#!/usr/bin/env python3
"""Per-layer activation similarity to dense, for one (or a few) MMBench samples.

Runs the SAME MMBench forward through three model states -- dense (unpruned),
joint-pruned, split-pruned -- captures every T5 encoder and decoder block's
output, and plots, layer by layer, how close each pruned model stays to dense.
The point is a single readable figure answering "who tracks the full model?".

Metrics (all computed; the plots default to relative-L2):

  * relative L2 drift   ||pruned - dense|| / ||dense||   (LOWER = closer)
        The headline. Cosine saturates near 1 and ignores magnitude; pruning
        changes magnitude, so a direction-only metric hides real differences.
  * cosine              direction agreement                (higher = closer)
  * centered cosine     cosine after removing the per-token mean

Everything is split by token group, because the encoder sequence is
[32 visual-prefix positions] + [text], and those two behave differently.

IMPORTANT -- precision. In bf16 the per-layer gap between split and joint is
often ~1e-3, near bf16's resolution, and on a single sample it can be noise.
Two remedies, both supported:
  * --fp32       run the whole forward in float32 (needs ~16GB for T5-XL; one
                 model is resident at a time). This is what actually resolves
                 the difference.
  * (default)    aggregate over --max_samples; the small gaps are stable in the
                 mean even when any single sample is noisy.
A single --sample_index in bf16 is the least reliable combination; the script
warns when you pick it.

Usage (one sample, high resolution):
  python scripts/blip2/visualize_layer_activation_similarity.py \
      --eval_json /path/mmbench_dev.json --images_dir /path/images \
      --ckpt joint=/path/joint.pth --ckpt split=/path/merged_split.pth \
      --out_dir /path/out/layer_similarity --sample_index 0 --fp32

Usage (stable aggregate, default precision):
  python scripts/blip2/visualize_layer_activation_similarity.py \
      --eval_json /path/mmbench_dev.json --images_dir /path/images \
      --ckpt joint=/path/joint.pth --ckpt split=/path/merged_split.pth \
      --out_dir /path/out/layer_similarity --max_samples 32
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from split_joint_analysis_common import (
    AUTO_INPUT_FIELDS,
    AUTO_OUTPUT_FIELDS,
    EncoderForward,
    build_vis_processor,
    ensure_dir,
    extract_text,
    iter_batches,
    load_batch_images,
    load_blip2,
    load_rows,
    parse_labeled_paths,
    select_rows,
    setup_matplotlib,
    write_csv,
)

PANELS = [
    ("encoder", "visual", "Encoder / visual prefix"),
    ("encoder", "text", "Encoder / text"),
    ("decoder", "answer", "Decoder / answer"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize per-layer activation similarity of pruned models to dense.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eval_json", required=True, help="MMBench (or any) eval rows.")
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dense_ckpt", default=None, help="Dense reference. Omit for pretrained blip2_t5.")
    parser.add_argument("--ckpt", action="append", required=True, metavar="LABEL=PATH",
                        help="Repeatable, e.g. --ckpt joint=/a.pth --ckpt split=/b.pth")
    parser.add_argument("--sample_index", type=int, default=None,
                        help="Row index for a single-sample trace. Omit to aggregate over --max_samples.")
    parser.add_argument("--max_samples", type=int, default=32,
                        help="Aggregate window when --sample_index is not given.")
    parser.add_argument("--model_name", default="blip2_t5")
    parser.add_argument("--model_type", default="pretrain_flant5xl")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--image_field", default="image")
    parser.add_argument("--text_field", default="auto")
    parser.add_argument("--output_field", default="auto")
    parser.add_argument("--max_txt_len", type=int, default=32)
    parser.add_argument("--fp32", action="store_true",
                        help="Run the whole forward in float32 to resolve sub-bf16 differences.")
    parser.add_argument("--no_decoder", action="store_true",
                        help="Skip the teacher-forced decoder pass (no answer needed).")
    parser.add_argument("--metric", choices=["rel_l2", "cosine", "centered_cosine"],
                        default="rel_l2", help="Metric drawn in the main figure.")
    return parser.parse_args()


class BlockCapture:
    def __init__(self, model: Any, torch: Any, include_decoder: bool):
        self.torch = torch
        self.buffers: Dict[Tuple[str, int], Any] = {}
        self.handles = []
        for index, block in enumerate(model.t5_model.encoder.block):
            self.handles.append(block.register_forward_hook(self._hook("encoder", index)))
        if include_decoder:
            for index, block in enumerate(model.t5_model.decoder.block):
                self.handles.append(block.register_forward_hook(self._hook("decoder", index)))

    def _hook(self, part: str, index: int):
        def hook(_m: Any, _i: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            self.buffers[(part, index)] = hidden.detach().to(self.torch.float32)
        return hook

    def clear(self) -> None:
        self.buffers = {}

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def token_metrics(dense: Any, pruned: Any, mask: Any, torch: Any) -> Dict[str, np.ndarray]:
    """Per-token metrics for the masked positions. dense/pruned: [B,T,H], mask: [B,T]."""
    d = dense[mask]  # [N, H]
    p = pruned[mask]
    diff = torch.linalg.norm(p - d, dim=-1)
    dn = torch.linalg.norm(d, dim=-1).clamp_min(1e-8)
    rel_l2 = (diff / dn)
    cos = torch.nn.functional.cosine_similarity(d, p, dim=-1)
    dc = d - d.mean(dim=-1, keepdim=True)
    pc = p - p.mean(dim=-1, keepdim=True)
    ccos = torch.nn.functional.cosine_similarity(dc, pc, dim=-1)
    return {
        "rel_l2": rel_l2.cpu().numpy(),
        "cosine": cos.cpu().numpy(),
        "centered_cosine": ccos.cpu().numpy(),
    }


def run_pass(
    label: str,
    checkpoint: Optional[str],
    args: argparse.Namespace,
    rows: List[Any],
    original_indices: List[int],
    torch: Any,
    Image: Any,
    include_decoder: bool,
    dense_cache: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    is_dense = dense_cache is None
    print("\n>>> [%s] %s" % (label, checkpoint or "pretrained dense"))

    model = load_blip2(args.model_name, args.model_type, args.device, checkpoint, args.max_txt_len)
    if args.fp32:
        model.float()
    capture = BlockCapture(model, torch, include_decoder)
    forward = EncoderForward(model, torch, padding="max_length", fp32=args.fp32)
    vis_processor = build_vis_processor(args.image_size)

    cache: Dict[str, Any] = {"blocks": {}, "masks": {}} if is_dense else {}
    # (part, block, group) -> {metric -> list}
    acc: Dict[tuple, Dict[str, List[float]]] = {}

    def accumulate(part, block, group, dense_h, pruned_h, mask, batch_index):
        m = token_metrics(dense_h, pruned_h, mask, torch)
        bucket = acc.setdefault((part, block, group), {k: [] for k in m})
        for k, v in m.items():
            if not np.all(np.isfinite(v)):
                raise SystemExit(
                    "[FATAL] non-finite %s at %s block %d group %s -- likely fp16 overflow; "
                    "use --fp32 or check inputs." % (k, part, block, group)
                )
            bucket[k].extend(v.tolist())

    try:
        for batch_index, (start, batch_rows) in enumerate(iter_batches(rows, args.batch_size)):
            images = load_batch_images(
                batch_rows, start, original_indices, os.path.abspath(args.images_dir),
                args.image_field, vis_processor, torch, Image,
            ).to(args.device)
            texts = [
                extract_text(row, args.text_field, AUTO_INPUT_FIELDS, original_indices[start + i])
                for i, row in enumerate(batch_rows)
            ]

            capture.clear()
            out = forward.run(images, texts, args.device)
            enc_masks = {"visual": out["visual_mask"], "text": out["text_mask"]}

            answer_mask = None
            if include_decoder:
                answers = [
                    extract_text(row, args.output_field, AUTO_OUTPUT_FIELDS, original_indices[start + i])
                    for i, row in enumerate(batch_rows)
                ]
                with torch.no_grad():
                    with forward._amp(torch.bfloat16):
                        target = model.t5_tokenizer(
                            answers, padding="max_length", truncation=True,
                            max_length=args.max_txt_len, return_tensors="pt",
                        ).to(args.device)
                        labels = target.input_ids.masked_fill(
                            target.input_ids == model.t5_tokenizer.pad_token_id, -100
                        )
                        model.t5_model(
                            encoder_outputs=(out["encoder_hidden"],),
                            attention_mask=out["encoder_attention"],
                            decoder_attention_mask=target.attention_mask,
                            labels=labels, return_dict=True,
                        )
                answer_mask = labels != -100

            if is_dense:
                cache["masks"][batch_index] = {
                    "visual": enc_masks["visual"].cpu().numpy(),
                    "text": enc_masks["text"].cpu().numpy(),
                    "answer": answer_mask.cpu().numpy() if answer_mask is not None else None,
                }
                for (part, block), hidden in capture.buffers.items():
                    cache["blocks"].setdefault((part, block), {})[batch_index] = (
                        hidden.to(torch.float32).cpu().numpy()
                    )
            else:
                masks_np = dense_cache["masks"][batch_index]
                for (part, block), hidden in capture.buffers.items():
                    dense_h = torch.from_numpy(
                        dense_cache["blocks"][(part, block)][batch_index]
                    ).to(hidden.device, torch.float32)
                    if dense_h.shape != hidden.shape:
                        raise SystemExit(
                            "[FATAL] shape mismatch %s block %d: dense=%s %s=%s"
                            % (part, block, tuple(dense_h.shape), label, tuple(hidden.shape))
                        )
                    if part == "encoder":
                        for group in ("visual", "text"):
                            mask = torch.from_numpy(masks_np[group]).to(hidden.device)
                            if bool(mask.any()):
                                accumulate(part, block, group, dense_h, hidden, mask, batch_index)
                    elif masks_np["answer"] is not None:
                        mask = torch.from_numpy(masks_np["answer"]).to(hidden.device)
                        if bool(mask.any()):
                            accumulate(part, block, "answer", dense_h, hidden, mask, batch_index)

            print("    batch %d  seq_len=%d" % (batch_index, out["seq_len"]))
    finally:
        capture.close()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if is_dense:
        return cache

    rows_out: List[Dict[str, Any]] = []
    for (part, block, group), metrics in sorted(acc.items()):
        row: Dict[str, Any] = {
            "model": label, "part": part, "block": block, "token_group": group,
            "tokens": len(metrics["rel_l2"]),
        }
        for metric_name, values in metrics.items():
            arr = np.asarray(values, dtype=np.float64)
            row["%s_mean" % metric_name] = float(arr.mean())
            row["%s_median" % metric_name] = float(np.median(arr))
        rows_out.append(row)
    return {"rows": rows_out}


def plot_figure(plt, rows, labels, metric, single_sample, path):
    if plt is None or not rows:
        return
    higher_better = metric in ("cosine", "centered_cosine")
    ylabel = {"rel_l2": "relative L2 to dense  (lower = closer)",
              "cosine": "cosine to dense  (higher = closer)",
              "centered_cosine": "centered cosine to dense  (higher = closer)"}[metric]
    key = "%s_mean" % metric
    colors = {"split": "#54A24B", "joint": "#E45756"}
    palette = ["#4C78A8", "#F58518", "#72B7B2", "#B279A2"]

    active = [p for p in PANELS if any(r["part"] == p[0] and r["token_group"] == p[1] for r in rows)]
    if not active:
        return

    fig, axes = plt.subplots(2, len(active), figsize=(5.6 * len(active), 8.4), squeeze=False)
    for col, (part, group, title) in enumerate(active):
        ax = axes[0][col]
        series = {}
        for i, label in enumerate(labels):
            sel = [r for r in rows if r["model"] == label and r["part"] == part and r["token_group"] == group]
            sel.sort(key=lambda r: r["block"])
            if not sel:
                continue
            blocks = [r["block"] for r in sel]
            values = [r[key] for r in sel]
            series[label] = (blocks, dict(zip(blocks, values)))
            ax.plot(blocks, values, marker="o", markersize=4, linewidth=1.9,
                    color=colors.get(label, palette[i % len(palette)]), label=label)
        ax.set_title(title)
        ax.set_xlabel("Block index")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.28)
        ax.legend()

        # who-wins delta panel
        axd = axes[1][col]
        if "split" in series and "joint" in series:
            common = sorted(set(series["split"][1]) & set(series["joint"][1]))
            s = np.array([series["split"][1][b] for b in common])
            j = np.array([series["joint"][1][b] for b in common])
            # positive bar = split closer to dense than joint
            delta = (j - s) if metric == "rel_l2" else (s - j)
            ax_colors = ["#54A24B" if d >= 0 else "#E45756" for d in delta]
            axd.bar(common, delta, color=ax_colors, alpha=0.85)
            axd.axhline(0.0, color="black", linewidth=1.0)
            axd.set_title("%s: who is closer to dense" % title)
            axd.set_xlabel("Block index")
            axd.set_ylabel("green = split closer\nred = joint closer")
            axd.grid(True, axis="y", alpha=0.28)
        else:
            axd.axis("off")

    mode = ("single sample #%d" % single_sample) if single_sample is not None else "aggregate"
    fig.suptitle("Per-layer activation similarity to dense (%s)" % mode, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Missing runtime dependency: %s" % exc) from exc

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(args.out_dir)

    pruned = parse_labeled_paths(args.ckpt)
    for label, path in pruned.items():
        if not os.path.isfile(path):
            raise FileNotFoundError("Checkpoint not found for %s: %s" % (label, path))

    all_rows = load_rows(os.path.abspath(args.eval_json))
    if args.sample_index is not None:
        if not (0 <= args.sample_index < len(all_rows)):
            raise SystemExit("sample_index %d out of range [0, %d)" % (args.sample_index, len(all_rows)))
        rows = [all_rows[args.sample_index]]
        original_indices = [args.sample_index]
        if not args.fp32:
            print("[WARN] single sample in bf16 is the noisiest setting -- the split/joint gap may be\n"
                  "       below bf16 resolution. Consider --fp32, or drop --sample_index to aggregate.")
    else:
        rows, original_indices = select_rows(all_rows, args.max_samples, args.shuffle, args.seed)

    include_decoder = not args.no_decoder
    print("device:", args.device, "| precision:", "fp32" if args.fp32 else "bf16/fp16 autocast")
    print("samples:", len(rows), "| decoder:", include_decoder)
    print("models:", ", ".join(["dense"] + list(pruned)))

    dense_cache = run_pass("dense", args.dense_ckpt, args, rows, original_indices,
                           torch, Image, include_decoder, None)

    result_rows: List[Dict[str, Any]] = []
    for label, path in pruned.items():
        res = run_pass(label, path, args, rows, original_indices, torch, Image, include_decoder, dense_cache)
        result_rows.extend(res["rows"])

    write_csv(os.path.join(args.out_dir, "per_layer_similarity_to_dense.csv"), result_rows)

    plt = setup_matplotlib()
    plot_figure(plt, result_rows, list(pruned), args.metric, args.sample_index,
                os.path.join(args.out_dir, "per_layer_similarity_%s.png" % args.metric))
    # also emit the cosine view so you can eyeball both
    if args.metric != "cosine":
        plot_figure(plt, result_rows, list(pruned), "cosine", args.sample_index,
                    os.path.join(args.out_dir, "per_layer_similarity_cosine.png"))

    # ---- verdict ----
    def agg(label, part, group, metric):
        vals = [r["%s_mean" % metric] for r in result_rows
                if r["model"] == label and r["part"] == part and r["token_group"] == group]
        return float(np.mean(vals)) if vals else float("nan")

    print("\n=== mean over blocks, relative-L2 drift to dense (lower = closer) ===")
    for part, group, title in PANELS:
        line = "  %-22s" % title
        present = False
        for label in pruned:
            v = agg(label, part, group, "rel_l2")
            if np.isfinite(v):
                present = True
                line += "  %s=%.5f" % (label, v)
        if present:
            print(line)

    summary = {
        "eval_json": os.path.abspath(args.eval_json),
        "mode": "single_sample" if args.sample_index is not None else "aggregate",
        "sample_index": args.sample_index,
        "num_samples": len(rows),
        "precision": "fp32" if args.fp32 else "autocast",
        "models": {label: os.path.abspath(path) for label, path in pruned.items()},
        "rel_l2_mean": {
            label: {"%s_%s" % (p, g): agg(label, p, g, "rel_l2") for p, g, _ in PANELS}
            for label in pruned
        },
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if "split" in pruned and "joint" in pruned:
        print("\n=== verdict ===")
        for part, group, title in PANELS:
            s = agg("split", part, group, "rel_l2")
            j = agg("joint", part, group, "rel_l2")
            if not (np.isfinite(s) and np.isfinite(j)):
                continue
            if abs(s - j) < 1e-5:
                who = "indistinguishable"
            else:
                who = "split closer" if s < j else "joint closer"
            print("  %-22s split=%.5f joint=%.5f  -> %s" % (title, s, j, who))
        if not args.fp32:
            print("\n  (bf16: treat sub-1e-3 gaps as noise. Re-run with --fp32 to trust them.)")

    print("\n[OK] wrote:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
