#!/usr/bin/env python3
"""Step 2 -- where does the pruning damage actually land?

Every model is run on the SAME rows and compared to dense position-by-position,
keeping the token groups apart:

  encoder blocks:  visual-prefix positions  vs  text positions
  decoder blocks:  answer positions (teacher-forced)
  answer logits:   KL(dense || pruned), per token, PAIRED across models

The decoder matters. A first pass that hooked only the encoder found split and
joint indistinguishable there (cosine to dense within 1e-3 of each other) while
their answer-logit KL differed by 13%. A difference that is absent in the
encoder but present in the logits has to be born downstream of the encoder, so
the decoder is not optional -- it is the only place left for it to come from.

The KL comparison is PAIRED: the same answer token is scored under both models
and the per-token difference is what gets tested. Comparing two independent
means would not tell you whether a 0.36-vs-0.41 gap is a real shift or just two
noisy averages.

Text is padded to a fixed length so positions line up across checkpoints, and
the dense cache is float32 -- NOT float16, whose 65504 ceiling silently turns
T5's activation outliers into inf and every downstream cosine into nan.

Usage:
  python scripts/blip2/analyze_pruned_drift_by_token_group.py \
      --eval_json /path/mmbench_dev.json --images_dir /path/images \
      --ckpt split=/path/merged_split.pth \
      --ckpt joint=/path/joint.pth \
      --out_dir /path/out/step2_drift \
      --max_samples 64 --batch_size 2 --logit_kl
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

ENC_GROUPS = ("visual", "text")
DEC_GROUP = "answer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-layer, per-token-group activation drift from dense.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eval_json", required=True, help="Evaluation rows -- NOT the calibration set.")
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dense_ckpt", default=None, help="Dense reference. Omit for pretrained blip2_t5.")
    parser.add_argument("--ckpt", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--model_name", default="blip2_t5")
    parser.add_argument("--model_type", default="pretrain_flant5xl")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--image_field", default="image")
    parser.add_argument("--text_field", default="auto")
    parser.add_argument("--output_field", default="auto")
    parser.add_argument("--max_txt_len", type=int, default=32)
    parser.add_argument(
        "--logit_kl",
        action="store_true",
        help="Teacher-force the answer: enables decoder-block drift AND answer-logit KL.",
    )
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Bootstrap resamples for the paired-KL confidence interval.")
    return parser.parse_args()


class BlockCapture:
    """Capture the output hidden states of every T5 encoder AND decoder block."""

    def __init__(self, model: Any, torch: Any):
        self.torch = torch
        self.buffers: Dict[Tuple[str, int], Any] = {}
        self.handles = []
        for index, block in enumerate(model.t5_model.encoder.block):
            self.handles.append(block.register_forward_hook(self._make_hook("enc", index)))
        for index, block in enumerate(model.t5_model.decoder.block):
            self.handles.append(block.register_forward_hook(self._make_hook("dec", index)))

    def _make_hook(self, part: str, index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            self.buffers[(part, index)] = hidden.detach().to(self.torch.float32)

        return hook

    def clear(self) -> None:
        self.buffers = {}

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def run_model_pass(
    label: str,
    checkpoint: Optional[str],
    args: argparse.Namespace,
    rows: List[Any],
    original_indices: List[int],
    torch: Any,
    Image: Any,
    dense_cache: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """One full pass. dense_cache is None => we ARE dense and build the cache."""
    is_dense = dense_cache is None
    print("\n>>> [%s] %s" % (label, checkpoint or "pretrained dense"))

    model = load_blip2(args.model_name, args.model_type, args.device, checkpoint, args.max_txt_len)
    capture = BlockCapture(model, torch)
    forward = EncoderForward(model, torch, padding="max_length")
    vis_processor = build_vis_processor(args.image_size)

    cache: Dict[str, Any] = {"blocks": {}, "logits": {}} if is_dense else {}
    cos_acc: Dict[tuple, List[float]] = {}
    l2_acc: Dict[tuple, List[float]] = {}
    kl_tokens: List[float] = []          # per answer token, in a stable order
    census: Dict[str, float] = {g: 0.0 for g in ENC_GROUPS}
    census[DEC_GROUP] = 0.0
    nonfinite: Dict[tuple, int] = {}

    def compare(part: str, block: int, group: str, hidden: Any, mask: Any, batch_index: int) -> None:
        reference = torch.from_numpy(dense_cache["blocks"][(part, block)][batch_index]).to(
            hidden.device, torch.float32
        )
        if reference.shape != hidden.shape:
            raise SystemExit(
                "[FATAL] shape mismatch at %s block %d batch %d: dense=%s %s=%s"
                % (part, block, batch_index, tuple(reference.shape), label, tuple(hidden.shape))
            )
        cos = torch.nn.functional.cosine_similarity(reference, hidden, dim=-1)
        rel = torch.linalg.norm(hidden - reference, dim=-1) / torch.linalg.norm(
            reference, dim=-1
        ).clamp_min(1e-6)
        cos_values = cos[mask].detach().cpu().numpy()
        rel_values = rel[mask].detach().cpu().numpy()
        bad = int((~np.isfinite(cos_values)).sum())
        if bad:
            nonfinite[(part, block, group)] = nonfinite.get((part, block, group), 0) + bad
        cos_acc.setdefault((part, block, group), []).extend(cos_values.tolist())
        l2_acc.setdefault((part, block, group), []).extend(rel_values.tolist())

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
            logits = None
            if args.logit_kl:
                answers = [
                    extract_text(row, args.output_field, AUTO_OUTPUT_FIELDS, original_indices[start + i])
                    for i, row in enumerate(batch_rows)
                ]
                with torch.no_grad():
                    with model.maybe_autocast(dtype=torch.bfloat16):
                        target_tokens = model.t5_tokenizer(
                            answers, padding="max_length", truncation=True,
                            max_length=args.max_txt_len, return_tensors="pt",
                        ).to(args.device)
                        labels = target_tokens.input_ids.masked_fill(
                            target_tokens.input_ids == model.t5_tokenizer.pad_token_id, -100
                        )
                        # Reuse the encoder states we already ran; only the decoder is new.
                        decoder_out = model.t5_model(
                            encoder_outputs=(out["encoder_hidden"],),
                            attention_mask=out["encoder_attention"],
                            decoder_attention_mask=target_tokens.attention_mask,
                            labels=labels,
                            return_dict=True,
                        )
                        logits = decoder_out.logits.to(torch.float32)
                answer_mask = labels != -100

            # census
            for group in ENC_GROUPS:
                census[group] += float(enc_masks[group].sum().item())
            if answer_mask is not None:
                census[DEC_GROUP] += float(answer_mask.sum().item())

            if is_dense:
                for (part, block), hidden in capture.buffers.items():
                    cache["blocks"].setdefault((part, block), {})[batch_index] = (
                        hidden.detach().to(torch.float32).cpu().numpy()
                    )
                if logits is not None:
                    cache["logits"][batch_index] = (
                        logits.detach().to(torch.float32).cpu().numpy(),
                        answer_mask.detach().cpu().numpy(),
                    )
            else:
                for (part, block), hidden in capture.buffers.items():
                    if part == "enc":
                        for group in ENC_GROUPS:
                            compare(part, block, group, hidden, enc_masks[group], batch_index)
                    elif answer_mask is not None:
                        compare(part, block, DEC_GROUP, hidden, answer_mask, batch_index)

                if logits is not None:
                    dense_logits_np, dense_valid_np = dense_cache["logits"][batch_index]
                    dense_logits = torch.from_numpy(dense_logits_np).to(logits.device, torch.float32)
                    dense_valid = torch.from_numpy(dense_valid_np).to(logits.device)
                    use = answer_mask & dense_valid
                    log_p = torch.log_softmax(dense_logits, dim=-1)
                    log_q = torch.log_softmax(logits, dim=-1)
                    kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1)  # [B, L]
                    # flatten in a fixed order so tokens pair up across models
                    kl_tokens.extend(kl[use].detach().cpu().numpy().tolist())

            if batch_index % 10 == 0:
                print("    batch %d  seq_len=%d" % (batch_index, out["seq_len"]))
    finally:
        capture.close()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total = sum(census.values())
    print("  token census: " + "  ".join(
        "%s=%.0f (%.1f%%)" % (g, census[g], 100.0 * census[g] / max(total, 1.0)) for g in census
    ))
    for group in ENC_GROUPS:
        if census[group] <= 0:
            raise SystemExit("[FATAL] token group %r is empty -- the mask is broken." % group)
    if args.logit_kl and census[DEC_GROUP] <= 0:
        raise SystemExit("[FATAL] no answer tokens -- check --output_field.")
    if nonfinite:
        raise SystemExit(
            "[FATAL] %d non-finite cosines -- numerics bug, do NOT interpret the output. %s"
            % (sum(nonfinite.values()), sorted(nonfinite)[:8])
        )

    if is_dense:
        return cache

    rows_out: List[Dict[str, Any]] = []
    for (part, block, group), values in sorted(cos_acc.items()):
        cos = np.asarray(values, dtype=np.float64)
        rel = np.asarray(l2_acc[(part, block, group)], dtype=np.float64)
        rows_out.append(
            {
                "model": label,
                "part": "encoder" if part == "enc" else "decoder",
                "block": block,
                "token_group": group,
                "tokens": int(cos.size),
                "cos_to_dense_mean": float(cos.mean()),
                "cos_to_dense_median": float(np.median(cos)),
                "cos_to_dense_p10": float(np.percentile(cos, 10)),
                "rel_l2_to_dense_mean": float(rel.mean()),
            }
        )
    return {"rows": rows_out, "kl_tokens": np.asarray(kl_tokens, dtype=np.float64)}


def paired_kl_report(
    label_a: str, kl_a: np.ndarray, label_b: str, kl_b: np.ndarray, bootstrap: int, seed: int
) -> Dict[str, Any]:
    """Per-token paired comparison. Positive diff => b is further from dense than a."""
    n = min(kl_a.size, kl_b.size)
    a = kl_a[:n]
    b = kl_b[:n]
    diff = b - a

    rng = np.random.default_rng(seed)
    means = np.empty(bootstrap, dtype=np.float64)
    for i in range(bootstrap):
        means[i] = diff[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])

    return {
        "a": label_a,
        "b": label_b,
        "answer_tokens": int(n),
        "kl_%s_mean" % label_a: float(a.mean()),
        "kl_%s_mean" % label_b: float(b.mean()),
        "paired_diff_mean": float(diff.mean()),
        "paired_diff_ci95_lo": float(lo),
        "paired_diff_ci95_hi": float(hi),
        "fraction_tokens_%s_worse" % label_b: float((diff > 0).mean()),
        "significant": bool(lo > 0 or hi < 0),
    }


def plot_drift(plt, rows: List[Dict[str, Any]], labels: List[str], path: str) -> None:
    if plt is None or not rows:
        return
    colors = {"split": "#54A24B", "joint": "#E45756"}
    palette = ["#4C78A8", "#F58518", "#72B7B2", "#B279A2"]
    panels = [("encoder", "visual", "Encoder / visual prefix"),
              ("encoder", "text", "Encoder / text"),
              ("decoder", DEC_GROUP, "Decoder / answer")]
    panels = [p for p in panels
              if any(r["part"] == p[0] and r["token_group"] == p[1] for r in rows)]
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(6.3 * len(panels), 4.8), squeeze=False)
    for ax, (part, group, title) in zip(axes[0], panels):
        for i, label in enumerate(labels):
            sel = [r for r in rows if r["model"] == label
                   and r["part"] == part and r["token_group"] == group]
            sel.sort(key=lambda r: r["block"])
            if not sel:
                continue
            ax.plot([r["block"] for r in sel], [r["cos_to_dense_mean"] for r in sel],
                    marker="o", markersize=4, linewidth=1.8,
                    color=colors.get(label, palette[i % len(palette)]), label=label)
        ax.set_title(title)
        ax.set_xlabel("Block index")
        ax.set_ylabel("cosine to dense (per token, mean)")
        ax.grid(True, alpha=0.28)
        ax.legend()
    fig.suptitle("Activation drift from dense")
    fig.tight_layout()
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

    rows, original_indices = select_rows(
        load_rows(os.path.abspath(args.eval_json)), args.max_samples, args.shuffle, args.seed
    )
    print("device:", args.device)
    print("samples:", len(rows))
    print("models:", ", ".join(["dense"] + list(pruned)))
    if not args.logit_kl:
        print("[NOTE] --logit_kl is off: no decoder blocks, no KL. The encoder alone has already\n"
              "       been shown not to separate split from joint -- you probably want it on.")

    dense_cache = run_model_pass(
        "dense", args.dense_ckpt, args, rows, original_indices, torch, Image, None
    )

    all_rows: List[Dict[str, Any]] = []
    kl_by_model: Dict[str, np.ndarray] = {}
    for label, path in pruned.items():
        result = run_model_pass(
            label, path, args, rows, original_indices, torch, Image, dense_cache
        )
        all_rows.extend(result["rows"])
        if result["kl_tokens"].size:
            kl_by_model[label] = result["kl_tokens"]

    write_csv(os.path.join(args.out_dir, "drift_by_layer_and_token_group.csv"), all_rows)
    plt = setup_matplotlib()
    labels = list(pruned)
    plot_drift(plt, all_rows, labels, os.path.join(args.out_dir, "drift_by_token_group.png"))

    def mean_cos(label: str, part: str, group: str) -> float:
        values = [r["cos_to_dense_mean"] for r in all_rows
                  if r["model"] == label and r["part"] == part and r["token_group"] == group]
        return float(np.mean(values)) if values else float("nan")

    print("\n=== mean cosine to dense (averaged over blocks) ===")
    header = "  %-8s  %-12s %-12s %-12s" % ("model", "enc/visual", "enc/text", "dec/answer")
    print(header)
    for label in labels:
        print("  %-8s  %-12.5f %-12.5f %-12.5f"
              % (label, mean_cos(label, "encoder", "visual"),
                 mean_cos(label, "encoder", "text"),
                 mean_cos(label, "decoder", DEC_GROUP)))

    paired: List[Dict[str, Any]] = []
    if len(kl_by_model) >= 2:
        keys = list(kl_by_model)
        report = paired_kl_report(
            keys[0], kl_by_model[keys[0]], keys[1], kl_by_model[keys[1]], args.bootstrap, args.seed
        )
        paired.append(report)
        write_csv(os.path.join(args.out_dir, "paired_answer_logit_kl.csv"), paired)
        a, b = report["a"], report["b"]
        print("\n=== answer-logit KL(dense || pruned), PAIRED over %d answer tokens ==="
              % report["answer_tokens"])
        print("  %-8s mean=%.4f" % (a, report["kl_%s_mean" % a]))
        print("  %-8s mean=%.4f" % (b, report["kl_%s_mean" % b]))
        print("  paired diff (%s - %s) = %+.4f   95%% CI [%+.4f, %+.4f]"
              % (b, a, report["paired_diff_mean"],
                 report["paired_diff_ci95_lo"], report["paired_diff_ci95_hi"]))
        print("  %s is worse on %.1f%% of individual answer tokens"
              % (b, 100.0 * report["fraction_tokens_%s_worse" % b]))
        if report["significant"]:
            print("  -> the CI excludes zero: the gap is a real shift, not two noisy means.")
        else:
            print("  -> the CI CONTAINS zero: this gap is NOT statistically distinguishable from\n"
                  "     noise at this sample size. Do not build a story on it -- raise\n"
                  "     --max_samples and re-measure first.")

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "eval_json": os.path.abspath(args.eval_json),
                "samples": len(rows),
                "models": {label: os.path.abspath(path) for label, path in pruned.items()},
                "mean_cos_to_dense": {
                    label: {
                        "encoder_visual": mean_cos(label, "encoder", "visual"),
                        "encoder_text": mean_cos(label, "encoder", "text"),
                        "decoder_answer": mean_cos(label, "decoder", DEC_GROUP),
                    }
                    for label in labels
                },
                "paired_answer_logit_kl": paired,
            },
            handle,
            indent=2,
        )

    print("\n[OK] wrote:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
