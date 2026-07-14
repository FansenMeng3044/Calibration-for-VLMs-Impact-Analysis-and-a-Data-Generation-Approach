#!/usr/bin/env python3
"""Step 2 -- where does the pruning damage actually land?

Step 1 says *which* weights joint pruning throws away. This says what that costs
in the forward pass, and it answers the question the last-layer-only comparison
cannot: does joint hurt the *language* pathway more than the visual one?

For every T5 encoder block, every model is run on the SAME evaluation rows and
compared to dense position-by-position, with visual-prefix positions and text
positions kept apart:

    cos(dense[n, t, :], pruned[n, t, :])   averaged within each token group

Predicted signature if the visual-hijack story from step 1 is right:
joint's drift on TEXT positions is much worse than split's, while on VISUAL
positions the two are comparable -- or joint is even better, because that is
exactly what it optimized its mask for.  Benchmarks punish the former.

``--logit_kl`` additionally teacher-forces the answer and reports
KL(dense || pruned) on the answer logits, which is the quantity that actually
decides whether the model gets the question right.

Text is padded to a fixed length so positions line up across checkpoints.

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
from typing import Any, Dict, List, Optional

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

GROUPS = ("visual", "text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-layer, per-token-group activation drift from dense.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eval_json", required=True,
                        help="Evaluation rows -- NOT the calibration set.")
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dense_ckpt", default=None,
                        help="Dense reference. Omit to use pretrained blip2_t5.")
    parser.add_argument("--ckpt", action="append", required=True, metavar="LABEL=PATH",
                        help="Repeatable. e.g. --ckpt split=/a.pth --ckpt joint=/b.pth")
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
    parser.add_argument("--max_txt_len", type=int, default=32,
                        help="Fixed text length; sequence is 32 visual + this many text positions.")
    parser.add_argument("--logit_kl", action="store_true",
                        help="Also teacher-force the answer and report KL(dense || pruned) on logits.")
    return parser.parse_args()


class BlockCapture:
    """Capture each T5 encoder block's output hidden states."""

    def __init__(self, model: Any, torch: Any):
        self.torch = torch
        self.buffers: Dict[int, Any] = {}
        self.handles = []
        for index, block in enumerate(model.t5_model.encoder.block):
            self.handles.append(block.register_forward_hook(self._make_hook(index)))
        self.num_blocks = len(model.t5_model.encoder.block)

    def _make_hook(self, index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            self.buffers[index] = hidden.detach().to(self.torch.float32)

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
    logit_cache: Optional[Dict[int, Any]],
) -> Dict[str, Any]:
    """One full pass. If dense_cache is None we ARE dense: build the cache.

    Otherwise compare against it batch by batch and return drift stats.
    """
    is_dense = dense_cache is None
    print("\n>>> [%s] %s" % (label, checkpoint or "pretrained dense"))

    model = load_blip2(
        args.model_name, args.model_type, args.device, checkpoint, args.max_txt_len
    )
    capture = BlockCapture(model, torch)
    forward = EncoderForward(model, torch, padding="max_length")
    vis_processor = build_vis_processor(args.image_size)

    cache: Dict[str, Any] = {"blocks": {}, "masks": {}} if is_dense else {}
    # (block, group) -> list of per-token cosines / relative L2
    cos_acc: Dict[tuple, List[float]] = {}
    l2_acc: Dict[tuple, List[float]] = {}
    kl_acc: List[float] = []
    token_census: Dict[str, float] = {g: 0.0 for g in GROUPS}
    nonfinite: Dict[tuple, int] = {}

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

            masks = {
                "visual": out["visual_mask"],
                "text": out["text_mask"],
            }

            for group in GROUPS:
                token_census[group] += float(masks[group].sum().item())

            if is_dense:
                cache["masks"][batch_index] = {
                    g: masks[g].detach().cpu().numpy() for g in GROUPS
                }
                for block_index, hidden in capture.buffers.items():
                    # float32, NOT float16: T5's activation outliers routinely exceed
                    # fp16's 65504 ceiling, which silently turns the cache into inf and
                    # every downstream cosine into nan.
                    cache["blocks"].setdefault(block_index, {})[batch_index] = (
                        hidden.detach().to(torch.float32).cpu().numpy()
                    )
            else:
                for block_index, hidden in capture.buffers.items():
                    reference = torch.from_numpy(
                        dense_cache["blocks"][block_index][batch_index]
                    ).to(hidden.device, torch.float32)
                    if reference.shape != hidden.shape:
                        raise ValueError(
                            "Shape mismatch at block %d batch %d: dense=%s %s=%s. "
                            "Sequence lengths must match across checkpoints."
                            % (block_index, batch_index, tuple(reference.shape), label, tuple(hidden.shape))
                        )
                    cos = torch.nn.functional.cosine_similarity(reference, hidden, dim=-1)  # [B, T]
                    diff = torch.linalg.norm(hidden - reference, dim=-1)
                    denom = torch.linalg.norm(reference, dim=-1).clamp_min(1e-6)
                    rel = diff / denom
                    for group in GROUPS:
                        mask = masks[group]
                        if not bool(mask.any()):
                            continue
                        cos_values = cos[mask].detach().cpu().numpy()
                        rel_values = rel[mask].detach().cpu().numpy()
                        bad = int((~np.isfinite(cos_values)).sum())
                        if bad:
                            nonfinite[(block_index, group)] = (
                                nonfinite.get((block_index, group), 0) + bad
                            )
                        cos_acc.setdefault((block_index, group), []).extend(cos_values.tolist())
                        l2_acc.setdefault((block_index, group), []).extend(rel_values.tolist())

            if args.logit_kl:
                answers = [
                    extract_text(row, args.output_field, AUTO_OUTPUT_FIELDS,
                                 original_indices[start + i])
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
                        logits = decoder_out.logits.to(torch.float32)  # [B, L, V]

                valid = (labels != -100)
                if is_dense:
                    logit_cache[batch_index] = (
                        logits.detach().to(torch.float16).cpu().numpy(),
                        valid.detach().cpu().numpy(),
                    )
                else:
                    dense_logits_np, dense_valid_np = logit_cache[batch_index]
                    dense_logits = torch.from_numpy(dense_logits_np).to(logits.device, torch.float32)
                    dense_valid = torch.from_numpy(dense_valid_np).to(logits.device)
                    use = valid & dense_valid
                    if bool(use.any()):
                        log_p = torch.log_softmax(dense_logits, dim=-1)
                        log_q = torch.log_softmax(logits, dim=-1)
                        kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1)  # [B, L]
                        kl_acc.extend(kl[use].detach().cpu().numpy().tolist())

            if batch_index % 5 == 0:
                print("    batch %d  seq_len=%d" % (batch_index, out["seq_len"]))
    finally:
        capture.close()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_tokens = sum(token_census.values())
    print("  token census: " + "  ".join(
        "%s=%.0f (%.1f%%)" % (g, token_census[g], 100.0 * token_census[g] / max(total_tokens, 1.0))
        for g in GROUPS
    ))
    for group in GROUPS:
        if token_census[group] <= 0:
            raise SystemExit(
                "[FATAL] token group %r is empty -- the mask is broken, not the model. "
                "Every metric for it would be nan." % group
            )
    if nonfinite:
        total_bad = sum(nonfinite.values())
        raise SystemExit(
            "[FATAL] %d non-finite cosine values (e.g. inf/nan hidden states). This is a\n"
            "        numerics bug in the analysis, not a property of the checkpoints -- do not\n"
            "        interpret the output. Affected (block, group): %s"
            % (total_bad, sorted(nonfinite)[:8])
        )

    if is_dense:
        return cache

    rows_out: List[Dict[str, Any]] = []
    for (block_index, group), values in sorted(cos_acc.items()):
        cos = np.asarray(values, dtype=np.float64)
        rel = np.asarray(l2_acc[(block_index, group)], dtype=np.float64)
        rows_out.append(
            {
                "model": label,
                "block": block_index,
                "token_group": group,
                "tokens": int(cos.size),
                "cos_to_dense_mean": float(cos.mean()),
                "cos_to_dense_median": float(np.median(cos)),
                "cos_to_dense_p10": float(np.percentile(cos, 10)),
                "rel_l2_to_dense_mean": float(rel.mean()),
            }
        )
    result: Dict[str, Any] = {"rows": rows_out}
    if args.logit_kl and kl_acc:
        kl = np.asarray(kl_acc, dtype=np.float64)
        result["logit_kl"] = {
            "model": label,
            "answer_tokens": int(kl.size),
            "kl_dense_to_pruned_mean": float(kl.mean()),
            "kl_dense_to_pruned_median": float(np.median(kl)),
            "kl_dense_to_pruned_p90": float(np.percentile(kl, 90)),
        }
    return result


def plot_drift(plt, rows: List[Dict[str, Any]], labels: List[str], path: str) -> None:
    if plt is None or not rows:
        return
    colors = {"split": "#54A24B", "joint": "#E45756"}
    palette = ["#4C78A8", "#F58518", "#72B7B2", "#B279A2"]

    fig, axes = plt.subplots(1, len(GROUPS), figsize=(7.2 * len(GROUPS), 5.0), squeeze=False)
    for ax, group in zip(axes[0], GROUPS):
        for i, label in enumerate(labels):
            sel = [r for r in rows if r["model"] == label and r["token_group"] == group]
            sel.sort(key=lambda r: r["block"])
            if not sel:
                continue
            ax.plot(
                [r["block"] for r in sel],
                [r["cos_to_dense_mean"] for r in sel],
                marker="o", markersize=4, linewidth=1.8,
                color=colors.get(label, palette[i % len(palette)]),
                label=label,
            )
        ax.set_title("%s tokens" % ("visual prefix" if group == "visual" else "text"))
        ax.set_xlabel("T5 encoder block")
        ax.set_ylabel("cosine to dense (per token, mean)")
        ax.grid(True, alpha=0.28)
        ax.legend()
    fig.suptitle("Activation drift from dense, split by token group")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_delta(plt, rows: List[Dict[str, Any]], a: str, b: str, path: str) -> None:
    """cos(dense, a) - cos(dense, b) per layer. Positive => a is closer to dense."""
    if plt is None or not rows:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"visual": "#E45756", "text": "#4C78A8"}
    for group in GROUPS:
        rows_a = {r["block"]: r["cos_to_dense_mean"] for r in rows
                  if r["model"] == a and r["token_group"] == group}
        rows_b = {r["block"]: r["cos_to_dense_mean"] for r in rows
                  if r["model"] == b and r["token_group"] == group}
        blocks = sorted(set(rows_a) & set(rows_b))
        if not blocks:
            continue
        ax.plot(
            blocks,
            [rows_a[i] - rows_b[i] for i in blocks],
            marker="o", markersize=4, linewidth=1.8,
            color=colors[group],
            label="%s tokens" % ("visual prefix" if group == "visual" else "text"),
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("T5 encoder block")
    ax.set_ylabel("cos(dense, %s) - cos(dense, %s)" % (a, b))
    ax.set_title("Positive = %s stays closer to dense than %s" % (a, b))
    ax.grid(True, alpha=0.28)
    ax.legend()
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
        load_rows(os.path.abspath(args.eval_json)),
        args.max_samples,
        args.shuffle,
        args.seed,
    )
    print("device:", args.device)
    print("samples:", len(rows))
    print("models:", ", ".join(["dense"] + list(pruned)))

    logit_cache: Optional[Dict[int, Any]] = {} if args.logit_kl else None

    dense_cache = run_model_pass(
        "dense", args.dense_ckpt, args, rows, original_indices, torch, Image, None, logit_cache
    )

    all_rows: List[Dict[str, Any]] = []
    kl_rows: List[Dict[str, Any]] = []
    for label, path in pruned.items():
        result = run_model_pass(
            label, path, args, rows, original_indices, torch, Image, dense_cache, logit_cache
        )
        all_rows.extend(result["rows"])
        if "logit_kl" in result:
            kl_rows.append(result["logit_kl"])

    write_csv(os.path.join(args.out_dir, "drift_by_layer_and_token_group.csv"), all_rows)
    if kl_rows:
        write_csv(os.path.join(args.out_dir, "answer_logit_kl.csv"), kl_rows)

    plt = setup_matplotlib()
    labels = list(pruned)
    plot_drift(plt, all_rows, labels, os.path.join(args.out_dir, "drift_by_token_group.png"))
    if len(labels) >= 2:
        plot_delta(plt, all_rows, labels[0], labels[1],
                   os.path.join(args.out_dir, "drift_delta_%s_minus_%s.png" % (labels[0], labels[1])))

    # ---- verdict ----
    def mean_cos(label: str, group: str) -> float:
        values = [r["cos_to_dense_mean"] for r in all_rows
                  if r["model"] == label and r["token_group"] == group]
        return float(np.mean(values)) if values else float("nan")

    print("\n=== mean cosine to dense, averaged over T5 encoder blocks ===")
    for label in labels:
        print("  %-8s  visual=%.4f   text=%.4f"
              % (label, mean_cos(label, "visual"), mean_cos(label, "text")))
    if kl_rows:
        print("\n=== answer-logit KL(dense || pruned) ===")
        for row in kl_rows:
            print("  %-8s  mean=%.4f  median=%.4f  p90=%.4f"
                  % (row["model"], row["kl_dense_to_pruned_mean"],
                     row["kl_dense_to_pruned_median"], row["kl_dense_to_pruned_p90"]))

    summary: Dict[str, Any] = {
        "eval_json": os.path.abspath(args.eval_json),
        "samples": len(rows),
        "models": {label: os.path.abspath(path) for label, path in pruned.items()},
        "mean_cos_to_dense": {
            label: {group: mean_cos(label, group) for group in GROUPS} for label in labels
        },
        "answer_logit_kl": kl_rows,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if "split" in labels and "joint" in labels:
        d_text = mean_cos("split", "text") - mean_cos("joint", "text")
        d_visual = mean_cos("split", "visual") - mean_cos("joint", "visual")
        print("\n=== verdict ===")
        print("  split - joint, mean cosine to dense:  text=%+.4f   visual=%+.4f"
              % (d_text, d_visual))
        if d_text > 0 and d_text > 2.0 * abs(d_visual):
            print(
                "  Joint's damage is concentrated on the TEXT positions -- it holds the visual\n"
                "  prefix roughly as well as split does, but the language pathway drifts much\n"
                "  further from dense. Together with step 1 that closes the argument: joint\n"
                "  spent T5's weight budget on the visual prefix and paid for it in language.\n"
                "  Now go fix the statistic and re-prune -- that is the actual contribution."
            )
        elif d_visual > 0 and d_visual > 2.0 * abs(d_text):
            print(
                "  Joint's damage is concentrated on the VISUAL positions, which is the\n"
                "  opposite of the visual-hijack prediction. Re-check step 1, and look at the\n"
                "  ViT itself -- the sparsity allocation (step 0) is the likelier culprit."
            )
        else:
            print(
                "  Damage is spread across both token groups. Joint is uniformly worse rather\n"
                "  than trading one pathway for another, which points at the sparsity budget\n"
                "  (step 0) or at sequential contamination -- joint prunes ViT first and then\n"
                "  calibrates T5 through the already-pruned ViT -- rather than at the token mix."
            )

    print("\n[OK] wrote:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
