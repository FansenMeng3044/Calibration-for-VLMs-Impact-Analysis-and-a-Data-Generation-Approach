#!/usr/bin/env python3
"""Step 1b -- diff the masks the two runs ACTUALLY produced.

Step 1 simulated joint's Wanda statistic on the *dense* model and found that a
text-only statistic reproduces ~94% of that mask: swapping the calibration
distribution barely moves Wanda, because |W| dominates ||X||.

But the real joint run prunes ViT first and only then calibrates T5 -- through
the already-pruned ViT.  So the true joint T5 mask is not the one step 1
simulated.  Rather than model that, just read both checkpoints and compare the
zero patterns directly.  No GPU, no forward pass, no approximation.

Read it like this:

  * overlap(split_T5, joint_T5) close to step 1's simulated 0.94
        -> the two runs really did pick near-identical T5 weights. A ~6% mask
           difference then has to carry the entire accuracy gap; if the gap is
           large, the cause is NOT the T5 mask and you should look at ViT, at
           the merge, or at the eval harness.
  * overlap(split_T5, joint_T5) well BELOW 0.94
        -> the extra divergence is what pruning ViT first did to T5's
           calibration. Sequential contamination is real and quantified.
  * ViT overlap low while T5 overlap is high
        -> the damage lives on the visual side, not in the language model.

Usage:
  python scripts/blip2/compare_checkpoint_masks.py \
      --ckpt split=/path/merged_split.pth \
      --ckpt joint=/path/joint.pth \
      --out_dir /path/out/step1b_maskdiff
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from split_joint_analysis_common import (
    ensure_dir,
    load_state_dict,
    parse_labeled_paths,
    prunable_block_group,
    setup_matplotlib,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Directly compare the zero patterns of two pruned checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", action="append", required=True, metavar="LABEL=PATH",
                        help="Exactly two. e.g. --ckpt split=/a.pth --ckpt joint=/b.pth")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--simulated_overlap",
        type=float,
        default=None,
        help="overlap(joint, text-only) from step 1, drawn as a reference line "
             "(the T5 overlap you would expect if ViT-first contamination did nothing).",
    )
    return parser.parse_args()


def plot_overlap(plt, rows: List[Dict[str, Any]], simulated: float, path: str) -> None:
    if plt is None or not rows:
        return
    panels = []
    for model, submodel, title in (
        ("vit", "blocks", "ViT"),
        ("t5", "encoder", "T5 encoder"),
        ("t5", "decoder", "T5 decoder"),
    ):
        if any(r["model"] == model and r["submodel"] == submodel for r in rows):
            panels.append((model, submodel, title))
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.8), squeeze=False)
    for ax, (model, submodel, title) in zip(axes[0], panels):
        sel = [r for r in rows if r["model"] == model and r["submodel"] == submodel]
        sel.sort(key=lambda r: r["block"])
        blocks = [r["block"] for r in sel]
        ax.plot(blocks, [r["keep_overlap"] for r in sel],
                marker="o", markersize=4, linewidth=1.8, color="#4C78A8",
                label="actual mask overlap")
        baselines = [r["random_baseline"] for r in sel]
        ax.plot(blocks, baselines, linestyle="--", linewidth=1.0, color="black",
                label="independent masks")
        if simulated is not None and model == "t5":
            ax.axhline(simulated, color="#E45756", linestyle=":", linewidth=1.6,
                       label="step 1 simulated (%.3f)" % simulated)
        ax.set_title(title)
        ax.set_xlabel("Block index")
        ax.set_ylabel("Kept-weight overlap between the two runs")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.28)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Do split and joint keep the same weights?")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    import torch

    ckpts = parse_labeled_paths(args.ckpt)
    if len(ckpts) != 2:
        raise SystemExit("Need exactly two --ckpt entries, got %d" % len(ckpts))
    (label_a, path_a), (label_b, path_b) = list(ckpts.items())
    ensure_dir(args.out_dir)

    print("[read] %-8s %s" % (label_a, path_a))
    state_a = load_state_dict(path_a)
    print("[read] %-8s %s" % (label_b, path_b))
    state_b = load_state_dict(path_b)

    tensor_rows: List[Dict[str, Any]] = []
    missing = 0

    for name, tensor_a in state_a.items():
        try:
            ndim = tensor_a.dim()
        except AttributeError:
            continue
        info = prunable_block_group(name, ndim)
        if info is None:
            continue
        if name not in state_b:
            missing += 1
            continue
        model, submodel, index, group = info
        tensor_b = state_b[name]
        if tensor_a.shape != tensor_b.shape:
            missing += 1
            continue

        keep_a = (tensor_a != 0)
        keep_b = (tensor_b != 0)
        inter = float((keep_a & keep_b).sum().item())
        union = float((keep_a | keep_b).sum().item())
        n_a = float(keep_a.sum().item())
        n_b = float(keep_b.sum().item())
        numel = float(tensor_a.numel())
        if n_a == 0 or numel == 0:
            continue

        # Two independent masks keeping n_a and n_b of numel would share
        # n_a*n_b/numel weights on average.
        expected = (n_a * n_b / numel) / n_a

        tensor_rows.append(
            {
                "model": model,
                "submodel": submodel,
                "block": index,
                "group": group,
                "tensor": name,
                "numel": int(numel),
                "keep_%s" % label_a: int(n_a),
                "keep_%s" % label_b: int(n_b),
                "sparsity_%s" % label_a: 1.0 - n_a / numel,
                "sparsity_%s" % label_b: 1.0 - n_b / numel,
                "shared_kept": int(inter),
                "keep_overlap": inter / n_a,
                "iou": inter / union if union else float("nan"),
                "random_baseline": expected,
                "excess_over_random": inter / n_a - expected,
            }
        )

    if missing:
        print("[WARN] %d prunable tensors skipped (absent or shape-mismatched in %s)"
              % (missing, label_b))
    if not tensor_rows:
        raise SystemExit("No comparable prunable tensors found.")

    write_csv(os.path.join(args.out_dir, "per_tensor_mask_overlap.csv"), tensor_rows)

    # aggregate to blocks (weight by kept count so big matrices count more)
    by_block: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in tensor_rows:
        by_block[(row["model"], row["submodel"], row["block"])].append(row)

    block_rows: List[Dict[str, Any]] = []
    for (model, submodel, index), group in sorted(by_block.items()):
        kept = sum(float(r["keep_%s" % label_a]) for r in group)
        shared = sum(float(r["shared_kept"]) for r in group)
        base = sum(float(r["random_baseline"]) * float(r["keep_%s" % label_a]) for r in group) / kept
        block_rows.append(
            {
                "model": model,
                "submodel": submodel,
                "block": index,
                "tensors": len(group),
                "keep_overlap": shared / kept,
                "random_baseline": base,
                "excess_over_random": shared / kept - base,
            }
        )
    write_csv(os.path.join(args.out_dir, "per_block_mask_overlap.csv"), block_rows)

    plt = setup_matplotlib()
    plot_overlap(plt, block_rows, args.simulated_overlap,
                 os.path.join(args.out_dir, "mask_overlap_by_block.png"))

    # ---- summary ----
    print("\n=== actual kept-weight overlap: %s vs %s ===" % (label_a, label_b))
    summary: Dict[str, Any] = {}
    for model in ("vit", "t5"):
        sel = [r for r in tensor_rows if r["model"] == model]
        if not sel:
            continue
        kept = sum(float(r["keep_%s" % label_a]) for r in sel)
        shared = sum(float(r["shared_kept"]) for r in sel)
        base = sum(float(r["random_baseline"]) * float(r["keep_%s" % label_a]) for r in sel) / kept
        overlap = shared / kept
        summary[model] = {"keep_overlap": overlap, "random_baseline": base}
        print("  %-4s  overlap=%.4f   (independent-mask baseline=%.4f)" % (model, overlap, base))

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoints": {label_a: path_a, label_b: path_b},
                "simulated_overlap_step1": args.simulated_overlap,
                "overlap": summary,
            },
            handle,
            indent=2,
        )

    print("\n=== verdict ===")
    t5 = summary.get("t5", {}).get("keep_overlap")
    vit = summary.get("vit", {}).get("keep_overlap")

    if t5 is not None and args.simulated_overlap is not None:
        delta = args.simulated_overlap - t5
        print("  T5: simulated %.4f  ->  actual %.4f   (gap = %.4f)"
              % (args.simulated_overlap, t5, delta))
        if delta > 0.03:
            print(
                "  The real joint T5 mask is measurably further from split than a pure change of\n"
                "  calibration distribution can explain. That extra divergence is what pruning ViT\n"
                "  FIRST did to T5's calibration -- sequential contamination, now quantified."
            )
        else:
            print(
                "  The real T5 masks are as close as the dense simulation predicted, so pruning\n"
                "  ViT first cost T5 essentially nothing. The T5 mask is NOT where split and joint\n"
                "  differ."
            )

    if t5 is not None and vit is not None:
        if vit < t5 - 0.03:
            print(
                "\n  ViT overlap (%.4f) is clearly below T5's (%.4f): the two runs disagree mainly\n"
                "  about the VISUAL encoder. Point step 2 at the ViT, not the language model."
                % (vit, t5)
            )
        elif t5 < vit - 0.03:
            print(
                "\n  T5 overlap (%.4f) is below ViT's (%.4f): the disagreement is concentrated in\n"
                "  the language model." % (t5, vit)
            )
        else:
            print(
                "\n  ViT (%.4f) and T5 (%.4f) disagree about equally. If both are high, the two\n"
                "  checkpoints are nearly the same model -- and a large accuracy gap between them\n"
                "  would then be suspicious. Check that the reported gap is real and reproducible\n"
                "  (same eval config, same seed, non-trivial vs run-to-run noise) BEFORE hunting\n"
                "  for a mechanism to explain it." % (vit, t5)
            )

    print("\n[OK] wrote:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
