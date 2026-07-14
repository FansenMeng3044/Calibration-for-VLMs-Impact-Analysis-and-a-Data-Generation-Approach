#!/usr/bin/env python3
"""Step 0 -- are split and joint even iso-sparse?

Counts actual zeros per prunable block in each checkpoint, using the exact same
block grouping the pruner uses (BLIPT5LayerWandaPruner.get_sparsity).  No GPU,
no model instantiation -- it just reads state_dicts.

This must run before any activation analysis.  In joint mode the sparsity
budget is allocated across ViT *and* T5 blocks from one pool
(``per_model_group=[t5, vit]`` with ``prune_per_model=False``), and the two
modalities' importance scores are not on a comparable scale.  If that pushed
sparsity onto one side, then "split beats joint" is a budget artifact and there
is nothing to explain at the activation level.

Usage:
  python scripts/blip2/analyze_split_joint_sparsity_allocation.py \
      --ckpt split=/path/merged_split.pth \
      --ckpt joint=/path/joint.pth \
      --out_dir /path/out/step0_sparsity \
      --target_sparsity 0.5
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, List

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
        description="Per-block realized sparsity for split vs joint checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeatable. e.g. --ckpt split=/a.pth --ckpt joint=/b.pth",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--target_sparsity",
        type=float,
        default=0.5,
        help="Nominal sparsity both runs were supposed to hit; drawn as a reference line.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Overall per-model sparsity gap above which the runs are called mismatched.",
    )
    return parser.parse_args()


def block_sparsity_table(path: str) -> List[Dict[str, object]]:
    state = load_state_dict(path)
    totals: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])  # group -> [zeros, params]
    meta: Dict[str, tuple] = {}

    for name, tensor in state.items():
        try:
            ndim = tensor.dim()
        except AttributeError:
            continue
        info = prunable_block_group(name, ndim)
        if info is None:
            continue
        model, submodel, index, group = info
        zeros = float((tensor == 0).sum().item())
        params = float(tensor.numel())
        totals[group][0] += zeros
        totals[group][1] += params
        meta[group] = (model, submodel, index)

    rows: List[Dict[str, object]] = []
    for group, (zeros, params) in totals.items():
        model, submodel, index = meta[group]
        rows.append(
            {
                "model": model,
                "submodel": submodel,
                "block": index,
                "group": group,
                "params": int(params),
                "zeros": int(zeros),
                "sparsity": zeros / params if params else float("nan"),
            }
        )
    rows.sort(key=lambda r: (r["model"], r["submodel"], r["block"]))
    return rows


def plot_curves(plt, per_ckpt: Dict[str, List[Dict[str, object]]], target: float, path: str) -> None:
    if plt is None:
        return
    panels = [("vit", "blocks", "ViT (visual_encoder.blocks)"),
              ("t5", "encoder", "T5 encoder"),
              ("t5", "decoder", "T5 decoder")]
    active = []
    for model, submodel, title in panels:
        if any(
            any(r["model"] == model and r["submodel"] == submodel for r in rows)
            for rows in per_ckpt.values()
        ):
            active.append((model, submodel, title))
    if not active:
        return

    fig, axes = plt.subplots(1, len(active), figsize=(6.2 * len(active), 4.6), squeeze=False)
    for ax, (model, submodel, title) in zip(axes[0], active):
        for label, rows in per_ckpt.items():
            sel = [r for r in rows if r["model"] == model and r["submodel"] == submodel]
            sel.sort(key=lambda r: r["block"])
            if not sel:
                continue
            ax.plot(
                [r["block"] for r in sel],
                [r["sparsity"] for r in sel],
                marker="o",
                markersize=3.5,
                linewidth=1.6,
                label=label,
            )
        ax.axhline(target, color="black", linestyle="--", linewidth=1.0, label="target")
        ax.set_title(title)
        ax.set_xlabel("Block index")
        ax.set_ylabel("Realized sparsity")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.28)
        ax.legend()
    fig.suptitle("Per-block realized sparsity (zeros / params)")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ckpts = parse_labeled_paths(args.ckpt)
    ensure_dir(args.out_dir)
    plt = setup_matplotlib()

    per_ckpt: Dict[str, List[Dict[str, object]]] = {}
    all_rows: List[Dict[str, object]] = []
    for label, path in ckpts.items():
        if not os.path.isfile(path):
            raise FileNotFoundError("Checkpoint not found: %s" % path)
        print("[read] %-8s %s" % (label, path))
        rows = block_sparsity_table(path)
        if not rows:
            raise ValueError(
                "No prunable blocks found in %s -- unexpected state_dict layout." % path
            )
        per_ckpt[label] = rows
        for row in rows:
            enriched = {"checkpoint": label}
            enriched.update(row)
            all_rows.append(enriched)

    write_csv(os.path.join(args.out_dir, "per_block_sparsity.csv"), all_rows)

    # Overall sparsity per (checkpoint, model).
    summary: List[Dict[str, object]] = []
    overall: Dict[tuple, float] = {}
    for label, rows in per_ckpt.items():
        agg: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])
        for row in rows:
            agg[str(row["model"])][0] += float(row["zeros"])
            agg[str(row["model"])][1] += float(row["params"])
        for model, (zeros, params) in agg.items():
            sparsity = zeros / params if params else float("nan")
            overall[(label, model)] = sparsity
            blocks = [r["sparsity"] for r in rows if r["model"] == model]
            summary.append(
                {
                    "checkpoint": label,
                    "model": model,
                    "params": int(params),
                    "zeros": int(zeros),
                    "overall_sparsity": sparsity,
                    "block_sparsity_min": float(np.min(blocks)),
                    "block_sparsity_max": float(np.max(blocks)),
                    "block_sparsity_std": float(np.std(blocks)),
                }
            )
    write_csv(os.path.join(args.out_dir, "overall_sparsity.csv"), summary)
    plot_curves(plt, per_ckpt, args.target_sparsity, os.path.join(args.out_dir, "per_block_sparsity.png"))

    print("\n=== overall realized sparsity ===")
    for row in summary:
        print(
            "  %-8s %-4s  overall=%.4f  per-block[min=%.3f max=%.3f std=%.3f]"
            % (
                row["checkpoint"],
                row["model"],
                row["overall_sparsity"],
                row["block_sparsity_min"],
                row["block_sparsity_max"],
                row["block_sparsity_std"],
            )
        )

    print("\n=== verdict ===")
    models = sorted({m for (_, m) in overall})
    labels = list(per_ckpt)
    mismatched = False
    for model in models:
        values = [(label, overall[(label, model)]) for label in labels if (label, model) in overall]
        if len(values) < 2:
            continue
        spread = max(v for _, v in values) - min(v for _, v in values)
        flag = "MISMATCH" if spread > args.tolerance else "ok"
        if spread > args.tolerance:
            mismatched = True
        print(
            "  %-4s  %s  spread=%.4f  (%s)"
            % (
                model,
                flag,
                spread,
                ", ".join("%s=%.4f" % (label, value) for label, value in values),
            )
        )

    if mismatched:
        print(
            "\n  The checkpoints are NOT iso-sparse. The accuracy gap is at least partly a\n"
            "  budget-allocation artifact, not a calibration-quality effect. Re-run the two\n"
            "  arms at matched per-model sparsity before reading anything into activations."
        )
    else:
        print(
            "\n  Iso-sparse within tolerance. The gap cannot be explained by how many weights\n"
            "  each side lost, so it has to come from *which* weights were chosen. Proceed to\n"
            "  step 1 (analyze_wanda_token_attribution.py)."
        )

    print("\n[OK] wrote:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
