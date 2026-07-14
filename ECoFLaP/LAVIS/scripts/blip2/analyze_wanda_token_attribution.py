#!/usr/bin/env python3
"""Step 1 -- who actually decides T5's Wanda mask: the image or the text?

Wanda scores a weight as ``|W_ij| * ||X_j||_2``, where ``||X_j||`` is the norm
of input channel j aggregated over *every token in the calibration sequence*
(``WrappedGPT.add_batch`` flattens [B, T, C] and does not care where a token
came from).

In BLIP2-T5 the encoder sequence is

    [32 Q-Former visual prefix tokens] + [caption text tokens] + [padding]

so if the visual prefix dominates that aggregate -- either because it
out-numbers the text (CC3M captions are short) or because Q-Former outputs
carry much larger activations than word embeddings -- then joint pruning picks
T5's mask *for the image*, and the language pathway is what gets thrown away.
Split pruning calibrates T5 on text alone and never has this problem.

This script tests that directly, on the DENSE model, with one forward pass.
No pruned checkpoint required.

It reports, per T5-encoder linear:

  * energy share:  how much of the Wanda statistic each token group contributes
  * per-token energy ratio: visual-vs-text magnitude with the token *count*
    effect divided out, so you can tell "there are more of them" apart from
    "each one is louder"
  * mask agreement: rebuild Wanda's mask from the all-token statistic (= what
    joint pruning uses) and from the text-only statistic (= the split-like
    statistic), and measure how much they overlap

The signature to look for: energy share of visual >> text, and
``overlap(all, visual)`` near 1.0 while ``overlap(all, text)`` sits near the
random baseline.  That means the joint mask *is* the visual mask.

Usage:
  python scripts/blip2/analyze_wanda_token_attribution.py \
      --calib_json /path/cc3m_calib128.json \
      --images_dir /path/images \
      --out_dir /path/out/step1_attribution \
      --max_samples 128 --batch_size 8 --sparsity 0.5
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from split_joint_analysis_common import (
    AUTO_INPUT_FIELDS,
    EncoderForward,
    build_vis_processor,
    ensure_dir,
    extract_text,
    iter_batches,
    load_batch_images,
    load_blip2,
    load_rows,
    mask_agreement,
    random_baselines,
    select_rows,
    setup_matplotlib,
    t5_encoder_linears,
    wanda_keep_mask,
    write_csv,
)

GROUPS = ("visual", "text", "pad")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attribute T5's Wanda calibration statistic to visual vs text tokens.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--calib_json", required=True, help="The SAME calibration set the pruning used.")
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Dense checkpoint. Omit to use the pretrained blip2_t5 weights.",
    )
    parser.add_argument("--model_name", default="blip2_t5")
    parser.add_argument("--model_type", default="pretrain_flant5xl")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--image_field", default="image")
    parser.add_argument("--text_field", default="auto")
    parser.add_argument("--max_txt_len", type=int, default=None)
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.5,
        help="Sparsity at which to rebuild the masks (match the pruning run).",
    )
    parser.add_argument(
        "--padding",
        choices=["longest", "max_length"],
        default="longest",
        help="Keep 'longest' to reproduce what the pruner actually saw.",
    )
    return parser.parse_args()


def plot_energy_share(plt, block_rows: List[Dict[str, Any]], path: str) -> None:
    if plt is None or not block_rows:
        return
    blocks = [r["block"] for r in block_rows]
    visual = np.asarray([r["energy_share_visual"] for r in block_rows])
    text = np.asarray([r["energy_share_text"] for r in block_rows])
    pad = np.asarray([r["energy_share_pad"] for r in block_rows])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(
        blocks,
        visual,
        text,
        pad,
        labels=["visual prefix", "text", "padding"],
        colors=["#E45756", "#4C78A8", "#BAB0AC"],
        alpha=0.88,
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("T5 encoder block")
    ax.set_ylabel("Share of Wanda statistic  (sum of squared activations)")
    ax.set_title("Which tokens produce T5's Wanda calibration statistic")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(min(blocks), max(blocks))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_per_token_ratio(plt, block_rows: List[Dict[str, Any]], path: str) -> None:
    if plt is None or not block_rows:
        return
    blocks = [r["block"] for r in block_rows]
    ratio = np.asarray([r["per_token_energy_ratio_visual_over_text"] for r in block_rows])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(blocks, ratio, marker="o", markersize=4, color="#E45756", linewidth=1.8)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="equal per-token energy")
    ax.set_yscale("log")
    ax.set_xlabel("T5 encoder block")
    ax.set_ylabel("mean per-token energy: visual / text")
    ax.set_title("Per-token activation magnitude, token-count effect removed")
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_mask_agreement(plt, block_rows: List[Dict[str, Any]], sparsity: float, path: str) -> None:
    if plt is None or not block_rows:
        return
    blocks = [r["block"] for r in block_rows]
    with_visual = np.asarray([r["overlap_all_vs_visual"] for r in block_rows])
    with_text = np.asarray([r["overlap_all_vs_text"] for r in block_rows])
    rand_overlap, _ = random_baselines(sparsity)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(blocks, with_visual, marker="o", markersize=4, linewidth=1.8,
            color="#E45756", label="joint mask vs visual-only mask")
    ax.plot(blocks, with_text, marker="s", markersize=4, linewidth=1.8,
            color="#4C78A8", label="joint mask vs text-only mask")
    ax.axhline(1.0, color="green", linestyle=":", linewidth=1.2, label="identical")
    ax.axhline(rand_overlap, color="black", linestyle="--", linewidth=1.0,
               label="independent masks (%.2f)" % rand_overlap)
    ax.set_xlabel("T5 encoder block")
    ax.set_ylabel("Kept-weight overlap")
    ax.set_ylim(min(0.0, rand_overlap - 0.1), 1.02)
    ax.set_title("Whose mask is the joint (all-token) mask?  sparsity=%.2f" % sparsity)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="lower right")
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

    rows, original_indices = select_rows(
        load_rows(os.path.abspath(args.calib_json)),
        args.max_samples,
        args.shuffle,
        args.seed,
    )
    print("device:", args.device)
    print("samples:", len(rows))

    model = load_blip2(
        args.model_name, args.model_type, args.device, args.ckpt, args.max_txt_len
    )
    print("model.max_txt_len:", model.max_txt_len)

    linears = t5_encoder_linears(model, torch)
    print("T5 encoder linears hooked:", len(linears))

    # name -> group -> per-input-channel sum of squared activations
    energy: Dict[str, Dict[str, Any]] = {}
    for _, _, key, module in linears:
        in_features = module.weight.shape[1]
        energy[key] = {
            g: torch.zeros(in_features, dtype=torch.float64, device="cpu") for g in GROUPS
        }

    token_counts: Dict[str, float] = {g: 0.0 for g in GROUPS}

    handles = []

    def make_hook(key: str):
        def hook(_module: Any, inputs: Any, _output: Any) -> None:
            x = inputs[0].detach()
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = x.to(torch.float32)
            masks = EncoderForward.current_masks()
            squared = x * x
            for group in GROUPS:
                mask = masks[group].to(x.device).unsqueeze(-1).to(x.dtype)  # [B, T, 1]
                energy[key][group] += (squared * mask).sum(dim=(0, 1)).double().cpu()

        return hook

    for _, _, key, module in linears:
        handles.append(module.register_forward_hook(make_hook(key)))

    forward = EncoderForward(model, torch, padding=args.padding)
    vis_processor = build_vis_processor(args.image_size)

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
            out = forward.run(images, texts, args.device)
            for group in GROUPS:
                token_counts[group] += float(out["%s_mask" % group].sum().item())
            if batch_index % 5 == 0:
                print("  batch %d  seq_len=%d  num_query=%d"
                      % (batch_index, out["seq_len"], out["num_query"]))
    finally:
        for handle in handles:
            handle.remove()

    total_tokens = sum(token_counts.values())
    print("\n=== token census over the calibration set ===")
    for group in GROUPS:
        print("  %-7s %10.0f  (%.1f%% of all encoder positions)"
              % (group, token_counts[group], 100.0 * token_counts[group] / max(total_tokens, 1.0)))

    # ---- per-linear statistics + mask rebuild ----
    rand_overlap, rand_iou = random_baselines(args.sparsity)
    linear_rows: List[Dict[str, Any]] = []

    for block_index, sub_name, key, module in linears:
        e_vis = energy[key]["visual"]
        e_txt = energy[key]["text"]
        e_pad = energy[key]["pad"]
        e_all = e_vis + e_txt + e_pad

        total_energy = float(e_all.sum().item())
        if total_energy <= 0:
            continue

        share_vis = float(e_vis.sum().item()) / total_energy
        share_txt = float(e_txt.sum().item()) / total_energy
        share_pad = float(e_pad.sum().item()) / total_energy

        per_tok_vis = float(e_vis.sum().item()) / max(token_counts["visual"], 1.0)
        per_tok_txt = float(e_txt.sum().item()) / max(token_counts["text"], 1.0)
        ratio = per_tok_vis / per_tok_txt if per_tok_txt > 0 else float("inf")

        weight = module.weight.detach().to("cpu")
        keep_all = wanda_keep_mask(weight, e_all, args.sparsity, torch)
        keep_vis = wanda_keep_mask(weight, e_vis, args.sparsity, torch)
        keep_txt = wanda_keep_mask(weight, e_txt, args.sparsity, torch)
        # text+pad = the statistic a text-only calibration pass would have produced
        keep_txtpad = wanda_keep_mask(weight, e_txt + e_pad, args.sparsity, torch)

        ov_vis, iou_vis = mask_agreement(keep_all, keep_vis, torch)
        ov_txt, iou_txt = mask_agreement(keep_all, keep_txt, torch)
        ov_txtpad, iou_txtpad = mask_agreement(keep_all, keep_txtpad, torch)

        linear_rows.append(
            {
                "block": block_index,
                "linear": sub_name,
                "sparsity_key": key,
                "in_features": int(weight.shape[1]),
                "out_features": int(weight.shape[0]),
                "energy_share_visual": share_vis,
                "energy_share_text": share_txt,
                "energy_share_pad": share_pad,
                "per_token_energy_visual": per_tok_vis,
                "per_token_energy_text": per_tok_txt,
                "per_token_energy_ratio_visual_over_text": ratio,
                "overlap_all_vs_visual": ov_vis,
                "overlap_all_vs_text": ov_txt,
                "overlap_all_vs_textpad": ov_txtpad,
                "iou_all_vs_visual": iou_vis,
                "iou_all_vs_text": iou_txt,
                "iou_all_vs_textpad": iou_txtpad,
            }
        )

    write_csv(os.path.join(args.out_dir, "per_linear_token_attribution.csv"), linear_rows)

    # ---- aggregate to blocks ----
    by_block: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in linear_rows:
        by_block[int(row["block"])].append(row)

    numeric_keys = [
        "energy_share_visual",
        "energy_share_text",
        "energy_share_pad",
        "per_token_energy_ratio_visual_over_text",
        "overlap_all_vs_visual",
        "overlap_all_vs_text",
        "overlap_all_vs_textpad",
        "iou_all_vs_visual",
        "iou_all_vs_text",
    ]
    block_rows: List[Dict[str, Any]] = []
    for block_index in sorted(by_block):
        group = by_block[block_index]
        row: Dict[str, Any] = {"block": block_index, "num_linears": len(group)}
        for key in numeric_keys:
            values = np.asarray([float(item[key]) for item in group], dtype=np.float64)
            values = values[np.isfinite(values)]
            row[key] = float(values.mean()) if values.size else float("nan")
        block_rows.append(row)

    write_csv(os.path.join(args.out_dir, "per_block_token_attribution.csv"), block_rows)

    plt = setup_matplotlib()
    plot_energy_share(plt, block_rows, os.path.join(args.out_dir, "energy_share_by_block.png"))
    plot_per_token_ratio(plt, block_rows, os.path.join(args.out_dir, "per_token_energy_ratio.png"))
    plot_mask_agreement(plt, block_rows, args.sparsity,
                        os.path.join(args.out_dir, "mask_agreement_by_block.png"))

    # ---- verdict ----
    mean_share_visual = float(np.mean([r["energy_share_visual"] for r in block_rows]))
    mean_share_text = float(np.mean([r["energy_share_text"] for r in block_rows]))
    mean_ratio = float(np.nanmean([r["per_token_energy_ratio_visual_over_text"] for r in block_rows]))
    mean_ov_vis = float(np.mean([r["overlap_all_vs_visual"] for r in block_rows]))
    mean_ov_txt = float(np.mean([r["overlap_all_vs_text"] for r in block_rows]))
    visual_token_frac = token_counts["visual"] / max(total_tokens, 1.0)

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "calib_json": os.path.abspath(args.calib_json),
                "checkpoint": os.path.abspath(args.ckpt) if args.ckpt else "pretrained",
                "samples": len(rows),
                "sparsity": args.sparsity,
                "padding": args.padding,
                "token_counts": token_counts,
                "visual_token_fraction": visual_token_frac,
                "mean_energy_share_visual": mean_share_visual,
                "mean_energy_share_text": mean_share_text,
                "mean_per_token_energy_ratio_visual_over_text": mean_ratio,
                "mean_overlap_all_vs_visual": mean_ov_vis,
                "mean_overlap_all_vs_text": mean_ov_txt,
                "random_overlap_baseline": rand_overlap,
                "random_iou_baseline": rand_iou,
            },
            handle,
            indent=2,
        )

    print("\n=== Wanda statistic attribution (mean over T5 encoder blocks) ===")
    print("  energy share   visual=%.3f  text=%.3f  pad=%.3f"
          % (mean_share_visual, mean_share_text,
             float(np.mean([r["energy_share_pad"] for r in block_rows]))))
    print("  per-token energy ratio visual/text = %.2fx" % mean_ratio)
    print("  visual tokens are %.1f%% of all encoder positions" % (100.0 * visual_token_frac))
    print("\n=== mask agreement (joint mask = all-token mask) ===")
    print("  overlap(joint, visual-only) = %.4f" % mean_ov_vis)
    print("  overlap(joint, text-only)   = %.4f" % mean_ov_txt)
    print("  independent-mask baseline   = %.4f" % rand_overlap)

    print("\n=== verdict ===")
    if mean_ov_vis > mean_ov_txt + 0.05:
        gap_v = mean_ov_vis - rand_overlap
        gap_t = mean_ov_txt - rand_overlap
        print(
            "  The joint mask tracks the VISUAL statistic, not the text one\n"
            "  (%.3f vs %.3f above the random baseline -- %.1fx).\n"
            "  Joint pruning is choosing T5's weights to serve the %d visual prefix tokens;\n"
            "  the weights that carry the language prior are what it drops. That is a\n"
            "  sufficient explanation for split > joint.\n"
            "  Next: confirm the damage lands on text positions (step 2), then fix the\n"
            "  statistic (exclude or per-group-normalize the visual prefix) and re-prune."
            % (gap_v, gap_t, gap_v / gap_t if gap_t > 1e-6 else float("inf"),
               int(round(token_counts["visual"] / max(len(rows), 1))))
        )
    elif mean_ov_txt > mean_ov_vis + 0.05:
        print(
            "  The joint mask tracks the TEXT statistic. The visual prefix is NOT hijacking\n"
            "  the calibration -- this hypothesis is dead. Look elsewhere: sequential\n"
            "  contamination (joint prunes ViT first, then calibrates T5 through the already\n"
            "  pruned ViT) or the sparsity allocation from step 0."
        )
    else:
        print(
            "  Neither statistic dominates the joint mask. The visual-hijack story does not\n"
            "  hold on its own; the two groups are contributing comparably. Go to step 2 and\n"
            "  see where the drift actually shows up before theorizing further."
        )

    print("\n[OK] wrote:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
