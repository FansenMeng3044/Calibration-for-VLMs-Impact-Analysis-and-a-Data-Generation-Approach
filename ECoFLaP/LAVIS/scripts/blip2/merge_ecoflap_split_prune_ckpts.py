#!/usr/bin/env python3
"""
Merge a T5-only pruned checkpoint with a ViT-only pruned checkpoint into one state_dict .pth.

Typical use: separate pruning produced
  - t5_only.pth  (ViT untouched / dense in that file)
  - vit_only.pth (T5 untouched in that file)

We take all keys from the T5-side file and overwrite visual_encoder.* (and optional ln_vision.*)
from the ViT-side file so a single file can be passed as --ckpt everywhere.

Usage (from ECoFLaP/LAVIS):
  python scripts/blip2/merge_ecoflap_split_prune_ckpts.py \\
    --t5_ckpt pruned_checkpoint/t5_only.pth \\
    --vit_ckpt pruned_checkpoint/vit_only.pth \\
    --out pruned_checkpoint/merged_t5_vit.pth
"""
from __future__ import annotations

import argparse
import os

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t5_ckpt", required=True, help="Checkpoint from T5-only prune (full state_dict)")
    ap.add_argument("--vit_ckpt", required=True, help="Checkpoint from ViT-only prune (full state_dict)")
    ap.add_argument("--out", required=True, help="Output merged .pth path")
    args = ap.parse_args()

    for p in (args.t5_ckpt, args.vit_ckpt):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    sd_t5 = torch.load(args.t5_ckpt, map_location="cpu")
    sd_vit = torch.load(args.vit_ckpt, map_location="cpu")

    if not isinstance(sd_t5, dict) or not isinstance(sd_vit, dict):
        raise TypeError("Expected top-level dict state_dict in both checkpoints")

    # Base: full state_dict from T5-side file (T5 + Q-Former + 非 ViT 部分与单侧 T5 剪枝一致).
    # Overwrite only ViT 子模块权重，与 load_blip2_t5_for_eval(vit_ckpt=..., t5_ckpt=...) 行为对齐.
    merged = dict(sd_t5)
    n_ov = 0
    for k, v in sd_vit.items():
        if k.startswith("visual_encoder.") or k.startswith("visual."):
            merged[k] = v
            n_ov += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save(merged, args.out)
    print(f"Wrote {args.out} (overwrote {n_ov} keys from vit_ckpt; base from t5_ckpt).")


if __name__ == "__main__":
    main()
