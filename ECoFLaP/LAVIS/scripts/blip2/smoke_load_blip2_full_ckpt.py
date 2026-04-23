#!/usr/bin/env python3
"""
最小自检：用与 MMMU/MathVista eval 相同的 load_model(checkpoint=...) 加载整份 .pth。
OKVQA 的 evaluate_blip 路径更复杂；若本脚本通过，至少说明 full state_dict 可被 LAVIS 吃进。

用法（在 LAVIS 根目录）:
  python scripts/blip2/smoke_load_blip2_full_ckpt.py --ckpt pruned_checkpoint/xxx.pth
  python scripts/blip2/smoke_load_blip2_full_ckpt.py --ckpt xxx.pth --forward
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_LAVIS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _LAVIS_ROOT not in sys.path:
    sys.path.insert(0, _LAVIS_ROOT)

from lavis.models import load_model
from lavis.processors import load_processor
from PIL import Image
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="完整 BLIP2-T5 pruned .pth（state_dict）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--forward",
        action="store_true",
        help="额外跑一步 predict_answers（随机图+短文本）",
    )
    args = ap.parse_args()

    ckpt = os.path.abspath(args.ckpt)
    if not os.path.isfile(ckpt):
        print("[FATAL] 找不到 ckpt:", ckpt, file=sys.stderr)
        sys.exit(1)

    print("[smoke] load_model blip2_t5 pretrain_flant5xl checkpoint=", ckpt)
    model = load_model(
        "blip2_t5",
        "pretrain_flant5xl",
        is_eval=True,
        device=args.device,
        checkpoint=ckpt,
    )
    print("[smoke] OK 权重已加载。")

    if args.forward:
        vis = load_processor("blip_image_eval").build(image_size=224)
        txt = load_processor("blip_question").build(max_words=50)
        rng = np.random.default_rng(0)
        pil = Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
        image_tensor = vis(pil).unsqueeze(0).to(args.device)
        text_input = [txt("What is in the image?")]
        model.eval()
        with torch.no_grad():
            out = model.predict_answers(
                {"image": image_tensor, "text_input": text_input},
                num_beams=1,
                max_len=8,
                min_len=1,
                inference_method="generate",
            )
        print("[smoke] predict_answers sample:", out)


if __name__ == "__main__":
    main()
