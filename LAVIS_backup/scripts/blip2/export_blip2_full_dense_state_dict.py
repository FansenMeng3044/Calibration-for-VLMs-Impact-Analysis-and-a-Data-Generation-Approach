#!/usr/bin/env python3
"""Export a full dense BLIP2-T5 state_dict for pruning-mask comparison.

LAVIS BLIP2-T5 loads different parts from different places:
  - BLIP2 bridge/Q-Former checkpoint from --ckpt
  - Flan-T5 weights from the HuggingFace cache/config
  - EVA ViT weights from the model config/cache

The common blip2_pretrained_flant5xl.pth file is therefore not a full dense
state_dict. Use this script to materialize the complete in-memory model into a
single checkpoint that can be compared against pruned full-state checkpoints.
"""

from __future__ import annotations

import argparse
import os
import sys


_LAVIS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _LAVIS_ROOT not in sys.path:
    sys.path.insert(0, _LAVIS_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export full dense BLIP2-T5 state_dict.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", required=True, help="BLIP2 pretrain checkpoint, e.g. blip2_pretrained_flant5xl.pth")
    parser.add_argument("--out", required=True, help="Output .pth path for the full dense state_dict.")
    parser.add_argument("--model_name", default="blip2_t5")
    parser.add_argument("--model_type", default="pretrain_flant5xl")
    parser.add_argument("--device", default="cpu", help="Use cpu to avoid GPU memory pressure.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from lavis.models import load_model
    except ImportError as exc:
        raise SystemExit("Missing LAVIS/PyTorch dependency: %s" % exc) from exc

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print("Loading model:", args.model_name, args.model_type)
    print("Bridge/Q-Former ckpt:", os.path.abspath(args.ckpt))
    print("Device:", args.device)
    model = load_model(
        args.model_name,
        args.model_type,
        is_eval=True,
        device=args.device,
        checkpoint=args.ckpt,
    )
    model.eval()

    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state, out)

    t5_count = sum(1 for key in state if key.startswith("t5_model."))
    vit_count = sum(1 for key in state if key.startswith("visual_encoder.") or key.startswith("ln_vision."))
    print("[OK] wrote:", out)
    print("total tensors:", len(state))
    print("t5_model tensors:", t5_count)
    print("visual tensors:", vit_count)


if __name__ == "__main__":
    main()
