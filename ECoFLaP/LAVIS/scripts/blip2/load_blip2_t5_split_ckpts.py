#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Load BLIP2-T5 for eval: single full ckpt, or ViT/T5 from two split prune checkpoints."""
from __future__ import annotations

from typing import Optional

import torch

from lavis.models import load_model


def load_blip2_t5_for_eval(
    device: str,
    *,
    ckpt: Optional[str] = None,
    vit_ckpt: Optional[str] = None,
    t5_ckpt: Optional[str] = None,
):
    """
    - ckpt: full state_dict (legacy).
    - vit_ckpt + t5_ckpt: ViT weights from vit-only prune file, T5 from t5-only prune file
      (same pattern as evaluate_blip.py --vit_pruned_checkpoint / --t5_pruned_checkpoint).
    """
    if ckpt is not None and (vit_ckpt is not None or t5_ckpt is not None):
        raise ValueError("Use either ckpt= or (vit_ckpt= and t5_ckpt=), not both")

    model = load_model(
        "blip2_t5",
        "pretrain_flant5xl",
        is_eval=True,
        device=device,
        checkpoint=None,
    )

    if ckpt is not None:
        model.load_checkpoint(ckpt)
        return model

    if vit_ckpt is None and t5_ckpt is None:
        return model

    if not vit_ckpt or not t5_ckpt:
        raise ValueError("vit_ckpt and t5_ckpt must both be set for split checkpoint loading")

    if getattr(model, "t5_model", None) is not None:
        prune_state_dict = torch.load(t5_ckpt, map_location="cpu")
        prune_state_dict = {k: v for k, v in prune_state_dict.items() if k.startswith("t5_model")}
        prune_state_dict = {k.replace("t5_model.", ""): v for k, v in prune_state_dict.items()}
        model.t5_model.load_state_dict(prune_state_dict)

    prune_state_dict = torch.load(vit_ckpt, map_location="cpu")
    model_prefix = None
    for candidate_prefix in ("visual.", "visual_encoder."):
        if any(k.startswith(candidate_prefix) for k in prune_state_dict.keys()):
            model_prefix = candidate_prefix
            break
    if model_prefix is None:
        raise RuntimeError("Could not find visual.* or visual_encoder.* keys in vit_ckpt")

    prune_state_dict = {k: v for k, v in prune_state_dict.items() if k.startswith(model_prefix)}
    prune_state_dict = {k.replace(model_prefix, ""): v for k, v in prune_state_dict.items()}

    original_state_dict = model.visual_encoder.state_dict()
    for k, v in prune_state_dict.items():
        if k in original_state_dict:
            original_state_dict[k] = v
    prune_state_dict = original_state_dict

    from lavis.models.eva_vit import interpolate_pos_embed

    interpolate_pos_embed(model.visual_encoder, prune_state_dict)
    model.visual_encoder.load_state_dict(prune_state_dict)

    return model
