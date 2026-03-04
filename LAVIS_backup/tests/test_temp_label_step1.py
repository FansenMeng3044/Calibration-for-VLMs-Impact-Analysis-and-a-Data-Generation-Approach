"""
Minimal self-check for Step 1: temp_label injection in Blip2T5.forward().

Run from LAVIS_backup root with HF mirror (recommended in China):
  HF_ENDPOINT=https://hf-mirror.com python tests/test_temp_label_step1.py

Note: First run downloads EVA-CLIP-G (~1.9GB) and flan-t5-small from HuggingFace;
      use HF_ENDPOINT to avoid timeout.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def main():
    from omegaconf import OmegaConf
    from lavis.models.blip2_models.blip2_t5 import Blip2T5

    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lavis/configs/models/blip2/blip2_pretrain_flant5xl.yaml")
    full_cfg = OmegaConf.load(cfg_path)
    cfg = full_cfg.model

    # Use small T5 to speed up; build model without loading LAVIS checkpoint (only HF weights)
    cfg.t5_model = "google/flan-t5-small"
    model = Blip2T5(
        vit_model=cfg.get("vit_model", "eva_clip_g"),
        img_size=cfg.get("image_size", 224),
        drop_path_rate=cfg.get("drop_path_rate", 0),
        use_grad_checkpoint=cfg.get("use_grad_checkpoint", False),
        vit_precision=cfg.get("vit_precision", "fp16"),
        freeze_vit=cfg.get("freeze_vit", True),
        num_query_token=cfg.get("num_query_token", 32),
        t5_model=cfg.get("t5_model", "google/flan-t5-small"),
        prompt=cfg.get("prompt", ""),
        max_txt_len=cfg.get("max_txt_len", 32),
        apply_lemmatizer=cfg.get("apply_lemmatizer", False),
    )

    model.eval()
    device = next(model.parameters()).device
    num_query_token = model.query_tokens.shape[1]

    batch_size = 2
    samples = {
        "image": torch.randn(batch_size, 3, 224, 224, device=device, dtype=torch.float16),
        "text_input": ["a photo of a cat"] * batch_size,
        "text_output": ["cat"] * batch_size,
    }

    with torch.no_grad():
        out = model(samples)

    # ---- Self-check ----
    checks = []
    has_attr = hasattr(model, "temp_label")
    checks.append(("hasattr(model, 'temp_label')", has_attr))

    if not has_attr:
        print("FAIL: model has no temp_label after forward.")
        for name, ok in checks:
            print("  ", name, "=>", ok)
        return 1

    t = model.temp_label
    checks.append(("model.temp_label.dtype == torch.bool", t.dtype == torch.bool))
    checks.append(("model.temp_label.shape[0] == batch_size", t.shape[0] == batch_size))
    encoder_seq_len = t.shape[1]
    num_query = min(num_query_token, encoder_seq_len)
    prefix_ok = t[:, :num_query].all().item()
    suffix_ok = (num_query >= encoder_seq_len) or (~t[:, num_query:]).all().item()
    checks.append(("temp_label[:, :num_query].all() == True", bool(prefix_ok)))
    checks.append(("temp_label[:, num_query:].any() == False", bool(suffix_ok)))
    checks.append(("forward return has 'loss'", "loss" in out))
    checks.append(("forward return has 'logits'", "logits" in out))

    all_ok = all(ok for _, ok in checks)
    print("Step 1 self-check (temp_label injection):")
    for name, ok in checks:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", name))
    print("Overall:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    exit(main())
