"""
Self-check for Step 3: T5 encoder calibration returns image_masks when return_image_masks=True.

Checks:
- Return value is 4-tuple (inps, outs, caches, image_masks) when return_image_masks=True.
- Return value is 3-tuple when return_image_masks=False (default).
- len(image_masks) == len(inps); image_masks[i].shape == (inps[i].shape[0], inps[i].shape[1]); dtype is bool.
- When model has no temp_label and return_image_masks=True, RuntimeError is raised.

Run from LAVIS_backup root (optional HF mirror):
  HF_ENDPOINT=https://hf-mirror.com python tests/test_step3_image_masks.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def _make_fake_dataloader(batch_size=2, num_batches=2, device="cuda"):
    """Yield batches compatible with Blip2T5 forward (image, text_input, text_output)."""
    for _ in range(num_batches):
        yield {
            "image": torch.randn(batch_size, 3, 224, 224, device=device, dtype=torch.float16),
            "text_input": ["a photo"] * batch_size,
            "text_output": ["photo"] * batch_size,
        }


def main():
    from omegaconf import OmegaConf
    from lavis.models.blip2_models.blip2_t5 import Blip2T5
    from lavis.compression.pruners.wanda_pruner import T5LayerWandaPruner

    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lavis/configs/models/blip2/blip2_pretrain_flant5xl.yaml")
    full_cfg = OmegaConf.load(cfg_path)
    cfg = full_cfg.model
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

    # Minimal pruner only for prepare_calibration_input_encoder (no real prune)
    class FakeDataLoader:
        def __init__(self, batches):
            self._batches = list(batches)
        def __iter__(self):
            return iter(self._batches)
    dataloader = FakeDataLoader(_make_fake_dataloader(batch_size=2, num_batches=2, device=device))
    pruner = T5LayerWandaPruner(
        model=model,
        data_loader=dataloader,
        model_prefix="t5_model",
        num_samples=4,
    )

    # ---- 1) return_image_masks=False: 3-tuple ----
    out3 = pruner.prepare_calibration_input_encoder(
        model, dataloader, device, "t5_model", n_samples=4,
        module_to_process="t5_model.encoder.block",
        return_image_masks=False,
    )
    checks = []
    checks.append(("return_image_masks=False returns 3-tuple", len(out3) == 3))
    if len(out3) == 3:
        inps, outs, caches = out3
        checks.append(("inps is list", isinstance(inps, list)))
        checks.append(("len(inps) > 0", len(inps) > 0))

    # ---- 2) return_image_masks=True: 4-tuple, image_masks aligned with inps ----
    dataloader2 = FakeDataLoader(_make_fake_dataloader(batch_size=2, num_batches=2, device=device))
    out4 = pruner.prepare_calibration_input_encoder(
        model, dataloader2, device, "t5_model", n_samples=4,
        module_to_process="t5_model.encoder.block",
        return_image_masks=True,
    )
    checks.append(("return_image_masks=True returns 4-tuple", len(out4) == 4))
    if len(out4) == 4:
        inps, outs, caches, image_masks = out4
        checks.append(("image_masks is list", isinstance(image_masks, list)))
        checks.append(("len(image_masks) == len(inps)", len(image_masks) == len(inps)))
        for i in range(len(inps)):
            B, S, D = inps[i].shape
            checks.append((f"image_masks[{i}].shape == (B,S) = ({B},{S})", image_masks[i].shape == (B, S)))
            checks.append((f"image_masks[{i}].dtype == torch.bool", image_masks[i].dtype == torch.bool))
        if len(image_masks) > 0:
            checks.append(("image_masks elements on CPU", image_masks[0].device.type == "cpu"))

    # ---- 3) model has no temp_label + return_image_masks=True => RuntimeError ----
    original_forward = model.forward
    def forward_then_clear_temp_label(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        if hasattr(model, "temp_label"):
            del model.temp_label
        return out
    model.forward = forward_then_clear_temp_label
    try:
        pruner.prepare_calibration_input_encoder(
            model, FakeDataLoader(_make_fake_dataloader(batch_size=1, num_batches=1, device=device)),
            device, "t5_model", n_samples=2,
            module_to_process="t5_model.encoder.block",
            return_image_masks=True,
        )
        raised = False
    except RuntimeError as e:
        raised = "temp_label" in str(e) or "Step 1" in str(e)
    finally:
        model.forward = original_forward
    checks.append(("RuntimeError when no temp_label and return_image_masks=True", raised))

    all_ok = all(ok for _, ok in checks)
    print("Step 3 self-check (calibration returns image_masks):")
    for name, ok in checks:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", name))
    print("Overall:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    exit(main())
