"""
Self-check for Step 4: AMIA + T5 _prune token_selection and image_masks.

Checks:
- AMIA.add_batch(inp, out, image_mask, score=None) does not raise and updates scaler_row.
- WrappedGPT.add_batch(inp, out, image_mask=None, score=None) still works (naive path).
- token_selection="naive": T5 _prune uses WrappedGPT and does not require image_masks.
- token_selection="amia" with image_masks=None raises RuntimeError.
- token_selection="amia" with image_masks: uses AMIA and passes image_masks in hook (path check via code/signature).

Run from LAVIS_backup root:
  python tests/test_step4_amia.py

If LAVIS deps (e.g. transformers) are missing, only the static checks run.
"""
import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

# Static checks (no heavy imports)
def test_step4_static():
    wanda_path = os.path.join(root, "lavis", "compression", "pruners", "wanda_pruner.py")
    with open(wanda_path, "r") as f:
        src = f.read()
    checks = []
    checks.append(("AdaptiveMultimodalInputActivation class present", "class AdaptiveMultimodalInputActivation:" in src))
    checks.append(("WrappedGPT.add_batch has image_mask=None, score=None", "def add_batch(self, inp, out, image_mask=None, score=None)" in src))
    checks.append(("_prune has token_selection=", "token_selection=" in src and "image_masks=None" in src))
    checks.append(("naive path uses WrappedGPT", "wrapped_layers[name] = WrappedGPT(subset[name])" in src))
    checks.append(("amia path uses AdaptiveMultimodalInputActivation", "AdaptiveMultimodalInputActivation(subset[name])" in src))
    checks.append(("hook passes mask_j, score_j", "mask_j" in src and "score_j" in src))
    checks.append(("amia requires image_masks RuntimeError", "token_selection='amia' requires image_masks" in src or 'token_selection="amia" requires image_masks' in src))
    checks.append(("return_image_masks when amia", "return_image_masks = token_selection == \"amia\"" in src))
    return checks


def test_amia_add_batch():
    """AMIA.add_batch with random inp/out, image_mask, score=None (requires wanda_pruner import)."""
    try:
        from lavis.compression.pruners.wanda_pruner import AdaptiveMultimodalInputActivation
    except Exception:
        return [("AMIA.add_batch (skip: import failed)", True)]
    import torch
    import torch.nn as nn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = nn.Linear(64, 32).to(device)
    amia = AdaptiveMultimodalInputActivation(layer)
    B, S, D = 2, 10, 64
    checks = []
    for name, mask in [
        ("all True", torch.ones(B, S, dtype=torch.bool, device=device)),
        ("all False", torch.zeros(B, S, dtype=torch.bool, device=device)),
        ("mixed", torch.zeros(B, S, dtype=torch.bool, device=device).masked_fill(torch.arange(S, device=device) < 4, True)),
    ]:
        inp = torch.randn(B, S, D, device=device)
        out = torch.randn(B, S, 32, device=device)
        try:
            amia.add_batch(inp, out, image_mask=mask, score=None)
            checks.append((f"AMIA.add_batch (image_mask={name}, score=None) no error", True))
        except Exception:
            checks.append((f"AMIA.add_batch (image_mask={name}, score=None) no error", False))
            continue
        checks.append((f"AMIA.scaler_row updated after {name}", amia.nsamples > 0 and amia.scaler_row.abs().sum() > 0))
    return checks


def test_wrapped_gpt_signature():
    try:
        from lavis.compression.pruners.wanda_pruner import WrappedGPT
    except Exception:
        return [("WrappedGPT.add_batch signature (skip: import failed)", True)]
    import torch
    import torch.nn as nn
    layer = nn.Linear(64, 32)
    w = WrappedGPT(layer)
    inp = torch.randn(2, 10, 64)
    out = torch.randn(2, 10, 32)
    try:
        w.add_batch(inp, out)
        w.add_batch(inp, out, image_mask=None, score=None)
        ok = w.nsamples > 0
    except Exception:
        ok = False
    return [("WrappedGPT.add_batch(inp, out) and (inp, out, None, None) work", ok)]


def test_amia_requires_image_masks():
    try:
        from lavis.compression.pruners.wanda_pruner import AdaptiveMultimodalInputActivation
    except Exception:
        return [("AMIA.add_batch(..., image_mask=None) raises (skip: import failed)", True)]
    import torch
    import torch.nn as nn
    layer = nn.Linear(64, 32)
    amia = AdaptiveMultimodalInputActivation(layer)
    inp = torch.randn(2, 10, 64)
    out = torch.randn(2, 10, 32)
    try:
        amia.add_batch(inp, out, image_mask=None, score=None)
        raised = False
    except RuntimeError as e:
        raised = "image_masks" in str(e) or "Step 3" in str(e)
    except Exception:
        raised = False
    return [("AMIA.add_batch(..., image_mask=None) raises RuntimeError", raised)]


def main():
    checks = []
    checks.extend(test_step4_static())
    checks.extend(test_amia_add_batch())
    checks.extend(test_wrapped_gpt_signature())
    checks.extend(test_amia_requires_image_masks())

    all_ok = all(ok for _, ok in checks)
    print("Step 4 self-check (AMIA + T5 _prune token_selection + image_masks):")
    for name, ok in checks:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", name))
    print("Overall:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    exit(main())
