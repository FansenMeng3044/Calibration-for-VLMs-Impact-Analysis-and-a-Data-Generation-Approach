"""
Lightweight self-check for Step 3 (no Blip2T5): verifies prepare_calibration_input_encoder
signature, return_image_masks default, and 3/4-tuple return logic in code.
Run from LAVIS_backup root: python tests/test_step3_image_masks_light.py
"""
import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wanda_path = os.path.join(root, "lavis", "compression", "pruners", "wanda_pruner.py")

def main():
    with open(wanda_path, "r") as f:
        src = f.read()
    checks = []
    checks.append(("return_image_masks parameter present", "return_image_masks=False" in src or "return_image_masks = False" in src))
    checks.append(("except ValueError branch with image_masks", "if return_image_masks:" in src and "image_masks.append" in src))
    checks.append(("RuntimeError when no temp_label", "model.temp_label not found" in src and "RuntimeError" in src))
    checks.append(("return 4-tuple when return_image_masks", "return inps, outs, caches, image_masks" in src))
    checks.append(("return 3-tuple when not", "return inps, outs, caches" in src))
    checks.append(("_prune unpacks result for 3 or 4 elements", "result[0], result[1], result[2]" in src and ("result[3] if len(result) == 4" in src or "result[3] if len(result)==4" in src)))
    checks.append(("image_masks alignment assert", "len(image_masks) == len(inps)" in src and "image_masks[i].shape" in src))

    all_ok = all(ok for _, ok in checks)
    print("Step 3 lightweight self-check (signature & logic in wanda_pruner.py):")
    for name, ok in checks:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", name))
    print("Overall:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1

if __name__ == "__main__":
    exit(main())
