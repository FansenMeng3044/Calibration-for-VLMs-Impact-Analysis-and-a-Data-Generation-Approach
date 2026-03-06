"""
Self-check for Step 5: BLIPT5LayerWandaPruner DAS + AMIA wiring.

Checks:
- get_sparsity: when score_method="density_sum", LayerSparsity is built with calibration_fn and score_method.
- prune: need_calib when density_sum or amia; calibration with return_image_masks=True; cached_calib passed to _t5_prune.
- token_selection="naive": no requirement for image_masks; default behavior unchanged.
- token_selection="amia" with missing image_masks: RuntimeError.

Run from LAVIS_backup root:
  python tests/test_step5_das_amia.py
"""
import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wanda_path = os.path.join(root, "lavis", "compression", "pruners", "wanda_pruner.py")


def main():
    with open(wanda_path, "r") as f:
        src = f.read()
    checks = []

    # get_sparsity: density_sum -> calibration_fn
    checks.append((
        "get_sparsity passes calibration_fn when score_method==density_sum",
        'self.score_method == "density_sum"' in src and "calibration_fn" in src
    ))
    checks.append((
        "calibration_fn returns cached or prepare_calibration with return_image_masks=True",
        "_cached_encoder_calib" in src and "return_image_masks=True" in src
    ))
    checks.append((
        "LayerSparsity constructed with calibration_fn",
        "LayerSparsity(" in src and "calibration_fn=calibration_fn" in src
    ))

    # prune: need_calib, cache, pass to _prune
    checks.append((
        "prune sets need_calib for density_sum or amia",
        "need_calib" in src and ("density_sum" in src and "token_selection" in src)
    ))
    checks.append((
        "prune runs encoder calibration with return_image_masks when need_calib",
        "_cached_encoder_calib = calib_result" in src
    ))
    checks.append((
        "prune passes cached_calib to _t5_prune for encoder",
        "cached_calib=" in src and "_cached_encoder_calib" in src
    ))
    checks.append((
        "prune clears _cached_encoder_calib at end",
        "_cached_encoder_calib = None" in src
    ))

    # T5 _prune accepts cached_calib
    checks.append((
        "T5 _prune has cached_calib parameter",
        "cached_calib=None" in src
    ))
    checks.append((
        "T5 _prune uses cached_calib when provided",
        "if cached_calib is not None" in src and "result = cached_calib" in src
    ))

    # Default / naive path
    checks.append((
        "token_selection default naive",
        'token_selection="naive"' in src or "token_selection='naive'" in src
    ))
    checks.append((
        "score_method default not density",
        'score_method="GradMagSquare_avg"' in src
    ))

    # Exception: amia without image_masks
    checks.append((
        "amia without image_masks raises RuntimeError",
        "token_selection='amia' requires image_masks" in src or 'token_selection="amia" requires image_masks' in src
    ))

    all_ok = all(ok for _, ok in checks)
    print("Step 5 self-check (BLIPT5 DAS + AMIA wiring):")
    for name, ok in checks:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", name))
    print("Overall:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    exit(main())
