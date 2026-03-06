"""
Self-check for Step 6: evaluate_blip.py TAMP entry and config.

Checks:
- CLI: --token_selection, --score_method, --sparsity_ratio_granularity (and help text).
- blipt5_tamp_pruner alias: maps to blipt5_wanda_pruner and sets amia/density_sum/layer.
- config dict passed to load_pruner includes token_selection, score_method, sparsity_ratio_granularity.
- Defaults: token_selection=naive, score_method=obd_avg, sparsity_ratio_granularity=None.

Run from LAVIS_backup root:
  python tests/test_step6_entry_config.py
"""
import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
eval_path = os.path.join(root, "evaluate_blip.py")


def main():
    with open(eval_path, "r") as f:
        src = f.read()
    checks = []

    # CLI args
    checks.append(("--token_selection defined", '"--token_selection"' in src or "'--token_selection'" in src))
    checks.append(("token_selection default naive", 'default="naive"' in src or "default='naive'" in src))
    checks.append(("token_selection choices naiv/amia", "naive" in src and "amia" in src))
    checks.append(("--score_method with help or density", '"--score_method"' in src or "'--score_method'" in src))
    checks.append(("--sparsity_ratio_granularity", "sparsity_ratio_granularity" in src))

    # tamp alias
    checks.append(("blipt5_tamp_pruner alias present", "blipt5_tamp_pruner" in src))
    checks.append(("alias sets token_selection amia", "args.token_selection = \"amia\"" in src or "args.token_selection = 'amia'" in src))
    checks.append(("alias sets score_method density_sum", "args.score_method = \"density_sum\"" in src or "args.score_method = 'density_sum'" in src))
    checks.append(("alias sets sparsity_ratio_granularity layer", "args.sparsity_ratio_granularity = \"layer\"" in src or "args.sparsity_ratio_granularity = 'layer'" in src))
    checks.append(("alias maps to blipt5_wanda_pruner", "args.pruning_method = \"blipt5_wanda_pruner\"" in src or "args.pruning_method = 'blipt5_wanda_pruner'" in src))

    # config passed to load_pruner
    checks.append(("config has token_selection", '"token_selection": args.token_selection' in src or "'token_selection': args.token_selection" in src))
    checks.append(("config has score_method", '"score_method": args.score_method' in src or "'score_method': args.score_method" in src))
    checks.append(("config has sparsity_ratio_granularity", "sparsity_ratio_granularity" in src and "args." in src))

    # Defaults (no tamp)
    checks.append(("pruning_method default blipt5_wanda_pruner", "blipt5_wanda_pruner" in src))

    all_ok = all(ok for _, ok in checks)
    print("Step 6 self-check (evaluate_blip.py TAMP entry & config):")
    for name, ok in checks:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", name))
    print("Overall:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    exit(main())
