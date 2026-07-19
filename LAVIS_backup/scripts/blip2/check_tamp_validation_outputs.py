#!/usr/bin/env python3
"""Check validation evidence files for the BLIP2-T5 TAMP migration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing validation JSON: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def missing_keys(data: Dict[str, Any], keys: Iterable[str]) -> List[str]:
    return [key for key in keys if key not in data]


def check_static(path: Path) -> Dict[str, Any]:
    data = load_json(path)
    checks = data.get("checks", [])
    require(data.get("ok") is True, f"{path} does not report ok=true")
    require(isinstance(checks, list) and len(checks) > 0, f"{path} has no checks")
    failed = [item for item in checks if not item.get("passed")]
    require(not failed, f"{path} has failed checks: {failed[:3]}")
    return {"checks": len(checks)}


def check_core(path: Path) -> Dict[str, Any]:
    data = load_json(path)
    require(data.get("ok") is True, f"{path} does not report ok=true")
    required = ["score_shape", "amia_selected_rows", "valid_rows", "das_language_density"]
    miss = missing_keys(data, required)
    require(not miss, f"{path} missing keys: {miss}")
    require(int(data["amia_selected_rows"]) > 0, f"{path} AMIA selected no rows")
    require(int(data["valid_rows"]) > 0, f"{path} has no valid rows")
    require(abs(float(data["das_language_density"])) < 1e-8, f"{path} DAS includes padded language token")
    return {
        "amia_selected_rows": int(data["amia_selected_rows"]),
        "valid_rows": int(data["valid_rows"]),
    }


def check_runtime(path: Path) -> Dict[str, Any]:
    data = load_json(path)
    require(data.get("ok") is True, f"{path} does not report ok=true")
    required = [
        "mask_summary",
        "cache_summary",
        "encoder_input_summary",
        "amia_score_summary",
        "amia_selected_rows",
        "das_summary",
        "token_selection",
        "score_method",
        "sparsity_ratio_granularity",
        "importance_scope",
        "prune_t5",
        "prune_vit",
        "sparsity",
        "max_sparsity_per_layer",
        "amia_valid_rows_first_batch",
        "amia_selected_fraction_first_batch",
    ]
    miss = missing_keys(data, required)
    require(not miss, f"{path} missing keys: {miss}")
    require(data["token_selection"] == "amia", f"{path} did not use AMIA")
    require(data["score_method"] == "density_sum", f"{path} did not use DAS density_sum")
    require(data["sparsity_ratio_granularity"] == "layer", f"{path} did not use layer sparsity")
    require(data["importance_scope"] == "llm_only", f"{path} did not use T5-only TAMP runtime scope")
    require(data["prune_t5"] is True, f"{path} did not keep T5 pruning enabled")
    require(data["prune_vit"] is False, f"{path} should not prune ViT in TAMP runtime smoke")
    expected_max = min(1.0, float(data["sparsity"]) + 0.1)
    require(
        abs(float(data["max_sparsity_per_layer"]) - expected_max) < 1e-8,
        f"{path} max_sparsity_per_layer is not sparsity+0.1",
    )

    mask = data["mask_summary"]
    enc = data["encoder_input_summary"]
    score = data["amia_score_summary"]
    das = data["das_summary"]
    require(int(mask["physical_samples"]) > 0, f"{path} has no cached physical samples")
    require(int(mask["min_valid_text_tokens"]) > 0, f"{path} has no valid text tokens")
    require(int(enc["total_visual_query_tokens"]) > 0, f"{path} has no visual query tokens")
    require(int(enc["total_valid_text_tokens"]) > 0, f"{path} has no valid text tokens")
    require(float(score["valid_mean"]) > 0, f"{path} has non-positive AMIA valid score mean")
    require(float(score["invalid_abs_max"]) == 0.0, f"{path} has nonzero AMIA PAD scores")
    require(int(data["amia_selected_rows"]) > 0, f"{path} AMIA selected no rows")
    require(int(data["amia_valid_rows_first_batch"]) > 0, f"{path} has no AMIA valid rows")
    require(
        int(data["amia_selected_rows"]) <= int(data["amia_valid_rows_first_batch"]),
        f"{path} AMIA selected more rows than valid non-PAD tokens",
    )
    require(
        0.0 < float(data["amia_selected_fraction_first_batch"]) <= 1.0,
        f"{path} AMIA selected fraction is outside (0, 1]",
    )
    require(int(das.get("encoder_keys", 0)) > 0, f"{path} has no DAS encoder keys")
    require(int(das.get("decoder_fallback_keys", 0)) > 0, f"{path} has no decoder fallback keys")
    return {
        "rows_used": int(data.get("rows_used", 0)),
        "visual_query_tokens": int(enc["total_visual_query_tokens"]),
        "valid_text_tokens": int(enc["total_valid_text_tokens"]),
        "das_encoder_keys": int(das["encoder_keys"]),
        "das_decoder_fallback_keys": int(das["decoder_fallback_keys"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check static/core/runtime validation JSONs emitted by run_tamp_migration_validation.sh.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out_dir", default="lavis/output/tamp_migration_validation")
    parser.add_argument("--require_runtime", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    try:
        static_summary = check_static(out_dir / "static_validation.json")
    except Exception as exc:
        raise SystemExit(f"[ERROR][static] {exc}") from exc

    try:
        core_summary = check_core(out_dir / "core_smoke.json")
    except Exception as exc:
        raise SystemExit(f"[ERROR][core] {exc}") from exc

    runtime_path = out_dir / "runtime_smoke.json"
    runtime_summary: Dict[str, Any] = {}
    try:
        if runtime_path.is_file():
            runtime_summary = check_runtime(runtime_path)
        elif args.require_runtime:
            raise FileNotFoundError(f"missing required runtime validation JSON: {runtime_path}")
    except Exception as exc:
        raise SystemExit(f"[ERROR][runtime] {exc}") from exc

    summary = {
        "ok": True,
        "out_dir": str(out_dir),
        "static": static_summary,
        "core": core_summary,
        "runtime": runtime_summary,
        "runtime_present": runtime_path.is_file(),
        "runtime_required": bool(args.require_runtime),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
