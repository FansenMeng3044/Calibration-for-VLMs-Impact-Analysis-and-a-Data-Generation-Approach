#!/usr/bin/env python3
"""Preflight checks for the BLIP2-T5 ATV migration validation run.

This script intentionally does not load BLIP2 or run CUDA work. It checks the
filesystem and configuration assumptions that would otherwise fail late during a
multi-seed pruning/evaluation run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


Row = Dict[str, str]
MASK_BASE_REQUIRED_GROUPS = ("t5_model", "visual_encoder", "Qformer", "query_tokens", "t5_proj")


def parse_bool_text(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def split_words(text: str) -> List[str]:
    return [x for x in str(text or "").replace(",", " ").split() if x]


def normalize_method(name: str) -> str:
    value = str(name or "").strip().lower()
    if value == "naive":
        return "wanda"
    if value == "amia":
        return "tamp"
    return value


def add_row(rows: List[Row], level: str, item: str, status: str, detail: str) -> None:
    rows.append({"level": level, "item": item, "status": status, "detail": detail})


def check_path(rows: List[Row], item: str, path: Path, *, kind: str, required: bool = True) -> None:
    if kind == "file":
        ok = path.is_file()
    elif kind == "dir":
        ok = path.is_dir()
    else:
        ok = path.exists()
    if ok:
        add_row(rows, "required" if required else "optional", item, "PASS", str(path))
    else:
        add_row(rows, "required" if required else "optional", item, "FAIL" if required else "WARN", str(path))


def ensure_writable_dir(rows: List[Row], item: str, path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".atv_preflight_write_test"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        add_row(rows, "required", item, "PASS", str(path))
    except Exception as exc:  # pragma: no cover - defensive filesystem guard
        add_row(rows, "required", item, "FAIL", "%s (%s)" % (path, exc))


def extract_json_rows(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("annotations", "data", "samples", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    return []


def validate_calibration_json(
    rows: List[Row],
    json_path: Path,
    images_dir: Path,
    expected_samples: int,
    image_field: str,
    text_fields: Sequence[str],
    image_probe_samples: int,
) -> None:
    if not json_path.is_file():
        return
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        add_row(rows, "required", "CC3M calibration JSON parse", "FAIL", "%s (%s)" % (json_path, exc))
        return

    data_rows = extract_json_rows(payload)
    if len(data_rows) >= expected_samples:
        add_row(
            rows,
            "required",
            "CC3M calibration sample count",
            "PASS",
            "rows=%d, expected>=%d" % (len(data_rows), expected_samples),
        )
    else:
        add_row(
            rows,
            "required",
            "CC3M calibration sample count",
            "FAIL",
            "rows=%d, expected>=%d" % (len(data_rows), expected_samples),
        )

    if not data_rows:
        return

    missing_image_field = [i for i, row in enumerate(data_rows[:expected_samples]) if not str(row.get(image_field, "")).strip()]
    missing_text_field = [
        i for i, row in enumerate(data_rows[:expected_samples])
        if not any(str(row.get(field, "")).strip() for field in text_fields)
    ]
    if missing_image_field:
        add_row(
            rows,
            "required",
            "CC3M calibration image field",
            "FAIL",
            "missing '%s' in first expected rows; examples=%s" % (image_field, missing_image_field[:8]),
        )
    else:
        add_row(rows, "required", "CC3M calibration image field", "PASS", "field='%s'" % image_field)

    if missing_text_field:
        add_row(
            rows,
            "required",
            "CC3M calibration text field",
            "FAIL",
            "none of %s present in first expected rows; examples=%s" % (list(text_fields), missing_text_field[:8]),
        )
    else:
        add_row(rows, "required", "CC3M calibration text field", "PASS", "fields=%s" % ",".join(text_fields))

    if not images_dir.is_dir() or missing_image_field:
        return
    probe_rows = data_rows[: min(len(data_rows), image_probe_samples)]
    missing_images = []
    for row in probe_rows:
        image_value = str(row.get(image_field, "")).strip()
        image_path = Path(image_value)
        if not image_path.is_absolute():
            image_path = images_dir / image_value
        if not image_path.is_file():
            missing_images.append(str(image_path))
    if missing_images:
        add_row(
            rows,
            "required",
            "CC3M calibration image file probe",
            "FAIL",
            "missing %d/%d probed images; examples=%s" % (len(missing_images), len(probe_rows), missing_images[:4]),
        )
    else:
        add_row(
            rows,
            "required",
            "CC3M calibration image file probe",
            "PASS",
            "checked=%d" % len(probe_rows),
        )


def check_mask_base_summary(rows: List[Row], mask_base_ckpt: Path) -> None:
    summary_json = Path(str(mask_base_ckpt) + ".summary.json")
    if not summary_json.is_file():
        add_row(
            rows,
            "warning",
            "dense full mask-base summary",
            "WARN",
            "summary not found: %s; run export_blip2_full_dense_state_dict.py for auditable group counts" % summary_json,
        )
        return
    try:
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
    except Exception as exc:
        add_row(rows, "required", "dense full mask-base summary parse", "FAIL", "%s (%s)" % (summary_json, exc))
        return

    groups = payload.get("groups", {})
    missing = []
    for group in MASK_BASE_REQUIRED_GROUPS:
        info = groups.get(group, {}) if isinstance(groups, dict) else {}
        try:
            tensors = int(info.get("tensors", 0))
        except Exception:
            tensors = 0
        if tensors <= 0:
            missing.append(group)
    if missing:
        add_row(
            rows,
            "required",
            "dense full mask-base summary groups",
            "FAIL",
            "missing/empty groups in %s: %s" % (summary_json, ",".join(missing)),
        )
    else:
        add_row(
            rows,
            "required",
            "dense full mask-base summary groups",
            "PASS",
            "%s contains %s" % (summary_json, ",".join(MASK_BASE_REQUIRED_GROUPS)),
        )


def seed_tamp_ckpt(seed: str, args: argparse.Namespace) -> Tuple[str, str]:
    env_key = "CKPT_TAMP_SEED%s" % seed
    by_env = os.environ.get(env_key, "").strip()
    by_cli = args.ckpt_tamp_seed.get(seed, "")
    if by_cli:
        return by_cli, "--ckpt_tamp_seed %s=..." % seed
    if by_env:
        return by_env, env_key
    if args.ckpt_tamp_template:
        return (
            args.ckpt_tamp_template.replace("{seed}", seed).replace("%SEED%", seed),
            "--ckpt_tamp_template",
        )
    if args.ckpt_tamp and args.allow_shared_tamp_ckpt:
        return args.ckpt_tamp, "--ckpt_tamp shared"
    return "", ""


def expected_generated_ckpts(
    lavis_root: Path,
    stamp: str,
    seeds: Sequence[str],
    models: Sequence[str],
) -> Iterable[Tuple[str, Path]]:
    model_set = {normalize_method(x) for x in models}
    for seed in seeds:
        if "atv" in model_set:
            yield "ATV alpha=1 checkpoint seed %s" % seed, lavis_root / "pruned_checkpoint" / (
                "atv_cc3m_t5only_%s_seed%s.pth" % (stamp, seed)
            )
        if "wanda" in model_set:
            yield "naive Wanda checkpoint seed %s" % seed, lavis_root / "pruned_checkpoint" / (
                "naive_wanda_cc3m_t5only_%s_seed%s.pth" % (stamp, seed)
            )
        if "atv_alpha0" in model_set:
            yield "ATV alpha=0 checkpoint seed %s" % seed, lavis_root / "pruned_checkpoint" / (
                "atv_cc3m_t5only_alpha0_%s_seed%s.pth" % (stamp, seed)
            )
        if "atv_alpha4" in model_set:
            yield "ATV alpha=4 checkpoint seed %s" % seed, lavis_root / "pruned_checkpoint" / (
                "atv_cc3m_t5only_alpha4_%s_seed%s.pth" % (stamp, seed)
            )


def write_rows(path: Path, rows: Sequence[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["level", "item", "status", "detail"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check required inputs before running ATV migration validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--lavis_root", required=True, type=Path)
    p.add_argument("--original_atv_root", required=True, type=Path)
    p.add_argument("--base", required=True, type=Path)
    p.add_argument("--report_dir", required=True, type=Path)
    p.add_argument("--stamp", required=True)
    p.add_argument("--seeds", default="42 43 44")
    p.add_argument("--models", default="dense atv atv_alpha0 atv_alpha4 naive tamp")
    p.add_argument("--run_prune", default="1")
    p.add_argument("--run_eval", default="1")
    p.add_argument("--cc3m_json", type=Path, default=None)
    p.add_argument("--cc3m_images_dir", type=Path, default=None)
    p.add_argument("--expected_calib_samples", type=int, default=128)
    p.add_argument("--calib_image_field", default="image")
    p.add_argument("--calib_text_fields", default="caption,text,question")
    p.add_argument("--image_probe_samples", type=int, default=16)
    p.add_argument("--dense_pretrain_ckpt", type=Path, default=None)
    p.add_argument(
        "--mask_base_ckpt",
        type=Path,
        default=Path(os.environ["MASK_BASE_CKPT"]) if os.environ.get("MASK_BASE_CKPT") else None,
        help=(
            "Full dense BLIP2-T5 state_dict used for dense-base-aware pruning-mask inference. "
            "This must include T5/ViT keys; the partial blip2_pretrained_flant5xl.pth bridge "
            "checkpoint is not enough."
        ),
    )
    p.add_argument(
        "--allow_zero_only_mask_inference",
        action="store_true",
        help="Allow debugging without a dense mask-base checkpoint; not valid for strict evidence.",
    )
    p.add_argument("--mmbench_root", type=Path, default=None)
    p.add_argument("--mmmu_root", type=Path, default=None)
    p.add_argument("--mathvista_eval_json", type=Path, default=None)
    p.add_argument("--mathvista_images_dir", type=Path, default=None)
    p.add_argument("--ckpt_tamp", default=os.environ.get("CKPT_TAMP", ""))
    p.add_argument("--ckpt_tamp_template", default=os.environ.get("CKPT_TAMP_TEMPLATE", ""))
    p.add_argument("--ckpt_tamp_seed", action="append", default=[])
    p.add_argument("--allow_shared_tamp_ckpt", action="store_true")
    p.add_argument("--out_csv", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.ckpt_tamp_seed = dict(x.split("=", 1) for x in args.ckpt_tamp_seed if "=" in x)
    seeds = split_words(args.seeds)
    models = {normalize_method(x) for x in split_words(args.models)}
    run_prune = parse_bool_text(args.run_prune)
    run_eval = parse_bool_text(args.run_eval)

    lavis_root = args.lavis_root
    base = args.base
    report_dir = args.report_dir
    cc3m_json = args.cc3m_json or base / "CC3M_calib_128" / "cc3m_calib_128.json"
    cc3m_images_dir = args.cc3m_images_dir or base / "CC3M_calib_128" / "images"
    dense_ckpt = args.dense_pretrain_ckpt or base / "model_cache" / "torch" / "hub" / "checkpoints" / "blip2_pretrained_flant5xl.pth"
    mask_base_ckpt = args.mask_base_ckpt
    mmbench_root = args.mmbench_root or base / "MMBench_eval"
    mmmu_root = args.mmmu_root or base / "MMMU_single_image"
    mathvista_eval_json = args.mathvista_eval_json or base / "MathVista_eval_testmini_mc" / "mathvista_multi_choice_eval.json"
    mathvista_images_dir = args.mathvista_images_dir or base / "MathVista_eval_testmini_mc" / "images"
    out_csv = args.out_csv or report_dir / "atv_preflight_report.csv"

    rows: List[Row] = []
    check_path(rows, "LAVIS root", lavis_root, kind="dir")
    check_path(rows, "original ATV root", args.original_atv_root, kind="dir")
    check_path(rows, "original ATV activation-aware pruner", args.original_atv_root / "qwen" / "activation_aware_pruner.py", kind="file")
    check_path(rows, "LAVIS evaluate_blip.py", lavis_root / "evaluate_blip.py", kind="file")
    check_path(rows, "LAVIS Wanda/ATV pruner", lavis_root / "lavis" / "compression" / "pruners" / "wanda_pruner.py", kind="file")
    check_path(rows, "ATV validator", lavis_root / "scripts" / "blip2" / "validate_atv_migration.py", kind="file")
    check_path(rows, "ATV strict audit script", lavis_root / "scripts" / "blip2" / "audit_atv_validation_report.py", kind="file")
    check_path(rows, "dense full state export script", lavis_root / "scripts" / "blip2" / "export_blip2_full_dense_state_dict.py", kind="file")
    check_path(rows, "CC3M runtime cfg materializer", lavis_root / "scripts" / "blip2" / "materialize_cc3m_calib_cfg.py", kind="file")
    check_path(rows, "ATV full verification driver", lavis_root / "scripts" / "blip2" / "run_atv_full_verify.sh", kind="file")
    check_path(rows, "ATV eval matrix driver", lavis_root / "scripts" / "blip2" / "run_atv_eval_matrix_fourbench.sh", kind="file")
    check_path(rows, "CC3M calibration yaml", lavis_root / "lavis" / "projects" / "blip2" / "eval" / "cc_prefix_derivative_compute_cc3m_calib128.yaml", kind="file")
    check_path(rows, "OKVQA eval yaml", lavis_root / "lavis" / "projects" / "blip2" / "eval" / "okvqa_zeroshot_flant5xl_eval_overall.yaml", kind="file")
    check_path(rows, "CC3M-128 JSON", cc3m_json, kind="file")
    check_path(rows, "CC3M-128 images dir", cc3m_images_dir, kind="dir")
    validate_calibration_json(
        rows,
        cc3m_json,
        cc3m_images_dir,
        args.expected_calib_samples,
        args.calib_image_field,
        split_words(args.calib_text_fields),
        args.image_probe_samples,
    )
    check_path(rows, "BLIP2 bridge pretrained checkpoint", dense_ckpt, kind="file")
    if run_prune and not args.allow_zero_only_mask_inference:
        if mask_base_ckpt is None:
            add_row(
                rows,
                "required",
                "dense full mask-base checkpoint",
                "FAIL",
                "set MASK_BASE_CKPT or --mask_base_ckpt to a full dense state_dict with T5/ViT keys",
            )
        else:
            check_path(rows, "dense full mask-base checkpoint", mask_base_ckpt, kind="file")
            if mask_base_ckpt.is_file():
                check_mask_base_summary(rows, mask_base_ckpt)
            if mask_base_ckpt.name == "blip2_pretrained_flant5xl.pth":
                add_row(
                    rows,
                    "warning",
                    "dense full mask-base checkpoint shape",
                    "WARN",
                    "blip2_pretrained_flant5xl.pth is usually a partial bridge checkpoint; export a full dense model.state_dict()",
                )
    elif run_prune and args.allow_zero_only_mask_inference:
        add_row(
            rows,
            "warning",
            "dense full mask-base checkpoint",
            "WARN",
            "zero-only mask inference allowed for debugging; strict mask evidence will not pass",
        )
    ensure_writable_dir(rows, "report dir writable", report_dir)

    if run_eval:
        check_path(rows, "MMBench eval root", mmbench_root, kind="dir")
        check_path(rows, "MMMU eval root", mmmu_root, kind="dir")
        check_path(rows, "MathVista eval JSON", mathvista_eval_json, kind="file")
        check_path(rows, "MathVista images dir", mathvista_images_dir, kind="dir")
        check_path(rows, "MMBench/MMMU eval script", lavis_root / "scripts" / "blip2" / "mmmu_eval_by_discipline.py", kind="file")
        check_path(rows, "MathVista eval script", lavis_root / "scripts" / "blip2" / "mathvista_mc_eval.py", kind="file")

    if run_eval and not run_prune:
        for label, path in expected_generated_ckpts(lavis_root, args.stamp, seeds, models):
            check_path(rows, label, path, kind="file", required=True)

    if run_eval and "tamp" in models:
        for seed in seeds:
            ckpt, source = seed_tamp_ckpt(seed, args)
            if not ckpt:
                add_row(
                    rows,
                    "required",
                    "TAMP/AMIA checkpoint seed %s" % seed,
                    "FAIL",
                    "set CKPT_TAMP_SEED%s, --ckpt_tamp_template, or CKPT_TAMP with --allow_shared_tamp_ckpt" % seed,
                )
            else:
                check_path(rows, "TAMP/AMIA checkpoint seed %s (%s)" % (seed, source), Path(ckpt), kind="file")
        if args.ckpt_tamp and args.allow_shared_tamp_ckpt:
            add_row(rows, "warning", "shared TAMP/AMIA checkpoint", "WARN", "shared baseline checkpoint weakens seed-specific provenance")

    if not seeds:
        add_row(rows, "required", "seed list", "FAIL", "empty --seeds")
    if "atv" not in models:
        add_row(rows, "warning", "model list", "WARN", "MODELS does not include ATV alpha=1")
    if run_eval and not {"dense", "atv", "wanda", "tamp"}.issubset(models):
        add_row(rows, "warning", "model list", "WARN", "strict paper matrix expects dense, atv, naive/wanda, and tamp/amia")

    write_rows(out_csv, rows)
    failures = [r for r in rows if r["level"] == "required" and r["status"] == "FAIL"]
    warnings = [r for r in rows if r["status"] == "WARN"]
    for row in rows:
        prefix = "[%s]" % row["status"]
        print("%s %s: %s" % (prefix, row["item"], row["detail"]))
    print("[OK] wrote preflight report: %s" % out_csv)
    if warnings:
        print("[WARN] preflight warnings: %d" % len(warnings))
    if failures:
        print("[FATAL] preflight failures: %d" % len(failures))
        return 2
    print("[OK] preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
