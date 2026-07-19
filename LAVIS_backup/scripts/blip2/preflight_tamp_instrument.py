#!/usr/bin/env python3
"""Pre-flight verification before running the TAMP calibration-probe experiment.

Answers one question: is the instrument trustworthy enough to run the full
5-calibration-set sweep and interpret the differences?

Checks (hard = blocks the run, warn = proceed with awareness):

  C1  config sanity            hard   paths exist; max_samples % batch_size == 0
                                      (ActivationDensity averages per BATCH, so a
                                      short trailing batch is over-weighted)
  C2  calibration data shape   warn   ordering structure in the JSON, duplicate
                                      samples, and valid-text length per set
  C3  calibration contract     hard   calib returns 5-tuple; [32 query][text][PAD]
                                      layout holds; attention masks align
  C4  determinism              hard   two in-process repeats are bit-identical
  C5  AMIA regime              hard   select ratio not degenerate (>0.95 / <0.05)
  C6  batch-size invariance    warn   DAS layer sparsity should not depend on the
                                      calibration batch size (nuisance parameter)
  C7  production equivalence   warn   DAS layer sparsity from this diagnostic path
                                      matches a real prune run's sparsity_dict yaml

C2's length check and the JSON structure check are model-free; pass --extra_calib
to include the other calibration sets in those without loading them through the model.

Example:

  CUDA_VISIBLE_DEVICES=0 python scripts/blip2/preflight_tamp_instrument.py \
    --calib_json /data/data2/mfs/MMBench_calibration/mmbench_calibration_train.json \
    --images_dir /data/data2/mfs/MMBench_calibration/images \
    --ckpt /data/data2/mfs/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth \
    --max_samples 16 --batch_size 8 \
    --extra_calib mmmu=/data/data2/mfs/MMMU_calibration/mmmu_calibration_train.json \
    --extra_calib cc3m=/data/data2/mfs/CC3M_calib_128/cc3m_calib_128.json \
    --reference_sparsity_yaml sparsity_dict/tamp_calibMMBench_XXX.yaml \
    --out_json preflight_tamp_instrument.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_LAVIS_ROOT = Path(__file__).resolve().parents[2]
if str(_LAVIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAVIS_ROOT))

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from diagnose_tamp_instrument import (  # noqa: E402
    TEXT_FIELDS,
    _pearson,
    das_layer_vector,
    iter_batches,
    load_rows,
    run_d1_amia_selection,
    spearman,
)

CATEGORICAL_FIELDS = ("category", "l2-category", "l2_category", "source", "split", "task", "topic")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-flight verification for the TAMP calibration probe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--calib_json", required=True, help="Primary calibration set (loaded through the model).")
    p.add_argument("--images_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument(
        "--extra_calib",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Additional calibration JSONs for model-free checks. Repeatable.",
    )
    p.add_argument("--model_name", default="blip2_t5")
    p.add_argument("--model_type", default="pretrain_flant5xl")
    p.add_argument("--device", default=None)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--max_samples", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_txt_len", type=int, default=None)
    p.add_argument("--sparsity", type=float, default=0.5)
    p.add_argument("--expected_query_tokens", type=int, default=32)
    p.add_argument("--planned_max_samples", type=int, default=128, help="What the full sweep will use.")
    p.add_argument("--alt_batch_size", type=int, default=4, help="Second batch size for C6.")
    p.add_argument("--reference_sparsity_yaml", default=None, help="sparsity_dict/<job>.yaml from a real prune run.")
    p.add_argument("--probe_blocks", default="0,11,23", help="Blocks probed for C5 (keep small).")
    p.add_argument("--probe_linears", default="SelfAttention.v,DenseReluDense.wo")
    p.add_argument("--skip_model_checks", action="store_true", help="Only run C1/C2 (no GPU needed).")
    p.add_argument("--out_json", default="preflight_tamp_instrument.json")
    return p.parse_args()


# --------------------------------------------------------------------- report


class Report:
    def __init__(self) -> None:
        self.checks: List[Dict[str, Any]] = []

    def add(self, cid: str, name: str, status: str, detail: Any = None, note: str = "") -> None:
        assert status in ("PASS", "WARN", "FAIL", "SKIP")
        self.checks.append(
            {"id": cid, "name": name, "status": status, "detail": detail, "note": note}
        )
        icon = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[status]
        print(f"{icon} {cid} {name}")
        if detail is not None:
            print(f"       {detail}")
        if note:
            print(f"       -> {note}")

    @property
    def failed(self) -> List[Dict[str, Any]]:
        return [c for c in self.checks if c["status"] == "FAIL"]

    @property
    def warned(self) -> List[Dict[str, Any]]:
        return [c for c in self.checks if c["status"] == "WARN"]


def tv_distance(a: collections.Counter, b: collections.Counter) -> float:
    """Total variation distance between two categorical distributions."""
    na, nb = sum(a.values()), sum(b.values())
    if na == 0 or nb == 0:
        return float("nan")
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in keys)


# --------------------------------------------------------------- model-free


def check_config(args: argparse.Namespace, rep: Report) -> None:
    missing = []
    if not os.path.isfile(args.calib_json):
        missing.append(f"--calib_json {args.calib_json}")
    if not os.path.isdir(args.images_dir):
        missing.append(f"--images_dir {args.images_dir}")
    if not os.path.isfile(args.ckpt):
        missing.append(f"--ckpt {args.ckpt}")
    if missing:
        rep.add("C1", "config sanity", "FAIL", detail="missing: " + ", ".join(missing))
        return

    problems = []
    if args.max_samples % args.batch_size != 0:
        problems.append(f"max_samples({args.max_samples}) % batch_size({args.batch_size}) != 0")
    if args.planned_max_samples % args.batch_size != 0:
        problems.append(
            f"planned_max_samples({args.planned_max_samples}) % batch_size({args.batch_size}) != 0"
        )
    if problems:
        rep.add(
            "C1",
            "config sanity",
            "FAIL",
            detail="; ".join(problems),
            note=(
                "ActivationDensity averages per BATCH (count += 1 per batch), so a short "
                "trailing batch is over-weighted in DAS. Keep sample counts divisible."
            ),
        )
        return
    rep.add(
        "C1",
        "config sanity",
        "PASS",
        detail=f"paths ok; {args.max_samples}%{args.batch_size}==0; planned {args.planned_max_samples} ok",
    )


def check_calib_data(args: argparse.Namespace, rep: Report) -> None:
    sets: List[Tuple[str, str]] = [("primary", args.calib_json)]
    for item in args.extra_calib:
        if "=" not in item:
            rep.add("C2", "calibration data", "FAIL", detail=f"--extra_calib must be LABEL=PATH, got {item}")
            return
        label, path = item.split("=", 1)
        sets.append((label, path))

    n = args.planned_max_samples
    per_set: Dict[str, Any] = {}
    structured: List[str] = []

    for label, path in sets:
        if not os.path.isfile(path):
            per_set[label] = {"error": f"missing {path}"}
            continue
        rows = load_rows(path, n)
        info: Dict[str, Any] = {"n_rows_used": len(rows)}

        if len(rows) < n:
            info["short_pool"] = f"only {len(rows)} rows available for planned {n}"

        # duplicate detection on the text field actually used for calibration
        texts = []
        for r in rows:
            for f in TEXT_FIELDS:
                v = r.get(f)
                if v:
                    texts.append(str(v).strip())
                    break
        dup = len(texts) - len(set(texts))
        info["duplicate_text_rows"] = dup

        # ordering structure: first half vs second half on categorical fields
        half = len(rows) // 2
        struct = {}
        for field in CATEGORICAL_FIELDS:
            if field not in rows[0]:
                continue
            a = collections.Counter(str(r.get(field)) for r in rows[:half])
            b = collections.Counter(str(r.get(field)) for r in rows[half:])
            d = tv_distance(a, b)
            struct[field] = round(d, 4)
            if d > 0.30:
                structured.append(f"{label}.{field}(TV={d:.2f})")
        info["halfsplit_tv_distance"] = struct
        per_set[label] = info

    note = ""
    status = "PASS"
    if structured:
        status = "WARN"
        note = (
            "Ordered/stratified JSON detected: " + ", ".join(structured) + ". "
            "The diagnostic uses RANDOM repeated split-half for the DAS noise floor, "
            "so this does not invalidate it, but do not use first/second-half splits elsewhere."
        )
    dup_sets = [k for k, v in per_set.items() if isinstance(v, dict) and v.get("duplicate_text_rows", 0) > 0]
    if dup_sets:
        status = "WARN"
        note += f" Duplicate calibration texts in: {dup_sets}."
    short = [k for k, v in per_set.items() if isinstance(v, dict) and "short_pool" in v]
    if short:
        status = "WARN"
        note += f" Pool smaller than planned_max_samples in: {short}."

    rep.add("C2", "calibration data shape", status, detail=per_set, note=note.strip())


# ------------------------------------------------------------- model checks


def build_pruner(mods, model, batches, n_rows, args, batch_size_note=""):
    BLIPT5LayerWandaPruner = mods["BLIPT5LayerWandaPruner"]
    return BLIPT5LayerWandaPruner(
        model=model,
        data_loader=batches,
        t5_prune_spec="24-%.6f-1.0-1.0" % (1.0 - args.sparsity),
        vit_prune_spec=None,
        t5_pruning_method="none",
        vit_pruning_method="none",
        num_samples=n_rows,
        num_data_first_stage=n_rows,
        sparsity_ratio_granularity="layer",
        max_sparsity_per_layer=min(1.0, args.sparsity + 0.1),
        score_method="density_sum",
        token_selection="amia",
        prune_t5=True,
        prune_vit=False,
        importance_scope="llm_only",
    )


def run_calibration(mods, pruner, model, batches, device, n_rows):
    torch = mods["torch"]
    T5LayerWandaPruner = mods["T5LayerWandaPruner"]
    with torch.no_grad():
        calib = T5LayerWandaPruner.prepare_calibration_input_encoder(
            pruner, model, batches, device, "t5_model", n_rows,
            module_to_process="t5_model.encoder.block", return_image_masks=True,
        )
    return calib


def check_contract(calib, expected_query: int, rep: Report) -> bool:
    if len(calib) < 5:
        rep.add(
            "C3", "calibration contract", "FAIL",
            detail=f"calibration returned {len(calib)} elements, expected 5",
            note="AMIA/DAS need temp_label + temp_encoder_atts on Blip2T5.",
        )
        return False
    inps, _outs, _caches, image_masks, attn_masks = calib[:5]
    problems = []
    if not (len(inps) == len(image_masks) == len(attn_masks)):
        problems.append("cached batch count mismatch")
    total_vis = total_text = total_pad = 0
    for j, (inp, img, attn) in enumerate(zip(inps, image_masks, attn_masks)):
        img_b, attn_b = img.bool(), attn.bool()
        if img_b.dim() == 1:
            img_b, attn_b = img_b.unsqueeze(0), attn_b.unsqueeze(0)
        B, S = img_b.shape
        if tuple(inp.shape[:2]) != (B, S):
            problems.append(f"batch {j}: mask {(B, S)} vs input {tuple(inp.shape[:2])}")
            continue
        if int(img_b[:, :expected_query].sum().item()) != B * expected_query:
            problems.append(f"batch {j}: first {expected_query} positions are not all visual")
        if int(img_b[:, expected_query:].sum().item()) != 0:
            problems.append(f"batch {j}: visual flags leak into the text suffix")
        if int(attn_b[:, :expected_query].sum().item()) != B * expected_query:
            problems.append(f"batch {j}: visual query tokens are masked out")
        total_vis += int(img_b[:, :expected_query].sum().item())
        total_text += int(((~img_b[:, expected_query:]) & attn_b[:, expected_query:]).sum().item())
        total_pad += int((~attn_b[:, expected_query:]).sum().item())
    if problems:
        rep.add("C3", "calibration contract", "FAIL", detail=problems[:5])
        return False
    rep.add(
        "C3", "calibration contract", "PASS",
        detail={
            "cached_batches": len(inps),
            "visual_tokens": total_vis,
            "valid_text_tokens": total_text,
            "pad_text_tokens": total_pad,
        },
    )
    return True


das_vector = das_layer_vector  # production get_sparsity path, shared with the diagnostic


def check_determinism(pruner, calib, sparsity: float, rep: Report) -> None:
    keys_a, va = das_vector(pruner, calib, sparsity)
    keys_b, vb = das_vector(pruner, calib, sparsity)
    if keys_a != keys_b:
        rep.add("C4", "determinism", "FAIL", detail="layer key sets differ between repeats")
        return
    diffs = [abs(x - y) for x, y in zip(va, vb)]
    max_diff = max(diffs) if diffs else 0.0
    if max_diff == 0.0:
        rep.add("C4", "determinism", "PASS", detail=f"{len(keys_a)} layers bit-identical across 2 repeats")
    else:
        rep.add(
            "C4", "determinism", "FAIL",
            detail=f"max |Δ| = {max_diff:.3e} over {len(keys_a)} layers",
            note=(
                "Non-deterministic DAS: run-to-run noise is mixed into any cross-calibration "
                "difference. Set torch.use_deterministic_algorithms(True) or treat this as the "
                "true noise floor before claiming any effect."
            ),
        )


def check_amia_regime(mods, model, pruner_mod, calib, args, rep: Report) -> None:
    torch = mods["torch"]
    recs = run_d1_amia_selection(
        torch, model, pruner_mod, calib, "preflight", args.probe_blocks, args.probe_linears
    )
    if not recs:
        rep.add("C5", "AMIA regime", "FAIL", detail="no AMIA selection observations recorded")
        return
    ratios = [r["select_ratio"] for r in recs]
    sel = [r["n_selected"] for r in recs]
    mean_ratio = sum(ratios) / len(ratios)
    zero_vis = sum(1 for r in recs if r["n_visual_selected"] == 0) / len(recs)
    detail = {
        "observations": len(recs),
        "select_ratio_mean": round(mean_ratio, 4),
        "select_ratio_min": round(min(ratios), 4),
        "select_ratio_max": round(max(ratios), 4),
        "n_selected_mean": round(sum(sel) / len(sel), 2),
        "zero_visual_selected_frac": round(zero_vis, 4),
    }
    if mean_ratio > 0.95:
        rep.add("C5", "AMIA regime", "FAIL", detail=detail,
                note="AMIA keeps ~everything: it is effectively not running. Re-calibrate sigma / MMD threshold.")
    elif mean_ratio < 0.05:
        rep.add("C5", "AMIA regime", "FAIL", detail=detail,
                note="AMIA collapses to a handful of tokens: scaler_row is estimated from too little data.")
    elif zero_vis > 0.05:
        rep.add("C5", "AMIA regime", "WARN", detail=detail,
                note=f"{zero_vis:.1%} of observations selected NO visual token; modality signal is unstable there.")
    else:
        rep.add("C5", "AMIA regime", "PASS", detail=detail)


def check_batch_invariance(mods, model, pruner_mod, args, rows, vis_processor, device, ref_vec, ref_keys, rep: Report) -> None:
    torch = mods["torch"]
    Image = mods["Image"]
    if args.max_samples % args.alt_batch_size != 0:
        rep.add("C6", "batch-size invariance", "SKIP",
                detail=f"max_samples({args.max_samples}) not divisible by alt_batch_size({args.alt_batch_size})")
        return
    batches = list(iter_batches(rows, args.images_dir, vis_processor, torch, Image, device, args.alt_batch_size))
    pruner = build_pruner(mods, model, batches, len(rows), args)
    calib = run_calibration(mods, pruner, model, batches, device, len(rows))
    keys, vec = das_vector(pruner, calib, args.sparsity)
    if keys != ref_keys:
        rep.add("C6", "batch-size invariance", "FAIL", detail="layer key sets differ across batch sizes")
        return
    diffs = [abs(x - y) for x, y in zip(ref_vec, vec)]
    detail = {
        "batch_sizes": [args.batch_size, args.alt_batch_size],
        "spearman": round(spearman(ref_vec, vec), 6),
        "max_abs_diff": round(max(diffs), 6),
        "mean_abs_diff": round(sum(diffs) / len(diffs), 6),
        "layer_sparsity_range": round(max(ref_vec) - min(ref_vec), 6),
    }
    ratio = detail["mean_abs_diff"] / max(detail["layer_sparsity_range"], 1e-9)
    if ratio > 0.10:
        rep.add("C6", "batch-size invariance", "WARN", detail=detail,
                note=(f"batch size moves DAS by {ratio:.1%} of its dynamic range; keep batch_size "
                      "identical across ALL calibration sets or it becomes a confound."))
    else:
        rep.add("C6", "batch-size invariance", "PASS", detail=detail)


def check_production_equivalence(args, ref_keys, ref_vec, rep: Report) -> None:
    path = args.reference_sparsity_yaml
    if not path:
        rep.add("C7", "production equivalence", "SKIP",
                detail="no --reference_sparsity_yaml given",
                note=("Strongest available check: pass sparsity_dict/<job>.yaml from a real prune "
                      "run with the SAME calibration and sample count to prove the diagnostic "
                      "path reproduces production DAS."))
        return
    if not os.path.isfile(path):
        rep.add("C7", "production equivalence", "FAIL", detail=f"missing {path}")
        return
    try:
        import yaml
    except ModuleNotFoundError:
        rep.add("C7", "production equivalence", "SKIP", detail="pyyaml not installed")
        return
    with open(path, "r", encoding="utf-8") as fh:
        ref = yaml.safe_load(fh)
    shared = [k for k in ref_keys if k in ref]
    if not shared:
        rep.add("C7", "production equivalence", "FAIL",
                detail="no overlapping encoder keys between diagnostic and reference yaml")
        return
    a = [ref_vec[ref_keys.index(k)] for k in shared]
    b = [float(ref[k]) for k in shared]
    diffs = [abs(x - y) for x, y in zip(a, b)]
    detail = {
        "shared_layers": len(shared),
        "spearman": round(spearman(a, b), 6),
        "pearson": round(_pearson(a, b), 6),
        "max_abs_diff": round(max(diffs), 6),
        "mean_abs_diff": round(sum(diffs) / len(diffs), 6),
    }
    if detail["max_abs_diff"] < 1e-6:
        rep.add("C7", "production equivalence", "PASS", detail=detail,
                note="diagnostic DAS is identical to the production prune run")
    elif detail["spearman"] > 0.99:
        rep.add("C7", "production equivalence", "WARN", detail=detail,
                note=("same ordering but not identical values -- likely a different sample count "
                      "or seed. Match them before quoting diagnostic numbers as production behaviour."))
    else:
        rep.add("C7", "production equivalence", "FAIL", detail=detail,
                note="diagnostic does not reproduce production DAS; do not extrapolate D1/D2 to real runs.")


# ------------------------------------------------------------------------ main


def main() -> int:
    args = parse_args()
    rep = Report()
    print("=" * 72)
    print("TAMP instrument pre-flight")
    print("=" * 72)

    check_config(args, rep)
    if rep.failed:
        return finish(args, rep)
    check_calib_data(args, rep)

    if args.skip_model_checks:
        return finish(args, rep)

    try:
        import torch
        from PIL import Image
        from lavis.models import load_model
        from lavis.processors import load_processor
        from lavis.compression.pruners import wanda_pruner as pruner_mod
        from lavis.compression.pruners.wanda_pruner import (
            BLIPT5LayerWandaPruner,
            T5LayerWandaPruner,
        )
    except ModuleNotFoundError as exc:
        rep.add("C3", "calibration contract", "FAIL",
                detail=f"missing runtime module: {exc.name}",
                note="Run inside the LAVIS/ecoflap conda env, or pass --skip_model_checks.")
        return finish(args, rep)

    mods = {
        "torch": torch, "Image": Image,
        "BLIPT5LayerWandaPruner": BLIPT5LayerWandaPruner,
        "T5LayerWandaPruner": T5LayerWandaPruner,
    }
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows(args.calib_json, args.max_samples)
    vis_processor = load_processor("blip_image_eval").build(image_size=args.image_size)

    print(f"\n[load] {args.model_name}/{args.model_type} device={device}")
    model = load_model(args.model_name, args.model_type, is_eval=True, device=device, checkpoint=args.ckpt)
    model.eval()
    if args.max_txt_len is not None:
        model.max_txt_len = int(args.max_txt_len)

    batches = list(iter_batches(rows, args.images_dir, vis_processor, torch, Image, device, args.batch_size))
    pruner = build_pruner(mods, model, batches, len(rows), args)
    calib = run_calibration(mods, pruner, model, batches, device, len(rows))

    if not check_contract(calib, args.expected_query_tokens, rep):
        return finish(args, rep)

    check_determinism(pruner, calib, args.sparsity, rep)
    ref_keys, ref_vec = das_vector(pruner, calib, args.sparsity)
    check_amia_regime(mods, model, pruner_mod, calib, args, rep)
    check_batch_invariance(mods, model, pruner_mod, args, rows, vis_processor, device, ref_vec, ref_keys, rep)
    check_production_equivalence(args, ref_keys, ref_vec, rep)

    return finish(args, rep)


def finish(args: argparse.Namespace, rep: Report) -> int:
    print("\n" + "=" * 72)
    n_fail, n_warn = len(rep.failed), len(rep.warned)
    if n_fail:
        print(f"RESULT: NOT READY -- {n_fail} hard failure(s), {n_warn} warning(s)")
        for c in rep.failed:
            print(f"  FAIL {c['id']} {c['name']}: {c['note'] or c['detail']}")
    elif n_warn:
        print(f"RESULT: READY WITH CAVEATS -- {n_warn} warning(s)")
        for c in rep.warned:
            print(f"  WARN {c['id']} {c['name']}: {c['note'] or ''}")
    else:
        print("RESULT: READY -- all checks passed")
    print("=" * 72)

    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "ready": n_fail == 0,
                "n_fail": n_fail,
                "n_warn": n_warn,
                "args": {k: v for k, v in vars(args).items()},
                "checks": rep.checks,
            },
            fh, indent=2, ensure_ascii=False, default=str,
        )
    print(f"[done] wrote {args.out_json}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
