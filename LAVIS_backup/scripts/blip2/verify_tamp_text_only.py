#!/usr/bin/env python3
"""Verification for TAMP's single-modality reduction (the --tamp_text_only path).

This path has never been executed before. It runs AMIA + DAS over text-only
calibration, with the diversity terms averaged over the modality pairs that
actually exist (s = s_l) instead of counting absent visual pairs as zero
similarity. This script proves the path works, proves the new branch is the one
being taken, and answers the only question that decides whether the direction is
worth pursuing:

    does anything survive? if AMIA keeps ~all tokens AND DAS comes out uniform,
    the whole path is vanilla Wanda with extra steps.

Checks (hard = blocks the run, warn = proceed with awareness):

  T1  view contract          hard  the text view exposes t5_model / maybe_autocast /
                                   temp_label (all-False) / temp_encoder_atts, and
                                   calibration returns the 5-tuple AMIA and DAS need
  T2  single-modality branch hard  cos_pairwise_density reports defined == ('l',) and
                                   the AMIA density equals s_l, not s_l/3 -- i.e. the
                                   five-line fix is actually in effect
  T3  old-vs-new delta       info  what the pre-fix code would have produced, so the
                                   size of the correction is on record
  T4  short-text safety      hard  no calibration sample has <2 valid tokens (which
                                   would raise from the new "no measurable pair" guard)
  T5  non-degeneracy         hard  GO/NO-GO: AMIA select_ratio is not ~1.0 or ~0, and
                                   DAS layer sparsity has non-trivial dynamic range
  T6  determinism            hard  two DAS repeats are bit-identical
  T7  vs degraded Wanda      warn  DAS layer sparsity differs from uniform, i.e. this
                                   path is not just the old degenerate path renamed
  T8  multimodal untouched   warn  with a multimodal calib, every term is still defined,
                                   so the multimodal path keeps its byte-identical branch

Example:

  CUDA_VISIBLE_DEVICES=0 python scripts/blip2/verify_tamp_text_only.py \
    --ckpt /data/data2/mfs/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth \
    --text_calib /data/data2/mfs/text_calib_128/c4_text_calib_128.json \
    --max_samples 32 --batch_size 8 \
    --mm_calib /data/data2/mfs/MMBench_calibration/mmbench_calibration_train.json \
    --mm_images /data/data2/mfs/MMBench_calibration/images
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_LAVIS_ROOT = Path(__file__).resolve().parents[2]
if str(_LAVIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAVIS_ROOT))
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from diagnose_tamp_instrument import (  # noqa: E402
    das_layer_vector,
    iter_batches,
    load_rows,
    run_d1_amia_selection,
    spearman,
)
from tamp_calib_study import iter_text_batches, load_text_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify TAMP's single-modality (text-only) reduction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument("--text_calib", required=True, help="build_text_calib.py output (JSON list of strings).")
    p.add_argument("--mm_calib", default=None, help="Optional multimodal calib JSON for T8.")
    p.add_argument("--mm_images", default=None, help="Image dir for --mm_calib.")
    p.add_argument("--model_name", default="blip2_t5")
    p.add_argument("--model_type", default="pretrain_flant5xl")
    p.add_argument("--device", default=None)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--max_samples", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_txt_len", type=int, default=None)
    p.add_argument("--sparsity", type=float, default=0.5)
    p.add_argument("--t5_c4_encoder_only", action="store_true")
    p.add_argument("--probe_blocks", default="0,11,23")
    p.add_argument("--probe_linears", default="SelfAttention.v,DenseReluDense.wo")
    p.add_argument("--min_select_ratio", type=float, default=0.02)
    p.add_argument("--max_select_ratio", type=float, default=0.95)
    p.add_argument("--min_sparsity_range", type=float, default=0.02,
                   help="DAS dynamic range below this counts as effectively uniform.")
    p.add_argument("--out_json", default="verify_tamp_text_only.json")
    return p.parse_args()


class Report:
    def __init__(self) -> None:
        self.checks: List[Dict[str, Any]] = []

    def add(self, cid, name, status, detail=None, note="") -> None:
        assert status in ("PASS", "WARN", "FAIL", "SKIP", "INFO")
        self.checks.append({"id": cid, "name": name, "status": status,
                            "detail": detail, "note": note})
        print(f"[{status}] {cid} {name}")
        if detail is not None:
            print(f"       {detail}")
        if note:
            print(f"       -> {note}")

    @property
    def failed(self):
        return [c for c in self.checks if c["status"] == "FAIL"]

    @property
    def warned(self):
        return [c for c in self.checks if c["status"] == "WARN"]


def build_pruner(mods, model, batches, n_rows, args):
    return mods["BLIPT5LayerWandaPruner"](
        model=model, data_loader=batches,
        t5_prune_spec="24-%.6f-1.0-1.0" % (1.0 - args.sparsity),
        vit_prune_spec=None, t5_pruning_method="none", vit_pruning_method="none",
        num_samples=n_rows, num_data_first_stage=n_rows,
        sparsity_ratio_granularity="layer",
        max_sparsity_per_layer=min(1.0, args.sparsity + 0.1),
        score_method="density_sum", token_selection="amia",
        prune_t5=True, prune_vit=False, importance_scope="llm_only",
    )


def calibrate(mods, pruner, model, batches, n_rows):
    torch = mods["torch"]
    with torch.no_grad():
        return mods["T5LayerWandaPruner"].prepare_calibration_input_encoder(
            pruner, model, batches, mods["device"], "t5_model", n_rows,
            module_to_process="t5_model.encoder.block", return_image_masks=True,
        )


def main() -> int:
    args = parse_args()
    rep = Report()
    print("=" * 72)
    print("TAMP single-modality (--tamp_text_only) verification")
    print("=" * 72)

    for label, path, isdir in (("--ckpt", args.ckpt, False), ("--text_calib", args.text_calib, False)):
        ok = os.path.isdir(path) if isdir else os.path.isfile(path)
        if not ok:
            rep.add("T1", "view contract", "FAIL", detail=f"missing {label} {path}")
            return finish(args, rep)

    try:
        import torch
        from PIL import Image
        from lavis.models import load_model
        from lavis.processors import load_processor
        from lavis.compression.pruners import wanda_pruner as pruner_mod
        from lavis.compression.pruners.wanda_pruner import (
            BLIPT5LayerWandaPruner, T5LayerWandaPruner, _cos_pairwise_density_single,
        )
        from lavis.compression.pruners.layer_single_base_pruner import cos_pairwise_density
        from lavis.compression.unimodal_prune import wrap_model_for_unimodal_prune
    except ModuleNotFoundError as exc:
        rep.add("T1", "view contract", "FAIL", detail=f"missing runtime module: {exc.name}",
                note="Run inside the LAVIS/ecoflap conda env.")
        return finish(args, rep)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[load] {args.model_name}/{args.model_type} device={device}")
    base_model = load_model(args.model_name, args.model_type, is_eval=True,
                            device=device, checkpoint=args.ckpt)
    base_model.eval()
    if args.max_txt_len is not None:
        base_model.max_txt_len = int(args.max_txt_len)

    mods = {"torch": torch, "Image": Image, "pruner_mod": pruner_mod,
            "BLIPT5LayerWandaPruner": BLIPT5LayerWandaPruner,
            "T5LayerWandaPruner": T5LayerWandaPruner, "device": device}

    view, _ = wrap_model_for_unimodal_prune(
        base_model, "t5_c4_text", t5_c4_encoder_only=args.t5_c4_encoder_only)
    view.eval()
    view_name = type(view).__name__
    print(f"[text-only] view = {view_name}")
    if args.t5_c4_encoder_only:
        rep.add("T0", "fp16 default on encoder-only view", "WARN",
                detail=f"{view_name}.maybe_autocast defaults to float16",
                note=("This repo has a recorded fp16 overflow -> NaN issue on text tokens. "
                      "All pruning call sites pass bfloat16 explicitly, so it should not "
                      "trigger, but prefer the seq2seq view unless you need encoder-only."))

    # ---------------- T1 view contract ----------------
    texts = load_text_rows(args.text_calib, args.max_samples)
    if len(texts) % args.batch_size != 0:
        rep.add("T1b", "sample divisibility", "WARN",
                detail=f"{len(texts)} texts % batch_size {args.batch_size} != 0",
                note="ActivationDensity averages per batch; a short trailing batch is over-weighted.")
    batches = list(iter_text_batches(texts, args.batch_size))
    pruner = build_pruner(mods, view, batches, len(texts), args)

    for attr in ("t5_model", "maybe_autocast"):
        if not hasattr(view, attr):
            rep.add("T1", "view contract", "FAIL", detail=f"view lacks .{attr}")
            return finish(args, rep)

    calib = calibrate(mods, pruner, view, batches, len(texts))
    if len(calib) < 5:
        rep.add("T1", "view contract", "FAIL",
                detail=f"calibration returned {len(calib)} elements, expected 5",
                note="AMIA/DAS need temp_label and temp_encoder_atts on the text view.")
        return finish(args, rep)
    inps, _outs, _caches, image_masks, attn_masks = calib[:5]

    problems, min_valid = [], None
    for j, (inp, img, att) in enumerate(zip(inps, image_masks, attn_masks)):
        img_b, att_b = img.bool(), att.bool()
        if img_b.dim() == 1:
            img_b, att_b = img_b.unsqueeze(0), att_b.unsqueeze(0)
        B, S = img_b.shape
        if tuple(inp.shape[:2]) != (B, S):
            problems.append(f"batch {j}: mask {(B,S)} vs input {tuple(inp.shape[:2])}")
        if int(img_b.sum().item()) != 0:
            problems.append(f"batch {j}: temp_label has True entries; text-only must be all-False")
        for b in range(B):
            v = int(att_b[b].sum().item())
            min_valid = v if min_valid is None else min(min_valid, v)
    if problems:
        rep.add("T1", "view contract", "FAIL", detail=problems[:5])
        return finish(args, rep)
    rep.add("T1", "view contract", "PASS",
            detail={"view": view_name, "cached_batches": len(inps),
                    "seq_len": int(inps[0].shape[1]),
                    "min_valid_tokens_per_sample": min_valid})

    # ---------------- T4 short-text safety ----------------
    if min_valid is not None and min_valid < 2:
        rep.add("T4", "short-text safety", "FAIL",
                detail=f"a sample has only {min_valid} valid token(s)",
                note=("The single-modality reduction raises when no modality pair can be "
                      "measured. Drop or merge sub-2-token calibration lines."))
    else:
        rep.add("T4", "short-text safety", "PASS",
                detail=f"min valid tokens per sample = {min_valid} (>= 2)")

    # ---------------- T2 single-modality branch ----------------
    with torch.no_grad():
        out0 = inps[0].float()
    v, l, vl, cv, cl, cvl = cos_pairwise_density(
        out0, image_masks[0], attention_mask=attn_masks[0], return_counts=True)
    defined = tuple(n for n, c in (("v", cv), ("l", cl), ("vl", cvl)) if c)
    dens_new = _cos_pairwise_density_single(
        out0[0], image_masks[0][0], attention_mask=attn_masks[0][0])
    branch_ok = defined == ("l",) and abs(dens_new - l) < 5e-3
    detail = {"defined_terms": defined, "s_l": round(l, 6),
              "amia_density_new": round(float(dens_new), 6),
              "amia_density_if_unfixed": round(l / 3.0, 6)}
    if branch_ok:
        rep.add("T2", "single-modality branch", "PASS", detail=detail,
                note="DAS sees defined==('l',) and AMIA density equals s_l, not s_l/3.")
    else:
        rep.add("T2", "single-modality branch", "FAIL", detail=detail,
                note="The five-line fix is not in effect on this code path.")

    # ---------------- T3 old-vs-new delta ----------------
    imp_new = (1.0 - l) * 3.0
    imp_old = (1.0 - 0.0) + (1.0 - l) + (1.0 - 0.0)
    thr_new = 0.1 * math.sqrt(max(0.0, 1.0 - l))
    thr_old = 0.1 * math.sqrt(max(0.0, 1.0 - l / 3.0))
    rep.add("T3", "old-vs-new correction size", "INFO",
            detail={"das_importance_old": round(imp_old, 5),
                    "das_importance_new": round(imp_new, 5),
                    "das_constant_floor_removed": round(2.0, 5),
                    "amia_threshold_old": round(thr_old, 6),
                    "amia_threshold_new": round(thr_new, 6),
                    "amia_threshold_inflation_removed": round(thr_old / max(thr_new, 1e-12), 4)})

    # ---------------- T6 determinism + DAS vector ----------------
    keys_a, vec_a = das_layer_vector(pruner, calib, args.sparsity)
    keys_b, vec_b = das_layer_vector(pruner, calib, args.sparsity)
    if keys_a != keys_b:
        rep.add("T6", "determinism", "FAIL", detail="layer key sets differ between repeats")
    else:
        md = max((abs(x - y) for x, y in zip(vec_a, vec_b)), default=0.0)
        if md == 0.0:
            rep.add("T6", "determinism", "PASS",
                    detail=f"{len(keys_a)} layers bit-identical across 2 repeats")
        else:
            rep.add("T6", "determinism", "FAIL", detail=f"max |delta| = {md:.3e}",
                    note="Run-to-run noise would contaminate any cross-calibration comparison.")

    sp_range = (max(vec_a) - min(vec_a)) if vec_a else 0.0

    # ---------------- T5 non-degeneracy (GO / NO-GO) ----------------
    recs = run_d1_amia_selection(torch, view, pruner_mod, calib, "textonly",
                                 args.probe_blocks, args.probe_linears)
    ratios = [r["select_ratio"] for r in recs] or [1.0]
    sel = [r["n_selected"] for r in recs] or [0]
    mean_ratio = statistics.mean(ratios)
    detail = {"amia_observations": len(recs),
              "select_ratio_mean": round(mean_ratio, 5),
              "select_ratio_min": round(min(ratios), 5),
              "select_ratio_max": round(max(ratios), 5),
              "n_selected_mean": round(statistics.mean(sel), 3),
              "valid_tokens_mean": round(statistics.mean([r["n_valid"] for r in recs] or [0]), 2),
              "das_layer_sparsity_range": round(sp_range, 6)}
    amia_dead = mean_ratio > args.max_select_ratio
    amia_collapsed = mean_ratio < args.min_select_ratio
    das_dead = sp_range < args.min_sparsity_range
    if amia_dead and das_dead:
        rep.add("T5", "non-degeneracy (GO/NO-GO)", "FAIL", detail=detail,
                note=("NO-GO: AMIA keeps ~everything AND DAS is effectively uniform. "
                      "This path is vanilla Wanda with extra steps -- same as the old "
                      "degraded route. Do not spend runs on it."))
    elif amia_collapsed:
        rep.add("T5", "non-degeneracy (GO/NO-GO)", "FAIL", detail=detail,
                note=("AMIA collapsed: the input activation rests on almost no tokens, so "
                      "scaler_row is not a meaningful estimate. Re-calibrate the MMD "
                      "threshold before using this path."))
    elif amia_dead or das_dead:
        rep.add("T5", "non-degeneracy (GO/NO-GO)", "WARN", detail=detail,
                note=("Only one component is alive ("
                      + ("DAS" if amia_dead else "AMIA")
                      + "). Report the path as that single mechanism, not as TAMP."))
    else:
        rep.add("T5", "non-degeneracy (GO/NO-GO)", "PASS", detail=detail,
                note="GO: both AMIA and DAS do non-trivial work under text-only calibration.")

    # ---------------- T7 vs degraded Wanda ----------------
    uniform = [args.sparsity] * len(vec_a)
    diffs = [abs(x - y) for x, y in zip(vec_a, uniform)]
    d7 = {"mean_abs_diff_vs_uniform": round(statistics.mean(diffs), 6),
          "max_abs_diff_vs_uniform": round(max(diffs), 6),
          "das_layer_sparsity_range": round(sp_range, 6)}
    if statistics.mean(diffs) < 1e-4:
        rep.add("T7", "vs degraded Wanda", "WARN", detail=d7,
                note="DAS output is indistinguishable from uniform sparsity: identical to the old path.")
    else:
        rep.add("T7", "vs degraded Wanda", "PASS", detail=d7,
                note="DAS allocation differs from uniform, so this is not the old degraded route.")

    del calib, batches, pruner
    torch.cuda.empty_cache()

    # ---------------- T8 multimodal untouched ----------------
    if args.mm_calib and args.mm_images and os.path.isfile(args.mm_calib) and os.path.isdir(args.mm_images):
        vis = load_processor("blip_image_eval").build(image_size=args.image_size)
        mm_rows = load_rows(args.mm_calib, args.max_samples)
        mm_batches = list(iter_batches(mm_rows, args.mm_images, vis, torch, Image, device, args.batch_size))
        mm_pruner = build_pruner(mods, base_model, mm_batches, len(mm_rows), args)
        mm_calib_res = calibrate(mods, mm_pruner, base_model, mm_batches, len(mm_rows))
        mv, ml, mvl, mcv, mcl, mcvl = cos_pairwise_density(
            mm_calib_res[0][0].float(), mm_calib_res[3][0],
            attention_mask=mm_calib_res[4][0], return_counts=True)
        B = int(mm_calib_res[3][0].shape[0])
        all_defined = (mcv == B and mcl == B and mcvl == B)
        d8 = {"batch_size": B, "defined_counts": {"v": mcv, "l": mcl, "vl": mcvl}}
        if all_defined:
            rep.add("T8", "multimodal untouched", "PASS", detail=d8,
                    note=("Every term is defined for every sample, so compute_density takes the "
                          "verbatim three-term branch and the multimodal path is byte-identical."))
        else:
            rep.add("T8", "multimodal untouched", "WARN", detail=d8,
                    note=("Some multimodal samples lack a measurable term. The five-line change "
                          "WILL alter multimodal numbers there. Run the LAVIS_DAS_DIAGNOSTIC "
                          "audit over all sets before reusing existing multimodal results."))
        del mm_calib_res, mm_batches, mm_pruner
        torch.cuda.empty_cache()
    else:
        rep.add("T8", "multimodal untouched", "SKIP",
                detail="pass --mm_calib and --mm_images to check",
                note="This is the check that protects your existing multimodal results.")

    return finish(args, rep)


def finish(args, rep) -> int:
    print("\n" + "=" * 72)
    nf, nw = len(rep.failed), len(rep.warned)
    if nf:
        print(f"RESULT: NOT READY -- {nf} hard failure(s), {nw} warning(s)")
        for c in rep.failed:
            print(f"  FAIL {c['id']} {c['name']}: {c['note'] or c['detail']}")
    elif nw:
        print(f"RESULT: READY WITH CAVEATS -- {nw} warning(s)")
        for c in rep.warned:
            print(f"  WARN {c['id']} {c['name']}: {c['note'] or ''}")
    else:
        print("RESULT: READY -- all checks passed")
    print("=" * 72)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump({"ready": nf == 0, "n_fail": nf, "n_warn": nw,
                   "args": vars(args), "checks": rep.checks},
                  fh, indent=2, ensure_ascii=False, default=str)
    print(f"[done] wrote {args.out_json}")
    return 1 if nf else 0


if __name__ == "__main__":
    raise SystemExit(main())
