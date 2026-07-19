#!/usr/bin/env python3
"""One-shot verification of TAMP's single-modality (text-only) reduction.

Run this and it answers, in order and with a stop-on-failure gate:

  PHASE 1  implementation      does the path run, is the single-modality branch the
                               one being taken, is it deterministic, and are existing
                               multimodal results still safe?
  PHASE 2  non-degeneracy      GO/NO-GO: do AMIA and DAS each do non-trivial work, or
                               is this vanilla Wanda with extra steps?
  PHASE 3  component ablation  four configurations sharing one calibration pass:
                                 A  naive  + uniform   (the old degraded-Wanda route)
                                 B  AMIA   + uniform
                                 C  naive  + DAS
                                 D  AMIA   + DAS       (--tamp_text_only)
                               computed in-process, no checkpoints written
  PHASE 4  divergence          pairwise pruning-mask overlap between A/B/C/D. If every
                               pair overlaps ~100%, the components change nothing and
                               the direction is dead regardless of benchmark scores.

The one thing this script cannot do is run the four benchmarks; that needs hours and
the eval harness. Everything short of that is covered here.

Note on PHASE 3/4: masks are computed on the dense model for a few probe layers,
whereas real pruning is progressive (layer i is pruned before layer i+1 is calibrated).
That is a deviation in absolute terms, but it is identical across A/B/C/D, so the
*comparison* between configurations is valid -- which is what these phases test.

Example:

  CUDA_VISIBLE_DEVICES=0 python scripts/blip2/verify_text_only_tamp_full.py \
    --ckpt /data/data2/mfs/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth \
    --text_calib /data/data2/mfs/text_calib_128/c4_text_calib_128.json \
    --mm_calib /data/data2/mfs/MMBench_calibration/mmbench_calibration_train.json \
    --mm_images /data/data2/mfs/MMBench_calibration/images \
    --max_samples 32 --batch_size 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
)
from tamp_calib_study import iter_text_batches, load_text_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One-shot verification of the text-only TAMP reduction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument("--text_calib", required=True, help="build_text_calib.py output (JSON list of strings).")
    p.add_argument("--mm_calib", default=None, help="Multimodal calib JSON (protects existing results).")
    p.add_argument("--mm_images", default=None)
    p.add_argument("--model_name", default="blip2_t5")
    p.add_argument("--model_type", default="pretrain_flant5xl")
    p.add_argument("--device", default=None)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--max_samples", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_txt_len", type=int, default=None)
    p.add_argument("--sparsity", type=float, default=0.5)
    p.add_argument("--t5_c4_encoder_only", action="store_true")
    p.add_argument("--probe_blocks", default="0,11,23", help="Blocks used for AMIA probing and ablation.")
    p.add_argument("--probe_linears", default="SelfAttention.v,DenseReluDense.wo")
    p.add_argument("--min_select_ratio", type=float, default=0.02)
    p.add_argument("--max_select_ratio", type=float, default=0.95)
    p.add_argument("--min_sparsity_range", type=float, default=0.02)
    p.add_argument("--min_mask_divergence", type=float, default=0.005,
                   help="Fraction of differing mask entries below which two configs count as identical.")
    p.add_argument("--out_json", default="verify_text_only_tamp_full.json")
    return p.parse_args()


class Report:
    def __init__(self) -> None:
        self.checks: List[Dict[str, Any]] = []
        self.phase = ""

    def start(self, name: str) -> None:
        self.phase = name
        print("\n" + "-" * 72)
        print(f"PHASE {name}")
        print("-" * 72)

    def add(self, cid, name, status, detail=None, note="") -> None:
        assert status in ("PASS", "WARN", "FAIL", "SKIP", "INFO")
        self.checks.append({"phase": self.phase, "id": cid, "name": name,
                            "status": status, "detail": detail, "note": note})
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


# ------------------------------------------------------------------ helpers


def build_pruner(mods, model, batches, n_rows, args, token_selection="amia", granularity="layer"):
    return mods["BLIPT5LayerWandaPruner"](
        model=model, data_loader=batches,
        t5_prune_spec="24-%.6f-1.0-1.0" % (1.0 - args.sparsity),
        vit_prune_spec=None, t5_pruning_method="none", vit_pruning_method="none",
        num_samples=n_rows, num_data_first_stage=n_rows,
        sparsity_ratio_granularity=granularity,
        max_sparsity_per_layer=min(1.0, args.sparsity + 0.1),
        score_method="density_sum", token_selection=token_selection,
        prune_t5=True, prune_vit=False, importance_scope="llm_only",
    )


def calibrate(mods, pruner, model, batches, n_rows):
    with mods["torch"].no_grad():
        return mods["T5LayerWandaPruner"].prepare_calibration_input_encoder(
            pruner, model, batches, mods["device"], "t5_model", n_rows,
            module_to_process="t5_model.encoder.block", return_image_masks=True,
        )


def parse_probe_blocks(spec: str, n_blocks: int) -> List[int]:
    if spec.strip().lower() == "all":
        return list(range(n_blocks))
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def compute_scaler_rows(mods, model, pruner_mod, calib, args, token_selection) -> Dict[str, Any]:
    """scaler_row per probe Linear, on the dense model, using production wrappers."""
    torch = mods["torch"]
    inps, _outs, caches, image_masks, attn_masks = calib[:5]
    blocks = pruner_mod.get_module_recursive(model, "t5_model.encoder.block")
    probe = parse_probe_blocks(args.probe_blocks, len(blocks))
    suffixes = [s.strip() for s in args.probe_linears.split(",") if s.strip()]

    layer_caches = [dict(c) for c in caches]
    hidden = list(inps)
    out: Dict[str, Any] = {}

    for i in range(len(blocks)):
        layer = blocks[i]
        if i in probe:
            subset = {k: v for k, v in pruner_mod.find_layers(layer).items()
                      if any(k.endswith(s) for s in suffixes)}
            wrapped = {}
            for name, mod_ in subset.items():
                wrapped[name] = (pruner_mod.AdaptiveMultimodalInputActivation(mod_)
                                 if token_selection == "amia" else pruner_mod.WrappedGPT(mod_))
            for j in range(len(hidden)):
                score_j = None
                if token_selection == "amia":
                    with torch.no_grad():
                        with model.maybe_autocast(dtype=torch.bfloat16):
                            _, _, aw = pruner_mod._normal_t5_block_forward(
                                layer, hidden[j], layer_caches[j], output_attentions=True)
                    score_j = pruner_mod._encoder_attention_column_scores(aw, attn_masks[j])
                    if score_j is None:
                        raise RuntimeError(f"block {i}: could not derive attention scores")

                def mk(w, mj, sj, aj):
                    def hook(_m, inp, o):
                        ot = o[0] if isinstance(o, (tuple, list)) else o
                        if token_selection == "amia":
                            w.add_batch(inp[0].data, ot.data, mj, sj, attention_mask=aj)
                        else:
                            w.add_batch(inp[0].data, ot.data, mj, sj)
                    return hook

                handles = [subset[n].register_forward_hook(
                    mk(wrapped[n], image_masks[j], score_j, attn_masks[j])) for n in wrapped]
                try:
                    with torch.no_grad():
                        with model.maybe_autocast(dtype=torch.bfloat16):
                            pruner_mod._normal_t5_block_forward(layer, hidden[j], layer_caches[j])
                finally:
                    for h in handles:
                        h.remove()
            for name, w in wrapped.items():
                out[f"t5_model.encoder.block.{i}.{name}.weight"] = (
                    subset[name].weight.data, w.scaler_row.clone())
        new_hidden = []
        for j in range(len(hidden)):
            with torch.no_grad():
                with model.maybe_autocast(dtype=torch.bfloat16):
                    o, _, _ = pruner_mod._normal_t5_block_forward(layer, hidden[j], layer_caches[j])
            new_hidden.append(o.detach())
        hidden = new_hidden
    return out


def masks_for(mods, scaler_rows: Dict[str, Any], sparsity_lookup) -> Dict[str, Any]:
    """Reproduce _prune's unstructured mask: per output row, drop the lowest-metric inputs."""
    torch = mods["torch"]
    out = {}
    for key, (W, scaler) in scaler_rows.items():
        metric = torch.abs(W) * torch.sqrt(scaler.reshape((1, -1)))
        s = float(sparsity_lookup(key))
        k = int(metric.shape[1] * s)
        mask = torch.zeros_like(metric, dtype=torch.bool)
        if k > 0:
            idx = torch.sort(metric, dim=-1, stable=True)[1][:, :k]
            mask.scatter_(1, idx, True)
        out[key] = mask.cpu()
    return out


def mask_divergence(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    diff = total = 0
    for k in a:
        if k not in b:
            continue
        d = (a[k] != b[k]).sum().item()
        diff += int(d)
        total += int(a[k].numel())
    return {"differing_frac": round(diff / max(1, total), 6),
            "differing_entries": diff, "total_entries": total}


# ------------------------------------------------------------------ main


def main() -> int:
    args = parse_args()
    rep = Report()
    print("=" * 72)
    print("TAMP text-only reduction -- full verification")
    print("=" * 72)

    for lbl, p in (("--ckpt", args.ckpt), ("--text_calib", args.text_calib)):
        if not os.path.isfile(p):
            rep.start("1 implementation")
            rep.add("P1.0", "inputs", "FAIL", detail=f"missing {lbl} {p}")
            return finish(args, rep)

    try:
        import torch
        from PIL import Image
        from lavis.models import load_model
        from lavis.processors import load_processor
        from lavis.compression.pruners import wanda_pruner as pruner_mod
        from lavis.compression.pruners.wanda_pruner import (
            BLIPT5LayerWandaPruner, T5LayerWandaPruner, _cos_pairwise_density_single)
        from lavis.compression.pruners.layer_single_base_pruner import cos_pairwise_density
        from lavis.compression.unimodal_prune import wrap_model_for_unimodal_prune
    except ModuleNotFoundError as exc:
        rep.start("1 implementation")
        rep.add("P1.0", "runtime", "FAIL", detail=f"missing module: {exc.name}",
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

    # ============================ PHASE 1 ============================
    rep.start("1 implementation")
    if args.t5_c4_encoder_only:
        rep.add("P1.0", "fp16 default on encoder-only view", "WARN",
                detail=f"{type(view).__name__}.maybe_autocast defaults to float16",
                note=("This repo has a recorded fp16 -> NaN issue on text tokens. Call sites pass "
                      "bfloat16 explicitly, but prefer the seq2seq view unless you need encoder-only."))

    texts = load_text_rows(args.text_calib, args.max_samples)
    if len(texts) % args.batch_size != 0:
        rep.add("P1.1", "sample divisibility", "WARN",
                detail=f"{len(texts)} % {args.batch_size} != 0",
                note="DAS averages per batch; a short trailing batch is over-weighted.")
    batches = list(iter_text_batches(texts, args.batch_size))
    pruner = build_pruner(mods, view, batches, len(texts), args)
    calib = calibrate(mods, pruner, view, batches, len(texts))

    if len(calib) < 5:
        rep.add("P1.2", "view contract", "FAIL",
                detail=f"calibration returned {len(calib)} elements, expected 5",
                note="AMIA/DAS need temp_label and temp_encoder_atts on the text view.")
        return finish(args, rep)
    inps, _o, _c, image_masks, attn_masks = calib[:5]

    problems, min_valid = [], None
    for j, (inp, img, att) in enumerate(zip(inps, image_masks, attn_masks)):
        ib, ab = img.bool(), att.bool()
        if ib.dim() == 1:
            ib, ab = ib.unsqueeze(0), ab.unsqueeze(0)
        B, S = ib.shape
        if tuple(inp.shape[:2]) != (B, S):
            problems.append(f"batch {j}: mask {(B,S)} vs input {tuple(inp.shape[:2])}")
        if int(ib.sum().item()) != 0:
            problems.append(f"batch {j}: temp_label has True entries; text-only must be all-False")
        for b in range(B):
            v = int(ab[b].sum().item())
            min_valid = v if min_valid is None else min(min_valid, v)
    if problems:
        rep.add("P1.2", "view contract", "FAIL", detail=problems[:5])
        return finish(args, rep)
    rep.add("P1.2", "view contract", "PASS",
            detail={"view": type(view).__name__, "cached_batches": len(inps),
                    "seq_len": int(inps[0].shape[1]), "min_valid_tokens": min_valid})

    if min_valid is not None and min_valid < 2:
        rep.add("P1.3", "short-text safety", "FAIL",
                detail=f"a sample has only {min_valid} valid token(s)",
                note="The reduction raises when no modality pair can be measured. Drop such lines.")
        return finish(args, rep)
    rep.add("P1.3", "short-text safety", "PASS", detail=f"min valid tokens = {min_valid}")

    v, l, vl, cv, cl, cvl = cos_pairwise_density(
        inps[0].float(), image_masks[0], attention_mask=attn_masks[0], return_counts=True)
    defined = tuple(n for n, c in (("v", cv), ("l", cl), ("vl", cvl)) if c)
    dens_new = _cos_pairwise_density_single(inps[0][0].float(), image_masks[0][0],
                                            attention_mask=attn_masks[0][0])
    d = {"defined_terms": defined, "s_l": round(l, 6),
         "amia_density_new": round(float(dens_new), 6),
         "amia_density_if_unfixed": round(l / 3.0, 6),
         "das_importance_new": round((1.0 - l) * 3.0, 5),
         "das_importance_if_unfixed": round(3.0 - l, 5)}
    if defined == ("l",) and abs(float(dens_new) - l) < 5e-3:
        rep.add("P1.4", "single-modality branch in effect", "PASS", detail=d,
                note=("DAS sees defined==('l',); AMIA density is s_l, not s_l/3. The old code's "
                      "constant +2 floor on DAS importance is gone."))
    else:
        rep.add("P1.4", "single-modality branch in effect", "FAIL", detail=d,
                note="The five-line fix is not active on this path.")
        return finish(args, rep)

    keys_a, vec_a = das_layer_vector(pruner, calib, args.sparsity)
    keys_b, vec_b = das_layer_vector(pruner, calib, args.sparsity)
    md = max((abs(x - y) for x, y in zip(vec_a, vec_b)), default=0.0) if keys_a == keys_b else None
    if md == 0.0:
        rep.add("P1.5", "determinism", "PASS", detail=f"{len(keys_a)} layers bit-identical")
    else:
        rep.add("P1.5", "determinism", "FAIL", detail=f"max |delta| = {md}",
                note="Run-to-run noise would contaminate every downstream comparison.")
        return finish(args, rep)

    # multimodal safety
    if args.mm_calib and args.mm_images and os.path.isfile(args.mm_calib) and os.path.isdir(args.mm_images):
        vis = load_processor("blip_image_eval").build(image_size=args.image_size)
        mm_rows = load_rows(args.mm_calib, args.max_samples)
        mm_batches = list(iter_batches(mm_rows, args.mm_images, vis, torch, Image, device, args.batch_size))
        mm_pruner = build_pruner(mods, base_model, mm_batches, len(mm_rows), args)
        mm_calib = calibrate(mods, mm_pruner, base_model, mm_batches, len(mm_rows))
        _mv, _ml, _mvl, mcv, mcl, mcvl = cos_pairwise_density(
            mm_calib[0][0].float(), mm_calib[3][0], attention_mask=mm_calib[4][0], return_counts=True)
        B = int(mm_calib[3][0].shape[0])
        dd = {"batch_size": B, "defined_counts": {"v": mcv, "l": mcl, "vl": mcvl}}
        if mcv == B and mcl == B and mcvl == B:
            rep.add("P1.6", "existing multimodal results safe", "PASS", detail=dd,
                    note="All terms defined for all samples -> compute_density takes the verbatim "
                         "three-term branch -> multimodal output is byte-identical.")
        else:
            rep.add("P1.6", "existing multimodal results safe", "WARN", detail=dd,
                    note="Some multimodal samples lack a measurable term; the change WILL alter "
                         "multimodal numbers there. Run the LAVIS_DAS_DIAGNOSTIC audit first.")
        del mm_calib, mm_batches, mm_pruner
        torch.cuda.empty_cache()
    else:
        rep.add("P1.6", "existing multimodal results safe", "SKIP",
                detail="pass --mm_calib and --mm_images",
                note="This is the check that protects your existing multimodal results.")

    # ============================ PHASE 2 ============================
    rep.start("2 non-degeneracy (GO/NO-GO)")
    recs = run_d1_amia_selection(torch, view, pruner_mod, calib, "textonly",
                                 args.probe_blocks, args.probe_linears)
    ratios = [r["select_ratio"] for r in recs] or [1.0]
    mean_ratio = statistics.mean(ratios)
    sp_range = (max(vec_a) - min(vec_a)) if vec_a else 0.0
    det = {"amia_observations": len(recs),
           "select_ratio_mean": round(mean_ratio, 5),
           "select_ratio_min": round(min(ratios), 5), "select_ratio_max": round(max(ratios), 5),
           "n_selected_mean": round(statistics.mean([r["n_selected"] for r in recs] or [0]), 3),
           "valid_tokens_mean": round(statistics.mean([r["n_valid"] for r in recs] or [0]), 2),
           "das_layer_sparsity_range": round(sp_range, 6)}
    amia_dead, amia_collapsed = mean_ratio > args.max_select_ratio, mean_ratio < args.min_select_ratio
    das_dead = sp_range < args.min_sparsity_range
    if amia_dead and das_dead:
        rep.add("P2.1", "GO/NO-GO", "FAIL", detail=det,
                note="NO-GO: AMIA keeps ~everything AND DAS is uniform. Identical to the old "
                     "degraded route. Stop here.")
        return finish(args, rep)
    if amia_collapsed:
        rep.add("P2.1", "GO/NO-GO", "FAIL", detail=det,
                note="AMIA collapsed: scaler_row rests on almost no tokens. Re-calibrate the "
                     "MMD threshold before using this path.")
        return finish(args, rep)
    if amia_dead or das_dead:
        rep.add("P2.1", "GO/NO-GO", "WARN", detail=det,
                note=f"Only {'DAS' if amia_dead else 'AMIA'} is alive. Report the path as that "
                     "single mechanism, not as TAMP.")
    else:
        rep.add("P2.1", "GO/NO-GO", "PASS", detail=det,
                note="Both AMIA and DAS do non-trivial work under text-only calibration.")

    # ============================ PHASE 3 ============================
    rep.start("3 component ablation (in-process, no checkpoints)")
    uniform = lambda _k: args.sparsity
    das_map = dict(zip(keys_a, vec_a))
    das_lookup = lambda k: das_map.get(k, args.sparsity)

    print("  computing scaler_row for naive ...")
    sr_naive = compute_scaler_rows(mods, view, pruner_mod, calib, args, "naive")
    print("  computing scaler_row for amia ...")
    sr_amia = compute_scaler_rows(mods, view, pruner_mod, calib, args, "amia")

    configs = {
        "A_naive_uniform": masks_for(mods, sr_naive, uniform),
        "B_amia_uniform": masks_for(mods, sr_amia, uniform),
        "C_naive_das": masks_for(mods, sr_naive, das_lookup),
        "D_amia_das": masks_for(mods, sr_amia, das_lookup),
    }
    rep.add("P3.1", "four configurations built", "PASS",
            detail={"probe_layers": len(sr_naive), "configs": list(configs)},
            note="A is the old degraded-Wanda route; D is --tamp_text_only.")

    # ============================ PHASE 4 ============================
    rep.start("4 divergence between configurations")
    names = list(configs)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dv = mask_divergence(configs[names[i]], configs[names[j]])
            pairs.append({"pair": f"{names[i]} vs {names[j]}", **dv})
    for p in pairs:
        print(f"       {p['pair']:34} differing = {p['differing_frac']:.4%}")

    amia_effect = next(p for p in pairs if p["pair"] == "A_naive_uniform vs B_amia_uniform")
    das_effect = next(p for p in pairs if p["pair"] == "A_naive_uniform vs C_naive_das")
    total_effect = next(p for p in pairs if p["pair"] == "A_naive_uniform vs D_amia_das")
    det = {"amia_only_vs_baseline": amia_effect["differing_frac"],
           "das_only_vs_baseline": das_effect["differing_frac"],
           "both_vs_baseline": total_effect["differing_frac"],
           "threshold": args.min_mask_divergence, "all_pairs": pairs}
    if total_effect["differing_frac"] < args.min_mask_divergence:
        rep.add("P4.1", "does the method change anything", "FAIL", detail=det,
                note="D produces essentially the same pruning mask as A. The text-only reduction "
                     "is the old degraded route in disguise; benchmark scores cannot differ "
                     "meaningfully. Do not spend eval runs on it.")
    else:
        contributions = []
        if amia_effect["differing_frac"] >= args.min_mask_divergence:
            contributions.append("AMIA")
        if das_effect["differing_frac"] >= args.min_mask_divergence:
            contributions.append("DAS")
        rep.add("P4.1", "does the method change anything", "PASS", detail=det,
                note=f"Mask differs from the degraded baseline by "
                     f"{total_effect['differing_frac']:.2%}; contributing component(s): "
                     f"{', '.join(contributions) or 'interaction only'}. "
                     "Benchmark evaluation is now worth running.")

    del calib, batches, pruner, sr_naive, sr_amia, configs
    torch.cuda.empty_cache()
    return finish(args, rep)


def finish(args, rep) -> int:
    print("\n" + "=" * 72)
    nf, nw = len(rep.failed), len(rep.warned)
    if nf:
        print(f"VERDICT: NOT READY -- {nf} hard failure(s), {nw} warning(s)")
        for c in rep.failed:
            print(f"  FAIL [{c['phase']}] {c['id']} {c['name']}")
            print(f"       {c['note'] or c['detail']}")
    elif nw:
        print(f"VERDICT: READY WITH CAVEATS -- {nw} warning(s)")
        for c in rep.warned:
            print(f"  WARN [{c['phase']}] {c['id']} {c['name']}: {c['note'] or ''}")
    else:
        print("VERDICT: READY -- every phase passed")
    print("\nNot covered here: the four-benchmark evaluation. Run it only if PHASE 4 passed.")
    print("=" * 72)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump({"ready": nf == 0, "n_fail": nf, "n_warn": nw,
                   "args": vars(args), "checks": rep.checks},
                  fh, indent=2, ensure_ascii=False, default=str)
    print(f"[done] wrote {args.out_json}")
    return 1 if nf else 0


if __name__ == "__main__":
    raise SystemExit(main())
