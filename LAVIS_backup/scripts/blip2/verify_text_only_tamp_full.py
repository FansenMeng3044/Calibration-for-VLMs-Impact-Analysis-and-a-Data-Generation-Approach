#!/usr/bin/env python3
"""One-shot verification of TAMP's single-modality (text-only) reduction.

Runs every phase to completion. A failure in one phase is recorded and the rest
still run, so a single invocation gives the whole picture instead of stopping at
the first problem.

  P0  code state          the edits under test are actually present in the files
                          being imported (catches "you are running a stale copy")
  P1  structural          the text view exposes t5_model.* through named_parameters,
                          module paths resolve, and the pruner can find what it
                          needs -- this is the class of bug that hides behind a
                          confusing "no T5 encoder Linear parameters" error
  P2  calibration         view contract, all-False temp_label, encoder attention
                          masks, no sub-2-token samples
  P3  reduction active    DAS sees defined==('l',) and AMIA density is s_l, not
                          s_l/3 -- proves the five-line change is in effect
  P4  determinism         two DAS repeats are bit-identical
  P5  multimodal safety   with multimodal calibration every diversity term is still
                          defined, so the multimodal path keeps its verbatim branch
                          and existing multimodal results stay valid
  P6  GO / NO-GO          do AMIA and DAS each do non-trivial work, or is this
                          vanilla Wanda with extra steps?
  P7  ablation            A naive+uniform (baseline) / B AMIA+uniform /
                          C naive+DAS / D AMIA+DAS, compared by pruning-mask
                          divergence, in-process, no checkpoints written

Not covered: the four-benchmark evaluation. Run that only if P7 passes.

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
import os
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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
    p.add_argument("--text_calib", required=True)
    p.add_argument("--mm_calib", default=None)
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
    p.add_argument("--probe_blocks", default="0,11,23")
    p.add_argument("--probe_linears", default="SelfAttention.v,DenseReluDense.wo")
    p.add_argument("--min_select_ratio", type=float, default=0.02)
    p.add_argument("--max_select_ratio", type=float, default=0.95)
    p.add_argument("--min_sparsity_range", type=float, default=0.02)
    p.add_argument("--min_mask_divergence", type=float, default=0.005)
    p.add_argument("--out_json", default="verify_text_only_tamp_full.json")
    return p.parse_args()


# ------------------------------------------------------------------- report


class Report:
    def __init__(self) -> None:
        self.checks: List[Dict[str, Any]] = []
        self.phase = ""

    def start(self, name: str) -> None:
        self.phase = name
        print("\n" + "-" * 74)
        print(f"PHASE {name}")
        print("-" * 74)

    def add(self, cid, name, status, detail=None, note="") -> None:
        self.checks.append({"phase": self.phase, "id": cid, "name": name,
                            "status": status, "detail": detail, "note": note})
        print(f"[{status}] {cid} {name}")
        if detail is not None:
            print(f"       {detail}")
        if note:
            print(f"       -> {note}")

    def crash(self, cid, name, exc) -> None:
        tb = traceback.format_exc(limit=6)
        self.add(cid, name, "FAIL", detail=f"{type(exc).__name__}: {exc}",
                 note="phase aborted; later phases still attempted")
        print("       " + tb.replace("\n", "\n       "))
        self.checks[-1]["traceback"] = tb

    def status_of(self, cid) -> Optional[str]:
        for c in self.checks:
            if c["id"] == cid:
                return c["status"]
        return None

    @property
    def failed(self):
        return [c for c in self.checks if c["status"] == "FAIL"]

    @property
    def warned(self):
        return [c for c in self.checks if c["status"] == "WARN"]


def phase(rep: Report, cid: str, name: str, fn: Callable[[], None]) -> bool:
    """Run one phase; record any exception and keep going."""
    try:
        fn()
        return True
    except Exception as exc:  # noqa: BLE001 - deliberate: keep the sweep alive
        rep.crash(cid, name, exc)
        return False


# ------------------------------------------------------------------ helpers


def build_pruner(ctx, model, batches, n_rows, token_selection="amia", granularity="layer"):
    a = ctx["args"]
    return ctx["BLIPT5LayerWandaPruner"](
        model=model, data_loader=batches,
        t5_prune_spec="24-%.6f-1.0-1.0" % (1.0 - a.sparsity),
        vit_prune_spec=None, t5_pruning_method="none", vit_pruning_method="none",
        num_samples=n_rows, num_data_first_stage=n_rows,
        sparsity_ratio_granularity=granularity,
        max_sparsity_per_layer=min(1.0, a.sparsity + 0.1),
        score_method="density_sum", token_selection=token_selection,
        prune_t5=True, prune_vit=False, importance_scope="llm_only",
    )


def calibrate(ctx, pruner, model, batches, n_rows):
    with ctx["torch"].no_grad():
        return ctx["T5LayerWandaPruner"].prepare_calibration_input_encoder(
            pruner, model, batches, ctx["device"], "t5_model", n_rows,
            module_to_process="t5_model.encoder.block", return_image_masks=True,
        )


def parse_probe_blocks(spec: str, n: int) -> List[int]:
    return list(range(n)) if spec.strip().lower() == "all" else [
        int(x) for x in spec.split(",") if x.strip()]


def compute_scaler_rows(ctx, model, calib, token_selection) -> Dict[str, Any]:
    torch, pm, a = ctx["torch"], ctx["pruner_mod"], ctx["args"]
    inps, _o, caches, image_masks, attn_masks = calib[:5]
    blocks = pm.get_module_recursive(model, "t5_model.encoder.block")
    probe = parse_probe_blocks(a.probe_blocks, len(blocks))
    sufs = [s.strip() for s in a.probe_linears.split(",") if s.strip()]
    layer_caches = [dict(c) for c in caches]
    hidden = list(inps)
    out: Dict[str, Any] = {}

    for i in range(len(blocks)):
        layer = blocks[i]
        if i in probe:
            subset = {k: v for k, v in pm.find_layers(layer).items()
                      if any(k.endswith(s) for s in sufs)}
            wrapped = {n: (pm.AdaptiveMultimodalInputActivation(m)
                           if token_selection == "amia" else pm.WrappedGPT(m))
                       for n, m in subset.items()}
            for j in range(len(hidden)):
                score_j = None
                if token_selection == "amia":
                    with torch.no_grad():
                        with model.maybe_autocast(dtype=torch.bfloat16):
                            _, _, aw = pm._normal_t5_block_forward(
                                layer, hidden[j], layer_caches[j], output_attentions=True)
                    score_j = pm._encoder_attention_column_scores(aw, attn_masks[j])
                    if score_j is None:
                        raise RuntimeError(f"block {i}: attention scores unavailable")

                def mk(w, mj, sj, aj):
                    def hook(_m, inp, o):
                        ot = o[0] if isinstance(o, (tuple, list)) else o
                        if token_selection == "amia":
                            w.add_batch(inp[0].data, ot.data, mj, sj, attention_mask=aj)
                        else:
                            w.add_batch(inp[0].data, ot.data, mj, sj)
                    return hook

                hs = [subset[n].register_forward_hook(
                    mk(wrapped[n], image_masks[j], score_j, attn_masks[j])) for n in wrapped]
                try:
                    with torch.no_grad():
                        with model.maybe_autocast(dtype=torch.bfloat16):
                            pm._normal_t5_block_forward(layer, hidden[j], layer_caches[j])
                finally:
                    for h in hs:
                        h.remove()
            for n, w in wrapped.items():
                out[f"t5_model.encoder.block.{i}.{n}.weight"] = (
                    subset[n].weight.data, w.scaler_row.clone())
        nh = []
        for j in range(len(hidden)):
            with torch.no_grad():
                with model.maybe_autocast(dtype=torch.bfloat16):
                    o, _, _ = pm._normal_t5_block_forward(layer, hidden[j], layer_caches[j])
            nh.append(o.detach())
        hidden = nh
    return out


def masks_for(torch, scaler_rows, sparsity_lookup):
    out = {}
    for key, (W, scaler) in scaler_rows.items():
        metric = torch.abs(W) * torch.sqrt(scaler.reshape((1, -1)))
        k = int(metric.shape[1] * float(sparsity_lookup(key)))
        mask = torch.zeros_like(metric, dtype=torch.bool)
        if k > 0:
            mask.scatter_(1, torch.sort(metric, dim=-1, stable=True)[1][:, :k], True)
        out[key] = mask.cpu()
    return out


def mask_divergence(a, b):
    diff = total = 0
    for k in a:
        if k in b:
            diff += int((a[k] != b[k]).sum().item())
            total += int(a[k].numel())
    return {"differing_frac": round(diff / max(1, total), 6),
            "differing_entries": diff, "total_entries": total}


# -------------------------------------------------------------------- main


def main() -> int:
    args = parse_args()
    rep = Report()
    ctx: Dict[str, Any] = {"args": args}
    print("=" * 74)
    print("TAMP text-only reduction -- one-shot verification")
    print("=" * 74)

    # ---------------------------------------------------------- P0 code state
    rep.start("0 code state")

    def p0():
        root = _LAVIS_ROOT
        ev = (root / "evaluate_blip.py").read_text(encoding="utf-8")
        up = (root / "lavis" / "compression" / "unimodal_prune.py").read_text(encoding="utf-8")
        ls = (root / "lavis" / "compression" / "pruners" /
              "layer_single_base_pruner.py").read_text(encoding="utf-8")
        wp = (root / "lavis" / "compression" / "pruners" / "wanda_pruner.py").read_text(encoding="utf-8")
        facts = {
            "degradation_branch_removed": 'args.token_selection = "naive"' not in ev,
            "reduction_branch_present": "single-modality reduction" in ev,
            "vit_mode_guard_present": "does not support --prune_calib_mode" in ev,
            "view_registers_t5_first": up.index("self.t5_model = blip_model.t5_model")
                                       < up.index("self._blip = blip_model"),
            "das_reduction_present": "sum(terms) * (3.0 / len(terms))" in ls,
            "amia_reduction_present": "sum(terms) / len(terms)" in wp,
            "audit_counters_present": "LAVIS_DAS_DIAGNOSTIC" in ls,
        }
        missing = [k for k, v in facts.items() if not v]
        if missing:
            rep.add("P0.1", "edits present in imported files", "FAIL", detail=facts,
                    note=f"missing: {missing}. You are running a stale copy of the code.")
        else:
            rep.add("P0.1", "edits present in imported files", "PASS", detail=facts)

    phase(rep, "P0.1", "edits present in imported files", p0)

    # --------------------------------------------------------------- load
    rep.start("1 structural")

    def load():
        import torch
        from PIL import Image
        from lavis.models import load_model
        from lavis.processors import load_processor
        from lavis.compression.pruners import wanda_pruner as pruner_mod
        from lavis.compression.pruners.wanda_pruner import (
            BLIPT5LayerWandaPruner, T5LayerWandaPruner, _cos_pairwise_density_single)
        from lavis.compression.pruners.layer_single_base_pruner import cos_pairwise_density
        from lavis.compression.unimodal_prune import wrap_model_for_unimodal_prune

        ctx.update(torch=torch, Image=Image, load_processor=load_processor,
                   pruner_mod=pruner_mod, BLIPT5LayerWandaPruner=BLIPT5LayerWandaPruner,
                   T5LayerWandaPruner=T5LayerWandaPruner,
                   cos_pairwise_density=cos_pairwise_density,
                   _cos_pairwise_density_single=_cos_pairwise_density_single)
        ctx["device"] = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[load] {args.model_name}/{args.model_type} device={ctx['device']}")
        base = load_model(args.model_name, args.model_type, is_eval=True,
                          device=ctx["device"], checkpoint=args.ckpt)
        base.eval()
        if args.max_txt_len is not None:
            base.max_txt_len = int(args.max_txt_len)
        ctx["base_model"] = base
        view, _ = wrap_model_for_unimodal_prune(
            base, "t5_c4_text", t5_c4_encoder_only=args.t5_c4_encoder_only)
        view.eval()
        ctx["view"] = view

        names = [n for n, _ in view.named_parameters()]
        t5_named = [n for n in names if n.startswith("t5_model.encoder.block.")]
        blip_hidden = [n for n in names if n.startswith("_blip.t5_model.")]
        det = {"view": type(view).__name__, "total_params": len(names),
               "t5_encoder_block_params": len(t5_named),
               "params_hidden_behind_blip": len(blip_hidden)}
        if not t5_named:
            rep.add("P1.1", "named_parameters exposes t5_model.*", "FAIL", detail=det,
                    note=("PyTorch de-duplicates shared Parameters under the first name it "
                          "reaches. Register the pruned submodule BEFORE _blip in the view, "
                          "or the pruner finds nothing and DAS raises."))
        else:
            rep.add("P1.1", "named_parameters exposes t5_model.*", "PASS", detail=det)

        ctx["pruner_mod"].get_module_recursive(view, "t5_model.encoder.block")
        rep.add("P1.2", "module path resolves through the view", "PASS",
                detail="t5_model.encoder.block reachable")

        if args.t5_c4_encoder_only:
            rep.add("P1.3", "fp16 default on encoder-only view", "WARN",
                    detail=f"{type(view).__name__}.maybe_autocast defaults to float16",
                    note="Recorded fp16 -> NaN issue on text tokens in this repo. Call sites "
                         "pass bfloat16 explicitly; prefer the seq2seq view regardless.")

    if not phase(rep, "P1.0", "model / view load", load):
        return finish(args, rep)

    # ---------------------------------------------------------- P2 calibration
    rep.start("2 calibration contract")

    def p2():
        torch = ctx["torch"]
        texts = load_text_rows(args.text_calib, args.max_samples)
        if len(texts) % args.batch_size != 0:
            rep.add("P2.0", "sample divisibility", "WARN",
                    detail=f"{len(texts)} % {args.batch_size} != 0",
                    note="DAS averages per batch; a short trailing batch is over-weighted.")
        batches = list(iter_text_batches(texts, args.batch_size))
        pruner = build_pruner(ctx, ctx["view"], batches, len(texts))
        calib = calibrate(ctx, pruner, ctx["view"], batches, len(texts))
        if len(calib) < 5:
            raise RuntimeError(f"calibration returned {len(calib)} elements, expected 5 "
                               "(needs temp_label + temp_encoder_atts)")
        ctx["pruner"], ctx["calib"] = pruner, calib
        inps, _o, _c, imasks, amasks = calib[:5]
        problems, min_valid = [], None
        for j, (inp, img, att) in enumerate(zip(inps, imasks, amasks)):
            ib, ab = img.bool(), att.bool()
            if ib.dim() == 1:
                ib, ab = ib.unsqueeze(0), ab.unsqueeze(0)
            B, S = ib.shape
            if tuple(inp.shape[:2]) != (B, S):
                problems.append(f"batch {j}: mask {(B,S)} vs input {tuple(inp.shape[:2])}")
            if int(ib.sum().item()) != 0:
                problems.append(f"batch {j}: temp_label has True entries (text-only must be all-False)")
            for b in range(B):
                v = int(ab[b].sum().item())
                min_valid = v if min_valid is None else min(min_valid, v)
        det = {"cached_batches": len(inps), "seq_len": int(inps[0].shape[1]),
               "min_valid_tokens": min_valid}
        if problems:
            rep.add("P2.1", "view contract", "FAIL", detail=problems[:5])
        else:
            rep.add("P2.1", "view contract", "PASS", detail=det)
        if min_valid is not None and min_valid < 2:
            rep.add("P2.2", "short-text safety", "FAIL",
                    detail=f"a sample has {min_valid} valid token(s)",
                    note="The reduction raises when no modality pair is measurable.")
        else:
            rep.add("P2.2", "short-text safety", "PASS", detail=f"min valid tokens = {min_valid}")

    phase(rep, "P2.0", "calibration", p2)

    # ------------------------------------------------------- P3 reduction active
    rep.start("3 reduction active")

    def p3():
        calib = ctx["calib"]
        inps, _o, _c, imasks, amasks = calib[:5]
        cpd, cpds = ctx["cos_pairwise_density"], ctx["_cos_pairwise_density_single"]
        _v, l_b, _vl, cv, cl, cvl = cpd(inps[0].float(), imasks[0],
                                        attention_mask=amasks[0], return_counts=True)
        defined = tuple(n for n, c in (("v", cv), ("l", cl), ("vl", cvl)) if c)
        _v0, l_s0, _vl0, _c0, cl0, _cv0 = cpd(inps[0][0:1].float(), imasks[0][0:1],
                                              attention_mask=amasks[0][0:1], return_counts=True)
        dens = float(cpds(inps[0][0].float(), imasks[0][0], attention_mask=amasks[0][0]))
        ef, eu = abs(dens - l_s0), abs(dens - l_s0 / 3.0)
        det = {"defined_terms": defined, "s_l_batch_mean": round(l_b, 6),
               "s_l_sample0": round(l_s0, 6), "amia_density_sample0": round(dens, 6),
               "err_vs_fixed(s_l)": round(ef, 8), "err_vs_unfixed(s_l/3)": round(eu, 8),
               "das_importance_new": round((1.0 - l_b) * 3.0, 5),
               "das_importance_if_unfixed": round(3.0 - l_b, 5)}
        if defined == ("l",) and cl0 == 1 and ef <= 1e-6 and ef < eu:
            rep.add("P3.1", "single-modality reduction in effect", "PASS", detail=det,
                    note="DAS takes the reduction branch; AMIA density is s_l, not s_l/3; "
                         "the old +2 constant floor on DAS importance is gone.")
        elif defined == ("l",) and ef < eu:
            rep.add("P3.1", "single-modality reduction in effect", "WARN", detail=det,
                    note="Reduction is active but the two paths disagree numerically; check "
                         "that the same mask and dtype reach both.")
        else:
            rep.add("P3.1", "single-modality reduction in effect", "FAIL", detail=det,
                    note="The reduction is not active on this path.")

    phase(rep, "P3.0", "reduction branch", p3)

    # --------------------------------------------------------- P4 determinism
    rep.start("4 determinism + DAS vector")

    def p4():
        ka, va = das_layer_vector(ctx["pruner"], ctx["calib"], args.sparsity)
        kb, vb = das_layer_vector(ctx["pruner"], ctx["calib"], args.sparsity)
        if ka != kb:
            rep.add("P4.1", "determinism", "FAIL", detail="layer key sets differ")
            return
        md = max((abs(x - y) for x, y in zip(va, vb)), default=0.0)
        ctx["das_keys"], ctx["das_vec"] = ka, va
        if md == 0.0:
            rep.add("P4.1", "determinism", "PASS",
                    detail={"layers": len(ka), "max_abs_diff": 0.0,
                            "layer_sparsity_range": round(max(va) - min(va), 6)})
        else:
            rep.add("P4.1", "determinism", "FAIL", detail=f"max |delta| = {md:.3e}",
                    note="Run-to-run noise would contaminate every downstream comparison.")

    phase(rep, "P4.0", "determinism", p4)

    # ---------------------------------------------------- P5 multimodal safety
    rep.start("5 multimodal safety (protects existing results)")

    def p5():
        if not (args.mm_calib and args.mm_images
                and os.path.isfile(args.mm_calib) and os.path.isdir(args.mm_images)):
            rep.add("P5.1", "multimodal untouched", "SKIP",
                    detail="pass --mm_calib and --mm_images",
                    note="This is the check that protects your existing multimodal results.")
            return
        torch, Image = ctx["torch"], ctx["Image"]
        vis = ctx["load_processor"]("blip_image_eval").build(image_size=args.image_size)
        rows = load_rows(args.mm_calib, args.max_samples)
        mb = list(iter_batches(rows, args.mm_images, vis, torch, Image, ctx["device"], args.batch_size))
        mp = build_pruner(ctx, ctx["base_model"], mb, len(rows))
        mc = calibrate(ctx, mp, ctx["base_model"], mb, len(rows))
        cpd = ctx["cos_pairwise_density"]
        worst = None
        for j in range(len(mc[0])):
            _v, _l, _vl, cv, cl, cvl = cpd(mc[0][j].float(), mc[3][j],
                                           attention_mask=mc[4][j], return_counts=True)
            B = int(mc[3][j].shape[0])
            short = (B - cv) + (B - cl) + (B - cvl)
            if worst is None or short > worst[0]:
                worst = (short, {"batch": j, "B": B, "defined": {"v": cv, "l": cl, "vl": cvl}})
        det = worst[1] | {"undefined_slots_worst_batch": worst[0]}
        if worst[0] == 0:
            rep.add("P5.1", "multimodal untouched", "PASS", detail=det,
                    note="Every diversity term is defined for every sample, so compute_density "
                         "takes the verbatim three-term branch -> multimodal output is "
                         "byte-identical to before the change.")
        else:
            rep.add("P5.1", "multimodal untouched", "WARN", detail=det,
                    note="Some multimodal samples lack a measurable term; the change WILL alter "
                         "multimodal numbers there. Audit all calibration sets before reusing "
                         "existing multimodal results.")
        del mc, mb, mp
        ctx["torch"].cuda.empty_cache()

    phase(rep, "P5.0", "multimodal safety", p5)

    # ------------------------------------------------------------ P6 GO/NO-GO
    rep.start("6 non-degeneracy (GO / NO-GO)")

    def p6():
        torch = ctx["torch"]
        recs = run_d1_amia_selection(torch, ctx["view"], ctx["pruner_mod"], ctx["calib"],
                                     "textonly", args.probe_blocks, args.probe_linears)
        ctx["amia_recs"] = recs
        ratios = [r["select_ratio"] for r in recs] or [1.0]
        mean_ratio = statistics.mean(ratios)
        va = ctx.get("das_vec") or []
        sp_range = (max(va) - min(va)) if va else float("nan")
        det = {"amia_observations": len(recs),
               "select_ratio_mean": round(mean_ratio, 5),
               "select_ratio_min": round(min(ratios), 5),
               "select_ratio_max": round(max(ratios), 5),
               "n_selected_mean": round(statistics.mean([r["n_selected"] for r in recs] or [0]), 3),
               "valid_tokens_mean": round(statistics.mean([r["n_valid"] for r in recs] or [0]), 2),
               "das_layer_sparsity_range": round(sp_range, 6) if va else "unavailable"}
        amia_dead = mean_ratio > args.max_select_ratio
        amia_col = mean_ratio < args.min_select_ratio
        das_dead = bool(va) and sp_range < args.min_sparsity_range
        if amia_dead and das_dead:
            rep.add("P6.1", "GO / NO-GO", "FAIL", detail=det,
                    note="NO-GO: AMIA keeps ~everything AND DAS is effectively uniform. This is "
                         "vanilla Wanda with extra steps. Do not spend eval runs on it.")
        elif amia_col:
            rep.add("P6.1", "GO / NO-GO", "FAIL", detail=det,
                    note="AMIA collapsed: scaler_row rests on almost no tokens. Re-calibrate the "
                         "MMD threshold before using this path.")
        elif amia_dead or das_dead:
            rep.add("P6.1", "GO / NO-GO", "WARN", detail=det,
                    note=f"Only {'DAS' if amia_dead else 'AMIA'} is alive. Report the path as "
                         "that single mechanism, not as TAMP.")
        else:
            rep.add("P6.1", "GO / NO-GO", "PASS", detail=det,
                    note="Both AMIA and DAS do non-trivial work under text-only calibration.")

    phase(rep, "P6.0", "non-degeneracy", p6)

    # ------------------------------------------------------------ P7 ablation
    rep.start("7 component ablation (in-process, no checkpoints)")

    def p7():
        torch = ctx["torch"]
        uniform = lambda _k: args.sparsity
        das_map = dict(zip(ctx.get("das_keys") or [], ctx.get("das_vec") or []))
        if not das_map:
            rep.add("P7.1", "ablation", "SKIP",
                    detail="DAS vector unavailable (phase 4 failed)",
                    note="A/B comparison would still work but C/D cannot be built.")
            return
        das_lookup = lambda k: das_map.get(k, args.sparsity)
        print("  scaler_row for naive ...")
        sr_naive = compute_scaler_rows(ctx, ctx["view"], ctx["calib"], "naive")
        print("  scaler_row for amia ...")
        sr_amia = compute_scaler_rows(ctx, ctx["view"], ctx["calib"], "amia")
        cfg = {"A_naive_uniform": masks_for(torch, sr_naive, uniform),
               "B_amia_uniform": masks_for(torch, sr_amia, uniform),
               "C_naive_das": masks_for(torch, sr_naive, das_lookup),
               "D_amia_das": masks_for(torch, sr_amia, das_lookup)}
        names = list(cfg)
        pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pairs.append({"pair": f"{names[i]} vs {names[j]}",
                              **mask_divergence(cfg[names[i]], cfg[names[j]])})
        for p in pairs:
            print(f"       {p['pair']:34} differing = {p['differing_frac']:.4%}")
        g = lambda s: next(p for p in pairs if p["pair"] == s)
        amia_e = g("A_naive_uniform vs B_amia_uniform")
        das_e = g("A_naive_uniform vs C_naive_das")
        tot_e = g("A_naive_uniform vs D_amia_das")
        det = {"amia_only_vs_baseline": amia_e["differing_frac"],
               "das_only_vs_baseline": das_e["differing_frac"],
               "both_vs_baseline": tot_e["differing_frac"],
               "threshold": args.min_mask_divergence, "probe_layers": len(sr_naive),
               "all_pairs": pairs}
        if tot_e["differing_frac"] < args.min_mask_divergence:
            rep.add("P7.1", "does the method change anything", "FAIL", detail=det,
                    note="D produces essentially the same mask as the naive+uniform baseline. "
                         "Benchmark scores cannot differ meaningfully.")
        else:
            contrib = [n for n, e in (("AMIA", amia_e), ("DAS", das_e))
                       if e["differing_frac"] >= args.min_mask_divergence]
            rep.add("P7.1", "does the method change anything", "PASS", detail=det,
                    note=f"Mask differs from baseline by {tot_e['differing_frac']:.2%}; "
                         f"contributing: {', '.join(contrib) or 'interaction only'}. "
                         "Benchmark evaluation is now worth running.")

    phase(rep, "P7.0", "ablation", p7)
    return finish(args, rep)


def finish(args, rep) -> int:
    print("\n" + "=" * 74)
    nf, nw = len(rep.failed), len(rep.warned)
    ordered = {}
    for c in rep.checks:
        ordered.setdefault(c["phase"], []).append(c)
    print("SUMMARY")
    for ph, cs in ordered.items():
        line = "  ".join(f"{c['id']}:{c['status']}" for c in cs)
        print(f"  {ph:44} {line}")
    print()
    if nf:
        print(f"VERDICT: NOT READY -- {nf} failure(s), {nw} warning(s)")
        for c in rep.failed:
            print(f"  FAIL [{c['phase']}] {c['id']} {c['name']}")
            print(f"       {c['note'] or c['detail']}")
    elif nw:
        print(f"VERDICT: READY WITH CAVEATS -- {nw} warning(s)")
        for c in rep.warned:
            print(f"  WARN [{c['phase']}] {c['id']} {c['name']}: {c['note'] or ''}")
    else:
        print("VERDICT: READY -- every phase passed")
    print("\nNot covered: the four-benchmark evaluation. Run it only if P7 passed.")
    print("=" * 74)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump({"ready": nf == 0, "n_fail": nf, "n_warn": nw,
                   "args": vars(args), "checks": rep.checks},
                  fh, indent=2, ensure_ascii=False, default=str)
    print(f"[done] wrote {args.out_json}")
    return 1 if nf else 0


if __name__ == "__main__":
    raise SystemExit(main())
