#!/usr/bin/env python3
"""End-to-end TAMP calibration-probe study for BLIP2-T5.

Loads the model ONCE, sweeps every calibration set, and produces the cross-set
comparison that actually answers the research question:

    does the calibration set change TAMP's read-out by more than the noise floor?

Per calibration set it records (production code paths only, nothing reimplemented):

    A  contamination audit   how often a v/l/vl diversity term could not be measured
                             and silently defaulted to 0.0 ("maximally diverse"), plus
                             how much the >0 similarity filter discards
    B  DAS layer sparsity    the full 168-dim vector, and a within-set noise floor from
                             randomised repeated split-half
    C  AMIA selection        per-block selected/valid ratio and visual/text composition
    D  sequence lengths      valid text tokens per sample (length-confound check)

Then across sets:

    - between-set DAS difference vs within-set noise floor  -> signal or noise
    - Spearman matrix over layer sparsity
    - AMIA modality-vs-depth curves per calibration set
    - contamination / length tables, to catch confounds before interpreting anything

Nothing is pruned. Weights are never modified.

Example:

  CUDA_VISIBLE_DEVICES=0 python scripts/blip2/tamp_calib_study.py \
    --ckpt /data/data2/mfs/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth \
    --calib mmbench=/data/data2/mfs/MMBench_calibration/mmbench_calibration_train.json:/data/data2/mfs/MMBench_calibration/images \
    --calib cc3m=/data/data2/mfs/CC3M_calib_128/cc3m_calib_128.json:/data/data2/mfs/CC3M_calib_128/images \
    --max_samples 128 --batch_size 8 \
    --out_dir tamp_calib_study_$(date +%Y%m%d_%H%M%S)

Use --base to fill in the five standard sets without typing paths:

  python scripts/blip2/tamp_calib_study.py --ckpt ... --base /data/data2/mfs
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
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
    _pearson,
    append_csv,
    das_layer_vector,
    iter_batches,
    load_rows,
    run_d1_amia_selection,
    run_d2_das_noise_floor,
    run_d3_lengths,
    spearman,
)

STANDARD_SETS = {
    "mmbench": ("MMBench_calibration/mmbench_calibration_train.json", "MMBench_calibration/images"),
    "mmmu": ("MMMU_calibration/mmmu_calibration_train.json", "MMMU_calibration/images"),
    "okvqa": ("datasets/okvqa/annotations/okvqa_train.json", "datasets/okvqa/images"),
    "mathvista": ("MathVista_calibration/mathvista_calibration_train.json", "MathVista_calibration/images"),
    "cc3m": ("CC3M_calib_128/cc3m_calib_128.json", "CC3M_calib_128/images"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-calibration-set TAMP probe study (no pruning).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument(
        "--calib", action="append", default=[], metavar="LABEL=JSON:IMAGES",
        help="Calibration set. Repeatable. Overrides/extends --base.",
    )
    p.add_argument("--base", default=None, help="Data root; fills in the five standard sets.")
    p.add_argument("--model_name", default="blip2_t5")
    p.add_argument("--model_type", default="pretrain_flant5xl")
    p.add_argument("--device", default=None)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--max_samples", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_txt_len", type=int, default=None)
    p.add_argument("--sparsity", type=float, default=0.5)
    p.add_argument("--expected_query_tokens", type=int, default=32)
    p.add_argument("--probe_blocks", default="all", help="Blocks probed for AMIA: 'all' or 0,11,23")
    p.add_argument(
        "--probe_linears", default="SelfAttention.v,DenseReluDense.wo",
        help="Linear suffixes probed for AMIA ('all' is ~3x slower).",
    )
    p.add_argument("--skip_amia", action="store_true", help="Skip section C (much faster).")
    p.add_argument("--signal_noise_ratio", type=float, default=3.0,
                   help="between/within ratio above which a pair counts as real signal.")
    p.add_argument("--out_dir", default="tamp_calib_study")
    return p.parse_args()


def resolve_sets(args) -> List[Tuple[str, str, str]]:
    out: "collections.OrderedDict[str, Tuple[str, str]]" = collections.OrderedDict()
    if args.base:
        for label, (j, im) in STANDARD_SETS.items():
            out[label] = (os.path.join(args.base, j), os.path.join(args.base, im))
    for item in args.calib:
        if "=" not in item or ":" not in item.split("=", 1)[1]:
            raise SystemExit(f"[ERROR] --calib must be LABEL=JSON:IMAGES, got {item}")
        label, rest = item.split("=", 1)
        j, im = rest.rsplit(":", 1)
        out[label] = (j, im)
    if not out:
        raise SystemExit("[ERROR] no calibration sets: pass --base and/or --calib")
    resolved = []
    for label, (j, im) in out.items():
        if not os.path.isfile(j):
            print(f"[WARN] skip {label}: missing {j}")
            continue
        if not os.path.isdir(im):
            print(f"[WARN] skip {label}: missing {im}")
            continue
        resolved.append((label, j, im))
    if not resolved:
        raise SystemExit("[ERROR] none of the calibration sets exist")
    return resolved


# ------------------------------------------------------------------ per set


def run_one_set(mods, model, label, calib_json, images_dir, args, out_dir) -> Dict[str, Any]:
    torch = mods["torch"]
    Image = mods["Image"]
    pruner_mod = mods["pruner_mod"]
    BLIPT5LayerWandaPruner = mods["BLIPT5LayerWandaPruner"]
    T5LayerWandaPruner = mods["T5LayerWandaPruner"]
    device = mods["device"]
    vis_processor = mods["vis_processor"]

    rows = load_rows(calib_json, args.max_samples)
    if len(rows) % args.batch_size != 0:
        print(f"[WARN] {label}: {len(rows)} rows not divisible by batch_size {args.batch_size}; "
              "the trailing short batch is over-weighted in DAS.")
    batches = list(iter_batches(rows, images_dir, vis_processor, torch, Image, device, args.batch_size))

    pruner = BLIPT5LayerWandaPruner(
        model=model, data_loader=batches,
        t5_prune_spec="24-%.6f-1.0-1.0" % (1.0 - args.sparsity),
        vit_prune_spec=None, t5_pruning_method="none", vit_pruning_method="none",
        num_samples=len(rows), num_data_first_stage=len(rows),
        sparsity_ratio_granularity="layer",
        max_sparsity_per_layer=min(1.0, args.sparsity + 0.1),
        score_method="density_sum", token_selection="amia",
        prune_t5=True, prune_vit=False, importance_scope="llm_only",
    )

    with torch.no_grad():
        calib = T5LayerWandaPruner.prepare_calibration_input_encoder(
            pruner, model, batches, device, "t5_model", len(rows),
            module_to_process="t5_model.encoder.block", return_image_masks=True,
        )
    if len(calib) < 5:
        raise SystemExit(f"[ERROR] {label}: calibration lacks encoder_attention_masks.")
    print(f"  cached_batches={len(calib[0])}")

    res: Dict[str, Any] = {"label": label, "calib_json": calib_json, "n_rows": len(rows)}

    # --- D: lengths ---
    d3_rows, d3 = run_d3_lengths(calib, label, args.expected_query_tokens)
    append_csv(os.path.join(out_dir, "d_lengths.csv"),
               ["label", "batch", "sample", "seq_len", "n_valid_text", "n_pad_text"], d3_rows)
    res["lengths"] = d3
    print(f"  [D] valid text tokens: mean={d3['valid_text_mean']} "
          f"min={d3['valid_text_min']} max={d3['valid_text_max']}")

    # --- A + B: contamination audit runs inside compute_density ---
    audit_path = os.path.join(out_dir, f"a_audit_{label}.json")
    os.environ["LAVIS_DAS_DIAGNOSTIC"] = "1"
    os.environ["LAVIS_DAS_DIAGNOSTIC_JSON"] = audit_path
    try:
        keys, vec = das_layer_vector(pruner, calib, args.sparsity)
    finally:
        os.environ.pop("LAVIS_DAS_DIAGNOSTIC", None)
        os.environ.pop("LAVIS_DAS_DIAGNOSTIC_JSON", None)
    if os.path.isfile(audit_path):
        with open(audit_path, encoding="utf-8") as fh:
            res["audit"] = json.load(fh)
        print(f"  [A] contaminated_frac={res['audit']['contaminated_frac']} "
              f"l_positive_pair_frac={res['audit']['l_positive_pair_frac']} "
              f"no_attention_mask={res['audit']['no_attention_mask']}")
    else:
        res["audit"] = {}
        print("  [A] audit file not written (LAVIS_DAS_DIAGNOSTIC not honoured?)")

    res["das_keys"] = keys
    res["das_vector"] = vec
    append_csv(
        os.path.join(out_dir, "b_layer_sparsity.csv"),
        ["label", "layer_key", "block", "linear", "sparsity"],
        [{"label": label, "layer_key": k,
          "block": int(k.split("encoder.block.")[1].split(".")[0]),
          "linear": k.split("encoder.block.")[1].split(".", 1)[1].rsplit(".weight", 1)[0],
          "sparsity": round(v, 8)} for k, v in zip(keys, vec)],
    )

    # --- B: within-set noise floor ---
    d2 = run_d2_das_noise_floor(pruner, calib, args.sparsity)
    d2.pop("_layer_keys", None)
    d2.pop("_layer_vector", None)
    res["noise_floor"] = d2
    append_csv(os.path.join(out_dir, "b_noise_floor.csv"),
               ["label"] + list(d2.keys()), [dict(d2, label=label)])
    print(f"  [B] within-set noise: mean|Δ|={d2['mean_abs_diff']} "
          f"spearman={d2['spearman']} SNR(range/noise)={d2['snr_range_over_noise']}")

    # --- C: AMIA selection ---
    if not args.skip_amia:
        recs = run_d1_amia_selection(torch, model, pruner_mod, calib, label,
                                     args.probe_blocks, args.probe_linears)
        append_csv(
            os.path.join(out_dir, "c_amia_selection.csv"),
            ["label", "block", "linear", "batch", "n_valid", "n_selected", "select_ratio",
             "n_visual_selected", "n_text_selected", "n_visual_available"], recs,
        )
        by_block: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
        for r in recs:
            by_block[int(r["block"])].append(r)
        curve = {}
        for b, rs in sorted(by_block.items()):
            fr = [r["n_visual_selected"] / max(1, r["n_selected"]) for r in rs]
            curve[b] = {
                "vis_frac_mean": round(statistics.mean(fr), 5),
                "vis_frac_sem": round(statistics.stdev(fr) / (len(fr) ** 0.5), 5) if len(fr) > 1 else 0.0,
                "n_selected_mean": round(statistics.mean(r["n_selected"] for r in rs), 3),
                "select_ratio_mean": round(statistics.mean(r["select_ratio"] for r in rs), 5),
                "n_obs": len(rs),
            }
        res["amia_curve"] = curve
        ratios = [r["select_ratio"] for r in recs] or [0.0]
        res["amia_summary"] = {
            "observations": len(recs),
            "select_ratio_mean": round(statistics.mean(ratios), 5),
            "n_selected_mean": round(statistics.mean([r["n_selected"] for r in recs] or [0]), 3),
            "zero_visual_frac": round(sum(1 for r in recs if r["n_visual_selected"] == 0) / max(1, len(recs)), 5),
        }
        append_csv(
            os.path.join(out_dir, "c_amia_by_block.csv"),
            ["label", "block", "vis_frac_mean", "vis_frac_sem", "n_selected_mean",
             "select_ratio_mean", "n_obs"],
            [dict(v, label=label, block=b) for b, v in curve.items()],
        )
        print(f"  [C] AMIA select_ratio={res['amia_summary']['select_ratio_mean']} "
              f"m={res['amia_summary']['n_selected_mean']} "
              f"zero_visual_frac={res['amia_summary']['zero_visual_frac']}")

    del calib, batches, pruner
    torch.cuda.empty_cache()
    return res


# ------------------------------------------------------------------ analysis


def cross_set_analysis(results: List[Dict[str, Any]], args, out_dir) -> Dict[str, Any]:
    labels = [r["label"] for r in results]
    ref_keys = results[0]["das_keys"]
    for r in results:
        if r["das_keys"] != ref_keys:
            raise SystemExit("[ERROR] layer key sets differ between calibration sets")

    pairs, matrix_rows = [], []
    for i, a in enumerate(results):
        row = {"label": a["label"]}
        for j, b in enumerate(results):
            if i == j:
                row[b["label"]] = 1.0
                continue
            va, vb = a["das_vector"], b["das_vector"]
            diffs = [abs(x - y) for x, y in zip(va, vb)]
            between = statistics.mean(diffs)
            within = statistics.mean([
                a["noise_floor"]["mean_abs_diff"], b["noise_floor"]["mean_abs_diff"]
            ])
            snr = between / max(within, 1e-12)
            row[b["label"]] = round(spearman(va, vb), 6)
            if i < j:
                pairs.append({
                    "set_a": a["label"], "set_b": b["label"],
                    "between_mean_abs_diff": round(between, 6),
                    "within_noise_floor": round(within, 6),
                    "snr": round(snr, 2),
                    "spearman": round(spearman(va, vb), 6),
                    "max_abs_diff": round(max(diffs), 6),
                    "verdict": "SIGNAL" if snr >= args.signal_noise_ratio else "NOISE",
                })
        matrix_rows.append(row)

    append_csv(os.path.join(out_dir, "x_das_spearman_matrix.csv"), ["label"] + labels, matrix_rows)
    append_csv(
        os.path.join(out_dir, "x_pairwise_signal_vs_noise.csv"),
        ["set_a", "set_b", "between_mean_abs_diff", "within_noise_floor", "snr",
         "spearman", "max_abs_diff", "verdict"], pairs,
    )
    return {"labels": labels, "spearman_matrix": matrix_rows, "pairs": pairs}


def write_report(results, cross, args, out_dir) -> str:
    L: List[str] = []
    A = L.append
    A("# TAMP calibration-probe study\n")
    A(f"- calibration sets: {', '.join(r['label'] for r in results)}")
    A(f"- samples per set: {args.max_samples} (batch {args.batch_size}), target sparsity {args.sparsity}")
    A("- no weights were pruned; all numbers come from the production DAS/AMIA code paths\n")

    A("## 1. Instrument health (read this before interpreting anything)\n")
    A("| set | contaminated_frac | no_attention_mask | l_positive_pair_frac | valid_text_mean |")
    A("|---|---|---|---|---|")
    for r in results:
        au = r.get("audit", {})
        A(f"| {r['label']} | {au.get('contaminated_frac', 'n/a')} | {au.get('no_attention_mask', 'n/a')} "
          f"| {au.get('l_positive_pair_frac', 'n/a')} | {r['lengths']['valid_text_mean']} |")
    bad = [r["label"] for r in results if r.get("audit", {}).get("contaminated_frac", 0)]
    nomask = [r["label"] for r in results if r.get("audit", {}).get("no_attention_mask", 0)]
    A("")
    if nomask:
        A(f"> **FAIL** PAD guard missing in: {nomask}. PAD tokens are being counted as language tokens.")
    if bad:
        A(f"> **WARN** unmeasurable diversity terms defaulted to 0.0 in: {bad}. "
          "Layer importance there is inflated toward 'maximally diverse'. "
          "If the rate differs across sets it is a confound, not a calibration effect.")
    if not bad and not nomask:
        A("> **OK** every diversity term was measurable in every set; no contamination.")
    lens = [r["lengths"]["valid_text_mean"] for r in results]
    if lens and (max(lens) - min(lens)) > 0.25 * max(max(lens), 1):
        A(f"> **WARN** valid-text length varies a lot across sets ({min(lens)}..{max(lens)}). "
          "T5 relative-position effects scale with length, so length co-varies with your independent variable.")
    A("")

    A("## 2. Does the calibration set move DAS beyond the noise floor?\n")
    A("Within-set noise floor = randomised repeated split-half on the same set.\n")
    A("| set | within-set mean\\|Δ\\| | spearman(split-half) | layer sparsity range | SNR |")
    A("|---|---|---|---|---|")
    for r in results:
        nf = r["noise_floor"]
        A(f"| {r['label']} | {nf['mean_abs_diff']} | {nf['spearman']} | "
          f"{nf['layer_sparsity_range']} | {nf['snr_range_over_noise']} |")
    A("")
    A("| pair | between mean\\|Δ\\| | within floor | SNR | spearman | verdict |")
    A("|---|---|---|---|---|---|")
    for p in cross["pairs"]:
        A(f"| {p['set_a']} vs {p['set_b']} | {p['between_mean_abs_diff']} | {p['within_noise_floor']} "
          f"| **{p['snr']}x** | {p['spearman']} | **{p['verdict']}** |")
    n_sig = sum(1 for p in cross["pairs"] if p["verdict"] == "SIGNAL")
    A("")
    A(f"> {n_sig}/{len(cross['pairs'])} pairs exceed {args.signal_noise_ratio}x the noise floor.")
    if n_sig == 0:
        A("> **DAS is insensitive to the calibration set at this sample size.** Either calibration "
          "genuinely does not move layer allocation, or the probe cannot resolve it. Do not report "
          "a calibration effect from DAS without a larger sample or a more sensitive read-out.")
    A("")

    if any("amia_curve" in r for r in results):
        A("## 3. AMIA: modality reliance vs depth\n")
        A("Fraction of selected tokens that are visual. Chance level = visual/(visual+text) available.\n")
        blocks = sorted({b for r in results if "amia_curve" in r for b in r["amia_curve"]})
        A("| block | " + " | ".join(r["label"] for r in results if "amia_curve" in r) + " |")
        A("|---" * (1 + sum(1 for r in results if "amia_curve" in r)) + "|")
        for b in blocks:
            cells = []
            for r in results:
                if "amia_curve" not in r:
                    continue
                c = r["amia_curve"].get(b)
                cells.append(f"{c['vis_frac_mean']:.3f}±{c['vis_frac_sem']:.3f}" if c else "-")
            A(f"| {b} | " + " | ".join(cells) + " |")
        A("")
        A("| set | select_ratio | m (tokens kept) | zero-visual frac |")
        A("|---|---|---|---|")
        for r in results:
            if "amia_summary" not in r:
                continue
            s = r["amia_summary"]
            A(f"| {r['label']} | {s['select_ratio_mean']} | {s['n_selected_mean']} | {s['zero_visual_frac']} |")
        A("")
        A("> select_ratio near 1.0 means AMIA is effectively off; near 0 means the input activation "
          "is estimated from too few tokens. A depth-varying visual fraction reproduces TAMP's "
          "Figure-4 claim; if the curves differ across calibration sets, that difference is the "
          "calibration effect on modality reliance.")
        A("")

    A("## 4. Files\n")
    A("| file | contents |")
    A("|---|---|")
    A("| `a_audit_<set>.json` | contamination audit per set |")
    A("| `b_layer_sparsity.csv` | full DAS layer sparsity vector per set |")
    A("| `b_noise_floor.csv` | within-set split-half noise floor |")
    A("| `c_amia_selection.csv` | every AMIA selection observation |")
    A("| `c_amia_by_block.csv` | modality-vs-depth curve per set |")
    A("| `d_lengths.csv` | valid text tokens per sample |")
    A("| `x_das_spearman_matrix.csv` | cross-set Spearman |")
    A("| `x_pairwise_signal_vs_noise.csv` | signal/noise verdict per pair |")

    text = "\n".join(L)
    path = os.path.join(out_dir, "REPORT.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main() -> int:
    args = parse_args()
    sets = resolve_sets(args)
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        import torch
        from PIL import Image
        from lavis.models import load_model
        from lavis.processors import load_processor
        from lavis.compression.pruners import wanda_pruner as pruner_mod
        from lavis.compression.pruners.wanda_pruner import (
            BLIPT5LayerWandaPruner, T5LayerWandaPruner,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(f"[ERROR] needs the LAVIS runtime; missing module: {exc.name}") from exc

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] {args.model_name}/{args.model_type} device={device}")
    model = load_model(args.model_name, args.model_type, is_eval=True,
                       device=device, checkpoint=args.ckpt)
    model.eval()
    if args.max_txt_len is not None:
        model.max_txt_len = int(args.max_txt_len)

    mods = {
        "torch": torch, "Image": Image, "pruner_mod": pruner_mod,
        "BLIPT5LayerWandaPruner": BLIPT5LayerWandaPruner,
        "T5LayerWandaPruner": T5LayerWandaPruner,
        "device": device,
        "vis_processor": load_processor("blip_image_eval").build(image_size=args.image_size),
    }

    results = []
    for label, cj, cim in sets:
        print(f"\n=== {label} ===")
        results.append(run_one_set(mods, model, label, cj, cim, args, args.out_dir))

    print("\n=== cross-set analysis ===")
    cross = cross_set_analysis(results, args, args.out_dir)
    report = write_report(results, cross, args, args.out_dir)

    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump({"args": vars(args), "results": results, "cross": cross},
                  fh, indent=2, ensure_ascii=False, default=str)

    print("\n" + report)
    print(f"\n[done] {args.out_dir}/REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
