#!/usr/bin/env python3
"""Part 2 / Stage B -- deep analysis of the calibration statistics (CPU).

Consumes the per-dataset NPZ files written by extract_wanda_statistics.py and
digs, from several angles, into WHY the calibration dataset behaves as a global
main effect (some sets good everywhere, some bad everywhere) rather than a
task-matching effect (diagonal wins). Emits tidy CSVs so you can plot yourself.

Angles (each a falsifiable question -> a CSV you can visualize):

  1. statistic geometry   pairwise cosine of the sqrt(scaler_row) vectors across
     (similarity_*.csv,    datasets; centrality = mean similarity to the others.
      centrality_*.csv,     Are some datasets central and some outliers?
      mds_*.csv)

  2. where it lives        per-block cross-dataset disagreement -- which layers,
     (per_block_*.csv)      and ViT vs T5, do the calibrations diverge in?

  3. why central/outlier   structure descriptors: activation RMS scale, tail
     (structure_*.csv)      heaviness (kurtosis), top-1% channel energy. Is the
                            outlier dataset just spikier / out-of-scale?

  4. link to accuracy      centrality vs the column-centered accuracy main
     (crosslink_*.csv,      effect (the real test), AND similarity(calib, eval)
      accuracy_link.csv)    vs accuracy(calib, eval) -- does matching the eval
                            distribution help (diagonal), or does being central
                            help everywhere?

Give it the SAME datasets you evaluated on plus any pretraining set, e.g.
--stats MMBench=.../MMBench --stats MMMU=.../MMMU ... --stats cc3m=.../cc3m.
The 4 eval benchmarks double as their own eval-reference distributions.

Usage:
  python scripts/blip2/analyze_calibration_statistics.py \
      --stats MMBench=/path/stats/MMBench \
      --stats MMMU=/path/stats/MMMU \
      --stats OKVQA=/path/stats/OKVQA \
      --stats mathvista=/path/stats/mathvista \
      --stats cc3m=/path/stats/cc3m \
      --out_dir /path/out/part2_stat_analysis \
      --group all --component both \
      --accuracy_csv /path/accuracy_matrix.csv
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from split_joint_analysis_common import (
    ensure_dir,
    parse_labeled_paths,
    setup_matplotlib,
    write_csv,
)
from analyze_calibration_mask_mechanism import classical_mds, load_accuracy_effects

T5_GROUPS = ("visual", "text", "pad")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deep statistic analysis across calibration datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stats", action="append", required=True, metavar="LABEL=DIR_OR_NPZ",
                   help="Repeatable. Output of extract_wanda_statistics.py per dataset.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--component", choices=["t5", "vit", "both"], default="both")
    p.add_argument("--group", choices=["all", "visual", "text", "pad"], default="all",
                   help="Which T5 token group's statistic to analyze (ViT is always 'all').")
    p.add_argument("--accuracy_csv", default=None,
                   help="rows=calib, cols=eval accuracy matrix for the link-to-accuracy test.")
    p.add_argument("--no_plots", action="store_true", help="CSV only; skip sanity PNGs.")
    return p.parse_args()


# --------------------------------------------------------------------------
def find_npz(path: str) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        return path
    cand = os.path.join(path, "wanda_statistics.npz")
    if os.path.isfile(cand):
        return cand
    raise FileNotFoundError("No wanda_statistics.npz under %s" % path)


def load_meta(npz_path: str) -> Dict[str, Any]:
    meta_path = os.path.join(os.path.dirname(npz_path), "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as h:
            return json.load(h)
    return {}


def statistic_vectors(
    data: Dict[str, np.ndarray], component: str, group: str
) -> Dict[str, Tuple[int, np.ndarray]]:
    """linear_key -> (block, sqrt(scaler) vector) for the requested component/group.

    Wanda uses sqrt(scaler_row), so the geometry that matters is that of the
    sqrt vectors. T5 'all' = visual+text+pad summed.
    """
    prefix = "t5enc::" if component == "t5" else "vit::"
    grp = group if component == "t5" else "all"
    out: Dict[str, Tuple[int, np.ndarray]] = {}
    for key in data.files:
        if not key.startswith(prefix) or not key.endswith("::sumsq"):
            continue
        # key = "<comp>::<block>::<linear>::<grp>::sumsq"
        parts = key.split("::")
        block = int(parts[1])
        linear = parts[2]
        this_grp = parts[3]
        lin_key = "%s::%s" % (parts[1], linear)
        if component == "t5":
            if grp == "all":
                if this_grp != "visual":
                    continue
                sumsq = np.zeros_like(data[key], dtype=np.float64)
                for g in T5_GROUPS:
                    k = "t5enc::%s::%s::%s::sumsq" % (parts[1], linear, g)
                    if k in data.files:
                        sumsq = sumsq + data[k].astype(np.float64)
            else:
                if this_grp != grp:
                    continue
                sumsq = data[key].astype(np.float64)
        else:
            if this_grp != "all":
                continue
            sumsq = data[key].astype(np.float64)
        vec = np.sqrt(np.clip(sumsq, 0.0, None))
        out[lin_key] = (block, vec)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def pairwise_similarity(
    vecs: Dict[str, Dict[str, Tuple[int, np.ndarray]]], labels: List[str]
) -> Tuple[np.ndarray, Dict[Tuple[str, str, int], List[float]]]:
    """Channel-weighted mean per-linear cosine between each dataset pair.

    Returns (NxN sim matrix, per-(a,b,block) list of cosines for the block view).
    """
    n = len(labels)
    sim = np.eye(n)
    per_block: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = vecs[labels[i]], vecs[labels[j]]
            num = 0.0
            den = 0.0
            for lin_key, (block, va) in a.items():
                if lin_key not in b:
                    continue
                vb = b[lin_key][1]
                if va.shape != vb.shape:
                    continue
                c = cosine(va, vb)
                if not np.isfinite(c):
                    continue
                w = va.size
                num += c * w
                den += w
                per_block[(labels[i], labels[j], block)].append(c)
            s = num / den if den > 0 else float("nan")
            sim[i, j] = sim[j, i] = s
    return sim, per_block


def structure_descriptors(
    vecs: Dict[str, Tuple[int, np.ndarray]], meta: Dict[str, Any], component: str, group: str
) -> Dict[str, float]:
    """Per-dataset scale + tail-heaviness of the channel statistic (why central/outlier)."""
    all_stat = []          # per-channel scaler = vec^2 (scale-free ratios use this)
    rms_terms = []
    kurts = []
    top1 = []
    # token normalization for absolute RMS (T5 only, from meta)
    tok = None
    if component == "t5":
        counts = meta.get("t5_token_counts", {})
        if group == "all":
            tok = sum(counts.get(g, 0.0) for g in T5_GROUPS)
        else:
            tok = counts.get(group, 0.0)
    for _lin, (_block, vec) in vecs.items():
        s = vec.astype(np.float64) ** 2  # per-channel sum of squares
        total = s.sum()
        if total <= 0:
            continue
        all_stat.append(s)
        if tok and tok > 0:
            rms_terms.append(float(np.sqrt(s.sum() / (tok * s.size))))  # mean RMS activation
        # tail heaviness of the channel-statistic distribution
        m = s.mean()
        sd = s.std()
        if sd > 0:
            kurts.append(float(((s - m) ** 4).mean() / (sd ** 4)))
        k = max(1, int(0.01 * s.size))
        top1.append(float(np.sort(s)[-k:].sum() / total))
    return {
        "rms_activation": float(np.mean(rms_terms)) if rms_terms else float("nan"),
        "channel_kurtosis": float(np.mean(kurts)) if kurts else float("nan"),
        "top1pct_energy_fraction": float(np.mean(top1)) if top1 else float("nan"),
        "num_linears": len(all_stat),
    }


def centrality_of(sim: np.ndarray, labels: List[str]) -> Dict[str, float]:
    n = len(labels)
    return {labels[i]: float(np.mean([sim[i, j] for j in range(n) if j != i])) for i in range(n)}


# --------------------------------------------------------------------------
def analyze_component(component: str, group: str, datasets, metas, labels, args, plt):
    vecs = {lab: statistic_vectors(datasets[lab], component, group) for lab in labels}
    present = [lab for lab in labels if vecs[lab]]
    if len(present) < 2:
        print("[skip] %s: <2 datasets have statistics" % component)
        return None
    labels = present

    sim, per_block = pairwise_similarity(vecs, labels)
    centrality = centrality_of(sim, labels)

    tag = "%s_%s" % (component, group) if component == "t5" else component

    sim_rows = [{"component": component, "group": group if component == "t5" else "all",
                 "dataset_a": labels[i], "dataset_b": labels[j], "cosine_similarity": sim[i, j]}
                for i in range(len(labels)) for j in range(len(labels))]
    write_csv(os.path.join(args.out_dir, "similarity_%s.csv" % tag), sim_rows)

    cen_rows = [{"component": component, "group": group if component == "t5" else "all",
                 "dataset": lab, "centrality": centrality[lab]}
                for lab in sorted(centrality, key=centrality.get, reverse=True)]
    write_csv(os.path.join(args.out_dir, "centrality_%s.csv" % tag), cen_rows)

    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    coords = classical_mds(dist)
    mds_rows = [{"component": component, "dataset": labels[i],
                 "mds_1": float(coords[i, 0]), "mds_2": float(coords[i, 1]),
                 "centrality": centrality[labels[i]]} for i in range(len(labels))]
    write_csv(os.path.join(args.out_dir, "mds_%s.csv" % tag), mds_rows)

    block_rows = []
    blocks = defaultdict(list)
    for (a, b, block), cs in per_block.items():
        blocks[block].extend([1.0 - c for c in cs])
    for block in sorted(blocks):
        vals = np.asarray(blocks[block])
        block_rows.append({"component": component, "block": block,
                           "mean_disagreement": float(vals.mean()),
                           "max_disagreement": float(vals.max())})
    write_csv(os.path.join(args.out_dir, "per_block_disagreement_%s.csv" % tag), block_rows)

    struct_rows = []
    for lab in labels:
        d = structure_descriptors(vecs[lab], metas.get(lab, {}), component, group)
        d.update({"component": component, "group": group if component == "t5" else "all",
                  "dataset": lab, "centrality": centrality[lab]})
        struct_rows.append(d)
    write_csv(os.path.join(args.out_dir, "structure_%s.csv" % tag), struct_rows)

    if plt is not None and not args.no_plots:
        _plot_heatmap(plt, labels, sim, "%s statistic similarity" % tag,
                      os.path.join(args.out_dir, "similarity_%s.png" % tag))
        _plot_mds(plt, labels, coords, centrality, "%s statistic space" % tag,
                  os.path.join(args.out_dir, "mds_%s.png" % tag))

    return {"labels": labels, "sim": sim, "centrality": centrality}


def _plot_heatmap(plt, labels, mat, title, path):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(1.1 * n + 2.5, 1.1 * n + 2))
    im = ax.imshow(mat, cmap="viridis", vmin=float(np.nanmin(mat)), vmax=1.0)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, "%.3f" % mat[i, j], ha="center", va="center", fontsize=9,
                    color="white" if mat[i, j] < (np.nanmin(mat) + 1) / 2 else "black")
    ax.set_title(title); fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def _plot_mds(plt, labels, coords, centrality, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    c = [centrality[l] for l in labels]
    sc = ax.scatter(coords[:, 0], coords[:, 1], s=240, c=c, cmap="viridis",
                    edgecolors="black", zorder=3)
    for i, l in enumerate(labels):
        ax.annotate(l, (coords[i, 0], coords[i, 1]), fontsize=11, xytext=(8, 6),
                    textcoords="offset points")
    ax.set_title(title); ax.grid(True, alpha=0.28)
    fig.colorbar(sc, ax=ax, shrink=0.82, label="centrality")
    fig.tight_layout(); fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def accuracy_link(result, args, plt):
    if result is None or not args.accuracy_csv:
        return None
    effects, evals, mat, acc_labels = load_accuracy_effects(args.accuracy_csv)
    labels = result["labels"]
    sim = result["sim"]
    centrality = result["centrality"]
    idx = {lab: i for i, lab in enumerate(labels)}

    # (1) centrality vs global effect
    shared = [l for l in labels if l in effects]
    corr_cen = None
    if len(shared) >= 3:
        xs = np.asarray([centrality[l] for l in shared])
        ys = np.asarray([effects[l] for l in shared])
        if xs.std() > 0 and ys.std() > 0:
            corr_cen = float(np.corrcoef(xs, ys)[0, 1])

    # (2) similarity(calib, eval) vs accuracy(calib, eval)
    link_rows = []
    for ci, cal in enumerate(acc_labels):
        for ei, ev in enumerate(evals):
            if cal in idx and ev in idx:
                s = float(sim[idx[cal], idx[ev]])
                link_rows.append({"calib": cal, "eval": ev, "statistic_similarity": s,
                                  "accuracy": float(mat[ci, ei]), "is_diagonal": int(cal == ev)})
    write_csv(os.path.join(args.out_dir, "accuracy_link.csv"), link_rows)

    corr_link = None
    off = [(r["statistic_similarity"], r["accuracy"]) for r in link_rows if not r["is_diagonal"]]
    if len(off) >= 3:
        xs = np.asarray([o[0] for o in off]); ys = np.asarray([o[1] for o in off])
        if xs.std() > 0 and ys.std() > 0:
            corr_link = float(np.corrcoef(xs, ys)[0, 1])

    if plt is not None and not args.no_plots and len(shared) >= 3:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        xs = [centrality[l] for l in shared]; ys = [effects[l] for l in shared]
        ax.scatter(xs, ys, s=140, color="#4C78A8", edgecolors="black", zorder=3)
        for l in shared:
            ax.annotate(l, (centrality[l], effects[l]), fontsize=11, xytext=(7, 5),
                        textcoords="offset points")
        if np.std(xs) > 0:
            b, a = np.polyfit(xs, ys, 1)
            xl = np.linspace(min(xs), max(xs), 40)
            ax.plot(xl, b * xl + a, color="#E45756",
                    label="r=%.2f" % (corr_cen if corr_cen is not None else float("nan")))
            ax.legend()
        ax.set_xlabel("statistic centrality"); ax.set_ylabel("global accuracy effect")
        ax.set_title("Central statistic -> better pruning?")
        ax.grid(True, alpha=0.28); fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "centrality_vs_accuracy.png"), dpi=220, bbox_inches="tight")
        plt.close(fig)

    return {"centrality_accuracy_r": corr_cen, "similarity_accuracy_r": corr_link,
            "row_effects": effects}


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    stats = parse_labeled_paths(args.stats)
    labels = list(stats)

    datasets: Dict[str, Any] = {}
    metas: Dict[str, Any] = {}
    for lab, path in stats.items():
        npz = find_npz(path)
        print("[load] %-10s %s" % (lab, npz))
        datasets[lab] = np.load(npz)
        metas[lab] = load_meta(npz)

    plt = setup_matplotlib()
    components = ["t5", "vit"] if args.component == "both" else [args.component]

    summary: Dict[str, Any] = {"datasets": {l: find_npz(p) for l, p in stats.items()},
                               "group": args.group, "components": {}}
    primary = None
    for comp in components:
        res = analyze_component(comp, args.group, datasets, metas, labels, args, plt)
        if res is not None:
            summary["components"][comp] = {"centrality": res["centrality"]}
            if primary is None or comp == "t5":
                primary = res

    link = accuracy_link(primary, args, plt)
    if link is not None:
        summary["accuracy_link"] = {"centrality_accuracy_r": link["centrality_accuracy_r"],
                                    "similarity_accuracy_r": link["similarity_accuracy_r"]}

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as h:
        json.dump(summary, h, indent=2)

    # ---- verdict ----
    for comp in components:
        if comp not in summary["components"]:
            continue
        cen = summary["components"][comp]["centrality"]
        ranked = sorted(cen, key=cen.get, reverse=True)
        print("\n=== %s statistic centrality (group=%s) ===" % (comp.upper(), args.group))
        print("  most central -> %s ;  most outlier -> %s" % (ranked[0], ranked[-1]))
        print("  " + "  ".join("%s=%.4f" % (l, cen[l]) for l in ranked))

    if link is not None:
        print("\n=== link to accuracy ===")
        r1 = link["centrality_accuracy_r"]; r2 = link["similarity_accuracy_r"]
        if r1 is not None:
            print("  r(centrality, global accuracy effect) = %+.3f" % r1)
        if r2 is not None:
            print("  r(similarity(calib,eval), accuracy)   = %+.3f  (off-diagonal pairs)" % r2)
        print("\n  Read:")
        print("   - r(centrality, .) strongly positive  -> representativeness IS the cause:")
        print("     a central statistic prunes better on every benchmark; task-matching is a red herring.")
        print("   - r(similarity(calib,eval), .) near 0 -> matching the eval distribution does NOT help,")
        print("     confirming the diagonal is not special.")
        print("   - if BOTH are near 0 -> the cause is not in the statistic geometry; look at the")
        print("     structure_*.csv (scale / tail) or move downstream to the pruned-model activations.")

    print("\n[OK] wrote CSVs to:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
