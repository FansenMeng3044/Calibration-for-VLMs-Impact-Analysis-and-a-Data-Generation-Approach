#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a lightweight evidence package for validating the ATV-Pruning migration
from the original Qwen-VL implementation to BLIP2-T5.

This script intentionally does not run pruning or evaluation by itself. It
checks static code evidence, runs synthetic golden tests for the ATV token
selection math, and can optionally parse ATV logs and compare checkpoint masks
after GPU pruning jobs have finished.

Example:
  python scripts/blip2/validate_atv_migration.py \
    --original_atv_root /data/data2/mfs/ATV-Pruning \
    --lavis_root /data/data2/mfs/2/LAVIS_backup \
    --out_dir /data/data2/mfs/atv_validation_report

Optional post-run evidence:
  python scripts/blip2/validate_atv_migration.py ... \
    --atv_log alpha1=/path/to/atv_alpha1.log \
    --mask_pair atv_vs_wanda /path/to/atv.pth /path/to/wanda.pth \
    --metrics_jsonl /path/to/fourbench_metrics.jsonl \
    --okvqa_eval_txt atv_cc3m=/path/to/OKVQA/evaluate.txt

The script always writes the artifact names required by the validation plan.
When optional GPU evidence is absent, the corresponding CSV/PNG files are
explicit placeholders and the final report marks them as missing evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import random
import shutil
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


MASK_BASE_REQUIRED_GROUPS = ("t5_model", "visual_encoder", "Qformer", "query_tokens", "t5_proj")


@dataclass
class Evidence:
    name: str
    source: str
    fragment: str
    path: Path
    line: Optional[int]

    @property
    def ok(self) -> bool:
        return self.line is not None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_line(text: str, fragment: str) -> Optional[int]:
    for i, line in enumerate(text.splitlines(), start=1):
        if fragment in line:
            return i
    return None


def make_evidence(name: str, source: str, fragment: str, path: Path, text: str) -> Evidence:
    return Evidence(
        name=name,
        source=source,
        fragment=fragment,
        path=path,
        line=find_line(text, fragment),
    )


def collect_static_evidence(original_root: Path, lavis_root: Path) -> List[Evidence]:
    original_pruner = original_root / "qwen" / "activation_aware_pruner.py"
    lavis_pruner = lavis_root / "lavis" / "compression" / "pruners" / "wanda_pruner.py"
    blip2_t5 = lavis_root / "lavis" / "models" / "blip2_models" / "blip2_t5.py"
    evaluate_blip = lavis_root / "evaluate_blip.py"
    run_atv = lavis_root / "scripts" / "blip2" / "run_atv_cc3m_prune_then_eval.sh"
    validator = lavis_root / "scripts" / "blip2" / "validate_atv_migration.py"
    run_full_verify = lavis_root / "scripts" / "blip2" / "run_atv_full_verify.sh"
    preflight = lavis_root / "scripts" / "blip2" / "preflight_atv_validation.py"

    files = {
        original_pruner: read_text(original_pruner),
        lavis_pruner: read_text(lavis_pruner),
        blip2_t5: read_text(blip2_t5),
        evaluate_blip: read_text(evaluate_blip),
        run_atv: read_text(run_atv),
        validator: read_text(validator),
        run_full_verify: read_text(run_full_verify),
        preflight: read_text(preflight),
    }

    checks = [
        ("original WrappedATV exists", "original", "class WrappedATV", original_pruner),
        ("original uses ATV branch", "original", "if self.pruning_method == 'atv':", original_pruner),
        ("original uses cosine distance", "original", "cos_dist = 1 - cos_sim", original_pruner),
        (
            "original uses ATV k formula",
            "original",
            "k = round(min(1, self.alpha * cos_dist_avg) * num_text_tokens)",
            original_pruner,
        ),
        ("original clamps k to image-token count", "original", "k = min(num_img_tokens, k)", original_pruner),
        ("original registers WrappedATV layer", "original", "wrapped_layers[name] = WrappedATV(subset[name])", original_pruner),
        ("lavis WrappedATV exists", "lavis", "class WrappedATV", lavis_pruner),
        ("lavis keeps valid text tokens", "lavis", "text_tokens = inp_b[text_mask]", lavis_pruner),
        ("lavis removes padded text tokens", "lavis", "text_mask = text_mask & attn_b", lavis_pruner),
        ("lavis selects image/query tokens", "lavis", "img_tokens = inp_b[mask_b]", lavis_pruner),
        ("lavis computes cosine distance", "lavis", "cos_dist = (1.0 - cos).detach()", lavis_pruner),
        ("lavis selects ATV tokens per sample", "lavis", "for b, cos_dist in enumerate(cos_dist_list[j]):", lavis_pruner),
        ("lavis uses ATV k formula", "lavis", "k = int(round(min(1.0, alpha * cos_dist_avg) * num_text))", lavis_pruner),
        ("lavis clamps k to query-token count", "lavis", "k = max(0, min(num_img, k))", lavis_pruner),
        ("lavis top-k selected query tokens", "lavis", "torch.topk(cos_dist, k=k)", lavis_pruner),
        ("lavis scaler uses sample normalization", "lavis", "tmp = active_samples", lavis_pruner),
        ("lavis requires temp_label masks", "lavis", "requires image_masks (multimodal calib + temp_label)", lavis_pruner),
        ("blip2 creates T5 query embeddings", "blip2_t5", "inputs_t5 = self.t5_proj(query_output.last_hidden_state)", blip2_t5),
        ("blip2 concatenates query and text embeddings", "blip2_t5", "inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)", blip2_t5),
        ("blip2 marks query tokens in temp_label", "blip2_t5", "temp_label[:, :num_query] = True", blip2_t5),
        ("evaluate_blip exposes atv token selection", "evaluate_blip", 'choices=["naive", "amia", "atv"]', evaluate_blip),
        ("evaluate_blip exposes ATV alpha", "evaluate_blip", "--atv_alpha", evaluate_blip),
        ("evaluate_blip passes alpha into pruner config", "evaluate_blip", '"alpha": args.atv_alpha', evaluate_blip),
        ("evaluate_blip aliases blipt5_atv_pruner", "evaluate_blip", 'args.pruning_method == "blipt5_atv_pruner"', evaluate_blip),
        ("atv alias forces uniform sparsity", "evaluate_blip", "args.sparsity_ratio_granularity = None", evaluate_blip),
        ("run script forwards ATV alpha", "run_atv", '--atv_alpha "$ATV_ALPHA"', run_atv),
        ("run script disables ViT pruning for T5-only ATV", "run_atv", "--no_prune_vit", run_atv),
        ("validator uses dense-base mask inference", "validator", "dense_base_nonzero_to_zero", validator),
        ("validator masks only dense-nonzero eligible weights", "validator", "eligible = base[key] != 0", validator),
        ("full verify requires mask base checkpoint", "run_atv_full_verify", "MASK_BASE_CKPT is required", run_full_verify),
        ("preflight checks mask-base export summary", "preflight", "check_mask_base_summary", preflight),
    ]
    return [make_evidence(name, source, fragment, path, files[path]) for name, source, fragment, path in checks]


def rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def write_static_mapping(out_dir: Path, evidence: Sequence[Evidence], lavis_root: Path) -> None:
    rows = [
        (
            "ATV visual token",
            "Qwen-VL image tokens",
            "BLIP2-T5 Q-Former query tokens after t5_proj",
            "BLIP2 raw ViT patch tokens do not directly enter the T5 encoder.",
        ),
        (
            "ATV text token",
            "Language token activations",
            "T5 text embeddings concatenated after query tokens",
            "temp_label marks query tokens as True and text tokens as False.",
        ),
        (
            "ATV salience",
            "1 - cosine_similarity(block input, block output)",
            "Same formula over query-token positions in each T5 encoder block",
            "Computed before WrappedATV accumulates scaler_row.",
        ),
        (
            "ATV keep rule",
            "All text tokens plus top-k visual tokens",
            "All text tokens plus top-k query tokens",
            "k is controlled by alpha and clamped by the number of query tokens.",
        ),
        (
            "Wanda metric",
            "abs(W) * sqrt(scaler_row)",
            "Same layer-wise metric after ATV-filtered activation accumulation",
            "Mask generation remains Wanda-style unstructured pruning.",
        ),
        (
            "Uniform sparsity",
            "ATV paper evaluates uniform sparsity for fairness",
            "blipt5_atv_pruner alias sets sparsity_ratio_granularity=None",
            "TAMP/AMIA DAS allocation is not part of ATV.",
        ),
    ]

    md = out_dir / "static_mapping.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# ATV Migration Static Mapping\n\n")
        f.write("## Semantic Mapping\n\n")
        f.write("| Concept | Original ATV | BLIP2-T5 Migration | Validation Note |\n")
        f.write("|---|---|---|---|\n")
        for row in rows:
            f.write("| %s | %s | %s | %s |\n" % row)

        f.write("\n## Source Evidence\n\n")
        f.write("| Status | Check | Source | File:Line | Fragment |\n")
        f.write("|---|---|---|---|---|\n")
        for e in evidence:
            status = "PASS" if e.ok else "FAIL"
            location = "%s:%s" % (rel(e.path, lavis_root.parent), e.line if e.line is not None else "-")
            fragment = e.fragment.replace("|", "\\|")
            f.write("| %s | %s | %s | `%s` | `%s` |\n" % (status, e.name, e.source, location, fragment))


def write_query_token_mapping(out_dir: Path, lavis_root: Path) -> List[Dict[str, object]]:
    blip2_t5 = lavis_root / "lavis" / "models" / "blip2_models" / "blip2_t5.py"
    blip2_base = lavis_root / "lavis" / "models" / "blip2_models" / "blip2.py"
    flant5xl_cfg = lavis_root / "lavis" / "configs" / "models" / "blip2" / "blip2_pretrain_flant5xl.yaml"

    rows: List[Dict[str, object]] = []

    def add_row(
        check: str,
        path: Path,
        pattern: str,
        expected: str,
        note: str,
        exact_fragment: Optional[str] = None,
    ) -> None:
        text = read_text(path) if path.exists() else ""
        observed = ""
        line: Optional[int] = None
        if exact_fragment is not None:
            line = find_line(text, exact_fragment)
            observed = exact_fragment if line is not None else ""
            ok = line is not None
        else:
            match = re.search(pattern, text)
            ok = bool(match and match.group(1) == expected)
            if match:
                observed = match.group(1)
                line = text[: match.start()].count("\n") + 1

        rows.append(
            {
                "check": check,
                "path": str(path),
                "line": line if line is not None else "",
                "expected": expected,
                "observed": observed,
                "status": "PASS" if ok else "FAIL",
                "note": note,
            }
        )

    add_row(
        "Blip2T5 constructor default query-token count",
        blip2_t5,
        r"num_query_token\s*=\s*(\d+)",
        "32",
        "The model default must expose 32 Q-Former query tokens.",
    )
    add_row(
        "pretrain_flant5xl config query-token count",
        flant5xl_cfg,
        r"num_query_token\s*:\s*(\d+)",
        "32",
        "The target BLIP2-T5 model_type must configure 32 query tokens.",
    )
    add_row(
        "Q-Former query token parameter allocation",
        blip2_base,
        "",
        "torch.zeros(1, num_query_token, encoder_config.hidden_size)",
        "Query-token count must flow into the Q-Former learned query parameter.",
        exact_fragment="torch.zeros(1, num_query_token, encoder_config.hidden_size)",
    )
    add_row(
        "Q-Former query length config",
        blip2_base,
        "",
        "encoder_config.query_length = num_query_token",
        "The Q-Former BERT config must receive the same query length.",
        exact_fragment="encoder_config.query_length = num_query_token",
    )

    write_csv(
        out_dir / "query_token_mapping.csv",
        ["check", "path", "line", "expected", "observed", "status", "note"],
        rows,
    )
    return rows


def atv_select_indices(
    cos_dist_by_sample: Sequence[Sequence[float]],
    num_text_by_sample: Sequence[int],
    alpha: float,
) -> Tuple[List[List[int]], List[int], float]:
    flat = [x for sample in cos_dist_by_sample for x in sample]
    cos_dist_avg = sum(flat) / len(flat) if flat else 0.0
    selected: List[List[int]] = []
    ks: List[int] = []
    for dist, num_text in zip(cos_dist_by_sample, num_text_by_sample):
        num_img = len(dist)
        k = int(round(min(1.0, alpha * cos_dist_avg) * num_text))
        k = max(0, min(num_img, k))
        order = sorted(range(num_img), key=lambda i: dist[i], reverse=True)
        selected.append(sorted(order[:k]))
        ks.append(k)
    return selected, ks, cos_dist_avg


def scaler_row_reference(
    inp: List[List[float]],
    image_mask: List[bool],
    selected_idxs: Sequence[int],
    attention_mask: Optional[Sequence[bool]] = None,
) -> List[float]:
    if attention_mask is None:
        kept = [row for row, is_img in zip(inp, image_mask) if not is_img]
    else:
        kept = [
            row
            for row, is_img, is_valid in zip(inp, image_mask, attention_mask)
            if (not is_img) and is_valid
        ]
    img_rows = [row for row, is_img in zip(inp, image_mask) if is_img]
    kept.extend(img_rows[i] for i in selected_idxs if i < len(img_rows))
    if not kept:
        return [0.0 for _ in inp[0]]
    cols = len(kept[0])
    out = []
    for c in range(cols):
        out.append(sum(row[c] * row[c] for row in kept))
    return out


def scaler_row_all_tokens(inp: List[List[float]]) -> List[float]:
    if not inp:
        return []
    cols = len(inp[0])
    return [sum(row[c] * row[c] for row in inp) for c in range(cols)]


def scaler_row_batch_reference(
    samples: Sequence[Tuple[List[List[float]], List[bool], Sequence[int]]]
) -> List[float]:
    if not samples:
        return []
    total = [0.0 for _ in samples[0][0][0]]
    for inp, image_mask, selected_idxs in samples:
        row = scaler_row_reference(inp, image_mask, selected_idxs)
        total = [a + b for a, b in zip(total, row)]
    return [x / len(samples) for x in total]


def wanda_metric(weight: List[List[float]], scaler: Sequence[float]) -> List[List[float]]:
    return [[abs(w) * math.sqrt(max(0.0, scaler[c])) for c, w in enumerate(row)] for row in weight]


def unstructured_wanda_mask(metric: List[List[float]], sparsity: float) -> List[List[bool]]:
    mask: List[List[bool]] = []
    for row in metric:
        prune_n = int(len(row) * sparsity)
        order = sorted(range(len(row)), key=lambda i: (row[i], i))
        pruned = set(order[:prune_n])
        mask.append([i in pruned for i in range(len(row))])
    return mask


def select_indices_with_quantized_scores(
    cos_dist: Sequence[float],
    k: int,
    decimals: int,
) -> List[int]:
    scored = [(round(float(v), decimals), i) for i, v in enumerate(cos_dist)]
    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    return sorted(i for _, i in scored[:k])


def run_golden_tests() -> List[Tuple[str, bool, str]]:
    tests: List[Tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        tests.append((name, bool(condition), detail))

    selected, ks, avg = atv_select_indices([[0.0, 0.0, 0.0]], [6], alpha=1.0)
    check("k_zero_when_cosdist_zero", selected == [[]] and ks == [0] and avg == 0.0, f"selected={selected}, ks={ks}, avg={avg}")

    selected, ks, avg = atv_select_indices([[0.30, 0.10, 0.20]], [6], alpha=1.0)
    check("middle_k_top_query", selected == [[0]] and ks == [1] and math.isclose(avg, 0.2), f"selected={selected}, ks={ks}, avg={avg}")

    selected, ks, avg = atv_select_indices([[0.30, 0.10, 0.20]], [6], alpha=10.0)
    check("k_clamped_to_num_img", selected == [[0, 1, 2]] and ks == [3], f"selected={selected}, ks={ks}, avg={avg}")

    selected, ks, avg = atv_select_indices([[0.20, 0.40, 0.10], [0.50, 0.10, 0.30]], [4, 8], alpha=1.0)
    check("variable_text_lengths", selected == [[1], [0, 2]] and ks == [1, 2], f"selected={selected}, ks={ks}, avg={avg}")

    selected, ks, avg = atv_select_indices([[0.50, 0.40, 0.30, 0.20]], [2], alpha=1.0)
    check(
        "valid_text_count_controls_k_not_padding",
        selected == [[0]] and ks == [1],
        f"selected={selected}, ks={ks}, avg={avg}; padded text count would over-select query tokens",
    )

    selected, ks, avg = atv_select_indices([[0.30, 0.0, 0.0], [0.0, 0.30, 0.0]], [6, 6], alpha=1.0)
    check(
        "batched_samples_select_queries_independently",
        selected == [[0], [1]] and ks == [1, 1],
        f"selected={selected}, ks={ks}, avg={avg}",
    )

    inp = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
    ]
    image_mask = [True, False, True, False]
    scaler = scaler_row_reference(inp, image_mask, selected_idxs=[0])
    expected = [
        3.0**2 + 7.0**2 + 1.0**2,
        4.0**2 + 8.0**2 + 2.0**2,
    ]
    check(
        "wrapped_atv_scaler_reference",
        all(math.isclose(a, b, rel_tol=1e-7) for a, b in zip(scaler, expected)),
        f"scaler={scaler}, expected={expected}",
    )

    batch_scaler = scaler_row_batch_reference(
        [
            (inp, image_mask, [0]),
            ([[2.0, 0.0], [0.0, 3.0], [4.0, 0.0], [0.0, 5.0]], image_mask, [1]),
        ]
    )
    batch_expected = [
        (expected[0] + (0.0**2 + 0.0**2 + 4.0**2)) / 2.0,
        (expected[1] + (3.0**2 + 5.0**2 + 0.0**2)) / 2.0,
    ]
    check(
        "wrapped_atv_batch_scaler_uses_sample_count",
        all(math.isclose(a, b, rel_tol=1e-7) for a, b in zip(batch_scaler, batch_expected)),
        f"batch_scaler={batch_scaler}, expected={batch_expected}",
    )

    padded_inp = [
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 3.0],
        [100.0, 100.0],
        [200.0, 200.0],
    ]
    padded_image_mask = [True, True, False, False, False]
    padded_attention_mask = [True, True, True, False, False]
    padded_scaler = scaler_row_reference(
        padded_inp,
        padded_image_mask,
        selected_idxs=[1],
        attention_mask=padded_attention_mask,
    )
    padded_expected = [2.0**2 + 0.0**2, 0.0**2 + 3.0**2]
    unmasked_scaler = scaler_row_reference(padded_inp, padded_image_mask, selected_idxs=[1])
    check(
        "wrapped_atv_scaler_ignores_padding_tokens",
        (
            all(math.isclose(a, b, rel_tol=1e-7) for a, b in zip(padded_scaler, padded_expected))
            and padded_scaler != unmasked_scaler
        ),
        f"padded_scaler={padded_scaler}, expected={padded_expected}, unmasked={unmasked_scaler}",
    )

    weight = [[1.0, 1.0, 1.0], [1.5, 0.8, 1.2]]
    inp_mask_case = [
        [0.1, 20.0, 2.0],
        [10.0, 1.0, 2.0],
        [0.1, 20.0, 2.0],
        [10.0, 1.0, 2.0],
    ]
    mask_case = [True, False, True, False]
    naive_mask = unstructured_wanda_mask(wanda_metric(weight, scaler_row_all_tokens(inp_mask_case)), sparsity=1.0 / 3.0)
    atv_all_mask = unstructured_wanda_mask(
        wanda_metric(weight, scaler_row_reference(inp_mask_case, mask_case, selected_idxs=[0, 1])),
        sparsity=1.0 / 3.0,
    )
    atv_text_only_mask = unstructured_wanda_mask(
        wanda_metric(weight, scaler_row_reference(inp_mask_case, mask_case, selected_idxs=[])),
        sparsity=1.0 / 3.0,
    )
    check(
        "atv_k_num_img_matches_naive_wanda_mask",
        atv_all_mask == naive_mask,
        f"atv_all={atv_all_mask}, naive={naive_mask}",
    )
    check(
        "atv_k_zero_differs_from_naive_wanda_mask",
        atv_text_only_mask != naive_mask,
        f"atv_text_only={atv_text_only_mask}, naive={naive_mask}",
    )

    cos_dist_fp32 = [0.10001, 0.40502, 0.30303, 0.20104]
    selected_fp32 = select_indices_with_quantized_scores(cos_dist_fp32, k=2, decimals=6)
    selected_low_precision = select_indices_with_quantized_scores(cos_dist_fp32, k=2, decimals=3)
    check(
        "selected_indices_stable_under_low_precision_rounding",
        selected_fp32 == selected_low_precision == [1, 2],
        f"fp32={selected_fp32}, low_precision={selected_low_precision}",
    )

    return tests


ATV_LOG_RE = re.compile(
    r"\[ATV\].*?layer\s+(?P<layer>\d+):\s+"
    r"cos_dist_avg=(?P<cos>[-+0-9.eE]+)\s+"
    r"k mean/min/max=(?P<kmean>[-+0-9.eE]+)/(?P<kmin>\d+)/(?P<kmax>\d+)\s+"
    r"(?:k==0=(?P<kzero>\d+)/(?P<kzero_n>\d+)\s+)?"
    r"num_img=(?P<num_img>\d+)\s+"
    r"k==num_img\(degenerate\)=(?P<deg>\d+)/(?P<n>\d+)"
)


def parse_atv_logs(log_specs: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for spec in log_specs:
        if "=" in spec:
            tag, path_s = spec.split("=", 1)
        else:
            path_s = spec
            tag = Path(path_s).stem
        path = Path(path_s)
        text = read_text(path)
        for m in ATV_LOG_RE.finditer(text):
            d = m.groupdict()
            d["tag"] = tag
            d["log"] = str(path)
            deg = int(d["deg"])
            n = int(d["n"])
            d["degenerate_rate"] = "%.6f" % (deg / n if n else 0.0)
            if d.get("kzero") not in (None, "") and d.get("kzero_n") not in (None, ""):
                kzero = int(d["kzero"])
                kzero_n = int(d["kzero_n"])
                d["k_zero_rate"] = "%.6f" % (kzero / kzero_n if kzero_n else 0.0)
            else:
                d["kzero"] = ""
                d["kzero_n"] = ""
                d["k_zero_rate"] = ""
            rows.append(d)
    return rows


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def csv_row_count(path: Path) -> int:
    return len(read_csv_rows(path))


def copy_or_empty_csv(
    out_path: Path,
    input_path: Optional[Path],
    default_header: Sequence[str],
    preserve_existing: bool = False,
) -> int:
    if input_path is not None:
        if not input_path.exists():
            if preserve_existing and out_path.exists():
                return csv_row_count(out_path)
            write_csv(out_path, default_header, [])
            return 0
        if input_path.resolve() != out_path.resolve():
            shutil.copyfile(input_path, out_path)
        with out_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            return max(0, sum(1 for _ in f) - 1)
    if preserve_existing and out_path.exists():
        return csv_row_count(out_path)
    write_csv(out_path, default_header, [])
    return 0


def parse_tagged_path(spec: str) -> Tuple[str, Path]:
    if "=" in spec:
        tag, path_s = spec.split("=", 1)
        tag = tag.strip()
    else:
        path_s = spec
        tag = Path(path_s).stem
    return tag, Path(path_s)


def eval_row(
    args: argparse.Namespace,
    benchmark: str,
    score: object,
    calibration: str = "",
    notes: str = "",
    method: str = "",
) -> Dict[str, object]:
    calibration = calibration or args.eval_calibration
    method = method or args.eval_method
    method_norm = normalize_method_name(method)
    t5_sparsity = args.eval_t5_sparsity
    vit_sparsity = args.eval_vit_sparsity
    if not t5_sparsity and method_norm == "dense":
        t5_sparsity = "0.0"
    if not vit_sparsity and method_norm == "dense":
        vit_sparsity = "0.0"
    if not t5_sparsity and method_norm in {"atv", "wanda", "tamp"}:
        t5_sparsity = str(args.expected_t5_sparsity)
    if not vit_sparsity and method_norm in {"atv", "wanda", "tamp"}:
        vit_sparsity = "0.0"
    return {
        "method": method,
        "calibration": calibration,
        "seed": args.eval_seed,
        "alpha": args.eval_alpha or infer_alpha_from_label(calibration) or ("1" if method_norm == "atv" else ""),
        "t5_sparsity": t5_sparsity,
        "vit_sparsity": vit_sparsity,
        "benchmark": benchmark,
        "score": score,
        "std": "",
        "notes": notes,
    }


def infer_method_from_label(label: str, default: str) -> str:
    s = label.casefold()
    if "dense" in s or "fullprec" in s or "full_precision" in s:
        return "dense"
    if "atv" in s:
        return "atv"
    if "tamp" in s or "amia" in s:
        return "tamp"
    if "wanda" in s or "naive" in s:
        return "wanda"
    return default


def infer_alpha_from_label(label: str) -> str:
    s = label.casefold()
    m = re.search(r"alpha[_-]?([-+]?\d+(?:[p.]\d+)?)", s)
    if not m:
        return ""
    return m.group(1).replace("p", ".")


def parse_metrics_jsonl(paths: Sequence[Path], args: argparse.Namespace) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                line = line.strip().lstrip("\ufeff")
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                benchmark = str(rec.get("benchmark") or "")
                if not benchmark:
                    continue
                value = rec.get("value", rec.get("score", rec.get("agg_metrics", "")))
                calibration = str(rec.get("calib_tag") or args.eval_calibration)
                method = str(rec.get("method") or infer_method_from_label(calibration, args.eval_method))
                note_bits = []
                for key in ("metric", "split", "n"):
                    if rec.get(key) not in (None, ""):
                        note_bits.append("%s=%s" % (key, rec[key]))
                note_bits.append("source=%s" % path)
                rows.append(eval_row(args, benchmark, value, calibration, "; ".join(note_bits), method=method))
    return rows


def last_agg_metrics(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    last: Optional[float] = None
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.strip().lstrip("\ufeff")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and rec.get("agg_metrics") not in (None, ""):
                try:
                    last = float(rec["agg_metrics"])
                except (TypeError, ValueError):
                    continue
    return last


def parse_okvqa_eval_txt(specs: Sequence[str], args: argparse.Namespace) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for spec in specs:
        calibration, path = parse_tagged_path(spec)
        value = last_agg_metrics(path)
        if value is None:
            continue
        rows.append(
            eval_row(
                args,
                "OKVQA",
                value,
                calibration=calibration,
                notes="metric=agg_metrics; source=%s" % path,
                method=infer_method_from_label(calibration, args.eval_method),
            )
        )
    return rows


def merge_eval_rows(
    existing: Sequence[Dict[str, object]],
    incoming: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    merged: Dict[Tuple[str, str, str, str, str], Dict[str, object]] = {}

    def key(row: Dict[str, object]) -> Tuple[str, str, str, str, str]:
        return (
            normalize_method_name(str(row.get("method", ""))),
            str(row.get("calibration", "")).strip(),
            str(row.get("seed", "")).strip(),
            str(row.get("alpha", "")).strip(),
            normalize_benchmark_name(str(row.get("benchmark", ""))),
        )

    for row in existing:
        merged[key(row)] = dict(row)
    for row in incoming:
        merged[key(row)] = dict(row)
    return list(merged.values())


def write_eval_results(out_path: Path, args: argparse.Namespace, header: Sequence[str]) -> int:
    if args.eval_csv is not None:
        incoming = read_csv_rows(args.eval_csv)
        if args.preserve_existing and out_path.exists():
            rows = merge_eval_rows(read_csv_rows(out_path), incoming)
            write_csv(out_path, header, rows)
            return len(rows)
        return copy_or_empty_csv(out_path, args.eval_csv, header, preserve_existing=args.preserve_existing)
    rows = parse_metrics_jsonl(args.metrics_jsonl, args)
    rows.extend(parse_okvqa_eval_txt(args.okvqa_eval_txt, args))
    if rows and args.preserve_existing and out_path.exists():
        rows = merge_eval_rows(read_csv_rows(out_path), rows)
    if not rows and args.preserve_existing and out_path.exists():
        return csv_row_count(out_path)
    write_csv(out_path, header, rows)
    return len(rows)


def write_eval_summary(path: Path, rows: Sequence[Dict[str, str]]) -> int:
    grouped: Dict[Tuple[str, str, str], List[float]] = {}
    for row in rows:
        if not score_is_present(row.get("score", "")):
            continue
        try:
            score = float(str(row.get("score", "")).rstrip("%"))
        except ValueError:
            continue
        method = normalize_method_name(str(row.get("method", "")))
        alpha = str(row.get("alpha", "")).strip()
        benchmark = normalize_benchmark_name(str(row.get("benchmark", "")))
        if not method or not benchmark:
            continue
        grouped.setdefault((method, alpha, benchmark), []).append(score)

    summary_rows: List[Dict[str, object]] = []
    for (method, alpha, benchmark), vals in sorted(grouped.items()):
        n = len(vals)
        mean = sum(vals) / n
        if n > 1:
            var = sum((v - mean) ** 2 for v in vals) / (n - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        summary_rows.append(
            {
                "method": method,
                "alpha": alpha,
                "benchmark": benchmark,
                "n": n,
                "mean": "%.6f" % mean,
                "std": "%.6f" % std,
            }
        )
    write_csv(path, ["method", "alpha", "benchmark", "n", "mean", "std"], summary_rows)
    return len(summary_rows)


def merge_eval_provenance_rows(
    existing: Sequence[Dict[str, str]],
    incoming: Sequence[Dict[str, str]],
) -> List[Dict[str, str]]:
    merged: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}

    def key(row: Dict[str, str]) -> Tuple[str, str, str, str]:
        method = normalize_method_name(str(row.get("method", "")))
        calibration = str(row.get("calibration", "")).strip()
        seed = str(row.get("seed", "")).strip()
        alpha = str(row.get("alpha", "")).strip()
        if method == "atv" and not alpha:
            alpha = infer_alpha_from_label(calibration) or "1"
        return (method, calibration, seed, alpha)

    for row in existing:
        merged[key(row)] = dict(row)
    for row in incoming:
        merged[key(row)] = dict(row)
    return list(merged.values())


def write_eval_provenance(
    out_path: Path,
    paths: Sequence[Path],
    preserve_existing: bool,
) -> int:
    header = ["seed", "method", "calibration", "alpha", "ckpt", "metrics_jsonl", "okvqa_eval_txt"]
    incoming: List[Dict[str, str]] = []
    for path in paths:
        if path.exists():
            incoming.extend(read_csv_rows(path))

    if incoming and preserve_existing and out_path.exists():
        rows = merge_eval_provenance_rows(read_csv_rows(out_path), incoming)
    elif incoming:
        rows = incoming
    elif preserve_existing and out_path.exists():
        return csv_row_count(out_path)
    else:
        rows = []
    write_csv(out_path, header, rows)
    return len(rows)


def prediction_score(row: Dict[str, str]) -> Optional[float]:
    for key in ("correct", "is_correct", "score", "value", "accuracy"):
        value = row.get(key, "")
        if value in (None, ""):
            continue
        s = str(value).strip().casefold()
        if s in {"true", "yes", "y", "correct"}:
            return 1.0
        if s in {"false", "no", "n", "incorrect", "wrong"}:
            return 0.0
        try:
            return float(s.rstrip("%")) / (100.0 if s.endswith("%") else 1.0)
        except ValueError:
            continue
    return None


def read_prediction_rows(specs: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for spec in specs:
        tag, path = parse_tagged_path(spec)
        if not path.exists():
            continue
        default_method = infer_method_from_label(tag, "")
        default_benchmark = normalize_benchmark_name(tag)
        default_alpha = infer_alpha_from_label(tag) or ("1" if default_method == "atv" else "")
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                sample_id = str(
                    raw.get("sample_id")
                    or raw.get("question_id")
                    or raw.get("id")
                    or raw.get("index")
                    or ""
                ).strip()
                if not sample_id:
                    continue
                score = prediction_score(raw)
                if score is None:
                    continue
                benchmark = normalize_benchmark_name(
                    str(raw.get("benchmark") or raw.get("dataset") or default_benchmark)
                )
                method = normalize_method_name(str(raw.get("method") or default_method))
                alpha = str(raw.get("alpha") or default_alpha).strip()
                seed = str(raw.get("seed") or "").strip()
                if not benchmark or not method:
                    continue
                rows.append(
                    {
                        "method": method,
                        "alpha": alpha,
                        "benchmark": benchmark,
                        "seed": seed,
                        "sample_id": sample_id,
                        "score": float(score),
                        "source": str(path),
                    }
                )
    return rows


def method_label(row: Dict[str, object]) -> str:
    method = normalize_method_name(str(row.get("method", "")))
    alpha = str(row.get("alpha", "")).strip()
    if method == "atv" and alpha:
        return "atv_alpha%s" % alpha.replace(".", "p")
    return method


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def write_paired_bootstrap_ci(
    path: Path,
    prediction_rows: Sequence[Dict[str, object]],
    num_bootstrap: int,
    seed: int,
) -> int:
    grouped: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for row in prediction_rows:
        benchmark = str(row.get("benchmark", "")).strip()
        eval_seed = str(row.get("seed", "")).strip()
        label = method_label(row)
        sample_id = str(row.get("sample_id", "")).strip()
        if not benchmark or not label or not sample_id:
            continue
        grouped.setdefault((benchmark, eval_seed, label), {})[sample_id] = float(row.get("score", 0.0))

    out_rows: List[Dict[str, object]] = []
    rng = random.Random(seed)
    labels_by_group: Dict[Tuple[str, str], List[str]] = {}
    for benchmark, eval_seed, label in grouped:
        labels_by_group.setdefault((benchmark, eval_seed), []).append(label)

    for (benchmark, eval_seed), labels in sorted(labels_by_group.items()):
        labels = sorted(set(labels))
        for i, a in enumerate(labels):
            for b in labels[i + 1 :]:
                scores_a = grouped.get((benchmark, eval_seed, a), {})
                scores_b = grouped.get((benchmark, eval_seed, b), {})
                common = sorted(set(scores_a) & set(scores_b))
                n = len(common)
                if n == 0:
                    continue
                diffs = [scores_a[sid] - scores_b[sid] for sid in common]
                observed = sum(diffs) / n
                boot = []
                if num_bootstrap > 0:
                    for _ in range(num_bootstrap):
                        boot.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
                ci_low = percentile(boot, 0.025) if boot else observed
                ci_high = percentile(boot, 0.975) if boot else observed
                out_rows.append(
                    {
                        "benchmark": benchmark,
                        "seed": eval_seed,
                        "method_a": a,
                        "method_b": b,
                        "n": n,
                        "mean_delta_a_minus_b": "%.6f" % observed,
                        "ci_low": "%.6f" % ci_low,
                        "ci_high": "%.6f" % ci_high,
                        "bootstrap_samples": num_bootstrap,
                    }
                )
    write_csv(
        path,
        [
            "benchmark",
            "seed",
            "method_a",
            "method_b",
            "n",
            "mean_delta_a_minus_b",
            "ci_low",
            "ci_high",
            "bootstrap_samples",
        ],
        out_rows,
    )
    return len(out_rows)


def try_import_pyplot():
    try:
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def write_minimal_png(path: Path, message: str) -> None:
    # Minimal valid 1x1 transparent PNG. The sidecar text explains why it is a
    # placeholder, so the file is never mistaken for real evidence.
    path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000a49444154789c636000000200015fe221bc0000000049454e44ae426082"
        )
    )
    path.with_suffix(path.suffix + ".txt").write_text(message + "\n", encoding="utf-8")


def write_placeholder_plot(path: Path, title: str, message: str) -> None:
    write_minimal_png(path, "%s\n\n%s" % (title, message))


def plot_importance_distribution(path: Path, csv_path: Optional[Path]) -> bool:
    if csv_path is None or not csv_path.exists():
        write_placeholder_plot(
            path,
            "ATV Importance Distribution",
            "GPU importance evidence was not provided.\n"
            "Pass --importance_csv after pruning to replace this placeholder.",
        )
        return False
    plt = try_import_pyplot()
    if plt is None:
        write_placeholder_plot(path, "ATV Importance Distribution", "matplotlib is unavailable.")
        return False

    values: List[float] = []
    labels: List[str] = []
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = None
            for key in ("importance", "wanda_importance", "mean_wanda_importance", "w_metric", "value"):
                if key in row and row[key] not in (None, ""):
                    value = row[key]
                    break
            if value is None:
                continue
            try:
                values.append(float(value))
                labels.append(row.get("method", row.get("tag", "importance")))
            except ValueError:
                continue

    if not values:
        write_placeholder_plot(path, "ATV Importance Distribution", "No numeric importance values found in --importance_csv.")
        return False

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(values, bins=min(80, max(10, int(len(values) ** 0.5))), color="#5B8FF9", alpha=0.78)
    ax.set_title("ATV Importance Score Distribution")
    ax.set_xlabel("importance score")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def plot_scaler_distribution(path: Path, csv_path: Optional[Path]) -> bool:
    if csv_path is None or not csv_path.exists():
        write_placeholder_plot(
            path,
            "ATV Scaler Row Distribution",
            "GPU scaler_row evidence was not provided.\n"
            "Pass --importance_csv after pruning to replace this placeholder.",
        )
        return False
    plt = try_import_pyplot()
    if plt is None:
        write_placeholder_plot(path, "ATV Scaler Row Distribution", "matplotlib is unavailable.")
        return False

    means: List[float] = []
    maxes: List[float] = []
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, bucket in (("scaler_row_mean", means), ("scaler_row_max", maxes)):
                value = row.get(key, "")
                if value in (None, ""):
                    continue
                try:
                    bucket.append(float(value))
                except ValueError:
                    continue

    if not means and not maxes:
        write_placeholder_plot(
            path,
            "ATV Scaler Row Distribution",
            "No numeric scaler_row_mean/scaler_row_max values found in --importance_csv.",
        )
        return False

    fig, ax = plt.subplots(figsize=(8, 4.8))
    if means:
        ax.hist(means, bins=min(80, max(10, int(len(means) ** 0.5))), color="#5AD8A6", alpha=0.65, label="scaler_row_mean")
    if maxes:
        ax.hist(maxes, bins=min(80, max(10, int(len(maxes) ** 0.5))), color="#F6BD16", alpha=0.45, label="scaler_row_max")
    ax.set_title("ATV scaler_row Distribution")
    ax.set_xlabel("scaler_row statistic")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def plot_selected_query_frequency(path: Path, csv_path: Optional[Path]) -> bool:
    if csv_path is None or not csv_path.exists():
        write_placeholder_plot(
            path,
            "ATV Selected Query Token Frequency",
            "Selected query-token indices were not provided.\n"
            "Pass --selected_query_csv after GPU pruning to replace this placeholder.",
        )
        return False
    plt = try_import_pyplot()
    if plt is None:
        write_placeholder_plot(path, "ATV Selected Query Token Frequency", "matplotlib is unavailable.")
        return False

    counts: Dict[int, float] = {}
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx_s = row.get("query_index", row.get("selected_query_index", row.get("index", "")))
            count_s = row.get("count", row.get("frequency", row.get("value", "1")))
            try:
                idx = int(idx_s)
                count = float(count_s)
            except ValueError:
                continue
            counts[idx] = counts.get(idx, 0.0) + count

    if not counts:
        write_placeholder_plot(path, "ATV Selected Query Token Frequency", "No query_index/count values found in --selected_query_csv.")
        return False

    xs = sorted(counts)
    ys = [counts[x] for x in xs]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(xs, ys, color="#61DDAA", edgecolor="#2F7F66")
    ax.set_title("ATV Selected Query Token Frequency")
    ax.set_xlabel("BLIP2 Q-Former query token index")
    ax.set_ylabel("selection count")
    ax.set_xticks(xs if len(xs) <= 32 else xs[:: max(1, len(xs) // 32)])
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def load_state_dict(path: Path):
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("torch is required for --mask_pair checkpoint comparison") from exc

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "module"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def layer_from_key(key: str) -> str:
    m = re.search(r"(?:encoder|decoder)\.block\.(\d+)", key)
    if m:
        return m.group(1)
    m = re.search(r"visual_encoder\.blocks\.(\d+)", key)
    if m:
        return m.group(1)
    return "unknown"


def ratio(num: int, den: int, default: float = 0.0) -> float:
    return num / den if den else default


def compare_mask_pair(
    name: str,
    a_path: Path,
    b_path: Path,
    prefix: str,
    min_numel: int,
    base_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("torch is required for --mask_pair checkpoint comparison") from exc

    sa = load_state_dict(a_path)
    sb = load_state_dict(b_path)
    base = load_state_dict(base_path) if base_path else None
    base_ckpt_s = str(base_path) if base_path else ""
    mask_inference = "dense_base_nonzero_to_zero" if base_path else "zero_weight_only"
    acc: Dict[str, Dict[str, int]] = {}
    module_rows: List[Dict[str, object]] = []
    for key, wa in sa.items():
        if key not in sb or not torch.is_tensor(wa):
            continue
        wb = sb[key]
        if not torch.is_tensor(wb) or wa.shape != wb.shape or wa.dim() < 2 or wa.numel() < min_numel:
            continue
        if prefix and not key.startswith(prefix):
            continue
        if base is not None:
            if key not in base or not torch.is_tensor(base[key]) or base[key].shape != wa.shape:
                continue
            eligible = base[key] != 0
            eligible_count = int(eligible.sum().item())
            if eligible_count == 0:
                continue
        else:
            eligible = torch.ones_like(wa, dtype=torch.bool)
            eligible_count = int(wa.numel())
        layer = layer_from_key(key)
        slot = acc.setdefault(layer, {"prune_inter": 0, "prune_union": 0, "keep_inter": 0, "keep_union": 0, "numel": 0, "tensors": 0})
        ma = (wa == 0) & eligible
        mb = (wb == 0) & eligible
        ka = (~ma) & eligible
        kb = (~mb) & eligible
        numel = eligible_count
        a_pruned = int(ma.sum().item())
        b_pruned = int(mb.sum().item())
        diff = int(((ma != mb) & eligible).sum().item())
        prune_inter = int((ma & mb).sum().item())
        prune_union = int((ma | mb).sum().item())
        keep_inter = int((ka & kb).sum().item())
        keep_union = int((ka | kb).sum().item())
        slot["prune_inter"] += int((ma & mb).sum().item())
        slot["prune_union"] += int((ma | mb).sum().item())
        slot["keep_inter"] += int((ka & kb).sum().item())
        slot["keep_union"] += int((ka | kb).sum().item())
        slot["a_pruned"] = slot.get("a_pruned", 0) + a_pruned
        slot["b_pruned"] = slot.get("b_pruned", 0) + b_pruned
        slot["diff"] = slot.get("diff", 0) + diff
        slot["numel"] += numel
        slot["tensors"] += 1

        module_rows.append(
            {
                "pair": name,
                "a_ckpt": str(a_path),
                "b_ckpt": str(b_path),
                "base_ckpt": base_ckpt_s,
                "mask_inference": mask_inference,
                "mask_prefix": prefix,
                "min_numel": min_numel,
                "module": key,
                "layer": layer,
                "numel": numel,
                "prune_iou": ratio(prune_inter, prune_union, default=1.0),
                "keep_iou": ratio(keep_inter, keep_union, default=1.0),
                "a_prune_ratio": ratio(a_pruned, numel),
                "b_prune_ratio": ratio(b_pruned, numel),
                "a_keep_ratio": 1.0 - ratio(a_pruned, numel),
                "b_keep_ratio": 1.0 - ratio(b_pruned, numel),
                "diff_ratio": ratio(diff, numel),
            }
        )

    layer_rows = []
    for layer, s in sorted(acc.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 10**9):
        numel = s["numel"]
        a_pruned = s.get("a_pruned", 0)
        b_pruned = s.get("b_pruned", 0)
        layer_rows.append(
            {
                "pair": name,
                "a_ckpt": str(a_path),
                "b_ckpt": str(b_path),
                "base_ckpt": base_ckpt_s,
                "mask_inference": mask_inference,
                "mask_prefix": prefix,
                "min_numel": min_numel,
                "layer": layer,
                "tensors": s["tensors"],
                "numel": numel,
                "prune_iou": ratio(s["prune_inter"], s["prune_union"], default=1.0),
                "keep_iou": ratio(s["keep_inter"], s["keep_union"], default=1.0),
                "a_prune_ratio": ratio(a_pruned, numel),
                "b_prune_ratio": ratio(b_pruned, numel),
                "a_keep_ratio": 1.0 - ratio(a_pruned, numel),
                "b_keep_ratio": 1.0 - ratio(b_pruned, numel),
                "diff_ratio": ratio(s.get("diff", 0), numel),
            }
        )
    module_rows.sort(key=lambda r: (int(r["layer"]) if str(r["layer"]).isdigit() else 10**9, str(r["module"])))
    return layer_rows, module_rows


def write_unit_results(out_dir: Path, tests: Sequence[Tuple[str, bool, str]]) -> None:
    path = out_dir / "unit_test_results.txt"
    with path.open("w", encoding="utf-8") as f:
        for name, ok, detail in tests:
            f.write("%s %s: %s\n" % ("PASS" if ok else "FAIL", name, detail))


def alpha_from_tag(tag: str) -> Optional[float]:
    m = re.search(r"alpha\s*([-+0-9.]+)", tag)
    if not m:
        m = re.search(r"alpha([-+0-9.]+)", tag)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def summarize_alpha_logs(log_rows: Sequence[Dict[str, str]]) -> Tuple[List[Dict[str, object]], Optional[bool]]:
    grouped: Dict[str, Dict[str, float]] = {}
    for row in log_rows:
        tag = row.get("tag", "")
        if not tag:
            continue
        slot = grouped.setdefault(
            tag,
            {
                "layers": 0.0,
                "kmean_sum": 0.0,
                "deg_sum": 0.0,
                "k_zero_sum": 0.0,
                "k_zero_layers": 0.0,
                "alpha": float("nan"),
            },
        )
        try:
            slot["kmean_sum"] += float(row.get("kmean", 0.0))
            slot["deg_sum"] += float(row.get("degenerate_rate", 0.0))
            slot["layers"] += 1.0
            k_zero_rate = row_float(row, "k_zero_rate")
            if k_zero_rate is not None and math.isfinite(k_zero_rate):
                slot["k_zero_sum"] += k_zero_rate
                slot["k_zero_layers"] += 1.0
        except ValueError:
            continue
        alpha = alpha_from_tag(tag)
        if alpha is not None:
            slot["alpha"] = alpha
    summaries: List[Dict[str, object]] = []
    for tag, slot in grouped.items():
        n = max(1.0, slot["layers"])
        summaries.append(
            {
                "tag": tag,
                "alpha": slot["alpha"],
                "layers": int(slot["layers"]),
                "mean_kmean": slot["kmean_sum"] / n,
                "mean_degenerate_rate": slot["deg_sum"] / n,
                "mean_k_zero_rate": (
                    slot["k_zero_sum"] / slot["k_zero_layers"]
                    if slot["k_zero_layers"] > 0
                    else float("nan")
                ),
            }
        )
    summaries.sort(key=lambda r: (math.inf if math.isnan(float(r["alpha"])) else float(r["alpha"]), str(r["tag"])))
    numeric = [r for r in summaries if not math.isnan(float(r["alpha"]))]
    monotonic: Optional[bool] = None
    if len(numeric) >= 2:
        monotonic = all(
            float(numeric[i]["mean_kmean"]) <= float(numeric[i + 1]["mean_kmean"]) + 1e-9
            for i in range(len(numeric) - 1)
        )
    return summaries, monotonic


def split_csv_arg(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def normalize_benchmark_name(name: str) -> str:
    s = name.strip().casefold().replace("-", "_")
    if s in {"mathvista", "mathvista_mc", "mathvista_multi_choice"}:
        return "mathvista"
    if s in {"okvqa", "ok_vqa"}:
        return "okvqa"
    if s in {"mmbench", "mmbench_dev"}:
        return "mmbench"
    if s == "mmmu":
        return "mmmu"
    return s


def normalize_method_name(name: str) -> str:
    s = name.strip().casefold()
    if s in {"naive", "naive_wanda", "wanda"}:
        return "wanda"
    if s in {"tamp", "amia"}:
        return "tamp"
    if s in {"fullprec", "full_precision", "dense"}:
        return "dense"
    return s


def row_float(row: Dict[str, object], key: str) -> Optional[float]:
    try:
        value = row.get(key, "")
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def seen_alphas(alpha_summary: Sequence[Dict[str, object]]) -> List[float]:
    vals = []
    for row in alpha_summary:
        alpha = row_float(row, "alpha")
        if alpha is not None and math.isfinite(alpha):
            vals.append(alpha)
    return vals


def alpha_coverage_ok(alpha_summary: Sequence[Dict[str, object]], required: Sequence[float]) -> Tuple[bool, str]:
    if not required:
        return True, "no required alpha grid configured"
    seen = seen_alphas(alpha_summary)
    missing = []
    for want in required:
        if not any(math.isclose(want, got, rel_tol=0.0, abs_tol=1e-6) for got in seen):
            missing.append(want)
    if missing:
        return False, "missing alpha values: %s" % ",".join("%.6g" % x for x in missing)
    return True, "covered alpha values: %s" % ",".join("%.6g" % x for x in required)


def alpha_layer_coverage_quality(
    alpha_summary: Sequence[Dict[str, object]],
    required: Sequence[float],
    min_layers: int,
) -> Tuple[bool, str]:
    if not required or min_layers <= 0:
        return True, "no per-alpha layer coverage requirement configured"
    problems: List[str] = []
    notes: List[str] = []
    for want in required:
        layer_counts: List[int] = []
        for row in alpha_summary:
            alpha = row_float(row, "alpha")
            if alpha is not None and math.isclose(want, alpha, rel_tol=0.0, abs_tol=1e-6):
                layers = row_float(row, "layers")
                if layers is not None:
                    layer_counts.append(int(layers))
        if not layer_counts:
            continue
        best = max(layer_counts)
        notes.append("alpha %.6g layers=%d" % (want, best))
        if best < min_layers:
            problems.append("alpha %.6g has %d layers, expected >=%d" % (want, best, min_layers))
    if problems:
        return False, "; ".join(problems[:8])
    return True, "; ".join(notes) if notes else "no alpha layer rows"


def alpha_num_img_quality(log_rows: Sequence[Dict[str, str]], expected_num_img: int) -> Tuple[bool, str]:
    if not log_rows:
        return False, "no alpha log rows"
    bad = []
    seen = set()
    for row in log_rows:
        tag = str(row.get("tag", ""))
        layer = str(row.get("layer", ""))
        try:
            num_img = int(float(row.get("num_img", "")))
        except ValueError:
            bad.append("%s/layer%s:nonnumeric" % (tag, layer))
            continue
        seen.add(num_img)
        if num_img != expected_num_img:
            bad.append("%s/layer%s:%d" % (tag, layer, num_img))
    if bad:
        return False, "expected num_img=%d but saw %s" % (expected_num_img, ",".join(bad[:8]))
    return True, "all alpha log rows have num_img=%d" % expected_num_img


def alpha_k_zero_quality(log_rows: Sequence[Dict[str, str]]) -> Tuple[bool, str]:
    if not log_rows:
        return False, "no alpha log rows"
    bad: List[str] = []
    rates: List[float] = []
    for row in log_rows:
        tag = str(row.get("tag", ""))
        layer = str(row.get("layer", ""))
        try:
            kzero = int(float(row.get("kzero", "")))
            kzero_n = int(float(row.get("kzero_n", "")))
            n = int(float(row.get("n", "")))
            rate = float(row.get("k_zero_rate", ""))
        except ValueError:
            bad.append("%s/layer%s:missing_or_nonnumeric" % (tag, layer))
            continue
        if kzero_n <= 0:
            bad.append("%s/layer%s:kzero_n=%d" % (tag, layer, kzero_n))
            continue
        if kzero < 0 or kzero > kzero_n:
            bad.append("%s/layer%s:kzero=%d/%d" % (tag, layer, kzero, kzero_n))
        if n != kzero_n:
            bad.append("%s/layer%s:kzero_n=%d differs from n=%d" % (tag, layer, kzero_n, n))
        expected = kzero / kzero_n
        if abs(rate - expected) > 1e-6:
            bad.append("%s/layer%s:k_zero_rate %.6f expected %.6f" % (tag, layer, rate, expected))
        rates.append(rate)
    if bad:
        return False, "; ".join(bad[:8])
    return True, "k==0 evidence rows=%d; mean k_zero_rate=%.6f" % (len(log_rows), sum(rates) / max(1, len(rates)))


def alpha_log_rows(log_rows: Sequence[Dict[str, str]], target_alpha: float) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for row in log_rows:
        alpha = alpha_from_tag(str(row.get("tag", "")))
        if alpha is not None and math.isclose(target_alpha, alpha, rel_tol=0.0, abs_tol=1e-6):
            rows.append(row)
    return rows


def alpha_zero_text_only_quality(log_rows: Sequence[Dict[str, str]]) -> Tuple[bool, str]:
    rows = alpha_log_rows(log_rows, 0.0)
    if not rows:
        return False, "no alpha=0 rows"

    bad: List[str] = []
    for row in rows:
        tag = str(row.get("tag", ""))
        layer = str(row.get("layer", ""))
        try:
            kmean = float(row.get("kmean", ""))
            kmin = int(float(row.get("kmin", "")))
            kmax = int(float(row.get("kmax", "")))
        except ValueError:
            bad.append("%s/layer%s:nonnumeric" % (tag, layer))
            continue
        if abs(kmean) > 1e-8 or kmin != 0 or kmax != 0:
            bad.append("%s/layer%s:kmean/kmin/kmax=%g/%d/%d" % (tag, layer, kmean, kmin, kmax))

    if bad:
        return False, "alpha=0 should select no query tokens but saw %s" % ",".join(bad[:8])
    return True, "alpha=0 rows=%d; all kmean/kmin/kmax are zero" % len(rows)


def token_mask_quality(
    rows: Sequence[Dict[str, str]],
    expected_samples: int,
    expected_layers: int,
) -> Tuple[bool, List[str], Dict[str, int]]:
    if not rows:
        return False, ["no token mask rows"], {"samples": 0, "layers": 0, "rows": 0}
    problems: List[str] = []
    samples = set()
    layers = set()
    required_prefix_fields = {
        "expected_query_tokens",
        "query_prefix_true_count",
        "text_suffix_true_count",
        "attention_query_true_count",
        "valid_text_tokens",
        "pad_text_tokens",
        "attention_layout_ok",
        "query_prefix_ok",
    }
    for i, row in enumerate(rows, start=1):
        try:
            seq_len = int(float(row.get("seq_len", "")))
            num_query = int(float(row.get("num_query_tokens", "")))
            num_text = int(float(row.get("num_text_tokens", "")))
            selected_k = int(float(row.get("selected_k", "")))
        except ValueError:
            problems.append("row %d has nonnumeric token counts" % i)
            continue
        missing_prefix_fields = sorted(k for k in required_prefix_fields if k not in row or row.get(k, "") == "")
        if missing_prefix_fields:
            problems.append(
                "row %d missing query-prefix layout evidence: %s"
                % (i, ",".join(missing_prefix_fields))
            )
        else:
            try:
                expected_query = int(float(row.get("expected_query_tokens", "")))
                query_prefix_true = int(float(row.get("query_prefix_true_count", "")))
                text_suffix_true = int(float(row.get("text_suffix_true_count", "")))
                attention_query_true = int(float(row.get("attention_query_true_count", "")))
                valid_text_tokens = int(float(row.get("valid_text_tokens", "")))
                pad_text_tokens = int(float(row.get("pad_text_tokens", "")))
                attention_layout_ok = int(float(row.get("attention_layout_ok", "")))
                query_prefix_ok = int(float(row.get("query_prefix_ok", "")))
            except ValueError:
                problems.append("row %d has nonnumeric query-prefix or attention-layout evidence" % i)
                expected_query = 32
                query_prefix_true = -1
                text_suffix_true = -1
                attention_query_true = -1
                valid_text_tokens = -1
                pad_text_tokens = -1
                attention_layout_ok = 0
                query_prefix_ok = 0
            if expected_query != 32:
                problems.append("row %d expected_query_tokens=%d, expected 32" % (i, expected_query))
            if query_prefix_true != 32:
                problems.append("row %d query_prefix_true_count=%d, expected 32" % (i, query_prefix_true))
            if text_suffix_true != 0:
                problems.append("row %d text_suffix_true_count=%d, expected 0" % (i, text_suffix_true))
            if attention_query_true != 32:
                problems.append(
                    "row %d attention_query_true_count=%d, expected 32" % (i, attention_query_true)
                )
            if valid_text_tokens <= 0:
                problems.append("row %d valid_text_tokens=%d, expected >0" % (i, valid_text_tokens))
            if pad_text_tokens < 0:
                problems.append("row %d pad_text_tokens=%d, expected >=0" % (i, pad_text_tokens))
            if valid_text_tokens + pad_text_tokens != num_text:
                problems.append(
                    "row %d valid_text_tokens+pad_text_tokens=%d, expected num_text_tokens=%d"
                    % (i, valid_text_tokens + pad_text_tokens, num_text)
                )
            if attention_layout_ok != 1:
                problems.append("row %d attention_layout_ok=%d, expected 1" % (i, attention_layout_ok))
            if query_prefix_ok != 1:
                problems.append("row %d query_prefix_ok=%d, expected 1" % (i, query_prefix_ok))
        sample_id = str(row.get("sample_id", "")).strip()
        layer = str(row.get("layer", "")).strip()
        if sample_id:
            samples.add(sample_id)
        if layer:
            layers.add(layer)
        if num_query != 32:
            problems.append("row %d num_query_tokens=%d, expected 32" % (i, num_query))
        if num_text <= 0:
            problems.append("row %d num_text_tokens=%d, expected >0" % (i, num_text))
        if seq_len != num_query + num_text:
            problems.append("row %d seq_len=%d, expected num_query+num_text=%d" % (i, seq_len, num_query + num_text))
        if selected_k < 0 or selected_k > num_query:
            problems.append("row %d selected_k=%d outside [0,%d]" % (i, selected_k, num_query))
    if expected_samples > 0 and len(samples) < expected_samples:
        problems.append("unique samples=%d, expected >=%d" % (len(samples), expected_samples))
    if expected_layers > 0 and len(layers) < expected_layers:
        problems.append("unique layers=%d, expected >=%d" % (len(layers), expected_layers))
    stats = {"samples": len(samples), "layers": len(layers), "rows": len(rows)}
    return not problems, problems[:10], stats


def calibration_batch_trace_quality(
    rows: Sequence[Dict[str, str]],
    expected_samples: int,
) -> Tuple[bool, str, Dict[str, int]]:
    if not rows:
        return False, "no calibration batch trace rows", {"rows": 0, "physical_samples": 0, "batches": 0}

    problems: List[str] = []
    physical_max = 0
    batch_indices = set()
    required_fields = {
        "requested_samples",
        "cached_batches",
        "batch_index",
        "batch_size",
        "sample_start",
        "sample_end_exclusive",
        "physical_samples_seen",
        "seq_len",
        "expected_query_tokens",
        "query_prefix_true_min",
        "query_prefix_true_max",
        "total_query_tokens_min",
        "total_query_tokens_max",
        "valid_text_tokens_min",
        "valid_text_tokens_max",
        "pad_text_tokens_min",
        "pad_text_tokens_max",
        "query_prefix_ok_all",
        "attention_layout_ok_all",
    }
    for i, row in enumerate(rows, start=1):
        missing = sorted(k for k in required_fields if k not in row or row.get(k, "") == "")
        if missing:
            problems.append("row %d missing fields=%s" % (i, ",".join(missing)))
            continue
        try:
            batch_index = int(float(row.get("batch_index", "")))
            batch_size = int(float(row.get("batch_size", "")))
            sample_start = int(float(row.get("sample_start", "")))
            sample_end = int(float(row.get("sample_end_exclusive", "")))
            physical_seen = int(float(row.get("physical_samples_seen", "")))
            expected_query = int(float(row.get("expected_query_tokens", "")))
            query_prefix_min = int(float(row.get("query_prefix_true_min", "")))
            query_prefix_max = int(float(row.get("query_prefix_true_max", "")))
            total_query_min = int(float(row.get("total_query_tokens_min", "")))
            total_query_max = int(float(row.get("total_query_tokens_max", "")))
            valid_text_min = int(float(row.get("valid_text_tokens_min", "")))
            valid_text_max = int(float(row.get("valid_text_tokens_max", "")))
            pad_text_min = int(float(row.get("pad_text_tokens_min", "")))
            query_prefix_ok = int(float(row.get("query_prefix_ok_all", "")))
            attention_ok = int(float(row.get("attention_layout_ok_all", "")))
        except ValueError:
            problems.append("row %d has nonnumeric batch-trace fields" % i)
            continue

        batch_indices.add(batch_index)
        physical_max = max(physical_max, physical_seen)
        if batch_size <= 0:
            problems.append("row %d batch_size=%d, expected >0" % (i, batch_size))
        if sample_end - sample_start != batch_size:
            problems.append(
                "row %d sample range length=%d, expected batch_size=%d"
                % (i, sample_end - sample_start, batch_size)
            )
        if physical_seen != sample_end:
            problems.append("row %d physical_samples_seen=%d, expected sample_end=%d" % (i, physical_seen, sample_end))
        if expected_query != 32:
            problems.append("row %d expected_query_tokens=%d, expected 32" % (i, expected_query))
        if query_prefix_min != 32 or query_prefix_max != 32:
            problems.append(
                "row %d query_prefix_true_min/max=%d/%d, expected 32/32"
                % (i, query_prefix_min, query_prefix_max)
            )
        if total_query_min != 32 or total_query_max != 32:
            problems.append(
                "row %d total_query_tokens_min/max=%d/%d, expected 32/32"
                % (i, total_query_min, total_query_max)
            )
        if valid_text_min <= 0 or valid_text_max <= 0:
            problems.append(
                "row %d valid_text_tokens_min/max=%d/%d, expected >0"
                % (i, valid_text_min, valid_text_max)
            )
        if pad_text_min < 0:
            problems.append("row %d pad_text_tokens_min=%d, expected >=0" % (i, pad_text_min))
        if query_prefix_ok != 1:
            problems.append("row %d query_prefix_ok_all=%d, expected 1" % (i, query_prefix_ok))
        if attention_ok != 1:
            problems.append("row %d attention_layout_ok_all=%d, expected 1" % (i, attention_ok))

    if expected_samples > 0 and physical_max < expected_samples:
        problems.append("physical_samples_seen=%d, expected >=%d" % (physical_max, expected_samples))

    stats = {"rows": len(rows), "physical_samples": physical_max, "batches": len(batch_indices)}
    if problems:
        return False, "; ".join(problems[:10]), stats
    return True, "batches=%d; physical_samples_seen=%d" % (len(batch_indices), physical_max), stats


def selected_query_quality(
    rows: Sequence[Dict[str, str]],
    expected_layers: int,
    expected_query_tokens: int = 32,
    token_mask_rows: Optional[Sequence[Dict[str, str]]] = None,
) -> Tuple[bool, str, Dict[str, int]]:
    expected_by_pair: Dict[Tuple[str, str], int] = {}
    positive_layers = set()
    if token_mask_rows:
        for row in token_mask_rows:
            layer = str(row.get("layer", "")).strip()
            sample_id = str(row.get("sample_id", "")).strip()
            if not layer or not sample_id:
                continue
            try:
                selected_k = int(round(float(str(row.get("selected_k", "0")).strip())))
            except ValueError:
                continue
            if selected_k > 0:
                expected_by_pair[(layer, sample_id)] = selected_k
                positive_layers.add(layer)

    if not rows:
        stats = {
            "rows": 0,
            "layers": 0,
            "samples": 0,
            "expected_pairs": len(expected_by_pair),
            "matched_pairs": 0,
        }
        if expected_by_pair:
            return False, "no selected query rows for %d positive selected_k pairs" % len(expected_by_pair), stats
        return False, "no selected query rows", stats
    problems: List[str] = []
    layers = set()
    samples = set()
    total_count = 0.0
    observed_by_pair: Dict[Tuple[str, str], float] = {}
    for i, row in enumerate(rows, start=1):
        idx_s = row.get("query_index", row.get("selected_query_index", row.get("index", "")))
        count_s = row.get("count", row.get("frequency", row.get("value", "1")))
        try:
            idx = int(float(idx_s))
            count = float(count_s)
        except ValueError:
            problems.append("row %d has nonnumeric query_index/count" % i)
            continue
        if idx < 0 or idx >= expected_query_tokens:
            problems.append("row %d query_index=%d outside [0,%d]" % (i, idx, expected_query_tokens - 1))
        if count <= 0:
            problems.append("row %d count=%.6f expected >0" % (i, count))
        total_count += count
        layer = str(row.get("layer", "")).strip()
        sample_id = str(row.get("sample_id", "")).strip()
        if not layer:
            problems.append("row %d missing layer" % i)
        if not sample_id:
            problems.append("row %d missing sample_id" % i)
        if layer:
            layers.add(layer)
        if sample_id:
            samples.add(sample_id)
        if layer and sample_id:
            key = (layer, sample_id)
            observed_by_pair[key] = observed_by_pair.get(key, 0.0) + count
    if expected_layers > 0 and len(layers) < expected_layers:
        problems.append("selected-query layers=%d, expected >=%d" % (len(layers), expected_layers))
    matched_pairs = 0
    if expected_by_pair:
        missing_pairs = []
        mismatched_pairs = []
        for key, expected_count in expected_by_pair.items():
            observed_count = observed_by_pair.get(key)
            if observed_count is None:
                missing_pairs.append(key)
                continue
            if abs(observed_count - expected_count) > 1e-6:
                mismatched_pairs.append((key, expected_count, observed_count))
                continue
            matched_pairs += 1
        extra_pairs = sorted(set(observed_by_pair) - set(expected_by_pair))
        if missing_pairs:
            problems.append(
                "missing selected-query rows for %d layer/sample pairs, e.g. %s"
                % (len(missing_pairs), missing_pairs[:3])
            )
        if mismatched_pairs:
            problems.append(
                "selected-query count mismatch for %d pairs, e.g. %s"
                % (len(mismatched_pairs), mismatched_pairs[:3])
            )
        if extra_pairs:
            problems.append(
                "selected-query rows for %d pairs with token_mask selected_k=0/missing, e.g. %s"
                % (len(extra_pairs), extra_pairs[:3])
            )
    if total_count <= 0:
        problems.append("selected-query total count is zero")
    stats = {
        "rows": len(rows),
        "layers": len(layers),
        "samples": len(samples),
        "expected_pairs": len(expected_by_pair),
        "matched_pairs": matched_pairs,
    }
    if problems:
        return False, "; ".join(problems[:10]), stats
    token_mask_note = ""
    if expected_by_pair:
        token_mask_note = "; matched selected_k for %d/%d layer/sample pairs across %d positive-k layers" % (
            matched_pairs,
            len(expected_by_pair),
            len(positive_layers),
        )
    return (
        True,
        "rows=%d, layers=%d, samples=%d; all query_index values in [0,%d]%s"
        % (len(rows), len(layers), len(samples), expected_query_tokens - 1, token_mask_note),
        stats,
    )


def score_is_present(value: object) -> bool:
    s = str(value).strip()
    if not s:
        return False
    try:
        float(s.rstrip("%"))
        return True
    except ValueError:
        return False


def eval_coverage(
    eval_csv: Path,
    required_benchmarks: Sequence[str],
    required_methods: Sequence[str],
) -> Tuple[bool, str, List[str], List[str], List[str]]:
    rows = read_csv_rows(eval_csv)
    seen_bench = {normalize_benchmark_name(str(r.get("benchmark", ""))) for r in rows}
    seen_method = {normalize_method_name(str(r.get("method", ""))) for r in rows}
    req_bench = {normalize_benchmark_name(x) for x in required_benchmarks}
    req_method = {normalize_method_name(x) for x in required_methods}
    missing_bench = sorted(req_bench - seen_bench)
    missing_method = sorted(req_method - seen_method)
    cells = set()
    nonnumeric_cells = []
    for row in rows:
        bench = normalize_benchmark_name(str(row.get("benchmark", "")))
        method = normalize_method_name(str(row.get("method", "")))
        if not bench or not method:
            continue
        if score_is_present(row.get("score", "")):
            cells.add((method, bench))
        else:
            nonnumeric_cells.append("%s:%s" % (method, bench))
    missing_cells = []
    for method in sorted(req_method):
        for bench in sorted(req_bench):
            if (method, bench) not in cells:
                missing_cells.append("%s:%s" % (method, bench))
    ok = not missing_bench and not missing_method and not missing_cells
    note = "benchmarks=%s; methods=%s" % (
        ",".join(sorted(x for x in seen_bench if x)),
        ",".join(sorted(x for x in seen_method if x)),
    )
    note += "; complete_cells=%d/%d" % (len(cells & {(m, b) for m in req_method for b in req_bench}), len(req_method) * len(req_bench))
    if nonnumeric_cells:
        note += "; nonnumeric_scores=%s" % ",".join(sorted(set(nonnumeric_cells))[:8])
    return ok, note, missing_bench, missing_method, missing_cells


def eval_metadata_quality(rows: Sequence[Dict[str, str]], args: argparse.Namespace) -> Tuple[bool, str]:
    if not rows:
        return False, "no eval rows"
    problems: List[str] = []
    for i, row in enumerate(rows, start=1):
        method = normalize_method_name(str(row.get("method", "")))
        if method not in {"dense", "atv", "wanda", "tamp"}:
            continue
        t5 = row_float(row, "t5_sparsity")
        vit = row_float(row, "vit_sparsity")
        if t5 is None or vit is None:
            problems.append("row %d method=%s missing numeric sparsity metadata" % (i, method))
            continue
        if method == "dense":
            if abs(t5) > args.sparsity_tol or abs(vit) > args.sparsity_tol:
                problems.append("row %d dense sparsity t5=%.6f vit=%.6f, expected 0/0" % (i, t5, vit))
        else:
            if abs(t5 - args.expected_t5_sparsity) > args.sparsity_tol:
                problems.append(
                    "row %d method=%s t5_sparsity %.6f outside %.6f +/- %.6f"
                    % (i, method, t5, args.expected_t5_sparsity, args.sparsity_tol)
                )
            if vit > args.expected_vit_sparsity_max:
                problems.append(
                    "row %d method=%s vit_sparsity %.6f > %.6f"
                    % (i, method, vit, args.expected_vit_sparsity_max)
                )
    if problems:
        return False, "; ".join(problems[:8])
    return True, "eval rows carry expected dense/pruned T5-only sparsity metadata"


def eval_atv_alpha_quality(
    rows: Sequence[Dict[str, str]],
    required_benchmarks: Sequence[str],
    required_alphas: Sequence[float],
) -> Tuple[bool, str]:
    if not required_alphas:
        return True, "no required ATV eval alpha grid configured"
    req_bench = {normalize_benchmark_name(x) for x in required_benchmarks}
    cells = set()
    seen_alphas = set()
    for row in rows:
        method = normalize_method_name(str(row.get("method", "")))
        if method != "atv":
            continue
        alpha = row_float(row, "alpha")
        bench = normalize_benchmark_name(str(row.get("benchmark", "")))
        if alpha is None or not math.isfinite(alpha) or bench not in req_bench:
            continue
        if not score_is_present(row.get("score", "")):
            continue
        for want in required_alphas:
            if math.isclose(alpha, want, rel_tol=0.0, abs_tol=1e-6):
                cells.add((want, bench))
                seen_alphas.add(want)
                break
    missing_alphas = [a for a in required_alphas if a not in seen_alphas]
    missing_cells = []
    for alpha in required_alphas:
        for bench in sorted(req_bench):
            if (alpha, bench) not in cells:
                missing_cells.append("alpha%.6g:%s" % (alpha, bench))
    if missing_alphas or missing_cells:
        note = "seen ATV eval alphas=%s" % ",".join("%.6g" % a for a in sorted(seen_alphas))
        if missing_alphas:
            note += "; missing_alphas=%s" % ",".join("%.6g" % a for a in missing_alphas)
        if missing_cells:
            note += "; missing_cells=%s" % ",".join(missing_cells[:16])
        return False, note
    return True, "ATV eval alpha cells complete for %s" % ",".join("%.6g" % a for a in required_alphas)


def eval_seed_quality(
    rows: Sequence[Dict[str, str]],
    required_benchmarks: Sequence[str],
    required_methods: Sequence[str],
    required_seeds: Sequence[str],
    required_atv_alphas: Sequence[float],
) -> Tuple[bool, str]:
    if not required_seeds:
        return True, "no required eval seed grid configured"
    req_bench = {normalize_benchmark_name(x) for x in required_benchmarks}
    req_methods = {normalize_method_name(x) for x in required_methods}
    required_seed_set = {str(x).strip() for x in required_seeds if str(x).strip()}
    if not required_seed_set:
        return True, "no required eval seed grid configured"

    cells = set()
    for row in rows:
        method = normalize_method_name(str(row.get("method", "")))
        bench = normalize_benchmark_name(str(row.get("benchmark", "")))
        seed = str(row.get("seed", "")).strip()
        if method not in req_methods or bench not in req_bench or seed not in required_seed_set:
            continue
        if not score_is_present(row.get("score", "")):
            continue
        alpha = row_float(row, "alpha")
        if method == "atv":
            for want in required_atv_alphas:
                if alpha is not None and math.isclose(alpha, want, rel_tol=0.0, abs_tol=1e-6):
                    cells.add((method, bench, seed, "alpha%.6g" % want))
                    break
        elif method != "dense":
            cells.add((method, bench, seed, ""))
        else:
            cells.add((method, bench, "", ""))

    missing = []
    for method in sorted(req_methods):
        if method == "dense":
            for bench in sorted(req_bench):
                if (method, bench, "", "") not in cells:
                    missing.append("%s:%s" % (method, bench))
        elif method == "atv":
            for alpha in required_atv_alphas:
                alpha_key = "alpha%.6g" % alpha
                for seed in sorted(required_seed_set):
                    for bench in sorted(req_bench):
                        if (method, bench, seed, alpha_key) not in cells:
                            missing.append("%s:%s:%s:%s" % (method, alpha_key, seed, bench))
        else:
            for seed in sorted(required_seed_set):
                for bench in sorted(req_bench):
                    if (method, bench, seed, "") not in cells:
                        missing.append("%s:%s:%s" % (method, seed, bench))

    if missing:
        return (
            False,
            "required eval seeds=%s; missing_seed_cells=%s"
            % (",".join(sorted(required_seed_set)), ",".join(missing[:24])),
        )
    return True, "required eval seeds complete for pruned methods: %s" % ",".join(sorted(required_seed_set))


def eval_provenance_quality(
    rows: Sequence[Dict[str, str]],
    required_methods: Sequence[str],
    required_seeds: Sequence[str],
    required_atv_alphas: Sequence[float],
    allow_shared_ckpts: bool,
) -> Tuple[bool, str]:
    if not rows:
        return False, "no eval provenance rows"
    req_methods = {normalize_method_name(x) for x in required_methods}
    required_seed_set = {str(x).strip() for x in required_seeds if str(x).strip()}
    cells = set()
    ckpts_by_method_alpha: Dict[Tuple[str, str], Dict[str, str]] = {}
    problems: List[str] = []

    for row in rows:
        method = normalize_method_name(str(row.get("method", "")))
        if method not in req_methods:
            continue
        seed = str(row.get("seed", "")).strip()
        calibration = str(row.get("calibration", "")).strip()
        alpha_s = str(row.get("alpha", "")).strip()
        if method == "atv" and not alpha_s:
            alpha_s = infer_alpha_from_label(calibration) or "1"
        ckpt = str(row.get("ckpt", "")).strip()
        metrics_jsonl = str(row.get("metrics_jsonl", "")).strip()
        okvqa_eval_txt = str(row.get("okvqa_eval_txt", "")).strip()

        if not metrics_jsonl:
            problems.append("method=%s seed=%s missing metrics_jsonl" % (method, seed))
        elif not Path(metrics_jsonl).exists():
            problems.append("method=%s seed=%s metrics_jsonl not found: %s" % (method, seed, metrics_jsonl))
        if not okvqa_eval_txt:
            problems.append("method=%s seed=%s missing okvqa_eval_txt" % (method, seed))
        elif not Path(okvqa_eval_txt).exists():
            problems.append("method=%s seed=%s okvqa_eval_txt not found: %s" % (method, seed, okvqa_eval_txt))
        if method != "dense" and not ckpt:
            problems.append("method=%s seed=%s missing checkpoint path" % (method, seed))
        elif method != "dense" and ckpt and not Path(ckpt).exists():
            problems.append("method=%s seed=%s checkpoint not found: %s" % (method, seed, ckpt))

        alpha_key = ""
        if method == "atv":
            alpha = row_float({"alpha": alpha_s}, "alpha")
            for want in required_atv_alphas:
                if alpha is not None and math.isclose(alpha, want, rel_tol=0.0, abs_tol=1e-6):
                    alpha_key = "alpha%.6g" % want
                    break
            if alpha_key:
                cells.add((method, seed, alpha_key))
        elif method == "dense":
            cells.add((method, "", ""))
        else:
            cells.add((method, seed, ""))

        if method != "dense" and ckpt:
            ckpts_by_method_alpha.setdefault((method, alpha_key), {})[seed] = ckpt

    missing = []
    for method in sorted(req_methods):
        if method == "dense":
            if (method, "", "") not in cells:
                missing.append("dense")
        elif method == "atv":
            for alpha in required_atv_alphas:
                alpha_key = "alpha%.6g" % alpha
                for seed in sorted(required_seed_set):
                    if (method, seed, alpha_key) not in cells:
                        missing.append("atv:%s:%s" % (alpha_key, seed))
        else:
            for seed in sorted(required_seed_set):
                if (method, seed, "") not in cells:
                    missing.append("%s:%s" % (method, seed))

    if not allow_shared_ckpts:
        for (method, alpha_key), by_seed in sorted(ckpts_by_method_alpha.items()):
            if method not in req_methods or len(by_seed) <= 1:
                continue
            reverse: Dict[str, List[str]] = {}
            for seed, ckpt in by_seed.items():
                if seed in required_seed_set:
                    reverse.setdefault(ckpt, []).append(seed)
            shared = ["%s:%s:%s" % (method, alpha_key or "main", ",".join(sorted(seeds))) for ckpt, seeds in reverse.items() if len(seeds) > 1]
            if shared:
                problems.append("shared checkpoint across seeds: %s" % ";".join(shared[:4]))

    if missing or problems:
        note = []
        if missing:
            note.append("missing_provenance=%s" % ",".join(missing[:24]))
        if problems:
            note.append("problems=%s" % "; ".join(problems[:8]))
        return False, "; ".join(note)
    return True, "eval provenance covers checkpoints and raw files for required method/seed/alpha cells"


def prune_provenance_quality(
    rows: Sequence[Dict[str, str]],
    args: argparse.Namespace,
) -> Tuple[bool, str]:
    if not rows:
        return False, "no pruning provenance rows"

    required_seed_set = {str(x).strip() for x in split_csv_arg(args.required_eval_seeds) if str(x).strip()}
    required_atv_alphas = [float(x) for x in split_csv_arg(args.required_eval_atv_alphas)]
    required_methods = {normalize_method_name(x) for x in split_csv_arg(args.required_prune_methods)}
    cells = set()
    required_rows: List[Dict[str, str]] = []
    problems: List[str] = []

    for i, row in enumerate(rows, start=1):
        method = normalize_method_name(str(row.get("method", "")))
        if method == "naive":
            method = "wanda"
        if method not in required_methods:
            continue
        role = str(row.get("role", "")).strip().casefold()
        if role == "smoke":
            continue
        seed = str(row.get("seed", "")).strip()
        alpha = row_float(row, "alpha")
        alpha_key = ""
        if method == "atv":
            for want in required_atv_alphas:
                if alpha is not None and math.isclose(alpha, want, rel_tol=0.0, abs_tol=1e-6):
                    alpha_key = "alpha%.6g" % want
                    break
            if not alpha_key:
                continue
        elif method == "wanda":
            alpha_key = ""

        if seed in required_seed_set:
            cells.add((method, seed, alpha_key))
            required_rows.append(row)

    missing = []
    for method in sorted(required_methods):
        if method == "atv":
            for alpha in required_atv_alphas:
                alpha_key = "alpha%.6g" % alpha
                for seed in sorted(required_seed_set):
                    if (method, seed, alpha_key) not in cells:
                        missing.append("atv:%s:%s" % (alpha_key, seed))
        elif method == "wanda":
            for seed in sorted(required_seed_set):
                if (method, seed, "") not in cells:
                    missing.append("wanda:%s" % seed)

    calib_jsons = {str(r.get("calib_json", "")).strip() for r in required_rows if str(r.get("calib_json", "")).strip()}
    image_dirs = {str(r.get("images_dir", "")).strip() for r in required_rows if str(r.get("images_dir", "")).strip()}
    t5_specs = {str(r.get("t5_spec", "")).strip() for r in required_rows if str(r.get("t5_spec", "")).strip()}

    if len(calib_jsons) != 1:
        problems.append("expected one calibration JSON across required pruning rows, saw %d" % len(calib_jsons))
    if len(image_dirs) != 1:
        problems.append("expected one image directory across required pruning rows, saw %d" % len(image_dirs))
    if len(t5_specs) != 1:
        problems.append("expected one T5 prune spec across required pruning rows, saw %d" % len(t5_specs))

    for path_s in sorted(calib_jsons):
        if not Path(path_s).is_file():
            problems.append("calibration JSON not found: %s" % path_s)
    for path_s in sorted(image_dirs):
        if not Path(path_s).is_dir():
            problems.append("image directory not found: %s" % path_s)

    for i, row in enumerate(required_rows, start=1):
        prefix = "row%d method=%s seed=%s" % (
            i,
            normalize_method_name(str(row.get("method", ""))),
            row.get("seed", ""),
        )
        ckpt = str(row.get("ckpt", "")).strip()
        calib_cfg = str(row.get("calib_cfg", "")).strip()
        prune_log = str(row.get("prune_log", "")).strip()
        sparsity_csv = str(row.get("sparsity_csv", "")).strip()
        if not ckpt or not Path(ckpt).is_file():
            problems.append("%s checkpoint not found: %s" % (prefix, ckpt or "<empty>"))
        if not calib_cfg or not Path(calib_cfg).is_file():
            problems.append("%s calibration cfg not found: %s" % (prefix, calib_cfg or "<empty>"))
        if str(row.get("run_prune", "")).strip() != "1":
            problems.append("%s run_prune is not 1" % prefix)
        if prune_log and not Path(prune_log).is_file():
            problems.append("%s prune_log not found: %s" % (prefix, prune_log))
        if sparsity_csv and not Path(sparsity_csv).is_file():
            problems.append("%s sparsity_csv not found: %s" % (prefix, sparsity_csv))
        if str(row.get("pruning_scope", "")).strip().casefold() != "t5_only":
            problems.append("%s pruning_scope is not t5_only" % prefix)
        num_data = row_float(row, "num_data")
        if num_data is None or int(num_data) != int(args.expected_prune_num_data):
            problems.append("%s num_data=%s expected=%d" % (prefix, row.get("num_data", ""), args.expected_prune_num_data))
        t5_target = row_float(row, "t5_sparsity_target")
        if t5_target is None or abs(t5_target - args.expected_t5_sparsity) > args.sparsity_tol:
            problems.append("%s t5_sparsity_target=%s" % (prefix, row.get("t5_sparsity_target", "")))
        vit_target = row_float(row, "vit_sparsity_target")
        if vit_target is None or vit_target > args.expected_vit_sparsity_max:
            problems.append("%s vit_sparsity_target=%s" % (prefix, row.get("vit_sparsity_target", "")))

    if missing or problems:
        note = []
        if missing:
            note.append("missing_prune_provenance=%s" % ",".join(missing[:24]))
        if problems:
            note.append("problems=%s" % "; ".join(problems[:10]))
        return False, "; ".join(note)
    return True, "prune provenance covers methods=%s seeds=%s alphas=%s with one calibration/spec" % (
        ",".join(sorted(required_methods)),
        ",".join(sorted(required_seed_set)),
        ",".join("%.6g" % a for a in required_atv_alphas),
    )


def importance_quality(rows: Sequence[Dict[str, str]], args: argparse.Namespace) -> Tuple[bool, str, Dict[str, int]]:
    if not rows:
        return False, "no importance rows", {"rows": 0, "layers": 0}
    problems: List[str] = []
    layers = set()
    required_numeric = [
        "numel",
        "mean_wanda_importance",
        "median_wanda_importance",
        "max_wanda_importance",
        "scaler_row_mean",
        "scaler_row_max",
        "weight_abs_mean",
        "mask_sparsity",
    ]
    for i, row in enumerate(rows, start=1):
        layer = str(row.get("layer", "")).strip()
        if layer:
            layers.add(layer)
        module = str(row.get("module", "")).strip()
        if not module:
            problems.append("row %d missing module" % i)
        for key in required_numeric:
            value = row_float(row, key)
            if value is None or not math.isfinite(value):
                problems.append("row %d %s is not finite numeric" % (i, key))
                continue
            if key == "numel" and value <= 0:
                problems.append("row %d numel %.6f expected >0" % (i, value))
            elif key != "mask_sparsity" and value < 0:
                problems.append("row %d %s %.6f expected >=0" % (i, key, value))
            elif key == "mask_sparsity":
                if value < 0 or value > 1:
                    problems.append("row %d mask_sparsity %.6f outside [0,1]" % (i, value))
                elif abs(value - args.expected_t5_sparsity) > args.sparsity_tol:
                    problems.append(
                        "row %d mask_sparsity %.6f outside %.6f +/- %.6f"
                        % (i, value, args.expected_t5_sparsity, args.sparsity_tol)
                    )
    if len(rows) < args.min_importance_rows:
        problems.append("importance rows=%d, expected >=%d" % (len(rows), args.min_importance_rows))
    if len(layers) < args.min_mask_layers:
        problems.append("importance layers=%d, expected >=%d" % (len(layers), args.min_mask_layers))
    stats = {"rows": len(rows), "layers": len(layers)}
    if problems:
        return False, "; ".join(problems[:10]), stats
    return (
        True,
        "rows=%d, layers=%d; finite importance/scaler/weight/mask sparsity values"
        % (len(rows), len(layers)),
        stats,
    )


def sparsity_quality(rows: Sequence[Dict[str, str]], args: argparse.Namespace) -> Tuple[bool, str]:
    if not rows:
        return False, "no sparsity rows"

    def rows_for(group: str) -> List[Dict[str, str]]:
        return [row for row in rows if str(row.get("group", "")) == group]

    problems: List[str] = []
    t5_rows = rows_for("t5_model.all")
    vit_rows = rows_for("visual_encoder")
    qformer_rows = rows_for("Qformer")
    t5_proj_rows = rows_for("t5_proj")
    if not t5_rows:
        problems.append("missing t5_model.all row")
    if not vit_rows:
        problems.append("missing visual_encoder row")
    if not qformer_rows:
        problems.append("missing Qformer row")
    if not t5_proj_rows:
        problems.append("missing t5_proj row")

    for row in t5_rows:
        sp = row_float(row, "sparsity")
        tag = str(row.get("tag", ""))
        if sp is None:
            problems.append("t5_model.all row has nonnumeric sparsity")
        elif abs(sp - args.expected_t5_sparsity) > args.sparsity_tol:
            problems.append(
                "%s t5 sparsity %.6f outside %.6f +/- %.6f"
                % (tag, sp, args.expected_t5_sparsity, args.sparsity_tol)
            )
    for row in vit_rows:
        sp = row_float(row, "sparsity")
        tag = str(row.get("tag", ""))
        if sp is None:
            problems.append("visual_encoder row has nonnumeric sparsity")
        elif sp > args.expected_vit_sparsity_max:
            problems.append("%s vit sparsity %.6f > %.6f" % (tag, sp, args.expected_vit_sparsity_max))
    for row in list(qformer_rows) + list(t5_proj_rows):
        sp = row_float(row, "sparsity")
        tag = str(row.get("tag", ""))
        group = str(row.get("group", ""))
        if sp is None:
            problems.append("%s row has nonnumeric sparsity" % group)
        elif sp > args.expected_non_t5_sparsity_max:
            problems.append(
                "%s %s sparsity %.6f > %.6f"
                % (tag, group, sp, args.expected_non_t5_sparsity_max)
            )

    if problems:
        return False, "; ".join(problems[:8])
    tags = sorted({str(row.get("tag", "")) for row in rows if str(row.get("tag", ""))})
    return True, "tags=%s; t5 target=%.6f +/- %.6f; vit max=%.6f; non_t5 max=%.6f" % (
        ",".join(tags),
        args.expected_t5_sparsity,
        args.sparsity_tol,
        args.expected_vit_sparsity_max,
        args.expected_non_t5_sparsity_max,
    )


def mask_rows_for_pair(rows: Sequence[Dict[str, object]], pair_substring: str) -> List[Dict[str, object]]:
    needle = pair_substring.strip().casefold()
    if not needle:
        return []
    return [row for row in rows if needle in str(row.get("pair", "")).casefold()]


def required_mask_pair_quality(
    mask_rows: Sequence[Dict[str, object]],
    module_mask_rows: Sequence[Dict[str, object]],
    args: argparse.Namespace,
) -> Tuple[bool, str, int]:
    required_pairs = split_csv_arg(args.required_mask_pairs)
    if not required_pairs:
        return True, "no required mask-pair coverage configured", 0

    problems: List[str] = []
    notes: List[str] = []
    covered = 0
    for pair in required_pairs:
        layer_rows = mask_rows_for_pair(mask_rows, pair)
        module_rows = mask_rows_for_pair(module_mask_rows, pair)
        if not layer_rows:
            problems.append("%s:missing layer rows" % pair)
            continue
        if not module_rows:
            problems.append("%s:missing module rows" % pair)
            continue
        layer_count = len({str(row.get("layer", "")) for row in layer_rows if str(row.get("layer", ""))})
        module_count = len({str(row.get("module", "")) for row in module_rows if str(row.get("module", ""))})
        if layer_count < args.min_mask_layers:
            problems.append("%s:layers=%d<%d" % (pair, layer_count, args.min_mask_layers))
        if module_count < args.min_mask_modules:
            problems.append("%s:modules=%d<%d" % (pair, module_count, args.min_mask_modules))
        if not problems or all(not p.startswith(pair + ":") for p in problems):
            covered += 1
            notes.append("%s layers=%d modules=%d" % (pair, layer_count, module_count))

    if problems:
        return False, "; ".join(problems[:10]), covered
    return True, "; ".join(notes), covered


def mean_iou(rows: Sequence[Dict[str, object]], key: str) -> Optional[float]:
    vals: List[float] = []
    for row in rows:
        value = row_float(row, key)
        if value is not None and math.isfinite(value):
            vals.append(value)
    if not vals:
        return None
    return sum(vals) / len(vals)


def alpha1_degenerate_rate(alpha_summary: Sequence[Dict[str, object]]) -> Optional[float]:
    vals: List[float] = []
    for row in alpha_summary:
        alpha = row_float(row, "alpha")
        deg = row_float(row, "mean_degenerate_rate")
        if alpha is not None and deg is not None and math.isclose(alpha, 1.0, rel_tol=0.0, abs_tol=1e-6):
            vals.append(deg)
    if not vals:
        return None
    return max(vals)


def atv_wanda_specificity_quality(
    mask_rows: Sequence[Dict[str, object]],
    alpha_summary: Sequence[Dict[str, object]],
    args: argparse.Namespace,
) -> Tuple[bool, str]:
    rows = mask_rows_for_pair(mask_rows, args.atv_wanda_pair_name)
    if not rows:
        return False, "missing mask pair containing '%s'" % args.atv_wanda_pair_name

    prune_mean = mean_iou(rows, "prune_iou")
    keep_mean = mean_iou(rows, "keep_iou")
    if prune_mean is None or keep_mean is None:
        return False, "mask pair has no numeric IoU values"

    identical = prune_mean >= args.identical_iou_threshold and keep_mean >= args.identical_iou_threshold
    deg = alpha1_degenerate_rate(alpha_summary)
    if identical:
        if deg is not None and deg >= args.alpha1_degenerate_explain_threshold:
            return True, (
                "ATV alpha=1 is mask-identical to Wanda but degeneracy is explicit "
                "(mean degenerate rate %.4f >= %.4f)"
                % (deg, args.alpha1_degenerate_explain_threshold)
            )
        return False, (
            "ATV alpha=1 is effectively identical to Wanda "
            "(mean prune_iou=%.6f, keep_iou=%.6f) without sufficient degeneracy evidence"
            % (prune_mean, keep_mean)
        )

    return True, "ATV differs from Wanda (mean prune_iou=%.6f, keep_iou=%.6f)" % (prune_mean, keep_mean)


def reproducibility_quality(
    mask_rows: Sequence[Dict[str, object]],
    args: argparse.Namespace,
) -> Tuple[bool, str]:
    if not args.repro_pair_name:
        return True, "reproducibility gate disabled"
    rows = mask_rows_for_pair(mask_rows, args.repro_pair_name)
    if not rows:
        return False, "missing same-seed ATV reproducibility mask pair containing '%s'" % args.repro_pair_name

    bad: List[str] = []
    for row in rows:
        layer = str(row.get("layer", ""))
        prune_iou = row_float(row, "prune_iou")
        keep_iou = row_float(row, "keep_iou")
        if prune_iou is None or keep_iou is None:
            bad.append("layer %s has nonnumeric IoU" % layer)
            continue
        if prune_iou < args.repro_iou_threshold or keep_iou < args.repro_iou_threshold:
            bad.append(
                "layer %s prune_iou=%.6f keep_iou=%.6f below %.6f"
                % (layer, prune_iou, keep_iou, args.repro_iou_threshold)
            )
    if bad:
        return False, "; ".join(bad[:6])
    return True, "same-seed ATV mask IoU >= %.6f for %d layers" % (args.repro_iou_threshold, len(rows))


def alpha_from_mask_pair_name(pair: str) -> Optional[float]:
    s = pair.casefold()
    m = re.search(r"atv[_-]?alpha([-+0-9.p]+).*wanda", s)
    if not m:
        return None
    raw = m.group(1).replace("p", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def alpha_mask_similarity_summary(mask_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[float, Dict[str, object]] = {}
    for row in mask_rows:
        alpha = alpha_from_mask_pair_name(str(row.get("pair", "")))
        if alpha is None:
            continue
        prune = row_float(row, "prune_iou")
        keep = row_float(row, "keep_iou")
        if prune is None or keep is None:
            continue
        slot = grouped.setdefault(alpha, {"alpha": alpha, "layers": 0, "similarity_sum": 0.0})
        slot["layers"] = int(slot["layers"]) + 1
        slot["similarity_sum"] = float(slot["similarity_sum"]) + (prune + keep) / 2.0

    out: List[Dict[str, object]] = []
    for alpha, slot in grouped.items():
        layers = int(slot["layers"])
        if layers <= 0:
            continue
        out.append(
            {
                "alpha": alpha,
                "layers": layers,
                "mean_mask_similarity": float(slot["similarity_sum"]) / layers,
            }
        )
    out.sort(key=lambda r: float(r["alpha"]))
    return out


def alpha_mask_trend_quality(
    mask_rows: Sequence[Dict[str, object]],
    args: argparse.Namespace,
) -> Tuple[bool, str]:
    summary = alpha_mask_similarity_summary(mask_rows)
    if len(summary) < args.min_alpha_mask_points:
        return False, "alpha-vs-Wanda mask pairs=%d, required >=%d" % (len(summary), args.min_alpha_mask_points)

    low_alpha = args.alpha_mask_low
    high_alpha = args.alpha_mask_high
    low_rows = [r for r in summary if math.isclose(float(r["alpha"]), low_alpha, rel_tol=0.0, abs_tol=1e-6)]
    high_rows = [r for r in summary if math.isclose(float(r["alpha"]), high_alpha, rel_tol=0.0, abs_tol=1e-6)]
    if not low_rows:
        return False, "missing alpha %.6g vs Wanda mask pair" % low_alpha
    if not high_rows:
        return False, "missing alpha %.6g vs Wanda mask pair" % high_alpha

    low_sim = float(low_rows[0]["mean_mask_similarity"])
    high_sim = float(high_rows[0]["mean_mask_similarity"])
    if high_sim + args.alpha_mask_trend_tol < low_sim:
        return False, (
            "alpha %.6g similarity %.6f is lower than alpha %.6g similarity %.6f"
            % (high_alpha, high_sim, low_alpha, low_sim)
        )

    ordered = ", ".join("%.6g:%.4f" % (float(r["alpha"]), float(r["mean_mask_similarity"])) for r in summary)
    return True, "alpha mask similarity trend ok (%s)" % ordered


def runtime_environment_quality(path: Path) -> Tuple[bool, str, int]:
    if not path.exists():
        return False, "runtime_environment.json missing; run capture_atv_runtime_env.py before final validation", 0
    try:
        payload = json.loads(read_text(path))
    except Exception as exc:
        return False, "could not parse runtime_environment.json: %s" % exc, 0

    source_files = payload.get("source_files", [])
    if not isinstance(source_files, list):
        return False, "runtime_environment.json has no source_files list", 0

    required_roles = {
        "original_atv_pruner",
        "evaluate_blip.py",
        "lavis/configs/models/blip2/blip2_pretrain_flant5xl.yaml",
        "lavis/compression/pruners/wanda_pruner.py",
        "lavis/models/blip2_models/blip2.py",
        "lavis/models/blip2_models/blip2_t5.py",
        "scripts/blip2/audit_atv_validation_report.py",
        "scripts/blip2/capture_atv_runtime_env.py",
        "scripts/blip2/check_ckpt_sparsity.py",
        "scripts/blip2/compare_ckpts.py",
        "scripts/blip2/export_blip2_full_dense_state_dict.py",
        "scripts/blip2/materialize_cc3m_calib_cfg.py",
        "scripts/blip2/preflight_atv_validation.py",
        "scripts/blip2/snapshot_atv_artifacts.py",
        "scripts/blip2/validate_atv_migration.py",
        "scripts/blip2/run_atv_cc3m_prune_then_eval.sh",
        "scripts/blip2/run_atv_eval_matrix_fourbench.sh",
        "scripts/blip2/run_atv_full_verify.sh",
        "scripts/blip2/run_atv_multiseed_validation.sh",
    }
    by_role = {str(row.get("role", "")): row for row in source_files if isinstance(row, dict)}
    missing_roles = sorted(role for role in required_roles if role not in by_role)
    missing_hashes = sorted(
        role
        for role in required_roles
        if role in by_role
        and (not by_role[role].get("exists") or not str(by_role[role].get("sha256", "")))
    )

    python_info = payload.get("python", {})
    git_info = payload.get("git", {})
    torch_info = payload.get("torch", {})
    has_python = isinstance(python_info, dict) and bool(python_info.get("executable"))
    lavis_commit = ""
    if isinstance(git_info, dict):
        lavis = git_info.get("lavis_root", {})
        if isinstance(lavis, dict):
            lavis_commit = str(lavis.get("commit", ""))

    problems = []
    if not has_python:
        problems.append("missing python executable")
    if missing_roles:
        problems.append("missing source roles=%s" % ",".join(missing_roles))
    if missing_hashes:
        problems.append("missing source hashes=%s" % ",".join(missing_hashes))
    if not lavis_commit:
        problems.append("lavis git commit unavailable")

    torch_note = ""
    if isinstance(torch_info, dict):
        if torch_info.get("available") is False:
            problems.append("torch import failed")
            torch_note = "; torch import failed: %s" % torch_info.get("error", "")
        elif torch_info.get("skipped"):
            problems.append("torch import skipped")
        elif torch_info.get("version"):
            torch_note = "; torch=%s cuda_available=%s" % (
                torch_info.get("version"),
                torch_info.get("cuda_available"),
            )
        else:
            problems.append("torch version unavailable")
    else:
        problems.append("missing torch provenance")

    if problems:
        return False, "; ".join(problems) + torch_note, len(source_files)
    return True, "source hashes=%d; lavis_commit=%s%s" % (len(source_files), lavis_commit[:12], torch_note), len(source_files)


def query_token_mapping_quality(rows: Sequence[Dict[str, str]]) -> Tuple[bool, str]:
    if not rows:
        return False, "query_token_mapping.csv missing or empty"
    failures = [r for r in rows if str(r.get("status", "")).upper() != "PASS"]
    expected_checks = {
        "Blip2T5 constructor default query-token count",
        "pretrain_flant5xl config query-token count",
        "Q-Former query token parameter allocation",
        "Q-Former query length config",
    }
    seen = {str(r.get("check", "")) for r in rows}
    missing = sorted(expected_checks - seen)
    if failures or missing:
        parts = []
        if failures:
            parts.append("failed checks=%s" % ",".join(str(r.get("check", "")) for r in failures))
        if missing:
            parts.append("missing checks=%s" % ",".join(missing))
        return False, "; ".join(parts)
    return True, "%d/%d query-token mapping checks passed; target query tokens=32" % (len(rows), len(expected_checks))


def mask_evidence_quality(
    rows: Sequence[Dict[str, object]],
    kind: str,
    args: argparse.Namespace,
) -> Tuple[bool, str]:
    if not rows:
        return False, "no %s mask rows" % kind
    required = ["pair", "a_ckpt", "b_ckpt", "mask_inference", "mask_prefix", "min_numel"]
    problems: List[str] = []
    pairs = set()
    ckpt_pairs = set()
    inference_modes = set()
    for i, row in enumerate(rows, start=1):
        missing = [key for key in required if str(row.get(key, "")).strip() == ""]
        if missing:
            problems.append("row %d missing %s" % (i, ",".join(missing)))
            continue
        pairs.add(str(row.get("pair", "")).strip())
        a_ckpt = str(row.get("a_ckpt", "")).strip()
        b_ckpt = str(row.get("b_ckpt", "")).strip()
        ckpt_pairs.add((a_ckpt, b_ckpt))
        if not Path(a_ckpt).is_file():
            problems.append("row %d a_ckpt not found: %s" % (i, a_ckpt))
        if not Path(b_ckpt).is_file():
            problems.append("row %d b_ckpt not found: %s" % (i, b_ckpt))
        inference = str(row.get("mask_inference", "")).strip()
        inference_modes.add(inference)
        if inference != "dense_base_nonzero_to_zero" and not args.allow_zero_only_mask_inference:
            problems.append(
                "row %d mask_inference=%s; strict validation requires dense_base_nonzero_to_zero"
                % (i, inference or "<empty>")
            )
        base_ckpt = str(row.get("base_ckpt", "")).strip()
        if inference == "dense_base_nonzero_to_zero":
            if not base_ckpt:
                problems.append("row %d missing base_ckpt for dense-base mask inference" % i)
            elif not Path(base_ckpt).is_file():
                problems.append("row %d base_ckpt not found: %s" % (i, base_ckpt))
        elif base_ckpt and not Path(base_ckpt).is_file():
            problems.append("row %d base_ckpt not found: %s" % (i, base_ckpt))
        min_numel = row_float(row, "min_numel")
        if min_numel is None or min_numel <= 0:
            problems.append("row %d min_numel=%s expected >0" % (i, row.get("min_numel", "")))
    if problems:
        return False, "; ".join(problems[:10])
    return True, "%s mask provenance rows=%d pairs=%d ckpt_pairs=%d inference=%s" % (
        kind,
        len(rows),
        len(pairs),
        len(ckpt_pairs),
        ",".join(sorted(inference_modes)),
    )


def write_dense_mask_base_summary(path: Path, mask_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    base_paths = sorted(
        {
            str(row.get("base_ckpt", "")).strip()
            for row in mask_rows
            if str(row.get("mask_inference", "")).strip() == "dense_base_nonzero_to_zero"
            and str(row.get("base_ckpt", "")).strip()
        }
    )
    rows: List[Dict[str, object]] = []
    for base_s in base_paths:
        base_path = Path(base_s)
        summary_path = Path(base_s + ".summary.json")
        if not base_path.is_file():
            rows.append(
                {
                    "base_ckpt": base_s,
                    "summary_json": str(summary_path),
                    "group": "__checkpoint__",
                    "tensors": 0,
                    "numel": 0,
                    "two_dim_tensors": 0,
                    "status": "FAIL",
                    "note": "base checkpoint not found",
                }
            )
            continue
        if not summary_path.is_file():
            rows.append(
                {
                    "base_ckpt": base_s,
                    "summary_json": str(summary_path),
                    "group": "__summary__",
                    "tensors": 0,
                    "numel": 0,
                    "two_dim_tensors": 0,
                    "status": "FAIL",
                    "note": "summary JSON missing; run export_blip2_full_dense_state_dict.py",
                }
            )
            continue
        try:
            payload = json.loads(read_text(summary_path))
        except Exception as exc:
            rows.append(
                {
                    "base_ckpt": base_s,
                    "summary_json": str(summary_path),
                    "group": "__summary__",
                    "tensors": 0,
                    "numel": 0,
                    "two_dim_tensors": 0,
                    "status": "FAIL",
                    "note": "summary JSON parse error: %s" % exc,
                }
            )
            continue
        groups = payload.get("groups", {})
        if not isinstance(groups, dict):
            rows.append(
                {
                    "base_ckpt": base_s,
                    "summary_json": str(summary_path),
                    "group": "__summary__",
                    "tensors": 0,
                    "numel": 0,
                    "two_dim_tensors": 0,
                    "status": "FAIL",
                    "note": "summary JSON has no groups dictionary",
                }
            )
            continue
        for group in MASK_BASE_REQUIRED_GROUPS:
            info = groups.get(group, {})
            if not isinstance(info, dict):
                info = {}
            tensors = int(row_float(info, "tensors") or 0)
            numel = int(row_float(info, "numel") or 0)
            two_dim_tensors = int(row_float(info, "two_dim_tensors") or 0)
            rows.append(
                {
                    "base_ckpt": base_s,
                    "summary_json": str(summary_path),
                    "group": group,
                    "tensors": tensors,
                    "numel": numel,
                    "two_dim_tensors": two_dim_tensors,
                    "status": "PASS" if tensors > 0 else "FAIL",
                    "note": "required dense BLIP2-T5 group",
                }
            )

    write_csv(
        path,
        ["base_ckpt", "summary_json", "group", "tensors", "numel", "two_dim_tensors", "status", "note"],
        rows,
    )
    return rows


def dense_mask_base_quality(rows: Sequence[Dict[str, object]], mask_rows: Sequence[Dict[str, object]]) -> Tuple[bool, str]:
    dense_bases = sorted(
        {
            str(row.get("base_ckpt", "")).strip()
            for row in mask_rows
            if str(row.get("mask_inference", "")).strip() == "dense_base_nonzero_to_zero"
            and str(row.get("base_ckpt", "")).strip()
        }
    )
    if not mask_rows:
        return False, "no mask rows; dense base cannot be audited"
    if not dense_bases:
        return False, "no dense-base mask inference rows"
    if not rows:
        return False, "dense_mask_base_summary.csv has no rows"
    failures = [row for row in rows if str(row.get("status", "")).upper() != "PASS"]
    present = {
        (str(row.get("base_ckpt", "")).strip(), str(row.get("group", "")).strip())
        for row in rows
        if str(row.get("status", "")).upper() == "PASS"
    }
    missing = [
        "%s:%s" % (base, group)
        for base in dense_bases
        for group in MASK_BASE_REQUIRED_GROUPS
        if (base, group) not in present
    ]
    if failures or missing:
        notes = []
        if failures:
            notes.append("fail_rows=%d" % len(failures))
        if missing:
            notes.append("missing=%s" % ",".join(missing[:8]))
        return False, "; ".join(notes)
    return True, "dense bases=%d; required groups=%s" % (len(dense_bases), ",".join(MASK_BASE_REQUIRED_GROUPS))


def build_validation_manifest(
    out_dir: Path,
    evidence: Sequence[Evidence],
    tests: Sequence[Tuple[str, bool, str]],
    log_rows: Sequence[Dict[str, str]],
    mask_rows: Sequence[Dict[str, object]],
    module_mask_rows: Sequence[Dict[str, object]],
    sparsity_rows: Sequence[Dict[str, str]],
    token_mask_rows: Sequence[Dict[str, str]],
    calibration_batch_trace_rows: Sequence[Dict[str, str]],
    importance_rows: Sequence[Dict[str, str]],
    selected_query_rows: Sequence[Dict[str, str]],
    eval_rows: int,
    eval_provenance_rows: Sequence[Dict[str, str]],
    prune_provenance_rows: Sequence[Dict[str, str]],
    has_importance_plot: bool,
    has_scaler_plot: bool,
    has_selected_query_plot: bool,
    alpha_summary: Sequence[Dict[str, object]],
    alpha_monotonic: Optional[bool],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], bool]:
    required_alphas = [float(x) for x in split_csv_arg(args.required_alphas)]
    required_benchmarks = split_csv_arg(args.required_benchmarks)
    required_methods = split_csv_arg(args.required_eval_methods)
    required_eval_atv_alphas = [float(x) for x in split_csv_arg(args.required_eval_atv_alphas)]
    required_eval_seeds = split_csv_arg(args.required_eval_seeds)

    static_ok = all(e.ok for e in evidence)
    unit_ok = all(ok for _, ok, _ in tests)
    alpha_grid_ok, alpha_note = alpha_coverage_ok(alpha_summary, required_alphas)
    alpha_layer_ok, alpha_layer_note = alpha_layer_coverage_quality(
        alpha_summary,
        required_alphas,
        args.expected_token_mask_layers,
    )
    alpha_num_img_ok, alpha_num_img_note = alpha_num_img_quality(log_rows, args.expected_alpha_num_img)
    alpha_k_zero_ok, alpha_k_zero_note = alpha_k_zero_quality(log_rows)
    alpha_zero_rows = alpha_log_rows(log_rows, 0.0)
    alpha_zero_ok, alpha_zero_note = alpha_zero_text_only_quality(log_rows)
    alpha_ok = (
        bool(log_rows)
        and alpha_grid_ok
        and alpha_layer_ok
        and (alpha_monotonic is not False)
        and alpha_num_img_ok
        and alpha_k_zero_ok
    )
    mask_layers = {str(r.get("layer", "")) for r in mask_rows if str(r.get("layer", ""))}
    mask_provenance_ok, mask_provenance_note = mask_evidence_quality(mask_rows, "layer", args)
    mask_ok = len(mask_rows) > 0 and len(mask_layers) >= args.min_mask_layers and mask_provenance_ok
    module_names = {str(r.get("module", "")) for r in module_mask_rows if str(r.get("module", ""))}
    module_mask_provenance_ok, module_mask_provenance_note = mask_evidence_quality(module_mask_rows, "module", args)
    module_mask_ok = len(module_names) >= args.min_mask_modules and module_mask_provenance_ok
    required_mask_pairs_ok, required_mask_pairs_note, required_mask_pairs_count = required_mask_pair_quality(
        mask_rows,
        module_mask_rows,
        args,
    )
    atv_specific_ok, atv_specific_note = atv_wanda_specificity_quality(mask_rows, alpha_summary, args)
    repro_ok, repro_note = reproducibility_quality(mask_rows, args)
    alpha_mask_ok, alpha_mask_note = alpha_mask_trend_quality(mask_rows, args)
    sparsity_ok, sparsity_note = sparsity_quality(sparsity_rows, args)
    importance_ok, importance_note, importance_stats = importance_quality(importance_rows, args)
    importance_plot_ok = has_importance_plot and importance_ok
    scaler_plot_ok = has_scaler_plot and importance_ok
    token_ok, token_problems, token_stats = token_mask_quality(
        token_mask_rows,
        args.expected_token_mask_samples,
        args.expected_token_mask_layers,
    )
    calib_trace_ok, calib_trace_note, calib_trace_stats = calibration_batch_trace_quality(
        calibration_batch_trace_rows,
        args.expected_token_mask_samples,
    )
    selected_query_ok, selected_query_note, selected_query_stats = selected_query_quality(
        selected_query_rows,
        args.min_selected_query_layers,
        args.expected_alpha_num_img,
        token_mask_rows,
    )
    selected_query_plot_ok = has_selected_query_plot and selected_query_ok
    eval_csv = out_dir / "eval_results.csv"
    eval_result_rows = read_csv_rows(eval_csv)
    eval_ok, eval_note, missing_bench, missing_methods, missing_cells = eval_coverage(
        eval_csv,
        required_benchmarks,
        required_methods,
    )
    eval_meta_ok, eval_meta_note = eval_metadata_quality(eval_result_rows, args)
    eval_atv_alpha_ok, eval_atv_alpha_note = eval_atv_alpha_quality(
        eval_result_rows,
        required_benchmarks,
        required_eval_atv_alphas,
    )
    eval_seed_ok, eval_seed_note = eval_seed_quality(
        eval_result_rows,
        required_benchmarks,
        required_methods,
        required_eval_seeds,
        required_eval_atv_alphas,
    )
    eval_prov_ok, eval_prov_note = eval_provenance_quality(
        eval_provenance_rows,
        required_methods,
        required_eval_seeds,
        required_eval_atv_alphas,
        args.allow_shared_eval_ckpts,
    )
    prune_prov_ok, prune_prov_note = prune_provenance_quality(prune_provenance_rows, args)
    runtime_env_ok, runtime_env_note, runtime_env_count = runtime_environment_quality(
        out_dir / "runtime_environment.json"
    )
    dense_mask_base_rows = read_csv_rows(out_dir / "dense_mask_base_summary.csv")
    dense_mask_base_ok, dense_mask_base_note = dense_mask_base_quality(dense_mask_base_rows, mask_rows)
    query_token_rows = read_csv_rows(out_dir / "query_token_mapping.csv")
    query_token_ok, query_token_note = query_token_mapping_quality(query_token_rows)
    if eval_rows == 0:
        eval_ok = False
    eval_ok = eval_ok and eval_meta_ok and eval_atv_alpha_ok and eval_seed_ok

    rows = [
        {
            "artifact": "static_mapping.md",
            "requirement": "static source mechanism mapping",
            "required_for_strict": "yes",
            "evidence_count": len(evidence),
            "status": "PASS" if static_ok else "FAIL",
            "note": "%d/%d static checks passed" % (sum(e.ok for e in evidence), len(evidence)),
        },
        {
            "artifact": "unit_test_results.txt",
            "requirement": "synthetic ATV golden tests",
            "required_for_strict": "yes",
            "evidence_count": len(tests),
            "status": "PASS" if unit_ok else "FAIL",
            "note": "%d/%d golden tests passed" % (sum(ok for _, ok, _ in tests), len(tests)),
        },
        {
            "artifact": "runtime_environment.json",
            "requirement": "runtime environment and source provenance",
            "required_for_strict": "yes",
            "evidence_count": runtime_env_count,
            "status": "PASS" if runtime_env_ok else "FAIL",
            "note": runtime_env_note,
        },
        {
            "artifact": "dense_mask_base_summary.csv",
            "requirement": "dense full checkpoint mask-base provenance",
            "required_for_strict": "yes",
            "evidence_count": len(dense_mask_base_rows),
            "status": "PASS" if dense_mask_base_ok else "FAIL",
            "note": dense_mask_base_note,
        },
        {
            "artifact": "query_token_mapping.csv",
            "requirement": "BLIP2-T5 Q-Former query-token count mapping",
            "required_for_strict": "yes",
            "evidence_count": len(query_token_rows),
            "status": "PASS" if query_token_ok else "FAIL",
            "note": query_token_note,
        },
        {
            "artifact": "alpha_sweep.csv",
            "requirement": "runtime alpha sweep behavior",
            "required_for_strict": "yes",
            "evidence_count": len(log_rows),
            "status": "PASS" if alpha_ok else "FAIL",
            "note": alpha_note
            + "; "
            + alpha_layer_note
            + "; "
            + alpha_num_img_note
            + "; "
            + alpha_k_zero_note
            + ("; monotonic=%s" % ("unknown" if alpha_monotonic is None else str(alpha_monotonic).lower())),
        },
        {
            "artifact": "alpha_sweep.csv",
            "requirement": "runtime k==0 ratio evidence",
            "required_for_strict": "yes",
            "evidence_count": len(log_rows),
            "status": "PASS" if alpha_k_zero_ok else "FAIL",
            "note": alpha_k_zero_note,
        },
        {
            "artifact": "alpha_sweep.csv",
            "requirement": "alpha=0 text-only token selection behavior",
            "required_for_strict": "yes",
            "evidence_count": len(alpha_zero_rows),
            "status": "PASS" if alpha_zero_ok else "FAIL",
            "note": alpha_zero_note,
        },
        {
            "artifact": "mask_iou_by_layer.csv",
            "requirement": "layer-wise mask difference evidence",
            "required_for_strict": "yes",
            "evidence_count": len(mask_rows),
            "status": "PASS" if mask_ok else "FAIL",
            "note": "covered layers=%d, required>=%d; %s"
            % (len(mask_layers), args.min_mask_layers, mask_provenance_note),
        },
        {
            "artifact": "mask_iou_by_module.csv",
            "requirement": "module-wise linear mask difference evidence",
            "required_for_strict": "yes",
            "evidence_count": len(module_mask_rows),
            "status": "PASS" if module_mask_ok else "FAIL",
            "note": "covered modules=%d, required>=%d; %s"
            % (len(module_names), args.min_mask_modules, module_mask_provenance_note),
        },
        {
            "artifact": "mask_iou_by_layer.csv; mask_iou_by_module.csv",
            "requirement": "required method mask-pair coverage",
            "required_for_strict": "yes",
            "evidence_count": required_mask_pairs_count,
            "status": "PASS" if required_mask_pairs_ok else "FAIL",
            "note": required_mask_pairs_note,
        },
        {
            "artifact": "mask_iou_by_layer.csv",
            "requirement": "ATV alpha=1 specificity versus naive Wanda",
            "required_for_strict": "yes",
            "evidence_count": len(mask_rows_for_pair(mask_rows, args.atv_wanda_pair_name)),
            "status": "PASS" if atv_specific_ok else "FAIL",
            "note": atv_specific_note,
        },
        {
            "artifact": "mask_iou_by_layer.csv",
            "requirement": "same-seed ATV mask reproducibility",
            "required_for_strict": "yes",
            "evidence_count": len(mask_rows_for_pair(mask_rows, args.repro_pair_name)),
            "status": "PASS" if repro_ok else "FAIL",
            "note": repro_note,
        },
        {
            "artifact": "mask_iou_by_layer.csv",
            "requirement": "alpha sweep mask trend toward naive Wanda",
            "required_for_strict": "yes",
            "evidence_count": len(alpha_mask_similarity_summary(mask_rows)),
            "status": "PASS" if alpha_mask_ok else "FAIL",
            "note": alpha_mask_note,
        },
        {
            "artifact": "sparsity_summary.csv",
            "requirement": "T5-only pruning scope and target sparsity",
            "required_for_strict": "yes",
            "evidence_count": len(sparsity_rows),
            "status": "PASS" if sparsity_ok else "FAIL",
            "note": sparsity_note,
        },
        {
            "artifact": "token_mask_integrity.csv",
            "requirement": "BLIP2 query/text token mask integrity",
            "required_for_strict": "yes",
            "evidence_count": len(token_mask_rows),
            "status": "PASS" if token_ok else "FAIL",
            "note": (
                "; ".join(token_problems)
                if token_problems
                else (
                    "rows=%d, samples=%d, layers=%d; "
                    "seq_len=32+text_len, attention/padding evidence, and selected_k valid"
                )
                % (token_stats["rows"], token_stats["samples"], token_stats["layers"])
            ),
        },
        {
            "artifact": "calibration_batch_trace.csv",
            "requirement": "calibration physical sample and batch trace",
            "required_for_strict": "yes",
            "evidence_count": calib_trace_stats["rows"],
            "status": "PASS" if calib_trace_ok else "FAIL",
            "note": calib_trace_note,
        },
        {
            "artifact": "importance_distribution.png",
            "requirement": "importance score distribution plot and numeric CSV quality",
            "required_for_strict": "yes",
            "evidence_count": importance_stats["rows"],
            "status": "PASS" if importance_plot_ok else "FAIL",
            "note": ("real plot; " if has_importance_plot else "placeholder only; ") + importance_note,
        },
        {
            "artifact": "scaler_row_distribution.png",
            "requirement": "scaler_row distribution plot and numeric CSV quality",
            "required_for_strict": "yes",
            "evidence_count": importance_stats["rows"],
            "status": "PASS" if scaler_plot_ok else "FAIL",
            "note": ("real plot; " if has_scaler_plot else "placeholder only; ") + importance_note,
        },
        {
            "artifact": "selected_query_token_frequency.png",
            "requirement": "selected query token frequency plot, index bounds, and selected_k consistency",
            "required_for_strict": "yes",
            "evidence_count": selected_query_stats["rows"],
            "status": "PASS" if selected_query_plot_ok else "FAIL",
            "note": (
                ("real plot; " if has_selected_query_plot else "placeholder only; ")
                + selected_query_note
            ),
        },
        {
            "artifact": "eval_results.csv",
            "requirement": "downstream benchmark coverage",
            "required_for_strict": "yes",
            "evidence_count": eval_rows,
            "status": "PASS" if eval_ok else "FAIL",
            "note": eval_note
            + "; "
            + eval_meta_note
            + "; "
            + eval_atv_alpha_note
            + "; "
            + eval_seed_note
            + ("; missing_benchmarks=%s" % ",".join(missing_bench) if missing_bench else "")
            + ("; missing_methods=%s" % ",".join(missing_methods) if missing_methods else "")
            + ("; missing_cells=%s" % ",".join(missing_cells[:16]) if missing_cells else ""),
        },
        {
            "artifact": "eval_provenance.csv",
            "requirement": "checkpoint and raw eval provenance",
            "required_for_strict": "yes",
            "evidence_count": len(eval_provenance_rows),
            "status": "PASS" if eval_prov_ok else "FAIL",
            "note": eval_prov_note,
        },
        {
            "artifact": "prune_provenance.csv",
            "requirement": "calibration and pruning provenance consistency",
            "required_for_strict": "yes",
            "evidence_count": len(prune_provenance_rows),
            "status": "PASS" if prune_prov_ok else "FAIL",
            "note": prune_prov_note,
        },
    ]
    preflight_rows = read_csv_rows(out_dir / "atv_preflight_report.csv")
    if preflight_rows:
        preflight_failures = [
            r for r in preflight_rows
            if str(r.get("level", "")).lower() == "required" and str(r.get("status", "")).upper() == "FAIL"
        ]
        preflight_warnings = [r for r in preflight_rows if str(r.get("status", "")).upper() == "WARN"]
        rows.append(
            {
                "artifact": "atv_preflight_report.csv",
                "requirement": "filesystem and input preflight",
                "required_for_strict": "yes",
                "evidence_count": len(preflight_rows),
                "status": "PASS" if not preflight_failures else "FAIL",
                "note": "required_failures=%d; warnings=%d" % (len(preflight_failures), len(preflight_warnings)),
            }
        )
    else:
        rows.append(
            {
                "artifact": "atv_preflight_report.csv",
                "requirement": "filesystem and input preflight",
                "required_for_strict": "yes",
                "evidence_count": 0,
                "status": "FAIL",
                "note": "missing; use run_atv_multiseed_validation.sh with RUN_PREFLIGHT=1 before final validation",
            }
        )
    strict_ok = all(row["status"] == "PASS" for row in rows if row["required_for_strict"] == "yes")
    return rows, strict_ok


def write_validation_manifest(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    write_csv(
        path,
        ["artifact", "requirement", "required_for_strict", "evidence_count", "status", "note"],
        rows,
    )


def write_traceability_matrix(path: Path, manifest_rows: Sequence[Dict[str, object]]) -> int:
    manifest_by_requirement = {
        str(row.get("requirement", "")): dict(row)
        for row in manifest_rows
    }
    trace_specs = [
        (
            "mechanism migration",
            "original ATV token selection, activation accumulation, Wanda importance, and mask generation are preserved",
            "static source mechanism mapping",
            "static_mapping.md",
            "Source fragments from original ATV and BLIP2-T5 migrated code must be present.",
        ),
        (
            "mechanism migration",
            "ATV math handles k=0, 0<k<num_img, k=num_img, variable text lengths, batching, precision, and padding",
            "synthetic ATV golden tests",
            "unit_test_results.txt",
            "Synthetic tests must pass without relying on real model runtime.",
        ),
        (
            "architecture mapping",
            "BLIP2-T5 visual tokens for ATV are 32 Q-Former query tokens, not raw ViT patches",
            "BLIP2-T5 Q-Former query-token count mapping",
            "query_token_mapping.csv",
            "Constructor/config/query parameter checks must all prove query-token count is 32.",
        ),
        (
            "architecture mapping",
            "T5 input layout is query-prefix followed by text tokens with correct attention and padding handling",
            "BLIP2 query/text token mask integrity",
            "token_mask_integrity.csv",
            "Runtime rows must prove per-sample query prefix, text suffix, attention mask, and padding counts.",
        ),
        (
            "architecture mapping",
            "calibration caching preserves physical sample counts rather than flattening or confusing batches",
            "calibration physical sample and batch trace",
            "calibration_batch_trace.csv",
            "Batch trace must prove cached batch sizes, cumulative physical sample count, 32 query tokens, and valid text/padding layout.",
        ),
        (
            "token selection behavior",
            "selected query token indices stay inside [0,31] and exactly match token_mask selected_k where k>0",
            "selected query token frequency plot, index bounds, and selected_k consistency",
            "selected_query_frequency.csv; selected_query_token_frequency.png",
            "Selected query evidence must include positive counts with in-range query indices and per-layer/sample counts matching selected_k.",
        ),
        (
            "alpha behavior",
            "alpha sweep covers 0,0.25,0.5,1,2,4 and every logged layer reports num_img=32",
            "runtime alpha sweep behavior",
            "alpha_sweep.csv",
            "Alpha logs must cover the required grid, expected layers, monotonic k trend, k==0 ratios, and per-sample num_img=32.",
        ),
        (
            "alpha behavior",
            "each alpha log row records the fraction of samples with k==0",
            "runtime k==0 ratio evidence",
            "alpha_sweep.csv",
            "Every alpha log row must include numeric kzero/kzero_n/k_zero_rate fields.",
        ),
        (
            "alpha behavior",
            "alpha=0 keeps no query tokens, proving the text-only negative-control path",
            "alpha=0 text-only token selection behavior",
            "alpha_sweep.csv",
            "Every alpha=0 log row must report kmean/kmin/kmax as zero.",
        ),
        (
            "mask behavior",
            "ATV alpha=1 is not silently identical to naive Wanda unless degeneracy logs explain it",
            "ATV alpha=1 specificity versus naive Wanda",
            "mask_iou_by_layer.csv",
            "ATV-vs-Wanda mask IoU must show specificity or a logged degeneracy explanation.",
        ),
        (
            "mask behavior",
            "high alpha moves toward naive Wanda relative to low alpha",
            "alpha sweep mask trend toward naive Wanda",
            "mask_iou_by_layer.csv",
            "Alpha-to-Wanda mask similarity must satisfy the expected trend.",
        ),
        (
            "mask behavior",
            "same-seed ATV rerun is mask-identical",
            "same-seed ATV mask reproducibility",
            "mask_iou_by_layer.csv",
            "Same-seed rerun keep/prune IoU must pass the reproducibility threshold.",
        ),
        (
            "mask behavior",
            "layer-wise T5 mask differences are available across enough layers",
            "layer-wise mask difference evidence",
            "mask_iou_by_layer.csv",
            "Layer mask evidence must cover the configured minimum number of T5 layers and infer masks from a dense base checkpoint.",
        ),
        (
            "mask behavior",
            "module-wise linear mask differences are available across target modules",
            "module-wise linear mask difference evidence",
            "mask_iou_by_module.csv",
            "Module mask evidence must cover the configured minimum number of linear modules and infer masks from a dense base checkpoint.",
        ),
        (
            "mask behavior",
            "mask-level comparisons cover ATV alpha ablations, naive Wanda, and TAMP/AMIA under dense-base inference",
            "required method mask-pair coverage",
            "mask_iou_by_layer.csv; mask_iou_by_module.csv",
            "Required mask-pair names must exist at both layer and module granularity with enough T5 coverage.",
        ),
        (
            "mask behavior",
            "dense base checkpoint used for mask inference is a full BLIP2-T5 state_dict",
            "dense full checkpoint mask-base provenance",
            "dense_mask_base_summary.csv",
            "Dense mask-base summary must prove T5, ViT, Q-Former, query token, and projection tensors are present.",
        ),
        (
            "scope and sparsity",
            "T5-only ATV reaches target T5 sparsity while visual and bridge modules remain dense",
            "T5-only pruning scope and target sparsity",
            "sparsity_summary.csv",
            "Sparsity rows must prove T5 target sparsity and near-zero visual_encoder, Qformer/query-token, and t5_proj sparsity.",
        ),
        (
            "importance evidence",
            "importance score distribution is numeric and not a placeholder",
            "importance score distribution plot and numeric CSV quality",
            "importance_distribution.csv; importance_distribution.png",
            "Importance CSV must contain finite Wanda/weight/mask statistics near target sparsity.",
        ),
        (
            "importance evidence",
            "scaler_row distribution is numeric and not a placeholder",
            "scaler_row distribution plot and numeric CSV quality",
            "importance_distribution.csv; scaler_row_distribution.png",
            "scaler_row CSV columns must be finite and plotted from real data.",
        ),
        (
            "runtime provenance",
            "Python/Torch/CUDA/git/source hashes are captured",
            "runtime environment and source provenance",
            "runtime_environment.json",
            "Runtime capture must include torch import, git commit, and source-role hashes.",
        ),
        (
            "runtime provenance",
            "filesystem and input preflight ran before expensive GPU jobs",
            "filesystem and input preflight",
            "atv_preflight_report.csv",
            "Preflight must prove calibration/eval/checkpoint paths and required fields exist.",
        ),
        (
            "runtime provenance",
            "ATV alpha ablations and naive Wanda use the same calibration/spec/scope across seeds",
            "calibration and pruning provenance consistency",
            "prune_provenance.csv",
            "Prune provenance must cover ATV alpha 0/1/4 and Wanda for seeds 42,43,44.",
        ),
        (
            "downstream evaluation",
            "dense, ATV, Wanda, and TAMP/AMIA cover MMBench, OKVQA, MMMU, and MathVista",
            "downstream benchmark coverage",
            "eval_results.csv; eval_summary_by_method.csv",
            "Eval rows must cover required methods, benchmarks, ATV alpha ablations, and pruned seeds.",
        ),
        (
            "downstream evaluation",
            "every eval cell is traceable to checkpoint path and raw metrics",
            "checkpoint and raw eval provenance",
            "eval_provenance.csv",
            "Eval provenance must map method/seed/alpha rows to checkpoints and raw eval files.",
        ),
    ]

    out_rows: List[Dict[str, object]] = []
    for area, objective_requirement, manifest_requirement, evidence_artifact, proof_standard in trace_specs:
        manifest = manifest_by_requirement.get(manifest_requirement, {})
        status = str(manifest.get("status", "MISSING"))
        out_rows.append(
            {
                "objective_area": area,
                "objective_requirement": objective_requirement,
                "manifest_requirement": manifest_requirement,
                "evidence_artifact": evidence_artifact,
                "current_status": status,
                "evidence_count": manifest.get("evidence_count", ""),
                "proof_standard": proof_standard,
                "manifest_note": manifest.get("note", "missing manifest row"),
            }
        )

    write_csv(
        path,
        [
            "objective_area",
            "objective_requirement",
            "manifest_requirement",
            "evidence_artifact",
            "current_status",
            "evidence_count",
            "proof_standard",
            "manifest_note",
        ],
        out_rows,
    )
    return len(out_rows)


def write_final_report(
    out_dir: Path,
    evidence: Sequence[Evidence],
    tests: Sequence[Tuple[str, bool, str]],
    log_rows: Sequence[Dict[str, str]],
    mask_rows: Sequence[Dict[str, object]],
    module_mask_rows: int,
    sparsity_rows: int,
    token_mask_rows: int,
    calibration_batch_trace_rows: int,
    eval_rows: int,
    eval_provenance_rows: int,
    prune_provenance_rows: int,
    bootstrap_rows: int,
    has_importance_plot: bool,
    has_scaler_plot: bool,
    has_selected_query_plot: bool,
    manifest_rows: Sequence[Dict[str, object]],
    strict_ok: bool,
) -> None:
    missing = []
    if not log_rows:
        missing.append("alpha_sweep.csv has only a header because no --atv_log was provided.")
    if not mask_rows:
        missing.append("mask_iou_by_layer.csv has only a header because no --mask_pair was provided.")
    if module_mask_rows == 0:
        missing.append("mask_iou_by_module.csv has only a header because no --mask_pair was provided.")
    if sparsity_rows == 0:
        missing.append("sparsity_summary.csv has only a header because no --sparsity_csv was provided.")
    if token_mask_rows == 0:
        missing.append("token_mask_integrity.csv has only a header because no --token_mask_csv was provided.")
    if calibration_batch_trace_rows == 0:
        missing.append(
            "calibration_batch_trace.csv has only a header because no "
            "--calibration_batch_trace_csv evidence was provided."
        )
    if eval_rows == 0:
        missing.append(
            "eval_results.csv has only a header because no --eval_csv, --metrics_jsonl, "
            "or --okvqa_eval_txt evidence was provided."
        )
    if eval_provenance_rows == 0:
        missing.append("eval_provenance.csv has only a header because no --eval_provenance_csv evidence was provided.")
    if prune_provenance_rows == 0:
        missing.append("prune_provenance.csv has only a header because no --prune_provenance_csv evidence was provided.")
    if bootstrap_rows == 0:
        missing.append(
            "paired_bootstrap_ci.csv has only a header because no usable --prediction_csv "
            "per-sample evidence was provided; this is optional unless per-sample predictions are available."
        )
    if not has_importance_plot:
        missing.append("importance_distribution.png is a placeholder because no usable --importance_csv was provided.")
    if not has_scaler_plot:
        missing.append("scaler_row_distribution.png is a placeholder because no usable scaler_row values were provided.")
    if not has_selected_query_plot:
        missing.append("selected_query_token_frequency.png is a placeholder because no usable --selected_query_csv was provided.")

    alpha_summary, alpha_monotonic = summarize_alpha_logs(log_rows)

    path = out_dir / "final_validation_report.md"
    with path.open("w", encoding="utf-8") as f:
        f.write("# ATV Migration Validation Report\n\n")
        f.write("## Current Evidence Summary\n\n")
        f.write("- Static source checks: %d / %d passed.\n" % (sum(e.ok for e in evidence), len(evidence)))
        f.write("- Golden unit tests: %d / %d passed.\n" % (sum(ok for _, ok, _ in tests), len(tests)))
        f.write("- Parsed ATV log rows: %d.\n" % len(log_rows))
        f.write("- Mask IoU rows: %d.\n" % len(mask_rows))
        f.write("- Module mask IoU rows: %d.\n" % module_mask_rows)
        f.write("- Sparsity summary rows: %d.\n" % sparsity_rows)
        f.write("- Token mask integrity rows: %d.\n" % token_mask_rows)
        f.write("- Calibration batch trace rows: %d.\n" % calibration_batch_trace_rows)
        f.write("- Eval result rows: %d.\n" % eval_rows)
        f.write("- Eval provenance rows: %d.\n" % eval_provenance_rows)
        f.write("- Pruning provenance rows: %d.\n" % prune_provenance_rows)
        f.write("- Paired bootstrap CI rows: %d.\n" % bootstrap_rows)
        f.write("- Importance distribution plot: %s.\n" % ("real" if has_importance_plot else "placeholder"))
        f.write("- scaler_row distribution plot: %s.\n" % ("real" if has_scaler_plot else "placeholder"))
        f.write("- Selected query-token frequency plot: %s.\n" % ("real" if has_selected_query_plot else "placeholder"))
        f.write("\n## Evidence Gate Verdict\n\n")
        f.write("- Strict validation readiness: %s.\n" % ("PASS" if strict_ok else "INCOMPLETE"))
        f.write("- Machine-readable checklist: `validation_manifest.csv`.\n\n")
        f.write("- Objective-to-evidence traceability: `validation_traceability.csv`.\n\n")
        f.write("| Artifact | Requirement | Strict | Count | Status | Note |\n")
        f.write("|---|---|---|---:|---|---|\n")
        for row in manifest_rows:
            note = str(row.get("note", "")).replace("|", "\\|")
            f.write(
                "| %s | %s | %s | %s | %s | %s |\n"
                % (
                    row.get("artifact", ""),
                    row.get("requirement", ""),
                    row.get("required_for_strict", ""),
                    row.get("evidence_count", ""),
                    row.get("status", ""),
                    note,
                )
            )

        manifest_by_requirement = {
            str(row.get("requirement", "")): str(row.get("status", ""))
            for row in manifest_rows
        }
        f.write("\n## Claim-To-Evidence Matrix\n\n")
        f.write(
            "These claims are the minimum scientific claims needed to argue that "
            "the original ATV mechanism was migrated correctly rather than merely "
            "made runnable.\n\n"
        )
        f.write("| Claim | Required evidence | Current status |\n")
        f.write("|---|---|---|\n")
        claim_rows = [
            (
                "Source-level mechanism equivalence",
                "static source mechanism mapping; synthetic ATV golden tests",
                "%s / %s"
                % (
                    manifest_by_requirement.get("static source mechanism mapping", "MISSING"),
                    manifest_by_requirement.get("synthetic ATV golden tests", "MISSING"),
                ),
            ),
            (
                "BLIP2-T5 architecture mapping is valid",
                "BLIP2-T5 Q-Former query-token count mapping; BLIP2 query/text token mask integrity; calibration physical sample and batch trace; selected query token frequency plot and index bounds",
                "%s / %s / %s / %s"
                % (
                    manifest_by_requirement.get("BLIP2-T5 Q-Former query-token count mapping", "MISSING"),
                    manifest_by_requirement.get("BLIP2 query/text token mask integrity", "MISSING"),
                    manifest_by_requirement.get("calibration physical sample and batch trace", "MISSING"),
                    manifest_by_requirement.get("selected query token frequency plot and index bounds", "MISSING"),
                ),
            ),
            (
                "ATV alpha behavior matches the token-selection formula",
                "runtime alpha sweep behavior; runtime k==0 ratio evidence; alpha=0 text-only token selection behavior; alpha sweep mask trend toward naive Wanda",
                "%s / %s / %s / %s"
                % (
                    manifest_by_requirement.get("runtime alpha sweep behavior", "MISSING"),
                    manifest_by_requirement.get("runtime k==0 ratio evidence", "MISSING"),
                    manifest_by_requirement.get("alpha=0 text-only token selection behavior", "MISSING"),
                    manifest_by_requirement.get("alpha sweep mask trend toward naive Wanda", "MISSING"),
                ),
            ),
            (
                "ATV is not silently identical to naive Wanda",
                "required method mask-pair coverage; ATV alpha=1 specificity versus naive Wanda; alpha sweep mask trend toward naive Wanda",
                "%s / %s / %s"
                % (
                    manifest_by_requirement.get("required method mask-pair coverage", "MISSING"),
                    manifest_by_requirement.get("ATV alpha=1 specificity versus naive Wanda", "MISSING"),
                    manifest_by_requirement.get("alpha sweep mask trend toward naive Wanda", "MISSING"),
                ),
            ),
            (
                "Pruning scope is controlled",
                "T5-only pruning scope and target sparsity; layer-wise mask difference evidence",
                "%s / %s"
                % (
                    manifest_by_requirement.get("T5-only pruning scope and target sparsity", "MISSING"),
                    manifest_by_requirement.get("layer-wise mask difference evidence", "MISSING"),
                ),
            ),
            (
                "Results are reproducible and fairly evaluated",
                "filesystem and input preflight; calibration and pruning provenance consistency; runtime environment and source provenance; same-seed ATV mask reproducibility; downstream benchmark coverage; checkpoint and raw eval provenance",
                "%s / %s / %s / %s / %s / %s"
                % (
                    manifest_by_requirement.get("filesystem and input preflight", "MISSING"),
                    manifest_by_requirement.get("calibration and pruning provenance consistency", "MISSING"),
                    manifest_by_requirement.get("runtime environment and source provenance", "MISSING"),
                    manifest_by_requirement.get("same-seed ATV mask reproducibility", "MISSING"),
                    manifest_by_requirement.get("downstream benchmark coverage", "MISSING"),
                    manifest_by_requirement.get("checkpoint and raw eval provenance", "MISSING"),
                ),
            ),
        ]
        for claim, required, status in claim_rows:
            f.write("| %s | %s | %s |\n" % (claim, required, status))

        if alpha_summary:
            f.write("\n## Alpha Sweep Behavior\n\n")
            f.write("| tag | alpha | layers | mean k | mean k==0 rate | mean degenerate rate |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for row in alpha_summary:
                alpha = row["alpha"]
                alpha_s = "" if math.isnan(float(alpha)) else "%.4g" % float(alpha)
                zero_rate = row.get("mean_k_zero_rate", float("nan"))
                zero_s = "" if math.isnan(float(zero_rate)) else "%.4f" % float(zero_rate)
                f.write(
                    "| %s | %s | %d | %.4f | %s | %.4f |\n"
                    % (
                        row["tag"],
                        alpha_s,
                        int(row["layers"]),
                        float(row["mean_kmean"]),
                        zero_s,
                        float(row["mean_degenerate_rate"]),
                    )
                )
            if alpha_monotonic is not None:
                f.write("\n- Mean selected-k monotonic with alpha: %s.\n" % ("yes" if alpha_monotonic else "no"))
        if missing:
            f.write("\n## Missing GPU Evidence\n\n")
            for item in missing:
                f.write("- %s\n" % item)
        f.write("\n## Interpretation Guardrails\n\n")
        f.write("- BLIP2-T5 ATV visual tokens mean the 32 Q-Former query tokens injected into T5, not raw ViT patches.\n")
        f.write("- A high alpha can degenerate ATV into naive Wanda if k reaches the number of query tokens.\n")
        f.write("- Correct migration requires mechanism agreement and reproducible mask behavior; benchmark gains alone are not proof.\n")
        f.write("- A downstream score gain without token-mask, alpha, sparsity, and mask-IoU evidence is not enough to validate migration.\n")
        f.write("- A downstream score drop does not by itself disprove migration if mechanism gates pass; it indicates the migrated method is not advantageous under that setting.\n")
        f.write("- Final paper tables should report dense, naive Wanda, TAMP/AMIA, ATV alpha=1, and ATV alpha ablations under identical calibration, sparsity, and evaluation settings.\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate ATV-Pruning migration evidence for BLIP2-T5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--original_atv_root", required=True, type=Path)
    p.add_argument("--lavis_root", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--atv_log", action="append", default=[], help="Optional TAG=path ATV pruning log; may be repeated.")
    p.add_argument("--mask_pair", nargs=3, action="append", default=[], metavar=("NAME", "A", "B"))
    p.add_argument("--mask_prefix", default="t5_model.encoder")
    p.add_argument("--min_numel", type=int, default=4096)
    p.add_argument(
        "--mask_base_ckpt",
        type=Path,
        default=None,
        help=(
            "Full dense checkpoint used to infer pruning masks as dense-nonzero -> pruned-zero. "
            "Strict validation requires this unless --allow_zero_only_mask_inference is set."
        ),
    )
    p.add_argument(
        "--allow_zero_only_mask_inference",
        action="store_true",
        help="Allow mask IoU from raw weight==0 without a dense base checkpoint; intended only for debugging.",
    )
    p.add_argument(
        "--token_mask_csv",
        type=Path,
        default=None,
        help="Optional precomputed token mask integrity CSV to copy into the report.",
    )
    p.add_argument(
        "--calibration_batch_trace_csv",
        type=Path,
        default=None,
        help="Optional CSV proving cached calibration batches and physical sample counts.",
    )
    p.add_argument(
        "--sparsity_csv",
        type=Path,
        default=None,
        help="Optional checkpoint sparsity CSV from check_ckpt_sparsity.py.",
    )
    p.add_argument(
        "--importance_csv",
        type=Path,
        default=None,
        help="Optional CSV with numeric importance/w_metric values for importance_distribution.png.",
    )
    p.add_argument(
        "--selected_query_csv",
        type=Path,
        default=None,
        help="Optional CSV with query_index,count rows for selected_query_token_frequency.png.",
    )
    p.add_argument(
        "--eval_csv",
        type=Path,
        default=None,
        help="Optional benchmark result CSV to copy into eval_results.csv.",
    )
    p.add_argument(
        "--metrics_jsonl",
        type=Path,
        action="append",
        default=[],
        help="Optional LAVIS metrics JSONL from MMBench/MMMU/MathVista; may be repeated.",
    )
    p.add_argument(
        "--okvqa_eval_txt",
        action="append",
        default=[],
        metavar="TAG=PATH",
        help="Optional OKVQA evaluate.txt, optionally tagged as calibration=path; may be repeated.",
    )
    p.add_argument(
        "--eval_provenance_csv",
        type=Path,
        action="append",
        default=[],
        help="Optional CSV mapping eval method/seed/alpha rows to checkpoints and raw eval files.",
    )
    p.add_argument(
        "--prune_provenance_csv",
        type=Path,
        default=None,
        help="Optional CSV mapping pruned checkpoints to calibration paths, alpha, seed, and sparsity settings.",
    )
    p.add_argument(
        "--prediction_csv",
        action="append",
        default=[],
        metavar="TAG=PATH",
        help=(
            "Optional per-sample prediction CSV for paired bootstrap CI. "
            "Rows should include sample_id and correct/is_correct/score; may be repeated."
        ),
    )
    p.add_argument("--bootstrap_samples", type=int, default=2000)
    p.add_argument("--bootstrap_seed", type=int, default=12345)
    p.add_argument("--eval_method", default="atv", help="Method label used when parsing JSONL/evaluate.txt evidence.")
    p.add_argument("--eval_calibration", default="", help="Fallback calibration label for parsed eval evidence.")
    p.add_argument("--eval_seed", default="", help="Seed label for parsed eval evidence.")
    p.add_argument("--eval_alpha", default="", help="Alpha label for parsed eval evidence.")
    p.add_argument("--eval_t5_sparsity", default="", help="T5 sparsity label for parsed eval evidence.")
    p.add_argument("--eval_vit_sparsity", default="", help="ViT sparsity label for parsed eval evidence.")
    p.add_argument(
        "--allow_shared_eval_ckpts",
        action="store_true",
        help="Allow the same pruned checkpoint path to satisfy multiple eval seeds in eval_provenance.csv.",
    )
    p.add_argument(
        "--preserve_existing",
        action="store_true",
        help="Keep existing report CSV/PNG artifacts when the corresponding new evidence is not supplied.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero unless all final validation evidence gates pass.",
    )
    p.add_argument(
        "--required_alphas",
        default="0,0.25,0.5,1,2,4",
        help="Comma-separated alpha values required by the strict alpha-sweep gate.",
    )
    p.add_argument(
        "--required_benchmarks",
        default="MMBench,OKVQA,MMMU,MathVista",
        help="Comma-separated benchmark names required by the strict eval gate.",
    )
    p.add_argument(
        "--required_eval_methods",
        default="dense,atv,wanda,tamp",
        help="Comma-separated method names required by the strict eval gate.",
    )
    p.add_argument(
        "--required_prune_methods",
        default="atv,wanda",
        help="Comma-separated pruned methods required by the strict pruning-provenance gate.",
    )
    p.add_argument(
        "--required_eval_atv_alphas",
        default="0,1,4",
        help="Comma-separated ATV alpha values required in eval_results.csv for strict eval ablation.",
    )
    p.add_argument(
        "--required_eval_seeds",
        default="42,43,44",
        help="Comma-separated seeds required in eval_results.csv for pruned-method stability evidence.",
    )
    p.add_argument(
        "--required_mask_pairs",
        default="atv_alpha0_vs_wanda,atv_alpha1_vs_wanda,atv_alpha4_vs_wanda,atv_alpha1_vs_tamp",
        help=(
            "Comma-separated mask-pair name substrings required at both layer and module granularity. "
            "Use this to prove ATV alpha ablations, naive Wanda, and TAMP/AMIA are included in mask analysis."
        ),
    )
    p.add_argument(
        "--expected_prune_num_data",
        type=int,
        default=128,
        help="Expected calibration sample count recorded in prune_provenance.csv.",
    )
    p.add_argument(
        "--min_mask_layers",
        type=int,
        default=20,
        help="Minimum distinct layers required in mask_iou_by_layer.csv for strict validation.",
    )
    p.add_argument(
        "--min_mask_modules",
        type=int,
        default=100,
        help="Minimum distinct linear modules required in mask_iou_by_module.csv for strict validation.",
    )
    p.add_argument(
        "--min_importance_rows",
        type=int,
        default=100,
        help="Minimum rows required in importance_distribution.csv for strict importance/scaler evidence.",
    )
    p.add_argument(
        "--atv_wanda_pair_name",
        default="atv_alpha1_vs_wanda",
        help="Substring identifying the ATV alpha=1 versus naive Wanda mask pair.",
    )
    p.add_argument(
        "--repro_pair_name",
        default="atv_repro",
        help="Substring identifying the same-seed ATV reproducibility mask pair; set empty to disable.",
    )
    p.add_argument(
        "--identical_iou_threshold",
        type=float,
        default=0.999999,
        help="Mean keep/prune IoU at or above this value is treated as mask-identical.",
    )
    p.add_argument(
        "--alpha1_degenerate_explain_threshold",
        type=float,
        default=0.95,
        help="If ATV and Wanda masks are identical, alpha=1 must have at least this degeneracy rate to explain it.",
    )
    p.add_argument(
        "--repro_iou_threshold",
        type=float,
        default=0.999999,
        help="Minimum per-layer keep/prune IoU for same-seed ATV reproducibility evidence.",
    )
    p.add_argument(
        "--min_alpha_mask_points",
        type=int,
        default=3,
        help="Minimum alpha-vs-Wanda mask pairs required for alpha mask trend validation.",
    )
    p.add_argument(
        "--alpha_mask_low",
        type=float,
        default=0.0,
        help="Low alpha endpoint used in the mask-trend validation.",
    )
    p.add_argument(
        "--alpha_mask_high",
        type=float,
        default=4.0,
        help="High alpha endpoint used in the mask-trend validation.",
    )
    p.add_argument(
        "--alpha_mask_trend_tol",
        type=float,
        default=1e-6,
        help="Tolerance when checking that high alpha is at least as Wanda-like as low alpha.",
    )
    p.add_argument(
        "--expected_token_mask_samples",
        type=int,
        default=128,
        help="Minimum unique calibration sample ids expected in token_mask_integrity.csv for strict validation.",
    )
    p.add_argument(
        "--expected_token_mask_layers",
        type=int,
        default=24,
        help="Minimum unique T5 encoder layers expected in token_mask_integrity.csv for strict validation.",
    )
    p.add_argument(
        "--expected_alpha_num_img",
        type=int,
        default=32,
        help="Expected per-sample query-token count reported as num_img in ATV alpha logs.",
    )
    p.add_argument(
        "--min_selected_query_layers",
        type=int,
        default=1,
        help=(
            "Minimum distinct layers expected in selected_query_frequency.csv. "
            "When token_mask_integrity.csv is present, every positive selected_k "
            "layer/sample pair must also match selected_query_frequency.csv exactly."
        ),
    )
    p.add_argument(
        "--expected_t5_sparsity",
        type=float,
        default=0.5,
        help="Expected T5 overall sparsity in sparsity_summary.csv for strict validation.",
    )
    p.add_argument(
        "--expected_vit_sparsity_max",
        type=float,
        default=0.01,
        help="Maximum allowed visual_encoder sparsity for T5-only ATV strict validation.",
    )
    p.add_argument(
        "--expected_non_t5_sparsity_max",
        type=float,
        default=0.01,
        help="Maximum allowed Q-Former/query-token/t5_proj sparsity for T5-only ATV strict validation.",
    )
    p.add_argument(
        "--sparsity_tol",
        type=float,
        default=0.05,
        help="Tolerance around expected T5 sparsity for strict validation.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    evidence = collect_static_evidence(args.original_atv_root, args.lavis_root)
    write_static_mapping(args.out_dir, evidence, args.lavis_root)
    write_query_token_mapping(args.out_dir, args.lavis_root)

    tests = run_golden_tests()
    write_unit_results(args.out_dir, tests)

    alpha_csv = args.out_dir / "alpha_sweep.csv"
    if args.atv_log or not (args.preserve_existing and alpha_csv.exists()):
        log_rows = parse_atv_logs(args.atv_log)
        write_csv(
            alpha_csv,
            [
                "tag",
                "layer",
                "cos",
                "kmean",
                "kmin",
                "kmax",
                "kzero",
                "kzero_n",
                "k_zero_rate",
                "num_img",
                "deg",
                "n",
                "degenerate_rate",
                "log",
            ],
            log_rows,
        )
    else:
        log_rows = read_csv_rows(alpha_csv)

    mask_csv = args.out_dir / "mask_iou_by_layer.csv"
    module_mask_csv = args.out_dir / "mask_iou_by_module.csv"
    if args.mask_pair or not (args.preserve_existing and mask_csv.exists()):
        mask_rows: List[Dict[str, object]] = []
        module_mask_rows: List[Dict[str, object]] = []
        for name, a_path, b_path in args.mask_pair:
            layer_part, module_part = compare_mask_pair(
                name,
                Path(a_path),
                Path(b_path),
                args.mask_prefix,
                args.min_numel,
                args.mask_base_ckpt,
            )
            mask_rows.extend(layer_part)
            module_mask_rows.extend(module_part)
        write_csv(
            mask_csv,
            [
                "pair",
                "a_ckpt",
                "b_ckpt",
                "base_ckpt",
                "mask_inference",
                "mask_prefix",
                "min_numel",
                "layer",
                "tensors",
                "numel",
                "prune_iou",
                "keep_iou",
                "a_prune_ratio",
                "b_prune_ratio",
                "a_keep_ratio",
                "b_keep_ratio",
                "diff_ratio",
            ],
            mask_rows,
        )
        write_csv(
            module_mask_csv,
            [
                "pair",
                "a_ckpt",
                "b_ckpt",
                "base_ckpt",
                "mask_inference",
                "mask_prefix",
                "min_numel",
                "module",
                "layer",
                "numel",
                "prune_iou",
                "keep_iou",
                "a_prune_ratio",
                "b_prune_ratio",
                "a_keep_ratio",
                "b_keep_ratio",
                "diff_ratio",
            ],
            module_mask_rows,
        )
    else:
        mask_rows = read_csv_rows(mask_csv)
        module_mask_rows = read_csv_rows(module_mask_csv)

    dense_mask_base_csv = args.out_dir / "dense_mask_base_summary.csv"
    write_dense_mask_base_summary(dense_mask_base_csv, mask_rows)

    sparsity_csv = args.out_dir / "sparsity_summary.csv"
    sparsity_count = copy_or_empty_csv(
        sparsity_csv,
        args.sparsity_csv,
        ["tag", "group", "zeros", "total", "tensors", "sparsity", "status", "note", "ckpt"],
        preserve_existing=args.preserve_existing,
    )
    sparsity_rows = read_csv_rows(sparsity_csv)

    token_mask_csv = args.out_dir / "token_mask_integrity.csv"
    token_mask_count = copy_or_empty_csv(
        token_mask_csv,
        args.token_mask_csv,
        [
            "sample_id",
            "layer",
            "seq_len",
            "expected_query_tokens",
            "num_query_tokens",
            "num_text_tokens",
            "query_prefix_true_count",
            "text_suffix_true_count",
            "attention_query_true_count",
            "valid_text_tokens",
            "pad_text_tokens",
            "attention_layout_ok",
            "query_prefix_ok",
            "selected_k",
            "selected_ratio",
            "is_degenerate",
            "source",
        ],
        preserve_existing=args.preserve_existing,
    )
    token_mask_rows = read_csv_rows(token_mask_csv)

    calibration_batch_trace_csv = args.out_dir / "calibration_batch_trace.csv"
    copy_or_empty_csv(
        calibration_batch_trace_csv,
        args.calibration_batch_trace_csv,
        [
            "source",
            "requested_samples",
            "cached_batches",
            "batch_index",
            "batch_size",
            "sample_start",
            "sample_end_exclusive",
            "physical_samples_seen",
            "seq_len",
            "expected_query_tokens",
            "query_prefix_true_min",
            "query_prefix_true_max",
            "total_query_tokens_min",
            "total_query_tokens_max",
            "valid_text_tokens_min",
            "valid_text_tokens_max",
            "pad_text_tokens_min",
            "pad_text_tokens_max",
            "query_prefix_ok_all",
            "attention_layout_ok_all",
        ],
        preserve_existing=args.preserve_existing,
    )
    calibration_batch_trace_rows = read_csv_rows(calibration_batch_trace_csv)

    eval_header = [
        "method",
        "calibration",
        "seed",
        "alpha",
        "t5_sparsity",
        "vit_sparsity",
        "benchmark",
        "score",
        "std",
        "notes",
    ]
    eval_csv_path = args.out_dir / "eval_results.csv"
    eval_rows = write_eval_results(eval_csv_path, args, eval_header)
    write_eval_summary(args.out_dir / "eval_summary_by_method.csv", read_csv_rows(eval_csv_path))
    eval_provenance_csv = args.out_dir / "eval_provenance.csv"
    eval_provenance_count = write_eval_provenance(
        eval_provenance_csv,
        args.eval_provenance_csv,
        args.preserve_existing,
    )
    eval_provenance_rows = read_csv_rows(eval_provenance_csv)
    prune_provenance_csv = args.out_dir / "prune_provenance.csv"
    copy_or_empty_csv(
        prune_provenance_csv,
        args.prune_provenance_csv,
        [
            "seed",
            "method",
            "role",
            "alpha",
            "job_id",
            "ckpt",
            "calib_cfg",
            "calib_json",
            "images_dir",
            "num_data",
            "t5_spec",
            "t5_sparsity_target",
            "vit_sparsity_target",
            "pruning_scope",
            "run_prune",
            "prune_log",
            "sparsity_csv",
        ],
        preserve_existing=args.preserve_existing,
    )
    prune_provenance_rows = read_csv_rows(prune_provenance_csv)
    bootstrap_rows = write_paired_bootstrap_ci(
        args.out_dir / "paired_bootstrap_ci.csv",
        read_prediction_rows(args.prediction_csv),
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    importance_csv = args.out_dir / "importance_distribution.csv"
    copy_or_empty_csv(
        importance_csv,
        args.importance_csv,
        [
            "source",
            "layer",
            "module",
            "tensor",
            "numel",
            "mean_wanda_importance",
            "median_wanda_importance",
            "max_wanda_importance",
            "scaler_row_mean",
            "scaler_row_max",
            "weight_abs_mean",
            "mask_sparsity",
        ],
        preserve_existing=args.preserve_existing,
    )
    importance_rows = read_csv_rows(importance_csv)
    importance_plot = args.out_dir / "importance_distribution.png"
    scaler_plot = args.out_dir / "scaler_row_distribution.png"
    selected_plot = args.out_dir / "selected_query_token_frequency.png"
    selected_query_csv = args.out_dir / "selected_query_frequency.csv"
    copy_or_empty_csv(
        selected_query_csv,
        args.selected_query_csv,
        ["layer", "sample_id", "query_index", "count", "source"],
        preserve_existing=args.preserve_existing,
    )
    selected_query_rows = read_csv_rows(selected_query_csv)
    if args.importance_csv is None and args.preserve_existing and importance_plot.exists():
        has_importance_plot = not importance_plot.with_suffix(importance_plot.suffix + ".txt").exists()
    else:
        has_importance_plot = plot_importance_distribution(importance_plot, importance_csv)
    if args.importance_csv is None and args.preserve_existing and scaler_plot.exists():
        has_scaler_plot = not scaler_plot.with_suffix(scaler_plot.suffix + ".txt").exists()
    else:
        has_scaler_plot = plot_scaler_distribution(scaler_plot, importance_csv)
    if args.selected_query_csv is None and args.preserve_existing and selected_plot.exists():
        has_selected_query_plot = not selected_plot.with_suffix(selected_plot.suffix + ".txt").exists()
    else:
        has_selected_query_plot = plot_selected_query_frequency(
            selected_plot,
            selected_query_csv,
        )

    alpha_summary, alpha_monotonic = summarize_alpha_logs(log_rows)
    manifest_rows, strict_ok = build_validation_manifest(
        args.out_dir,
        evidence,
        tests,
        log_rows,
        mask_rows,
        module_mask_rows,
        sparsity_rows,
        token_mask_rows,
        calibration_batch_trace_rows,
        importance_rows,
        selected_query_rows,
        eval_rows,
        eval_provenance_rows,
        prune_provenance_rows,
        has_importance_plot,
        has_scaler_plot,
        has_selected_query_plot,
        alpha_summary,
        alpha_monotonic,
        args,
    )
    write_validation_manifest(args.out_dir / "validation_manifest.csv", manifest_rows)
    write_traceability_matrix(args.out_dir / "validation_traceability.csv", manifest_rows)

    write_final_report(
        args.out_dir,
        evidence,
        tests,
        log_rows,
        mask_rows,
        len(module_mask_rows),
        sparsity_count,
        token_mask_count,
        len(calibration_batch_trace_rows),
        eval_rows,
        eval_provenance_count,
        len(prune_provenance_rows),
        bootstrap_rows,
        has_importance_plot,
        has_scaler_plot,
        has_selected_query_plot,
        manifest_rows,
        strict_ok,
    )

    static_ok = all(e.ok for e in evidence)
    tests_ok = all(ok for _, ok, _ in tests)
    status = "STRICT_PASS" if strict_ok else ("MECHANISM_PASS_GPU_INCOMPLETE" if static_ok and tests_ok else "CHECK")
    print("[OK] wrote ATV validation evidence to: %s" % args.out_dir)
    print("[RESULT] static=%s unit=%s strict=%s status=%s" % (static_ok, tests_ok, strict_ok, status))
    if args.strict and not strict_ok:
        print("[STRICT] final validation evidence is incomplete; see validation_manifest.csv", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
