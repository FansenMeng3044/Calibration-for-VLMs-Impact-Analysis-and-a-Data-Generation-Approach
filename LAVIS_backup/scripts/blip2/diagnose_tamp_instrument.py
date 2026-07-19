#!/usr/bin/env python3
"""Instrument diagnostics for using TAMP as a calibration probe on BLIP2-T5.

This script does NOT prune and does NOT change any pruning behaviour. It runs the
production code paths (AdaptiveMultimodalInputActivation._select_tokens,
LayerSparsity.compute_density) on real calibration data and records what they do,
so that calibration-comparison conclusions can be interpreted.

Three diagnostics, one calibration pass each:

  D1  AMIA sensitivity      per-layer selected/valid token ratio, and how many of
                            the selected tokens are visual vs text. Answers "is the
                            probe actually sensitive to the calibration data, or does
                            it collapse to a handful of tokens regardless?"

  D2  DAS noise floor       splits the calibration pool into two disjoint halves and
                            runs DAS on each. The Spearman correlation between the
                            two layer-importance vectors is the within-dataset noise
                            floor: any between-dataset difference smaller than this
                            is not interpretable.

  D3  Length confound       valid-text-token distribution per calibration set. T5's
                            relative position bias (and its absence in block-replay
                            calibration) scales with sequence length, so this checks
                            whether sequence length co-varies with the calibration
                            set being compared.

Run once per calibration set with a distinct --label; rows are appended to the same
CSVs so the sets can be compared directly.

Example (per calibration set):

  CUDA_VISIBLE_DEVICES=0 python scripts/blip2/diagnose_tamp_instrument.py \
    --label mmbench \
    --calib_json /data/data2/mfs/MMBench_calibration/mmbench_calib_128.json \
    --images_dir /data/data2/mfs/MMBench_calibration/images \
    --ckpt /data/data2/mfs/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth \
    --max_samples 128 --batch_size 8 \
    --out_dir tamp_instrument_diag
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


_LAVIS_ROOT = Path(__file__).resolve().parents[2]
if str(_LAVIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAVIS_ROOT))


TEXT_FIELDS = ("question", "caption", "text_input", "text", "prompt")
OUTPUT_FIELDS = ("text_output", "answer", "caption", "text", "question", "output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose TAMP as a calibration probe on BLIP2-T5 (no pruning).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--label", required=True, help="Calibration set name, e.g. mmbench / cc3m.")
    parser.add_argument("--calib_json", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model_name", default="blip2_t5")
    parser.add_argument("--model_type", default="pretrain_flant5xl")
    parser.add_argument("--device", default=None)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_txt_len", type=int, default=None)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--expected_query_tokens", type=int, default=32)
    parser.add_argument(
        "--probe_blocks",
        default="all",
        help="Which encoder blocks to probe for D1: 'all' or comma-separated indices.",
    )
    parser.add_argument(
        "--probe_linears",
        default="all",
        help=(
            "Which Linear sub-layers to probe for D1: 'all' or comma-separated name "
            "suffixes, e.g. 'SelfAttention.v,DenseReluDense.wo'."
        ),
    )
    parser.add_argument("--skip_d1", action="store_true")
    parser.add_argument("--skip_d2", action="store_true")
    parser.add_argument("--out_dir", default="tamp_instrument_diag")
    return parser.parse_args()


# --------------------------------------------------------------------------- io


def load_rows(path: str, max_samples: int) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    rows = [row for row in data if isinstance(row, dict)]
    if max_samples is not None:
        rows = rows[: int(max_samples)]
    if not rows:
        raise ValueError(f"No usable rows found in {path}.")
    return rows


def first_text(row: Dict[str, Any], fields: Sequence[str]) -> str:
    for field in fields:
        value = row.get(field)
        if value is None:
            continue
        if isinstance(value, list):
            value = " ".join(str(item) for item in value)
        text = str(value).strip()
        if text:
            return text
    return ""


def resolve_image_path(images_dir: str, image_value: Any) -> str:
    if image_value is None:
        raise ValueError("row has no image field")
    image_path = str(image_value)
    if os.path.isabs(image_path) and os.path.isfile(image_path):
        return image_path
    joined = os.path.join(images_dir, image_path)
    if os.path.isfile(joined):
        return joined
    raise FileNotFoundError(f"image not found: {image_path} under {images_dir}")


def iter_batches(rows, images_dir, vis_processor, torch, Image, device, batch_size) -> Iterable[Dict[str, Any]]:
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        images, text_inputs, text_outputs = [], [], []
        for row in batch_rows:
            image = Image.open(resolve_image_path(images_dir, row.get("image"))).convert("RGB")
            images.append(vis_processor(image))
            text_input = first_text(row, TEXT_FIELDS)
            if not text_input:
                raise ValueError(f"row missing text input fields {TEXT_FIELDS}: {row}")
            text_inputs.append(text_input)
            text_outputs.append(first_text(row, OUTPUT_FIELDS) or text_input)
        yield {
            "image": torch.stack(images, dim=0).to(device),
            "text_input": text_inputs,
            "text_output": text_outputs,
        }


def append_csv(path: str, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ------------------------------------------------------------------ statistics


def _ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    if da == 0.0 or db == 0.0:
        return float("nan")
    return num / (da * db)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    return _pearson(_ranks(a), _ranks(b))


# ------------------------------------------------------------------ diagnostics


def build_probed_amia(base_cls, recorder: List[Dict[str, Any]]):
    """Subclass the production AMIA so _select_tokens is observed, never reimplemented."""

    class _ProbedAMIA(base_cls):
        probe_label = ""
        probe_block = -1
        probe_name = ""
        probe_batch = -1

        def _select_tokens(self, out, image_mask, score, attention_mask=None, eps=1e-8):
            mask = super()._select_tokens(out, image_mask, score, attention_mask=attention_mask, eps=eps)
            n_valid = int(mask.numel())
            n_sel = int(mask.sum().item())
            img = image_mask.bool().to(mask.device)
            n_vis_sel = int((mask & img).sum().item())
            recorder.append(
                {
                    "label": self.probe_label,
                    "block": self.probe_block,
                    "linear": self.probe_name,
                    "batch": self.probe_batch,
                    "n_valid": n_valid,
                    "n_selected": n_sel,
                    "select_ratio": round(n_sel / max(1, n_valid), 6),
                    "n_visual_selected": n_vis_sel,
                    "n_text_selected": n_sel - n_vis_sel,
                    "n_visual_available": int(img.sum().item()),
                }
            )
            return mask

    return _ProbedAMIA


def run_d1_amia_selection(
    torch,
    model,
    pruner_mod,
    calib,
    label: str,
    probe_blocks: str,
    probe_linears: str,
) -> List[Dict[str, Any]]:
    """D1: record AMIA selection statistics on the dense model, layer by layer."""
    inps, _outs, caches, image_masks, encoder_attention_masks = calib[:5]
    recorder: List[Dict[str, Any]] = []
    probed_cls = build_probed_amia(pruner_mod.AdaptiveMultimodalInputActivation, recorder)

    blocks = pruner_mod.get_module_recursive(model, "t5_model.encoder.block")
    if probe_blocks.strip().lower() == "all":
        block_ids = list(range(len(blocks)))
    else:
        block_ids = [int(x) for x in probe_blocks.split(",") if x.strip() != ""]

    linear_filter = None
    if probe_linears.strip().lower() != "all":
        linear_filter = [x.strip() for x in probe_linears.split(",") if x.strip()]

    # Same block-0 cache for every block: the ECoFLaP calibration convention this
    # repository deliberately follows. Mirrors T5LayerWandaPruner._prune exactly.
    layer_caches = [dict(cache) for cache in caches]
    hidden = list(inps)

    for i in range(len(blocks)):
        layer = blocks[i]
        if i in block_ids:
            subset = pruner_mod.find_layers(layer)
            if linear_filter is not None:
                subset = {k: v for k, v in subset.items() if any(k.endswith(f) for f in linear_filter)}

            wrapped = {name: probed_cls(mod) for name, mod in subset.items()}
            for name, w in wrapped.items():
                w.probe_label, w.probe_block, w.probe_name = label, i, name

            for j in range(len(hidden)):
                attn_mask_j = encoder_attention_masks[j]
                with torch.no_grad():
                    with model.maybe_autocast(dtype=torch.bfloat16):
                        _, _, attn_weights = pruner_mod._normal_t5_block_forward(
                            layer, hidden[j], layer_caches[j], output_attentions=True
                        )
                score_j = pruner_mod._encoder_attention_column_scores(attn_weights, attn_mask_j)
                if score_j is None:
                    raise RuntimeError(
                        f"block {i}: could not derive encoder attention scores; "
                        "check the transformers version's T5Block output order."
                    )
                for w in wrapped.values():
                    w.probe_batch = j

                handles = [
                    subset[name].register_forward_hook(
                        _make_hook(wrapped[name], image_masks[j], score_j, attn_mask_j)
                    )
                    for name in wrapped
                ]
                try:
                    with torch.no_grad():
                        with model.maybe_autocast(dtype=torch.bfloat16):
                            pruner_mod._normal_t5_block_forward(layer, hidden[j], layer_caches[j])
                finally:
                    for h in handles:
                        h.remove()

        # propagate to next block (dense; no pruning is applied by this script)
        new_hidden = []
        for j in range(len(hidden)):
            with torch.no_grad():
                with model.maybe_autocast(dtype=torch.bfloat16):
                    out, _, _ = pruner_mod._normal_t5_block_forward(layer, hidden[j], layer_caches[j])
            new_hidden.append(out.detach())
        hidden = new_hidden

    return recorder


def _make_hook(wrapped, image_mask, score, attention_mask):
    def hook(_module, inp, out):
        out_tensor = out[0] if isinstance(out, (tuple, list)) else out
        wrapped.add_batch(inp[0].data, out_tensor.data, image_mask, score, attention_mask=attention_mask)

    return hook


def split_calib(calib, n_a: int):
    """Split a 5-tuple calibration result into two disjoint halves."""
    inps, outs, caches, image_masks, attn_masks = calib[:5]
    a = (inps[:n_a], [None] * n_a, caches[:n_a], image_masks[:n_a], attn_masks[:n_a])
    nb = len(inps) - n_a
    b = (inps[n_a:], [None] * nb, caches[n_a:], image_masks[n_a:], attn_masks[n_a:])
    return a, b


def run_d2_das_noise_floor(pruner, calib, sparsity: float) -> Dict[str, Any]:
    """D2: DAS on two disjoint calibration halves -> within-dataset noise floor."""
    n_batches = len(calib[0])
    if n_batches < 2:
        raise ValueError("D2 needs at least 2 cached calibration batches; increase --max_samples.")
    half = n_batches // 2
    calib_a, calib_b = split_calib(calib, half)

    pruner._cached_encoder_calib = calib_a
    sparsity_a = pruner.get_sparsity(sparsity, sparsity_ratio_granularity="layer")
    pruner._cached_encoder_calib = calib_b
    sparsity_b = pruner.get_sparsity(sparsity, sparsity_ratio_granularity="layer")
    pruner._cached_encoder_calib = None

    keys = sorted(set(sparsity_a) & set(sparsity_b))
    keys = [k for k in keys if "encoder.block." in k]
    va = [float(sparsity_a[k]) for k in keys]
    vb = [float(sparsity_b[k]) for k in keys]
    diffs = [abs(x - y) for x, y in zip(va, vb)]
    return {
        "n_layers": len(keys),
        "half_batches": half,
        "spearman": round(spearman(va, vb), 6),
        "pearson": round(_pearson(va, vb), 6),
        "max_abs_diff": round(max(diffs), 6) if diffs else float("nan"),
        "mean_abs_diff": round(sum(diffs) / len(diffs), 6) if diffs else float("nan"),
        "range_a": round(max(va) - min(va), 6) if va else float("nan"),
        "range_b": round(max(vb) - min(vb), 6) if vb else float("nan"),
    }


def run_d3_lengths(calib, label: str, expected_query: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """D3: valid text token distribution -> length confound check."""
    _inps, _outs, _caches, image_masks, attn_masks = calib[:5]
    rows: List[Dict[str, Any]] = []
    lengths: List[int] = []
    for j, (img_mask, attn_mask) in enumerate(zip(image_masks, attn_masks)):
        img = img_mask.bool()
        attn = attn_mask.bool()
        if img.dim() == 1:
            img, attn = img.unsqueeze(0), attn.unsqueeze(0)
        B, S = img.shape
        for b in range(B):
            valid_text = int(((~img[b, expected_query:]) & attn[b, expected_query:]).sum().item())
            pad_text = int((~attn[b, expected_query:]).sum().item())
            rows.append(
                {
                    "label": label,
                    "batch": j,
                    "sample": b,
                    "seq_len": int(S),
                    "n_valid_text": valid_text,
                    "n_pad_text": pad_text,
                }
            )
            lengths.append(valid_text)
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    summary = {
        "n_samples": n,
        "valid_text_mean": round(sum(lengths_sorted) / n, 3) if n else float("nan"),
        "valid_text_min": lengths_sorted[0] if n else 0,
        "valid_text_p50": lengths_sorted[n // 2] if n else 0,
        "valid_text_max": lengths_sorted[-1] if n else 0,
    }
    return rows, summary


# ------------------------------------------------------------------------ main


def main() -> int:
    args = parse_args()

    missing = []
    if not os.path.isfile(args.calib_json):
        missing.append(f"--calib_json {args.calib_json}")
    if not os.path.isdir(args.images_dir):
        missing.append(f"--images_dir {args.images_dir}")
    if not os.path.isfile(args.ckpt):
        missing.append(f"--ckpt {args.ckpt}")
    if missing:
        raise SystemExit("[ERROR] Missing input path(s): " + ", ".join(missing))

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
        raise SystemExit(
            "[ERROR] diagnose_tamp_instrument.py needs the full LAVIS runtime "
            f"(PyTorch, Pillow, this repo on PYTHONPATH). Missing module: {exc.name}"
        ) from exc

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    rows = load_rows(args.calib_json, args.max_samples)
    vis_processor = load_processor("blip_image_eval").build(image_size=args.image_size)

    print(f"[load] {args.model_name}/{args.model_type} device={device} label={args.label}")
    model = load_model(
        args.model_name, args.model_type, is_eval=True, device=device, checkpoint=args.ckpt
    )
    model.eval()
    if args.max_txt_len is not None:
        model.max_txt_len = int(args.max_txt_len)

    batches = list(
        iter_batches(rows, args.images_dir, vis_processor, torch, Image, device, args.batch_size)
    )

    pruner = BLIPT5LayerWandaPruner(
        model=model,
        data_loader=batches,
        t5_prune_spec="24-%.6f-1.0-1.0" % (1.0 - args.sparsity),
        vit_prune_spec=None,
        t5_pruning_method="none",
        vit_pruning_method="none",
        num_samples=len(rows),
        num_data_first_stage=len(rows),
        sparsity_ratio_granularity="layer",
        max_sparsity_per_layer=min(1.0, args.sparsity + 0.1),
        score_method="density_sum",
        token_selection="amia",
        prune_t5=True,
        prune_vit=False,
        importance_scope="llm_only",
    )

    print("[calib] running encoder calibration ...")
    with torch.no_grad():
        calib = T5LayerWandaPruner.prepare_calibration_input_encoder(
            pruner,
            model,
            batches,
            device,
            "t5_model",
            len(rows),
            module_to_process="t5_model.encoder.block",
            return_image_masks=True,
        )
    if len(calib) < 5:
        raise SystemExit(
            "[ERROR] calibration did not return encoder_attention_masks; "
            "AMIA/DAS diagnostics need temp_label + temp_encoder_atts on Blip2T5."
        )
    print(f"[calib] cached_batches={len(calib[0])}")

    summary: Dict[str, Any] = {
        "label": args.label,
        "calib_json": args.calib_json,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "sparsity": args.sparsity,
        "cached_batches": len(calib[0]),
        "note": "Diagnostics only. No weights were pruned by this script.",
    }

    # ---- D3 (cheapest, always) ----
    d3_rows, d3_summary = run_d3_lengths(calib, args.label, args.expected_query_tokens)
    append_csv(
        os.path.join(args.out_dir, "d3_calib_lengths.csv"),
        ["label", "batch", "sample", "seq_len", "n_valid_text", "n_pad_text"],
        d3_rows,
    )
    summary["d3_length"] = d3_summary
    print("[D3] valid-text length:", d3_summary)

    # ---- D1 AMIA selection ----
    if not args.skip_d1:
        print("[D1] probing AMIA token selection ...")
        d1_rows = run_d1_amia_selection(
            torch, model, pruner_mod, calib, args.label, args.probe_blocks, args.probe_linears
        )
        append_csv(
            os.path.join(args.out_dir, "d1_amia_selection.csv"),
            [
                "label", "block", "linear", "batch", "n_valid", "n_selected",
                "select_ratio", "n_visual_selected", "n_text_selected", "n_visual_available",
            ],
            d1_rows,
        )
        if d1_rows:
            ratios = [r["select_ratio"] for r in d1_rows]
            sel = [r["n_selected"] for r in d1_rows]
            vis0 = sum(1 for r in d1_rows if r["n_visual_selected"] == 0)
            summary["d1_amia"] = {
                "observations": len(d1_rows),
                "select_ratio_mean": round(sum(ratios) / len(ratios), 6),
                "select_ratio_min": round(min(ratios), 6),
                "select_ratio_max": round(max(ratios), 6),
                "n_selected_mean": round(sum(sel) / len(sel), 3),
                "n_selected_min": min(sel),
                "n_selected_max": max(sel),
                "zero_visual_selected_frac": round(vis0 / len(d1_rows), 6),
            }
            print("[D1]", summary["d1_amia"])

    # ---- D2 DAS noise floor ----
    if not args.skip_d2:
        print("[D2] DAS on two disjoint calibration halves ...")
        summary["d2_das_noise_floor"] = run_d2_das_noise_floor(pruner, calib, args.sparsity)
        append_csv(
            os.path.join(args.out_dir, "d2_das_noise_floor.csv"),
            ["label"] + list(summary["d2_das_noise_floor"].keys()),
            [dict(summary["d2_das_noise_floor"], label=args.label)],
        )
        print("[D2]", summary["d2_das_noise_floor"])

    out_json = os.path.join(args.out_dir, f"summary_{args.label}.json")
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"[done] wrote {out_json}")
    print(f"[done] CSVs appended under {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
