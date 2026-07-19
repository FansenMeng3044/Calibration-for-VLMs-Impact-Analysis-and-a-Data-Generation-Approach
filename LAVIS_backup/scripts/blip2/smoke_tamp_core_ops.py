#!/usr/bin/env python3
"""Torch-level smoke checks for the BLIP2-T5 TAMP core ops.

This test avoids importing the full LAVIS package, so it can run in lightweight
torch environments that lack BLIP2 dependencies. It extracts the current source
definitions for AMIA and DAS helpers, executes them in an isolated namespace, and
checks invariants that static string validation cannot prove:

  - encoder attention column scores zero out padding positions;
  - T5Block replay helper propagates self/cross position bias by tuple order;
  - AMIA selection is padding-aware and never selects padded rows;
  - AMIA updates a real Linear layer's Wanda scaler_row;
  - DAS cosine density ignores padded language positions.

It is not a replacement for smoke_tamp_migration_runtime.py, which still needs a
full BLIP2-T5 environment and checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict


def find_between(text: str, start: str, end: str) -> str:
    start_idx = text.find(start)
    if start_idx < 0:
        raise ValueError(f"start marker not found: {start}")
    end_idx = text.find(end, start_idx + len(start))
    if end_idx < 0:
        raise ValueError(f"end marker not found: {end}")
    return text[start_idx:end_idx]


def load_core_namespace(lavis_root: Path) -> Dict[str, object]:
    import torch
    import torch.nn as nn

    wanda_path = lavis_root / "lavis" / "compression" / "pruners" / "wanda_pruner.py"
    layer_path = lavis_root / "lavis" / "compression" / "pruners" / "layer_single_base_pruner.py"
    wanda = wanda_path.read_text(encoding="utf-8")
    layer = layer_path.read_text(encoding="utf-8")

    namespace: Dict[str, object] = {"torch": torch, "nn": nn, "math": math}

    wanda_parts = [
        find_between(wanda, "def _align_bool_vector(", "\n\ndef _align_float_vector("),
        find_between(wanda, "def _align_float_vector(", "\n\ndef _normal_t5_block_forward("),
        find_between(wanda, "def _normal_t5_block_forward(", "\n\ndef _encoder_attention_column_scores("),
        find_between(wanda, "def _encoder_attention_column_scores(", "\n\ndef _write_calibration_batch_trace("),
        find_between(wanda, "def _cos_pairwise_density_single(", "\n\nclass AdaptiveMultimodalInputActivation"),
        find_between(wanda, "class AdaptiveMultimodalInputActivation:", "\n\nclass WrappedGPT:"),
    ]
    exec("\n\n".join(wanda_parts), namespace)

    layer_parts = [
        find_between(layer, "def _align_bool_vector(", "\n\ndef _normal_t5_block_forward("),
        find_between(layer, "def _normal_t5_block_forward(", "\n\ndef cos_pairwise_density("),
        find_between(layer, "def cos_pairwise_density(", "\n\nclass ActivationDensity:"),
    ]
    layer_namespace: Dict[str, object] = {"torch": torch, "nn": nn}
    exec("\n\n".join(layer_parts), layer_namespace)
    namespace["das_normal_t5_forward"] = layer_namespace["_normal_t5_block_forward"]
    namespace["das_cos_pairwise_density"] = layer_namespace["cos_pairwise_density"]
    return namespace


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_checks(lavis_root: Path, device: str) -> Dict[str, object]:
    import torch
    import torch.nn as nn

    ns = load_core_namespace(lavis_root)
    normal_t5_forward = ns["_normal_t5_block_forward"]
    das_normal_t5_forward = ns["das_normal_t5_forward"]
    encoder_scores = ns["_encoder_attention_column_scores"]
    amia_cls = ns["AdaptiveMultimodalInputActivation"]
    das_density = ns["das_cos_pairwise_density"]

    B, H, S, D = 2, 2, 6, 4

    class FakeT5Block(nn.Module):
        def forward(self, hidden_states, **kwargs):
            hidden = hidden_states + 1.0
            self_bias = torch.full((1,), 2.0, device=hidden_states.device)
            self_attn = torch.full((1,), 3.0, device=hidden_states.device)
            cross_bias = torch.full((1,), 4.0, device=hidden_states.device)
            cross_attn = torch.full((1,), 5.0, device=hidden_states.device)
            present = ("fake_kv",)
            use_cache = bool(kwargs.get("use_cache", False))
            output_attentions = bool(kwargs.get("output_attentions", False))
            has_cross = kwargs.get("encoder_hidden_states") is not None

            outputs = (hidden,)
            if use_cache:
                outputs = outputs + (present,)
            outputs = outputs + (self_bias,)
            if output_attentions:
                outputs = outputs + (self_attn,)
            if has_cross:
                outputs = outputs + (cross_bias,)
                if output_attentions:
                    outputs = outputs + (cross_attn,)
            return outputs

    hidden_probe = torch.zeros(1, 2, D, device=device)
    out_probe, cache_probe, attn_probe = normal_t5_forward(
        FakeT5Block(),
        hidden_probe,
        {"attention_mask": None, "use_cache": False},
        output_attentions=True,
    )
    assert_true(tuple(out_probe.shape) == tuple(hidden_probe.shape), "T5 helper hidden shape mismatch")
    assert_true(float(cache_probe["position_bias"].item()) == 2.0, "encoder self position bias not propagated")
    assert_true(float(attn_probe.item()) == 3.0, "encoder attention weights index mismatch")

    _out_dec, cache_dec, attn_dec = normal_t5_forward(
        FakeT5Block(),
        hidden_probe,
        {"encoder_hidden_states": hidden_probe, "use_cache": True},
        output_attentions=True,
    )
    assert_true(float(cache_dec["position_bias"].item()) == 2.0, "decoder self position bias not propagated")
    assert_true(
        float(cache_dec["encoder_decoder_position_bias"].item()) == 4.0,
        "decoder cross position bias not propagated",
    )
    assert_true(float(attn_dec.item()) == 3.0, "decoder self attention weights index mismatch")

    das_out_probe, das_cache_probe = das_normal_t5_forward(
        FakeT5Block(),
        hidden_probe,
        {"attention_mask": None, "use_cache": False},
        output_attentions=True,
    )
    assert_true(
        tuple(das_out_probe.shape) == tuple(hidden_probe.shape),
        "DAS T5 helper hidden shape mismatch",
    )
    assert_true(
        float(das_cache_probe["position_bias"].item()) == 2.0,
        "DAS encoder self position bias not propagated",
    )

    _das_out_dec, das_cache_dec = das_normal_t5_forward(
        FakeT5Block(),
        hidden_probe,
        {"encoder_hidden_states": hidden_probe, "use_cache": True},
        output_attentions=True,
    )
    assert_true(
        float(das_cache_dec["position_bias"].item()) == 2.0,
        "DAS decoder self position bias not propagated",
    )
    assert_true(
        float(das_cache_dec["encoder_decoder_position_bias"].item()) == 4.0,
        "DAS decoder cross position bias not propagated",
    )

    attn = torch.ones(B, H, S, S, device=device)
    attn[:, :, :, -1] = 100.0
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]],
        dtype=torch.long,
        device=device,
    )
    scores = encoder_scores(attn, attention_mask)
    assert_true(scores is not None, "encoder attention scores are None")
    assert_true(tuple(scores.shape) == (B, S), "encoder score shape mismatch")
    assert_true(float(scores[1, 4:].abs().max().item()) == 0.0, "PAD score positions are nonzero")
    assert_true(torch.isfinite(scores).all().item(), "encoder scores contain non-finite values")

    layer = nn.Linear(D, 3, bias=False).to(device)
    wrapped = amia_cls(layer, keep_ratio=1.0)
    inp = torch.randn(B, S, D, device=device)
    out = torch.randn(B, S, D, device=device)
    inp[1, 4:] = float("nan")
    image_mask = torch.tensor(
        [[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]],
        dtype=torch.bool,
        device=device,
    )
    # Make padded positions extremely tempting. Correct masking still excludes them.
    scores_for_amia = torch.ones(B, S, device=device)
    scores_for_amia[1, 4:] = 1000.0
    wrapped.add_batch(
        inp,
        out,
        image_mask=image_mask,
        score=scores_for_amia,
        attention_mask=attention_mask.bool(),
    )
    valid_count = int(attention_mask.sum().item())
    assert_true(0 < int(wrapped.nsamples) <= valid_count, "AMIA selected no rows or too many rows")
    assert_true(
        torch.isfinite(wrapped.scaler_row).all().item(),
        "AMIA scaler_row has non-finite values, likely from padded rows",
    )
    assert_true(float(wrapped.scaler_row.sum().item()) > 0.0, "AMIA scaler_row did not update")

    missing_score_wrapped = amia_cls(nn.Linear(D, 3, bias=False).to(device), keep_ratio=1.0)
    try:
        missing_score_wrapped.add_batch(
            inp,
            out,
            image_mask=image_mask,
            score=None,
            attention_mask=attention_mask.bool(),
        )
    except RuntimeError as exc:
        assert_true(
            "attention contribution scores" in str(exc),
            "AMIA missing-score error should explain attention contribution scores",
        )
    else:
        raise AssertionError("AMIA accepted missing attention contribution scores")

    embeddings = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],  # visual
                [0.0, 1.0, 0.0, 0.0],  # one valid text token
                [0.0, 1.0, 0.0, 0.0],  # PAD, would fake language-language density if included
            ]
        ],
        device=device,
    )
    das_image_mask = torch.tensor([[1, 0, 0]], dtype=torch.bool, device=device)
    das_attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool, device=device)
    _v, l, _vl = das_density(embeddings, das_image_mask, attention_mask=das_attention_mask)
    assert_true(abs(float(l)) < 1e-8, "DAS language density included padded token")

    return {
        "ok": True,
        "device": device,
        "score_shape": list(scores.shape),
        "amia_selected_rows": int(wrapped.nsamples),
        "valid_rows": valid_count,
        "das_language_density": float(l),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run torch-only smoke checks for TAMP AMIA/DAS core operations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lavis_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out_json", default=None)
    return parser.parse_args()


def main() -> int:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "[ERROR] smoke_tamp_core_ops.py requires PyTorch. "
            "Run with the same environment used for LAVIS, or set PYTHON_BIN "
            "when calling run_tamp_migration_validation.sh."
        ) from exc

    args = parse_args()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    summary = run_checks(Path(args.lavis_root).resolve(), device)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print("[OK] TAMP core op smoke passed:", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
