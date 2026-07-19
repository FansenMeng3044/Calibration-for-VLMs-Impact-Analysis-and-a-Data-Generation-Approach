#!/usr/bin/env python3
"""Runtime smoke test for the BLIP2-T5 TAMP migration.

This is intentionally small and server-oriented. It loads the real BLIP2-T5
model, runs a few multimodal calibration samples through the T5 encoder catcher,
and checks the invariants that static source validation cannot prove:

  - temp_label and temp_encoder_atts have the expected [B, 32 + text] layout;
  - AMIA contribution scores can be derived from encoder attention;
  - AMIA can update a real Linear layer's Wanda scaler_row;
  - optional DAS layer sparsity can be computed over real T5 encoder Linear keys.

Example:

  CUDA_VISIBLE_DEVICES=0 python scripts/blip2/smoke_tamp_migration_runtime.py \
    --calib_json /data/data2/mfs/MMBench_calibration/mmbench_calib_128.json \
    --images_dir /data/data2/mfs/MMBench_calibration/images \
    --ckpt /data/data2/mfs/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth \
    --max_samples 2 --batch_size 2 --run_das
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


_LAVIS_ROOT = Path(__file__).resolve().parents[2]
if str(_LAVIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAVIS_ROOT))


TEXT_FIELDS = ("question", "caption", "text_input", "text", "prompt")
OUTPUT_FIELDS = ("text_output", "answer", "caption", "text", "question", "output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small real-model smoke test for BLIP2-T5 TAMP migration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--calib_json", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model_name", default="blip2_t5")
    parser.add_argument("--model_type", default="pretrain_flant5xl")
    parser.add_argument("--device", default=None)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_samples", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_txt_len", type=int, default=None)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--expected_query_tokens", type=int, default=32)
    parser.add_argument("--run_das", action="store_true", help="Also compute DAS layer sparsity.")
    parser.add_argument("--out_json", default=None)
    return parser.parse_args()


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


def iter_batches(
    rows: Sequence[Dict[str, Any]],
    images_dir: str,
    vis_processor: Any,
    torch: Any,
    Image: Any,
    device: str,
    batch_size: int,
) -> Iterable[Dict[str, Any]]:
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        images = []
        text_inputs = []
        text_outputs = []
        for row in batch_rows:
            image_path = resolve_image_path(images_dir, row.get("image"))
            image = Image.open(image_path).convert("RGB")
            images.append(vis_processor(image))
            text_input = first_text(row, TEXT_FIELDS)
            text_output = first_text(row, OUTPUT_FIELDS) or text_input
            if not text_input:
                raise ValueError(f"row missing text input fields {TEXT_FIELDS}: {row}")
            text_inputs.append(text_input)
            text_outputs.append(text_output)
        yield {
            "image": torch.stack(images, dim=0).to(device),
            "text_input": text_inputs,
            "text_output": text_outputs,
        }


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_mask_layout(image_masks: Sequence[Any], attention_masks: Sequence[Any], expected_query: int) -> Dict[str, Any]:
    total_samples = 0
    min_valid_text = None
    max_pad_text = 0
    for batch_idx, (image_mask, attention_mask) in enumerate(zip(image_masks, attention_masks)):
        assert_true(tuple(image_mask.shape) == tuple(attention_mask.shape), "image/attention mask shape mismatch")
        if image_mask.dim() == 1:
            image_mask = image_mask.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        B, S = image_mask.shape
        assert_true(S > expected_query, f"sequence too short in batch {batch_idx}: {S}")
        for b in range(B):
            img = image_mask[b].bool()
            attn = attention_mask[b].bool()
            prefix_true = int(img[:expected_query].sum().item())
            suffix_true = int(img[expected_query:].sum().item())
            valid_text = int((~img[expected_query:] & attn[expected_query:]).sum().item())
            pad_text = int((~attn[expected_query:]).sum().item())
            assert_true(prefix_true == expected_query, "visual query prefix is not fully marked True")
            assert_true(suffix_true == 0, "text suffix contains visual-token True labels")
            assert_true(int(attn[:expected_query].sum().item()) == expected_query, "visual query tokens are masked out")
            assert_true(valid_text > 0, "no valid text tokens in calibration sample")
            min_valid_text = valid_text if min_valid_text is None else min(min_valid_text, valid_text)
            max_pad_text = max(max_pad_text, pad_text)
            total_samples += 1
    return {
        "cached_batches": len(image_masks),
        "physical_samples": total_samples,
        "min_valid_text_tokens": int(min_valid_text or 0),
        "max_pad_text_tokens": int(max_pad_text),
    }


def check_cache_attention_layout(caches: Sequence[Dict[str, Any]], inps: Sequence[Any], raw_attention_masks: Sequence[Any]) -> Dict[str, Any]:
    """Check that cached T5 block masks and raw 0/1 masks have distinct, aligned roles."""
    checked_batches = 0
    batches_with_padding = 0
    for batch_idx, (cache, inp, raw_attn) in enumerate(zip(caches, inps, raw_attention_masks)):
        cache_mask = cache.get("attention_mask")
        assert_true(cache_mask is not None, f"cache[{batch_idx}] missing extended attention_mask")
        assert_true(raw_attn.dim() == 2, f"raw attention mask must be [B,S], got {tuple(raw_attn.shape)}")
        B, S = raw_attn.shape
        assert_true(inp.shape[0] == B and inp.shape[1] == S, "raw attention mask does not match cached input shape")
        assert_true(cache_mask.shape[0] == B, "extended attention mask batch dimension mismatch")
        assert_true(cache_mask.shape[-1] == S, "extended attention mask sequence dimension mismatch")
        assert_true(cache_mask.dim() >= 3, "T5 block attention_mask should be broadcast/extended, not raw [B,S]")

        cm = cache_mask.detach().float().cpu()
        raw = raw_attn.detach().bool().cpu()
        for b in range(B):
            raw_b = raw[b]
            if bool((~raw_b).any().item()):
                batches_with_padding += 1
                cols = cm[b].reshape(-1, S)
                valid_min = float(cols[:, raw_b].min().item())
                invalid_max = float(cols[:, ~raw_b].max().item())
                assert_true(
                    invalid_max < valid_min,
                    "extended attention mask does not suppress raw PAD columns",
                )
        checked_batches += 1
    return {
        "cache_batches_checked": int(checked_batches),
        "cache_batches_with_padding": int(batches_with_padding),
    }


def summarize_encoder_inputs(
    inps: Sequence[Any],
    image_masks: Sequence[Any],
    attention_masks: Sequence[Any],
    expected_query: int,
) -> Dict[str, Any]:
    """Machine-readable evidence for the [visual query][text][PAD] encoder layout."""
    shapes = []
    total_visual = 0
    total_valid_text = 0
    total_pad_text = 0
    total_tokens = 0
    for inp, image_mask, attention_mask in zip(inps, image_masks, attention_masks):
        if image_mask.dim() == 1:
            image_mask = image_mask.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        B, S = image_mask.shape
        shapes.append([int(B), int(S), int(inp.shape[-1])])
        img = image_mask.bool()
        attn = attention_mask.bool()
        total_visual += int(img[:, :expected_query].sum().item())
        total_valid_text += int(((~img[:, expected_query:]) & attn[:, expected_query:]).sum().item())
        total_pad_text += int((~attn[:, expected_query:]).sum().item())
        total_tokens += int(B * S)
    return {
        "cached_batches": int(len(inps)),
        "encoder_input_shapes": shapes,
        "total_tokens": int(total_tokens),
        "total_visual_query_tokens": int(total_visual),
        "total_valid_text_tokens": int(total_valid_text),
        "total_pad_text_tokens": int(total_pad_text),
    }


def main() -> int:
    args = parse_args()

    missing_paths = []
    if not os.path.isfile(args.calib_json):
        missing_paths.append(f"--calib_json {args.calib_json}")
    if not os.path.isdir(args.images_dir):
        missing_paths.append(f"--images_dir {args.images_dir}")
    if not os.path.isfile(args.ckpt):
        missing_paths.append(f"--ckpt {args.ckpt}")
    if missing_paths:
        raise SystemExit("[ERROR] Missing runtime smoke input path(s): " + ", ".join(missing_paths))

    try:
        import torch
        from PIL import Image
        from lavis.models import load_model
        from lavis.processors import load_processor
        from lavis.compression.pruners.wanda_pruner import (
            AdaptiveMultimodalInputActivation,
            BLIPT5LayerWandaPruner,
            T5LayerWandaPruner,
            _encoder_attention_column_scores,
            _normal_t5_block_forward,
            find_layers,
            get_module_recursive,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "[ERROR] smoke_tamp_migration_runtime.py requires the full LAVIS runtime "
            "environment with PyTorch, Pillow, and this repository on PYTHONPATH. "
            "Run with the same conda environment used for BLIP2 evaluation, or set "
            "PYTHON_BIN when calling run_tamp_migration_validation.sh. Missing module: "
            f"{exc.name}"
        ) from exc

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows(args.calib_json, args.max_samples)
    vis_processor = load_processor("blip_image_eval").build(image_size=args.image_size)

    print(f"[load] model={args.model_name}/{args.model_type} device={device}")
    model = load_model(
        args.model_name,
        args.model_type,
        is_eval=True,
        device=device,
        checkpoint=args.ckpt,
    )
    model.eval()
    if args.max_txt_len is not None:
        model.max_txt_len = int(args.max_txt_len)

    batches = list(
        iter_batches(
            rows,
            args.images_dir,
            vis_processor,
            torch,
            Image,
            device,
            args.batch_size,
        )
    )

    pruner = BLIPT5LayerWandaPruner(
        model=model,
        data_loader=batches,
        t5_prune_spec="24-%.6f-1.0-1.0" % (1.0 - args.sparsity),
        vit_prune_spec=None,
        t5_pruning_method="none",
        vit_pruning_method="none",
        num_samples=args.max_samples,
        sparsity_ratio_granularity="layer",
        max_sparsity_per_layer=min(1.0, args.sparsity + 0.1),
        score_method="density_sum",
        token_selection="amia",
        prune_t5=True,
        prune_vit=False,
        importance_scope="llm_only",
    )

    with torch.no_grad():
        calib = T5LayerWandaPruner.prepare_calibration_input_encoder(
            pruner,
            model,
            batches,
            device,
            "t5_model",
            args.max_samples,
            module_to_process="t5_model.encoder.block",
            return_image_masks=True,
        )
    inps, _outs, caches, image_masks, encoder_attention_masks = calib[:5]
    assert_true(len(inps) > 0, "no encoder inputs were cached")
    assert_true(len(inps) == len(image_masks) == len(encoder_attention_masks), "cached batch count mismatch")
    mask_summary = check_mask_layout(image_masks, encoder_attention_masks, args.expected_query_tokens)
    print("[OK] mask layout:", mask_summary)
    cache_summary = check_cache_attention_layout(caches, inps, encoder_attention_masks)
    print("[OK] T5 cache attention masks:", cache_summary)
    encoder_input_summary = summarize_encoder_inputs(
        inps,
        image_masks,
        encoder_attention_masks,
        args.expected_query_tokens,
    )
    print("[OK] encoder inputs:", encoder_input_summary)

    layers = get_module_recursive(model, "t5_model.encoder.block")
    layer0 = layers[0]
    cache0 = dict(caches[0])
    with torch.no_grad():
        _hidden, _next_cache, attn_weights = _normal_t5_block_forward(
            layer0,
            inps[0],
            cache0,
            output_attentions=True,
        )
    scores = _encoder_attention_column_scores(attn_weights, encoder_attention_masks[0])
    assert_true(scores is not None, "failed to derive AMIA attention scores")
    assert_true(tuple(scores.shape) == tuple(image_masks[0].shape), "AMIA score shape mismatch")
    assert_true(torch.isfinite(scores).all().item(), "AMIA scores contain non-finite values")
    valid = encoder_attention_masks[0].to(scores.device).bool()
    assert_true(float(scores[valid].mean().item()) > 0, "valid AMIA scores are not positive")
    invalid = ~valid
    invalid_abs_max = 0.0
    if bool(invalid.any().item()):
        invalid_abs_max = float(scores[invalid].abs().max().item())
        assert_true(invalid_abs_max == 0.0, "invalid token scores are not zero")
    amia_score_summary = {
        "shape": list(scores.shape),
        "valid_mean": float(scores[valid].mean().item()),
        "valid_min": float(scores[valid].min().item()),
        "valid_max": float(scores[valid].max().item()),
        "invalid_abs_max": float(invalid_abs_max),
    }
    print("[OK] AMIA scores:", amia_score_summary)

    subset = find_layers(layer0)
    first_name, first_linear = next(iter(subset.items()))
    wrapped = AdaptiveMultimodalInputActivation(first_linear)

    def hook(_module: Any, inp: Any, out: Any) -> None:
        out_tensor = out[0] if isinstance(out, (tuple, list)) else out
        wrapped.add_batch(
            inp[0].data,
            out_tensor.data,
            image_masks[0].to(out_tensor.device),
            scores.to(out_tensor.device),
            attention_mask=encoder_attention_masks[0].to(out_tensor.device),
        )

    handle = first_linear.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _normal_t5_block_forward(layer0, inps[0], dict(caches[0]))
    finally:
        handle.remove()
    assert_true(wrapped.nsamples > 0, "AMIA did not select any token rows")
    valid_rows_first_batch = int(valid.sum().item())
    assert_true(
        int(wrapped.nsamples) <= valid_rows_first_batch,
        "AMIA selected more rows than valid non-PAD encoder tokens",
    )
    assert_true(torch.isfinite(wrapped.scaler_row).all().item(), "AMIA scaler_row contains non-finite values")
    print(
        "[OK] AMIA scaler:",
        {
            "linear": first_name,
            "nsamples": int(wrapped.nsamples),
            "valid_rows_first_batch": valid_rows_first_batch,
        },
    )

    das_summary: Dict[str, Any] = {}
    if args.run_das:
        pruner._cached_encoder_calib = calib
        sparsity = pruner.get_sparsity(args.sparsity, sparsity_ratio_granularity="layer")
        encoder_keys = [key for key in sparsity if key.startswith("t5_model.encoder.block.")]
        decoder_keys = [key for key in sparsity if key.startswith("t5_model.decoder.block.")]
        assert_true(len(encoder_keys) > 0, "DAS returned no T5 encoder sparsity keys")
        assert_true(len(decoder_keys) > 0, "DAS fallback returned no T5 decoder sparsity keys")
        assert_true(
            all(abs(float(sparsity[key]) - args.sparsity) < 1e-8 for key in decoder_keys),
            "decoder fallback sparsity is not uniform original sparsity",
        )
        das_summary = {
            "encoder_keys": len(encoder_keys),
            "decoder_fallback_keys": len(decoder_keys),
            "encoder_sparsity_min": min(float(sparsity[key]) for key in encoder_keys),
            "encoder_sparsity_max": max(float(sparsity[key]) for key in encoder_keys),
            "decoder_uniform_sparsity": float(args.sparsity),
            "encoder_key_example": encoder_keys[0],
            "decoder_key_example": decoder_keys[0],
        }
        print("[OK] DAS sparsity:", das_summary)

    summary = {
        "ok": True,
        "calib_json": args.calib_json,
        "images_dir": args.images_dir,
        "device": device,
        "rows_used": len(rows),
        "batch_size": args.batch_size,
        "expected_query_tokens": args.expected_query_tokens,
        "token_selection": "amia",
        "score_method": "density_sum",
        "sparsity_ratio_granularity": "layer",
        "importance_scope": "llm_only",
        "prune_t5": True,
        "prune_vit": False,
        "sparsity": float(args.sparsity),
        "max_sparsity_per_layer": min(1.0, args.sparsity + 0.1),
        "mask_summary": mask_summary,
        "cache_summary": cache_summary,
        "encoder_input_summary": encoder_input_summary,
        "amia_linear": first_name,
        "amia_score_summary": amia_score_summary,
        "amia_selected_rows": int(wrapped.nsamples),
        "amia_valid_rows_first_batch": valid_rows_first_batch,
        "amia_selected_fraction_first_batch": float(wrapped.nsamples) / max(valid_rows_first_batch, 1),
        "das_summary": das_summary,
    }
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[OK] TAMP runtime smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
