#!/usr/bin/env python3
"""Reload and strictly validate a saved Cosmos ATV Reasoner checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoProcessor, Cosmos3EdgeForConditionalGeneration

import cosmos_atv_prune as atv


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def assert_zero_report(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    for branch in ("vision", "ar", "combined_target_linears"):
        for key in ("parameters", "zeros"):
            if int(actual[branch][key]) != int(expected[branch][key]):
                raise RuntimeError(
                    f"Reloaded {branch}.{key}={actual[branch][key]} != saved {expected[branch][key]}"
                )
        if not math.isclose(
            float(actual[branch]["zero_ratio"]),
            float(expected[branch]["zero_ratio"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"Reloaded {branch} zero ratio changed: "
                f"{actual[branch]['zero_ratio']} != {expected[branch]['zero_ratio']}"
            )


def assert_float(name: str, actual: Any, expected: float, tolerance: float = 1e-8) -> None:
    if not isinstance(actual, (int, float)) or not math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance
    ):
        raise RuntimeError(f"{name}={actual!r} != expected {expected!r}")


def compare_non_target_tensors(
    dense_model: torch.nn.Module,
    sparse_model: torch.nn.Module,
    ar_target_modules: list[str],
) -> dict[str, Any]:
    def tensor_map(model: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {
            **dict(model.named_parameters()),
            **dict(model.named_buffers()),
        }

    dense_tensors = tensor_map(dense_model)
    sparse_tensors = tensor_map(sparse_model)
    if set(dense_tensors) != set(sparse_tensors):
        missing = sorted(set(dense_tensors) - set(sparse_tensors))
        extra = sorted(set(sparse_tensors) - set(dense_tensors))
        raise RuntimeError(
            f"Loaded model tensor-key mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    target_keys = {f"{name}.weight" for name in ar_target_modules}
    missing_targets = sorted(target_keys - set(dense_tensors))
    if missing_targets:
        raise RuntimeError(f"AR target tensors missing from loaded model: {missing_targets[:5]}")

    compared = 0
    mismatches: list[str] = []
    for key in sorted(set(dense_tensors) - target_keys):
        dense = dense_tensors[key].detach().cpu()
        sparse = sparse_tensors[key].detach().cpu()
        if dense.dtype != sparse.dtype or dense.shape != sparse.shape or not torch.equal(dense, sparse):
            mismatches.append(key)
        compared += 1
    if mismatches:
        raise RuntimeError(
            "ATV changed non-target Reasoner tensors; first mismatches: "
            + ", ".join(mismatches[:10])
        )
    return {
        "target_tensor_count": len(target_keys),
        "non_target_tensor_count": compared,
        "non_target_bitwise_equal": compared,
        "non_target_mismatches": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--protocol", choices=("joint", "separate"), required=True)
    parser.add_argument("--preset", choices=("mmbench", "mmmu", "okvqa"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--expected-nsamples", type=int)
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--expected-ar-sparsity", type=float)
    parser.add_argument("--expected-alpha", type=float)
    parser.add_argument("--expected-dtype", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--expected-attention-implementation")
    parser.add_argument("--expected-min-image-pixels", type=int)
    parser.add_argument("--expected-max-image-pixels", type=int)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if state.get("phase") != "complete":
        raise RuntimeError(f"Pruning state is not complete: {state}")
    if metadata.get("protocol_short") != args.protocol:
        raise RuntimeError(f"Protocol mismatch: {metadata.get('protocol_short')} != {args.protocol}")
    if metadata.get("algorithm") != "ATV-Pruning":
        raise RuntimeError(f"Not an ATV checkpoint: {metadata.get('algorithm')!r}")
    if metadata.get("calibration_presets") != [args.preset]:
        raise RuntimeError(
            f"Calibration preset mismatch: {metadata.get('calibration_presets')!r} != {[args.preset]!r}"
        )
    scalar_expectations = (
        ("nsamples", args.expected_nsamples),
        ("seed", args.expected_seed),
        ("dtype", args.expected_dtype),
        ("attention_implementation", args.expected_attention_implementation),
        ("min_image_pixels", args.expected_min_image_pixels),
        ("max_image_pixels", args.expected_max_image_pixels),
    )
    for name, expected in scalar_expectations:
        if expected is not None and metadata.get(name) != expected:
            raise RuntimeError(
                f"Protocol metadata mismatch: {name}={metadata.get(name)!r} != {expected!r}"
            )
    if args.expected_ar_sparsity is not None:
        assert_float(
            "target_ar_linear_sparsity",
            metadata.get("target_ar_linear_sparsity"),
            args.expected_ar_sparsity,
        )
        assert_float(
            "achieved_ar_linear_sparsity",
            metadata.get("achieved_ar_linear_sparsity"),
            args.expected_ar_sparsity,
        )
    if args.expected_alpha is not None:
        assert_float("alpha", metadata.get("alpha"), args.expected_alpha)
    expected_variant = (
        "official_multimodal_visual_cosine"
        if args.protocol == "joint"
        else "text_only_zero_visual_ablation"
    )
    if metadata.get("algorithm_variant") != expected_variant:
        raise RuntimeError(
            f"ATV variant mismatch: {metadata.get('algorithm_variant')!r} != {expected_variant!r}"
        )
    if bool(metadata.get("official_multimodal_atv")) != (args.protocol == "joint"):
        raise RuntimeError("official_multimodal_atv flag contradicts the selected protocol")
    if metadata.get("partial_run") and not args.allow_partial:
        raise RuntimeError("Formal checkpoint cannot be a partial-layer run")
    if not metadata.get("partial_run"):
        if int(metadata.get("vision_layers_requested", -1)) != 0:
            raise RuntimeError("ATV checkpoint attempted vision-layer pruning")
        if int(metadata.get("ar_layers_requested", -1)) != 28:
            raise RuntimeError("Formal checkpoint did not prune all 28 AR layers")
    if not metadata.get("generator_excluded"):
        raise RuntimeError("Generator exclusion is not recorded")
    if (
        not metadata.get("vision_encoder_dense")
        or not metadata.get("projector_dense")
        or not metadata.get("embedding_norm_lm_head_dense")
    ):
        raise RuntimeError("Dense vision/projector/embedding/norm/lm_head contract changed")
    if float(metadata.get("vision_sparsity_target", -1)) != 0.0:
        raise RuntimeError("ATV metadata contains non-zero vision sparsity")
    layer_statistics = metadata.get("layer_statistics")
    expected_layers = int(metadata.get("ar_layers_requested", -1))
    if not isinstance(layer_statistics, list) or len(layer_statistics) != expected_layers:
        raise RuntimeError(
            f"Expected {expected_layers} ATV layer statistics, got "
            f"{None if not isinstance(layer_statistics, list) else len(layer_statistics)}"
        )
    for index, layer in enumerate(layer_statistics):
        if int(layer.get("sample_count", -1)) != int(metadata.get("nsamples", -2)):
            raise RuntimeError(f"ATV layer {index} sample count does not match metadata")
        if args.expected_alpha is not None:
            assert_float(f"ATV layer {index} alpha", layer.get("alpha"), args.expected_alpha)
        if args.protocol == "joint":
            if layer.get("mode") != "multimodal_atv":
                raise RuntimeError(f"Joint ATV layer {index} has wrong mode")
            if layer.get("mean_cosine_distance") is None or not layer.get("alpha_effective"):
                raise RuntimeError(f"Joint ATV layer {index} lacks effective visual selection")
            if any(int(value) <= 0 for value in layer.get("visual_tokens", [])):
                raise RuntimeError(f"Joint ATV layer {index} contains no visual tokens")
        else:
            if layer.get("mode") != "text_only_zero_visual":
                raise RuntimeError(f"Text-only ATV layer {index} has wrong mode")
            if layer.get("mean_cosine_distance") is not None or layer.get("selection_scale") is not None:
                raise RuntimeError(f"Text-only ATV layer {index} fabricated visual selection")
            if layer.get("alpha_effective"):
                raise RuntimeError(f"Text-only ATV layer {index} incorrectly enables alpha")
            if any(int(value) != 0 for value in layer.get("visual_tokens", [])):
                raise RuntimeError(f"Text-only ATV layer {index} contains visual tokens")
            if any(int(value) != 0 for value in layer.get("selected_visual_tokens", [])):
                raise RuntimeError(f"Text-only ATV layer {index} selected visual tokens")

    dataflow = metadata.get("ar_dataflow") or {}
    if int(dataflow.get("ar_forward_calls", 0)) <= 0:
        raise RuntimeError("ATV calibration did not execute the AR path")
    if args.protocol == "joint":
        if int(dataflow.get("vision_forward_calls", 0)) <= 0 or int(
            dataflow.get("projector_forward_calls", 0)
        ) <= 0:
            raise RuntimeError("Joint ATV did not execute dense vision and projector paths")
        if dataflow.get("visual_mask_mode") != "real_multimodal_token_types":
            raise RuntimeError("Joint ATV visual mask provenance is invalid")
    else:
        if int(dataflow.get("vision_forward_calls", -1)) != 0 or int(
            dataflow.get("projector_forward_calls", -1)
        ) != 0:
            raise RuntimeError("Text-only ATV executed a forbidden visual path")
        if dataflow.get("visual_mask_mode") != "forced_all_false_text_only":
            raise RuntimeError("Text-only ATV visual mask provenance is invalid")

    pruning_layers = (metadata.get("ar_pruning") or {}).get("layers") or []
    linear_reports = [
        report
        for layer in pruning_layers
        for report in (layer.get("linears") or {}).values()
    ]
    expected_linear_reports = expected_layers * 6
    if len(linear_reports) != expected_linear_reports:
        raise RuntimeError(
            f"Expected {expected_linear_reports} AR Linear reports, got {len(linear_reports)}"
        )
    for index, report in enumerate(linear_reports):
        if args.expected_ar_sparsity is not None:
            assert_float(
                f"AR Linear report {index} target_sparsity",
                report.get("target_sparsity"),
                args.expected_ar_sparsity,
            )
            assert_float(
                f"AR Linear report {index} actual_zero_ratio",
                report.get("actual_zero_ratio"),
                args.expected_ar_sparsity,
            )
        if int(report.get("activation_samples", -1)) != int(metadata.get("nsamples", -2)):
            raise RuntimeError(f"AR Linear report {index} activation sample count is invalid")

    verification = metadata.get("final_multimodal_verification") or {}
    if verification.get("skipped") or not verification.get("logits_finite"):
        raise RuntimeError("Saved ATV checkpoint lacks a finite final multimodal verification")
    forward_calls = verification.get("forward_calls") or {}
    if any(int(forward_calls.get(name, 0)) <= 0 for name in ("vision", "projector", "ar")):
        raise RuntimeError("Final multimodal verification did not execute the full Reasoner")

    checkpoint = run_dir / "checkpoint"
    required_files = ("config.json", "model.safetensors.index.json", "processor_config.json")
    for name in required_files:
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)

    device = torch.device(args.device)
    dtype_name = str(metadata["dtype"])
    model = Cosmos3EdgeForConditionalGeneration.from_pretrained(
        checkpoint,
        dtype=dtype_from_name(dtype_name),
        low_cpu_mem_usage=True,
        attn_implementation=str(metadata["attention_implementation"]),
    )
    model.eval()
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(checkpoint)

    audit = atv.module_audit(model)
    if audit["generator_modules"]:
        raise RuntimeError("Reloaded Reasoner unexpectedly contains Generator parameters")
    if int(audit["ar_target_linear_count"]) != 168:
        raise RuntimeError(f"Reloaded AR target count changed: {audit['ar_target_linear_count']}")
    if int(audit["ar_target_parameter_count"]) != 1_409_286_144:
        raise RuntimeError(
            f"Reloaded AR target parameter count changed: {audit['ar_target_parameter_count']}"
        )
    max_ar = int(metadata["ar_layers_requested"]) if metadata.get("partial_run") else 0
    actual_zero_report = atv.target_zero_report(
        model,
        max_ar_layers=max_ar,
    )
    assert_zero_report(actual_zero_report, metadata["zero_report"])
    dense_model = Cosmos3EdgeForConditionalGeneration.from_pretrained(
        Path(metadata["model_path"]).resolve(),
        dtype=dtype_from_name(dtype_name),
        low_cpu_mem_usage=True,
        attn_implementation=str(metadata["attention_implementation"]),
    )
    dense_model.eval()
    dense_model.requires_grad_(False)
    non_target_validation = compare_non_target_tensors(
        dense_model,
        model,
        audit["ar_linear_names"],
    )
    del dense_model
    gc.collect()

    model.to(device)

    calibration_args = SimpleNamespace(
        protocol=args.protocol,
        calibration_preset=[args.preset],
        preset_file=str(Path(__file__).with_name("calibration_presets.json")),
        calibration_json=[],
        ar_calibration_json=[],
        verification_json=[],
        image_root=[],
        nsamples=1,
        nsamples_per_file=0,
    )
    atv.apply_calibration_presets(calibration_args)
    calibration = atv.build_protocol_calibration(calibration_args)
    verification = atv.verify_full_multimodal_forward(
        model,
        processor,
        calibration.verification_records[0],
        device,
        4096,
        False,
        int(metadata.get("min_image_pixels", 65_536)),
        int(metadata.get("max_image_pixels", 1_048_576)),
    )
    validation = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "protocol": args.protocol,
        "preset": args.preset,
        "protocol_expectations": {
            "nsamples": args.expected_nsamples,
            "seed": args.expected_seed,
            "ar_sparsity": args.expected_ar_sparsity,
            "alpha": args.expected_alpha,
            "dtype": args.expected_dtype,
            "attention_implementation": args.expected_attention_implementation,
            "min_image_pixels": args.expected_min_image_pixels,
            "max_image_pixels": args.expected_max_image_pixels,
        },
        "partial_run": bool(metadata.get("partial_run")),
        "reload_zero_report": actual_zero_report,
        "non_target_tensor_validation": non_target_validation,
        "reload_multimodal_verification": verification,
        "checkpoint_bytes": sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file()),
        "validated": True,
    }
    validation_path = run_dir / "checkpoint_validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / ".checkpoint_validated").write_text(
        json.dumps({"validation": str(validation_path)}) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
