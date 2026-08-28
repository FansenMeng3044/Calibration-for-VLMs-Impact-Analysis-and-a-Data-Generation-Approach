#!/usr/bin/env python3
"""Reload and strictly validate a saved Cosmos TAMP Reasoner checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoProcessor, Cosmos3EdgeForConditionalGeneration

import cosmos_tamp_prune as tamp


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
            "TAMP changed non-target Reasoner tensors; first mismatches: "
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
    parser.add_argument("--expected-max-sparsity-per-linear", type=float)
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
    if metadata.get("algorithm") != "TAMP":
        raise RuntimeError(f"Not a TAMP checkpoint: {metadata.get('algorithm')!r}")
    if metadata.get("algorithm_components") != ["DAS", "AMIA", "WANDA"]:
        raise RuntimeError(f"Invalid TAMP components: {metadata.get('algorithm_components')!r}")
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
        achieved = float(metadata.get("achieved_ar_linear_sparsity", -1))
        if achieved > args.expected_ar_sparsity + 1e-12 or args.expected_ar_sparsity - achieved > 1e-3:
            raise RuntimeError(
                "achieved_ar_linear_sparsity is outside the row-floor tolerance: "
                f"{achieved} vs target {args.expected_ar_sparsity}"
            )
    if args.expected_max_sparsity_per_linear is not None:
        assert_float(
            "max_sparsity_per_linear",
            metadata.get("max_sparsity_per_linear"),
            args.expected_max_sparsity_per_linear,
        )
    expected_variant = (
        "joint_multimodal_reasoner_ar"
        if args.protocol == "joint"
        else "separate_text_only_reasoner_ar"
    )
    if metadata.get("algorithm_variant") != expected_variant:
        raise RuntimeError(
            f"TAMP variant mismatch: {metadata.get('algorithm_variant')!r} != {expected_variant!r}"
        )
    if metadata.get("partial_run") and not args.allow_partial:
        raise RuntimeError("Formal checkpoint cannot be a partial-layer run")
    if not metadata.get("partial_run"):
        if int(metadata.get("vision_layers_requested", -1)) != 0:
            raise RuntimeError("TAMP checkpoint attempted vision-layer pruning")
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
        raise RuntimeError("TAMP metadata contains non-zero vision sparsity")
    das = metadata.get("das") or {}
    layer_statistics = das.get("layers")
    expected_layers = int(metadata.get("ar_layers_requested", -1))
    if not isinstance(layer_statistics, list) or len(layer_statistics) != expected_layers:
        raise RuntimeError(
            f"Expected {expected_layers} DAS layer statistics, got "
            f"{None if not isinstance(layer_statistics, list) else len(layer_statistics)}"
        )
    for index, layer in enumerate(layer_statistics):
        linears = layer.get("linears") or {}
        if len(linears) != 6:
            raise RuntimeError(f"DAS layer {index} expected 6 Linear reports, got {len(linears)}")
        for name, report in linears.items():
            expected_terms = ["v", "l", "vl"] if args.protocol == "joint" else ["l"]
            if report.get("defined_terms") != expected_terms:
                raise RuntimeError(
                    f"DAS layer {index}/{name} terms={report.get('defined_terms')} != {expected_terms}"
                )
            expected_formula = (
                "(1-s_v)+(1-s_l)+(1-s_vl)" if args.protocol == "joint" else "3*(1-s_l)"
            )
            if report.get("formula") != expected_formula:
                raise RuntimeError(f"DAS layer {index}/{name} has wrong formula")
            if int(report.get("calls", -1)) != int(metadata.get("nsamples", -2)):
                raise RuntimeError(f"DAS layer {index}/{name} sample count mismatch")

    allocation = metadata.get("sparsity_allocation") or {}
    if allocation.get("granularity") != "per_linear_tensor":
        raise RuntimeError("TAMP allocation is not per-Linear tensor")
    if int(allocation.get("target_linear_count", -1)) != expected_layers * 6:
        raise RuntimeError("TAMP allocation target count does not match the pruned layers")
    if args.expected_ar_sparsity is not None:
        assert_float(
            "allocated_sparsity",
            allocation.get("allocated_sparsity"),
            args.expected_ar_sparsity,
            tolerance=1e-8,
        )

    dataflow = metadata.get("ar_dataflow") or {}
    if int(dataflow.get("ar_forward_calls", 0)) <= 0:
        raise RuntimeError("TAMP calibration did not execute the AR path")
    token_counts = dataflow.get("token_counts") or []
    if len(token_counts) != int(metadata.get("nsamples", -1)):
        raise RuntimeError("TAMP dataflow token-count records do not match nsamples")
    if args.protocol == "joint":
        if int(dataflow.get("vision_forward_calls", 0)) <= 0 or int(
            dataflow.get("projector_forward_calls", 0)
        ) <= 0:
            raise RuntimeError("Joint TAMP did not execute dense vision and projector paths")
        if dataflow.get("visual_mask_mode") != "real_multimodal_token_types":
            raise RuntimeError("Joint TAMP visual mask provenance is invalid")
        if any(
            int(item.get("image", 0)) <= 0
            or int(item.get("language", 0)) <= 0
            or int(item.get("video", -1)) != 0
            for item in token_counts
        ):
            raise RuntimeError("Joint TAMP contains a sample without real image+language tokens")
    else:
        if int(dataflow.get("vision_forward_calls", -1)) != 0 or int(
            dataflow.get("projector_forward_calls", -1)
        ) != 0:
            raise RuntimeError("Text-only TAMP executed a forbidden visual path")
        if dataflow.get("visual_mask_mode") != "forced_all_false_text_only":
            raise RuntimeError("Text-only TAMP visual mask provenance is invalid")
        if any(
            int(item.get("image", -1)) != 0
            or int(item.get("video", -1)) != 0
            or int(item.get("language", 0)) <= 0
            for item in token_counts
        ):
            raise RuntimeError("Text-only TAMP contains forbidden visual tokens")

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
        allocated = float(report.get("allocated_sparsity", -1))
        if not 0.0 <= allocated <= float(metadata.get("max_sparsity_per_linear", -1)) + 1e-12:
            raise RuntimeError(f"AR Linear report {index} has invalid DAS allocation {allocated}")
        if not math.isclose(float(report.get("target_sparsity", -1)), allocated, abs_tol=1e-12):
            raise RuntimeError(f"AR Linear report {index} target/allocation mismatch")
        columns = int(report.get("shape", [0, 0])[1])
        expected_row_zeros = int(columns * allocated)
        if int(report.get("pruned_per_output_row", -1)) != expected_row_zeros:
            raise RuntimeError(f"AR Linear report {index} violates rowwise floor rule")
        amia = report.get("amia") or {}
        if int(amia.get("calls", -1)) != int(metadata.get("nsamples", -2)):
            raise RuntimeError(f"AR Linear report {index} AMIA sample count is invalid")
        if int(amia.get("selected_tokens", 0)) <= 0:
            raise RuntimeError(f"AR Linear report {index} AMIA selected no tokens")

    verification = metadata.get("final_multimodal_verification") or {}
    if verification.get("skipped") or not verification.get("logits_finite"):
        raise RuntimeError("Saved TAMP checkpoint lacks a finite final multimodal verification")
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

    audit = tamp.module_audit(model)
    if audit["generator_modules"]:
        raise RuntimeError("Reloaded Reasoner unexpectedly contains Generator parameters")
    if int(audit["ar_target_linear_count"]) != 168:
        raise RuntimeError(f"Reloaded AR target count changed: {audit['ar_target_linear_count']}")
    if int(audit["ar_target_parameter_count"]) != 1_409_286_144:
        raise RuntimeError(
            f"Reloaded AR target parameter count changed: {audit['ar_target_parameter_count']}"
        )
    expected_allowlist = [f"{name}.weight" for name in audit["ar_linear_names"]]
    if metadata.get("target_allowlist") != expected_allowlist:
        raise RuntimeError("Saved TAMP target allow-list differs from the reloaded Reasoner")
    expected_allowlist_sha256 = hashlib.sha256(
        "\n".join(expected_allowlist).encode("utf-8")
    ).hexdigest()
    if metadata.get("target_allowlist_sha256") != expected_allowlist_sha256:
        raise RuntimeError("Saved TAMP target allow-list hash is invalid")
    max_ar = int(metadata["ar_layers_requested"]) if metadata.get("partial_run") else 0
    actual_zero_report = tamp.target_zero_report(
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
    tamp.apply_calibration_presets(calibration_args)
    calibration = tamp.build_protocol_calibration(calibration_args)
    verification = tamp.verify_full_multimodal_forward(
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
            "max_sparsity_per_linear": args.expected_max_sparsity_per_linear,
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
