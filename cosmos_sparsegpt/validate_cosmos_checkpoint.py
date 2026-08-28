#!/usr/bin/env python3
"""Reload and validate a saved Cosmos SparseGPT Reasoner checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoProcessor, Cosmos3EdgeForConditionalGeneration

import cosmos_sparsegpt_prune as sparsegpt


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


def validate_sparsegpt_metadata(metadata: dict[str, Any], protocol: str) -> None:
    expected_protocol = f"cosmos_sparsegpt_{protocol}_reasoner"
    if metadata.get("protocol") != expected_protocol:
        raise RuntimeError(
            f"Protocol name mismatch: {metadata.get('protocol')} != {expected_protocol}"
        )
    if metadata.get("model_scope") != "reasoner_only":
        raise RuntimeError("SparseGPT model scope is not locked to the Reasoner")
    audit = metadata.get("module_audit") or {}
    if not audit.get("target_parameter_ids_unique"):
        raise RuntimeError("SparseGPT target Parameter IDs were not proven unique")
    if int(audit.get("vision_ar_parameter_id_overlap", -1)) != 0:
        raise RuntimeError("Vision and AR SparseGPT targets overlap")
    if audit.get("generator_modules"):
        raise RuntimeError("Generator modules appear in the saved module audit")

    before = metadata.get("non_target_dense_zero_report_before")
    after = metadata.get("non_target_dense_zero_report_after")
    if not isinstance(before, dict) or before != after:
        raise RuntimeError("Non-target Reasoner zero counts changed during SparseGPT")

    nsamples = int(metadata["nsamples"])
    if metadata.get("vision_calibration_sample_ids") != metadata.get("ar_calibration_sample_ids"):
        raise RuntimeError("Vision and AR calibration sample IDs are not aligned")
    sparsegpt_contract = metadata["sparsegpt"]
    budget_mode = str(sparsegpt_contract["budget_mode"])
    for branch in ("vision", "ar"):
        payload = metadata.get(f"{branch}_pruning") or {}
        layers = payload.get("layers") or []
        requested = int(metadata[f"{branch}_layers_requested"])
        if len(layers) != requested:
            raise RuntimeError(
                f"{branch} report has {len(layers)} layers, expected {requested}"
            )
        for layer in layers:
            linears = layer.get("linears") or {}
            if not linears:
                raise RuntimeError(f"Empty SparseGPT Linear report: {layer}")
            for name, report in linears.items():
                if int(report.get("activation_samples", -1)) != nsamples:
                    raise RuntimeError(f"{name}: activation sample count mismatch")
                if int(report.get("activation_hook_calls", -1)) != nsamples:
                    raise RuntimeError(f"{name}: activation hook count mismatch")
                if int(report.get("valid_activation_tokens", 0)) <= 0:
                    raise RuntimeError(f"{name}: no valid Hessian tokens")
                if report.get("budget_mode") != budget_mode:
                    raise RuntimeError(f"{name}: budget mode changed")
                if budget_mode == "exact_k_budget" and int(
                    report.get("mask_pruned_weights", -1)
                ) != int(report.get("requested_pruned_weights", -2)):
                    raise RuntimeError(f"{name}: exact SparseGPT mask budget was not met")
                if not report.get("finite_after"):
                    raise RuntimeError(f"{name}: non-finite reconstructed weights")
                shape = report.get("shape") or []
                hessian_shape = report.get("hessian_shape") or []
                if len(shape) != 2 or hessian_shape != [shape[1], shape[1]]:
                    raise RuntimeError(f"{name}: Hessian/weight shape mismatch")
                for stage in ("hessian", "inverse_hessian"):
                    chol = (report.get("cholesky") or {}).get(stage) or {}
                    if int(chol.get("attempts", 0)) <= 0:
                        raise RuntimeError(f"{name}: missing {stage} Cholesky report")
                    if int(chol.get("retries", -1)) > int(
                        sparsegpt_contract["max_cholesky_retries"]
                    ):
                        raise RuntimeError(f"{name}: unbounded Cholesky retries")

    vision_flow = metadata.get("vision_dataflow") or {}
    if (
        int(vision_flow.get("vision_forward_calls", -1)) != nsamples
        or int(vision_flow.get("projector_forward_calls", -1)) != 0
        or int(vision_flow.get("ar_forward_calls", -1)) != 0
    ):
        raise RuntimeError(f"Vision-only SparseGPT dataflow violated: {vision_flow}")
    ar_flow = metadata.get("ar_dataflow") or {}
    if int(ar_flow.get("ar_forward_calls", -1)) != nsamples:
        raise RuntimeError(f"AR SparseGPT call count mismatch: {ar_flow}")
    token_counts = ar_flow.get("token_counts") or []
    if len(token_counts) != nsamples:
        raise RuntimeError("AR token audit does not cover every calibration sample")
    if protocol == "joint":
        if (
            int(ar_flow.get("vision_forward_calls", -1)) != nsamples
            or int(ar_flow.get("projector_forward_calls", -1)) != nsamples
            or any(int(item.get("image", 0)) <= 0 or int(item.get("language", 0)) <= 0 for item in token_counts)
        ):
            raise RuntimeError(f"Joint AR Hessian was not genuinely multimodal: {ar_flow}")
    else:
        if (
            int(ar_flow.get("vision_forward_calls", -1)) != 0
            or int(ar_flow.get("projector_forward_calls", -1)) != 0
            or any(int(item.get("image", -1)) != 0 or int(item.get("video", -1)) != 0 or int(item.get("language", 0)) <= 0 for item in token_counts)
        ):
            raise RuntimeError(f"Separate AR Hessian was not language-only: {ar_flow}")

    verification = metadata.get("final_multimodal_verification") or {}
    if verification.get("skipped"):
        raise RuntimeError("Saved checkpoint skipped final multimodal verification")
    if verification.get("forward_calls") != {"vision": 1, "projector": 1, "ar": 1}:
        raise RuntimeError("Final checkpoint verification was not a full Reasoner forward")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--protocol", choices=("joint", "separate"), required=True)
    parser.add_argument("--preset", choices=("mmbench", "mmmu", "okvqa"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if state.get("phase") != "complete":
        raise RuntimeError(f"Pruning state is not complete: {state}")
    if metadata.get("protocol_short") != args.protocol:
        raise RuntimeError(f"Protocol mismatch: {metadata.get('protocol_short')} != {args.protocol}")
    if metadata.get("algorithm") != "SparseGPT":
        raise RuntimeError(f"Algorithm mismatch: {metadata.get('algorithm')}")
    sparsegpt_contract = metadata.get("sparsegpt") or {}
    for key in ("hessian", "importance", "reconstruction", "blocksize", "percdamp", "budget_mode"):
        if key not in sparsegpt_contract:
            raise RuntimeError(f"SparseGPT metadata is missing {key}")
    validate_sparsegpt_metadata(metadata, args.protocol)
    if metadata.get("partial_run") and not args.allow_partial:
        raise RuntimeError("Formal checkpoint cannot be a partial-layer run")
    if not metadata.get("partial_run"):
        if int(metadata.get("vision_layers_requested", -1)) != 27:
            raise RuntimeError("Formal checkpoint did not prune all 27 vision layers")
        if int(metadata.get("ar_layers_requested", -1)) != 28:
            raise RuntimeError("Formal checkpoint did not prune all 28 AR layers")
    if not metadata.get("generator_excluded"):
        raise RuntimeError("Generator exclusion is not recorded")
    if not metadata.get("projector_dense") or not metadata.get("embedding_norm_lm_head_dense"):
        raise RuntimeError("Dense projector/embedding/norm/lm_head contract changed")

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
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(checkpoint)

    audit = sparsegpt.module_audit(model)
    if audit["generator_modules"]:
        raise RuntimeError("Reloaded Reasoner unexpectedly contains Generator parameters")
    max_vision = int(metadata["vision_layers_requested"]) if metadata.get("partial_run") else 0
    max_ar = int(metadata["ar_layers_requested"]) if metadata.get("partial_run") else 0
    actual_zero_report = sparsegpt.target_zero_report(
        model,
        max_vision_layers=max_vision,
        max_ar_layers=max_ar,
    )
    assert_zero_report(actual_zero_report, metadata["zero_report"])

    calibration_args = SimpleNamespace(
        protocol=args.protocol,
        calibration_preset=[args.preset],
        preset_file=str(Path(__file__).with_name("calibration_presets.json")),
        calibration_json=[],
        vision_calibration_json=[],
        ar_calibration_json=[],
        image_root=[],
        nsamples=1,
        nsamples_per_file=0,
    )
    sparsegpt.apply_calibration_presets(calibration_args)
    calibration = sparsegpt.build_protocol_calibration(calibration_args)
    verification = sparsegpt.verify_full_multimodal_forward(
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
        "partial_run": bool(metadata.get("partial_run")),
        "reload_zero_report": actual_zero_report,
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
