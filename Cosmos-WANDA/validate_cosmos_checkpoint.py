#!/usr/bin/env python3
"""Reload and validate a saved Cosmos WANDA Reasoner checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoProcessor, Cosmos3EdgeForConditionalGeneration

import cosmos_wanda_prune as wanda


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

    audit = wanda.module_audit(model)
    if audit["generator_modules"]:
        raise RuntimeError("Reloaded Reasoner unexpectedly contains Generator parameters")
    max_vision = int(metadata["vision_layers_requested"]) if metadata.get("partial_run") else 0
    max_ar = int(metadata["ar_layers_requested"]) if metadata.get("partial_run") else 0
    actual_zero_report = wanda.target_zero_report(
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
    wanda.apply_calibration_presets(calibration_args)
    calibration = wanda.build_protocol_calibration(calibration_args)
    verification = wanda.verify_full_multimodal_forward(
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
