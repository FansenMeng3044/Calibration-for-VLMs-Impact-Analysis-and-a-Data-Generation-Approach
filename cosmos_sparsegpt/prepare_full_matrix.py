#!/usr/bin/env python3
"""Create a resumable 6-pruning/21-evaluation shared task queue."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import validate_calibration_alignment as alignment


BENCHMARKS = ("mmbench", "mmmu", "okvqa")
PROTOCOLS = ("joint", "separate")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def add_task(tasks_root: Path, payload: dict[str, Any]) -> None:
    task_dir = tasks_root / payload["id"]
    task_dir.mkdir(parents=True, exist_ok=True)
    config_path = task_dir / "task.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"Refusing to change existing task definition: {config_path}")
    else:
        write_json(config_path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vision-sparsity", type=float, default=0.5)
    parser.add_argument("--ar-sparsity", type=float, default=0.5)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument(
        "--budget-mode",
        choices=("exact_k_budget", "legacy_llava_threshold"),
        default="exact_k_budget",
    )
    parser.add_argument("--max-cholesky-retries", type=int, default=8)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--min-image-pixels", type=int, default=65_536)
    parser.add_argument("--max-image-pixels", type=int, default=1_048_576)
    args = parser.parse_args()

    code_root = Path(__file__).resolve().parent
    run_root = Path(args.run_root).resolve()
    controller = run_root / "controller"
    tasks_root = controller / "tasks"
    manifest_path = controller / "experiment.json"
    if run_root.exists() and any(run_root.iterdir()) and not manifest_path.is_file():
        raise RuntimeError(f"Refusing non-empty unrecognized run root: {run_root}")
    tasks_root.mkdir(parents=True, exist_ok=True)

    alignment_path = controller / "calibration_alignment.json"
    if not alignment_path.is_file():
        saved_argv = sys.argv
        try:
            sys.argv = [
                "validate_calibration_alignment.py",
                "--nsamples",
                str(args.nsamples),
                "--output",
                str(alignment_path),
            ]
            alignment.main()
        finally:
            sys.argv = saved_argv

    tracked_code = (
        "cosmos_sparsegpt_prune.py",
        "sparsegpt_core.py",
        "test_sparsegpt_core.py",
        "calibration_presets.json",
        "cosmos_lmms_plugin/models/cosmos3_edge.py",
        "run_three_eval.sh",
        "validate_calibration_alignment.py",
        "validate_cosmos_checkpoint.py",
        "validate_eval_output.py",
        "prepare_full_matrix.py",
        "run_full_matrix_task.py",
        "run_full_matrix_worker.sh",
        "aggregate_matrix_results.py",
    )
    code_hashes = {
        name: sha256(code_root / name)
        for name in tracked_code
        if (code_root / name).is_file()
    }
    eval_data = [
        Path("/private/workspace/hycui/project/Tamp/MMBench_eval/en/dev-00000-of-00001.parquet"),
        Path("/private/workspace/hycui/mfs/okvqa/okvqa_val2014_local.jsonl"),
        *sorted(Path("/private/workspace/hycui/project/Tamp/MMMU_single_image").glob("*/validation-*.parquet")),
    ]
    eval_data_manifest = {
        str(path): {
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": sha256(path),
        }
        for path in eval_data
    }
    model_root = Path("/private/workspace/hycui/model/Cosmos3-Edge")
    model_index = model_root / "model.safetensors.index.json"
    index_payload = json.loads(model_index.read_text(encoding="utf-8"))
    referenced_weight_files = sorted(
        {model_root / relative for relative in index_payload["weight_map"].values()}
    )
    model_files = [model_root / "config.json", model_index, *referenced_weight_files]
    missing_model_files = [str(path) for path in model_files if not path.is_file()]
    if missing_model_files:
        raise FileNotFoundError(f"Indexed Reasoner weight files are missing: {missing_model_files}")
    model_manifest = {
        str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in model_files
    }
    external_eval_code = [
        Path("/private/workspace/hycui/project/Tamp/lmms_tasks/mmbench_en_dev_local.yaml"),
        Path("/private/workspace/hycui/project/Tamp/lmms_tasks/mmbench_local_utils.py"),
        Path("/private/workspace/hycui/project/Tamp/lmms_tasks/mmmu_local/mmmu_val_local.yaml"),
        Path("/private/workspace/hycui/project/Tamp/lmms_tasks/mmmu_local/mmmu_local_utils.py"),
        Path("/private/workspace/hycui/project/Tamp/lmms_tasks/okvqa_local/okvqa_val2014_local.yaml"),
        Path("/private/workspace/hycui/project/Tamp/lmms_tasks/okvqa_local/okvqa_local_utils.py"),
    ]
    manifest = {
        "schema_version": 1,
        "created_iso": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_root": str(run_root),
        "model_path": "/private/workspace/hycui/model/Cosmos3-Edge",
        "protocols": list(PROTOCOLS),
        "calibration_sources": list(BENCHMARKS),
        "evaluation_benchmarks": list(BENCHMARKS),
        "nsamples": args.nsamples,
        "seed": args.seed,
        "vision_sparsity": args.vision_sparsity,
        "ar_sparsity": args.ar_sparsity,
        "blocksize": args.blocksize,
        "percdamp": args.percdamp,
        "budget_mode": args.budget_mode,
        "max_cholesky_retries": args.max_cholesky_retries,
        "dtype": args.dtype,
        "attention_implementation": "eager",
        "enable_thinking": False,
        "max_length": 4096,
        "min_image_pixels": args.min_image_pixels,
        "max_image_pixels": args.max_image_pixels,
        "precision_label": f"dense/pruned unquantized {args.dtype}",
        "expected_eval_samples": {"mmbench": 4329, "mmmu": 900, "okvqa": 5046},
        "calibration_alignment": str(alignment_path),
        "code_sha256": code_hashes,
        "external_eval_code_sha256": {
            str(path): sha256(path) for path in external_eval_code
        },
        "base_model_files": model_manifest,
        "eval_data_files": eval_data_manifest,
        "calibration_policy": "transductive_input_only",
        "calibration_eval_overlap": {
            "mmbench": {
                "calibration_rows": 128,
                "overlapping_eval_inputs": 128,
                "calibration_split": "dev",
                "evaluation_split": "dev",
            },
            "mmmu": {
                "calibration_rows": 128,
                "overlapping_eval_inputs": 113,
                "calibration_split_counts": {"dev": 15, "validation": 113},
                "evaluation_split": "validation",
            },
            "okvqa": {
                "calibration_rows": 128,
                "overlapping_eval_inputs": 0,
                "calibration_split": "train2014",
                "evaluation_split": "val2014",
            },
        },
        "calibration_policy_note": (
            "Input-only transductive calibration. MMBench and MMMU overlap their same-name evaluation "
            "inputs as quantified in calibration_eval_overlap; OK-VQA is train2014 to val2014. "
            "No answer labels are consumed during SparseGPT Hessian statistics. These results must not "
            "be described as strictly held-out for MMBench or MMMU."
        ),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        locked = (
            "run_root",
            "model_path",
            "nsamples",
            "seed",
            "vision_sparsity",
            "ar_sparsity",
            "blocksize",
            "percdamp",
            "budget_mode",
            "max_cholesky_retries",
            "dtype",
            "min_image_pixels",
            "max_image_pixels",
        )
        for key in locked:
            if existing.get(key) != manifest.get(key):
                raise RuntimeError(f"Experiment manifest mismatch for {key}: {existing.get(key)} != {manifest.get(key)}")
    else:
        write_json(manifest_path, manifest)
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (controller / "pip_freeze.txt").write_text(freeze, encoding="utf-8")

    gate_id = "000_gate_partial_save_reload_eval"
    add_task(
        tasks_root,
        {
            "id": gate_id,
            "kind": "gate",
            "priority": 0,
            "protocol": "joint",
            "preset": "mmbench",
            "output_dir": str(run_root / "gate" / "partial_save_reload"),
            "eval_output_dir": str(run_root / "gate" / "partial_eval"),
        },
    )

    priority = 100
    prune_ids: dict[tuple[str, str], str] = {}
    for protocol in PROTOCOLS:
        for preset in BENCHMARKS:
            task_id = f"{priority:03d}_prune_{protocol}_{preset}"
            prune_ids[(protocol, preset)] = task_id
            add_task(
                tasks_root,
                {
                    "id": task_id,
                    "kind": "prune",
                    "priority": priority,
                    "protocol": protocol,
                    "preset": preset,
                    "output_dir": str(run_root / "pruning" / protocol / preset),
                    "nsamples": args.nsamples,
                    "seed": args.seed,
                    "vision_sparsity": args.vision_sparsity,
                    "ar_sparsity": args.ar_sparsity,
                    "blocksize": args.blocksize,
                    "percdamp": args.percdamp,
                    "budget_mode": args.budget_mode,
                    "max_cholesky_retries": args.max_cholesky_retries,
                    "dtype": args.dtype,
                    "min_image_pixels": args.min_image_pixels,
                    "max_image_pixels": args.max_image_pixels,
                    "requires_gate": True,
                },
            )
            priority += 1

    for index, benchmark in enumerate(BENCHMARKS):
        add_task(
            tasks_root,
            {
                "id": f"{200 + index:03d}_eval_dense_{benchmark}",
                "kind": "eval",
                "priority": 200 + index,
                "model_id": "dense",
                "model_path": "/private/workspace/hycui/model/Cosmos3-Edge",
                "benchmark": benchmark,
                "output_dir": str(run_root / "eval" / "dense" / benchmark),
                "expected_count": {"mmbench": 4329, "mmmu": 900, "okvqa": 5046}[benchmark],
                "dependency": None,
                "requires_gate": False,
            },
        )

    priority = 300
    eval_task_ids: list[str] = []
    for protocol in PROTOCOLS:
        for preset in BENCHMARKS:
            prune_output = run_root / "pruning" / protocol / preset
            for benchmark in BENCHMARKS:
                task_id = f"{priority:03d}_eval_{protocol}_{preset}_{benchmark}"
                eval_task_ids.append(task_id)
                add_task(
                    tasks_root,
                    {
                        "id": task_id,
                        "kind": "eval",
                        "priority": priority,
                        "model_id": f"{protocol}_{preset}",
                        "model_path": str(prune_output / "checkpoint"),
                        "benchmark": benchmark,
                        "output_dir": str(run_root / "eval" / f"{protocol}_{preset}" / benchmark),
                        "expected_count": {"mmbench": 4329, "mmmu": 900, "okvqa": 5046}[benchmark],
                        "dependency": prune_ids[(protocol, preset)],
                        "requires_gate": True,
                    },
                )
                priority += 1

    dense_task_ids = [f"{200 + index:03d}_eval_dense_{benchmark}" for index, benchmark in enumerate(BENCHMARKS)]
    add_task(
        tasks_root,
        {
            "id": "999_aggregate_validated_matrix",
            "kind": "aggregate",
            "priority": 999,
            "run_root": str(run_root),
            "dependencies": [*dense_task_ids, *eval_task_ids],
            "requires_gate": True,
        },
    )

    task_count = len(list(tasks_root.glob("*/task.json")))
    summary = {
        "run_root": str(run_root),
        "task_count": task_count,
        "gate_tasks": 1,
        "prune_tasks": 6,
        "dense_eval_tasks": 3,
        "pruned_eval_tasks": 18,
        "aggregate_tasks": 1,
    }
    write_json(controller / "queue_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
