#!/usr/bin/env python3
"""Execute one task from the shared Cosmos SparseGPT matrix queue."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback
from typing import Any


CODE_ROOT = Path(__file__).resolve().parent
PYTHON_BIN = "/private/workspace/hycui/envs/cosmos3-edge/bin/python"
MODEL_PATH = "/private/workspace/hycui/model/Cosmos3-Edge"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(command: list[str], env: dict[str, str]) -> None:
    print("COMMAND", json.dumps(command), flush=True)
    subprocess.run(command, env=env, check=True)


def ensure_clean_or_validated(path: Path, marker: str) -> bool:
    if (path / marker).is_file():
        print(f"SKIP validated output: {path}", flush=True)
        return True
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Refusing non-empty incomplete output: {path}")
    return False


def validate_checkpoint(run_dir: Path, protocol: str, preset: str, env: dict[str, str], allow_partial: bool = False) -> None:
    command = [
        PYTHON_BIN,
        str(CODE_ROOT / "validate_cosmos_checkpoint.py"),
        "--run-dir",
        str(run_dir),
        "--protocol",
        protocol,
        "--preset",
        preset,
        "--device",
        "cuda:0",
    ]
    if allow_partial:
        command.append("--allow-partial")
    run(command, env)


def run_prune(config: dict[str, Any], env: dict[str, str]) -> None:
    output = Path(config["output_dir"])
    if ensure_clean_or_validated(output, ".checkpoint_validated"):
        return
    protocol = str(config["protocol"])
    preset = str(config["preset"])
    preflight = [
        PYTHON_BIN,
        str(CODE_ROOT / "cosmos_sparsegpt_prune.py"),
        "--protocol",
        protocol,
        "--calibration-preset",
        preset,
        "--nsamples",
        str(config["nsamples"]),
        "--preflight-only",
    ]
    run(preflight, env)
    command = [
        PYTHON_BIN,
        str(CODE_ROOT / "cosmos_sparsegpt_prune.py"),
        "--protocol",
        protocol,
        "--calibration-preset",
        preset,
        "--model-path",
        MODEL_PATH,
        "--nsamples",
        str(config["nsamples"]),
        "--seed",
        str(config["seed"]),
        "--vision-sparsity",
        str(config["vision_sparsity"]),
        "--ar-sparsity",
        str(config["ar_sparsity"]),
        "--blocksize",
        str(config["blocksize"]),
        "--percdamp",
        str(config["percdamp"]),
        "--budget-mode",
        str(config["budget_mode"]),
        "--max-cholesky-retries",
        str(config["max_cholesky_retries"]),
        "--device",
        "cuda:0",
        "--dtype",
        str(config["dtype"]),
        "--attn-implementation",
        "eager",
        "--min-image-pixels",
        str(config.get("min_image_pixels", 65_536)),
        "--max-image-pixels",
        str(config.get("max_image_pixels", 1_048_576)),
        "--save-model",
        "--output-dir",
        str(output),
    ]
    run(command, env)
    validate_checkpoint(output, protocol, preset, env)


def run_eval(config: dict[str, Any], env: dict[str, str]) -> None:
    output = Path(config["output_dir"])
    marker = f".done_{config['benchmark']}"
    if ensure_clean_or_validated(output, marker):
        return
    run(
        [
            "bash",
            str(CODE_ROOT / "run_three_eval.sh"),
            str(config["model_path"]),
            str(config["benchmark"]),
            str(output),
        ],
        env,
    )
    run(
        [
            PYTHON_BIN,
            str(CODE_ROOT / "validate_eval_output.py"),
            "--output-dir",
            str(output),
            "--benchmark",
            str(config["benchmark"]),
            "--expected-count",
            str(config["expected_count"]),
        ],
        env,
    )


def run_gate(config: dict[str, Any], env: dict[str, str]) -> None:
    output = Path(config["output_dir"])
    eval_output = Path(config["eval_output_dir"])
    if not (output / ".checkpoint_validated").is_file():
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"Refusing non-empty incomplete gate output: {output}")
        run(
            [
                PYTHON_BIN,
                str(CODE_ROOT / "cosmos_sparsegpt_prune.py"),
                "--protocol",
                str(config["protocol"]),
                "--calibration-preset",
                str(config["preset"]),
                "--model-path",
                MODEL_PATH,
                "--nsamples",
                "1",
                "--seed",
                "42",
                "--vision-sparsity",
                "0.5",
                "--ar-sparsity",
                "0.5",
                "--blocksize",
                "128",
                "--percdamp",
                "0.01",
                "--budget-mode",
                "exact_k_budget",
                "--max-cholesky-retries",
                "8",
                "--max-vision-layers",
                "1",
                "--max-ar-layers",
                "1",
                "--allow-partial-save",
                "--device",
                "cuda:0",
                "--dtype",
                "bfloat16",
                "--attn-implementation",
                "eager",
                "--min-image-pixels",
                "65536",
                "--max-image-pixels",
                "1048576",
                "--save-model",
                "--output-dir",
                str(output),
            ],
            env,
        )
        validate_checkpoint(output, str(config["protocol"]), str(config["preset"]), env, allow_partial=True)

    gate_markers = [eval_output / f".done_{name}" for name in ("mmbench", "mmmu", "okvqa")]
    if all(path.is_file() for path in gate_markers):
        print(f"SKIP validated gate evaluation: {eval_output}", flush=True)
        return
    if eval_output.exists() and any(eval_output.iterdir()):
        raise RuntimeError(f"Refusing non-empty incomplete gate eval output: {eval_output}")
    gate_env = dict(env)
    gate_env["TAMP_EVAL_LIMIT"] = "1"
    run(
        [
            "bash",
            str(CODE_ROOT / "run_three_eval.sh"),
            str(output / "checkpoint"),
            "all",
            str(eval_output),
        ],
        gate_env,
    )
    for benchmark in ("mmbench", "mmmu", "okvqa"):
        run(
            [
                PYTHON_BIN,
                str(CODE_ROOT / "validate_eval_output.py"),
                "--output-dir",
                str(eval_output),
                "--benchmark",
                benchmark,
                "--expected-count",
                "1",
            ],
            env,
        )


def run_aggregate(config: dict[str, Any], env: dict[str, str]) -> None:
    run(
        [
            PYTHON_BIN,
            str(CODE_ROOT / "aggregate_matrix_results.py"),
            "--run-root",
            str(config["run_root"]),
        ],
        env,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-json", required=True)
    parser.add_argument("--physical-gpu", required=True)
    parser.add_argument("--worker-id", required=True)
    args = parser.parse_args()

    task_path = Path(args.task_json).resolve()
    task_dir = task_path.parent
    config = json.loads(task_path.read_text(encoding="utf-8"))
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.physical_gpu),
            "GPU_ID": str(args.physical_gpu),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(
        task_dir / "status.json",
        {
            "phase": "running",
            "task_id": config["id"],
            "kind": config["kind"],
            "worker_id": args.worker_id,
            "hostname": socket.gethostname(),
            "physical_gpu": str(args.physical_gpu),
            "pid": os.getpid(),
            "started_iso": started,
        },
    )
    try:
        if config["kind"] == "prune":
            run_prune(config, env)
        elif config["kind"] == "eval":
            run_eval(config, env)
        elif config["kind"] == "gate":
            run_gate(config, env)
        elif config["kind"] == "aggregate":
            run_aggregate(config, env)
        else:
            raise ValueError(f"Unknown task kind: {config['kind']}")
        completed = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(
            task_dir / "status.json",
            {
                "phase": "complete",
                "task_id": config["id"],
                "kind": config["kind"],
                "worker_id": args.worker_id,
                "hostname": socket.gethostname(),
                "physical_gpu": str(args.physical_gpu),
                "pid": os.getpid(),
                "started_iso": started,
                "completed_iso": completed,
            },
        )
        (task_dir / ".done").write_text(completed + "\n", encoding="utf-8")
        return 0
    except Exception as exc:
        failed = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(
            task_dir / "status.json",
            {
                "phase": "failed",
                "task_id": config["id"],
                "kind": config["kind"],
                "worker_id": args.worker_id,
                "hostname": socket.gethostname(),
                "physical_gpu": str(args.physical_gpu),
                "pid": os.getpid(),
                "started_iso": started,
                "failed_iso": failed,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
