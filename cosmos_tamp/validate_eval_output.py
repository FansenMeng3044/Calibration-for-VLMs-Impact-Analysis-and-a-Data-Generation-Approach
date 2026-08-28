#!/usr/bin/env python3
"""Validate lmms-eval sample counts, metrics, and benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


TASK_INFO: dict[str, dict[str, Any]] = {
    "mmbench": {
        "task": "mmbench_en_dev_local",
        "expected": 4329,
        "metric": "exact_match_score,none",
    },
    "mmmu": {
        "task": "mmmu_val_local",
        "expected": 900,
        "metric": "exact_match_score,none",
    },
    "okvqa": {
        "task": "okvqa_val2014_local",
        "expected": 5046,
        "metric": "exact_match,none",
    },
}


def exactly_one(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {label}, found {len(paths)}: {paths}")
    return paths[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--benchmark", choices=tuple(TASK_INFO), required=True)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    info = TASK_INFO[args.benchmark]
    task = str(info["task"])
    expected = int(args.expected_count or info["expected"])
    results_path = exactly_one(
        [
            path
            for path in output.rglob("*_results.json")
            if "submissions" not in path.parts
        ],
        "aggregated results JSON",
    )
    sample_path = exactly_one(
        list(output.rglob(f"*_samples_{task}.jsonl")),
        f"{task} sample JSONL",
    )

    results = json.loads(results_path.read_text(encoding="utf-8"))
    if task not in results.get("results", {}):
        raise RuntimeError(f"Task {task} is absent from {results_path}")
    metric_key = str(info["metric"])
    metric_value = results["results"][task].get(metric_key)
    if not isinstance(metric_value, (int, float)) or not math.isfinite(float(metric_value)):
        raise RuntimeError(f"Invalid {task} metric {metric_key}: {metric_value!r}")

    count = 0
    with sample_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            responses = row.get("filtered_resps")
            if not isinstance(responses, list) or not responses:
                raise RuntimeError(
                    f"Missing filtered response at {sample_path}:{line_number}"
                )
            count += 1
    if count != expected:
        raise RuntimeError(f"{task} sample count={count}, expected={expected}")

    submissions = output / "submissions"
    if args.benchmark == "mmbench":
        exactly_one(list(submissions.glob("mmbench_en_dev_local_results.json")), "MMBench score JSON")
        exactly_one(list(submissions.glob("mmbench_en_dev_local_results.xlsx")), "MMBench XLSX")
    elif args.benchmark == "mmmu":
        for name in (
            "mmmu_val_local_scores.json",
            "mmmu_val_local_predictions.json",
            "mmmu_val_local_records.json",
        ):
            path = submissions / name
            if not path.is_file():
                raise FileNotFoundError(path)
            json.loads(path.read_text(encoding="utf-8"))
    else:
        exactly_one(list(submissions.glob("ok_vqa-test-submission-*.json")), "OK-VQA submission JSON")

    validation = {
        "benchmark": args.benchmark,
        "task": task,
        "output_dir": str(output),
        "results_path": str(results_path),
        "samples_path": str(sample_path),
        "sample_count": count,
        "expected_count": expected,
        "metric_key": metric_key,
        "metric_value": float(metric_value),
        "validated": True,
    }
    validation_path = output / f"validation_{args.benchmark}.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / f".done_{args.benchmark}").write_text(
        json.dumps({"validation": str(validation_path)}) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
