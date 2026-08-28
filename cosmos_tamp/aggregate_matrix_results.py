#!/usr/bin/env python3
"""Aggregate validated dense and 6x3 pruned Cosmos TAMP results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


BENCHMARKS = ("mmbench", "mmmu", "okvqa")
MODELS = (
    "dense",
    "joint_mmbench",
    "joint_mmmu",
    "joint_okvqa",
    "separate_mmbench",
    "separate_mmmu",
    "separate_okvqa",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    rows: list[dict[str, Any]] = []
    nested: dict[str, Any] = {}
    for model_id in MODELS:
        checkpoint_metadata: dict[str, Any] | None = None
        if model_id == "dense":
            protocol = "dense"
            calibration = "none"
        else:
            protocol, calibration = model_id.split("_", 1)
            checkpoint_validation = root / "pruning" / protocol / calibration / "checkpoint_validation.json"
            if not checkpoint_validation.is_file():
                raise FileNotFoundError(checkpoint_validation)
            checkpoint_report = json.loads(checkpoint_validation.read_text(encoding="utf-8"))
            if not checkpoint_report.get("validated"):
                raise RuntimeError(f"Unvalidated checkpoint: {checkpoint_validation}")
            metadata_path = root / "pruning" / protocol / calibration / "metadata.json"
            checkpoint_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        nested[model_id] = {}
        for benchmark in BENCHMARKS:
            validation_path = root / "eval" / model_id / benchmark / f"validation_{benchmark}.json"
            if not validation_path.is_file():
                raise FileNotFoundError(validation_path)
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            if not validation.get("validated"):
                raise RuntimeError(f"Unvalidated result: {validation_path}")
            row = {
                "model_id": model_id,
                "protocol": protocol,
                "calibration": calibration,
                "algorithm_variant": (
                    "dense" if checkpoint_metadata is None else checkpoint_metadata["algorithm_variant"]
                ),
                "max_sparsity_per_linear": (
                    None
                    if checkpoint_metadata is None
                    else checkpoint_metadata["max_sparsity_per_linear"]
                ),
                "target_ar_sparsity": (
                    None if checkpoint_metadata is None else checkpoint_metadata["target_ar_linear_sparsity"]
                ),
                "achieved_ar_sparsity": (
                    None if checkpoint_metadata is None else checkpoint_metadata["achieved_ar_linear_sparsity"]
                ),
                "evaluation": benchmark,
                "metric_key": validation["metric_key"],
                "metric_value": validation["metric_value"],
                "sample_count": validation["sample_count"],
                "validation_path": str(validation_path),
            }
            rows.append(row)
            nested[model_id][benchmark] = row
    if len(rows) != 21:
        raise RuntimeError(f"Expected 21 validated results, found {len(rows)}")

    output_dir = root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cosmos_tamp_6x3_plus_dense.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "cosmos_tamp_6x3_plus_dense.json"
    json_path.write_text(
        json.dumps({"rows": rows, "matrix": nested}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    complete = {
        "validated_checkpoint_count": 6,
        "validated_result_count": len(rows),
        "csv": str(csv_path),
        "json": str(json_path),
    }
    (root / ".matrix_complete").write_text(json.dumps(complete) + "\n", encoding="utf-8")
    print(json.dumps(complete, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
