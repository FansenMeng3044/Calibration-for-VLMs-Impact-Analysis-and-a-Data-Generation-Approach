#!/usr/bin/env python3
"""
Export MMMU single-image eval results (by discipline) for all MMMU-calibrated checkpoints in LAVIS_backup to a CSV.

For each checkpoint matching:
  pruned_checkpoint/okvqa_mmmu_*.pth
this script runs:
  python scripts/blip2/mmmu_eval_by_discipline.py --ckpt CKPT ...
captures stdout, parses:
  - Overall accuracy
  - Per-discipline accuracy and sample count
and writes a CSV table.

Usage (from LAVIS_backup root):
  conda activate ecoflap
  cd /root/autodl-tmp/LAVIS_backup
  python scripts/blip2/export_mmmu_results_to_csv.py \
    --mmmu_root /root/autodl-tmp/MMMU_single_image \
    --split test \
    --batch_size 4 \
    --device cuda \
    --gpu 0 \
    --outfile mmmu_results_lavisbackup.csv

Notes:
- It is recommended to set HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 in the environment
  if models/tokenizers are already cached, to avoid network calls.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import subprocess
from typing import Dict, List, Tuple


def run_eval_for_ckpt(
    ckpt_path: str,
    mmmu_root: str,
    split: str,
    batch_size: int,
    device: str,
    gpu: str,
) -> str:
    """Run mmmu_eval_by_discipline.py for a single checkpoint and return stdout as text."""
    cmd = [
        "python",
        "scripts/blip2/mmmu_eval_by_discipline.py",
        "--mmmu_root",
        mmmu_root,
        "--split",
        split,
        "--ckpt",
        ckpt_path,
        "--batch_size",
        str(batch_size),
        "--device",
        device,
    ]
    env = os.environ.copy()
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        print(f"[WARN] Eval for {ckpt_path} exited with code {proc.returncode}")
    return proc.stdout


def parse_eval_output(stdout: str) -> Tuple[float, int, Dict[str, Tuple[float, int]]]:
    """
    Parse stdout from mmmu_eval_by_discipline.py.

    Returns:
        overall_acc (float),
        total_n (int),
        per_disc: {discipline: (acc, n)}
    """
    overall_acc = None
    total_n = None
    per_disc: Dict[str, Tuple[float, int]] = {}

    lines = stdout.splitlines()
    header_re = re.compile(r"^===== MMMU single-image (\w+) \(n=(\d+)\) =====")
    overall_re = re.compile(r"^Overall accuracy:\s*([0-9.]+)%")
    disc_re = re.compile(r"^\s*(.+?):\s*([0-9.]+)%\s*\((\d+)\)")

    in_block = False
    for line in lines:
        m = header_re.match(line)
        if m:
            in_block = True
            try:
                total_n = int(m.group(2))
            except ValueError:
                total_n = None
            continue
        if not in_block:
            continue
        m = overall_re.match(line)
        if m:
            overall_acc = float(m.group(1))
            continue
        m = disc_re.match(line)
        if m:
            name = m.group(1).strip()
            acc = float(m.group(2))
            n = int(m.group(3))
            per_disc[name] = (acc, n)

    if overall_acc is None or total_n is None:
        raise ValueError("Failed to parse MMMU eval output; header/overall missing.")
    return overall_acc, total_n, per_disc


def main():
    parser = argparse.ArgumentParser(
        description="Export MMMU eval (by discipline) for all MMMU-calibrated checkpoints to CSV (LAVIS_backup)."
    )
    parser.add_argument(
        "--ckpt_glob",
        default="pruned_checkpoint/okvqa_mmmu_*.pth",
        help="Glob pattern for MMMU-calibrated checkpoints.",
    )
    parser.add_argument(
        "--mmmu_root",
        default="/root/autodl-tmp/MMMU_single_image",
        help="MMMU_single_image root.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["dev", "validation", "test"],
        help="MMMU split to evaluate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for MMMU eval (use small value to avoid OOM).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help='Device for eval, e.g. "cuda" or "cpu".',
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help="CUDA_VISIBLE_DEVICES to use (e.g. 0,1). Empty string to leave unchanged.",
    )
    parser.add_argument(
        "--outfile",
        default="mmmu_results_lavisbackup.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    ckpts = sorted(glob.glob(args.ckpt_glob))
    if not ckpts:
        print(f"[WARN] No checkpoints found for pattern: {args.ckpt_glob}")
        return

    print(f"[INFO] Found {len(ckpts)} checkpoints:")
    for p in ckpts:
        print(f"  - {p}")

    rows: List[Dict[str, str]] = []

    for ckpt in ckpts:
        print(f"\n[INFO] Running MMMU eval for ckpt: {ckpt}")
        stdout = run_eval_for_ckpt(
            ckpt_path=ckpt,
            mmmu_root=args.mmmu_root,
            split=args.split,
            batch_size=args.batch_size,
            device=args.device,
            gpu=args.gpu,
        )
        try:
            overall_acc, total_n, per_disc = parse_eval_output(stdout)
        except Exception as e:
            print(f"[WARN] Failed to parse output for {ckpt}: {e}")
            continue

        base = os.path.basename(ckpt)
        calib_name = os.path.splitext(base)[0]

        rows.append(
            {
                "ckpt": base,
                "calibration_name": calib_name,
                "discipline": "OVERALL",
                "n": str(total_n),
                "accuracy": f"{overall_acc:.4f}",
            }
        )
        for disc_name, (acc, n) in per_disc.items():
            rows.append(
                {
                    "ckpt": base,
                    "calibration_name": calib_name,
                    "discipline": disc_name,
                    "n": str(n),
                    "accuracy": f"{acc:.4f}",
                }
            )

    if not rows:
        print("[WARN] No rows parsed; CSV will not be written.")
        return

    fieldnames = ["ckpt", "calibration_name", "discipline", "n", "accuracy"]
    with open(args.outfile, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\n[INFO] Wrote MMMU results CSV to: {args.outfile}")


if __name__ == "__main__":
    main()

