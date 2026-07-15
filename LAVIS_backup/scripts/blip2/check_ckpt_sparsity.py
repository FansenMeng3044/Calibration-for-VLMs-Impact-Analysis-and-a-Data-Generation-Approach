#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check BLIP2-T5 checkpoint sparsity by module group.

This is used by the ATV migration validation flow to verify that T5-only ATV
actually prunes T5 while leaving the visual encoder and BLIP2 bridge modules
effectively dense.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch


GROUP_RULES = [
    ("visual_encoder", lambda k: k.startswith("visual_encoder")),
    ("Qformer", lambda k: k.startswith("Qformer") or k.startswith("query_tokens")),
    ("t5_proj", lambda k: k.startswith("t5_proj")),
    ("t5_model.encoder", lambda k: k.startswith("t5_model.encoder")),
    ("t5_model.decoder", lambda k: k.startswith("t5_model.decoder")),
]


def unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "module"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def group_for_key(key: str) -> str:
    for group, fn in GROUP_RULES:
        if fn(key):
            return group
    return "other"


def iter_weight_tensors(state_dict, min_numel: int) -> Iterable[Tuple[str, torch.Tensor]]:
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        if value.dim() < 2 or value.numel() < min_numel:
            continue
        yield key, value


def empty_stat() -> Dict[str, int]:
    return {"zeros": 0, "total": 0, "tensors": 0}


def add_tensor(stat: Dict[str, int], tensor: torch.Tensor) -> None:
    stat["zeros"] += int((tensor == 0).sum().item())
    stat["total"] += int(tensor.numel())
    stat["tensors"] += 1


def sparsity(stat: Dict[str, int]) -> float:
    return float(stat["zeros"]) / float(stat["total"]) if stat["total"] else 0.0


def status_for_group(group: str, sp: float, args: argparse.Namespace) -> Tuple[str, str]:
    if group == "visual_encoder":
        ok = sp <= args.vit_max
        return ("PASS" if ok else "FAIL", "vit_max=%.6f" % args.vit_max)
    if group in {"Qformer", "t5_proj"}:
        ok = sp <= args.non_t5_max
        return ("PASS" if ok else "FAIL", "non_t5_max=%.6f" % args.non_t5_max)
    if group in {"t5_model.encoder", "t5_model.decoder", "t5_model.all"} and args.expect_t5 is not None:
        ok = abs(sp - args.expect_t5) <= args.tol
        return ("PASS" if ok else "FAIL", "expect_t5=%.6f,tol=%.6f" % (args.expect_t5, args.tol))
    return ("INFO", "")


def build_rows(stats: Dict[str, Dict[str, int]], args: argparse.Namespace) -> List[Dict[str, object]]:
    t5_all = empty_stat()
    for group in ("t5_model.encoder", "t5_model.decoder"):
        t5_all["zeros"] += stats[group]["zeros"]
        t5_all["total"] += stats[group]["total"]
        t5_all["tensors"] += stats[group]["tensors"]

    rows: List[Dict[str, object]] = []
    for group in [name for name, _ in GROUP_RULES] + ["t5_model.all", "other"]:
        stat = t5_all if group == "t5_model.all" else stats[group]
        if stat["total"] == 0:
            continue
        sp = sparsity(stat)
        status, note = status_for_group(group, sp, args)
        rows.append(
            {
                "tag": args.tag,
                "group": group,
                "zeros": stat["zeros"],
                "total": stat["total"],
                "tensors": stat["tensors"],
                "sparsity": "%.8f" % sp,
                "status": status,
                "note": note,
                "ckpt": str(args.ckpt),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["tag", "group", "zeros", "total", "tensors", "sparsity", "status", "note", "ckpt"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check sparsity of BLIP2-T5 checkpoint groups.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--tag", default="checkpoint")
    parser.add_argument("--expect_t5", type=float, default=None)
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--vit_max", type=float, default=0.01)
    parser.add_argument(
        "--non_t5_max",
        type=float,
        default=0.01,
        help="Maximum allowed sparsity for Q-Former/query-token and t5_proj bridge weights.",
    )
    parser.add_argument("--min_numel", type=int, default=4096)
    parser.add_argument("--out_csv", type=Path, default=None)
    parser.add_argument("--no_fail", action="store_true", help="Print warnings but exit zero.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    obj = torch.load(args.ckpt, map_location="cpu")
    state_dict = unwrap_state_dict(obj)
    if not isinstance(state_dict, dict):
        raise SystemExit("[FATAL] Could not parse state_dict from %s" % args.ckpt)

    stats = {group: empty_stat() for group, _ in GROUP_RULES}
    stats["other"] = empty_stat()

    matched = 0
    for key, tensor in iter_weight_tensors(state_dict, args.min_numel):
        add_tensor(stats[group_for_key(key)], tensor)
        matched += 1
    if matched == 0:
        raise SystemExit("[FATAL] No eligible 2D weight tensors found in %s" % args.ckpt)

    rows = build_rows(stats, args)
    print("\n===== sparsity: %s =====" % args.ckpt)
    for row in rows:
        print(
            "  {group:18s}: {sp:7.3f}%  zeros {zeros:,} / {total:,}  tensors={tensors}  {status}".format(
                group=str(row["group"]),
                sp=100.0 * float(row["sparsity"]),
                zeros=int(row["zeros"]),
                total=int(row["total"]),
                tensors=int(row["tensors"]),
                status=str(row["status"]),
            )
        )

    if args.out_csv is not None:
        write_csv(args.out_csv, rows)
        print("[OK] wrote sparsity CSV: %s" % args.out_csv)

    hard_fail = any(row["status"] == "FAIL" for row in rows)
    print("[RESULT]", "PASS" if not hard_fail else "CHECK")
    if hard_fail and not args.no_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
