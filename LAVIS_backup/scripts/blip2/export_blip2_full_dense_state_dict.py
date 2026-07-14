#!/usr/bin/env python3
"""Export a full dense BLIP2-T5 state_dict for pruning-mask comparison.

LAVIS BLIP2-T5 loads different parts from different places:
  - BLIP2 bridge/Q-Former checkpoint from --ckpt
  - Flan-T5 weights from the HuggingFace cache/config
  - EVA ViT weights from the model config/cache

The common blip2_pretrained_flant5xl.pth file is therefore not a full dense
state_dict. Use this script to materialize the complete in-memory model into a
single checkpoint that can be compared against pruned full-state checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple


_LAVIS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _LAVIS_ROOT not in sys.path:
    sys.path.insert(0, _LAVIS_ROOT)


GROUP_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("t5_model", ("t5_model.",)),
    ("visual_encoder", ("visual_encoder.",)),
    ("ln_vision", ("ln_vision.",)),
    ("Qformer", ("Qformer.",)),
    ("query_tokens", ("query_tokens",)),
    ("t5_proj", ("t5_proj.",)),
]

REQUIRED_GROUPS = ("t5_model", "visual_encoder", "Qformer", "query_tokens", "t5_proj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export full dense BLIP2-T5 state_dict.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", required=True, help="BLIP2 pretrain checkpoint, e.g. blip2_pretrained_flant5xl.pth")
    parser.add_argument("--out", required=True, help="Output .pth path for the full dense state_dict.")
    parser.add_argument("--model_name", default="blip2_t5")
    parser.add_argument("--model_type", default="pretrain_flant5xl")
    parser.add_argument("--device", default="cpu", help="Use cpu to avoid GPU memory pressure.")
    parser.add_argument("--summary_json", default="", help="Optional JSON summary path. Defaults to <out>.summary.json.")
    parser.add_argument("--summary_csv", default="", help="Optional CSV group summary path. Defaults to <out>.summary.csv.")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def group_for_key(key: str) -> str:
    for group, prefixes in GROUP_RULES:
        if any(key.startswith(prefix) for prefix in prefixes):
            return group
    return "other"


def empty_group() -> Dict[str, object]:
    return {
        "tensors": 0,
        "numel": 0,
        "two_dim_tensors": 0,
        "sample_keys": [],
    }


def summarize_state_dict(state: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    groups = {group: empty_group() for group, _ in GROUP_RULES}
    groups["other"] = empty_group()
    for key, value in state.items():
        group = group_for_key(key)
        row = groups[group]
        row["tensors"] = int(row["tensors"]) + 1
        if hasattr(value, "numel"):
            row["numel"] = int(row["numel"]) + int(value.numel())
        if hasattr(value, "dim") and int(value.dim()) >= 2:
            row["two_dim_tensors"] = int(row["two_dim_tensors"]) + 1
        sample_keys = row["sample_keys"]
        if isinstance(sample_keys, list) and len(sample_keys) < 5:
            sample_keys.append(key)
    return groups


def write_summary_json(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_summary_csv(path: str, groups: Dict[str, Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "tensors", "numel", "two_dim_tensors", "sample_keys"])
        writer.writeheader()
        for group in list(dict(GROUP_RULES).keys()) + ["other"]:
            row = groups[group]
            writer.writerow(
                {
                    "group": group,
                    "tensors": row["tensors"],
                    "numel": row["numel"],
                    "two_dim_tensors": row["two_dim_tensors"],
                    "sample_keys": ";".join(str(x) for x in row["sample_keys"]),
                }
            )


def missing_required_groups(groups: Dict[str, Dict[str, object]]) -> List[str]:
    return [group for group in REQUIRED_GROUPS if int(groups[group]["tensors"]) <= 0]


def main() -> None:
    args = parse_args()
    try:
        import torch
        from lavis.models import load_model
    except ImportError as exc:
        raise SystemExit("Missing LAVIS/PyTorch dependency: %s" % exc) from exc

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print("Loading model:", args.model_name, args.model_type)
    print("Bridge/Q-Former ckpt:", os.path.abspath(args.ckpt))
    print("Device:", args.device)
    model = load_model(
        args.model_name,
        args.model_type,
        is_eval=True,
        device=args.device,
        checkpoint=args.ckpt,
    )
    model.eval()

    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    groups = summarize_state_dict(state)
    missing = missing_required_groups(groups)
    if missing:
        raise SystemExit(
            "[FATAL] exported state_dict is not full BLIP2-T5; missing groups: %s" % ",".join(missing)
        )

    torch.save(state, out)

    summary_json = args.summary_json or out + ".summary.json"
    summary_csv = args.summary_csv or out + ".summary.csv"
    payload: Dict[str, object] = {
        "created_utc": utc_now(),
        "model_name": args.model_name,
        "model_type": args.model_type,
        "device": args.device,
        "bridge_ckpt": os.path.abspath(args.ckpt),
        "out": out,
        "total_tensors": len(state),
        "required_groups": list(REQUIRED_GROUPS),
        "missing_required_groups": missing,
        "groups": groups,
        "note": "The .pth file is a raw full dense model.state_dict() for dense-base-aware pruning-mask inference.",
    }
    write_summary_json(summary_json, payload)
    write_summary_csv(summary_csv, groups)

    print("[OK] wrote:", out)
    print("total tensors:", len(state))
    for group in REQUIRED_GROUPS:
        print("%s tensors: %s" % (group, groups[group]["tensors"]))
    print("[OK] wrote summary JSON:", summary_json)
    print("[OK] wrote summary CSV:", summary_csv)


if __name__ == "__main__":
    main()
