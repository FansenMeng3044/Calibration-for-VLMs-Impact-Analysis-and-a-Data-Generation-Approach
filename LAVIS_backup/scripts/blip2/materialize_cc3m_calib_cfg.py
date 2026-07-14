#!/usr/bin/env python3
"""Materialize a CC3M calibration config with explicit data paths."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional


DATASET_NAME = "prefix_conceptual_caption_3m"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Write a runtime CC3M calibration YAML with explicit annotation/image paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--src_cfg", required=True, type=Path)
    p.add_argument("--out_cfg", required=True, type=Path)
    p.add_argument("--cc3m_json", required=True, type=Path)
    p.add_argument("--cc3m_images_dir", required=True, type=Path)
    p.add_argument("--pretrained", type=Path, default=None)
    return p.parse_args()


def quote_yaml(value: Path) -> str:
    return json.dumps(str(value))


def skip_nested_block(lines: list[str], start: int, indent: int) -> int:
    j = start + 1
    while j < len(lines):
        stripped = lines[j].strip()
        line_indent = len(lines[j]) - len(lines[j].lstrip(" "))
        if stripped and line_indent <= indent:
            break
        j += 1
    return j


def save_with_text_fallback(
    src_cfg: Path,
    out_cfg: Path,
    cc3m_json: Path,
    cc3m_images_dir: Path,
    pretrained: Optional[Path],
) -> None:
    lines = src_cfg.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    stack: list[tuple[int, str]] = []
    i = 0
    replaced = {
        "url": False,
        "annotation_storage": False,
        "image_storage": False,
        "pretrained": pretrained is None,
    }

    while i < len(lines):
        line = lines[i]
        match = re.match(r"^(\s*)([A-Za-z0-9_]+):\s*(.*)$", line)
        if not match:
            out.append(line)
            i += 1
            continue

        indent = len(match.group(1))
        key = match.group(2)
        value = match.group(3).strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = [x[1] for x in stack] + [key]

        if path == ["model", "pretrained"] and pretrained is not None:
            out.append("%spretrained: %s" % (match.group(1), quote_yaml(pretrained)))
            replaced["pretrained"] = True
            i += 1
            continue

        if key == "url" and "annotations" in path and "train" in path and value == "":
            out.append(line)
            out.append("%s  - %s" % (match.group(1), quote_yaml(cc3m_json)))
            replaced["url"] = True
            i = skip_nested_block(lines, i, indent)
            continue

        if key == "storage" and "annotations" in path and "train" in path and value == "":
            out.append(line)
            out.append("%s  - %s" % (match.group(1), quote_yaml(cc3m_json)))
            replaced["annotation_storage"] = True
            i = skip_nested_block(lines, i, indent)
            continue

        if path[-3:] == ["build_info", "images", "storage"]:
            out.append("%sstorage: %s" % (match.group(1), quote_yaml(cc3m_images_dir)))
            replaced["image_storage"] = True
            i += 1
            continue

        out.append(line)
        if value == "":
            stack.append((indent, key))
        i += 1

    missing = [name for name, ok in replaced.items() if not ok]
    if missing:
        raise RuntimeError("text fallback could not replace fields: %s" % ",".join(missing))
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    out_cfg.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.src_cfg.is_file():
        raise FileNotFoundError("source config not found: %s" % args.src_cfg)
    if not args.cc3m_json.is_file():
        raise FileNotFoundError("CC3M JSON not found: %s" % args.cc3m_json)
    if not args.cc3m_images_dir.is_dir():
        raise FileNotFoundError("CC3M images dir not found: %s" % args.cc3m_images_dir)

    try:
        from omegaconf import OmegaConf  # type: ignore

        cfg = OmegaConf.load(str(args.src_cfg))
        dataset = cfg.datasets[DATASET_NAME]
        dataset.build_info.annotations.train.url = [str(args.cc3m_json)]
        dataset.build_info.annotations.train.storage = [str(args.cc3m_json)]
        dataset.build_info.images.storage = str(args.cc3m_images_dir)
        if args.pretrained is not None:
            cfg.model.pretrained = str(args.pretrained)
        args.out_cfg.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config=cfg, f=str(args.out_cfg))
    except ModuleNotFoundError:
        save_with_text_fallback(
            args.src_cfg,
            args.out_cfg,
            args.cc3m_json,
            args.cc3m_images_dir,
            args.pretrained,
        )
    print("[OK] wrote runtime CC3M cfg: %s" % args.out_cfg)
    print("[OK] annotation JSON: %s" % args.cc3m_json)
    print("[OK] images dir: %s" % args.cc3m_images_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
