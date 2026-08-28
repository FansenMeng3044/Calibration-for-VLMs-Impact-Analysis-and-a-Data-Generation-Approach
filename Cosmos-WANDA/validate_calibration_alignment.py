#!/usr/bin/env python3
"""Assert that joint and separate protocols use identical calibration pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cosmos_wanda_prune as wanda


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset-file", default=str(Path(__file__).with_name("calibration_presets.json")))
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    preset_path = Path(args.preset_file).resolve()
    payload = json.loads(preset_path.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "preset_file": str(preset_path),
        "preset_sha256": sha256(preset_path),
        "nsamples": args.nsamples,
        "datasets": {},
    }
    for name in ("mmbench", "mmmu", "okvqa"):
        dataset = payload["datasets"][name]
        joint_cfg = dataset["joint"]
        separate_cfg = dataset["separate"]
        joint = wanda.build_calibration_records(
            joint_cfg["calibration_json"],
            joint_cfg.get("image_roots", []),
            args.nsamples,
            0,
            True,
            f"{name}_joint_alignment",
        )
        vision = wanda.build_calibration_records(
            separate_cfg["vision_calibration_json"],
            separate_cfg.get("image_roots", []),
            args.nsamples,
            0,
            True,
            f"{name}_separate_vision_alignment",
        )
        ar = wanda.build_calibration_records(
            separate_cfg["ar_calibration_json"],
            separate_cfg.get("image_roots", []),
            args.nsamples,
            0,
            False,
            f"{name}_separate_ar_alignment",
        )
        joint_text = [wanda.normalized_pair_text(record.text) for record in joint]
        vision_text = [wanda.normalized_pair_text(record.text) for record in vision]
        ar_text = [wanda.normalized_pair_text(record.text) for record in ar]
        joint_images = [str(Path(record.image_path).resolve()) for record in joint]
        vision_images = [str(Path(record.image_path).resolve()) for record in vision]
        text_joint_separate = sum(a == b for a, b in zip(joint_text, vision_text))
        text_vision_ar = sum(a == b for a, b in zip(vision_text, ar_text))
        image_joint_separate = sum(a == b for a, b in zip(joint_images, vision_images))
        if (text_joint_separate, text_vision_ar, image_joint_separate) != (
            args.nsamples,
            args.nsamples,
            args.nsamples,
        ):
            raise RuntimeError(
                f"{name} calibration mismatch: joint/separate text={text_joint_separate}, "
                f"vision/AR text={text_vision_ar}, images={image_joint_separate}, "
                f"expected={args.nsamples}"
            )
        source_paths = {
            Path(record.source_path).resolve()
            for record in [*joint, *vision, *ar]
        }
        report["datasets"][name] = {
            "joint_separate_text_matches": text_joint_separate,
            "separate_vision_ar_text_matches": text_vision_ar,
            "joint_separate_image_path_matches": image_joint_separate,
            "source_sha256": {
                str(path): sha256(path) for path in sorted(source_paths)
            },
            "first_joint_sample_id": joint[0].sample_id,
            "first_image": joint_images[0],
        }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

