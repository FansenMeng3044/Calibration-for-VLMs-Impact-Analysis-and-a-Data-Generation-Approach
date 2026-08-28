#!/usr/bin/env python3
"""Assert that joint and text-only TAMP protocols use aligned calibration rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cosmos_tamp_prune as tamp


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
        def load(protocol: str):
            options = SimpleNamespace(
                protocol=protocol,
                calibration_preset=[name],
                preset_file=str(preset_path),
                calibration_json=[],
                ar_calibration_json=[],
                verification_json=[],
                image_root=[],
                nsamples=args.nsamples,
                nsamples_per_file=0,
            )
            tamp.apply_calibration_presets(options)
            return tamp.build_protocol_calibration(options)

        joint = load(tamp.PROTOCOL_JOINT)
        separate = load(tamp.PROTOCOL_SEPARATE)
        joint_text = [tamp.normalized_pair_text(record.text) for record in joint.ar_records]
        separate_text = [tamp.normalized_pair_text(record.text) for record in separate.ar_records]
        verification_text = [
            tamp.normalized_pair_text(record.text)
            for record in separate.verification_records
        ]
        joint_images = [
            str(Path(record.image_path).resolve()) for record in joint.ar_records
        ]
        verification_images = [
            str(Path(record.image_path).resolve())
            for record in separate.verification_records
        ]
        text_joint_separate = sum(a == b for a, b in zip(joint_text, separate_text))
        text_separate_verification = sum(
            a == b for a, b in zip(separate_text, verification_text)
        )
        image_joint_verification = sum(
            a == b for a, b in zip(joint_images, verification_images)
        )
        if (text_joint_separate, text_separate_verification, image_joint_verification) != (
            args.nsamples,
            args.nsamples,
            args.nsamples,
        ):
            raise RuntimeError(
                f"{name} calibration mismatch: joint/separate text={text_joint_separate}, "
                f"separate/verification text={text_separate_verification}, "
                f"joint/verification images={image_joint_verification}, "
                f"expected={args.nsamples}"
            )
        source_paths = {
            Path(record.source_path).resolve()
            for record in [
                *joint.ar_records,
                *separate.ar_records,
                *separate.verification_records,
            ]
        }
        report["datasets"][name] = {
            "joint_separate_text_matches": text_joint_separate,
            "separate_verification_text_matches": text_separate_verification,
            "joint_verification_image_path_matches": image_joint_verification,
            "separate_ar_has_no_images": all(
                record.image_path is None for record in separate.ar_records
            ),
            "source_sha256": {
                str(path): sha256(path) for path in sorted(source_paths)
            },
            "first_joint_sample_id": joint.ar_records[0].sample_id,
            "first_image": joint_images[0],
        }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
