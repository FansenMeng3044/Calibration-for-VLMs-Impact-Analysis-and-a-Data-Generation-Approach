#!/usr/bin/env python3
"""Layer-wise TAMP pruning for the Cosmos3-Edge Reasoner.

This implementation intentionally supports exactly two pruning protocols:

* joint: a real image+text fused sequence supplies DAS and AMIA statistics;
  only the AR language tower is pruned.
* separate: a genuinely text-only tokenizer -> embedding -> AR forward supplies
  the strict language-only DAS reduction and AMIA statistics. No
  vision/projector output is present in calibration and the vision tower is
  not pruned.

The vision encoder and projector stay dense in both protocols. The
diffusion/generator tower is never instantiated by this script.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers import Cosmos3EdgeForConditionalGeneration


PROTOCOL_JOINT = "joint"
PROTOCOL_SEPARATE = "separate"
PROTOCOL_INSPECT = "inspect"

VISION_PREFIX = "model.visual.encoder.layers"
AR_PREFIX = "model.language_model.layers"
PROJECTOR_PREFIX = "model.projector"

EXPECTED_VISION_LAYERS = 27
EXPECTED_AR_LAYERS = 28
EXPECTED_VISION_LINEARS = 162
EXPECTED_AR_LINEARS = 168
EXPECTED_VISION_LINEAR_PARAMETERS = 411_070_464
EXPECTED_AR_LINEAR_PARAMETERS = 1_409_286_144

GENERATOR_NAME_FRAGMENTS = (
    "action_modality_embed",
    "action_proj_in",
    "action_proj_out",
    "time_embedder",
    "norm_moe_gen",
    "input_layernorm_moe_gen",
    "post_attention_layernorm_moe_gen",
    "mlp_moe_gen",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "norm_added_q",
    "norm_added_k",
    "k_norm_und_for_gen",
)

TEXT_FIELDS = ("text", "text_input", "question", "caption", "prompt")
IMAGE_FIELDS = ("image", "image_path", "image_file", "images")
VIDEO_FIELDS = ("video", "video_path", "video_file", "videos")


class ProtocolError(RuntimeError):
    """Raised when a calibration path violates the locked experiment protocol."""


class _CaptureStop(RuntimeError):
    """Private control-flow exception used to stop after the first target layer."""


@dataclasses.dataclass(frozen=True)
class CalibrationRecord:
    sample_id: str
    source_path: str
    source_index: int
    dataset: str
    text: str | None
    image_path: str | None


@dataclasses.dataclass(frozen=True)
class ProtocolCalibration:
    ar_records: list[CalibrationRecord]
    verification_records: list[CalibrationRecord]
    pairing: dict[str, Any]


@dataclasses.dataclass
class LayerSample:
    hidden_states: torch.Tensor
    layer_kwargs: dict[str, Any]
    valid_mask: torch.Tensor | None
    visual_mask: torch.Tensor
    sample_id: str


@dataclasses.dataclass
class ActivationStats:
    columns: int
    sum_sq: torch.Tensor = dataclasses.field(init=False)
    nsamples: int = 0
    calls: int = 0

    def __post_init__(self) -> None:
        self.sum_sq = torch.zeros(self.columns, dtype=torch.float32, device="cpu")

    def add(self, inputs: torch.Tensor, valid_mask: torch.Tensor | None) -> None:
        if inputs.ndim == 2:
            batch_count = 1
            flattened = inputs
        elif inputs.ndim == 3:
            batch_count = int(inputs.shape[0])
            if valid_mask is not None:
                mask = valid_mask.to(device=inputs.device, dtype=torch.bool)
                if tuple(mask.shape) != tuple(inputs.shape[:-1]):
                    raise ProtocolError(
                        f"Activation/mask mismatch: activation={tuple(inputs.shape)}, "
                        f"mask={tuple(mask.shape)}"
                    )
                flattened = inputs[mask]
            else:
                flattened = inputs.reshape(-1, inputs.shape[-1])
        else:
            raise ProtocolError(f"Unsupported Linear input rank {inputs.ndim}: {tuple(inputs.shape)}")

        if flattened.shape[-1] != self.columns:
            raise ProtocolError(
                f"Linear input width changed: expected {self.columns}, got {flattened.shape[-1]}"
            )
        if flattened.numel() == 0:
            raise ProtocolError("No valid activation tokens reached a target Linear")

        contribution = flattened.detach().float().pow(2).sum(dim=0).cpu()
        self.sum_sq.add_(contribution)
        self.nsamples += batch_count
        self.calls += 1

    @property
    def scaler_row(self) -> torch.Tensor:
        if self.nsamples <= 0:
            raise ProtocolError("TAMP statistic was requested before any activation samples were collected")
        return self.sum_sq / float(self.nsamples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=(PROTOCOL_INSPECT, PROTOCOL_JOINT, PROTOCOL_SEPARATE), required=True)
    parser.add_argument("--model-path", default="/private/workspace/hycui/model/Cosmos3-Edge")
    parser.add_argument("--calibration-json", action="append", default=[])
    parser.add_argument("--ar-calibration-json", action="append", default=[])
    parser.add_argument("--verification-json", action="append", default=[])
    parser.add_argument(
        "--calibration-preset",
        action="append",
        choices=("mmbench", "mmmu", "okvqa"),
        default=[],
        help="Load named joint/separate paths from calibration_presets.json; repeatable.",
    )
    parser.add_argument(
        "--preset-file",
        default=str(Path(__file__).resolve().with_name("calibration_presets.json")),
    )
    parser.add_argument("--image-root", action="append", default=[])
    parser.add_argument("--output-dir")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--nsamples-per-file", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--min-image-pixels", type=int, default=65_536)
    parser.add_argument("--max-image-pixels", type=int, default=1_048_576)
    parser.add_argument(
        "--vision-sparsity",
        type=float,
        default=0.0,
        help="Compatibility guard; TAMP is AR/LLM-only and requires this to remain exactly zero.",
    )
    parser.add_argument("--ar-sparsity", type=float, default=0.5)
    parser.add_argument("--max-sparsity-per-linear", type=float, default=0.6)
    parser.add_argument("--max-ar-layers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa", "flash_attention_2"), default="eager")
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--allow-partial-save", action="store_true")
    parser.add_argument("--skip-final-forward", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def apply_calibration_presets(args: argparse.Namespace) -> None:
    if not args.calibration_preset:
        return
    if args.calibration_json or args.ar_calibration_json or args.verification_json:
        raise ValueError("Do not mix --calibration-preset with explicit calibration JSON arguments")
    preset_path = Path(args.preset_file).expanduser().resolve()
    with preset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError(f"Invalid preset file (missing datasets object): {preset_path}")
    for preset_name in args.calibration_preset:
        dataset_entry = datasets.get(preset_name)
        if not isinstance(dataset_entry, dict):
            raise ValueError(f"Preset {preset_name!r} is absent from {preset_path}")
        protocol_entry = dataset_entry.get(args.protocol)
        if not isinstance(protocol_entry, dict):
            raise ValueError(f"Preset {preset_name!r} has no {args.protocol!r} entry")
        args.image_root.extend(str(value) for value in protocol_entry.get("image_roots", []))
        if args.protocol == PROTOCOL_JOINT:
            args.calibration_json.extend(str(value) for value in protocol_entry.get("calibration_json", []))
        elif args.protocol == PROTOCOL_SEPARATE:
            args.ar_calibration_json.extend(
                str(value) for value in protocol_entry.get("ar_calibration_json", [])
            )
            args.verification_json.extend(
                str(value) for value in protocol_entry.get("verification_json", [])
            )


def validate_args(args: argparse.Namespace) -> None:
    if float(args.vision_sparsity) != 0.0:
        raise ValueError(
            "TAMP is AR/LLM-only; --vision-sparsity must remain exactly 0. "
            "The dense vision encoder is part of the Reasoner but is never pruned."
        )
    if not 0.0 < float(args.ar_sparsity) < 1.0:
        raise ValueError(f"--ar-sparsity must be strictly between 0 and 1, got {args.ar_sparsity}")
    if not float(args.ar_sparsity) <= float(args.max_sparsity_per_linear) < 1.0:
        raise ValueError(
            "--max-sparsity-per-linear must be >= --ar-sparsity and < 1, got "
            f"{args.max_sparsity_per_linear} for target {args.ar_sparsity}"
        )
    if args.nsamples <= 0:
        raise ValueError("--nsamples must be positive")
    if args.nsamples_per_file < 0:
        raise ValueError("--nsamples-per-file cannot be negative")
    if args.min_image_pixels <= 0 or args.max_image_pixels <= 0:
        raise ValueError("Image pixel bounds must be positive")
    if args.min_image_pixels > args.max_image_pixels:
        raise ValueError("--min-image-pixels cannot exceed --max-image-pixels")
    if args.protocol == PROTOCOL_JOINT:
        if args.ar_calibration_json or args.verification_json:
            raise ValueError("Joint protocol uses --calibration-json only")
        if not args.calibration_json:
            raise ValueError("Joint protocol requires at least one --calibration-json or preset")
    elif args.protocol == PROTOCOL_SEPARATE:
        if args.calibration_json:
            raise ValueError(
                "Separate TAMP calibration is text-only; use --ar-calibration-json, "
                "never a paired multimodal --calibration-json"
            )
        if not args.ar_calibration_json:
            raise ValueError("Separate protocol requires at least one text-only --ar-calibration-json")
        if not args.verification_json:
            raise ValueError(
                "Separate protocol requires --verification-json for the final multimodal forward; "
                "verification records are never used for pruning statistics"
            )
    if args.protocol != PROTOCOL_INSPECT:
        if not args.preflight_only and not args.output_dir:
            raise ValueError("--output-dir is required for pruning")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")


def set_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        for key in ("data", "records", "samples"):
            if isinstance(payload.get(key), list):
                records = payload[key]
                break
        else:
            raise ValueError(f"No list-like data/records/samples field in {path}")
    else:
        raise ValueError(f"Expected a JSON list or object in {path}, got {type(payload).__name__}")
    if not all(isinstance(item, dict) for item in records):
        raise ValueError(f"Every calibration item must be an object: {path}")
    return records


def first_text(record: dict[str, Any]) -> str:
    for field in TEXT_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    conversations = record.get("conversations")
    if isinstance(conversations, list):
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from", turn.get("role", ""))).lower()
            value = turn.get("value", turn.get("content"))
            if role in {"human", "user"} and isinstance(value, str) and value.strip():
                return value.replace("<image>", "").strip()
    raise ValueError(f"No usable text field among {TEXT_FIELDS}")


def first_image_value(record: dict[str, Any]) -> str:
    for field in IMAGE_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value and isinstance(value[0], str):
            return value[0].strip()
    raise ValueError(f"No usable image field among {IMAGE_FIELDS}")


def assert_text_only_record(record: dict[str, Any], context: str) -> None:
    for field in (*IMAGE_FIELDS, *VIDEO_FIELDS):
        value = record.get(field)
        if value not in (None, "", [], {}):
            raise ProtocolError(f"{context}: text-only TAMP record contains visual field {field!r}")
    rendered = json.dumps(record, ensure_ascii=False)
    for placeholder in ("<image>", "<video>"):
        if placeholder in rendered:
            raise ProtocolError(f"{context}: text-only TAMP record contains {placeholder} placeholder")


def resolve_image_path(
    raw_image: str,
    dataset: str,
    source_path: Path,
    image_roots: list[Path],
) -> Path:
    raw_path = Path(raw_image)
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            (
                source_path.parent / raw_path,
                source_path.parent / dataset / raw_path,
                source_path.parent / dataset / "images" / raw_path,
            )
        )
        for root in image_roots:
            candidates.extend((root / raw_path, root / dataset / raw_path, root / dataset / "images" / raw_path))

    seen: set[str] = set()
    unique_candidates = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)
            if candidate.is_file():
                return candidate.resolve()
    rendered = "\n  - ".join(str(path) for path in unique_candidates)
    raise FileNotFoundError(f"Could not resolve image {raw_image!r}. Tried:\n  - {rendered}")


def infer_sample_id(record: dict[str, Any], source: Path, index: int) -> str:
    for field in ("sample_id", "question_id", "id", "uid", "index"):
        value = record.get(field)
        if value is not None and str(value).strip():
            return f"{source.stem}:{value}"
    digest = hashlib.sha1(
        json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    return f"{source.stem}:{index}:{digest}"


def build_calibration_records(
    paths: list[str],
    image_roots: list[str],
    nsamples: int,
    nsamples_per_file: int,
    require_image: bool,
    branch_name: str,
    forbid_visual: bool = False,
) -> list[CalibrationRecord]:
    roots = [Path(value).expanduser().resolve() for value in image_roots]
    selected: list[CalibrationRecord] = []
    for path_string in paths:
        path = Path(path_string).expanduser().resolve()
        raw_records = load_json_records(path)
        if nsamples_per_file:
            raw_records = raw_records[:nsamples_per_file]
        for index, raw in enumerate(raw_records):
            try:
                if forbid_visual:
                    assert_text_only_record(raw, f"{branch_name}: {path}[{index}]")
                text = first_text(raw)
                dataset = str(raw.get("dataset", path.stem.split("_")[0])).strip().lower()
                if require_image:
                    raw_image = first_image_value(raw)
                    image_path: Path | None = resolve_image_path(raw_image, dataset, path, roots)
                else:
                    image_path = None
            except Exception as exc:
                raise type(exc)(f"{branch_name}: {path}[{index}]: {exc}") from exc
            if "<image>" in text:
                raise ProtocolError(f"{branch_name}: {path}[{index}] text still contains an <image> placeholder")
            selected.append(
                CalibrationRecord(
                    sample_id=infer_sample_id(raw, path, index),
                    source_path=str(path),
                    source_index=index,
                    dataset=dataset,
                    text=text,
                    image_path=str(image_path) if image_path is not None else None,
                )
            )
    if len(selected) < nsamples:
        raise ValueError(
            f"{branch_name}: requested {nsamples} samples but only {len(selected)} valid records were loaded"
        )
    return selected[:nsamples]


def normalized_pair_text(text: str | None) -> str:
    if text is None:
        return ""
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").strip().splitlines())


def build_protocol_calibration(args: argparse.Namespace) -> ProtocolCalibration:
    common_kwargs = {
        "image_roots": args.image_root,
        "nsamples": args.nsamples,
        "nsamples_per_file": args.nsamples_per_file,
    }
    if args.protocol == PROTOCOL_JOINT:
        records = build_calibration_records(
            paths=args.calibration_json,
            require_image=True,
            branch_name="joint",
            **common_kwargs,
        )
        return ProtocolCalibration(
            ar_records=records,
            verification_records=records,
            pairing={"mode": "paired_multimodal", "pairs": len(records), "text_mismatches": 0},
        )

    if args.protocol != PROTOCOL_SEPARATE:
        raise ValueError(f"Calibration records are not used by protocol {args.protocol}")

    ar_records = build_calibration_records(
        paths=args.ar_calibration_json,
        require_image=False,
        branch_name="separate_ar",
        forbid_visual=True,
        **common_kwargs,
    )
    verification_records = build_calibration_records(
        paths=args.verification_json,
        require_image=True,
        branch_name="separate_verification",
        **common_kwargs,
    )
    if len(verification_records) != len(ar_records):
        raise ProtocolError(
            "Separate text/verification length mismatch: "
            f"text={len(ar_records)}, verification={len(verification_records)}"
        )
    paired_ar: list[CalibrationRecord] = []
    paired_verification: list[CalibrationRecord] = []
    mismatches: list[dict[str, Any]] = []
    for index, (verification_record, ar_record) in enumerate(
        zip(verification_records, ar_records)
    ):
        verification_text = normalized_pair_text(verification_record.text)
        ar_text = normalized_pair_text(ar_record.text)
        if verification_text != ar_text:
            mismatches.append(
                {
                    "index": index,
                    "verification_source": verification_record.source_path,
                    "ar_source": ar_record.source_path,
                    "verification_sample_id": verification_record.sample_id,
                    "ar_sample_id": ar_record.sample_id,
                }
            )
            continue
        pair_digest = hashlib.sha1(ar_text.encode("utf-8")).hexdigest()[:12]
        pair_id = f"{verification_record.dataset}:tamp_text_pair:{index}:{pair_digest}"
        paired_ar.append(dataclasses.replace(ar_record, sample_id=pair_id))
        paired_verification.append(dataclasses.replace(verification_record, sample_id=pair_id))
    if mismatches:
        raise ProtocolError(
            f"Separate text/verification files are not aligned; "
            f"{len(mismatches)} text mismatches, first={mismatches[:3]}"
        )
    return ProtocolCalibration(
        ar_records=paired_ar,
        verification_records=paired_verification,
        pairing={
            "mode": "text_only_ar_plus_multimodal_verification",
            "pairs": len(paired_ar),
            "text_mismatches": 0,
            "ar_json": [str(Path(path).resolve()) for path in args.ar_calibration_json],
            "verification_json": [str(Path(path).resolve()) for path in args.verification_json],
            "verification_used_for_importance": False,
        },
    )


def recursive_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(recursive_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [recursive_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: recursive_to_cpu(item) for key, item in value.items()}
    return value


def recursive_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(recursive_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [recursive_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: recursive_to_device(item, device) for key, item in value.items()}
    return value


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: recursive_to_device(value, device) for key, value in batch.items()}


def find_linear_modules(module: nn.Module) -> dict[str, nn.Linear]:
    return {name: child for name, child in module.named_modules() if name and isinstance(child, nn.Linear)}


def count_parameters(parameters: Iterable[nn.Parameter]) -> int:
    seen: set[int] = set()
    count = 0
    for parameter in parameters:
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        count += parameter.numel()
    return count


def module_audit(model: Cosmos3EdgeForConditionalGeneration) -> dict[str, Any]:
    if not hasattr(model, "model"):
        raise ProtocolError("Cosmos Reasoner wrapper has no .model")
    reasoner = model.model
    for attr in ("visual", "projector", "language_model"):
        if not hasattr(reasoner, attr):
            raise ProtocolError(f"Cosmos Reasoner is missing expected module model.{attr}")

    vision_layers = reasoner.visual.encoder.layers
    ar_layers = reasoner.language_model.layers
    if len(vision_layers) != int(model.config.vision_config.num_hidden_layers):
        raise ProtocolError("Vision layer count does not match config")
    if len(ar_layers) != int(model.config.text_config.num_hidden_layers):
        raise ProtocolError("AR layer count does not match config")
    if len(vision_layers) != EXPECTED_VISION_LAYERS or len(ar_layers) != EXPECTED_AR_LAYERS:
        raise ProtocolError(
            "Cosmos3-Edge architecture fingerprint changed: "
            f"vision/ar layers={len(vision_layers)}/{len(ar_layers)}, expected "
            f"{EXPECTED_VISION_LAYERS}/{EXPECTED_AR_LAYERS}"
        )

    module_names = [name for name, _ in model.named_modules()]
    generator_names = [
        name for name in module_names if any(fragment in name for fragment in GENERATOR_NAME_FRAGMENTS)
    ]
    if generator_names:
        raise ProtocolError(f"Generator modules were instantiated in the Reasoner: {generator_names[:20]}")

    vision_linears = {
        f"{VISION_PREFIX}.{index}.{name}": child
        for index, layer in enumerate(vision_layers)
        for name, child in find_linear_modules(layer).items()
    }
    ar_linears = {
        f"{AR_PREFIX}.{index}.{name}": child
        for index, layer in enumerate(ar_layers)
        for name, child in find_linear_modules(layer).items()
    }
    overlap = set(vision_linears) & set(ar_linears)
    if overlap:
        raise ProtocolError(f"Vision/AR target-name overlap: {sorted(overlap)[:20]}")
    if not vision_linears or not ar_linears:
        raise ProtocolError("Empty vision or AR Linear allow-list")
    if len(vision_linears) != EXPECTED_VISION_LINEARS or len(ar_linears) != EXPECTED_AR_LINEARS:
        raise ProtocolError(
            "Cosmos3-Edge Linear allow-list fingerprint changed: "
            f"vision/ar={len(vision_linears)}/{len(ar_linears)}, expected "
            f"{EXPECTED_VISION_LINEARS}/{EXPECTED_AR_LINEARS}"
        )

    named_parameters = list(model.named_parameters())
    reasoner_parameter_count = count_parameters(parameter for _, parameter in named_parameters)
    vision_parameter_count = count_parameters(child.weight for child in vision_linears.values())
    ar_parameter_count = count_parameters(child.weight for child in ar_linears.values())
    projector_parameter_count = count_parameters(reasoner.projector.parameters())
    lm_head_parameter_count = count_parameters(model.lm_head.parameters())
    if vision_parameter_count != EXPECTED_VISION_LINEAR_PARAMETERS:
        raise ProtocolError(
            f"Vision Linear parameter fingerprint changed: {vision_parameter_count} != "
            f"{EXPECTED_VISION_LINEAR_PARAMETERS}"
        )
    if ar_parameter_count != EXPECTED_AR_LINEAR_PARAMETERS:
        raise ProtocolError(
            f"AR Linear parameter fingerprint changed: {ar_parameter_count} != "
            f"{EXPECTED_AR_LINEAR_PARAMETERS}"
        )

    return {
        "model_class": type(model).__name__,
        "vision_layer_count": len(vision_layers),
        "ar_layer_count": len(ar_layers),
        "vision_linear_count_dense": len(vision_linears),
        "vision_target_linear_count": 0,
        "ar_linear_count": len(ar_linears),
        "ar_target_linear_count": len(ar_linears),
        "vision_linear_names": sorted(vision_linears),
        "ar_linear_names": sorted(ar_linears),
        "projector_linear_names": sorted(
            f"{PROJECTOR_PREFIX}.{name}" for name in find_linear_modules(reasoner.projector)
        ),
        "generator_modules": generator_names,
        "reasoner_parameter_count": reasoner_parameter_count,
        "vision_linear_parameter_count_dense": vision_parameter_count,
        "vision_target_parameter_count": 0,
        "ar_target_parameter_count": ar_parameter_count,
        "all_target_parameter_count": ar_parameter_count,
        "projector_parameter_count_dense": projector_parameter_count,
        "lm_head_parameter_count_dense": lm_head_parameter_count,
    }


def dtype_from_name(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_reasoner(args: argparse.Namespace) -> tuple[Cosmos3EdgeForConditionalGeneration, Any]:
    from transformers import AutoProcessor, Cosmos3EdgeForConditionalGeneration

    dtype = dtype_from_name(args.dtype)
    model = Cosmos3EdgeForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    )
    model.to(torch.device(args.device))
    model.eval()
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.model_path)
    return model, processor


def image_size_kwargs(min_image_pixels: int, max_image_pixels: int) -> dict[str, dict[str, int]]:
    return {
        "size": {
            "shortest_edge": int(min_image_pixels),
            "longest_edge": int(max_image_pixels),
        }
    }


def apply_chat_template(
    processor: Any,
    record: CalibrationRecord,
    multimodal: bool,
    max_length: int,
    enable_thinking: bool,
    min_image_pixels: int,
    max_image_pixels: int,
) -> dict[str, torch.Tensor]:
    if record.text is None:
        raise ProtocolError(f"AR calibration record has no text: {record.sample_id}")
    content: list[dict[str, str]] = []
    if multimodal:
        if record.image_path is None:
            raise ProtocolError(f"Multimodal record has no image: {record.sample_id}")
        content.append({"type": "image", "image": record.image_path})
    content.append({"type": "text", "text": record.text})
    messages = [{"role": "user", "content": content}]
    batch = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "images_kwargs": image_size_kwargs(min_image_pixels, max_image_pixels),
            "text_kwargs": {"truncation": True, "max_length": max_length},
        },
        enable_thinking=enable_thinking,
    )
    return dict(batch)


def token_counts(batch: dict[str, torch.Tensor]) -> dict[str, int]:
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(batch["input_ids"], dtype=torch.long)
    valid = attention_mask.bool()
    token_types = batch.get("mm_token_type_ids")
    if token_types is None:
        token_types = torch.zeros_like(batch["input_ids"], dtype=torch.long)
    return {
        "valid": int(valid.sum().item()),
        "language": int(((token_types == 0) & valid).sum().item()),
        "image": int(((token_types == 1) & valid).sum().item()),
        "video": int(((token_types == 2) & valid).sum().item()),
    }


@contextlib.contextmanager
def forbidden_forward(modules: dict[str, nn.Module]):
    handles = []

    def make_hook(name: str):
        def hook(_module, _args, _kwargs):
            raise ProtocolError(f"Forbidden module executed in isolated calibration path: {name}")

        return hook

    try:
        for name, module in modules.items():
            handles.append(module.register_forward_pre_hook(make_hook(name), with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()


class CallCounter:
    def __init__(self, modules: dict[str, nn.Module]):
        self.counts = {name: 0 for name in modules}
        self.handles = [
            module.register_forward_pre_hook(self._make_hook(name), with_kwargs=True)
            for name, module in modules.items()
        ]

    def _make_hook(self, name: str):
        def hook(_module, _args, _kwargs):
            self.counts[name] += 1

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def capture_first_layer_inputs(
    layer: nn.Module,
    run_sample,
    records: list[CalibrationRecord],
) -> list[LayerSample]:
    captured: list[LayerSample] = []
    state: dict[str, Any] = {}

    def catcher(_module, args, kwargs):
        hidden_states = args[0] if args else kwargs["hidden_states"]
        captured.append(
            LayerSample(
                hidden_states=recursive_to_cpu(hidden_states),
                layer_kwargs=recursive_to_cpu(dict(kwargs)),
                valid_mask=recursive_to_cpu(state.get("valid_mask")),
                visual_mask=recursive_to_cpu(state["visual_mask"]),
                sample_id=str(state["sample_id"]),
            )
        )
        raise _CaptureStop()

    handle = layer.register_forward_pre_hook(catcher, with_kwargs=True)
    try:
        for record in records:
            state["sample_id"] = record.sample_id
            state["valid_mask"] = None
            state["visual_mask"] = None
            try:
                run_sample(record, state)
            except _CaptureStop:
                pass
            else:
                raise ProtocolError(f"First-layer catcher did not fire for {record.sample_id}")
    finally:
        handle.remove()
    if len(captured) != len(records):
        raise ProtocolError(f"Captured {len(captured)} layer inputs for {len(records)} records")
    for sample in captured:
        if sample.visual_mask is None:
            raise ProtocolError(f"No modality mask was captured for {sample.sample_id}")
        expected = tuple(sample.hidden_states.shape[:-1])
        if tuple(sample.visual_mask.shape) != expected:
            raise ProtocolError(
                f"Hidden/modality-mask mismatch for {sample.sample_id}: "
                f"hidden={tuple(sample.hidden_states.shape)}, visual={tuple(sample.visual_mask.shape)}"
            )
    return captured


def build_ar_cache(
    model: Cosmos3EdgeForConditionalGeneration,
    processor: Any,
    records: list[CalibrationRecord],
    protocol: str,
    device: torch.device,
    max_length: int,
    enable_thinking: bool,
    min_image_pixels: int,
    max_image_pixels: int,
) -> tuple[list[LayerSample], dict[str, Any]]:
    reasoner = model.model
    first_layer = reasoner.language_model.layers[0]
    counters = CallCounter(
        {
            "vision": reasoner.visual,
            "projector": reasoner.projector,
            "ar": reasoner.language_model,
        }
    )
    per_sample_counts: list[dict[str, Any]] = []

    def run_joint(record: CalibrationRecord, state: dict[str, Any]) -> None:
        batch = apply_chat_template(
            processor,
            record,
            multimodal=True,
            max_length=max_length,
            enable_thinking=enable_thinking,
            min_image_pixels=min_image_pixels,
            max_image_pixels=max_image_pixels,
        )
        counts = token_counts(batch)
        if counts["image"] <= 0 or counts["language"] <= 0:
            raise ProtocolError(f"Joint AR sample lacks image or language tokens: {record.sample_id}: {counts}")
        if counts["video"] != 0:
            raise ProtocolError(f"Image+text calibration unexpectedly contains video tokens: {record.sample_id}")
        if "pixel_values" not in batch or "image_grid_thw" not in batch:
            raise ProtocolError(f"Joint AR sample has no real image tensors: {record.sample_id}")
        per_sample_counts.append({"sample_id": record.sample_id, **counts})
        state["valid_mask"] = batch["attention_mask"].bool()
        state["visual_mask"] = (
            batch["mm_token_type_ids"].eq(1) & state["valid_mask"]
        )
        reasoner.rope_deltas = None
        model(
            **move_batch(batch, device),
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )

    def run_separate(record: CalibrationRecord, state: dict[str, Any]) -> None:
        batch = apply_chat_template(
            processor,
            record,
            multimodal=False,
            max_length=max_length,
            enable_thinking=enable_thinking,
            min_image_pixels=min_image_pixels,
            max_image_pixels=max_image_pixels,
        )
        forbidden_input_keys = {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}
        present = forbidden_input_keys & set(batch)
        if present:
            raise ProtocolError(f"Separate AR tokenizer path returned visual inputs: {sorted(present)}")
        input_ids = batch["input_ids"]
        forbidden_token_ids = {
            int(model.config.image_token_id),
            int(model.config.video_token_id),
        }
        actual_ids = set(int(value) for value in input_ids.unique().tolist())
        if forbidden_token_ids & actual_ids:
            raise ProtocolError(f"Separate AR input contains image/video placeholder token: {record.sample_id}")
        counts = token_counts(batch)
        if counts["image"] != 0 or counts["video"] != 0 or counts["language"] <= 0:
            raise ProtocolError(f"Separate AR token-type assertion failed: {record.sample_id}: {counts}")
        per_sample_counts.append({"sample_id": record.sample_id, **counts})
        state["valid_mask"] = batch["attention_mask"].bool()
        state["visual_mask"] = torch.zeros_like(state["valid_mask"], dtype=torch.bool)
        reasoner.rope_deltas = None
        reasoner.language_model(
            input_ids=input_ids.to(device),
            attention_mask=batch["attention_mask"].to(device),
            use_cache=False,
            return_dict=True,
        )

    try:
        if protocol == PROTOCOL_JOINT:
            cache = capture_first_layer_inputs(first_layer, run_joint, records)
        elif protocol == PROTOCOL_SEPARATE:
            with forbidden_forward({"vision": reasoner.visual, "projector": reasoner.projector}):
                cache = capture_first_layer_inputs(first_layer, run_separate, records)
        else:
            raise ValueError(f"Unsupported AR protocol: {protocol}")
    finally:
        counters.close()

    if counters.counts["ar"] != len(records):
        raise ProtocolError(f"Expected {len(records)} AR calls, got {counters.counts['ar']}")
    if protocol == PROTOCOL_JOINT:
        if counters.counts["vision"] != len(records) or counters.counts["projector"] != len(records):
            raise ProtocolError(f"Joint AR did not execute full visual path: {counters.counts}")
    else:
        if counters.counts["vision"] != 0 or counters.counts["projector"] != 0:
            raise ProtocolError(f"Separate AR touched visual path: {counters.counts}")

    return cache, {
        "vision_forward_calls": counters.counts["vision"],
        "projector_forward_calls": counters.counts["projector"],
        "ar_forward_calls": counters.counts["ar"],
        "token_counts": per_sample_counts,
        "visual_mask_mode": (
            "real_multimodal_token_types"
            if protocol == PROTOCOL_JOINT
            else "forced_all_false_text_only"
        ),
    }


def _layer_output_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise ProtocolError(f"Unexpected layer output type: {type(output).__name__}")


def _flatten_hidden_and_masks(
    hidden_states: torch.Tensor,
    visual_mask: torch.Tensor,
    valid_mask: torch.Tensor | None,
    context: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    if hidden_states.ndim == 3:
        if hidden_states.shape[0] != 1:
            raise ProtocolError(
                f"{context}: TAMP requires calibration batch size 1, got {tuple(hidden_states.shape)}"
            )
        flat_hidden = hidden_states[0]
    elif hidden_states.ndim == 2:
        flat_hidden = hidden_states
    else:
        raise ProtocolError(
            f"{context}: expected rank-2/3 hidden states, got {tuple(hidden_states.shape)}"
        )

    original_mask_shape = tuple(visual_mask.shape)
    flat_visual = visual_mask.to(device=flat_hidden.device, dtype=torch.bool).reshape(-1)
    if flat_visual.numel() != flat_hidden.shape[0]:
        raise ProtocolError(
            f"{context}: hidden/visual-mask mismatch: {flat_hidden.shape[0]} vs {flat_visual.numel()}"
        )
    if valid_mask is None:
        flat_valid = torch.ones_like(flat_visual)
    else:
        flat_valid = valid_mask.to(device=flat_hidden.device, dtype=torch.bool).reshape(-1)
        if flat_valid.numel() != flat_hidden.shape[0]:
            raise ProtocolError(
                f"{context}: hidden/valid-mask mismatch: {flat_hidden.shape[0]} vs {flat_valid.numel()}"
            )
    if not flat_valid.any().item():
        raise ProtocolError(f"{context}: all calibration tokens are padding")
    return flat_hidden, flat_visual, flat_valid, original_mask_shape


def collect_atv_distance_record(
    input_hidden: torch.Tensor,
    output_hidden: torch.Tensor,
    visual_mask: torch.Tensor,
    valid_mask: torch.Tensor | None,
    sample_id: str,
) -> dict[str, Any]:
    raise ProtocolError("Legacy ATV selection is forbidden in the Cosmos TAMP implementation")
    context = f"ATV multimodal sample {sample_id}"
    flat_input, flat_visual, flat_valid, mask_shape = _flatten_hidden_and_masks(
        input_hidden,
        visual_mask,
        valid_mask,
        context,
    )
    flat_output, output_visual, output_valid, _ = _flatten_hidden_and_masks(
        output_hidden,
        visual_mask,
        valid_mask,
        context + " output",
    )
    if flat_output.shape != flat_input.shape:
        raise ProtocolError(
            f"{context}: layer input/output shape changed: "
            f"{tuple(flat_input.shape)} vs {tuple(flat_output.shape)}"
        )
    if not torch.equal(flat_visual, output_visual) or not torch.equal(flat_valid, output_valid):
        raise ProtocolError(f"{context}: modality masks changed while computing ATV distance")

    visual_positions = torch.where(flat_visual & flat_valid)[0]
    text_positions = torch.where((~flat_visual) & flat_valid)[0]
    if visual_positions.numel() == 0 or text_positions.numel() == 0:
        raise ProtocolError(
            f"{context}: joint ATV requires visual and language tokens, got "
            f"visual={visual_positions.numel()}, language={text_positions.numel()}"
        )
    distances = 1 - torch.nn.functional.cosine_similarity(
        flat_input.index_select(0, visual_positions),
        flat_output.index_select(0, visual_positions),
        dim=-1,
    )
    if not torch.isfinite(distances).all().item():
        raise ProtocolError(f"{context}: non-finite visual-token cosine distance")
    return {
        "sample_id": sample_id,
        "mask_shape": mask_shape,
        "valid_mask": flat_valid.detach().cpu(),
        "visual_mask": flat_visual.detach().cpu(),
        "visual_positions": visual_positions.detach().cpu(),
        "distances": distances.detach(),
        "text_tokens": int(text_positions.numel()),
        "visual_tokens": int(visual_positions.numel()),
        "padding_tokens": int((~flat_valid).sum().item()),
    }


def finalize_atv_visual_selection(
    distance_records: list[dict[str, Any]],
    alpha: float,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    raise ProtocolError("Legacy ATV selection is forbidden in the Cosmos TAMP implementation")
    if not distance_records:
        raise ProtocolError("ATV visual selection received no calibration samples")
    all_distances = torch.cat([record["distances"] for record in distance_records])
    mean_distance = float(all_distances.mean().item())
    selection_scale = min(1.0, float(alpha) * mean_distance)
    if selection_scale < 0.0:
        raise ProtocolError(
            f"ATV selection scale became negative ({selection_scale}); cosine-distance contract failed"
        )

    retained_masks: list[torch.Tensor] = []
    text_tokens: list[int] = []
    visual_tokens: list[int] = []
    padding_tokens: list[int] = []
    selected_visual_tokens: list[int] = []
    selected_visual_indices: list[list[int]] = []
    for record in distance_records:
        k = round(selection_scale * int(record["text_tokens"]))
        k = min(int(record["visual_tokens"]), int(k))
        if k > 0:
            relative_indices = torch.topk(record["distances"], k=k).indices.sort().values
        else:
            relative_indices = torch.empty(
                0,
                dtype=torch.long,
                device=record["distances"].device,
            )
        relative_indices_cpu = relative_indices.cpu()
        selected_positions = record["visual_positions"].index_select(0, relative_indices_cpu)
        retained = record["valid_mask"] & ~record["visual_mask"]
        retained[selected_positions] = True
        retained_masks.append(retained.reshape(record["mask_shape"]))
        text_tokens.append(int(record["text_tokens"]))
        visual_tokens.append(int(record["visual_tokens"]))
        padding_tokens.append(int(record["padding_tokens"]))
        selected_visual_tokens.append(int(k))
        selected_visual_indices.append(relative_indices_cpu.tolist())

    return retained_masks, {
        "mode": "multimodal_atv",
        "sample_count": len(distance_records),
        "alpha": float(alpha),
        "alpha_effective": True,
        "mean_cosine_distance": mean_distance,
        "selection_scale": selection_scale,
        "text_tokens": text_tokens,
        "visual_tokens": visual_tokens,
        "padding_tokens": padding_tokens,
        "selected_visual_tokens": selected_visual_tokens,
        "selected_visual_indices": selected_visual_indices,
        "total_text_tokens": sum(text_tokens),
        "total_visual_tokens": sum(visual_tokens),
        "total_selected_visual_tokens": sum(selected_visual_tokens),
        "visual_token_selection": "topk_cosine_distance",
    }


def compute_atv_text_only_selection(
    cache: list[LayerSample],
    alpha: float,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    raise ProtocolError("Legacy ATV selection is forbidden in the Cosmos TAMP implementation")
    retained_masks: list[torch.Tensor] = []
    text_tokens: list[int] = []
    padding_tokens: list[int] = []
    for sample in cache:
        _, flat_visual, flat_valid, mask_shape = _flatten_hidden_and_masks(
            sample.hidden_states,
            sample.visual_mask,
            sample.valid_mask,
            f"ATV text-only sample {sample.sample_id}",
        )
        if (flat_visual & flat_valid).any().item():
            raise ProtocolError(
                f"ATV text-only sample {sample.sample_id} contains visual tokens"
            )
        retained_masks.append(flat_valid.detach().cpu().reshape(mask_shape))
        text_tokens.append(int(flat_valid.sum().item()))
        padding_tokens.append(int((~flat_valid).sum().item()))
    zeros = [0 for _ in cache]
    return retained_masks, {
        "mode": "text_only_zero_visual",
        "sample_count": len(cache),
        "alpha": float(alpha),
        "alpha_effective": False,
        "mean_cosine_distance": None,
        "selection_scale": None,
        "text_tokens": text_tokens,
        "visual_tokens": zeros,
        "padding_tokens": padding_tokens,
        "selected_visual_tokens": zeros,
        "selected_visual_indices": [[] for _ in cache],
        "total_text_tokens": sum(text_tokens),
        "total_visual_tokens": 0,
        "total_selected_visual_tokens": 0,
        "visual_token_selection": "forced_zero",
    }


def prune_linear_weight(
    linear: nn.Linear,
    stats: ActivationStats,
    sparsity: float,
) -> tuple[dict[str, Any], torch.Tensor]:
    weight = linear.weight.data
    rows, columns = weight.shape
    if columns != stats.columns:
        raise ProtocolError(f"Weight/stat width mismatch: weight={tuple(weight.shape)}, stats={stats.columns}")
    prune_per_row = int(columns * sparsity)
    zeros_before = int((weight == 0).sum().item())
    mask = torch.zeros_like(weight, dtype=torch.bool)
    importance_score_mean = 0.0
    if prune_per_row > 0:
        scaler = stats.scaler_row.to(device=weight.device, dtype=torch.float32).clamp_min_(0).sqrt_()
        metric = weight.detach().float().abs().mul_(scaler.unsqueeze(0))
        importance_score_mean = float(metric.abs().mean().item())
        # Match the LLaVA kernel exactly: stable per-row sort, then take
        # floor(columns * sparsity) lowest-importance input connections.
        indices = torch.sort(metric, dim=-1, stable=True).indices[:, :prune_per_row]
        mask.scatter_(1, indices, True)
        weight.masked_fill_(mask, 0)
        del metric, indices, scaler
    zeros_after = int((weight == 0).sum().item())
    report = {
        "shape": [rows, columns],
        "parameters": int(weight.numel()),
        "target_sparsity": float(sparsity),
        "pruned_per_output_row": prune_per_row,
        "activation_samples": stats.nsamples,
        "activation_hook_calls": stats.calls,
        "importance_score_mean": importance_score_mean,
        "zeros_before": zeros_before,
        "zeros_after": zeros_after,
        "actual_zero_ratio": zeros_after / float(weight.numel()),
    }
    return report, mask


def prune_ar_layers_atv(
    layers: nn.ModuleList,
    cache: list[LayerSample],
    sparsity: float,
    prefix: str,
    device: torch.device,
    max_layers: int,
    save_masks: bool,
    protocol: str,
    alpha: float,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], list[LayerSample]]:
    raise ProtocolError("Legacy ATV pruning is forbidden; use prune_ar_layers_tamp")
    limit = len(layers) if max_layers <= 0 else min(max_layers, len(layers))
    reports: list[dict[str, Any]] = []
    masks: dict[str, torch.Tensor] = {}

    for layer_index in range(limit):
        layer = layers[layer_index]
        linears = find_linear_modules(layer)
        if not linears:
            raise ProtocolError(f"No Linear modules in {prefix}.{layer_index}")

        if protocol == PROTOCOL_JOINT:
            distance_records: list[dict[str, Any]] = []
            with torch.inference_mode():
                for sample in cache:
                    hidden_states = sample.hidden_states.to(device=device, non_blocking=True)
                    kwargs = recursive_to_device(sample.layer_kwargs, device)
                    output = _layer_output_tensor(layer(hidden_states, **kwargs))
                    distance_records.append(
                        collect_atv_distance_record(
                            hidden_states,
                            output,
                            sample.visual_mask,
                            sample.valid_mask,
                            sample.sample_id,
                        )
                    )
                    del hidden_states, kwargs, output
            retained_masks, selection_report = finalize_atv_visual_selection(
                distance_records,
                alpha,
            )
        elif protocol == PROTOCOL_SEPARATE:
            retained_masks, selection_report = compute_atv_text_only_selection(cache, alpha)
        else:
            raise ValueError(f"Unsupported ATV protocol: {protocol}")
        selection_report["layer_index"] = layer_index
        selection_report["layer_name"] = f"{prefix}.{layer_index}"

        stats = {name: ActivationStats(module.in_features) for name, module in linears.items()}
        current_mask: list[torch.Tensor | None] = [None]
        handles = []

        def make_hook(name: str):
            def hook(_module, inputs, _output):
                stats[name].add(inputs[0], current_mask[0])

            return hook

        for name, module in linears.items():
            handles.append(module.register_forward_hook(make_hook(name)))

        try:
            with torch.inference_mode():
                for sample, retained_mask in zip(cache, retained_masks):
                    hidden_states = sample.hidden_states.to(device=device, non_blocking=True)
                    kwargs = recursive_to_device(sample.layer_kwargs, device)
                    current_mask[0] = retained_mask
                    _layer_output_tensor(layer(hidden_states, **kwargs))
                    del hidden_states, kwargs
        finally:
            for handle in handles:
                handle.remove()

        layer_report: dict[str, Any] = {
            "layer_index": layer_index,
            "layer_name": f"{prefix}.{layer_index}",
            "atv_selection": selection_report,
            "linears": {},
        }
        for name, linear in linears.items():
            full_name = f"{prefix}.{layer_index}.{name}"
            linear_report, mask = prune_linear_weight(linear, stats[name], sparsity)
            layer_report["linears"][full_name] = linear_report
            if save_masks:
                masks[full_name] = mask.cpu()
            del mask

        # Recompute with the newly-pruned current layer so every downstream layer
        # sees the same sequentially-pruned activations as the LLaVA implementation.
        next_cache: list[LayerSample] = []
        with torch.inference_mode():
            for sample in cache:
                hidden_states = sample.hidden_states.to(device=device, non_blocking=True)
                kwargs = recursive_to_device(sample.layer_kwargs, device)
                output = _layer_output_tensor(layer(hidden_states, **kwargs))
                next_cache.append(
                    LayerSample(
                        hidden_states=output.detach().cpu(),
                        layer_kwargs=sample.layer_kwargs,
                        valid_mask=sample.valid_mask,
                        visual_mask=sample.visual_mask,
                        sample_id=sample.sample_id,
                    )
                )
                del hidden_states, kwargs, output
        cache = next_cache
        layer_report["parameters"] = sum(
            item["parameters"] for item in layer_report["linears"].values()
        )
        layer_report["zeros_after"] = sum(
            item["zeros_after"] for item in layer_report["linears"].values()
        )
        layer_report["actual_zero_ratio"] = (
            layer_report["zeros_after"] / float(layer_report["parameters"])
        )
        reports.append(layer_report)
        print(
            f"[{prefix}] layer {layer_index + 1}/{limit}: "
            f"mode={selection_report['mode']} "
            f"selected_visual={selection_report['total_selected_visual_tokens']} "
            f"zero_ratio={layer_report['actual_zero_ratio']:.6f}",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return reports, masks, cache


@dataclasses.dataclass
class DasSimilarityStats:
    """Per-Linear DAS statistics matching the operational LLaVA code."""

    totals: dict[str, float] = dataclasses.field(
        default_factory=lambda: {"v": 0.0, "l": 0.0, "vl": 0.0}
    )
    counts: dict[str, int] = dataclasses.field(
        default_factory=lambda: {"v": 0, "l": 0, "vl": 0}
    )
    calls: int = 0

    def add(
        self,
        outputs: torch.Tensor,
        visual_mask: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> None:
        hidden, visual, valid, _ = _flatten_hidden_and_masks(
            outputs,
            visual_mask,
            valid_mask,
            context="DAS Linear output",
        )
        hidden = F.normalize(hidden.float(), dim=-1, eps=1e-8)
        visual_tokens = hidden[visual & valid]
        language_tokens = hidden[(~visual) & valid]

        def positive_pair_mean(tokens: torch.Tensor) -> float:
            similarities = torch.mm(tokens, tokens.T).triu(diagonal=1)
            values = similarities[similarities > 0]
            return float(values.mean().item()) if values.numel() else 0.0

        if visual_tokens.shape[0] >= 2:
            self.totals["v"] += positive_pair_mean(visual_tokens)
            self.counts["v"] += 1
        if language_tokens.shape[0] >= 2:
            self.totals["l"] += positive_pair_mean(language_tokens)
            self.counts["l"] += 1
        if visual_tokens.shape[0] >= 1 and language_tokens.shape[0] >= 1:
            self.totals["vl"] += float(
                torch.mm(visual_tokens, language_tokens.T).mean().item()
            )
            self.counts["vl"] += 1
        self.calls += 1

    def finalize(self, protocol: str) -> tuple[float, dict[str, Any]]:
        means = {
            name: self.totals[name] / self.counts[name]
            if self.counts[name]
            else None
            for name in ("v", "l", "vl")
        }
        if protocol == PROTOCOL_JOINT:
            if any(means[name] is None for name in ("v", "l", "vl")):
                raise ProtocolError(
                    f"Joint DAS is missing a modality-pair statistic: {means}"
                )
            terms = ("v", "l", "vl")
            formula = "(1-s_v)+(1-s_l)+(1-s_vl)"
        elif protocol == PROTOCOL_SEPARATE:
            if means["l"] is None:
                raise ProtocolError("Text-only DAS has no language-language pairs")
            if means["v"] is not None or means["vl"] is not None:
                raise ProtocolError(
                    f"Text-only DAS observed forbidden visual statistics: {means}"
                )
            terms = ("l",)
            formula = "3*(1-s_l)"
        else:
            raise ValueError(f"Unsupported TAMP protocol: {protocol}")

        score = sum(1.0 - float(means[name]) for name in terms)
        score *= 3.0 / float(len(terms))
        if not math.isfinite(score):
            raise ProtocolError(f"Non-finite DAS score: {score}")
        return score, {
            "similarities": means,
            "defined_terms": list(terms),
            "formula": formula,
            "score": score,
            "calls": self.calls,
        }


def clone_layer_cache(cache: list[LayerSample]) -> list[LayerSample]:
    return [
        LayerSample(
            hidden_states=sample.hidden_states.clone(),
            layer_kwargs=sample.layer_kwargs,
            valid_mask=sample.valid_mask,
            visual_mask=sample.visual_mask,
            sample_id=sample.sample_id,
        )
        for sample in cache
    ]


def collect_das_scores(
    model: Cosmos3EdgeForConditionalGeneration,
    cache: list[LayerSample],
    device: torch.device,
    protocol: str,
    max_layers: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Run the dense AR stack once and return one DAS score per Linear."""

    layers = model.model.language_model.layers
    layer_limit = len(layers) if max_layers <= 0 else min(max_layers, len(layers))
    current_cache = clone_layer_cache(cache)
    scores: dict[str, float] = {}
    reports: list[dict[str, Any]] = []

    for layer_index in range(layer_limit):
        layer = layers[layer_index]
        linears = find_linear_modules(layer)
        stats = {name: DasSimilarityStats() for name in linears}
        active_sample = {"index": -1}

        def make_hook(name: str):
            def hook(_module, _inputs, output):
                sample = current_cache[active_sample["index"]]
                stats[name].add(
                    _layer_output_tensor(output),
                    sample.visual_mask,
                    sample.valid_mask,
                )

            return hook

        handles = [module.register_forward_hook(make_hook(name)) for name, module in linears.items()]
        next_cache: list[LayerSample] = []
        try:
            for sample_index, sample in enumerate(current_cache):
                active_sample["index"] = sample_index
                hidden = sample.hidden_states.to(device=device, non_blocking=True)
                kwargs = recursive_to_device(sample.layer_kwargs, device)
                with torch.no_grad():
                    output = _layer_output_tensor(layer(hidden, **kwargs))
                next_cache.append(
                    LayerSample(
                        hidden_states=recursive_to_cpu(output),
                        layer_kwargs=sample.layer_kwargs,
                        valid_mask=sample.valid_mask,
                        visual_mask=sample.visual_mask,
                        sample_id=sample.sample_id,
                    )
                )
        finally:
            for handle in handles:
                handle.remove()

        linear_reports: dict[str, Any] = {}
        for name in sorted(linears):
            full_weight_name = f"{AR_PREFIX}.{layer_index}.{name}.weight"
            score, report = stats[name].finalize(protocol)
            scores[full_weight_name] = score
            linear_reports[name] = report
        reports.append({"layer": layer_index, "linears": linear_reports})
        current_cache = next_cache
        print(
            f"DAS layer {layer_index}: linears={len(linears)} "
            f"score_min={min(scores[f'{AR_PREFIX}.{layer_index}.{name}.weight'] for name in linears):.8f} "
            f"score_max={max(scores[f'{AR_PREFIX}.{layer_index}.{name}.weight'] for name in linears):.8f}",
            flush=True,
        )
    return scores, reports


def allocate_per_linear_sparsity(
    model: Cosmos3EdgeForConditionalGeneration,
    das_scores: dict[str, float],
    target_sparsity: float,
    max_sparsity_per_linear: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Allocate an exact global integer keep budget with a per-Linear cap."""

    named_linears = {
        f"{AR_PREFIX}.{layer_index}.{name}.weight": module
        for layer_index, layer in enumerate(model.model.language_model.layers)
        for name, module in find_linear_modules(layer).items()
        if f"{AR_PREFIX}.{layer_index}.{name}.weight" in das_scores
    }
    if set(named_linears) != set(das_scores):
        raise ProtocolError(
            "DAS/AR target mismatch: "
            f"missing={sorted(set(named_linears) - set(das_scores))[:10]}, "
            f"extra={sorted(set(das_scores) - set(named_linears))[:10]}"
        )

    names = sorted(named_linears)
    sizes = [int(named_linears[name].weight.numel()) for name in names]
    total_parameters = sum(sizes)
    target_keep = int(total_parameters * (1.0 - target_sparsity))
    minimum_keep = [int(math.ceil(size * (1.0 - max_sparsity_per_linear))) for size in sizes]
    if sum(minimum_keep) > target_keep:
        raise ProtocolError(
            "DAS max-sparsity cap makes the global keep budget infeasible: "
            f"minimum={sum(minimum_keep)} target={target_keep}"
        )

    keep = list(minimum_keep)
    safe_scores = [
        max(0.0, float(das_scores[name]))
        if math.isfinite(float(das_scores[name]))
        else 0.0
        for name in names
    ]
    working_scores = list(safe_scores)
    while sum(keep) < target_keep:
        active = [index for index, size in enumerate(sizes) if keep[index] < size]
        if not active:
            raise ProtocolError("DAS allocation exhausted all Linear capacities")
        score_total = sum(working_scores[index] for index in active)
        if not math.isfinite(score_total) or score_total <= 0:
            for index in range(len(working_scores)):
                working_scores[index] = 1.0 if index in active else 0.0
            score_total = float(len(active))

        remaining = target_keep - sum(keep)
        requested_additions = [0] * len(keep)
        for index in active:
            requested_additions[index] = int(
                math.ceil(working_scores[index] / score_total * remaining)
            )
        before = sum(keep)
        for index in active:
            keep[index] = min(
                sizes[index], keep[index] + requested_additions[index]
            )
            if keep[index] >= sizes[index]:
                working_scores[index] = 0.0

        if sum(keep) == before and sum(keep) < target_keep:
            need = target_keep - sum(keep)
            for index in active:
                can_add = min(need, sizes[index] - keep[index])
                keep[index] += can_add
                need -= can_add
                if need == 0:
                    break
            if need:
                raise ProtocolError("DAS allocator stalled before the global keep budget")

        if sum(keep) > target_keep:
            excess = sum(keep) - target_keep
            while excess > 0:
                removed = 0
                # Match torch.argsort(keep, descending=True, stable=True).
                for index in sorted(range(len(keep)), key=lambda item: (-keep[item], item)):
                    removable = max(0, keep[index] - minimum_keep[index])
                    count = min(excess, removable)
                    keep[index] -= count
                    excess -= count
                    removed += count
                    if excess == 0:
                        break
                if removed == 0:
                    raise ProtocolError(
                        "DAS could not remove rounded excess without violating the cap"
                    )

    if sum(keep) != target_keep:
        raise ProtocolError(f"DAS keep budget mismatch: {sum(keep)} != {target_keep}")
    if any(value < floor for value, floor in zip(keep, minimum_keep)):
        raise ProtocolError("DAS allocation violated max-sparsity cap")

    sparsities = {
        name: 1.0 - keep_count / size
        for name, keep_count, size in zip(names, keep, sizes)
    }
    return sparsities, {
        "granularity": "per_linear_tensor",
        "target_sparsity": float(target_sparsity),
        "max_sparsity_per_linear": float(max_sparsity_per_linear),
        "target_linear_count": len(names),
        "target_parameter_count": total_parameters,
        "target_keep_parameter_count": target_keep,
        "allocated_keep_parameter_count": sum(keep),
        "allocated_zero_parameter_count": total_parameters - sum(keep),
        "allocated_sparsity": (total_parameters - sum(keep)) / total_parameters,
        "per_linear": {
            name: {
                "das_score": float(das_scores[name]),
                "parameters": size,
                "keep_parameters": keep_count,
                "allocated_sparsity": sparsities[name],
            }
            for name, keep_count, size in zip(names, keep, sizes)
        },
    }


@dataclasses.dataclass
class AmiaActivationStats:
    columns: int
    sum_sq: torch.Tensor = dataclasses.field(init=False)
    selected_tokens: int = 0
    valid_tokens: int = 0
    calls: int = 0
    density_sum: float = 0.0

    def __post_init__(self) -> None:
        self.sum_sq = torch.zeros(self.columns, dtype=torch.float32, device="cpu")

    @staticmethod
    def gaussian_rbf(x: torch.Tensor) -> torch.Tensor:
        x_norm = x.pow(2).sum(dim=1).view(-1, 1)
        distances = (x_norm + x_norm.T - 2.0 * torch.mm(x, x.T)).clamp_min(0)
        return torch.exp(-distances / 2.0)

    @staticmethod
    def density(hidden: torch.Tensor, visual: torch.Tensor) -> float:
        visual_tokens = hidden[visual]
        language_tokens = hidden[~visual]

        def positive_pair_mean(tokens: torch.Tensor) -> float:
            similarities = torch.mm(tokens, tokens.T).triu(diagonal=1)
            values = similarities[similarities > 0]
            return float(values.mean().item()) if values.numel() else 0.0

        terms: list[float] = []
        if visual_tokens.shape[0] >= 2:
            terms.append(positive_pair_mean(visual_tokens))
        if language_tokens.shape[0] >= 2:
            terms.append(positive_pair_mean(language_tokens))
        if visual_tokens.shape[0] >= 1 and language_tokens.shape[0] >= 1:
            terms.append(float(torch.mm(visual_tokens, language_tokens.T).mean().item()))
        if not terms:
            raise ProtocolError("AMIA density is undefined for fewer than two usable tokens")
        return min(1.0, max(0.0, sum(terms) / len(terms)))

    def add(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        visual_mask: torch.Tensor,
        valid_mask: torch.Tensor | None,
        attention_score: torch.Tensor,
    ) -> None:
        flat_output, flat_visual, flat_valid, _ = _flatten_hidden_and_masks(
            outputs,
            visual_mask,
            valid_mask,
            context="AMIA Linear output",
        )
        if inputs.ndim == 3 and inputs.shape[0] == 1:
            flat_input = inputs[0]
        elif inputs.ndim == 2:
            flat_input = inputs
        else:
            raise ProtocolError(f"AMIA requires batch size 1, got {tuple(inputs.shape)}")
        if flat_input.shape[0] != flat_output.shape[0] or flat_input.shape[1] != self.columns:
            raise ProtocolError(
                f"AMIA input/output token mismatch: input={tuple(flat_input.shape)}, "
                f"output={tuple(flat_output.shape)}, columns={self.columns}"
            )
        score = attention_score.to(device=flat_output.device, dtype=torch.float32).reshape(-1)
        if score.numel() != flat_output.shape[0]:
            raise ProtocolError(
                f"AMIA attention/token mismatch: {score.numel()} != {flat_output.shape[0]}"
            )

        hidden = F.normalize(flat_output[flat_valid].float(), dim=-1, eps=1e-8)
        visual = flat_visual[flat_valid]
        score = torch.nan_to_num(score[flat_valid], nan=0.0, posinf=0.0, neginf=0.0)
        inputs_valid = flat_input[flat_valid]
        density = self.density(hidden, visual)
        distances = (1.0 - torch.mm(hidden, hidden.T)).clamp_min(0)
        num_tokens = int(hidden.shape[0])
        num_neigh = min(3, num_tokens - 1)

        if num_neigh < 1:
            selected = torch.tensor([0], dtype=torch.long, device=hidden.device)
        else:
            knn = torch.topk(distances, k=num_neigh + 1, largest=False).indices[:, 1:]
            graph_score = score + (
                torch.exp(-torch.gather(distances, 1, knn)) * score[knn]
            ).sum(dim=-1)
            kernel = self.gaussian_rbf(hidden)
            selected_set: set[int] = set()
            threshold = math.sqrt(max(0.0, 1.0 - density)) * 0.1
            while len(selected_set) < num_tokens:
                available = torch.ones(num_tokens, dtype=torch.bool, device=hidden.device)
                if selected_set:
                    available[
                        torch.tensor(sorted(selected_set), dtype=torch.long, device=hidden.device)
                    ] = False
                chosen = int(torch.argmax(graph_score.masked_fill(~available, -torch.inf)).item())
                neighbors = knn[chosen]
                graph_score[neighbors] -= torch.exp(-distances[chosen, neighbors] * 0.2) * torch.maximum(
                    graph_score[chosen], torch.zeros_like(graph_score[chosen])
                )
                selected_set.add(chosen)
                selected_tensor = torch.tensor(
                    sorted(selected_set), dtype=torch.long, device=hidden.device
                )
                graph_score[selected_tensor] = torch.min(graph_score) - 1
                mmd2 = (
                    kernel.mean()
                    + kernel[selected_tensor][:, selected_tensor].mean()
                    - 2.0 * kernel[:, selected_tensor].mean()
                )
                if not torch.isfinite(mmd2):
                    raise ProtocolError("AMIA MMD became non-finite")
                if float(mmd2.item()) <= threshold + 1e-8:
                    break
            selected = torch.tensor(sorted(selected_set), dtype=torch.long, device=hidden.device)

        selected_inputs = inputs_valid.index_select(0, selected)
        self.sum_sq.add_(selected_inputs.detach().float().pow(2).sum(dim=0).cpu())
        self.selected_tokens += int(selected.numel())
        self.valid_tokens += num_tokens
        self.calls += 1
        self.density_sum += density

    @property
    def scaler_row(self) -> torch.Tensor:
        if self.selected_tokens <= 0:
            raise ProtocolError("AMIA selected no activation tokens")
        return self.sum_sq / float(self.selected_tokens)

    @property
    def nsamples(self) -> int:
        # LLaVA's AMIA accumulator normalizes by the number of selected tokens.
        return self.selected_tokens

    def report(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "valid_tokens": self.valid_tokens,
            "selected_tokens": self.selected_tokens,
            "mean_density": self.density_sum / self.calls if self.calls else None,
        }


def layer_attention_scores(
    layer: nn.Module,
    cache: list[LayerSample],
    device: torch.device,
) -> list[torch.Tensor]:
    """Obtain LLaVA-compatible final-valid-query mean-head attention scores."""

    scores: list[torch.Tensor] = []
    for sample in cache:
        captured: dict[str, torch.Tensor] = {}

        def attention_hook(_module, _inputs, output):
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise ProtocolError("Cosmos attention did not return attention weights")
            captured["weights"] = output[1]

        handle = layer.self_attn.register_forward_hook(attention_hook)
        try:
            hidden = sample.hidden_states.to(device=device, non_blocking=True)
            kwargs = recursive_to_device(sample.layer_kwargs, device)
            with torch.no_grad():
                layer(hidden, **kwargs)
        finally:
            handle.remove()
        weights = captured.get("weights")
        if weights is None or weights.ndim != 4 or weights.shape[0] != 1:
            shape = None if weights is None else tuple(weights.shape)
            raise ProtocolError(f"Invalid eager attention tensor: {shape}")
        valid = (
            torch.ones(weights.shape[-1], dtype=torch.bool, device=weights.device)
            if sample.valid_mask is None
            else sample.valid_mask.to(device=weights.device, dtype=torch.bool).reshape(-1)
        )
        if valid.numel() != weights.shape[-1] or weights.shape[-2] != valid.numel():
            raise ProtocolError(
                f"Attention/valid-mask mismatch: attention={tuple(weights.shape)}, valid={valid.numel()}"
            )
        valid_queries = torch.where(valid)[0]
        if valid_queries.numel() == 0:
            raise ProtocolError("AMIA attention received an all-padding sample")
        score = weights[0, :, int(valid_queries[-1].item()), :].float().mean(dim=0)
        score = torch.nan_to_num(score, nan=0.0).masked_fill(~valid, 0.0)
        scores.append(score.detach().cpu())
    return scores


def prune_ar_layers_tamp(
    model: Cosmos3EdgeForConditionalGeneration,
    cache: list[LayerSample],
    sparsity_by_weight: dict[str, float],
    device: torch.device,
    max_layers: int,
    save_masks: bool,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], list[LayerSample]]:
    layers = model.model.language_model.layers
    layer_limit = len(layers) if max_layers <= 0 else min(max_layers, len(layers))
    current_cache = clone_layer_cache(cache)
    reports: list[dict[str, Any]] = []
    masks: dict[str, torch.Tensor] = {}

    for layer_index in range(layer_limit):
        layer = layers[layer_index]
        linears = find_linear_modules(layer)
        attention_scores = layer_attention_scores(layer, current_cache, device)
        stats = {
            name: AmiaActivationStats(module.in_features)
            for name, module in linears.items()
        }
        active_sample = {"index": -1}

        def make_hook(name: str):
            def hook(_module, inputs, output):
                sample_index = active_sample["index"]
                sample = current_cache[sample_index]
                stats[name].add(
                    inputs[0],
                    _layer_output_tensor(output),
                    sample.visual_mask,
                    sample.valid_mask,
                    attention_scores[sample_index],
                )

            return hook

        handles = [module.register_forward_hook(make_hook(name)) for name, module in linears.items()]
        try:
            for sample_index, sample in enumerate(current_cache):
                active_sample["index"] = sample_index
                hidden = sample.hidden_states.to(device=device, non_blocking=True)
                kwargs = recursive_to_device(sample.layer_kwargs, device)
                with torch.no_grad():
                    layer(hidden, **kwargs)
        finally:
            for handle in handles:
                handle.remove()

        linear_reports: dict[str, Any] = {}
        for name, linear in sorted(linears.items()):
            full_name = f"{AR_PREFIX}.{layer_index}.{name}"
            weight_name = f"{full_name}.weight"
            if weight_name not in sparsity_by_weight:
                raise ProtocolError(f"No DAS sparsity allocation for {weight_name}")
            linear_report, mask = prune_linear_weight(
                linear,
                stats[name],
                sparsity_by_weight[weight_name],
            )
            linear_report["amia"] = stats[name].report()
            linear_report["allocated_sparsity"] = sparsity_by_weight[weight_name]
            linear_reports[name] = linear_report
            if save_masks:
                masks[weight_name] = mask

        next_cache: list[LayerSample] = []
        for sample in current_cache:
            hidden = sample.hidden_states.to(device=device, non_blocking=True)
            kwargs = recursive_to_device(sample.layer_kwargs, device)
            with torch.no_grad():
                output = _layer_output_tensor(layer(hidden, **kwargs))
            next_cache.append(
                LayerSample(
                    hidden_states=recursive_to_cpu(output),
                    layer_kwargs=sample.layer_kwargs,
                    valid_mask=sample.valid_mask,
                    visual_mask=sample.visual_mask,
                    sample_id=sample.sample_id,
                )
            )
        layer_parameters = sum(item["parameters"] for item in linear_reports.values())
        layer_zeros = sum(item["zeros_after"] for item in linear_reports.values())
        reports.append(
            {
                "layer": layer_index,
                "parameters": layer_parameters,
                "zeros_after": layer_zeros,
                "zero_ratio": layer_zeros / float(layer_parameters)
                if layer_parameters
                else 0.0,
                "linears": linear_reports,
            }
        )
        current_cache = next_cache
        selected = sum(item["amia"]["selected_tokens"] for item in linear_reports.values())
        valid = sum(item["amia"]["valid_tokens"] for item in linear_reports.values())
        print(
            f"TAMP layer {layer_index}: linears={len(linears)} "
            f"AMIA selected/valid={selected}/{valid}",
            flush=True,
        )
        torch.cuda.empty_cache()
    return reports, masks, current_cache


def target_zero_report(
    model: Cosmos3EdgeForConditionalGeneration,
    max_ar_layers: int = 0,
) -> dict[str, Any]:
    groups: dict[str, list[nn.Linear]] = {"vision": [], "ar": []}
    vision_layers = model.model.visual.encoder.layers
    ar_layers = model.model.language_model.layers
    ar_limit = len(ar_layers) if max_ar_layers <= 0 else min(max_ar_layers, len(ar_layers))
    for layer in vision_layers:
        groups["vision"].extend(find_linear_modules(layer).values())
    for layer in ar_layers[:ar_limit]:
        groups["ar"].extend(find_linear_modules(layer).values())

    report: dict[str, Any] = {}
    for name, linears in groups.items():
        parameters = sum(module.weight.numel() for module in linears)
        zeros = sum(int((module.weight.data == 0).sum().item()) for module in linears)
        report[name] = {
            "parameters": parameters,
            "zeros": zeros,
            "zero_ratio": zeros / float(parameters) if parameters else 0.0,
        }
    combined_parameters = report["ar"]["parameters"]
    combined_zeros = report["ar"]["zeros"]
    report["combined_target_linears"] = {
        "parameters": combined_parameters,
        "zeros": combined_zeros,
        "zero_ratio": combined_zeros / float(combined_parameters) if combined_parameters else 0.0,
        "scope": AR_PREFIX,
    }
    return report


def verify_full_multimodal_forward(
    model: Cosmos3EdgeForConditionalGeneration,
    processor: Any,
    record: CalibrationRecord,
    device: torch.device,
    max_length: int,
    enable_thinking: bool,
    min_image_pixels: int,
    max_image_pixels: int,
) -> dict[str, Any]:
    batch = apply_chat_template(
        processor,
        record,
        multimodal=True,
        max_length=max_length,
        enable_thinking=enable_thinking,
        min_image_pixels=min_image_pixels,
        max_image_pixels=max_image_pixels,
    )
    counts = token_counts(batch)
    if counts["image"] <= 0 or counts["language"] <= 0:
        raise ProtocolError(f"Final verification is not multimodal: {counts}")
    counters = CallCounter(
        {
            "vision": model.model.visual,
            "projector": model.model.projector,
            "ar": model.model.language_model,
        }
    )
    try:
        model.model.rope_deltas = None
        with torch.inference_mode():
            outputs = model(
                **move_batch(batch, device),
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
        logits = outputs.logits
        if logits.numel() == 0 or not torch.isfinite(logits).all().item():
            raise ProtocolError("Final multimodal Reasoner forward produced empty or non-finite logits")
    finally:
        counters.close()
    expected = {"vision": 1, "projector": 1, "ar": 1}
    if counters.counts != expected:
        raise ProtocolError(f"Final multimodal path mismatch: expected {expected}, got {counters.counts}")
    return {
        "sample_id": record.sample_id,
        "token_counts": counts,
        "forward_calls": counters.counts,
        "logits_shape": list(logits.shape),
        "logits_finite": True,
    }


def prepare_output_dir(path_string: str) -> Path:
    path = Path(path_string).expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def summarize_layer_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    parameters = sum(report["parameters"] for report in reports)
    zeros = sum(report["zeros_after"] for report in reports)
    return {
        "layers_pruned": len(reports),
        "parameters": parameters,
        "zeros_after": zeros,
        "zero_ratio": zeros / float(parameters) if parameters else 0.0,
    }


def all_parameter_zero_report(model: nn.Module) -> dict[str, Any]:
    seen: set[int] = set()
    parameters = 0
    zeros = 0
    with torch.inference_mode():
        for parameter in model.parameters():
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            parameters += parameter.numel()
            zeros += int((parameter.data == 0).sum().item())
    return {
        "parameters": parameters,
        "zeros": zeros,
        "zero_ratio": zeros / float(parameters) if parameters else 0.0,
    }


def compact_audit(audit: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in audit.items() if not key.endswith("_names")}


def run_pruning(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.time()
    output_dir = prepare_output_dir(args.output_dir)
    state_path = output_dir / "state.json"
    write_json(
        state_path,
        {
            "phase": "starting",
            "protocol": args.protocol,
            "started_unix": started_at,
            "pid": os.getpid(),
        },
    )

    try:
        calibration = build_protocol_calibration(args)
        device = torch.device(args.device)
        model, processor = load_reasoner(args)
        audit = module_audit(model)
        print(json.dumps(compact_audit(audit), indent=2, ensure_ascii=False), flush=True)
        target_allowlist = [f"{name}.weight" for name in audit["ar_linear_names"]]
        target_allowlist_sha256 = hashlib.sha256(
            "\n".join(target_allowlist).encode("utf-8")
        ).hexdigest()

        total_ar_layers = audit["ar_layer_count"]
        ar_limit = total_ar_layers if args.max_ar_layers <= 0 else min(args.max_ar_layers, total_ar_layers)
        partial_run = ar_limit < total_ar_layers
        if partial_run and args.save_model and not args.allow_partial_save:
            raise ProtocolError(
                "Partial-layer smoke runs cannot save a checkpoint unless --allow-partial-save is explicit"
            )

        metadata: dict[str, Any] = {
            "schema_version": 1,
            "algorithm": "TAMP",
            "algorithm_components": ["DAS", "AMIA", "WANDA"],
            "algorithm_variant": (
                "joint_multimodal_reasoner_ar"
                if args.protocol == PROTOCOL_JOINT
                else "separate_text_only_reasoner_ar"
            ),
            "protocol": (
                "cosmos_tamp_joint_reasoner"
                if args.protocol == PROTOCOL_JOINT
                else "cosmos_tamp_separate_textonly_reasoner"
            ),
            "protocol_short": args.protocol,
            "model_path": str(Path(args.model_path).resolve()),
            "model_class": type(model).__name__,
            "transformers_version": __import__("transformers").__version__,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "dtype": args.dtype,
            "attention_implementation": args.attn_implementation,
            "min_image_pixels": args.min_image_pixels,
            "max_image_pixels": args.max_image_pixels,
            "seed": args.seed,
            "experiment_model": "Cosmos3-Edge Reasoner",
            "nsamples": len(calibration.ar_records),
            "calibration_json": [str(Path(path).resolve()) for path in args.calibration_json],
            "ar_calibration_json": [str(Path(path).resolve()) for path in args.ar_calibration_json],
            "verification_json": [str(Path(path).resolve()) for path in args.verification_json],
            "calibration_presets": list(args.calibration_preset),
            "calibration_pairing": calibration.pairing,
            "ar_calibration_sample_ids": [record.sample_id for record in calibration.ar_records],
            "ar_calibration_records": [
                {
                    "sample_id": record.sample_id,
                    "source_path": record.source_path,
                    "source_index": record.source_index,
                    "dataset": record.dataset,
                }
                for record in calibration.ar_records
            ],
            "calibration_modality": (
                "multimodal" if args.protocol == PROTOCOL_JOINT else "text"
            ),
            "vision_sparsity_target": 0.0,
            "target_ar_linear_sparsity": args.ar_sparsity,
            "ar_sparsity_target": args.ar_sparsity,
            "max_sparsity_per_linear": args.max_sparsity_per_linear,
            "uniform_sparsity": False,
            "sparsity_allocation": "DAS density_sum per Linear tensor",
            "token_selection": "AMIA",
            "target_scope": f"{AR_PREFIX}.*.nn.Linear.weight",
            "target_allowlist": target_allowlist,
            "target_allowlist_sha256": target_allowlist_sha256,
            "vision_target_count": 0,
            "projector_target_count": 0,
            "ar_target_linear_count": len(target_allowlist),
            "ar_target_parameter_count": audit["ar_target_parameter_count"],
            "ar_sequence_source": (
                "real_visual_plus_language_embeddings"
                if args.protocol == PROTOCOL_JOINT
                else "language_tokenizer_embedding_only"
            ),
            "tamp": {
                "das_joint_formula": "(1-s_v)+(1-s_l)+(1-s_vl)",
                "das_separate_formula": "3*(1-s_l)",
                "das_granularity": "per_linear_tensor",
                "amia": "causal-attention KNN graph selection with MMD stopping",
                "wanda_metric": "abs(W[o,i]) * sqrt(mean_over_AMIA_selected_tokens(x[...,i]^2))",
                "mask": "per-Linear, per-output-row bottom-k",
                "retained_tokens": (
                    "AMIA-selected valid tokens from the real fused visual+language sequence"
                    if args.protocol == PROTOCOL_JOINT
                    else "AMIA-selected valid language-only tokens; zero visual tokens"
                ),
                "global_ranking": False,
                "structured_nm": False,
            },
            "execution_order": [
                (
                    "capture fused image+text AR activations through the dense vision encoder and projector"
                    if args.protocol == PROTOCOL_JOINT
                    else "capture language-tokenizer-only AR activations with vision/projector forbidden"
                ),
                "dense AR pass: compute one DAS density score per target Linear",
                "allocate the exact global keep budget per Linear with the configured sparsity cap",
                "per AR layer: compute AMIA token selection, collect selected-token activations, apply WANDA, propagate sparse output",
                "verify full image+text Reasoner forward",
            ],
            "generator_excluded": True,
            "vision_encoder_dense": True,
            "projector_dense": True,
            "embedding_norm_lm_head_dense": True,
            "partial_run": partial_run,
            "vision_layers_requested": 0,
            "vision_layers_pruned": 0,
            "ar_layers_requested": ar_limit,
            "module_audit": audit,
            "started_unix": started_at,
        }
        metadata["pre_prune_zero_report"] = target_zero_report(
            model,
            max_ar_layers=args.max_ar_layers,
        )
        write_json(output_dir / "metadata.running.json", metadata)
        write_json(
            state_path,
            {
                "phase": "ar_calibration",
                "protocol": metadata["protocol"],
                "started_unix": started_at,
                "pid": os.getpid(),
            },
        )
        ar_cache, ar_dataflow = build_ar_cache(
            model=model,
            processor=processor,
            records=calibration.ar_records,
            protocol=args.protocol,
            device=device,
            max_length=args.max_length,
            enable_thinking=args.enable_thinking,
            min_image_pixels=args.min_image_pixels,
            max_image_pixels=args.max_image_pixels,
        )
        write_json(
            state_path,
            {
                "phase": "das_scoring",
                "protocol": metadata["protocol"],
                "started_unix": started_at,
                "pid": os.getpid(),
            },
        )
        das_scores, das_reports = collect_das_scores(
            model=model,
            cache=ar_cache,
            device=device,
            protocol=args.protocol,
            max_layers=args.max_ar_layers,
        )
        sparsity_by_weight, allocation_report = allocate_per_linear_sparsity(
            model=model,
            das_scores=das_scores,
            target_sparsity=args.ar_sparsity,
            max_sparsity_per_linear=args.max_sparsity_per_linear,
        )
        write_json(output_dir / "sparsity_allocation.json", allocation_report)
        write_json(
            state_path,
            {
                "phase": "amia_wanda_pruning",
                "protocol": metadata["protocol"],
                "started_unix": started_at,
                "pid": os.getpid(),
            },
        )
        ar_reports, ar_masks, _ = prune_ar_layers_tamp(
            model=model,
            cache=ar_cache,
            sparsity_by_weight=sparsity_by_weight,
            device=device,
            max_layers=args.max_ar_layers,
            save_masks=args.save_masks,
        )
        del ar_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        metadata["ar_dataflow"] = ar_dataflow
        metadata["vision_pruning"] = {
            "enabled": False,
            "reason": "TAMP prunes Reasoner AR/LLM Linear weights only",
            "summary": {
                "layers_pruned": 0,
                "parameters": 0,
                "zeros_after": 0,
                "zero_ratio": 0.0,
            },
            "layers": [],
        }
        metadata["ar_pruning"] = {
            "summary": summarize_layer_reports(ar_reports),
            "layers": ar_reports,
        }
        metadata["das"] = {
            "score_method": "density_sum",
            "protocol_formula": (
                "(1-s_v)+(1-s_l)+(1-s_vl)"
                if args.protocol == PROTOCOL_JOINT
                else "3*(1-s_l)"
            ),
            "layers": das_reports,
        }
        metadata["sparsity_allocation"] = allocation_report
        metadata["zero_report"] = target_zero_report(
            model,
            max_ar_layers=args.max_ar_layers,
        )
        if (
            metadata["zero_report"]["vision"]
            != metadata["pre_prune_zero_report"]["vision"]
        ):
            raise ProtocolError("Vision Linear weights changed during AR-only TAMP pruning")
        if (
            metadata["zero_report"]["ar"]["zeros"]
            <= metadata["pre_prune_zero_report"]["ar"]["zeros"]
        ):
            raise ProtocolError("No AR weights were pruned")
        metadata["achieved_ar_linear_sparsity"] = metadata["zero_report"]["ar"][
            "zero_ratio"
        ]

        if args.skip_final_forward:
            metadata["final_multimodal_verification"] = {"skipped": True}
        else:
            write_json(
                state_path,
                {
                    "phase": "verification",
                    "protocol": metadata["protocol"],
                    "started_unix": started_at,
                    "pid": os.getpid(),
                },
            )
            metadata["final_multimodal_verification"] = verify_full_multimodal_forward(
                model,
                processor,
                calibration.verification_records[0],
                device,
                args.max_length,
                args.enable_thinking,
                args.min_image_pixels,
                args.max_image_pixels,
            )

        metadata["whole_reasoner_zero_report"] = all_parameter_zero_report(model)
        metadata["peak_cuda_memory_bytes"] = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        )

        if args.save_masks:
            masks_path = output_dir / "tamp_masks.pt"
            torch.save(ar_masks, masks_path)
            metadata["mask_path"] = str(masks_path)
        if args.save_model:
            write_json(
                state_path,
                {
                    "phase": "saving",
                    "protocol": metadata["protocol"],
                    "started_unix": started_at,
                    "pid": os.getpid(),
                },
            )
            checkpoint_dir = output_dir / "checkpoint"
            model.save_pretrained(checkpoint_dir, safe_serialization=True, max_shard_size="4GB")
            processor.save_pretrained(checkpoint_dir)
            metadata["checkpoint_path"] = str(checkpoint_dir)
        else:
            metadata["checkpoint_path"] = None

        metadata["completed_unix"] = time.time()
        metadata["elapsed_seconds"] = metadata["completed_unix"] - started_at
        final_metadata_path = output_dir / "metadata.json"
        write_json(final_metadata_path, metadata)
        running_metadata = output_dir / "metadata.running.json"
        if running_metadata.exists():
            running_metadata.unlink()
        write_json(
            state_path,
            {
                "phase": "complete",
                "protocol": metadata["protocol"],
                "started_unix": started_at,
                "completed_unix": metadata["completed_unix"],
                "elapsed_seconds": metadata["elapsed_seconds"],
                "metadata_path": str(final_metadata_path),
                "checkpoint_path": metadata["checkpoint_path"],
                "pid": os.getpid(),
            },
        )
        print(json.dumps(metadata["zero_report"], indent=2), flush=True)
        print(f"Complete: {final_metadata_path}", flush=True)
        return metadata
    except Exception as exc:
        import traceback

        write_json(
            state_path,
            {
                "phase": "failed",
                "protocol": args.protocol,
                "started_unix": started_at,
                "failed_unix": time.time(),
                "pid": os.getpid(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def main() -> int:
    args = parse_args()
    apply_calibration_presets(args)
    validate_args(args)
    set_determinism(args.seed)
    if args.protocol == PROTOCOL_INSPECT:
        model, _processor = load_reasoner(args)
        audit = module_audit(model)
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return 0
    if args.preflight_only:
        calibration = build_protocol_calibration(args)
        report = {
            "protocol": args.protocol,
            "calibration_presets": args.calibration_preset,
            "ar_samples": len(calibration.ar_records),
            "verification_samples": len(calibration.verification_records),
            "pairing": calibration.pairing,
            "ar_sources": sorted({record.source_path for record in calibration.ar_records}),
            "verification_sources": sorted(
                {record.source_path for record in calibration.verification_records}
            ),
            "datasets": sorted({record.dataset for record in calibration.ar_records}),
            "verification_images_resolved": all(
                record.image_path is not None and Path(record.image_path).is_file()
                for record in calibration.verification_records
            ),
            "ar_has_no_images": all(record.image_path is None for record in calibration.ar_records)
            if args.protocol == PROTOCOL_SEPARATE else False,
            "first_pair": {
                "ar_sample_id": calibration.ar_records[0].sample_id,
                "verification_sample_id": calibration.verification_records[0].sample_id,
                "verification_image": calibration.verification_records[0].image_path,
                "text_equal": normalized_pair_text(calibration.verification_records[0].text)
                == normalized_pair_text(calibration.ar_records[0].text),
            },
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    run_pruning(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
