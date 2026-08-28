#!/usr/bin/env python3
"""Layer-wise WANDA pruning for the Cosmos3-Edge Reasoner.

This implementation intentionally supports exactly two pruning protocols:

* joint: image-only activations prune the vision encoder first, then a real
  image+text fused sequence prunes the AR tower (matching the existing LLaVA
  VIT-then-LLM execution order).
* separate: image-only activations prune the vision encoder, while a genuinely
  text-only tokenizer -> embedding -> AR forward prunes the AR tower.  No
  vision/projector output is present in the AR calibration path.

The diffusion/generator tower is never instantiated by this script.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, Cosmos3EdgeForConditionalGeneration


PROTOCOL_JOINT = "joint"
PROTOCOL_SEPARATE = "separate"
PROTOCOL_INSPECT = "inspect"

VISION_PREFIX = "model.visual.encoder.layers"
AR_PREFIX = "model.language_model.layers"
PROJECTOR_PREFIX = "model.projector"

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
    vision_records: list[CalibrationRecord]
    ar_records: list[CalibrationRecord]
    verification_records: list[CalibrationRecord]
    pairing: dict[str, Any]


@dataclasses.dataclass
class LayerSample:
    hidden_states: torch.Tensor
    layer_kwargs: dict[str, Any]
    valid_mask: torch.Tensor | None
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
            raise ProtocolError("WANDA statistic was requested before any activation samples were collected")
        return self.sum_sq / float(self.nsamples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=(PROTOCOL_INSPECT, PROTOCOL_JOINT, PROTOCOL_SEPARATE), required=True)
    parser.add_argument("--model-path", default="/private/workspace/hycui/model/Cosmos3-Edge")
    parser.add_argument("--calibration-json", action="append", default=[])
    parser.add_argument("--vision-calibration-json", action="append", default=[])
    parser.add_argument("--ar-calibration-json", action="append", default=[])
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
    parser.add_argument("--vision-sparsity", type=float, default=0.5)
    parser.add_argument("--ar-sparsity", type=float, default=0.5)
    parser.add_argument("--max-vision-layers", type=int, default=0)
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
    if args.calibration_json or args.vision_calibration_json or args.ar_calibration_json:
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
            args.vision_calibration_json.extend(
                str(value) for value in protocol_entry.get("vision_calibration_json", [])
            )
            args.ar_calibration_json.extend(
                str(value) for value in protocol_entry.get("ar_calibration_json", [])
            )


def validate_args(args: argparse.Namespace) -> None:
    for name in ("vision_sparsity", "ar_sparsity"):
        value = float(getattr(args, name))
        if not 0.0 <= value < 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1), got {value}")
        if args.protocol in {PROTOCOL_JOINT, PROTOCOL_SEPARATE} and value == 0.0:
            raise ValueError(
                f"Locked {args.protocol} Reasoner protocol requires both vision and AR pruning; "
                f"--{name.replace('_', '-')} cannot be zero"
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
        if args.vision_calibration_json or args.ar_calibration_json:
            raise ValueError("Joint protocol uses --calibration-json, not separate branch JSON arguments")
        if not args.calibration_json:
            raise ValueError("Joint protocol requires at least one --calibration-json or preset")
    elif args.protocol == PROTOCOL_SEPARATE:
        has_paired = bool(args.calibration_json)
        has_split = bool(args.vision_calibration_json or args.ar_calibration_json)
        if has_paired and has_split:
            raise ValueError("Separate protocol must use either paired JSON or explicit vision+AR JSON, not both")
        if has_split and not (args.vision_calibration_json and args.ar_calibration_json):
            raise ValueError("Separate protocol requires both --vision-calibration-json and --ar-calibration-json")
        if not has_paired and not has_split:
            raise ValueError("Separate protocol requires paired JSON or explicit vision+AR JSON")
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
            vision_records=records,
            ar_records=records,
            verification_records=records,
            pairing={"mode": "paired_multimodal", "pairs": len(records), "text_mismatches": 0},
        )

    if args.protocol != PROTOCOL_SEPARATE:
        raise ValueError(f"Calibration records are not used by protocol {args.protocol}")

    if args.calibration_json:
        records = build_calibration_records(
            paths=args.calibration_json,
            require_image=True,
            branch_name="separate_paired",
            **common_kwargs,
        )
        return ProtocolCalibration(
            vision_records=records,
            ar_records=records,
            verification_records=records,
            pairing={"mode": "derived_from_paired_multimodal", "pairs": len(records), "text_mismatches": 0},
        )

    vision_records = build_calibration_records(
        paths=args.vision_calibration_json,
        require_image=True,
        branch_name="separate_vision",
        **common_kwargs,
    )
    ar_records = build_calibration_records(
        paths=args.ar_calibration_json,
        require_image=False,
        branch_name="separate_ar",
        **common_kwargs,
    )
    if len(vision_records) != len(ar_records):
        raise ProtocolError(
            f"Separate branch length mismatch: vision={len(vision_records)}, AR={len(ar_records)}"
        )
    paired_vision: list[CalibrationRecord] = []
    paired_ar: list[CalibrationRecord] = []
    mismatches: list[dict[str, Any]] = []
    for index, (vision_record, ar_record) in enumerate(zip(vision_records, ar_records)):
        vision_text = normalized_pair_text(vision_record.text)
        ar_text = normalized_pair_text(ar_record.text)
        if vision_text != ar_text:
            mismatches.append(
                {
                    "index": index,
                    "vision_source": vision_record.source_path,
                    "ar_source": ar_record.source_path,
                    "vision_sample_id": vision_record.sample_id,
                    "ar_sample_id": ar_record.sample_id,
                }
            )
            continue
        pair_digest = hashlib.sha1(ar_text.encode("utf-8")).hexdigest()[:12]
        pair_id = f"{vision_record.dataset}:separate_pair:{index}:{pair_digest}"
        paired_vision.append(dataclasses.replace(vision_record, sample_id=pair_id))
        paired_ar.append(dataclasses.replace(ar_record, sample_id=pair_id))
    if mismatches:
        raise ProtocolError(
            f"Separate image/text calibration files are not aligned; "
            f"{len(mismatches)} text mismatches, first={mismatches[:3]}"
        )
    return ProtocolCalibration(
        vision_records=paired_vision,
        ar_records=paired_ar,
        verification_records=paired_vision,
        pairing={
            "mode": "explicit_image_only_plus_text_only",
            "pairs": len(paired_vision),
            "text_mismatches": 0,
            "vision_json": [str(Path(path).resolve()) for path in args.vision_calibration_json],
            "ar_json": [str(Path(path).resolve()) for path in args.ar_calibration_json],
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

    named_parameters = list(model.named_parameters())
    reasoner_parameter_count = count_parameters(parameter for _, parameter in named_parameters)
    vision_parameter_count = count_parameters(child.weight for child in vision_linears.values())
    ar_parameter_count = count_parameters(child.weight for child in ar_linears.values())
    projector_parameter_count = count_parameters(reasoner.projector.parameters())
    lm_head_parameter_count = count_parameters(model.lm_head.parameters())

    return {
        "model_class": type(model).__name__,
        "vision_layer_count": len(vision_layers),
        "ar_layer_count": len(ar_layers),
        "vision_linear_count": len(vision_linears),
        "ar_linear_count": len(ar_linears),
        "vision_linear_names": sorted(vision_linears),
        "ar_linear_names": sorted(ar_linears),
        "projector_linear_names": sorted(
            f"{PROJECTOR_PREFIX}.{name}" for name in find_linear_modules(reasoner.projector)
        ),
        "generator_modules": generator_names,
        "reasoner_parameter_count": reasoner_parameter_count,
        "vision_target_parameter_count": vision_parameter_count,
        "ar_target_parameter_count": ar_parameter_count,
        "all_target_parameter_count": vision_parameter_count + ar_parameter_count,
        "projector_parameter_count_dense": projector_parameter_count,
        "lm_head_parameter_count_dense": lm_head_parameter_count,
    }


def dtype_from_name(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_reasoner(args: argparse.Namespace) -> tuple[Cosmos3EdgeForConditionalGeneration, Any]:
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


def prepare_image_inputs(
    processor: Any,
    record: CalibrationRecord,
    min_image_pixels: int,
    max_image_pixels: int,
) -> dict[str, torch.Tensor]:
    if record.image_path is None:
        raise ProtocolError(f"Vision calibration record has no image: {record.sample_id}")
    with Image.open(record.image_path) as handle:
        image = handle.convert("RGB")
        batch = processor.image_processor(
            images=image,
            return_tensors="pt",
            **image_size_kwargs(min_image_pixels, max_image_pixels),
        )
    required = {"pixel_values", "image_grid_thw"}
    missing = required - set(batch)
    if missing:
        raise ProtocolError(f"Image processor did not return {sorted(missing)}")
    return {key: batch[key] for key in required}


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
                sample_id=str(state["sample_id"]),
            )
        )
        raise _CaptureStop()

    handle = layer.register_forward_pre_hook(catcher, with_kwargs=True)
    try:
        for record in records:
            state["sample_id"] = record.sample_id
            state["valid_mask"] = None
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
    return captured


def build_vision_cache(
    model: Cosmos3EdgeForConditionalGeneration,
    processor: Any,
    records: list[CalibrationRecord],
    device: torch.device,
    min_image_pixels: int,
    max_image_pixels: int,
) -> tuple[list[LayerSample], dict[str, Any]]:
    reasoner = model.model
    first_layer = reasoner.visual.encoder.layers[0]
    counters = CallCounter({"vision": reasoner.visual})
    patch_counts: list[int] = []

    def run_sample(record: CalibrationRecord, state: dict[str, Any]) -> None:
        batch = prepare_image_inputs(
            processor,
            record,
            min_image_pixels,
            max_image_pixels,
        )
        patch_counts.append(int(batch["pixel_values"].shape[0]))
        batch = move_batch(batch, device)
        state["valid_mask"] = None
        reasoner.visual(
            pixel_values=batch["pixel_values"],
            grid_thw=batch["image_grid_thw"],
            return_dict=True,
        )

    try:
        with forbidden_forward(
            {
                "projector": reasoner.projector,
                "ar_language_model": reasoner.language_model,
            }
        ):
            cache = capture_first_layer_inputs(first_layer, run_sample, records)
    finally:
        counters.close()

    if counters.counts["vision"] != len(records):
        raise ProtocolError(
            f"Vision-only path expected {len(records)} vision calls, got {counters.counts['vision']}"
        )
    return cache, {
        "vision_forward_calls": counters.counts["vision"],
        "projector_forward_calls": 0,
        "ar_forward_calls": 0,
        "patch_tokens_per_sample": patch_counts,
    }


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
    }


def _layer_output_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise ProtocolError(f"Unexpected layer output type: {type(output).__name__}")


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


def prune_layers_layerwise(
    layers: nn.ModuleList,
    cache: list[LayerSample],
    sparsity: float,
    prefix: str,
    device: torch.device,
    max_layers: int,
    save_masks: bool,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], list[LayerSample]]:
    limit = len(layers) if max_layers <= 0 else min(max_layers, len(layers))
    reports: list[dict[str, Any]] = []
    masks: dict[str, torch.Tensor] = {}

    for layer_index in range(limit):
        layer = layers[layer_index]
        linears = find_linear_modules(layer)
        if not linears:
            raise ProtocolError(f"No Linear modules in {prefix}.{layer_index}")
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
                for sample in cache:
                    hidden_states = sample.hidden_states.to(device=device, non_blocking=True)
                    kwargs = recursive_to_device(sample.layer_kwargs, device)
                    current_mask[0] = sample.valid_mask
                    _layer_output_tensor(layer(hidden_states, **kwargs))
                    del hidden_states, kwargs
        finally:
            for handle in handles:
                handle.remove()

        layer_report: dict[str, Any] = {
            "layer_index": layer_index,
            "layer_name": f"{prefix}.{layer_index}",
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
            f"zero_ratio={layer_report['actual_zero_ratio']:.6f}",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return reports, masks, cache


def target_zero_report(
    model: Cosmos3EdgeForConditionalGeneration,
    max_vision_layers: int = 0,
    max_ar_layers: int = 0,
) -> dict[str, Any]:
    groups: dict[str, list[nn.Linear]] = {"vision": [], "ar": []}
    vision_layers = model.model.visual.encoder.layers
    ar_layers = model.model.language_model.layers
    vision_limit = len(vision_layers) if max_vision_layers <= 0 else min(max_vision_layers, len(vision_layers))
    ar_limit = len(ar_layers) if max_ar_layers <= 0 else min(max_ar_layers, len(ar_layers))
    for layer in vision_layers[:vision_limit]:
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
    combined_parameters = report["vision"]["parameters"] + report["ar"]["parameters"]
    combined_zeros = report["vision"]["zeros"] + report["ar"]["zeros"]
    report["combined_target_linears"] = {
        "parameters": combined_parameters,
        "zeros": combined_zeros,
        "zero_ratio": combined_zeros / float(combined_parameters) if combined_parameters else 0.0,
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

        total_vision_layers = audit["vision_layer_count"]
        total_ar_layers = audit["ar_layer_count"]
        vision_limit = (
            total_vision_layers
            if args.max_vision_layers <= 0
            else min(args.max_vision_layers, total_vision_layers)
        )
        ar_limit = total_ar_layers if args.max_ar_layers <= 0 else min(args.max_ar_layers, total_ar_layers)
        partial_run = vision_limit < total_vision_layers or ar_limit < total_ar_layers
        if partial_run and args.save_model and not args.allow_partial_save:
            raise ProtocolError(
                "Partial-layer smoke runs cannot save a checkpoint unless --allow-partial-save is explicit"
            )

        metadata: dict[str, Any] = {
            "schema_version": 1,
            "protocol": (
                "cosmos_wanda_joint_reasoner"
                if args.protocol == PROTOCOL_JOINT
                else "cosmos_wanda_separate_reasoner"
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
            "nsamples": len(calibration.vision_records),
            "calibration_json": [str(Path(path).resolve()) for path in args.calibration_json],
            "vision_calibration_json": [
                str(Path(path).resolve()) for path in args.vision_calibration_json
            ],
            "ar_calibration_json": [str(Path(path).resolve()) for path in args.ar_calibration_json],
            "calibration_presets": list(args.calibration_preset),
            "calibration_pairing": calibration.pairing,
            "vision_calibration_sample_ids": [
                record.sample_id for record in calibration.vision_records
            ],
            "ar_calibration_sample_ids": [record.sample_id for record in calibration.ar_records],
            "vision_calibration_records": [
                {
                    "sample_id": record.sample_id,
                    "source_path": record.source_path,
                    "source_index": record.source_index,
                    "dataset": record.dataset,
                    "image_path": record.image_path,
                }
                for record in calibration.vision_records
            ],
            "ar_calibration_records": [
                {
                    "sample_id": record.sample_id,
                    "source_path": record.source_path,
                    "source_index": record.source_index,
                    "dataset": record.dataset,
                }
                for record in calibration.ar_records
            ],
            "vision_sparsity_target": args.vision_sparsity,
            "ar_sparsity_target": args.ar_sparsity,
            "wanda": {
                "metric": "abs(W[o,i]) * sqrt(mean_over_calibration_samples(sum_over_valid_tokens(x[...,i]^2)))",
                "normalization": "LLaVA-exact sample-normalized token-square-sum",
                "mask": "per-Linear, per-output-row bottom-k",
                "global_ranking": False,
                "structured_nm": False,
            },
            "execution_order": [
                "capture image-only vision activations",
                "prune vision layers sequentially",
                (
                    "capture fused image+text AR activations through the already-pruned vision encoder"
                    if args.protocol == PROTOCOL_JOINT
                    else "capture language-tokenizer-only AR activations with vision/projector forbidden"
                ),
                "prune AR layers sequentially",
                "verify full image+text Reasoner forward",
            ],
            "generator_excluded": True,
            "projector_dense": True,
            "embedding_norm_lm_head_dense": True,
            "partial_run": partial_run,
            "vision_layers_requested": vision_limit,
            "ar_layers_requested": ar_limit,
            "module_audit": audit,
            "started_unix": started_at,
        }
        write_json(output_dir / "metadata.running.json", metadata)
        write_json(
            state_path,
            {
                "phase": "vision_calibration",
                "protocol": metadata["protocol"],
                "started_unix": started_at,
                "pid": os.getpid(),
            },
        )

        vision_cache, vision_dataflow = build_vision_cache(
            model,
            processor,
            calibration.vision_records,
            device,
            args.min_image_pixels,
            args.max_image_pixels,
        )
        write_json(
            state_path,
            {
                "phase": "vision_pruning",
                "protocol": metadata["protocol"],
                "started_unix": started_at,
                "pid": os.getpid(),
            },
        )
        vision_reports, vision_masks, _ = prune_layers_layerwise(
            layers=model.model.visual.encoder.layers,
            cache=vision_cache,
            sparsity=args.vision_sparsity,
            prefix=VISION_PREFIX,
            device=device,
            max_layers=args.max_vision_layers,
            save_masks=args.save_masks,
        )
        del vision_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
                "phase": "ar_pruning",
                "protocol": metadata["protocol"],
                "started_unix": started_at,
                "pid": os.getpid(),
            },
        )
        ar_reports, ar_masks, _ = prune_layers_layerwise(
            layers=model.model.language_model.layers,
            cache=ar_cache,
            sparsity=args.ar_sparsity,
            prefix=AR_PREFIX,
            device=device,
            max_layers=args.max_ar_layers,
            save_masks=args.save_masks,
        )
        del ar_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        metadata["vision_dataflow"] = vision_dataflow
        metadata["ar_dataflow"] = ar_dataflow
        metadata["vision_pruning"] = {
            "summary": summarize_layer_reports(vision_reports),
            "layers": vision_reports,
        }
        metadata["ar_pruning"] = {
            "summary": summarize_layer_reports(ar_reports),
            "layers": ar_reports,
        }
        metadata["zero_report"] = target_zero_report(
            model,
            max_vision_layers=args.max_vision_layers,
            max_ar_layers=args.max_ar_layers,
        )
        if metadata["zero_report"]["vision"]["zeros"] <= 0:
            raise ProtocolError("No vision weights were pruned")
        if metadata["zero_report"]["ar"]["zeros"] <= 0:
            raise ProtocolError("No AR weights were pruned")

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
            masks_path = output_dir / "wanda_masks.pt"
            torch.save({**vision_masks, **ar_masks}, masks_path)
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
            "vision_samples": len(calibration.vision_records),
            "ar_samples": len(calibration.ar_records),
            "verification_samples": len(calibration.verification_records),
            "pairing": calibration.pairing,
            "vision_sources": sorted({record.source_path for record in calibration.vision_records}),
            "ar_sources": sorted({record.source_path for record in calibration.ar_records}),
            "datasets": sorted({record.dataset for record in calibration.vision_records}),
            "vision_images_resolved": all(
                record.image_path is not None and Path(record.image_path).is_file()
                for record in calibration.vision_records
            ),
            "ar_has_no_images": all(record.image_path is None for record in calibration.ar_records)
            if calibration.pairing["mode"] == "explicit_image_only_plus_text_only"
            else None,
            "first_pair": {
                "vision_sample_id": calibration.vision_records[0].sample_id,
                "ar_sample_id": calibration.ar_records[0].sample_id,
                "vision_image": calibration.vision_records[0].image_path,
                "text_equal": normalized_pair_text(calibration.vision_records[0].text)
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
