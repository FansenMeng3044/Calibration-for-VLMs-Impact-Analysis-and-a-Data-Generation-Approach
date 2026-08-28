"""Cosmos3-Edge adapter for lmms-eval 0.4.x generate-until tasks.

The adapter instantiates only ``Cosmos3EdgeForConditionalGeneration`` (the
Reasoner). It never loads or invokes the Cosmos generator/diffusion tower.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Optional, Tuple, Union

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Cosmos3EdgeForConditionalGeneration

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", "none", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[str(name).lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dtype {name!r}; choose bfloat16, float16, or float32"
        ) from exc


def _flatten_visuals(value: Any) -> list[Image.Image]:
    if value is None:
        return []
    if isinstance(value, Image.Image):
        return [value.convert("RGB")]
    if isinstance(value, (list, tuple)):
        flattened: list[Image.Image] = []
        for item in value:
            flattened.extend(_flatten_visuals(item))
        return flattened
    raise TypeError(
        f"Cosmos evaluation expects PIL images, got {type(value).__name__}"
    )


def _interleaved_content(
    context: str, visuals: list[Image.Image]
) -> list[dict[str, Any]]:
    """Preserve MMMU image-placeholder order and prepend images for VQA tasks."""

    normalized = re.sub(r"<image\s+\d+>", "<image>", context)
    placeholder_count = normalized.count("<image>")
    if placeholder_count == len(visuals) and placeholder_count > 0:
        parts = normalized.split("<image>")
        content: list[dict[str, Any]] = []
        for index, visual in enumerate(visuals):
            if parts[index]:
                content.append({"type": "text", "text": parts[index]})
            content.append({"type": "image", "image": visual})
        if parts[-1]:
            content.append({"type": "text", "text": parts[-1]})
        return content

    clean_context = normalized.replace("<image>", "").strip()
    return [
        *({"type": "image", "image": visual} for visual in visuals),
        {"type": "text", "text": clean_context},
    ]


def _stop_strings(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Iterable):
        return [str(value) for value in raw]
    raise TypeError(
        f"generation until must be a string or list, got {type(raw).__name__}"
    )


@register_model("cosmos3_edge")
class Cosmos3Edge(lmms):
    """Single-GPU Cosmos3-Edge Reasoner adapter for local image-text evals."""

    def __init__(
        self,
        pretrained: str = "/private/workspace/hycui/model/Cosmos3-Edge",
        device: str = "cuda:0",
        device_map: Optional[str] = None,
        batch_size: Union[int, str] = 1,
        dtype: str = "bfloat16",
        attn_implementation: str = "eager",
        max_length: int = 4096,
        min_image_pixels: int = 65_536,
        max_image_pixels: int = 1_048_576,
        use_cache: Any = True,
        enable_thinking: Any = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if kwargs:
            raise ValueError(
                f"Unexpected Cosmos3-Edge model arguments: {sorted(kwargs)}"
            )
        if int(batch_size) != 1:
            raise ValueError(
                "Cosmos3-Edge lmms adapter currently requires --batch_size 1"
            )
        if device_map not in (None, "", device, "cuda", "cuda:0"):
            raise ValueError(
                f"Only one explicit GPU is supported, got device_map={device_map!r}"
            )

        self._device = torch.device(device)
        self._rank = 0
        self._world_size = 1
        self.batch_size_per_gpu = 1
        self._max_length = int(max_length)
        self.min_image_pixels = int(min_image_pixels)
        self.max_image_pixels = int(max_image_pixels)
        if self.min_image_pixels <= 0 or self.min_image_pixels > self.max_image_pixels:
            raise ValueError(
                f"Invalid image pixel bounds: min={self.min_image_pixels}, max={self.max_image_pixels}"
            )
        self.use_cache = _as_bool(use_cache)
        self.enable_thinking = _as_bool(enable_thinking)

        self._model = Cosmos3EdgeForConditionalGeneration.from_pretrained(
            pretrained,
            dtype=_torch_dtype(dtype),
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
        )
        self._model.to(self._device)
        self._model.eval()
        self._model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(pretrained)
        self._tokenizer = self.processor.tokenizer
        self._config = self._model.config

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def loglikelihood(
        self, requests: List[Instance]
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError(
            "Cosmos3-Edge loglikelihood is not needed by the three generate-until evals"
        )

    def _generate_one(self, request: Instance) -> str:
        context, raw_gen_kwargs, doc_to_visual, doc_id, task, split = request.args
        document = self.task_dict[task][split][doc_id]
        visuals = _flatten_visuals(doc_to_visual(document))
        if not visuals:
            raise ValueError(
                f"{task}/{split}/{doc_id} has no image; this adapter is for image-text evals"
            )

        messages = [
            {
                "role": "user",
                "content": _interleaved_content(str(context), visuals),
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "images_kwargs": {
                    "size": {
                        "shortest_edge": self.min_image_pixels,
                        "longest_edge": self.max_image_pixels,
                    }
                },
                "text_kwargs": {
                    "truncation": True,
                    "max_length": self.max_length,
                },
            },
            enable_thinking=self.enable_thinking,
        )
        inputs = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }

        gen_kwargs = dict(raw_gen_kwargs or {})
        until = _stop_strings(gen_kwargs.pop("until", None))
        max_new_tokens = int(gen_kwargs.pop("max_new_tokens", 32))
        temperature = float(gen_kwargs.pop("temperature", 0.0) or 0.0)
        top_p = gen_kwargs.pop("top_p", None)
        num_beams = int(gen_kwargs.pop("num_beams", 1))
        requested_do_sample = _as_bool(
            gen_kwargs.pop("do_sample", temperature > 0.0)
        )
        do_sample = requested_do_sample and temperature > 0.0

        # These are meaningful to LLaVA adapters but not to Cosmos.
        gen_kwargs.pop("image_aspect_ratio", None)
        if gen_kwargs:
            raise ValueError(
                f"Unsupported Cosmos generation arguments: {sorted(gen_kwargs)}"
            )

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "use_cache": self.use_cache,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            if top_p is not None:
                generate_kwargs["top_p"] = float(top_p)

        # A new document must not inherit shape-dependent multimodal rotary
        # state from the preceding document.
        if hasattr(self.model, "model") and hasattr(
            self.model.model, "rope_deltas"
        ):
            self.model.model.rope_deltas = None

        prompt_tokens = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **generate_kwargs)
        continuation = generated[:, prompt_tokens:]
        answer = self.processor.batch_decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        for stop in until:
            if stop and stop in answer:
                answer = answer.split(stop, 1)[0].strip()
        return answer

    def generate_until(self, requests: List[Instance]) -> List[str]:
        answers: list[str] = []
        progress = tqdm(
            requests,
            disable=self.rank != 0,
            desc="Cosmos3-Edge responding",
        )
        for request in progress:
            answer = self._generate_one(request)
            answers.append(answer)
            self.cache_hook.add_partial(
                "generate_until", request.args[:2], answer
            )
        return answers

    def generate_until_multi_round(
        self, requests: List[Instance]
    ) -> List[str]:
        raise NotImplementedError(
            "The selected MMBench/MMMU/OKVQA tasks are single-round"
        )
