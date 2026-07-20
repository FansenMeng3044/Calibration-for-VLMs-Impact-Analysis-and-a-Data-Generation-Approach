"""
Unimodal calibration for BLIP-2 Wanda pruning: T5-only on text (e.g. C4), ViT-only on images (e.g. CC3M),
without full multimodal forward during calibration.

C4 text path default: full T5 seq2seq (encoder + decoder); optional encoder-only view for ablation.
ViT-only runs use full BLIP in evaluate_blip (no wrapper); importance_scope=vit_only limits LayerSparsity to visual_encoder.
"""

import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def _load_text_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError(f"empty list in {path}")
        if isinstance(data[0], str):
            return data
        if isinstance(data[0], dict):
            for key in ("text", "caption", "text_input", "output"):
                if key in data[0]:
                    return [row[key] for row in data]
            raise ValueError(f"list of dict in {path}: need a text field (text/caption/text_input/output)")
    raise ValueError(f"unsupported JSON in {path}")


class TextLinesDataset(Dataset):
    def __init__(self, texts, max_items=None):
        if max_items is not None:
            texts = texts[: int(max_items)]
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text_input": self.texts[idx]}


def collate_text_only(batch):
    return {"text_input": [b["text_input"] for b in batch]}


class T5EncoderTextOnlyView(nn.Module):
    """C4 text: T5 encoder only (no decoder forward). Legacy / ablation."""

    def __init__(self, blip_model):
        super().__init__()
        # Register the pruned submodule FIRST. named_parameters(remove_duplicate=True)
        # reports each shared Parameter under the first name it reaches, so registering
        # _blip first would hide every weight behind "_blip.t5_model.*" and the pruner
        # (which looks up "t5_model.encoder.block.<i>....weight") would find nothing.
        self.t5_model = blip_model.t5_model
        self._blip = blip_model
        self.t5_tokenizer = blip_model.t5_tokenizer
        self.max_txt_len = blip_model.max_txt_len

    def maybe_autocast(self, dtype=torch.float16):
        return self._blip.maybe_autocast(dtype=dtype)

    def forward(self, samples):
        texts = samples["text_input"]
        if isinstance(texts, str):
            texts = [texts]
        device = next(self.t5_model.parameters()).device
        input_tokens = self.t5_tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)
        inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
        bsz, seq_len = inputs_embeds.shape[:2]
        self.temp_label = torch.zeros((bsz, seq_len), dtype=torch.bool, device=inputs_embeds.device)
        self.temp_encoder_atts = input_tokens.attention_mask.detach()
        enc = self.t5_model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=input_tokens.attention_mask,
            return_dict=True,
        )
        # dict loss for LayerSparsity / loss_language
        return {"loss": enc.last_hidden_state.float().pow(2).mean()}


class T5Seq2SeqTextOnlyView(nn.Module):
    """
    C4 text: full T5 (encoder + decoder) with teacher-forced labels, same strings as
    input and target - matches `lavis.models.t5_models.t5.T5.forward` (no ViT/Q-Former).
    """

    def __init__(self, blip_model):
        super().__init__()
        # Register the pruned submodule FIRST. named_parameters(remove_duplicate=True)
        # reports each shared Parameter under the first name it reaches, so registering
        # _blip first would hide every weight behind "_blip.t5_model.*" and the pruner
        # (which looks up "t5_model.encoder.block.<i>....weight") would find nothing.
        self.t5_model = blip_model.t5_model
        self._blip = blip_model
        self.t5_tokenizer = blip_model.t5_tokenizer
        self.max_txt_len = blip_model.max_txt_len

    def maybe_autocast(self, dtype=torch.bfloat16):
        return self._blip.maybe_autocast(dtype=dtype)

    def forward(self, samples):
        texts = samples["text_input"]
        if isinstance(texts, str):
            texts = [texts]
        device = next(self.t5_model.parameters()).device
        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                texts,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)
            output_tokens = self.t5_tokenizer(
                texts,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)
            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            bsz, enc_len = inputs_embeds.shape[:2]
            # Blip2T5 convention: True = vision/query, False = language. C4 text-only: all language.
            self.temp_label = torch.zeros((bsz, enc_len), dtype=torch.bool, device=inputs_embeds.device)
            self.temp_encoder_atts = input_tokens.attention_mask.detach()
            out = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=input_tokens.attention_mask,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            return {"loss": out.loss, "logits": out.logits}


class VisualEncoderImageOnlyView(nn.Module):
    """ViT-only: same vision path as Blip2T5 (``ln_vision(visual_encoder(image))``), no Q-Former / T5."""

    def __init__(self, blip_model):
        super().__init__()
        # Same ordering requirement as the T5 views: the pruned submodule must be
        # registered first so named_parameters() reports "visual_encoder.*".
        self.visual_encoder = blip_model.visual_encoder
        self._blip = blip_model

    def maybe_autocast(self, dtype=torch.float16):
        return self._blip.maybe_autocast(dtype=dtype)

    def encode_image(self, image):
        # EVA VisionTransformer has ``forward`` only (no ``encode_image``); align with Blip2T5.
        with self.maybe_autocast():
            x = self.visual_encoder(image)
            if hasattr(self._blip, "ln_vision"):
                x = self._blip.ln_vision(x)
            return x

    def forward(self, samples):
        return self.encode_image(samples["image"])


def build_text_only_dataloader(c4_json_path, num_data, batch_size, distributed=False):
    if not os.path.isfile(c4_json_path):
        raise FileNotFoundError(f"c4 calib json not found: {c4_json_path}")
    texts = _load_text_lines(c4_json_path)
    ds = TextLinesDataset(texts, max_items=num_data)
    if num_data % batch_size != 0:
        raise ValueError(f"num_data ({num_data}) must be divisible by batch_size ({batch_size})")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, batch_size),
        collate_fn=collate_text_only,
    )
    return loader


def wrap_model_for_unimodal_prune(full_blip_model, mode, t5_c4_encoder_only=False):
    if mode == "t5_c4_text":
        if t5_c4_encoder_only:
            return T5EncoderTextOnlyView(full_blip_model), full_blip_model
        return T5Seq2SeqTextOnlyView(full_blip_model), full_blip_model
    if mode == "vit_image_only":
        return VisualEncoderImageOnlyView(full_blip_model), full_blip_model
    raise ValueError(mode)
