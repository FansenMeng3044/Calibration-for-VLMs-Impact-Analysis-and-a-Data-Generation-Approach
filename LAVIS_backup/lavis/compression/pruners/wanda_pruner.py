import math
import csv
import os
import torch
import torch.nn as nn

from time import time
from copy import deepcopy
from functools import partial

from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample
from lavis.models.blip2_models.blip2_t5 import Blip2T5
from lavis.models.t5_models.t5 import T5
from lavis.models.clip_models.eva_model import EVA_CLIP
from lavis.compression.pruners.utils import (
    loss_vision_language,
    loss_language,
    loss_vision,
    loss_vit_encode_l2,
    print_time,
)
from lavis.compression.pruners.layer_single_base_pruner import LayerWiseBasePruner, LayerSparsity


def _append_csv_rows(path, fieldnames, rows):
    if not path or not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def get_module_recursive(base, module_to_process):
    
    if module_to_process == "":
        return base
    
    splits = module_to_process.split(".")
    now = splits.pop(0)
    rest = ".".join(splits)
    
    base = getattr(base, now)

    return get_module_recursive(base, rest)


def _get_attention_flat(encoder_attention_masks, batch_index, sample_index, ref_mask):
    if encoder_attention_masks is None or batch_index >= len(encoder_attention_masks):
        return None
    attn_batch = encoder_attention_masks[batch_index]
    if attn_batch is None:
        return None
    if attn_batch.dim() == 1:
        attn_batch = attn_batch.unsqueeze(0)
    if sample_index >= attn_batch.shape[0]:
        return None
    attn_flat = attn_batch[sample_index].reshape(-1).to(ref_mask.device).bool()
    if attn_flat.numel() != ref_mask.numel():
        attn_flat = attn_flat[: ref_mask.numel()]
    if attn_flat.numel() != ref_mask.numel():
        return None
    return attn_flat


def _write_calibration_batch_trace(
    path,
    module_to_process,
    requested_samples,
    inps,
    image_masks,
    encoder_attention_masks,
    expected_query_tokens=32,
):
    if not path or image_masks is None:
        return
    rows = []
    physical_seen = 0
    total_batches = min(len(inps), len(image_masks))
    for batch_index in range(total_batches):
        inp = inps[batch_index]
        mask_batch = image_masks[batch_index].bool()
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        if mask_batch.dim() == 1:
            mask_batch = mask_batch.unsqueeze(0)
        if mask_batch.shape[0] != inp.shape[0] and mask_batch.numel() == inp.shape[0] * inp.shape[1]:
            mask_batch = mask_batch.reshape(inp.shape[0], inp.shape[1])

        batch_size = int(inp.shape[0])
        seq_len = int(inp.shape[1])
        query_prefix_counts = []
        total_query_counts = []
        valid_text_counts = []
        pad_text_counts = []
        query_prefix_ok_flags = []
        attention_ok_flags = []
        for sample_index in range(batch_size):
            mflat = mask_batch[sample_index].reshape(-1)
            if mflat.numel() != seq_len:
                mflat = mflat[:seq_len]
            query_prefix = mflat[:expected_query_tokens]
            text_suffix = mflat[expected_query_tokens:]
            query_prefix_true = int(query_prefix.sum().item())
            text_suffix_true = int(text_suffix.sum().item())
            total_query = int(mflat.sum().item())
            query_prefix_counts.append(query_prefix_true)
            total_query_counts.append(total_query)
            query_prefix_ok_flags.append(
                int(
                    mflat.numel() >= expected_query_tokens
                    and query_prefix_true == expected_query_tokens
                    and text_suffix_true == 0
                )
            )

            attn_flat = _get_attention_flat(encoder_attention_masks, batch_index, sample_index, mflat)
            if attn_flat is not None and attn_flat.numel() == mflat.numel():
                attention_query_true = int(attn_flat[:expected_query_tokens].sum().item())
                valid_text = int(attn_flat[expected_query_tokens:].sum().item())
                pad_text = int(attn_flat[expected_query_tokens:].numel() - valid_text)
                attention_ok = int(
                    attention_query_true == expected_query_tokens
                    and valid_text > 0
                    and valid_text + pad_text == int(text_suffix.numel())
                )
            else:
                valid_text = -1
                pad_text = -1
                attention_ok = 0
            valid_text_counts.append(valid_text)
            pad_text_counts.append(pad_text)
            attention_ok_flags.append(attention_ok)

        row = {
            "source": module_to_process,
            "requested_samples": int(requested_samples),
            "cached_batches": int(total_batches),
            "batch_index": int(batch_index),
            "batch_size": batch_size,
            "sample_start": int(physical_seen),
            "sample_end_exclusive": int(physical_seen + batch_size),
            "physical_samples_seen": int(physical_seen + batch_size),
            "seq_len": seq_len,
            "expected_query_tokens": int(expected_query_tokens),
            "query_prefix_true_min": int(min(query_prefix_counts)) if query_prefix_counts else 0,
            "query_prefix_true_max": int(max(query_prefix_counts)) if query_prefix_counts else 0,
            "total_query_tokens_min": int(min(total_query_counts)) if total_query_counts else 0,
            "total_query_tokens_max": int(max(total_query_counts)) if total_query_counts else 0,
            "valid_text_tokens_min": int(min(valid_text_counts)) if valid_text_counts else -1,
            "valid_text_tokens_max": int(max(valid_text_counts)) if valid_text_counts else -1,
            "pad_text_tokens_min": int(min(pad_text_counts)) if pad_text_counts else -1,
            "pad_text_tokens_max": int(max(pad_text_counts)) if pad_text_counts else -1,
            "query_prefix_ok_all": int(all(query_prefix_ok_flags)) if query_prefix_ok_flags else 0,
            "attention_layout_ok_all": int(all(attention_ok_flags)) if attention_ok_flags else 0,
        }
        rows.append(row)
        physical_seen += batch_size

    _append_csv_rows(
        path,
        [
            "source",
            "requested_samples",
            "cached_batches",
            "batch_index",
            "batch_size",
            "sample_start",
            "sample_end_exclusive",
            "physical_samples_seen",
            "seq_len",
            "expected_query_tokens",
            "query_prefix_true_min",
            "query_prefix_true_max",
            "total_query_tokens_min",
            "total_query_tokens_max",
            "valid_text_tokens_min",
            "valid_text_tokens_max",
            "pad_text_tokens_min",
            "pad_text_tokens_max",
            "query_prefix_ok_all",
            "attention_layout_ok_all",
        ],
        rows,
    )


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def _cos_pairwise_density_single(embeddings, image_mask, eps=1e-8):
    """Single-batch density for AMIA (mean v-v, l-l, v-l cosine similarities)."""
    with torch.no_grad():
        S, D = embeddings.shape
        emb = torch.nn.functional.normalize(embeddings.float(), dim=-1, eps=eps)
        mask = image_mask.to(embeddings.device).bool()
        v_idx = torch.where(mask)[0]
        l_idx = torch.where(~mask)[0]
        nv, nl = v_idx.numel(), l_idx.numel()
        v_sim, l_sim, vl_sim = 0.0, 0.0, 0.0
        if nv >= 2:
            v_emb = emb[v_idx]
            sim_vv = v_emb @ v_emb.T
            v_upper = sim_vv.triu(diagonal=1)
            v_vals = v_upper[v_upper > 0]
            v_sim = v_vals.mean().item() if v_vals.numel() > 0 else 0.0
        if nl >= 2:
            l_emb = emb[l_idx]
            sim_ll = l_emb @ l_emb.T
            l_upper = sim_ll.triu(diagonal=1)
            l_vals = l_upper[l_upper > 0]
            l_sim = l_vals.mean().item() if l_vals.numel() > 0 else 0.0
        if nv >= 1 and nl >= 1:
            vl_sim = (emb[v_idx] @ emb[l_idx].T).mean().item()
        return (v_sim + l_sim + vl_sim) / 3.0


class AdaptiveMultimodalInputActivation:
    """AMIA token selection: only selected tokens contribute to Wanda scaler_row."""

    def __init__(self, layer, layer_id=0, layer_name="none", keep_ratio=1.0, **kwargs):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0
        self.layer_id = layer_id
        self.layer_name = layer_name
        self.keep_ratio = float(keep_ratio)

    def _gaussian_rbf(self, X, Y, sigma=1.0):
        X_norm = (X ** 2).sum(dim=1).view(-1, 1)
        Y_norm = (Y ** 2).sum(dim=1).view(1, -1)
        pairwise_dists = X_norm + Y_norm - 2.0 * torch.mm(X, Y.T)
        return torch.exp(-pairwise_dists.clamp(min=0) / (2 * sigma ** 2))

    def _select_tokens(self, out, image_mask, score, eps=1e-8):
        N, D = out.shape
        out = torch.nn.functional.normalize(out.float(), dim=-1, eps=eps)
        if score is None:
            score = torch.ones(N, device=out.device, dtype=out.dtype)
        else:
            score = score.to(out.device).float().flatten()[:N]
            if score.numel() < N:
                score = torch.nn.functional.pad(score, (0, N - score.numel()), value=0.0)
        distances = 1.0 - torch.mm(out, out.T)
        distances = distances.clamp(min=0)
        num_neigh = min(3, N - 1)
        if num_neigh < 1:
            return torch.ones(N, dtype=torch.bool, device=out.device)
        knn_indices = torch.topk(distances, k=num_neigh + 1, largest=False).indices[:, 1:]
        neigh_dist = torch.exp(-torch.gather(distances, dim=1, index=knn_indices) * 1.0)
        neigh_scores = torch.gather(score.unsqueeze(0).expand(N, -1), dim=1, index=knn_indices)
        graph_score = score + (neigh_dist * neigh_scores).sum(dim=-1)
        K = self._gaussian_rbf(out, out)
        selected_indices = set()
        min_val = graph_score.min().item() - 1.0
        while True:
            idx = torch.argmax(graph_score).item()
            cur_score = graph_score[idx].item()
            selected_indices.add(idx)
            for si in selected_indices:
                if si < graph_score.shape[0]:
                    graph_score[si] = min_val
            neighbors = knn_indices[idx].tolist()
            for nb in neighbors:
                if nb < N:
                    dist_nb = distances[idx, nb].item()
                    decay = math.exp(-dist_nb * 0.2) * max(cur_score, 0.0)
                    graph_score[nb] = graph_score[nb].item() - decay
            if len(selected_indices) >= max(1, int(N * self.keep_ratio)):
                break
            temp_select = torch.tensor(list(selected_indices), device=out.device, dtype=torch.long)
            K_XX = K.mean().item()
            K_XY = K[:, temp_select].mean().item()
            K_YY = K[temp_select, :][:, temp_select].mean().item()
            MMD2 = K_XX + K_YY - 2.0 * K_XY
            try:
                density = _cos_pairwise_density_single(out, image_mask, eps=eps)
            except Exception:
                density = 0.5
            if MMD2 < (1.0 - density) ** 0.5 * 0.1:
                break
        score_mask = torch.zeros(N, dtype=torch.bool, device=out.device)
        for si in selected_indices:
            if si < N:
                score_mask[si] = True
        return score_mask

    def add_batch(self, inp, out, image_mask=None, score=None):
        if image_mask is None:
            raise RuntimeError(
                "token_selection='amia' requires image_masks. Enable return_image_masks in calibration."
            )
        out_tensor = out[0] if isinstance(out, (tuple, list)) else out
        inp_tensor = inp
        if out_tensor.dim() == 3:
            B, S, D = out_tensor.shape
            out_flat = out_tensor.reshape(-1, D)
        else:
            out_flat = out_tensor
            B, S, D = 1, out_flat.shape[0], out_flat.shape[1]
        if image_mask.dim() == 2:
            mask_flat = image_mask.reshape(-1)
        else:
            mask_flat = image_mask.flatten()
        if mask_flat.numel() != out_flat.shape[0]:
            mask_flat = mask_flat[: out_flat.shape[0]]
            if mask_flat.numel() < out_flat.shape[0]:
                mask_flat = torch.nn.functional.pad(
                    mask_flat.bool(), (0, out_flat.shape[0] - mask_flat.numel()), value=False
                )
        mask_flat = mask_flat.to(out_flat.device).bool()
        score_flat = None
        if score is not None:
            s = score.flatten()
            if s.numel() >= out_flat.shape[0]:
                score_flat = s[: out_flat.shape[0]]
        score_mask = self._select_tokens(out_flat, mask_flat, score_flat)
        if inp_tensor.dim() == 2:
            inp_tensor = inp_tensor.unsqueeze(0)
        inp_flat = inp_tensor.reshape(-1, inp_tensor.shape[-1])
        if inp_flat.shape[0] != score_mask.numel():
            score_mask = score_mask[: inp_flat.shape[0]]
        inp_selected = inp_flat[score_mask]
        if inp_selected.numel() == 0:
            return
        if isinstance(self.layer, nn.Linear):
            inp_selected = inp_selected.t()
        tmp = inp_selected.shape[1]
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp_selected = inp_selected.type(torch.float32)
        self.scaler_row += torch.norm(inp_selected, p=2, dim=1) ** 2 / self.nsamples


class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out, image_mask=None, score=None):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples


class WrappedATV:
    """ATV-Pruning token selection (CVPR 2026, "Mostly Text, Smart Visuals").

    Wanda scaler_row is accumulated over all valid text tokens + only the selected
    salient visual/query tokens. `selected_idxs` indexes into per-sample image/query
    positions and is precomputed in T5LayerWandaPruner._prune from block input->output
    cosine distance. The normalization follows the original WrappedATV: one update
    unit is one physical sample, not one token.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0
        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out, image_mask=None, score=None, selected_idxs=None, attention_mask=None):
        if image_mask is None:
            raise RuntimeError(
                "token_selection='atv' requires image_masks (multimodal calib + temp_label)."
            )
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        if image_mask.dim() == 1:
            image_mask = image_mask.unsqueeze(0)
        mask = image_mask.to(inp.device).bool()
        if mask.shape[0] != inp.shape[0] and mask.numel() == inp.shape[0] * inp.shape[1]:
            mask = mask.reshape(inp.shape[0], inp.shape[1])
        attn = None
        if attention_mask is not None:
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            attn = attention_mask.to(inp.device).bool()
            if attn.shape[0] != inp.shape[0] and attn.numel() == inp.shape[0] * inp.shape[1]:
                attn = attn.reshape(inp.shape[0], inp.shape[1])

        kept_parts = []
        active_samples = 0
        batch_size = inp.shape[0]
        for b in range(batch_size):
            inp_b = inp[b]
            mask_b = mask[b].reshape(-1)
            if mask_b.numel() != inp_b.shape[0]:
                mask_b = mask_b[: inp_b.shape[0]]
                inp_b = inp_b[: mask_b.numel()]
            attn_b = None
            if attn is not None and b < attn.shape[0]:
                attn_b = attn[b].reshape(-1)
                if attn_b.numel() != mask_b.numel():
                    attn_b = attn_b[: mask_b.numel()]

            text_mask = ~mask_b
            if attn_b is not None and attn_b.numel() == mask_b.numel():
                text_mask = text_mask & attn_b

            text_tokens = inp_b[text_mask]               # keep all valid text tokens per sample
            img_tokens = inp_b[mask_b]                   # all visual/query tokens for this sample
            sample_parts = []
            if text_tokens.numel() > 0:
                sample_parts.append(text_tokens)

            if selected_idxs is None:
                sel = None
            elif isinstance(selected_idxs, (list, tuple)):
                sel = selected_idxs[b] if b < len(selected_idxs) else None
            else:
                sel = selected_idxs if batch_size == 1 else None

            if sel is not None and sel.numel() > 0:
                sel = sel.to(img_tokens.device).long()
                sel = sel[sel < img_tokens.shape[0]]
                if sel.numel() > 0:
                    sample_parts.append(img_tokens[sel])  # + selected salient query tokens

            if sample_parts:
                kept_parts.append(torch.cat(sample_parts, dim=0))
                active_samples += 1

        kept_parts = [part for part in kept_parts if part.numel() > 0]
        if not kept_parts or active_samples == 0:
            return
        kept = torch.cat(kept_parts, dim=0)
        if kept.shape[0] == 0:
            return
        if isinstance(self.layer, nn.Linear):
            kept = kept.t()
        kept = kept.type(torch.float32)
        tmp = active_samples
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        self.scaler_row += torch.norm(kept, p=2, dim=1) ** 2 / self.nsamples


@registry.register_pruner("t5_wanda_pruner")
class T5LayerWandaPruner(LayerWiseBasePruner):
    pruner_name = "t5_wanda_pruner"
    def __init__(
        self,
        model,
        data_loader,
        prune_spec=None,
        importance_scores_cache=None,
        keep_indices_or_masks_cache=None,
        is_strct_pruning=False,
        num_samples=64,
        is_global=False,
        model_prefix="t5_model",
        sparsity_ratio_granularity=None,
        max_sparsity_per_layer=0.8,
        score_method="GradMagSquare_avg",
        num_data_first_stage=128,
        num_noise=1,
        sparsity_dict=None,
        noise_eps=1e-3,
        prune_per_model=False,
        **kwargs,
    ):
        super().__init__(
            model=model,
            data_loader=data_loader,
            prune_spec=prune_spec,
            is_strct_pruning=is_strct_pruning,
            importance_scores_cache=importance_scores_cache,
            keep_indices_or_masks_cache=keep_indices_or_masks_cache,
            is_global=is_global,
            num_samples=num_samples,
            model_prefix=model_prefix,
            sparsity_ratio_granularity=sparsity_ratio_granularity,
            max_sparsity_per_layer=max_sparsity_per_layer,
            score_method=score_method,
            num_data_first_stage=num_data_first_stage,
            num_noise=num_noise,
            sparsity_dict=sparsity_dict,
            noise_eps=noise_eps,
            prune_per_model=prune_per_model,
        )
        
        self.loss_func = loss_language

    def reweighting_after_pruning(self, original_weights, keep_masks):
        raise NotImplementedError

    def read_cache(self, cache_file):
        raise NotImplementedError
    
    def check_sparsity(self, model, module_to_process="encoder.block"):
        use_cache = getattr(model, self.model_prefix).config.use_cache 
        getattr(model, self.model_prefix).config.use_cache = False 

        layers = get_module_recursive(model, module_to_process)
        count = 0 
        total_params = 0
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            sub_count = 0
            sub_params = 0
            for name in subset:
                W = subset[name].weight.data
                count += (W==0).sum().item()
                total_params += W.numel()

                sub_count += (W==0).sum().item()
                sub_params += W.numel()

            print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

        getattr(model, self.model_prefix).config.use_cache = use_cache 
        return float(count)/total_params 
    
    def forward_to_cache(self, model, batch):
        return model(batch)
    
    def prepare_calibration_input_encoder(
        self,
        model,
        dataloader,
        device,
        model_prefix,
        n_samples,
        module_to_process="encoder.block",
        return_image_masks=False,
    ):
        use_cache = getattr(model, model_prefix).config.use_cache
        getattr(model, model_prefix).config.use_cache = False

        layers = get_module_recursive(model, module_to_process)

        inps = []
        caches = []
        image_masks = [] if return_image_masks else None
        encoder_attention_masks = [] if return_image_masks else None

        keys_to_cache = [
            "attention_mask",
            "position_bias",
            "encoder_attention_mask",
            "encoder_decoder_position_bias",
            "layer_head_mask",
            "cross_attn_layer_head_mask",
            "encoder_hidden_states",
        ]

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps.append(inp.detach())
                cache = {}
                for k in keys_to_cache:
                    cache[k] = kwargs[k]
                caches.append(cache)
                raise ValueError

        layers[0] = Catcher(layers[0])
        total_samples = 0
        for i, batch in enumerate(dataloader):
            if total_samples >= n_samples:
                break
            if batch.get("image") is not None:
                bs = batch["image"].shape[0]
            elif "text_input" in batch:
                ti = batch["text_input"]
                bs = len(ti) if isinstance(ti, list) else int(ti.shape[0])
            else:
                raise ValueError(
                    "calibration batch must contain 'image' (multimodal) or 'text_input' (T5 text-only)."
                )
            total_samples += bs
            try:
                self.forward_to_cache(model, batch)
            except ValueError:
                if return_image_masks:
                    if not hasattr(model, "temp_label"):
                        raise RuntimeError(
                            "model.temp_label not found; required for AMIA / DAS (set in Blip2T5.forward)."
                        )
                    with torch.no_grad():
                        mask = model.temp_label.detach()
                    if mask.dtype != torch.bool:
                        mask = mask.bool()
                    image_masks.append(mask.cpu())
                    if hasattr(model, "temp_encoder_atts"):
                        encoder_attention_masks.append(model.temp_encoder_atts.detach().cpu())
        layers[0] = layers[0].module

        outs = [None] * len(inps)

        getattr(model, model_prefix).config.use_cache = use_cache

        if return_image_masks:
            assert len(image_masks) == len(inps), (
                "image_masks vs inps length mismatch: %d vs %d" % (len(image_masks), len(inps))
            )
            for i in range(len(inps)):
                B, S = inps[i].shape[0], inps[i].shape[1]
                assert image_masks[i].shape == (B, S), (
                    "image_masks[%d].shape %s vs inps[%d] (%d,%d)"
                    % (i, image_masks[i].shape, i, B, S)
                )
            self._cached_encoder_attention_masks = encoder_attention_masks
            return inps, outs, caches, image_masks
        return inps, outs, caches

    @print_time
    def _prune(
        self,
        model,
        dataloader,
        device,
        model_prefix,
        module_to_process="encoder.block",
        n_samples=64,
        sparsity_ratio=0.5,
        token_selection="naive",
        image_masks=None,
        scores=None,
        cached_calib=None,
        alpha=1.0,
    ):
        use_cache = getattr(model, model_prefix).config.use_cache
        getattr(model, model_prefix).config.use_cache = False

        return_image_masks = token_selection in ("amia", "atv")
        print("loading calibdation data")
        with torch.no_grad():
            if cached_calib is not None:
                result = cached_calib
            else:
                result = self.prepare_calibration_input_encoder(
                    model,
                    dataloader,
                    device,
                    model_prefix,
                    n_samples,
                    module_to_process,
                    return_image_masks=return_image_masks,
                )
            inps, outs, caches = result[0], result[1], result[2]
            image_masks = result[3] if len(result) == 4 else image_masks
            encoder_attention_masks = getattr(self, "_cached_encoder_attention_masks", None)

        if token_selection in ("amia", "atv") and (image_masks is None or len(image_masks) == 0):
            raise RuntimeError(
                f"token_selection='{token_selection}' requires image_masks (multimodal calib + temp_label)."
            )

        requested_physical_samples = n_samples
        n_samples = min(n_samples, len(inps))
        batch_offsets = []
        sample_offset = 0
        for j in range(n_samples):
            batch_offsets.append(sample_offset)
            sample_offset += int(inps[j].shape[0]) if inps[j].dim() == 3 else 1
        atv_diag_dir = (
            os.environ.get("LAVIS_ATV_DIAGNOSTIC_DIR")
            or os.environ.get("ATV_DIAGNOSTIC_DIR")
        )
        atv_token_csv = None
        atv_query_csv = None
        atv_importance_csv = None
        atv_calib_trace_csv = None
        if atv_diag_dir and token_selection == "atv":
            os.makedirs(atv_diag_dir, exist_ok=True)
            atv_token_csv = os.path.join(atv_diag_dir, "token_mask_integrity.csv")
            atv_query_csv = os.path.join(atv_diag_dir, "selected_query_frequency.csv")
            atv_importance_csv = os.path.join(atv_diag_dir, "importance_distribution.csv")
            atv_calib_trace_csv = os.path.join(atv_diag_dir, "calibration_batch_trace.csv")
            _write_calibration_batch_trace(
                atv_calib_trace_csv,
                module_to_process,
                requested_physical_samples,
                inps[:n_samples],
                image_masks[:n_samples] if image_masks is not None else None,
                encoder_attention_masks[:n_samples] if encoder_attention_masks is not None else None,
            )

        layers = get_module_recursive(model, module_to_process)
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            # ---- ATV pre-pass: pick salient visual (query) tokens per sample ----
            # salience = 1 - cos(block input, block output) over image/query positions;
            # k = round(min(1, alpha * avg_cosdist) * #text_tokens), clamped to #img_tokens.
            selected_idxs_atv = None
            if token_selection == "atv":
                if image_masks is None:
                    raise RuntimeError("token_selection='atv' requires image_masks (multimodal calib).")
                cos_dist_list = []
                all_cos_dist = []
                for j in range(n_samples):
                    with torch.no_grad():
                        with model.maybe_autocast(dtype=torch.bfloat16):
                            out_j = layer(inps[j], **caches[j])[0]
                    inp_batch = inps[j] if inps[j].dim() == 3 else inps[j].unsqueeze(0)
                    out_batch = out_j if out_j.dim() == 3 else out_j.unsqueeze(0)
                    mask_batch = image_masks[j].to(inps[j].device).bool()
                    if mask_batch.dim() == 1:
                        mask_batch = mask_batch.unsqueeze(0)
                    if mask_batch.shape[0] != inp_batch.shape[0] and mask_batch.numel() == inp_batch.shape[0] * inp_batch.shape[1]:
                        mask_batch = mask_batch.reshape(inp_batch.shape[0], inp_batch.shape[1])

                    batch_cos = []
                    for b in range(inp_batch.shape[0]):
                        mask_b = mask_batch[b].reshape(-1)
                        inp_b = inp_batch[b]
                        out_b = out_batch[b]
                        if mask_b.numel() != inp_b.shape[0]:
                            mask_b = mask_b[: inp_b.shape[0]]
                            inp_b = inp_b[: mask_b.numel()]
                            out_b = out_b[: mask_b.numel()]
                        img_in = inp_b.float()[mask_b]
                        img_out = out_b.float()[mask_b]
                        if img_in.numel() == 0:
                            cos_dist = inp_b.new_zeros(0, dtype=torch.float32)
                        else:
                            cos = torch.nn.functional.cosine_similarity(img_in, img_out, dim=-1)
                            cos_dist = (1.0 - cos).detach()
                        batch_cos.append(cos_dist)
                        if cos_dist.numel() > 0:
                            all_cos_dist.append(cos_dist)
                    cos_dist_list.append(batch_cos)
                cat = torch.cat(all_cos_dist) if len(all_cos_dist) else inps[0].new_zeros(1)
                cos_dist_avg = cat.mean().item() if cat.numel() else 0.0
                selected_idxs_atv = [None] * n_samples
                ks = []
                num_imgs = []
                for j in range(n_samples):
                    mask_batch = image_masks[j].bool()
                    if mask_batch.dim() == 1:
                        mask_batch = mask_batch.unsqueeze(0)
                    if mask_batch.shape[0] != len(cos_dist_list[j]) and mask_batch.numel() == len(cos_dist_list[j]) * mask_batch.shape[-1]:
                        mask_batch = mask_batch.reshape(len(cos_dist_list[j]), -1)
                    selected_idxs_atv[j] = []
                    for b, cos_dist in enumerate(cos_dist_list[j]):
                        mflat = mask_batch[b].reshape(-1)
                        attn_flat = _get_attention_flat(encoder_attention_masks, j, b, mflat)
                        if attn_flat is not None:
                            num_text = int(((~mflat) & attn_flat).sum().item())
                        else:
                            num_text = int((~mflat).sum().item())
                        num_img = int(cos_dist.numel())
                        k = int(round(min(1.0, alpha * cos_dist_avg) * num_text))
                        k = max(0, min(num_img, k))
                        if k > 0:
                            selected = torch.topk(cos_dist, k=k).indices.sort().values
                        else:
                            selected = torch.empty(0, dtype=torch.long, device=cos_dist.device)
                        selected_idxs_atv[j].append(selected)
                        ks.append(k)
                        num_imgs.append(num_img)
                _nimg = int(num_imgs[0]) if num_imgs else 0
                _hit = sum(1 for k, nimg in zip(ks, num_imgs) if k >= nimg and nimg > 0)
                _zero = sum(1 for k in ks if k == 0)
                _ntotal = len(ks)
                print(
                    f"[ATV] {module_to_process} layer {i}: cos_dist_avg={cos_dist_avg:.4f} "
                    f"k mean/min/max={sum(ks)/max(1,len(ks)):.1f}/{min(ks) if ks else 0}/{max(ks) if ks else 0} "
                    f"k==0={_zero}/{_ntotal} "
                    f"num_img={_nimg} k==num_img(degenerate)={_hit}/{_ntotal}"
                )
                if atv_token_csv:
                    token_rows = []
                    query_rows = []
                    for j in range(n_samples):
                        mask_batch = image_masks[j].bool()
                        if mask_batch.dim() == 1:
                            mask_batch = mask_batch.unsqueeze(0)
                        if mask_batch.shape[0] != len(selected_idxs_atv[j]) and mask_batch.numel() == len(selected_idxs_atv[j]) * mask_batch.shape[-1]:
                            mask_batch = mask_batch.reshape(len(selected_idxs_atv[j]), -1)
                        for b, selected in enumerate(selected_idxs_atv[j]):
                            mflat = mask_batch[b].reshape(-1)
                            expected_query = 32
                            query_prefix = mflat[:expected_query]
                            text_suffix = mflat[expected_query:]
                            attn_flat = _get_attention_flat(encoder_attention_masks, j, b, mflat)
                            if attn_flat is not None and attn_flat.numel() == mflat.numel():
                                attention_query_true = int(attn_flat[:expected_query].sum().item())
                                valid_text_tokens = int(attn_flat[expected_query:].sum().item())
                                pad_text_tokens = int(attn_flat[expected_query:].numel() - valid_text_tokens)
                                attention_layout_ok = int(
                                    attention_query_true == expected_query
                                    and 0 <= valid_text_tokens <= int(text_suffix.numel())
                                )
                            else:
                                attention_query_true = -1
                                valid_text_tokens = -1
                                pad_text_tokens = -1
                                attention_layout_ok = 0
                            num_query = int(mflat.sum().item())
                            num_text = int((~mflat).sum().item())
                            query_prefix_true = int(query_prefix.sum().item())
                            text_suffix_true = int(text_suffix.sum().item())
                            query_prefix_ok = int(
                                mflat.numel() >= expected_query
                                and query_prefix_true == expected_query
                                and text_suffix_true == 0
                            )
                            selected_cpu = selected.detach().cpu().tolist() if selected is not None else []
                            k = len(selected_cpu)
                            sample_id = batch_offsets[j] + b
                            token_rows.append(
                                {
                                    "sample_id": sample_id,
                                    "layer": i,
                                    "seq_len": int(mflat.numel()),
                                    "expected_query_tokens": expected_query,
                                    "num_query_tokens": num_query,
                                    "num_text_tokens": num_text,
                                    "query_prefix_true_count": query_prefix_true,
                                    "text_suffix_true_count": text_suffix_true,
                                    "attention_query_true_count": attention_query_true,
                                    "valid_text_tokens": valid_text_tokens,
                                    "pad_text_tokens": pad_text_tokens,
                                    "attention_layout_ok": attention_layout_ok,
                                    "query_prefix_ok": query_prefix_ok,
                                    "selected_k": k,
                                    "selected_ratio": k / max(1, num_query),
                                    "is_degenerate": int(k >= num_query and num_query > 0),
                                    "source": module_to_process,
                                }
                            )
                            for idx in selected_cpu:
                                query_rows.append(
                                    {
                                        "layer": i,
                                        "sample_id": sample_id,
                                        "query_index": int(idx),
                                        "count": 1,
                                        "source": module_to_process,
                                    }
                                )
                    _append_csv_rows(
                        atv_token_csv,
                        [
                            "sample_id",
                            "layer",
                            "seq_len",
                            "expected_query_tokens",
                            "num_query_tokens",
                            "num_text_tokens",
                            "query_prefix_true_count",
                            "text_suffix_true_count",
                            "attention_query_true_count",
                            "valid_text_tokens",
                            "pad_text_tokens",
                            "attention_layout_ok",
                            "query_prefix_ok",
                            "selected_k",
                            "selected_ratio",
                            "is_degenerate",
                            "source",
                        ],
                        token_rows,
                    )
                    _append_csv_rows(
                        atv_query_csv,
                        ["layer", "sample_id", "query_index", "count", "source"],
                        query_rows,
                    )

            wrapped_layers = {}
            for name in subset:
                if token_selection == "amia":
                    wrapped_layers[name] = AdaptiveMultimodalInputActivation(subset[name])
                elif token_selection == "atv":
                    wrapped_layers[name] = WrappedATV(subset[name])
                else:
                    wrapped_layers[name] = WrappedGPT(subset[name])

            def add_batch(name, j_ref):
                def tmp(_, inp, out):
                    out_tensor = out[0] if isinstance(out, (tuple, list)) else out
                    inp_data = inp[0].data
                    mask_j = image_masks[j_ref] if image_masks is not None else None
                    score_j = scores[j_ref] if scores is not None else None
                    if token_selection == "atv":
                        sel_j = selected_idxs_atv[j_ref] if selected_idxs_atv is not None else None
                        attn_j = (
                            encoder_attention_masks[j_ref]
                            if encoder_attention_masks is not None and j_ref < len(encoder_attention_masks)
                            else None
                        )
                        wrapped_layers[name].add_batch(
                            inp_data,
                            out_tensor.data,
                            mask_j,
                            score_j,
                            sel_j,
                            attention_mask=attn_j,
                        )
                    else:
                        wrapped_layers[name].add_batch(inp_data, out_tensor.data, mask_j, score_j)

                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name, 0)))

            for j in range(n_samples):
                if j > 0:
                    for h in handles:
                        h.remove()
                    handles = []
                    for name in wrapped_layers:
                        handles.append(subset[name].register_forward_hook(add_batch(name, j)))
                with torch.no_grad():
                    with model.maybe_autocast(dtype=torch.bfloat16):
                        outs[j] = layer(inps[j], **caches[j])[0]
            for h in handles:
                h.remove()

            for name in subset:
                if token_selection == "naive":
                    assert wrapped_layers[name].nsamples == len(inps) * inps[0].shape[0]
                else:
                    assert wrapped_layers[name].nsamples > 0
                print(f"pruning layer {i} name {name}")
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                    wrapped_layers[name].scaler_row.reshape((1, -1))
                )

                W_mask = torch.zeros_like(W_metric) == 1
                if self.prune_n != 0:
                    for ii in range(W_metric.shape[1]):
                        if ii % self.prune_m == 0:
                            tmp = W_metric[:, ii : (ii + self.prune_m)].float()
                            W_mask.scatter_(
                                1,
                                ii
                                + torch.topk(tmp, self.prune_n, dim=1, largest=False)[1],
                                True,
                            )
                else:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    sparsity_key = f"{module_to_process}.{i}.{name}.weight"
                    indices = sort_res[1][
                        :, : int(W_metric.shape[1] * sparsity_ratio[sparsity_key])
                    ]
                    W_mask.scatter_(1, indices, True)

                if atv_importance_csv and token_selection == "atv":
                    wm = W_metric.detach().float()
                    scaler = wrapped_layers[name].scaler_row.detach().float()
                    weight_abs = subset[name].weight.data.detach().float().abs()
                    _append_csv_rows(
                        atv_importance_csv,
                        [
                            "source",
                            "layer",
                            "module",
                            "tensor",
                            "numel",
                            "mean_wanda_importance",
                            "median_wanda_importance",
                            "max_wanda_importance",
                            "scaler_row_mean",
                            "scaler_row_max",
                            "weight_abs_mean",
                            "mask_sparsity",
                        ],
                        [
                            {
                                "source": module_to_process,
                                "layer": i,
                                "module": name,
                                "tensor": f"{module_to_process}.{i}.{name}.weight",
                                "numel": int(wm.numel()),
                                "mean_wanda_importance": wm.mean().item(),
                                "median_wanda_importance": wm.median().item(),
                                "max_wanda_importance": wm.max().item(),
                                "scaler_row_mean": scaler.mean().item(),
                                "scaler_row_max": scaler.max().item(),
                                "weight_abs_mean": weight_abs.mean().item(),
                                "mask_sparsity": W_mask.float().mean().item(),
                            }
                        ],
                    )

                subset[name].weight.data[W_mask] = 0

            for j in range(n_samples):
                with torch.no_grad():
                    with model.maybe_autocast(dtype=torch.bfloat16):
                        outs[j] = layer(inps[j], **caches[j])[0]
            inps, outs = outs, inps

        getattr(model, model_prefix).config.use_cache = use_cache
        torch.cuda.empty_cache()

        return model
    
    def get_sparsity(self, original_sparsity, sparsity_ratio_granularity=None):
        if self.sparsity_dict is not None:
            import yaml
            with open(self.sparsity_dict, "r") as f:
                return yaml.load(f, Loader=yaml.FullLoader)

        if sparsity_ratio_granularity == None:
            layer_to_group_mapping = {}
        
        else:
            def check(name, v):
                if len(v.shape) == 2 and \
                        ".block" in name and \
                            "relative_attention_bias.weight" not in name and \
                                name.startswith(self.model_prefix):
                    return True
                return False
            parameters_to_prune = [
                k for k, v in self.model.named_parameters() if check(k, v)
            ]

            if sparsity_ratio_granularity == "layer":
                layer_to_group_mapping = {
                    k: k
                    for k in parameters_to_prune
                }
            elif sparsity_ratio_granularity == "block":
                layer_to_group_mapping = {
                    k: ".".join(k.split(".")[:4])
                    for k in parameters_to_prune
                }
            else:
                raise NotImplementedError
        
        sparsity_module = LayerSparsity(
            self.model, 
            self.data_loader, 
            loss_language, 
            self.num_data_first_stage,
            original_sparsity,
            self.max_sparsity_per_layer,
            self.score_method,
            self.num_noise,
            self.noise_eps,
            layer_to_group_mapping,
            prune_per_model=self.prune_per_model,
        )
        
        return sparsity_module.return_sparsity()
        
    @print_time
    def prune(self, importance_scores=None, keep_indices_or_masks=None):
        print("In: ", self.pruner_name)
        dtype_record, requires_grad_record, device = self.model_setup_and_record_attributes(self.model)

        if self.prune_spec is None:
            return self.model, None

        _, keep_ratio, _, _ = self.convert_spec_to_list(self.prune_spec)
        
        sparsity_ratio = 1 - keep_ratio
        
        sparsity_dict = self.get_sparsity(
            sparsity_ratio,
            sparsity_ratio_granularity=self.sparsity_ratio_granularity
        )
        
        self.model = self._prune(
            self.model, self.data_loader, device, 
            model_prefix=self.model_prefix,
            module_to_process=f"{self.model_prefix}.encoder.block",
            n_samples=self.num_samples, sparsity_ratio=sparsity_dict,
        )
        self.model = self._prune(
            self.model, self.data_loader, device, 
            model_prefix=self.model_prefix,
            module_to_process=f"{self.model_prefix}.decoder.block",
            n_samples=self.num_samples, sparsity_ratio=sparsity_dict,
        )

        # let the pruned model has the original
        self.model_reset(self.model, dtype_record, requires_grad_record, device)
        
        return self.model, sparsity_dict


@registry.register_pruner("vit_wanda_pruner")
class VITLayerWandaPruner(LayerWiseBasePruner):
    pruner_name = "vit_wanda_pruner"
    def __init__(
        self,
        model,
        data_loader,
        prune_spec=None,
        importance_scores_cache=None,
        keep_indices_or_masks_cache=None,
        is_strct_pruning=False,
        num_samples=64,
        is_global=False,
        model_prefix="visual",
        sparsity_ratio_granularity=None,
        max_sparsity_per_layer=0.8,
        score_method="GradMagSquare_avg",
        num_data_first_stage=128,
        num_noise=1,
        sparsity_dict=None,
        noise_eps=1e-3,
        prune_per_model=False,
        **kwargs,
    ):
        super().__init__(
            model=model,
            data_loader=data_loader,
            prune_spec=prune_spec,
            is_strct_pruning=is_strct_pruning,
            importance_scores_cache=importance_scores_cache,
            keep_indices_or_masks_cache=keep_indices_or_masks_cache,
            is_global=is_global,
            num_samples=num_samples,
            model_prefix=model_prefix,
            sparsity_ratio_granularity=sparsity_ratio_granularity,
            max_sparsity_per_layer=max_sparsity_per_layer,
            score_method=score_method,
            num_data_first_stage=num_data_first_stage,
            num_noise=num_noise,
            sparsity_dict=sparsity_dict,
            noise_eps=noise_eps,
            prune_per_model=prune_per_model,
        )
        
        self.loss_func = loss_vision

    def reweighting_after_pruning(self, original_weights, keep_masks):
        raise NotImplementedError

    def read_cache(self, cache_file):
        raise NotImplementedError
    
    def check_sparsity(self, model, module_to_process="encoder.block"):
        layers = get_module_recursive(model, module_to_process)
        count = 0 
        total_params = 0
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            sub_count = 0
            sub_params = 0
            for name in subset:
                W = subset[name].weight.data
                count += (W==0).sum().item()
                total_params += W.numel()

                sub_count += (W==0).sum().item()
                sub_params += W.numel()

            print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

        return float(count)/total_params 
    
    def forward_to_cache(self, model, batch):
        return model.encode_image(batch["image"])
    
    def prepare_calibration_input_encoder(self, model, dataloader, device, model_prefix, n_samples, module_to_process="encoder.block"):
        layers = get_module_recursive(model, module_to_process)

        dtype = next(iter(model.parameters())).dtype
        inps = []
        
        print(dtype)
        
        caches = []
        
        keys_to_cache = [
            "rel_pos_bias"
        ]

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            def forward(self, inp, rel_pos_bias):
                inps.append(inp)
                inps[-1].requires_grad = False
                
                cache = {}
                cache["rel_pos_bias"] = rel_pos_bias
                caches.append(cache)
                raise ValueError

        layers[0] = Catcher(layers[0])
        
        total_samples = 0
        for i, batch in enumerate(dataloader):
            if total_samples >= n_samples:
                break
            total_samples += batch["image"].shape[0]
            try:
                self.forward_to_cache(model, batch)
            except ValueError:
                pass 
        layers[0] = layers[0].module

        outs = [None] * len(inps)

        return inps, outs, caches
    
    @print_time
    def _prune(self, model, dataloader, device, model_prefix, module_to_process="encoder.block", n_samples=64, sparsity_ratio=0.5):
        print("loading calibdation data")
        with torch.no_grad():
            inps, outs, caches = self.prepare_calibration_input_encoder(model, dataloader, device, model_prefix, n_samples, module_to_process)

        n_samples = min(n_samples, len(inps))

        layers = get_module_recursive(model, module_to_process)
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            # if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            #     dev = model.hf_device_map[f"model.layers.{i}"]
            #     inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    # print(inp[0].data.shape)
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(n_samples):
                with torch.no_grad():
                    with model.maybe_autocast():
                        outs[j] = layer(inps[j], **caches[j])

            for h in handles:
                h.remove()

            for name in subset:
                assert wrapped_layers[name].nsamples == len(inps) * inps[0].shape[0]
                print(f"pruning layer {i} name {name}")
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
                
                # setattr(subset[name].weight, "importance_score", W_metric.cpu().abs().mean().item())
                
                W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                if self.prune_n != 0:
                    # structured n:m sparsity
                    for ii in range(W_metric.shape[1]):
                        if ii % self.prune_m == 0:
                            tmp = W_metric[:,ii:(ii+self.prune_m)].float()
                            W_mask.scatter_(1,ii+torch.topk(tmp, self.prune_n, dim=1, largest=False)[1], True)
                else:
                    sparsity_key = f"{module_to_process}.{i}.{name}.weight"
                    
                    thres = torch.sort(W_metric.flatten())[0][int(W_metric.numel() * sparsity_ratio[sparsity_key])]
                    W_mask = (W_metric <= thres)

                subset[name].weight.data[W_mask] = 0  ## set weights to zero 

            for j in range(n_samples):
                with torch.no_grad():
                    with model.maybe_autocast():
                        outs[j] = layer(inps[j], **caches[j])
            inps, outs = outs, inps

        torch.cuda.empty_cache()

        return model
    
    def get_sparsity(self, original_sparsity, sparsity_ratio_granularity=None):
        if self.sparsity_dict is not None:
            import yaml
            with open(self.sparsity_dict, "r") as f:
                sparsity_dict = yaml.load(f, Loader=yaml.FullLoader)
                
            sparsity_dict = {k.replace("visual_encoder.", "visual."): v for k, v in sparsity_dict.items()}
            
            if "visual.blocks.39.attn.qkv.weight" not in sparsity_dict:
                # get from multi-modal pruning
                sparsity_dict["visual.blocks.39.attn.qkv.weight"] = 0
                sparsity_dict["visual.blocks.39.attn.proj.weight"] = 0
                sparsity_dict["visual.blocks.39.mlp.fc1.weight"] = 0
                sparsity_dict["visual.blocks.39.mlp.fc2.weight"] = 0
            
            return sparsity_dict

        if sparsity_ratio_granularity == None:
            layer_to_group_mapping = {}
        
        else:
            def check(name, v):
                if len(v.shape) == 2 and \
                        ".blocks" in name and \
                            name.startswith(self.model_prefix):
                    return True
                return False
            parameters_to_prune = [
                k for k, v in self.model.named_parameters() if check(k, v)
            ]

            if sparsity_ratio_granularity == "layer":
                layer_to_group_mapping = {
                    k: k
                    for k in parameters_to_prune
                }
            elif sparsity_ratio_granularity == "block":
                layer_to_group_mapping = {
                    k: ".".join(k.split(".")[:3])
                    for k in parameters_to_prune
                }
            else:
                raise NotImplementedError
        
        sparsity_module = LayerSparsity(
            self.model, 
            self.data_loader, 
            loss_vision, 
            self.num_data_first_stage,
            original_sparsity,
            self.max_sparsity_per_layer,
            self.score_method,
            self.num_noise,
            self.noise_eps,
            layer_to_group_mapping,
            prune_per_model=self.prune_per_model,
        )
        
        return sparsity_module.return_sparsity()

    @print_time
    def prune(self, importance_scores=None, keep_indices_or_masks=None):
        print("In: ", self.pruner_name)
        dtype_record, requires_grad_record, device = self.model_setup_and_record_attributes(self.model)

        if self.prune_spec is None:
            return self.model, None

        _, keep_ratio, _, _ = self.convert_spec_to_list(self.prune_spec)
        
        sparsity_ratio = 1 - keep_ratio
        
        sparsity_dict = self.get_sparsity(
            sparsity_ratio,
            sparsity_ratio_granularity=self.sparsity_ratio_granularity
        )
        
        self.model = self._prune(
            self.model, self.data_loader, device, 
            model_prefix=self.model_prefix,
            module_to_process=f"{self.model_prefix}.blocks",
            n_samples=self.num_samples, sparsity_ratio=sparsity_dict,
        )

        # let the pruned model has the original
        self.model_reset(self.model, dtype_record, requires_grad_record, device)
        
        return self.model, sparsity_dict


@registry.register_pruner("blipt5_wanda_pruner")
class BLIPT5LayerWandaPruner(LayerWiseBasePruner):
    pruner_name = "blipt5_wanda_pruner"
    def __init__(
        self,
        model,
        data_loader,
        t5_prune_spec=None,
        vit_prune_spec=None,
        t5_pruning_method=None,
        vit_pruning_method=None,
        t5_importance_scores_cache=None,
        t5_keep_indices_or_masks_cache=None,
        vit_importance_scores_cache=None,
        vit_keep_indices_or_masks_cache=None,
        importance_scores_cache=None,
        keep_indices_or_masks_cache=None,
        is_strct_pruning=False,
        num_samples=64,
        is_global=False,
        t5_model_prefix="t5_model",
        vit_model_prefix="visual_encoder",
        sparsity_ratio_granularity=None,
        max_sparsity_per_layer=0.8,
        score_method="GradMagSquare_avg",
        num_data_first_stage=128,
        num_noise=1,
        sparsity_dict=None,
        noise_eps=1e-3,
        prune_per_model=False,
        token_selection="naive",
        alpha=1.0,
        prune_t5=True,
        prune_vit=True,
        t5_unimodal_text_skip_decoder=False,
        importance_scope="joint",
        **kwargs,
    ):
        super().__init__(
            model=model,
            data_loader=data_loader,
            prune_spec=None,
            is_strct_pruning=is_strct_pruning,
            importance_scores_cache=importance_scores_cache,
            keep_indices_or_masks_cache=keep_indices_or_masks_cache,
            is_global=is_global,
            num_samples=num_samples,
            model_prefix="tmp",
            sparsity_ratio_granularity=sparsity_ratio_granularity,
            max_sparsity_per_layer=max_sparsity_per_layer,
            score_method=score_method,
            num_data_first_stage=num_data_first_stage,
            num_noise=num_noise,
            sparsity_dict=sparsity_dict,
            noise_eps=noise_eps,
            prune_per_model=prune_per_model,
        )
        
        self.t5_prune_spec = t5_prune_spec
        self.vit_prune_spec = vit_prune_spec
        
        assert t5_pruning_method is not None
        assert vit_pruning_method is not None
        
        self.t5_model_prefix = t5_model_prefix
        self.vit_model_prefix = vit_model_prefix
        self.token_selection = token_selection
        self.alpha = alpha
        self.prune_t5 = prune_t5
        self.prune_vit = prune_vit
        self.t5_unimodal_text_skip_decoder = t5_unimodal_text_skip_decoder
        assert importance_scope in (
            "joint",
            "llm_only",
            "vit_only",
            "vit_only_encode",
        )
        self.importance_scope = importance_scope

    def get_sparsity(self, original_sparsity, sparsity_ratio_granularity=None):
        if self.sparsity_dict is not None:
            import yaml
            with open(self.sparsity_dict, "r") as f:
                return yaml.load(f, Loader=yaml.FullLoader)

        if sparsity_ratio_granularity == None:
            layer_to_group_mapping = {}
        
        else:
            def check(name, v):
                if len(v.shape) != 2 or ".block" not in name:
                    return False
                if "relative_attention_bias.weight" in name:
                    return False
                if self.importance_scope == "llm_only":
                    return name.startswith(self.t5_model_prefix)
                if self.importance_scope in ("vit_only", "vit_only_encode"):
                    return name.startswith(self.vit_model_prefix)
                return name.startswith(self.t5_model_prefix) or name.startswith(
                    self.vit_model_prefix
                )

            parameters_to_prune = [
                k for k, v in self.model.named_parameters() if check(k, v)
            ]

            if sparsity_ratio_granularity == "model":
                
                def return_group(name):
                    if name.startswith(self.t5_model_prefix):
                        return self.t5_model_prefix
                    elif name.startswith(self.vit_model_prefix):
                        return self.vit_model_prefix
                    else:
                        return "other"
                
                layer_to_group_mapping = {
                    k: return_group(k)
                    for k in parameters_to_prune
                }
                
            elif sparsity_ratio_granularity == "layer":
                layer_to_group_mapping = {
                    k: k
                    for k in parameters_to_prune
                }
            elif sparsity_ratio_granularity == "block":
                def return_group(name):
                    if name.startswith(self.t5_model_prefix):
                        return ".".join(name.split(".")[:4])
                    elif name.startswith(self.vit_model_prefix):
                        return ".".join(name.split(".")[:3])
                    else:
                        return "other"
                layer_to_group_mapping = {
                    k: return_group(k)
                    for k in parameters_to_prune
                }
            else:
                raise NotImplementedError

        if self.importance_scope == "llm_only":
            loss_fn = loss_language
            per_model_group = [self.t5_model_prefix]
        elif self.importance_scope == "vit_only":
            # Multimodal CE; groups / budget only over ViT blocks.
            loss_fn = loss_vision_language
            per_model_group = [self.vit_model_prefix]
        elif self.importance_scope == "vit_only_encode":
            # Pure-ViT forward: normalized encode_image features, mean square (surrogate).
            loss_fn = loss_vit_encode_l2
            per_model_group = [self.vit_model_prefix]
        else:
            loss_fn = loss_vision_language
            per_model_group = [self.t5_model_prefix, self.vit_model_prefix]

        calibration_fn = None
        if self.score_method == "density_sum" and self.importance_scope in ("joint", "llm_only"):
            def calibration_fn(model, data_loader, device):
                if getattr(self, "_cached_encoder_calib", None) is not None:
                    return self._cached_encoder_calib
                return T5LayerWandaPruner.prepare_calibration_input_encoder(
                    self,
                    model,
                    data_loader,
                    device,
                    self.t5_model_prefix,
                    self.num_data_first_stage,
                    module_to_process=f"{self.t5_model_prefix}.encoder.block",
                    return_image_masks=True,
                )

        sparsity_module = LayerSparsity(
            self.model, 
            self.data_loader, 
            loss_fn, 
            self.num_data_first_stage,
            original_sparsity,
            self.max_sparsity_per_layer,
            self.score_method,
            self.num_noise,
            self.noise_eps,
            layer_to_group_mapping,
            prune_per_model=self.prune_per_model,
            per_model_group=per_model_group,
            calibration_fn=calibration_fn,
        )
        
        return sparsity_module.return_sparsity()
        
    def forward_to_cache(self, model, batch):
        return model(batch)

    @print_time
    def prune(self, importance_scores=None, keep_indices_or_masks=None):
        print("In: ", self.pruner_name)
        dtype_record, requires_grad_record, device = self.model_setup_and_record_attributes(self.model)

        if not self.prune_vit and not self.prune_t5:
            raise ValueError("At least one of prune_vit or prune_t5 must be True")

        self._cached_encoder_calib = None
        need_calib = self.prune_t5 and (
            (self.sparsity_ratio_granularity is not None and self.score_method == "density_sum")
            or self.token_selection in ("amia", "atv")
        )
        if need_calib and self.t5_prune_spec is not None:
            self.prepare_calibration_input_encoder = partial(
                T5LayerWandaPruner.prepare_calibration_input_encoder,
                self,
            )
            calib_result = self.prepare_calibration_input_encoder(
                self.model,
                self.data_loader,
                device,
                self.t5_model_prefix,
                self.num_data_first_stage,
                module_to_process=f"{self.t5_model_prefix}.encoder.block",
                return_image_masks=True,
            )
            if self.token_selection in ("amia", "atv") and len(calib_result) != 4:
                raise RuntimeError(
                    f"token_selection='{self.token_selection}' requires image_masks from calibration and temp_label on Blip2T5."
                )
            self._cached_encoder_calib = calib_result

        global_sparsity_dict = None
        if self.sparsity_ratio_granularity is not None:
            vit_kr = None
            t5_kr = None
            if self.vit_prune_spec is not None:
                _, vit_kr, _, _ = self.convert_spec_to_list(self.vit_prune_spec)
            if self.t5_prune_spec is not None:
                _, t5_kr, _, _ = self.convert_spec_to_list(self.t5_prune_spec)

            if self.prune_vit and self.prune_t5:
                assert vit_kr is not None and t5_kr is not None, (
                    "vit_prune_spec and t5_prune_spec required when pruning both under joint allocation"
                )
                assert vit_kr == t5_kr
                budget_kr = vit_kr
            elif self.prune_vit:
                budget_kr = vit_kr if vit_kr is not None else t5_kr
                assert budget_kr is not None, (
                    "vit_prune_spec required when prune_vit with sparsity_ratio_granularity set "
                    "(or pass t5_prune_spec to reuse its keep ratio for the global budget)."
                )
            else:
                # T5-only joint allocation: prefer t5_prune_spec; many YAMLs only define vit_prune_spec.
                budget_kr = t5_kr if t5_kr is not None else vit_kr
                assert budget_kr is not None, (
                    "Pass --t5_prune_spec (e.g. 24-0.5-1.0-1.0) when pruning T5 with "
                    "sparsity_ratio_granularity set, or add --vit_prune_spec with the same keep ratio "
                    "to reuse it for the budget, or use --sparsity_dict / set granularity to None."
                )

            global_sparsity_dict = self.get_sparsity(
                1 - budget_kr,
                sparsity_ratio_granularity=self.sparsity_ratio_granularity,
            )

        if self.prune_vit:
            if global_sparsity_dict is not None:
                sparsity_dict = global_sparsity_dict
            else:
                assert self.vit_prune_spec is not None, (
                    "vit_prune_spec is required for ViT Wanda when no joint sparsity_dict was built "
                    "(e.g. sparsity_ratio_granularity is None)."
                )
                _, keep_ratio, _, _ = self.convert_spec_to_list(self.vit_prune_spec)
                sparsity_ratio = 1 - keep_ratio
                sparsity_dict = self.get_sparsity(
                    sparsity_ratio,
                    sparsity_ratio_granularity=None
                )
            
            _vit_prune = partial(VITLayerWandaPruner._prune, self)
            self.prepare_calibration_input_encoder = partial(
                VITLayerWandaPruner.prepare_calibration_input_encoder,
                self,
                )
            
            self.model = _vit_prune(
                self.model, self.data_loader, device, 
                model_prefix=self.vit_model_prefix,
                module_to_process=f"{self.vit_model_prefix}.blocks",
                n_samples=self.num_samples, sparsity_ratio=sparsity_dict,
            )
            
        if self.prune_t5:
            if global_sparsity_dict is not None:
                sparsity_dict = global_sparsity_dict
            else:
                assert self.t5_prune_spec is not None, (
                    "t5_prune_spec is required for T5 Wanda when no joint sparsity_dict was built "
                    "(e.g. sparsity_ratio_granularity is None)."
                )
                _, keep_ratio, _, _ = self.convert_spec_to_list(self.t5_prune_spec)
                sparsity_ratio = 1 - keep_ratio
                sparsity_dict = self.get_sparsity(
                    sparsity_ratio,
                    sparsity_ratio_granularity=None
                )
            
            _t5_prune = partial(T5LayerWandaPruner._prune, self)
            self.prepare_calibration_input_encoder = partial(
                T5LayerWandaPruner.prepare_calibration_input_encoder,
                self,
                )
            
            self.model = _t5_prune(
                self.model, self.data_loader, device, 
                model_prefix=self.t5_model_prefix,
                module_to_process=f"{self.t5_model_prefix}.encoder.block",
                n_samples=self.num_samples, sparsity_ratio=sparsity_dict,
                token_selection=self.token_selection,
                cached_calib=self._cached_encoder_calib if need_calib else None,
                alpha=self.alpha,
            )
            
            if not self.t5_unimodal_text_skip_decoder:
                self.model = _t5_prune(
                    self.model, self.data_loader, device, 
                    model_prefix=self.t5_model_prefix,
                    module_to_process=f"{self.t5_model_prefix}.decoder.block",
                    n_samples=self.num_samples, sparsity_ratio=sparsity_dict,
                    token_selection="naive",
                )

        self._cached_encoder_calib = None
        # let the pruned model has the original
        self.model_reset(self.model, dtype_record, requires_grad_record, device)
        
        return self.model, global_sparsity_dict
