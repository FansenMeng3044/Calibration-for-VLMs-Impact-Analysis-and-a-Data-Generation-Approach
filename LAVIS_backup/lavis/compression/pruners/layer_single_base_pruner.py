import torch
import torch.nn as nn
import numpy as np
import collections
import contextlib
import json
import os

from time import time
from copy import deepcopy
from functools import partial

from lavis.datasets.data_utils import prepare_sample
from lavis.models.blip2_models.blip2_t5 import Blip2T5
from lavis.models.t5_models.t5 import T5
from lavis.models.clip_models.eva_model import EVA_CLIP
from lavis.compression.pruners.utils import (
    loss_vision_language, loss_language, loss_vision, print_time
)
from lavis.compression.pruners.base_pruner import BasePruner


def _get_module_by_path(model, path):
    """Get nested module by dot path, e.g. 't5_model.encoder.block' -> model.t5_model.encoder.block."""
    obj = model
    for name in path.split("."):
        obj = getattr(obj, name)
    return obj


def find_layers(module, layers=(nn.Linear,), name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for child_name, child in module.named_children():
        full_name = child_name if name == "" else name + "." + child_name
        res.update(find_layers(child, layers=layers, name=full_name))
    return res


def _align_bool_vector(mask, length, device, fill_value=False):
    if mask is None:
        return None
    mask = mask.reshape(-1).to(device).bool()
    if mask.numel() > length:
        mask = mask[:length]
    elif mask.numel() < length:
        pad = torch.full(
            (length - mask.numel(),),
            bool(fill_value),
            dtype=torch.bool,
            device=device,
        )
        mask = torch.cat([mask, pad], dim=0)
    return mask


def _normal_t5_block_forward(layer, hidden_states, cache, output_attentions=False):
    kwargs = dict(cache)
    if output_attentions:
        kwargs["output_attentions"] = True
    outputs = layer(hidden_states, **kwargs)
    if not isinstance(outputs, (tuple, list)):
        return outputs, dict(cache)
    uses_cache = bool(kwargs.get("use_cache", False))
    bias_offset = 2 if uses_cache else 1
    next_cache = dict(cache)
    if len(outputs) > bias_offset and outputs[bias_offset] is not None:
        next_cache["position_bias"] = outputs[bias_offset].detach()
    if next_cache.get("encoder_hidden_states") is not None:
        cross_bias_index = bias_offset + 2 if output_attentions else bias_offset + 1
        if len(outputs) > cross_bias_index and outputs[cross_bias_index] is not None:
            next_cache["encoder_decoder_position_bias"] = outputs[cross_bias_index].detach()
    return outputs[0], next_cache


def cos_pairwise_density(
    embeddings, image_mask, attention_mask=None, eps=1e-8, stats=None, return_counts=False
):
    """
    Vision-vision, language-language, and vision-language mean cosine similarities (TAMP / DAS).

    `stats` is an optional Counter for observability only. When a term cannot be
    measured (too few tokens of a modality, or the >0 filter leaves nothing) the
    corresponding similarity stays 0.0, which downstream reads as "maximally
    diverse". Counting how often that happens tells us whether any DAS number is
    contaminated by not-measured values masquerading as measured zeros.
    Passing stats=None keeps the numerical behaviour byte-identical.

    With return_counts=True the per-term "defined sample" counts are returned as
    well, and each similarity is averaged over the samples where it was actually
    measurable instead of over all B samples. When every term is defined for every
    sample -- the normal multimodal case, where nv is always the 32 query tokens --
    the two are the same division and the result is bit-identical. Run the
    LAVIS_DAS_DIAGNOSTIC audit to confirm that holds for your calibration sets.
    """
    with torch.no_grad():
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
            image_mask = image_mask.unsqueeze(0)

        B, S, D = embeddings.shape
        device = embeddings.device

        if image_mask.dim() == 1:
            image_mask = image_mask.unsqueeze(0).expand(B, -1)
        image_mask = image_mask.to(device=device)

        embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1, eps=eps)

        v_sims = []
        l_sims = []
        vl_sims = []
        n_v_defined = n_l_defined = n_vl_defined = 0

        for b in range(B):
            emb = embeddings[b]
            mask = _align_bool_vector(image_mask[b], S, device, fill_value=False)
            valid = None
            if attention_mask is not None:
                if attention_mask.dim() == 1:
                    attn_b = attention_mask
                else:
                    attn_b = attention_mask[b]
                valid = _align_bool_vector(attn_b, S, device, fill_value=False)
            if valid is None:
                # No attention mask reached us: PAD positions would be treated as
                # real language tokens. Should never happen on the DAS/AMIA paths.
                if stats is not None:
                    stats["no_attention_mask"] += 1
                valid = torch.ones(S, dtype=torch.bool, device=device)
            mask = mask & valid
            v_idx = torch.where(mask)[0]
            l_idx = torch.where((~mask) & valid)[0]
            nv = v_idx.numel()
            nl = l_idx.numel()

            v_mean_sim_b = 0.0
            l_mean_sim_b = 0.0
            vl_mean_sim_b = 0.0

            if stats is not None:
                stats["samples"] += 1
                stats["n_visual_total"] += int(nv)
                stats["n_language_total"] += int(nl)
                stats["n_pad_total"] += int(S - int(valid.sum().item()))
                if nv < 2:
                    stats["v_undefined_too_few"] += 1
                if nl < 2:
                    stats["l_undefined_too_few"] += 1
                if nv < 1 or nl < 1:
                    stats["vl_undefined_too_few"] += 1

            if nv >= 2:
                v_emb = emb[v_idx]
                sim_vv = v_emb @ v_emb.T
                v_upper = sim_vv.triu(diagonal=1)
                v_vals = v_upper[v_upper > 0]
                if stats is not None:
                    stats["v_pairs_total"] += int(nv * (nv - 1) // 2)
                    stats["v_pairs_kept_positive"] += int(v_vals.numel())
                    if v_vals.numel() == 0:
                        stats["v_empty_after_positive_filter"] += 1
                v_mean_sim_b = v_vals.mean().item() if v_vals.numel() > 0 else 0.0
            if nl >= 2:
                l_emb = emb[l_idx]
                sim_ll = l_emb @ l_emb.T
                l_upper = sim_ll.triu(diagonal=1)
                l_vals = l_upper[l_upper > 0]
                if stats is not None:
                    stats["l_pairs_total"] += int(nl * (nl - 1) // 2)
                    stats["l_pairs_kept_positive"] += int(l_vals.numel())
                    if l_vals.numel() == 0:
                        stats["l_empty_after_positive_filter"] += 1
                l_mean_sim_b = l_vals.mean().item() if l_vals.numel() > 0 else 0.0
            if nv >= 1 and nl >= 1:
                v_emb = emb[v_idx]
                l_emb = emb[l_idx]
                sim_vl = (v_emb @ l_emb.T).reshape(-1)
                vl_mean_sim_b = sim_vl.mean().item()

            v_sims.append(v_mean_sim_b)
            l_sims.append(l_mean_sim_b)
            vl_sims.append(vl_mean_sim_b)
            # A term is "defined" for this sample when the modality pair exists at
            # all. The >0 filter leaving an empty set is a separate condition,
            # tracked by the audit counters but still treated as a measurement.
            n_v_defined += int(nv >= 2)
            n_l_defined += int(nl >= 2)
            n_vl_defined += int(nv >= 1 and nl >= 1)

        if not return_counts:
            v_mean_sim = sum(v_sims) / B
            l_mean_sim = sum(l_sims) / B
            vl_mean_sim = sum(vl_sims) / B
            return float(v_mean_sim), float(l_mean_sim), float(vl_mean_sim)

        v_mean_sim = sum(v_sims) / n_v_defined if n_v_defined else 0.0
        l_mean_sim = sum(l_sims) / n_l_defined if n_l_defined else 0.0
        vl_mean_sim = sum(vl_sims) / n_vl_defined if n_vl_defined else 0.0
        return (
            float(v_mean_sim), float(l_mean_sim), float(vl_mean_sim),
            n_v_defined, n_l_defined, n_vl_defined,
        )


class ActivationDensity:
    """Accumulates DAS density stats over encoder layer forwards."""

    def __init__(self, stats=None):
        self.sum_v = 0.0
        self.sum_l = 0.0
        self.sum_vl = 0.0
        self.count = 0
        # Per-term batch counts: a term only contributes where it was measurable.
        self.count_v = 0
        self.count_l = 0
        self.count_vl = 0
        # Observability only; None keeps behaviour byte-identical.
        self.stats = stats

    def add_batch(self, out, image_mask, attention_mask=None, **kwargs):
        if isinstance(out, (tuple, list)) and len(out) == 1:
            out = out[0]
        v, l, vl, cv, cl, cvl = cos_pairwise_density(
            out, image_mask, attention_mask=attention_mask, stats=self.stats,
            return_counts=True,
        )
        if cv:
            self.sum_v += v
            self.count_v += 1
        if cl:
            self.sum_l += l
            self.count_l += 1
        if cvl:
            self.sum_vl += vl
            self.count_vl += 1
        self.count += 1

    def get_stats(self):
        """(v, l, vl, defined) where `defined` names the terms that were measured."""
        v = self.sum_v / self.count_v if self.count_v else 0.0
        l = self.sum_l / self.count_l if self.count_l else 0.0
        vl = self.sum_vl / self.count_vl if self.count_vl else 0.0
        defined = tuple(
            name for name, c in (("v", self.count_v), ("l", self.count_l), ("vl", self.count_vl)) if c
        )
        return v, l, vl, defined


class LayerWiseBasePruner(BasePruner):
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
            is_strct_pruning=is_strct_pruning,
            importance_scores_cache=importance_scores_cache,
            keep_indices_or_masks_cache=keep_indices_or_masks_cache,
            is_global=is_global,
            num_samples=num_samples,
        )

        self.sparsity_ratio_granularity = sparsity_ratio_granularity
        self.max_sparsity_per_layer = max_sparsity_per_layer
        self.score_method = score_method
        self.num_data_first_stage = num_data_first_stage
        self.num_noise = num_noise
        self.sparsity_dict = sparsity_dict
        self.noise_eps = noise_eps
        self.prune_per_model=prune_per_model

        self.prune_spec = prune_spec
        self.model_prefix = model_prefix
        self.prune_n = 0
        self.prune_m = 0
        self.model_stem = getattr(self.model, model_prefix, None) # self.model.t5_model, self.model.visual, etc
        
    def compute_importance_scores(self, model, data_loader, loss_func):
        raise NotImplementedError

    def get_params(self, model):
        params = []
        names = []

        for name, param in model.named_parameters():
            names.append(name)
            params.append(param)

        return names, params

    def model_setup_and_record_attributes(self, model):
        dtype_record = {}
        requires_grad_record = {}
        # for n, p in model.state_dict().items():
        for n, p in model.named_parameters():
            dtype_record[n] = p.data.dtype
            # p.data = p.data.type(torch.bfloat16)

        # set requires_grad to be true for getting model's derivatives
        for n, p in model.named_parameters():
            requires_grad_record[n] = p.requires_grad
            p.requires_grad = True

        device = next(iter(model.parameters())).device
        # self.model.to("cpu")

        return dtype_record, requires_grad_record, device

    def model_reset(self, model, dtype_record, requires_grad_record, device):
        # set to original requires grad
        for n, p in model.named_parameters():
            p.requires_grad = requires_grad_record[n]

        # for n, p in model.state_dict().items():
        for n, p in model.named_parameters():
            p.data = p.data.type(dtype_record[n])
            
        model.to(device)
            
    def convert_spec_to_list(self, spec):
        num_layers, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = spec.split("-")

        num_layers = int(num_layers)
        res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = float(res_keep_ratio), float(attn_keep_ratio), float(ffn_keep_ratio)

        return num_layers, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio
    
    def create_pruned_arch(self, *args, **kwargs):
        return NotImplementedError


class LayerSparsity:
    def __init__(
            self, 
            model, 
            data_loader, 
            loss_func, 
            num_samples, 
            original_sparsity, 
            max_sparsity_per_layer=0.8, 
            score_method="GradMagSquare_avg", 
            num_noise=1, 
            noise_eps=1e-3, 
            layer_to_group_mapping={}, 
            prune_per_model=False,
            per_model_group=[],
            calibration_fn=None,
            model_for_calibration=None,
            data_loader_for_calibration=None,
        ):
        self.importance_measure = {}
        self.model = model
        self.data_loader = data_loader
        self.loss_func = loss_func
        self.num_samples = num_samples
        self.original_sparsity = original_sparsity
        self.layer_to_group_mapping = layer_to_group_mapping
        self.max_sparsity_per_layer = max_sparsity_per_layer
        self.num_noise = num_noise
        self.noise_eps = noise_eps
        self.prune_per_model = prune_per_model
        
        self.score_method = score_method
        self.per_model_group = per_model_group
        
        if score_method is not None:
            self.score_compute, self.score_aggregate = score_method.split("_")
            if self.score_compute.startswith("density"):
                self.score_aggregate = "sum"

        self.calibration_fn = calibration_fn
        self.model_for_calibration = model_for_calibration
        self.data_loader_for_calibration = data_loader_for_calibration
        
        assert self.max_sparsity_per_layer >= self.original_sparsity
        
    def get_mask(self, importance_scores, p, max_sparsity_per_layer):
        # Set top (1 - max_sparsity)% of parameters to be very large value to avoid 
        # them being pruned
        
        for k, v in importance_scores.items():
            num_to_set = int(importance_scores[k].numel() * (1 - max_sparsity_per_layer))
            
            if num_to_set > 0:
                threshold, _ = torch.topk(importance_scores[k].flatten(), num_to_set, largest=True)
                threshold = threshold[-1] # take the last value

                importance_scores[k][torch.where(v >= threshold)] = torch.finfo(v.dtype).max
        
        # Flatten all tensors and concatenate them
        all_scores = torch.cat([t.flatten() for t in importance_scores.values()])
        
        # Sort and find the threshold
        num_to_zero_out = int(p * all_scores.numel())
        threshold, _ = torch.topk(all_scores, num_to_zero_out, largest=False)
        threshold = threshold[-1]
        
        # Create mask based on threshold
        masks = {}
        for k, v in importance_scores.items():
            masks[k] = (v > threshold).type(v.dtype)
        
        return masks
    
    def get_layerwise_mask(self, importance_scores, p):
        # Set top (1 - max_sparsity)% of parameters to be very large value to avoid 
        # them being pruned
        
        masks = {}
        for k, v in importance_scores.items():
            all_scores = importance_scores[k].flatten().cuda()
            num_to_zero_out = int(p * all_scores.numel())
            threshold, _ = torch.topk(all_scores, num_to_zero_out, largest=False)
            threshold = threshold[-1].cpu()

            masks[k] = (v > threshold).type(v.dtype)

        return masks
        
    def global_iterative_pruning(self, target_sparsity, dict_layers_to_prune, iteratation=1, max_sparsity_per_layer=1.0):
        
        weight_copy = {}
        total_parameters = 0
        names = []
        params = []
        for k, v in self.model.named_parameters():  
            if k in dict_layers_to_prune:
                names.append(k)
                params.append(v)
                weight_copy[k] = torch.clone(v).cpu()

        masks = None
        for i in range(1, iteratation+1):
            p_i = target_sparsity ** (iteratation / i) # Compute modified sparsity for the i^th iteration
            
            importance_measure = self.compute_importance_scores(
                dict_layers_to_prune
            )
            
            importance_measure = {k: v for k, v in importance_measure.items() if k in dict_layers_to_prune}
            
            if masks is not None:
                # Apply mask to importance scores (this step is to simulate pruning in iterations)
                for k in importance_measure:
                    importance_measure[k] *= masks[k]

            masks = self.get_mask(importance_measure, p_i, max_sparsity_per_layer)

            # prune the model
            for k, v in self.model.named_parameters():
                if k in masks:
                    v.data *= masks[k].type(v.dtype).to(v.device)
                    
            print(f"Step {i}, target sparsity: {p_i:.4f}")
        
        sparsity_dict = {}
        for k, v in self.model.named_parameters():
            sparsity_dict[k] = ((v == 0).float().sum() / v.numel()).item()
            
        for k, p in zip(names, params):
            # use current_batch_index rather than self.num_samples because sometimes
            # the batch size might not be 1, and the loss is already normalized by 
            # batch size, now when only have to normalize it by num_batches now
            p.data = weight_copy[k].to(p.device)
        
        return sparsity_dict
    
    def compute_the_sparsity_per_group(self, total_parameters_to_keep, group_scores, group_num_parameters, max_sparsity_per_layer=0.8):
        scores = torch.FloatTensor(list(group_scores.values()))
        num_parameters = torch.LongTensor(list(group_num_parameters.values()))
        
        parameters_to_keep_per_group = torch.zeros_like(scores, dtype=int)
        
        parameters_to_keep_per_group += torch.ceil(num_parameters * (1 - max_sparsity_per_layer)).int() # to gaurantee the max_sparsity
        
        while parameters_to_keep_per_group.sum() < total_parameters_to_keep:
            total_ratio = torch.sum(scores)
            
            rest_total_parameters_to_keep = total_parameters_to_keep - parameters_to_keep_per_group.sum()
            
            parameters_to_add = torch.ceil((scores / total_ratio) * rest_total_parameters_to_keep)
            
            parameters_to_keep_per_group = parameters_to_keep_per_group + parameters_to_add
            
            scores[parameters_to_keep_per_group >= num_parameters] = 0 # make sure they are not going to add more parameters
            
            parameters_to_keep_per_group = torch.clamp(parameters_to_keep_per_group, max=num_parameters) # remove the extra parameters

            # they are to make sure the sum of parameters_to_keep_per_group is EXACTLY the same as total_parameters_to_keep
            if parameters_to_add.sum() == 0: # for some reason the algo cannot add more parameters
                # the algo stuck
                current_sum = parameters_to_keep_per_group.sum()
                if current_sum < total_parameters_to_keep:
                    num_need_to_add = total_parameters_to_keep - current_sum
                    
                    while num_need_to_add > 0:
                        # distributed the parameters to the rest of groups
                        for index in torch.where(scores > 0)[0]:
                            parameters_can_add = min(
                                num_need_to_add, num_parameters[index] - parameters_to_keep_per_group[index]
                            )
                            parameters_to_keep_per_group[index] += parameters_can_add
                            
                            num_need_to_add -= parameters_can_add
                            
                            if num_need_to_add == 0:
                                break
                            
            if parameters_to_keep_per_group.sum() > total_parameters_to_keep: # for some reason the algo cannot add more parameters
                # the algo stuck
                current_sum = parameters_to_keep_per_group.sum()

                num_need_to_remove = current_sum - total_parameters_to_keep
                
                while num_need_to_remove > 0:
                    # remove the parameters from full groups
                    for index in torch.argsort(parameters_to_keep_per_group, descending=True, stable=True):
                        parameters_can_remove = min(
                            num_need_to_remove, 
                            parameters_to_keep_per_group[index] - (num_parameters[index] * (1 - max_sparsity_per_layer)).int() # extra parameters
                        )
                        parameters_to_keep_per_group[index] += parameters_can_remove
                        
                        num_need_to_remove -= parameters_can_remove
                        
                        if num_need_to_remove == 0:
                            break
                        
        # convert the group parameters to keep to sparsity    
        group_sparsity = {}
        
        for k, param_to_keep, group_max_param in zip(group_num_parameters.keys(), parameters_to_keep_per_group, num_parameters):
            group_sparsity[k] = torch.clamp(1 - param_to_keep / group_max_param, min=0, max=1).item()
            
        return group_sparsity
    
    @print_time
    def return_sparsity(self):
        original_sparsity = self.original_sparsity
        layer_to_group_mapping = self.layer_to_group_mapping
        
        if self.score_compute.startswith("Real"):
            # get the layer sparsity perform the real global pruning
            return self.global_iterative_pruning(
                original_sparsity, layer_to_group_mapping, iteratation=3, max_sparsity_per_layer=1.0
            )

        if layer_to_group_mapping is None or len(layer_to_group_mapping) == 0:
            class uniform_sparsity_module:
                def __getitem__(self, key):
                    return original_sparsity
            return uniform_sparsity_module()

        # compute the global information
        if len(self.importance_measure) == 0:
            if self.score_compute.startswith("MEZO"):
                # use zeroth-order gradient
                self.importance_measure = self.compute_importance_scores_mezo(layer_to_group_mapping)
            elif self.score_compute.startswith("density"):
                self.importance_measure = self.compute_density(layer_to_group_mapping)
            else:
                # use first-order gradient
                self.importance_measure = self.compute_importance_scores(layer_to_group_mapping)

        # create the layer list that for each group
        group_to_layer_mapping = {}
        for k, v in layer_to_group_mapping.items():
            if v not in group_to_layer_mapping:
                group_to_layer_mapping[v] = []

            group_to_layer_mapping[v].append(k)
        
        # store the num of parameters for each group and the total paramters
        num_parameters_dict = {}
        total_parameters = 0
        for k, v in self.model.named_parameters():
            if k in layer_to_group_mapping:
                num_parameters_dict[k] = v.numel()
                total_parameters += v.numel()
        
        # total params to keep
        total_parameters_to_keep = int(total_parameters * (1 - original_sparsity))
        
        # store the importance per parameter for each group
        group_scores = {}
        group_num_parameters = {}
        for group_name, layers in group_to_layer_mapping.items():
            if group_name not in group_scores:
                group_scores[group_name] = 0
            
            num_params = 0
            for l in layers:
                group_scores[group_name] += self.importance_measure[l].sum()
                
                num_params += num_parameters_dict[l]
            
            if self.score_aggregate == "avg":
                group_scores[group_name] /= num_params

            group_num_parameters[group_name] = num_params

        if self.prune_per_model:
            group_sparsity = {}
            for submodel_prefix in self.per_model_group:
                print(submodel_prefix)
                submodel_group_scores = {k: v for k, v in group_scores.items() if k.startswith(submodel_prefix)}
                submodel_group_num_parameters = {k: v for k, v in group_num_parameters.items() if k.startswith(submodel_prefix)}
                
                submodel_total_parameters_to_keep = int(sum(list(submodel_group_num_parameters.values())) * (1 - original_sparsity))
                submodel_group_sparsity = self.compute_the_sparsity_per_group(
                    submodel_total_parameters_to_keep, 
                    submodel_group_scores, 
                    submodel_group_num_parameters, 
                    max_sparsity_per_layer=self.max_sparsity_per_layer,
                )
                group_sparsity.update(submodel_group_sparsity)
        else:
            group_sparsity = self.compute_the_sparsity_per_group(
                total_parameters_to_keep, 
                group_scores, 
                group_num_parameters, 
                max_sparsity_per_layer=self.max_sparsity_per_layer,
            )
        
        compute_total_keep_parameters = 0
        for k in group_num_parameters:
            compute_total_keep_parameters += (1 - group_sparsity[k]) * group_num_parameters[k]

        # sanity check
        print(compute_total_keep_parameters, total_parameters_to_keep)
        
        layer_sparsity = {
            k: group_sparsity[v]
            for k, v in layer_to_group_mapping.items()
        }
        
        return layer_sparsity

    def compute_density(self, layer_to_group_mapping):
        """Per-Linear DAS importance from T5 encoder hidden states and modality masks."""
        default_t5_encoder_module_path = "t5_model.encoder.block"
        default_t5_encoder_prefix = f"{default_t5_encoder_module_path}."
        encoder_prefixes = sorted(
            {
                key.split("encoder.block.")[0] + "encoder.block."
                for key in layer_to_group_mapping
                if "encoder.block." in key
            },
            key=len,
            reverse=True,
        )
        param_encoder_prefix = encoder_prefixes[0] if encoder_prefixes else default_t5_encoder_prefix
        t5_encoder_module_path = param_encoder_prefix.rstrip(".")
        has_t5_encoder_keys = bool(encoder_prefixes)
        try:
            _get_module_by_path(self.model, t5_encoder_module_path)
            has_t5_encoder_module = True
        except AttributeError:
            has_t5_encoder_module = False

        if not (has_t5_encoder_keys or has_t5_encoder_module):
            raise NotImplementedError(
                "compute_density is only implemented for T5 encoder "
                f"(keys containing 'encoder.block.' or model with {default_t5_encoder_module_path})"
            )

        if self.calibration_fn is None:
            raise ValueError(
                "compute_density requires calibration_fn (encoder calib with image_masks)"
            )

        model = self.model_for_calibration if self.model_for_calibration is not None else self.model
        data_loader = self.data_loader_for_calibration if self.data_loader_for_calibration is not None else self.data_loader
        device = next(iter(model.parameters())).device
        model.eval()

        with torch.no_grad():
            calib_result = self.calibration_fn(model, data_loader, device)
        if len(calib_result) == 3:
            inps, outs, caches = calib_result
            image_masks = None
            encoder_attention_masks = None
        elif len(calib_result) >= 5:
            inps, outs, caches, image_masks, encoder_attention_masks = calib_result[:5]
        else:
            inps, outs, caches, image_masks = calib_result
            encoder_attention_masks = None

        if image_masks is None or len(image_masks) == 0:
            raise ValueError(
                "compute_density requires calibration_fn to return image_masks; got none"
            )
        if encoder_attention_masks is None or len(encoder_attention_masks) == 0:
            raise ValueError(
                "compute_density requires calibration_fn to return raw encoder_attention_masks; got none"
            )
        if len(image_masks) != len(inps) or len(encoder_attention_masks) != len(inps):
            raise ValueError(
                "compute_density calibration batch mismatch: "
                f"inps={len(inps)} image_masks={len(image_masks)} "
                f"encoder_attention_masks={len(encoder_attention_masks)}"
            )
        for idx, (inp, img_mask, attn_mask) in enumerate(
            zip(inps, image_masks, encoder_attention_masks)
        ):
            B, S = inp.shape[0], inp.shape[1]
            if tuple(img_mask.shape) != (B, S):
                raise ValueError(
                    f"image_masks[{idx}].shape {tuple(img_mask.shape)} vs input {(B, S)}"
                )
            if tuple(attn_mask.shape) != (B, S):
                raise ValueError(
                    f"encoder_attention_masks[{idx}].shape {tuple(attn_mask.shape)} vs input {(B, S)}"
                )

        n_samples = len(inps)
        blocks = _get_module_by_path(model, t5_encoder_module_path)
        num_blocks = len(blocks)

        maybe_autocast = getattr(
            model, "maybe_autocast", lambda dtype=None: contextlib.nullcontext()
        )

        density_dict = {name: 0.0 for name in layer_to_group_mapping}
        matched_names = set()
        # Opt-in contamination audit: counts how often a v/l/vl term could not be
        # measured and silently defaulted to 0.0 (which downstream reads as
        # "maximally diverse"). Off by default -- the counters force GPU syncs.
        das_stats = collections.Counter() if os.environ.get("LAVIS_DAS_DIAGNOSTIC") else None
        # ECoFLaP calibration convention: every block replays with the arguments captured
        # at block 0, so position_bias stays None and blocks >0 fall back to a zero bias.
        # Kept deliberately for comparability with ECoFLaP / Wanda / SparseGPT numbers --
        # do NOT propagate per-block position_bias without re-running affected experiments.
        layer_caches = [dict(cache) for cache in caches]

        for i in range(num_blocks):
            layer = blocks[i]
            subset = find_layers(layer)
            wrapped_layers = {}
            for name, sub_layer in subset.items():
                full_name = f"{param_encoder_prefix}{i}.{name}.weight"
                if full_name in density_dict:
                    wrapped_layers[name] = ActivationDensity(stats=das_stats)

            def add_batch(name, batch_index):
                def tmp(_, inp, out):
                    img_mask_j = image_masks[batch_index] if batch_index < len(image_masks) else image_masks[0]
                    attn_j = (
                        encoder_attention_masks[batch_index]
                        if encoder_attention_masks is not None and batch_index < len(encoder_attention_masks)
                        else None
                    )
                    wrapped_layers[name].add_batch(out, img_mask_j, attention_mask=attn_j)

                return tmp

            new_inps = []
            for j in range(n_samples):
                handles = [
                    subset[name].register_forward_hook(add_batch(name, j))
                    for name in wrapped_layers
                ]
                try:
                    with torch.no_grad():
                        with maybe_autocast(dtype=torch.bfloat16):
                            out, _ = _normal_t5_block_forward(
                                layer,
                                inps[j],
                                layer_caches[j],
                            )
                finally:
                    for h in handles:
                        h.remove()
                new_inps.append(out.detach())
            inps = new_inps
            for name, act_density in wrapped_layers.items():
                full_name = f"{param_encoder_prefix}{i}.{name}.weight"
                v, l, vl, defined = act_density.get_stats()
                if len(defined) == 3:
                    # Multimodal: original TAMP expression, kept verbatim so the
                    # multimodal path stays bit-identical.
                    density_dict[full_name] = (1.0 - v) + (1.0 - l) + (1.0 - vl)
                elif defined:
                    # Single-modality calibration: average over the modality pairs
                    # that exist, rescaled to the 3-term range so layer importances
                    # remain on the same scale as the multimodal case.
                    terms = []
                    if "v" in defined:
                        terms.append(1.0 - v)
                    if "l" in defined:
                        terms.append(1.0 - l)
                    if "vl" in defined:
                        terms.append(1.0 - vl)
                    density_dict[full_name] = sum(terms) * (3.0 / len(terms))
                else:
                    raise RuntimeError(
                        f"DAS could not measure any modality-pair diversity for {full_name}; "
                        "the calibration batch has too few tokens."
                    )
                matched_names.add(full_name)

        missing_names = sorted(set(layer_to_group_mapping) - matched_names)
        if missing_names:
            preview = ", ".join(missing_names[:5])
            raise RuntimeError(
                "compute_density did not observe all requested T5 encoder Linear layers; "
                f"missing {len(missing_names)} keys, e.g. {preview}"
            )

        if das_stats is not None:
            n = max(1, das_stats["samples"])
            report = {
                "measurements": das_stats["samples"],
                "no_attention_mask": das_stats["no_attention_mask"],
                "v_undefined_too_few": das_stats["v_undefined_too_few"],
                "l_undefined_too_few": das_stats["l_undefined_too_few"],
                "vl_undefined_too_few": das_stats["vl_undefined_too_few"],
                "v_empty_after_positive_filter": das_stats["v_empty_after_positive_filter"],
                "l_empty_after_positive_filter": das_stats["l_empty_after_positive_filter"],
                "contaminated_frac": round(
                    (
                        das_stats["v_undefined_too_few"]
                        + das_stats["l_undefined_too_few"]
                        + das_stats["vl_undefined_too_few"]
                        + das_stats["v_empty_after_positive_filter"]
                        + das_stats["l_empty_after_positive_filter"]
                    )
                    / (3.0 * n),
                    6,
                ),
                "v_positive_pair_frac": round(
                    das_stats["v_pairs_kept_positive"] / max(1, das_stats["v_pairs_total"]), 6
                ),
                "l_positive_pair_frac": round(
                    das_stats["l_pairs_kept_positive"] / max(1, das_stats["l_pairs_total"]), 6
                ),
                "mean_visual_tokens": round(das_stats["n_visual_total"] / n, 3),
                "mean_language_tokens": round(das_stats["n_language_total"] / n, 3),
                "mean_pad_tokens": round(das_stats["n_pad_total"] / n, 3),
            }
            print("[DAS-AUDIT]", json.dumps(report))
            out_path = os.environ.get("LAVIS_DAS_DIAGNOSTIC_JSON")
            if out_path:
                with open(out_path, "w", encoding="utf-8") as fh:
                    json.dump(report, fh, indent=2)
                print(f"[DAS-AUDIT] wrote {out_path}")

        importance_measure = {
            name: torch.FloatTensor([density_dict[name]]).abs()
            for name in layer_to_group_mapping
        }

        return importance_measure

    @print_time
    def compute_importance_scores(self, layer_to_group_mapping):
        model = self.model
        data_loader = self.data_loader
        loss_func = self.loss_func
        
        names = []
        params = []
        for k, v in model.named_parameters():
            if k in layer_to_group_mapping:
                names.append(k)
                params.append(v)
            
        gradients_dict = {k: 0 for k in names}
        
        device = next(iter(model.parameters())).device

        accum_samples = 0
        current_batch_index = 0
        
        for d in data_loader:
            # print(accum_samples)
            if accum_samples >= self.num_samples:
                break
            
            loss, batch_len = loss_func(model, d, device != "cpu")

            accum_samples += batch_len
            current_batch_index += 1

            grads = torch.autograd.grad(loss, params)
            
            assert len(grads) == len(names) == len(params)

            for k, v in zip(names, grads):
                
                if self.score_compute == "GradMagSquare":
                    gradients_dict[k] += v.cpu().data.float() ** 2
                else:
                    gradients_dict[k] += v.cpu().data.float().abs()

        for k in names:
            # use current_batch_index rather than self.num_samples because sometimes
            # the batch size might not be 1, and the loss is already normalized by 
            # batch size, now when only have to normalize it by num_batches now
            gradients_dict[k] /= current_batch_index
        
        if "GradMagSquare" in self.score_compute:
            # using square of magnitude multiplied by diagonal fisher as importance scores
            importance_measure = {k: (v.cpu().data.float() ** 2) * gradients_dict[k] for k, v in zip(names, params)}
        elif "GradMagAbs" in self.score_compute:
            importance_measure = {k: (v.cpu().data.float().abs()) * gradients_dict[k].abs() for k, v in zip(names, params)}
        elif "GradOnly" in self.score_compute:
            importance_measure = {k: gradients_dict[k].abs() for k, v in zip(names, params)}
        
        return importance_measure
    
    def zo_perturb_parameters(self, params, random_seed=1, scaling_factor=1, zo_eps=1e-3):
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed)
        
        for param in params:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * zo_eps
    
    def compute_importance_scores_mezo(self, layer_to_group_mapping):
        model = self.model
        data_loader = self.data_loader
        loss_func = self.loss_func
        
        names = []
        params = []
        model.eval()
        for k, v in model.named_parameters():  
            if k in layer_to_group_mapping:
                names.append(k)
                params.append(v)
        
        gradients_dict = {k: 0 for k in names}
        
        device = next(iter(model.parameters())).device

        accum_samples = 0
        current_batch_index = 0
        
        zo_eps = self.noise_eps
        
        n_mezo = self.num_noise
        
        for i, (name, param) in enumerate(zip(names, params)):
            print(i, name)
            accum_samples = 0
            current_batch_index = 0
            
            for d in data_loader:
                if accum_samples >= self.num_samples:
                    break
                
                per_gradients_dict = {name: 0}
                
                for _ in range(n_mezo):
                    
                    if accum_samples >= self.num_samples:
                        break
                    
                    zo_random_seed = np.random.randint(1000000000)
                    
                    self.zo_perturb_parameters([param], random_seed=zo_random_seed, scaling_factor=1, zo_eps=zo_eps)
                    with torch.no_grad():
                        loss1, batch_len = loss_func(model, d, device != "cpu")
                    
                    self.zo_perturb_parameters([param], random_seed=zo_random_seed, scaling_factor=-2, zo_eps=zo_eps)
                    with torch.no_grad():
                        loss2, batch_len = loss_func(model, d, device != "cpu")
                
                    # recover the weight
                    self.zo_perturb_parameters([param], random_seed=zo_random_seed, scaling_factor=1, zo_eps=zo_eps)

                    accum_samples += batch_len
                    current_batch_index += 1
                    
                    projected_grad = ((loss1 - loss2) / (2 * zo_eps)).item()

                    torch.manual_seed(zo_random_seed)
                    per_gradients_dict[name] += abs(projected_grad)
                        
                gradients_dict[name] += torch.FloatTensor([per_gradients_dict[name]]).abs()
                
        if self.score_compute == "MEZO-GradOnly":
            # only use gradient
            importance_measure = {k: gradients_dict[k].abs() for k, v in zip(names, params)}
        elif self.score_compute == "MEZO-GradMagAbs":
            # gradient * magnitude
            importance_measure = {k: v.cpu().data.float().abs() * gradients_dict[k].abs() for k, v in zip(names, params)}
        elif self.score_compute == "MEZO-GradMagSquare":
            # (gradient * magnitude) ** 2
            importance_measure = {k: v.cpu().data.float() ** 2 * gradients_dict[k] ** 2 for k, v in zip(names, params)}
            
        return importance_measure
