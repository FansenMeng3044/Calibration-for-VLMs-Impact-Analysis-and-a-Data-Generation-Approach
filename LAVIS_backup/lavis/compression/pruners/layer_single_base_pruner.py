import torch
import torch.nn as nn
import numpy as np
import contextlib

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


def cos_pairwise_density(embeddings, image_mask, eps=1e-8):
    """
    Compute vision-vision, language-language, and vision-language mean cosine **similarities**
    (same convention as TAMP LLaVA). Used for DAS (Density-Aware Sparsity).
    Downstream typically uses importance = (1 - v) + (1 - l) + (1 - vl) to get density-based importance.

    Args:
        embeddings: (S, D) or (B, S, D)
        image_mask: (S,) or (B, S), bool; True = vision token, False = language token
        eps: small constant for L2 normalization

    Returns:
        tuple (v_mean_sim, l_mean_sim, vl_mean_sim) as python floats (cosine similarity in [−1,1]).
        - v_mean_sim: mean cosine similarity among vision tokens (upper-triangular, excl. diagonal).
          Same as TAMP: only positive similarities are averaged (> 0). 0 if <2 vision tokens or all <= 0.
        - l_mean_sim: mean cosine similarity among language tokens. 0 if <2 language tokens or all <= 0.
        - vl_mean_sim: mean cosine similarity between vision and language. 0 if either side empty.
    """
    with torch.no_grad():
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
            image_mask = image_mask.unsqueeze(0)
            squeeze_out = True
        else:
            squeeze_out = False

        B, S, D = embeddings.shape
        device = embeddings.device

        # Ensure image_mask shape (B, S) for 3D embeddings
        if image_mask.dim() == 1:
            image_mask = image_mask.unsqueeze(0).expand(B, -1)
        image_mask = image_mask.to(device=device)

        # L2 normalize (same as TAMP: normalized dot product = cosine similarity)
        embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1, eps=eps)

        v_sims = []
        l_sims = []
        vl_sims = []

        for b in range(B):
            emb = embeddings[b]   # (S, D)
            mask = image_mask[b]   # (S,)
            v_idx = torch.where(mask)[0]
            l_idx = torch.where(~mask)[0]
            nv = v_idx.numel()
            nl = l_idx.numel()

            v_mean_sim_b = 0.0
            l_mean_sim_b = 0.0
            vl_mean_sim_b = 0.0

            if nv >= 2:
                v_emb = emb[v_idx]   # (Nv, D)
                sim_vv = v_emb @ v_emb.T   # (Nv, Nv), cosine similarity
                v_upper = sim_vv.triu(diagonal=1)   # same as TAMP: upper triangular, exclude diagonal
                v_vals = v_upper[v_upper > 0]   # TAMP uses > 0: exclude zeros and negative similarities
                v_mean_sim_b = v_vals.mean().item() if v_vals.numel() > 0 else 0.0
            if nl >= 2:
                l_emb = emb[l_idx]   # (Nl, D)
                sim_ll = l_emb @ l_emb.T
                l_upper = sim_ll.triu(diagonal=1)
                l_vals = l_upper[l_upper > 0]
                l_mean_sim_b = l_vals.mean().item() if l_vals.numel() > 0 else 0.0
            if nv >= 1 and nl >= 1:
                v_emb = emb[v_idx]
                l_emb = emb[l_idx]
                sim_vl = (v_emb @ l_emb.T).reshape(-1)
                vl_mean_sim_b = sim_vl.mean().item()

            v_sims.append(v_mean_sim_b)
            l_sims.append(l_mean_sim_b)
            vl_sims.append(vl_mean_sim_b)

        v_mean_sim = sum(v_sims) / B
        l_mean_sim = sum(l_sims) / B
        vl_mean_sim = sum(vl_sims) / B

        return float(v_mean_sim), float(l_mean_sim), float(vl_mean_sim)


class ActivationDensity:
    """
    Accumulates vision-vision, language-language, and vision-language mean cosine similarities
    over multiple batches (e.g. from encoder layer hooks). Used by DAS compute_density.
    Downstream uses importance = (1 - v) + (1 - l) + (1 - vl) with get_stats() values.
    """

    def __init__(self):
        self.sum_v = 0.0
        self.sum_l = 0.0
        self.sum_vl = 0.0
        self.count = 0

    def add_batch(self, out, image_mask, **kwargs):
        """
        Add one batch of layer output and accumulate density stats.

        Args:
            out: Layer output tensor, shape (B, S, D) or (S, D). If tuple/list with single
                 element (e.g. from a hook), the first element is used.
            image_mask: (S,) or (B, S), bool; True = vision token, False = language token.
            **kwargs: Optional, reserved for future use (e.g. AMIA valid_mask).
        """
        if isinstance(out, (tuple, list)) and len(out) == 1:
            out = out[0]
        v, l, vl = cos_pairwise_density(out, image_mask)
        self.sum_v += v
        self.sum_l += l
        self.sum_vl += vl
        self.count += 1

    def get_stats(self):
        """
        Return accumulated mean cosine similarities (same semantics as cos_pairwise_density).

        Returns:
            tuple (v_mean_sim, l_mean_sim, vl_mean_sim) as floats. If count==0, returns (0.0, 0.0, 0.0).
            Downstream typically uses importance = (1 - v_mean_sim) + (1 - l_mean_sim) + (1 - vl_mean_sim).
        """
        if self.count == 0:
            return 0.0, 0.0, 0.0
        return (
            self.sum_v / self.count,
            self.sum_l / self.count,
            self.sum_vl / self.count,
        )

    @property
    def v_density(self):
        """TAMP-compatible: same as get_stats()[0]."""
        return self.sum_v / self.count if self.count > 0 else 0.0

    @property
    def l_density(self):
        """TAMP-compatible: same as get_stats()[1]."""
        return self.sum_l / self.count if self.count > 0 else 0.0

    @property
    def vl_dist(self):
        """TAMP-compatible: same as get_stats()[2]."""
        return self.sum_vl / self.count if self.count > 0 else 0.0


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
            # density_sum (and any score_method starting with "density") must use sum aggregation
            if self.score_compute.startswith("density"):
                self.score_aggregate = "sum"
        
        # Optional: for compute_density (DAS). Default None, do not break existing callers.
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
                # DAS: density-based importance (T5 encoder only). Requires calibration_fn returning (inps, outs, caches, image_masks).
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
            
            # density_sum => score_aggregate="sum" (no normalization). Others may use "avg".
            if self.score_aggregate == "avg":
                group_scores[group_name] /= num_params # normalization
            
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
        """
        Compute per-layer importance from vision/language density (DAS).
        Only supported for BLIP-2 T5 encoder (t5_model.encoder.block).
        Requires self.calibration_fn to be set and to return (inps, outs, caches, image_masks).
        """
        # 1) Only support T5 encoder: keys contain t5_model.encoder.block. or model has t5_model.encoder.block
        t5_encoder_prefix = "t5_model.encoder.block."
        has_t5_encoder_keys = any(
            t5_encoder_prefix in k for k in layer_to_group_mapping
        )
        try:
            _get_module_by_path(self.model, "t5_model.encoder.block")
            has_t5_encoder_module = True
        except AttributeError:
            has_t5_encoder_module = False

        if not (has_t5_encoder_keys or has_t5_encoder_module):
            raise NotImplementedError(
                "compute_density is only implemented for T5 encoder "
                "(layer_to_group_mapping keys containing 't5_model.encoder.block.' or model with t5_model.encoder.block)"
            )

        if self.calibration_fn is None:
            raise ValueError(
                "compute_density requires calibration_fn to be set (e.g. prepare_calibration_input_encoder returning inps, outs, caches, image_masks)"
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
        else:
            inps, outs, caches, image_masks = calib_result

        if image_masks is None or len(image_masks) == 0:
            raise ValueError(
                "compute_density requires calibration_fn to return image_masks (Step 3); got none"
            )

        n_samples = len(inps)
        blocks = _get_module_by_path(model, "t5_model.encoder.block")
        num_blocks = len(blocks)

        maybe_autocast = getattr(
            model, "maybe_autocast", lambda dtype=None: contextlib.nullcontext()
        )

        # 2) Per-block density: for each block i, run forward on all samples, accumulate ActivationDensity.
        #    Inps must be updated after each block so the next block receives this block's output (same as TAMP inps, outs = outs, inps).
        importance_per_block = [0.0] * num_blocks

        for i in range(num_blocks):
            layer = blocks[i]
            act_density = ActivationDensity()
            new_inps = []
            for j in range(n_samples):
                with torch.no_grad():
                    with maybe_autocast(dtype=torch.bfloat16):
                        out = layer(inps[j], **caches[j])
                if isinstance(out, (tuple, list)):
                    out = out[0]
                img_mask_j = image_masks[j] if j < len(image_masks) else image_masks[0]
                act_density.add_batch(out, img_mask_j)
                new_inps.append(out.detach())
            inps = new_inps
            v, l, vl = act_density.get_stats()
            # TAMP formula: importance = (1 - v_density) + (1 - l_density) + (1 - vl_dist); we use similarity so same
            importance_per_block[i] = (1.0 - v) + (1.0 - l) + (1.0 - vl)

        # 3) Map block importance to each param key in layer_to_group_mapping (same block -> same importance).
        #    按 TAMP 原版：先对 layer_to_group_mapping 里所有 key 赋默认值，再只填 T5 encoder 的 density。
        #    非 T5 的 key（如 ViT）用 1.0 而非 0：否则 prune_per_model 下 ViT 单独分配时 total_ratio=0 会得 NaN sparsity。
        importance_measure = {name: torch.FloatTensor([1.0]) for name in layer_to_group_mapping}
        for name in layer_to_group_mapping:
            if t5_encoder_prefix not in name:
                continue
            parts = name.split(".")
            try:
                block_idx = int(parts[3])
            except (IndexError, ValueError):
                continue
            if 0 <= block_idx < num_blocks:
                imp = importance_per_block[block_idx]
                importance_measure[name] = torch.FloatTensor([imp])

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