import torch
import torch.nn as nn
import math

from time import time
from copy import deepcopy
from functools import partial

from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample
from lavis.models.blip2_models.blip2_t5 import Blip2T5
from lavis.models.t5_models.t5 import T5
from lavis.models.clip_models.eva_model import EVA_CLIP
from lavis.compression.pruners.utils import (
    loss_vision_language, loss_language, loss_vision, print_time
)
from lavis.compression.pruners.layer_single_base_pruner import LayerWiseBasePruner, LayerSparsity


def get_module_recursive(base, module_to_process):
    
    if module_to_process == "":
        return base
    
    splits = module_to_process.split(".")
    now = splits.pop(0)
    rest = ".".join(splits)
    
    base = getattr(base, now)

    return get_module_recursive(base, rest)


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
    """
    Single-batch density for AMIA: mean of v-v, l-l, v-l cosine similarities (positive only for v/l).
    embeddings: (S, D), image_mask: (S,) bool. Returns scalar float.
    No extra deps; used only when token_selection='amia'.
    """
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
    """
    Token selection for calibration: only selected tokens update scaler_row (for Wanda |W|*sqrt(scaler_row)).
    Compatible with T5 encoder: score may be None (use density-only / uniform).
    """
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
        # out: (N, D), image_mask: (N,) bool, score: (N,) or None
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
                "token_selection='amia' requires image_masks. Please enable return_image_masks in calibration (Step 3)."
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
                mask_flat = torch.nn.functional.pad(mask_flat.bool(), (0, out_flat.shape[0] - mask_flat.numel()), value=False)
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
        # image_mask, score ignored (signature compatible with AMIA)
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
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples


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
    
    def prepare_calibration_input_encoder(self, model, dataloader, device, model_prefix, n_samples, module_to_process="encoder.block", return_image_masks=False):
        """
        Run calibration for T5 encoder: collect first-layer inputs (inps), caches, and optionally image_masks.
        When return_image_masks=True (TAMP/DAS path): also collect model.temp_label per batch and return
        (inps, outs, caches, image_masks). Semantics aligned with TAMP:
        - TAMP (LLaVA): Catcher on model.layers[0], image_masks.append(model.temp_label) per batch.
        - Here (BLIP-2): Catcher on t5_model.encoder.block[0], same append per batch; temp_label is set
          in Blip2T5.forward (Step 1) before T5 encoder runs, so it is available when Catcher fires.
        - image_masks[i] aligns with inps[i]: same batch, shape (B, S) with inps[i].shape (B, S, D).
        """
        use_cache = getattr(model, model_prefix).config.use_cache
        getattr(model, model_prefix).config.use_cache = False
        
        layers = get_module_recursive(model, module_to_process)

        dtype = next(iter(model.parameters())).dtype
        inps = []
        
        caches = []
        image_masks = [] if return_image_masks else None
        
        keys_to_cache = [
            "attention_mask", "position_bias", "encoder_attention_mask", "encoder_decoder_position_bias",
            "layer_head_mask", "cross_attn_layer_head_mask", "encoder_hidden_states",
        ]
        
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            def forward(self, inp, **kwargs):
                # 不能对非叶子张量设 requires_grad，用 detach 存一份不需梯度的副本
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
            total_samples += batch["image"].shape[0]
            try:
                self.forward_to_cache(model, batch)
            except ValueError:
                if return_image_masks:
                    if not hasattr(model, "temp_label"):
                        raise RuntimeError(
                            "model.temp_label not found. Please complete Step 1 (set temp_label in Blip2T5.forward)."
                        )
                    with torch.no_grad():
                        mask = model.temp_label.detach()
                    if mask.dtype != torch.bool:
                        mask = mask.bool()
                    image_masks.append(mask.cpu())
                pass
        layers[0] = layers[0].module
        
        outs = [None] * len(inps)
        
        getattr(model, model_prefix).config.use_cache = use_cache

        if return_image_masks:
            # Self-check: image_masks aligned with inps (same length, same batch/seq dims).
            assert len(image_masks) == len(inps), (
                "image_masks vs inps length mismatch: %d vs %d" % (len(image_masks), len(inps))
            )
            for i in range(len(inps)):
                B, S = inps[i].shape[0], inps[i].shape[1]
                assert image_masks[i].shape == (B, S), (
                    "image_masks[%d].shape %s vs inps[%d] (B,S) (%d,%d)" % (i, image_masks[i].shape, i, B, S)
                )
            return inps, outs, caches, image_masks
        return inps, outs, caches
    
    @print_time
    def _prune(self, model, dataloader, device, model_prefix, module_to_process="encoder.block", n_samples=64, sparsity_ratio=0.5, token_selection="naive", image_masks=None, scores=None, cached_calib=None):
        use_cache = getattr(model, model_prefix).config.use_cache 
        getattr(model, model_prefix).config.use_cache = False 

        return_image_masks = token_selection == "amia"
        print("loading calibdation data")
        with torch.no_grad():
            if cached_calib is not None:
                result = cached_calib
            else:
                result = self.prepare_calibration_input_encoder(
                    model, dataloader, device, model_prefix, n_samples, module_to_process,
                    return_image_masks=return_image_masks,
                )
            inps, outs, caches = result[0], result[1], result[2]
            image_masks = result[3] if len(result) == 4 else None

        if token_selection == "amia" and (image_masks is None or len(image_masks) == 0):
            raise RuntimeError(
                "token_selection='amia' requires image_masks. Enable return_image_masks in prepare_calibration_input_encoder (Step 3) and ensure model.temp_label is set (Step 1)."
            )

        n_samples = min(n_samples, len(inps))

        layers = get_module_recursive(model, module_to_process)
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            wrapped_layers = {}
            for name in subset:
                if token_selection == "amia":
                    wrapped_layers[name] = AdaptiveMultimodalInputActivation(subset[name])
                else:
                    wrapped_layers[name] = WrappedGPT(subset[name])

            def add_batch(name, j_ref):
                def tmp(_, inp, out):
                    out_tensor = out[0] if isinstance(out, (tuple, list)) else out
                    inp_data = inp[0].data
                    mask_j = image_masks[j_ref] if image_masks is not None else None
                    score_j = scores[j_ref] if scores is not None else None
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
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)

                    # unstructured pruning
                    sparsity_key = f"{module_to_process}.{i}.{name}.weight"
                    indices = sort_res[1][:,:int(W_metric.shape[1] * sparsity_ratio[sparsity_key])]
                    W_mask.scatter_(1, indices, True)

                subset[name].weight.data[W_mask] = 0  ## set weights to zero 

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
                inps.append(inp.detach())
                
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
                        (name.startswith(self.t5_model_prefix) or \
                            name.startswith(self.vit_model_prefix)):
                    return True
                return False
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
        
        # When score_method is density_sum, LayerSparsity.compute_density needs calibration_fn returning (inps, outs, caches, image_masks).
        calibration_fn = None
        if self.score_method == "density_sum":
            def calibration_fn(model, data_loader, device):
                if getattr(self, "_cached_encoder_calib", None) is not None:
                    return self._cached_encoder_calib
                return T5LayerWandaPruner.prepare_calibration_input_encoder(
                    self, model, data_loader, device, self.t5_model_prefix,
                    self.num_data_first_stage, module_to_process=f"{self.t5_model_prefix}.encoder.block",
                    return_image_masks=True,
                )
        
        sparsity_module = LayerSparsity(
            self.model, 
            self.data_loader, 
            loss_vision_language, 
            self.num_data_first_stage,
            original_sparsity,
            self.max_sparsity_per_layer,
            self.score_method,
            self.num_noise,
            self.noise_eps,
            layer_to_group_mapping,
            prune_per_model=self.prune_per_model,
            per_model_group=[self.t5_model_prefix, self.vit_model_prefix],
            calibration_fn=calibration_fn,
        )
        
        return sparsity_module.return_sparsity()
        
    def forward_to_cache(self, model, batch):
        return model(batch)

    @print_time
    def prune(self, importance_scores=None, keep_indices_or_masks=None):
        print("In: ", self.pruner_name)
        dtype_record, requires_grad_record, device = self.model_setup_and_record_attributes(self.model)

        self._cached_encoder_calib = None
        need_calib = (
            (self.sparsity_ratio_granularity is not None and self.score_method == "density_sum")
            or self.token_selection == "amia"
        )
        if need_calib and self.t5_prune_spec is not None:
            self.prepare_calibration_input_encoder = partial(
                T5LayerWandaPruner.prepare_calibration_input_encoder,
                self,
            )
            calib_result = self.prepare_calibration_input_encoder(
                self.model, self.data_loader, device, self.t5_model_prefix,
                self.num_data_first_stage, module_to_process=f"{self.t5_model_prefix}.encoder.block",
                return_image_masks=True,
            )
            if self.token_selection == "amia" and len(calib_result) != 4:
                raise RuntimeError(
                    "token_selection='amia' requires image_masks. Enable return_image_masks in prepare_calibration_input_encoder (Step 3) and ensure model.temp_label is set (Step 1)."
                )
            self._cached_encoder_calib = calib_result

        global_sparsity_dict = None
        if self.sparsity_ratio_granularity is not None: 
            _, vit_keep_ratio, _, _ = self.convert_spec_to_list(self.vit_prune_spec)
            _, t5_keep_ratio, _, _ = self.convert_spec_to_list(self.t5_prune_spec) 
            assert vit_keep_ratio == t5_keep_ratio

            global_sparsity_dict = self.get_sparsity(
                1 - vit_keep_ratio, # same as 1 - t5_keep_ratio
                sparsity_ratio_granularity=self.sparsity_ratio_granularity
            )
            
        if self.vit_prune_spec is not None:
            _, keep_ratio, _, _ = self.convert_spec_to_list(self.vit_prune_spec)
        
            sparsity_ratio = 1 - keep_ratio
            
            if global_sparsity_dict is not None:
                sparsity_dict = global_sparsity_dict
            else:
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
            
        if self.t5_prune_spec is not None:
            _, keep_ratio, _, _ = self.convert_spec_to_list(self.t5_prune_spec)
        
            sparsity_ratio = 1 - keep_ratio
            
            if global_sparsity_dict is not None:
                sparsity_dict = global_sparsity_dict
            else:
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
                token_selection=getattr(self, "token_selection", "naive"),
                cached_calib=self._cached_encoder_calib if need_calib else None,
            )
            
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