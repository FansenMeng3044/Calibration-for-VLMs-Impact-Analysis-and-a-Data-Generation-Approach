"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import random
import time
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *

from lavis.compression import load_pruner
from lavis.compression.unimodal_prune import (
    build_text_only_dataloader,
    wrap_model_for_unimodal_prune,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    parser.add_argument(
        "--side_pretrained_weight",
        type=str,
        default=None,
        help="The pre-trained config for the distilled transformer."
    )

    parser.add_argument(
        "--vit_side_pretrained_weight",
        type=str,
        default=None,
        help="The pre-trained config for the distilled transformer."
    )

    parser.add_argument(
        "--distillation_init",
        type=str,
        default="sum",
        help="Whether to init the distilled transformer."
    )

    parser.add_argument(
        "--distilled_block_ids",
        type=str,
        default=None,
        help="The layer assignment to merge the distilled transformer."
    )

    parser.add_argument(
        "--distilled_block_weights",
        type=str,
        default=None,
        help="The weight assignments to merge the distilled transformer."
    )

    parser.add_argument(
        "--modules_to_merge",
        type=str,
        default=".*",
        help="The type of modules to merge."
    )

    parser.add_argument(
        "--permute_before_merge",
        action="store_true",
        default=False,
        help="Whether to permute the layers before merging (permute based on the first layer)"
    )

    parser.add_argument(
        "--permute_on_block_before_merge",
        action="store_true",
        default=False,
        help="Whether to permute the layers before merging (permute independently based on blocks)"
    )

    parser.add_argument(
        "--job_id",
        type=str,
        default=None,
        help="The id of the Job"
    )

    parser.add_argument(
        "--vit_ffn_ratio", type=float, default=1.0
    )

    parser.add_argument(
        "--distilled_merge_ratio", type=float, default=0.5
    )

    parser.add_argument(
        "--exact", action="store_true"
    )

    parser.add_argument(
        "--normalization", action="store_true"
    )

    parser.add_argument(
        "--metric", type=str, default="dot"
    )

    parser.add_argument(
        "--distill_merge_ratio", type=float, default=0.5
    )

    parser.add_argument(
        "--to_one", action="store_true"
    )

    parser.add_argument(
        "--importance", action="store_true"
    )

    parser.add_argument(
        "--num_data", type=int, default=128
    )

    parser.add_argument(
        "--power", type=int, default=2
    )

    parser.add_argument(
        "--num_logits", type=int, default=1
    )

    parser.add_argument(
        "--get_derivative_info", action="store_true"
    )

    parser.add_argument(
        "--get_activation_info", action="store_true"
    )

    parser.add_argument(
        "--use_input_activation", action="store_true"
    )

    parser.add_argument(
        "--save_pruned_indices", action="store_true"
    )

    parser.add_argument(
        "--vit_pruned_indices", type=str, default=None
    )

    parser.add_argument(
        "--t5_pruned_indices", type=str, default=None
    )

    parser.add_argument(
        "--save_importance_measure", action="store_true"
    )

    parser.add_argument(
        "--vit_importance_measure", type=str, default=None
    )

    parser.add_argument(
        "--t5_importance_measure", type=str, default=None
    )
    
    parser.add_argument(
        "--t5_pruned_checkpoint", type=str, default=None
    )
    
    parser.add_argument(
        "--vit_pruned_checkpoint", type=str, default=None
    )
    
    parser.add_argument(
        "--t5_prune_spec", type=str, default=None
    )
    
    parser.add_argument(
        "--vit_prune_spec", type=str, default=None
    )

    parser.add_argument(
        "--vision_weight", type=float, default=0.0
    )

    parser.add_argument(
        "--save_final_activations", action="store_true"
    )
    
    parser.add_argument(
        "--pruning_method",
        type=str,
        default="blipt5_wanda_pruner",
        help=(
            "Registry pruner name, e.g. blipt5_wanda_pruner. "
            "blipt5_tamp_pruner is an alias: sets amia + density_sum + layer, then runs wanda."
        ),
    )

    parser.add_argument(
        "--token_selection",
        type=str,
        default="naive",
        choices=["naive", "amia"],
        help="T5 encoder Wanda calibration: naive (all tokens) or amia (adaptive multimodal).",
    )
    
    parser.add_argument(
        "--save_pruned_model", action="store_true"
    )
    
    parser.add_argument(
        "--sparsity_ratio_granularity",
        type=str,
        default=None,
        help="DAS / joint budgeting: layer, block, model, or None. Use with density_sum (see --score_method).",
    )
    
    parser.add_argument(
        "--max_sparsity_per_layer", type=float, default=0.8
    )
    
    parser.add_argument(
        "--score_method",
        type=str,
        default="obd_avg",
        help="LayerSparsity score, e.g. obd_avg, MEZO-*, GradMagSquare_avg, density_sum (DAS).",
    )
    
    parser.add_argument(
        "--num_data_first_stage", type=int, default=32
    )
    
    parser.add_argument(
        "--num_noise", default=1, type=int,
    )
    
    parser.add_argument(
        "--noise_eps", default=1e-3, type=float,
    )
    
    parser.add_argument(
        "--sparsity_dict",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--prune_per_model",
        action="store_true"
    )
    
    parser.add_argument(
        "--is_global",
        action="store_true"
    )
    
    parser.add_argument(
        "--iteration",
        type=int,
        default=1,
    )
    
    parser.add_argument(
        "--prunining_dataset_batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--prune_calib_mode",
        type=str,
        default="multimodal",
        choices=["multimodal", "t5_c4_text", "vit_cc3m_image", "vit_image_only"],
        help=(
            "multimodal: full BLIP-2 forward for calibration. "
            "t5_c4_text: pure-text C4 + --c4_calib_json; full T5 encoder+decoder (no ViT/Q-Former). "
            "vit_cc3m_image: full BLIP-2 + CC3M-style loader; loss_vision_language; importance_scope=vit_only. "
            "vit_image_only: CC3M-style loader but only ViT encode_image; LayerNorm(feature)+mean_square loss; "
            "importance_scope=vit_only_encode."
        ),
    )
    parser.add_argument(
        "--t5_c4_encoder_only",
        action="store_true",
        help=(
            "Only for prune_calib_mode=t5_c4_text: use T5 encoder forward only and skip decoder Wanda "
            "(legacy). Default is full T5 seq2seq on text (same line as input and target)."
        ),
    )
    parser.add_argument(
        "--c4_calib_json",
        type=str,
        default=None,
        help="Required for prune_calib_mode=t5_c4_text.",
    )
    parser.add_argument(
        "--vit_calib_full_multimodal",
        action="store_true",
        help=(
            "If set, do not auto-switch multimodal -> vit_cc3m_image when doing ViT-only pruning; "
            "stay on prune_calib_mode=multimodal (then set --importance_scope vit_only yourself if needed)."
        ),
    )

    parser.add_argument(
        "--importance_scope",
        type=str,
        default=None,
        choices=["joint", "llm_only", "vit_only", "vit_only_encode"],
        help=(
            "blipt5_wanda_pruner LayerSparsity: joint = ViT+LLM global allocation (ECoFLaP default). "
            "llm_only / vit_only = only T5 or only ViT params; vit_only uses loss_vision_language. "
            "vit_only_encode = only ViT params + encode_image surrogate loss (use with vit_image_only). "
            "Default: auto — llm_only for t5_c4_text, vit_only for vit_cc3m_image, vit_only_encode for "
            "vit_image_only, joint otherwise."
        ),
    )

    parser.add_argument(
        "--no_prune_t5",
        action="store_true",
        help=(
            "blipt5_wanda_pruner only: when sparsity_ratio_granularity is set, still run joint "
            "ViT+T5 importance / sparsity allocation (same as full pipeline), but skip T5 Wanda "
            "weight pruning (only prune ViT). For identical budgeting to end-to-end prune, pass "
            "the same t5_prune_spec as you would for full run (keep ratio must match vit)."
        ),
    )

    parser.add_argument(
        "--no_prune_vit",
        action="store_true",
        help=(
            "Legacy: skip ViT pruning. Default is already to skip ViT; use --prune_vit to enable ViT pruning."
        ),
    )
    parser.add_argument(
        "--prune_vit",
        action="store_true",
        help=(
            "Enable ViT-side pruning. Default: ViT is not pruned (only T5). "
            "When --no_prune_t5 is set (ViT-only run), ViT is always pruned; this flag is ignored."
        ),
    )

    args = parser.parse_args()
    if args.prune_calib_mode == "t5_c4_text" and not args.c4_calib_json:
        parser.error("--c4_calib_json is required when --prune_calib_mode t5_c4_text")
    if args.no_prune_t5 and args.no_prune_vit:
        parser.error("cannot use --no_prune_t5 and --no_prune_vit together (nothing would be pruned)")
    if args.prune_vit and args.no_prune_vit:
        parser.error("cannot use --prune_vit and --no_prune_vit together")

    if args.no_prune_t5:
        args.no_prune_vit = False
    elif args.prune_vit:
        args.no_prune_vit = False
    else:
        args.no_prune_vit = True
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_final_activations(args, cfg, task, model, datasets):
    runner = RunnerBase(
        cfg=cfg, job_id=None, task=task, model=model, datasets=datasets
    )
    start = time.time()

    print("Start to get final activation")
    outputs = runner.get_last_activations(num_data=args.num_data, power=args.power)

    end = time.time()
    print(f"Finish get final activation, using {end - start:.3f}s")

    return outputs


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    args = parse_args()

    if args.pruning_method == "blipt5_tamp_pruner":
        args.token_selection = "amia"
        args.score_method = "density_sum"
        args.sparsity_ratio_granularity = "layer"
        args.pruning_method = "blipt5_wanda_pruner"

    if args.importance_scope is None:
        if args.prune_calib_mode == "t5_c4_text":
            args.importance_scope = "llm_only"
        elif args.prune_calib_mode == "vit_cc3m_image":
            args.importance_scope = "vit_only"
        elif args.prune_calib_mode == "vit_image_only":
            args.importance_scope = "vit_only_encode"
        else:
            args.importance_scope = "joint"

    if args.prune_calib_mode == "t5_c4_text" and args.importance_scope != "llm_only":
        print(
            f"[prune] t5_c4_text: importance_scope must be llm_only (got {args.importance_scope}); overriding."
        )
        args.importance_scope = "llm_only"
    if args.prune_calib_mode == "vit_cc3m_image" and args.importance_scope != "vit_only":
        print(
            f"[prune] vit_cc3m_image: importance_scope must be vit_only (got {args.importance_scope}); overriding."
        )
        args.importance_scope = "vit_only"
    if args.prune_calib_mode == "vit_image_only" and args.importance_scope != "vit_only_encode":
        print(
            f"[prune] vit_image_only: importance_scope must be vit_only_encode (got {args.importance_scope}); overriding."
        )
        args.importance_scope = "vit_only_encode"

    if (
        args.prune_calib_mode == "multimodal"
        and args.sparsity_ratio_granularity is None
        and args.no_prune_t5
        and not args.no_prune_vit
        and not args.vit_calib_full_multimodal
    ):
        args.prune_calib_mode = "vit_cc3m_image"
        print(
            "[prune] ViT-only separate pruning: prune_calib_mode -> vit_cc3m_image "
            "(full BLIP calib; LayerSparsity same as joint, scope vit_only). "
            "Use --vit_calib_full_multimodal to keep mode=multimodal."
        )

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    if args.job_id is not None:
        job_id = args.job_id
    else:
        job_id = now()

    cfg = Config(args)

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    
    orig_total_size = sum(
        param.numel() for param in model.parameters()
    )
    
    if args.t5_pruned_checkpoint is not None and getattr(model, "t5_model", None) is not None:
        print("Load t5 pruned weight")
        prune_state_dict = torch.load(args.t5_pruned_checkpoint, map_location="cpu")
        
        prune_state_dict = {k: v for k, v in prune_state_dict.items() if k.startswith("t5_model")}
        
        prune_state_dict = {k.replace("t5_model.", ""): v for k, v in prune_state_dict.items()}
        model.t5_model.load_state_dict(prune_state_dict)
        
    if args.vit_pruned_checkpoint is not None:
        print("Load vit pruned weight")
        prune_state_dict = torch.load(args.vit_pruned_checkpoint, map_location="cpu")
        
        model_prefix = None
        for candidate_prefix in ["visual.", "visual_encoder."]:
            if any(k.startswith(candidate_prefix) for k in prune_state_dict.keys()):
                model_prefix = candidate_prefix
                break
            
        assert model_prefix is not None
        
        prune_state_dict = {k: v for k, v in prune_state_dict.items() if k.startswith(model_prefix)}
        
        print(f"VIT checkpoint prefix: {model_prefix}")
        
        prune_state_dict = {k.replace(model_prefix, ""): v for k, v in prune_state_dict.items()}
        
        # additional_keys = [
        #     "norm.weight", "norm.bias", "head.weight", "head.bias"
        # ]
        
        original_state_dict = model.visual_encoder.state_dict()
        
        for k, v in prune_state_dict.items():
            if k in original_state_dict:
                original_state_dict[k] = v
                
        prune_state_dict = original_state_dict
        
        from lavis.models.eva_vit import interpolate_pos_embed
        
        interpolate_pos_embed(model.visual_encoder, prune_state_dict)
        # for additional_key in additional_keys:
        #     del prune_state_dict[additional_key]

        model.visual_encoder.load_state_dict(prune_state_dict)

    # Pre-pruned checkpoint + eval only: weights are already pruned; skip Wanda (would need
    # t5_prune_spec / vit_prune_spec or sparsity_ratio_granularity, and re-pruning is wrong).
    if (
        not args.save_pruned_model
        and (
            args.t5_pruned_checkpoint is not None
            or args.vit_pruned_checkpoint is not None
        )
    ):
        distilled_total_size = sum(
            (param != 0).float().sum() for param in model.parameters()
        )
        print(distilled_total_size / orig_total_size * 100)
        runner = RunnerBase(
            cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
        )
        runner.orig_total_size = orig_total_size
        runner.distilled_total_size = distilled_total_size
        runner.evaluate(skip_reload=True)
        return

    runner = RunnerBase(
        cfg=cfg, job_id=None, task=task, model=model, datasets=datasets
    )
    if args.prune_calib_mode == "multimodal":
        data_loader = runner.get_dataloader_for_importance_computation(
            num_data=args.num_data, power=args.power, batch_size=args.prunining_dataset_batch_size
        )
    elif args.prune_calib_mode == "t5_c4_text":
        data_loader = build_text_only_dataloader(
            args.c4_calib_json,
            args.num_data,
            args.prunining_dataset_batch_size,
        )
    elif args.prune_calib_mode in ("vit_cc3m_image", "vit_image_only"):
        data_loader = runner.get_dataloader_for_importance_computation(
            num_data=args.num_data, power=args.power, batch_size=args.prunining_dataset_batch_size
        )
    else:
        raise ValueError(args.prune_calib_mode)

    base_model = runner.unwrap_dist_model(runner.model)
    prune_model = base_model
    if args.prune_calib_mode == "t5_c4_text":
        prune_model, _ = wrap_model_for_unimodal_prune(
            base_model, "t5_c4_text", t5_c4_encoder_only=args.t5_c4_encoder_only
        )
    elif args.prune_calib_mode == "vit_image_only":
        prune_model, _ = wrap_model_for_unimodal_prune(base_model, "vit_image_only")

    config = {
        "t5_prune_spec": args.t5_prune_spec if args.t5_pruned_checkpoint is None else None,
        "vit_prune_spec": args.vit_prune_spec if args.vit_pruned_checkpoint is None else None,
        "t5_pruning_method": "none",
        "vit_pruning_method": "none",
        "importance_scores_cache": None,
        "keep_indices_cache": None,
        "is_strct_pruning": False,
        "is_global": args.is_global,
        "num_samples": args.num_data,
        "sparsity_ratio_granularity": args.sparsity_ratio_granularity,
        "max_sparsity_per_layer": args.max_sparsity_per_layer,
        "score_method": args.score_method,
        "token_selection": args.token_selection,
        "num_data_first_stage": args.num_data_first_stage,
        "num_noise": args.num_noise,
        "noise_eps": args.noise_eps,
        "sparsity_dict": args.sparsity_dict,
        "prune_per_model": args.prune_per_model,
        "iteration": args.iteration,
        "prune_t5": (not args.no_prune_t5),
        "prune_vit": (not args.no_prune_vit),
        "t5_unimodal_text_skip_decoder": (
            args.prune_calib_mode == "t5_c4_text" and args.t5_c4_encoder_only
        ),
        "importance_scope": args.importance_scope,
    }
    
    pruner = load_pruner(
        args.pruning_method,
        prune_model.eval(),
        data_loader,
        cfg=config
    )
    
    start = time.time()
    
    _, sparsity_dict = pruner.prune()

    # model, _ = pruner.prune()

    distilled_total_size = sum(
        (param != 0).float().sum() for param in model.parameters()
    )
    
    print(distilled_total_size / orig_total_size * 100)
    
    if args.save_pruned_model:
        saved_folder = "pruned_checkpoint"
        os.makedirs(saved_folder, exist_ok=True)
        
        torch.save(
            model.state_dict(), 
            os.path.join(saved_folder, job_id + ".pth")
        )

        print(os.path.join(saved_folder, job_id + ".pth"))
        
        # save sparsity dict
        if sparsity_dict is not None and isinstance(sparsity_dict, dict):
            saved_folder = "sparsity_dict"
            os.makedirs(saved_folder, exist_ok=True)
            
            import yaml
            with open(os.path.join(saved_folder, job_id + ".yaml"), "w") as f:
                yaml.dump(sparsity_dict, f)
                
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
        
        processing_time = time.time() - start
        
        training_dict = {
            "memory": peak_memory,
            "time": processing_time
        }
        
        saved_folder = "training_statistics"
        os.makedirs(saved_folder, exist_ok=True)
        
        import yaml
        with open(os.path.join(saved_folder, job_id + ".yaml"), "w") as f:
            yaml.dump(training_dict, f)
             
        # saved_folder = "importance_scores"
        # os.makedirs(saved_folder, exist_ok=True)
        
        # torch.save(
        #     {k: v.importance_score for k, v in model.named_parameters() if getattr(v, "importance_score", None) is not None}, 
        #     os.path.join(saved_folder, job_id + ".pth")
        # )

        exit()

    runner = RunnerBase(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )

    runner.orig_total_size = orig_total_size
    runner.distilled_total_size = distilled_total_size

    runner.evaluate(skip_reload=True)


if __name__ == "__main__":
    main()
