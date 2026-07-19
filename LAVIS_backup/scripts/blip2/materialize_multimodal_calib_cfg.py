#!/usr/bin/env python3
"""Write a runtime multimodal calibration YAML with explicit JSON/image paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET_NAME = "prefix_conceptual_caption_3m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a BLIP2 multimodal calibration config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src_cfg", required=True, type=Path)
    parser.add_argument("--out_cfg", required=True, type=Path)
    parser.add_argument("--annotation_json", required=True, type=Path)
    parser.add_argument("--images_dir", required=True, type=Path)
    parser.add_argument("--pretrained", required=True, type=Path)
    parser.add_argument("--run_seed", type=int, default=None)
    return parser.parse_args()


def write_fallback_cfg(args: argparse.Namespace) -> None:
    """Fallback used only if OmegaConf is unavailable in a lightweight env."""
    seed = 42 if args.run_seed is None else int(args.run_seed)
    cfg = f"""model:
  arch: blip2_t5
  model_type: pretrain_flant5xl
  use_grad_checkpoint: False
  pretrained: {json.dumps(str(args.pretrained))}

datasets:
  {DATASET_NAME}:
    vis_processor:
      train:
        name: "blip2_image_train"
        image_size: 224
    text_processor:
      train:
        name: "blip_caption"
    build_info:
      annotations:
        train:
          url:
            - {json.dumps(str(args.annotation_json))}
          storage:
            - {json.dumps(str(args.annotation_json))}
      images:
        storage: {json.dumps(str(args.images_dir))}

run:
  task: image_text_pretrain
  lr_sched: "linear_warmup_cosine_lr"
  init_lr: 1e-4
  min_lr: 1e-5
  warmup_lr: 1e-6
  weight_decay: 0.05
  max_epoch: 1
  batch_size_train: 16
  batch_size_eval: 16
  num_workers: 4
  warmup_steps: 1000
  seed: {seed}
  output_dir: "output/BLIP2/hybrid_c4_multimodal_calibration"
  amp: True
  resume_ckpt_path: null
  evaluate: False
  train_splits: ["train"]
  test_splits: ["train"]
  device: "cuda"
  world_size: 1
  dist_url: "env://"
  distributed: True
"""
    args.out_cfg.parent.mkdir(parents=True, exist_ok=True)
    args.out_cfg.write_text(cfg, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.src_cfg.is_file():
        raise FileNotFoundError("source cfg not found: %s" % args.src_cfg)
    if not args.annotation_json.is_file():
        raise FileNotFoundError("annotation JSON not found: %s" % args.annotation_json)
    if not args.images_dir.is_dir():
        raise FileNotFoundError("images dir not found: %s" % args.images_dir)
    if not args.pretrained.is_file():
        raise FileNotFoundError("pretrained checkpoint not found: %s" % args.pretrained)

    try:
        from omegaconf import OmegaConf  # type: ignore

        cfg = OmegaConf.load(str(args.src_cfg))
        dataset = cfg.datasets[DATASET_NAME]
        train_ann = dataset.build_info.annotations.train
        train_ann.url = [str(args.annotation_json)]
        train_ann.storage = [str(args.annotation_json)]
        dataset.build_info.images.storage = str(args.images_dir)
        cfg.model.pretrained = str(args.pretrained)
        if args.run_seed is not None:
            cfg.run.seed = int(args.run_seed)

        args.out_cfg.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config=cfg, f=str(args.out_cfg))
    except ModuleNotFoundError:
        write_fallback_cfg(args)

    print("[OK] wrote runtime multimodal cfg: %s" % args.out_cfg)
    print("[OK] annotation JSON: %s" % args.annotation_json)
    print("[OK] images dir: %s" % args.images_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
