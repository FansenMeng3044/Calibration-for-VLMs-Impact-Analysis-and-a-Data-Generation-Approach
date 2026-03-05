#!/usr/bin/env python3
"""
Run OK-VQA evaluation per category (11 categories) for a CF-calibration pruned model.
Usage:
  cd /root/autodl-tmp/UKMP-main/LAVIS
  # Set PRUNED_CKPT to your pruned checkpoint (e.g. after CF-calibration pruning)
  export PRUNED_CKPT=pruned_checkpoint/ukmp_prune/okvqa_CF-128data-.../pytorch_model.bin
  python scripts/structured_blip2/run_okvqa_eval_by_category.py

Or override from command line:
  PRUNED_CKPT=path/to/pytorch_model.bin python scripts/structured_blip2/run_okvqa_eval_by_category.py
"""

import os
import subprocess
import sys

# Default: CF-calibration pruned checkpoint (set your job_id folder name)
PRUNED_CKPT = os.environ.get(
    "PRUNED_CKPT",
    "pruned_checkpoint/ukmp_prune/okvqa_CF-128data-taylor+knowledge-param_first-param_norm-0.5-blockwise-global-select_loss-multimodal/pytorch_model.bin",
)
GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

# 11 OK-VQA categories: (eval_yaml_basename_suffix, label for job_id)
CATEGORIES = [
    ("VT", "VT"),
    ("BCP", "BCP"),
    ("OMC", "OMC"),
    ("SR", "SR"),
    ("CF", "CF"),
    ("GHLC", "GHLC"),
    ("PEL", "PEL"),
    ("PA", "PA"),
    ("ST", "ST"),
    ("WC", "WC"),
    ("Other", "Other"),
]


def main():
    _safetensors_index_bak = None  # 用于最后恢复 model.safetensors.index.json
    # 若 ECoFLaP 等已在默认路径下过 flan-t5-xl，则复用该缓存，避免重复下载
    default_hub = "/root/.cache/huggingface/hub"
    existing_flan = os.path.join(default_hub, "models--google--flan-t5-xl")
    use_existing = os.path.isdir(existing_flan) and os.path.isdir(
        os.path.join(existing_flan, "blobs")
    )
    if use_existing:
        try:
            size_gb = sum(
                os.path.getsize(os.path.join(r, f))
                for r, _, files in os.walk(os.path.join(existing_flan, "blobs"))
                for f in files
            ) / (1024 ** 3)
            # 只认“完整版”缓存：≥10GB（14G 的 pytorch 版），避免误用只有几 G 的 safetensors 半成品
            use_existing = size_gb >= 10.0
        except Exception:
            use_existing = False

    if use_existing:
        os.environ["HF_HOME"] = "/root/.cache/huggingface"
        os.environ["HUGGINGFACE_HUB_CACHE"] = default_hub
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        # 让 T5 从 snapshot 目录加载，使用已有的 pytorch .bin（需暂时隐藏 safetensors index）
        snap_dir = os.path.join(existing_flan, "snapshots")
        flan_snapshot = None
        if os.path.isdir(snap_dir):
            for rev in os.listdir(snap_dir):
                p = os.path.join(snap_dir, rev)
                if os.path.isdir(p) and os.path.isfile(os.path.join(p, "pytorch_model.bin.index.json")):
                    flan_snapshot = p
                    break
        if flan_snapshot:
            os.environ["FLAN_T5_XL_SNAPSHOT"] = flan_snapshot
            safetensors_index = os.path.join(flan_snapshot, "model.safetensors.index.json")
            if os.path.isfile(safetensors_index):
                _safetensors_index_bak = safetensors_index + ".bak"
                try:
                    os.rename(safetensors_index, _safetensors_index_bak)
                except Exception:
                    _safetensors_index_bak = None
        print(f"[INFO] Reusing existing HF cache (full flan-t5-xl from ECoFLaP): {default_hub}")
    else:
        hf_cache_root = "/root/autodl-tmp/.cache/huggingface"
        os.environ["HF_HOME"] = hf_cache_root
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_cache_root, "hub")
        os.makedirs(os.environ["HUGGINGFACE_HUB_CACHE"], exist_ok=True)
        if "TRANSFORMERS_OFFLINE" in os.environ:
            del os.environ["TRANSFORMERS_OFFLINE"]
        print(f"[INFO] HF cache: {os.environ['HUGGINGFACE_HUB_CACHE']}")

    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print(f"[INFO] HF endpoint: {os.environ.get('HF_ENDPOINT', 'default')}")

    if not os.path.isabs(PRUNED_CKPT) and not os.path.exists(PRUNED_CKPT):
        print(
            f"[ERROR] Pruned checkpoint not found: {PRUNED_CKPT}\n"
            "Set PRUNED_CKPT to your pruned pytorch_model.bin path (e.g. pruned_checkpoint/ukmp_prune/<job_id>/pytorch_model.bin)"
        )
        sys.exit(1)

    cwd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if os.path.basename(cwd) != "LAVIS":
        print("[ERROR] Run this script from UKMP-main/LAVIS (or ensure script path is correct).")
        sys.exit(1)

    eval_dir = "lavis/projects/blip2/eval"
    try:
        for label, job_label in CATEGORIES:
            cfg_name = f"okvqa_zeroshot_flant5xl_eval_{label}.yaml"
            cfg_path = f"{eval_dir}/{cfg_name}"
            job_id = f"okvqa_eval_{job_label}"
            cmd = (
                f"CUDA_VISIBLE_DEVICES={GPU} python -u evaluate_blip2_pruned.py"
                f" --cfg-path {cfg_path}"
                f" --pruned_ckpt {PRUNED_CKPT}"
                f" --job_id {job_id}"
            )
            print(f"\n[RUN] OK-VQA eval category: {job_label} ({cfg_name})")
            print(cmd)
            ret = subprocess.call(cmd, shell=True, cwd=cwd)
            if ret != 0:
                print(f"[WARN] Eval failed for {job_label} with exit code {ret}")
    finally:
        if _safetensors_index_bak and os.path.isfile(_safetensors_index_bak):
            try:
                os.rename(_safetensors_index_bak, _safetensors_index_bak.replace(".bak", ""))
            except Exception:
                pass
    print("\n[DONE] All 11 category evals finished. Check output/BLIP2/OKVQA or logs for per-category accuracy.")


if __name__ == "__main__":
    main()
