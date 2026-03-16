#!/usr/bin/env python3
"""
Run OK-VQA evaluation per category (11 categories).
Usage:
  cd /root/autodl-tmp/UKMP-main/LAVIS

  # 全精度（不剪枝）BLIP2，按 11 类跑 eval：
  PRUNED_CKPT= python scripts/structured_blip2/run_okvqa_eval_by_category.py
  # 或
  export PRUNED_CKPT=
  python scripts/structured_blip2/run_okvqa_eval_by_category.py

  # 指定剪枝模型跑 11 类 eval：
  export PRUNED_CKPT=pruned_checkpoint/ukmp_prune/okvqa_CF-.../pytorch_model.bin
  python scripts/structured_blip2/run_okvqa_eval_by_category.py

  # 多组 calibration 时区分结果：用 EVAL_RUN_PREFIX 避免覆盖（如 calibVT -> okvqa_eval_calibVT_VT）
  export PRUNED_CKPT=... EVAL_RUN_PREFIX=calibVT python scripts/structured_blip2/run_okvqa_eval_by_category.py
"""

import os
import subprocess
import sys

# 不设或设为空 = 全精度；否则 = 剪枝模型路径
PRUNED_CKPT = os.environ.get("PRUNED_CKPT", "").strip() or None
if PRUNED_CKPT is not None and PRUNED_CKPT.lower() in ("none", "full", "fullprecision", "full_precision"):
    PRUNED_CKPT = None
GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
# 多组 calibration 时加前缀，结果目录为 okvqa_eval_<PREFIX>_<Label>，不设则 okvqa_eval_<Label>
EVAL_RUN_PREFIX = os.environ.get("EVAL_RUN_PREFIX", "").strip() or None

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

    if PRUNED_CKPT is not None and not os.path.isabs(PRUNED_CKPT) and not os.path.exists(PRUNED_CKPT):
        print(
            f"[ERROR] Pruned checkpoint not found: {PRUNED_CKPT}\n"
            "Set PRUNED_CKPT to your pruned pytorch_model.bin path, or leave unset for full-precision eval."
        )
        sys.exit(1)

    use_fullprecision = PRUNED_CKPT is None
    if use_fullprecision:
        print("[INFO] Full-precision eval (no pruned checkpoint).")

    cwd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if os.path.basename(cwd) != "LAVIS":
        print("[ERROR] Run this script from UKMP-main/LAVIS (or ensure script path is correct).")
        sys.exit(1)

    eval_dir = "lavis/projects/blip2/eval"
    job_prefix = "okvqa_fullprecision_eval" if use_fullprecision else "okvqa_eval"
    if EVAL_RUN_PREFIX:
        job_prefix = f"{job_prefix}_{EVAL_RUN_PREFIX}"
    # 全精度模型显存占用大，降低 batch_size_eval 避免 OOM（24GB 卡）
    fullprecision_options = ' --options "run.batch_size_eval=16"' if use_fullprecision else ""
    try:
        for label, job_label in CATEGORIES:
            cfg_name = f"okvqa_zeroshot_flant5xl_eval_{label}.yaml"
            cfg_path = f"{eval_dir}/{cfg_name}"
            job_id = f"{job_prefix}_{job_label}"
            cmd = (
                f"CUDA_VISIBLE_DEVICES={GPU} python -u evaluate_blip2_pruned.py"
                f" --cfg-path {cfg_path}"
                f" --job_id {job_id}"
                f"{fullprecision_options}"
            )
            if PRUNED_CKPT:
                cmd += f" --pruned_ckpt {PRUNED_CKPT}"
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
