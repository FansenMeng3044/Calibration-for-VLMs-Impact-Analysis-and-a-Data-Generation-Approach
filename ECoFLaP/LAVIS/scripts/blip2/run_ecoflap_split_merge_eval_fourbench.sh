#!/usr/bin/env bash
# =============================================================================
# ECoFLaP：T5 单侧剪枝权重 + ViT 单侧剪枝权重 → 合并（可选）→ MMBench / MMMU / OKVQA / MathVista 评测
#
# 依赖：在 ECoFLaP/LAVIS 根目录运行；已存在两侧剪枝产物 pruned_checkpoint/*.pth
#
# 环境变量（按需覆盖）:
#   CKPT_T5_ONLY   T5-only 剪枝完整权重（默认示例路径见下）
#   CKPT_VIT_ONLY  ViT-only 剪枝完整权重
#   MERGED_CKPT    合并输出路径；默认 REPO/pruned_checkpoint/merged_ecoflap_t5_<stem>_vit_<stem>.pth
#   RUN_MERGE=1    是否先 merge（默认 1）；若为 0 则评测全程用 --vit_ckpt/--t5_ckpt 组合加载，不写合并文件
#   RUN_EVAL=1     是否跑四基准（默认 1）
#   SKIP_MMBENCH=1 / SKIP_MMMU=1 / SKIP_OKVQA=1 / SKIP_MATHVISTA=1  跳过其中某一项（默认 0，全跑）
#
#   MMBENCH_ROOT / MMMU_ROOT / MathVista JSON 等同仓库内其它 eval 脚本
#   HF_HOME        已下载的 HuggingFace 缓存根目录（默认见下）；内含 hub/transformers
#   也可直接设 BERT_BASE_UNCASED_SNAPSHOT / FLAN_T5_XL_SNAPSHOT 指向 hub snapshot 目录
#
# 默认权重对应两次剪枝（evaluate_blip.py --save_pruned_model，job_id 即文件名）:
#   - ViT: bash scripts/run_prune_vit_cc3m_128.sh（CC3M128 + vit_image_only + MEZO-GradOnly_sum）→ 默认 ecoflap_vit_encode_proxy.pth
#   - T5:  bash scripts/run_prune_t5_c4_128.sh（C4 文本 + t5_c4_text + MEZO-GradOnly_sum）；合并脚本默认名 ecoflap_separate_t5_only.pth 请设 JOB_ID=ecoflap_separate_t5_only
# 用法:
#   cd /root/autodl-tmp/ECoFLaP/LAVIS
#   bash scripts/blip2/run_ecoflap_split_merge_eval_fourbench.sh
# 分开剪枝默认产物 → 四评测（薄封装，见）:
#   bash scripts/blip2/run_ecoflap_fourbench_eval_from_split_prune.sh
#
# 「联合剪枝」单文件 pth 四评测：设 JOINT_SINGLE_CKPT=/path/to/joint.pth（跳过 merge 与双文件检查）
#   bash scripts/blip2/run_ecoflap_joint_cc3m128_prune_then_fourbench.sh
# 纯 Wanda 等单文件 pth 薄封装（也可用 JOB_ID=xxx）:
#   bash scripts/blip2/run_wanda_pruned_fourbench_eval.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 联合剪枝一条权重评四基准：单 pth，不 merge
JOINT_SINGLE_CKPT="${JOINT_SINGLE_CKPT:-}"

if [[ -n "$JOINT_SINGLE_CKPT" ]]; then
  RUN_MERGE=0
  CKPT_T5_ONLY="$JOINT_SINGLE_CKPT"
  CKPT_VIT_ONLY="$JOINT_SINGLE_CKPT"
  STEM_T5="$(basename "$JOINT_SINGLE_CKPT" .pth)"
  STEM_VIT="$STEM_T5"
  MERGED_CKPT="$JOINT_SINGLE_CKPT"
else
  CKPT_T5_ONLY="${CKPT_T5_ONLY:-$REPO_ROOT/pruned_checkpoint/ecoflap_separate_t5_only.pth}"
  CKPT_VIT_ONLY="${CKPT_VIT_ONLY:-$REPO_ROOT/pruned_checkpoint/ecoflap_vit_encode_proxy.pth}"
  STEM_T5="$(basename "$CKPT_T5_ONLY" .pth)"
  STEM_VIT="$(basename "$CKPT_VIT_ONLY" .pth)"
  MERGED_CKPT="${MERGED_CKPT:-$REPO_ROOT/pruned_checkpoint/merged_ecoflap_t5_${STEM_T5}__vit_${STEM_VIT}.pth}"
fi

RUN_MERGE="${RUN_MERGE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
if [[ -n "${JOINT_SINGLE_CKPT:-}" ]]; then
  RUN_MERGE=0
fi

MMBENCH_ROOT="${MMBENCH_ROOT:-/root/autodl-tmp/MMBench_eval}"
MMMU_ROOT="${MMMU_ROOT:-/root/autodl-tmp/MMMU_single_image}"
MATHVISTA_EVAL_JSON="${MATHVISTA_EVAL_JSON:-/root/autodl-tmp/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
export MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
export MMMU_SPLIT="${MMMU_SPLIT:-test}"

MASTER_PORT="${MASTER_PORT:-29600}"
if [[ -n "${JOINT_SINGLE_CKPT:-}" ]]; then
  EVAL_JOB_OKVQA="${EVAL_JOB_OKVQA:-okvqa_eval_joint_${STEM_T5}}"
else
  EVAL_JOB_OKVQA="${EVAL_JOB_OKVQA:-okvqa_eval_merged_split_${STEM_T5}__${STEM_VIT}}"
fi

# HF：使用本机已下载缓存（离线）。默认 HF_HOME；BERT/Flan 的 snapshot 从 hub 目录自动解析，无需手写 commit hash。
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache_moved/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

_resolve_hub_snapshot_dir() {
  local repo_dir="$1"
  [[ -d "$repo_dir/snapshots" ]] || { echo ""; return 0; }
  find "$repo_dir/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | head -1
}

# HUGGINGFACE_HUB_CACHE 已是 .../huggingface/hub，勿再拼 /hub
HUB_ROOT="${HUGGINGFACE_HUB_CACHE}"
if [[ -z "${BERT_BASE_UNCASED_SNAPSHOT:-}" ]]; then
  BERT_BASE_UNCASED_SNAPSHOT="$(_resolve_hub_snapshot_dir "$HUB_ROOT/models--bert-base-uncased")"
fi
if [[ -z "${FLAN_T5_XL_SNAPSHOT:-}" ]]; then
  FLAN_T5_XL_SNAPSHOT="$(_resolve_hub_snapshot_dir "$HUB_ROOT/models--google--flan-t5-xl")"
fi
export BERT_BASE_UNCASED_SNAPSHOT
export FLAN_T5_XL_SNAPSHOT

if [[ ! -d "$BERT_BASE_UNCASED_SNAPSHOT" ]]; then
  echo "[FATAL] 未找到 bert-base-uncased 本地快照。请设置 HF_HOME（当前: $HF_HOME）或 export BERT_BASE_UNCASED_SNAPSHOT=/path/to/hub/.../snapshots/<hash>" >&2
  exit 1
fi
if [[ ! -d "$FLAN_T5_XL_SNAPSHOT" ]]; then
  echo "[FATAL] 未找到 google/flan-t5-xl 本地快照。请设置 HF_HOME 或 export FLAN_T5_XL_SNAPSHOT=..." >&2
  exit 1
fi
echo "[INFO] HF_HOME=$HF_HOME"
echo "[INFO] BERT_BASE_UNCASED_SNAPSHOT=$BERT_BASE_UNCASED_SNAPSHOT"
echo "[INFO] FLAN_T5_XL_SNAPSHOT=$FLAN_T5_XL_SNAPSHOT"

check_file() {
  if [[ ! -f "$1" ]]; then
    echo "[FATAL] 找不到文件: $1" >&2
    exit 1
  fi
}

if [[ -n "${JOINT_SINGLE_CKPT:-}" ]]; then
  check_file "$JOINT_SINGLE_CKPT"
  echo "[INFO] 联合剪枝单权重: $JOINT_SINGLE_CKPT"
else
  check_file "$CKPT_T5_ONLY"
  check_file "$CKPT_VIT_ONLY"
  echo "[INFO] T5-only:  $CKPT_T5_ONLY"
  echo "[INFO] ViT-only: $CKPT_VIT_ONLY"
  echo "[INFO] MERGED:   $MERGED_CKPT (RUN_MERGE=$RUN_MERGE)"
fi

if [[ "$RUN_MERGE" == "1" ]] && [[ -z "${JOINT_SINGLE_CKPT:-}" ]]; then
  python "$SCRIPT_DIR/merge_ecoflap_split_prune_ckpts.py" \
    --t5_ckpt "$CKPT_T5_ONLY" \
    --vit_ckpt "$CKPT_VIT_ONLY" \
    --out "$MERGED_CKPT"
fi

# 评测用：JOINT_SINGLE / RUN_MERGE=1 合并文件 → 单 ckpt；否则双路径
if [[ -n "${JOINT_SINGLE_CKPT:-}" ]]; then
  CKPT_FOR_SINGLE="$JOINT_SINGLE_CKPT"
elif [[ "$RUN_MERGE" == "1" ]]; then
  CKPT_FOR_SINGLE="$MERGED_CKPT"
else
  CKPT_FOR_SINGLE=""
fi

run_mmbench() {
  echo ""
  echo "========== MMBench ($MMBENCH_ROOT, split=$MMBENCH_SPLIT) =========="
  export LAVIS_METRICS_BENCHMARK="MMBench"
  if [[ -n "$CKPT_FOR_SINGLE" ]] && [[ -f "$CKPT_FOR_SINGLE" ]]; then
    python scripts/blip2/mmmu_eval_by_discipline.py \
      --mmmu_root "$MMBENCH_ROOT" \
      --split "$MMBENCH_SPLIT" \
      --ckpt "$CKPT_FOR_SINGLE" \
      --batch_size "$EVAL_BATCH_SIZE" \
      --device cuda \
      --overall_only
  else
    python scripts/blip2/mmmu_eval_by_discipline.py \
      --mmmu_root "$MMBENCH_ROOT" \
      --split "$MMBENCH_SPLIT" \
      --vit_ckpt "$CKPT_VIT_ONLY" \
      --t5_ckpt "$CKPT_T5_ONLY" \
      --batch_size "$EVAL_BATCH_SIZE" \
      --device cuda \
      --overall_only
  fi
}

run_mmmu() {
  echo ""
  echo "========== MMMU ($MMMU_ROOT, split=$MMMU_SPLIT) =========="
  export LAVIS_METRICS_BENCHMARK="MMMU"
  if [[ -n "$CKPT_FOR_SINGLE" ]] && [[ -f "$CKPT_FOR_SINGLE" ]]; then
    python scripts/blip2/mmmu_eval_by_discipline.py \
      --mmmu_root "$MMMU_ROOT" \
      --split "$MMMU_SPLIT" \
      --ckpt "$CKPT_FOR_SINGLE" \
      --batch_size "$EVAL_BATCH_SIZE" \
      --device cuda \
      --overall_only
  else
    python scripts/blip2/mmmu_eval_by_discipline.py \
      --mmmu_root "$MMMU_ROOT" \
      --split "$MMMU_SPLIT" \
      --vit_ckpt "$CKPT_VIT_ONLY" \
      --t5_ckpt "$CKPT_T5_ONLY" \
      --batch_size "$EVAL_BATCH_SIZE" \
      --device cuda \
      --overall_only
  fi
}

run_okvqa() {
  echo ""
  echo "========== OKVQA zeroshot overall =========="
  local P=$MASTER_PORT
  MASTER_PORT=$((MASTER_PORT + 1))
  if [[ -n "$CKPT_FOR_SINGLE" ]] && [[ -f "$CKPT_FOR_SINGLE" ]]; then
    python -m torch.distributed.run --nproc_per_node=1 --master_port="$P" evaluate_blip.py \
      --cfg-path lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml \
      --t5_pruned_checkpoint "$CKPT_FOR_SINGLE" \
      --vit_pruned_checkpoint "$CKPT_FOR_SINGLE" \
      --job_id "$EVAL_JOB_OKVQA"
  else
    python -m torch.distributed.run --nproc_per_node=1 --master_port="$P" evaluate_blip.py \
      --cfg-path lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml \
      --t5_pruned_checkpoint "$CKPT_T5_ONLY" \
      --vit_pruned_checkpoint "$CKPT_VIT_ONLY" \
      --job_id "$EVAL_JOB_OKVQA"
  fi
}

run_mathvista() {
  echo ""
  echo "========== MathVista MC ($MATHVISTA_EVAL_JSON) =========="
  export LAVIS_METRICS_BENCHMARK="MathVista_MC"
  if [[ -n "$CKPT_FOR_SINGLE" ]] && [[ -f "$CKPT_FOR_SINGLE" ]]; then
    python scripts/blip2/mathvista_mc_eval.py \
      --eval_json "$MATHVISTA_EVAL_JSON" \
      --ckpt "$CKPT_FOR_SINGLE" \
      --batch_size "$EVAL_BATCH_SIZE" \
      --device cuda
  else
    python scripts/blip2/mathvista_mc_eval.py \
      --eval_json "$MATHVISTA_EVAL_JSON" \
      --vit_ckpt "$CKPT_VIT_ONLY" \
      --t5_ckpt "$CKPT_T5_ONLY" \
      --batch_size "$EVAL_BATCH_SIZE" \
      --device cuda
  fi
}

if [[ "$RUN_EVAL" == "1" ]]; then
  if [[ -n "${JOINT_SINGLE_CKPT:-}" ]]; then
    export LAVIS_EVAL_CALIB_TAG="joint_${STEM_T5}"
  else
    export LAVIS_EVAL_CALIB_TAG="merged_split_${STEM_T5}__${STEM_VIT}"
  fi
  if [[ "${SKIP_MMBENCH:-0}" != "1" ]]; then run_mmbench; else echo "[SKIP] MMBench"; fi
  if [[ "${SKIP_MMMU:-0}" != "1" ]]; then run_mmmu; else echo "[SKIP] MMMU"; fi
  if [[ "${SKIP_OKVQA:-0}" != "1" ]]; then run_okvqa; else echo "[SKIP] OKVQA"; fi
  if [[ "${SKIP_MATHVISTA:-0}" != "1" ]]; then run_mathvista; else echo "[SKIP] MathVista"; fi
  echo ""
  echo "[INFO] 四基准评测结束。"
else
  echo "[INFO] RUN_EVAL=0，跳过评测。"
fi
