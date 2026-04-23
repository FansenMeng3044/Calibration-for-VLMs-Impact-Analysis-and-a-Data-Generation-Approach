#!/usr/bin/env bash
# ViT-only Wanda：CC3M 128 图 + vit_image_only（encode 代理 loss），--no_prune_t5
# 与联合剪对齐：MEZO-GradOnly_sum + block + max_sparsity_per_layer=0.6
# 单侧剪时「整体」仅在 ViT 内部做 block 间分配（不与 T5 抢预算）。
#
# 默认仅使用本机 HF hub 快照 + 本地 BLIP2 pth（与 run_ecoflap_split_merge_eval_fourbench.sh 一致）。
#
# 用法（在 ECoFLaP/LAVIS 根目录）:
#   bash scripts/run_prune_vit_cc3m_128.sh
# 产出默认 job_id → pruned_checkpoint/${JOB_ID}.pth（与 run_ecoflap_split_merge_eval_fourbench 默认 ViT ckpt 名对齐可设 JOB_ID=ecoflap_vit_encode_proxy）
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

# --- HuggingFace / Transformers：离线，BERT & Flan-T5 从本地 snapshot 加载 ---
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache_moved/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

_resolve_hub_snapshot_dir() {
  local repo_dir="$1"
  [[ -d "$repo_dir/snapshots" ]] || { echo ""; return 0; }
  find "$repo_dir/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | head -1
}

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
  echo "[FATAL] 未找到 bert-base-uncased 本地目录。请设置 HF_HOME 或 export BERT_BASE_UNCASED_SNAPSHOT=..." >&2
  exit 1
fi
if [[ ! -d "$FLAN_T5_XL_SNAPSHOT" ]]; then
  echo "[FATAL] 未找到 google/flan-t5-xl 本地目录。请设置 HF_HOME 或 export FLAN_T5_XL_SNAPSHOT=..." >&2
  exit 1
fi

BLIP2_PRETRAINED="${BLIP2_PRETRAINED:-/root/autodl-tmp/cache_moved/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth}"
if [[ ! -f "$BLIP2_PRETRAINED" ]]; then
  echo "[FATAL] 未找到本地 BLIP2 预训练权重: $BLIP2_PRETRAINED" >&2
  exit 1
fi

echo "[INFO] HF_HOME=$HF_HOME"
echo "[INFO] BERT_BASE_UNCASED_SNAPSHOT=$BERT_BASE_UNCASED_SNAPSHOT"
echo "[INFO] FLAN_T5_XL_SNAPSHOT=$FLAN_T5_XL_SNAPSHOT"
echo "[INFO] BLIP2_PRETRAINED=$BLIP2_PRETRAINED"

CFG="${CFG:-${REPO}/lavis/projects/blip2/eval/cc_prefix_derivative_compute_cc3m_calib128.yaml}"
VIT_SPEC="${VIT_SPEC:-39-0.5-1.0-1.0}"
JOB_ID="${JOB_ID:-ecoflap_vit_encode_proxy}"
NUM_DATA="${NUM_DATA:-128}"
BS="${BS:-8}"
SCORE_METHOD="${SCORE_METHOD:-MEZO-GradOnly_sum}"
SPARSITY_GRANULARITY="${SPARSITY_GRANULARITY:-block}"
MAX_SPARSITY_PER_LAYER="${MAX_SPARSITY_PER_LAYER:-0.6}"
MASTER_PORT="${MASTER_PORT:-29507}"

cd "${REPO}"

python -m torch.distributed.run --nproc_per_node=1 --master_port "${MASTER_PORT}" evaluate_blip.py \
  --cfg-path "${CFG}" \
  --options model.pretrained="${BLIP2_PRETRAINED}" \
  --prune_calib_mode vit_image_only \
  --no_prune_t5 \
  --pruning_method blipt5_wanda_pruner \
  --score_method "${SCORE_METHOD}" \
  --sparsity_ratio_granularity "${SPARSITY_GRANULARITY}" \
  --max_sparsity_per_layer "${MAX_SPARSITY_PER_LAYER}" \
  --vit_prune_spec "${VIT_SPEC}" \
  --num_data "${NUM_DATA}" \
  --prunining_dataset_batch_size "${BS}" \
  --num_data_first_stage "${NUM_DATA}" \
  --job_id "${JOB_ID}" \
  --save_pruned_model \
  "$@"
