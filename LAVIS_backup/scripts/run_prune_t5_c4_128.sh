#!/usr/bin/env bash
# T5-only Wanda：C4 纯文本校准，默认完整 T5（encoder+decoder）；加 --t5_c4_encoder_only 则仅 encoder
# 与联合剪（ecoflap_zeroth / CC3M）对齐：
#   MEZO-GradOnly_sum + sparsity_ratio_granularity=block + max_sparsity_per_layer=0.6
# 单侧剪时「整体」仅在 T5 内部做 block 间分配（不与 ViT 抢预算）。
#
# 默认仅使用本机 HF hub 快照 + 本地 BLIP2 pth（与 run_ecoflap_split_merge_eval_fourbench.sh 一致）。
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

# --- HuggingFace / Transformers：离线，BERT & Flan-T5 从本地 snapshot 加载 ---
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
mkdir -p "${HF_HOME}/hub" "${HF_HOME}/transformers" 2>/dev/null || true
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
  echo "[FATAL] 未找到 bert-base-uncased 本地目录。请设置 HF_HOME 或 export BERT_BASE_UNCASED_SNAPSHOT=.../hub/models--bert-base-uncased/snapshots/<hash>" >&2
  exit 1
fi
if [[ ! -d "$FLAN_T5_XL_SNAPSHOT" ]]; then
  echo "[FATAL] 未找到 google/flan-t5-xl 本地目录。请设置 HF_HOME 或 export FLAN_T5_XL_SNAPSHOT=..." >&2
  exit 1
fi

# BLIP-2 聚合权重（覆盖 cfg 里默认的 GCS URL，避免联网下载）
BLIP2_PRETRAINED="${BLIP2_PRETRAINED:-/root/autodl-tmp/cache_moved/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth}"
if [[ ! -f "$BLIP2_PRETRAINED" ]]; then
  echo "[FATAL] 未找到本地 BLIP2 预训练权重: $BLIP2_PRETRAINED （请下载或 export BLIP2_PRETRAINED=你的路径）" >&2
  exit 1
fi

echo "[INFO] HF_HOME=$HF_HOME"
echo "[INFO] BERT_BASE_UNCASED_SNAPSHOT=$BERT_BASE_UNCASED_SNAPSHOT"
echo "[INFO] FLAN_T5_XL_SNAPSHOT=$FLAN_T5_XL_SNAPSHOT"
echo "[INFO] BLIP2_PRETRAINED=$BLIP2_PRETRAINED"

CFG="${CFG:-${REPO}/lavis/projects/blip2/eval/t5_c4_text_prune_calib.yaml}"
C4_JSON="${C4_JSON:-/data/data2/mfs/c4_calib_128.json}"
T5_SPEC="${T5_SPEC:-24-0.5-1.0-1.0}"
JOB_ID="${JOB_ID:-ecoflap_separate_t5_only}"
NUM_DATA="${NUM_DATA:-128}"
BS="${BS:-8}"
SCORE_METHOD="${SCORE_METHOD:-MEZO-GradOnly_sum}"
SPARSITY_GRANULARITY="${SPARSITY_GRANULARITY:-block}"
MAX_SPARSITY_PER_LAYER="${MAX_SPARSITY_PER_LAYER:-0.6}"

cd "${REPO}"

if [[ ! -f "${C4_JSON}" ]]; then
  echo "missing C4 json: ${C4_JSON}" >&2
  exit 1
fi

python evaluate_blip.py \
  --cfg-path "${CFG}" \
  --options model.pretrained="${BLIP2_PRETRAINED}" \
  --prune_calib_mode t5_c4_text \
  --c4_calib_json "${C4_JSON}" \
  --no_prune_vit \
  --pruning_method blipt5_wanda_pruner \
  --score_method "${SCORE_METHOD}" \
  --sparsity_ratio_granularity "${SPARSITY_GRANULARITY}" \
  --max_sparsity_per_layer "${MAX_SPARSITY_PER_LAYER}" \
  --t5_prune_spec "${T5_SPEC}" \
  --num_data "${NUM_DATA}" \
  --prunining_dataset_batch_size "${BS}" \
  --num_data_first_stage "${NUM_DATA}" \
  --job_id "${JOB_ID}" \
  --save_pruned_model \
  "$@"
