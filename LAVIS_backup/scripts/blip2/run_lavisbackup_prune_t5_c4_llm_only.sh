#!/usr/bin/env bash
# LAVIS_backup：C4 纯文本校准 + 只剪 T5（LLM）。
# 默认剪枝器为 blipt5_tamp_pruner；纯文本下运行 TAMP 单模态归约（s = s_l，AMIA 选文本 token，
# DAS 逐层分配）。这不是已发表的多模态 TAMP。naive+uniform 基线用 blipt5_wanda_pruner。
# 可选 Wanda：PRUNE_METHOD=blipt5_wanda_pruner bash ...
# 需本地 C4 JSON（与 ECoFLaP/scripts/run_prune_t5_c4_128.sh 相同格式）。
#
# 用法:
#   cd /root/autodl-tmp/LAVIS_backup
#   bash scripts/blip2/run_lavisbackup_prune_t5_c4_llm_only.sh
#
# 覆盖示例:
#   C4_JSON=/path/to/c4_calib_128.json JOB_ID=my_t5_c4_t5only bash scripts/blip2/run_lavisbackup_prune_t5_c4_llm_only.sh
#
# 默认 blipt5_tamp_pruner，在 t5_c4_text 下运行单模态归约；
# 若要显式指定 Wanda：
#   PRUNE_METHOD=blipt5_wanda_pruner bash scripts/blip2/run_lavisbackup_prune_t5_c4_llm_only.sh
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO}" || exit 1

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

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

if [[ ! -d "${BERT_BASE_UNCASED_SNAPSHOT}" ]]; then
  echo "[FATAL] 未找到 bert-base-uncased 本地 snapshot。请设置 HF_HOME。" >&2
  exit 1
fi
if [[ ! -d "${FLAN_T5_XL_SNAPSHOT}" ]]; then
  echo "[FATAL] 未找到 flan-t5-xl 本地 snapshot。请设置 HF_HOME。" >&2
  exit 1
fi

BLIP2_PRETRAINED="${BLIP2_PRETRAINED:-/root/autodl-tmp/cache_moved/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth}"
if [[ ! -f "${BLIP2_PRETRAINED}" ]]; then
  echo "[FATAL] 未找到 BLIP2_PRETRAINED: ${BLIP2_PRETRAINED}" >&2
  exit 1
fi

CFG="${CFG:-${REPO}/lavis/projects/blip2/eval/t5_c4_text_prune_calib.yaml}"
C4_JSON="${C4_JSON:-/root/autodl-tmp/c4_calib_128.json}"
T5_SPEC="${T5_SPEC:-24-0.5-1.0-1.0}"
JOB_ID="${JOB_ID:-lavisbackup_t5_c4_llm_only}"
NUM_DATA="${NUM_DATA:-128}"
BS="${BS:-8}"
SCORE_METHOD="${SCORE_METHOD:-MEZO-GradOnly_sum}"
SPARSITY_GRANULARITY="${SPARSITY_GRANULARITY:-block}"
MAX_SPARSITY_PER_LAYER="${MAX_SPARSITY_PER_LAYER:-0.6}"
PRUNE_METHOD="${PRUNE_METHOD:-blipt5_tamp_pruner}"

if [[ ! -f "${C4_JSON}" ]]; then
  echo "[FATAL] 缺少 C4 校准 JSON: ${C4_JSON}" >&2
  echo "  可从 ECoFLaP 仓库运行 scripts/fetch_c4_calib128_hf_mirror.py 或自备与 ECoFLaP run_prune_t5_c4_128.sh 相同格式的文件。" >&2
  exit 1
fi

echo "[INFO] REPO=$REPO"
echo "[INFO] C4_JSON=$C4_JSON JOB_ID=$JOB_ID PRUNE_METHOD=$PRUNE_METHOD"

EXTRA=()
if [[ "${T5_C4_ENCODER_ONLY:-0}" == "1" ]]; then
  EXTRA+=(--t5_c4_encoder_only)
fi

python evaluate_blip.py \
  --cfg-path "${CFG}" \
  --options model.pretrained="${BLIP2_PRETRAINED}" \
  --prune_calib_mode t5_c4_text \
  --c4_calib_json "${C4_JSON}" \
  --pruning_method "${PRUNE_METHOD}" \
  --score_method "${SCORE_METHOD}" \
  --sparsity_ratio_granularity "${SPARSITY_GRANULARITY}" \
  --max_sparsity_per_layer "${MAX_SPARSITY_PER_LAYER}" \
  --t5_prune_spec "${T5_SPEC}" \
  --num_data "${NUM_DATA}" \
  --prunining_dataset_batch_size "${BS}" \
  --num_data_first_stage "${NUM_DATA}" \
  --job_id "${JOB_ID}" \
  --save_pruned_model \
  "${EXTRA[@]}" \
  "$@"
