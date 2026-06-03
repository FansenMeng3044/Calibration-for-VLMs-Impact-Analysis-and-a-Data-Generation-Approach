#!/usr/bin/env bash
# SparseGPT 单侧剪枝（**不**设 sparsity_ratio_granularity，即与 argparse 默认 None 一致）：
#
#   ViT：--prune_calib_mode vit_image_only → importance_scope=vit_only_encode → loss_vit_encode_l2
#   T5：--prune_calib_mode t5_c4_text     → importance_scope=llm_only        → loss_language
#
# 两次剪枝各存一份完整 state_dict；评测前可用 scripts/blip2/merge_ecoflap_split_prune_ckpts.py 合并，
# 或用 evaluate_blip 同时传 --vit_pruned_checkpoint / --t5_pruned_checkpoint。
#
# 与本机校准数据对应：C4 文本 /data/data2/mfs/c4_calib_128.json ，CC3M 图
# /data/data2/mfs/CC3M_calib_128（json + images/）。可用 CC3M_ROOT / C4_JSON 覆盖。
#
# 在 ECoFLaP/LAVIS 根目录执行，例如：
#   bash scripts/blip2/run_sparsegpt_unimodal_split_no_granularity.sh vit
#   bash scripts/blip2/run_sparsegpt_unimodal_split_no_granularity.sh t5
#   bash scripts/blip2/run_sparsegpt_unimodal_split_no_granularity.sh merge   # 需先设 VIT_CKPT / T5_CKPT / MERGED_OUT
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

MODE="${1:-}"

MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-/data/data2/mfs/model_cache}"
ECOFLAP_ENV="${ECOFLAP_ENV:-${MODEL_CACHE_ROOT}/ecoflap_model_env.sh}"
if [[ -f "${ECOFLAP_ENV}" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "${ECOFLAP_ENV}"
  set -u
fi

# --- HF / Transformers：与 download_ecoflap_blip2_hf_assets.sh 一致；勿默认 ~/.cache 而忽略 mfs hub ---
if [[ -z "${HF_HOME:-}" ]]; then
  if [[ -d "${MODEL_CACHE_ROOT}/huggingface/hub" ]]; then
    export HF_HOME="${MODEL_CACHE_ROOT}/huggingface"
  else
    export HF_HOME="${HOME}/.cache/huggingface"
  fi
fi
mkdir -p "${HF_HOME}/hub" "${HF_HOME}/transformers" 2>/dev/null || true
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# --- timm / Torch Hub：LAVIS 用 timm 缓存 eva_vit_g.pth；勿只用 ~/.cache/torch ---
export TORCH_HOME="${TORCH_HOME:-${MODEL_CACHE_ROOT}/torch}"
mkdir -p "${TORCH_HOME}/hub/checkpoints" 2>/dev/null || true
if [[ -z "${EVA_VIT_G_PTH:-}" ]]; then
  if [[ -f "${TORCH_HOME}/hub/checkpoints/eva_vit_g.pth" ]]; then
    export EVA_VIT_G_PTH="${TORCH_HOME}/hub/checkpoints/eva_vit_g.pth"
  elif [[ -f "${HOME}/.cache/torch/hub/checkpoints/eva_vit_g.pth" ]]; then
    export EVA_VIT_G_PTH="${HOME}/.cache/torch/hub/checkpoints/eva_vit_g.pth"
  fi
fi

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

BLIP2_PRETRAINED="${BLIP2_PRETRAINED:-${MODEL_CACHE_ROOT}/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth}"

CFG_VIT="${CFG_VIT:-${REPO}/lavis/projects/blip2/eval/cc_prefix_derivative_compute_cc3m_calib128.yaml}"
CFG_T5="${CFG_T5:-${REPO}/lavis/projects/blip2/eval/t5_c4_text_prune_calib.yaml}"
C4_JSON="${C4_JSON:-/data/data2/mfs/c4_calib_128.json}"
CC3M_ROOT="${CC3M_ROOT:-/data/data2/mfs/CC3M_calib_128}"
CC3M_JSON="${CC3M_JSON:-${CC3M_ROOT}/cc3m_calib_128.json}"
CC3M_IMAGES="${CC3M_IMAGES:-${CC3M_ROOT}/images}"

VIT_SPEC="${VIT_SPEC:-39-0.5-1.0-1.0}"
T5_SPEC="${T5_SPEC:-24-0.5-1.0-1.0}"
NUM_DATA="${NUM_DATA:-128}"
BS="${BS:-8}"
# granularity 关闭：不要设置 SPARSITY_GRANULARITY / 不要传 --sparsity_ratio_granularity
SCORE_METHOD="${SCORE_METHOD:-MEZO-GradOnly_sum}"
MAX_SPARSITY_PER_LAYER="${MAX_SPARSITY_PER_LAYER:-0.6}"
MASTER_PORT="${MASTER_PORT:-29517}"

JOB_VIT="${JOB_VIT:-sparsegpt_vit_image_only_nogran}"
JOB_T5="${JOB_T5:-sparsegpt_t5_c4_nogran}"

cd "${REPO}"

if [[ ! -d "${REPO}/lavis/datasets" ]]; then
  echo "[FATAL] 缺少目录: ${REPO}/lavis/datasets/" >&2
  echo "当前 LAVIS 不是完整拷贝，lavis/__init__.py 会 import lavis.datasets 并失败。" >&2
  echo "请从官方 LAVIS 仓库补全 lavis/datasets（或换到含该目录的完整克隆后再跑）。" >&2
  exit 1
fi

run_vit() {
  if [[ ! -f "${CC3M_JSON}" ]]; then
    echo "[FATAL] CC3M calib json not found: ${CC3M_JSON} (set CC3M_ROOT or CC3M_JSON=...)" >&2
    exit 1
  fi
  if [[ ! -d "${CC3M_IMAGES}" ]]; then
    echo "[FATAL] CC3M images dir not found: ${CC3M_IMAGES} (set CC3M_IMAGES=...)" >&2
    exit 1
  fi
  # 仅覆盖 BLIP2 pth；CC3M 路径写在 yaml（避免 --options 里 url.0 与 OmegaConf 合并出 List/Dict 冲突）
  python -m torch.distributed.run --nproc_per_node=1 --master_port "${MASTER_PORT}" evaluate_blip.py \
    --cfg-path "${CFG_VIT}" \
    --options model.pretrained="${BLIP2_PRETRAINED}" \
    --pruning_method blipt5_sparsegpt_pruner \
    --prune_calib_mode vit_image_only \
    --importance_scope vit_only_encode \
    --no_prune_t5 \
    --vit_prune_spec "${VIT_SPEC}" \
    --score_method "${SCORE_METHOD}" \
    --max_sparsity_per_layer "${MAX_SPARSITY_PER_LAYER}" \
    --num_data "${NUM_DATA}" \
    --prunining_dataset_batch_size "${BS}" \
    --num_data_first_stage "${NUM_DATA}" \
    --job_id "${JOB_VIT}" \
    --save_pruned_model \
    "${@:2}"
}

run_t5() {
  if [[ ! -f "${C4_JSON}" ]]; then
    echo "[FATAL] C4 calib json not found: ${C4_JSON} (set C4_JSON=...)" >&2
    exit 1
  fi
  python evaluate_blip.py \
    --cfg-path "${CFG_T5}" \
    --options model.pretrained="${BLIP2_PRETRAINED}" \
    --pruning_method blipt5_sparsegpt_pruner \
    --prune_calib_mode t5_c4_text \
    --importance_scope llm_only \
    --c4_calib_json "${C4_JSON}" \
    --no_prune_vit \
    --t5_prune_spec "${T5_SPEC}" \
    --score_method "${SCORE_METHOD}" \
    --max_sparsity_per_layer "${MAX_SPARSITY_PER_LAYER}" \
    --num_data "${NUM_DATA}" \
    --prunining_dataset_batch_size "${BS}" \
    --num_data_first_stage "${NUM_DATA}" \
    --job_id "${JOB_T5}" \
    --save_pruned_model \
    "${@:2}"
}

run_merge() {
  local t5c="${T5_CKPT:?set T5_CKPT=pruned_checkpoint/<t5_job>.pth}"
  local vic="${VIT_CKPT:?set VIT_CKPT=pruned_checkpoint/<vit_job>.pth}"
  local out="${MERGED_OUT:?set MERGED_OUT=pruned_checkpoint/merged_sparsegpt_unimodal.pth}"
  python "${REPO}/scripts/blip2/merge_ecoflap_split_prune_ckpts.py" \
    --t5_ckpt "${t5c}" \
    --vit_ckpt "${vic}" \
    --out "${out}"
}

case "${MODE}" in
  vit) run_vit ;;
  t5)  run_t5 ;;
  merge) run_merge ;;
  *)
    echo "usage: $0 vit|t5|merge [extra evaluate_blip.py args...]" >&2
    exit 1
    ;;
esac
