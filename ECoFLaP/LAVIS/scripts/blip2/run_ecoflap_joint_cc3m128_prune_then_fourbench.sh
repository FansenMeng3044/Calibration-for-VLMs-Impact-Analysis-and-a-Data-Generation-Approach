#!/usr/bin/env bash
# =============================================================================
# 原版 ECoFLaP「联合剪枝」：CC3M 128 条多模态 calibration（multimodal + joint）
# → 保存单份 pruned_checkpoint/${JOB_ID}.pth → 四基准评测（MMBench / MMMU / OKVQA / MathVista）
#
# 与 yaml 注释一致：MEZO-GradOnly_sum、block、max_sparsity_per_layer=0.6、bs=8、num_data=128
# 默认加 --prune_vit，使 ViT+T5 均参与 Wanda（与仅剪 T5 的注释示例不同；若只要 T5：PRUNE_VIT=0）
#
# 依赖：CC3M_calib_128 数据、本地 HF 快照、BLIP2 pth（同其它 ecoflap 脚本）
#
# 用法:
#   cd /root/autodl-tmp/ECoFLaP/LAVIS
#   bash scripts/blip2/run_ecoflap_joint_cc3m128_prune_then_fourbench.sh
#
# 只评测（已有 joint pth）:
#   RUN_PRUNE=0 JOB_ID=ecoflap_joint_cc3m128_wanda bash scripts/blip2/run_ecoflap_joint_cc3m128_prune_then_fourbench.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# --- HuggingFace / Transformers：离线 ---
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

if [[ ! -d "$BERT_BASE_UNCASED_SNAPSHOT" ]] || [[ ! -d "$FLAN_T5_XL_SNAPSHOT" ]]; then
  echo "[FATAL] 未找到 bert-base-uncased 或 flan-t5-xl 本地 snapshot。请设置 HF_HOME。" >&2
  exit 1
fi

BLIP2_PRETRAINED="${BLIP2_PRETRAINED:-/root/autodl-tmp/cache_moved/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth}"
if [[ ! -f "$BLIP2_PRETRAINED" ]]; then
  echo "[FATAL] 未找到 BLIP2_PRETRAINED: $BLIP2_PRETRAINED" >&2
  exit 1
fi

JOB_ID="${JOB_ID:-ecoflap_joint_cc3m128_wanda}"
MASTER_PORT_PRUNE="${MASTER_PORT_PRUNE:-29507}"
NUM_DATA="${NUM_DATA:-128}"
BS="${BS:-8}"
NUM_DATA_FIRST_STAGE="${NUM_DATA_FIRST_STAGE:-32}"
T5_SPEC="${T5_SPEC:-24-0.5-1.0-1.0}"
VIT_SPEC="${VIT_SPEC:-39-0.5-1.0-1.0}"
PRUNE_VIT="${PRUNE_VIT:-1}"
RUN_PRUNE="${RUN_PRUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

CFG="${CFG:-$REPO_ROOT/lavis/projects/blip2/eval/cc_prefix_derivative_compute_cc3m_calib128.yaml}"
JOINT_CKPT="$REPO_ROOT/pruned_checkpoint/${JOB_ID}.pth"

cd "$REPO_ROOT" || exit 1

PRUNE_EXTRA=()
if [[ "$PRUNE_VIT" == "1" ]]; then
  PRUNE_EXTRA+=(--prune_vit)
fi

if [[ "$RUN_PRUNE" == "1" ]]; then
  echo "[INFO] 联合剪枝 CC3M128: JOB_ID=$JOB_ID cfg=$CFG"
  python -m torch.distributed.run --nproc_per_node=1 --master_port "$MASTER_PORT_PRUNE" evaluate_blip.py \
    --cfg-path "$CFG" \
    --options model.pretrained="${BLIP2_PRETRAINED}" \
    --pruning_method blipt5_wanda_pruner \
    --save_pruned_model \
    --score_method MEZO-GradOnly_sum \
    --sparsity_ratio_granularity block \
    --max_sparsity_per_layer 0.6 \
    --prunining_dataset_batch_size "$BS" \
    --num_data "$NUM_DATA" \
    --num_data_first_stage "$NUM_DATA_FIRST_STAGE" \
    --t5_prune_spec "$T5_SPEC" \
    --vit_prune_spec "$VIT_SPEC" \
    "${PRUNE_EXTRA[@]}" \
    --job_id "$JOB_ID" \
    "$@"
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  if [[ ! -f "$JOINT_CKPT" ]]; then
    echo "[FATAL] 找不到联合剪枝权重: $JOINT_CKPT（先 RUN_PRUNE=1 或检查 JOB_ID）" >&2
    exit 1
  fi
  export JOINT_SINGLE_CKPT="$JOINT_CKPT"
  exec bash "$SCRIPT_DIR/run_ecoflap_split_merge_eval_fourbench.sh"
fi
