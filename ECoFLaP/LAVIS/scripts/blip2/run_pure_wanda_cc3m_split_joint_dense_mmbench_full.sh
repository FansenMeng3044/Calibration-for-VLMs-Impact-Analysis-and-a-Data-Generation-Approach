#!/usr/bin/env bash
# =============================================================================
# 纯 Wanda | CC3M 标定 | 三个模型在「MMBench 全量」上真实评测：
#   1) 全精度 dense（不加载任何剪枝权重）
#   2) CC3M 联合剪枝（ViT+T5 一起剪，单个整模 pth）
#   3) CC3M 分开剪枝（ViT-only pth + T5-only pth，评测时组合加载）
#
# 「纯 Wanda」= blipt5_wanda_pruner，不传 --sparsity_ratio_granularity、不传 MEZO/block，
#   即均匀稀疏的原始 Wanda（|W|·‖X‖），CC3M 多模态标定。
#
# MMBench「全量」= 不传 --max_samples（默认跑完所有图），不加 --overall_only（输出各学科细分）。
# 评测复用 scripts/blip2/mmmu_eval_by_discipline.py（--mmmu_root=$MMBENCH_ROOT --split=dev 即 MMBench）。
#
# 用法：
#   cd /data/data2/mfs/2/ECoFLaP/LAVIS
#   bash scripts/blip2/run_pure_wanda_cc3m_split_joint_dense_mmbench_full.sh
# 只评测已有权重（跳过剪枝，需设 JOB_STAMP 与剪枝时一致，或用 CKPT_* 直接指定）：
#   RUN_PRUNE=0 JOB_STAMP=<STAMP> bash scripts/.../run_pure_wanda_cc3m_split_joint_dense_mmbench_full.sh
# 只跑其中某些模型：
#   MODELS="dense joint" bash ...          # 可选 dense / joint / split
#
# 主要环境变量：
#   BASE, RUN_PRUNE/RUN_EVAL(默认1), MODELS(默认 "dense joint split"), JOB_STAMP
#   CC3M_CFG, T5_SPEC(24-0.5-1.0-1.0), VIT_SPEC(39-0.5-1.0-1.0), NUM_DATA(128), BS(8), SEED(42)
#   CKPT_JOINT / CKPT_VIT / CKPT_T5   直接指定已有权重（覆盖按 JOB_STAMP 推导的路径）
#   MMBENCH_ROOT, MMBENCH_SPLIT(dev), EVAL_BATCH_SIZE(2), MAX_SAMPLES(空=全量)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

BASE="${BASE:-/data/data2/mfs}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1

export HF_HOME="${HF_HOME:-$BASE/model_cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

_resolve_hub_snapshot_dir() {
  local repo_dir="$1"
  [[ -d "$repo_dir/snapshots" ]] || { echo ""; return 0; }
  find "$repo_dir/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | head -1
}
HUB_ROOT="${HUGGINGFACE_HUB_CACHE}"
[[ -z "${BERT_BASE_UNCASED_SNAPSHOT:-}" ]] && BERT_BASE_UNCASED_SNAPSHOT="$(_resolve_hub_snapshot_dir "$HUB_ROOT/models--bert-base-uncased")"
[[ -z "${FLAN_T5_XL_SNAPSHOT:-}" ]] && FLAN_T5_XL_SNAPSHOT="$(_resolve_hub_snapshot_dir "$HUB_ROOT/models--google--flan-t5-xl")"
export BERT_BASE_UNCASED_SNAPSHOT FLAN_T5_XL_SNAPSHOT

BLIP2_PRETRAINED="${BLIP2_PRETRAINED:-$BASE/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth}"
[[ -f "${BLIP2_PRETRAINED}" ]] || { echo "[FATAL] 未找到 BLIP2_PRETRAINED: ${BLIP2_PRETRAINED}" >&2; exit 1; }

# ---- 标定 & 剪枝超参 ----
CC3M_CFG="${CC3M_CFG:-lavis/projects/blip2/eval/cc_prefix_derivative_compute_cc3m_calib128.yaml}"
[[ -f "$REPO_ROOT/$CC3M_CFG" || -f "$CC3M_CFG" ]] || { echo "[FATAL] 找不到 CC3M cfg: $CC3M_CFG（用 CC3M_CFG=... 指定）" >&2; exit 1; }
T5_SPEC="${T5_SPEC:-24-0.5-1.0-1.0}"
VIT_SPEC="${VIT_SPEC:-39-0.5-1.0-1.0}"
NUM_DATA="${NUM_DATA:-128}"
NUM_DATA_FIRST_STAGE="${NUM_DATA_FIRST_STAGE:-$NUM_DATA}"
BS="${BS:-8}"
SEED="${SEED:-42}"
export LAVIS_DISTRIBUTED_SAMPLER_SEED="${LAVIS_DISTRIBUTED_SAMPLER_SEED:-$SEED}"

# ---- 评测资源 ----
MMBENCH_ROOT="${MMBENCH_ROOT:-$BASE/MMBench_eval}"
MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-}"           # 空 = 全量；调试可设小数字
MASTER_PORT_START="${MASTER_PORT_START:-29841}"

RUN_PRUNE="${RUN_PRUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
MODELS="${MODELS:-dense joint split}"
JOB_STAMP="${JOB_STAMP:-$(date +%Y%m%d_%H%M%S)}"

# ---- 权重路径（按 JOB_STAMP 推导，可用 CKPT_* 覆盖）----
JID_JOINT="pure_wanda_cc3m_joint_${JOB_STAMP}_seed${SEED}"
JID_VIT="pure_wanda_cc3m_vitonly_${JOB_STAMP}_seed${SEED}"
JID_T5="pure_wanda_cc3m_t5only_${JOB_STAMP}_seed${SEED}"
CKPT_JOINT="${CKPT_JOINT:-$REPO_ROOT/pruned_checkpoint/${JID_JOINT}.pth}"
CKPT_VIT="${CKPT_VIT:-$REPO_ROOT/pruned_checkpoint/${JID_VIT}.pth}"
CKPT_T5="${CKPT_T5:-$REPO_ROOT/pruned_checkpoint/${JID_T5}.pth}"

SUMMARY_DIR="$REPO_ROOT/lavis/output/BLIP2"
mkdir -p "$SUMMARY_DIR"
SUMMARY_LOG="$SUMMARY_DIR/mmbench_full_pure_wanda_cc3m_${JOB_STAMP}_seed${SEED}.log"

_wants() { case " $MODELS " in *" $1 "*) return 0;; *) return 1;; esac; }

# =============================== 剪枝 ===============================
prune_joint() {
  echo ""; echo ">>> [剪枝] 纯 Wanda | CC3M | 联合 ViT+T5 → $CKPT_JOINT"
  python -m torch.distributed.run --nproc_per_node=1 --master_port "$1" evaluate_blip.py \
    --cfg-path "$CC3M_CFG" \
    --options model.pretrained="${BLIP2_PRETRAINED}" \
    --pruning_method blipt5_wanda_pruner \
    --save_pruned_model \
    --prunining_dataset_batch_size "$BS" \
    --num_data "$NUM_DATA" --num_data_first_stage "$NUM_DATA_FIRST_STAGE" \
    --t5_prune_spec "$T5_SPEC" --vit_prune_spec "$VIT_SPEC" \
    --prune_vit \
    --job_id "$JID_JOINT"
  [[ -f "$CKPT_JOINT" ]] || { echo "[FATAL] 联合剪枝后未找到: $CKPT_JOINT" >&2; exit 1; }
}
prune_split_vit() {
  echo ""; echo ">>> [剪枝] 纯 Wanda | CC3M | 只剪 ViT（--no_prune_t5）→ $CKPT_VIT"
  python -m torch.distributed.run --nproc_per_node=1 --master_port "$1" evaluate_blip.py \
    --cfg-path "$CC3M_CFG" \
    --options model.pretrained="${BLIP2_PRETRAINED}" \
    --pruning_method blipt5_wanda_pruner \
    --save_pruned_model \
    --prunining_dataset_batch_size "$BS" \
    --num_data "$NUM_DATA" --num_data_first_stage "$NUM_DATA_FIRST_STAGE" \
    --t5_prune_spec "$T5_SPEC" --vit_prune_spec "$VIT_SPEC" \
    --no_prune_t5 \
    --job_id "$JID_VIT"
  [[ -f "$CKPT_VIT" ]] || { echo "[FATAL] ViT-only 剪枝后未找到: $CKPT_VIT" >&2; exit 1; }
}
prune_split_t5() {
  echo ""; echo ">>> [剪枝] 纯 Wanda | CC3M | 只剪 T5（默认）→ $CKPT_T5"
  python -m torch.distributed.run --nproc_per_node=1 --master_port "$1" evaluate_blip.py \
    --cfg-path "$CC3M_CFG" \
    --options model.pretrained="${BLIP2_PRETRAINED}" \
    --pruning_method blipt5_wanda_pruner \
    --save_pruned_model \
    --prunining_dataset_batch_size "$BS" \
    --num_data "$NUM_DATA" --num_data_first_stage "$NUM_DATA_FIRST_STAGE" \
    --t5_prune_spec "$T5_SPEC" --vit_prune_spec "$VIT_SPEC" \
    --job_id "$JID_T5"
  [[ -f "$CKPT_T5" ]] || { echo "[FATAL] T5-only 剪枝后未找到: $CKPT_T5" >&2; exit 1; }
}

# =============================== MMBench 全量评测 ===============================
# 用法: mmbench_full <tag> <ckpt-args...>
mmbench_full() {
  local tag="$1"; shift
  export LAVIS_METRICS_BENCHMARK="MMBench"
  export LAVIS_EVAL_CALIB_TAG="$tag"
  local cap=()
  [[ -n "$MAX_SAMPLES" ]] && cap=(--max_samples "$MAX_SAMPLES")
  echo "" | tee -a "$SUMMARY_LOG"
  echo "========== MMBench 全量 | 模型=$tag | split=$MMBENCH_SPLIT ==========" | tee -a "$SUMMARY_LOG"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMBENCH_ROOT" --split "$MMBENCH_SPLIT" \
    "$@" \
    --batch_size "$EVAL_BATCH_SIZE" --device cuda \
    "${cap[@]}" 2>&1 | tee -a "$SUMMARY_LOG"
}

# =============================== 主流程 ===============================
echo "========== 纯 Wanda CC3M | dense/joint/split → MMBench 全量 | STAMP=$JOB_STAMP SEED=$SEED ==========" | tee "$SUMMARY_LOG"
echo "[INFO] MODELS=$MODELS  RUN_PRUNE=$RUN_PRUNE RUN_EVAL=$RUN_EVAL  MAX_SAMPLES=${MAX_SAMPLES:-<全量>}" | tee -a "$SUMMARY_LOG"
echo "[INFO] CC3M_CFG=$CC3M_CFG  T5_SPEC=$T5_SPEC VIT_SPEC=$VIT_SPEC" | tee -a "$SUMMARY_LOG"
echo "[INFO] joint=$CKPT_JOINT" | tee -a "$SUMMARY_LOG"
echo "[INFO] split=vit:$CKPT_VIT + t5:$CKPT_T5" | tee -a "$SUMMARY_LOG"

if [[ "$RUN_PRUNE" == "1" ]]; then
  p=$MASTER_PORT_START
  if _wants joint; then prune_joint "$p"; p=$((p+1)); fi
  if _wants split; then prune_split_vit "$p"; p=$((p+1)); prune_split_t5 "$p"; p=$((p+1)); fi
else
  echo "[INFO] RUN_PRUNE=0 — 使用已有权重"
fi

if [[ "$RUN_EVAL" != "1" ]]; then
  echo "[INFO] RUN_EVAL!=1 — 跳过评测"; exit 0
fi

# 1) dense 全精度：不传任何 ckpt → 用 LAVIS 预训练权重
if _wants dense; then
  mmbench_full "dense_fullprec_${JOB_STAMP}"
fi

# 2) 联合剪枝：单 ckpt
if _wants joint; then
  [[ -f "$CKPT_JOINT" ]] || { echo "[FATAL] 缺少联合权重: $CKPT_JOINT" >&2; exit 1; }
  mmbench_full "cc3m_joint_wanda_${JOB_STAMP}" --ckpt "$CKPT_JOINT"
fi

# 3) 分开剪枝：ViT-only + T5-only 组合
if _wants split; then
  [[ -f "$CKPT_VIT" && -f "$CKPT_T5" ]] || { echo "[FATAL] 缺少分开权重: $CKPT_VIT | $CKPT_T5" >&2; exit 1; }
  mmbench_full "cc3m_split_wanda_${JOB_STAMP}" --vit_ckpt "$CKPT_VIT" --t5_ckpt "$CKPT_T5"
fi

echo "" | tee -a "$SUMMARY_LOG"
echo "========== 全部完成 ==========" | tee -a "$SUMMARY_LOG"
echo "[INFO] 汇总日志: $SUMMARY_LOG" | tee -a "$SUMMARY_LOG"
