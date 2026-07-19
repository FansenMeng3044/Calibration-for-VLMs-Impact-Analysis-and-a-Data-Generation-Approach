#!/usr/bin/env bash
# =============================================================================
# C4 纯文本标定 → blipt5_tamp_pruner 退化为 vanilla Wanda、只剪 T5（llm-only）→ 保存 pth → 四基准评测；
# 再用既有 CC3M TAMP T5-only 权重再跑一遍四基准评测。
#
# 默认路径面向本机 /data/data2/mfs（可用环境变量覆盖）。
#
# 用法:
#   cd /data/data2/mfs/2/LAVIS_backup
#   bash scripts/blip2/run_c4_tamp_llmonly_prune_then_dual_fourbench_eval.sh
#
# 仅评测（已剪好 C4 权重）:
#   RUN_PRUNE=0 JOB_ID=你的job_id bash scripts/blip2/run_c4_tamp_llmonly_prune_then_dual_fourbench_eval.sh
#
# 环境变量:
#   BASE, RUN_PRUNE(默认1), RUN_EVAL(默认1), JOB_STAMP, JOB_ID
#   C4_JSON, CC3M_TAMP_CKPT（第二套评测用）
#   MMBENCH_ROOT, MMMU_ROOT, MATHVISTA_EVAL_JSON, MATHVISTA_IMAGES_DIR, EVAL_BATCH_SIZE
#   CUDA_VISIBLE_DEVICES, HF_HOME, BLIP2_PRETRAINED
#   MASTER_PORT_C4 / MASTER_PORT_CC3M — OKVQA 分布式端口（默认 29750 / 29760）
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

BASE="${BASE:-/data/data2/mfs}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export HF_HOME="${HF_HOME:-$BASE/model_cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

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
  echo "[FATAL] 未找到 bert-base-uncased 本地 snapshot。请设置 HF_HOME 或 BERT_BASE_UNCASED_SNAPSHOT。" >&2
  exit 1
fi
if [[ ! -d "${FLAN_T5_XL_SNAPSHOT}" ]]; then
  echo "[FATAL] 未找到 flan-t5-xl 本地 snapshot。请设置 HF_HOME 或 FLAN_T5_XL_SNAPSHOT。" >&2
  exit 1
fi

BLIP2_PRETRAINED="${BLIP2_PRETRAINED:-$BASE/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth}"
if [[ ! -f "${BLIP2_PRETRAINED}" ]]; then
  echo "[FATAL] 未找到 BLIP2_PRETRAINED: ${BLIP2_PRETRAINED}" >&2
  exit 1
fi

CFG="${CFG:-$REPO_ROOT/lavis/projects/blip2/eval/t5_c4_text_prune_calib.yaml}"
C4_JSON="${C4_JSON:-$BASE/c4_calib_128.json}"
JOB_STAMP="${JOB_STAMP:-$(date +%Y%m%d_%H%M%S)}"
JOB_ID="${JOB_ID:-tamp_c4_llmonly_${JOB_STAMP}}"
CC3M_TAMP_CKPT="${CC3M_TAMP_CKPT:-$REPO_ROOT/pruned_checkpoint/calib-cc_prefix_derivative_compute_cc3m_calib128__TAMP__t5_only__t20260504_192855__seed42.pth}"

T5_SPEC="${T5_SPEC:-24-0.5-1.0-1.0}"
NUM_DATA="${NUM_DATA:-128}"
BS="${BS:-8}"
PRUNE_METHOD="${PRUNE_METHOD:-blipt5_tamp_pruner}"
SCORE_METHOD="${SCORE_METHOD:-MEZO-GradOnly_sum}"
SPARSITY_GRANULARITY="${SPARSITY_GRANULARITY:-block}"
MAX_SPARSITY_PER_LAYER="${MAX_SPARSITY_PER_LAYER:-0.6}"

RUN_PRUNE="${RUN_PRUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

MMBENCH_ROOT="${MMBENCH_ROOT:-$BASE/MMBench_eval}"
MMMU_ROOT="${MMMU_ROOT:-$BASE/MMMU_single_image}"
MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
MMMU_SPLIT="${MMMU_SPLIT:-test}"
MATHVISTA_EVAL_JSON="${MATHVISTA_EVAL_JSON:-$BASE/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json}"
MATHVISTA_IMAGES_DIR="${MATHVISTA_IMAGES_DIR:-$BASE/MathVista_eval_testmini_mc/images}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
MASTER_PORT_C4="${MASTER_PORT_C4:-29750}"
MASTER_PORT_CC3M="${MASTER_PORT_CC3M:-29760}"

C4_CKPT="$REPO_ROOT/pruned_checkpoint/${JOB_ID}.pth"

four_bench_eval() {
  local ckpt="$1"
  local eval_tag="$2"
  local okvqa_job="$3"
  local master_port="$4"

  if [[ ! -f "$ckpt" ]]; then
    echo "[FATAL] 找不到权重: $ckpt" >&2
    exit 1
  fi

  export LAVIS_EVAL_CALIB_TAG="$eval_tag"
  echo ""
  echo "========== 四基准评测 | tag=$eval_tag =========="
  echo "[INFO] CKPT=$(readlink -f "$ckpt")"
  echo "[INFO] LAVIS_EVAL_CALIB_TAG=$LAVIS_EVAL_CALIB_TAG OKVQA_JOB_ID=$okvqa_job master_port=$master_port"

  echo ""
  echo ">>> MMBench ($MMBENCH_ROOT, split=$MMBENCH_SPLIT)"
  export LAVIS_METRICS_BENCHMARK="MMBench"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMBENCH_ROOT" \
    --split "$MMBENCH_SPLIT" \
    --ckpt "$ckpt" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --device cuda \
    --overall_only

  echo ""
  echo ">>> OKVQA full val (--job_id $okvqa_job)"
  python -m torch.distributed.run --nproc_per_node=1 --master_port="$master_port" evaluate_blip.py \
    --cfg-path lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml \
    --t5_pruned_checkpoint "$ckpt" \
    --vit_pruned_checkpoint "$ckpt" \
    --job_id "$okvqa_job"

  echo ""
  echo ">>> MMMU ($MMMU_ROOT, split=$MMMU_SPLIT)"
  export LAVIS_METRICS_BENCHMARK="MMMU"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMMU_ROOT" \
    --split "$MMMU_SPLIT" \
    --ckpt "$ckpt" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --device cuda \
    --overall_only

  echo ""
  echo ">>> MathVista MC"
  if [[ ! -f "$MATHVISTA_EVAL_JSON" ]]; then
    echo "[WARN] 跳过 MathVista：缺少 $MATHVISTA_EVAL_JSON"
  else
    export LAVIS_METRICS_BENCHMARK="MathVista_MC"
    python scripts/blip2/mathvista_mc_eval.py \
      --eval_json "$MATHVISTA_EVAL_JSON" \
      --images_dir "$MATHVISTA_IMAGES_DIR" \
      --ckpt "$ckpt" \
      --batch_size "$EVAL_BATCH_SIZE" \
      --device cuda
  fi
  echo "[INFO] 四基准评测结束: $eval_tag"
}

echo "========== C4 TAMP-degenerated Wanda llm-only 剪枝 + 双四基准评测 | STAMP=$JOB_STAMP =========="
echo "[INFO] REPO_ROOT=$REPO_ROOT"
echo "[INFO] C4_JSON=$C4_JSON JOB_ID=$JOB_ID → $C4_CKPT"
echo "[INFO] CC3M_TAMP_CKPT=$CC3M_TAMP_CKPT"
echo "[INFO] RUN_PRUNE=$RUN_PRUNE RUN_EVAL=$RUN_EVAL"
echo "================================================================"

if [[ ! -f "$C4_JSON" ]]; then
  echo "[FATAL] 缺少 C4 校准 JSON: $C4_JSON" >&2
  exit 1
fi

if [[ "$RUN_PRUNE" == "1" ]]; then
  echo ""
  echo ">>> 剪枝: C4 标定 + TAMP-degenerated vanilla Wanda + 只剪 T5"
  EXTRA=()
  if [[ "${T5_C4_ENCODER_ONLY:-0}" == "1" ]]; then
    EXTRA+=(--t5_c4_encoder_only)
  fi
  python evaluate_blip.py \
    --cfg-path "${CFG}" \
    --options "model.pretrained=${BLIP2_PRETRAINED}" \
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
    "${EXTRA[@]}"
  if [[ ! -f "$C4_CKPT" ]]; then
    echo "[FATAL] 剪枝后未找到: $C4_CKPT" >&2
    exit 1
  fi
else
  echo "[INFO] RUN_PRUNE=0 — 跳过剪枝，使用已有: $C4_CKPT"
  if [[ ! -f "$C4_CKPT" ]]; then
    echo "[FATAL] 缺少权重，请设置正确 JOB_ID 或先 RUN_PRUNE=1" >&2
    exit 1
  fi
fi

if [[ "$RUN_EVAL" != "1" ]]; then
  echo "[INFO] RUN_EVAL!=1 — 跳过评测"
  exit 0
fi

if [[ ! -f "$CC3M_TAMP_CKPT" ]]; then
  echo "[FATAL] 第二套评测指定权重不存在: $CC3M_TAMP_CKPT" >&2
  exit 1
fi

four_bench_eval "$C4_CKPT" "t5_c4_tamp_${JOB_ID}" "okvqa_eval_t5_c4_tamp_${JOB_ID}_fullval" "$MASTER_PORT_C4"

four_bench_eval "$CC3M_TAMP_CKPT" "cc3m_tamp_t5only_seed42_20260504" "okvqa_eval_cc3m_tamp_t5only_seed42_fullval" "$MASTER_PORT_CC3M"

echo ""
echo "[INFO] 全部完成。C4 权重: $C4_CKPT | CC3M 权重: $CC3M_TAMP_CKPT"
