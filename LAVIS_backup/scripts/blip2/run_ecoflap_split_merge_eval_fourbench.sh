#!/usr/bin/env bash
# =============================================================================
# T5 单侧剪枝 + ViT 单侧剪枝 → 合并（可选）→ MMBench / MMMU / OKVQA / MathVista
#
# 在 LAVIS 根目录执行。依赖数据目录（parquet / json）与 HF 本地快照。
#
# 环境变量（常用）:
#   CKPT_T5_ONLY / CKPT_VIT_ONLY / MERGED_CKPT
#   RUN_MERGE=1（默认）|0    RUN_EVAL=1（默认）|0
#   SKIP_MMBENCH / SKIP_MMMU / SKIP_OKVQA / SKIP_MATHVISTA =1 可跳过单项
#   JOINT_SINGLE_CKPT=...   若设则视为「单文件联合剪枝」，跳过 merge 与双文件逻辑
#   MMBENCH_ROOT=/data/data2/mfs/MMBench_eval（含 dev-* 等 parquet，如 Other/）
#   MMMU_ROOT=/data/data2/mfs/MMMU_single_image
#   MATHVISTA_EVAL_JSON=/data/data2/mfs/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json
#   默认即以上路径；若目录不存在会再尝试 datasets/、$HOME、autodl-tmp 等。
#   ECOFLAP_BENCH_ROOT=/data/data2/mfs  可一键设置上述三者（子目录名: MMBench_eval / MMMU_single_image / MathVista_eval_testmini_mc）
#   MMMU_SPLIT 默认 test：遍历各学科下 test-*.parquet 中的单图题（全量 single-image test）
#   MODEL_CACHE_ROOT / ECOFLAP_ENV（见 run_sparsegpt_unimodal_split_no_granularity.sh）
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-/data/data2/mfs/model_cache}"
ECOFLAP_ENV="${ECOFLAP_ENV:-${MODEL_CACHE_ROOT}/ecoflap_model_env.sh}"
if [[ -f "${ECOFLAP_ENV}" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "${ECOFLAP_ENV}"
  set -u
fi

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

export TORCH_HOME="${TORCH_HOME:-${MODEL_CACHE_ROOT}/torch}"
mkdir -p "${TORCH_HOME}/hub/checkpoints" 2>/dev/null || true
if [[ -z "${EVA_VIT_G_PTH:-}" ]]; then
  if [[ -f "${TORCH_HOME}/hub/checkpoints/eva_vit_g.pth" ]]; then
    export EVA_VIT_G_PTH="${TORCH_HOME}/hub/checkpoints/eva_vit_g.pth"
  elif [[ -f "${HOME}/.cache/torch/hub/checkpoints/eva_vit_g.pth" ]]; then
    export EVA_VIT_G_PTH="${HOME}/.cache/torch/hub/checkpoints/eva_vit_g.pth"
  fi
fi

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

_pick_existing_dir() {
  for _d in "$@"; do
    [[ -n "$_d" && -d "$_d" ]] || continue
    echo "$_d"
    return 0
  done
  return 1
}

_pick_existing_file() {
  for _f in "$@"; do
    [[ -n "$_f" && -f "$_f" ]] || continue
    echo "$_f"
    return 0
  done
  return 1
}

ECOFLAP_BENCH_ROOT="${ECOFLAP_BENCH_ROOT:-}"
if [[ -n "$ECOFLAP_BENCH_ROOT" ]]; then
  export MMBENCH_ROOT="${MMBENCH_ROOT:-$ECOFLAP_BENCH_ROOT/MMBench_eval}"
  export MMMU_ROOT="${MMMU_ROOT:-$ECOFLAP_BENCH_ROOT/MMMU_single_image}"
  export MATHVISTA_EVAL_JSON="${MATHVISTA_EVAL_JSON:-$ECOFLAP_BENCH_ROOT/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json}"
fi

# 本机评测数据（用户约定在 /data/data2/mfs/ 下；MMMU 默认 split=test = 各学科 test-*.parquet 内全部单图题）
MMBENCH_ROOT="${MMBENCH_ROOT:-/data/data2/mfs/MMBench_eval}"
MMMU_ROOT="${MMMU_ROOT:-/data/data2/mfs/MMMU_single_image}"
MATHVISTA_EVAL_JSON="${MATHVISTA_EVAL_JSON:-/data/data2/mfs/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json}"

if [[ ! -d "${MMBENCH_ROOT:-}" ]]; then
  if _d="$(_pick_existing_dir \
    /data/data2/mfs/MMBench_eval \
    /data/data2/mfs/datasets/MMBench_eval \
    "${HOME}/MMBench_eval" \
    /root/autodl-tmp/MMBench_eval)"; then
    MMBENCH_ROOT="$_d"
    echo "[INFO] MMBENCH_ROOT 自动选用: $MMBENCH_ROOT"
  fi
fi

if [[ ! -d "${MMMU_ROOT:-}" ]]; then
  if _d="$(_pick_existing_dir \
    /data/data2/mfs/MMMU_single_image \
    /data/data2/mfs/datasets/MMMU_single_image \
    "${HOME}/MMMU_single_image" \
    /root/autodl-tmp/MMMU_single_image)"; then
    MMMU_ROOT="$_d"
    echo "[INFO] MMMU_ROOT 自动选用: $MMMU_ROOT"
  fi
fi

if [[ ! -f "${MATHVISTA_EVAL_JSON:-}" ]]; then
  if _f="$(_pick_existing_file \
    /data/data2/mfs/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json \
    /data/data2/mfs/datasets/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json \
    "${HOME}/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json" \
    /root/autodl-tmp/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json)"; then
    MATHVISTA_EVAL_JSON="$_f"
    echo "[INFO] MATHVISTA_EVAL_JSON 自动选用: $MATHVISTA_EVAL_JSON"
  fi
fi

export MMBENCH_ROOT MMMU_ROOT MATHVISTA_EVAL_JSON
echo "[INFO] 四评测数据: MMBENCH_ROOT=${MMBENCH_ROOT:-<无>} MMMU_ROOT=${MMMU_ROOT:-<无>} MATHVISTA_EVAL_JSON=${MATHVISTA_EVAL_JSON:-<无>}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
export MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
export MMMU_SPLIT="${MMMU_SPLIT:-test}"

MASTER_PORT="${MASTER_PORT:-29600}"
if [[ -n "${JOINT_SINGLE_CKPT:-}" ]]; then
  EVAL_JOB_OKVQA="${EVAL_JOB_OKVQA:-okvqa_eval_joint_${STEM_T5}}"
else
  EVAL_JOB_OKVQA="${EVAL_JOB_OKVQA:-okvqa_eval_merged_split_${STEM_T5}__${STEM_VIT}}"
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

if [[ ! -d "$BERT_BASE_UNCASED_SNAPSHOT" ]]; then
  echo "[FATAL] 未找到 bert-base-uncased 本地快照。请设置 HF_HOME（当前: $HF_HOME）或 BERT_BASE_UNCASED_SNAPSHOT" >&2
  exit 1
fi
if [[ ! -d "$FLAN_T5_XL_SNAPSHOT" ]]; then
  echo "[FATAL] 未找到 google/flan-t5-xl 本地快照。请设置 HF_HOME 或 FLAN_T5_XL_SNAPSHOT" >&2
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

if [[ -n "${JOINT_SINGLE_CKPT:-}" ]]; then
  CKPT_FOR_SINGLE="$JOINT_SINGLE_CKPT"
elif [[ "$RUN_MERGE" == "1" ]]; then
  CKPT_FOR_SINGLE="$MERGED_CKPT"
else
  CKPT_FOR_SINGLE=""
fi

OKVQA_CFG="${OKVQA_CFG:-$REPO_ROOT/lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml}"

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
  echo "========== OKVQA zeroshot overall ($OKVQA_CFG) =========="
  local P=$MASTER_PORT
  MASTER_PORT=$((MASTER_PORT + 1))
  if [[ -n "$CKPT_FOR_SINGLE" ]] && [[ -f "$CKPT_FOR_SINGLE" ]]; then
    python -m torch.distributed.run --nproc_per_node=1 --master_port="$P" evaluate_blip.py \
      --cfg-path "$OKVQA_CFG" \
      --t5_pruned_checkpoint "$CKPT_FOR_SINGLE" \
      --vit_pruned_checkpoint "$CKPT_FOR_SINGLE" \
      --job_id "$EVAL_JOB_OKVQA"
  else
    python -m torch.distributed.run --nproc_per_node=1 --master_port="$P" evaluate_blip.py \
      --cfg-path "$OKVQA_CFG" \
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
  _eval_preflight_failed=0
  if [[ "${SKIP_MMBENCH:-0}" != "1" ]] && [[ ! -d "${MMBENCH_ROOT:-}" ]]; then
    echo "[FATAL] MMBench 数据目录不存在: MMBENCH_ROOT=${MMBENCH_ROOT:-<空>}" >&2
    echo "      请准备 MMBench_eval（含各 split 的 parquet），然后 export MMBENCH_ROOT=/path" >&2
    echo "      或 export ECOFLAP_BENCH_ROOT=/path/to/parent，或 SKIP_MMBENCH=1" >&2
    _eval_preflight_failed=1
  fi
  if [[ "${SKIP_MMMU:-0}" != "1" ]] && [[ ! -d "${MMMU_ROOT:-}" ]]; then
    echo "[FATAL] MMMU 数据目录不存在: MMMU_ROOT=${MMMU_ROOT:-<空>}" >&2
    echo "      请设置 MMMU_ROOT 或 ECOFLAP_BENCH_ROOT，或 SKIP_MMMU=1" >&2
    _eval_preflight_failed=1
  fi
  if [[ "${SKIP_MATHVISTA:-0}" != "1" ]] && [[ ! -f "${MATHVISTA_EVAL_JSON:-}" ]]; then
    echo "[FATAL] MathVista 评测 JSON 不存在: MATHVISTA_EVAL_JSON=${MATHVISTA_EVAL_JSON:-<空>}" >&2
    echo "      请设置 MATHVISTA_EVAL_JSON 或 ECOFLAP_BENCH_ROOT，或 SKIP_MATHVISTA=1" >&2
    _eval_preflight_failed=1
  fi
  if [[ "${SKIP_OKVQA:-0}" != "1" ]] && [[ ! -f "${OKVQA_CFG:-}" ]]; then
    echo "[FATAL] OKVQA 配置不存在: OKVQA_CFG=${OKVQA_CFG:-}" >&2
    _eval_preflight_failed=1
  fi
  if [[ "$_eval_preflight_failed" -ne 0 ]]; then
    exit 1
  fi

  mkdir -p "$REPO_ROOT/training_statistics"
  export LAVIS_METRICS_JSONL="${LAVIS_METRICS_JSONL:-$REPO_ROOT/training_statistics/sparsegpt_split_fourbench_metrics.jsonl}"
  if [[ -n "${JOINT_SINGLE_CKPT:-}" ]]; then
    export LAVIS_EVAL_CALIB_TAG="joint_${STEM_T5}"
  else
    export LAVIS_EVAL_CALIB_TAG="merged_split_${STEM_T5}__${STEM_VIT}"
  fi
  echo "[INFO] LAVIS_METRICS_JSONL=$LAVIS_METRICS_JSONL"
  if [[ "${SKIP_MMBENCH:-0}" != "1" ]]; then run_mmbench; else echo "[SKIP] MMBench"; fi
  if [[ "${SKIP_MMMU:-0}" != "1" ]]; then run_mmmu; else echo "[SKIP] MMMU"; fi
  if [[ "${SKIP_OKVQA:-0}" != "1" ]]; then run_okvqa; else echo "[SKIP] OKVQA"; fi
  if [[ "${SKIP_MATHVISTA:-0}" != "1" ]]; then run_mathvista; else echo "[SKIP] MathVista"; fi
  echo ""
  echo "[INFO] 四基准评测结束。"
else
  echo "[INFO] RUN_EVAL=0，跳过评测。"
fi
