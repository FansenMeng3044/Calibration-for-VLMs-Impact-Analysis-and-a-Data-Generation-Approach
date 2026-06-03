#!/usr/bin/env bash
# =============================================================================
# LAVIS_backup：五套 calibration（MMBench / MMMU / OKVQA train / MathVista / CC3M）
# + TAMP 剪枝（blipt5_tamp_pruner，默认只剪 T5）+ 每份权重跑四基准评测。
#
# DistributedSampler seed：MMBench/MMMU/OKVQA/MathVista 默认 0、30、42 各跑一轮（4×3=12 份 pth）。
# CC3M 标定文件固定 128 条，只剪枝一次 + 四基准评测（不随 seed 重复）。
#
# 权重命名（pruned_checkpoint/）:
#   tamp_calibMMBench_${RUN_STAMP}_calibseed${SEED}.pth  （SEED=0|30|42）
#   tamp_calibMMMU_... / OKVQAtrain_... / MathVista_...
#   tamp_calibCC3M_${RUN_STAMP}_calib128.pth              （仅一份）
#
# 用法:
#   cd /data/data2/mfs/2/LAVIS_backup
#   bash scripts/blip2/run_lavisbackup_tamp_fivecalib_prune_eval_fourbench_20260602_171200.sh
#
# 只跑 sampler seed 42:
#   SAMPLER_SEEDS=42 bash scripts/blip2/run_lavisbackup_tamp_fivecalib_prune_eval_fourbench_20260602_171200.sh
#
# 只剪枝、不评测:
#   RUN_EVAL=0 bash scripts/blip2/...
#
# 只评测（pth 已存在，需 RUN_STAMP 与剪枝时一致）:
#   RUN_PRUNE=0 RUN_STAMP=20260602_171200 SAMPLER_SEEDS=42 bash scripts/blip2/...
#
# 只跑部分 calibration:
#   CALIBS="mmbench okvqa cc3m" bash scripts/blip2/...
#
# 环境变量:
#   BASE, RUN_STAMP, SAMPLER_SEEDS / SEEDS, RUN_PRUNE, RUN_EVAL
#   NUM_DATA(128), BS(8), T5_SPEC, PRUNE_METHOD(blipt5_tamp_pruner)
#   BLIP2_PRETRAINED, HF_HOME, CUDA_VISIBLE_DEVICES, MASTER_PORT
#   MMBENCH_ROOT, MMMU_ROOT, MATHVISTA_EVAL_JSON, MATHVISTA_IMAGES_DIR
#   SKIP_MMBENCH / SKIP_OKVQA / SKIP_MMMU / SKIP_MATHVISTA =1（评测阶段）
#   CC3M_CALIB_SEED         CC3M 剪枝用 sampler seed（默认 42；池仅 128 条，与 seed 无关）
#   SKIP_CC3M=1             跳过 CC3M 剪枝与评测
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

BASE="${BASE:-/data/data2/mfs}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$BASE/model_cache}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# 本机已下载模型（与 model_cache/ecoflap_model_env.sh 一致）
ECOFLAP_ENV="${ECOFLAP_ENV:-${MODEL_CACHE_ROOT}/ecoflap_model_env.sh}"
if [[ -f "${ECOFLAP_ENV}" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "${ECOFLAP_ENV}"
  set -u
fi

export TORCH_HOME="${TORCH_HOME:-${MODEL_CACHE_ROOT}/torch}"
export HF_HOME="${HF_HOME:-${MODEL_CACHE_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export BERT_BASE_UNCASED_SNAPSHOT="${BERT_BASE_UNCASED_SNAPSHOT:-${MODEL_CACHE_ROOT}/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594}"
export FLAN_T5_XL_SNAPSHOT="${FLAN_T5_XL_SNAPSHOT:-${MODEL_CACHE_ROOT}/huggingface/hub/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001}"
BLIP2_PRETRAINED="${BLIP2_PRETRAINED:-${MODEL_CACHE_ROOT}/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth}"

if [[ -z "${EVA_VIT_G_PTH:-}" ]]; then
  if [[ -f "${TORCH_HOME}/hub/checkpoints/eva_vit_g.pth" ]]; then
    export EVA_VIT_G_PTH="${TORCH_HOME}/hub/checkpoints/eva_vit_g.pth"
  elif [[ -f "${HOME}/.cache/torch/hub/checkpoints/eva_vit_g.pth" ]]; then
    export EVA_VIT_G_PTH="${HOME}/.cache/torch/hub/checkpoints/eva_vit_g.pth"
  fi
fi

if [[ ! -d "${BERT_BASE_UNCASED_SNAPSHOT}" ]] || [[ ! -d "${FLAN_T5_XL_SNAPSHOT}" ]]; then
  echo "[FATAL] 未找到 bert-base-uncased 或 flan-t5-xl 本地 snapshot。" >&2
  echo "  BERT_BASE_UNCASED_SNAPSHOT=${BERT_BASE_UNCASED_SNAPSHOT}" >&2
  echo "  FLAN_T5_XL_SNAPSHOT=${FLAN_T5_XL_SNAPSHOT}" >&2
  exit 1
fi
if [[ ! -f "${BLIP2_PRETRAINED}" ]]; then
  echo "[FATAL] 未找到 BLIP2_PRETRAINED: ${BLIP2_PRETRAINED}" >&2
  exit 1
fi
if [[ -z "${EVA_VIT_G_PTH:-}" ]] || [[ ! -e "${EVA_VIT_G_PTH}" ]]; then
  echo "[FATAL] 未找到 EVA ViT-G 权重，请设置 EVA_VIT_G_PTH（本机常见: ${TORCH_HOME}/hub/checkpoints/eva_vit_g.pth）" >&2
  exit 1
fi

# 与剪枝/评测 run 绑定的时间戳（可 export 覆盖以复用已有 pth）
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SAMPLER_SEEDS="${SAMPLER_SEEDS:-${SEEDS:-0 30 42}}"
CALIBS="${CALIBS:-mmbench mmmu okvqa mathvista cc3m}"
CC3M_CALIB_SEED="${CC3M_CALIB_SEED:-42}"
SKIP_CC3M="${SKIP_CC3M:-0}"

# 从 CALIBS 拆出：四套大池走多 seed；CC3M 固定 128 条单独跑一轮
CALIBS_SEEDED=""
RUN_CC3M=0
for _c in $CALIBS; do
  if [[ "$_c" == "cc3m" ]]; then
    RUN_CC3M=1
  else
    CALIBS_SEEDED+=" $_c"
  fi
done
CALIBS_SEEDED="${CALIBS_SEEDED# }"

RUN_PRUNE="${RUN_PRUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

NUM_DATA="${NUM_DATA:-128}"
BS="${BS:-8}"
NUM_DATA_FIRST_STAGE="${NUM_DATA_FIRST_STAGE:-128}"
T5_SPEC="${T5_SPEC:-24-0.5-1.0-1.0}"
VIT_SPEC="${VIT_SPEC:-39-0.5-1.0-1.0}"
PRUNE_METHOD="${PRUNE_METHOD:-blipt5_tamp_pruner}"

MMBENCH_ROOT="${MMBENCH_ROOT:-$BASE/MMBench_eval}"
MMMU_ROOT="${MMMU_ROOT:-$BASE/MMMU_single_image}"
MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
MMMU_SPLIT="${MMMU_SPLIT:-test}"
MATHVISTA_EVAL_JSON="${MATHVISTA_EVAL_JSON:-$BASE/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json}"
MATHVISTA_IMAGES_DIR="${MATHVISTA_IMAGES_DIR:-$BASE/MathVista_eval_testmini_mc/images}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
# OKVQA 评测 DataLoader workers（默认 0，避免 spacy/thinc 在子进程 segfault）
OKVQA_EVAL_NUM_WORKERS="${OKVQA_EVAL_NUM_WORKERS:-0}"

SUMMARY_DIR="$REPO_ROOT/lavis/output/BLIP2"
mkdir -p "$REPO_ROOT/pruned_checkpoint"
mkdir -p "$SUMMARY_DIR"

MASTER_PORT_BASE="${MASTER_PORT:-29700}"
MASTER_PORT="$MASTER_PORT_BASE"

if (( NUM_DATA % BS != 0 )); then
  echo "[FATAL] NUM_DATA ($NUM_DATA) 必须能被 BS ($BS) 整除。" >&2
  exit 1
fi

# calib 短名 -> yaml 文件名、job/eval 标签名
calib_tag_name() {
  case "$1" in
    mmbench)    echo "MMBench" ;;
    mmmu)       echo "MMMU" ;;
    okvqa)      echo "OKVQAtrain" ;;
    mathvista)  echo "MathVista" ;;
    cc3m)       echo "CC3M" ;;
    *)
      echo "[FATAL] 未知 calibration: $1（mmbench/mmmu/okvqa/mathvista/cc3m）" >&2
      exit 1
      ;;
  esac
}

calib_cfg_yaml() {
  case "$1" in
    mmbench)    echo "cc_prefix_derivative_compute_okvqa_mmbench.yaml" ;;
    mmmu)       echo "cc_prefix_derivative_compute_okvqa_mmmu_overall.yaml" ;;
    okvqa)      echo "cc_prefix_derivative_compute_okvqa_okvqa_train_overall.yaml" ;;
    mathvista)  echo "cc_prefix_derivative_compute_okvqa_mathvista.yaml" ;;
    cc3m)       echo "cc_prefix_derivative_compute_cc3m_calib128.yaml" ;;
    *)
      echo "[FATAL] 未知 calibration: $1" >&2
      exit 1
      ;;
  esac
}

warn_if_missing_calib_json() {
  local j=""
  case "$1" in
    mmbench)   j="$BASE/MMBench_calibration/mmbench_calibration_train.json" ;;
    mmmu)      j="$BASE/MMMU_calibration/mmmu_calibration_train.json" ;;
    okvqa)     j="$BASE/datasets/okvqa/annotations/okvqa_train.json" ;;
    mathvista) j="$BASE/MathVista_calibration/mathvista_calibration_train.json" ;;
    cc3m)      j="$BASE/CC3M_calib_128/cc3m_calib_128.json" ;;
    *) return 0 ;;
  esac
  if [[ ! -f "$j" ]]; then
    echo "[WARN] 未找到 calibration 标注: $j" >&2
  fi
}

prune_one_calib() {
  local calib="$1"
  local seed="$2"
  local tag
  tag="$(calib_tag_name "$calib")"
  local cfg_name cfg job_id ckpt
  cfg_name="$(calib_cfg_yaml "$calib")"
  cfg="$REPO_ROOT/lavis/projects/blip2/eval/$cfg_name"
  if [[ ! -f "$cfg" ]]; then
    echo "[FATAL] 找不到 cfg: $cfg" >&2
    exit 1
  fi
  warn_if_missing_calib_json "$calib"

  job_id="tamp_calib${tag}_${RUN_STAMP}_calibseed${seed}"
  ckpt="$REPO_ROOT/pruned_checkpoint/${job_id}.pth"

  echo ""
  echo ">>> [PRUNE] calib=$calib tag=$tag seed=$seed job_id=$job_id"
  local P
  P=$MASTER_PORT
  MASTER_PORT=$((MASTER_PORT + 1))

  python -m torch.distributed.run --nproc_per_node=1 --master_port="$P" evaluate_blip.py \
    --cfg-path "$cfg" \
    --options "model.pretrained=${BLIP2_PRETRAINED}" \
    --pruning_method "$PRUNE_METHOD" \
    --save_pruned_model \
    --t5_prune_spec "$T5_SPEC" \
    --vit_prune_spec "$VIT_SPEC" \
    --num_data "$NUM_DATA" \
    --num_data_first_stage "$NUM_DATA_FIRST_STAGE" \
    --prunining_dataset_batch_size "$BS" \
    --job_id "$job_id"

  if [[ ! -f "$ckpt" ]]; then
    echo "[FATAL] 剪枝后未找到: $ckpt" >&2
    exit 1
  fi
  echo "[INFO] 已保存: $ckpt"
}

cc3m_job_id() {
  echo "tamp_calibCC3M_${RUN_STAMP}_calib128"
}

prune_cc3m_once() {
  local seed="${CC3M_CALIB_SEED}"
  export LAVIS_DISTRIBUTED_SAMPLER_SEED="$seed"
  local job_id ckpt
  job_id="$(cc3m_job_id)"
  ckpt="$REPO_ROOT/pruned_checkpoint/${job_id}.pth"
  warn_if_missing_calib_json cc3m

  echo ""
  echo ">>> [PRUNE] calib=cc3m（固定 128 条，只跑一次） job_id=$job_id sampler_seed=$seed"
  local P
  P=$MASTER_PORT
  MASTER_PORT=$((MASTER_PORT + 1))

  python -m torch.distributed.run --nproc_per_node=1 --master_port="$P" evaluate_blip.py \
    --cfg-path "$REPO_ROOT/lavis/projects/blip2/eval/cc_prefix_derivative_compute_cc3m_calib128.yaml" \
    --options "model.pretrained=${BLIP2_PRETRAINED}" \
    --pruning_method "$PRUNE_METHOD" \
    --save_pruned_model \
    --t5_prune_spec "$T5_SPEC" \
    --vit_prune_spec "$VIT_SPEC" \
    --num_data "$NUM_DATA" \
    --num_data_first_stage "$NUM_DATA_FIRST_STAGE" \
    --prunining_dataset_batch_size "$BS" \
    --job_id "$job_id"

  if [[ ! -f "$ckpt" ]]; then
    echo "[FATAL] 剪枝后未找到: $ckpt" >&2
    exit 1
  fi
  echo "[INFO] 已保存: $ckpt"
}

eval_four_bench() {
  local calib="$1"
  local seed="$2"
  local tag ckpt eval_tag okvqa_job
  tag="$(calib_tag_name "$calib")"
  local job_id="tamp_calib${tag}_${RUN_STAMP}_calibseed${seed}"
  ckpt="$REPO_ROOT/pruned_checkpoint/${job_id}.pth"
  eval_tag="eval_${tag}_${RUN_STAMP}_calibseed${seed}"
  okvqa_job="okvqa_eval_${eval_tag}_fullval"

  if [[ ! -f "$ckpt" ]]; then
    echo "[FATAL] 评测缺少权重: $ckpt" >&2
    exit 1
  fi

  echo ""
  echo "#####################################################################"
  echo "# EVAL  calib=$calib  calibseed=$seed"
  echo "# CKPT: $(readlink -f "$ckpt")"
  echo "#####################################################################"
  export LAVIS_EVAL_CALIB_TAG="$eval_tag"

  if [[ "${SKIP_MMBENCH:-0}" != "1" ]]; then
    echo ""
    echo ">>> [$eval_tag] MMBench"
    export LAVIS_METRICS_BENCHMARK="MMBench"
    python scripts/blip2/mmmu_eval_by_discipline.py \
      --mmmu_root "$MMBENCH_ROOT" \
      --split "$MMBENCH_SPLIT" \
      --ckpt "$ckpt" \
      --batch_size "$EVAL_BATCH_SIZE" \
      --device cuda \
      --overall_only
  fi

  if [[ "${SKIP_OKVQA:-0}" != "1" ]]; then
    echo ""
    echo ">>> [$eval_tag] OKVQA full val"
    local P
    P=$MASTER_PORT
    MASTER_PORT=$((MASTER_PORT + 1))
    python -m torch.distributed.run --nproc_per_node=1 --master_port="$P" evaluate_blip.py \
      --cfg-path lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml \
      --options "run.num_workers=${OKVQA_EVAL_NUM_WORKERS}" \
      --t5_pruned_checkpoint "$ckpt" \
      --vit_pruned_checkpoint "$ckpt" \
      --job_id "$okvqa_job"
  fi

  if [[ "${SKIP_MMMU:-0}" != "1" ]]; then
    echo ""
    echo ">>> [$eval_tag] MMMU"
    export LAVIS_METRICS_BENCHMARK="MMMU"
    python scripts/blip2/mmmu_eval_by_discipline.py \
      --mmmu_root "$MMMU_ROOT" \
      --split "$MMMU_SPLIT" \
      --ckpt "$ckpt" \
      --batch_size "$EVAL_BATCH_SIZE" \
      --device cuda \
      --overall_only
  fi

  if [[ "${SKIP_MATHVISTA:-0}" != "1" ]]; then
    echo ""
    echo ">>> [$eval_tag] MathVista MC"
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
  fi

}

run_one_sampler_seed() {
  local SEED="$1"
  export SEED
  export LAVIS_DISTRIBUTED_SAMPLER_SEED="$SEED"

  local metrics_jsonl summary_md summary_tsv
  metrics_jsonl="$SUMMARY_DIR/tamp_fivecalib_fourbench_${RUN_STAMP}_calibseed${SEED}.jsonl"
  summary_md="$SUMMARY_DIR/tamp_fivecalib_fourbench_${RUN_STAMP}_calibseed${SEED}.md"
  summary_tsv="$SUMMARY_DIR/tamp_fivecalib_fourbench_${RUN_STAMP}_calibseed${SEED}.tsv"
  export LAVIS_METRICS_JSONL="$metrics_jsonl"
  : > "$LAVIS_METRICS_JSONL"

  echo ""
  echo "########################################################################"
  echo "# RUN_STAMP=$RUN_STAMP  LAVIS_DISTRIBUTED_SAMPLER_SEED=$SEED"
  echo "# CALIBS_SEEDED=$CALIBS_SEEDED  PRUNE_METHOD=$PRUNE_METHOD (T5-only 默认)"
  echo "# metrics: $LAVIS_METRICS_JSONL"
  echo "########################################################################"

  local -a suite_specs=()

  for calib in $CALIBS_SEEDED; do
    if [[ "$RUN_PRUNE" == "1" ]]; then
      prune_one_calib "$calib" "$SEED"
    fi
  done

  for calib in $CALIBS_SEEDED; do
    if [[ "$RUN_EVAL" != "1" ]]; then
      continue
    fi
    local tag eval_tag okvqa_job
    tag="$(calib_tag_name "$calib")"
    eval_tag="eval_${tag}_${RUN_STAMP}_calibseed${SEED}"
    okvqa_job="okvqa_eval_${eval_tag}_fullval"
    # calib_tag 须与 LAVIS_EVAL_CALIB_TAG 一致，collect 才能从 jsonl 读到 MMBench/MMMU/MathVista
    suite_specs+=("${eval_tag}:${okvqa_job}")
    eval_four_bench "$calib" "$SEED"
  done

  if [[ "$RUN_EVAL" == "1" ]] && [[ ${#suite_specs[@]} -gt 0 ]]; then
    echo ""
    echo "========== 汇总 calibseed=$SEED =========="
    python "$SCRIPT_DIR/collect_lavisbackup_eval_summary.py" \
      --repo-root "$REPO_ROOT" \
      --metrics-jsonl "$LAVIS_METRICS_JSONL" \
      --out-md "$summary_md" \
      --out-tsv "$summary_tsv" \
      --suites "${suite_specs[@]}"
    echo "已写入: $summary_md"
    echo "TSV:    $summary_tsv"
  fi

  echo ""
  echo "========== 完成 calibseed=$SEED =========="
}

run_cc3m_once() {
  if [[ "$RUN_CC3M" != "1" ]] || [[ "$SKIP_CC3M" == "1" ]]; then
    return 0
  fi

  local job_id ckpt eval_tag okvqa_job
  job_id="$(cc3m_job_id)"
  ckpt="$REPO_ROOT/pruned_checkpoint/${job_id}.pth"
  eval_tag="eval_CC3M_${RUN_STAMP}_calib128"
  okvqa_job="okvqa_eval_${eval_tag}_fullval"

  local metrics_jsonl summary_md summary_tsv
  metrics_jsonl="$SUMMARY_DIR/tamp_fivecalib_fourbench_${RUN_STAMP}_cc3m_calib128.jsonl"
  summary_md="$SUMMARY_DIR/tamp_fivecalib_fourbench_${RUN_STAMP}_cc3m_calib128.md"
  summary_tsv="$SUMMARY_DIR/tamp_fivecalib_fourbench_${RUN_STAMP}_cc3m_calib128.tsv"

  echo ""
  echo "########################################################################"
  echo "# CC3M：固定 128 条标定 — 剪枝 1 次 + 四基准评测 1 次"
  echo "# job_id=$job_id"
  echo "########################################################################"

  if [[ "$RUN_PRUNE" == "1" ]]; then
    prune_cc3m_once
  fi

  if [[ "$RUN_EVAL" == "1" ]]; then
    if [[ ! -f "$ckpt" ]]; then
      echo "[FATAL] CC3M 评测缺少权重: $ckpt" >&2
      exit 1
    fi
    export LAVIS_METRICS_JSONL="$metrics_jsonl"
    : > "$LAVIS_METRICS_JSONL"
    export LAVIS_EVAL_CALIB_TAG="$eval_tag"

    echo ""
    echo "#####################################################################"
    echo "# EVAL  calib=cc3m（calib128 固定集）"
    echo "# CKPT: $(readlink -f "$ckpt")"
    echo "#####################################################################"

    if [[ "${SKIP_MMBENCH:-0}" != "1" ]]; then
      echo ""
      echo ">>> [$eval_tag] MMBench"
      export LAVIS_METRICS_BENCHMARK="MMBench"
      python scripts/blip2/mmmu_eval_by_discipline.py \
        --mmmu_root "$MMBENCH_ROOT" \
        --split "$MMBENCH_SPLIT" \
        --ckpt "$ckpt" \
        --batch_size "$EVAL_BATCH_SIZE" \
        --device cuda \
        --overall_only
    fi

    if [[ "${SKIP_OKVQA:-0}" != "1" ]]; then
      echo ""
      echo ">>> [$eval_tag] OKVQA full val"
      local P
      P=$MASTER_PORT
      MASTER_PORT=$((MASTER_PORT + 1))
      python -m torch.distributed.run --nproc_per_node=1 --master_port="$P" evaluate_blip.py \
        --cfg-path lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml \
        --options "run.num_workers=${OKVQA_EVAL_NUM_WORKERS}" \
        --t5_pruned_checkpoint "$ckpt" \
        --vit_pruned_checkpoint "$ckpt" \
        --job_id "$okvqa_job"
    fi

    if [[ "${SKIP_MMMU:-0}" != "1" ]]; then
      echo ""
      echo ">>> [$eval_tag] MMMU"
      export LAVIS_METRICS_BENCHMARK="MMMU"
      python scripts/blip2/mmmu_eval_by_discipline.py \
        --mmmu_root "$MMMU_ROOT" \
        --split "$MMMU_SPLIT" \
        --ckpt "$ckpt" \
        --batch_size "$EVAL_BATCH_SIZE" \
        --device cuda \
        --overall_only
    fi

    if [[ "${SKIP_MATHVISTA:-0}" != "1" ]]; then
      echo ""
      echo ">>> [$eval_tag] MathVista MC"
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
    fi

    echo ""
    echo "========== 汇总 CC3M calib128 =========="
    python "$SCRIPT_DIR/collect_lavisbackup_eval_summary.py" \
      --repo-root "$REPO_ROOT" \
      --metrics-jsonl "$LAVIS_METRICS_JSONL" \
      --out-md "$summary_md" \
      --out-tsv "$summary_tsv" \
      --suites "${eval_tag}:${okvqa_job}"
    echo "已写入: $summary_md"
    echo "TSV:    $summary_tsv"
    echo "指标行: $LAVIS_METRICS_JSONL"
  fi

  echo ""
  echo "========== 完成 CC3M calib128 =========="
}

echo "========== LAVIS_backup TAMP 五套 calib 剪枝 + 四基准评测 =========="
echo "[INFO] REPO_ROOT=$REPO_ROOT"
echo "[INFO] BLIP2_PRETRAINED=$BLIP2_PRETRAINED"
echo "[INFO] EVA_VIT_G_PTH=$EVA_VIT_G_PTH"
echo "[INFO] BERT_BASE_UNCASED_SNAPSHOT=$BERT_BASE_UNCASED_SNAPSHOT"
echo "[INFO] FLAN_T5_XL_SNAPSHOT=$FLAN_T5_XL_SNAPSHOT"
echo "[INFO] HF_HOME=$HF_HOME"
echo "[INFO] RUN_STAMP=$RUN_STAMP"
echo "[INFO] SAMPLER_SEEDS=$SAMPLER_SEEDS"
echo "[INFO] CALIBS=$CALIBS"
echo "[INFO] CALIBS_SEEDED=$CALIBS_SEEDED RUN_CC3M=$RUN_CC3M SKIP_CC3M=$SKIP_CC3M"
echo "[INFO] CC3M 权重: pruned_checkpoint/tamp_calibCC3M_${RUN_STAMP}_calib128.pth"
echo "[INFO] NUM_DATA=$NUM_DATA BS=$BS PRUNE_METHOD=$PRUNE_METHOD"
echo "[INFO] RUN_PRUNE=$RUN_PRUNE RUN_EVAL=$RUN_EVAL"
echo "================================================================"

if [[ -n "$CALIBS_SEEDED" ]]; then
  for _seed in $SAMPLER_SEEDS; do
    run_one_sampler_seed "$_seed"
  done
fi

run_cc3m_once

echo ""
echo "[DONE] 全部完成。RUN_STAMP=$RUN_STAMP seeds=$SAMPLER_SEEDS"
echo "[DONE] 多 seed 权重: pruned_checkpoint/tamp_calib*_${RUN_STAMP}_calibseed*.pth"
if [[ "$RUN_CC3M" == "1" ]] && [[ "$SKIP_CC3M" != "1" ]]; then
  echo "[DONE] CC3M 权重: $REPO_ROOT/pruned_checkpoint/tamp_calibCC3M_${RUN_STAMP}_calib128.pth"
fi
