#!/usr/bin/env bash
# =============================================================================
# Joint ViT+T5 SparseGPT：分别以 CC3M(128) / MMBench / MMMU / OKVQA(train) / MathVista
# 为 calibration，各剪一份权重（keep ratio 0.5，即约 50% 稀疏预算，与 24-0.5 / 39-0.5 规格一致），
# 再对每个 pth 跑四基准：MMBench / MMMU / OKVQA / MathVista。
#
# 对应 yaml（均在 lavis/projects/blip2/eval/，数据路径在 yaml 内，换机器请改）:
#   cc3m:      cc_prefix_derivative_compute_cc3m_calib128.yaml
#   mmbench:   cc_prefix_derivative_compute_okvqa_mmbench.yaml
#   mmmu:      cc_prefix_derivative_compute_okvqa_mmmu_overall.yaml
#   okvqa:     cc_prefix_derivative_compute_okvqa_okvqa_train_overall.yaml
#   mathvista: cc_prefix_derivative_compute_okvqa_mathvista.yaml
#
# 用法（在 ECoFLaP/LAVIS 根目录）:
#   bash scripts/blip2/run_joint_sparsegpt_fivecalib128_prune_then_fourbench_each.sh
#
# 只跑部分 calibration（空格分隔，名字须为 cc3m/mmbench/mmmu/okvqa/mathvista）:
#   CALIBS="mmbench okvqa" bash scripts/blip2/run_joint_sparsegpt_fivecalib128_prune_then_fourbench_each.sh
#
# 只剪枝不评测 / 只评测已有 pth:
#   RUN_EVAL=0 bash .../run_joint_sparsegpt_fivecalib128_prune_then_fourbench_each.sh
#   RUN_PRUNE=0 bash .../run_joint_sparsegpt_fivecalib128_prune_then_fourbench_each.sh
#
# 环境变量（可选）:
#   JOB_PREFIX=joint_sparsegpt_calib128   → pruned_checkpoint/${JOB_PREFIX}_<calib>.pth
#   NUM_DATA=128 BS=8  T5_SPEC  VIT_SPEC
#   SCORE_METHOD=MEZO-GradOnly_sum  SPARSITY_GRANULARITY=block  MAX_SPARSITY_PER_LAYER=0.6
#   NUM_DATA_FIRST_STAGE=32  BLIP2_PRETRAINED  HF_HOME
#   MASTER_PORT_PRUNE_BASE=29810  MASTER_PORT_EVAL_BASE=29900
#   LAVIS_METRICS_JSONL  追加四基准指标
#   EVAL_JOB_OKVQA_OVERRIDE=固定名  若设，每轮 OKVQA 评测均用该 job_id
#
# 注意:
#   - num_data %% batch_size == 0（Runner 断言）。
#   - blipt5_sparsegpt_pruner 在 joint + granularity 下要求 vit/t5 keep 一致（本脚本默认一致）。
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "$REPO_ROOT" || exit 1

mkdir -p "$REPO_ROOT/pruned_checkpoint"
mkdir -p "$REPO_ROOT/training_statistics"
export LAVIS_METRICS_JSONL="${LAVIS_METRICS_JSONL:-$REPO_ROOT/training_statistics/joint_sparsegpt_fivecalib128_fourbench_metrics.jsonl}"
echo "[INFO] LAVIS_METRICS_JSONL=$LAVIS_METRICS_JSONL"

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

JOB_PREFIX="${JOB_PREFIX:-joint_sparsegpt_calib128}"
NUM_DATA="${NUM_DATA:-128}"
BS="${BS:-8}"
NUM_DATA_FIRST_STAGE="${NUM_DATA_FIRST_STAGE:-32}"
# 约 50% 稀疏（keep 0.5）：T5 24 层 / ViT 39 block，与仓库常用写法一致
T5_SPEC="${T5_SPEC:-24-0.5-1.0-1.0}"
VIT_SPEC="${VIT_SPEC:-39-0.5-1.0-1.0}"
SCORE_METHOD="${SCORE_METHOD:-MEZO-GradOnly_sum}"
SPARSITY_GRANULARITY="${SPARSITY_GRANULARITY:-block}"
MAX_SPARSITY_PER_LAYER="${MAX_SPARSITY_PER_LAYER:-0.6}"
RUN_PRUNE="${RUN_PRUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
MASTER_PORT_PRUNE_BASE="${MASTER_PORT_PRUNE_BASE:-29810}"
MASTER_PORT_EVAL_BASE="${MASTER_PORT_EVAL_BASE:-29900}"

CALIBS="${CALIBS:-cc3m mmbench mmmu okvqa mathvista}"

if (( NUM_DATA % BS != 0 )); then
  echo "[FATAL] NUM_DATA ($NUM_DATA) 必须能被 BS ($BS) 整除。" >&2
  exit 1
fi

warn_if_missing_calib_json() {
  local j=""
  case "$1" in
    cc3m) j="/root/autodl-tmp/CC3M_calib_128/cc3m_calib_128.json" ;;
    mmbench) j="/root/autodl-tmp/MMBench_calibration/mmbench_calibration_train.json" ;;
    mmmu) j="/root/autodl-tmp/MMMU_calibration/mmmu_calibration_train.json" ;;
    okvqa) j="/root/autodl-tmp/datasets/okvqa/annotations/okvqa_train.json" ;;
    mathvista) j="/root/autodl-tmp/MathVista_calibration/mathvista_calibration_train.json" ;;
    *) return 0 ;;
  esac
  if [[ ! -f "$j" ]]; then
    echo "[WARN] 未找到默认 calibration 标注（若已改 yaml 路径可忽略）: $j" >&2
  fi
}

cfg_yaml_for() {
  case "$1" in
    cc3m) echo "cc_prefix_derivative_compute_cc3m_calib128.yaml" ;;
    mmbench) echo "cc_prefix_derivative_compute_okvqa_mmbench.yaml" ;;
    mmmu) echo "cc_prefix_derivative_compute_okvqa_mmmu_overall.yaml" ;;
    okvqa) echo "cc_prefix_derivative_compute_okvqa_okvqa_train_overall.yaml" ;;
    mathvista) echo "cc_prefix_derivative_compute_okvqa_mathvista.yaml" ;;
    *)
      echo "[FATAL] 未知 calibration 名: $1（须为 cc3m/mmbench/mmmu/okvqa/mathvista）" >&2
      exit 1
      ;;
  esac
}

idx=0
for calib in $CALIBS; do
  CFG_NAME="$(cfg_yaml_for "$calib")"
  CFG="$REPO_ROOT/lavis/projects/blip2/eval/$CFG_NAME"
  if [[ ! -f "$CFG" ]]; then
    echo "[FATAL] 找不到 cfg: $CFG" >&2
    exit 1
  fi
  warn_if_missing_calib_json "$calib"

  JOB_ID="${JOB_PREFIX}_${calib}"
  JOINT_CKPT="$REPO_ROOT/pruned_checkpoint/${JOB_ID}.pth"
  PORT_PRUNE=$((MASTER_PORT_PRUNE_BASE + idx))
  export MASTER_PORT=$((MASTER_PORT_EVAL_BASE + idx * 10))

  echo ""
  echo "#####################################################################"
  echo "#  SparseGPT joint  calibration=$calib  JOB_ID=$JOB_ID"
  echo "#  cfg=$CFG"
  echo "#####################################################################"

  if [[ "$RUN_PRUNE" == "1" ]]; then
    python -m torch.distributed.run --nproc_per_node=1 --master_port "$PORT_PRUNE" evaluate_blip.py \
      --cfg-path "$CFG" \
      --options model.pretrained="${BLIP2_PRETRAINED}" \
      --pruning_method blipt5_sparsegpt_pruner \
      --save_pruned_model \
      --prunining_dataset_batch_size "$BS" \
      --num_data "$NUM_DATA" \
      --num_data_first_stage "$NUM_DATA_FIRST_STAGE" \
      --t5_prune_spec "$T5_SPEC" \
      --vit_prune_spec "$VIT_SPEC" \
      --score_method "$SCORE_METHOD" \
      --sparsity_ratio_granularity "$SPARSITY_GRANULARITY" \
      --max_sparsity_per_layer "$MAX_SPARSITY_PER_LAYER" \
      --job_id "$JOB_ID" \
      "$@"
  fi

  if [[ "$RUN_EVAL" == "1" ]]; then
    if [[ ! -f "$JOINT_CKPT" ]]; then
      echo "[FATAL] 找不到剪枝权重: $JOINT_CKPT" >&2
      exit 1
    fi
    export JOINT_SINGLE_CKPT="$JOINT_CKPT"
    if [[ -n "${EVAL_JOB_OKVQA_OVERRIDE:-}" ]]; then
      export EVAL_JOB_OKVQA="$EVAL_JOB_OKVQA_OVERRIDE"
    else
      export EVAL_JOB_OKVQA="okvqa_eval_${JOB_ID}"
    fi
    bash "$SCRIPT_DIR/run_ecoflap_split_merge_eval_fourbench.sh"
  fi

  idx=$((idx + 1))
done

echo ""
echo "[INFO] 全部流程结束（CALIBS=$CALIBS）。权重前缀 JOB_PREFIX=$JOB_PREFIX"
