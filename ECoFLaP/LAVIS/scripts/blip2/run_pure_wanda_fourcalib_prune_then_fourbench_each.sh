#!/usr/bin/env bash
# =============================================================================
# 分别以 MMBench / MMMU / OKVQA(train) / MathVista 为 calibration，做「纯 Wanda」剪枝
# （默认只剪 T5；PRUNE_VIT=1 时额外剪 ViT）。不传 --sparsity_ratio_granularity / 不传 ECoFLaP 的 MEZO+block。
# 每个 calibration 各产出一份 pruned_checkpoint/${JOB_ID}.pth，并对该权重依次跑四基准：MMBench / MMMU / OKVQA / MathVista。
#
# 对应 cfg（数据路径在 yaml 内，可按机器修改）:
#   MMBench:    cc_prefix_derivative_compute_okvqa_mmbench.yaml
#   MMMU:       cc_prefix_derivative_compute_okvqa_mmmu_overall.yaml  （需先生成 MMMU_calibration）
#   OKVQA:      cc_prefix_derivative_compute_okvqa_okvqa_train_overall.yaml
#   MathVista:  cc_prefix_derivative_compute_okvqa_mathvista.yaml
#
# 用法（在 ECoFLaP/LAVIS 根目录）:
#   bash scripts/blip2/run_pure_wanda_fourcalib_prune_then_fourbench_each.sh
#
# 只跑部分 calibration（空格分隔）:
#   CALIBS="mmbench mathvista" bash scripts/blip2/run_pure_wanda_fourcalib_prune_then_fourbench_each.sh
#
# 只剪枝不评测:
#   RUN_EVAL=0 bash scripts/blip2/run_pure_wanda_fourcalib_prune_then_fourbench_each.sh
#
# 已有 pth 只评测（需 JOB_ID 与当时一致，或改 JOINT_SINGLE_CKPT）:
#   RUN_PRUNE=0 JOB_ID=pure_wanda_calib_mmbench bash ...  # 单条需手写循环或单独 export JOINT_SINGLE_CKPT
#
# 环境变量（可选）:
#   JOB_PREFIX=pure_wanda_calib   产出 job_id: ${JOB_PREFIX}_<mmbench|mmmu|okvqa|mathvista>
#   NUM_DATA=128  BS=8  T5_SPEC  VIT_SPEC  PRUNE_VIT=0（默认）/1 启用 ViT 剪枝  BLIP2_PRETRAINED
#   MASTER_PORT_PRUNE_BASE=29510  MASTER_PORT_EVAL_BASE=29600  （每轮递增，避免端口冲突）
#   EVAL_JOB_OKVQA_OVERRIDE=固定名  若设，四轮 OKVQA 评测均用该 --job_id（默认每轮 okvqa_eval_${JOB_PREFIX}_<calib>）
#
# 注意:
#   - Runner 要求 num_data % batch_size == 0（见 runner_base.get_dataloader_for_importance_computation）。
#   - 各 yaml 内数据路径默认为 /root/autodl-tmp/...，换机器请改 yaml。
#   - MMMU 需先按 cc_prefix_derivative_compute_okvqa_mmmu_overall.yaml 顶注释生成 calibration JSON。
#
# 跑完后汇总四基准数字:
#   python scripts/blip2/collect_pure_wanda_fourbench_summary.py
#
# 若当时未写 LAVIS_METRICS_JSONL，已有四份 pth 时只补 MMBench/MMMU/MathVista（跳过 OKVQA）:
#   bash scripts/blip2/run_pure_wanda_fourcalib_backfill_three_bench.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "$REPO_ROOT" || exit 1

mkdir -p "$REPO_ROOT/pruned_checkpoint"
mkdir -p "$REPO_ROOT/training_statistics"
# MMBench/MMMU/MathVista 结构化指标追加到此文件（供 collect_pure_wanda_fourbench_summary.py 汇总）
export LAVIS_METRICS_JSONL="${LAVIS_METRICS_JSONL:-$REPO_ROOT/training_statistics/pure_wanda_fourbench_metrics.jsonl}"
echo "[INFO] LAVIS_METRICS_JSONL=$LAVIS_METRICS_JSONL"

# --- HuggingFace：离线 + 本地 snapshot ---
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

JOB_PREFIX="${JOB_PREFIX:-pure_wanda_calib}"
NUM_DATA="${NUM_DATA:-128}"
BS="${BS:-8}"
NUM_DATA_FIRST_STAGE="${NUM_DATA_FIRST_STAGE:-32}"
T5_SPEC="${T5_SPEC:-24-0.5-1.0-1.0}"
VIT_SPEC="${VIT_SPEC:-39-0.5-1.0-1.0}"
PRUNE_VIT="${PRUNE_VIT:-0}"
RUN_PRUNE="${RUN_PRUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
MASTER_PORT_PRUNE_BASE="${MASTER_PORT_PRUNE_BASE:-29510}"
MASTER_PORT_EVAL_BASE="${MASTER_PORT_EVAL_BASE:-29600}"

CALIBS="${CALIBS:-mmbench mmmu okvqa mathvista}"

if (( NUM_DATA % BS != 0 )); then
  echo "[FATAL] NUM_DATA ($NUM_DATA) 必须能被 BS ($BS) 整除（LAVIS Runner 断言）。" >&2
  exit 1
fi

# 可选：启动前检查 yaml 内默认标注是否存在（路径与 yaml 一致；若你改过 yaml 可忽略 WARN）
warn_if_missing_calib_json() {
  local j=""
  case "$1" in
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
    mmbench) echo "cc_prefix_derivative_compute_okvqa_mmbench.yaml" ;;
    mmmu) echo "cc_prefix_derivative_compute_okvqa_mmmu_overall.yaml" ;;
    okvqa) echo "cc_prefix_derivative_compute_okvqa_okvqa_train_overall.yaml" ;;
    mathvista) echo "cc_prefix_derivative_compute_okvqa_mathvista.yaml" ;;
    *)
      echo "[FATAL] 未知 calibration 名: $1（需为 mmbench/mmmu/okvqa/mathvista）" >&2
      exit 1
      ;;
  esac
}

PRUNE_EXTRA=()
if [[ "$PRUNE_VIT" == "1" ]]; then
  PRUNE_EXTRA+=(--prune_vit)
fi

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
  echo "#  calibration=$calib  JOB_ID=$JOB_ID"
  echo "#  cfg=$CFG"
  echo "#####################################################################"

  if [[ "$RUN_PRUNE" == "1" ]]; then
    python -m torch.distributed.run --nproc_per_node=1 --master_port "$PORT_PRUNE" evaluate_blip.py \
      --cfg-path "$CFG" \
      --options model.pretrained="${BLIP2_PRETRAINED}" \
      --pruning_method blipt5_wanda_pruner \
      --save_pruned_model \
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
      echo "[FATAL] 找不到剪枝权重: $JOINT_CKPT" >&2
      exit 1
    fi
    export JOINT_SINGLE_CKPT="$JOINT_CKPT"
    # 每轮必须重设，不能用 EVAL_JOB_OKVQA:-默认值（首轮 export 后后续轮会错误沿用第一次）。
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
echo "[INFO] 全部 calibration 流程结束（CALIBS=$CALIBS）。"
