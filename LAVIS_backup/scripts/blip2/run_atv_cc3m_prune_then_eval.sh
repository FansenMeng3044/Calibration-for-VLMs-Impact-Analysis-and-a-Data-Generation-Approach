#!/usr/bin/env bash
# =============================================================================
# ATV-Pruning（CVPR 2026）在 BLIP2-T5 上复现：
#   CC3M 多模态标定 → blipt5_atv_pruner（只剪 T5，均匀稀疏，token_selection=atv）→ 保存 pth
#   → 四基准评测（MMBench / OKVQA / MMMU / MathVista）
#
# 关键约束：
#   - ATV 必须多模态标定（需要 temp_label/视觉 query token）；不能用 t5_c4_text。
#   - 只剪 T5（不传 --prune_vit）；ViT/Q-Former/t5_proj 保持稠密，忠实 ATV「只剪 LLM」。
#   - 均匀稀疏（别名 blipt5_atv_pruner 自动置 sparsity_ratio_granularity=None）。
#   - --atv_alpha 控制 k=round(min(1,alpha·avg_cosdist)·#text_tokens)，clamp≤#query(32)。
#   - 剪枝日志会打印每层 [ATV] cos_dist_avg / k / 退化占比（k==num_img）—— 用于诊断。
#
# 用法：
#   cd /data/data2/mfs/2/LAVIS_backup
#   bash scripts/blip2/run_atv_cc3m_prune_then_eval.sh
# 仅评测（已剪好）：RUN_PRUNE=0 JOB_ID=<你的job_id> bash scripts/.../run_atv_cc3m_prune_then_eval.sh
#
# 环境变量：BASE, RUN_PRUNE/RUN_EVAL(默认1), RUN_ENV_CAPTURE(默认1), JOB_STAMP, JOB_ID, ATV_ALPHA(1.0),
#   CC3M_CFG, CC3M_JSON, CC3M_IMAGES_DIR, T5_SPEC(24-0.5-1.0-1.0), NUM_DATA(128), BS(8), SEED(42)
#   MMBENCH_ROOT/MMMU_ROOT/MATHVISTA_EVAL_JSON/MATHVISTA_IMAGES_DIR/EVAL_BATCH_SIZE, MASTER_PORT_*
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
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
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
[[ -d "${BERT_BASE_UNCASED_SNAPSHOT}" ]] || { echo "[FATAL] 未找到 bert-base-uncased snapshot" >&2; exit 1; }
[[ -d "${FLAN_T5_XL_SNAPSHOT}" ]]      || { echo "[FATAL] 未找到 flan-t5-xl snapshot" >&2; exit 1; }

BLIP2_PRETRAINED="${BLIP2_PRETRAINED:-$BASE/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth}"
[[ -f "${BLIP2_PRETRAINED}" ]] || { echo "[FATAL] 未找到 BLIP2_PRETRAINED: ${BLIP2_PRETRAINED}" >&2; exit 1; }

CC3M_CFG="${CC3M_CFG:-$REPO_ROOT/lavis/projects/blip2/eval/cc_prefix_derivative_compute_cc3m_calib128.yaml}"
[[ -f "$CC3M_CFG" ]] || { echo "[FATAL] 找不到 CC3M cfg: $CC3M_CFG（用 CC3M_CFG=... 指定）" >&2; exit 1; }

JOB_STAMP="${JOB_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-42}"
export LAVIS_DISTRIBUTED_SAMPLER_SEED="${LAVIS_DISTRIBUTED_SAMPLER_SEED:-$SEED}"
JOB_ID="${JOB_ID:-atv_cc3m_t5only_${JOB_STAMP}_seed${SEED}}"

ATV_ALPHA="${ATV_ALPHA:-1.0}"
T5_SPEC="${T5_SPEC:-24-0.5-1.0-1.0}"
NUM_DATA="${NUM_DATA:-128}"
BS="${BS:-8}"

RUN_PRUNE="${RUN_PRUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_ENV_CAPTURE="${RUN_ENV_CAPTURE:-1}"
REPORT_DIR="${REPORT_DIR:-$BASE/atv_validation_report_${JOB_STAMP}_seed${SEED}}"
ATV_ROOT="${ATV_ROOT:-$BASE/ATV-Pruning}"
VALIDATE_REPORT="${VALIDATE_REPORT:-1}"
PYTHON="${PYTHON:-python}"
PRUNE_PROVENANCE_CSV="${PRUNE_PROVENANCE_CSV:-$REPORT_DIR/prune_provenance.csv}"
PRUNE_PROVENANCE_ROLE="${PRUNE_PROVENANCE_ROLE:-main}"
MATERIALIZE_CC3M_CFG="${MATERIALIZE_CC3M_CFG:-1}"

MMBENCH_ROOT="${MMBENCH_ROOT:-$BASE/MMBench_eval}"
MMMU_ROOT="${MMMU_ROOT:-$BASE/MMMU_single_image}"
MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
MMMU_SPLIT="${MMMU_SPLIT:-test}"
MATHVISTA_EVAL_JSON="${MATHVISTA_EVAL_JSON:-$BASE/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json}"
MATHVISTA_IMAGES_DIR="${MATHVISTA_IMAGES_DIR:-$BASE/MathVista_eval_testmini_mc/images}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
MASTER_PORT_PRUNE="${MASTER_PORT_PRUNE:-29720}"
MASTER_PORT_OKVQA="${MASTER_PORT_OKVQA:-29730}"

CKPT="$REPO_ROOT/pruned_checkpoint/${JOB_ID}.pth"
PRUNE_LOG="$REPORT_DIR/${JOB_ID}_prune.log"
SPARSITY_CSV="$REPORT_DIR/${JOB_ID}_sparsity.csv"
export LAVIS_METRICS_JSONL="${LAVIS_METRICS_JSONL:-$REPORT_DIR/${JOB_ID}_fourbench_metrics.jsonl}"
mkdir -p "$REPORT_DIR"

csv_cell() {
  local value="${1:-}"
  value="${value//\"/\"\"}"
  printf '"%s"' "$value"
}

append_prune_provenance() {
  if [[ ! -s "$PRUNE_PROVENANCE_CSV" ]]; then
    printf 'seed,method,role,alpha,job_id,ckpt,calib_cfg,calib_json,images_dir,num_data,t5_spec,t5_sparsity_target,vit_sparsity_target,pruning_scope,run_prune,prune_log,sparsity_csv\n' > "$PRUNE_PROVENANCE_CSV"
  fi
  {
    csv_cell "$SEED"; printf ','
    csv_cell "atv"; printf ','
    csv_cell "$PRUNE_PROVENANCE_ROLE"; printf ','
    csv_cell "$ATV_ALPHA"; printf ','
    csv_cell "$JOB_ID"; printf ','
    csv_cell "$CKPT"; printf ','
    csv_cell "$CC3M_CFG_RUNTIME"; printf ','
    csv_cell "${CC3M_JSON:-}"; printf ','
    csv_cell "${CC3M_IMAGES_DIR:-}"; printf ','
    csv_cell "$NUM_DATA"; printf ','
    csv_cell "$T5_SPEC"; printf ','
    csv_cell "0.5"; printf ','
    csv_cell "0.0"; printf ','
    csv_cell "t5_only"; printf ','
    csv_cell "$RUN_PRUNE"; printf ','
    csv_cell "$PRUNE_LOG"; printf ','
    csv_cell "$SPARSITY_CSV"; printf '\n'
  } >> "$PRUNE_PROVENANCE_CSV"
}

CC3M_CFG_RUNTIME="$CC3M_CFG"
CC3M_JSON="${CC3M_JSON:-$BASE/CC3M_calib_128/cc3m_calib_128.json}"
CC3M_IMAGES_DIR="${CC3M_IMAGES_DIR:-$BASE/CC3M_calib_128/images}"
if [[ "$MATERIALIZE_CC3M_CFG" == "1" ]]; then
  CC3M_CFG_RUNTIME="$REPORT_DIR/cc3m_calib_runtime.yaml"
  python scripts/blip2/materialize_cc3m_calib_cfg.py \
    --src_cfg "$CC3M_CFG" \
    --out_cfg "$CC3M_CFG_RUNTIME" \
    --cc3m_json "$CC3M_JSON" \
    --cc3m_images_dir "$CC3M_IMAGES_DIR" \
    --pretrained "$BLIP2_PRETRAINED"
fi

rm -f "$REPORT_DIR/runtime_environment.json" "$REPORT_DIR/runtime_environment.md"
if [[ "$RUN_ENV_CAPTURE" == "1" ]]; then
  echo "========== Runtime environment provenance =========="
  "$PYTHON" scripts/blip2/capture_atv_runtime_env.py \
    --lavis_root "$REPO_ROOT" \
    --original_atv_root "$ATV_ROOT" \
    --report_dir "$REPORT_DIR" \
    --base "$BASE" \
    --stamp "$JOB_STAMP" \
    --seeds "$SEED" \
    --models "atv" \
    --run_prune "$RUN_PRUNE" \
    --run_eval "$RUN_EVAL"
else
  echo "[INFO] RUN_ENV_CAPTURE=0, skipping runtime environment provenance capture."
fi

four_bench_eval() {
  local ckpt="$1" eval_tag="$2" okvqa_job="$3" master_port="$4"
  [[ -f "$ckpt" ]] || { echo "[FATAL] 找不到权重: $ckpt" >&2; exit 1; }
  export LAVIS_EVAL_CALIB_TAG="$eval_tag"
  echo ""; echo "========== 四基准评测 | tag=$eval_tag =========="

  export LAVIS_METRICS_BENCHMARK="MMBench"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMBENCH_ROOT" --split "$MMBENCH_SPLIT" --ckpt "$ckpt" \
    --batch_size "$EVAL_BATCH_SIZE" --device cuda --overall_only

  python -m torch.distributed.run --nproc_per_node=1 --master_port="$master_port" evaluate_blip.py \
    --cfg-path lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml \
    --t5_pruned_checkpoint "$ckpt" --vit_pruned_checkpoint "$ckpt" \
    --job_id "$okvqa_job"

  export LAVIS_METRICS_BENCHMARK="MMMU"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMMU_ROOT" --split "$MMMU_SPLIT" --ckpt "$ckpt" \
    --batch_size "$EVAL_BATCH_SIZE" --device cuda --overall_only

  if [[ -f "$MATHVISTA_EVAL_JSON" ]]; then
    export LAVIS_METRICS_BENCHMARK="MathVista_MC"
    python scripts/blip2/mathvista_mc_eval.py \
      --eval_json "$MATHVISTA_EVAL_JSON" --images_dir "$MATHVISTA_IMAGES_DIR" \
      --ckpt "$ckpt" --batch_size "$EVAL_BATCH_SIZE" --device cuda
  else
    echo "[WARN] 跳过 MathVista：缺少 $MATHVISTA_EVAL_JSON"
  fi
  echo "[INFO] 四基准评测结束: $eval_tag"
}

echo "========== ATV-Pruning CC3M | 只剪 T5 | 均匀稀疏 | alpha=$ATV_ALPHA | STAMP=$JOB_STAMP =========="
echo "[INFO] REPO_ROOT=$REPO_ROOT  CC3M_CFG=$CC3M_CFG_RUNTIME"
echo "[INFO] JOB_ID=$JOB_ID → $CKPT"
echo "[INFO] RUN_PRUNE=$RUN_PRUNE RUN_EVAL=$RUN_EVAL  T5_SPEC=$T5_SPEC"
echo "[INFO] REPORT_DIR=$REPORT_DIR"

if [[ "$RUN_PRUNE" == "1" ]]; then
  echo ""; echo ">>> 剪枝: CC3M 多模态标定 + ATV + 只剪 T5"
  LAVIS_ATV_DIAGNOSTIC_DIR="${LAVIS_ATV_DIAGNOSTIC_DIR:-$REPORT_DIR}" \
  python -m torch.distributed.run --nproc_per_node=1 --master_port="$MASTER_PORT_PRUNE" evaluate_blip.py \
    --cfg-path "$CC3M_CFG_RUNTIME" \
    --options "model.pretrained=${BLIP2_PRETRAINED}" \
    --pruning_method blipt5_atv_pruner \
    --atv_alpha "$ATV_ALPHA" \
    --no_prune_vit \
    --save_pruned_model \
    --prunining_dataset_batch_size "$BS" \
    --num_data "$NUM_DATA" \
    --num_data_first_stage "$NUM_DATA" \
    --t5_prune_spec "$T5_SPEC" \
    --job_id "$JOB_ID" 2>&1 | tee "$PRUNE_LOG"
  [[ -f "$CKPT" ]] || { echo "[FATAL] 剪枝后未找到: $CKPT" >&2; exit 1; }
else
  echo "[INFO] RUN_PRUNE=0 — 用已有权重: $CKPT"
  [[ -f "$CKPT" ]] || { echo "[FATAL] 缺少权重，请设对 JOB_ID 或 RUN_PRUNE=1" >&2; exit 1; }
fi

echo ""; echo ">>> 稀疏度检查: T5-only ATV should prune T5 and keep non-T5 modules dense"
python scripts/blip2/check_ckpt_sparsity.py \
  --ckpt "$CKPT" --tag "atv_cc3m_t5only_${JOB_STAMP}" --expect_t5 0.5 --tol 0.05 \
  --out_csv "$SPARSITY_CSV"
append_prune_provenance

if [[ "$RUN_EVAL" != "1" ]]; then
  echo "[INFO] RUN_EVAL!=1 — 跳过评测"; exit 0
fi

: > "$LAVIS_METRICS_JSONL"
EVAL_TAG="atv_cc3m_t5only_${JOB_STAMP}"
OKVQA_JOB="okvqa_eval_atv_cc3m_${JOB_STAMP}_fullval"
four_bench_eval "$CKPT" "$EVAL_TAG" "$OKVQA_JOB" "$MASTER_PORT_OKVQA"

if [[ "$VALIDATE_REPORT" == "1" ]]; then
  echo ""; echo "========== ATV migration validation report =========="
  if [[ -f "$ATV_ROOT/qwen/activation_aware_pruner.py" ]]; then
    python scripts/blip2/validate_atv_migration.py \
      --original_atv_root "$ATV_ROOT" \
      --lavis_root "$REPO_ROOT" \
      --out_dir "$REPORT_DIR" \
      --atv_log "alpha${ATV_ALPHA}=$PRUNE_LOG" \
      --sparsity_csv "$SPARSITY_CSV" \
      --prune_provenance_csv "$PRUNE_PROVENANCE_CSV" \
      --token_mask_csv "$REPORT_DIR/token_mask_integrity.csv" \
      --importance_csv "$REPORT_DIR/importance_distribution.csv" \
      --selected_query_csv "$REPORT_DIR/selected_query_frequency.csv" \
      --metrics_jsonl "$LAVIS_METRICS_JSONL" \
      --okvqa_eval_txt "$EVAL_TAG=$REPO_ROOT/lavis/output/BLIP2/OKVQA/$OKVQA_JOB/evaluate.txt" \
      --eval_method "atv" \
      --eval_calibration "$EVAL_TAG" \
      --eval_seed "$SEED" \
      --eval_alpha "$ATV_ALPHA" \
      --eval_t5_sparsity "0.5" \
      --eval_vit_sparsity "0.0"
  else
    echo "[WARN] skip validate_atv_migration.py: ATV_ROOT not found: $ATV_ROOT"
  fi
fi

echo "[INFO] metrics jsonl: $LAVIS_METRICS_JSONL"
echo "[INFO] validation report: $REPORT_DIR"
echo ""
echo "[INFO] 完成。ATV 权重: $CKPT"
echo "[INFO] MMBench 全量 + 三模型对比可用: scripts/blip2/run_pure_wanda_cc3m_split_joint_dense_mmbench_full.sh"
echo "       （把本 ATV ckpt 以 CKPT_JOINT=$CKPT MODELS=joint 传入即可与 dense/Wanda 同表）"
