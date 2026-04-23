#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# MathVista / 多标定评测套件（单脚本）
#
# Sampler 组合：**仅 seed 30 与 42**（不再使用 44）。
#
# 1) Phase A 剪枝：
#    - LAVIS_backup：MathVista 标定 ×2（TAMP，s30 + s42）
#    - ECoFLaP：MathVista 标定（Wanda+MEZO+block），默认 **只剪 s42**；s30 视为已有，不重复剪
#      （若需重剪 s30：RUN_PRUNE_ECOFLAP_MATHVISTA_S30=1）
# 2) Phase B：LAVIS_backup 两颗 MathVista 权重 → 各跑四项 benchmark
# 3) Phase B′：ECoFLaP 两颗 MathVista 权重（已有 s30 + 新 s42）→ 各跑四项
# 4) Phase C/D：ECoFLaP / LAVIS_backup 三标定（OKVQA overall / MMBench / MMMU）× s30/s42 → 仅 MathVista MC
# 5) Phase E/F：ECoFLaP CC3M、LAVIS_backup CC3M → 各四项
#
# 环境变量（常用）：
#   RUN_PRUNE              默认 0：跳过 Phase A（仅跑 B–F，适合 A 已跑完）
#                          设 1 重跑 Phase A（LB TAMP s30/s42 + ECoFLaP Wanda 按下方开关）
#   RUN_EVAL               默认 1：Phase B–F
#   PRUNING_CALIB_BATCH    默认 8（仅 ECoFLaP MathVista Wanda 剪枝）
#   MATHVISTA_CALIB_DATE_TAG   job_id 中段，默认 $(date +%m%d)
#   THREE_CALIB_DATE_TAG       三标定 pth 中段；未设时优先从 pruned_checkpoint 的
#                              okvqa_cf_0.5_overall_*_s30.pth 推断（与 MathVista 日期解耦）
#   RUN_EVAL_PHASE_B / BP / CD / EF   默认均为 1；仅重跑 C/D 可设 B=0 BP=0 EF=0、CD=1
#   仅跑 6 次 OKVQA overall（MathVista×4 仓库 + CC3M×2）：见仓库根目录 run_okvqa_six_mathvista_cc3m.sh
#   RUN_PRUNE_ECOFLAP_MATHVISTA      默认 1；设 0 跳过整段 ECoFLaP MathVista 剪枝
#   RUN_PRUNE_ECOFLAP_MATHVISTA_S30  默认 0（不重剪 s30）；设 1 则也剪 s30
#   RUN_PRUNE_ECOFLAP_MATHVISTA_S42  默认 1
#   ECOFLAP_CKPT_* / LAVIS_CKPT_*    三标定权重路径覆盖（S30/S42）
#
# 依赖：MathVista MC 数据、MMBench/MMMU、三标定权重、CC3M 权重等。
# =============================================================================

AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
ECOFLAP_ROOT="$AUTODL_TMP/ECoFLaP/LAVIS"
LB_ROOT="$AUTODL_TMP/LAVIS_backup"
EVAL_JSON="${EVAL_JSON:-$AUTODL_TMP/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json}"
IMAGES_DIR="${IMAGES_DIR:-$AUTODL_TMP/MathVista_eval_testmini_mc/images}"

MATHVISTA_CFG="lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa_mathvista.yaml"
SEEDS=(30 42)

RUN_PRUNE="${RUN_PRUNE:-0}"
RUN_EVAL="${RUN_EVAL:-1}"

# MathVista 权重文件名：okvqa_cf_0.5_mathvista_overall_${MATHVISTA_CALIB_DATE_TAG}_s{30,42}.pth
# 未设置且跳过 A 时，尝试从 LAVIS_backup 的 s30 文件推断；否则用当天 mmdd（易与 A 不一致，请显式 export）
MATHVISTA_CALIB_DATE_TAG="${MATHVISTA_CALIB_DATE_TAG:-}"
if [[ -z "$MATHVISTA_CALIB_DATE_TAG" && "$RUN_PRUNE" == "0" ]]; then
  _MV_INF="$(ls -1 "$LB_ROOT/pruned_checkpoint"/okvqa_cf_0.5_mathvista_overall_*_s30.pth 2>/dev/null | head -1 || true)"
  if [[ -n "$_MV_INF" ]]; then
    MATHVISTA_CALIB_DATE_TAG="$(basename "$_MV_INF" .pth | sed -n 's/^okvqa_cf_0.5_mathvista_overall_\(.*\)_s30$/\1/p')"
    echo "[INFO] MATHVISTA_CALIB_DATE_TAG 由 LB pruned_checkpoint 推断: ${MATHVISTA_CALIB_DATE_TAG}"
  fi
fi
MATHVISTA_CALIB_DATE_TAG="${MATHVISTA_CALIB_DATE_TAG:-$(date +%m%d)}"

THREE_CALIB_DATE_TAG="${THREE_CALIB_DATE_TAG:-}"
if [[ -z "$THREE_CALIB_DATE_TAG" ]]; then
  _TC_INF="$(ls -1 "$LB_ROOT/pruned_checkpoint"/okvqa_cf_0.5_overall_*_s30.pth 2>/dev/null | head -1 || true)"
  if [[ -z "$_TC_INF" ]]; then
    _TC_INF="$(ls -1 "$ECOFLAP_ROOT/pruned_checkpoint"/okvqa_cf_0.5_overall_*_s30.pth 2>/dev/null | head -1 || true)"
  fi
  if [[ -n "$_TC_INF" ]]; then
    THREE_CALIB_DATE_TAG="$(basename "$_TC_INF" .pth | sed -n 's/^okvqa_cf_0.5_overall_\(.*\)_s30$/\1/p')"
    echo "[INFO] THREE_CALIB_DATE_TAG 由 okvqa_cf_0.5_overall_*_s30.pth 推断: ${THREE_CALIB_DATE_TAG}"
  fi
fi
THREE_CALIB_DATE_TAG="${THREE_CALIB_DATE_TAG:-$MATHVISTA_CALIB_DATE_TAG}"

PRUNING_CALIB_BATCH="${PRUNING_CALIB_BATCH:-8}"

RUN_EVAL_PHASE_B="${RUN_EVAL_PHASE_B:-1}"
RUN_EVAL_PHASE_BP="${RUN_EVAL_PHASE_BP:-1}"
RUN_EVAL_PHASE_CD="${RUN_EVAL_PHASE_CD:-1}"
RUN_EVAL_PHASE_EF="${RUN_EVAL_PHASE_EF:-1}"

RUN_PRUNE_ECOFLAP_MATHVISTA="${RUN_PRUNE_ECOFLAP_MATHVISTA:-1}"
RUN_PRUNE_ECOFLAP_MATHVISTA_S30="${RUN_PRUNE_ECOFLAP_MATHVISTA_S30:-0}"
RUN_PRUNE_ECOFLAP_MATHVISTA_S42="${RUN_PRUNE_ECOFLAP_MATHVISTA_S42:-1}"

MASTER_PORT_PRUNE="${MASTER_PORT_PRUNE_START:-29810}"

# CC3M job_id（可覆盖）
ECOFLAP_CC3M_JOB_ID="${ECOFLAP_CC3M_JOB_ID:-cc3m_calib128-blipt5_wanda_pruner_0.5-1.0-1.0_MEZO-GradOnly_sum0.6_block_bs8}"
LB_CC3M_JOB_ID="${LB_CC3M_JOB_ID:-cc3m_calib128-blipt5_tamp_pruner_0.5-1.0-1.0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1

export_hf_common() {
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export HF_HOME="${HF_HOME:-$AUTODL_TMP/cache_moved/huggingface}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export BERT_BASE_UNCASED_SNAPSHOT="${BERT_BASE_UNCASED_SNAPSHOT:-$AUTODL_TMP/cache_moved/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594}"
  export FLAN_T5_XL_SNAPSHOT="${FLAN_T5_XL_SNAPSHOT:-$AUTODL_TMP/cache_moved/huggingface/hub/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001}"
}

export MMBENCH_ROOT="${MMBENCH_ROOT:-$AUTODL_TMP/MMBench_eval}"
export MMMU_ROOT="${MMMU_ROOT:-$AUTODL_TMP/MMMU_single_image}"
export MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
export MMMU_SPLIT="${MMMU_SPLIT:-test}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"

export_hf_common

# --- 三标定 job_id（+ _s30 / _s42）---
job_threecalib_okvqa_train_overall() { echo "okvqa_cf_0.5_overall_${THREE_CALIB_DATE_TAG}_s$1"; }
job_threecalib_mmbench_calib() { echo "okvqa_cf_0.5_MMBench_${THREE_CALIB_DATE_TAG}_s$1"; }
job_threecalib_mmu_overall() { echo "okvqa_cf_0.5_MMMU_overall_${THREE_CALIB_DATE_TAG}_s$1"; }

job_ecoflap_okvqa_train_overall() { job_threecalib_okvqa_train_overall "$1"; }
job_ecoflap_mmbench_calib() { job_threecalib_mmbench_calib "$1"; }
job_ecoflap_mmu_overall() { job_threecalib_mmu_overall "$1"; }

# ECoFLaP / LB 三标定路径（可 ECOFLAP_CKPT_* / LAVIS_CKPT_* 覆盖，后缀 S30/S42）
ecoflap_pth_okvqa() {
  local s="$1"
  local ev
  ev=$(eval echo \"\$\{ECOFLAP_CKPT_OKVQA_S${s}:-\}\")
  if [[ -n "$ev" ]]; then echo "$ev"
  else echo "$ECOFLAP_ROOT/pruned_checkpoint/$(job_threecalib_okvqa_train_overall "$s").pth"
  fi
}
ecoflap_pth_mmbench() {
  local s="$1"
  local ev
  ev=$(eval echo \"\$\{ECOFLAP_CKPT_MMBENCH_S${s}:-\}\")
  if [[ -n "$ev" ]]; then echo "$ev"
  else echo "$ECOFLAP_ROOT/pruned_checkpoint/$(job_threecalib_mmbench_calib "$s").pth"
  fi
}
ecoflap_pth_mmu() {
  local s="$1"
  local ev
  ev=$(eval echo \"\$\{ECOFLAP_CKPT_MMU_S${s}:-\}\")
  if [[ -n "$ev" ]]; then echo "$ev"
  else echo "$ECOFLAP_ROOT/pruned_checkpoint/$(job_threecalib_mmu_overall "$s").pth"
  fi
}

lb_pth_okvqa() {
  local s="$1"
  local ev
  ev=$(eval echo \"\$\{LAVIS_CKPT_OKVQA_S${s}:-\}\")
  if [[ -n "$ev" ]]; then echo "$ev"
  else echo "$LB_ROOT/pruned_checkpoint/$(job_threecalib_okvqa_train_overall "$s").pth"
  fi
}
lb_pth_mmbench() {
  local s="$1"
  local ev
  ev=$(eval echo \"\$\{LAVIS_CKPT_MMBENCH_S${s}:-\}\")
  if [[ -n "$ev" ]]; then echo "$ev"
  else echo "$LB_ROOT/pruned_checkpoint/$(job_threecalib_mmbench_calib "$s").pth"
  fi
}
lb_pth_mmu() {
  local s="$1"
  local ev
  ev=$(eval echo \"\$\{LAVIS_CKPT_MMU_S${s}:-\}\")
  if [[ -n "$ev" ]]; then echo "$ev"
  else echo "$LB_ROOT/pruned_checkpoint/$(job_threecalib_mmu_overall "$s").pth"
  fi
}

# MathVista 标定剪枝/评测共用 job_id（两仓库同形）
job_id_mathvista_ckpt() {
  local s="$1"
  echo "okvqa_cf_0.5_mathvista_overall_${MATHVISTA_CALIB_DATE_TAG}_s${s}"
}
ckpt_lb_mathvista() { echo "$LB_ROOT/pruned_checkpoint/$(job_id_mathvista_ckpt "$1").pth"; }
ckpt_ecoflap_mathvista() { echo "$ECOFLAP_ROOT/pruned_checkpoint/$(job_id_mathvista_ckpt "$1").pth"; }
ckpt_ecoflap_cc3m() { echo "$ECOFLAP_ROOT/pruned_checkpoint/${ECOFLAP_CC3M_JOB_ID}.pth"; }
ckpt_lb_cc3m() { echo "$LB_ROOT/pruned_checkpoint/${LB_CC3M_JOB_ID}.pth"; }

mathvista_mc_eval_run() {
  local repo_root="$1"
  local ckpt="$2"
  local calib_tag="$3"
  local bench="${4:-MathVista_MC}"
  if [[ ! -f "$ckpt" ]]; then
    echo "[WARN] skip MathVista MC — missing: $ckpt"
    return 0
  fi
  if [[ ! -f "$EVAL_JSON" ]]; then
    echo "[FATAL] eval JSON missing: $EVAL_JSON"
    exit 1
  fi
  export LAVIS_REPO_ROOT="$repo_root"
  export CKPT_PATH="$ckpt"
  export LAVIS_EVAL_CALIB_TAG="$calib_tag"
  export LAVIS_METRICS_BENCHMARK="$bench"
  if [[ -z "${LAVIS_METRICS_JSONL:-}" ]]; then
    echo "[WARN] LAVIS_METRICS_JSONL unset; MathVista 指标不会写入 jsonl"
  fi
  (
    cd "$repo_root"
    python scripts/blip2/mathvista_mc_eval.py \
      --eval_json "$EVAL_JSON" \
      --images_dir "$IMAGES_DIR" \
      --ckpt "$CKPT_PATH" \
      --batch_size "$EVAL_BATCH_SIZE" \
      --device cuda
  )
  echo "[OK] MathVista MC: $ckpt"
}

run_four_evals() {
  local repo_root="$1"
  local ckpt="$2"
  local calib_tag="$3"
  local eval_job_okvqa="$4"
  local p_okvqa_base="$5"
  local mv_tag_suffix="$6"

  if [[ ! -f "$ckpt" ]]; then
    echo "[WARN] skip four-evals — missing: $ckpt"
    return 0
  fi

  export LAVIS_EVAL_CALIB_TAG="$calib_tag"
  cd "$repo_root"

  export LAVIS_METRICS_BENCHMARK="MMBench"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMBENCH_ROOT" --split "$MMBENCH_SPLIT" --ckpt "$ckpt" \
    --batch_size "$EVAL_BATCH_SIZE" --device cuda --overall_only

  local P_OKVQA=$((p_okvqa_base + ${LAVIS_DISTRIBUTED_SAMPLER_SEED:-0}))
  # LAVIS_backup 默认 pruning_method 为 TAMP，无 prune_spec 时会崩；评测只加载 ckpt 需显式 Wanda
  python -m torch.distributed.run --nproc_per_node=1 --master_port="$P_OKVQA" evaluate_blip.py \
    --cfg-path lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml \
    --pruning_method blipt5_wanda_pruner \
    --t5_pruned_checkpoint "$ckpt" --vit_pruned_checkpoint "$ckpt" \
    --job_id "$eval_job_okvqa"

  export LAVIS_METRICS_BENCHMARK="MMMU"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMMU_ROOT" --split "$MMMU_SPLIT" --ckpt "$ckpt" \
    --batch_size "$EVAL_BATCH_SIZE" --device cuda --overall_only

  mathvista_mc_eval_run "$repo_root" "$ckpt" "${calib_tag}_${mv_tag_suffix}" "MathVista_MC"
}

collect_summary() {
  local repo_root="$1"
  local collect_py="$2"
  local jsonl="$3"
  local md="$4"
  local tsv="$5"
  shift 5
  (
    cd "$repo_root"
    python "scripts/blip2/$collect_py" \
      --repo-root "$repo_root" \
      --metrics-jsonl "$jsonl" \
      --out-md "$md" \
      --out-tsv "$tsv" \
      "$@"
  )
}

prune_ecoflap_mathvista_one() {
  local SEED="$1"
  export LAVIS_DISTRIBUTED_SAMPLER_SEED="$SEED"
  local JOB_ID
  JOB_ID="$(job_id_mathvista_ckpt "$SEED")"
  local P="$MASTER_PORT_PRUNE"
  MASTER_PORT_PRUNE=$((MASTER_PORT_PRUNE + 1))
  echo "[PRUNE] ECoFLaP MathVista Wanda+MEZO seed=$SEED job_id=$JOB_ID port=$P"
  (
    cd "$ECOFLAP_ROOT"
    python -m torch.distributed.run --nproc_per_node=1 --master_port="$P" evaluate_blip.py \
      --cfg-path "$MATHVISTA_CFG" \
      --pruning_method 'blipt5_wanda_pruner' --save_pruned_model \
      --score_method MEZO-GradOnly_sum --sparsity_ratio_granularity block \
      --max_sparsity_per_layer 0.6 --prunining_dataset_batch_size "${PRUNING_CALIB_BATCH}" \
      --t5_prune_spec 24-0.5-1.0-1.0 --vit_prune_spec 39-0.5-1.0-1.0 \
      --job_id "$JOB_ID"
  )
  local CK
  CK="$(ckpt_ecoflap_mathvista "$SEED")"
  [[ -f "$CK" ]] || { echo "[FATAL] ECoFLaP 剪枝后未找到: $CK"; exit 1; }
  echo "[OK] $CK"
}

SUITE_STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_DIR_ECO="$ECOFLAP_ROOT/lavis/output/BLIP2"
SUMMARY_DIR_LB="$LB_ROOT/lavis/output/BLIP2"
mkdir -p "$SUMMARY_DIR_ECO" "$SUMMARY_DIR_LB"

JSONL_LB_MV="$SUMMARY_DIR_LB/suite_lb_mathvista_ckpts_${SUITE_STAMP}.jsonl"
JSONL_ECO_MV="$SUMMARY_DIR_ECO/suite_ecoflap_mathvista_ckpts_${SUITE_STAMP}.jsonl"
JSONL_ECO_CROSS="$SUMMARY_DIR_ECO/suite_ecoflap_threecalib_mathvista_${SUITE_STAMP}.jsonl"
JSONL_LB_CROSS="$SUMMARY_DIR_LB/suite_lavisbackup_threecalib_mathvista_${SUITE_STAMP}.jsonl"
JSONL_ECO_CC3M="$SUMMARY_DIR_ECO/suite_ecoflap_cc3m_${SUITE_STAMP}.jsonl"
JSONL_LB_CC3M="$SUMMARY_DIR_LB/suite_lavisbackup_cc3m_${SUITE_STAMP}.jsonl"
: > "$JSONL_LB_MV"
: > "$JSONL_ECO_MV"
: > "$JSONL_ECO_CROSS"
: > "$JSONL_LB_CROSS"
: > "$JSONL_ECO_CC3M"
: > "$JSONL_LB_CC3M"

# --------------------------------------------------------------------------- #
# Phase A — MathVista 标定剪枝：LAVIS_backup（s30/s42 TAMP）+ ECoFLaP（默认仅 s42 Wanda）
# --------------------------------------------------------------------------- #
if [[ "$RUN_PRUNE" == "1" ]]; then
  echo "========== Phase A: MathVista prune | DATE=${MATHVISTA_CALIB_DATE_TAG} | SEEDS=${SEEDS[*]} =========="

  echo "---- A1 LAVIS_backup TAMP s30 s42 ----"
  for SEED in "${SEEDS[@]}"; do
    export LAVIS_DISTRIBUTED_SAMPLER_SEED="$SEED"
    JOB_ID="$(job_id_mathvista_ckpt "$SEED")"
    P="$MASTER_PORT_PRUNE"
    MASTER_PORT_PRUNE=$((MASTER_PORT_PRUNE + 1))
    echo "[PRUNE] LAVIS_backup seed=$SEED job_id=$JOB_ID port=$P"
    (
      cd "$LB_ROOT"
      python -m torch.distributed.run --nproc_per_node=1 --master_port="$P" evaluate_blip.py \
        --cfg-path "$MATHVISTA_CFG" \
        --pruning_method blipt5_tamp_pruner --save_pruned_model \
        --t5_prune_spec 24-0.5-1.0-1.0 --vit_prune_spec 39-0.5-1.0-1.0 \
        --job_id "$JOB_ID"
    )
    CK="$(ckpt_lb_mathvista "$SEED")"
    [[ -f "$CK" ]] || { echo "[FATAL] 剪枝后未找到: $CK"; exit 1; }
    echo "[OK] $CK"
  done

  if [[ "$RUN_PRUNE_ECOFLAP_MATHVISTA" == "1" ]]; then
    echo "---- A2 ECoFLaP Wanda+MEZO（默认只剪 s42；s30 设 RUN_PRUNE_ECOFLAP_MATHVISTA_S30=1 才剪）----"
    if [[ "$RUN_PRUNE_ECOFLAP_MATHVISTA_S30" == "1" ]]; then
      prune_ecoflap_mathvista_one 30
    else
      echo "[SKIP] ECoFLaP MathVista s30 prune（沿用已有权重: $(ckpt_ecoflap_mathvista 30)）"
    fi
    if [[ "$RUN_PRUNE_ECOFLAP_MATHVISTA_S42" == "1" ]]; then
      prune_ecoflap_mathvista_one 42
    else
      echo "[SKIP] ECoFLaP MathVista s42 prune（RUN_PRUNE_ECOFLAP_MATHVISTA_S42=0）"
    fi
  else
    echo "[SKIP] ECoFLaP MathVista 剪枝（RUN_PRUNE_ECOFLAP_MATHVISTA=0）"
  fi

  echo "[DONE] Phase A"
else
  echo "[SKIP] Phase A (RUN_PRUNE=0)"
fi

if [[ "$RUN_EVAL" != "1" ]]; then
  echo "[SKIP] Phase B–F (RUN_EVAL=0)"
  echo "========== ALL DONE =========="
  exit 0
fi

echo "========== Phase B–F: eval | MATHVISTA_CALIB_DATE_TAG=${MATHVISTA_CALIB_DATE_TAG} | THREE_CALIB_DATE_TAG=${THREE_CALIB_DATE_TAG} =========="

echo "---------- Phase A 产出的 MathVista 权重路径（B / B′ 使用）----------"
for _s in 30 42; do
  echo "  LAVIS_backup:  $(ckpt_lb_mathvista "$_s")"
  echo "  ECoFLaP:       $(ckpt_ecoflap_mathvista "$_s")"
done
if [[ "$RUN_PRUNE" == "0" && ( "$RUN_EVAL_PHASE_B" == "1" || "$RUN_EVAL_PHASE_BP" == "1" ) ]]; then
  _mv_miss=0
  for _s in 30 42; do
    [[ -f "$(ckpt_lb_mathvista "$_s")" ]] || { echo "[FATAL] 缺少 LB MathVista ckpt（检查 MATHVISTA_CALIB_DATE_TAG）: $(ckpt_lb_mathvista "$_s")"; _mv_miss=1; }
    [[ -f "$(ckpt_ecoflap_mathvista "$_s")" ]] || { echo "[FATAL] 缺少 ECoFLaP MathVista ckpt: $(ckpt_ecoflap_mathvista "$_s")"; _mv_miss=1; }
  done
  [[ "$_mv_miss" -eq 0 ]] || { echo "[HINT] 导出与 Phase A 一致的日期，例如: export MATHVISTA_CALIB_DATE_TAG=0403"; exit 1; }
fi

# --------------------------------------------------------------------------- #
# Phase B — LAVIS_backup：MathVista s30 / s42 → 各四项
# --------------------------------------------------------------------------- #
if [[ "$RUN_EVAL_PHASE_B" == "1" ]]; then
  echo "---------- Phase B: LAVIS_backup MathVista ckpts × 4 benchmarks ----------"
  export LAVIS_METRICS_JSONL="$JSONL_LB_MV"
  for SEED in "${SEEDS[@]}"; do
    export LAVIS_DISTRIBUTED_SAMPLER_SEED="$SEED"
    CKPT_PATH="$(ckpt_lb_mathvista "$SEED")"
    MID="${MATHVISTA_CALIB_DATE_TAG}_s${SEED}"
    TAG="MathVistaoverallLb_samplerSeed${SEED}_ckpt${MID}"
    EJOB="okvqa_eval_calibMathVistaLb_samplerSeed${SEED}_ckpt${MID}_fullval"
    [[ -f "$CKPT_PATH" ]] || { echo "[FATAL] 缺权重: $CKPT_PATH"; exit 1; }
    echo "[EVAL LB MV] seed=$SEED ckpt=$CKPT_PATH"
    run_four_evals "$LB_ROOT" "$CKPT_PATH" "$TAG" "$EJOB" 30911 "mv_mc_lb"
  done
  collect_summary "$LB_ROOT" "collect_lavisbackup_eval_summary.py" "$JSONL_LB_MV" \
    "$SUMMARY_DIR_LB/suite_lb_mathvista_ckpts_${SUITE_STAMP}.md" \
    "$SUMMARY_DIR_LB/suite_lb_mathvista_ckpts_${SUITE_STAMP}.tsv" \
    --suites \
    "MathVistaoverallLb_samplerSeed30_ckpt${MATHVISTA_CALIB_DATE_TAG}_s30:okvqa_eval_calibMathVistaLb_samplerSeed30_ckpt${MATHVISTA_CALIB_DATE_TAG}_s30_fullval" \
    "MathVistaoverallLb_samplerSeed42_ckpt${MATHVISTA_CALIB_DATE_TAG}_s42:okvqa_eval_calibMathVistaLb_samplerSeed42_ckpt${MATHVISTA_CALIB_DATE_TAG}_s42_fullval"
else
  echo "[SKIP] Phase B (RUN_EVAL_PHASE_B=0)"
fi

# --------------------------------------------------------------------------- #
# Phase B′ — ECoFLaP：MathVista s30（已有）+ s42（新剪）→ 各四项
# --------------------------------------------------------------------------- #
if [[ "$RUN_EVAL_PHASE_BP" == "1" ]]; then
  echo "---------- Phase B′: ECoFLaP MathVista ckpts × 4 benchmarks ----------"
  export LAVIS_METRICS_JSONL="$JSONL_ECO_MV"
  for SEED in "${SEEDS[@]}"; do
    export LAVIS_DISTRIBUTED_SAMPLER_SEED="$SEED"
    CKPT_PATH="$(ckpt_ecoflap_mathvista "$SEED")"
    MID="${MATHVISTA_CALIB_DATE_TAG}_s${SEED}"
    TAG="MathVistaoverallEcoflap_samplerSeed${SEED}_ckpt${MID}"
    EJOB="okvqa_eval_calibMathVistaEcoflap_samplerSeed${SEED}_ckpt${MID}_fullval"
    [[ -f "$CKPT_PATH" ]] || { echo "[FATAL] 缺 ECoFLaP MathVista 权重: $CKPT_PATH（先 Phase A 或检查 DATE_TAG）"; exit 1; }
    echo "[EVAL ECO MV] seed=$SEED ckpt=$CKPT_PATH"
    run_four_evals "$ECOFLAP_ROOT" "$CKPT_PATH" "$TAG" "$EJOB" 29911 "mv_mc_eco"
  done
  collect_summary "$ECOFLAP_ROOT" "collect_ecoflap_eval_summary.py" "$JSONL_ECO_MV" \
    "$SUMMARY_DIR_ECO/suite_ecoflap_mathvista_ckpts_${SUITE_STAMP}.md" \
    "$SUMMARY_DIR_ECO/suite_ecoflap_mathvista_ckpts_${SUITE_STAMP}.tsv" \
    --suites \
    "MathVistaoverallEcoflap_samplerSeed30_ckpt${MATHVISTA_CALIB_DATE_TAG}_s30:okvqa_eval_calibMathVistaEcoflap_samplerSeed30_ckpt${MATHVISTA_CALIB_DATE_TAG}_s30_fullval" \
    "MathVistaoverallEcoflap_samplerSeed42_ckpt${MATHVISTA_CALIB_DATE_TAG}_s42:okvqa_eval_calibMathVistaEcoflap_samplerSeed42_ckpt${MATHVISTA_CALIB_DATE_TAG}_s42_fullval"
else
  echo "[SKIP] Phase B′ (RUN_EVAL_PHASE_BP=0)"
fi

if [[ "$RUN_EVAL_PHASE_CD" == "1" ]]; then
  echo "---------- 三标定权重路径（Phase C/D，THREE_CALIB_DATE_TAG=${THREE_CALIB_DATE_TAG}）----------"
  for _s in 30 42; do
    echo "  ECo OKVQA:   $(ecoflap_pth_okvqa "$_s")"
    echo "  ECo MMBench: $(ecoflap_pth_mmbench "$_s")"
    echo "  ECo MMMU:    $(ecoflap_pth_mmu "$_s")"
    echo "  LB  OKVQA:   $(lb_pth_okvqa "$_s")"
    echo "  LB  MMBench: $(lb_pth_mmbench "$_s")"
    echo "  LB  MMMU:    $(lb_pth_mmu "$_s")"
  done
fi

# --------------------------------------------------------------------------- #
# Phase C/D — ECoFLaP：三标定 × s30 / s42 → 仅 MathVista MC
# --------------------------------------------------------------------------- #
if [[ "$RUN_EVAL_PHASE_CD" == "1" ]]; then
  echo "---------- Phase C/D (1/2): ECoFLaP three-calib ckpts → MathVista only ----------"
  export LAVIS_METRICS_JSONL="$JSONL_ECO_CROSS"
  for SEED in "${SEEDS[@]}"; do
    export LAVIS_DISTRIBUTED_SAMPLER_SEED="$SEED"
    for triple in \
      "okvqa|$(ecoflap_pth_okvqa "$SEED")|OKVQAtrain_${THREE_CALIB_DATE_TAG}_s${SEED}" \
      "mmbench|$(ecoflap_pth_mmbench "$SEED")|MMBench_${THREE_CALIB_DATE_TAG}_s${SEED}" \
      "mmu|$(ecoflap_pth_mmu "$SEED")|MMMUoverall_${THREE_CALIB_DATE_TAG}_s${SEED}"; do
      IFS='|' read -r _which _ck _shorttag <<< "$triple"
      mathvista_mc_eval_run "$ECOFLAP_ROOT" "$_ck" "ecoflap_${_shorttag}_samplerSeed${SEED}_mv_mc" "MathVista_MC"
    done
  done
  echo "[INFO] ECoFLaP 三标定 MathVista: $JSONL_ECO_CROSS"

  # --------------------------------------------------------------------------- #
  # Phase C/D — LAVIS_backup：三标定 × s30 / s42 → 仅 MathVista MC
  # --------------------------------------------------------------------------- #
  echo "---------- Phase C/D (2/2): LAVIS_backup three-calib ckpts → MathVista only ----------"
  export LAVIS_METRICS_JSONL="$JSONL_LB_CROSS"
  for SEED in "${SEEDS[@]}"; do
    export LAVIS_DISTRIBUTED_SAMPLER_SEED="$SEED"
    for triple in \
      "okvqa|$(lb_pth_okvqa "$SEED")|OKVQAtrain_${THREE_CALIB_DATE_TAG}_s${SEED}" \
      "mmbench|$(lb_pth_mmbench "$SEED")|MMBench_${THREE_CALIB_DATE_TAG}_s${SEED}" \
      "mmu|$(lb_pth_mmu "$SEED")|MMMUoverall_${THREE_CALIB_DATE_TAG}_s${SEED}"; do
      IFS='|' read -r _which _ck _shorttag <<< "$triple"
      mathvista_mc_eval_run "$LB_ROOT" "$_ck" "lavisbackup_${_shorttag}_samplerSeed${SEED}_mv_mc" "MathVista_MC"
    done
  done
  echo "[INFO] LAVIS_backup 三标定 MathVista: $JSONL_LB_CROSS"
else
  echo "[SKIP] Phase C/D (RUN_EVAL_PHASE_CD=0)"
fi

# --------------------------------------------------------------------------- #
# Phase E — ECoFLaP CC3M → 四项
# --------------------------------------------------------------------------- #
if [[ "$RUN_EVAL_PHASE_EF" == "1" ]]; then
  echo "---------- Phase E: ECoFLaP CC3M → 4 benchmarks ----------"
  export LAVIS_METRICS_JSONL="$JSONL_ECO_CC3M"
  export LAVIS_DISTRIBUTED_SAMPLER_SEED="${LAVIS_DISTRIBUTED_SAMPLER_SEED:-30}"
  echo "  ECoFLaP CC3M ckpt: $(ckpt_ecoflap_cc3m)  (覆盖请设 ECOFLAP_CC3M_JOB_ID)"
  CCE="$(ckpt_ecoflap_cc3m)"
  TAG="CC3M_ecoflap_${ECOFLAP_CC3M_JOB_ID}"
  EJOB="okvqa_eval_ecoflapCC3M_${SUITE_STAMP}_fullval"
  run_four_evals "$ECOFLAP_ROOT" "$CCE" "$TAG" "$EJOB" 29911 "mv_mc_cc3m"
  collect_summary "$ECOFLAP_ROOT" "collect_ecoflap_eval_summary.py" "$JSONL_ECO_CC3M" \
    "$SUMMARY_DIR_ECO/suite_ecoflap_cc3m_${SUITE_STAMP}.md" \
    "$SUMMARY_DIR_ECO/suite_ecoflap_cc3m_${SUITE_STAMP}.tsv" \
    --suites "CC3M_ecoflap_${ECOFLAP_CC3M_JOB_ID}:okvqa_eval_ecoflapCC3M_${SUITE_STAMP}_fullval"

  # --------------------------------------------------------------------------- #
  # Phase F — LAVIS_backup CC3M → 四项
  # --------------------------------------------------------------------------- #
  echo "---------- Phase F: LAVIS_backup CC3M → 4 benchmarks ----------"
  export LAVIS_METRICS_JSONL="$JSONL_LB_CC3M"
  export LAVIS_DISTRIBUTED_SAMPLER_SEED="${LAVIS_DISTRIBUTED_SAMPLER_SEED:-30}"
  echo "  LAVIS_backup CC3M ckpt: $(ckpt_lb_cc3m)  (覆盖请设 LB_CC3M_JOB_ID)"
  CCL="$(ckpt_lb_cc3m)"
  TAG="CC3M_lavisbackup_${LB_CC3M_JOB_ID}"
  EJOB="okvqa_eval_lbCC3M_${SUITE_STAMP}_fullval"
  run_four_evals "$LB_ROOT" "$CCL" "$TAG" "$EJOB" 31911 "mv_mc_cc3m_lb"
  collect_summary "$LB_ROOT" "collect_lavisbackup_eval_summary.py" "$JSONL_LB_CC3M" \
    "$SUMMARY_DIR_LB/suite_lavisbackup_cc3m_${SUITE_STAMP}.md" \
    "$SUMMARY_DIR_LB/suite_lavisbackup_cc3m_${SUITE_STAMP}.tsv" \
    --suites "CC3M_lavisbackup_${LB_CC3M_JOB_ID}:okvqa_eval_lbCC3M_${SUITE_STAMP}_fullval"
else
  echo "[SKIP] Phase E/F (RUN_EVAL_PHASE_EF=0)"
fi

echo "========== ALL DONE =========="
echo "Metrics jsonl:"
echo "  LB MathVista s30/s42 四项: $JSONL_LB_MV"
echo "  ECoFLaP MathVista s30/s42 四项: $JSONL_ECO_MV"
echo "  ECoFLaP 三标定×2 仅 MathVista: $JSONL_ECO_CROSS"
echo "  LAVIS_backup 三标定×2 仅 MathVista: $JSONL_LB_CROSS"
echo "  ECoFLaP CC3M 四项: $JSONL_ECO_CC3M"
echo "  LAVIS_backup CC3M 四项: $JSONL_LB_CC3M"
