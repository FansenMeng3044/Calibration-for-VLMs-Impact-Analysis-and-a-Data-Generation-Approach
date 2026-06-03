#!/usr/bin/env bash
# =============================================================================
# CC3M calib128 标定 — 联合剪枝 ×2（ECo Wanda+MEZO / LB TAMP）+ 组合评测 ×2（各四项）
#
# 剪枝：与 cc_prefix_derivative_compute_cc3m_calib128.yaml，ViT+T5 同时剪（不设 --no_prune_*）
# 评测：每个仓库一个整模 pth → MMBench / OKVQA overall / MMMU / MathVista MC（单 ckpt 加载）
#
# 合计：2 次剪枝 + 8 次 eval 子任务（2 × 4）；仅 LAVIS_backup：RUN_LAVISBACKUP_ONLY=1 → 1 剪枝 + 4 eval
# evaluate_blip 默认已不剪 ViT；本脚本 PRUNE_VIT=1 时传 --prune_vit 做 ViT+T5 联合剪枝
#
# 环境变量:
#   RUN_LAVISBACKUP_ONLY=1     只跑 LAVIS_backup（跳过 ECoFLaP 剪枝与评测）
#   RUN_PRUNE=1 RUN_EVAL=1     默认均 1；仅评测设 RUN_PRUNE=0 且 JOB_STAMP 与剪枝一致
#   JOB_STAMP                  时间戳（t…），写入 pth / jsonl / 评测 job 名；仅评测时需与剪枝时一致
#   LAVIS_DISTRIBUTED_SAMPLER_SEED / SEED   默认 42（CC3M 128 条子集与 dataloader 可复现；改 seed 即换一批 128）
#   NUM_DATA / NUM_DATA_FIRST_STAGE         默认 128
#   PRUNING_CALIB_BATCH                     默认 8
#   MASTER_PORT_START                       剪枝两步 master_port，默认 29831
#   CC3M_CFG           相对各 repo 根，默认 cc_prefix_derivative_compute_cc3m_calib128_seed20260411.yaml
#   PRUNE_VIT          默认 0：依赖 evaluate_blip 默认（不剪 ViT）；设 1 传 --prune_vit
# =============================================================================

set -euo pipefail

AUTODL_TMP="${AUTODL_TMP:-/data/data2/mfs}"
ECOFLAP_ROOT="$AUTODL_TMP/2/ECoFLaP/LAVIS"
LB_ROOT="$AUTODL_TMP/2/LAVIS_backup"

# ECoFLaP 用 seed20260411 版 YAML；LAVIS_backup 没有该版本，用基础版
CC3M_CFG_ECO="${CC3M_CFG_ECO:-lavis/projects/blip2/eval/cc_prefix_derivative_compute_cc3m_calib128_seed20260411.yaml}"
CC3M_CFG_LB="${CC3M_CFG_LB:-lavis/projects/blip2/eval/cc_prefix_derivative_compute_cc3m_calib128.yaml}"

EVAL_JSON="${EVAL_JSON:-$AUTODL_TMP/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json}"
IMAGES_DIR="${IMAGES_DIR:-$AUTODL_TMP/MathVista_eval_testmini_mc/images}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$AUTODL_TMP/model_cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export BERT_BASE_UNCASED_SNAPSHOT="${BERT_BASE_UNCASED_SNAPSHOT:-$AUTODL_TMP/model_cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594}"
export FLAN_T5_XL_SNAPSHOT="${FLAN_T5_XL_SNAPSHOT:-$AUTODL_TMP/model_cache/huggingface/hub/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001}"

export MMBENCH_ROOT="${MMBENCH_ROOT:-$AUTODL_TMP/MMBench_eval}"
export MMMU_ROOT="${MMMU_ROOT:-$AUTODL_TMP/MMMU_single_image}"
export MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
export MMMU_SPLIT="${MMMU_SPLIT:-test}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"

SEED="${LAVIS_DISTRIBUTED_SAMPLER_SEED:-${SEED:-42}}"
export LAVIS_DISTRIBUTED_SAMPLER_SEED="$SEED"

NUM_DATA="${NUM_DATA:-128}"
NUM_DATA_FIRST_STAGE="${NUM_DATA_FIRST_STAGE:-128}"
PRUNING_CALIB_BATCH="${PRUNING_CALIB_BATCH:-8}"
T5_SPEC="${T5_PRUNE_SPEC:-24-0.5-1.0-1.0}"
VIT_SPEC="${VIT_PRUNE_SPEC:-39-0.5-1.0-1.0}"

MASTER_PORT_START="${MASTER_PORT_START:-29831}"
JOB_STAMP="${JOB_STAMP:-$(date +%Y%m%d_%H%M%S)}"

RUN_PRUNE="${RUN_PRUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_LAVISBACKUP_ONLY="${RUN_LAVISBACKUP_ONLY:-0}"
PRUNE_VIT="${PRUNE_VIT:-0}"

# --- job_id / 权重文件名：标定(cfg 基名) + 剪枝范围 + 时间 + seed ---
_cfg_basename_noext() {
  basename "$1" .yaml
}
CALIB_SLUG_ECO="$(_cfg_basename_noext "$CC3M_CFG_ECO")"
CALIB_SLUG_LB="$(_cfg_basename_noext "$CC3M_CFG_LB")"
if [[ "$PRUNE_VIT" == "1" ]]; then
  PRUNE_SCOPE_SLUG="vit_and_t5"
else
  PRUNE_SCOPE_SLUG="t5_only"
fi
# 片段: calib-<yaml> | wanda|tamp | t5_only|vit_and_t5 | t<时间戳> | seed<>
JID_ECO="calib-${CALIB_SLUG_ECO}__wandaMEZO__${PRUNE_SCOPE_SLUG}__t${JOB_STAMP}__seed${SEED}"
JID_LB="calib-${CALIB_SLUG_LB}__TAMP__${PRUNE_SCOPE_SLUG}__t${JOB_STAMP}__seed${SEED}"

SUMMARY_DIR_ECO="$ECOFLAP_ROOT/lavis/output/BLIP2"
SUMMARY_DIR_LB="$LB_ROOT/lavis/output/BLIP2"
mkdir -p "$SUMMARY_DIR_ECO" "$SUMMARY_DIR_LB"

STEM_EVAL_ECO="eval_eco__${CALIB_SLUG_ECO}__${PRUNE_SCOPE_SLUG}__t${JOB_STAMP}__seed${SEED}"
STEM_EVAL_LB="eval_lb__${CALIB_SLUG_LB}__${PRUNE_SCOPE_SLUG}__t${JOB_STAMP}__seed${SEED}"
JSONL_ECO="$SUMMARY_DIR_ECO/${STEM_EVAL_ECO}.jsonl"
JSONL_LB="$SUMMARY_DIR_LB/${STEM_EVAL_LB}.jsonl"
if [[ "$RUN_LAVISBACKUP_ONLY" == "1" ]]; then
  : > "$JSONL_LB"
else
  : > "$JSONL_ECO"
  : > "$JSONL_LB"
fi

CKPT_ECO="$ECOFLAP_ROOT/pruned_checkpoint/${JID_ECO}.pth"
CKPT_LB="$LB_ROOT/pruned_checkpoint/${JID_LB}.pth"

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
  export LAVIS_EVAL_CALIB_TAG="$calib_tag"
  export LAVIS_METRICS_BENCHMARK="$bench"
  (
    cd "$repo_root"
    python scripts/blip2/mathvista_mc_eval.py \
      --eval_json "$EVAL_JSON" \
      --images_dir "$IMAGES_DIR" \
      --ckpt "$ckpt" \
      --batch_size "$EVAL_BATCH_SIZE" \
      --device cuda
  )
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

run_eco_joint() {
  local _port="$1"
  if [[ "$PRUNE_VIT" == "1" ]]; then
    echo "========== [剪枝 1/2] ECoFLaP | CC3M | Wanda+MEZO | 联合剪枝 ViT+T5 =========="
  else
    echo "========== [剪枝 1/2] ECoFLaP | CC3M | Wanda+MEZO | 只剪 T5（PRUNE_VIT=0，已关 ViT）=========="
  fi
  (
    cd "$ECOFLAP_ROOT"
    set -- \
      python -m torch.distributed.run --nproc_per_node=1 --master_port="$_port" evaluate_blip.py \
      --cfg-path "$CC3M_CFG_ECO" \
      --pruning_method blipt5_wanda_pruner \
      --save_pruned_model \
      --score_method MEZO-GradOnly_sum \
      --sparsity_ratio_granularity block \
      --max_sparsity_per_layer 0.6 \
      --prunining_dataset_batch_size "$PRUNING_CALIB_BATCH" \
      --num_data "$NUM_DATA" \
      --num_data_first_stage "$NUM_DATA_FIRST_STAGE" \
      --t5_prune_spec "$T5_SPEC" \
      --vit_prune_spec "$VIT_SPEC" \
      --job_id "$JID_ECO"
    if [[ "$PRUNE_VIT" == "1" ]]; then
      set -- "$@" --prune_vit
    fi
    "$@"
  )
}

run_lb_joint() {
  local _port="$1"
  if [[ "$PRUNE_VIT" == "1" ]]; then
    echo "========== [剪枝 2/2] LAVIS_backup | CC3M | TAMP | 联合剪枝 ViT+T5 =========="
  else
    echo "========== [剪枝 2/2] LAVIS_backup | CC3M | TAMP | 只剪 T5（PRUNE_VIT=0，已关 ViT）=========="
  fi
  (
    cd "$LB_ROOT"
    set -- \
      python -m torch.distributed.run --nproc_per_node=1 --master_port="$_port" evaluate_blip.py \
      --cfg-path "$CC3M_CFG_LB" \
      --pruning_method blipt5_tamp_pruner \
      --save_pruned_model \
      --num_data "$NUM_DATA" \
      --num_data_first_stage "$NUM_DATA_FIRST_STAGE" \
      --prunining_dataset_batch_size "$PRUNING_CALIB_BATCH" \
      --t5_prune_spec "$T5_SPEC" \
      --vit_prune_spec "$VIT_SPEC" \
      --job_id "$JID_LB"
    if [[ "$PRUNE_VIT" == "1" ]]; then
      set -- "$@" --prune_vit
    fi
    "$@"
  )
}

echo "========== CC3M joint prune + eval | STAMP=${JOB_STAMP} SEED=${SEED} =========="
echo "  PRUNE_VIT=${PRUNE_VIT} (0=只剪 T5, 1=ViT+T5) → 文件名片段: ${PRUNE_SCOPE_SLUG}"
echo "  标定(cfg): ECo=${CALIB_SLUG_ECO} | LB=${CALIB_SLUG_LB}"
echo "  job_id(ECo)=${JID_ECO}"
echo "  job_id(LB) =${JID_LB}"
if [[ "$RUN_LAVISBACKUP_ONLY" == "1" ]]; then
  echo "  [MODE] RUN_LAVISBACKUP_ONLY=1 — 仅 LAVIS_backup"
else
  echo "  ECo ckpt: $CKPT_ECO"
fi
echo "  LB  ckpt: $CKPT_LB"

P1="$MASTER_PORT_START"
P2=$((MASTER_PORT_START + 1))

if [[ "$RUN_PRUNE" == "1" ]]; then
  if [[ "$RUN_LAVISBACKUP_ONLY" == "1" ]]; then
    run_lb_joint "$P1"
  else
    run_eco_joint "$P1"
    run_lb_joint "$P2"
  fi
else
  echo "[SKIP] pruning (RUN_PRUNE=0)"
fi

if [[ "$RUN_EVAL" != "1" ]]; then
  echo "[SKIP] eval (RUN_EVAL=0)"
  echo "========== DONE =========="
  exit 0
fi

if [[ "$RUN_LAVISBACKUP_ONLY" != "1" ]]; then
  echo "========== [评测] ECoFLaP 联合权重 → 四项 =========="
  export LAVIS_METRICS_JSONL="$JSONL_ECO"
  TAG_ECO="tag_${STEM_EVAL_ECO}"
  EJOB_ECO="okvqa_${STEM_EVAL_ECO}_fullval"
  run_four_evals "$ECOFLAP_ROOT" "$CKPT_ECO" "$TAG_ECO" "$EJOB_ECO" 29911 "mv_mc_${STEM_EVAL_ECO}"

  collect_summary "$ECOFLAP_ROOT" "collect_ecoflap_eval_summary.py" "$JSONL_ECO" \
    "$SUMMARY_DIR_ECO/${STEM_EVAL_ECO}.md" \
    "$SUMMARY_DIR_ECO/${STEM_EVAL_ECO}.tsv" \
    --suites "${TAG_ECO}:${EJOB_ECO}"
else
  echo "[SKIP] ECoFLaP 评测（RUN_LAVISBACKUP_ONLY=1）"
fi

echo "========== [评测] LAVIS_backup 联合权重 → 四项 =========="
export LAVIS_METRICS_JSONL="$JSONL_LB"
TAG_LB="tag_${STEM_EVAL_LB}"
EJOB_LB="okvqa_${STEM_EVAL_LB}_fullval"
run_four_evals "$LB_ROOT" "$CKPT_LB" "$TAG_LB" "$EJOB_LB" 31911 "mv_mc_${STEM_EVAL_LB}"

collect_summary "$LB_ROOT" "collect_lavisbackup_eval_summary.py" "$JSONL_LB" \
  "$SUMMARY_DIR_LB/${STEM_EVAL_LB}.md" \
  "$SUMMARY_DIR_LB/${STEM_EVAL_LB}.tsv" \
  --suites "${TAG_LB}:${EJOB_LB}"

echo "========== ALL DONE =========="
if [[ "$RUN_LAVISBACKUP_ONLY" != "1" ]]; then
  echo "  ECo metrics jsonl: $JSONL_ECO"
fi
echo "  LB  metrics jsonl: $JSONL_LB"
