#!/usr/bin/env bash
# =============================================================================
# CC3M calib128 标定 — 四套单侧剪枝 + 分项评测
#
# 剪枝（与 cc_prefix_derivative_compute_cc3m_calib128.yaml 一致，128 条）:
#   1) ECoFLaP   Wanda+MEZO+block | 只剪 ViT
#   2) ECoFLaP   Wanda+MEZO+block | 只剪 T5
#   3) LAVIS_backup  TAMP        | 只剪 ViT
#   4) LAVIS_backup  TAMP        | 只剪 T5
#
# 评测（默认 PRUNE_VIT=0：不跑 ViT 单侧剪枝；组合评测时 ViT/T5 均从「只剪 T5」的 pth 加载，ViT 为预训练未剪）:
#   PRUNE_VIT=1 时恢复：vit 单侧 pth + t5 单侧 pth 组合
#
# 环境变量:
#   RUN_LAVISBACKUP_ONLY=1     只跑 LAVIS_backup（强制跳过剪枝 1–2 与 ECo 评测；剪枝 3–4 / LB 评测仍受 RUN_SPLIT_3/4 控制）
#   RUN_PRUNE=1 RUN_EVAL=1     剪枝 + 评测（默认均 1）；仅评测可设 RUN_PRUNE=0 并设 JOB_STAMP 与剪枝时一致
#   RUN_SPLIT_1..4             同前，控制四步剪枝是否执行
#   JOB_STAMP                  剪枝 job_id 与输出文件名；仅评测时需与剪枝时一致
#   LAVIS_DISTRIBUTED_SAMPLER_SEED / SEED   默认 42（换 seed 即换 CC3M 上采样的 128 条）
#   NUM_DATA / NUM_DATA_FIRST_STAGE         默认 128（与 calib128 对齐）
#   PRUNING_CALIB_BATCH                     默认 8（须整除 NUM_DATA）
#   MASTER_PORT_START                       剪枝四步 master_port 起始，默认 29831
#   CC3M_CFG           相对各 repo 根，默认 cc_prefix_derivative_compute_cc3m_calib128_seed20260411.yaml
#   PRUNE_VIT          默认 0：跳过剪枝 1/3（ViT 单侧），评测用 T5-only ckpt 提供 dense ViT；设 1 恢复双侧单侧剪枝+组合
# =============================================================================

set -euo pipefail

AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
ECOFLAP_ROOT="$AUTODL_TMP/ECoFLaP/LAVIS"
LB_ROOT="$AUTODL_TMP/LAVIS_backup"

# 默认使用 CC3M_calib_128_seed20260411；换回旧集可 export CC3M_CFG=.../cc_prefix_derivative_compute_cc3m_calib128.yaml
CC3M_CFG="${CC3M_CFG:-lavis/projects/blip2/eval/cc_prefix_derivative_compute_cc3m_calib128_seed20260411.yaml}"
CC3M_CFG_ECO="$CC3M_CFG"
CC3M_CFG_LB="$CC3M_CFG"

EVAL_JSON="${EVAL_JSON:-$AUTODL_TMP/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json}"
IMAGES_DIR="${IMAGES_DIR:-$AUTODL_TMP/MathVista_eval_testmini_mc/images}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1

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
RUN_SPLIT_1="${RUN_SPLIT_1:-1}"
RUN_SPLIT_2="${RUN_SPLIT_2:-1}"
RUN_SPLIT_3="${RUN_SPLIT_3:-1}"
RUN_SPLIT_4="${RUN_SPLIT_4:-1}"
RUN_LAVISBACKUP_ONLY="${RUN_LAVISBACKUP_ONLY:-0}"
PRUNE_VIT="${PRUNE_VIT:-0}"
if [[ "$RUN_LAVISBACKUP_ONLY" == "1" ]]; then
  RUN_SPLIT_1=0
  RUN_SPLIT_2=0
  echo "[INFO] RUN_LAVISBACKUP_ONLY=1 → 跳过 ECoFLaP 剪枝（SPLIT_1/2=0）"
fi
if [[ "$PRUNE_VIT" != "1" ]]; then
  RUN_SPLIT_1=0
  RUN_SPLIT_3=0
  echo "[INFO] PRUNE_VIT=0 → 跳过 ViT 单侧剪枝（SPLIT_1/3=0），评测组合用 T5-only ckpt 中的 dense ViT"
fi

SUMMARY_DIR_ECO="$ECOFLAP_ROOT/lavis/output/BLIP2"
SUMMARY_DIR_LB="$LB_ROOT/lavis/output/BLIP2"
mkdir -p "$SUMMARY_DIR_ECO" "$SUMMARY_DIR_LB"

JSONL_ECO="$SUMMARY_DIR_ECO/cc3m_split_eval_ecoflap_${JOB_STAMP}_seed${SEED}.jsonl"
JSONL_LB="$SUMMARY_DIR_LB/cc3m_split_eval_lavisbackup_${JOB_STAMP}_seed${SEED}.jsonl"
if [[ "$RUN_LAVISBACKUP_ONLY" == "1" ]]; then
  : > "$JSONL_LB"
else
  : > "$JSONL_ECO"
  : > "$JSONL_LB"
fi

# job_id 与 pruned_checkpoint 文件名一致：JOB_STAMP（时间）+ seed
JID_ECO_VIT="split_cc3m_vitonly_eco_wanda_${JOB_STAMP}_seed${SEED}"
JID_ECO_T5="split_cc3m_t5only_eco_wanda_${JOB_STAMP}_seed${SEED}"
JID_LB_VIT="split_cc3m_vitonly_lb_tamp_${JOB_STAMP}_seed${SEED}"
JID_LB_T5="split_cc3m_t5only_lb_tamp_${JOB_STAMP}_seed${SEED}"

CKPT_ECO_VIT="$ECOFLAP_ROOT/pruned_checkpoint/${JID_ECO_VIT}.pth"
CKPT_ECO_T5="$ECOFLAP_ROOT/pruned_checkpoint/${JID_ECO_T5}.pth"
CKPT_LB_VIT="$LB_ROOT/pruned_checkpoint/${JID_LB_VIT}.pth"
CKPT_LB_T5="$LB_ROOT/pruned_checkpoint/${JID_LB_T5}.pth"

mathvista_mc_eval_run() {
  local repo_root="$1"
  local ckpt_vit="$2"
  local ckpt_t5="$3"
  local calib_tag="$4"
  local bench="${5:-MathVista_MC}"
  if [[ ! -f "$ckpt_vit" || ! -f "$ckpt_t5" ]]; then
    echo "[WARN] skip MathVista MC — missing vit or t5 ckpt: $ckpt_vit | $ckpt_t5"
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
      --vit_ckpt "$ckpt_vit" --t5_ckpt "$ckpt_t5" \
      --batch_size "$EVAL_BATCH_SIZE" \
      --device cuda
  )
}

run_four_evals() {
  local repo_root="$1"
  local ckpt_vit="$2"
  local ckpt_t5="$3"
  local calib_tag="$4"
  local eval_job_okvqa="$5"
  local p_okvqa_base="$6"
  local mv_tag_suffix="$7"

  if [[ ! -f "$ckpt_vit" || ! -f "$ckpt_t5" ]]; then
    echo "[WARN] skip four-evals — missing vit or t5 ckpt: $ckpt_vit | $ckpt_t5"
    return 0
  fi

  export LAVIS_EVAL_CALIB_TAG="$calib_tag"
  cd "$repo_root"

  export LAVIS_METRICS_BENCHMARK="MMBench"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMBENCH_ROOT" --split "$MMBENCH_SPLIT" \
    --vit_ckpt "$ckpt_vit" --t5_ckpt "$ckpt_t5" \
    --batch_size "$EVAL_BATCH_SIZE" --device cuda --overall_only

  local P_OKVQA=$((p_okvqa_base + ${LAVIS_DISTRIBUTED_SAMPLER_SEED:-0}))
  python -m torch.distributed.run --nproc_per_node=1 --master_port="$P_OKVQA" evaluate_blip.py \
    --cfg-path lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml \
    --pruning_method blipt5_wanda_pruner \
    --vit_pruned_checkpoint "$ckpt_vit" --t5_pruned_checkpoint "$ckpt_t5" \
    --job_id "$eval_job_okvqa"

  export LAVIS_METRICS_BENCHMARK="MMMU"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMMU_ROOT" --split "$MMMU_SPLIT" \
    --vit_ckpt "$ckpt_vit" --t5_ckpt "$ckpt_t5" \
    --batch_size "$EVAL_BATCH_SIZE" --device cuda --overall_only

  mathvista_mc_eval_run "$repo_root" "$ckpt_vit" "$ckpt_t5" "${calib_tag}_${mv_tag_suffix}" "MathVista_MC"
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

# --- 剪枝 ---
run_eco_vit_only() {
  echo "========== [剪枝 1/4] ECoFLaP | CC3M | Wanda+MEZO | 只剪 ViT =========="
  (
    cd "$ECOFLAP_ROOT"
    python -m torch.distributed.run --nproc_per_node=1 --master_port="$1" evaluate_blip.py \
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
      --no_prune_t5 \
      --job_id "$JID_ECO_VIT"
  )
}

run_eco_t5_only() {
  echo "========== [剪枝 2/4] ECoFLaP | CC3M | Wanda+MEZO | 只剪 T5 =========="
  (
    cd "$ECOFLAP_ROOT"
    python -m torch.distributed.run --nproc_per_node=1 --master_port="$1" evaluate_blip.py \
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
      --job_id "$JID_ECO_T5"
  )
}

run_lb_vit_only() {
  echo "========== [剪枝 3/4] LAVIS_backup | CC3M | TAMP | 只剪 ViT =========="
  (
    cd "$LB_ROOT"
    python -m torch.distributed.run --nproc_per_node=1 --master_port="$1" evaluate_blip.py \
      --cfg-path "$CC3M_CFG_LB" \
      --pruning_method blipt5_tamp_pruner \
      --save_pruned_model \
      --num_data "$NUM_DATA" \
      --num_data_first_stage "$NUM_DATA_FIRST_STAGE" \
      --prunining_dataset_batch_size "$PRUNING_CALIB_BATCH" \
      --t5_prune_spec "$T5_SPEC" \
      --vit_prune_spec "$VIT_SPEC" \
      --no_prune_t5 \
      --job_id "$JID_LB_VIT"
  )
}

run_lb_t5_only() {
  echo "========== [剪枝 4/4] LAVIS_backup | CC3M | TAMP | 只剪 T5 =========="
  (
    cd "$LB_ROOT"
    python -m torch.distributed.run --nproc_per_node=1 --master_port="$1" evaluate_blip.py \
      --cfg-path "$CC3M_CFG_LB" \
      --pruning_method blipt5_tamp_pruner \
      --save_pruned_model \
      --num_data "$NUM_DATA" \
      --num_data_first_stage "$NUM_DATA_FIRST_STAGE" \
      --prunining_dataset_batch_size "$PRUNING_CALIB_BATCH" \
      --t5_prune_spec "$T5_SPEC" \
      --vit_prune_spec "$VIT_SPEC" \
      --job_id "$JID_LB_T5"
  )
}

echo "========== CC3M split prune + eval | STAMP=${JOB_STAMP} SEED=${SEED} =========="
echo "  Calib: see CC3M_CFG (default: cc3m_calib_128_seed20260411 in yaml)"
echo "  PRUNE_VIT=${PRUNE_VIT} (0=不剪 ViT / 组合用 T5-only ckpt 提供 dense ViT, 1=双侧单侧+组合)"
if [[ "$RUN_LAVISBACKUP_ONLY" == "1" ]]; then
  echo "  [MODE] RUN_LAVISBACKUP_ONLY=1 — 仅 LAVIS_backup"
elif [[ "$PRUNE_VIT" == "1" ]]; then
  echo "  ECo ckpt: $CKPT_ECO_VIT | $CKPT_ECO_T5"
else
  echo "  ECo ckpt: (无 vit-only) 组合评测 ViT 来自: $CKPT_ECO_T5"
fi
if [[ "$PRUNE_VIT" == "1" ]]; then
  echo "  LB  ckpt: $CKPT_LB_VIT | $CKPT_LB_T5"
else
  echo "  LB  ckpt: (无 vit-only) 组合评测 ViT 来自: $CKPT_LB_T5"
fi

P1="$MASTER_PORT_START"
P2=$((MASTER_PORT_START + 1))
P3=$((MASTER_PORT_START + 2))
P4=$((MASTER_PORT_START + 3))

if [[ "$RUN_PRUNE" == "1" ]]; then
  if [[ "$RUN_SPLIT_1" == "1" ]]; then run_eco_vit_only "$P1"; else echo "[SKIP] prune 1"; fi
  if [[ "$RUN_SPLIT_2" == "1" ]]; then run_eco_t5_only "$P2"; else echo "[SKIP] prune 2"; fi
  if [[ "$RUN_SPLIT_3" == "1" ]]; then run_lb_vit_only "$P3"; else echo "[SKIP] prune 3"; fi
  if [[ "$RUN_SPLIT_4" == "1" ]]; then run_lb_t5_only "$P4"; else echo "[SKIP] prune 4"; fi
else
  echo "[SKIP] all pruning (RUN_PRUNE=0)"
fi

if [[ "$RUN_EVAL" != "1" ]]; then
  echo "[SKIP] eval (RUN_EVAL=0)"
  echo "========== DONE =========="
  exit 0
fi

if [[ "$RUN_LAVISBACKUP_ONLY" != "1" ]]; then
  if [[ "$PRUNE_VIT" == "1" ]]; then
    echo "========== [评测] ECoFLaP | ViT单侧 + T5单侧 组合 → MMBench / OKVQA / MMMU / MathVista =========="
    _ECO_VIT="$CKPT_ECO_VIT"
    TAG_ECO_COMBO="CC3Msplit_eco_vit_t5_${JOB_STAMP}_seed${SEED}"
  else
    echo "========== [评测] ECoFLaP | 仅 T5 剪枝 ckpt（dense ViT）→ 四项 =========="
    _ECO_VIT="$CKPT_ECO_T5"
    TAG_ECO_COMBO="CC3Msplit_eco_t5only_densevit_${JOB_STAMP}_seed${SEED}"
  fi
  export LAVIS_METRICS_JSONL="$JSONL_ECO"
  EJOB_ECO_COMBO="okvqa_eval_cc3m_split_eco_combo_${JOB_STAMP}_seed${SEED}_fullval"

  run_four_evals "$ECOFLAP_ROOT" "$_ECO_VIT" "$CKPT_ECO_T5" "$TAG_ECO_COMBO" "$EJOB_ECO_COMBO" 29911 "mv_mc_cc3m_split_eco"

  collect_summary "$ECOFLAP_ROOT" "collect_ecoflap_eval_summary.py" "$JSONL_ECO" \
    "$SUMMARY_DIR_ECO/cc3m_split_eval_ecoflap_${JOB_STAMP}_seed${SEED}.md" \
    "$SUMMARY_DIR_ECO/cc3m_split_eval_ecoflap_${JOB_STAMP}_seed${SEED}.tsv" \
    --suites \
    "${TAG_ECO_COMBO}:${EJOB_ECO_COMBO}"
else
  echo "[SKIP] ECoFLaP 评测（RUN_LAVISBACKUP_ONLY=1）"
fi

# --- LAVIS_backup：组合 → 四项 ---
if [[ "$PRUNE_VIT" == "1" ]]; then
  echo "========== [评测] LAVIS_backup | ViT单侧 + T5单侧 组合 → MMBench / OKVQA / MMMU / MathVista =========="
  _LB_VIT="$CKPT_LB_VIT"
  TAG_LB_COMBO="CC3Msplit_lb_vit_t5_${JOB_STAMP}_seed${SEED}"
else
  echo "========== [评测] LAVIS_backup | 仅 T5 剪枝 ckpt（dense ViT）→ 四项 =========="
  _LB_VIT="$CKPT_LB_T5"
  TAG_LB_COMBO="CC3Msplit_lb_t5only_densevit_${JOB_STAMP}_seed${SEED}"
fi
export LAVIS_METRICS_JSONL="$JSONL_LB"

EJOB_LB_COMBO="okvqa_eval_cc3m_split_lb_combo_${JOB_STAMP}_seed${SEED}_fullval"

run_four_evals "$LB_ROOT" "$_LB_VIT" "$CKPT_LB_T5" "$TAG_LB_COMBO" "$EJOB_LB_COMBO" 31911 "mv_mc_cc3m_split_lb"

collect_summary "$LB_ROOT" "collect_lavisbackup_eval_summary.py" "$JSONL_LB" \
  "$SUMMARY_DIR_LB/cc3m_split_eval_lavisbackup_${JOB_STAMP}_seed${SEED}.md" \
  "$SUMMARY_DIR_LB/cc3m_split_eval_lavisbackup_${JOB_STAMP}_seed${SEED}.tsv" \
  --suites \
  "${TAG_LB_COMBO}:${EJOB_LB_COMBO}"

echo "========== ALL DONE =========="
if [[ "$RUN_LAVISBACKUP_ONLY" != "1" ]]; then
  echo "  ECo metrics jsonl: $JSONL_ECO"
fi
echo "  LB  metrics jsonl: $JSONL_LB"
