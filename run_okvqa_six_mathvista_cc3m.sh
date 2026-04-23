#!/usr/bin/env bash
# 仅跑 6 次 OKVQA overall（full val），与 run_mathvista_prune_eval_suite.sh 中对应权重一致：
#   1–2) LAVIS_backup  MathVista 标定剪枝 s30、s42
#   3–4) ECoFLaP       MathVista 标定剪枝 s30、s42
#   5)   ECoFLaP       CC3M 剪枝
#   6)   LAVIS_backup  CC3M 剪枝
#
# 环境变量与套件一致：MATHVISTA_CALIB_DATE_TAG、ECOFLAP_CC3M_JOB_ID、LB_CC3M_JOB_ID、AUTODL_TMP 等。

set -euo pipefail

AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
ECOFLAP_ROOT="$AUTODL_TMP/ECoFLaP/LAVIS"
LB_ROOT="$AUTODL_TMP/LAVIS_backup"

MATHVISTA_CALIB_DATE_TAG="${MATHVISTA_CALIB_DATE_TAG:-}"
if [[ -z "$MATHVISTA_CALIB_DATE_TAG" ]]; then
  _MV_INF="$(ls -1 "$LB_ROOT/pruned_checkpoint"/okvqa_cf_0.5_mathvista_overall_*_s30.pth 2>/dev/null | head -1 || true)"
  if [[ -n "$_MV_INF" ]]; then
    MATHVISTA_CALIB_DATE_TAG="$(basename "$_MV_INF" .pth | sed -n 's/^okvqa_cf_0.5_mathvista_overall_\(.*\)_s30$/\1/p')"
    echo "[INFO] MATHVISTA_CALIB_DATE_TAG 推断: ${MATHVISTA_CALIB_DATE_TAG}"
  fi
fi
MATHVISTA_CALIB_DATE_TAG="${MATHVISTA_CALIB_DATE_TAG:-$(date +%m%d)}"

ECOFLAP_CC3M_JOB_ID="${ECOFLAP_CC3M_JOB_ID:-cc3m_calib128-blipt5_wanda_pruner_0.5-1.0-1.0_MEZO-GradOnly_sum0.6_block_bs8}"
LB_CC3M_JOB_ID="${LB_CC3M_JOB_ID:-cc3m_calib128-blipt5_tamp_pruner_0.5-1.0-1.0}"

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

job_id_mathvista_ckpt() { echo "okvqa_cf_0.5_mathvista_overall_${MATHVISTA_CALIB_DATE_TAG}_s${1}"; }
ckpt_lb_mathvista() { echo "$LB_ROOT/pruned_checkpoint/$(job_id_mathvista_ckpt "$1").pth"; }
ckpt_ecoflap_mathvista() { echo "$ECOFLAP_ROOT/pruned_checkpoint/$(job_id_mathvista_ckpt "$1").pth"; }
ckpt_ecoflap_cc3m() { echo "$ECOFLAP_ROOT/pruned_checkpoint/${ECOFLAP_CC3M_JOB_ID}.pth"; }
ckpt_lb_cc3m() { echo "$LB_ROOT/pruned_checkpoint/${LB_CC3M_JOB_ID}.pth"; }

okvqa_evaluate_txt() {
  local repo_root="$1"
  local job_id="$2"
  echo "${repo_root}/lavis/output/BLIP2/OKVQA/${job_id}/evaluate.txt"
}

# 读取 LAVIS VQA 任务写入的 evaluate.txt（JSON 行），取最后一行的 agg_metrics = overall accuracy
print_okvqa_agg_line() {
  local title="$1"
  local repo_root="$2"
  local job_id="$3"
  local ckpt_path="${4:-}"

  if [[ -n "$ckpt_path" && ! -f "$ckpt_path" ]]; then
    printf '%-42s  %s\n' "$title" "(未跑：缺权重)"
    return
  fi

  local ev
  ev="$(okvqa_evaluate_txt "$repo_root" "$job_id")"
  if [[ ! -f "$ev" ]]; then
    printf '%-42s  %s\n' "$title" "(无 evaluate.txt: $ev)"
    return
  fi

  local val
  val="$(python3 -c "
import json, sys
path = sys.argv[1]
try:
    lines = [ln.strip() for ln in open(path, encoding='utf-8') if ln.strip()]
    if not lines:
        print('(evaluate.txt 为空)')
    else:
        d = json.loads(lines[-1])
        a = d.get('agg_metrics')
        print(f'{a:.2f}%' if isinstance(a, (int, float)) else str(a))
except Exception as e:
    print(f'(解析失败: {e})')
" "$ev")"
  printf '%-42s  overall %s\n' "$title" "$val"
}

run_okvqa_overall_one() {
  local repo_root="$1"
  local ckpt="$2"
  local eval_job="$3"
  local master_port="$4"
  local sampler_seed="$5"

  if [[ ! -f "$ckpt" ]]; then
    echo "[WARN] 跳过 — 缺权重: $ckpt"
    return 0
  fi
  export LAVIS_DISTRIBUTED_SAMPLER_SEED="$sampler_seed"
  echo "========== OKVQA overall | $eval_job | port=$master_port | ckpt=$ckpt =========="
  (
    cd "$repo_root"
    python -m torch.distributed.run --nproc_per_node=1 --master_port="$master_port" evaluate_blip.py \
      --cfg-path lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml \
      --pruning_method blipt5_wanda_pruner \
      --t5_pruned_checkpoint "$ckpt" --vit_pruned_checkpoint "$ckpt" \
      --job_id "$eval_job"
  )
  echo "[OK] $eval_job"
}

STAMP="$(date +%Y%m%d_%H%M%S)"
EJOB_ECO_CC3M="okvqa_eval_ecoflapCC3M_${STAMP}_fullval"
EJOB_LB_CC3M="okvqa_eval_lbCC3M_${STAMP}_fullval"

echo "========== 仅 OKVQA overall ×6 | MATHVISTA_CALIB_DATE_TAG=${MATHVISTA_CALIB_DATE_TAG} | STAMP=${STAMP} =========="

for SEED in 30 42; do
  CK="$(ckpt_lb_mathvista "$SEED")"
  MID="${MATHVISTA_CALIB_DATE_TAG}_s${SEED}"
  EJOB="okvqa_eval_calibMathVistaLb_samplerSeed${SEED}_ckpt${MID}_fullval"
  run_okvqa_overall_one "$LB_ROOT" "$CK" "$EJOB" $((30911 + SEED)) "$SEED"
done

for SEED in 30 42; do
  CK="$(ckpt_ecoflap_mathvista "$SEED")"
  MID="${MATHVISTA_CALIB_DATE_TAG}_s${SEED}"
  EJOB="okvqa_eval_calibMathVistaEcoflap_samplerSeed${SEED}_ckpt${MID}_fullval"
  run_okvqa_overall_one "$ECOFLAP_ROOT" "$CK" "$EJOB" $((29911 + SEED)) "$SEED"
done

export LAVIS_DISTRIBUTED_SAMPLER_SEED="${LAVIS_DISTRIBUTED_SAMPLER_SEED:-30}"
CCE="$(ckpt_ecoflap_cc3m)"
run_okvqa_overall_one "$ECOFLAP_ROOT" "$CCE" "$EJOB_ECO_CC3M" $((29911 + LAVIS_DISTRIBUTED_SAMPLER_SEED)) "$LAVIS_DISTRIBUTED_SAMPLER_SEED"

CCL="$(ckpt_lb_cc3m)"
run_okvqa_overall_one "$LB_ROOT" "$CCL" "$EJOB_LB_CC3M" $((31911 + LAVIS_DISTRIBUTED_SAMPLER_SEED)) "$LAVIS_DISTRIBUTED_SAMPLER_SEED"

echo ""
echo "========== OKVQA overall 汇总（agg_metrics = official overall accuracy）=========="
print_okvqa_agg_line "1) LB  MathVista s30" "$LB_ROOT" \
  "okvqa_eval_calibMathVistaLb_samplerSeed30_ckpt${MATHVISTA_CALIB_DATE_TAG}_s30_fullval" \
  "$(ckpt_lb_mathvista 30)"
print_okvqa_agg_line "2) LB  MathVista s42" "$LB_ROOT" \
  "okvqa_eval_calibMathVistaLb_samplerSeed42_ckpt${MATHVISTA_CALIB_DATE_TAG}_s42_fullval" \
  "$(ckpt_lb_mathvista 42)"
print_okvqa_agg_line "3) ECo MathVista s30" "$ECOFLAP_ROOT" \
  "okvqa_eval_calibMathVistaEcoflap_samplerSeed30_ckpt${MATHVISTA_CALIB_DATE_TAG}_s30_fullval" \
  "$(ckpt_ecoflap_mathvista 30)"
print_okvqa_agg_line "4) ECo MathVista s42" "$ECOFLAP_ROOT" \
  "okvqa_eval_calibMathVistaEcoflap_samplerSeed42_ckpt${MATHVISTA_CALIB_DATE_TAG}_s42_fullval" \
  "$(ckpt_ecoflap_mathvista 42)"
print_okvqa_agg_line "5) ECo CC3M" "$ECOFLAP_ROOT" "$EJOB_ECO_CC3M" "$CCE"
print_okvqa_agg_line "6) LB  CC3M" "$LB_ROOT" "$EJOB_LB_CC3M" "$CCL"
echo "================================================================================"
echo "========== 6× OKVQA overall 完成 | STAMP=${STAMP} =========="
