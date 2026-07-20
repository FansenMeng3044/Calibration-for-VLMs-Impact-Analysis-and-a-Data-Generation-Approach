#!/usr/bin/env bash
# =============================================================================
# 六源「纯文本」标定 → TAMP 只剪 T5（llm-only, 无图像/无视觉 token）→ 各自四基准评测
#
# 标定源（各取纯文本，无跨模态信号）：
#   c4        —— 已是纯文本 list（直接用 $C4_JSON）
#   cc3m      —— 取 caption
#   okvqa     —— 取 question
#   mathvista —— 取 question(+choices)
#   mmbench   —— 取 question(+choices)
#   mmmu      —— 取 question(+choices)
#
# 流程（每个源独立一遍）：
#   build_text_calib.py 抽纯文本 128 条 → evaluate_blip.py（t5_c4_text + blipt5_tamp_pruner，
#   只剪 T5，保存整模 pth）→ four_bench_eval：MMBench / OKVQA / MMMU / MathVista
#
# 注意（务必知情）：
#   blipt5_tamp_pruner 在纯文本标定下运行 TAMP 的「单模态归约」：
#   s = s_l（视觉/跨模态多样性项无定义，按存在的项求平均），AMIA 在文本 token 上选择。
#   这不是已发表的多模态 TAMP，报告时应写成 TAMP variant / single-modality reduction。
#   若要 naive+uniform 基线，用 PRUNE_METHOD=blipt5_wanda_pruner（其默认值即是）。
#
# 用法：
#   cd /data/data2/mfs/2/LAVIS_backup
#   bash scripts/blip2/run_lavisbackup_tamp_textcalib_six_prune_eval.sh
# 只跑其中几个源：
#   SOURCES="okvqa mathvista" bash scripts/blip2/run_lavisbackup_tamp_textcalib_six_prune_eval.sh
# 仅评测（已剪好）：
#   RUN_PRUNE=0 JOB_STAMP=你的STAMP bash scripts/.../run_lavisbackup_tamp_textcalib_six_prune_eval.sh
#
# 主要环境变量：
#   BASE, SOURCES, RUN_BUILD/RUN_PRUNE/RUN_EVAL(默认1), FORCE_BUILD(默认0), JOB_STAMP
#   NUM_DATA(128) BS(8) T5_SPEC(24-0.5-1.0-1.0) MAX_SPARSITY_PER_LAYER(0.6)
#   RAW_<SRC> / TEXT_<SRC>  覆盖各源的「源标注」与「已抽好的纯文本 JSON」路径
#   MMBENCH_ROOT MMMU_ROOT MATHVISTA_EVAL_JSON MATHVISTA_IMAGES_DIR EVAL_BATCH_SIZE
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
[[ -d "${BERT_BASE_UNCASED_SNAPSHOT}" ]] || { echo "[FATAL] 未找到 bert-base-uncased snapshot（设 HF_HOME 或 BERT_BASE_UNCASED_SNAPSHOT）" >&2; exit 1; }
[[ -d "${FLAN_T5_XL_SNAPSHOT}" ]]      || { echo "[FATAL] 未找到 flan-t5-xl snapshot（设 HF_HOME 或 FLAN_T5_XL_SNAPSHOT）" >&2; exit 1; }

BLIP2_PRETRAINED="${BLIP2_PRETRAINED:-$BASE/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth}"
[[ -f "${BLIP2_PRETRAINED}" ]] || { echo "[FATAL] 未找到 BLIP2_PRETRAINED: ${BLIP2_PRETRAINED}" >&2; exit 1; }

CFG="${CFG:-$REPO_ROOT/lavis/projects/blip2/eval/t5_c4_text_prune_calib.yaml}"
BUILDER="$SCRIPT_DIR/build_text_calib.py"

# ---- 剪枝超参（与你现有 TAMP t5_c4 脚本一致）----
NUM_DATA="${NUM_DATA:-128}"
BS="${BS:-8}"
T5_SPEC="${T5_SPEC:-24-0.5-1.0-1.0}"
PRUNE_METHOD="${PRUNE_METHOD:-blipt5_tamp_pruner}"
SCORE_METHOD="${SCORE_METHOD:-MEZO-GradOnly_sum}"          # 会被 tamp 别名覆盖为 density_sum
SPARSITY_GRANULARITY="${SPARSITY_GRANULARITY:-block}"       # 会被 tamp 别名覆盖为 layer
MAX_SPARSITY_PER_LAYER="${MAX_SPARSITY_PER_LAYER:-0.6}"
SEED="${SEED:-42}"

# ---- 评测资源 ----
MMBENCH_ROOT="${MMBENCH_ROOT:-$BASE/MMBench_eval}"
MMMU_ROOT="${MMMU_ROOT:-$BASE/MMMU_single_image}"
MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
MMMU_SPLIT="${MMMU_SPLIT:-test}"
MATHVISTA_EVAL_JSON="${MATHVISTA_EVAL_JSON:-$BASE/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json}"
MATHVISTA_IMAGES_DIR="${MATHVISTA_IMAGES_DIR:-$BASE/MathVista_eval_testmini_mc/images}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
MASTER_PORT_START="${MASTER_PORT_START:-29780}"

RUN_BUILD="${RUN_BUILD:-1}"
RUN_PRUNE="${RUN_PRUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
FORCE_BUILD="${FORCE_BUILD:-0}"
JOB_STAMP="${JOB_STAMP:-$(date +%Y%m%d_%H%M%S)}"

SOURCES="${SOURCES:-c4 cc3m okvqa mathvista mmbench mmmu}"
TEXT_DIR="${TEXT_DIR:-$BASE/text_calib_128}"
SUMMARY_DIR="$REPO_ROOT/lavis/output/BLIP2"
mkdir -p "$TEXT_DIR" "$SUMMARY_DIR"

# ---- 每个源：源标注(RAW) / 已抽好的纯文本(TEXT) / 抽取选项(OPT) ----
raw_for()  { local s="$1"; local v="RAW_${s^^}";  echo "${!v:-$(_default_raw  "$s")}"; }
text_for() { local s="$1"; local v="TEXT_${s^^}"; echo "${!v:-$(_default_text "$s")}"; }
_default_text() { echo "$TEXT_DIR/${1}_text_calib_${NUM_DATA}.json"; }
_default_raw() {
  case "$1" in
    c4)        echo "$BASE/c4_calib_128.json" ;;
    cc3m)      echo "$BASE/CC3M_calib_128/cc3m_calib_128.json" ;;
    okvqa)     echo "$BASE/datasets/okvqa/annotations/okvqa_train.json" ;;
    mathvista) echo "$BASE/MathVista_calibration/mathvista_calibration_train.json" ;;
    mmbench)   echo "$BASE/MMBench_calibration/mmbench_calibration_train.json" ;;
    mmmu)      echo "$BASE/MMMU_calibration/mmmu_calibration_train.json" ;;
    *) echo "" ;;
  esac
}
_build_opts() {   # MC 数据集把选项拼进标定文本，使其接近真实 prompt
  case "$1" in
    mathvista|mmbench|mmmu) echo "--include-choices" ;;
    *) echo "" ;;
  esac
}

# =============================== 抽取纯文本 ===============================
build_one() {
  local s="$1" raw text opts
  raw="$(raw_for "$s")"; text="$(text_for "$s")"; opts="$(_build_opts "$s")"
  if [[ "$s" == "c4" ]]; then
    # C4 本身就是纯文本 list，若默认 TEXT 不存在则直接指向 RAW
    if [[ -f "$(text_for c4)" ]]; then echo "[build] c4 使用已存在: $(text_for c4)"; return 0; fi
    [[ -f "$raw" ]] || { echo "[FATAL] 缺少 C4 文本: $raw" >&2; exit 1; }
    ln -sf "$raw" "$text" 2>/dev/null || cp "$raw" "$text"
    echo "[build] c4 -> $text"; return 0
  fi
  if [[ -f "$text" && "$FORCE_BUILD" != "1" ]]; then
    echo "[build] $s 已存在，跳过（FORCE_BUILD=1 可强制重建）: $text"; return 0
  fi
  [[ -f "$raw" ]] || { echo "[FATAL] $s 源标注不存在: $raw（用 RAW_${s^^}=... 指定）" >&2; exit 1; }
  echo "[build] $s: $raw -> $text  $opts"
  python "$BUILDER" --input "$raw" --output "$text" \
    --num "$NUM_DATA" --seed "$SEED" --shuffle $opts
}

# =============================== 四基准评测 ===============================
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

# =============================== 剪枝 ===============================
prune_one() {
  local s="$1" text job ckpt
  text="$(text_for "$s")"; job="tamp_txt_${s}_${JOB_STAMP}"
  ckpt="$REPO_ROOT/pruned_checkpoint/${job}.pth"
  [[ -f "$text" ]] || { echo "[FATAL] $s 纯文本标定不存在: $text（先 RUN_BUILD=1）" >&2; exit 1; }
  echo ""; echo ">>> [剪枝] $s | t5_c4_text + TAMP | 只剪 T5 | calib=$text"
  python evaluate_blip.py \
    --cfg-path "${CFG}" \
    --options "model.pretrained=${BLIP2_PRETRAINED}" \
    --prune_calib_mode t5_c4_text \
    --c4_calib_json "${text}" \
    --pruning_method "${PRUNE_METHOD}" \
    --score_method "${SCORE_METHOD}" \
    --sparsity_ratio_granularity "${SPARSITY_GRANULARITY}" \
    --max_sparsity_per_layer "${MAX_SPARSITY_PER_LAYER}" \
    --t5_prune_spec "${T5_SPEC}" \
    --num_data "${NUM_DATA}" \
    --prunining_dataset_batch_size "${BS}" \
    --num_data_first_stage "${NUM_DATA}" \
    --job_id "${job}" \
    --save_pruned_model
  [[ -f "$ckpt" ]] || { echo "[FATAL] 剪枝后未找到: $ckpt" >&2; exit 1; }
  echo "[OK] 剪枝完成: $ckpt"
}

# =============================== 主流程 ===============================
echo "========== 六源纯文本 TAMP(llm-only) 剪枝+评测 | STAMP=$JOB_STAMP =========="
echo "[INFO] REPO_ROOT=$REPO_ROOT  SOURCES=$SOURCES"
echo "[INFO] RUN_BUILD=$RUN_BUILD RUN_PRUNE=$RUN_PRUNE RUN_EVAL=$RUN_EVAL FORCE_BUILD=$FORCE_BUILD"
echo "[INFO] TEXT_DIR=$TEXT_DIR"
echo "[INFO] TAMP 纯文本标定 = 单模态归约（s = s_l；AMIA 在文本 token 上选择；DAS 逐层分配）。"

i=0
for s in $SOURCES; do
  echo ""
  echo "############################## 源: $s ##############################"
  [[ "$RUN_BUILD" == "1" ]] && build_one "$s"

  job="tamp_txt_${s}_${JOB_STAMP}"
  ckpt="$REPO_ROOT/pruned_checkpoint/${job}.pth"
  export LAVIS_METRICS_JSONL="$SUMMARY_DIR/tamp_textcalib_${s}_${JOB_STAMP}.jsonl"
  : > "$LAVIS_METRICS_JSONL"

  if [[ "$RUN_PRUNE" == "1" ]]; then
    prune_one "$s"
  else
    echo "[INFO] RUN_PRUNE=0，用已有权重: $ckpt"
    [[ -f "$ckpt" ]] || { echo "[FATAL] 缺少 $ckpt（设对 JOB_STAMP 或 RUN_PRUNE=1）" >&2; exit 1; }
  fi

  if [[ "$RUN_EVAL" == "1" ]]; then
    port=$((MASTER_PORT_START + i))
    four_bench_eval "$ckpt" "tamp_txtcalib_${s}_${JOB_STAMP}" \
      "okvqa_tamp_txtcalib_${s}_${JOB_STAMP}_fullval" "$port"
  fi
  echo "[DONE] 源 $s → 权重: $ckpt | 指标: $LAVIS_METRICS_JSONL"
  i=$((i + 1))
done

echo ""
echo "========== 全部完成 =========="
echo "[INFO] 纯文本标定: $TEXT_DIR/<src>_text_calib_${NUM_DATA}.json"
echo "[INFO] 权重:       $REPO_ROOT/pruned_checkpoint/tamp_txt_<src>_${JOB_STAMP}.pth"
echo "[INFO] 指标:       $SUMMARY_DIR/tamp_textcalib_<src>_${JOB_STAMP}.jsonl"
