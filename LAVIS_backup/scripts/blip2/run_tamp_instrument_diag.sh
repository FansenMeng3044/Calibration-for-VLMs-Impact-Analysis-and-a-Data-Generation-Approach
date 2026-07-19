#!/usr/bin/env bash
# =============================================================================
# TAMP 探针诊断：在五个 calibration 集上跑 diagnose_tamp_instrument.py
#
# 目的：把 TAMP 当作观察 calibration 规律的仪器之前，先标定这个仪器。
#   D1  AMIA 逐层选择率 / 模态构成  -> 探针对 calibration 数据有没有灵敏度
#   D2  DAS 同数据集两半的 Spearman -> 噪声地板（小于它的跨集差异不可解释）
#   D3  有效文本长度分布            -> 长度是否与 calibration 集共变（混淆）
#
# 本脚本不剪枝、不写权重，只产出 CSV/JSON。
#
# 用法:
#   cd /data/data2/mfs/2/LAVIS_backup
#   bash scripts/blip2/run_tamp_instrument_diag.sh
#
# 只跑部分 calibration:
#   CALIBS="mmbench cc3m" bash scripts/blip2/run_tamp_instrument_diag.sh
#
# 快速冒烟（少样本、只探 3 个 block、只探两种 Linear）:
#   MAX_SAMPLES=16 PROBE_BLOCKS=0,11,23 \
#   PROBE_LINEARS=SelfAttention.v,DenseReluDense.wo \
#   bash scripts/blip2/run_tamp_instrument_diag.sh
#
# 环境变量:
#   BASE, OUT_DIR, MAX_SAMPLES(128), BS(8), SPARSITY(0.5)
#   PROBE_BLOCKS(all), PROBE_LINEARS(all), CUDA_VISIBLE_DEVICES
#   BLIP2_PRETRAINED, PYTHON_BIN
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
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

ECOFLAP_ENV="${ECOFLAP_ENV:-${MODEL_CACHE_ROOT}/ecoflap_model_env.sh}"
if [[ -f "${ECOFLAP_ENV}" ]]; then
  set +u; source "${ECOFLAP_ENV}"; set -u
fi

export TORCH_HOME="${TORCH_HOME:-${MODEL_CACHE_ROOT}/torch}"
export HF_HOME="${HF_HOME:-${MODEL_CACHE_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

BLIP2_PRETRAINED="${BLIP2_PRETRAINED:-${MODEL_CACHE_ROOT}/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CALIBS="${CALIBS:-mmbench mmmu okvqa mathvista cc3m}"
MAX_SAMPLES="${MAX_SAMPLES:-128}"
BS="${BS:-8}"
SPARSITY="${SPARSITY:-0.5}"
PROBE_BLOCKS="${PROBE_BLOCKS:-all}"
PROBE_LINEARS="${PROBE_LINEARS:-all}"
OUT_DIR="${OUT_DIR:-tamp_instrument_diag_$(date +%Y%m%d_%H%M%S)}"

calib_json_path() {
  case "$1" in
    mmbench)   echo "$BASE/MMBench_calibration/mmbench_calibration_train.json" ;;
    mmmu)      echo "$BASE/MMMU_calibration/mmmu_calibration_train.json" ;;
    okvqa)     echo "$BASE/datasets/okvqa/annotations/okvqa_train.json" ;;
    mathvista) echo "$BASE/MathVista_calibration/mathvista_calibration_train.json" ;;
    cc3m)      echo "$BASE/CC3M_calib_128/cc3m_calib_128.json" ;;
    *) echo "" ;;
  esac
}

calib_images_dir() {
  case "$1" in
    mmbench)   echo "$BASE/MMBench_calibration/images" ;;
    mmmu)      echo "$BASE/MMMU_calibration/images" ;;
    okvqa)     echo "$BASE/datasets/okvqa/images" ;;
    mathvista) echo "$BASE/MathVista_calibration/images" ;;
    cc3m)      echo "$BASE/CC3M_calib_128/images" ;;
    *) echo "" ;;
  esac
}

if [[ ! -f "$BLIP2_PRETRAINED" ]]; then
  echo "[FATAL] 找不到 BLIP2 权重: $BLIP2_PRETRAINED" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] CALIBS=$CALIBS MAX_SAMPLES=$MAX_SAMPLES BS=$BS SPARSITY=$SPARSITY"
echo "[INFO] PROBE_BLOCKS=$PROBE_BLOCKS PROBE_LINEARS=$PROBE_LINEARS"

FAILED=""
for calib in $CALIBS; do
  cj="$(calib_json_path "$calib")"
  cim="$(calib_images_dir "$calib")"
  if [[ -z "$cj" || ! -f "$cj" ]]; then
    echo "[WARN] 跳过 $calib：找不到标注 $cj" >&2
    FAILED+=" $calib(json)"
    continue
  fi
  if [[ ! -d "$cim" ]]; then
    echo "[WARN] 跳过 $calib：找不到图像目录 $cim" >&2
    FAILED+=" $calib(images)"
    continue
  fi

  echo ""
  echo ">>> [DIAG] calib=$calib"
  if ! "$PYTHON_BIN" scripts/blip2/diagnose_tamp_instrument.py \
      --label "$calib" \
      --calib_json "$cj" \
      --images_dir "$cim" \
      --ckpt "$BLIP2_PRETRAINED" \
      --max_samples "$MAX_SAMPLES" \
      --batch_size "$BS" \
      --sparsity "$SPARSITY" \
      --probe_blocks "$PROBE_BLOCKS" \
      --probe_linears "$PROBE_LINEARS" \
      --out_dir "$OUT_DIR"; then
    echo "[ERROR] $calib 诊断失败" >&2
    FAILED+=" $calib(run)"
  fi
done

echo ""
echo "==================== 汇总 ===================="
echo "[DONE] 输出目录: $REPO_ROOT/$OUT_DIR"
echo "  d1_amia_selection.csv    每次 AMIA 选择的 valid/selected/模态构成"
echo "  d2_das_noise_floor.csv   每个 calib 集的 DAS 两半 Spearman(= 噪声地板)"
echo "  d3_calib_lengths.csv     每条样本的有效文本长度"
echo "  summary_<calib>.json     每个 calib 集的汇总"
if [[ -n "$FAILED" ]]; then
  echo "[WARN] 失败/跳过:$FAILED" >&2
fi
echo ""
echo "怎么读："
echo "  D1  select_ratio 若普遍 >0.95 -> AMIA 形同未开；<0.05 -> 估计量退化。"
echo "      zero_visual_selected_frac 明显 >0 -> 存在整层不选视觉 token 的情况。"
echo "  D2  spearman 越接近 1 越稳。若同数据集两半的 spearman 就已经很低，"
echo "      则跨 calibration 集的层稀疏率差异不可解释。"
echo "  D3  比较各 calib 集的 valid_text_mean。若差异很大，长度是潜在混淆。"
