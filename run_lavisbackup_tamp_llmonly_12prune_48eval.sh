#!/usr/bin/env bash
# =============================================================================
# LAVIS_backup TAMP pruner — 多模态 calibration × 4 源 × 3 seed = 12 剪枝 → 48 eval
#
# Calibration 数据:
#   MMBench    → /data/data2/mfs/MMBench_calibration
#   MMMU       → /data/data2/mfs/MMMU_calibration
#   OKVQA      → 默认 okvqa_train.json；图像根须含 train2014/（COCO train2014 与标注相对路径一致）。
#               若你只有 val2014：请下载/软链 MSCOCO train2014 到 $OKVQA_CALIB_IMAGES/train2014/，
#               或 export OKVQA_CALIB_JSON=... 改为只用 val 图的校准子集 json。
#   MathVista  → 默认识别 images.zip 解压后的 .../MathVista_calibration/images/images/*.jpg
#
# Seeds: 0, 30, 42
#
# 剪枝规格: TAMP (amiA + density_sum + layer), 只剪 T5 (不剪 ViT)
# 评测: MMBench / MMMU / OKVQA / MathVista MC
#
# 命名: {calib}_tamp_llmonly_{stamp}_seed{seed}.pth
#
# 环境变量覆盖:
#   RUN_PRUNE=1 RUN_EVAL=1   默认均执行；仅评测设 RUN_PRUNE=0 JOB_STAMP=xxx
#   JOB_STAMP                 时间戳 (默认自动生成)
#   PRUNING_CALIB_BATCH       校准 batch size (默认 8)
#   NUM_DATA                  校准样本数 (默认 128)
#   CUDA_VISIBLE_DEVICES      GPU (默认 0)
#   OKVQA_CALIB_JSON          覆盖 OKVQA 校准标注路径
#   OKVQA_CALIB_IMAGES        显式指定 OKVQA 图像根（跳过自动探测）
#   OKVQA_CALIB_IMAGE_CANDIDATES  冒号分隔的额外探测目录（在默认列表之前优先尝试）
#   MATHVISTA_CALIB_IMAGES        MathVista 剪枝校准图根目录（须含 json 里 image 文件名，且为有效 JPEG，勿用 0 字节占位）
#
# 断点续跑剪枝（本次已从 OKVQA 校准继续）：
#   PRUNE_SKIP_CALIBS   空格分隔，跳过这些 calibration 的剪枝（mmbench/mmmu 已跑完）。
#                       仍会评测若 pruned_checkpoint 下对应 pth 存在。
#   JOB_STAMP           必须与已跑完 mmbench/mmmu 那次相同，否则找不到旧 pth。
#   全量重跑四套剪枝：   PRUNE_SKIP_CALIBS="" bash ...
# =============================================================================

set -euo pipefail

BASE="${BASE:-/data/data2/mfs}"
LB_ROOT="$BASE/2/LAVIS_backup"
YAML_TPL="$LB_ROOT/lavis/projects/blip2/eval/cc_prefix_derivative_compute_calib_envsubst.template.yaml"

BLIP2_PRETRAINED="$BASE/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth"

# --- Benchmark eval paths ---
MMBENCH_ROOT="${MMBENCH_ROOT:-$BASE/MMBench_eval}"
MMMU_ROOT="${MMMU_ROOT:-$BASE/MMMU_single_image}"
MATHVISTA_JSON="${MATHVISTA_JSON:-$BASE/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json}"
MATHVISTA_IMG="${MATHVISTA_IMG:-$BASE/MathVista_eval_testmini_mc/images}"

# --- Calibration data definitions ---
declare -A CALIB_JSON CALIB_IMG
CALIB_JSON[mmbench]="$BASE/MMBench_calibration/mmbench_calibration_train.json"
CALIB_IMG[mmbench]="$BASE/MMBench_calibration/images"
CALIB_JSON[mmmu]="$BASE/MMMU_calibration/mmmu_calibration_train.json"
CALIB_IMG[mmmu]="$BASE/MMMU_calibration/images"
CALIB_JSON[okvqa]="${OKVQA_CALIB_JSON:-$BASE/datasets/okvqa/annotations/okvqa_train.json}"
# OKVQA train 图：默认可被自动探测覆盖（见 _resolve_okvqa_calib_image_root）
CALIB_IMG[okvqa]="${OKVQA_CALIB_IMAGES:-$BASE/datasets/okvqa}"
CALIB_JSON[mathvista]="${MATHVISTA_CALIB_JSON:-$BASE/MathVista_calibration/mathvista_calibration_train.json}"
# HF 的 images.zip 解压后多为 .../images/images/*.jpg，与 json 里「相对 images 根」的 1.jpg 对齐
if [[ -n "${MATHVISTA_CALIB_IMAGES:-}" ]]; then
  CALIB_IMG[mathvista]="$MATHVISTA_CALIB_IMAGES"
elif [[ -f "$BASE/MathVista_calibration/images/images/1.jpg" ]]; then
  CALIB_IMG[mathvista]="$BASE/MathVista_calibration/images/images"
else
  CALIB_IMG[mathvista]="$BASE/MathVista_calibration/images"
fi

SEEDS=(0 30 42)
CALIBS=(mmbench mmmu okvqa mathvista)

# 2026-05-05：mmbench / mmmu 剪枝已跑完，从 okvqa 继续。全量重跑前设为空： PRUNE_SKIP_CALIBS=""
PRUNE_SKIP_CALIBS="${PRUNE_SKIP_CALIBS:-mmbench mmmu}"

# --- Config ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1
export HF_HOME="${HF_HOME:-$BASE/model_cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export LAVIS_METRICS_JSONL="${LAVIS_METRICS_JSONL:-}"

export BERT_BASE_UNCASED_SNAPSHOT="${BERT_BASE_UNCASED_SNAPSHOT:-$HF_HOME/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594}"
export FLAN_T5_XL_SNAPSHOT="${FLAN_T5_XL_SNAPSHOT:-$HF_HOME/hub/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001}"

RUN_PRUNE="${RUN_PRUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
JOB_STAMP="${JOB_STAMP:-$(date +%Y%m%d_%H%M%S)}"
NUM_DATA="${NUM_DATA:-128}"
NUM_DATA_FIRST_STAGE="${NUM_DATA_FIRST_STAGE:-128}"
PRUNING_CALIB_BATCH="${PRUNING_CALIB_BATCH:-8}"
T5_SPEC="${T5_PRUNE_SPEC:-24-0.5-1.0-1.0}"
VIT_SPEC="${VIT_PRUNE_SPEC:-39-0.5-1.0-1.0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
MASTER_PORT_START="${MASTER_PORT_START:-29831}"

echo "========== TAMP LLM-only | 12 prune + 48 eval | STAMP=$JOB_STAMP =========="
echo "Calibs: ${CALIBS[*]}"
echo "Seeds:  ${SEEDS[*]}"
echo "Num calib samples: $NUM_DATA"
if [[ -n "${PRUNE_SKIP_CALIBS:-}" ]]; then
  echo "PRUNE_SKIP_CALIBS (跳过剪枝、仅挂载已有 pth): $PRUNE_SKIP_CALIBS"
  echo "（请确认 JOB_STAMP 与已产出 mmbench/mmmu 权重时一致）"
fi
echo "================================================================"

# 在未显式 export OKVQA_CALIB_IMAGES 时，在常见目录下查找 train2014/COCO_train2014_*.jpg
_resolve_okvqa_calib_image_root() {
  if [[ -n "${OKVQA_CALIB_IMAGES:-}" ]]; then
    echo "[INFO] OKVQA 图像根（来自 OKVQA_CALIB_IMAGES）: ${OKVQA_CALIB_IMAGES}"
    CALIB_IMG[okvqa]="${OKVQA_CALIB_IMAGES}"
    return 0
  fi
  local j="${CALIB_JSON[okvqa]}"
  local found
  export BASE
  found="$(
    python3 - "$j" <<'PY'
import json, os, sys

j_path = sys.argv[1]
base = os.environ.get("BASE", "/data/data2/mfs").rstrip("/")

candidates = []
extra = os.environ.get("OKVQA_CALIB_IMAGE_CANDIDATES", "")
if extra:
    for part in extra.split(":"):
        p = part.strip()
        if p and p not in candidates:
            candidates.append(p)

defaults = [
    os.path.join(base, "datasets", "okvqa"),
    os.path.join(base, "datasets", "coco"),
    os.path.join(base, "datasets", "coco2014"),
    os.path.join(base, "datasets", "mscoco"),
    os.path.join(base, "coco"),
    os.path.join(base, "COCO"),
    os.path.join(base, "coco2014"),
    "/data/data2/mfs/datasets/coco",
    "/data/data2/mfs/datasets/coco2014",
    "/data/data2/mfs/coco",
    "/data/coco",
    "/data/coco2014",
    "/data/datasets/coco",
    "/data/datasets/coco2014",
    "/mnt/data/coco",
    "/mnt/data2/coco",
]
h = os.environ.get("HOME", "")
if h:
    defaults.append(os.path.join(h, "data", "coco"))
    defaults.append(os.path.join(h, "datasets", "coco"))
ad = os.environ.get("AUTODL_TMP", "")
if ad:
    defaults.append(os.path.join(ad, "coco"))
    defaults.append(os.path.join(ad, "datasets", "coco"))
for d in defaults:
    if d and d not in candidates:
        candidates.append(d)

with open(j_path, encoding="utf-8") as f:
    data = json.load(f)
rels = [a.get("image", "").strip() for a in data[:30] if a.get("image")]
if not rels:
    sys.exit(1)
probe = rels[0]
for root in candidates:
    if not root or not os.path.isdir(root):
        continue
    p = os.path.join(root, probe)
    if os.path.isfile(p):
        print(root)
        sys.exit(0)
sys.exit(1)
PY
  )" || true

  if [[ -n "$found" ]]; then
    CALIB_IMG[okvqa]="$found"
    echo "[INFO] OKVQA 图像根（自动探测）: $found"
    return 0
  fi
  echo "[WARN] 未在默认路径找到 OKVQA train 图（${CALIB_JSON[okvqa]} 首条相对路径）。使用默认: ${CALIB_IMG[okvqa]}"
  echo "      请 export OKVQA_CALIB_IMAGES=/含 train2014 的父目录 或设置 OKVQA_CALIB_IMAGE_CANDIDATES=路径1:路径2"
}

_resolve_okvqa_calib_image_root

# OKVQA train 标注引用 train2014/*.jpg；缺图时在此失败并提示，避免 torch 里才崩。
_check_okvqa_calib_images() {
  local j="${CALIB_JSON[okvqa]}" r="${CALIB_IMG[okvqa]}"
  [[ -f "$j" ]] || {
    echo "[FATAL] OKVQA 校准 JSON 不存在: $j" >&2
    exit 1
  }
  python3 - "$j" "$r" <<'PY' || exit 1
import json, os, sys
j_path, root = sys.argv[1], sys.argv[2]
with open(j_path) as f:
    data = json.load(f)
for i, ann in enumerate(data[:50]):
    rel = ann.get("image", "")
    if not rel:
        continue
    p = os.path.join(root, rel)
    if os.path.isfile(p):
        print("[INFO] OKVQA calib image probe OK (sample %d): %s" % (i, p))
        sys.exit(0)
print("[FATAL] OKVQA 校准：在图像根目录下找不到标注里的图片。", file=sys.stderr)
print("  JSON: %s" % j_path, file=sys.stderr)
print("  图像根: %s" % root, file=sys.stderr)
print("  已检查前 50 条里的相对路径，例如应存在:", file=sys.stderr)
for ann in data[:3]:
    print("    %s" % os.path.join(root, ann.get("image", "")), file=sys.stderr)
print("", file=sys.stderr)
print("  okvqa_train.json 使用 COCO train2014，请保证:", file=sys.stderr)
print("    %s" % os.path.join(root, "train2014", "<COCO_train2014_*.jpg>"), file=sys.stderr)
print("  常见做法：下载 COCO2014 train images，解压或 ln -s 为:", file=sys.stderr)
print("    %s -> <你的 train2014 目录>" % os.path.join(root, "train2014"), file=sys.stderr)
sys.exit(1)
PY
}

# MathVista 校准 JSON 里的 1.jpg 等须为真实图片；仓库里常见 0 字节占位会导致 PIL.UnidentifiedImageError
_check_mathvista_calib_images() {
  local j="${CALIB_JSON[mathvista]}" r="${CALIB_IMG[mathvista]}"
  [[ -f "$j" ]] || {
    echo "[FATAL] MathVista 校准 JSON 不存在: $j" >&2
    exit 1
  }
  python3 - "$j" "$r" <<'PY' || exit 1
import json, os, sys
from PIL import Image

j_path, root = sys.argv[1], sys.argv[2]
with open(j_path, encoding="utf-8") as f:
    data = json.load(f)
for ann in data[:50]:
    rel = (ann.get("image") or "").strip()
    if not rel:
        continue
    p = os.path.join(root, rel)
    if not os.path.isfile(p):
        continue
    sz = os.path.getsize(p)
    if sz < 64:
        print("[FATAL] MathVista 校准图过小或为空占位文件（%d bytes）: %s" % (sz, p), file=sys.stderr)
        print("  请从 MathVista 官方数据拷贝对应图片到该目录，或 export MATHVISTA_CALIB_IMAGES=含这些文件名的目录", file=sys.stderr)
        sys.exit(1)
    try:
        with Image.open(p) as im:
            im.verify()
    except Exception as e:
        print("[FATAL] MathVista 校准图 PIL 无法识别: %s (%s)" % (p, e), file=sys.stderr)
        sys.exit(1)
    print("[INFO] MathVista calib image OK: %s (%d bytes)" % (p, sz))
    sys.exit(0)
print("[FATAL] MathVista: 在前 50 条标注中未找到可用图片。root=%s" % root, file=sys.stderr)
sys.exit(1)
PY
}

# ---- Helper: convert calibration JSON to {image, caption} format ----
_calib_to_caption_json() {
  local src="$1" dst="$2"
  python3 -c "
import json, sys
with open('$src') as f:
    data = json.load(f)
out = []
for item in data:
    img = item.get('image', '')
    cap = item.get('caption') or item.get('question', '')
    ans = item.get('answer', '')
    if ans:
        if isinstance(ans, list):
            ans = ans[0] if ans else ''
        cap = str(cap) + ' ' + str(ans)
    out.append({'image': img, 'caption': cap.strip()})
with open('$dst', 'w') as f:
    json.dump(out, f)
print(f'Converted {len(out)} samples: $src -> $dst')
" 2>&1
}

# ---- Step 1: Pruning ----
declare -A CKPT
TMP_DIR="$LB_ROOT/tmp_calib"
mkdir -p "$TMP_DIR" "$LB_ROOT/pruned_checkpoint"

_port_counter=0
for calib in "${CALIBS[@]}"; do
  SRC_JSON="${CALIB_JSON[$calib]}"
  SRC_IMG="${CALIB_IMG[$calib]}"

  # 已跑过的 calibration：不再跑剪枝，只把 CKPT 指到已有 pth（需 JOB_STAMP 一致）
  if [[ -n "${PRUNE_SKIP_CALIBS:-}" ]] && [[ " ${PRUNE_SKIP_CALIBS} " == *" ${calib} "* ]]; then
    echo "---------- [SKIP PRUNE] calib=$calib（见 PRUNE_SKIP_CALIBS）----------"
    for seed in "${SEEDS[@]}"; do
      CKPT_NAME="${calib}_tamp_llmonly_${JOB_STAMP}_seed${seed}"
      _f="$LB_ROOT/pruned_checkpoint/${CKPT_NAME}.pth"
      CKPT["${calib}_${seed}"]="$_f"
      if [[ -f "$_f" ]]; then
        echo "  [OK] 复用: $_f"
      else
        echo "  [WARN] 未找到 $_f — 请 export JOB_STAMP=生成该权重时的同一时间戳，或移除此 calib 出 PRUNE_SKIP_CALIBS 后重剪"
      fi
    done
    continue
  fi

  if [[ "$calib" == "okvqa" ]] && [[ "${RUN_PRUNE:-1}" == "1" ]]; then
    _check_okvqa_calib_images
  fi
  if [[ "$calib" == "mathvista" ]] && [[ "${RUN_PRUNE:-1}" == "1" ]]; then
    _check_mathvista_calib_images
  fi

  CAP_JSON="$TMP_DIR/calib_${calib}_captions.json"
  _calib_to_caption_json "$SRC_JSON" "$CAP_JSON"

  for seed in "${SEEDS[@]}"; do
    _port=$((MASTER_PORT_START + _port_counter))
    _port_counter=$((_port_counter + 1))
    export LAVIS_DISTRIBUTED_SAMPLER_SEED="$seed"

    CKPT_NAME="${calib}_tamp_llmonly_${JOB_STAMP}_seed${seed}"
    CKPT["${calib}_${seed}"]="$LB_ROOT/pruned_checkpoint/${CKPT_NAME}.pth"

    # Generate YAML via envsubst
    CFG_TMP="$TMP_DIR/calib_${calib}_seed${seed}.yaml"

    cat > "$CFG_TMP" <<ENVEOF
model:
  arch: blip2_t5
  model_type: pretrain_flant5xl
  use_grad_checkpoint: False
  pretrained: "$BLIP2_PRETRAINED"

datasets:
  prefix_conceptual_caption_3m:
    vis_processor:
      train:
        name: "blip2_image_train"
        image_size: 224
    text_processor:
      train:
        name: "blip_caption"
    build_info:
      annotations:
        train:
          url:
            - "$CAP_JSON"
          storage:
            - "$CAP_JSON"
      images:
        storage: "$SRC_IMG"

run:
  task: image_text_pretrain
  lr_sched: "linear_warmup_cosine_lr"
  init_lr: 1e-4
  min_lr: 1e-5
  warmup_lr: 1e-6
  weight_decay: 0.05
  max_epoch: 1
  batch_size_train: 16
  batch_size_eval: 16
  num_workers: 4
  warmup_steps: 1000
  seed: $seed
  output_dir: "output/BLIP2/Pretrain_stage2"
  amp: True
  resume_ckpt_path: null
  evaluate: False
  test_splits: ["train"]
  device: "cuda"
  world_size: 1
  dist_url: "env://"
  distributed: True
ENVEOF

    if [[ "$RUN_PRUNE" != "1" ]]; then
      echo "[SKIP PRUNE] $CKPT_NAME"
      continue
    fi

    echo "---------- [PRUNE] $CKPT_NAME | seed=$seed | port=$_port ----------"
    (
      cd "$LB_ROOT"
      python -m torch.distributed.run --nproc_per_node=1 --master_port="$_port" evaluate_blip.py \
        --cfg-path "$CFG_TMP" \
        --pruning_method blipt5_tamp_pruner \
        --save_pruned_model \
        --num_data "$NUM_DATA" \
        --num_data_first_stage "$NUM_DATA_FIRST_STAGE" \
        --prunining_dataset_batch_size "$PRUNING_CALIB_BATCH" \
        --t5_prune_spec "$T5_SPEC" \
        --vit_prune_spec "$VIT_SPEC" \
        --job_id "$CKPT_NAME"
    )
    echo "[DONE PRUNE] $CKPT_NAME → ${CKPT[${calib}_${seed}]}"
  done
done

# ---- Step 2: Evaluation ----
if [[ "$RUN_EVAL" != "1" ]]; then
  echo "[SKIP] All evals (RUN_EVAL=0)"
  echo "========== DONE =========="
  exit 0
fi

SUMMARY_DIR="$LB_ROOT/lavis/output/BLIP2"
mkdir -p "$SUMMARY_DIR"
METRICS_JSONL="$SUMMARY_DIR/tamp_llmonly_12prune_48eval_${JOB_STAMP}.jsonl"
: > "$METRICS_JSONL"
export LAVIS_METRICS_JSONL="$METRICS_JSONL"

for calib in "${CALIBS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    ckpt="${CKPT[${calib}_${seed}]}"
    CKPT_NAME="${calib}_tamp_llmonly_${JOB_STAMP}_seed${seed}"
    TAG="${CKPT_NAME}"

    if [[ ! -f "$ckpt" ]]; then
      echo "[WARN] CKPT not found, skip eval: $ckpt"
      continue
    fi

    _port=$((MASTER_PORT_START + _port_counter))
    _port_counter=$((_port_counter + 1))

    echo "===== [EVAL] $TAG ====="

    # -- MMBench --
    export LAVIS_EVAL_CALIB_TAG="$TAG"
    export LAVIS_METRICS_BENCHMARK="MMBench"
    echo "  [MMBench]"
    (
      cd "$LB_ROOT"
      python scripts/blip2/mmmu_eval_by_discipline.py \
        --mmmu_root "$MMBENCH_ROOT" --split dev \
        --ckpt "$ckpt" --batch_size "$EVAL_BATCH_SIZE" \
        --device cuda --overall_only
    ) || echo "  [WARN] MMBench eval failed for $TAG"

    # -- MMMU --
    export LAVIS_METRICS_BENCHMARK="MMMU"
    echo "  [MMMU]"
    (
      cd "$LB_ROOT"
      python scripts/blip2/mmmu_eval_by_discipline.py \
        --mmmu_root "$MMMU_ROOT" --split test \
        --ckpt "$ckpt" --batch_size "$EVAL_BATCH_SIZE" \
        --device cuda --overall_only
    ) || echo "  [WARN] MMMU eval failed for $TAG"

    # -- OKVQA --
    export LAVIS_METRICS_BENCHMARK="OKVQA"
    echo "  [OKVQA]"
    (
      cd "$LB_ROOT"
      python -m torch.distributed.run --nproc_per_node=1 --master_port="$_port" evaluate_blip.py \
        --cfg-path lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml \
        --pruning_method blipt5_wanda_pruner \
        --t5_pruned_checkpoint "$ckpt" --vit_pruned_checkpoint "$ckpt" \
        --job_id "okvqa_eval_${CKPT_NAME}"
    ) || echo "  [WARN] OKVQA eval failed for $TAG"

    # -- MathVista MC --
    export LAVIS_METRICS_BENCHMARK="MathVista_MC"
    echo "  [MathVista MC]"
    (
      cd "$LB_ROOT"
      python scripts/blip2/mathvista_mc_eval.py \
        --eval_json "$MATHVISTA_JSON" \
        --images_dir "$MATHVISTA_IMG" \
        --ckpt "$ckpt" --batch_size "$EVAL_BATCH_SIZE" \
        --device cuda
    ) || echo "  [WARN] MathVista eval failed for $TAG"

    echo "[DONE EVAL] $TAG"
    echo ""
  done
done

echo "========== ALL DONE =========="
echo "Metrics jsonl: $METRICS_JSONL"
echo "Checkpoints in: $LB_ROOT/pruned_checkpoint/"
echo "Calibration configs/tmp: $TMP_DIR/"
