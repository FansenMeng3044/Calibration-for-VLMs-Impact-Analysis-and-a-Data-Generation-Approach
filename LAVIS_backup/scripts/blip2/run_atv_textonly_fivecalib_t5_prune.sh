#!/usr/bin/env bash
# =============================================================================
# Text-only ATV entry for five calibration sources.
#
# What this runs:
#   CC3M / MathVista / OKVQA / MMBench / MMMU annotations
#   -> extract 128 text-only calibration strings
#   -> evaluate_blip.py --prune_calib_mode t5_c4_text
#   -> --pruning_method blipt5_atv_pruner
#   -> prune T5 only, save one full-model checkpoint per source.
#   -> run four evals for every checkpoint:
#      MMBench / OKVQA / MMMU / MathVista.
#
# Important:
#   This deliberately passes no images into the calibration dataloader.
#   Therefore there are no BLIP2 query/visual tokens for ATV to select.
#   In this setting the ATV entry degenerates to language-token-only Wanda-style
#   T5 pruning with uniform sparsity. Use the multimodal ATV script when you
#   need true ATV query-token selection.
#
# Usage from LAVIS_backup:
#   cd /data/data2/mfs/2/LAVIS_backup
#   bash scripts/blip2/run_atv_textonly_fivecalib_t5_prune.sh
#
# Run a subset:
#   SOURCES="mmbench mmmu" bash scripts/blip2/run_atv_textonly_fivecalib_t5_prune.sh
#
# Common overrides:
#   BASE=/data/data2/mfs
#   NUM_DATA=128 BS=8 T5_SPEC=24-0.5-1.0-1.0 ATV_ALPHA=1.0
#   RAW_CC3M=... RAW_MATHVISTA=... RAW_OKVQA=... RAW_MMBENCH=... RAW_MMMU=...
#   TEXT_DIR=/data/data2/mfs/atv_textonly_calib_128
#   RUN_EVAL=0
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

BASE="${BASE:-/data/data2/mfs}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

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
if [[ -z "${BERT_BASE_UNCASED_SNAPSHOT:-}" ]]; then
  BERT_BASE_UNCASED_SNAPSHOT="$(_resolve_hub_snapshot_dir "$HUB_ROOT/models--bert-base-uncased")"
fi
if [[ -z "${FLAN_T5_XL_SNAPSHOT:-}" ]]; then
  FLAN_T5_XL_SNAPSHOT="$(_resolve_hub_snapshot_dir "$HUB_ROOT/models--google--flan-t5-xl")"
fi
export BERT_BASE_UNCASED_SNAPSHOT
export FLAN_T5_XL_SNAPSHOT

BLIP2_PRETRAINED="${BLIP2_PRETRAINED:-$BASE/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth}"
CFG="${CFG:-$REPO_ROOT/lavis/projects/blip2/eval/t5_c4_text_prune_calib.yaml}"

NUM_DATA="${NUM_DATA:-128}"
BS="${BS:-8}"
T5_SPEC="${T5_SPEC:-24-0.5-1.0-1.0}"
ATV_ALPHA="${ATV_ALPHA:-1.0}"
SEED="${SEED:-42}"
JOB_STAMP="${JOB_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_BUILD="${RUN_BUILD:-1}"
RUN_PRUNE="${RUN_PRUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
FORCE_BUILD="${FORCE_BUILD:-0}"

SOURCES="${SOURCES:-mmbench mathvista mmmu okvqa cc3m}"
if [[ "$SOURCES" == "all" ]]; then
  SOURCES="mmbench mathvista mmmu okvqa cc3m"
fi
TEXT_DIR="${TEXT_DIR:-$BASE/atv_textonly_calib_${NUM_DATA}}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/lavis/output/BLIP2/atv_textonly_logs_${JOB_STAMP}_seed${SEED}}"
SUMMARY_DIR="${SUMMARY_DIR:-$REPO_ROOT/lavis/output/BLIP2}"
mkdir -p "$TEXT_DIR" "$LOG_DIR" "$SUMMARY_DIR" "$REPO_ROOT/pruned_checkpoint"

MMBENCH_ROOT="${MMBENCH_ROOT:-$BASE/MMBench_eval}"
MMMU_ROOT="${MMMU_ROOT:-$BASE/MMMU_single_image}"
MMBENCH_SPLIT="${MMBENCH_SPLIT:-dev}"
MMMU_SPLIT="${MMMU_SPLIT:-test}"
MATHVISTA_EVAL_JSON="${MATHVISTA_EVAL_JSON:-$BASE/MathVista_eval_testmini_mc/mathvista_multi_choice_eval.json}"
MATHVISTA_IMAGES_DIR="${MATHVISTA_IMAGES_DIR:-$BASE/MathVista_eval_testmini_mc/images}"
OKVQA_EVAL_CFG="${OKVQA_EVAL_CFG:-$REPO_ROOT/lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval_overall.yaml}"
OKVQA_EVAL_NUM_WORKERS="${OKVQA_EVAL_NUM_WORKERS:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
MASTER_PORT="${MASTER_PORT:-29700}"

if (( NUM_DATA % BS != 0 )); then
  echo "[FATAL] NUM_DATA ($NUM_DATA) must be divisible by BS ($BS)." >&2
  exit 1
fi

_default_raw() {
  case "$1" in
    cc3m)      echo "$BASE/CC3M_calib_128/cc3m_calib_128.json" ;;
    mathvista) echo "$BASE/MathVista_calibration/mathvista_calibration_train.json" ;;
    okvqa)     echo "$BASE/datasets/okvqa/annotations/okvqa_train.json" ;;
    mmbench)   echo "$BASE/MMBench_calibration/mmbench_calibration_train.json" ;;
    mmmu)      echo "$BASE/MMMU_calibration/mmmu_calibration_train.json" ;;
    *) echo "" ;;
  esac
}

raw_for() {
  local source="$1"
  local var="RAW_${source^^}"
  echo "${!var:-$(_default_raw "$source")}"
}

text_for() {
  local source="$1"
  local var="TEXT_${source^^}"
  echo "${!var:-$TEXT_DIR/${source}_text_calib_${NUM_DATA}.json}"
}

include_choices_for() {
  case "$1" in
    mathvista|mmbench|mmmu) echo "1" ;;
    *) echo "0" ;;
  esac
}

preflight() {
  [[ -f "$BLIP2_PRETRAINED" ]] || {
    echo "[FATAL] BLIP2_PRETRAINED not found: $BLIP2_PRETRAINED" >&2
    exit 1
  }
  [[ -f "$CFG" ]] || {
    echo "[FATAL] CFG not found: $CFG" >&2
    exit 1
  }
  [[ -d "${BERT_BASE_UNCASED_SNAPSHOT:-}" ]] || {
    echo "[FATAL] bert-base-uncased snapshot not found. Set BERT_BASE_UNCASED_SNAPSHOT or HF_HOME." >&2
    exit 1
  }
  [[ -d "${FLAN_T5_XL_SNAPSHOT:-}" ]] || {
    echo "[FATAL] flan-t5-xl snapshot not found. Set FLAN_T5_XL_SNAPSHOT or HF_HOME." >&2
    exit 1
  }

  if [[ "$RUN_EVAL" == "1" ]]; then
    [[ -d "$MMBENCH_ROOT" ]] || {
      echo "[FATAL] MMBENCH_ROOT not found: $MMBENCH_ROOT" >&2
      exit 1
    }
    [[ -d "$MMMU_ROOT" ]] || {
      echo "[FATAL] MMMU_ROOT not found: $MMMU_ROOT" >&2
      exit 1
    }
    [[ -f "$OKVQA_EVAL_CFG" ]] || {
      echo "[FATAL] OKVQA_EVAL_CFG not found: $OKVQA_EVAL_CFG" >&2
      exit 1
    }
    [[ -f "$MATHVISTA_EVAL_JSON" ]] || {
      echo "[FATAL] MATHVISTA_EVAL_JSON not found: $MATHVISTA_EVAL_JSON" >&2
      exit 1
    }
    [[ -d "$MATHVISTA_IMAGES_DIR" ]] || {
      echo "[FATAL] MATHVISTA_IMAGES_DIR not found: $MATHVISTA_IMAGES_DIR" >&2
      exit 1
    }
  fi
}

job_for() {
  local source="$1"
  echo "atv_textonly_${source}_t5only_${JOB_STAMP}_seed${SEED}"
}

ckpt_for() {
  local source="$1"
  echo "$REPO_ROOT/pruned_checkpoint/$(job_for "$source").pth"
}

eval_label_for() {
  case "$1" in
    mmbench) echo "MMBench" ;;
    mathvista) echo "MathVista" ;;
    mmmu) echo "MMMU" ;;
    okvqa) echo "OKVQA" ;;
    cc3m) echo "CC3M" ;;
    *) echo "$1" ;;
  esac
}

build_text_json() {
  local source="$1"
  local raw text include_choices
  raw="$(raw_for "$source")"
  text="$(text_for "$source")"
  include_choices="$(include_choices_for "$source")"

  if [[ -f "$text" && "$FORCE_BUILD" != "1" ]]; then
    echo "[build] reuse $source text calibration: $text"
    return 0
  fi
  [[ -f "$raw" ]] || {
    echo "[FATAL] $source raw calibration JSON not found: $raw" >&2
    echo "        Override with RAW_${source^^}=..." >&2
    exit 1
  }

  echo "[build] $source: $raw -> $text"
  python - "$source" "$raw" "$text" "$NUM_DATA" "$SEED" "$include_choices" <<'PY'
import json
import os
import random
import sys

source, raw_path, out_path, num, seed, include_choices = sys.argv[1:7]
num = int(num)
seed = int(seed)
include_choices = include_choices == "1"

TEXT_KEYS = (
    "text",
    "caption",
    "text_input",
    "question",
    "sent",
    "query",
    "prompt",
    "output",
)
CHOICE_KEYS = ("choices", "options", "option")
LETTER_KEYS = tuple("ABCDEFG")


def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("annotations", "data", "questions", "items", "samples"):
            if isinstance(data.get(key), list):
                return data[key]
        return list(data.values())
    raise ValueError("unsupported top-level JSON type: %s" % type(data).__name__)


def stringify(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    if isinstance(value, (list, tuple)):
        return " ".join(stringify(v) for v in value if stringify(v)).strip()
    if isinstance(value, dict):
        parts = []
        for key, val in value.items():
            text = stringify(val)
            if text:
                parts.append("%s. %s" % (key, text))
        return " ".join(parts).strip()
    return str(value).strip()


def extract_text(row):
    if isinstance(row, str):
        return row.strip()
    if not isinstance(row, dict):
        return ""

    base = ""
    for key in TEXT_KEYS:
        if key in row:
            base = stringify(row.get(key))
            if base:
                break
    if not base:
        return ""

    parts = [base]
    hint = stringify(row.get("hint"))
    if hint and "hint:" not in base.lower():
        parts.append("Hint: " + hint)

    if include_choices:
        for key in CHOICE_KEYS:
            choice_text = stringify(row.get(key))
            if choice_text and choice_text not in base:
                parts.append(choice_text)
                break
        for key in LETTER_KEYS:
            choice_text = stringify(row.get(key))
            if choice_text and ("%s." % key) not in base:
                parts.append("%s. %s" % (key, choice_text))
    return "\n".join(parts).strip()


rows = load_rows(raw_path)
texts = []
seen = set()
for row in rows:
    text = extract_text(row)
    if not text or text in seen:
        continue
    seen.add(text)
    texts.append(text)

if not texts:
    raise SystemExit("[FATAL] no usable text extracted from %s" % raw_path)

rng = random.Random(seed)
rng.shuffle(texts)
if len(texts) < num:
    print("[WARN] %s unique texts=%d < NUM_DATA=%d; repeating to fill." % (source, len(texts), num))
    reps = (num // len(texts)) + 1
    texts = (texts * reps)[:num]
else:
    texts = texts[:num]

os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(texts, f, ensure_ascii=False, indent=2)
    f.write("\n")
print("[OK] wrote %s text rows: %s" % (len(texts), out_path))
PY
}

prune_one() {
  local source="$1"
  local text job ckpt log
  text="$(text_for "$source")"
  job="$(job_for "$source")"
  ckpt="$(ckpt_for "$source")"
  log="$LOG_DIR/${job}.log"

  [[ -f "$text" ]] || {
    echo "[FATAL] text calibration JSON not found: $text" >&2
    exit 1
  }

  echo ""
  echo ">>> [prune] $source | text-only ATV entry | T5 only"
  echo "    text_json=$text"
  echo "    ckpt=$ckpt"
  python evaluate_blip.py \
    --cfg-path "$CFG" \
    --options "model.pretrained=${BLIP2_PRETRAINED}" \
    --prune_calib_mode t5_c4_text \
    --c4_calib_json "$text" \
    --pruning_method blipt5_atv_pruner \
    --atv_alpha "$ATV_ALPHA" \
    --no_prune_vit \
    --t5_prune_spec "$T5_SPEC" \
    --num_data "$NUM_DATA" \
    --num_data_first_stage "$NUM_DATA" \
    --prunining_dataset_batch_size "$BS" \
    --job_id "$job" \
    --save_pruned_model 2>&1 | tee "$log"

  [[ -f "$ckpt" ]] || {
    echo "[FATAL] pruning finished but checkpoint not found: $ckpt" >&2
    exit 1
  }
  echo "[OK] $source checkpoint: $ckpt"
}

eval_four_bench() {
  local source="$1"
  local label eval_tag ckpt metrics_jsonl summary_md summary_tsv okvqa_job okvqa_port
  label="$(eval_label_for "$source")"
  eval_tag="atv_textonly_${source}_${JOB_STAMP}_seed${SEED}"
  ckpt="$(ckpt_for "$source")"
  metrics_jsonl="$SUMMARY_DIR/atv_textonly_${source}_fourbench_${JOB_STAMP}_seed${SEED}.jsonl"
  summary_md="$SUMMARY_DIR/atv_textonly_${source}_fourbench_${JOB_STAMP}_seed${SEED}.md"
  summary_tsv="$SUMMARY_DIR/atv_textonly_${source}_fourbench_${JOB_STAMP}_seed${SEED}.tsv"
  okvqa_job="okvqa_eval_${eval_tag}_fullval"

  [[ -f "$ckpt" ]] || {
    echo "[FATAL] eval checkpoint not found for $source: $ckpt" >&2
    echo "        If reusing an old checkpoint, set JOB_STAMP/SEED to match it or run RUN_PRUNE=1." >&2
    exit 1
  }

  export LAVIS_METRICS_JSONL="$metrics_jsonl"
  export LAVIS_EVAL_CALIB_TAG="$eval_tag"
  : > "$LAVIS_METRICS_JSONL"

  echo ""
  echo ">>> [eval] $label ATV text-only checkpoint on four benchmarks"
  echo "    ckpt=$ckpt"
  echo "    metrics_jsonl=$LAVIS_METRICS_JSONL"

  echo ""
  echo ">>> [$eval_tag] MMBench"
  export LAVIS_METRICS_BENCHMARK="MMBench"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMBENCH_ROOT" \
    --split "$MMBENCH_SPLIT" \
    --ckpt "$ckpt" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --device cuda \
    --overall_only

  echo ""
  echo ">>> [$eval_tag] OKVQA full val"
  export LAVIS_METRICS_BENCHMARK="OKVQA"
  okvqa_port="$MASTER_PORT"
  MASTER_PORT=$((MASTER_PORT + 1))
  python -m torch.distributed.run --nproc_per_node=1 --master_port="$okvqa_port" evaluate_blip.py \
    --cfg-path "$OKVQA_EVAL_CFG" \
    --options "run.num_workers=${OKVQA_EVAL_NUM_WORKERS}" \
    --t5_pruned_checkpoint "$ckpt" \
    --vit_pruned_checkpoint "$ckpt" \
    --job_id "$okvqa_job"

  echo ""
  echo ">>> [$eval_tag] MMMU"
  export LAVIS_METRICS_BENCHMARK="MMMU"
  python scripts/blip2/mmmu_eval_by_discipline.py \
    --mmmu_root "$MMMU_ROOT" \
    --split "$MMMU_SPLIT" \
    --ckpt "$ckpt" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --device cuda \
    --overall_only

  echo ""
  echo ">>> [$eval_tag] MathVista MC"
  export LAVIS_METRICS_BENCHMARK="MathVista_MC"
  python scripts/blip2/mathvista_mc_eval.py \
    --eval_json "$MATHVISTA_EVAL_JSON" \
    --images_dir "$MATHVISTA_IMAGES_DIR" \
    --ckpt "$ckpt" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --device cuda

  if [[ -f "$SCRIPT_DIR/collect_lavisbackup_eval_summary.py" ]]; then
    echo ""
    echo ">>> [summary] $source"
    python "$SCRIPT_DIR/collect_lavisbackup_eval_summary.py" \
      --repo-root "$REPO_ROOT" \
      --metrics-jsonl "$LAVIS_METRICS_JSONL" \
      --out-md "$summary_md" \
      --out-tsv "$summary_tsv" \
      --suites "${eval_tag}:${okvqa_job}" || true
  fi

  echo "[OK] $source four-benchmark eval finished"
  echo "     summary_md=$summary_md"
  echo "     summary_tsv=$summary_tsv"
}

preflight

echo "========== text-only ATV-entry T5 pruning =========="
echo "[INFO] REPO_ROOT=$REPO_ROOT"
echo "[INFO] SOURCES=$SOURCES"
echo "[INFO] NUM_DATA=$NUM_DATA BS=$BS T5_SPEC=$T5_SPEC ATV_ALPHA=$ATV_ALPHA"
echo "[INFO] TEXT_DIR=$TEXT_DIR"
echo "[INFO] RUN_BUILD=$RUN_BUILD RUN_PRUNE=$RUN_PRUNE RUN_EVAL=$RUN_EVAL"
echo "[INFO] eval roots: MMBENCH_ROOT=$MMBENCH_ROOT | MMMU_ROOT=$MMMU_ROOT"
echo "[INFO] MathVista eval: $MATHVISTA_EVAL_JSON"
echo "[INFO] JOB_STAMP=$JOB_STAMP seed=$SEED"
echo "[WARN] No images are passed; ATV query-token selection degenerates to text-only Wanda-style scaling."
echo "===================================================="

for source in $SOURCES; do
  case "$source" in
    cc3m|mathvista|okvqa|mmbench|mmmu) ;;
    *)
      echo "[FATAL] unsupported source '$source'. Use: cc3m mathvista okvqa mmbench mmmu" >&2
      exit 1
      ;;
  esac

  echo ""
  echo "############################## $source ##############################"
  if [[ "$RUN_BUILD" == "1" ]]; then
    build_text_json "$source"
  fi
  if [[ "$RUN_PRUNE" == "1" ]]; then
    prune_one "$source"
  else
    echo "[INFO] RUN_PRUNE=0, skip pruning for $source"
    [[ -f "$(ckpt_for "$source")" ]] || {
      echo "[FATAL] RUN_PRUNE=0 but checkpoint is missing: $(ckpt_for "$source")" >&2
      exit 1
    }
  fi

  if [[ "$RUN_EVAL" == "1" ]]; then
    eval_four_bench "$source"
  else
    echo "[INFO] RUN_EVAL=0, skip four-benchmark eval for $source"
  fi
done

echo ""
echo "========== all requested text-only ATV-entry pruning jobs finished =========="
echo "[INFO] text JSONs: $TEXT_DIR/<source>_text_calib_${NUM_DATA}.json"
echo "[INFO] ckpts:      $REPO_ROOT/pruned_checkpoint/atv_textonly_<source>_t5only_${JOB_STAMP}_seed${SEED}.pth"
echo "[INFO] logs:       $LOG_DIR"
echo "[INFO] eval jsonl: $SUMMARY_DIR/atv_textonly_<source>_fourbench_${JOB_STAMP}_seed${SEED}.jsonl"
