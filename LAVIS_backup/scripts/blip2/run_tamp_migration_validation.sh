#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Validate the BLIP2-T5 TAMP migration.

Always runs:
  1. scripts/blip2/validate_tamp_migration.py
  2. scripts/blip2/smoke_tamp_core_ops.py

Also runs the real BLIP2-T5 runtime smoke when all three data arguments are set:
  --calib_json PATH --images_dir PATH --ckpt PATH

Writes validation evidence under --out_dir:
  static_validation.json, core_smoke.json, and runtime_smoke.json when runtime smoke runs.

Example:
  CUDA_VISIBLE_DEVICES=0 bash scripts/blip2/run_tamp_migration_validation.sh \
    --calib_json /data/data2/mfs/MMBench_calibration/mmbench_calib_128.json \
    --images_dir /data/data2/mfs/MMBench_calibration/images \
    --ckpt /data/data2/mfs/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth \
    --max_samples 2 --batch_size 2 --run_das \
    --out_dir lavis/output/tamp_migration_validation
USAGE
}

CALIB_JSON=""
IMAGES_DIR=""
CKPT=""
MAX_SAMPLES=2
BATCH_SIZE=2
DEVICE="auto"
RUN_DAS=0
OUT_DIR="${OUT_DIR:-lavis/output/tamp_migration_validation}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --calib_json)
      CALIB_JSON="$2"
      shift 2
      ;;
    --images_dir)
      IMAGES_DIR="$2"
      shift 2
      ;;
    --ckpt)
      CKPT="$2"
      shift 2
      ;;
    --max_samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --out_dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --run_das)
      RUN_DAS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a PYTHON_CMD <<< "$PYTHON_BIN"
mkdir -p "$OUT_DIR"

echo "[1/3] Static migration validation"
STATIC_JSON="$OUT_DIR/static_validation.json"
"${PYTHON_CMD[@]}" scripts/blip2/validate_tamp_migration.py --lavis_root . --out_json "$STATIC_JSON"
echo "[OK] wrote static validation JSON: $STATIC_JSON"

echo "[2/3] Torch-only T5 replay + AMIA/DAS core smoke"
CORE_JSON="$OUT_DIR/core_smoke.json"
CORE_ARGS=(
  scripts/blip2/smoke_tamp_core_ops.py
  --lavis_root .
  --out_json "$CORE_JSON"
)
if [[ "$DEVICE" != "auto" ]]; then
  CORE_ARGS+=(--device "$DEVICE")
fi
"${PYTHON_CMD[@]}" "${CORE_ARGS[@]}"
echo "[OK] wrote core smoke JSON: $CORE_JSON"

if [[ -z "$CALIB_JSON" && -z "$IMAGES_DIR" && -z "$CKPT" ]]; then
  "${PYTHON_CMD[@]}" scripts/blip2/check_tamp_validation_outputs.py --out_dir "$OUT_DIR"
  echo "[3/3] Real BLIP2-T5 runtime smoke skipped: pass --calib_json --images_dir --ckpt to enable it."
  exit 0
fi

if [[ -z "$CALIB_JSON" || -z "$IMAGES_DIR" || -z "$CKPT" ]]; then
  echo "[ERROR] runtime smoke needs all of --calib_json, --images_dir, and --ckpt." >&2
  exit 2
fi

echo "[3/3] Real BLIP2-T5 runtime smoke"
RUNTIME_JSON="$OUT_DIR/runtime_smoke.json"
RUNTIME_ARGS=(
  scripts/blip2/smoke_tamp_migration_runtime.py
  --calib_json "$CALIB_JSON"
  --images_dir "$IMAGES_DIR"
  --ckpt "$CKPT"
  --max_samples "$MAX_SAMPLES"
  --batch_size "$BATCH_SIZE"
  --out_json "$RUNTIME_JSON"
)
if [[ "$DEVICE" != "auto" ]]; then
  RUNTIME_ARGS+=(--device "$DEVICE")
fi
if [[ "$RUN_DAS" -eq 1 ]]; then
  RUNTIME_ARGS+=(--run_das)
fi
"${PYTHON_CMD[@]}" "${RUNTIME_ARGS[@]}"
echo "[OK] wrote runtime smoke JSON: $RUNTIME_JSON"
"${PYTHON_CMD[@]}" scripts/blip2/check_tamp_validation_outputs.py --out_dir "$OUT_DIR" --require_runtime
