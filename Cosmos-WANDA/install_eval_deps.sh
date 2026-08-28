#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/private/workspace/hycui/envs/cosmos3-edge/bin/python}"

# lmms-eval 0.4.0 pins NumPy 1.26.4, which has no Python 3.13 wheel. Install
# the harness without dependencies so Cosmos keeps its tested Torch,
# Transformers, and NumPy stack, then add only the runtime dependencies used by
# the three local tasks.
"${PYTHON_BIN}" -m pip install --no-deps lmms-eval==0.4.0
"${PYTHON_BIN}" -m pip install \
  datasets evaluate ftfy jsonlines loguru more-itertools numexpr openai \
  openpyxl pandas pytablewriter python-dateutil python-dotenv sacrebleu \
  scikit-learn sqlitedict tenacity==8.3.0 wandb word2number zstandard

"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as metadata
import lmms_eval
import torch
import transformers

print("lmms-eval", metadata.version("lmms-eval"))
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("lmms_eval_import", lmms_eval.__file__)
PY

