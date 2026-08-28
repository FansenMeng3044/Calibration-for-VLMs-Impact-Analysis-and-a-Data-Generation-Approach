# Cosmos3-Edge TAMP migration

This directory implements TAMP for the Cosmos3-Edge Reasoner and exposes the
same three local calibration/evaluation datasets used by the LLaVA baseline.
The locked scientific definition is in `MIGRATION_MEMO.md`.

## Scope

Both protocols prune only the 168 `nn.Linear.weight` tensors under:

```text
model.language_model.layers
```

The 27-layer Vision Encoder, projector, embeddings, norms, biases and
`lm_head` remain dense. The Generator/diffusion/VAE model is never loaded.

- `joint`: real image+text calibration; DAS and AMIA see the fused AR sequence.
- `separate`: pure text from the data boundary; tokenizer/embedding/AR only;
  Vision Encoder and projector calls are forbidden and must remain zero.

TAMP execution is:

```text
dense AR DAS scoring
  -> exact global per-Linear keep-budget allocation (cap defaults to 0.6)
  -> per-layer AMIA token selection
  -> WANDA rowwise weight pruning
  -> propagate sparse layer output
```

Separate is not an image-only Vision branch and does not prune the Vision
Encoder. Its DAS reduction is exactly `3*(1-s_l)`.

## Files

```text
cosmos_tamp_prune.py               TAMP core and strict Joint/Separate paths
calibration_presets.json           MMBench/MMMU/OK-VQA calibration sources
validate_calibration_alignment.py  Joint/Text-only sample alignment gate
validate_tamp_migration.py         LLaVA source and Cosmos static contract gate
test_tamp_core.py                  CPU DAS/AMIA/allocation/WANDA tests
validate_cosmos_checkpoint.py      reload and non-target bitwise validation
cosmos_lmms_plugin/                Cosmos Reasoner lmms-eval adapter
run_prune.sh                       one protocol × one calibration dataset
run_smoke.sh                       one sample × one AR layer smoke
run_three_dataset_matrix.sh        three calibration checkpoints for one protocol
run_three_eval.sh                  MMBench/MMMU/OK-VQA evaluation entry
validate_eval_output.py            exact sample/result artifact validator
```

## Calibration interface

Named presets:

```text
mmbench
mmmu
okvqa
```

Joint consumes the paired ShareGPT4V-style JSON and resolves real image paths.
Separate consumes the aligned text-only JSON, while the paired multimodal JSON
is used only for final checkpoint verification and never for importance.

Preflight all three datasets without loading model weights:

```bash
bash /private/workspace/hycui/mfs/cosmos_tamp/preflight_three_datasets.sh joint 128
bash /private/workspace/hycui/mfs/cosmos_tamp/preflight_three_datasets.sh separate 128
```

Verify row-by-row text/image alignment:

```bash
/private/workspace/hycui/envs/cosmos3-edge/bin/python \
  /private/workspace/hycui/mfs/cosmos_tamp/validate_calibration_alignment.py \
  --nsamples 128 \
  --output /private/workspace/hycui/mfs/cosmos_tamp/calibration_alignment.json
```

## Static and CPU validation

```bash
cd /private/workspace/hycui/mfs/cosmos_tamp
/private/workspace/hycui/envs/cosmos3-edge/bin/python -m unittest -v test_tamp_core.py
/private/workspace/hycui/envs/cosmos3-edge/bin/python validate_tamp_migration.py
```

The static gate pins the current operational LLaVA TAMP source hashes. A hash
change means the source must be re-audited before pruning.

## Smoke test

```bash
bash /private/workspace/hycui/mfs/cosmos_tamp/run_smoke.sh joint 0 mmbench
bash /private/workspace/hycui/mfs/cosmos_tamp/run_smoke.sh separate 0 mmbench
```

The smoke processes one calibration example and one AR layer without saving a
checkpoint. It still executes DAS, allocation, AMIA and WANDA.

## One full pruning run

```bash
bash /private/workspace/hycui/mfs/cosmos_tamp/run_prune.sh \
  joint 0 \
  /private/workspace/hycui/Results/mfs/cosmos_tamp_joint_mmbench \
  mmbench 128 0.5 0.6
```

Arguments are:

```text
PROTOCOL PHYSICAL_GPU OUTPUT_DIR DATASET NSAMPLES AR_SPARSITY MAX_SPARSITY_PER_LINEAR
```

Separate example:

```bash
bash /private/workspace/hycui/mfs/cosmos_tamp/run_prune.sh \
  separate 0 \
  /private/workspace/hycui/Results/mfs/cosmos_tamp_separate_okvqa \
  okvqa 128 0.5 0.6
```

## Three calibration checkpoints

```bash
bash /private/workspace/hycui/mfs/cosmos_tamp/run_three_dataset_matrix.sh \
  joint 0 /private/workspace/hycui/Results/mfs/cosmos_tamp_joint 128

bash /private/workspace/hycui/mfs/cosmos_tamp/run_three_dataset_matrix.sh \
  separate 0 /private/workspace/hycui/Results/mfs/cosmos_tamp_separate 128
```

Each checkpoint is reloaded and validated. Formal validation checks:

- 168 AR Linear targets and 1,409,286,144 target weights;
- DAS formula and six Linear reports per each of 28 AR layers;
- per-Linear exact allocation plus rowwise floor behavior;
- AMIA called for every calibration sample and selected valid tokens;
- Vision/projector call provenance for Joint vs Separate;
- all non-target tensors bitwise equal to the dense Reasoner;
- normal image+text forward remains finite after reload.

## Evaluation interface

The adapter uses the local LLaVA task definitions and scorers:

| Benchmark | lmms-eval task | Expected samples |
|---|---|---:|
| MMBench | `mmbench_en_dev_local` | 4,329 |
| MMMU | `mmmu_val_local` | 900 |
| OK-VQA | `okvqa_val2014_local` | 5,046 |

Run all three on a dense or pruned Reasoner checkpoint:

```bash
GPU_ID=0 bash /private/workspace/hycui/mfs/cosmos_tamp/run_three_eval.sh \
  /path/to/checkpoint all /path/to/eval_output
```

Run one benchmark by replacing `all` with `mmbench`, `mmmu`, or `okvqa`.
`run_three_eval.sh` automatically invokes `validate_eval_output.py`, which
requires the exact sample count, finite metric, raw sample JSONL, result JSON
and benchmark-specific submission/scoring artifacts.

For a limited interface smoke:

```bash
TAMP_EVAL_LIMIT=1 GPU_ID=0 bash \
  /private/workspace/hycui/mfs/cosmos_tamp/run_three_eval.sh \
  /private/workspace/hycui/model/Cosmos3-Edge mmbench \
  /private/workspace/hycui/Results/mfs/cosmos_tamp_eval_smoke
```

## Locked runtime defaults

```text
dtype                 = bfloat16
quantization          = none
attention             = eager
thinking              = false
min_image_pixels      = 65536
max_image_pixels      = 1048576
AR target sparsity    = 0.5
max per-Linear sparse = 0.6
nsamples              = 128
seed                  = 42
```

Changing any scientific parameter must be explicit in the run command and
recorded in metadata. Never implement Separate using a dummy/zero image.
