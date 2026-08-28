# Cosmos3-Edge Reasoner SparseGPT

This directory migrates the existing TAMP/LLaVA layer-wise SparseGPT kernel to
the Hugging Face `Cosmos3EdgeForConditionalGeneration` Reasoner. It never
loads, counts, hooks, prunes, or saves the Cosmos Generator/diffusion/VAE.

The complete experiment model is the Reasoner. The default prunable weights
are exactly the `nn.Linear.weight` tensors in:

- `model.visual.encoder.layers`: 27 layers, 162 Linear modules;
- `model.language_model.layers`: 28 layers, 168 Linear modules.

The projector, token embedding, norms, bias tensors, and `lm_head` stay dense.

## Locked protocols

`joint` (`cosmos_sparsegpt_joint_reasoner`):

1. Real images provide vision Hessians and the vision encoder is pruned first.
2. The same paired image+text samples run through the already-pruned vision
   encoder and dense projector.
3. The fused visual+language sequence provides AR Hessians and the AR layers
   are pruned second.

`separate` (`cosmos_sparsegpt_separate_reasoner`):

1. Real images locally prune only the vision encoder; projector and AR calls
   are forbidden.
2. Text follows tokenizer -> token embedding -> AR layers. Pixel/grid inputs,
   visual placeholders, vision outputs, and projector outputs are forbidden;
   the visual-token count is exactly zero.
3. Both disjoint sets of sparse weights remain in one complete Reasoner. The
   final verification and all evaluations use the normal image+text forward.

The only intended joint/separate variable is whether the AR Hessian has seen
real visual representations.

## SparseGPT kernel

`sparsegpt_core.py` implements:

- full FP32 `H = 2/n * sum(X X^T)` over valid calibration tokens;
- bounded Cholesky/inverse-Cholesky retries;
- 128-input-column blocks by default;
- `W^2 / diag(Hinv)^2` block ranking;
- sequential OBS/SparseGPT error propagation and layerwise recomputation.

The default `exact_k_budget` selects exactly `floor(block_elements*sparsity)`
weights per block. `legacy_llava_threshold` is available only to reproduce the
TAMP/LLaVA `<= threshold` off-by-one and is always recorded in metadata.

## Files

- `cosmos_sparsegpt_prune.py`: Reasoner adapter, protocol/dataflow guards,
  layerwise orchestration, metadata and save logic.
- `sparsegpt_core.py`: standalone SparseGPT statistics and reconstruction.
- `test_sparsegpt_core.py`: deterministic CPU unit test for exact/legacy masks.
- `calibration_presets.json`: aligned MMBench/MMMU/OK-VQA calibration paths.
- `validate_calibration_alignment.py`: proves joint and separate samples align.
- `validate_cosmos_checkpoint.py`: reloads checkpoint, checks zeros and runs a
  real multimodal Reasoner forward.
- `run_smoke.sh`, `run_prune.sh`, `run_three_dataset_matrix.sh`: direct runners.
- `prepare_full_matrix.py`, `run_full_matrix_task.py`,
  `run_full_matrix_worker.sh`: resumable matrix queue.
- `MIGRATION_MEMO.md`: source-code locator, exact protocol and migration redlines.

## Validation before GPU work

```bash
PY=/private/workspace/hycui/envs/cosmos3-edge/bin/python
CODE=/private/workspace/hycui/mfs/cosmos_sparsegpt

$PY -m py_compile $CODE/*.py $CODE/cosmos_lmms_plugin/models/*.py
$PY $CODE/test_sparsegpt_core.py
bash -n $CODE/*.sh
bash $CODE/preflight_three_datasets.sh joint 128
bash $CODE/preflight_three_datasets.sh separate 128
```

## One-layer smoke

```bash
bash /private/workspace/hycui/mfs/cosmos_sparsegpt/run_smoke.sh joint 0 mmbench
bash /private/workspace/hycui/mfs/cosmos_sparsegpt/run_smoke.sh separate 0 mmbench
```

The smoke uses one sample, one vision layer and one AR layer, does not save a
model, and still performs a final full image+text Reasoner verification.

## Formal single run

```bash
bash /private/workspace/hycui/mfs/cosmos_sparsegpt/run_prune.sh \
  joint 0 /private/workspace/hycui/Results/mfs/cosmos_sparsegpt_joint_mmbench \
  mmbench 128

bash /private/workspace/hycui/mfs/cosmos_sparsegpt/run_prune.sh \
  separate 0 /private/workspace/hycui/Results/mfs/cosmos_sparsegpt_separate_mmbench \
  mmbench 128
```

The runner deliberately accepts a dataset preset instead of a raw JSON path.
For `separate`, the preset always resolves to the matching image-only vision
JSON plus text-only AR JSON. Direct Python calls may instead pass both
`--vision-calibration-json` and `--ar-calibration-json`; never mix paired and
split arguments.

## MMBench, MMMU and OK-VQA evaluation

Evaluation deliberately reuses the exact local TAMP/LLaVA task definitions and
scorers. Only the lmms-eval model adapter changes from `llava` to the
Reasoner-only `cosmos3_edge` plugin.

```bash
GPU_ID=0 bash /private/workspace/hycui/mfs/cosmos_sparsegpt/run_three_eval.sh \
  /path/to/cosmos/checkpoint all /path/to/eval_output
```

The second argument may be `all`, `mmbench`, `mmmu`, or `okvqa`. Before GPU
loading, the runner checks the selected dataset assets; MMMU additionally
checks all 30 subject shards, schema and 900-example count. After inference it
runs `validate_eval_output.py`, verifies the metric and every logged response,
requires exact sample counts (MMBench 4,329; MMMU 900; OK-VQA 5,046), checks
benchmark-specific submission artifacts, writes `validation_<benchmark>.json`,
and creates `.done_<benchmark>` only after validation succeeds.

The shared task contracts are:

- MMBench: `mmbench_en_dev_local`, local exact-choice scoring, 32 new tokens;
- MMMU: `mmmu_val_local`, official lmms-eval parser/evaluator, 16 new tokens;
- OK-VQA: `okvqa_val2014_local`, official VQA exact-match processing, 32 new
  tokens.

All use batch size 1, greedy decoding, normal image+text Reasoner inference,
BF16, eager attention, thinking disabled and the locked 65,536/1,048,576 image
pixel bounds. The same evaluation path is used for dense, joint and separate
checkpoints.

## Output contract

Each run contains:

- `state.json`: atomic phase/failure state;
- `metadata.json`: algorithm, protocol, model scope, samples, token/call audits,
  target modules, Hessian settings, per-Linear Cholesky reports, sparsity and
  final multimodal verification;
- `checkpoint/`: optional saved sparse Reasoner and processor;
- `sparsegpt_masks.pt`: optional masks when `--save-masks` is explicit.

Formal defaults are unquantized BF16, eager attention, thinking disabled,
vision/AR sparsity 0.5, 128 samples, seed 42, `blocksize=128`, `percdamp=0.01`,
`exact_k_budget`, and image pixel bounds 65,536/1,048,576.
