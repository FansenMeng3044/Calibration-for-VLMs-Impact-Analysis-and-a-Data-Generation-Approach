# Cosmos3-Edge Reasoner ATV migration

This directory implements the LLaVA/official ATV algorithm for the Hugging
Face `Cosmos3EdgeForConditionalGeneration` Reasoner. It never loads the Cosmos
Generator/diffusion/VAE path.

The scientific contract is frozen in `MIGRATION_MEMO.md` on the server. The
implementation deliberately prunes **AR/LLM Linear weights only**. The vision
encoder, projector, embedding, norms, biases, and `lm_head` stay dense.

## Locked protocols

`joint` (`cosmos_atv_joint_reasoner`):

1. Run a real paired image+text Reasoner forward through the dense vision
   encoder and dense projector.
2. Capture the fused AR sequence and its aligned valid/visual token masks.
3. At every AR layer, compute visual-token input/output cosine distance.
4. Keep all valid language tokens and the official ATV top-k visual tokens.
5. Apply the WANDA metric to those retained activations and prune only AR
   `nn.Linear.weight` tensors.

`separate` (`cosmos_atv_separate_textonly_reasoner`):

1. Calibration records are strictly text-only and may contain no image/video
   field or visual placeholder.
2. Run only tokenizer -> token embedding -> AR language transformer.
3. Vision and projector calls are forbidden and audited to equal zero.
4. Visual masks and selected-visual counts are forced to zero; alpha is
   recorded as ineffective.
5. Prune only AR `nn.Linear.weight` tensors. The final checkpoint still
   contains the complete dense vision encoder.

The word `joint` refers to multimodal AR activation provenance, not a joint
vision/AR weight target. The word `separate` refers to an isolated text-only AR
subsequence, not image-only vision pruning.

## Paths

```text
code:  /private/workspace/hycui/mfs/cosmos_atv
model: /private/workspace/hycui/model/Cosmos3-Edge
env:   /private/workspace/hycui/envs/cosmos3-edge
```

## Architecture audit

```bash
CUDA_VISIBLE_DEVICES=0 /private/workspace/hycui/envs/cosmos3-edge/bin/python \
  /private/workspace/hycui/mfs/cosmos_atv/cosmos_atv_prune.py \
  --protocol inspect \
  --device cuda:0
```

The audit is fail-closed and expects:

- 27 dense vision layers / 162 dense vision Linear modules;
- 28 AR layers / 168 target AR Linear modules;
- 0 vision target modules;
- 1,409,286,144 AR target weight parameters;
- no Generator modules.

## Calibration presets and preflight

`calibration_presets.json` defines independent MMBench, MMMU, and OK-VQA
sources. Joint uses paired image+text JSON. Separate uses text-only JSON for
importance and a paired JSON only for the final multimodal checkpoint check.
The paired verification rows never contribute pruning activations.

```bash
bash /private/workspace/hycui/mfs/cosmos_atv/preflight_three_datasets.sh joint 128
bash /private/workspace/hycui/mfs/cosmos_atv/preflight_three_datasets.sh separate 128

/private/workspace/hycui/envs/cosmos3-edge/bin/python \
  /private/workspace/hycui/mfs/cosmos_atv/validate_calibration_alignment.py \
  --nsamples 128 \
  --output /private/workspace/hycui/mfs/cosmos_atv/calibration_alignment.json
```

Separate preflight fails if text records contain visual fields/placeholders,
if text/verification questions differ, or if verification images are missing.

## One-layer smoke tests

These prune only the first AR layer, run the final multimodal forward, and do
not save a checkpoint:

```bash
bash /private/workspace/hycui/mfs/cosmos_atv/run_smoke.sh joint 0 mmbench
bash /private/workspace/hycui/mfs/cosmos_atv/run_smoke.sh separate 0 mmbench
```

## Full single-source pruning

```bash
bash /private/workspace/hycui/mfs/cosmos_atv/run_prune.sh \
  joint 0 \
  /private/workspace/hycui/Results/mfs/cosmos_atv_joint_mmbench_s50 \
  mmbench 128 0.5 1.0

bash /private/workspace/hycui/mfs/cosmos_atv/run_prune.sh \
  separate 0 \
  /private/workspace/hycui/Results/mfs/cosmos_atv_separate_mmbench_s50 \
  mmbench 128 0.5 1.0
```

The explicit `--vision-sparsity` compatibility guard must remain exactly zero.
Any non-zero value is rejected before model pruning.

## Three calibration-source checkpoints

```bash
bash /private/workspace/hycui/mfs/cosmos_atv/run_three_dataset_matrix.sh \
  joint 0 \
  /private/workspace/hycui/Results/mfs/cosmos_atv_joint_three_calib 128

bash /private/workspace/hycui/mfs/cosmos_atv/run_three_dataset_matrix.sh \
  separate 0 \
  /private/workspace/hycui/Results/mfs/cosmos_atv_separate_three_calib 128
```

Each source creates a separate checkpoint; the three 128-sample sources are
not concatenated into one 384-sample checkpoint.

## Output contract

Each run writes:

- `state.json`: starting, AR calibration/pruning, verification, saving,
  complete, or failed with traceback;
- `metadata.json`: algorithm variant, source sample IDs, module allow-list,
  dataflow call counts, per-layer ATV selection, per-Linear mask statistics,
  AR sparsity, and final multimodal verification;
- `checkpoint/`: complete Reasoner plus processor when saving is enabled;
- `atv_masks.pt`: optional AR-only masks with `--save-masks`.

The output directory must be empty. Partial-layer runs cannot save a formal
checkpoint unless explicitly allowed.

## Strict checkpoint validation

```bash
CUDA_VISIBLE_DEVICES=0 /private/workspace/hycui/envs/cosmos3-edge/bin/python \
  /private/workspace/hycui/mfs/cosmos_atv/validate_cosmos_checkpoint.py \
  --run-dir /path/to/completed/run \
  --protocol joint \
  --preset mmbench \
  --device cuda:0
```

The validator reloads the checkpoint, confirms all 28 layer statistics and AR
zero counts, runs a normal image+text forward, and compares every non-target
Reasoner tensor bitwise with the dense base model. Only the 168 AR Linear
weight tensors may differ.

## Evaluation

Evaluation always returns to the normal multimodal Reasoner path, including
for text-only calibrated checkpoints:

```text
image + question -> dense vision -> dense projector -> pruned AR -> answer
```

```bash
GPU_ID=0 bash /private/workspace/hycui/mfs/cosmos_atv/run_three_eval.sh \
  /path/to/checkpoint all \
  /private/workspace/hycui/Results/mfs/cosmos_atv_eval/example
```

This reuses the existing local MMBench, MMMU, and OK-VQA task definitions and
scorers under `/private/workspace/hycui/project/Tamp`. Only the model adapter is
Cosmos-specific.
