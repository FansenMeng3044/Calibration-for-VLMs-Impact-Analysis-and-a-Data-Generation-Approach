# Cosmos3-Edge Reasoner WANDA migration

This directory contains the Cosmos migration of the existing LLaVA layer-wise WANDA implementation. It is deliberately limited to the Hugging Face `Cosmos3EdgeForConditionalGeneration` Reasoner. The diffusion/Generator tower and VAE are not loaded, counted, hooked, pruned, or saved.

## Locked protocols

`joint` (`cosmos_wanda_joint_reasoner`):

1. Images are processed locally through `model.visual` and prune the 27 vision encoder layers.
2. The already-pruned vision encoder and dense projector produce real visual embeddings.
3. The fused image+text AR sequence prunes the 28 `model.language_model.layers`.

This VIT-first/AR-second ordering exactly matches the current LLaVA `LLaVALayerWandaPruner.prune()` behavior.

`separate` (`cosmos_wanda_separate_reasoner`):

1. Images alone prune `model.visual.encoder.layers`; projector and AR calls are forbidden.
2. Text alone follows language tokenizer → token embedding → `model.language_model.layers`; vision and projector calls are forbidden and the number of visual tokens must be zero.
3. Both disjoint sets of masks remain in one complete Reasoner. Final verification is a normal image+text forward.

Both protocols prune only `nn.Linear.weight` inside the vision and AR transformer blocks. Projector, embeddings, norms, biases, and `lm_head` stay dense.

Here, **AR sparsity means the sparsity of the target Linear weights in the
autoregressive language tower**. It is not input/token sparsity. Images and
text are calibration inputs used only to collect the activation term in the
WANDA score; the resulting mask is applied to `nn.Linear.weight`.

For the completed 50%/50% experiment, the target set contains 411,070,464
vision-encoder Linear weights and 1,409,286,144 language-tower Linear weights.
Exactly 50% of each target set is zeroed. Because the projector, embeddings,
normalization parameters, and LM head remain dense, the resulting sparsity
relative to all 2,435,616,496 Reasoner parameters is approximately 37.37%.

## Server paths

```text
code:  /private/workspace/hycui/mfs/cosmos_wanda
model: /private/workspace/hycui/model/Cosmos3-Edge
env:   /private/workspace/hycui/envs/cosmos3-edge
```

## Module audit

```bash
CUDA_VISIBLE_DEVICES=0 /private/workspace/hycui/envs/cosmos3-edge/bin/python \
  /private/workspace/hycui/mfs/cosmos_wanda/cosmos_wanda_prune.py \
  --protocol inspect \
  --device cuda:0
```

The audit must report 27 vision layers, 28 AR layers, non-empty Linear allow-lists, and no Generator modules.

## One-layer smoke test

```bash
bash /private/workspace/hycui/mfs/cosmos_wanda/run_smoke.sh separate 0
bash /private/workspace/hycui/mfs/cosmos_wanda/run_smoke.sh joint 0
```

Smoke tests use one OK-VQA record and prune only the first vision/AR layer. They do not save a checkpoint.

## Calibration dataset presets

`calibration_presets.json` defines the three required calibration sources:

- `mmbench`
- `mmmu`
- `okvqa`

Each source contains two different interfaces:

- joint: one ShareGPT4V image+text JSON.
- separate vision: the corresponding image-only JSON and image root.
- separate AR: the corresponding text-only JSON.

Before loading model weights, validate all paths, all 128 images, and the one-to-one text alignment between each separate image/text pair:

```bash
bash /private/workspace/hycui/mfs/cosmos_wanda/preflight_three_datasets.sh joint 128
bash /private/workspace/hycui/mfs/cosmos_wanda/preflight_three_datasets.sh separate 128
```

The separate preflight fails if any paired row has different question text, a missing image, or unequal branch lengths.

## Full pruning matrix

Create one checkpoint per calibration source (three checkpoints per protocol), matching the existing LLaVA calibration-source matrix design:

```bash
bash /private/workspace/hycui/mfs/cosmos_wanda/run_three_dataset_matrix.sh \
  joint \
  0 \
  /private/workspace/hycui/Results/mfs/cosmos_wanda_joint_three_calib \
  128

bash /private/workspace/hycui/mfs/cosmos_wanda/run_three_dataset_matrix.sh \
  separate \
  0 \
  /private/workspace/hycui/Results/mfs/cosmos_wanda_separate_three_calib \
  128
```

The matrix runner processes `mmbench mmmu okvqa` separately. It does not concatenate them into one 384-sample checkpoint.

## Single-source full pruning

```bash
CUDA_VISIBLE_DEVICES=0 /private/workspace/hycui/envs/cosmos3-edge/bin/python \
  /private/workspace/hycui/mfs/cosmos_wanda/cosmos_wanda_prune.py \
  --protocol separate \
  --calibration-preset okvqa \
  --nsamples 128 \
  --vision-sparsity 0.5 \
  --ar-sparsity 0.5 \
  --device cuda:0 \
  --output-dir /private/workspace/hycui/Results/mfs/cosmos_wanda_separate_okvqa_both50
```

Change `separate` to `joint`, or change the preset to `mmbench`/`mmmu`, as needed.

Explicit paths are also supported. Joint uses repeated `--calibration-json`. Separate accepts either paired `--calibration-json`, or the strict split interface with repeated `--vision-calibration-json` and `--ar-calibration-json`. Do not mix presets and explicit JSON paths in one invocation.

## Output contract

Each run writes:

- `state.json`: `starting`, `vision_calibration`, `vision_pruning`, `ar_calibration`, `ar_pruning`, `verification`, `saving`, `complete`, or `failed`.
- `metadata.json`: module allow-lists, sample IDs, modality token counts, call counts, WANDA normalization, per-layer/per-Linear sparsity, final zero counts, and final multimodal verification.
- `checkpoint/`: pruned Reasoner and processor when model saving is enabled.
- `wanda_masks.pt`: optional masks with `--save-masks`.

The script refuses to overwrite a non-empty output directory. A failed run keeps its traceback in `state.json`.

## Mandatory protocol assertions

- Joint AR samples have both image and language tokens and execute vision, projector, and AR exactly once per sample.
- Separate vision executes no projector or AR module.
- Separate AR has no pixel/grid input, no image/video placeholder token, no visual token type, and executes no vision/projector module.
- Vision and AR target names are disjoint and belong only to the Reasoner.
- Both vision and AR must have non-zero target sparsity and non-zero resulting weights.
- Final verification executes the complete image+text Reasoner and produces finite logits.

## Evaluation: MMBench, MMMU, and OKVQA

The evaluation path reuses the same three local lmms-eval tasks and scorers as
the existing LLaVA runners under `/private/workspace/hycui/project/Tamp`:

- `mmbench_en_dev_local`: local exact option parsing, category/L2 breakdown,
  JSON summary, and XLSX predictions.
- `mmmu_val_local`: official lmms-eval MMMU parser/evaluator plus overall,
  domain, and subject JSON summaries.
- `okvqa_val2014_local`: official lmms-eval OK-VQA answer processing and local
  val2014 images.

`cosmos_lmms_plugin` is loaded through `LMMS_EVAL_PLUGINS`, so the installed
TAMP lmms-eval package is not patched. The adapter loads only
`Cosmos3EdgeForConditionalGeneration` (the Reasoner), accepts either the dense
model or a saved WANDA checkpoint, preserves MMMU image-placeholder order, and
resets Cosmos multimodal rotary state between examples.

Install the compatible evaluation layer once:

```bash
bash /private/workspace/hycui/mfs/cosmos_wanda/install_eval_deps.sh
```

Run all three evals on one physical GPU:

```bash
GPU_ID=0 bash /private/workspace/hycui/mfs/cosmos_wanda/run_three_eval.sh \
  /private/workspace/hycui/model/Cosmos3-Edge all \
  /private/workspace/hycui/Results/mfs/cosmos_eval/dense
```

The second positional argument can also be `mmbench`, `mmmu`, or `okvqa`.
`TAMP_EVAL_LIMIT=1` performs a one-example interface smoke test. Passing a
joint/separate pruning checkpoint as the first argument evaluates that pruned
Reasoner with exactly the same task and metric code.

## Completed 6x3 matrix

The repository includes the complete validated evaluation package for the
BF16, `vision_sparsity=0.5`, `ar_sparsity=0.5`, `nsamples=128`, `seed=42`
experiment:

- [human-readable result table](results/6x3_bf16_s50_n128_seed42/README.md)
- [aggregate CSV](results/6x3_bf16_s50_n128_seed42/summary/cosmos_wanda_6x3_plus_dense.csv)
- [aggregate JSON](results/6x3_bf16_s50_n128_seed42/summary/cosmos_wanda_6x3_plus_dense.json)
- [LLaVA-to-Cosmos migration memo](docs/migration_llava_code_locator.md)

The result package contains all 21 raw lmms-eval result JSON files, all 21
per-sample JSONL files, validation manifests, and benchmark submission files.
Model checkpoints are intentionally excluded from Git because the six saved
checkpoints occupy approximately 24 GB; their metadata and validation reports
are included under `results/.../pruning/`.
