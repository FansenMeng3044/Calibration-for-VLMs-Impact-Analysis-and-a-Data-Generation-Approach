# BLIP2-T5 TAMP Migration Validation

This note records the validation gates for the BLIP2-T5 TAMP migration fixes.
It is intentionally tied to runnable scripts so the migration can be audited
before new pruning/evaluation numbers are used.

## What Is Covered

The validation targets the known TAMP-to-BLIP2-T5 migration hazards:

- AMIA must use attention contribution scores, not an all-one fallback.
- T5 encoder AMIA scores must use column-wise encoder attention under the raw
  encoder attention mask, not a causal-LM last attention row.
- TAMP alias must derive `max_sparsity_per_layer` as `sparsity + 0.1`.
- Decoder and non-encoder tensors must fall back to uniform sparsity under DAS,
  rather than participating in the encoder DAS allocation with a hardcoded score.
- DAS must be computed per T5 encoder `Linear`, not once per block.
- PAD tokens must be excluded from AMIA/DAS and AMIA token selection must be
  per sample, not flattened across the batch.
- AMIA selection must avoid reselection and avoid the old O(N^3) density loop.
- T5 block replay must propagate `position_bias` between layers.

## Local Checks

From the LAVIS root:

```bash
python scripts/blip2/validate_tamp_migration.py \
  --lavis_root . \
  --out_json lavis/output/tamp_migration_validation/static_validation.json

python scripts/blip2/smoke_tamp_core_ops.py \
  --lavis_root . \
  --out_json lavis/output/tamp_migration_validation/core_smoke.json

python scripts/blip2/check_tamp_validation_outputs.py \
  --out_dir lavis/output/tamp_migration_validation
```

The static validation should report `ok=true` with no failed checks. The core
smoke should report positive `amia_selected_rows` and `das_language_density=0.0`.

## Real BLIP2-T5 Runtime Gate

Run this in the same environment used for BLIP2 evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/blip2/run_tamp_migration_validation.sh \
  --calib_json /data/data2/mfs/MMBench_calibration/mmbench_calib_128.json \
  --images_dir /data/data2/mfs/MMBench_calibration/images \
  --ckpt /data/data2/mfs/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth \
  --max_samples 2 \
  --batch_size 2 \
  --run_das \
  --out_dir lavis/output/tamp_migration_validation
```

If the desired Python executable needs to be selected explicitly:

```bash
PYTHON_BIN="conda run -n ecoflap python" CUDA_VISIBLE_DEVICES=0 \
  bash scripts/blip2/run_tamp_migration_validation.sh ...
```

The run is complete only when
`lavis/output/tamp_migration_validation/runtime_smoke.json` exists and
`check_tamp_validation_outputs.py --require_runtime` passes.

The runtime JSON must contain:

- `encoder_input_summary.total_visual_query_tokens > 0`
- `encoder_input_summary.total_valid_text_tokens > 0`
- `importance_scope == "llm_only"`
- `prune_t5 == true` and `prune_vit == false`
- `max_sparsity_per_layer == min(1.0, sparsity + 0.1)`
- `amia_score_summary.valid_mean > 0`
- `amia_score_summary.invalid_abs_max == 0`
- `amia_selected_rows > 0`
- `amia_selected_rows <= amia_valid_rows_first_batch`
- `0 < amia_selected_fraction_first_batch <= 1`
- `das_summary.encoder_keys > 0`
- `das_summary.decoder_fallback_keys > 0`

Until this runtime gate passes, the source-level and torch-only fixes are
validated, but the migration is not fully validated on a real BLIP2-T5 model.
