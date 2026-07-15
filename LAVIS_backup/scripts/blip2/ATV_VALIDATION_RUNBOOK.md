# ATV Migration Validation Runbook

This runbook validates the migration of original ATV-Pruning to BLIP2-T5. The
goal is not to show that pruning runs once, but to produce enough evidence that
the migration is mechanistically correct, architecture-aware, reproducible, and
comparable under a shared evaluation protocol.

## Required Evidence Gates

The final report is only strict-pass when all gates pass:

1. Static source mapping from original ATV to BLIP2-T5.
2. Synthetic golden tests for ATV token selection and scaler accumulation.
   These include the edge cases where `k=num_img` must match naive Wanda,
   `k=0` must differ from naive Wanda, and selected indices remain stable under
   low-precision rounding. They also include a BLIP2 batching case where
   padding tokens must not inflate `k` or contribute to `scaler_row`. The
   static source mapping also checks that both original ATV and BLIP2-T5 ATV
   clamp `k` to the available visual/query-token count.
3. `atv_preflight_report.csv` proves that the calibration JSON, image root,
   required row count, image/text fields, bridge checkpoint, eval scripts, and
   required baseline checkpoints exist before expensive GPU runs. Strict
   validation fails if this report is missing or has required failures.
4. `query_token_mapping.csv` proves that the target `pretrain_flant5xl`
   BLIP2-T5 config and constructor use 32 Q-Former query tokens, and that this
   count flows into the learned Q-Former query parameter. This makes the
   "visual token = query token, not raw ViT patch" mapping machine-checkable.
5. `runtime_environment.json` proves the Python/Torch/CUDA environment, git
   commits/dirty state, and hashes of the migrated source/config files. Strict
   validation fails if Torch cannot be imported or this file is missing. The
   required source hashes include the original ATV pruner, BLIP2-T5 model
   implementation, migrated Wanda/ATV pruner, validator, preflight checker,
   runtime provenance capture script, final strict audit script, checkpoint
   comparison/sparsity helpers, dense-base exporter, CC3M config materializer,
   snapshot helper, and all ATV prune/eval drivers.
6. Runtime alpha sweep over `0,0.25,0.5,1,2,4`, with every logged layer
   reporting `num_img=32` so the evidence proves per-sample BLIP2 query-token
   selection rather than flattened-batch selection. Each required alpha must
   cover the expected T5 encoder layer count; a partial log is not enough.
   Every logged layer must also report `k==0=a/b`, and the validator writes
   this as `kzero`, `kzero_n`, and `k_zero_rate` in `alpha_sweep.csv`.
   For `alpha=0`, every logged layer must report `kmean/kmin/kmax=0/0/0`,
   proving the text-only negative-control path.
7. Layer-wise mask comparison across at least 20 T5 encoder layers.
8. Module-wise linear mask comparison across the T5 encoder target modules,
   including keep ratios, prune ratios, mask difference ratios, and the
   checkpoint/prefix/min-size provenance used to infer pruning masks. Strict
   validation requires dense-base-aware mask inference: a weight counts as
   pruned only if it was nonzero in the dense full checkpoint and is zero in the
   pruned checkpoint. Strict mask-pair coverage requires
   `atv_alpha0_vs_wanda`, `atv_alpha1_vs_wanda`, `atv_alpha4_vs_wanda`, and
   `atv_alpha1_vs_tamp`, each at both layer and module granularity.
9. `prune_provenance.csv` proves that the pruned checkpoints used for ATV
   alpha ablations and naive Wanda were produced with the same calibration JSON,
   image directory, sample count, T5 prune spec, T5-only pruning scope, and
   target T5/non-T5 sparsity settings. Strict validation requires ATV alpha
   `0,1,4` and naive Wanda rows for seeds 42, 43, and 44.
10. Alpha-vs-Wanda mask trend: high alpha must be at least as close to naive
   Wanda as low alpha, using mean keep/prune mask IoU.
11. ATV alpha=1 is not silently identical to naive Wanda, unless logs prove
   near-total query-token degeneracy.
12. Same-seed ATV rerun produces identical masks.
13. T5 sparsity is near the target and non-T5 modules remain dense for
    T5-only ATV. This includes `visual_encoder`, `Qformer`/`query_tokens`, and
    `t5_proj`.
   The pruning commands for ATV and the naive Wanda comparator must explicitly
   disable ViT pruning, and `sparsity_summary.csv` must confirm non-T5 groups
   stay dense.
14. BLIP2 token mask integrity proves the first 32 T5 input positions are
    Q-Former query tokens, and all following positions are text tokens. This
    must be checked per sample, not after flattening a whole batch. The same
    CSV must also prove the actual T5 attention mask layout:
    `attention_query_true_count=32`, `valid_text_tokens>0`,
    `pad_text_tokens>=0`, `valid_text_tokens + pad_text_tokens =
    num_text_tokens`, and `attention_layout_ok=1`.
15. `calibration_batch_trace.csv` proves cached batch sizes and cumulative
    physical sample count. This guards against confusing batch count with
    calibration sample count when `batch_size > 1`; strict validation expects at
    least 128 physical samples, 32 query tokens per sample, valid text tokens,
    and audited padding counts.
16. Real importance, scaler_row, and selected-query-token plots are present.
    The importance/scaler plots must be backed by `importance_distribution.csv`
    with finite `mean_wanda_importance`, `scaler_row_*`, `weight_abs_mean`, and
    per-module `mask_sparsity` values near the target T5 sparsity.
17. The full method-by-benchmark matrix is present: dense, ATV, Wanda, and
    TAMP/AMIA each have MMBench, OKVQA, MMMU, and MathVista scores. ATV must
    include the alpha=1 main setting plus alpha=0 and alpha=4 eval ablations.
    Strict validation expects pruned-method evidence for seeds 42, 43, and 44,
    and the validator derives `eval_summary_by_method.csv` with mean/std/n.
    It also requires `eval_provenance.csv`, which maps every method/seed/alpha
    eval row to the checkpoint path and raw evaluation files; by default,
    pruned checkpoints may not be silently shared across required seeds.
    If per-sample prediction files are available, pass them with
    `--prediction_csv TAG=path` so `paired_bootstrap_ci.csv` reports paired
    bootstrap confidence intervals between methods.

## Step 1: Full Mechanism Verification

First materialize the full dense mask-base checkpoint. This file is used only
to infer pruning masks as dense-nonzero-to-pruned-zero; it is not the partial
BLIP2 bridge checkpoint.

```bash
cd /data/data2/mfs/2/LAVIS_backup
python scripts/blip2/export_blip2_full_dense_state_dict.py \
  --ckpt /data/data2/mfs/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth \
  --out /data/data2/mfs/model_cache/full_dense_blip2_t5_flant5xl_state_dict.pth \
  --device cpu
```

The exporter also writes `<out>.summary.json` and `<out>.summary.csv`. Inspect
them before the expensive validation run: `t5_model`, `visual_encoder`,
`Qformer`, `query_tokens`, and `t5_proj` must all have positive tensor counts.
The preflight script reads `<MASK_BASE_CKPT>.summary.json` when present and
fails early if any of these required groups is missing.

Recommended top-level command for final evidence:

```bash
cd /data/data2/mfs/2/LAVIS_backup
BASE=/data/data2/mfs \
ATV_ROOT=/data/data2/mfs/ATV-Pruning \
MASK_BASE_CKPT=/data/data2/mfs/model_cache/full_dense_blip2_t5_flant5xl_state_dict.pth \
CKPT_TAMP_SEED42=/path/to/tamp_or_amia_seed42.pth \
CKPT_TAMP_SEED43=/path/to/tamp_or_amia_seed43.pth \
CKPT_TAMP_SEED44=/path/to/tamp_or_amia_seed44.pth \
bash scripts/blip2/run_atv_multiseed_validation.sh
```

This wrapper uses seed 42 for the mechanism evidence gates and accumulates the
four-benchmark eval matrix for seeds 42, 43, and 44 into one shared
`REPORT_DIR`. It is the least error-prone way to create a strict validation
report. It also writes `strict_audit_summary.csv` and
`strict_audit_summary.md`, which summarize every strict manifest and
traceability gate. By default it first runs a filesystem/input preflight and writes
`REPORT_DIR/atv_preflight_report.csv`; set `RUN_PREFLIGHT=0` only when you are
intentionally debugging a partial setup. The preflight also parses the
calibration JSON, checks that it has at least 128 rows, verifies the image/text
fields, and probes several image files under the image directory. Override these
checks with `EXPECTED_CALIB_SAMPLES`, `CALIB_IMAGE_FIELD`,
`CALIB_TEXT_FIELDS`, and `IMAGE_PROBE_SAMPLES` when intentionally validating a
different calibration file. When `CC3M_JSON` or `CC3M_IMAGES_DIR` is set, the
pruning driver materializes `REPORT_DIR/cc3m_calib_runtime.yaml` and actually
uses that YAML for pruning, so the preflight-checked data and the runtime
calibration data stay aligned.

`MASK_BASE_CKPT` is required for strict mask evidence. It must point to a full
dense BLIP2-T5 `state_dict` that contains the T5 and ViT tensors used by the
pruned checkpoints. Do not use the partial `blip2_pretrained_flant5xl.pth`
bridge checkpoint as this base; that file is sufficient for model loading, but
not for dense-nonzero-to-pruned-zero mask inference. If you are only debugging
locally and intentionally accept weaker mask evidence, set
`ALLOW_ZERO_ONLY_MASK_INFERENCE=1`.

The wrapper also writes
`runtime_environment.json` and `runtime_environment.md` before pruning/eval so
the strict report can trace Python, Torch/CUDA, git commits, dirty status,
selected environment variables, and source-file hashes for the migrated ATV
implementation. Set `RUN_ENV_CAPTURE=0` only for local debugging. By default,
the main seed runs the full alpha grid
`0,0.25,0.5,1,2,4`; non-main seeds only build alpha `0` and `4` checkpoints,
which are the ablation checkpoints required by the eval seed grid. Dense eval
is also run once by default because the strict seed gate only requires
multi-seed evidence for pruned methods. Set `RUN_DENSE_ONCE=0` or
`EVAL_SEED_ALPHA_GRID="0 0.25 0.5 2 4"` if you intentionally want the more
expensive exhaustive version. For TAMP/AMIA, the wrapper expects seed-specific
checkpoint variables such as `CKPT_TAMP_SEED42`; alternatively use
`CKPT_TAMP_TEMPLATE="/path/to/tamp_seed{seed}.pth"`. Reusing one TAMP/AMIA
checkpoint for every seed is allowed only with `ALLOW_SHARED_TAMP_CKPT=1`, and
should be reported as a weaker baseline. Use the lower-level steps below when
debugging a single component.

After the final validator runs, the wrapper also writes
`artifact_snapshot.csv` and `artifact_snapshot.md`. These files record paths,
sizes, timestamps, and SHA256 hashes for report artifacts and source files.
Large checkpoints are recorded by path and size by default; set
`SNAPSHOT_HASH_MAX_BYTES` higher if you want checkpoint hashes as well.

The shared `REPORT_DIR` keeps `eval_results.csv` and
`eval_summary_by_method.csv` as the merged table, but raw evaluation provenance
is kept seed-specific: `eval_matrix_fourbench_metrics_seed<seed>.jsonl` for
MMBench/MMMU/MathVista and `OKVQA/..._seed<seed>_fullval/evaluate.txt` for
OKVQA. This avoids later seeds overwriting the source files referenced by
earlier eval rows.

Run this on the GPU server:

```bash
cd /data/data2/mfs/2/LAVIS_backup
BASE=/data/data2/mfs \
ATV_ROOT=/data/data2/mfs/ATV-Pruning \
SKIP_MMBENCH=1 \
bash scripts/blip2/run_atv_full_verify.sh
```

This lower-level single-seed script also writes `runtime_environment.json` and
`runtime_environment.md` in its own `REPORT_DIR`. It disables runtime capture in
the child prune jobs so the parent report keeps one consistent provenance file.
Use `RUN_ENV_CAPTURE=0` only for temporary debugging; strict validation expects
the runtime file to exist and to show an importable Torch environment. It also
writes a nonfatal `strict_audit_summary.md` by default; this is useful for seeing
which mechanism gates passed, but the single-seed report is expected to remain
incomplete until the multi-seed eval matrix is added.
It is still a mechanism-debugging entry point: final strict validation should
use `run_atv_multiseed_validation.sh`, because the strict manifest also expects
`atv_preflight_report.csv` and the multi-seed four-benchmark evaluation matrix.

Important outputs:

- `REPORT_DIR/atv_preflight_report.csv` when using the multi-seed wrapper with
  `RUN_PREFLIGHT=1`
- `REPORT_DIR/runtime_environment.json`
- `REPORT_DIR/runtime_environment.md`
- `REPORT_DIR/strict_audit_summary.md`
- `REPORT_DIR/query_token_mapping.csv`
- `REPORT_DIR/static_mapping.md`
- `REPORT_DIR/unit_test_results.txt`
- `REPORT_DIR/calibration_batch_trace.csv`
- `REPORT_DIR/token_mask_integrity.csv`
- `REPORT_DIR/alpha_sweep.csv`
- `REPORT_DIR/mask_iou_by_layer.csv`
- `REPORT_DIR/mask_iou_by_module.csv`
- `REPORT_DIR/dense_mask_base_summary.csv`
- `REPORT_DIR/prune_provenance.csv`
- `REPORT_DIR/sparsity_summary.csv`
- `REPORT_DIR/importance_distribution.csv`
- `REPORT_DIR/importance_distribution.png`
- `REPORT_DIR/scaler_row_distribution.png`
- `REPORT_DIR/selected_query_frequency.csv`
- `REPORT_DIR/selected_query_token_frequency.png`
- `REPORT_DIR/validation_manifest.csv`
- `REPORT_DIR/validation_traceability.csv`
- `REPORT_DIR/eval_provenance.csv`
- `REPORT_DIR/eval_summary_by_method.csv`
- `REPORT_DIR/paired_bootstrap_ci.csv`
- `REPORT_DIR/artifact_snapshot.csv`
- `REPORT_DIR/artifact_snapshot.md`
- `REPORT_DIR/final_validation_report.md`

This step intentionally runs a same-seed ATV rerun by default. Set
`RUN_REPRO_CHECK=0` only for debugging; do not use that run as final evidence.

## Step 2: Four-Benchmark Evaluation Matrix

After mechanism verification creates the ATV and naive Wanda checkpoints, run:

```bash
cd /data/data2/mfs/2/LAVIS_backup
BASE=/data/data2/mfs \
ATV_ROOT=/data/data2/mfs/ATV-Pruning \
REPORT_DIR=/data/data2/mfs/atv_validation_report_<STAMP>_seed42 \
CKPT_ATV=/path/to/atv_cc3m_t5only_<STAMP>_seed42.pth \
CKPT_ATV_ALPHA0=/path/to/atv_cc3m_t5only_alpha0_<STAMP>_seed42.pth \
CKPT_ATV_ALPHA4=/path/to/atv_cc3m_t5only_alpha4_<STAMP>_seed42.pth \
CKPT_NAIVE=/path/to/naive_wanda_cc3m_t5only_<STAMP>_seed42.pth \
CKPT_TAMP=/path/to/tamp_or_amia_t5only_seed42.pth \
VALIDATE_STRICT=1 \
bash scripts/blip2/run_atv_eval_matrix_fourbench.sh
```

`VALIDATE_STRICT=1` should be used only after Step 1 has already populated the
same `REPORT_DIR`. The eval script uses `--preserve_existing`, so the mechanism
evidence remains in place while `eval_results.csv` is refreshed.

Run the eval matrix for `SEED=42`, `SEED=43`, and `SEED=44` against the same
`REPORT_DIR`. `validate_atv_migration.py --preserve_existing` merges new eval
rows with existing rows, so the final `eval_results.csv` contains the multi-seed
pool and `eval_summary_by_method.csv` contains mean/std/n.

If an evaluation script emits per-sample rows, provide CSV files with at least
`sample_id` and one of `correct`, `is_correct`, or `score`; optional columns are
`method`, `benchmark`, `seed`, and `alpha`. The validator pairs methods within
each `benchmark` and `seed` over common `sample_id` values and writes 95% paired
bootstrap intervals to `paired_bootstrap_ci.csv`.

When `VALIDATE_STRICT=1`, missing method checkpoints are fatal. This is
intentional: a strict report must contain all 16 method-by-benchmark cells, not
just a partial table.

## Publication-Grade Reporting Protocol

For a paper or rebuttal, report the migration evidence in two separate blocks.
Do not merge them into one accuracy table:

1. Mechanism validation table:
   - static source mapping pass/fail;
   - golden test pass count;
   - token-mask integrity pass/fail with `num_query_tokens=32` and attention
     mask/padding evidence;
   - alpha sweep coverage and selected-k monotonicity;
   - same-seed mask reproducibility;
   - T5 sparsity plus visual encoder, Q-Former/query-token, and projection
     sparsity for the T5-only ATV setting.
2. Downstream performance table:
   - dense, naive Wanda, TAMP/AMIA, ATV alpha=1, ATV alpha=0, and ATV alpha=4;
   - MMBench, OKVQA, MMMU, and MathVista;
   - mean and standard deviation over seeds 42, 43, and 44 for pruned methods;
   - checkpoint path and raw metric-file provenance for every reported cell.

The mechanism table is the evidence that the migration is correct. The
performance table is the evidence about whether the migrated method is useful
under this calibration/evaluation setup. A score improvement without the
mechanism table is not a valid migration proof.

Recommended negative controls:

- ATV `alpha=0`: must select no query tokens (`kmean/kmin/kmax=0/0/0` in
  every logged layer), so it is the strict text-only token accumulation control.
- ATV high alpha, e.g. `alpha=4`: should move toward naive Wanda because more
  query tokens are retained.
- naive Wanda under the same calibration and sparsity: shows whether ATV token
  selection changes the mask instead of reproducing ordinary activation
  accumulation.
- same-seed ATV rerun: should produce identical masks. If it does not, first
  debug nondeterministic data order, CUDA settings, or checkpoint loading before
  interpreting benchmark scores.

Recommended statistical reporting:

- Use mean plus standard deviation across seeds for every method/benchmark.
- If per-sample prediction files are available, include paired bootstrap
  confidence intervals from `paired_bootstrap_ci.csv`.
- Keep dense evaluation separate from multi-seed pruning variance. Dense can be
  evaluated once, but every pruned method should use seed-specific calibration
  and checkpoint provenance.

Failure interpretation:

- If static mapping or golden tests fail, the migration is mechanically wrong.
- If token-mask integrity fails, the BLIP2-T5 architecture mapping is wrong.
- If alpha=1 is identical to naive Wanda and logs do not show near-total
  `k=num_img` degeneracy, ATV token selection is likely not active.
- If `visual_encoder`, `Qformer`/`query_tokens`, or `t5_proj` sparsity is
  nonzero beyond the configured tolerance in the T5-only ATV experiment, the
  pruning scope is contaminated.
- If downstream scores are weak but mechanism gates pass, the implementation may
  still be correct; the result should be reported as a negative or
  non-improving transfer rather than an implementation failure.

## Interpreting the Result

Start with the strict audit summary:

```bash
cat "$REPORT_DIR/strict_audit_summary.md"
```

The audit is generated from the machine-readable source of truth:

```bash
cat "$REPORT_DIR/validation_manifest.csv"
```

Every row with `required_for_strict=yes` must be `PASS`. A green static mapping
or golden test result alone is not enough to claim successful migration; GPU
evidence is required for alpha behavior, mask behavior, sparsity, token masks,
plots, and downstream evaluation.

If you run a partial/debug validation and still want audit files without a
failing shell exit, use:

```bash
python scripts/blip2/audit_atv_validation_report.py \
  --report_dir "$REPORT_DIR" \
  --no_fail
```

For auditability, also inspect:

```bash
cat "$REPORT_DIR/validation_traceability.csv"
```

This maps each migration claim in the validation plan to the manifest gate and
artifact that must prove it.

## Architecture Guardrail

In BLIP2-T5, ATV visual tokens mean the 32 Q-Former query tokens after
`t5_proj` and before concatenation with T5 text embeddings. They are not raw ViT
patch tokens. ViT-side ATV would be a separate method and should not be used as
evidence for the current T5-side ATV migration.

The runtime `token_mask_integrity.csv` must therefore include prefix-layout
evidence: `expected_query_tokens=32`, `query_prefix_true_count=32`,
`text_suffix_true_count=0`, and `query_prefix_ok=1` for every checked sample and
layer. It must also include attention/padding evidence from the same forward
pass: `attention_query_true_count=32`, `valid_text_tokens>0`,
`pad_text_tokens>=0`, `valid_text_tokens + pad_text_tokens = num_text_tokens`,
and `attention_layout_ok=1`. In the migrated BLIP2-T5 implementation, ATV uses
the attention mask to count and accumulate valid text tokens only; padding tokens
are audited but do not determine `k` or `scaler_row`. If pruning is run with
`batch_size > 1`, each physical dataset sample must still have its own
`sample_id` row with `num_query_tokens=32`; a row with `batch_size * 32` query
tokens is invalid evidence.

The selected-query evidence is checked separately in
`selected_query_frequency.csv`. Every selected `query_index` must be in `[0,31]`
with a positive count, and a plot alone is not accepted as strict evidence. When
`token_mask_integrity.csv` is available, the validator also checks the exact
`(layer, sample_id)` consistency: every row with `selected_k>0` must have the
same number of selected-query entries, while a row with `selected_k=0` must not
silently require a query-index row. Full layer coverage is proven by the alpha
logs and `token_mask_integrity.csv`; a layer with `selected_k=0` is valid and
naturally has no selected query-index row.

`calibration_batch_trace.csv` is the batch-level companion to
`token_mask_integrity.csv`. It records each cached batch's size, cumulative
physical sample count, sequence length, query-token count, valid text-token
range, and padding range before layer-wise pruning begins. Use it to confirm
that the report's `NUM_DATA=128` claim corresponds to at least 128 real
calibration samples, not merely 128 cached tensors or flattened tokens.

For ATV `scaler_row`, the normalization unit is also the physical sample, matching
the original implementation. The selected valid-text/query token energies are
summed within each sample and then averaged over samples; they are not averaged
over the number of kept tokens.
