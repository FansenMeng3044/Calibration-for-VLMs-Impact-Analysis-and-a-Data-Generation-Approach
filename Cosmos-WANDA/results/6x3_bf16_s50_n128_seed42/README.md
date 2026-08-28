# Cosmos3-Edge WANDA 6x3 evaluation results

All 29 queue tasks completed successfully. Six pruning checkpoints passed the
checkpoint validator, and all 21 evaluations passed exact sample-count and
result-file validation.

## Protocol

- Model: Cosmos3-Edge Reasoner only; Generator excluded.
- Precision: unquantized BF16.
- Calibration: 128 examples per source, seed 42.
- Target sparsity: 50% per output row for every target Linear in the 27-layer
  vision encoder and the 28-layer autoregressive language tower.
- Image processing: minimum 65,536 pixels and maximum 1,048,576 pixels.
- Evaluation counts: MMBench 4,329; MMMU 900; OK-VQA 5,046.
- Joint: language-tower WANDA statistics use the fused image+text sequence
  after the already-pruned vision encoder.
- Separate: vision statistics are image-only; language-tower statistics are
  tokenizer/embedding text-only, with vision and projector calls forbidden.

Images and text are calibration inputs only. The implementation prunes
`nn.Linear.weight`; it does not prune tokens, image patches, or activations.

## Scores

All values below are percentages. The parenthesized value is the absolute
percentage-point change relative to the dense BF16 baseline. `Macro avg.` is a
simple unweighted average for convenient comparison, not an official metric.

| Protocol | Calibration | MMBench | MMMU | OK-VQA | Macro avg. |
|---|---|---:|---:|---:|---:|
| Dense | None | **74.40** | **37.56** | **47.73** | **53.23** |
| Joint | MMBench | 56.79 (-17.61) | **30.56 (-7.00)** | 35.12 (-12.61) | **40.82** |
| Joint | MMMU | 48.80 (-25.60) | 28.44 (-9.12) | 30.00 (-17.73) | 35.75 |
| Joint | OK-VQA | 47.34 (-27.06) | 26.78 (-10.78) | **35.94 (-11.80)** | 36.68 |
| Separate | MMBench | **57.04 (-17.35)** | 27.56 (-10.00) | 33.10 (-14.63) | **39.23** |
| Separate | MMMU | 48.71 (-25.69) | 28.78 (-8.78) | 28.90 (-18.83) | 35.46 |
| Separate | OK-VQA | 49.83 (-24.57) | 27.56 (-10.00) | 35.27 (-12.47) | 37.55 |

The highest pruned macro average is Joint calibrated on MMBench (40.82). The
best pruned result per benchmark is Separate/MMBench for MMBench (57.04),
Joint/MMBench for MMMU (30.56), and Joint/OK-VQA for OK-VQA (35.94).

## Files

- `summary/`: the 21-row aggregate CSV and JSON.
- `eval/<model>/<benchmark>/`: validation manifest, raw lmms-eval result JSON,
  complete per-sample JSONL, and any generated submission/summary files.
- `pruning/<protocol>/<calibration>/`: pruning metadata, state, and checkpoint
  validation report. Model weight shards are not stored in Git.
- `controller/`: frozen experiment manifest, code SHA256 values, calibration
  alignment audit, dependency freeze, and queue summary.
- `.matrix_complete`: aggregate completion marker containing the validated
  result count and summary paths from the original server run.

The original server run root was:

```text
/private/workspace/hycui/Results/mfs/cosmos_wanda_full_matrix_20260828_bf16_s50_n128_seed42_v2_imagecap1m
```
