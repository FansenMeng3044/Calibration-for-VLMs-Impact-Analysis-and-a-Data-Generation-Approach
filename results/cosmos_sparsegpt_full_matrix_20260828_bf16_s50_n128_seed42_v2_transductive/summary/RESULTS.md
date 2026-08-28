# Cosmos3-Edge SparseGPT full-matrix results

Run root: `/private/workspace/hycui/Results/mfs/cosmos_sparsegpt_full_matrix_20260828_bf16_s50_n128_seed42_v2_transductive`

Protocol: Joint uses multimodal Reasoner calibration; separate uses the locked vision-local plus text-only AR protocol.

All values below are percentages. Parentheses show the percentage-point difference from the dense baseline.

| Model | Protocol | Calibration | MMBench | MMMU | OK-VQA |
|---|---|---|---:|---:|---:|
| dense | dense | none | 74.40 | 37.56 | 47.73 |
| joint_mmbench | joint | mmbench | 63.06 (-11.34) | 31.56 (-6.00) | 38.36 (-9.37) |
| joint_mmmu | joint | mmmu | 56.70 (-17.70) | 30.56 (-7.00) | 33.55 (-14.18) |
| joint_okvqa | joint | okvqa | 47.51 (-26.89) | 26.11 (-11.45) | 39.74 (-7.99) |
| separate_mmbench | separate | mmbench | 63.14 (-11.25) | 30.89 (-6.67) | 36.33 (-11.40) |
| separate_mmmu | separate | mmmu | 56.53 (-17.87) | 28.22 (-9.34) | 31.86 (-15.87) |
| separate_okvqa | separate | okvqa | 11.60 (-62.80) | 27.33 (-10.23) | 20.76 (-26.97) |

## Validation

- Scheduler tasks: 29/29 complete; no failed or blocked tasks.
- Formal eval validations: 21/21 valid.
- Pruned checkpoint validations: 6/6 valid.
- Formal sample counts: MMBench 4329, MMMU 900, OK-VQA 5046.
- Checkpoints and model weights are intentionally excluded from Git.
