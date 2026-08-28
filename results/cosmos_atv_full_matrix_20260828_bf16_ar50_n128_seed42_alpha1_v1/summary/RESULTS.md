# Cosmos3-Edge ATV full-matrix results

Run root: `/private/workspace/hycui/Results/mfs/cosmos_atv_full_matrix_20260828_bf16_ar50_n128_seed42_alpha1_v1`

Protocol: Joint uses official multimodal visual-cosine ATV; separate uses text-only zero-visual ablation. Vision/projector remain dense.

All values below are percentages. Parentheses show the percentage-point difference from the dense baseline.

| Model | Protocol | Calibration | MMBench | MMMU | OK-VQA |
|---|---|---|---:|---:|---:|
| dense | dense | none | 74.40 | 37.56 | 47.73 |
| joint_mmbench | joint | mmbench | 65.98 (-8.42) | 32.11 (-5.45) | 41.55 (-6.19) |
| joint_mmmu | joint | mmmu | 61.77 (-12.63) | 31.89 (-5.67) | 37.47 (-10.27) |
| joint_okvqa | joint | okvqa | 58.76 (-15.64) | 30.44 (-7.12) | 41.08 (-6.65) |
| separate_mmbench | separate | mmbench | 64.09 (-10.31) | 28.67 (-8.89) | 39.24 (-8.49) |
| separate_mmmu | separate | mmmu | 60.65 (-13.75) | 29.11 (-8.45) | 36.77 (-10.96) |
| separate_okvqa | separate | okvqa | 55.93 (-18.47) | 30.11 (-7.45) | 38.51 (-9.23) |

## Validation

- Scheduler tasks: 29/29 complete; no failed or blocked tasks.
- Formal eval validations: 21/21 valid.
- Pruned checkpoint validations: 6/6 valid.
- Formal sample counts: MMBench 4329, MMMU 900, OK-VQA 5046.
- Checkpoints and model weights are intentionally excluded from Git.
