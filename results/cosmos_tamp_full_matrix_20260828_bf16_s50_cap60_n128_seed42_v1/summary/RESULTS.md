# Cosmos3-Edge TAMP full-matrix results

Run root: `/private/workspace/hycui/Results/mfs/cosmos_tamp_full_matrix_20260828_bf16_s50_cap60_n128_seed42_v1`

Protocol: TAMP combines DAS, AMIA, and WANDA. Joint uses fused multimodal AR calibration; separate uses text-only AR calibration. Only Reasoner AR/LLM linears are pruned; vision/projector remain dense.

All values below are percentages. Parentheses show the percentage-point difference from the dense baseline.

| Model | Protocol | Calibration | MMBench | MMMU | OK-VQA |
|---|---|---|---:|---:|---:|
| dense | dense | none | 74.40 | 37.56 | 47.73 |
| joint_mmbench | joint | mmbench | 55.15 (-19.24) | 31.00 (-6.56) | 28.05 (-19.68) |
| joint_mmmu | joint | mmmu | 52.66 (-21.74) | 30.44 (-7.12) | 31.46 (-16.27) |
| joint_okvqa | joint | okvqa | 49.31 (-25.09) | 28.67 (-8.89) | 37.03 (-10.70) |
| separate_mmbench | separate | mmbench | 0.60 (-73.80) | 26.11 (-11.45) | 1.67 (-46.06) |
| separate_mmmu | separate | mmmu | 36.34 (-38.06) | 25.67 (-11.89) | 19.69 (-28.05) |
| separate_okvqa | separate | okvqa | 22.85 (-51.55) | 24.44 (-13.12) | 19.96 (-27.77) |

## Validation

- Scheduler tasks: 29/29 complete; no failed or blocked tasks.
- Formal eval validations: 21/21 valid.
- Pruned checkpoint validations: 6/6 valid.
- Formal sample counts: MMBench 4329, MMMU 900, OK-VQA 5046.
- Checkpoints and model weights are intentionally excluded from Git.
