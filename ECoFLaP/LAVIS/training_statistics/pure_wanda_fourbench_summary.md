# 纯 Wanda 四 calibration × 四基准 汇总

- repo: `/root/autodl-tmp/ECoFLaP/LAVIS`
- job 前缀: `pure_wanda_calib_<calib>`（与 pruned_checkpoint 文件名一致）
- metrics jsonl: `/root/autodl-tmp/ECoFLaP/LAVIS/training_statistics/pure_wanda_fourbench_metrics.jsonl`（MMBench / MMMU / MathVista）
- OKVQA: `/root/autodl-tmp/ECoFLaP/LAVIS/lavis/output/BLIP2/OKVQA/okvqa_eval_<job_id>/evaluate.txt`

| calibration 来源 | stem (ckpt) | MMBench % | MMMU % | OKVQA agg | MathVista % |
|---|---|--:|--:|--:|--:|
| mmbench | `pure_wanda_calib_mmbench` | — | — | 34.95 | — |
| mmmu | `pure_wanda_calib_mmmu` | — | — | 33.32 | — |
| okvqa | `pure_wanda_calib_okvqa` | — | — | 35.68 | — |
| mathvista | `pure_wanda_calib_mathvista` | — | — | 33.43 | — |

## 说明

- 若 MMBench/MMMU/MathVista 为 「—」，检查是否在本次评测前启用了 `LAVIS_METRICS_JSONL`（`run_pure_wanda_fourcalib_prune_then_fourbench_each.sh` 会默认追加到 `training_statistics/pure_wanda_fourbench_metrics.jsonl`）。旧跑次可重跑四 eval 或手动补记。
- OKVQA 为 `evaluate_blip` 全量 val 的 `agg_metrics`。
- 权重路径：`pruned_checkpoint/<stem>.pth`。

