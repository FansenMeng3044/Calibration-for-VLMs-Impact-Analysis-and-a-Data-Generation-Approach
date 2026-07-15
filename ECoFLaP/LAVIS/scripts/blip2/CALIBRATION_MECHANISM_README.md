# Part 2 — calibration mechanism exploration

Goal: find the *real* reason different calibration datasets produce a **global
main effect** on downstream accuracy (some sets good on every benchmark, some
bad on every benchmark) instead of a **task-matching effect** (calibrate on task
X → best on task X). The accuracy table shows OKVQA-calibration wins every
column while the diagonal does not — this suite tests *why*.

The causal chain under investigation:

```
calibration data → activations → Wanda statistic (scaler_row) → mask
                → pruned weights → eval activations → accuracy
```

## Scripts

| Stage | Script | Cost | Produces |
|------|--------|------|----------|
| A | `extract_wanda_statistics.py` | GPU, 1 forward/dataset | `wanda_statistics.npz` + `meta.json` per dataset |
| B | `analyze_calibration_statistics.py` | CPU | statistic geometry / structure / accuracy-link CSVs |
| C | `analyze_calibration_mask_mechanism.py` | CPU, reads pruned `.pth` | mask-space CSVs |
| — | `run_calibration_mechanism_suite.sh` | orchestrates A→B→C | everything |

Run everything:

```bash
DATASETS=$'MMBench|/p/mmbench_calib.json|/p/mmbench_images
MMMU|/p/mmmu_calib.json|/p/mmmu_images
OKVQA|/p/okvqa_calib.json|/p/okvqa_images
mathvista|/p/mathvista_calib.json|/p/mathvista_images
cc3m|/p/cc3m_calib_128.json|/p/cc3m_images' \
ACC_CSV=/p/accuracy_matrix.csv \
JOINT_CKPTS="MMBench=/p/joint_mmbench.pth,MMMU=/p/joint_mmmu.pth,OKVQA=/p/joint_okvqa.pth,mathvista=/p/joint_mathvista.pth,cc3m=/p/joint_cc3m.pth" \
bash scripts/blip2/run_calibration_mechanism_suite.sh
```

`accuracy_matrix.csv` (rows = calibration, cols = eval):

```
calib,MMBench,MMMU,OKVQA,mathvista
MMBench,52.53,25.29,33.33,34.95
MMMU,50.38,25.1,35,33.32
OKVQA,52.97,25.57,35.37,35.68
mathvista,52.67,25.27,35.27,33.43
cc3m,50.13,24.6,34.77,34.07
```

## Outputs → what to plot → what it decides

Everything is CSV so you can visualize however you like. Per component: `t5`
statistics come in three token-group flavors (`t5_all`, `t5_text`, `t5_visual`);
`vit` is `all`.

| CSV | Plot | Question it answers |
|-----|------|---------------------|
| `similarity_<comp>.csv` | N×N heatmap | Are the statistics/masks nearly identical (weak leverage) or spread out? |
| `centrality_<comp>.csv` | bar chart | Which datasets are central vs outlier? (predict OKVQA central, cc3m outlier) |
| `mds_<comp>.csv` | 2D scatter (mds_1, mds_2), color = centrality | The geometry: one tight cluster + outliers? |
| `per_block_disagreement_<comp>.csv` | line vs block | Where in the net (which layers, ViT vs T5) do calibrations diverge? |
| `structure_<comp>.csv` | bar / scatter vs centrality | WHY central/outlier: RMS scale, kurtosis (tail), top-1% channel energy |
| `accuracy_link.csv` | scatter: statistic_similarity vs accuracy, color by is_diagonal | Does matching the eval distribution help (diagonal), or being central? |
| `centrality_vs_accuracy` (in summary) | scatter centrality vs row effect | **The linchpin**: does a central statistic predict global accuracy? |
| Stage C `overlap_*.csv`, `mask_space_*.csv` | heatmap + MDS | Same story at the mask level, from the checkpoints that actually made the numbers |

## How to read the result

- **`similarity`/`overlap` near 1.0 everywhere** → calibration has weak leverage
  (|W| dominates Wanda, as in Part 1). Then task-matching *cannot* exist and the
  only thing left is a coarse main effect. This is the expected precondition.
- **`centrality_vs_accuracy` slopes up (r > 0.5)** → representativeness is the
  cause: the most central (representative) calibration prunes best on every
  benchmark. The best calibration is not the task-matched one, it is the central
  one.
- **`accuracy_link` off-diagonal slope ≈ 0 and diagonal not high-accuracy** →
  matching calibration to the eval task does nothing; the diagonal is a red
  herring.
- **`structure_*` central vs outlier** → the concrete property: e.g. the outlier
  (cc3m) has a spikier / heavier-tailed / out-of-scale channel statistic. This is
  the mechanistic "real reason" in one number.
- **If everything is near 0 / flat** → the cause is NOT in the calibration
  statistic geometry. Next step: go downstream — run each pruned checkpoint on
  eval data and measure output drift (reuse `analyze_pruned_drift_by_token_group.py`),
  i.e. the effect is in how the pruned model behaves, not in which weights it cut.

## Which token group for T5?

`group=all` reproduces exactly what the pruner saw. `group=text` isolates the
language pathway (Part 1 showed the split/joint difference lived in the text
positions). Comparing `centrality_t5_all` vs `centrality_t5_text` tells you
whether the calibration main effect is driven by the visual prefix or the text —
a direct bridge from Part 1 to Part 2.
