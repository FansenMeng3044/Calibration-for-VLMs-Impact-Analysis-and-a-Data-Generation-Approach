# LLM Input Embedding Similarity Experiment

This experiment separates two questions that are easy to mix up:

1. Data semantic similarity: is a calibration set close to the final evaluation
   set under the dense BLIP2-T5 representation?
2. Pruned-model fidelity: after pruning with a given calibration set, does the
   pruned model preserve the dense model's embedding that enters the T5 LLM
   during evaluation?

For BLIP2-T5, the LLM input embedding is:

```text
concat([t5_proj(Q-Former(image)), T5 embed_tokens(text)])
```

The first part is saved as `visual_prefix`; the second part is saved as
`text_embed`.

## Main Design

Use the dense model for semantic comparison:

- Run dense BLIP2-T5 on every calibration candidate dataset.
- Run dense BLIP2-T5 on the final evaluation/reference datasets.
- Compare dataset-level embeddings with a calibration-by-eval heatmap.

Use dense and pruned models for fidelity comparison:

- Pick one evaluation dataset and fix the exact JSON/parquet rows and image
  order.
- Extract dense LLM-input embeddings on those rows.
- For every calibration-pruned checkpoint, extract embeddings on the exact same
  eval rows.
- Compare each pruned checkpoint to dense with paired cosine similarity.

## Scripts

- `extract_llm_input_embeddings.py`
  - writes `llm_input_embeddings.npz`
  - keys: `visual_prefix`, `text_embed`, `sample_index`
  - supports `--input_mode multimodal` and `--input_mode text_only`

- `analyze_llm_embeddings.py --mode semantic`
  - writes `calib_eval_semantic_similarity_<part>.csv/png`
  - writes `semantic_similarity_<part>.csv/png`
  - writes `mean_semantic_similarity_to_evals.csv`

- `analyze_llm_embeddings.py --mode fidelity`
  - writes `llm_input_fidelity_<part>.csv/png`
  - optionally writes `fidelity_vs_accuracy.png`

## Semantic Similarity Metrics

The semantic CSV contains:

- `centroid_cosine`: cosine between dataset mean embeddings.
- `mean_pairwise_cosine`: average all-pairs sample cosine.
- `mean_max_calib_to_eval`: for each calibration sample, nearest eval sample
  similarity, then averaged.
- `mean_max_eval_to_calib`: for each eval sample, nearest calibration sample
  similarity, then averaged.

Use `--part both` for a multimodal semantic view. Use `--part text` when C4 or
other text-only calibration is included. `--part visual` isolates image-to-LLM
handoff similarity.

## Example A: Dense Semantic Similarity

```bash
cd /data/data2/mfs/2/ECoFLaP/LAVIS
export PYTHONPATH=$PWD:${PYTHONPATH:-}

OUT=/data/data2/mfs/llm_embedding_similarity
CKPT=/data/data2/mfs/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth

python scripts/blip2/extract_llm_input_embeddings.py \
  --label calib_CC3M \
  --input_mode multimodal \
  --calib_json /data/data2/mfs/CC3M_calib_128/cc3m_calib_128.json \
  --images_dir /data/data2/mfs/CC3M_calib_128/images \
  --ckpt "$CKPT" \
  --out_dir "$OUT/dense/calib_CC3M" \
  --text_field caption \
  --max_samples 128 --batch_size 8

python scripts/blip2/extract_llm_input_embeddings.py \
  --label eval_MMBench \
  --input_mode multimodal \
  --calib_json /data/data2/mfs/MMBench_eval/mmbench_dev.parquet \
  --images_dir /data/data2/mfs/MMBench_eval/images \
  --ckpt "$CKPT" \
  --out_dir "$OUT/dense/eval_MMBench" \
  --text_field question \
  --max_samples 512 --batch_size 8

python scripts/blip2/analyze_llm_embeddings.py \
  --mode semantic \
  --part both \
  --emb calib_CC3M="$OUT/dense/calib_CC3M" \
  --emb eval_MMBench="$OUT/dense/eval_MMBench" \
  --calibs calib_CC3M \
  --evals eval_MMBench \
  --out_dir "$OUT/semantic_both"
```

Add more `--emb calib_...=...` and `--emb eval_...=...` entries to get a full
calibration-by-eval heatmap.

## Example B: Pruned-vs-Dense Fidelity on One Eval Set

```bash
python scripts/blip2/extract_llm_input_embeddings.py \
  --label dense_eval_MMBench \
  --input_mode multimodal \
  --calib_json /data/data2/mfs/MMBench_eval/mmbench_dev.parquet \
  --images_dir /data/data2/mfs/MMBench_eval/images \
  --ckpt "$CKPT" \
  --out_dir "$OUT/fidelity/MMBench/dense" \
  --text_field question \
  --max_samples 512 --batch_size 8

python scripts/blip2/extract_llm_input_embeddings.py \
  --label pruned_CC3M_on_MMBench \
  --input_mode multimodal \
  --calib_json /data/data2/mfs/MMBench_eval/mmbench_dev.parquet \
  --images_dir /data/data2/mfs/MMBench_eval/images \
  --ckpt /data/data2/mfs/2/ECoFLaP/LAVIS/pruned_checkpoint/cc3m_pruned.pth \
  --out_dir "$OUT/fidelity/MMBench/CC3M" \
  --text_field question \
  --max_samples 512 --batch_size 8

python scripts/blip2/analyze_llm_embeddings.py \
  --mode fidelity \
  --part visual \
  --dense "$OUT/fidelity/MMBench/dense" \
  --emb CC3M="$OUT/fidelity/MMBench/CC3M" \
  --out_dir "$OUT/fidelity/MMBench/visual"
```

Repeat the pruned extraction for each calibration-pruned checkpoint. The dense
and pruned runs must use the exact same eval rows and order.

## Example C: Five-Calib by Four-Eval Semantic Matrix and Fidelity

Use the wrapper when you want five calibration datasets to be compared against
four evaluation datasets under the dense model. The same wrapper can also
compare each calibration-pruned checkpoint against the dense model on all four
evaluation datasets.

```bash
cd /data/data2/mfs/2/ECoFLaP/LAVIS
export PYTHONPATH=$PWD:${PYTHONPATH:-}

export DENSE_CKPT=/data/data2/mfs/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth
export OUT_ROOT=/data/data2/mfs/llm_embedding_fidelity_fourbench

export CALIB_SPECS=$'MMBench|/path/mmbench_calib.json_or_parquet|/path/mmbench_images|question|multimodal|128
MMMU|/path/mmmu_calib.json|/path/mmmu_images|question|multimodal|128
OKVQA|/path/okvqa_calib.json|/path/okvqa_images|question|multimodal|128
MathVista|/path/mathvista_calib.json|/path/mathvista_images|question|multimodal|128
CC3M|/path/cc3m_calib.json|/path/cc3m_images|caption|multimodal|128'

export EVAL_SPECS=$'MMBench|/path/mmbench_eval.json_or.parquet|/path/mmbench_images|question|multimodal|512
MMMU|/path/mmmu_eval.json_or.parquet|/path/mmmu_images|question|multimodal|512
OKVQA|/path/okvqa_eval.json|/path/okvqa_images|question|multimodal|512
MathVista|/path/mathvista_eval.json|/path/mathvista_images|question|multimodal|512'

RUN_FIDELITY=0 SEMANTIC_PARTS="both text visual" BATCH_SIZE=8 \
  bash scripts/blip2/run_llm_embedding_fidelity_fourbench.sh
```

For semantic-only runs, the wrapper writes:

```text
$OUT_ROOT/semantic_dense/semantic_both/calib_eval_semantic_similarity_both.csv
$OUT_ROOT/semantic_dense/semantic_both/calib_eval_semantic_similarity_both.png
$OUT_ROOT/semantic_dense/semantic_text/calib_eval_semantic_similarity_text.csv
$OUT_ROOT/semantic_dense/semantic_visual/calib_eval_semantic_similarity_visual.csv
```

To also run pruned-vs-dense fidelity on the same four eval datasets, add the
five calibration-pruned checkpoints and set `RUN_FIDELITY=1`:

```bash
export PRUNED_CKPTS="MMBench=/path/pruned_by_mmbench.pth,MMMU=/path/pruned_by_mmmu.pth,OKVQA=/path/pruned_by_okvqa.pth,MathVista=/path/pruned_by_mathvista.pth,CC3M=/path/pruned_by_cc3m.pth"

RUN_FIDELITY=1 PARTS="visual both" SEMANTIC_PARTS="both text visual" BATCH_SIZE=8 \
  bash scripts/blip2/run_llm_embedding_fidelity_fourbench.sh
```

The fidelity branch writes one folder per eval dataset and a combined summary:

```text
$OUT_ROOT/MMBench/fidelity_visual/llm_input_fidelity_visual.csv
$OUT_ROOT/MMMU/fidelity_visual/llm_input_fidelity_visual.csv
$OUT_ROOT/OKVQA/fidelity_visual/llm_input_fidelity_visual.csv
$OUT_ROOT/MathVista/fidelity_visual/llm_input_fidelity_visual.csv
$OUT_ROOT/fidelity_summary_all_evals.csv
```

## Interpretation

- If semantic similarity is high but accuracy is not high, semantic task
  matching is probably not the main pruning mechanism.
- If pruned-vs-dense LLM-input fidelity is high and correlates with accuracy,
  preserving the image/text handoff into T5 is a plausible mechanism.
- If a checkpoint prunes only T5 layers, the LLM input embedding may be almost
  identical to dense. In that case, use downstream T5 encoder hidden-state or
  last-layer output similarity to see the pruning effect inside the LLM.
- For multimodal BLIP2-T5, `visual_prefix` means the 32 Q-Former query tokens
  after `t5_proj`; it is not raw ViT patch tokens.

## Example D: T5 Layer-Wise Hidden-State Fidelity

The LLM-input fidelity above only compares the embedding before T5 encoder
block 0.  To see how pruning drift accumulates inside the T5 encoder, use the
layer-wise wrapper:

```bash
cd /data/data2/mfs/2/ECoFLaP/LAVIS
export PYTHONPATH=$PWD:${PYTHONPATH:-}

export DENSE_CKPT=/data/data2/mfs/model_cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth
export OUT_ROOT=/data/data2/mfs/t5_layer_fidelity_fourbench

export PRUNED_CKPTS="MMBench=/path/pruned_by_mmbench.pth,MMMU=/path/pruned_by_mmmu.pth,OKVQA=/path/pruned_by_okvqa.pth,MathVista=/path/pruned_by_mathvista.pth,CC3M=/path/pruned_by_cc3m.pth"

export EVAL_SPECS=$'MMBench|/path/mmbench_eval.json_or.parquet|/path/mmbench_images|question|multimodal|512
MMMU|/path/mmmu_eval.json_or.parquet|/path/mmmu_images|question|multimodal|512
OKVQA|/path/okvqa_eval.json|/path/okvqa_images|question|multimodal|512
MathVista|/path/mathvista_eval.json|/path/mathvista_images|question|multimodal|512'

T5_LAYER_PARTS="both" BATCH_SIZE=4 \
  bash scripts/blip2/run_t5_layer_fidelity_fourbench.sh
```

This produces one main curve plot per eval dataset:

```text
$OUT_ROOT/MMBench/t5_layer_both/t5_layer_fidelity_both.png
$OUT_ROOT/MMMU/t5_layer_both/t5_layer_fidelity_both.png
$OUT_ROOT/OKVQA/t5_layer_both/t5_layer_fidelity_both.png
$OUT_ROOT/MathVista/t5_layer_both/t5_layer_fidelity_both.png
$OUT_ROOT/t5_layer_fidelity_summary_all_evals.csv
```

Each plot has five calibration curves.  The x-axis is the T5 encoder layer
after each encoder block, and the y-axis is mean cosine similarity between the
pruned model and dense model hidden states on the same eval rows.
