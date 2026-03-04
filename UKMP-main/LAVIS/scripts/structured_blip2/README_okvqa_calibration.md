# UKMP 用 OK-VQA 做 calibration 剪枝

已在 UKMP 的 LAVIS 里加入与 ECoFLaP 相同的 OK-VQA calibration 支持。**路径**在 `calibration.yaml` 里改，**cfg** 用带标注的 `cc595k_prefix_derivative_compute_okvqa.yaml`。

## 文件名标注（与 ECoFLaP 同样备注）

| 用途 | 配置（prune） |
|------|------------------|
| **原 / CC595k calibration** | `cc595k_prefix_derivative_compute_cc595k.yaml` |
| **新 / OK-VQA calibration** | `cc595k_prefix_derivative_compute_okvqa.yaml` |

路径配置：`lavis/configs/datasets/okvqa/calibration.yaml`（改此处即可）。

---

## 1. 改路径（必做）

编辑 **`lavis/configs/datasets/okvqa/calibration.yaml`**：

- **annotations.train.url / storage**：指向你的 LAVIS 格式 `okvqa_train.json`（与 ECoFLaP 用同路径即可）。
- **images.storage**：指向 COCO 图像根目录（含 `train2014/`、`val2014/`）。

示例（按你本机改）：

```yaml
annotations:
  train:
    url:
        - /你的路径/datasets/okvqa/annotations/okvqa_train.json
    storage:
        - /你的路径/datasets/okvqa/annotations/okvqa_train.json
images:
  storage: /你的路径/datasets/okvqa_official/images
```

某一 knowledge category 时，换成该类别的 `okvqa_train.json`，例如：
`/你的路径/datasets/okvqa_by_category/Science_and_Technology/okvqa_train.json`。

---

## 2. 跑剪枝（用 OK-VQA calibration 时改 cfg）

**cfg** 用 **`cc595k_prefix_derivative_compute_okvqa.yaml`**（新配置 / 当前使用）：

```bash
cd /root/autodl-tmp/UKMP-main/LAVIS

CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.run \
  --nproc_per_node=1 --master_port=18082 ukmp_prune.py \
  --cfg-path lavis/projects/blip2/prune/cc595k_prefix_derivative_compute_okvqa.yaml \
  --device cuda \
  --save_ckpt_log_name ukmp_prune \
  --job_id "okvqa-xxx" \
  --pruning_ratio 0.5 \
  --granularity block \
  --pruner_type taylor+knowledge \
  --taylor param_first \
  --num_examples 1000 \
  --channel_per_step 1000 \
  --global_pruning \
  --imp_normalizer param \
  --select_loss \
  --entropy_importance \
  --multimodal
```

或改 **`ukmp_0.5.py`** 第 28 行：

```python
# 原 CC595k
# f" --cfg-path lavis/projects/blip2/prune/cc595k_prefix_derivative_compute.yaml"
# 改为 OK-VQA calibration（新）
f" --cfg-path lavis/projects/blip2/prune/cc595k_prefix_derivative_compute_okvqa.yaml"
```

并把 `job_id` 改成带 `okvqa` 前缀便于区分。

---

## 3. 新增/修改的文件（UKMP LAVIS 内）

- `lavis/datasets/datasets/okvqa_calibration_dataset.py`：OK-VQA → (image, text_input, text_output)。
- `lavis/datasets/builders/vqa_builder.py`：注册 `prefix_okvqa_calibration`。
- **`lavis/configs/datasets/okvqa/calibration.yaml`**：**改这里的路径**（与 ECoFLaP 同路径即可）。
- **`lavis/projects/blip2/prune/cc595k_prefix_derivative_compute_okvqa.yaml`**：【新】OK-VQA calibration 剪枝用。
- **`lavis/projects/blip2/prune/cc595k_prefix_derivative_compute_cc595k.yaml`**：【原】CC595k calibration 剪枝用。
- 原 `cc595k_prefix_derivative_compute.yaml`、`okvqa_prefix_derivative_compute.yaml` 顶部已加备注，指向上述带标注副本。

全精度模型仍用你之前下好的缓存，不会重复下载。
