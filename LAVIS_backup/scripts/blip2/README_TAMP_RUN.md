# 在 LAVIS_backup 下用 TAMP 方式跑 BLIP-2 剪枝（Wanda + DAS + AMIA）

## 0. Conda 环境（首次使用）

在 **LAVIS_backup** 根目录下创建并激活环境，再安装项目依赖：

```bash
cd /root/autodl-tmp/LAVIS_backup
conda env create -f environment.yml
conda activate lavis_backup
pip install -e .
```

或使用脚本一键完成（在 LAVIS_backup 根目录执行）：

```bash
bash scripts/conda_env_create.sh
```

- `environment.yml` 使用 Python 3.8、PyTorch 1.13 + CUDA 11.7；若只需 CPU，编辑 `environment.yml` 注释掉 cuda 相关行并取消 CPU 版 pytorch 注释。
- 国内建议设置：`export HF_ENDPOINT=https://hf-mirror.com`。

---

## 1. 环境与目录

- 在 **LAVIS_backup** 根目录下执行（不是 ECoFLaP/LAVIS）。
- 需安装 LAVIS 依赖（torch、transformers、datasets 等）；国内可设 `HF_ENDPOINT=https://hf-mirror.com`。

```bash
cd /root/autodl-tmp/LAVIS_backup
export HF_ENDPOINT=https://hf-mirror.com   # 可选，国内镜像
```

---

## 2. 一键 TAMP 剪枝（推荐）

使用 pruner 别名 `blipt5_tamp_pruner`，会自动使用：
- `token_selection=amia`
- `score_method=density_sum`
- `sparsity_ratio_granularity=layer`

**Calibration 数据集**：参考 **ECoFLaP 的 CF 类 OK-VQA calibration** 做法：
- 使用数据集 `prefix_okvqa_calibration`，配置 `cc_prefix_derivative_compute_okvqa.yaml`。
- 数据路径在 **`lavis/configs/datasets/okvqa/calibration.yaml`**：默认 **CF（Cooking_and_Food）**，即 `okvqa_by_category/Cooking_and_Food/okvqa_train.json`；要换 category（如 GHLC）只需改该文件里 `annotations.train.storage/url`。
- 若要用 CC3M 校准，请把 `--cfg-path` 改为 `lavis/projects/blip2/eval/cc_prefix_derivative_compute.yaml`。

**单卡示例**（0.5 稀疏率，保存剪枝后权重）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
  --nproc_per_node=1 --master_port 29500 evaluate_blip.py \
  --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa.yaml \
  --pruning_method blipt5_tamp_pruner \
  --save_pruned_model \
  --t5_prune_spec 24-0.5-1.0-1.0 \
  --vit_prune_spec 39-0.5-1.0-1.0 \
  --job_id okvqa_cf_0.5
```

**用脚本跑**（与现有 wanda 脚本一致，仅改 method）：

```bash
# 用法: python scripts/blip2/tamp_wanda.py <GPU_ID> <MASTER_PORT>
python scripts/blip2/tamp_wanda.py 0 29500
```

---

## 3. 显式指定 TAMP 参数（不用别名）

等价于上面，但不用 `blipt5_tamp_pruner`：

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
  --nproc_per_node=1 --master_port 29500 evaluate_blip.py \
  --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa.yaml \
  --pruning_method blipt5_wanda_pruner \
  --token_selection amia \
  --score_method density_sum \
  --sparsity_ratio_granularity layer \
  --save_pruned_model \
  --t5_prune_spec 24-0.5-1.0-1.0 \
  --vit_prune_spec 39-0.5-1.0-1.0 \
  --job_id tamp_manual_0.5
```

---

## 4. 参数说明

| 参数 | 含义 | TAMP 推荐 |
|------|------|-----------|
| `--cfg-path` | 任务/模型配置（校准用数据集、模型结构） | `cc_prefix_derivative_compute_okvqa.yaml`（OK-VQA CF 类，路径在 `configs/datasets/okvqa/calibration.yaml`）；或 `cc_prefix_derivative_compute.yaml`（CC3M） |
| `--pruning_method` | 剪枝器 | `blipt5_tamp_pruner` 或 `blipt5_wanda_pruner` |
| `--token_selection` | naive / amia | amia（TAMP） |
| `--score_method` | 重要性度量 | density_sum（DAS） |
| `--sparsity_ratio_granularity` | 稀疏率粒度 | layer（DAS 按层） |
| `--t5_prune_spec` | T5 剪枝规格 | 如 `24-0.5-1.0-1.0`（24 层，0.5 保留率） |
| `--vit_prune_spec` | ViT 剪枝规格 | 如 `39-0.5-1.0-1.0` |
| `--save_pruned_model` | 是否保存剪枝后权重 | 按需 |
| `--job_id` | 本次任务 ID（保存文件名用） | 任意字符串 |

---

## 5. OK-VQA 用哪个 category（CF / GHLC / 全量）

唯一决定位置：**`lavis/configs/datasets/okvqa/calibration.yaml`** 里的 `build_info.annotations.train.storage`（以及同路径的 `url`）。

| 用途 | train.storage / url 填的路径（示例） |
|------|--------------------------------------|
| **CF（Cooking_and_Food）**（默认） | `.../okvqa_by_category/Cooking_and_Food/okvqa_train.json` |
| GHLC | `.../okvqa_by_category/Geography_History_Language_and_Culture/okvqa_train.json` |
| 全量 | `.../okvqa/annotations/okvqa_train.json` |

图片根目录在同一文件里 `build_info.images.storage`（如 `.../okvqa_official/images`），按本机路径修改即可。

---

## 6. 输出

- 使用 `--save_pruned_model` 时，权重会保存到 `pruned_checkpoint/<job_id>.pth`。
- 稀疏率字典会保存到 `sparsity_dict/<job_id>.yaml`。

---

## 6.1 用 CF-calibration 剪枝模型按 OK-VQA 十一类分别跑 eval

剪枝完成后，用同一份 `.pth` 对 OK-VQA 的 **11 个 category** 分别跑一次 eval，看每个类上的效果（借鉴 ECoFLaP 形式）。

**前提**：已有 CF-calibration 剪枝好的模型，例如 `pruned_checkpoint/okvqa_cf_0.5-1.0-1.0.pth`。

**用法**（在 LAVIS_backup 根目录执行）：

```bash
# 默认：GPU 0，port 29501，ckpt=pruned_checkpoint/okvqa_cf_0.5-1.0-1.0.pth，data_root=/root/autodl-tmp/datasets
python scripts/blip2/eval_okvqa_by_category.py 0 29501

# 指定数据根和 checkpoint
python scripts/blip2/eval_okvqa_by_category.py 0 29501 --data_root /root/autodl-tmp/datasets --ckpt pruned_checkpoint/okvqa_cf_0.5-1.0-1.0.pth
```

- 脚本会依次对 11 个 category 各跑一次 `evaluate_blip.py`（仅 OK-VQA eval，加载上述剪枝权重）。
- 每个 category 的 eval 配置会写到 `lavis/projects/blip2/eval/okvqa_per_category/okvqa_zeroshot_flant5xl_eval_<Category>.yaml`，数据为该类下的 `okvqa_by_category/<Category>/okvqa_val_eval.json`。
- 11 类默认列表见脚本内 `OK_VQA_CATEGORIES`（Cooking_and_Food、Geography_History_Language_and_Culture、Science_and_Technology 等）；若你只有 10 类或目录名不同，可用 `--categories` 覆盖，例如：`--categories Cat1 Cat2 ...`。
- 结果与日志在 `output/BLIP2/OKVQA` 或终端输出，按 job_id 区分每个 category。

---

## 7. 默认行为（非 TAMP）

不传 TAMP 相关参数时，与改动前一致（naive + 原 score_method）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
  --nproc_per_node=1 --master_port 29500 evaluate_blip.py \
  --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute.yaml \
  --pruning_method blipt5_wanda_pruner \
  --save_pruned_model \
  --t5_prune_spec 24-0.5-1.0-1.0 \
  --vit_prune_spec 39-0.5-1.0-1.0 \
  --job_id naive_0.5
```
