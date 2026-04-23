# 用 OK-VQA 做 calibration 得到剪枝模型：跑哪些代码、用的哪个 category

## 零、国内网络：HuggingFace 超时

若报错 `Connection to huggingface.co timed out`（下载 bert-base-uncased 等），先设镜像再跑：

```bash
export HF_ENDPOINT=https://hf-mirror.com
# 然后再执行下面的剪枝命令
```

或一行命令：`HF_ENDPOINT=https://hf-mirror.com python scripts/blip2/ecoflap_zeroth.py 0 29501`

---

## 一、要跑哪些代码才能得到「OK-VQA calibration 剪枝后的模型」

### 1. 只做剪枝（得到 .pth）

在 **ECoFLaP/LAVIS** 目录下执行（需先 `cd` 到该目录）：

```bash
cd /root/autodl-tmp/ECoFLaP/LAVIS

# GPU 和 port 按你的环境改，例如 GPU=0 port=29500
python scripts/blip2/ecoflap_zeroth.py <GPU> <port>
```

- 脚本会先跑 **剪枝**：用当前配置里的 OK-VQA calibration 数据，得到剪枝后的模型并保存。
- 保存路径：`pruned_checkpoint/okvqa-<...>.pth`（job_id 前缀为 `okvqa-`）。
- 之后脚本还会用这个 checkpoint 去跑 VQAv2、GQA、OK-VQA、NoCaps、Flickr 等评测；若你**只要剪枝模型**，拿到 `.pth` 后可以停掉或只保留第一次调用。

### 2. 若只想单独跑「剪枝」这一步（不跑后面评测）

可以不跑整段脚本，只执行和剪枝等价的一条命令（把 `<GPU>`、`<port>` 换成实际值）：

```bash
cd /root/autodl-tmp/ECoFLaP/LAVIS

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
  --nproc_per_node=1 --master_port 29500 evaluate_blip.py \
  --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute_okvqa.yaml \
  --pruning_method 'blipt5_wanda_pruner' --save_pruned_model \
  --score_method MEZO-GradOnly_sum \
  --sparsity_ratio_granularity block \
  --max_sparsity_per_layer 0.6 \
  --prunining_dataset_batch_size 8 \
  --t5_prune_spec "24-0.5-1.0-1.0" --vit_prune_spec "39-0.5-1.0-1.0" \
  --job_id "okvqa-blipt5_wanda_pruner_0.5-1.0-1.0_MEZO-GradOnly_sum0.6_block_bs8"
```

得到的剪枝模型同样在 `pruned_checkpoint/` 下，文件名里的 `job_id` 与上面一致。

---

## 二、OK-VQA 用的是「哪一个 category」从哪里看 / 怎么改

### 1. 从哪里看：唯一决定位置

**文件路径：**

`ECoFLaP/LAVIS/lavis/configs/datasets/okvqa/calibration.yaml`

里面 **`build_info.annotations.train.storage`** 的路径，决定 calibration 用的是「全量 OK-VQA」还是「某一个 category」。

### 2. 当前默认（你现在的配置）

```yaml
# calibration.yaml 里当前是：
build_info:
  annotations:
    train:
      storage:
          - /root/autodl-tmp/datasets/okvqa/annotations/okvqa_train.json
  images:
      storage: /root/autodl-tmp/datasets/okvqa_official/images
```

- **含义**：用的是 **全量 OK-VQA train**（所有 category 混在一起），不是某一个单独 category。
- 对应数据：`datasets/okvqa/annotations/okvqa_train.json`，即 LAVIS 格式的完整 train。

### 3. 若想改成「某一个 category」

把 **`annotations.train.storage`** 改成该类别的 `okvqa_train.json` 即可，**images 不用改**（仍用 `okvqa_official/images`）。

例如用 **Science_and_Technology** 这一类做 calibration：

```yaml
build_info:
  annotations:
    train:
      storage:
          - /root/autodl-tmp/datasets/okvqa_by_category/Science_and_Technology/okvqa_train.json
  images:
      storage: /root/autodl-tmp/datasets/okvqa_official/images
```

其他类别同理，只要把路径换成对应子目录下的 `okvqa_train.json`：

| category 目录名（示例） | calibration 里 train.storage 填的路径 |
|-------------------------|----------------------------------------|
| 全量（当前） | `/root/autodl-tmp/datasets/okvqa/annotations/okvqa_train.json` |
| Science_and_Technology | `/root/autodl-tmp/datasets/okvqa_by_category/Science_and_Technology/okvqa_train.json` |
| Vehicles_and_Transportation | `/root/autodl-tmp/datasets/okvqa_by_category/Vehicles_and_Transportation/okvqa_train.json` |
| … | `.../okvqa_by_category/<Category>/okvqa_train.json` |

类别名和目录一致，见：`datasets/okvqa_by_category/` 下各子文件夹名，或 `datasets/okvqa_by_category/configs/okvqa_<Category>.yaml` 里的名字。

### 4. 小结

- **要得到「用 OK-VQA 当 calibration 的剪枝模型」**：跑上面 **一** 里的脚本或单条剪枝命令。
- **当前用的是哪个 OK-VQA category**：看 **`lavis/configs/datasets/okvqa/calibration.yaml`** 里 **`annotations.train.storage`**；当前是**全量**，要单类别就改成对应 `okvqa_by_category/<Category>/okvqa_train.json`。
