# UKMP 剪枝（OK-VQA calibration）+ 复用 ECoFLaP 已下载的全精度模型

**前提**：在**同一台机器、同一用户、同一 conda 环境（ecoflap）**下跑。  
PyTorch / HuggingFace 的缓存目录（如 `~/.cache/torch/hub/`、`~/.cache/huggingface/`）会共用，**不会重复下载**全精度模型。

---

## 一条命令跑 UKMP 剪枝（OK-VQA calibration）

```bash
cd /root/autodl-tmp/UKMP-main/LAVIS

# 国内网络可加 HF 镜像（与 ECoFLaP 一致，避免再次请求外网）
export HF_ENDPOINT=https://hf-mirror.com

# 使用与 ECoFLaP 相同的 conda 环境，会复用已有模型缓存
conda activate ecoflap

CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.run \
  --nproc_per_node=1 --master_port=18083 \
  ukmp_prune.py \
  --cfg-path lavis/projects/blip2/prune/cc595k_prefix_derivative_compute_okvqa.yaml \
  --device cuda \
  --save_ckpt_log_name ukmp_prune \
  --job_id "okvqa-1000data-taylor+knowledge-param_first-param_norm-0.5-blockwise-global-select_loss-multimodal" \
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

- 端口 `18083` 若被占用可改成 `18084` 等。
- `--job_id` 可自定义，建议带 `okvqa` 前缀以便和 CC595k 剪枝区分。

---

## 为何不会重新下全精度模型

| 缓存内容 | 默认路径 | 说明 |
|----------|----------|------|
| eva_vit_g.pth、blip2_pretrained_flant5xl.pth 等 | `~/.cache/torch/hub/checkpoints/` | 与 ECoFLaP 共用 |
| bert-base-uncased、google/flan-t5-xl | `~/.cache/huggingface/hub/` | 与 ECoFLaP 共用 |

只要在同一用户、同一环境（如 ecoflap）下跑，上述路径一致，会直接读本地缓存，不会再次下载。若之前用 `HF_ENDPOINT=https://hf-mirror.com` 下过，建议继续加上该环境变量再跑 UKMP。

---

## 只跑剪枝、不跑后面的 eval/finetune

上面命令只执行 **ukmp_prune.py**，得到 `pruned_checkpoint/ukmp_prune/<job_id>/pytorch_model.bin` 等。  
若要像 `ukmp_0.5.py` 那样再跑 eval 和 finetune，可继续用该脚本或按需仿照其后续命令。

---

## 路径确认

OK-VQA 数据路径在 **`lavis/configs/datasets/okvqa/calibration.yaml`** 中。若与 ECoFLaP 用同一份数据，保持该文件里的路径与 ECoFLaP 一致即可（或已改为你的本机路径即可）。
