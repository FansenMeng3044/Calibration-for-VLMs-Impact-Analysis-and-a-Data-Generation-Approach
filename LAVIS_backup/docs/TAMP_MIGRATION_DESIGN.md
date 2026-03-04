# TAMP 迁移到 LAVIS_backup（BLIP-2）方案

目标：把原版 TAMP 的 **Wanda + DAS + AMIA** 原封不动地迁移到 `/root/autodl-tmp/LAVIS_backup`，在 BLIP-2（Blip2T5）上复现 TAMP 的方法与思路。**本文档只做方案设计，不直接改代码**，后续按步骤由你指导逐步实现。

---

## 一、原版 TAMP 与 LAVIS_backup 现状对照

| 组件 | 原版 TAMP（LLaVA） | LAVIS_backup（BLIP-2） |
|------|---------------------|-------------------------|
| **Wanda** | LLaMALayerWandaPruner：\|W\|×√(激活 L2) 剪权重 | T5LayerWandaPruner、VITLayerWandaPruner、BLIPT5LayerWandaPruner 已有，但 **无** image_mask / token_selection |
| **DAS** | LayerSparsity + `score_method="density_sum"`，`compute_density()` 用 vision/language token 密度算层重要性 | LayerSparsity 仅有 GradMagSquare_avg、MEZO 等，**无** density_sum、**无** compute_density |
| **AMIA** | AdaptiveMultimodalInputActivation：按 density + 图/MMD 选 token，只用工选 token 的激活更新 scaler_row | **无**；WrappedGPT 用全部 token，**无** image_mask |
| **image_mask 来源** | LLaVA forward 里设 `model.temp_label`（哪些位置是 image token） | BLIP-2 forward **未**设 temp_label；encoder 前 num_query_token 为 vision |
| **校准数据** | prepare_calibration 抓 model.layers 输入，返回 inps, outs, caches, **image_masks** | prepare_calibration 只返回 inps, outs, caches，**无** image_masks |

结论：在 LAVIS_backup 上要**新增** DAS（density 分支 + compute_density）、AMIA（AdaptiveMultimodalInputActivation + 校准时 image_mask）、以及 BLIP-2 的 temp_label；并**扩展现有** Wanda 的校准与 _prune 接口以支持 token_selection 与 image_masks。

---

## 二、迁移步骤总览

| 步骤 | 内容 | 涉及文件 |
|------|------|----------|
| **Step 1** | BLIP-2 forward 里为 encoder 设置 temp_label（vision=前 num_query_token） | `lavis/models/blip2_models/blip2_t5.py` |
| **Step 2** | layer_single_base 增加 DAS：cos_pairwise_density、ActivationDensity、compute_density，LayerSparsity 支持 density_sum | `lavis/compression/pruners/layer_single_base_pruner.py` |
| **Step 3** | T5 校准返回 image_masks：prepare_calibration 在 BLIP-2 下收集 model.temp_label | `lavis/compression/pruners/wanda_pruner.py`（T5 部分） |
| **Step 4** | wanda_pruner 增加 AMIA：AdaptiveMultimodalInputActivation，T5 _prune 支持 token_selection 与 image_masks | `lavis/compression/pruners/wanda_pruner.py` |
| **Step 5** | BLIPT5 pruner 接入 DAS+AMIA：get_sparsity 用 density_sum，T5 校准时传 token_selection=amia | `lavis/compression/pruners/wanda_pruner.py`（BLIPT5LayerWandaPruner） |
| **Step 6** | 入口与配置：evaluate_blip 支持 tamp 相关参数，并注册/使用 blipt5_tamp_pruner（可选） | `evaluate_blip.py`、配置/脚本 |

下面按步骤写清「做什么、在哪做、原版 TAMP 对应逻辑」。

---

## 三、Step 1：Blip2T5 的 forward 里设置 temp_label

**目的**：让后续校准阶段能拿到「哪些位置是 vision token」的 mask，供 DAS 和 AMIA 使用。

**原版 TAMP**：LLaVA 在 `llava_arch.py` 的 prepare_inputs_labels_for_model 里设 `self.temp_label = image_token_masks`（与序列等长的 bool）。

**BLIP-2 对应**：T5 encoder 的输入是 `inputs_embeds = [inputs_t5; text_embeds]`，前 `num_query_token`（如 32）个位置是 vision，其余是 text。

**设计**：

- **文件**：`LAVIS_backup/lavis/models/blip2_models/blip2_t5.py`
- **位置**：在 `forward()` 里，在 `inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)` 之后、`self.t5_model(...)` 之前。
- **行为**：
  - `num_query = inputs_t5.shape[1]`（即 query_tokens 长度）。
  - `bsz, seq_len, _ = inputs_embeds.shape`。
  - 定义与 encoder 序列一致的 mask：前 `num_query` 为 True，其余为 False。
  - 设 `self.temp_label = ...`，形状 `(bsz, seq_len)`，device 与 `inputs_embeds` 一致，dtype=bool。
- **注意**：不改变原有 loss/forward 逻辑，仅增加一次赋值，保证每次 `model(batch)` 后 `model.temp_label` 可用。

---

## 四、Step 2：LayerSparsity 增加 DAS（density_sum + compute_density）

**目的**：按「多模态 token 密度」算每层重要性，得到每层稀疏率（DAS），与 TAMP 一致。

**原版 TAMP**：`layer_single_base_pruner.py` 中  
- `cos_pairwise_density(embeddings, image_mask)`：按 image_mask 分出 v/l，算 v-v、l-l、v-l 的余弦相似度（或距离）均值。  
- `ActivationDensity`：对某一层输出做 cos_pairwise_density，累积 v_density, l_density, vl_dist。  
- `compute_density(layer_to_group_mapping)`：对 `model.layers` 逐层 forward，用 ActivationDensity 收集每层输出的 density，再合成 `importance_measure`（例如 (1-v_density)+(1-l_density)+vl_dist 等），用于后续按层分配稀疏率。  
- `LayerSparsity.return_sparsity()` 里若 `score_compute.startswith("density")`，则调用 `compute_density`，不再用梯度。

**设计**：

- **文件**：`LAVIS_backup/lavis/compression/pruners/layer_single_base_pruner.py`
- **新增**：
  1. **函数** `cos_pairwise_density(embeddings, image_mask)`：与 TAMP 一致，输入 2D embeddings 和 1D bool image_mask，返回 v_mean_dist, l_mean_dist, vl_mean_dist（或 TAMP 里用的三个量）。
  2. **类** `ActivationDensity`：对一层输出 `out` 和 `image_mask` 调用 `cos_pairwise_density`，在 `add_batch` 里累积；接口与 TAMP 一致（入参 out, image_mask）。
  3. **函数** `prepare_calibration_input_encoder(...)`（或复用现有逻辑）：需要能对 **BLIP-2 的 T5 encoder** 跑校准并得到每层的 **image_masks**。这里先约定：由 Step 3 在 wanda_pruner 里实现「BLIP-2 的 T5 encoder 校准并返回 image_masks」，layer_single_base 的 `compute_density` 通过参数或依赖注入拿到「带 image_masks 的校准数据」或「能返回 inps, outs, caches, image_masks 的接口」。
  4. **方法** `LayerSparsity.compute_density(self, layer_to_group_mapping)`：  
     - 对 BLIP-2，layer_to_group_mapping 的 key 是 `t5_model.encoder.block.{i}.{name}.weight` 等；需要知道 `module_to_process = "t5_model.encoder.block"`（或由调用方传入）。  
     - 跑校准得到 inps, outs, caches, image_masks（来自 Step 3 的 T5 校准）。  
     - 逐层 forward，每层用 ActivationDensity + 对应 image_masks 收集 density，再转为每层一个 importance（与 TAMP 公式一致），写入 `self.importance_measure`。  
     - 返回的 importance_measure 会被后续 `return_sparsity` 里已有的「按 group 分配稀疏率」逻辑使用（group_scores 等）。
  5. **修改** `LayerSparsity.return_sparsity()`：在 `if len(self.importance_measure) == 0` 分支中，增加 `elif self.score_compute.startswith("density"): self.importance_measure = self.compute_density(layer_to_group_mapping)`；并保证 `score_aggregate` 对 density 仍为 `sum`（与 TAMP 的 density_sum 一致）。
- **依赖**：Step 3 完成后，T5 的 prepare_calibration 能返回 image_masks，供 compute_density 使用（或 Step 2 里先实现一个仅用于 encoder、且能拿到 model.temp_label 的校准流程）。

说明：TAMP 的 compute_density 是给 LLaVA 的 `model.layers` 用的；这里改为给 `t5_model.encoder.block` 用，层名与 layer_to_group_mapping 的 key 要一致（例如 `t5_model.encoder.block.0.layer.0.linear.weight` 等）。

---

## 五、Step 3：T5 校准返回 image_masks

**目的**：T5 encoder 校准时，每一批/每个样本的 vision/language mask 能传给 DAS 和 AMIA。

**原版 TAMP**：prepare_calibration_input_encoder 里，Catcher 在第一次 forward 时触发，在 `except ValueError` 里 `image_masks.append(model.temp_label)`，返回 inps, outs, caches, image_masks。

**设计**：

- **文件**：`LAVIS_backup/lavis/compression/pruners/wanda_pruner.py`（T5LayerWandaPruner 及其调用的校准逻辑）。
- **当前**：BLIP-2 的 forward 是 `model(batch)`，会先跑 visual_encoder → Q-Former → t5_proj，再拼成 encoder 输入并进入 T5 encoder；Catcher 挂在 `t5_model.encoder.block[0]` 上，第一次 forward 会执行到该层并 append inp，然后 raise ValueError。此时 model 已经执行过 forward 前半段，**model.temp_label 已由 Step 1 设好**。
- **修改**：
  - 在 T5 的 `prepare_calibration_input_encoder` 中，在 `except ValueError` 分支里，除了 append inps、caches 外，**append 当前 batch 的 model.temp_label**（即 `image_masks.append(model.temp_label)`，若 batch_size>1 则按样本数展开或按需存成 list）。
  - 返回值从 `(inps, outs, caches)` 改为 `(inps, outs, caches, image_masks)`。
  - 所有调用 `prepare_calibration_input_encoder` 的地方（T5 的 _prune、以及若 Step 2 的 compute_density 会间接调用）都要适配三元组改为四元组，并向下传递 image_masks。

这样 T5 encoder 的校准与 TAMP 的「带 image_masks 的校准」一致，为 Step 4/5 的 AMIA 和 Step 2 的 compute_density 提供输入。

---

## 六、Step 4：Wanda 的 T5 分支支持 AMIA（AdaptiveMultimodalInputActivation）

**目的**：校准时用 AMIA 选 token，只用工选 token 的激活更新 scaler_row，再用于 Wanda 的 |W|×√scaler_row 剪枝。

**原版 TAMP**：  
- `AdaptiveMultimodalInputActivation`：在 `add_batch(inp, out, image_mask, score)` 里用 `cos_pairwise_density(out, image_mask)` 得 density，用 distance + KNN + 图传播得 graph_score，再迭代选 token 直到 MMD 条件满足，得到 score_mask；最后 `inp = inp[score_mask]`，用这部分 inp 更新 scaler_row。  
- LLaVA 的 _prune 里，若 `token_selection == 'amia'` 则用 AdaptiveMultimodalInputActivation，否则 WrappedGPT；hook 里调用 `add_batch(..., image_masks[j], scores[j])`。  
- LLaVA 的 scores 来自 causal attention 最后一 query 的 attention 权重；T5 encoder 非 causal，可选：对 T5 不传 score（或传 None），AMIA 内对 score 做判空，无 score 时用均匀或仅 density 做选择（与 TAMP 思路一致即可）。

**设计**：

- **文件**：`LAVIS_backup/lavis/compression/pruners/wanda_pruner.py`
- **新增**：
  1. **类** `AdaptiveMultimodalInputActivation`：从 TAMP 的 `llava/pruners/wanda_pruner.py` 拷贝实现；`add_batch(self, inp, out, image_mask=None, score=None)`，当 `score is None` 时内部用均匀或仅 density 做 token 选择（保证 T5 encoder 可用）。
  2. **修改** T5 的 `_prune`（或 T5LayerWandaPruner._prune 调用的内部 _prune）：
     - 增加参数 `token_selection='naive'` 与 `image_masks=None`（以及可选 `scores=None`）。
     - 校准时若返回了 image_masks，则传入 _prune。
     - 对每一层：若 `token_selection == 'amia'` 则用 `AdaptiveMultimodalInputActivation`，否则用现有 `WrappedGPT`。
     - 在 register_forward_hook 的闭包里：`add_batch(inp[0].data, out.data, image_masks[j] if image_masks else None, scores[j] if scores else None)`。
  3. **WrappedGPT**：当前 LAVIS 的 `add_batch(self, inp, out)` 无 image_mask；为接口统一可改为 `add_batch(self, inp, out, image_mask=None, score=None)`，内部忽略这两个参数（保持原行为），这样与 TAMP 的 WrappedGPT 签名一致，便于共用。

这样 T5 encoder 的 Wanda 剪枝就支持「naive / amia」两种 token 选择，与 TAMP 一致。

---

## 七、Step 5：BLIPT5LayerWandaPruner 接入 DAS + AMIA

**目的**：BLIP-2 的入口 pruner 使用 density_sum 做层稀疏（DAS），T5 encoder 使用 amia 做 token 选择（AMIA）。

**原版 TAMP**：LLaVALayerWandaPruner 的 get_sparsity 里，若 sparsity_ratio_granularity 非空则用 LayerSparsity，score_method 用 `density_sum`；prune() 里对 LLM 调用 _prune 时传 `token_selection=self.token_selection`（amia/naive）。

**设计**：

- **文件**：`LAVIS_backup/lavis/compression/pruners/wanda_pruner.py` 中的 `BLIPT5LayerWandaPruner`。
- **修改**：
  1. **构造函数**：增加参数 `token_selection='naive'`、`score_method` 默认或显式支持 `'density_sum'`（若当前只有 GradMagSquare_avg 等，则增加 density_sum 选项）。
  2. **get_sparsity**：当调用方传入 `score_method='density_sum'` 时，传给 LayerSparsity 的 score_method 即为 `density_sum`；LayerSparsity 在 Step 2 已支持 density 分支，会调用 compute_density，从而得到按层/按组的稀疏率。
  3. **prune()**：在调用 T5 的 _prune 时，传入 `token_selection=self.token_selection` 和校准时得到的 `image_masks`（来自 Step 3 的 prepare_calibration 返回值）。
  4. **compute_density 的调用链**：get_sparsity → LayerSparsity.return_sparsity() → compute_density(layer_to_group_mapping)。compute_density 需要拿到「T5 encoder 的校准 inps, outs, caches, image_masks」——可通过在 LayerSparsity 构造时传入 data_loader 与 model，并在 compute_density 内调用与 T5 相同的 prepare_calibration 逻辑（或抽成公共函数），保证 BLIP-2 下只对 t5_model.encoder.block 跑、且使用 model.temp_label 作为 image_masks。
- **ViT / Decoder**：  
  - ViT：暂无 DAS/AMIA 需求时可保持现有 Wanda；若要做 DAS，再为 ViT 单独算一层 importance（或统一用 density，但 ViT 无 text token，需约定 image_mask 全 True）。  
  - T5 decoder：无 vision token，可保持 token_selection='naive'，image_masks 传全 False或 None。

这样 BLIP-2 的 T5 部分就完整接上 TAMP 的 DAS + AMIA；ViT 可后续再扩展。

---

## 八、Step 6：入口与配置

**目的**：能从命令行/配置里以「TAMP 方式」跑 BLIP-2 剪枝（Wanda + DAS + AMIA）。

**设计**：

- **文件**：`LAVIS_backup/evaluate_blip.py`（以及可选的项目配置/脚本）。
- **修改**：
  - 增加/支持参数：例如 `--token_selection amia`、`--score_method density_sum`、`--sparsity_ratio_granularity layer`（若尚未支持）。
  - 在 load_pruner 或构造 BLIPT5LayerWandaPruner 时，把这些参数传入；这样默认即可跑「TAMP 配置」。
- **可选**：新增一个 pruner 名如 `blipt5_tamp_pruner`，内部等价于 BLIPT5LayerWandaPruner 且固定 `token_selection='amia'`、`score_method='density_sum'`、`sparsity_ratio_granularity='layer'`，便于一键复现 TAMP。

---

## 九、依赖顺序与建议实现顺序

- **Step 1** 必须先做：否则没有 temp_label，Step 3/4/5 的 image_masks 无法得到。
- **Step 3** 建议紧接 Step 1：T5 校准返回 image_masks，不改 DAS/AMIA 逻辑也能先打通数据流。
- **Step 4** 依赖 Step 3** 的 image_masks**：在 T5 _prune 里接 AMIA 与 image_masks。
- **Step 2** 依赖 Step 3** 的校准+image_masks**（或 Step 2 内自己实现一遍 BLIP-2 的 encoder 校准+temp_label），用于 compute_density。
- **Step 5** 在 Step 2、4 都就绪后，把 DAS 和 AMIA 接到 BLIPT5LayerWandaPruner。
- **Step 6** 最后：入口参数与可选 blipt5_tamp_pruner。

建议你后续指导时的顺序：**1 → 3 → 4 → 2 → 5 → 6**（若你希望先做 AMIA 再做 DAS，也可 1→3→4→5，再补 2 和 6）。

---

## 十、文件改动汇总（不改代码，仅清单）

| 文件 | 改动类型 |
|------|----------|
| `lavis/models/blip2_models/blip2_t5.py` | 在 forward 中设置 `self.temp_label`（Step 1） |
| `lavis/compression/pruners/layer_single_base_pruner.py` | 新增 cos_pairwise_density、ActivationDensity、compute_density；LayerSparsity.return_sparsity 增加 density 分支（Step 2） |
| `lavis/compression/pruners/wanda_pruner.py` | T5 校准返回 image_masks；新增 AdaptiveMultimodalInputActivation；T5 _prune 支持 token_selection 与 image_masks；WrappedGPT 签名兼容；BLIPT5LayerWandaPruner 接入 DAS+AMIA（Step 3、4、5） |
| `evaluate_blip.py` | 增加/传递 token_selection、score_method、sparsity_ratio_granularity；可选注册 blipt5_tamp_pruner（Step 6） |

以上即为按原版 TAMP 方法与思路、原封不动迁移到 LAVIS_backup 的完整方案，后续你可按步骤一步步指导实现，每步再决定具体代码写法。
