# TAMP 迁移：Padding 被算进「language」的修复方案

本文档说明 **在完成 Step 2 / Step 3 之后**，如何单独加一步（或嵌入 Step 2/3 的扩展）来排除 padding，让 `cos_pairwise_density` 只对「真实 vision + 真实 text」算 density，避免 padding 拉偏 `l_mean_sim`。

---

## 一、问题是什么

- Blip2T5 的 encoder 输入序列长度固定：`S = num_query + max_txt_len`，短句会 **padding**。
- 当前 `temp_label` 只区分：前 `num_query` = vision（True），其余 = 非 vision（False）。
- **没有**区分「真实 text」和「padding」，所以 `cos_pairwise_density` 里会把 **padding 位置也算成 language**。
- 后果：`l_emb` 里包含大量 padding 的 embedding（常为 0 或重复），会拉低或扭曲 `l_mean_sim`，进而影响 DAS 的层重要性。

---

## 二、在「哪一步」解决（插入位置）

| 主步骤 | 你要做的事 |
|--------|------------|
| **Step 2 做完之后** | 扩展 `cos_pairwise_density` 和 / 或 `compute_density`，支持「只对有效 token 算 density」（见下）。 |
| **Step 3 做完之后** | 扩展校准返回值，**除了 image_masks 再存 encoder 的 attention_mask**，并传给 compute_density。 |

建议顺序：**先完成 Step 2、Step 3 的原始设计**（能跑通 DAS、拿到 image_masks），再按本方案加「排除 padding」的扩展，这样改动最小、也便于对比有无 padding 修复的效果。

---

## 三、方案 A：valid_mask 传入 cos_pairwise_density（推荐）

### 1. Step 3 扩展（wanda_pruner：校准多返回 attention_mask）

- **文件**：`lavis/compression/pruners/wanda_pruner.py`
- **位置**：T5 的 `prepare_calibration_input_encoder` 里，Catcher 触发时（`except ValueError` 分支）。
- **当前**：已 append `model.temp_label` 到 `image_masks`，返回 `(inps, outs, caches, image_masks)`。
- **扩展**：
  - 在 Catcher 的 `forward(inp, **kwargs)` 里，`kwargs` 中已有 encoder 的 `attention_mask`（或 `encoder_attention_mask`，以 T5 实际 key 为准）。
  - 新增列表 `encoder_attention_masks = []`，在 append inps/caches/image_masks 的同一处执行：
    - `encoder_attention_masks.append(kwargs["attention_mask"])`（或对应的 key）。
  - 返回值改为 `(inps, outs, caches, image_masks, encoder_attention_masks)`。
- **调用方**：所有解包 `prepare_calibration_input_encoder` 返回值的地方，从四元组改为五元组，并向下传递 `encoder_attention_masks`（例如传给 `compute_density`）。

说明：T5 encoder 的 cache 里通常已有 `attention_mask`，可直接从 Catcher 的 `kwargs` 取出，无需在 blip2_t5 里新增字段。

### 2. Step 2 扩展（layer_single_base_pruner：cos_pairwise_density 支持 valid_mask）

- **文件**：`lavis/compression/pruners/layer_single_base_pruner.py`
- **函数**：`cos_pairwise_density(embeddings, image_mask, valid_mask=None, eps=1e-8)`。
- **语义**：
  - `valid_mask`：可选，形状与 `image_mask` 一致（(S,) 或 (B, S)），**True = 有效 token**（参与 v/l 划分），**False = 无效（如 padding，不参与）**。
  - 若 `valid_mask is None`：行为与现在完全一致，所有位置按 `image_mask` 分 v/l（兼容现有调用）。
  - 若 `valid_mask` 非 None：只对 **valid_mask 为 True** 的位置做划分：
    - **vision**：`valid_mask & image_mask`
    - **language**：`valid_mask & (~image_mask)`
  - 这样 padding（valid_mask=False）既不进 v 也不进 l，不会被算进 `l_emb`。
- **实现要点**：
  - 在循环内：`valid = valid_mask[b]`（若 1D 则先 broadcast 成 (B, S) 再取 `valid_mask[b]`）。
  - `v_idx = torch.where(valid & mask)[0]`，`l_idx = torch.where(valid & (~mask))[0]`。
  - 其余（L2 归一化、上三角、> 0 取平均、边界 nv/nl&lt;2 返回 0）不变。

### 3. compute_density 里传入 valid_mask

- **文件**：`lavis/compression/pruners/layer_single_base_pruner.py`
- **位置**：`compute_density` 里，对每一层、每个样本调用 `ActivationDensity.add_batch` 或直接调 `cos_pairwise_density` 的地方。
- **修改**：
  - 若校准时返回了 `encoder_attention_masks`，则对第 j 个样本：`valid_mask_j = encoder_attention_masks[j]`（与 `inps[j]` 的 seq 维度一致，bool）。
  - 调用 `cos_pairwise_density(out, image_mask=image_masks[j], valid_mask=valid_mask_j)`；
  - 若 `ActivationDensity.add_batch` 的接口改为支持 `valid_mask`，则这里把 `valid_mask_j` 传入 `add_batch(..., valid_mask=valid_mask_j)`，在 `add_batch` 内再调 `cos_pairwise_density(..., valid_mask=valid_mask)`。

这样，**只在 Step 2/3 都做完并跑通后**，按「Step 3 扩展 → Step 2 扩展（valid_mask）→ compute_density 传 valid_mask」顺序加一遍，即可排除 padding。

---

## 四、方案 B：compute_density 里先 mask 再调 cos_pairwise_density

不改 `cos_pairwise_density` 的签名，只在 **compute_density** 里用 attention_mask 把 padding 位置的 embedding 先处理掉，再调现有的 `cos_pairwise_density(embeddings, image_mask)`。

### 1. Step 3 扩展

- 与方案 A 相同：校准时多存并返回 `encoder_attention_masks`，调用方解包五元组并传给 compute_density。

### 2. compute_density 内「先 mask 再调」

- 对第 j 个样本：`out_j = outs[j]`，形状 (B, S, D)（或 (S, D)）；`attn_j = encoder_attention_masks[j]`，形状 (B, S) 或 (S,)。
- **做法一（置零）**：`out_j = out_j * attn_j.unsqueeze(-1)`（padding 位置置 0），再 `cos_pairwise_density(out_j, image_masks[j])`。缺点：0 向量参与 L2 归一化会得到 NaN/异常，需在 `cos_pairwise_density` 里跳过范数为 0 的行，或改用做法二。
- **做法二（只取有效位置，再组 batch）**：对每个样本，用 `attn_j` 和 `image_masks[j]` 取出有效位置的 embedding，组成 (N_valid, D)，并构造等长的 `image_mask` 指示这些位置是 v 还是 l；对该 (N_valid, D) 调用 `cos_pairwise_density`。这样无需改 `cos_pairwise_density`，但需要按样本拼接/循环，代码稍复杂。

若不想动 `cos_pairwise_density` 的接口，可用方案 B 的做法二；若希望接口清晰、后续复用方便，推荐方案 A。

---

## 五、插入到主流程的时机小结

| 顺序 | 动作 |
|------|------|
| 1 | 按 TAMP_MIGRATION_DESIGN 完成 **Step 2**（cos_pairwise_density、ActivationDensity、compute_density、return_sparsity 的 density 分支）。 |
| 2 | 按 TAMP_MIGRATION_DESIGN 完成 **Step 3**（校准时收集 temp_label → image_masks，返回四元组）。 |
| 3 | **（本方案）Step 3 扩展**：校准时再存 `encoder_attention_masks`，返回五元组，调用方适配。 |
| 4 | **（本方案）Step 2 扩展**：`cos_pairwise_density(..., valid_mask=None)`，只对 valid 位置分 v/l。 |
| 5 | **（本方案）compute_density**：从校准拿到 `encoder_attention_masks`，调用 `cos_pairwise_density` 或 `add_batch` 时传入 `valid_mask=encoder_attention_masks[j]`。 |

完成 1–2 即可跑通 DAS；完成 3–5 后，padding 不再被算进 language，`l_mean_sim` 只反映真实 text 的密度。
