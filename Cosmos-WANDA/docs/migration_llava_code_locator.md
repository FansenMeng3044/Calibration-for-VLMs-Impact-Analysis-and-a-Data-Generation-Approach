# LLaVA WANDA / 单模态分开剪枝代码定位

本文件记录从当前 workspace 中定位到的 LLaVA-NeXT 8B 剪枝实现，作为迁移到 Cosmos3-Edge/Reasoner 的基准。

> **迁移协议已锁定：** 本项目只剪并评测图文回答所用的完整 Reasoner，Generator 完全排除。`joint` 中 vision encoder 用图片激活，AR 用真实 visual embeddings + language embeddings 的融合激活；`separate` 中 vision encoder 只用图片做局部 WANDA，AR 只用 language tokenizer 路径、严格不接收 vision encoder 输出，再把两套 mask 合并回同一个 Reasoner。任何只剪 AR、用 dummy image、或从 joint hidden states 事后删 visual tokens 的实现都不属于这里定义的 separate。详细硬定义和验收条件见第 9-14 节。

本地基准目录：

- `E:\1study\calibration\TAMP`

远端预期对应目录：

- `/private/workspace/hycui/project/Tamp`

已通过 SSH 验证服务器版本：

- host：`10.103.92.120`
- ports：`1254/1255/1256/1257/1260` 均可连接。
- 服务器目录：`/private/workspace/hycui/project/Tamp`
- git head：`8c044e1058d98d86df9e6bb38dd8d863e8d4c9cc`
- branch：`main`

本地 `E:\1study\calibration\TAMP` 与服务器 `/private/workspace/hycui/project/Tamp` 的关键迁移文件 SHA256 完全一致：

| 文件 | SHA256 |
|---|---|
| `llava/evaluate.py` | `97c0d15120a58ddb2b62a6de30fccdf6b7b71c5bb30b9c723239bd7e787e248c` |
| `llava/pruners/wanda_pruner.py` | `d06e643ad8cbea6e69f825ae499842bb4f0f53c4a4fa81f65412a34ffc314244` |
| `llava/pruners/layer_single_base_pruner.py` | `86adb6365a47efc0863d26177b14e609bc5d0542ef2cc4051521c1cefcb70b42` |
| `llava/pruners/data_loader.py` | `977f30321bc14e35b0e20a9b7a29223847534604fac3c7dcefc922437fe9f820` |
| `llava/model/llava_arch.py` | `962f392f5d06130c2a4a3051f3be8cf3dc3cc4665ed5648a87afc3431c524289` |
| `scripts/prune/wanda_joint_both50_fourcalib_eval.sh` | `7484a2f60803a16e97ed440367b2fc5e8d86d8c8ae62dbac24180932156a89d1` |
| `scripts/prune/tamp_fourcalib_prune_eval_common.sh` | `0f8fe992d19d7a44e2828a790268abdbab0fe0c76c2863d25f9666a3a431b23f` |
| `scripts/prune/tamp_text_fourcalib_llava_next_8b_prune_eval.sh` | `9014d685468c93bb4e1391e846e800290bb7114c9d66ef2e2c03ea9bd083dfb6` |

注意：服务器 TAMP worktree 是 dirty 状态，包含 LLaVA 剪枝相关未提交改动和新增脚本/数据目录。迁移 Cosmos 时应以当前 dirty tree 为准，而不是只看 git commit。

## 1. 真正要迁移的 LLaVA 路径

优先使用：

- `TAMP/llava/evaluate.py`
- `TAMP/llava/pruners/wanda_pruner.py`
- `TAMP/llava/pruners/layer_single_base_pruner.py`
- `TAMP/llava/pruners/data_loader.py`
- `TAMP/llava/model/llava_arch.py`

不要把 `TAMP/videollama2/pruners/wanda_pruner.py` 当作主迁移源；那套是 VideoLLaMA2 侧的相似实现。LLaVA-NeXT 8B 脚本实际调用的是 `TAMP/llava/evaluate.py`。

## 2. evaluate.py：入口和方法选择

文件：`TAMP/llava/evaluate.py`

关键定位：

- `--calibration_modality`：约第 65-75 行，取值 `multimodal` 或 `text`。
- `--prune_method`：约第 85-88 行，支持 `wanda/sparsegpt/magnitude/atv/tamp`。
- `--token_selection`：约第 117-120 行，支持 `naive/amia/atv`。
- TAMP alias：约第 124-180 行。
  - 用户传 `--prune_method tamp` 后，内部改成 `args.prune_method = "wanda"`。
  - 同时强制：
    - `token_selection = "amia"`
    - `score_method = "density_sum"`
    - `sparsity_ratio_granularity = "layer"`
  - 因此 TAMP 本质是 Wanda kernel + AMIA token selection + DAS layer sparsity。
- ATV guard：约第 185-217 行。
  - ATV 强制 LLM-only、uniform sparsity。
- 构造 pruner：约第 253-352 行。
  - `wanda` → `LLaVALayerWandaPruner`
  - `atv` → `LLaVALayerATVPruner`
  - `sparsegpt` → `LLaVALayerSparseGPTPruner`
  - `magnitude` → `LLaVALayerMagnitudePruner`
- 保存 metadata：约第 369-666 行。
  - `tamp_metadata.json`
  - `atv_metadata.json`
  - `wanda_metadata.json`
  - `tamp_text_metadata.json`

LLaVA 模块前缀：

- LLM：`model.layers.`
- VIT：`model.vision_tower.vision_tower.vision_model.encoder.layers.`

对应函数：

- `compute_llm_linear_sparsity()`：约第 45-46 行。
- `compute_vit_linear_sparsity()`：约第 49-53 行。

## 3. data_loader.py：校准数据如何变成 forward batch

文件：`TAMP/llava/pruners/data_loader.py`

关键定位：

- `_normalize_text_calibration_record()`：约第 13-53 行。
  - text-only 校准只允许纯文本字段。
  - 禁止 `image/images/video` 字段。
  - 禁止 `<image>` placeholder。
- `LazySupervisedDataset.__init__()`：约第 57-188 行。
  - 支持 JSON/YAML/多 JSON。
  - 约第 137-142 行根据 `calibration_modality == "text"` 规范化纯文本样本。
  - 约第 144-184 行按 `task_split_path/task_name/nsamples/sample_select` 选校准样本。
- `_get_item()`：约第 286-391 行。
  - 有 `image` 字段时读图。
  - 有 `video` 字段时读视频。
  - 否则是纯文本。
  - 若 `is_multimodal` 为真但样本无图，会塞 zero image；text 模式要避免这个分支。
- `DataCollatorForSupervisedDataset.__call__()`：约第 407-451 行。
  - 生成 `input_ids/labels/attention_mask`。
  - 有视觉输入才生成 `images/image_sizes/modalities`。
- `create_data_loader()`：约第 454-460 行。
  - batch size 固定为 1。

迁 Cosmos 时，需要等价实现：

- 多模态校准：文本+图像输入，能得到 text token 和 visual token 的位置 mask。
- 文本校准：完全绕开 vision encoder，构造 all-false visual mask。

## 4. llava_arch.py：modality mask 从哪里来

文件：`TAMP/llava/model/llava_arch.py`

关键定位：

- `prepare_inputs_labels_for_multimodal()`：约第 251 行开始。
- text-only/bypass 分支：约第 254-267 行。
  - `images is None` 时：
    - `self.temp_label = torch.zeros_like(input_ids, dtype=torch.bool)`
    - `self.temp_attention_mask = attention_mask.detach().bool()`
- multimodal 插入图像 token：约第 457-589 行。
  - 文本 token mask 置 False。
  - 图像 feature token mask 置 True。
  - padding 后写：
    - `self.temp_label = image_token_masks`
    - `self.temp_attention_mask = attention_mask.detach().bool()`

迁 Cosmos 时，这是最重要的接口之一：pruner 不直接理解 tokenizer/processor，只要求第一层 hidden states 对齐的：

- `image_mask` / modality mask
- `token_attention_mask` / padding-valid mask

## 5. wanda_pruner.py：普通 Wanda 核心

文件：`TAMP/llava/pruners/wanda_pruner.py`

关键定位：

- `find_layers()`：约第 41-60 行。
  - 递归找所有 `nn.Linear`。
- `WrappedGPT`：约第 63-92 行。
  - 普通 Wanda activation wrapper。
  - `add_batch()` 把输入展平成 `[hidden, tokens]`。
  - 统计：
    - `scaler_row += ||inp||_2^2 / nsamples`
- `LLaMALayerWandaPruner.prepare_calibration_input_encoder()`：约第 815-954 行。
  - 替换第一层为 `Catcher`。
  - 跑一次模型 forward。
  - 拦截第一层输入 `inps` 和 cache。
  - 同时保存 `image_masks` 和 `token_attention_masks`。
  - text-only 时手动构造全 False image mask。
- `LLaMALayerWandaPruner._prune()`：约第 957-1266 行。
  - 逐层循环。
  - 每层找到所有 Linear。
  - 注册 forward hook 收集 activation statistic。
  - Wanda metric：
    - `W_metric = abs(W) * sqrt(scaler_row)`
  - unstructured 剪枝：
    - 每个 output row 内排序。
    - 剪最低的 `int(columns * sparsity)` 个输入连接。
  - structured `n:m` 也在这里。
- `VITLayerWandaPruner`：约第 1296-1575 行。
  - VIT 侧同样使用 `WrappedGPT` 和 Wanda metric。
  - module path 是 `...encoder.layers`。
- `LLaVALayerWandaPruner`：约第 1578-1813 行。
  - LLaVA 总控。
  - 若 `vit_sparsity_ratio > 0`，先剪 VIT。
  - 若 `llm_sparsity_ratio > 0`，再剪 LLM。
  - LLM module path：
    - `f"{llm_model_prefix}{peft_postfix}.layers"`
  - VIT module path：
    - `f"{vit_model_prefix}.layers"`

普通 Wanda 的定义：

- `prune_method = wanda`
- `token_selection = naive`
- `sparsity_ratio_granularity = none`
- `sparsity_dict = None`
- `use_variant = False`

这种情况下没有 AMIA、没有 DAS、没有 ATV。

## 6. wanda_pruner.py：ATV / text-only ATV

文件：`TAMP/llava/pruners/wanda_pruner.py`

关键定位：

- `compute_atv_visual_token_selection()`：约第 178-306 行。
  - 对每层输入/输出的 visual tokens 计算 cosine distance。
  - 全局 mean distance 决定 `selection_scale = min(1, alpha * mean_distance)`。
  - 每样本 `k = round(selection_scale * text_tokens)`。
  - 选 distance 最高的 visual tokens。
- `compute_atv_text_only_selection()`：约第 309-372 行。
  - text-only ATV 显式退化：
    - visual tokens 必须为 0。
    - selected visual tokens 必须为 0。
    - 所有有效 text tokens 保留。
- `WrappedATV`：约第 375-470 行。
  - activation statistic 使用：
    - all valid text tokens
    - selected visual tokens
  - text-only 时只用 text tokens。
- `LLaVALayerATVPruner`：约第 1835-1903 行。
  - 继承 `LLaVALayerWandaPruner`。
  - 强制：
    - `token_selection = "atv"`
    - `vit_sparsity_ratio = 0`
    - uniform sparsity
    - 不允许 `sparsity_dict`

## 7. layer_single_base_pruner.py：TAMP / AMIA / DAS

文件：`TAMP/llava/pruners/layer_single_base_pruner.py`

关键定位：

- `cos_pairwise_density()`：约第 18-80 行。
  - 计算 visual-visual、language-language、visual-language density。
  - 会用 `attention_mask` 排除 padding。
- `das_diversity_score()`：约第 83-103 行。
  - multimodal 时是原始三项：
    - `(1-s_v) + (1-s_l) + (1-s_vl)`
  - pure text 时只存在 `s_l`：
    - `3 * (1-s_l)`
  - 这就是单模态分开 Text-DAS 的数学定义。
- `ActivationDensity`：约第 204-265 行。
  - 用于 LayerSparsity 的 density statistic。
- `LayerSparsity`：约第 492-912 行。
  - 根据 layer/group score 分配 per-layer sparsity。
  - 若 `layer_to_group_mapping` 为空，返回 uniform sparsity。
  - `score_method=density_sum` 时走 `compute_density()`。
  - 用整数 keep budget 保证全局目标参数数精确。
- `compute_density()`：约第 1247-1331 行。
  - 逐层 forward，收集 `ActivationDensity`。
  - 生成每个 Linear weight 的 DAS importance。

TAMP separate 的本质：

- 纯文本校准。
- vision tower bypass。
- 只剪 LLM。
- token selection 用 AMIA 的单模态 reduction。
- layer sparsity 用 Text-DAS：`3*(1-s_l)`。
- 最终每个 Linear 内仍用 Wanda metric 做 unstructured mask。

## 8. 脚本层定位

### 8.1 老版 LLM-only naive Wanda

文件：`TAMP/scripts/prune/wanda.sh`

关键定位：

- 约第 22-48 行。
- 固定：
  - model=`llama3-llava-next-8b`
  - calibration_source=`sharegpt4v`
  - `token_selection=naive`
  - `--prune_method wanda`
  - `--llm_sparsity_ratio $sparsity_ratio`
  - `--vit_sparsity_ratio 0`

这是最简单的 LLM-only Wanda，不是四校准源矩阵脚本。

### 8.2 四校准源 joint pure Wanda

文件：`TAMP/scripts/prune/wanda_joint_both50_fourcalib_eval.sh`

关键定位：

- usage：约第 4-28 行。
- 四源：约第 75-88 行，`mathvista/mmbench/mmmu/okvqa`。
- prune command：约第 384-402 行。
- 固定：
  - `--calibration_modality multimodal`
  - `--llm_sparsity_ratio 0.5`
  - `--vit_sparsity_ratio 0.5`
  - `--sparsity_ratio_granularity none`
  - `--prune_method wanda`
  - `--token_selection naive`

这是“pure Wanda joint both50”的标准迁移样板。

### 8.3 TAMP joint/separate 四校准源

文件：

- `TAMP/scripts/prune/tamp_fourcalib_prune_eval_common.sh`
- `TAMP/scripts/prune/tamp_joint_fourcalib_llava_next_8b_prune_eval.sh`
- `TAMP/scripts/prune/tamp_separate_fourcalib_llava_next_8b_prune_eval.sh`

关键定位：

- mode 分支：约第 129-145 行。
  - `joint`：
    - `calibration_modality=multimodal`
    - image+text calibration
  - `separate`：
    - `calibration_modality=text`
    - pure-text calibration JSON
- prune command：约第 448-467 行。
  - `--prune_method tamp`
  - `--llm_sparsity_ratio ${sparsity_ratio}`
  - `--vit_sparsity_ratio 0`
  - joint 模式额外传 `--image_folder`

注意：这里叫 TAMP，不是 pure naive Wanda。它内部仍然用 Wanda kernel，但打开了 AMIA + DAS。

### 8.4 单独 text TAMP runner

文件：`TAMP/scripts/prune/tamp_text_fourcalib_llava_next_8b_prune_eval.sh`

关键定位：

- usage：约第 4-29 行。
- prune command：约第 186-201 行。
- 固定：
  - `--calibration_modality text`
  - `--llm_sparsity_ratio ${sparsity_ratio}`
  - `--vit_sparsity_ratio 0`
  - `--prune_method tamp`

### 8.5 pure-text preflight

文件：`TAMP/scripts/prune/check_tamp_text_llava_next_8b.py`

关键定位：

- 约第 23-53 行检查每行必须是纯文本。
- 约第 77-93 行检查 LLaVA-NeXT Llama-3 8B 配置：
  - `num_hidden_layers=32`
  - `hidden_size=4096`
- 约第 136-183 行检查 task split 和 nsamples。

## 9. Cosmos 迁移的实验边界：只研究完整 Reasoner

本项目的下游任务是“图像 + 文本问题 → 文本回答”，所以迁移和评测范围锁定为 **Cosmos3-Edge Reasoner**：

- Reasoner 是本实验中的完整模型范围，包括 vision encoder、视觉到 AR 的连接/投影路径、AR language transformer 和文本输出头。
- Generator、diffusion transformer、VAE，以及图像/视频/音频/action 生成分支全部排除。
- Generator 的参数不得进入参数量、稀疏率、重要性统计、hook 列表、checkpoint 或运行时显存统计。
- 最终评测始终使用完整的图文问答 Reasoner 做端到端前向。所谓 `separate` 只描述校准和重要性统计方式，不表示推理时拆成两个模型。

还必须区分“模型范围”和“默认可剪参数范围”：

- 模型范围是整个 Reasoner。
- 为与现有 LLaVA WANDA 对齐，默认可剪参数是 vision encoder blocks 和 AR transformer blocks 内的 `nn.Linear.weight`。
- tokenizer、embedding、norm、bias、视觉连接器/projector、`lm_head` 默认保持 dense，除非另开实验明确把它们纳入。
- 因此不能只写一个含糊的“Reasoner 50%”：必须同时报告 vision-linear sparsity、AR-linear sparsity、全部目标 Linear 的加权总 sparsity，以及整个 Reasoner 全参数口径下的实际 zero ratio。

若 Cosmos 代码把 AR 与 diffusion 称为 shared attention，需要按实际参数对象处理：

- attention/SDPA 的矩阵乘、softmax 等共享算子本身没有可剪参数。
- Reasoner 的 AR Q/K/V/O 投影属于 Reasoner，可进入 AR WANDA。
- Generator/diffusion 的 Q/K/V/O 投影不进入本实验。
- 若存在字面意义上同一个 `Parameter` 被多个模块引用，必须按 `id(parameter)` 去重；不能在 Reasoner 和 Generator 名字下重复计数或重复置 mask。

## 10. 两个 WANDA 实验的唯一正确含义

为避免继续混用 LLaVA 脚本中的 `joint/separate/text-only` 命名，Cosmos 迁移固定成下面两个协议。二者使用相同校准样本 ID、相同可剪模块白名单、相同每部分 sparsity budget 和相同 row-wise WANDA 规则；**唯一核心变量是 AR 侧重要性是否看过视觉表示**。

### 10.1 联合剪枝：`cosmos_wanda_joint_reasoner`

联合剪枝使用真实的图文校准前向：

```text
image ──> vision encoder ──> visual embeddings ──┐
                                                  ├─> fused AR sequence ──> AR transformer ──> answer
text  ──> language tokenizer ──> token embeddings ┘
```

重要性来源必须满足：

- vision encoder 的 WANDA 激活来自校准图片经过 vision encoder 时各层的真实输入。
- AR transformer 的 WANDA 激活来自真实融合后的 AR sequence；该 sequence 同时包含语言 token embedding 和 vision encoder 输出经过连接/投影后的 visual embedding。
- AR 统计不能删掉 visual tokens，也不能退化成 text-only。
- 对 padding 用 valid-token mask 排除；同时保存 visual-token mask 和 language-token mask，作为运行时断言和审计信息。
- “联合”指两部分的重要性来自同一套多模态样本和完整图文数据流，不等于把 vision 与 AR 的全部权重放进一次全局排序。默认仍是各 Linear 内按 output row 做 bottom-k，vision 与 AR 分别执行其 sparsity budget。

与现有 LLaVA 基线完全对齐时还有一个容易漏掉的顺序语义：`LLaVALayerWandaPruner.prune()` 先实际剪掉 VIT，再采集/计算 LLM WANDA。因此 LLaVA joint 中的 LLM/AR 重要性看到的是**已剪 vision encoder**产生的视觉表示。Cosmos 的第一版 exact migration 应保留这个顺序：

1. 用图像校准并剪 vision encoder。
2. 用同一协议重新跑完整图文前向。
3. 用已剪 vision encoder 产生的 visual embeddings 统计并剪 AR transformer。

如果先在 dense Reasoner 上同时缓存两侧重要性、最后统一落 mask，那是另一个合理但不同的消融，必须命名为类似 `joint_dense_stats`，不能与 LLaVA exact joint 结果混报。

### 10.2 单模态分开剪枝：`cosmos_wanda_separate_reasoner`

这里的 separate **不是只剪 text/AR，也不是在一次联合前向后把 visual tokens 从统计张量中过滤掉**。它由两个真正隔离的局部 WANDA 校准分支组成，最后把两套 mask 合并到同一个 Reasoner。

#### A. Vision-only 局部分支

```text
image ──> image processor ──> vision encoder ──> stop
```

- 只给 vision encoder 输入图片。
- hook 和 WANDA 统计只覆盖 vision encoder blocks 内的目标 Linear。
- 每个 vision Linear 使用其本层真实图像/patch hidden-state 输入计算激活尺度并直接剪 vision encoder。
- 不使用问题文本来计算 vision 权重重要性；不需要执行 AR transformer 来收集任何统计。
- 即便工程上从顶层模型入口进入，也必须在 vision encoder 层完成捕获后停止，且不得让 AR hook 收到样本。

#### B. Language-only AR 局部分支

```text
text ──> language tokenizer ──> input_ids ──> token embeddings ──> AR transformer ──> stop

image/vision encoder/projector ──X─> 不进入这次前向
```

- 输入只来自 language tokenizer 产生的 token IDs；WANDA 实际 hook 到的是这些 token IDs 经 embedding 后、进入各 AR Linear 的连续 hidden states。
- 不传 `pixel_values`，不调用 vision encoder，不调用视觉 projector，不插入 vision encoder 输出。
- prompt 中不得保留会触发视觉分支的 `<image>` placeholder；也不能用 zero image、dummy image 或全零 visual embeddings 代替“无图”。
- AR sequence 中 visual-token 数必须严格为 0；只对 valid language tokens 统计激活。
- hook 和剪枝目标是 **Reasoner 去掉 vision encoder 后的 AR transformer blocks**。视觉 projector/connector 在这条分支没有输入，默认保持 dense，不能用伪激活去剪。
- 该分支得到的 WANDA mask 只应用到 AR transformer，不得覆盖 vision-only 分支已经得到的 vision mask。

最终 separate checkpoint 的组成是：

```text
[vision encoder：image-only WANDA 后的权重]
        + [dense connector/projector]
        + [AR transformer：language-only WANDA 后的权重]
        + [其余默认 dense 的 Reasoner 模块]
```

最终 separate 模型在评测时仍按正常图文流程运行：图片经过已剪 vision encoder，视觉表示与文本表示进入已剪 AR transformer，再生成文本答案。这一方案有意制造“单模态局部重要性 → 多模态端到端推理”的设置，不能在评测时继续绕开视觉分支。

### 10.3 联合与分开的对照表

| 项目 | joint Reasoner WANDA | separate Reasoner WANDA |
|---|---|---|
| 最终模型 | 完整 Reasoner | 完整 Reasoner |
| Generator | 不加载/不统计/不剪 | 不加载/不统计/不剪 |
| vision encoder 激活 | 图片 | 图片，仅局部 vision forward |
| AR 激活 | text embeddings + visual embeddings 的融合序列 | 仅 language-token embeddings；visual token 数为 0 |
| vision encoder 是否剪 | 是 | 是 |
| AR transformer 是否剪 | 是 | 是 |
| connector/projector | 参与 joint 前向但默认不剪 | AR 校准时完全不用，最终模型中保持 dense |
| mask 合并方式 | vision 与 AR 各自 row-wise WANDA；按各自 budget | 两次隔离统计产生两套不重叠 mask，再写入同一 Reasoner |
| 最终评测输入 | image + text | image + text |
| 绝对禁止的误实现 | AR 只统计文本 token，却仍叫 joint | 从 joint hidden states 后处理删 visual token；传 dummy/zero image；只剪 AR 不剪 vision |

## 11. WANDA 数学与统计口径必须保持一致

对目标 Linear 权重 `W[o, i]`，沿校准激活的 token/batch 维累计输入通道尺度：

```text
s[i] = sum ||X[..., i]||² / normalization
importance[o, i] = abs(W[o, i]) * sqrt(s[i])
```

然后对每个 output row 独立排序，剪掉 importance 最小的目标比例输入连接。迁移时不可误改为：

- 对 `abs(W)` 单独做 magnitude pruning。
- 在整个 layer、整个 vision encoder 或整个 Reasoner 上做一次全局 bottom-k。
- 用 Linear 输出激活代替 Linear 输入激活。
- 把 vision 与 AR 的 scaler 混成同一个统计量。

现有 LLaVA `WrappedGPT.add_batch()` 在 flatten token 前用 batch 维更新 `nsamples`；batch size 为 1 时，本质上每个样本把全部 token 的平方和贡献进 `scaler_row`，不是严格按 token 总数归一化。Exact migration 应先保留这一行为；若改成 token-normalized WANDA，应作为单独消融并在 metadata 中记录。

## 12. LLaVA → Cosmos3-Edge Reasoner 的模块映射

| LLaVA 概念 | LLaVA 代码位置 | Cosmos 迁移对应 |
|---|---|---|
| 完整剪枝模型 | LLaVA 图文回答模型 | Cosmos3-Edge Reasoner；明确排除 Generator |
| LLM decoder layers | `model.layers` | Reasoner AR transformer layers |
| VIT encoder layers | `model.vision_tower.vision_tower.vision_model.encoder.layers` | Reasoner vision encoder layers；joint 和 separate 都要剪 |
| multimodal connector | LLaVA projector | Cosmos visual-to-AR connector/projector；参与 joint 数据流，默认保持 dense |
| image token mask | `model.temp_label` | joint 融合 AR sequence 上的 visual-token bool mask；separate AR 中必须全 False/数量为 0 |
| valid token mask | `model.temp_attention_mask` | Cosmos AR sequence attention/padding mask |
| AR first-layer catcher | `LLaMALayerWandaPruner.prepare_calibration_input_encoder` | hook Reasoner AR 第一层并缓存 layer inputs/cache |
| vision first-layer catcher | `VITLayerWandaPruner.prepare_calibration_input_encoder` | hook Reasoner vision encoder 第一层；separate 时捕获后停止 |
| per-layer Linear discovery | `find_layers(layer)` | 只在显式 allow-list 的 vision/AR blocks 内递归找 `nn.Linear` |
| Wanda statistic | `WrappedGPT.scaler_row` | 原样复用，并记录 normalization 口径 |
| Wanda mask | `abs(W)*sqrt(scaler_row)` row-wise bottom-k | 原样复用 |
| joint AR calibration | multimodal loader + image token insertion | AR 输入必须含真实 visual embeddings 与 language embeddings |
| separate vision calibration | LLaVA VIT local catcher | 仅图片 → vision encoder，剪 vision encoder |
| separate AR calibration | `calibration_modality=text` 的 bypass 思路 | 仅 tokenizer/text embeddings → AR，不调用 vision 路径，剪非 vision 的 AR blocks |

## 13. 实现时的硬断言和验收清单

每次剪枝运行都应把以下内容写入 metadata，并在不满足时直接报错，避免“脚本跑完但实验定义错了”。

### 13.1 模块范围

- 打印并保存 `reasoner_module_names`、`vision_prunable_names`、`ar_prunable_names`。
- `vision_prunable_names` 与 `ar_prunable_names` 的交集必须为空。
- 所有目标参数必须属于 Reasoner；任何 Generator/diffusion/VAE 前缀出现都立即失败。
- 按参数对象 ID 去重，记录每部分参数量和各口径 denominator。
- connector/projector、embedding、norm、`lm_head` 是否 dense 要逐项记录，不能靠默认猜测。

### 13.2 数据流断言

- joint vision：每个有效样本有真实 `pixel_values`，vision hook count 大于 0。
- joint AR：每个有效样本同时有 `num_visual_tokens > 0` 和 `num_language_tokens > 0`。
- separate vision：vision hook count 大于 0，AR hook count 必须为 0。
- separate AR：`pixel_values is None`，vision forward count 和 projector forward count 都必须为 0，`num_visual_tokens == 0`，AR hook count 大于 0。
- text-only loader 中禁止 image 字段、`<image>` placeholder、zero image 和 dummy image。

### 13.3 稀疏率与结果

- joint 与 separate 使用相同的 vision/AR module allow-list 和相同 sparsity budget，才是可比实验。
- 分别核验每个目标 Linear、vision 总体、AR 总体和整个 Reasoner 的 zero count。
- separate 合并后必须同时存在 vision zeros 和 AR zeros；任一部分仍全 dense 都视为失败。
- 最终保存前跑一次正常的 image+text Reasoner 前向，确认 logits 有限、生成非空且 Generator 没有被调用。
- metadata 至少保存：协议名、模型 revision、校准样本 ID、模态、图像/语言 token 数、模块清单、WANDA normalization、执行顺序、目标/实际 sparsity、每部分 zero count、输出 checkpoint。

## 14. 与现有 LLaVA 命名的关系

当前 LLaVA 代码中：

- `wanda_joint_both50_fourcalib_eval.sh` 是最接近 `cosmos_wanda_joint_reasoner` 的迁移源：多模态校准、vision 和 LLM 都剪、naive WANDA、uniform sparsity。
- `tamp_separate_*` 只是现有脚本的 TAMP text-only 模式，打开了 AMIA + Text-DAS，而且只剪 LLM；它**不是**这里定义的 `cosmos_wanda_separate_reasoner`。
- `evaluate.py --calibration_modality text --prune_method wanda` 只覆盖 separate 的 Language-only AR 分支；还必须另加 Image-only vision 分支并合并两套 mask，才构成完整的单模态分开 WANDA。
- TAMP/ATV 如果以后迁移，应另用 `cosmos_tamp_*` / `cosmos_atv_*` 命名，不能混入这两个 pure naive WANDA 实验。

最终固定只使用以下两个主实验名：

1. `cosmos_wanda_joint_reasoner`：vision 与 AR 都剪；AR 重要性来自真实图文融合序列。
2. `cosmos_wanda_separate_reasoner`：vision 用 image-only 激活局部剪；AR 用 language-only 激活局部剪；合并为一个完整的稀疏 Reasoner。

## 15. 服务器迁移实现与已验证状态（2026-08-28）

服务器实现目录：

- `/private/workspace/hycui/mfs/cosmos_wanda`
- 主程序：`/private/workspace/hycui/mfs/cosmos_wanda/cosmos_wanda_prune.py`
- 使用说明：`/private/workspace/hycui/mfs/cosmos_wanda/README.md`
- smoke runner：`/private/workspace/hycui/mfs/cosmos_wanda/run_smoke.sh`
- full runner：`/private/workspace/hycui/mfs/cosmos_wanda/run_prune.sh`
- 当前主程序 SHA256：`5a728e373e764b911650ae505ee47cec6b5eb3e3f2ed93000b95f31814735c73`

代码使用 Transformers 中 Cosmos3-Edge Reasoner 的真实模块路径：

- vision blocks：`model.visual.encoder.layers`，27 层。
- projector：`model.projector`，保持 dense。
- AR blocks：`model.language_model.layers`，28 层。
- 输出头：`lm_head`，保持 dense。

服务器模块审计结果：

- 完整 Reasoner 参数量：`2,435,616,496`。
- vision 目标：162 个 Linear，`411,070,464` 个可剪权重。
- AR 目标：168 个 Linear，`1,409,286,144` 个可剪权重。
- vision + AR 目标总计：`1,820,356,608` 个可剪权重。
- dense projector：`76,692,992` 参数。
- dense `lm_head`：`268,435,456` 参数。
- 实例化的 Generator 模块数：0。

实现保留了 LLaVA exact WANDA 的关键语义：

- 逐层收集 Linear 输入激活。
- `scaler_row` 使用按校准样本归一化的 token 平方和。
- `abs(W) * sqrt(scaler_row)`。
- 每个 output row 内 `torch.sort(..., stable=True)`，剪 `floor(columns * sparsity)` 个最小项。
- 当前层置零后重新 forward，再把已剪层输出交给下一层。
- joint 保留先剪 vision、再通过已剪 vision 统计 AR 的执行顺序。

最终代码版本已完成一层 50% smoke：

### joint smoke

- metadata：`/private/workspace/hycui/Results/mfs/cosmos_wanda_smoke_joint_20260828_123852/metadata.json`
- vision-only 阶段：vision=1、projector=0、AR=0。
- joint AR 阶段：vision=1、projector=1、AR=1。
- joint AR token：300 visual + 30 language。
- 第一层 vision zero ratio：0.5。
- 第一层 AR zero ratio：0.5。
- 最终完整图文 Reasoner forward：vision=1、projector=1、AR=1，logits finite。

### separate smoke

- metadata：`/private/workspace/hycui/Results/mfs/cosmos_wanda_smoke_separate_20260828_124011/metadata.json`
- vision-only 阶段：vision=1、projector=0、AR=0。
- separate AR 阶段：vision=0、projector=0、AR=1。
- separate AR token：0 visual + 28 language。
- 第一层 vision zero ratio：0.5。
- 第一层 AR zero ratio：0.5。
- 最终完整图文 Reasoner forward：vision=1、projector=1、AR=1，logits finite。

以上 smoke 只剪第一层且不保存 checkpoint；它们用于验证模块定位、激活来源、禁止路径、row-wise mask、稀疏率和最终端到端前向。完整 27+28 层剪枝需通过 `run_prune.sh` 单独启动。

## 16. MMBench / MMMU / OK-VQA 校准数据接口

服务器 preset 文件：

- `/private/workspace/hycui/mfs/cosmos_wanda/calibration_presets.json`

三个 source 的协议固定为：

| source | joint 图文 JSON | separate vision JSON | separate AR JSON | image root |
|---|---|---|---|---|
| MMBench | `calibration_sharegpt4v/mmbench_calibration_sharegpt4v.json` | `mmbench_20260626_181458_seed42_image_only.json` | `mmbench_20260626_181458_seed42_text_only.json` | `mfs/mmbench/images` |
| MMMU | `calibration_sharegpt4v/mmmu_calibration_sharegpt4v.json` | `mmmu_20260626_181458_seed42_image_only.json` | `mmmu_20260626_181458_seed42_text_only.json` | `mfs/mmmu/images` |
| OK-VQA | `calibration_sharegpt4v/okvqa_calibration_sharegpt4v.json` | `okvqa_20260626_181458_seed42_image_only.json` | `okvqa_20260626_181458_seed42_text_only.json` | `mfs/okvqa` |

主程序接口：

- named preset：`--calibration-preset mmbench|mmmu|okvqa`。
- joint 显式路径：重复 `--calibration-json`。
- separate 显式路径：重复 `--vision-calibration-json` 和 `--ar-calibration-json`；两类参数必须同时存在。
- `--preflight-only`：不加载模型，先检查 JSON、图片路径、样本数和 separate 两分支文本对齐。

Separate 的输入构造不再要求一个 record 同时有图和文：

- vision records 只用于图片 → vision encoder；要求图片实际存在。
- AR records 只用于 tokenizer → AR；记录对象内 `image_path=None`。
- image-only 与 text-only 文件按索引配对，并对规范化问题文本逐条做 exact equality；任一错位立即失败。
- 配对成功后为两个分支写相同 `sample_id`，方便 metadata 审计。

六组 128-sample 预检均已通过：

- MMBench joint：128/128 图片解析成功。
- MMBench separate：128 image + 128 text，mismatch=0，missing image=0。
- MMMU joint：128/128 图片解析成功。
- MMMU separate：128 image + 128 text，mismatch=0，missing image=0。
- OK-VQA joint：128/128 图片解析成功。
- OK-VQA separate：128 image + 128 text，mismatch=0，missing image=0。

最终接口 smoke：

- MMBench separate：`/private/workspace/hycui/Results/mfs/cosmos_wanda_smoke_separate_mmbench_20260828_130353/metadata.json`。
  - AR 校准 vision=0、projector=0、AR=1，visual tokens=0。
  - 最终图文验证 vision=1、projector=1、AR=1，logits finite。
- MMMU joint：`/private/workspace/hycui/Results/mfs/cosmos_wanda_smoke_joint_mmmu_20260828_130353/metadata.json`。
  - AR 校准 vision=1、projector=1、AR=1，visual tokens=351、language tokens=39。
  - 最终图文验证 vision=1、projector=1、AR=1，logits finite。

矩阵 runner：

- `/private/workspace/hycui/mfs/cosmos_wanda/run_three_dataset_matrix.sh`
- 默认依次为 MMBench、MMMU、OK-VQA 各生成一个独立 checkpoint，保持和 LLaVA “每个 calibration source 一个模型”的实验设计一致；不会把三个 source 合成一个 384-sample checkpoint。

## 17. LLaVA 三项评测定位与 Cosmos 评测接口

LLaVA 并不是用三个独立 Python evaluator；三项都通过 TAMP 环境的
`lmms-eval 0.4.0` 执行本地 task：

| benchmark | LLaVA shell 入口 | task | 数据 | 计分 |
|---|---|---|---|---|
| MMBench | `scripts/eval/mmbench_en_dev_local.sh` | `mmbench_en_dev_local` | `MMBench_eval/en/dev-00000-of-00001.parquet` | `MMBench_Evaluator.can_infer` 本地选项解析，主题/L2 分类汇总，不调用 API |
| MMMU | `scripts/eval/mmmu_val_local.sh` | `mmmu_val_local` | `MMMU_single_image/*/validation-*.parquet`，30 subjects / 900 samples | lmms-eval 官方 MMMU parser + evaluator，总体/domain/subject 汇总 |
| OK-VQA | `scripts/eval/okvqa_val2014_local.sh` | `okvqa_val2014_local` | `mfs/okvqa/okvqa_val2014_local.jsonl` | lmms-eval 官方 OK-VQA answer normalization/accuracy |

三者公共调用链是 shell → `python -m lmms_eval` → model adapter 的
`generate_until` → task `process_results`/aggregation。LLaVA adapter 参数是
`pretrained=...,conv_template=llava_llama_3,device_map=cuda:0`，固定
`batch_size=1`、`--log_samples`，并允许用 `TAMP_EVAL_LIMIT` 做接口测试。

Cosmos 迁移保持 task、prompt、generation kwargs、parser、metric 和输出格式
不变，只替换模型 adapter：

- plugin：`/private/workspace/hycui/mfs/cosmos_wanda/cosmos_lmms_plugin/models/cosmos3_edge.py`
- runner：`/private/workspace/hycui/mfs/cosmos_wanda/run_three_eval.sh`
- 一次性依赖脚本：`/private/workspace/hycui/mfs/cosmos_wanda/install_eval_deps.sh`
- 模型名：`cosmos3_edge`，由 `LMMS_EVAL_PLUGINS=cosmos_lmms_plugin` 注册，不修改 TAMP 安装包。
- 只加载 `Cosmos3EdgeForConditionalGeneration` Reasoner；不加载 Generator。
- 支持 dense 模型路径，也支持 joint/separate WANDA 的 `checkpoint/` 路径。
- MMMU 的 `<image N>` 位置保持图文交错；MMBench/OK-VQA 在问题前放图。
- 每个样本前清空 `model.model.rope_deltas`，防止跨样本继承多模态 RoPE shape 状态。
- Python 3.13 环境使用 `lmms-eval==0.4.0 --no-deps`，不允许其把 Cosmos 的 NumPy 2.5、Torch 2.10、Transformers 5.16.dev0 降级。

三项 dense 端到端 1-sample smoke 已通过：

- 输出根目录：`/private/workspace/hycui/Results/mfs/cosmos_eval/smoke_all_20260828_v4`
- MMBench prediction/target：`B` / `B`，本地 exact score=100（仅 smoke）。
- MMMU prediction/target：`B` / `B`，官方 parser score=100（仅 smoke）。
- OK-VQA prediction/target：`Ride` / `race`，单样本 VQA score=0.6（仅 smoke）。
- 已生成 MMBench JSON+XLSX、MMMU scores/predictions/records JSON、OK-VQA submission JSON，以及三项 sample JSONL 和 lmms-eval aggregated results JSON。
- smoke 退出后物理 GPU 0 显存恢复为 0 MiB。

运行示例：

```bash
GPU_ID=0 bash /private/workspace/hycui/mfs/cosmos_wanda/run_three_eval.sh \
  /private/workspace/hycui/model/Cosmos3-Edge all \
  /private/workspace/hycui/Results/mfs/cosmos_eval/dense
```

第二个参数可为 `all|mmbench|mmmu|okvqa`。只有设置
`TAMP_EVAL_LIMIT=1` 时才是 smoke；正式评测不得设置 limit。
