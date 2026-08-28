# Cosmos3-Edge TAMP 迁移前定位与协议备忘

审计日期：2026-08-28
当前阶段：已完成 LLaVA 运行代码、TAMP 算法组件、Joint/Text-only 数据流和 Cosmos3-Edge 目标模块定位；迁移代码、三数据集校准/评测接口和真实一层 smoke 已落地。后续正式 runner、metadata、validator 和实验汇总必须继续以本文件为硬协议。

## 1. 最高优先级定义：本实验中的“整个模型”是 Reasoner

本实验任务是：

```text
图像 + 文本问题 -> 文本回答
```

因此，本次 TAMP 实验的模型边界固定为 Cosmos3-Edge **Reasoner**：

```text
experiment model = Reasoner

Reasoner
├─ vision encoder
├─ projector / connector / multimodal fusion
├─ language tokenizer / token embedding
├─ AR language transformer
└─ lm_head

Generator / diffusion / VAE
└─ 完全排除：不加载、不前向、不 hook、不统计重要性、不剪枝、不计稀疏率、不保存
```

“只看 Reasoner 算重要性”的准确含义是：所有校准激活、DAS 分数、AMIA token 选择、WANDA 权重分数和稀疏率分配都只能来自 Reasoner；Generator 永远不进入计算图。

同时必须区分“实验模型范围”和“实际被剪权重范围”：

- 实验加载、保存和评测的是完整 Reasoner。
- TAMP 的实际剪枝目标仅是 Reasoner 的 AR/LLM Transformer Linear 权重。
- Vision Encoder、projector、token embedding、norm、bias 和 `lm_head` 不属于 TAMP 剪枝目标。
- Vision Encoder 在 Joint 中参与产生视觉表示，在 Separate 中完全不调用；两种模式下它的权重都保持完整 dense。

> **最高优先级禁错项：** TAMP 与本项目的 WANDA/SparseGPT separate 定义不同。本次 TAMP separate 不包含 image-only Vision Encoder 剪枝分支；它只有 text-only AR/LLM 剪枝。不得因为名称中有 `separate` 就对 Vision Encoder 生成 mask 或置零。

## 2. 两套 TAMP 的冻结定义

### 2.1 `cosmos_tamp_joint_reasoner`

这里的 `joint` 表示 **使用真实图像与文本组成的 Reasoner 融合 AR 序列来计算 TAMP 重要性**，不表示 Vision Encoder 和 AR 权重一起被剪。

```text
真实 image -> dense Vision Encoder -> dense Projector -> visual embeddings ----┐
                                                                              ├─> fused AR sequence
真实 text  -> tokenizer -> token embedding -> language embeddings -----------┘
                                              |
                                              v
                                DAS + AMIA/WANDA importance
                                              |
                                              v
                                  prune AR Linear weights only
```

固定语义：

1. 校准输入必须是配对的真实图片和文本问题。
2. Vision Encoder 和 projector 必须真实执行，产生真实 visual embeddings。
3. visual embeddings 与 language embeddings 必须按 Cosmos Reasoner 官方顺序形成 AR 第一层输入。
4. DAS 在有效 visual/language token 上计算联合多模态多样性，并给每个 AR Linear 分配稀疏率。
5. AMIA 在同一真实融合 AR 序列上选择代表 token；WANDA 激活统计只能使用这些被选中的有效 token。
6. 只允许稀疏化 Reasoner AR/LLM Transformer 内白名单 Linear 的权重。
7. Vision Encoder 与 projector 只参与前向数据流，其参数始终 dense、冻结且不进入稀疏率 denominator。

### 2.2 `cosmos_tamp_separate_textonly_reasoner`

这里的 `separate` 表示 **从数据入口就隔离出纯语言 AR subsequence 来计算 TAMP 重要性**。它不是“Vision 和 AR 两侧分别剪枝”，也没有 image-only 分支。

```text
纯 text -> tokenizer -> token embedding -> AR language transformer
                                         |
                                         v
                         text-only DAS + AMIA/WANDA importance
                                         |
                                         v
                             prune AR Linear weights only

image / Vision Encoder / projector:
校准阶段零输入、零 placeholder、零调用、零输出、零剪枝
```

固定语义：

1. 校准样本从入口开始只允许有文本；不得携带 `image`、`images`、`video`、`pixel_values`、图像尺寸或视觉 placeholder。
2. 唯一合法数据流是 `tokenizer -> token embedding -> AR language transformer`。
3. AR 第一层输入只允许来自 language tokenizer/token embedding；不得混入 Vision Encoder 或 projector 的任何输出。
4. 不允许先运行多模态/Joint forward，再按 mask 删除 visual token；被视觉融合影响过的 hidden states 不是纯文本 subsequence。
5. Vision Encoder 和 projector 的校准 forward 调用数必须严格为 0。
6. modality mask 必须全 False，visual token 数必须为 0；padding 必须由有效 token mask 排除。
7. Text-only DAS 只能使用 language-language 项，严格化简为 `3 * (1 - s_l)`。
8. AMIA 只能在有效 language tokens 上执行，不能选到视觉输出或 padding。
9. 只允许稀疏化 Reasoner AR/LLM Transformer 内与 Joint 完全相同的 Linear 白名单。
10. Vision Encoder 不输入图片、不计算重要性、不生成 mask、不被剪；最终 checkpoint 仍完整保存 dense Vision Encoder 和 dense projector。

用户所说的“剪除了 Vision Encoder 的 Reasoner”在本文中只允许解释为：

```text
TAMP target scope = Reasoner - Vision Encoder - Projector - non-AR modules
calibration dataflow = Reasoner 的 language-only AR subsequence
```

它绝不表示删除最终 checkpoint 中的 Vision Encoder。最终模型仍是：

```text
dense Vision Encoder + dense Projector + pruned AR/LLM + dense lm_head
```

## 3. Joint 与 Separate 绝不能混淆的对照

| 项目 | Joint TAMP | Separate text-only TAMP |
|---|---|---|
| 实验模型 | 完整 Reasoner | 完整 Reasoner |
| 校准输入 | 真实图片 + 文本 | 只有文本 |
| Vision Encoder 调用 | 必须大于 0 | 必须严格为 0 |
| Projector 调用 | 必须大于 0 | 必须严格为 0 |
| AR 第一层输入 | visual + language embeddings | 只有 language embeddings |
| visual-token mask | 同时包含视觉/语言位置 | 全 False |
| DAS | 三项联合多模态公式 | `3 * (1 - s_l)` |
| AMIA 候选 token | 有效 visual + language token | 仅有效 language token |
| 被剪权重 | 仅 AR/LLM Linear | 仅 AR/LLM Linear |
| 目标模块集合 | 固定白名单 | 与 Joint 完全相同 |
| Vision Encoder 权重 | dense、不剪 | dense、不剪 |
| Projector 权重 | dense、不剪 | dense、不剪 |
| Generator | 完全排除 | 完全排除 |
| 最终评测输入 | 正常 image + text | 正常 image + text |

下列任一情况必须 fail closed，不能仅打印 warning：

- Joint 没有真实图像、没有 visual token，或 Vision Encoder/projector 调用数为 0；
- Joint 用 dummy/zero image 代替真实图片；
- Separate 出现任何图像张量、视觉 placeholder、Vision Encoder/projector 调用或 visual embeddings；
- Separate 先运行多模态前向，再事后切掉 visual token；
- Separate 对 Vision Encoder 做 image-only TAMP、WANDA 或任何其他剪枝；
- Joint 与 Separate 使用了不同的 AR Linear 目标集合；
- 任一模式修改 Vision Encoder、projector、embedding、norm、bias 或 `lm_head`；
- 任一模式加载、调用、hook 或保存 Generator；
- 将 Joint 解释为 Vision 与 AR 权重的全局共同排序；
- 将 Separate 解释为 Vision/AR 两个独立 mask 的相加。

## 4. TAMP 算法组件的准确口径

当前 LLaVA 实现中，`--prune_method tamp` 是一个受保护的算法别名，必须强制：

```text
token_selection = amia
score_method = density_sum
sparsity_ratio_granularity = layer
weight-pruning kernel = WANDA
vit_sparsity_ratio = 0
```

因此 TAMP 不是单独一种权重置零内核，而是以下两阶段组合：

```text
Phase A: DAS 根据表征多样性给目标 Linear 分配非均匀稀疏率
Phase B: AMIA 选择代表 token，WANDA 按激活加权权重重要性执行置零
```

### 4.1 Phase A：DAS 稀疏率分配

对归一化 hidden states，定义：

```text
s_v  = visual-visual token pair 的平均正余弦相似度
s_l  = language-language token pair 的平均正余弦相似度
s_vl = visual-language token pair 的平均余弦相似度
```

Joint 使用原始三项公式：

```text
D_joint = (1 - s_v) + (1 - s_l) + (1 - s_vl)
```

Separate 没有视觉模态，只允许 language-language 项；为保持与分配器预期的三项尺度一致，严格化简为：

```text
D_text = 3 * (1 - s_l)
```

不得把不存在的视觉项伪造为零相似度后继续代入三项公式。padding 不属于语言 token，也不得进入 `s_l`。

DAS 在全局目标 keep/zero 预算下进行分配；多样性更高的目标 Linear 应获得更多保留参数。当前 LLaVA 基线的单目标最大稀疏率为 `0.6`，整体目标稀疏率通常为 `0.5`；迁移时必须作为显式协议参数保存，不能静默改变。

需要特别锁定一个容易误读的实现细节：LLaVA 的 `sparsity_ratio_granularity="layer"` 当前把每个完整 `*.weight` 参数名映射到自身，因此实际是 **per-Linear tensor 分配**，不是每个 Transformer block 共用一个稀疏率。Cosmos 迁移必须保持这一语义，或者在改变前单独报告并重新定义实验。

### 4.2 Phase B：AMIA token 选择与 WANDA 置零

AMIA 只在当前协议允许的有效 token 集合上工作：

- Joint：真实融合序列中的有效 visual + language token；
- Separate：只有 tokenizer/embedding 产生的有效 language token；
- 两者都必须排除 padding。

LLaVA 实现使用最终有效 causal query 的多头平均注意力作为 token score，并结合余弦距离图、KNN 与 MMD 选出代表 token。迁移不能把 AMIA 简化成随机 token、前 N 个 token 或仅按模态 mask 采样。

对某个目标 Linear 的输入列 `j` 和权重元素 `[r,j]`：

```text
scaler_row[j] = calibration selected-token 上输入激活平方和的样本平均
W_metric[r,j] = abs(W[r,j]) * sqrt(scaler_row[j])
```

随后按 DAS 为该完整 Linear 分配的稀疏率，在每个 output row 内将 `W_metric` 最小的连接置零。每一层剪完后必须重新前向并把稀疏输出传播到下一层，不能始终复用 dense 层输出。

TAMP 中的 token 选择只用于校准重要性统计：

- 不删除最终 checkpoint 中的 token；
- 不改变正式评测的图文输入序列；
- 不剪 tokenizer、embedding 或 Vision Encoder；
- 不是推理时 token pruning。

## 5. LLaVA 权威迁移源定位

迁移源仓库：

```text
/private/workspace/hycui/project/Tamp
origin: https://github.com/G-JWLee/TAMP.git
branch: main
Git HEAD / origin main: 8c044e1058d98d86df9e6bb38dd8d863e8d4c9cc
```

该 worktree 含正式实验所依赖的未提交修改与新增脚本。迁移必须以当前服务器实际文件和下列 SHA256 为准，不能只重新 clone origin main：

| 角色 | 文件 | SHA256 |
|---|---|---|
| 旧版上游入口参考 | `scripts/prune/tamp.sh` | `31c612123dacc15ca661005c450eafb48392adf8217428717d6bac1acf4f1dc7` |
| Joint 正式入口 | `scripts/prune/tamp_joint_fourcalib_llava_next_8b_prune_eval.sh` | `0d2fdc0d5075aaf7d7f9c9783f7f31e226c80bbe807d2e5b7f72c40d265f23a2` |
| Separate 正式入口 | `scripts/prune/tamp_separate_fourcalib_llava_next_8b_prune_eval.sh` | `5407908cce5d4c5782a0467911a8b1ed90a22e2f828f86c8e2fa09460f4de1e0` |
| 两模式公共 runner | `scripts/prune/tamp_fourcalib_prune_eval_common.sh` | `0f8fe992d19d7a44e2828a790268abdbab0fe0c76c2863d25f9666a3a431b23f` |
| CLI、TAMP alias、metadata | `llava/evaluate.py` | `97c0d15120a58ddb2b62a6de30fccdf6b7b71c5bb30b9c723239bd7e787e248c` |
| 校准数据与 text-only 拒绝规则 | `llava/pruners/data_loader.py` | `977f30321bc14e35b0e20a9b7a29223847534604fac3c7dcefc922437fe9f820` |
| AMIA、WANDA 与逐层传播 | `llava/pruners/wanda_pruner.py` | `d06e643ad8cbea6e69f825ae499842bb4f0f53c4a4fa81f65412a34ffc314244` |
| DAS 与全局稀疏率分配 | `llava/pruners/layer_single_base_pruner.py` | `86adb6365a47efc0863d26177b14e609bc5d0542ef2cc4051521c1cefcb70b42` |
| modality mask 构造 | `llava/model/llava_arch.py` | `962f392f5d06130c2a4a3051f3be8cf3dc3cc4665ed5648a87afc3431c524289` |
| LLaVA 顶层模型接线 | `llava/model/language_model/llava_llama.py` | `2d4c0fbece1637240eb0361f252f7ffb50b8007d0686f1b22b8286921437c816` |
| checkpoint checker | `scripts/prune/check_tamp_pruned_model.py` | `ec628b2195936e3b0a37a9fb6c562a0e937b3addf2f190ceafe53255ef33ef64` |
| text-only preflight | `scripts/prune/check_tamp_text_llava_next_8b.py` | `a1275647cbb5ee5d9097d087d65a9124a49cc7d8e64e42d91bfcab3b5108070d` |
| text reduction tests | `tests/test_tamp_text_reduction.py` | `ce23836b8a739c2ae5f631765a37428546df32c026bf225eab9083ebcf8f42c9` |

函数级迁移责任：

| 文件/符号 | 必须迁移的语义 |
|---|---|
| `evaluate.py` 的 TAMP alias guards | TAMP 强制 AMIA + density_sum + per-Linear DAS；拒绝错误参数组合 |
| `layer_single_base_pruner.py::cos_pairwise_density()` | 只在有效 modality token 上计算相似度，padding 排除 |
| `layer_single_base_pruner.py::das_diversity_score()` | Joint 三项公式与 Separate `3*(1-s_l)` 严格化简 |
| `layer_single_base_pruner.py::prepare_calibration_input_encoder()` | 捕获 DAS 所需的 AR 第一层输入、mask 与有效 token |
| `wanda_pruner.py::prepare_calibration_input_encoder()` | 捕获 AMIA/WANDA 的逐层 AR hidden states 与 mask |
| `WrappedGPT`/AMIA 相关实现 | 代表 token 选择与激活平方统计 |
| `wanda_pruner.py::_prune()` | 逐层 metric、rowwise mask、稀疏输出传播 |
| `get_sparsity(..., "layer")` | 每个完整 AR Linear 权重张量独立分配稀疏率 |

### 5.1 LLaVA 当前目标范围证据

LLaVA-NeXT 8B 当前 TAMP 目标为：

```text
32 AR blocks × 7 Linear = 224 target Linear weights
target Linear weight parameters = 6,979,321,856
vit_sparsity_ratio = 0
```

已生成的各 checkpoint 稀疏率 YAML 均有 224 个条目，说明 TAMP 对每个 AR Linear 独立分配，而不是按整个 block 共用一个值。已有模型 checker 通过，但它未逐张量 bitwise 验证全部非目标参数；Cosmos validator 必须补上这一缺口。

### 5.2 LLaVA 当前 Separate 数据证据

当前纯文本校准文件：

```text
/private/workspace/hycui/mfs/four_calibration_text_only_seed42.json
/private/workspace/hycui/mfs/four_calibration_text_only_seed42_task_split.json
```

它由四个来源各 128 条组成；对已核验 slice，Separate 的问题文本与 Joint 去掉 `<image>` 后逐条一致。Separate batch 设置 `is_multimodal=False` 且没有 `images`，不是 dummy/zero image。Cosmos 正式矩阵只使用 MMBench、MMMU、OK-VQA 三个来源时，应保持三源各自的 128-sample/seed 协议；是否包含 MathVista 是实验矩阵选择，不是 TAMP 算法定义。

## 6. Cosmos3-Edge 模块映射与目标白名单

Cosmos TAMP 应复用已经在 Cosmos WANDA/SparseGPT/ATV 中验证的 Reasoner-only 加载、数据隔离、AR 第一层捕获、保存和评测框架，但不能复用它们错误的目标范围。

当前 Cosmos3-Edge Reasoner 指纹：

| 语义 | Cosmos 模块 | TAMP 状态 |
|---|---|---|
| Vision Encoder | `model.visual.encoder.layers`，27 层 | Joint 仅参与前向；两模式均 dense、不剪 |
| Projector/connector | `model.projector` | Joint 仅参与前向；两模式均 dense、不剪 |
| AR Transformer | `model.language_model.layers`，28 层 | 唯一 TAMP 剪枝范围 |
| token embedding / norms | language model 对应模块 | dense、不剪 |
| `lm_head` | Reasoner 输出头 | dense、不剪 |
| Generator/diffusion/VAE | Reasoner 外 | 不加载、不调用、不保存 |

预期 AR allow-list 指纹：

```text
model.language_model.layers.* 下 168 个 nn.Linear.weight
target AR Linear weight parameters = 1,409,286,144
Vision Encoder target count = 0
Projector target count = 0
Generator target count = 0
```

实现必须先枚举并保存完整目标名清单。如果模型版本、28/168 数量或权重数与指纹不一致，必须在剪枝前失败并重新审计，不能自动扩大到 Vision Encoder，也不能静默漏掉 AR Linear。

Joint 与 Separate 必须使用同一份 allow-list hash。两者唯一允许的算法差异是校准数据流及其产生的 DAS/AMIA 统计。

## 7. 校准数据接口

正式三套校准源固定为：

```text
MMBench
MMMU
OK-VQA
```

Joint 每条样本至少包含：

```text
task
sample_id
真实 image/images
question/prompt text
```

Separate 必须由相同 task、sample ID、顺序和问题文本派生纯文本记录，只保留 tokenizer 所需的语言内容。答案标签可以存在于数据记录中用于评分，但不得拼入校准输入造成 label leakage。

每一数据源必须记录：

```text
calibration_task
sample_ids in order
sample_id hash
normalized prompt hash
nsamples
seed
prompt/template version
Joint image path/content hash and pixel protocol
Separate forbidden-visual-field audit
```

Joint/Separate 可比性 gate：

- `task`、sample ID、顺序和规范化问题文本逐条一致；
- 唯一区别是 Joint 有真实图像，Separate 从入口移除所有视觉输入；
- Separate 不得通过 processor 默认行为生成 zero image；
- 所有校准实际用到的 sample ID 必须写入 checkpoint metadata。

## 8. Cosmos 实现必须分开的两条捕获路径

不得用一个含条件分支但无法证明隔离性的 cache 函数模糊处理两套协议。至少在接口与审计字段上明确区分：

```text
build_ar_cache(protocol="joint")
build_ar_cache(protocol="separate_text_only")
```

### 8.1 Joint cache gate

- Vision Encoder 调用数 `> 0`；
- projector 调用数 `> 0`；
- 每个样本有效 visual token 数 `> 0`；
- 每个样本有效 language token 数 `> 0`；
- modality mask、valid-token mask 与 AR sequence 长度严格对齐；
- visual embeddings 非 dummy、非全零，来自当前真实图像；
- DAS 与 AMIA 使用同一协议下的融合 AR states；
- Generator 调用数 `= 0`。

### 8.2 Separate cache gate

- batch key 白名单不含任何视觉字段；
- tokenizer 输入无视觉 placeholder；
- Vision Encoder 调用数 `= 0`；
- projector 调用数 `= 0`；
- visual mask 全 False；
- visual token 数 `= 0`；
- AR hidden states 与纯文本 attention mask 严格对齐；
- DAS 只报告 `language_language` 项；
- AMIA selected token 全部属于有效 language token；
- padding selected count `= 0`；
- Generator 调用数 `= 0`。

## 9. 稀疏率与参数统计口径

TAMP 的主要稀疏率 denominator 只能是目标 allow-list 内的 AR Linear 权重：

```text
denominator = numel(model.language_model.layers.* target nn.Linear.weight)
```

不得把以下 dense 参数加入 denominator：

- Vision Encoder；
- projector/connector；
- token embedding；
- norm、bias；
- `lm_head`；
- Generator 任意参数。

metadata 至少同时报告：

```text
requested_target_ar_sparsity
allocated_target_ar_sparsity
achieved_target_ar_sparsity
reasoner_overall_zero_fraction  # 只能作为补充
```

由于 rowwise `int(columns * sparsity)` 会产生取整，实际稀疏率可有可解释的极小误差，但必须保存精确 numerator/denominator，不能只写四舍五入后的 `0.5`。

## 10. Checkpoint 与 metadata 硬合同

所有 checkpoint 至少保存：

```text
algorithm = "TAMP"
algorithm_components = ["DAS", "AMIA", "WANDA"]
experiment_model = "Cosmos3-Edge Reasoner"
generator_excluded = true
target_scope = "Reasoner AR/LLM Linear only"
target_allowlist
target_allowlist_sha256
ar_target_linear_count = 168
vision_target_count = 0
projector_target_count = 0
non_ar_target_count = 0
requested_target_ar_sparsity
max_sparsity_per_linear
achieved_target_ar_sparsity
calibration_task
num_samples
seed
score_method = "density_sum"
token_selection = "amia"
sparsity_allocation_granularity = "per_linear_tensor"
per_linear_das_statistics
per_linear_allocated_sparsity
per_linear_achieved_sparsity
```

Joint 必须另外保存：

```text
protocol = "joint_multimodal_reasoner_ar"
calibration_modality = "multimodal"
ar_sequence_source = "real_visual_plus_language_embeddings"
vision_encoder_call_count > 0
projector_call_count > 0
visual_token_count > 0
language_token_count > 0
das_terms = ["visual_visual", "language_language", "visual_language"]
```

Separate 必须另外保存：

```text
protocol = "separate_text_only_reasoner_ar"
calibration_modality = "text"
ar_sequence_source = "language_tokenizer_embedding_only"
vision_encoder_call_count = 0
projector_call_count = 0
visual_token_count = 0
visual_placeholder_count = 0
das_terms = ["language_language"]
das_formula = "3 * (1 - s_l)"
vision_encoder_pruned = false
```

## 11. 剪枝后强验证

每个 checkpoint 必须与同一个 dense Reasoner 基线逐张量比较：

1. 所有 Vision Encoder 张量 bitwise equal；
2. 所有 projector/connector 张量 bitwise equal；
3. token embedding、norm、bias、`lm_head` bitwise equal；
4. 只有 allow-list 中的 AR Linear weights 允许变化和新增零值；
5. Generator 权重不得出现在 checkpoint 或 target 清单；
6. 实际变化的权重名集合必须是 allow-list 的子集，目标模块数量必须准确；
7. 每个 Linear 的实际零数必须与已分配稀疏率、rowwise 取整规则一致；
8. 全局精确 numerator/denominator 必须达到协议容差；
9. Joint 与 Separate 的 target allow-list hash 必须相同；
10. metadata、调用计数、token 计数与协议完全一致。

此外，每个 checkpoint 在进入正式评测前都必须重载并完成至少一个正常图文回答 smoke：

```text
image + question
  -> dense Vision Encoder
  -> dense Projector
  -> pruned AR/LLM
  -> finite logits / valid text answer
```

Separate 的纯文本隔离只发生在剪枝校准阶段；不能把最终模型改成 text-only，也不能在三项评测中跳过图像。

## 12. 三项评测接口的不可变边界

每个 dense、Joint-pruned 和 Separate-pruned Reasoner 均使用相同的正常图文评测协议：

| 评测 | LLaVA 已核验入口 | 完整样本数 |
|---|---|---:|
| MMBench | `scripts/eval/mmbench_en_dev_local.sh` | 4,329 |
| MMMU | `scripts/eval/mmmu_val_local.sh` | 900 |
| OK-VQA | `scripts/eval/okvqa_val2014_local.sh` | 5,046 |

Cosmos 侧只替换模型 adapter；数据集 split、sample IDs、prompt/template、generation config、scorer 和样本数验证必须保持一致。所有结果必须保存原始 prediction JSON/JSONL、scored result、validation JSON 和日志。

## 13. 实现与运行前 fail-closed 清单

### Joint

- [ ] 只加载 Reasoner，Generator 排除。
- [ ] 使用真实 image+text paired calibration。
- [ ] Vision Encoder/projector 真实调用但不剪。
- [ ] AR 第一层输入含真实 visual + language embeddings。
- [ ] DAS 使用三项联合多模态公式并排除 padding。
- [ ] AMIA 只在有效融合 token 上选择代表 token。
- [ ] 只剪 AR Linear allow-list。

### Separate text-only

- [ ] 只加载 Reasoner，Generator 排除。
- [ ] 输入从数据边界开始只有文本，无图像/placeholder。
- [ ] 直接走 tokenizer/embedding/AR subsequence。
- [ ] Vision Encoder/projector 调用数严格为 0。
- [ ] visual mask 全 False，visual token 数为 0。
- [ ] DAS 严格使用 `3*(1-s_l)`。
- [ ] AMIA 只使用有效 language token，padding 为 0。
- [ ] 不存在 image-only Vision Encoder 剪枝分支。
- [ ] 只剪与 Joint 相同的 AR Linear allow-list。
- [ ] dense Vision Encoder 仍完整保存在最终 checkpoint。

### 两者共同

- [ ] Cosmos 模块指纹与 28 AR layers / 168 Linear / 1,409,286,144 weights 一致。
- [ ] target allow-list 和 hash 完全一致。
- [ ] DAS 是 per-Linear tensor 分配，不误写为 per-block。
- [ ] AMIA + WANDA 逐层传播稀疏输出。
- [ ] 非目标 Reasoner 张量逐位不变。
- [ ] 目标稀疏率 numerator/denominator 验证通过。
- [ ] 正常 image+text checkpoint reload smoke 通过。
- [ ] 三项评测恢复标准多模态前向。

## 14. 迁移阶段结论

本次 Cosmos TAMP 的两套实验只能定义为：

```text
Joint TAMP:
  真实 image+text -> 完整 Reasoner 多模态前向 -> 融合 AR 序列算 DAS/AMIA/WANDA
  -> 只剪 Reasoner AR/LLM Linear

Separate TAMP:
  纯 text -> tokenizer/embedding -> language-only AR subsequence 算 DAS/AMIA/WANDA
  -> 只剪 Reasoner AR/LLM Linear
  -> Vision Encoder/projector 校准调用为 0，权重完整 dense
```

两者都是对“完整 Reasoner checkpoint”开展实验，但 TAMP 的目标权重范围始终是 `Reasoner - Vision Encoder - Projector - non-AR modules`。两者的差别是 **重要性统计的数据流**，不是剪枝目标集合。任何未来实现若偏离这一定义，必须在运行前停止并更新科研协议，禁止静默修改。

## 15. 已落地迁移代码与验证状态

实现目录：

```text
/private/workspace/hycui/mfs/cosmos_tamp
```

主要实现：

```text
cosmos_tamp_prune.py               DAS + AMIA + WANDA、Joint/Text-only 数据流
calibration_presets.json           MMBench/MMMU/OK-VQA 两协议校准接口
validate_calibration_alignment.py  三数据集逐样本对齐 gate
validate_tamp_migration.py         LLaVA 源码哈希和算法静态 gate
test_tamp_core.py                  CPU 协议与算法测试
validate_cosmos_checkpoint.py      重载、目标范围和非目标 bitwise gate
cosmos_lmms_plugin/                Cosmos Reasoner-only lmms-eval adapter
run_prune.sh                       单 checkpoint 正式入口
run_three_dataset_matrix.sh        三校准源 checkpoint 入口
run_three_eval.sh                  三项正常图文评测与自动结果验证
```

已通过：

```text
Python compile                       PASS
all shell scripts bash -n            PASS
TAMP CPU unit tests                  7/7 PASS
LLaVA/Cosmos static migration gate   PASS
MMBench calibration alignment        128/128
MMMU calibration alignment           128/128
OK-VQA calibration alignment         128/128
三套 Separate AR 记录无图片          true
```

真实模型一层 smoke：

```text
Joint:
/private/workspace/hycui/Results/mfs/cosmos_tamp_smoke_joint_mmbench_20260828_191052

Separate:
/private/workspace/hycui/Results/mfs/cosmos_tamp_smoke_separate_mmbench_20260828_191052
```

Joint smoke 合同：

```text
Vision/Projector/AR calibration calls = 1/1/1
valid language/visual tokens           = 237/80
DAS formula                            = (1-s_v)+(1-s_l)+(1-s_vl)
DAS allocation                         = 6 Linear, exact 0.5 budget
AMIA selected/valid across 6 Linear    = 108/1902
Vision target count                    = 0
actual AR one-layer sparsity            = 0.4997355143229167
final normal multimodal logits finite  = true
```

Separate smoke 合同：

```text
Vision/Projector/AR calibration calls = 0/0/1
valid language/visual tokens           = 235/0
DAS formula                            = 3*(1-s_l)
DAS allocation                         = 6 Linear, exact 0.5 budget
AMIA selected/valid across 6 Linear    = 70/1410
Vision target count                    = 0
actual AR one-layer sparsity            = 0.49981689453125
final normal multimodal logits finite  = true
```

三项评测接口均已用 dense Reasoner 各跑 1 条并通过自动结果 validator：

```text
MMBench: /private/workspace/hycui/Results/mfs/cosmos_tamp_eval_smoke_mmbench_20260828
MMMU:    /private/workspace/hycui/Results/mfs/cosmos_tamp_eval_smoke_mmmu_20260828
OK-VQA:  /private/workspace/hycui/Results/mfs/cosmos_tamp_eval_smoke_okvqa_20260828
```

这些 limited smoke 只验证接口与产物合同，不能作为正式性能结果。正式 checkpoint 仍需运行 128 条校准、全部 28 层、保存后非目标逐张量 bitwise 验证，并完成 4,329/900/5,046 条全量评测。
