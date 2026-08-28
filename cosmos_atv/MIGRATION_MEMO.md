# Cosmos3-Edge ATV 迁移前定位备忘

审计日期：2026-08-28
当前阶段：LLaVA/官方 ATV 源码定位、算法语义、joint/text-only 数据流、Cosmos3-Edge 实现、校准接口、checkpoint 合同和三项评测接口已经核验；本文件冻结正式全矩阵运行使用的 Cosmos3-Edge Reasoner ATV 定义。正式运行必须由 fail-closed gate、任务依赖、checkpoint/eval validator 和最终聚合完成标记共同验收。

## 1. 本次实验的最高优先级协议锁

本实验处理的是：

```text
图像 + 文本问题 -> 文本回答
```

因此，实验中的“整个模型”固定指 **Cosmos Reasoner**，不指完整 Cosmos 世界模型：

```text
experiment model = Reasoner

Reasoner
├─ vision encoder
├─ projector / connector
├─ language tokenizer / token embedding
├─ AR language transformer
└─ lm_head

Generator / diffusion / VAE
└─ 完全排除：不加载、不前向、不统计重要性、不计稀疏率、不保存
```

ATV 的官方算法只剪 AR/LLM Linear 权重，不剪 vision encoder。因此，本迁移的两套 ATV 都必须满足：

- ATV 权重重要性只在 Reasoner 内计算；Generator 永远不进入实验。
- 实际稀疏化目标只允许是 `model.language_model.layers.*` 内的 `nn.Linear.weight`。
- `model.visual.encoder.layers.*` 必须保持完整 dense，不计算或应用 ATV mask。
- projector、token embedding、norm、bias、`lm_head` 默认保持 dense。
- “对整个 Reasoner 做实验”表示加载、保存和评测的是完整 Reasoner checkpoint；不表示 ATV 必须剪 Reasoner 内每一个子模块。

> **绝对禁止误解：** 本次 ATV 的“单模态分开剪枝”不再像 WANDA/SparseGPT separate 那样包含 image-only vision 剪枝。它只用纯文本 AR subsequence 计算 ATV/WANDA 激活重要性；vision encoder 完整保留、不剪、校准时零调用。

## 2. 两套 ATV 的冻结定义

### 2.1 `cosmos_atv_joint_reasoner`

这里的 `joint` 指 **AR 重要性来自真实图文联合数据流**，不指 vision 与 AR 权重一起被剪。

```text
image -> dense vision encoder -> dense projector -> visual embeddings ----┐
                                                                         ├─> fused AR sequence
text  -> tokenizer -> token embedding -> language embeddings -----------┘
                                      |
                                      v
                         ATV activation statistics
                                      |
                                      v
                     prune AR Linear weights only
```

固定语义：

1. 输入必须是成对的真实图片和文本问题。
2. vision encoder 和 projector 必须真实执行，产生真实 visual embeddings。
3. visual embeddings 与 language embeddings 必须按照 Cosmos 官方 Reasoner 的融合顺序形成 AR 第一层输入。
4. 对每个 AR decoder layer，使用该层 visual-token 输入/输出 hidden states 的余弦距离执行官方 ATV top-k 选择。
5. AR Linear 的激活统计使用全部有效 language tokens 加 ATV 选中的 visual tokens。
6. 只剪 `model.language_model.layers.*` 内的 Linear 权重。
7. vision encoder 和 projector 只参与联合激活数据流，权重保持 dense。

### 2.2 `cosmos_atv_separate_textonly_reasoner`

这里的 `separate` 指 **从 Reasoner 中隔离出纯语言 AR subsequence 计算重要性**。它不是 vision/AR 两侧分别剪枝，也不包含 image-only vision 分支。

```text
text -> tokenizer -> token embedding -> AR language transformer
                                      |
                                      v
                  text-only zero-visual activation statistics
                                      |
                                      v
                     prune AR Linear weights only

image / vision encoder / projector: calibration 时零输入、零调用
```

固定语义：

1. 校准样本只允许包含文本；不得携带 `image/images/video` 字段或视觉 placeholder。
2. 只执行 `tokenizer -> token embedding -> model.language_model.layers`。
3. AR 第一层输入只允许来自 language tokenizer/token embedding；不得包含 vision encoder 或 projector 的任何输出。
4. 不允许先跑 joint forward，再从融合 hidden states 中事后删除 visual-token 位置；这不等价于真正的 text-only subsequence。
5. visual-token mask 必须全 False；visual token 数、selected visual token 数必须全部为 0。
6. `mean_cosine_distance` 和 `selection_scale` 必须记录为 `null`，不能伪造视觉选择统计。
7. `alpha_effective=false`。此时 ATV 退化为仅使用全部有效文本 token 的 WANDA 激活统计，但实验名称必须明确记录为 `text_only_zero_visual_ablation`，不能标成官方 multimodal ATV。
8. 只剪 `model.language_model.layers.*` 内的 Linear 权重。
9. vision encoder 不输入图片、不计算重要性、不生成 mask、不被剪；最终 checkpoint 仍完整保留 dense vision encoder。

“剪除了 vision encoder 的 Reasoner”在本备忘中的准确含义只能是：

```text
AR 校准数据流 = Reasoner 去掉 vision 数据流后的 language-only subsequence
```

它不表示从最终 checkpoint 删除 vision encoder。最终保存和评测的仍是完整 Reasoner：dense vision encoder + dense projector + pruned AR。

## 3. joint 与 separate 绝不能混淆的对照

| 项目 | Joint ATV | Separate text-only ATV |
|---|---|---|
| 实验模型 | 完整 Reasoner | 完整 Reasoner |
| 校准输入 | 真实图片 + 文本 | 只输入文本 |
| vision encoder 调用 | 必须真实调用 | 必须为 0 |
| projector 调用 | 必须真实调用 | 必须为 0 |
| AR 第一层输入 | visual embeddings + language embeddings | 只有 language embeddings |
| visual-token mask | 必须含 True 和 False | 必须全 False |
| ATV top-k 视觉选择 | 官方公式，必须执行 | 强制 0；alpha 不生效 |
| AR 激活统计 | 全部有效文本 + 选中视觉 token | 全部有效文本 token |
| 被剪模块 | 仅 AR Linear | 仅 AR Linear |
| vision encoder 权重 | dense、不剪 | dense、不剪 |
| projector 权重 | dense、不剪 | dense、不剪 |
| Generator | 完全排除 | 完全排除 |
| 算法标记 | official multimodal ATV | text-only zero-visual ablation |

任何出现下列情况的运行都必须失败，不得只给 warning：

- joint 模式没有真实视觉 token，或 vision/projector 调用数为 0；
- separate 模式出现图片张量、视觉 placeholder、vision/projector 调用或 visual embeddings；
- 任一模式修改 vision encoder、projector、embedding、norm、bias 或 `lm_head`；
- 任一模式加载或调用 Generator；
- separate 模式从 joint hidden states 事后切片得到 text tokens；
- separate 模式将 `alpha_effective` 标成 true；
- 将 separate text-only ATV 宣称为官方 multimodal ATV；
- 将“joint”解释成 vision 与 AR 权重的全局排序或共同 mask。

## 4. 官方 ATV 算法定位与公式

官方仓库：

```text
/private/workspace/hycui/project/ATV-Pruning
Git HEAD: c75fe920c2ce1eb2688b1066008af55f13a4e123
```

官方干净版本必须用 Git 对象读取，因为当前 worktree 含本地实验修改：

```bash
git show HEAD:qwen/activation_aware_pruner.py
git show HEAD:qwen/prune.py
```

干净官方文件指纹：

```text
qwen/activation_aware_pruner.py
SHA256: 7491566401b172ced1df96370f88315bbb525f28b9faa1b93886887ff102f561

qwen/prune.py
SHA256: 62e41431383ec8014832f3299157fb80594d6122bbdca230de6d1ab8ed301d0f
```

官方 `qwen/prune.py` 明确设置：

```python
model_prefix = "language_model"
module_to_process = "language_model.layers"
```

因此官方 ATV 没有 vision encoder 剪枝路径。

对 AR 第 `l` 层、样本 `s` 的第 `i` 个 visual token：

```text
d[l,s,i] = 1 - cosine(h_in[l,s,i], h_out[l,s,i])

mean_distance[l] = mean of all d[l,s,i] over calibration visual tokens
selection_scale[l] = min(1, alpha * mean_distance[l])

k[l,s] = min(
    valid_visual_token_count[s],
    round(selection_scale[l] * valid_text_token_count[s])
)
```

每个样本选择余弦距离最大的 top-k visual tokens。AR Linear 的激活样本集合为：

```text
all valid text tokens U selected visual tokens
```

对 Linear 输入列 `j`：

```text
scaler_row[j]
  = calibration-sample mean of the summed squared activation on retained tokens

W_metric[r,j]
  = abs(W[r,j]) * sqrt(scaler_row[j])
```

然后在每个 output row 内剪掉 `W_metric` 最小的固定比例连接。ATV 的权重重要性内核仍是 WANDA；ATV 特有部分是 multimodal 模式下的 visual-token 选择。

ATV visual-token 选择只服务于校准重要性统计：

- 不会删除最终 checkpoint 中的 token embedding；
- 不会剪 vision encoder；
- 不会改变正常评测时的图文输入序列；
- 不是推理阶段 token pruning。

## 5. LLaVA 权威迁移源定位

LLaVA 迁移源：

```text
/private/workspace/hycui/project/Tamp
branch: main
Git HEAD: 8c044e1058d98d86df9e6bb38dd8d863e8d4c9cc
```

TAMP worktree 是 dirty 状态；ATV 实现和运行脚本主要存在于未提交修改/新增文件中。迁移必须以当前实际工作副本及下列 SHA256 为准，不能只看 Git HEAD。

| 角色 | 文件 | SHA256 |
|---|---|---|
| ATV 选择、激活统计和 AR 剪枝 | `llava/pruners/wanda_pruner.py` | `d06e643ad8cbea6e69f825ae499842bb4f0f53c4a4fa81f65412a34ffc314244` |
| CLI、pruner 构造、metadata | `llava/evaluate.py` | `97c0d15120a58ddb2b62a6de30fccdf6b7b71c5bb30b9c723239bd7e787e248c` |
| multimodal/text-only 校准数据 | `llava/pruners/data_loader.py` | `977f30321bc14e35b0e20a9b7a29223847534604fac3c7dcefc922437fe9f820` |
| visual/language token 对齐 mask | `llava/model/llava_arch.py` | `962f392f5d06130c2a4a3051f3be8cf3dc3cc4665ed5648a87afc3431c524289` |
| LLaVA 顶层模型接线 | `llava/model/language_model/llava_llama.py` | `2d4c0fbece1637240eb0361f252f7ffb50b8007d0686f1b22b8286921437c816` |

函数级定位：

| 位置 | 符号 | 迁移语义 |
|---|---|---|
| `wanda_pruner.py:178` | `compute_atv_visual_token_selection()` | joint 官方余弦距离、alpha、k、top-k 规则 |
| `wanda_pruner.py:309` | `compute_atv_text_only_selection()` | separate 强制零视觉 token 退化 |
| `wanda_pruner.py:375` | `WrappedATV` | 全部有效文本 + 选中视觉 token 的激活统计 |
| `wanda_pruner.py:815` | `prepare_calibration_input_encoder()` | 捕获 AR 第一层输入、modality mask、valid-token mask |
| `wanda_pruner.py:957` | `_prune()` | 逐 AR layer 收集激活、生成 mask、传播已剪输出 |
| `wanda_pruner.py:1835` | `LLaVALayerATVPruner` | 强制 LLM-only、vision sparsity=0、uniform sparsity |
| `evaluate.py:185-216` | ATV CLI guards | 拒绝 vision pruning、非 uniform sparsity 和非法 text-only 组合 |
| `evaluate.py:277-302` | pruner 构造 | 选择 multimodal 或 text-only calibration mode |
| `evaluate.py:430-475` | `atv_metadata.json` | 保存算法变体、激活来源、层统计和实际稀疏率 |
| `data_loader.py:13` | `_normalize_text_calibration_record()` | separate 禁止所有视觉字段与 placeholder |
| `llava_arch.py:251` | `prepare_inputs_labels_for_multimodal()` | 构造与 AR hidden states 对齐的 visual/valid mask |

## 6. LLaVA 两套 runner 的真实调用链

Joint：

```text
scripts/prune/atv_joint_fourcalib_llava_next_8b_prune_eval.sh
  -> scripts/prune/atv_fourcalib_prune_eval_common.sh joint
     -> scripts/prune/atv_llava_next_8b.sh
        -> llava/evaluate.py --prune_method atv
```

Separate/text-only：

```text
scripts/prune/atv_separate_fourcalib_llava_next_8b_prune_eval.sh
  -> scripts/prune/atv_fourcalib_prune_eval_common.sh separate
     -> scripts/prune/atv_textonly_fourcalib_llava_next_8b_prune.sh
        -> llava/evaluate.py
             --prune_method atv
             --calibration_modality text
             --vit_sparsity_ratio 0
```

公共脚本已经明确：

```text
joint    = multimodal joint-token
separate = text-only zero-visual-token
sparsity = uniform, LLM only
```

不能根据文件名把 LLaVA `atv_separate` 解释成 image-only vision + text-only AR。

## 7. Cosmos3-Edge 精确模块映射

Cosmos ATV 迁移应复用现有 Cosmos WANDA/SparseGPT 已验证的 Reasoner-only 加载、数据隔离、第一层捕获、保存和评测框架。

| 语义 | Cosmos 模块 | ATV 权重状态 |
|---|---|---|
| vision encoder | `model.visual.encoder.layers`，27 层 | 两种模式均 dense、不剪 |
| projector | `model.projector` | dense、不剪 |
| AR transformer | `model.language_model.layers`，28 层 | 唯一 ATV 剪枝目标 |
| token embedding / norms | language model 对应模块 | dense、不剪 |
| `lm_head` | Reasoner 输出头 | dense、不剪 |
| Generator/diffusion/VAE | Reasoner 外模块 | 不加载、不调用、不计数 |

AR 默认目标白名单：

```text
model.language_model.layers.* 下的 168 个 nn.Linear.weight
总目标权重参数：1,409,286,144
```

如果模型版本、模块数量或权重数与上述指纹不同，必须在剪枝前失败并重新审计，不能自动扩大或缩小目标范围。

### 7.1 Joint 捕获要求

Joint AR cache 必须来自完整 Reasoner 的真实图文前向：

```text
build_ar_cache(protocol="joint")
```

验收至少包括：

- vision encoder 调用数大于 0；
- projector 调用数大于 0；
- 每个样本 visual token 数大于 0；
- 每个样本 language token 数大于 0；
- visual/language mask 与 AR hidden sequence 长度完全对齐；
- AR 输入包含真实 visual embeddings，而不是 dummy image 或零向量；
- Generator 调用数为 0。

### 7.2 Separate 捕获要求

Separate AR cache 必须直接来自 language subsequence：

```text
build_ar_cache(protocol="separate")
```

等价数据流：

```text
processor/tokenizer 的纯文本 token IDs
  -> language token embedding
  -> model.language_model.layers
```

验收至少包括：

- 输入 batch 不含 `pixel_values`、图像对象、图像尺寸或视觉 placeholder；
- vision encoder 调用数严格等于 0；
- projector 调用数严格等于 0；
- visual mask 全 False；
- visual-token count 和 selected-visual-token count 全为 0；
- AR hidden states 与纯文本 attention mask 完全对齐；
- padding token 不参与激活统计；
- Generator 调用数为 0。

## 8. 校准数据接口

预期校准源：

```text
MMBench
MMMU
OK-VQA
```

Joint 每条样本必须提供：

```text
sample_id
image / images
question text
task name
```

Separate 必须从相同 sample ID 派生纯文本记录，只保留问题文本和必要 prompt，不得把答案标签作为模型输入，也不得保留任何图像字段或 placeholder。

为保证 joint/separate 可比较，必须记录：

- calibration task；
- 原始 sample ID；
- sample 顺序；
- `nsamples`；
- seed；
- prompt/template 版本；
- joint 图片像素协议；
- separate vision/projector 零调用审计。

不能用 dummy image 实现 separate，也不能让 processor 因缺图自动生成 zero image 后仍执行 vision encoder。

## 9. 稀疏率和参数口径

ATV 稀疏率 denominator 只能是 AR 目标白名单内的 Linear 权重：

```text
denominator = numel(model.language_model.layers.*.nn.Linear.weight)
```

不得把以下参数加入 denominator：

- dense vision encoder；
- projector；
- token embedding；
- norm、bias；
- `lm_head`；
- Generator 任意参数。

因此，metadata 必须同时区分：

```text
target_ar_linear_sparsity
achieved_ar_linear_sparsity
reasoner_overall_zero_fraction
```

主要实验稀疏率使用前两者。Reasoner overall zero fraction 只能作为补充说明，不能冒充 ATV 目标稀疏率。

## 10. 保存与 metadata 合同

每个 checkpoint 至少保存：

```text
algorithm = "ATV-Pruning"
experiment_model = "Cosmos3-Edge Reasoner"
generator_excluded = true
target_scope = "model.language_model.layers.*.nn.Linear.weight"
vision_target_count = 0
projector_target_count = 0
ar_target_linear_count = 168
target_ar_linear_sparsity
achieved_ar_linear_sparsity
calibration_task
num_samples
seed
alpha
uniform_sparsity = true
layer_statistics
```

Joint 必须记录：

```text
algorithm_variant = "official_multimodal_visual_cosine"
official_multimodal_atv = true
calibration_modality = "multimodal"
alpha_effective = true
visual_token_selection = "atv_cosine_topk"
vision_encoder_called = true
projector_called = true
```

Separate 必须记录：

```text
algorithm_variant = "text_only_zero_visual_ablation"
official_multimodal_atv = false
calibration_modality = "text"
alpha_effective = false
visual_token_selection = "forced_zero"
vision_encoder_call_count = 0
projector_call_count = 0
```

每层 separate 统计必须满足：

```text
mode = "text_only_zero_visual"
mean_cosine_distance = null
selection_scale = null
all visual_tokens = 0
all selected_visual_tokens = 0
```

## 11. Checkpoint 强验证

剪枝后必须将 checkpoint 与 dense Reasoner 做逐张量比较：

- 所有 vision encoder 张量 bitwise equal；
- 所有 projector 张量 bitwise equal；
- token embedding、norm、bias、`lm_head` bitwise equal；
- 只有 AR Linear weights 允许变化和新增零值；
- Generator 权重不得出现在 checkpoint 或比较清单中；
- AR 目标实际稀疏率必须与协议目标一致；
- layer statistics 数量必须等于 28 个 AR layers；
- joint/separate metadata 与各自数据流严格一致。

任何非目标张量变化都必须阻止评测。

## 12. 评测协议

无论 checkpoint 来自 joint 还是 separate，评测都必须恢复为正常图文回答：

```text
image + question
  -> dense vision encoder
  -> dense projector
  -> pruned AR transformer
  -> text answer
```

Separate 的 text-only 隔离只用于计算剪枝重要性，不能改变最终模型结构或评测输入。

三项评测继续使用既有本地任务/scorer：

| 评测 | LLaVA 权威入口 | 样本数 |
|---|---|---:|
| MMBench | `scripts/eval/mmbench_en_dev_local.sh` | 4,329 |
| MMMU | `scripts/eval/mmmu_val_local.sh` | 900 |
| OK-VQA | `scripts/eval/okvqa_val2014_local.sh` | 5,046 |

Cosmos 侧只替换模型 adapter，不修改数据、prompt、generation config、scorer 和样本集合。

## 13. LLaVA 现有产物给出的合同证据

LLaVA 实际产物：

```text
joint checkpoints:
/private/workspace/hycui/model/atv_pruned

text-only checkpoints:
/private/workspace/hycui/model/atv_textonly_pruned

joint eval:
/private/workspace/hycui/project/Tamp/results/atv_joint_fourcalib

text-only eval:
/private/workspace/hycui/project/Tamp/results/atv_separate_fourcalib
```

已核验的 MMBench checkpoint：

```text
joint:
  AR layers = 32
  achieved LLM Linear sparsity = 0.5
  vision tensors bitwise equal = 391/391
  projector tensors bitwise equal = 4/4

text-only:
  AR layers = 32
  achieved LLM Linear sparsity = 0.5
  vision tensors bitwise equal = 391/391
  projector tensors bitwise equal = 4/4
```

代码验证状态：

```text
23 ATV unit tests: PASS
ATV_LLAVA_MIGRATION_STATIC_VALIDATION_OK
ATV saved-model validation: READY FOR EVAL
```

这直接证明 LLaVA 的两套 ATV 都是 LLM-only；现有 `separate` 是 text-only zero-visual ablation，不是 vision/AR 两侧分别剪枝。

## 14. 实现前最终检查表

### Joint

- [ ] 只加载 Reasoner，Generator 排除。
- [ ] 使用真实 image+text paired calibration。
- [ ] vision 和 projector 真实调用，但权重 dense。
- [ ] AR 输入包含真实 visual + language embeddings。
- [ ] 每层按官方余弦距离和 alpha 选择 visual tokens。
- [ ] 激活统计只包含 valid text + selected visual tokens。
- [ ] 只剪 AR Linear。

### Separate text-only

- [ ] 只加载 Reasoner，Generator 排除。
- [ ] 输入只含文本，无图片/placeholder。
- [ ] 直接走 tokenizer/embedding/AR subsequence。
- [ ] vision encoder 和 projector 调用数严格为 0。
- [ ] visual mask 全 False，visual/selected counts 全 0。
- [ ] alpha_effective=false，视觉距离/scale 为 null。
- [ ] 只剪 AR Linear。
- [ ] dense vision encoder 仍完整保存在最终 checkpoint。

### 两者共同

- [ ] target allow-list 只含 28 个 AR layers 下的 168 个 Linear。
- [ ] uniform unstructured sparsity。
- [ ] padding 不参与激活统计。
- [ ] 非目标 Reasoner 张量 bitwise equal。
- [ ] 最终评测重新使用正常 image+text Reasoner forward。
- [ ] metadata 不把 text-only ablation 写成官方 multimodal ATV。

以上定义为 Cosmos ATV 迁移的科研协议。后续代码、runner、validator、checkpoint metadata 和结果汇总必须引用并遵守本文件；若需改变目标模块、稀疏率口径或 separate 数据流，必须先更新协议并明确报告，不能静默修改。

## 15. 已落地迁移代码与验证状态

实现目录：

```text
/private/workspace/hycui/mfs/cosmos_atv
```

主要文件：

```text
cosmos_atv_prune.py                 ATV core、两套数据流、保存与 metadata
calibration_presets.json            MMBench/MMMU/OK-VQA joint/text-only 接口
validate_calibration_alignment.py   逐样本校准对齐验证
validate_cosmos_checkpoint.py       重载、非目标张量 bitwise、正常图文前向验证
validate_atv_migration.py           官方/LLaVA/Cosmos 公式与目标范围静态对照
test_atv_core.py                    CPU 算法和协议单元测试
run_smoke.sh                        单样本/单 AR layer smoke
run_prune.sh                        单数据源完整剪枝入口
run_three_dataset_matrix.sh         三数据源 checkpoint 入口
run_three_eval.sh                   三项正常图文评测入口
cosmos_lmms_plugin/                 Reasoner-only lmms-eval adapter
```

实现已经用 fail-closed 模块指纹锁定：

```text
vision layers / Linear = 27 / 162（全部 dense，target count=0）
AR layers / Linear     = 28 / 168（唯一 ATV target）
vision Linear weights = 411,070,464（dense）
AR target weights      = 1,409,286,144
Generator modules      = 0
```

校准预检状态：

```text
MMBench joint/separate text match = 128/128
MMMU joint/separate text match    = 128/128
OK-VQA joint/separate text match  = 128/128
三套 separate AR 记录无图片       = true
三套 verification 图片存在        = true
verification_used_for_importance  = false
```

对齐报告：

```text
/private/workspace/hycui/mfs/cosmos_atv/calibration_alignment.json
```

算法单元测试：6/6 通过，包括官方 alpha/k/top-k、visual-token clamp、text-only forced zero、视觉泄漏拒绝、样本归一化激活统计和视觉字段/placeholder 拒绝。

官方/LLaVA/Cosmos 静态对照已通过：

```text
COSMOS_ATV_MIGRATION_STATIC_VALIDATION_OK
```

一层真实模型 smoke 已通过：

```text
joint:
/private/workspace/hycui/Results/mfs/cosmos_atv_smoke_joint_mmbench_20260828_174947

separate:
/private/workspace/hycui/Results/mfs/cosmos_atv_smoke_separate_mmbench_20260828_174609
```

Joint smoke 合同：

```text
vision/projector/AR calibration calls = 1/1/1
language/visual tokens                = 237/80
selected visual tokens                = 2
mean cosine distance                  = 0.006500244140625
vision layers pruned                  = 0
AR layer-0 achieved sparsity          = 0.5
final normal multimodal forward       = valid finite logits
```

Separate smoke 合同：

```text
vision/projector/AR calibration calls = 0/0/1
language/visual tokens                = 235/0
selected visual tokens                = 0
mean cosine distance/selection scale  = null/null
alpha_effective                       = false
vision layers pruned                  = 0
AR layer-0 achieved sparsity          = 0.5
final normal multimodal forward       = valid finite logits
```

这两个 smoke 都证明：校准阶段的数据流按协议严格区分，但最终 checkpoint/模型验证始终回到正常 image+text Reasoner forward。
