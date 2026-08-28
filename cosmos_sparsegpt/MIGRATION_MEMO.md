# Cosmos SparseGPT 迁移前定位备忘（LLaVA 基线）

审计日期：2026-08-28
当前阶段：源代码定位、语义核对和迁移边界已冻结；独立 Cosmos SparseGPT 迁移代码正在按本备忘实现和验证，**尚未启动正式剪枝/评测矩阵**。

## 1. 冻结结论

1. 服务器 TAMP/LLaVA 中存在并完成过的是 **LLaVA-NeXT 8B joint SparseGPT**：同一个多模态校准源下先剪 vision encoder，再用已剪 vision encoder 产生的图文融合序列剪 LLM。
2. 这里的 `joint` 不是把 vision 与 LLM 的全部权重放进一个全局 Hessian 或全局排序；它是两个目标子网按顺序做 layer-wise SparseGPT，二者使用同一套多模态样本。
3. 服务器 TAMP/LLaVA 中**没有 separate SparseGPT** 的脚本或类。`evaluate.py` 虽允许 `calibration_modality=text` 与 `vit_sparsity=0`，但 SparseGPT 的 LLM catcher 仍无条件访问 `batch['images']`，所以现有 SparseGPT text-only 路径并未真正接通。
4. 本地 ECoFLaP/LAVIS 中存在 BLIP2-T5 的 separate SparseGPT：text-only 剪 T5、image-only 剪 ViT、再合并 checkpoint。它可以作为 separate 实验契约参考，但**不是 LLaVA 代码，也不能直接映射 Cosmos 模块或 forward**。
5. Cosmos 的可复用基础不是 LLaVA text-only SparseGPT，而是现有 `cosmos_wanda` 已验证的数据隔离、第一层捕获、模块白名单、保存和验收框架；算法内核需从 LLaVA `SparseGPT` 迁入。

### 1.1 用户确认的模型范围和重要性范围（协议锁）

本实验做的是“图像 + 文本问题 → 文本回答”，因此从这一版开始固定以下术语，不允许在实现中换口径：

- **整个实验模型 = Cosmos Reasoner**。这里的“整个模型”不是完整 Cosmos 世界模型；Generator、diffusion、VAE 及生成图像/视频/action 的模块一律不属于本实验模型。
- **重要性只在 Reasoner 内计算**。任何 Generator 权重、激活、参数量、Hessian、稀疏率 denominator 和 checkpoint shard 都不得出现。
- **Reasoner 内的默认 SparseGPT 可剪权重**分成两个互不重叠的集合：
  - vision target：`model.visual.encoder.layers.*` 内的 `nn.Linear.weight`；
  - AR target：`model.language_model.layers.*` 内的 `nn.Linear.weight`。
- tokenizer 只是把文本变成 token IDs，不是被剪权重；embedding 负责把 token IDs 变成 AR hidden states。默认沿用 LLaVA 基线，projector、embedding、norm、bias、`lm_head` 保持 dense。
- “AR subsequence 剪除了 vision encoder 的 Reasoner”准确表示：**计算 AR SparseGPT 重要性时，只执行 Reasoner 中去掉 vision 数据流后的 language subsequence**，即 `tokenizer → token embedding → AR transformer`；它不表示最终 checkpoint 删除 vision encoder。最终 checkpoint 仍同时包含已经 image-only SparseGPT 剪过的 vision encoder 和已经 text-only SparseGPT 剪过的 AR transformer。

只允许使用下面这个范围关系：

```text
experiment model = Reasoner

Reasoner
├─ vision encoder Linear weights  ── SparseGPT target
├─ projector / connector           ── dense
├─ token embedding / norms         ── dense
├─ AR transformer Linear weights   ── SparseGPT target
└─ lm_head                         ── dense

Generator / diffusion / VAE        ── outside experiment; never load/count/prune
```

## 2. 权威代码位置与版本指纹

服务器：`root@10.103.92.120:1256`
权威 TAMP 根目录：`/private/workspace/hycui/project/Tamp`
检查时 Git HEAD：`8c044e1`；工作区包含修改和新增文件，因此必须同时用下面的 SHA-256 锁定实际运行版本。

| 角色 | 服务器文件 | SHA-256 |
|---|---|---|
| SparseGPT 数学内核与 LLaVA 两侧 pruner | `llava/pruners/sparsegpt_pruner.py` | `1f0bd3964888d5de927c3ab2d148eb722c0610fbe625889bcc7ba7c3e9f829eb` |
| CLI、pruner 构造和 metadata | `llava/evaluate.py` | `97c0d15120a58ddb2b62a6de30fccdf6b7b71c5bb30b9c723239bd7e787e248c` |
| 校准数据集和 collator | `llava/pruners/data_loader.py` | `977f30321bc14e35b0e20a9b7a29223847534604fac3c7dcefc922437fe9f820` |
| first-layer catcher 公共支持 | `llava/pruners/layer_single_base_pruner.py` | `86adb6365a47efc0863d26177b14e609bc5d0542ef2cc4051521c1cefcb70b42` |
| SparseGPT 4×4 包装入口 | `scripts/prune/sparsegpt_joint_both50_fourcalib_eval.sh` | `420ae6e2f49f34fb3d080800f2d1a9e1cf32ca54aa3197430d089f0af3bedc71` |
| joint 4×4 公共调度器 | `scripts/prune/wanda_joint_both50_fourcalib_eval.sh` | `7484a2f60803a16e97ed440367b2fc5e8d86d8c8ae62dbac24180932156a89d1` |
| checkpoint 合同检查器 | `scripts/prune/check_sparsegpt_joint_pruned_model.py` | `58f91747693244cb6ea1b1c435d4d07257da5a53d637a93b9e5153a7208b8a02` |

### 2.1 LLaVA 三项评测的权威定位

这三项评测不是写在 `llava/evaluate.py` 里的剪枝逻辑，也不是三套独立
模型推理实现。权威链路是：评测 Shell 入口 → lmms-eval 的 `llava` adapter
→ 本地 task YAML/doc adapter/scorer。Cosmos 迁移必须复用相同 task 和 scorer，
只把模型 adapter 换成 Reasoner-only `cosmos3_edge`。

| 评测 | LLaVA Shell 入口 | task/scorer | 样本数与 metric |
|---|---|---|---|
| MMBench | `scripts/eval/mmbench_en_dev_local.sh` | `lmms_tasks/mmbench_en_dev_local.yaml` + `mmbench_local_utils.py` | 4,329；`exact_match_score,none` |
| MMMU | `scripts/eval/mmmu_val_local.sh` | `lmms_tasks/mmmu_local/mmmu_val_local.yaml` + `mmmu_local_utils.py` | 900/30 subjects；`exact_match_score,none` |
| OK-VQA | `scripts/eval/okvqa_val2014_local.sh` | `lmms_tasks/okvqa_local/okvqa_val2014_local.yaml` + `okvqa_local_utils.py` | 5,046；`exact_match,none` |

LLaVA 的模型 adapter 实际文件是环境中的
`lmms_eval/models/simple/llava.py`，SHA-256 为
`d9a81fa9709dafe0bafeb4bb3841d8d6181256deb9de0b7b4d18d8a4a3ab9a32`。
它完成图片预处理、LLaVA conversation template、image token 插入和
`model.generate()`；task YAML 负责 prompt、生成上限和评分。Cosmos adapter
必须用 `AutoProcessor.apply_chat_template()` 处理相同 task 产生的文本与图片，
同时正确保留 MMMU 多图 placeholder 顺序。

冻结的生成协议：batch size 1、temperature 0、num_beams 1、do_sample=false；
MMBench/OK-VQA `max_new_tokens=32`，MMMU `max_new_tokens=16`。评测永远是正常
image+text Reasoner forward，不因 checkpoint 来自 joint 或 separate 而改变。

本地 `TAMP/llava/pruners/sparsegpt_pruner.py` 仍保留一个未使用的 `activation_density` import，哈希与服务器不同；服务器只删除了该无用 import，SparseGPT 数学主体相同。后续迁移必须以服务器哈希为准。

`git diff` 还表明：服务器的 `sparsegpt_pruner.py` 相对 TAMP 基线几乎未改算法，只删除了未使用 import；用户侧新增工作的核心在运行矩阵、checkpoint 校验和 SparseGPT metadata。因此不能把 TAMP/WANDA 的 AMIA、DAS、ATV 逻辑误归入 SparseGPT。

## 3. 完整调用链

```text
scripts/prune/sparsegpt_joint_both50_fourcalib_eval.sh
  └─ 设置 WANDA_MATRIX_ALGORITHM=sparsegpt
     └─ scripts/prune/wanda_joint_both50_fourcalib_eval.sh
        ├─ 校验校准 JSON、task split、图片和评测资产
        ├─ llava/evaluate.py --prune_method sparsegpt
        │  ├─ create_data_loader(...), batch_size=1
        │  ├─ LLaVALayerSparseGPTPruner(...)
        │  └─ pruner.prune()
        │     ├─ VITLayerSparseGPTPruner._prune(...)
        │     │  └─ 每个 ViT block 内每个 nn.Linear → SparseGPT
        │     └─ LLaMALayerSparseGPTPruner._prune(...)
        │        └─ 每个 decoder block 内每个 nn.Linear → SparseGPT
        ├─ check_sparsegpt_joint_pruned_model.py
        └─ 每个 checkpoint 依次跑 mmbench / okvqa / mmmu / mathvista
```

实际 4×4 参数协议：

- 校准源：`mathvista mmbench mmmu okvqa`，每个源独立产生一个 checkpoint，不是把四源拼成 512 样本。
- `nsamples=128`、`seed=42`、`sample_select=random`。
- `calibration_modality=multimodal`。
- `llm_sparsity_ratio=0.5`、`vit_sparsity_ratio=0.5`。
- `sparsity_type=unstructured`、`sparsity_ratio_granularity=none`。
- `token_selection=naive`、`use_variant=false`、无 `sparsity_dict`。
- SparseGPT 内核固定 `blocksize=128`、`percdamp=0.01`。

## 4. 函数级标准定位

以下行号对应服务器权威 `llava/pruners/sparsegpt_pruner.py`。

### 4.1 SparseGPT 内核

| 行号 | 符号 | 准确语义 |
|---|---|---|
| 35-44 | `find_layers()` | 递归寻找 `type(module) is nn.Linear` 的模块。只因调用入口限制在 transformer blocks，才不会剪 projector、embedding、norm、lm_head。 |
| 47-58 | `SparseGPT.__init__()` | 为一个 Linear 分配 `H[columns, columns]`，即按输入维构造完整 Hessian 近似。 |
| 60-72 | `SparseGPT.add_batch()` | Linear 输入展平成 token×hidden，再转为 hidden×token；累计 `H += X X^T`。`nsamples` 按展平前 batch 维更新，不按 token 数更新。 |
| 75-90 | `fastprune()` 前处理 | 权重转 FP32；零 Hessian 对角对应 dead input channel，先把该列权重清零。 |
| 94-155 | Hessian 稳定化/逆分解 | 处理 inf，按 `0.01 * mean(diag)` 加 damping，Cholesky → inverse → upper Cholesky，得到后续误差传播使用的 `Hinv`。 |
| 157-159 | 权重重要性 | `W^2 / diag(Hinv)^2`；这里只把平均值写到 `weight.importance_score`，真正 mask 在后面的 block 内重算。 |
| 163-180 | 非结构化 mask | 以 128 个输入列为一块，在当前块的**所有 output rows × columns 上整体 flatten 排序**；不是 WANDA 的逐 output-row bottom-k。 |
| 183-204 | OBS/SparseGPT 更新 | 按列依次置零并用 `Hinv` 做块内和跨块误差补偿；这是 SparseGPT 与只置 mask 的 magnitude/WANDA 的本质区别。 |
| 211 | 写回权重 | 把重构后的稀疏 FP32 权重转回原 dtype。 |

必须保留的算法区别：SparseGPT 用完整 `H` 和逐列误差补偿；不能把 Cosmos 现有 WANDA 的一维 `sum_sq/scaler_row` 政名后当 SparseGPT。

### 4.2 LLM/AR 侧

| 行号 | 符号 | 准确语义 |
|---|---|---|
| 336-388 | `LLaMALayerSparseGPTPruner.prepare_calibration_input_encoder()` | 临时替换第一个 decoder layer，跑顶层 LLaVA forward，缓存第一层融合 hidden state、attention/position/cache kwargs 和 `model.temp_label`。 |
| 369 | 同上 | 无条件执行 `batch['images'] = ...`；这是当前 text-only SparseGPT 不可用的直接证据。 |
| 390-445 | `LLaMALayerSparseGPTPruner._prune()` | 逐 decoder layer：给该层全部 Linear 注册 hook、累计 Hessian、逐 Linear `fastprune()`、再用已剪层重算输出并传给下一层。 |
| 419-421 | 同上 | 使用缓存的整段融合 hidden state；`image_masks` 虽被收集，但 naive SparseGPT 从未读取。 |

LLM 目标路径：`model.layers`。在 joint 模式中，这个输入同时含 language embeddings 与经 vision tower/projector 插入的 visual embeddings。

### 4.3 Vision 侧

| 行号 | 符号 | 准确语义 |
|---|---|---|
| 609-649 | `VITLayerSparseGPTPruner.prepare_calibration_input_encoder()` | 从顶层模型 forward 进入 vision tower，在第一个 vision encoder layer 截获 patch hidden states 后停止。 |
| 651-701 | `VITLayerSparseGPTPruner._prune()` | 对 vision encoder 逐层、逐 Linear 累计 Hessian、剪枝、重算并传播。 |

Vision 目标路径：`model.vision_tower.vision_tower.vision_model.encoder.layers`。虽然样本 JSON 也含问题文本，但 catcher 在 vision 第一层停止，文本不会进入 ViT 层激活；因此 vision 统计在数据流上是 image-only 局部统计。

### 4.4 joint 总控顺序

| 行号 | 符号 | 准确语义 |
|---|---|---|
| 802-886 | `LLaVALayerSparseGPTPruner.get_sparsity()` | `granularity=none` 时返回统一稀疏率，不运行 DAS/AMIA 分配。 |
| 907-932 | `LLaVALayerSparseGPTPruner.prune()` vision 段 | **先实际剪 ViT**。 |
| 934-959 | 同函数 LLM 段 | 再重新跑多模态 forward，使用已经稀疏的 ViT 产生视觉表示，剪 LLM。 |

因此 LLaVA exact-joint 顺序必须写成：

```text
image-only vision statistics → prune vision weights
                                  ↓
same paired image+text → pruned vision → projector → fused sequence
                                  ↓
                         SparseGPT prune AR/LLM weights
```

若在 dense 模型上同时缓存 vision 与 AR 统计、最后统一落权重，那是另一个消融，不是现有 LLaVA joint 的等价迁移。

## 5. 校准输入的真实路径

`llava/pruners/data_loader.py`：

- `LazySupervisedDataset.__init__()` 约 57-188 行：按 task split 取对应校准源，再用 seed 控制的随机索引取 128 条。
- `_get_item()` 约 286-391 行：读取真实图片，执行 LLaVA multimodal conversation preprocessing。
- `DataCollatorForSupervisedDataset.__call__()` 约 394-451 行：输出 `input_ids/labels/attention_mask/images/image_sizes/modalities`。
- `create_data_loader()` 454-460 行：`batch_size=1`。

`llava/model/llava_arch.py::prepare_inputs_labels_for_multimodal()`：

- 图片经 vision tower 和 projector。
- `<image>` 位置被实际 visual features 替换，形成图文融合的 `inputs_embeds`。
- `model.temp_label` 标记 visual token，`model.temp_attention_mask` 标记有效 token。

但现有 SparseGPT naive kernel 有两个重要事实：

1. LLM catcher 收集 `image_masks`，`SparseGPT.add_batch()` 却完全不接收或使用它，所以 joint 使用全部 visual + language token，符合 joint 定义。
2. 它也不使用 `attention_mask` 排除 padding。当前 batch size=1，样本通常没有跨样本 padding，因此影响有限；迁 Cosmos 时仍应显式使用并记录 valid-token mask，不能把是否过滤 padding 变成未记录的实现差异。

## 6. 已有产物对代码语义的验证

服务器实际产物：

- checkpoint 根目录：`/private/workspace/hycui/model/sparsegpt_joint_both50_pruned`
- 4×4 评测根目录：`/private/workspace/hycui/project/Tamp/results/sparsegpt_joint_both50_fourcalib`
- controller：`/private/workspace/hycui/project/Tamp/results/sparsegpt_joint_both50_controller/state.json`

controller 已记录：4/4 checkpoint、16/16 eval、`phase=complete`。四份 `sparsegpt_metadata.json` 均满足：

- `algorithm=SparseGPT`
- `calibration_modality=multimodal`
- `pure_sparsegpt_contract=true`
- `tamp_components={amia:false,das:false,atv:false}`
- 目标 LLM/ViT sparsity 均为 0.5
- 实际 LLM sparsity 约 `0.50000162-0.50000164`
- 实际 ViT sparsity 约 `0.50000660-0.50000675`

实际略高于 0.5 不是统计误差：当前 mask 使用零基下标 `int(numel*sparsity)` 再做 `<= threshold`，无 ties 时也会多剪一个元素/块。Cosmos 迁移必须显式选择并记录：

- `legacy_llava_threshold`：复现该 off-by-one，获得最大基线一致性；或
- `exact_k_budget`：每块严格剪 K 个，获得严格 50% 零数。

两者不能不标版本地混报。

## 7. separate SparseGPT 的代码边界

### 7.1 TAMP/LLaVA：不存在

对服务器 `scripts/` 和 `llava/` 的文件名、符号和文本搜索均未发现：

- `sparsegpt_separate`
- `sparsegpt_unimodal`
- `sparsegpt_split`
- 负责 image-only ViT + text-only LLM 合并的 SparseGPT orchestrator

现有 `scripts/prune/tamp_separate_*` 和 `atv_separate_*` 属于 TAMP/ATV，不是 pure SparseGPT。不能把它们当成 SparseGPT separate 基线。

### 7.2 ECoFLaP/LAVIS：有 BLIP2-T5 参考，但不是 LLaVA

本地参考文件：

| 角色 | 文件 | SHA-256 |
|---|---|---|
| BLIP2 SparseGPT pruner | `ECoFLaP/LAVIS/lavis/compression/pruners/sparsegpt_pruner.py` | `4f7ad17cf2637465c9c89f404015fe7f6cf6d27b26973183bdf6748020d75fc6` |
| 五校准源 separate 编排 | `scripts/blip2/run_sparsegpt_fivecalib_unimodal_split_then_fourbench_eval.sh` | `feb48f3ec4ea06ffda343211d1cc260a91193ac1325bfeed7baf147654acf290` |
| 单侧/合并入口 | `scripts/blip2/run_sparsegpt_unimodal_split_no_granularity.sh` | `17c5712201f70960134c4484d8d86de385137281fa5d68b445c49778cf74e787` |
| checkpoint 合并工具 | `scripts/blip2/merge_ecoflap_split_prune_ckpts.py` | `154c1056b0498145ded463636bc2f4e173fb896d8aa300bf46e12c20ac6820c5` |

可借用的 only contract：

```text
paired sample
  ├─ text-only JSON  → tokenizer/T5 → prune T5
  └─ image-only JSON → ViT only     → prune ViT
merge disjoint target weights → one full multimodal checkpoint → normal multimodal eval
```

不可照抄：BLIP2 的 `visual_encoder.blocks`、`t5_model.encoder/decoder.block`、loss/importance scope、checkpoint 格式和模型 forward。

## 8. LLaVA → Cosmos3-Edge Reasoner 精确映射

Cosmos 现有、已跑通的定位代码：`/private/workspace/hycui/mfs/cosmos_wanda/cosmos_wanda_prune.py`。

| LLaVA SparseGPT 概念 | Cosmos 对应 |
|---|---|
| 完整图文回答模型 | `Cosmos3EdgeForConditionalGeneration` Reasoner；Generator 完全排除 |
| ViT layers | `model.visual.encoder.layers`，27 层 |
| AR/LLM layers | `model.language_model.layers`，28 层 |
| projector | `model.projector`；参与 joint forward，但默认保持 dense |
| ViT first-layer catcher | 复用 `build_vision_cache()` / `capture_first_layer_inputs()` 的隔离路径 |
| joint AR catcher | 复用 `build_ar_cache(protocol=joint)`，要求真实 vision+projector+AR 调用 |
| separate AR catcher | 复用 `build_ar_cache(protocol=separate)`，直接 `language_model(input_ids, attention_mask)`，vision/projector 调用数必须为 0 |
| WANDA `ActivationStats` | 替换为完整 Hessian `SparseGPTStats`；保留 LayerSample、数据流断言和逐层传播框架 |
| `find_layers()` | 继续只在显式 vision/AR block allow-list 中找 `nn.Linear` |

服务器已有 metadata 审计出的 Cosmos 目标范围：

| 口径 | 数值 |
|---|---:|
| Reasoner 全参数 | 2,435,616,496 |
| Vision layers / Linear | 27 / 162 |
| AR layers / Linear | 28 / 168 |
| Vision 目标 Linear 权重 | 411,070,464 |
| AR 目标 Linear 权重 | 1,409,286,144 |
| 全部目标 Linear 权重 | 1,820,356,608 |
| Dense projector 参数 | 76,692,992 |
| Dense lm_head 参数 | 268,435,456 |
| Generator modules | 0 |

每层实际 Linear 形状：

- Vision：`q/k/v/out_proj [1152,1152]`，`fc1 [4304,1152]`，`fc2 [1152,4304]`。
- AR：`q_proj [2048,2048]`，`k/v_proj [1024,2048]`，`o_proj [2048,2048]`，`fc1 [9216,2048]`，`fc2 [2048,9216]`。

FP32 Hessian 常驻量估计：

- 一个 vision layer 的 6 个 H 合计约 96 MiB。
- 一个 AR layer 的 6 个 H 合计恰为 404 MiB；其中 `fc2` 的 `9216×9216` H 单独约 324 MiB。
- Cholesky/inverse 和 FP32 权重副本会产生额外峰值。实现必须逐层、逐 Linear 及时释放，记录 peak memory；不能为全部 28 层同时保存 H。

## 9. Cosmos 两套 SparseGPT 的冻结定义

### 9.1 `cosmos_sparsegpt_joint_reasoner`

联合剪枝的“联合”是指 AR 重要性来自 Reasoner 的真实图文联合数据流，不是 vision/AR 权重做一次全局排序。

```text
阶段 A：image ──> vision encoder ──> vision Hessian ──> 剪 vision

阶段 B：image ──> 已剪 vision encoder ──> projector ──> visual embeddings ─┐
                                                                          ├─> fused AR sequence
         text ──> tokenizer ──> token embedding ──> language embeddings ──┘
                                              └─> AR Hessian ──> 剪 AR
```

固定执行语义：

1. 用校准图片局部进入 `model.visual.encoder.layers`，对 vision Linear 逐层累计完整 Hessian、执行 SparseGPT，并立即写回 vision 权重。
2. 用同一批、同一顺序、同一 sample ID 的 paired image+text 重新跑完整 Reasoner；视觉表示必须来自第一步已经剪过的 vision encoder。
3. projector 参与这一阶段的数据流，把真实 visual embeddings 送入 AR，但 projector 权重默认不剪。
4. 捕获融合 AR 第一层输入，逐层 SparseGPT `model.language_model.layers`。AR Hessian 必须同时统计有效 visual tokens 和有效 language tokens，只排除 padding。
5. 不允许先在 dense vision 上缓存 AR hidden states；不允许只保留 language tokens；不允许把一次 text-only AR 前向命名为 joint。
6. 整个过程只加载和保存 Reasoner；Generator 不加载、不调用、不计算重要性。

### 9.2 `cosmos_sparsegpt_separate_reasoner`

单模态分开剪枝不是只剪一边，也不是把完整 Reasoner 拆成两个推理模型。它是在同一个 Reasoner 中做两次互相隔离的局部 SparseGPT，再合并两套不相交的稀疏权重。

#### A. Vision encoder：image-only 局部 SparseGPT

```text
image ──> image processor ──> model.visual.encoder.layers ──> stop
                              └─> vision Hessian ──> 只剪 vision encoder Linear

text / projector / AR transformer ──X─> 不进入本分支
```

固定要求：

1. vision encoder 的输入只有真实校准图片产生的 patch hidden states；问题文本不能影响 vision Hessian。
2. 只对 `model.visual.encoder.layers.*` 内的 Linear 累计完整 H、做 SparseGPT 重构并写回。
3. 捕获和逐层传播都局限在 vision encoder；projector forward count 和 AR forward count 必须为 0。
4. 这是一个局部的 vision-only SparseGPT，不允许把 AR loss、AR hidden state 或图文融合表示用于 vision 重要性。

#### B. AR subsequence：language-only 局部 SparseGPT

```text
text ──> language tokenizer ──> input_ids ──> token embedding
                                             └─> model.language_model.layers ──> stop
                                                 └─> AR Hessian ──> 只剪 AR Linear

image / vision encoder output / projector output ──X─> 不进入本分支
```

固定要求：

1. AR 校准输入从 language tokenizer 产生的 `input_ids` 开始，经 Reasoner 自己的 token embedding 形成连续 hidden states，再进入 AR transformer；SparseGPT hook 使用的是各 AR Linear 的真实语言 hidden-state 输入。
2. 不传 `pixel_values`、`image_grid_thw` 或视频输入；文本中不允许 image/video placeholder。
3. 不执行 vision encoder，不读取它预先缓存的输出，也不执行 projector；vision/projector forward count 必须严格为 0。
4. AR sequence 的 visual token 数必须严格为 0；AR Hessian 只由有效 language tokens 构成，padding 必须排除。
5. 只对 `model.language_model.layers.*` 内的 Linear 做 SparseGPT。逻辑上这是“Reasoner 去掉 vision 数据流后的 AR subsequence”；projector 因无视觉输入无法形成合法 Hessian，保持 dense。
6. 绝对禁止从 joint fused hidden states 中事后删 visual tokens来伪造 text-only，也禁止 zero/dummy image 或全零 visual embeddings。

#### C. 合并和最终评测

```text
[image-only SparseGPT 后的 vision encoder]
        + [dense projector / embedding / norm / lm_head]
        + [language-only SparseGPT 后的 AR transformer]
        = 一个完整的 sparse Reasoner checkpoint
```

两个分支必须使用一一对齐的同一批 sample IDs，vision/AR target 集合必须不相交。最终评测恢复正常 `image+text` 完整 Reasoner forward：图片仍通过已剪 vision encoder 和 dense projector，再与文本一起进入已剪 AR transformer。`separate` 只描述校准和重要性统计方式，不表示推理时绕开视觉分支。

这两套实验唯一应变化的是 AR Hessian 是否看过视觉表示；样本 ID、seed、nsamples、像素协议、目标 Linear 白名单、vision/AR sparsity、SparseGPT damping/blocksize/budget mode 都必须一致。

### 9.3 joint 与 separate 的不可混淆对照

| 对照项 | Joint SparseGPT | Separate SparseGPT |
|---|---|---|
| 实验完整模型 | Reasoner | Reasoner |
| Generator | 完全排除 | 完全排除 |
| Vision Hessian | 真实图片、局部 vision forward | 真实图片、局部 vision-only forward |
| Vision 权重目标 | vision encoder Linear | 同一组 vision encoder Linear |
| AR Hessian | visual embeddings + language embeddings | language tokenizer/embedding 路径，visual token=0 |
| AR 是否调用 vision/projector | 必须调用，且使用已剪 vision | 严禁调用 |
| AR 权重目标 | AR transformer Linear | 同一组 AR transformer Linear |
| Projector | joint AR 数据流中执行但保持 dense | AR 校准不执行，最终模型中保持 dense |
| 最终 checkpoint | 一个完整 sparse Reasoner | 一个完整 sparse Reasoner |
| 最终评测 | 正常 image+text | 正常 image+text |
| 唯一核心变量 | AR 重要性看过视觉表示 | AR 重要性完全没看过视觉表示 |

## 10. 迁移时绝对不能错的点

1. **算法不能偷换**：必须累计完整 `H=X X^T` 并执行 SparseGPT 重构/误差传播，不能复用 WANDA 一维 activation norm 后只置零。
2. **整个实验模型只等于 Reasoner**：所有重要性和剪枝目标都必须属于 Reasoner；任何 Generator/diffusion/VAE 参数进入模块清单、Hessian、稀疏率 denominator 或 checkpoint 都必须失败。
3. **joint 顺序**：vision 先落剪枝，再采 AR Hessian；dense vision 上预缓存 AR 属于另一个实验。
4. **separate AR 必须是真正从 tokenizer 开始的 language-only subsequence**：`input_ids → token embedding → AR layers`；不能从 joint hidden state 事后删 visual tokens，不能 dummy/zero image，不能读取 vision cache，也不能给 projector 伪输入。
5. **目标白名单一致**：joint/separate 都只剪相同 162 个 vision Linear 和 168 个 AR Linear；projector/lm_head 等保持 dense。
6. **Shared attention 归属按参数对象**：无参数的 attention/SDPA 算子不计；Reasoner q/k/v/o 计入 AR 或 vision；若同一 Parameter 被别名引用，按 `id(parameter)` 去重。
7. **valid token 口径要记录**：LLaVA legacy kernel 未用 padding mask；Cosmos 应使用 `LayerSample.valid_mask`，并在 metadata 标明 `hessian_token_filter=valid_tokens`。图文有效 token 均保留，不能把 visual token 过滤掉还叫 joint。
8. **稀疏预算要版本化**：LLaVA legacy threshold 有每块多剪一个的 off-by-one；如修成 exact-K，必须改 metadata contract，不能和 legacy 无标识混用。
9. **数值稳定不能照搬无限循环**：LLaVA `while True + bare except` 在 damping 为 0/NaN 或 Cholesky 持续失败时可能永久循环，且含 `pdb.set_trace()`。Cosmos 必须限定 retry 次数、记录实际 damping、对 NaN/inf fail-fast，并提供明确 traceback。
10. **Hessian 生命周期**：每层只保留当前 6 个 Linear 的 H；每个 `fastprune` 后立即 free，层结束后把 hidden states 传给下一层，禁止全模型缓存 H。
11. **重算传播不能省**：每层剪完后必须用已剪层重新计算输出，再喂给下一层；否则后层 Hessian 与 SparseGPT layer-wise 假设不一致。
12. **科学协议不静默改变**：BF16 dense base、0.5/0.5、128、seed42、eager、thinking=false、像素 65536/1048576 均写 metadata。
13. **不要误解“剪除 vision encoder 的 Reasoner”**：它表示 separate AR 的重要性路径排除 vision encoder 输出；最终模型并不删除 vision encoder，而是把 image-only 剪过的 vision 权重与 language-only 剪过的 AR 权重合并回同一个 Reasoner。
14. **对比只能改变一个变量**：joint 与 separate 的目标模块、样本 ID、预算和 SparseGPT 数学必须相同；唯一变化只能是 AR Hessian 是否包含真实视觉表示。

## 11. 实现前后的验收门

### 实现前静态门

- 输出并锁定 330 个目标 Linear 的全名、形状、参数对象 ID 和所属 vision/AR 分组。
- `vision ∩ AR = ∅`，Generator target 数为 0。
- `bash -n`、`py_compile`、参数合同和校准对齐检查全部通过。

### 一层/一样本算法门

- 对人工可逆的小 Linear 验证 H 累计、mask 数量、Cholesky、误差传播和写回 dtype。
- 分别跑 joint/separate 1 样本、vision 1 层、AR 1 层。
- joint AR：vision/projector/AR call 都大于 0，visual/language token 都大于 0。
- separate vision：projector/AR call 为 0。
- separate AR：vision/projector call 为 0，visual token 为 0，language token 大于 0。

### checkpoint 门

- 每个目标 Linear 记录 target/actual sparsity、H 样本数、有效 token 数、damping/retry、非有限值计数。
- vision、AR、目标总计和整个 Reasoner 分口径报告 zero ratio。
- projector、embedding、norm、lm_head 保持 dense；Generator 不存在。
- 保存后重新加载，运行正常 image+text Reasoner forward，logits 有限、输出非空。
- 只有上述门通过后，才允许扩为 MMBench/MMMU/OK-VQA 三校准源 × joint/separate × 三评测矩阵。

## 12. 服务器迁移代码的实现边界

服务器迁移使用独立目录 `/private/workspace/hycui/mfs/cosmos_sparsegpt`，不把 SparseGPT 混入已经产生正式 WANDA 结果的 `cosmos_wanda` 运行目录。代码层面的复用边界为：

- 复用 `cosmos_wanda_prune.py` 已验证的校准 record、paired 对齐、processor、first-layer catcher、joint/separate dataflow assertions、module audit、逐层 cache 传播、保存与 reload 验证。
- 在 `sparsegpt_core.py` 新增 `SparseGPTStats` 和有界数值稳定版 `sparsegpt_prune_linear()`，替换 `ActivationStats + prune_linear_weight()`。
- 算法名、metadata schema、输出根目录、controller task 名全部使用 `sparsegpt`，绝不覆盖或复用 WANDA checkpoint。

实现提供两种明确版本：正式默认 `exact_k_budget` 严格执行每块 K 个权重预算；`legacy_llava_threshold` 仅用于复现 LLaVA 轻微超 50% 的旧阈值行为。实际模式必须写入 metadata，不能混报。

主要实现文件：

- `/private/workspace/hycui/mfs/cosmos_sparsegpt/cosmos_sparsegpt_prune.py`
- `/private/workspace/hycui/mfs/cosmos_sparsegpt/sparsegpt_core.py`
- `/private/workspace/hycui/mfs/cosmos_sparsegpt/test_sparsegpt_core.py`
- `/private/workspace/hycui/mfs/cosmos_sparsegpt/validate_cosmos_checkpoint.py`
- `/private/workspace/hycui/mfs/cosmos_sparsegpt/prepare_full_matrix.py`
- `/private/workspace/hycui/mfs/cosmos_sparsegpt/run_full_matrix_task.py`
- `/private/workspace/hycui/mfs/cosmos_sparsegpt/run_full_matrix_worker.sh`

正式矩阵启动前必须先通过 CPU kernel test、`py_compile`、`bash -n`、三数据集 joint/separate preflight，以及 joint/separate 各一个 1-sample/1-layer GPU smoke。
