# TAMP 多选评测 vs 当前 MMMU 评测 对比

## TAMP 里 LLaVA-NeXT 怎么做 MMMU evaluation

TAMP 仓库里**没有**单独的「MMMU 数据集」脚本，但 **LLaVA-NeXT 做 MMMU 评测**走的是同一套通用流程：

1. **推理**：`llava/eval/model_vqa.py`  
   - 读入 **question 文件**（如 `tables/question.jsonl`，来自 [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) 或自建的 MMMU→LLaVA 格式）。  
   - 每行有 `metadata.dataset`、`metadata.question_type`、`conversations`（gt 在 `[1]["value"]`）、`image` 等。  
   - 若 question 里包含 MMMU（dataset 名为 `"MMMU"` 或按学科），就会生成对应的 `pred_response` / `gt_response`。

2. **结果文件**：推理时写入的 answers 文件（如 `answer.jsonl`）每行包含：  
   `dataset`, `sample_id`, `pred_response`, `gt_response`, `question_type`。  
   评测时通常把该文件放到某 `result-dir` 下并命名为 `result.jsonl`。

3. **打分**：`llava/eval/evaluate_interleave.py`  
   - 读 `result-dir/result.jsonl`，按 `dataset` 分组。  
   - 对每个 dataset（**包括 MMMU**）：若 `question_type == 'multi-choice'` 且不在 `image_choice_dataset_list`，则用 **`evaluate_multichoice`**。  
   - MMMU 不在 `image_choice_dataset_list`（该列表只有 RecipeQA_VisualCloze 等），所以 **MMMU 用的就是 evaluate_interleave 里的多选逻辑**：process(gt)、process(pred)、从 pred 按 `":"` 抽第一个单字母 a–h、再 `pred_ans == gt_ans`。

因此：**TAMP 里“LLaVA-NeXT 做 MMMU evaluation”= model_vqa.py 推理 + evaluate_interleave.py 的多选分支**；没有单独的 MMMU 脚本。下面把这一套和「当前 MMMU 脚本」以及「VideoMME」的差异都列出来。

---

## 1. 数据与流程

| 项目 | TAMP (evaluate_interleave / eval_video_mcqa_videomme) | 当前 MMMU (mmmu_eval_by_discipline.py) |
|------|--------------------------------------------------------|----------------------------------------|
| 数据来源 | result.jsonl（含 gt_response / pred_response）或 VideoMME 格式 | MMMU_single_image parquet，单图 test |
| GT 形式 | 多选：gt 为选项字母或选项文本 | GT 取 `answer` 字段首字符，仅保留 A/B/C/D |
| 是否只评估「有答案」样本 | **VideoMME：是**（见下） | **否**：所有有效 gt 样本都进分母 |

---

## 2. GT（标准答案）的处理

### TAMP · evaluate_interleave.py（多选）

- 先对 **gt_response** 做 **process()**：
  - 换行/制表符 → 空格
  - **processPunctuation**：去掉或替换标点（`; / [ ] " { } ( ) = + \ _ - > < @ ` , ? !` 等）
  - 去掉首尾 `' " ) (`
  - 最后 **.strip().lower()**
- 多选时 gt 一般是 `"a"` / `"b"` 等，process 后仍为小写单字母。

### TAMP · eval_video_mcqa_videomme.py

- **不做** process：直接用 `question[gt_answer_key]`（即 `"answer"` 字段），一般为 **"A"/"B"/"C"/"D"**。
- 比较时：`extration == gt_answer`（大小写取决于数据集里存的格式）。

### 当前 MMMU 脚本

- GT：`raw_answer = str(row.get("answer","")).strip().upper()`，再 `answer_letter = raw_answer[0] if raw_answer[0] in "ABCD" else ""`。
- **不做** processPunctuation，不做小写；只取首字母并统一成大写，非 A/B/C/D 的样本跳过（不进入统计）。

**差异小结**：  
- TAMP interleave 会对 gt 做**标点清洗 + 小写**；VideoMME 和当前 MMMU 都不对 gt 做 process，只当「选项字母」用。  
- 当前和 VideoMME 更接近；和 interleave 的差别在「gt 是否 process」。

---

## 3. 从模型输出（pred）里提取选项字母

### TAMP · evaluate_interleave.py

- 先对 **pred_response** 做 **process()**（同上：标点、首尾符号、小写）。
- 再从 process 后的 pred 里取「选项字母」：
  - 若 pred 里包含 **":"**，则按 `":"` 分割，取**第一个**「长度为 1 且字符 ∈ [a,b,c,d,e,f,g,h]」的 token 作为 pred_ans。
  - 否则整段 pred 当作 pred_ans（可能是一整句话）。
- 比较：**pred_ans == gt_ans**（两者都已 process，都是小写）。

特点：**先整段规范化，再按 ":" 找单字母**；支持 **a–h 共 8 个选项**。

### TAMP · eval_video_mcqa_videomme.py · extract_characters_regex(s)

```python
s = s.strip()
# 前缀：任意位置 replace 掉（不是仅开头）
answer_prefixes = [
    "The best answer is", "The correct answer is", "The answer is",
    "The answer", "The best option is" "The correct option is",
    "Best answer:" "Best option:",
]
for answer_prefix in answer_prefixes:
    s = s.replace(answer_prefix, "")

if len(s.split()) > 10 and not re.search("[ABCD]", s):
    return ""
matches = re.search(r'[ABCD]', s)
return matches[0] if matches else ""
```

- 前缀是**在整句里 replace 掉**（不是仅开头）。
- 若句子 **>10 词且没有 [ABCD]**，返回 `""`。
- 否则用正则 **第一个 [ABCD]**（大写）作为选项字母；**不支持 E/F/G/H**。

### 当前 MMMU · extract_answer_letter(pred)

- `s = pred.strip()`。
- 只对**开头**做前缀剥离：`if s.lower().startswith(prefix): s = s[len(prefix):].strip()`，前缀列表类似（the best answer is, the correct answer is, the answer is, the answer, best answer:, best option:, correct answer:）。
- 若 `len(s.split()) > 10` 且没有 `[ABCD]`，返回 `""`。
- 用 `re.search(r'[ABCD]', s, re.IGNORECASE)` 取**第一个** A/B/C/D，再 `.upper()`。

**差异小结**：

| 项目 | TAMP interleave | TAMP VideoMME | 当前 MMMU |
|------|-----------------|---------------|-----------|
| 是否先对 pred 做 process（标点、小写） | 是 | 否 | 否 |
| 前缀 | 无 | 整句 replace | 仅开头 startswith 去掉 |
| 选项字母提取 | 按 ":" 拆，取单字母 a–h | 正则第一个 [ABCD] | 正则第一个 [ABCD]（不区分大小写） |
| 选项范围 | a–h | A–D | A–D |

---

## 4. 统计方式（谁进分母）

### TAMP · eval_video_mcqa_videomme.py

```python
if extration != "":
    ...["answered"] += 1
    ...["correct"] += (extration == gt_answer)
```

- 只有 **extraction 非空**（即模型输出里抽到了 A/B/C/D）时，才计入 **answered** 和 **correct**。
- 准确率 = correct / **answered** → **「未抽到选项字母」的样本不进入分母**。

### 当前 MMMU

- 只要 **gt_letter 有效**（A/B/C/D），就 `total_count += 1`，并 `match = 1 if (pred_letter and pred_letter == gt_letter) else 0`。
- 准确率 = overall_correct / **total_count** → **所有有效 GT 样本都进分母**；pred 抽不到字母算错（match=0）。

**差异小结**：  
- **TAMP VideoMME**：分母 = 「模型至少输出了一个可识别的 A/B/C/D」的样本数。  
- **当前 MMMU**：分母 = 所有「GT 为 A/B/C/D」的样本数。  
→ 在「很多题模型没输出 A/B/C/D」时，当前方式会**压低准确率**；TAMP VideoMME 会**只算「有答案」的题**，数字会更高。

---

## 5. 逐项对照表

| 维度 | TAMP (interleave 多选) | TAMP (VideoMME) | 当前 MMMU |
|------|------------------------|-----------------|-----------|
| GT 是否 process（标点+小写） | 是 | 否 | 否 |
| GT 形式 | process 后整段或单字母 | 原始 answer 字母 | 首字母且 ∈ A/B/C/D |
| Pred 是否 process | 是 | 否 | 否 |
| Pred 选项提取 | 按 ":" 取单字母 a–h | 整句 replace 前缀 + 正则 [ABCD] | 仅开头去前缀 + 正则 [ABCD] |
| 前缀处理 | 无 | 整句 replace | 仅开头 startswith |
| 选项范围 | a–h | A–D | A–D |
| 分母 | 全部预测样本 | 仅「抽到字母」的样本 | 全部有效 GT 样本 |
| 抽不到字母 | 算错（整段 pred 与 gt 比） | **不进入统计** | 算错 |

---

## 6. 总结：主要不同点

1. **分母是否含「未抽到选项」**  
   - TAMP VideoMME：**不含**，只对「模型输出了 A/B/C/D」的题算准确率。  
   - 你当前：**含**，没抽到字母直接算错并进分母。  
   → 若希望和 TAMP VideoMME 一致，需要改成：只有 `pred_letter != ""` 时才计入 denominator（或单独报「answered 准确率」和「overall 准确率」）。

2. **前缀处理**  
   - TAMP VideoMME：`s.replace(prefix, "")`，**整句**任意位置都删。  
   - 你当前：`s.lower().startswith(prefix)`，只删**句首**。  
   → 若模型经常在中间写 "The answer is A"，你当前会保留 "The answer is A"，仍能抽到 A；若希望和 TAMP 完全一致，可改为整句 replace 前缀。

3. **GT / pred 是否做 process**  
   - TAMP interleave：gt 和 pred 都做 process（标点+小写），再比或再抽字母。  
   - TAMP VideoMME 和你当前：都不对 gt/pred 做 process，只做「选项字母」提取与比较。  
   → 当前和 VideoMME 更接近；若 MMMU 的 answer 有时带标点/空格，可以考虑对 gt 做一次轻量 strip/lower 再取首字母。

4. **TAMP 里 MMMU 用哪套**  
   - **MMMU** 在 TAMP/LLaVA-NeXT 里走的是 **evaluate_interleave** 的多选逻辑（process + 按 ":" 抽字母 a–h），**不是** VideoMME 的 extract_characters_regex。  
   - VideoMME 是另一套（eval_video_mcqa_videomme.py），分母只含「抽到字母」的样本。

如需，我可以按上述 1～3 点直接改一版 `mmmu_eval_by_discipline.py`（例如：分母改为仅 answered，前缀改为 replace，可选对 gt 做 strip/lower）。

---

## 7. 还可能存在的其他问题（已排查/已修）

- **题干/选项预处理把小数点去掉**  
  `BlipQuestionProcessor.pre_question` 会去掉 `.!\"()*#:;~`，导致选项里的 "6.33" 变成 "633"、"A. Option" 里的点也被删。  
  **已修**：处理器增加 `remove_punctuation` 参数（默认 True 保持原行为）；MMMU 脚本里用 `remove_punctuation=False`，只做 lower + strip + max_words，保留小数和选项结构。

- **输入长度**  
  推理路径 `predict_answers` 里 T5 tokenizer 用 `padding="longest"` 且未设 `max_length`，不会截断；题干+选项当前 p99 约 674 字符，远低于 T5 的 512 token，暂无问题。若以后题干更长，需留意 T5 encoder 的 max_position_embeddings。

- **batch_size / OOM**  
  若显存不足可把 `--batch_size` 调小（如 8 或 4）。

- **E/F/G/H 选项**  
  当前数据已过滤为仅 A/B/C/D；若将来用未过滤 parquet，GT 的 `normalize_gt_to_option_letter` 只把首字母 A–D 规范成单字母，E–H 会走 `process_answer(gt_raw)`，仍能得到 "e"/"f" 等；pred 端已支持 a–h 抽取，逻辑一致。

- **学科未映射**  
  新 subject 会进 "Other"，不影响正确性，只影响分领域统计的命名。
