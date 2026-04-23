#!/usr/bin/env python3
"""
MMMU 单图 test 集上跑 BLIP2-T5 推理，并按 6 大领域汇总准确率。

依赖：在 ECoFLaP/LAVIS 的 conda 环境中运行（需 torch, lavis, transformers, pandas 等）。

用法（在 ECoFLaP/LAVIS 目录下）:
  python scripts/blip2/mmmu_eval_by_discipline.py \\
    --mmmu_root /root/autodl-tmp/MMMU_single_image \\
    --split test \\
    [--ckpt pruned_checkpoint/okvqa_ghlc-xxx.pth] \\
    [--batch_size 16] [--device cuda]

- 从 MMMU_single_image 读 test parquet，只评估单图题。
- 若提供 --ckpt 则加载剪枝后的模型，否则用 LAVIS 预训练权重。
- 输出：Overall 准确率 + 6 大领域各自准确率。
"""
from __future__ import annotations

import argparse
import ast
import io
import os
import re
import sys

import pandas as pd
import torch
from PIL import Image

# 保证从 LAVIS 根目录运行时能 import lavis
_LAVIS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _LAVIS_ROOT not in sys.path:
    sys.path.insert(0, _LAVIS_ROOT)

from lavis.models import load_model
from lavis.processors import load_processor


# 与 mmmu_to_calibration_format.py 保持一致
DISCIPLINES = {
    "Art & Design": ["Art", "Art_Theory", "Design"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": ["Agriculture", "Biology", "Chemistry", "Geography", "Math", "Materials", "Physics"],
    "Health & Medicine": ["Basic_Medical_Science", "Clinical_Medicine", "Diagnostics_and_Laboratory_Medicine", "Pharmacy", "Public_Health"],
    "Humanities & Social Science": ["History", "Literature", "Music", "Psychology", "Sociology"],
    "Tech & Engineering": ["Architecture_and_Engineering", "Computer_Science", "Electronics", "Energy_and_Power", "Mechanical_Engineering"],
}


def subject_to_discipline(subject: str) -> str:
    for disc, subs in DISCIPLINES.items():
        if subject in subs:
            return disc
    return "Other"


def has_bytes(val):
    return isinstance(val, dict) and val.get("bytes") is not None


def count_images(row):
    n = 0
    for col in [f"image_{i}" for i in range(1, 8)]:
        if has_bytes(row.get(col)):
            n += 1
    return n


def get_first_image_bytes(row):
    for col in [f"image_{i}" for i in range(1, 8)]:
        val = row.get(col)
        if has_bytes(val):
            return val["bytes"]
    return None


def parse_options(options_val):
    if options_val is None:
        return []
    if isinstance(options_val, list):
        return [str(x) for x in options_val]
    s = str(options_val).strip()
    if not s:
        return []
    try:
        out = ast.literal_eval(s)
        return [str(x) for x in out] if isinstance(out, list) else []
    except (ValueError, SyntaxError):
        return []


def answer_letter_to_text(answer_letter, options_list):
    letter = str(answer_letter).strip().upper()
    if not letter or not options_list:
        return str(answer_letter).strip()
    idx = ord(letter[0]) - ord("A")
    if 0 <= idx < len(options_list):
        return options_list[idx]
    return str(answer_letter).strip()


def format_options_for_prompt(options_val):
    """把 parquet 的 options 列格式化为选项文本 + 字母列表。
    Returns (formatted_str, option_letters_str)，例如 ("A. x B. x", "A, B, C, D")，无选项时 ("", "")。
    """
    opts = parse_options(options_val)
    if not opts:
        return "", ""
    opts = [str(x).strip() for x in opts]
    formatted = " ".join("%s. %s" % (chr(65 + i), x) for i, x in enumerate(opts))
    letters = ", ".join(chr(65 + i) for i in range(len(opts)))
    return formatted, letters


def normalize_answer(s: str) -> str:
    """Lowercase, strip, collapse whitespace for matching."""
    if not isinstance(s, str):
        s = str(s)
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


# --- TAMP-style answer processing & multi-choice judging ---

_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
_PUNCT = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]


def process_punctuation(in_text: str) -> str:
    """Match TAMP evaluate_interleave.processPunctuation."""
    out_text = in_text
    for p in _PUNCT:
        if (p + " " in in_text or " " + p in in_text) or (
            re.search(_COMMA_STRIP, in_text) is not None
        ):
            out_text = out_text.replace(p, "")
        else:
            out_text = out_text.replace(p, " ")
    out_text = _PERIOD_STRIP.sub("", out_text)
    return out_text


def process_answer(answer: str) -> str:
    """TAMP-style规范化：去换行/制表符/标点，strip 引号与括号，小写。"""
    if answer is None:
        answer = ""
    answer = str(answer)
    answer = answer.replace("\n", " ")
    answer = answer.replace("\t", " ")
    answer = answer.strip()
    answer = process_punctuation(answer)
    answer = answer.strip("'")
    answer = answer.strip('"')
    answer = answer.strip(")")
    answer = answer.strip("(")
    answer = answer.strip().lower()
    return answer


def normalize_gt_to_option_letter(gt_raw: str) -> str:
    """把 parquet 的 answer 规范成单字母，与 TAMP question.json 多选 GT 一致。
    parquet 可能是 "A" / "A." / "A. Baroque" 等，这里统一成首字母 A–H（小写），
    与 pred 端 a–h 抽取一致；否则返回 process 后的整段（兼容非选项格式）。
    """
    s = str(gt_raw or "").strip()
    if not s:
        return ""
    first = s[0].upper()
    if first in "ABCDEFGH":
        return first.lower()
    # 非 A–H 的样本（若未过滤）仍走整段 process
    return process_answer(gt_raw)


def judge_multi_choice_tamp_style(gt_raw: str, pred_raw: str) -> int:
    """与 TAMP evaluate_interleave 的多选逻辑对齐的判定函数."""
    # GT：先规范成单字母再 process，避免 parquet 里 "A. Baroque" 导致 gt_ans="a baroque" 与 pred_ans="a" 不匹配
    gt_normalized = normalize_gt_to_option_letter(gt_raw)
    gt_ans = process_answer(gt_normalized) if gt_normalized else process_answer(gt_raw)
    pred_ans = process_answer(pred_raw)

    if not gt_ans:
        # 与 TAMP evaluate_rouge 保持一致：空 GT 直接跳过；上层不计入分母
        return -1

    # 按 ":" 抽第一个 a–h 单字母；若无冒号则从整段中抽第一个 a–h，避免 "the answer is a" 整段与 "a" 不相等
    if ":" in pred_ans:
        parts = [a.strip() for a in pred_ans.split(":")]
        for a in parts:
            if len(a) == 1 and a in ["a", "b", "c", "d", "e", "f", "g", "h"]:
                pred_ans = a
                break
    else:
        m = re.search(r"[a-h]", pred_ans)
        if m:
            pred_ans = m.group(0)

    return 1 if pred_ans == gt_ans else 0


def extract_answer_letter(pred: str) -> str:
    """Extract option letter A/B/C/D from model output (TAMP-style, see eval_video_mcqa_videomme.py).
    Returns single letter or "" if not found.
    """
    if not pred:
        return ""
    s = pred.strip()
    answer_prefixes = [
        "the best answer is",
        "the correct answer is",
        "the answer is",
        "the answer",
        "best answer:",
        "best option:",
        "correct answer:",
    ]
    for prefix in answer_prefixes:
        if s.lower().startswith(prefix):
            s = s[len(prefix):].strip()
    if len(s.split()) > 10 and not re.search(r"[ABCD]", s, re.IGNORECASE):
        return ""
    m = re.search(r"[ABCD]", s, re.IGNORECASE)
    return m.group(0).upper() if m else ""


def load_mmmu_single_image_test(mmmu_root: str, split: str = "test"):
    """Yield (subject, sample_id, image_bytes, question, answer_raw, options_raw) for each single-image sample.
    answer_raw: 原始 GT 字段；options_raw: 原始 options 列，用于拼进 prompt。
    """
    subjects = sorted(
        d
        for d in os.listdir(mmmu_root)
        if os.path.isdir(os.path.join(mmmu_root, d)) and not d.startswith(".")
    )
    for subject in subjects:
        subj_dir = os.path.join(mmmu_root, subject)
        for fname in os.listdir(subj_dir):
            if not fname.endswith(".parquet"):
                continue
            file_split = fname.split("-")[0].lower()
            if file_split != split.lower():
                continue
            path = os.path.join(subj_dir, fname)
            df = pd.read_parquet(path)
            for _, row in df.iterrows():
                if count_images(row) != 1:
                    continue
                img_bytes = get_first_image_bytes(row)
                if img_bytes is None:
                    continue
                sample_id = str(row.get("id", "")).strip()
                raw_answer = str(row.get("answer", "")).strip()
                question = str(row.get("question", "")).strip()
                question = re.sub(r"\s*<image\s*\d*>\s*", " ", question, flags=re.IGNORECASE).strip()
                options_raw = row.get("options")
                yield subject, sample_id, img_bytes, question, raw_answer, options_raw


def main():
    parser = argparse.ArgumentParser(description="MMMU single-image test eval by 6 disciplines")
    parser.add_argument("--mmmu_root", default="/root/autodl-tmp/MMMU_single_image", help="MMMU_single_image root")
    parser.add_argument("--split", default="test", choices=["dev", "validation", "test"], help="Split to evaluate")
    parser.add_argument("--ckpt", default=None, help="Optional: path to pruned .pth (full state_dict)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for inference")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap number of samples (for debugging)")
    parser.add_argument(
        "--overall_only",
        action="store_true",
        help="全量单遍评测，只打印 Overall（不打印按领域分解）",
    )
    args = parser.parse_args()

    # 收集 test 样本
    samples = list(load_mmmu_single_image_test(args.mmmu_root, args.split))
    if not samples:
        print("No single-image samples found for split %r in %s" % (args.split, args.mmmu_root))
        return
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
        print("Capped to %d samples (--max_samples)" % len(samples))
    print("Loaded %d single-image %s samples" % (len(samples), args.split))

    # 加载模型与处理器（与 okvqa eval 一致：blip2_t5 + blip_image_eval 224 + blip_question）
    #
    # UKMP 的 ukmp_prune.py 保存格式是 torch.save({"model": pruned_model_obj}, path)
    # 其中 "model" 是一个已被剪枝过、tensor 维度已经改变的 nn.Module。
    # 直接走 load_model(... checkpoint=...) 会把 checkpoint["model"] 当成 state_dict，
    # 从而触发:
    #   - Expected state_dict to be dict-like
    # 或 size mismatch。
    # 因此这里对 UKMP pruned .bin 做兼容加载：如果 checkpoint['model'] 是 nn.Module，
    # 就直接使用它作为评测模型；否则退回到原始 load_model 逻辑。
    if args.ckpt is None:
        model = load_model(
            "blip2_t5",
            "pretrain_flant5xl",
            is_eval=True,
            device=args.device,
        )
    else:
        # UKMP 的剪枝权重保存为 torch.save({"model": pruned_model_obj}, path)；
        # pruned_model_obj 的某些生成/解码相关组件可能没有 base 模型完整初始化，
        # 直接用它做 predict_answers 可能导致空输出，影响准确率。
        # 因此这里采用“先构建 base 模型（带 tokenizer/生成配置），再替换剪枝后的子模块”的方式，
        # 与 UKMP 的 evaluate_blip2_pruned.py 处理逻辑保持一致。
        ckpt_obj = torch.load(args.ckpt, map_location="cpu")
        if (
            isinstance(ckpt_obj, dict)
            and "model" in ckpt_obj
            and isinstance(ckpt_obj["model"], torch.nn.Module)
        ):
            pruned_model = ckpt_obj["model"]
            model = load_model(
                "blip2_t5",
                "pretrain_flant5xl",
                is_eval=True,
                device=args.device,
            )
            # 替换被剪枝的模块；pruned_model 内部维度已经变化，
            # base_model 会因此在后续 forward/generate 走剪枝后的子模块。
            if hasattr(model, "visual_encoder") and hasattr(pruned_model, "visual_encoder"):
                if hasattr(pruned_model.visual_encoder, "blocks"):
                    model.visual_encoder.blocks = pruned_model.visual_encoder.blocks
            if hasattr(model, "t5_model") and hasattr(pruned_model, "t5_model"):
                model.t5_model = pruned_model.t5_model
            model.eval()
            model.to(args.device)
        else:
            model = load_model(
                "blip2_t5",
                "pretrain_flant5xl",
                is_eval=True,
                device=args.device,
                checkpoint=args.ckpt,
            )
    # MMMU 与 TAMP 一致：题干+选项不设长度上限，由模型/encoder 自然长度限制
    # remove_punctuation=False 保留选项中的小数点和数字（如 6.33、$759,000），避免 pre_question 把 "6.33" 变成 "633"
    vis_processor = load_processor("blip_image_eval").build(image_size=224)
    text_processor = load_processor("blip_question").build(max_words=99999, remove_punctuation=False)
    max_len, min_len, num_beams = 10, 1, 5

    # 按 batch 推理
    correct_by_disc = {d: 0 for d in DISCIPLINES}
    correct_by_disc["Other"] = 0
    total_by_disc = {d: 0 for d in DISCIPLINES}
    total_by_disc["Other"] = 0
    overall_correct = 0
    total_count = 0

    for i in range(0, len(samples), args.batch_size):
        batch = samples[i : i + args.batch_size]
        images = []
        questions = []
        for _, _, img_bytes, question, _, options_raw in batch:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            images.append(vis_processor(img))
            opts_str, option_letters = format_options_for_prompt(options_raw)
            full_question = question + (" Options: " + opts_str if opts_str else "")
            if option_letters:
                full_prompt = "Question: %s Answer with one letter only (%s):" % (full_question, option_letters)
            else:
                full_prompt = "Question: %s Short answer:" % full_question
            questions.append(text_processor(full_prompt))
        image_tensor = torch.stack(images).to(args.device)
        with torch.no_grad():
            preds = model.predict_answers(
                {"image": image_tensor, "text_input": questions},
                num_beams=num_beams,
                max_len=max_len,
                min_len=min_len,
                inference_method="generate",
            )
        for (subject, _, _, _, gt_raw, _), pred in zip(batch, preds):
            score = judge_multi_choice_tamp_style(gt_raw, pred)
            if score == -1:
                # 空 GT，跳过（不计入分母）
                continue
            total_count += 1
            overall_correct += score
            if not args.overall_only:
                disc = subject_to_discipline(subject)
                total_by_disc[disc] = total_by_disc.get(disc, 0) + 1
                correct_by_disc[disc] = correct_by_disc.get(disc, 0) + score

    total = total_count
    overall_acc = 100.0 * overall_correct / total if total else 0
    print("\n===== MMMU single-image %s (n=%d) =====" % (args.split, total))
    print("Overall accuracy: %.2f%%" % overall_acc)
    if not args.overall_only:
        print("\nBy discipline:")
        for disc in list(DISCIPLINES.keys()) + ["Other"]:
            n = total_by_disc.get(disc, 0)
            if n == 0:
                continue
            acc = 100.0 * correct_by_disc.get(disc, 0) / n
            print("  %s: %.2f%% (%d)" % (disc, acc, n))
    print("")

    _mp = os.environ.get("LAVIS_METRICS_JSONL")
    if _mp:
        import json

        _bench = os.environ.get("LAVIS_METRICS_BENCHMARK", "MMMU_parquet")
        _calib = os.environ.get("LAVIS_EVAL_CALIB_TAG", "")
        rec = {
            "calib_tag": _calib,
            "benchmark": _bench,
            "split": args.split,
            "metric": "overall_accuracy_percent",
            "value": round(overall_acc, 4),
            "n": int(total),
            "mmmu_root": os.path.abspath(args.mmmu_root),
        }
        with open(_mp, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
