#!/usr/bin/env python3
"""Generate calibration hints with an OpenAI vision model.

Example:

  export OPENAI_API_KEY="..."
  python scripts/blip2/generate_calibration_hints_openai.py \
    --calib_json /data/data2/mfs/MMMU_calibration/mmmu_calibration_train.json \
    --images_dir /data/data2/mfs/MMMU_calibration/images \
    --out_hints_jsonl /data/data2/mfs/MMMU_calibration/mmmu_openai_hints.jsonl \
    --out_calib_json /data/data2/mfs/MMMU_calibration/mmmu_calibration_with_openai_hint.json \
    --max_samples 128 \
    --add_hint_metadata

The answer is never sent to the API. It is used locally only to reject hints
that directly reveal the ground-truth answer.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import mimetypes
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SYSTEM_PROMPT = """You write supplementary hints for visual question answering calibration data.

A useful hint is an extension of the question, not an answer and not a generic image caption. It should:
- clarify the exact intent of the question and what must be determined;
- identify concrete visual evidence in the supplied image that should be inspected;
- explain how the relevant labels, values, regions, objects, or relationships connect to the requested result.

Do not solve the problem, calculate the result, select a choice, reveal the answer, or provide chain-of-thought reasoning.
Do not merely repeat or paraphrase the question.
Do not invent details that are not visible in the image.
Return only the hint text. Multiple sentences are allowed."""


PROMPT_NO_OPTIONS = """Create an informative supplementary hint for the visual question below.

Inspect the image carefully. Clarify what the question is really asking, point to the specific image evidence that matters, and explain what relationship should be considered before answering. Use concrete, question-specific details rather than generic phrases.

Do not answer the question. Do not give a final value, class, identity, or conclusion. Do not repeat the question. Do not describe unrelated parts of the image.

Question:
{question}

Hint:"""


PROMPT_WITH_OPTIONS = """Create an informative supplementary hint for the visual multiple-choice question below.

Inspect the image carefully. Use the question and choices only to understand the task and expected answer type. Clarify what the question is really asking, point to the specific image evidence that matters, and explain what relationship should be considered before answering. Use concrete, question-specific details rather than generic phrases.

Do not answer the question. Do not mention option letters, select a choice, copy a choice, or reveal the correct choice. Do not give a final value, class, identity, or conclusion. Do not repeat the question. Do not describe unrelated parts of the image.

Question:
{question}

Choices:
{options}

Hint:"""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate image-aware calibration hints with the OpenAI Responses API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--calib_json", required=True, help="Input calibration JSON list.")
    ap.add_argument(
        "--images_dir",
        default=None,
        help="Image directory. Defaults to sibling images/ next to --calib_json.",
    )
    ap.add_argument("--out_hints_jsonl", required=True, help="Per-attempt audit JSONL output.")
    ap.add_argument("--out_calib_json", required=True, help="Calibration JSON with accepted hints.")
    ap.add_argument("--model", default="gpt-5-mini", help="Vision-capable OpenAI model.")
    ap.add_argument(
        "--api_key_env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key.",
    )
    ap.add_argument(
        "--base_url",
        default=None,
        help="Optional OpenAI-compatible API base URL.",
    )
    ap.add_argument(
        "--image_detail",
        choices=["auto", "low", "high"],
        default="high",
        help="Image detail sent to the vision model.",
    )
    ap.add_argument(
        "--max_output_tokens",
        type=int,
        default=400,
        help="Maximum API output-token budget for one hint.",
    )
    ap.add_argument(
        "--hint_min_words",
        type=int,
        default=12,
        help="Reject shorter hints and regenerate them.",
    )
    ap.add_argument(
        "--max_hint_attempts",
        type=int,
        default=3,
        help="Content-generation attempts before keeping the original row unchanged.",
    )
    ap.add_argument(
        "--request_retries",
        type=int,
        default=5,
        help="API retries for a temporary request failure.",
    )
    ap.add_argument(
        "--retry_base_seconds",
        type=float,
        default=2.0,
        help="Base delay for exponential API retry backoff.",
    )
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--question_field", default="question")
    ap.add_argument("--image_field", default="image")
    ap.add_argument("--answer_field", default="answer")
    ap.add_argument(
        "--append_template",
        default="{question}\nHint: {hint}",
        help="Template for accepted hints. Available fields: question, hint.",
    )
    ap.add_argument(
        "--add_hint_metadata",
        action="store_true",
        help="Add generated_hint/status/reasons to output calibration rows.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Reuse successful row_index entries already present in the audit JSONL.",
    )
    ap.add_argument("--log_every", type=int, default=20)
    return ap.parse_args()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_rows(path: str, max_samples: Optional[int]) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise TypeError("Input calibration JSON must be a list of objects.")
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError("Row %d is not a JSON object." % idx)
    return rows if max_samples is None else rows[:max_samples]


def resolve_image_path(images_dir: str, image_value: Any) -> str:
    image_name = str(image_value)
    if os.path.isabs(image_name):
        return image_name
    return os.path.join(images_dir, image_name)


def build_options_text(row: Dict[str, Any]) -> str:
    choices = row.get("choices")
    if isinstance(choices, list) and choices:
        return "\n".join(
            "%s. %s" % (chr(ord("A") + idx), str(choice))
            for idx, choice in enumerate(choices)
        )

    lines = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        value = row.get(letter)
        if value is not None and str(value).strip():
            lines.append("%s. %s" % (letter, str(value).strip()))
    return "\n".join(lines)


def build_prompt(question: str, row: Dict[str, Any]) -> str:
    options = build_options_text(row)
    if options:
        return PROMPT_WITH_OPTIONS.format(question=question, options=options)
    return PROMPT_NO_OPTIONS.format(question=question)


def build_retry_prompt(
    base_prompt: str,
    rejected_hint: Any,
    reasons: Sequence[str],
) -> str:
    reason_text = ", ".join(reasons) if reasons else "invalid hint"
    return (
        base_prompt
        + "\n\nA previous attempt was rejected.\n"
        + "Rejected hint: %s\n" % str(rejected_hint or "").strip()
        + "Rejected because: %s\n" % reason_text
        + "Write a substantially different, more informative hint while following every rule above.\n"
        + "Hint:"
    )


def image_to_data_url(path: str) -> str:
    mime_type = mimetypes.guess_type(path)[0] or "image/jpeg"
    supported = {"image/png", "image/jpeg", "image/webp", "image/gif"}
    if mime_type not in supported:
        raise ValueError(
            "Unsupported image type %r for %s. Use PNG, JPEG, WEBP, or non-animated GIF."
            % (mime_type, path)
        )
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return "data:%s;base64,%s" % (mime_type, encoded)


def strip_wrapping_quotes(text: str) -> str:
    value = text.strip()
    pairs = [('"', '"'), ("'", "'"), ("`", "`"), ("\u201c", "\u201d")]
    changed = True
    while changed and len(value) >= 2:
        changed = False
        for left, right in pairs:
            if value.startswith(left) and value.endswith(right):
                value = value[1:-1].strip()
                changed = True
    return value


def clean_hint(raw_hint: Any) -> str:
    value = str(raw_hint or "").replace("\r", " ").replace("\n", " ")
    value = re.sub(r"\s+", " ", value).strip()
    value = strip_wrapping_quotes(value)
    value = re.sub(
        r"^(?:hint|visual hint|question-focused hint)\s*:\s*",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()
    return strip_wrapping_quotes(value)


def normalize_text(text: Any) -> str:
    value = str(text or "").casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def word_tokens(text: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").casefold())


def iter_answer_values(answer: Any) -> Iterable[str]:
    if isinstance(answer, list):
        for item in answer:
            yield str(item)
    elif answer is not None:
        yield str(answer)


def contains_answer_leak(hint: str, answer: Any) -> bool:
    hint_norm = normalize_text(hint)
    if not hint_norm:
        return False

    for value in iter_answer_values(answer):
        answer_text = value.strip()
        if not answer_text or re.fullmatch(r"[A-Za-z]", answer_text):
            continue
        answer_norm = normalize_text(answer_text)
        if not answer_norm or len(answer_norm) < 3:
            continue
        if re.search(r"(^| )%s( |$)" % re.escape(answer_norm), hint_norm):
            return True
    return False


def has_option_letter_leak(text: str) -> bool:
    if re.fullmatch(r"\s*[A-D]\s*", text):
        return True
    return bool(re.search(r"(?<![A-Za-z0-9])\(?[A-D]\)?[\).:]", text))


def filter_hint(
    raw_hint: Any,
    clean: str,
    answer: Any,
    min_words: int,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    combined = ("%s %s" % (str(raw_hint or ""), clean)).strip()

    if not clean:
        reasons.append("empty")
    elif len(word_tokens(clean)) < min_words:
        reasons.append("too_short")

    if has_option_letter_leak(combined):
        reasons.append("option_letter")
    if contains_answer_leak(combined, answer):
        reasons.append("answer_leak")

    return ("ok" if not reasons else "filtered"), reasons


def make_output_row(
    row: Dict[str, Any],
    hint: str,
    reasons: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    out = copy.deepcopy(row)
    original_question = str(row.get(args.question_field, ""))
    out[args.question_field] = args.append_template.format(
        question=original_question,
        hint=hint,
    )
    if args.add_hint_metadata:
        out["generated_hint"] = hint
        out["generated_hint_status"] = "ok"
        out["generated_hint_reasons"] = list(reasons)
        out["generated_hint_model"] = args.model
    return out


def write_json(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_resumed_hints(path: str) -> Dict[int, Tuple[str, List[str]]]:
    resumed: Dict[int, Tuple[str, List[str]]] = {}
    if not os.path.isfile(path):
        return resumed

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    "[WARN] ignored invalid audit JSON at line %d in %s"
                    % (line_number, path)
                )
                continue
            if record.get("status") != "ok":
                continue
            row_index = record.get("row_index")
            hint = str(record.get("hint_clean", "")).strip()
            if isinstance(row_index, int) and hint:
                resumed[row_index] = (hint, list(record.get("reasons") or []))
    return resumed


def get_usage(response: Any) -> Dict[str, Optional[int]]:
    usage = getattr(response, "usage", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def request_hint(
    client: Any,
    image_data_url: str,
    prompt: str,
    args: argparse.Namespace,
) -> Tuple[str, Optional[str], Dict[str, Optional[int]]]:
    last_error: Optional[Exception] = None
    for request_attempt in range(1, args.request_retries + 1):
        try:
            response = client.responses.create(
                model=args.model,
                instructions=SYSTEM_PROMPT,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": image_data_url,
                                "detail": args.image_detail,
                            },
                        ],
                    }
                ],
                max_output_tokens=args.max_output_tokens,
            )
            return (
                str(response.output_text or ""),
                getattr(response, "id", None),
                get_usage(response),
            )
        except Exception as exc:
            last_error = exc
            if request_attempt >= args.request_retries:
                break
            delay = args.retry_base_seconds * (2 ** (request_attempt - 1))
            print(
                "[WARN] API request failed (%d/%d): %s; retrying in %.1fs"
                % (request_attempt, args.request_retries, exc, delay)
            )
            time.sleep(delay)

    assert last_error is not None
    raise last_error


def main() -> None:
    args = parse_args()

    if args.max_hint_attempts < 1:
        raise ValueError("--max_hint_attempts must be >= 1")
    if args.request_retries < 1:
        raise ValueError("--request_retries must be >= 1")
    if args.hint_min_words < 1:
        raise ValueError("--hint_min_words must be >= 1")
    if args.max_output_tokens < 1:
        raise ValueError("--max_output_tokens must be >= 1")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing OpenAI Python SDK. Install or upgrade it with: pip install -U openai"
        ) from exc

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(
            "Environment variable %s is not set. Put the API key there before running."
            % args.api_key_env
        )

    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)
    if not hasattr(client, "responses"):
        raise SystemExit(
            "The installed OpenAI SDK has no Responses API. Upgrade it with: pip install -U openai"
        )

    calib_json = os.path.abspath(args.calib_json)
    images_dir = (
        os.path.abspath(args.images_dir)
        if args.images_dir
        else os.path.join(os.path.dirname(calib_json), "images")
    )
    rows = load_rows(calib_json, args.max_samples)
    if not rows:
        raise RuntimeError("No rows found in %s" % calib_json)

    resumed_hints = (
        load_resumed_hints(args.out_hints_jsonl) if args.resume else {}
    )
    ensure_parent_dir(args.out_hints_jsonl)
    audit_mode = "a" if args.resume else "w"

    print("Loaded %d calibration rows from %s" % (len(rows), calib_json))
    print("images_dir:", images_dir)
    print("model:", args.model)
    if resumed_hints:
        print("resumed accepted hints:", len(resumed_hints))

    output_rows: List[Optional[Dict[str, Any]]] = [None] * len(rows)
    ok_count = 0
    resumed_count = 0
    retry_count = 0
    keep_original_count = 0

    with open(args.out_hints_jsonl, audit_mode, encoding="utf-8") as audit_f:
        for row_index, row in enumerate(rows):
            if args.question_field not in row:
                raise KeyError(
                    "Row %d missing question field %r"
                    % (row_index, args.question_field)
                )
            if args.image_field not in row:
                raise KeyError(
                    "Row %d missing image field %r"
                    % (row_index, args.image_field)
                )

            if row_index in resumed_hints:
                hint, reasons = resumed_hints[row_index]
                output_rows[row_index] = make_output_row(row, hint, reasons, args)
                ok_count += 1
                resumed_count += 1
            else:
                question = str(row.get(args.question_field, "")).strip()
                image_path = resolve_image_path(
                    images_dir,
                    row.get(args.image_field),
                )
                if not os.path.isfile(image_path):
                    raise FileNotFoundError(
                        "Image not found for row %d: %s" % (row_index, image_path)
                    )

                image_data_url = image_to_data_url(image_path)
                base_prompt = build_prompt(question, row)
                previous_raw = ""
                previous_reasons: List[str] = []

                for attempt in range(1, args.max_hint_attempts + 1):
                    prompt = (
                        base_prompt
                        if attempt == 1
                        else build_retry_prompt(
                            base_prompt,
                            previous_raw,
                            previous_reasons,
                        )
                    )
                    response_id: Optional[str] = None
                    usage: Dict[str, Optional[int]] = {
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                    }
                    api_error: Optional[str] = None

                    try:
                        raw_hint, response_id, usage = request_hint(
                            client,
                            image_data_url,
                            prompt,
                            args,
                        )
                        clean = clean_hint(raw_hint)
                        status, reasons = filter_hint(
                            raw_hint,
                            clean,
                            row.get(args.answer_field),
                            args.hint_min_words,
                        )
                    except Exception as exc:
                        raw_hint = ""
                        clean = ""
                        status = "filtered"
                        reasons = ["api_error"]
                        api_error = "%s: %s" % (type(exc).__name__, exc)

                    audit_status = "ok" if status == "ok" else "retry"
                    audit = {
                        "row_index": row_index,
                        "image": row.get(args.image_field),
                        "question": row.get(args.question_field),
                        "answer": row.get(args.answer_field),
                        "model": args.model,
                        "response_id": response_id,
                        "hint_prompt": prompt,
                        "hint_raw": raw_hint,
                        "hint_clean": clean,
                        "status": audit_status,
                        "attempt": attempt,
                        "reasons": list(reasons),
                        "usage": usage,
                    }
                    if api_error:
                        audit["api_error"] = api_error

                    if status == "ok":
                        output_rows[row_index] = make_output_row(
                            row,
                            clean,
                            reasons,
                            args,
                        )
                        ok_count += 1
                    else:
                        retry_count += 1
                        previous_raw = raw_hint
                        previous_reasons = list(reasons)
                        if attempt >= args.max_hint_attempts:
                            audit["status"] = "failed_keep_original"
                            output_rows[row_index] = copy.deepcopy(row)
                            keep_original_count += 1

                    audit_f.write(json.dumps(audit, ensure_ascii=False) + "\n")
                    audit_f.flush()
                    if status == "ok":
                        break

            done = row_index + 1
            if args.log_every > 0 and (
                done == len(rows) or done % args.log_every == 0
            ):
                print(
                    "Processed %d/%d rows "
                    "(ok=%d resumed=%d retries=%d keep_original=%d)"
                    % (
                        done,
                        len(rows),
                        ok_count,
                        resumed_count,
                        retry_count,
                        keep_original_count,
                    )
                )

    final_rows: List[Dict[str, Any]] = []
    for row_index, row in enumerate(output_rows):
        if row is None:
            raise RuntimeError("Internal error: row %d has no output." % row_index)
        final_rows.append(row)

    write_json(args.out_calib_json, final_rows)
    print("[OK] wrote audit hints:", os.path.abspath(args.out_hints_jsonl))
    print("[OK] wrote calibration JSON:", os.path.abspath(args.out_calib_json))
    print(
        "[OK] accepted hints: %d | resumed: %d | retry attempts: %d | kept original: %d"
        % (ok_count, resumed_count, retry_count, keep_original_count)
    )


if __name__ == "__main__":
    main()
