#!/usr/bin/env python3
"""Draw paper-style calibration sample cards for CC3M and C4.

The figure is designed for an introduction/motivation panel:
  - one CC3M multimodal card: image on top, caption below;
  - one C4 text-only card;
  - the same CC3M sample split into image-only and caption-only cards.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import textwrap
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont


TEXT_BG = "#A5CDE2"
INK = "#23313B"
MUTED = "#5A6A73"
PAPER = "#FFFFFF"
SHADOW = "#CAD3D8"
RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render rounded calibration cards for CC3M/C4 paper figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cc3m_json", default="/data/data2/mfs/CC3M_calib_128/cc3m_calib_128.json")
    parser.add_argument("--cc3m_images_dir", default="/data/data2/mfs/CC3M_calib_128/images")
    parser.add_argument("--c4_json", default="/data/data2/mfs/c4_calib_128.json")
    parser.add_argument("--cc3m_index", type=int, default=0)
    parser.add_argument("--c4_index", type=int, default=0)
    parser.add_argument("--out_dir", default="/data/data2/mfs/calibration_intro_cards")
    parser.add_argument("--out_prefix", default="calibration_intro_cards")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=2250)
    parser.add_argument("--font", default=None, help="Optional path to a .ttf/.otf font.")
    parser.add_argument("--hide_titles", action="store_true")
    parser.add_argument("--show_connectors", action="store_true", help="Draw light connector lines from the multimodal card to the split cards.")
    parser.add_argument("--no_split_vector", action="store_true", help="Do not export the three separate SVG/PDF vector panels.")
    parser.add_argument("--max_caption_chars", type=int, default=175)
    parser.add_argument("--max_c4_chars", type=int, default=330)
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rows_from_json(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("annotations", "data", "items", "questions", "samples"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        return list(data.values())
    raise ValueError("unsupported JSON top-level type: %s" % type(data).__name__)


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    if isinstance(value, (list, tuple)):
        return " ".join(stringify(v) for v in value if stringify(v)).strip()
    if isinstance(value, dict):
        parts = []
        for key, val in value.items():
            text = stringify(val)
            if text:
                parts.append("%s. %s" % (key, text))
        return " ".join(parts).strip()
    return str(value).strip()


def first_text(row: Any, keys: Sequence[str]) -> str:
    if isinstance(row, str):
        return row.strip()
    if not isinstance(row, dict):
        return stringify(row)
    for key in keys:
        text = stringify(row.get(key))
        if text:
            return text
    return ""


def shorten(text: str, max_chars: int) -> str:
    text = clean_display_text(text)
    if len(text) <= max_chars:
        return text
    cut = text[: max(0, max_chars - 1)].rstrip()
    last_space = cut.rfind(" ")
    if last_space >= int(max_chars * 0.65):
        cut = cut[:last_space]
    return cut.rstrip(" ,;:.") + "..."


def clean_display_text(text: str) -> str:
    text = " ".join(str(text or "").split())
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    return text


def pick_cc3m_sample(rows: Sequence[Any], images_dir: str | Path, index: int) -> tuple[dict[str, Any], Path, str]:
    if not rows:
        raise ValueError("empty CC3M rows")
    start = min(max(index, 0), len(rows) - 1)
    order = list(range(start, len(rows))) + list(range(0, start))
    for i in order:
        row = rows[i]
        if not isinstance(row, dict):
            continue
        image_name = stringify(row.get("image"))
        if not image_name:
            continue
        image_path = Path(image_name)
        if not image_path.is_absolute():
            image_path = Path(images_dir) / image_name
        if not image_path.is_file():
            continue
        caption = first_text(row, ("caption", "text", "text_input", "question", "prompt", "output"))
        if caption:
            return row, image_path, caption
    raise FileNotFoundError("no CC3M row with an existing image and usable text")


def pick_c4_text(rows: Sequence[Any], index: int) -> str:
    if not rows:
        raise ValueError("empty C4 rows")
    start = min(max(index, 0), len(rows) - 1)
    order = list(range(start, len(rows))) + list(range(0, start))
    for i in order:
        text = first_text(rows[i], ("text", "caption", "text_input", "output", "question", "prompt"))
        if text:
            return text
    raise ValueError("no usable C4 text found")


def font_candidates() -> Iterable[Path]:
    paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/msyhl.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/source-han-sans/SourceHanSansCN-Regular.otf",
        "/usr/share/fonts/opentype/source-han-sans/SourceHanSansCN-Bold.otf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in paths:
        p = Path(path)
        if p.is_file():
            yield p


def load_font(size: int, requested: Optional[str] = None, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = pick_font_path(requested, bold=bold)
    if path:
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def pick_font_path(requested: Optional[str] = None, bold: bool = False) -> Optional[Path]:
    if requested:
        return Path(requested)
    candidates = list(font_candidates())
    if bold:
        bold_candidates = [
            p
            for p in candidates
            if "Bold" in p.name or "bd" in p.stem.lower() or p.name.lower() == "msyhbd.ttc"
        ]
        for path in bold_candidates:
            return path
    for path in candidates:
        return path
    return None


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def lerp(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * t))


def vertical_gradient(size: tuple[int, int], top: str, bottom: str) -> Image.Image:
    w, h = size
    out = Image.new("RGB", size)
    t_rgb = hex_to_rgb(top)
    b_rgb = hex_to_rgb(bottom)
    draw = ImageDraw.Draw(out)
    denom = max(1, h - 1)
    for y in range(h):
        t = y / denom
        rgb = tuple(lerp(t_rgb[i], b_rgb[i], t) for i in range(3))
        draw.line([(0, y), (w, y)], fill=rgb)
    return out


def rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=255)
    return mask


def paste_round(canvas: Image.Image, image: Image.Image, xy: tuple[int, int], radius: int) -> None:
    mask = rounded_mask(image.size, radius)
    canvas.paste(image, xy, mask)


def cover_resize(image: Image.Image, target: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    tw, th = target
    scale = max(tw / image.width, th / image.height)
    nw = int(math.ceil(image.width * scale))
    nh = int(math.ceil(image.height * scale))
    resized = image.resize((nw, nh), RESAMPLE_LANCZOS)
    left = max(0, (nw - tw) // 2)
    top = max(0, (nh - th) // 2)
    return resized.crop((left, top, left + tw, top + th))


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    if not text:
        return 0
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def font_px(font: ImageFont.ImageFont, fallback: int = 28) -> int:
    return int(getattr(font, "size", fallback))


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if text_width(draw, candidate, font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
                current = word
            else:
                lines.append(word)
                current = ""
    if current:
        lines.append(current)
    return lines


def fit_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    max_height: int,
    line_gap: int,
) -> list[str]:
    lines = wrap_text(draw, text, font, max_width)
    line_height = int(font_px(font) * 1.12)
    max_lines = max(1, max_height // (line_height + line_gap))
    if len(lines) <= max_lines:
        return lines
    lines = lines[:max_lines]
    last = lines[-1]
    while last and text_width(draw, last + "...", font) > max_width:
        last = last.rsplit(" ", 1)[0] if " " in last else last[:-1]
    lines[-1] = last.rstrip(" ,;:.") + "..."
    return lines


def draw_centered_lines(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    line_gap: int = 10,
) -> None:
    x0, y0, x1, y1 = box
    max_width = x1 - x0
    max_height = y1 - y0
    lines = fit_lines(draw, text, font, max_width, max_height, line_gap)
    line_height = int(font_px(font) * 1.12)
    total_h = len(lines) * line_height + max(0, len(lines) - 1) * line_gap
    y = y0 + max(0, (max_height - total_h) // 2)
    for line in lines:
        w = text_width(draw, line, font)
        draw.text((x0 + (max_width - w) / 2, y), line, font=font, fill=fill)
        y += line_height + line_gap


def draw_left_lines(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    line_gap: int = 10,
    valign: str = "top",
) -> None:
    x0, y0, x1, y1 = box
    max_height = y1 - y0
    lines = fit_lines(draw, text, font, x1 - x0, max_height, line_gap)
    line_height = int(font_px(font) * 1.12)
    total_h = len(lines) * line_height + max(0, len(lines) - 1) * line_gap
    if valign == "center":
        y = y0 + max(0, (max_height - total_h) // 2)
    else:
        y = y0
    for line in lines:
        draw.text((x0, y), line, font=font, fill=fill)
        y += line_height + line_gap


def draw_title(
    draw: ImageDraw.ImageDraw,
    text: str,
    center_x: int,
    y: int,
    font: ImageFont.ImageFont,
    fill: str = INK,
) -> None:
    w = text_width(draw, text, font)
    draw.text((center_x - w / 2, y), text, font=font, fill=fill)


def draw_multimodal_card(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    image_path: Path,
    caption: str,
    fonts: dict[str, ImageFont.ImageFont],
    title: Optional[str] = None,
) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    radius = 42

    top_h = int(h * 0.58)
    card = Image.new("RGB", (w, h), TEXT_BG)
    d = ImageDraw.Draw(card)

    img_box = (0, 0, w, top_h)
    image = cover_resize(Image.open(image_path), (img_box[2] - img_box[0], img_box[3] - img_box[1]))
    card.paste(image, (img_box[0], img_box[1]))

    text = shorten(caption, 175)
    text_box = (118, top_h + 28, w - 118, h - 34)
    draw_centered_lines(d, text_box, text, fonts["body"], INK, line_gap=8)

    paste_round(canvas, card, (x0, y0), radius)
    if title:
        draw = ImageDraw.Draw(canvas)
        draw_title(draw, title, x0 + w // 2, y0 - 72, fonts["title"])


def draw_text_card(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    text: str,
    fonts: dict[str, ImageFont.ImageFont],
    title: Optional[str] = None,
    max_chars: int = 330,
) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    radius = 42
    card = Image.new("RGB", (w, h), TEXT_BG)
    d = ImageDraw.Draw(card)
    text_box = (148, 64, w - 148, h - 64)
    draw_centered_lines(d, text_box, shorten(text, max_chars), fonts["body"], INK, line_gap=8)
    paste_round(canvas, card, (x0, y0), radius)
    if title:
        draw = ImageDraw.Draw(canvas)
        draw_title(draw, title, x0 + w // 2, y0 - 72, fonts["title"])


def draw_image_only_card(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    image_path: Path,
    fonts: dict[str, ImageFont.ImageFont],
    title: Optional[str] = None,
) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    radius = 38
    image = cover_resize(Image.open(image_path), (w, h))
    paste_round(canvas, image, (x0, y0), radius)
    if title:
        draw = ImageDraw.Draw(canvas)
        draw_title(draw, title, x0 + w // 2, y0 - 62, fonts["small_title"])


def draw_caption_only_card(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    caption: str,
    fonts: dict[str, ImageFont.ImageFont],
    title: Optional[str] = None,
) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    radius = 38
    card = Image.new("RGB", (w, h), TEXT_BG)
    d = ImageDraw.Draw(card)
    text_box = (132, 46, w - 132, h - 46)
    draw_centered_lines(d, text_box, shorten(caption, 145), fonts["small_body"], INK, line_gap=8)
    paste_round(canvas, card, (x0, y0), radius)
    if title:
        draw = ImageDraw.Draw(canvas)
        draw_title(draw, title, x0 + w // 2, y0 - 62, fonts["small_title"])


def draw_connector(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    x0, y0 = start
    x1, y1 = end
    color = "#8FA8B5"
    draw.line((x0, y0, x1, y1), fill=color, width=3)
    r = 9
    draw.ellipse((x0 - r, y0 - r, x0 + r, y0 + r), fill=color)
    draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=color)


def draw_dash_dot_line(draw: ImageDraw.ImageDraw, x0: int, x1: int, y: int, color: str = "#8FA8B5") -> None:
    x = x0
    while x < x1:
        dash_end = min(x + 34, x1)
        draw.line((x, y, dash_end, y), fill=color, width=3)
        x = dash_end + 16
        if x < x1:
            r = 4
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
            x += 22


def vector_wrap(text: str, max_chars: int) -> str:
    return "\n".join(vector_lines(text, max_chars))


def vector_lines(text: str, max_chars: int) -> list[str]:
    return textwrap.wrap(clean_display_text(text), width=max_chars, break_long_words=False)


def add_round_rect(ax: Any, box: tuple[float, float, float, float], radius: float, color: str) -> Any:
    from matplotlib.patches import FancyBboxPatch

    x0, y0, x1, y1 = box
    patch = FancyBboxPatch(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0,
        facecolor=color,
    )
    ax.add_patch(patch)
    return patch


def add_clipped_rect(ax: Any, clip: Any, box: tuple[float, float, float, float], color: str) -> None:
    from matplotlib.patches import Rectangle

    x0, y0, x1, y1 = box
    patch = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=0, facecolor=color)
    patch.set_clip_path(clip)
    ax.add_patch(patch)


def add_clipped_vertical_gradient(
    ax: Any,
    clip: Any,
    box: tuple[float, float, float, float],
    top: str,
    bottom: str,
    steps: int = 80,
) -> None:
    x0, y0, x1, y1 = box
    top_rgb = hex_to_rgb(top)
    bottom_rgb = hex_to_rgb(bottom)
    h = y1 - y0
    for i in range(steps):
        t0 = i / steps
        t1 = (i + 1) / steps
        rgb = tuple(lerp(top_rgb[j], bottom_rgb[j], (t0 + t1) / 2) / 255.0 for j in range(3))
        add_clipped_rect(ax, clip, (x0, y0 + h * t0, x1, y0 + h * t1 + 0.5), rgb)


def add_vector_image(
    ax: Any,
    image_path: Path,
    box: tuple[float, float, float, float],
    radius: float,
    clip_path: Optional[Any] = None,
) -> None:
    from matplotlib.patches import FancyBboxPatch
    import numpy as np

    x0, y0, x1, y1 = box
    image = cover_resize(Image.open(image_path), (int(x1 - x0), int(y1 - y0)))
    arr = np.asarray(image)
    im = ax.imshow(arr, extent=(x0, x1, y1, y0), origin="upper", zorder=3)
    if clip_path is not None:
        im.set_clip_path(clip_path)
        return
    clip = FancyBboxPatch(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0,
        facecolor="none",
    )
    ax.add_patch(clip)
    im.set_clip_path(clip)


def add_vector_title(ax: Any, title: str, center_x: float, y: float, fontprops: Any, size: int = 27) -> None:
    ax.text(center_x, y, title, ha="center", va="center", color=INK, fontsize=size, fontproperties=fontprops)


def setup_vector_ax(width: int, height: int) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    return fig, ax


def draw_vector_multimodal(
    ax: Any,
    box: tuple[float, float, float, float],
    image_path: Path,
    caption: str,
    fontprops: dict[str, Any],
) -> None:
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    radius = 42
    top_h = h * 0.58
    clip = add_round_rect(ax, box, radius, TEXT_BG)
    add_vector_image(ax, image_path, (x0, y0, x1, y0 + top_h), 0, clip_path=clip)
    font_size = 20
    lines = vector_lines(shorten(caption, 175), 48)
    line_step = font_size * 1.32
    text_y0 = y0 + top_h + 28
    text_y1 = y1 - 34
    total_h = max(font_size, len(lines) * font_size + max(0, len(lines) - 1) * (line_step - font_size))
    text_y = text_y0 + max(0, (text_y1 - text_y0 - total_h) / 2)
    ax.text(
        (x0 + x1) / 2,
        text_y + total_h / 2,
        "\n".join(lines),
        ha="center",
        va="center",
        color=INK,
        fontsize=font_size,
        linespacing=1.32,
        fontproperties=fontprops["body"],
    )


def draw_vector_text(
    ax: Any,
    box: tuple[float, float, float, float],
    text: str,
    fontprops: dict[str, Any],
    max_chars: int,
    wrap_chars: int,
    font_size: int = 23,
) -> None:
    add_round_rect(ax, box, 42, TEXT_BG)
    x0, y0, x1, y1 = box
    ax.text(
        (x0 + x1) / 2,
        (y0 + y1) / 2,
        vector_wrap(shorten(text, max_chars), wrap_chars),
        ha="center",
        va="center",
        color=INK,
        fontsize=font_size,
        linespacing=1.35,
        fontproperties=fontprops["body"],
    )


def draw_vector_image_only(ax: Any, box: tuple[float, float, float, float], image_path: Path) -> None:
    add_vector_image(ax, image_path, box, 38)


def draw_vector_dash_dot(ax: Any, x0: float, x1: float, y: float, color: str = "#8FA8B5") -> None:
    x = x0
    while x < x1:
        dash_end = min(x + 34, x1)
        ax.plot([x, dash_end], [y, y], color=color, linewidth=2.2, solid_capstyle="round")
        x = dash_end + 16
        if x < x1:
            ax.scatter([x], [y], s=18, color=color)
            x += 22


def save_vector(fig: Any, out_svg: Path, out_pdf: Path) -> None:
    import matplotlib.pyplot as plt

    fig.savefig(out_svg, format="svg", transparent=False)
    fig.savefig(out_pdf, format="pdf", transparent=False)
    plt.close(fig)


def export_split_vector_panels(
    args: argparse.Namespace,
    out_dir: Path,
    cc3m_image: Path,
    cc3m_caption: str,
    c4_text: str,
) -> dict[str, dict[str, str]]:
    import matplotlib as mpl
    from matplotlib.font_manager import FontProperties

    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    regular_font = pick_font_path(args.font, bold=False)
    bold_font = pick_font_path(args.font, bold=True)
    fontprops = {
        "title": FontProperties(fname=str(bold_font)) if bold_font else FontProperties(weight="bold"),
        "body": FontProperties(fname=str(regular_font)) if regular_font else FontProperties(),
    }

    outputs: dict[str, dict[str, str]] = {}

    width, height = 1000, 730
    fig, ax = setup_vector_ax(width, height)
    if not args.hide_titles:
        add_vector_title(ax, "Multimodal Calibration", width / 2, 58, fontprops["title"])
    draw_vector_multimodal(ax, (60, 120, 940, 670), cc3m_image, shorten(cc3m_caption, args.max_caption_chars), fontprops)
    svg = out_dir / f"{args.out_prefix}_multimodal.svg"
    pdf = out_dir / f"{args.out_prefix}_multimodal.pdf"
    save_vector(fig, svg, pdf)
    outputs["multimodal"] = {"svg": str(svg), "pdf": str(pdf)}

    width, height = 1000, 560
    fig, ax = setup_vector_ax(width, height)
    if not args.hide_titles:
        add_vector_title(ax, "Unimodal Calibration", width / 2, 58, fontprops["title"])
    draw_vector_text(ax, (60, 120, 940, 500), c4_text, fontprops, args.max_c4_chars, wrap_chars=48, font_size=20)
    svg = out_dir / f"{args.out_prefix}_unimodal.svg"
    pdf = out_dir / f"{args.out_prefix}_unimodal.pdf"
    save_vector(fig, svg, pdf)
    outputs["unimodal"] = {"svg": str(svg), "pdf": str(pdf)}

    width, height = 1000, 850
    fig, ax = setup_vector_ax(width, height)
    if not args.hide_titles:
        add_vector_title(ax, "Split Multimodal Calibration", width / 2, 58, fontprops["title"])
    image_box = (60, 120, 940, 439)
    caption_box = (60, 506, 940, 816)
    draw_vector_image_only(ax, image_box, cc3m_image)
    draw_vector_dash_dot(ax, 156, 844, 472)
    draw_vector_text(ax, caption_box, cc3m_caption, fontprops, args.max_caption_chars, wrap_chars=48, font_size=20)
    svg = out_dir / f"{args.out_prefix}_split_multimodal.svg"
    pdf = out_dir / f"{args.out_prefix}_split_multimodal.pdf"
    save_vector(fig, svg, pdf)
    outputs["split_multimodal"] = {"svg": str(svg), "pdf": str(pdf)}

    return outputs


def main() -> int:
    args = parse_args()
    cc3m_rows = rows_from_json(load_json(args.cc3m_json))
    c4_rows = rows_from_json(load_json(args.c4_json))
    cc3m_row, cc3m_image, cc3m_caption = pick_cc3m_sample(cc3m_rows, args.cc3m_images_dir, args.cc3m_index)
    c4_text = pick_c4_text(c4_rows, args.c4_index)

    fonts = {
        "title": load_font(40, args.font, bold=True),
        "small_title": load_font(30, args.font, bold=True),
        "body": load_font(25, args.font),
        "small_body": load_font(24, args.font),
        "caption": load_font(26, args.font),
    }

    canvas = Image.new("RGBA", (args.width, args.height), PAPER)
    draw = ImageDraw.Draw(canvas)

    card_w = min(900, args.width - 220)
    card_x = (args.width - card_w) // 2
    title_x = args.width // 2

    mm_box = (card_x, 150, card_x + card_w, 700)
    c4_box = (card_x, 840, card_x + card_w, 1225)
    split_image_h = int((mm_box[3] - mm_box[1]) * 0.58)
    image_box = (card_x, 1425, card_x + card_w, 1425 + split_image_h)
    caption_box = (card_x, image_box[3] + 47, card_x + card_w, image_box[3] + 357)

    if not args.hide_titles:
        draw_title(draw, "Multimodal Calibration", title_x, 84, fonts["title"])
        draw_title(draw, "Unimodal Calibration", title_x, 774, fonts["title"])
        draw_title(draw, "Split Multimodal Calibration", title_x, 1358, fonts["title"])

    draw_multimodal_card(canvas, mm_box, cc3m_image, shorten(cc3m_caption, args.max_caption_chars), fonts)
    draw_text_card(canvas, c4_box, shorten(c4_text, args.max_c4_chars), fonts)
    draw_image_only_card(canvas, image_box, cc3m_image, fonts)
    draw_caption_only_card(canvas, caption_box, shorten(cc3m_caption, args.max_caption_chars), fonts)

    draw = ImageDraw.Draw(canvas)
    draw_dash_dot_line(draw, card_x + 96, card_x + card_w - 96, (image_box[3] + caption_box[1]) // 2)

    if args.show_connectors:
        draw = ImageDraw.Draw(canvas)
        draw_connector(draw, ((mm_box[0] + mm_box[2]) // 2, mm_box[3] + 80), ((image_box[0] + image_box[2]) // 2, image_box[1] - 28))
        draw_connector(draw, ((mm_box[0] + mm_box[2]) // 2, mm_box[3] + 80), ((caption_box[0] + caption_box[2]) // 2, caption_box[1] - 28))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{args.out_prefix}.png"
    out_pdf = out_dir / f"{args.out_prefix}.pdf"
    out_meta = out_dir / f"{args.out_prefix}_samples.json"
    rgb = canvas.convert("RGB")
    rgb.save(out_png, quality=95)
    rgb.save(out_pdf)
    vector_outputs = {}
    if not args.no_split_vector:
        vector_outputs = export_split_vector_panels(args, out_dir, cc3m_image, cc3m_caption, c4_text)

    meta = {
        "cc3m_json": str(args.cc3m_json),
        "cc3m_images_dir": str(args.cc3m_images_dir),
        "cc3m_index_requested": args.cc3m_index,
        "cc3m_image": stringify(cc3m_row.get("image")) if isinstance(cc3m_row, dict) else "",
        "cc3m_image_resolved": str(cc3m_image),
        "cc3m_caption": cc3m_caption,
        "c4_json": str(args.c4_json),
        "c4_index_requested": args.c4_index,
        "c4_text": c4_text,
        "colors": {
            "image_background": "none",
            "text_background": TEXT_BG,
        },
        "outputs": {"png": str(out_png), "pdf": str(out_pdf), "split_vector": vector_outputs},
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("[OK] wrote:", out_png)
    print("[OK] wrote:", out_pdf)
    for panel, paths in vector_outputs.items():
        print("[OK] wrote %s SVG:" % panel, paths["svg"])
        print("[OK] wrote %s PDF:" % panel, paths["pdf"])
    print("[OK] wrote:", out_meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
