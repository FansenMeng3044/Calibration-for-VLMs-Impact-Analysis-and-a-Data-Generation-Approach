#!/usr/bin/env python
"""Compose already-rendered SVG figures into paper-ready SVG/PDF panels."""

from __future__ import annotations

import argparse
import copy
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


DEFAULT_LAYER_VISUAL = r"E:\1study\calibration\paper_figures_layer_similarity\encoder_similarity_encoder_visual_rel_l2.svg"
DEFAULT_LAYER_TEXT = r"E:\1study\calibration\paper_figures_layer_similarity\encoder_similarity_encoder_text_rel_l2.svg"
DEFAULT_SEMANTIC_HEATMAP = r"E:\1study\calibration\paper_figures_semantic_l2\semantic_heatmap_okvqa_mmbench_l2_semantic_heatmap.svg"
DEFAULT_SEMANTIC_OKVQA = r"E:\1study\calibration\paper_figures_semantic_l2\semantic_heatmap_okvqa_mmbench_l2_okvqa_l2.svg"
DEFAULT_SEMANTIC_MMBENCH = r"E:\1study\calibration\paper_figures_semantic_l2\semantic_heatmap_okvqa_mmbench_l2_mmbench_l2.svg"
DEFAULT_OUT_DIR = r"E:\1study\calibration\paper_figures_combined_svg"


@dataclass
class SvgPanel:
    path: str
    root: ET.Element
    width: float
    height: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine existing SVG panels without replotting their contents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--layer_visual", default=DEFAULT_LAYER_VISUAL)
    parser.add_argument("--layer_text", default=DEFAULT_LAYER_TEXT)
    parser.add_argument("--semantic_heatmap", default=DEFAULT_SEMANTIC_HEATMAP)
    parser.add_argument("--semantic_okvqa", default=DEFAULT_SEMANTIC_OKVQA)
    parser.add_argument("--semantic_mmbench", default=DEFAULT_SEMANTIC_MMBENCH)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--layer_name", default="encoder_similarity_layer_two_panel")
    parser.add_argument("--semantic_name", default="semantic_heatmap_okvqa_mmbench_three_panel")
    parser.add_argument("--margin", type=float, default=18.0)
    parser.add_argument("--gap", type=float, default=22.0)
    parser.add_argument(
        "--no_pdf",
        action="store_true",
        help="Only write SVG. By default the script also writes PDF via CairoSVG.",
    )
    return parser.parse_args()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def parse_length(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.match(r"\s*([0-9.+-eE]+)", value)
    if not match:
        return None
    return float(match.group(1))


def read_svg(path: str) -> SvgPanel:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    root = ET.parse(path).getroot()
    view_box = root.attrib.get("viewBox")
    if view_box:
        parts = [float(x) for x in re.split(r"[\s,]+", view_box.strip()) if x]
        if len(parts) != 4:
            raise ValueError("Invalid viewBox in %s: %s" % (path, view_box))
        width, height = parts[2], parts[3]
    else:
        width = parse_length(root.attrib.get("width"))
        height = parse_length(root.attrib.get("height"))
        if width is None or height is None:
            raise ValueError("SVG has neither valid viewBox nor width/height: %s" % path)

    return SvgPanel(path=path, root=root, width=width, height=height)


def collect_ids(root: ET.Element) -> Dict[str, str]:
    ids: Dict[str, str] = {}
    for elem in root.iter():
        elem_id = elem.attrib.get("id")
        if elem_id:
            ids[elem_id] = elem_id
    return ids


def prefix_svg_ids(root: ET.Element, prefix: str) -> None:
    old_ids = collect_ids(root)
    id_map = {old: "%s_%s" % (prefix, old) for old in old_ids}

    def replace_url(match: re.Match[str]) -> str:
        old = match.group(1)
        return "url(#%s)" % id_map.get(old, old)

    for elem in root.iter():
        if "id" in elem.attrib and elem.attrib["id"] in id_map:
            elem.attrib["id"] = id_map[elem.attrib["id"]]

        for attr, value in list(elem.attrib.items()):
            if attr == "id":
                continue
            if value.startswith("#") and value[1:] in id_map:
                elem.attrib[attr] = "#%s" % id_map[value[1:]]
            elif "url(#" in value:
                elem.attrib[attr] = re.sub(r"url\(#([^)]+)\)", replace_url, value)


def panel_children(panel: SvgPanel, prefix: str) -> List[ET.Element]:
    root = copy.deepcopy(panel.root)
    prefix_svg_ids(root, prefix)
    children: List[ET.Element] = []
    for child in list(root):
        if local_name(child.tag) == "metadata":
            continue
        children.append(child)
    return children


def make_svg_root(width: float, height: float) -> ET.Element:
    root = ET.Element(
        "{%s}svg" % SVG_NS,
        {
            "width": "%.6gpt" % width,
            "height": "%.6gpt" % height,
            "viewBox": "0 0 %.6f %.6f" % (width, height),
            "version": "1.1",
        },
    )
    ET.SubElement(
        root,
        "{%s}rect" % SVG_NS,
        {
            "x": "0",
            "y": "0",
            "width": "%.6f" % width,
            "height": "%.6f" % height,
            "fill": "white",
        },
    )
    return root


def add_panel(
    root: ET.Element,
    panel: SvgPanel,
    prefix: str,
    x: float,
    y: float,
    scale: float = 1.0,
    scale_y: float | None = None,
) -> None:
    if scale_y is None:
        scale_y = scale
    group = ET.SubElement(
        root,
        "{%s}g" % SVG_NS,
        {"transform": "translate(%.6f %.6f) scale(%.8f %.8f)" % (x, y, scale, scale_y)},
    )
    for child in panel_children(panel, prefix):
        group.append(child)


def write_svg(root: ET.Element, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    print("[OK] SVG:", path)


def write_pdf(svg_path: str, pdf_path: str) -> None:
    try:
        import cairosvg
    except ImportError as exc:
        raise RuntimeError(
            "CairoSVG is required for PDF output. Install it with: pip install cairosvg"
        ) from exc
    cairosvg.svg2pdf(url=svg_path, write_to=pdf_path)
    print("[OK] PDF:", pdf_path)


def save_outputs(root: ET.Element, out_dir: str, name: str, write_pdf_output: bool) -> None:
    svg_path = os.path.join(out_dir, "%s.svg" % name)
    pdf_path = os.path.join(out_dir, "%s.pdf" % name)
    write_svg(root, svg_path)
    if write_pdf_output:
        write_pdf(svg_path, pdf_path)


def compose_layer_panels(panels: Sequence[SvgPanel], margin: float, gap: float) -> ET.Element:
    visual, text = panels
    column_width = max(visual.width, text.width)
    visual_scale = column_width / visual.width
    text_scale = column_width / text.width
    visual_height = visual.height * visual_scale
    text_height = text.height * text_scale
    height = max(visual_height, text_height)
    width = 2 * column_width + gap

    root = make_svg_root(width + 2 * margin, height + 2 * margin)
    add_panel(
        root,
        visual,
        "layer_visual",
        margin,
        margin + (height - visual_height) / 2,
        visual_scale,
    )
    add_panel(
        root,
        text,
        "layer_text",
        margin + column_width + gap,
        margin + (height - text_height) / 2,
        text_scale,
    )
    return root


def compose_semantic_panels(
    panels: Sequence[SvgPanel],
    margin: float,
    gap: float,
) -> ET.Element:
    heatmap, okvqa, mmbench = panels
    vertical_gap = gap
    horizontal_gap = 0.0
    right_height = okvqa.height + vertical_gap + mmbench.height
    total_height = right_height

    heatmap_scale = total_height / heatmap.height
    heatmap_width = heatmap.width * heatmap_scale
    right_width = max(okvqa.width, mmbench.width)
    total_width = heatmap_width + horizontal_gap + right_width
    root = make_svg_root(margin + total_width, total_height + 2 * margin)

    add_panel(
        root,
        heatmap,
        "semantic_heatmap",
        margin,
        margin,
        heatmap_scale,
    )
    right_x = margin + heatmap_width + horizontal_gap
    add_panel(
        root,
        okvqa,
        "semantic_okvqa",
        right_x,
        margin,
    )
    add_panel(
        root,
        mmbench,
        "semantic_mmbench",
        right_x,
        margin + okvqa.height + vertical_gap,
    )
    return root


def main() -> None:
    args = parse_args()
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))

    layer_panels = [read_svg(args.layer_visual), read_svg(args.layer_text)]
    semantic_panels = [
        read_svg(args.semantic_heatmap),
        read_svg(args.semantic_okvqa),
        read_svg(args.semantic_mmbench),
    ]

    layer_root = compose_layer_panels(layer_panels, args.margin, args.gap)
    semantic_root = compose_semantic_panels(
        semantic_panels,
        args.margin,
        args.gap,
    )

    save_outputs(layer_root, out_dir, args.layer_name, not args.no_pdf)
    save_outputs(semantic_root, out_dir, args.semantic_name, not args.no_pdf)
    print("[OK] output dir:", out_dir)


if __name__ == "__main__":
    main()
