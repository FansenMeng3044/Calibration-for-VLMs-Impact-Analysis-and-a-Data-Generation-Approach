#!/usr/bin/env python3
"""Static, fail-closed audit of the official/LLaVA-to-Cosmos ATV migration."""

from __future__ import annotations

import argparse
import ast
import subprocess
from pathlib import Path


def require_text(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise SystemExit(f"FAIL: {label}: missing {needle!r}")
    print(f"PASS: {label}")


def definitions(text: str, filename: str) -> set[str]:
    tree = ast.parse(text, filename=filename)
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path("/private/workspace/hycui/project/ATV-Pruning"),
    )
    parser.add_argument(
        "--tamp-root",
        type=Path,
        default=Path("/private/workspace/hycui/project/Tamp"),
    )
    parser.add_argument(
        "--cosmos-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()

    official = subprocess.run(
        ["git", "show", "HEAD:qwen/activation_aware_pruner.py"],
        cwd=args.official_root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    llava_path = args.tamp_root / "llava/pruners/wanda_pruner.py"
    cosmos_path = args.cosmos_root / "cosmos_atv_prune.py"
    runner_path = args.cosmos_root / "run_three_dataset_matrix.sh"
    for path in (llava_path, cosmos_path, runner_path):
        if not path.is_file():
            raise SystemExit(f"FAIL: required file missing: {path}")
    llava = llava_path.read_text(encoding="utf-8")
    cosmos = cosmos_path.read_text(encoding="utf-8")
    runner = runner_path.read_text(encoding="utf-8")

    official_defs = definitions(official, "official_activation_aware_pruner.py")
    llava_defs = definitions(llava, str(llava_path))
    cosmos_defs = definitions(cosmos, str(cosmos_path))
    for name in ("WrappedATV", "ActivationAwarePruner"):
        if name not in official_defs:
            raise SystemExit(f"FAIL: official definition missing: {name}")
    for name in (
        "compute_atv_visual_token_selection",
        "compute_atv_text_only_selection",
        "LLaVALayerATVPruner",
    ):
        if name not in llava_defs:
            raise SystemExit(f"FAIL: LLaVA definition missing: {name}")
    for name in (
        "collect_atv_distance_record",
        "finalize_atv_visual_selection",
        "compute_atv_text_only_selection",
        "prune_ar_layers_atv",
    ):
        if name not in cosmos_defs:
            raise SystemExit(f"FAIL: Cosmos definition missing: {name}")
        print(f"PASS: Cosmos defines {name}")

    require_text(official, "cos_dist = 1 - cos_sim", "official cosine-distance rule")
    require_text(
        official,
        "k = round(min(1, self.alpha * cos_dist_avg) * num_text_tokens)",
        "official alpha/k rule",
    )
    require_text(official, "module_to_process='language_model.layers'", "official AR-only target")
    require_text(
        cosmos,
        "distances = 1 - torch.nn.functional.cosine_similarity(",
        "Cosmos cosine-distance mapping",
    )
    require_text(
        cosmos,
        "selection_scale = min(1.0, float(alpha) * mean_distance)",
        "Cosmos alpha scale mapping",
    )
    require_text(
        cosmos,
        'k = round(selection_scale * int(record["text_tokens"]))',
        "Cosmos alpha/k mapping",
    )
    require_text(
        cosmos,
        'retained = record["valid_mask"] & ~record["visual_mask"]',
        "all valid language tokens retained",
    )
    require_text(cosmos, 'retained[selected_positions] = True', "selected visual tokens retained")
    require_text(cosmos, '"mode": "text_only_zero_visual"', "explicit text-only mode")
    require_text(cosmos, '"mean_cosine_distance": None', "no fabricated text-only visual distance")
    require_text(cosmos, 'if float(args.vision_sparsity) != 0.0:', "vision sparsity hard guard")
    require_text(cosmos, '"vision_target_count": 0', "zero vision target contract")
    require_text(
        cosmos,
        'with forbidden_forward({"vision": reasoner.visual, "projector": reasoner.projector})',
        "separate vision/projector execution guard",
    )
    if "def build_vision_cache(" in cosmos:
        raise SystemExit("FAIL: Cosmos ATV still exposes a vision-pruning cache path")
    print("PASS: no Cosmos ATV vision-pruning cache path")
    require_text(runner, "--vision-sparsity 0", "runner fixes dense vision encoder")
    require_text(runner, "--alpha 1.0", "runner records ATV alpha")
    print("COSMOS_ATV_MIGRATION_STATIC_VALIDATION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
