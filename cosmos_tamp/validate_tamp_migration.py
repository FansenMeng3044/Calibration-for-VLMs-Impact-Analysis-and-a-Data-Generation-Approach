#!/usr/bin/env python3
"""Fail-closed static comparison of LLaVA TAMP and the Cosmos migration."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LLAVA_ROOT = Path("/private/workspace/hycui/project/Tamp")

LLAVA_SHA256 = {
    "llava/evaluate.py": "97c0d15120a58ddb2b62a6de30fccdf6b7b71c5bb30b9c723239bd7e787e248c",
    "llava/pruners/data_loader.py": "977f30321bc14e35b0e20a9b7a29223847534604fac3c7dcefc922437fe9f820",
    "llava/pruners/wanda_pruner.py": "d06e643ad8cbea6e69f825ae499842bb4f0f53c4a4fa81f65412a34ffc314244",
    "llava/pruners/layer_single_base_pruner.py": "86adb6365a47efc0863d26177b14e609bc5d0542ef2cc4051521c1cefcb70b42",
    "scripts/prune/tamp_fourcalib_prune_eval_common.sh": "0f8fe992d19d7a44e2828a790268abdbab0fe0c76c2863d25f9666a3a431b23f",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(source: str, needles: list[str], context: str) -> None:
    missing = [needle for needle in needles if needle not in source]
    if missing:
        raise RuntimeError(f"{context} is missing locked semantics: {missing}")


def main() -> int:
    observed_hashes: dict[str, str] = {}
    for relative, expected in LLAVA_SHA256.items():
        path = LLAVA_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        observed_hashes[relative] = actual
        if actual != expected:
            raise RuntimeError(
                f"LLaVA migration source changed: {relative} {actual} != {expected}"
            )

    evaluate_source = (LLAVA_ROOT / "llava/evaluate.py").read_text(encoding="utf-8")
    require(
        evaluate_source,
        [
            'args.token_selection = "amia"',
            'args.score_method = "density_sum"',
            'args.sparsity_ratio_granularity = "layer"',
            'requested_prune_method == "tamp"',
        ],
        "LLaVA TAMP alias",
    )
    das_source = (LLAVA_ROOT / "llava/pruners/layer_single_base_pruner.py").read_text(
        encoding="utf-8"
    )
    require(
        das_source,
        [
            "def das_diversity_score",
            "score = sum(terms) * (3.0 / len(terms))",
            'elif self.score_compute.startswith("density")',
        ],
        "LLaVA DAS",
    )
    wanda_source = (LLAVA_ROOT / "llava/pruners/wanda_pruner.py").read_text(encoding="utf-8")
    require(
        wanda_source,
        [
            "class AdaptiveMultimodalInputActivation",
            "num_neigh = min(3, num_tokens - 1)",
            "MMD2 = K_XX + K_YY - 2 * K_XY",
            "W_metric = torch.abs(subset[name].weight.data) * torch.sqrt",
            "last_valid_query",
            'elif sparsity_ratio_granularity == "layer"',
        ],
        "LLaVA AMIA/WANDA",
    )

    cosmos_path = ROOT / "cosmos_tamp_prune.py"
    cosmos_source = cosmos_path.read_text(encoding="utf-8")
    ast.parse(cosmos_source, filename=str(cosmos_path))
    require(
        cosmos_source,
        [
            "class DasSimilarityStats",
            'formula = "3*(1-s_l)"',
            "def collect_das_scores",
            "def allocate_per_linear_sparsity",
            "class AmiaActivationStats",
            "def layer_attention_scores",
            "def prune_ar_layers_tamp",
            '"algorithm_components": ["DAS", "AMIA", "WANDA"]',
            '"sparsity_allocation": "DAS density_sum per Linear tensor"',
            '"vision_target_count": 0',
            '"projector_target_count": 0',
            "with forbidden_forward({\"vision\": reasoner.visual, \"projector\": reasoner.projector})",
        ],
        "Cosmos TAMP",
    )
    run_source = cosmos_source[cosmos_source.index("def run_pruning") :]
    if "prune_ar_layers_atv(" in run_source:
        raise RuntimeError("Cosmos TAMP run path still calls the ATV pruning kernel")
    if "prune_ar_layers_tamp(" not in run_source:
        raise RuntimeError("Cosmos TAMP run path does not call its TAMP kernel")
    require(
        cosmos_source,
        [
            "Legacy ATV selection is forbidden in the Cosmos TAMP implementation",
            "Legacy ATV pruning is forbidden; use prune_ar_layers_tamp",
        ],
        "Cosmos TAMP legacy-path guards",
    )

    presets = json.loads((ROOT / "calibration_presets.json").read_text(encoding="utf-8"))
    for dataset in ("mmbench", "mmmu", "okvqa"):
        entry = presets.get("datasets", {}).get(dataset)
        if not isinstance(entry, dict) or set(entry) != {"joint", "separate"}:
            raise RuntimeError(f"Calibration preset {dataset} lacks joint/separate entries")
        for protocol in ("joint", "separate"):
            for values in entry[protocol].values():
                if isinstance(values, list):
                    for value in values:
                        if value.endswith((".json", ".jsonl")) and not Path(value).is_file():
                            raise FileNotFoundError(value)

    project = LLAVA_ROOT
    eval_contract = {
        "mmbench": project / "lmms_tasks/mmbench_en_dev_local.yaml",
        "mmmu": project / "lmms_tasks/mmmu_local/mmmu_val_local.yaml",
        "okvqa": project / "lmms_tasks/okvqa_local/okvqa_val2014_local.yaml",
    }
    for path in eval_contract.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (ROOT / "cosmos_lmms_plugin/models/cosmos3_edge.py").is_file():
        raise FileNotFoundError("Cosmos lmms-eval adapter")

    report = {
        "llava_source_sha256": observed_hashes,
        "cosmos_source_sha256": sha256(cosmos_path),
        "calibration_datasets": ["mmbench", "mmmu", "okvqa"],
        "protocols": ["joint", "separate"],
        "eval_tasks": {name: str(path) for name, path in eval_contract.items()},
        "target_scope": "Cosmos Reasoner AR/LLM Linear only",
        "vision_target_count": 0,
        "validated": True,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("COSMOS_TAMP_MIGRATION_STATIC_VALIDATION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
