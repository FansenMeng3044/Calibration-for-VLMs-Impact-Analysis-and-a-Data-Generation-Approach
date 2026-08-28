#!/usr/bin/env python3
"""CPU unit checks for the standalone Cosmos SparseGPT kernel."""

from __future__ import annotations

import torch
import torch.nn as nn

from sparsegpt_core import (
    BUDGET_EXACT,
    BUDGET_LLAVA_LEGACY,
    SparseGPTStats,
    sparsegpt_prune_linear,
)


def collect_stats(linear: nn.Linear) -> SparseGPTStats:
    stats = SparseGPTStats(linear.in_features, torch.device("cpu"))
    generator = torch.Generator().manual_seed(42)
    for _ in range(10):
        inputs = torch.randn(1, 3, linear.in_features, generator=generator)
        valid = torch.tensor([[True, True, False]])
        stats.add(inputs, valid)
    return stats


def run_case(mode: str) -> tuple[int, dict]:
    torch.manual_seed(7)
    linear = nn.Linear(8, 4, bias=False)
    report, mask = sparsegpt_prune_linear(
        linear,
        collect_stats(linear),
        0.5,
        blocksize=4,
        percdamp=0.01,
        budget_mode=mode,
        max_cholesky_retries=4,
    )
    assert report["activation_samples"] == 10
    assert report["activation_hook_calls"] == 10
    assert report["valid_activation_tokens"] == 20
    assert report["finite_after"]
    assert torch.isfinite(linear.weight).all()
    assert int(mask.sum()) == report["mask_pruned_weights"]
    return int((linear.weight == 0).sum().item()), report


def main() -> None:
    exact_zeros, exact = run_case(BUDGET_EXACT)
    legacy_zeros, legacy = run_case(BUDGET_LLAVA_LEGACY)
    assert exact_zeros == 16, exact
    assert exact["mask_pruned_weights"] == 16
    assert legacy_zeros >= exact_zeros
    assert legacy["mask_pruned_weights"] >= exact["mask_pruned_weights"]
    print(
        {
            "exact_zeros": exact_zeros,
            "legacy_zeros": legacy_zeros,
            "exact_budget": exact["requested_pruned_weights"],
            "status": "ok",
        }
    )


if __name__ == "__main__":
    main()
