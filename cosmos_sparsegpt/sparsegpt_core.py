#!/usr/bin/env python3
"""Numerically bounded SparseGPT kernel used by the Cosmos Reasoner adapter.

The implementation keeps the LLaVA/TAMP layer-wise algorithmic contract:

* full input Hessian approximation ``H = 2 / n * sum(X X^T)``;
* 128-input-column reconstruction blocks by default;
* OBS/SparseGPT sequential error propagation;
* vision and AR statistics are supplied by the caller and never mixed here.

Unlike the legacy adapter, Cholesky retries are bounded and every numerical
choice is returned in a machine-readable report.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import torch
import torch.nn as nn


BUDGET_EXACT = "exact_k_budget"
BUDGET_LLAVA_LEGACY = "legacy_llava_threshold"
BUDGET_MODES = (BUDGET_EXACT, BUDGET_LLAVA_LEGACY)


class SparseGPTError(RuntimeError):
    """Raised when SparseGPT statistics or reconstruction are invalid."""


@dataclasses.dataclass
class SparseGPTStats:
    """Full Hessian statistics for one target Linear."""

    columns: int
    device: torch.device
    hessian: torch.Tensor | None = dataclasses.field(init=False)
    nsamples: int = 0
    calls: int = 0
    valid_tokens: int = 0

    def __post_init__(self) -> None:
        if self.columns <= 0:
            raise ValueError(f"columns must be positive, got {self.columns}")
        self.device = torch.device(self.device)
        self.hessian = torch.zeros(
            (self.columns, self.columns),
            dtype=torch.float32,
            device=self.device,
        )

    def _flatten_valid(
        self,
        inputs: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int]:
        if inputs.ndim == 2:
            batch_count = 1
            flattened = inputs
            if valid_mask is not None:
                mask = valid_mask.to(device=inputs.device, dtype=torch.bool).reshape(-1)
                if mask.numel() == inputs.shape[0]:
                    flattened = inputs[mask]
                elif mask.numel() != 1:
                    raise SparseGPTError(
                        "Rank-2 Linear input/mask mismatch: "
                        f"activation={tuple(inputs.shape)}, mask={tuple(valid_mask.shape)}"
                    )
        elif inputs.ndim == 3:
            batch_count = int(inputs.shape[0])
            if valid_mask is None:
                flattened = inputs.reshape(-1, inputs.shape[-1])
            else:
                mask = valid_mask.to(device=inputs.device, dtype=torch.bool)
                if tuple(mask.shape) != tuple(inputs.shape[:-1]):
                    raise SparseGPTError(
                        "Linear input/mask mismatch: "
                        f"activation={tuple(inputs.shape)}, mask={tuple(mask.shape)}"
                    )
                flattened = inputs[mask]
        else:
            raise SparseGPTError(
                f"Unsupported Linear input rank {inputs.ndim}: {tuple(inputs.shape)}"
            )

        if flattened.ndim != 2 or flattened.shape[-1] != self.columns:
            raise SparseGPTError(
                f"Linear input width changed: expected {self.columns}, "
                f"got {tuple(flattened.shape)}"
            )
        if flattened.shape[0] == 0:
            raise SparseGPTError("No valid activation tokens reached a target Linear")
        return flattened, batch_count

    def add(self, inputs: torch.Tensor, valid_mask: torch.Tensor | None) -> None:
        if self.hessian is None:
            raise SparseGPTError("Cannot add activations after Hessian was released")
        flattened, batch_count = self._flatten_valid(inputs.detach(), valid_mask)
        previous = self.nsamples
        updated = previous + batch_count
        self.hessian.mul_(previous / float(updated))
        matrix = flattened.to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=True,
        ).transpose(0, 1).contiguous()
        matrix.mul_(math.sqrt(2.0 / float(updated)))
        self.hessian.addmm_(matrix, matrix.transpose(0, 1))
        self.nsamples = updated
        self.calls += 1
        self.valid_tokens += int(flattened.shape[0])
        del matrix

    @property
    def hessian_bytes(self) -> int:
        if self.hessian is None:
            return 0
        return int(self.hessian.numel() * self.hessian.element_size())

    def take_hessian(self) -> torch.Tensor:
        if self.hessian is None:
            raise SparseGPTError("Hessian has already been released")
        if self.nsamples <= 0 or self.calls <= 0 or self.valid_tokens <= 0:
            raise SparseGPTError("SparseGPT Hessian was requested before collecting activations")
        result = self.hessian
        self.hessian = None
        return result

    def free(self) -> None:
        self.hessian = None


def _bounded_cholesky(
    matrix: torch.Tensor,
    *,
    upper: bool,
    base_damp: float,
    max_retries: int,
    stage: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    if not torch.isfinite(matrix).all().item():
        raise SparseGPTError(f"{stage} matrix contains NaN or inf")
    if not math.isfinite(base_damp) or base_damp < 0:
        raise SparseGPTError(f"{stage} damping is invalid: {base_damp}")

    diagonal = torch.arange(matrix.shape[0], device=matrix.device)
    last_info = -1
    for attempt in range(max_retries + 1):
        added_damp = base_damp * attempt
        candidate = matrix.clone()
        if added_damp:
            candidate[diagonal, diagonal] += added_damp
        factor, info = torch.linalg.cholesky_ex(candidate, upper=upper, check_errors=False)
        last_info = int(info.max().item()) if info.numel() else int(info.item())
        if last_info == 0 and torch.isfinite(factor).all().item():
            return factor, {
                "stage": stage,
                "attempts": attempt + 1,
                "retries": attempt,
                "base_damp": base_damp,
                "added_damp": added_damp,
                "upper": upper,
            }
        del candidate, factor, info
        if base_damp == 0:
            break
    raise SparseGPTError(
        f"{stage} Cholesky failed after {max_retries + 1} attempts; "
        f"base_damp={base_damp}, last_info={last_info}"
    )


def _select_block_mask(
    metric: torch.Tensor,
    sparsity: float,
    budget_mode: str,
) -> tuple[torch.Tensor, int]:
    if budget_mode not in BUDGET_MODES:
        raise ValueError(f"Unsupported budget mode: {budget_mode}")
    count = int(metric.numel() * sparsity)
    mask = torch.zeros_like(metric, dtype=torch.bool)
    if count <= 0:
        return mask, 0
    flattened = metric.reshape(-1)
    if budget_mode == BUDGET_EXACT:
        selected = torch.argsort(flattened, stable=True)[:count]
        mask.reshape(-1)[selected] = True
        del selected
    else:
        # Exact reproduction of the TAMP/LLaVA zero-based threshold.  With
        # unique scores this selects count + 1 values, which explains the
        # slightly-above-target sparsity in the existing LLaVA checkpoints.
        threshold_index = min(count, flattened.numel() - 1)
        threshold = torch.sort(flattened, stable=True).values[threshold_index]
        mask = metric <= threshold
    return mask, count


def sparsegpt_prune_linear(
    linear: nn.Linear,
    stats: SparseGPTStats,
    sparsity: float,
    *,
    blocksize: int = 128,
    percdamp: float = 0.01,
    budget_mode: str = BUDGET_EXACT,
    max_cholesky_retries: int = 8,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Prune and reconstruct one Linear weight matrix in-place."""

    if type(linear) is not nn.Linear:
        raise TypeError(f"SparseGPT target must be exactly nn.Linear, got {type(linear).__name__}")
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if blocksize <= 0:
        raise ValueError(f"blocksize must be positive, got {blocksize}")
    if percdamp < 0 or not math.isfinite(percdamp):
        raise ValueError(f"percdamp must be finite and non-negative, got {percdamp}")

    weight = linear.weight.data
    rows, columns = weight.shape
    if columns != stats.columns:
        raise SparseGPTError(
            f"Weight/Hessian width mismatch: weight={tuple(weight.shape)}, stats={stats.columns}"
        )
    zeros_before = int((weight == 0).sum().item())
    hessian_bytes = stats.hessian_bytes
    hessian = stats.take_hessian()
    if hessian.device != weight.device:
        hessian = hessian.to(device=weight.device, non_blocking=True)
    if not torch.isfinite(hessian).all().item():
        raise SparseGPTError("Input Hessian contains NaN or inf")

    reconstructed = weight.detach().float().clone()
    dead = torch.diag(hessian) == 0
    dead_columns = int(dead.sum().item())
    if dead_columns:
        hessian[dead, dead] = 1
        reconstructed[:, dead] = 0

    first_base_damp = float(percdamp * torch.mean(torch.diag(hessian)).item())
    first_factor, first_report = _bounded_cholesky(
        hessian,
        upper=False,
        base_damp=first_base_damp,
        max_retries=max_cholesky_retries,
        stage="hessian",
    )
    inverse = torch.cholesky_inverse(first_factor)
    del first_factor, hessian
    if not torch.isfinite(inverse).all().item():
        raise SparseGPTError("Inverse Hessian contains NaN or inf")
    second_base_damp = float(percdamp * torch.mean(torch.diag(inverse).abs()).item())
    hinv, second_report = _bounded_cholesky(
        inverse,
        upper=True,
        base_damp=second_base_damp,
        max_retries=max_cholesky_retries,
        stage="inverse_hessian",
    )
    del inverse

    diagonal = torch.diag(hinv).reshape(1, -1)
    if (diagonal == 0).any().item() or not torch.isfinite(diagonal).all().item():
        raise SparseGPTError("Inverse-Hessian Cholesky has invalid diagonal")
    global_metric = reconstructed.square() / diagonal.square()
    importance_score_mean = float(global_metric.abs().mean().item())
    del global_metric, diagonal

    losses = torch.zeros(rows, dtype=torch.float32, device=weight.device)
    mask = torch.zeros_like(weight, dtype=torch.bool)
    requested_pruned = 0
    for block_start in range(0, columns, blocksize):
        block_end = min(block_start + blocksize, columns)
        count = block_end - block_start
        block_weight = reconstructed[:, block_start:block_end].clone()
        quantized = torch.zeros_like(block_weight)
        errors = torch.zeros_like(block_weight)
        block_losses = torch.zeros_like(block_weight)
        block_hinv = hinv[block_start:block_end, block_start:block_end]
        block_diagonal = torch.diag(block_hinv).reshape(1, -1)
        metric = block_weight.square() / block_diagonal.square()
        block_mask, requested = _select_block_mask(metric, sparsity, budget_mode)
        requested_pruned += requested
        mask[:, block_start:block_end] = block_mask
        del metric, block_diagonal

        for column_index in range(count):
            column = block_weight[:, column_index]
            diagonal_value = block_hinv[column_index, column_index]
            if diagonal_value == 0 or not torch.isfinite(diagonal_value).item():
                raise SparseGPTError(
                    f"Invalid Hinv diagonal at input column {block_start + column_index}"
                )
            updated = column.clone()
            updated[block_mask[:, column_index]] = 0
            quantized[:, column_index] = updated
            block_losses[:, column_index] = (column - updated).square() / diagonal_value.square()
            error = (column - updated) / diagonal_value
            block_weight[:, column_index:] -= error.unsqueeze(1).matmul(
                block_hinv[column_index, column_index:].unsqueeze(0)
            )
            errors[:, column_index] = error

        reconstructed[:, block_start:block_end] = quantized
        losses += block_losses.sum(dim=1) / 2.0
        if block_end < columns:
            reconstructed[:, block_end:] -= errors.matmul(hinv[block_start:block_end, block_end:])
        del block_weight, quantized, errors, block_losses, block_hinv, block_mask

    if not torch.isfinite(reconstructed).all().item():
        raise SparseGPTError("SparseGPT reconstruction produced NaN or inf weights")
    weight.copy_(reconstructed.to(dtype=weight.dtype))
    if weight.device.type == "cuda":
        torch.cuda.synchronize(weight.device)
    zeros_after = int((weight == 0).sum().item())
    mask_pruned = int(mask.sum().item())
    report = {
        "shape": [rows, columns],
        "parameters": int(weight.numel()),
        "target_sparsity": float(sparsity),
        "blocksize": int(blocksize),
        "percdamp": float(percdamp),
        "budget_mode": budget_mode,
        "requested_pruned_weights": requested_pruned,
        "mask_pruned_weights": mask_pruned,
        "activation_samples": stats.nsamples,
        "activation_hook_calls": stats.calls,
        "valid_activation_tokens": stats.valid_tokens,
        "hessian_shape": [columns, columns],
        "hessian_bytes": hessian_bytes,
        "dead_input_columns": dead_columns,
        "importance_score_mean": importance_score_mean,
        "reconstruction_loss_sum": float(losses.sum().item()),
        "cholesky": {
            "hessian": first_report,
            "inverse_hessian": second_report,
        },
        "zeros_before": zeros_before,
        "zeros_after": zeros_after,
        "actual_zero_ratio": zeros_after / float(weight.numel()),
        "finite_after": bool(torch.isfinite(weight).all().item()),
    }
    del reconstructed, hinv, losses
    stats.free()
    return report, mask
