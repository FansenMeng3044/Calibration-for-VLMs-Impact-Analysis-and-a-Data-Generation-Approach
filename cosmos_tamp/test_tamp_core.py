#!/usr/bin/env python3
"""CPU tests for Cosmos TAMP DAS, AMIA, allocation, and scope contracts."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

import cosmos_tamp_prune as tamp


class _FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4, bias=False)
        self.b = nn.Linear(4, 2, bias=False)


class _FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(), _FakeLayer()])


class _FakeReasoner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _FakeLanguageModel()


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeReasoner()


class TampCoreTests(unittest.TestCase):
    def test_joint_das_uses_three_terms(self) -> None:
        stats = tamp.DasSimilarityStats()
        outputs = torch.tensor(
            [[[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]]]
        )
        stats.add(
            outputs,
            torch.tensor([[True, True, False, False]]),
            torch.ones(1, 4, dtype=torch.bool),
        )
        score, report = stats.finalize(tamp.PROTOCOL_JOINT)
        self.assertEqual(report["defined_terms"], ["v", "l", "vl"])
        self.assertEqual(report["formula"], "(1-s_v)+(1-s_l)+(1-s_vl)")
        self.assertTrue(0.0 <= score <= 3.0)

    def test_text_das_is_strict_three_times_language_reduction(self) -> None:
        stats = tamp.DasSimilarityStats()
        outputs = torch.tensor([[[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]])
        stats.add(
            outputs,
            torch.zeros(1, 3, dtype=torch.bool),
            torch.tensor([[True, True, False]]),
        )
        score, report = stats.finalize(tamp.PROTOCOL_SEPARATE)
        self.assertEqual(report["defined_terms"], ["l"])
        self.assertEqual(report["formula"], "3*(1-s_l)")
        self.assertAlmostEqual(score, 3.0 * (1.0 - report["similarities"]["l"]), places=7)
        self.assertIsNone(report["similarities"]["v"])
        self.assertIsNone(report["similarities"]["vl"])

    def test_text_das_rejects_visual_leakage(self) -> None:
        stats = tamp.DasSimilarityStats()
        stats.add(
            torch.randn(1, 4, 3),
            torch.tensor([[True, True, False, False]]),
            torch.ones(1, 4, dtype=torch.bool),
        )
        with self.assertRaises(tamp.ProtocolError):
            stats.finalize(tamp.PROTOCOL_SEPARATE)

    def test_allocator_preserves_exact_budget_and_cap(self) -> None:
        model = _FakeModel()
        scores = {
            f"{tamp.AR_PREFIX}.{layer_index}.{name}.weight": float(index + 1)
            for index, (layer_index, layer) in enumerate(
                (item for item in enumerate(model.model.language_model.layers))
            )
            for name in tamp.find_linear_modules(layer)
        }
        allocation, report = tamp.allocate_per_linear_sparsity(
            model,
            scores,
            target_sparsity=0.5,
            max_sparsity_per_linear=0.6,
        )
        self.assertEqual(
            report["allocated_keep_parameter_count"],
            report["target_keep_parameter_count"],
        )
        self.assertAlmostEqual(report["allocated_sparsity"], 0.5, places=12)
        self.assertTrue(all(0.0 <= value <= 0.6 + 1e-12 for value in allocation.values()))
        self.assertGreater(max(allocation.values()) - min(allocation.values()), 0.0)

    def test_amia_uses_only_valid_tokens(self) -> None:
        stats = tamp.AmiaActivationStats(columns=2)
        inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
        outputs = torch.tensor([[[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]])
        stats.add(
            inputs,
            outputs,
            visual_mask=torch.zeros(1, 3, dtype=torch.bool),
            valid_mask=torch.tensor([[True, True, False]]),
            attention_score=torch.tensor([0.8, 0.2, 0.0]),
        )
        self.assertEqual(stats.calls, 1)
        self.assertEqual(stats.valid_tokens, 2)
        self.assertGreaterEqual(stats.selected_tokens, 1)
        self.assertTrue(torch.isfinite(stats.scaler_row).all().item())
        self.assertLess(float(stats.scaler_row.max().item()), 100.0)

    def test_rowwise_wanda_uses_allocated_sparsity(self) -> None:
        layer = nn.Linear(5, 2, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.arange(10, dtype=torch.float32).reshape(2, 5) + 1)
        stats = tamp.AmiaActivationStats(columns=5)
        stats.sum_sq.fill_(1.0)
        stats.selected_tokens = 1
        stats.calls = 1
        report, mask = tamp.prune_linear_weight(layer, stats, sparsity=0.4)
        self.assertEqual(report["pruned_per_output_row"], 2)
        self.assertEqual(int(mask.sum().item()), 4)
        self.assertEqual(int((layer.weight == 0).sum().item()), 4)

    def test_text_only_input_rejects_visual_fields_and_placeholder(self) -> None:
        with self.assertRaises(tamp.ProtocolError):
            tamp.assert_text_only_record({"text": "question", "image": "x.jpg"}, "test")
        with self.assertRaises(tamp.ProtocolError):
            tamp.assert_text_only_record({"text": "<image> question"}, "test")


if __name__ == "__main__":
    unittest.main()
