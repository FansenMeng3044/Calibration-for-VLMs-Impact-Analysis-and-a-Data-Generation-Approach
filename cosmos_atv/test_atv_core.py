#!/usr/bin/env python3
"""CPU unit tests for the Cosmos ATV selection and activation contracts."""

from __future__ import annotations

import unittest

import torch

import cosmos_atv_prune as atv


class ATVSelectionTests(unittest.TestCase):
    def distance_record(self) -> dict:
        valid = torch.ones(6, dtype=torch.bool)
        visual = torch.tensor([False, True, False, True, False, False])
        return {
            "sample_id": "sample",
            "mask_shape": (1, 6),
            "valid_mask": valid,
            "visual_mask": visual,
            "visual_positions": torch.tensor([1, 3]),
            "distances": torch.tensor([0.5, 0.1]),
            "text_tokens": 4,
            "visual_tokens": 2,
            "padding_tokens": 0,
        }

    def test_official_alpha_k_and_topk_rule(self) -> None:
        masks, stats = atv.finalize_atv_visual_selection([self.distance_record()], alpha=1.0)
        self.assertAlmostEqual(stats["mean_cosine_distance"], 0.3, places=6)
        self.assertEqual(stats["selected_visual_tokens"], [1])
        self.assertEqual(stats["selected_visual_indices"], [[0]])
        self.assertEqual(masks[0].tolist(), [[True, True, True, False, True, True]])

    def test_selection_clamps_to_available_visual_tokens(self) -> None:
        masks, stats = atv.finalize_atv_visual_selection([self.distance_record()], alpha=100.0)
        self.assertEqual(stats["selection_scale"], 1.0)
        self.assertEqual(stats["selected_visual_tokens"], [2])
        self.assertTrue(masks[0].all().item())

    def test_text_only_forces_zero_visual_selection(self) -> None:
        sample = atv.LayerSample(
            hidden_states=torch.zeros(1, 4, 3),
            layer_kwargs={},
            valid_mask=torch.tensor([[True, True, True, False]]),
            visual_mask=torch.zeros(1, 4, dtype=torch.bool),
            sample_id="text",
        )
        masks, stats = atv.compute_atv_text_only_selection([sample], alpha=7.0)
        self.assertEqual(masks[0].tolist(), [[True, True, True, False]])
        self.assertEqual(stats["visual_tokens"], [0])
        self.assertEqual(stats["selected_visual_tokens"], [0])
        self.assertIsNone(stats["mean_cosine_distance"])
        self.assertIsNone(stats["selection_scale"])
        self.assertFalse(stats["alpha_effective"])

    def test_text_only_rejects_visual_tokens(self) -> None:
        sample = atv.LayerSample(
            hidden_states=torch.zeros(1, 2, 3),
            layer_kwargs={},
            valid_mask=torch.ones(1, 2, dtype=torch.bool),
            visual_mask=torch.tensor([[False, True]]),
            sample_id="bad",
        )
        with self.assertRaises(atv.ProtocolError):
            atv.compute_atv_text_only_selection([sample], alpha=1.0)

    def test_activation_stat_is_sample_normalized(self) -> None:
        stats = atv.ActivationStats(columns=2)
        stats.add(torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[True, False]]))
        stats.add(torch.tensor([[[2.0, 1.0], [9.0, 9.0]]]), torch.tensor([[True, False]]))
        self.assertTrue(torch.equal(stats.scaler_row, torch.tensor([2.5, 2.5])))
        self.assertEqual(stats.nsamples, 2)

    def test_text_only_record_rejects_image_field_and_placeholder(self) -> None:
        with self.assertRaises(atv.ProtocolError):
            atv.assert_text_only_record({"text": "question", "image": "x.jpg"}, "test")
        with self.assertRaises(atv.ProtocolError):
            atv.assert_text_only_record({"text": "<image> question"}, "test")


if __name__ == "__main__":
    unittest.main()
