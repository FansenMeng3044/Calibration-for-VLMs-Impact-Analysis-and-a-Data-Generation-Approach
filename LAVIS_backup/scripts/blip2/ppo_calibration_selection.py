#!/usr/bin/env python3
"""PPO calibration selection from Wanda W_metric and mask distances.

The environment starts from an empty calibration subset.  At every step the
action is one unselected sample index.  The reward is the incremental change of

    R(S) = lambda_cov * Coverage(S) - lambda_red * Redundancy(S)

where Coverage and Redundancy are computed from a pairwise similarity matrix
derived from D_final = alpha * D_wmetric + beta * D_mask.

This script intentionally consumes precomputed pairwise distances.  For BLIP2
T5-XL, storing a full W_metric matrix per sample is too large; compute pairwise
distances offline, then use this lightweight selector to choose the subset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DISTANCE_KEYS = (
    "D_final",
    "D",
    "distance",
    "distances",
    "pairwise_distance",
    "pairwise_distances",
)
WMETRIC_KEYS = (
    "D_wmetric",
    "wmetric_distance",
    "wmetric_distances",
    "D",
    "distance",
)
MASK_KEYS = (
    "D_mask",
    "mask_distance",
    "mask_distances",
    "D",
    "distance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a calibration subset with sequential PPO over W_metric/mask distances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--calib_json", required=True, help="Original candidate calibration JSON/JSONL.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out_calib_json", default="", help="Selected calibration JSON. Defaults under out_dir.")
    parser.add_argument("--select_k", type=int, required=True, help="Number of samples selected from the candidate pool.")
    parser.add_argument("--max_candidates", type=int, default=None, help="Use only the first N candidates/dist rows.")
    parser.add_argument(
        "--candidate_indices_json",
        default="",
        help="Optional JSON list mapping distance-matrix rows to original calib_json row indices.",
    )

    source = parser.add_argument_group("distance inputs")
    source.add_argument("--distance_npz", default="", help="NPZ/NPY/CSV/JSON containing precomputed D_final.")
    source.add_argument("--wmetric_distance_npz", default="", help="Distance file for D_wmetric.")
    source.add_argument("--mask_distance_npz", default="", help="Distance file for D_mask.")
    source.add_argument("--distance_key", default="", help="Array key for --distance_npz.")
    source.add_argument("--wmetric_key", default="", help="Array key for --wmetric_distance_npz.")
    source.add_argument("--mask_key", default="", help="Array key for --mask_distance_npz.")
    source.add_argument("--alpha", type=float, default=0.7, help="Weight for D_wmetric.")
    source.add_argument("--beta", type=float, default=0.3, help="Weight for D_mask.")
    source.add_argument(
        "--normalize_distances",
        choices=["percentile", "max", "none"],
        default="percentile",
        help="Normalize D_wmetric and D_mask before alpha/beta combination.",
    )
    source.add_argument("--normalization_percentile", type=float, default=95.0)
    source.add_argument("--tau", type=float, default=0.0, help="Similarity temperature. 0 means median nonzero D_final.")

    reward = parser.add_argument_group("reward")
    reward.add_argument("--lambda_cov", type=float, default=1.0)
    reward.add_argument("--lambda_red", type=float, default=1.0)

    ppo = parser.add_argument_group("ppo")
    ppo.add_argument("--train", action="store_true", help="Train PPO before selecting. If omitted, run greedy only.")
    ppo.add_argument("--updates", type=int, default=200)
    ppo.add_argument("--episodes_per_update", type=int, default=16)
    ppo.add_argument("--ppo_epochs", type=int, default=4)
    ppo.add_argument("--minibatch_size", type=int, default=256)
    ppo.add_argument("--hidden_dim", type=int, default=128)
    ppo.add_argument("--lr", type=float, default=3e-4)
    ppo.add_argument("--gamma", type=float, default=1.0)
    ppo.add_argument("--gae_lambda", type=float, default=0.95)
    ppo.add_argument("--clip_range", type=float, default=0.2)
    ppo.add_argument("--entropy_coef", type=float, default=0.01)
    ppo.add_argument("--value_coef", type=float, default=0.5)
    ppo.add_argument("--max_grad_norm", type=float, default=1.0)
    ppo.add_argument("--eval_rollouts", type=int, default=32)
    ppo.add_argument("--deterministic_eval", action="store_true")
    ppo.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")

    misc = parser.add_argument_group("misc")
    misc.add_argument("--seed", type=int, default=42)
    misc.add_argument("--also_write_greedy", action="store_true")
    misc.add_argument("--no_plots", action="store_true")
    misc.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def load_rows(path: str) -> List[Any]:
    if path.lower().endswith(".jsonl"):
        rows: List[Any] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError("%s must contain a JSON list or JSONL rows." % path)
    return rows


def write_rows(path: str, rows: Sequence[Any]) -> None:
    ensure_dir(os.path.dirname(path))
    if path.lower().endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
        return
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(list(rows), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_array_file(path: str, keys: Sequence[str], explicit_key: str = "") -> np.ndarray:
    if not path:
        raise ValueError("Empty array path.")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.asarray(np.load(path), dtype=np.float64)
    if ext == ".npz":
        data = np.load(path)
        if explicit_key:
            if explicit_key not in data.files:
                raise KeyError("Key %r not found in %s. Available: %s" % (explicit_key, path, ", ".join(data.files)))
            return np.asarray(data[explicit_key], dtype=np.float64)
        for key in keys:
            if key in data.files:
                return np.asarray(data[key], dtype=np.float64)
        matrix_keys = [key for key in data.files if np.asarray(data[key]).ndim == 2]
        if len(matrix_keys) == 1:
            return np.asarray(data[matrix_keys[0]], dtype=np.float64)
        raise KeyError("Could not infer distance key in %s. Available: %s" % (path, ", ".join(data.files)))
    if ext == ".csv":
        return np.loadtxt(path, delimiter=",", dtype=np.float64)
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            obj = json.load(handle)
        if explicit_key:
            obj = obj[explicit_key]
        elif isinstance(obj, dict):
            for key in keys:
                if key in obj:
                    obj = obj[key]
                    break
        return np.asarray(obj, dtype=np.float64)
    raise ValueError("Unsupported distance file extension: %s" % path)


def validate_distance(name: str, matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("%s must be square [N,N], got %s." % (name, tuple(matrix.shape)))
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("%s contains NaN or inf." % name)
    matrix = np.maximum(matrix, 0.0)
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def nonzero_values(matrix: np.ndarray) -> np.ndarray:
    values = matrix[np.triu_indices(matrix.shape[0], k=1)]
    return values[values > 0]


def normalize_distance(matrix: np.ndarray, mode: str, percentile: float) -> np.ndarray:
    if mode == "none":
        return matrix
    values = nonzero_values(matrix)
    if values.size == 0:
        return matrix.copy()
    if mode == "max":
        scale = float(values.max())
    else:
        scale = float(np.percentile(values, percentile))
    if scale <= 0 or not math.isfinite(scale):
        return matrix.copy()
    return matrix / scale


def load_final_distance(args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "normalize_distances": args.normalize_distances,
        "normalization_percentile": float(args.normalization_percentile),
    }
    if args.distance_npz:
        d_final = read_array_file(args.distance_npz, DISTANCE_KEYS, args.distance_key)
        d_final = validate_distance("D_final", d_final)
        meta["distance_npz"] = os.path.abspath(args.distance_npz)
        meta["distance_source"] = "precomputed_final"
        return d_final, meta

    if not args.wmetric_distance_npz and not args.mask_distance_npz:
        raise ValueError("Provide --distance_npz or at least one of --wmetric_distance_npz/--mask_distance_npz.")

    parts: List[Tuple[float, str, np.ndarray]] = []
    if args.wmetric_distance_npz:
        d_w = validate_distance(
            "D_wmetric",
            read_array_file(args.wmetric_distance_npz, WMETRIC_KEYS, args.wmetric_key),
        )
        parts.append((float(args.alpha), "D_wmetric", normalize_distance(d_w, args.normalize_distances, args.normalization_percentile)))
        meta["wmetric_distance_npz"] = os.path.abspath(args.wmetric_distance_npz)
    if args.mask_distance_npz:
        d_m = validate_distance(
            "D_mask",
            read_array_file(args.mask_distance_npz, MASK_KEYS, args.mask_key),
        )
        parts.append((float(args.beta), "D_mask", normalize_distance(d_m, args.normalize_distances, args.normalization_percentile)))
        meta["mask_distance_npz"] = os.path.abspath(args.mask_distance_npz)

    shape = parts[0][2].shape
    if any(part[2].shape != shape for part in parts):
        raise ValueError("D_wmetric and D_mask shapes do not match.")
    weight_sum = sum(max(weight, 0.0) for weight, _name, _matrix in parts)
    if weight_sum <= 0:
        raise ValueError("alpha/beta weights must include at least one positive value.")
    d_final = np.zeros(shape, dtype=np.float64)
    for weight, _name, matrix in parts:
        if weight > 0:
            d_final += weight * matrix
    d_final /= weight_sum
    np.fill_diagonal(d_final, 0.0)
    meta["distance_source"] = "combined_parts"
    return validate_distance("D_final", d_final), meta


def infer_tau(d_final: np.ndarray, user_tau: float) -> float:
    if user_tau and user_tau > 0:
        return float(user_tau)
    values = nonzero_values(d_final)
    if values.size == 0:
        return 1.0
    tau = float(np.median(values))
    return tau if tau > 0 and math.isfinite(tau) else 1.0


def make_similarity(d_final: np.ndarray, tau: float) -> np.ndarray:
    sim = np.exp(-d_final / max(tau, 1e-12))
    sim = 0.5 * (sim + sim.T)
    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float64)


@dataclass
class RewardState:
    selected: np.ndarray
    coverage_vector: np.ndarray
    pair_sum: float
    selected_count: int
    coverage: float
    redundancy: float
    reward: float


class CalibrationSelectionEnv:
    def __init__(self, d_final: np.ndarray, sim: np.ndarray, select_k: int, lambda_cov: float, lambda_red: float):
        self.d_final = d_final
        self.sim = sim
        self.n = int(sim.shape[0])
        self.select_k = int(select_k)
        self.lambda_cov = float(lambda_cov)
        self.lambda_red = float(lambda_red)
        values = nonzero_values(d_final)
        self.dist_scale = float(np.percentile(values, 95)) if values.size else 1.0
        if self.dist_scale <= 0 or not math.isfinite(self.dist_scale):
            self.dist_scale = 1.0
        self.reset()

    def reset(self) -> RewardState:
        self.selected = np.zeros((self.n,), dtype=bool)
        self.coverage_vector = np.zeros((self.n,), dtype=np.float64)
        self.pair_sum = 0.0
        self.selected_count = 0
        self.coverage = 0.0
        self.redundancy = 0.0
        self.reward_value = 0.0
        return self.current_reward_state()

    def current_reward_state(self) -> RewardState:
        return RewardState(
            selected=self.selected.copy(),
            coverage_vector=self.coverage_vector.copy(),
            pair_sum=float(self.pair_sum),
            selected_count=int(self.selected_count),
            coverage=float(self.coverage),
            redundancy=float(self.redundancy),
            reward=float(self.reward_value),
        )

    def reward_from_parts(self, coverage: float, redundancy: float) -> float:
        return self.lambda_cov * coverage - self.lambda_red * redundancy

    def redundancy_from_pair_sum(self, pair_sum: float, count: int) -> float:
        if count < 2:
            return 0.0
        return float(pair_sum / (count * (count - 1) / 2.0))

    def candidate_features(self) -> np.ndarray:
        features = np.zeros((self.n, 10), dtype=np.float32)
        selected_float = self.selected.astype(np.float64)
        features[:, 0] = selected_float.astype(np.float32)
        features[:, 1] = self.coverage_vector.astype(np.float32)

        if self.selected_count > 0:
            selected_idx = np.flatnonzero(self.selected)
            dist_to_selected = self.d_final[:, selected_idx]
            sim_to_selected = self.sim[:, selected_idx]
            min_dist = dist_to_selected.min(axis=1) / self.dist_scale
            max_sim = sim_to_selected.max(axis=1)
            mean_sim = sim_to_selected.mean(axis=1)
        else:
            mean_dist = self.d_final.mean(axis=1) / self.dist_scale
            min_dist = mean_dist
            max_sim = np.zeros((self.n,), dtype=np.float64)
            mean_sim = np.zeros((self.n,), dtype=np.float64)

        features[:, 2] = min_dist.astype(np.float32)
        features[:, 3] = max_sim.astype(np.float32)
        features[:, 4] = mean_sim.astype(np.float32)

        current_coverage = self.coverage
        current_redundancy = self.redundancy
        for i in range(self.n):
            if self.selected[i]:
                coverage_gain = 0.0
                redundancy_increase = 0.0
            else:
                new_coverage = float(np.maximum(self.coverage_vector, self.sim[:, i]).mean())
                coverage_gain = new_coverage - current_coverage
                if self.selected_count == 0:
                    new_redundancy = 0.0
                else:
                    add_pair_sum = float(self.sim[i, self.selected].sum())
                    new_redundancy = self.redundancy_from_pair_sum(
                        self.pair_sum + add_pair_sum,
                        self.selected_count + 1,
                    )
                redundancy_increase = new_redundancy - current_redundancy
            features[i, 5] = float(coverage_gain)
            features[i, 6] = float(redundancy_increase)

        progress = self.selected_count / max(self.select_k, 1)
        features[:, 7] = float(progress)
        features[:, 8] = float(self.coverage)
        features[:, 9] = float(self.redundancy)
        return features

    def step(self, action: int) -> Tuple[RewardState, float, bool, Dict[str, Any]]:
        action = int(action)
        if action < 0 or action >= self.n:
            raise IndexError("Action index out of range: %d" % action)
        if self.selected[action]:
            raise ValueError("Action %d was already selected." % action)

        old_reward = self.reward_value
        if self.selected_count > 0:
            self.pair_sum += float(self.sim[action, self.selected].sum())
        self.selected[action] = True
        self.selected_count += 1
        self.coverage_vector = np.maximum(self.coverage_vector, self.sim[:, action])
        self.coverage = float(self.coverage_vector.mean())
        self.redundancy = self.redundancy_from_pair_sum(self.pair_sum, self.selected_count)
        self.reward_value = self.reward_from_parts(self.coverage, self.redundancy)
        reward_delta = self.reward_value - old_reward
        done = self.selected_count >= self.select_k
        info = {
            "coverage": self.coverage,
            "redundancy": self.redundancy,
            "objective": self.reward_value,
            "selected_count": self.selected_count,
        }
        return self.current_reward_state(), float(reward_delta), done, info


def greedy_select(env: CalibrationSelectionEnv) -> Tuple[List[int], List[Dict[str, Any]]]:
    env.reset()
    selected: List[int] = []
    trace: List[Dict[str, Any]] = []
    for step in range(env.select_k):
        features = env.candidate_features()
        gains = features[:, 5] * env.lambda_cov - features[:, 6] * env.lambda_red
        gains[env.selected] = -np.inf
        action = int(np.argmax(gains))
        _state, reward_delta, done, info = env.step(action)
        selected.append(action)
        trace.append(
            {
                "step": step + 1,
                "action": action,
                "reward_delta": reward_delta,
                "coverage": info["coverage"],
                "redundancy": info["redundancy"],
                "objective": info["objective"],
                "method": "greedy",
            }
        )
        if done:
            break
    return selected, trace


def require_torch() -> Any:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PPO training requires PyTorch. Install torch or run without --train for greedy.") from exc
    return torch, nn, F


def resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


class ActorCriticModule:  # real base class assigned dynamically after torch import
    pass


def build_actor_critic(torch: Any, nn: Any, feature_dim: int, hidden_dim: int) -> Any:
    class ActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            self.actor = nn.Linear(hidden_dim, 1)
            self.critic = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, states: Any) -> Tuple[Any, Any]:
            # states: [B,N,F]
            emb = self.encoder(states)
            logits = self.actor(emb).squeeze(-1)
            values = self.critic(emb.mean(dim=1)).squeeze(-1)
            selected_mask = states[:, :, 0] > 0.5
            logits = logits.masked_fill(selected_mask, -1.0e9)
            return logits, values

    return ActorCritic()


def compute_gae(rewards: List[float], values: List[float], gamma: float, gae_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros((len(rewards),), dtype=np.float32)
    last_gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * gae_lambda * last_gae
        advantages[t] = float(last_gae)
        next_value = values[t]
    returns = advantages + np.asarray(values, dtype=np.float32)
    return returns.astype(np.float32), advantages.astype(np.float32)


def collect_episode(
    env: CalibrationSelectionEnv,
    model: Any,
    torch: Any,
    device: str,
    gamma: float,
    gae_lambda: float,
) -> Dict[str, Any]:
    env.reset()
    states: List[np.ndarray] = []
    actions: List[int] = []
    log_probs: List[float] = []
    values: List[float] = []
    rewards: List[float] = []
    trace: List[Dict[str, Any]] = []

    for step in range(env.select_k):
        state_np = env.candidate_features()
        state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, value = model(state)
            dist = torch.distributions.Categorical(logits=logits.squeeze(0))
            action_tensor = dist.sample()
            log_prob = dist.log_prob(action_tensor)
        action = int(action_tensor.item())
        _next_state, reward_delta, done, info = env.step(action)
        states.append(state_np)
        actions.append(action)
        log_probs.append(float(log_prob.item()))
        values.append(float(value.item()))
        rewards.append(float(reward_delta))
        trace.append(
            {
                "step": step + 1,
                "action": action,
                "reward_delta": float(reward_delta),
                "coverage": info["coverage"],
                "redundancy": info["redundancy"],
                "objective": info["objective"],
                "method": "ppo_train_sample",
            }
        )
        if done:
            break

    returns, advantages = compute_gae(rewards, values, gamma, gae_lambda)
    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "old_log_probs": np.asarray(log_probs, dtype=np.float32),
        "values": np.asarray(values, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "returns": returns,
        "advantages": advantages,
        "trace": trace,
        "final_objective": trace[-1]["objective"] if trace else 0.0,
        "final_coverage": trace[-1]["coverage"] if trace else 0.0,
        "final_redundancy": trace[-1]["redundancy"] if trace else 0.0,
    }


def train_ppo(env: CalibrationSelectionEnv, args: argparse.Namespace) -> Tuple[Any, List[Dict[str, Any]], str]:
    torch, nn, _F = require_torch()
    device = resolve_device(torch, args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    feature_dim = int(env.candidate_features().shape[1])
    model = build_actor_critic(torch, nn, feature_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    training_rows: List[Dict[str, Any]] = []

    for update in range(1, args.updates + 1):
        episodes = [
            collect_episode(env, model, torch, device, args.gamma, args.gae_lambda)
            for _ in range(args.episodes_per_update)
        ]
        states = np.concatenate([ep["states"] for ep in episodes], axis=0)
        actions = np.concatenate([ep["actions"] for ep in episodes], axis=0)
        old_log_probs = np.concatenate([ep["old_log_probs"] for ep in episodes], axis=0)
        returns = np.concatenate([ep["returns"] for ep in episodes], axis=0)
        advantages = np.concatenate([ep["advantages"] for ep in episodes], axis=0)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.tensor(states, dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)

        total = int(states_t.shape[0])
        indices = np.arange(total)
        losses: List[float] = []
        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []

        for _epoch in range(args.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, total, args.minibatch_size):
                batch_idx_np = indices[start : start + args.minibatch_size]
                batch_idx = torch.tensor(batch_idx_np, dtype=torch.long, device=device)
                logits, values = model(states_t[batch_idx])
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions_t[batch_idx])
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_log_probs - old_log_probs_t[batch_idx])
                unclipped = ratio * advantages_t[batch_idx]
                clipped = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * advantages_t[batch_idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = (returns_t[batch_idx] - values).pow(2).mean()
                loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                losses.append(float(loss.detach().cpu().item()))
                policy_losses.append(float(policy_loss.detach().cpu().item()))
                value_losses.append(float(value_loss.detach().cpu().item()))
                entropies.append(float(entropy.detach().cpu().item()))

        row = {
            "update": update,
            "mean_episode_objective": float(np.mean([ep["final_objective"] for ep in episodes])),
            "mean_episode_coverage": float(np.mean([ep["final_coverage"] for ep in episodes])),
            "mean_episode_redundancy": float(np.mean([ep["final_redundancy"] for ep in episodes])),
            "loss": float(np.mean(losses)) if losses else 0.0,
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }
        training_rows.append(row)
        if args.log_every and (update == 1 or update % args.log_every == 0 or update == args.updates):
            print(
                "[PPO] update %d/%d objective=%.6f coverage=%.6f redundancy=%.6f"
                % (update, args.updates, row["mean_episode_objective"], row["mean_episode_coverage"], row["mean_episode_redundancy"])
            )

    return model, training_rows, device


def rollout_policy(
    env: CalibrationSelectionEnv,
    model: Any,
    torch: Any,
    device: str,
    deterministic: bool,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    env.reset()
    selected: List[int] = []
    trace: List[Dict[str, Any]] = []
    for step in range(env.select_k):
        state_np = env.candidate_features()
        state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _value = model(state)
            if deterministic:
                action = int(torch.argmax(logits.squeeze(0)).item())
            else:
                dist = torch.distributions.Categorical(logits=logits.squeeze(0))
                action = int(dist.sample().item())
        _state, reward_delta, done, info = env.step(action)
        selected.append(action)
        trace.append(
            {
                "step": step + 1,
                "action": action,
                "reward_delta": reward_delta,
                "coverage": info["coverage"],
                "redundancy": info["redundancy"],
                "objective": info["objective"],
                "method": "ppo",
            }
        )
        if done:
            break
    return selected, trace


def select_with_ppo(env: CalibrationSelectionEnv, model: Any, device: str, args: argparse.Namespace) -> Tuple[List[int], List[Dict[str, Any]]]:
    torch, _nn, _F = require_torch()
    best_selected: List[int] = []
    best_trace: List[Dict[str, Any]] = []
    best_objective = -float("inf")

    rollout_count = 1 if args.deterministic_eval else max(1, int(args.eval_rollouts))
    for rollout in range(rollout_count):
        selected, trace = rollout_policy(
            env,
            model,
            torch,
            device,
            deterministic=bool(args.deterministic_eval),
        )
        objective = float(trace[-1]["objective"]) if trace else -float("inf")
        if objective > best_objective:
            best_objective = objective
            best_selected = selected
            best_trace = trace
        if args.deterministic_eval:
            break

    if not args.deterministic_eval:
        det_selected, det_trace = rollout_policy(env, model, torch, device, deterministic=True)
        det_objective = float(det_trace[-1]["objective"]) if det_trace else -float("inf")
        if det_objective > best_objective:
            best_selected = det_selected
            best_trace = det_trace

    return best_selected, best_trace


def make_plots(out_dir: str, training_rows: Sequence[Dict[str, Any]], final_trace: Sequence[Dict[str, Any]]) -> List[str]:
    paths: List[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print("[WARN] matplotlib unavailable, skipping plots: %s" % exc)
        return paths

    if training_rows:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot([row["update"] for row in training_rows], [row["mean_episode_objective"] for row in training_rows])
        ax.set_xlabel("PPO update")
        ax.set_ylabel("Mean episode objective")
        ax.set_title("PPO Calibration Selection Training")
        ax.grid(True, alpha=0.25)
        path = os.path.join(out_dir, "ppo_training_objective.png")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    if final_trace:
        fig, ax = plt.subplots(figsize=(10, 5))
        steps = [row["step"] for row in final_trace]
        ax.plot(steps, [row["objective"] for row in final_trace], label="objective")
        ax.plot(steps, [row["coverage"] for row in final_trace], label="coverage")
        ax.plot(steps, [row["redundancy"] for row in final_trace], label="redundancy")
        ax.set_xlabel("Selection step")
        ax.set_ylabel("Value")
        ax.set_title("Selected Calibration Subset Reward Trace")
        ax.legend()
        ax.grid(True, alpha=0.25)
        path = os.path.join(out_dir, "selection_reward_trace.png")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    return paths


def load_candidate_indices(path: str, n: int) -> List[int]:
    if not path:
        return list(range(n))
    with open(path, "r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if isinstance(obj, dict):
        for key in ("candidate_indices", "indices", "original_indices"):
            if key in obj:
                obj = obj[key]
                break
    if not isinstance(obj, list):
        raise ValueError("--candidate_indices_json must contain a list or a dict with candidate_indices.")
    indices = [int(v) for v in obj]
    if len(indices) < n:
        raise ValueError("candidate_indices length %d is smaller than distance N=%d." % (len(indices), n))
    return indices[:n]


def subset_distance(d_final: np.ndarray, n: int) -> np.ndarray:
    if n > d_final.shape[0]:
        raise ValueError("--max_candidates=%d exceeds distance matrix size %d." % (n, d_final.shape[0]))
    return d_final[:n, :n]


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)

    rows = load_rows(args.calib_json)
    d_final, distance_meta = load_final_distance(args)
    if args.max_candidates is not None:
        d_final = subset_distance(d_final, int(args.max_candidates))
    n = int(d_final.shape[0])
    if args.select_k < 1 or args.select_k > n:
        raise ValueError("--select_k must be in [1, %d], got %d." % (n, args.select_k))
    candidate_indices = load_candidate_indices(args.candidate_indices_json, n)
    if max(candidate_indices) >= len(rows):
        raise ValueError("candidate index exceeds calib_json rows: max=%d rows=%d." % (max(candidate_indices), len(rows)))

    tau = infer_tau(d_final, args.tau)
    sim = make_similarity(d_final, tau)
    env = CalibrationSelectionEnv(d_final, sim, args.select_k, args.lambda_cov, args.lambda_red)

    np.savez_compressed(
        os.path.join(args.out_dir, "ppo_selection_distance_matrices.npz"),
        D_final=d_final.astype(np.float32),
        Sim=sim.astype(np.float32),
        candidate_indices=np.asarray(candidate_indices, dtype=np.int64),
    )

    greedy_selected, greedy_trace = greedy_select(env)
    greedy_objective = float(greedy_trace[-1]["objective"]) if greedy_trace else 0.0
    print("[GREEDY] objective=%.6f coverage=%.6f redundancy=%.6f" % (
        greedy_trace[-1]["objective"],
        greedy_trace[-1]["coverage"],
        greedy_trace[-1]["redundancy"],
    ))

    training_rows: List[Dict[str, Any]] = []
    method = "greedy"
    selected_positions = greedy_selected
    final_trace = greedy_trace
    model_path = ""
    device = "cpu"

    if args.train:
        model, training_rows, device = train_ppo(env, args)
        ppo_selected, ppo_trace = select_with_ppo(env, model, device, args)
        ppo_objective = float(ppo_trace[-1]["objective"]) if ppo_trace else -float("inf")
        if ppo_objective >= greedy_objective:
            method = "ppo"
            selected_positions = ppo_selected
            final_trace = ppo_trace
        else:
            method = "greedy_fallback_after_ppo"
            selected_positions = greedy_selected
            final_trace = greedy_trace
        try:
            import torch

            model_path = os.path.join(args.out_dir, "ppo_calibration_selector.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_dim": int(env.candidate_features().shape[1]),
                    "hidden_dim": int(args.hidden_dim),
                    "args": vars(args),
                },
                model_path,
            )
        except Exception as exc:  # pragma: no cover
            print("[WARN] failed to save PPO model: %s" % exc)

    selected_original_indices = [candidate_indices[i] for i in selected_positions]
    selected_rows = [rows[i] for i in selected_original_indices]
    out_calib_json = args.out_calib_json or os.path.join(args.out_dir, "selected_calibration.json")
    write_rows(out_calib_json, selected_rows)

    if args.also_write_greedy:
        greedy_rows = [rows[candidate_indices[i]] for i in greedy_selected]
        write_rows(os.path.join(args.out_dir, "greedy_selected_calibration.json"), greedy_rows)
        write_csv(os.path.join(args.out_dir, "greedy_selection_trace.csv"), greedy_trace)

    write_csv(os.path.join(args.out_dir, "ppo_training_log.csv"), training_rows)
    write_csv(os.path.join(args.out_dir, "selection_trace.csv"), final_trace)
    write_json(
        os.path.join(args.out_dir, "selected_indices.json"),
        {
            "method": method,
            "selected_positions": [int(i) for i in selected_positions],
            "selected_original_indices": [int(i) for i in selected_original_indices],
            "select_k": int(args.select_k),
            "candidate_count": n,
            "objective": float(final_trace[-1]["objective"]) if final_trace else 0.0,
            "coverage": float(final_trace[-1]["coverage"]) if final_trace else 0.0,
            "redundancy": float(final_trace[-1]["redundancy"]) if final_trace else 0.0,
            "greedy_objective": greedy_objective,
            "tau": float(tau),
            "lambda_cov": float(args.lambda_cov),
            "lambda_red": float(args.lambda_red),
            "distance_metadata": distance_meta,
            "calib_json": os.path.abspath(args.calib_json),
            "out_calib_json": os.path.abspath(out_calib_json),
            "ppo_model_path": os.path.abspath(model_path) if model_path else "",
            "device": device,
        },
    )

    if not args.no_plots:
        for path in make_plots(args.out_dir, training_rows, final_trace):
            print("[OK] plot:", path)

    print("[OK] method:", method)
    print("[OK] selected calibration:", out_calib_json)
    print("[OK] selected indices:", os.path.join(args.out_dir, "selected_indices.json"))
    print("[OK] trace:", os.path.join(args.out_dir, "selection_trace.csv"))


if __name__ == "__main__":
    main()
