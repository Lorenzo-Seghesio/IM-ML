"""Shared scalar-weight training helpers; a test point is not a mini-batch."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler


class InsufficientVariabilityWarning(UserWarning):
    """Training targets cannot reliably normalize one of the loss terms."""


def encode_test_points(test_point_ids):
    ids = np.asarray(test_point_ids)
    if ids.ndim != 1 or len(ids) == 0 or pd.isna(ids).any():
        raise ValueError("Test-point IDs must be a nonempty 1D array without missing IDs")
    codes, labels = pd.factorize(ids, sort=False)
    return codes, labels


def normalization_factors(training_weights, training_test_point_ids, epsilon=1e-8):
    """Fit fixed factors using ONLY the actual training portion of one fold."""
    weights = np.asarray(training_weights, dtype=np.float64).reshape(-1)
    codes, labels = encode_test_points(training_test_point_ids)
    if len(codes) != len(weights) or not np.isfinite(weights).all():
        raise ValueError("Training weights and test-point IDs must be finite and aligned")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("normalization_epsilon must be finite and positive")
    s_w = float(np.abs(weights - weights.mean()).mean())
    s_within = float(np.mean([
        np.abs(weights[codes == j] - weights[codes == j].mean()).mean()
        for j in range(len(labels))
    ]))
    for name, value in (("s_w", s_w), ("s_within", s_within)):
        if value < epsilon:
            warnings.warn(
                f"Training data have insufficient variability: {name}={value:g}; "
                f"using floor {epsilon:g} for normalization.",
                InsufficientVariabilityWarning, stacklevel=2)
    return {"s_w": max(s_w, epsilon), "s_within": max(s_within, epsilon),
            "raw_s_w": s_w, "raw_s_within": s_within, "epsilon": epsilon,
            "n_training_parts": len(weights), "n_training_test_points": len(labels)}


def test_point_loss(predicted, actual, test_point_codes, beta, factors):
    """Return combined loss, part MAE, and equally TP-weighted centered MAE.

    Both means use exactly the same mini-batch subset. Predicted means stay
    attached to autograd, including for unequal test-point sizes.
    """
    predicted, actual = predicted.reshape(-1), actual.reshape(-1)
    codes = test_point_codes.reshape(-1)
    if predicted.numel() == 0 or predicted.shape != actual.shape or codes.shape != actual.shape:
        raise ValueError("Predictions, targets and test-point codes must be nonempty and aligned")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("within_test_point_beta must be finite and nonnegative")
    weight_mae = (predicted - actual).abs().mean()
    within_mae = torch.stack([
        ((predicted[codes == tp] - predicted[codes == tp].mean())
         - (actual[codes == tp] - actual[codes == tp].mean())).abs().mean()
        for tp in torch.unique(codes)
    ]).mean()
    # Preserve the original MAE exactly at beta=0, even with floored factors.
    combined = weight_mae if beta == 0 else (
        weight_mae + beta * (factors["s_w"] / factors["s_within"]) * within_mae)
    return combined, weight_mae, within_mae


class TestPointMiniBatchSampler(Sampler):
    """Random TP/part queues with full epoch coverage and no within-batch duplicates.

    Consume shuffled per-TP queues before revisiting parts. Top up exhausted
    queues only to finish a mini-batch or cover the epoch tail. Small TPs use
    every available distinct part, yielding explicitly warned smaller batches.
    At least two distinct TPs are always required in every mini-batch.
    """
    def __init__(self, test_point_ids, mini_batch_size=64,
                 test_points_per_mini_batch=4, parts_per_test_point=16, seed=42):
        self.codes, labels = encode_test_points(test_point_ids)
        self.tp_count = test_points_per_mini_batch
        self.parts_per_tp = parts_per_test_point
        for value in (mini_batch_size, self.tp_count, self.parts_per_tp):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError("Mini-batch sampling sizes must be positive integers")
        if self.tp_count < 2 or mini_batch_size != self.tp_count * self.parts_per_tp:
            raise ValueError("batch_size must equal test_points_per_mini_batch * "
                             "parts_per_test_point, with at least two test points")
        if len(labels) < self.tp_count:
            raise ValueError(f"Training split contains {len(labels)} test points; "
                             f"{self.tp_count} distinct test points are required per mini-batch")
        self.indices = [np.flatnonzero(self.codes == tp) for tp in range(len(labels))]
        short = [str(labels[tp]) for tp, idx in enumerate(self.indices) if len(idx) < self.parts_per_tp]
        if short:
            warnings.warn("Test points with fewer than parts_per_test_point use all distinct "
                          f"parts and smaller mini-batches (no exclusion/replacement): {short}",
                          UserWarning, stacklevel=2)
        self.seed, self.epoch = seed, 0

    def _mini_batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        queues = [list(rng.permutation(idx)) for idx in self.indices]
        mini_batches = []
        chunks = np.array([(len(queue) + self.parts_per_tp - 1) // self.parts_per_tp
                           for queue in queues])
        # Minimum feasible epoch length with distinct TPs in each mini-batch.
        # TPs needing every remaining step are mandatory; randomize all other
        # choices. This avoids stranding one TP and needlessly repeating others.
        remaining_steps = max(int(chunks.max()),
                              (int(chunks.sum()) + self.tp_count - 1) // self.tp_count)
        while any(queues):
            active = [tp for tp, queue in enumerate(queues) if queue]
            selected = [tp for tp in active if chunks[tp] == remaining_steps]
            candidates = [tp for tp in active if tp not in selected]
            take_tps = min(self.tp_count - len(selected), len(candidates))
            selected += list(rng.choice(candidates, take_tps, replace=False))
            if len(selected) < self.tp_count:
                remaining = [tp for tp in range(len(queues)) if tp not in selected]
                selected += list(rng.choice(remaining, self.tp_count - len(selected), replace=False))
            mini_batch = []
            for tp in selected:
                take = min(self.parts_per_tp, len(queues[tp]))
                chosen = queues[tp][:take]
                del queues[tp][:take]
                if take:
                    chunks[tp] -= 1
                required = min(self.parts_per_tp, len(self.indices[tp])) - take
                if required:
                    candidates = np.setdiff1d(self.indices[tp], chosen)
                    chosen += list(rng.choice(candidates, required, replace=False))
                mini_batch.extend(chosen)
            rng.shuffle(mini_batch)
            mini_batches.append(mini_batch)
            remaining_steps -= 1
        return mini_batches

    def __iter__(self):
        mini_batches = self._mini_batches()
        self.epoch += 1
        return iter(mini_batches)

    def __len__(self):
        return len(self._mini_batches())


def test_point_metrics(actual, predicted, test_point_ids):
    """Retrospective raw-gram metrics using complete evaluated TP groups.

    Standard deviations use the population convention (ddof=0), including
    singleton TPs. No TP target mean is ever supplied to the network.
    """
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    codes, labels = encode_test_points(test_point_ids)
    if actual.shape != predicted.shape or len(codes) != len(actual):
        raise ValueError("Evaluation weights and test-point IDs must be aligned")
    rows = []
    for tp, label in enumerate(labels):
        a, p = actual[codes == tp], predicted[codes == tp]
        rows.append({"test_point_id": str(label), "n_parts": len(a),
                     "actual_mean_g": float(a.mean()), "predicted_mean_g": float(p.mean()),
                     "actual_std_g": float(a.std(ddof=0)), "predicted_std_g": float(p.std(ddof=0)),
                     "MAE_TPavg": float(abs(p.mean() - a.mean())),
                     "MAE_within": float(np.abs((p - p.mean()) - (a - a.mean())).mean())})
    return {"MAE_w": float(np.abs(predicted - actual).mean()),
            "MAE_TPavg": float(np.mean([row["MAE_TPavg"] for row in rows])),
            "MAE_within": float(np.mean([row["MAE_within"] for row in rows])),
            "test_points": rows}
