"""Batch-mean/deviation targets, losses, and complete-production-batch sampling.

Packed targets have columns [part weight, batch mean, deviation, batch code].
Batch codes are labels for losses/sampling only; they are never model inputs.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler


LOSS_VARIANTS = ("components", "components_weight", "components_weight_balance")
DEFAULT_BETAS = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]


def resolve_decomposition(config: dict) -> tuple[list[float], list[dict]]:
    raw = config.get("beta", DEFAULT_BETAS)
    betas = [float(v) for v in (raw if isinstance(raw, (list, tuple)) else [raw])]
    if (not betas or len(set(betas)) != len(betas)
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in betas)):
        raise ValueError("decomposition.beta must be a scalar or unique nonempty list in [0, 1]")
    gamma = float(config.get("gamma", 1.0))
    balance = float(config.get("balance_lambda", 1.0))
    if any(not math.isfinite(v) or v < 0 for v in (gamma, balance)):
        raise ValueError("gamma and balance_lambda must be nonnegative and finite")
    variants = config.get("variants", list(LOSS_VARIANTS))
    if (not variants or len(set(variants)) != len(variants)
            or any(v not in LOSS_VARIANTS for v in variants)):
        raise ValueError(f"decomposition.variants must contain unique names from {LOSS_VARIANTS}")
    settings = [{"variant": variant, "beta": beta,
                 "gamma": gamma if variant != "components" else 0.0,
                 "balance_lambda": balance if variant == "components_weight_balance" else 0.0}
                for variant in variants for beta in betas]
    return betas, settings


def part_weights(targets):
    return targets[:, 0] if targets.ndim == 2 else targets


def pack_batch_targets(weights, batch_ids) -> np.ndarray:
    """Compute actual batch means, never nominal setpoints or predicted means.

    Callers must keep entire batches on one side of every train/val/test split.
    This ensures training targets never depend on held-out labels.
    """
    weights = np.asarray(weights, dtype=np.float32)
    batch_ids = np.asarray(batch_ids)
    if (weights.ndim != 1 or batch_ids.shape != weights.shape
            or not np.isfinite(weights).all() or pd.isna(batch_ids).any()):
        raise ValueError("Batch decomposition requires finite weights and one nonmissing batch ID per part")
    codes, _ = pd.factorize(batch_ids, sort=False)
    if codes.max(initial=0) >= 2 ** 24:
        raise ValueError("Too many batches for the float32 packed target representation")
    means = pd.Series(weights).groupby(codes).transform("mean").to_numpy(dtype=np.float32)
    return np.column_stack([weights, means, weights - means, codes]).astype(np.float32)


def decomposition_loss(predictions: torch.Tensor, targets: torch.Tensor, settings: dict):
    """Sample-weighted component MAEs plus optional total and batch-balance terms."""
    if predictions.ndim != 2 or predictions.shape[1] != 2 or targets.shape != (len(predictions), 4):
        raise ValueError("Expected (N,2) [mean, deviation] predictions and (N,4) packed targets")
    avg, dev = predictions.unbind(dim=1)
    mae_avg = (avg - targets[:, 1]).abs().mean()
    mae_dev = (dev - targets[:, 2]).abs().mean()
    mae_w = (avg + dev - targets[:, 0]).abs().mean()
    _, inverse, counts = torch.unique(targets[:, 3], return_inverse=True, return_counts=True)
    batch_sums = dev.new_zeros(len(counts)).index_add(0, inverse, dev)
    balance = (batch_sums / counts.to(dev.dtype)).abs().mean()
    beta = settings["beta"]
    loss = (beta * mae_avg + (1-beta) * mae_dev
            + settings.get("gamma", 0) * mae_w
            + settings.get("balance_lambda", 0) * balance)
    return loss, {"mae_avg": mae_avg, "mae_dev": mae_dev,
                  "mae_w": mae_w, "balance_loss": balance}


class WholeGroupBatchSampler(Sampler):
    """Pack whole production batches; batch_size is a soft sample-count limit.

    A production batch larger than batch_size is yielded intact. This avoids
    penalizing a random fragment's deviation mean as if it were a whole batch.
    """
    def __init__(self, batch_codes, batch_size: int, generator=None):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        codes = np.asarray(batch_codes)
        self.groups = [np.flatnonzero(codes == c).tolist() for c in pd.unique(codes)]
        self.batch_size = batch_size
        self.generator = generator

    def __iter__(self):
        pending = []
        for group_id in torch.randperm(len(self.groups), generator=self.generator).tolist():
            group = self.groups[group_id]
            if pending and len(pending) + len(group) > self.batch_size:
                yield pending
                pending = []
            pending.extend(group)
        if pending:
            yield pending


def feature_drop_columns(data_cfg: dict) -> list[str]:
    """Keep nominal PPs as inputs when requested, while retaining edge metadata."""
    edge_cols = data_cfg.get("edge_group_split", {}).get("columns", [])
    requested = list(dict.fromkeys([*data_cfg.get("drop_cols", []), *edge_cols]))
    if data_cfg.get("include_nominal_pp", False):
        nominal = set(data_cfg.get("nominal_pp_cols", edge_cols))
        requested = [col for col in requested if col not in nominal]
    return requested


def copy_artifacts(source: Path, destination: Path):
    """Copy a selected checkpoint's files, excluding sweep subdirectories."""
    destination.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    for path in source.iterdir():
        if path.is_file() and path.name != "experiment_variants.json":
            shutil.copy2(path, destination / path.name)


def promote_m1_candidate(candidate: Path, destination: Path) -> bool:
    """Compare full-weight validation MAE, never select beta on test labels."""
    metadata = json.loads((candidate / "best_metrics.json").read_text())
    previous_path = destination / "best_metrics.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    candidate_dp = json.loads((candidate / "data_processing.json").read_text())
    previous_dp_path = destination / "data_processing.json"
    previous_dp = json.loads(previous_dp_path.read_text()) if previous_dp_path.exists() else {}
    same_contract = (previous.get("model_cfg", {}).get("weight_decomposition", False)
                     == metadata["model_cfg"].get("weight_decomposition", False)
                     and previous_dp.get("pp_cols") == candidate_dp.get("pp_cols")
                     and previous_dp.get("val_part_ids") == candidate_dp.get("val_part_ids")
                     and previous_dp.get("test_part_ids") == candidate_dp.get("test_part_ids")
                     and previous.get("signal_names") == metadata.get("signal_names"))
    if not same_contract or metadata["selection_MAE"] < previous.get("selection_MAE", math.inf):
        copy_artifacts(candidate, destination)
        return True
    return False
