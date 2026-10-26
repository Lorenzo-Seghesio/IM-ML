"""Deterministic, leakage-safe whole-group splitting helpers."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from typing import Iterable

import numpy as np


def _normalise(value) -> str:
    """Return a stable, JSON-safe representation of a group value."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "<missing>"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return str(int(value)) if float(value).is_integer() else str(value)
    return str(value)


def make_group_keys(group_values: Iterable, config: dict | None = None) -> np.ndarray:
    """Normalise group IDs and optionally merge configured protected IDs."""
    config = config or {}
    keys = np.asarray([_normalise(v) for v in group_values], dtype=object)
    protected = ({_normalise(v) for v in config.get("protected_group_ids", [])}
                 if grouping_enabled(config) else set())
    if protected:
        merged = "protected:" + "+".join(sorted(protected))
        keys = np.asarray([merged if key in protected else key for key in keys], dtype=object)
    return keys


def grouping_enabled(config: dict | None) -> bool:
    # The caller obtains the configured group column from its data section;
    # this object only controls whether grouping is enabled and how groups are
    # normalised/protected.
    return bool(config and config.get("enabled", False))


def edge_grouping_enabled(config: dict | None, partition: str = "validation") -> bool:
    """Whether edge groups must be kept out of the requested partition."""
    if not config or not config.get("enabled", False):
        return False
    flag = "exclude_from_test" if partition == "test" else "exclude_from_validation"
    return bool(config.get(flag, True))


def group_split_required(group_config: dict | None,
                         edge_config: dict | None = None,
                         partition: str = "validation") -> bool:
    """Whether this partition needs whole-group splitting."""
    return grouping_enabled(group_config) or edge_grouping_enabled(edge_config, partition)


def _numeric_edge_values(edge_values, n: int) -> np.ndarray:
    """Validate and convert the configured nominal process parameters."""
    values = np.asarray(edge_values)
    if values.ndim != 2 or values.shape[0] != n or values.shape[1] == 0:
        raise ValueError("edge_values must be a non-empty 2D array with one row per sample")
    try:
        numeric = values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Edge-group nominal columns must contain numeric values") from exc
    if not np.isfinite(numeric).all():
        raise ValueError("Edge-group nominal columns cannot contain missing or non-finite values")
    return numeric


def edge_group_details(edge_values, group_values: Iterable,
                       indices: np.ndarray | None = None,
                       group_config: dict | None = None,
                       edge_config: dict | None = None) -> dict:
    """Describe groups touching any parameter minimum or maximum.

    Extrema are calculated only from ``indices`` when provided. Any row at
    any column's minimum or maximum marks its entire configured group as an
    edge group. The returned mask and group keys align with the full input.
    """
    if edge_config and not edge_config.get("enabled", False):
        n = len(group_values) if group_values is not None else len(edge_values)
        return {"enabled": False, "edge_row_mask": np.zeros(n, dtype=bool),
                "edge_group_keys": set(), "extrema": {}}

    n = len(group_values) if group_values is not None else len(edge_values)
    numeric = _numeric_edge_values(edge_values, n)
    configured_columns = (edge_config or {}).get("columns", [])
    if configured_columns and len(configured_columns) != numeric.shape[1]:
        raise ValueError(
            "edge_config.columns must contain one name for each edge_values column")
    scope = (np.arange(n, dtype=np.intp) if indices is None
             else np.asarray(indices, dtype=np.intp))
    if len(scope) == 0:
        raise ValueError("Cannot calculate edge-group extrema from an empty split input")
    if np.any(scope < 0) or np.any(scope >= n):
        raise ValueError("indices contain values outside the edge-value array")

    subset = numeric[scope]
    minima = np.min(subset, axis=0)
    maxima = np.max(subset, axis=0)
    at_edge = np.zeros(len(scope), dtype=bool)
    for column in range(numeric.shape[1]):
        values = subset[:, column]
        at_edge |= np.isclose(values, minima[column], rtol=1e-9, atol=1e-12)
        at_edge |= np.isclose(values, maxima[column], rtol=1e-9, atol=1e-12)

    row_mask = np.zeros(n, dtype=bool)
    row_mask[scope[at_edge]] = True
    if group_values is None:
        all_group_keys = np.asarray([str(i) for i in range(n)], dtype=object)
    else:
        all_group_keys = make_group_keys(group_values, group_config)
    protected_keys = set(all_group_keys[row_mask].tolist())

    columns = (edge_config or {}).get("columns", [])
    extrema = {}
    for column, low, high in zip(columns, minima.tolist(), maxima.tolist()):
        extrema[str(column)] = {"min": float(low), "max": float(high)}
    if not columns:
        extrema = {str(i): {"min": float(low), "max": float(high)}
                   for i, (low, high) in enumerate(zip(minima, maxima))}

    return {
        "enabled": True,
        "edge_row_mask": row_mask,
        "edge_group_keys": protected_keys,
        "extrema": extrema,
        "edge_row_count": int(row_mask.sum()),
        "edge_group_count": len(protected_keys),
        "edge_group_ids": sorted(protected_keys),
        "scope_size": int(len(scope)),
    }


def edge_group_policy_metadata(edge_values, group_values: Iterable,
                               edge_config: dict | None,
                               group_config: dict | None = None) -> dict:
    """Return JSON-serialisable edge policy and a stable config-contract hash."""
    config = dict(edge_config or {})
    contract = {
        "enabled": bool(config.get("enabled", False)),
        "columns": list(config.get("columns", [])),
        "exclude_from_test": bool(config.get("exclude_from_test", True)),
        "exclude_from_validation": bool(config.get("exclude_from_validation", True)),
        "grouping_enabled": grouping_enabled(group_config),
        "protected_group_ids": sorted(
            _normalise(value) for value in (group_config or {}).get("protected_group_ids", [])
        ) if grouping_enabled(group_config) else [],
        "rule": "any_parameter_min_or_max",
        "normalization_version": "numeric-v1",
    }
    policy_hash = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if not contract["enabled"]:
        return {**contract, "policy_hash": policy_hash}
    details = edge_group_details(edge_values, group_values, group_config=group_config,
                                 edge_config=config)
    return {
        **contract,
        "policy_hash": policy_hash,
        "extrema": details["extrema"],
        "edge_group_ids": details["edge_group_ids"],
        "edge_row_count": details["edge_row_count"],
        "edge_group_count": details["edge_group_count"],
    }


def validate_edge_groups_excluded(indices: np.ndarray, edge_values,
                                  group_values: Iterable,
                                  edge_config: dict | None,
                                  group_config: dict | None = None,
                                  scope_indices: np.ndarray | None = None,
                                  partition: str = "validation") -> None:
    """Raise if an edge group in the split scope occurs in a held-out partition."""
    if not edge_grouping_enabled(edge_config, partition):
        return
    details = edge_group_details(edge_values, group_values, scope_indices,
                                 group_config, edge_config)
    keys = make_group_keys(
        group_values if group_values is not None else np.arange(len(edge_values)),
        group_config)
    selected = set(keys[np.asarray(indices, dtype=np.intp)].tolist())
    overlap = selected & details["edge_group_keys"]
    if overlap:
        raise ValueError(
            f"Edge groups must remain in training, but are present in {partition}: "
            f"{sorted(overlap)}")


def _excluded_group_keys(group_values: Iterable, group_config: dict | None,
                         edge_values=None, edge_config: dict | None = None,
                         scope_indices: np.ndarray | None = None,
                         partition: str = "validation") -> set[str]:
    config = group_config or {}
    protected = ({_normalise(v) for v in config.get("protected_group_ids", [])}
                 if partition == "test" and grouping_enabled(config) else set())
    if protected:
        protected = {"protected:" + "+".join(sorted(protected))}
    if edge_values is not None and edge_grouping_enabled(edge_config, partition):
        details = edge_group_details(edge_values, group_values, scope_indices,
                                     group_config, edge_config)
        protected.update(details["edge_group_keys"])
    return protected


def _whole_group_subset(group_sizes: dict[str, int], target: float,
                        excluded: set[str], seed: int) -> set[str]:
    """Select whole groups with size closest to target, deterministically.

    Dynamic programming gives an exact closest-size subset for the usual dataset
    sizes while the seeded ordering makes ties reproducible.
    """
    rng = np.random.default_rng(seed)
    groups = sorted((g for g in group_sizes if g not in excluded),
                    key=lambda g: (rng.random(), g))
    # sum -> tuple of selected groups; retaining the first candidate is enough
    # because groups have a deterministic order.
    states: dict[int, tuple[str, ...]] = {0: ()}
    for group in groups:
        size = int(group_sizes[group])
        additions = [(total + size, selected + (group,))
                     for total, selected in list(states.items())]
        for total, selected in additions:
            if total not in states:
                states[total] = selected
    best_total, best_selected = min(
        states.items(), key=lambda item: (abs(item[0] - target), item[0] > target, item[0]))
    return set(best_selected)


def grouped_train_test_split(n: int, test_fraction: float, seed: int,
                             group_values: Iterable, config: dict | None = None,
                             strat_labels=None, edge_values=None,
                             edge_config: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint sample indices whose groups never cross train/test."""
    config = config or {}
    keys = make_group_keys(group_values, config)
    if len(keys) != n:
        raise ValueError("group_values must have one value per sample")
    group_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(keys):
        group_to_idx[key].append(i)
    protected = _excluded_group_keys(
        group_values, config, edge_values, edge_config,
        np.arange(n, dtype=np.intp), "test")
    target = max(1.0, float(n) * float(test_fraction))
    test_groups = _whole_group_subset(
        {g: len(idx) for g, idx in group_to_idx.items()}, target, protected, seed)
    test = np.array([i for g in sorted(test_groups) for i in group_to_idx[g]], dtype=np.intp)
    train = np.array([i for i in range(n) if i not in set(test.tolist())], dtype=np.intp)
    if len(test) == 0 or len(train) == 0:
        raise ValueError("Grouped split cannot create non-empty train and test partitions")
    return train, test


def grouped_train_val_split(indices: np.ndarray, val_fraction: float, seed: int,
                            group_values: Iterable, config: dict | None = None,
                            strat_labels=None, edge_values=None,
                            edge_config: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Split an existing partition into whole-group train and validation sets."""
    indices = np.asarray(indices, dtype=np.intp)
    values = np.asarray(group_values, dtype=object)
    keys = make_group_keys(values, config)
    local_keys = keys[indices]
    group_to_idx: dict[str, list[int]] = defaultdict(list)
    for idx, key in zip(indices.tolist(), local_keys.tolist()):
        group_to_idx[key].append(idx)
    target = max(1.0, float(len(indices)) * float(val_fraction))
    protected = _excluded_group_keys(
        values, config, edge_values, edge_config, indices, "validation")
    val_groups = _whole_group_subset(
        {g: len(v) for g, v in group_to_idx.items()}, target, protected, seed)
    val = np.array([i for g in sorted(val_groups) for i in group_to_idx[g]], dtype=np.intp)
    train = np.array([i for i in indices.tolist() if i not in set(val.tolist())], dtype=np.intp)
    if len(val) == 0 or len(train) == 0:
        raise ValueError("Grouped validation split cannot create two non-empty partitions")
    return train, val


def grouped_kfold(indices: np.ndarray, n_splits: int, seed: int,
                  group_values: Iterable, config: dict | None = None,
                  strat_labels=None, edge_values=None,
                  edge_config: dict | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic group-balanced K-fold partitions.

    Groups are assigned as indivisible units using a largest-first greedy
    bin-packing strategy.  When labels are supplied, each fold also tracks
    per-label counts to retain material balance as far as group sizes allow.
    """
    indices = np.asarray(indices, dtype=np.intp)
    keys = make_group_keys(np.asarray(group_values, dtype=object), config)
    groups: dict[str, list[int]] = defaultdict(list)
    for idx in indices.tolist():
        groups[keys[idx]].append(idx)
    protected = _excluded_group_keys(
        group_values, config, edge_values, edge_config, indices, "validation")
    eligible_groups = [g for g in groups if g not in protected]
    if len(eligible_groups) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} non-edge groups for grouped K-fold, "
            f"got {len(eligible_groups)} (total groups: {len(groups)})")

    rng = np.random.default_rng(seed)
    ordered = sorted(eligible_groups,
                     key=lambda g: (-len(groups[g]), rng.random(), g))
    fold_groups = [[] for _ in range(n_splits)]
    fold_sizes = np.zeros(n_splits, dtype=int)
    fold_label_sizes = [defaultdict(int) for _ in range(n_splits)]
    labels = np.asarray(strat_labels) if strat_labels is not None else None
    for group in ordered:
        candidates = list(range(n_splits))
        if labels is None:
            fold = min(candidates, key=lambda j: (fold_sizes[j], j))
        else:
            group_labels = labels[groups[group]]
            fold = min(candidates, key=lambda j: (
                sum(fold_label_sizes[j][str(v)] for v in group_labels),
                fold_sizes[j], j))
        fold_groups[fold].append(group)
        fold_sizes[fold] += len(groups[group])
        if labels is not None:
            for value in group_labels:
                fold_label_sizes[fold][str(value)] += 1

    all_set = set(indices.tolist())
    folds = []
    for selected in fold_groups:
        test = np.array([i for g in selected for i in groups[g]], dtype=np.intp)
        train = np.array(sorted(all_set - set(test.tolist())), dtype=np.intp)
        folds.append((train, test))
    # Each fold's train/test pair must be group-disjoint, and test groups must
    # occur in exactly one fold.  Training groups intentionally occur in the
    # other folds, so validating the pairs against one another would be wrong.
    seen_test: set[str] = set()
    for train, test in folds:
        if set(keys[train].tolist()) & set(keys[test].tolist()):
            raise ValueError("A group crosses a grouped K-fold train/test boundary")
        test_keys = set(keys[test].tolist())
        if seen_test & test_keys:
            raise ValueError("A group occurs in more than one grouped K-fold test set")
        seen_test.update(test_keys)
    return folds


def validate_partitions(partitions: list[tuple[np.ndarray, np.ndarray]],
                        group_values: Iterable) -> None:
    """Validate group-disjoint train/test pairs and unique fold test groups."""
    keys = make_group_keys(group_values)
    seen_test: set[str] = set()
    for left, right in partitions:
        left_keys = set(keys[np.asarray(left, dtype=np.intp)].tolist())
        right_keys = set(keys[np.asarray(right, dtype=np.intp)].tolist())
        if left_keys & right_keys:
            raise ValueError("A group crosses a grouped train/test boundary")
        if seen_test & right_keys:
            raise ValueError("A group occurs in more than one grouped test set")
        seen_test.update(right_keys)


def validate_group_disjoint(*partitions: np.ndarray,
                            group_values: Iterable,
                            config: dict | None = None) -> None:
    """Raise when any two supplied partitions share a normalised group."""
    keys = make_group_keys(np.asarray(group_values, dtype=object), config)
    seen: set[str] = set()
    for partition in partitions:
        current = set(keys[np.asarray(partition, dtype=np.intp)].tolist())
        if seen & current:
            raise ValueError("A group occurs in more than one data partition")
        seen.update(current)


def validate_protected_excluded(indices: np.ndarray, group_values: Iterable,
                                config: dict | None = None) -> None:
    """Raise if a configured protected group is present in a test partition."""
    config = config or {}
    protected = {_normalise(v) for v in config.get("protected_group_ids", [])}
    if not protected:
        return
    keys = make_group_keys(np.asarray(group_values, dtype=object), config)
    protected_key = "protected:" + "+".join(sorted(protected))
    if protected_key in set(keys[np.asarray(indices, dtype=np.intp)].tolist()):
        raise ValueError("A protected group is present in the final test partition")


def group_assignments(indices: np.ndarray, group_values: Iterable,
                      config: dict | None = None) -> dict[str, list]:
    """Return serialisable group-to-part-ID assignments for split metadata."""
    indices = np.asarray(indices, dtype=np.intp)
    keys = make_group_keys(np.asarray(group_values, dtype=object), config)
    selected_keys = keys[indices]
    return {key: indices[selected_keys == key].tolist()
            for key in sorted(set(selected_keys.tolist()))}
