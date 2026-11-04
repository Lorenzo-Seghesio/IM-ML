"""Shared data, split, preprocessing, and artifact contracts for hybrid models."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .group_splitting import (
    grouped_train_test_split, grouped_train_val_split, validate_group_disjoint,
    validate_protected_excluded, validate_edge_groups_excluded,
    edge_group_policy_metadata, grouped_kfold,
)

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=lambda v: v.item() if
                          isinstance(v, np.generic) else str(v)))


def resolve(path):
    return ROOT / path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(*items):
    return hashlib.sha256(json.dumps(items, sort_keys=True, default=str).encode()).hexdigest()


def fit_imputer(values):
    values = np.asarray(values, float)
    # keep_empty_features preserves the saved schema for entirely unavailable descriptors.
    return SimpleImputer(strategy="median", keep_empty_features=True).fit(
        np.where(np.isfinite(values), values, np.nan))


def impute(imputer, values):
    values = np.asarray(values, float)
    return imputer.transform(np.where(np.isfinite(values), values, np.nan)).astype(np.float32)


@dataclass
class HybridData:
    ids: np.ndarray
    pp: np.ndarray
    curves: list
    y: np.ndarray
    groups: np.ndarray
    edges: np.ndarray | None
    pp_names: list
    sources: list
    curve_columns: list
    config: dict
    digest: str


def load_data(cfg):
    data = cfg["data"]
    id_col = data["id_col"]
    scalar_path = resolve(data["scalar_csv"])
    scalar = pd.read_csv(scalar_path, index_col=id_col)
    if scalar.index.has_duplicates:
        raise ValueError("Scalar part IDs must be unique")
    sources = data.get("timeseries_csvs", data.get("pressure_csvs"))
    if sources is None:
        sources = [{"name": "pressure", "path": data["pressure_csv"]}]
    frames = [pd.read_csv(resolve(source["path"]), index_col=id_col) for source in sources]
    if any(frame.index.has_duplicates for frame in frames):
        raise ValueError("Signal part IDs must be unique")
    ids = scalar.index
    for frame in frames:
        ids = ids[ids.isin(frame.index)]
    scalar = scalar.loc[ids]
    group_col = data.get("group_col")
    if not group_col or group_col not in scalar or not data.get("group_split", {}).get("enabled"):
        raise ValueError("Hybrids require data.group_col and group_split.enabled=true")
    groups = scalar[group_col].to_numpy()
    if pd.isna(groups).any():
        raise ValueError("Test-point IDs cannot be missing")
    edge_cols = data.get("edge_group_split", {}).get("columns", [])
    edges = scalar[edge_cols].to_numpy(float) if edge_cols else None
    excluded = {*data.get("drop_cols", []), *edge_cols, group_col, data["target_col"]}
    pp_names = [name for name in scalar if name not in excluded]
    pp = scalar[pp_names].to_numpy(np.float32)
    y = scalar[data["target_col"]].to_numpy(np.float32)
    if not np.isfinite(y).all() or not len(ids):
        raise ValueError("Weight targets must be finite and the joined dataset non-empty")
    digest = fingerprint(data, [(str(path), file_hash(path)) for path in
        [scalar_path, *[resolve(s["path"]) for s in sources]]])
    return HybridData(ids.to_numpy(), pp, [frame.loc[ids].to_numpy(np.float32) for frame in frames],
                      y, groups, edges, pp_names, sources, [list(frame) for frame in frames], cfg, digest)


def split_data(dataset, cfg, source_dir=None):
    data, prep = cfg["data"], cfg["preprocessing"]
    group_cfg, edge_cfg = data["group_split"], data.get("edge_group_split", {})
    source = source_dir or cfg.get("split", {}).get("source_dir")
    if source:
        contract = read_json(resolve(source) / "data_processing.json")
        positions = {value: i for i, value in enumerate(dataset.ids.tolist())}
        result = []
        for key in ("train_part_ids", "val_part_ids", "test_part_ids"):
            missing = set(contract[key]) - positions.keys()
            if missing:
                raise ValueError(f"Split {key} contains {len(missing)} parts absent from input data")
            result.append(np.asarray([positions[value] for value in contract[key]], dtype=np.intp))
        train, val, test = result
        if contract.get("pp_cols") and contract["pp_cols"] != dataset.pp_names:
            raise ValueError("PP feature names/order differ from the source split/checkpoint")
        if contract.get("group_split", {}) != group_cfg:
            raise ValueError("Configured group policy differs from the authoritative split")
    else:
        seed = prep.get("split_seed", prep["random_seed"])
        dev, test = grouped_train_test_split(len(dataset.y),
            prep.get("test_fraction", .15), seed, dataset.groups, group_cfg,
            edge_values=dataset.edges, edge_config=edge_cfg)
        if prep.get("remove_target_outliers", False):
            from src.M1_PPPT_to_W import _dev_y_outlier_keep_mask
            keep, _ = _dev_y_outlier_keep_mask(dataset.y, dataset.ids, dev, prep)
            dev = dev[keep[dev]]
        train, val = grouped_train_val_split(dev, cfg["training"].get("val_fraction", .15),
            seed, dataset.groups, group_cfg, edge_values=dataset.edges, edge_config=edge_cfg)
    if not all(len(part) for part in (train, val, test)):
        raise ValueError("Train, validation and independent test partitions must all be non-empty")
    if any(len(np.unique(part)) != len(part) for part in (train, val, test)):
        raise ValueError("Duplicate part IDs in split")
    validate_group_disjoint(train, val, test, group_values=dataset.groups, config=group_cfg)
    validate_protected_excluded(val, dataset.groups, group_cfg, partition="validation")
    validate_protected_excluded(test, dataset.groups, group_cfg, partition="test")
    scope = np.concatenate([train, val, test])
    if dataset.edges is not None:
        validate_edge_groups_excluded(test, dataset.edges, dataset.groups, edge_cfg,
                                     group_cfg, scope_indices=scope, partition="test")
        validate_edge_groups_excluded(val, dataset.edges, dataset.groups, edge_cfg,
                                     group_cfg, scope_indices=np.concatenate([train, val]),
                                     partition="validation")
    contract = {"id_col": data["id_col"], "pp_cols": dataset.pp_names,
                "group_col": data["group_col"], "group_split": group_cfg,
                "edge_group_split": edge_group_policy_metadata(dataset.edges, dataset.groups,
                                                                edge_cfg, group_cfg),
                **{key: dataset.ids[index].tolist() for key, index in zip(
                    ("train_part_ids", "val_part_ids", "test_part_ids"), (train, val, test))}}
    return (train, val, test), contract


def cv_splits(dataset, indices, n_folds, seed):
    data = dataset.config["data"]
    return list(grouped_kfold(np.asarray(indices), n_folds, seed, dataset.groups,
        data["group_split"], edge_values=dataset.edges, edge_config=data.get("edge_group_split", {})))


def inner_split(dataset, indices, cfg, seed):
    data = dataset.config["data"]
    return grouped_train_val_split(np.asarray(indices), cfg.get("val_fraction", .15), seed,
        dataset.groups, data["group_split"], edge_values=dataset.edges,
        edge_config=data.get("edge_group_split", {}))


def checkpoint_valid(directory, signature, files):
    directory = Path(directory)
    manifest = directory / "manifest.json"
    if not manifest.exists():
        return False
    saved = read_json(manifest)
    return (saved.get("signature") == signature and all((directory / name).is_file()
            and saved.get("files", {}).get(name) == file_hash(directory / name) for name in files))


def save_manifest(directory, signature, files, **metadata):
    write_json(Path(directory) / "manifest.json", {"signature": signature,
        "files": {name: file_hash(Path(directory) / name) for name in files}, **metadata})


def dump_preprocessing(directory, value):
    joblib.dump(value, Path(directory) / "preprocessing.joblib")
