"""Batch-balanced SHAP sampling across full Train, internal test and external Test."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def batch_balanced_indices(groups, requested, seed):
    groups = np.asarray(groups)
    if groups.ndim != 1 or pd.isna(groups).any():
        raise ValueError("SHAP batch labels must be a one-dimensional non-missing array")
    if not len(groups) or int(requested) <= 0:
        raise ValueError("SHAP requires non-empty data and a positive sample count")
    rng = np.random.default_rng(seed)
    batches = pd.unique(groups)
    target = min(len(groups), max(int(requested), len(batches)))
    pools = [list(rng.permutation(np.flatnonzero(groups == batch))) for batch in batches]
    selected = []
    while len(selected) < target:
        for pool in pools:
            if pool:
                selected.append(pool.pop())
                if len(selected) == target:
                    break
    return np.asarray(selected, dtype=np.intp)


def scoped_fusion_xai(callback, background, background_groups, domains, directory, cfg, seed):
    """Callback receives a sampled fixed background and one explanation domain."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    bg_idx = batch_balanced_indices(background_groups, cfg.get("background_samples", cfg.get("shap_bg", 100)), seed)
    manifest = []
    for partition, pp, ids, groups in domains:
        selected = batch_balanced_indices(groups, cfg.get("explain_samples", 64), seed)
        out = directory / partition
        out.mkdir(parents=True, exist_ok=True)
        ids, groups = np.asarray(ids), np.asarray(groups)
        pd.DataFrame({"part_id": ids[selected], "batch_id": groups[selected], "partition": partition}).to_csv(
            out / "explained_samples.csv", index=False)
        callback(background[bg_idx], pp[selected], ids[selected], out, partition)
        record = {"partition": partition, "n_available": len(pp), "n_explained": len(selected),
                  "n_batches_available": len(pd.unique(groups)),
                  "n_batches_explained": len(pd.unique(groups[selected])),
                  "all_batches_represented": set(groups.tolist()) == set(groups[selected].tolist()),
                  "background_samples": len(bg_idx), "seed": seed,
                  "id_namespace": partition, "selection_uses_targets": False}
        (out / "sampling.json").write_text(json.dumps(record, indent=2))
        manifest.append(record)
    (directory / "partitions.json").write_text(json.dumps(manifest, indent=2))
