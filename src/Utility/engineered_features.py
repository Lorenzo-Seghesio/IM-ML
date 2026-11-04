"""Physical curve descriptors. No fitted dataset statistics or target access.

Times are seconds, amplitudes retain their input units, integrals are amplitude*s,
and slopes are amplitude/s. The first occurrence of the maximum is the peak.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ENGINEERED_FEATURES = [
    "max", "min", "mean", "std", "range", "start", "end", "duration_s",
    "peak_time_s", "area", "area_above_baseline", "time_weighted_mean",
    "max_rise_slope", "max_fall_slope", "rise_time_10_90_s",
    "fall_time_90_10_s", "rise_slope_10_90", "fall_slope_90_10",
]
VERSION = 2


def signal_time_options(source: dict, options: dict, base_dir=None) -> dict:
    """Resolve explicit timing first, then sibling info/signal_time_grid.csv.

    Relative paths are interpreted against base_dir when supplied. Discovery is
    local to each signal's dataset directory, so unrelated grids are never used.
    """
    name = source.get("name", "")
    config = {**options, **options.get("signals", {}).get(name, {}), **source.get("time", {})}
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    for key in ("timestamps_csv", "time_grid_csv"):
        if config.get(key):
            config[key] = str(root / config[key])
    explicit = (config.get("timestamps_s") is not None or config.get("timestamps_csv")
                or config.get("column_times_seconds", False)
                or config.get("sampling_frequency_hz") is not None
                or config.get("sampling_interval_s") is not None)
    if not explicit:
        if config.get("time_grid_csv"):
            config["timestamps_csv"] = config["time_grid_csv"]
        elif config.get("auto_discover_time_grid", True) and source.get("path"):
            candidate = (root / source["path"]).parent / "info" / "signal_time_grid.csv"
            if candidate.is_file():
                config["timestamps_csv"] = str(candidate)
    return config


def time_axis(n: int, options: dict, columns=None) -> np.ndarray:
    """Require explicit seconds, timestamp columns, or a known sample frequency."""
    if options.get("timestamps_s") is not None:
        t = np.asarray(options["timestamps_s"], dtype=float)
    elif options.get("timestamps_csv"):
        frame = pd.read_csv(options["timestamps_csv"])
        time_column = options.get("timestamp_column")
        if time_column is None:
            time_column = next((name for name in ("time_seconds", "time_s") if name in frame), None)
        if time_column not in frame:
            raise ValueError("Time-grid CSV must contain time_seconds/time_s or the configured timestamp_column")
        label_column = options.get("timestamp_label_column", "column")
        if label_column in frame:
            labels = frame[label_column]
            if labels.isna().any() or labels.astype(str).duplicated().any():
                raise ValueError("Time-grid column labels must be non-missing and unique")
            if columns is not None:
                expected = list(map(str, columns))
                actual = labels.astype(str).tolist()
                if len(expected) != n or len(set(expected)) != n or set(actual) != set(expected):
                    raise ValueError("Time-grid column labels must exactly match the signal CSV columns")
                frame = frame.assign(**{label_column: actual}).set_index(label_column).loc[expected]
        t = frame[time_column].to_numpy(float)
    elif options.get("column_times_seconds", False):
        t = np.asarray(columns, dtype=float)
    elif options.get("sampling_frequency_hz") is not None:
        frequency = float(options["sampling_frequency_hz"])
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError("sampling_frequency_hz must be positive and finite")
        t = np.arange(n, dtype=float) / frequency + float(options.get("start_time_s", 0))
    elif options.get("sampling_interval_s") is not None:
        interval = float(options["sampling_interval_s"])
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError("sampling_interval_s must be positive and finite")
        t = np.arange(n, dtype=float) * interval + float(options.get("start_time_s", 0))
    else:
        raise ValueError("Physical time metadata required: configure timestamps_s, timestamps_csv, "
                         "sampling_interval_s, or sampling_frequency_hz for each signal. "
                         "Alternatively provide info/signal_time_grid.csv beside the signal CSV. "
                         "Sample labels t0, t1, ... are not seconds.")
    if t.shape != (n,) or n < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Timestamps must be finite, strictly increasing, and match the curve width")
    return t


def _crossings(t, y, level, rising):
    mask = ((y[:-1] <= level) & (y[1:] > level) if rising else
            (y[:-1] >= level) & (y[1:] < level))
    indices = np.flatnonzero(mask)
    values = t[indices] + (level - y[indices]) * np.diff(t)[indices] / np.diff(y)[indices]
    # A threshold exactly at the final endpoint is a valid crossing too.
    if len(y) >= 2 and y[-1] == level and (y[-2] < level if rising else y[-2] > level):
        values = np.append(values, t[-1])
    return values


def curve_features(values, timestamps, options=None) -> dict:
    options = options or {}
    y, t = np.asarray(values, float), np.asarray(timestamps, float)
    if y.ndim != 1 or y.shape != t.shape or len(y) < 2:
        raise ValueError("A curve needs at least two samples and matching timestamps")
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing")
    valid = np.isfinite(y)
    policy = options.get("invalid_values", "interpolate")
    if policy not in ("raise", "interpolate", "nan"):
        raise ValueError("invalid_values must be raise, interpolate, or nan")
    if not valid.all():
        if policy == "raise":
            raise ValueError("Non-finite curve amplitude")
        if policy == "nan" or valid.sum() < 2:
            return dict.fromkeys(ENGINEERED_FEATURES, float("nan"))
        y = np.interp(t, t[valid], y[valid])
    baseline_mode = options.get("baseline", "start")
    if isinstance(baseline_mode, (float, int)):
        baseline = float(baseline_mode)
    elif baseline_mode == "start":
        baseline = y[0]
    elif baseline_mode == "min":
        baseline = y.min()
    elif baseline_mode == "initial_mean":
        count = int(options.get("baseline_samples", 5))
        if count < 1:
            raise ValueError("baseline_samples must be positive")
        baseline = y[:count].mean()
    else:
        raise ValueError("baseline must be start, min, initial_mean, or a number")
    if not np.isfinite(baseline):
        raise ValueError("Baseline must be finite")
    peak = int(np.argmax(y))
    amplitude = y[peak] - baseline
    duration = t[-1] - t[0]
    area = float(np.trapezoid(y, t))
    derivative = np.diff(y) / np.diff(t)  # adjacent secant slopes on the actual time grid
    # Integrate the piecewise-linear positive part exactly, inserting baseline crossings.
    centered = y - baseline
    crossing = np.flatnonzero(centered[:-1] * centered[1:] < 0)
    tc = t[crossing] - centered[crossing] * np.diff(t)[crossing] / np.diff(centered)[crossing]
    augmented = np.sort(np.concatenate([t, tc]))
    above = float(np.trapezoid(np.maximum(np.interp(augmented, t, centered), 0), augmented))
    result = dict(zip(ENGINEERED_FEATURES[:14], [
        y.max(), y.min(), y.mean(), y.std(ddof=0), np.ptp(y), y[0], y[-1],
        duration, t[peak], area, above, area / duration,
        max(0., derivative.max()), min(0., derivative.min()),
    ]))
    rise_time = fall_time = rise_slope = fall_slope = float("nan")
    if amplitude > float(options.get("amplitude_epsilon", 1e-12)):
        low, high = baseline + .1 * amplitude, baseline + .9 * amplitude
        # Last rising crossings before the first peak; first falling crossings after it.
        r90 = _crossings(t[:peak+1], y[:peak+1], high, True)
        r10 = _crossings(t[:peak+1], y[:peak+1], low, True)
        f90 = _crossings(t[peak:], y[peak:], high, False)
        f10 = _crossings(t[peak:], y[peak:], low, False)
        if len(r90) and len(r10):
            earlier = r10[r10 < r90[-1]]
            if len(earlier):
                rise_time = r90[-1] - earlier[-1]
                rise_slope = .8 * amplitude / rise_time
        if len(f90) and len(f10):
            later = f10[f10 > f90[0]]
            if len(later):
                fall_time = later[0] - f90[0]
                fall_slope = -.8 * amplitude / fall_time
    elif np.ptp(y) <= float(options.get("amplitude_epsilon", 1e-12)):
        rise_time = fall_time = rise_slope = fall_slope = 0.
    result.update(zip(ENGINEERED_FEATURES[14:], [rise_time, fall_time, rise_slope, fall_slope]))
    return {key: float(value) for key, value in result.items()}


def extract_features(curves, sources, options, part_ids, cache_dir=None, columns=None,
                     base_dir=None, return_time_grids=False):
    """Extract/cache in signal-major order. Cache includes values, IDs, times and options."""
    if len(curves) != len(sources):
        raise ValueError("One source is required per curve matrix")
    chosen = options.get("features", ENGINEERED_FEATURES)
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(ENGINEERED_FEATURES):
        raise ValueError("features must be a non-empty unique subset of ENGINEERED_FEATURES")
    names, grids, settings, time_grids = [], [], [], {}
    digest = hashlib.sha256(json.dumps([VERSION, options, list(map(str, part_ids))], sort_keys=True).encode())
    for i, (matrix, source) in enumerate(zip(curves, sources)):
        name = source.get("name", f"signal_{i+1}")
        config = signal_time_options(source, options, base_dir)
        grid = time_axis(matrix.shape[1], config, columns[i] if columns else None)
        grids.append(grid)
        settings.append(config)
        names.extend(f"{name}__{feature}" for feature in chosen)
        digest.update(name.encode())
        digest.update(json.dumps(config, sort_keys=True).encode())
        digest.update(np.ascontiguousarray(matrix, dtype=np.float64).tobytes())
        digest.update(grid.tobytes())
        time_grids[name] = {"timestamps_s": grid.tolist(), "n_samples": len(grid),
                            "timestamps_csv": config.get("timestamps_csv")}
    if len(set(names)) != len(names):
        raise ValueError("Signal names must be distinct")
    path = Path(cache_dir) / (digest.hexdigest() + ".npz") if cache_dir else None
    if path and path.exists():
        with np.load(path, allow_pickle=False) as data:
            result = (data["values"], data["names"].tolist())
            return (*result, time_grids) if return_time_grids else result
    blocks = []
    for matrix, grid, config in zip(curves, grids, settings):
        rows = [curve_features(row, grid, config) for row in matrix]
        blocks.append(np.asarray([[row[key] for key in chosen] for row in rows]))
    values = np.concatenate(blocks, axis=1)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, values=values, names=np.asarray(names))
    return (values, names, time_grids) if return_time_grids else (values, names)
