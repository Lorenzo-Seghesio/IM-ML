"""Held-out test metrics and plots grouped by configured batch."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _as_2d(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array; got shape {array.shape}.")
    return array


def _display_batch_id(value: Any) -> str:
    try:
        if pd.isna(value):
            return "(missing)"
    except (TypeError, ValueError):
        pass
    return str(value)


def save_batch_test_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    batch_ids: Sequence[Any] | np.ndarray | None,
    sample_ids: Sequence[Any] | np.ndarray,
    out_dir: Path,
    target_names: Sequence[str] | None = None,
    title: str = "Final held-out test",
    partition: str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Save per-batch test metrics, per-sample predictions, and one plot.

    Prediction CSV is long-form (one row per sample and target) and includes
    residual = predicted - measured. The metric CSV has one row per batch and
    target. If batch IDs are unavailable, no artifacts are produced.
    """
    if batch_ids is None:
        print("  [batch diagnostics] skipped: no configured group/batch column")
        return None

    truth = _as_2d(y_true, "y_true")
    pred = _as_2d(y_pred, "y_pred")
    groups = np.asarray(batch_ids, dtype=object)
    ids = np.asarray(sample_ids, dtype=object)
    if truth.shape != pred.shape:
        raise ValueError(
            f"y_true and y_pred must have matching shapes; got {truth.shape} "
            f"and {pred.shape}."
        )
    if groups.ndim != 1 or ids.ndim != 1:
        raise ValueError("batch_ids and sample_ids must be one-dimensional.")
    if len(groups) != len(truth) or len(ids) != len(truth):
        raise ValueError(
            "Held-out targets, predictions, batch IDs, and sample IDs must "
            "have matching sample counts."
        )
    if not len(truth):
        raise ValueError("Cannot report batch metrics for an empty test set.")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("Held-out targets and predictions must be finite.")

    n_targets = truth.shape[1]
    labels = (list(target_names) if target_names is not None
              else ["target" if n_targets == 1 else f"target_{i}"
                    for i in range(n_targets)])
    if len(labels) != n_targets:
        raise ValueError(
            f"target_names has {len(labels)} entries for {n_targets} targets."
        )
    labels = [str(label) for label in labels]

    group_labels = np.asarray([_display_batch_id(value) for value in groups], dtype=object)
    unique_groups = list(pd.unique(group_labels))
    if not unique_groups:
        print("  [batch diagnostics] skipped: test set has no batch IDs")
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for target_idx, target_name in enumerate(labels):
        residuals = pred[:, target_idx] - truth[:, target_idx]
        for sample_idx in range(len(truth)):
            prediction_rows.append({
                "sample_id": ids[sample_idx],
                "batch_id": group_labels[sample_idx],
                "target": target_name,
                "y_true": float(truth[sample_idx, target_idx]),
                "y_pred": float(pred[sample_idx, target_idx]),
                "residual": float(residuals[sample_idx]),
            })
        for batch_id in unique_groups:
            mask = group_labels == batch_id
            batch_true = truth[mask, target_idx]
            batch_pred = pred[mask, target_idx]
            mse = float(mean_squared_error(batch_true, batch_pred))
            metric_rows.append({
                "batch_id": batch_id,
                "target": target_name,
                "n_samples": int(mask.sum()),
                "MAE": float(mean_absolute_error(batch_true, batch_pred)),
                "MSE": mse,
                "RMSE": float(np.sqrt(mse)),
                "R2": (float(r2_score(batch_true, batch_pred))
                       if int(mask.sum()) >= 2 else float("nan")),
            })

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    metrics_path = out_dir / f"{partition}_metrics_per_batch.csv"
    predictions_path = out_dir / f"{partition}_predictions_per_batch.csv"
    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    n_groups = len(unique_groups)
    color_map = plt.get_cmap("turbo", max(n_groups, 1))
    colors = {batch_id: color_map(index) for index, batch_id in enumerate(unique_groups)}
    fig, axes = plt.subplots(
        n_targets, 2,
        figsize=(13, max(5, 4.2 * n_targets)),
        squeeze=False,
    )
    for target_idx, target_name in enumerate(labels):
        ax_scatter, ax_residual = axes[target_idx]
        y_t = truth[:, target_idx]
        y_p = pred[:, target_idx]
        residuals = y_p - y_t
        for batch_id in unique_groups:
            mask = group_labels == batch_id
            ax_scatter.scatter(
                y_t[mask], y_p[mask], s=34, alpha=0.82,
                color=colors[batch_id], edgecolors="white", linewidths=0.35,
                label=batch_id,
            )
            ax_residual.scatter(
                y_p[mask], residuals[mask], s=34, alpha=0.82,
                color=colors[batch_id], edgecolors="white", linewidths=0.35,
            )

        lo = float(min(y_t.min(), y_p.min()))
        hi = float(max(y_t.max(), y_p.max()))
        if lo == hi:
            padding = max(abs(lo) * 0.05, 1e-6)
            lo, hi = lo - padding, hi + padding
        ax_scatter.plot([lo, hi], [lo, hi], "--", color="#B2182B", linewidth=1.2)
        ax_scatter.set_xlim(lo, hi)
        ax_scatter.set_ylim(lo, hi)
        ax_scatter.set_xlabel("Measured")
        ax_scatter.set_ylabel("Predicted")
        ax_scatter.set_title(f"{target_name}: measured vs predicted")

        ax_residual.axhline(0, color="#555555", linestyle="--", linewidth=1.2)
        ax_residual.set_xlabel("Predicted")
        ax_residual.set_ylabel("Residual (predicted − measured)")
        ax_residual.set_title(f"{target_name}: residuals vs predicted")

    fig.suptitle(f"{title} — colored by batch", fontsize=14)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, title="Batch", loc="lower center",
        bbox_to_anchor=(0.5, 0.005), ncol=min(5, n_groups),
        fontsize=8, title_fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    plot_path = out_dir / f"{partition}_scatter_residuals_by_batch.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(f"  Batch test metrics saved → {metrics_path}")
    print(f"  Batch test predictions saved → {predictions_path}")
    print(f"  Batch scatter/residual plot saved → {plot_path}")
    return metrics_df, predictions_df
