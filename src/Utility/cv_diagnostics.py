"""Plot held-out predictions from indicative cross-validation folds."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


_COLORS = {
    "scatter": "#2166AC",
    "ideal": "#B2182B",
    "residual": "#1B7837",
    "zero": "#555555",
}


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return value or "target"


def _draw_diagnostics(axes, y_true: np.ndarray, y_pred: np.ndarray,
                      fold_label: str, target_label: str) -> None:
    residuals = y_pred - y_true
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")

    ax = axes[0]
    ax.scatter(y_true, y_pred, color=_COLORS["scatter"], s=28, alpha=0.78,
               edgecolors="white", linewidths=0.4, zorder=3)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    if lo == hi:
        pad = max(abs(lo) * 0.05, 1e-6)
        lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], color=_COLORS["ideal"], linestyle="--",
            linewidth=1.2, label="Perfect prediction")
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{fold_label}: measured vs predicted\nMAE={mae:.4g}, R²={r2:.3f}")
    ax.legend(loc="best")

    ax = axes[1]
    ax.scatter(y_pred, residuals, color=_COLORS["residual"], s=28, alpha=0.78,
               edgecolors="white", linewidths=0.4, zorder=3)
    ax.axhline(0, color=_COLORS["zero"], linestyle="--", linewidth=1.2)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (predicted − measured)")
    ax.set_title(f"{fold_label}: residuals vs predicted")

    ax = axes[2]
    ax.hist(residuals, bins=min(20, max(5, int(np.sqrt(len(residuals))))),
            color=_COLORS["residual"], edgecolor="white", alpha=0.85)
    ax.axvline(0, color=_COLORS["zero"], linestyle="--", linewidth=1.2)
    ax.set_xlabel("Residual (predicted − measured)")
    ax.set_ylabel("Count")
    ax.set_title(f"{fold_label}: residual distribution")


def save_cv_diagnostics(fold_records: list[dict], out_dir: Path,
                        title_prefix: str, target_names: list[str] | None = None
                        ) -> list[Path]:
    """Save fold-level and combined diagnostic plots for out-of-fold predictions.

    Each record must contain ``fold``, ``y_true``, and ``y_pred``. Arrays may
    be one-dimensional or ``(samples, targets)``. Multi-output models receive
    one compact all-fold diagnostic figure per target.
    """
    if not fold_records:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    records = sorted(fold_records, key=lambda record: record["fold"])
    normalized = []
    for record in records:
        y_true = np.asarray(record["y_true"], dtype=float)
        y_pred = np.asarray(record["y_pred"], dtype=float)
        if y_true.ndim == 1:
            y_true = y_true[:, None]
        if y_pred.ndim == 1:
            y_pred = y_pred[:, None]
        if y_true.shape != y_pred.shape:
            raise ValueError("CV diagnostic targets and predictions must have matching shapes")
        normalized.append((record["fold"], y_true, y_pred))

    n_targets = normalized[0][1].shape[1]
    if any(y_true.shape[1] != n_targets for _, y_true, _ in normalized):
        raise ValueError("All CV folds must have the same number of targets")
    labels = target_names or [f"target_{i}" for i in range(n_targets)]
    if len(labels) != n_targets:
        raise ValueError("target_names length must match the prediction target count")

    saved_paths = []
    for target_idx, target_label in enumerate(labels):
        n_folds = len(normalized)
        fig, axes = plt.subplots(n_folds, 3,
                                 figsize=(15, max(4.2, 3.7 * n_folds)),
                                 squeeze=False)
        for row, (fold, y_true, y_pred) in enumerate(normalized):
            _draw_diagnostics(axes[row], y_true[:, target_idx],
                              y_pred[:, target_idx], f"Fold {fold + 1}",
                              str(target_label))
        fig.suptitle(f"{title_prefix} — {target_label}", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        if n_targets == 1:
            filename = "cv_diagnostics_folds.png"
        else:
            filename = f"cv_diagnostics_{_slug(target_label)}.png"
        path = out_dir / filename
        fig.savefig(path)
        plt.close(fig)
        saved_paths.append(path)

        for fold, y_true, y_pred in normalized:
            fold_fig, fold_axes = plt.subplots(1, 3, figsize=(15, 4.5))
            _draw_diagnostics(fold_axes, y_true[:, target_idx],
                              y_pred[:, target_idx], f"Fold {fold + 1}",
                              str(target_label))
            fold_fig.suptitle(
                f"{title_prefix} — {target_label} — Fold {fold + 1}", fontsize=13)
            fold_fig.tight_layout(rect=(0, 0, 1, 0.94))
            suffix = "" if n_targets == 1 else f"_{_slug(target_label)}"
            fold_path = out_dir / f"cv_fold_{fold + 1:02d}_diagnostics{suffix}.png"
            fold_fig.savefig(fold_path)
            plt.close(fold_fig)
            saved_paths.append(fold_path)

    print(f"CV diagnostic plots saved → {out_dir}")
    return saved_paths
