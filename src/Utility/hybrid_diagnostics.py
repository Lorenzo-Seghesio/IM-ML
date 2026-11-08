"""Shared prediction plots for neural, tree and PP-only hybrid stages."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .batch_test_diagnostics import save_batch_test_diagnostics
from .cv_diagnostics import _draw_diagnostics, _slug, save_cv_diagnostics


def save_partition_plots(directory, partition, truth, prediction, names, title,
                         groups, ids):
    """Plot observed targets only, retaining physical units and partition labels."""
    directory = Path(directory) / partition
    directory.mkdir(parents=True, exist_ok=True)
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.ndim == 1:
        truth, prediction = truth[:, None], prediction[:, None]
    for i, name in enumerate(names):
        valid = np.isfinite(truth[:, i]) & np.isfinite(prediction[:, i])
        if not valid.any():
            continue
        target_dir = directory if len(names) == 1 else directory / f"{i:03d}_{_slug(name)}"
        target_dir.mkdir(parents=True, exist_ok=True)
        figures = [plt.subplots(figsize=(6, 4.5)) for _ in range(3)]
        _draw_diagnostics([ax for _, ax in figures], truth[valid, i], prediction[valid, i],
                          partition.title(), name)
        suffix = "final_test" if partition == "test" else partition
        for (fig, _), prefix in zip(figures, ("scatter", "residuals_scatter", "residuals_hist")):
            fig.suptitle(title)
            fig.tight_layout()
            fig.savefig(target_dir / f"{prefix}_{suffix}.png")
            plt.close(fig)
        save_batch_test_diagnostics(truth[valid, i], prediction[valid, i],
            np.asarray(groups)[valid], np.asarray(ids)[valid], target_dir,
            target_names=[name], title=f"{title} — {partition}", partition=partition)


def save_cv_plots(directory, records, names, title, cfg):
    """Use the original pipeline's fold plots and a compact metric comparison."""
    if not records or not cfg.get("plots", True):
        return
    directory = Path(directory)
    save_cv_diagnostics(records, directory, title, names,
                        save_per_fold_plots=cfg.get("save_per_fold_plots", len(names) == 1))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for record in records:
        actual, pred = np.asarray(record["y_true"]), np.asarray(record["y_pred"])
        valid = np.isfinite(actual) & np.isfinite(pred)
        errors = (pred - actual)[valid]
        if errors.size:
            ax.scatter(record["fold"] + 1, np.abs(errors).mean(), color="#2166AC")
            ax.scatter(record["fold"] + 1, np.sqrt((errors ** 2).mean()), color="#1B7837")
    ax.plot([], [], "o", color="#2166AC", label="MAE")
    ax.plot([], [], "o", color="#1B7837", label="RMSE")
    ax.set(xlabel="CV fold", ylabel="Error in target units", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "cv_metrics_folds.png")
    plt.close(fig)
