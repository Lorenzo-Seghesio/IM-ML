"""Report the selected Fusion M2's ability to reconstruct frozen M1 features."""
from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .batch_test_diagnostics import save_batch_test_diagnostics
from .cv_diagnostics import _draw_diagnostics


def save_fusion_reconstruction(truth, prediction, feature_names, sample_ids,
                               batch_ids, out_dir, material, partition="test",
                               target_scale=None, plots=True):
    """Evaluate already-computed selected-model features, without fitting anything.

    Raw errors retain M1 feature units. NMAE uses M2's saved training scaler,
    never the range of the evaluated partition.
    """
    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    ids = np.asarray(sample_ids)
    if (truth.ndim != 2 or truth.shape != prediction.shape or not truth.size
            or truth.shape[1] != len(feature_names) or len(ids) != len(truth)):
        raise ValueError("Reconstruction targets, predictions, feature names and IDs must align")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Reconstruction targets and predictions must be finite")
    scale = np.asarray(target_scale, dtype=float) if target_scale is not None else None
    if scale is not None and (scale.shape != (truth.shape[1],) or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("target_scale must contain one positive fitted scale per feature")
    if partition not in ("test", "test_1", "test_2", "validation"):
        raise ValueError("Reconstruction partition must be test or validation")
    directory = Path(out_dir) / "m2_reconstruction"
    directory.mkdir(parents=True, exist_ok=True)
    errors = prediction - truth
    per_target = {}
    for i, name in enumerate(feature_names):
        mse = float(np.mean(errors[:, i] ** 2))
        per_target[name] = {"MAE": float(np.abs(errors[:, i]).mean()), "MSE": mse,
                            "RMSE": math.sqrt(mse),
                            "R2": float(r2_score(truth[:, i], prediction[:, i])) if len(truth) > 1 else None}
        if scale is not None:
            per_target[name]["NMAE"] = float(np.abs(errors[:, i] * scale[i]).mean())
    report = {f"mean_{key}": float(np.mean([values[key] for values in per_target.values()]))
              for key in ("MAE", "MSE", "RMSE")}
    report["mean_R2"] = float(np.mean([v["R2"] for v in per_target.values()])) if len(truth) > 1 else None
    if scale is not None:
        report["mean_NMAE"] = float(np.mean([v["NMAE"] for v in per_target.values()]))
    report.update(per_target=per_target, source=f"final_{partition}" if partition.startswith("test") else "validation",
                  partition=partition, role="Selected Fusion M2 reconstruction", n_samples=len(truth),
                  n_features=len(feature_names), target_reference="frozen M1 extracted signal features",
                  normalization="saved M2 training scaler" if scale is not None else None)
    (directory / f"{partition}_metrics.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame([{"feature": name, **metrics} for name, metrics in per_target.items()]).to_csv(
        directory / f"{partition}_feature_metrics.csv", index=False)
    frame = pd.DataFrame({"ID_Part": ids})
    for i, name in enumerate(feature_names):
        frame[f"{name}_true"], frame[f"{name}_pred"] = truth[:, i], prediction[:, i]
    frame.to_csv(directory / f"{partition}_feature_predictions.csv", index=False)
    if plots:
        ncols = min(len(feature_names), 3)
        nrows = math.ceil(len(feature_names) / ncols)
        figures = [plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False) for _ in range(3)]
        for i, name in enumerate(feature_names):
            _draw_diagnostics([axes.flat[i] for _, axes in figures], truth[:, i], prediction[:, i],
                              partition.title(), name)
        suffix = f"final_{partition}" if partition.startswith("test") else "validation"
        for (fig, axes), prefix in zip(figures, ("scatter", "residuals_scatter", "residuals_hist")):
            for i in range(len(feature_names), nrows*ncols):
                axes.flat[i].set_visible(False)
            fig.suptitle(f"[{material}] Selected Fusion M2 — {partition} reconstruction\n"
                         f"mean feature MAE={report['mean_MAE']:.6g}")
            fig.tight_layout()
            fig.savefig(directory / f"{prefix}_{suffix}.png")
            plt.close(fig)
        save_batch_test_diagnostics(truth, prediction, batch_ids, ids, directory,
            target_names=feature_names, title=f"[{material}] Selected Fusion M2 — {partition} reconstruction",
            partition=partition)
    print(f"  Selected Fusion M2 {partition}: mean_MAE={report['mean_MAE']:.6f}; "
          f"mean_NMAE={report.get('mean_NMAE')}; outputs → {directory}", flush=True)
    return report
