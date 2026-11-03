"""Compare complete validation and test MAEs within a single beta sweep."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_METRICS = (
    ("MAE_w", "Part weight (MAE_w)"),
    ("MAE_TPavg", "Test-point average (MAE_TPavg)"),
    ("MAE_within", "Within test point (MAE_within)"),
)


def save_beta_mae_comparison(selection: dict, out_dir: Path,
                             title: str) -> list[Path]:
    """Save a three-panel PNG and its raw-gram values as CSV.

    Read metrics from each experiment's run directory, never best-overall
    directories, which may belong to other runs. M1 stores test metrics in
    ``best_metrics.json``; references also write ``test_metrics.json``.
    The selection record supplies the validation MAE if detailed metrics are
    unavailable. Missing metrics stay blank in the CSV and are not plotted.
    """
    rows = []
    for experiment in sorted(selection["experiments"],
                             key=lambda item: item["within_test_point_beta"]):
        beta = experiment["within_test_point_beta"]
        row = {"beta": beta, "selected": beta == selection["selected_beta"]}
        run_dir = Path(experiment["run_dir"])
        for split in ("validation", "test"):
            path = run_dir / f"{split}_metrics.json"
            if split == "test" and not path.exists():
                path = run_dir / "best_metrics.json"
            metrics = json.loads(path.read_text()) if path.exists() else {}
            for metric, _ in _METRICS:
                value = metrics.get(metric)
                if metric == "MAE_w" and value is None:
                    value = metrics.get("MAE")
                    if split == "validation" and value is None:
                        value = experiment["validation_mae_g"]
                row[f"{split}_{metric}_g"] = value
        rows.append(row)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "beta_mae_comparison.csv"
    columns = ["beta", "selected"] + [
        f"{split}_{metric}_g" for split in ("validation", "test")
        for metric, _ in _METRICS]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    try:
        betas = [row["beta"] for row in rows]
        for ax, (metric, label) in zip(axes, _METRICS):
            plotted = False
            for split, color, marker, style in (
                    ("validation", "#2166AC", "o", "-"),
                    ("test", "#B2182B", "s", "--")):
                values = [row[f"{split}_{metric}_g"] for row in rows]
                values = np.array([np.nan if value is None else value
                                   for value in values], dtype=float)
                if np.isfinite(values).any():
                    ax.plot(betas, values, color=color, marker=marker,
                            linestyle=style, label=split.capitalize())
                    plotted = True
            ax.axvline(selection["selected_beta"], color="#555555",
                       linestyle=":", linewidth=1.2, label="Selected beta")
            ax.set(title=label, xlabel="Beta (β)", ylabel="MAE (g)")
            ax.set_xticks(betas, [f"{beta:g}" for beta in betas])
            ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
            ax.grid(True, linestyle="--", alpha=0.35)
            if plotted:
                ax.legend()
            else:
                ax.text(0.5, 0.5, "Metrics unavailable", ha="center",
                        va="center", transform=ax.transAxes)
        fig.suptitle(
            f"{title}\nSelected β={selection['selected_beta']:g} by validation part-weight MAE")
        fig.tight_layout()
        plot_path = out_dir / "beta_mae_comparison.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)
    return [plot_path, csv_path]
