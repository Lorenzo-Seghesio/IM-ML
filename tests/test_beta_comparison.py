import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.Utility.beta_comparison import save_beta_mae_comparison


class BetaComparisonTests(unittest.TestCase):
    def test_plot_and_csv_use_each_runs_metrics_in_numeric_beta_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiments = []
            for beta in (1., 0., .25):
                run_dir = root / f"beta_{beta}"
                run_dir.mkdir()
                for split, offset in (("validation", 1.), ("test", 10.)):
                    metrics = {"MAE_w": offset + beta,
                               "MAE_TPavg": offset + beta + .1,
                               "MAE_within": offset + beta + .2}
                    (run_dir / f"{split}_metrics.json").write_text(json.dumps(metrics))
                # Reference test_metrics.json takes precedence over this file.
                (run_dir / "best_metrics.json").write_text('{"MAE": 999}')
                experiments.append({"within_test_point_beta": beta,
                                    "validation_mae_g": 1. + beta,
                                    "run_dir": str(run_dir)})
            selection = {"selected_beta": 0., "experiments": experiments}
            with patch("src.Utility.beta_comparison.plt.close", wraps=plt.close) as close:
                paths = save_beta_mae_comparison(selection, root / "summary", "Example run")
            self.assertTrue(all(path.is_file() for path in paths))
            with paths[1].open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([float(row["beta"]) for row in rows], [0., .25, 1.])
            self.assertEqual([row["selected"] for row in rows], ["True", "False", "False"])
            self.assertEqual(float(rows[0]["validation_MAE_w_g"]), 1.)
            self.assertEqual(float(rows[0]["test_MAE_w_g"]), 10.)
            self.assertEqual(float(rows[0]["test_MAE_within_g"]), 10.2)
            figure = close.call_args.args[0]
            for axis, metric in zip(figure.axes, ("MAE_w", "MAE_TPavg", "MAE_within")):
                for line, split in zip(axis.lines[:2], ("validation", "test")):
                    np.testing.assert_allclose(line.get_xdata(), [0., .25, 1.])
                    np.testing.assert_allclose(
                        line.get_ydata(), [float(row[f"{split}_{metric}_g"]) for row in rows])
                np.testing.assert_allclose(axis.lines[2].get_xdata(), [0., 0.])
            self.assertNotIn(figure.number, plt.get_fignums())

    def test_single_beta_uses_m1_test_file_and_missing_metrics_stay_blank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "best_metrics.json").write_text('{"MAE": 5}')
            selection = {"selected_beta": .25, "experiments": [
                {"within_test_point_beta": .25, "validation_mae_g": 2.,
                 "run_dir": str(root)}]}
            with patch("src.Utility.beta_comparison.plt.close", wraps=plt.close) as close:
                paths = save_beta_mae_comparison(selection, root / "summary", "Single beta")
            with paths[1].open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(float(row["validation_MAE_w_g"]), 2.)
            self.assertEqual(float(row["test_MAE_w_g"]), 5.)
            self.assertEqual(row["test_MAE_within_g"], "")
            figure = close.call_args.args[0]
            np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), [2.])
            np.testing.assert_allclose(figure.axes[0].lines[1].get_ydata(), [5.])
            self.assertEqual(len(figure.axes[2].lines), 1)  # Selection marker only.


if __name__ == "__main__":
    unittest.main()
