import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.Utility.batch_test_diagnostics import save_batch_test_diagnostics


class BatchTestDiagnosticsTests(unittest.TestCase):
    def test_saves_metrics_predictions_and_combined_plot(self):
        y_true = np.array([1.0, 3.0, 2.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        batch_ids = np.array(["A", "A", "B"])
        sample_ids = np.array([101, 102, 103])

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            result = save_batch_test_diagnostics(
                y_true, y_pred, batch_ids, sample_ids, out_dir,
                target_names=["weight"], title="unit test",
            )

            self.assertIsNotNone(result)
            metrics, predictions = result
            batch_a = metrics.loc[metrics["batch_id"] == "A"].iloc[0]
            batch_b = metrics.loc[metrics["batch_id"] == "B"].iloc[0]
            self.assertEqual(batch_a["n_samples"], 2)
            self.assertAlmostEqual(batch_a["MAE"], 0.5)
            self.assertAlmostEqual(batch_a["MSE"], 0.5)
            self.assertAlmostEqual(batch_a["RMSE"], np.sqrt(0.5))
            self.assertAlmostEqual(batch_a["R2"], 0.5)
            self.assertEqual(batch_b["n_samples"], 1)
            self.assertTrue(np.isnan(batch_b["R2"]))
            self.assertEqual(predictions["sample_id"].tolist(), [101, 102, 103])
            self.assertEqual(predictions["residual"].tolist(), [0.0, -1.0, 2.0])

            metrics_path = out_dir / "test_metrics_per_batch.csv"
            predictions_path = out_dir / "test_predictions_per_batch.csv"
            plot_path = out_dir / "test_scatter_residuals_by_batch.png"
            self.assertTrue(metrics_path.exists())
            self.assertTrue(predictions_path.exists())
            self.assertGreater(plot_path.stat().st_size, 0)
            self.assertEqual(len(pd.read_csv(metrics_path)), 2)

    def test_multitarget_metrics_are_reported_per_target(self):
        y_true = np.array([[1.0, 2.0], [3.0, 4.0], [2.0, 1.0], [4.0, 3.0]])
        y_pred = y_true + np.array([[0.1, -0.2], [0.2, 0.3], [-0.1, 0.1], [0.0, -0.1]])

        with tempfile.TemporaryDirectory() as temp_dir:
            metrics, predictions = save_batch_test_diagnostics(
                y_true, y_pred, ["A", "A", "B", "B"],
                [10, 11, 12, 13], Path(temp_dir),
                target_names=["f_p1_0", "f_T_0"],
            )

        self.assertEqual(set(metrics["target"]), {"f_p1_0", "f_T_0"})
        self.assertEqual(len(metrics), 4)
        self.assertEqual(len(predictions), 8)

    def test_skips_when_batch_ids_are_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = save_batch_test_diagnostics(
                [1.0], [1.2], None, [7], Path(temp_dir),
            )
            self.assertIsNone(result)
            self.assertEqual(list(Path(temp_dir).iterdir()), [])

    def test_rejects_misaligned_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "matching sample counts"):
                save_batch_test_diagnostics(
                    [1.0, 2.0], [1.0, 2.0], ["A"], [10, 11], Path(temp_dir),
                )


if __name__ == "__main__":
    unittest.main()
