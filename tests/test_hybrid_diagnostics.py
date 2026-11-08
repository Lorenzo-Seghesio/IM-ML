"""Plot artifacts distinguish final test data and retain only observed targets."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.Utility.hybrid_diagnostics import save_partition_plots, save_cv_plots


class HybridDiagnosticsTests(unittest.TestCase):
    def test_final_and_validation_plots_keep_partition_and_observed_ids(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            for partition in ("validation", "test"):
                save_partition_plots(root, partition, [1., np.nan, 3.], [1.2, 2., 2.8],
                                     ["weight"], "Fusion", ["A", "A", "B"], [10, 11, 12])
                output = root / partition
                suffix = "final_test" if partition == "test" else partition
                for prefix in ("scatter", "residuals_scatter", "residuals_hist"):
                    self.assertGreater((output / f"{prefix}_{suffix}.png").stat().st_size, 0)
                self.assertTrue((output / f"{partition}_scatter_residuals_by_batch.png").exists())
                frame = pd.read_csv(output / f"{partition}_predictions_per_batch.csv")
                self.assertEqual(frame["sample_id"].tolist(), [10, 12])
            self.assertFalse(any((root / "validation").glob("*test*")))

    def test_cv_draws_all_folds_and_handles_unavailable_feature_targets(self):
        records = [{"fold": i, "y_true": np.array([[1., np.nan], [2., np.nan]]),
                    "y_pred": np.array([[1.1, 3.], [1.9, 3.]])} for i in range(2)]
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            save_cv_plots(root, records, ["f1", "f2"], "M2 CV", {"save_per_fold_plots": True})
            self.assertTrue((root / "cv_metrics_folds.png").exists())
            self.assertTrue((root / "cv_diagnostics_f1.png").exists())
            self.assertTrue((root / "cv_diagnostics_f2.png").exists())
            self.assertEqual(len(list(root.glob("cv_fold_*_diagnostics_*.png"))), 4)


if __name__ == "__main__":
    unittest.main()
