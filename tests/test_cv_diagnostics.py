import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from src.Utility.cv_diagnostics import save_cv_diagnostics


class CvDiagnosticsTests(unittest.TestCase):
    def _records(self):
        return [
            {"fold": 0, "y_true": np.array([[1.0, 2.0], [2.0, 4.0]]),
             "y_pred": np.array([[1.1, 1.8], [2.2, 4.1]])},
            {"fold": 1, "y_true": np.array([[3.0, 2.0], [4.0, 1.0]]),
             "y_pred": np.array([[2.9, 2.1], [4.2, 0.9]])},
        ]

    def test_aggregate_only_mode_skips_per_fold_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            saved = save_cv_diagnostics(
                self._records(), out_dir, "test", ["f_0", "f_1"],
                save_per_fold_plots=False,
            )

            self.assertEqual(len(saved), 2)
            self.assertEqual(
                {path.name for path in saved},
                {"cv_diagnostics_f_0.png", "cv_diagnostics_f_1.png"},
            )
            self.assertTrue(all(path.exists() for path in saved))
            self.assertEqual(list(out_dir.glob("cv_fold_*_diagnostics*.png")), [])

    def test_per_fold_images_remain_enabled_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            saved = save_cv_diagnostics(
                self._records(), out_dir, "test", ["f_0", "f_1"])

            self.assertEqual(len(saved), 6)
            self.assertEqual(len(list(out_dir.glob("cv_fold_*_diagnostics*.png"))), 4)


if __name__ == "__main__":
    unittest.main()
