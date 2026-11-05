"""Selected Fusion reconstruction uses saved training scales and part IDs."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.Utility.fusion_reconstruction import save_fusion_reconstruction


class FusionReconstructionTests(unittest.TestCase):
    def test_reports_feature_units_and_training_scaled_mae_with_plots(self):
        truth = np.array([[10., 0.], [20., 2.], [30., 4.]])
        pred = truth + [1., -2.]
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            out = Path(directory)
            report = save_fusion_reconstruction(truth, pred, ['p_feature', 'T_feature'],
                [101, 103, 109], ['A', 'A', 'B'], out, 'DOE1', target_scale=[.1, 2.])
            self.assertAlmostEqual(report['mean_MAE'], 1.5)
            self.assertAlmostEqual(report['mean_NMAE'], 2.05)
            self.assertAlmostEqual(report['per_target']['p_feature']['NMAE'], .1)
            self.assertAlmostEqual(report['per_target']['T_feature']['NMAE'], 4.)
            self.assertEqual(report['target_reference'], 'frozen M1 extracted signal features')
            recon = out / 'm2_reconstruction'
            self.assertEqual(json.loads((recon / 'test_metrics.json').read_text())['n_samples'], 3)
            frame = pd.read_csv(recon / 'test_feature_predictions.csv')
            self.assertEqual(frame['ID_Part'].tolist(), [101, 103, 109])
            np.testing.assert_array_equal(frame['T_feature_pred'], pred[:, 1])
            for name in ('scatter_final_test.png', 'residuals_scatter_final_test.png',
                         'residuals_hist_final_test.png', 'test_scatter_residuals_by_batch.png'):
                self.assertGreater((recon / name).stat().st_size, 0)
            batches = pd.read_csv(recon / 'test_metrics_per_batch.csv')
            self.assertEqual(len(batches), 4)

    def test_validation_mode_does_not_create_test_artifacts(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            out = Path(directory)
            report = save_fusion_reconstruction([[1.], [2.]], [[1.1], [2.1]], ['f'],
                [1, 2], None, out, 'DOE1', partition='validation', target_scale=[.5], plots=False)
            self.assertEqual(report['source'], 'validation')
            self.assertFalse(any(out.rglob('test_*')))
            self.assertTrue((out / 'm2_reconstruction' / 'validation_metrics.json').is_file())


if __name__ == '__main__':
    unittest.main()
