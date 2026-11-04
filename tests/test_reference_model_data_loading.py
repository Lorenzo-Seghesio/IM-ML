import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import Reference_Models_W_Pred as ref


class ReferenceModelDataLoadingTests(unittest.TestCase):
    def test_load_data_joins_multiple_timeseries_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pd.DataFrame({
                'ID': [1, 2],
                'weight': [10.0, 20.0],
                'feature': [3.0, 4.0],
            }).to_csv(root / 'scalar.csv', index=False)
            pd.DataFrame({
                'ID': [1, 2],
                'v0': [0.1, 0.2],
                'v1': [0.3, 0.4],
            }).to_csv(root / 'pressure.csv', index=False)
            pd.DataFrame({
                'ID': [1, 2],
                't0': [50.0, 60.0],
            }).to_csv(root / 'temperature.csv', index=False)

            data_cfg = {
                'scalar_csv': 'scalar.csv',
                'id_col': 'ID',
                'target_col': 'weight',
                'timeseries_csvs': [
                    {'name': 'p1', 'path': 'pressure.csv'},
                    {'name': 'T1', 'path': 'temperature.csv'},
                ],
            }
            with patch.object(ref, 'BASE_DIR', root):
                result = ref.load_data(
                    data_cfg, {'remove_target_outliers': False}, material='ABS',
                )

        part_ids, x_pp, x_pt, y, strat_labels, pp_cols, group_ids, edge_values = result
        np.testing.assert_array_equal(part_ids, [1, 2])
        np.testing.assert_array_equal(x_pp, [[3.0], [4.0]])
        np.testing.assert_allclose(x_pt, [[0.1, 0.3, 50.0], [0.2, 0.4, 60.0]])
        np.testing.assert_array_equal(y, [10.0, 20.0])
        self.assertIsNone(strat_labels)
        self.assertEqual(pp_cols, ['feature'])
        self.assertIsNone(group_ids)
        self.assertIsNone(edge_values)

    def test_timeseries_sources_preserve_legacy_pressure_configs(self):
        pressure_sources = [{'name': 'p1', 'path': 'p1.csv'}]
        self.assertEqual(
            ref._timeseries_sources({'pressure_csvs': pressure_sources}),
            pressure_sources,
        )
        self.assertEqual(
            ref._timeseries_sources({'pressure_csv': 'pressure.csv'}),
            [{'name': 'pressure', 'path': 'pressure.csv'}],
        )

    def test_timeseries_sources_require_at_least_one_signal(self):
        with self.assertRaisesRegex(ValueError, 'data must define'):
            ref._timeseries_sources({})
        with self.assertRaisesRegex(ValueError, 'time-series source'):
            ref._timeseries_sources({'timeseries_csvs': []})
