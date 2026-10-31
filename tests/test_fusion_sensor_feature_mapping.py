import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd

from src.Fusion_M1M2_WPred import get_f_positions, _select_m2_feature_columns


class FusionSensorFeatureMappingTests(unittest.TestCase):
    def test_new_m2_signal_names_map_by_column_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            m1_dir = Path(temp_dir)
            pd.DataFrame(
                [[1.0, 2.0, 3.0]],
                index=pd.Index([10], name="ID_Part"),
                columns=["f_p1_0", "f_p1_1", "f_T_0"],
            ).to_csv(m1_dir / "sensors_features_f.csv")

            positions = get_f_positions(m1_dir, ["f_p1_1", "f_T_0"])
            self.assertEqual(positions, [1, 2])

    def test_legacy_m2_names_map_by_preserved_feature_position(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            m1_dir = Path(temp_dir)
            df = pd.DataFrame(
                [[1.0, 2.0, 3.0]],
                index=pd.Index([10], name="ID_Part"),
                columns=["f_p1_0", "f_p1_1", "f_T_0"],
            )
            df.to_csv(m1_dir / "sensors_features_f.csv")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                positions = get_f_positions(m1_dir, ["f_1"])
            selected = _select_m2_feature_columns(df, ["f_1"], positions)

            self.assertEqual(positions, [1])
            self.assertEqual(selected.columns.tolist(), ["f_1"])
            self.assertEqual(selected.iloc[0, 0], 2.0)

    def test_legacy_feature_file_is_supported_with_warning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            m1_dir = Path(temp_dir)
            (m1_dir / "pressure_features_f.csv").write_text("ID_Part,f_0\n10,1.0\n")

            with self.assertWarnsRegex(RuntimeWarning, "legacy artifact"):
                positions = get_f_positions(m1_dir, ["f_0"])
            self.assertEqual(positions, [0])


if __name__ == "__main__":
    unittest.main()
