import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from src.M1_PPPT_to_W import (
    _build_sensor_feature_schema,
    _migrate_legacy_best_features,
    _timeseries_sources,
    _validate_curve_encoder_config,
    extract_features,
)


class M1SensorFeatureSchemaTests(unittest.TestCase):
    def test_signal_count_resolves_to_one_time_concatenated_encoder(self):
        data = {"timeseries_csvs": [{"name": "p1"}, {"name": "p2"}]}
        model = {"conv_channels": [3, 8], "n_f_features": 4}
        _validate_curve_encoder_config(data, model)
        self.assertEqual(model["n_signals"], 2)
        self.assertEqual(model["conv_channels"], [1, 8])
        self.assertEqual(model["encoder_input_layout"], "concatenate_time")

    def test_old_training_configs_resolve_to_one_encoder_with_total_feature_width(self):
        data = {"timeseries_csvs": [{"name": "p1"}, {"name": "T"}]}
        model = {"curve_encoders": [
            {"conv_channels": [1, 8], "conv_kernels": [3], "n_f_features": 4},
            {"conv_channels": [1, 16], "conv_kernels": [5], "n_f_features": 2}]}
        _validate_curve_encoder_config(data, model)
        self.assertNotIn("curve_encoders", model)
        self.assertEqual(model["conv_channels"], [1, 8])
        self.assertEqual(model["n_f_features"], 6)

    def test_source_names_support_current_and_legacy_configuration(self):
        sources = [{"name": "p1", "path": "p1.csv"},
                   {"name": "p2", "path": "p2.csv"}]
        for key in ("timeseries_csvs", "pressure_csvs"):
            self.assertEqual(_timeseries_sources({key: sources}), sources)
        self.assertEqual(_timeseries_sources({"pressure_csv": "pressure.csv"}),
                         [{"name": "pressure", "path": "pressure.csv"}])
        with self.assertRaisesRegex(ValueError, "at least one signal"):
            _timeseries_sources({"timeseries_csvs": []})

    def test_feature_columns_follow_signal_and_curve_order(self):
        columns, signals = _build_sensor_feature_schema(
            ["p1", "p2", "temp"], [2, 1, 2],
        )

        self.assertEqual(
            columns,
            ["f_p1_0", "f_p1_1", "f_p2_0", "f_T_0", "f_T_1"],
        )
        self.assertEqual([signal["n_features"] for signal in signals], [2, 1, 2])
        self.assertEqual([signal["signal_name"] for signal in signals], ["p1", "p2", "T"])

    def test_rejects_duplicate_signal_names(self):
        with self.assertRaisesRegex(ValueError, "duplicate sensor feature columns"):
            _build_sensor_feature_schema(["temp", "T"], [1, 1])

    def test_rejects_curve_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "matching lengths"):
            _build_sensor_feature_schema(["p1"], [1, 2])

    def test_extract_features_saves_named_columns_and_summary(self):
        class DummyModel:
            @staticmethod
            def get_f_by_curve(curves):
                n_samples = curves[0].shape[0]
                return [torch.ones((n_samples, 2)), torch.zeros((n_samples, 1))]

        x_pp = np.array([[0.0], [1.0]], dtype=np.float32)
        x_pt = [
            np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32),
            np.array([[10.0], [20.0]], dtype=np.float32),
        ]
        scalers = (
            MinMaxScaler().fit(x_pp),
            [{"min": 0.0, "max": 2.0}, {"min": 10.0, "max": 20.0}],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "sensors_features_f.csv"
            frame, summary = extract_features(
                DummyModel(), scalers, x_pp, x_pt, np.array([10, 11]),
                csv_path, "ID_Part", torch.device("cpu"),
                ["p1", "temp"], {"pp_hidden": [4, 8]},
            )

            self.assertEqual(list(frame.columns), ["f_p1_0", "f_p1_1", "f_T_0"])
            self.assertTrue(csv_path.exists())
            loaded = pd.read_csv(csv_path, index_col=0)
            self.assertEqual(list(loaded.columns), list(frame.columns))
            saved_summary = json.loads(
                (Path(temp_dir) / "feature_summary.json").read_text())
            self.assertEqual(saved_summary, summary)
            self.assertEqual(summary["curve_feature_total"], 3)
            self.assertEqual(summary["pp_embedding"]["n_features"], 8)
            self.assertEqual(summary["merge_head_input_width"], 11)

    def test_migrates_existing_best_features_using_saved_model_metadata(self):
        model_cfg = {
            "curve_encoders": [
                {"n_f_features": 2}, {"n_f_features": 1},
            ],
            "pp_hidden": [4, 3],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            best_dir = Path(temp_dir)
            (best_dir / "best_metrics.json").write_text(
                json.dumps({"model_cfg": model_cfg}))
            legacy = pd.DataFrame(
                [[10.0, 20.0, 30.0]],
                index=pd.Index([101], name="Product ID"),
                columns=["f_0", "f_1", "f_2"],
            )
            legacy.to_csv(best_dir / "pressure_features_f.csv")

            _migrate_legacy_best_features(
                best_dir,
                {"id_col": "Product ID"},
                ["p1", "temp"],
                n_pp=4,
            )

            migrated = pd.read_csv(
                best_dir / "sensors_features_f.csv", index_col="Product ID")
            self.assertEqual(
                migrated.columns.tolist(), ["f_p1_0", "f_p1_1", "f_T_0"])
            self.assertEqual(migrated.iloc[0].tolist(), [10.0, 20.0, 30.0])
            self.assertTrue((best_dir / "pressure_features_f.csv").exists())
            summary = json.loads((best_dir / "feature_summary.json").read_text())
            self.assertEqual(summary["pp_embedding"]["n_features"], 3)
            self.assertEqual(summary["merge_head_input_width"], 6)


if __name__ == "__main__":
    unittest.main()
