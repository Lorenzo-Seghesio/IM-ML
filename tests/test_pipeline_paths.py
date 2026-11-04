"""Flat DoE1 routing and first-launch dependencies; no training subprocesses."""
import io
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src.Utility import family_launcher as launcher
from src.Utility import pipeline_runtime as runtime
from src.Utility.hybrid_data import ROOT, read_json, write_json
from src.Utility.hybrid_models import stage_dirs


class PipelinePathTests(unittest.TestCase):
    def test_doe1_configuration_sources_match_original_outputs(self):
        configs = {path.name: read_json(path) for path in (ROOT / "config" / "DoE1").glob("*.json")}
        m1 = configs["M1_DoE1_config.json"]["output"]
        m2 = configs["M2_DoE1_config.json"]["output"]
        self.assertEqual(Path(m1["best_overall_dir"]), Path(m1["output_dir"]) / "best_overall")
        self.assertEqual(Path(m2["best_overall_dir"]), Path(m2["output_dir"]) / "best_overall")
        self.assertEqual(Path(configs["M2_DoE1_config.json"]["data"]["features_csv"]),
                         Path(m1["best_overall_dir"]) / "sensors_features_f.csv")
        self.assertEqual(configs["Fusion_DoE1_config.json"]["models"]["m1_best_dir"], m1["best_overall_dir"])
        self.assertEqual(configs["Fusion_DoE1_config.json"]["models"]["m2_best_dir"], m2["best_overall_dir"])
        self.assertEqual(configs["RefModels_DoE1_config.json"]["m1_split_dir"], m1["best_overall_dir"])
        for name, cfg in configs.items():
            with self.subTest(config=name):
                data = cfg.get("data", {})
                if data.get("scalar_csv"):
                    self.assertTrue((ROOT / data["scalar_csv"]).is_file())
                for source in data.get("timeseries_csvs", []):
                    self.assertTrue((ROOT / source["path"]).is_file())
                for path in cfg.get("configs", {}).values():
                    self.assertTrue((ROOT / path).is_file())
                if "split" in cfg:
                    self.assertEqual(cfg["split"]["source_dir"], m1["best_overall_dir"])
                if cfg.get("teacher", {}).get("checkpoint_dir"):
                    self.assertEqual(cfg["teacher"]["checkpoint_dir"], m1["best_overall_dir"])
                for path in cfg.get("output", {}).values():
                    if isinstance(path, str) and path.startswith("outputs/"):
                        self.assertEqual(Path(path).parts.count("DOE1"), 1)

    def test_runtime_paths_flatten_doe1_and_keep_other_material_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {"IM_ML_RUN_DIR": str(root), "IM_ML_RUN_SEED": "42"}):
                for material in ("DOE1", "PP"):
                    with self.subTest(material=material):
                        m1 = runtime.configure({}, "M1", material)
                        m2 = runtime.configure({"data": {}}, "M2", material)
                        fusion = runtime.configure({}, "Fusion", material)
                        reference = runtime.configure({}, "RefModels", material)
                        parent = root / "M1" if material == "DOE1" else root / "M1" / "PP"
                        self.assertEqual(Path(m1["output"]["best_overall_dir"]), parent / "best_overall")
                        self.assertEqual(m2["data"]["features_csv"], str(parent / "best_overall" / "sensors_features_f.csv"))
                        self.assertEqual(fusion["models"]["m1_best_dir"], m1["output"]["best_overall_dir"])
                        self.assertEqual(fusion["models"]["m2_best_dir"], m2["output"]["best_overall_dir"])
                        self.assertEqual(reference["m1_split_dir"], m1["output"]["best_overall_dir"])
                        _, best = stage_dirs(root, "M1", material)
                        self.assertEqual(best, parent / "best_overall")

    def test_first_original_run_publishes_coherent_pair_using_validation_only(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            scalar = root / "scalars.csv"
            scalar.write_text("ID,weight\n1,10\n")
            paths = {}
            for stage in ("M1", "M2", "Fusion"):
                paths[stage] = root / f"{stage}.json"
                write_json(paths[stage], {"data": {"scalar_csv": str(scalar)},
                    "output": {"best_overall_dir": str(root / "original" / stage / "best_overall")}})
            candidate = {"validation": .5, "test": 100.}
            calls = []
            def fake_subprocess(command, *, cwd, env, check):
                stage = "M1" if Path(command[1]).name.startswith("M1_") else "M2" if Path(command[1]).name.startswith("M2_") else "Fusion"
                run = Path(env["IM_ML_RUN_DIR"])
                with patch.dict(os.environ, env):
                    cfg = runtime.configure(read_json(paths[stage]), stage, "DOE1")
                local_m1 = run / "M1" / "best_overall"
                if stage == "Fusion":
                    self.assertEqual(Path(cfg["models"]["m1_best_dir"]), local_m1)
                    self.assertTrue((Path(cfg["models"]["m2_best_dir"]) / "scalers.json").is_file())
                    return
                target = Path(cfg["output"]["best_overall_dir"])
                target.mkdir(parents=True, exist_ok=True)
                (target / "best_model.pt").write_text(run.name)
                write_json(target / "best_metrics.json", {"source": "final_test", "MAE": candidate["test"],
                                                          "validation_MAE": candidate["validation"]})
                if stage == "M1":
                    write_json(target / "data_processing.json", {"run_ts": run.name, "pp_cols": ["pp"],
                        "train_part_ids": [1], "val_part_ids": [2], "test_part_ids": [3]})
                    (target / "sensors_features_f.csv").write_text("ID,f\n1,2\n")
                else:
                    self.assertEqual(Path(cfg["data"]["features_csv"]), local_m1 / "sensors_features_f.csv")
                    self.assertTrue(Path(cfg["data"]["features_csv"]).is_file())
                    write_json(target / "scalers.json", {"m1_run_ts": run.name})
                calls.append(stage)
            canonical = root / "original" / "M1" / "best_overall"
            self.assertFalse(canonical.exists())
            with patch.object(launcher, "original_configs", return_value=paths), \
                    patch.object(launcher.subprocess, "run", side_effect=fake_subprocess):
                launcher._run_original("DOE1", root / "run_001", 42, None, {"reference": {"enabled": False}})
                self.assertEqual(read_json(canonical / "data_processing.json")["run_ts"], "run_001")
                candidate.update(validation=.8, test=0.)
                launcher._run_original("DOE1", root / "run_002", 1051, None, {"reference": {"enabled": False}})
                self.assertEqual(read_json(canonical / "data_processing.json")["run_ts"], "run_001")
                candidate.update(validation=.2, test=1000.)
                launcher._run_original("DOE1", root / "run_003", 2060, None, {"reference": {"enabled": False}})
            published_m2 = root / "original" / "M2" / "best_overall"
            self.assertEqual(read_json(canonical / "data_processing.json")["run_ts"], "run_003")
            self.assertEqual(read_json(published_m2 / "scalers.json")["m1_run_ts"], "run_003")
            self.assertEqual(calls, ["M1", "M2"] * 3)
            self.assertFalse((root / "run_001" / "M1" / "DOE1").exists())


if __name__ == "__main__":
    unittest.main()
