"""Tiny synthetic experiment checks; never runs the real dataset pipeline."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src import M1_PPPT_to_W as m1
from src import M2_PP_to_F as m2
from src import Fusion_M1M2_WPred as fusion


class DecompositionExperimentTests(unittest.TestCase):
    def test_tiny_three_variant_beta_selection_and_matching_downstream_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(42)
            groups = np.repeat(np.arange(20), 3)
            ids = np.arange(len(groups))
            nominal = groups / 20.
            scalar = pd.DataFrame({"Batch ID": groups, "nominal.pressure": nominal,
                                   "measured": nominal + rng.normal(0, .02, len(groups)),
                                   "Weight": 2 + nominal + rng.normal(0, .01, len(groups))},
                                  index=pd.Index(ids, name="ID_Part"))
            scalar.to_csv(root / "scalar.csv")
            pd.DataFrame(rng.normal(size=(len(ids), 8)), index=scalar.index).to_csv(root / "signal.csv")
            data = {"scalar_csv": str(root / "scalar.csv"), "id_col": "ID_Part",
                    "group_col": "Batch ID", "target_col": "Weight",
                    "timeseries_csvs": [{"name": "p1", "path": str(root / "signal.csv")}],
                    "drop_cols": ["Batch ID", "nominal.pressure"], "include_nominal_pp": True,
                    "group_split": {"enabled": True},
                    "edge_group_split": {"enabled": False, "columns": ["nominal.pressure"]}}
            prep = {"random_seed": 42, "test_fraction": .2,
                    "iqr_multiplier": 1.5, "remove_target_outliers": False}
            training = {"batch_size": 8, "lr": .005, "weight_decay": 0., "l1_lambda": 0.,
                        "epochs": 1, "final_epochs": 1, "patience": 1, "final_patience": 1,
                        "scheduler_factor": 1, "scheduler_patience": 1,
                        "print_every": 100, "k_folds": 2, "val_fraction": .25}
            cfg = {"mode": "optuna", "data": data, "preprocessing": prep,
                   "model": {"curve_encoders": [{"conv_channels": [1, 2], "conv_kernels": [3],
                                                  "n_f_features": 2}],
                             "pp_hidden": [4], "pp_latent_dim": 3,
                             "merge_hidden": [4], "dropout": 0.},
                   "training": training,
                   "output": {"output_dir": str(root / "M1"),
                              "run_best_dir": str(root / "M1/run_best"),
                              "best_overall_dir": str(root / "M1/best_overall")},
                   "decomposition": {"enabled": True, "beta": [.25, .75],
                                     "gamma": 1., "balance_lambda": 1., "share_hpo": True},
                   "optuna": {"n_trials": 1, "n_startup_trials": 1, "sampler_seed": 42,
                              "n_folds": 2, "epochs": 1, "patience": 1,
                              "hyperband_min_resource": 1, "hyperband_reduction_factor": 3,
                              "study_name": "synthetic_decomposition_test", "storage": None,
                              "search_space": {"n_f_features": [2, 2], "pp_n_features": [3, 3]}}}
            with contextlib.ExitStack() as stack:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                for name in ["save_cv_plots", "save_cv_diagnostics", "save_final_test_plots",
                             "save_batch_test_diagnostics"]:
                    stack.enter_context(patch.object(m1, name))
                hpo = stack.enter_context(patch.object(m1, "run_hpo", wraps=m1.run_hpo))
                m1.run_decomposition_experiments(cfg, "PP", root / "m1.json")
                self.assertEqual(hpo.call_count, 1)
            run = root / "M1/run_best" / m1._RUN_TS
            comparison = pd.read_csv(run / "beta_comparison.csv")
            self.assertEqual(len(comparison), 6)
            manifest = json.loads((root / "M1/best_overall/experiment_variants.json").read_text())
            self.assertEqual(len(manifest["variants"]), 3)
            for entry in manifest["variants"]:
                candidates = comparison[comparison.variant == entry["variant"]]
                expected = candidates.loc[candidates.selection_MAE.idxmin()]
                self.assertEqual(entry["beta"], expected.beta)
                saved = json.loads((Path(entry["m1_dir"]) / "best_metrics.json").read_text())
                self.assertEqual(saved["model_cfg"]["loss_settings"]["beta"], expected.beta)
                dp = json.loads((Path(entry["m1_dir"]) / "data_processing.json").read_text())
                self.assertIn("nominal.pressure", dp["pp_cols"])
                self.assertIn(entry["variant"], dp["run_ts"])
                self.assertFalse(set(dp["train_groups"]) & set(dp["val_groups"]))
                for beta in [.25, .75]:
                    self.assertTrue((run / entry["variant"] / f"beta_{beta:g}" / "best_model.pt").exists())

            cfg_m2 = {"mode": "manual", "data": copy.deepcopy(data), "preprocessing": prep,
                      "model": {"hidden_dims": [4], "dropout": 0.}, "training": training,
                      "output": {"output_dir": str(root / "M2"),
                                 "run_best_dir": str(root / "M2/run_best"),
                                 "best_overall_dir": str(root / "M2/best_overall")}}
            cfg_m2["data"]["features_csv"] = str(root / "M1/best_overall/sensors_features_f.csv")
            cfg_m2["data"]["drop_cols"] = ["Batch ID", "Weight", "nominal.pressure"]
            (root / "m2.json").write_text(json.dumps(cfg_m2))
            with contextlib.ExitStack() as stack:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(patch("sys.argv", ["M2", "--material", "PP", "--config", str(root / "m2.json")]))
                for name in ["save_cv_plots", "save_cv_diagnostics", "save_final_test_plots",
                             "save_batch_test_diagnostics"]:
                    stack.enter_context(patch.object(m2, name))
                m2.main()
            m2_manifest = json.loads((root / "M2/best_overall/experiment_variants.json").read_text())
            for entry in m2_manifest["variants"]:
                dp = json.loads((Path(entry["m1_dir"]) / "data_processing.json").read_text())
                scales = json.loads((Path(entry["m2_dir"]) / "scalers.json").read_text())
                self.assertEqual(dp["run_ts"], scales["m1_run_ts"])

            cfg_fusion = {"data": data, "models": {"m1_best_dir": str(root / "M1/best_overall"),
                                                    "m2_best_dir": str(root / "M2/best_overall")},
                          "output": {"output_base_dir": str(root / "Fusion")},
                          "finetuning": {"alpha": [.3, .7], "epochs": 1, "patience": 1,
                                         "batch_size": 8, "lr": .001}}
            (root / "fusion.json").write_text(json.dumps(cfg_fusion))
            # Run gradient-based XAI on tiny fixtures, suppress only plot generation.
            with contextlib.ExitStack() as stack:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(patch("sys.argv", ["Fusion", "--material", "PP", "--config",
                    str(root / "fusion.json"), "--n-ig-steps", "2", "--eg-bg", "1",
                    "--shap-bg", "2", "--ih-samples", "1"]))
                for name in dir(fusion):
                    if callable(getattr(fusion, name)) and (name.startswith("save_") or name.startswith("_save_")):
                        stack.enter_context(patch.object(fusion, name))
                fusion.main()
            fusion_run = next((root / "Fusion").iterdir())
            results = pd.read_csv(fusion_run / "variant_alpha_comparison.csv")
            self.assertEqual(len(results), 6)
            for entry in manifest["variants"]:
                m1_dir = Path(entry["m1_dir"])
                original = torch.load(m1_dir / "best_model.pt", weights_only=True)
                for alpha in [.3, .7]:
                    folder = fusion_run / entry["variant"] / f"alpha_{alpha}"
                    info = json.loads((folder / "finetuning_info.json").read_text())
                    self.assertEqual(info["weight_loss_settings"]["beta"], entry["beta"])
                    self.assertEqual(info["weight_loss_settings"]["variant"], entry["variant"])
                    components = pd.read_csv(folder / "test_component_predictions.csv")
                    np.testing.assert_allclose(components.avg_pred + components.dev_pred,
                                               components.weight_pred, rtol=1e-6, atol=1e-6)
                    self.assertTrue((folder / "jacobian_values.csv").exists())
                    if fusion._HAS_SHAP:
                        self.assertTrue((folder / "shap_values.csv").exists())
                    state = torch.load(folder / "fusion_model.pt", weights_only=True)
                    for key, value in original.items():
                        if key.startswith(("pp_mlp.", "merge.")):
                            torch.testing.assert_close(state[key], value, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
