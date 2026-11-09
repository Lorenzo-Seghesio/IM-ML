import copy
import io
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import optuna
import pandas as pd
import torch

from src import Reference_Models_W_Pred as ref
from src.Utility.multi_encoder_reference import MultiEncoderModel, fit_scalers, scale_inputs, suggest_params, gradcam
from src.Utility.hybrid_data import load_data, split_data, read_json, write_json
from src.Utility.external_evaluation import external_dataset
from src.Utility.pipeline_runtime import configure
from test_hybrid_models import divided_fixture, tiny_training


def architecture(lengths=(5, 7, 9)):
    return {"curve_encoders": [{"conv_channels": [1, 3], "conv_kernels": [3], "n_f_features": 2}
                               for _ in lengths], "signal_lengths": list(lengths), "head_hidden": [4], "dropout": 0.}


class MultiEncoderReferenceTests(unittest.TestCase):
    def test_each_signal_has_its_own_encoder_and_no_pp_branch(self):
        cfg = architecture()
        model = MultiEncoderModel(**cfg).eval()
        signals = [torch.rand(2, 1, length, requires_grad=True) for length in cfg["signal_lengths"]]
        features = torch.cat([enc(x) for enc, x in zip(model.encoders, signals)], dim=1)
        prediction = model(torch.cat(signals, dim=-1))
        torch.testing.assert_close(prediction, model.merge(features))
        self.assertEqual(len(model.encoders), 3)
        self.assertEqual(model.merge.net[0].in_features, 6)
        self.assertFalse(any("pp" in name for name, _ in model.named_parameters()))
        prediction.sum().backward()
        self.assertTrue(all(any(p.grad is not None for p in enc.parameters()) for enc in model.encoders))
        with self.assertRaisesRegex(ValueError, "Expected sensor input"):
            model(torch.rand(2, 1, 5))

    def test_optuna_samples_independent_encoders_and_head(self):
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=4))
        trial = study.ask()
        space = {"enc_n_conv_layers": [1, 2], "enc_channels": [2, 4], "enc_kernel_size": [3, 5],
                 "enc_pool_kernel_size": 1, "n_f_features": [2, 4], "head_n_layers": [1, 2],
                 "head_hidden_size": [3, 5], "dropout": 0., "l1_lambda": [1e-6, 1e-3]}
        before = copy.deepcopy(architecture())
        cfg, training = suggest_params(trial, space, before)
        self.assertEqual(before, architecture())
        self.assertTrue(all(f"enc_c{i}_n_conv_layers" in trial.params for i in range(3)))
        self.assertIn("merge_n_layers", trial.params)
        self.assertEqual(training["l1_lambda"], trial.params["l1_lambda"])
        self.assertFalse(any(key.startswith("pp_") for key in trial.params))
        replay, _ = suggest_params(optuna.trial.FixedTrial(trial.params), space, before)
        self.assertEqual(cfg, replay)
        self.assertEqual(MultiEncoderModel(**cfg)(torch.rand(2, 1, 21)).shape, (2,))

    def test_per_signal_scaling_fits_only_training_data(self):
        training = np.array([[0., 2., 100., 200.], [1., 3., 150., 250.]], dtype=np.float32)
        low, high = fit_scalers(training, [2, 2])
        self.assertEqual(low, [0., 100.])
        self.assertEqual(high, [3., 250.])
        external = np.full((1, 4), 1000.)
        self.assertTrue((scale_inputs(external, low, high, [2, 2]) > 1).all())
        self.assertEqual(low, [0., 100.])

    def test_gradcam_writes_joint_prediction_attribution_for_each_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            gradcam(MultiEncoderModel(**architecture()), np.random.default_rng(4).normal(size=(3, 21)).astype(np.float32),
                    Path(tmp), torch.device("cpu"))
            self.assertTrue((Path(tmp) / "xai_gradcam.png").is_file())

    def test_full_reference_hpo_cv_final_training_external_test_and_reload(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            cfg, _ = divided_fixture(root)
            data = load_data(cfg)
            splits, _ = split_data(data, cfg)
            cfg.update(training=tiny_training())
            cfg["evaluation"].update(enabled=True, plots=False)
            cfg["multi_encoder_hpo"] = {"n_trials": 1, "n_startup_trials": 1, "sampler_seed": 17,
                "n_folds": 2, "epochs": 1, "patience": 1, "search_space": {
                    "enc_n_conv_layers": 1, "enc_channels": 2, "enc_kernel_size": 3,
                    "n_f_features": 2, "head_n_layers": 1, "head_hidden_size": 3, "dropout": 0.}}
            cfg["encoder_hpo"] = cfg["multi_encoder_hpo"]
            stack.enter_context(patch.object(ref, "BASE_DIR", root))
            stack.enter_context(patch.object(ref, "BASE_OUT", root / "outputs"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            for name in ("save_cv_plots", "save_cv_diagnostics", "save_final_test_plots", "_save_batch_test_artifacts"):
                stack.enter_context(patch.object(ref, name))
            result = ref.run_encoder(np.concatenate(data.curves, axis=1), data.y, cfg, "DOE1", "DOE1", 42,
                torch.device("cpu"), None, np.concatenate(splits[:2]), splits[2],
                data.groups, cfg["data"]["group_split"], splits[0], splits[1],
                part_ids=data.ids, multi_encoder=True)
            out = Path(result["run_dir"])
            for name in ("best_model.pt", "best_metrics.json", "configuration.json", "data_processing.json",
                         "scalers.json", "hpo_trials.csv", "hpo_best_config.json", "training_history.json",
                         "test_2_metrics.json", "test_2_predictions.csv", "xai_gradcam.png"):
                self.assertTrue((out / name).exists(), name)
            meta, scale = read_json(out / "best_metrics.json"), read_json(out / "scalers.json")
            self.assertEqual(meta["input_layout"], "per_signal")
            model = MultiEncoderModel(**{key: meta[key] for key in ("curve_encoders", "signal_lengths", "head_hidden", "dropout")}).eval()
            model.load_state_dict(torch.load(out / "best_model.pt", weights_only=True))
            ext = external_dataset(cfg)
            inputs = scale_inputs(np.concatenate(ext.curves, axis=1), scale["pt_min"], scale["pt_max"], scale["signal_lengths"])
            with torch.no_grad():
                expected = model(ref._enc_prepare(inputs)).numpy()
            saved = pd.read_csv(out / "test_2_predictions.csv")
            np.testing.assert_allclose(saved.y_pred, expected, rtol=1e-6)
            self.assertEqual(saved.part_id.tolist(), ext.ids.tolist())

    def test_downstream_stage_inherits_reused_teacher_data_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher = {"scalar_csv": "legacy.csv", "target_col": "weight", "drop_cols": ["extra"],
                       "timeseries_csvs": [{"name": "p", "path": "legacy_pressure.csv"}], "id_col": "ID"}
            write_json(root / "M1/best_overall/configuration.json", {"data": teacher})
            for stage in ("M2", "Fusion", "RefModels"):
                with patch.dict("os.environ", {"IM_ML_RUN_DIR": str(root)}):
                    configured = configure({"data": {"scalar_csv": "new.csv"}}, stage, "DOE1")
                self.assertEqual(configured["data"]["scalar_csv"], "legacy.csv")
                self.assertEqual(configured["data"]["drop_cols"], ["extra", "weight"] if stage == "M2" else ["extra"])


if __name__ == "__main__":
    unittest.main()
