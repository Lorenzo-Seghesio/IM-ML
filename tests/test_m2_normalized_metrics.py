"""M2 normalized selection scores and original-unit MAE reporting."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import optuna
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from src import M2_PP_to_F as m2


class ConstantScaledOutput(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.register_buffer("output", torch.as_tensor(output, dtype=torch.float32))

    def forward(self, x):
        return self.output.unsqueeze(0).expand(len(x), -1)


def metrics(mean_mae, mean_nmae, pred):
    return {
        "mean_MAE": mean_mae,
        "mean_NMAE": mean_nmae,
        "mean_RMSE": 0.0,
        "mean_R2": 0.0,
        "mean_MSE": 0.0,
        "per_target": {},
        "pred": pred,
    }


class M2NormalizedMetricTests(unittest.TestCase):
    def test_training_uses_only_prediction_mae_even_with_legacy_l1_setting(self):
        model = nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.fill_(2.)
            model.bias.zero_()
        x = np.zeros((4, 1), dtype=np.float32)
        y = np.ones((4, 1), dtype=np.float32)
        cfg = {'lr': .01, 'weight_decay': 0., 'scheduler_factor': 1.,
               'batch_size': 4, 'epochs': 1, 'patience': 1, 'print_every': 99,
               'l1_lambda': 100.}
        trained, _ = m2.train_fold(model, x, y, x, y, cfg, torch.device('cpu'))
        # These weights have zero prediction gradient because every input is zero.
        # A parameter penalty would still shrink them and increase training loss.
        torch.testing.assert_close(trained.weight, torch.full_like(trained.weight, 2.), rtol=0, atol=0)
        self.assertEqual(trained.m2_training_info['history'][0]['train_loss'], 1.)
        self.assertGreater(trained.bias.item(), 0.)

    def test_hpo_does_not_sample_legacy_l1_regularization(self):
        trial = optuna.trial.FixedTrial({'lr': .001})
        _, updates = m2.suggest_hyperparams(trial, {'lr': [.0001, .01],
                                                  'l1_lambda': [.000001, .01]}, 1)
        self.assertEqual(updates, {'lr': .001})
        self.assertNotIn('l1_lambda', trial.params)

    def test_evaluation_keeps_original_unit_mae_and_adds_range_normalized_mae(self):
        y_train = np.array([[0.0, 0.0], [10.0, 100.0]], dtype=np.float32)
        y_sc = MinMaxScaler().fit(y_train)
        output = y_sc.transform(np.array([[1.0, 1.0]], dtype=np.float32))[0]
        y_test = y_train.copy()

        result = m2.evaluate_model(
            ConstantScaledOutput(output),
            np.zeros((2, 1), dtype=np.float32),
            y_test,
            y_sc,
            ["f_small_range", "f_large_range"],
            torch.device("cpu"),
        )

        self.assertAlmostEqual(result["per_target"]["f_small_range"]["MAE"], 5.0)
        self.assertAlmostEqual(result["per_target"]["f_large_range"]["MAE"], 50.0)
        self.assertAlmostEqual(result["mean_MAE"], 27.5)
        self.assertAlmostEqual(result["mean_NMAE"], 0.5)

    def test_cv_best_fold_is_selected_by_normalized_mae(self):
        x = np.arange(6, dtype=np.float32).reshape(-1, 1)
        y = np.column_stack((x[:, 0], 10.0 * x[:, 0])).astype(np.float32)
        splits = [
            {"tr_idx": np.array([0, 1]), "val_idx": np.array([2]),
             "test_idx": np.array([3, 4])},
            {"tr_idx": np.array([3, 4]), "val_idx": np.array([5]),
             "test_idx": np.array([0, 1])},
        ]
        observed = [
            metrics(1.0, 0.5, np.zeros((2, 2))),
            metrics(100.0, 0.1, np.zeros((2, 2))),
        ]
        with patch.object(m2, "train_fold", side_effect=lambda model, *_: (model, 0.0)), \
                patch.object(m2, "evaluate_model", side_effect=observed), \
                redirect_stdout(io.StringIO()):
            result = m2.run_cv(
                x, y, {"hidden_dims": [2], "dropout": 0.0}, {}, ["f0", "f1"],
                torch.device("cpu"), 42, precomputed_splits=splits,
            )

        fold_metrics, best_fold_idx, best_fold_nmae = result[:3]
        self.assertEqual(best_fold_idx, 1)
        self.assertAlmostEqual(best_fold_nmae, 0.1)
        np.testing.assert_allclose(fold_metrics["mean_NMAE"], [0.5, 0.1])

    def test_optuna_objective_averages_normalized_mae(self):
        x = np.arange(8, dtype=np.float32).reshape(-1, 1)
        y = np.column_stack((x[:, 0], 10.0 * x[:, 0])).astype(np.float32)
        observed = [
            metrics(100.0, 0.1, np.zeros((4, 2))),
            metrics(1.0, 0.3, np.zeros((4, 2))),
        ]
        optuna_cfg = {
            "search_space": {},
            "n_folds": 2,
            "epochs": 1,
            "patience": 1,
            "val_fraction": 0.25,
        }
        train_cfg = {"val_fraction": 0.25}
        with patch.object(m2, "train_fold", side_effect=lambda model, *_: (model, 0.0)), \
                patch.object(m2, "evaluate_model", side_effect=observed), \
                redirect_stdout(io.StringIO()):
            score = m2.optuna_objective(
                optuna.trial.FixedTrial({}), x, y, optuna_cfg,
                {"hidden_dims": [2], "dropout": 0.0}, train_cfg,
                ["f0", "f1"], 42, torch.device("cpu"),
            )

        self.assertAlmostEqual(score, 0.2)

    def test_persistent_optuna_study_uses_new_name_for_normalized_objective(self):
        study = Mock()
        study.best_trial = SimpleNamespace(number=0, value=0.1, params={})
        optuna_cfg = {
            "n_trials": 1,
            "n_startup_trials": 1,
            "n_folds": 2,
            "epochs": 1,
            "patience": 1,
            "study_name": "M2_test_hpo",
            "storage": "sqlite:///unused-for-test.db",
        }
        with patch.object(m2.optuna, "create_study", return_value=study) as create_study, \
                redirect_stdout(io.StringIO()):
            m2.run_hpo(
                np.zeros((2, 1)), np.zeros((2, 1)), optuna_cfg,
                {"hidden_dims": [2], "dropout": 0.0}, {}, ["f"],
                42, torch.device("cpu"),
            )

        self.assertEqual(create_study.call_args.kwargs["study_name"], "M2_test_hpo_nmae_no_l1")

    def test_best_model_promotion_uses_nmae_and_saves_raw_test_mae(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run_best"
            overall_dir = root / "best_overall"
            run_dir.mkdir()
            overall_dir.mkdir()
            (overall_dir / "best_metrics.json").write_text(json.dumps({
                "mean_MAE": 0.1,
                "mean_NMAE": 0.3,
            }))
            final_metrics = metrics(100.0, 0.2, np.zeros((2, 2)))

            with redirect_stdout(io.StringIO()):
                updated = m2.save_models(
                    nn.Linear(1, 1), final_metrics, run_dir, overall_dir,
                    1, 2, ["pp"], ["f0", "f1"], {}, "ID",
                )

            self.assertTrue(updated)
            saved = json.loads((overall_dir / "best_metrics.json").read_text())
            self.assertEqual(saved["mean_MAE"], 100.0)
            self.assertEqual(saved["mean_NMAE"], 0.2)

    def test_legacy_best_without_nmae_is_rebased_on_normalized_metric(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run_best"
            overall_dir = root / "best_overall"
            run_dir.mkdir()
            overall_dir.mkdir()
            (overall_dir / "best_metrics.json").write_text(json.dumps({
                "mean_MAE": 0.1,
            }))
            final_metrics = metrics(100.0, 0.2, np.zeros((2, 2)))

            with redirect_stdout(io.StringIO()):
                updated = m2.save_models(
                    nn.Linear(1, 1), final_metrics, run_dir, overall_dir,
                    1, 2, ["pp"], ["f0", "f1"], {}, "ID",
                )

            self.assertTrue(updated)
            saved = json.loads((overall_dir / "best_metrics.json").read_text())
            self.assertEqual(saved["mean_MAE"], 100.0)
            self.assertEqual(saved["mean_NMAE"], 0.2)
