"""Synthetic checks for independent HPO and final validation fractions."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import optuna
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from src import M1_PPPT_to_W as m1, M2_PP_to_F as m2, Reference_Models_W_Pred as ref
from src.Utility.split_config import split_fraction, split_ids_changed, validation_fraction


class TinyModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, *inputs):
        return self.linear(inputs[-1]).squeeze(-1)


class SplitConfigTests(unittest.TestCase):
    def test_fraction_validation_and_legacy_fallback(self):
        cfg = {"val_fraction": .1}
        self.assertEqual(validation_fraction(cfg, {"val_fraction": .3}), .3)
        self.assertEqual(validation_fraction(cfg, {}), .1)
        self.assertEqual(validation_fraction(cfg), .1)
        self.assertEqual(cfg, {"val_fraction": .1})
        for value in (0, 1, -1, 20, True, None, "0.2", float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "fraction"):
                split_fraction(value, "test_fraction")

    def test_test_fraction_applies_to_all_split_modes_and_models(self):
        for module in (m1, m2, ref):
            for mode in ("plain", "stratified", "grouped"):
                kwargs = {}
                if mode == "stratified":
                    kwargs["strat_labels"] = np.tile([0, 1], 30)
                elif mode == "grouped":
                    kwargs.update(group_values=np.repeat(np.arange(30), 2),
                                  group_cfg={"enabled": True})
                for fraction, expected in ((.1, 6), (.3, 18)):
                    with self.subTest(module=module.__name__, mode=mode, fraction=fraction):
                        dev, test = module._initial_split(60, fraction, 42, **kwargs)
                        self.assertEqual(len(test), expected)
                        self.assertEqual(len(dev) + len(test), 60)
                        self.assertFalse(set(dev) & set(test))

    def test_changed_partition_membership_requires_checkpoint_refresh(self):
        previous = {"train_part_ids": [1, 2], "val_part_ids": [3], "test_part_ids": [4]}
        reordered = {**previous, "train_part_ids": [2, 1]}
        self.assertFalse(split_ids_changed(previous, reordered))
        for current in (
            {**previous, "train_part_ids": [1], "val_part_ids": [2, 3]},
            {**previous, "train_part_ids": [1], "test_part_ids": [2, 4]},
        ):
            self.assertTrue(split_ids_changed(previous, current))

    def test_beta_sweep_promotes_changed_split_even_with_worse_validation_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "best_overall"
            destination.mkdir()
            previous = {"train_part_ids": [1, 2], "val_part_ids": [3], "test_part_ids": [4]}
            current = {**previous, "train_part_ids": [1], "val_part_ids": [2, 3]}
            (destination / "data_processing.json").write_text(json.dumps(previous))
            (destination / "best_metrics.json").write_text(json.dumps({"validation_MAE": .01}))
            cfg = {"mode": "manual", "training": {"test_point_training": {"enabled": True}},
                   "output": {"output_dir": str(root / "outputs"),
                              "best_overall_dir": str(destination)}}

            def experiment(*args):
                run_dir = root / "run"
                run_dir.mkdir()
                (run_dir / "data_processing.json").write_text(json.dumps(current))
                (run_dir / "best_model.pt").write_bytes(b"matching checkpoint")
                return {"run_dir": str(run_dir), "validation_mae_g": .5,
                        "within_test_point_beta": 0.}

            with patch.object(m1, "run_experiment", side_effect=experiment), \
                    patch.object(m1, "save_beta_mae_comparison", return_value=[]), \
                    redirect_stdout(io.StringIO()):
                m1.run_beta_experiments(cfg, "DOE1", "unused", [0])
            self.assertEqual(json.loads((destination / "data_processing.json").read_text()), current)
            self.assertEqual((destination / "best_model.pt").read_bytes(), b"matching checkpoint")


class PhaseSplitTests(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(60, dtype=np.float32)[:, None]
        self.y = self.x[:, 0]
        self.cfg = {"val_fraction": .1, "k_folds": 3, "epochs": 1, "patience": 1}
        self.hpo = {"val_fraction": .3, "n_folds": 3, "epochs": 1,
                    "patience": 1, "search_space": {}}
        self.device = torch.device("cpu")

    def test_m1_and_m2_hpo_cv_and_final_training_use_separate_fractions(self):
        for module, constructor in ((m1, "M1Model"), (m2, "M2Model")):
            for mode in ("plain", "stratified", "grouped"):
                with self.subTest(model=module.__name__, mode=mode):
                    observed = []
                    kwargs = {}
                    if mode == "stratified":
                        kwargs["strat_labels"] = np.tile([0, 1], 30)
                    elif mode == "grouped":
                        kwargs.update(group_values=np.repeat(np.arange(30), 2),
                                      group_cfg={"enabled": True})

                    def train(model, *args):
                        train_y, val_y = (args[2], args[5]) if module is m1 else (args[1], args[3])
                        observed.append((len(train_y), len(val_y)))
                        model.m1_training_info = {}
                        return model, 0.

                    def evaluate(model, *args):
                        target = args[2] if module is m1 else args[1]
                        metrics = {key: 0. for key in ("MAE", "RMSE", "R2", "MSE")}
                        if module is m2:
                            metrics = {"mean_" + key: value for key, value in metrics.items()}
                            metrics["mean_NMAE"] = 0.
                            metrics["per_target"] = {}
                        return {**metrics, "pred": np.zeros_like(target)}

                    with patch.object(module, constructor, TinyModel), \
                            patch.object(module, "train_fold", side_effect=train), \
                            patch.object(module, "evaluate_model", side_effect=evaluate), \
                            redirect_stdout(io.StringIO()):
                        trial = optuna.trial.FixedTrial({})
                        if module is m1:
                            module.optuna_objective(trial, self.x, [], self.y, self.hpo, {},
                                                    self.cfg, 42, self.device, **kwargs)
                            module.run_cv(self.x, [], self.y, {}, self.cfg, self.device, 42, **kwargs)
                            module.final_train_m1(self.x, [], self.y, np.arange(60), {},
                                                  self.cfg, 42, self.device, **kwargs)
                        else:
                            model_cfg = {"hidden_dims": [2], "dropout": 0.}
                            module.optuna_objective(trial, self.x, self.x, self.hpo, model_cfg,
                                                    self.cfg, ["f"], 42, self.device, **kwargs)
                            module.run_cv(self.x, self.x, model_cfg, self.cfg, ["f"],
                                          self.device, 42, **kwargs)
                            # M2's fallback final split supports plain or grouped inputs.
                            kwargs.pop("strat_labels", None)
                            module.final_train_m2(self.x, self.x, np.arange(60), np.array([], dtype=int),
                                MinMaxScaler().fit(self.x), model_cfg, self.cfg, 42, self.device, **kwargs)
                    self.assertEqual(observed[:3], [(28, 12)] * 3)
                    self.assertEqual(observed[3:6], [(36, 4)] * 3)
                    self.assertEqual(observed[6:], [(54, 6)])
                    self.assertEqual(self.cfg["val_fraction"], .1)

    def test_reference_nn_and_custom_gbdt_hpo_override(self):
        groups = np.repeat(np.arange(30), 2)
        group_cfg = {"enabled": True}
        for kind in ("encoder", "mlp", "custom_gbdt"):
            with self.subTest(kind=kind):
                observed = []
                original = ref._inner_split

                def split(*args, **kwargs):
                    train, val = original(*args, **kwargs)
                    observed.append((len(train), len(val)))
                    return train, val

                def train(model, *args):
                    model.ref_training_info = {}
                    return model, 0.

                with patch.object(ref, "EncoderModel", TinyModel), \
                        patch.object(ref, "MLPModel", TinyModel), \
                        patch.object(ref, "_inner_split", side_effect=split), \
                        patch.object(ref, "train_fold_nn", side_effect=train), \
                        redirect_stdout(io.StringIO()):
                    trial = optuna.trial.FixedTrial({})
                    if kind == "custom_gbdt":
                        cfg = {**self.cfg, "batch_size": 4,
                               "test_point_training": {"enabled": True,
                                                       "test_points_per_mini_batch": 2,
                                                       "parts_per_test_point": 2}}
                        ref._obj_custom_gbdt(trial, self.x, self.y, self.hpo, cfg, {}, 42,
                                             group_values=groups, group_cfg=group_cfg)
                    else:
                        getattr(ref, "_obj_" + kind)(trial, self.x, self.y, self.hpo,
                            self.cfg, 42, self.device, None, groups, group_cfg)
                    ref._inner_split(np.arange(60), self.cfg, 42, 0, None, groups, group_cfg)
                self.assertEqual(observed[:3], [(28, 12)] * 3)
                self.assertEqual(observed[3:], [(54, 6)])


if __name__ == "__main__":
    unittest.main()
