"""Synthetic split checks without dataset training or HPO searches."""
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import optuna
import torch
from sklearn.preprocessing import MinMaxScaler

from src import M2_PP_to_F as m2


class M2ProtectedGroupTests(unittest.TestCase):
    def setUp(self):
        self.groups = np.repeat(np.arange(12), 2)
        self.x = np.arange(24, dtype=np.float32)[:, None]
        self.y = self.x ** 2
        self.group_cfg = {"enabled": True, "protected_group_ids": [0, 11]}
        self.model_cfg = {"hidden_dims": [4], "dropout": 0.}
        self.train_cfg = {"k_folds": 3, "val_fraction": .25, "epochs": 1, "patience": 1}
        self.device = torch.device("cpu")

    def assert_training_only(self, train, *holdouts):
        self.assertTrue({0, 11}.issubset(set(self.groups[train])))
        for indices in holdouts:
            self.assertFalse({0, 11} & set(self.groups[indices]))

    def test_cv_and_hpo_use_only_training_protected_groups(self):
        observed = []
        original = m2.apply_scalers

        def record_scaling(x, *args):
            observed.append(x[:, 0].astype(int))
            return original(x, *args)

        with patch.object(m2, "apply_scalers", side_effect=record_scaling), \
                patch.object(m2, "train_fold", side_effect=lambda model, *args: (model, 0.)), \
                redirect_stdout(io.StringIO()):
            m2.run_cv(self.x, self.y, self.model_cfg, self.train_cfg, ["f"], self.device,
                      42, group_values=self.groups, group_cfg=self.group_cfg)
            trial = optuna.trial.FixedTrial({})
            m2.optuna_objective(
                trial, self.x, self.y,
                {"search_space": {}, "n_folds": 3, "epochs": 1, "patience": 1},
                self.model_cfg, self.train_cfg, ["f"], 42, self.device,
                group_values=self.groups, group_cfg=self.group_cfg)
        self.assertEqual(len(observed), 18)
        for offset in range(0, len(observed), 3):
            self.assert_training_only(*observed[offset:offset + 3])

    def test_final_training_fallback_preserves_protected_groups(self):
        observed = []
        original = m2.apply_scalers

        def record_scaling(x, *args):
            observed.append(x[:, 0].astype(int))
            return original(x, *args)

        with patch.object(m2, "apply_scalers", side_effect=record_scaling), \
                patch.object(m2, "train_fold", side_effect=lambda model, *args: (model, 0.)), \
                redirect_stdout(io.StringIO()):
            m2.final_train_m2(
                self.x, self.y, np.arange(24), np.array([], dtype=int),
                MinMaxScaler().fit(self.x), self.model_cfg, self.train_cfg, 42,
                self.device, group_values=self.groups, group_cfg=self.group_cfg)
        self.assertEqual(len(observed), 2)
        self.assert_training_only(*observed)

    def test_imported_final_validation_and_cv_holdouts_are_checked(self):
        train, val, holdout = np.arange(2, 20), np.arange(2), np.arange(20, 24)
        with self.assertRaisesRegex(ValueError, "protected group.*validation"):
            m2.final_train_m2(
                self.x, self.y, train, val, MinMaxScaler().fit(self.x[train]),
                self.model_cfg, self.train_cfg, 42, self.device,
                group_values=self.groups, group_cfg=self.group_cfg)
        for validation, test in ((val, holdout), (holdout, val)):
            splits = [{"tr_idx": train, "val_idx": validation, "test_idx": test}]
            with self.subTest(validation=validation), redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(ValueError, "protected group.*CV"):
                m2.run_cv(self.x, self.y, self.model_cfg, self.train_cfg, ["f"],
                          self.device, 42, precomputed_splits=splits,
                          group_values=self.groups,
                          group_cfg={"enabled": True, "protected_group_ids": [0]})


if __name__ == "__main__":
    unittest.main()
