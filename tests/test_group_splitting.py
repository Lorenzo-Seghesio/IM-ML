import unittest

import numpy as np

from src.Utility.group_splitting import (
    edge_group_details,
    grouped_kfold,
    grouped_train_test_split,
    grouped_train_val_split,
    make_group_keys,
    group_split_required,
    validate_group_disjoint,
)


class EdgeGroupSplitTests(unittest.TestCase):
    def setUp(self):
        # Eight indivisible groups with two rows per group. Groups 0, 2, 4, 5,
        # and 7 touch at least one parameter minimum or maximum.
        per_group = np.array([
            [0, 2, 220, 10],
            [1, 2, 230, 20],
            [2, 2, 240, 30],
            [3, 2, 230, 40],
            [4, 2, 220, 50],
            [2, 0, 230, 30],
            [2, 1, 230, 30],
            [2, 3, 230, 30],
        ], dtype=float)
        self.edge_values = np.repeat(per_group, 2, axis=0)
        self.group_values = np.repeat(np.arange(len(per_group)), 2)
        self.group_cfg = {"enabled": True, "protected_group_ids": []}
        self.edge_cfg = {
            "enabled": True,
            "columns": ["pressure", "time", "temperature", "speed"],
            "exclude_from_test": True,
            "exclude_from_validation": True,
        }
        self.indices = np.arange(len(self.group_values), dtype=np.intp)
        self.edge_keys = edge_group_details(
            self.edge_values, self.group_values, group_config=self.group_cfg,
            edge_config=self.edge_cfg)["edge_group_keys"]

    def assert_partition_excludes_edges(self, partition, scope=None):
        details = edge_group_details(
            self.edge_values, self.group_values, scope,
            self.group_cfg, self.edge_cfg)
        keys = make_group_keys(self.group_values, self.group_cfg)
        self.assertFalse(set(keys[partition].tolist()) & details["edge_group_keys"])

    def test_initial_test_split_keeps_edge_groups_in_development(self):
        dev, test = grouped_train_test_split(
            len(self.indices), 0.25, 13, self.group_values, self.group_cfg,
            edge_values=self.edge_values, edge_config=self.edge_cfg)
        self.assert_partition_excludes_edges(test)
        validate_group_disjoint(dev, test, group_values=self.group_values,
                                config=self.group_cfg)
        keys = make_group_keys(self.group_values, self.group_cfg)
        self.assertTrue(self.edge_keys.issubset(set(keys[dev].tolist())))

    def test_inner_validation_keeps_edge_groups_in_training(self):
        train, val = grouped_train_val_split(
            self.indices, 0.25, 19, self.group_values, self.group_cfg,
            edge_values=self.edge_values, edge_config=self.edge_cfg)
        self.assert_partition_excludes_edges(val, self.indices)
        validate_group_disjoint(train, val, group_values=self.group_values,
                                config=self.group_cfg)
        keys = make_group_keys(self.group_values, self.group_cfg)
        self.assertTrue(self.edge_keys.issubset(set(keys[train].tolist())))

    def test_every_cv_validation_fold_keeps_edge_groups_in_training(self):
        folds = grouped_kfold(
            self.indices, 3, 23, self.group_values, self.group_cfg,
            edge_values=self.edge_values, edge_config=self.edge_cfg)
        tested_groups = set()
        keys = make_group_keys(self.group_values, self.group_cfg)
        for train, validation in folds:
            self.assert_partition_excludes_edges(validation, self.indices)
            self.assertTrue(self.edge_keys.issubset(set(keys[train].tolist())))
            tested_groups.update(keys[validation].tolist())
            validate_group_disjoint(train, validation, group_values=self.group_values,
                                    config=self.group_cfg)
        self.assertEqual(tested_groups & self.edge_keys, set())

    def test_nested_cv_validation_uses_outer_training_extrema(self):
        folds = grouped_kfold(
            self.indices, 3, 29, self.group_values, self.group_cfg,
            edge_values=self.edge_values, edge_config=self.edge_cfg)
        for outer_train, outer_test in folds:
            inner_train, validation = grouped_train_val_split(
                outer_train, 0.25, 37, self.group_values, self.group_cfg,
                edge_values=self.edge_values, edge_config=self.edge_cfg)
            inner_edges = edge_group_details(
                self.edge_values, self.group_values, outer_train,
                self.group_cfg, self.edge_cfg)["edge_group_keys"]
            keys = make_group_keys(self.group_values, self.group_cfg)
            self.assertFalse(set(keys[validation].tolist()) & inner_edges)
            self.assertTrue(inner_edges.issubset(set(keys[inner_train].tolist())))
            self.assertFalse(set(keys[outer_test].tolist()) & self.edge_keys)

    def test_extrema_are_recomputed_for_the_requested_subset(self):
        subset = np.arange(2, len(self.indices) - 2, dtype=np.intp)
        details = edge_group_details(
            self.edge_values, self.group_values, subset,
            self.group_cfg, self.edge_cfg)
        self.assertEqual(details["extrema"]["pressure"]["min"], 1.0)
        self.assertEqual(details["extrema"]["pressure"]["max"], 4.0)
        self.assertEqual(details["scope_size"], len(subset))

    def test_disabled_edge_policy_leaves_all_groups_eligible_for_cv(self):
        disabled = {**self.edge_cfg, "enabled": False}
        folds = grouped_kfold(
            self.indices, 8, 31, self.group_values, self.group_cfg,
            edge_values=self.edge_values, edge_config=disabled)
        tested_groups = {
            key for _, validation in folds
            for key in make_group_keys(self.group_values, self.group_cfg)[validation]
        }
        self.assertEqual(tested_groups, set(make_group_keys(self.group_values)))

    def test_edge_policy_requires_grouped_split_when_batch_grouping_is_off(self):
        ordinary_group_cfg = {"enabled": False, "protected_group_ids": []}
        self.assertTrue(group_split_required(ordinary_group_cfg, self.edge_cfg, "test"))
        dev, test = grouped_train_test_split(
            len(self.indices), 0.25, 41, self.group_values, ordinary_group_cfg,
            edge_values=self.edge_values, edge_config=self.edge_cfg)
        keys = make_group_keys(self.group_values, ordinary_group_cfg)
        self.assertFalse(set(keys[test].tolist()) & self.edge_keys)
        validate_group_disjoint(dev, test, group_values=self.group_values,
                                config=ordinary_group_cfg)

    def test_non_numeric_nominal_values_are_rejected(self):
        bad_values = self.edge_values.astype(object)
        bad_values[0, 0] = "not-numeric"
        with self.assertRaisesRegex(ValueError, "numeric values"):
            edge_group_details(
                bad_values, self.group_values, group_config=self.group_cfg,
                edge_config=self.edge_cfg)


if __name__ == "__main__":
    unittest.main()
