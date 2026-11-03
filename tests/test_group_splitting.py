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
    validate_protected_excluded,
)


class ProtectedGroupSplitTests(unittest.TestCase):
    def setUp(self):
        self.values = np.repeat(np.arange(8), 2)
        self.indices = np.arange(len(self.values))
        self.config = {"enabled": True, "protected_group_ids": [0.0, "7"]}
        self.protected_rows = set(self.indices[np.isin(self.values, [0, 7])])

    def assert_training_only(self, train, held_out, scope):
        self.assertTrue(self.protected_rows.issubset(set(train)))
        self.assertFalse(self.protected_rows & set(held_out))
        self.assertEqual(set(train) | set(held_out), set(scope))
        validate_group_disjoint(train, held_out, group_values=self.values,
                                config=self.config)

    def test_protected_groups_stay_in_training_through_test_and_validation(self):
        for seed in range(10):
            dev, test = grouped_train_test_split(
                len(self.values), .25, seed, self.values, self.config)
            self.assert_training_only(dev, test, self.indices)
            train, val = grouped_train_val_split(
                dev, .25, seed, self.values, self.config)
            self.assert_training_only(train, val, dev)

    def test_nested_cv_keeps_protected_groups_in_every_training_split(self):
        folds = grouped_kfold(self.indices, 3, 42, self.values, self.config)
        tested = []
        for outer_train, holdout in folds:
            self.assert_training_only(outer_train, holdout, self.indices)
            train, val = grouped_train_val_split(
                outer_train, .25, 42, self.values, self.config)
            self.assert_training_only(train, val, outer_train)
            tested.extend(holdout)
        self.assertEqual(len(tested), len(set(tested)))
        self.assertEqual(set(tested), set(self.indices) - self.protected_rows)

    def test_disabled_grouping_does_not_protect_configured_ids(self):
        config = {**self.config, "enabled": False}
        folds = grouped_kfold(self.indices, 8, 42, self.values, config)
        self.assertEqual(set(np.concatenate([held_out for _, held_out in folds])),
                         set(self.indices))
        validate_protected_excluded(self.indices, self.values, config)

    def test_imported_holdouts_reject_protected_groups(self):
        for partition in ("validation", "CV validation", "CV holdout", "final test"):
            with self.subTest(partition=partition):
                with self.assertRaisesRegex(ValueError, partition):
                    validate_protected_excluded(self.indices, self.values,
                                                self.config, partition=partition)
                validate_protected_excluded(np.array([2, 3]), self.values,
                                            self.config, partition=partition)

    def test_exclusion_combines_with_edges(self):
        edges = np.repeat(np.array([3, 0, 2, 3, 4, 5, 6, 4]), 2)[:, None]
        edge_config = {"enabled": True}
        excluded = set(self.indices[np.isin(self.values, [0, 1, 6, 7])])
        for split in (grouped_train_val_split(
                self.indices, .25, 42, self.values, self.config,
                edge_values=edges, edge_config=edge_config), *grouped_kfold(
                self.indices, 4, 42, self.values, self.config,
                edge_values=edges, edge_config=edge_config)):
            train, held_out = split
            self.assertTrue(excluded.issubset(set(train)))
            self.assertFalse(excluded & set(held_out))

    def test_impossible_splits_raise_without_using_protected_groups(self):
        all_protected = {"enabled": True, "protected_group_ids": list(range(8))}
        with self.assertRaisesRegex(ValueError, "non-empty partitions"):
            grouped_train_val_split(self.indices, .25, 42, self.values,
                                    all_protected)
        with self.assertRaisesRegex(ValueError, "eligible groups"):
            grouped_kfold(self.indices, 7, 42, self.values, self.config)


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
