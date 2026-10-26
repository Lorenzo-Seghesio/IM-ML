import unittest

import numpy as np

from src.M1_PPPT_to_W import _dev_y_outlier_keep_mask


class DevTargetOutlierFilterTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([1, 2, 3, 4, 100, 1000], dtype=np.float32)
        self.part_ids = np.array(
            ["d1", "d2", "d3", "d4", "dev_outlier", "test_outlier"])
        self.dev_idx = np.arange(5, dtype=np.intp)
        self.prep_cfg = {"iqr_multiplier": 1.5}

    def test_enabled_removes_only_development_outliers(self):
        keep_mask, removed_ids = _dev_y_outlier_keep_mask(
            self.y, self.part_ids, self.dev_idx,
            {**self.prep_cfg, "remove_target_outliers": True})

        np.testing.assert_array_equal(
            keep_mask, [True, True, True, True, False, True])
        self.assertEqual(removed_ids, ["dev_outlier"])

    def test_disabled_or_missing_flag_keeps_all_rows(self):
        for cfg in (
            {**self.prep_cfg, "remove_target_outliers": False},
            self.prep_cfg,
        ):
            with self.subTest(cfg=cfg):
                keep_mask, removed_ids = _dev_y_outlier_keep_mask(
                    self.y, self.part_ids, self.dev_idx, cfg)
                self.assertTrue(keep_mask.all())
                self.assertEqual(removed_ids, [])


if __name__ == "__main__":
    unittest.main()
