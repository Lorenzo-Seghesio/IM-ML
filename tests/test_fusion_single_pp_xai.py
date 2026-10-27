import unittest

import numpy as np
import torch
import torch.nn as nn

from src.Fusion_M1M2_WPred import (
    FlatFusionWrapper,
    SinglePPFusionWrapper,
    build_per_sample_df,
    build_single_pp_summary,
)


class _LinearTwoPathModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("indirect_weights", torch.tensor([1.5, -2.0]))
        self.register_buffer("direct_weights", torch.tensor([-0.5, 3.0]))

    def forward(self, indirect, direct):
        return (indirect * self.indirect_weights
                + direct * self.direct_weights).sum(dim=1)


class SinglePPFusionWrapperTests(unittest.TestCase):
    def setUp(self):
        self.model = _LinearTwoPathModel()
        self.n_pp = 2

    def test_single_input_predictions_match_equal_split_inputs(self):
        values = torch.tensor([[1.0, 2.0], [-3.0, 0.5]])
        flat_values = torch.cat([values, values], dim=1)
        tied = SinglePPFusionWrapper(self.model, self.n_pp)
        split = FlatFusionWrapper(self.model, self.n_pp)

        torch.testing.assert_close(tied(values), split(flat_values))

    def test_single_input_gradient_combines_both_path_effects(self):
        values = torch.tensor([[1.0, 2.0]], requires_grad=True)
        tied = SinglePPFusionWrapper(self.model, self.n_pp)

        gradient, = torch.autograd.grad(tied(values).sum(), values)
        expected = self.model.indirect_weights + self.model.direct_weights
        torch.testing.assert_close(gradient[0], expected)

    def test_single_input_rejects_wrong_feature_width(self):
        tied = SinglePPFusionWrapper(self.model, self.n_pp)
        with self.assertRaisesRegex(ValueError, "Expected input"):
            tied(torch.zeros(3, self.n_pp + 1))

    def test_single_pp_summary_and_sample_columns_are_aligned(self):
        pp_cols = ["pressure", "temperature"]
        shap_values = np.array([[1.0, -2.0], [-3.0, 4.0]])
        eg_values = np.array([[-2.0, 1.0], [2.0, 3.0]])
        part_ids = np.array([101, 205])

        summary = build_single_pp_summary(pp_cols, shap_values, eg_values)
        samples = build_per_sample_df(shap_values, part_ids, pp_cols)

        self.assertEqual(summary.index.tolist(), pp_cols)
        self.assertEqual(
            summary.columns.tolist(),
            ["shap_single_pp_mean", "shap_single_pp_mean_abs",
             "eg_single_pp_mean", "eg_single_pp_mean_abs"],
        )
        self.assertEqual(summary.loc["pressure", "shap_single_pp_mean_abs"], 2.0)
        self.assertEqual(samples.columns.tolist(), ["ID_Part", *pp_cols])
        self.assertEqual(samples["ID_Part"].tolist(), part_ids.tolist())
        np.testing.assert_array_equal(samples[pp_cols].to_numpy(), shap_values)

    def test_single_pp_summary_rejects_split_width_values(self):
        with self.assertRaisesRegex(ValueError, "single-PP values"):
            build_single_pp_summary(
                ["pressure", "temperature"], np.zeros((3, 4)), None)


if __name__ == "__main__":
    unittest.main()