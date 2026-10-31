import unittest

import torch
from torch import nn

from src.M1_PPPT_to_W import (
    M1Model,
    apply_manual_pp_n_features,
    _resolve_pp_branch_dims,
    suggest_hyperparams,
)
from src.Fusion_M1M2_WPred import M1Model as FusionM1Model


class RecordingTrial:
    def __init__(self):
        self.calls = []

    def suggest_int(self, name, low, high, log=False):
        self.calls.append((name, low, high, log))
        return low


class M1PPLatentDimensionTests(unittest.TestCase):
    def _model_cfg(self, latent_dim=7):
        return {
            "curve_encoders": [{
                "conv_channels": [1, 2],
                "conv_kernels": [3],
                "enc_pool_kernels": [],
                "n_f_features": 3,
            }],
            "pp_hidden": [5],
            "pp_latent_dim": latent_dim,
            "merge_hidden": [4],
            "dropout": 0.0,
        }

    def test_fixed_latent_width_is_separate_from_hidden_dims(self):
        hidden, latent = _resolve_pp_branch_dims(self._model_cfg())
        self.assertEqual(hidden, [5])
        self.assertEqual(latent, 7)

        model = M1Model(n_pp=4, mcfg=self._model_cfg())
        fusion_model = FusionM1Model(n_pp=4, mcfg=self._model_cfg())
        pp_linears = [layer for layer in model.pp_mlp.net if isinstance(layer, nn.Linear)]
        fusion_pp_linears = [
            layer for layer in fusion_model.pp_mlp.net if isinstance(layer, nn.Linear)
        ]
        self.assertEqual([layer.out_features for layer in pp_linears], [5, 7])
        self.assertEqual(
            [layer.out_features for layer in fusion_pp_linears], [5, 7])
        self.assertEqual(model.merge.net[0].in_features, 3 + 7)
        self.assertEqual(fusion_model.merge.net[0].in_features, 3 + 7)

        prediction = model(
            [torch.randn(2, 1, 8)], torch.randn(2, 4))
        self.assertEqual(tuple(prediction.shape), (2,))

    def test_legacy_config_keeps_last_pp_hidden_as_latent_width(self):
        legacy_cfg = self._model_cfg()
        legacy_cfg.pop("pp_latent_dim")
        legacy_cfg["pp_hidden"] = [5, 7]
        self.assertEqual(_resolve_pp_branch_dims(legacy_cfg), ([5], 7))

        model = M1Model(n_pp=4, mcfg=legacy_cfg)
        fusion_model = FusionM1Model(n_pp=4, mcfg=legacy_cfg)
        model_linears = [layer for layer in model.pp_mlp.net
                 if isinstance(layer, nn.Linear)]
        fusion_linears = [layer for layer in fusion_model.pp_mlp.net
                  if isinstance(layer, nn.Linear)]
        self.assertEqual(model_linears[-1].out_features, 7)
        self.assertEqual(fusion_linears[-1].out_features, 7)

    def test_range_latent_dim_is_sampled_but_scalar_is_not(self):
        search_space = {
            "pp_n_layers": [2, 2],
            "pp_hidden_size": [4, 16],
            "pp_n_features": [6, 12],
        }
        range_trial = RecordingTrial()
        range_model_cfg = self._model_cfg()
        model_updates, _ = suggest_hyperparams(
            range_trial, search_space, 4, range_model_cfg)

        self.assertEqual(model_updates["pp_hidden"], [4, 4])
        self.assertEqual(model_updates["pp_latent_dim"], 6)
        self.assertIn(("pp_n_features", 6, 12, False), range_trial.calls)
        self.assertEqual(len([call for call in range_trial.calls
                             if call[0].startswith("pp_h_")]), 2)

        fixed_trial = RecordingTrial()
        fixed_model_cfg = self._model_cfg(9)
        fixed_search = {**search_space, "pp_n_features": 9}
        fixed_updates, _ = suggest_hyperparams(
            fixed_trial, fixed_search, 4, fixed_model_cfg)
        self.assertEqual(fixed_updates["pp_hidden"], [4, 4])
        self.assertEqual(fixed_updates["pp_latent_dim"], 9)
        self.assertFalse(any(call[0] == "pp_n_features" for call in fixed_trial.calls))

    def test_latent_range_is_sampled_with_fixed_hidden_layers(self):
        trial = RecordingTrial()
        model_cfg = self._model_cfg()
        updates, _ = suggest_hyperparams(
            trial, {"pp_n_features": [3, 5]}, 4, model_cfg)

        self.assertEqual(updates["pp_latent_dim"], 3)
        self.assertNotIn("pp_hidden", updates)
        self.assertEqual(trial.calls, [("pp_n_features", 3, 5, False)])

    def test_manual_mode_uses_fixed_search_space_value(self):
        model_cfg = self._model_cfg()
        model_cfg.pop("pp_latent_dim")
        resolved = apply_manual_pp_n_features(
            model_cfg, {"search_space": {"pp_n_features": 11}})
        self.assertEqual(resolved["pp_latent_dim"], 11)
        self.assertNotIn("pp_latent_dim", model_cfg)

    def test_manual_mode_rejects_search_range(self):
        with self.assertRaisesRegex(ValueError, "mode='optuna'"):
            apply_manual_pp_n_features(
                self._model_cfg(),
                {"search_space": {"pp_n_features": [4, 12]}},
            )

    def test_unresolved_range_is_rejected_by_model_constructor(self):
        with self.assertRaisesRegex(ValueError, "Use mode='optuna'"):
            M1Model(n_pp=4, mcfg=self._model_cfg([4, 10]))


if __name__ == "__main__":
    unittest.main()
