import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import optuna
import torch
from sklearn.preprocessing import MinMaxScaler

from src import M1_PPPT_to_W as m1
from src import Reference_Models_W_Pred as reference
from src import Fusion_M1M2_WPred as fusion
from src.Utility.signal_channels import stack_signal_arrays


def config(channels=3):
    return {"conv_channels": [channels, 4], "conv_kernels": [3], "n_f_features": 5,
            "pp_hidden": [6], "pp_latent_dim": 2, "merge_hidden": [4], "dropout": 0.}


class MultichannelEncoderTests(unittest.TestCase):
    def test_m1_uses_one_encoder_with_all_signals_and_preserves_merge(self):
        for count in (1, 2, 3, 4):
            with self.subTest(count=count):
                model = m1.M1Model(2, config(count)).eval()
                self.assertEqual(len(model.encoders), 1)
                self.assertEqual(model.encoders[0].convs[0].in_channels, count)
                self.assertEqual(model.n_curves, count)
                x = torch.randn(3, count, 12)
                pp = torch.randn(3, 2)
                signals = list(x.split(1, dim=1))
                torch.testing.assert_close(model(x, pp), model(signals, pp))
                features = model.get_f(signals)
                self.assertEqual(features.shape, (3, 5))
                torch.testing.assert_close(features, model.encoders[0](x))
                torch.testing.assert_close(model(x, pp), model.merge(torch.cat([features, model.pp_mlp(pp)], dim=1)))
                self.assertEqual(model.merge.net[0].in_features, 5 + 2)

    def test_joint_features_depend_on_every_channel_and_channel_order(self):
        model = m1.M1Model(2, config()).eval()
        conv = model.encoders[0].convs[0]
        with torch.no_grad():
            conv.weight[:, 0].fill_(1.)
            conv.weight[:, 1].fill_(2.)
            conv.weight[:, 2].fill_(3.)
            conv.bias.zero_()
            model.encoders[0].fc.weight.fill_(1.)
        x = torch.ones(2, 3, 10, requires_grad=True)
        features = model._joint_features(x)
        features.sum().backward()
        self.assertTrue((x.grad.abs().sum(dim=(0, 2)) > 0).all())
        changed = x.detach().clone()
        changed[:, 0] *= 2
        self.assertFalse(torch.allclose(model.get_f(changed), model.get_f(changed[:, [2, 1, 0]])))

    def test_signal_lengths_and_channel_counts_are_validated(self):
        with self.assertRaisesRegex(ValueError, "same.*time-step"):
            stack_signal_arrays([np.zeros((2, 4)), np.zeros((2, 5))])
        model = m1.M1Model(2, config()).eval()
        with self.assertRaisesRegex(ValueError, "same.*time-step"):
            model.get_f([torch.zeros(2, 1, 8), torch.zeros(2, 1, 9), torch.zeros(2, 1, 8)])
        with self.assertRaisesRegex(ValueError, "Expected 3 signal channels"):
            model.get_f(torch.zeros(2, 2, 8))

    def test_m1_hpo_samples_one_shared_stack_with_fixed_input_channels(self):
        space = {"enc_n_conv_layers": [2, 2], "enc_channels": [4, 8],
                 "enc_kernel_size": [3, 5], "enc_pool_kernel_size": [1, 2], "n_f_features": [5, 10]}
        params = {"enc_n_conv_layers": 2, "enc_ch_0": 4, "enc_ch_1": 8,
                  "enc_k_0": 3, "enc_k_1": 5, "enc_pk_0": 2, "n_f_features": 6}
        for count in (1, 3, 4):
            trial = optuna.trial.FixedTrial(params)
            update, _ = m1.suggest_hyperparams(trial, space, 2, config(count))
            self.assertNotIn("curve_encoders", update)
            self.assertEqual(update["conv_channels"], [count, 4, 8])
            self.assertEqual(update["n_f_features"], 6)
            self.assertFalse(any(name.startswith(("enc_c0_", "enc_c1_", "enc_c2_")) for name in trial.params))
            model = m1.M1Model(2, {**config(count), **update})
            self.assertEqual(model.get_f(torch.ones(2, count, 16)).shape, (2, 6))

    def test_joint_feature_exports_describe_shared_features(self):
        model = m1.M1Model(2, config()).eval()
        xpp = np.array([[0., 1.], [1., 2.]], dtype=np.float32)
        curves = [np.arange(16, dtype=np.float32).reshape(2, 8) + i for i in range(3)]
        scalers = (MinMaxScaler().fit(xpp), [{"min": float(c.min()), "max": float(c.max())} for c in curves])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sensors_features_f.csv"
            frame, summary = m1.extract_features(model, scalers, xpp, curves, np.array([10, 11]),
                path, "ID", torch.device("cpu"), ["p1", "p2", "T1"], config())
            self.assertEqual(frame.columns.tolist(), [f"f_joint_{i}" for i in range(5)])
            self.assertEqual(summary["n_input_channels"], 3)
            self.assertEqual(summary["input_signals"], ["p1", "p2", "T1"])
            self.assertEqual(summary["curve_feature_total"], 5)
            self.assertEqual(summary["merge_head_input_width"], 7)

    def test_fusion_loads_shared_m1_checkpoint_without_changing_predictions(self):
        model = m1.M1Model(2, config()).eval()
        x, pp = torch.randn(2, 3, 12), torch.randn(2, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            torch.save(model.state_dict(), path / "best_model.pt")
            (path / "best_metrics.json").write_text(json.dumps({"model_cfg": config(), "MAE": 1., "R2": 0.}))
            restored, _, _ = fusion.load_m1(path, 2, torch.device("cpu"))
            torch.testing.assert_close(restored(list(x.split(1, dim=1)), pp), model(x, pp))
            self.assertEqual(fusion.get_m1_n_f_features(config(), restored, path), 5)

    def test_reference_hpo_and_input_preparation_use_signal_channels(self):
        x = np.arange(72, dtype=np.float32).reshape(2, 3, 12)
        torch.testing.assert_close(reference._enc_prepare(x), torch.tensor(x))
        update, _ = reference.suggest_encoder_params(optuna.trial.FixedTrial({}),
            {"enc_n_conv_layers": 1, "enc_channels": 4, "enc_kernel_size": 3}, n_input_channels=3)
        self.assertEqual(update["channels"], [3, 4])
        model = reference.EncoderModel(update["channels"], update["kernels"], [], [4], 0.)
        self.assertEqual(model(reference._enc_prepare(x)).shape, (2,))
        self.assertEqual(reference._enc_prepare(x[:, 0]).shape, (2, 1, 12))

    def test_reference_hpo_trains_the_multichannel_encoder_in_every_fold(self):
        x = np.random.default_rng(4).normal(size=(24, 3, 12)).astype(np.float32)
        y = np.linspace(1., 3., 24, dtype=np.float32)
        hpo = {"search_space": {"enc_n_conv_layers": 1, "enc_channels": 4,
                               "enc_kernel_size": 3, "head_n_layers": 1, "head_hidden_size": 4},
               "n_folds": 2, "epochs": 1, "patience": 1}
        training = {"batch_size": 4, "lr": .001, "weight_decay": 0., "l1_lambda": 0.,
                    "scheduler_patience": 1, "scheduler_factor": .5, "val_fraction": .25}
        with patch.object(reference, "EncoderModel", wraps=reference.EncoderModel) as build:
            result = reference._obj_encoder(optuna.trial.FixedTrial({}), x, y, hpo,
                training, 42, torch.device("cpu"), None)
        self.assertTrue(np.isfinite(result))
        self.assertEqual(build.call_count, 2)
        self.assertTrue(all(call.args[0][0] == 3 for call in build.call_args_list))


if __name__ == "__main__":
    unittest.main()
