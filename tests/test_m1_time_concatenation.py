import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import optuna
import torch
from sklearn.preprocessing import MinMaxScaler

from src import M1_PPPT_to_W as m1
from src import Fusion_M1M2_WPred as fusion


def config(count=3):
    return {"encoder_input_layout": "concatenate_time", "n_signals": count,
            "conv_channels": [1, 4], "conv_kernels": [3], "n_f_features": 5,
            "pp_hidden": [6], "pp_latent_dim": 2, "merge_hidden": [4], "dropout": 0.}


class M1TimeConcatenationTests(unittest.TestCase):
    def test_single_cnn_receives_signals_end_to_end_in_configured_order(self):
        for lengths in ([4], [2, 3], [2, 3, 5], [2, 3, 5, 7]):
            with self.subTest(lengths=lengths):
                model = m1.M1Model(2, config(len(lengths))).eval()
                signals = [torch.full((2, 1, length), float(i + 1)) for i, length in enumerate(lengths)]
                pp = torch.randn(2, 2)
                inputs = []
                hook = model.encoders[0].convs[0].register_forward_pre_hook(
                    lambda module, args: inputs.append(args[0].detach().clone()))
                prediction = model(signals, pp)
                hook.remove()
                self.assertEqual(len(model.encoders), 1)
                self.assertEqual(model.encoders[0].convs[0].in_channels, 1)
                self.assertEqual(inputs[0].shape, (2, 1, sum(lengths)))
                torch.testing.assert_close(inputs[0], torch.cat(signals, dim=-1))
                features = model.get_f(signals)
                self.assertEqual(features.shape, (2, 5))
                torch.testing.assert_close(features, model.encoders[0](torch.cat(signals, dim=-1)))
                torch.testing.assert_close(prediction, model(torch.cat(signals, dim=-1), pp))
                torch.testing.assert_close(prediction, model.merge(torch.cat([features, model.pp_mlp(pp)], dim=1)))
                self.assertEqual(model.merge.net[0].in_features, 7)

    def test_features_receive_gradients_from_every_sensor_segment(self):
        model = m1.M1Model(2, config()).eval()
        with torch.no_grad():
            model.encoders[0].convs[0].weight.fill_(1.)
            model.encoders[0].convs[0].bias.zero_()
            model.encoders[0].fc.weight.fill_(1.)
        signals = [torch.ones(2, 1, length, requires_grad=True) for length in (2, 3, 5)]
        model.encoders[0](model._concatenate_curves(signals)).sum().backward()
        self.assertTrue(all(signal.grad.abs().sum() > 0 for signal in signals))

    def test_wrong_signal_count_or_channel_layout_is_rejected(self):
        model = m1.M1Model(2, config()).eval()
        with self.assertRaisesRegex(ValueError, "Expected 3 sensor signals"):
            model.get_f([torch.ones(2, 1, 4)])
        with self.assertRaisesRegex(ValueError, "Each sensor signal"):
            model.get_f([torch.ones(2, 2, 4)] * 3)
        with self.assertRaisesRegex(ValueError, "Concatenated sensor input"):
            model.get_f(torch.ones(2, 3, 4))

    def test_hpo_samples_one_stack_and_total_feature_width(self):
        params = {"enc_n_conv_layers": 2, "enc_ch_0": 4, "enc_ch_1": 8,
                  "enc_k_0": 3, "enc_k_1": 5, "enc_pk_0": 2, "n_f_features": 6}
        trial = optuna.trial.FixedTrial(params)
        updates, _ = m1.suggest_hyperparams(trial,
            {"enc_n_conv_layers": [2, 2], "enc_channels": [4, 8], "enc_kernel_size": [3, 5],
             "enc_pool_kernel_size": [1, 2], "n_f_features": [5, 10]}, 2, config())
        self.assertNotIn("curve_encoders", updates)
        self.assertEqual(updates["conv_channels"], [1, 4, 8])
        self.assertEqual(updates["n_f_features"], 6)
        self.assertFalse(any(name.startswith(("enc_c0_", "enc_c1_", "enc_c2_")) for name in trial.params))
        model = m1.M1Model(2, {**config(), **updates})
        self.assertEqual(model.get_f([torch.ones(2, 1, length) for length in (4, 6, 8)]).shape, (2, 6))

    def test_feature_exports_keep_scaling_and_record_signal_boundaries(self):
        model = m1.M1Model(2, config()).eval()
        xpp = np.array([[0., 1.], [1., 2.]], dtype=np.float32)
        curves = [np.arange(2 * length, dtype=np.float32).reshape(2, length) + 10 * i
                  for i, length in enumerate((2, 3, 5))]
        scalers = (MinMaxScaler().fit(xpp), [{"min": float(c.min()), "max": float(c.max())} for c in curves])
        inputs = []
        hook = model.encoders[0].convs[0].register_forward_pre_hook(
            lambda module, args: inputs.append(args[0].detach().numpy()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sensors_features_f.csv"
            frame, summary = m1.extract_features(model, scalers, xpp, curves, np.array([10, 11]),
                path, "ID", torch.device("cpu"), ["p1", "p2", "T1"], config())
            self.assertEqual(frame.columns.tolist(), [f"f_concat_{i}" for i in range(5)])
            self.assertEqual(summary["input_signals"], ["p1", "p2", "T1"])
            self.assertEqual(summary["signal_lengths"], [2, 3, 5])
            self.assertEqual(summary["concatenated_length"], 10)
            self.assertEqual(summary["n_input_channels"], 1)
            self.assertEqual(summary["merge_head_input_width"], 7)
        hook.remove()
        expected = np.concatenate([(c - sc["min"]) / (sc["max"] - sc["min"] + 1e-8)
                                   for c, sc in zip(curves, scalers[1])], axis=1)[:, None, :]
        np.testing.assert_allclose(inputs[0], expected)

    def test_fusion_loads_new_checkpoint_with_identical_predictions(self):
        model = m1.M1Model(2, config()).eval()
        signals = [torch.randn(2, 1, length) for length in (4, 6, 8)]
        pp = torch.randn(2, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            torch.save(model.state_dict(), path / "best_model.pt")
            (path / "best_metrics.json").write_text(json.dumps({"model_cfg": config(), "MAE": 1., "R2": 0.}))
            restored, _, _ = fusion.load_m1(path, 2, torch.device("cpu"))
            torch.testing.assert_close(restored(signals, pp), model(signals, pp))
            self.assertEqual(fusion.get_m1_n_f_features(config(), restored, path), 5)

    def test_old_per_signal_checkpoints_still_use_their_saved_architecture(self):
        cfg = config()
        cfg.pop("encoder_input_layout")
        cfg.pop("n_signals")
        cfg["curve_encoders"] = [{"conv_channels": [1, 4], "conv_kernels": [3], "n_f_features": 2}
                                 for _ in range(3)]
        model = m1.M1Model(2, cfg).eval()
        restored = m1.M1Model(2, cfg).eval()
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        self.assertFalse(restored.concatenate_signals)
        self.assertEqual(len(restored.encoders), 3)
        signals, pp = [torch.randn(2, 1, length) for length in (4, 6, 8)], torch.randn(2, 2)
        torch.testing.assert_close(restored(signals, pp), model(signals, pp))


if __name__ == "__main__":
    unittest.main()
