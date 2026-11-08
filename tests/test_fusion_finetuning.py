import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from src.Fusion_M1M2_WPred import (
    FusionModel, M2Model, PPMLP, MergeHead, TorchMinMaxScaler,
    finetune_m2, finetuning_split, fusion_mae_loss, resolve_finetuning, parse_args,
    fusion_loss_normalization, train_alpha_sweep,
)


class FusionFinetuningTests(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(4)
        scaler = TorchMinMaxScaler(MinMaxScaler().fit([[0., 0.], [1., 1.]]))
        model = FusionModel(
            M2Model(2, 2, [4], 0.2), PPMLP(2, [3], 0.5),
            MergeHead(6, [4], 0.5), copy.deepcopy(scaler),
            copy.deepcopy(scaler), scaler, 3, [2, 0])
        model.eval()
        return model

    def test_loss_matches_per_feature_average_and_endpoints(self):
        pred_w, true_w = torch.tensor([2., 6.]), torch.tensor([1., 2.])
        pred_f = torch.tensor([[0., 4.], [2., 8.]])
        true_f = torch.zeros_like(pred_f)
        for alpha in (0., 0.25, 1.):
            loss, w, f, nw, nf = fusion_mae_loss(
                pred_w, true_w, pred_f, true_f, alpha, 2., torch.tensor([1., 3.]))
            self.assertAlmostEqual(w.item(), 2.5)
            self.assertAlmostEqual(f.item(), 3.5)
            self.assertAlmostEqual(nw.item(), 1.25)
            self.assertAlmostEqual(nf.item(), 1.5)
            self.assertAlmostEqual(loss.item(), alpha * 1.25 + (1-alpha) * 1.5)

    def test_raw_loss_matches_original_maes_and_endpoints(self):
        pred_w, true_w = torch.tensor([2., 6.]), torch.tensor([1., 2.])
        pred_f = torch.tensor([[0., 4.], [2., 8.]])
        for alpha in (0., .25, 1.):
            loss, mae_w, mae_f = fusion_mae_loss(
                pred_w, true_w, pred_f, torch.zeros_like(pred_f), alpha)
            self.assertAlmostEqual(mae_w.item(), 2.5)
            self.assertAlmostEqual(mae_f.item(), 3.5)
            self.assertAlmostEqual(loss.item(), alpha * 2.5 + (1-alpha) * 3.5)

    def test_normalize_loss_config_and_cli_precedence(self):
        _, settings = resolve_finetuning({}, SimpleNamespace())
        self.assertTrue(settings["normalize_loss"])
        for configured in (False, True):
            cfg = {"finetuning": {"normalize_loss": configured}}
            for flag, expected in ((None, configured), ("--normalize-loss", True),
                                   ("--no-normalize-loss", False)):
                argv = ["fusion", "--material", "DOE1"] + ([flag] if flag else [])
                with self.subTest(configured=configured, flag=flag), patch("sys.argv", argv):
                    _, settings = resolve_finetuning(cfg, parse_args())
                    self.assertIs(settings["normalize_loss"], expected)
                self.assertIs(cfg["finetuning"]["normalize_loss"], configured)
        for invalid in ("false", 0, 1, None):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "normalize_loss"):
                resolve_finetuning({"finetuning": {"normalize_loss": invalid}}, SimpleNamespace())

    def test_disabled_normalization_skips_factors_and_uses_raw_loss_for_train_and_val(self):
        model = self.make_model()
        X = np.random.default_rng(3).uniform(size=(12, 2)).astype(np.float32)
        # Zero variances are valid for the original objective.
        y, features = np.zeros(12), np.zeros((12, 2))
        settings = {"epochs": 2, "lr": .001, "batch_size": 4, "patience": 2,
                    "normalize_loss": False}
        with tempfile.TemporaryDirectory() as tmp, patch(
                "src.Fusion_M1M2_WPred.fusion_loss_normalization",
                side_effect=AssertionError("Raw loss must not compute normalization factors")):
            run_dir = Path(tmp)
            train_alpha_sweep(model, X, y, features, np.arange(8), np.arange(8, 12),
                [.25], settings, 42, torch.device("cpu"), run_dir, np.arange(12), plots=False)
            self.assertFalse((run_dir / "loss_normalization.json").exists())
            folder = run_dir / "alpha_0.25"
            info = json.loads((folder / "finetuning_info.json").read_text())
            self.assertIs(info["normalize_loss"], False)
            self.assertIsNone(info["normalization"])
            self.assertEqual(info["loss"], "alpha * MAE_w + (1-alpha) * mean_i(MAE_f_i)")
            history = pd.read_csv(folder / "training_history.csv")
            self.assertFalse(any("normalized" in name for name in history.columns))
            for partition in ("train", "val"):
                np.testing.assert_allclose(history[f"{partition}_loss"],
                    .25 * history[f"{partition}_mae_w"] + .75 * history[f"{partition}_mae_f"], rtol=1e-6)
            report = json.loads((folder / "validation_metrics.json").read_text())
            self.assertNotIn("normalized_MAE", report)
            self.assertNotIn("normalized_feature_MAE", report)
            metrics = info["val_metrics"]
            self.assertAlmostEqual(info["best_val_loss"], .25 * metrics["mae_w"] + .75 * metrics["mae_f"])

    def test_normalization_uses_only_training_population_std_and_relative_floor(self):
        y = np.array([0., 2., 1000., np.nan])
        features = np.array([[0., 0., 0., 0.], [0., .002, 4., 8.],
                             [1e6, 1e6, 1e6, 1e6], [np.nan] * 4])
        factors = fusion_loss_normalization(y, features, np.arange(2))
        self.assertEqual(factors["std_w"], 1.)
        np.testing.assert_allclose(factors["std_f"], [0., .001, 2., 4.])
        self.assertAlmostEqual(factors["median_std_f"], 1.0005, places=6)
        np.testing.assert_allclose(factors["feature_scale"], [.010005, .010005, 2., 4.])
        self.assertFalse(factors["unbiased"])
        loss = fusion_mae_loss(torch.ones(2), torch.zeros(2), torch.ones(2, 4),
            torch.zeros(2, 4), .5, factors["std_w"], torch.tensor(factors["feature_scale"]))[0]
        self.assertTrue(torch.isfinite(loss))

    def test_degenerate_normalization_is_rejected(self):
        train = np.arange(3)
        with self.assertRaisesRegex(ValueError, "std_w"):
            fusion_loss_normalization(np.ones(3), np.arange(6).reshape(3, 2), train)
        with self.assertRaisesRegex(ValueError, r"median\(std_f\)"):
            fusion_loss_normalization(np.arange(3), np.zeros((3, 2)), train)

    def test_eps_rel_default_config_and_cli_override(self):
        _, settings = resolve_finetuning({}, SimpleNamespace())
        self.assertEqual(settings["eps_rel"], .01)
        config = {"finetuning": {"eps_rel": .02}}
        _, settings = resolve_finetuning(config, SimpleNamespace())
        self.assertEqual(settings["eps_rel"], .02)
        with patch("sys.argv", ["fusion", "--material", "DOE1", "--eps-rel", ".03"]):
            _, settings = resolve_finetuning(config, parse_args())
        self.assertEqual(settings["eps_rel"], .03)
        for invalid in (0., -.01, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "eps_rel"):
                resolve_finetuning({"finetuning": {"eps_rel": invalid}}, SimpleNamespace())

    def test_sweep_fits_factors_once_and_logs_fixed_normalized_metrics(self):
        from src import Fusion_M1M2_WPred as module
        model = self.make_model()
        X = np.random.default_rng(3).uniform(size=(12, 2)).astype(np.float32)
        y, features = np.arange(12, dtype=np.float32), X.copy()
        features[8:] *= 1000
        settings = {"epochs": 2, "lr": .001, "batch_size": 4, "patience": 2}
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                module, "fusion_loss_normalization", wraps=fusion_loss_normalization) as fit:
            run_dir = Path(tmp)
            train_alpha_sweep(model, X, y, features, np.arange(8), np.arange(8, 12),
                [0., .5], settings, 42, torch.device("cpu"), run_dir, np.arange(12), plots=False)
            self.assertEqual(fit.call_count, 1)
            factors = json.loads((run_dir / "loss_normalization.json").read_text())
            self.assertAlmostEqual(factors["std_w"], float(y[:8].std()), places=6)
            np.testing.assert_allclose(factors["std_f"], features[:8].std(axis=0))
            for alpha in (0., .5):
                folder = run_dir / f"alpha_{alpha}"
                info = json.loads((folder / "finetuning_info.json").read_text())
                self.assertEqual(info["normalization"], factors)
                history = pd.read_csv(folder / "training_history.csv")
                for partition in ("train", "val"):
                    np.testing.assert_allclose(history[f"{partition}_normalized_mae_w"],
                        history[f"{partition}_mae_w"] / factors["std_w"], rtol=1e-6)
                    np.testing.assert_allclose(history[f"{partition}_loss"],
                        alpha * history[f"{partition}_normalized_mae_w"]
                        + (1-alpha) * history[f"{partition}_normalized_mae_f"], rtol=1e-6)
                with torch.no_grad():
                    model.load_state_dict(torch.load(folder / "fusion_model.pt", weights_only=True))
                    _, pred = model(torch.tensor(X[8:]), torch.tensor(X[8:]), return_features=True)
                expected = (np.abs(pred.numpy() - features[8:]).mean(axis=0)
                            / np.asarray(factors["feature_scale"])).mean()
                self.assertAlmostEqual(info["val_metrics"]["normalized_mae_f"], expected, places=3)

    def test_embedding_preserves_permuted_positions(self):
        model = self.make_model()
        features = torch.tensor([[5., 7.]], requires_grad=True)
        embedded = model.embed_features(features)
        torch.testing.assert_close(embedded, torch.tensor([[7., 0., 5.]]))
        embedded.sum().backward()
        torch.testing.assert_close(features.grad, torch.ones_like(features))

    def test_scalar_vector_and_invalid_alpha_configuration(self):
        alphas, _ = resolve_finetuning({"finetuning": {"alpha": 0.25}}, SimpleNamespace())
        self.assertEqual(alphas, [0.25])
        alphas, _ = resolve_finetuning({}, SimpleNamespace(alpha=[0, 1]))
        self.assertEqual(alphas, [0., 1.])
        for invalid in ([], [-0.1], [float("nan")], [0.1, 0.1], [1.1]):
            with self.assertRaises(ValueError):
                resolve_finetuning({}, SimpleNamespace(alpha=invalid))

    def test_validation_fraction_config_cli_precedence_and_inheritance(self):
        _, settings = resolve_finetuning({}, SimpleNamespace())
        self.assertIsNone(settings["val_fraction"])
        cfg = {"finetuning": {"val_fraction": .25}}
        with patch("sys.argv", ["fusion", "--material", "DOE1"]):
            _, settings = resolve_finetuning(cfg, parse_args())
        self.assertEqual(settings["val_fraction"], .25)
        with patch("sys.argv", ["fusion", "--material", "DOE1", "--val-fraction", ".4"]):
            _, settings = resolve_finetuning(cfg, parse_args())
        self.assertEqual(settings["val_fraction"], .4)
        self.assertEqual(cfg["finetuning"]["val_fraction"], .25)
        for value in (0, 1, -1, 20, True, "0.2", float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "val_fraction"):
                resolve_finetuning({"finetuning": {"val_fraction": value}}, SimpleNamespace())

    def test_split_reuses_ids_and_rejects_test_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "data_processing.json"
            path.write_text(json.dumps({"train_part_ids": [10, 30], "val_part_ids": [20]}))
            args = (directory, np.array([30, 20, 10, 40]), np.array([0, 1, 2]),
                    np.array([3]), 42, None)
            train, val, _ = finetuning_split(*args)
            np.testing.assert_array_equal(train, [2, 0])
            np.testing.assert_array_equal(val, [1])
            path.write_text(json.dumps({"train_part_ids": [10, 40], "val_part_ids": [20]}))
            with self.assertRaises(ValueError):
                finetuning_split(*args)

    def test_requested_fraction_overrides_saved_validation_without_changing_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            dp = {"train_part_ids": list(range(36)), "val_part_ids": list(range(36, 40))}
            path = directory / "data_processing.json"
            path.write_text(json.dumps(dp))
            for fraction, expected in ((.25, 10), (.5, 20)):
                args = (directory, np.arange(50), np.arange(40), np.arange(40, 50), 42, fraction)
                train, val, source = finetuning_split(*args)
                train_again, val_again, _ = finetuning_split(*args)
                np.testing.assert_array_equal(train, train_again)
                np.testing.assert_array_equal(val, val_again)
                self.assertEqual(len(val), expected)
                self.assertEqual(set(train) | set(val), set(range(40)))
                self.assertFalse(set(train) & set(val))
                self.assertEqual(source, "configured dev split")
            self.assertEqual(json.loads(path.read_text()), dp)

    def test_unavailable_saved_split_uses_legacy_ten_percent_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            train, val, source = finetuning_split(
                Path(tmp), np.arange(50), np.arange(40), np.arange(40, 50), 42)
        self.assertEqual((len(train), len(val)), (36, 4))
        self.assertEqual(source, "reconstructed dev split")

    def test_protected_groups_stay_in_training_in_reconstructed_split(self):
        groups = np.repeat(np.arange(8), 2)
        config = {"enabled": True, "protected_group_ids": [0, 7]}
        test = np.flatnonzero(groups == 6)
        dev = np.flatnonzero(groups != 6)
        with tempfile.TemporaryDirectory() as tmp:
            train, val, _ = finetuning_split(
                Path(tmp), np.arange(len(groups)), dev, test, 42, .25,
                groups=groups, group_cfg=config)
        self.assertTrue({0, 7}.issubset(set(groups[train])))
        self.assertFalse({0, 7} & set(groups[val]))
        self.assertEqual(set(train) | set(val), set(dev))

    def test_imported_split_rejects_protected_groups_in_validation(self):
        groups = np.repeat(np.arange(4), 2)
        config = {"enabled": True, "protected_group_ids": [0]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data_processing.json"
            args = (Path(tmp), np.arange(8), np.arange(6), np.arange(6, 8), 42, None)
            path.write_text(json.dumps({"train_part_ids": [0, 1, 2, 3], "val_part_ids": [4, 5]}))
            train, val, _ = finetuning_split(*args, groups=groups, group_cfg=config)
            np.testing.assert_array_equal(train, [0, 1, 2, 3])
            np.testing.assert_array_equal(val, [4, 5])
            path.write_text(json.dumps({"train_part_ids": [2, 3, 4, 5], "val_part_ids": [0, 1]}))
            with self.assertRaisesRegex(ValueError, "protected group.*validation"):
                finetuning_split(*args, groups=groups, group_cfg=config)

    def test_requested_grouped_fraction_preserves_protected_and_edge_groups(self):
        groups = np.repeat(np.arange(12), 2)
        dev, test = np.arange(20), np.arange(20, 24)
        edge_values = groups[:, None].astype(float)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "data_processing.json").write_text(json.dumps({
                "train_part_ids": list(range(18)), "val_part_ids": [18, 19]}))
            for fraction, expected in ((.2, 4), (.4, 8)):
                train, val, source = finetuning_split(
                    directory, np.arange(24), dev, test, 42, fraction,
                    groups=groups, group_cfg={"enabled": True, "protected_group_ids": [1]},
                    edge_values=edge_values, edge_cfg={"enabled": True, "exclude_from_test": False,
                                                       "exclude_from_validation": True})
                self.assertEqual(len(val), expected)
                self.assertEqual(source, "configured grouped dev split")
                self.assertFalse(set(groups[train]) & set(groups[val]))
                self.assertFalse({0, 1, 9} & set(groups[val]))
                self.assertTrue({0, 1, 9}.issubset(set(groups[train])))
                self.assertEqual(set(train) | set(val), set(dev))

    def test_only_m2_changes_and_saved_checkpoint_matches_selected_model(self):
        model = self.make_model()
        before = copy.deepcopy(model.state_dict())
        X = np.random.default_rng(3).uniform(size=(12, 2)).astype(np.float32)
        settings = {"epochs": 4, "lr": 0.01, "batch_size": 4, "patience": 4}
        m1_modes = []
        hook = model.merge.register_forward_pre_hook(
            lambda module, inputs: m1_modes.append(module.training))
        with tempfile.TemporaryDirectory() as tmp:
            info = finetune_m2(model, X, np.arange(12), X.copy(),
                               np.arange(8), np.arange(8, 12), 0., settings,
                               42, torch.device("cpu"), Path(tmp))
            self.assertGreater(info["best_epoch"], 0)
            saved = torch.load(Path(tmp) / "fusion_model.pt", weights_only=True)
            self.assertTrue(any(not torch.equal(before[k], v)
                                for k, v in model.state_dict().items() if k.startswith("m2.")))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(saved[key], value)
                if not key.startswith("m2."):
                    torch.testing.assert_close(before[key], value, rtol=0, atol=0)
            self.assertFalse(any(m1_modes))
            self.assertFalse(any(p.requires_grad for p in model.parameters()))
        hook.remove()

    def test_weight_loss_gradient_passes_through_frozen_m1_to_m2(self):
        model = self.make_model()
        for p in model.m2.parameters():
            p.requires_grad_(True)
        x = torch.ones(4, 2)
        w, f = model(x, x, return_features=True)
        loss = fusion_mae_loss(w, w.detach() + 1, f, f.detach(), 1., 2., torch.ones(2))[0]
        loss.backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.m2.parameters()), 0)
        self.assertTrue(all(p.grad is None for p in model.merge.parameters()))

    def test_selected_xai_helper_writes_explanations_to_best_fusion(self):
        from contextlib import ExitStack, redirect_stdout
        import io
        from src import Fusion_M1M2_WPred as fusion_module

        model = self.make_model()
        X = np.random.default_rng(5).uniform(size=(5, 2)).astype(np.float32)
        args = SimpleNamespace(shap_bg=2, n_ig_steps=2, eg_bg=2, ih_samples=1)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            output = Path(directory) / "best_fusion"
            output.mkdir()
            stack.enter_context(redirect_stdout(io.StringIO()))
            for name in ('run_shap', 'run_ig', 'run_shap_m1merge', 'run_ig_m1merge'):
                stack.enter_context(patch.object(fusion_module, name,
                    side_effect=lambda model, background, samples, *args: np.zeros_like(samples)))
            stack.enter_context(patch.object(fusion_module, 'run_jacobian',
                side_effect=lambda model, samples, device: np.zeros_like(samples)))
            stack.enter_context(patch.object(fusion_module, 'run_integrated_hessians',
                return_value=np.zeros((4, 4))))
            fusion_module.run_fusion_xai(model, X[:3], X[3:], np.array([40, 41]),
                ['pp_a', 'pp_b'], 3, [2, 0], args, torch.device('cpu'), output, 'synthetic')
            import pandas as pd
            self.assertEqual(pd.read_csv(output / 'shap_values.csv')['ID_Part'].tolist(), [40, 41])
            self.assertTrue((output / 'feature_attributions_single_pp.csv').is_file())
            self.assertTrue((output / 'xai_m1merge_shap_beeswarm.png').is_file())

    def test_selection_uses_a_finetuned_epoch_even_when_baseline_has_zero_loss(self):
        model = self.make_model()
        X = np.random.default_rng(2).uniform(size=(12, 2)).astype(np.float32)
        with torch.no_grad():
            weight, features = model(torch.tensor(X), torch.tensor(X), return_features=True)
        settings = {"epochs": 2, "lr": 0.01, "batch_size": 4, "patience": 2}
        before = copy.deepcopy(model.m2.state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            info = finetune_m2(model, X, weight.numpy(), features.numpy(),
                               np.arange(8), np.arange(8, 12), 0., settings,
                               42, torch.device("cpu"), Path(tmp))
        self.assertEqual(info["initial_val_loss"], 0.)
        self.assertGreaterEqual(info["best_epoch"], 1)
        self.assertTrue(any(not torch.equal(before[k], v)
                            for k, v in model.m2.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
