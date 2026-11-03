import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from src.Fusion_M1M2_WPred import (
    FusionModel, M2Model, PPMLP, MergeHead, TorchMinMaxScaler,
    finetune_m2, finetuning_split, fusion_mae_loss, resolve_finetuning,
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
            loss, w, f = fusion_mae_loss(pred_w, true_w, pred_f, true_f, alpha)
            self.assertAlmostEqual(w.item(), 2.5)
            self.assertAlmostEqual(f.item(), 3.5)
            self.assertAlmostEqual(loss.item(), alpha * 2.5 + (1-alpha) * 3.5)

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

    def test_split_reuses_ids_and_rejects_test_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "data_processing.json"
            path.write_text(json.dumps({"train_part_ids": [10, 30], "val_part_ids": [20]}))
            args = (directory, np.array([30, 20, 10, 40]), np.array([0, 1, 2]),
                    np.array([3]), 42, 0.1)
            train, val, _ = finetuning_split(*args)
            np.testing.assert_array_equal(train, [2, 0])
            np.testing.assert_array_equal(val, [1])
            path.write_text(json.dumps({"train_part_ids": [10, 40], "val_part_ids": [20]}))
            with self.assertRaises(ValueError):
                finetuning_split(*args)

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
            args = (Path(tmp), np.arange(8), np.arange(6), np.arange(6, 8), 42, .25)
            path.write_text(json.dumps({"train_part_ids": [0, 1, 2, 3], "val_part_ids": [4, 5]}))
            train, val, _ = finetuning_split(*args, groups=groups, group_cfg=config)
            np.testing.assert_array_equal(train, [0, 1, 2, 3])
            np.testing.assert_array_equal(val, [4, 5])
            path.write_text(json.dumps({"train_part_ids": [2, 3, 4, 5], "val_part_ids": [0, 1]}))
            with self.assertRaisesRegex(ValueError, "protected group.*validation"):
                finetuning_split(*args, groups=groups, group_cfg=config)

    def test_only_m2_changes_and_saved_checkpoint_matches_selected_model(self):
        model = self.make_model()
        before = copy.deepcopy(model.state_dict())
        X = np.random.default_rng(3).uniform(size=(12, 2)).astype(np.float32)
        settings = {"epochs": 4, "lr": 0.01, "batch_size": 4, "patience": 4}
        m1_modes = []
        hook = model.merge.register_forward_pre_hook(
            lambda module, inputs: m1_modes.append(module.training))
        with tempfile.TemporaryDirectory() as tmp:
            info = finetune_m2(model, X, np.zeros(12), np.zeros((12, 2)),
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
        loss, _, _ = fusion_mae_loss(w, w.detach() + 1, f, f.detach(), 1.)
        loss.backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.m2.parameters()), 0)
        self.assertTrue(all(p.grad is None for p in model.merge.parameters()))

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
