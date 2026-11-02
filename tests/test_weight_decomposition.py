import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from src.Utility.weight_decomposition import (
    LOSS_VARIANTS, WholeGroupBatchSampler, decomposition_loss,
    feature_drop_columns, pack_batch_targets, promote_m1_candidate,
    resolve_decomposition,
)
from src.M1_PPPT_to_W import M1Model, train_fold
from src.Fusion_M1M2_WPred import (
    M1Model as FusionM1Model, FusionModel, M2Model, TorchMinMaxScaler,
    M1MergeWrapper, fusion_mae_loss, finetune_m2,
)


def model_config(settings):
    return {"curve_encoders": [{"conv_channels": [1, 2], "conv_kernels": [3],
                                "n_f_features": 2}],
            "pp_hidden": [4], "pp_latent_dim": 3, "merge_hidden": [4],
            "dropout": 0., "weight_decomposition": True, "loss_settings": settings}


class WeightDecompositionTests(unittest.TestCase):
    def test_batch_targets_use_actual_means_and_signed_deviations(self):
        targets = pack_batch_targets([10., 14., 20.], ["A", "A", "B"])
        np.testing.assert_allclose(targets[:, 1], [12, 12, 20])
        np.testing.assert_allclose(targets[:, 2], [-2, 2, 0])
        np.testing.assert_allclose(targets[:, 1] + targets[:, 2], targets[:, 0])
        np.testing.assert_allclose(targets[:, 3], [0, 0, 1])

    def test_all_three_objectives_and_balancing_are_exact(self):
        targets = torch.tensor(pack_batch_targets([10., 14., 20.], ["A", "A", "B"]))
        predictions = torch.tensor([[11., -1.], [13., 1.], [20., 2.]], requires_grad=True)
        _, settings = resolve_decomposition({"beta": .25, "gamma": 2, "balance_lambda": 3})
        for loss_settings in settings:
            loss, terms = decomposition_loss(predictions, targets, loss_settings)
            self.assertAlmostEqual(terms["mae_avg"].item(), 2/3, places=6)
            self.assertAlmostEqual(terms["mae_dev"].item(), 4/3, places=6)
            self.assertAlmostEqual(terms["mae_w"].item(), 2/3, places=6)
            # Batch A's dev mean is 0; B's is 2. Both batches count equally.
            self.assertAlmostEqual(terms["balance_loss"].item(), 1.)
            expected = .25 * (2/3) + .75 * (4/3)
            expected += loss_settings["gamma"] * (2/3) + loss_settings["balance_lambda"]
            self.assertAlmostEqual(loss.item(), expected, places=6)
        settings[-1]["beta"] = .5
        loss, _ = decomposition_loss(predictions, targets, settings[-1])
        loss.backward()
        self.assertGreater(predictions.grad[:, 0].abs().sum(), 0)
        self.assertGreater(predictions.grad[:, 1].abs().sum(), 0)

    def test_sample_weighted_average_loss_gives_large_batches_more_weight(self):
        targets = torch.tensor(pack_batch_targets([0., 0., 0., 0.], [1, 1, 1, 2]))
        predictions = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [5., 0.]])
        loss, _ = decomposition_loss(predictions, targets, {"beta": 1})
        self.assertEqual(loss.item(), 2.)

    def test_config_scalar_vector_endpoints_and_invalid_values(self):
        self.assertEqual(resolve_decomposition({"beta": .3})[0], [.3])
        betas, settings = resolve_decomposition({"beta": [0, 1]})
        self.assertEqual(betas, [0, 1])
        self.assertEqual(len(settings), 6)
        self.assertEqual({s["variant"] for s in settings}, set(LOSS_VARIANTS))
        for invalid in ([], [-.1], [1.1], [float("nan")], [.5, .5]):
            with self.assertRaises(ValueError):
                resolve_decomposition({"beta": invalid})
        with self.assertRaises(ValueError):
            resolve_decomposition({"gamma": -1})

    def test_complete_group_sampler_never_splits_a_production_batch(self):
        codes = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2, 3])
        sampler = WholeGroupBatchSampler(codes, 3, torch.Generator().manual_seed(42))
        batches = list(sampler)
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(len(codes))))
        for code in np.unique(codes):
            indices = set(np.flatnonzero(codes == code))
            self.assertEqual(sum(bool(indices & set(batch)) for batch in batches), 1)

    def test_nominal_pps_can_be_inputs_without_disabling_edge_metadata(self):
        data = {"drop_cols": ["Batch ID", "nominal.pressure"],
                "edge_group_split": {"columns": ["nominal.pressure"]}}
        self.assertIn("nominal.pressure", feature_drop_columns(data))
        data["include_nominal_pp"] = True
        self.assertEqual(feature_drop_columns(data), ["Batch ID"])

    def test_both_m1_loaders_reproduce_two_outputs_and_their_sum(self):
        cfg = model_config({"beta": .5, "gamma": 1, "balance_lambda": 1})
        torch.manual_seed(10)
        model = M1Model(2, cfg).eval()
        mirrored = FusionM1Model(2, cfg).eval()
        mirrored.load_state_dict(model.state_dict(), strict=True)
        curves, pp = [torch.randn(5, 1, 8)], torch.randn(5, 2)
        components = model(curves, pp, return_components=True)
        self.assertEqual(tuple(components.shape), (5, 2))
        torch.testing.assert_close(model(curves, pp), components.sum(dim=1))
        torch.testing.assert_close(mirrored(curves, pp), model(curves, pp))

    def test_m1_training_and_validation_use_total_weight_mae_for_selection(self):
        _, settings = resolve_decomposition({"beta": .5})
        rng = np.random.default_rng(42)
        pp = rng.normal(size=(12, 2)).astype(np.float32)
        curves = [rng.normal(size=(12, 8)).astype(np.float32)]
        targets = pack_batch_targets(np.linspace(1, 2, 12), np.repeat(np.arange(4), 3))
        training = {"epochs": 2, "patience": 2, "batch_size": 4, "lr": .01,
                    "weight_decay": 0, "scheduler_factor": 1, "print_every": 100}
        for loss in settings:
            model = M1Model(2, model_config(loss))
            model, best = train_fold(model, pp[:9], [curves[0][:9]], targets[:9],
                                     pp[9:], [curves[0][9:]], targets[9:], training, torch.device("cpu"))
            self.assertAlmostEqual(best, min(row["val_mae_w"] for row in model.training_history))
            with torch.no_grad():
                prediction = model([torch.tensor(curves[0][9:, None, :])], torch.tensor(pp[9:]))
            self.assertAlmostEqual(best, (prediction-torch.tensor(targets[9:, 0])).abs().mean().item())

    def test_fusion_inherits_component_loss_and_freezes_m1_in_all_variants(self):
        _, settings = resolve_decomposition({"beta": .25})
        rng = np.random.default_rng(2)
        pp = rng.uniform(size=(12, 2)).astype(np.float32)
        targets = pack_batch_targets(np.linspace(1, 2, 12), np.repeat(np.arange(4), 3))
        features = np.zeros((12, 2), dtype=np.float32)
        scaler = TorchMinMaxScaler(MinMaxScaler().fit([[0., 0.], [1., 1.]]))
        for loss in settings:
            m1 = M1Model(2, model_config(loss))
            fusion = FusionModel(M2Model(2, 2, [4], 0), m1.pp_mlp, m1.merge,
                                 copy.deepcopy(scaler), copy.deepcopy(scaler), copy.deepcopy(scaler),
                                 2, [0, 1], loss)
            frozen = {k: v.clone() for k, v in fusion.state_dict().items() if not k.startswith("m2.")}
            old_m2 = copy.deepcopy(fusion.m2.state_dict())
            with tempfile.TemporaryDirectory() as temp:
                info = finetune_m2(fusion, pp, targets, features, np.arange(9), np.arange(9, 12),
                                   .5, {"epochs": 2, "patience": 2, "batch_size": 4, "lr": .01},
                                   42, torch.device("cpu"), Path(temp))
                self.assertEqual(info["weight_loss_settings"], loss)
                self.assertIn("mae_avg", info["val_metrics"])
                self.assertIn("mae_w", info["val_metrics"])
            for key, value in frozen.items():
                torch.testing.assert_close(fusion.state_dict()[key], value, atol=0, rtol=0)
            self.assertTrue(any(not torch.equal(old_m2[k], v) for k, v in fusion.m2.state_dict().items()))
            x = torch.tensor(pp)
            total = fusion(x, x)
            torch.testing.assert_close(total, fusion(x, x, return_components=True).sum(dim=1))
            merge_input = torch.cat([torch.tensor(features), x], dim=1)
            self.assertEqual(tuple(M1MergeWrapper(fusion, 2, 2)(merge_input).shape), (12,))

    def test_alpha_weights_the_whole_decomposed_objective(self):
        targets = torch.tensor(pack_batch_targets([1., 3.], [0, 0]))
        predictions = torch.tensor([[1., 1.], [2., 3.]])
        features = torch.ones(2, 3)
        loss_settings = {"beta": .25, "gamma": 2., "balance_lambda": 3.}
        weight_loss, _ = decomposition_loss(predictions, targets, loss_settings)
        combined, weighted, feature_mae = fusion_mae_loss(
            predictions, targets, features, torch.zeros_like(features), .4, loss_settings)
        torch.testing.assert_close(weighted, weight_loss)
        torch.testing.assert_close(combined, .4 * weight_loss + .6 * feature_mae)

    def test_checkpoint_promotion_uses_validation_not_test_weight_mae(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            overall = root / "overall"
            for index, (val_mae, test_mae) in enumerate([(1., 10.), (2., .1), (.5, 20.)]):
                candidate = root / str(index)
                candidate.mkdir()
                (candidate / "best_metrics.json").write_text(json.dumps({
                    "selection_MAE": val_mae, "MAE": test_mae,
                    "model_cfg": {"weight_decomposition": True}, "signal_names": ["p1"]}))
                (candidate / "data_processing.json").write_text(json.dumps({
                    "pp_cols": ["pressure"], "val_part_ids": [1], "test_part_ids": [2]}))
                self.assertEqual(promote_m1_candidate(candidate, overall), index != 1)
            self.assertEqual(json.loads((overall / "best_metrics.json").read_text())["selection_MAE"], .5)


if __name__ == "__main__":
    unittest.main()
