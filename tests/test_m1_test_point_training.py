"""Focused synthetic checks; no dataset training or HPO jobs are launched."""
import copy
import io
import json
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from src import M1_PPPT_to_W as m1
from src.Utility.m1_test_point_training import (
    TestPointMiniBatchSampler, InsufficientVariabilityWarning,
    normalization_factors, test_point_loss as loss_fn,
    test_point_metrics as grouped_metrics,
)


class SamplerTests(unittest.TestCase):
    def test_default_four_test_points_sixteen_distinct_parts_and_coverage(self):
        ids = np.repeat(np.arange(12), 32)
        sampler = TestPointMiniBatchSampler(ids, seed=7)
        expected_length = len(sampler)
        mini_batches = list(sampler)
        self.assertEqual(len(mini_batches), expected_length)
        for indices in mini_batches:
            self.assertEqual(len(indices), 64)
            self.assertEqual(len(set(indices)), 64)
            tps, counts = np.unique(ids[indices], return_counts=True)
            self.assertEqual(len(tps), 4)
            np.testing.assert_array_equal(counts, [16] * 4)
        self.assertEqual(set(np.concatenate(mini_batches)), set(range(len(ids))))
        self.assertEqual(len(np.concatenate(mini_batches)), len(ids))
        self.assertNotEqual(mini_batches, list(sampler))
        self.assertEqual(mini_batches, list(TestPointMiniBatchSampler(ids, seed=7)))

    def test_no_epoch_repeats_when_four_balanced_test_points(self):
        ids = np.repeat(np.arange(4), 32)
        samples = np.concatenate(list(TestPointMiniBatchSampler(ids)))
        self.assertEqual(len(samples), len(ids))
        self.assertEqual(len(set(samples)), len(ids))

    def test_small_test_point_and_epoch_tail_are_retained(self):
        ids = np.repeat(np.arange(5), [3, 17, 19, 16, 33])
        with self.assertWarnsRegex(UserWarning, "smaller mini-batches"):
            sampler = TestPointMiniBatchSampler(ids)
        mini_batches = list(sampler)
        for indices in mini_batches:
            self.assertEqual(len(indices), len(set(indices)))
            tps, counts = np.unique(ids[indices], return_counts=True)
            self.assertEqual(len(tps), 4)
            for tp, count in zip(tps, counts):
                self.assertEqual(count, min(16, np.sum(ids == tp)))
        self.assertEqual(set(np.concatenate(mini_batches)), set(range(len(ids))))

    def test_invalid_sampling_sizes_and_missing_ids_raise(self):
        ids = np.repeat(np.arange(4), 16)
        for options in ({"mini_batch_size": 32}, {"test_points_per_mini_batch": 1},
                        {"parts_per_test_point": 0}, {"mini_batch_size": 64.0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                TestPointMiniBatchSampler(ids, **options)
        with self.assertRaisesRegex(ValueError, "distinct test points"):
            TestPointMiniBatchSampler(np.zeros(100))
        with self.assertRaisesRegex(ValueError, "missing"):
            TestPointMiniBatchSampler([1, 2, None, 4])


class LossTests(unittest.TestCase):
    def setUp(self):
        self.actual = torch.tensor([0., 2., 10., 14., 18.], dtype=torch.double)
        self.pred = torch.tensor([1., 5., 11., 11., 17.], dtype=torch.double)
        self.ids = torch.tensor([0, 0, 1, 1, 1])
        self.factors = {"s_w": 6., "s_within": 2.}

    def test_subset_means_part_denominator_and_equal_test_point_weighting(self):
        combined, absolute, within = loss_fn(self.pred, self.actual, self.ids, .5, self.factors)
        # TP 0: actual mean=1, predicted mean=3, centered MAE=1.
        # TP 1: actual mean=14, predicted mean=13, centered MAE=4/3.
        self.assertAlmostEqual(absolute.item(), 9/5)
        self.assertAlmostEqual(within.item(), (1 + 4/3)/2)
        self.assertNotAlmostEqual(within.item(), (2*1 + 3*4/3)/5)
        self.assertAlmostEqual(combined.item(), 9/5 + .5*3*(7/6))

    def test_equal_group_sizes_match_per_part_centered_error(self):
        actual = self.actual[:4]
        predicted = self.pred[:4]
        ids = self.ids[:4]
        within = loss_fn(predicted, actual, ids, 1, self.factors)[2]
        a = actual.reshape(2, 2)
        p = predicted.reshape(2, 2)
        per_part = ((p - p.mean(1, keepdim=True)) - (a - a.mean(1, keepdim=True))).abs().mean()
        torch.testing.assert_close(within, per_part)

    def test_zero_for_exact_predictions_and_constant_error_per_test_point(self):
        for predicted in (self.actual, self.actual + torch.tensor([7., 7., -3., -3., -3.])):
            self.assertEqual(loss_fn(predicted, self.actual, self.ids, 1, self.factors)[2].item(), 0)

    def test_beta_zero_is_original_mae_exactly(self):
        combined, absolute, _ = loss_fn(self.pred, self.actual, self.ids, 0, self.factors)
        self.assertTrue(torch.equal(combined, nn.L1Loss()(self.pred, self.actual)))
        self.assertTrue(torch.equal(combined, absolute))

    def test_gradients_flow_through_predicted_means(self):
        pred = torch.tensor([0., 0., 3.], dtype=torch.double, requires_grad=True)
        actual = torch.zeros(3, dtype=torch.double)
        ids = torch.zeros(3, dtype=torch.long)
        within = loss_fn(pred, actual, ids, 1, self.factors)[2]
        within.backward()
        torch.testing.assert_close(pred.grad, torch.tensor([-2/9, -2/9, 4/9], dtype=torch.double))
        self.assertAlmostEqual(pred.grad.sum().item(), 0)
        self.assertTrue(torch.autograd.gradcheck(
            lambda p: loss_fn(p, actual, ids, 1, self.factors)[2], (pred,)))

    def test_singleton_test_point_and_negative_beta(self):
        within = loss_fn(self.pred[:3], self.actual[:3], self.ids[:3], 1, self.factors)[2]
        self.assertAlmostEqual(within.item(), .5)  # Singleton contributes zero with equal TP weight.
        with self.assertRaises(ValueError):
            loss_fn(self.pred, self.actual, self.ids, -1, self.factors)


class NormalizationAndMetricTests(unittest.TestCase):
    def test_normalization_uses_part_weighting_and_equal_test_point_weighting(self):
        factors = normalization_factors([0, 2, 10, 14, 18], ['a', 'a', 'b', 'b', 'b'])
        self.assertAlmostEqual(factors['s_w'], 6.24)
        self.assertAlmostEqual(factors['s_within'], (1 + 8/3)/2)
        self.assertEqual(factors['n_training_parts'], 5)
        self.assertEqual(factors['n_training_test_points'], 2)

    def test_near_zero_variability_warns_and_floors(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            factors = normalization_factors([3, 3, 3], [0, 0, 1])
        self.assertEqual(len(caught), 2)
        self.assertTrue(all(issubclass(w.category, InsufficientVariabilityWarning) for w in caught))
        self.assertEqual(factors['s_w'], 1e-8)
        self.assertEqual(factors['s_within'], 1e-8)
        with self.assertRaises(ValueError):
            normalization_factors([1, 2], [0, 0], epsilon=0)

    def test_evaluation_uses_complete_test_points_and_reports_population_std(self):
        actual = np.array([0., 2., 4., 6., 100., 102.])
        predicted = np.array([0., 0., 8., 8., 110., 112.])
        ids = ['a']*4 + ['b']*2
        result = grouped_metrics(actual, predicted, ids)
        self.assertAlmostEqual(result['MAE_w'], 28/6)
        self.assertAlmostEqual(result['MAE_TPavg'], 5.5)
        self.assertAlmostEqual(result['MAE_within'], 1.)
        self.assertAlmostEqual(result['test_points'][0]['actual_std_g'], np.sqrt(5))
        self.assertEqual(result['test_points'][0]['predicted_std_g'], 4.)
        # Arbitrary two-part centering would yield 1 for TP a, instead of 2.
        arbitrary = [grouped_metrics(actual[i:i+2], predicted[i:i+2], ['a']*2)['MAE_within']
                     for i in (0, 2)]
        self.assertEqual(np.mean(arbitrary), 1.)
        self.assertEqual(result['test_points'][0]['MAE_within'], 2.)
        permutation = [0, 4, 2, 5, 1, 3]
        interleaved = grouped_metrics(actual[permutation], predicted[permutation], np.array(ids)[permutation])
        self.assertEqual(result, interleaved)


class TinyModel(nn.Module):
    """Cheap substitute to exercise the real M1 training and evaluation wiring."""
    def __init__(self, n_pp=1, mcfg=None):
        super().__init__()
        self.linear = nn.Linear(n_pp, 1)

    def forward(self, sensor_curves, process_parameters):
        return self.linear(process_parameters).squeeze(-1)


def tiny_config():
    return {'batch_size': 4, 'lr': .001, 'weight_decay': 0., 'epochs': 1,
            'patience': 2, 'scheduler_factor': .5, 'scheduler_patience': 1,
            'print_every': 999, 'k_folds': 2, 'val_fraction': .25,
            'within_test_point_beta': .25,
            'test_point_training': {'enabled': True, 'test_points_per_mini_batch': 2,
                                    'parts_per_test_point': 2}}


class TrainingIntegrationTests(unittest.TestCase):
    def test_one_step_per_combined_mini_batch_fixed_training_only_factors_and_fair_sampling(self):
        pp = np.arange(8, dtype=np.float32).reshape(-1, 1)
        y = np.array([0, 2, 10, 14, 20, 22, 30, 34], dtype=np.float32)
        ids = np.repeat(np.arange(4), 2)
        factors_seen, mini_batches_seen = [], []
        original_loss = m1.test_point_loss
        def record_loss(prediction, actual, codes, beta, factors):
            factors_seen.append(factors)
            mini_batches_seen.append((actual.clone(), codes.clone()))
            return original_loss(prediction, actual, codes, beta, factors)
        for beta, validation_y in ((0., [1e6, -1e6]), (.5, [-1e9, 1e9])):
            cfg = {**tiny_config(), 'epochs': 2, 'within_test_point_beta': beta}
            with patch.object(m1, 'test_point_loss', side_effect=record_loss):
                trained, _ = m1.train_fold(TinyModel(), pp, [], y, pp[:2], [],
                                          np.array(validation_y, dtype=np.float32), cfg,
                                          torch.device('cpu'), ids)
            info = trained.m1_training_info
            self.assertEqual(info['normalization'], normalization_factors(y, ids))
            self.assertEqual([h['optimizer_steps'] for h in info['history']], [2, 2])
        self.assertTrue(all(factors is factors_seen[0] for factors in factors_seen[:4]))
        for before, after in zip(mini_batches_seen[:4], mini_batches_seen[4:]):
            torch.testing.assert_close(before[0], after[0])
            torch.testing.assert_close(before[1], after[1])
            self.assertEqual(len(torch.unique(before[1])), 2)

    def test_cv_and_hpo_recompute_factors_on_each_training_fold_only(self):
        ids = np.repeat(np.arange(12), 2)
        y = np.arange(24, dtype=np.float32)**2
        pp = np.arange(24, dtype=np.float32).reshape(-1, 1)
        observed = []
        group_cfg = {'enabled': True, 'protected_group_ids': [0, 11]}
        original = m1.train_fold
        def record_training(*args, **kwargs):
            self.assertTrue({0, 11}.issubset(set(args[9])))
            self.assertFalse({0, 11} & set(args[10]))
            trained, score = original(*args, **kwargs)
            expected = normalization_factors(args[3], args[9])
            self.assertEqual(trained.m1_training_info['normalization'], expected)
            self.assertEqual(expected['n_training_parts'], len(args[3]))
            self.assertLess(len(args[3]), len(y))
            observed.append(expected)
            return trained, score
        trial = m1.optuna.trial.FixedTrial({})
        optuna_cfg = {'search_space': {'batch_size': [1, 1]}, 'n_folds': 2, 'epochs': 1, 'patience': 2}
        original_evaluate = m1.evaluate_model
        def check_holdout(*args, **kwargs):
            self.assertFalse({0, 11} & set(args[5]))
            return original_evaluate(*args, **kwargs)
        with patch.object(m1, 'M1Model', TinyModel), patch.object(m1, 'train_fold', side_effect=record_training), patch.object(m1, 'evaluate_model', side_effect=check_holdout), redirect_stdout(io.StringIO()):
            metrics, _, _, records = m1.run_cv(pp, [], y, {}, tiny_config(), torch.device('cpu'),
                                              42, group_values=ids, group_cfg=group_cfg)
            self.assertEqual(len(records), 2)
            self.assertIn('MAE_within', metrics)
            m1.optuna_objective(trial, pp, [], y, optuna_cfg, {}, tiny_config(), 42,
                                torch.device('cpu'), group_values=ids, group_cfg=group_cfg)
        self.assertEqual(len(observed), 4)
        self.assertNotEqual(observed[0]['s_w'], observed[1]['s_w'])
        self.assertEqual(observed[0], observed[2])
        self.assertEqual(observed[1], observed[3])
        self.assertIn('fold_0_training', trial.user_attrs)

    def test_final_training_uses_training_only_factors_and_preserves_group_splits(self):
        ids = np.repeat(np.arange(12), 2)
        pp = np.arange(24, dtype=np.float32).reshape(-1, 1)
        y = pp[:, 0] ** 2
        dev_idx = np.arange(20)
        with patch.object(m1, 'M1Model', TinyModel), redirect_stdout(io.StringIO()):
            model, _, train_idx, val_idx = m1.final_train_m1(
                pp, [], y, dev_idx, {}, tiny_config(), 42, torch.device('cpu'),
                group_values=ids, group_cfg={'enabled': True, 'protected_group_ids': [0, 9]})
        self.assertTrue({0, 9}.issubset(set(ids[train_idx])))
        self.assertFalse({0, 9} & set(ids[val_idx]))
        self.assertFalse(set(ids[train_idx]) & set(ids[val_idx]))
        self.assertEqual(set(train_idx) | set(val_idx), set(dev_idx))
        self.assertEqual(model.m1_training_info['normalization'],
                         normalization_factors(y[train_idx], ids[train_idx]))
        self.assertIn('MAE_within', model.m1_training_info['best_validation_metrics'])
        self.assertIn('val_MAE_within_g', model.m1_training_info['history'][0])

    def test_legacy_final_stratification_uses_only_development_labels(self):
        pp = np.arange(24, dtype=np.float32).reshape(-1, 1)
        labels = np.tile([0, 1], 12)
        cfg = {**tiny_config(), 'test_point_training': {'enabled': False},
               'within_test_point_beta': 0.}
        with patch.object(m1, 'M1Model', TinyModel), redirect_stdout(io.StringIO()):
            _, _, train_idx, val_idx = m1.final_train_m1(
                pp, [], pp[:, 0], np.arange(20), {}, cfg, 42,
                torch.device('cpu'), strat_labels=labels)
        self.assertEqual(set(train_idx) | set(val_idx), set(range(20)))
        self.assertEqual(set(labels[val_idx]), {0, 1})

    def test_test_point_training_rejects_ungrouped_cv(self):
        with self.assertRaisesRegex(ValueError, 'prevent test-point leakage'):
            m1.run_cv(np.zeros((24, 1)), [], np.zeros(24), {}, tiny_config(),
                      torch.device('cpu'), 42, group_values=np.repeat(np.arange(12), 2))

    def test_evaluate_model_calls_full_group_metrics(self):
        pp = np.array([[0], [0], [8], [8]], dtype=np.float32)
        model = TinyModel()
        with torch.no_grad():
            model.linear.weight.fill_(1)
            model.linear.bias.zero_()
        result = m1.evaluate_model(model, pp, [], np.array([0, 2, 4, 6]),
                                   torch.device('cpu'), ['a']*4)
        self.assertEqual(result['MAE_within'], 2.)
        self.assertEqual(len(result['test_points']), 1)

    def test_checkpoint_promotion_uses_validation_mae_and_preserves_state_dict(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            run_dir, overall = Path(directory) / 'run', Path(directory) / 'overall'
            run_dir.mkdir()
            overall.mkdir()
            (overall / 'best_metrics.json').write_text(json.dumps(
                {'MAE': 0., 'validation_MAE': 2.}))
            model = TinyModel()
            metrics = {'MAE': 100., 'RMSE': 100., 'R2': 0., 'MSE': 10000.,
                       'validation_MAE': 1., 'MAE_within': .25}
            self.assertTrue(m1.save_models(model, metrics, 1, [], {}, run_dir, overall))
            saved = torch.load(overall / 'best_model.pt', weights_only=True)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(saved[key], value)
            self.assertEqual(json.loads((overall / 'best_metrics.json').read_text())['MAE_within'], .25)
            metrics.update({'MAE': 0., 'validation_MAE': 3.})
            self.assertFalse(m1.save_models(model, metrics, 1, [], {}, run_dir, overall))
            self.assertEqual(json.loads((overall / 'best_metrics.json').read_text())['validation_MAE'], 1.)

    def test_complete_promotion_replaces_old_test_1_and_nested_reports(self):
        from src.Utility.dataset_layout import label_test_1

        with tempfile.TemporaryDirectory() as directory:
            source, overall = Path(directory) / 'winner', Path(directory) / 'overall'
            source.mkdir()
            overall.mkdir()
            (source / 'test_predictions_per_batch.csv').write_text('winning predictions')
            (source / 'scatter_final_test.png').write_bytes(b'winning plot')
            (source / 'test_2').mkdir()
            (source / 'test_2' / 'scatter_test_2.png').write_bytes(b'winning external plot')
            (overall / 'test_1_predictions_per_batch.csv').write_text('first run predictions')
            (overall / 'scatter_final_test_1.png').write_bytes(b'first run plot')
            (overall / 'test_2').mkdir()
            (overall / 'test_2' / 'obsolete.png').write_bytes(b'obsolete plot')
            (overall / 'manifest.json').write_text('stale checkpoint hashes')
            (Path(directory) / 'dataset/Train').mkdir(parents=True)
            label_test_1(source, {'dataset_dir': str(Path(directory) / 'dataset')})
            m1._promote_complete_run(source, overall)
            self.assertEqual((overall / 'test_1_predictions_per_batch.csv').read_text(),
                             'winning predictions')
            self.assertEqual((overall / 'scatter_final_test_1.png').read_bytes(), b'winning plot')
            self.assertEqual((overall / 'test_2/scatter_test_2.png').read_bytes(),
                             b'winning external plot')
            self.assertFalse((overall / 'test_2/obsolete.png').exists())
            self.assertFalse((overall / 'manifest.json').exists())

    def test_beta_resolution(self):
        self.assertEqual(m1.resolve_beta_values(.25), [.25])
        self.assertEqual(m1.resolve_beta_values([0, .1]), [0., .1])
        for invalid in ([], [-1], [float('nan')], [float('inf')], [0, 0]):
            with self.assertRaises(ValueError):
                m1.resolve_beta_values(invalid)

    def test_sweep_selection_and_promotion_use_validation_not_test(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = {'mode': 'manual', 'training': {'test_point_training': {'enabled': True}},
                   'output': {key: str(Path(directory) / key)
                              for key in ('output_dir', 'run_best_dir', 'best_overall_dir')}}
            original = copy.deepcopy(cfg)
            calls = []
            def fake_experiment(experiment, mat, config_path):
                beta = experiment['training']['within_test_point_beta']
                calls.append(experiment)
                label = 'beta_' + str(beta).removesuffix('.0')
                run_dir = Path(experiment['output']['run_best_dir']) / m1._RUN_TS / label
                run_dir.mkdir(parents=True)
                validation = 1. if beta == .25 else 2.
                # The best test result intentionally has the worse validation result.
                metrics = {'MAE': 100. if beta == .25 else 0., 'validation_MAE': validation}
                (run_dir / 'best_metrics.json').write_text(json.dumps(metrics))
                (run_dir / 'data_processing.json').write_text(json.dumps({'edge_group_split': {}}))
                (run_dir / 'test_2').mkdir()
                (run_dir / 'test_2' / 'predictions.csv').write_text(str(beta))
                return {'validation_mae_g': validation, 'run_dir': str(run_dir),
                        'within_test_point_beta': beta}
            with patch.object(m1, 'run_experiment', side_effect=fake_experiment), redirect_stdout(io.StringIO()):
                result = m1.run_beta_experiments(cfg, 'DOE1', 'unused', [0, .25])
            self.assertEqual(result['selected_beta'], .25)
            self.assertEqual(cfg, original)
            self.assertNotEqual(calls[0]['output'], calls[1]['output'])
            for call, record in zip(calls, result['experiments']):
                self.assertEqual(call['output']['run_best_dir'], cfg['output']['run_best_dir'])
                beta = record['within_test_point_beta']
                label = 'beta_' + str(beta).removesuffix('.0')
                self.assertEqual(Path(record['run_dir']),
                                 Path(cfg['output']['run_best_dir']) / m1._RUN_TS / label)
                self.assertEqual(Path(call['output']['best_overall_dir']),
                                 Path(cfg['output']['best_overall_dir']) / label)
            promoted = json.loads((Path(cfg['output']['best_overall_dir']) / 'best_metrics.json').read_text())
            self.assertEqual(promoted['MAE'], 100.)
            self.assertEqual(promoted['validation_MAE'], 1.)
            self.assertEqual((Path(cfg['output']['best_overall_dir']) /
                              'test_2/predictions.csv').read_text(), '0.25')
            summary = Path(cfg['output']['output_dir']) / 'beta_experiments' / m1._RUN_TS
            for name in ('beta_mae_comparison.png', 'beta_mae_comparison.csv'):
                self.assertTrue((summary / name).exists())
                self.assertTrue((Path(cfg['output']['best_overall_dir']) / name).exists())


if __name__ == '__main__':
    unittest.main()
