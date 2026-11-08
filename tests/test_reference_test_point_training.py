"""Synthetic reference-model integration checks; no real HPO or dataset jobs."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import optuna
import torch
from torch import nn

from src import Reference_Models_W_Pred as ref
from src.Utility import m1_test_point_training as shared


class TinyNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        values = x.reshape(len(x), -1).mean(dim=1, keepdim=True)
        return self.linear(values).squeeze(-1)


class TinyTree:
    fits = []
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, x, y):
        self.value = float(np.mean(y))
        self.fits.append((x.copy(), y.copy()))
        return self

    def predict(self, x):
        return np.full(len(x), self.value)


def tiny_config():
    return {'batch_size': 4, 'lr': .001, 'weight_decay': 0., 'l1_lambda': .01,
            'epochs': 1, 'patience': 2, 'scheduler_factor': .5, 'scheduler_patience': 1,
            'print_every': 999, 'k_folds': 2, 'val_fraction': .25,
            'within_test_point_beta': .25,
            'test_point_training': {'enabled': True, 'test_points_per_mini_batch': 2,
                                    'parts_per_test_point': 2}}


class ReferenceTrainingTests(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(24, dtype=np.float32).reshape(-1, 1)
        self.y = self.x[:, 0] ** 2
        self.ids = np.repeat(np.arange(12), 2)
        self.device = torch.device('cpu')

    def test_shares_exact_loss_sampler_normalization_and_metrics_with_m1(self):
        self.assertIs(ref.TestPointMiniBatchSampler, shared.TestPointMiniBatchSampler)
        self.assertIs(ref.test_point_loss, shared.test_point_loss)
        self.assertIs(ref.normalization_factors, shared.normalization_factors)
        self.assertIs(ref.test_point_metrics, shared.test_point_metrics)

    def test_default_64_part_mini_batch_and_identical_sampling_across_betas(self):
        x = np.arange(64, dtype=np.float32).reshape(-1, 1)
        y = x[:, 0] / 10
        ids = np.repeat(np.arange(4), 16)
        observed = []
        factors_seen = []
        original_loss = ref.test_point_loss
        def record_loss(predicted, actual, codes, beta, factors):
            observed.append((actual.clone(), codes.clone()))
            factors_seen.append(factors)
            return original_loss(predicted, actual, codes, beta, factors)
        for beta, validation in ((0, [1e6, -1e6]), (.5, [-1e9, 1e9])):
            cfg = {**tiny_config(), 'batch_size': 64, 'epochs': 2,
                   'within_test_point_beta': beta,
                   'test_point_training': {'enabled': True, 'test_points_per_mini_batch': 4,
                                           'parts_per_test_point': 16}}
            with patch.object(ref, 'test_point_loss', side_effect=record_loss):
                model, _ = ref.train_fold_nn(TinyNN(), ref._mlp_prepare, x, y,
                                              x[:2], np.array(validation, dtype=np.float32),
                                              cfg, self.device, ids)
            self.assertEqual(model.ref_training_info['normalization'], shared.normalization_factors(y, ids))
            self.assertEqual([h['optimizer_steps'] for h in model.ref_training_info['history']], [1, 1])
        self.assertIs(factors_seen[0], factors_seen[1])
        for first, second in zip(observed[:2], observed[2:]):
            torch.testing.assert_close(first[0], second[0])
            torch.testing.assert_close(first[1], second[1])
            self.assertEqual(len(torch.unique(first[0])), 64)
            torch.testing.assert_close(torch.unique(first[1], return_counts=True)[1], torch.tensor([16]*4))

    def test_beta_zero_preserves_mae_plus_existing_l1_regularization(self):
        model = TinyNN()
        cfg = {**tiny_config(), 'lr': 0., 'within_test_point_beta': 0.}
        x, y, ids = self.x[:8], self.y[:8], self.ids[:8]
        with torch.no_grad():
            expected = nn.L1Loss()(model(torch.tensor(x)), torch.tensor(y)).item()
            l1_penalty = cfg['l1_lambda'] * sum(p.abs().sum().item() for p in model.parameters())
        trained, _ = ref.train_fold_nn(model, ref._mlp_prepare, x, y,
                                       self.x[8:12], self.y[8:12], cfg, self.device,
                                       ids, self.ids[8:12])
        history = trained.ref_training_info['history'][0]
        self.assertAlmostEqual(history['train_loss'], expected + l1_penalty, places=5)
        self.assertAlmostEqual(history['train_weight_mae_g'], expected, places=5)
        self.assertAlmostEqual(history['train_l1_penalty'], l1_penalty, places=6)

    def test_legacy_training_uses_part_batches_and_mae_plus_l1_without_tp_ids(self):
        cfg = {**tiny_config(), 'lr': 0., 'batch_size': 3,
               'within_test_point_beta': 0.,
               'test_point_training': {'enabled': False, 'beta_values': [0, .25]}}
        model = TinyNN()
        x, y = self.x[:8], self.y[:8]
        with torch.no_grad():
            expected_mae = nn.L1Loss()(model(torch.tensor(x)), torch.tensor(y)).item()
            expected_l1 = cfg['l1_lambda'] * sum(p.abs().sum().item() for p in model.parameters())
        with patch.object(ref, 'TestPointMiniBatchSampler') as sampler, \
                patch.object(ref, 'normalization_factors') as factors, \
                patch.object(ref, 'test_point_loss') as tp_loss:
            trained, _ = ref.train_fold_nn(model, ref._mlp_prepare, x, y,
                self.x[8:12], self.y[8:12], cfg, self.device)
        sampler.assert_not_called()
        factors.assert_not_called()
        tp_loss.assert_not_called()
        info = trained.ref_training_info
        self.assertIsNone(info['normalization'])
        self.assertEqual(info['history'][0]['sampled_parts'], 8)
        self.assertEqual(info['history'][0]['optimizer_steps'], 3)
        self.assertAlmostEqual(info['history'][0]['train_weight_mae_g'], expected_mae, places=5)
        self.assertAlmostEqual(info['history'][0]['train_l1_penalty'], expected_l1, places=6)
        hpo = {'search_space': {'batch_size': [2, 8]}}
        self.assertEqual(ref._nn_search_space(hpo, cfg), hpo['search_space'])

    def test_cv_and_hpo_recompute_training_only_factors_for_both_networks(self):
        hpo_cfg = {'search_space': {'batch_size': [1, 1]}, 'n_folds': 2,
                   'epochs': 1, 'patience': 2}
        group_cfg = {'enabled': True, 'protected_group_ids': [0, 11]}
        for kind in ('Encoder', 'MLP'):
            with self.subTest(kind=kind):
                observed = []
                original_train = ref.train_fold_nn
                def record_train(*args, **kwargs):
                    trained, mae = original_train(*args, **kwargs)
                    expected = shared.normalization_factors(args[3], args[8])
                    self.assertEqual(trained.ref_training_info['normalization'], expected)
                    self.assertFalse(set(args[8]) & set(args[9]))
                    self.assertTrue({0, 11}.issubset(set(args[8])))
                    self.assertFalse({0, 11} & set(args[9]))
                    self.assertLess(len(args[3]), len(self.y))
                    observed.append(expected)
                    return trained, mae
                trial = optuna.trial.FixedTrial({})
                original_evaluate = ref.evaluate_nn
                def check_holdout(*args, **kwargs):
                    self.assertFalse({0, 11} & set(args[5]))
                    return original_evaluate(*args, **kwargs)
                with patch.object(ref, kind + 'Model', TinyNN), patch.object(ref, 'train_fold_nn', side_effect=record_train), patch.object(ref, 'evaluate_nn', side_effect=check_holdout), redirect_stdout(io.StringIO()):
                    if kind == 'Encoder':
                        result = ref.run_cv_encoder(self.x, self.y, [1, 2], [1], [], [2],
                            0., tiny_config(), self.device, 42, group_values=self.ids, group_cfg=group_cfg)
                        ref._obj_encoder(trial, self.x, self.y, hpo_cfg, tiny_config(), 42,
                            self.device, None, self.ids, group_cfg)
                    else:
                        result = ref.run_cv_mlp(self.x, self.y, 1, [2], 0., tiny_config(),
                            self.device, 42, group_values=self.ids, group_cfg=group_cfg)
                        ref._obj_mlp(trial, self.x, self.y, hpo_cfg, tiny_config(), 42,
                            self.device, None, self.ids, group_cfg)
                self.assertEqual(len(observed), 4)
                self.assertNotEqual(observed[0]['s_w'], observed[1]['s_w'])
                self.assertEqual(observed[:2], observed[2:])
                self.assertIn('MAE_within', result[0])
                self.assertIn('training_info', result[3][0])
                self.assertIn('fold_0_training', trial.user_attrs)
                self.assertEqual(trial.user_attrs['fold_0_training']['training_config']['batch_size'], 4)

    def test_final_training_reuses_authoritative_split_for_both_networks(self):
        train_idx, val_idx, dev_idx = np.arange(12), np.arange(12, 20), np.arange(20)
        for kind in ('Encoder', 'MLP'):
            with self.subTest(kind=kind), patch.object(ref, kind + 'Model', TinyNN), redirect_stdout(io.StringIO()):
                kwargs = dict(group_values=self.ids, group_cfg={'enabled': True, 'protected_group_ids': [0, 5]},
                              fixed_train_idx=train_idx, fixed_val_idx=val_idx)
                if kind == 'Encoder':
                    model, _ = ref.final_train_encoder(self.x, self.y, dev_idx, [1, 2],
                        [1], [], [2], 0., tiny_config(), self.device, 42, **kwargs)
                else:
                    model, _ = ref.final_train_mlp(self.x, self.y, dev_idx, 1, [2],
                        0., tiny_config(), self.device, 42, **kwargs)
                info = model.ref_training_info
                self.assertEqual(info['train_indices'], train_idx.tolist())
                self.assertEqual(info['validation_indices'], val_idx.tolist())
                self.assertEqual(info['normalization'], shared.normalization_factors(
                    self.y[train_idx], self.ids[train_idx]))
                self.assertIn('MAE_within', info['best_validation_metrics'])
                self.assertIn('val_MAE_within_g', info['history'][0])

    def test_imported_validation_rejects_protected_groups_including_legacy_loss(self):
        for enabled in (False, True):
            cfg = {**tiny_config(), 'test_point_training': {'enabled': enabled}}
            kwargs = dict(group_values=self.ids,
                          group_cfg={'enabled': True, 'protected_group_ids': [6]},
                          fixed_train_idx=np.arange(12), fixed_val_idx=np.arange(12, 20))
            with self.subTest(enabled=enabled, kind='Encoder'), \
                    self.assertRaisesRegex(ValueError, 'protected group.*validation'):
                ref.final_train_encoder(self.x, self.y, np.arange(20), [1, 2], [1], [],
                                        [2], 0., cfg, self.device, 42, **kwargs)
            with self.subTest(enabled=enabled, kind='MLP'), \
                    self.assertRaisesRegex(ValueError, 'protected group.*validation'):
                ref.final_train_mlp(self.x, self.y, np.arange(20), 1, [2], 0.,
                                   cfg, self.device, 42, **kwargs)

    def test_ungrouped_cv_and_overlapping_final_test_points_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'prevent test-point leakage'):
            ref.run_cv_mlp(self.x, self.y, 1, [2], 0., tiny_config(), self.device, 42,
                           group_values=self.ids)
        with self.assertRaisesRegex(ValueError, 'more than one data partition'), redirect_stdout(io.StringIO()):
            ref.final_train_mlp(self.x, self.y, np.arange(20), 1, [2], 0., tiny_config(),
                self.device, 42, group_values=self.ids, group_cfg={'enabled': True},
                fixed_train_idx=np.arange(11), fixed_val_idx=np.arange(11, 20))

    def test_early_stopping_still_uses_absolute_validation_mae(self):
        cfg = {**tiny_config(), 'lr': 0., 'epochs': 5, 'patience': 1}
        with redirect_stdout(io.StringIO()):
            model, score = ref.train_fold_nn(TinyNN(), ref._mlp_prepare,
                self.x[:8], self.y[:8], self.x[8:12], self.y[8:12], cfg, self.device,
                self.ids[:8], self.ids[8:12])
        self.assertEqual(len(model.ref_training_info['history']), 2)
        self.assertEqual(score, model.ref_training_info['history'][0]['val_mae_g'])

    def test_metrics_use_complete_groups_for_neural_and_tree_models(self):
        actual = np.array([0., 2., 4., 6., 100., 102.])
        predicted = np.array([0., 0., 8., 8., 110., 112.])
        ids = ['a'] * 4 + ['b'] * 2
        class IdentityNN(nn.Module):
            def forward(self, x):
                return x[:, 0]
        class IdentityTree:
            def predict(self, x):
                return x[:, 0]
        neural = ref.evaluate_nn(IdentityNN(), ref._mlp_prepare,
                                 predicted[:, None], actual, self.device, ids)
        tree = ref.evaluate_gbdt(IdentityTree(), predicted[:, None], actual, ids)
        for result in (neural, tree):
            self.assertAlmostEqual(result['MAE_w'], 28/6)
            self.assertEqual(result['MAE_TPavg'], 5.5)
            self.assertEqual(result['MAE_within'], 1.)
            self.assertEqual(result['test_points'][0]['predicted_std_g'], 4.)
            self.assertAlmostEqual(result['test_points'][0]['actual_std_g'], np.sqrt(5))

    def test_tree_cv_retains_full_fold_fitting_and_adds_test_point_metrics(self):
        TinyTree.fits = []
        group_cfg = {'enabled': True, 'protected_group_ids': [0, 11]}
        with redirect_stdout(io.StringIO()):
            metrics, _, _, records = ref.run_cv_gbdt(TinyTree, {}, self.x, self.y,
                tiny_config(), 42, group_values=self.ids, group_cfg=group_cfg)
            ref._obj_gbdt(optuna.trial.FixedTrial({}), self.x, self.y,
                          {'search_space': {}, 'n_folds': 2}, TinyTree,
                          lambda trial, space: {}, 42, None, self.ids, group_cfg)
        self.assertEqual(len(TinyTree.fits), 4)
        for x, y in TinyTree.fits:
            self.assertEqual(len(y), 14)
            self.assertTrue({0, 11}.issubset(set(self.ids[x[:, 0].astype(int)])))
        self.assertIn('MAE_within', metrics)
        self.assertTrue(all(record['training_info'] is None for record in records))
        self.assertEqual(len(records[0]['test_points']), 5)


class ReferenceArtifactTests(unittest.TestCase):
    def test_nn_checkpoint_selection_ignores_test_mae(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            run_dir, overall = Path(directory) / 'run', Path(directory) / 'overall'
            run_dir.mkdir()
            overall.mkdir()
            (overall / 'best_metrics.json').write_text(json.dumps({'MAE': 0., 'validation_MAE': 2.}))
            model = TinyNN()
            metrics = {'MAE': 100., 'MSE': 1e4, 'RMSE': 100., 'R2': 0.,
                       'validation_MAE': 1., 'MAE_within': .3}
            self.assertTrue(ref.save_model_nn(model, metrics, run_dir, overall, {}, 'MLP'))
            saved = torch.load(overall / 'best_model.pt', weights_only=True)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(saved[key], value)
            metrics.update({'MAE': 0., 'validation_MAE': 3.})
            self.assertFalse(ref.save_model_nn(model, metrics, run_dir, overall, {}, 'MLP'))
            self.assertEqual(json.loads((overall / 'best_metrics.json').read_text())['validation_MAE'], 1.)

    def test_beta_sweeps_are_independent_per_model_with_shared_seeds(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(ref, 'BASE_OUT', Path(directory)), redirect_stdout(io.StringIO()):
            cfg = {'training': tiny_config(), 'encoder_hpo': {'sampler_seed': 7, 'search_space': {}},
                   'mlp_hpo': {'sampler_seed': 8, 'search_space': {}}}
            original_cfg = copy.deepcopy(cfg)
            for label, winner in (('Encoder', .25), ('MLP', 0.)):
                calls = []
                def fake_experiment(experiment):
                    beta = experiment['training']['within_test_point_beta']
                    calls.append(experiment)
                    run_dir, _ = ref._nn_output_dirs(label, 'DOE1', experiment['training'])
                    run_dir.mkdir(parents=True)
                    mae = 1. if beta == winner else 2.
                    (run_dir / 'best_metrics.json').write_text(json.dumps(
                        {'MAE': 100. if beta == winner else 0., 'validation_MAE': mae}))
                    (run_dir / 'data_processing.json').write_text(json.dumps({'train_part_ids': [1]}))
                    return {'validation_mae_g': mae, 'run_dir': str(run_dir), 'within_test_point_beta': beta}
                result = ref.run_nn_beta_experiments(label, cfg, 'DOE1', 42, fake_experiment, [0, .25])
                self.assertEqual(result['selected_beta'], winner)
                hpo_key = label.lower() + '_hpo'
                self.assertEqual(calls[0][hpo_key]['sampler_seed'], calls[1][hpo_key]['sampler_seed'])
                self.assertNotEqual(calls[0][hpo_key]['study_name'], calls[1][hpo_key]['study_name'])
                promoted = json.loads((Path(directory) / label / 'best_overall' / 'best_metrics.json').read_text())
                self.assertEqual(promoted['MAE'], 100.)
                model_root = Path(directory) / label
                for name in ('beta_mae_comparison.png', 'beta_mae_comparison.csv'):
                    self.assertTrue((model_root / 'beta_experiments' / ref._RUN_TS / name).exists())
                    self.assertTrue((model_root / 'best_overall' / name).exists())
            self.assertEqual(cfg, original_cfg)

    def test_beta_values_scalar_default_list_invalid_values_and_legacy_mode(self):
        self.assertEqual(ref.resolve_beta_values(.25), [.25])
        self.assertEqual(ref.resolve_beta_values([0, .1]), [0., .1])
        for invalid in ([], [-1], [float('nan')], [float('inf')], [0, 0]):
            with self.assertRaises(ValueError):
                ref.resolve_beta_values(invalid)
        with redirect_stdout(io.StringIO()):
            result = ref.run_nn_beta_experiments('MLP', {'training': {}}, 'PP', 42,
                                                 lambda config: 'legacy')
        self.assertEqual(result, 'legacy')
        with self.assertRaises(ValueError):
            ref.run_nn_beta_experiments('MLP', {'training': {}}, 'PP', 42, lambda config: None, [0])

    def test_synthetic_neural_runs_write_all_beta_artifacts_without_hpo_search(self):
        x = np.arange(24, dtype=np.float32).reshape(-1, 1)
        y = x[:, 0] ** 2
        ids = np.repeat(np.arange(12), 2)
        train_idx, val_idx, test_idx = np.arange(12), np.arange(12, 20), np.arange(20, 24)
        cfg = {'training': tiny_config(), 'data': {'id_col': 'part', 'group_col': 'TP'},
               'encoder_hpo': {'search_space': {'batch_size': [1, 1]}, 'sampler_seed': 7,
                               'n_trials': 1, 'n_startup_trials': 1, 'n_folds': 2},
               'mlp_hpo': {'search_space': {'batch_size': [1, 1]}, 'sampler_seed': 7,
                           'n_trials': 1, 'n_startup_trials': 1, 'n_folds': 2}}
        study = SimpleNamespace(best_trial=SimpleNamespace(params={}, number=0, value=.5))
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(ref, 'BASE_DIR', Path(directory)))
            stack.enter_context(patch.object(ref, 'BASE_OUT', Path(directory) / 'outputs'))
            stack.enter_context(patch.object(ref, '_run_hpo', return_value=study))
            stack.enter_context(patch.object(ref, 'EncoderModel', TinyNN))
            stack.enter_context(patch.object(ref, 'MLPModel', TinyNN))
            for name in ('save_cv_plots', 'save_cv_diagnostics', 'save_final_test_plots',
                         '_save_batch_test_artifacts', '_xai_gradcam_encoder', '_xai_shap_mlp', '_xai_ig_mlp'):
                stack.enter_context(patch.object(ref, name))
            stack.enter_context(redirect_stdout(io.StringIO()))
            for label in ('Encoder', 'MLP'):
                def run(experiment):
                    common = dict(group_values=ids, group_cfg={'enabled': True},
                                  fixed_train_idx=train_idx, fixed_val_idx=val_idx,
                                  part_ids=np.arange(100, 124))
                    if label == 'Encoder':
                        return ref.run_encoder(x, y, experiment, 'DOE1', 'DOE1', 42,
                            torch.device('cpu'), None, np.arange(20), test_idx, **common)
                    return ref.run_mlp(x, y, experiment, 'DOE1', 'DOE1', 42,
                        torch.device('cpu'), None, np.arange(20), test_idx, ['x'], **common)
                result = ref.run_nn_beta_experiments(label, cfg, 'DOE1', 42, run, [0, .25])
                for record in result['experiments']:
                    run_dir = Path(record['run_dir'])
                    beta_dir = ref._beta_label(record['within_test_point_beta'])
                    model_root = Path(directory) / 'outputs' / label
                    self.assertEqual(run_dir, model_root / 'run_best' / ref._RUN_TS / beta_dir)
                    self.assertTrue((model_root / 'best_overall' / beta_dir / 'best_model.pt').exists())
                    for name in ('best_model.pt', 'best_metrics.json', 'configuration.json',
                                 'normalization_factors.json', 'training_history.json', 'scalers.json',
                                 'test_metrics.json', 'validation_metrics.json', 'test_test_points.csv',
                                 'validation_test_points.csv', 'cv_fold_1_training.json',
                                 'cv_fold_1_test_points.csv', 'data_processing.json'):
                        self.assertTrue((run_dir / name).exists(), f'{label}: {name}')
                    split = json.loads((run_dir / 'data_processing.json').read_text())
                    self.assertEqual(split['train_part_ids'], list(range(100, 112)))
                    self.assertEqual(split['val_part_ids'], list(range(112, 120)))
                    self.assertEqual(split['test_part_ids'], list(range(120, 124)))
                    factors = json.loads((run_dir / 'normalization_factors.json').read_text())
                    self.assertEqual(factors, shared.normalization_factors(y[train_idx], ids[train_idx]))
                    selected = json.loads((run_dir / 'best_metrics.json').read_text())
                    self.assertEqual(selected['validation_MAE'], record['validation_mae_g'])
                    self.assertIn('MAE_within', selected)
                root = Path(directory) / 'outputs' / label / 'best_overall'
                self.assertTrue((root / 'beta_selection.json').exists())
                self.assertTrue((root / 'scalers.json').exists())

    def test_main_runs_neural_sweeps_and_each_tree_only_once(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg_path = Path(directory) / 'ref_config.json'
            cfg = {'data': {'target_col': 'weight'}, 'preprocessing': {'random_seed': 42, 'remove_target_outliers': False},
                   'training': tiny_config(), 'output_base_dir': directory}
            cfg_path.write_text(json.dumps(cfg))
            x = np.arange(24, dtype=np.float32).reshape(-1, 1)
            data = (np.arange(24), x, x, x[:, 0], None, ['x'], np.repeat(np.arange(12), 2), None)
            cfg['data']['group_split'] = {'enabled': True}
            cfg_path.write_text(json.dumps(cfg))
            with ExitStack() as stack:
                stack.enter_context(patch.object(ref, '_CFG_MAP', {'DOE1': str(cfg_path)}))
                stack.enter_context(patch.object(ref, 'load_data', return_value=data))
                stack.enter_context(patch.object(ref, 'BASE_OUT', Path(directory)))
                stack.enter_context(patch('sys.argv', ['ref', '--material', 'DOE1', '--beta-values', '0', '.25']))
                sweeps = stack.enter_context(patch.object(ref, 'run_nn_beta_experiments'))
                trees = [stack.enter_context(patch.object(ref, name)) for name in ('run_lgbm', 'run_xgboost')]
                stack.enter_context(redirect_stdout(io.StringIO()))
                ref.main()
            self.assertEqual(sweeps.call_count, 2)
            self.assertEqual([call.args[0] for call in sweeps.call_args_list], ['Encoder', 'MLP'])
            self.assertTrue(all(tree.call_count == 1 for tree in trees))

    def test_custom_selection_keeps_other_models_in_legacy_training_mode(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            cfg = {'data': {'target_col': 'weight', 'group_split': {'enabled': True}},
                   'preprocessing': {'random_seed': 42}, 'output_base_dir': directory,
                   'training': {**tiny_config(), 'test_point_training': {'enabled': False}},
                   'active_models': dict.fromkeys(('encoder', 'mlp', 'lgbm', 'xgboost', 'custom_gbdt'), 1)}
            path = root / 'config.json'
            path.write_text(json.dumps(cfg))
            x = np.arange(24, dtype=np.float32).reshape(-1, 1)
            stack.enter_context(patch.object(ref, '_CFG_MAP', {'DOE1': str(path)}))
            stack.enter_context(patch.object(ref, 'BASE_OUT', root))
            stack.enter_context(patch.object(ref, 'load_data', return_value=(
                np.arange(24), x, x, x[:, 0], None, ['x'], np.repeat(np.arange(12), 2), None)))
            stack.enter_context(patch('sys.argv', ['ref', '--material', 'DOE1', '--beta-values', '0', '.25']))
            runners = [stack.enter_context(patch.object(ref, name)) for name in
                       ('run_encoder', 'run_mlp', 'run_lgbm', 'run_xgboost')]
            def sweep(label, experiment, mat_dir, seed, run, override):
                self.assertEqual(label, 'CustomGBDT')
                self.assertEqual(override, [0, .25])
                return run(experiment)
            stack.enter_context(patch.object(ref, 'run_test_point_beta_experiments', side_effect=sweep))
            custom = stack.enter_context(patch.object(ref, 'run_custom_gbdt'))
            output = stack.enter_context(redirect_stdout(io.StringIO()))
            ref.main()
            for runner in runners:
                runner.assert_called_once()
                self.assertEqual(runner.call_args.args[2], cfg)
            custom.assert_called_once()
            self.assertTrue(custom.call_args.args[2]['training']['test_point_training']['enabled'])
            self.assertEqual(json.loads(path.read_text()), cfg)
            self.assertIn('5/5 reference model(s) complete', output.getvalue())

    def test_main_legacy_mode_runs_original_models_when_custom_is_disabled(self):
        for tp_cfg in ({'test_point_training': {'enabled': False, 'beta_values': [0, .25]}}, {}):
            for grouped in (False, True):
                with self.subTest(tp_cfg=tp_cfg, grouped=grouped), \
                        tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                    cfg_path = Path(directory) / 'ref_config.json'
                    cfg = {'data': {'target_col': 'weight', 'group_split': {'enabled': grouped}},
                           'preprocessing': {'random_seed': 42},
                           'training': {**tiny_config(), **tp_cfg},
                           'output_base_dir': directory,
                           'active_models': dict.fromkeys(
                               ('encoder', 'mlp', 'lgbm', 'xgboost', 'custom_gbdt'), 1)}
                    if not tp_cfg:
                        cfg['training'].pop('test_point_training')
                    cfg['active_models']['custom_gbdt'] = 0
                    cfg_path.write_text(json.dumps(cfg))
                    x = np.arange(24, dtype=np.float32).reshape(-1, 1)
                    ids = np.repeat(np.arange(12), 2) if grouped else None
                    stack.enter_context(patch.object(ref, '_CFG_MAP', {'DOE1': str(cfg_path)}))
                    stack.enter_context(patch.object(ref, 'load_data', return_value=(
                        np.arange(24), x, x, x[:, 0], None, ['x'], ids, None)))
                    stack.enter_context(patch.object(ref, 'BASE_OUT', Path(directory)))
                    stack.enter_context(patch('sys.argv', ['ref', '--material', 'DOE1']))
                    runners = [stack.enter_context(patch.object(ref, name)) for name in
                               ('run_encoder', 'run_mlp', 'run_lgbm', 'run_xgboost')]
                    custom = stack.enter_context(patch.object(ref, 'run_custom_gbdt'))
                    comparison = stack.enter_context(patch.object(ref, 'save_beta_mae_comparison'))
                    output = stack.enter_context(redirect_stdout(io.StringIO()))
                    ref.main()
                    for runner in runners:
                        runner.assert_called_once()
                        self.assertEqual(runner.call_args.args[2], cfg)
                    custom.assert_not_called()
                    comparison.assert_not_called()
                    self.assertIn('active_models.custom_gbdt = 0', output.getvalue())
                    self.assertIn('4/5 reference model(s) complete', output.getvalue())


if __name__ == '__main__':
    unittest.main()
