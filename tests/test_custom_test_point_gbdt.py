"""Tiny synthetic checks for exact-loss stochastic tree boosting, never dataset jobs."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import optuna
import torch

from src import Reference_Models_W_Pred as ref
from src.Utility import custom_test_point_gbdt as custom
from src.Utility import m1_test_point_training as shared


def config(**overrides):
    return {"batch_size": 8, "epochs": 3, "patience": 3, "print_every": 999,
            "scheduler_factor": .5, "scheduler_patience": 1, "k_folds": 2,
            "val_fraction": .25, "within_test_point_beta": .25,
            "test_point_training": {"enabled": True, "test_points_per_mini_batch": 4,
                                    "parts_per_test_point": 2}, **overrides}


def fixture():
    ids = np.repeat(np.arange(16), 2)
    x = np.column_stack([np.linspace(-2, 2, 32), np.tile([-1., 1.], 16)])
    y = 20 + x[:, 0] + .2 * x[:, 1]
    return x, y, ids


def train(beta=.25, **kwargs):
    x, y, ids = fixture()
    model = custom.CustomTestPointGBDT(learning_rate=.3, max_depth=2, min_samples_leaf=1)
    return custom.train_custom_gbdt(model, x[:16], y[:16], x[16:24], y[16:24],
                                    config(within_test_point_beta=beta, **kwargs), ids[:16], ids[16:24])[0]


class CustomLossTests(unittest.TestCase):
    def test_uses_exact_shared_loss_and_sampler(self):
        self.assertIs(custom.test_point_loss, shared.test_point_loss)
        self.assertIs(custom.TestPointMiniBatchSampler, shared.TestPointMiniBatchSampler)

    def test_pseudo_residuals_include_gradients_through_test_point_means(self):
        prediction = np.array([0., 0., 3., 1., 5.])
        actual = np.zeros(5)
        codes = np.array([0, 0, 0, 1, 1])
        factors = {'s_w': 4., 's_within': 2.}
        result, negative_gradient = custom.loss_and_negative_gradient(prediction, actual, codes, .5, factors)
        # Absolute gradient is sign(error)/N. Within gradients subtract the
        # per-TP mean sign, then divide by B*m_j (equal TP weighting).
        gradient = np.sign(prediction) / 5
        for tp in (0, 1):
            mask = codes == tp
            centered_sign = np.sign(prediction[mask] - prediction[mask].mean())
            gradient[mask] += (centered_sign - centered_sign.mean()) / (2 * mask.sum())
        np.testing.assert_allclose(negative_gradient, -5 * gradient)
        self.assertAlmostEqual(result['weight_mae_g'], 9/5)
        zero_result, zero_gradient = custom.loss_and_negative_gradient(prediction, actual, codes, 0, factors)
        np.testing.assert_array_equal(zero_gradient, -np.sign(prediction))
        self.assertEqual(zero_result['loss'], zero_result['weight_mae_g'])

    def test_exact_line_search_matches_shared_objective_at_all_breakpoints(self):
        prediction = np.array([1., 3., 8., 10., 13.])
        actual = np.array([0., 2., 10., 14., 18.])
        direction = np.array([-1., 2., 1., 3., 2.])
        codes = np.array([0, 0, 1, 1, 1])
        factors = {'s_w': 6., 's_within': 2.}
        def objective(gamma, beta):
            return shared.test_point_loss(torch.tensor(prediction + gamma*direction),
                torch.tensor(actual), torch.tensor(codes), beta, factors)[0].item()
        for beta in (0, .25, 1.):
            with self.subTest(beta=beta):
                gamma = custom.exact_tree_step(prediction, actual, direction, codes, beta, factors)
                errors = prediction - actual
                candidates = [0., *(-errors / direction)]
                for tp in (0, 1):
                    mask = codes == tp
                    offsets = errors[mask] - errors[mask].mean()
                    slopes = direction[mask] - direction[mask].mean()
                    valid = slopes != 0
                    candidates.extend((-offsets[valid] / slopes[valid]).tolist())
                best = min(objective(candidate, beta) for candidate in candidates if candidate >= 0)
                self.assertAlmostEqual(objective(gamma, beta), best)
                self.assertLessEqual(objective(.3*gamma, beta), objective(0, beta) + 1e-12)
        self.assertEqual(custom.exact_tree_step(prediction, actual, np.zeros(5), codes, 1, factors), 0.)

    def test_one_tree_update_never_increases_its_mini_batch_loss(self):
        x, y, ids = fixture()
        factors = shared.normalization_factors(y[:8], ids[:8])
        for beta in (0., .25, 1.):
            model = custom.CustomTestPointGBDT(learning_rate=.3, max_depth=2, min_samples_leaf=1)
            model._initialize(x[:8], y[:8])
            initial = model.predict(x[:8])
            result = model.boost_step(x[:8], y[:8], ids[:8], beta, factors, initial)
            self.assertLessEqual(result['loss_after_step'], result['loss'])
            self.assertEqual(model.update_attempts_, 1)
            self.assertLessEqual(len(model.trees_), 1)
            np.testing.assert_allclose(model.predict(x[:8]), initial + result['scale'] * result['tree'].predict(x[:8]))


class CustomTrainingTests(unittest.TestCase):
    def test_training_improves_training_mae_and_predicts_one_absolute_weight(self):
        model = train(beta=0)
        x, y, ids = fixture()
        initial_mae = np.abs(np.median(y[:16]) - y[:16]).mean()
        self.assertLess(np.abs(model.predict(x[:16]) - y[:16]).mean(), initial_mae)
        self.assertEqual(model.predict(x[:16]).shape, (16,))
        self.assertEqual(model.initial_weight_, np.median(y[:16]))
        self.assertTrue(len(model.trees_) > 0)
        self.assertTrue(all(info['optimizer_steps'] == 2 for info in model.ref_training_info['history']))
        self.assertTrue(all(info['sampled_parts'] == 16 for info in model.ref_training_info['history']))

    def test_sampling_identical_across_betas_and_factors_fixed_training_only(self):
        x, y, ids = fixture()
        observed, factors_seen = [], []
        original = custom.loss_and_negative_gradient
        def record(predicted, actual, codes, beta, factors):
            observed.append((actual.copy(), codes.copy()))
            factors_seen.append(factors)
            return original(predicted, actual, codes, beta, factors)
        for beta, validation_shift in ((0., 0.), (.5, 1000.)):
            model = custom.CustomTestPointGBDT(max_depth=2, min_samples_leaf=1)
            with patch.object(custom, 'loss_and_negative_gradient', side_effect=record):
                custom.train_custom_gbdt(model, x[:16], y[:16], x[16:24], y[16:24] + validation_shift,
                                         config(epochs=2, within_test_point_beta=beta), ids[:16], ids[16:24])
            self.assertEqual(model.ref_training_info['normalization'], shared.normalization_factors(y[:16], ids[:16]))
        self.assertEqual(len(observed), 8)
        self.assertTrue(all(factors is factors_seen[0] for factors in factors_seen[:4]))
        for first, second in zip(observed[:4], observed[4:]):
            np.testing.assert_array_equal(first[0], second[0])
            np.testing.assert_array_equal(first[1], second[1])
            np.testing.assert_array_equal(np.unique(first[1], return_counts=True)[1], [2]*4)

    def test_default_four_test_points_sixteen_distinct_parts(self):
        x = np.arange(96, dtype=float)[:, None]
        y = 20 + x[:, 0] * .01
        ids = np.repeat(np.arange(6), 16)
        observed = []
        original = custom.loss_and_negative_gradient
        def record(predicted, actual, codes, beta, factors):
            observed.append((actual.copy(), codes.copy()))
            return original(predicted, actual, codes, beta, factors)
        cfg = config(epochs=1, batch_size=64,
                     test_point_training={'enabled': True, 'test_points_per_mini_batch': 4,
                                          'parts_per_test_point': 16})
        with patch.object(custom, 'loss_and_negative_gradient', side_effect=record):
            custom.train_custom_gbdt(custom.CustomTestPointGBDT(), x[:64], y[:64], x[64:], y[64:],
                                     cfg, ids[:64], ids[64:])
        self.assertEqual(len(observed), 1)
        self.assertEqual(len(np.unique(observed[0][0])), 64)
        np.testing.assert_array_equal(np.unique(observed[0][1], return_counts=True)[1], [16]*4)

    def test_early_stop_and_checkpoint_restore_best_epoch_not_last(self):
        # Feed validation scores deliberately out of order, while actual trees
        # keep growing. This isolates selection from the combined training loss.
        original = custom.test_point_metrics
        scores = iter([3., 1., 2., 4.])
        def scripted_metrics(actual, predicted, ids):
            return {**original(actual, predicted, ids), 'MAE_w': next(scores)}
        with patch.object(custom, 'test_point_metrics', side_effect=scripted_metrics), redirect_stdout(io.StringIO()):
            model = train(epochs=10, patience=2)
        info = model.ref_training_info
        self.assertEqual(info['best_epoch'], 2)
        self.assertEqual(info['best_validation_mae_g'], 1.)
        self.assertEqual(len(info['history']), 4)
        self.assertEqual(len(model.trees_), info['history'][1]['n_trees'])
        self.assertLess(len(model.trees_), info['history'][-1]['n_trees'])

    def test_plateau_scheduler_reduces_tree_shrinkage(self):
        original = custom.test_point_metrics
        def constant_metrics(actual, predicted, ids):
            return {**original(actual, predicted, ids), 'MAE_w': 1.}
        with patch.object(custom, 'test_point_metrics', side_effect=constant_metrics):
            model = train(epochs=4, patience=5, scheduler_patience=0)
        rates = [row['learning_rate'] for row in model.ref_training_info['history']]
        np.testing.assert_allclose(rates, [.3, .3, .15, .075])

    def test_saved_model_has_identical_predictions_and_no_class_import_dependency(self):
        model = train()
        x, _, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'best_model.joblib'
            model.save(path)
            restored = custom.CustomTestPointGBDT.load(path)
            np.testing.assert_array_equal(model.predict(x), restored.predict(x))
            self.assertEqual(model.ref_training_info, restored.ref_training_info)
            self.assertEqual(model.get_params(), restored.get_params())
            self.assertEqual(len(model.feature_importances_), x.shape[1])

    def test_constant_training_targets_warn_and_produce_finite_zero_tree_model(self):
        x, _, ids = fixture()
        with self.assertWarns(shared.InsufficientVariabilityWarning):
            model = custom.CustomTestPointGBDT().fit(
                x[:16], np.full(16, 20.), test_point_ids=ids[:16],
                X_val=x[16:24], y_val=np.full(8, 20.), validation_test_point_ids=ids[16:24],
                training_config=config(epochs=1))
        self.assertEqual(len(model.trees_), 0)
        self.assertTrue(np.isfinite(model.predict(x)).all())
        np.testing.assert_array_equal(model.predict(x), np.full(32, 20.))
        self.assertEqual(model.ref_training_info['best_validation_mae_g'], 0.)

    def test_disabled_missing_leaky_or_invalid_training_inputs_are_rejected(self):
        x, y, ids = fixture()
        for cfg in (config(test_point_training={'enabled': False}), config(epochs=0),
                    config(within_test_point_beta=-1), config(batch_size=7)):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                custom.train_custom_gbdt(custom.CustomTestPointGBDT(), x[:16], y[:16], x[16:24], y[16:24],
                                         cfg, ids[:16], ids[16:24])
        with self.assertRaisesRegex(ValueError, 'both CustomGBDT training and validation'):
            custom.train_custom_gbdt(custom.CustomTestPointGBDT(), x[:16], y[:16], x[:8], y[:8],
                                     config(), ids[:16], ids[:8])
        with self.assertRaisesRegex(ValueError, 'learning_rate'):
            model = custom.CustomTestPointGBDT(learning_rate=1.1)
            custom.train_custom_gbdt(model, x[:16], y[:16], x[16:24], y[16:24], config(), ids[:16], ids[16:24])


class CustomPipelineTests(unittest.TestCase):
    def test_cv_hpo_recompute_training_only_factors_and_preserve_pruning(self):
        x, y, ids = fixture()
        group_cfg = {'enabled': True, 'protected_group_ids': [0, 15]}
        observed = []
        original = ref.train_custom_gbdt
        def record_training(*args, **kwargs):
            model, score = original(*args, **kwargs)
            expected = shared.normalization_factors(args[2], args[6])
            self.assertEqual(model.ref_training_info['normalization'], expected)
            self.assertFalse(set(args[6]) & set(args[7]))
            self.assertTrue({0, 15}.issubset(set(args[6])))
            self.assertFalse({0, 15} & set(args[7]))
            self.assertLess(len(args[2]), len(y))
            observed.append(expected)
            return model, score
        hpo = {'search_space': {'batch_size': [1, 1]}, 'n_folds': 2, 'epochs': 1, 'patience': 2}
        params = {'max_depth': 2, 'min_samples_leaf': 1}
        trial = optuna.trial.FixedTrial({})
        with patch.object(ref, 'train_custom_gbdt', side_effect=record_training), redirect_stdout(io.StringIO()):
            metrics, _, _, records = ref.run_cv_custom_gbdt(x, y, params, config(epochs=1), 42,
                group_values=ids, group_cfg=group_cfg)
            ref._obj_custom_gbdt(trial, x, y, hpo, config(epochs=1), params, 42,
                                  group_values=ids, group_cfg=group_cfg)
        self.assertEqual(len(observed), 4)
        self.assertNotEqual(observed[0]['s_w'], observed[1]['s_w'])
        self.assertEqual(observed[:2], observed[2:])
        self.assertIn('MAE_within', metrics)
        self.assertEqual(records[0]['training_info']['normalization'], observed[0])
        self.assertEqual(trial.user_attrs['fold_0_training']['normalization'], observed[0])
        class PrunedTrial(optuna.trial.FixedTrial):
            def should_prune(self):
                return True
        with self.assertRaises(optuna.TrialPruned), redirect_stdout(io.StringIO()):
            ref._obj_custom_gbdt(PrunedTrial({}), x, y, hpo, config(epochs=1), params, 42,
                                  group_values=ids, group_cfg={'enabled': True})

    def test_final_training_uses_authoritative_m1_ids_and_training_only_normalization(self):
        x, y, ids = fixture()
        tr_idx, val_idx, dev_idx = np.arange(16), np.arange(16, 24), np.arange(24)
        with redirect_stdout(io.StringIO()):
            model = ref.final_train_custom_gbdt(x, y, dev_idx, {'min_samples_leaf': 1},
                config(epochs=1), 42, group_values=ids, group_cfg={'enabled': True},
                fixed_train_idx=tr_idx, fixed_val_idx=val_idx)
        self.assertEqual(model.ref_training_info['train_indices'], tr_idx.tolist())
        self.assertEqual(model.ref_training_info['validation_indices'], val_idx.tolist())
        self.assertEqual(model.ref_training_info['normalization'], shared.normalization_factors(y[tr_idx], ids[tr_idx]))
        with self.assertRaisesRegex(ValueError, 'protected group.*validation'):
            ref.final_train_custom_gbdt(x, y, dev_idx, {}, config(epochs=1), 42,
                group_values=ids, group_cfg={'enabled': True, 'protected_group_ids': [8]},
                fixed_train_idx=tr_idx, fixed_val_idx=val_idx)
        with self.assertRaisesRegex(ValueError, 'more than one data partition'), redirect_stdout(io.StringIO()):
            ref.final_train_custom_gbdt(x, y, dev_idx, {}, config(epochs=1), 42,
                group_values=ids, group_cfg={'enabled': True}, fixed_train_idx=np.arange(15),
                fixed_val_idx=np.arange(15, 24))

    def test_synthetic_pipeline_writes_per_beta_and_selected_tree_artifacts(self):
        x, y, ids = fixture()
        cfg = {'training': config(epochs=1), 'data': {'id_col': 'part', 'group_col': 'TP'},
               'custom_gbdt': {'max_depth': 2, 'min_samples_leaf': 1},
               'custom_gbdt_hpo': {'n_trials': 0, 'search_space': {}, 'sampler_seed': 7}}
        tr_idx, val_idx, test_idx = np.arange(16), np.arange(16, 24), np.arange(24, 32)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(ref, 'BASE_DIR', Path(directory)))
            stack.enter_context(patch.object(ref, 'BASE_OUT', Path(directory) / 'outputs'))
            hpo = stack.enter_context(patch.object(ref, '_run_hpo'))
            for name in ('save_cv_plots', 'save_cv_diagnostics', 'save_final_test_plots', '_save_batch_test_artifacts'):
                stack.enter_context(patch.object(ref, name))
            stack.enter_context(redirect_stdout(io.StringIO()))
            original = copy.deepcopy(cfg)
            def run(experiment):
                return ref.run_custom_gbdt(x, y, experiment, 'DOE1', 'DOE1', 42, None,
                    np.arange(24), test_idx, ['pp1', 'pp2'], group_values=ids,
                    group_cfg={'enabled': True}, fixed_train_idx=tr_idx, fixed_val_idx=val_idx,
                    part_ids=np.arange(100, 132))
            selection = ref.run_test_point_beta_experiments('CustomGBDT', cfg, 'DOE1', 42, run, [0, .25])
            hpo.assert_not_called()
            self.assertEqual(cfg, original)
            for result in selection['experiments']:
                output = Path(result['run_dir'])
                for name in ('best_model.joblib', 'best_metrics.json', 'configuration.json',
                             'normalization_factors.json', 'training_history.json', 'model_parameters.json',
                             'validation_metrics.json', 'test_metrics.json', 'test_test_points.csv',
                             'validation_test_points.csv', 'cv_fold_1_training.json', 'cv_fold_metrics.csv',
                             'feature_importances.csv', 'scalers.json', 'data_processing.json'):
                    self.assertTrue((output / name).exists(), name)
                loaded = custom.CustomTestPointGBDT.load(output / 'best_model.joblib')
                metrics = json.loads((output / 'best_metrics.json').read_text())
                expected = shared.test_point_metrics(y[test_idx], loaded.predict(x[test_idx]), ids[test_idx])
                for name in ('MAE_w', 'MAE_TPavg', 'MAE_within'):
                    self.assertAlmostEqual(metrics[name], expected[name])
                self.assertEqual(metrics['validation_MAE'], result['validation_mae_g'])
                split = json.loads((output / 'data_processing.json').read_text())
                self.assertEqual(split['train_part_ids'], list(range(100, 116)))
                self.assertEqual(split['val_part_ids'], list(range(116, 124)))
                self.assertEqual(split['test_part_ids'], list(range(124, 132)))
            root = Path(directory) / 'outputs' / 'CustomGBDT' / 'best_overall'
            self.assertTrue((root / 'best_model.joblib').exists())
            self.assertTrue((root / 'beta_selection.json').exists())

    def test_custom_hpo_orchestration_replays_tree_params_without_changing_batch_size(self):
        x, y, ids = fixture()
        cfg = {'training': config(epochs=1), 'data': {}, 'custom_gbdt': {'min_samples_leaf': 1},
               'custom_gbdt_hpo': {'n_trials': 1, 'n_startup_trials': 1, 'n_folds': 2,
                   'epochs': 1, 'patience': 2, 'search_space': {
                       'learning_rate': [.1, .3], 'max_depth': [1, 2], 'batch_size': [1, 1]}}}
        trial_params = {'boost_learning_rate': .2, 'max_depth': 2}
        def fixed_hpo(study_name, n_trials, n_startup, objective, out_dir, seed, **kwargs):
            trial = optuna.trial.FixedTrial(trial_params)
            score = objective(trial)
            self.assertEqual(trial.user_attrs['fold_0_training']['training_config']['batch_size'], 8)
            return SimpleNamespace(best_trial=SimpleNamespace(params=trial_params, number=0, value=score))
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(ref, 'BASE_DIR', Path(directory)))
            stack.enter_context(patch.object(ref, 'BASE_OUT', Path(directory) / 'outputs'))
            hpo = stack.enter_context(patch.object(ref, '_run_hpo', side_effect=fixed_hpo))
            for name in ('save_cv_plots', 'save_cv_diagnostics', 'save_final_test_plots', '_save_batch_test_artifacts'):
                stack.enter_context(patch.object(ref, name))
            stack.enter_context(redirect_stdout(io.StringIO()))
            result = ref.run_custom_gbdt(x, y, cfg, 'DOE1', 'DOE1', 42, None,
                np.arange(24), np.arange(24, 32), ['p1', 'p2'], group_values=ids,
                group_cfg={'enabled': True}, fixed_train_idx=np.arange(16), fixed_val_idx=np.arange(16, 24))
            hpo.assert_called_once()
            model = custom.CustomTestPointGBDT.load(Path(result['run_dir']) / 'best_model.joblib')
            self.assertEqual(model.learning_rate, .2)
            self.assertEqual(model.max_depth, 2)
            effective = json.loads((Path(result['run_dir']) / 'configuration.json').read_text())
            self.assertEqual(effective['custom_gbdt'], model.get_params())
            self.assertEqual(model.ref_training_info['training_config']['batch_size'], 8)

    def test_beta_selection_ignores_test_results(self):
        cfg = {'training': config(), 'custom_gbdt_hpo': {'n_trials': 0, 'search_space': {}}}
        with tempfile.TemporaryDirectory() as directory, patch.object(ref, 'BASE_OUT', Path(directory)), redirect_stdout(io.StringIO()):
            def run(experiment):
                beta = experiment['training']['within_test_point_beta']
                output, _ = ref._test_point_output_dirs('CustomGBDT', 'DOE1', experiment['training'])
                output.mkdir(parents=True)
                val_mae = 1. if beta == .25 else 2.
                (output / 'best_metrics.json').write_text(json.dumps(
                    {'MAE': 100. if beta == .25 else 0., 'validation_MAE': val_mae}))
                (output / 'data_processing.json').write_text(json.dumps({'train_part_ids': [1]}))
                return {'validation_mae_g': val_mae, 'run_dir': str(output), 'within_test_point_beta': beta}
            selected = ref.run_test_point_beta_experiments('CustomGBDT', cfg, 'DOE1', 42, run, [0, .25])
            self.assertEqual(selected['selected_beta'], .25)
            root = Path(directory) / 'CustomGBDT' / 'best_overall'
            self.assertEqual(json.loads((root / 'best_metrics.json').read_text())['MAE'], 100.)

    def test_main_registers_custom_model_and_passes_shared_split(self):
        x, y, ids = fixture()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            path = Path(directory) / 'config.json'
            cfg = {'training': config(), 'data': {'target_col': 'weight', 'group_split': {'enabled': True}},
                   'preprocessing': {'random_seed': 42}, 'output_base_dir': directory,
                   'active_models': {'encoder': 1, 'mlp': 1, 'lgbm': 1, 'xgboost': 1, 'custom_gbdt': 1}}
            path.write_text(json.dumps(cfg))
            stack.enter_context(patch.object(ref, '_CFG_MAP', {'DOE1': str(path)}))
            stack.enter_context(patch.object(ref, 'BASE_OUT', Path(directory)))
            stack.enter_context(patch.object(ref, 'load_data', return_value=(
                np.arange(32), x, x, y, None, ['p1', 'p2'], ids, None)))
            stack.enter_context(patch('sys.argv', ['ref', '--material', 'DOE1', '--models', 'custom_gbdt', '--beta-values', '0', '.25']))
            def sweep(label, cfg, mat_dir, seed, run, override):
                self.assertEqual(label, 'CustomGBDT')
                self.assertEqual(override, [0, .25])
                return run(cfg)
            stack.enter_context(patch.object(ref, 'run_test_point_beta_experiments', side_effect=sweep))
            runner = stack.enter_context(patch.object(ref, 'run_custom_gbdt'))
            stack.enter_context(redirect_stdout(io.StringIO()))
            ref.main()
            runner.assert_called_once()
            args = runner.call_args.args
            self.assertEqual(set(ids[args[7]]) & set(ids[args[8]]), set())



if __name__ == '__main__':
    unittest.main()
