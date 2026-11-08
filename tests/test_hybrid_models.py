"""Small real fits on synthetic TPs. Never starts dataset experiments."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout, ExitStack
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src import M1_PPPT_to_W as m1
from src.Utility import engineered_features as ef
from src.Utility import hybrid_models as hm
from src.Utility import hybrid_pipeline as hp
from src.Utility.hybrid_data import read_json, write_json, load_data, split_data
from src.Utility.family_launcher import validate_pipeline_config, run_families
from src.Utility.dataset_layout import resolve_data_config, internal_test_name


def tiny_training():
    return {"batch_size": 8, "lr": .002, "weight_decay": 0., "l1_lambda": 0.,
            "epochs": 2, "patience": 2, "scheduler_patience": 1, "scheduler_factor": 1.,
            "print_every": 999, "k_folds": 2, "val_fraction": .25, "device": "cpu",
            "test_point_training": {"enabled": False}}


def fixture(root):
    groups = np.repeat(np.arange(16), 3)
    n = len(groups)
    x = groups.astype(float) / 16
    frame = pd.DataFrame({"ID": np.arange(n), "TP": groups, "pp_a": x,
                          "pp_b": np.sin(x), "weight": 10 + x + np.tile([-.02, 0, .02], 16)})
    frame.to_csv(root / "scalar.csv", index=False)
    sources = []
    for number, name in enumerate(("p1", "T1")):
        curve = np.array([0., 1., 3., 5., 4., 2., 1., 0.])
        values = (1 + x[:, None]) * curve + number * 5
        out = pd.DataFrame(values, columns=[f"t{i}" for i in range(8)])
        out.insert(0, "ID", np.arange(n))
        out.to_csv(root / f"{name}.csv", index=False)
        sources.append({"name": name, "path": str(root / f"{name}.csv")})
    data = {"scalar_csv": str(root / "scalar.csv"), "id_col": "ID", "target_col": "weight",
            "group_col": "TP", "group_split": {"enabled": True, "protected_group_ids": [0]},
            "timeseries_csvs": sources}
    cfg = {"data": data, "preprocessing": {"random_seed": 42, "test_fraction": .125},
           "training": {"val_fraction": .25, "early_stopping_rounds": 2},
           "xgboost": {"n_estimators": 8, "max_depth": 2, "learning_rate": .2,
                       "objective": "reg:absoluteerror", "tree_method": "hist", "n_jobs": 1},
           "optuna": {"enabled": True, "n_trials": 1, "n_startup_trials": 1, "sampler_seed": 17,
                      "n_folds": 2, "search_space": {"max_depth": [1, 2]}},
           "cv": {"enabled": True, "n_folds": 2, "plots": False}, "evaluation": {"enabled": False, "plots": False},
           "checkpoint": {"reuse": True}, "output": {"output_dir": str(root / "output")}}
    model_cfg = {"curve_encoders": [{"conv_channels": [1, 2], "conv_kernels": [3],
                                    "n_f_features": 2} for _ in sources],
                 "pp_hidden": [3], "pp_latent_dim": 2, "merge_hidden": [3], "dropout": 0.}
    return cfg, model_cfg


def write_time_grid(root, times, labels=None):
    path = root / "info" / "signal_time_grid.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"column": labels if labels is not None else [f"t{i}" for i in range(len(times))],
                  "time_seconds": times}).to_csv(path, index=False)
    return path


def divided_fixture(root):
    train, test = root/'Train', root/'Test'
    train.mkdir(); test.mkdir()
    cfg, model = fixture(train)
    for old, suffix in [('scalar.csv', 'Scalars.csv'), ('p1.csv', 'Pressure_1.csv'), ('T1.csv', 'Temperature.csv')]:
        new = train / ('Training_'+suffix)
        (train/old).rename(new)
        frame = pd.read_csv(new).iloc[:6].copy()
        if suffix == 'Scalars.csv':
            frame['weight'] += 100
        else:
            frame.loc[:, frame.columns != 'ID'] += .25
        frame.to_csv(test/('External_'+suffix), index=False)
        if suffix == 'Scalars.csv':
            cfg['data']['scalar_csv'] = str(new)
        else:
            for source in cfg['data']['timeseries_csvs']:
                if Path(source['path']).name == old:
                    source['path'] = str(new)
    cfg['data']['dataset_dir'] = str(root)
    cfg['evaluation']['test_2_dir'] = str(test)
    cfg['data'] = resolve_data_config(cfg['data'])
    write_time_grid(train, np.arange(8)*.1)
    write_time_grid(test, np.arange(8)*.1)
    return cfg, model


class EngineeredTests(unittest.TestCase):
    def test_discovers_dataset_grid_aligns_labels_and_preserves_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_time_grid(root, [2., 0., .5], ["t2", "t0", "t1"])
            source = {"name": "p1", "path": "signal.csv"}
            settings = ef.signal_time_options(source, {}, base_dir=root)
            self.assertEqual(settings["timestamps_csv"], str(root / "info" / "signal_time_grid.csv"))
            grid = ef.time_axis(3, settings, ["t0", "t1", "t2"])
            np.testing.assert_array_equal(grid, [0., .5, 2.])
            explicit = ef.signal_time_options(source, {"sampling_interval_s": 1.}, base_dir=root)
            self.assertNotIn("timestamps_csv", explicit)
            np.testing.assert_array_equal(ef.time_axis(3, explicit), [0., 1., 2.])
            disabled = ef.signal_time_options(source, {"auto_discover_time_grid": False}, base_dir=root)
            with self.assertRaisesRegex(ValueError, "Physical time metadata required"):
                ef.time_axis(3, disabled)
            for times, labels, error in (
                ([0., 1., 2.], ["t0", "t1", "other"], "exactly match"),
                ([0., 1., 2.], ["t0", "t1", "t1"], "unique"),
                ([0., 2., 1.], ["t0", "t1", "t2"], "strictly increasing"),
                ([0., 1.], ["t0", "t1"], "exactly match"),
            ):
                write_time_grid(root, times, labels)
                with self.subTest(error=error, labels=labels), self.assertRaisesRegex(ValueError, error):
                    ef.time_axis(3, settings, ["t0", "t1", "t2"])

    def test_discovered_grid_changes_invalidate_feature_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_time_grid(root, [0., .5, 1.])
            args = ([np.array([[0., 2., 0.]])], [{"name": "p", "path": "signal.csv"}], {}, [7])
            kwargs = dict(cache_dir=root / "cache", columns=[["t0", "t1", "t2"]],
                          base_dir=root, return_time_grids=True)
            first, names, provenance = ef.extract_features(*args, **kwargs)
            self.assertEqual(provenance["p"]["timestamps_csv"], str(path))
            with patch.object(ef, "curve_features", side_effect=AssertionError("cache miss")):
                cached, _, _ = ef.extract_features(*args, **kwargs)
            np.testing.assert_array_equal(first, cached)
            write_time_grid(root, [0., 1., 2.])
            second, _, provenance = ef.extract_features(*args, **kwargs)
            self.assertEqual(provenance["p"]["timestamps_s"], [0., 1., 2.])
            self.assertEqual(second[0, names.index("p__area")], 2 * first[0, names.index("p__area")])
            self.assertEqual(len(list((root / "cache").glob("*.npz"))), 2)

    def test_triangle_units_and_interpolated_crossings_on_actual_time(self):
        result = ef.curve_features([0, 10, 0], [0, 2, 4])
        self.assertEqual(list(result), ef.ENGINEERED_FEATURES)
        self.assertEqual(result["area"], 20.)
        self.assertEqual(result["time_weighted_mean"], 5.)
        self.assertEqual(result["peak_time_s"], 2.)
        self.assertAlmostEqual(result["rise_time_10_90_s"], 1.6)
        self.assertAlmostEqual(result["fall_time_90_10_s"], 1.6)
        self.assertEqual(result["max_rise_slope"], 5.)
        self.assertEqual(result["max_fall_slope"], -5.)
        self.assertAlmostEqual(result["rise_slope_10_90"], 5.)
        self.assertAlmostEqual(result["fall_slope_90_10"], -5.)

    def test_constant_missing_invalid_and_positive_baseline_integral(self):
        result = ef.curve_features([2, 2, 2], [0, 1, 3])
        self.assertEqual(result["area"], 6.)
        for key in ("area_above_baseline", "max_rise_slope", "max_fall_slope",
                    "rise_time_10_90_s", "fall_time_90_10_s"):
            self.assertEqual(result[key], 0.)
        self.assertTrue(np.isnan(ef.curve_features([0, 2, 4], [0, 1, 2])["fall_time_90_10_s"]))
        self.assertEqual(ef.curve_features([0, np.nan, 4], [0, 1, 2])["mean"], 2.)
        self.assertTrue(np.isnan(ef.curve_features([np.nan, np.nan], [0, 1])["max"]))
        self.assertEqual(ef.curve_features([-1, 1], [0, 2], {"baseline": 0})["area_above_baseline"], .5)
        with self.assertRaises(ValueError):
            ef.curve_features([1, float("inf")], [0, 1], {"invalid_values": "raise"})
        with self.assertRaises(ValueError):
            ef.time_axis(3, {})
        np.testing.assert_allclose(ef.time_axis(3, {"sampling_frequency_hz": 2}), [0, .5, 1])

    def test_cache_invalidates_for_curve_and_time_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            curves = [np.array([[0., 2., 0.]])]
            args = ([{"name": "p"}], {"sampling_interval_s": .5}, [10], directory)
            first, names = ef.extract_features(curves, *args)
            with patch.object(ef, "curve_features", side_effect=AssertionError("cache miss")):
                second, _ = ef.extract_features(curves, *args)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(names[0], "p__max")
            ef.extract_features([curves[0] * 2], *args)
            ef.extract_features(curves, args[0], {"sampling_interval_s": 1.}, *args[2:])
            self.assertEqual(len(list(Path(directory).glob("*.npz"))), 3)
            values, names = ef.extract_features(curves * 3,
                [{"name": name} for name in ("p1", "p2", "T1")],
                {"sampling_interval_s": .1}, [10])
            self.assertEqual(values.shape, (1, 54))
            self.assertEqual(len(set(names)), 54)


class HybridIntegrationTests(unittest.TestCase):
    def test_head_imputation_uses_training_only_and_ignores_independent_test_targets(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            cfg, _ = fixture(root)
            cfg["optuna"]["enabled"] = False
            cfg["cv"]["enabled"] = False
            dataset = load_data(cfg)
            splits, contract = split_data(dataset, cfg)
            train, val, test = splits
            dataset.pp[val, 0] = 1e9
            features = dataset.pp.copy()
            features[train[0], 0] = np.nan
            model, imputer = hm.train_xgb(cfg, dataset, splits, contract,
                features, dataset.pp_names, root / "first", 42)
            self.assertAlmostEqual(imputer.statistics_[0], np.nanmedian(features[train, 0]), places=6)
            before = model.predict(imputer.transform(features[val]))
            dataset.y[test] += 1e6
            features[test] -= 1e6
            after, _ = hm.train_xgb(cfg, dataset, splits, contract,
                features, dataset.pp_names, root / "second", 42)
            np.testing.assert_array_equal(before, after.predict(imputer.transform(features[val])))
            self.assertFalse((root / "first" / "test_metrics.json").exists())

    def test_hybrid_split_seed_changes_new_splits_and_preserves_imported_contracts(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            cfg, _ = fixture(root)
            cfg.update(family='gbdt_engineered', feature_extraction={
                'sampling_interval_s': .1, 'cache_dir': str(root / 'cache')})
            cfg['preprocessing']['split_seed'] = 17
            path = root / 'config.json'
            write_json(path, cfg)
            contracts = []
            for split_seed in (42, 43):
                run = root / f'new_{split_seed}'
                hp.run_hybrid('gbdt_engineered', path, run_dir=run, seed=99,
                              stages=('m1',), split_seed=split_seed)
                self.assertEqual(read_json(run / 'configuration.json')['preprocessing']['split_seed'], split_seed)
                contracts.append(read_json(run / 'data_processing.json'))
            self.assertNotEqual(contracts[0]['test_part_ids'], contracts[1]['test_part_ids'])
            cfg['split'] = {'source_dir': str(root / 'new_42')}
            write_json(path, cfg)
            hp.run_hybrid('gbdt_engineered', path, run_dir=root / 'imported', seed=101,
                          stages=('m1',), split_seed=123)
            imported = read_json(root / 'imported/data_processing.json')
            for key in ('train_part_ids', 'val_part_ids', 'test_part_ids'):
                self.assertEqual(imported[key], contracts[0][key])

    def test_all_families_train_reload_reuse_and_enable_final_testing(self):
        self._check_hybrid_families(False)

    def test_all_families_keep_external_test_separate_and_reuse_training_checkpoints(self):
        self._check_hybrid_families(True)

    def _check_hybrid_families(self, divided):
        torch.set_num_threads(1)
        for family in hp.FAMILIES:
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                root = Path(directory)
                cfg, mc = divided_fixture(root) if divided else fixture(root)
                cfg["family"] = family
                dataset = load_data(cfg)
                splits, contract = split_data(dataset, cfg)
                tr, va, te = splits
                # Held-out extremes must never affect fitted scalers/imputers or gradients.
                self.assertFalse(set(dataset.groups[tr]) & set(dataset.groups[te]))
                write_json(root / "split" / "data_processing.json", contract)
                write_json(root / "split" / "reference_run.json", {
                    "run_dir": "/original/runs/2026-11-05_12-00-00/run_001",
                    "run_timestamp": "2026-11-05_12-00-00", "fusion_test_MAE": .123})
                cfg["split"] = {"source_dir": str(root / "split")}
                if family == "gbdt_engineered":
                    write_time_grid(root/'Train' if divided else root, np.arange(8) * .1)
                    cfg["feature_extraction"] = {"auto_discover_time_grid": True,
                                                 "cache_dir": str(root / "cache")}
                else:
                    teacher_cfg = {"data": cfg["data"], "model": mc, "training": tiny_training(),
                                   "mode": "manual", "checkpoint": {"reuse": True},
                                   "evaluation": {"enabled": False, "plots": False,
                                                  "test_2_dir": cfg['evaluation'].get('test_2_dir')},
                                   "cv": {"enabled": True, "plots": False,
                                          "save_fold_checkpoints": family == "gbdt_ppdirect_pt", "n_folds": 2}}
                    if family == "gbdt_ppdirect_pt":
                        teacher_cfg["model"]["pp_branch"] = "direct"
                        cfg["teacher"] = {"train": True, "config": str(root / "m1.json")}
                        teacher_cfg.update(mode="optuna", optuna={"enabled": True, "n_trials": 1,
                            "n_startup_trials": 1, "sampler_seed": 3, "n_folds": 2, "epochs": 1,
                            "patience": 1, "search_space": {"lr": [.001, .002]}})
                        write_json(root / "m1.json", teacher_cfg)
                    else:
                        teacher = hm.train_teacher(teacher_cfg, dataset, splits, contract, root / "pretrained", 42)
                        cfg["teacher"] = {"train": False, "checkpoint_dir": str(root / "pretrained")}
                reconstruction = {"data": cfg["data"], "preprocessing": {"random_seed": 42},
                    "model": {"hidden_dims": [4], "dropout": 0.}, "training": tiny_training(),
                    "mode": "manual", "evaluation": {"enabled": False, "plots": False,
                        "test_2_dir": cfg['evaluation'].get('test_2_dir')}, "checkpoint": {"reuse": True}}
                if family == "gbdt_ppdirect_pt":
                    reconstruction.update(mode="optuna", normalization={"pp": "standard", "targets": "standard"},
                        cv={"enabled": True, "n_folds": 2, "plots": False}, optuna={"enabled": True, "n_trials": 1,
                            "n_startup_trials": 1, "sampler_seed": 4, "n_folds": 2, "epochs": 1,
                            "patience": 1, "search_space": {"lr": [.001, .002]}})
                fusion = {"evaluation": {"enabled": False, "plots": False},
                          "cv": {"enabled": True, "n_folds": 2, "plots": False},
                          "xai": {"enabled": family == "gbdt_engineered", "partition": "validation",
                                  "background_samples": 2, "explain_samples": 2, "nsamples": 16,
                                  "pathway_analysis": True}}
                fusion['evaluation']['test_2_dir'] = cfg['evaluation'].get('test_2_dir')
                write_json(root / "m2.json", reconstruction)
                write_json(root / "fusion.json", fusion)
                cfg["configs"] = {"m2": str(root / "m2.json"), "fusion": str(root / "fusion.json")}
                write_json(root / "head.json", cfg)
                run = root / "run"
                hp.run_hybrid(family, root / "head.json", run_dir=run)
                fusion_dir = run / "Fusion" / "run_best" / "model"
                predictor = hm.HybridPredictor.load(fusion_dir / "predictor")
                prediction = predictor.predict(pd.DataFrame(dataset.pp[va], columns=dataset.pp_names))
                self.assertEqual(prediction.shape, (len(va),))
                np.testing.assert_allclose(prediction, pd.read_csv(fusion_dir / "validation_predictions.csv")["y_pred"])
                self.assertFalse(any(run.rglob("test_metrics.json")))
                self.assertTrue((run / "comparison.csv").exists())
                self.assertEqual(read_json(run / "reference_run.json")["reference_run"]["run_timestamp"],
                                 "2026-11-05_12-00-00")
                for path in fusion_dir.glob("cv_fold_*_split.json"):
                    fold = read_json(path)
                    self.assertFalse(set(fold["heldout_part_ids"]) & set(dataset.ids[te]))
                    self.assertFalse(set(fold["heldout_part_ids"]) & set(fold["train_part_ids"]))
                    self.assertFalse(set(fold["heldout_part_ids"]) & set(fold["val_part_ids"]))
                self.assertEqual(len(list(fusion_dir.glob("cv_fold_*_split.json"))), 2)
                self.assertTrue(all(not p.requires_grad for p in predictor.reconstructor.parameters()))
                if family == "gbdt_engineered":
                    self.assertTrue((fusion_dir / "xai/train" / "xai_shap_single_pp_beeswarm.png").exists())
                    self.assertTrue((fusion_dir / "xai/train" / "xai_bar_shap_pathway_split.png").exists())
                    self.assertTrue((fusion_dir / "xai/train" / "tree_head_shap_values.csv").exists())
                    fusion["xai"]["enabled"] = False
                    write_json(root / "fusion.json", fusion)
                else:
                    self.assertIsInstance(predictor.teacher.model.merge, torch.nn.Identity)
                with self.assertRaises(ValueError):
                    predictor.predict(pd.DataFrame(dataset.pp[va], columns=list(reversed(dataset.pp_names))))
                with patch.object(hp, "train_teacher", side_effect=AssertionError("teacher retrained")), \
                        patch.object(hp, "train_xgb", side_effect=AssertionError("head retrained")), \
                        patch.object(hp, "train_m2", side_effect=AssertionError("M2 retrained")):
                    hp.run_hybrid(family, root / "head.json", run_dir=run)
                    # Enabling final testing must also evaluate reused checkpoints.
                    cfg["evaluation"]["enabled"] = True
                    reconstruction["evaluation"]["enabled"] = True
                    fusion["evaluation"]["enabled"] = True
                    write_json(root / "head.json", cfg)
                    write_json(root / "m2.json", reconstruction)
                    write_json(root / "fusion.json", fusion)
                    if family == "gbdt_ppdirect_pt":
                        teacher_cfg["evaluation"]["enabled"] = True
                        write_json(root / "m1.json", teacher_cfg)
                    hp.run_hybrid(family, root / "head.json", run_dir=run)
                    internal = internal_test_name(cfg['data'])
                    for stage in ("XGBoost", "Fusion") + (("M1",) if family != "gbdt_engineered" else ()):
                        stage_dir = run / stage / "run_best" / "model"
                        self.assertEqual(set(pd.read_csv(stage_dir / f"{internal}_predictions.csv")["part_id"]),
                                         set(dataset.ids[te]))
                    m2_predictions = run / "M2" / "run_best" / "model" / f"{internal}_feature_predictions.csv"
                    self.assertEqual(set(pd.read_csv(m2_predictions)["part_id"]), set(dataset.ids[te]))
                    if divided:
                        for stage in ('XGBoost', 'M2', 'Fusion') + (('M1',) if family != 'gbdt_engineered' else ()):
                            stage_dir = run/stage/'run_best/model'
                            report = read_json(stage_dir/'test_2_metrics.json')
                            name = 'test_2_feature_predictions.csv' if stage == 'M2' else 'test_2_predictions.csv'
                            self.assertEqual(pd.read_csv(stage_dir/name)['part_id'].tolist(), list(range(6)))
                        before = pd.read_csv(fusion_dir/'test_2_predictions.csv')['y_pred'].to_numpy()
                        ext_path = root/'Test/External_Scalars.csv'
                        ext_frame = pd.read_csv(ext_path); ext_frame['weight'] += 1000
                        ext_frame.to_csv(ext_path, index=False)
                        hp.run_hybrid(family, root/'head.json', run_dir=run)
                        np.testing.assert_array_equal(before, pd.read_csv(fusion_dir/'test_2_predictions.csv')['y_pred'])
                        # Independent stage controls: only Fusion uses a different external folder.
                        alternate = root/'AlternateTest'
                        alternate.mkdir()
                        for path in (root/'Test').glob('*.csv'):
                            pd.read_csv(path).iloc[:3].to_csv(alternate/path.name, index=False)
                        write_time_grid(alternate, np.arange(8)*.1)
                        cfg['evaluation']['test_2_dir'] = None
                        reconstruction['evaluation']['test_2_dir'] = None
                        fusion['evaluation']['test_2_dir'] = str(alternate)
                        write_json(root/'head.json', cfg)
                        write_json(root/'m2.json', reconstruction)
                        write_json(root/'fusion.json', fusion)
                        if family == 'gbdt_ppdirect_pt':
                            teacher_cfg['evaluation']['test_2_dir'] = None
                            write_json(root/'m1.json', teacher_cfg)
                        hp.run_hybrid(family, root/'head.json', run_dir=run)
                        for stage in ('XGBoost', 'M2', 'M1'):
                            self.assertFalse((run/stage/'run_best/model/test_2_metrics.json').exists())
                            self.assertFalse((run/stage/'best_overall/test_2_metrics.json').exists())
                        self.assertEqual(pd.read_csv(fusion_dir/'test_2_predictions.csv')['part_id'].tolist(), [0,1,2])
                        fusion['evaluation']['test_2_dir'] = None
                        write_json(root/'fusion.json', fusion)
                        hp.run_hybrid(family, root/'head.json', run_dir=run)
                        self.assertFalse((fusion_dir/'test_2_metrics.json').exists())
                        self.assertFalse((run/'Fusion/best_overall/test_2_metrics.json').exists())
                        self.assertNotIn('test_2', pd.read_csv(run/'comparison.csv')['partition'].tolist())
                    if family == "gbdt_engineered":
                        cfg["checkpoint"]["source_dir"] = str(run / "XGBoost" / "best_overall")
                        reconstruction["checkpoint"]["source_dir"] = str(run / "M2" / "best_overall")
                        write_json(root / "head.json", cfg)
                        write_json(root / "m2.json", reconstruction)
                        hp.run_hybrid(family, root / "head.json", run_dir=root / "resumed_elsewhere")
                        self.assertTrue((root / "resumed_elsewhere" / "comparison.csv").exists())
                        write_time_grid(root/'Train' if divided else root, np.arange(8) * .2)
                        with self.assertRaisesRegex(ValueError, "checkpoint source.*incompatible"):
                            hp.run_hybrid(family, root / "head.json", run_dir=run)
                if family != "gbdt_engineered":
                    notes = read_json(run / "XGBoost" / "run_best" / "model" / "cv_interpretation.json")["notes"]
                    self.assertTrue(any("fold-specific" in n for n in notes) if family == "gbdt_ppdirect_pt"
                                    else any("not an unbiased" in n for n in notes))

    def test_original_m1_m2_training_supports_validation_only_evaluation(self):
        self._check_original_evaluation(False)

    def test_original_m1_m2_fusion_final_evaluation_uses_test_ids(self):
        self._check_original_evaluation(True)

    def test_original_sweep_selects_validation_winner_before_all_candidate_tests_and_xai(self):
        self._check_original_evaluation(True, alpha_sweep=True)

    def test_original_raw_loss_sweep_and_saved_evaluation_use_raw_composite(self):
        self._check_original_evaluation(True, alpha_sweep=True, divided=True, normalize_loss=False)

    def test_original_sweep_can_select_pretrained_fusion_with_all_candidate_tests_and_xai(self):
        self._check_original_evaluation(True, alpha_sweep=True, baseline_wins=True)

    def test_original_train_pipeline_and_independent_external_test_with_overlapping_ids(self):
        self._check_original_evaluation(True, alpha_sweep=True, divided=True)

    def test_original_explicit_pretrained_fusion_evaluates_without_finetuning(self):
        self._check_original_evaluation(True, pretrained_only=True)

    def _check_original_evaluation(self, test_enabled, alpha_sweep=False, baseline_wins=False, divided=False,
                                   pretrained_only=False, normalize_loss=True):
        from src import M2_PP_to_F as m2
        from src import Fusion_M1M2_WPred as original_fusion
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            cfg, mc = divided_fixture(root) if divided else fixture(root)
            cfg["evaluation"]["enabled"] = test_enabled
            cfg.update(model=mc, training=tiny_training(), mode="manual")
            cfg["preprocessing"].update(iqr_multiplier=1.5, remove_target_outliers=False)
            output = root / "M1"
            cfg["output"] = {"output_dir": str(output), "run_best_dir": str(output / "run_best"),
                             "best_overall_dir": str(output / "best_overall")}
            stack.enter_context(patch.object(m1, 'BASE_DIR', root))
            stack.enter_context(patch.object(m2, 'BASE_DIR', root))
            for module in (m1, m2):
                for name in ('save_cv_plots', 'save_cv_diagnostics', 'save_final_test_plots', 'save_batch_test_diagnostics'):
                    stack.enter_context(patch.object(module, name))
            stack.enter_context(redirect_stdout(io.StringIO()))
            m1.run_experiment(cfg, "DOE1", "synthetic")
            metadata = read_json(output / "best_overall" / "best_metrics.json")
            self.assertEqual(metadata["source"], ("final_test_1" if divided else "final_test") if test_enabled else "validation")
            self.assertIn("validation_MAE", metadata)
            second = {"data": {**cfg["data"], "features_csv": str(output / "best_overall" / "sensors_features_f.csv"),
                               "drop_cols": ["weight"]}, "preprocessing": cfg["preprocessing"],
                      "model": {"hidden_dims": [4], "dropout": 0.}, "training": tiny_training(),
                      "mode": "manual", "evaluation": {"enabled": test_enabled,
                          "test_2_dir": cfg['evaluation'].get('test_2_dir')},
                      "output": {"output_dir": str(root / "M2"), "run_best_dir": str(root / "M2" / "run_best"),
                                 "best_overall_dir": str(root / "M2" / "best_overall")}}
            write_json(root / "m2.json", second)
            stack.enter_context(patch.object(m2, '_CFG_MAP', {"DOE1": str(root / "m2.json")}))
            stack.enter_context(patch('sys.argv', ['m2', '--material', 'DOE1']))
            m2.main()
            metadata = read_json(root / "M2" / "best_overall" / "best_metrics.json")
            self.assertEqual(metadata["source"], ("final_test_1" if divided else "final_test") if test_enabled else "validation")
            self.assertIn("validation_NMAE", metadata)
            fusion_cfg = {"data": cfg["data"], "preprocessing": {"random_seed": 42},
                "models": {"m1_best_dir": str(output / "best_overall"),
                           "m2_best_dir": str(root / "M2" / "best_overall")},
                "output": {"output_base_dir": str(root / "Fusion")},
                "evaluation": {"enabled": test_enabled, "plots": False,
                    "test_2_dir": cfg['evaluation'].get('test_2_dir')}, "xai": {"enabled": alpha_sweep},
                "finetuning": {"alpha": [0, .5, 1] if alpha_sweep else [0], "epochs": 1, "lr": .001,
                               "batch_size": 8, "patience": 1, "val_fraction": None,
                               "normalize_loss": normalize_loss}}
            write_json(root / "config" / "ProBayes" / "Fusion_PP_config.json", fusion_cfg)
            stack.enter_context(patch.object(original_fusion, 'BASE_DIR', root))
            for name in ('_save_scatter', '_save_residuals_scatter', '_save_residuals_hist',
                         'save_batch_test_diagnostics'):
                stack.enter_context(patch.object(original_fusion, name))
            if alpha_sweep:
                events = []
                actual_train, actual_evaluate = original_fusion.finetune_m2, original_fusion.evaluate_fusion
                active = {"alpha": None}
                def train_alpha(*args, **kwargs):
                    alpha = args[6]
                    active["alpha"] = alpha
                    events.append(("train", alpha))
                    info = actual_train(*args, **kwargs)
                    # Composite loss favors alpha=0; weight MAE must choose .5.
                    info["best_val_loss"] = .1 if alpha == 0 else .9
                    return info
                def evaluate_alpha(model, X, y, device, partition="test"):
                    events.append((partition, active["alpha"]))
                    report = actual_evaluate(model, X, y, device, partition=partition)
                    if partition == "validation":
                        report["MAE"] = ({0.: .3, .5: .1, 1.: .2}[active["alpha"]]
                                         if active["alpha"] is not None else .05 if baseline_wins else .4)
                    return report
                stack.enter_context(patch.object(original_fusion, 'finetune_m2', side_effect=train_alpha))
                stack.enter_context(patch.object(original_fusion, 'evaluate_fusion', side_effect=evaluate_alpha))
                xai = stack.enter_context(patch.object(original_fusion, 'run_fusion_xai'))
            if pretrained_only:
                stack.enter_context(patch.object(original_fusion, 'finetune_m2',
                    side_effect=AssertionError('Pretrained-only Fusion must not fine-tune')))
            cli = ['fusion', '--material', 'PP'] + (['--pretrained-only'] if pretrained_only else [])
            with patch('sys.argv', cli):
                original_fusion.main()
            partition = internal_test_name(cfg['data']) if test_enabled else "validation"
            reports = list((root / "Fusion").rglob(f"best_fusion/{partition}_predictions.csv"))
            self.assertEqual(len(reports), 1)
            predicted_ids = set(pd.read_csv(reports[0])["ID_Part"])
            contract = read_json(output / "best_overall" / "data_processing.json")
            self.assertEqual(predicted_ids, set(contract["test_part_ids" if test_enabled else "val_part_ids"]))
            self.assertFalse(predicted_ids & set(contract["val_part_ids" if test_enabled else "test_part_ids"]))
            reconstruction_dir = reports[0].parent / 'm2_reconstruction'
            reconstruction = read_json(reconstruction_dir / f'{partition}_metrics.json')
            feature_predictions = pd.read_csv(reconstruction_dir / f'{partition}_feature_predictions.csv')
            self.assertEqual(set(feature_predictions['ID_Part']), predicted_ids)
            selected_m2 = read_json(reports[0].parent / 'm2_best_metrics.json')
            self.assertAlmostEqual(selected_m2['mean_MAE'], reconstruction['mean_MAE'])
            self.assertAlmostEqual(selected_m2['mean_NMAE'], reconstruction['mean_NMAE'])
            fusion_metrics = read_json(reports[0].parent / f'{partition}_metrics.json')
            self.assertAlmostEqual(fusion_metrics['feature_MAE'], reconstruction['mean_MAE'], places=5)
            if not normalize_loss:
                self.assertNotIn('normalized_MAE', fusion_metrics)
                self.assertNotIn('normalized_feature_MAE', fusion_metrics)
                alpha = fusion_metrics['alpha']
                self.assertAlmostEqual(fusion_metrics['composite_loss'],
                    alpha * fusion_metrics['MAE'] + (1-alpha) * fusion_metrics['feature_MAE'])
            if not test_enabled:
                self.assertFalse(any(root.rglob("test_metrics.json")))
            if pretrained_only:
                best = reports[0].parent
                selection = read_json(best / 'selection.json')
                self.assertEqual(selection['source_folder'], 'no_finetuning')
                self.assertEqual(selection['n_alphas'], 0)
                self.assertFalse(selection['fine_tuned'])
                self.assertFalse(any(path.is_dir() for path in best.parent.glob('alpha_*')))
                original_m2 = torch.load(root / 'M2/best_overall/best_model.pt', weights_only=True)
                evaluated_m2 = torch.load(best / 'm2_best_model.pt', weights_only=True)
                for key in original_m2:
                    torch.testing.assert_close(original_m2[key], evaluated_m2[key], rtol=0, atol=0)
            if alpha_sweep:
                best = reports[0].parent
                selection = read_json(best / "selection.json")
                self.assertEqual(selection["alpha"], None if baseline_wins else .5)
                self.assertEqual(selection["candidate"], 'no_finetuning' if baseline_wins else 'alpha_0.5')
                self.assertEqual(selection["fine_tuned"], not baseline_wins)
                self.assertEqual(selection["n_alphas"], 3)
                self.assertEqual(selection["n_candidates"], 4)
                self.assertEqual(selection["selection_metric"], "validation_MAE")
                self.assertFalse(selection["test_used_for_selection"])
                self.assertEqual([event for event in events if event[0] == "train"],
                                 [("train", 0.), ("train", .5), ("train", 1.)])
                self.assertEqual(len([event for event in events if event[0] == "test"]), 4)
                self.assertGreater(next(i for i, event in enumerate(events) if event[0] == "test"),
                                   max(i for i, event in enumerate(events) if event[0] == "train"))
                self.assertEqual(len(list((root / "Fusion").rglob(f"{partition}_metrics.json"))), 10)
                for folder in ('no_finetuning', 'alpha_0.0', 'alpha_0.5', 'alpha_1.0'):
                    candidate_dir = best.parent / folder
                    self.assertTrue((candidate_dir / f'{partition}_predictions.csv').exists())
                    actual = read_json(candidate_dir / f'{partition}_metrics.json')
                    self.assertEqual(actual['candidate'], folder)
                    self.assertTrue((candidate_dir / 'xai' / partition / 'explained_samples.csv').exists())
                    if divided:
                        self.assertTrue((candidate_dir / 'test_2_metrics.json').exists())
                baseline_predictions = pd.read_csv(best.parent / 'no_finetuning' / 'validation_predictions.csv')
                self.assertEqual(set(baseline_predictions['ID_Part']), set(contract['val_part_ids']))
                baseline_info = read_json(best.parent / 'no_finetuning' / 'finetuning_info.json')
                self.assertFalse(baseline_info['enabled'])
                self.assertEqual(baseline_info['epochs_run'], 0)
                self.assertEqual(xai.call_count, 12 if divided else 8)
                winner_index = ['no_finetuning', 'alpha_0.0', 'alpha_0.5', 'alpha_1.0'].index(selection['source_folder'])
                calls_per_candidate = 3 if divided else 2
                internal_call = xai.call_args_list[winner_index * calls_per_candidate + 1]
                self.assertEqual(internal_call.args[9], best.parent/selection['source_folder']/'xai'/partition)
                self.assertEqual(set(internal_call.args[3]), set(contract["test_part_ids"]))
                if divided:
                    self.assertEqual(set(xai.call_args_list[winner_index * calls_per_candidate + 2].args[3]), set(range(6)))
                    self.assertEqual(pd.read_csv(best/'test_2_predictions.csv')['part_id'].tolist(), list(range(6)))
                    self.assertEqual(pd.read_csv(output/'best_overall/test_2_predictions.csv')['part_id'].tolist(), list(range(6)))
                    self.assertEqual(read_json(root/'M2/best_overall/test_2_metrics.json')['n_samples'], 6)
                selected_state = torch.load(best / "fusion_model.pt", weights_only=True)
                evaluated_state = xai.call_args.args[0].state_dict()
                for key, value in selected_state.items():
                    torch.testing.assert_close(value, evaluated_state[key])
                with torch.inference_mode():
                    selected_model = xai.call_args.args[0]
                    selected_pp = torch.tensor(internal_call.args[2], dtype=torch.float32)
                    _, selected_features = selected_model(selected_pp, selected_pp, return_features=True)
                feature_columns = [name for name in feature_predictions.columns if name.endswith('_pred')]
                sampled_predictions = pd.DataFrame(selected_features.numpy(), index=internal_call.args[3]).loc[feature_predictions['ID_Part']].to_numpy()
                np.testing.assert_allclose(feature_predictions[feature_columns].to_numpy(), sampled_predictions,
                                           rtol=1e-6, atol=1e-6)
                comparison = pd.read_csv(best.parent / "alpha_comparison.csv")
                self.assertEqual(comparison.loc[comparison["selected"], "candidate"].tolist(),
                                 ['no_finetuning' if baseline_wins else 'alpha_0.5'])
                self.assertEqual(set(comparison["source"]), {"validation"})
                if baseline_wins:
                    original_m2 = torch.load(root / 'M2' / 'best_overall' / 'best_model.pt', weights_only=True)
                    restored_m2 = torch.load(best / 'm2_best_model.pt', weights_only=True)
                    for key, value in original_m2.items():
                        torch.testing.assert_close(value, restored_m2[key], rtol=0, atol=0)
                    self.assertIsNone(read_json(best / f'{partition}_metrics.json')['composite_loss'])
                    self.assertEqual(read_json(best / 'm2_best_metrics.json')['source'], 'fusion_pretrained')
                from src.Utility.family_launcher import _original_comparison
                _original_comparison(root, 'DOE1', 42)
                combined = pd.read_csv(root / 'comparison.csv')
                fusion_rows = combined.loc[combined['role'] == 'Fusion (PP only)']
                self.assertEqual(len(fusion_rows), 14 if divided else 9)
                self.assertEqual((fusion_rows['partition'] == partition).sum(), 5)
                self.assertEqual(((fusion_rows['partition'] == partition) & fusion_rows['selected']).sum(), 1)
                selected_validation = fusion_rows.loc[
                    (fusion_rows['partition'] == 'validation') & (fusion_rows['selected'] == True)]
                self.assertEqual(selected_validation['candidate'].tolist(),
                                 ['no_finetuning' if baseline_wins else 'alpha_0.5'])

                if divided:
                    checkpoint_bytes = {path: path.read_bytes() for path in best.parent.rglob('fusion_model.pt')}
                    selection_bytes = (best.parent / 'best_alpha_selection.json').read_bytes()
                    with patch.object(original_fusion, 'train_alpha_sweep') as retrain, \
                            patch('sys.argv', ['fusion', '--material', 'PP', '--evaluate-run', str(best.parent)]):
                        original_fusion.main()
                    retrain.assert_not_called()
                    self.assertEqual((best.parent / 'best_alpha_selection.json').read_bytes(), selection_bytes)
                    for path, contents in checkpoint_bytes.items():
                        self.assertEqual(path.read_bytes(), contents)
                    self.assertEqual(xai.call_count, 24)
                    if not normalize_loss:
                        reevaluated = read_json(best / f'{partition}_metrics.json')
                        self.assertAlmostEqual(reevaluated['composite_loss'], fusion_metrics['composite_loss'])
                        self.assertNotIn('normalized_MAE', reevaluated)

    def test_direct_pp_model_has_no_pp_parameters_and_ignores_pp_hpo(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, mc = fixture(Path(directory))
            mc["pp_branch"] = "direct"
            model = m1.M1Model(2, mc)
            self.assertIsInstance(model.pp_mlp, torch.nn.Identity)
            mu, _ = m1.suggest_hyperparams(__import__('optuna').trial.FixedTrial({}),
                {"pp_n_layers": [2, 3], "pp_n_features": [4, 8]}, 2, mc)
            self.assertFalse(any(key.startswith("pp_") for key in mu))

    def test_family_launcher_counts_full_repetitions_sequentially_and_offsets_seeds(self):
        cfg = {"data_split": {"seed": 17, "seed_step": 2},
               "models": {"gbdt_engineered": {"enabled": True, "n_runs": 2}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "cfg.json"
            write_json(path, {"preprocessing": {"random_seed": 42}, "output": {"output_dir": str(root)}})
            with patch('src.Utility.family_launcher.default_config', return_value=path), \
                    patch('src.Utility.family_launcher.run_hybrid') as run, redirect_stdout(io.StringIO()):
                self.assertFalse(run_families(cfg, ["DOE1"], optuna_seed=7))
            self.assertEqual(run.call_count, 2)
            self.assertEqual([c.args[4] for c in run.call_args_list], [42, 1051])
            self.assertEqual([c.args[5] for c in run.call_args_list], [7, 1016])
            self.assertEqual([c.kwargs["split_seed"] for c in run.call_args_list], [17, 19])
        with self.assertRaises(ValueError):
            validate_pipeline_config({"models": {}, "epochs": 100})
        with self.assertRaises(ValueError):
            validate_pipeline_config({"models": {"gbdt_pppt": {"n_runs": 2, "lr": .01}}})

    def test_pipeline_preflight_uses_discovered_grid_before_starting_any_family(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            model, _ = fixture(root)
            model["feature_extraction"] = {"auto_discover_time_grid": True}
            path = root / "head.json"
            write_json(path, model)
            write_time_grid(root, np.arange(8) * .1)
            selection = {"models": {"gbdt_engineered": {"enabled": True, "n_runs": 1}}}
            with patch('src.Utility.family_launcher.default_config', return_value=path), \
                    patch('src.Utility.family_launcher.run_hybrid') as runner:
                self.assertFalse(run_families(selection, ["DOE1"]))
                runner.assert_called_once()
            write_time_grid(root, np.arange(7) * .1)
            selection["models"]["original"] = {"enabled": True, "n_runs": 1}
            with patch('src.Utility.family_launcher.default_config', return_value=path), \
                    patch('src.Utility.family_launcher.run_hybrid') as runner, \
                    patch('src.Utility.family_launcher._run_original') as original:
                with self.assertRaisesRegex(ValueError, "Time-grid column labels"):
                    run_families(selection, ["DOE1"])
                runner.assert_not_called()
                original.assert_not_called()


if __name__ == '__main__':
    unittest.main()
