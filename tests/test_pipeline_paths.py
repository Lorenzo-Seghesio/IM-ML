"""Flat DoE1 routing and first-launch dependencies; no training subprocesses."""
import io
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from contextlib import redirect_stdout
from unittest.mock import patch

from src.Utility import family_launcher as launcher
from src.Utility import pipeline_runtime as runtime
from src.Utility.hybrid_data import ROOT, read_json, write_json
from src.Utility.hybrid_models import stage_dirs


class PipelinePathTests(unittest.TestCase):
    def test_original_stage_counts_require_nonnegative_integers_and_are_original_only(self):
        for stage in ('m1', 'm2', 'fusion'):
            key = f'n_{stage}_run_per_pipeline_run'
            for count in (-1, True, 1.5, '2', None):
                with self.subTest(stage=stage, count=count), self.assertRaisesRegex(ValueError, 'nonnegative integer'):
                    launcher.validate_pipeline_config({'models': {'original': {key: count}}})
            launcher.validate_pipeline_config({'models': {'original': {
                'n_m1_run_per_pipeline_run': 0, 'n_m2_run_per_pipeline_run': 0,
                'n_fusion_run_per_pipeline_run': 0, key: 0}}})
        with self.assertRaises(ValueError):
            launcher.validate_pipeline_config({'models': {'gbdt_pppt': {'n_m1_run_per_pipeline_run': 2}}})
        launcher.validate_pipeline_config({'models': {'original': {'n_m1_run_per_pipeline_run': 3}}})

    def test_partial_original_runs_reuse_dependencies_and_promote_coherent_snapshots(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            scalar = root / 'scalars.csv'
            scalar.write_text('ID,weight\n1,10\n')
            paths = {}
            for stage in ('M1', 'M2', 'Fusion'):
                paths[stage] = root / f'{stage}.json'
                write_json(paths[stage], {'preprocessing': {'random_seed': 42},
                    'data': {'scalar_csv': str(scalar)},
                    'output': {'pipeline_root': str(root / 'original')}})
            calls, scores = [], {'test': 10.}
            def fake_subprocess(command, *, cwd, env, check):
                stage = next(name for name in paths if str(paths[name]) == env['IM_ML_MODEL_CONFIG'])
                run = Path(env['IM_ML_RUN_DIR'])
                calls.append((stage, int(env['IM_ML_RUN_SEED']), '--pretrained-only' in command))
                with patch.dict(os.environ, env):
                    cfg = runtime.configure(read_json(paths[stage]), stage, 'DOE1')
                if stage == 'M1':
                    target = Path(cfg['output']['best_overall_dir'])
                    write_json(target / 'data_processing.json', {'run_ts': 'fixed-teacher',
                        'pp_cols': ['pp'], 'test_part_ids': [3], 'train_part_ids': [1], 'val_part_ids': [2]})
                    (target / 'best_model.pt').write_bytes(b'fixed M1')
                    (target / 'sensors_features_f.csv').write_text('ID,f\n1,2\n')
                    write_json(target / 'configuration.json', {'data': {'scalar_csv': str(scalar)}})
                    write_json(target / 'best_metrics.json', {'MAE': 1.})
                elif stage == 'M2':
                    self.assertEqual(Path(cfg['data']['features_csv']).read_text(), 'ID,f\n1,2\n')
                    target = Path(cfg['output']['best_overall_dir'])
                    write_json(target / 'scalers.json', {'m1_run_ts': 'fixed-teacher'})
                    (target / 'best_model.pt').write_text(run.name)
                    write_json(target / 'best_metrics.json', {'mean_MAE': 1.})
                else:
                    for key in ('m1_best_dir', 'm2_best_dir'):
                        self.assertTrue((Path(cfg['models'][key]) / 'best_model.pt').exists())
                    candidate = 'no_finetuning' if '--pretrained-only' in command else 'alpha_0.5'
                    folder = run / 'Fusion' / str(len(calls)) / 'best_fusion'
                    write_json(folder / 'test_1_metrics.json', {'MAE': scores['test'], 'source': 'final_test_1'})
                    write_json(folder / 'selection.json', {'source_folder': candidate,
                        'candidate': candidate, 'fine_tuned': candidate != 'no_finetuning',
                        'selection_metric': 'validation_MAE'})
            with patch.object(launcher, 'original_configs', return_value=paths), \
                    patch.object(launcher.subprocess, 'run', side_effect=fake_subprocess):
                launcher._run_original('DOE1', root / 'initial', 42, 7, {'reference': {'enabled': False}})
                reference = root / 'original/best_reference_run'
                for i, (m2_count, fusion_count) in enumerate(((2, 3), (0, 2), (2, 0)), 1):
                    scores['test'] = 10. - i
                    calls.clear()
                    run = root / f'partial_{i}'
                    spec = {'n_m1_run_per_pipeline_run': 0, 'n_m2_run_per_pipeline_run': m2_count,
                            'n_fusion_run_per_pipeline_run': fusion_count, 'reference': {'enabled': False}}
                    previous_reference = {path.relative_to(reference): path.read_bytes()
                                          for path in reference.rglob('*') if path.is_file()}
                    launcher._run_original('DOE1', run, 100, 7, spec)
                    self.assertEqual([c[0] for c in calls], ['M2'] * m2_count + ['Fusion'] * fusion_count)
                    self.assertEqual([c[1] for c in calls if c[0] == 'M2'], list(range(100, 100 + m2_count)))
                    self.assertEqual([c[1] for c in calls if c[0] == 'Fusion'], list(range(100, 100 + fusion_count)))
                    self.assertFalse(any(call[2] for call in calls))
                    self.assertEqual((reference / 'M1/best_overall/best_model.pt').read_bytes(), b'fixed M1')
                    if fusion_count:
                        self.assertEqual(read_json(reference / 'reference_run.json')['run_dir'], str(run.resolve()))
                    else:
                        self.assertFalse((run / 'Fusion').exists())
                        actual_reference = {path.relative_to(reference): path.read_bytes()
                                            for path in reference.rglob('*') if path.is_file()}
                        self.assertEqual(actual_reference, previous_reference)
                    self.assertEqual(set(read_json(run / 'reused_stages.json')['stages']),
                                     {'M1', 'M2'} if m2_count == 0 else {'M1'})
                    self.assertTrue((run / 'comparison.csv').exists())
                scores['test'] = 999.
                spec['n_m2_run_per_pipeline_run'] = 0
                spec['n_fusion_run_per_pipeline_run'] = 1
                launcher._run_original('DOE1', root / 'loser', 200, 7, spec)
                self.assertEqual(read_json(reference / 'reference_run.json')['run_dir'], str((root / 'partial_2').resolve()))

    def test_missing_reference_fails_before_launching_partial_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'm1.json'
            write_json(path, {'output': {'pipeline_root': directory}})
            with patch.object(launcher, 'original_configs', return_value={'M1': path}), \
                    patch.object(launcher.subprocess, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'Cannot reuse reference M1'):
                    launcher._run_original('DOE1', Path(directory) / 'run', 42, None,
                                           {'n_m1_run_per_pipeline_run': 0})
                run.assert_not_called()

    def test_original_repeats_m1_before_downstream_stages_in_every_full_run(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            scalar = root/'scalars.csv'; scalar.write_text('ID,weight\n1,10\n')
            paths = {}
            for stage in ('M1', 'M2', 'Fusion', 'RefModels'):
                paths[stage] = root/f'{stage}.json'
                write_json(paths[stage], {'preprocessing': {'random_seed': 42},
                    'data': {'scalar_csv': str(scalar)}, 'checkpoint': {'reuse': True},
                    'output': {'pipeline_root': str(root/'original')}})
            calls = []
            def fake_subprocess(command, *, cwd, env, check):
                stage = next(name for name, path in paths.items() if str(path) == env['IM_ML_MODEL_CONFIG'])
                with patch.dict(os.environ, env):
                    effective = runtime.configure(read_json(paths[stage]), stage, 'DOE1')
                run = Path(env['IM_ML_RUN_DIR'])
                seed = effective['preprocessing']['random_seed']
                sampler_seed = int(command[command.index('--optuna-seed')+1]) if '--optuna-seed' in command else None
                calls.append((stage, run, seed, sampler_seed))
                if stage == 'M1':
                    base_seed = 42 if run.name == 'run_001' else 1051
                    self.assertEqual(effective['preprocessing']['split_seed'], base_seed)
                    target = Path(effective['output']['best_overall_dir'])
                    score = [3., 1., 2.][seed-base_seed]
                    previous = read_json(target/'best_metrics.json') if (target/'best_metrics.json').exists() else {}
                    if score < previous.get('validation_MAE', float('inf')):
                        write_json(target/'best_metrics.json', {'validation_MAE': score})
                        (target/'best_model.pt').write_text(str(seed))
                        write_json(target/'data_processing.json', {'run_ts': str(seed)})
                        (target/'sensors_features_f.csv').write_text(str(seed))
                elif stage in ('M2', 'Fusion'):
                    selected = run/'M1/best_overall'
                    expected = '43' if run.name == 'run_001' else '1052'
                    self.assertEqual((selected/'best_model.pt').read_text(), expected)
                    self.assertEqual([c[0] for c in calls if c[1] == run][:3], ['M1']*3)
                    if stage == 'M2':
                        self.assertEqual(Path(effective['data']['features_csv']).read_text(), expected)
                        target = Path(effective['output']['best_overall_dir']); target.mkdir(parents=True)
                        (target/'best_model.pt').write_text('M2')
                        write_json(target/'scalers.json', {'m1_run_ts': expected})
                    else:
                        self.assertEqual(Path(effective['models']['m1_best_dir']), selected)
            spec = {'enabled': True, 'n_runs': 2, 'n_m1_run_per_pipeline_run': 3,
                    'reference': {'enabled': True, 'n_runs': 2}}
            with patch.object(launcher, 'original_configs', return_value=paths), \
                    patch.object(launcher.subprocess, 'run', side_effect=fake_subprocess), \
                    patch.object(launcher, '_original_comparison'), \
                    patch.object(launcher, '_publish_best_reference_run'):
                self.assertFalse(launcher.run_families({'models': {'original': spec}}, ['DOE1'], optuna_seed=7))
            self.assertEqual([c[0] for c in calls], ['M1', 'M1', 'M1', 'M2', 'Fusion', 'RefModels', 'RefModels']*2)
            m1_calls = [c for c in calls if c[0] == 'M1']
            self.assertEqual([c[2] for c in m1_calls], [42, 43, 44, 1051, 1052, 1053])
            self.assertEqual([c[3] for c in m1_calls], [7, 8, 9, 1016, 1017, 1018])
            preview = io.StringIO()
            with patch.object(launcher, 'original_configs', return_value=paths), \
                    patch.object(launcher.subprocess, 'run') as subprocess_run, redirect_stdout(preview):
                self.assertFalse(launcher.run_families({'models': {'original': spec}}, ['DOE1'], dry_run=True))
            subprocess_run.assert_not_called()
            self.assertEqual(preview.getvalue().count('M1 launches per pipeline run: 3'), 2)

    def test_pipeline_split_seed_overrides_model_default(self):
        with patch.dict(os.environ, {'IM_ML_RUN_SEED': '43', 'IM_ML_SPLIT_SEED': '42'}):
            cfg = runtime.configure({'preprocessing': {'split_seed': 17}}, 'M1', 'DOE1')
            self.assertEqual(cfg['preprocessing'], {'split_seed': 42, 'random_seed': 43})

    def test_pipeline_split_seed_validation(self):
        for options in (None, [], {'unknown': 1}, {'seed': True}, {'seed': -1},
                        {'seed': 1.5}, {'seed': '42'}, {'seed_step': -1}, {'seed_step': True},
                        {'seed': 2**32}, {'seed': 2**32 - 1, 'seed_step': 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                launcher.validate_pipeline_config({'data_split': options,
                    'models': {'original': {'n_runs': 2}}})
        launcher.validate_pipeline_config({'data_split': {'seed': 0, 'seed_step': 0}, 'models': {}})

    def test_pipeline_split_seed_controls_membership_independently_of_training(self):
        from src.M1_PPPT_to_W import _initial_split
        from src.Utility.group_splitting import grouped_train_val_split

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            scalar = root / 'scalars.csv'
            scalar.write_text('ID,weight\n1,10\n')
            config = root / 'm1.json'
            write_json(config, {'preprocessing': {'random_seed': 99, 'split_seed': 17},
                'data': {'scalar_csv': str(scalar)}, 'output': {'pipeline_root': str(root / 'original')}})
            observed = []
            def fake_subprocess(command, *, cwd, env, check):
                with patch.dict(os.environ, env):
                    effective = runtime.configure(read_json(config), 'M1', 'DOE1')
                prep = effective['preprocessing']
                split_seed = prep['split_seed']
                groups = np.repeat(np.arange(40), 3)
                dev, test = _initial_split(120, .2, split_seed, group_values=groups,
                                          group_cfg={'enabled': True})
                train, val = grouped_train_val_split(dev, .2, split_seed, groups, {'enabled': True})
                observed.append((prep['random_seed'], split_seed,
                                 tuple(tuple(sorted(part)) for part in (train, val, test))))
                target = Path(effective['output']['best_overall_dir'])
                write_json(target / 'data_processing.json', {'random_seed': split_seed})
                (target / 'best_model.pt').write_text(str(prep['random_seed']))
            spec = {'enabled': True, 'n_runs': 3, 'n_m1_run_per_pipeline_run': 2,
                    'n_m2_run_per_pipeline_run': 0, 'n_fusion_run_per_pipeline_run': 0,
                    'reference': {'enabled': False}}
            for step in (0, 1):
                observed.clear()
                with patch.object(launcher, 'original_configs', return_value={'M1': config}), \
                        patch.object(launcher.subprocess, 'run', side_effect=fake_subprocess), \
                        patch.object(launcher, '_original_comparison'):
                    failures = launcher.run_families({'data_split': {'seed': 42, 'seed_step': step},
                                                     'models': {'original': spec}}, ['DOE1'])
                self.assertFalse(failures)
                self.assertEqual([item[0] for item in observed], [99, 100, 1108, 1109, 2117, 2118])
                self.assertEqual([item[1] for item in observed], [42, 42, 42+step, 42+step, 42+2*step, 42+2*step])
                for index in (0, 2, 4):
                    self.assertEqual(observed[index][2], observed[index+1][2])
                memberships = {item[2] for item in observed}
                self.assertEqual(len(memberships), 1 if step == 0 else 3)

    def test_changing_pipeline_split_seed_invalidates_reuse_manifest(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            scalar = root / 'scalars.csv'
            scalar.write_text('ID,weight\n1,10\n')
            config = root / 'm1.json'
            write_json(config, {'preprocessing': {'random_seed': 42, 'split_seed': 17},
                'data': {'scalar_csv': str(scalar)}, 'checkpoint': {'reuse': True}})
            seeds = []
            def fake_subprocess(command, *, cwd, env, check):
                seeds.append(int(env['IM_ML_SPLIT_SEED']))
                target = root / 'run/M1/best_overall'
                write_json(target / 'data_processing.json', {'random_seed': seeds[-1]})
                (target / 'best_model.pt').write_bytes(b'checkpoint')
            spec = {'n_m2_run_per_pipeline_run': 0, 'n_fusion_run_per_pipeline_run': 0,
                    'reference': {'enabled': False}}
            with patch.object(launcher, 'original_configs', return_value={'M1': config}), \
                    patch.object(launcher.subprocess, 'run', side_effect=fake_subprocess), \
                    patch.object(launcher, '_original_comparison'):
                for split_seed in (42, 42, 43):
                    launcher._run_original('DOE1', root / 'run', 99, 7, spec, split_seed=split_seed)
            self.assertEqual(seeds, [42, 43])

    def test_doe1_configuration_sources_match_original_outputs(self):
        configs = {path.name: read_json(path) for path in (ROOT / "config" / "DoE1").glob("*.json")}
        m1 = configs["M1_DoE1_config.json"]["output"]
        m2 = configs["M2_DoE1_config.json"]["output"]
        self.assertEqual(Path(m1["best_overall_dir"]), Path(m1["output_dir"]) / "best_overall")
        self.assertEqual(Path(m2["best_overall_dir"]), Path(m2["output_dir"]) / "best_overall")
        self.assertEqual(Path(configs["M2_DoE1_config.json"]["data"]["features_csv"]),
                         Path(m1["best_overall_dir"]) / "sensors_features_f.csv")
        self.assertEqual(configs["Fusion_DoE1_config.json"]["models"]["m1_best_dir"], m1["best_overall_dir"])
        self.assertEqual(configs["Fusion_DoE1_config.json"]["models"]["m2_best_dir"], m2["best_overall_dir"])
        self.assertEqual(configs["RefModels_DoE1_config.json"]["m1_split_dir"], m1["best_overall_dir"])
        for name, cfg in configs.items():
            with self.subTest(config=name):
                data = cfg.get("data", {})
                if data.get("scalar_csv"):
                    self.assertTrue((ROOT / data["scalar_csv"]).is_file())
                for source in data.get("timeseries_csvs", []):
                    self.assertTrue((ROOT / source["path"]).is_file())
                for path in cfg.get("configs", {}).values():
                    self.assertTrue((ROOT / path).is_file())
                if "split" in cfg:
                    self.assertEqual(cfg["split"]["source_dir"], "outputs/DOE1/original/best_reference_run/M1/best_overall")
                if cfg.get("teacher", {}).get("checkpoint_dir"):
                    self.assertEqual(cfg["teacher"]["checkpoint_dir"], "outputs/DOE1/original/best_reference_run/M1/best_overall")
                if "evaluation" in cfg:
                    self.assertTrue(cfg["evaluation"]["enabled"])
                for path in cfg.get("output", {}).values():
                    if isinstance(path, str) and path.startswith("outputs/"):
                        self.assertEqual(Path(path).parts.count("DOE1"), 1)

    def test_runtime_paths_flatten_doe1_and_keep_other_material_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {"IM_ML_RUN_DIR": str(root), "IM_ML_RUN_SEED": "42"}):
                for material in ("DOE1", "PP"):
                    with self.subTest(material=material):
                        m1 = runtime.configure({}, "M1", material)
                        m2 = runtime.configure({"data": {}}, "M2", material)
                        fusion = runtime.configure({}, "Fusion", material)
                        reference = runtime.configure({}, "RefModels", material)
                        parent = root / "M1" if material == "DOE1" else root / "M1" / "PP"
                        self.assertEqual(Path(m1["output"]["best_overall_dir"]), parent / "best_overall")
                        self.assertEqual(m2["data"]["features_csv"], str(parent / "best_overall" / "sensors_features_f.csv"))
                        self.assertEqual(fusion["models"]["m1_best_dir"], m1["output"]["best_overall_dir"])
                        self.assertEqual(fusion["models"]["m2_best_dir"], m2["output"]["best_overall_dir"])
                        self.assertEqual(reference["m1_split_dir"], m1["output"]["best_overall_dir"])
                        _, best = stage_dirs(root, "M1", material)
                        self.assertEqual(best, parent / "best_overall")

    def test_original_publishes_complete_reference_using_fusion_test_mae(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            scalar = root / "scalars.csv"
            scalar.write_text("ID,weight\n1,10\n")
            paths = {}
            for stage in ("M1", "M2", "Fusion"):
                paths[stage] = root / f"{stage}.json"
                write_json(paths[stage], {"data": {"scalar_csv": str(scalar)},
                    "output": {"pipeline_root": str(root / "original"),
                               "best_overall_dir": str(root / "original" / stage / "best_overall")}})
            candidate = {"validation": .5, "test": 100.}
            calls = []
            def fake_subprocess(command, *, cwd, env, check):
                stage = "M1" if Path(command[1]).name.startswith("M1_") else "M2" if Path(command[1]).name.startswith("M2_") else "Fusion"
                run = Path(env["IM_ML_RUN_DIR"])
                with patch.dict(os.environ, env):
                    cfg = runtime.configure(read_json(paths[stage]), stage, "DOE1")
                local_m1 = run / "M1" / "best_overall"
                if stage == "Fusion":
                    self.assertEqual(Path(cfg["models"]["m1_best_dir"]), local_m1)
                    self.assertTrue((Path(cfg["models"]["m2_best_dir"]) / "scalers.json").is_file())
                    candidate_folder = candidate.get("folder", "alpha_0.5")
                    candidate_alpha = .5 if candidate_folder == "alpha_0.5" else None
                    partition = 'test_1' if candidate_folder == 'no_finetuning' else 'test'
                    write_json(run / "Fusion" / "sweep" / "best_fusion" / f"{partition}_metrics.json",
                               {"source": f"final_{partition}", "MAE": candidate["test"], "alpha": candidate_alpha})
                    write_json(run / "Fusion" / "sweep" / "best_fusion" / "test_2_metrics.json",
                               {"source": "final_test_2", "MAE": -1000.})
                    write_json(run / "Fusion" / "sweep" / "best_fusion" / "selection.json",
                               {"alpha": candidate_alpha, "source_folder": candidate_folder,
                                "candidate": candidate_folder, "fine_tuned": candidate_alpha is not None,
                                "selection_metric": "validation_MAE"})
                    # Stray legacy per-alpha tests cannot override the selected model.
                    write_json(run / "Fusion" / "sweep" / "alpha_1.0" / "test_metrics.json",
                               {"source": "final_test", "MAE": -1.})
                    write_json(run / "RefModels" / f"{run.name}.json", {"run": run.name})
                    return
                target = Path(cfg["output"]["best_overall_dir"])
                target.mkdir(parents=True, exist_ok=True)
                (target / "best_model.pt").write_text(run.name)
                write_json(target / "best_metrics.json", {"source": "final_test", "MAE": candidate["validation"],
                                                          "validation_MAE": candidate["validation"]})
                if stage == "M1":
                    write_json(target / "data_processing.json", {"run_ts": run.name, "pp_cols": ["pp"],
                        "train_part_ids": [1], "val_part_ids": [2], "test_part_ids": [3]})
                    (target / "sensors_features_f.csv").write_text("ID,f\n1,2\n")
                else:
                    self.assertEqual(Path(cfg["data"]["features_csv"]), local_m1 / "sensors_features_f.csv")
                    self.assertTrue(Path(cfg["data"]["features_csv"]).is_file())
                    write_json(target / "scalers.json", {"m1_run_ts": run.name})
                calls.append(stage)
            snapshot = root / "original" / "best_reference_run"
            canonical = snapshot / "M1" / "best_overall"
            self.assertFalse(canonical.exists())
            with patch.object(launcher, "original_configs", return_value=paths), \
                    patch.object(launcher.subprocess, "run", side_effect=fake_subprocess):
                launcher._run_original("DOE1", root / "run_001", 42, None, {"reference": {"enabled": False}})
                self.assertEqual(read_json(canonical / "data_processing.json")["run_ts"], "run_001")
                candidate.update(validation=.8, test=0., folder="no_finetuning")
                launcher._run_original("DOE1", root / "run_002", 1051, None, {"reference": {"enabled": False}})
                self.assertEqual(read_json(canonical / "data_processing.json")["run_ts"], "run_002")
                candidate.update(validation=.2, test=1000.)
                launcher._run_original("DOE1", root / "run_003", 2060, None, {"reference": {"enabled": False}})
            published_m2 = snapshot / "M2" / "best_overall"
            self.assertEqual(read_json(canonical / "data_processing.json")["run_ts"], "run_002")
            self.assertEqual(read_json(published_m2 / "scalers.json")["m1_run_ts"], "run_002")
            self.assertEqual(read_json(snapshot / "reference_run.json")["selected_alpha"], "no_finetuning")
            self.assertEqual(read_json(snapshot / "reference_run.json")["selected_fusion_candidate"], "no_finetuning")
            self.assertFalse(read_json(snapshot / "reference_run.json")["fusion_fine_tuned"])
            self.assertEqual(read_json(snapshot / "reference_run.json")["alpha_selection_metric"], "validation_MAE")
            self.assertIn("best_fusion", read_json(snapshot / "reference_run.json")["fusion_metrics_path"])
            self.assertIn("test_1_metrics.json", read_json(snapshot / "reference_run.json")["fusion_metrics_path"])
            self.assertTrue((snapshot / "comparison.csv").exists())
            self.assertTrue((snapshot / "RefModels" / "run_002.json").exists())
            self.assertFalse((snapshot / "RefModels" / "run_001.json").exists())
            # A validation-only repetition cannot replace a scored snapshot.
            for path in (root / "run_003" / "Fusion").rglob("*_metrics.json"):
                if path.name in ('test_metrics.json', 'test_1_metrics.json'):
                    path.rename(path.with_name("validation_metrics.json"))
            launcher._publish_best_reference_run("DOE1", root / "run_003", paths)
            self.assertEqual(read_json(canonical / "data_processing.json")["run_ts"], "run_002")
            self.assertEqual(calls, ["M1", "M2"] * 3)
            self.assertFalse((root / "run_001" / "M1" / "DOE1").exists())


if __name__ == "__main__":
    unittest.main()
