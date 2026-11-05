"""Dual final tests, scoped product IDs, and batch-balanced explanation domains."""
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from test_hybrid_models import divided_fixture
from src.Utility.dataset_layout import (
    resolve_data_config, divided_dataset, training_data_identity, label_test_1,
)
from src.Utility.hybrid_data import load_data, split_data, write_json, read_json
from src.Utility.external_evaluation import external_dataset, finish_original_stage
from src.Utility.fusion_xai_sampling import batch_balanced_indices, scoped_fusion_xai
from src.Utility.doe1_columns import DOE1_SOURCE_COLUMNS, DOE1_DISPLAY_COLUMNS
from src import Fusion_M1M2_WPred as fusion
from src import Reference_Models_W_Pred as references


class DividedDatasetTests(unittest.TestCase):
    def test_original_stage_with_test2_disabled_keeps_test1_and_removes_old_external_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cfg, _ = divided_fixture(root)
            cfg['evaluation']['enabled'] = True
            cfg['evaluation']['test_2_dir'] = None
            for stage in ('M1', 'M2', 'MLP'):
                out = root/stage; out.mkdir()
                write_json(out/'best_metrics.json', {'source': 'final_test', 'MAE': 1.0})
                write_json(out/'test_2_metrics.json', {'MAE': 99.0})
                with patch('src.Utility.external_evaluation.external_dataset', return_value=None) as loader:
                    finish_original_stage(cfg, stage, out)
                self.assertIsNone(loader.call_args.args[0]['evaluation']['test_2_dir'])
                self.assertFalse((out/'test_2_metrics.json').exists())
                self.assertEqual(read_json(out/'test_1_metrics.json')['source'], 'final_test_1')

    def test_existing_test_folder_is_ignored_without_explicit_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cfg, _ = divided_fixture(root)
            cfg['evaluation'].pop('test_2_dir')
            self.assertIsNone(external_dataset(cfg))
            cfg['evaluation']['test_2_dir'] = None
            self.assertIsNone(external_dataset(cfg))
            self.assertEqual(len(load_data(cfg).ids), 48)
            self.assertTrue((root/'Test').is_dir())

    def test_external_folder_can_be_outside_dataset_root_without_changing_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cfg, _ = divided_fixture(root)
            identity = training_data_identity(cfg['data'])
            custom = root/'different_external_dataset'
            (root/'Test').rename(custom)
            cfg['evaluation']['test_2_dir'] = str(custom)
            external = external_dataset(cfg)
            self.assertEqual(len(external.ids), 6)
            self.assertTrue(all(str(custom) in source['path'] for source in external.sources))
            self.assertEqual(len(load_data(cfg).ids), 48)
            self.assertEqual(identity, training_data_identity(cfg['data']))
            cfg['evaluation']['test_2_dir'] = str(root/'missing')
            with self.assertRaisesRegex(FileNotFoundError, 'evaluation.test_2_dir'):
                external_dataset(cfg)

    def test_resolves_distinct_files_and_never_combines_reused_product_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, _ = divided_fixture(Path(directory))
            train, external = load_data(cfg), external_dataset(cfg)
            self.assertEqual((len(train.ids), len(external.ids)), (48, 6))
            self.assertEqual(set(external.ids), set(range(6)))
            self.assertTrue(set(train.ids) & set(external.ids))
            self.assertGreater(external.y.mean(), train.y.mean()+90)
            self.assertEqual(external.pp_names, train.pp_names)
            self.assertTrue(all('/Test/' in source['path'] for source in external.sources))
            identity = training_data_identity(cfg['data'])
            path = Path(directory)/'Test/External_Scalars.csv'
            frame = pd.read_csv(path); frame['weight'] += 1000; frame.to_csv(path, index=False)
            self.assertEqual(identity, training_data_identity(cfg['data']))

    def test_flat_folder_fallback_has_no_second_final_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _ = divided_fixture(root)
            for file in (root/'Train').glob('*.csv'):
                shutil.move(file, root/file.name)
            shutil.rmtree(root/'Train'); shutil.rmtree(root/'Test')
            cfg['evaluation']['test_2_dir'] = None
            resolved = resolve_data_config(cfg['data'])
            self.assertTrue(Path(resolved['scalar_csv']).is_file())
            self.assertFalse(divided_dataset(resolved))
            self.assertIsNone(external_dataset({**cfg, 'data': resolved}))
            self.assertEqual(len(load_data({**cfg, 'data': resolved}).ids), 48)

    def test_refuses_old_split_even_if_product_ids_happen_to_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cfg, _ = divided_fixture(root)
            data = load_data(cfg)
            _, contract = split_data(data, cfg)
            contract.pop('training_data_identity')
            write_json(root/'old/data_processing.json', contract)
            cfg['split'] = {'source_dir': str(root/'old')}
            with self.assertRaisesRegex(ValueError, 'different training dataset'):
                split_data(data, cfg)

    def test_external_test_cannot_silently_drop_parts_missing_signal_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cfg, _ = divided_fixture(root)
            curve = root/'Test/External_Pressure_1.csv'
            pd.read_csv(curve).iloc[1:].to_csv(curve, index=False)
            with self.assertRaisesRegex(ValueError, 'refusing to drop test_2 rows'):
                external_dataset(cfg)

    def test_labels_internal_artifacts_without_altering_external_or_split_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cfg, _ = divided_fixture(root)
            out = root/'artifacts'; out.mkdir()
            for name in ('test_metrics.json', 'test_2_metrics.json', 'best_metrics.json'):
                write_json(out/name, {'source': 'final_test_2' if 'test_2' in name else 'final_test'})
            write_json(out/'train_test_split.json', {'test_part_ids': [1]})
            (out/'residuals_hist_final_test.png').write_bytes(b'plot')
            (out/'validation_test_points.csv').write_text('batch,mae\n')
            label_test_1(out, cfg['data']); label_test_1(out, cfg['data'])
            self.assertTrue((out/'test_1_metrics.json').is_file())
            self.assertTrue((out/'residuals_hist_final_test_1.png').is_file())
            self.assertTrue((out/'validation_test_points.csv').is_file())
            self.assertEqual(read_json(out/'test_2_metrics.json')['source'], 'final_test_2')
            self.assertEqual(read_json(out/'train_test_split.json')['test_part_ids'], [1])

    def test_every_reference_kind_evaluates_all_external_rows_without_fitting(self):
        class Tree:
            def fit(self, *args):
                raise AssertionError('External testing must not fit models')
            def predict(self, values):
                return values.sum(axis=1)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory); cfg, _ = divided_fixture(root)
            external = external_dataset(cfg)
            cfg['evaluation'].update(enabled=True, plots=False)
            for label in ('Encoder', 'MLP', 'LightGBM', 'XGBoost', 'CustomGBDT'):
                model = (references.EncoderModel([1,2], [3], [], [2], 0.) if label == 'Encoder' else
                         references.MLPModel(2, [3], 0.) if label == 'MLP' else Tree())
                state = copy.deepcopy(model.state_dict()) if isinstance(model, torch.nn.Module) else None
                out = root/label; out.mkdir()
                write_json(out/'test_metrics.json', {'source': 'final_test'})
                with patch('torch.optim.Adam', side_effect=AssertionError('Must not train')):
                    finish_original_stage(cfg, label, out, model=model)
                predicted = pd.read_csv(out/'test_2_predictions.csv')
                self.assertEqual(predicted['part_id'].tolist(), external.ids.tolist())
                np.testing.assert_allclose(predicted['y_true'], external.y, rtol=1e-7)
                self.assertTrue((out/'test_1_metrics.json').is_file())
                if state:
                    for name, values in state.items():
                        torch.testing.assert_close(values, model.state_dict()[name], rtol=0, atol=0)


class FusionSamplingTests(unittest.TestCase):
    def test_sampler_covers_every_batch_despite_small_requested_budget(self):
        groups = np.repeat(np.arange(8), [100, 2, 3, 4, 1, 7, 8, 9])
        selected = batch_balanced_indices(groups, 3, 42)
        self.assertEqual(set(groups[selected]), set(range(8)))
        self.assertEqual(len(selected), len(set(selected)))
        np.testing.assert_array_equal(selected, batch_balanced_indices(groups, 3, 42))

    def test_scope_order_and_namespace_keep_test2_ids_separate(self):
        pp = np.arange(24).reshape(12,2); groups = np.repeat([0,1,2], 4)
        domains = [('train', pp, np.arange(12), groups), ('test_1', pp[:4], np.arange(4), groups[:4]),
                   ('test_2', pp[:4]+100, np.arange(4), groups[:4])]
        seen = []
        with tempfile.TemporaryDirectory() as directory:
            scoped_fusion_xai(lambda background, x, ids, out, partition: seen.append((partition, x.copy())),
                pp, groups, domains, directory, {'background_samples': 2, 'explain_samples': 2}, 42)
            self.assertEqual([p for p,_ in seen], ['train', 'test_1', 'test_2'])
            self.assertGreater(seen[2][1].mean(), seen[1][1].mean()+90)
            for entry in read_json(Path(directory)/'partitions.json'):
                self.assertTrue(entry['all_batches_represented'])

    def test_fusion_display_names_cover_source_and_pathway_feature_labels(self):
        for raw, display in zip(DOE1_SOURCE_COLUMNS, DOE1_DISPLAY_COLUMNS):
            self.assertEqual(fusion._strip_prefix(raw), display)
            self.assertEqual(fusion._strip_prefix(raw+'_indirect'), display+'_indirect')
            self.assertEqual(fusion._strip_prefix(raw+'_direct'), display+'_direct')


if __name__ == '__main__':
    unittest.main()
