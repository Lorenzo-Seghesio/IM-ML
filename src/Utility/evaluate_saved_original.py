"""Evaluate an existing original DoE1 run without fitting any model or scaler.

Run from the repository root with ``python -m src.Utility.evaluate_saved_original
--run-dir ...``. Test outputs are isolated from the historical run artifacts.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path
import shutil
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src import M1_PPPT_to_W as m1
from src import M2_PP_to_F as m2
from src import Fusion_M1M2_WPred as fusion_module
from src import Reference_Models_W_Pred as refs
from .batch_test_diagnostics import save_batch_test_diagnostics
from .doe1_columns import doe1_column_name
from .group_splitting import validate_group_disjoint
from .hybrid_data import ROOT, read_json, write_json, file_hash
from .hybrid_models import FrozenTeacher


def map_ids(part_ids, requested):
    """Require every saved ID, preserving the exact saved order."""
    positions = {value: i for i, value in enumerate(part_ids.tolist())}
    if len(positions) != len(part_ids):
        raise ValueError("Duplicate input part IDs")
    missing = set(requested) - positions.keys()
    if missing:
        raise ValueError(f"Saved partition has {len(missing)} IDs absent from input data")
    return np.array([positions[value] for value in requested], dtype=np.intp)


def copy_stage(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, destination / path.name)
    cfg_path = destination / "configuration.json"
    if cfg_path.exists():
        cfg = read_json(cfg_path)
        cfg.setdefault("evaluation", {}).update(enabled=True)
        write_json(cfg_path, cfg)
    checkpoint = next((source / name for name in ("best_model.pt", "best_model.joblib", "fusion_model.pt")
                       if (source / name).exists()), None)
    write_json(destination / "evaluation_info.json", {"mode": "saved_checkpoint_only",
        "source_directory": str(source), "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": file_hash(checkpoint) if checkpoint else None,
        "retrained": False, "scalers_refitted": False})


def weight_report(out, partition, ids, truth, pred, groups, label, source_meta=None):
    out.mkdir(parents=True, exist_ok=True)
    report = refs._metrics(truth, pred, groups)
    report["source"] = "final_test" if partition == "test" else "validation"
    report.update(model=label, partition=partition, n_samples=len(ids))
    write_json(out / f"{partition}_metrics.json", report)
    pd.DataFrame({"ID_Part": ids, "y_true": truth, "y_pred": pred,
                  "error_g": pred - truth}).to_csv(out / f"{partition}_predictions.csv", index=False)
    if report.get("test_points"):
        pd.DataFrame(report["test_points"]).to_csv(out / f"{partition}_test_points.csv", index=False)
    if partition == "test":
        write_json(out / "best_metrics.json", {**(source_meta or {}), **report})
    return report


def replay_check(source, label, saved_score, score):
    """Check loaded models against saved validation metrics before testing."""
    delta = abs(float(saved_score) - float(score))
    # Historical checkpoints were evaluated on CUDA; CPU roundoff can differ.
    if not np.isfinite(score) or delta > 5e-5:
        raise ValueError(f"{label}: saved validation score {saved_score} differs from replay {score}")
    record = {"model": label, "saved_validation_score": saved_score,
              "replayed_validation_score": score, "absolute_difference": delta,
              "tolerance": 5e-5, "passed": True, "source": str(source)}
    print(f"VALIDATION REPLAY {label}: saved={saved_score:.8f}, replay={score:.8f}", flush=True)
    return record


def prediction_plots(out, truth, pred, report, ids, groups, label):
    if label == "M1":
        m1.save_final_test_plots(truth, pred, report, out, "DOE1")
    elif label.startswith("Reference/"):
        refs.save_final_test_plots(truth, pred, report, out, "DOE1", label.split('/')[-1])
        fusion_module._save_residuals_scatter(truth, pred, out / "residuals_scatter_final_test.png", "DOE1")
    else:
        fusion_module._save_scatter(truth, pred, report, out / "scatter_test.png", "DOE1")
        fusion_module._save_residuals_scatter(truth, pred, out / "residuals_scatter_test.png", "DOE1")
        fusion_module._save_residuals_hist(truth, pred, out / "residuals_hist_test.png", "DOE1")
    save_batch_test_diagnostics(truth, pred, groups, ids, out,
        target_names=["weight"], title=f"[DOE1 | {label}] Test evaluation")


def feature_report(out, model, pp, targets, indices, ids, groups, xs, ys, names, device):
    report = m2.evaluate_model(model, xs.transform(pp[indices]).astype(np.float32),
                               targets[indices], ys, names, device)
    pred = report.pop("pred")
    report.update(source="final_test", n_samples=len(indices))
    write_json(out / "test_metrics.json", report)
    pd.DataFrame([{"feature": name, **metrics} for name, metrics in report["per_target"].items()]).to_csv(
        out / "test_feature_metrics.csv", index=False)
    frame = pd.DataFrame({"ID_Part": ids[indices]})
    for i, name in enumerate(names):
        frame[f"{name}_true"], frame[f"{name}_pred"] = targets[indices, i], pred[:, i]
    frame.to_csv(out / "test_feature_predictions.csv", index=False)
    frame.to_csv(out / "f_predictions.csv", index=False)
    m2.save_final_test_plots(targets[indices], pred, report, out, "DOE1", names)
    save_batch_test_diagnostics(targets[indices], pred, groups[indices], ids[indices], out,
                               target_names=names, title="[DOE1 | M2] Test reconstruction")
    return report, pred


def evaluate_run(run_dir, output, xai=True, threads=4):
    torch.set_num_threads(threads)
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    if output == run_dir or run_dir.is_relative_to(output):
        raise ValueError("Evaluation output must be separate from original run artifacts")
    if output.exists():
        raise FileExistsError(f"Use a new evaluation directory: {output}")
    source_files = [p for p in run_dir.rglob('*') if p.is_file()
                    and 'test_evaluation' not in p.relative_to(run_dir).parts
                    and p.suffix in ('.pt', '.joblib', '.json')]
    source_hashes = {str(p): file_hash(p) for p in source_files}
    m1_source, m2_source = run_dir / "M1/best_overall", run_dir / "M2/best_overall"
    cfg, contract = read_json(m1_source / "configuration.json"), read_json(m1_source / "data_processing.json")
    (ids, pp, curves, y, _, pp_cols, groups, _) = m1.load_data(cfg["data"])
    if pp_cols != contract["pp_cols"]:
        raise ValueError("Current data feature ordering differs from saved M1")
    if not np.isfinite(pp).all() or any(not np.isfinite(c).all() for c in curves):
        raise ValueError("Missing input values require original fitted imputers")
    tr, va, te = [map_ids(ids, contract[key]) for key in ('train_part_ids', 'val_part_ids', 'test_part_ids')]
    validate_group_disjoint(tr, va, te, group_values=groups, config=contract["group_split"])
    dev = np.concatenate([tr, va])
    output.mkdir(parents=True)
    write_json(output / "data_processing.json", contract)
    pd.DataFrame({"ID_Part": ids[te], "batch_id": groups[te], "y_true": y[te]}).to_csv(
        output / "test_dataset.csv", index=False)
    audit, rows, xai_jobs = [], [], []

    def summarize(label, report, out, weight=True):
        rows.append({"model": label, "partition": "test", "n_samples": len(te), "target": "weight_g" if weight else "sensor_features",
                     "output_directory": str(out), **{k: v for k, v in report.items() if isinstance(v, (float, int))}})
        pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
        write_json(output / "validation_replay_checks.json", audit)
        print(f"TEST COMPLETE {label}: {report.get('MAE', report.get('mean_MAE'))}", flush=True)

    # M1 uses its exact saved PP/signal scalers and network weights.
    teacher = FrozenTeacher.load(m1_source)
    val_pred = teacher.predict_weight(pp[va], [c[va] for c in curves])
    val_score = refs._metrics(y[va], val_pred, groups[va])["MAE"]
    m1_meta = read_json(m1_source / "best_metrics.json")
    audit.append(replay_check(m1_source, "M1", m1_meta['validation_MAE'], val_score))
    out = output / "M1/best_overall"
    copy_stage(m1_source, out)
    pred = teacher.predict_weight(pp[te], [c[te] for c in curves])
    report = weight_report(out, 'test', ids[te], y[te], pred, groups[te], 'M1', m1_meta)
    prediction_plots(out, y[te], pred, report, ids[te], groups[te], 'M1')
    summarize('M1', report, out)

    # M2 reconstructs the saved M1 features, in the saved column order.
    reconstructor, m2_cfg, m2_meta = fusion_module.load_m2(m2_source, device)
    xs = m2._dict_to_scaler(contract['pp_sc'])
    sc = read_json(m2_source / 'scalers.json')
    if sc['m1_run_ts'] != contract['run_ts']:
        raise ValueError('M1/M2 checkpoint run identifiers do not match')
    ys = m2._dict_to_scaler(sc['y_sc'])
    names = m2_meta['f_cols']
    targets = pd.read_csv(m1_source / 'sensors_features_f.csv', index_col=0).reindex(ids)[names].to_numpy(np.float32)
    val = m2.evaluate_model(reconstructor, xs.transform(pp[va]).astype(np.float32), targets[va], ys, names, device)
    audit.append(replay_check(m2_source, 'M2', m2_meta['mean_MAE'], val['mean_MAE']))
    out = output / 'M2/best_overall'
    copy_stage(m2_source, out)
    report, _ = feature_report(out, reconstructor, pp, targets, te, ids, groups, xs, ys, names, device)
    write_json(out / 'best_metrics.json', {**m2_meta, **report})
    summarize('M2', report, out, weight=False)

    # Rebuild original Fusion, then load every already-trained alpha for validation only.
    fm = fusion_module
    m1_model, _, _ = fm.load_m1(m1_source, len(pp_cols), device)
    positions = fm.get_f_positions(m1_source, names)
    n_f = fm.get_m1_n_f_features(m1_meta['model_cfg'], m1_model, m1_source)
    fusion = fm.FusionModel(reconstructor, m1_model.pp_mlp, m1_model.merge,
        fm.TorchMinMaxScaler(xs).to(device), fm.TorchMinMaxScaler(ys).to(device),
        fm.TorchMinMaxScaler(xs).to(device), n_f, positions).to(device).eval()
    pretrained_state = copy.deepcopy(fusion.state_dict())
    sweeps = sorted((run_dir / 'Fusion').glob('*/sweep_config.json'))
    if len(sweeps) != 1:
        raise ValueError('Expected exactly one saved Fusion sweep in this repetition')
    sweep_source = sweeps[0].parent
    sweep_out = output / 'Fusion' / sweep_source.name
    sweep_out.mkdir(parents=True)
    for name in ('sweep_config.json', 'finetuning_split.json'):
        shutil.copy2(sweep_source / name, sweep_out / name)
    fine_split = read_json(sweep_source / 'finetuning_split.json')
    if set(fine_split['val_part_ids']) != set(contract['val_part_ids']):
        raise ValueError('Fusion saved validation IDs differ; evaluate with its exact saved partition')
    candidates = [('no_finetuning', None, None)]
    candidate_dirs = [p for p in sweep_source.glob('alpha_*') if p.is_dir() and (p / 'fusion_model.pt').is_file()]
    candidates += [(p.name, read_json(p / 'finetuning_info.json')['alpha'], p)
                   for p in sorted(candidate_dirs, key=lambda p: float(p.name[6:]))]
    candidate_rows = []
    for label, alpha, source in candidates:
        candidate_dir = sweep_out / label
        candidate_dir.mkdir()
        if source is None:
            fusion.load_state_dict(pretrained_state)
            info = {'enabled': False, 'alpha': None, 'best_epoch': 0, 'epochs_run': 0}
        else:
            fusion.load_state_dict(torch.load(source / 'fusion_model.pt', map_location=device, weights_only=True))
            info = {**read_json(source / 'finetuning_info.json'), 'enabled': True}
            for name in ('training_history.csv', 'm2_best_metrics.json'):
                if (source / name).exists():
                    shutil.copy2(source / name, candidate_dir / name)
        fusion.eval()
        torch.save(fusion.state_dict(), candidate_dir / 'fusion_model.pt')
        torch.save(fusion.m2.state_dict(), candidate_dir / 'm2_best_model.pt')
        write_json(candidate_dir / 'finetuning_info.json', info)
        validation = fm.evaluate_fusion(fusion, pp[va], y[va], device, partition='validation')
        expected = read_json(source / 'validation_metrics.json')['MAE'] if source else read_json(
            sweep_source / 'pretrained_validation_metrics.json')['MAE']
        audit.append(replay_check(source or sweep_source, f'Fusion/{label}', expected, validation['MAE']))
        report = weight_report(candidate_dir, 'validation', ids[va], y[va], validation['pred'], groups[va], f'Fusion/{label}')
        with torch.inference_mode():
            vx = torch.as_tensor(pp[va], dtype=torch.float32, device=device)
            _, vfeatures = fusion(vx, vx, return_features=True)
        vfeatures = vfeatures.cpu().numpy()
        verrors = np.abs(vfeatures - targets[va]).mean(axis=0)
        feature_mae = float(verrors.mean())
        info.update(val_metrics={**info.get('val_metrics', {}), 'mae_w': report['MAE'], 'mae_f': feature_mae})
        write_json(candidate_dir / 'finetuning_info.json', info)
        report.update(alpha=alpha, candidate=label, fine_tuned=source is not None, feature_MAE=feature_mae,
                      composite_loss=(alpha * report['MAE'] + (1-alpha) * feature_mae) if alpha is not None else None)
        write_json(candidate_dir / 'validation_metrics.json', report)
        pd.DataFrame({'feature': names, 'MAE': verrors}).to_csv(candidate_dir / 'validation_feature_metrics.csv', index=False)
        vf = pd.DataFrame({'ID_Part': ids[va]})
        for i, name in enumerate(names):
            vf[f'{name}_true'], vf[f'{name}_pred'] = targets[va, i], vfeatures[:, i]
        vf.to_csv(candidate_dir / 'validation_feature_predictions.csv', index=False)
        fm._save_scatter(y[va], validation['pred'], report, candidate_dir / 'scatter_validation.png', 'DOE1')
        fm._save_residuals_scatter(y[va], validation['pred'], candidate_dir / 'residuals_scatter_validation.png', 'DOE1')
        fm._save_residuals_hist(y[va], validation['pred'], candidate_dir / 'residuals_hist_validation.png', 'DOE1')
        save_batch_test_diagnostics(y[va], validation['pred'], groups[va], ids[va], candidate_dir,
            target_names=['weight'], title=f'Fusion {label}: validation', partition='validation')
        candidate_rows.append({'candidate': label, 'alpha': alpha, 'validation_MAE': report['MAE'],
                               'fine_tuned': source is not None, 'folder': label})
    winner = min(candidate_rows, key=lambda row: row['validation_MAE'])
    selection = {**winner, 'source_folder': winner['candidate'], 'selection_metric': 'validation_MAE',
                 'test_used_for_selection': False, 'retrained': False,
                 'tie_breaker': 'no fine-tuning first, then ascending saved alpha'}
    best = sweep_out / 'best_fusion'
    shutil.copytree(sweep_out / winner['candidate'], best)
    write_json(best / 'selection.json', selection)
    write_json(sweep_out / 'best_alpha_selection.json', selection)
    comparison = pd.DataFrame(candidate_rows)
    comparison['selected'] = comparison['candidate'] == winner['candidate']
    comparison.to_csv(sweep_out / 'alpha_comparison.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(comparison['candidate'], comparison['validation_MAE'], color=['#1B7837' if s else '#2166AC' for s in comparison['selected']])
    ax.set(ylabel='Validation weight MAE (g)', title='Saved Fusion candidates — validation only')
    ax.tick_params(axis='x', labelrotation=25)
    fig.tight_layout()
    fig.savefig(sweep_out / 'alpha_comparison.png')
    plt.close(fig)
    fusion.load_state_dict(torch.load(best / 'fusion_model.pt', map_location=device, weights_only=True))
    for parameter in fusion.parameters():
        parameter.requires_grad_(False)
    final = fm.evaluate_fusion(fusion, pp[te], y[te], device)
    report = weight_report(best, 'test', ids[te], y[te], final['pred'], groups[te], 'Fusion')
    report.update(alpha=winner['alpha'], candidate=winner['candidate'], fine_tuned=winner['fine_tuned'],
                  validation_MAE=winner['validation_MAE'], alpha_selection_metric='validation_MAE')
    prediction_plots(best, y[te], final['pred'], report, ids[te], groups[te], 'Fusion')
    with torch.inference_mode():
        tx = torch.as_tensor(pp[te], dtype=torch.float32, device=device)
        _, fpred = fusion(tx, tx, return_features=True)
    fpred = fpred.cpu().numpy()
    errors = np.abs(fpred - targets[te]).mean(axis=0)
    report.update(feature_MAE=float(errors.mean()), composite_loss=(winner['alpha'] * report['MAE'] +
                  (1-winner['alpha']) * float(errors.mean())) if winner['alpha'] is not None else None)
    reconstruction = fm.save_fusion_reconstruction(targets[te], fpred, names, ids[te], groups[te], best,
        'DOE1', target_scale=fusion.m2_y_sc.scale_.detach().cpu().numpy())
    report.update(feature_NMAE=reconstruction['mean_NMAE'], feature_RMSE=reconstruction['mean_RMSE'],
                  feature_R2=reconstruction['mean_R2'])
    write_json(best / 'test_metrics.json', report)
    write_json(best / 'best_metrics.json', report)
    selected_info = read_json(best / 'finetuning_info.json')
    selected_m2_meta = {**m2_meta, 'finetuning': selected_info,
                        'source': 'fusion_finetuning' if winner['fine_tuned'] else 'fusion_pretrained'}
    if winner['fine_tuned']:
        for key in ('mean_MAE', 'mean_NMAE', 'mean_RMSE', 'mean_R2', 'mean_MSE', 'per_target'):
            selected_m2_meta.pop(key, None)
    selected_m2_meta.update({key: reconstruction[key] for key in
        ('mean_MAE', 'mean_NMAE', 'mean_RMSE', 'mean_R2', 'mean_MSE', 'per_target')})
    selected_m2_meta['reconstruction_evaluation'] = reconstruction
    write_json(best / 'm2_best_metrics.json', selected_m2_meta)
    pd.DataFrame({'feature': names, 'MAE': errors}).to_csv(best / 'test_feature_metrics.csv', index=False)
    frame = pd.DataFrame({'ID_Part': ids[te]})
    for i, name in enumerate(names):
        frame[f'{name}_true'], frame[f'{name}_pred'] = targets[te, i], fpred[:, i]
    frame.to_csv(best / 'test_feature_predictions.csv', index=False)
    for src_dir, prefix in ((m1_source, 'm1'), (m2_source, 'm2_pretrained')):
        for name in ('best_model.pt', 'best_metrics.json', 'hpo_best_config.json'):
            if (src_dir / name).exists():
                shutil.copy2(src_dir / name, best / f'{prefix}_{name}')
    shutil.copy2(m1_source / 'sensors_features_f.csv', best / 'sensors_features_f.csv')
    source_info = read_json(next(sweep_source.glob('alpha_*/fusion_model_info.json')))
    write_json(best / 'fusion_model_info.json', {**source_info, 'alpha': winner['alpha'],
        'alpha_selection': selection, 'finetuning': read_json(best / 'finetuning_info.json'),
        'test_metrics': report, 'validation_metrics': read_json(best / 'validation_metrics.json'),
        'm2_reconstruction_metrics': reconstruction,
        'evaluation_partition': 'test', 'source_run': str(run_dir),
        'evaluation_mode': 'saved_checkpoint_only'})
    write_json(best / 'evaluation_info.json', {'mode': 'saved_checkpoint_only', 'source_run': str(run_dir),
        'candidate': winner['candidate'], 'retrained': False, 'scalers_refitted': False,
        'source_checkpoints': [str(m1_source / 'best_model.pt'), str(m2_source / 'best_model.pt')]
        if not winner['fine_tuned'] else [str(sweep_source / winner['candidate'] / 'fusion_model.pt')]})
    summarize(f"Fusion ({winner['candidate']})", report, best)
    if xai:
        settings = SimpleNamespace(**{k: source_info[k] for k in ('n_ig_steps', 'ih_samples', 'shap_bg', 'eg_bg')})
        xai_jobs.append(('Fusion', lambda: fm.run_fusion_xai(fusion, pp[dev], pp[te], ids[te], pp_cols,
            n_f, positions, settings, device, best, 'DOE1'), best,
            ['feature_attributions.csv', 'feature_attributions_single_pp.csv', 'ih_matrix.csv', 'jacobian_values.csv',
             'shap_values.csv', 'eg_values.csv', 'shap_m1merge_values.csv']))

    # Reference networks use their own serialized preprocessing and exact saved architecture.
    for source in sorted((run_dir / 'RefModels').glob('*/best_overall')):
        label = source.parent.name
        rcfg, meta = read_json(source / 'configuration.json'), read_json(source / 'best_metrics.json')
        channel_input = label == 'Encoder' and (meta.get('input_layout') == 'channels' or meta.get('channels', [1])[0] > 1)
        rids, rpp, rpt, ry, _, rcols, rgroups, _ = refs.load_data(
            rcfg['data'], rcfg['preprocessing'], 'DOE1', multichannel=channel_input)
        if rcols != [doe1_column_name(c) for c in pp_cols]:
            raise ValueError(f'{label}: reference PP feature ordering differs')
        rtest, rdev = map_ids(rids, contract['test_part_ids']), map_ids(rids, contract['train_part_ids'] + contract['val_part_ids'])
        saved_val = pd.read_csv(source / 'validation_predictions_per_batch.csv')['sample_id'].drop_duplicates().tolist()
        rval = map_ids(rids, saved_val)
        out = output / 'RefModels' / label / 'best_overall'
        copy_stage(source, out)
        if label == 'Encoder':
            model = refs.EncoderModel(**{k: meta[k] for k in ('channels', 'kernels', 'pool_kernels', 'head_hidden', 'dropout')}).to(device)
            scale = read_json(source / 'scalers.json')
            rx = ((rpt - scale['pt_min']) / (scale['pt_max'] - scale['pt_min'] + 1e-8)).astype(np.float32)
            prepare = refs._enc_prepare
        elif label == 'MLP':
            model = refs.MLPModel(meta['n_in'], meta['hidden_dims'], meta['dropout']).to(device)
            rx = m2._dict_to_scaler(read_json(source / 'scalers.json')).transform(rpp).astype(np.float32)
            prepare = refs._mlp_prepare
        else:
            model = joblib.load(source / 'best_model.joblib')
            if hasattr(model, 'set_params'):
                model.set_params(n_jobs=threads)
            rx, prepare = rpp, None
        if prepare is not None:
            model.load_state_dict(torch.load(source / 'best_model.pt', map_location=device, weights_only=True))
            model.eval()
            def predict(indices, model=model, prepare=prepare, rx=rx):
                with torch.inference_mode():
                    return model(prepare(rx[indices]).to(device)).cpu().numpy()
        else:
            def predict(indices, model=model, rx=rx):
                return model.predict(rx[indices])
        replay = refs._metrics(ry[rval], predict(rval), rgroups[rval])
        audit.append(replay_check(source, f'Reference/{label}', meta['MAE'], replay['MAE']))
        pred = predict(rtest)
        report = weight_report(out, 'test', rids[rtest], ry[rtest], pred, rgroups[rtest], f'Reference/{label}', meta)
        prediction_plots(out, ry[rtest], pred, report, rids[rtest], rgroups[rtest], f'Reference/{label}')
        summarize(f'Reference/{label}', report, out)
        if not xai:
            continue
        if label == 'Encoder':
            job = lambda model=model, rx=rx, rtest=rtest, out=out: refs._xai_gradcam_encoder(model, rx[rtest], out, 'Encoder', device)
            expected = ['xai_gradcam.png']
        elif label == 'MLP':
            def job(model=model, rx=rx, rtest=rtest, rdev=rdev, rcols=rcols, out=out):
                refs._xai_shap_mlp(model, rx[rdev], rx[rtest], rcols, out, 'MLP', device)
                refs._xai_ig_mlp(model, rx[rtest], rx[rdev], rcols, out, 'MLP', device)
            expected = ['xai_shap_bar.png', 'xai_shap_beeswarm.png', 'xai_ig_bar.png']
        else:
            job = lambda model=model, rx=rx, rtest=rtest, rdev=rdev, rcols=rcols, out=out, label=label: refs._xai_shap_gbdt(model, rx[rdev], rx[rtest], rcols, out, label)
            expected = ['xai_shap_bar.png', 'xai_shap_beeswarm.png']
        # Remove validation XAI copied from history before creating test explanations.
        for path in out.glob('xai_*'):
            if path.is_file():
                path.unlink()
        xai_jobs.append((f'Reference/{label}', job, out, expected))

    xai_status = []
    for label, job, out, expected in xai_jobs:
        print(f'XAI START {label}', flush=True)
        torch.manual_seed(42)
        np.random.seed(42)
        job()
        missing = [name for name in expected if not (out / name).is_file()]
        xai_status.append({'model': label, 'directory': str(out), 'missing_files': missing,
                           'complete': not missing, 'partition': 'test'})
        write_json(output / 'xai_status.json', xai_status)
        print(f'XAI COMPLETE {label}; missing={missing}', flush=True)

    unchanged = all(file_hash(path) == digest for path, digest in source_hashes.items())
    write_json(output / 'evaluation_manifest.json', {'source_run': str(run_dir),
        'evaluated_at': datetime.now(ZoneInfo('Europe/Brussels')).isoformat(), 'device': str(device),
        'mode': 'saved_checkpoint_only', 'retrained': False, 'scalers_refitted': False,
        'test_part_ids': contract['test_part_ids'], 'test_groups': sorted(set(groups[te].tolist())),
        'source_files_sha256': source_hashes, 'source_artifacts_unchanged': unchanged,
        'fusion_selection': selection, 'validation_replays': audit, 'xai': xai_status,
        'cv': 'Historical CV artifacts copied unchanged; no fold models were retrained.',
        'unavailable_models': ['CustomGBDT: not trained or saved in this historical run']})
    if not unchanged:
        raise ValueError('A historical model artifact changed during evaluation')
    write_summary(output, rows, selection, xai_status)
    print(f'ALL EVALUATIONS COMPLETE: {output}', flush=True)
    return rows


def write_summary(output, rows, selection, xai_status):
    weight = sorted((r for r in rows if r['target'] == 'weight_g'), key=lambda r: r['MAE'])
    lines = ['# Saved-run test evaluation', '',
             'Existing weights and fitted preprocessing were loaded; no model was retrained. '
             f"All models use the same {rows[0]['n_samples']} held-out parts from the saved M1 split.", '',
             f"Fusion was selected on validation only: **{selection['candidate']}**, "
             f"validation MAE **{selection['validation_MAE']:.6f} g**.", '',
             '| Model | Test MAE (g) | Test RMSE (g) | Test R² |',
             '| --- | ---: | ---: | ---: |']
    for row in weight:
        lines.append(f"| {row['model']} | {row['MAE']:.6f} | {row['RMSE']:.6f} | {row['R2']:.4f} |")
    for row in rows:
        if row['target'] == 'sensor_features':
            lines.extend(['', f"M2 reconstruction: mean feature MAE **{row['mean_MAE']:.6f}**, "
                          f"normalized MAE **{row['mean_NMAE']:.6f}**, mean feature R² **{row['mean_R2']:.4f}**."])
    lines.extend(['', 'See `comparison.csv`, `validation_replay_checks.json`, `evaluation_manifest.json`, '
                  'and each stage folder for metrics, part predictions, test-point/batch reports, plots and XAI.', '',
                  'Historical CV plots and training records were copied unchanged. '
                  'CustomGBDT has no checkpoint in this run. No reference run was promoted.'])
    if any(not item['complete'] for item in xai_status):
        lines.extend(['', 'Some XAI files are missing; see `xai_status.json`.'])
    (output / 'TEST_REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--skip-xai', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    stamp = datetime.now(ZoneInfo('Europe/Brussels')).strftime('%Y-%m-%d_%H-%M-%S')
    output = args.output_dir or args.run_dir / 'test_evaluation' / stamp
    evaluate_run(args.run_dir, output, xai=not args.skip_xai, threads=args.threads)


if __name__ == '__main__':
    main()
