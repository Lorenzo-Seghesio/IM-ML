"""Stage orchestration for the three hybrid families, sequential and resumable."""
from __future__ import annotations

import argparse
import copy
import hashlib
from datetime import datetime
from pathlib import Path
import shutil
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor

from src import M2_PP_to_F as baseline_m2
from .hybrid_data import (
    ROOT, resolve, read_json, write_json, load_data, split_data, fingerprint,
    file_hash, fit_imputer, impute, checkpoint_valid, save_manifest, split_provenance,
)
from .hybrid_models import (
    FAMILIES, FrozenTeacher, HybridPredictor, train_teacher, train_xgb, train_m2,
    save_evaluation, stage_dirs, promote, evaluate_teacher, evaluate_m2, fusion_cv,
)
from .engineered_features import extract_features
from .dataset_layout import resolve_data_config, internal_test_name
from .external_evaluation import external_dataset, clear_test_2_artifacts

CONFIG_NAMES = {"gbdt_pppt": "M1_GBDT_PPPT_to_W", "gbdt_ppdirect_pt": "M1_GBDT_PPdirect_PT_to_W",
                "gbdt_engineered": "M1_GBDT_engineered_to_W"}


def default_config(family, material):
    directory = "DoE1" if material == "DOE1" else "ProBayes"
    return ROOT / "config" / directory / f"{CONFIG_NAMES[family]}_{material}_config.json"


def training_config(cfg):
    return {key: value for key, value in cfg.items() if key not in
            ("evaluation", "xai", "output", "checkpoint", "_comment")}


def array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _load_model_config(path, seed, optuna_seed):
    cfg = read_json(resolve(path))
    if cfg.get("data"):
        cfg["data"] = resolve_data_config(cfg["data"])
    cfg.setdefault("preprocessing", {})["random_seed"] = seed
    if optuna_seed is not None:
        cfg.setdefault("optuna", {})["sampler_seed"] = optuna_seed
    return cfg


def _snapshot_teacher(source, destination):
    """Copy artifacts into this run; never change the original pretrained directory."""
    if Path(source).resolve() == Path(destination).resolve():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("best_model.pt", "best_metrics.json", "data_processing.json", "configuration.json",
                 "preprocessing.joblib", "signal_schema.json", "feature_summary.json"):
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)
    for pattern in ("cv_*.png", "cv_*.csv", "cv_*_training.json", "training_history.json", "pipeline_source.json"):
        for path in source.glob(pattern):
            shutil.copy2(path, destination / path.name)
    if (source / "folds").is_dir():
        shutil.copytree(source / "folds", destination / "folds", dirs_exist_ok=True)


def _reuse_checkpoint(cfg, directory, signature, files, training_selected=True):
    settings = cfg.get("checkpoint", {})
    if training_selected and not settings.get("reuse", True):
        return False
    if checkpoint_valid(directory, signature, files):
        return True
    source = settings.get("source_dir")
    if source:
        source = resolve(source)
        if not checkpoint_valid(source, signature, files):
            raise ValueError(f"Requested checkpoint source is missing/incompatible: {source}")
        shutil.copytree(source, directory, dirs_exist_ok=True)
        print(f"  Reusing compatible checkpoint from {source}", flush=True)
        return True
    return False


def _feature_names(teacher, dataset):
    widths = [encoder.fc.out_features for encoder in teacher.model.encoders]
    return [f"f_{source.get('name', f'signal_{number+1}')}__{i}"
            for number, (source, width) in enumerate(zip(dataset.sources, widths)) for i in range(width)]


def _fold_feature_provider(teacher_dir, dataset, family, splits, cfg, pp_imp, final_features, teacher):
    candidates = list((Path(teacher_dir) / "folds").glob("fold_*"))
    candidates += [resolve(path) for path in cfg.get("cv", {}).get("fold_checkpoint_dirs", [])]
    final_contract = read_json(Path(teacher_dir) / "data_processing.json")
    test_ids = set(dataset.ids[splits[2]].tolist())
    def get_features(number, train, val, holdout):
        heldout_ids = set(dataset.ids[holdout].tolist())
        for path in candidates:
            contract = read_json(path / "data_processing.json")
            seen = set(contract["train_part_ids"]) | set(contract["val_part_ids"])
            if (set(contract["test_part_ids"]) == heldout_ids and not seen & (heldout_ids | test_ids)
                    and seen.issubset(set(dataset.ids[np.concatenate([train, val])].tolist()))):
                fold_teacher = FrozenTeacher.load(path, dataset)
                if fold_teacher.model_config != teacher.model_config:
                    raise ValueError("Fold-specific encoder architecture differs from final teacher")
                pp = (fold_teacher.pp_features(dataset.pp) if family == "gbdt_pppt"
                      else dataset.pp)
                sensor = fold_teacher.sensor_features(dataset.curves)
                return np.concatenate([pp, sensor], axis=1), (
                    f"fold_{number}: fold-specific frozen encoder; "
                    "architecture selection remains conditional on development data")
        overlap = heldout_ids & set(final_contract["train_part_ids"])
        note = (f"fold_{number}: head-only CV with the final pretrained encoder; "
                f"{len(overlap)} held-out parts were in encoder training. "
                "This is not an unbiased evaluation of the entire hybrid.")
        if number == 0:
            warnings.warn(note, UserWarning, stacklevel=2)
        return final_features, note
    return get_features


def _save_feature_table(directory, dataset, sensor, names):
    frame = pd.DataFrame(sensor, columns=names)
    frame.insert(0, dataset.config["data"]["id_col"], dataset.ids)
    frame.to_csv(Path(directory) / "sensors_features_f.csv", index=False)
    write_json(Path(directory) / "feature_schema.json", {"sensor_columns": names,
        "pp_columns": dataset.pp_names, "signal_sources": dataset.sources})


def _stage_evaluation(cfg, directory, predictor, dataset, splits, role, sensor=None, external=None, external_sensor=None):
    _, val, test = splits
    evaluations = [("validation", dataset, val, sensor)]
    if cfg.get("evaluation", {}).get("enabled", False):
        evaluations.append((internal_test_name(dataset.config["data"]), dataset, test, sensor))
        if external is not None:
            evaluations.append(("test_2", external, np.arange(len(external.ids)), external_sensor))
    results = []
    for partition, current, indices, current_sensor in evaluations:
        if predictor.reconstructor is None:
            pred = predictor.predict_teacher(current.pp[indices],
                [matrix[indices] for matrix in current.curves], current_sensor[indices] if current_sensor is not None else None)
        else:
            pred = predictor.predict(current.pp[indices])
        report = save_evaluation(directory, partition, current, indices, pred, role,
                                 evaluation_cfg=cfg.get("evaluation", {}))
        if predictor.reconstructor is not None and current_sensor is not None:
            actual, predicted = current_sensor[indices], predictor.predict_features(current.pp[indices])
            valid = np.isfinite(actual)
            report["M2_feature_MAE"] = float(np.abs(actual - predicted)[valid].mean()) if valid.any() else None
            ys = predictor.m2_preprocessing["y_scaler"]
            differences = np.abs(ys.transform(actual) - ys.transform(predicted))
            report["M2_feature_NMAE"] = float(differences[valid].mean()) if valid.any() else None
            write_json(Path(directory) / f"{partition}_metrics.json", {"role": role, "partition": partition, **report})
        results.append({"role": role, "partition": partition,
                        **{key: value for key, value in report.items() if key != "test_points"}})
    return results


def _saved_evaluation_rows(directory, role, test_enabled):
    rows = []
    for partition in ("validation", "test", "test_1", "test_2") if test_enabled else ("validation",):
        path = Path(directory) / f"{partition}_metrics.json"
        if path.exists():
            report = read_json(path)
            rows.append({"role": role, "partition": partition,
                         **{key: value for key, value in report.items() if isinstance(value, (int, float))}})
    return rows


def _write_comparison(rows, run_dir, family, material, seed, provenance):
    for row in rows:
        row.update(family=family, material=material, seed=seed, run_dir=str(run_dir))
        if provenance:
            reference = provenance.get("reference_run", {})
            row.update(reference_run_dir=reference.get("run_dir"),
                       reference_run_timestamp=reference.get("run_timestamp"),
                       reference_split_sha256=provenance["split_sha256"])
    pd.DataFrame(rows).to_csv(run_dir / "comparison.csv", index=False)


def run_hybrid(family, config_path, material="DOE1", run_dir=None, seed=None,
               optuna_seed=None, stages=("m1", "xgboost", "m2", "fusion"), split_seed=None):
    if family not in FAMILIES:
        raise ValueError(f"Unknown hybrid family: {family}")
    cfg = read_json(config_path)
    cfg["data"] = resolve_data_config(cfg["data"])
    if cfg.get("family") != family:
        raise ValueError("Model configuration family differs from the requested entry point")
    seed = int(seed if seed is not None else cfg["preprocessing"]["random_seed"])
    cfg["preprocessing"]["random_seed"] = seed
    if split_seed is not None:
        cfg["preprocessing"]["split_seed"] = split_seed
    if optuna_seed is not None:
        cfg.setdefault("optuna", {})["sampler_seed"] = optuna_seed
    run_dir = Path(run_dir) if run_dir else resolve(cfg["output"]["output_dir"]) / "runs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{family}] seed={seed}; stages={', '.join(stages)}; outputs={run_dir}", flush=True)
    dataset = load_data(cfg)
    external = external_dataset(cfg)
    if external is not None:
        write_json(run_dir / "test_2_dataset.json", {"namespace": "test_2", "data": external.config["data"],
            "data_fingerprint": external.digest, "part_ids": external.ids.tolist(),
            "n_samples": len(external.ids), "used_in_training": False})
    else:
        clear_test_2_artifacts(run_dir)
    splits, contract = split_data(dataset, cfg)
    provenance = split_provenance(cfg)
    if provenance:
        write_json(run_dir / "reference_run.json", provenance)
    write_json(run_dir / "data_processing.json", contract)
    write_json(run_dir / "configuration.json", cfg)
    write_json(run_dir / "run_info.json", {"family": family, "material": material,
        "seed": seed, "optuna_seed_override": optuna_seed, "stages": list(stages),
        "data_fingerprint": dataset.digest, "split_reference": provenance})
    train, val, test = splits
    teacher = None
    m1_dir, m1_best = stage_dirs(run_dir, "M1", material)
    teacher_settings = cfg.get("teacher", {})
    teacher_cfg = None
    if family != "gbdt_engineered":
        if teacher_settings.get("train", False):
            teacher_cfg = _load_model_config(teacher_settings["config"], seed, optuna_seed)
            if split_seed is not None:
                teacher_cfg["preprocessing"]["split_seed"] = split_seed
            if teacher_cfg["data"] != cfg["data"]:
                raise ValueError("Teacher and hybrid data configurations must match")
            expected_direct = family == "gbdt_ppdirect_pt"
            if (teacher_cfg["model"].get("pp_branch") == "direct") != expected_direct:
                raise ValueError("Teacher PP branch does not match the hybrid family")
            signature = fingerprint(dataset.digest, contract, training_config(teacher_cfg), seed)
            files = ["best_model.pt", "best_metrics.json", "preprocessing.joblib", "data_processing.json"]
            if not _reuse_checkpoint(teacher_cfg, m1_dir, signature, files, "m1" in stages):
                if "m1" not in stages:
                    raise ValueError("M1 checkpoint missing/incompatible; run --stages m1 first")
                teacher = train_teacher(teacher_cfg, dataset, splits, contract, m1_dir, seed)
                save_manifest(m1_dir, signature, files)
            else:
                print("  Reusing compatible M1 checkpoint", flush=True)
            teacher = teacher or FrozenTeacher.load(m1_dir, dataset)
        else:
            source = resolve(teacher_settings["checkpoint_dir"])
            # Match source split IDs exactly, so the encoder never saw this hybrid's test parts.
            source_contract = read_json(source / "data_processing.json")
            for key in ("train_part_ids", "val_part_ids", "test_part_ids"):
                if set(source_contract[key]) != set(contract[key]):
                    raise ValueError("Pretrained encoder and hybrid split IDs differ; reuse its authoritative split")
            signature = fingerprint(dataset.digest, contract, *[file_hash(source / name) for name in
                ("best_model.pt", "best_metrics.json", "data_processing.json")])
            _snapshot_teacher(source, m1_dir)
            teacher = FrozenTeacher.load(m1_dir, dataset)
            if family == "gbdt_ppdirect_pt" and teacher.model_config.get("pp_branch") != "direct":
                raise ValueError("Model B requires a pretrained direct-PP M1_pp")
            if family == "gbdt_pppt" and teacher.model_config.get("pp_branch") == "direct":
                raise ValueError("Model A requires the original M1 with a learned PP MLP")
            save_manifest(m1_dir, signature, ["best_model.pt", "best_metrics.json", "data_processing.json"])
        evaluate_teacher(teacher, teacher_cfg or cfg, dataset, splits, m1_dir)
        if not (teacher_cfg or cfg).get("evaluation", {}).get("test_2_dir"):
            clear_test_2_artifacts(m1_best)
        sensor = teacher.sensor_features(dataset.curves)
        sensor_names = _feature_names(teacher, dataset)
        signature = fingerprint(signature, file_hash(m1_dir / "best_model.pt"),
            [(str(path.relative_to(m1_dir)), file_hash(path)) for path in sorted(m1_dir.rglob("*"))
             if path.is_file() and (path.suffix == ".joblib" or path.parent.parent.name == "folds"
                                   and path.name in ("best_model.pt", "data_processing.json"))])
        shutil.copytree(m1_dir, m1_best, dirs_exist_ok=True)
    else:
        extraction = copy.deepcopy(cfg["feature_extraction"])
        sensor, sensor_names, time_grids = extract_features(dataset.curves, dataset.sources, extraction,
            dataset.ids, resolve(cfg["feature_extraction"]["cache_dir"]), dataset.curve_columns,
            base_dir=ROOT, return_time_grids=True)
        signature = fingerprint(dataset.digest, extraction, time_grids)
        write_json(m1_dir / "data_processing.json", contract)
        write_json(m1_dir / "configuration.json", {"feature_extraction": extraction,
                                                    "resolved_signal_time_grids": time_grids})
        write_json(m1_dir / "signal_time_grids.json", time_grids)
        write_json(m1_dir / "diagnostics_info.json", {"prediction_diagnostics": "not_applicable",
            "reason": "M1 is a deterministic physical feature extractor. Weight diagnostics are in XGBoost."})
    _save_feature_table(m1_dir, dataset, sensor, sensor_names)
    def external_features(current):
        if current is None:
            return None
        if teacher is not None:
            return teacher.sensor_features(current.curves)
        else:
            values, external_names, _ = extract_features(current.curves, current.sources, extraction,
                current.ids, resolve(cfg["feature_extraction"]["cache_dir"]), current.curve_columns,
                base_dir=ROOT, return_time_grids=True)
            if external_names != sensor_names:
                raise ValueError("test_2 engineered feature order differs from training")
            return values
    external_sensor = external_features(external)
    if family == "gbdt_engineered":
        shutil.copytree(m1_dir, m1_best, dirs_exist_ok=True)
    pp_imp = fit_imputer(dataset.pp[train])
    direct = teacher.pp_features(impute(pp_imp, dataset.pp)) if family == "gbdt_pppt" else dataset.pp
    features = np.concatenate([direct, sensor], axis=1)
    direct_names = [f"pp_embedding_{i}" for i in range(direct.shape[1])] if family == "gbdt_pppt" else dataset.pp_names
    head_names = direct_names + sensor_names
    xgb_dir, xgb_best = stage_dirs(run_dir, "XGBoost", material)
    if external is None:
        clear_test_2_artifacts(xgb_dir)
        clear_test_2_artifacts(xgb_best)
    head_signature = fingerprint(training_config(cfg), dataset.digest, contract, signature, seed)
    head_files = ["best_model.ubj", "imputer.joblib", "best_metrics.json", "feature_schema.json"]
    head_valid = _reuse_checkpoint(cfg, xgb_dir, head_signature, head_files, "xgboost" in stages)
    if "xgboost" in stages and not head_valid:
        provider = (_fold_feature_provider(m1_dir, dataset, family, splits, cfg, pp_imp, features, teacher)
                    if teacher is not None else None)
        xgb_model, head_imp = train_xgb(cfg, dataset, splits, contract, features, head_names,
                                       xgb_dir, seed, provider)
        meta = read_json(xgb_dir / "best_metrics.json")
        write_json(xgb_dir / "best_metrics.json", {**meta, "signature": head_signature})
        save_manifest(xgb_dir, head_signature, head_files)
    elif head_valid:
        xgb_model = XGBRegressor()
        xgb_model.load_model(xgb_dir / "best_model.ubj")
        head_imp = joblib.load(xgb_dir / "imputer.joblib")
        print("  Reusing compatible XGBoost checkpoint", flush=True)
    elif set(stages).issubset({"m1"}):
        return {"run_dir": str(run_dir), "family": family}
    else:
        raise ValueError("XGBoost checkpoint missing/incompatible; run --stages xgboost first")
    predictor = HybridPredictor(family, xgb_model, head_imp, dataset.pp_names, sensor_names,
                                teacher=teacher, pp_imputer=pp_imp)
    predictor.save(xgb_dir / "predictor")
    rows = _stage_evaluation(cfg, xgb_dir, predictor, dataset, splits,
                             "GBDT teacher (measured signals)", sensor, external, external_sensor)
    if teacher is not None:
        rows += _saved_evaluation_rows(m1_dir, "M1 neural (measured signals)",
            (teacher_cfg or cfg).get("evaluation", {}).get("enabled", False))
    promote(xgb_dir, xgb_best)
    if not set(stages) & {"m2", "fusion"}:
        _write_comparison(rows, run_dir, family, material, seed, provenance)
        return {"run_dir": str(run_dir), "family": family}
    m2_cfg = _load_model_config(cfg["configs"]["m2"], seed, optuna_seed)
    # M2 owns its own training/HPO config but uses the authoritative feature/split schema.
    if m2_cfg["data"] != cfg["data"]:
        raise ValueError("M2 and teacher data configurations must match")
    m2_dir, m2_best = stage_dirs(run_dir, "M2", material)
    m2_external = external_dataset(m2_cfg)
    if m2_external is None:
        clear_test_2_artifacts(m2_dir)
        clear_test_2_artifacts(m2_best)
    reconstruction_signature = fingerprint(training_config(m2_cfg), dataset.digest, contract,
                                            signature, array_hash(sensor), seed)
    m2_files = ["best_model.pt", "preprocessing.joblib", "configuration.json", "feature_schema.json", "best_metrics.json"]
    valid = _reuse_checkpoint(m2_cfg, m2_dir, reconstruction_signature, m2_files, "m2" in stages)
    if "m2" in stages and not valid:
        reconstructor, m2_prep = train_m2(m2_cfg, dataset, splits, contract, sensor, sensor_names, m2_dir, seed)
        meta = read_json(m2_dir / "best_metrics.json")
        write_json(m2_dir / "best_metrics.json", {**meta, "signature": reconstruction_signature})
        save_manifest(m2_dir, reconstruction_signature, m2_files)
    elif valid:
        selected_cfg = read_json(m2_dir / "configuration.json")["model"]
        reconstructor = baseline_m2.M2Model(len(dataset.pp_names), len(sensor_names),
            selected_cfg["hidden_dims"], selected_cfg["dropout"])
        reconstructor.load_state_dict(torch.load(m2_dir / "best_model.pt", map_location="cpu", weights_only=True))
        reconstructor.eval()
        m2_prep = joblib.load(m2_dir / "preprocessing.joblib")
        print("  Reusing compatible M2 checkpoint", flush=True)
        evaluate_m2(reconstructor, m2_prep, m2_cfg, dataset, splits, sensor, sensor_names, m2_dir)
    else:
        raise ValueError("M2 checkpoint missing/incompatible; run --stages m2 first")
    evaluate_m2(reconstructor, m2_prep, m2_cfg, dataset, splits, sensor, sensor_names, m2_dir,
                external=m2_external, external_sensor=external_features(m2_external), only_external=True)
    promote(m2_dir, m2_best)
    rows += _saved_evaluation_rows(m2_dir, "M2 feature reconstruction",
                                    m2_cfg.get("evaluation", {}).get("enabled", False))
    if "fusion" in stages:
        fusion_cfg = _load_model_config(cfg["configs"]["fusion"], seed, optuna_seed)
        if fusion_cfg.get("finetuning", {}).get("enabled", False):
            raise ValueError("GBDT Fusion does not support fine-tuning; train M2 for feature reconstruction")
        fusion_dir, fusion_best = stage_dirs(run_dir, "Fusion", material)
        fusion_external = external_dataset(fusion_cfg, data_cfg=cfg["data"])
        if fusion_external is None:
            clear_test_2_artifacts(fusion_dir)
            clear_test_2_artifacts(fusion_best)
        predictor.reconstructor, predictor.m2_preprocessing = reconstructor, m2_prep
        predictor.m2_model_config = read_json(m2_dir / "configuration.json")["model"]
        for parameter in reconstructor.parameters():
            parameter.requires_grad_(False)
        predictor.save(fusion_dir / "predictor")
        rows += _stage_evaluation(fusion_cfg, fusion_dir, predictor, dataset, splits,
                                  "Fusion (PP only)", sensor, fusion_external, external_features(fusion_external))
        score = next(row["MAE"] for row in rows if row["role"] == "Fusion (PP only)" and row["partition"] == "validation")
        write_json(fusion_dir / "best_metrics.json", {"signature": fingerprint(head_signature, reconstruction_signature),
            "source": "validation", "validation_MAE": score})
        write_json(fusion_dir / "configuration.json", fusion_cfg)
        write_json(fusion_dir / "data_processing.json", contract)
        cv_signature = fingerprint(head_signature, reconstruction_signature, fusion_cfg.get("cv", {}))
        cv_manifest = fusion_dir / "cv_manifest.json"
        if (not fusion_cfg.get("checkpoint", {}).get("reuse", True) or not cv_manifest.exists()
                or read_json(cv_manifest).get("signature") != cv_signature):
            provider = (_fold_feature_provider(m1_dir, dataset, family, splits, cfg, pp_imp, features, teacher)
                        if teacher is not None else None)
            fusion_cv(fusion_cfg, read_json(xgb_dir / "configuration.json"),
                read_json(m2_dir / "configuration.json"), dataset, splits, features, sensor_names,
                fusion_dir, seed, provider)
            write_json(cv_manifest, {"signature": cv_signature})
        if fusion_cfg.get("xai", {}).get("enabled", True):
            xai_cfg = fusion_cfg["xai"]
            from .hybrid_xai import run_xai
            from .fusion_xai_sampling import scoped_fusion_xai
            domains = [("train", dataset.pp, dataset.ids, dataset.groups)]
            if fusion_cfg.get("evaluation", {}).get("enabled", False):
                domains.append((internal_test_name(cfg["data"]), dataset.pp[test], dataset.ids[test], dataset.groups[test]))
                if fusion_external is not None:
                    domains.append(("test_2", fusion_external.pp, fusion_external.ids, fusion_external.groups))
            def explain(background, values, ids, destination, partition):
                run_xai(predictor, background, values, ids, destination,
                        {**xai_cfg, "partition": partition, "preselected": True}, f"{material} | {partition}", seed)
            scoped_fusion_xai(explain, dataset.pp[train], dataset.groups[train], domains,
                             fusion_dir / "xai", xai_cfg, seed)
        promote(fusion_dir, fusion_best)
    _write_comparison(rows, run_dir, family, material, seed, provenance)
    return {"run_dir": str(run_dir), "family": family}


def hybrid_main(family):
    parser = argparse.ArgumentParser(description=f"{family}: XGBoost teacher and PP-only Fusion")
    parser.add_argument("--material", type=str.upper, choices=["DOE1", "PP", "ABS", "ALL"], required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--optuna-seed", type=int)
    parser.add_argument("--stages", nargs="+", choices=["m1", "xgboost", "m2", "fusion"],
                        default=["m1", "xgboost"])
    args = parser.parse_args()
    path = args.config or default_config(family, args.material)
    if not path.exists():
        parser.error(f"Model-specific config not found: {path}; supply --config")
    run_hybrid(family, path, args.material, args.run_dir, args.seed, args.optuna_seed, args.stages)
