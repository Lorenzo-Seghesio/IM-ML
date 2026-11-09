"""Restore evaluation and downstream contracts for a saved M1 without retraining."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer

from src import M1_PPPT_to_W as m1
from .dataset_layout import resolve_data_config
from .external_evaluation import external_dataset, save_weight_evaluation
from .hybrid_data import ROOT, load_data, read_json, write_json, file_hash
from .hybrid_models import FrozenTeacher


def repair(model_dir, data_dir, test_2_dir, base_config):
    model_dir, data_dir = Path(model_dir).resolve(), Path(data_dir).resolve()
    historical = model_dir / "legacy_artifacts"
    meta = read_json((historical if (historical / "best_metrics.json").exists() else model_dir) / "best_metrics.json")
    contract = read_json(model_dir / "data_processing.json")
    checkpoint = model_dir / "best_model.pt"
    original_hash = file_hash(checkpoint)
    cfg = copy.deepcopy(read_json(base_config))
    id_col, pp_cols = contract["id_col"], contract["pp_cols"]
    scalar = next(p for p in data_dir.glob("*Scalars.csv"))
    external_scalar = next(Path(test_2_dir).glob("*Scalars.csv"))
    columns = list(dict.fromkeys([*pd.read_csv(scalar, nrows=0).columns,
                                 *pd.read_csv(external_scalar, nrows=0).columns]))
    target = contract["y_filter"]["target_col"]
    signals = read_json(model_dir / "feature_summary.json")["curve_signals"]
    sources = []
    for signal, suffix in zip(signals, ("Pressure_1.csv", "Pressure_2.csv", "Temperature.csv")):
        sources.append({"name": signal["source_name"], "path": str(next(data_dir.glob("*" + suffix)))})
    if len(sources) != len(meta["T_list"]):
        raise ValueError("Signal source schema differs from the checkpoint")
    cfg["data"] = {"scalar_csv": str(scalar), "timeseries_csvs": sources,
        "id_col": id_col, "target_col": target, "group_col": contract["group_col"],
        "group_split": contract["group_split"], "edge_group_split": contract["edge_group_split"],
        "drop_cols": [c for c in columns if c not in (id_col, target, *pp_cols)]}
    cfg["model"] = meta["model_cfg"]
    cfg["mode"] = "manual"
    cfg["optuna"]["n_trials"] = 0
    cfg["preprocessing"].update(random_seed=contract["random_seed"], remove_target_outliers=False)
    cfg["evaluation"] = {"enabled": True, "plots": True, "test_2_dir": str(Path(test_2_dir).resolve())}
    hpo = read_json(model_dir / "hpo_best_config.json")
    for saved, name in (("train_lr", "lr"), ("train_weight_decay", "weight_decay"), ("train_batch_size", "batch_size")):
        cfg["training"][name] = hpo[saved]
    cfg["training"]["test_point_training"]["enabled"] = False
    cfg["output"].update(output_dir=str(model_dir.parent), run_best_dir=str(model_dir.parent / "run_best"),
                         best_overall_dir=str(model_dir))
    cfg["recovery"] = {"mode": "saved_checkpoint_only", "training_settings_complete": False,
        "note": "Architecture, optimizer values, scalers and splits are saved originals; unavailable historical settings are template defaults, not recovered training history."}
    # The historical loader filled missing PP values with medians from this entire
    # original dataset. Reconstruct that rule, without using external Test rows.
    raw = pd.read_csv(scalar, index_col=id_col)
    expected_ids = set(contract["train_part_ids"] + contract["val_part_ids"] + contract["test_part_ids"])
    if set(raw.index) != expected_ids:
        raise ValueError("Recovery dataset must contain exactly the saved historical part IDs")
    pp_imputer = SimpleImputer(strategy="median").fit(raw[pp_cols].to_numpy(dtype=np.float32))
    from src.M2_PP_to_F import _dict_to_scaler
    preprocessing = {"pp_imputer": pp_imputer, "pp_scaler": _dict_to_scaler(contract["pp_sc"]),
                     "pt_scalers": contract["pt_scalers"], "curve_widths": meta["T_list"]}
    model = m1.M1Model(len(pp_cols), meta["model_cfg"]).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    teacher = FrozenTeacher(model, preprocessing, pp_cols, sources, meta["model_cfg"])
    data = load_data(cfg)
    assert data.pp_names == pp_cols
    feature_source = historical if (historical / "sensors_features_f.csv").exists() else model_dir
    old_features = pd.read_csv(feature_source / "sensors_features_f.csv", index_col=id_col).reindex(data.ids)
    if old_features.isna().any().any():
        raise ValueError("Historical feature rows are incomplete")
    lookup = {value: i for i, value in enumerate(data.ids)}
    indices = {part: np.array([lookup[value] for value in contract[f"{part}_part_ids"]])
               for part in ("train", "val", "test")}
    with torch.inference_mode():
        cached = model.merge(torch.tensor(np.concatenate([
            old_features.to_numpy(dtype=np.float32), teacher.pp_features(data.pp)], axis=1))).numpy()
    cached_mae = float(np.abs(cached[indices["test"]] - data.y[indices["test"]]).mean())
    if abs(cached_mae - meta["MAE"]) > 5e-5:
        raise ValueError(f"Historical checkpoint/feature replay failed: {cached_mae} vs {meta['MAE']}")
    features = teacher.sensor_features(data.curves)
    predicted = teacher.predict_weight(data.pp, data.curves)
    feature_delta = np.abs(features - old_features.to_numpy())
    backup = model_dir / "legacy_artifacts"
    backup.mkdir(exist_ok=True)
    for name in ("best_metrics.json", "sensors_features_f.csv", "feature_summary.json", "run_info.json"):
        if not (backup / name).exists():
            shutil.copy2(model_dir / name, backup / name)
    write_json(model_dir / "configuration.json", cfg)
    joblib.dump(preprocessing, model_dir / "preprocessing.joblib")
    write_json(model_dir / "signal_schema.json", {"sources": sources, "widths": meta["T_list"]})
    write_json(model_dir / "normalization_factors.json", None)
    refreshed = pd.DataFrame(features, index=pd.Index(data.ids, name=id_col), columns=old_features.columns)
    refreshed.to_csv(model_dir / "sensors_features_f.csv")
    reports = {}
    for part, partition in (("val", "validation"), ("test", "test_1")):
        ix = indices[part]
        subset = copy.copy(data)
        subset.ids, subset.y, subset.groups = data.ids[ix], data.y[ix], data.groups[ix]
        reports[partition] = save_weight_evaluation(model_dir, partition, subset, predicted[ix], "M1", plots=True)
    external = external_dataset(cfg)
    if external.pp_names != pp_cols:
        raise ValueError("External PP schema differs from historical M1")
    external_predictions = teacher.predict_weight(external.pp, external.curves)
    reports["test_2"] = save_weight_evaluation(model_dir, "test_2", external, external_predictions, "M1", plots=True)
    external_features = teacher.sensor_features(external.curves)
    pd.DataFrame(external_features, index=pd.Index(external.ids, name=id_col), columns=old_features.columns).to_csv(
        model_dir / "test_2_sensor_features.csv")
    ext_scalar = pd.read_csv(resolve_data_config({**cfg["data"], "test_2_dir": cfg["evaluation"]["test_2_dir"]},
                                                partition="test_2")["scalar_csv"], index_col=id_col)
    overlaps = []
    if "Start Time" in raw and "Start Time" in ext_scalar:
        overlaps = ext_scalar["Start Time"].isin(raw["Start Time"]).loc[lambda x: x].index.tolist()
    write_json(model_dir / "test_2_dataset.json", {"namespace": "Test", "part_ids": external.ids.tolist(),
        "n_samples": len(external.ids), "data": external.config["data"],
        "overlapping_historical_start_times": overlaps, "refitted": False})
    write_json(model_dir / "best_metrics.json", {**meta, **reports["test_1"],
        "validation_MAE": reports["validation"]["MAE"], "recovered_from_checkpoint": True})
    summary = read_json(model_dir / "feature_summary.json")
    summary["n_samples"] = len(data.ids)
    write_json(model_dir / "feature_summary.json", summary)
    recovery = {"mode": "saved_checkpoint_only", "checkpoint_sha256": original_hash, "retrained": False,
        "scalers_refitted": False, "pp_imputation": "Reconstructed original full-historical-dataset median rule",
        "original_saved_test_MAE": meta["MAE"], "cached_feature_replay_test_MAE": cached_mae,
        "live_test_1_MAE": reports["test_1"]["MAE"], "test_2_MAE": reports["test_2"]["MAE"],
        "max_refreshed_feature_difference": float(feature_delta.max()),
        "mean_refreshed_feature_difference": float(feature_delta.mean()),
        "refreshed_features": True, "historical_artifacts_backup": "legacy_artifacts",
        "train_count": len(indices["train"]), "validation_count": len(indices["val"]),
        "test_1_count": len(indices["test"]), "test_2_count": len(external.ids),
        "unavailable_history": ["training_history.json", "cv_fold_*_training.json"],
        "overlapping_historical_start_times": overlaps}
    write_json(model_dir / "recovery_info.json", recovery)
    assert file_hash(checkpoint) == original_hash
    run_copy = model_dir.parent / "run_best" / contract["run_ts"]
    info = read_json(model_dir / "run_info.json")
    old_info = read_json(backup / "run_info.json")
    info.update(historical_run_best_dir=old_info["run_best_dir"], run_best_dir=str(run_copy),
                recovered_from_legacy=True)
    write_json(model_dir / "run_info.json", info)
    if (run_copy / "best_model.pt").is_file() and file_hash(run_copy / "best_model.pt") == original_hash:
        for path in model_dir.iterdir():
            if path.name == "best_model.pt":
                continue
            if path.is_dir():
                shutil.copytree(path, run_copy / path.name, dirs_exist_ok=True)
            else:
                shutil.copy2(path, run_copy / path.name)
    root = model_dir.parents[1]
    config_dir = root / "recovered_configs"
    for stage, filename in (("M2", "M2_DoE1_config.json"), ("Fusion", "Fusion_DoE1_config.json"), ("RefModels", "RefModels_DoE1_config.json")):
        stage_cfg = read_json(ROOT / "config/DoE1" / filename)
        stage_cfg["data"] = copy.deepcopy(cfg["data"])
        stage_cfg.setdefault("evaluation", {}).update(cfg["evaluation"])
        if stage == "M2":
            stage_cfg["data"]["drop_cols"].append(target)
            stage_cfg["data"]["features_csv"] = str(model_dir / "sensors_features_f.csv")
            stage_cfg["output"].update(output_dir=str(root / "M2"), run_best_dir=str(root / "M2/run_best"),
                                       best_overall_dir=str(root / "M2/best_overall"))
        elif stage == "Fusion":
            stage_cfg["models"] = {"m1_best_dir": str(model_dir), "m2_best_dir": str(root / "M2/best_overall")}
            stage_cfg["output"]["output_base_dir"] = str(root / "Fusion")
        else:
            stage_cfg.update(m1_split_dir=str(model_dir), output_base_dir=str(root / "RefModels"))
        write_json(config_dir / filename, stage_cfg)
    return recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--test-2-dir", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=ROOT / "config/DoE1/M1_DoE1_config.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(repair(args.model_dir, args.data_dir, args.test_2_dir, args.base_config), indent=2))


if __name__ == "__main__":
    main()
