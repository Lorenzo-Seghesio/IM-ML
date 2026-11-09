"""Independent test_2 inference with frozen models and training preprocessing."""
import copy
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from .dataset_layout import resolve_data_config, label_test_1, divided_dataset, internal_test_name
from .hybrid_data import load_data, read_json, write_json
from .hybrid_diagnostics import save_partition_plots
from .m1_test_point_training import test_point_metrics
from .doe1_columns import doe1_column_name


def external_dataset(cfg, data_cfg=None):
    configured = cfg.get("evaluation", {}).get("test_2_dir")
    if not configured:
        return None
    data = resolve_data_config({**(data_cfg if data_cfg is not None else cfg["data"]),
                                "test_2_dir": configured}, partition="test_2")
    if data is None:
        return None
    return load_data({**cfg, "data": data})


def clear_test_2_artifacts(directory):
    """Remove obsolete evaluation reports when resuming with test_2 disabled."""
    directory = Path(directory)
    paths = list(directory.glob("test_2*"))
    paths += list((directory/"m2_reconstruction").glob("*test_2*"))
    paths += [directory/"xai/test_2"]
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()


def save_weight_evaluation(directory, partition, data, pred, role, plots=True):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pred = np.asarray(pred)
    if pred.shape != data.y.shape or not np.isfinite(pred).all():
        raise ValueError("Evaluation predictions must be finite and aligned")
    mse = float(mean_squared_error(data.y, pred))
    report = {"MAE": float(mean_absolute_error(data.y, pred)), "MSE": mse, "RMSE": mse**.5,
              "R2": float(r2_score(data.y, pred)), "partition": partition, "role": role,
              "source": f"final_{partition}", "n_samples": len(data.y),
              "relative_MAE_percent": 100 * float(np.abs(data.y-pred).mean()) / abs(float(data.y.mean()))}
    report.update(test_point_metrics(data.y, pred, data.groups))
    write_json(directory / f"{partition}_metrics.json", report)
    pd.DataFrame({"part_id": data.ids, "test_point_id": data.groups, "y_true": data.y,
                  "y_pred": pred, "partition": partition}).to_csv(directory / f"{partition}_predictions.csv", index=False)
    pd.DataFrame(report["test_points"]).to_csv(directory / f"{partition}_test_points.csv", index=False)
    if plots:
        save_partition_plots(directory, partition, data.y, pred, [doe1_column_name(data.config["data"]["target_col"])],
                             role, data.groups, data.ids)
    return report


def _copy_external(source, destination):
    if destination is None:
        return
    source, destination = Path(source), Path(destination)
    for path in source.glob("test_2*"):
        if path.is_dir():
            shutil.copytree(path, destination/path.name, dirs_exist_ok=True)
        else:
            shutil.copy2(path, destination/path.name)


def finish_original_stage(cfg, stage, directory, overall=None, model=None, transform=None):
    """Called only after final training/model selection and all scalers are saved."""
    enabled = cfg.get("evaluation", {}).get("enabled", True)
    if not enabled:
        return
    if not cfg.get("evaluation", {}).get("test_2_dir") and not divided_dataset(cfg.get("data", {})):
        return
    directory = Path(directory)
    if (directory/"best_metrics.json").exists():
        write_json(directory/"test_metrics.json", read_json(directory/"best_metrics.json"))
    evaluation_cfg = cfg
    if stage == "M2" and cfg.get("evaluation", {}).get("test_2_dir"):
        from .dataset_layout import ROOT
        teacher_dir = (ROOT/cfg["data"]["features_csv"]).parent
        evaluation_cfg = {**cfg, "data": read_json(teacher_dir/"configuration.json")["data"]}
    external = external_dataset(evaluation_cfg)
    if external is None:
        clear_test_2_artifacts(directory)
        if overall:
            clear_test_2_artifacts(overall)
        label_test_1(directory, cfg["data"])
        if overall:
            label_test_1(overall, cfg["data"])
            write_json(Path(overall)/"test_1_metrics.json", read_json(Path(overall)/"best_metrics.json"))
        return
    plots = cfg.get("evaluation", {}).get("plots", True)
    if not np.isfinite(external.pp).all() or any(not np.isfinite(c).all() for c in external.curves):
        raise ValueError("Original test_2 inputs must be finite; external statistics cannot be fitted")
    if stage == "M1":
        from .hybrid_models import FrozenTeacher
        teacher = FrozenTeacher.load(directory)
        if external.pp_names != teacher.pp_names:
            raise ValueError("test_2 PP feature ordering differs from the trained M1")
        pred = teacher.predict_weight(external.pp, external.curves)
        save_weight_evaluation(directory, "test_2", external, pred, "M1", plots)
    elif stage == "M2":
        from src import Fusion_M1M2_WPred as fm
        from .hybrid_models import FrozenTeacher
        from .fusion_reconstruction import save_fusion_reconstruction
        source = Path(cfg["data"]["features_csv"]).parent
        if not source.is_absolute():
            from .dataset_layout import ROOT
            source = ROOT/source
        teacher = FrozenTeacher.load(source)
        meta = read_json(directory/"best_metrics.json")
        positions = fm.get_f_positions(source, meta["f_cols"])
        targets = teacher.sensor_features(external.curves)[:, positions]
        xs = fm._dict_to_scaler(read_json(source/"data_processing.json")["pp_sc"])
        ys = fm._dict_to_scaler(read_json(directory/"scalers.json")["y_sc"])
        dev = next(model.parameters()).device
        with torch.inference_mode():
            pred = ys.inverse_transform(model(torch.as_tensor(xs.transform(external.pp), dtype=torch.float32, device=dev)).cpu().numpy())
        report = save_fusion_reconstruction(targets, pred, meta["f_cols"], external.ids, external.groups,
            directory/"test_2", "DOE1", partition="test_2", target_scale=ys.scale_, plots=plots)
        write_json(directory/"test_2_metrics.json", report)
    else:
        if stage in ("Encoder", "MultiEncoder"):
            values = np.concatenate(external.curves, axis=1)
        else:
            values = external.pp
        values = transform(values) if transform else values
        if isinstance(model, torch.nn.Module):
            dev = next(model.parameters()).device
            tensor = torch.as_tensor(values, dtype=torch.float32, device=dev)
            if stage in ("Encoder", "MultiEncoder"):
                tensor = tensor[:, None, :]
            model.eval()
            with torch.inference_mode():
                pred = model(tensor).cpu().numpy()
        else:
            pred = model.predict(values)
        save_weight_evaluation(directory, "test_2", external, pred, f"Reference/{stage}", plots)
    # Folder-local IDs may overlap; test_2 is never joined to training feature tables.
    write_json(directory/"test_2_dataset.json", {"namespace": "test_2", "data": external.config["data"],
        "data_fingerprint": external.digest, "part_ids": external.ids.tolist(),
        "n_samples": len(external.ids), "refitted": False})
    _copy_external(directory, overall)
    label_test_1(directory, cfg["data"])
    if overall:
        label_test_1(overall, cfg["data"])
        write_json(Path(overall)/f"{internal_test_name(cfg['data'])}_metrics.json", read_json(Path(overall)/"best_metrics.json"))
