"""Frozen-feature XGBoost teachers and PP-only M2/Fusion inference.

Network optimizers and architecture search reuse M1/M2 baseline implementations.
XGBoost is only ever called through NumPy prediction/fit APIs.
"""
from __future__ import annotations

import copy
from pathlib import Path
import shutil

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor

from src import M1_PPPT_to_W as m1
from src import M2_PP_to_F as m2
from .hybrid_data import (
    read_json, write_json, fingerprint, fit_imputer, impute, cv_splits, inner_split,
)
from .m1_test_point_training import test_point_metrics
from .optuna_seeding import resolve_optuna_seed
from .pipeline_runtime import stage_directory

FAMILIES = ("gbdt_pppt", "gbdt_ppdirect_pt", "gbdt_engineered")


def device_for(cfg):
    requested = cfg.get("training", {}).get("device", "auto")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available()
                        else "cpu" if requested == "auto" else requested)


def study_search(objective, cfg, directory, seed, selection_metric="validation_MAE"):
    sampler_seed = resolve_optuna_seed(cfg)
    study = optuna.create_study(direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=sampler_seed,
            n_startup_trials=cfg.get("n_startup_trials", 5)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=cfg.get("n_startup_trials", 5)),
        study_name=f"{cfg.get('study_name', 'hybrid')}_{fingerprint(str(directory), cfg, seed)[:12]}",
        storage=cfg.get("storage"),
        load_if_exists=bool(cfg.get("storage")))
    study.optimize(objective, n_trials=int(cfg["n_trials"]),
                   timeout=cfg.get("timeout_s"), n_jobs=1)
    study.trials_dataframe().to_csv(Path(directory) / "hpo_trials.csv", index=False)
    write_json(Path(directory) / "hpo_best_config.json", {
        "params": study.best_trial.params, selection_metric: study.best_value,
        "selection_metric": selection_metric,
        "sampler_seed": sampler_seed})
    return study


def hpo_enabled(cfg):
    return cfg.get("enabled", True) and int(cfg.get("n_trials", 0)) > 0


def feature_scaler(values, method):
    if method == "minmax":
        return MinMaxScaler().fit(values)
    if method == "standard":
        return StandardScaler().fit(values)
    raise ValueError("Neural feature normalization must be minmax or standard")


def metrics(y, prediction, groups):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    error = prediction - y
    mse = float(np.mean(error ** 2))
    denominator = float(np.sum((y - y.mean()) ** 2))
    result = {"MAE": float(np.mean(np.abs(error))), "MSE": mse, "RMSE": mse ** .5,
              "R2": 1 - float(np.sum(error**2)) / denominator if denominator else None,
              "relative_MAE_percent": 100 * float(np.mean(np.abs(error))) / abs(float(y.mean()))
                  if y.mean() else None}
    result.update(test_point_metrics(y, prediction, groups))
    return result


def save_evaluation(directory, split, dataset, indices, prediction, label, evaluation_cfg=None):
    directory = Path(directory)
    result = metrics(dataset.y[indices], prediction, dataset.groups[indices])
    write_json(directory / f"{split}_metrics.json", {"role": label, "partition": split, **result})
    pd.DataFrame({"part_id": dataset.ids[indices], "test_point_id": dataset.groups[indices],
                  "y_true": dataset.y[indices], "y_pred": prediction,
                  "error_g": prediction - dataset.y[indices]}).to_csv(
                      directory / f"{split}_predictions.csv", index=False)
    pd.DataFrame(result["test_points"]).to_csv(directory / f"{split}_test_points.csv", index=False)
    if (evaluation_cfg if evaluation_cfg is not None else dataset.config.get("evaluation", {})).get("plots", True):
        from .batch_test_diagnostics import save_batch_test_diagnostics
        save_batch_test_diagnostics(dataset.y[indices], prediction, dataset.groups[indices],
            dataset.ids[indices], directory / split, target_names=[dataset.config["data"]["target_col"]],
            title=f"{label} — {split}")
    return result


def stage_dirs(run_dir, stage, material):
    parent = stage_directory(run_dir, stage, material)
    output = parent / "run_best" / "model"
    output.mkdir(parents=True, exist_ok=True)
    return output, parent / "best_overall"


def promote(directory, overall):
    """Only compare validation metrics; artifacts are already isolated by repetition."""
    current = read_json(Path(directory) / "best_metrics.json")
    previous = read_json(Path(overall) / "best_metrics.json") if (Path(overall) / "best_metrics.json").exists() else {}
    if (current.get("signature") != previous.get("signature") or
            current["validation_MAE"] <= previous.get("validation_MAE", float("inf"))):
        shutil.copytree(directory, overall, dirs_exist_ok=True)


def _train_teacher_once(dataset, train, val, model_cfg, train_cfg, seed, device):
    pp_imp = fit_imputer(dataset.pp[train])
    pp = impute(pp_imp, dataset.pp)
    curve_imputers = [fit_imputer(matrix[train]) for matrix in dataset.curves]
    curves = [impute(imp, matrix) for imp, matrix in zip(curve_imputers, dataset.curves)]
    pp_sc, pt_sc = m1.fit_scalers(pp[train], [matrix[train] for matrix in curves])
    xtr, ctr = m1.apply_scalers(pp[train], [matrix[train] for matrix in curves], pp_sc, pt_sc)
    xval, cval = m1.apply_scalers(pp[val], [matrix[val] for matrix in curves], pp_sc, pt_sc)
    torch.manual_seed(seed)
    model = m1.M1Model(pp.shape[1], model_cfg).to(device)
    trained, score = m1.train_fold(model, xtr, ctr, dataset.y[train], xval, cval,
        dataset.y[val], {**train_cfg, "mini_batch_seed": seed}, device,
        dataset.groups[train], dataset.groups[val])
    preprocessing = {"pp_imputer": pp_imp, "curve_imputers": curve_imputers,
                     "pp_scaler": pp_sc, "pt_scalers": pt_sc,
                     "curve_widths": [matrix.shape[1] for matrix in dataset.curves]}
    return trained, preprocessing, score


def _save_teacher(model, preprocessing, model_cfg, dataset, split, contract, directory, score):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / "best_model.pt")
    joblib.dump(preprocessing, directory / "preprocessing.joblib")
    write_json(directory / "best_metrics.json", {"model_cfg": model_cfg, "n_pp": len(dataset.pp_names),
        "validation_MAE": score, "source": "validation"})
    write_json(directory / "data_processing.json", {**contract,
        "pp_sc": m1._scaler_to_dict(preprocessing["pp_scaler"]),
        "pt_scalers": preprocessing["pt_scalers"]})
    write_json(directory / "training_history.json", model.m1_training_info)
    write_json(directory / "signal_schema.json", {"sources": dataset.sources,
        "widths": [matrix.shape[1] for matrix in dataset.curves]})


def train_teacher(cfg, dataset, splits, contract, directory, seed):
    """Model B's direct-PP M1, using the baseline sampler/loss/optimizer and HPO."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    train, val, test = splits
    dev = np.concatenate([train, val])
    device = device_for(cfg)
    model_cfg, train_cfg = copy.deepcopy(cfg["model"]), copy.deepcopy(cfg["training"])
    hpo = cfg.get("optuna", {})
    if hpo_enabled(hpo) and cfg.get("mode", "manual") == "optuna":
        def objective(trial):
            model_up, train_up = m1.suggest_hyperparams(trial, hpo["search_space"],
                                                      len(dataset.pp_names), model_cfg)
            fold_cfg = {**train_cfg, **train_up, "epochs": hpo["epochs"],
                        "patience": hpo["patience"], "print_every": 999999}
            scores = []
            for number, (tv, holdout) in enumerate(cv_splits(dataset, dev, hpo["n_folds"], seed)):
                tr, va = inner_split(dataset, tv, fold_cfg, seed + number)
                model, prep, _ = _train_teacher_once(dataset, tr, va, {**model_cfg, **model_up},
                                                     fold_cfg, seed + number, device)
                predictor = FrozenTeacher(model, prep, dataset.pp_names, dataset.sources)
                scores.append(float(np.mean(np.abs(predictor.predict_weight(
                    dataset.pp[holdout], [matrix[holdout] for matrix in dataset.curves]) - dataset.y[holdout]))))
                trial.report(float(np.mean(scores)), number)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(scores))
        study = study_search(objective, hpo, directory, seed)
        mu, tu = m1.suggest_hyperparams(optuna.trial.FixedTrial(study.best_params),
                                       hpo["search_space"], len(dataset.pp_names), model_cfg)
        model_cfg.update(mu)
        train_cfg.update(tu)
    final_cfg = {**train_cfg, "epochs": train_cfg.get("final_epochs", train_cfg["epochs"]),
                 "patience": train_cfg.get("final_patience", train_cfg["patience"])}
    model, prep, score = _train_teacher_once(dataset, train, val, model_cfg, final_cfg, seed, device)
    _save_teacher(model, prep, model_cfg, dataset, splits, contract, directory, score)
    write_json(directory / "configuration.json", {**cfg, "model": model_cfg, "training": final_cfg})
    # Optional fold-specific encoders keep outer held-out TPs out of encoder fitting.
    cv = cfg.get("cv", {})
    if cv.get("save_fold_checkpoints", False):
        for number, (tv, holdout) in enumerate(cv_splits(dataset, dev, cv["n_folds"], seed)):
            tr, va = inner_split(dataset, tv, train_cfg, seed + number)
            fold_model, fold_prep, fold_score = _train_teacher_once(
                dataset, tr, va, model_cfg, train_cfg, seed + number, device)
            fold_contract = {**contract, "train_part_ids": dataset.ids[tr].tolist(),
                "val_part_ids": dataset.ids[va].tolist(), "test_part_ids": dataset.ids[holdout].tolist()}
            _save_teacher(fold_model, fold_prep, model_cfg, dataset, (tr, va, holdout),
                           fold_contract, directory / "folds" / f"fold_{number}", fold_score)
    return FrozenTeacher(model, prep, dataset.pp_names, dataset.sources, model_cfg)


class FrozenTeacher:
    def __init__(self, model, preprocessing, pp_names, sources, model_config=None):
        self.model, self.preprocessing = model.cpu().eval(), preprocessing
        self.pp_names, self.sources = list(pp_names), sources
        self.model_config = model_config
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def load(cls, directory, dataset=None):
        directory = Path(directory)
        meta = read_json(directory / "best_metrics.json")
        contract = read_json(directory / "data_processing.json")
        cfg = meta["model_cfg"]
        names = contract["pp_cols"]
        model = m1.M1Model(len(names), cfg)
        state = torch.load(directory / "best_model.pt", map_location="cpu", weights_only=True)
        state = {(key.replace("encoder.", "encoders.0.", 1) if key.startswith("encoder.") else key): value
                 for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        if (directory / "preprocessing.joblib").exists():
            prep = joblib.load(directory / "preprocessing.joblib")
        else:
            prep = {"pp_scaler": m2._dict_to_scaler(contract["pp_sc"]),
                    "pt_scalers": contract["pt_scalers"] if "pt_scalers" in contract else
                        [{"min": contract["pt_min"], "max": contract["pt_max"]}]}
        saved_cfg = read_json(directory / "configuration.json") if (directory / "configuration.json").exists() else {}
        sources = saved_cfg.get("data", {}).get("timeseries_csvs", [])
        schema_path = directory / "signal_schema.json"
        if schema_path.exists():
            sources = read_json(schema_path)["sources"]
        widths = meta.get("T_list", read_json(schema_path).get("widths") if schema_path.exists() else None)
        if widths:
            prep["curve_widths"] = widths
        if dataset is not None:
            if dataset.pp_names != names:
                raise ValueError("Pretrained M1 PP feature ordering differs from configured inputs")
            if widths and widths != [matrix.shape[1] for matrix in dataset.curves]:
                raise ValueError("Pretrained M1 curve widths differ from input data")
            if sources and [(s.get("name"), s.get("path")) for s in sources] != [
                    (s.get("name"), s.get("path")) for s in dataset.sources]:
                raise ValueError("Pretrained M1 signal names, order or paths differ from configured inputs")
            if model.n_curves != len(dataset.curves):
                raise ValueError("Pretrained M1 signal count differs from input data")
            sources = dataset.sources
        return cls(model, prep, names, sources, cfg)

    def _pp(self, values):
        prep = self.preprocessing
        if "pp_imputer" in prep:
            values = impute(prep["pp_imputer"], values)
        elif not np.isfinite(values).all():
            raise ValueError("Pretrained M1 has no saved imputer; supply finite PP inputs")
        return prep["pp_scaler"].transform(values).astype(np.float32)

    def pp_features(self, pp, batch_size=512):
        scaled = self._pp(pp)
        with torch.inference_mode():
            return np.concatenate([self.model.pp_mlp(torch.from_numpy(scaled[i:i+batch_size])).numpy()
                                   for i in range(0, len(pp), batch_size)])

    def sensor_features(self, curves, batch_size=512):
        if curves is None or len(curves) != self.model.n_curves:
            raise ValueError("Wrong number of sensor curves")
        if len({len(matrix) for matrix in curves}) != 1:
            raise ValueError("Sensor curve rows must align")
        widths = self.preprocessing.get("curve_widths")
        if widths and widths != [matrix.shape[1] for matrix in curves]:
            raise ValueError("Sensor widths differ from the saved schema")
        scaled = []
        for i, (matrix, sc) in enumerate(zip(curves, self.preprocessing["pt_scalers"])):
            if "curve_imputers" in self.preprocessing:
                matrix = impute(self.preprocessing["curve_imputers"][i], matrix)
            elif not np.isfinite(matrix).all():
                raise ValueError("Pretrained M1 has no signal imputer; supply finite curves")
            scaled.append(((matrix - sc["min"]) / (sc["max"] - sc["min"] + 1e-8)).astype(np.float32))
        if len(scaled) != self.model.n_curves:
            raise ValueError("Wrong number of sensor curves")
        with torch.inference_mode():
            return np.concatenate([self.model.get_f([torch.from_numpy(x[i:i+batch_size, None, :])
                for x in scaled]).numpy() for i in range(0, len(scaled[0]), batch_size)])

    def predict_weight(self, pp, curves):
        # Use the saved neural merge only for M1_pp diagnostics/training, never hybrid inference.
        features = np.concatenate([self.sensor_features(curves), self.pp_features(pp)], axis=1)
        with torch.inference_mode():
            return self.model.merge(torch.from_numpy(features)).numpy()


def xgb_suggest(trial, search_space):
    parameters = {}
    integer_keys = {"n_estimators", "max_depth", "max_leaves", "max_bin", "min_child_weight"}
    for key, spec in search_space.items():
        if key.startswith("_"):
            continue
        if isinstance(spec, dict):
            if "choices" in spec:
                parameters[key] = trial.suggest_categorical(key, spec["choices"])
            elif spec.get("type") == "int":
                parameters[key] = trial.suggest_int(key, spec["low"], spec["high"], log=spec.get("log", False))
            else:
                parameters[key] = trial.suggest_float(key, spec["low"], spec["high"], log=spec.get("log", False))
        elif isinstance(spec, list):
            if len(spec) != 2:
                raise ValueError(f"Use choices for categorical XGBoost parameter {key}")
            method = trial.suggest_int if key in integer_keys else trial.suggest_float
            parameters[key] = method(key, *spec)
        else:
            parameters[key] = spec
    return parameters


def fit_xgb(train_x, train_y, val_x, val_y, parameters, training, seed):
    params = {**parameters, "random_state": seed, "eval_metric": "mae"}
    patience = int(training.get("early_stopping_rounds", 30))
    if patience > 0:
        params["early_stopping_rounds"] = patience
    model = XGBRegressor(**params)
    model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)
    return model


def train_xgb(cfg, dataset, splits, contract, features, names, directory, seed,
              fold_features=None):
    train, val, test = splits
    parameters = copy.deepcopy(cfg["xgboost"])
    hpo, training = cfg.get("optuna", {}), cfg["training"]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dev = np.concatenate([train, val])
    cv = cfg.get("cv", {})
    caveats = []
    fold_cache = {}
    def prepare(n_folds, settings):
        key = (n_folds, settings.get("val_fraction", .15))
        if key not in fold_cache:
            prepared = []
            for number, (tv, holdout) in enumerate(cv_splits(dataset, dev, n_folds, seed)):
                tr, va = inner_split(dataset, tv, settings, seed + number)
                matrix, note = fold_features(number, tr, va, holdout) if fold_features else (features, None)
                imp = fit_imputer(matrix[tr])
                prepared.append((tr, va, holdout, imp, matrix))
                if note:
                    caveats.append(note)
            fold_cache[key] = prepared
        return fold_cache[key]
    def objective(trial):
        params = {**parameters, **xgb_suggest(trial, hpo["search_space"])}
        scores = []
        for number, (tr, va, ho, imp, matrix) in enumerate(prepare(hpo["n_folds"], {**training, **hpo})):
            model = fit_xgb(impute(imp, matrix[tr]), dataset.y[tr],
                impute(imp, matrix[va]), dataset.y[va], params, training, seed + number)
            score = float(np.mean(np.abs(model.predict(impute(imp, matrix[ho])) - dataset.y[ho])))
            scores.append(score)
            trial.report(float(np.mean(scores)), number)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))
    if hpo_enabled(hpo):
        study = study_search(objective, hpo, directory, seed)
        parameters.update(xgb_suggest(optuna.trial.FixedTrial(study.best_params), hpo["search_space"]))
    if cv.get("enabled", True):
        records = []
        for number, (tr, va, ho, imp, matrix) in enumerate(prepare(cv.get("n_folds", 3), training)):
            model = fit_xgb(impute(imp, matrix[tr]), dataset.y[tr],
                impute(imp, matrix[va]), dataset.y[va], parameters, training, seed + number)
            prediction = model.predict(impute(imp, matrix[ho]))
            report = metrics(dataset.y[ho], prediction, dataset.groups[ho])
            records.append({"fold": number, **{key: value for key, value in report.items() if key != "test_points"}})
            pd.DataFrame({"part_id": dataset.ids[ho], "y_true": dataset.y[ho],
                          "y_pred": prediction}).to_csv(directory / f"cv_fold_{number}_predictions.csv", index=False)
            write_json(directory / f"cv_fold_{number}_split.json", {
                "train_part_ids": dataset.ids[tr].tolist(), "val_part_ids": dataset.ids[va].tolist(),
                "heldout_part_ids": dataset.ids[ho].tolist()})
        pd.DataFrame(records).to_csv(directory / "cv_fold_metrics.csv", index=False)
    imp = fit_imputer(features[train])
    model = fit_xgb(impute(imp, features[train]), dataset.y[train],
        impute(imp, features[val]), dataset.y[val], parameters, training, seed)
    model.save_model(directory / "best_model.ubj")
    joblib.dump(imp, directory / "imputer.joblib")
    write_json(directory / "feature_schema.json", {"columns": names, "scaling": "none"})
    write_json(directory / "configuration.json", {**cfg, "xgboost": parameters})
    write_json(directory / "data_processing.json", contract)
    write_json(directory / "cv_interpretation.json", {"notes": sorted(set(caveats)),
        "independent_test_used_in_hpo": False,
        "description": "Head-only CV is conditional on pretrained representations. Even fold-specific "
                       "checkpoints do not undo architecture selection on development data."})
    result = save_evaluation(directory, "validation", dataset, val,
                            model.predict(impute(imp, features[val])), "M1 teacher (measured signals)")
    if cfg.get("evaluation", {}).get("enabled", False):
        save_evaluation(directory, "test", dataset, test,
                        model.predict(impute(imp, features[test])), "M1 teacher (measured signals)")
    write_json(directory / "best_metrics.json", {"source": "validation", "validation_MAE": result["MAE"],
        **result, "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else None})
    return model, imp


def train_m2(cfg, dataset, splits, contract, sensor_features, sensor_names, directory, seed):
    train, val, test = splits
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Missing engineered descriptors are imputed from this training partition only.
    pp_imp, target_imp = fit_imputer(dataset.pp[train]), fit_imputer(sensor_features[train])
    x, y = impute(pp_imp, dataset.pp), impute(target_imp, sensor_features)
    model_cfg, training = copy.deepcopy(cfg["model"]), copy.deepcopy(cfg["training"])
    device = device_for(cfg)
    dev = np.concatenate([train, val])
    hpo = cfg.get("optuna", {})
    normalization = cfg.get("normalization", {"pp": "minmax", "targets": "minmax"})
    if training.get("loss", "mae") != "mae":
        raise ValueError("M2 reconstruction uses configurable MAE training; training.loss must be mae")
    def fit_fold(mc, tc, tr, va, ho, fold_seed):
        xi, yi = fit_imputer(dataset.pp[tr]), fit_imputer(sensor_features[tr])
        xf, yf = impute(xi, dataset.pp), impute(yi, sensor_features)
        xs = feature_scaler(xf[tr], normalization.get("pp", "minmax"))
        ys = feature_scaler(yf[tr], normalization.get("targets", "minmax"))
        torch.manual_seed(fold_seed)
        model = m2.M2Model(x.shape[1], y.shape[1], mc["hidden_dims"], mc["dropout"]).to(device)
        model, _ = m2.train_fold(model, xs.transform(xf[tr]), ys.transform(yf[tr]),
            xs.transform(xf[va]), ys.transform(yf[va]), tc, device)
        return m2.evaluate_model(model, xs.transform(xf[ho]), yf[ho], ys, sensor_names, device)
    if cfg.get("mode") == "optuna" and hpo_enabled(hpo):
        def objective(trial):
            mu, tu = m2.suggest_hyperparams(trial, hpo["search_space"], x.shape[1])
            scores = []
            for number, (tv, ho) in enumerate(cv_splits(dataset, dev, hpo["n_folds"], seed)):
                fold_training = {**training, **tu, "epochs": hpo["epochs"],
                                 "patience": hpo["patience"], "print_every": 999999}
                tr, va = inner_split(dataset, tv, {**fold_training, **hpo}, seed + number)
                report = fit_fold({**model_cfg, **mu}, fold_training, tr, va, ho, seed + number)
                scores.append(report["mean_NMAE"])
                trial.report(float(np.mean(scores)), number)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(scores))
        study = study_search(objective, hpo, directory, seed, "validation_mean_NMAE")
        mu, tu = m2.suggest_hyperparams(optuna.trial.FixedTrial(study.best_params),
                                       hpo["search_space"], x.shape[1])
        model_cfg.update(mu)
        training.update(tu)
    cv = cfg.get("cv", {})
    if cv.get("enabled", False):
        records = []
        for number, (tv, ho) in enumerate(cv_splits(dataset, dev, cv.get("n_folds", training["k_folds"]), seed)):
            tr, va = inner_split(dataset, tv, training, seed + number)
            report = fit_fold(model_cfg, training, tr, va, ho, seed + number)
            records.append({"fold": number, **{key: value for key, value in report.items() if key not in ("pred", "per_target")}})
            write_json(directory / f"cv_fold_{number}_metrics.json", {key: value for key, value in report.items() if key != "pred"})
            output = pd.DataFrame(report["pred"], columns=sensor_names)
            output.insert(0, "part_id", dataset.ids[ho])
            output.to_csv(directory / f"cv_fold_{number}_predictions.csv", index=False)
        pd.DataFrame(records).to_csv(directory / "cv_fold_metrics.csv", index=False)
    xs = feature_scaler(x[train], normalization.get("pp", "minmax"))
    ys = feature_scaler(y[train], normalization.get("targets", "minmax"))
    torch.manual_seed(seed)
    model = m2.M2Model(x.shape[1], y.shape[1], model_cfg["hidden_dims"], model_cfg["dropout"]).to(device)
    final_training = {**training, "epochs": training.get("final_epochs", training["epochs"]),
                      "patience": training.get("final_patience", training["patience"])}
    model, _ = m2.train_fold(model, xs.transform(x[train]), ys.transform(y[train]),
        xs.transform(x[val]), ys.transform(y[val]), final_training, device)
    model = model.cpu().eval()
    torch.save(model.state_dict(), directory / "best_model.pt")
    write_json(directory / "training_history.json", model.m2_training_info)
    preprocessing = {"pp_imputer": pp_imp, "target_imputer": target_imp,
                     "x_scaler": xs, "y_scaler": ys}
    joblib.dump(preprocessing, directory / "preprocessing.joblib")
    write_json(directory / "configuration.json", {**cfg, "model": model_cfg, "training": training})
    write_json(directory / "feature_schema.json", {"pp_columns": dataset.pp_names, "sensor_columns": sensor_names})
    write_json(directory / "data_processing.json", contract)
    reports = {}
    for name, indices in [("validation", val)] + ([("test", test)] if cfg.get("evaluation", {}).get("enabled", False) else []):
        report = m2.evaluate_model(model, xs.transform(x[indices]), y[indices], ys, sensor_names, torch.device("cpu"))
        pred = report.pop("pred")
        # Missing actual descriptors are not scored as if training-median fills were observations.
        actual = sensor_features[indices]
        observed = np.isfinite(actual)
        errors = np.where(observed, np.abs(pred - actual), np.nan)
        counts = observed.sum(axis=0)
        feature_mae = np.divide(np.nansum(errors, axis=0), counts,
                                out=np.full(len(counts), np.nan), where=counts > 0)
        report["observed_feature_MAE"] = float(np.nanmean(feature_mae)) if np.any(counts) else None
        pd.DataFrame({"feature": sensor_names, "MAE": feature_mae, "n_observed": counts}).to_csv(
            directory / f"{name}_feature_metrics.csv", index=False)
        output = pd.DataFrame({"part_id": dataset.ids[indices]})
        for i, column in enumerate(sensor_names):
            output[f"{column}__true"] = actual[:, i]
            output[f"{column}__pred"] = pred[:, i]
        output.to_csv(directory / f"{name}_feature_predictions.csv", index=False)
        write_json(directory / f"{name}_metrics.json", report)
        reports[name] = report
    write_json(directory / "best_metrics.json", {"source": "validation", "model_cfg": model_cfg,
        "n_in": x.shape[1], "n_out": y.shape[1], "validation_MAE": reports["validation"]["mean_NMAE"],
        "selection_metric": "validation_mean_NMAE", **reports["validation"]})
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, preprocessing


class HybridPredictor:
    """Self-contained PP-only predictor; no sensor/target arguments at inference."""
    def __init__(self, family, xgb, head_imputer, pp_names, sensor_names,
                 teacher=None, reconstructor=None, m2_preprocessing=None, pp_imputer=None):
        self.family, self.xgb, self.head_imputer = family, xgb, head_imputer
        self.pp_names, self.sensor_names = list(pp_names), list(sensor_names)
        self.teacher, self.reconstructor = teacher, reconstructor
        self.m2_preprocessing, self.pp_imputer = m2_preprocessing, pp_imputer

    def _array(self, pp):
        if isinstance(pp, pd.DataFrame):
            if list(pp.columns) != self.pp_names:
                raise ValueError("PP DataFrame columns/order differ from the saved schema")
            pp = pp.to_numpy(np.float32)
        pp = np.asarray(pp, np.float32)
        if pp.ndim != 2 or pp.shape[1] != len(self.pp_names) or len(pp) == 0:
            raise ValueError("PP inputs must be a non-empty matrix with the saved feature width")
        return pp

    def direct_features(self, pp):
        if self.family == "gbdt_pppt":
            return self.teacher.pp_features(impute(self.pp_imputer, pp))
        return pp

    def predict_features(self, pp):
        pp = self._array(pp)
        if self.reconstructor is None:
            raise ValueError("Load the Fusion checkpoint to predict features from PP")
        prep = self.m2_preprocessing
        x = prep["x_scaler"].transform(impute(prep["pp_imputer"], pp)).astype(np.float32)
        with torch.inference_mode():
            scaled = np.concatenate([self.reconstructor(torch.from_numpy(x[i:i+512])).numpy()
                                     for i in range(0, len(x), 512)])
        return prep["y_scaler"].inverse_transform(scaled)

    def predict_paths(self, indirect_pp, direct_pp):
        indirect_pp, direct_pp = self._array(indirect_pp), self._array(direct_pp)
        if len(indirect_pp) != len(direct_pp):
            raise ValueError("Direct/indirect PP rows must align")
        features = np.concatenate([self.direct_features(direct_pp), self.predict_features(indirect_pp)], axis=1)
        return self.xgb.predict(impute(self.head_imputer, features))

    def predict(self, pp):
        return self.predict_paths(pp, pp)

    def predict_teacher(self, pp, curves=None, engineered=None):
        pp = self._array(pp)
        if isinstance(engineered, pd.DataFrame):
            if list(engineered.columns) != self.sensor_names:
                raise ValueError("Engineered feature columns/order differ from the saved schema")
            engineered = engineered.to_numpy(np.float32)
        sensors = engineered if self.family == "gbdt_engineered" else self.teacher.sensor_features(curves)
        if sensors is None or sensors.shape != (len(pp), len(self.sensor_names)):
            raise ValueError("Measured sensor features must match the saved schema")
        features = np.concatenate([self.direct_features(pp), sensors], axis=1)
        return self.xgb.predict(impute(self.head_imputer, features))

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # Portable paths: no live class pickle, no absolute dependencies on training folders.
        self.xgb.save_model(directory / "xgboost.ubj")
        joblib.dump({"head_imputer": self.head_imputer, "pp_imputer": self.pp_imputer,
                     "m2_preprocessing": self.m2_preprocessing}, directory / "preprocessing.joblib")
        meta = {"version": 1, "family": self.family, "pp_columns": self.pp_names,
                "sensor_columns": self.sensor_names, "has_m2": self.reconstructor is not None}
        if self.teacher is not None:
            torch.save({key: value for key, value in self.teacher.model.state_dict().items()
                        if not key.startswith("merge.")}, directory / "teacher.pt")
            joblib.dump(self.teacher.preprocessing, directory / "teacher_preprocessing.joblib")
            # Store the resolved architecture alongside the state, including direct-PP variants.
            meta["teacher_model_cfg"] = self.teacher.model_config
            meta["sources"] = self.teacher.sources
        if self.reconstructor is not None:
            torch.save(self.reconstructor.state_dict(), directory / "m2.pt")
            meta["m2_model_cfg"] = self.m2_model_config
        write_json(directory / "predictor.json", meta)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = read_json(directory / "predictor.json")
        if meta["version"] != 1:
            raise ValueError("Unsupported hybrid checkpoint version")
        xgb = XGBRegressor()
        xgb.load_model(directory / "xgboost.ubj")
        prep = joblib.load(directory / "preprocessing.joblib")
        teacher = None
        if "teacher_model_cfg" in meta:
            model = m1.M1Model(len(meta["pp_columns"]), meta["teacher_model_cfg"])
            model.merge = nn.Identity()  # The saved inference teacher has no neural fusion head.
            model.load_state_dict(torch.load(directory / "teacher.pt", map_location="cpu", weights_only=True))
            teacher = FrozenTeacher(model, joblib.load(directory / "teacher_preprocessing.joblib"),
                                    meta["pp_columns"], meta["sources"])
            teacher.model_config = meta["teacher_model_cfg"]
        reconstructor = None
        if meta["has_m2"]:
            cfg = meta["m2_model_cfg"]
            reconstructor = m2.M2Model(len(meta["pp_columns"]), len(meta["sensor_columns"]),
                                       cfg["hidden_dims"], cfg["dropout"])
            reconstructor.load_state_dict(torch.load(directory / "m2.pt", map_location="cpu", weights_only=True))
            reconstructor.eval()
            for parameter in reconstructor.parameters():
                parameter.requires_grad_(False)
        result = cls(meta["family"], xgb, prep["head_imputer"], meta["pp_columns"],
            meta["sensor_columns"], teacher, reconstructor, prep["m2_preprocessing"], prep["pp_imputer"])
        if meta["has_m2"]:
            result.m2_model_config = meta["m2_model_cfg"]
        return result
