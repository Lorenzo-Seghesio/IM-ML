"""Read-only audit of saved DoE1 trials; writes summaries beside this script.

Run from the repository root with .venv/bin/python. No models are retrained,
production configs changed, or final-test scores used for recommendations.
"""
import ast
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def family(name):
    if name.endswith("_n_f_features"):
        return "sensor_feature_widths"
    # Shuffle conditional layer counts and their dependent dimensions together.
    # Otherwise permutation creates impossible depth/width/missing-value patterns.
    if name.startswith("enc_c"):
        return "cnn_architecture"
    if name.startswith("pp_h_") or name == "pp_n_layers":
        return "pp_architecture"
    if name.startswith("merge_"):
        return "merge_architecture"
    if re.match(r"h_\d+", name) or name == "n_layers":
        return "hidden_architecture"
    return name


def load_studies():
    copies = defaultdict(list)
    roots = ["outputs/DOE1/original", "outputs/DoE1_oldfolder/M1", "outputs/DoE1_oldfolder/M2"]
    for root in roots:
        for path in (ROOT / root).rglob("hpo_trials.csv"):
            if "M1" in path.parts or "M2" in path.parts:
                copies[hashlib.sha256(path.read_bytes()).hexdigest()].append(path)
    manifest, studies = [], []
    for digest, paths in copies.items():
        recent = [p for p in paths if "outputs/DOE1/original/" in str(p)]
        candidates = recent or paths
        path = min(candidates, key=lambda p: (
            not (p.parent / "configuration.json").exists(),
            "run_best" not in p.parts, len(str(p)), str(p)))
        cfg = read_json(path.parent / "configuration.json")
        stage = "M1" if "M1" in path.parts else "M2"
        header = pd.read_csv(path, nrows=0).columns
        columns = [c for c in header if c in ("number", "value", "state") or c.startswith("params_")]
        frame = pd.read_csv(path, usecols=columns)
        frame.columns = [c.removeprefix("params_") for c in frame.columns]
        ss = cfg.get("optuna", {}).get("search_space", {})
        cohort = ("M1_standard_MAE" if stage == "M1" else
                  "M2_MAE_plus_L1" if "l1_lambda" in ss else "M2_MAE_only")
        record = {"study": digest[:12], "stage": stage, "cohort": cohort,
                  "scope": "recent_original" if recent else "older_archive",
                  "canonical_path": str(path.relative_to(ROOT)), "copies": len(paths),
                  "rows": len(frame), "config_available": bool(cfg),
                  "all_paths": json.dumps([str(p.relative_to(ROOT)) for p in sorted(paths)])}
        manifest.append(record)
        if not recent:
            continue
        assert cfg, f"Missing metadata: {path}"
        complete = frame[frame.state.eq("COMPLETE") & np.isfinite(frame.value)].copy()
        complete["study"] = record["study"]
        complete["cohort"] = cohort
        complete["relative_loss"] = complete.value / complete.value.min()
        complete["within_study_rank"] = complete.value.rank(method="average", pct=True)
        # Each study contributes its best ceil(20% * completed) trials.
        complete["top20"] = complete.value.rank(method="first") <= int(np.ceil(.2 * len(complete)))
        studies.append({**record, "frame": frame, "complete": complete, "cfg": cfg, "path": path})
    pd.DataFrame(manifest).to_csv(OUT / "study_manifest.csv", index=False)
    return studies, manifest


def summarize(studies):
    rows, trajectories = [], []
    for s in studies:
        cfg, frame, complete = s["cfg"], s["frame"], s["complete"].sort_values("number")
        startup = cfg["optuna"]["n_startup_trials"]
        first = complete[complete.number < startup]
        best_start = float(first.value.min())
        best = complete.loc[complete.value.idxmin()]
        incumbent, post_records = np.inf, 0
        for row in complete.itertuples():
            if row.value < incumbent:
                post_records += int(row.number >= startup)
                incumbent = row.value
        params = [c for c in frame if c not in ("number", "value", "state")]
        spec = {k: v for k, v in cfg["optuna"]["search_space"].items() if not k.startswith("_")}
        rows.append({k: s[k] for k in ("study", "stage", "cohort", "canonical_path", "copies")}
                    | {"n_trials": len(frame), "complete": len(complete), "pruned": int(frame.state.eq("PRUNED").sum()),
                       "nominal_startup": startup, "best_startup": best_start, "best_final": float(best.value),
                       "best_trial_zero_based": int(best.number), "post_startup_new_records": post_records,
                       "improvement_percent": 100 * (best_start - best.value) / best_start,
                       "max_possible_sampled_params": len(params),
                       "median_sampled_params": float(frame[params].notna().sum(axis=1).median()),
                       "seed": cfg.get("preprocessing", {}).get("random_seed"),
                       "hpo_validation_fraction": cfg["optuna"].get("val_fraction", cfg["training"]["val_fraction"]),
                       "data": cfg["data"]["scalar_csv"], "search_space": json.dumps(spec, sort_keys=True)})
        curve = frame.set_index("number").value.where(frame.set_index("number").state.eq("COMPLETE"))
        curve = curve.reindex(range(int(frame.number.max()) + 1)).cummin().ffill()
        for number, value in curve.items():
            trajectories.append({"study": s["study"], "cohort": s["cohort"], "number": number,
                                 "best_relative_to_startup": value / best_start})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "study_summary.csv", index=False)
    pd.DataFrame(trajectories).to_csv(OUT / "incumbent_trajectories.csv", index=False)
    all_trials = pd.concat([s["complete"] for s in studies], ignore_index=True)
    all_trials.to_csv(OUT / "completed_trials.csv", index=False)
    return summary, all_trials, pd.DataFrame(trajectories)


def parameter_tables(studies, all_trials):
    rows, bins = [], []
    for cohort, frame in all_trials.groupby("cohort"):
        parameters = [c for c in studies[next(i for i, s in enumerate(studies) if s["cohort"] == cohort)]["frame"]
                      if c not in ("number", "value", "state")]
        for name in parameters:
            valid = frame[frame[name].notna()]
            top = valid[valid.top20]
            winners = valid.loc[valid.groupby("study").value.idxmin()]
            correlations = [spearmanr(g[name], g.value).statistic for _, g in valid.groupby("study")
                            if len(g) >= 10 and g[name].nunique() > 1]
            rows.append({"cohort": cohort, "parameter": name, "family": family(name),
                         "n": len(valid), "top20_n": len(top), "studies": valid.study.nunique(),
                         "median_within_study_spearman": float(np.nanmedian(correlations)) if correlations else None,
                         **{f"all_{q}": float(valid[name].quantile(q)) for q in (.1, .5, .9)},
                         **{f"top20_{q}": float(top[name].quantile(q)) for q in (.1, .5, .9)},
                         "winner_min": float(winners[name].min()), "winner_median": float(winners[name].median()),
                         "winner_max": float(winners[name].max())})
            # Quantile bins are descriptive associations, not controlled ablations.
            categorization = (valid[name].astype(str) if valid[name].nunique() <= 5
                              else pd.qcut(valid[name], 4, duplicates="drop").astype(str))
            for value, g in valid.groupby(categorization):
                bins.append({"cohort": cohort, "parameter": name, "bin": value, "n": len(g),
                             "top20_rate": float(g.top20.mean()),
                             "median_within_study_rank": float(g.within_study_rank.median())})
    pd.DataFrame(rows).to_csv(OUT / "parameter_summary.csv", index=False)
    pd.DataFrame(bins).to_csv(OUT / "parameter_bins.csv", index=False)


def predictive_importance(all_trials):
    """Hold entire studies out; permutation preserves study and conditional groups.

    These are observational predictive importances, not causal ablation effects.
    The current M2 cohort has only two studies, so no importance is estimated.
    """
    rows = []
    for cohort, frame in all_trials.groupby("cohort"):
        frame = frame.reset_index(drop=True)
        if frame.study.nunique() < 4:
            continue
        metadata = {"number", "value", "state", "study", "cohort", "relative_loss", "within_study_rank", "top20"}
        columns = [c for c in frame if c not in metadata and frame[c].notna().any()]
        X = frame[columns].copy()
        for name in ("lr", "weight_decay", "l1_lambda"):
            if name in X:
                X[name] = np.log10(X[name])
        X = X.fillna(-1.).to_numpy()
        y = frame.within_study_rank.to_numpy()
        groups = frame.study.to_numpy()
        families = {name: [i for i, c in enumerate(columns) if family(c) == name]
                    for name in sorted(set(map(family, columns)))}
        for fold, (train, test) in enumerate(GroupKFold(min(5, frame.study.nunique())).split(X, y, groups)):
            model = RandomForestRegressor(n_estimators=250, min_samples_leaf=12, max_features=.8,
                                          random_state=71 + fold, n_jobs=1)
            model.fit(X[train], y[train])
            baseline = r2_score(y[test], model.predict(X[test]))
            rng = np.random.default_rng(42 + fold)
            for name, indices in families.items():
                for repeat in range(5):
                    shuffled = X[test].copy()
                    for group in np.unique(groups[test]):
                        where = np.flatnonzero(groups[test] == group)
                        order = rng.permutation(where)
                        shuffled[np.ix_(where, indices)] = shuffled[np.ix_(order, indices)]
                    rows.append({"cohort": cohort, "fold": fold, "family": name, "repeat": repeat,
                                 "heldout_r2": baseline,
                                 "r2_drop_after_permutation": baseline - r2_score(y[test], model.predict(shuffled))})
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "predictive_importance_detail.csv", index=False)
    detail.groupby(["cohort", "family"]).agg(
        mean_r2_drop=("r2_drop_after_permutation", "mean"),
        sd_r2_drop=("r2_drop_after_permutation", "std"),
        mean_heldout_r2=("heldout_r2", "mean")).reset_index().to_csv(OUT / "predictive_importance.csv", index=False)


def training_dynamics(studies):
    rows = []
    for s in studies:
        if s["stage"] != "M1":
            continue
        raw = pd.read_csv(s["path"], usecols=lambda c: c in ("number", "state") or c.endswith("_training"))
        top = set(s["complete"].loc[s["complete"].top20, "number"])
        for trial in raw.to_dict("records"):
            if trial["state"] != "COMPLETE":
                continue
            for key, value in trial.items():
                if not key.endswith("_training") or not isinstance(value, str):
                    continue
                info = ast.literal_eval(value)
                history = info.get("history", [])
                if not history:
                    continue
                best = min(history, key=lambda e: e["val_mae_g"])
                budget = s["cfg"]["optuna"]["epochs"]
                rows.append({"study": s["study"], "number": trial["number"], "fold": key,
                             "top20": trial["number"] in top, "epochs_run": len(history),
                             "best_epoch": best["epoch"], "budget": budget,
                             "best_in_final_quarter": best["epoch"] > .75 * budget,
                             "hit_budget": len(history) == budget,
                             "best_val_mae": best["val_mae_g"],
                             "end_train_mae": history[-1]["train_weight_mae_g"]})
    pd.DataFrame(rows).to_csv(OUT / "m1_training_dynamics.csv", index=False)


def sampler_audit(studies):
    rows = []
    for s in studies:
        cfg = s["cfg"]["optuna"]
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=cfg.get("hyperband_min_resource", 1), max_resource=cfg["n_folds"],
            reduction_factor=cfg.get("hyperband_reduction_factor", 3))
        study_name = cfg.get("study_name", f"{s['stage']}_hpo")
        study = SimpleNamespace(study_name=study_name)
        pruner._try_initialization(study)
        counts = [0] * pruner._n_brackets
        starts = {}
        guided = 0
        incumbent = np.inf
        guided_records = random_records = 0
        best_random = np.inf
        best_phase = None
        for trial in s["frame"].sort_values("number").itertuples():
            bracket = pruner._get_bracket_id(study, SimpleNamespace(number=trial.number))
            model_based = counts[bracket] >= cfg["n_startup_trials"]
            if model_based:
                starts.setdefault(bracket, int(trial.number))
                guided += 1
            if trial.state == "COMPLETE" and np.isfinite(trial.value):
                if not model_based:
                    best_random = min(best_random, trial.value)
                if trial.value < incumbent:
                    incumbent = trial.value
                    best_phase = "TPE" if model_based else "random"
                    guided_records += int(model_based)
                    random_records += int(not model_based)
            if trial.state in ("COMPLETE", "PRUNED"):
                counts[bracket] += 1
        rows.append({"study": s["study"], "stage": s["stage"], "study_name": study_name,
                     "installed_optuna": optuna.__version__, "n_brackets": len(counts),
                     "nominal_startup": cfg["n_startup_trials"],
                     "first_model_based_trial_per_bracket": json.dumps(starts),
                     "model_based_trials_reconstructed": guided, "trials": len(s["frame"]),
                     "tpe_incumbent_records": guided_records, "random_incumbent_records": random_records,
                     "winning_sampling_phase_reconstructed": best_phase,
                     "best_random_completed": best_random, "best_final": incumbent,
                     "gain_over_best_random_percent": 100 * (best_random - incumbent) / best_random,
                     "assumptions": "sequential trials; saved study_name; installed Optuna bracket/filter behavior"})
    pd.DataFrame(rows).to_csv(OUT / "sampler_audit.csv", index=False)


def plot(trajectory, summary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    audit = pd.read_csv(OUT / "sampler_audit.csv")
    for ax, cohort in zip(axes, ("M1_standard_MAE", "M2_MAE_only")):
        subset = trajectory[trajectory.cohort.eq(cohort)]
        for _, g in subset.groupby("study"):
            ax.plot(g.number, g.best_relative_to_startup, alpha=.25, linewidth=1)
        median = subset.groupby("number").best_relative_to_startup.median()
        ax.plot(median.index, median.values, color="black", linewidth=2, label="Median")
        nominal = int(summary.loc[summary.cohort.eq(cohort), "nominal_startup"].iloc[0])
        ax.axvline(nominal, color="#888888", linestyle="--", label="Configured startup count")
        selected = audit[audit.study.isin(subset.study)]
        fully_started = max(max(json.loads(value).values()) for value in selected.first_model_based_trial_per_bracket)
        ax.axvline(fully_started, color="#c55a11", linestyle=":", label="All brackets past startup")
        ax.axhline(1, color="#bbbbbb", linewidth=1)
        ax.set(title=f"{cohort}: {subset.study.nunique()} distinct studies", xlabel="Trial number (zero based)",
               ylabel="Best completed objective / best startup objective", ylim=(0, 1.8))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Configured versus effective startup: saved searches and Optuna 4.4 bracket reconstruction")
    fig.tight_layout()
    fig.savefig(OUT / "search_progress.png", dpi=180)
    plt.close(fig)


def main():
    studies, manifest = load_studies()
    summary, trials, trajectory = summarize(studies)
    parameter_tables(studies, trials)
    predictive_importance(trials)
    training_dynamics(studies)
    sampler_audit(studies)
    plot(trajectory, summary)
    print("Inventory:", len(manifest), "distinct histories; recent studies:", len(studies))
    print(summary.groupby("cohort").agg(studies=("study", "count"), complete=("complete", "sum"),
        improvement_percent=("improvement_percent", "median"), post_startup_records=("post_startup_new_records", "median")).to_string())


if __name__ == "__main__":
    main()
