"""Prediction-based SHAP for frozen tree Fusion; no autograd through trees."""
from pathlib import Path

import numpy as np
import pandas as pd

from .hybrid_data import write_json, impute


def run_xai(predictor, training_pp, explain_pp, part_ids, directory, cfg, material, seed):
    import shap
    from src import Fusion_M1M2_WPred as plots
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    background_count = min(int(cfg.get("background_samples", 30)), len(training_pp))
    count = min(int(cfg.get("explain_samples", 40)), len(explain_pp))
    if min(background_count, count) <= 0:
        raise ValueError("XAI sample counts must be positive")
    bg = training_pp[rng.choice(len(training_pp), background_count, replace=False)]
    selected = rng.choice(len(explain_pp), count, replace=False)
    x, ids = explain_pp[selected], np.asarray(part_ids)[selected]
    # SHAP maskers need finite background data. Imputation statistics are training-only.
    pp_imp = predictor.m2_preprocessing["pp_imputer"]
    bg, x = impute(pp_imp, bg), impute(pp_imp, x)
    nsamples = cfg.get("nsamples", 256)
    np.random.seed(seed)
    tied = shap.KernelExplainer(predictor.predict, bg)
    tied_values = np.asarray(tied.shap_values(x, nsamples=nsamples, l1_reg=cfg.get("l1_reg", 0)))
    summary = plots.build_single_pp_summary(predictor.pp_names, tied_values, None)
    summary.to_csv(directory / "feature_attributions_single_pp.csv")
    pd.DataFrame(tied_values, index=ids, columns=predictor.pp_names).to_csv(
        directory / "shap_single_pp_values.csv", index_label="part_id")
    plots.save_single_pp_shap_beeswarm(tied_values, x, predictor.pp_names, directory, material)
    plots.save_single_pp_importance_plot(summary, predictor.pp_names, directory, material)
    if cfg.get("pathway_analysis", True):
        n = len(predictor.pp_names)
        def pathways(flat):
            return predictor.predict_paths(flat[:, :n], flat[:, n:])
        separated = shap.KernelExplainer(pathways, np.concatenate([bg, bg], axis=1))
        values = np.asarray(separated.shap_values(np.concatenate([x, x], axis=1),
                            nsamples=nsamples, l1_reg=cfg.get("l1_reg", 0)))
        split_summary = plots.build_summary(predictor.pp_names, values, None, None, n)
        split_summary.to_csv(directory / "feature_attributions.csv")
        columns = [f"indirect__{name}" for name in predictor.pp_names] + [f"direct__{name}" for name in predictor.pp_names]
        pd.DataFrame(values, index=ids, columns=columns).to_csv(directory / "shap_values.csv", index_label="part_id")
        plots._bar_direct_indirect(np.abs(values[:, :n]).mean(axis=0), np.abs(values[:, n:]).mean(axis=0),
            predictor.pp_names, f"[{material}] SHAP — direct and reconstructed-sensor pathways",
            directory / "xai_bar_shap_pathway_split.png",
            direct_label="Direct (PP embedding)" if predictor.family == "gbdt_pppt" else "Direct (raw PP)")
        plots.save_shap_beeswarm(values, x, predictor.pp_names, directory, material, n)
        plots.save_total_effect_plot(split_summary, predictor.pp_names, directory, material)
        plots.save_net_effect_bars(split_summary, predictor.pp_names, directory, material)
        plots.save_pathway_agreement_scatter(split_summary, predictor.pp_names, directory, material)
        plots.save_cancellation_bars(split_summary, predictor.pp_names, directory, material)
    # Exact tree contribution values for the actual Fusion head inputs (not input-PP SHAP).
    features = np.concatenate([predictor.direct_features(x), predictor.predict_features(x)], axis=1)
    head_x = impute(predictor.head_imputer, features)
    import xgboost as xgb
    contributions = predictor.xgb.get_booster().predict(xgb.DMatrix(head_x), pred_contribs=True,
        iteration_range=(0, predictor.xgb.best_iteration + 1) if hasattr(predictor.xgb, "best_iteration") else (0, 0))
    head_names = ([f"pp_embedding_{i}" for i in range(features.shape[1] - len(predictor.sensor_names))]
                  if predictor.family == "gbdt_pppt" else predictor.pp_names) + predictor.sensor_names
    pd.DataFrame(contributions, index=ids, columns=head_names + ["bias"]).to_csv(
        directory / "tree_head_shap_values.csv", index_label="part_id")
    write_json(directory / "xai_method.json", {"method": "KernelSHAP on PP-only Fusion predictions",
        "background_partition": "training", "explanation_partition": cfg.get("partition", "validation"),
        "background_samples": background_count, "explain_samples": count, "nsamples": nsamples,
        "seed": seed, "gradients_through_xgboost": False,
        "pathway_interpretation": "Duplicated PP inputs separate direct/indirect intervention paths; "
            "tied-input SHAP is the primary total PP effect. Correlated-feature caveats apply."})
