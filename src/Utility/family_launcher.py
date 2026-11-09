"""Four-family sequential launcher. Pipeline files contain execution and split-seed settings."""
from __future__ import annotations

from datetime import datetime
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

from .hybrid_data import ROOT, read_json, write_json, fingerprint, file_hash, checkpoint_valid, save_manifest
from .hybrid_pipeline import default_config, run_hybrid, training_config
from .pipeline_runtime import stage_directory

FAMILIES = ("original", "gbdt_pppt", "gbdt_ppdirect_pt", "gbdt_engineered")


def validate_pipeline_config(cfg):
    extra = set(cfg) - {"models", "data_split", "_comment"}
    if extra:
        raise ValueError(f"Family pipeline config only accepts models/data_split; move settings {sorted(extra)} to individual model configs")
    split = cfg.get("data_split", {})
    if not isinstance(split, dict) or set(split) - {"seed", "seed_step", "_comment"}:
        raise ValueError("data_split accepts only seed and seed_step")
    for key, value in split.items():
        if key == "_comment":
            continue
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 2**32:
            raise ValueError(f"data_split.{key} must be an integer between 0 and 4294967295")
    for model in cfg["models"].values():
        count = model.get("n_runs", 1)
        if isinstance(count, int) and count > 0 and "seed" in split:
            if split["seed"] + (count - 1) * split.get("seed_step", 0) >= 2**32:
                raise ValueError("data_split seed exceeds 4294967295 for the requested repetitions")
    unknown = set(cfg["models"]) - set(FAMILIES)
    if unknown:
        raise ValueError(f"Unknown pipeline families: {sorted(unknown)}")
    for name, model in cfg["models"].items():
        allowed = {"enabled", "n_runs", "_comment", "reference"}
        if name == "original":
            allowed.update({"n_m1_run_per_pipeline_run", "n_m2_run_per_pipeline_run", "n_fusion_run_per_pipeline_run"})
        if set(model) - allowed:
            raise ValueError(f"{name}: only execution flags/counts belong in the pipeline; allowed: {sorted(allowed)}")
        count = model.get("n_runs", 1)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"{name}.n_runs must be a nonnegative integer counting full repetitions")
        if name == "original":
            for stage in ("m1", "m2", "fusion"):
                count = model.get(f"n_{stage}_run_per_pipeline_run", 1)
                if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                    raise ValueError(f"original.n_{stage}_run_per_pipeline_run must be a nonnegative integer")
            if (model.get("n_m1_run_per_pipeline_run", 1) > 0
                    and model.get("n_m2_run_per_pipeline_run", 1) == 0
                    and model.get("n_fusion_run_per_pipeline_run", 1) > 0):
                raise ValueError("Training M1 before Fusion also requires M2 training for the new teacher")
        reference = model.get("reference", {})
        if reference and (name != "original" or set(reference) - {"enabled", "n_runs"}):
            raise ValueError("reference execution flags/counts are supported only under original")
        if reference:
            ref_count = reference.get("n_runs", 1)
            if not isinstance(ref_count, int) or isinstance(ref_count, bool) or ref_count < 0:
                raise ValueError("reference.n_runs must be a nonnegative integer")


def original_configs(material):
    folder = "DoE1" if material == "DOE1" else "ProBayes"
    suffix = "AllData" if material == "ALL" else "DoE1" if material == "DOE1" else material
    paths = {stage: ROOT / "config" / folder / f"{stage}_{suffix}_config.json" for stage in ("M1", "M2", "Fusion")}
    ref_suffix = "DoE1" if material == "DOE1" else "AllData" if material == "ALL" else material
    paths["RefModels"] = ROOT / "config" / folder / f"RefModels_{ref_suffix}_config.json"
    if material == "ALL":
        paths.pop("Fusion")
    return paths


def _run_original(material, run_dir, seed, optuna_seed, spec, split_seed=None):
    configs = original_configs(material)
    scripts = {"M1": "M1_PPPT_to_W.py", "M2": "M2_PP_to_F.py", "Fusion": "Fusion_M1M2_WPred.py",
               "RefModels": "Reference_Models_W_Pred.py"}
    counts = {stage: spec.get(f"n_{stage.lower()}_run_per_pipeline_run", 1)
              for stage in ("M1", "M2", "Fusion")}
    reused = []
    if counts["M1"] == 0 and (counts["M2"] or counts["Fusion"]):
        reused.append("M1")
    if counts["M2"] == 0 and counts["Fusion"]:
        reused.append("M2")
    if reused:
        m1_cfg = read_json(configs["M1"])
        reference_root = ROOT / m1_cfg["output"].get("pipeline_root", f"outputs/{material}/original") / "best_reference_run"
        # Check all dependencies before creating the new run.
        for stage in reused:
            source = stage_directory(reference_root, stage, material) / "best_overall"
            required = ["best_model.pt", "data_processing.json", "sensors_features_f.csv", "configuration.json"] if stage == "M1" else ["best_model.pt", "scalers.json"]
            missing = [name for name in required if not (source / name).is_file()]
            if missing:
                raise ValueError(f"Cannot reuse reference {stage} from {source}; missing {missing}")
        if "M2" in reused:
            split = read_json(stage_directory(reference_root, "M1", material) / "best_overall/data_processing.json")
            scalers = read_json(stage_directory(reference_root, "M2", material) / "best_overall/scalers.json")
            if scalers.get("m1_run_ts") != split.get("run_ts"):
                raise ValueError("Reference M1/M2 originate from different runs")
        provenance = {"reference_dir": str(reference_root.resolve()), "stages": {},
                      "reference_run": read_json(reference_root / "reference_run.json") if (reference_root / "reference_run.json").exists() else {}}
        for stage in reused:
            source = stage_directory(reference_root, stage, material)
            target = stage_directory(run_dir, stage, material)
            shutil.copytree(source, target)
            # The old execution manifest describes the reference launch, not this one.
            (target / "best_overall/manifest.json").unlink(missing_ok=True)
            provenance["stages"][stage] = {"source": str(source.resolve()),
                "model_sha256": file_hash(source / "best_overall/best_model.pt")}
        write_json(run_dir / "reused_stages.json", provenance)
    stages = [(stage, counts[stage]) for stage in ("M1", "M2", "Fusion") if stage in configs and counts[stage]]
    reference = spec.get("reference", {})
    if reference.get("enabled", True):
        stages.append(("RefModels", int(reference.get("n_runs", 1))))
    for stage, count in stages:
        config_path = configs[stage]
        cfg = read_json(config_path)
        for repetition in range(count):
            stage_seed = seed + repetition
            sampler_seed = optuna_seed + repetition if optuna_seed is not None else None
            env = {**os.environ, "IM_ML_RUN_DIR": str(run_dir), "IM_ML_RUN_SEED": str(stage_seed),
                   "IM_ML_MODEL_CONFIG": str(config_path), "PYTHONDONTWRITEBYTECODE": "1"}
            if stage == "M1":
                env["IM_ML_SPLIT_SEED"] = str(split_seed if split_seed is not None else cfg.get("preprocessing", {}).get("split_seed", seed))
            cmd = [sys.executable, str(ROOT / "src" / scripts[stage]), "--material", material]
            if stage == "Fusion":
                cmd.extend(["--seed", str(stage_seed)])
            elif sampler_seed is not None:
                cmd.extend(["--optuna-seed", str(sampler_seed)])
            # Reuse only a manifest made with this exact requested configuration and seed.
            target = stage_directory(run_dir, stage, material) / "best_overall"
            data_paths = [cfg.get("data", {}).get("scalar_csv")]
            data_paths += [source["path"] for source in cfg.get("data", {}).get("timeseries_csvs", [])]
            if cfg.get("data", {}).get("pressure_csv"):
                data_paths.append(cfg["data"]["pressure_csv"])
            data_hashes = [(path, file_hash(ROOT / path)) for path in data_paths if path]
            signature = fingerprint(training_config(cfg), cfg.get("evaluation", {}), stage_seed, sampler_seed, data_hashes, env.get("IM_ML_SPLIT_SEED"))
            if stage == "M2":
                teacher_path = stage_directory(run_dir, "M1", material) / "best_overall" / "best_model.pt"
                signature = fingerprint(signature, file_hash(teacher_path))
            files = ["best_model.pt", "data_processing.json"] if stage == "M1" else ["best_model.pt", "scalers.json"]
            if (stage in ("M1", "M2") and cfg.get("checkpoint", {}).get("reuse", False)
                    and checkpoint_valid(target, signature, files)):
                print(f"  [original] Reusing {stage}", flush=True)
                continue
            print(f"  [original] {stage} {repetition+1}/{count}; seed={stage_seed}; {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)
            if stage in ("M1", "M2"):
                save_manifest(target, signature, files)
    _original_comparison(run_dir, material, seed)
    if counts["Fusion"] > 0 and "Fusion" in configs:
        _publish_best_reference_run(material, run_dir, configs)
    else:
        print("  [original] Results saved separately; best_reference_run is updated only after a requested Fusion run.", flush=True)


def _publish_best_reference_run(material, run_dir, configs):
    """Snapshot a complete, coherent original repetition ranked by Fusion test MAE."""
    if "Fusion" not in configs:
        # ALL has no Fusion and therefore cannot be ranked by this criterion.
        return
    candidates = []
    for path in (run_dir / "Fusion").rglob("*_metrics.json"):
        if path.name not in ("test_metrics.json", "test_1_metrics.json"):
            continue
        if path.parent.name.startswith("alpha_") and (path.parent.parent / "best_fusion").exists():
            continue
        report = read_json(path)
        score = report.get("MAE")
        if ((path.parent.name == "best_fusion" or path.parent.name.startswith("alpha_"))
                and report.get("source", "final_test") in ("final_test", "final_test_1")
                and isinstance(score, (int, float)) and math.isfinite(score)):
            candidates.append((float(score), path))
    if not candidates:
        print("  [original] No Fusion test score; best_reference_run was not updated. "
              "Enable evaluation.enabled in the Fusion config to rank this run.", flush=True)
        return
    score, fusion_metrics_path = min(candidates, key=lambda item: (item[0], str(item[1])))
    selection_path = fusion_metrics_path.parent / "selection.json"
    alpha_selection = read_json(selection_path) if selection_path.exists() else {}
    sources = {stage: stage_directory(run_dir, stage, material) / "best_overall" for stage in ("M1", "M2")}
    requested = {stage: read_json(configs[stage]) for stage in sources}
    output_cfg = requested["M1"]["output"]
    output_root = ROOT / output_cfg.get("pipeline_root", f"outputs/{material}/original")
    destination = output_root / "best_reference_run"
    split = read_json(sources["M1"] / "data_processing.json")
    scalers = read_json(sources["M2"] / "scalers.json")
    if scalers.get("m1_run_ts") != split.get("run_ts"):
        raise ValueError("Cannot publish original checkpoints: M1/M2 originate from different runs")
    saved_configuration = sources["M1"] / "configuration.json"
    data_cfg = (read_json(saved_configuration).get("data", requested["M1"]["data"])
                if saved_configuration.is_file() else requested["M1"]["data"])
    data_paths = [data_cfg["scalar_csv"]]
    data_paths += [source["path"] for source in data_cfg.get("timeseries_csvs", data_cfg.get("pressure_csvs", []))]
    if data_cfg.get("pressure_csv"):
        data_paths.append(data_cfg["pressure_csv"])
    # Compare runs on the same task and exact test membership, even when training
    # seeds, validation membership or model hyperparameters differ.
    signature = fingerprint(data_cfg,
        [(path, file_hash(ROOT / path)) for path in data_paths],
        {key: split.get(key) for key in ("pp_cols", "test_part_ids", "group_split", "edge_group_split")})
    manifest_path = destination / "reference_run.json"
    previous = read_json(manifest_path) if manifest_path.exists() else {}
    complete = all((stage_directory(destination, stage, material) / "best_overall" / "best_model.pt").exists()
                   for stage in sources)
    if (complete and previous.get("signature") == signature
            and score >= previous.get("fusion_test_MAE", float("inf"))):
        return
    publication = {"signature": signature, "fusion_test_MAE": score, "run_dir": str(run_dir.resolve()),
        "run_timestamp": run_dir.parent.name if run_dir.name.startswith("run_") else run_dir.name,
        "repetition": run_dir.name, "m1_run_ts": split.get("run_ts"),
        "selection_metric": "Fusion test_1 MAE" if fusion_metrics_path.name.startswith("test_1") else "Fusion test MAE",
        "selected_alpha": alpha_selection.get("source_folder", fusion_metrics_path.parent.name),
        "selected_fusion_candidate": alpha_selection.get("candidate", alpha_selection.get("source_folder")),
        "fusion_fine_tuned": alpha_selection.get("fine_tuned"),
        "alpha_selection_metric": alpha_selection.get("selection_metric"),
        "fusion_metrics_path": str(fusion_metrics_path.relative_to(run_dir)),
        "published_at": datetime.now().astimezone().isoformat()}
    output_root.mkdir(parents=True, exist_ok=True)
    # Replace the entire snapshot, so no models or reference artifacts from the
    # previous winner can survive a promotion.
    with tempfile.TemporaryDirectory(prefix=".reference-", dir=output_root) as temporary:
        snapshot, backup = Path(temporary) / "snapshot", Path(temporary) / "previous"
        shutil.copytree(run_dir, snapshot)
        write_json(snapshot / "reference_run.json", publication)
        for stage in sources:
            write_json(stage_directory(snapshot, stage, material) / "best_overall" / "pipeline_source.json", publication)
        if destination.exists():
            destination.rename(backup)
        try:
            snapshot.rename(destination)
        except Exception:
            if backup.exists():
                backup.rename(destination)
            raise
    print(f"  [original] Best reference run: {destination}; Fusion test MAE={score:.6g}; "
          f"source={run_dir}", flush=True)


def _original_comparison(run_dir, material, seed):
    rows = []
    for stage in ("M1", "M2"):
        path = stage_directory(run_dir, stage, material) / "best_overall" / "best_metrics.json"
        if path.exists():
            report = read_json(path)
            rows.append({"family": "original", "material": material, "seed": seed,
                "run_dir": str(run_dir), "role": "M1 teacher (measured signals)" if stage == "M1" else "M2 feature reconstruction",
                "partition": report.get("source", "").replace("final_", "") if report.get("source", "").startswith("final_test") else "validation",
                **{key: value for key, value in report.items() if isinstance(value, (int, float))}})
            external_path = path.parent / "test_2_metrics.json"
            if external_path.exists():
                report_2 = read_json(external_path)
                rows.append({"family": "original", "material": material, "seed": seed,
                    "run_dir": str(run_dir), "role": "M1 teacher (measured signals)" if stage == "M1" else "M2 feature reconstruction",
                    "partition": "test_2", **{k: v for k, v in report_2.items() if isinstance(v, (int, float))}})
    for path in (run_dir / "RefModels").rglob("best_metrics.json") if (run_dir / "RefModels").exists() else []:
        if path.parent.name != "best_overall":
            continue
        for partition, report_path in ((None, path), ("test_2", path.parent/"test_2_metrics.json")):
            if not report_path.exists():
                continue
            report = read_json(report_path)
            partition = partition or (report.get("source", "").replace("final_", "")
                                      if report.get("source", "").startswith("final_test") else "validation")
            rows.append({"family": "original", "material": material, "seed": seed,
                "run_dir": str(run_dir), "role": f"Reference/{path.parent.parent.name}", "partition": partition,
                **{k: v for k, v in report.items() if isinstance(v, (int, float))}})
    for path in (run_dir / "Fusion").rglob("*_metrics.json") if (run_dir / "Fusion").exists() else []:
        if path.parent.name == "m2_reconstruction":
            continue
        if path.name not in ("test_metrics.json", "test_1_metrics.json", "test_2_metrics.json", "validation_metrics.json"):
            continue
        if (path.parent.name == "best_fusion" and path.name == "validation_metrics.json"
                and any(path.parent.parent.glob("alpha_*/validation_metrics.json"))):
            continue
        report = read_json(path)
        selection_path = path.parent.parent / "best_alpha_selection.json"
        selection = read_json(selection_path) if selection_path.exists() else {}
        rows.append({"family": "original", "material": material, "seed": seed,
            "run_dir": str(run_dir), "role": "Fusion (PP only)", "alpha": report.get("alpha", path.parent.name),
            "candidate": report.get("candidate", path.parent.name),
            "selected": path.parent.name == "best_fusion" or (path.name == "validation_metrics.json"
                and path.parent.name == selection.get("source_folder")),
            "partition": path.name.removesuffix("_metrics.json"),
            **{key: value for key, value in report.items() if isinstance(value, (int, float))}})
    if rows:
        pd.DataFrame(rows).to_csv(run_dir / "comparison.csv", index=False)


def run_families(cfg, materials, optuna_seed=None, dry_run=False):
    validate_pipeline_config(cfg)
    # Catch missing physical metadata before any potentially expensive earlier family.
    engineered_spec = cfg["models"].get("gbdt_engineered", {})
    if not dry_run and engineered_spec.get("enabled", False) and engineered_spec.get("n_runs", 1) > 0:
        from .engineered_features import time_axis, signal_time_options
        for material in materials:
            engineered_cfg = read_json(default_config("gbdt_engineered", material))
            options = engineered_cfg.get("feature_extraction", {})
            for source in engineered_cfg.get("data", {}).get("timeseries_csvs", []):
                settings = signal_time_options(source, options, base_dir=ROOT)
                columns = [name for name in pd.read_csv(ROOT / source["path"], nrows=0).columns
                           if name != engineered_cfg["data"]["id_col"]]
                try:
                    time_axis(len(columns), settings, columns)
                except ValueError as exc:
                    raise ValueError(f"gbdt_engineered/{source['name']}: {exc} "
                                     "Provide the dataset info/signal_time_grid.csv or configure explicit timing "
                                     "in this model's feature_extraction settings.") from exc
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    all_rows, failures, comparison_roots = [], [], set()
    for material in materials:
        for family in FAMILIES:
            spec = cfg["models"].get(family, {"enabled": False})
            if not spec.get("enabled", False) or spec.get("n_runs", 1) == 0:
                continue
            if family == "original":
                paths = original_configs(material)
                model_cfg = read_json(paths["M1"])
                output = ROOT / model_cfg["output"].get("pipeline_root", f"outputs/{material}/original")
            else:
                path = default_config(family, material)
                if not path.exists():
                    raise ValueError(f"Missing model-specific configuration: {path}")
                model_cfg = read_json(path)
                output = ROOT / model_cfg["output"]["output_dir"]
            for repetition in range(spec.get("n_runs", 1)):
                seed = int(model_cfg["preprocessing"]["random_seed"]) + repetition * 1009
                sampler_seed = optuna_seed + repetition * 1009 if optuna_seed is not None else None
                split_options = cfg.get("data_split", {})
                base_split_seed = split_options.get("seed", model_cfg["preprocessing"].get(
                    "split_seed", seed))
                split_seed = int(base_split_seed) + repetition * split_options.get("seed_step", 0)
                if not 0 <= split_seed < 2**32:
                    raise ValueError("Resolved data split seed must be between 0 and 4294967295")
                run_dir = output / "runs" / timestamp / f"run_{repetition+1:03d}"
                print(f"[{material}/{family}] complete repetition {repetition+1}/{spec.get('n_runs', 1)}; "
                      f"seed={seed}; split_seed={split_seed}; outputs={run_dir}", flush=True)
                if family == "original":
                    for stage in ("M1", "M2", "Fusion"):
                        count = spec.get(f"n_{stage.lower()}_run_per_pipeline_run", 1)
                        print(f"  {stage} launches per pipeline run: {count}", flush=True)
                    if spec.get("n_m1_run_per_pipeline_run", 1) == 0:
                        print("  Reuse M1 and its saved split from best_reference_run; split_seed does not resplit it", flush=True)
                    if spec.get("n_m2_run_per_pipeline_run", 1) == 0 and spec.get("n_fusion_run_per_pipeline_run", 1) > 0:
                        print("  Reuse M2 from best_reference_run", flush=True)
                    if spec.get("n_m2_run_per_pipeline_run", 1) > 0 and spec.get("n_fusion_run_per_pipeline_run", 1) == 0:
                        print("  M2-only results stay separate; best_reference_run is unchanged", flush=True)
                if family != "original" and model_cfg.get("split", {}).get("source_dir"):
                    print("  Reuse the configured source split; split_seed applies only when generating a new split", flush=True)
                if dry_run:
                    continue
                try:
                    if family == "original":
                        _run_original(material, run_dir, seed, sampler_seed, spec, split_seed=split_seed)
                    else:
                        run_hybrid(family, path, material, run_dir, seed, sampler_seed, split_seed=split_seed)
                    comparison_path = run_dir / "comparison.csv"
                    if comparison_path.exists():
                        all_rows.append(pd.read_csv(comparison_path))
                        comparison_roots.add(output.parent)
                        combined = pd.concat(all_rows, ignore_index=True)
                        for root in comparison_roots:
                            root.mkdir(parents=True, exist_ok=True)
                            combined.to_csv(root / "comparison.csv", index=False)
                            combined.to_csv(root / f"comparison_{timestamp}.csv", index=False)
                except Exception as exc:
                    failures.append({"family": family, "material": material, "run": repetition+1,
                                     "error": str(exc)})
                    write_json(run_dir / "failure.json", failures[-1])
                    print(f"FAILED: {failures[-1]}", file=sys.stderr, flush=True)
    return failures
