"""Run-local paths and seeds for unchanged baseline stage ordering.

Only orchestration metadata travels through the environment. Model settings are
read from each stage's own configuration file.
"""
import copy
import json
import os
from pathlib import Path
from .dataset_layout import resolve_data_config


def stage_directory(run_dir, stage, material):
    """DoE1 is already identified by its output root; keep legacy material folders elsewhere."""
    directory = Path(run_dir) / stage
    return directory if material.upper() == "DOE1" else directory / material


def model_config(default_path, stage, material):
    path = Path(os.environ.get("IM_ML_MODEL_CONFIG", default_path))
    cfg = json.loads(path.read_text())
    return configure(cfg, stage, material), path


def configure(cfg, stage, material):
    cfg = copy.deepcopy(cfg)
    if cfg.get("data"):
        cfg["data"] = resolve_data_config(cfg["data"])
    seed = os.environ.get("IM_ML_RUN_SEED")
    if seed is not None:
        cfg.setdefault("preprocessing", {})["random_seed"] = int(seed)
    split_seed = os.environ.get("IM_ML_SPLIT_SEED")
    if stage == "M1" and split_seed is not None:
        cfg.setdefault("preprocessing", {}).setdefault("split_seed", int(split_seed))
    run_dir = os.environ.get("IM_ML_RUN_DIR")
    if run_dir:
        root = Path(run_dir)
        if stage in ("M1", "M2"):
            base = stage_directory(root, stage, material)
            cfg["output"] = {**cfg.get("output", {}), "output_dir": str(base),
                "run_best_dir": str(base / "run_best"), "best_overall_dir": str(base / "best_overall")}
        elif stage == "Fusion":
            cfg.setdefault("output", {})["output_base_dir"] = str(root / "Fusion")
            cfg["models"] = {"m1_best_dir": str(stage_directory(root, "M1", material) / "best_overall"),
                             "m2_best_dir": str(stage_directory(root, "M2", material) / "best_overall")}
        elif stage == "RefModels":
            cfg["output_base_dir"] = str(root / "RefModels")
            cfg["m1_split_dir"] = str(stage_directory(root, "M1", material) / "best_overall")
        if stage == "M2":
            cfg["data"]["features_csv"] = str(stage_directory(root, "M1", material) / "best_overall" / "sensors_features_f.csv")
    return cfg


def finalize_evaluation(directory, test_enabled):
    """Legacy plot writers have fixed test filenames; label validation output honestly."""
    if test_enabled:
        return
    directory = Path(directory)
    for path in list(directory.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        # Split contracts keep their established filenames and independent test IDs.
        if name == "train_test_split.json" or name.endswith("_test_points.csv") and not name.startswith("test_"):
            continue
        new_name = name.replace("final_test", "validation")
        if new_name.startswith("test_"):
            new_name = "validation_" + new_name[5:]
        else:
            new_name = new_name.replace("_test_", "_validation_", 1)
        if name.endswith("_test.png"):
            new_name = new_name.replace("_test.png", "_validation.png")
        if new_name != name:
            destination = path.with_name(new_name)
            if destination.exists():
                path.unlink()
            else:
                path.rename(destination)
