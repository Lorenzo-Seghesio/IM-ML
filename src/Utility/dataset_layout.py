"""Resolve legacy flat data and independent Train/Test dataset folders."""
import copy
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def dataset_root(data, base_dir=ROOT):
    if data.get("dataset_dir"):
        return Path(base_dir) / data["dataset_dir"]
    if not data.get("scalar_csv"):
        return None
    parent = (Path(base_dir) / data["scalar_csv"]).parent
    return parent.parent if parent.name in ("Train", "Test") else parent


def divided_dataset(data, base_dir=ROOT):
    """Training CSVs live in Train/, independently of optional test_2 settings."""
    root = dataset_root(data, base_dir)
    return bool(root and (root / "Train").is_dir())


def _matching_csv(directory, suffix):
    matches = sorted(p for p in directory.glob("*.csv") if p.name.lower().endswith(suffix.lower()))
    if len(matches) != 1:
        raise ValueError(f"Expected one *{suffix} file in {directory}; found {len(matches)}")
    return str(matches[0])


def resolve_data_config(data, partition=None, base_dir=ROOT):
    result = copy.deepcopy(data)
    partition = partition or data.get("_dataset_partition", "train")
    root = dataset_root(data, base_dir)
    structured = divided_dataset(data, base_dir)
    if partition == "test_2":
        configured = data.get("test_2_dir")
        if not configured:
            return None
        if not isinstance(configured, (str, Path)):
            raise ValueError("evaluation.test_2_dir must be a folder path or null")
        directory = Path(base_dir) / configured
        if not directory.is_dir():
            raise FileNotFoundError(f"Configured evaluation.test_2_dir does not exist: {directory}")
    elif not structured and not data.get("dataset_dir"):
        return result
    else:
        directory = root / "Train" if structured else root
    if structured or partition == "test_2":
        result["_dataset_partition"] = partition
    result["scalar_csv"] = _matching_csv(directory, "Scalars.csv")
    sources = result.get("timeseries_csvs", result.get("pressure_csvs", []))
    for source in sources:
        path = Path(source["path"])
        suffixes = ("Pressure_1.csv", "Pressure_2.csv", "Temperature.csv", "Pressure.csv")
        suffix = next((s for s in suffixes if path.name.lower().endswith(s.lower())), None)
        if suffix is None:
            raise ValueError(f"Cannot identify signal CSV type: {path}")
        source["path"] = _matching_csv(directory, suffix)
    if result.get("pressure_csv"):
        suffix = "Pressure_1.csv" if list(directory.glob("*Pressure_1.csv")) else "Pressure.csv"
        result["pressure_csv"] = _matching_csv(directory, suffix)
    return result


def training_data_identity(data, base_dir=ROOT):
    """Hash training files only; changing external evaluation data never refits models."""
    data = resolve_data_config(data, partition="train", base_dir=base_dir)
    paths = [data["scalar_csv"]]
    paths += [s["path"] for s in data.get("timeseries_csvs", data.get("pressure_csvs", []))]
    if data.get("pressure_csv"):
        paths.append(data["pressure_csv"])
    digest = hashlib.sha256()
    for name in paths:
        path = (Path(base_dir) / name).resolve()
        digest.update(str(path).encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def internal_test_name(data, base_dir=ROOT):
    return "test_1" if divided_dataset(data, base_dir) else "test"


def label_test_1(directory, data):
    """Rename legacy test writers for paired datasets, retaining split contract keys."""
    if not divided_dataset(data):
        return
    for path in list(Path(directory).rglob("*")):
        if not path.is_file() or path.name in ("train_test_split.json", "data_processing.json"):
            continue
        if path.name.endswith("_test_points.csv") and not path.name.startswith("test_"):
            continue
        name = re.sub(r"(?<![A-Za-z0-9])test(?!_[12]|_points)(?=[_.]|$)", "test_1", path.name)
        if name != path.name:
            destination = path.with_name(name)
            if destination.exists():
                path.unlink()
            else:
                path.rename(destination)
    for path in Path(directory).rglob("*metrics.json"):
        report = json.loads(path.read_text())
        changed = False
        for key, before, after in (("source", "final_test", "final_test_1"), ("partition", "test", "test_1")):
            if report.get(key) == before:
                report[key] = after
                changed = True
        if changed:
            path.write_text(json.dumps(report, indent=2))
