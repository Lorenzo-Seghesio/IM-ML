"""Recreate Reference-MLP SHAP plots with configurable x-axis limits.

The utility loads the saved MLP checkpoint from a reference-model output
folder, reconstructs the same development/test split and MinMax scaling used by
Reference_Models_W_Pred.py, recomputes GradientExplainer SHAP values, and saves
new plots without overwriting the original XAI images.

Examples
--------
    python src/Utility/replot_reference_mlp_shap.py \
        --material DOE1 \
        --bar-xlim 0 0.6 \
        --beeswarm-xlim -0.18 0.15

    python src/Utility/replot_reference_mlp_shap.py \
        --material PP \
        --model-dir outputs/ProBayes/RefModels/MLP/PP/best_overall \
        --bar-xlim 0 1.0 \
        --beeswarm-xlim -0.25 0.25

Use ``--auto-limits`` to let SHAP/matplotlib choose both axis ranges.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.Reference_Models_W_Pred import (  # noqa: E402
    MLPModel,
    _CFG_MAP,
    _MAT_DIR,
    _initial_split,
    _inner_split,
    load_data,
)


DEFAULT_BAR_XLIM = (0.0, 0.06)
DEFAULT_BEESWARM_XLIM = (-0.18, 0.15)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replot saved Reference-MLP SHAP results with custom x-axis limits."
    )
    parser.add_argument(
        "--material",
        type=str.upper,
        choices=sorted(_CFG_MAP),
        default="DOE1",
        help="Dataset/material to load (default: DOE1).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="MLP best_overall directory. Defaults to the configured output path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the new images. Defaults to --model-dir.",
    )
    parser.add_argument(
        "--suffix",
        default="_custom",
        help="Suffix before .png for new images (default: _custom).",
    )
    parser.add_argument(
        "--bar-xlim",
        nargs=2,
        type=float,
        metavar=("XMIN", "XMAX"),
        default=DEFAULT_BAR_XLIM,
        help="Bar-plot x-axis limits (default: 0 0.6).",
    )
    parser.add_argument(
        "--beeswarm-xlim",
        nargs=2,
        type=float,
        metavar=("XMIN", "XMAX"),
        default=DEFAULT_BEESWARM_XLIM,
        help="Beeswarm x-axis limits (default: -0.18 0.15).",
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=100,
        help="Maximum development samples used as SHAP background (default: 100).",
    )
    parser.add_argument(
        "--auto-limits",
        action="store_true",
        help="Ignore the configured limits and use automatic axis limits.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="Torch device. Defaults to CUDA when available, otherwise CPU.",
    )
    return parser.parse_args()


def _resolve_model_dir(material: str, cfg: dict, requested: Path | None) -> Path:
    if requested is not None:
        model_dir = requested
        if not model_dir.is_absolute():
            model_dir = BASE_DIR / model_dir
        return model_dir.resolve()

    output_base = BASE_DIR / cfg.get("output_base_dir", "outputs/ProBayes/RefModels")
    return (output_base / "MLP" / _MAT_DIR[material] / "best_overall").resolve()


def _indices_from_ids(ids: list, part_ids: np.ndarray) -> np.ndarray:
    id_to_idx = {part_id: i for i, part_id in enumerate(part_ids.tolist())}
    return np.array(
        [id_to_idx[part_id] for part_id in ids if part_id in id_to_idx],
        dtype=np.intp,
    )


def _load_split(
    material: str,
    cfg: dict,
    part_ids: np.ndarray,
    strat_labels,
    group_ids,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Reconstruct the split and final-training scaler fit indices."""
    m1_train_idx = m1_val_idx = None
    m1_split_dir_cfg = cfg.get("m1_split_dir") if material == "DOE1" else None
    dp_path = BASE_DIR / m1_split_dir_cfg / "data_processing.json" if m1_split_dir_cfg else None

    if dp_path is not None and dp_path.exists():
        dp = json.loads(dp_path.read_text())
        train_idx = _indices_from_ids(dp["train_part_ids"], part_ids)
        val_idx = _indices_from_ids(dp["val_part_ids"], part_ids)
        test_idx = _indices_from_ids(dp["test_part_ids"], part_ids)
        dev_idx = np.concatenate([train_idx, val_idx])
        m1_train_idx, m1_val_idx = train_idx, val_idx
        print(
            f"Using M1 split from {dp_path}  "
            f"→ train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
        )
        return dev_idx, test_idx, m1_train_idx, m1_val_idx

    dev_idx, test_idx = _initial_split(
        len(part_ids),
        cfg["preprocessing"].get("test_fraction", 0.2),
        split_seed,
        strat_labels,
        group_ids,
        cfg["data"].get("group_split", {}),
    )
    print(f"Reconstructed split  → dev={len(dev_idx)} test={len(test_idx)}")
    return dev_idx, test_idx, m1_train_idx, m1_val_idx


def _load_model(model_dir: Path, metrics: dict, device: torch.device) -> MLPModel:
    model_path = model_dir / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"MLP checkpoint not found: {model_path}")

    n_in = int(metrics["n_in"])
    hidden_dims = [int(value) for value in metrics["hidden_dims"]]
    dropout = float(metrics["dropout"])
    model = MLPModel(n_in, hidden_dims, dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def _compute_shap(
    model: MLPModel,
    X_dev_sc: np.ndarray,
    X_test_sc: np.ndarray,
    background_size: int,
    device: torch.device,
) -> np.ndarray:
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "The shap package is required. Install the project requirements first."
        ) from exc

    if background_size <= 0:
        raise ValueError("--background-size must be greater than zero.")
    if len(X_dev_sc) == 0 or len(X_test_sc) == 0:
        raise ValueError("The reconstructed development/test split is empty.")

    rng = np.random.default_rng(42)
    bg_idx = rng.choice(
        len(X_dev_sc), min(background_size, len(X_dev_sc)), replace=False
    )
    bg_t = torch.tensor(X_dev_sc[bg_idx], dtype=torch.float32, device=device)
    test_t = torch.tensor(X_test_sc, dtype=torch.float32, device=device)

    class _Wrap2D(torch.nn.Module):
        def __init__(self, wrapped_model: torch.nn.Module):
            super().__init__()
            self.wrapped_model = wrapped_model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.wrapped_model(x).unsqueeze(-1)

    explainer = shap.GradientExplainer(_Wrap2D(model), bg_t)
    raw = explainer.shap_values(test_t)
    shap_values = np.asarray(raw[0] if isinstance(raw, list) else raw)
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]
    if shap_values.shape != X_test_sc.shape:
        raise ValueError(
            f"Unexpected SHAP shape {shap_values.shape}; expected {X_test_sc.shape}."
        )
    return shap_values.astype(np.float32, copy=False)


def _validate_limits(limits: tuple[float, float] | None, name: str) -> None:
    if limits is None:
        return
    if limits[0] >= limits[1]:
        raise ValueError(f"{name} must satisfy xmin < xmax; received {limits}.")


def _save_plots(
    shap_values: np.ndarray,
    X_test_sc: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    suffix: str,
    bar_xlim: tuple[float, float] | None,
    beeswarm_xlim: tuple[float, float] | None,
) -> tuple[Path, Path]:
    import shap

    output_dir.mkdir(parents=True, exist_ok=True)
    bar_path = output_dir / f"xai_shap_bar{suffix}.png"
    beeswarm_path = output_dir / f"xai_shap_beeswarm{suffix}.png"

    with plt.rc_context(plt.rcParamsDefault):
        shap.summary_plot(
            shap_values,
            X_test_sc,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            color="#2166AC",
        )
        bar_ax = plt.gca()
        if bar_xlim is not None:
            bar_ax.set_xlim(*bar_xlim)
        bar_ax.set_xlabel("mean(|SHAP value|)")
        bar_ax.xaxis.grid(True, color="#d0d0d0", linewidth=0.8, zorder=0)
        bar_ax.set_axisbelow(True)
        plt.savefig(bar_path, bbox_inches="tight", dpi=300)
        plt.close()

        shap.summary_plot(
            shap_values,
            X_test_sc,
            feature_names=feature_names,
            plot_type="dot",
            show=False,
        )
        beeswarm_ax = plt.gca()
        if beeswarm_xlim is not None:
            beeswarm_ax.set_xlim(*beeswarm_xlim)
        beeswarm_ax.xaxis.grid(True, color="#d0d0d0", linewidth=0.8, zorder=0)
        beeswarm_ax.set_axisbelow(True)
        plt.savefig(beeswarm_path, bbox_inches="tight", dpi=300)
        plt.close()

    return bar_path, beeswarm_path


def main() -> None:
    args = _parse_args()
    if args.background_size <= 0:
        raise ValueError("--background-size must be greater than zero.")

    material = args.material
    cfg_path = BASE_DIR / "config" / _CFG_MAP[material]
    cfg = json.loads(cfg_path.read_text())
    model_dir = _resolve_model_dir(material, cfg, args.model_dir)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else model_dir
    )
    if not output_dir.is_absolute():
        output_dir = BASE_DIR / output_dir
    output_dir = output_dir.resolve()

    metrics_path = model_dir / "best_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    metrics = json.loads(metrics_path.read_text())
    if "hidden_dims" not in metrics or "n_in" not in metrics:
        raise ValueError(f"{metrics_path} does not contain MLP architecture metadata.")

    run_info_path = model_dir / "run_info.json"
    run_info = json.loads(run_info_path.read_text()) if run_info_path.exists() else {}
    split_seed = int(run_info.get("split_seed", cfg["preprocessing"]["random_seed"]))

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(device_name)

    part_ids, X_pp, _X_pt, _y, strat_labels, pp_cols, group_ids = load_data(
        cfg["data"], cfg["preprocessing"], material
    )
    dev_idx, test_idx, fixed_train_idx, fixed_val_idx = _load_split(
        material, cfg, part_ids, strat_labels, group_ids, split_seed
    )

    if fixed_train_idx is not None and fixed_val_idx is not None:
        scaler_train_idx = fixed_train_idx
    else:
        scaler_train_idx, _ = _inner_split(
            dev_idx,
            cfg["training"],
            split_seed,
            0,
            strat_labels,
            group_ids,
            cfg["data"].get("group_split", {}),
        )

    scaler = MinMaxScaler().fit(X_pp[scaler_train_idx])
    X_dev_sc = scaler.transform(X_pp[dev_idx]).astype(np.float32)
    X_test_sc = scaler.transform(X_pp[test_idx]).astype(np.float32)

    model = _load_model(model_dir, metrics, device)
    if model.net[0].in_features != X_pp.shape[1]:
        raise ValueError(
            f"Checkpoint expects {model.net[0].in_features} inputs, "
            f"but the selected data contains {X_pp.shape[1]} PP features."
        )

    shap_values = _compute_shap(
        model, X_dev_sc, X_test_sc, args.background_size, device
    )
    feature_names = [_strip_prefix(name) for name in pp_cols]
    bar_xlim = None if args.auto_limits else tuple(args.bar_xlim)
    beeswarm_xlim = None if args.auto_limits else tuple(args.beeswarm_xlim)
    _validate_limits(bar_xlim, "--bar-xlim")
    _validate_limits(beeswarm_xlim, "--beeswarm-xlim")

    bar_path, beeswarm_path = _save_plots(
        shap_values,
        X_test_sc,
        feature_names,
        output_dir,
        args.suffix,
        bar_xlim,
        beeswarm_xlim,
    )

    print(f"Material       : {material}")
    print(f"Model directory: {model_dir}")
    print(f"SHAP test rows : {len(test_idx)}")
    print(f"SHAP background: {min(args.background_size, len(X_dev_sc))} dev rows")
    print(f"Bar plot       : {bar_path}")
    print(f"Beeswarm plot  : {beeswarm_path}")


def _strip_prefix(name: str) -> str:
    for prefix in ("TCE_", "TCN_"):
        if name.startswith(prefix):
            return name[len(prefix):] + prefix[:-1]
    for prefix in ("DXP_", "QUA_", "SCA_", "MSS_", "IHR_", "SPE_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


if __name__ == "__main__":
    main()
