"""Sensor-only reference: independent M1 encoders followed by a merge MLP."""
import copy

import numpy as np
import torch
from torch import nn


class MultiEncoderModel(nn.Module):
    def __init__(self, curve_encoders, signal_lengths, head_hidden, dropout):
        super().__init__()
        from src.M1_PPPT_to_W import PressureEncoder, MergeHead
        if len(curve_encoders) != len(signal_lengths) or not signal_lengths:
            raise ValueError("One encoder and time-series length are required per signal")
        self.signal_lengths = [int(length) for length in signal_lengths]
        if any(length <= 0 for length in self.signal_lengths):
            raise ValueError("Signal lengths must be positive")
        self.encoders = nn.ModuleList([
            PressureEncoder(cfg["conv_channels"], cfg["conv_kernels"], cfg["n_f_features"],
                            cfg.get("enc_pool_kernels", [])) for cfg in curve_encoders])
        self.merge = MergeHead(sum(cfg["n_f_features"] for cfg in curve_encoders), head_hidden, dropout)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[-1] != sum(self.signal_lengths):
            raise ValueError("Expected sensor input (B, 1, sum(signal_lengths))")
        signals = x.split(self.signal_lengths, dim=-1)
        features = torch.cat([encoder(signal) for encoder, signal in zip(self.encoders, signals)], dim=1)
        return self.merge(features)


def suggest_params(trial, search_space, base_config):
    from src.M1_PPPT_to_W import suggest_hyperparams
    space = dict(search_space)
    if "head_n_layers" in space:
        space["merge_n_layers"] = space.pop("head_n_layers")
    if "head_hidden_size" in space:
        space["merge_hidden_size"] = space.pop("head_hidden_size")
    base = {"curve_encoders": base_config["curve_encoders"], "pp_branch": "direct",
            "merge_hidden": base_config.get("head_hidden", [32]), "dropout": base_config.get("dropout", .2)}
    model, training = suggest_hyperparams(trial, space, 0, base)
    if "l1_lambda" in space:
        value = space["l1_lambda"]
        training["l1_lambda"] = (trial.suggest_float("l1_lambda", float(value[0]), float(value[1]), log=True)
                                 if isinstance(value, list) else float(value))
    return {"curve_encoders": model.get("curve_encoders", copy.deepcopy(base["curve_encoders"])),
            "signal_lengths": base_config["signal_lengths"],
            "head_hidden": model.get("merge_hidden", base["merge_hidden"]),
            "dropout": model.get("dropout", base["dropout"])}, training


def fit_scalers(values, signal_lengths=None):
    if signal_lengths is None:
        return float(values.min()), float(values.max())
    signals = np.split(values, np.cumsum(signal_lengths)[:-1], axis=1)
    return [float(s.min()) for s in signals], [float(s.max()) for s in signals]


def scale_inputs(values, low, high, signal_lengths=None):
    if signal_lengths is None:
        return ((values - low) / (high - low + 1e-8)).astype(np.float32)
    low = np.repeat(np.asarray(low), signal_lengths)
    high = np.repeat(np.asarray(high), signal_lengths)
    return ((values - low) / (np.asarray(high) - low + 1e-8)).astype(np.float32)


def gradcam(model, values, out_dir, device):
    """Backpropagate the joint weight prediction to each signal's last Conv1D."""
    import matplotlib.pyplot as plt
    activations, gradients, hooks = {}, {}, []
    model.eval()
    try:
        for i, encoder in enumerate(model.encoders):
            conv = [m for m in encoder.convs if isinstance(m, nn.Conv1d)][-1]
            hooks.append(conv.register_forward_hook(
                lambda module, inp, out, i=i: activations.__setitem__(i, out.detach())))
            hooks.append(conv.register_full_backward_hook(
                lambda module, gin, gout, i=i: gradients.__setitem__(i, gout[0].detach())))
        tensor = torch.as_tensor(values[:, None, :], dtype=torch.float32, device=device)
        model.zero_grad(set_to_none=True)
        model(tensor).sum().backward()
    finally:
        for hook in hooks:
            hook.remove()
    fig, axes = plt.subplots(len(model.encoders), 1, figsize=(10, 3 * len(model.encoders)), squeeze=False)
    for i, (signal, axis) in enumerate(zip(np.split(values, np.cumsum(model.signal_lengths)[:-1], axis=1), axes[:, 0])):
        cam = (gradients[i].mean(dim=-1, keepdim=True) * activations[i]).sum(dim=1).clamp_min(0).cpu().numpy()
        cam /= np.maximum(cam.max(axis=1, keepdims=True), 1e-12)
        heat = np.mean([np.interp(np.linspace(0, 1, signal.shape[1]), np.linspace(0, 1, len(row)), row)
                        for row in cam], axis=0)
        axis.plot(signal.mean(axis=0), label=f"Signal {i + 1} mean")
        other = axis.twinx()
        other.fill_between(np.arange(len(heat)), heat, alpha=.3, color="red")
        other.set_ylabel("GradCAM")
        axis.set(xlabel="Time step", ylabel="Scaled signal", title=f"MultiEncoder — signal {i + 1}")
        axis.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "xai_gradcam.png")
    plt.close(fig)
