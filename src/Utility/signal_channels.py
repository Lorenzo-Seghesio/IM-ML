"""Preserve signal order when forming multichannel Conv1D inputs."""

import numpy as np
import torch


def stack_signal_arrays(signals):
    """List of (N, T) signals -> (N, C, T), without resampling or padding."""
    if not signals or any(signal.ndim != 2 for signal in signals):
        raise ValueError("Expected nonempty time-series arrays with shape (N, T)")
    if len({signal.shape for signal in signals}) != 1:
        raise ValueError("Multichannel signals must have the same sample count and time-step length")
    return np.stack(signals, axis=1)


def stack_signal_tensors(signals):
    """Accept (B, C, T) or concatenate ordered (B, 1, T) signal tensors."""
    if isinstance(signals, (list, tuple)):
        if not signals or any(s.ndim != 3 or s.shape[1] != 1 for s in signals):
            raise ValueError("Expected one (B, 1, T) tensor per signal")
        if len({(s.shape[0], s.shape[2]) for s in signals}) != 1:
            raise ValueError("Multichannel signals must have the same sample count and time-step length")
        return torch.cat(signals, dim=1)
    if signals.ndim != 3:
        raise ValueError("Expected multichannel input with shape (B, C, T)")
    return signals
