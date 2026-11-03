"""Fraction settings shared by the weight-prediction models."""

import math
from numbers import Real


def split_fraction(value, setting):
    """Validate a fractional share (for example, 0.2 means 20%)."""
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not 0 < value < 1):
        raise ValueError(f"{setting} must be a fraction strictly between 0 and 1 "
                         "(for example, 0.2 for 20%)")
    return float(value)


def validation_fraction(train_cfg, hpo_cfg=None):
    """Use the HPO override when present, otherwise the final-training share."""
    if hpo_cfg is not None and "val_fraction" in hpo_cfg:
        return split_fraction(hpo_cfg["val_fraction"], "HPO val_fraction")
    return split_fraction(train_cfg["val_fraction"], "training.val_fraction")


def split_ids_changed(previous, current):
    """A new split needs a matching promoted checkpoint for downstream models."""
    return any(set(previous.get(key, [])) != set(current.get(key, [])) for key in (
        "train_part_ids", "val_part_ids", "test_part_ids"))
