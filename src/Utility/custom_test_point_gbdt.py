"""Stochastic first-order tree boosting for the shared scalar-weight TP loss.

Trees fit negative loss gradients. An exact nonnegative one-dimensional line
search minimizes the same nonsmooth, TP-coupled MAE along each tree direction.
One complete multi-TP mini-batch produces at most one additive tree update.
"""
from __future__ import annotations

import joblib
import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from .m1_test_point_training import (
    TestPointMiniBatchSampler, encode_test_points, normalization_factors,
    test_point_loss, test_point_metrics,
)


def _features(values, expected_features=None):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("CustomGBDT requires nonempty finite 2D feature arrays")
    if expected_features is not None and values.shape[1] != expected_features:
        raise ValueError("CustomGBDT prediction features do not match the training schema")
    return values


def _targets(values, n_parts):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) != n_parts or not np.isfinite(values).all():
        raise ValueError("CustomGBDT targets must be finite with one scalar weight per part")
    return values


def loss_and_negative_gradient(predicted, actual, test_point_codes, beta, factors):
    """Return the shared loss terms and -N_mb*dL/dprediction (not a Hessian).

    Scaling the entire gradient by N_mb changes its magnitude, not its direction.
    The subsequent line search compensates for this scaling. In particular,
    gradients still flow through the predicted mini-batch TP means.
    """
    pred = torch.tensor(predicted, dtype=torch.float64, requires_grad=True)
    actual_t = torch.as_tensor(actual, dtype=torch.float64)
    codes_t = torch.as_tensor(test_point_codes, dtype=torch.long)
    loss, weight_mae, within_mae = test_point_loss(pred, actual_t, codes_t, beta, factors)
    gradient, = torch.autograd.grad(loss, pred)
    return ({"loss": loss.item(), "weight_mae_g": weight_mae.item(),
             "within_mae_g": within_mae.item()}, -len(actual) * gradient.detach().numpy())


def exact_tree_step(predicted, actual, direction, test_point_codes, beta, factors):
    """Minimize L(predicted + gamma*direction) for gamma >= 0 exactly.

    Every absolute term is |offset + gamma*slope|. Its nonconstant part is
    |slope|*|gamma - (-offset/slope)|, so a weighted median of these breakpoints
    minimizes the convex sum. Within-TP slopes and offsets are centered over
    matching mini-batch TP subsets, with weights lambda/(B_mb*m_j).
    """
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    direction = np.asarray(direction, dtype=np.float64).reshape(-1)
    codes = np.asarray(test_point_codes).reshape(-1)
    if (not len(actual) or predicted.shape != actual.shape or direction.shape != actual.shape
            or codes.shape != actual.shape):
        raise ValueError("Line-search predictions, targets, directions and TP codes must align")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("within_test_point_beta must be finite and nonnegative")
    error = predicted - actual
    breakpoints, weights = [], []

    def add_terms(offset, slope, multiplier):
        nonzero = slope != 0
        breakpoints.extend((-offset[nonzero] / slope[nonzero]).tolist())
        weights.extend((multiplier * np.abs(slope[nonzero])).tolist())

    add_terms(error, direction, 1 / len(actual))
    if beta:
        represented = np.unique(codes)
        coefficient = beta * factors["s_w"] / factors["s_within"]
        for tp in represented:
            mask = codes == tp
            centered_error = error[mask] - error[mask].mean()
            centered_direction = direction[mask] - direction[mask].mean()
            add_terms(centered_error, centered_direction,
                      coefficient / (len(represented) * int(mask.sum())))
    if not weights:
        return 0.0
    breakpoints, weights = np.asarray(breakpoints), np.asarray(weights)
    if not np.isfinite(breakpoints).all() or not np.isfinite(weights).all():
        raise ValueError("CustomGBDT line search encountered non-finite values")
    order = np.argsort(breakpoints, kind="stable")
    cumulative = np.cumsum(weights[order])
    median_idx = np.searchsorted(cumulative, cumulative[-1] / 2, side="left")
    return max(0.0, float(breakpoints[order[median_idx]]))


class CustomTestPointGBDT(RegressorMixin, BaseEstimator):
    """PP -> one absolute weight, as an additive ensemble of regression trees.

    The ensemble is custom; sklearn supplies individual CART weak learners.
    Neural-network parameter penalties/AdamW do not apply to these tree splits.
    """
    def __init__(self, learning_rate=0.1, max_depth=3, min_samples_leaf=4,
                 min_samples_split=2, max_features=None, random_state=42):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

    def _initialize(self, x, actual):
        if not np.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1:
            raise ValueError("CustomGBDT learning_rate must be in (0, 1]")
        if (isinstance(self.random_state, bool)
                or not isinstance(self.random_state, (int, np.integer)) or self.random_state < 0):
            raise ValueError("CustomGBDT random_state must be a nonnegative integer")
        self.n_features_in_ = x.shape[1]
        self.initial_weight_ = float(np.median(actual))  # Training targets only.
        self.trees_, self.tree_weights_ = [], []
        self.learning_rate_ = float(self.learning_rate)
        self.update_attempts_ = 0

    def fit(self, X, y, *, test_point_ids, X_val, y_val,
            validation_test_point_ids, training_config):
        train_custom_gbdt(self, X, y, X_val, y_val, training_config,
                          test_point_ids, validation_test_point_ids)
        return self

    def predict(self, X):
        check_is_fitted(self, "initial_weight_")
        x = _features(X, self.n_features_in_)
        prediction = np.full(len(x), self.initial_weight_, dtype=np.float64)
        for tree, scale in zip(self.trees_, self.tree_weights_):
            prediction += scale * tree.predict(x)
        return prediction

    @property
    def feature_importances_(self):
        check_is_fitted(self, "initial_weight_")
        importance = np.zeros(self.n_features_in_)
        for tree, scale in zip(self.trees_, self.tree_weights_):
            importance += abs(scale) * tree.feature_importances_
        return importance / importance.sum() if importance.sum() else importance

    def save(self, path):
        """Portable checkpoint: no dependency on script-vs-package import names."""
        check_is_fitted(self, "initial_weight_")
        state_keys = ("n_features_in_", "initial_weight_", "trees_", "tree_weights_",
                      "learning_rate_", "update_attempts_", "ref_training_info")
        joblib.dump({"format": "custom_test_point_gbdt_v1", "parameters": self.get_params(),
                     "state": {key: getattr(self, key) for key in state_keys}}, path)

    @classmethod
    def load(cls, path):
        payload = joblib.load(path)
        if payload.get("format") != "custom_test_point_gbdt_v1":
            raise ValueError("Unsupported CustomGBDT checkpoint format")
        model = cls(**payload["parameters"])
        for key in ("n_features_in_", "initial_weight_", "trees_", "tree_weights_",
                    "learning_rate_", "update_attempts_", "ref_training_info"):
            setattr(model, key, payload["state"][key])
        return model

    def boost_step(self, x, actual, codes, beta, factors, prediction):
        """Fit one direction and add a shrunk exact-loss step, if it helps."""
        metrics, negative_gradient = loss_and_negative_gradient(prediction, actual, codes, beta, factors)
        tree = DecisionTreeRegressor(
            max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split, max_features=self.max_features,
            random_state=self.random_state + self.update_attempts_)
        self.update_attempts_ += 1
        tree.fit(x, negative_gradient)
        direction = tree.predict(x)
        gamma = exact_tree_step(prediction, actual, direction, codes, beta, factors)
        scale = self.learning_rate_ * gamma
        after_loss = metrics["loss"]
        if scale:
            with torch.no_grad():
                after_loss = test_point_loss(
                    torch.as_tensor(prediction + scale * direction, dtype=torch.float64),
                    torch.as_tensor(actual, dtype=torch.float64), torch.as_tensor(codes), beta, factors)[0].item()
            # Guard numerical roundoff: an accepted step must not increase its
            # mini-batch objective. Other mini-batches/validation can still worsen.
            if after_loss > metrics["loss"]:
                scale, after_loss = 0.0, metrics["loss"]
        if scale:
            self.trees_.append(tree)
            self.tree_weights_.append(scale)
        return {**metrics, "loss_after_step": after_loss, "tree": tree, "scale": scale}


def train_custom_gbdt(model, X_tr, y_tr, X_val, y_val, tcfg,
                       test_point_ids_tr, test_point_ids_val):
    """Epoch/mini-batch schedule matching the neural reference experiments.

    Every mini-batch gets one functional-gradient boosting attempt. Validation
    MAE controls shrinkage scheduling, early stopping and ensemble restoration.
    """
    tp_cfg = tcfg.get("test_point_training", {})
    if not tp_cfg.get("enabled", False):
        raise ValueError("CustomGBDT requires test_point_training.enabled=true and valid TP IDs")
    x_tr, x_val = _features(X_tr), _features(X_val)
    if x_tr.shape[1] != x_val.shape[1]:
        raise ValueError("CustomGBDT training/validation feature schemas must match")
    actual_tr, actual_val = _targets(y_tr, len(x_tr)), _targets(y_val, len(x_val))
    codes, _ = encode_test_points(test_point_ids_tr)
    val_codes, _ = encode_test_points(test_point_ids_val)
    if len(codes) != len(actual_tr) or len(val_codes) != len(actual_val):
        raise ValueError("CustomGBDT test-point IDs must align with training/validation parts")
    combined_codes, _ = encode_test_points(np.concatenate([
        np.asarray(test_point_ids_tr, dtype=object), np.asarray(test_point_ids_val, dtype=object)]))
    if set(combined_codes[:len(codes)]) & set(combined_codes[len(codes):]):
        raise ValueError("A test point occurs in both CustomGBDT training and validation")
    for key in ("epochs", "patience", "print_every"):
        if not isinstance(tcfg[key], int) or tcfg[key] < 1:
            raise ValueError(f"CustomGBDT {key} must be a positive integer")
    factor = tcfg.get("scheduler_factor", 1.0)
    scheduler_patience = tcfg.get("scheduler_patience", 10)
    if not 0 < factor <= 1 or not isinstance(scheduler_patience, int) or scheduler_patience < 0:
        raise ValueError("CustomGBDT requires scheduler_factor in (0,1] and nonnegative integer patience")
    beta = tcfg.get("within_test_point_beta", 0.0)
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("within_test_point_beta must be finite and nonnegative")
    factors = normalization_factors(actual_tr, test_point_ids_tr,
                                    tp_cfg.get("normalization_epsilon", 1e-8))
    sampler = TestPointMiniBatchSampler(
        test_point_ids_tr, tcfg["batch_size"], tp_cfg.get("test_points_per_mini_batch", 4),
        tp_cfg.get("parts_per_test_point", 16), tcfg.get("mini_batch_seed", 42))
    model._initialize(x_tr, actual_tr)
    train_prediction = np.full(len(x_tr), model.initial_weight_)
    val_prediction = np.full(len(x_val), model.initial_weight_)
    history = []
    best_val_mae, best_tree_count, best_epoch = float("inf"), 0, 0
    best_validation_metrics = {}
    best_learning_rate = model.learning_rate_
    wait = scheduler_wait = 0
    scheduler_best = float("inf")
    for epoch in range(1, tcfg["epochs"] + 1):
        total_loss = total_weight_mae = total_within_mae = total_loss_after = 0.0
        seen_parts = steps = 0
        trees_before = len(model.trees_)
        epoch_learning_rate = model.learning_rate_
        for indices in sampler:
            idx = np.asarray(indices)
            result = model.boost_step(x_tr[idx], actual_tr[idx], codes[idx], beta, factors,
                                       train_prediction[idx])
            if result["scale"]:
                # Update caches incrementally; do not re-evaluate every prior tree
                # for every mini-batch. Trees never use validation targets to fit.
                train_prediction += result["scale"] * result["tree"].predict(x_tr)
                val_prediction += result["scale"] * result["tree"].predict(x_val)
            total_loss += result["loss"]
            total_loss_after += result["loss_after_step"]
            total_weight_mae += result["weight_mae_g"] * len(idx)
            total_within_mae += result["within_mae_g"]
            seen_parts += len(idx)
            steps += 1
        validation_metrics = test_point_metrics(actual_val, val_prediction, test_point_ids_val)
        val_mae = validation_metrics["MAE_w"]
        history.append({"epoch": epoch, "train_loss": total_loss / steps,
                        "train_loss_after_step": total_loss_after / steps,
                        "train_weight_mae_g": total_weight_mae / seen_parts,
                        "train_within_mae_g": total_within_mae / steps,
                        "val_mae_g": val_mae, "sampled_parts": seen_parts,
                        "optimizer_steps": steps, "trees_added": len(model.trees_) - trees_before,
                        "n_trees": len(model.trees_), "learning_rate": epoch_learning_rate,
                        **{f"val_{key}_g": value for key, value in validation_metrics.items()
                           if key != "test_points"}})
        # Same relative plateau threshold as ReduceLROnPlateau's default; tree
        # shrinkage is adjusted instead of neural-network optimizer parameters.
        if val_mae < scheduler_best * (1 - 1e-4):
            scheduler_best, scheduler_wait = val_mae, 0
        else:
            scheduler_wait += 1
            if factor < 1 and scheduler_wait > scheduler_patience:
                model.learning_rate_ *= factor
                scheduler_wait = 0
        if epoch % tcfg["print_every"] == 0:
            print(f"    ep {epoch:4d}  train_loss={total_loss / steps:.4f}  "
                  f"val_mae={val_mae:.4f}  trees={len(model.trees_)}")
        if val_mae < best_val_mae:
            best_val_mae, best_tree_count, best_epoch = val_mae, len(model.trees_), epoch
            best_validation_metrics = validation_metrics
            best_learning_rate = epoch_learning_rate
            wait = 0
        else:
            wait += 1
            if wait >= tcfg["patience"]:
                print(f"    Early stop  ep={epoch}  best_val_mae={best_val_mae:.4f}")
                break
    # Trees are immutable once added, so prefix restoration is sufficient.
    model.trees_ = model.trees_[:best_tree_count]
    model.tree_weights_ = model.tree_weights_[:best_tree_count]
    model.learning_rate_ = best_learning_rate
    model.ref_training_info = {
        "normalization": factors, "within_test_point_beta": beta,
        "training_config": dict(tcfg), "model_parameters": model.get_params(),
        "update_rule": "one first-order tree direction plus exact-loss line search per mini-batch",
        "best_validation_mae_g": best_val_mae, "best_validation_metrics": best_validation_metrics,
        "best_epoch": best_epoch, "best_n_trees": best_tree_count,
        "attempted_boosting_steps": model.update_attempts_, "history": history,
    }
    return model, best_val_mae
