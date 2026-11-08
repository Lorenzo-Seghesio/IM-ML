# CustomGBDT reference for Fusion

`CustomGBDT` is a fifth reference model. It predicts one absolute part weight in
grams from the same unscaled PP feature matrix as the existing tree baselines.
Its ensemble/training loop is custom; individual weak learners are scikit-learn
`DecisionTreeRegressor` trees. M1, M2, Fusion, LightGBM, and XGBoost are unchanged.

## Run only this reference

The DoE1 reference configuration enables `active_models.custom_gbdt = 1`.
Use the new `--models` option to avoid launching other reference models:

```sh
python src/Reference_Models_W_Pred.py --material DOE1 --models custom_gbdt \
  --beta-values 0 0.1 0.25 0.5 1
```

This launches the configured training/HPO jobs; it is not a lightweight test.
`--optuna-seed` also controls the CustomGBDT study. The original active-model
configuration remains the default when `--models` is omitted. Configurations
without the new active-model key keep CustomGBDT off for compatibility.
ProBayes keeps it explicitly disabled until valid TP IDs and grouped splits are
supplied. Selecting CustomGBDT always runs its required TP-aware training,
independently of `training.test_point_training.enabled`, which controls the
optional TP mode for Encoder/MLP. It inherits the configured beta values and
sampler settings without changing the other models' training mode. Missing TP
IDs or disabled grouped splitting raise an error before any training starts,
instead of silently skipping an explicitly selected model. CustomGBDT reuses the
same authoritative M1 final train/validation/test split as the other references.

## Exactly the shared loss

The data objective is the existing shared implementation:

`L = L_w + beta * (s_w / s_within) * L_within`.

For every training mini-batch, the model:

1. Predicts all selected parts using its current additive tree ensemble.
2. Calls the same PyTorch loss as M1/MLP/Encoder and differentiates with respect
   to the predictions. Both TP means use the same mini-batch subset. Predicted
   TP means remain attached to autograd.
3. Fits one shallow CART tree to `-N_mb * dL/dprediction` across the **combined**
   multi-TP mini-batch. Multiplying by `N_mb` rescales the whole direction; the
   subsequent step-size search compensates for it.
4. Chooses a nonnegative step size that minimizes the **same exact objective**
   along that tree's prediction direction, then applies shrinkage via
   `custom_gbdt.learning_rate`.

This is first-order functional gradient boosting. The neural-network AdamW
parameter update is replaced by a tree addition. There is no fabricated Hessian,
no smoothing of the absolute loss, and no TP-by-TP optimizer step. Absolute loss
uses PyTorch's subgradient convention at zero. A zero or non-improving tree step
is skipped and remains recorded as an attempted boosting step.

For a fixed direction `h`, both loss terms are sums of `|a + gamma*b|`.
The absolute term uses `a = prediction - actual`, `b = h`, and weight `1/N_mb`.
Within each mini-batch TP, `a` and `b` are centered and weighted by
`beta * (s_w/s_within) / (B_mb*m_j)`. Each nonzero-slope term has breakpoint
`-a/b` and weight proportional to `|b|`. A weighted median of those breakpoints,
clipped at zero, gives an exact minimizer over `gamma >= 0`. Shrinkage in `(0,1]`
keeps the step between zero and that minimizer. Numerical checks reject a step
that increases its mini-batch objective. This does not guarantee improvement on
other mini-batches or validation data.

The general negative-gradient tree fitting approach is described in the
[scikit-learn gradient boosting documentation](https://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting).
The coupled-loss line search above is implemented explicitly in this repository.

## Training parity and differences

The training protocol matches Encoder/MLP where applicable:

- The same TP-aware sampler (default 4 distinct TPs × 16 distinct parts), epochs,
  randomized coverage, and explicit handling of small TPs.
- Fixed, equally TP-weighted normalization fitted from each actual training
  portion only; never validation/test targets. Near-zero factors retain warnings.
- Grouped CV/HPO, protected groups, edge exclusions, and authoritative M1 final
  train/validation/test split reuse. CV/HPO receive development data only.
- Epoch-level early stopping and learning-rate plateau scheduling driven by
  complete validation **absolute part-weight MAE**. The best epoch's tree prefix
  is restored. The epoch-zero constant predictor is not an extra candidate.
- The same beta sweep, with a separate HPO search per beta and selection solely
  by final validation part MAE. Test results do not select beta or checkpoints.
- The same full-group raw-gram metrics and actual/predicted per-TP standard
  deviations. No group target statistic is needed at inference.

Tree shrinkage is configured under `custom_gbdt.learning_rate`, separately from
neural `training.lr`. Neural `l1_lambda`, AdamW `weight_decay`, and dropout do not
apply to tree splits; tree capacity is controlled by depth, minimum leaf sizes,
and feature sampling. Thus beta zero is **pure MAE tree boosting**. It is not a
claim of identical fitting behavior to sklearn/LightGBM/XGBoost or their scores.

The initial constant is the **training-only median** weight, which minimizes the
absolute term for a constant predictor. Model initialization and feature splits
never use validation targets. Validation predictions are updated to monitor the
ensemble, not to fit trees or choose individual mini-batch step sizes.

Each accepted mini-batch step adds a tree; `epochs` therefore does not equal the
number of trees. History records attempted steps and accepted tree counts. This
CPU implementation favors a transparent objective over histogram-boosting speed.
A PP-only reference given identical inputs within a TP predicts identical weights
for those parts, regardless of loss choice.

## Configuration

`training.test_point_training` contains the existing sampler, beta-list, and
normalization settings. Tree parameters are separate:

```json
"custom_gbdt": {
  "learning_rate": 0.1,
  "max_depth": 3,
  "min_samples_leaf": 4,
  "min_samples_split": 2,
  "max_features": null
}
```

`custom_gbdt_hpo` defines its own search space, folds, epochs, patience, trials,
and sampler seed. The batch-size geometry remains fixed for all beta values.
Set `custom_gbdt_hpo.n_trials = 0` to skip HPO and use the fixed parameters above;
CV and final training still run. Final training uses the existing
`training.final_epochs` / `final_patience` overrides when present.

## Artifacts and inference

Artifacts follow `CustomGBDT/DOE1/run_best/<timestamp>/beta_<value>/`, with
per-beta and validation-selected best-overall directories as for Encoder/MLP.
They include a checkpoint, effective parameters, normalization, configuration,
history, raw metrics, per-TP CSVs, split IDs, feature schema, CV/HPO reports, and
standard diagnostic plots. `feature_importances.csv` contains split-impurity
importances aggregated using absolute tree step weights; it is not SHAP.

`best_model.joblib` is a versioned payload of parameters and tree state, avoiding
script-versus-package Python class import dependencies. Load it explicitly:

```python
from src.Utility.custom_test_point_gbdt import CustomTestPointGBDT

model = CustomTestPointGBDT.load("path/to/best_model.joblib")
weights_g = model.predict(X_pp)  # PP columns in the saved order; no TP IDs/means
```

The checkpoint stores the selected tree prefix. The saved `scalers.json` records
`scaling: none` and the PP column order. Direct `joblib.load` returns the payload;
use the class loader to obtain a prediction object.

## Lightweight verification

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/im-ml-mpl \
  .venv/bin/python -m unittest discover -s tests -p 'test_custom_test_point_gbdt.py'
```

These checks use small synthetic datasets, direct fixed Optuna trials, and mocked
plotting/HPO orchestration. They do not run a real dataset experiment or search.
