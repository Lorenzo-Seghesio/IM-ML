# Reference models: within-test-point weight variation

Encoder, MLP and the new CustomGBDT share M1's test-point-aware sampler,
normalization, loss, and retrospective evaluation helpers. Their original architectures and scalar
absolute-weight outputs are preserved. LightGBM and XGBoost receive the same
complete-test-point evaluation metrics and retain their existing tree-fitting
methods. Each standard tree baseline runs once, independently of the TP-aware beta sweeps.
CustomGBDT is described in [CustomGBDT](custom_test_point_gbdt.md).

## Configure and run

The DoE1 reference configuration enables:

```json
{
  "batch_size": 64,
  "test_point_training": {
    "enabled": true,
    "beta_values": [0, 0.1, 0.25, 0.5, 1.0],
    "test_points_per_mini_batch": 4,
    "parts_per_test_point": 16,
    "normalization_epsilon": 1e-8
  }
}
```

These settings belong in `training`. Set `enabled` to `false` (or omit
`test_point_training`) to run the original four references as before: Encoder
and MLP use shuffled part mini-batches, MAE plus L1 regularization, and the
configured HPO batch-size search, with one run each and no beta sweep.
LightGBM/XGBoost keep their standard fitting methods. Selected CustomGBDT always
uses the TP-aware loss, sampler and beta sweep, even when the neural TP switch
is false. Its local configuration enables TP training without changing Encoder
or MLP. It requires valid TP IDs and grouped splitting; missing prerequisites
raise an error before training. Existing split settings and M1 split reuse still
apply. With neural TP training disabled, `--beta-values` applies only to an active
CustomGBDT.

`beta_values` accepts a scalar or list;
`within_test_point_beta` records the active coefficient in each experiment.
For example:

```sh
python src/Reference_Models_W_Pred.py --material DOE1 --beta-values 0 0.1 0.25 0.5 1
python src/Reference_Models_W_Pred.py --material DOE1 --beta-values 0.25
```

These commands launch the configured training/HPO jobs. Active-model flags still
control which models run. Encoder, MLP and CustomGBDT each select their own beta using final
**validation absolute part-weight MAE**, with configured order breaking ties.
Test metrics never select beta. The original grouped split policy, protected
groups, edge exclusions, and authoritative M1 split reuse remain in place.
Across M1, M2, Fusion, and all reference models, enabled grouped splitting keeps
`data.group_split.protected_group_ids` in training only, including CV/HPO folds
and reconstructed validation splits. Imported M1 splits are checked too; if an
older split puts a protected group in validation or test, rerun M1 to regenerate
the split before running downstream models.

The ProBayes configurations explicitly disable TP training because they have no
TP IDs. Supply a valid `data.group_col` and enable `data.group_split.enabled`
before enabling those experiments. DoE1 assumes `Batch ID` identifies one nominal
process-parameter setting. Each part ID must be unique.

## Training and evaluation

The neural data loss is `L_w + beta * (s_w / s_within) * L_within`, exactly as in
[M1](M1_test_point_training.md). Means for the within loss use matching TP subsets
inside the current training mini-batch; predicted means remain differentiable.
The part MAE is part-weighted, and the within loss is equally TP-weighted.
Normalization is fitted once from each actual training portion, including every
CV/HPO fold, with explicit warnings and a floor for near-zero factors.

The original reference-model `l1_lambda` parameter penalty is added afterwards.
Thus beta zero restores the original MAE data term **plus the existing L1
regularization**. AdamW weight decay, scheduling, early stopping, Optuna pruning,
and XAI reports are preserved. Early stopping and scheduling use absolute
validation MAE, excluding the combined loss and parameter penalty.

All betas share the sampling strategy and seeds. The sampler covers every
training part each epoch, prevents duplicate parts within a mini-batch, and
limits repeats needed to finish an epoch. It warns and uses a smaller mini-batch
when a selected TP has too few distinct parts. Sampling-size consistency and
multiple TPs per mini-batch are enforced. Optuna cannot vary `batch_size`
independently in this mode; other search parameters remain available. Each beta
runs HPO independently, so adaptive trials may differ between beta values.

Validation, test, and CV reports add raw-gram `MAE_w`, `MAE_TPavg`, `MAE_within`,
and per-TP actual/predicted standard deviations (`ddof=0`). Group means use every
available part of each evaluated TP and are never inference inputs. Standard LightGBM/XGBoost
use the same retrospective metrics on their held-out CV folds and final test
set; their existing fitting methods do not use the new loss or beta selection.
CustomGBDT uses exact-loss functional gradient boosting on TP mini-batches.

A PP-only MLP given identical inputs within a TP necessarily predicts identical
weights for those parts. The new objective does not add per-part information to
its inputs.

## Artifacts

Each neural model writes, for example:

- `Encoder/DOE1/run_best/<timestamp>/beta_0.25/`: checkpoint, normalization factors,
  configuration, training history, validation/test metrics, per-TP CSVs, scalers,
  part-ID split contract, CV fold factors/histories, HPO results, and XAI reports.
- `Encoder/DOE1/beta_experiments/<timestamp>/beta_selection.json`: beta comparison.
- `Encoder/DOE1/best_overall/`: validation-selected artifacts, retaining existing
  filenames. Per-beta best checkpoints are in `best_overall/beta_<value>/`.

MLP and CustomGBDT follow the same structure. All betas from one run share the
timestamp folder. Final training preserves the authoritative M1
training and validation IDs when available. The original best-overall location
is updated when validation MAE improves or the split contract changes. Legacy
checkpoints lacking validation MAE are replaced on the first successful sweep.

Each Encoder, MLP, and CustomGBDT sweep also writes `beta_mae_comparison.png`
and `beta_mae_comparison.csv` beside its `beta_selection.json`. Three panels
compare validation and held-out test `MAE_w`, `MAE_TPavg`, and `MAE_within` in
grams across the tested betas, marking the validation-selected beta. The plot
and CSV are copied to `best_overall` when the selected run is promoted.

## Lightweight tests

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/im-ml-mpl \
  .venv/bin/python -m unittest discover -s tests -p 'test_reference_test_point_training.py'
```

Tests use synthetic parts, tiny networks, and mocked HPO searches/XAI plots.
The shared mathematical and sampler checks remain in
`tests/test_m1_test_point_training.py`.
