# M1 within-test-point loss experiment

M1 keeps its original PP branch, independent sensor encoders, fusion head, and
single scalar absolute-weight output. M2 and Fusion training are unchanged.
A **test point (TP)** contains parts with the same nominal process parameters;
a **training mini-batch** contains parts from multiple TPs for one optimizer step.

Configure `training` as follows (the DoE1 configuration already enables this):

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

`beta_values` accepts a number or a list. CLI overrides are also supported:

```sh
python src/M1_PPPT_to_W.py --material DOE1 --beta-values 0 0.1 0.25 0.5 1
python src/M1_PPPT_to_W.py --material DOE1 --beta-values 0.25
```

These commands launch the configured training/HPO pipeline; they are not tests.
Optuna, early stopping, learning-rate scheduling, grouped CV, feature export, and
checkpointing remain available. The DoE1 configuration still defaults to Optuna.

## Loss and normalization

Each optimizer step uses `L_w + within_test_point_beta * s_w / s_within * L_within`.
`L_w` is conventional per-part MAE. `L_within` centers predictions and targets
using the **same mini-batch TP subset**, averages centered absolute errors within
each TP, then averages across TPs. Predicted means remain in the autograd graph.
At beta zero the optimization loss is exactly the original unnormalized MAE;
the sampler remains identical for all beta values.

`s_w` is the training parts' global mean absolute deviation. `s_within` is the
equally TP-weighted mean absolute deviation around each full training TP mean.
Both are fitted once per actual training split, including each CV/HPO fold, and
remain fixed during training. Small factors produce explicit warnings and are
floored by `normalization_epsilon`; saved records contain raw and floored values.
No validation or test targets enter normalization.

## Sampling and split compatibility

`batch_size` must equal `test_points_per_mini_batch * parts_per_test_point`.
At least two TPs per mini-batch are required, and a training split with fewer than
the configured number of distinct TPs raises an error. Each epoch randomly draws
distinct TPs from those with remaining shuffled part queues. To avoid stranding
TP queues, TPs that need every remaining optimizer step are selected first; other
choices are random. The epoch uses the minimum feasible number of mini-batches
for those queues. All training parts are visited. Exhausted queues are topped up only to complete a mini-batch or the
epoch tail, always avoiding duplicate parts within a mini-batch. Consequently an
epoch can have more sampled parts than the training dataset; histories record the
actual number of parts and optimizer steps. If a TP has fewer than the requested
parts, the sampler warns and uses all its distinct parts; the mini-batch is smaller.
No TP is silently excluded and there is no within-mini-batch replacement.

With grouped splitting enabled, M1 keeps `data.group_split.protected_group_ids`
in training only: they never enter the final test set, final validation set,
CV holdouts, or inner validation splits used by CV and HPO. Edge exclusions
still apply independently. `data.group_col` must supply valid TP IDs and `data.group_split.enabled`
must be true when this experiment is enabled. Raw TP IDs define loss/metrics;
protected groups merged by the split policy are still separate TPs in the loss.
Part IDs must be unique. DoE1 uses `Batch ID` as the TP identifier, following its
existing data grouping contract. This assumes each Batch ID has one nominal PP
setting. The ProBayes configurations have no TP IDs and explicitly retain legacy
random-part MAE training; provide valid TP IDs and grouped splits before enabling.

An independent Optuna `batch_size` search is disabled when TP training is enabled;
change all three sampling sizes together in `training`. Other HPO parameters are
unchanged. HPO studies have distinct beta/run names, and all betas share the split
seed and initial Optuna sampler seed. Each beta runs HPO independently, so adaptive
trials can differ once their validation results differ.

## Metrics, artifacts, and selection

Evaluation uses all available parts per TP in the evaluated split. Reports retain
MAE/MSE/RMSE/R² and add raw-gram `MAE_w`, `MAE_TPavg`, `MAE_within`, and per-TP actual
and predicted means and population standard deviations (`ddof=0`). TP means are
used retrospectively for metrics only, never as inference inputs.

Each beta has separate output/run-best/best-overall directories, for example
`run_best/<timestamp>/beta_0.25/`. All betas from one run share the timestamp
folder. Each beta stores the checkpoint, configuration,
normalization factors, training history, complete validation and test metrics,
per-TP CSVs, CV fold factors/histories, existing diagnostic plots, feature exports,
and split contracts. HPO trial user attributes (also exported in `hpo_trials.csv`)
include fold factors/histories and per-TP metrics.

The lowest final **validation absolute part-weight MAE** selects beta; ties follow
the configured beta order. Test results do not select beta. The selection record
is written under `beta_experiments/<timestamp>/beta_selection.json`. The selected
run updates the original `best_overall` location when validation MAE improves (or
the existing split policy changes), preserving downstream checkpoint/feature/split
filenames. Earlier checkpoints without validation MAE are replaced on the first
successful experiment; their test MAE is not used for comparison.

Each sweep also saves `beta_mae_comparison.png` and `beta_mae_comparison.csv`
beside `beta_selection.json`. The plot compares validation and held-out test
`MAE_w`, `MAE_TPavg`, and `MAE_within` in grams across the tested betas, marking
the validation-selected beta. These files are copied to `best_overall` when
the selected run is promoted.

## Lightweight verification

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/im-ml-mpl \
  .venv/bin/python -m unittest discover -s tests -p 'test_m1_test_point_training.py'
```

The tests use synthetic targets and tiny linear models, including one-epoch fold
wiring checks. They do not run dataset training or an Optuna search.
