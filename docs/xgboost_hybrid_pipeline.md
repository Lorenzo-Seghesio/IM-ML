# Four weight-prediction families

The original M1 → M2 → Fusion models keep their neural training and Fusion
fine-tuning. Three additional families replace the weight head with XGBoost:

| Family | Teacher inputs to XGBoost | PP-only Fusion inputs to XGBoost |
| --- | --- | --- |
| `gbdt_pppt` (A) | Frozen pretrained M1 PP embedding + measured CNN features | Same frozen PP embedding + M2-predicted CNN features |
| `gbdt_ppdirect_pt` (B) | Original PP + M1_pp CNN features | Original PP + M2-predicted CNN features |
| `gbdt_engineered` (C) | Original PP + physical signal descriptors | Original PP + M2-predicted descriptors |

M1_pp trains a baseline M1 with `model.pp_branch: "direct"`. Its normalized PP
inputs go directly into the neural merge head during representation training;
there are no PP MLP parameters or PP embedding HPO parameters. The tree then
receives **unscaled original PP** and the trained CNN embeddings. A uses the
saved M1 PP scaler and MLP. The serialized hybrid inference teacher omits the
neural merge head entirely.

All hybrid Fusion models train M2 only for feature reconstruction, then freeze
it. They perform no Fusion fine-tuning and no differentiation through XGBoost.
The original Fusion's alpha sweeps, gradient XAI and fine-tuning remain available.

## Before running the DoE1 experiments

The cleaned DoE1 CSVs contain sample labels (`t0`, `t1`, …). Model C automatically
searches beside each signal CSV for **`info/signal_time_grid.csv`**. The supplied
DoE1 metadata maps `column` to `time_seconds`: 599 samples spanning 0 to
22.49975883960724 seconds, shared by pressure 1, pressure 2 and temperature.
Feature extraction uses this exact time vector; no manually entered sampling
frequency is needed.

The default configuration is:

```json
{
  "feature_extraction": {
    "auto_discover_time_grid": true
  }
}
```

Column labels must uniquely match the signal CSV's samples. Metadata rows are
aligned by label, then timestamps are checked for finite, strictly increasing
seconds and matching curve widths. The full launcher and feature extractor use
the same resolver; malformed or missing metadata fails before training starts.

Explicit `timestamps_s`, `timestamps_csv`, `sampling_interval_s`, or
`sampling_frequency_hz` settings override automatic discovery. They may be
configured globally or per signal in `feature_extraction.signals`; a source's
`time` settings have highest priority. `timestamps_csv` accepts `time_seconds`
or `time_s`, with optional `column` labels; `timestamp_column` and
`timestamp_label_column` customize those names. Set `time_grid_csv` to select
another metadata file, or `auto_discover_time_grid: false` to disable the
dataset-local search. Relative paths use the repository root. Numeric sample
headers can explicitly be interpreted as seconds with `column_times_seconds`.

The M1 feature-artifact folder saves `signal_time_grids.json` and resolved grids
inside `configuration.json`. Feature caches and downstream checkpoint
signatures include the resolved times, so editing a grid invalidates reuse
even when the signal CSVs and model configurations have not changed.

Model A defaults to the existing pretrained checkpoint at
`outputs/DOE1/original/M1/best_overall`. It checks the PP columns/order, signal
sources/order/count/widths, saved architecture, and exact split membership.
The original checkpoint is read and copied into the hybrid run; the hybrid
does not modify it. B and C also default to that checkpoint's authoritative split, so
the independent test TPs remain comparable. To change these sources, edit each
hybrid's own `teacher` and `split` settings.

On the first full pipeline launch, the original family runs before the hybrids
and publishes a matching M1/M2 pair at the `output.best_overall_dir` paths in
the original configurations. These directories need not exist beforehand.
Across compatible original repetitions, publication selects M1 by validation
MAE and copies its corresponding M2, preserving their shared feature space.
The canonical M1 path supplies the teacher and split contract for subsequent
hybrid families. Running a hybrid alone requires this original checkpoint to
have been produced first (or an explicitly configured compatible source).

## Execute

Activate the repository environment and launch from the repository root:

```sh
source .venv/bin/activate
python Run_WPred_Pipeline.py --material DOE1 --dry-run
python Run_WPred_Pipeline.py --material DOE1
```

`config/WPred_Pipeline_config.json` contains only family selection and execution
counts. `models.<family>.n_runs` counts **complete independent repetitions**,
not trials or epochs. Every enabled family/repetition executes sequentially.
An original repetition still runs M1 → M2 → Fusion → optional Reference.
`original.reference.enabled` and `n_runs` control that optional baseline stage.
Legacy pipeline files using `n_runs_m1`, `n_runs_m2`, etc. still follow their
previous stage-loop procedure. `--material`, `--config`, and `--optuna-seed`
remain supported.

Training seeds come from the model configurations; repetition `r` adds
`1009 * (r - 1)`. The same offset applies to an explicit CLI Optuna seed.
`preprocessing.split_seed` keeps baseline partitions fixed while training seeds
vary. Hybrid split contracts use the exact configured source checkpoint IDs.
DoE1 model-specific configurations are supplied; other datasets need their own
individual hybrid configurations and valid TP IDs/grouped splitting.

Run any teacher independently (defaults to M1 preparation and XGBoost only):

```sh
python src/M1_GBDT_PPPT_to_W.py --material DOE1
python src/M1_GBDT_PPdirect_PT_to_W.py --material DOE1
python src/M1_GBDT_engineered_to_W.py --material DOE1
```

Run all hybrid stages, or resume selected stages in the **same** directory:

```sh
python src/M1_GBDT_PPPT_to_W.py --material DOE1 \
  --run-dir outputs/DOE1/gbdt_pppt/my_run --stages m1 xgboost m2 fusion
python src/M1_GBDT_PPPT_to_W.py --material DOE1 \
  --run-dir outputs/DOE1/gbdt_pppt/my_run --stages m2 fusion
```

The analogous options work for B and C. `--config` selects another individual
hybrid configuration. Configuration/content hashes, saved preprocessing, input
data hashes, feature ordering, source checkpoint bytes and split membership
must match for checkpoint reuse. `checkpoint.reuse: false` forces training
of a selected stage. New pipeline repetitions get separate directories;
resuming a stage explicitly reuses its directory. Feature extraction caches
are shared safely across repetitions using content/time/options hashes.
To reuse a compatible new-format M1_pp, XGBoost, or M2 artifact from another
run, set `checkpoint.source_dir` in that stage's individual configuration to
its `run_best/model` or `best_overall` directory. Its manifest must match; an
incompatible explicit source fails instead of silently loading different
features, preprocessing or partitions.

## Individual configuration files

For each family, the `M1_GBDT_*_DOE1_config.json` contains data/preprocessing,
teacher source, split source, XGBoost parameters, XGBoost `training`, `optuna`,
`cv`, checkpoint and evaluation settings. Its `configs` entries reference the
separate M2/Fusion configurations. Model B additionally references
`M1_pp_DoE1_config.json` for its neural architecture, optimizer, epochs, HPO,
sampling and loss. The existing M1 configuration is used if A's `teacher.train`
is explicitly enabled.

Each `M2_gbdt_*_DOE1_config.json` owns M2 architecture, optimizer, loss, epochs,
HPO, indicative CV, normalization, checkpoint reuse and evaluation.
`normalization.pp` and `normalization.targets` accept `minmax` (baseline default)
or `standard`. Both are fitted feature-wise on the actual training partition.
Reconstruction minimizes MAE in normalized feature space; validation model
selection uses normalized reconstruction MAE so mixed physical units do not
dominate the objective. Predictions are inverse-transformed before XGBoost.

Each `Fusion_gbdt_*_DOE1_config.json` owns final evaluation and XAI settings.
Hybrid `finetuning.enabled` must stay false. Pipeline configuration never
overrides these training, architecture, HPO or evaluation parameters.

XGBoost HPO uses the baseline group/edge/protected-TP helpers, TPE, MAE and
separate inner grouped early-stopping sets. `optuna.enabled: false` or
`optuna.n_trials: 0` skips search and uses `xgboost` parameters directly.
`cv.enabled/n_folds` independently control the reported head CV.
XGBoost inputs retain their original scales; only missing values are imputed
using training medians, with all-missing columns kept at zero. M1/M2 neural
inputs use their saved normalization. All preprocessing is saved.

## Engineered feature definitions

Each configured signal produces all 18 requested descriptors by default
(54 for the current three signals). `feature_extraction.features` can select
an ordered subset. Names are signal-prefixed, e.g. `p1__area` and
`T1__rise_time_10_90_s`, and retain physical amplitude units.

- Baseline: first value by default; alternatively `min`, `initial_mean`
  (`baseline_samples`), or an explicit numeric baseline.
- Maximum: the first maximum if there are ties. `peak_time_s` uses its actual
  timestamp. `duration_s` is last minus first timestamp.
- Statistics: arithmetic mean and population standard deviation (`ddof=0`).
- Integration: trapezoidal integration on the physical time grid.
  `area_above_baseline` integrates only the positive piecewise-linear amplitude
  above baseline, inserting zero crossings. `time_weighted_mean = area/duration`.
- Derivatives: adjacent secant slopes using actual time differences. Maximum
  rise slope is bounded below by zero; maximum fall slope is bounded above by
  zero and retains its negative sign.
- Thresholds: baseline plus 10%/90% of the baseline-to-peak amplitude. Use
  linearly interpolated last rising crossings before the first peak, and first
  falling crossings after it. Interval slopes use 80% of that amplitude;
  falling slopes are negative.
- Constant curves: interval times/slopes and above-baseline area are zero.
  Missing crossings are NaN, not fabricated durations. `amplitude_epsilon`
  configures the constant/amplitude tolerance.
- Invalid amplitudes: `invalid_values` selects `interpolate`, `raise`, or `nan`.
  Interpolation uses only the individual curve and physical time grid. An
  unrepairable curve produces NaN descriptors; a training-only imputer handles
  them downstream. Timestamp errors always fail explicitly.

Amplitude features retain sensor units; areas have sensor-units × seconds,
slopes sensor-units/second, and time features seconds. No weight target or
dataset statistic enters extraction. Changing curves, IDs, options or physical
timestamps invalidates its cache.

## Evaluation and scientific interpretation

DoE1 configurations explicitly set `evaluation.enabled: false`. Runs report
validation results and can generate validation XAI without scoring the
independent test set. Enable final testing **separately in each model's own
configuration** when ready. Test XAI also requires Fusion's evaluation switch.
Original legacy configurations without this new key retain their previous
standalone behavior; the supplied DoE1 configurations explicitly disable it.
Checkpoint promotion in the configured DoE1 runs uses validation metrics.

Weight reports contain MAE, RMSE, R², relative MAE as a percent of mean absolute
part weight, per-TP errors, TP-average MAE and within-TP MAE using the existing
shared retrospective metric definitions. M2 saves per-feature reconstruction
MAE and normalized MAE. Missing actual descriptors are identified in M2 reports;
`observed_feature_MAE` excludes unavailable descriptors rather than scoring
imputed values as observations. Fusion comparison rows include reconstruction
MAE/NMAE alongside weight metrics.

Teacher reports require measured signals; Fusion inference requires only PP.
No group means or weight labels are inference inputs. Test metrics never select
hybrid hyperparameters, beta values, checkpoints or repetitions.

For A, head-only CV with an already-trained encoder is conditional on that
representation and **is not an unbiased evaluation of the complete hybrid**.
`cv_interpretation.json` reports fold overlap with encoder training. B saves
fold-specific neural checkpoints by default; the head uses one only when its
held-out IDs match and neither encoder training nor early stopping used those
TPs. `cv.fold_checkpoint_dirs` can supply existing compatible checkpoints.
Architecture selection on development data remains a separate source of
conditioning even with fold-specific fits. The independent test set is the
final assessment when explicitly enabled.

## Outputs and inference

Outputs use `outputs/DOE1/<family>/runs/<timestamp>/run_001/`, with
`M1`, `M2`, `XGBoost` and `Fusion` stages, without a redundant final `DOE1`
folder. Each stage retains
`run_best` and `best_overall`. Original repeats also contain `RefModels`.
C's M1 folder contains physical feature tables/schema/cache references rather
than a neural checkpoint. Every repetition saves its exact split contract,
resolved configurations, seeds, model states, preprocessing and predictions.
`outputs/DOE1/comparison.csv` and a timestamped copy consolidate this launch's
reports, with explicit family, role and partition columns.

Load a self-contained Fusion predictor, including feature schema and all fitted
preprocessing, without the original pretrained output folders:

```python
from src.Utility.hybrid_models import HybridPredictor

predictor = HybridPredictor.load(
    "outputs/DOE1/gbdt_pppt/my_run/Fusion/best_overall/predictor"
)
weights_g = predictor.predict(pp_dataframe)  # columns must match saved order
sensor_features = predictor.predict_features(pp_dataframe)  # original feature units
```

NumPy input is also supported with the saved column order. Teacher checkpoints
under `XGBoost/.../predictor` provide `predict_teacher(pp, curves=...)`; C accepts
`engineered=...` in the saved descriptor order. Fusion `.predict` accepts PP only.

## XAI

Hybrid Fusion uses prediction-based KernelSHAP for true PP-only total effects
and a separate direct/indirect intervention-path analysis. Background parts
come only from training. The existing Fusion presentation functions produce
single-PP beeswarms/importance, pathway bars, net effects, pathway agreement and
cancellation plots. An additional exact tree-contribution CSV explains the
head's actual PP/feature inputs. KernelSHAP sample/background/evaluation counts
are configurable in each Fusion file. Duplicate-path explanations have the
usual correlated-input/intervention caveats; tied-PP SHAP is the primary total
effect. No gradient approximation is presented as a tree gradient.

## Lightweight verification

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/im-ml-mpl \
  .venv/bin/python -m unittest discover -s tests
```

Synthetic tests cover physical feature math/cache invalidation, actual tiny
XGBoost/Optuna fits, all three teacher/Fusion paths, M1_pp/M2 HPO, normalized
reconstruction, frozen PP-only inference, self-contained save/load, checkpoint
reuse, SHAP plots, original validation-only M1/M2 training, and launcher counts.
These are synthetic smoke tests, not real-data experiments or final testing.
