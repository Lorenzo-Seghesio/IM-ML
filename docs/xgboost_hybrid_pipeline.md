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

All supplied DoE1 configurations now use
`data/DoE1_Dataset_July2026_Arburg_ABS/Clean_Data_1_2_Train_Test_Divided`.
`data.dataset_dir` selects this root. If it contains `Train/`, that folder is
used for training regardless of whether an external test is configured:

- Only `Train/` participates in training, HPO, CV, fitting preprocessing and
  train/validation/internal-test splitting. The current Train folder has 2,893 parts.
- The internal held-out partition is evaluated as **test_1**.
- All 204 parts in the explicitly configured external folder are evaluated as **test_2**, using the same fixed
  weights and fitted preprocessing. No test_2 filtering, splitting or fitting is done.
- M1, M2, all reference models, GBDT heads and selected Fusion models produce
  separate test_1/test_2 metrics, predictions and plots. M2 targets for test_2
  are generated from its own curves by the frozen M1 encoder; engineered
  descriptors use that folder's own physical time grid.

The filenames differ between folders (`DoE_1_Dataset_*.csv` and
`Val_Dataset_*.csv`); routing discovers their scalar/pressure/temperature suffixes.
Product and batch IDs are scoped to each folder: Test IDs restart at zero and
are never joined against Train's saved feature rows. Evaluation files and SHAP
sampling tables explicitly identify the partition.

The optional external path is explicit in **each model's own config**, including
the separate M2 and Fusion configs:

```json
"evaluation": {
    "enabled": true,
    "plots": true,
    "test_2_dir": "data/DoE1_Dataset_July2026_Arburg_ABS/Clean_Data_1_2_Train_Test_Divided/Test"
}
```

Paths are relative to the repository root (absolute paths also work). The folder
can live anywhere and need not be named `Test`. Set `test_2_dir` to `null`, an
empty string, or omit it to disable test_2 and its SHAP scope. A neighboring
`Test/` folder by itself never enables testing. A configured missing folder is
an error. The existing `evaluation.enabled` switch still controls final testing
overall. Stage configs are independent: disabling test_2 for GBDT or M2 does not
disable it for Fusion. Resumed hybrid stages remove stale test_2 reports when
the path is unset and reevaluate a changed path without refitting the models.

If both subfolders are absent, the configured flat folder is used with the
existing train/validation/test pipeline and legacy `test` filenames. Test_2 is
omitted unless an external path is explicitly configured. Existing explicit CSV
configurations remain supported.

Training-file fingerprints accompany M1 split contracts. Historical checkpoints
from the unified dataset are incompatible with the newly divided Train folder,
even when some numeric IDs coincide. Run original first to create a new compatible
`best_reference_run`, then run the hybrids. Test_2 changes do not invalidate
training checkpoints. Existing historical runs are preserved.

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
`outputs/DOE1/original/best_reference_run/M1/best_overall`. It checks the PP columns/order, signal
sources/order/count/widths, saved architecture, and exact split membership.
The original checkpoint is read and copied into the hybrid run; the hybrid
does not modify it. B and C also default to that checkpoint's authoritative split, so
the independent test TPs remain comparable. To change these sources, edit each
hybrid's own `teacher` and `split` settings.

On the first full pipeline launch, the original family runs before the hybrids
and publishes a complete original repetition under
`outputs/DOE1/original/best_reference_run/`. These directories need not exist
beforehand. Among original repetitions with the same data and independent test
membership, the run with the lowest **Fusion test_1 MAE** wins (legacy flat data
continues to use its single test MAE). Test_2 never ranks runs. Within each run,
every alpha is trained and the winner is chosen by **validation weight MAE**
among unchanged pretrained Fusion and all fine-tuned alphas.
Only that winner is evaluated on the test partition and passed through XAI.
Its `Fusion/<timestamp>/best_fusion/test_1_metrics.json` supplies the run's score;
the selected alpha and its validation selection criterion are recorded.
The whole repetition is copied together: M1, M2, Fusion,
Reference (when enabled), and comparison reports. Replacing the snapshot removes
stale artifacts from the previous winner. Validation-only runs cannot replace it.
`ALL` has no Fusion and therefore does not publish a best reference snapshot.

`best_reference_run/reference_run.json` identifies the original run directory,
date/time, repetition, M1 timestamp, selected alpha, and Fusion test MAE. Each
hybrid reads the authoritative split from this snapshot's M1 and saves its own
`reference_run.json` and `run_info.json` with that identity and the source split's
SHA-256 hash. Its comparison rows also identify the original run and timestamp.
These records remain useful after a later original run replaces the snapshot.
Running a hybrid alone requires this snapshot to have been produced first (or
an explicitly configured compatible source). Old validation-only outputs are
not automatically promoted to a test-selected reference.

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
An original repetition runs all M1 launches → M2 → Fusion → optional Reference.
`models.original.n_m1_run_per_pipeline_run` is a positive integer (default `1`)
controlling the M1 launches within each full original repetition. For example:

```json
"original": {
    "enabled": true,
    "n_runs": 2,
    "n_m1_run_per_pipeline_run": 3,
    "reference": {"enabled": true, "n_runs": 1}
}
```

This runs M1 three times, then M2/Fusion/Reference once, and repeats that entire
sequence twice: six M1 launches total. Each M1 launch includes its configured HPO,
CV, final training and beta experiments. M1 training seeds increment by one within
a full repetition while its split seed stays fixed. Explicit Optuna seeds also
increment by one; otherwise each launch generates a fresh sampler seed. M1's
existing validation-MAE selection retains the best checkpoint in that repetition's
`M1/best_overall`; M2 and Fusion use that checkpoint. All M1 launch artifacts stay
in the same repetition's M1 folders, with their existing timestamped run directories.
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

DoE1 configurations set **`evaluation.enabled: true`** for original M1, M2,
Fusion and Reference, and for every hybrid's neural M1, GBDT, M2 and Fusion.
Test_1 scores use the independent `test_part_ids` saved by M1; test_2 evaluates
the entire separate Test folder. Validation remains
the source for early stopping and hyperparameter selection. Previously these
configs explicitly set the switch to false: reports labeled validation really
were computed on validation data, rather than merely having misleading names.
Set a model's switch to false only when intentionally requesting validation-only
evaluation. Hybrid checkpoint reuse still produces final test reports when
testing is subsequently enabled. Test SHAP also requires Fusion's evaluation
switch; global Train SHAP is available without enabling final tests.

Weight reports contain MAE, RMSE, R², relative MAE as a percent of mean absolute
part weight, per-TP errors, TP-average MAE and within-TP MAE using the existing
shared retrospective metric definitions. M2 saves per-feature reconstruction
MAE and normalized MAE. Missing actual descriptors are identified in M2 reports;
`observed_feature_MAE` excludes unavailable descriptors rather than scoring
imputed values as observations. Fusion comparison rows include reconstruction
MAE/NMAE alongside weight metrics.

Teacher reports require measured signals; Fusion inference requires only PP.
No group means or weight labels are inference inputs. Test metrics do not select
hybrid hyperparameters, beta values or training checkpoints. The requested
original reference-run ranking **does use Fusion test_1 MAE across repetitions**:
the winning reference score is therefore a run-selection score, not an untouched
estimate after selection. Alpha selection within a repetition uses validation
MAE exclusively. A separate unseen dataset is needed for an untouched estimate
after choosing the reference repetition.

For A, head-only CV with an already-trained encoder is conditional on that
representation and **is not an unbiased evaluation of the complete hybrid**.
`cv_interpretation.json` reports fold overlap with encoder training. B saves
fold-specific neural checkpoints by default; the head uses one only when its
held-out IDs match and neither encoder training nor early stopping used those
TPs. `cv.fold_checkpoint_dirs` can supply existing compatible checkpoints.
Architecture selection on development data remains a separate source of
conditioning even with fold-specific fits. The independent test set is the
final assessment for each fixed fitted model.

Hybrid CV now saves the original-style measured/predicted scatter, residual
scatter and residual histogram for each fold, an all-fold diagnostic figure,
and a fold error comparison. Neural M1/M1_pp and XGBoost get weight diagnostics;
M2 gets per-feature reconstruction diagnostics in physical units. Model A
copies its pretrained original M1's CV plots into the hybrid M1 snapshot.
`cv.plots` controls these figures and `cv.save_per_fold_plots` controls separate
fold figures (default true for weight and false for multi-feature output).

Fusion has its own `cv.enabled/n_folds/plots` settings. Each development fold
refits both the tree head and M2 using training/early-stopping partitions, then
scores PP-only predictions on that fold's held-out TPs. Compatible fold-specific
encoders are used when available; otherwise the final representation is reused
and that conditioning is recorded in `cv_interpretation.json`. Hyperparameters
are already selected, so all these CV plots remain indicative. Independent test
IDs are excluded from CV. The engineered family's M1 is a deterministic feature
extractor, so predictive M1 metrics are not applicable; its weight predictions
and plots live under XGBoost.

Final hybrid plots live under each stage's `validation/`, `test_1/` and `test_2/` folders:
measured/predicted scatter, residual scatter, residual histogram, and batch-colored
scatter/residuals with per-batch CSVs. Multi-output M2 has one numbered subfolder
per feature. Missing actual descriptors are excluded from diagnostic plots.
`evaluation.plots` controls these figures. Validation filenames are explicitly
labeled validation. Evaluation does not refit models on the test partition.

## Outputs and inference

Outputs use `outputs/DOE1/<family>/runs/<timestamp>/run_001/`, with
`M1`, `M2`, `XGBoost` and `Fusion` stages, without a redundant final `DOE1`
folder. Each stage retains
`run_best` and `best_overall`. Original repeats also contain `RefModels`.
The original root has two functional subfolders: `runs/` for all experiments
and `best_reference_run/` for the complete winning snapshot. Direct standalone
original scripts use `runs/standalone/`; the family launcher publishes the
coherent reference after a complete repetition. Existing historical output
directories are left intact.

Original Fusion sweeps keep training histories, checkpoints, validation metrics,
predictions and plots in `alpha_<value>/`. The unchanged pretrained Fusion is
saved in `no_finetuning/` with its original checkpoints, validation metrics,
predictions and plots. It uses the same validation IDs and is eligible to win.
Its metadata has `fine_tuned: false` and `alpha: null`; its fine-tuning info
records `enabled: false` and zero training epochs.
`alpha_comparison.csv/.png` compares validation weight MAE across this baseline
and all alphas and marks the winner. Each alpha's epoch
is still chosen by its validation composite loss; the comparison between alphas
uses weight MAE. Ties prefer no fine-tuning, then the first configured alpha.
`best_alpha_selection.json` records the decision. `best_fusion/` contains a copy
of the selected checkpoint and validation artifacts, `selection.json`, final
test metrics/predictions/plots, model metadata and XAI outputs. The pretrained
baseline also receives final testing and XAI if it wins; all other candidates
retain validation results only. With `evaluation.enabled: false`,
the selected model produces validation outputs and XAI without testing.

The restored winner also evaluates its M2 reconstruction on the same final
partition. `best_fusion/m2_reconstruction/` contains `test_metrics.json`,
`test_feature_metrics.csv`, part-level feature targets/predictions, measured vs
predicted scatter, residual scatter and histograms, plus batch-colored plots
and per-batch reports. Errors compare M2 predictions with the frozen M1 sensor
features in their original units. Per-feature and mean NMAE use M2's saved
training scaler. These scores are also included in `m2_best_metrics.json` and
`fusion_model_info.json`; they describe M2 after the selected fine-tuning, or
unchanged M2 if `no_finetuning` wins. This diagnostic does not affect model
selection. With final testing disabled, artifacts are labeled validation.

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

Every Fusion family now explains the selected model in this order:
`xai/train/` for global importance over the **full Train dataset**, then
`xai/test_1/`, then `xai/test_2/`. Legacy flat datasets have `xai/train/` and
`xai/test/`. Final-test scopes are omitted when evaluation is disabled. Model
selection has already finished before any of these explanations are computed.

Every scope includes separate direct/indirect pathway SHAP, tied-input
`single_pp` SHAP, beeswarms, importance/pathway bars and comparison CSVs.
Sampling is balanced across batches. `xai.explain_samples` and
`xai.background_samples` control sample budgets; budgets automatically increase
to at least the number of batches so every batch is represented. Background
parts come only from the fitted training partition and the same background is
used for all scopes. `explained_samples.csv`, `sampling.json` and
`partitions.json` record selected IDs, batch coverage and dataset namespaces.
The default explanation/background budgets are 64. This is a representative
sample of the complete Train distribution, rather than an explanation of only
its validation subset.

All Fusion plots use the labels in `DOE1_DISPLAY_COLUMNS` from
`src/Utility/doe1_columns.py`, including pathway and interaction labels. Model
input schemas and inference column ordering are preserved.

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
reuse, SHAP plots, original and hybrid final-test membership, validation-only
evaluation, CV and final diagnostic plots, complete reference-run promotion,
split provenance, and launcher counts.
These are synthetic smoke tests, not real-data experiments or final testing.
