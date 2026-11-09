# Sensor-only MultiEncoder reference and recovered four-PP M1

`Reference_Models_W_Pred.py` now supports `multi_encoder` alongside the existing
`encoder`. The existing Encoder concatenates signals over time before one CNN.
MultiEncoder instead gives each signal its own M1 `PressureEncoder`, concatenates
the latent features, and predicts weight with M1's merge MLP. There is no PP
branch and no process-parameter input to this network.

Enable it with `active_models.multi_encoder: 1` or run only this reference:

```bash
.venv/bin/python src/Reference_Models_W_Pred.py --material DOE1 --models multi_encoder
```

DoE1's `multi_encoder_hpo` section controls an independent Optuna study, initially
60 trials, 10 startup trials, 120 epochs and patience 30. Each signal independently
samples encoder depth, channels, kernels, pooling, and latent feature width.
`head_n_layers` and `head_hidden_size` control the merge head; their trial names
use `merge_*`. Training parameters and dropout are also searched. For other
material configs, an omitted `multi_encoder_hpo` inherits the original encoder
space with a feature-width range added.

Signal lengths come from the configured CSVs and may differ. Scaling is fitted
separately per signal on each fold's training rows. The saved `scalers.json`
contains the signal lengths and per-signal extrema for frozen evaluation.

The model uses the same group-safe splits, optional test-point loss and beta
selection, training procedure, validation selection, final tests and artifact
writers as the existing neural references. It writes to `RefModels/MultiEncoder`,
including architecture metadata, checkpoint, Optuna trials, training histories,
validation/test metrics, external Test 2 predictions, and per-signal GradCAM
using gradients of the joint weight prediction. Saved-checkpoint evaluation,
external evaluation and the network visualization loader support the new model.

## Recovered M1 reference

The user-selected M1 at
`outputs/DOE1/original/best_reference_run/M1/best_overall` was recovered using
the original `Clean_Data_1` dataset, its four saved PP columns, unchanged saved
scalers, original split IDs and unchanged checkpoint. The historical and current
divided datasets use different ID layouts; substituting the current Train folder
would silently change which physical parts the saved IDs identify.

Generated artifacts include `configuration.json`, `preprocessing.joblib`,
`signal_schema.json`, validation and Test 1 metrics/predictions, Test 2
metrics/predictions/plots/features, refreshed sensor features, recovery provenance,
and downstream-loader validation. They are also present in the matching
`M1/run_best/2026-10-28_01-39-46` copy. Original overwritten reports/features are
preserved under `legacy_artifacts`. Checkpoint bytes are unchanged.

The cached historical features and checkpoint reproduce the saved historical
test MAE: 0.02720810 g versus 0.02720826 g saved. Fresh feature extraction with
the current implementation has a maximum feature difference of approximately
0.00564 and an internal test MAE of 0.02730062 g. The refreshed features are used
consistently for downstream training; the audit records this distinction rather
than attributing an unverified cause to it. External Test 2 has 204 rows and MAE
0.01740782 g, with no overlapping historical start timestamps.

Original split counts remain 2,084 training / 232 validation / 579 Test 1 rows.
M2 receives four PP inputs and eleven encoder targets. Missing PP values use
the original loader's whole-historical-dataset median rule, reconstructed without
using external Test data. No M1 weights or MinMax scalers were fitted again.
Historical epoch/fold training logs cannot be reconstructed and were not invented.

Ready-to-use downstream configs live under
`outputs/DOE1/original/best_reference_run/recovered_configs`. For example:

```bash
IM_ML_MODEL_CONFIG=outputs/DOE1/original/best_reference_run/recovered_configs/M2_DoE1_config.json \
  .venv/bin/python src/M2_PP_to_F.py --material DOE1

IM_ML_MODEL_CONFIG=outputs/DOE1/original/best_reference_run/recovered_configs/RefModels_DoE1_config.json \
  .venv/bin/python src/Reference_Models_W_Pred.py --material DOE1 --models multi_encoder
```

Train M2 before launching Fusion with the recovered Fusion config. M1 does not
need retraining. Original-pipeline reuse now propagates the reused M1's saved
data schema to M2, Fusion and RefModels, preserving its four-PP feature order and
historical ID namespace.

The repair is reproducible with:

```bash
.venv/bin/python -m src.Utility.repair_legacy_m1 \
  --model-dir outputs/DOE1/original/best_reference_run/M1/best_overall \
  --data-dir data/DoE1_Dataset_July2026_Arburg_ABS/Clean_Data_1 \
  --test-2-dir data/DoE1_Dataset_July2026_Arburg_ABS/Clean_Data_1_2_Train_Test_Divided/Test
```
