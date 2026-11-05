# Configuring test and validation shares

M1, M2, and the weight reference models use fractions: `0.2` means 20%.
All fractions must be strictly between 0 and 1.

The DoE1 defaults now point to
`Clean_Data_1_2_Train_Test_Divided` using `data.dataset_dir`. Its `Train/`
folder alone is split using the fractions below; the internal held-out portion
produces **test_1**. The optional `evaluation.test_2_dir` selects the complete
external folder for **test_2**, using frozen models and saved preprocessing.
The supplied configs point this to `Test/`; change it to any compatible dataset
folder, or set it to `null`/omit it to disable test_2 even if `Test/` exists.
The setting is independent in each model config, including M2 and Fusion.
Test_2 never participates
in HPO, CV, early stopping or selection. Product IDs can overlap between the two
folders and are kept in separate namespaces.

With a flat dataset folder, the original single-test behavior and filenames are
preserved and no test_2 evaluation is attempted unless its path is explicitly
configured. Training-data fingerprints prevent
old unified-dataset checkpoints from silently supplying splits for the new Train
folder. Original must generate a compatible reference before running the hybrids.

For M1 and M2, edit the following fields in the selected JSON configuration
under `config/DoE1` or `config/ProBayes`:

```json
{
    "preprocessing": {
        "test_fraction": 0.2
    },
    "optuna": {
        "val_fraction": 0.15
    },
    "training": {
        "val_fraction": 0.1
    }
}
```

These are excerpts; keep the other settings in each section.

| Setting | Share held out | Used for |
| --- | --- | --- |
| `preprocessing.test_fraction` | Fraction of the cleaned dataset | One held-out test set, created before HPO and CV |
| `optuna.val_fraction` | Fraction of each HPO fold's training pool | Early stopping during HPO |
| `training.val_fraction` | Fraction of each final CV fold's training pool, or of the development pool for final training | Early stopping and final model selection |

The development pool contains the data remaining after the initial test split.
For example, with 1,000 cleaned parts, `test_fraction: 0.2` reserves 200 parts for
testing. Final training with `training.val_fraction: 0.1` uses 720 training parts
and 80 validation parts from the remaining 800.

In HPO and final CV, the number of folds controls the outer evaluation share
(`optuna.n_folds` and `training.k_folds`). The validation fractions above apply
inside each fold's training pool. For example, a five-fold run on 800 development
parts evaluates on 160 parts per fold. Its remaining 640 parts are split into
576 training and 64 validation parts when `val_fraction` is `0.1`.

If the HPO `val_fraction` is omitted, it falls back to `training.val_fraction`,
preserving the behavior of older configurations. Supplied configurations keep
their previous fractions as defaults.

## Reference models

The reference-model configuration uses the same `preprocessing.test_fraction`
and `training.val_fraction` fields. Set the HPO validation share separately in
`encoder_hpo.val_fraction`, `mlp_hpo.val_fraction`, and
`custom_gbdt_hpo.val_fraction`.

Standard LightGBM and XGBoost use their configured fold counts for HPO and CV
evaluation and fit the full training pool without an inner early-stopping
validation split. Their final models fit all development parts.

## Reusing M1 splits

M2 loads M1's exact train/validation/test IDs when its saved split is available.
DoE1 reference models do the same when `m1_split_dir` points to an available M1
`data_processing.json`. Their local test and final-training validation fractions
apply when they create fresh splits; final CV and HPO use their local settings.
Original Fusion uses M1's held-out test_1 set and the separate full test_2 folder,
and has no separate HPO or CV stage.
Hybrid Fusion also reuses this test set and has optional indicative development
CV, configured separately with `cv.enabled` and `cv.n_folds`.

DoE1 configurations enable final testing with `evaluation.enabled: true`.
Setting it to false evaluates validation data instead. For four-family pipeline
runs, hybrids read M1's exact IDs from
`outputs/DOE1/original/best_reference_run/M1/best_overall/`, and record the
selected reference run's timestamp and source split hash. The original snapshot
contains the complete repetition selected by Fusion test MAE; see
[the family pipeline guide](xgboost_hybrid_pipeline.md).

To change the shared held-out test set or final train/validation split, change
M1's settings and rerun M1 before rerunning downstream models. M1 promotes a
checkpoint with changed split IDs to `best_overall` together with its scalers,
features, and split metadata so downstream models receive a matching set of
artifacts. A changed HPO validation share alone does not change these final IDs.
M1's `data_processing.json` records requested and achieved test and final
validation fractions, plus the HPO validation fraction.

Group boundaries, protected groups, edge-group exclusions, and integer rounding
can make the achieved percentages differ from the requested shares. Existing
group protections remain in force. Optional target-outlier filtering removes
development parts after the initial split, so it can also affect the achieved
test share.

## Fusion fine-tuning

Set `finetuning.val_fraction` in `Fusion_DoE1_config.json`,
`Fusion_PP_config.json`, or `Fusion_ABS_config.json`:

```json
"finetuning": {
    "val_fraction": 0.2
}
```

Keep the other fine-tuning settings in that section. This example reserves 20%
of M1's development pool for validation during fine-tuning. It creates a new
train/validation split even when M1 has saved validation IDs. M1's test IDs stay
fixed, and the split respects configured group, protected-group, and edge-group
rules. Every alpha in the sweep uses the same split.

All alphas finish fine-tuning before Fusion selects the lowest validation weight
MAE among the unchanged pretrained model and the fine-tuned candidates. The
pretrained candidate has its own `no_finetuning/` folder with validation reports,
plots and saved original weights. Only the winner is evaluated on the independent
test IDs and sent through XAI. Its checkpoint, final reports and explanations are saved in
`best_fusion/`; the per-alpha folders keep training and validation artifacts.
The sweep's comparison table and plot use validation metrics only.

The default `null` (or an omitted field) reuses M1's saved train/validation IDs.
If those IDs are unavailable, Fusion reserves 10% of development data for
validation. Set a numeric value strictly between 0 and 1 to choose another share.

The command-line option overrides the configuration:

```bash
python src/Fusion_M1M2_WPred.py --material DOE1 --val-fraction 0.2
```

Fusion records the source, exact part IDs, requested fraction, and achieved
fraction in `finetuning_split.json`. Pretrained model weights and scalers are
loaded as before; the fine-tuning fraction only controls the fine-tuning split.
When reconstruction of missing scalers is needed, it uses the original M1
training validation fraction from saved metadata/configuration (or the legacy
10% fallback).
