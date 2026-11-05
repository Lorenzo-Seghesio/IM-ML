# Saved-run test evaluation

Existing weights and fitted preprocessing were loaded; no model was retrained. All models use the same 465 held-out parts from the saved M1 split.

Fusion was selected on validation only: **alpha_0.1**, validation MAE **0.022283 g**.

| Model | Test MAE (g) | Test RMSE (g) | Test R² |
| --- | ---: | ---: | ---: |
| Reference/LightGBM | 0.025612 | 0.034600 | 0.8778 |
| Reference/XGBoost | 0.027072 | 0.035312 | 0.8728 |
| M1 | 0.037522 | 0.046142 | 0.7828 |
| Fusion (alpha_0.1) | 0.038016 | 0.046151 | 0.7827 |
| Reference/Encoder | 0.038921 | 0.048418 | 0.7608 |
| Reference/MLP | 0.083004 | 0.093429 | 0.1093 |

M2 reconstruction: mean feature MAE **0.153680**, normalized MAE **0.062135**, mean feature R² **0.7082**.

See `comparison.csv`, `validation_replay_checks.json`, `evaluation_manifest.json`, and each stage folder for metrics, part predictions, test-point/batch reports, plots and XAI.

Historical CV plots and training records were copied unchanged. CustomGBDT has no checkpoint in this run. No reference run was promoted.

Selected Fusion M2 after fine-tuning (alpha 0.1): mean feature MAE **0.153075**, normalized MAE **0.059448**, mean feature R² **0.7013**. Detailed metrics and plots are in `Fusion/2026-11-04_23-38-40_DOE1/best_fusion/m2_reconstruction/`.
