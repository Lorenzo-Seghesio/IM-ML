# Batch-average and part-deviation weight prediction

The DOE1 M1 config enables a two-output merge head. It predicts an average
component and a signed deviation component; their sum is the part-weight
prediction. The shared merge head sees both sensor features and the process
parameter embedding. The losses supervise the intended interpretation of the
components; they do not restrict the average head to PP inputs alone.

`model(..., return_components=True)` returns `(N, 2)` columns `[average,
deviation]`. The ordinary forward call returns their sum, preserving weight
metrics and XAI interfaces. Older one-output checkpoints/configs remain supported.

## Targets and three loss variants

For each actual production batch j, M1 computes its mean measured part weight
μj and each part's target deviation `weight_i - μj`. Batch IDs, measured means,
and deviations are targets/metadata, never inference inputs. Whole production
batches stay together in every training/validation/test split, including HPO
and CV, so training means cannot incorporate held-out labels. The exact targets
are saved in `batch_targets.csv`; Fusion reloads them instead of recomputing
means from a potentially different scalar-data subset.

Each configured beta is trained under all three variants:

| Variant | Weight objective |
| --- | --- |
| `components` | `beta * MAE_avg + (1-beta) * MAE_dev` |
| `components_weight` | component objective `+ gamma * MAE_weight` |
| `components_weight_balance` | previous objective `+ balance_lambda * mean_j(abs(mean_i_in_j(predicted_deviation_i)))` |

`MAE_weight` compares `predicted_average + predicted_deviation` with measured
part weight. Every MAE uses original weight units. The component MAEs average
over parts, exactly as specified: a batch with more parts has more influence.
Repeated batch-mean labels are intentional per-part supervision, not an extra
copy of the data. The balance term instead gives each production batch equal
weight. It encourages zero-mean predicted deviations; finite loss coefficients
do not guarantee exact equality or convergence.

For the balance variant, optimizer batches contain complete production batches.
The configured training batch size becomes a soft sample-count limit: a larger
production batch is kept intact. This avoids pushing the mean of a randomly
selected fragment to zero. At beta 0 or 1, one component has no direct component
supervision; the total-weight/balance terms still apply in their variants.

## Configuration

`config/DoE1/M1_DoE1_config.json` includes:

```json
"decomposition": {
    "enabled": true,
    "beta": [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1],
    "gamma": 1.0,
    "balance_lambda": 1.0,
    "variants": ["components", "components_weight", "components_weight_balance"],
    "share_hpo": true,
    "hpo_beta": 0.5
}
```

`beta` also accepts a scalar. `--beta 0.25 0.5` overrides the configured beta
list. `--config path.json` is supported by M1, M2, and Fusion.

With `share_hpo: true`, architecture/training hyperparameters are optimized
once, using the first variant and the configured beta nearest `hpo_beta`.
All subsequent betas/variants reuse that configuration and the same training
seed and splits. Set `share_hpo: false` to repeat the configured HPO search for
each beta/variant. Manual mode skips HPO. Each beta still receives its own CV
and final training, with the configured epoch budgets.

`data.include_nominal_pp: true` keeps the nominal parameters listed in
`data.nominal_pp_cols` (or, by default, `edge_group_split.columns`) as inputs.
They remain available to the edge-group split policy as well. M2 and Fusion
reuse M1's exact PP list and order. Batch decomposition requires a group column
and `group_split.enabled: true`.

## Selection and output structure

Early stopping and beta selection use **validation MAE of the full part weight**.
Test metrics are reported for each model and never used to select beta. Each
variant keeps its own winning beta. A winner across all variants is also saved
at the run root; compatible best-overall checkpoints are updated when the
validation full-weight MAE improves. A changed input/signal/split contract
starts a new comparison rather than comparing incompatible checkpoints.

```text
M1/DOE1/run_best/<timestamp>/
    beta_comparison.csv
    best_model.pt, best_metrics.json, ...   # overall run winner
    experiment_variants.json
    components/
        beta_0/, beta_0.1/, ...
        best_model.pt, ...                 # best beta in this variant
    components_weight/...
    components_weight_balance/...

M1/DOE1/variants/<variant>/best_overall/...
M2/DOE1/variants/<variant>/{run_best/<timestamp>,best_overall}/...

Fusion/<timestamp>_DOE1/
    variant_alpha_comparison.csv
    components/alpha_0.0/...
    components_weight/alpha_0.0/...
    components_weight_balance/alpha_0.0/...
```

Every M1 candidate saves training history, test component predictions, batch
targets, feature exports, splits/scalers, and resolved model/loss settings.
Best-overall promotion keeps these artifacts together. Distinct beta/variant
provenance IDs prevent accidental mixing of encoder feature spaces.

## M2 and Fusion

M2 detects M1's `experiment_variants.json` and trains a matching feature
predictor for each variant's selected beta. An M2 checkpoint from a different
M1 provenance is replaced even when its old feature MAE was lower: those
feature spaces are not interchangeable.

Fusion detects the same variants and runs the configured alpha sweep for each.
For a decomposed checkpoint, its objective is:

```text
alpha * M1_weight_objective(beta, gamma, balance_lambda)
    + (1-alpha) * mean_feature_MAE
```

All weight-objective settings come from the selected M1 checkpoint. Only M2
is trainable; M1's PP branch and two-output merge head remain frozen, with
dropout disabled. Test predictions include both components and their sum.
Existing split-path, tied-input, and M1-merge XAI explain the total predicted
weight. A mismatched decomposed M1/M2 pair fails before fine-tuning.

The existing launcher continues to run M1 → M2 → Fusion → Reference. The first
three scripts automatically handle their loss variants; no launcher changes
or separate manual variant invocations are needed. To run later:

```bash
python Run_WPred_Pipeline.py --material DOE1
```

The implementation was tested with unit tests and tiny synthetic one-epoch
experiments. These do not run the production dataset pipeline.
