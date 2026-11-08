# M1 sensor inputs concatenated over time

New M1 and M1_pp runs concatenate the sensor signals end to end in
`data.timeseries_csvs` order, after their existing per-signal scaling. One
single-channel Conv1D encoder processes the result. Three signals of 599
samples become `(batch, 1, 1797)`. Signals may have different lengths; they
are not padded or resampled. The reference encoder already uses this layout.

The model configuration defines the encoder once:

```json
{
    "encoder_input_layout": "concatenate_time",
    "n_signals": 3,
    "conv_channels": [1, 8, 16],
    "conv_kernels": [5, 3],
    "enc_pool_kernels": [2],
    "n_f_features": 12
}
```

Training resolves `n_signals` from the configured sources and fixes the first
convolution's input channel count at one. HPO samples one CNN stack and one
total `n_f_features` output width. Its encoder parameters use `enc_ch_0`,
`enc_k_0`, etc., rather than independent `enc_c0_*` parameters for each signal.

The encoder's features are concatenated with the existing PP features before
the existing merge head. PP layers, merge layers, scaling, losses, splits,
optimizers, and stopping rules retain their previous behavior.

`sensors_features_f.csv` exports `f_concat_0`, `f_concat_1`, etc. These features
describe the entire concatenated sequence. `feature_summary.json` records the
source order, individual lengths, total input length, and feature width. M2,
Fusion, and the hybrid feature loaders use these exported features as before.

Old checkpoints still load with their saved per-signal architecture. Old
training configs are converted to one shared CNN using the first branch's
convolution settings and the sum of the old feature widths. Retrain M1 to use
the new architecture, then regenerate M2 and Fusion from its new features.
