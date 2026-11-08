# Multichannel signal encoders

New M1 runs use one Conv1D encoder. Each source in `data.timeseries_csvs`
enters a separate channel, in configuration order: `(batch, signals, time)`.
All signals must have the same number of time steps; the loader does not
resample or pad them. The DoE1 signals currently all have 599 time steps.
Removing or adding a source automatically changes the first convolution's
input channel count when training starts.

The M1 model configuration defines the shared stack once:

```json
{
    "conv_channels": [3, 8, 16],
    "conv_kernels": [5, 3],
    "enc_pool_kernels": [2],
    "n_f_features": 12
}
```

`n_f_features` is the total joint feature width. Those features are concatenated
with the existing PP embedding (or direct PP inputs for M1_pp) before entering
the existing merge head. Scaling, PP layers, merge layers, losses, splits,
optimizers, and stopping rules retain their existing behavior.

M1 HPO samples one shared stack using `enc_n_conv_layers`, `enc_channels`,
`enc_kernel_size`, `enc_pool_kernel_size`, and `n_f_features`. The input channel
count comes from the configured sources and is never searched. Trial parameters
use `enc_ch_0`, `enc_k_0`, etc., rather than separate `enc_c0_*`, `enc_c1_*`
parameters for every signal. `n_f_features` now controls the shared output,
rather than a separate width for each signal.

`sensors_features_f.csv` exports `f_joint_0`, `f_joint_1`, etc. Each feature can
depend on every input signal. `feature_summary.json` records the ordered input
signals, channel count, joint feature width, and merge width. M2 and Fusion use
these columns in their existing feature alignment paths.

The reference Encoder also uses one signal per channel. Its existing encoder
HPO optimizes a single stack and scalar prediction head; the number of input
channels is fixed from the data. GradCAM shows the signals on their shared time
axis, together with the encoder's joint temporal activation.

Historical M1 checkpoints with `curve_encoders` still load using their saved
architecture. Old training configurations are converted to a shared stack at
training startup, taking the first branch's convolution settings and summing
the old output widths. Historical reference checkpoints retain their original
concatenated input layout during saved-checkpoint and external evaluation.

Train new M1 and reference checkpoints to use the new architecture. Regenerate
M2 and Fusion from the new M1 features as well; historical M2 targets represent
the old encoder features.
