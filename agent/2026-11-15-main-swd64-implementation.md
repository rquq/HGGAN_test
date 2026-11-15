# MAIN SWD64 experiment — 2026-11-15

User approved the previously proposed experiment and requested commit/push.
Base MAIN: d881b59; implementation: `f849d9a`. Change only `training.distribution_projections:32→64` in
gan_iam_32.yml, gan_iam_64.yml, gan_iam_local.yml and gan_image.yml.
Distribution samples remain64; weight remains0.1. All other configuration,
architecture, optimizer, teacher, sampling and evaluation settings are unchanged.
ALPHA and both notebooks are untouched by this increment.

Motivation is reduced Monte Carlo projection noise, not a proven KID gain.
Directions already refresh each training call. This experiment must be assessed
at matched update budgets with unchanged metric streams, also monitoring HWD.
No new loss, metric tuning or Inception supervision was added.

## Checkpoint boundary

Existing32-direction MAIN GAN checkpoints store SWD direction buffers with
width32; the new loss has width64. The current loader restores these buffers
strictly. Consequently an old32-direction GAN checkpoint cannot resume directly
under these configs. Use a fresh GAN experiment, or retain the original32-setting
code/config for continuing the old run. No buffers or optimizer state were
silently discarded, and no compatibility mechanism was added. Existing W+B/R
pretrains and network architecture are unchanged and reusable.

## Verification

Check the full parsed YAML against baseline: the projection count must be the
only differing value. Bounded CPU multiscale SWD64 forward/backward and strict
save/reload/next-step equivalence test exercise64/128/256-channel features.
An explicit32→64 strict-load rejection confirms the stated continuation boundary.
No full GAN training, physical T4 timing/memory or KID improvement is established.

All listed bounded checks passed before commit. The existing diagram remains
pinned to the documented d881b59 baseline (32 projections); this config experiment
does not change its topology. No agents or Astra delegation were used.
