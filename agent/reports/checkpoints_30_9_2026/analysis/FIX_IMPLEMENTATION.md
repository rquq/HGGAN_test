# Follow-up implementation: DEV and alpha

This addendum records code changes made after the checkpoint audit in `REPORT.md`.

## DEV changes

- `StyleEncoder` now layer-normalizes the global pooled vector before global projection, and normalizes attended local evidence before its residual with the local query. Local tokens bypass the global MLP that had reduced their reference-dependent variation by about 20× in the checkpoint probes.
- Local tokens are deterministic from the reference. The stochastic Gaussian posterior and KL regularizer apply only to the global token. A bounded local-evidence gate always preserves at least half of the local signal.
- Style-transfer CTC's final weight is raised from 0.08 to 0.20 in the 32px, 64px, and local GAN configs.
- Generator features are masked at each valid width inside the residual block, before later convolutions can read conditional-normalization values in padded columns. Training BatchNorm moments ignore padded columns. The output-normalization stage receives the same valid lengths.

## alpha synchronization

All tracked model source, library, metric, tool, and data files were copied from DEV and checked byte-for-byte. DEV's ignored model backup directories were copied as well; pretrained checkpoint SHA-256 hashes match. The only active configuration difference across the 32px, 64px, and local GAN configs is `num_critic_train`: DEV uses 4 and alpha uses 2. Dataset, schedule, and loss settings match.

The old checkpoint files and metric tables were not edited. Strict loading still succeeds because no trainable parameter shapes changed, but the encoder semantics and KL objective changed; retrain from the same initialization/seed to compare the ratio fairly.

## Verification performed

- Python byte-compilation passed for DEV network/library code.
- Both existing checkpoints strictly loaded into the changed models; generated tensors and latent statistics were finite.
- CPU FP32 train-mode forward/backward passed for both saved states; finite, nonzero gradients reached G and E.
- For `.`, `,`, `/`, `1`, `123`, and `handwriting`, output pixels inside the valid target were invariant to padding widths 1/11/20 within `0.001/255` max error for both checkpoints. Prior to the block-internal mask, the observed mean error for a period was 8.56/255 in DEV and 16.32/255 in alpha.
- A fresh 64px generator passed the same boundary check in both eval (max error `0.000031/255`) and training mode (`0.000457/255`).
- YAML comparison verified that synchronized GAN config files differ only in `num_critic_train` (4 versus 2).
- No full GAN training or FID/KID/HWD validation was run. The background-quality and cross-content robustness improvements remain hypotheses until the changed models are retrained and evaluated.

## Why critic ratio 2 can correlate with the RC failure

The training loop updates D/P every outer iteration and updates G/E every `num_critic_train` iterations. A ratio of 2 therefore gives G/E twice as many update opportunities per critic update as ratio 4. In the audited files, DEV had 174,825 G/E updates and 699,300 D/P updates; RC had 303,863 G/E and 607,725 D/P updates. RC's generator received about 1.74× more updates overall and its G-to-critic update ratio doubled.

That can let G/E optimize against weaker or less frequently updated critics and can reward shortcuts such as repeating a character-shaped texture when style evidence is poor. It is a plausible contributor, not a proven sole cause: the saved checkpoints did not share the same initialization/resume point, so ratio, optimizer history, and prior training path are confounded. The synchronized worktrees now set up the controlled test: same source/config/data seed and initialization, with only ratio 4 versus 2 changed. Compare at matched G/E update counts as well as matched epochs to answer this causally.
