# Review of the September 30 fixes

Reviewed DEV commit `536cb873f0158c699de98f52bfdd6e54360c76d9` and the synchronized alpha implementation. Four narrow corrections were applied to both worktrees; no architecture parameters, pretrained weights, metric definitions, or active hyperparameters were added or changed.

## Corrections

1. **Unintended KL rescaling.** Switching from an all-token mean to a global-token-only mean multiplied the global token's KL contribution by eight with the current eight-token encoder. The loss now excludes local tokens but divides the global term by the token count, preserving its previous contribution. This is not a claim that the old total KL remains identical: the local terms are intentionally removed.
2. **Incorrect gate initialization.** The new `0.5 + 0.5 * sigmoid(logit)` mapping made a configured initial reliability of 0.75 become 0.875. Initialization now uses the inverse of that mapping and validates its achievable range. This changes fresh initialization, not the values already learned in a loaded checkpoint. The lower bound does not guarantee that the gate avoids saturation near one or that local tokens learn diverse evidence.
3. **Optional spatial attention ignored valid lengths.** Both the ignored plural argument and missing key mask were corrected. Invalid pixels are excluded from attention and have zero output/gradient. D now passes post-downsampling lengths to attention. Spatial attention remains disabled in the active configs, so this correction does not add attention to the current model.
4. **Standalone GBlock length fallback.** When an upsampling block received input lengths without explicit output lengths, it kept the original lengths and clipped the upsampled valid region. It now derives output lengths from the actual width expansion. The normal generator already supplies explicit lengths; this was a latent helper-path defect.

## Verification

- All 17 distinct regression checks passed, including existing background/critic checks and six review checks. The complete 17-test suite passed in alpha; DEV's existing checks and added checks also passed.
- Masked BatchNorm matches native BatchNorm outputs and running statistics for fully valid inputs.
- Optional attention preserves valid pixels under added padding for G and D at both 32px and 64px; padded input gradients are zero.
- Real-data DEV training and sampling passed in FP32 on CPU and CUDA at both heights. Each bounded run used eight outer iterations, actual pretrained W+B/R, R1 every iteration, and EMA. D/P each updated eight times; G/E and EMA updated twice with ratio 4.
- alpha's 32px real-data CUDA run passed: D/P each updated eight times; G/E and EMA updated four times with ratio 2.
- CUDA peak allocated memory for the bounded batch-two tests was approximately 484 MiB (DEV 32px), 811 MiB (DEV 64px), and 483 MiB (alpha 32px). These are smoke-test measurements, not batch-eight production VRAM estimates.
- alpha had no local IAM dataset links. Only its smoke-test process was pointed at DEV's existing links; production linking remains managed by the notebooks.
- Network source is byte-identical between the two worktrees. The three GAN configs differ only in critic ratio: DEV 4, alpha 2. Git diff whitespace checks passed. Test files remain ignored; no commit or push was performed.

## Limits

No full training, FID/KID/HWD rerun, or Kaggle session was performed. These checks prove execution and targeted invariants, not better generation quality or the absence of every possible bug. The previous encoder representation change still needs fresh GAN training for a fair comparison. W+B/R do not need retraining for these corrections. Alternative normalization modes such as cross-replica/custom BN were not validated; the active configs use ordinary BN.
