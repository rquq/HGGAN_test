# ALPHA metric and W&B alignment — 2026-11-08

## Findings

ALPHA's FID/KID/HWD/Inception/IS and PSNR/MSSIM code is AST-identical to
MAIN's. The CMMD module differs in comments/temporary variable names, not
the preprocessing operations. No metric formulas were replaced.

ALPHA's native 32px and 64px configs disabled HWD and selected
`valid.dset_name: iam_word_org`, whereas MAIN selects `iam_word`. Actual
data equivalence depends on the notebook-created file links; an alias
difference alone does not prove different dataset contents.

ALPHA used automatic W&B log steps, so the displayed event counter could
be 2763 at epoch 52. A W&B event step is neither an epoch nor necessarily
an optimizer iteration. MAIN explicitly supplies iteration steps, but its
current samples are also iteration-based; blindly copying MAIN would not
provide the requested epoch axis for sample panels.

## Implemented in ALPHA only

- Matched MAIN's validation dataset selector, metric toggles, evaluation
  seed, repeat count, batch sizes, Inception dimensions and KID estimator
  settings for 32px, 64px, local and the legacy default IAM config (the
  default uses MAIN's 64px config as reference).
- Enabled HWD in all four IAM GAN configs. Training datasets and model
  architecture/losses/checkpoint selection are unchanged.
- Samples use `samples/epoch` as their custom W&B x-axis, with fractional
  epoch and iteration in the caption. Local image filenames retain
  `iter_<iteration>.png` for compatibility.
- GAN logs explicitly carry the training iteration. Loss/sample/eval
  events accumulate with `commit=False`, then flush once per iteration,
  preventing same-step entries from being dropped.
- Validation uses fractional epochs on `valid/epoch`, matching MAIN.
- Pretraining logging remains untouched.

## Verification and limitations

Passed YAML comparison of all four IAM validation configs against MAIN,
Python compilation, AST equality for the primary metric modules, and an
execution test of the actual logging AST with a W&B stand-in. The test
asserts that loss/sample/validation entries sharing an iteration survive
in one row and that sample captions/custom axis use the epoch.

No GPU evaluation, full training, or live W&B dashboard test was run.
Existing W&B history cannot be rewritten by this patch; existing custom
panels may need their x-axis switched to `samples/epoch`. A resumed/new
process must load the updated checkout. No GAN or teacher retraining is
required, but changing which validation files an alias resolves to can
make old and new metric series incomparable; retain dataset provenance.
