# 32px late-training adjustment — 15 September 2026

Dataset, sample labels, evaluation protocol, generator architecture, critic
architecture, LR schedule, GAN EMA and pretrained teachers are unchanged.

## Evidence and limits

- The recorded 32px KID improves to epoch 54 and deteriorates later while FID
  continues improving. This establishes the symptom, not its unique cause.
- A 64-reference CPU gradient probe found a median weighted writer/visual-critic
  image-gradient norm ratio of 0.246. It does not show general writer dominance.
  There are strong outliers (maximum ratio 60.19). This probe uses eval-mode
  online networks, four fixed target words, and omits critic augmentation/masking
  and other losses; it is not a full-training gradient attribution.
- A paired GPU texture-screening test on 1,024 references found slightly worse
  small-sample KID when texture was halved or disabled. Original/half/off scores
  were 0.091252/0.092126/0.094543. These use paired 512-sample subsets and float64
  arithmetic and are NOT replacement benchmark scores. Texture was not changed.

## Implemented mitigation

In MAIN and DEV, `gan_iam_32.yml` now keeps writer weight 0.5 through epoch 48,
linearly reduces it to 0.25 at epoch 60, and holds 0.25 afterwards. This reduces
late writer-gradient outliers while retaining writer supervision, without adding
a loss or reweighting/removing dataset records.

The production GAN loops use the existing `linear_epoch_weight` helper. Missing
writer-schedule fields preserve the prior constant weight, including all current
64px configs. Weighted `g_writer` logging automatically reflects the schedule.
Setting `lambda_wid_final` equal to `lambda_wid` disables the taper.

This is a conservative, testable treatment of one plausible source of late
pressure, not a proven fix for KID or a claim that writer loss caused all drift.
Full training and benchmark evaluation are still required.

## Verification and continuation

Focused tests passed for both branches: coefficient boundaries, monotonicity,
absolute-epoch resume consistency, unchanged 64px/default behavior, opting out,
and exact halving of the writer gradient without changing an auxiliary gradient.
Both edited model files parse, and both Git whitespace checks passed.

Start a controlled fresh run with the same teachers, or resume the same run from
a saved checkpoint near the onset of drift if available. An already-completed
epoch-84 checkpoint cannot undo historical drift by changing a configuration.
Resuming it requires extending the configured epoch limit and is a separate
recovery experiment, not an equivalent test of the epoch-48 taper. No epoch limit
or resume path was changed automatically.

Keep baseline sampling/evaluation settings fixed and inspect rolling KID together
with FID/HWD. No full GAN training or full benchmark was run in this change.

Scripts/results in this folder are Git-ignored. Existing checkpoint files were
read only, and no current pretrained export was overwritten.
