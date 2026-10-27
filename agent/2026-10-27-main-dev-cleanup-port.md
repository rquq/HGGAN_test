# MAIN/DEV shared cleanup port and existing-checkpoint continuity

Date: 2026-10-27, Asia/Saigon (UTC+07:00).

## Approved scope

The user approved porting the reviewed ALPHA cleanup into MAIN, then DEV, and
committing both. MAIN started clean at `3fb207ce...`; DEV started clean at
`d7b997d...`. ALPHA's existing 14-file dirty refactor is source material only.
Its Git history and uncommitted state must not be changed.

Port the shared 14-file cleanup, retaining MAIN gain-only modulation and DEV's
additive SpatialStyleRefinement/out_proj. Do not copy fusion.py or branch Git
metadata. Preserve all parameter/buffer keys, shapes and optimizer ordering,
current active losses/weights, FP32, LR/EMA/evaluation schedules, data and teachers.
Only unused YAML options are removed. Commit locally; pushing is not requested.

## Checkpoint acceptance boundary

No direct access to the active Kaggle session/checkpoint files is available here.
Prove current architecture compatibility and run disposable CPU old-source save /
new-source load / continuation tests instead of claiming those remote files were
opened. Use each branch's own pre-cleanup HEAD as the baseline source. Keep true
evaluation results distinct from synthetic metadata used to test CSV restoration.

No fresh GAN/teacher training should be required solely for cleanup. Floating-point
batch-grouping changes can alter subsequent numerical trajectories; CUDA speed,
VRAM and final quality still need real-run observation.

## Verification ledger

- [x] MAIN pre-port cleanup tests fail as expected; reviewed ALPHA diff applied to
  exactly 14 tracked MAIN files, byte-identical to ALPHA afterward.
- [x] DEV pre-port cleanup tests fail as expected, then shared cleanup applied
  without modifying its additive fusion or checkpoint keys.
- [x] Current contracts, strict compatibility, 32px/64px native CPU smokes and
  pre-cleanup-source checkpoint save/new-source resume checks pass for both.
- [x] Independent review, scoped commits and final clean working-tree checks.
- [x] Record exact commits and limitations; leave ALPHA/CLASSIC/checkpoints alone.

Related: [ALPHA implementation](2026-10-27-alpha-cleanup-implementation.md),
[MAIN-only initial port](2026-10-27-main-cleanup-port.md).

## Implemented changes

Each product checkout has the same 14-file diff: 275 insertions and 1,432
deletions. Unused blocks/utilities and stale CTC-decoder options were removed;
duplicated B/R CNN construction and training-loop helpers were consolidated.
The builder creates separate B and R weights, not shared parameters.
Generated random and transferred images share one frozen-backbone/global-style
encoding call. Width-mask work is cached locally within a block forward;
valid-region semantics and gradients remain intact. Unsupported cross-replica
normalization now raises an explicit error instead of referencing missing SyncBN.

Neither fusion.py changed. MAIN retains gain-only gain_proj modulation; DEV
retains additive out_proj retrieval. No active loss, precision, dataset, teacher,
LR, critic ratio, parameter dimension or optimizer group changed. Semantic YAML
comparison found only the 13 expected unused subtree/key removals per branch;
all remaining values match each branch's pre-cleanup HEAD. All 30 tracked Python
files per branch parse successfully, and git diff --check passes.

## Verification results

- MAIN current targeted suite: **53/53 passed**.
- DEV current targeted suite: **47/47 passed**, explicitly importing DEV rather
  than accidentally exercising MAIN through a reused helper.
- Both branches passed native IAM **32px and 64px** CPU training and continuation
  checks, using real strict-loaded frozen B/R/W teachers, lazy R1, two critic
  batches and a generator/EMA update before and after checkpoint reload.
- The source checkpoint was saved by each branch's pre-cleanup HEAD model.py
  with its old YAML, then resumed by its cleaned model.py. The neural classes
  used by the fixture were the current classes with separately verified unchanged
  state layouts, rather than a complete isolated checkout of every old module.
- All seven raw network states, two EMA network states and three Adam states
  matched exactly immediately after restore. Continuation finished epoch 2;
  EMA steps advanced from 1 to 2, G Adam steps to 2, D/P steps to 4, and all
  three schedulers advanced to epoch 2. Teachers remained frozen/unchanged.
- RNG metadata was present. Embedded evaluation history and the exported CSV
  survived reload. The history row used a **synthetic schema fixture**, not an
  actual quality measurement. Sampling and guided/random generation passed;
  independent OCR used stub predictions to check wiring, not CER/WER quality.
- No existing checkpoint or teacher file was modified. Disposable checkpoint
  fixtures were removed with their temporary directories. Tests/logs stay in
  ignored agent/tests and are excluded from product commits.

The historical ignored test folders are not all green: MAIN discovery ran 71
tests with 6 failure cases and 31 error cases; DEV ran 20 with 1 failure and 8
errors. They include retired local-slot constructors/APIs, old source-string and
KL expectations, and an obsolete LR experiment expectation. DEV also has a stale
test import. These are reported separately from the current contracts above;
this task does not rewrite historical tests merely to obtain a green discovery.

## Resume instructions and limits

Update code between Kaggle sessions, then resume the existing same-branch,
same-resolution checkpoint normally. **No fresh GAN or teacher pretraining is
required solely for this cleanup.** Do not hot-reload running processes. Arbitrary
older architectures are not made compatible by this port.

The combined encoder call is mathematically equivalent for the current frozen
backbone/evaluation mode and per-sample style normalization, but FP32 kernel
batching can alter rounding and subsequent Adam/GAN trajectories. Prior paired
probes measured encoder discrepancies around 1e-7 and post-update mean image
differences around 2–3e-5 in [0,1]. This is not a bitwise-reproducibility claim.
Larger combined convolution workspaces can change CUDA peak memory; no T4 speed,
VRAM or final FID/KID/HWD guarantee is established by these CPU checks. The actual
currently running remote Kaggle checkpoint files were not directly inspected.

## Review and local commits

Independent read-only review found no blocking/actionable defect and independently
reran the four focused safety tests on each branch: MAIN 4/4 and DEV 4/4 passed.
The reviewer confirmed unchanged optimizer construction, scheduler restore,
epoch loss weights and branch-specific fusion. The full resume suite was inspected
by the reviewer, not independently rerun.

- MAIN: `7537a50` — `refactor: simplify training without breaking resume`.
- DEV: `a4a7eb6` — `refactor: simplify training without breaking resume`.

Each commit contains only the 14 scoped product files. ALPHA's 14 dirty source
files still match their recorded pre-port hashes. CLASSIC remains untouched.
No checkpoint, generated test artifact or training log was committed. No push,
force-push, reset or running-session mutation was performed. Documentation is
retained separately in the local agent documentation branch.
