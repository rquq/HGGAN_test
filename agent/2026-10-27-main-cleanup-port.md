# Port the reviewed ALPHA cleanup into MAIN

Date: 2026-10-27, Asia/Saigon (UTC+07:00).

Historical initial plan: the user subsequently extended scope to DEV and
requested local commits. The completed scope/results are recorded in
[MAIN/DEV cleanup and checkpoint continuity](2026-10-27-main-dev-cleanup-port.md).
Statements below about leaving DEV untouched or not committing describe only
the original MAIN-only request, not the final action.

## Approved scope

Apply the current reviewed ALPHA-versus-MAIN cleanup to MAIN. Both start from
`3fb207ce4e3fd2d16df046836cec192ab78137df`. MAIN is clean before this task; ALPHA
contains 14 user-owned uncommitted cleanup/refactor files, including the follow-up
corrections. Port only those tracked file contents, never .git, checkpoints,
teachers, datasets, runs, notebooks or ignored tests. ALPHA and DEV stay unchanged.

Source: [ALPHA correction report](2026-10-27-alpha-cleanup-implementation.md) and
[initial refactor review](2026-10-27-alpha-refactor-review.md).

## Behavior boundary

Retain MAIN's gain-only architecture, all parameter/buffer names/shapes and
optimizer ordering, 64D memory, global32, existing active loss formulas/weights,
FP32, evaluation and learning-rate schedules. Only unused YAML options disappear;
no remaining value changes. The unsupported cross_replica=True path gets an
explicit error instead of a missing SyncBN reference. Current false defaults
remain supported. Combined generated-image encoding is checkpoint-compatible
but can change FP32 rounding, Adam updates and larger convolution workspaces.

No fresh GAN/teacher training is required solely for this port. Do not hot-reload
an active job; update between sessions and resume a compatible MAIN checkpoint.
No commit, push, merge, reset or history rewrite is requested in this task.

## Plan and verification ledger

- [ ] Run the shared cleanup tests on pre-port MAIN and confirm expected RED.
- [ ] Apply only ALPHA's exact 14 tracked file diffs to MAIN using apply_patch.
- [ ] Verify target/source byte equality, unchanged active YAML values and strict
  state compatibility; run the current contracts and native 32px/64px CPU smokes.
- [ ] Compare matched MAIN/ALPHA update fingerprints under the same batch grouping;
  report historical test failures separately and verify ALPHA/DEV untouched.
- [ ] Record final results, limitations and retraining requirements here.
