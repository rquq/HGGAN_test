# Shared memory width and objective balance implementation plan

> Execution: native, as delegated by the user; one independent review before publishing.

Date: 2026-10-24 (Asia/Bangkok, UTC+07:00).

## Design and scope

Use a learned 128-to-64 channel memory projection in DEV and MAIN, retaining the
32D global latent, four rows / maximum 32 memory columns, and 128-wide four-head
attention. Keep DEV additive retrieval and MAIN gain-only modulation as the only
architectural difference. Use writer weight 0.25 and transfer-CTC decay 0.5 to 0.3
over epochs 8–28. All other objectives, teachers, data, metrics and schedules stay
unchanged. This is a prospective experiment, not a proven quality improvement.

Basis: [capacity/gradient audit](analysis/2026-10-24-dev-memory-capacity/REPORT.md)
and [modulation design](2026-10-24-main-gain-only-modulation.md).

## Files and interfaces

- Both `networks/{module,fusion,BigGAN_networks}.py`: matching 64-channel defaults.
- Both `configs/gan_iam_{32,64,local}.yml`: matching encoder/G widths and loss balance.
- Local ignored `agent/tests`: behavioral contracts and two-height real-loop smoke.
- Dated shared report and `agent/INDEX.md`: design, verification, limits, retraining.
- Commit product changes separately on DEV and MAIN; publish docs on agent branch.

## Review focus

1. Config/default disagreement: build default and configured encoder/G together.
2. Width confusion: memory has 64 channels; attention still has 128 and four heads.
3. Empty/mixed reference rows: exact random bypass and finite guided gradients.
4. Old GAN state: strict 128D state fails instead of partial silent migration.
5. Fair experiment: shared configuration and teachers; only add versus gain differs.

## Tasks and ledger

- [x] Add tests for default/configured 64D memory generation and gradients at both
  heights; real loss schedule values; legacy-state rejection. Run RED first.
- [x] Patch matching defaults and three GAN configs in each branch. Run GREEN plus
  existing current-architecture contracts and bounded real training loops.
- [x] Inspect the existing full test suite and report obsolete failures honestly.
- [x] Independent read-only review, fix actionable findings, fresh verification.
- [x] Document and publish scoped DEV/MAIN commits and agent documentation; verify
  remote hashes and clean product working trees. Do not touch ALPHA or dirty master.

Ruling: no more design-approval round — the user explicitly delegated selecting
and applying changes, including commit/push. Costs if wrong: a new controlled GAN
run may regress; no existing checkpoint or training result is overwritten.

Ruling: retain full attention width despite narrower memory — activation variation
is compressible, but all heads and low-variance gradient directions are active.
No channel pruning, PCA deployed weights, mean subtraction, teacher changes,
special-character routing or new loss is introduced.

Ruling: tests stay in ignored `agent/tests` per the user's clean-repository rule;
verification results and test descriptions are published in dated documentation.

Verification: DEV 16/16 current contracts, MAIN 49/49 current contracts; four real
CPU native-height training-loop smokes passed. Full historical discovery remains
red (DEV20 with1failure/8errors; MAIN71 with6failures/31errors), due to retired
interfaces/settings, explicitly listed in the companion report. Ruling: do not
restore retired model behavior or alter ignored historical tests to manufacture
a green full-discovery claim. Publish only on the supported current-contract and
real-loop evidence, with this limitation stated.

Independent review: no Critical/Important findings, 38 independently rerun
contracts passed. Minor old-document dimension/count ambiguity fixed by adding
supersession notices rather than rewriting the 128D checkpoint audit.

Product publication verified: DEV `d7b997d22e1d0390ba771dde41a33ff36fcda341`, MAIN
`3fb207ce4e3fd2d16df046836cec192ab78137df`, both non-force pushes. Documentation
is published separately to agent in the commit containing this final ledger.

Final review rulings: quality metrics/long-run stability and CUDA/Kaggle speed
remain unmeasured; old GAN compatibility is intentionally rejected; the known
black/empty-reference outlier is not claimed fixed. Costs: fresh controlled GAN
runs are still required and may regress, despite passing CPU wiring tests.
