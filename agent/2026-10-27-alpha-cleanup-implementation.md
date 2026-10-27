# ALPHA cleanup corrections and checkpoint-safe encoder batching

Date: 2026-10-27, Asia/Saigon (UTC+07:00).

## Scope and design

Implement the approved follow-up to the Gemini refactor review in ALPHA only.
Preserve its existing dirty refactor, all parameter/buffer names and shapes,
losses/weights, configs, precision, evaluation cadence and branch history.
MAIN, DEV, CLASSIC and all existing checkpoints/teachers stay untouched.

1. Reject unsupported cross_replica=True explicitly in both normalization classes.
2. Keep width masking, drop mathematically unnecessary clamp operations, and
   reuse the width mask within each GBlock/DBlock forward. Cache lifetime is one
   call; it retains source length tensors to avoid Python object-ID reuse.
3. Concatenate generated random/style images for one frozen-B/global-only-E call;
   split global codes and return only the transfer half of feature maps/sequences
   for existing writer/contextual supervision. Do not use no_grad or detach here.
4. Correct obsolete local-token/context docstrings.

Alternatives declined: deleting active losses changes objectives; flattening the
global tensor API or relocating demos broadens the patch; dataset aliasing needs
separate collation/pinning analysis; changing metrics removes useful monitoring.
No dependency, precision option or new model mechanism is introduced.

## Verification plan and ledger

- [x] RED: tests reject the silently ignored flag, demand mask reuse and one
  generated-image B/E call with baseline output/gradient parity.
- [x] GREEN: implement at the existing responsible layers and rerun tests.
- [x] Run the 27 current regression checks, strict state compatibility and native
  CPU 32px/64px real GAN-loop smokes. Report obsolete discovery failures separately.
- [x] Compare disposable MAIN/ALPHA optimizer updates numerically, allowing stated
  floating-point tolerances rather than promising bit-identical CUDA trajectories.
- [x] Independently review the scoped implementation, document rollout safety,
  verify MAIN/DEV remain unchanged. No commit/push/history rewrite is requested.

Ruling: implement directly under the user's explicit approval to correct the
reviewed refactor. This is not approval to erase ALPHA's dirty changes or port the
whole branch over other experiments.

Expected compatibility: no GAN/teacher retraining solely for these changes.
Batching may change floating-point rounding and peak workspace use; long-run
quality and Kaggle speed/VRAM remain empirical, not established by CPU tests.

## Implemented files

- `alpha/networks/BigGAN_layers.py`: explicit unsupported SyncBN flag errors;
  forward-local mask reuse in GBlock and DBlock. All masks remain active.
- `alpha/networks/model.py`: `_encode_generated_styles` combines random/transfer
  B+E work and splits the codes, transfer features and transfer sequence for the
  original loss formulas. Gradients remain attached through the frozen B to G.
- `alpha/networks/module.py`: context docstring describes current spatial memory.

No parameter/buffer is added, removed, renamed or reshaped by this patch. The
singleton global-style dimension, normalization divisor, all active losses and
YAML values remain unchanged. Existing Gemini changes in the other files remain
as they were; no history operation, commit or push was performed.

## Verification results

Execution: CPU FP32, two Torch threads, CUDA hidden. No GPU/T4 workload, library
installation, pretrained teacher update or full quality metric run.

- Four new tests failed before the change for the expected missing behavior,
  then passed. They verify unsupported-flag rejection, mask reuse/boundaries,
  masked NaNs/gradients, one B call and separate-versus-joined encoder output and
  input/parameter gradient agreement at 32px and 64px (atol/rtol 3e-5).
- The current regression suite passes **31/31** (27 existing + four new).
- An additional **16/16** current reference-memory/modulation contracts pass
  against ALPHA. The MAIN-named local helpers were explicitly remapped to ALPHA
  and the imported production module path was checked; these are not MAIN results
  being mislabeled as ALPHA results.
- Final combined execution of these current-contract suites passes **47/47**.
- Strict state loading between current MAIN and corrected ALPHA passes for all
  seven components, including B/R/W. Parameter counts and tensor shapes match.
- Both branches ran two native-data critic batches and one G/EMA update at both
  heights, with real frozen teachers, lazy R1, actual samples and random/guided
  generation. OCR generation wiring used a prediction stub, not quality scores.
- Existing MAIN Adam state dictionaries loaded into optimizers built around the
  corrected ALPHA parameters with preserved step counters. Scheduler/EMA progress
  schemas were not changed.
- Source/diff whitespace checks pass. MAIN/DEV remain clean at their original
  `3fb207ce...` / `d7b997d...` revisions. CLASSIC remains clean at `25ff2339...`.
- Independent read-only review found no Critical/Important defect and independently
  reran all four new tests successfully.

### Numerical trajectory is not bit-identical

The mathematically equivalent combined encoder can use different FP32 convolution
and reduction orders. A strict 3e-5 raw-parameter comparison after a real update
failed; it was investigated rather than silently relaxed and called exact parity.

Maximum absolute raw G-parameter differences after one update were 1.3748e-4 at
32px and 1.1525e-4 at 64px. Largest differences occur in convolution biases with
near-zero gradients. For example, one 32px bias's Adam exp_avg is -9.92e-10 versus
5.46e-9, with Adam epsilon 1e-8; normalization by the tiny second moment/epsilon
can amplify that small gradient perturbation. Raw E differences were 5.10e-7 and
2.42e-7. B/R/W and raw global D states remained exactly equal in this probe.

Post-update generated-image differences on the tested references/targets,
measured on [0,1]:

| Height | Raw mean absolute pixel difference | EMA mean difference | Largest pixel difference across both |
|---|---:|---:|---:|
| 32 | 2.36e-5 | 2.01e-5 | 6.11e-5 |
| 64 | 2.64e-5 | 2.80e-5 | 6.47e-5 |

These are small functional differences in a bounded one-step smoke, not proof of
identical long-run GAN trajectories or identical final metrics. Checkpoint schema
compatibility does not imply bitwise reproducibility after resuming with the
optimized batch grouping.

### Performance boundary

The generated-image path calls B once for 2B images instead of twice for B images;
the same total images and neural arithmetic are processed. Tests verify actual
mask reuse rather than fewer source lines. No T4 speedup percentage is claimed.
In the configured 32px global-D forward, actual arange calls fall from MAIN's 17
to corrected ALPHA's 8; both now have one clamp call, instead of the 17 clamp
operations found in Gemini's uncorrected shared-mask version.

The returned transfer slices are views into combined feature tensors, retaining
the full 2B masked feature allocation. Combined convolution workspaces may also
be larger. This is a bounded memory tradeoff, not a detach/autograd defect; measure
peak VRAM and throughput at native batch8 on Kaggle before a broad deployment.
No extra toggle or unmeasured automatic fallback was added.

### Historical discovery remains non-green

The ignored historical suite ran **71 tests: five failure cases and 31 error
cases**. Failures are old local-slot assertions in
`test_compact_style_context.test_encoder_local_visual_evidence_and_global_only_sampling`
(both heights), `test_luna_review.test_global_kl_preserves_previous_global_contribution`
(old source-text contract), and
`test_reference_local_evidence.test_visual_reference_changes_centered_codes`
(both heights; old empty-local-slice diagnostic).

Errors are retired eight-token/slot options and old helper APIs in
`test_critic_experiment.test_no_texture_head_both_heights`,
`test_global_only_integration`,
`test_luna_review.test_local_evidence_has_no_gate_bias_or_checkpoint_initialization_shim`,
`test_luna_review.test_optional_attention_padding_invariance_both_heights`,
`test_reference_local_evidence` and `test_spatial_style_fusion`. Exact names and
traces are preserved in local `agent/tests/alpha_cleanup_full_discovery.log`.
They were not edited, deleted or restored into product code to manufacture a
green full-discovery claim. Current-contract and real-loop results are distinct.

## Checkpoints and repo-wide rollout

These corrections alone require **no fresh GAN or teacher pretraining** for a
compatible current architecture. Existing current MAIN/DEV checkpoint files were
not modified; the test states were fresh disposable outputs in a separate /tmp
directory, not user training checkpoints. Numerical continuation can diverge
slightly due to changed batch grouping, as documented above.
The four disposable test state files (about 516 MiB total) were removed after
their comparisons; reproducible helper scripts and numerical summaries remain
in ignored agent/tests. No user checkpoint file was deleted.

- MAIN: a scoped port can preserve current gain-only model, model keys, optimizer
  ordering, EMA state and configuration. Test a real resume after porting.
- DEV: port helpers/operations only, preserving its additive `SpatialStyleRefinement`
  and `out_proj`. Copying ALPHA's entire G/fusion tree would replace these with
  MAIN's `SpatialStyleModulation`/`gain_proj` and is NOT a checkpoint-safe cleanup.
- CLASSIC_OPTIMIZED: has a different encoder interface without the current
  `global_only`/reference-memory contract. It needs a branch-native refactor and
  parity/resume tests; do not apply this helper verbatim or change the benchmark
  architecture under the name of cleanup.

Apply future ports **between training sessions**, then resume the saved checkpoint
with optimizer/scheduler/EMA/RNG state intact. Do not hot-edit code used by an
active job or replace whole branch folders. No repo-wide port is performed here:
the user asked whether it is safe, not to execute it on every branch.

The review's minor observations (full combined feature allocations and cache
invariants) are documented rather than prompting unmeasured extra mechanisms.
Source lengths must not mutate in-place during a block forward; current call sites
do not do so. Cache objects must remain local to the forward, as implemented.
