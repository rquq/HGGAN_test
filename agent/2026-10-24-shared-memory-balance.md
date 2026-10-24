# DEV and MAIN: compact reference memory and readability balance

Date: 2026-10-24, Asia/Bangkok (UTC+07:00).

## Decision

Both branches now use a **64-channel spatial reference memory**, retaining the
32D global latent and 128-wide, four-head cross-attention. Writer supervision is
reduced moderately and late transfer-CTC supervision is retained more strongly.
These are prospective experimental settings, not measured FID/KID/HWD gains.

MAIN includes the previously implemented shared-backbone synchronization and
gain-only experiment. DEV remains additive. ALPHA is untouched.

## Exact common settings

| Setting | Before this shared change | Now, both branches |
|---|---:|---:|
| Global latent | 32 | 32 |
| B intermediate source channels | 128 | 128 |
| EncModel.memory_dim | 128 | 64 |
| GenModel.reference_memory_dim | 128 | 64 |
| Reference grid | 4 rows, at most 32 columns | unchanged |
| Attention width / heads | 128 / 4 | unchanged |
| Writer loss coefficient | 0.5 | 0.25 |
| Transfer-CTC initial coefficient | 0.5 | 0.5 |
| Transfer-CTC final coefficient | 0.2 | 0.3 |
| Transfer-CTC decay epochs | 8–28 | unchanged |

The three IAM GAN YAMLs (`gan_iam_32`, `gan_iam_64`, `gan_iam_local`) are updated
in each branch. Encoder, generator and retrieval-class defaults also agree on
64 channels, so the custom-image config inherits a consistent width. All nine
MAIN/DEV YAMLs are semantically identical. Normal GAN batch size remains 8;
local settings remain local settings. Critic ratio stays 2, G/D/P LR stays 2e-4,
and the existing 80-epoch / decay-after-36 schedule is unchanged.

No loss is added or deleted. Random-CTC, reconstruction, contextual, global
style cycle/information/content-adversarial/KL and adversarial settings stay
as they were. Frozen W+B/R, dataset/polarity, augmentations, reference sampling,
FID/KID/HWD and independent CER/WER protocols are unchanged. No precision change
or additional W&B diagnostic series is introduced.

## Why these choices

### Memory width: compress the representation, not the attention mechanism

The [epoch-34 audit](analysis/2026-10-24-dev-memory-capacity/REPORT.md) found a
held-out memory effective covariance rank of 18.96, with approximately 30
principal directions explaining 99% of pooled variation. It did **not** find
dead channels or identical heads. All channels received finite fresh gradients,
and low-variance directions retained loss sensitivity. Therefore 16D/32D memory
or removing attention would be more aggressive than the evidence supports.

64D is the conservative first trained compression experiment. It is a fresh,
learned 128-to-64 projection, not selection of 64 named old channels, a deployed
PCA basis or a guarantee of retaining a given fraction of trained information.
The functional PCA probe preserved the original 128D reference mean and all
original attention weights, so its small image error cannot be claimed as the
performance of this new architecture.

Keeping attention at 128 retains four 32D heads and their target-position
selection capacity. Only the source memory and K/V input widths shrink.

### Objective balance: reduce pressure, do not delete writer learning

In six selected two-image diagnostic batches, the writer objective produced the
largest mean weighted gradient norm on the memory projection: 11.71, versus
1.91 for adversarial and 0.85 for transfer CTC. This is limited sample evidence,
not proof that writer supervision dominates every historical step.

Halving its coefficient from 0.5 to 0.25 reduces its direct contribution for a
fixed forward pass without discarding writer identity learning. Raising the
late transfer-CTC coefficient from 0.2 to 0.3 retains more readability pressure
while keeping it below its initial 0.5. At epoch 18 its coefficient is 0.4;
from epoch 28 it is 0.3. Neither setting is established as optimal. The OCR
teacher is still frozen and no new teacher or text-dependent routing is used.

## Shared dataflow and controlled difference

```text
Reference -> shared frozen B
             | deep sequence -> context + pooling -> global32 -> GBlocks
             | intermediate128 map -> valid-region pool -> 1x1 projection64
             |                                      -> LN -> memory64
             |                                                   |
Text -> content context -> spatial seed -> GBlock1 -> attention128/4 heads
                                                    Q from GBlock1
                                                    K,V from memory64
                                                         |
                    DEV: H + scale * output(attended) -----+
                   MAIN: H * (1 + scale * gain(attended)) --+
                                                         |
                                                   GBlocks2–4 -> image
```

Random-style generation has no reference and bypasses attention. Style transfer
and reconstruction use the actual reference memory. Validity masks, valid-width
pooling and output padding contracts remain in place; no all-masked softmax is
run. There is no new high-frequency encoder, learned slot bank, normalization
cap, confidence heuristic, character-ID check or special-case fallback.

The only computational architectural difference is the additive versus
multiplicative handoff. Class/projection names differ to prevent silent loading
of the wrong branch's GAN weights; two existing error messages also differ in
wording. `networks/model.py`, `metric/ocr_eval.py` and all configurations match.

## Size, speed and retraining

| Native height | G parameters | E parameters | G+E |
|---|---:|---:|---:|
| 32px | 3,950,685 | 953,521 | 4,904,206 |
| 64px | 3,954,845 | 953,521 | 4,908,366 |

Both branches have identical parameter counts at the same height. Relative to
the pre-change 128D shared-memory version, G+E saves 24,576 parameters, about
0.50%. The stored spatial-memory activation width halves. Attention's 128D
Q/K/V outputs and attention matrix size do not shrink, and CNN cost is largely
unchanged. No major overall throughput gain or T4 memory saving is claimed.

**Start a fresh GAN run in each branch. Existing W+B and OCR pretrains are
reusable, and were not modified.** Existing 128D-memory GAN states fail strict
loading because projection shapes changed. No partial-loader compatibility
shim or automatic checkpoint conversion is added. The supplied checkpoints and
their embedded historical CSVs remain intact. Do not compare an old continued
DEV run against fresh MAIN as a controlled experiment.

## Verification performed

CPU FP32, CUDA hidden, two Torch threads. No laptop GPU allocation or new full
metric evaluation was performed.

- Test-first RED: new contracts failed on the old 128D defaults/configuration,
  old loss schedule and silent same-shape legacy acceptance.
- GREEN: DEV current memory/balance suite **16/16**, MAIN current suite **49/49**.
  MAIN includes the 27 current background/checkpoint/patch regressions and six
  gain-only formula contracts in addition to the shared memory tests.
- Both native heights and both branches ran the real GAN loop on two bounded
  native-data critic batches and one G/EMA update. Real W+B/R weights loaded
  strictly and stayed frozen; R1 second-order backward ran; actual sample PNGs
  were written; guided/random generation and independent OCR generation wiring
  passed. OCR predictions in this interface smoke are stubbed, not quality scores.
- Finite memory-projection gradient norms in the smoke: DEV 16.77/21.59 and
  MAIN 10.10/16.01 at 32px/64px. These magnitudes are runtime checks, not evidence
  of which branch will train better.
- Thirty tracked Python files AST-parsed in each branch. All nine paired YAMLs
  agree semantically. `git diff --check` passes. All five paired teacher-file
  SHA-256 hashes agree; none is staged or changed.
- Independent read-only review found no Critical/Important issue and reran 38
  contracts successfully (six shared and ten reference tests per branch, plus
  six MAIN modulation tests). Its minor historical-document ambiguity was
  resolved with explicit supersession notices.

### Full historical discovery is not green

The ignored local test folders contain obsolete architecture contracts. They
were run, not hidden: MAIN discovery has 71 tests, six failure cases and 31 error
cases; DEV discovery has 20 tests, one failure and eight error cases. No ignored
historical script was deleted, committed or changed to make this pass.

MAIN affected names/groups:

- `test_compact_style_context.test_encoder_local_visual_evidence_and_global_only_sampling`
  expects retired local-slot structure (two height cases).
- `test_critic_experiment.test_experiment_settings` retains an earlier LR
  expectation; `test_no_texture_head_both_heights` passes an old eight-token input.
- `test_global_only_integration` targets retired sampling/centering interfaces.
- `test_luna_review.test_local_evidence_has_no_gate_bias_or_checkpoint_initialization_shim`,
  `test_optional_attention_padding_invariance_both_heights` and
  `test_global_kl_preserves_previous_global_contribution` target old options/input
  shapes or an old source-text assertion.
- `test_reference_local_evidence` and `test_spatial_style_fusion` target retired
  local-slot/refinement interfaces; empty old local slices also produce NaN in
  a test's own diagnostic, not in current model output.

DEV affected names/groups:

- `test_background_consistency` imports removed `MultiScaleStyleContext`.
- `test_critic_experiment.test_no_texture_head_both_heights` uses eight tokens.
- `test_generalization.test_reference_modulation_uses_context_with_unbounded_learned_strength`
  and `test_weak_local_evidence_can_fall_back_to_global_without_floor` expect
  retired local modules.
- `test_luna_review.test_evidence_gate_initializes_at_configured_probability`,
  `test_optional_attention_padding_invariance_both_heights` and
  `test_global_kl_preserves_previous_global_contribution` expect old modules,
  token shapes or source text.

An initial attempt to reuse the MAIN regression module names directly under DEV
also exposed absent DEV copies of two helpers and the obsolete background import.
DEV's current suite was then run separately, using its actual current contracts.
This is not an all-historical-tests-green claim or a guarantee of a crash-free
full Kaggle/CUDA session. Runtime helper scripts, raw logs, temporary model outputs
and generated samples remain ignored under `agent/tests`.

## What the next comparison can establish

Run fresh DEV and MAIN with the same resolution, teachers, seed, batch size and
protocol. Compare at matched epochs, retaining both FID/KID/HWD and independent
CER/WER plus the fixed short/long-reference examples. Only their handoff differs.
Because width and objective balance change together in this pair, an improvement
over the old DEV checkpoint would not isolate which shared change caused it.
The old checkpoint remains useful as a historical baseline, not a controlled arm.

The black/empty-reference global-vector outlier is not guaranteed fixed by these
settings; no evidence-free image threshold or global-norm clamp is added. Memory
compression and gain-only modulation can still regress allograph/style fidelity.
Full training is needed to decide whether the choices help. No FID target is
promised.

## Publication

Independent review completed with no Critical/Important findings. Product commits
were pushed without force and verified against remote branch hashes:

- DEV: [`d7b997d22e1d0390ba771dde41a33ff36fcda341`](https://github.com/rquq/HGGAN_test/commit/d7b997d22e1d0390ba771dde41a33ff36fcda341)
- MAIN: [`3fb207ce4e3fd2d16df046836cec192ab78137df`](https://github.com/rquq/HGGAN_test/commit/3fb207ce4e3fd2d16df046836cec192ab78137df)

Both product working trees are clean. ALPHA remains at
`90225a2b1ba5380104729053814454b291ae24b0`. Documentation is delivered on the
shared `agent` branch, separate from product code; this report and its companion
audit data are included in that documentation commit. Test/runtime files and
generated samples are not included. The unrelated dirty outer master checkout
is not staged, reset, merged or committed.
