# Port MAIN's plumbing fixes to DEV — 2026-11-08

Requested: apply the MAIN fixes from `e454c59` to DEV, without copying the
MAIN experiment. Starting DEV commit: `c9c17fd` (clean tracked worktree).

Published DEV commit: `e30f7c66b1c24c8cc56fdccb0fdba8c7d8c208ce`.
Normal push completed and a fresh remote query matched local HEAD.

## Changes

The four-file port matches MAIN's six runtime/API corrections:

1. Accumulate GAN loss/sample/evaluation W&B payloads at a shared iteration
   and flush once, avoiding dropped entries from repeated committed steps.
2. Flush only when a logging event is due; no empty history rows between
   logging intervals.
3. Honor `wandb.enabled: false` before initialization/authentication.
4. Publish terminal notebook job status after W&B finish/upload returns.
5. Load requested checkpoint modules by name or existing object selectors;
   apply the same selection to EMA modules.
6. Export current `G`/`E`/`B` checkpoint keys, prefer `_EMA` states when
   requested, and reject missing required modules instead of writing an
   incomplete deployment checkpoint.

Changed files: `deploy_checkpoint.py`, `lib/utils.py`, `networks/model.py`,
`train.py`. No configs, network definitions, learned parameter shapes,
training losses, teacher weights, memory conditioning or metric estimators
were modified. DEV still uses its own memory path/contextual objective and
FID checkpoint selection; MAIN's KID-first redesign was not copied.

## Verification

- Six applicable regression tests failed before patching and pass after
  patching (12 discovered; six branch-specific tests skipped). Tests cover
  same-step payload retention, no empty rows, disabled W&B, shutdown status
  ordering, named partial loading and current-key/EMA deployment export.
- Actual CPU continuation at 32px and 64px passed. The helper executes the
  pre-patch HEAD trainer, saves a checkpoint, constructs the patched trainer,
  checks exact restored raw/EMA/Adam states, then continues through epoch 2.
  Each resolution finished with G optimizer step 2, D/P step 4, EMA step 2,
  scheduler epoch 2, preserved CSV history, finite memory-projection
  gradients, sample PNG output and frozen teacher weights.
- Changed Python sources and DEV YAML configs parse; whitespace checks pass.
- Full legacy discovery is not green: 20 tests, two failures and nine errors
  including subtests. The failing-name inventory is below. Several failures
  concern old helper names or architectural assertions. They were not fixed
  by changing DEV's model, which is outside this plumbing-only port.

Generated test helpers and evidence logs remain excluded from Git. Logs:
`agent/analysis/2026-11-08-bug-audit/hggan-dev-port-*.log`.

## Compatibility and limits

No fresh GAN training or teacher pretraining is required. Refresh the DEV
checkout before resuming. Already running processes do not acquire a push
automatically. Existing missing W&B events cannot be recovered retroactively.

CPU fixture/continuation tests do not establish Kaggle/T4 fit, live W&B UI
behavior, full-dataset metric scores or generation-quality gains. No full
GPU training or full pretraining job was run.

## Legacy discovery failure inventory

- `ERROR: test_background_consistency (unittest.loader._FailedTest)`
- `ERROR: test_no_texture_head_both_heights (test_critic_experiment.CriticExperiment) (height=32)`
- `ERROR: test_no_texture_head_both_heights (test_critic_experiment.CriticExperiment) (height=64)`
- `ERROR: test_full_fp32_training_paths_and_new_checkpoint_roundtrip_both_heights (test_generalization.LearnedEvidence)`
- `ERROR: test_reference_modulation_uses_context_with_unbounded_learned_strength (test_generalization.LearnedEvidence)`
- `ERROR: test_weak_local_evidence_can_fall_back_to_global_without_floor (test_generalization.LearnedEvidence)`
- `ERROR: test_evidence_gate_initializes_at_configured_probability (test_luna_review.LunaReview)`
- `ERROR: test_optional_attention_padding_invariance_both_heights (test_luna_review.LunaReview) (height=32)`
- `ERROR: test_optional_attention_padding_invariance_both_heights (test_luna_review.LunaReview) (height=64)`
- `FAIL: test_geometry_sampler_and_readout_ignore_word_labels (test_generalization.LearnedEvidence)`
- `FAIL: test_global_kl_preserves_previous_global_contribution (test_luna_review.LunaReview)`
