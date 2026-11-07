# MAIN / ALPHA bug audit — 2026-11-08

## Scope and baseline

Requested: inspect MAIN and ALPHA and fix reproducible bugs. The audit
focused on training orchestration, sampling, evaluation, checkpoint APIs,
deployment export, and optional pretraining paths. It is not a proof that
every possible bug is absent.

Starting source commits: MAIN `d0b812c`, ALPHA `0532c68`. Both worktrees had
no tracked modifications; ALPHA's three existing untracked pretrained
binaries were left untouched. No DEV/CLASSIC code was changed.

## Verified source publication

- MAIN: `e454c5967b71785309ff0adb7215eb58108e5416`
- ALPHA: `84fbab2de435b4319a3ed50d9aa1dcb11662f748`

Both were normally pushed and verified against fresh remote heads.

## Confirmed bugs and fixes

| ID | Scope | Failure and correction |
| --- | --- | --- |
| M1 | MAIN | Loss, sample and validation logs could each commit the same W&B step, dropping subsequent entries. Accumulate with `commit=False` and flush once per logged iteration. |
| M2 | MAIN | `wandb.enabled: false` was ignored and authentication was still attempted. Honor the switch before importing/initializing W&B. |
| M3 | MAIN | Terminal job status was published before `wandb.finish()`, letting a notebook monitor stop the runtime while upload was pending. Publish final status after finish returns. |
| B1 | Both | `load(..., modules=['G'])` compared strings against module objects and silently loaded nothing. Match module names as well as existing object selectors; MAIN's EMA loop also respects the selection. |
| B2 | Both | Exporters looked for legacy long module keys instead of saved `G`, `E`, `B`, and MAIN used the wrong EMA-key convention. Use current keys and `G_EMA`/`E_EMA`; reject incomplete exports before writing. ALPHA now has the same CLI export entrypoint instead of executing a hard-coded export at import. |
| B3 | Both | Unconditional empty W&B flushes create history rows on unlogged iterations. Flush only when loss/sample/evaluation events are due. |
| A1 | ALPHA | Sample transfer images were padded using reconstruction width. When widths differed, grid concatenation failed and sampling silently returned without a PNG. Pad from the transfer image's actual width. |
| A2 | ALPHA | Fractional-epoch validation logged correctly to W&B but CSV upserts still used the integer epoch, overwriting earlier evaluations in the same epoch. Record the actual fractional evaluation epoch. |
| A3 | ALPHA | Frozen-backbone writer optimizer put `lr` into the optimizer dictionary instead of Adam, leaving the wrong LR and an invalid scalar optimizer entry. Pass LR to Adam. |
| A4 | ALPHA | Random-corpus evaluation passed capitalization ratio as maximum word length, causing truncation or float-slice exceptions. Bind the arguments correctly and move generated content labels/lengths to the model device before G. |
| A5 | ALPHA | Encoding a one-item text batch returned a flat single-word list, breaking the final batch in random-corpus evaluation. Preserve the batched tensor/length return for list inputs, including length one. Single-string encoding is unchanged. |
| A6 | ALPHA | Standalone test mode forced PSNR/MS-SSIM and WIER even when disabled. Respect each toggle independently. |
| A7 | ALPHA | Resume recomputed epoch solely from global iteration count using the current loader length. After a batch-size change, a saved epoch-52 checkpoint could resume at epoch 105. Keep saved `Epoch` authoritative, matching MAIN's existing principle. |

The resume fix preserves the iteration counter. It does not promise
bit-exact data order or a mathematically exact mid-epoch sample offset after
changing the dataset or batch size; that requires data-loader state the old
checkpoints do not store.

## Regression evidence

The ignored regression helper is
`agent/tests/test_bug_audit_20261108.py`. Reproductions were run before
their responsible fixes. The sample test initially had an incorrect
test-only channel count; that fixture was corrected and the genuine ALPHA
width mismatch was reproduced independently before patching.

Final focused results:

- MAIN audit: 12 discovered, 7 applicable passed, 5 ALPHA-only skipped.
- ALPHA audit: 12 discovered, 9 applicable passed, 3 MAIN-only skipped.
- MAIN KID-first contracts: 8 passed (selection, projection RNG/state,
  feature-distribution gradients, native-height forward/backward, PatchD).
- ALPHA runtime/cache/RNG tests: 3 passed, including actual metric arithmetic
  comparison against the saved pre-runtime-optimization implementation.
- Native 32px and 64px CPU training smokes passed for both branches:
  actual GAN updates, finite weights, strict frozen-teacher loading and
  checkpoint save/reload. MAIN also exercised sample PNG output, EMA/loss
  state loading and native Inception/KID on synthetic fixture records.
- All active source Python and config YAML parsed; staged whitespace checks
  passed before publication.

Full legacy discovery is **not green**:

- MAIN: 71 tests, 20 failures, 29 errors.
- ALPHA: 41 tests, 10 failures, 35 errors (including subtest errors).

Examples include tests requiring removed `GatedStyleContext`/spatial memory,
old multi-token generator inputs, or best-FID filenames in MAIN's explicitly
KID-selected experiment. ALPHA tests also refer to old helper names and
interfaces. These are not justification to reintroduce removed architecture
or change the ablations. No before/after full-legacy baseline was measured
in this turn, so this report does not label every legacy failure pre-existing.
The complete failing-name inventory is appended below.

Local evidence logs (excluded from Git) are under
`agent/analysis/2026-11-08-bug-audit/`, covering audit contracts, CPU smokes,
the MAIN KID contracts, ALPHA runtime checks and full legacy output.

## Compatibility and limitations

- No GAN architecture, parameter shapes, loss formula/weight, teacher
  architecture/weights, training dataset or metric estimator was changed.
  MAIN remains KID-selected; ALPHA remains FID-selected.
- Existing GAN checkpoints remain compatible. Resume from the refreshed
  checkout; no fresh GAN training or teacher pretraining is required for
  these fixes.
- Lost old W&B events or overwritten CSV rows cannot be reconstructed from
  missing data. Previously broken deployment exports should be regenerated
  from the full training checkpoint.
- If a prior run already saved incorrect epoch metadata after a bad resume,
  this patch cannot infer its true epoch automatically.
- Tests were CPU-only with synthetic mini-batches. No Kaggle/T4 run, full
  dataset metric evaluation, live W&B dashboard verification, DDP run or
  quality/convergence measurement was performed. W&B was isolated in tests;
  no test run was uploaded.
- Broader legacy pretraining/resume and inference permutations are not
  exhaustively validated by this bounded audit. Passing these regressions
  is not a guarantee of crash-free training in every environment.
- Generated tests, fixture checkpoints, logs and the unrelated outer-master
  deletions are not part of the model commits.

## Legacy discovery failure inventory

### MAIN

- `ERROR: test_emergency_publishes_weights_progress_and_preserves_best (test_checkpoint_eval_regressions.CheckpointEvalRegressions)`
- `ERROR: test_no_texture_head_both_heights (test_critic_experiment.CriticExperiment) (height=32)`
- `ERROR: test_no_texture_head_both_heights (test_critic_experiment.CriticExperiment) (height=64)`
- `ERROR: test_content_probe_is_global_only_with_global_grl_and_local_zero_gradient (test_global_only_integration.GlobalOnlyIntegration)`
- `ERROR: test_legacy_gan_state_is_rejected_by_production_strict_loader (test_global_only_integration.GlobalOnlyIntegration)`
- `ERROR: test_manual_interpolation_centers_endpoints_and_preserves_global_lerp (test_global_only_integration.GlobalOnlyIntegration)`
- `ERROR: test_manual_random_centers_sampler_outputs_and_preserves_global_repeats (test_global_only_integration.GlobalOnlyIntegration)`
- `ERROR: test_sample_images_centers_eval_prior_without_mutating_sampler (test_global_only_integration.GlobalOnlyIntegration)`
- `ERROR: test_train_centers_both_priors_and_routes_fake_objectives_through_wrapped_global_forward (test_global_only_integration.GlobalOnlyIntegration)`
- `ERROR: test_unguided_image_generator_centers_random_prior_without_changing_rng (test_global_only_integration.GlobalOnlyIntegration)`
- `ERROR: test_local_evidence_has_no_gate_bias_or_checkpoint_initialization_shim (test_luna_review.LunaReview)`
- `ERROR: test_optional_attention_padding_invariance_both_heights (test_luna_review.LunaReview) (height=32)`
- `ERROR: test_optional_attention_padding_invariance_both_heights (test_luna_review.LunaReview) (height=64)`
- `ERROR: test_hinge_penalizes_each_spatial_decision_before_averaging (test_multiscale_patch.MultiscalePatch)`
- `ERROR: test_single_scale_keeps_equal_word_weight_and_branch_separation (test_multiscale_patch.MultiscalePatch)`
- `ERROR: test_word_and_scale_balanced_readout (test_multiscale_patch.MultiscalePatch)`
- `ERROR: test_global_only_skips_local_modules_and_preserves_features (test_reference_local_evidence.ReferenceLocalEvidence) (height=32)`
- `ERROR: test_global_only_skips_local_modules_and_preserves_features (test_reference_local_evidence.ReferenceLocalEvidence) (height=64)`
- `ERROR: test_global_only_without_optional_returns_does_not_mask_feature_maps (test_reference_local_evidence.ReferenceLocalEvidence)`
- `ERROR: test_global_seam_preserves_writer_calculation_and_vae_sampling (test_reference_local_evidence.ReferenceLocalEvidence)`
- `ERROR: test_position_only_changes_keys (test_reference_local_evidence.ReferenceLocalEvidence)`
- `ERROR: test_uniform_values_cannot_create_local_style (test_reference_local_evidence.ReferenceLocalEvidence)`
- `ERROR: test_weak_visual_deviation_is_not_renormalized (test_reference_local_evidence.ReferenceLocalEvidence)`
- `ERROR: test_zero_and_single_local_slot_contract (test_reference_local_evidence.ReferenceLocalEvidence) (height=32, token_count=2)`
- `ERROR: test_zero_and_single_local_slot_contract (test_reference_local_evidence.ReferenceLocalEvidence) (height=64, token_count=2)`
- `ERROR: test_generator_gradients_logging_and_checkpoint_roundtrip (test_spatial_style_fusion.SpatialStyleFusion)`
- `ERROR: test_global_conditioning_still_changes_first_block (test_spatial_style_fusion.SpatialStyleFusion)`
- `ERROR: test_local_style_first_changes_features_after_first_gblock (test_spatial_style_fusion.SpatialStyleFusion)`
- `ERROR: test_seed_is_reference_independent_at_both_heights (test_spatial_style_fusion.SpatialStyleFusion)`
- `FAIL: test_emergency_before_first_batch_is_resumable (test_checkpoint_eval_regressions.CheckpointEvalRegressions)`
- `FAIL: test_encoder_local_visual_evidence_and_global_only_sampling (test_compact_style_context.CompactStyleContext) (height=32)`
- `FAIL: test_encoder_local_visual_evidence_and_global_only_sampling (test_compact_style_context.CompactStyleContext) (height=64)`
- `FAIL: test_experiment_settings (test_critic_experiment.CriticExperiment) (config='gan_iam_32.yml')`
- `FAIL: test_global_kl_preserves_previous_global_contribution (test_luna_review.LunaReview)`
- `FAIL: test_configured_critic_all_parameters_train_at_both_image_heights (test_multiscale_patch.MultiscalePatch)`
- `FAIL: test_crop_strata_cover_horizontal_extent_and_vertical_bands (test_multiscale_patch.MultiscalePatch)`
- `FAIL: test_relative_scales_counts_and_padding_invariance (test_multiscale_patch.MultiscalePatch)`
- `FAIL: test_sampler_rejects_invalid_lengths (test_multiscale_patch.MultiscalePatch)`
- `FAIL: test_short_words_do_not_read_or_backpropagate_through_padding (test_multiscale_patch.MultiscalePatch)`
- `FAIL: test_visual_reference_changes_centered_codes (test_reference_local_evidence.ReferenceLocalEvidence) (height=32)`
- `FAIL: test_visual_reference_changes_centered_codes (test_reference_local_evidence.ReferenceLocalEvidence) (height=64)`
- `FAIL: test_zero_and_single_local_slot_contract (test_reference_local_evidence.ReferenceLocalEvidence) (height=32, token_count=1)`
- `FAIL: test_zero_and_single_local_slot_contract (test_reference_local_evidence.ReferenceLocalEvidence) (height=64, token_count=1)`
- `FAIL: test_small_local_values_produce_small_correction (test_spatial_style_fusion.SpatialStyleFusion)`
- `FAIL: test_spatial_padding_blocks_tail_output_and_gradients (test_spatial_style_fusion.SpatialStyleFusion)`
- `FAIL: test_spatial_queries_keep_vertical_information (test_spatial_style_fusion.SpatialStyleFusion)`
- `FAIL: test_spatial_strength_is_unbounded_and_slots_are_permutation_invariant (test_spatial_style_fusion.SpatialStyleFusion)`
- `FAIL: test_value_projection_receives_raw_local_residuals (test_spatial_style_fusion.SpatialStyleFusion)`
- `FAIL: test_zero_local_codes_are_neutral_for_any_global (test_spatial_style_fusion.SpatialStyleFusion)`

### ALPHA

- `ERROR: test_background_consistency (unittest.loader._FailedTest)`
- `ERROR: test_cmmd_cache_distinguishes_dataset_file_and_revision (test_checkpoint_eval_regressions.CheckpointEvalRegressions)`
- `ERROR: test_cmmd_cache_distinguishes_heights (test_checkpoint_eval_regressions.CheckpointEvalRegressions)`
- `ERROR: test_cmmd_cache_distinguishes_subsets (test_checkpoint_eval_regressions.CheckpointEvalRegressions)`
- `ERROR: test_emergency_before_first_batch_is_resumable (test_checkpoint_eval_regressions.CheckpointEvalRegressions)`
- `ERROR: test_matching_wier_checkpoint_loads_strictly_and_reuses_models (test_checkpoint_eval_regressions.CheckpointEvalRegressions)`
- `ERROR: test_wier_rejects_wrong_head_and_restores_rng_on_failure (test_checkpoint_eval_regressions.CheckpointEvalRegressions)`
- `ERROR: test_compact_style_context (unittest.loader._FailedTest)`
- `ERROR: test_experiment_settings (test_critic_experiment.CriticExperiment) (config='gan_iam_32.yml')`
- `ERROR: test_experiment_settings (test_critic_experiment.CriticExperiment) (config='gan_iam_64.yml')`
- `ERROR: test_no_texture_head_both_heights (test_critic_experiment.CriticExperiment) (height=32)`
- `ERROR: test_no_texture_head_both_heights (test_critic_experiment.CriticExperiment) (height=64)`
- `ERROR: test_global_only_integration (unittest.loader._FailedTest)`
- `ERROR: test_local_style_contract (unittest.loader._FailedTest)`
- `ERROR: test_local_evidence_has_no_gate_bias_or_checkpoint_initialization_shim (test_luna_review.LunaReview)`
- `ERROR: test_masked_batchnorm_matches_native_with_full_lengths (test_luna_review.LunaReview)`
- `ERROR: test_optional_attention_padding_invariance_both_heights (test_luna_review.LunaReview) (height=32)`
- `ERROR: test_optional_attention_padding_invariance_both_heights (test_luna_review.LunaReview) (height=64)`
- `ERROR: test_hinge_penalizes_each_spatial_decision_before_averaging (test_multiscale_patch.MultiscalePatch)`
- `ERROR: test_single_scale_keeps_equal_word_weight_and_branch_separation (test_multiscale_patch.MultiscalePatch)`
- `ERROR: test_word_and_scale_balanced_readout (test_multiscale_patch.MultiscalePatch)`
- `ERROR: test_global_only_skips_local_modules_and_preserves_features (test_reference_local_evidence.ReferenceLocalEvidence) (height=32)`
- `ERROR: test_global_only_skips_local_modules_and_preserves_features (test_reference_local_evidence.ReferenceLocalEvidence) (height=64)`
- `ERROR: test_global_only_without_optional_returns_does_not_mask_feature_maps (test_reference_local_evidence.ReferenceLocalEvidence)`
- `ERROR: test_global_seam_preserves_writer_calculation_and_vae_sampling (test_reference_local_evidence.ReferenceLocalEvidence)`
- `ERROR: test_position_only_changes_keys (test_reference_local_evidence.ReferenceLocalEvidence)`
- `ERROR: test_uniform_values_cannot_create_local_style (test_reference_local_evidence.ReferenceLocalEvidence)`
- `ERROR: test_visual_reference_changes_centered_codes (test_reference_local_evidence.ReferenceLocalEvidence) (height=32)`
- `ERROR: test_visual_reference_changes_centered_codes (test_reference_local_evidence.ReferenceLocalEvidence) (height=64)`
- `ERROR: test_weak_visual_deviation_is_not_renormalized (test_reference_local_evidence.ReferenceLocalEvidence)`
- `ERROR: test_zero_and_single_local_slot_contract (test_reference_local_evidence.ReferenceLocalEvidence) (height=32, token_count=1)`
- `ERROR: test_zero_and_single_local_slot_contract (test_reference_local_evidence.ReferenceLocalEvidence) (height=32, token_count=2)`
- `ERROR: test_zero_and_single_local_slot_contract (test_reference_local_evidence.ReferenceLocalEvidence) (height=64, token_count=1)`
- `ERROR: test_zero_and_single_local_slot_contract (test_reference_local_evidence.ReferenceLocalEvidence) (height=64, token_count=2)`
- `ERROR: test_spatial_style_fusion (unittest.loader._FailedTest)`
- `FAIL: test_emergency_publishes_weights_progress_and_preserves_best (test_checkpoint_eval_regressions.CheckpointEvalRegressions)`
- `FAIL: test_missing_wier_teacher_reports_path (test_checkpoint_eval_regressions.CheckpointEvalRegressions)`
- `FAIL: test_experiment_settings (test_critic_experiment.CriticExperiment) (config='gan_iam_local.yml')`
- `FAIL: test_attention_ignores_padding_and_blocks_tail_gradients (test_luna_review.LunaReview)`
- `FAIL: test_global_kl_preserves_previous_global_contribution (test_luna_review.LunaReview)`
- `FAIL: test_configured_critic_all_parameters_train_at_both_image_heights (test_multiscale_patch.MultiscalePatch)`
- `FAIL: test_crop_strata_cover_horizontal_extent_and_vertical_bands (test_multiscale_patch.MultiscalePatch)`
- `FAIL: test_relative_scales_counts_and_padding_invariance (test_multiscale_patch.MultiscalePatch)`
- `FAIL: test_sampler_rejects_invalid_lengths (test_multiscale_patch.MultiscalePatch)`
- `FAIL: test_short_words_do_not_read_or_backpropagate_through_padding (test_multiscale_patch.MultiscalePatch)`
