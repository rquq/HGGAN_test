# Verification inventory — 2026-11-09

Focused MAIN case-policy tests: four passed. MAIN KID contracts: eight passed.
Applicable bug-audit regressions: MAIN seven passed, ALPHA nine passed.
ALPHA context tests: two passed (native-height/padding/gradients and paired-seed shared initialization).
CPU 32/64 GAN smokes passed for both branches. Supplied MAIN checkpoint continuation passed.

These gates do not establish quality gains. Full legacy discovery remains red.
No claim that every failure below predates this turn is made. Several tests require
retired modules or architecture/config contracts that differ from the experiments.
Generated helpers and full logs are excluded from Git.

## MAIN

Ran 71 tests in 1.793s
FAILED (failures=20, errors=29)

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

## ALPHA

Ran 51 tests in 0.820s
FAILED (failures=16, errors=38)

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
- `ERROR: test_generator_gradients_logging_and_checkpoint_roundtrip (test_spatial_style_fusion.SpatialStyleFusion)`
- `ERROR: test_global_conditioning_still_changes_first_block (test_spatial_style_fusion.SpatialStyleFusion)`
- `ERROR: test_local_style_first_changes_features_after_first_gblock (test_spatial_style_fusion.SpatialStyleFusion)`
- `ERROR: test_seed_is_reference_independent_at_both_heights (test_spatial_style_fusion.SpatialStyleFusion)`
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
- `FAIL: test_small_local_values_produce_small_correction (test_spatial_style_fusion.SpatialStyleFusion)`
- `FAIL: test_spatial_padding_blocks_tail_output_and_gradients (test_spatial_style_fusion.SpatialStyleFusion)`
- `FAIL: test_spatial_queries_keep_vertical_information (test_spatial_style_fusion.SpatialStyleFusion)`
- `FAIL: test_spatial_strength_is_unbounded_and_slots_are_permutation_invariant (test_spatial_style_fusion.SpatialStyleFusion)`
- `FAIL: test_value_projection_receives_raw_local_residuals (test_spatial_style_fusion.SpatialStyleFusion)`
- `FAIL: test_zero_local_codes_are_neutral_for_any_global (test_spatial_style_fusion.SpatialStyleFusion)`
