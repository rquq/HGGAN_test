# Task 3 report: preserve weak local evidence in spatial G fusion

## Change

`SpatialStyleRefinement` still normalizes `style_seq[:, 1:]` for K, but now sends the raw centered local codes directly to V. It no longer subtracts normalized global style from local values. Kept `SpatialStyleRefinement.forward`, the GBlock placement, Q/K/O paths, and the existing constructor-owned `reference_scale` parameter and initialization unchanged. The plan's `residual_scale` wording was descriptive; no parameter rename was made.

The ignored spatial test file now has the requested contract tests:

- zero local codes are neutral regardless of global code;
- V receives the raw local residuals;
- with K held fixed, reducing raw local values sharply reduces the correction.

## RED

Command:

```text
/home/quq/miniconda3/envs/HiGANplus/bin/python -B tests/test_spatial_style_fusion.py
```

Output before the production edit (the three new tests failed on the expected old value contract; remaining eight tests passed):

```text
test_content_context_short_long_padding_and_gradients (__main__.SpatialStyleFusion) ... ok
test_generator_gradients_logging_and_checkpoint_roundtrip (__main__.SpatialStyleFusion) ... ok
test_global_conditioning_still_changes_first_block (__main__.SpatialStyleFusion) ... ok
test_local_style_first_changes_features_after_first_gblock (__main__.SpatialStyleFusion) ... ok
test_seed_is_reference_independent_at_both_heights (__main__.SpatialStyleFusion) ... ok
test_small_local_values_produce_small_correction (__main__.SpatialStyleFusion) ... FAIL
test_spatial_padding_blocks_tail_output_and_gradients (__main__.SpatialStyleFusion) ... ok
test_spatial_queries_keep_vertical_information (__main__.SpatialStyleFusion) ... ok
test_spatial_strength_is_unbounded_and_slots_are_permutation_invariant (__main__.SpatialStyleFusion) ... ok
test_value_projection_receives_raw_local_residuals (__main__.SpatialStyleFusion) ... FAIL
test_zero_local_codes_are_neutral_for_any_global (__main__.SpatialStyleFusion) ... FAIL

======================================================================
FAIL: test_small_local_values_produce_small_correction (__main__.SpatialStyleFusion)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/quq/machineLearning/HTG/HGGAN_test/alpha/tests/test_spatial_style_fusion.py", line 130, in test_small_local_values_produce_small_correction
    torch.testing.assert_close(values[0], 0.01 * values[1], atol=1e-7, rtol=1e-6)
AssertionError: Tensor-likes are not close!

Mismatched elements: 448 / 448 (100.0%)
Greatest absolute difference: 4.131281852722168 at index (1, 3, 31) (up to 1e-07 allowed)
Greatest relative difference: 4833.68310546875 at index (0, 2, 28) (up to 1e-06 allowed)

======================================================================
FAIL: test_value_projection_receives_raw_local_residuals (__main__.SpatialStyleFusion)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/quq/machineLearning/HTG/HGGAN_test/alpha/tests/test_spatial_style_fusion.py", line 102, in test_value_projection_receives_raw_local_residuals
    torch.testing.assert_close(seen[0], local_codes, atol=0, rtol=0)
AssertionError: Tensor-likes are not equal!

Mismatched elements: 448 / 448 (100.0%)
Greatest absolute difference: 2.930229425430298 at index (1, 4, 15) (up to 0 allowed)
Greatest relative difference: 100.49958038330078 at index (1, 5, 5) (up to 0 allowed)

======================================================================
FAIL: test_zero_local_codes_are_neutral_for_any_global (__main__.SpatialStyleFusion)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/quq/machineLearning/HTG/HGGAN_test/alpha/tests/test_spatial_style_fusion.py", line 91, in test_zero_local_codes_are_neutral_for_any_global
    torch.testing.assert_close(module(feature, style), feature, atol=0, rtol=0)
AssertionError: Tensor-likes are not equal!

Mismatched elements: 6144 / 6144 (100.0%)
Greatest absolute difference: 0.38946855068206787 at index (0, 29, 0, 0) (up to 0 allowed)
Greatest relative difference: 1409.1915283203125 at index (0, 12, 7, 10) (up to 0 allowed)

----------------------------------------------------------------------
Ran 11 tests in 0.923s

FAILED (failures=3)
```

## GREEN

Focused command:

```text
/home/quq/miniconda3/envs/HiGANplus/bin/python -B tests/test_spatial_style_fusion.py
```

Final output:

```text
test_content_context_short_long_padding_and_gradients (__main__.SpatialStyleFusion) ... ok
test_generator_gradients_logging_and_checkpoint_roundtrip (__main__.SpatialStyleFusion) ... ok
test_global_conditioning_still_changes_first_block (__main__.SpatialStyleFusion) ... ok
test_local_style_first_changes_features_after_first_gblock (__main__.SpatialStyleFusion) ... ok
test_seed_is_reference_independent_at_both_heights (__main__.SpatialStyleFusion) ... ok
test_small_local_values_produce_small_correction (__main__.SpatialStyleFusion) ... ok
test_spatial_padding_blocks_tail_output_and_gradients (__main__.SpatialStyleFusion) ... ok
test_spatial_queries_keep_vertical_information (__main__.SpatialStyleFusion) ... ok
test_spatial_strength_is_unbounded_and_slots_are_permutation_invariant (__main__.SpatialStyleFusion) ... ok
test_value_projection_receives_raw_local_residuals (__main__.SpatialStyleFusion) ... ok
test_zero_local_codes_are_neutral_for_any_global (__main__.SpatialStyleFusion) ... ok

----------------------------------------------------------------------
Ran 11 tests in 0.909s

OK
```

CPU unittest discovery command:

```text
/home/quq/miniconda3/envs/HiGANplus/bin/python -B -m unittest discover -s tests -p 'test_*.py'
```

Final output:

```text
...........--> Saved last checkpoint: last_fid_inf.pth
.--> Saved last checkpoint: last_fid_inf.pth
.--> Saved last checkpoint: last_fid_6.0000.pth
--> Saved new best checkpoint: best_fid_6.0000.pth (FID: 6.0000)
--> Saved last checkpoint: last_fid_6.0000.pth
.load pretrained test_writer_identifier: /tmp/hggan-checkpoint-regression-l83ob0s4/matching-writer.pth
.................................................
----------------------------------------------------------------------
Ran 63 tests in 3.968s

OK
```

## Self-review

- Production diff is confined to `SpatialStyleRefinement`'s local-value handoff; normalized locals remain the K input and raw locals are the V input.
- The existing `reference_scale` name, shape, initialization, reset behavior, and logging consumer are unchanged.
- No GBlock, ContentContext, D/P, loss, configuration, or other branch/worktree files were changed by this task. Other pre-existing modified files were left untouched.
- Focused spatial tests and discovered `test_*.py` suite pass. No GPU test, training, metric evaluation, commit, or staging was performed.
