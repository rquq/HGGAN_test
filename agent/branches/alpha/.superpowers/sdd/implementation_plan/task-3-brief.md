### Task 3: Preserve weak local evidence in spatial G fusion

**Files:**
- Modify: `networks/fusion.py`, `SpatialStyleRefinement`.
- Modify: ignored `tests/test_spatial_style_fusion.py`.

**Interfaces:**
- Consume `(B,N,D)` global-plus-centered-local codes from Task 2/Task 1.
- Keep `SpatialStyleRefinement.forward` signature, GBlock-1 placement, spatial axes, Q/K/O paths and constructor-owned per-channel `residual_scale` initialized to `0.25`.
- Keys use normalized locals; the bias-free V projection consumes raw `styles[:, 1:]`, not normalized locals minus normalized global.

- [ ] **Write failing assertions** `test_zero_local_codes_are_neutral_for_any_global`, `test_value_projection_receives_raw_local_residuals`, `test_small_local_values_produce_small_correction`. Use a V-projection hook and compare fixed-key attention while scaling values; don't assume attention weights remain fixed if keys are independently changed.
- [ ] **Run:** `/home/quq/miniconda3/envs/HiGANplus/bin/python -B tests/test_spatial_style_fusion.py`. Expected: the old global-subtraction value contract fails.
- [ ] **Change only the local value handoff**; do not add confidence gates, forced attention diversity, ceilings, new centering, or normalization of values.
- [ ] **Run the full spatial test file:** retain content-first seed independence, global BSSP influence, vertical detail, local-slot permutation invariance, padding/tail gradient checks, constructor-owned parameters and strict state roundtrip. Expected: all pass at 32/64px.
- [ ] **Review boundary:** root checks no incidental ContentContext/GBlock/D/P changes. Do not commit.

