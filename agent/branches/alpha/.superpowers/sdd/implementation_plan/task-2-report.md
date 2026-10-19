# Task 2 implementation report

Status: implemented and CPU-verified; ready for root review. No design blocker.

## Owned changes

- `networks/module.py`: changes are confined to `StyleEncoder`. Added `_encode_global_style` and `_build_local_attention_inputs`, appended `global_only=False`, added constructor-owned bias-free `local_projection`, and made cross-attention bias-free. Removed gate options/aliases, reliability descriptors/MLP, sigmoid attention strength, query output residual, local output normalization, shared global `mu` use on local appearance, and the checkpoint initialization shim.
- `configs/gan_iam_32.yml`, `configs/gan_iam_64.yml`, `configs/gan_iam_local.yml`: removed only the three obsolete gate values from each encoder config. Updated the stale 32px checkpoint-compatibility comment; all non-obsolete parsed config values are identical to the Task 2 baseline.
- Ignored tests: created `tests/test_reference_local_evidence.py`; updated `tests/test_compact_style_context.py` and `tests/test_luna_review.py`.
- Did not modify `predict_content`, `global_token`, `model.py`, G, B/W/R/D/P definitions, loss helpers, or prior helpers. Existing Task 1 changes in `loss.py`/`utils.py` were retained untouched.

## Contract and geometry

Global and full forwards share the same width-context, valid mean, global norm/MLP, `mu`, and logvar-clamp calculation. VAE randomness remains global-only; local samples equal local means and local logvars remain zero. Global-only outputs have `(B,1,D)` and preserve the existing optional tuple ordering.

The local helper receives selected feature maps but the full backbone `map_lengths` list. It indexes lengths via `self.feature_indices`; subset/reordering `(4,2)` is tested explicitly. The existing pointwise projections and four-row pooling remain. Each visual token is normalized once before position is added to keys only. Values are normalized visual features minus the per-map valid-spatial mean, with invalid values zeroed. The 1D global context stream is not appended. Attention output goes through the dedicated bias-free projector and then local-slot centering, with no post-attention renormalization.

Global-only mode skips the helper, all local map projections, cross-attention, and the local projector. Optional selected original maps remain width-masked, and the original backbone sequence is returned unchanged. Both masked and unmasked optional contracts are covered. E handles N=1 without empty-mean NaNs and N=2 with an exactly zero centered local slot; G's minimum-token validation is untouched.

## RED evidence

Before changing production behavior:

```text
/home/quq/miniconda3/envs/HiGANplus/bin/python -B tests/test_reference_local_evidence.py
Ran 7 tests: FAILED (failures=4, errors=6), exit 1.
```

Expected RED causes: missing `_encode_global_style`, missing `_build_local_attention_inputs`, missing `local_projection`, non-centered locals at both heights (maximum local mean deviations approximately 0.3903/0.3893), and nonzero one-local-slot output at both heights. N=1 was already finite and remained so.

```text
/home/quq/miniconda3/envs/HiGANplus/bin/python -B tests/test_luna_review.py LunaReview.test_local_evidence_has_no_gate_bias_or_checkpoint_initialization_shim
Ran 1 test: FAILED (failures=1), exit 1.
```

Expected RED cause: obsolete `local_evidence_gate` still existed.

## GREEN evidence

All commands ran from ALPHA on CPU, with no GPU or training/metric run:

```text
/home/quq/miniconda3/envs/HiGANplus/bin/python -B tests/test_reference_local_evidence.py
Ran 7 tests in 0.889s: OK, exit 0 (final focused rerun).

/home/quq/miniconda3/envs/HiGANplus/bin/python -B tests/test_compact_style_context.py
Ran 5 tests in 0.749s: OK, exit 0.

/home/quq/miniconda3/envs/HiGANplus/bin/python -B tests/test_luna_review.py
Ran 6 tests in 0.214s: OK, exit 0.

/home/quq/miniconda3/envs/HiGANplus/bin/python -B -m unittest discover -s tests -p 'test_*.py' -v
Ran 62 tests in 4.146s: OK, exit 0 (final discovery rerun).

git diff --check
Exit 0, no whitespace errors.
```

New coverage includes position-only keys, uniform-evidence zero locals through real full E forwards, weak visual perturbations around a nondegenerate channel vector (approximately 100x attenuation rather than renormalization), reference-dependent centered codes, finite gradients through queries/attention/projector/map projections/input evidence, mixed-width padding poisoning and zero tail gradients at both heights, global-only module bypass and features, same-seed VAE parity, and N=1/N=2.

Additional inline Python diagnostic commands (exit 0):

- Parsed the saved `task-2-baseline.json` and current `module.py` AST: the only changed top-level definition is `StyleEncoder`; `predict_content` and `global_token` ASTs are unchanged. This also confirms B/W/R and unrelated module definitions were untouched.
- Parsed all three YAML files against the Task 2 snapshot after removing the obsolete keys: every other value is identical. All three configured encoders construct successfully; query-row Gram matrices match the existing orthogonal initialization scale.
- Executed the saved original encoder module in an isolated in-memory namespace, copied identical global-path weights, and compared old/new full forwards at 32px and 64px with and without width metadata. Global mean, clamped logvar, and same-seed VAE samples are bitwise identical (`rtol=0`, `atol=0`).
- `git check-ignore` confirmed the three test files and this report are ignored.

## Self-review and concerns

- No implementation blocker remains. Source/config ownership boundaries were verified above; no staging, commit, push, checkpoint conversion, or GPU command was used.
- One initial GREEN attempt revealed a test-only expectation issue: after slicing local tokens from concatenated global/local output, autograd may give global-context parameters zero gradient tensors rather than `None`. Assertions now require absent-or-zero global-context gradients for local-only losses. The separate global-loss context-gradient test remains and passes.
- New GAN architecture intentionally requires fresh GAN training and fails strict loading of obsolete gate keys or a missing local projector. No compatibility shim was added. Existing W+B/R definitions were not changed, but real pretrained-checkpoint load verification is left to root's integration gate.
- Existing ignored exploratory scripts such as `analyze_short_references.py` and `checkpoint_probe_14_10.py` still refer to the removed gate; they are outside this task's assigned test ownership and were not altered. They are not part of unittest discovery.
- No empirical image-quality, optimization-stability, or metric improvement claim is made from these CPU regressions. GPU/integrated objective checks remain root-owned.
