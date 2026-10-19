### Task 1: Centered-prior and existing-objective contracts

**Files:**
- Modify: `networks/utils.py` — one style-prior tensor helper.
- Modify: `networks/loss.py` — small testable wrappers around the existing L1 and BCE expressions, not new objectives.
- Create: ignored `tests/test_local_style_contract.py`.

**Interfaces:**
- Produce `center_style_prior(styles: torch.Tensor) -> torch.Tensor`: accepts `(B,N,D)` with `N >= 1`, strips Tensor subclass without detaching, preserves token zero and returns centered remaining slots. `N=1` is a no-op on values; reject invalid rank/zero token count clearly. Do not mutate the input or sampler RNG state.
- Produce `global_style_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, num_style_tokens: int) -> torch.Tensor`: accepts full or global-only style tensors, applies existing L1 to `[:, :1]` and divides by the original positive integer `N`.
- Produce `global_content_adversarial_loss(logits: torch.Tensor, labels: torch.Tensor, num_style_tokens: int) -> torch.Tensor`: logits are `(B,1,C)`; preserve bag-of-character targets, blank index zero, positive fraction clamped to `[1e-4,0.5]`, positive weight capped at `10`, mean BCE divided by original `N`.

- [ ] **Write failing tests** named `test_prior_centering_preserves_global_and_gradients`, `test_distribution_prior_returns_base_tensor`, `test_global_l1_matches_former_global_gradient` and `test_global_bce_matches_former_global_gradient`. Check local means near zero, no mutation, exact global values, finite gradients, invalid shapes and `N=1`; compare the global gradient against the old all-token mean, not against an unscaled global mean. Changing only local slots must leave global L1 unchanged.
- [ ] **Run the tests before implementation:** `PYTHONDONTWRITEBYTECODE=1 /home/quq/miniconda3/envs/HiGANplus/bin/python -B tests/test_local_style_contract.py`. Expected: missing-helper import failure, not a missing dependency failure.
- [ ] **Implement the three named helpers** in the existing files. Preserve autograd when converting the `Distribution` subclass with `as_subclass(torch.Tensor)`; never use `.detach()` or `torch.tensor(existing_tensor)` for this conversion.
- [ ] **Run the same command:** all tests pass. Confirm these helpers add no parameters, config keys or dependencies.
- [ ] **Review boundary:** root checks the exact `/N` math, BCE target equivalence and sampler-state preservation. Do not commit.

