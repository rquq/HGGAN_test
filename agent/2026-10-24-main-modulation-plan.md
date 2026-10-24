# MAIN spatial modulation and DEV dimension audit implementation plan

> For agentic workers: use inline execution with test-driven development and a final independent review. The user approved implementing the gain-only experiment on 2026-10-24.

**Goal:** Make MAIN the DEV shared-backbone baseline with gain-only reference modulation, and quantify redundancy in the supplied DEV epoch-34 checkpoint.

**Architecture:** Keep one shared frozen B, one global 32D latent and spatial 128D memory. MAIN keeps the same Q/K/V attention after GBlock1 but replaces `H + scale * projection(attention)` by `H * (1 + scale * gain_projection(attention))`. No new attention, losses, guards, gates or auxiliary encoders.

**Tech stack:** Existing PyTorch, CPU FP32 diagnostic execution, existing IAM HDF5 files, NumPy/PIL.

**Spec:** This self-contained plan records the design approved in the conversation. It is an ablation, not a promised quality improvement.

## Global constraints

- Edit MAIN only; DEV and ALPHA remain unchanged.
- Copy only the DEV product-code/config differences needed for synchronization; preserve each nested repository and all teacher checkpoints/runtime files.
- Match DEV's training and evaluation hyperparameters. Clear MAIN's old GAN resume default for fresh architecture training; keep W+B/R paths.
- Keep normal W&B losses/metrics/samples/resources; no new training diagnostics.
- Use a distinct module/projection state name so old additive checkpoints cannot silently become a modulation experiment.
- Documentation goes under `HGGAN_test/agent`; local helpers/tests stay in ignored `agent/tests`.
- No optimizer steps on the audited DEV checkpoint. Its saved Adam moments are historical summaries; fresh probe gradients must be labeled separately.
- No commit or push without a current request.

## Review focus

- Missing/all-false memory must bypass retrieval without an all-masked softmax.
- Zero generator features must remain zero at the modulation handoff, even with nonzero memory.
- Both heights and mixed random/reference batches must support finite G/E gradients and frozen teachers.
- Old additive checkpoints must not silently load as the new experiment.
- Low per-reference rank is not by itself channel collapse; distinguish pooled, within-reference and between-reference spectra, head usage, optimizer evidence and functional compression sensitivity.

## Task 1: Implement one controlled MAIN experiment

- [x] Write isolated gain-only tests, and run them against unchanged MAIN to observe expected failures.
- [x] Synchronize DEV's nine YAML files plus `networks/{module,fusion,BigGAN_networks,model}.py` and `metric/ocr_eval.py` into MAIN using patches. Do not copy Git metadata, cached graph indexes or checkpoints.
- [x] Introduce `SpatialStyleModulation` with `gain_proj`, update its G construction, and use multiplicative residual modulation. Preserve masks, dimensions, initialization scale and bypass semantics.
- [x] Check MAIN's GAN resume paths: inherited DEV defaults are already empty. Run the new tests and inherited reference-memory contracts on MAIN.

## Task 2: Audit DEV checkpoint capacity and use

- [x] Load `/home/quq/machineLearning/HTG/chkpoints/24_10_2026/last_fid_2.6501_dev.pth` strictly in CPU FP32.
- [x] Measure memory covariance spectra, per-reference/global variation, projection spectra, attention-head diversity and query-varying contribution.
- [x] Inspect saved Adam moments, then compute separately labeled fresh loss-specific gradients without stepping optimizers.
- [x] Perform held-out PCA-memory and common-reference-update interventions. Report inference sensitivity, not retrained low-dimensional model scores.

## Task 3: Integration and delivery

- [x] Run native 32/64px real train-loop smokes, including R1, EMA, sampling and evaluation-interface wiring.
- [x] Run the discoverable existing MAIN test suite; distinguish stale architecture tests from new failures.
- [x] Review the final diff independently, verify DEV/ALPHA and teachers stayed unchanged, and write dated implementation and dimension-audit reports.
- [x] Explain fresh GAN training versus reusable pretrains and the limits of CPU/runtime and compression evidence.

## Progress and rulings

- 2026-10-24: MAIN and DEV working trees were clean before edits.
- Ruling: edit the existing MAIN checkout directly because the user specifically requested the experiment in MAIN; no new worktree or branch is needed.
- Ruling: keep the known black-reference/global-latent failure unchanged in this experiment to avoid adding a second architectural variable.
- Task 1: complete. Observed 6 intended failures before implementation; 6 modulation and 10 shared-memory contracts pass afterwards.
- Integration: actual native32/64 training-loop smoke passed (2 critic batches, 1 G/EMA update per height). Teachers remain frozen; R1 and sample generation exercised.
- Ruling: no pytest is installed in the existing ML environment; used unittest discovery without installing dependencies. The historical ignored suite ran 71 tests with 6 failures and 31 errors due to retired token interfaces/settings/source-text assertions. Do not present the overall historical suite as green or change product behavior to satisfy the old experiment.
- Review: independent read-only reviewer found no Critical/Important product issue. Two diagnostic precision issues (pre-scale versus actual delta statistics and aggregate L2 versus RMS) are being corrected and the audit rerun. Weight-space V/output composition is explicitly not the complete multi-head attention operator.
- Review fixes: actual residual-scale-weighted delta statistics and valid-token gradient RMS/L2 were separated and verified by a full audit rerun. No product-code repair was needed.
- Task 2: complete. 256 calibration plus 160 held-out/short references, 48 functional intervention cases, and six two-image raw-gradient batches, without optimizer steps.
- Task 3: complete. 43 focused/current-contract tests pass; the historical ignored suite limitation remains explicit. Both dated reports and the index are updated. No commit/push performed.
