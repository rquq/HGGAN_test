# Lightweight handwriting evaluation and paired local-path ablation

Status: approved and implemented on 2026-10-19. The user's final amendment supersedes the broader metric proposal below: ONLY independent CER and WER are implemented, with no NED, accuracy or additional teacher series. Existing checkpoints and datasets were not rewritten.

## Purpose and constraints

Make the next MAIN/ALPHA paired run identify an actual architectural contribution, while exposing readability and short-reference failures that existing same-text FID/KID/HWD do not resolve. Support native 32px and 64px, single-GPU per branch, FP32, batch 8 training, Kaggle epoch monitoring and bounded local checkpoint evaluation. Preserve official benchmark data, preprocessing and historical score definitions.

No claim that finite tests can eliminate all future bottlenecks. No metric-driven character fallback, no special-case generator behavior for punctuation/digits, no new training losses, no replacement of the benchmark dataset, no automatic heavy downloads or background training. Do not change running Kaggle jobs or auto-launch the next jobs.

## Research basis: recent GAN papers, not a universal leaderboard

The reviewed works use different data processing and reporting setups. Their printed scores do not establish one universal ranking and cannot be compared directly with this repository without matching the full protocol.

- **SpiS-GAN (2026), sections 4.2–4.5:** test-set replication plus IV/OOV × seen/unseen writer scenarios; FID/KID/HWD; downstream recognizer augmentation using 5,000 real plus 25,000 synthetic samples. The downstream CER/WER are not direct readings by the GAN's training teacher. Source: https://arxiv.org/html/2607.06949v1
- **FW-GAN (2025), sections 4.3–4.5:** distribution metrics across vocabulary/writer scenarios and separate synthetic-data HTR augmentation. Source: https://arxiv.org/html/2508.21040v1
- **GL-GAN (COLING 2025), sections 4.1–4.5:** IAM and out-of-domain CVL evaluation with FID/KID/HWD; module ablations; word recognition metrics also discussed. KID table scale is ×100. Source: https://aclanthology.org/2025.coling-main.166.pdf
- **HiGAN+ (TOG 2022):** retain its reference-guided versus random-style distinction and separate perceptual, style and recognition assessment. Official implementation: https://github.com/ganji15/HiGANplus
- **Rethinking HTG Evaluation (ECCV workshop 2024):** distinguishes readable synthetic samples from useful/diverse synthetic training data. HTG_HTR trains HTR on synthetic and tests on real; HTG_style assesses writer imitation; HTG_OOV reads generated unseen words. Source: https://arxiv.org/html/2409.02683v1

The proposal below is a small monitoring adaptation of these ideas, not a reproduction of their 25,000-image experiments or downstream tables.

## Next paired experiment: decide whether the local path helps

### MAIN: control

Use current MAIN architecture: 32D global summary, four local tokens, spatial local attention, existing G/D/P, teachers and losses. Keep the existing settings unless needed to make both runs identical in training duration.

### ALPHA: global-only ablation

Mirror the same active model/training implementation and hyperparameters, then disable local extraction and bypass G's local spatial refinement. One global token remains and still conditions all GBlocks. Do not feed zero placeholders through attention or instantiate a one-key attention pretending it tests selection. Do not instantiate unused trainable local modules just to keep a parameter count artificially large.

Keep the explicit global-loss normalization divisor **8** for info, cycle, content adversarial and global KL. It must not silently become 1 when total tokens become 1. Preserve global latent dimension, loss schedules, D/P update frequency, teacher states, datasets, batch size and optimization policy.

Use fresh GAN training for the final comparison; keep existing W+B and OCR pretrains. Existing 4/7-local checkpoint continuations are not clean global-only training controls. A checkpoint-inference bypass is diagnostic only.

Use the same data manifests and record seed, data/teacher fingerprints, optimizer settings, active configuration and exact G/E parameter counts. Equal seeds do not imply identical RNG consumption across different architectures; repeated seeds are required before claiming superiority. Where feasible initialize shared parameters identically, without transplanting trained local-model weights into only one arm.

### Interpretation and later experiments

- If global-only is competitive across realism, handwriting distance and independent readability, simplify the pipeline rather than preserve a local module by assumption.
- If locals help, the next pair is shared-query versus spatial-query attention, holding the four-token extractor fixed. The shared query comes from pooled valid G features, with the same key/value/out projections, then broadcasts its attended result. Uniform local averaging is forbidden as a control because centered locals sum to zero.
- Only afterward compare pooled multiscale local evidence with learned-query retrieval.
- Test prior/posterior scale alignment separately. Do not jointly change extractor, normalization, token count and LR and attribute the outcome to one cause.
- A larger global-only capacity control is a later requirement if claiming local decomposition beats equally budgeted global representation, rather than merely that the implemented extra path helps.

## Evaluation layers

### 1. Existing official benchmark: unchanged

Keep the current complete-set FID, KID, HWD, seed, same-text style-guided generation, data polarity, resizing/cropping and checkpoint-best-FID selection. Preserve the embedded CSV schema and old histories. Do not replace full FID with a 256-sample estimate or compare tiny-subset KID directly with published/full scores.

New diagnostics are separate named outputs. They must not silently change the meaning of historical `cer`, `wer`, `fid`, `kid` or `hwd` columns. A corrected recognizer-metric version must be explicit in metadata.

### 2. Fast fixed transfer monitor: 256 core generated samples

Create one deterministic manifest, reused byte-for-byte by both branches when corpus metadata match. Different resolution corpora with different metadata require separate manifests. Four cells of 64 examples each: IV-S, IV-U, OOV-S, OOV-U. For every example record source split, original writer ID, reference index, target text and vocabulary classification. Target text must differ from reference text. IDs are split-local: seen/unseen labels require an explicit declaration that writer splits are disjoint.

Seen/unseen refers to GAN writer exposure, not OCR writer classification accuracy. Record original writer IDs rather than relying on separately renormalized split labels.

Record two vocabulary checks: absence from real GAN training transcripts, and absence from the configured GAN lexicon/sampling corpus. A word absent only from real transcripts but sampled during training is not fully unexposed OOV. Define primary OOV against all configured textual exposure; fail or explicitly mark unavailable if insufficient eligible words exist. Do not invent malformed character strings to fill cells.

Balance references over writers where practical and report counts; do not let prolific writers silently dominate. Choose naturally occurring target words with the configured alphabet/length bounds. Use an independent fixed manifest RNG so training seed/RNG state cannot change evaluation cases.

The existing test corpus is already used for monitoring/model selection. Label these diagnostic subsets accordingly; do not call them a pristine independently untouched final test set. A paper-level final generalization claim needs an independently reserved audit or additional untouched dataset.

### 3. Short-reference transfer probe: 32 writer-matched pairs

Select naturally short and longer references from the same writer; generate the same target from each. Use generic length strata only for evaluation analysis, never to change model behavior. Record unavailable strata explicitly. Reference length is not a perfect proxy for information; include pixel widths and reference text in outputs.

This adds 64 generated images, for a default total of 320. Save a compact contact sheet and per-example predictions. Include targets with repeated letters and naturally occurring rare characters without changing their frequency in the main benchmark.

### 4. Readability metrics

Report only corpus CER = sum character edit distances / sum target characters and exact-word WER = fraction of unequal strings (word-level dataset). Keep case/punctuation; do not lowercase, strip unknown characters or autocorrect words to improve results.

**Teacher diagnostics:** not added under the final two-metric-only scope. The GAN's frozen R is not an independent evaluator.

**Independent OCR:** use a recognizer that never supplied GAN gradients. Same weights, preprocessing and decoding for all compared models. Calibrate on matched real images and cache real OCR results by dataset, manifest, recognizer hash and preprocessing revision. Do not feed target text/lexicon hints to the evaluator.

A separately trained compact CRNN can be used for epoch evaluation. A public TrOCR recognizer is an alternative for bounded offline local evaluation; first calibrate its text-line-to-word crop behavior on real images and disclose checkpoint/split provenance. No independent OCR checkpoint is currently established by the branch pretrained folders; their R is a training teacher. Never silently fall back to it under an `independent` metric name.

If the independent evaluator is unavailable, mark those metrics disabled/missing with a clear startup note; teacher metrics and image exports can still run. Do not fabricate empty OCR results as zero error.

## Confirmed evaluation bug that must be fixed

`lib/alphabet.py:decode` is a CTC-path decoder by default. GAN and pretrain validation call it both on literal target labels and on predictions that `ctc_greedy_decoder` has already collapsed. In the inspected MAIN/ALPHA code:

| True transcript | Current target decode |
|---|---|
| letter | leter |
| bookkeeper | bokeper |
| 11 | 1 |
| /.. | /. |

A correctly decoded CTC path for `letter` is collapsed again into `leter`. This makes the current OCR scores unsuitable as a new reference without correction.

Fix at transcript decoding/evaluation ownership: separate literal label-to-text conversion from CTC path collapse. Ground truth is mapped literally using valid label length; predictions collapse CTC paths once and then map IDs literally. Do not globally change decoder semantics and risk unrelated existing callers. Audit GAN and OCR pretrain validation in both target branches; record metric revision so old pretrain scores are not silently comparable. No retraining is required merely to correct evaluation decoding.

## Runtime and resource behavior

- Streaming generation and OCR; no complete generated dataset duplicated on GPU or host.
- Training batch stays 8. Default diagnostic batch at most 4, configurable down to 1 for local evaluation.
- FP32 only. Preserve configured cuDNN behavior; no global optimization switches added.
- Local standalone process loads only G/E/B and the selected evaluator, not D/P/optimizers for inference. Prefer CPU checkpoint deserialization and release unused state before moving modules to GPU.
- Independent large OCR is not placed permanently alongside two near-full training processes. Standalone local/offline evaluation or an explicit compact epoch evaluator avoids this memory cliff.
- Record elapsed generation/OCR time and peak allocated/reserved CUDA memory. Do not promise a runtime before measuring on the actual GPU.
- Use model eval/inference mode and restore exact module train flags, active EMA selection and all process/sampler RNG state after the epoch diagnostic. Evaluating must not change subsequent training draws or teacher BN statistics.
- The standalone tool accepts CPU or explicit CUDA device; bounded CPU mode must not accidentally launch the full dataset.
- Keep full existing official evaluations as configured. Default new 320-image diagnostics per epoch; independent OCR frequency depends on evaluator size and measured cost.

## Outputs and integration

The same evaluation implementation and protocol in MAIN and ALPHA. Extend existing validation/CSV/W&B seams, not a second competing training framework. No paper-specific protocol names in model code or forced benchmark comments.

Artifacts per run: deterministic manifest with fingerprints; compact metrics JSON/CSV with evaluator identity and disabled fields; per-example target/prediction/edit counts; optional small image sheet; runtime/memory summary. W&B plots use epoch axes and explicit teacher versus independent metric names. Existing best-FID-only checkpoint selection remains unchanged.

Standalone command must accept checkpoint, config, dataset location, manifest/output directory, device, batch size and independent recognizer path/model choice. These are requirements, not commands that currently exist. Notebook launch/status/monitor behavior and existing commented cells stay unchanged unless a minimal invocation is separately needed.

## Verification acceptance

1. Literal transcripts retain repeated letters, spaces, digits and punctuation; blank padding is not text.
2. CTC `[a,a,blank,a]` decodes to `aa`, not `a`; perfect repeated-letter predictions score zero errors.
3. Counts, denominators, fraction units and empty-string policy are covered by focused deterministic tests.
4. Two branches select exactly the same manifest; split/writer/text exposure classifications are validated, including absent or insufficient OOV cases.
5. Both heights support global-only and local conditioning without unused local-parameter gradients or unintended strength/divisor changes.
6. Diagnostic eval preserves RNG/model flags and official FID/KID/HWD behavior; unavailable independent evaluator cannot masquerade as a valid result.
7. Strict checkpoint/config consistency checks: never identify architecture from a renamed filename alone.
8. CPU tiny tests and bounded real-checkpoint inference first; CUDA smoke only if resources are available and explicitly safe. Full Kaggle runtime and quality remain unverified until a native run.
9. Temporary tests/results are ignored and remain outside tracked model code where feasible; no generated test artifacts or reports auto-committed/pushed.

## Review checkpoint

Approve this design before implementation planning. In particular, the next run is **MAIN current four-token control versus ALPHA global-only**, not a simultaneous extractor-plus-prior modification. No claim of zero remaining bottlenecks or guaranteed metric gains is made.
