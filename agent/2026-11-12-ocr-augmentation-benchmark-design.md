# Paired IAM synthetic-augmentation benchmark

Date: 2026-11-12. Updated: 2026-11-14.
Status: approved design; implementation and final review complete at source
`f2f8498cabfac095bcc4d5e7586f1b9133a54030`. No full OCR benchmark is claimed.

User amendments after approval: active generator pool is only SpiS-GAN, FW-GAN
and HiGAN+ until own-model training finishes. Datasets must be independently
downloadable after generation. The user selected one IAM32 comparison with an
explicitly labeled HiGAN+ native64→export32 conversion. These decisions override
the broader adapter inventory and native-only default described below.

Version 1 scope: only the pinned `trocr-small-stage1` revision and FP32 are
implemented. The alternate OCR bases and optional FP16 discussed below are
deferred extensions, not available configuration options. All current arms use
the same base and precision; no experiment silently switches either.

## Goal and decisions confirmed by the user

Create an independent `ocr_benchmark` Git branch checked out at
`/home/quq/machineLearning/HTG/HGGAN_test/ocr_benchmark`.
It is a downstream recognition-utility experiment, not an evaluation with the
GAN's own recognition teacher. Existing model branches and notebooks stay intact.

The user selected:

- Train each recognizer on X real IAM samples plus at most 25,000 generated samples.
- Include a real-only control using the identical X real samples.
- Evaluate on a separate real IAM test set.
- Save the generated datasets before OCR training.
- Run independent TrOCR jobs, one per T4, rather than one distributed OCR job.

The chosen implementation approach is a small file-driven pipeline with separate
generation adapters and one shared OCR trainer. The alternatives considered were
on-the-fly GAN augmentation (harder exact resume and paired-input verification)
and two-GPU distributed OCR training (fewer simultaneous comparison arms).

## Paper provenance and reproduction limits

[SpiS-GAN section 4.5 and Table 3](https://arxiv.org/html/2607.06949v1#S4.SS5)
describe a fixed pool of 5,000 real IAM images plus 25,000 synthetic images for
each competing generator. The synthetic lexicon is derived from that real pool.
The recognizer follows TrOCR. Reported metrics are CER, NED and WER.
[FW-GAN section 4.4](https://arxiv.org/html/2508.21040v1#S4.SS4)
describes the same broad augmentation experiment.

Consequently the default will be `real_count: 5000` and
`synthetic_count: 25000`, with the former configurable and the latter bounded.
This matches the paper's experiment structure. It is NOT an exact numerical
reproduction: the papers do not provide exact OCR split IDs, checkpoint variant,
initialization, training recipe, decoding settings, or an unambiguous NED formula.
Our choices will be explicit and the report will separate published numbers
from results measured by this pipeline.

Do not reuse SpiS-GAN's `gen_random_images` defaults: its shipped configuration
selects test references, and its code samples an external lexicon rather than
implementing the paper's frozen 5,000-image pool. Test references must never be
used to generate OCR training samples in this benchmark.

## Data boundaries

1. Default cohort: native IAM 32px, because the available SpiS-GAN/FW-GAN
   checkpoints are native 32px. A separate 64px cohort is supported.
2. Local source files are `iam/iam_train_32.hdf5` and
   `iam/iam_test_32.hdf5`, or their corresponding 64px files.
3. Test remains the supplied IAM test HDF5. It is not randomly split from generated
   images and is never used for target sampling, references or OCR selection.
4. When a separate real validation file is available, accept it explicitly.
   Otherwise deterministically reserve 10% of training writers for real validation,
   using only writer IDs within the training file. Store exact row indices.
5. Select X real training rows from the remaining training partition using a fixed
   seed and writer-balanced sampling. Synthetic references come ONLY from these
   same X rows, and target texts come ONLY from their lexicon.
6. Compute file SHA-256, image-only hashes and image-plus-transcript fingerprints.
   Preserve the complete supplied test set, and reserve validation writers before
   selecting the real pool. Exclude exact image-plus-transcript fingerprint
   groups present in test or validation from eligible training/reference rows;
   record exclusions before sampling X. Do not rewrite the source HDF5 files.
   This is duplicate separation, not OCR- or appearance-based quality filtering.
   Image-only collisions with different transcripts are reported as label/image
   ambiguity, not silently removed. Packed files lack original IAM sample IDs:
   do not claim official line/form split equivalence or cross-file writer-ID
   identity from local integer IDs.
7. Do not threshold, denoise, repair or remove difficult examples based on OCR,
   foreground appearance or generator quality. Invalid storage or geometry fails
   validation. Any explicit split policy is recorded, not hidden.
8. Existing GAN checkpoints may have been selected using test-set FID in past
   experiments. This history must be disclosed; freezing GANs here prevents new
   OCR-test tuning but does not retroactively create an untouched GAN test set.
9. Record each GAN's original training-data/teacher provenance when known. A GAN
   trained on the full IAM training set has more prior data exposure than the
   downstream recognizer's selected 5,000 rows. This follows the practical
   augmentation question, not a claim that the entire system used only 5,000
   labeled examples. Unknown GAN provenance must remain marked unknown.
   Known test-trained checkpoints are excluded from the primary comparison;
   an explicit diagnostic opt-in may run them with a test-exposure label.
   Unknown provenance is allowed for exploratory arms but visibly flagged in
   reports and cannot support a strict equal-supervision claim. Validation
   carved from the old GAN training file is held out from the new OCR trainer,
   not necessarily unseen by the upstream GAN or its teachers.

Current inventory: 32px train/test contain 66,603/25,766 rows; 64px contain
63,186/25,432. Therefore these are not the exact same rows at different scales.
Each cohort gets its own split and manifest hashes. Never identify a 64px row
with a 32px row just because their row indices match.

The read-only duplicate audit found 6 shared image-plus-transcript fingerprints
in 32px (48 train rows, 35 test rows), and 5 in 64px (43 train, 29 test).
All these shared images are constant-zero arrays, with punctuation or short
transcripts. They do not establish duplicated original IAM identities; the
conservative fingerprint separation above avoids putting the same input/target
pair in OCR training and testing, while retaining every difficult test example.

## Matched generation contract

A single immutable JSONL manifest contains exactly N generation cases:
`sample_id`, `reference_row`, reference image/text hash, `writer`, `target_text`,
and `seed`. A metadata file binds it to the real-pool/split/data fingerprints,
alphabet intersection, text-length constraints, sampling policy and version.

All participating generators receive the exact same reference images and target
texts in manifest order. Restrict target characters to the declared common
alphabet and report excluded real-pool lexicon entries. Do not silently replace
unsupported characters. Sample target text independently of reference content,
excluding a target identical to that reference transcript to avoid deliberate
same-word reconstruction. Repeated words across different references are valid;
"25,000 new samples" does not mean 25,000 unique English word types.

Use per-record RNG seeds, not only one process seed. Resume, batch-size changes
and sharding cannot change the intended random draws. Models with noise use
their own native noise dimensionality/distribution; a common seed is not a claim
that incompatible latent vectors are identical. Deterministic reference encoders
remain deterministic. Use the native inference path, not a new noise mechanism.
Numerical equality across different CUDA/library versions is not promised.

Generation runs in separate subprocesses to avoid collisions among external
repositories' top-level `networks` and `lib` packages. Each adapter constructs
only G and required style encoder/backbone, in eval/inference mode. It must not
construct D, PatchD, R, W, GAN optimizers or GAN data loaders.
"Generator only" includes the modules needed to encode a reference style.

Source revisions, local modifications, model configuration, checkpoint SHA-256,
raw-versus-EMA policy and required-module keys are recorded. Prefer a complete
available EMA G/E pair; never mix raw and EMA components silently. Frozen
backbones are loaded from the matching checkpoint. Strict state loading is the
default; errors are not hidden with `strict=False`. Own checkpoints' saved model
configuration takes precedence over a current YAML's possibly different geometry.
External checkpoints need an explicit pinned compatible config.

Known source compatibility trap: the November 9 ALPHA checkpoint lacks the
ContentContext introduced in current ALPHA. It requires the pre-context source
revision `0c435bdc41b29119b53d64f47960716f299b5cd9`, not current ALPHA with missing
weights ignored. Existing local SpiS/FW files also have compatibility edits, so
record effective-file hashes/patch provenance, not Git HEAD alone. Do not modify
or revert those source repositories as part of this pipeline.

One `.py` entrypoint and one YAML per adapter under `generators/` and
`configs/generators/`: `main`, `dev`, `alpha`, `classic`, `spis_gan`, `fw_gan`,
`higanplus`. Shared functions avoid copy-pasted export code. Configuration exposes
source root/revision, checkpoint, native geometry, polarity and output cohort.
Other GANs can later implement the same small adapter interface.

Native-resolution comparisons are the default. Original available HiGAN+ is
64px; it belongs in a 64px cohort. Running that checkpoint in a 32px cohort
requires an explicit declared height-conversion option and a separate report
label. Do not pretend downsampling is native 32px HiGAN+ or silently upsample
other generators. Native 64px SpiS/FW checkpoints are required for a fully native
64px comparison; the currently supplied files cannot establish that comparison.

## Dataset output and resume

Output is IAM-compatible packed HDF5 with `imgs`, `img_seek_idxs`, `img_lens`,
`lbs`, `lb_seek_idxs`, `lb_lens`, and `wids`. Images are uint8 white-paper /
black-ink with true valid widths, without batch-padding tails. Model-specific
polarity conversion is an explicit adapter boundary, not image thresholding.
Additional provenance goes into attributes and sidecar metadata.

Use small atomic, indexed generation shards as the resumable intermediate; only
complete shards count. The final HDF5 is streamed, verified for exact record
count/order/transcripts/widths, then atomically renamed. Never append repeatedly
to an unverified partial packed file. Dataset completion requires the manifest
hash and all expected sample IDs, not just an existing filename.

Preserve native generated width semantics; do not force images to the reference
word's width. The identical OCR processor later sees canonical RGB images from
all real and synthetic sources. Save a small fixed contact sheet, target text
and reference for human checking, without using OCR scores to cherry-pick data.

## TrOCR training and metrics

Each arm starts independently from the identical pinned
[`microsoft/trocr-small-stage1`](https://huggingface.co/microsoft/trocr-small-stage1)
pretrained-only weights and processor, not `trocr-small-handwritten`.
The latter is already fine-tuned for handwriting and introduces an avoidable
IAM-supervision confound. This is fresh task fine-tuning, not random training of
all 61M parameters from scratch. Support an explicit alternate base model/revision
but do not compare arms using different bases as generator-only comparisons.

Proposed initial recipe, held constant across arms:

- Full TrOCR fine-tuning, native processor image geometry, AdamW, LR 5e-5,
  weight decay 0.01, gradient clipping 1.0, 5% warmup and linear decay.
- Fixed 10,000 optimizer-update budget, effective batch 32, seed 123456.
  Same update budget applies to real-only and augmented arms; report actual
  example exposures because their dataset sizes differ.
- Conservative FP32 default, microbatch 2 and gradient accumulation 16 per T4;
  configurable AMP FP16 with GradScaler only as an explicit all-arms setting.
  T4 does not support native BF16. No silent GAN precision changes.
- Real validation every 500 updates, checkpoint every 500, best selected by
  validation CER, then WER for ties. No OCR filtering of synthetic training data.
- Greedy decoding by default, `max_new_tokens: 64`. Tokenization checks fail on
  transcript truncation; generation length-limit hits are reported.
- Literal case/punctuation-sensitive corpus CER; token-edit WER; mean
  `edit_distance / max(len(reference), len(prediction), 1)` as explicitly defined
  NED. Also report exact-word accuracy, sample counts and raw edit denominators.
  No case folding, punctuation stripping or percent/fraction ambiguity.
- Stream predictions to CSV/JSONL and aggregate counts without retaining logits
  for the entire test set. Evaluate the full real test set only after best
  checkpoint selection. An interrupted final evaluation can restart safely.

The published paper leaves NED normalization unspecified, so do not promise that
our NED exactly reproduces its number. Store percentage-form report columns plus
raw fraction/count fields. A one-seed run is a pilot, not a significant ranking;
the runner supports the same repeated seeds across all arms for the final report.
Keep separate `split_seed`, `generation_seed` and `ocr_seed` fields. Default
repeats freeze the split, X real rows, generation manifest and generated datasets,
varying only paired OCR-training seeds. Their mean/std measures OCR-training
variance, not generation-resampling variance. Resampling augmentation is a
separately named experiment requiring a new manifest and dataset identity.

The final report mirrors the paper's compact layout, in a separate table per
height cohort: `Method | Real | Synthetic | CER (%) | NED (%) | WER (%)`.
Rows include real-only and each completed generator arm; missing, paused or
failed arms are clearly marked rather than given fabricated numbers. Sidecar
CSV/JSON adds validation-selected checkpoint/update, seeds, test sample counts,
exact-word accuracy, relative/absolute changes against real-only, native versus
converted generator geometry, provenance flags and measured resource use.
Published paper scores appear only in a separately labeled reference table,
never in the same measured-results column as our new runs.

## Kaggle two-T4 notebook and observability

One new standalone notebook in the benchmark branch; no replacement of the user's
existing GAN notebooks. Cells cover configuration/paths, pinned dependency setup,
data/source/checkpoint preflight, split/manifest preparation, generation queue,
two-GPU OCR queue, monitoring, and final results export.

Launch exactly one child process per visible T4 with its own
`CUDA_VISIBLE_DEVICES`, log, output directory and W&B run. Generation and OCR
training for an arm are separate stages; release generator memory before TrOCR.
No assumption that two 16GB GPUs form a 32GB memory pool. Keep source processes
isolated and cap CPU/data-loader workers per job.

Resume accepts a previous Kaggle output mounted read-only under `/kaggle/input`.
Copy only the selected state into writable `/kaggle/working`; preserve manifests,
generated shards/final datasets, optimizer/scheduler/scaler/RNG/trainer states,
best checkpoint, epoch/update metrics CSV, completion state and W&B run ID.
Verify experiment identity before resume. Changed data, seed, processor, source
checkpoint or training schedule must not silently continue the old experiment.

Use the supported Hugging Face trainer/checkpoint machinery rather than inventing
a second optimizer-resume mechanism. A session time budget requests a clean
checkpoint-and-stop before the Kaggle limit. Resume continues the same planned
10,000-update schedule, not a newly restarted learning-rate schedule.

W&B initializes before model loading so startup stats and errors are captured.
Support online/offline/disabled modes and API keys from Kaggle Secrets/environment,
never committed credentials. Stable run IDs bind to experiment identity; one
worker owns one run. Logs: losses, LR, real validation metrics, test metrics,
small fixed samples, resources and lineage. Avoid parameter/gradient histogram
spam and uploading every large checkpoint by default.

The monitor terminates when all child processes exit, checks return codes and
stage status, and raises on failure. A printed RESULTS table is for users, not
the sole completion detector. Distinguish complete, safely-paused and failed
jobs so a Save Version session stops instead of monitoring forever. Preserve
both best and resumable last checkpoints; exports are local even without W&B.

## Proposed source layout

```text
ocr_benchmark/
  AGENTS.md
  .gitignore
  requirements.txt
  benchmark.py                 # prepare / generate / train / evaluate / report
  configs/benchmark_32.yml
  configs/benchmark_64.yml
  configs/generators/*.yml
  generators/{main,dev,alpha,classic,spis_gan,fw_gan,higanplus}.py
  pipeline/                    # data, manifest, export, resume, OCR, logging, queue
  kaggle_ocr_benchmark.ipynb
```

Design/implementation/verification reports remain in shared `HGGAN_test/agent`,
published on its existing documentation branch. Generated tests live in an
ignored work area. No images, datasets, checkpoints, caches, W&B credentials,
logs or test artifacts are staged. The new branch contains the pipeline only,
not a second copy of every GAN repository. Pin source checkouts independently.

## Acceptance checklist and limits

- [x] Confirm the requested downstream benchmark, mixed-data/control arm and
  independent two-GPU scheduling with the user.
- [x] Inspect local IAM schema and current generator interfaces/checkpoint keys.
- [x] Verify paper protocol and identify unspecified details/leakage traps.
- [x] Approve this design and its explicit defaults.
- [x] Create isolated branch/folder and implementation checklist.
- [x] Implement matched train/validation/test boundaries and generation resume.
- [x] Implement/test every available adapter without discriminator loading.
- [x] Prove tiny end-to-end generation/export/OCR update/save/resume/evaluation.
- [x] Check notebook Run All control flow, process failure/clean pause handling,
  offline W&B and experiment-identity mismatch rejection.
- [x] Independent review, dated verification report, scoped commit/push and fresh
  remote-SHA check.

CPU tests and small available-checkpoint smokes cannot prove full 25k generation,
full test-set results, 2xT4 peak memory, throughput, live W&B or a completed Kaggle
Save Version session. A real T4 preflight is required before claiming measured
fit; a full benchmark needs multiple sessions/models and is not run implicitly
on the user's laptop. Native 32px/64px teacher/GAN retraining is outside this task.
