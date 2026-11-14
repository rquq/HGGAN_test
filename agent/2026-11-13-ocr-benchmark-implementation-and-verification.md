# IAM synthetic-augmentation benchmark: implementation and verification

Started: 2026-11-13. Updated: 2026-11-14. Working branch: `ocr_benchmark`.
Status: all four implementation tasks delivered and final review passed at
`f2f8498cabfac095bcc4d5e7586f1b9133a54030`. This is an implemented pipeline,
not a completed OCR benchmark. Publication follows the verification gate below.

## Purpose

Measure whether a generator supplies useful additional OCR training data, rather
than only producing a low FID or an easily reconstructed reference word. Each
arm fine-tunes a separate recognizer from the same pretrained-only TrOCR base.
One real-only control separates the value of synthetic augmentation from the
recognizer's own baseline performance.

```text
Supplied real IAM training HDF5
            |
       writer holdout ------------------------> real validation
            |
       fixed X real rows
            |--------------------------> same X real OCR training rows
            |
   frozen reference/text/seed manifest
            |                 |                    |
        SpiS-GAN            FW-GAN              HiGAN+
        native32            native32         native64 ->32
            |                 |                    |
       IAM HDF5 + ZIP     IAM HDF5 + ZIP       IAM HDF5 + ZIP
            |                 |                    |
   separate fresh TrOCR fine-tuning: X real + 25,000 synthetic
            |                 |                    |
      select best on the SAME real validation set
            |                 |                    |
          evaluate on the SAME complete real IAM test HDF5
```

Only SpiS-GAN, FW-GAN and HiGAN+ are active. MAIN/DEV/ALPHA are not adapters in
this version: the user wants to finish their training first. Existing model
branches and their two training notebooks are not modified.

## Fairness and dataset boundaries

Default X is 5,000, configurable up to available eligible training data.
Synthetic N is configurable from 1 to 25,000. References and the synthetic
lexicon are drawn only from the selected real training pool. No target word is
identical to its own reference transcript. The same case IDs, references, target
texts and per-case seeds feed every generator; no generator gets extra text
augmentation, quality filtering, OCR rejection sampling, thresholding or denoising.

The default holds out 10% of training writers for validation. Local writer IDs
from a separately supplied validation file are not assumed to be globally
identical to the training file's independently remapped integers. Exact
image-plus-transcript duplicate groups are excluded from eligible training when
present in validation/test; validation/test collisions are removed from validation
only. The supplied final test remains unchanged, including difficult constant
images. These fingerprints do not establish original IAM sample identity.

Actual IAM32 preparation was exercised read-only against the user's files:

| Item | Observed |
|---|---:|
| Real selected training rows | 5,000 |
| Real validation rows | 10,499 |
| Held-out training writers | 34 |
| Writers represented in selected training pool | 304 |
| Eligible target-word lexicon | 1,879 |
| Frozen generation cases | 25,000 |
| Full final test rows | 25,766 |
| Exact train/val, train/test, val/test fingerprint intersections | 0 / 0 / 0 |
| Same-reference-transcript target cases | 0 |

Preparation took 187.719s locally; immutable reprepare plus an independent full
fingerprint-boundary audit took 1.47s. Manifest SHA-256:
`7359604e0099289aebe2dff9a335dcd3e517f893e5cadf00f64e6b44254380b6`.
This created a manifest and real subsets, not 25,000 generated images.

Known upstream limitations remain visible: frozen GANs can have different original
training corpora/teachers, some checkpoints were selected using IAM test FID,
and exact original IAM IDs are not available in the packed files. The comparison
is controlled downstream utility under declared frozen-generator provenance,
not proof of identical end-to-end supervision or a previously untouched GAN test.

## Generator-only inference and exports

Each adapter constructs only the generator plus required style encoder/backbone.
It does not construct D, PatchD, recognition teacher, writer identifier or GAN
optimizers. The same requirement applies to original HiGAN+'s package initializer,
which is bypassed because it imports the full GAN/metric stack.

The exercised frozen inputs were:

| Adapter | Official source revision | Checkpoint filename | Checkpoint SHA-256 |
|---|---|---|---|
| SpiS-GAN | `6949fc2e128d76d8aba6ce90af4d1a4683b21789` | `bestIAM.pth` | `e5094b78691d7d07ad854b5fc0748e054d385dd52f48cbcfa9020294b42c6b60` |
| FW-GAN | `8ec41ce253e6208ca0575c09934401a7d25c61d8` | `FW-GAN.pth` | `c4046f681292c261f78eb1af6f3381206653d1124ac9df54854b3f0eee426bfb` |
| HiGAN+ | `5d37de733f79e2a72a79c0f66e1865329ff56b4e` | `deploy_HiGAN+.pth` | `aa2fbdad65c21dddefff5675a11eb482d51622dbc3d60fd2a8bd9d79d37029b3` |

Local SpiS/FW checkouts contain an existing DropPath-import compatibility edit;
they were not reverted. Effective source hashes record local code in addition
to Git revision. The Kaggle notebook checks out clean official pins and installs
the declared `timm` dependency. These hashes establish which artifacts ran, not
the truth of every upstream dataset/teacher/test-exposure claim.
Portable metadata and result rows explicitly record upstream training data,
teachers, test-data training exposure, test-based checkpoint selection and
supporting evidence. Unsupported facts are `unknown`, with an exploratory
qualification, rather than assumed clean. A checkpoint explicitly declared to
have trained on test data is rejected in the ordinary comparison. Test-based
checkpoint selection is recorded separately and is not mislabeled as training
on test images. There is no extra diagnostic-override system in this version.

SpiS/FW use native encoded style96 plus per-case CPU-seeded Gaussian noise32.
HiGAN+ uses its native deterministic mean-style32 path. A common seed controls
native randomness; it does not make unlike latent spaces semantically identical.
Images export as uint8, white paper and dark ink, with valid target widths and
no batch-padding tail. The user explicitly chose the HiGAN+ native64-to-export32
bilinear bridge for the shared IAM32 comparison; it is never called native32
HiGAN+ reproduction.

Checkpoint, source revision, effective Python source, source YAML, model settings
and worker content bind generation resume. Weight loading is strict, CPU-mapped,
and weights-only, with a narrow NumPy-scalar metadata allowlist for legacy files;
there is no arbitrary-pickle or random-weight fallback. Complete shards have
hashed commit markers; unfinished shards are not completion evidence.

Each model exports `synthetic.hdf5`, `metadata.json`, the identical
`manifest.jsonl`, a reference/target preview, and `dataset.zip`. The four-file
archive is independently portable: OCR verification does not require the original
GAN source, checkpoint, cache shards or generation status file. Generator resume
still requires the original pinned provenance. Integrity hashes are not signatures
against a party rewriting all artifacts.

## TrOCR training, selection and continuation

Base: `microsoft/trocr-small-stage1`, revision
`d94c4048159050bb51e4105bad0a45dbfd5f3a02`, 61,596,672 parameters.
This is pretrained-only, not the already IAM-fine-tuned handwritten checkpoint.
Production uses the native 384x384 processor and literal transcripts.
Version 1 deliberately accepts only this pinned base/revision and FP32. The
design's possible alternate-backbone/FP16 extensions are deferred, not exposed
as working settings. Changing bases or precision would require a separately
versioned all-arms experiment and a new continuation identity.

Default recipe: full-parameter FP32 fine-tuning, AdamW, LR5e-5, weight decay0.01,
clip norm1, 5% warmup/linear decay, 10,000 optimizer updates, microbatch2 and
accumulation16. Effective batch32 is nominal; native incomplete epoch-tail groups
can be smaller. Equal update budgets do not mean equal epoch counts or data
repetitions for 5k real-only versus 30k mixed training.

Real validation every500 updates selects lexicographically by CER then WER;
an exact tie retains the earlier checkpoint. Real test inference only follows
the full configured update horizon and final validation. Final predictions and
metrics are streamed/atomically published only after every real test row finishes.
An interrupted final test restarts safely, rather than publishing a partial score.

Continuation restores complete native HF model/optimizer/scheduler/RNG/trainer
state and keeps best plus last checkpoints. Paths can be relocated into writable
Kaggle working storage without retaining stale absolute best-checkpoint paths.
The original LR horizon remains 10,000 on continuation, not remaining updates.
Canonical complete `checkpoint-N` saves are authoritative. Best/last aliases
and `best.json` are derived views and can lag after an interruption between a
native save and receipt publication. Restore handles preserved links, omitted
aliases and verified materialized copies, repairs only the writable destination,
and does not modify a read-only previous output. Safe rotated-away older alias
targets can advance; future, traversal, absolute or unrelated aliases reject.

An installed Transformers5.9 bug was reproduced: native Trainer raw state loading
does not apply DeiT's legacy encoder-key conversion. A narrow loader bridge uses
the supported `from_pretrained` conversion and strictly copies canonical weights
into the existing parameters; native optimizer/RNG/scheduler/data-skip mechanics
remain untouched. A pause after a nonzero encoder update now resumes to exactly
the same final model tensors and nested optimizer, scheduler and RNG states as
uninterrupted training. A zero-LR first step alone would not have proved this.

CSV derives from checkpoint-persisted log history, not unsaved runtime summaries.
Budget expiry can pause before the first validation or at the final update with
validation still pending, without inventing a best or doing an extra update.

## Metrics and reporting

- CER: total character edit operations / total reference characters.
- WER: total whitespace-token edit operations / total reference words.
- NED: mean sample character edit distance divided by max(reference length,
  prediction length,1).
- Exact word accuracy: exact literal transcript matches / sample count.

Case, punctuation, repeated characters and raw character spacing are preserved.
No target text is reconstructed through a normalizing tokenizer before scoring.
NED and all table percentages have explicit units; the papers do not provide an
unambiguous NED denominator, so numerical equivalence is not asserted.
Before publishing a completed score, reporting checks the saved experiment
against the requested cohort, test content, recipe, arm, seed and full update
horizon using read-only validation. An old pilot under a changed configuration
is marked incompatible instead of silently joining the comparison. Rows also
include real/synthetic counts, selected checkpoint/update, split/generation
seeds and percentage-point differences from the paired real-only control when
both tests completed. Unmeasured runtime/memory fields stay `unknown`.

W&B starts before model loading and supports online/offline/disabled modes with
stable job identities. Credentials are environment/Secrets-only. A real offline
SDK0.27 record scan confirmed startup prints in `output_raw` events. The lack of
an offline `output.log` text file is not itself missing console capture. Live
online Logs-tab behavior and offline-to-cloud sync have not been verified.

## Bounded verification, not quality results

The latest independent complete suite passed137 tests in52.79s, with12 known SentencePiece SWIG/W&B
dependency deprecation warnings and zero failures. It includes real subprocess
GPU-slot isolation, bounded notebook-cell execution, portable ZIP/restore checks,
native TrOCR nonzero-update continuation, and interrupted-save-before-receipts
recovery. Tests and generated artifacts remain ignored, not committed. Native
tiny DeiT+TrOCR tests use the actual pinned tokenizer vocabulary, not a mock
language model. Every notebook code cell compiles as ordinary Python; the
10-cell notebook passes schema validation. Python compilation and diff checks
also passed. This is bounded wiring evidence, not a fresh Kaggle execution.

| Native check | Hardware | Result |
|---|---|---|
| All3 real checkpoints, paired four-case HDF5/ZIP exports | CPU | Passed |
| Repeated seed / changed noise seed / one-character geometry | CPU | Passed |
| SpiS native32 `hello`, generator-only | RTX4060 laptop | Passed;108.77MiB peak allocated |
| FW native32 `hello`, generator-only | RTX4060 laptop | Passed;107.15MiB peak allocated |
| HiGAN native64 `hello`, generator-only | RTX4060 laptop | Passed;43.19MiB peak allocated |
| Full pretrained TrOCR384 forward+greedy | CPU | Finite loss;FP32;0 updates |
| Full pretrained TrOCR batch2 FP32 forward/backward/AdamW | RTX4060 laptop | One diagnostic update;1199.29MiB allocated/1374MiB reserved |
| New CLI queue, all3 real GANs, four matched cases each, offline W&B | RTX4060 laptop, one physical slot | All workers exited0;3 dataset ZIPs plus prepared ZIP; no OCR launched |

The one GPU optimizer update was not saved and is not a trained OCR result.
Its6.7s included startup and must not be extrapolated into T4 throughput.
Source samples retain faint SpiS punctuation and blobby FW punctuation: these
were observed and retained, not filtered to improve downstream scores.

The actual all-three CLI run used manifest SHA-256
`618e00f7f2573e315e360f01bfa5ba8eeb3a0657d7d5caeece3ba0c09dcac850`.
Its outputs were reopened and the monitor ended automatically. This is separate
from fixture-only two-slot tests: the local laptop has one GPU, not two T4s.

Not yet established: clean Kaggle venv install, two-T4
Save Version execution, measured T4 peak memory/throughput, full25k generation
per model, 10k-update OCR training, final OCR scores or SOTA ranking.
Checkpoint continuation is tested; future failures cannot be guaranteed absent.

## Final review and publication

The whole-branch review and its single scoped correction wave are complete.
All four final findings were addressed: current-experiment result validation,
explicit upstream provenance, contextual result rows/control deltas, and compact
split/ambiguity audits. The correction did not change GAN mathematics, RNG draw
order, selected data, image pixels, OCR recipe or notebook settings. Existing
prepared exports without the additive audit remain readable without rewriting.
Generator continuation hashes can change with new implementation/provenance;
completed legacy portable datasets remain usable and explicitly unknown.

Release source: `ocr_benchmark` at `f2f8498cabfac095bcc4d5e7586f1b9133a54030`.
Shared documentation is published separately on `agent`; no generated tests,
checkpoints, datasets, logs or source clones belong in either release commit.

The four recorded scope decisions and their costs are:

1. Keep a persistent standalone sibling checkout rather than a managed temporary
   worktree. Cost: separate checkout maintenance; no implicit merging into GAN code.
2. Restrict adapters to the user's approved SpiS/FW/HiGAN pool. Cost: own-model
   comparisons wait until their checkpoints are ready.
3. Use the user-approved HiGAN64→32 bridge for one IAM32 table. Cost: its output
   is resampled and cannot be claimed as native32 HiGAN+ reproduction.
4. Fix the pretrained-only OCR revision and FP32 for version1. Cost: optional
   alternate-base and FP16 speed experiments are deferred.
