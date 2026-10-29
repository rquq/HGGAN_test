# Full configured evaluation-set OCR

Date: 30 October 2026, Asia/Saigon. User explicitly requested replacing the sampled OCR monitor with the same configured evaluation dataset used by other metrics, a full local four-source run, and commits/pushes. ALPHA is to remain local because its history diverges from origin.

## Correction

The old training CER/WER path used a320-case mixed training/test-reference transfer diagnostic, not the complete configured evaluation dataset. The current GAN configs point `valid.dset_split` to `test`; IAM32 is `/home/quq/machineLearning/HTG/iam/iam_test_32.hdf5`,25,766 records. It is the configured evaluation split, not a newly created official validation partition. No dataset split was fabricated or renamed.

DEV, MAIN, and ALPHA now supply OCR with `get_cached_generator()` and the existing `eval_dloader`. Thus OCR scores the exact cached `org_imgs`/valid widths produced for distribution/HWD metrics, including same-transcript guided synthesis and restored original word widths. It does not regenerate stress words, read training references, or invoke a320-case manifest. CLASSIC uses its existing full evaluation generator/loader and the same independent evaluator instead of its configured training R. No architecture or GAN loss was changed.

## Implementation

- Shared `metric/ocr_eval.py:evaluate_validation`: all batches, literal transcript decoding (repeated letters preserved), true-width crops, independent pinned TrOCR, CER and isolated-word WER, real-image calibration, full prediction CSV, protocol metadata and real-OCR caching. Mismatched stream lengths/transcripts and incomplete declared dataset coverage fail explicitly.
- Real-calibration cache keys include dataset/evaluator identity and actual image pixels/transcript. Matching real/fake pixels can reuse the exact prediction without a second identical OCR pass.
- DEV/MAIN/ALPHA training uses the full cached metric stream. Existing metric switches and cadence remain; CPU fallback still evaluates once per process. Missing OCR dependencies/OOM leave cells empty with a message, never switch to training R or report zero.
- Obsolete per-cell/reference-pair/writer-strata configuration knobs were removed from the three GAN configs in DEV/MAIN/ALPHA. Existing learning-rate/loss/model settings were preserved.
- DEV/MAIN/ALPHA `tools/evaluate_ocr.py` now defaults to full test-set OCR and the actual same-text `AdversarialModel.image_generator`. The old sampled new-text tool requires explicit `--transfer-diagnostic`. `--limit` is rejected for default full-set evaluation so a bounded smoke cannot masquerade as complete validation.
- The new protocol is `full-validation-same-text-v1`. Earlier CSV/W&B CER/WER came from a different task and are not directly comparable to new values. Old histories are preserved; model weights/checkpoints remain compatible, with no GAN or teacher retraining required merely for this evaluation change.

## Verification

- Test-first failure reproduced missing full-loader API; four focused tests now pass in each of the four branches (16 total): more-than320-record coverage, repeated literal labels/true-width cropping, unequal stream rejection, mismatched transcript rejection, and declared dataset coverage checking across the test methods.
- Existing42-test current contract suite passes in DEV and MAIN (84 total).
- Actual epoch71 DEV checkpoint, EMA G/E/B, native32, full-loader CLI path, independent cached TrOCR: an eight-record temporary HDF5 fixture completed with protocol and coverage assertions. This is a wiring/runtime smoke, not the requested full benchmark result. Temporary fixture/output were removed normally when its context closed.
- AST/YAML and `git diff --check` passed for the scoped files. No claim of a new full Kaggle/T4 run or all historical tests passing.
- No changes to pretrained weights, dataset files, FID/KID/HWD algorithms, or model dimensions.

## Full local comparison: running, not yet complete

Runner: `agent/tests/full_test_ocr_20261030.py`; outputs under `agent/analysis/2026-10-30-full-test-ocr/`.

- All25,766 IAM32 test records, no removal of38 all-zero images.
- DEV best-FID epoch71 EMA, SpiS-GAN epoch98, HiGAN+ deployment, and actual IAM.
- Sequential CPU FP32, two threads, fixed sampling seed, batch32 generation, OCR batch2, native-width restoration using the production rescaling rule. HiGAN+ native64 outputs use the existing documented64->32 adapter.
- HiGAN+ cannot encode records9423 (`*?2s`) and23769 (`cafe*?2`). Both remain explicitly reported as unsupported; no character substitutions. Report source-specific full coverage and a common25,764-supported-record comparison.
- Sharded OCR checkpoints allow continuation without losing a completed model's OCR. Output predictions are complete and uncensored; sample illustrations take two examples each from length1–2,3–5,and6+ plus the known black record. Those are display categories, not a sample budget for metrics.
- The runner automatically writes `results.json`, `REPORT.md`, `samples.png` and `samples.pdf` when all sources finish. Until then, `run.log`/`progress.json` are progress only, not final scores.

The full CPU workload is an hours-long job. Starting it does not establish full results; no earlier320-case result is reused as a full-set score. Dataset/checkpoint/model architecture is unchanged by the comparison.

### User-approved GPU switch

After the user requested using the local7GB-class GPU, the owned CPU controller/child were stopped after confirming their exact command lines.13,664 partial DEV CPU output PNGs were preserved recoverably under `dev-cpu-partial/`. A fresh GPU DEV generation was started rather than mixing CPU/GPU generator samples into one completed run. IAM references remain the actual source pixels.

RTX4060 Laptop GPU:8,188MiB physical reported VRAM; initial57°C and about5,080MiB free in nvidia-smi. Bounded sequential FP32 generation uses batch8; OCR batch2. The PyTorch allocator cap is2GiB per phase; GPU OOM or>=82°C triggers CPU continuation/retry. Other GPU processes are not touched. Initial active-run observation: about3.3GiB total device use,4.6GiB free,61–68°C. These are observations, not a guarantee against driver/thermal failures.

The restarted controller PID is129623. The runner reuses fresh full-test real-image calibration across source models and for the IAM row; it does not reuse the old320-case diagnostic's scores. Full test character count is101,954, so189 edits contribute0.18538 CER percentage points. No full-set result exists until the job finishes.

## Version-control scope

DEV/MAIN changes comprise the evaluator, its training call, standalone tool, and obsolete OCR config knobs. CLASSIC changes only evaluation integration and the shared evaluator file. ALPHA staging must include only this evaluation patch, preserving its unrelated pre-existing uncommitted architecture/training edits. Do not force-push ALPHA. AI tests, generated images, large predictions and job artifacts stay outside model-branch commits.

Scoped code commits: DEV `97fbe7b`, MAIN `4b77b93`, CLASSIC `de0974c` plus RNG-isolation follow-up `23db62e`, ALPHA `a1831bf`. DEV, MAIN and CLASSIC were pushed and remote hashes verified. ALPHA commit is local only per explicit user choice. Its pre-existing unrelated working-tree edits were not staged into this commit. The full local job remains active; its eventual scores are not implied by these commits.
