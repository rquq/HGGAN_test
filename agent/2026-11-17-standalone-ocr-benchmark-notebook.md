# Standalone OCR augmentation notebook — 2026-11-17

Canonical user-facing file: `notebooks/kaggle_ocr_benchmark.ipynb`.
The benchmark branch contains the byte-identical `kaggle_ocr_benchmark.ipynb`.

This is a complete separate Kaggle entry point, not a cell to add to a GAN
controller. Run only this notebook: it fetches the independent benchmark branch,
installs the pinned environment, downloads/reuses IAM, obtains generator source
and checkpoints, prepares/restores the cohort, generates saved paired synthetic
datasets and trains independent recognizers. No other notebook needs to run first.

## Sole task

Handwriting-generation OCR benchmarking here means data augmentation utility:
X real IAM + N synthetic samples → train TrOCR → real validation selects best →
complete held-out real IAM test. This is not frozen-reader scoring of GAN images.

Defaults remain DATA_AUGMENT=True, TRAIN_TROCR=True, SpiS-GAN and HiGAN+,
5,000 real rows and 25,000 synthetic samples per generator. All generation arms
share the frozen train-only reference/text/per-case-seed manifest. OCR arms are
real_only, spis_gan, higanplus; each uses a separate model/run directory and the
same initialization/recipe. Native64 HiGAN+ explicitly exports32 for IAM32.

## This pass

Expanded the independent notebook into eight clearly labeled steps:

1. Configurations and initial Kaggle checklist.
2. Isolated runtime, W&B and read-only GPU/disk overview.
3. IAM download/reuse and split roles.
4. Frozen generator source/checkpoint setup (not GAN training).
5. Previous-output restore, fixed manifest and read-only saved-OCR-job inventory.
6. Generation → independent OCR training queue → final real test, live log stream.
7. Validated final results and saved reference/generated previews.
8. Download links, separated output paths and continuation instructions.

Checkpoint inventory is only an overview of saved folders/receipts; the pipeline
still owns full identity/state validation before continuation. Preview display
does not filter generated data. No architecture, loss, training recipe or worker
logic was changed in this pass. Existing GAN/pretraining notebooks were confirmed
byte-identical to their published versions.

## Continuation

Outputs stay under /kaggle/working/ocr_benchmark_outputs. Keep prepared/,
datasets/, and ocr/<arm>/seed_<seed>/ with full checkpoint-N directories. Attach
that output next session and set PREVIOUS_OUTPUT_DIR to its benchmark root.
Optimizer, scheduler, RNG and validation-selected best resume through the existing
pipeline. Dataset ZIPs alone do not include OCR training state. OCR-only
continuation uses DATA_AUGMENT=False, TRAIN_TROCR=True and needs no GAN weights.

## Verification and limitations

- 31 focused tests passed in 21.05s: notebook bootstrap/configuration, queue/status,
  checkpoint transport/resume and new standalone utilities.
- Both notebook copies pass nbformat validation and every code cell parses.
- Copies are byte-identical, settings remain end-to-end by default.
- New inventory exercised on empty and paused fixture outputs without creating
  training folders; absent previews do not fail.
- No full-size GPU generation, fresh installation, 25k export, complete TrOCR run
  or two-T4 Kaggle Save Version result was measured in this pass.
- Session limit remains cooperative; retain time for saving and continue paused
  jobs next session. Completed benchmark scores require completed final tests.
