# OCR benchmark: compact reporting and a larger T4 batch (2026-11-19)

Scope: the downstream IAM OCR augmentation benchmark. The paired SpiS-GAN
and HiGAN+ exports, cohort, seeds, TrOCR initialization and optimizer recipe
are unchanged. No GAN or teacher was touched.

- The Kaggle default is `OCR_BATCH_MULTIPLIER=2`: train microbatch 32,
  accumulation 2, validation/test batch 64, one independent worker per T4.
  Effective train batch remains 64; both 30,000-row arms keep 32 epochs and
  15,008 optimizer updates. Precision stays FP32.
- The previous x1 setting (16/4/32) retains the `epoch32_t4` namespace and
  can resume the old matching checkpoint. x2 uses `epoch32_t4_x2` and starts
  a fresh TrOCR run on the restored datasets. x4 (64/1/128) is opt-in with
  its own namespace; do not use it as a mid-run resume setting. The CLI
  step-based presets and Save Version folder layout remain unchanged.
- Validation and final-test reporting expose WER, CER and NED. The
  validation-only checkpoint selector still minimizes CER, breaking ties
  with WER; final test still evaluates the entire real IAM test split.
  NED is mean per-item edit distance divided by the longer transcript
  length. The displayed metrics all use percentages. Final receipts retain
  counts and provenance for audit; old receipts remain readable without
  showing the old exact-match accuracy score.
- W&B OCR history contains `train/loss`, `train/lr`, and `val/{wer,cer,ned}_pct`.
  Completed test scores `test/{wer,cer,ned}_pct` go into the run summary.
  The SDK's System metrics remain available, with console streaming in
  Kaggle rather than duplicating it in W&B. Existing W&B history is not
  deleted or rewritten.

The prior 28% reading is observed usage, not a worst-case peak. x2 was
chosen over x4 as a conservative increase. Neither T4 fit, speed gain,
final OCR quality nor an online W&B sync was measured locally.

Verification: 167 CPU/fixture tests passed, including the paired-worker
run, native epoch checkpoint/resume and the three-metric report. Both
notebook copies pass schema/cell validation and are byte-identical. The
pinned W&B SDK passed an offline history and final-summary smoke. The
x1 effective config and run name exactly match the previous notebook.
