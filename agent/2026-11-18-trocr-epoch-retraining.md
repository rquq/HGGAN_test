# Epoch-based TrOCR retraining — 2026-11-18

This is a new OCR recipe for the SpiS-GAN and HiGAN+ augmentation arms, not
continuation of the 10,000-update experiment. Reuse the exact paired datasets
and prepared IAM split from results (8).zip. Both models start from the pinned
TrOCR-small-stage1 initialization. The real-only control is deferred.

Microsoft's published Fairseq IAM example for Base/Large uses 384px input,
Adam, LR 2e-5, 500 warmup updates, inverse-square-root decay, weight decay 1e-4,
FP16, batch 8/device on eight GPUs, 300-epoch cap and patience 20. It also uses
DA2 preprocessing and beam 10 in its evaluation example:
https://github.com/microsoft/unilm/blob/master/trocr/README.md#fine-tuning-on-iam

The new benchmark borrows the LR, decay shape, absolute warmup count and weight
decay, while keeping the stage1 Small model, AdamW, FP32 and greedy decoding
for the controlled comparison. It is not an exact reproduction of stock
Fairseq or of the paper's data/patience/decoding schedule.

- 32 full epochs, each on 5,000 real plus 25,000 synthetic rows.
- Train microbatch 16, 4 accumulation steps (nominal effective batch 64),
  validation and test batch 32. One independent arm per T4.
- 469 optimizer updates per epoch, 15,008 total per arm with the last
  partial batch retained. 960,000 training-row presentations per arm.
- Validation and native checkpoint saves every epoch; cooperative budget
  pause still saves at an optimizer boundary, including mid-epoch.
- Best selected by real validation CER/WER, final test on held-out IAM only.
- Logs from first training update, then every 25 updates, plus validation
  and final-test progress. CUDA logs include peak allocated and reserved VRAM.
- Set only max_length=65 for the one-token decoder start + 64 generated
  tokens. This retains the prior cap without competing length arguments
  or global Transformers warning suppression.

`PREVIOUS_OUTPUT_DIR` accepts the previous output tree or single archive.
The notebook defaults to OCR-only with `RUN_REAL_ONLY=False`. The new
checkpoint suffix `_epoch32_t4` leaves the old seed_123456 folders intact.
Do not change batch size, epoch count or recipe to continue that namespace;
use a different name for a fresh recipe. Old step-based CLI configs remain
supported. The old real-only score is not a matched control for this recipe.

Larger batches reduce the relative overhead of launching tiny batches, but
W&B VRAM usage is not GPU compute utilization. Batch 16/32 is a starting
preset, not a measured T4 fit or 90–100% utilization guarantee. More than one
Kaggle session may be needed; restore the new resume bundle on continuation.
No generation model, data cohort, archive format or teacher is changed.

Verification: 163 CPU/fixture regression tests passed, including native tiny
TrOCR epoch-based validation/checkpoint scheduling, mid-epoch pause/resume,
final-test reporting, and legacy step-mode coverage. The longer separate
all-stage fixture was not run. Both notebook copies are byte-identical and pass
schema and cell compilation. No full-size or dual-T4 run was performed.
