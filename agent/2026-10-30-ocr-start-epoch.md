# GAN OCR start epoch — 2026-10-30

Added `valid.num_start_ocr: 80` to GAN configurations in DEV, MAIN, ALPHA and CLASSIC_OPTIMIZED. Automatic independent CER/WER evaluation skips epochs below 80; epoch 80 is eligible. Existing enable switches and OCR cadence remain in effect. Missing settings default to 80, and negative values are rejected.

This affects GAN evaluation only, not OCR or writer pretraining, GAN OCR supervision, FID/KID/HWD, or standalone evaluations without a training epoch. Skipped OCR scores remain absent rather than being recorded as zero. No architecture or checkpoint tensor changes are involved. Restart a running training process with the updated code/config to activate this setting.

ALPHA changes are committed locally only under the existing instruction not to push its divergent history; unrelated local modifications are preserved and excluded from the commit.
