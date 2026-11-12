# Repository workflow

- This independent branch is the downstream OCR augmentation benchmark, not a GAN training branch. Do not modify sibling model repositories or their running jobs.
- Active adapters are SpiS-GAN, FW-GAN and HiGAN+. Add own-model adapters only when requested after their training is complete.
- After approved changes, commit and push `ocr_benchmark`, then verify fresh remote HEAD. Never force-push or mix unrelated changes.
- Stage only intended source/config/notebook files. Do not commit checkpoints, datasets, generated tests, run artifacts, source clones or credentials.
- Keep dated design, implementation, verification and limitations under shared `HGGAN_test/agent` and publish documentation separately.
- Test-first, bounded CPU/native-checkpoint checks before large runs. Passing small tests is not measured two-T4 fit, a finished Kaggle session or an OCR result.
- Keep real test samples out of synthetic-data generation, OCR training and checkpoint selection. Preserve literal transcripts and raw test rows.
