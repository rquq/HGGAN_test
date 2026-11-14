# OCR augmentation: evaluation-set evidence

Date: 2026-11-12. These are protocol findings, not new experimental scores.

## What the papers actually specify

[SpiS-GAN §4.1](https://arxiv.org/html/2607.06949v1#S4.SS1) and
[FW-GAN §4.1](https://arxiv.org/html/2508.21040v1#S4.SS1) describe a revised IAM
split, following HWT/VATr practice, with 339 training writers and 161 test writers.
They distinguish test-set replication from IV/OOV and seen/unseen-style
generative evaluations. Those generated evaluation images are not the real
held-out test corpus for a downstream recognizer.

[SpiS-GAN §4.5](https://arxiv.org/html/2607.06949v1#S4.SS5) and
[FW-GAN §4.4](https://arxiv.org/html/2508.21040v1#S4.SS4) describe augmenting a
fixed 5,000-image real IAM pool with 25,000 synthetic images per model and
retraining a TrOCR-style recognizer. The lexicon derives from that real pool.
The reported recognition metrics are CER, NED and WER.

Neither downstream-augmentation section supplies exact OCR evaluation sample IDs,
an independently specified official OCR split manifest, a complete validation
recipe, recognizer initialization/revision, decoding details, or a reproducible
training schedule. The general 339/161 split is evidence of intended writer
separation, not proof that an arbitrary HDF5 file is byte-identical to their
OCR evaluation data. Do not claim exact numerical reproduction from these details.

## Local declared protocol

- Final OCR test is the complete supplied `iam_test_32.hdf5`, or the corresponding
  64px file in a separate experiment. Never synthetic images.
- Real validation is an explicit independent file when supplied; otherwise a
  deterministic 10% writer holdout from the training HDF5.
- Select the fixed real training pool only from remaining eligible training rows.
- References and target lexicon for synthetic training data come only from that
  real pool, not test or validation.
- Canonical image+transcript duplicate groups are separated before pool selection;
  the source datasets and full final test set are never rewritten.
- All arms use identical real rows, manifest, test rows, OCR initialization,
  training budget, validation-selection policy and decoding.

The local files have the expected writer counts (339 train, 161 test), but writer
IDs are remapped independently within files (train 0..338 and test 0..160).
Without original IAM identifiers, overlapping integer IDs are not evidence of
overlapping real writers, and counts alone do not prove the official split.

The 32px files contain 66,603 train and 25,766 test records; 64px contain 63,186
train and 25,432 test records. Comparing their OCR scores measures both rendering
resolution and dataset membership differences. It is not a pure resize ablation.

GAN checkpoints may already have used IAM test FID for selection. The new OCR
trainer must not use test labels for training or selection, but this does not
make those historical GANs entirely test-unexposed. Report upstream provenance.

## Public generation-code warning

The inspected SpiS-GAN checkout is pinned at
`6949fc2e128d76d8aba6ce90af4d1a4683b21789`.
`networks/model32.py:675` uses `opt.valid.dset_split` in `gen_random_images`;
the IAM configuration selects `test`. Its target text source is an external
English lexicon. That helper is not a faithful implementation of the downstream
5,000-real-pool protocol and must not be called unchanged to create OCR training
data. The new benchmark uses explicit train-only references and a frozen manifest.
