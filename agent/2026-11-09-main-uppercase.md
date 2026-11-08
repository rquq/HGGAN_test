# MAIN uppercase generation: investigation and training correction — 2026-11-09

Source published and remote-verified: `d881b59fb2111f92b0a8a580cedd4ad1e6438672`.

## Exact checkpoint and reproduced failure

Input: `chkpoints/9_11_2026/last_kid_0.068900_main.pth`.
SHA256: `25d27c0931f5cc5ce2174431f6fd09f14e6409675c2fbb1edd7120901cf3bfdd`.
Saved epoch 56, FID 2.945582, KID 0.0689001, HWD 0.1072682.
The 56-row history records its best KID at epoch 53: 0.0626459, with
FID 2.719437. No full-set metrics were recomputed for this investigation.

Controlled CPU FP32 inference used strict EMA G/E and frozen B/R weights,
four references from different IAM test writers, three words in three case
forms, and two style sources (reference-guided and random). These 72 outputs
are a diagnostic cohort, not a benchmark. The internal teacher recognized:

| Target form | Exact matches |
| --- | --- |
| Lowercase | 24/24 |
| Title case | 23/24 |
| Full uppercase | 0/24 |

The image gallery itself confirms malformed uppercase strokes; this is not
merely a recognizer disagreeing with otherwise-correct uppercase images.
Gallery: `agent/analysis/2026-11-09-uppercase-history/uppercase_before.png`.

## What was ruled out, and what remains an inference

- Alphabet encoding retains literal case and contains all A–Z classes.
  The current sampler does generate uppercase, so the earlier suggestion
  that it might be absent would not be a supported diagnosis.
- Raw G and EMA G both fail. Switching only generator BatchNorms to training
  statistics does not repair the problem.
- Removing text attention destroys normal generation. Disabling its local
  mixer or halving both residuals does not restore accurate uppercase words.
  No such inference ablation was promoted into production.
- Uppercase embedding rows are not identical/dead: average pairwise cosine
  is about 0.255 versus 0.132 for lowercase rows. This does not establish full
  semantic disentanglement, but rules out complete identical-vector collapse.
- On a controlled three-word/zero-latent test, teacher CTC losses are 0.0133
  for lowercase, 0.0043 for title case, and 5.4250 for uppercase. Embedding
  gradient norms are respectively 0.2123, 0.0600 and 4.9040. The loss sees the
  defect and its gradient reaches the content path.
- A small calibration on the first 16 real uppercase training examples of
  lengths 3–8 returned all 16 labels correctly. The corresponding lowercase
  calibration also returned 16/16. This is not held-out teacher accuracy and
  includes repeated transcripts; it only rules out universal teacher blindness
  to uppercase. That filtered training population contains 169 uppercase
  versus 33,322 lowercase alphabetic records, not a full-IAM case census.

The checkpoint has a learned content-generalization failure. The evidence
does not prove one coding typo caused it, or uniquely isolate sampling,
adversarial pressure, context generalization and loss balance as causes.

## Implemented remedy: training coverage and existing content supervision

1. Added `training.uppercase_ratio: 0.5`, the fraction of the existing case
   augmentation allocated to full uppercase. With `capitalize_ratio: 0.5`,
   a lowercase lexicon word now has approximately 50% unchanged, 25% title,
   25% uppercase probability, instead of 50/40/10. This is a generic casing
   policy, not a rule for specific character IDs, words, writers or image
   resolutions. It applies symmetrically to D-step and G-step fake text.
2. Retain the initial OCR supervision late in training: random-content CTC
   stays at 1.0 instead of decaying to 0.4; transferred-content CTC stays at
   0.5 instead of decaying to 0.4. The stronger correction is on random style;
   transferred style gets only the modest 0.4→0.5 change. No OCR term was
   added to the reconstruction branch and no new loss was introduced.
3. Apply the case-mixture option to preview sampling so the affected case
   is visible during training. No uppercase-to-lowercase fallback is used.

The sampler's API default remains 0.2 for old callers. The validation
`image_generator` path was left unchanged, including its optional random
corpus distribution; default validation remains same-text on the full test
loader. No evaluation transcript normalization, metric formula, record
filtering, teacher, discriminator or model architecture was changed.

The new setting is explicit in native 32px, 64px, local and image configs.
Existing GAN checkpoints remain strict-compatible. Use the refreshed YAML
when continuing the supplied epoch-56 checkpoint; checkpoint metadata does
not replace the active training YAML. No teacher retraining is required.

## Tests, negative result, and limits

- Regression initially failed for the missing case-mixture argument/config;
  four tests now pass: empirical case frequencies, legacy-default frequencies,
  literal case round-trip, non-decaying CTC schedule, and invalid probability
  checks (some behaviors share a test).
- Eight MAIN KID-first contracts and seven applicable runtime regressions
  pass. CPU 32/64 D/G/P/R1 update, sample, strict save/reload and mini-fixture
  native KID validation smoke passed. Source/YAML parsing passes.
- An additional one-update continuation loaded the actual supplied epoch-56
  checkpoint (including its Adam/EMA state), resumed at epoch 57 with the new
  active case/CTC profile and finished with finite weights. This verifies
  resumability only, not uppercase quality after continued GAN training.
- A diagnostic CTC-only adaptation on eight training words, evaluated on
  different words, reduced training loss but did **not** recover exact held-out
  uppercase spelling after 200 updates. The shorter content-only-parameter
  probe also did not repair it. These are negative results, not evidence that
  the production GAN remedy has succeeded. No diagnostic weights were saved
  as a promoted training checkpoint or substituted for the user's file.
- Full legacy suites remain incompatible with several retired experiments;
  their final counts and failing names are in
  `2026-11-09-verification-inventory.md` and the evidence logs.
- No full continued GAN run, independent-OCR benchmark or GPU/T4 test was
  performed. The production changes are an implemented training remedy, not
  a demonstrated recovery of uppercase quality. Further training and a new
  controlled case panel are required before declaring this issue solved.

Changing the current sampling/objective balance can alter KID/FID/style
quality. There is no guaranteed score gain. Keep the original checkpoint as
the control and compare both content cases and full-set metrics after the
next run; a lower KID alone does not prove uppercase content is correct.

All diagnostic artifacts/scripts remain in the ignored `agent` analysis/tests
directories. The original supplied checkpoint was not modified.
