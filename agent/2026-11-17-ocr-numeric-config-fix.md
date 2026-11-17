# TrOCR optimizer learning-rate type failure — 2026-11-17

The supplied Kaggle log reports both 25,000-sample exports complete and all three
OCR arms failed at update zero. AdamW construction raises TypeError comparing a
float to a string. The encoder pooler missing-weight / tokenizer BOS alignment
messages precede the failure but are not its cause.

## Reproduction and fix

The notebook serializes learning_rate=5e-5 into runtime_config.json as 5e-05.
load_config incorrectly used yaml.safe_load even on JSON. PyYAML's implicit
numeric rules produce the string '5e-05' for this literal. OCR validation checked
integer budgets but did not normalize/validate floating-point hyperparameters.
Direct in-memory/tiny test recipes omitted this real serialization boundary.

JSON now uses json.load. YAML remains supported; scientific-notation numeric
strings in the four floating settings are normalized explicitly. LR must be
positive/finite; decay, clipping and warmup are validated as finite nonnegative
values with the existing upper bound on warmup. Booleans and nonnumeric strings
are rejected before model loading. The notebook validates its actual serialized
recipe and prints the effective LR and type before generation starts.

## Safe continuation

Generated dataset identities and export code are unchanged. Both completed
datasets can be restored and reused; no need to regenerate images for this fix.

Previously failed OCR folders have an identity containing the string LR. A narrow
repair handles only failed global_step=0 jobs with no checkpoint, best/last alias
or test artifacts, valid old hash, and an otherwise identical normalized recipe
and payload. Original identity is preserved as identity.before-numeric-fix.json
before replacement. Repeated application is idempotent. Trained jobs, changed
recipes and conflicting backups are rejected rather than rewritten. The older
failed status/logs remain; normal training publishes the new active/final state.
Rollback before training can restore the saved identity; no checkpoint is deleted.

## Verification

- Reproduced old loader: '5e-05' (str); fixed loader: 5e-05 (float).
- 12 numeric/migration tests cover JSON/YAML scientific notation, NaN/Inf/invalid
  LR, booleans, zero/negative LR, idempotent repair, backup preservation, and
  rejection of checkpoints/nonzero progress/changed recipes.
- Real serialized recipe → native tiny TrOCR Trainer → two optimizer updates →
  saved optimizer/checkpoint → completed-job reload passed on CPU. This is an
  optimizer/training integration check, not a full pretrained-model quality run.
- 159 regression tests passed in 89.41s (existing long all-stage fixture excluded;
  the explicit serialized-config optimizer test was included).
- Shared and benchmark notebooks remain identical and schema/syntax checked;
  release notebook refreshed. Tests and run artifacts remain ignored.

Continuation: attach the completed benchmark output and set PREVIOUS_OUTPUT_DIR
to its root. DATA_AUGMENT=False / TRAIN_TROCR=True trains from both saved datasets;
True/True also verifies/skips the completed exports before OCR. Keep the actual
learning rate 0.00005 and all other recipe/data settings unchanged.
No full-size Kaggle TrOCR run, dual-T4 completion or OCR scores are claimed.
