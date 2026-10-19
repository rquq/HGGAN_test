## Global Constraints

- ALPHA means `/home/quq/machineLearning/HTG/HGGAN_test/alpha`; MAIN means `/home/quq/machineLearning/HTG/HGGAN_test/main`.
- Keep frozen B/W/R, global D, StrokePatchD, vocabulary, datasets, data polarity, preprocessing, evaluator/protocol, seeds and FP32 unchanged.
- Retain content-first ContentContext, ordinary GBlocks, BSSP global conditioning, valid-width padding and the common 32/64px topology.
- Standard style shape remains `(batch, 8, 32)`; implement generically over configured token count and dimension. Token zero is global; remaining tokens are centered local residuals.
- No character/length/resolution-specific feature rules, ink thresholds, background filtering, new loss types, extra architecture toggles or LR/EMA tuning.
- Preserve existing global-only VAE sampling and KL `/N`; preserve current loss weights and BCE positive-weight policy.
- New GAN architecture requires fresh training. Do not add old-checkpoint conversion, partial initialization or missing-key shims. Existing W+B/R checkpoints remain reusable.
- Preserve nested `.git`, branch identity, runs, W&B logs, datasets and checkpoints. DEV and all other branches remain untouched.
- MAIN keeps every non-obsolete configuration value from `main_config_snapshot.json`, including 32px LR `2e-5`, epochs `90`, decay `74/16`, floor `0.25`, batch `8`, critic count `2` and rare-word ratio `0.25`.
- Source edits use `apply_patch`. Tests/plans/results live only under Git-ignored `tests/` or `test_artifacts/`; no automatic staging, commit or push.
- Use `gpt-6-luna` with `xhigh` for bounded implementations/tests. Root owns design decisions, integration review and failure triage. Serialize GPU tests; do not launch full training or metric evaluation.
