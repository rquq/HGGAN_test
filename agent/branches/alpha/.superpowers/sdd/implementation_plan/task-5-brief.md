### Task 5: Synchronize MAIN, preserve its configurations and verify handoff

**Files:**
- Synchronize active ALPHA Python files to corresponding MAIN paths. Expected differing sources after Tasks 1–4: `networks/utils.py`, `networks/loss.py`, `networks/module.py`, `networks/fusion.py`, `networks/model.py`, plus existing `networks/BigGAN_networks.py`.
- Modify MAIN `configs/gan_iam_32.yml`, `configs/gan_iam_64.yml`, `configs/gan_iam_local.yml` only to remove the three obsolete encoder entries.
- Create ignored `test_artifacts/local_style_simplification_14_10/verify_sync.py` and `implementation_results.md`.

**Interfaces:**
- Snapshot is the original MAIN YAML-values/SHA record in `main_config_snapshot.json`.
- Pre-copy class/body/asset hashes distinguish GAN-only edits from frozen B/W/R and D/P. Tests remain ignored and are not copied as active product source.

- [ ] **Write a failing sync check** that compares tracked active Python source paths/bytes and MAIN YAML values against the snapshot after removing only obsolete gate keys. Assert branch names/HEADs are unchanged and runtime/pretrain/data paths still exist. Expected initial failure: MAIN still has old fusion/generator wiring.
- [ ] **Synchronize active source using `apply_patch`**, preserving `.git` and all assets/runtime/report folders. Unchanged files need no rewrite. Remove only `EncModel.local_attention_residual_init`, `local_evidence_gate_init`, `local_evidence_gate_hidden` from MAIN configs; preserve every other YAML value, including 32px CER/WER toggles.
- [ ] **Run sync verification:** every active source matches; MAIN snapshot comparison passes; no edits appear in DEV or unrelated branches; tests/artifacts satisfy `git check-ignore`; original teacher asset hashes and B/W/R/D/P class-body hashes match.
- [ ] **Run MAIN CPU focused tests** with MAIN on `sys.path` using ignored copies of the test harness, not imports accidentally resolved to ALPHA. Strict-load native32/native64 W+B/R with unchanged definitions. Reuse the proven identical core source for GPU proof; run one extra MAIN32 bounded smoke to verify its own config/path assembly.
- [ ] **Measure constructor parameter counts** before/after for E/G and report unchanged D/P/B/W/R counts. Record measured smoke peak memory/runtime without extrapolating full epoch speed or claiming metric improvement.
- [ ] **Write the ignored result report and final handoff:** exact changed files, removed mechanisms, tests passed/failed/not run, parameter counts, preserved MAIN hyperparameters, and fresh-GAN/teacher-reuse instructions. Explain that MAIN32's preserved `2e-5` is conservative for scratch training; do not silently retune it. Recommend controlled identical-seed/evaluator retraining for quality comparison.
- [ ] **Final review boundary:** root reviews the complete product diff and acceptance coverage. No commit, push, full training or full metric evaluation unless separately authorized.

## Plan self-review

- **Spec coverage:** representation/prior contract → Task 1; E/value grounding/global VAE/obsolete gates → Task 2; spatial raw-value handoff → Task 3; global-only objectives, wrapper routing, reused features and runtime/checkpoint proof → Task 4; MAIN copy, original hyperparameters, frozen assets and transparent reporting → Task 5.
- **Interfaces:** global-only E uses `(B,1,D)` while objective scaling consumes original `N`; optional feature returns retain tuple order; all callers use the same prior helper. No alternative public interfaces or extra config modes are left undecided.
- **Review Focus:** all five conditions have explicit assertions in their owner tasks. Full GPU/multi-GPU training and metric improvement are not claimed by these tests.
- **Proportion/scope:** five independently reviewable tasks, no extra architecture subsystem, loss type, dependency or checkpoint migration. Test/helper code is not prewritten into the plan.

**Status:** written spec and implementation plan approved; implementation in progress, tracked by the ignored SDD ledger.
