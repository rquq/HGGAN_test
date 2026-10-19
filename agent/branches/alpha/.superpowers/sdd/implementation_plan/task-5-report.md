# Task 5 report: synchronize MAIN and verify handoff

Status: complete for the bounded synchronization and verification scope; ready for root's final product-diff review.

MAIN's six differing active Python files now match ALPHA byte-for-byte. Its three GAN configs have only the obsolete encoder entries removed; MAIN32 also replaces one stale comment that incorrectly claimed old-checkpoint shape compatibility with the current multiscale visual-evidence purpose. The original snapshot confirms all parsed YAML values are unchanged apart from removal of those obsolete keys, including MAIN32's `2e-5` learning rates and CER/WER toggles. MAIN and Random branch/HEAD identities are unchanged, DEV is clean, and MAIN's original pretrained hashes plus B/W/R/D/P AST body hashes are preserved. The final ignored verifier passed for all 28 tracked Python files and all stated preservation checks.

Validation evidence:

- Pre-sync `verify_sync.py` failed on the expected six Python and three config mismatches; post-sync it passed.
- The initial MAIN default test discovery was red (42 tests: 3 failures and 2 errors) because old local tests asserted removed gate/fusion behavior or the wrong MAIN32 LR. After promoting the current Main-import regression harness and preserving the approved `2e-5` Main32 fixture, final default discovery passed 71/71 tests. The retired `test_reference_fusion.py` was archived as ignored `main/test_artifacts/legacy_tests/legacy_reference_fusion.py`; it is excluded from test discovery and its old architecture is not restored. `git diff --check` passed in both nested repos.
- Strict native MAIN W+B/R checkpoint loading passed for 32px and 64px.
- MAIN32 bounded CUDA train/sample/checkpoint proof passed with G/E=4 updates, D/P=8 updates, finite FP32 values, optimizer ownership, full strict restoration, legacy-state rejection, and OCR input backward. Peak allocated/reserved CUDA memory: 465.6/514.0 MiB; peak host RSS 2,581,420 KiB; wall time 26.39 s. No full training or metric evaluation was run.
- Constructor counts: G 3,915,461→3,942,493 (32px), 3,919,621→3,946,653 (64px); E 1,391,666→1,332,913 at both heights. B/W/R/D/P counts are unchanged and final MAIN equals final ALPHA.

The complete test/resource/config/asset details are in ignored `test_artifacts/local_style_simplification_14_10/implementation_results.md`. Existing Task 4 32/64 CUDA proofs were reused because all tracked Python source is now identical. Fresh GAN training is required; W+B/R teachers remain reusable. MAIN32's `2e-5` is deliberately retained, not presented as an empirical scratch-training optimum. No metric gain is claimed; quality comparison requires controlled identical-seed/evaluator retraining.

Remaining pre-existing caveat: MAIN64 and local configs reference absent `wid_iam_test64.pth`, while `validate_wier` is disabled. It remains untouched and is not needed for the verified training path. No commit, push, or staging was performed.

## Final default-discovery follow-up

The reviewer's final default-discovery run was reproduced before repair and recorded in MAIN's ignored `test_artifacts/local_style_simplification_14_10/task5_default_main_red.log`: 42 tests, 3 failures and 2 errors. The stale failures were the gate-constructor test (`local_evidence_gate_init` is intentionally unsupported), the old local-only-to-global-context depthwise-gradient assertion (two image-height subtests), the old critic fixture expecting `2e-4` instead of the preserved MAIN32 `2e-5`, and the retired fusion module import. No removed product architecture was restored.

The six copied ignored tests were promoted into MAIN's normal `tests/test_*.py` suite, four additional current spatial/local-evidence tests were added, and the Main-specific critic fixture continues to assert the snapshot learning rates. The retired architecture test was copied before removal to ignored `main/test_artifacts/legacy_tests/legacy_reference_fusion.py`; the filename is deliberately outside `test_*.py` discovery. MAIN's default discovery then passed 71 tests in 6.166 seconds, recorded in `main/test_artifacts/local_style_simplification_14_10/task5_default_main_green.log`.

The only additional product-file edit in this follow-up is a comment-only correction in `main/configs/gan_iam_32.yml`; it now describes the shared multiscale visual-evidence path instead of promising old-checkpoint shape compatibility. All parsed configuration values remain snapshot-identical except the three removed obsolete encoder-gate keys. No product Python source was changed in this follow-up. No GPU tests were rerun.
