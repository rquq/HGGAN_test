# Repository workflow

- After completing user-approved changes, commit and push the affected branch before reporting completion, unless the user explicitly requests local-only work or no commit/push.
- Verify publication with a fresh `git ls-remote --heads origin <branch>` and compare the remote SHA with local `HEAD`. A local commit or successful-looking push output alone is not proof.
- If publication fails, say explicitly that the change is not available to Kaggle/GitHub. Resolve safe normal-push issues; never silently leave approved changes local-only.
- Inspect divergence before integration. Preserve unrelated edits and remote history; do not force-push, rewrite history, or mix unrelated model experiments without explicit approval.
- Stage only task-related source, configuration, and intended documentation. Do not commit generated tests/artifacts, run outputs, datasets, checkpoint binaries, credentials, or unrelated dirty files.
- Record dated design, implementation, verification, and limitations under the shared `HGGAN_test/agent` documentation directory. Keep reports out of model source directories.
- Maintain shared `HGGAN_test/agent/ALPHA_INCREMENT_LEDGER.md` for every ALPHA increment. Record its base/result SHA, exact architecture/objective/protocol delta, unchanged controls, verification, checkpoint/pretrain compatibility and dated report. Separate workflow/utility changes from model contributions, and proposed/implemented/trained results; never carry retired-lineage results forward as a current ablation.
- In the handoff, list the affected branches and verified remote commit hashes, and distinguish source checks from actual training/quality evidence.
