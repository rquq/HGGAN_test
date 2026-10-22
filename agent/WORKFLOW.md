# HGGAN change documentation

User preference confirmed on 2026-10-23, Asia/Bangkok.

For each code-changing task, maintain a dated report under `HGGAN_test/agent/`.
Describe the requested/approved design, relevant source revisions, actual code
changes, affected configs, verification, untested boundaries and retraining
requirements. Distinguish implementation evidence from measured model quality.
Keep architecture documentation accurate when later stages supersede earlier ones.

Keep model branch directories focused on code. Do not commit AI runtime probes,
test artifacts, caches, datasets, generated samples or checkpoint results as
documentation. Existing versioned pretrained teachers are handled separately.

When the user requests commit/push delivery, review the staged diff, run relevant
checks and publish the scoped code and associated documentation. Preserve unrelated
dirty files and branch history. Record and verify the resulting remote commits.
