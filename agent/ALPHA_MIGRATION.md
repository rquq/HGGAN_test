# ALPHA naming migration — 2026-10-20

ALPHA is the global-only style experiment paired with MAIN's four-local-token control.
This is a naming/documentation cleanup, not an architecture, metric or training change.

- The active checkout is HGGAN_test/alpha and its Git branch is alpha.
- GitHub's branch-rename operation preserves the existing commit ancestry.
- All four model checkouts recognize alpha when choosing their displayed branch/W&B name.
- Notebook clone lists, data linking, checkpoint dictionaries, launch settings, packaging and commit targets use alpha.
- Shared versioned notebooks are under HGGAN_test/notebooks. Existing HTG-level notebook paths are symlinks to these files.
- Nine checkpoint filenames were renamed; binary hashes were verified unchanged.
- Documentation text, editable diagrams, filenames and report-builder paths use ALPHA/alpha.
- Additional checkpoint-analysis documents and assets were collected under agent/reports.

Historical Git commits, audit records, binary checkpoint internals, W&B/run logs and already-rendered binary/raster snapshots remain historical evidence. They are not rewritten to pretend that past runs used today's label. The original relocation manifest retains original source names/checksums; alpha_rename_manifest.json maps changed filesystem paths. Editable document sources use the new label.

The shared documentation/notebooks are published on a separate agent archive branch, without copying models/checkpoints into it. The four training branches stay focused on model code. The local container's existing master branch and unrelated pending deletions are not staged or rewritten.

No GAN/OCR pretraining rerun is required solely because of this rename. An existing Kaggle notebook must be replaced/reimported with the updated notebook before it can clone alpha. Do not run its old clone/commit cells after the migration.
