# Single-file resume upload — 2026-11-17

The user reported Kaggle dataset creation failing with duplicate identity.json.
The benchmark intentionally uses identical basenames in separate prepared/model/
OCR directories. Flattening those files at upload makes them collide. The exact
Kaggle UI action was not observed; no claim of reproducing its UI is made.

Added pipeline.bundle to export a single path-preserving ZIP with intentional
.zip.keep suffix for opaque single-file upload. It includes native checkpoint-N
states and data, omitting redundant best/last directory aliases, restored alias
backups and PID files. Canonical native state rebuilds best/last during restore.
It never renames identity/status/metadata files inside the output tree.

restore_outputs now accepts either a directory or a ZIP archive file regardless
of suffix. Archive paths, duplicate entries and symlinks are checked before
extraction. Extraction uses a task-owned temporary folder on the destination's
filesystem rather than assuming /tmp has room for large OCR states. The existing
prepared/checkpoint/conflict validators still gate final restoration.

The notebook packages OUTPUT_DIR in a finally block after the pipeline returns,
including Python-level failed runs. Packaging failure is reported without hiding
the original run exception. No guarantee is made for process/kernel/VM termination
before that block executes. Full archives need extra disk space and time to write.

PREVIOUS_OUTPUT_DIR accepts the attached archive path; no additional resume mode
or alternate OCR engine was introduced. Future bundles are named
/kaggle/working/ocr_benchmark_resume.zip.keep. Upload this ONE file, not all its
contents. Keep the suffix when using an uploader that automatically unpacks .zip.

For the available local results (6).zip recovery, produced:
E:/Downloads/ocr_benchmark_resume_spis_only.zip.keep (184 MiB).
This bundle contains only the older completed SpiS-GAN export and shared cohort;
the later completed HiGAN+ output was not supplied locally. Do not label it as a
two-model complete output. The original ZIP is unchanged.

Verification: 31 focused bundle/notebook/queue/checkpoint tests passed, plus a
follow-up bundle gate after destination-filesystem extraction adjustment. The
real recovered output was exported to this single file, restored to a new local
folder, and its 25,000-sample SpiS-GAN export revalidated. Both notebook copies
pass schema/syntax and are identical. No full GPU/OCR run or Kaggle UI automation
was performed. Recovered data/bundles/tests are not staged or uploaded publicly.
