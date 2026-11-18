# Kaggle flat output publication — 2026-11-18

The prior benchmark exposed the nested output tree as well as its single-file
bundle. Kaggle dataset creation reported duplicate `identity.json` names. The
new notebook keeps the live output and IAM cache in `/kaggle/temp`; it publishes
only `ocr_benchmark_resume.zip.keep` and up to three uniquely named
`ocr_benchmark_*` reports in `/kaggle/working`.

The archived directory structure, datasets, and native TrOCR checkpoint states
are unchanged. The notebook tries to package at normal/pause or Python-error
exit. A successful training invocation with failed packaging now fails the run
cell rather than claiming a saved output. A hard termination before packaging
can still lose the current session's temporary state. This does not touch old
outputs. Kaggle's saved Notebook Output can be attached directly as an input,
or upload only the opaque bundle file as a dataset without unpacking it.

The existing local `chkpoints/18_11_2026/results (8).zip.keep` was opened in
place and verified to have the benchmark root (1,165 entries). No additional
4.1 GB copy was needed. The epoch-based training configuration is unchanged.
