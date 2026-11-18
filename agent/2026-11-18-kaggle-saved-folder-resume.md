# Kaggle saved-folder continuation — 2026-11-18

Replaces the flat publication workflow. The notebook writes the canonical
benchmark output tree directly to `/kaggle/working/ocr_benchmark_outputs`.
There is no automatic resume bundle and no extra copies of the reports.
IAM downloads stay temporary. Existing archive inputs remain readable.

Successive Save Versions do not share their writable working directories.
Attach the previous saved Notebook Output as an input and set
`PREVIOUS_OUTPUT_DIR` to its mounted `ocr_benchmark_outputs` directory.
Restoration checks the prepared dataset and checkpoint layout without editing
the attached source. The restored state is written to the new session's
writable working directory. Nested identity files must keep their paths;
do not flatten the output via the dataset uploader.

Keeps the previous OCR-only, epoch-based two-arm recipe and checkpoint
namespace unchanged. A later run reuses the saved synthetic datasets and
continues compatible TrOCR states. A Kaggle Save Version run was not launched
locally.
