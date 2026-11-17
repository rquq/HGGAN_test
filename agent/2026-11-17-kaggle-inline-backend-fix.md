# Kaggle inherited Matplotlib backend — 2026-11-17

The exact supplied traceback failed during the bootstrap's worker import smoke:
Matplotlib rejected module://matplotlib_inline.backend_inline. The notebook
copied Kaggle's parent environment into a standalone Python worker, inheriting
a Jupyter-specific MPLBACKEND unavailable in the isolated worker.

Reproduced the same ValueError in the clean pinned Python 3.11 environment by
setting that exact MPLBACKEND and importing matplotlib.pyplot.

Fix: WORKER_ENV explicitly overrides MPLBACKEND=Agg. The subprocess runner also
overrides it for every generator/OCR job, covering notebook and CLI launch paths.
The parent process environment is preserved; the Jupyter kernel can still use
its inline backend. No notebook-kernel packages are added to the worker.

The actual settings cell was then exercised with the bad parent MPLBACKEND:
the clean child imported matplotlib.pyplot successfully with backend Agg, while
the parent's value stayed unchanged. Regression tests cover both notebook and
runner propagation. Existing paired export identities remain unchanged because
neither adapter nor generator export implementation changed.

Defaults remain DATA_AUGMENT=True, TRAIN_TROCR=True, PREVIOUS_OUTPUT_DIR=None.
No GAN architecture, losses, seeds, dataset rules or TrOCR recipe changed.
Shared and benchmark notebooks remain identical; release notebook refreshed.
No full Kaggle/GPU benchmark or model-quality result is asserted by this check.
