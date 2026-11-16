# Kaggle ensurepip bootstrap failure — 2026-11-17

The supplied traceback shows Kaggle's Python 3.13 kernel invoking `python -m
venv`, which fails in its child ensurepip invocation before any model is loaded.
The underlying ensurepip stderr was not supplied; no more specific OS/package
cause is asserted. This was a notebook environment setup failure, not a GPU,
checkpoint, W&B, generator or TrOCR loss failure.

Replaced the host-venv assumption with a stdlib bootstrap script that installs
pinned uv 0.12.14 into a private /tmp target and uses a managed Python 3.11 worker
with seeded pip. The notebook kernel and its installed packages are not changed.
Every reused worker must pass `import pip` and Python-version checks. Incomplete
or wrong-version environments are renamed to an .incomplete timestamp backup;
no existing environment is recursively deleted. Old clean code checkouts missing
the new bootstrap can fast-forward from the remote; edited checkouts fail clearly
instead of being reset. Notebook copies are kept identical.

Official runtime mechanism: https://docs.astral.sh/uv/concepts/python-versions/

Verification:

- Fresh actual local bootstrap downloaded managed CPython 3.11.16 and created a
  worker with pip successfully, without calling host `venv` or ensurepip.
- A dry-run resolution of the unchanged pinned requirements against that worker
  resolved all 68 packages, including torch 2.7.1+cu118 / transformers 5.9.0.
  This was resolution, not a complete CUDA environment installation.
- 33 focused notebook, queue, checkpoint restore and new bootstrap tests passed
  before the small old-checkout migration guard; focused bootstrap/Secrets tests
  rerun afterward. Partial-environment preservation and healthy reuse are covered.
- No full Kaggle session, full-size GPU training or OCR quality result is claimed.

The shared local notebook and the downloadable release notebook are refreshed.
Import the refreshed notebook in Kaggle; the old uploaded notebook is a snapshot
and will keep executing its old `python -m venv` cell if it is not replaced.
