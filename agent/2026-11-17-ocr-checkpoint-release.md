# OCR augmentation benchmark release assets — 2026-11-17

Release: https://github.com/rquq/HGGAN_test/releases/tag/ocr-benchmark-weights-v1

User-authorized mirrors of the existing released baseline inference checkpoints.
No weights were modified or retrained. Attribution remains with the upstream
authors; redistribution here does not imply authorship or new licensing terms.

| Model | Asset | Upstream |
|---|---|---|
| SpiS-GAN | bestIAM.pth | https://github.com/DAIR-Group/SpiS-GAN |
| HiGAN+ | deploy_HiGAN+.pth | https://github.com/ganji15/HiGANplus |
| FW-GAN (optional) | FW-GAN.pth | https://github.com/DAIR-Group/FW-GAN |

Exact byte identities are in the accompanying checkpoint_assets.json. These
checkpoints include the generator/style-encoder/backbone required by the adapters;
there is no need for separate OCR/writer teachers to run generator inference.
Original training/test-exposure provenance remains unknown where undocumented.

The notebook fetches the selected weights into /tmp/ocr_benchmark_checkpoints and
checks SHA-256 before native strict loading. A manually supplied CHECKPOINT_PATHS
entry remains an explicit override. No Kaggle checkpoint attachment is required
for the default run. OCR-only continuation does not download GAN checkpoints.

IAM downloads remain on the existing data release. The pinned TrOCR-small-stage1
base/processor come from Hugging Face automatically. Newly trained TrOCR states
are benchmark outputs, not pretrained input assets; existing test fixtures and
private training artifacts are not uploaded.

The release also includes the standalone Kaggle notebook and checksum manifest.
Only notebook download plumbing changed; generator configs and identities were
not extended with release URLs, avoiding needless generation-resume incompatibility.

Verification: release is public/published (not draft); GitHub asset digests match
all three local source files. All three public browser-download URLs were fetched
through the actual pipeline.assets CLI into a new /tmp cache and SHA-256 verified.
The uploaded notebook digest also matches the canonical shared notebook exactly.
23 focused tests passed (asset integrity/cache/no-overwrite/failed-publication,
notebook utilities, runtime config and runner). Both notebook copies pass schema
and Python-cell syntax validation. No full GPU benchmark was launched.

Benchmark source commit: 33e1f4ec622745cf02e755ddc1d08d9a34a1ec67, pushed and
verified against fresh remote HEAD. Trained future OCR checkpoints still belong
in each Kaggle output folder for continuation, not this input-weights release.
