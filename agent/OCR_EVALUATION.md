# Independent OCR evaluation and local-style ablation

## Next experiment

MAIN retains one global plus four local style tokens and spatial style refinement.
ALPHA (`alpha`) uses one global token and no local extractor/refinement.
The normal 32px/64px training settings are matched; the global-loss normalization divisor stays 8.
Start **fresh GAN training** for a controlled comparison. Keep the existing W+B and R pretrains.
An old local-style checkpoint cannot resume into the new global-only architecture; the standalone tool instead reconstructs the saved checkpoint architecture strictly.
MAIN remains compatible with its existing local-style checkpoints, but continuing only one arm is not a controlled ablation.

## Exactly two OCR metrics

- CER = total character edit distance / total target characters.
- WER = fraction of unequal target/prediction strings for this isolated-word dataset.

Both are fractions, preserve case/punctuation and have no lexicon correction.
CER can exceed 1 when the recognizer inserts characters.
Literal transcript decoding now preserves repeated letters; old double-CTC-collapsed OCR scores are not directly comparable.
Checkpoint metadata records `literal-transcripts-v1`.

The independent evaluator is the pinned `microsoft/trocr-small-handwritten` revision
`b4648cfa171985a6745f37ddd637e98c0da958ac`: 61,596,672 parameters, about 246 MB of weights, FP32.
It supplies no GAN gradients and does not use the training teacher R.
The pretrained decoder does not use the encoder pooler; Transformers can warn that the unused pooler weights are absent.
TrOCR is not perfect on narrow punctuation/word crops, so matched **real-image CER/WER** are also saved as calibration, not extra metric types.
These monitoring scores are not a substitute for training a separate downstream recognizer on synthetic data.

## Cases and provenance

The fixed manifest contains IV-S, IV-U, OOV-S, OOV-U (64 each), plus 32 short/long reference pairs: 320 generated words.
Targets differ from reference transcripts. Short/long references are paired within writer.
OOV excludes the configured lexicon, training transcripts, case variants and text exposed through possible substring cropping.
Writer IDs in the HDF5 splits are split-local. The IAM configs explicitly declare disjoint writer splits; numeric ID equality across splits is not evidence of shared identity.
For other datasets leave `ocr_eval_disjoint_writer_splits: false` unless disjointness is known; labels become train-ref/test-ref rather than asserted seen/unseen.
Insufficient cases are reported, not invented.
Reuse one manifest between comparable branch runs. Different 32px/64px corpus metadata may require different manifests.
This monitored test set is not an untouched final generalization holdout.

## Frequency and memory

Normal configs: `ocr_eval_every_n_epochs: 1`, batch 2, 320 cases.
Local bug-test config: cadence 0, batch 1, 40 cases; 0 means **first eligible validation once per process**, including after resume, not an immediate pre-training evaluation.
A positive N runs on the first eligible validation and subsequent epochs divisible by N.
CPU/low-VRAM fallback runs only once per process.
Missing evaluator weights/dependencies leave CER/WER empty with a clear message; never silently fall back to R or report zero.
CUDA out-of-memory in this optional evaluator disables it for the session rather than terminating GAN training.
Weights are released after evaluation; no permanent extra OCR model accompanies training.

On this RTX 4060 laptop, a bounded 320-case MAIN evaluation took approximately 42 seconds including evaluator loading.
An earlier measured 320-case run peaked at about 301 MiB allocated GPU memory (372 MiB reserved).
This is not a native Kaggle T4 timing or a guarantee under two concurrent jobs.
Adjust cadence to 5 or 10 if the extra runtime is material.

## Commands

Run from this branch directory after installing the notebook dependencies:

```bash
python tools/evaluate_ocr.py --prepare-ocr-only --download-ocr
python tools/evaluate_ocr.py --checkpoint /absolute/path/model.pth --dataset-root /home/quq/machineLearning/HTG/iam --device cpu --batch-size 1 --limit 8 --output runs/ocr_smoke
```

The first command downloads the independent evaluator once; ordinary evaluation uses cached weights only.
The notebook preparation cell also fills this cache without using the GPU.
For an internet-disabled Kaggle run, prepare an accessible Hugging Face cache beforehand.
Remove `--limit 8` for all 320 cases; use `--device cuda:0` for bounded GPU inference.
For a second branch, pass `--manifest /absolute/path/first_run/manifest.json`.
Use `--disjoint-writer-splits` only when this dataset property is known.

Outputs: manifest.json, predictions.csv, metrics.csv, metrics.json and samples.png.
Real calibration cache keys include image-file fingerprints, evaluator revision and preprocessing version.
During training these live under `runs/<run>/ocr_eval/`; existing epoch CSV/W&B CER and WER columns receive the independent scores.
Official full-dataset FID/KID/HWD, old histories and best-FID-only selection are unchanged.
No new loss, teacher retraining, precision change or dataset rewrite is introduced.
