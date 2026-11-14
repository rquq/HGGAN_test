# Run the IAM augmentation benchmark

Date: 2026-11-14. Source branch: `ocr_benchmark`.

Notebook:
`HGGAN_test/ocr_benchmark/kaggle_ocr_benchmark.ipynb`.
This is a separate notebook; the existing GAN/pretraining notebooks are unchanged.

## What it compares

One real-only TrOCR control and three independent augmented TrOCR runs:

| Arm | Real IAM training rows | Synthetic training rows | Generator |
|---|---:|---:|---|
| real_only | Same X | 0 | None |
| spis_gan | Same X | Same N paired cases | Native32 SpiS-GAN |
| fw_gan | Same X | Same N paired cases | Native32 FW-GAN |
| higanplus | Same X | Same N paired cases | HiGAN+ native64, explicitly exported32 |

Defaults are X=5,000 and N=25,000 per model. Common references, target words and
per-case seeds are frozen in a manifest. References/lexicon come only from X real
training rows, never from test. Target text differs from its own reference text.
Style encoder/backbone modules needed by each generator are included; no GAN
discriminator, recognition teacher, writer identifier or GAN optimizer is built.

The recognizers all start from the same pinned pretrained-only
`microsoft/trocr-small-stage1`, not the already handwriting-fine-tuned model.
Each gets the same 10,000-update FP32 recipe. Real validation chooses best by
CER then WER. Final metrics use every row in the supplied real IAM test file.
Version 1 fixes the base revision and FP32; alternate OCR backbones and mixed
precision are not supported settings in this version.

## Kaggle: generate and download first

1. Import the new notebook. Select **GPU T4 x2**, enable Internet, and attach a
   Kaggle input containing your two external checkpoint files:
   `bestIAM.pth` for SpiS-GAN and `FW-GAN.pth` for FW-GAN.
2. In the settings cell, keep `RUN_STAGE = 'generation'` for generation only.
   Set `CHECKPOINT_PATHS` explicitly if either filename appears in more than one
   attached input. With one exact match under `/kaggle/input`, automatic discovery
   works. HiGAN+'s deploy checkpoint comes from its pinned official checkout.
3. Keep `COHORT_HEIGHT = 32` for the three-model comparison. Set `REAL_COUNT` and
   `SYNTHETIC_COUNT` before the first run; N must be between1 and25,000.
4. The default writable `DATA_DIR` downloads the selected IAM train/test release
   files only when missing. Alternatively point it to an attached directory
   containing both matching files, named `iam_train_32.hdf5` and
   `iam_test_32.hdf5`. A read-only mount must already contain both files.
5. Choose W&B mode, then **Save Version / Run All**. Default `disabled` needs no
   credentials. For live logging set `WANDB_MODE = 'online'` and enable your
   existing Kaggle Secret **`WANDB_KEY`**. An existing `WANDB_API_KEY` environment
   variable is also accepted. Never paste a key into the notebook.

The notebook creates an isolated pinned worker environment, external source
checkouts and HF cache under `/tmp`, avoiding modifications to Kaggle's base
Python environment and avoiding gigabytes of caches in downloaded outputs.
Two independent workers see one physical GPU each and use logical `cuda:0`.
This is not a single distributed model or a pooled32GB GPU.

Default session budget is10.5h including notebook setup. It is cooperative:
running model loads, optimizer updates or file writes may finish after the
boundary. Queued jobs stop, active jobs save safe state where supported, and the
monitor ends when child processes exit. No printed RESULTS token is used as the
sole completion detector. A paused job is not reported as a finished benchmark.

## Download these artifacts

Default output directory:

```text
/kaggle/working/ocr_benchmark_outputs/
  prepared.zip
  prepared/
  datasets/
    spis_gan/dataset.zip
    fw_gan/dataset.zip
    higanplus/dataset.zip
  logs/
  results.csv
  results.json
```

Each `dataset.zip` contains exactly `synthetic.hdf5`, `metadata.json`,
`manifest.jsonl`, and `preview.png`. The HDF5 uses the IAM packed seven-array
schema, uint8 white-paper/dark-ink images and valid widths, not padded word tails.
Preview shows raw reference/transcript beside target/generated image.

Keep **all three dataset ZIPs plus `prepared.zip`**. The latter contains the exact
shared real training/validation subsets and manifest. A completed archive is
published only after count/order/content checks. If generation pauses, download
or retain the entire output directory, including shards, for generation resume;
an incomplete model may not have a final ZIP yet.

The final notebook cell exposes download links. Saved files also remain in the
Kaggle Output panel if browser FileLinks are not followed. The generation results
table describes generated counts/status; the separate OCR table remains
`not_started` until recognizer training actually runs. Those are not OCR scores.

To assemble downloaded archives manually:

```text
new_output/prepared/             <- extract prepared.zip contents here
new_output/datasets/spis_gan/     <- extract that model's dataset.zip here
new_output/datasets/fw_gan/       <- extract that model's dataset.zip here
new_output/datasets/higanplus/    <- extract that model's dataset.zip here
```

Do not merge all three ZIPs into the same directory: their filenames intentionally
match, and doing that would overwrite one model's data with another's.

## Train TrOCR in a later session

Attach the prior output as a Kaggle input. Set:

```python
RUN_STAGE = 'ocr'
PREVIOUS_OUTPUT_DIR = Path('/kaggle/input/YOUR_PREVIOUS_OUTPUT/ocr_benchmark_outputs')
```

Use the actual folder that contains `prepared`, `datasets`, and any existing
`ocr` subfolder; the example dataset name is only a placeholder.
Run All again. The pipeline copies required state into writable `OUTPUT_DIR`
without modifying `/kaggle/input`, refuses conflicting existing files, skips GAN
source/checkpoint setup in OCR-only mode, and trains/continues the four OCR arms.
The real IAM source files must still be mounted or downloaded for their content
identity checks and final real-test evaluation.

`RUN_STAGE = 'all'` performs generation followed by OCR in one invocation. It may
need multiple Kaggle sessions. `RUN_STAGE = 'generation'` never launches OCR.

For continuation keep X, N, split/generation/OCR seeds, processor revision, batch,
accumulation and training schedule unchanged. A session deadline is not a change
to the original10,000-update LR horizon. Rebuild the temporary runtime each fresh
session; do not copy an old virtual environment into the outputs.

OCR paths:

```text
ocr/<arm>/seed_123456/
  checkpoint-<update>/           # complete native HF training state
  best.json                     # validation-selected canonical checkpoint
  best/                         # convenience alias
  last/                         # convenience alias
  history.csv
  status.json
  test_metrics.json              # exists only after complete final real test
  test_predictions.jsonl
```

Keep canonical `checkpoint-N` folders and selection metadata, not just a
model-only export. Best/last shortcuts are convenience views; restore validates
canonical state and handles link/omitted/materialized transport representations.
If an interruption leaves shortcuts or `best.json` behind a newer complete
native save, canonical trainer state wins and writable mirrors are repaired.
Unsafe or future targets and unverifiable unrelated alias folders still reject.
The next run resumes optimizer, scheduler, RNG and data position. An interrupted
final test restarts safely; no partial final score is published.

## CLI equivalents

Run from the benchmark checkout with its pinned Python environment:

```bash
python benchmark.py prepare --config configs/benchmark_32.yml
python benchmark.py run --config configs/benchmark_32.yml --stage generation --gpu-slots 0,1 --time-budget-seconds 37800
python benchmark.py run --config configs/benchmark_32.yml --stage ocr --gpu-slots 0,1 --time-budget-seconds 37800
python benchmark.py report --config configs/benchmark_32.yml
```

On this workstation defaults point at `../../iam` and the existing external
source/checkpoint folders. Kaggle writes resolved runtime JSON instead of
changing source YAML. Each generator also has its own executable entrypoint:

```bash
python generators/spis_gan.py --config configs/benchmark_32.yml --device cuda:0
python generators/fw_gan.py --config configs/benchmark_32.yml --device cuda:0
python generators/higanplus.py --config configs/benchmark_32.yml --device cuda:0
```

The optional64 config is a separate native64 HiGAN+-only cohort with its own
real-only control. Available native32 SpiS/FW weights are not silently presented
as native64 models. The32 and64 IAM files have different row memberships.

## Reading the report honestly

CSV/JSON show CER, NED, WER and exact word accuracy only for completed final tests.
Empty/paused/invalid rows are not zeros. Case/punctuation/spacing are literal.
An old completed pilot does not count as a result under a changed recipe or
cohort: it is marked incompatible. The saved table includes counts, selected
checkpoint, seeds and paired real-only differences. Upstream checkpoint
training/teacher/exposure facts without evidence remain visibly `unknown`;
software hashes alone do not prove equal upstream supervision.
The compact table mirrors the papers' augmentation question, not an exact
reproduction of their unspecified OCR split/recipe. See the dated paper-protocol
and implementation/verification reports under shared `HGGAN_test/agent`.

The pipeline has bounded local/native tests, including full-size FP32 training
memory diagnostics on RTX4060; it has not completed a physical two-T4 Save
Version benchmark. One seed is a pilot, not a statistically established model
ranking. Do not infer OCR improvements from successful export or smoke tests.
