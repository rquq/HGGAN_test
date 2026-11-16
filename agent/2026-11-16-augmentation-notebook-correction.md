# OCR benchmark notebook correction — 2026-11-16

## Required meaning

In this project, handwriting-generation OCR benchmarking means downstream data
augmentation: train an independent recognizer on X real IAM training samples plus
N synthetic samples; compare against an X-real-only control on held-out real IAM.
A frozen reader scoring GAN images is not this benchmark. The earlier notebook
integration was incorrect and is superseded by this change.

## Implementation checklist

- Remove the new frozen-reader mode/worker from GAN/pretraining notebooks.
- Preserve uploaded GAN/pretraining settings and repaired best-checkpoint carry.
- Reuse the existing ocr_benchmark branch; no second training/evaluation engine.
- Publish a separate kaggle_ocr_benchmark.ipynb under shared notebooks and the
  benchmark branch, with DATA_AUGMENT=True and TRAIN_TROCR=True.
- Default generators: SpiS-GAN and HiGAN+. Same train-only reference/text/seed
  manifest, 25,000 synthetic samples each, X=5,000 real examples and one control.
- Automatically proceed from complete generation to independent TrOCR training.
- Keep prepared data, per-model datasets and per-arm/seed checkpoints separate.
- Restore a previous output root, retaining optimizer/scheduler/RNG, validation
  selection and the original training horizon. Never resume from GAN checkpoints
  or a model-only TrOCR export. Completed jobs are verified and skipped.
- Train-only real-writer validation selects best; full unchanged IAM test is final
  evaluation only. No test refs/texts for synthetic generation or OCR training.
- Keep W&B and cooperative session budgets; portable ZIP generation outputs.
- Test notebook mode boundaries, stage transition, paired inputs and checkpoint
  transport/resume. Commit/push benchmark source and shared notebooks separately.

## Deliberate settings

Two single-GPU workers (one T4 each), not one distributed OCR model. Three OCR
arms queue on those slots: real_only, spis_gan, higanplus. Native64 HiGAN+ output
is explicitly exported32; the comparison is not a native32 HiGAN+ reproduction.
TrOCR-small-stage1, fixed revision, FP32, microbatch2 × accumulation16, 10,000
updates. W&B mode defaults online when a key is available, otherwise disabled.
These existing fair-recipe defaults remain unchanged and are editable together.

PREVIOUS_OUTPUT_DIR accepts the saved benchmark output folder containing
prepared/, datasets/ and ocr/. Its checkpoint-N folders include full Trainer
state; best/last are aliases, not independent models. Keep the complete output
for continuation. Identity checks reject changing the recipe/data while resuming.

## Verification and limitations

No request to launch a 25k/full-TrOCR run on the local laptop is inferred from
the request to prepare the Kaggle workflow. A successful bounded test does not
prove a finished two-T4 session or a measured OCR score.

Implemented all checklist items. The shared notebook and benchmark-branch
notebook have identical bytes. The obsolete shared frozen-reader worker was
deleted, and its earlier report is explicitly marked superseded.

The GAN launcher had also retained an invalid one-argument call to the repaired
carry helper; it now passes branch, status path and launch timestamp and records
carry_done for monitor retries. GAN/pretraining choices from the uploaded
notebook remain unchanged; no model branch was edited.

Verification on the installed HiGANplus Python environment:

- 137 existing benchmark tests passed in 55.67 seconds after updating the
  notebook expectations for the intentional two-model/all-stage default and
  preserving stable notebook cell IDs. Includes paired exports, test-boundary
  checks, native Trainer resume, optimizer/scheduler/RNG equivalence, transported
  checkpoint aliases, validation selection, status/error paths and W&B setup.
- One additional actual CLI all-stage test passed in 117.66 seconds: six paired
  synthetic fixtures per generator, identical manifests, different target versus
  reference transcripts; separate real-only and two augmented tiny native TrOCR
  training jobs, two optimizer updates each, selected-best/full-test artifacts.
  An OCR-only repeat skipped all completed jobs without extra updates.
  Generator loads were substituted with deterministic fixture adapters and OCR
  used a tiny trainable DeiT/TrOCR model on CPU. These are wiring/resume tests,
  not performance/quality results from the released GANs or full-size TrOCR.
- Two shared-notebook contract tests passed: no frozen-reader mode remains in
  either training notebook; default augmentation+training, both selected models,
  25k count, two GPU slots, resume control, source parity and syntax checked.
- All four notebook artifacts passed nbformat validation and plain-Python-cell
  syntax checks; the training notebooks' IPython shell cells were not executed.
- The GAN carry call now supplies all three arguments; previous nonblocking,
  run-receipt and no-overwrite semantics are preserved.

No full 25,000-sample generation, full-size training, fresh runtime installation,
physical two-T4 fit or Kaggle Save Version completion was claimed or measured.
W&B and native sources use the existing pinned runtime/adapters unchanged.

## Minimal Kaggle use

1. Import `notebooks/kaggle_ocr_benchmark.ipynb`, choose T4 x2 and Internet.
2. Attach SpiS-GAN bestIAM.pth or specify CHECKPOINT_PATHS['spis_gan']; HiGAN+
   uses the released deploy checkpoint unless explicitly overridden.
3. Keep DATA_AUGMENT=True and TRAIN_TROCR=True. Run All generates/resumes both
   25k datasets and proceeds automatically to the three independent OCR arms.
4. Outputs are in /kaggle/working/ocr_benchmark_outputs. Download dataset.zip
   files for reusable synthetic datasets; retain the full folder for continuation.
5. Next session attach prior Output, set PREVIOUS_OUTPUT_DIR to its benchmark
   output root, and rerun with the same data/recipe. Generation-only is True/False;
   OCR-only is False/True and does not require GAN checkpoints. Native TrOCR
   checkpoint-N folders, not model-only exports, are the supported resume format.

## Publication

Benchmark notebook committed/pushed on ocr_benchmark as
`4d7de5a42736dfd54878bdddac02028dc8c504be`, with fresh remote HEAD verified.
Shared notebooks, obsolete-worker deletion and this correction report are
published separately on agent. Tests/fixtures/logs remain local and ignored.
