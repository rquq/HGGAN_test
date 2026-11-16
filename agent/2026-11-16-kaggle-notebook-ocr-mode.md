# Kaggle notebook promotion and baseline OCR mode — 2026-11-16

**SUPERSEDED the same day:** the frozen-reader mode described here was the wrong
interpretation of the requested OCR benchmark and has been removed. Use the
separate `notebooks/kaggle_ocr_benchmark.ipynb` for saved synthetic datasets →
independent TrOCR training → held-out real IAM evaluation. The GAN/pretraining
notebooks contain no OCR benchmark mode. See
`2026-11-16-augmentation-notebook-correction.md` for the correction.

## Scope

Promoted the exact supplied `C:/Users/Lenovo/Downloads/higan-edits (2).ipynb`
to `notebooks/one_for_all_kaggle_HTG.ipynb`. The original upload remains unchanged.
Updated the authoritative local notebook as well; the workspace-root notebook
links continue to point to these files. No generator, discriminator, loss,
teacher, training config, or model-branch checkpoint was changed.

## Controls

The Kaggle `#configs` cell preserves TIME_LIMIT=12 and the supplied GAN and
teacher branch/config choices. `RUN_MODE="ocr_metrics"` now selects the new
evaluation mode and gates both training launchers off. Set it to `"training"`
to use the original RUN_GAN_JOBS / RUN_TEACHER_PRETRAIN controls.

Default evaluation: SpiS-GAN on physical GPU0 and HiGAN+ on physical GPU1,
one independent process each, FP32, OCR batch size 2. The local notebook keeps
training mode as its default and offers a single-GPU SpiS-GAN preset; change
the model list to higanplus for the next run. All settings are in OCR_EVAL:
model list, GPU slots, checkpoint/source paths, dataset, polarity, output folder,
batch size, sample limit, seed, evaluator/revision, decoding length, W&B mode,
real-image control, conversion and unsupported-character policy.

SpiS-GAN requires an attached bestIAM.pth (or an explicit checkpoint path).
Auto-discovery refuses multiple candidates. HiGAN+ defaults to its released
deploy_HiGAN+.pth in the pinned upstream checkout. Source Git revisions and
effective source/checkpoint hashes are recorded by the existing frozen adapters.
The Kaggle worker uses a temporary isolated environment with the benchmark's
pinned requirements; the notebook's base environment is not replaced. First-run
setup needs network access. Source and runtime setup precede job launch and are
not included in the monitor's TIME_LIMIT clock.

## Evaluation definition and limits

This is direct, independent frozen TrOCR evaluation, NOT OCR pretraining or
the real-plus-synthetic augmentation benchmark. Each IAM test image is the style
reference; its own transcript is the generated target. This is a same-transcript
guided-generation/reconstruction condition, not proof of novel-text generation.

Evaluator: microsoft/trocr-small-handwritten at revision
b4648cfa171985a6745f37ddd637e98c0da958ac, FP32, greedy decoding, max_new_tokens=48.
CER is total character edit distance divided by total reference characters.
WER here is exact word-crop failure rate, consistent with MAIN's independent
word evaluator, not a multiword token-WER interpretation. Case is preserved.
Both generated and optional real-IAM control images go through the same reader.

The default IAM32 test file has 25,766 records. The two `*` targets are absent
from HiGAN+'s vocabulary. The explicit common_subset default uses the intersection
of all selected generators' alphabets, excluding the same two records for both
models and the real control: 25,764 evaluable records. Counts and selected-row
hashes are saved. Use unsupported_policy="error" to reject such a dataset rather
than report a subset. No remapping or invisible replacement is performed.

HiGAN+ is native64 and explicitly bilinear-downsampled to IAM32; this is labeled
in results. SpiS-GAN stays native32. These are not claims of native32 HiGAN+
reproduction or the papers' downstream OCR augmentation protocol. Source
checkpoint training/test provenance remains unknown unless supplied independently.

Outputs: `<MASTER_PATH>/ocr_metrics/<model>/metrics.json` and
`predictions.jsonl`, plus model log/status/PID files and the effective evaluation
config in the parent output folder. In Kaggle these remain downloadable under
/kaggle/working/ocr_metrics. Predictions are streamed to a .partial file and final
metrics are published only on success. Interrupted evaluation must be rerun;
there is no mid-evaluation resume. The summary table requires a completed status
with the current launch PID, so a failed job cannot display a previous score.
W&B is configurable; online is selected only when a key is available.

## Best checkpoint carry-forward

Reused the previously repaired nonblocking carry helper in both notebook monitor
paths. It waits for this launch's run receipt, validates the branch/run directory,
and copies the actual previous best checkpoint into that run's ckpts directory.
It does not guess the latest run folder, call a last checkpoint the best, overwrite
an existing checkpoint, or silently choose among multiple best candidates.
Pending copies are retried by the monitor. Explicit best inputs and compatible
FID/KID names remain supported. Ambiguous candidates require an explicit path.

Removed the uploaded setup's plaintext W&B-key write-back to the downloadable
working folder. Existing secret/environment-based authentication is retained.

## Verification

- Both notebooks: JSON, plain-Python cell syntax, mode/settings ordering and
  cleared execution outputs checked. IPython pip/shell cells were not executed.
- Both native baseline checkpoints strictly loaded G/E/B and generated two IAM32
  samples each on CPU. A fixture reader exercised the full result-writing path;
  both selected the same row hash. These fixture scores are NOT OCR results.
- Real checkpoint native-height conversion and two unsupported records checked.
- Carry regression checks: missing/stale receipts, exact destination, retry,
  preservation of an existing destination checkpoint.
- Invalid worker config produces a terminal failed receipt; both notebook result
  tables refuse to publish a score from a failed launch.
- No full TrOCR evaluation, fresh dependency installation, dual-T4 run, Kaggle
  save-version run or runtime/VRAM guarantee is claimed. No new quality score
  or model improvement is asserted. Generated fixtures remain local analysis
  artifacts and are excluded from the publication commit.
