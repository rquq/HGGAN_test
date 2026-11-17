# Kaggle output recovery and HiGAN+ dependency fix — 2026-11-17

## Exact evidence

Input: E:/Downloads/results (6).zip, preserved unchanged.
SHA-256: 2855be8dfc6679d0831d57fe5ee7a5b53c5ff72ba506ad24fda82bf1c7089610.
ZIP CRC test passed; 546 entries, 425,167,918 uncompressed bytes. Paths/symlinks
were checked before extraction into the ignored benchmark work directory.

The actual HiGAN+ traceback is in logs/generate-higanplus.log. Its pinned upstream
networks/BigGAN_layers.py imports matplotlib.pyplot at module import time. The
isolated requirements omitted matplotlib, producing ModuleNotFoundError before
any completed generation shard. The per-model status.json records this exact
error; the CLI's final receipt was absent because the exception escaped main.
This is a missing runtime dependency, not evidence of faulty generator weights,
CUDA exhaustion or poor synthesis quality. The prior broad local ML environment
already contained matplotlib and therefore did not expose this missing requirement.

## Recovery

Extracted output root:
ocr_benchmark/work/kaggle_2026_11_17_output/ocr_benchmark_outputs.
The IAM files included in the ZIP are under the adjacent iam directory.

Verified prepared dataset hashes, split/manifest and the complete portable
SpiS-GAN export (25,000 rows, label/width/writer constraints, artifact/ZIP hashes).
Manifest SHA-256:
7359604e0099289aebe2dff9a335dcd3e517f893e5cadf00f64e6b44254380b6.
No OCR jobs/checkpoints exist in this output: training correctly did not start
when one required generation arm failed.

Also exercised the actual generation-resume function with the clean pinned
SpiS-GAN source/config/checkpoint identities. It returned complete/25,000 without
loading a generator or regenerating samples. The generator export implementation
hash remains unchanged. Existing completed output is reusable.

## Changes

- Pin matplotlib==3.10.0 in the isolated requirements; no upstream model edits.
- Notebook worker import smoke includes matplotlib.pyplot before generation.
- Refresh clean benchmark code via fast-forward before reading requirements,
  allowing an existing /tmp checkout to receive dependency fixes. Dirty checkouts
  fail clearly, never reset.
- Failed runner rows retain the error-log path and last traceback lines, instead
  of only a secondary missing-receipt error. This does not change image export
  identities or introduce a new loss/model/evaluation recipe.
- Shared and benchmark-branch notebooks stay byte-identical; release copy updated.

## Verification

- Installed all fixed pinned requirements in a fresh, isolated managed Python
  3.11.16 environment, without inherited site packages (76 packages resolved).
- Clean upstream HiGAN+ source revision 5d37de733f79e2a72a79c0f66e1865329ff56b4e,
  exact released checkpoint, first 100 cases from the captured training manifest:
  GPU FP32 inference passed, with expected native64 geometry for each target.
- TrOCRProcessor, VisionEncoderDecoderModel and Seq2SeqTrainer imports passed in
  that same fresh environment; this was not a full TrOCR training run.
- 30 focused tests passed in 18.54 seconds: status/trace reporting, queue,
  notebook configuration and transported checkpoint restore.
- Input ZIP remains untouched. No original GAN branch, teacher or checkpoint was
  edited. Environments, copied sources, recovered data and test logs remain in
  ignored work paths, not Git or GitHub release assets.

## Next Kaggle continuation

Use the refreshed standalone notebook. Attach the prior output tree and point
PREVIOUS_OUTPUT_DIR to the folder containing prepared/, datasets/ and logs/.
Keep DATA_AUGMENT=True, TRAIN_TROCR=True and the exact seeds/counts/recipe.
The completed SpiS-GAN generation is checked/skipped; HiGAN+ generates its missing
samples, then real_only and both augmented TrOCR runs start. A local WSL path
cannot be accessed by Kaggle; the saved output must be attached there.

The extracted local output is for inspection/reuse, not automatically uploaded
to a public release. Full 25k HiGAN+ generation and full TrOCR training were not
run locally. No physical dual-T4 completion or OCR-quality score is claimed.
