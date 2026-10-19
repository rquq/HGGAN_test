# Checkpoint and teacher audit — 14 September 2026

## Executive conclusion

**The strongest actionable finding is a defective 32px training dataset, not a failed 32px writer architecture.** Thousands of exactly identical word images have different writer labels. Your 32px writer is already almost optimal against those contradictory labels. Training it longer cannot resolve that contradiction.

The 32px GAN is only 4.06% behind the 64px checkpoint in FID, but 55.79% behind in KID and 24.51% behind in HWD. Its late training improves FID and HWD while KID worsens. That is a genuine difference in the recorded trajectory, but this audit does **not** establish how much of the GAN gap comes from data, capacity, optimization, or the evaluation protocol. A controlled clean-data run is needed to establish causality.

Do not remove the encoder/fusion improvements or replace the writer backbone as the first response. Correct the data and evaluation issues first, then test resolution-specific generator/patch-critic changes separately.

## 1. What was actually inspected and run

- Inspected both PTH files in the requested folder, including every embedded evaluation-history row, embedded CSV, configuration snapshot, model manifest, optimizer learning rate and saved model/EMA tensor.
- Exported complete histories: [32px, 84 epochs](/home/quq/machineLearning/HTG/chkpoints/14_9_2026/analysis/gan_32_all_epochs.csv) and [64px, 71 epochs](/home/quq/machineLearning/HTG/chkpoints/14_9_2026/analysis/gan_64_all_epochs.csv). Both embedded CSVs agree with their structured histories.
- Ran six complete teacher evaluations: your 32px/64px OCR and W+B, plus the OCR and writer/shared-backbone components from `chkpoints/SpiS-GAN/bestIAM.pth`.
- Used the RTX 4060 Laptop GPU, FP32, batch size 16, no data-loader workers, sequential processes, no training, and a 45% PyTorch GPU-memory cap. Peak allocated model/tensor memory in each evaluation was approximately 0.10 GiB; this excludes CUDA context and other applications.
- Ran small CPU forward/gradient probes for both complete resolution paths. These are not a full GPU training soak test.
- Audited raw HDF5 image duplication, labels, sample counts and train/test overlap.

GAN FID/KID/HWD values below are **read from the checkpoint histories**, not newly recomputed. The OCR/writer results are fresh full evaluations, not stored metadata or smoke-test scores.

Current inspected trees: DEV `475302107b7fd5ab2d986985fb5a67bb4cd0c647`, MAIN `e75d6c14628ad7e999019bb531e53045307e250e`. DEV has pre-existing uncommitted work; it was preserved. These local revisions do not prove which exact Kaggle revision produced the checkpoints: the saved artifacts lack an authoritative source-code commit hash.

## 2. GAN results and full-history interpretation

Lower is better. KID uses this codebase's displayed scale, **raw MMD² × 100**.

| Checkpoint | Resolution | Epoch | FID | KID | HWD |
|---|---:|---:|---:|---:|---:|
| `best_fid_3.2594.pth` | 64 | 71 | 3.259370 | 0.080285 | 0.090112 |
| `last_fid_3.3916.pth` | 32 | 84 | 3.391603 | 0.125076 | 0.112198 |

| History landmark | 32px | 64px |
|---|---|---|
| Best recorded FID | 3.391603, epoch 84 | 3.259370, epoch 71 |
| Best recorded KID | 0.092320, epoch 54 | 0.071451, epoch 51 |
| Best recorded HWD | 0.104658, epoch 79 | 0.077500, epoch 50 |

The best values above occur at different epochs; they are not the metrics of a single hypothetical checkpoint. This folder does not contain the epoch-54 32px checkpoint, only its history row.

### The 32px KID regression is not just one noisy evaluation

| Epoch window | Mean 32px FID | Mean 32px KID | Mean 32px HWD |
|---|---:|---:|---:|
| 36–47 | 4.300547 | 0.124196 | 0.138554 |
| 48–59 | 3.872918 | 0.115648 | 0.130266 |
| 60–71 | 3.665304 | 0.125735 | 0.122868 |
| 72–84 | 3.521124 | 0.127071 | 0.113828 |

The latest KID is about 35.5% worse than its epoch-54 minimum. Meanwhile, FID and HWD improve. Consequently, “more epochs” or “undo the decay” is not a sufficiently supported KID fix. The 64px run's mean KID improves from 0.089834 at epochs 48–59 to 0.085404 at epochs 60–71.

Both saved schedules use linear decay starting at epoch 36, 48 decay epochs and a 0.15 LR floor. Saved G learning rates are 0.000030 at 32px and 0.000054167 at 64px. There is no evidence here of decay simply stopping all learning.

No NaN/Inf was found in the saved G, D, P, E, R, W, B or G/E EMA tensors. The saved R/W/B weights match the currently installed resolution-specific pretrained files exactly, including buffers. Current MAIN and DEV copies of those four teacher files are byte-identical.

## 3. Critical finding: contradictory writer labels in the 32px file

This is an audit of raw pixel bytes in `iam_train_32.hdf5`, not a similarity search or a guess based on model predictions.

| Training-data property | 32px | 64px |
|---|---:|---:|
| Samples | 66,603 | 63,186 |
| Unique image byte sequences | 63,509 | 63,140 |
| Exact-image groups with conflicting writer IDs | 3,035 | 3 |
| Such groups with non-uniform images | **3,032** | **0** |
| Samples participating in writer conflicts | 6,123 | 48 |
| Minimum image-only writer errors implied by conflicts | 3,085 | 43 |
| Corresponding WIER floor | 4.631923% | 0.068053% |

“Non-uniform” excludes constant-value images, such as completely blank crops. Most 32px conflicts are ordinary word images, not visually ambiguous dots or blank paper.

Examples, using zero-based HDF5 row numbers:

| Text | First row / writer | Identical-image row / writer |
|---|---|---|
| PRESIDENT | 1497 / 3 | 50479 / 190 |
| KENNEDY | 1498 / 3 | 50480 / 190 |
| is | 1499 / 3 | 50481 / 190 |
| ready | 1500 / 3 | 50482 / 190 |
| to | 1501 / 3 | 50483 / 190 |

The image seek offsets are distinct and agree with cumulative image widths. Thus these are duplicate pixel records already stored in the HDF5 file, not a current loader accidentally returning one seek offset for two samples.

For each identical-image group, a deterministic image-only classifier can correctly choose at most its most frequent writer label. Summing `group size − largest writer count` produces the stated error floor. Padding/batch-context artifacts are not a legitimate way to resolve conflicting labels.

### This explains the apparent writer-teacher gap

- Your 32px writer made **3,092 errors**, only 7 more than the 3,085-error floor.
- Of those errors, 3,086 occur among contradictory images.
- On the remaining **60,480 unambiguous images**, it made **6 errors**: WIER **0.009921%**.
- SpiS-GAN made 38 errors on that same unambiguous subset: WIER **0.062831%**.
- Your 64px writer made 58 total errors, including 45 in its much smaller set of blank/constant-image conflicts.

Therefore, the overall 4.64% versus 0.092% WIER comparison does **not** mean the 32px writer is intrinsically around 50 times worse. Most of that difference is label inconsistency.

### Why this can hurt the GAN

The [writer loss](/home/quq/machineLearning/HTG/HGGAN_test/dev/networks/model.py:2407) classifies generated style-transfer images against `real_wids`. Identical reference pixels can therefore ask the model to satisfy different writer targets. The frozen teacher cannot make those targets consistent. This can create pressure to change style away from the visual reference to satisfy an arbitrary ID.

That is a verified source of contradictory supervision and a strong candidate contributor to style/KID problems. **The precise improvement in KID/HWD after repairing it has not been measured.** SpiS-GAN also encounters the 32px data problem; it is not proof that this alone explains the entire GAN gap versus SpiS-GAN.

Full evidence: [duplicate audit](/home/quq/machineLearning/HTG/chkpoints/14_9_2026/analysis/dataset_duplicate_audit.json), [all conflicting 32px training rows](/home/quq/machineLearning/HTG/chkpoints/14_9_2026/analysis/iam_train_32_conflicting_images.csv), [teacher subgroup results](/home/quq/machineLearning/HTG/chkpoints/14_9_2026/analysis/teacher_subgroups.json).

## 4. Fresh OCR and writer evaluations

### Scores using the existing DEV evaluation's repeat-collapse behavior

All table values are percentages. CER = character error rate; WER = whole-word error rate; WIER = writer-identification error rate.

| Model | OCR test samples | CER ↓ | WER ↓ | Writer samples | WIER ↓ |
|---|---:|---:|---:|---:|---:|
| Your 32px R / W+B | 25,766 | **9.2116%** | **22.3822%** | 66,603 | **4.6424%** |
| Your 64px R / W+B | 25,432 | 8.5460% | 21.1938% | 63,186 | 0.0918% |
| SpiS-GAN native 32px R / W+S | 25,766 | 11.3235% | 27.7265% | 66,603 | 4.7085% |

Your 32px OCR beats SpiS-GAN by **2.1119 percentage points CER** and **5.3443 points WER** on the same 32px test file. Your writer is slightly better overall—44 fewer mistakes—not dramatically better on this contradictory-label corpus.

The writer evaluation uses the training corpus with the 339 known writer IDs, matching current writer-pretrain validation. It is **not held-out writer recognition**. Test-set integer IDs represent another writer universe; using those integers directly with a 339-class training-writer head would be invalid. High training WRR also does not establish generalization to unseen writers.

Only a native 32px SpiS-GAN checkpoint is available here. Its R/W/S were extracted from the supplied epoch-98 GAN checkpoint and loaded with the native architecture. SpiS-GAN's code jointly optimizes R/W/S during GAN training, unlike your frozen-teacher GAN. These are comparisons of the supplied systems, not a controlled pretraining-architecture experiment.

Each model received its native expected input polarity: inverted black-paper/white-ink tensors for your teachers, white-paper/black-ink tensors for SpiS-GAN. Feeding SpiS-GAN the opposite polarity would produce an unfair failure. No input-resolution interpolation was used in these six evaluations.

### Actual OCR evaluation bug: repeated letters are collapsed twice

[Recognizer validation](/home/quq/machineLearning/HTG/HGGAN_test/dev/networks/model.py:2966) already applies `ctc_greedy_decoder`, then calls the label converter's CTC-style `decode` again. It also applies that converter to ground-truth text. The [converter](/home/quq/machineLearning/HTG/HGGAN_test/dev/lib/alphabet.py:90) removes adjacent repeated labels unless `raw=True`.

For example, a legitimate double letter can disappear from the reference or already-decoded prediction. That can hide real spelling errors. Corrected metrics below preserve ground-truth transcripts and apply CTC collapsing only once:

| Model | Correct transcript CER ↓ | Correct transcript WER ↓ |
|---|---:|---:|
| Your 32px OCR | 9.2924% | 22.7121% |
| Your 64px OCR | 8.6637% | 21.6460% |
| SpiS-GAN native 32px OCR | 11.4110% | 28.1223% |

The ranking remains unchanged. This is a reporting/checkpoint-selection bug, not evidence that the training CTC target encoding itself collapses repeats. Fix the validation call sites, not the converter's default indiscriminately: other callers may intentionally need CTC decoding.

There are also a few `*` characters in the data, while your alphabet lacks `*`; the current encoder maps unsupported characters to blank. This merits an explicit alphabet validation policy but is far too rare to explain the headline KID/HWD gap.

Results: [comparison CSV](/home/quq/machineLearning/HTG/chkpoints/14_9_2026/analysis/teacher_comparison.csv). The same directory contains the six full JSON results with checkpoint hashes and six per-sample prediction CSVs. Files with `_smoke` in their names are preliminary checks and were not used in these tables.

## 5. Is the 32px implementation broken, or are the improvements unsuitable?

### Checked and working

The saved checkpoints load strictly into the current DEV implementation at their respective resolutions. For a six-character-wide input, both resolutions produce:

- Backbone sequence: `[B, 256, 12]`.
- OCR output: `[B, 12, 80]`.
- Encoder style tokens: `[B, 8, 32]`.
- Matching intermediate backbone feature shapes after the resolution-specific initial stride.

The [backbone](/home/quq/machineLearning/HTG/HGGAN_test/dev/networks/module.py:79) and [recognizer](/home/quq/machineLearning/HTG/HGGAN_test/dev/networks/module.py:669) correctly use a width reduction of 8 at 32px and 16 at 64px. The short-reference support calculation uses image-height-relative character widths, so the one-character safeguard is not simply twice as strong at 32px.

Both probes passed generated-height/width checks, black-tail masking under the internal polarity convention, finite style/image outputs, frozen-teacher gradients to inputs, and second derivatives through global D and StrokePatchD. No wrong-height or half-length masking failure appeared in these checks. The teacher weights and their expected input polarity match the GAN.

[Probe results](/home/quq/machineLearning/HTG/chkpoints/14_9_2026/analysis/resolution_path_probes.json). This is bounded verification, not proof that every training path is bug-free.

### Structural differences that remain relevant after data repair

| Property | 32px | 64px |
|---|---:|---:|
| G parameters | 1,573,764 | 4,021,956 |
| GBlock stages | 3 | 4 |
| Initial G feature channels | 256 | 512 |
| Global D parameters | 2,366,145 | 2,366,145 |
| StrokePatchD parameters | 810,305 | 810,305 |
| Native stroke crop | 16 × 16 | 32 × 32 |
| StrokePatchD output per crop | 2 × 2 | 4 × 4 |
| Training batch size | 16 | 8 |
| Saved EMA/G-update counter | 174,804 | 280,379 |

The [32px generator architecture](/home/quq/machineLearning/HTG/HGGAN_test/dev/networks/BigGAN_networks.py:20) has only 39.1% of the 64px generator's parameters, while both critics retain their parameter counts. That is a substantial capacity change, not merely rendering the same model at half height. Parameter ratios alone do not prove discriminator domination.

The [patch critic](/home/quq/machineLearning/HTG/HGGAN_test/dev/networks/BigGAN_networks.py:515) applies the same three downsampling blocks to a crop of half the side length. It runs correctly, but the 2×2 score map and fixed-size oriented kernels may be less well matched to fine 32px strokes. A useful isolated experiment is to preserve the final spatial resolution at 32px—for example, disable its last downsampling operation—rather than immediately increasing patch count or discriminator strength.

The [texture module](/home/quq/machineLearning/HTG/HGGAN_test/dev/networks/texture_generator.py:36) uses 1×7, 7×1 and 3×3 kernels at both heights. Its effective coverage relative to a character doubles at 32px. This is a plausible ablation target, not proof that the module hurts KID. The saved EMA detail gain is about 0.04337 at 32px versus 0.04442 at 64px; there is no obvious scalar-gain explosion. Synthetic-input probes are not a valid quality ablation.

The 32px run has 37.65% fewer G updates despite having more epochs. Both use `num_critic_train: 2`; the larger 32px batch roughly halves updates per epoch. Decay and EMA therefore operate on different update/example timescales. This is a comparison confound, not a reason to indiscriminately raise learning rates or double training duration.

### Current MAIN has code/config drift beyond resolution

Current MAIN still hardcodes generated-image metric caching to int8, whereas DEV supports and defaults to float32: [MAIN cache](/home/quq/machineLearning/HTG/HGGAN_test/main/networks/model.py:1155), [DEV cache](/home/quq/machineLearning/HTG/HGGAN_test/dev/networks/model.py:1208).

MAIN's current 32px config also uses `local_attention_gate_init` and `feature_scales`, but its older StyleEncoder does not implement those options the same way as DEV. Its local-attention initializer defaults to 0.25 unless the recognized `local_attention_residual_init` argument is used. DEV additionally has soft patch-character confidence absent from current MAIN.

These are verified current-tree consistency problems. However, the 32px checkpoint snapshot already contains several newer DEV-style options. Without the training code hash, it would be incorrect to assert that the current MAIN cache or ignored options caused its historical scores.

## 6. Dataset and metric comparability cautions

The supplied files are not exact paired resizes:

| Split | 32px samples | 64px samples |
|---|---:|---:|
| Train | 66,603 | 63,186 |
| Test | 25,766 | 25,432 |

Writer/text fingerprint matching strongly suggests different writer-ID permutations, too. An inferred assignment increases shared training writer/text occurrences from 13,734 to 62,989. This is evidence against treating numeric IDs as interchangeable, not authenticated writer metadata suitable for automatic relabeling. Do not drop the 64px writer head into the existing 32px dataset without a verified label mapping.

No non-uniform exact-image train/test overlap was found at either resolution. A few uniform images overlap; that does not explain the OCR improvements through ordinary word-image leakage. This audit does not rule out near-duplicate leakage.

HWD currently [resizes every image to height 32](/home/quq/machineLearning/HTG/HGGAN_test/dev/metric/hwd.py:483), using nearest-neighbor interpolation by default. Thus the 64px run does not simply get a 64px HWD feature input. Its generation/training advantages and the downsampling operation still matter.

The current GAN HWD call compares global mean feature vectors; it does not average a separate distance per writer. Therefore, a low score does not prove correct writer-conditioned allographs. Keep that interpretation distinct from the separate writer-ID evaluation.

The current FID/KID preprocessing uses width `4 × height`, resize to 299², and converts padded -1 tails to +1. It does not invert the valid black-paper model image into SpiS-GAN's native white-paper presentation. HWD's tensor-to-PIL path likewise preserves the supplied valid-region polarity. Applying the same formula to different input polarities is not a strictly matched baseline protocol. Before making a new cross-system benchmark claim, verify that generated **and real** inputs have the same canonical presentation, sample set, cache precision, crop policy, feature weights and random sampling for all systems. Do not silently change this protocol and append the resulting scores to the old CSV as if nothing changed.

## 7. Recommended order of work

1. **Repair the 32px training data first.** Best option: rebuild paired 32px/64px files from one authoritative source with stable sample IDs and writer IDs. A practical alternative is deriving 32px images from the cleaner 64px corpus while preserving its labels and splits. Investigate the few blank/label-conflict cases too. Do not guess the correct writer of a conflicting pair or merely keep the first record.
2. **Keep the current teacher architectures initially.** The 32px writer does not need a more elaborate head to solve this apparent accuracy deficit. Retrain or adapt W+B against the corrected identity mapping; never reuse the old classifier labels blindly. The current OCR is a useful starting point and can be revalidated/fine-tuned rather than automatically discarded. For a clean GAN comparison, use a new run with clearly versioned data/teacher provenance.
3. **Fix double-collapse OCR reporting and synchronize intended MAIN/DEV options.** Keep corrected metrics distinguishable from historical ones. Record the code revision and dataset hashes in future checkpoints. Align metric-cache precision and the baseline's complete input protocol before comparing new KID numbers.
4. **Hold fusion/token count/critic ratio steady for the first clean-data experiment.** This isolates the strongest verified problem. Do not simultaneously remove style fusion, weaken OCR and add a new texture loss.
5. **Only then test 32px-specific capacity/locality changes:** preserve a 4×4 patch score grid; modestly increase G capacity while keeping 32px output; separately test reduced or disabled texture refinement. Use matched training budgets and a fixed evaluation suite. None is guaranteed to improve KID without an ablation.
6. **Track KID alongside FID when choosing what to continue.** Continue saving only best-FID if that is your preference, but retain periodic/last checkpoints long enough to test a KID-favorable epoch. The history already shows that best-FID selection is not equivalent to best-KID selection.

For the current 64px run: preserve the epoch-71 checkpoint. Its trajectory is healthy enough that these findings do not justify an architectural reset. For the current 32px run: preserve epoch 84, but do not spend another long run hoping LR changes alone will repair contradictory supervision. I would prioritize the clean-data experiment.

## 8. Scope and reproducibility

No training/config/model source, pretrained file, dataset or existing checkpoint was changed. Analysis artifacts are under this directory, with a local `.gitignore` to keep them out of commits.

Scripts: `evaluate_teachers.py` (six full GPU validations), `inspect_evidence.py` (checkpoint/CSV extraction), `audit_dataset_duplicates.py` (exact-image audit), `probe_resolution_paths.py` (small CPU shape/gradient probes), `summarize_results.py` (result tables and diagnostic subgroups). JSON evidence includes file hashes, sample counts and settings where applicable. The full comparison CSV uses explicit percentage column names.

No new FID/KID/HWD evaluation or clean-data retraining was performed. These results identify a concrete data defect, verify the supplied teachers, and narrow the architectural hypotheses; they do not promise a particular future KID/HWD score.
