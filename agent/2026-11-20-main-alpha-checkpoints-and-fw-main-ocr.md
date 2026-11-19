# MAIN / ALPHA checkpoint audit, MAIN target pairing, and FW-GAN→TrOCR

Date: 2026-11-20. Inputs are the two exact checkpoints in
`chkpoints/20_11_2026`. MAIN source before the patch: `f849d9a`.
ALPHA source: `31cab5d`. Read-only evidence, local inference, a separate
benchmark launcher, and one narrow MAIN sampling correction; no GAN or
full TrOCR retraining.

Implementation commits: MAIN `a7606ca`, OCR benchmark `4d98352`.

## Checkpoints and CSV

| Snapshot | Epoch | FID | KID ×100 | HWD |
|---|---:|---:|---:|---:|
| MAIN supplied last | 76 | 2.928871 | 0.072698 | 0.114874 |
| MAIN at the matched checkpoint epoch | 61 | 2.933865 | 0.061596 | 0.097973 |
| ALPHA supplied best FID | 61 | 6.428895 | 0.265485 | 0.128992 |

- MAIN reached both best FID **2.834267** and best KID **0.056662** at epoch
  58 in this history; best HWD **0.086303** at 67. At 76, KID is about 28%
  above its low and HWD about 33% above its low. The supplied weights are
  epoch 76, not a recoverable copy of the epoch-58 model. MAIN's saved
  SWD direction buffers really do have 64 columns at all three scales.
  The older 15/11 MAIN run recorded a lower KID, 0.049210. This is not a
  matched comparison isolating the SWD projection count.
- ALPHA C5-BSSP32 has best FID at epoch 61. Its lowest saved KID is
  **0.263764** at epoch 43 and HWD **0.115350** at epoch 20. The epoch-58/60
  FID spikes and recovery at 61 do not establish a steady late gain.
  The earlier epoch-69 ALPHA checkpoint with FID 6.271029 is not a matched
  control for isolating BSSP; do not claim C5 improved or regressed quality
  on that comparison alone.
- Both embedded CSVs match every structured row (76 and 61). All audited
  model tensors and optimizer moments are finite. MAIN G/E has 316,350
  Adam updates and D/P 632,700; ALPHA G/E has 253,913 and D+P 507,825.
  ALPHA P is part of OPT.D, not untrained. Saved G/D LR is 3.33e-5 in
  MAIN vs 5.11e-5 in ALPHA. ALPHA G/E/B strictly load into its current
  branch, and the checkpoint contains all four BSSP projections.
- MAIN uses 128D global style, EMA and 5,277,230 G/E parameters.
  ALPHA uses 32D global style, no EMA and 3,342,813 G/E parameters. The
  training dataset alias also differs (iam_word vs iam_word_org).
  Both validation snapshots use iam_word/test and same-transcript
  synthesis. The table reports stored values, not newly rerun metric
  parity or a module ablation.

Local only: `analysis/2026-11-20-main-alpha/{main,alpha}_eval_metrics.csv`,
`checkpoint_summary.json` and the audit script. SHA-256 of MAIN:
`052465338af6ea5e79763fd38fea27883cb3ed14ed20ac6589627066d8c9226c`.
SHA-256 of ALPHA:
`76a6aef3da0422d47c20ad55a1e40b2f9f6c4837bbb7be8a1b3d310111b560b3`.

## MAIN sample investigation and changes

The supplied folder has no original sample PNG. In MAIN's seven-band sample
panel the two bands immediately before the target caption are **new target +
random prior** and **the same new target + reference style**. Neither is
same-transcript reconstruction. The current FID/KID/HWD stream runs with
`use_rand_corpus: false`, so score quality there does not certify these bands.

Bounded RTX 4060 inference strictly loaded MAIN G_EMA, E_EMA, B and R.
On eight reproducible IAM32 references, the in-domain R teacher reads
8/8 on same-transcript generation in both branches. The original sampler's
novel targets read 8/8 prior and 7/8 transfer. A fixed mixed uppercase/digit
stress set reads 5/8 and 4/8 (e.g. transfer `2026` → `2Or`, `57` → `S`).
These are diagnostics, not independent OCR results. Simply shortening
random targets does not solve the cases: the length-limited illustrative
set reads 7/8 and 4/8, but changes the words/pairings and is not an ablation.
The maximum valid-pixel difference for singleton vs padded batched G was
under 0.002 on the [-1,1] scale. This does not support padding as the
cause of a whole illegible word. A selected IAM32 train ref labelled "."
also visibly does not resemble a dot; this is not edited or filtered.

Two separate code issues corrected:

1. The preview used a CTC-collapse decoder for ground-truth labels, displayed
   the width-warped D image instead of the actual ref sent to E, and
   overwrote the penultimate target with the last. It now displays the actual
   ref, preserves literal repeats, keeps independently drawn targets, and
   labels routes in the W&B caption. It still shows the stress cases.
   The patched sample method ran using actual epoch-76 EMA weights and
   wrote `analysis/2026-11-20-main-alpha/fixed-sample/iter_632699.png`.
2. In training, MAIN sorts references by canonical width (equal to transcript
   length ×16) and previously sorted sampled targets by length too. In 225
   non-degenerate illustrative batches from IAM32 transcript lengths,
   the mean within-batch length correlation was **0.696** with that ordering
   versus **−0.022** without it, keeping each batch's exact target multiset.
   D and G now retain independent target order. R supports unsorted packed
   sequences. Case mix, corpus, length cap, weights/losses, LR and network
   shapes are unchanged. No new loss, augmentation, character rule, rerank,
   or inference change to the frozen checkpoint. Existing checkpoints can
   load; whether continued training improves novel-target writing is untested.
   Raw G/R from the supplied checkpoint passed native finite CTC backward on
   unsorted lengths [4, 2, 9, 2], without an optimizer update.

The labelled local GPU panel and CSV are `main_native32_probe.png` and
`main_native32_probe.csv` in the same analysis directory. They are not the
missing original W&B panel.

## OCR benchmark — FW-GAN versus exact supplied MAIN e76

The standalone `notebooks/kaggle_ocr_benchmark.ipynb` defaults to
`DATA_AUGMENT=True`, `TRAIN_TROCR=True`, arms FW-GAN and MAIN, seed 123456
and IAM32. It uses 5,000 real + 25,000 paired synthetic samples for each
arm, identical reference images, proposed target strings, sample IDs,
and seeds. Both generators are native32. Each T4 runs a separate TrOCR
arm, FP32, batch32 × accumulation2, eval64, 32 epochs; real-only remains
off for this run. Real validation taken from IAM train chooses by CER
then WER; complete real IAM test gives WER/CER/NED. This is downstream
augmentation utility, not GAN self-OCR. MAIN's upstream GAN monitored
the IAM32 test split; do not present this as an upstream blind-test
comparison. FW's upstream exposure is declared unknown.

Attach `last_kid_0.072698_main.pth` as a **private** Kaggle input, or
set `CHECKPOINT_PATHS['main']`. File bytes are checked against the pinned
hash. It was **not** uploaded to a release. FW-GAN uses its existing
verified release file and pinned source. MAIN uses the independently
pinned compatible pre-fix source `f849d9a35e3928b9f9aaa37066d7d3ad11a36cae`
and only initializes/strict-loads G_EMA/E_EMA/B. MAIN keeps original
native reference width and the trained ink polarity. It does not use
R/W/D/P for export. Both output packages are kept under the existing
Kaggle saved-folder layout. Start a fresh pair with no previous output;
resume this pair's own saved output, and use OCR-only once both exports
are complete. Prior SpiS/HiGAN+ arms can still be selected manually.

Local readiness: exact pinned clean source revisions and GAN checkpoint
hashes produced four native32 records per arm from the same manifest.
Both portable bundles verified, and OCR preflight + per-arm real/synthetic
training identities accepted both. Full 25k export, TrOCR and Kaggle
runtime were not run. 56 focused OCR and notebook tests passed; the two
notebook copies are byte-identical and pass schema/cell compilation.
The referenced old model scores are not OCR scores.

## Next ALPHA increment: C6 content-only seed (chosen, not implemented)

Keep C5's 32D style and four BSSP mappings. In the current ALPHA G, remove
only the repeated style concatenation at the linear spatial seed: feed
contextualized target embeddings to that one projection, leaving style
conditioning in all four GBlocks. This asks one question: is the
style-at-seed route still useful once BSSP is present? It is not the
global-vector deletion or the old spatial memory experiment.

Control: reproduce C5 vs C6 with fixed seeds, native32 training data and
teacher weights, critic/crop policy, losses, steps and evaluation. Compare
matched-epoch FID/KID/HWD plus the same fixed novel-text/ref panel on both,
without adding GAN CER/WER back into training. Compare more than one seed
before assigning causality. C6 changes the filter-linear input shape, so
use fresh GANs for the controlled experiment; no OCR or W+B retraining.
Do not port SWD or MAIN's larger E concurrently. Ledger updated separately;
no ALPHA source or checkpoint changed in this task.
