# MAIN / ALPHA checkpoint review — 2026-11-15

Inputs: `chkpoints/15_11_2026/best_kid_0.049210_main.pth` and
`last_fid_6.2710_alpha.pth`. Source inspected: MAIN `d881b59`, ALPHA `14f6eea`.
This is a checkpoint/history audit and a bounded CPU inference diagnostic, not
a new full FID/KID/HWD evaluation. The saved CSV was inspected, not just names.

## Outcome

MAIN is numerically healthy and substantially ahead on the saved metrics. There
is no evidence here of missing optimizer updates, NaN corruption or an encoder
that has trivially collapsed. Its remaining problems are conditional behavior
and what the evaluation measures, rather than a demonstrated broken GAN loop.

| Saved evidence | MAIN | ALPHA |
|---|---:|---:|
| Epoch / metric-history rows | 73 / 73 | 69 / 69 |
| FID, current and saved best | 2.491548 | 6.271029 |
| KID ×100, current and saved best | 0.049210 | 0.265770 |
| HWD, current | 0.091702 | 0.118055 |
| Best HWD in history | 0.079665 at63 | 0.112234 at28 |
| G+E parameters | 5,277,230 | 3,338,589 |
| G/E Adam updates | 303,863 | 287,213 |
| D updates | 607,725 | 574,425 |
| Saved LR | 4.58333e-5 | 1.70213e-5 |
| Evaluated generator/encoder | EMA | Live weights |

At the matched epoch69, MAIN already records FID2.671178, KID0.062865 and
HWD0.085332. Four extra epochs therefore do not explain the whole gap. The
comparison still changes many variables and is not a causal module ablation.
KID0.049210 is raw KID approximately0.00049210, not4.92%: the implementation
multiplies the polynomial MMD estimate by100.

## Problems and evidence boundaries

1. **Short references remain unstable.** A native CPU probe used G_EMA/E_EMA/B
   only, four IAM32 references and five fixed targets. All four references happen
   to be writer0. Word references produce readable output; a `.` reference
   produces grey/slanted output and less clear uppercase strokes. The dot's
   latent L2 distance to word-reference codes is22–25, compared with5–6.6 among
   the three word references. This is actual conditional sensitivity, but one
   writer/four examples cannot establish general writer invariance, collapse,
   or a specific faulty layer. A dot provides very little handwriting evidence.
   A fix should not hardcode punctuation, character IDs or reference lengths.
2. **Best KID is not best HWD.** MAIN's HWD is15.1% above its best at epoch63,
   despite its newest best FID/KID. This is a saved-metric tradeoff, not a
   statistically established regression. SWD matches feature populations and
   does not by itself guarantee a particular reference's writer identity.
3. **The metric stream is same-transcript, reference-guided generation.** Both
   saved configs use `use_rand_corpus:false`; evaluation encodes the reference
   and generates its own transcript. Low FID/KID therefore does not establish
   arbitrary-target spelling, digits, capitals or conditional writer matching.
   No CER/WER/WIER results are stored in these checkpoints. Their removal from
   GAN training is respected; this review does not silently re-enable them.
4. **The configured evaluation split is `test`.** Epoch-wise selection on that
   split makes these test-selected results if it is the final benchmark test.
   This does not make the FID/KID formula wrong or prove leakage into training;
   it limits a claim of an untouched final held-out estimate. Apply the same
   declared benchmark convention to every comparison, and use a separate locked
   test for a stricter final claim.
5. **MAIN versus ALPHA is not a controlled comparison.** Capacity differs58.1%,
   style dimension is128 versus32, EMA is MAIN-only, LR differs2.69×, and
   references, losses and augmentation differ. Do not attribute the gap to
   ContentContext, SWD, PatchD or BSSP alone.
6. **History lacks per-row source/data identity.** Neither file stores source
   commit/data hashes for each historical metric row. Current strict loading
   proves compatibility now, not an unchanged metric protocol throughout all
   resumed runs. Both checkpoints explicitly resume earlier runs.

One minor utility bug is conditional, not an explanation of these scores:
MAIN's RESULTS helper takes the minimum for PSNR/MS-SSIM, although those metrics
are higher-is-better. Both are disabled in these runs. Do not transplant that
summary logic blindly into ALPHA; FID/KID/HWD reporting is not affected by it.

## Checks that passed

- G/E/D/P/R/W/B strictly load from each checkpoint's saved configuration into
  its compatible current branch, with no missing/unexpected keys or shapes.
- All saved model tensors and optimizer moments are finite. All trained
  G/E/D/P parameters have compatible optimizer state. MAIN and ALPHA updates
  agree with two critic updates per generator update.
- ALPHA's P parameters are inside OPT.D. The absence of separate OPT.P is not
  an untrained PatchD bug.
- MAIN's embedding rank is81 and global-mean projection rank128; fusion scales
  are finite/nonzero. These reject trivial stored zero/rank defects, but are
  not substitutes for gradient measurements or image-quality evaluation.
- All20 probe outputs are finite native32 images with exact16px/character widths.
- MAIN's embedded CSV agrees with all73 structured history rows. ALPHA has69
  structured rows but no embedded CSV payload.

## What I recommend next for MAIN

Keep this checkpoint; nothing found requires restarting MAIN. Finish the planned
schedule before changing several hyperparameters. Preserve the epoch63 HWD-best
as a diagnostic control if it is available. Do not raise LR or add a new loss
based only on a noisy final metric row.

The highest-value next diagnostic is a prospective, fixed target/reference panel
with matched-writer short and normal references, including lowercase, capitals
and digits. Score style/reference behavior separately from same-text FID/KID.
Do not tune on these four displayed examples and then present them as held out.
If changing training later, isolate reference robustness from SWD/identity-weight
tradeoffs instead of hiding several changes under one contribution.

## Local evidence

Detailed inventory, hashes, full histories, extracted CSV, strict-load/optimizer
maps and probe measurements are preserved under
`agent/analysis/2026-11-15-main-alpha-review/`. The contact sheet is
`main_native32_ema_contact_sheet.png`. Generated probes/scripts remain local,
not model-source files or newly published benchmark results.

Code anchors: MAIN `networks/model.py:1060` (evaluation stream), `:1840`
(style normalization), `:2371` (distribution loss), `:2510` (KID selection),
`metric/val_metrics.py:468` (KID scale); ALPHA `networks/model.py:669` (evaluation).

No GPU, training, full metric rerun, teacher retraining, architecture change or
hyperparameter edit was performed for this diagnosis.
