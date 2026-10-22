# DEV: two-view spatial reference memory

Date: 2026-10-23, Asia/Bangkok (UTC+07:00).

DEV implementation commit: `efca5aaa5d3bdb384242c85ce58788650a635cfd`
(`feat: add dual-view spatial reference conditioning`).

## Scope and source

DEV was synchronized from MAIN at `dbd906615fa531f5925a381289f1e05c1995185c`,
including the existing independent CER/WER evaluator and pretrained teacher
files. Its Git repository, data links and training runs were preserved. MAIN
and ALPHA are unchanged. The DEV changes include that synchronization, the
spatial-memory implementation, the high-frequency extension below, config
reorganization and W&B cleanup.

The preceding implementation was the first, original-image-only stage of the
proposed experiment. It deliberately deferred the high-frequency branch. The
user's latest requested diagram includes both sources; this revision implements
that complete two-view path.

## Implemented architecture

```text
Reference image ── existing frozen B ── intermediate spatial map ── pool/project ──┐
                                                                                ├─ learned merge ─ memory
Reference image ── signed Laplacian ── small trainable CNN ── pool/project ────────┘
                                                                                                  │
Text ── content context ── spatial seed ── GBlock1 ── one cross-attention ◄──────────────────────────┘
                                                          │
                                            GBlocks2–4 ── image head

Reference image ── existing B/global encoder ── 1 × 32D global latent
                                                      └─ block-specific global conditioning
```

The two image views produce aligned evidence on the same normalized spatial
grid. Their receptive fields are different; alignment here does not assert
pixel-identical receptive fields. The memory is `[B, N, 128]`, with four height
rows and at most 32 width columns, so `N <= 128`. It is separate from the global
latent `[B, 1, 32]`. Two views do not double the number of memory positions.

The existing B remains frozen and supplies its intermediate 128-channel map.
Each sample is cropped to its valid feature width before adaptive pooling, then
batch-padded. The original path uses a learned 1x1 projection to 128 dimensions.
All three original B maps remain available to the unchanged contextual loss.

The high-frequency view uses a fixed signed 3x3 Laplacian divided by four.
Each sample's own edge is extended before filtering, then padding responses are
cleared. This prevents unrelated batch padding from creating a false boundary
and makes any constant image produce a zero high-frequency view. This is a
filter on the reference input to an architectural branch; dataset contents,
image polarity, generated output and metric preprocessing are unchanged.

The trainable high-frequency CNN contains a 1-to-32 channel strided stem,
32-to-64 channel downsampling, then a depthwise 3x3/pointwise 1x1 residual block.
It uses per-position channel LayerNorm rather than spatial/batch statistics.
Its first stride follows the existing geometry rule, producing comparable
sampling density at 32px and 64px with shared architecture and channel widths.
It has **23,712 parameters**. A separate 64-to-128 projection maps its evidence
to the reference-memory dimension.

Let `P` be normalized original-image features and `F` normalized high-frequency
features, after pooling/projection to the common grid. The merge is

```text
gate   = sigmoid(Linear(concat(P, F)))
memory = LayerNorm(P + gate * F)
```

The gate is per position and channel. Its weights start at zero and its bias at
-2, giving an initial high-frequency contribution of about 0.119 before the
final normalization. All gate parameters are trainable. This is initialization,
not a character-dependent condition or an imposed fixed contribution.

The complete extension adds **64,800 parameters** over the raw-memory encoder:
E changes from 961,713 to 1,026,513 parameters. Activation cost and actual T4
training speed must be measured; parameter counts do not establish throughput.

GBlock1's spatial features query this memory through one 128D, four-head cross-
attention. Positions affect keys; values carry visual evidence. Its learned
residual scale is preserved. There is no second retrieval for the high-frequency
view, separate allograph classifier, extra contrastive loss or second full B.
The general use of original/high-frequency reference evidence has precedent in
[One-DM](https://arxiv.org/abs/2409.04004); this smaller GAN encoder and merge are
our experimental implementation, not a reproduction of One-DM's full system.

## Training and evaluation routing

- Random-style generation samples only the 32D global latent and bypasses
  reference retrieval. Gaussian local/spatial features are not fabricated.
- Style transfer receives the encoded global latent and merged reference memory.
- Reconstruction receives the same global latent and reference memory.
- The three outputs remain batched through G. Empty random-row memory masks
  bypass attention before softmax; no all-masked softmax is evaluated.
- Only the global posterior receives KL regularization. Existing global style,
  content and reconstruction losses preserve the baseline normalization divisor.
- G/E EMA, samples, FID/KID/HWD generation, interactive reference generation and
  independent OCR generation all use the same reference-memory interface.
- Calls used only for global style/content losses skip both memory branches.

The new CNN, projections and merge are part of E, so the existing G/E optimizer,
gradient clipping, EMA and checkpoint saving cover them automatically. W+B and
R teachers are unchanged, strictly loaded and frozen. There are no character ID,
punctuation, digit, writer or transcription checks in either memory branch.

`EncModel.use_high_frequency: true` enables the new path in GAN configs. False
constructs the original-image-only encoder for a fresh ablation. The global
style dimension remains 32 and memory dimension remains 128 in both cases.

## Configuration cleanup

All nine existing DEV YAML filenames are retained. Sections now group runtime,
data geometry, training, evaluation, model components and inference options.
Training keys are grouped by initialization, loaders, optimization/cadence,
schedule, EMA, text sampling, objectives, regularization and outputs. Evaluation
groups metric switches, independent OCR, feature extraction and KID estimation.

Parsed YAML dictionaries were compared before/after reordering: all values were
preserved, except adding the explicit `use_high_frequency: true` setting to GAN
encoders. This cleanup does not retune LR, critic count, batch size, losses, EMA,
teacher selection or metric protocols. Both normal GAN configs retain batch 8,
critic cadence 2 and G/D/P learning rates 0.0002.

## W&B cleanup

Removed fusion strength and fusion-scale minimum/maximum from training meters,
the console progress line and W&B scalar payloads. No high-frequency gate or
attention diagnostics were added. Kept losses and loss components, evaluation
metrics, generated samples, epoch/learning rates, startup parameter summaries,
the existing direct Logs bridge and W&B's normal system/resource monitoring.
The monitor changes do not remove trainable residual-scale parameters.

## Verification on 2026-10-23

Ten CPU FP32 contract tests passed. They cover both heights, finite/nonzero
gradients through original projection, high-frequency CNN/projection, merge
gate and Q/K/V, uniform-paper responses, poisoned batch padding, reference
padding invariance, mixed random/reference batches, global-only calls and
strict G/E checkpoint roundtrips. A fixed encoder's high-frequency intervention
changes memory while preserving the global latent.

The real DEV training loop also passed at 32px and 64px using native IAM samples
and the existing pretrained W+B/R states: two critic batches, one G/EMA update,
R1 enabled, and real sample-grid generation. Both teacher routes remained
frozen and received no gradients. High-frequency stem gradient norms were
approximately 0.711 and 0.757 respectively; these are gradient-flow diagnostics,
not quality scores.

Guided/random image generation and the independent OCR generation interface
passed. The latter used an OCR stub to test routing without claiming recognition
accuracy. A captured W&B payload contained losses, LR/epoch and samples and no
fusion diagnostics. All product Python sources parse and nine YAML files load.

Runtime helpers/results are under the local `agent/tests/` directory and ignored
by Git. They are not included in code or documentation commits. Historical run
artifacts and unrelated dirty files in the outer checkout are not staged.

## Retraining and limits

Start a **fresh DEV GAN run**. Old learned-local-slot GAN checkpoints, and the
earlier raw-memory-only GAN states, do not represent this architecture. Reuse
the existing W+B and R teachers; this change does not require pretraining them.
No compatibility loader or silent partial weight loading was introduced.

No full training, FID/KID/HWD quality evaluation or CUDA/Kaggle T4 run was performed.
The tests establish implementation/runtime contracts, not an improvement in KID,
allographs or training speed. A fair comparison keeps data, evaluation protocol,
schedule and teacher weights matched and reports run/seed variation.
