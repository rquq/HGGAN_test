# DEV epoch 34: spatial-memory dimensions, redundancy and gradient audit

Date: 2026-10-24, Asia/Saigon (UTC+07:00).

Checkpoint: `/home/quq/machineLearning/HTG/chkpoints/24_10_2026/last_fid_2.6501_dev.pth`.
Architecture: current shared-backbone reference-memory DEV, not the earlier learned
local-slot encoder. No source, weights, data or checkpoint were modified by this
audit, and no optimizer was stepped.

Follow-up: the [shared DEV/MAIN experiment](../../2026-10-24-shared-memory-balance.md)
subsequently applies a learned 64D memory and a modest objective rebalance. All
audit numbers below still refer to the original 128D epoch-34 checkpoint; they
are not results for the new model.

## Conclusion

The memory has not collapsed to one direction, identical slots, duplicate heads
or dead channels. It is active and spatially consequential. However, its 128-channel
representation is statistically redundant: roughly 30 principal directions
explain 99% of pooled variation in the tested images. This is a compressible,
anisotropic representation, not evidence that 98 channels stopped learning.

The distinction matters: all channels receive fresh gradients and saved Adam
second moments are nonzero; nevertheless variation lives mostly in a smaller,
rotated subspace. Removing individual named channels is not equivalent to
retaining that subspace. Low-variance directions also retain gradient sensitivity.

If testing a smaller memory later, 64 channels is the conservative first ablation;
32 is a more aggressive candidate. Neither was applied to DEV or MAIN in this
task. MAIN's new handoff experiment retains 128 channels to isolate its variable.

## What "local dimensions" means here

There is one global token `[B,1,32]`. Local evidence is a variable spatial matrix
`[B,N,128]`, with four height rows and at most 32 columns, not 4/8/12 learned local
style slots. Two axes must not be conflated:

- N: number of spatial reference positions.
- 128: feature channels at every position.

The audit measures both channel variation and whether retrieval varies across
target positions. It does not assign allograph labels to channels or prove
writer/content disentanglement.

## Data and methods

All execution used CPU FP32 and two Torch threads, with double-precision linear
algebra for covariance/spectra where appropriate. Spectral and functional probes
use strictly loaded G_EMA/E_EMA with the saved frozen B. Saved optimizer moments
and fresh loss-gradient probes refer to raw G/E training weights.

- Calibration: 256 deterministic random native IAM32 training references.
- Held-out covariance: 128 random native test references plus 32 additional
  short references. Calibration and held-out test sets are separate.
- Functional interventions: 48 held-out references, 24 random and 24 short, with
  fixed target words and fixed global vectors.
- Fresh gradients: six two-image training-reference batches, three short and
  three longer, with the saved teachers/critics and representative existing
  weighted GAN objectives. No training update was performed.

Covariance uses equal reference weight so long words do not dominate simply by
having more memory entries. It is decomposed into within-reference spatial
variation and between-reference mean variation. Effective rank means entropy
rank of covariance eigenvalue energy, not numerical matrix rank. Per-position
LayerNorm creates a channel-sum constraint, so rank below 128 is expected even
without learned collapse.

## 1. Activation spectra: redundant, but not rank-one collapse

| Representation | Effective rank | Directions for 90% energy | 95% | 99% | Largest direction |
|---|---:|---:|---:|---:|---:|
| Calibration total | 18.86 | 16 | 20 | 30 | 12.27% |
| Held-out random total | 18.96 | 16 | 20 | 30 | 12.14% |
| Held-out random within-reference | 18.79 | 16 | 20 | 31 | 12.69% |
| Held-out reference means | 9.68 | 8 | 11 | 19 | 24.41% |
| Extra-short-reference total | 18.11 | 16 | 20 | 30 | 14.82% |

The strong agreement between calibration and held-out covariance is evidence
that this is not merely one reference's accidental low rank. A single held-out
image's centered spatial matrix has mean effective rank 11.46, median 12.06,
range 6.57-14.52. Different references nevertheless populate a broader common
subspace, so a per-image rank of 11 does not mean 11 channels suffice globally.

The pooled B source map already has effective rank 21.31. Its top 32 directions
explain 96.45%, versus 99.22% in the projected held-out memory. Thus a substantial
part of the anisotropy comes from the frozen visual representation; E compresses
it somewhat further. It is not all created by a failed memory projection.

### Are channels duplicates or dead?

- No channel has variance below one-millionth of the median channel variance.
- Mean absolute off-diagonal channel correlation: 0.185; 95th percentile: 0.444.
- No channel pair has absolute correlation above 0.95 in this probe.

Low joint rank and modest pairwise correlation can coexist: many channels can
encode different mixtures of the same smaller latent subspace. This is why
dropping an arbitrary subset of channels is not equivalent to PCA compression.

## 2. Weight matrices: full numerical rank

EMA matrices, numerical rank threshold `1e-5 * largest singular value`:

| Matrix | Shape | Numerical rank | Effective energy rank |
|---|---|---:|---:|
| E memory projection | 128 x 128 | 128 | 37.29 |
| Query projection | 128 x 256 | 128 | 73.81 |
| Key projection | 128 x 128 | 128 | 59.95 |
| Value projection | 128 x 128 | 128 | 54.08 |
| Output projection | 256 x 128 | 128 | 66.14 |

No matrix has collapsed to a one-dimensional projection. The memory projection
is ill-conditioned relative to an orthogonal map (condition number about 1217),
but that alone is not a numerical failure: memory tensors and measured gradients
remain finite, and its representation is intentionally anisotropic.

The scaled weight-space product `diag(reference_scale) @ W_out @ W_value` has
effective rank 20.61 and full numerical rank 128. This is not the complete
multi-head attention operator: each head uses a different attention matrix.
It is included only as a channel-transform diagnostic, not an equivalence proof.

## 3. Attention heads are not duplicates

On 48 held-out reference/target cases:

- Mean pairwise total-variation distance between head attention maps: 0.730.
  Zero would mean identical maps; these heads do not select identical evidence.
- Mean normalized attention entropy: 0.533, rather than uniform attention.
- Zeroing one head's V output changes images by the following mean absolute
  amount on the [0,1] image scale:

| Intervention | Mean output pixel difference |
|---|---:|
| Remove head 0 | 3.72% |
| Remove head 1 | 6.68% |
| Remove head 2 | 5.36% |
| Remove head 3 | 4.96% |

All heads have a functional effect. A larger difference is not automatically a
better head: useful style and harmful artifacts can both cause sensitivity.

## 4. Does memory merely provide another global vector?

It has both common and target-position-varying effects.

For the actual additive update, including the learned channel scale, an average
76.34% of squared update energy is in a component constant across target query
positions. The remaining approximately 23.66% varies spatially. The centered
query-varying update has mean effective rank 6.44.

That common component can carry shared stroke/background/writer information; it
is not automatically redundant with the explicit global latent. A fixed-ridge
linear predictor trained on 256 references explains about 45.4% of held-out
memory-mean variance from the global vector, and the reverse explains about 46.2%
of held-out global variance. There is overlap, but not linear equivalence. This
does not prove semantic independence or writer-only information.

Two interventions distinguish a strong common component from complete collapse:

- Replace each retrieved update with its mean across target positions, preserving
  the reference-dependent common update: image difference 8.45% on average.
- Remove the entire reference memory path: difference 16.42% on average.

Thus spatial retrieval is consequential beyond a common reference bias. It has
not reduced to an identical update at every location.

## 5. Functional dimension compression

A PCA basis was fitted only to calibration within-reference variation. On each
held-out reference, keep its full 128D mean unchanged and reconstruct only its
spatial deviations with k principal directions. The explicit global vector,
target, trained weights, entry count and original attention width stay unchanged.

| Kept spatial directions | Spatial variation energy retained | Mean image MAE | Worst of 48 image MAEs |
|---|---:|---:|---:|
| 8 | 67.32% | 3.891% | 8.119% |
| 16 | 90.47% | 1.604% | 3.327% |
| 32 | 99.13% | 0.515% | 1.118% |
| 64 | 99.93% | 0.109% | 0.270% |
| 96 | 99.995% | 0.026% | 0.045% |

This is meaningful compression evidence. But it is not a trained 32D or 64D
encoder: the full 128D per-reference mean remains, projections still operate at
128D, and no FID/KID/HWD/CER evaluation was rerun for these interventions. Small
pixel MAE can conceal an important rare-letter difference.

The conservative first retraining ablation would be 64D memory, not 16D. Changing
only memory width from 128 to 64 while keeping attention width 128 saves about
24,576 projection weights, roughly 0.5% of G+E at 32px. It halves stored memory
activations, but leaves most convolutions and attention width unchanged. Do not
expect a large overall speedup from this change alone.

## 6. Saved optimizer evidence

All inspected memory-projection, Q/K/V/output and residual-scale parameters have
Adam step 141,525, consistent with the checkpoint's G update count. Every entry
in their second-moment tensors is nonzero. All 128 memory-projection output rows
have nonzero historical second-moment RMS.

These are exponential moving summaries of training gradients, not saved
instantaneous `.grad` tensors. They also reflect the actual training clipping
and optimizer history. A nonzero second moment proves past updates, not that
every direction has equal usefulness or never became redundant later.

## 7. Fresh gradients: not dead, and not confined to the largest-variance PCs

The raw checkpoint was loaded into disposable CPU models. Six two-image batches
used existing adversarial/global+patch, transfer CTC, writer, contextual,
reconstruction, global-cycle/info/content/KL and random CTC terms. Objectives
use the epoch-34 weights. BN was held in eval mode for stable attribution;
gradients are before clipping. This is not a historical training replay.

All tested gradients were finite. All 128 memory channels receive a nonzero
total gradient across the six batches. Actual per-channel RMS over 760 valid
memory-token gradient observations ranges from about 0.01075 to 0.16093; the
median is about 0.01679. Aggregate L2 is reported separately in JSON rather than
mislabeling it as RMS.

Mean weighted gradient L2 on E's memory-projection matrix:

| Existing objective | Mean gradient norm |
|---|---:|
| Writer identity | 11.710 |
| Global + patch adversarial | 1.914 |
| Transfer CTC | 0.849 |
| Reconstruction | 0.770 |
| Contextual | 0.317 |
| Combined remaining terms | 0.143 |

Writer identity exerts the strongest pressure in this selected diagnostic sample.
That is a reason to investigate objective balance, not proof it dominates every
historical step or that its coefficient must be changed. The six batches include
writer/reference-dependent variation; G/E BN modes differ from the training
forward. No loss weights were changed as a consequence of this audit.

The top 32 calibration spatial-variation PCs contain only about 31-37% of fresh
memory-gradient energy across objectives; the top 64 contain about 60-65%.
Therefore low activation variance does not equal zero loss sensitivity. The PCA
basis comes from EMA calibration activations while gradients come from raw
weights, another reason to interpret this as a sensitivity diagnostic rather
than an exact training-subspace decomposition.

Reconstruction/CTC gradients on memory point in conflicting directions in some
batches and aligned directions in others; there is no consistent six-batch
conflict that would justify deleting one objective outright.

## Practical recommendation

1. Do not describe this checkpoint as local-style collapse. Its failure is more
   consistent with useful but content-contaminated reference evidence than an
   unused local branch.
2. Do not increase memory width above 128 based on these measurements. There is
   no evidence here that lack of channel capacity is the bottleneck.
3. If a later capacity experiment is desired, test 64D first and retain identical
   evaluation/reference cases. Test rare-letter behavior as well as aggregate
   metrics. A 32D retraining is plausible but less conservative.
4. Keep the approved MAIN modulation experiment at 128D. Combining the two
   changes would obscure whether the retrieval handoff or capacity caused a result.
5. Do not prune named channels or treat the PCA probe as a benchmark improvement.
   Preserve the fixed dataset and metric protocol.

## Artifacts and limits

- [Spectra, channel correlations, weight ranks and saved Adam summaries](spectral_audit.json)
- [Per-reference compression/head/query interventions](functional_audit.json)
- [Fresh loss-specific gradients](gradient_audit.json)
- Calibration basis and indices are saved locally in `calibration.pt`; it is a
  diagnostic artifact, not a deployable replacement checkpoint.
- Local helper: `HGGAN_test/agent/tests/audit_dev_memory_capacity.py` (Git-ignored).

No GPU job, optimizer update, full metric validation, multi-seed run, teacher
retraining or production dimension change was performed. This audit describes
the supplied epoch-34 checkpoint, not the future behavior of a fully trained DEV.
