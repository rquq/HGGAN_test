# DEV: corrected shared-backbone reference memory

Date: 2026-10-23, Asia/Bangkok (UTC+07:00).

DEV correction commit: `69f78bef9ad0fdac8daaca469c83454e530360c1`
(`fix: restore shared-backbone reference memory`).

## Authoritative design

The user corrected the requested diagram after the two-view implementation was
published. This report supersedes the high-frequency design in
[the earlier report](2026-10-23-dev-dual-reference-memory.md).

```text
Reference -> one shared, frozen B
               |
               +-> deep sequence -> global context/pooling -> 32D global vector -> GBlocks
               |
               +-> intermediate spatial map -> reduction/projection -> reference memory
                                                                         |
Text -> content context -> spatial seed -> GBlock1 -> cross-attention <-----+
                                             |            |
                                             +-- queries  +-> GBlocks2-4 -> image head
```

There is one reference backbone and one reference cross-attention in G. The
global latent has shape `[B, 1, 32]`. Reference memory has shape `[B, N, 128]`,
with `N = 4 * min(valid_intermediate_map_width, 32)` before batch padding and a
maximum of 128 entries. These entries are spatial evidence, not learned local
query slots. One global token therefore does not reduce reference attention to
a one-key softmax.

The global path uses B's deep sequence, the existing gated global context,
valid-width pooling, normalization and global posterior projection. Each GBlock
receives its existing block-specific global conditioning. The memory path uses
the same B call's intermediate 128-channel map. Each sample is reduced to four
height rows and at most 32 columns before a small learned 1x1 projection and
per-position normalization. Pooling excludes sample padding before resizing.

GBlock1 supplies spatial queries. Reference memory supplies keys/values; spatial
position is added to keys only. One residual cross-attention refines GBlock1's
features before GBlocks2-4. No extra reference encoder or high-frequency view is
present. The attention's existing trainable residual scale is retained.

## Correction from efca5aa

Removed `HighFrequencyEncoder`, its fixed Laplacian buffer, high-frequency
projection, learned source-merge gate and `use_high_frequency` config option.
These modules are deleted rather than disabled placeholders. E returns to
**961,713 parameters**, removing **64,800** parameters from the two-view version.

The previous commit already contained the requested shared-backbone spatial
memory foundation, so recopying MAIN or resetting branch history was unnecessary.
This is a forward correction on DEV. Its existing MAIN synchronization remains
based on `dbd906615fa531f5925a381289f1e05c1995185c`.

Retained the organized config files, loss/metric/sample/resource logging cleanup,
global-loss normalization, sampling/evaluation routing and the existing MAIN
training settings. The correction changes only the style encoder and removes
the obsolete option from the four GAN YAML files. MAIN and ALPHA are unchanged.

## Training and inference contracts

E obtains both paths from one B forward for a reference. In GAN training, real
B features are cached and reused when E is rerun for gradients; the global and
memory projections remain trainable while B stays frozen. Additional B passes
on generated images for existing teacher losses are separate from this real-
reference encoding contract.

Random generation samples only the global latent and bypasses reference
retrieval. Style transfer and reconstruction receive the encoded global latent
and the same reference memory. Empty random-row masks bypass attention before
softmax. Global-only objective calls skip memory projection. Existing contextual
supervision still uses all three B feature maps.

Sampling, validation and independent OCR generation use the same G/E interface.
There are no transcription, punctuation, digit or writer-specific conditions
in the memory extraction. Valid-padding masks remain necessary metadata and do
not constitute foreground/ink masking.

## Verification on 2026-10-23

Ten CPU FP32 tests passed, including two direct architecture checks:

- For both 32px and 64px, E makes exactly one B forward to obtain global and
  spatial outputs. Supplying cached B features causes zero additional B forwards
  and returns matching global vectors, memories and masks.
- G hook order is exactly `content context, GBlock1, reference retrieval,
  GBlock2, GBlock3, GBlock4`.

Other tests cover finite nonzero memory/Q/K/V gradients, padding invariance,
poisoned masked memory, random-row identity bypass, VAE/global-only outputs and
strict checkpoint roundtrips. No high-frequency/gate state remains in E.

The actual training loop passed at both heights with native IAM samples and
strictly loaded W+B/R teachers: two critic batches, one G/EMA update, R1 enabled,
sample-grid generation, guided/random image generation and independent OCR
generation wiring. OCR predictions were stubbed for that interface test and
are not recognition-quality evidence. Captured W&B payloads contain losses,
LR/epoch and samples without fusion diagnostics. Runtime helpers stay Git-ignored
under the local `agent/tests/` directory.

## Running this experiment

Use DEV's existing `configs/gan_iam_32.yml` or `configs/gan_iam_64.yml` for a fresh
GAN run. Reuse the existing W+B and R teachers. Learned-slot GAN checkpoints and
the superseded high-frequency GAN checkpoints do not match this architecture;
no silent partial-load compatibility was introduced.

No full training, quality-metric rerun or CUDA/Kaggle T4 test was performed.
Implementation checks establish the requested flow, not a promised KID/FID/HWD
improvement or measured GPU speedup.
