# DEV spatial reference memory experiment

Historical stage 1, implemented on 2026-10-23. This document describes the initial
original-image-only implementation and its checks. The current two-view design,
including the subsequently requested high-frequency branch, is documented in
[the dated implementation report](2026-10-23-dev-dual-reference-memory.md).

Approved design: copy MAIN's versioned code and pretrained teachers into DEV,
preserving DEV's repository, datasets and runs. Keep MAIN and ALPHA unchanged.

Replace the encoder's four 32D learned local slots with spatial evidence from
the intermediate (128-channel) frozen-backbone map. Pool each sample's valid
map to four height rows and at most 32 width columns, then project to 128D.
Carry positions in attention keys and visual features in values. Preserve one
32D global latent and its existing objectives and block-specific projections.
Apply one 128D/four-head retrieval residual after GBlock1.

Random-style rows have no reference memory and bypass retrieval. Transfer and
reconstruction share the real reference memory; the combined three-way G batch
uses empty validity masks only for bypassed rows, never an all-masked softmax.
Global-only objective calls skip memory construction. Existing contextual
feature supervision still uses all three backbone maps. No new loss, high
frequency encoder, teacher architecture, evaluation metric or schedule is added.

Implementation scope: encoder, spatial attention, G arguments, training batch
routing, sample/evaluation/interactive paths, independent OCR generation and GAN
configs. New GAN training is required; unchanged W+B and R teachers are reused.

Verification: CPU FP32 shape/gradient tests at 32px and 64px, mixed rows with
no memory, padding invariance and valid-region pooling, VAE/global-only paths,
strict checkpoint roundtrip, and integration of training/evaluation callers.
Tests and generated evidence stay under agent rather than product repositories.

## Implemented interfaces

`E(..., return_memory=True)` returns `(encoded, memory, valid_mask)`.
`encoded` is `[B, 1, 32]`, or the usual `(sample, mean, logvar)` in VAE mode.
`memory` is `[B, N, 128]` with `N <= 128`; its mask is boolean `[B, N]`.
Without `return_memory`, E retains its tensor/feature-return interfaces for the
existing style reconstruction, content-adversarial and contextual objectives.

`G(global_style, labels, label_lengths, reference_memory=..., memory_valid_mask=...)`
uses the memory after the first generator block. Omitting both optional inputs
performs random/global-only generation. In the three-way training batch, rows
are ordered random/transfer/reconstruction. Empty random-row masks bypass the
attention module's softmax; transfer and reconstruction reuse the same memory.

Pooling crops each sample to its valid backbone width before adaptive averaging.
The batch then pads the pooled maps, preventing unrelated longer references
from changing the short reference's pooling bins. One shared 4-by-at-most-32
policy serves both 32px and 64px; there are no text/character-specific branches.
The per-sample pooling uses a small length-list CPU transfer, so training speed
and GPU memory still require measurement on Kaggle.

## Verification completed

- Eight CPU FP32 contract tests passed, including per-sample padding invariance,
  mixed random/reference batches, masked NaNs, finite nonzero Q/K/V and memory
  projection gradients, VAE/global-only paths, and strict G/E state roundtrips.
- The actual GAN training loop passed at both 32px and 64px, with two native IAM
  batches, one G/EMA update, R1 enabled, and the existing pretrained W+B/R states
  loaded strictly. Teacher parameters remained frozen and received no gradients.
- The actual sample_images and guided/random image_generator paths passed at
  both heights. The independent OCR generation interface also passed using EMA
  G/E and a stub OCR predictor; this is wiring evidence, not OCR quality evidence.
- MAIN and ALPHA remain unmodified. DEV retains its Git identity and existing
  run/data artifacts; pretrained teacher files were copied from MAIN unchanged.

No full GAN training, FID/KID/HWD rerun, CUDA/T4 test, commit or push was performed.
The new architecture must start a fresh GAN run rather than resume old local-slot
G/E/optimizer states. W+B and OCR pretraining is not required again for this change.
