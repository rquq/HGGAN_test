# MAIN: gain-only spatial style modulation experiment

Date: 2026-10-24, Asia/Saigon (UTC+07:00).

Status: initial gain-only implementation CPU-verified. The subsequent
[shared-width/objective update](2026-10-24-shared-memory-balance.md) sets current
DEV/MAIN memory to 64D and records the combined publication. The 128D counts and
settings below describe the initial handoff-only implementation, not the final
published settings. No full GAN quality training was performed.

## Approved experiment

The user approved testing the previously discussed gain-only alternative, despite
its potential to reduce the local branch's expressive freedom. MAIN is now based
on the current DEV shared-backbone reference-memory code. DEV and ALPHA are left
unchanged.

```text
Reference -> shared frozen B
                +-> deep context/pooling -> global 32D -> all GBlocks
                +-> intermediate map -> projection -> spatial memory 128D
                                                               |
Text -> content context -> seed -> GBlock1 -> cross-attention <--+
                                              |
                                       residual channel gains
                                              |
                                  multiply GBlock1 features
                                              |
                                         GBlocks2-4 -> image
```

Same retrieval, different handoff:

```text
DEV:   H_out = H + scale * out_proj(Attention(Q(H), K(M), V(M)))
MAIN:  H_out = H * (1 + scale * gain_proj(Attention(Q(H), K(M), V(M))))
```

MAIN's `SpatialStyleModulation` replaces the additive reference feature path with
a per-target-position, per-channel residual gain. It has no additive beta term.
The gain and existing channel scale remain unconstrained; no sigmoid ceiling,
tanh clamp, normalization cap, confidence network, character-ID rule or
reference-length fallback was added.

This is a gain-only, spatial attention-conditioned adaptation of the general
feature modulation principle in [FiLM](https://arxiv.org/abs/1709.07871), not a
reproduction of the complete FiLM architecture and not literature proof of a
handwriting-quality improvement.

## What stayed matched to DEV

- One shared frozen style backbone, existing global head/context and one 32D
  latent token.
- The same 128D spatial memory, four pooled rows and at most 32 pooled columns.
- Same Q/K/V projections, four attention heads, total attention dimension 128,
  keys-only positional code, and retrieval after GBlock1.
- Same generator stages, global D, StrokePatchD, OCR and writer teachers.
- Same losses, loss schedules, dataset/polarity assumptions, augmentation, LR
  schedule, FP32 execution, EMA, sampling and evaluation protocol.
- All nine YAML files are semantically identical to DEV. Normal GAN configs use
  batch size 8, critic ratio 2, G/D/P LR 2e-4, 80 epochs, decay after 36, and LR
  floor 0.15x. Local settings remain DEV's local settings.
- GAN resume defaults are already empty; no extra resume mechanism was added.
- W&B remains losses, metrics, samples, resources and normal run metadata; no
  new fusion diagnostics were added.

Parameter count is identical to DEV at each height:

| Height | G | E | G+E |
|---|---:|---:|---:|
| 32px | 3,967,069 | 961,713 | 4,928,782 |
| 64px | 3,971,229 | 961,713 | 4,932,942 |

The 64px final upsampling stage has the inherited additional shortcut projection;
this is the same in DEV and MAIN, not a modulation-specific capacity change.

## File scope and repository preservation

Synchronized the DEV differences in the nine files under `main/configs`, plus
`main/metric/ocr_eval.py` and `main/networks/{module,fusion,BigGAN_networks,model}.py`.
The net diff from old MAIN is larger because old MAIN still used learned local
slots; relative to current DEV, the product change is the modulation class,
projection name/formula/docstring and two neutral error-message wording changes.

Did not copy or replace `.git`, branch names, graph caches, runtime results,
notebooks or pretrained weights. The five MAIN teacher checkpoint files are
byte-identical to the corresponding DEV files. They are outside the product diff.
DEV and ALPHA working trees are clean after the changes.

## Training and checkpoint compatibility

Start a fresh GAN experiment in MAIN. Reuse the existing 32px or 64px W+B and R
teachers. No teacher pretraining is required by this change.

The module name is `SpatialStyleModulation` and the output weight is named
`gain_proj.weight`. A DEV additive checkpoint expects `out_proj.weight`; strict
loading rejects it rather than silently treating it as a modulation run. A fresh
round-trip of the new architecture passes. No partial-load shim was added.

Local command from `HGGAN_test/main`:

```bash
python train.py --config configs/gan_iam_32.yml
```

Use `configs/gan_iam_64.yml` for 64px. In the notebook, select MAIN and no GAN
resume checkpoint for this experiment. Existing notebook data-link handling is
unchanged.

## Verification

- Six gain-only behavior tests were written and failed before the new module
  existed, then passed. They cover a hand-calculated multiplicative result,
  no injection into zero features, missing/empty memory, NaN-poisoned padding,
  finite Q/K/V/gain/memory gradients, and rejection of additive state keys.
- Ten inherited shared-memory contracts pass for MAIN, including both heights,
  one shared B forward, cached features, padding invariance, mixed random/style/
  reconstruction routing, first-GBlock placement and strict round-trip.
- Real native-data train-loop smokes passed at 32px and 64px: two critic batches,
  one G/EMA update, R1 enabled, strict real teacher loading, frozen teachers,
  real sample-grid generation, guided/random generation, and independent OCR
  generation-interface wiring. OCR was stubbed only in the interface smoke and
  is not a recognition-quality result.
- Additional existing current-contract suites passed: background consistency
  8/8, checkpoint/evaluation regressions 10/10, StrokePatchD 9/9. Together with
  the 16 new/shared-memory contracts, that is 43 focused tests passing.
- All 30 tracked MAIN Python files parsed, all nine YAMLs matched DEV semantically,
  teacher hashes matched, and `git diff --check` passed.
- An independent read-only reviewer reran the 16 primary tests and found no
  Critical/Important product issue. Diagnostic report precision issues were
  corrected and the checkpoint probes rerun.

### Test-suite limitation, explicitly not all-green

`pytest` is not installed in the existing ML environment. No dependency was
installed just to change this. `unittest discover -s tests -p 'test_*.py'` did run
the historical ignored MAIN suite: 71 tests, six failure cases and 31 errors.
These include old 8-token/learned-slot interfaces, retired `use_local_style`
options, former LR expectations and source-text assertions. They are recorded
in the local `agent/tests/main_existing_unittest_suite.log`, not silently hidden.
They were not changed or deleted to produce an all-green claim.

The focused current-contract tests and real-loop smoke establish the new wiring;
they do not establish that every historical test remains applicable or that every
possible Kaggle/CUDA run is crash-free.

## Limitations and intended comparison

This remains an ablation. Gain-only modulation can shift effective authority
toward the global path and may weaken some writer-specific letter construction.
It removes direct additive reference feature injection, not every possible path
to artifacts. Later convolutions and learned gains can still distort glyphs.

The known completely black-reference/global-vector failure is intentionally not
fixed here; changing it would introduce another variable. No dimensions were
reduced based on the companion capacity audit. Compare fresh MAIN against DEV at
matched epochs and identical protocols, including short/full-reference samples,
independent CER/WER, and FID/KID/HWD. A lower FID alone is insufficient evidence
that this handoff is better.

Companion analysis: [DEV memory capacity audit](analysis/2026-10-24-dev-memory-capacity/REPORT.md).
