# ALPHA BSSP32 and notebook repair — 2026-11-15

User clarified that implementation belongs in ALPHA, not MAIN. No subagents
or Astra model were selected for this implementation. MAIN source is unchanged.

## Implemented scope

- C5-BSSP32: replace the shared linear/split style mapping with four independent
  SN-linear32→32, SiLU, SN-linear32→32 projections. Adds4,224 parameters.
  Preserve the32D encoder/style, existing mixed content/style seed, ContentContext,
  native32/64 geometry, GBlocks, teachers, losses, LR/cadence/EMA/evaluator settings.
- Construct/initialize the new projections inside isolated CPU RNG after the
  baseline initialization. Shared parameters/buffers and subsequent RNG state
  match the previous generator exactly under the same seed.
- ALPHA checkpoints now embed metric CSV/columns and a class/parameter manifest.
  Keep unknown historical metric columns, restore the run CSV immediately on
  checkpoint load, accept older history-only checkpoints, and reject empty
  checkpoint objects instead of silently continuing with random weights.
- RESULTS adds finish UTC time, CSV path and parameter count. Existing FID-best
  selection and last/best identity are unchanged. No MAIN PSNR-minimum logic was
  copied. Interruption/DDP rewrites were not bundled without a dedicated test.
- Kaggle copy is nonblocking; its existing monitor retries pending carry until
  startup/terminal status is available. Both notebooks' monitors use the helper;
  local archived GAN launch code remains commented. Generic inputs preserve
  unique FID/KID bests; explicit best aliases work; ambiguity is reported rather
  than guessed. Atomic same-directory hard-link publication never overwrites an
  existing destination. No new notebook user settings or model startup API.
- Normal launches carry their timestamp; older startup receipts reject. Monitor
  fallback/manual attachment without launch metadata cannot prove the launch's
  identity. Overlapping jobs reusing status files remain unsupported; callers
  must not launch two jobs into the same GPU status/PID files.

## Verification

Focused CPU checks pass at both native32 and native64 using full-width G64:
shared-state/RNG equality; exact +4,224 parameter delta; output geometry;
finite gradients through every BSSP; Adam update; save/strict model+optimizer
reload. Carry fixtures cover missing/late/old startup, generic and explicit KID
aliases, preservation of existing bytes and temporary-file cleanup. Both
notebooks parse and plain Python cells/monitor wiring compile. Unknown CSV
column preservation/export passes. Python compilation/diff checks pass.

The focused test harness needed two harness-only corrections (import path and
AST source locations); neither required changing model behavior. Tests and raw
artifacts remain under ignored/local agent analysis, not model source commits.
No full legacy-suite claim, full GAN training, live Kaggle run, online W&B or
KID improvement is asserted.

## Compatibility

Fresh ALPHA GAN training is required: old G keys contain style_linear, whereas
new G contains bssp. No permissive old-G loading shim was added. Existing W+B/R
pretrains remain unchanged/reusable. Do not link the epoch69 GAN as the new
experiment's resume input. MAIN's checkpoint and model remain usable unchanged.
