# ALPHA versus MAIN: review of Gemini's simplification

Date: 2026-10-27, Asia/Saigon (UTC+07:00).

Scope: read-only product review of the supplied Gemini output and actual local
source. No model/configuration/teacher/checkpoint was edited, committed or pushed
by this review. Disposable CPU checks and this report are stored under agent.

## Verdict

The implemented cleanup is largely good and preserves the current configured
model in the tested cases. It is a maintainability refactor, not a smaller neural
architecture or a demonstrated quality improvement. Several proposed next steps
are worthwhile; the proposed loss deletion is not justified as a no-regression
cleanup. Fix unsupported-option handling and avoid extra mask preprocessing
before claiming a cleaner implementation is also faster.

## Exact revisions and branch state

- MAIN: clean at `3fb207ce4e3fd2d16df046836cec192ab78137df`.
- Local ALPHA: also based at that commit, with 14 uncommitted modified files.
- Refactor diff: +242 / -1,418 lines, net -1,176. This matches the supplied summary.
- Remote MAIN still points at `3fb207c...`; remote ALPHA still points at
  `90225a2b1ba5380104729053814454b291ae24b0` at review time.
- ALPHA reflog records `reset: moving to FETCH_HEAD`. Its old branch history was
  reset to MAIN rather than merely receiving copied files as a new commit.
  Relative to its cached origin/alpha, Git reports ahead 112 / behind 78.

The refactor is not yet committed or pushed. Do not reflexively force-push this
state if preserving ALPHA history is intended. The old tip remains accessible;
history repair/publication is a separate authorized task, not performed here.

ALPHA is no longer the old global-only experiment. Its model now matches MAIN's
64D reference memory and gain-only spatial modulation. Old ALPHA checkpoint scores
cannot establish the quality of this new source tree.

## What actually differs in code

| Area | Change | Assessment |
|---|---|---|
| block.py | Remove unused layers, recurrent helpers and normalization variants | Good for the active configured pipeline; deliberately reduces old helper APIs |
| utils.py | Remove old unused samplers, augmenters and wrappers | Good; active StrokePatch sampling and augmentation stay intact |
| loss.py | Remove unused Gram/contrastive implementations and duplicate mask helper | Good; no active GAN loss removed |
| module.py | One CNN construction helper for B and R | Good; fresh modules are instantiated separately, not weight-shared between B and R |
| BigGAN_layers.py | Shared G/D width-mask helper; remove unavailable SyncBN references | Mostly good, with option/performance caveats below |
| BigGAN_networks.py | Remove unused embedding/attention-report/33px architecture paths | No active 32px/64px architecture change |
| model.py | Extract scheduler restore, fake-triplet generation, epoch weights and W&B payload | Good boundaries; moves bookkeeping rather than neural work |
| train.py / test.py | Remove duplicated rank logic; loop over metric switches | Equivalent in the examined entry paths |
| YAML | Remove unused CTC-decoder and discriminator options | No remaining YAML value changed |

All active dimensions, objectives and optimizer settings remain MAIN's: global32,
memory64, four-row/max32-column memory grid, attention128/four heads, gain-only
retrieval, batch8 in normal configs, critic2, LR2e-4, writer0.25, late transfer
CTC0.3, and the existing EMA/epoch/evaluation settings. Fusion.py is unchanged.

The removed Gram and supervised-contrastive code was not in the active objective.
Deleting it cannot by itself improve KID or reduce executed training work.

Parameter counts remain identical across MAIN and ALPHA:

| Height | G | E | B | R | W | D | P |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | 3,950,685 | 953,521 | 1,617,328 | 2,428,673 | 317,780 | 2,366,145 | 794,945 |
| 64 | 3,954,845 | 953,521 | 1,617,328 | 2,428,673 | 317,780 | 2,366,145 | 794,945 |

## Findings worth addressing

### 1. Unsupported cross_replica silently changes meaning

`alpha/networks/BigGAN_layers.py:349` and `:411` still accept `cross_replica`.
The flag is stored but no longer influences either normalization implementation.
The old branch would fail because SyncBN2d was undefined; the new one silently
constructs ordinary local normalization instead. This is not equivalent failure
behavior or a working synchronized-BN implementation.

Reproduction: `bn(4, cross_replica=True)` has the flag set, ordinary running stats
allocated and no sync module. Current configs use false, so this does not explain
a current default-config quality regression. Reject true explicitly, or implement
proper support only if it is genuinely needed. Do not advertise the option while
silently ignoring it.

### 2. Shared masking adds repeated preprocessing in D

`alpha/networks/BigGAN_layers.py:147` casts/clamps lengths on every mask call.
DBlock invokes it four times (`:544`). The old D helper compared existing lengths
directly. For one configured 32px global-D forward, CPU operator tracing shows:

| Operator | MAIN | ALPHA |
|---|---:|---:|
| aten::clamp | 1 | 17 |
| aten::to | 8 | 40 |
| aten::_to_copy | 4 | 4 |
| aten::masked_fill | 16 | 16 |

`to` counts do not imply a copy every time; actual `_to_copy` counts are unchanged.
However, 16 additional clamp operations are real work, and on CUDA may add small
kernel-launch overhead. No T4 slowdown was measured, so do not attach a percentage.
Keep the masks, but normalize lengths at an appropriate entry/stage boundary and
reuse valid lengths/masks instead of reprocessing them for every feature tensor.
The CPU profiler emitted a device-enumeration warning with CUDA hidden; the trace
completed successfully and no CUDA workload was run.

### 3. Some explanation/API cleanup is incomplete

`alpha/networks/module.py:184` still says multi-scale local-token attention remains
in StyleEncoder; the current encoder uses projected spatial memory, not learned
local-slot attention. Update stale explanations. Removing Recognizer.frozen_bn
is safe for the tracked current callers, which use the shared free function, but
the claim of preserving every historical public helper API would be too broad.

No blocking defect was found in the tested default FP32 32px/64px pipeline.

## Assessment of Gemini's proposed next changes

### Batch generated random/style images through B+E: first candidate

The two generated paths have matching canvas widths/target lengths. B is frozen
and in eval mode; the current E path has per-position/per-image normalization,
not training BatchNorm or dropout. A fresh CPU probe comparing the separate calls
with one concatenated B+E call, E in train mode and B eval, gives maximum forward
differences of 9.69e-8 (32px) and 1.19e-7 (64px), including returned feature maps.

This is promising implementation simplification, not a proven CUDA speed result.
It processes 2B images in one call instead of B twice: it does not halve CNN FLOPs
or eliminate one batch's arithmetic/backward graph. Test gradient equivalence,
full update parity and peak VRAM/throughput at the actual Kaggle batch size before
adoption. A larger single call may require different convolution workspaces.

### Remove center_style_prior: reasonable, with interface checks

It is a no-op for the sole current global token. Removing its internal calls is
low risk if all current entry points and checkpoint/sample/metric helpers are
verified. Changing every style from [B,1,D] to [B,D] is a much larger API cleanup
with almost no tensor-compute benefit; the singleton dimension is not an extra
learned token or large activation. Do not combine that rewrite casually with a
speed optimization.

### Fold the fixed normalization divisor into weights: algebraically valid

For current single-token runs, removing the fixed division by eight and changing
the effective four coefficients to 0.125, 0.125, 0.0025 and 0.0125 can preserve the
objective. Verify division is removed exactly once, config defaults and old resume
metadata are handled consistently, and logs are still the same weighted values.
It is clarity work, not an expected quality gain.

### Drop style-cycle or content-adversarial loss: not justified as cleanup

The terms constrain different things:

- Writer loss: correct writer classification on the transfer image.
- Contextual loss: reference/generated feature-set similarity across different text.
- Style cycle: recover the reference global code from the transferred image.
- Reconstruction: pixel L1 on **recn_imgs**, not on **style_imgs**.
- Content-adversarial: make the reference global code less predictive of the
  reference transcription through the existing gradient-reversal probe.
- KL: regularize the global Gaussian posterior toward its prior.

The implementation is explicit at model.py:2473, :2478, :2484 and :2518. KL and
content-adversarial are not interchangeable, and sharing a backbone does not make
writer/contextual/style-cycle objectives identical. A spatial memory does not
remove the need to learn a usable global path. Any deletion is a training ablation
that may alter writer fidelity, readability or content leakage, not a proven
no-quality-change refactor. Retain them for the cleanup comparison.

### Alias raw/style dataset tensors: useful but narrower than claimed

When process_style=False, the current dataset transforms the same source pixels
twice and collates separate equal canvases. There is genuine avoidable work.
The implementation does not literally already alias the two tensors. A safe
optimization must account for subsequent sorted collation, pinning and consumers
that may mutate tensors; these can recreate copies or make aliasing observable.
There are also augmented images and other batch fields, so this is not proof of
halving total DataLoader traffic or training time.

### Reduce evaluator cadence: optional visibility tradeoff

This changes monitoring frequency, not the GAN objective. Existing code already
caches real FID/HWD representations and generated batches within evaluation;
it is not recomputing all real features every epoch. OCR cadence zero means the
existing initial/session evaluation policy, not unconditional disabling.

Given the user's focus on KID/HWD and short-reference readability, keep FID/KID/HWD
visible and measure evaluation costs first. Running independent OCR less often
can be reasonable if it is expensive, but use matched cadence between experiments
and retain final independent CER/WER. Best-FID selection alone does not make HWD
or readability monitoring redundant.

### Move interactive demos out of model.py: good maintenance follow-up

Keep eval_demo.py behavior while moving its input()/plotting responsibilities out
of the training class. This can reduce navigation cost, not active neural FLOPs.
Do not split the remaining training loop into dozens of tiny one-line wrappers.

## Independent verification performed

All execution used CPU FP32 with CUDA hidden and two Torch threads. No full
FID/KID/HWD/CER/WER evaluation, GPU benchmark or long-run training was performed.

1. The three reported current regression modules independently pass 27/27 tests.
2. Seven model constructors at each height have identical state keys, shapes,
   initial tensor values and parameter counts. Strict state loads work both ways.
3. Finite test forward outputs and gradients match exactly (max error 0) between
   MAIN and ALPHA at both heights, in configured eval and GAN-training modes.
4. A real native-data GAN loop per branch/height runs two critic batches and one
   G/EMA update, including lazy R1, real frozen teachers and actual sample writing.
5. After that loop, SHA-256 fingerprints match for all seven raw model states,
   both EMA states, all three Adam states, all three scheduler states and EMA step.
6. No remaining YAML value differs. Only the listed unused options were removed.
7. MAIN stays clean; ALPHA retains exactly the 14 pre-existing refactor modifications.

These checks support this refactor's behavior preservation within tested inputs,
not bit-identical CUDA behavior under every configuration, unchanged NaN handling,
all removed historical APIs, or any metric improvement. Historical ignored suites
targeting retired slot architectures were not used to manufacture a green claim.

Local ignored evidence/helpers: agent/tests/alpha_refactor_differential_20261027.log,
alpha_refactor_real_step_20261027.log, main_refactor_real_step_20261027.log and
review_alpha_refactor_20261027.py / review_refactor_real_step_20261027.py.

## Retraining and recommendation

This cleanup alone does not require retraining GAN or teachers. A compatible
current MAIN gain-only 64D-memory checkpoint should retain its weights and behavior
under ALPHA; strict tensor compatibility is verified. Old global-only ALPHA or
128D/additive states are a different architecture and cannot be assumed compatible.

Keep the implemented cleanup, address the option/mask details, then prioritize
batching the generated-image encoder work and removing truly unused helpers.
Treat loss deletion, fewer style dimensions or different evaluation protocols as
separate experiments. No code fix, commit, push or history repair was authorized
or performed in this review.
