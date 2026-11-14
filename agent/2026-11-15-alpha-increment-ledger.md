# ALPHA increment record and next experiment — 2026-11-15

Status: existing increments audited; next architecture increment is proposed,
not implemented. The user has been offered the narrow BSSP-only factor plus
utility repairs for approval. No choice is inferred from a preselected UI option.

Recordkeeping update `c70f441c4f97b6306e4dc29076bef0f30fe8416f` adds one ALPHA
AGENTS instruction to maintain `agent/ALPHA_INCREMENT_LEDGER.md`. It was normally
pushed and fresh-remote verified. `AGENTS.md` is the only changed tracked file;
the model/config/teachers and running-checkpoint compatibility are unchanged.

## Current decision

Recommend **C5-BSSP32** first: replace ALPHA's shared linear-and-split style
projection with four independent nonlinear32→32→32SiLU projections. Keep
style/text concatenation into the seed, ContentContext,32D E, GBlocks/geometry,
D/P, teachers, all losses/weights/sampling/LR/EMA/selection unchanged. This adds
4,224 parameters and answers a clean, modest conditioning question—not a claim
of first-ever per-block style vectors. Fresh GAN training is needed; W+B/R do
not need retraining.

The larger historical A4 bundle (content-only seed **plus** BSSP) is a sensible
subsequent experiment, but changes two levers. It must not silently replace the
narrow proposal if that is what the user approves. SWD-for-contextual loss is a
separate objective experiment; no extra loss is to be added to C5-BSSP32.

Latest saved ALPHA evidence is epoch69, FID6.271029, KID (×100) =0.265770,
HWD0.118055. The source-compatible checkpoint does not establish a controlled
C3→C4 gain or prove a single unchanged protocol throughout its resumed history.
MAIN-vs-ALPHA scores cannot fill the missing paired controls.

## Audited historical and current record

Date: 2026-11-15. Read-only model/source-history review. The accompanying documentation/diagram publication and ALPHA recordkeeping instruction do not alter the architecture, objectives or training. The proposed model/utility implementation still requires approval.

## Current verified source, not branch-name inference

- MAIN HEAD: `d881b59fb2111f92b0a8a580cedd4ad1e6438672`.
- ALPHA HEAD: `14f6eea61e892343ec958912529c5f25cf34491a`.
- Both tracked worktrees are clean. ALPHA retains existing untracked `pretrained/deploy_HiGAN+.pth`, `ocr_iam_new.pth`, `wid_iam_new.pth`; these are not the configured active teachers.
- Direct `git ls-tree -r HEAD` blob comparison: ALPHA 52 tracked paths, MAIN 51; 48 common, 20 identical, 28 different. ALPHA-only: three legacy default YAMLs and `metric/fid.py`. MAIN-only: three external benchmark/OCR tools. No copy/merge decision is inferred from common ancestry. A read-only alternate-object merge-base check happens to find `3fb207c`; this is not a useful definition of the current architecture delta after ALPHA's rebuild.
- Identical tracked blobs include `networks/fusion.py`, all five tracked pretrained binaries, `deploy_checkpoint.py`, HWD/Inception/MS-SSIM/PSNR modules. FID/KID/IS `val_metrics.py` differs only in comments/formatting; CMMD differs in comments/local variable names. There is no new metric formula to port.
- Current ALPHA is Classic-derived G/E/global D, native32/64, 32D global style, DEV-compatible B/W/R, overlapping StrokePatchD, original adaptive losses, and target-only ContentContext. It is not the October gain-only/spatial-memory experiment. MAIN currently also has no reference memory, but its style128/encoder/loss/normalization/recipe differs substantially.

Key live anchors: ALPHA `networks/BigGAN_networks.py:25-29,121-124,173-203`; `networks/module.py:191-242`; `networks/model.py:1170-1175,1418-1496`. MAIN `networks/BigGAN_networks.py:19-35,40-63,153-160,200-216`; `networks/module.py:266-345`; `configs/gan_iam_32.yml:33-97`.

## Three real options

| Option | Actual delta available in MAIN | Question and attribution boundary |
|---|---|---|
| **1. Content-first generator conditioning (larger bundle)** | Content-only spatial seed, with global style delivered through independent nonlinear per-GBlock projections. Reuse ALPHA's already-present ContentContext; preserve its 32D style and geometry. | Does separating initial content layout from global style improve the current ALPHA? A coherent conditioning contribution, but a two-lever bundle (seed route + nonlinear projections), not proof of either component alone. |
| 2. Nonlinear style projection only (recommended narrow next test) | MAIN `BlockSpecificStyleProjection` adapted at 32D replaces `style_linear`, while retaining ALPHA style/text seed concatenation. | Cleanest narrow architectural isolate. ALPHA already generates four distinct block vectors via one linear map and split; novelty is nonlinear two-layer per-block mapping, not the first per-block conditioning. Adds only 4,224 trainable projection parameters (4,224 → 8,448 at 32D), so do not oversell this as a large model redesign. |
| 3. SWD instead of contextual matching | MAIN `SlicedWassersteinLoss` replaces, rather than joins, ALPHA CXLoss. | Meaningful style-objective experiment without new G/E layers. Keep every other ALPHA objective/weight/schedule fixed; MAIN's 0.1 is only an initial coefficient, not an empirically matched replacement for contextual 2.0. No evidence that MAIN's bundled Nov2 results isolate SWD. |

Option 1 also follows the earlier **planned, not run** ladder: `agent/2026-10-29-dev-vs-higanplus-contributions-and-ablations.md:403-408` lists target context A3, then content-only seed + block-specific nonlinear projection A4, explicitly calling A4 a conditioning bundle. Current ALPHA completed the context idea on Nov9. This historical ladder is motivation, not permission and not a claim ALPHA matches every earlier A0–A3 control.

### Status of the historical A0–A4 ladder

| Historical ID | Original meaning | Status as of this review |
|---|---|---|
| A0 | Aligned HiGAN+ with original P and one frozen recipe | Proposed control, not an established trained result. The active C0 rebuild below already bundles modern P, teachers and native32; it is not the exact historical A0. |
| A1 | Change only canonical/native reference geometry | No standalone matched A0→A1 training evidence found in the reviewed reports. Reference preprocessing in current ALPHA must be recorded, not retroactively credited as a measured increment. |
| A2 | GatedStyleContext + normalization before global pooling | Not present in current ALPHA E; MAIN has this mechanism. No current-ALPHA A2 implementation/run claim. |
| A3 | Target-only ContentContext, retaining original style-to-seed route | Mechanism implemented in current ALPHA at `14f6eea`; validated wiring/shared initialization, no controlled trained gain. This does not imply earlier historical A0–A2 arms were executed. |
| A4 | Content-only seed + block-specific nonlinear global projection | Proposed next coherent conditioning bundle; not implemented/approved/run. BSSP-only would be a narrower factor of A4, not the whole historical A4. |

These labels are not silently reused for the retired global-only/memory models or for operational publication/logging commits. The C0–C5 table below records the actual active source chain.

### Why option 1 now

ContentContext already gives characters neighboring-target context, but ALPHA still repeats the same style code at every character and concatenates it into the initial spatial seed (`BigGAN_networks.py:180-187`). MAIN initializes that seed from contextualized content only (`:211-221`), while the same global style conditions all four synthesis stages through four learned MLPs (`:40-63,209,240-243`). This gives a concrete next mechanistic question without resurrecting local tokens, reference K/V or memory M. It does not guarantee disentanglement, stroke quality, spelling or a lower KID.

Do not copy MAIN's entire generator. MAIN32 uses a 4×4 seed and no last-stage upsampling; ALPHA32 uses a 2×2 seed with all four upsampling operations. MAIN also changes width masking and normalization (`main/networks/BigGAN_layers.py:381-408`). Those are separate behavior changes, not dependencies of the proposed conditioning port.

### Exact proposed implementation scope — not implemented

1. `alpha/networks/BigGAN_networks.py`: add MAIN's parameterized `BlockSpecificStyleProjection` class; instantiate `style_dim=32`, four blocks, `style_chunk_size=32`, ALPHA's existing spectral-linear factory. Replace `style_linear` with BSSP. Change `filter_linear` input from 152 (120 content + 32 style) to 120. Feed only existing ContentContext output into the spatial seed; feed the resulting four style vectors into the unchanged GBlocks.
2. Keep ALPHA's 2D `(B,32)` latent API; BSSP already accepts 2D. Do not copy MAIN's `(B,1,128)` sampler/encoder API, style128, E sequence model/content probe, log-variance initialization, GBlock normalization/masking, native32 stage schedule, or D readout.
3. The four IAM GAN YAMLs (`gan_iam.yml`, `gan_iam_32.yml`, `gan_iam_64.yml`, `gan_iam_local.yml`) and `gan_image.yml` may record the conditioning choice explicitly if a constructor selector is used. Keep every existing numeric dimension, loss, batch, LR, critic count, sampling and evaluation setting unchanged. No new loss/config tuning is part of the port.
4. Initialization must preserve paired-seed equality of all unaffected model parameters/buffers and the subsequent global RNG stream, as the Nov9 context increment did. Merely replacing an earlier constructor changes RNG consumption and confounds later shared-layer initialization. New/reshaped targeted projection parameters are expected to differ; record them explicitly.
5. Existing W+B/R pretrains remain exactly reusable; their definitions/configs/files do not change. Old ALPHA GAN checkpoints are intentionally incompatible because `style_linear` disappears, BSSP state is added and `filter_linear.weight` changes shape. Use **fresh matched GAN runs**, not strict=False, a hidden adaptation shim, or resumed old GAN optimizer state.
6. Dated design/implementation/verification/limits belong under `agent`; generated tests/artifacts outside model branch folders. A later approved implementation would follow normal branch publication rules unless explicitly told local-only. This review does not commit or push.

### Minimum proof and fair control

- Control: exact current `14f6eea` on the same refreshed behavior-neutral utilities. Treatment: control + option 1 only. Hold data files/split/polarity, teachers and SHA256, image height, width restoration, text/reference sampling, batch, initialization policy, optimizers/LR/adaptive loss, critic ratio, steps, evaluation seed/repeats and metric preprocessing fixed.
- Start native32 as the bounded primary experiment; native64 still needs shape/backward/loading checks. Report matched-step and wall-time/VRAM costs separately. Prefer paired seeds and repeat seeds before promotion; a single pair is preliminary.
- Both arms keep ALPHA's current **FID checkpoint selection** and report KID/HWD at identical saved steps. If changing the primary selection criterion to KID, make that a separately approved shared evaluation-protocol decision applied to both; never silently compare MAIN's KID-best against ALPHA's FID-best as if selection were identical.
- A fixed prospective lowercase/title/full-uppercase, short/long target panel should check content/style failures; independent OCR is a separately named diagnostic/evaluation output, not the training teacher score. Do not promote a case-specific sampler/objective change along with this conditioning test.
- Acceptance before training: native32/64 exact output geometry; seed invariance to style for a fixed target; all four BSSP paths receive finite gradients and affect their intended stage; no memory/reference-content input; same context/padding contract as current ALPHA; strict teacher loads; unchanged state key/shape sets outside the named projection modules; actual D+G update and save/strict reload of the new design. These establish wiring, not quality.
- A control-vs-treatment gain establishes the **bundle's** conditional effect. To distinguish its components later, run one BSSP-only or seed-only arm under the same recipe. MAIN-vs-ALPHA run metrics, September scores and retired ALPHA checkpoints are descriptive cross-run evidence, not causal ablations of this next increment.

### If option 3 is selected instead

Required scope is `networks/loss.py` (SWD class), `networks/model.py` (replacement construction/call/logs + save/load projection RNG state), and GAN YAMLs (explicit weight/projections/samples). Real B features detached; fake gradients still flow through frozen B; valid per-scale widths from the existing `StyleBackbone.feature_lengths`. Isolate projections' RNG and checkpoint seed/counter. Remove contextual contribution rather than summing SWD on top; retain reconstruction, all three CTC branches and adaptive coefficients. Do not import MAIN's zero reconstruction, fixed-loss rewrite, global-content adversary, KL rescaling, teacher modes, style128 or KID selection as SWD dependencies. ALPHA's E currently calls B without `x_lens` (`module.py:212`); a masked loss cannot be relabeled as importing MAIN's full padding-aware feature extraction. Changing that backbone-forward behavior is a separate explicit treatment. SWD loss identity/permutation/padding/gradient/RNG/reload tests are necessary; lower KID is not guaranteed.

## Safe operational differences — selective ports, not whole files

These are **candidates requiring focused equivalence checks**, not changes already performed:

| Candidate | MAIN source | Preserve in ALPHA |
|---|---|---|
| Checkpoint descriptive metadata | `networks/model.py:413-432,466,470` | Add class/parameter/module manifest, portable primitive config serialization and format version only. No EMA, distribution state, selection criterion or objective changes. Keep old metadata optional on load. |
| Richer lossless evaluation ledger | `networks/model.py:270-399,474-479` | Embedded CSV + column list, historical unknown-column retention and immediate CSV recreation after load. Keep ALPHA active metric columns/toggles, fractional epoch, estimator and FID-best policy. Preserve old CER/WER history without re-enabling automatic GAN OCR. |
| Reject empty checkpoint clearly | `networks/model.py:681-682` vs ALPHA `:432-433` | Error instead of returning epoch0 and carrying on with uninitialized weights. No tensor remapping or permissive loading. |
| Read-only RESULTS enrichment | `lib/utils.py:105-177` | UTC finish time, run/evaluation CSV paths, parameter count; retain ALPHA source_height, last-vs-best identity and epoch-axis logs. Do not transplant all best-metric logic blindly: MAIN `_results_metrics` chooses min for PSNR/MS-SSIM, which would be wrong when enabled. |
| Interruption checkpoint with accurate progress | `train.py:106-115`; `networks/model.py:101-104,400-411` | Port as a unit with ALPHA-native current/completed epoch and last completed iteration tracking; test interrupt before first batch and mid-epoch, preserve existing best checkpoint. Copying `model.save('interrupted')` alone into ALPHA would use default epoch0 and omit iter_count. |
| DDP-aware startup/logging guards | `train.py:55-69`; `networks/model.py:177-195` | Only if distributed use is actually in scope: avoid reinitializing an already initialized group; master-only side effects. No distributed sampler, batch, seeding or training-order rewrite hidden in the port. Needs a real DDP test, not a single-process claim. |

Already present in ALPHA; **do not count again**: atomic publish-before-cleanup checkpoints; missing requested checkpoint/teacher failure; strict named module loading; current deploy export; one W&B lifecycle/direct Logs; gated same-step flush; epoch-based sample axes; detailed weighted/raw losses; pinned/nonblocking transfers; workers=0-safe loader options; cached real FID/KID/HWD/CMMD features/validation loader; training RNG isolation for validation; batched scalar extraction; detached adaptive gradients without higher-order graphs; `zero_grad(set_to_none=True)`; authoritative saved Epoch handling.

Not safe neutral utilities: EMA (changes evaluation model), LR/scheduler changes including resume LR corrections (changes optimization trajectory), separate P optimizer/LR, R1/DiffAug/gradient clipping, BN/teacher mode changes, merged training forwards, AMP/TF32, new weighted/rare/case samplers, uppercase remedy, style prior centering, loss normalization, reconstruction removal, CTC coefficient changes, KID-best selection, geometry changes. Some are architecture-compatible; compatibility is not experimental neutrality. ALPHA's W&B naming/offline/configurable project behavior is in some respects more explicit than MAIN; do not overwrite it just for parity.

## Complete ALPHA increment ledger

Scope: all named ALPHA experiments since its October global-only identity plus the entire active Classic-derived increment chain. Earlier generic inherited project commits are ancestry, not claimed ALPHA-specific incremental experiments.

### Retired lineage — not cumulative components of current ALPHA

| Stage | Source/provenance | What it was; status now |
|---|---|---|
| Original global-only ablation | `232cf85de903f7818f1dbb657d8de299868c9dcc`; remote old tip rename `90225a2b1ba5380104729053814454b291ae24b0` | Remove local-style modules; global-only modern model with literal independent OCR. Rename is metadata, not architecture. Naming migration in `ALPHA_MIGRATION.md`. Retired. |
| Inherited MAIN OCR/naming history | `baad4a0` then `dbd9066` on the reset lineage before `3fb207c` | Independent CER/WER monitoring and alpha-name/document centralization. These are inherited commits in the later reset ancestry, not two additional model mechanisms in today's ALPHA. The originally named ALPHA remote had its separate `232cf85` → `90225a2` history. |
| MAIN reset / gain-only reference memory | Local ALPHA reset onto MAIN `3fb207ce4e3fd2d16df046836cec192ab78137df`; documented `2026-10-27-alpha-refactor-review.md:19-34` | Not a clean continuation of the original global-only experiment. 64D reference memory/gain-only spatial modulation inherited from MAIN. Old scores cannot be assigned to this source. Retired; no permission to restore memory. |
| Gemini simplification + corrections | Local dirty 14-file refactor, then `2026-10-27-alpha-cleanup-implementation.md`; no standalone published ALPHA SHA established | Width-mask/helper cleanup; combined frozen encoder work; tested checkpoint/functional equivalence but not bit-identical updates. Historical CPU tests belong to this memory model, not current Classic-derived ALPHA. Replaced. |
| Full metric-stream OCR | `a1831bf197f36483bd1cfb8c3c745d2abbc5e3bd` | Full configured evaluation OCR replaces 320-case sampled monitor. Metrics-only; later automatic GAN OCR removed. `2026-10-30-full-validation-ocr-fix.md`. |
| Delayed GAN OCR | `0dc4f6c6586bfcd9a65895a644507d811b55dfb9` | Start at epoch80; later superseded by automatic GAN OCR removal. `2026-10-30-ocr-start-epoch.md:3-9`. |
| Interrupted DEV-whole-model copy / competing grid proposals | `2026-10-30-dev-to-alpha-overlap-grid-plan.md`; `2026-10-30-classic-baseline-alpha-grid-design.md` | Superseded, not final approved design, not an active source increment or a quality result. Pre-replacement dirty source archive preserved under `agent/analysis/2026-10-30-alpha-grid/alpha-before-dev-copy.tar.gz`. |

### Active Classic-derived chain

| Order | Commit / dated record | Accumulated change and evidence boundary |
|---|---|---|
| C0: approved rebuild | `0c435bdc41b29119b53d64f47960716f299b5cd9` (committed Nov3; design/implementation Oct30), `2026-10-30-classic-to-alpha-stage1.md` | Classic G/E/global D + DEV StrokePatchD/SN-depthwise + per-word hinge reduction, patch0.45, deterministic overlapping H/2 grid/H/4 stride + compatible strict B/W/R teachers + explicit native32 adaptation. This is a **bundle**, not pure crop-policy evidence. Fresh GAN required; teachers reused. 7 stage1 +3 grid contracts and actual CPU32/64 update/save/reload recorded; no trained quality gain. |
| C0 operations folded into rebuild | Same `0c435bd`; `2026-10-30-alpha-runtime-efficiency.md`, `2026-10-30-alpha-operations.md` | Higher-order adaptive graph removal, gradients/scalar transfer, real-feature/loader caches, validation RNG isolation; W&B/RESULTS/status, atomic checkpoint safety, missing-path checks, detail logs and loader controls. Separate conceptual operation increments, not separate source SHAs. Matched CPU updates reported exact model+optimizer parity for runtime port; fixed validation RNG intentionally differs from old consuming-evaluation trajectory. |
| Publication repair | `03ff873364dffb63b904ddd17a1a4755025d4022`, `2026-11-03-alpha-publication-repair.md` | History-only ours merge with `90225a2` as second parent; tree exactly `0c435bd` (tree `383685323d4e53465f788a98ca0a296b52b48935`). Preserves old remote lineage without restoring old model. No model increment. |
| Workflow rule | `26936b0e4156f5e8ff3e8cd6910b1e8b13a2adfa` | AGENTS publication/verification requirement. No model increment. |
| C1 recipe/runtime settings | `c406eb37eef9a7ca1bfa3fd0aa664f4bcbfb497a`, `2026-11-04-config-publication.md` | Train/eval batch8 for32/64, sample interval500. Batch change is a training-recipe confound, not architecture or a quality proof. |
| C2 evaluation/log parity | `0532c68cc84a189dd5415f5c9ac46f716d298116`, `2026-11-08-alpha-metrics-wandb-parity.md` | Validation selector/settings aligned with MAIN, HWD enabled; fractional epoch sample panel, explicit iter/gated W&B flushing. Same formulas. Alias resolution/provenance may make pre/post metrics incomparable. No model/teacher retraining needed. |
| C3 correctness repairs | `84fbab2de435b4319a3ed50d9aa1dcb11662f748`, `2026-11-08-main-alpha-bug-audit.md` | Named module loading/current deploy export; transfer sample width; fractional-epoch CSV; writer optimizer LR placement; corpus sampler argument/device; singleton batch encoding; standalone toggles; authoritative resume Epoch. No new model/objective. Applicable9 audit tests and CPU smokes recorded; old metadata lost by earlier bugs cannot be recovered automatically. |
| C4 architecture increment — current | `14f6eea61e892343ec958912529c5f25cf34491a`, `2026-11-09-alpha-content-context.md` | Target-only relative self-attention + depthwise local gated FFN before old style/text seed; 4heads/128attention/360FFN/residual0.25, +194,532 G parameters at32. Shared initialization/RNG parity tested. Old GAN incompatible; fresh GAN, same teachers. Two context contracts +9 audit regressions + CPU32/64 updates/reload recorded. No trained FID/KID or T4 claim. |
| C5-BSSP32 proposed, not approved | This review, no implementation SHA | Nonlinear per-stage style projections only; preserve the mixed seed and all C4 recipe/teachers/geometry. Full content-first A4 is a separate larger follow-up. No trained gain is established. |

## Remaining unknowns and stop boundary

The quoted historical passing tests are recorded evidence, not rerun in this review. Nov9 inventory reports ALPHA legacy discovery 51 tests with16 failures/38 errors including subtests; a green focused suite is not green project-wide evidence. The separately documented checkpoint audit and bounded MAIN inference probe are current diagnostics, not a new training/quality benchmark, GPU fit/speed test or live W&B/DDP verification. The supplied checkpoint does not establish a controlled C0→C4 quality trajectory. Source publication was freshly verified for the recordkeeping update; no model implementation or trained gain is inferred. Approval is needed before C5 or any utility port.
