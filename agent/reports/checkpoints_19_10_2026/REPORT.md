# MAIN versus ALPHA: checkpoint audit, 19 October 2026

> **Update after the initial audit:** the user subsequently supplied `best_fid_2.5769_main.pth`. It has been verified as MAIN's genuine 5-token checkpoint at epoch 77: FID 2.5769258222, KID 0.0719623566, HWD 0.1054325774, 77 history rows, finite G/E and EMA tensors. Its full CSV is now exported alongside the others. References below to the missing MAIN best describe the initial four-file submission, not the updated folder. The duplicate `best_fid_2.7509_main.pth` remains a ALPHA copy. The rank/norm probes below were run on MAIN **last**, not the newly supplied MAIN best.

## Executive assessment

The runs are not failing to learn. Both now operate around FID 2.6–3.1 on their saved 32px evaluation protocol. The useful questions are representation quality, the FID/KID–HWD trade-off, late-training variance, and experimental attribution—not a wholesale architecture replacement.

The largest confirmed issues are:

1. **The file labeled MAIN best is actually a byte-identical copy of ALPHA best.** MAIN's actual best weights are absent from this supplied folder.
2. **The local style representation is strongly low-rank.** More learned slots do not translate into proportionally richer local style evidence.
3. **Reference-encoded local codes and Gaussian local-prior codes have substantially different scales and structures.** One shared G is trained against both.
4. **The default metrics cover same-reference-text generation, not the difficult short-reference-to-unseen-word transfer task.** Aggregate scores cannot settle punctuation-reference robustness.
5. **The two histories suggest different strengths, but are not repeated-seed, fully provenance-controlled ablations.** They do not establish a universal winning token count.

No training source, config, checkpoint, dataset, or Git branch was modified for this analysis. All new outputs are in this report directory, outside the branch checkouts. No GPU evaluation or training was run.

## 1. Scope and verification

Inspected all four supplied PTH files, not just filenames. Extracted their complete embedded evaluation CSVs; compared them against structured `eval_history`; checked state hashes, saved configs, model manifests, finite values, Adam state, scheduler state, iteration counts, and EMA steps.

Current inspected source heads:

- MAIN: `3a8b00f31e3759a230ba1d789581252592a7204a`.
- ALPHA checkout (`alpha`): `7eed74c6b2d5286dc7b3e446ec1db3b165322c39`.

Inference used saved configurations, strict-loaded EMA G/E and frozen B, CPU FP32, original 32px references, fixed test selections, and the normal dataset polarity convention. Eleven selected references provide short/long same-writer examples; an independent fixed random sample of 128 references provides the broader token-rank probe. These are diagnostics, not a fresh FID/KID/HWD benchmark or a statistically representative visual-quality score.

## 2. Checkpoint identity and missing MAIN best

| Supplied file | Actual architecture | Epoch | FID | KID, as logged | HWD |
|---|---|---:|---:|---:|---:|
| best_fid_2.7509_main.pth | 1 global + 7 local | 84 | 2.750891 | 0.073509 | 0.105745 |
| best_fid_2.7509_alpha.pth | 1 global + 7 local | 84 | 2.750891 | 0.073509 | 0.105745 |
| last_fid_2.8067_alpha.pth | 1 global + 7 local | 86 | 2.806748 | 0.079064 | 0.093287 |
| last_fid_2.9572_main.pth | 1 global + 4 local | 80 | 2.957154 | 0.097320 | 0.113494 |

The two `best_fid_2.7509_*` files have the same SHA-256:

`d068f0fc3e0144b9ab61a682257df5b477dc7610bc6efb9d0094363ce9af0587`

Their entire files, G/E/D/P states, frozen teachers, EMA states, configs, and 84 history rows are identical. This is not merely rounding coincidence. Renaming a checkpoint does not change its token count or provenance.

MAIN last embeds `best_fid = 2.5769258221824316`, agreeing with its CSV at epoch 77. That row has KID 0.07196236 and HWD 0.10543258. No matching `best_fid_2.5769*` file was found among the searched checkpoint and branch-pretrained paths. The CSV proves that the evaluation was recorded; it does **not** recover epoch-77 weights from epoch-80 weights.

The notebook does contain best-checkpoint carry-forward logic. It is a plausible mechanism for inherited best artifacts being collected under a later experiment, but the supplied files alone do not prove the exact copying/renaming event. This is an artifact-provenance problem, not evidence that MAIN independently reproduced FID 2.7509.

Do not load the mislabeled 8-token best as MAIN's 5-token E: `style_queries` has shape `[1,7,256]` rather than `[1,4,256]`.

## 3. What the experiment actually differs in

The active network source files are identical between branches except `networks/model.py`, where MAIN adds an explicit normalization divisor for global-only objectives. Both share the same G architecture, fusion code, StyleBackbone, StyleEncoder implementation, global D, StrokePatchD, OCR and writer implementations, dataset pipeline, and metric code.

Saved last-checkpoint configs differ in:

| Field | MAIN | ALPHA |
|---|---|---|
| Local queries | 4 | 7 |
| Total style tokens | 5 | 8 |
| Explicit global-loss normalization | 8 | default total-token divisor = 8 |
| Configured final epoch | 80 | 100 |
| Recorded resume path | `last_fid_3.1002.pth` | `last_fid_3.0487.pth` |

Both use 32px, batch 8, critic ratio 2, G/D/P initial LR 2e-4, Adam betas 0.5/0.999, EMA beta 0.9998, the same loss schedules, and identical frozen B/W/R tensor hashes. Their metric implementations and saved metric settings match.

The global-loss compensation is mathematically correct: MAIN's function divides by 5 and its multiplier is 5/8, yielding a net divisor of 8. KL divides directly by 8. ALPHA divides by 8 normally. Thus MAIN has **not** accidentally strengthened its global info/cycle/content/KL terms by 8/5.

G has 3,942,493 learnable parameters in both. E has 1,332,145 in MAIN and 1,332,913 in ALPHA: only **768 parameters** differ. Reducing local queries reduces attention work in the local retrieval paths, but is not a meaningful whole-model parameter compression. No T4 speed or VRAM advantage was measured here.

The resume paths are different, and history contains no per-epoch code/config fingerprints or complete resume-event manifest. A final saved config cannot certify the exact configuration used at every historical row. The runs are useful single-run observations, not a fully isolated repeated-seed causal study.

## 4. Read the histories, not only best filenames

### Independently lowest metrics recorded in each history

| Metric | MAIN minimum | Epoch | ALPHA minimum | Epoch |
|---|---:|---:|---:|---:|
| FID | 2.576926 | 77 | 2.750891 | 84 |
| KID | 0.061819 | 65 | 0.069984 | 73 |
| HWD | 0.099161 | 79 | 0.090255 | 62 |

These minima are from different checkpoints. Do not combine them into one imaginary model. At each FID-selected checkpoint, the complete rows are MAIN epoch 77: `(2.576926, 0.071962, 0.105433)` and ALPHA epoch 84: `(2.750891, 0.073509, 0.105745)`.

MAIN's recorded minimum FID is about 6.3% lower; its independent minimum KID is about 11.7% lower. ALPHA's independent minimum HWD is about 9.0% lower. Minima alone exaggerate certainty because they select favorable epochs from noisy training.

### Matched epoch-window means

| Epochs | MAIN FID / KID / HWD | ALPHA FID / KID / HWD |
|---|---|---|
| 41–60 | 3.31746 / 0.104514 / 0.129338 | 3.36170 / 0.108329 / 0.116086 |
| 61–70 | 2.89294 / 0.085349 / 0.117029 | 3.07380 / 0.096936 / 0.110766 |
| 71–80 | 2.98749 / 0.105422 / 0.113817 | 3.04444 / 0.095379 / 0.107232 |
| 61–80 | 2.94021 / 0.095385 / 0.115423 | 3.05912 / 0.096158 / 0.108999 |

Interpretation:

- Four local slots are sufficient for competitive aggregate realism. Seven slots are not automatically superior.
- MAIN has a modest matched-window FID advantage, but the 61–80 average KID difference is only about 0.00077 in logged units. That is much less compelling than comparing their best-ever KID values.
- ALPHA's HWD advantage persists across all these windows. This is evidence of a style/handwriting-feature trade-off worth preserving, not grounds for deleting local conditioning.
- MAIN's KID mean worsens about 23.5% between 61–70 and 71–80, while HWD improves about 2.7%. There is no single scalar notion of progress here.
- ALPHA epochs 81–86 average FID 2.99628, KID 0.098287, HWD 0.104950. It is still fluctuating rather than collapsing.
- Late FID/KID are highly correlated within both histories, approximately 0.970 for epochs 61 onward. This is descriptive correlation, not proof of any background-related cause.

## 5. Confirmed representation bottleneck: slots are mostly collinear

For each reference, take the centered local-code matrix `L` with shape `number_of_local_slots × 32`. Compute singular values `s_i`, energy fractions `p_i = s_i² / sum(s_j²)`, and effective rank `exp(-sum(p_i log p_i))`.

| Probe, 128 fixed random test references, last EMA | MAIN | ALPHA |
|---|---:|---:|
| Mean per-reference local effective rank | 1.0398 | 1.1413 |
| Median per-reference effective rank | 1.0247 | 1.1340 |
| Mean energy in leading direction | 99.30% | 97.49% |
| Mean effective rank before final local projection | 1.0751 | 1.3373 |
| Effective rank of pooled local codes across references | 1.7262 | 1.3077 |
| Mean absolute cosine between reference principal directions | 0.8688 | 0.9753 |
| Effective rank of final projection weights themselves | 17.56 | 19.08 |

This is stronger than merely observing a weak final linear layer. The 256-dimensional attended evidence is already low-rank across slots; the final projection's weights are not themselves rank one. Increasing the final projection width does not directly address this mechanism.

The subtraction `local_mu -= local_mu.mean(dim=1, keepdim=True)` guarantees rank at most `L-1`: 3 for MAIN, 6 for ALPHA. However, this mathematical ceiling alone does **not** force rank near one. Learned queries, attention selection, the value path, and downstream incentives jointly settle on a much smaller effective subspace.

Orthogonal initialization does not enforce persistent query/evidence diversity after training. More slots can become several positive/negative versions of nearly the same visual factor. This does not mean all style information is gone: global codes remain available and the direction can change by reference. It means the advertised local capacity is substantially underused on this probe.

Zeroing local codes changes diagnostic generated images by mean absolute differences of 24.66/255 in MAIN and 24.05/255 in ALPHA last. This is an intervention, but an out-of-distribution one: it proves local conditioning is active, **not** that removing it improves images or metrics.

### Most justified architecture experiment

Do not add more slots or another attention stage first. Test a simpler local evidence extractor against the existing one:

- Retain frozen B and the existing global summary.
- Build four generic spatial descriptors using normalized spatial pooling/grid or stripes over the same multiscale features.
- Project/fuse those descriptors into the existing local-code interface; keep the current spatial attention in G.
- Preserve the existing objectives, dataset, teachers, and metric protocol.

This replaces learned-query retrieval with a directly structured evidence source rather than adding a diversity loss. It has no character IDs, punctuation checks, writer-specific rules, additional data, or resolution-specific branch. Pooling choices are hypotheses: they can discard useful non-aligned evidence, so only a controlled comparison can show whether this beats the learned retrieval.

For a narrower diagnostic, remove only the across-slot centering in a separate retrained ablation while retaining per-feature-map value centering. This tests one source of coupling without changing the entire fusion. It may reintroduce overlap between local/global codes and is not guaranteed to resolve collapse.

## 6. Prior/posterior mismatch and objective allocation

The local posterior is deterministic and unregularized by KL; random local prior tokens are independent Gaussians followed by across-slot centering. Those are not the same distribution.

| Statistic | MAIN | ALPHA |
|---|---:|---:|
| Mean encoded local-token norm, 128-reference probe | 15.46 | 31.54 |
| Gaussian-prior RMS norm after centering | 4.90 | 5.24 |
| Ratio | 3.16× | 6.02× |
| Mean global posterior sigma | 0.0920 | 0.0993 |
| Mean across-reference global coordinate std | 1.1986 | 1.2500 |

The prior RMS expression is `sqrt(32*(1-1/L))`. Changing L from 7 to 4 also changes per-component prior variance from 6/7 to 3/4. Thus the token experiment changes the centered-prior geometry/scale in addition to slot capacity; the explicit global-loss divisor does not compensate for that.

The encoded codes are both much larger and much lower-rank than the random local prior. G processes both because training batches combine random, style-transfer, and reconstruction images. This can make the two conditioning regimes demand different responses from one shared generator. The global posterior's small conditional noise is not by itself pathological: its aggregate mean varies substantially by reference.

The mean learned local residual scale in G is about 0.0806 in MAIN and 0.0546 in ALPHA last, versus initialization 0.25. Some channels are negative; there is no hard saturation cap. A smaller scale can compensate for larger local values. Scale alone is **not** a calibrated measure of conditioning importance; the zero-local experiment demonstrates why.

Recommended order:

1. Measure random-versus-encoded local moments and rank over training references as well as test references.
2. Run matched random-generation and reference-transfer diagnostics, since standard saved metrics predominantly assess the latter.
3. Test a learned, scale-controlled local interface with a matched prior, preserving writer/style variation. Do not hard-clamp all style amplitudes or indiscriminately add local KL; those can erase useful information.
4. If unconditional random style is not a research requirement, consider a separate simplification experiment that draws random styles from training-reference encodings instead. This changes the method's unconditional-generation capability and must be disclosed; it is not a drop-in equivalent replacement.

Global info/cycle/content objectives explicitly re-encode with `global_only=True`; there is no local-token reconstruction/cycle term. Local evidence is supervised indirectly through generated images, not directly taught to be diverse or invariant to reference content. This asymmetry helps explain why the local branch can be active yet low-rank. It does not prove another auxiliary loss is necessary.

## 7. LR, updates, EMA and numerical health

All saved model tensors and Adam moments are finite. Frozen B/R BatchNorm variances are positive. The active EMA Gs also have positive `stored_var` buffers, with minima around 0.00736–0.00794. Strict inference loads succeeded for MAIN last and ALPHA best/last.

At MAIN epoch 80, D/P have each taken 666,000 optimizer steps and G/E 333,000. At ALPHA epoch 86, D/P have 715,950 and G/E 357,975. The EMA step counts agree with G updates. These are exactly consistent with 8,325 minibatches per epoch and critic ratio 2. There is no evidence of the critic-ratio/optimizer-step bug seen in older experiments.

The scheduler is:

`lr(epoch) = 2e-4 * max(0.15, 1 - max(0,(epoch-36)/48))`.

- Epochs 1–36: 2e-4.
- Epoch 62: approximately 9.17e-5.
- Epoch 70: approximately 5.83e-5.
- Epoch 76: approximately 3.33e-5.
- Epoch 77 onward: floor 3e-5.

The floor starts around epoch 77, **not** epoch 84, because `min_lr_ratio=0.15` clips the declining line before it reaches zero. All current optimizer and scheduler states agree on the 3e-5 floor.

MAIN's best FID occurs at epoch 77; ALPHA's at 84. These histories do not support the blanket claim that decay destroys learning or that restoring all LRs to 2e-4 is the answer. CTC schedules finish by epochs 20/28 and reconstruction by 42, so they are not newly changing during the 61–86 tail.

EMA beta 0.9998 has a late-training half-life of about 3,465 G updates, roughly 0.83 epoch here. It is not averaging ten epochs. Its parameters are smoothed but BN and spectral-normalization buffers are copied from the live model. This is a plausible source of evaluation variation when buffers respond to recent batches, not a demonstrated bug: separate buffer calibration and parameter smoothing before changing EMA behavior.

Suggested optimizer experiment, only after preserving a control: continue from the same checkpoint with G LR 3e-5 and D/P LR 1.5e-5 versus unchanged 3e-5 for all. This tests a weaker late critic without changing update counts. Values are trial candidates, not an established optimum. Avoid simultaneously changing critic ratio, all LRs, crop weights, and objectives.

The checkpoints do not contain discriminator margins, per-objective gradient norms, clipping frequency, or wall-clock profiler traces. Therefore they cannot establish whether D is overpowering G, contextual loss dominates gradients, or a particular module is the T4 runtime bottleneck.

## 8. Evaluation coverage and visual findings

Current style-guided validation with `use_rand_corpus=False` encodes the reference word and generates its **same label**. It can score well without being robust at generating a different, complex target from a dot, slash, or isolated digit. This is an evaluation coverage limitation, not an accusation that existing metrics are invalid.

The fixed visual sheet includes punctuation/digit/word references, with several same-writer short/long pairs. It shows readable cross-text generations rather than wholesale early-epoch collapse. Short references still produce different weight/slant/spacing from fuller references. A dot or digit genuinely cannot identify every allograph or pen trajectory of a writer; the objective should be calibrated uncertainty and plausible transfer, not an impossible guarantee of recovering hidden style.

No slash reference was present in the chosen exact-token lookup; the probe explicitly skips unavailable reference tokens. The generated target list also includes `abdg/17`; it is a diagnostic target, not a claim about a complete rare-character benchmark.

Padding/batch checks over the selected references:

- Adding 64 invalid background columns changes style codes by at most about 1.1e-4.
- Changing target batch companions changes generated values by at most about 1.1e-5.
- Permuting local slots together changes output by at most about 2.8e-6, as expected for attention over an unordered key/value set.

These checks do not reproduce a large inference padding-leakage issue. They do not validate every GPU backward path or every input size.

FID/KID preprocessing converts to the chosen normalized polarity, replaces invalid tail padding, pads/crops width to 4H (128px at H32), then resizes to 299×299. HWD uses valid-width word images. Thus these metrics emphasize different feature properties and geometry. Do not change this established benchmark preprocessing merely to improve the score; add separate diagnostic evaluations instead. Paper/background structure inside valid image regions still reaches the feature extractors, but this audit cannot quantify how much of a particular score comes from background rather than ink.

OCR/WER/writer metrics are disabled in these saved evaluations. Consequently, FID/KID/HWD alone cannot establish transcription accuracy, rare-character correctness, or allograph diversity. Use a separately disclosed diagnostic CER/WER and matched reference-transfer panel, retaining the official benchmark protocol unchanged.

## 9. What is established, what is not

Established for these artifacts:

- Both 32px runs have learned useful handwriting and substantially outperform their own early histories.
- Four local slots are viable; more slots are not automatically better on FID/KID.
- Seven slots retain an observed HWD advantage across matched late windows.
- Both local extractors underuse their nominal slot dimension; the encoded local distribution differs markedly from its random prior.
- MAIN's supplied best file is not its model; recovering the actual epoch-77 best is the first practical action.

Not established:

- Statistical superiority of either architecture across seeds.
- A causal link from token count alone to the observed differences.
- That rank expansion necessarily improves handwriting metrics.
- That late critic dominance, contextual loss, or background artifacts are the definitive cause of the plateau.
- That 32px cannot match 64px: this folder contains no 64px run.
- That these numbers beat another paper's scores without matching its data, split, image preprocessing, feature model and KID scale. Logged KID here is raw polynomial MMD multiplied by 100.

## 10. Prioritized next steps

1. **Recover and preserve MAIN epoch-77 best**, without overwriting any existing checkpoint. Validate identity from saved config and query shape, not filename. If unavailable, report MAIN's minimum as a historical score, not a recoverable model.
2. **Keep the existing models as controls.** Do not replace the whole G or D: evidence points first to local evidence extraction/distribution, not insufficient total G capacity.
3. **Compare a simple four-descriptor local extractor against learned-query extraction.** Same B/W/R, G, D/P, losses, dataset and update budget. Aim for stable useful diversity, not rank inflation as a goal in itself.
4. **Probe conditioning-distribution alignment separately.** Do not confound it with step 3. Make any scale normalization/prior change explicit and measure both random-style and reference-style behavior.
5. **Add an evaluation-only transfer stress set**, using existing test writers/references and fixed target words. Include short/full references, actual corpus rare characters, and image examples. No character-triggered model fallback, no new training data, no test-set fitting.
6. **Run the low-LR critic ablation if resources allow**, preserving the unchanged continuation. Compare windows and multiple evaluation seeds, not the single most favorable epoch. Fix the official score seed for reporting; use additional seeds only to characterize uncertainty.
7. **Profile before speed edits.** Frozen B and R still need input gradients on generated images. Contextual matching, repeated fake branches, and critics may dominate; removing useful supervision based on forward parameter counts is unjustified.

For a defensible token-capacity contribution, use repeated seeds and either clean scratch runs with identical frozen teachers and schedules or an explicitly documented architecture-conversion protocol. Different token shapes cannot be strictly loaded from the same E without choosing how slots are transformed. Do not pretend those choices are neutral.

## Artifacts

- `audit.py`, `audit.json`: reproducible CPU extraction, hashes, model/optimizer/scheduler checks and complete histories.
- Four `*.csv` exports: full embedded epoch tables, including disabled metric columns.
- `probe.py`, `probe.csv`, `probe_summary.json`: fixed short/long-reference sampling and invariance/ablation diagnostics.
- `token_probe.py`, `token_probe.json`: 128-reference representation measurements.
- `matched_references.png`: native 32px outputs enlarged 2×, without display autocontrast.

Useful source anchors in the inspected MAIN tree:

- `networks/module.py:407`: learned local queries.
- `networks/module.py:548`: local evidence retrieval and centering.
- `networks/fusion.py:65`: spatial local-style attention.
- `networks/BigGAN_networks.py:208`: global conditioning and G stage flow.
- `networks/model.py:1789`: global-loss divisor compensation.
- `networks/model.py:2358`: global-only re-encoding objectives.
- `networks/model.py:1071`: default evaluation generator.
- `networks/model.py:31`: EMA implementation.
- `networks/utils.py:121`: LR schedule.
- `networks/utils.py:244`: centered local Gaussian prior.
- `networks/utils.py:445`: half-height StrokePatchD crops.
- `metric/val_metrics.py:145`: FID/KID image preprocessing.
- `metric/val_metrics.py:468`: logged KID ×100 convention.
