# What changed after September 23 before FID first fell below 3?

Investigation date: 2026-11-09. This corrects the earlier answer that focused
too heavily on the later spatial-memory experiment. The first supplied
sub-3 checkpoint predates that architecture.

## 1. Checkpoint evidence, not just filenames

All evaluation rows below were read from checkpoint `eval_history`. These
are recorded scores, not newly recomputed full-set metrics. Historical
`alpha` filenames refer to the older branch then named random_crop; they
do not mean its architecture equals today's Classic-based ALPHA experiment.

| Archive / model | Saved epoch | Saved FID | First recorded FID < 3 |
| --- | ---: | ---: | --- |
| Sep 23 / DEV best | 74 | 3.489184 | None |
| Sep 23 / older ALPHA last | 84 | 3.456305 | None; embedded best was 3.228380 at epoch 69 |
| Sep 30 / older ALPHA best | 73 | **2.689235** | Epoch **62**, FID **2.990634** |
| Sep 30 / DEV last | 84 | 3.005266 | None |
| Oct 7 / MAIN best | 74 | 2.546381 | Epoch 57, FID 2.999139 |
| Oct 14 / older ALPHA best | 97 | 2.4463 | See extracted history |
| Oct 28 / memory DEV best | 71 | 1.368739 | Epoch 19, FID 2.979966 |

The Sep 30 winning row also records KID 0.0673208 and HWD 0.0939449.
The Sep 23 older-ALPHA best-history row records KID 0.1057196 and HWD
0.1126602. These are different trajectories/resume points, not paired seeds.

## 2. Strongest source/config alignment

Comparing the Sep 30 sub-3 checkpoint's saved config with historical ALPHA
YAML shows that commit **611410e65f**, September 26, matches the inspected
training/G/D/E/validation sections **except its user-supplied resume path**.

The checkpoint resumes from `./pretrained/best_fid_3.0212.pth`. Its snapshot
still has 84 epochs and style-transfer CTC final weight 0.08. The later Sep 30
commits set 100 epochs and then 0.20 respectively, so those later settings
are not present in this saved run. A config match is strong provenance
evidence, not proof of exact producing source: the checkpoint contains no
verified training commit SHA, and non-parameter forward changes cannot be
dated from weights alone.

The before/after frozen StyleBackbone B, Recognizer R and WriterIdentifier W
states are byte-identical across both Sep 23 files and the Sep 30 sub-3 file.
No teacher-pretraining improvement explains this crossing.

## 3. The actual September 24–26 changes

### September 24: ca9a597204 (DEV equivalent: 1bbc0822e0)

[ALPHA commit](https://github.com/rquq/HGGAN_test/commit/ca9a597204ff845aa1be6a88cbd787514d3a7242)

- Removed artificial stripe/patch erasure before the local discriminator.
  This preserves actual ink boundaries and paper patterns rather than asking
  PatchD to judge deliberately damaged crops.
- Made real/fake PatchD preprocessing symmetric: use unwarped real crops,
  rather than mixing an extra geometrically augmented real-patch population
  against the fake-crop path.
- Stopped PatchD downsampling once either feature axis would fall below four
  cells. Small H/2 crops no longer collapse to an unnecessarily tiny grid.
- Reintroduced bounded, valid-width contextual feature matching for transfer
  after a temporary FFT-only experiment. The archive already has
  `lambda_ctx: 0.1`, so **adding a contextual-loss scalar is not a new
  contribution relative to the Sep 23 saved run**. The bounded/cropped
  implementation and training path are the relevant changes.
- Changed 32px batch/eval batch 16→8 and removed the late writer-loss decay
  (0.5→0.25), retaining writer supervision at 0.5.

### September 26: 611410e65f (DEV equivalent: 10c66e580a)

[ALPHA commit](https://github.com/rquq/HGGAN_test/commit/611410e65bef5cfacc66d2579611be458362e068)

- **Padding-aware B/E feature extraction:** propagate true convolution and
  pooling widths, mask intermediate padding rather than only the final
  sequence, and use those lengths for multiscale reference features and
  style-context aggregation. A final mask cannot undo convolutions that
  already read invalid batch-tail activations.
- **Padding-aware D blocks:** prevent invalid right-tail activations leaking
  back into valid positions across convolutions/downsampling.
- **No synthetic constant-colour seams in D augmentation:** horizontal scale
  and translation now extend each word's boundary pixels inside its valid
  canvas (`replicate` padding), rather than inserting constant paper bands.
  Invalid batch tails keep the canonical fill. This is directly relevant to
  the reported non-rectangular/fading background artifacts, and is not an
  evaluation-image postprocessing trick.
- **Resolution/length-consistent D readout:** replace height-dependent sums
  and per-character normalization with valid spatial means and one shared
  gain of 32 at every supported resolution.
- Raised D learning rate 1e-4→2e-4 and PatchD 1.2e-4→2e-4; G stayed 2e-4.
- Deleted the texture-refinement implementation and old image-masking module.
  The Sep 23 DEV control already disabled texture refinement, while the
  older-ALPHA control enabled it. Its removal is not isolated by this run.
- ALPHA retained `num_critic_train: 2`; DEV changed to 4. Thus the winning
  Sep 30 run is **not evidence that critic ratio 4 caused the improvement**.

Valid-width padding masks here are not synthetic image corruption or
character-specific rules. They delimit actual data in padded batches and
apply to arbitrary word widths and either image height.

## 4. Rank of plausible impact, with confounds

1. **Highest-confidence structural candidate: correct B/E/D handling of
   variable-width padding.** It removes batch-composition contamination of
   reference appearance and critic evidence. Together with removing artificial
   constant-colour augmentation seams, it is directly relevant to the uneven
   backgrounds and style inconsistency observed at 32px.
2. **High-impact training candidate: corrected D readout plus stronger D/P
   optimization.** These change the scale and quality of adversarial feedback
   without introducing a new learned style architecture.
3. **Local supervision cleanup:** undamaged, symmetric PatchD crops and the
   minimum four-cell feature grid make the local critic's task more coherent.
4. **Batch/update budget and guidance:** halving batch size doubles the
   approximate number of minibatches per epoch; at the same critic ratio it
   also doubles G updates per epoch. Stronger retained writer supervision may
   matter. An epoch-matched comparison is therefore not update-matched.

The snapshots differ in resume source too: Sep 23 older ALPHA used
`last_fid_3.2284.pth`, while the Sep 30 file used `best_fid_3.0212.pth`.
The latter source is not supplied here. We cannot allocate the final gain
among inherited quality, extra updates, normalization, PatchD and losses.
No one-change controlled ablation exists in these archives.

## 5. What should not receive credit for this first crossing

- **Spatial reference memory:** introduced/restored in October, after sub-3
  results already existed. It is a plausible driver of the later move toward
  FID 1.x, not the first September crossing.
- **The later Sep 30 local-posterior/BN changes:** `beac08c976` added valid-only
  G BatchNorm, normalized style aggregation, deterministic local posterior
  tokens/global-only KL, and stronger transfer CTC. These changes deserve
  separate investigation for subsequent runs. The supplied sub-3 snapshot
  instead matches the Sep 26 settings and does not record the Sep 30 config
  changes. Without producing-code provenance, do not claim these caused its
  earlier crossing.
- **Merely increasing epochs to 100:** the saved run was configured for 84,
  and crossed at epoch 62.
- **Changing the FID implementation:** the inspected Sep 23→Sep 30 source
  range changes no `metric` files, dataset loader or path mapping. The saved
  validation settings also match. Historical dataset hashes are unavailable,
  so this is not a fresh end-to-end metric-reproduction proof.
- **New local tokens/allograph module:** the saved E configuration remains
  1 global + 7 local tokens, 32D style, 256D queries and four heads. The main
  refinements in this window were correctness and optimization, not a new
  global/local topology.

## 6. Evidence and reproducibility

Generated, Git-excluded evidence is in
`agent/analysis/2026-11-09-uppercase-history/`:
`checkpoint_histories.json` contains full extracted rows/configs for the
inspected Sep 23 through Nov 9 archives; `teacher_hashes.json` records exact
state hashes. Diagnostic helpers are under ignored `agent/tests`.

Critical checkpoint SHA256 values:

| File | SHA256 |
| --- | --- |
| Sep 23 DEV best | `ff0d57945efa50e806cceb8546e38591e3537788099e73a6ffc2343752c01e96` |
| Sep 23 older ALPHA last | `6c4cbf5c75ecbf81b71a8c9b0f20beda62ec844aec7c6c3f1ddccd7c0ca26856` |
| Sep 30 older ALPHA best | `63b089ae01e96d692b06f8a9ae77acc2e489510dcd0f78be6b0fe97ab12f8ce7` |

No old checkpoint was modified or relabeled. This investigation changes no
historic scores and makes no causal claim based solely on chronological
correlation.
