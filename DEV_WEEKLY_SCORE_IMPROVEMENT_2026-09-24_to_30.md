# DEV: changes behind the September 30 checkpoint improvement

## 1. Scope and conclusion

This report covers DEV's commits from **September 24, 2026 through September 30, 2026, 00:47:18 (UTC+7)**. The endpoint is your request to analyze `chkpoints/30_9_2026`. Changes made in response to that analysis, and subsequent MAIN experiments, are excluded.

The checkpoint being explained is [last_fid_3.0053_dev_branch.pth](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/last_fid_3.0053_dev_branch.pth), saved at epoch 84. Its comparison is the older [best_fid_3.4892_dev.pth](/home/quq/machineLearning/HTG/chkpoints/23_9_2026/best_fid_3.4892_dev.pth), saved at epoch 74.

The strongest explanation supported by the code is a **bundle of cleaner style features, better-calibrated discriminator supervision, less destructive patch preprocessing, and a changed optimization budget**. There was no new fusion architecture, no new pretrained teacher, and no improvement to the metric formula in these commits.

At matched epoch 74, the newer history has **10.77% lower FID, 31.74% lower KID, and 9.35% lower HWD**. Its final epoch 84 reaches FID **3.00527**, KID **0.09024**, HWD **0.10096**. These are measured improvements, but the contribution of each individual change is not isolated by an ablation.

## 2. Commit boundary: what counts, and what does not

All times below are UTC+7. The generic Git commit messages do not describe the changes adequately; this report uses their actual diffs.

| Commit | Date/time | Actual contents | Relationship to achieved scores |
|---|---|---|---|
| [1bbc082](https://github.com/rquq/HGGAN_test/commit/1bbc0822e04c42a722eb2584714f1beef4cc4eae) | Sep 24, 23:34:12 | Contextual-loss restoration and bounded computation; cleaner StrokePatchD inputs; minimum patch feature grid; batch changes; removal of writer-loss decay | Relevant pre-analysis changes |
| [10c66e5](https://github.com/rquq/HGGAN_test/commit/10c66e580a1ed8891a8bf79d9c89419eadaa9d8f) | Sep 26, 10:55:15 | Padding-aware style extraction and D blocks; corrected D pooling; edge-filled DiffAug; critic/LR changes; removal of texture and masking code | Relevant pre-analysis changes |
| [fa5a1fc](https://github.com/rquq/HGGAN_test/commit/fa5a1fce5f62a9088e3d8660b11f0e7051231d61) | Sep 30, 00:37:41 | GAN 32/64 epoch limit: 84 → 100 | Before the request, but cannot explain a checkpoint already saved at epoch 84 |
| 536cb87 | Sep 30, 01:46:50 | Additional encoder/global-local/KL/generator fixes | **Excluded: after the analysis request** |
| 8fdef9e | Oct 1, 14:13:17 | Subsequent fixes | **Excluded** |

The September 23 frequency-loss experiment is outside this reporting week, but matters as the immediate parent of the September 24 diff. The important distinction is that the older saved DEV checkpoint already used contextual loss; its settings are not identical to that transient Git parent.

Checkpoint files do not contain a per-epoch source commit ID. Therefore, commit order establishes the eligible changes, but cannot establish the exact epoch at which every edit entered a resumed run.

## 3. What improved in the saved histories

All three metrics below are lower-is-better. KID uses the project's stored reporting scale, not an assumed raw value from another paper.

| Checkpoint/history point | Epoch | FID | KID | HWD |
|---|---:|---:|---:|---:|
| Older DEV, saved best-FID checkpoint | 74 | 3.48918 | 0.12485 | 0.10591 |
| Newer DEV, matched epoch | 74 | 3.11335 | 0.08522 | 0.09601 |
| Newer DEV, best-KID epoch | 76 | 3.16112 | 0.08483 | 0.10239 |
| Newer DEV, final/best-FID checkpoint | 84 | 3.00527 | 0.09024 | 0.10096 |

### Matched-epoch comparison

Comparing epoch 74 with epoch 74 gives reductions of:

- FID: **10.77%**.
- KID: **31.74%**.
- HWD: **9.35%**.

Comparing the two saved checkpoint endpoints, epoch 74 versus epoch 84, gives FID **−13.87%**, KID **−27.72%**, HWD **−4.67%**. This endpoint comparison includes ten additional epochs, so it is less controlled.

There is another necessary qualification: **0.12485 was not the older run's best KID**. Its minimum KID was **0.10119 at epoch 53**. The newer run's minimum, **0.08483 at epoch 76**, is **16.16% lower** than that older minimum. The larger 31.74% figure describes the matched epoch, not a best-versus-best KID comparison.

### The improvement was mainly late, not universal

The newer run was not better at every earlier epoch. For example:

| Epoch | Older FID / KID / HWD | Newer FID / KID / HWD |
|---:|---|---|
| 50 | 4.08576 / 0.10909 / 0.12007 | 4.81859 / 0.17342 / 0.14305 |
| 60 | 4.09271 / 0.15822 / 0.13545 | 4.19886 / 0.14928 / 0.14103 |
| 70 | 3.72861 / 0.14165 / 0.12395 | 3.60483 / 0.12370 / 0.12567 |
| 74 | 3.48918 / 0.12485 / 0.10591 | 3.11335 / 0.08522 / 0.09601 |

Across epochs 60–74, newer mean FID improves **4.18%** and mean KID improves **15.99%**, but mean HWD is **0.54% worse**. Across epochs 71–74, mean FID/KID/HWD improve **12.14% / 35.12% / 10.17%**. Thus the late improvement is stronger than the whole-window improvement, especially for HWD.

The newer final ten epochs, 75–84, average FID **3.18950**, KID **0.09948**, HWD **0.10651**. Epoch 84's best FID does not also give the best KID or HWD.

Evidence: [new DEV epoch CSV](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/dev_metrics.csv), embedded history in the older checkpoint, and [checkpoint audit](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/checkpoint_audit.json). The histories have different early trajectories; they are not a controlled identical-prefix continuation.

## 4. Style encoder/backbone: remove padding contamination

### 4.1 Valid widths reach the frozen backbone

Previously, valid image lengths did not consistently govern the backbone computation. Batch padding could enter intermediate convolutions and subsequently affect boundary features. The September 26 change passes reference lengths into the style backbone and masks invalid feature columns as features move through it.

This distinction matters: **padding exclusion is not deletion of real background texture**. Gray paper, uneven illumination, and scan texture inside a word's valid image remain visible. Only the canvas beyond the supplied valid width is excluded.

### 4.2 Exact feature-length propagation

Feature widths are now propagated using actual convolution, padding, pooling, and stride geometry, including constant-padding offsets. This replaces proportional estimates such as “input length divided by canvas width, multiplied by feature width.”

The proportional estimate can be wrong for short references or odd widths. Incorrect lengths allow padded features into attention or exclude valid evidence. Exact lengths are used for the final style features, intermediate features, and contextual-loss inputs.

### 4.3 HeavyCNN becomes padding-aware; it is not newly invented

The existing HeavyCNN attention block remains the multibranch block already in DEV. This week's changes make its operations respect valid positions:

- Invalid columns are masked at its input, after its convolution branches, and after residual combination.
- GroupNorm statistics exclude invalid columns.
- Squeeze/excitation pooling averages valid positions instead of the complete padded canvas.
- Style-token attention receives exact feature lengths.

This reduces a dependency on what other samples happen to determine the batch's maximum width. It is a plausible improvement to style consistency and distribution learning, particularly when short and long words share a batch.

The changes do **not** introduce the later compact MAIN encoder, later global/local normalization, or a new allograph module. The backbone's no-length path used in pretraining retains its prior BatchNorm behavior.

Implementation also avoids repeated masking/length calculation in width-preserving layers and caches width masks within a forward pass. These are computational cleanups; no controlled throughput measurement is available to assign a speedup.

Historical source: [valid-width utilities and backbone traversal](https://github.com/rquq/HGGAN_test/blob/10c66e580a1ed8891a8bf79d9c89419eadaa9d8f/networks/module.py#L19), [HeavyCNN](https://github.com/rquq/HGGAN_test/blob/10c66e580a1ed8891a8bf79d9c89419eadaa9d8f/networks/module.py#L146), [encoder forward](https://github.com/rquq/HGGAN_test/blob/10c66e580a1ed8891a8bf79d9c89419eadaa9d8f/networks/module.py#L535).

## 5. Global D: correct the supervision's geometry

### 5.1 Pool valid image area, not feature area divided by character count

The old readout accumulated feature-map values and normalized by label length. Consequently, the scale of the discriminator feature depended on feature-map height, image width per character, and resolution.

The revised readout first averages over height, then averages over valid feature width, and applies the configured fixed `pooling_gain: 32` before the final linear head. Existing optional width-context processing is retained; this is not a new attention architecture.

This makes the meaning of the readout more consistent across image geometry. A 64px image does not inherently get a larger accumulated score merely because it has more vertical feature cells. The same rule serves 32px and 64px; it is not a benchmark-specific branch.

**This changes adversarial training calibration, not FID/KID calculation.** Scores cannot improve numerically just because this pooling expression changed; its effect must occur through the trained generator.

### 5.2 D residual blocks track valid widths

DBlock receives valid lengths, excludes invalid input/intermediate columns, propagates lengths through downsampling, and masks the residual result. This limits padded-canvas leakage into valid boundary features.

Generator valid-width/normalization fixes made after the September 30 analysis are not part of this explanation.

Historical source: [D residual block](https://github.com/rquq/HGGAN_test/blob/10c66e580a1ed8891a8bf79d9c89419eadaa9d8f/networks/BigGAN_layers.py#L461), [global D pooling and forward](https://github.com/rquq/HGGAN_test/blob/10c66e580a1ed8891a8bf79d9c89419eadaa9d8f/networks/BigGAN_networks.py#L443).

## 6. StrokePatchD: cleaner local targets and a useful feature grid

### 6.1 Remove destructive stripe corruption

The previous patch preprocessing could erase horizontal and vertical stripes. Its nominal `mask_ratio_range` argument did not actually constrain the operation. A one- or two-pixel stripe is proportionally much more destructive in a 16×16 crop than in a 32×32 crop.

The September 24 commit removes these calls from GAN training and removes their configuration. September 26 then deletes the unused masking module.

This means StrokePatchD is asked to judge actual local handwriting/paper appearance rather than partly erased patches. It does **not** mean removing CTC lengths, attention padding masks, evaluation cropping, or the valid-width handling described above.

### 6.2 Match real/fake patch preprocessing

Previously, the real patch path mixed original-real and DiffAug-real crops, while generated patches were unwarped. The revised patch loss uses unwarped real crops against unwarped fake crops.

Global D still uses symmetric DiffAug on real and fake. Only the asymmetric local discriminator preparation was removed. This reduces an avoidable shortcut where P can identify augmentation differences rather than handwriting quality.

### 6.3 Do not downsample small patches to a nearly collapsed grid

StrokePatchBlock downsamples only if both resulting spatial dimensions remain at least four. A 16×16 patch therefore retains at least a 4×4 readout instead of collapsing to 2×2. A 32×32 patch can still reach 4×4.

This is one geometry-derived rule for all patch sizes, with no additional parameters. It preserves more positions for stroke boundaries and paper variation in smaller crops. It is a plausible reason the revised training signal is more useful at 32px, not proof that this block alone lowered KID.

Historical source: [September 24 patch changes](https://github.com/rquq/HGGAN_test/commit/1bbc0822e04c42a722eb2584714f1beef4cc4eae), [pre-analysis patch block](https://github.com/rquq/HGGAN_test/blob/10c66e580a1ed8891a8bf79d9c89419eadaa9d8f/networks/BigGAN_networks.py#L505).

## 7. DiffAug: do not create artificial paper seams

Width scaling and translation previously filled uncovered positions with constant background values. Inside a valid word image with real paper texture, that can introduce a conspicuous artificial strip.

The revised path uses edge replication where these transformations need fill inside the valid region. Outside the valid word, the normal padding value remains. Real and fake follow the same policy.

This removes an augmentation-induced discrepancy; it does not force genuinely irregular paper to become rectangular, and does not guarantee artifact-free generation. Its plausible metric benefit is indirect: D is less encouraged to exploit artificial fill seams.

Historical source: [valid-region augmentation](https://github.com/rquq/HGGAN_test/blob/10c66e580a1ed8891a8bf79d9c89419eadaa9d8f/networks/utils.py#L556).

## 8. Losses: what changed, and what cannot receive credit

### 8.1 Contextual loss restored after the frequency experiment

The September 24 Git diff changes all three GAN configs to `lambda_ctx: 0.1` and `lambda_frequency: 0.0`, replacing the preceding frequency-loss experiment.

Contextual supervision compares frozen-backbone features of the **real style reference** with features of the **generated style-transfer image**. Their text need not match, so this is non-aligned feature correspondence, not a pixel reconstruction target. Reference features are detached; generated features remain differentiable.

The implementation reuses reference and generated features already needed elsewhere, avoiding redundant backbone forwards.

However, **the older September 23 saved DEV checkpoint also had `lambda_ctx: 0.1`**. Therefore, “we introduced contextual loss and it caused the improvement over that checkpoint” would be incorrect. The meaningful changes are its bounded computation and more reliable valid feature support, alongside the other training changes.

### 8.2 Bound contextual-loss cost

`CXLoss(max_tokens=256)` crops each feature map to its valid width before adaptive pooling. It limits pairwise comparison to at most 256 positions per feature map, rather than an unbounded spatial quadratic computation. A positive-integer budget is validated; the optional full-resolution mode remains available.

This controls cost and excludes padding from the feature correspondence. Pooling changes the loss's effective spatial sampling, so it is an approximation rather than a promise of identical gradients. Its quality contribution was not individually measured.

### 8.3 Keep writer supervision active late in training

The old 32px config decayed writer loss from 0.5 to 0.25 across epochs 48–60. The new config keeps it at **0.5**, consistent with the 64px config.

This gives the generator stronger writer-consistency supervision late in training, when adversarial improvements otherwise could trade away style identity. It is relevant to the late improvement, but cannot be assigned a measured share of HWD or KID without an ablation.

### 8.4 Frequency loss is disabled in the measured newer run

Its optional implementation was made geometry-relative: half-height patches by default instead of fixed 32px patches, with validated arguments. It compares averaged log Fourier magnitude, not phase or exact location.

With `lambda_frequency: 0.0`, it is not computed and cannot explain the checkpoint's improvement. This is available experimental code, not active supervision.

Historical source: [bounded CX and optional spectral loss](https://github.com/rquq/HGGAN_test/blob/1bbc0822e04c42a722eb2584714f1beef4cc4eae/networks/loss.py#L117), [GAN contextual wiring](https://github.com/rquq/HGGAN_test/blob/10c66e580a1ed8891a8bf79d9c89419eadaa9d8f/networks/model.py#L2410).

## 9. Optimization: the batch/critic interaction is the important detail

These are settings saved in the actual checkpoints, rather than only the latest working-tree config:

| Setting | Older DEV | Newer DEV |
|---|---:|---:|
| Image height | 32 | 32 |
| Training batch | 16 | 8 |
| `num_critic_train` | 2 | 4 |
| Base G LR | 0.0002 | 0.0002 |
| Base D LR | 0.0001 | 0.0002 |
| Base P LR | 0.00012 | 0.0002 |
| Contextual weight | 0.1 | 0.1 |
| Frequency weight | Absent | 0.0 |
| Writer weight late in training | 0.25 | 0.5 |
| Decay start / duration / floor | 36 / 48 / 0.15 | 36 / 48 / 0.15 |
| Saved epoch limit | 84 | 84 |

For approximately N training examples per epoch:

- Old G/E updates: `N / (16 × 2) = N / 32`.
- New G/E updates: `N / (8 × 4) = N / 32`.
- Old D/P updates: `N / 16`.
- New D/P updates: `N / 8`.

Therefore, **the generator/encoder gets approximately the same number of updates per epoch, while D/P gets twice as many**, with higher D/P learning rates. It is misleading to describe this solely as “critic 4 slowed G by half,” because the batch size simultaneously halved.

The optimizer counters support this: older G has 153,994 updates over 74 epochs (2,081/epoch), while newer G has 174,825 over 84 epochs (about 2,081/epoch). Older D/P has 307,988 updates; newer D/P has 699,300, corresponding to approximately 4,162 versus 8,325 updates per epoch.

Smaller batches also change gradient noise and BatchNorm behavior. These differences and the LR/critic changes are intertwined; a pure critic-ratio conclusion cannot be extracted from this comparison.

The 32px `training.eval_batch_size` changed from 16 to 8, but the saved official **`valid.batch_size` stayed 32**. Do not conflate these two settings.

The decay schedule did not change in the eligible commits. The September 30 extension to 100 epochs did not move its start or duration, and the analyzed checkpoint still contains the 84-epoch configuration. Consequently, delayed decay cannot be credited for this gain.

## 10. Texture removal, logging, and cleanup

The texture refinement module was deleted, along with its import, constructor arguments, reset/freezing logic, residual path, and configuration toggles. Generator output follows the ordinary output layer and `tanh` path.

But **the older DEV checkpoint already had texture refinement disabled**. Removing its unused machinery is a design/state cleanup, not evidence that changing the active texture forward path caused this DEV improvement. The gain should not be described as a new texture generator's success or as a measured texture on/off ablation.

Other changes include restoring contextual-loss console/W&B meters, logging optional losses only when active, validating finite nonnegative loss weights, deleting the orphaned masking file, and adding the earlier weekly report. These improve cleanliness, cost, and inspectability; logging or documentation alone does not lower a metric.

### Complete changed-file inventory

The eligible commits touch these 13 paths:

| Path relative to DEV | Change |
|---|---|
| `configs/gan_iam_32.yml` | Batch, active losses, writer schedule, D/P LR, critic ratio, pooling gain, texture toggle, epoch limit |
| `configs/gan_iam_64.yml` | Active losses, D/P LR, critic ratio, pooling gain, texture toggle, epoch limit |
| `configs/gan_iam_local.yml` | Active losses and common training settings; deliberately smaller local batch retained |
| `configs/gan_image.yml` | Dead texture toggle removed |
| `networks/BigGAN_layers.py` | Valid-width D residual blocks |
| `networks/BigGAN_networks.py` | D pooling, patch grid, texture-path removal |
| `networks/loss.py` | Bounded contextual computation and optional geometry-relative spectral loss |
| `networks/model.py` | Loss wiring, valid reference lengths, clean patch preparation, optional-loss logging |
| `networks/module.py` | Exact width propagation and padding-aware backbone/HeavyCNN/style features |
| `networks/utils.py` | Edge-filled valid-region augmentation |
| `networks/masking.py` | Deleted |
| `networks/texture_generator.py` | Deleted |
| `DEV_WEEKLY_COMMIT_REPORT_2026-09-18_to_25.md` | Historical documentation added |

## 11. What stayed unchanged or cannot explain the gain

### Pretrained teachers

The paths recorded as the GAN resume source differ between the two checkpoints. Nevertheless, fresh tensor comparisons show that their **R, W, and B state tensors are identical**. Improved OCR/writer/backbone pretraining weights therefore do not explain this difference.

Their runtime use can still differ: padding-aware execution of the same B weights changes extracted features.

### Fusion and generator topology

There is no `fusion.py` change in the eligible commits. Earlier allograph/content-style attention, local-evidence mechanisms, shared 32/64 generator topology, and single-character sampling support were inherited, not introduced this week.

Apart from deletion of disabled texture machinery, this week's major active changes are around feature extraction, discriminator supervision, losses, and optimization—not a replacement G backbone.

### Evaluation protocol

The complete saved `valid` configuration is identical between old and new DEV: FID/KID/HWD enabled, the same evaluation seed 123456, random corpus disabled, float32 cache, validation batch 32, Inception dimensions 2048, KID 50 subsets of size 1,000, polynomial degree 3, and coefficient 1.

No metric or dataset implementation file changes in the eligible commits. The saved KID is `mean(MMD²) × 100`; newer final KID 0.09024 corresponds to raw MMD² approximately **0.0009024**. Cross-paper comparisons must use matching units and protocol.

The checkpoints do not preserve a historical HDF5 content hash. Identical declared data/protocol is supported; byte-identical historical input files cannot be proven from this metadata alone.

### The later fixes

The post-analysis global/local encoder normalization, deterministic local-token changes, global-only KL, local-evidence adjustments, style CTC weight increase, and G masking/normalization fixes were **not responsible for scores already saved on September 30**.

## 12. How confidently can we explain the improvement?

| Assessment | Evidence level |
|---|---|
| The newer history improves late FID/KID, and reaches lower best HWD | Directly measured from saved histories |
| Frozen R/W/B weights and declared validation settings are unchanged | Direct checkpoint comparison |
| Style features, D pooling, patch targets, augmentation seams, and optimizer balance changed | Direct historical diffs and saved settings |
| These corrections plausibly improve handwriting/paper distribution learning | Mechanistic interpretation, not isolated measurement |
| One specific change caused a particular percentage of the KID reduction | **Not established** |
| Critic 4 is uniquely necessary for the improvement | **Not established** |
| All special-character/style-reference cases are solved | **Not established** |

As supporting context, the same September 30 folder's RANDOM_CROP checkpoint reaches FID **2.68924**, KID **0.06732**, HWD **0.09394** with batch 8 and critic ratio 2. It has approximately twice DEV's G/E update opportunities per epoch and different GAN resume provenance. It is not a clean critic-ratio ablation. It does show that good aggregate scores do not require critic ratio 4; the shared fixes and optimization budget matter too. The original audit also found reference-conditioning failures despite those strong aggregate scores.

The most defensible priority order for explaining DEV is:

1. **Cleaner training evidence:** valid-width style features and normalization, matched uncorrupted local patches, and fewer augmentation-induced seams.
2. **More coherent adversarial supervision:** valid-area D readout, padding-aware D features, and retained small-patch spatial detail.
3. **Changed late-training pressure:** sustained writer supervision plus more D/P updates and higher D/P LR, while roughly preserving G/E updates per epoch.
4. **Loss/cost cleanup:** bounded contextual supervision and feature reuse; inactive frequency/texture code receives no direct quality credit.

This ranking expresses a technical assessment, not an ablation ranking.

## 13. Short version suitable for a research progress report

During September 24–26, DEV was revised to make style extraction and adversarial supervision more consistent with valid image geometry. The changes included padding-aware backbone and attention normalization, exact feature-length propagation, valid-area discriminator pooling, preservation of small-patch spatial support, removal of destructive/asymmetric patch preprocessing, and edge-filled augmentation. Contextual supervision was retained with bounded spatial comparison, and late writer supervision was maintained. The training batch decreased from 16 to 8 while the critic ratio increased from 2 to 4, approximately preserving generator/encoder updates per epoch and doubling discriminator updates; D/P base learning rates increased to 2e-4. At matched epoch 74, the revised history improved FID/KID/HWD by 10.77%/31.74%/9.35%, and the final epoch 84 reached FID 3.0053. These results support the combined revision, but individual contributions are not isolated because the saved trajectories use different GAN resumes and lack per-change controlled ablations. Later September 30–October 1 fixes are excluded from this attribution.

## Evidence links

- [Older DEV checkpoint and embedded history](/home/quq/machineLearning/HTG/chkpoints/23_9_2026/best_fid_3.4892_dev.pth).
- [Newer DEV checkpoint](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/last_fid_3.0053_dev_branch.pth).
- [Full newer DEV metric CSV](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/dev_metrics.csv).
- [Saved checkpoint/config audit](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/checkpoint_audit.json).
- [Original September 30 analysis](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/REPORT.md).
- [Later fix implementation—excluded from score attribution](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/FIX_IMPLEMENTATION.md).
- [Earlier weekly code report](/home/quq/machineLearning/HTG/HGGAN_test/dev/DEV_WEEKLY_COMMIT_REPORT_2026-09-18_to_25.md).

Prepared from historical Git diffs and CPU checkpoint inspection. No model/configuration edits, training, or fresh GPU evaluation were performed for this report.
