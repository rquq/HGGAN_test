# `dev` versus `alpha`: training speed and FID convergence

## Executive summary

The current `dev` experiment is slower **per epoch** mainly because it performs a generator update on every training batch, while `alpha` performs one generator update every four batches:

- `dev`: `num_critic_train: 1`
- `alpha`: `num_critic_train: 4`

The discriminator is still updated every batch in both versions. Therefore, for the same number of batches in an epoch, `dev` executes approximately four times as many expensive generator-side forward/backward passes. The generator-side update is the expensive part of this system because it includes the generator, global and patch discriminators, OCR recognizer, writer classifier, multiple style-encoder passes, perceptual losses, and a single backward pass through the combined graph.

This also explains a large part of the faster FID convergence **when measured by epoch**. An epoch is no longer an equal unit of optimization:

| Quantity per epoch | `dev` | `alpha` |
|---|---:|---:|
| Discriminator updates | \(N\) | \(N\) |
| Generator updates | \(N\) | approximately \(N/4\) |
| Relative G-update density | 4× | 1× |

Here, \(N\) is the number of batches per epoch.

Using the reported results:

- `dev` at epoch 3 has performed about \(3N\) generator updates and \(3N\) discriminator updates.
- `alpha` at epoch 20 has performed about \(5N\) generator updates and \(20N\) discriminator updates.

So `dev` reaches FID 15 with approximately:

- **40% fewer generator updates**: \(3N\) instead of \(5N\)
- **85% fewer discriminator updates**: \(3N\) instead of \(20N\)
- **6.67× fewer epochs**

This means `dev` is not merely making four times as much progress per epoch. It is also more sample- and update-efficient: even after correcting for the different generator-update frequency, it reaches FID 15 with fewer generator updates.

The correct conclusion is:

> `dev` is slower per epoch because each epoch contains much more generator training, but it is substantially faster in convergence. Unless a `dev` epoch takes more than about 6.67 times as long as a `alpha` epoch, `dev` is also faster in wall-clock time to FID 15.

## Scope of the comparison

In this report, `alpha` refers to the `HGGAN_test/alpha` experiment tree. The comparison uses the current working-tree versions, including the current uncommitted changes in both experiment directories.

The two branch histories do not have a usable common merge base in the local clone, so this is a direct tree and training-path comparison, not a conventional merge-base diff.

## Major implementation differences

### 1. Generator update frequency

The training loop updates the discriminator every batch, then conditionally updates the generator using:

```python
if iter_count % self.opt.training.num_critic_train == 0:
```

The current configurations set this value to 1 in `dev` and 4 in `alpha`.

This is the dominant reason that `dev` is slower per epoch. A generator update is not just one generator forward pass. In both versions it includes:

1. Style encoding of the real reference.
2. Generation of three image groups in one batched call:
   - random-style generated images,
   - style-transfer images,
   - reconstruction images.
3. Global-discriminator evaluation of all generated images.
4. Extraction and patch-discriminator evaluation of all generated patches.
5. OCR recognition of all three generated image groups.
6. Style re-encoding of generated images.
7. Writer classification and backbone feature extraction.
8. Contextual, Gram, reconstruction, KL, and adversarial losses.
9. Backpropagation through the combined generator and style-encoder graph.
10. Gradient clipping and EMA updates.

`dev` pays this cost every batch. `alpha` pays it only on one batch out of four.

### 2. `dev` uses the original word image as the style reference

The dataset exposes three distinct views:

- `org_img`: the original word image with its natural geometry and aspect ratio.
- `style_img`: a resized version with width fixed to `CharWidth × text length`.
- `aug_img`: an augmented view of `style_img`; with writer augmentation enabled, this includes `RandomClip`.

`alpha` encodes `style_imgs`, while `dev` explicitly loads `org_imgs` as `style_refs` and feeds those to the style encoder.

This is important for both quality and speed:

- **Quality:** the original image preserves writer-specific spacing, aspect ratio, slant, character widths, and natural word geometry. Resizing every word to a fixed per-character width partially normalizes away those cues.
- **Speed:** original images have variable natural widths. Wider examples and batch padding produce more backbone activations and more attention/fusion tokens than the fixed-width normalized style view. This can increase both computation and memory use.

Despite the experiment name, the random crop is primarily the `aug_img` view. It is not the only style input in the pipeline.

### 3. More structured and compact style representation

`alpha` creates 32 learned style-query tokens because its number of style tokens is set equal to `style_dim`.

`dev` uses eight tokens:

- token 0 is an explicit global style summary;
- the remaining seven tokens capture local stroke/style details.

The `dev` encoder also:

- masks padded width positions in the backbone and projected multi-scale features;
- uses masked pooling for the global style token;
- supplies a key-padding mask to style cross-attention;
- constructs all projection layers before optimizer creation, avoiding forward-time module replacement.

These changes improve the meaning and stability of the style code. They also reduce cross-attention cost relative to 32 style queries. This is a speed-saving change, but it is not large enough to offset the fourfold increase in generator-update frequency and the additional style-cycle computation.

### 4. Safer style/content fusion in the generator

`alpha` replaces the content embedding directly with the output of the Mamba style/content fusion path from the first update.

`dev` uses a learnable residual interpolation:

```python
y_mixed = content + tanh(fusion_alpha) * (fused_content - content)
```

`fusion_alpha` starts at zero, so training starts from the reliable content-only path and gradually learns how much fused style information to use.

This likely improves early FID because it prevents an untrained fusion module from corrupting character conditioning at initialization. The model first preserves readable content and then learns style modulation.

`dev` also masks invalid text positions and runs the bidirectional Mamba scan separately for groups of equal valid sequence length. This prevents padded word width from leaking into valid character representations.

The grouped-by-length fusion is more correct, but the Python loop over unique lengths and multiple smaller Mamba launches can reduce GPU utilization compared with one padded batched scan. It is a plausible secondary source of slowdown.

### 5. Better discriminator semantics

`alpha` sends both the complete real word and the randomly clipped word to the global discriminator as real samples.

`dev` removes the cropped sample from the global discriminator and retains it only for the patch discriminator.

That distinction is conceptually correct:

- the global discriminator should learn the distribution of complete words;
- the patch discriminator can treat local crops as valid handwriting texture/stroke samples.

Training the global discriminator to regard truncated words as real weakens its global-shape signal. Removing that contradiction likely contributes to the large early FID improvement.

`dev` also adds lazy R1 regularization to the global discriminator every 16 iterations. R1 requires gradients with respect to real images and therefore makes those iterations noticeably more expensive. Because it is applied only once every 16 iterations, it is a secondary rather than dominant slowdown.

The discriminator learning rate is also reduced:

- `dev`: \(1\times10^{-4}\)
- `alpha`: \(2\times10^{-4}\), inherited from the shared learning rate

Together with four times more frequent generator updates, this shifts the optimization balance strongly toward the generator and reduces the chance that the discriminator outruns it early in training.

### 6. Revised generator objectives

`alpha` dynamically estimates loss weights from five separate `torch.autograd.grad` calls. These gradient probes retain the training graph and add extra GPU work on generator-update batches. They also produce noisy, batch-dependent weights based on gradients with respect to different intermediate tensors.

`dev` removes those five gradient probes and uses fixed, interpretable coefficients:

- CTC: 1.0
- latent/style information: 1.0
- writer identity: 0.5
- reconstruction: 10.0
- style-transfer cycle: 1.0
- content-removal adversary: 0.02

It adds two objectives:

- **Style-transfer cycle loss:** re-encodes the style-transfer output and matches its global style token to the source style. This requires one additional style-encoder/backbone forward pass on every generator update.
- **Content adversarial loss:** a gradient-reversal content probe discourages the style representation from retaining character identity. Its linear head is cheap, but its gradient changes the encoder representation.

The additional style-encoder pass is a meaningful secondary slowdown. In `dev`, a generator update uses the style encoder for:

1. the real original style reference;
2. randomly styled generated images;
3. transferred generated images.

`alpha` performs the first two but not the third.

The objective rescaling also matters:

- reconstruction receives a strong fixed weight of 10;
- contextual and Gram weights drop from 2.0 to 0.1;
- Gram loss is normalized by feature energy;
- KL changes from a dimension-wise sum to a mean and its configured coefficient changes from 0.0001 to 0.1.

The KL coefficients cannot be compared as a simple 1000× increase because the loss reduction itself changed from sum to mean. Overall, `dev` gives more stable, dimension-independent loss magnitudes and places greater emphasis on faithful reconstruction, content readability, and transferable writer style.

### 7. Corrected EMA behavior

`dev` improves the exponential moving average by:

- updating parameters in place;
- copying model buffers such as running statistics and spectral-normalization buffers;
- keeping EMA models in evaluation mode;
- maintaining EMA only for the trainable generator and encoder rather than duplicating the frozen backbone.

FID is commonly evaluated with EMA weights in this code path. A better synchronized EMA can therefore improve reported FID even when the live model changes are less dramatic.

EMA updates happen four times as often per epoch in `dev`, matching the increased generator-update frequency. This adds some memory-bandwidth cost but is smaller than the generator backward pass itself.

### 8. Data sampling and crop alignment

`dev` adds a writer-balanced batch sampler, but the current configuration sets:

```yaml
writer_balanced_batches: false
```

Therefore it is not active in the current run and does not explain the present speed or FID difference.

The random crop start position is now aligned to `CharWidth` units and includes the last valid aligned position. This improves crop/label geometry consistency, but its CPU cost is negligible compared with model computation.

### 9. Evaluation cadence is a major benchmarking confounder

The current local configurations differ:

- `dev`: `eval_epoch_val: 1`
- `alpha`: `eval_epoch_val: 0.01`

The code converts this value into:

```python
eval_interval_iters = int(eval_epoch_val * len(train_loader))
```

Thus `0.01` requests approximately 100 validation runs per epoch, whereas `1` requests one validation run per epoch. Full FID/KID/HWD/CMMD validation can be expensive.

This setting should make the current `alpha` end-to-end job slower, not faster. If `alpha` is observed to train faster, one of the following is likely true:

1. “training speed” is being judged from the interval between training log lines, excluding validation;
2. the compared run was launched with a different saved configuration;
3. validation metrics were disabled or the run did not reach the validation trigger;
4. the comparison is per training iteration or per epoch, not total process time.

For a valid timing comparison, both experiments must use the same validation cadence, validation metrics, batch sizes, checkpoint initialization, hardware, and data-worker settings.

## Why FID improves so much earlier

The improvement from FID 15 at epoch 20 to FID 15 at epoch 3 is most plausibly the combined effect of the following changes, in approximate order of importance.

### High-confidence causes

1. **Four times as many generator updates per epoch.** This directly compresses generator learning into fewer epochs.
2. **Original, geometry-preserving style references.** The encoder sees natural writer spacing and aspect ratio instead of only fixed-width normalized words.
3. **Correct global-versus-local discriminator targets.** Randomly truncated words no longer teach the global discriminator that incomplete words are real.
4. **Stable fixed loss weights.** The generator is no longer driven by noisy batchwise gradient-ratio weights.
5. **Residual-gated fusion initialized to the content path.** Early generated text remains structurally reliable while style fusion is learned.

### Medium-confidence causes

6. **Explicit global style token plus compact local tokens.**
7. **Padding-aware style encoding and length-aware Mamba fusion.**
8. **Style-transfer cycle consistency and content removal from the style code.**
9. **Corrected EMA parameters and buffers.**
10. **Lower discriminator learning rate and lazy R1 regularization.**

FID alone cannot attribute the gain to one item. A controlled ablation is required to measure each contribution.

## Is `dev` actually slower to reach the target?

Let:

- \(T_r\) be the wall-clock time of one `alpha` epoch;
- \(T_d\) be the wall-clock time of one `dev` epoch;
- \(s=T_d/T_r\) be the per-epoch slowdown.

The approximate time ratio to FID 15 is:

\[
\frac{\text{dev time to FID 15}}{\text{random\_crop time to FID 15}}
=
\frac{3T_d}{20T_r}
=
\frac{3s}{20}
\]

Examples:

| `dev` slowdown per epoch | Time ratio to FID 15 | Interpretation |
|---:|---:|---|
| 2× | 0.30× | `dev` reaches the target about 3.3× faster |
| 3× | 0.45× | `dev` reaches the target about 2.2× faster |
| 4× | 0.60× | `dev` reaches the target about 1.7× faster |
| 6× | 0.90× | `dev` is still about 11% faster |
| 6.67× | 1.00× | equal wall-clock time |

Therefore, an epoch-level slowdown is acceptable—and expected—if the actual goal is time to a target FID.

## Recommended measurements

The current training logs do not provide a clean component-level timing breakdown. The slowdown attribution above is based on the executed code paths, not a CUDA profiler trace. The next comparison should record:

1. seconds per data batch;
2. seconds for the discriminator phase;
3. seconds for the generator phase;
4. number of D and G optimizer steps;
5. seconds spent in validation;
6. peak allocated GPU memory;
7. FID versus wall-clock time;
8. FID versus generator updates;
9. FID versus images seen.

The most useful plots are:

- FID vs epoch;
- FID vs wall-clock hours;
- FID vs generator step;
- FID vs total optimizer steps.

The epoch plot alone is misleading because the update schedules differ.

## Recommended speed improvements with the lowest quality risk

1. **Keep `num_critic_train: 1` until its contribution is ablated.** It is probably a major part of the early FID gain.
2. **Enable mixed precision for G, D, P, backbone, and attention operations**, while keeping numerically sensitive R1, CTC loss, and normalized loss calculations in FP32.
3. **Batch the two generated-image style-encoder calls together.** `fake_imgs` and `style_imgs` can potentially be padded and concatenated, encoded in one backbone call, and split afterward. This preserves the objectives while reducing kernel-launch overhead.
4. **Profile the grouped length-aware Mamba scan.** If it is a hotspot, use length buckets in the data loader so each batch contains fewer unique lengths. That retains padding correctness while reducing the number of scan launches.
5. **Report validation separately.** Use the same `eval_epoch_val` for both experiments and exclude validation time when reporting raw training throughput.
6. **Consider `num_critic_train: 2` only as a controlled speed/quality ablation.** It may reduce epoch time, but it also halves generator updates and changes GAN balance. Compare by generator step and wall-clock time, not only epoch.
7. **Do not remove the style-cycle or padding fixes without an ablation.** They are plausible contributors to the FID gain and are better optimization targets for batching than deletion.

## Suggested ablation order

To determine which changes produced the gain without running an impractically large experiment grid:

1. `dev` unchanged.
2. `dev` with `num_critic_train: 2`.
3. `dev` with `num_critic_train: 4`.
4. `dev` with the style-transfer cycle disabled.
5. `dev` using resized `style_imgs` instead of original `org_imgs`.
6. `dev` with global-D crop handling reverted.
7. `dev` with direct fusion instead of residual-gated fusion.

For every run, hold the seed, initialization, checkpoint, data order, batch size, evaluation corpus, and evaluation cadence constant. Record FID against both wall-clock time and generator updates.

## Final assessment

The present behavior is internally consistent:

- `dev` epochs are slower because they contain four times as many generator updates and each generator update includes an additional style-transfer encoding path.
- `dev` converges much faster because it trains the generator much more often and improves the correctness of style input, fusion, discriminator targets, loss scaling, padding handling, and EMA evaluation.
- Reaching FID 15 at epoch 3 instead of epoch 20 is a genuine efficiency improvement, not merely an epoch-count illusion: after adjusting for the update ratio, `dev` still uses about 40% fewer generator updates.
- The most meaningful remaining question is not “which epoch is faster?” but “which system reaches the desired FID in less wall-clock time and with fewer optimizer updates?”
