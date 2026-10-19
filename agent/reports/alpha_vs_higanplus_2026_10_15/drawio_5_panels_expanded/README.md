# ALPHA — five training panels + one HiGAN+ module-comparison tab

Open `alpha_training_architecture.drawio` in diagrams.net using **File → Open From → Device**. All modules, inner operations, connectors, labels and formulas are editable native objects. `preview.html` is an offline, self-contained SVG preview of those same native pages.

## The important correction to panel (a)

Panel **(a) Reference-guided GAN training** displays only the reference-transfer branch:

`reference image → frozen B → E → S_ref → G(S_ref, y′) → generated word → D / R / W+B → losses`.

There is **no independently sampled prior z entering G in panel (a)**. The earlier compact version merged three branches and made their different style sources ambiguous. This revision separates their presentation, not their implementation.

The actual current GAN still computes all three branches together, using shared weights:

- Random generation: `G(z, y′)`.
- Reference transfer: `G(S_ref, y′)`.
- Reconstruction: `G(S_ref, y)`.

The equations under (a) explicitly describe the **full shared training objective**, not only the illustrated transfer branch. Repeated labels `y′` and `w` beside losses are copies of batch supervision, not additional model inputs.

## Six tabs

1. **(a) GAN training** — reference-guided image flow; whole-word adversarial loss, OCR CTC and writer CE; full shared objective below.
2. **(b) Reconstruction** — pixel reconstruction, random global-code recovery, reference global-cycle recovery, posterior KL and content-adversarial regularization.
3. **(c) Local refinement** — real and generated native patches, height-relative stratified cropping, StrokePatchD and multiscale contextual supervision.
4. **(d) Random-style inference** — seeded Gaussian sampling, local-slot centering, text context, shared G and generated image. No reference input.
5. **(e) Reference-style inference** — reference B/E features and target text through shared G. Evaluation uses the global posterior mean, not independent prior sampling.
6. **(f) Modules vs HiGAN+** — six matched side-by-side module pairs, with original HiGAN+ on the left and current ALPHA on the right. Rows cover E, G/fusion, crop + P, W, D, and backbone B geometry. Every row has a structural-change explanation and source pointers. No forward arrow connects the two model versions.

The comparison tab uses a taller **2880 × 3000** page. Existing tabs (a)–(e) are preserved, including their content and geometry; the builder checks this when updating the file. This addition does not silently apply the separate precision corrections identified in the earlier (a)/(b) verification.

### Comparison source and attribution

The baseline is the original local checkout `/home/quq/machineLearning/HTG/HiGANplus/HiGAN+`, commit `5d37de733f79e2a72a79c0f66e1865329ff56b4e`, not `classic_optimized`. Its tracked Python/YAML source was inspected directly. Existing bytecode/cache files in that checkout were left untouched.

- [Original E / W / B source](https://github.com/ganji15/HiGANplus/blob/5d37de733f79e2a72a79c0f66e1865329ff56b4e/HiGAN%2B/networks/module.py)
- [Original G / D / P source](https://github.com/ganji15/HiGANplus/blob/5d37de733f79e2a72a79c0f66e1865329ff56b4e/HiGAN%2B/networks/BigGAN_networks.py)
- [Original exhaustive patch extraction](https://github.com/ganji15/HiGANplus/blob/5d37de733f79e2a72a79c0f66e1865329ff56b4e/HiGAN%2B/networks/utils.py#L308)

C1/C2/C3 mark the three main redesigns. W pooling, global D readout and B geometry/padding are supporting changes. HiGAN+ already has multiscale backbone maps, CBN-conditioned residual GBlocks, a global critic, a patch critic, OCR supervision and writer supervision. The comparison does not claim these inherited elements are new or prove that any one change improved a metric.

## Where z comes from

`networks/rand_dist.py:128` constructs a seeded normal `Distribution`. Training calls `self.z.sample_()` and `center_style_prior(self.z)` in `networks/model.py:2258`. The configured shape per sample is `8 × 32`: one global token and seven local tokens. `networks/utils.py:244` subtracts the local-slot mean and leaves the global token unchanged.

This prior is not E's output, not a reference image, and not E's learned query vectors. Reference E instead returns `S_ref = [g, L]`. During VAE training its global token is sampled **conditional on reference features**; deterministic reference inference uses `g = μ_g`. These two sources of stochasticity must not be conflated.

## Mini-diagrams and attribution

The 2880 × 1960 canvas retains the five-panel organization, while expanding these modules internally:

- **C1 — global/local E:** visual backbone maps branch into global context and multiscale learned-query local evidence; the resulting tokens are packed together.
- **C2 — content-first G/fusion:** text context produces a content-only spatial seed; global style uses block-specific CBN projections; local evidence enters spatial cross-attention after GBlock 1.
- **C3 — local critic:** height-relative crop policy and StrokePatchD have their own connected internal paths.
- Writer attentive-statistics pooling and whole-word D pooling are shown as supporting changes. Retained backbone B, OCR R, random latent sampling and ordinary adversarial/reconstruction objectives are not labeled as new contributions merely because they appear in this architecture.

Solid arrows show forward computation; orange dashed arrows indicate supervision/target dependencies; green arrows indicate global-style conditioning. A dashed module boundary marks frozen parameters in the depicted GAN phase. Frozen teachers still propagate input gradients. Repeated modules share weights.

## Source and verification scope

Source snapshot: `/home/quq/machineLearning/HTG/HGGAN_test/alpha`, commit `7eed74c6b2d5286dc7b3e446ec1db3b165322c39`, normal 32/64 GAN configurations.

The builder checks native XML integrity, endpoint connectivity, one connected operation graph per training page, twelve intentionally independent forward graphs on the paired-comparison page, orthogonal authored routes, module overlap and unrelated-box transit. The official draw.io `Graph` / `mxCodec` renderer is exercised offline with GPU disabled; its report is in `native_renderer_verification.json`. The full network-dependent diagrams.net editor UI is not claimed as tested.

Read `verification.json` for external crossing counts. Remaining inner fan-in/crossings use jump arcs; the diagrams are not claimed to be globally crossing-free. This work changes documentation only, not model code, configuration, checkpoints or notebooks.
