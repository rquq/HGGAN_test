# ALPHA — five end-to-end panels

Open `alpha_training_architecture.drawio` in diagrams.net using
**File → Open From → Device**. It contains exactly five editable pages,
corresponding to the five panels of the supplied HiGAN+ overview:

1. (a) Training with disentangled representations: input/style/text → G →
   generated words → D/R/W+B → adversarial, CTC and writer CE comparisons.
2. (b) Regularization with reconstruction: three branches through shared G,
   pixel reconstruction, global information/cycle recovery, global KL and the
   gradient-reversed content probe.
3. (c) Refining local patches and reference appearance: real and generated
   images → shared adaptive cropping policy → StrokePatchD; reference/generated
   B features → contextual comparison.
4. (d) Testing with random styles: sampled prior + text → Generator → image.
5. (e) Testing with reference-guided styles: reference → B → E → G → image,
   with independently requested target text.

Each page has one connected diagram component, with **zero isolated model or
loss blocks**. Detailed operations are inside the relevant module. Explanation,
branch assignments, equations and important implementation qualifications are
plain editable text under the figure, not extra disconnected architecture nodes.

The palette, simple block treatment, solid forward arrows and dashed objective
paths follow the supplied reference. This is a source-grounded ALPHA
diagram, not a redraw of HiGAN+ with renamed modules. In particular:

- There is an explicit pretrained backbone B and a learned 1-global/7-local E.
- G uses content attention/local FFN, block-specific global CBN and one spatial
  local-style attention stage.
- Global D sees whole words; StrokePatchD alone sees crops of side H/2.
- CTC is applied only to random and transfer words; writer/context/cycle only
  to transfer; pixel L1 only to reconstruction; global latent losses retain /8.
- There is no texture generator, frequency objective, digit-specific routing or
  imported Spiral/Star/Wave module.

Normal 32/64 configs and source commit:
`7eed74c6b2d5286dc7b3e446ec1db3b165322c39` (inspected 2026-10-15).
No model, config, notebook, checkpoint or Git branch is modified.

## Verification

`build_five_panels.py` reuses the existing native XML and routing helpers, but
defines a new five-page composition. `verification.json` checks XML, geometry,
resolved connectors, no node overlaps or edge transit, and a single connected
diagram component on every page. The official draw.io codec/renderer independently
loads all five pages; its exact-file hash and native render checks are in
`native_renderer_verification.json`. The preview uses the native renderer's
inline SVG, including all text and equations, and is self-contained.

The full online editor UI is not claimed to have been tested. The verified
artifact uses its native editable mxGraph format and the official renderer.

Abstraction ledger: the prior 15-tab atlas is consolidated into these five
HiGAN+-style panels. OCR/W+B pretraining and evaluation are mentioned beneath
the relevant diagram instead of receiving separate tabs; they are not claimed
to be part of the inference network. Repeated blocks denote shared weights.
