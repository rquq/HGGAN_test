# MAIN diagram refresh — 2026-11-15

Authoritative file: `HGGAN_test/handwritten.drawio`.
Code target: MAIN `d881b59fb2111f92b0a8a580cedd4ad1e6438672`, current IAM32
and IAM64 configurations. This documents the model; it does not modify it.

## Six updated tabs

| Tab | Current content |
|---|---|
| (a) GAN training | Reference encoding, random/transfer/same-text branches, D/P hinge, R CTC, W identity and branch-specific losses |
| (b) Same-text and latent losses | Global posterior128D, VAE, GRL content probe, random-latent information loss, KL; same-text remains a GAN fake while pixel L1 is off |
| (c) Patch and distribution losses | Spatial patch hinge and transfer-only frozen-B multiscale Sliced Wasserstein; actual targets and reductions |
| (d) Modules vs HiGAN+ | Original local HiGAN+ source versus MAIN encoder, generator conditioning, patch critic, writer readout, global-D readout and backbone geometry |
| (e) Overlap-grid PatchD | H/2 squares, H/4 stride, three rows at every column, paper-filled right remainder and equal-word/branch reduction |
| (f) Inference | Reference→B→E→one128D global mean; target-only content seed; four nonlinear BSSP paths into four GBlocks; separate random-prior mode |

All six original page IDs/order are retained. The minimal palette, rounded boxes,
paired comparison layout and broad end-to-end flow language are preserved.
Obsolete internal boxes/wires were replaced, not merely relabeled.

## Fidelity corrections

- Removed the nonexistent memory M, local style tokens, learned style queries,
  reference K/V attention and spatial style retrieval. MAIN now has one128D
  global code, not the old32D global plus local/memory design.
- Showed GatedStyleContext's two sequential DWk7/gated residual blocks, valid
  pooling/LN and the global posterior; content self-attention remains distinct
  from this style path.
- Showed the content-only seed and four independent nonlinear global projections
  into CBN. At native32, GBlock4 refines without upsampling; at native64 it
  performs the final2× upsample.
- Corrected CTC to random/transfer only, writer CE to transfer only, Info to
  re-encoded random generation versus detached prior, and content BCE to the
  reference global mean through gradient reversal. Kept the explicit1/8 style
  normalization and actual current coefficients.
- Replaced obsolete cycle/CX/Gram/frequency/texture objectives with the actual
  multiscale SW objective. Its targets are detached real-reference B maps,
  compared with transfer B maps; this is not an Inception/KID training loss.
- Corrected the old two-row/non-overlap and stratified crop claims. For H32/W64,
  s16/stride8 gives three rows×seven columns=21 crops.
- Removed obsolete GAN CER/WER evaluation claims. Current enabled metrics are
  EMA-guided same-reference-text FID/KID/HWD on the configured test split.
  Guided inference uses the posterior mean; random-prior generation is separate.

## Verification and preservation

Original SHA256:
`c32236438be06654fee7a937273b298f1619189ff49c94e90dca6bab4622aaca`.
Promoted authoritative SHA256:
`698749c4f9396028ec06b693dafbcb6c305a77fade5c9ab8af1e42f21802b328`.
The generated candidate differs only by the source editor's final newline.

Structural checks: six pages, original IDs/order, no duplicate IDs/dangling
edges and orthogonal connectors. All six pages were rendered with the existing
official draw.io/mxGraph bundle in a headless browser, without network loading,
then visually reviewed. Clipped legend/real-image labels were enlarged; the
detached critic-step wire was given a separate source port from the G-step
wire. Final render checks cover text clipping/overflow, dangling/unrendered
edges and transit through unrelated blocks.

The old file is preserved at
`agent/analysis/2026-11-15-main-diagram/handwritten.before.drawio`.
Native PNG/SVG previews, renderer report, source mapping and edit evidence are
local analysis artifacts in that directory. The editable `.drawio` is the
intended deliverable; generated scripts/raster checks are not model-source files.

Rendering is not a model-quality test or proof of a visible desktop-editor tab.
No connected document session was present. A Windows-backed temporary renderer
entry was used after Node's UNC ancestor-package lookup failed; the repository
and authoritative diagram stayed in WSL. Interoperability diagnostics passed;
no global settings, file associations or permission profile were changed.
