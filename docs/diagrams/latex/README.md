# DEV architecture diagrams in TikZ/LaTeX

These sources describe the current DEV implementation only:

- `dev_full_architecture.tex`: complete HiGAN+-style overview of DEV training,
  Generator internals, global and stroke-level critics, auxiliary objectives,
  and inference paths.
- `dev_style_encoder.tex`: `StyleBackbone`, `HeavyCNNAttention`, and `StyleEncoder`.
- `dev_fusion.tex`: `StyleContentAttentionFusion`, `AllographicModulation`, and the outer fusion residual handoff. It stops before `filter_linear` and all GBlocks.
- `dev_strokepatchd.tex`: adaptive patch preparation, `StrokePatchBlock`, `PatchDiscriminator`, and the patch-only adversarial objective. It does not include global D.
- `dev_strokepatchd_mechanism.tex`: concrete 64-pixel-high word example showing
  the 32x32 adaptive crop geometry, matched five-source sampling, local evidence,
  StrokePatchD forward path, and the explicit patch-vs-global-D routing rule.

All source code uses free, standard TikZ/LaTeX packages. No Mermaid or hosted diagram service is required.

## Compile

Run from this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error dev_full_architecture.tex
pdflatex -interaction=nonstopmode -halt-on-error dev_style_encoder.tex
pdflatex -interaction=nonstopmode -halt-on-error dev_fusion.tex
pdflatex -interaction=nonstopmode -halt-on-error dev_strokepatchd.tex
pdflatex -interaction=nonstopmode -halt-on-error dev_strokepatchd_mechanism.tex
```

The `standalone` document class crops every PDF to its diagram bounds. The shared colors, node styles, arrow styles, and typography are defined in `dev_diagram_style.tex`.

If using Overleaf, upload all four `.tex` files and choose one of the three diagram files as the main document.
