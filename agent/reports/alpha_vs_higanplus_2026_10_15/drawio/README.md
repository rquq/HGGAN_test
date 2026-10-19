# ALPHA — editable training diagrams

Open **alpha_training_architecture.drawio** in
[diagrams.net](https://app.diagrams.net/) using **File → Open From → Device**.
The file is native, uncompressed mxGraph XML. All pages are in one file; all
modules, text, loss cells, panel boundaries and connectors remain editable.
No Mermaid subscription, plugin, font download or embedded raster is needed.

## Pages

| Tab | Purpose |
| --- | --- |
| 00 Read me | Notation, frozen/trainable semantics, navigation and references |
| 01 GAN overview | Three generation branches and their objective families |
| 02 Alternating updates | D/P step versus G/E step; EMA ownership |
| 03 Global discriminator | Whole-word real/fake hinge, generator adversarial and lazy R1 |
| 04 StrokePatchD | Height-relative crop policy, critic network and logit-level hinge |
| 05 Random + latent recovery | Random-prior generation, CTC and global latent recovery |
| 06 Style transfer | CTC, writer identity, global cycle and contextual comparisons |
| 07 Reconstruction + disentanglement | Pixel target, global posterior KL and GRL content probe |
| 08 Style encoder | Frozen B, global context and multiscale local evidence retrieval |
| 09 Fusion + Generator | Content attention, local FFN, global CBN and spatial local attention |
| 10 OCR pretraining | Real-image CTC, optional EMA, validation and checkpoint export |
| 11 Writer + backbone pretraining | OCR initialization, joint B/W training, pooling and writer CE |
| 12 Inference | Prior-style and reference-guided forward generation |
| 13 Validation + checkpoints | Measurement-only metrics, epoch CSV/history and best-FID selection |
| 14 Loss ledger | Every active GAN objective, weight, branch, prediction and target |

## Snapshot and scope

- Source: `/home/quq/machineLearning/HTG/HGGAN_test/alpha`.
- Git commit: `7eed74c6b2d5286dc7b3e446ec1db3b165322c39`, inspected 2026-10-15.
- Normal `gan_iam_32.yml` / `gan_iam_64.yml` configurations; 8 total style tokens.
- Only ALPHA is depicted. MAIN's four-local-token experiment is not shown.
- This is source-based documentation, not a new accuracy or speed experiment.
- No source, config, checkpoint, notebook or Git branch was modified.

Solid connectors show forward objective dependencies. Dashed brown connectors
show a target, stop-gradient target or optimizer action. Dashed module borders
mean frozen parameters in the depicted phase—not a broken gradient path to the
input. Repeated module names denote shared parameters, not extra networks.

## Presentation references

The original architecture images in the local official implementation folders
were visually reviewed:

- [HiGAN+ implementation and overview](https://github.com/ganji15/HiGANplus):
  multi-panel training / reconstruction / inference composition.
- [SpiS-GAN paper](https://arxiv.org/abs/2607.06949): split training panels,
  repeated shared modules, distinct objective paths.
- [FW-GAN paper](https://arxiv.org/abs/2508.21040): generator/critic detail
  separated from complete training flows.

These are layout references, not claims that their Spiral/Star/Wave operations,
Sobel loss or frequency loss exist in ALPHA. Their images are not embedded
in the deliverable. The model graph is independently drawn from the local code.

## Generation and verification

`build_drawio.py` creates the native file, offline `preview.html`, the shared
`diagram_graph.json`, and `verification.json`. The checks cover:

- XML parsing; unique page/cell IDs; resolved source/target/parent and page links;
- native vertices/edges and zero embedded images;
- orthogonal connector geometry and no transit through unrelated module boxes;
- non-overlapping module boxes, page bounds and measured text-height budgets;
- panel-title avoidance during connector routing.

The preview passed the Diagram Design offline/accessibility self-check and its
15 pages were rendered in a headless, CPU-only browser for visual QA. The native
file was independently parsed with `drawio_extract.py --page all` (15 pages,
no dangling connectors). The loss ledger is a table, so its 44 editable cells
are not 44 architecture modules. Larger training pages are split into panels.

`check_editor.cjs` attempts a full load in the official web editor; network /
embedded-editor availability is distinct from the structural XML tests.
Only claim a full editor import if `editor_verification.json` records success.
QA screenshots are verification artifacts, not substitutes for editable shapes.

The complete network-dependent editor UI did not initialize in the verification
browser. As a separate stronger-than-XML check, the official draw.io viewer's
`Graph` and `mxCodec` were run offline against every page. All native vertices,
connectors and hyperlinks matched their expected counts; every connector had
rendered geometry and resolved terminals. This result is recorded in
`native_renderer_verification.json`, tied to the exact file SHA-256. It is not
misrepresented as a completed full-editor UI test.

Abstraction ledger: repeated G/B/E instances are repeated for readability;
GBlocks 2–4 and repeated convolution/residual units are grouped. Every active
loss has its own comparison definition on its flow page or loss ledger. Loader,
logging and low-level kernel bookkeeping are intentionally not architecture
nodes.
