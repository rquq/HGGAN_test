# MAIN: compact style-context and reference-fusion experiment

MAIN was synchronized from alpha and now experiments with a compact
encoder context module and simpler reference-conditioned fusion. The generator
CNN, global critic, metrics, teachers and training schedule remain unchanged,
including `num_critic_train: 2`. StrokePatchD now uses the supervision described
below. DEV and alpha remain unchanged controls.

## Context path

For a backbone sequence of shape `[batch, 256, width]`, apply two residual blocks:

1. Depthwise 7-position convolution, dilation 1.
2. LayerNorm over channels independently at each position.
3. Pointwise expansion into two 512-channel branches.
4. Gated channel mixing: `value * SiLU(gate)`.
5. Pointwise projection back to 256 channels.
6. Per-channel residual scale, initialized at 0.1, plus the incoming sequence.

The second block repeats the same operations with convolution dilation 2.
Their combined convolutional receptive field is 19 feature positions. This
retains broad word context while allowing channel interactions at both stages.
The residual skips retain the backbone's original feature magnitude/evidence;
the nonzero scales allow branch weights to learn from the first update.

Batch padding is cleared before each spatial convolution and at each block
output. Channel-only normalization does not mix statistics across valid and
padded positions. The same topology is used at both 32px and 64px.

The encoder still provides one global token and seven local tokens. Its global
pooling/MLP, three projected spatial feature maps, local query cross-attention,
evidence gate, deterministic local codes, global VAE, and content probe are
unchanged. Reference extraction remains separate from target-dependent retrieval
in the fusion stage.

## Reference-conditioned fusion

The target embedding sequence passes through three pre-normalized residual
sublayers: global-style-conditioned self-attention, reference cross-attention,
and a joint gated local feed-forward block. Both attentions use four heads and
128 total attention channels. Cross-attention projects the seven 32-dimensional
local tokens directly into K/V; it needs no second character vocabulary or
special allograph labels. A 360-channel feed-forward block uses depthwise
kernel-5 mixing after retrieval, so neighbouring target positions can interact
with the retrieved writer evidence.

Each sublayer has one direct per-channel residual scale initialized at 0.25,
without a sigmoid ceiling. The former dual routing scores, character gain,
projection gate, uniform attention mixture, modulation/RMS caps and outer G
fusion gate are removed. Input normalization and padding masks remain; padding
is cleared before the depthwise convolution as well as at sublayer outputs.
Global-token GBlock conditioning and the generator CNN are unchanged. Training
logs report actual residual scales, not sigmoid gate probabilities.

## Capacity and training

### StrokePatchD supervision

The same unconditional anisotropic residual critic processes two square crop
scales: half image height and full image height. Thus 32px words use 16/32px
crops and 64px words use 32/64px crops; there is no resolution-specific branch.
Each scale takes `ceil(valid_width / crop_size)` crops, bounded to 4--8 per word.
Horizontal crop starts are stratified and vertical starts alternate upper/lower
bands. Full-height crops retain ascenders, baseline, descenders and background
together. Short words use the existing -1 padding value outside their valid
width, without reading the batch canvas or changing dataset pixels.

Character projection and its approximate uniform-width labels/ink-confidence
thresholds are removed. The OCR, writer and content objectives still supervise
content/style; the patch critic judges realistic local appearance.

For D, compute hinge penalties per spatial score first. Average spatial scores,
then crops within each word, then the two scales, then words and the three fake
branches. For G, average raw scores in the same hierarchy and negate them.
This keeps long words, bigger score maps and extra fake branches from acquiring
larger loss weights. Real, random fake, style-transfer and reconstruction paths
use the same unwarped crop policy; global D retains symmetric DiffAug.
The existing generator patch weight stays at 0.45. Crops never enter global D.

The 64->128->192->192 trunk remains unchanged. Its directional depthwise kernels
use the largest individual filter L2 norm, the exact singular value of their
block-diagonal kernel matrix, instead of treating independent channel filters
as a dense matrix. This is kernel-matrix normalization, not an exact full spatial
convolution Lipschitz bound. Dense convolutions retain their original SN.
Removing the character embedding reduces P from 810,305 to 794,945 parameters.
Two crop scales increase compute and activation memory; this is a quality-first
supervision change, not a promised speed improvement. Old unused PatchDiscModel
options were removed from the active configs; D_ch, D_max_ch and D_layers now
make its actual capacity explicit.

Context parameters: 1,741,633 before, 794,624 now.
Complete encoder parameters, excluding the backbone: 2,338,642 before,
1,391,633 now. This is still much larger than HiGAN+'s 148,032-parameter encoder
head. The reduction removes parallel dense context branches and their separate
branch/SE gates, not the multi-scale reference representation.

Train a fresh MAIN GAN: old GAN checkpoints contain different context/fusion/P keys and
cannot strictly load into this architecture. The existing 32px/64px W+B and OCR
pretrains are unchanged and reusable. Select no GAN resume checkpoint in the
notebook when launching this experiment; no partial-load compatibility layer
has been added.

Regression and bounded training tests establish execution, finite gradients,
short-reference support, and padding invariance. They do not establish an
improvement in handwriting quality or FID/KID/HWD; that requires controlled
training and evaluation against alpha.
