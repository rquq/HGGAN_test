# ALPHA incremental experiment: target-text context — 2026-11-09

Source published and remote-verified: `14f6eea61e892343ec958912529c5f25cf34491a`.

User approval: add the proposed MAIN target-content contextualization to
ALPHA, without adding reference memory or other MAIN model changes.

## Design and narrow implementation plan

Reuse MAIN's existing `ContentContext` implementation rather than introducing
a new attention mechanism. Target character embeddings pass through one
relative-position self-attention sublayer and a local depthwise gated
feed-forward sublayer before ALPHA's existing style concatenation and spatial
seed. Initial learnable residual scales are 0.25, attention dimension 128,
four heads, and feed-forward width three times the embedding dimension.

Preserve the style encoder/backbone, 32D global style vector, four GBlocks,
style-linear conditioning, discriminator/PatchD, all losses, training and
evaluation settings. The module consumes target text and lengths only, not
reference image content, writer IDs or glyph-category rules.

Acceptance: native 32/64 outputs have unchanged geometry; padding in target
text cannot affect valid context tokens; context parameters receive finite
gradients; a real GAN update with frozen teachers and the added module runs;
existing non-context parameter names/shapes are unchanged. No KID/FID gain
is claimed without training.

This adds generator parameters, so old ALPHA GAN checkpoints do not strict-load
into the new experiment. Use a fresh GAN run; existing W+B and R teachers
remain compatible. Do not silently ignore missing state during resume.

## Implementation and measured checks

Copied MAIN's target-only `ContentContext` into `alpha/networks/fusion.py`
and inserted it between `text_embedding` and the original style/text
concatenation in G. No reference keys/values, local-query style branch,
new losses or teacher changes were added. Constructor defaults select the
same four-head/128D context design as MAIN at both image heights. The four
IAM GAN YAMLs and the custom-image YAML explicitly record the context
dimensions/residual initialization for future checkpoint provenance.

At 32px, G increases from 2,996,025 to 3,190,557 parameters: +194,532
(about 6.5%). Every old non-context state key and tensor shape is unchanged.
Context is constructed after initializing the baseline modules, inside an
isolated CPU RNG scope. A paired-seed test verifies exact equality of every
old shared parameter/buffer and the post-construction global RNG state.
This avoids confounding the ablation by changing the initial shared weights
or the next encoder/discriminator initialization draws.

Adding these new state keys is nevertheless a deliberate GAN-checkpoint
incompatibility; no permissive loading shim was added.

Passed: regression first failed because the context module was absent,
then two tests passed native 32/64 geometry, finite backward/context gradients and
valid-token context invariance under extra batch padding. The actual ALPHA
GAN D+G update/teacher-loading/save/reload CPU smoke passed at both heights.
The applicable nine ALPHA bug-audit tests also pass (12 discovered, three
MAIN/DEV-specific tests skipped).

No trained FID/KID improvement or T4 speed/memory improvement is established.
The pre-existing full legacy suite targets several retired model versions;
it is not a green acceptance suite for this new architectural ablation.
Final legacy discovery reports 51 tests, 16 failures and 38 errors including
subtests; see `2026-11-09-verification-inventory.md` for the full inventory.
