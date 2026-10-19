"""Render DEV's shared forward path and exact GAN loss/target connections.

Standard-library HTML/SVG authoring. Each objective is its own small diagram
to preserve traceable input/target edges without a page-wide feedback tangle.
Verified against networks/model.py at 4753021, configs/gan_iam_64.yml.
"""
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'dev_model_loss_map'
PAPER = '#fffefa'
INK, MUTED = '#293440', '#596775'
GREEN, PINK, PURPLE, PEACH = '#e8f5ee', '#f8e7ee', '#eeebf8', '#fff0df'
parts = []


def text(x, y, value, size=20, bold=False, color=INK, anchor='start'):
    parts.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans, Arial, sans-serif" '
                 f'font-size="{size}" font-weight="{600 if bold else 400}" '
                 f'fill="{color}" text-anchor="{anchor}">{escape(value)}</text>')


def rect(x, y, w, h, fill='white', stroke='#bac3c8', dash=False):
    dash_style = ' stroke-dasharray="6 4"' if dash else ''
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dash_style}/>')


def arrow(d, dash=False):
    dash_style = ' stroke-dasharray="6 4"' if dash else ''
    parts.append(f'<path d="{d}" fill="none" stroke="{MUTED}" stroke-width="2" '
                 f'marker-end="url(#arrow)"{dash_style}/>')


def node(x, y, w, h, lines, fill, dashed=False, size=20):
    rect(x, y, w, h, fill, dash=dashed)
    gap = 28
    baseline = y + h / 2 - (len(lines) - 1) * gap / 2 + 8
    for i, value in enumerate(lines):
        text(x+w/2, baseline+i*gap, value, size, i == 0, anchor='middle')


def loss_card(x, y, title, weight, prediction, target, loss, formula, updates,
              pred_fill=GREEN, target_stop=False):
    # One independent prediction → objective ← target schematic per card.
    rect(x, y, 928, 240, PAPER, '#ccd2d5')
    text(x+24, y+36, title, 24, True)
    text(x+904, y+36, weight, 16, color=MUTED, anchor='end')
    arrow(f'M{x+416} {y+112} H{x+456}')
    arrow(f'M{x+632} {y+112} H{x+592}', target_stop)
    node(x+24, y+72, 392, 80, prediction, pred_fill)
    node(x+456, y+80, 136, 64, [loss], PEACH)
    node(x+632, y+72, 272, 80, target, '#f0f2f3', target_stop)
    text(x+24, y+188, formula, 20)
    text(x+24, y+220, updates, 16, True, '#596278')


parts.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2000 2352" '
             'width="2000" height="2352" role="img" '
             'aria-labelledby="dev-model-loss-map-title dev-model-loss-map-desc">')
parts.append('<title id="dev-model-loss-map-title">DEV architecture and training loss map</title>')
parts.append('<desc id="dev-model-loss-map-desc">A shared handwriting generator with explicit module paths, targets, loss weights, frozen teachers and gradient destinations for reconstruction, recognition, style, distribution regularization and adversarial training.</desc>')
parts.append('''<defs>
<marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="#596775"/></marker>
<marker id="arrow-accent" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="#ad6688"/></marker>
<marker id="arrow-link" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="#729783"/></marker>
</defs>''')
parts.append(f'<rect width="2000" height="2352" fill="{PAPER}"/>')
text(48, 56, 'DEV · Architecture & exact loss connections', 32, True)
text(48, 92, 'Prediction → objective ← comparison target. Each panel states which parameters receive gradients.', 20, color=MUTED)
text(1952, 56, '4753021 · GAN64 CONFIG', 16, color=MUTED, anchor='end')

text(48, 136, '01  SHARED FORWARD PATH', 20, True)
arrow('M288 248 H340')
arrow('M760 248 H872')
arrow('M1472 248 H1564')
arrow('M1172 136 V168')
text(1172, 120, 'target text t or reference text c', 16, color=MUTED, anchor='middle')
node(48, 196, 240, 104, ['Original reference x', 'text c · writer w'], '#f0f2f3')
node(340, 176, 420, 144,
     ['B frozen → E trainable', '8 style tokens: 1 global + 7 local', 'q = N(μ, diag(exp(logvar)))'], GREEN)
text(816, 220, 's ~ q', 16, color=MUTED, anchor='middle')
node(872, 168, 600, 152,
     ['Shared G', 'Text embedding → 3-stage fusion → GBlocks', '+ texture refinement → tanh'], PINK)
node(1564, 176, 388, 144,
     ['Random: F = G(z, t)', 'Transfer: S = G(s, t)', 'Reconstruction: C = G(s, c)'], PURPLE)
text(48, 364, 'x = original-width reference; r = the same word resized to the model’s per-character width.', 20)
text(48, 400, 'z ~ N(0, I) bypasses B/E. Eμ denotes deterministic re-encoding; μ is the mean of the original reference.', 20)

text(48, 456, '02  GENERATOR / ENCODER OBJECTIVES', 20, True)
text(1952, 456, 'Weights: gan_iam_64.yml · “sg” = stop-gradient', 16, color=MUTED, anchor='end')
loss_card(48, 488, 'Readability · random + transfer', 'λrand 1→0.4; λstyle 0.5→0.08',
          ['F → R   and   S → R', 'R is frozen; CTC log probabilities'],
          ['Sampled text t', 'valid text / image lengths'], 'CTC',
          'CTC(R(F), t) + CTC(R(S), t), with separate weights; no CTC on C.',
          'Updates: G for both branches; E through transfer S. R stays frozen.')
loss_card(1024, 488, 'Writer identity · transfer', 'weight 0.5',
          ['S → B → W', 'B and W are frozen'], ['Writer ID w', 'of reference x'], 'CE',
          'Cross-entropy compares writer logits to the reference writer label.',
          'Updates: G + E through S. No writer-teacher parameter updates.')

loss_card(48, 752, 'Pixel reconstruction', 'λrec 10→2.5',
          ['C = G(s, c)', 'same writer, same reference text'],
          ['Real image r', 'width-normalized target'], 'L1',
          'Mean absolute pixel error over valid widths; padded tails are masked.',
          'Updates: G + E through encoded style s.', pred_fill=PINK)
loss_card(1024, 752, 'Contextual style · transfer', 'weight 0.1',
          ['S → shared backbone B', 'multi-scale features Bℓ(S)'],
          ['Original x → B', 'features Bℓ(x)'], 'CX',
          'Sumℓ CX(Bℓ(x), Bℓ(S)); valid-width feature matching, not pixel L1.',
          'Updates: G + E through S. B is frozen; reference features are constant.')

loss_card(48, 1016, 'Random latent recovery', 'weight 1.0',
          ['F → B → Eμ', 're-encode the random-style image'],
          ['sg(z)', 'original random tokens'], 'L1',
          'mean |Eμ(B(F)) − sg(z)|',
          'Updates: G + E in the re-encoding path. B stays frozen.',
          pred_fill=PURPLE, target_stop=True)
loss_card(1024, 1016, 'Transfer style cycle', 'weight 1.0',
          ['S → B → Eμ', 're-encode the transferred image'],
          ['sg(μ)', 'reference posterior mean'], 'L1',
          'mean |Eμ(B(S)) − sg(μ)|; the target is μ, not the sampled s.',
          'Updates: G + E (source and re-encoding). Only the target μ is stopped.',
          pred_fill=PURPLE, target_stop=True)

loss_card(48, 1280, 'Style prior regularization', 'weight 0.1',
          ['x → B → E: μ, logvar', 'encoded posterior q(s | x)'],
          ['N(0, I)', 'standard normal prior'], 'KL',
          '½ mean[μ² + exp(logvar) − 1 − logvar], averaged over tokens/dims.',
          'Updates: E only. This loss does not pass through G.', pred_fill=PURPLE)
loss_card(1024, 1280, 'Content disentanglement', 'weight 0.02',
          ['μ → GRL → probe Q', 'Q is a trainable head inside E'],
          ['Multi-hot glyphs of c', 'blank class excluded'], 'BCE',
          'Weighted BCE: Q learns glyph presence; GRL reverses its gradient to E.',
          'Updates: Q normally, E representation in reverse. No G path.', pred_fill=PURPLE)

text(48, 1576, '03  ADVERSARIAL OBJECTIVES · DIFFERENT D/P AND G UPDATE PHASES', 20, True)


def critic_card(x, title, real, fake, score, df, gf, note):
    y = 1616
    rect(x, y, 600, 380, PAPER, '#ccd2d5')
    text(x+24, y+36, title, 24, True)
    # Two score paths enter their shared hinge objective at separate ports.
    arrow(f'M{x+156} {y+144} V{y+160} Q{x+156} {y+168} {x+164} {y+168} H{x+204} Q{x+212} {y+168} {x+212} {y+176} V{y+192}')
    arrow(f'M{x+444} {y+144} V{y+160} Q{x+444} {y+168} {x+436} {y+168} H{x+396} Q{x+388} {y+168} {x+388} {y+176} V{y+192}')
    node(x+24, y+64, 264, 80, real, GREEN, size=16)
    node(x+312, y+64, 264, 80, fake, '#f0f2f3', dashed=True, size=16)
    node(x+116, y+192, 368, 56, ['Hinge: real ≥ +1, fake ≤ −1'], PEACH, size=16)
    text(x+24, y+280, df, 20)
    text(x+24, y+312, gf, 20, True)
    text(x+24, y+348, score, 16, color=MUTED)
    text(x+24, y+372, note, 16, color=MUTED)


critic_card(48, 'Global discriminator D',
            ['Real r → A → D', 'real score a'],
            ['sg(F, S, C) → A → D', 'fake scores b'],
            'D step: update D only. G step: gradients through frozen D.',
            'L_D = mean[1 − a]₊ + mean[1 + b]₊',
            'G step: Ladv,D = −mean D(A(F, S, C))',
            'A = word-safe DiffAug; local crops never enter global D.')
critic_card(700, 'StrokePatchD · P',
            ['r and A(r) → crop → P', 'real patch score a'],
            ['sg(F, S, C) → crop → P', 'fake patch scores b'],
            'P step: update P only. G step: 0.45 × Ladv,P.',
            'L_P = mean[1 − a]₊ + mean[1 + b]₊',
            'G step: Ladv,P = −mean P(crops(F, S, C))',
            '32×32; 4–8 / image; character IDs + soft confidence.')

x, y = 1352, 1616
rect(x, y, 600, 380, PAPER, '#ccd2d5')
text(x+24, y+36, 'R1 · global D only', 24, True)
arrow(f'M{x+300} {y+144} V{y+192}')
node(x+104, y+64, 392, 80, ['Real r → A → D', 'differentiate the score with respect to r'], GREEN, size=16)
node(x+52, y+192, 496, 56, ['R1 = ½ mean ‖∇r D(A(r))‖²'], PEACH, size=20)
text(x+24, y+280, 'Penalizes input gradients toward zero.', 20)
text(x+24, y+312, 'Applied every 32 steps: 0.01 × 32 × R1', 20, True)
text(x+24, y+348, 'Updates: D only; no R1 on P or G.', 16, color=MUTED)
text(x+24, y+372, 'A regularizer, not an image-to-image comparison.', 16, color=MUTED)

rect(48, 2032, 1904, 160, '#f3f4f5', '#c6cdd2')
text(72, 2068, 'HOW THE OBJECTIVES COMBINE', 20, True)
text(72, 2104, 'G/E step: Ladv,D + 0.45 Ladv,P + λrand Lctc,F + λstyle Lctc,S + 0.5 Lwriter + λrec Lrec', 20)
text(72, 2136, '                 + Llatent + Lcycle + 0.1 Lcontext + 0.1 LKL + 0.02 Lprobe (with GRL).', 20)
text(72, 2168, 'D/P step: L_D + L_P + scheduled R1. Fake images are detached. G/E optimizer runs every 2 D/P batches.', 20)

text(48, 2228, 'Frozen D/P pass gradients to G for F/S/C, and to E through S/C. B/R/W are frozen but remain differentiable.', 20)
text(48, 2264, 'Linear weight transitions: CTC random epochs 10–20; CTC transfer 8–28; reconstruction 24–42.', 20, color=MUTED)
text(48, 2300, 'B: backbone · E: style encoder · R: OCR · W: writer classifier · Q: content probe · [u]₊ = max(u, 0)', 16, color=MUTED)
text(48, 2332, 'Each generated family is averaged equally in adversarial losses; real P averages clean and augmented groups. FID/KID/HWD are evaluation metrics, not training losses.', 16, color=MUTED)
parts.append('</svg>')
svg = '\n'.join(parts)
OUT.with_suffix('.svg').write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + svg, encoding='utf-8')
html = '''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>DEV architecture and exact loss map</title>
<style>body{margin:0;background:#fffefa;color:#293440;font-family:Arial,sans-serif}main{max-width:2000px;margin:auto}svg{display:block;width:100%;height:auto}details{margin:24px 3%;line-height:1.6}a{color:#5b4b84}</style></head><body><main>''' + svg + '''
<details><summary>Implementation details and source evidence</summary>
<p>The diagram follows GlobalLocalAdversarialModel.train in networks/model.py, loss definitions in networks/loss.py, StyleEncoder in networks/module.py and configs/gan_iam_64.yml. All weights and schedules shown are from the current DEV GAN64 configuration. The overview figure remains available separately.</p>
<p>Real reference x is batch.org_imgs; real reconstruction/discriminator target r is batch.style_imgs, rescaled to char_width × label_length. E uses original-width x. The encoder’s posterior has 8 tokens of 32 dimensions; sampled s is used for transfer/reconstruction while its mean μ is the detached style-cycle target. Re-encoding uses E in deterministic, non-VAE mode.</p>
<p>CTC uses frozen R with zero_infinity=True and mean reduction. It receives random/transfer images directly in the internal model polarity, not inverted displays. Writer CE is only on transfer. W uses B’s already-computed transfer features. Contextual loss compares width-masked B features at selected scales, with valid-width clipping; E projections are not part of these feature tensors.</p>
<p>The content probe Q operates on every style token and predicts the reference transcription’s multi-hot character presence, excluding blank. Its BCE uses a positive-class weight from batch prevalence, capped at 10. Q minimizes this objective while gradient reversal makes the upstream encoder oppose the same prediction. Q parameters belong to E and are optimized by the G/E optimizer.</p>
<p>For global D, fake loss is an equal average over F/S/C and real loss uses A(r). For P, fake groups F/S/C are averaged equally; real groups r and A(r) are averaged equally. P uses character-aligned approximate crops, optional patch masking, character IDs and confidence. The 0.45 patch coefficient scales G’s patch objective, not P’s own hinge objective. Low confidence attenuates only character projection, not the unconditional critic score.</p>
<p>During the D/P phase, generated images are detached. During the G/E phase, D, P, R, W and B weights are frozen, but their image derivatives remain active. The style-cycle target μ is stopped; gradients still reach E via its source-style path through S and via re-encoding. KL and content-probe losses only update E/Q and have no G path. No Gram or contrastive loss contributes to the current GAN total.</p>
<p>CTC random weight interpolates 1.0→0.4 over epochs 10–20; CTC transfer 0.5→0.08 over 8–28; reconstruction 10→2.5 over 24–42. Other weights shown remain fixed. R1 is multiplied by lambda_r1 × interval to compensate its every-32-step application. The native R1 function includes the ½ factor.</p>
<p>Source snapshot: DEV 4753021, checked 11 September 2026. Pure documentation; no model or training changes. Typeface: offline DejaVu Sans / Arial.</p>
</details></main></body></html>'''
OUT.with_suffix('.html').write_text(html, encoding='utf-8')
print(OUT.with_suffix('.html'))
print(OUT.with_suffix('.svg'))
