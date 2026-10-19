"""Build a compact, editable overview of DEV at commit 4753021.

Run with Python 3. Standard library only; outputs HTML and standalone SVG.
Pastel groups follow the user's HiGAN+ reference; repeated training paths and
per-layer internals are intentionally collapsed. No model code is changed.
"""
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'dev_model_overview'
PAPER, INK, MUTED = '#fffefa', '#293440', '#596775'
STYLE, PINK, VIOLET, PEACH = '#e8f5ee', '#f8e7ee', '#eeebf8', '#fff0df'
svg = []


def text(x, y, value, size=20, weight=400, color=INK, anchor='start', family='DejaVu Sans, Arial, sans-serif'):
    svg.append(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
               f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(value)}</text>')


def rect(x, y, w, h, fill=PAPER, stroke='#b7c0c5', radius=8):
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
               f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')


def path(d, color=MUTED, dashed=False):
    dash = ' stroke-dasharray="7 5"' if dashed else ''
    svg.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" '
               f'marker-end="url(#arrow)"{dash}/>')


def label(x, y, value, width):
    svg.append(f'<rect x="{x}" y="{y-20}" width="{width}" height="24" fill="{PAPER}"/>')
    text(x + width / 2, y, value, 16, color=MUTED, anchor='middle')


svg.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 1040" '
           'width="1600" height="1040" role="img" '
           'aria-labelledby="dev-model-overview-title dev-model-overview-desc">')
svg.append('<title id="dev-model-overview-title">DEV handwriting generation architecture</title>')
svg.append('<desc id="dev-model-overview-desc">Style encoding and target text feed three-stage fusion and a GBlock generator with texture refinement, supervised during training by discriminators, frozen recognition and writer teachers, and reconstruction and style regularization.</desc>')
svg.append('''<defs>
<marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="#596775"/></marker>
<marker id="arrow-accent" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="#ad6688"/></marker>
<marker id="arrow-link" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="#6b8b79"/></marker>
</defs>''')
svg.append(f'<rect width="1600" height="1040" fill="{PAPER}"/>')
text(48, 56, 'DEV · Handwriting generation', 32, 600)
text(48, 92, 'One shared generator for random style, style transfer and reconstruction', 20, color=MUTED)
text(1552, 56, 'CURRENT CODE · 4753021', 16, color=MUTED, anchor='end')
text(48, 140, 'GENERATION', 16, 600, MUTED)

# Data paths: distinct ports; all elbows rounded and routed in open canvas.
path('M264 224 H780 Q788 224 788 232 V280')
path('M264 424 H320')
path('M580 424 H640')
path('M940 400 H1008')
path('M1288 400 H1356')
# Style also conditions GBlocks and output texture, beyond character fusion.
path('M520 520 V552 Q520 560 528 560 H1136 Q1144 560 1144 552 V488')
label(660, 208, 'Text embedding + IDs', 256)
label(592, 396, 'z', 32)
label(960, 376, 'fused', 48)
label(656, 548, 'z → block conditioning + texture statistics', 464)

# Three downstream bundles represent training-only consumers, with no repeated G.
path('M1392 488 V640 Q1392 648 1384 648 H272 Q264 648 264 656 V720', dashed=True)
path('M1456 488 V664 Q1456 672 1448 672 H808 Q800 672 800 680 V720', dashed=True)
path('M1520 488 V720', dashed=True)

rect(48, 176, 216, 96, '#f0f2f3')
text(156, 212, 'Target text', 24, 600, anchor='middle')
text(156, 248, '“handwriting”', 20, anchor='middle')

rect(48, 360, 216, 128, '#f0f2f3')
text(156, 400, 'Style reference', 20, 600, anchor='middle')
text(156, 432, 'word / letter / “.”', 20, anchor='middle')
text(156, 464, 'real image x', 16, color=MUTED, anchor='middle')

rect(320, 328, 260, 192, STYLE, '#729783')
text(450, 364, 'Style encoder', 24, 600, anchor='middle')
text(450, 396, 'B frozen · E trainable', 16, color=MUTED, anchor='middle')
text(450, 432, 'Multi-scale features', 20, anchor='middle')
text(450, 464, '1 global + 7 local tokens', 16, 600, anchor='middle')
text(450, 496, 'Short-reference stabilization', 16, color=MUTED, anchor='middle')

rect(640, 280, 300, 224, PINK, '#ad6688')
text(790, 320, 'Content–style fusion', 24, 600, anchor='middle')
text(660, 356, '1', 20, 600, '#ad6688')
text(692, 352, 'Style-conditioned', 20)
text(692, 376, 'self-attention', 20)
text(660, 412, '2', 20, 600, '#ad6688')
text(692, 412, 'Local depthwise mixing', 16)
text(660, 452, '3', 20, 600, '#ad6688')
text(692, 452, 'Character-aware', 20)
text(692, 476, 'allograph refinement', 20)

rect(1008, 312, 280, 176, PINK, '#ad6688')
text(1148, 352, 'Generator G', 24, 600, anchor='middle')
text(1148, 392, 'Conditioned GBlocks', 20, anchor='middle')
text(1148, 428, '+ texture refinement', 20, anchor='middle')
text(1148, 464, 'Output head + tanh', 16, color=MUTED, anchor='middle')

rect(1356, 344, 196, 144, '#f0f2f3')
text(1454, 380, 'Generated images', 16, 600, anchor='middle')
text(1454, 416, 'Random · Transfer', 16, anchor='middle')
text(1454, 448, 'Reconstruction', 16, anchor='middle')

# The random path substitutes normal tokens; same G and same conditioning routes.
text(48, 608, 'Style modes:', 20, 600)
text(200, 608, 'encoded z = E(B(x))   or   random z ~ N(0, I)', 20)
text(788, 608, 'Reconstruction uses the reference text.', 20, color=MUTED)

text(48, 696, 'TRAINING ONLY', 16, 600, MUTED)
rect(48, 720, 432, 216, PEACH, '#c1a07b')
text(72, 756, 'Realism · D + P', 24, 600)
text(72, 792, 'D: whole words + real images', 20)
text(72, 824, 'P: matched 32×32 stroke crops', 20)
text(72, 856, '4–8 crops / image; char confidence', 16, color=MUTED)
text(72, 904, 'Hinge adversarial + R1 on D', 20, 600)

rect(584, 720, 432, 216, STYLE, '#729783')
text(608, 756, 'Content & writer · frozen teachers', 20, 600)
text(608, 792, 'R: CTC → target text', 20)
text(608, 824, 'B + W: writer CE → writer identity', 20)
text(608, 856, 'B features: contextual style match', 20)
text(608, 904, 'CTC + writer CE + contextual loss', 16, 600)

rect(1120, 720, 432, 216, VIOLET, '#9186b1')
text(1144, 756, 'Reconstruction & regularization', 20, 600)
text(1144, 792, 'Image L1: reconstruction vs. real', 20)
text(1144, 824, 'E reuse: latent L1 + style cycle', 20)
text(1144, 856, 'KL on encoded style', 20)
text(1144, 904, 'Content-adversarial probe (GRL)', 16, 600)

svg.append('<line x1="48" y1="964" x2="1552" y2="964" stroke="#cbd1d4"/>')
svg.append('<path d="M48 992 H96" stroke="#596775" stroke-width="2" fill="none"/>')
text(108, 1000, 'Generation', 16, color=MUTED)
svg.append('<path d="M256 992 H304" stroke="#596775" stroke-width="2" stroke-dasharray="7 5" fill="none"/>')
text(316, 1000, 'Training supervision', 16, color=MUTED)
text(608, 1000, 'Fusion belongs to G; B is shared. 64 px shown (32 px uses 16×16 crops).', 16, color=MUTED)
svg.append('</svg>')
markup = '\n'.join(svg)
OUT.with_suffix('.svg').write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + markup, encoding='utf-8')
html = '''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DEV · Handwriting generation</title>
<style>*{box-sizing:border-box}body{margin:0;background:#fffefa;color:#293440;font-family:Arial,sans-serif}main{max-width:1600px;margin:auto}svg{display:block;width:100%;height:auto}details{margin:12px 3%;font-size:14px;line-height:1.6}a{color:#5b4b84}</style>
</head><body><main>''' + markup + '''
<details><summary>Scope and source</summary>
<p>Verified against DEV commit 4753021, 11 September 2026. Visual grouping follows the supplied HiGAN+ reference. Generation paths are collapsed into one shared G; loss targets are grouped rather than drawn as repeated feedback loops. The random mode bypasses B/E and supplies normal style tokens to the same fusion and decoder paths. Fusion is implemented inside Generator.</p>
<p>R applies CTC to random and transfer images. W and contextual matching supervise transfer images. Image L1 supervises reconstruction. E is reused on random/transfer images for latent recovery and style cycle. KL and the content-adversarial probe act on encoded reference style, not on generated image pixels; the third dashed connector groups these related training objectives. B, R and W are frozen during GAN training. D/P use real as well as generated images.</p>
<p>Sources: networks/BigGAN_networks.py, networks/texture_generator.py, networks/fusion.py, networks/module.py, networks/model.py and configs/gan_iam_64.yml. No training code or model weights were changed. This overview is not a claim of measured improvement.</p>
<p>The figure uses offline DejaVu Sans / Arial for portable rendering.</p>
</details></main></body></html>'''
OUT.with_suffix('.html').write_text(html, encoding='utf-8')
print(OUT.with_suffix('.html'))
print(OUT.with_suffix('.svg'))
