"""Three connected DEV training flows; repeated boxes share model weights.

Documentation only. Source: model.py and gan_iam_64.yml at DEV 4753021.
Run with Python 3 to rebuild the self-contained HTML and vector SVG.
"""
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'dev_three_training_flows'
W, H = 2304, 2664
PAPER, INK, MUTED = '#fffefa', '#293440', '#596775'
GREEN, PINK, PURPLE, PEACH = '#e8f5ee', '#f8e7ee', '#eeebf8', '#fff0df'
FEEDBACK = '#93617b'
layers = {key: [] for key in ('zones', 'edges', 'labels', 'nodes')}


def text(x, y, value, size=20, bold=False, color=INK, anchor='start', layer='nodes'):
    layers[layer].append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans, Arial, sans-serif" '
                         f'font-size="{size}" font-weight="{600 if bold else 400}" '
                         f'fill="{color}" text-anchor="{anchor}">{escape(value)}</text>')


def rect(x, y, w, h, fill, layer='nodes', stroke='#bdc8ce'):
    layers[layer].append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
                         f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')


def node(x, y, w, h, title, lines=(), fill=GREEN, small=20):
    rect(x, y, w, h, fill)
    gap = 28 if small == 20 else 24
    baseline = y + h / 2 - len(lines) * gap / 2 + 8
    text(x+w/2, baseline, title, 20, True, anchor='middle')
    for n, line in enumerate(lines, 1):
        text(x+w/2, baseline+n*gap, line, small, anchor='middle')


def edge(points, feedback=False, arrow=True):
    """Rounded orthogonal path; separate explicit branch junctions where needed."""
    for a, b in zip(points, points[1:]):
        assert a[0] == b[0] or a[1] == b[1], (a, b)
    d = f'M{points[0][0]} {points[0][1]}'
    for i in range(1, len(points)-1):
        a, b, c = points[i-1], points[i], points[i+1]
        length1 = abs(a[0]-b[0])+abs(a[1]-b[1])
        length2 = abs(c[0]-b[0])+abs(c[1]-b[1])
        radius = min(8, length1/2, length2/2)
        before = (b[0]+(a[0]-b[0])*radius/length1,
                  b[1]+(a[1]-b[1])*radius/length1)
        after = (b[0]+(c[0]-b[0])*radius/length2,
                 b[1]+(c[1]-b[1])*radius/length2)
        d += f' L{before[0]} {before[1]} Q{b[0]} {b[1]} {after[0]} {after[1]}'
    d += f' L{points[-1][0]} {points[-1][1]}'
    path(d, feedback, arrow)


def path(d, feedback=False, arrow=True):
    color = FEEDBACK if feedback else MUTED
    extra = ' stroke-dasharray="8 6"' if feedback else ''
    if arrow:
        extra += f' marker-end="url(#{"arrow-accent" if feedback else "arrow"})"'
    layers['edges'].append(f'<path d="{d}" fill="none" stroke="{color}" '
                           f'stroke-width="2.4"{extra}/>')


def label(x, y, value, color=MUTED):
    width = ((len(value)*10 + 24)//4)*4
    rect(x-width/2, y-20, width, 28, PAPER, 'labels', 'none')
    text(x, y, value, 16, color=color, anchor='middle', layer='labels')


def zone(y, h, number, title, subtitle):
    rect(36, y, 2232, h, '#fdfcf8', 'zones', '#cdd3d6')
    text(68, y+40, number, 24, True)
    text(124, y+40, title, 28, True)
    text(68, y+72, subtitle, 16, color=MUTED)


def adversarial(y, patch=False):
    name = 'P' if patch else 'D'
    title = 'StrokePatchD · local stroke realism' if patch else 'Global D · whole-word realism'
    zone(y, 512, '02' if patch else '01', title,
         'Read left → right; upper route is real, lower route is generated. The two critic boxes are the same network.')
    # Shared forward route, repeated intentionally in both sections.
    node(76, y+280, 208, 112, 'Reference x', ['text c · writer w'], '#f0f2f3')
    node(340, y+280, 232, 112, 'B → E', ['B frozen', 'style sample s'], GREEN)
    node(636, y+280, 336, 112, 'G', ['fusion → GBlocks', 'texture refinement → tanh'], PINK)
    node(636, y+112, 336, 88, 'Text + random style', ['t, c and z ~ N(0, I)'], '#f0f2f3')
    edge([(804, y+200), (804, y+280)])
    label(804, y+240, 'CONDITIONING')
    node(1032, y+280, 224, 112, 'Generated images', ['F = G(z, t)', 'S = G(s, t); C = G(s, c)'], PURPLE, 16)
    for a, b in ((284,340), (572,636), (972,1032)):
        edge([(a,y+336), (b,y+336)])
    node(1032, y+112, 224, 88, 'Real image r', ['width-normalized x'], '#f0f2f3', 16)
    for yy in (y+156, y+336):
        edge([(1256,yy), (1308,yy)])
        edge([(1576,yy), (1632,yy)])
    if patch:
        node(1308,y+112,268,88,'Real patches',['crop r and A(r)'],GREEN)
        node(1308,y+280,268,112,'Fake patches',['same crop policy','crop F, S and C'],GREEN)
        real_sub, fake_sub = ['real patch scores'], ['fake patch scores']
    else:
        node(1308,y+112,268,88,'Word-safe DiffAug A',['A(r) · full word'],GREEN)
        node(1308,y+280,268,112,'Word-safe DiffAug A',['A(F), A(S), A(C)','full words, never crops'],GREEN,16)
        real_sub, fake_sub = ['real score a'], ['fake scores b']
    node(1632,y+112,160,88,name,real_sub,GREEN,16)
    node(1632,y+280,160,112,name,fake_sub,GREEN,16)
    # Different score ports feed critic and generator objectives.
    edge([(1792,y+156),(1872,y+156)])
    edge([(1792,y+308),(1824,y+308),(1824,y+236),(1872,y+236)])
    edge([(1792,y+364),(1836,y+364),(1836,y+352),(1872,y+352)])
    dlines = ['hinge(real, fake)', 'real ≥ +1; fake ≤ −1', 'no R1 on P'] if patch else [
        'hinge(real, fake) + R1', 'real ≥ +1; fake ≤ −1', 'R1: 0.01 × 32, every 32 steps']
    node(1872,y+108,344,156,f'{name} objective',dlines,PEACH,16)
    node(1872,y+300,344,104,'G/E objective',
         ['−mean fake scores', 'weight = 0.45' if patch else 'weight = 1'],PEACH)
    # Local feedback loops stay inside the section.
    edge([(2080,y+108),(2080,y+92),(1712,y+92),(1712,y+112)],True)
    label(1900,y+80,f'UPDATE {name}',FEEDBACK)
    edge([(2040,y+404),(2040,y+448),(804,y+448),(804,y+392)],True)
    label(1400,y+428,'BACKPROP TO G; TO E THROUGH S/C',FEEDBACK)
    text(68,y+480,
         'P step: detach F/S/C; update only P.  G step: freeze P weights, keep image gradients.  Each fake family has equal weight.' if patch else
         'D step: detach F/S/C; update only D.  G step: freeze D weights, keep image gradients.  B stays frozen in both phases.',16,color=MUTED)
    text(68,y+504,
         'GAN64: 32×32 patches; ceil(valid width / 32), clamped to 4–8. Character IDs + soft confidence condition P. Real groups r / A(r) are averaged equally.' if patch else
         'Hinge = mean[1 − a]₊ + mean[1 + b]₊.  R1 = ½ mean ‖∇r D(A(r))‖²; global D only.  Losses compare scores to margins, not paired real/fake pixels.',16,color=MUTED)


text(48,56,'DEV · Three connected training flows',40,True)
text(48,96,'Follow the forward arrows into each loss, then follow the dashed feedback to the updated modules.',24,color=MUTED)
text(48,132,'Repeated B / E / G / D / P boxes share weights. F = random; S = style transfer; C = reconstruction. Current GAN64 configuration.',20,color=MUTED)
adversarial(168)
adversarial(724, patch=True)

y=1280
zone(y,1216,'03','Reconstruction · content and style supervision',
     'One shared generator fans out to the auxiliary objectives. Each loss names its comparison target; these are NOT extra discriminators.')

# The deliberately repeated generator has one port per supervised image path.
node(76,y+412,208,136,'Reference x',['transcription c','writer ID w'],'#f0f2f3')
node(340,y+396,232,168,'B → E',['B frozen','s ~ q; μ, logvar','E trainable'],GREEN)
edge([(284,y+480),(340,y+480)])
rect(660,y+108,340,688,PINK)
text(830,y+360,'Shared G',28,True,anchor='middle')
for yy,value in [(400,'Text embedding'),(432,'↓'),(464,'Three-stage fusion'),(496,'↓'),(528,'GBlocks + texture'),(560,'↓'),(592,'Generated image')]:
    text(830,y+yy,value,20,anchor='middle')
text(830,y+680,'Inputs: s or z; text t or c',16,anchor='middle',color=MUTED)
edge([(572,y+480),(660,y+480)])
label(616,y+460,'s')
node(76,y+148,496,100,'Target text + random style',['t / c and random tokens z'],'#f0f2f3')
edge([(572,y+196),(660,y+196)])

# Loss receivers, laid out as a connected fan-out/fan-in, not isolated cards.
rows = [
    ('C',None,'Pixel reconstruction · λrec',
     ['L1(C, r) over valid pixels','Target r: width-normalized real x'], 'Updates G + E'),
    ('F, S',('OCR R',['frozen; predicts text']), 'Readability · λrand / λstyle',
     ['CTC(R(F), t) and CTC(R(S), t)','Target t: sampled transcription; no CTC on C'], 'Updates G; E through S'),
    ('F',('B → Eμ',['deterministic re-encoding']),'Random latent recovery · 1.0',
     ['L1(Eμ(B(F)), sg(z))','Target z: original random style tokens'], 'Updates G + E (re-encoder)'),
    ('S',('B → Eμ',['deterministic re-encoding']),'Transfer style cycle · 1.0',
     ['L1(Eμ(B(S)), sg(μ))','Target μ: reference posterior mean, not s'], 'Updates G + E (both paths)'),
    ('S',('B → W',['frozen writer teacher']),'Writer identity · 0.5',
     ['CE(W(B(S)), w)','Target w: the reference writer label'], 'Updates G + E through S'),
    ('S',('Backbone B',['frozen; multi-scale features']),'Contextual style · 0.1',
     ['Σℓ CX(Bℓ(x), Bℓ(S))','Target: original-reference backbone features'], 'Updates G + E through S'),
    (None,('Posterior q(s | x)',['μ, logvar from E']),'Style prior · 0.1',
     ['KL(q(s | x) || N(0, I))','Target: standard normal style prior'], 'Updates E only; no G path'),
    (None,('GRL → probe Q',['Q belongs to E']),'Content disentanglement · 0.02',
     ['weighted BCE(Q(GRL(μ)), glyphs(c))','Target: multi-hot characters in c; blank excluded'], 'Updates Q normally; E reversed'),
]
for i,(image,head,title,lines,update) in enumerate(rows):
    cy=y+156+i*120
    if image:
        if head:
            edge([(1000,cy),(1128,cy)])
        else:
            edge([(1000,cy),(1480,cy)])
        label(1064,cy-20,image)
    if head:
        node(1128,cy-40,276,80,head[0],head[1],GREEN,16)
        edge([(1404,cy),(1480,cy)])
    node(1480,cy-44,500,88,title,lines,PEACH,16)
    text(1492,cy+64,update,16,color=FEEDBACK)
    edge([(1980,cy),(2072,cy)])

# E's posterior also feeds KL/probe directly: neither path passes through G.
path(f'M456 {y+564} V{y+996}',arrow=False)
for cy in (y+876,y+996):
    layers['edges'].append(f'<circle cx="456" cy="{cy}" r="4" fill="{MUTED}"/>')
    edge([(456,cy),(1128,cy)])
label(612,y+856,'μ, logvar')
label(612,y+976,'μ')
node(76,y+660,276,156,'Same reference word',
     ['x = original-width image','r = width-normalized x','Targets stay constant.'],'#f0f2f3',16)

rect(2072,y+108,160,948,'#eeebf8')
for yy,val,size,bold in [(384,'Weighted',20,True),(416,'sum',24,True),(464,'↓',24,False),
                         (508,'G + E',24,True),(540,'update',20,True),
                         (612,'Q is in E',16,False),(680,'B / R / W',16,True),(708,'stay frozen',16,False)]:
    text(2152,y+yy,val,size,bold,anchor='middle')
# The return path hops over the two posterior-data routes rather than joining them.
path(f'M2152 {y+1056} V{y+1096} Q2152 {y+1104} 2144 {y+1104} H848 '
     f'Q840 {y+1104} 840 {y+1096} V{y+1004} a8 8 0 0 0 0 -16 '
     f'V{y+884} a8 8 0 0 0 0 -16 V{y+796}',feedback=True)
label(1320,y+1084,'G: IMAGE LOSSES ONLY',FEEDBACK)
path(f'M2192 {y+1056} V{y+1128} Q2192 {y+1136} 2184 {y+1136} H552 '
     f'Q544 {y+1136} 544 {y+1128} V{y+1004} a8 8 0 0 0 0 -16 '
     f'V{y+884} a8 8 0 0 0 0 -16 V{y+564}',feedback=True)
label(1360,y+1116,'E: IMAGE + KL; PROBE VIA GRL',FEEDBACK)
text(68,y+1156,'The same G/E optimizer also receives global adversarial loss (section 01) and 0.45 × patch adversarial loss (section 02).',20)
text(68,y+1188,'KL bypasses G. GRL reverses only the probe gradient entering E. Frozen teachers still transmit image gradients. sg(·) stops only the named target.',16,color=MUTED)

path('M48 2532 H2256',arrow=False)
edge([(52,2572),(116,2572)])
text(136,2580,'Forward data / loss value',20)
edge([(484,2572),(548,2572)],True)
text(568,2580,'Backward training signal',20)
text(1036,2580,'Weights: λrec 10→2.5; λrand 1→0.4; λstyle 0.5→0.08.',20)
text(48,2620,'Schedules: reconstruction epochs 24–42; random CTC 10–20; transfer CTC 8–28. G/E updates every 2 D/P batches. FID/KID/HWD are evaluation-only.',16,color=MUTED)
text(48,2652,'Source: DEV 4753021 · networks/model.py · configs/gan_iam_64.yml · documentation only · offline DejaVu Sans / Arial',16,color=MUTED)

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"
role="img" aria-labelledby="dev-three-training-flows-title dev-three-training-flows-desc">
<title id="dev-three-training-flows-title">DEV: three connected training flows</title>
<desc id="dev-three-training-flows-desc">Three sections trace global adversarial training, StrokePatchD training and reconstruction with auxiliary supervision, showing shared modules, explicit comparison targets and backward loss signals.</desc>
<defs>
<marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="{MUTED}"/></marker>
<marker id="arrow-accent" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="{FEEDBACK}"/></marker>
<marker id="arrow-link" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="{MUTED}"/></marker>
</defs><rect width="{W}" height="{H}" fill="{PAPER}"/>
''' + '\n'.join('\n'.join(layers[key]) for key in layers) + '</svg>'
OUT.with_suffix('.svg').write_text('<?xml version="1.0" encoding="UTF-8"?>\n'+svg,encoding='utf-8')
html='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DEV — three connected training flows</title><style>
body{margin:0;background:#fffefa;color:#293440;font-family:Arial,sans-serif}main{max-width:2304px;margin:auto}svg{display:block;width:100%;height:auto}
details{margin:24px 3%;line-height:1.7}a{color:#72516a} @media print{details{display:none}}
</style></head><body><main>'''+svg+'''
<details><summary>Notation, sharing and implementation details</summary>
<p>This is one DEV model, shown in three sections. Repeated boxes are reused calls with shared weights, not additional networks. The drawing is a logical computation graph, not a claim that every repeated B/R/G call is executed separately: the code batches generator/OCR paths and reuses backbone features.</p>
<p>x = batch.org_imgs, the original-width reference. r = batch.style_imgs, the same word rescaled to char_width × label length. c and w are its transcription and writer. t is sampled target text. E extracts eight 32-dimensional style tokens (one global and seven local); s is the posterior sample. μ is the posterior mean. z is an independent normal style sample. F=G(z,t), S=G(s,t), C=G(s,c).</p>
<p>D/P training detaches all generated images and updates only D/P. G/E training freezes D/P weights but backpropagates through their image inputs. E receives source-style adversarial gradients from S and C, not F. B, R and W remain frozen. R1 differentiates global D’s real-image score with respect to r and is applied only during D training.</p>
<p>Global and patch fake terms average the F/S/C groups equally. P’s real term equally averages clean-real and augmented-real patch groups. P uses character-aligned approximate crops, soft character confidence and the configured masking policy. The coefficient 0.45 applies to G’s patch adversarial loss; P’s own hinge loss has coefficient 1. “Matched” means the same sampling policy, not exactly paired coordinates across differently sized words.</p>
<p>CTC receives F and S directly in internal model polarity, with valid image/text lengths and zero_infinity=True. Writer CE only receives S. Contextual loss compares valid-width masked multi-scale B features of S with those of original x. Pixel L1 uses C versus normalized r, masking padded tails.</p>
<p>Both latent losses use deterministic E re-encoding. Transfer cycle stops the target μ only; E also receives gradients through S and through its re-encoder. The content probe predicts per-token multi-hot reference glyph presence, excludes blank and uses positive-class BCE weighting capped at 10. Q minimizes BCE normally; GRL reverses the gradient to upstream E. KL is ½ mean[μ²+exp(logvar)−1−logvar].</p>
<p>The G/E total is global adversarial + 0.45 patch adversarial + λrand CTC(F) + λstyle CTC(S) + λrec reconstruction + latent recovery + transfer cycle + 0.5 writer CE + 0.1 contextual + 0.1 KL + 0.02 probe BCE. The D/P total is global hinge + patch hinge + scheduled R1. No Gram or contrastive loss enters these totals.</p>
<p>Source: networks/model.py, networks/loss.py, networks/module.py, configs/gan_iam_64.yml at DEV 4753021. This revision changes documentation only.</p>
</details></main></body></html>'''
OUT.with_suffix('.html').write_text(html,encoding='utf-8')
print(OUT.with_suffix('.html'))
print(OUT.with_suffix('.svg'))
