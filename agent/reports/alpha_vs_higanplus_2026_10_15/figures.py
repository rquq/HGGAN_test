"""Deterministic vector architecture figures; no remote image service required.

SVG is generated with the standard library. The same drawing commands can be
drawn into ReportLab PDF by build_report.py, preserving editable vector text.
"""
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
import math

INK = '#263746'
BLUE = '#e8f0f8'
GREEN = '#e6f3ee'
GOLD = '#fff2d6'
RED = '#f9e9e5'
GRAY = '#f1f3f5'


@dataclass
class Diagram:
    width: int
    height: int
    commands: list = field(default_factory=list)

    def box(self, x, y, w, h, label, fill=BLUE, size=15):
        self.commands.append(('box', x, y, w, h, label, fill, size))

    def text(self, x, y, label, size=15, color=INK, anchor='middle'):
        self.commands.append(('text', x, y, label, size, color, anchor))

    def arrow(self, points, label='', color=INK, dashed=False, label_pos=None):
        self.commands.append(('arrow', points, color, dashed))
        if label:
            x, y = label_pos or points[len(points)//2]
            self.text(x, y - 8, label, 13, color)

    def svg(self):
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" role="img">',
               '<rect width="100%" height="100%" fill="white"/>']
        for command in self.commands:
            kind, *args = command
            if kind == 'box':
                x, y, w, h, label, fill, size = args
                out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill}" stroke="{INK}" stroke-width="1.4"/>')
                lines = label.split('\n')
                size = min(size, (w - 16) / (max(map(len, lines)) * 0.56))
                for i, line in enumerate(lines):
                    ty = y + h/2 + (i-(len(lines)-1)/2)*size*1.32
                    out.append(f'<text x="{x+w/2}" y="{ty}" dominant-baseline="middle" text-anchor="middle" font-family="Arial,DejaVu Sans,sans-serif" font-size="{size}" fill="{INK}">{escape(line)}</text>')
            elif kind == 'text':
                x, y, label, size, color, anchor = args
                out.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial,DejaVu Sans,sans-serif" font-size="{size}" fill="{color}">{escape(label)}</text>')
            else:
                points, color, dashed = args
                coordinates = ' '.join(f'{x},{y}' for x, y in points)
                dash = ' stroke-dasharray="6 4"' if dashed else ''
                out.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="1.6"{dash}/>')
                x, y = points[-1]
                px, py = points[-2]
                angle = math.atan2(y-py, x-px)
                base = [(x-8*math.cos(angle)+4*math.sin(angle), y-8*math.sin(angle)-4*math.cos(angle)),
                        (x-8*math.cos(angle)-4*math.sin(angle), y-8*math.sin(angle)+4*math.cos(angle))]
                out.append(f'<polygon points="{x},{y} {base[0][0]},{base[0][1]} {base[1][0]},{base[1][1]}" fill="{color}"/>')
        out.append('</svg>')
        return '\n'.join(out)


def make_figures():
    figures = {}
    d = Diagram(1340, 570)
    d.box(25, 45, 140, 65, 'Reference x\nvalid width', GRAY)
    d.box(210, 45, 140, 65, 'Backbone B\npretrained, frozen')
    d.box(395, 45, 140, 65, 'Style encoder E\nlearned in GAN')
    d.box(585, 40, 150, 65, 'Global code g\n32-D', GREEN)
    d.box(585, 150, 150, 65, '7 local codes L\n7 × 32-D', GREEN)
    d.box(805, 40, 180, 65, 'Block-specific\nstyle projections', GREEN)
    d.arrow([(165, 77), (210, 77)])
    d.arrow([(350, 77), (395, 77)])
    d.arrow([(535, 77), (585, 77)])
    d.arrow([(465, 110), (465, 182), (585, 182)])
    d.arrow([(735, 72), (805, 72)])
    d.box(25, 300, 140, 65, 'Target text y\ncharacter IDs', GRAY)
    d.box(210, 300, 140, 65, 'Embedding +\nContentContext')
    d.box(395, 300, 140, 65, 'Content-only\nspatial seed')
    d.box(585, 300, 140, 65, 'GBlock 1\nCBN(global)')
    d.box(790, 300, 170, 65, 'Spatial style\nrefinement', GREEN)
    d.box(1000, 300, 140, 65, 'GBlocks 2–4\nCBN(global)')
    d.box(1180, 300, 140, 65, 'Image x̂\nH = 32 / 64', GRAY)
    for a,b in [(165,210),(350,395),(535,585),(725,790),(960,1000),(1140,1180)]:
        d.arrow([(a,332),(b,332)])
    d.arrow([(855,105),(855,250),(655,250),(655,300)])
    d.arrow([(985,72),(1070,72),(1070,300)])
    d.arrow([(735,182),(875,182),(875,300)], 'local K,V', label_pos=(805,182))
    d.box(1180, 445, 140, 70, 'Global D\nStrokePatchD P', RED)
    d.box(790, 445, 350, 70, 'Frozen R, W+B; global re-encoder E\nCTC / writer / context / latent losses', GOLD)
    d.arrow([(1250,365),(1250,445)])
    d.arrow([(1180,345),(1160,345),(1160,480),(1140,480)])
    d.text(25, 550, 'Solid arrows: forward tensors. Loss endpoints are expanded in Figures 5–7. CBN: conditional batch normalization.', 14, anchor='start')
    figures['01_overview'] = d

    d = Diagram(1250, 650)
    d.box(25, 270, 140, 75, 'Reference image\nx, valid width', GRAY)
    d.box(205, 270, 170, 75, 'Shared frozen B\nF2 / F3 / F4')
    d.arrow([(165,307),(205,307)])
    d.box(420, 45, 180, 70, '256-D sequence\nfrom deepest B map')
    d.box(645, 45, 180, 70, 'GatedStyleContext\nDW k7, dilation 1→2')
    d.box(875, 45, 160, 70, 'Valid mean\nLayerNorm + MLP')
    d.box(1080, 45, 145, 70, 'μg, log σ²g\ng = μg + σg ε', GREEN)
    d.arrow([(290,270),(290,80),(420,80)])
    d.arrow([(600,80),(645,80)])
    d.arrow([(825,80),(875,80)])
    d.arrow([(1035,80),(1080,80)])
    d.box(420, 270, 180, 75, 'Pool to 4 height rows\n1×1 conv → 256\nper scale')
    d.box(645, 245, 180, 75, 'Normalized visual U\nK = U + 2-D PE\nV = U − valid mean', GREEN)
    d.box(645, 395, 180, 70, '7 learned queries\northogonal init', GOLD)
    d.box(875, 270, 160, 75, '4-head\ncross-attention')
    d.box(1080, 270, 145, 75, 'Bias-free\n256→32 projection')
    d.arrow([(375,307),(420,307)])
    d.arrow([(600,307),(625,307),(625,282),(645,282)])
    d.arrow([(825,282),(850,282),(850,307),(875,307)])
    d.arrow([(825,430),(955,430),(955,345)])
    d.arrow([(1035,307),(1080,307)])
    d.box(1080, 430, 145, 75, 'Center over slots\nLj ← Lj − mean(L)\n7 local tokens', GREEN)
    d.arrow([(1152,345),(1152,430)])
    d.box(420, 540, 600, 65, 'Output S = [g, L1, …, L7] ∈ R^(B×8×32)\nOnly global is stochastic / KL-regularized; local evidence is deterministic', GRAY)
    d.arrow([(1225,80),(1238,80),(1238,625),(720,625),(720,605)])
    d.arrow([(1080,467),(1055,467),(1055,572),(1020,572)])
    d.text(25, 635, 'Position selects evidence through K; positional vectors and learned queries are not added into appearance V.', 14, anchor='start')
    figures['02_style_encoder'] = d

    d = Diagram(1260, 590)
    d.box(25,45,160,65,'Text embeddings\nB × T × 120',GRAY)
    d.box(230,45,210,65,'Relative-position\nself-attention: 4 heads')
    d.box(485,45,220,65,'Gated local FFN\nDWConv1D k5; hidden 360')
    d.box(750,45,190,65,'Linear → reshape\nspatial seed')
    d.arrow([(185,77),(230,77)])
    d.arrow([(440,77),(485,77)])
    d.arrow([(705,77),(750,77)])
    d.text(445,145,'Two learned residual handoffs; initial scale 0.25, no sigmoid ceiling',14)
    d.box(750,210,190,65,'GBlock 1 feature F\nB × 256 × 8 × 4T')
    d.arrow([(845,110),(845,210)])
    d.box(25,245,160,65,'Local codes L\nB × 7 × 32',GREEN)
    d.box(230,210,210,65,'K = Wk LN(L)\nV = Wv L',GREEN)
    d.box(485,210,220,65,'Spatial queries\nQ = Wq LN(flatten F)')
    d.arrow([(185,277),(205,277),(205,242),(230,242)])
    d.arrow([(750,242),(705,242)])
    d.box(485,365,220,75,'Attention(Q,K,V)\n→ output projection\n→ residual into F',GREEN)
    d.arrow([(335,275),(335,403),(485,403)])
    d.arrow([(595,275),(595,365)])
    d.box(750,370,190,65,'GBlocks 2–4\nimage refinement')
    d.arrow([(705,403),(750,403)])
    d.box(1000,370,230,65,'Output\n32px: 32 × 16T\n64px: 64 × 32T',GRAY)
    d.arrow([(940,403),(1000,403)])
    d.box(1000,210,230,65,'Global g → BSSP\nCBN in all four blocks',GOLD)
    d.arrow([(1000,242),(940,242)])
    d.arrow([(1115,275),(1115,320),(845,320),(845,370)])
    d.text(25,520,'Content self-attention: relationships among target characters.',15,anchor='start')
    d.text(25,550,'Spatial cross-attention: image locations select reference-style evidence. They are different operations, not two copies.',15,anchor='start')
    figures['03_fusion_generator'] = d

    d = Diagram(1220, 570)
    d.box(25,45,190,75,'Valid word image\nH × W; square s=H/2',GRAY)
    d.box(265,45,240,75,'Geometry-only sampling\nK = clamp(ceil(W/s),4,8)\nhorizontal strata + two y bands')
    d.box(555,45,180,75,'K crops per word\n32px → 16×16\n64px → 32×32',GREEN)
    d.box(785,45,180,75,'3×3 stem\n1→64 channels')
    d.arrow([(215,82),(265,82)])
    d.arrow([(505,82),(555,82)])
    d.arrow([(735,82),(785,82)])
    d.box(265,235,250,90,'StrokePatchBlock ×3\nchannels 128 → 192 → 192\nshared minimum spatial grid 4×4',GREEN)
    d.arrow([(875,120),(875,185),(390,185),(390,235)])
    d.box(565,235,220,90,'Block anatomy (inset)\n3×3 conv + (DW 1×5 + DW 5×1)\n1×1 fuse + shortcut',GRAY)
    d.arrow([(515,305),(565,305)],dashed=True)
    d.box(835,235,220,90,'1×1 spatial logit head\nno early global averaging',RED)
    d.arrow([(515,255),(535,255),(535,210),(945,210),(945,235)])
    d.box(565,410,490,80,'D phase: hinge per spatial decision → mean map → mean crops/word\nG phase: negative score → same per-word aggregation\nThen average branches equally, not by crop count',GOLD)
    d.arrow([(945,325),(945,410)])
    d.text(25,535,'Real, random fake, transferred fake and reconstruction use the same sampling policy; coordinates are sampled independently.',14,anchor='start')
    figures['04_strokepatchd'] = d

    for key, mode in [('05_random_flow','random'),('06_transfer_flow','transfer'),('07_reconstruction_flow','reconstruction')]:
        d = Diagram(1260, 470)
        if mode == 'random':
            entry='Random prior [zg,Lz]\nTarget text y′'
            aux='Frozen R\nCTC against y′'
            side='Frozen B → E(global)\nInfo L1 against stopgrad(zg) / 8'
            loss='Global + patch adversarial\n+ random CTC + Info'
        elif mode == 'transfer':
            entry='Reference x → B → E\nTarget text y′'
            aux='Frozen R and W+B\nCTC(y′) + writer CE(w)'
            side='Frozen B features + E(global)\nContext against x; cycle against stopgrad(gx)'
            loss='Global + patch adversarial\n+ style CTC + writer + context + cycle'
        else:
            entry='Reference x → B → E\nOriginal transcript y'
            aux='Reference posterior μg,log σ²g\nGlobal KL against N(0,I) / 8'
            side='Masked pixel L1\ncompare generated x̂rec with real x'
            loss='Global + patch adversarial\n+ pixel reconstruction; KL on E'
        d.box(25,65,230,80,entry,GRAY)
        d.box(310,65,180,80,'Shared generator G\n(global + local routes)')
        d.box(545,65,160,80,'Generated image\n'+{'random':'x̂rand','transfer':'x̂style','reconstruction':'x̂rec'}[mode],GREEN)
        d.box(760,65,215,80,'Global D: whole word\nP: stratified stroke crops',RED)
        d.box(1020,65,215,80,'Real image x\nReal/fake hinge targets',GRAY)
        d.arrow([(255,105),(310,105)])
        d.arrow([(490,105),(545,105)])
        d.arrow([(705,105),(760,105)])
        d.arrow([(1020,105),(975,105)])
        d.box(330,245,300,75,aux,GOLD)
        d.box(690,245,370,75,side,GOLD)
        if mode == 'reconstruction':
            d.arrow([(140,145),(140,282),(330,282)])
        else:
            d.arrow([(595,145),(595,205),(480,205),(480,245)])
        d.arrow([(655,145),(655,205),(875,205),(875,245)])
        d.box(330,365,730,65,loss,GOLD)
        d.arrow([(480,320),(480,365)])
        d.arrow([(875,320),(875,365)])
        d.arrow([(867,145),(867,175),(1120,175),(1120,397),(1060,397)])
        d.text(25,455,'Same G/E/D/P weights across branches. D/P update: fake detached. G/E update: critics/teachers frozen, image gradients retained.',14,anchor='start')
        figures[key]=d
    return figures


if __name__ == '__main__':
    folder = Path(__file__).resolve().parent / 'figures'
    folder.mkdir(exist_ok=True)
    for name, diagram in make_figures().items():
        (folder / (name + '.svg')).write_text(diagram.svg(), encoding='utf-8')
    print(f'Generated {len(make_figures())} deterministic SVG figures.')
