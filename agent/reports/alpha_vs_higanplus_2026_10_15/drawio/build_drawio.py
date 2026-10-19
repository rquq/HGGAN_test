"""Native, editable mxGraph pages for ALPHA; no model code is modified.

Run with the HiGANplus Python. All diagrams share one semantic graph with the
offline preview and geometry checks. No rasterized architecture is embedded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from html import escape
from heapq import heappush, heappop
from itertools import count
from hashlib import sha256
import json
import math
import subprocess
import textwrap
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/quq/machineLearning/HTG/HGGAN_test/alpha')
W, H = 1600, 1040
INK, MUTED, PAPER = '#263746', '#526171', '#ffffff'
COLORS = {'data': '#f1f3f5', 'module': '#e8f0f8', 'style': '#e6f3ee',
          'loss': '#fff2d6', 'critic': '#f9e9e5', 'frozen': '#f7f3e9'}
LOSS_INK = '#8a4b23'


@dataclass
class Node:
    id: str
    x: int
    y: int
    w: int
    h: int
    title: str
    sub: str = ''
    kind: str = 'module'
    frozen: bool = False
    link: str = ''

    def port(self, side, fraction=.5):
        return {'R': (self.x+self.w, self.y+self.h*fraction),
                'L': (self.x, self.y+self.h*fraction),
                'T': (self.x+self.w*fraction, self.y),
                'B': (self.x+self.w*fraction, self.y+self.h)}[side]


@dataclass
class Edge:
    source: str
    target: str
    out: str = 'R'
    into: str = 'L'
    a: float = .5
    b: float = .5
    role: str = 'forward'
    points: list = field(default_factory=list)


@dataclass
class Page:
    id: str
    name: str
    title: str
    subtitle: str
    nodes: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)
    zones: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    source: str = 'networks/model.py'

    def n(self, id, x, y, w, title, sub='', kind='module', h=80, frozen=False, link=''):
        assert id not in self.nodes
        self.nodes[id] = Node(id, x, y, w, h, title, sub, kind, frozen, link)
        return id

    def e(self, source, target, out='R', into='L', a=.5, b=.5, role='forward'):
        self.edges.append(Edge(source, target, out, into, a, b, role))

    def zone(self, x, y, w, h, label):
        self.zones.append((x, y, w, h, label))

    def note(self, text):
        self.notes.append(text)


def interior_crossing(a, b, rect, pad=0):
    """True only for a positive-length segment inside an expanded rectangle."""
    x, y, w, h = rect
    l, r, t, bot = x-pad, x+w+pad, y-pad, y+h+pad
    if a[0] == b[0]:
        return l < a[0] < r and max(min(a[1], b[1]), t) < min(max(a[1], b[1]), bot)
    if a[1] == b[1]:
        return t < a[1] < bot and max(min(a[0], b[0]), l) < min(max(a[0], b[0]), r)
    return True


def interaction_cost(a, b, previous, cross_penalty=100):
    cost = 0
    for c, d in previous:
        if a[0] == b[0] == c[0] == d[0]:
            overlap = min(max(a[1], b[1]), max(c[1], d[1])) - max(min(a[1], b[1]), min(c[1], d[1]))
            if overlap > 1:
                cost += overlap * 6 + 160
        elif a[1] == b[1] == c[1] == d[1]:
            overlap = min(max(a[0], b[0]), max(c[0], d[0])) - max(min(a[0], b[0]), min(c[0], d[0]))
            if overlap > 1:
                cost += overlap * 6 + 160
        elif a[0] == b[0] and c[1] == d[1]:
            if min(c[0], d[0]) < a[0] < max(c[0], d[0]) and min(a[1], b[1]) < c[1] < max(a[1], b[1]):
                cost += cross_penalty
        elif a[1] == b[1] and c[0] == d[0]:
            if min(a[0], b[0]) < c[0] < max(a[0], b[0]) and min(c[1], d[1]) < a[1] < max(c[1], d[1]):
                cost += cross_penalty
    return cost


def simplify(points):
    result = []
    for p in points:
        if result and p == result[-1]:
            continue
        if len(result) >= 2 and ((result[-2][0] == result[-1][0] == p[0]) or
                                 (result[-2][1] == result[-1][1] == p[1])):
            result[-1] = p
        else:
            result.append(p)
    return result


def route(page, width=1600, top=132, bottom=868, margin=40,
          clearance=20, obstacle_pad=8, cross_penalty=100):
    """Orthogonal obstacle-aware routing, with penalties for shared strokes."""
    previous = []
    normals = {'R': (1, 0), 'L': (-1, 0), 'T': (0, -1), 'B': (0, 1)}
    for edge in page.edges:
        src, dst = page.nodes[edge.source], page.nodes[edge.target]
        begin, end = src.port(edge.out, edge.a), dst.port(edge.into, edge.b)
        ns, nt = normals[edge.out], normals[edge.into]
        start = (begin[0] + ns[0]*clearance, begin[1] + ns[1]*clearance)
        goal = (end[0] + nt[0]*clearance, end[1] + nt[1]*clearance)
        rects = [(n.x, n.y, n.w, n.h) for n in page.nodes.values()]
        # Panel headers are obstacles too: a connector must not bisect a title.
        for x,y,w,h,label in page.zones:
            header_width=min(w-40,math.ceil((14*.59*len(label)+24)/4)*4)
            rects.append((x+16,y+4,header_width,32))
        # Keep obstacle margins thin enough to route between nearby cards.
        xs = sorted({float(margin), float(width-margin), start[0], goal[0]} |
                    {float(v) for x,y,w,h in rects for v in (x-clearance, x+w+clearance) if margin <= v <= width-margin})
        ys = sorted({float(top), float(bottom), start[1], goal[1]} |
                    {float(v) for x,y,w,h in rects for v in (y-clearance, y+h+clearance) if top <= v <= bottom})
        xi, yi = {v:i for i,v in enumerate(xs)}, {v:i for i,v in enumerate(ys)}
        initial = (xi[start[0]], yi[start[1]], 0)
        target_xy = (xi[goal[0]], yi[goal[1]])
        serial = count()
        heap = [(0., next(serial), initial)]
        best, parents = {initial: 0.}, {}
        found = None
        while heap:
            _, _, state = heappop(heap)
            i, j, direction = state
            if (i, j) == target_xy:
                found = state
                break
            a = (xs[i], ys[j])
            for di, dj, nd in ((1,0,1),(-1,0,1),(0,1,2),(0,-1,2)):
                ni, nj = i+di, j+dj
                if not (0 <= ni < len(xs) and 0 <= nj < len(ys)):
                    continue
                b = (xs[ni], ys[nj])
                if any(interior_crossing(a,b,r,pad=obstacle_pad) for r in rects):
                    continue
                length = abs(b[0]-a[0])+abs(b[1]-a[1])
                score = best[state] + length + (36 if direction and direction != nd else 0) + interaction_cost(a,b,previous,cross_penalty)
                nxt = (ni,nj,nd)
                if score < best.get(nxt, math.inf):
                    best[nxt], parents[nxt] = score, state
                    heur = abs(b[0]-goal[0])+abs(b[1]-goal[1])
                    heappush(heap,(score+heur,next(serial),nxt))
        if found is None:
            raise ValueError(f'Unroutable: {page.name}: {edge.source} -> {edge.target}')
        walk = []
        while True:
            walk.append((xs[found[0]],ys[found[1]]))
            if found == initial:
                break
            found = parents[found]
        edge.points = simplify([begin]+list(reversed(walk))+[end])
        previous.extend(zip(edge.points,edge.points[1:]))


def pages():
    result=[]
    def p(id,name,title,subtitle,source='networks/model.py'):
        x=Page(id,name,title,subtitle,source=source)
        result.append(x)
        return x

    d=p('guide','00 Read me','ALPHA · training atlas',
        'Native editable diagrams · current code snapshot · 32 px and 64 px · no baseline modules copied')
    d.zone(52,148,1496,176,'HOW TO READ')
    d.n('notation',80,200,680,'Notation',
        'x_ref: original reference geometry · x: canonical reconstruction target\ny: reference transcript · y\u2032: sampled target text · w: writer ID',kind='data',h=96)
    d.n('styles',800,200,720,'Style representation S = [g, L]',
        'g: 1 global token, 32-D · L: 7 local tokens, 7 × 32-D\nPrior z has the same layout; local slots are centered.',kind='style',h=96)
    d.n('train',80,380,680,'TRAINABLE vs FROZEN',
        'Dashed module border = frozen parameters in that phase.\nFrozen modules can still transmit gradients to generated pixels.',kind='frozen',h=104)
    d.n('flow',800,380,720,'Solid vs dashed arrows',
        'Solid: forward tensor or measured prediction.\nDashed brown: target, detached target, initialization, or optimizer action.',kind='loss',h=104)
    d.n('part1',80,548,440,'01–04 · GAN training',
        'Overview · alternating updates\nGlobal discriminator · StrokePatchD',kind='module',h=104,link='data:page/id,overview')
    d.n('part2',580,548,440,'05–09 · Branches + architecture',
        'Random · transfer · reconstruction\nStyle encoder · fusion / Generator',kind='style',h=104,link='data:page/id,random')
    d.n('part3',1080,548,440,'10–14 · Pretrain + evaluation',
        'OCR · writer + backbone\nInference · metrics · loss ledger',kind='loss',h=104,link='data:page/id,ocr')
    d.n('scope',80,720,1440,'Scope / presentation references',
        'All pages describe ALPHA only. HiGAN+, SpiS-GAN and FW-GAN figures inform panel layout and the loss-arrow convention.\nBoxes and connectors are native mxGraph objects, not one flattened image.',kind='data',h=104)
    d.note('Open the file with File → Open From → Device in diagrams.net; choose tabs along the bottom. Ctrl/Cmd + mouse wheel zooms.')
    d.note('Repeated B, E, G, D, P, R or W labels denote shared weights within the corresponding training phase, not separate networks.')

    d=p('overview','01 GAN overview','Three connected generation branches',
        'Same G / E weights across branches · image paths and loss families only; details follow in separate tabs')
    for yy,label in ((148,'A · RANDOM GENERATION'),(380,'B · REFERENCE-GUIDED TRANSFER'),(612,'C · RECONSTRUCTION')):
        d.zone(52,yy,1496,196,label)
    d.n('z',80,212,252,'Prior z + target y\u2032','1 global + 7 centered locals',kind='style')
    d.n('g1',392,212,216,'Generator G','trainable; shared weights')
    d.n('r1',668,212,252,'Random word x_rand','new style, new text',kind='data')
    d.n('l1',992,212,528,'D + P · R · B → E_global','Adversarial · CTC · global latent recovery',kind='loss')
    d.n('e2',80,444,252,'x_ref → B → E','B frozen; E trainable',kind='style')
    d.n('g2',392,444,216,'Generator G + y\u2032','new text; reference style')
    d.n('r2',668,444,252,'Transfer word x_style','same writer, different text',kind='data')
    d.n('l2',992,444,528,'D + P · R · W+B · E_global','Adversarial · CTC · writer · context · style cycle',kind='loss')
    d.n('e3',80,676,252,'x_ref → B → E','same encoded S as transfer',kind='style')
    d.n('g3',392,676,216,'Generator G + y','original reference transcript')
    d.n('r3',668,676,252,'Reconstruction x_rec','canonical target geometry',kind='data')
    d.n('l3',992,676,528,'D + P · valid-pixel L1','Adversarial · reconstruction (no reconstruction CTC)',kind='loss')
    for row in (('z','g1','r1','l1'),('e2','g2','r2','l2'),('e3','g3','r3','l3')):
        for a,b in zip(row,row[1:]):d.e(a,b)
    d.note('Reference E also receives global-posterior KL and a gradient-reversed content probe; those are expanded on page 07.')
    d.note('The code concatenates the three branches into one batched G forward for efficiency; the separate rows show their semantics.')

    d=p('alternation','02 Alternating updates','Who learns at each GAN step?',
        'Current normal 32/64 configs: num_critic_train = 2 · one G/E optimizer, separate D and P optimizers')
    d.zone(52,148,1496,264,'A · EVERY TRAINING BATCH — UPDATE D AND P')
    d.n('data',80,236,252,'Real batch + text','x_ref, target x, transcript y, writer w',kind='data',h=96)
    d.n('fake',392,236,304,'Generate three fake groups','B/E/G forward with no_grad\nrandom · transfer · reconstruction')
    d.n('dp',756,236,300,'D / P hinge + lazy R1','detach fake images; train D/P',kind='critic')
    d.n('upd',1116,236,404,'Adam step: D and P','G/E are not updated by this phase',kind='loss')
    d.e('data','fake');d.e('fake','dp');d.e('dp','upd')
    d.zone(52,484,1496,320,'B · EVERY SECOND CRITIC BATCH — UPDATE G AND E')
    d.n('newfake',80,588,288,'Regenerate with autograd','G + E trainable\nB/R/W/D/P parameter-frozen',h=104)
    d.n('objectives',432,588,332,'All generator objectives','D/P + CTC + writer + L1\nlatent/cycle + content + context + KL',kind='loss',h=104)
    d.n('ge',828,588,300,'Adam step: G and E','gradients through frozen teachers',h=104)
    d.n('ema',1192,588,328,'EMA update: G and E','beta = 0.9998\nused by configured evaluation',kind='style',h=104)
    d.e('newfake','objectives');d.e('objectives','ge');d.e('ge','ema')
    d.note('D/P remain differentiable with respect to generated pixels during G/E training; parameter freezing is not torch.no_grad().')
    d.note('B/W run in eval mode. R uses eval behavior except its recurrent block is kept in training mode for cuDNN backward support.')

    d=p('global-d','03 Global discriminator','Whole-word adversarial learning',
        'No patches are presented to the global D · shared D weights are repeated below to keep the loss comparisons explicit')
    d.zone(52,148,1496,308,'A · CRITIC UPDATE — TRAIN D ONLY')
    d.n('real',80,212,260,'Real word x','valid width',kind='data')
    d.n('fake',80,352,260,'Three fake groups','x_rand / x_style / x_rec; detach',kind='data')
    d.n('aug1',404,212,280,'Differentiable augmentation','mild translation + width scaling')
    d.n('aug2',404,352,280,'Same augmentation policy','independent draws, same strength')
    d.n('d1',748,212,232,'Global D','whole-word score',kind='critic')
    d.n('d2',748,352,232,'Global D','shared parameters',kind='critic')
    d.n('hinge',1052,212,468,'Discriminator hinge loss',
        'L_D = mean ReLU(1 − D(real))\n+ mean_branches mean ReLU(1 + D(fake))',kind='loss',h=104)
    d.n('r1',1052,352,468,'Lazy real-image R1',
        '0.5 × λ_R1 × interval × ||∇_x D(x)||²\nλ_R1 = 0.01; interval = 32',kind='loss',h=80)
    d.e('real','aug1');d.e('fake','aug2');d.e('aug1','d1');d.e('aug2','d2');d.e('d1','hinge',a=.3,b=.25)
    d.e('d2','hinge',into='B');d.e('d1','r1',a=.75,b=.5)
    d.zone(52,520,1496,272,'B · GENERATOR UPDATE — FREEZE D PARAMETERS, KEEP INPUT GRADIENTS')
    d.n('live',80,616,292,'Generated words','not detached; all three branches',kind='data')
    d.n('aug3',436,616,292,'Differentiable augmentation','same global-D input policy')
    d.n('d3',792,616,260,'Global D','frozen parameters',kind='critic',frozen=True)
    d.n('gadv',1116,600,404,'Global generator adversarial',
        'L_adv = −mean_branches mean D(fake)\nweight = 1.0 → updates G/E',kind='loss',h=112)
    d.e('live','aug3');d.e('aug3','d3');d.e('d3','gadv')
    d.note('D: residual CNN → optional width context → valid-area mean × 32 → scalar score. Width context is OFF in normal 32/64 configs.')
    d.note('Augmented real and fake whole words use the same global-D policy. Patch crops are taken from the unwarped image paths.')

    d=p('patch-d','04 StrokePatchD','Local stroke critic: sampling, comparison and network',
        'One policy for real and all three fake branches · patch logits are supervised before spatial/crop reductions',
        'networks/utils.py; networks/BigGAN_networks.py; networks/model.py')
    d.zone(52,148,1496,284,'A · SHARED CROP POLICY — NO CHARACTER-ID OR DATASET-SPECIFIC RULE')
    d.n('images',80,224,264,'Real / random / transfer / rec','whole image + valid width',kind='data',h=96)
    d.n('policy',404,204,364,'Stratified stroke patches',
        'side s = H/2\nK = clamp(ceil(valid_width/s), 4, 8)\nx: strata; y: alternate upper/lower',h=136)
    d.n('patches',828,224,280,'K native patches per word',
        'H32 → 16×16 · H64 → 32×32\nshort-width outside fill = −1',kind='data',h=96)
    d.n('patchnet',1172,224,348,'Shared StrokePatchD P',
        'one spatial logit map per crop\nnever enters global D',kind='critic',h=96)
    d.e('images','policy');d.e('policy','patches');d.e('patches','patchnet')
    d.zone(52,472,1496,176,'B · P NETWORK — SAME STRUCTURE AT BOTH HEIGHTS')
    d.n('stem',80,528,252,'3×3 stem','1 → 64 channels',kind='critic')
    d.n('blocks',392,528,472,'StrokePatchBlock ×3','128 → 192 → 192; 3×3 + axial DW(1×5, 5×1)\n1×1 fusion + residual; pool only when grid stays ≥4',kind='critic')
    d.n('map',924,528,260,'1×1 logit head','spatial score map',kind='critic')
    d.n('reduce',1244,528,276,'Per-word reduction','map → crop → word\nequal word / branch weighting',kind='loss')
    d.e('stem','blocks');d.e('blocks','map');d.e('map','reduce')
    d.e('patchnet','stem',out='B',into='L',b=.25)
    d.n('lp',80,716,684,'Critic: L_P = Hinge(P(real), +1) + mean_b Hinge(P(fake_b), −1)',
        'Train P; fake tensors detached. Hinge is applied per logit BEFORE spatial and crop averaging.',kind='loss',h=104)
    d.n('lgp',824,716,696,'Generator: L_adv,P = −mean_b mean_word P(fake_b)',
        'Freeze P; preserve input gradients. Generator weight λ_patch_adv = 0.45.',kind='loss',h=104)
    d.e('reduce','lp',out='B',into='R',a=.3,role='forward')
    d.e('reduce','lgp',out='B',into='T',a=.7,role='forward')
    d.note('Independent coordinates are drawn for each image; “same policy” does not mean pixel-aligned real/fake crops.')
    d.note('P receives no writer/text labels. It learns local realism from adversarial real-vs-fake supervision, not predefined stroke classes.')

    d=p('random','05 Random + latent recovery','Random generation: content and global-style recovery',
        'Train G/E · freeze B/R and both critics · local prior slots are centered; no local-token recovery loss')
    d.zone(52,148,1496,216,'A · GENERATE')
    d.n('prior',80,212,304,'Sample prior z = [z_g, z_L]','z_g ∼ N(0,I); z_L centered over slots',kind='style')
    d.n('text',444,212,248,'Sample target text y\u2032','lexicon; encoded character IDs',kind='data')
    d.n('g',756,212,276,'Generator G','input: z and y\u2032')
    d.n('image',1096,212,424,'Random word x_rand','H × (character_width × text_length)',kind='data')
    d.e('prior','g',out='B',into='B',b=.25);d.e('text','g');d.e('g','image')
    d.zone(52,420,1496,384,'B · DIFFERENTIABLE SUPERVISION')
    d.n('ocr',80,504,280,'Frozen recognizer R','CNN + BiLSTM + log-softmax',frozen=True)
    d.n('ctc',424,504,368,'Random CTC loss','CTC(R(x_rand), y\u2032)\nweight: 1.0 → 0.4 (epochs 10–20)',kind='loss')
    d.n('b',80,688,280,'Frozen backbone B','input gradients retained',frozen=True)
    d.n('eg',424,688,296,'E_global','skip local-token retrieval')
    d.n('info',784,680,348,'Global information loss','mean |g_hat − stop(z_g)| / 8\nλ_info = 1.0',kind='loss',h=96)
    d.n('adv',1196,504,324,'Global D + StrokePatchD','L_adv + 0.45 L_adv,P\nsee pages 03 and 04',kind='loss',h=104)
    d.e('image','ocr',out='L',into='T',a=.3);d.e('ocr','ctc')
    d.e('text','ctc',out='B',into='T',role='target')
    d.e('image','b',out='B',into='T',a=.3);d.e('b','eg');d.e('eg','info')
    d.e('prior','info',out='B',into='T',a=.3,role='target')
    d.e('image','adv',out='B',into='T',a=.7)
    d.note('z_g is a detached target in the information loss; E and G still receive gradients through the re-encoding prediction.')
    d.note('All losses on these pages are forward objective dependencies. Backpropagation is not drawn as a second set of reversed arrows.')

    d=p('transfer','06 Style transfer','Reference-guided generation: which signal compares to what?',
        'Train G/E · freeze B/W/R · W and cycle/context supervision reuse the same generated B features')
    d.zone(52,148,1496,208,'A · REFERENCE → NEW TEXT')
    d.n('ref',80,208,232,'Reference x_ref','original word geometry',kind='data')
    d.n('be',372,208,268,'B → E','B frozen; E trainable')
    d.n('style',700,208,232,'Reference S = [g,L]','g sampled; locals deterministic',kind='style')
    d.n('g',992,208,240,'G + target text y\u2032','different text, same style')
    d.n('fake',1292,208,228,'Transfer x_style','generated word',kind='data')
    for a,b in (('ref','be'),('be','style'),('style','g'),('g','fake')):d.e(a,b)
    d.zone(52,404,1496,416,'B · FROZEN TEACHERS AND TRAINABLE GLOBAL RE-ENCODER')
    d.n('r',80,468,232,'Recognizer R','frozen',frozen=True)
    d.n('ctc',372,468,312,'CTC vs target y\u2032','weight: 0.5 → 0.20\nepochs 8–28',kind='loss')
    d.n('bfake',80,692,232,'Backbone B','frozen; cache fake features',frozen=True)
    d.n('w',372,612,248,'Writer head W','frozen; known train writers',frozen=True)
    d.n('wid',684,600,312,'Writer CE vs w_ref','λ_wid = 0.5\nwriter label of reference image',kind='loss',h=104)
    d.n('eg',372,724,248,'E_global','trainable; local path skipped')
    d.n('cycle',684,716,312,'Global style-cycle L1','vs stop(mu_g,ref); divide by 8\nλ_style_cycle = 1.0',kind='loss',h=96)
    d.n('ctx',1060,612,460,'Multiscale contextual loss',
        'fake B maps vs stop(reference B maps)\nnon-aligned content; bounded spatial sampling\nλ_ctx = 0.1',kind='loss',h=112)
    d.n('adv',1060,468,460,'D + P adversarial','L_adv + 0.45 L_adv,P',kind='loss')
    d.e('fake','r',out='B',into='T',a=.25);d.e('r','ctc')
    d.e('fake','bfake',out='B',into='T',a=.45)
    d.e('bfake','w',a=.25);d.e('w','wid');d.e('bfake','eg',a=.75);d.e('eg','cycle')
    d.e('bfake','ctx',out='B',into='B',a=.5,b=.5)
    d.e('be','ctx',out='B',into='T',a=.25,role='target')
    d.e('style','cycle',out='B',into='B',role='target')
    d.e('fake','adv',out='B',into='T',a=.75)
    d.note('CTC compares against y\u2032, NOT the reference transcript y. Writer CE compares against the reference writer ID w.')
    d.note('The reference B pass and generated B pass supply both E and contextual features; duplicate drawing does not imply duplicate computation.')

    d=p('recon','07 Reconstruction + disentanglement','Exact reconstruction and reference-posterior regularization',
        'A reconstruction target, a prior target and a content-adversary target are different signals — do not merge them')
    d.zone(52,148,1496,292,'A · IMAGE RECONSTRUCTION — SAME REFERENCE TRANSCRIPT y')
    d.n('ref',80,212,240,'Reference x_ref','original geometry',kind='data')
    d.n('be',380,212,268,'Frozen B → trainable E','posterior mu_g, logvar_g; L')
    d.n('g',708,212,260,'G(S_ref, y)','reference text, NOT y\u2032')
    d.n('rec',1028,212,220,'x_rec','canonical geometry',kind='data')
    d.n('pix',1308,196,212,'Pixel L1','x_rec versus x\nvalid image area only',kind='loss',h=112)
    d.n('target',1028,352,492,'Canonical real target x','same sample; original x_ref and target x may differ in width',kind='data',h=64)
    for a,b in (('ref','be'),('be','g'),('g','rec'),('rec','pix')):d.e(a,b)
    d.e('target','pix',out='T',into='B',a=.85,b=(1028+492*.85-1308)/212,role='target')
    d.zone(52,464,1496,344,'B · REGULARIZE REFERENCE GLOBAL CODE — NO GENERATED IMAGE IS REQUIRED HERE')
    d.n('mu',80,552,272,'Global posterior parameters','mu_g, logvar_g\nfrom the reference E pass',kind='style',h=104)
    d.n('kl',416,552,440,'Global KL to N(0,I)','L_KL = KL(q(g|x_ref) || N(0,I)) / 8\nλ_KL = 0.1; local tokens have no KL',kind='loss',h=104)
    d.n('grl',80,704,272,'mu_g → gradient reversal','identity forward; reverse gradient to E')
    d.n('probe',416,704,260,'Content probe inside E','trainable linear prediction')
    d.n('bce',740,688,400,'Content-adversarial BCE','vs bag of reference characters\nBCE / 8; λ_content_adv = 0.02',kind='loss',h=112)
    d.n('bag',1204,688,316,'Bag of characters in y','multi-hot content target\nnot a handcrafted character case',kind='data',h=112)
    d.e('be','mu',out='B',into='T',a=.3);d.e('mu','kl');d.e('mu','grl',out='B',into='T')
    d.e('grl','probe');d.e('probe','bce');d.e('bag','bce',out='L',into='R',role='target')
    d.note('Reconstruction also receives both adversarial losses (pages 03/04). Pixel weight decays 10 → 2.5 over epochs 24–42.')
    d.note('The probe learns to recover content; gradient reversal trains the global encoder code to hide it. No extra OCR loss is applied to x_rec.')

    d=p('encoder','08 Style encoder','Style encoder E: separate global statistics and local evidence',
        'B is frozen in GAN; E is trainable · 8 total style tokens (1 global + 7 local) · token dimension 32',
        'networks/module.py: StyleBackbone; GatedStyleContext; StyleEncoder')
    d.zone(52,148,1496,280,'A · GLOBAL PATH — STOCHASTIC STYLE POSTERIOR')
    d.n('b',80,232,260,'Reference → backbone B','valid-width-aware; frozen',frozen=True)
    d.n('deep',400,232,260,'Deep 256-D sequence','deepest B features')
    d.n('context',720,212,352,'GatedStyleContext: 2 blocks','DW k7, dilation 1 then 2\ngated channel MLP + residual',h=112)
    d.n('global',1132,212,388,'Valid mean → LN → MLP','mu_g, logvar_g → reparameterize\ng = mu_g + exp(0.5 logvar_g) ε',kind='style',h=112)
    for a,b in (('b','deep'),('deep','context'),('context','global')):d.e(a,b)
    d.zone(52,480,1496,336,'B · LOCAL PATH — DETERMINISTIC MULTISCALE EVIDENCE')
    d.n('multi',80,552,260,'B maps: 64 / 128 / 256','three spatial scales',kind='style')
    d.n('proj',400,552,280,'Pool height to 4 rows','per-scale 1×1 projection → 256\nflatten + normalization')
    d.n('kv',740,536,348,'K = visual + 2-D position','V = visual − valid spatial mean\npositions address evidence; not appearance',kind='style',h=112)
    d.n('queries',400,712,280,'7 learned local queries','orthogonally initialized',kind='style')
    d.n('attention',1148,536,372,'4-head cross-attention','Q: learned queries; K/V: B evidence\nbias-free local output projection',h=112)
    d.n('center',1148,712,372,'256 → 32; center over slots','L_j ← L_j − mean_k L_k\n7 deterministic local tokens',kind='style')
    d.e('b','multi',out='B',into='T');d.e('multi','proj');d.e('proj','kv');d.e('kv','attention')
    d.e('queries','attention',out='B',into='B',a=.7,b=.25);d.e('attention','center',out='B',into='T')
    d.note('Output S = [g, L1, …, L7] has shape batch × 8 × 32. Only the global posterior is sampled and KL-regularized.')
    d.note('global_only=True skips the local retrieval path during fake re-encoding. There is no special rule for “/”, digits, or reference text length.')

    d=p('generator','09 Fusion + Generator','Content-first generation with two distinct style routes',
        'Same four-stage G at H32/H64; final-stage geometry changes with requested height, not dataset identity',
        'networks/fusion.py; networks/BigGAN_networks.py: Generator; BlockSpecificStyleProjection')
    d.zone(52,148,1496,260,'A · TEXT LAYOUT — CONTENT ONLY, BEFORE STYLE REFINEMENT')
    d.n('text',80,220,228,'Target character IDs','text length T',kind='data')
    d.n('embed',368,220,240,'Embedding','120-D per character')
    d.n('sa',668,200,336,'Content self-attention','relative-position; 4 heads\nattention width 128; residual',h=112)
    d.n('ffn',1064,200,456,'Local gated FFN → spatial seed','DWConv1D k5; hidden width 360\nlinear + reshape → 512 × 4 × (4T)',h=112)
    for a,b in (('text','embed'),('embed','sa'),('sa','ffn')):d.e(a,b)
    d.zone(52,456,1496,388,'B · STYLE-CONDITIONED IMAGE SYNTHESIS')
    d.n('global',80,524,228,'Global token g','32-D',kind='style')
    d.n('bssp',368,508,336,'Block-specific style projections','4 independent MLPs: 32 → 32 → 32\none vector per GBlock → CBN',kind='style',h=112)
    d.n('local',80,704,228,'7 local tokens L','7 × 32-D',kind='style')
    d.n('g1',368,704,216,'GBlock 1','512 → 256; H=8')
    d.n('refine',648,684,332,'Spatial style refinement','Q: spatial content features\nK: LN(local); V: raw local\n4 heads, width 128; learned residual',h=136)
    d.n('g234',1044,684,256,'GBlocks 2–4','256 → 128 → 64 → 64\neach has global-conditioned BN',h=136)
    d.n('image',1360,704,160,'BN → conv','tanh image',kind='data')
    d.e('global','bssp');d.e('ffn','g1',out='B',into='T',a=.3,b=.75)
    d.e('g1','refine');d.e('local','refine',out='B',into='B',b=.3)
    d.e('bssp','g1',out='B',into='T',a=.25,b=.25,role='forward')
    d.e('bssp','g234',out='R',into='T',a=.5,role='forward')
    d.e('refine','g234');d.e('g234','image')
    d.note('GBlock stages: H=8 → 16 → 32 → (32 or 64). Output width is 16T at H32 and 32T at H64; padding is not a foreground mask.')
    d.note('Local attention is applied ONCE after GBlock 1. No Mamba, allograph-ID gate, texture generator, or frequency loss is active here.')

    d=p('ocr','10 OCR pretraining','Pretrain recognizer R on real labeled word images',
        'R alone is trainable · no GAN, no writer classifier, no local-style encoder · height-matched 32/64 configurations',
        'networks/model.py: RecognizeModel; networks/module.py: Recognizer; configs/ocr_iam_{32,64}.yml')
    d.zone(52,148,1496,300,'A · SUPERVISED CTC TRAINING — TRAINING SPLIT: trnval')
    d.n('data',80,220,252,'Real image + transcript','normalized image, valid width\nsource white → model black polarity',kind='data',h=112)
    d.n('aug',392,220,276,'Handwriting augmentation','scale + mild handwriting transform',h=112)
    d.n('cnn',728,220,248,'R: residual CNN','width reduction: 8 or 16',h=112)
    d.n('rnn',1036,220,224,'2-layer BiLSTM','packed valid sequence',h=112)
    d.n('ctc',1320,200,200,'Linear + CTC','81-class logits\nCTC vs transcript',kind='loss',h=152)
    for a,b in (('data','aug'),('aug','cnn'),('cnn','rnn'),('rnn','ctc')):d.e(a,b)
    d.e('data','ctc',out='B',into='B',a=.3,b=.5,role='target')
    d.zone(52,504,1496,300,'B · OPTIMIZATION, EVALUATION AND EXPORT')
    d.n('opt',80,600,308,'Adam update: R','100 epochs; batch 128; lr 1e−3\nlinear decay epochs 60–100',kind='module',h=112)
    d.n('ema',448,600,280,'Optional EMA(R)','teacher_ema toggle\nbeta = 0.999',kind='style',h=112)
    d.n('valid',788,600,336,'Validate on test split','every 10 epochs; batch 256\ndecode → CER / WER',kind='loss',h=112)
    d.n('export',1184,600,336,'Export OCR checkpoint','ocr_iam_train32.pth\nor ocr_iam_train64.pth',kind='data',h=112)
    d.e('ctc','opt',out='B',into='T',role='update');d.e('opt','ema');d.e('ema','valid');d.e('valid','export')
    d.note('When teacher_ema=false, validation/export use current online R weights directly; EMA is not a second independently trained OCR.')
    d.note('OCR CNN weights can initialize B for W+B pretraining. In GAN training, R is frozen and generated images are NOT inverted before R.')

    d=p('writer','11 Writer + backbone pretraining','Pretrain the shared writer backbone B and writer head W',
        'Both B and W train here · E is not part of pretraining · a single writer checkpoint stores both networks',
        'networks/model.py: WriterIdentifyModel; networks/module.py: MaskedAttentiveStatsPool; WriterIdentifier')
    d.zone(52,148,1496,328,'A · WRITER CLASSIFICATION — KNOWN TRAINING WRITERS')
    d.n('sample',80,224,280,'Writer-balanced real batch','trnval; inverse-count sampling\npower 0.5; capped weight 3',kind='data',h=112)
    d.n('select',420,224,280,'Mixed image input policy','raw or augmented / canonical\nraw probability = 0.5',h=112)
    d.n('b',760,224,232,'Backbone B','trainable; OCR CNN init',h=112)
    d.n('pool',1052,204,232,'Attentive statistics','valid-width weights\nweighted mean + std\n512 → 256 projection',h=152)
    d.n('w',1344,224,176,'W: MLP','339 writer logits',h=112)
    for a,b in (('sample','select'),('select','b'),('b','pool'),('pool','w')):d.e(a,b)
    d.zone(52,524,1496,296,'B · LOSS, WEIGHT UPDATE AND EXPORT')
    d.n('ce',80,612,312,'Writer cross-entropy','prediction W(B(x)) vs writer ID w\nno OCR / CTC objective here',kind='loss',h=112)
    d.n('opt',452,612,300,'Adam update: W + B','100 epochs; batch 128; lr 1e−3\nlinear decay epochs 60–100',h=112)
    d.n('ema',812,612,300,'Optional EMA(W + B)','beta 0.999; validate every 10 epochs\ntrnval split: closed-set writer error',kind='style',h=112)
    d.n('export',1172,612,348,'Export W and B together','wid_iam_train32.pth\nor wid_iam_train64.pth',kind='data',h=112)
    d.e('w','ce',out='B',into='T');d.e('sample','ce',out='B',into='T',a=.25,b=.25,role='target')
    d.e('ce','opt');d.e('opt','ema');d.e('ema','export')
    d.note('Default frozen_backbone=false means B IS updated. The writer checkpoint carries StyleBackbone and WriterIdentifier weights.')
    d.note('Writer validation uses trnval known identities; this is not an unseen-writer identification test. GAN test-writer metrics are a separate task.')

    d=p('inference','12 Inference','Two ways to generate handwriting — no training critics required',
        'Inference uses G only for a prior style, or B/E/G for reference-guided style; configured GAN evaluation uses EMA G/E')
    d.zone(52,148,1496,260,'A · RANDOM STYLE — NO REFERENCE IMAGE')
    d.n('prior',80,236,292,'Sample z = [z_g, z_L]','Gaussian prior; center local slots',kind='style')
    d.n('text1',436,236,252,'Target text y\u2032','arbitrary supported text',kind='data')
    d.n('g1',752,236,280,'Generator G','same weights as training')
    d.n('out1',1096,236,424,'Generated word','no D/P/R/W at inference',kind='data')
    d.e('prior','g1',out='B',into='B',b=.25);d.e('text1','g1');d.e('g1','out1')
    d.zone(52,480,1496,328,'B · STYLE TRANSFER — REFERENCE WORD CAN HAVE DIFFERENT TEXT')
    d.n('ref',80,572,252,'Style reference x_ref','real image + valid width',kind='data')
    d.n('b',392,572,248,'Frozen backbone B','pretrained writer backbone',frozen=True)
    d.n('e',700,572,260,'Style encoder E','g=mu_g at deterministic eval\n7 deterministic local tokens',kind='style')
    d.n('g2',1020,572,232,'Generator G','S_ref + target y\u2032')
    d.n('out2',1312,572,208,'Styled output','same writer, new text',kind='data')
    d.n('text2',1020,716,232,'Target text y\u2032','not the reference transcript',kind='data',h=64)
    for a,b in (('ref','b'),('b','e'),('e','g2'),('g2','out2')):d.e(a,b)
    d.e('text2','g2',out='T',into='B')
    d.note('B is loaded from the W+B checkpoint. W is unnecessary for forward generation once B is available; R and critics are training/eval tools.')
    d.note('Display inversion is separate from model-space tensors. Source images and fake tensors use one consistent polarity throughout the model.')

    d=p('validation','13 Validation + checkpoints','Evaluation is a measurement flow, not a GAN loss',
        'Frozen evaluation networks · fixed validation seed · enabled metrics are configured independently',
        'networks/model.py: validate; metric/val_metrics.py; checkpoint and CSV utilities')
    d.zone(52,148,1496,332,'A · GENERATE THE VALIDATION SET')
    d.n('set',80,224,292,'Held-out real test words','real images + valid lengths\nstyle references and target text',kind='data',h=112)
    d.n('gen',436,224,332,'Reference B → EMA E/G','deterministic evaluation mode\nno parameter updates',frozen=True,h=112)
    d.n('fake',832,224,296,'Generated validation words','same real/fake protocol\nfixed eval seed',kind='data',h=112)
    d.n('pairs',1192,224,328,'Real and generated inputs','length-aware metric preprocessing\nnot training DiffAug or P patches',kind='data',h=112)
    d.e('set','gen');d.e('gen','fake');d.e('fake','pairs');d.e('set','pairs',out='B',into='B',role='target')
    d.zone(52,528,1496,280,'B · METRICS, HISTORY AND SELECTION')
    d.n('ik',80,608,312,'Inception features','configured FID / KID / IS\nFID/KID use their own preprocessing',frozen=True,h=112)
    d.n('hwd',452,608,300,'Handwriting features / OCR','HWD; optional OCR CER/WER\nother configured metrics',frozen=True,h=112)
    d.n('table',812,608,308,'Epoch metrics history','all metric columns retained\ndisabled fields left empty',kind='loss',h=112)
    d.n('save',1180,608,340,'CSV + checkpoint state','eval_metrics.csv in run directory\nhistory embedded in .pth; best by FID',kind='data',h=112)
    d.e('pairs','ik',out='B',into='T',a=.25);d.e('pairs','hwd',out='B',into='T',a=.75)
    d.e('ik','table',out='B',into='B');d.e('hwd','table');d.e('table','save')
    d.note('Metrics do not backpropagate into G/E. Normal GAN configs currently enable FID, KID and HWD; additional metrics remain toggles.')
    d.note('All epochs persist across compatible resumes; checkpoints without history are allowed. Best-checkpoint selection remains FID, not KID.')

    d=p('losses','14 Loss ledger','Complete GAN objective — ownership and comparison targets',
        'Normal gan_iam_32.yml / gan_iam_64.yml weights · schedules are epoch-based; this page is a reference, not an extra graph')
    d.zone(52,148,1496,684,'GENERATOR / ENCODER — L_G,E = SUM OF THE WEIGHTED TERMS BELOW')
    rows=[
        ('Global adversarial','1.0','all three generated branches','− mean D(fake); gradients through frozen D'),
        ('Patch adversarial','0.45','all three generated branches','− mean P(patches(fake)); equal word/branch reduction'),
        ('Random CTC','1.0 → 0.4','random only; epochs 10–20','R(x_rand) compared against sampled text y\u2032'),
        ('Transfer CTC','0.5 → 0.20','transfer only; epochs 8–28','R(x_style) compared against sampled text y\u2032'),
        ('Global information','1.0','random only','mean |E_g(B(x_rand)) − stop(z_g)| / 8'),
        ('Global style cycle','1.0','transfer only','mean |E_g(B(x_style)) − stop(mu_g,ref)| / 8'),
        ('Content adversary','0.02','reference global posterior','GRL + learned content probe BCE vs bag(y), divided by 8'),
        ('Writer identity','0.5','transfer only','CE(W(B(x_style)), writer ID of reference)'),
        ('Pixel reconstruction','10 → 2.5','reconstruction; epochs 24–42','valid-pixel L1(x_rec, canonical real target x)'),
        ('Contextual','0.1','transfer vs reference','non-aligned multiscale B-feature comparison'),
        ('Global KL','0.1','reference posterior only','KL(q(g|x_ref) || N(0,I)) / 8; no local KL'),
    ]
    # Native, editable table cells; no connectors are needed on this page.
    for i,(title,weight,where,target) in enumerate(rows):
        y=224+i*52
        d.n('term'+str(i),80,y,280,title,kind='loss',h=44)
        d.n('weight'+str(i),360,y,160,weight,kind='data',h=44)
        d.n('branch'+str(i),520,y,360,where,kind='data',h=44)
        d.n('target'+str(i),880,y,640,target,kind='data',h=44)
    d.note('Critic objective: L_D + L_P + lazy R1 on real whole-word D. P has no R1 term in the current training loop.')
    d.note('Each /8 is the explicit num_style_tokens normalization for this 8-token model. No frequency/Gram/texture-head objective is active.')
    return result


def text_cell(root, id, x,y,w,h,value, size=14, align='left', bold=False, color=INK, link=''):
    attrs={'id':id,'value':escape(value).replace('\n','<br>'), 'vertex':'1','parent':'notes',
           'style':f'text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;overflow=hidden;align={align};verticalAlign=middle;fontFamily=Arial;fontSize={size};fontColor={color};fontStyle={1 if bold else 0};spacing=0;'}
    parent=root
    if link:
        parent=ET.SubElement(root,'UserObject',{'id':attrs.pop('id'),'label':attrs.pop('value'),'link':link})
    cell=ET.SubElement(parent,'mxCell',attrs)
    ET.SubElement(cell,'mxGeometry',{'x':str(x),'y':str(y),'width':str(w),'height':str(h),'as':'geometry'})
    return cell


def xml_file(all_pages, head):
    mx=ET.Element('mxfile',{'host':'app.diagrams.net','pages':str(len(all_pages)),'compressed':'false',
                          'vars':json.dumps({'branch':'alpha','commit':head,'snapshot':'2026-10-15'})})
    for page in all_pages:
        dia=ET.SubElement(mx,'diagram',{'id':page.id,'name':page.name})
        model=ET.SubElement(dia,'mxGraphModel',{'dx':'1600','dy':'1040','grid':'1','gridSize':'8','guides':'1',
            'tooltips':'1','connect':'1','arrows':'1','fold':'1','page':'1','pageScale':'1',
            'pageWidth':str(W),'pageHeight':str(H),'background':PAPER,'math':'0','shadow':'0'})
        root=ET.SubElement(model,'root')
        ET.SubElement(root,'mxCell',{'id':'0'})
        for id,label in [('zones','Panel boundaries'),('edges','Forward and objective connections'),('1','Editable modules'),('notes','Titles and annotations')]:
            ET.SubElement(root,'mxCell',{'id':id,'parent':'0','value':label})
        for i,(x,y,w,h,label) in enumerate(page.zones):
            c=ET.SubElement(root,'mxCell',{'id':f'zone-{i}','value':'','vertex':'1','parent':'zones',
                'style':'rounded=1;arcSize=4;html=1;fillColor=#ffffff;strokeColor=#cbd3da;strokeWidth=1;dashed=1;dashPattern=5 4;pointerEvents=0;'})
            ET.SubElement(c,'mxGeometry',{'x':str(x),'y':str(y),'width':str(w),'height':str(h),'as':'geometry'})
            text_cell(root,f'zone-label-{i}',x+20,y+8,w-40,28,label,14,bold=True,color=MUTED)
        for i,e in enumerate(page.edges):
            src,dst=page.nodes[e.source],page.nodes[e.target]
            x1,y1=src.port(e.out,e.a);x2,y2=dst.port(e.into,e.b)
            ex,ey=(x1-src.x)/src.w,(y1-src.y)/src.h
            ix,iy=(x2-dst.x)/dst.w,(y2-dst.y)/dst.h
            dashed=e.role!='forward'
            color=LOSS_INK if dashed else INK
            style=(f'edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;'
                f'endArrow=block;endFill=1;strokeWidth=1.6;strokeColor={color};'
                f'jumpStyle=arc;jumpSize=8;exitX={ex};exitY={ey};exitDx=0;exitDy=0;'
                f'entryX={ix};entryY={iy};entryDx=0;entryDy=0;exitPerimeter=0;entryPerimeter=0;'
                +('dashed=1;dashPattern=6 4;' if dashed else ''))
            c=ET.SubElement(root,'mxCell',{'id':f'edge-{i}','value':'','edge':'1','parent':'edges',
                                       'source':e.source,'target':e.target,'style':style})
            g=ET.SubElement(c,'mxGeometry',{'relative':'1','as':'geometry'})
            if e.points[1:-1]:
                a=ET.SubElement(g,'Array',{'as':'points'})
                for x,y in e.points[1:-1]:ET.SubElement(a,'mxPoint',{'x':str(x),'y':str(y)})
        for n in page.nodes.values():
            size=18 if page.id!='losses' else 14
            label=f'<b>{escape(n.title)}</b>'
            if n.sub:label+=f'<br><span style="font-size:14px">{escape(n.sub).replace(chr(10),"<br>")}</span>'
            style=(f'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor={COLORS[n.kind]};'
                f'strokeColor={INK};strokeWidth=1.3;fontColor={INK};fontFamily=Arial;fontSize={size};'
                'align=center;verticalAlign=middle;spacing=8;overflow=hidden;'
                +('dashed=1;dashPattern=5 3;' if n.frozen else ''))
            attrs={'id':n.id,'value':label,'vertex':'1','parent':'1','style':style}
            parent=root
            if n.link:
                parent=ET.SubElement(root,'UserObject',{'id':attrs.pop('id'),'label':attrs.pop('value'),'link':n.link})
            c=ET.SubElement(parent,'mxCell',attrs)
            ET.SubElement(c,'mxGeometry',{'x':str(n.x),'y':str(n.y),'width':str(n.w),'height':str(n.h),'as':'geometry'})
        text_cell(root,'title',64,28,1472,48,page.title,32,bold=True)
        text_cell(root,'subtitle',64,84,1472,40,page.subtitle,15,color=MUTED)
        for i,note in enumerate(page.notes):text_cell(root,f'note-{i}',64,892+i*32,1472,32,note,14,color=MUTED)
        if page.id=='losses':
            for i,(x,w,label) in enumerate(((80,280,'TERM'),(360,160,'WEIGHT'),(520,360,'APPLIES TO'),(880,640,'PREDICTION / TARGET'))):
                text_cell(root,f'table-header-{i}',x,196,w,20,label,12,align='center',bold=True,color=MUTED)
        if page.id=='guide':
            for i,(name,url) in enumerate((('HiGAN+ · original implementation','https://github.com/ganji15/HiGANplus'),
                                          ('SpiS-GAN · paper','https://arxiv.org/abs/2607.06949'),
                                          ('FW-GAN · paper','https://arxiv.org/abs/2508.21040'))):
                text_cell(root,f'paper-{i}',80+i*500,852,440,28,name,14,color=MUTED,link=url)
        text_cell(root,'footer',64,976,1472,24,f'ALPHA · {head[:10]} · {page.source} · 2026-10-15',12,color=MUTED)
        text_cell(root,'legend',64,1008,1352,20,'Solid arrow = forward dependency     Dashed brown = target / initialization / update     Dashed module border = parameter-frozen',12,color=MUTED)
        text_cell(root,'index-link',1440,1008,96,20,'↖ Index',12,align='right',link='data:page/id,guide')
    ET.indent(mx,space='  ')
    return ET.tostring(mx,encoding='utf-8',xml_declaration=True)


def wrapped_lines(text,width,size):
    # PIL measures the same offline font used by the PDF renderer.
    from PIL import ImageFont
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',int(size))
    result=[]
    for paragraph in text.split('\n'):
        line=''
        for word in paragraph.split():
            test=f'{line} {word}'.strip()
            if line and font.getlength(test)>width:
                result.append(line);line=word
            else:line=test
        result.append(line)
    return result


def display_graph(page):
    """An offline, accessible SVG preview derived from the native graph."""
    titleid,descid=f'{page.id}-title',f'{page.id}-desc'
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="{titleid} {descid}">',
         f'<title id="{titleid}">{escape(page.title)}</title>',
         f'<desc id="{descid}">{escape(page.subtitle)}. Solid arrows represent forward dependencies; dashed arrows represent targets or optimizer actions.</desc>',
         '<rect width="1600" height="1040" fill="white"/>']
    def text(x,y,label,size=14,anchor='start',bold=False,color=INK):
        weight=' font-weight="bold"' if bold else ''
        out.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}"{weight}>{escape(label)}</text>')
    for x,y,w,h,label in page.zones:
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="white" stroke="#cbd3da" stroke-dasharray="5 4"/>')
        text(x+20,y+28,label,14,bold=True,color=MUTED)
    # Crossings are bridged in draw.io via jumpStyle=arc; white under-strokes
    # make the same non-junction semantics visible in the static preview.
    for e in page.edges:
        color=INK if e.role=='forward' else LOSS_INK
        pts=e.points
        coords=' '.join(f'{x},{y}' for x,y in pts)
        dash='' if e.role=='forward' else ' stroke-dasharray="6 4"'
        out.append(f'<polyline points="{coords}" fill="none" stroke="white" stroke-width="5" stroke-linejoin="round"/>')
        out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="1.6" stroke-linejoin="round"{dash}/>')
        x,y=pts[-1];px,py=pts[-2];angle=math.atan2(y-py,x-px)
        back=[(x-10*math.cos(angle)+4*math.sin(angle),y-10*math.sin(angle)-4*math.cos(angle)),
              (x-10*math.cos(angle)-4*math.sin(angle),y-10*math.sin(angle)+4*math.cos(angle))]
        out.append(f'<polygon points="{x},{y} {back[0][0]},{back[0][1]} {back[1][0]},{back[1][1]}" fill="{color}"/>')
    for n in page.nodes.values():
        dash=' stroke-dasharray="5 3"' if n.frozen else ''
        out.append(f'<rect x="{n.x}" y="{n.y}" width="{n.w}" height="{n.h}" rx="6" fill="{COLORS[n.kind]}" stroke="{INK}" stroke-width="1.3"{dash}/>')
        ts=18 if page.id!='losses' else 14
        title=wrapped_lines(n.title,n.w-24,ts)
        sub=wrapped_lines(n.sub,n.w-24,14) if n.sub else []
        total=len(title)*ts*1.22+(len(sub)*18+6 if sub else 0)
        y=n.y+(n.h-total)/2+ts*.85
        for line in title:text(n.x+n.w/2,y,line,ts,'middle',True);y+=ts*1.22
        if sub:y+=4
        for line in sub:text(n.x+n.w/2,y,line,14,'middle');y+=18
    text(64,64,page.title,32,bold=True)
    text(64,108,page.subtitle,15,color=MUTED)
    for i,note in enumerate(page.notes):text(64,912+i*32,note,14,color=MUTED)
    if page.id=='losses':
        for x,w,label in ((80,280,'TERM'),(360,160,'WEIGHT'),(520,360,'APPLIES TO'),(880,640,'PREDICTION / TARGET')):
            text(x+w/2,210,label,12,'middle',True,MUTED)
    if page.id=='guide':
        for i,label in enumerate(('HiGAN+ · original implementation','SpiS-GAN · paper','FW-GAN · paper')):
            text(80+i*500,872,label,14,color=MUTED)
    text(64,992,f'ALPHA · {page.source} · snapshot 2026-10-15',12,color=MUTED)
    text(64,1024,'Solid = forward dependency     Dashed brown = target / update     Dashed module = parameter-frozen',12,color=MUTED)
    out.append('</svg>')
    return '\n'.join(out)


def validate(all_pages, data):
    problems=[];results=[]
    for p in all_pages:
        overlaps=[];hits=[];overflow=[]
        nodes=list(p.nodes.values())
        for i,a in enumerate(nodes):
            for b in nodes[i+1:]:
                if max(a.x,b.x)<min(a.x+a.w,b.x+b.w) and max(a.y,b.y)<min(a.y+a.h,b.y+b.h):overlaps.append([a.id,b.id])
            if not (40<=a.x and a.x+a.w<=1560 and 132<=a.y and a.y+a.h<=868):overflow.append(a.id)
            ts=18 if p.id!='losses' else 14
            lines=wrapped_lines(a.title,a.w-24,ts)
            sub=wrapped_lines(a.sub,a.w-24,14) if a.sub else []
            total=len(lines)*ts*1.22+(len(sub)*18+6 if sub else 0)
            if total>a.h-14:problems.append(f'{p.id}:{a.id}:text height {total:.1f} > {a.h-14}')
        for i,e in enumerate(p.edges):
            assert e.source in p.nodes and e.target in p.nodes
            for a,b in zip(e.points,e.points[1:]):
                if a[0]!=b[0] and a[1]!=b[1]:problems.append(f'{p.id}: diagonal edge {i}')
                for n in nodes:
                    if n.id not in (e.source,e.target) and interior_crossing(a,b,(n.x,n.y,n.w,n.h),pad=0):hits.append([i,n.id])
        if overlaps or hits or overflow:problems.append(f'{p.id}: overlaps={overlaps}; transit={hits}; overflow={overflow}')
        results.append({'page':p.name,'module_or_table_cells':len(nodes),'connectors':len(p.edges),
                        'node_overlaps':len(overlaps),'edge_through_unrelated_nodes':len(hits),'out_of_bounds':len(overflow)})
    root=ET.fromstring(data)
    pageids=[p.get('id') for p in root.findall('diagram')]
    assert len(set(pageids))==len(pageids)
    for page in root.findall('diagram'):
        cells=page.findall('.//mxCell')
        owners=[c for c in page.iter() if c.get('id') and c.tag!='diagram']
        ids={c.get('id') for c in owners}
        assert len(ids)==len(owners)
        for c in cells:
            for attr in ('source','target','parent'):
                if c.get(attr):assert c.get(attr) in ids,(page.get('name'),attr,c.get(attr))
        for c in page.findall('.//UserObject'):
            if c.get('link','').startswith('data:page/id,'):assert c.get('link').split(',')[1] in pageids
    assert b'data:image/' not in data and b'<image' not in data
    return {'ok':not problems,'problems':problems,'pages':results,
            'native_xml_pages':len(pageids),'image_embeds':0,
            'drawio_sha256':sha256(data).hexdigest(),
            'verification_scope':'XML, page references, native cells, explicit geometry, node overlap, orthogonality, text budgets. Browser editor import not yet exercised.'}


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    ps=pages()
    for p in ps:
        route(p)
        print(p.name,len(p.nodes),'nodes',len(p.edges),'edges',flush=True)
    data=xml_file(ps,head)
    check=validate(ps,data)
    (ROOT/'verification.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
    if not check['ok']:
        print('\n'.join(check['problems']))
        raise SystemExit(1)
    (ROOT/'alpha_training_architecture.drawio').write_bytes(data)
    # Small graph JSON is the deterministic preview source, not a second model.
    from dataclasses import asdict
    (ROOT/'diagram_graph.json').write_text(json.dumps({'head':head,'width':W,'height':H,'pages':[asdict(p) for p in ps]},ensure_ascii=False),encoding='utf-8')
    html=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
          '<title>ALPHA training atlas</title><style>body{margin:0;background:#e9edf0;color:#263746;font-family:Arial,sans-serif}nav{padding:20px;display:flex;gap:12px;flex-wrap:wrap}a{color:#263746}section{margin:24px 0;background:white}.diagram-container{width:100%;overflow-x:auto}svg{width:100%;min-width:1600px;display:block;font-family:Arial,DejaVu Sans,sans-serif}@media print{nav{display:none}section{break-after:page;margin:0}.diagram-container{overflow-x:visible}svg{min-width:0}body{background:white}}</style>',
          '<nav>'+''.join(f'<a href="#{p.id}">{escape(p.name)}</a>' for p in ps)+'</nav>']
    for p in ps:html.append(f'<section id="{p.id}"><div class="diagram-container">{display_graph(p)}</div></section>')
    html.append('</html>')
    (ROOT/'preview.html').write_text('\n'.join(html),encoding='utf-8')
    print('OK:',len(ps),'native pages;',len(data),'bytes')


if __name__=='__main__':main()
