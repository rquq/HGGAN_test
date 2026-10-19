"""Five end-to-end ALPHA panels, composed like the HiGAN+ overview.

The diagram is native editable mxGraph; prose and equations sit below its
colored panel. Reuses the tested routing/codec helpers, not the 15-page layout.
No model, configuration, checkpoint or notebook is changed.
"""
from pathlib import Path
from dataclasses import asdict
from html import escape
from hashlib import sha256
import importlib.util
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('atlas_core',ROOT.parent/'drawio'/'build_drawio.py')
core=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=core
spec.loader.exec_module(core)
W,H=1600,1480
INK='#202020'
PANEL_COLORS=['#e7edda','#eae5ef','#dceef3','#e3f0f3','#e9efdf']
FILLS={'data':'#ffffff','module':'#f1dddd','style':'#ded6e9',
       'loss':'#ffead7','critic':'#f1dddd','frozen':'#f1dddd'}


def make_pages():
    out=[]
    def page(id,name,title,subtitle):
        p=core.Page(id,name,title,subtitle)
        p.explain=[]
        out.append(p)
        return p
    def note(p,kind,text):p.explain.append((kind,text))
    def loss(p,*args,**kwargs):
        kwargs['role']='objective'
        p.e(*args,**kwargs)

    p=page('a','(a) GAN training','(a) Training with disentangled representations',
           'ALPHA · same forward graph at 32 px / 64 px · details live inside modules, not in extra detached insets')
    p.n('text',64,184,220,'Text targets','y′: sampled target text\ny: reference transcript',kind='data',h=104)
    p.n('prior',552,184,244,'Random style z','1 global + 7 centered locals\n8 × 32 dimensions',kind='style',h=104)
    p.n('sample',64,528,192,'Real sample','Reference x_ref; target x\nTranscript y; writer ID w',kind='data',h=144)
    p.n('b',312,528,184,'B','Pretrained residual CNN\nFrozen in GAN',h=144,frozen=True)
    p.n('e',552,508,244,'E','Global: context → q(g|x_ref)\nLocal: 7 evidence queries\nS_ref = [g, L]',h=164)
    p.n('g',852,384,252,'G','Text context → content seed\ng → block-specific CBN\nL → spatial cross-attention\n4 image-generation stages',h=224)
    p.n('fake',1160,420,212,'Generated words','x_rand = G(z, y′)\nx_style = G(S_ref, y′)\nx_rec = G(S_ref, y)',kind='data',h=144)
    p.n('real',312,348,184,'Real word x','Canonical training image',kind='data',h=96)
    p.n('d',1160,184,176,'D','DiffAug + whole-word critic\nAll three fake groups',kind='critic',h=120)
    p.n('adv',1400,196,136,'Adv','Real/fake hinge\nWord realism',kind='loss',h=96)
    p.n('r',1160,620,176,'R','CNN + BiLSTM\nrand + style only',h=96,frozen=True)
    p.n('ctc',1400,620,136,'CTC','Prediction vs y′\nNo CTC on x_rec',kind='loss',h=96)
    p.n('wb',1160,748,176,'W + B','Frozen writer teacher\nTransfer image only',h=100,frozen=True)
    p.n('ce',1400,748,136,'CE','Writer CE\nvs writer w',kind='loss',h=100)
    p.e('sample','b');p.e('b','e');p.e('e','g',b=.86)
    p.e('text','g',out='B',into='T',a=.3,b=.25)
    p.e('prior','g',out='B',into='T',b=.75)
    p.e('g','fake')
    p.e('sample','real',out='T',into='L',a=.5)
    p.e('real','d',out='R',into='L')
    p.e('fake','d',out='T',into='B',a=88/212)
    loss(p,'d','adv')
    p.e('fake','r',out='B',into='T',a=88/212)
    loss(p,'r','ctc')
    loss(p,'text','ctc',out='B',into='T',a=.75)
    p.e('fake','wb',out='B',into='T',a=.8)
    loss(p,'wb','ce')
    loss(p,'sample','ce',out='B',into='B',a=.25)
    note(p,'h','Forward flow and supervision')
    note(p,'p','The same G generates random, reference-transfer and reconstruction words in one batched forward. D sees all three; R sees only random/transfer; W+B sees only transfer. The reference and canonical target x may have different widths.')
    note(p,'eq','L_D = E[(1 − D(x))₊] + (1/3) Σ_b E[(1 + D(stop(x_b)))₊] + L_R1,    b ∈ {rand, style, rec}.')
    note(p,'eq','L_adv = −(1/3) Σ_b E[D(x_b)];    L_CTC = λ_r CTC(R(x_rand), y′) + λ_s CTC(R(x_style), y′);    L_writer = CE(W(B(x_style)), w).')
    note(p,'eq','L_GE = L_adv + 0.45 L_adv,P + L_CTC + 0.5 L_writer + λ_rec L_rec + L_info + L_cycle + 0.02 L_content + 0.1 L_ctx + 0.1 L_KL.')
    note(p,'p','D/P update every batch; G/E update every second critic batch. In the G/E phase, D/P/B/R/W parameters are frozen but input gradients remain. R and W+B are pretrained on real images; W+B starts from the OCR CNN. See (b)/(c) for the other losses.')
    note(p,'p','Normal config: λ_r 1→0.4 (epochs 10–20), λ_s 0.5→0.20 (8–28), λ_rec 10→2.5 (24–42). Lazy R1 = (0.01 × 32 / 2) E||∇_x D(x)||² every 32 critic batches. D includes the same mild differentiable augmentation policy for real and fake.')

    p=page('b','(b) Reconstruction','(b) Regularization with reconstruction',
           'Image reconstruction and two global-code recovery paths share G/E; the local tokens are deterministic and have no KL term')
    p.n('sample',64,400,184,'Real sample','x_ref; canonical x\nTranscript y; writer w',kind='data',h=128)
    p.n('be',304,360,248,'B → E','B: frozen reference features\nE: global posterior + local tokens\nmu_g, logvar_g, L',h=208)
    p.n('g',616,360,264,'G','Shared Generator\nReference S with y → x_rec\nReference S with y′ → x_style\nPrior z with y′ → x_rand',h=208)
    p.n('text',616,184,264,'Text targets','y: reconstruction\ny′: random and transfer',kind='data',h=104)
    p.n('prior',64,660,184,'Random prior z','Global N(0,I)\n7 centered local slots',kind='style',h=104)
    p.n('rec',936,244,192,'x_rec','Reconstructed image',kind='data',h=88)
    p.n('style',936,428,192,'x_style','Same style; new text y′',kind='data',h=88)
    p.n('rand',936,660,192,'x_rand','Random style; text y′',kind='data',h=88)
    p.n('canonical',1184,184,184,'Real target x','Canonical valid image area',kind='data',h=88)
    p.n('pixel',1416,244,120,'L1_rec','x_rec vs x',kind='loss',h=88)
    p.n('eg1',1184,428,184,'B → E_g','Global-only E\nB frozen; E learns',h=88)
    p.n('cycle',1416,428,120,'L1_cycle','vs stop(mu_g,ref)',kind='loss',h=88)
    p.n('eg2',1184,660,184,'B → E_g','Global-only E\nSame B / E weights',h=88)
    p.n('info',1416,660,120,'L1_info','vs stop(z_g)',kind='loss',h=88)
    p.n('kl',304,184,248,'KL','Global posterior vs N(0,I)\nNo local posterior penalty',kind='loss',h=104)
    p.n('content',64,184,184,'Content BCE','GRL → learned probe\nvs bag of characters in y',kind='loss',h=112)
    p.e('sample','be');p.e('be','g');p.e('text','g',out='B',into='T')
    p.e('prior','g',out='R',into='B',b=.25)
    p.e('g','rec',a=.2);p.e('g','style',a=.5);p.e('g','rand',a=.8)
    loss(p,'rec','pixel');loss(p,'canonical','pixel',out='R',into='T')
    p.e('sample','canonical',out='T',into='L',a=.75)
    p.e('style','eg1');loss(p,'eg1','cycle')
    p.e('rand','eg2');loss(p,'eg2','info')
    loss(p,'be','cycle',out='T',into='T',a=.75)
    loss(p,'prior','info',out='B',into='B',a=.75)
    loss(p,'be','kl',out='T',into='B',a=.5)
    loss(p,'be','content',out='T',into='R',a=.25)
    loss(p,'sample','content',out='T',into='B',a=.5)
    note(p,'h','What is reconstructed — and what is not')
    note(p,'p','Pixel L1 compares x_rec to the canonical real image x over valid pixels, not to the original-width style reference. Global information recovery reconstructs the prior z_g; transfer cycle recovers the reference global mean. These are separate targets, not three copies of one loss.')
    note(p,'eq','L_rec = mean_valid |x_rec − x|;    L_info = mean |E_g(B(x_rand)) − stop(z_g)| / 8;    L_cycle = mean |E_g(B(x_style)) − stop(mu_g,ref)| / 8.')
    note(p,'eq','L_KL = KL[N(mu_g, diag(exp(logvar_g))) || N(0,I)] / 8;    g = mu_g + exp(0.5 logvar_g) ⊙ ε;    ε ∼ N(0,I);    ℓ_g = log σ_g².')
    note(p,'eq','L_content = balanced-BCE(Probe(GRL(mu_g,ref)), bag(y)) / 8;    GRL is identity forward and reverses only the gradient toward E.')
    note(p,'p','The probe learns reference character presence while E is pushed to hide it in g. Local L is a deterministic, centered 7-token residual; it is neither KL-regularized nor reconstructed against random local slots. E_g skips local retrieval. All repeated B/E/G blocks share weights.')
    note(p,'p','These three generated branches also receive the global and patch adversarial losses in (a)/(c). Only x_rand and x_style receive OCR CTC. The /8 factors preserve the explicit eight-token normalization implemented in the current loss functions.')

    p=page('c','(c) Local refinement','(c) Refining local patches and reference appearance',
           'A complete real/reference → G → generated-image → StrokePatchD/contextual flow; no separate texture generator is present')
    p.n('sample',64,500,184,'Real sample','Reference x_ref\nCanonical real x; transcript y',kind='data',h=128)
    p.n('be',304,480,248,'B → E','Cached real B features\nS_ref = [g, L] → G',h=160)
    p.n('text',304,184,248,'Text targets','y′ for random / transfer\ny for reconstruction',kind='data',h=104)
    p.n('prior',64,184,184,'Random prior z','1 global + 7 locals',kind='style',h=104)
    p.n('g',608,432,264,'G','Text context + content seed\nGlobal CBN + local attention\nGenerate all 3 fake branches',h=184)
    p.n('fake',928,400,216,'Generated images','x_rand · x_style · x_rec\nNative height; valid widths',kind='data',h=136)
    p.n('real',608,184,264,'Real image x','Canonical valid-width image',kind='data',h=104)
    p.n('crop_r',928,184,216,'Stratified crops','s = H/2; K = ceil(W/s)\nK clamped to [4,8]',kind='data',h=112)
    p.n('crop_f',1200,400,176,'Same crop policy','All 3 fake branches\nIndependent coordinates',kind='data',h=136)
    p.n('p',1200,184,176,'P','StrokePatchD\nSpatial patch logits',kind='critic',h=112)
    p.n('adv',1432,192,104,'Adv_P','Patch hinge',kind='loss',h=96)
    p.n('bf',928,704,216,'B','Frozen multiscale features\nx_style only',h=112,frozen=True)
    p.n('ctx',1200,704,176,'CTX','Fake B maps vs\nstop(reference B maps)',kind='loss',h=112)
    p.e('sample','be');p.e('be','g');p.e('text','g',out='B',into='T',a=.65,b=.75)
    p.e('prior','g',out='B',into='T',b=.25)
    p.e('sample','real',out='T',into='L',a=.25)
    p.e('real','crop_r');p.e('crop_r','p')
    p.e('g','fake');p.e('fake','crop_f');p.e('crop_f','p',out='T',into='B')
    loss(p,'p','adv')
    p.e('fake','bf',out='B',into='T',a=.4)
    loss(p,'bf','ctx')
    loss(p,'be','ctx',out='B',into='B',a=.6)
    note(p,'h','One crop policy; two complementary supervision signals')
    note(p,'p','Crop side s = H/2 (16 at H32, 32 at H64). Horizontal start positions are stratified; vertical locations alternate upper/lower halves. Real and all fake branches use the same policy, not identical coordinates. Only P receives these crops; global D always receives complete words.')
    note(p,'eq','K_n = clamp(ceil(W_n / (H/2)), 4, 8);    L_P = HingeMap(P(C(x)), +1) + (1/3) Σ_b HingeMap(P(C(stop(x_b))), −1).')
    note(p,'eq','L_adv,P = −(1/3) Σ_b MeanWord[MeanCrop[MeanMap[P(C(x_b))]]];    generator contribution = 0.45 L_adv,P.')
    note(p,'eq','L_ctx = Σ_l CX(stop(B_l(x_ref)), B_l(x_style));    generator contribution = 0.1 L_ctx;    l ∈ {three selected backbone scales}.')
    note(p,'p','P: 3×3 stem (1→64) → three residual StrokePatchBlocks (128→192→192; 3×3 plus axial DW 1×5 / 5×1 and 1×1 fusion) → spatial logit head. Pooling preserves a minimum 4×4 grid. Hinge is applied BEFORE map/crop reduction, then each word and fake branch is weighted equally.')
    note(p,'p','CTX matches non-aligned multiscale writer features to new text, with at most 256 sampled spatial features per comparison. Cached B maps are reused. The context comparison is not a reconstruction pixel target, frequency loss, foreground mask or independent texture-refinement network.')

    p=page('d','(d) Random-style inference','(d) Testing with random styles',
           'No reference image is needed · only the Generator and a sampled style prior are required at inference')
    p.n('text',64,272,224,'Target text y′','Supported character sequence\nExample: “there”',kind='data',h=112)
    p.n('content',344,272,264,'Text representation','120-D embedding per character\nRelative-position self-attention\nGated local FFN, DW k5',h=144)
    p.n('prior',64,544,224,'Sample N(0,I)','z_g: 1 × 32\nz_L: 7 × 32',kind='style',h=144)
    p.n('pack',344,544,264,'Style z = [z_g, L]','L_j = z_L,j − mean_k z_L,k\nCenter over the 7 local slots',kind='style',h=144)
    p.n('g',688,272,456,'G','Content-only spatial seed: 512 × 4 × (4T)\n\nGlobal z_g → four block-specific style MLPs → CBN\n\nGBlock 1 → spatial style attention(Q=features, K/V=locals)\n→ GBlocks 2–4 → BN / convolution / tanh',h=416)
    p.n('out',1208,416,328,'Generated image x_rand','H × (character_width × T)\nNo D, P, R or writer classifier',kind='data',h=144)
    p.e('text','content');p.e('prior','pack')
    p.e('content','g',b=.24);p.e('pack','g',b=.8)
    p.e('g','out')
    note(p,'h','Random-style forward pass')
    note(p,'p','Text establishes spatial content before local style is injected. The same style layout is used by random and reference-guided generation: one global token for block-wide conditioning and seven centered local tokens for spatial appearance. No character-specific fallback or single-digit switch is used.')
    note(p,'eq','z_g ∼ N(0, I_32);    Z_L ∼ N(0, I) with shape 7 × 32;    L_j = Z_L,j − (1/7) Σ_k Z_L,k;    x_rand = G([z_g,L], y′).')
    note(p,'eq','Spatial refinement: Q = W_Q LN(F);    K = W_K LN(L);    V = W_V L;    F_out = F + α ⊙ W_O Attention(Q,K,V).')
    note(p,'p','Content attention and spatial style attention each use 4 heads and attention width 128. Spatial style refinement happens once after the first GBlock. α is learned per feature channel (initialized at 0.25), not a hardcoded character gate.')
    note(p,'p','Both native heights retain four GBlocks (512→256→128→64→64 channels). Feature heights are 4→8→16→32→32 at H32, or 4→8→16→32→64 at H64. The final stage refines at constant resolution or upsamples according to requested geometry.')
    note(p,'eq','T = valid target character count;    output width = 16T at H32 or 32T at H64.    Inference: no loss, backward pass or critic optimizer step.')

    p=page('e','(e) Reference-style inference','(e) Testing with reference-guided styles',
           'Reference image → pretrained B → learned E → G → generated word · the target transcript need not match the reference transcript')
    p.n('ref',64,420,184,'Style reference','x_ref + valid width\nAny supported word image',kind='data',h=144)
    p.n('b',304,420,216,'B','Pretrained residual backbone\n3 scales: 64 / 128 / 256\nFrozen feature extractor',h=144,frozen=True)
    p.n('e',576,372,328,'E','Global: two gated context blocks\n→ valid mean / LN / MLP → mu_g\n\nLocal: multiscale 4-row evidence\n→ 7 learned-query cross-attention slots\n→ 32-D projection + slot centering',h=240)
    p.n('g',968,372,320,'G','Target text → content context / seed\nGlobal g → block-specific CBN\nLocal L → spatial style attention\n4 residual image stages → tanh',h=240)
    p.n('text',968,184,320,'Target text y′','New content, independent of reference text',kind='data',h=104)
    p.n('out',1352,420,184,'Styled output','x_style = G(S_ref, y′)\nReference writer appearance',kind='data',h=144)
    p.e('ref','b');p.e('b','e');p.e('e','g');p.e('g','out')
    p.e('text','g',out='B',into='T')
    note(p,'h','A learned global/local representation, not a reference-length rule')
    note(p,'p','B supplies multiscale visual evidence. E separates global pooled style from local residual evidence. During deterministic evaluation, g = mu_g; the seven local tokens are always deterministic. Reference text labels and writer IDs are not inputs to this inference path.')
    note(p,'eq','g = mu_g(B(x_ref));    L = CenterSlots[W_out Attention(Q_learned, K_visual+2Dposition, V_visual−valid_mean)];    S_ref = [g,L] ∈ R^(8×32).')
    note(p,'eq','x_style = G(S_ref, y′);    if y′ = y and the target geometry matches the training reconstruction target, this becomes the reconstruction path.')
    note(p,'p','The global branch uses two residual gated DW k7 context blocks with dilations 1 and 2. Local evidence uses three B scales pooled to four height rows, projected to 256 channels, and queried by seven orthogonally initialized learned slots with four attention heads. Positions address evidence through K; they are not added to appearance V.')
    note(p,'p','Configured GAN sampling/evaluation uses EMA G/E (beta 0.9998) and fixed pretrained B. W, R, D and P are unnecessary for generation. FID/KID/HWD are downstream measurements, not inputs to this graph or objectives backpropagated during evaluation.')
    note(p,'p','Shared notation: x_ref = original reference geometry; x = canonical real reconstruction target; y = its transcript; y′ = requested text; w = training writer ID. The diagrams summarize normal gan_iam_32/64 configurations, not MAIN’s separate token-capacity experiment.')
    return out


def math_markup(value):
    value=escape(value)
    tokens={
        'mu_g,ref':'μ<sub>g,ref</sub>', 'logvar_g':'ℓ<sub>g</sub>', 'mu_g':'μ<sub>g</sub>',
        'ℓ_g':'ℓ<sub>g</sub>', 'σ_g':'σ<sub>g</sub>',
        'L_adv,P':'L<sub>adv,P</sub>', 'L_adv':'L<sub>adv</sub>',
        'L_CTC':'L<sub>CTC</sub>', 'L_writer':'L<sub>writer</sub>',
        'L_GE':'L<sub>G,E</sub>', 'L_R1':'L<sub>R1</sub>', 'L_D':'L<sub>D</sub>',
        'L_rec':'L<sub>rec</sub>', 'L_info':'L<sub>info</sub>', 'L_cycle':'L<sub>cycle</sub>',
        'L_content':'L<sub>content</sub>', 'L_ctx':'L<sub>ctx</sub>', 'L_KL':'L<sub>KL</sub>',
        'L_P':'L<sub>P</sub>', 'L_j':'L<sub>j</sub>',
        'x_rand':'x<sub>rand</sub>', 'x_style':'x<sub>style</sub>', 'x_rec':'x<sub>rec</sub>',
        'x_ref':'x<sub>ref</sub>', 'x_b':'x<sub>b</sub>',
        'E_g':'E<sub>g</sub>', 'B_l':'B<sub>l</sub>', 'S_ref':'S<sub>ref</sub>',
        'z_g':'z<sub>g</sub>', 'z_L,j':'z<sub>L,j</sub>', 'z_L,k':'z<sub>L,k</sub>',
        'Z_L,j':'Z<sub>L,j</sub>', 'Z_L,k':'Z<sub>L,k</sub>', 'Z_L':'Z<sub>L</sub>',
        'W_n':'W<sub>n</sub>', 'K_n':'K<sub>n</sub>', 'I_32':'I<sub>32</sub>',
        'W_out':'W<sub>out</sub>', 'W_Q':'W<sub>Q</sub>', 'W_K':'W<sub>K</sub>',
        'W_V':'W<sub>V</sub>', 'W_O':'W<sub>O</sub>', 'F_out':'F<sub>out</sub>',
        'Q_learned':'Q<sub>learned</sub>', 'K_visual+2Dposition':'K<sub>visual+2Dpos</sub>',
        'V_visual−valid_mean':'V<sub>visual−valid mean</sub>',
        'λ_rec':'λ<sub>rec</sub>', 'λ_r':'λ<sub>r</sub>', 'λ_s':'λ<sub>s</sub>',
        'Σ_b':'∑<sub>b</sub>', 'Σ_k':'∑<sub>k</sub>', 'Σ_l':'∑<sub>l</sub>',
        '∇_x':'∇<sub>x</sub>', 'mean_valid':'mean<sub>valid</sub>',
        'R^(8×32)':'ℝ<sup>8×32</sup>',
    }
    return re.sub('|'.join(re.escape(k) for k in sorted(tokens,key=len,reverse=True)),
                  lambda match:tokens[match.group()],value)


def text(root,id,x,y,w,h,value,size=16,bold=False,formula=False):
    font='Times New Roman' if formula else 'Arial'
    label=math_markup(value) if formula else escape(value).replace('\n','<br>')
    c=ET.SubElement(root,'mxCell',{'id':id,'value':label,'vertex':'1','parent':'notes',
      'style':f'text;html=1;whiteSpace=wrap;overflow=hidden;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;spacing=0;fontFamily={font};fontSize={size};fontStyle={1 if bold else 0};fontColor={INK};'})
    ET.SubElement(c,'mxGeometry',{'x':str(x),'y':str(y),'width':str(w),'height':str(h),'as':'geometry'})


def annotation_layout(p):
    y=916
    result=[]
    for kind,value in p.explain:
        size=19 if kind=='h' else 19 if kind=='eq' else 16
        lines=core.wrapped_lines(value,1472,size)
        lineheight=26 if kind=='h' else 27 if kind=='eq' else 23
        height=len(lines)*lineheight+8
        result.append((kind,value,y,height,size))
        y+=height+8
    if y>1430:raise ValueError(f'{p.id}: explanation overflow ends at {y}')
    return result


def native_file(ps,head):
    xml=ET.fromstring(core.xml_file(ps,head))
    for i,(p,dia) in enumerate(zip(ps,xml.findall('diagram'))):
        m=dia.find('mxGraphModel');m.set('pageHeight',str(H));m.set('dy',str(H))
        root=m.find('root')
        # Drop the old atlas-only navigation and relocate the source/legend.
        for elem in list(root):
            if elem.get('id') in ('index-link',):root.remove(elem)
        background=ET.Element('mxCell',{'id':'paper-panel','value':'','vertex':'1','parent':'zones',
             'style':f'rounded=0;html=1;fillColor={PANEL_COLORS[i]};strokeColor=none;pointerEvents=0;'})
        ET.SubElement(background,'mxGeometry',{'x':'48','y':'124','width':'1504','height':'744','as':'geometry'})
        root.insert(5,background)
        for elem in root.findall('mxCell'):
            if elem.get('id')=='title':
                elem.set('style',elem.get('style').replace('fontSize=32','fontSize=30'))
            elif elem.get('id')=='subtitle':
                elem.set('style',elem.get('style').replace('fontSize=15','fontSize=14'))
            elif elem.get('id')=='footer':
                elem.find('mxGeometry').set('y','1440')
                elem.set('value',escape(f'ALPHA · {head[:10]} · networks/model.py / module.py / fusion.py / BigGAN_networks.py · 2026-10-15'))
            elif elem.get('id')=='legend':
                elem.find('mxGeometry').set('y','880')
                elem.find('mxGeometry').set('width','1472')
                elem.set('value','Solid = forward tensors     Dashed orange = loss / target dependency     Dashed module border = frozen parameters in the depicted phase')
            elif elem.get('id') in p.nodes:
                n=p.nodes[elem.get('id')]
                title_size=28 if n.title in ('G','E','B','D','R','P') else 19
                label=f'<b style="font-size:{title_size}px">{escape(n.title)}</b>'
                if n.sub:label+='<br><span style="font-size:14px;line-height:1.25">'+escape(n.sub).replace('\n','<br>')+'</span>'
                elem.set('value',label)
                style=elem.get('style')
                style=re.sub(r'fillColor=#[0-9a-fA-F]+',f'fillColor={FILLS[n.kind]}',style)
                style=re.sub(r'strokeColor=#[0-9a-fA-F]+',f'strokeColor={INK}',style)
                elem.set('style',style.replace('arcSize=8','arcSize=3'))
        # No extra diagram nodes: explanatory prose is native text below it.
        for j,(kind,value,y,h,size) in enumerate(annotation_layout(p)):
            text(root,f'explanation-{j}',64,y,1472,h,value,size,bold=kind=='h',formula=kind=='eq')
    ET.indent(xml,space='  ')
    return ET.tostring(xml,encoding='utf-8',xml_declaration=True)


def connectivity(ps):
    report=[]
    for p in ps:
        adjacent={id:set() for id in p.nodes}
        for e in p.edges:
            adjacent[e.source].add(e.target);adjacent[e.target].add(e.source)
        unseen=set(adjacent);components=[]
        while unseen:
            pending=[next(iter(unseen))];found=set()
            while pending:
                n=pending.pop()
                if n in found:continue
                found.add(n);pending.extend(adjacent[n]-found)
            unseen-=found;components.append(sorted(found))
        report.append({'page':p.name,'diagram_components':len(components),'disconnected_blocks':[k for k,v in adjacent.items() if not v],
                       'nodes':len(p.nodes),'edges':len(p.edges)})
        assert len(components)==1,(p.name,components)
    return report


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    ps=make_pages()
    for p in ps:
        core.route(p)
        print(p.name, len(p.nodes),'nodes',len(p.edges),'connections',flush=True)
    head=subprocess.check_output(['git','-C',str(core.REPO),'rev-parse','HEAD'],text=True).strip()
    data=native_file(ps,head)
    check=core.validate(ps,data)
    check['end_to_end']=connectivity(ps)
    check['description']='Exactly five HiGAN+-inspired end-to-end ALPHA panels; formulas/prose beneath each diagram.'
    (ROOT/'verification.json').write_text(json.dumps(check,indent=2,ensure_ascii=False),encoding='utf-8')
    if not check['ok']:
        print('\n'.join(check['problems']));raise SystemExit(1)
    (ROOT/'alpha_training_architecture.drawio').write_bytes(data)
    (ROOT/'diagram_graph.json').write_text(json.dumps({'head':head,'width':W,'height':H,'pages':[dict(asdict(p),explanations=p.explain) for p in ps]},ensure_ascii=False),encoding='utf-8')
    # Preview is populated with exact native-renderer screenshots after codec
    # verification; the primary artifact does not depend on those screenshots.
    print('OK:',len(ps),'pages; all diagram blocks belong to one connected component per page')


if __name__=='__main__':main()
