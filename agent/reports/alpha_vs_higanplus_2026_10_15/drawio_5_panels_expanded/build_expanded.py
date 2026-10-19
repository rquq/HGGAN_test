"""Five wide paper panels with native, connected mini-architectures.

Only documentation changes. Source-of-truth: ALPHA current code. The
reference-guided training view (a) intentionally does not use a random prior.
"""
from pathlib import Path
from dataclasses import asdict
from html import escape
from hashlib import sha256
import importlib.util
import json
import math
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import comparison_panel
spec=importlib.util.spec_from_file_location('five_compact',ROOT.parent/'drawio_5_panels'/'build_five_panels.py')
compact=importlib.util.module_from_spec(spec);sys.modules[spec.name]=compact;spec.loader.exec_module(compact)
core=compact.core
W,H=2880,1960
TOP,BOTTOM=140,1240
INK='#202020'


def snap(v):return int(round(v/4)*4)


def crossing_count(edges):
    crossings=set()
    for i,e in enumerate(edges):
        for j,f in enumerate(edges[i+1:],i+1):
            for a,b in zip(e.points,e.points[1:]):
                for c,d in zip(f.points,f.points[1:]):
                    if a[0]==b[0] and c[1]==d[1]:
                        x,y=a[0],c[1]
                        hit=min(a[1],b[1])<y<max(a[1],b[1]) and min(c[0],d[0])<x<max(c[0],d[0])
                    elif a[1]==b[1] and c[0]==d[0]:
                        x,y=c[0],a[1]
                        hit=min(c[1],d[1])<y<max(c[1],d[1]) and min(a[0],b[0])<x<max(a[0],b[0])
                    else:hit=False
                    if hit:crossings.add((i,j,x,y))
    return len(crossings)


def make_pages():
    ps=compact.make_pages()
    for p in ps:
        p.minis={}
        # The main (a) panel is specifically reference-guided, like HiGAN+.
        # Keep the full three-branch objective in the explanatory text.
        if p.id=='a':
            del p.nodes['prior']
            p.edges=[e for e in p.edges if e.source!='prior' and not(e.source=='sample' and e.target=='real')]
            p.nodes['text'].title='Target text y′';p.nodes['text'].sub='New requested content'
            p.nodes['fake'].title='Style-transfer word';p.nodes['fake'].sub='x_style = G(S_ref, y′)\nStyle comes from the reference'
            p.nodes['d'].sub='Shared whole-word critic\nReal x / transfer x_style'
            p.nodes['r'].sub='Frozen CNN + BiLSTM\nTransfer prediction'
            p.nodes['g'].sub='Reference-guided branch only'
            # Repeat ground-truth labels beside the objective instead of
            # carrying target wires across the whole architecture.
            p.edges=[e for e in p.edges if not ((e.source=='text' and e.target=='ctc') or (e.source=='sample' and e.target=='ce'))]
            p.n('ctc_target',1400,484,136,'Target y′','Copied text label',kind='data',h=72)
            p.n('writer_target',1400,904,136,'Writer ID w','From the real sample',kind='data',h=72)
            p.e('ctc_target','ctc',out='B',into='T',role='objective')
            p.e('writer_target','ce',out='T',into='B',role='objective')
            p.title='(a) Reference-guided GAN training'
            p.subtitle='Style comes from x_ref → B → E, not from a random z · the shared full GAN also trains random and reconstruction branches'
            p.explain[1]=('p','This panel isolates the reference-guided training path: x_ref → B → E → S_ref → G(S_ref,y′) → D/R/W+B → losses. There is NO random-style input in this path. The same shared G/D/P also train on random x_rand and reconstruction x_rec, shown in (b)/(c)/(d); the equations below describe that full objective.')
            p.explain.append(('p','Repeated target y′ and writer ID w boxes are the same ground-truth batch labels, not additional inputs to E or G. They are placed beside CTC/CE to keep supervision wires short.'))
        if p.id=='b':
            # Real x is a ground-truth field, not the output of a learned
            # x_ref-to-x transform. Copy the detached reference target near the
            # cycle loss to avoid a long diagonal dependency across the page.
            p.edges=[e for e in p.edges if not ((e.source=='sample' and e.target=='canonical') or (e.source=='be' and e.target=='cycle'))]
            p.n('mu_target',1416,356,120,'Reference μ_g','Copied detached target',kind='data',h=56)
            p.e('mu_target','cycle',out='B',into='T',role='objective')
            p.explain.append(('p','The nearby μ_g target is a copy of E’s reference posterior mean, not a new encoder or trainable module. Repeating this target avoids a wire crossing the image-generation paths.'))
        if p.id=='c':
            p.edges=[e for e in p.edges if not(e.source=='sample' and e.target=='real')]
        for n in p.nodes.values():
            n.x=snap(n.x*1.8);n.y=snap(n.y*1.22)
            n.w=snap(n.w*1.8);n.h=snap(n.h*1.22)
        for id in ('e','be'):
            if id in p.nodes:
                p.nodes[id].h=320 if p.nodes[id].w<560 else 380
        if 'g' in p.nodes:p.nodes['g'].h=448
        for id in ('eg1','eg2'):
            if id in p.nodes:p.nodes[id].h=180
        for id in ('wb','d'):
            if id in p.nodes:p.nodes[id].h=264
        if 'p' in p.nodes:p.nodes['p'].h=300
        for id in ('crop_r','crop_f','prior','pack','content'):
            if id in p.nodes and p.nodes[id].kind!='loss':p.nodes[id].h=264
        if p.id=='c':p.nodes['crop_f'].y=580
        if p.id=='d':p.nodes['g'].h=480
        for edge in p.edges:
            if edge.target=='g' and edge.source in ('e','be'):edge.into='L';edge.b=.70
            if p.id=='a' and edge.source=='real' and edge.target=='d':edge.into='L';edge.b=.22
        for id in ('e','be','g','eg1','eg2','p','crop_r','crop_f','wb','d','prior','pack','content'):
            if id in p.nodes and p.nodes[id].kind!='loss':p.minis[id]=Mini(p.nodes[id],p)
        # Header bands stay clear: top-boundary ports occupy the narrow corner
        # rails, then turn toward the actual inner operator.
        for id in p.minis:
            for attr,sideattr,frac in (('source','out','a'),('target','into','b')):
                top_edges=[e for e in p.edges if getattr(e,attr)==id and getattr(e,sideattr)=='T']
                for j,e in enumerate(top_edges):setattr(e,frac,.02 if j%2==0 else .98)
        core.route(p,width=W,top=TOP,bottom=BOTTOM,clearance=10,obstacle_pad=4,cross_penalty=8000)
        for i,e in enumerate(p.edges):
            if e.source in p.minis:p.minis[e.source].bind(i,e,'source')
            if e.target in p.minis:p.minis[e.target].bind(i,e,'target')
        for mini in p.minis.values():mini.finish()
    return ps


class Mini:
    def __init__(self,parent,page):
        self.parent,self.page=parent,page
        self.graph=core.Page('mini','mini','','')
        self.binding={}
        self.kind=('encoder' if parent.id in ('e','be') else 'global' if parent.id.startswith('eg') else
                   'generator' if parent.id=='g' else 'patch' if parent.id=='p' else
                   'crop' if parent.id.startswith('crop') else 'writer' if parent.id=='wb' else
                   'critic' if parent.id=='d' else 'rng' if parent.id=='prior' else
                   'center' if parent.id=='pack' else 'content')
        self.title={'encoder':'E · Global–local style encoder [C1]',
                    'global':'B → E_global · global-only path',
                    'generator':'G · Content-first style fusion [C2]',
                    'patch':'P · StrokePatchD [C3]',
                    'crop':'Height-relative crop policy [C3]',
                    'writer':'W + B · attentive statistics (supporting change)',
                    'critic':'D · whole-word critic (supporting change)',
                    'rng':'Random-number generator → prior z',
                    'center':'Pack the sampled style prior',
                    'content':'ContentContext · text context [C2]'}[self.kind]
        self.graph.zone(0,4,parent.w,40,'Header band ' * 100)
        self.nodes()

    def n(self,id,x,y,w,h,title,sub='',kind='module'):
        self.graph.n(id,snap(x),snap(y),snap(w),title,sub,kind=kind,h=snap(h))

    def e(self,a,b,out=None,into=None,role='forward'):
        a0,b0=self.graph.nodes[a],self.graph.nodes[b]
        dx=(b0.x+b0.w/2)-(a0.x+a0.w/2);dy=(b0.y+b0.h/2)-(a0.y+a0.h/2)
        if out is None:out=('R' if dx>=0 else 'L') if abs(dx)>=abs(dy) else ('B' if dy>=0 else 'T')
        if into is None:into=('L' if dx>=0 else 'R') if abs(dx)>=abs(dy) else ('T' if dy>=0 else 'B')
        self.graph.e(a,b,out,into,role=role)

    def nodes(self):
        w,h=self.parent.w,self.parent.h
        if self.kind=='encoder':
            self.n('maps',20,h/2-28,88,64,'B maps','3 scales' if self.parent.id=='e' else 'Frozen B')
            self.n('global',132,64,w-264,80,'Global context','DW context → valid pool\nμ → g (deterministic)' if self.page.id=='e' else 'DW context → valid pool\nμ / logσ² → sampled g')
            self.n('local',132,h-92,w-264,72,'Local evidence','3 scales → 7-query MHA\n32-D projection → center')
            self.n('pack',w-112,h/2-44,92,88,'S_ref','[g, L]\n8 × 32',kind='style')
            self.e('maps','global','R','L');self.e('maps','local','R','L')
            self.e('global','pack','R','T');self.e('local','pack','R','B')
            self.first,self.last='maps','pack'
        elif self.kind=='generator':
            sx,sy=w/460,h/440
            def g(id,x,y,bw,bh,title,sub=''):
                self.n(id,x*sx,y*sy,bw*sx,bh*sy,title,sub)
            external_context=self.page.id=='d'
            g('content',20,64,188,56,'Text features' if external_context else 'ContentContext',
              'From expanded ContentContext' if external_context else 'RPE MHA → gated DW-FFN')
            g('seed',264,64,176,56,'Content-only seed','Linear + reshape 512×4×4T')
            g('g1',20,188,112,80,'GBlock 1','CBN(g₁)')
            g('spatial',172,188,124,80,'Spatial style attention','Q = image features\nK/V = local L')
            g('g234',336,188,104,80,'GBlocks 2–4','CBN(g₂…g₄)')
            g('split',20,288,100,32,'g | L')
            g('bssp',20,352,188,56,'Global style projections','g → 4 MLPs → CBN')
            g('head',336,352,104,56,'Image head','Conv + tanh')
            self.e('content','seed','R','L');self.e('seed','g1','B','T')
            self.e('g1','spatial','R','L');self.e('spatial','g234','R','L')
            self.e('g234','head','B','T')
            self.e('split','bssp','B','T');self.e('split','spatial','R','B')
            self.e('bssp','g1','T','B',role='style')
            self.e('bssp','g234','R','B',role='style')
            self.first,self.last='split','head'
        elif self.kind=='global':
            self.n('b',20,64,100,72,'Frozen B','Feature sequence')
            self.n('eg',156,64,w-176,72,'Global E only','Context → pool → μ_g')
            self.e('b','eg','R','L');self.first,self.last='b','eg'
        else:
            data={
                'patch':[('stem','3×3 stem','1 → 64'),('blocks','StrokePatchBlock ×3','128 → 192 → 192\n3×3 + axial DW + residual'),('out','Spatial logit head','1×1 conv → score map')],
                'crop':[('size','Native patch size','s = H/2'),('strata','Horizontal strata + vertical alternation','Upper / lower image halves'),('out','4–8 native patches per word','K = clamp(ceil(W/s), 4, 8)')],
                'writer':[('b','Frozen B maps',''),('pool','Attentive mean + standard deviation','Concat → 256-D projection'),('out','Writer MLP','339 logits')],
                'critic':[('cnn','Residual whole-word CNN','Symmetric DiffAug at input'),('pool','Valid-area mean × 32','Ignore padding, not background'),('out','Scalar score','Real vs generated full words')],
                'rng':[('rng','Seeded PyTorch RNG: N(0,I)','Independent of any reference image'),('split','Draw batch × 8 × 32 values','Keep 1 global; separate 7 locals'),('out','Raw sampled slots' if self.page.id=='d' else 'Center ONLY the local slots','Centering follows in the next block' if self.page.id=='d' else 'z = [z_g, Z_L − mean_slots(Z_L)]')],
                'center':[('split','Input sampled global/local slots','z_g: 1×32 · Z_L: 7×32'),('mean','Subtract local-slot mean','Global token unchanged'),('out','Packed prior z','Shape: 8×32')],
                'content':[('embed','Character embedding','120-D'),('attention','Relative-position self-attention','4 heads · attention width 128'),('out','Gated local feed-forward','DW k5 · hidden width 360')],
            }[self.kind]
            start=52
            gap=16 if h>=260 else 10
            nh=(h-start-16-gap*2)/3
            for i,(id,title,sub) in enumerate(data):
                self.n(id,24,start+i*(nh+gap),w-48,nh,title,sub)
            for a,b in zip(data,data[1:]):self.e(a[0],b[0],'B','T')
            self.first,self.last=data[0][0],data[-1][0]

    def bind(self,index,edge,mode):
        p=self.parent
        side=edge.out if mode=='source' else edge.into
        fraction=edge.a if mode=='source' else edge.b
        px,py=p.port(side,fraction);px-=p.x;py-=p.y
        x=(0 if side=='L' else p.w-4) if side in ('L','R') else px-2
        y=(0 if side=='T' else p.h-4) if side in ('T','B') else py-2
        id=f'port-{index}-{mode}'
        self.n(id,x,y,4,4,'',kind='style')
        self.binding[(index,mode)]=(id,side)
        if mode=='target':
            anchor='content' if self.kind=='generator' and edge.source in ('text','content') else self.first
            self.e(id,anchor,{'L':'R','R':'L','T':'B','B':'T'}[side])
        else:
            anchor=self.last
            if self.kind=='encoder' and edge.target in ('content','kl','cycle'):anchor='global'
            if self.kind=='encoder' and edge.target=='ctx':anchor='maps'
            if self.kind=='rng' and edge.target=='info':anchor='split'
            self.e(anchor,id,into={'L':'R','R':'L','T':'B','B':'T'}[side])

    def finish(self):
        # Distinct attachment points on every shared mini-node face.
        for attr,sideattr,frac in (('source','out','a'),('target','into','b')):
            grouped={}
            for e in self.graph.edges:
                if getattr(e,attr).startswith('port-'):continue
                grouped.setdefault((getattr(e,attr),getattr(e,sideattr)),[]).append(e)
            for group in grouped.values():
                for i,e in enumerate(group):setattr(e,frac,(i+1)/(len(group)+1))
        try:
            core.route(self.graph,width=self.parent.w,top=40,bottom=self.parent.h-4,
                       margin=4,clearance=4,obstacle_pad=2,cross_penalty=10000)
        except ValueError as error:
            raise ValueError(f'{self.page.id}/{self.parent.id}: {error}') from error


def cell(root,id,parent,n,title=None,sub=None,fill='#ffffff',port=False):
    value='' if port else '<b>'+escape(n.title if title is None else title)+'</b>'
    caption=n.sub if sub is None else sub
    if caption and not port:value+='<br><span style="font-size:13px">'+escape(caption).replace('\n','<br>')+'</span>'
    style=(f'rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={INK};'
           f'strokeWidth=1.2;fontColor={INK};fontFamily=Arial;fontSize=14;align=center;verticalAlign=middle;spacing=4;overflow=hidden;')
    if port:style='ellipse;html=1;fillColor=#202020;strokeColor=none;resizable=0;movable=0;'
    c=ET.SubElement(root,'mxCell',{'id':id,'value':value,'vertex':'1','parent':parent,'style':style})
    ET.SubElement(c,'mxGeometry',{'x':str(n.x),'y':str(n.y),'width':str(n.w),'height':str(n.h),'as':'geometry'})
    return c


def edge_cell(root,id,parent,e,nodes,prefix='',source=None,target=None,port_s=None,port_t=None):
    a,b=nodes[e.source],nodes[e.target]
    x,y=a.port(e.out,e.a);u,v=b.port(e.into,e.b)
    exitx,exity=(x-a.x)/a.w,(y-a.y)/a.h
    entryx,entryy=(u-b.x)/b.w,(v-b.y)/b.h
    if port_s:
        exitx,exity={'L':(0,.5),'R':(1,.5),'T':(.5,0),'B':(.5,1)}[port_s]
    if port_t:
        entryx,entryy={'L':(0,.5),'R':(1,.5),'T':(.5,0),'B':(.5,1)}[port_t]
    color='#98612f' if e.role=='objective' else '#47776b' if e.role=='style' else INK
    dash='dashed=1;dashPattern=6 4;' if e.role=='objective' else ''
    style=(f'edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;endFill=1;'
           f'jumpStyle=arc;jumpSize=8;strokeColor={color};strokeWidth=1.4;{dash}'
           f'exitX={exitx};exitY={exity};entryX={entryx};entryY={entryy};exitPerimeter=0;entryPerimeter=0;')
    c=ET.SubElement(root,'mxCell',{'id':id,'edge':'1','parent':parent,'source':source or prefix+e.source,'target':target or prefix+e.target,'style':style})
    g=ET.SubElement(c,'mxGeometry',{'relative':'1','as':'geometry'})
    if e.points[1:-1]:
        arr=ET.SubElement(g,'Array',{'as':'points'})
        for x,y in e.points[1:-1]:ET.SubElement(arr,'mxPoint',{'x':str(x),'y':str(y)})


def native_file(ps,head):
    xml=ET.Element('mxfile',{'host':'app.diagrams.net','compressed':'false','pages':str(len(ps)+1)})
    for index,p in enumerate(ps):
        d=ET.SubElement(xml,'diagram',{'id':p.id,'name':p.name})
        m=ET.SubElement(d,'mxGraphModel',{'grid':'1','gridSize':'8','page':'1','pageScale':'1','pageWidth':str(W),'pageHeight':str(H),'background':'#ffffff'})
        root=ET.SubElement(m,'root');ET.SubElement(root,'mxCell',{'id':'0'})
        for id in ('zones','edges','1','notes'):ET.SubElement(root,'mxCell',{'id':id,'parent':'0'})
        bg=core.Node('paper',48,128,W-96,1120,'')
        bc=cell(root,'paper','zones',bg,fill=compact.PANEL_COLORS[index])
        bc.set('value','');bc.set('style',f'fillColor={compact.PANEL_COLORS[index]};strokeColor=none;')
        for id,n in p.nodes.items():
            if id not in p.minis:
                c=cell(root,id,'1',n,fill=compact.FILLS[n.kind])
                c.set('style',c.get('style').replace('fontSize=14','fontSize=18').replace('spacing=4','spacing=10'))
                if n.frozen:c.set('style',c.get('style')+'dashed=1;dashPattern=5 3;')
            else:
                mini=p.minis[id]
                c=cell(root,id,'1',n,title='',sub='',fill=compact.FILLS[n.kind])
                c.set('value','')
                c.set('style',c.get('style')+'container=1;collapsible=0;recursiveResize=0;')
                if n.frozen:c.set('style',c.get('style')+'dashed=1;dashPattern=5 3;')
                header=core.Node('',16,8,n.w-32,36,mini.title)
                hc=cell(root,id+'-header',id,header,sub='',fill=compact.FILLS[n.kind])
                hc.set('style','text;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;fontFamily=Arial;fontSize=16;fontStyle=1;strokeColor=none;fillColor=none;spacing=0;')
                for childid,child in mini.graph.nodes.items():
                    port=childid.startswith('port-')
                    cell(root,id+'-'+childid,id,child,fill='#e9e1ef' if child.kind=='style' else '#fffaf6',port=port)
                for j,e in enumerate(mini.graph.edges):edge_cell(root,f'{id}-inner-edge-{j}',id,e,mini.graph.nodes,prefix=id+'-')
        for i,e in enumerate(p.edges):
            source,target=e.source,e.target
            srcside=dstside=None
            if source in p.minis:
                pin,srcside=p.minis[source].binding[(i,'source')];source=source+'-'+pin
            if target in p.minis:
                pin,dstside=p.minis[target].binding[(i,'target')];target=target+'-'+pin
            edge_cell(root,f'outer-edge-{i}','edges',e,p.nodes,source=source,target=target,port_s=srcside,port_t=dstside)
        compact.text(root,'title',64,28,W-128,48,p.title,30,True)
        compact.text(root,'subtitle',64,88,W-128,32,p.subtitle,17)
        compact.text(root,'legend',64,1272,W-128,28,'C1: global/local E   ·   C2: content-first fusion   ·   C3: StrokePatchD/crops   |   Solid: forward   ·   Dashed: loss/target   ·   Green: global-style conditioning',16)
        y=1320
        for j,(kind,value) in enumerate(p.explain):
            size=21 if kind=='h' else 22 if kind=='eq' else 18
            lines=core.wrapped_lines(value,W-128,size)
            h=len(lines)*(30 if kind=='eq' else 27)+10
            compact.text(root,f'explain-{j}',64,y,W-128,h,value,size,kind=='h',kind=='eq')
            y+=h+10
        assert y<1904,(p.id,'explanations overflow',y)
        compact.text(root,'source',64,1920,W-128,24,f'ALPHA · {head[:10]} · 2026-10-15 · Reference and random priors are distinct style sources; nested mini-diagrams share the same model parameters.',13)
    comparison_panel.append_panel(xml,core,cell,edge_cell,compact.text,head)
    ET.indent(xml,space='  ')
    return ET.tostring(xml,encoding='utf-8',xml_declaration=True)


def validate(ps,data):
    problems=[];reports=[]
    for p in ps:
        all_graphs=[('outer',p,p.nodes)]+[(id,m.graph,m.graph.nodes) for id,m in p.minis.items()]
        for name,graph,nodes in all_graphs:
            ns=[n for id,n in nodes.items() if not id.startswith('port-')]
            for i,n in enumerate(ns):
                for k in ns[i+1:]:
                    if max(n.x,k.x)<min(n.x+n.w,k.x+k.w) and max(n.y,k.y)<min(n.y+n.h,k.y+k.h):
                        problems.append(f'{p.id}/{name}: overlap {n.id}/{k.id}')
            for i,e in enumerate(graph.edges):
                for a,b in zip(e.points,e.points[1:]):
                    if a[0]!=b[0] and a[1]!=b[1]:problems.append(f'{p.id}/{name}: diagonal edge {i}')
                    for n in ns:
                        if n.id not in (e.source,e.target) and core.interior_crossing(a,b,(n.x,n.y,n.w,n.h)):
                            problems.append(f'{p.id}/{name}: transit edge {i} through {n.id}')
        reports.append({'page':p.name,'external_edges':len(p.edges),'external_crossings':crossing_count(p.edges),
                        'mini_diagrams':len(p.minis),'internal_edges':sum(len(m.graph.edges) for m in p.minis.values())})
    tree=ET.fromstring(data)
    for dia in tree.findall('diagram'):
        cells=dia.findall('.//mxCell');ids={c.get('id') for c in cells}
        assert len(ids)==len(cells)
        # Count graph components over actual leaf tensors/operations/ports,
        # excluding layout containers and explanatory text cells.
        edges=[c for c in cells if c.get('edge')=='1']
        endpoints={c.get(a) for c in edges for a in ('source','target')}
        adjacent={id:set() for id in endpoints}
        for e in edges:
            a,b=e.get('source'),e.get('target');assert a in ids and b in ids
            adjacent[a].add(b);adjacent[b].add(a)
        unseen=set(adjacent);components=0
        while unseen:
            seen=set();pending=[next(iter(unseen))]
            while pending:
                n=pending.pop()
                if n not in seen:seen.add(n);pending.extend(adjacent[n]-seen)
            unseen-=seen;components+=1
        expected=12 if dia.get('id')=='f' else 1
        if components!=expected:problems.append(dia.get('name')+f': expected {expected} independent forward graph(s), got {components}')
        if dia.get('id')=='f':
            reports.append({'page':dia.get('name'),'paired_modules':6,
                            'independent_forward_graphs':components,'edges':len(edges)})
    return {'ok':not problems,'problems':problems,'pages':reports,'sha256':sha256(data).hexdigest(),
            'scope':'Exact XML; nested and external connectivity; orthogonality; crossings and transit geometry. Native renderer checked separately.'}


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    ps=make_pages()
    head=subprocess.check_output(['git','-C',str(core.REPO),'rev-parse','HEAD'],text=True).strip()
    target=ROOT/'alpha_training_architecture.drawio'
    previous=ET.fromstring(target.read_bytes()) if target.exists() else None
    snapshot=lambda d:ET.canonicalize(ET.tostring(d,encoding='unicode'),strip_text=True)
    old_training_pages={d.get('id'):snapshot(d) for d in previous.findall('diagram') if d.get('id') in 'abcde'} if previous is not None else {}
    data=native_file(ps,head);check=validate(ps,data)
    current=ET.fromstring(data)
    check['existing_training_pages_unchanged']=all(
        old_training_pages.get(d.get('id'),snapshot(d))==snapshot(d)
        for d in current.findall('diagram') if d.get('id') in 'abcde'
    )
    if not check['existing_training_pages_unchanged']:
        check['ok']=False;check['problems'].append('Existing training pages changed while adding comparison tab')
    (ROOT/'verification.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    if not check['ok']:
        print('\n'.join(check['problems']));raise SystemExit(1)
    (ROOT/'alpha_training_architecture.drawio').write_bytes(data)
    print(json.dumps(check['pages'],indent=2))
    print('OK: five training pages preserved; sixth tab compares six module pairs')


if __name__=='__main__':main()
