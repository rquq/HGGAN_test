"""Native side-by-side module comparisons; no model code is imported or changed.

Each row is a pair of independent forward graphs. No arrow between versions
is allowed: a comparison relationship is not a computational dependency.
"""
from html import escape
import xml.etree.ElementTree as ET

WIDTH, HEIGHT = 2880, 3000
BASELINE_HEAD = '5d37de733f79e2a72a79c0f66e1865329ff56b4e'
BASELINE = '/home/quq/machineLearning/HTG/HiGANplus/HiGAN+'
CURRENT = '/home/quq/machineLearning/HTG/HGGAN_test/alpha'


def append_panel(xml, core, draw_cell, draw_edge, text, current_head):
    diagram = ET.SubElement(xml, 'diagram', {'id': 'f', 'name': '(f) Modules vs HiGAN+'})
    model = ET.SubElement(diagram, 'mxGraphModel', {
        'grid': '1', 'gridSize': '8', 'page': '1', 'pageScale': '1',
        'pageWidth': str(WIDTH), 'pageHeight': str(HEIGHT), 'background': '#ffffff',
    })
    root = ET.SubElement(model, 'root')
    ET.SubElement(root, 'mxCell', {'id': '0'})
    for layer in ('zones', 'edges', '1', 'notes'):
        ET.SubElement(root, 'mxCell', {'id': layer, 'parent': '0'})
    graphs = []
    pairs = []

    def frame(key, x, y, height, title, side):
        node = core.Node(key, x, y, 1280, height, '')
        c = draw_cell(root, key, '1', node, fill='#eae5ef' if side == 'before' else '#e7edda')
        c.set('value', '')
        c.set('style', c.get('style') + 'container=1;collapsible=0;recursiveResize=0;')
        header = core.Node(key + '-title', 24, 12, 1232, 32, title)
        h = draw_cell(root, key + '-title', key, header)
        h.set('style', 'text;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;'
              'fontFamily=Arial;fontSize=20;fontStyle=1;strokeColor=none;fillColor=none;spacing=0;')
        graph = core.Page(key, key, '', '')
        graphs.append((key, node, graph))
        return graph

    def n(graph, key, x, y, width, height, title, caption='', kind='module'):
        graph.n(key, x, y, width, title, caption, kind=kind, h=height)

    def e(graph, source, target, out='R', into='L', a=.5, b=.5, role='forward'):
        graph.e(source, target, out, into, a, b, role)

    def chain(graph, ids):
        for source, target in zip(ids, ids[1:]):
            e(graph, source, target)

    def pair(number, y, height, title, delta, baseline_source, current_source):
        left = frame(f'row{number}-before', 64, y, height, title, 'before')
        right = frame(f'row{number}-after', 1536, y, height, title, 'after')
        text(root, f'delta{number}', 1360, y + height / 2 - 28, 160, 56,
             f'Δ {number}', 26, True)
        text(root, f'change{number}', 64, y + height + 8, WIDTH - 128, 32, delta, 18)
        text(root, f'refs{number}', 64, y + height + 44, WIDTH - 128, 24,
             f'HiGAN+: {baseline_source}    |    ALPHA: {current_source}', 14)
        pairs.append({'module': title, 'delta': delta, 'baseline_source': baseline_source,
                      'current_source': current_source})
        return left, right

    text(root, 'title', 64, 24, WIDTH - 128, 48,
         '(f) Module-by-module comparison with HiGAN+', 30, True)
    text(root, 'subtitle', 64, 80, WIDTH - 128, 32,
         'Matched scope per row · native editable blocks · C1/C2/C3 = primary redesigns; supporting changes are labeled separately', 18)
    text(root, 'before-heading', 64, 128, 1280, 32, 'HiGAN+ · original implementation', 24, True)
    text(root, 'after-heading', 1536, 128, 1280, 32, 'ALPHA · current implementation', 24, True)

    # 1. E receives a final sequence AND intermediate maps. Do not imply that
    # the global branch directly fuses all three local maps.
    left, right = pair(1, 176, 408, 'C1 · Style encoder E',
        'CHANGED: one pooled 32-D style → contextual global token + seven centered local evidence tokens. Multiscale B maps already exist in HiGAN+.',
        'module.py:65–116', 'module.py:301–585; configs/gan_iam_32.yml:226–236')
    for key, x, title, caption in [
        ('input', 32, 'Final B sequence', 'f ∈ R^(256×T_B)'),
        ('pool', 280, 'Valid-width mean', 'One pooled 256-D vector'),
        ('mlp', 528, 'Two-layer MLP', '256 → 256 → 256'),
        ('posterior', 776, 'Global posterior', 'μ / logσ² · 32 dimensions'),
        ('output', 1024, 'Single style vector', 'Sample at train; μ at eval'),
    ]:
        n(left, key, x, 168, 224, 88, title, caption)
    chain(left, ['input', 'pool', 'mlp', 'posterior', 'output'])
    n(right, 'input', 32, 168, 192, 88, 'B outputs', 'Final sequence f\n3 intermediate maps F₁,F₂,F₃')
    n(right, 'context', 304, 68, 336, 96, 'Global sequence context', '2 gated DW k7 blocks (d=1,2)\n→ valid mean → LN → MLP')
    n(right, 'posterior', 720, 68, 224, 96, 'Global token g', 'μ / logσ², 32-D\nSample train; μ eval')
    n(right, 'local', 304, 276, 336, 96, 'Multiscale evidence retrieval', '3 maps → 4 height rows each\n7 learned queries · MHA, 4 heads')
    n(right, 'project', 720, 276, 224, 96, 'Local tokens L', '256 → 32, bias-free\nSubtract local-slot mean')
    n(right, 'output', 1040, 168, 208, 88, 'Style pack S=[g,L]', '1 global + 7 locals\n8 × 32; no local KL', kind='style')
    e(right, 'input', 'context', b=.5, a=.3)
    e(right, 'input', 'local', b=.5, a=.7)
    e(right, 'context', 'posterior'); e(right, 'local', 'project')
    e(right, 'posterior', 'output', into='T'); e(right, 'project', 'output', into='B')

    # 2. Both variants already use four residual GBlocks with CBN. The change
    # is the conditioning topology, not inventing a residual image decoder.
    left, right = pair(2, 696, 520, 'C2 · Content / style fusion inside G',
        'REWIRED: early content–style concatenation → content-first seed + separate global CBN and local spatial retrieval. Four residual GBlocks are retained.',
        'BigGAN_networks.py:113–196', 'BigGAN_networks.py:153–278; fusion.py:14–215')
    n(left, 'text', 32, 80, 232, 80, 'Target y → embedding', '120-D per character')
    n(left, 'concat', 336, 80, 256, 80, 'Early concatenation', '[embedding(y_t), repeat(s)]')
    n(left, 'seed', 664, 80, 232, 80, 'Mixed spatial seed', 'Linear → 512 × 4 × 4T')
    n(left, 'blocks', 968, 80, 280, 96, 'GBlocks 1–4', 'Residual image stages\nEach block uses CBN(s_i)')
    n(left, 'style', 32, 344, 232, 80, 'Style s', '32-D reference code or prior', kind='style')
    n(left, 'project', 336, 344, 560, 80, 'One shared linear style projection', '32 → 128 → split into four 32-D CBN conditions')
    n(left, 'output', 968, 344, 280, 80, 'Image head', 'BN → ReLU → Conv → tanh')
    chain(left, ['text', 'concat', 'seed', 'blocks'])
    e(left, 'style', 'concat', out='T', into='B')
    e(left, 'style', 'project', role='style')
    e(left, 'project', 'blocks', into='B', role='style')
    e(left, 'blocks', 'output', out='B', into='T')
    n(right, 'text', 32, 80, 232, 80, 'Target y → embedding', '120-D per character')
    n(right, 'context', 336, 80, 256, 80, 'ContentContext', 'RPE self-attn → gated DW k5 FFN\nLearned residual scales')
    n(right, 'seed', 664, 80, 232, 80, 'Content-only seed', 'Linear → 512 × 4 × 4T')
    n(right, 'g1', 32, 240, 232, 88, 'GBlock 1', 'CBN(g₁) → early image features')
    n(right, 'spatial', 336, 240, 256, 88, 'Spatial style refinement', 'Q=features; K/V=L\n4 heads · width 128\nLearned residual')
    n(right, 'g234', 664, 240, 232, 88, 'GBlocks 2–4', 'CBN(g₂…g₄)')
    n(right, 'output', 968, 240, 280, 88, 'Image head', 'BN → ReLU → Conv → tanh')
    n(right, 'style', 32, 408, 232, 80, 'Style pack [g,L]', 'Global and local paths separated', kind='style')
    n(right, 'project', 336, 408, 560, 80, 'Four independent global-style MLPs', 'Each: 32 → 32 → SiLU → 32; separate condition per GBlock')
    chain(right, ['text', 'context', 'seed'])
    e(right, 'seed', 'g1', out='B', into='T')
    chain(right, ['g1', 'spatial', 'g234', 'output'])
    e(right, 'style', 'project', role='style')
    e(right, 'style', 'spatial', out='T', into='B')
    e(right, 'project', 'g1', out='T', into='B', a=.2, role='style')
    e(right, 'project', 'g234', out='T', into='B', a=.8, role='style')

    left, right = pair(3, 1328, 312, 'C3 · Crop policy + local discriminator P',
        'REPLACED: exhaustive fixed patches + scalar crop critic → stratified H/2 crops + directional spatial critic; hinge precedes map/crop reduction.',
        'utils.py:308–337; BigGAN_networks.py:330–358; model.py:744–761',
        'utils.py:445–521; BigGAN_networks.py:508–608; model.py:2162–2217')
    for graph, entries in [(left, [
        ('image', 'Valid image', 'Real / generated word'),
        ('crop', 'Exhaustive sliding windows', '32 × 32; stride 8'),
        ('critic', 'Residual patch D', 'Same Discriminator family\nInput resolution 32'),
        ('readout', 'One scalar per patch', 'Global sum → SN linear'),
        ('loss', 'Patch-level hinge / mean', 'Average all collected patches'),
    ]), (right, [
        ('image', 'Valid image', 'Real / all three fake branches'),
        ('crop', 'Stratified crop policy', 's=H/2; K=clamp(ceil(W/s),4,8)\nHorizontal strata; upper/lower alternation'),
        ('critic', 'StrokePatchD', 'Stem → 3 residual stroke blocks\n3×3 + DW 1×5 / 5×1 + 1×1'),
        ('readout', 'Spatial score map', '1×1 logit head; no global sum'),
        ('loss', 'Map-wise hinge, then mean', 'Map → crops → equal-weight words\nFake branches averaged equally'),
    ])]:
        for index, (key, title, caption) in enumerate(entries):
            n(graph, key, 32+index*248, 104, 224, 112, title, caption)
        chain(graph, ['image', 'crop', 'critic', 'readout', 'loss'])

    left, right = pair(4, 1760, 224, 'Supporting change · Writer head W',
        'CHANGED: uniform temporal mean → learned attentive mean + standard deviation, projected back to 256-D. Writer supervision itself already exists in HiGAN+.',
        'module.py:119–146', 'module.py:588–674')
    for graph in (left, right):
        n(graph, 'input', 32, 80, 232, 88, 'Final B sequence', 'B is frozen during GAN training')
        n(graph, 'head', 968, 80, 280, 88, 'Writer MLP → logits', 'Writer identity CE target w')
    n(left, 'pool', 336, 80, 256, 88, 'Valid-width mean', 'Uniform average → 256-D')
    chain(left, ['input', 'pool', 'head'])
    n(right, 'pool', 336, 80, 256, 88, 'Attentive statistics', 'Weighted mean + std → 512-D')
    n(right, 'projection', 664, 80, 232, 88, 'Feature projection', '512 → 256 → LN → LeakyReLU')
    chain(right, ['input', 'pool', 'projection', 'head'])

    left, right = pair(5, 2104, 224, 'Supporting change · Whole-word discriminator D',
        'CHANGED: spatial evidence sum divided by text length → valid-area mean × shared gain 32. Symmetric mild DiffAug is used on the current whole-word D path.',
        'BigGAN_networks.py:330–351', 'BigGAN_networks.py:467–503; model.py:2149–2192')
    for graph in (left, right):
        n(graph, 'input', 32, 80, 232, 88, 'Complete word image', 'Not a set of local crops')
        n(graph, 'cnn', 336, 80, 256, 88, 'Residual CNN D', 'Same family of spatial critic blocks')
        n(graph, 'head', 968, 80, 280, 88, 'SN linear → scalar', 'Whole-word real/fake supervision')
    n(left, 'pool', 664, 80, 232, 88, 'Valid spatial sum', 'Divide by target label length')
    n(right, 'pool', 664, 80, 232, 88, 'Valid-area mean × 32', 'Pool height and valid width\nWidthContext option is OFF')
    chain(left, ['input', 'cnn', 'pool', 'head']); chain(right, ['input', 'cnn', 'pool', 'head'])

    left, right = pair(6, 2448, 248, 'Supporting change · Backbone B geometry / padding',
        'RETAINED: residual CNN topology and intermediate maps. CHANGED: geometry-derived front stride, exact width propagation and valid-padding handling for native 32/64 px.',
        'module.py:9–62', 'module.py:11–103,167–241')
    for graph in (left, right):
        n(graph, 'image', 32, 88, 232, 88, 'Normalized image', 'Same residual-CNN family')
        n(graph, 'cnn', 664, 88, 232, 88, 'Residual CNN stages', 'Intermediate maps: 64 / 128 / 256')
        n(graph, 'output', 968, 88, 280, 104, 'Sequence + spatial maps', 'Final sequence feeds global E / W\nMaps are available in BOTH versions')
    n(left, 'stem', 336, 88, 256, 88, 'Fixed stride-2 front end', 'Original native 64px path\nHeight collapse uses squeeze')
    n(right, 'stem', 336, 88, 256, 88, 'Geometry-aware front end', 'stride=max(1,H//32)\nExact valid lengths through CNN')
    chain(left, ['image', 'stem', 'cnn', 'output']); chain(right, ['image', 'stem', 'cnn', 'output'])

    problems = []
    for key, container, graph in graphs:
        # Separate attachment points preserve branches without stacking arrows.
        for attr, sideattr, fraction in (('source', 'out', 'a'), ('target', 'into', 'b')):
            groups = {}
            for edge in graph.edges:
                groups.setdefault((getattr(edge, attr), getattr(edge, sideattr)), []).append(edge)
            for edges in groups.values():
                if len(edges) > 1:
                    for index, edge in enumerate(edges):
                        setattr(edge, fraction, (index+1)/(len(edges)+1))
        core.route(graph, width=1280, top=52, bottom=container.h-8,
                   margin=8, clearance=12, obstacle_pad=4, cross_penalty=3000)
        nodes = list(graph.nodes.values())
        for index, node in enumerate(nodes):
            if not (0 <= node.x and node.x+node.w <= 1280 and 52 <= node.y and node.y+node.h <= container.h):
                problems.append(f'{key}: node out of bounds: {node.id}')
            for other in nodes[index+1:]:
                if max(node.x, other.x) < min(node.x+node.w, other.x+other.w) and max(node.y, other.y) < min(node.y+node.h, other.y+other.h):
                    problems.append(f'{key}: overlapping nodes {node.id}/{other.id}')
        for index, edge in enumerate(graph.edges):
            for a, b in zip(edge.points, edge.points[1:]):
                if a[0] != b[0] and a[1] != b[1]:
                    problems.append(f'{key}: diagonal edge {index}')
                for node in nodes:
                    if node.id not in (edge.source, edge.target) and core.interior_crossing(a,b,(node.x,node.y,node.w,node.h)):
                        problems.append(f'{key}: unrelated-box transit {index}/{node.id}')
        for node in nodes:
            c = draw_cell(root, key+'-'+node.id, key, node,
                          fill='#ded6e9' if node.kind=='style' else '#fffaf6')
            c.set('value', '<b>'+escape(node.title)+'</b>' +
                  ('<br><span style="font-size:15px">'+escape(node.sub).replace('\n','<br>')+'</span>' if node.sub else ''))
            c.set('style', c.get('style').replace('fontSize=14', 'fontSize=17'))
        for index, edge in enumerate(graph.edges):
            draw_edge(root, key+f'-edge-{index}', key, edge, graph.nodes, prefix=key+'-')

    text(root, 'legend', 64, 2812, WIDTH-128, 32,
         'Columns are independent models, not a pipeline between versions. Solid arrows = forward tensors; green arrows = global-style conditioning.', 17)
    text(root, 'attribution', 64, 2860, WIDTH-128, 48,
         'Retained: backbone / OCR family, four GBlocks with CBN, global+patch adversarial training, OCR and writer supervision. These diagrams establish structural differences, not measured FID/KID gains.', 17)
    text(root, 'sources', 64, 2928, WIDTH-128, 48,
         f'2026-10-16 · HiGAN+ {BASELINE_HEAD[:10]} vs ALPHA {current_head[:10]} · Sources: original HiGANplus/HiGAN+/networks and ALPHA networks; normal gan_iam_32/64 configs.', 14)
    if problems:
        raise ValueError('\n'.join(problems))
    return {'page': '(f) Modules vs HiGAN+', 'module_pairs': pairs,
            'independent_forward_graphs': len(graphs),
            'native_internal_edges': sum(len(g.edges) for _,_,g in graphs),
            'baseline_head': BASELINE_HEAD, 'current_head': current_head,
            'baseline_path': BASELINE, 'current_path': CURRENT,
            'scope': 'Six independent paired module comparisons; no causal or metric-improvement claim.'}
