// Exercise the official draw.io graph codec and renderer offline. This is not
// a claim that the network-dependent, complete web editor UI was tested.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { chromium } = require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async () => {
  const folder = process.argv[2];
  const canvasHeight = Number(process.argv[4] || 1040);
  const makePreview = process.argv[5] === 'preview';
  const bundle = fs.readFileSync(process.argv[3], 'utf8');
  const xml = fs.readFileSync(path.join(folder, 'alpha_training_architecture.drawio'), 'utf8');
  const canvasWidth = Number(xml.match(/pageWidth="(\d+)"/)?.[1] || 1600);
  const browser = await chromium.launch({ executablePath: 'C:/Program Files (x86)/Microsoft/EdgeCore/155.0.4283.45/msedge.exe', headless: true, args: ['--disable-gpu', '--no-sandbox'] });
  try {
    const page = await browser.newPage({ viewport: { width: canvasWidth, height: canvasHeight } });
    await page.route('**/*', route => route.abort());
    await page.setContent(`<!doctype html><html><body style="margin:0"><div id="graph" style="width:${canvasWidth}px;height:${canvasHeight}px;overflow:hidden;background:white"></div></body></html>`);
    await page.addScriptTag({ content: 'window.urlParams={};window.mxLoadResources=false;window.mxLoadStylesheets=false;' });
    await page.addScriptTag({ content: bundle });
    const report = await page.evaluate(({ xml }) => {
      const documentXml = mxUtils.parseXml(xml);
      const diagrams = [...documentXml.getElementsByTagName('diagram')];
      window.__nativePages = diagrams;
      window.__nativeXml = documentXml;
      window.__nativeGraph = new Graph(document.getElementById('graph'));
      window.__nativeGraph.setEnabled(false);
      return { diagrams: diagrams.map(x => x.getAttribute('name')),
        sizes: diagrams.map(x => {
          const m=x.getElementsByTagName('mxGraphModel')[0];
          return {width:Number(m.getAttribute('pageWidth'))||1600,height:Number(m.getAttribute('pageHeight'))||1040};
        }), Graph: typeof Graph, mxCodec: typeof mxCodec };
    }, { xml });
    const results = [];
    const svgPages = [];
    for (let i=0; i<report.diagrams.length; i++) {
      await page.setViewportSize(report.sizes[i]);
      await page.evaluate(size => {
        const container=document.getElementById('graph');
        container.style.width=size.width+'px';container.style.height=size.height+'px';
      },report.sizes[i]);
      const result = await page.evaluate(index => {
        const graph = window.__nativeGraph;
        // IDs are page-local in mxfile. Match the editor's separate page parse;
        // decoding against the entire mxfile would falsely flag repeated IDs.
        const modelDoc = mxUtils.parseXml(mxUtils.getXml(window.__nativePages[index].getElementsByTagName('mxGraphModel')[0]));
        const codec = new mxCodec(modelDoc);
        codec.decode(modelDoc.documentElement, graph.getModel());
        graph.view.scaleAndTranslate(1, 0, 0);
        graph.refresh();
        const model = graph.getModel();
        const cells = Object.values(model.cells);
        const edges = cells.filter(c => model.isEdge(c));
        const vertices = cells.filter(c => model.isVertex(c));
        const missing = edges.filter(e => !model.getTerminal(e, true) || !model.getTerminal(e, false));
        const unrendered = edges.filter(e => !graph.view.getState(e)?.absolutePoints?.length);
        const expectedVertices = [...window.__nativePages[index].getElementsByTagName('mxCell')].filter(c => c.getAttribute('vertex') === '1').length;
        const expectedEdges = [...window.__nativePages[index].getElementsByTagName('mxCell')].filter(c => c.getAttribute('edge') === '1').length;
        const expectedLinks = window.__nativePages[index].getElementsByTagName('UserObject').length;
        const links = vertices.filter(c => !!graph.getLinkForCell(c)).length;
        const visibleMarkup = [...document.querySelectorAll('#graph svg text')]
          .map(n => n.textContent).filter(t => /<\/?(?:b|span|br|sub|sup)(?:\s|>)/.test(t));
        const ancestor = (candidate, child) => {
          for (let p=child; p; p=p.parent) if (p===candidate) return true;
          return false;
        };
        const layer = model.getCell('1');
        const modules = vertices.filter(c => c!==layer && ancestor(layer,c) &&
          !String(c.style || '').startsWith('text;') && !c.id.includes('-port-'));
        const transit = [];
        for (const edge of edges) {
          const points = graph.view.getState(edge)?.absolutePoints || [];
          for (let j=1;j<points.length;j++) {
            const a=points[j-1], b=points[j];
            if (!a || !b) continue;
            for (const node of modules) {
              if (ancestor(node,edge.source) || ancestor(node,edge.target)) continue;
              const r = graph.view.getState(node);
              if (!r) continue;
              const hitV = Math.abs(a.x-b.x)<.01 && r.x+.5<a.x && a.x<r.x+r.width-.5 && Math.max(Math.min(a.y,b.y),r.y+.5)<Math.min(Math.max(a.y,b.y),r.y+r.height-.5);
              const hitH = Math.abs(a.y-b.y)<.01 && r.y+.5<a.y && a.y<r.y+r.height-.5 && Math.max(Math.min(a.x,b.x),r.x+.5)<Math.min(Math.max(a.x,b.x),r.x+r.width-.5);
              if (hitV || hitH) transit.push([edge.id,node.id]);
            }
          }
        }
        return { name: window.__nativePages[index].getAttribute('name'), vertices: vertices.length, expectedVertices, expectedEdges,
          links, expectedLinks, edges: edges.length, dangling_edges: missing.length, unrendered_edges: unrendered.length, native_edge_box_transit: transit,
          visible_markup_labels: visibleMarkup };
      }, i);
      await page.screenshot({ path: path.join(folder, `native_${String(i).padStart(2,'0')}.png`) });
      results.push(result);
      if (makePreview) {
        const svg = await page.evaluate(index => {
          const graph = window.__nativeGraph;
          const svg = graph.getSvg('#ffffff', 1, 0);
          // Drop only the exporter's fallback-help link. Actual SVG/HTML text
          // remains intact and no remote dependency is needed in the preview.
          for (const link of svg.querySelectorAll('a')) {
            const href=link.getAttribute('href') || link.getAttribute('xlink:href') || '';
            if (href.includes('svg-export-text-problems')) link.remove();
          }
          const prefix = 'panel-' + index + '-';
          const idMap = new Map();
          for (const el of svg.querySelectorAll('[id]')) {
            idMap.set(el.id, prefix + el.id);
            el.id = prefix + el.id;
          }
          for (const el of svg.querySelectorAll('*')) {
            for (const attr of [...el.attributes]) {
              let value = attr.value;
              for (const [oldId,newId] of idMap) value=value.split('url(#'+oldId+')').join('url(#'+newId+')');
              if (value!==attr.value) el.setAttribute(attr.name,value);
            }
          }
          const title = document.createElementNS('http://www.w3.org/2000/svg','title');
          const desc = document.createElementNS('http://www.w3.org/2000/svg','desc');
          title.id=prefix+'title';title.textContent=window.__nativePages[index].getAttribute('name');
          desc.id=prefix+'desc';desc.textContent=window.__nativePages[index].getAttribute('id')==='f'
            ? 'Six paired architecture comparisons: original HiGAN+ on the left, current ALPHA on the right, with change explanations beneath each pair.'
            : 'End-to-end ALPHA flow, with explanation and objective formulas beneath the diagram.';
          svg.insertBefore(desc,svg.firstChild);svg.insertBefore(title,svg.firstChild);
          svg.setAttribute('role','img');svg.setAttribute('aria-labelledby',title.id+' '+desc.id);
          return new XMLSerializer().serializeToString(svg);
        }, i);
        svgPages.push(svg);
      }
      console.log(result.name, result.edges, 'native edges rendered');
    }
    const success = results.every(r => r.vertices === r.expectedVertices && r.links === r.expectedLinks && r.edges === r.expectedEdges && r.dangling_edges === 0 && r.unrendered_edges === 0 && r.native_edge_box_transit.length === 0 && r.visible_markup_labels.length === 0);
    fs.writeFileSync(path.join(folder, 'native_renderer_verification.json'), JSON.stringify({ success,
      renderer: 'https://viewer.diagrams.net/js/viewer-static.min.js', network_during_render: false,
      drawio_sha256: crypto.createHash('sha256').update(xml).digest('hex'),
      gpu_disabled: true, full_editor_ui_tested: false, pages: results }, null, 2));
    if (makePreview) {
      const links=report.diagrams.map((name,i)=>`<a href="#p${i}">${name.replace(/&/g,'&amp;')}</a>`).join(' ');
      const html='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>ALPHA — training panels and module comparison</title><style>body{margin:0;background:#eee;font-family:Arial,sans-serif;color:#202020}nav{padding:20px;display:flex;gap:24px;flex-wrap:wrap}a{color:#202020}section{margin:20px 0;background:white}.diagram-container{width:100%;overflow-x:auto}svg{width:100%;min-width:'+canvasWidth+'px;display:block}@media print{nav{display:none}.diagram-container{overflow-x:visible}svg{min-width:0}section{break-after:page;margin:0}}</style><nav>'+links+'</nav>'+svgPages.map((svg,i)=>'<section id="p'+i+'"><div class="diagram-container">'+svg+'</div></section>').join('')+'</html>';
      fs.writeFileSync(path.join(folder,'preview.html'),html);
    }
    if (!success) throw new Error('Native codec/renderer verification failed');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode=1; });
