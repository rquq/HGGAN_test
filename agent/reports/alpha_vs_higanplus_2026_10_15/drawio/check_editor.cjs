// Read-only editor-load verification. No training, no model checkpoint loading.
const fs = require('fs');
const path = require('path');
const { chromium } = require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const folder = path.dirname(process.argv[2]);
  const xml = fs.readFileSync(process.argv[2], 'utf8');
  const browser = await chromium.launch({
    executablePath: 'C:/Program Files (x86)/Microsoft/EdgeCore/155.0.4283.45/msedge.exe',
    headless: true,
    args: ['--disable-gpu', '--no-sandbox', '--disable-dev-shm-usage'],
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1800, height: 1200 }, deviceScaleFactor: 1 });
    await page.addInitScript(({ xml }) => {
      if (window.top !== window) return;
      window.__embedEvents = [];
      window.addEventListener('message', (event) => {
        let data;
        try { data = typeof event.data === 'string' ? JSON.parse(event.data) : event.data; }
        catch (_) { return; }
        if (!data || typeof data !== 'object') return;
        if (data.event) window.__embedEvents.push({ event: data.event, error: data.error || null });
        if (data.event === 'init' && !window.__diagramSent) {
          window.__diagramSent = true;
          event.source.postMessage(JSON.stringify({ action: 'load', xml, title: 'ALPHA training atlas', autosave: 0, fit: 1 }), '*');
        }
      });
    }, { xml });
    console.log('Opening official draw.io embedded editor (headless CPU-only)...');
    const wrapper = '<!doctype html><html><body style="margin:0"><iframe style="border:0;width:100vw;height:100vh" src="https://embed.diagrams.net/?embed=1&proto=json&spin=1&libraries=0&noSaveBtn=1&noExitBtn=1"></iframe></body></html>';
    await page.goto('data:text/html,' + encodeURIComponent(wrapper), { waitUntil: 'domcontentloaded', timeout: 45000 });
    try {
      await page.waitForFunction(() => window.__embedEvents.some(e => e.event === 'load'), null, { timeout: 45000 });
    } catch (error) {
      const events = await page.evaluate(() => window.__embedEvents);
      console.log('Observed embed events:', JSON.stringify(events));
      await page.screenshot({ path: path.join(folder, 'editor_attempt.png') });
      throw error;
    }
    await page.waitForTimeout(2000);
    const frame = page.frames().find(f => f.url().includes('embed.diagrams.net'));
    const initial = { events: await page.evaluate(() => window.__embedEvents), text: await frame.locator('body').innerText() };
    const names = [...xml.matchAll(/<diagram id="[^"]+" name="([^"]+)"/g)].map(m => m[1].replace(/&amp;/g, '&'));
    const present = names.filter(name => initial.text.includes(name));
    console.log('Editor acknowledged load; visible page tabs:', present.length, '/', names.length);
    await page.screenshot({ path: path.join(folder, 'editor_loaded.png'), fullPage: false });
    for (const tab of ['01 GAN overview', '06 Style transfer', '08 Style encoder', '10 OCR pretraining']) {
      const locator = frame.getByText(tab, { exact: true }).first();
      if (await locator.count()) {
        await locator.click({ timeout: 5000 });
        await page.waitForTimeout(700);
        await page.screenshot({ path: path.join(folder, 'editor_' + tab.slice(0, 2) + '.png'), fullPage: false });
      }
    }
    const report = { success: true, editor: 'https://embed.diagrams.net/', events: initial.events,
      expected_pages: names, page_tabs_found: present, all_tabs_present: present.length === names.length,
      gpu_disabled: true, note: 'Actual official editor accepted native uncompressed mxfile; screenshots are editor renders.' };
    fs.writeFileSync(path.join(folder, 'editor_verification.json'), JSON.stringify(report, null, 2));
    if (present.length !== names.length) throw new Error('Some page tabs were not observed');
  } finally { await browser.close(); }
})().catch(error => { console.error(error.message); process.exitCode = 1; });
