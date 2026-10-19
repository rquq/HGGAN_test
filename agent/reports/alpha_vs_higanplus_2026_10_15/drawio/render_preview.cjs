// Offline visual QA only. The deliverable remains the native .drawio file.
const fs = require('fs');
const path = require('path');
const { chromium } = require('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async () => {
  const folder = process.argv[2];
  const browser = await chromium.launch({ executablePath: 'C:/Program Files (x86)/Microsoft/EdgeCore/155.0.4283.45/msedge.exe', headless: true, args: ['--disable-gpu', '--no-sandbox'] });
  try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1100 } });
    await page.setContent(fs.readFileSync(path.join(folder, 'preview.html'), 'utf8'));
    const count = await page.locator('svg').count();
    for (let i = 0; i < count; i++) {
      await page.locator('svg').nth(i).screenshot({ path: path.join(folder, `qa_${String(i).padStart(2, '0')}.png`) });
    }
    console.log(`Rendered ${count} offline preview pages (CPU-only)`);
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
