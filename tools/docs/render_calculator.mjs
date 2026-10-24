#!/usr/bin/env node
// Render the README screenshots of the fit calculator (docs/assets/config-calculator-{light,dark}.png)
// from docs/config-calculator.html.
//
// GitHub strips scripts from READMEs, so the README shows these pictures and links them to the live
// page on GitHub Pages (https://ashalliants.github.io/ninfer-3090/config-calculator.html). Re-run
// this whenever the calculator's look or default result changes, so the picture matches the page.
//
// Needs Node 18+, Google Chrome (or Chromium) and the playwright-core package:
//
//   cd tools/docs && npm install --no-save playwright-core
//   node tools/docs/render_calculator.mjs                 # writes docs/assets/config-calculator-*.png
//   node tools/docs/render_calculator.mjs --out-dir /tmp/calc
//
// It uses the CPU only. It loads no model and never touches the GPU.
import { createRequire } from 'node:module';
import { mkdirSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..', '..');

function option(name, fallback) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? resolve(process.argv[i + 1]) : fallback;
}
const outDir = option('--out-dir', join(root, 'docs', 'assets'));

function loadPlaywright() {
  for (const base of [here, process.cwd()]) {
    try { return createRequire(join(base, 'noop.js'))('playwright-core'); } catch { /* try the next place */ }
  }
  console.error('playwright-core is not installed: run `cd tools/docs && npm install --no-save playwright-core`');
  process.exit(1);
}

const { chromium } = loadPlaywright();
const browser = await chromium.launch({ channel: 'chrome' }).catch(() => chromium.launch());
mkdirSync(outDir, { recursive: true });

for (const scheme of ['light', 'dark']) {
  const page = await browser.newPage({ viewport: { width: 1180, height: 900 }, deviceScaleFactor: 2, colorScheme: scheme });
  await page.goto(pathToFileURL(join(root, 'docs', 'config-calculator.html')).href);
  await page.evaluate(() => document.fonts.ready);
  // The header plus the Configure / Result panels: the part a reader would interact with.
  const top = await page.locator('.wrap').boundingBox();
  const grid = await page.locator('.grid').first().boundingBox();
  const out = join(outDir, `config-calculator-${scheme}.png`);
  await page.screenshot({
    path: out,
    clip: { x: top.x, y: 0, width: top.width, height: Math.ceil(grid.y + grid.height + 24) },
  });
  console.log(`wrote ${out} (${Math.round(statSync(out).size / 1024)} KiB)`);
  await page.close();
}
await browser.close();
