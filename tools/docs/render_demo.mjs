#!/usr/bin/env node
// Render the README demo animation (docs/assets/ninfer-demo.gif and .mp4) from
// tools/docs/demo/demo.html.
//
// The page is a pure function of time (window.renderAt), so this steps it frame by frame in
// headless Chrome and encodes the screenshots with ffmpeg. The result is identical on every run.
// The animation is an illustration, not a recording; the page says so on every frame.
//
// Needs Node 18+, ffmpeg, Google Chrome (or Chromium) and the playwright-core package:
//
//   cd tools/docs && npm install --no-save playwright-core
//   node tools/docs/render_demo.mjs                  # writes docs/assets/ninfer-demo.{gif,mp4}
//   node tools/docs/render_demo.mjs --out-dir /tmp/demo --keep-frames /tmp/frames
//
// It uses the CPU only. It loads no model and never touches the GPU.
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { mkdirSync, mkdtempSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..', '..');
const FPS = 30;       // frames rendered and used for the MP4
const GIF_FPS = 15;   // a README GIF has to stay light; the MP4 keeps the full 30 fps at 1920x1080
const GIF_WIDTH = 840;

function option(name, fallback) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? resolve(process.argv[i + 1]) : fallback;
}
const outDir = option('--out-dir', join(root, 'docs', 'assets'));
const keepFrames = option('--keep-frames', null);

function loadPlaywright() {
  for (const base of [here, process.cwd()]) {
    try { return createRequire(join(base, 'noop.js'))('playwright-core'); } catch { /* try the next place */ }
  }
  console.error('playwright-core is not installed: run `cd tools/docs && npm install --no-save playwright-core`');
  process.exit(1);
}

function run(command, args) {
  const result = spawnSync(command, args, { stdio: ['ignore', 'ignore', 'pipe'] });
  if (result.status !== 0) {
    console.error(`${command} failed:\n${result.stderr}`);
    process.exit(1);
  }
}

const { chromium } = loadPlaywright();
const browser = await chromium.launch({ channel: 'chrome' }).catch(() => chromium.launch());
const page = await browser.newPage({ viewport: { width: 960, height: 540 }, deviceScaleFactor: 2 });
await page.goto(pathToFileURL(join(here, 'demo', 'demo.html')).href + '?capture=1');
await page.evaluate(() => document.fonts.ready);
const duration = await page.evaluate(() => window.DURATION);

mkdirSync(outDir, { recursive: true });
const scratch = mkdtempSync(join(tmpdir(), 'ninfer-demo-'));
const frames = keepFrames ?? scratch;
mkdirSync(frames, { recursive: true });

const count = Math.round(duration * FPS);
for (let i = 0; i < count; i++) {
  await page.evaluate((t) => window.renderAt(t), i / FPS);
  await page.screenshot({ path: join(frames, `f${String(i).padStart(4, '0')}.png`) });
}
await browser.close();

const pattern = join(frames, 'f%04d.png');
const palette = join(scratch, 'palette.png');
const gif = join(outDir, 'ninfer-demo.gif');
const mp4 = join(outDir, 'ninfer-demo.mp4');
const scale = `fps=${GIF_FPS},scale=${GIF_WIDTH}:-1:flags=lanczos`;
run('ffmpeg', ['-y', '-framerate', String(FPS), '-i', pattern, '-vf', `${scale},palettegen=max_colors=96:stats_mode=diff`, palette]);
run('ffmpeg', ['-y', '-framerate', String(FPS), '-i', pattern, '-i', palette, '-lavfi',
  `${scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle`, '-loop', '0', gif]);
run('ffmpeg', ['-y', '-framerate', String(FPS), '-i', pattern, '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
  '-crf', '18', '-movflags', '+faststart', mp4]);
rmSync(scratch, { recursive: true, force: true });

for (const path of [gif, mp4]) console.log(`${path}: ${(statSync(path).size / 1e6).toFixed(1)} MB`);
