// Renders scene.mjs under every chain at every zoom, then tiles the frames into one sheet: a row per
// chain, a column per zoom (close, mid, default, far), each frame at half size.
//
//   node test/spikes/anime-look/run.mjs [--out dir] [--chains a,b] [--zooms close,far]
//   NG_SITE_PUBLIC  the test site's public/ folder (see scene.mjs)
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { encodePNG } from '../../gpu/png.mjs';
import { readPNG } from './png-read.mjs';

const here = fileURLToPath(new URL('.', import.meta.url));
const root = resolve(here, '../../..');
const arg = (name, fallback) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : fallback; };
const out = resolve(arg('out', join(root, 'test/spikes/anime-look/out')));
const chains = arg('chains', 'faithful,anime,outline1,ink,ink-ss').split(',');
const zooms = arg('zooms', 'close,mid,default,far').split(',');
const W = 1280, H = 720;
mkdirSync(out, { recursive: true });

const frames = [];
for (const chain of chains) {
    for (const zoom of zooms) {
        const file = join(out, `${chain}-${zoom}.png`);
        execFileSync(process.execPath, [join(root, 'test/gpu/render.mjs'), join(here, 'scene.mjs'), '--out', file, '--width', `${W}`, '--height', `${H}`],
            { env: { ...process.env, NG_LOOK_CHAIN: chain, NG_LOOK_ZOOM: zoom }, stdio: ['ignore', 'pipe', 'inherit'] });
        frames.push(readPNG(readFileSync(file)));
        console.log(`rendered ${chain} at ${zoom}`);
    }
}

// The sheet: each frame halved by averaging 2 × 2 pixels.
const w = W / 2, h = H / 2, sheet = new Uint8Array(zooms.length * w * chains.length * h * 4);
const sheetWidth = zooms.length * w;
frames.forEach((frame, k) => {
    const ox = (k % zooms.length) * w, oy = Math.floor(k / zooms.length) * h;
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            for (let c = 0; c < 4; c++) {
                const at = (dy, dx) => frame.rgba[((2 * y + dy) * W + 2 * x + dx) * 4 + c];
                sheet[((oy + y) * sheetWidth + ox + x) * 4 + c] = (at(0, 0) + at(0, 1) + at(1, 0) + at(1, 1) + 2) >> 2;
            }
        }
    }
});
writeFileSync(join(out, 'sheet.png'), encodePNG(sheet, sheetWidth, chains.length * h));
console.log(`anime-look: ${chains.length} chains × ${zooms.length} zooms → ${join(out, 'sheet.png')} (rows: ${chains.join(', ')}; columns: ${zooms.join(', ')})`);
