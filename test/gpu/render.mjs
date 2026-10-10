// npm run render -- <scene.mjs> [--out shot.png] [--width 512] [--height 512] [--frames 1]
//                                [--json report.json] [--allow-blank]
//
// Render any NullGraph scene to a PNG, headless, on this machine's GPU (Dawn) — no browser, no
// server. Built for people AND agents: an agent writes a scene, runs this, reads the PNG, and
// gets a non-zero exit with a reason when the frame is wrong.
//
// A scene is an ES module with one default export, called once with the initialised engine:
//
//   export default async function scene({ engine, width, height, lib }) {
//       // build passes and batches once…
//       return (frame, seconds) => [ /* the render sequence for this frame */ ];
//   }
//
// `lib` carries the built library — lib.core, lib.geometry, lib.materials, lib.lights,
// lib.cameras, lib.visibility, lib.shadows, lib.environment, lib.post, lib.loaders — plus
// lib.glMatrix, so a scene file needs no import paths and runs from anywhere. The sequence must
// end in a pass with `isMainScreenPass: true`: that pass is the image. `seconds` advances 1/60
// per frame, deterministically.
//
// Exit codes: 0 rendered; 1 a WebGPU error was reported, or the frame is uniform (nothing drew —
// unless --allow-blank); 2 bad arguments, scene failed to load, or the scene threw.

import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { installNodeGPU, describeDevice } from './node-gpu.mjs';
import { encodePNG, pixelStats } from './png.mjs';
import { readTextureRGBA8 } from './readback.mjs';

const argv = process.argv.slice(2);
const flag = (name, fallback) => { const i = argv.indexOf(`--${name}`); return i >= 0 ? argv[i + 1] : fallback; };
const VALUE_FLAGS = new Set(['--out', '--width', '--height', '--frames', '--json']);
const positional = [];
for (let i = 0; i < argv.length; i++) {
    if (VALUE_FLAGS.has(argv[i])) { i++; continue; }
    if (!argv[i].startsWith('--')) positional.push(argv[i]);
}
const sceneArg = positional[0];

function usage(msg) {
    if (msg) console.error(msg);
    console.error('usage: npm run render -- <scene.mjs> [--out shot.png] [--width 512] [--height 512] ' +
        '[--frames 1] [--json report.json] [--allow-blank]');
    process.exit(2);
}
if (!sceneArg) usage('No scene file given.');

const width = Number(flag('width', '512'));
const height = Number(flag('height', '512'));
const frames = Number(flag('frames', '1'));
const out = flag('out', 'render.png');
const jsonPath = flag('json', null);
const allowBlank = argv.includes('--allow-blank');
if (![width, height, frames].every(n => Number.isInteger(n) && n > 0)) usage('--width, --height and --frames must be positive integers.');

const repo = resolve(fileURLToPath(import.meta.url), '../../..');
const dist = (name) => import(pathToFileURL(resolve(repo, 'dist', `${name}.js`)).href);

const host = installNodeGPU();
const t0 = performance.now();

const lib = {
    core: await dist('index'), geometry: await dist('geometry'), materials: await dist('materials'),
    lights: await dist('lights'), cameras: await dist('cameras'), visibility: await dist('visibility'),
    shadows: await dist('shadows'), environment: await dist('environment'), post: await dist('post'),
    loaders: await dist('loaders'), glMatrix: await import('gl-matrix'),
};

const engine = new lib.core.NullGraph();
const init = await engine.init(null, { width, height });
if (!init.success) { console.error(`Engine init failed: ${init.error}`); process.exit(2); }

let sceneFn;
try {
    sceneFn = (await import(pathToFileURL(resolve(sceneArg)).href)).default;
} catch (e) {
    console.error(`Could not load scene ${sceneArg}:\n${e.stack ?? e}`);
    process.exit(2);
}
if (typeof sceneFn !== 'function') {
    console.error(`${sceneArg} has no default export function — export default async function scene({ engine, width, height, lib }) { … }`);
    process.exit(2);
}

let frameFn;
try {
    frameFn = await sceneFn({ engine, width, height, lib });
    if (typeof frameFn !== 'function') throw new Error('the scene must RETURN a function (frame, seconds) => renderSequence');
    for (let f = 0; f < frames; f++) {
        const sequence = await frameFn(f, f / 60);
        if (!Array.isArray(sequence)) throw new Error(`frame ${f}: expected a render sequence array, got ${typeof sequence}`);
        if (!sequence.some(n => n?.isMainScreenPass)) {
            throw new Error(`frame ${f}: the sequence has no pass with isMainScreenPass: true — that pass is the image this tool saves`);
        }
        engine.render(sequence);
    }
} catch (e) {
    await host.settle();
    console.error(`Scene threw: ${e.stack ?? e}`);
    for (const ev of host.events.slice(0, 5)) console.error(`  [${ev.kind}] ${ev.message.split('\n')[0]}`);
    process.exit(2);
}
await host.settle();

const rgba = await readTextureRGBA8(engine.device, engine.screenTexture, width, height, engine.ctx.format);
writeFileSync(out, encodePNG(rgba, width, height));
const stats = pixelStats(rgba, width, height);
const blank = stats.stdDev < 0.5 && stats.coverage < 0.001;

const report = {
    scene: sceneArg, out: resolve(out), width, height, frames,
    device: describeDevice(engine.device), ms: Math.round(performance.now() - t0),
    gpuErrors: host.events, ...stats, blank,
};
if (jsonPath) writeFileSync(jsonPath, JSON.stringify(report, null, 2));

console.log(`Rendered ${sceneArg} → ${report.out} (${width}x${height}, ${frames} frame(s), ${report.ms} ms, ${report.device})`);
console.log(`  luma mean ${stats.meanLuma.toFixed(1)}, std-dev ${stats.stdDev.toFixed(1)}, ` +
    `${(stats.coverage * 100).toFixed(1)}% of pixels differ from the corner (${stats.corner.join(',')})`);

let ok = true;
if (host.events.length) {
    ok = false;
    console.log(`\n${host.events.length} WebGPU error(s):`);
    for (const ev of host.events.slice(0, 8)) console.log(`  [${ev.kind}] ${ev.message.split('\n').slice(0, 3).join('\n      ')}`);
}
if (blank && !allowBlank) {
    ok = false;
    console.log('\nThe frame is uniform — nothing drew. Usual causes: no batch in the main-screen pass; a batch ' +
        'with 0 instances (non-indirect batches with currentInstanceCount 0 are skipped); the camera never ' +
        'updated (engine.updateCamera); geometry outside the view. Pass --allow-blank if a flat frame is intended.');
}
for (const d of host.devices) { try { d.destroy(); } catch { /* gone */ } }
process.exit(ok ? 0 : 1);
