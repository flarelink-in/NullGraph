// Bundle the real site demo for an isolated browser drive (no React/server/package publication).
// Output is temporary in the site's public directory; remove it after driving.
import { createRequire } from 'node:module';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
const site = process.env.NG_SITE ?? 'D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts';
const esbuild = createRequire(join(site, 'package.json'))('esbuild');
const out = join(site, 'public/.ng-palette-validation');
mkdirSync(out, { recursive: true });
const alias = Object.fromEntries(['agents', 'geometry', 'loaders', 'visibility', 'text', 'debug-ui', 'picking', 'profiler', 'materials', 'cameras', 'post', 'inference']
    .map(name => [`null-graph/${name}`, resolve(`dist/${name}.js`)]));
alias['null-graph'] = resolve('dist/index.js');
await esbuild.build({
    stdin: { contents: `
import { NullGraph, Camera } from 'null-graph';
import { Picker } from 'null-graph/picking';
import { setupHerdWolvesGPU } from './src/data/engine-demos/demos/AgentsExperiments/HerdWolvesGPU/index.ts';
import { HERD_DEFAULTS } from './src/data/engine-demos/demos/AgentsExperiments/HerdWolvesGPU/options.ts';
const engine = new NullGraph();
const canvas = document.querySelector('canvas');
globalThis.fonts = document.fonts;
const init = await engine.init(canvas, { desiredFeatures: ['timestamp-query'] });
if (!init.success) throw new Error(init.error);
const camera = new Camera(45, 1, 0.1, 2000);
camera.updateView([0, 110, 130], [0, 0, 0]);
const state = { ...HERD_DEFAULTS, paused: false, anime: false, showNames: true };
const input = { x: -2, y: -2, buttons: 0, keys: [], wheel: 0, click: null };
const pick = Picker.prototype.pick;
Picker.prototype.pick = async function(...args) {
    const hit = await pick.apply(this, args);
    if (hit) window.lastHerdHit = { index: hit.index, pixel: args.slice(0, 2) };
    return hit;
};
const demo = await setupHerdWolvesGPU(engine, camera, () => state, () => input);
let latest = [], frames = 0, running = true;
function frame() {
    if (!running) return;
    engine.updateCamera(camera);
    latest = demo.update(performance.now() / 1000);
    engine.render(latest); demo.afterRender(); frames++;
    requestAnimationFrame(frame);
}
frame();
window.herd = { state, input, engine, camera, demo,
    get frames() { return frames; },
    get palettes() { return latest.filter(s => s.name === 'Herd visible animation palette').length; },
    get stages() { return latest.map(s => s.name); },
    get draws() { return latest.flatMap(s => s.batches ?? []).filter(b => b.isIndirect).map(b => ({ name: b.name, count: b.currentInstanceCount })); },
    resize(w, h) { canvas.width = w; canvas.height = h; engine.resize(w, h); camera.updateProjection(w / h); },
    stop() { running = false; demo.destroy(); }
};
console.log('isolated Herd demo ready');
`, resolveDir: site, sourcefile: 'palette-demo.ts', loader: 'ts' },
    alias, bundle: true, format: 'esm', platform: 'browser', outfile: join(out, 'demo.js'),
    loader: { '.woff2': 'dataurl' },
    plugins: [{ name: 'font-url', setup(build) { build.onResolve({ filter: /\.woff2\?url$/ }, args => ({ path: createRequire(join(site, 'package.json')).resolve(args.path.replace('?url', '')) })); } }],
});
writeFileSync(join(out, 'index.html'), '<!doctype html><html><head><title>Herd palette validation</title></head><body style="margin:0;background:#111"><canvas width="768" height="768"></canvas><script type="module" src="./demo.js"></script></body></html>');
console.log(out);