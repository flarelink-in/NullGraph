// Exercise the site's actual cow/fox hooks, not a second implementation of their blend logic.
// node test/gpu/run-harness.mjs test/spikes/animated-models/visible-palette.mjs __HERD_PALETTE_RESULTS__
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { resolve, join } from 'node:path';
import { mat4 } from 'gl-matrix';
import { NullGraph, Camera } from '../../../dist/index.js';
import { VisibilitySystem, InstanceBuffer } from '../../../dist/visibility.js';
import { MegabufferBuilder } from '../../../dist/geometry.js';
import { harnessEnv } from '../../gpu/harness-env.mjs';
import { readTextureRGBA8 } from '../../gpu/readback.mjs';

const env = harnessEnv('__HERD_PALETTE_RESULTS__');
const site = process.env.NG_SITE ?? 'D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts';
const esbuild = createRequire(join(site, 'package.json'))('esbuild');
const source = readFileSync(join(site, 'src/data/engine-demos/demos/AgentsExperiments/HerdWolvesGPU/models.ts'), 'utf8');
// Preserve the old strided producer as a controlled baseline; the linear mode is the shipped code.
const oldSource = source
    .replace(/let linear = gid\.x \+ gid\.y \* \$\{Math\.ceil\(capacity \/ 64\) \* 64\}u;\s*let slot = linear \/ animalParts;/,
        'let slot = gid.x;')
    .replace('let part = linear % animalParts;', 'let part = gid.y;');
assert.notEqual(oldSource, source, 'baseline must change the producer');
const bundle = await esbuild.build({
    stdin: { contents: oldSource, resolveDir: join(site, 'src'), loader: 'ts' },
    bundle: true, write: false, format: 'esm', platform: 'node',
    alias: { 'null-graph/loaders': resolve('dist/loaders.js'), 'null-graph/visibility': resolve('dist/visibility.js') },
});
const app = await import(`data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString('base64')}`);
const linearBundle = await esbuild.build({
    stdin: { contents: source, resolveDir: join(site, 'src'), loader: 'ts' },
    bundle: true, write: false, format: 'esm', platform: 'node',
    alias: { 'null-graph/loaders': resolve('dist/loaders.js'), 'null-graph/visibility': resolve('dist/visibility.js') },
});
const linearApp = await import(`data:text/javascript;base64,${Buffer.from(linearBundle.outputFiles[0].text).toString('base64')}`);
globalThis.fetch = async url => {
    const bytes = readFileSync(join(site, 'public', url));
    return { ok: true, arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) };
};
const models = await app.loadHerdModels({ textureManager: { load: async () => ({}) } });
const engine = new NullGraph();
const init = await engine.init(null, { width: 512, height: 512, desiredFeatures: ['timestamp-query'] });
assert.ok(init.success, init.error);
const device = engine.device;
const camera = new Camera(45, 1, 0.1, 1000);
const own = [];
const keep = value => { own.push(value); return value; };
const upload = data => {
    const b = keep(device.createBuffer({ size: Math.max(16, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }));
    device.queue.writeBuffer(b, 0, data); return b;
};
const max = Number(process.env.NG_SPIKE_MAX ?? 16384);
const counts = [128, 4096, 16384].filter(n => n <= max);
const query = device.features.has('timestamp-query') ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null;
const resolved = query && keep(device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }));
const read = query && keep(device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
function mark(index) {
    const e = device.createCommandEncoder();
    e.beginComputePass({ timestampWrites: { querySet: query, beginningOfPassWriteIndex: index } }).end();
    device.queue.submit([e.finish()]);
}
async function timed(fn) {
    if (!query) { fn(); await device.queue.onSubmittedWorkDone(); return NaN; }
    mark(0); fn(); mark(1);
    const e = device.createCommandEncoder(); e.resolveQuerySet(query, 0, 2, resolved, 0); e.copyBufferToBuffer(resolved, 0, read, 0, 16); device.queue.submit([e.finish()]);
    await read.mapAsync(GPUMapMode.READ); const t = new BigUint64Array(read.getMappedRange().slice(0)); read.unmap();
    return Number(t[1] - t[0]) / 1e6;
}
const median = xs => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
// Amortize the ~65 us timestamp steps observed on these D3D12 adapters.
const repeat = 8;
async function sample(sequence) {
    return await timed(() => { for (let i = 0; i < repeat; i++) engine.render(sequence); }) / repeat;
}
const fragmentBody = `@fragment fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(abs(normalize(in.worldNormal)) * 0.7 + 0.2, 1.0);
}`;
try {
    for (const count of counts) {
        const rows = new InstanceBuffer(count), state = new Float32Array(count * 16), phases = new Float32Array(count);
        const side = Math.ceil(Math.sqrt(count));
        for (let i = 0; i < count; i++) {
            const m = mat4.create(); mat4.translate(m, m, [(i % side - side / 2) * 2.5, 0, (Math.floor(i / side) - side / 2) * 2.5]);
            const name = i % 2 ? 'prey' : 'wolf';
            rows.write(i, m, [0, 0, 0, models[name].radius], i % 2);
            // Deliberately interleave sources and include idle, both blend bands, run and eating.
            const speed = [0, 0.08, 0.3, 0.9, 0.5][i % 5];
            state[i * 16 + 4] = speed * 4; state[i * 16 + 10] = 1; state[i * 16 + 11] = i % 5 === 4 ? 1 : 0;
            phases[i] = (i * 0.137) % 1;
        }
        const agents = upload(state), phase = upload(phases), top = upload(new Float32Array([4, 0, 0, 0]));
        camera.updateView([side * 0.9, side * 2.5, side * 2.5], [0, 0, 0]); camera.updateProjection(1); engine.updateCamera(camera);
        const times = {}, images = {};
        const modes = ['hook', 'palette', 'linear'];
        if (process.env.NG_PALETTE_REVERSE) modes.reverse();
        for (const mode of modes) {
            const vis = new VisibilitySystem(engine, { maxInstances: count });
            const mega = new MegabufferBuilder(8), parts = [];
            // Padding catches absolute vertex-index errors on both species.
            mega.addMesh('padding', new Float32Array(24), [0, 1, 2]); parts.push(0, 0, 0);
            for (const name of ['wolf', 'prey']) { const model = models[name]; mega.addMesh(name, model.vertices, model.indices); parts.push(...model.partOfVertex); }
            vis.setGeometry(mega); vis.setInstances(rows.data, count); vis.upload();
            const pass = engine.createPass({ name: mode, isMainScreenPass: false,
                colorAttachments: [{ view: engine.screenTexture.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
                depthStencilAttachment: { view: engine.ctx.depthTexture.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } }), stages = [];
            for (const [mesh, name] of ['wolf', 'prey'].entries()) {
                const args = [engine, models[name], new Uint32Array(parts), agents, phase, top];
                const palette = mode === 'hook' ? null : (mode === 'linear' ? linearApp : app).animalPalette(...args, count, keep);
                const hook = palette?.hook ?? app.animalHook(...args, keep);
                const batch = vis.createInstanceDrawBatch(pass, name, mesh, { vertexHook: hook, fragmentBody });
                if (palette) stages.push(palette.attach(batch));
            }
            const sequence = [...vis.getComputeStages(), ...stages, pass];
            const samples = [];
            for (let frame = 0; frame < 24; frame++) {
                const ms = await sample(sequence); if (frame >= 8) samples.push(ms);
            }
            times[mode] = median(samples);
            // Diagnostic only: isolated stages have different submission/cache behaviour from total.
            const components = {};
            for (const [label, sequence] of [['cull', vis.getComputeStages()], ['producer', stages], ['draw', [pass]]]) {
                if (!sequence.length) continue;
                const samples = [];
                for (let i = 0; i < 20; i++) {
                    const ms = await sample(sequence);
                    if (i >= 4) samples.push(ms);
                }
                components[label] = Number(median(samples).toFixed(4));
            }
            env.log(`${count} ${mode}: isolated stage medians (not additive) ${JSON.stringify(components)}`);
            images[mode] = await readTextureRGBA8(device, engine.screenTexture, 512, 512, engine.ctx.format);
            // Same batch, decreasing then zero then growing survivor counts. No CPU count readback.
            if (mode !== 'hook') {
                for (const n of [1, 0, count]) { vis.setInstances(rows.data, n); vis.upload(); engine.render(sequence); }
                const again = await readTextureRGBA8(device, engine.screenTexture, 512, 512, engine.ctx.format);
                env.check(`${count} ${mode}: survivor count shrink/empty/grow restores image`, Buffer.compare(Buffer.from(again), Buffer.from(images[mode])) === 0);
            }
            vis.destroy(); engine.clearPasses();
        }
        for (const mode of ['palette', 'linear']) {
            let differences = 0, covered = 0;
            for (let i = 0; i < images.hook.length; i += 4) {
                if (images.hook[i] || images.hook[i + 1] || images.hook[i + 2]) covered++;
                if ([0, 1, 2].some(k => Math.abs(images.hook[i + k] - images[mode][i + k]) > 2)) differences++;
            }
            env.check(`${count} ${mode}: actual cow/fox palette matches legacy hook pixels`, covered > 100 && differences <= 8, `${covered} covered; ${differences} differing pixels`);
        }
        env.log(`${count}: cull + palette + draw median/16 at 512x512: hook ${times.hook.toFixed(3)} ms; palette ${times.palette.toFixed(3)} ms; linear ${times.linear.toFixed(3)} ms`);
        await env.saveImage(`herd-palette-${count}`, images.palette, 512, 512);
    }
    await env.finish('herd visible palette done');
} finally {
    await device.queue.onSubmittedWorkDone(); own.forEach(r => r.destroy()); query?.destroy();
}