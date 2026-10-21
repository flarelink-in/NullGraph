import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { WgslReflect } from '../node_modules/wgsl_reflect/wgsl_reflect.module.js';
import { standIn } from './stand-in-engine.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
test('13.4: physics guide example type-checks, bundles, runs and rejects broken wiring', async () => {
    const guide = readFileSync(join(root, 'docs/api/physics.md'), 'utf8');
    const examples = [...guide.matchAll(/```ts\r?\n([\s\S]*?)```/g)].map(m => m[1]);
    assert.equal(examples.length, 1, 'check every TypeScript example');
    const dir = mkdtempSync(join(root, 'test/.physics-api-'));
    try {
        writeFileSync(join(dir, 'example.ts'), examples[0]);
        writeFileSync(join(dir, 'tsconfig.json'), JSON.stringify({ compilerOptions: {
            target: 'ES2022', module: 'ESNext', moduleResolution: 'Bundler', strict: true,
            noEmit: true, skipLibCheck: true, types: ['@webgpu/types'], lib: ['ES2022', 'DOM', 'DOM.Iterable'],
        }, files: ['example.ts'] }));
        try { execFileSync(process.execPath, [join(root, 'node_modules/typescript/bin/tsc'), '--project', join(dir, 'tsconfig.json')], { stdio: 'pipe' }); }
        catch (e) { throw new Error(`Physics example TypeScript failed:\n${e.stdout}\n${e.stderr}`); }
        let moduleId = 0;
        async function load(source) {
            const output = await build({ stdin: { contents: source, loader: 'ts', resolveDir: root },
                bundle: true, platform: 'browser', format: 'esm', write: false, logLevel: 'silent' });
            const file = join(dir, `example${moduleId++}.mjs`);
            writeFileSync(file, output.outputFiles[0].text);
            return import(pathToFileURL(file).href);
        }
        async function verify(source) {
            const { createContactDemo } = await load(source), { engine, stages, buffers } = standIn();
            const rendered = [];
            engine.render = chain => rendered.push(chain);
            const demo = createContactDemo(engine);
            try {
                assert.deepEqual(demo.contacts.options.layout.alive, { offset: 13, mask: 1 });
                assert.deepEqual(demo.contacts.options.layout.fixed, { offset: 13, mask: 2 });
                assert.deepEqual(demo.contacts.options.layout.velocity, { offset: 4 });
                assert.equal(demo.contacts.options.dt, demo.agents.definition.dt);
                for (const s of stages.filter(s => s.name.startsWith('DiscContacts') || s.name === 'AgentSystem think')) new WgslReflect(s.shaderCode);
                const built = stages.length;
                demo.frame(0); assert.equal(rendered.at(-1).length, 0);
                for (const n of [1, 3]) {
                    demo.frame(n * demo.agents.definition.dt);
                    const chain = rendered.at(-1), contacts = demo.contacts.getComputeStages();
                    assert.equal(chain.filter(s => s.name === 'DiscContacts tick').length, n);
                    for (let at = 0; at < chain.length; at++) if (chain[at].name === 'AgentSystem act') {
                        assert.deepEqual(chain.slice(at + 1, at + 1 + contacts.length), contacts);
                    }
                }
                assert.equal(stages.length, built, 'frame compiles nothing');
            } finally { demo.destroy(); }
            assert.ok(buffers.every(b => b.destroyed), 'example releases every owned buffer');
            assert.ok(stages.every(s => !s.enabled), 'destroy disables every stage');
            const calls = rendered.length; demo.frame(1); demo.destroy(); assert.equal(rendered.length, calls);
        }
        await verify(examples[0]);
        for (const [before, after] of [
            ["agents.addStepStage({ name: 'contacts', stages: contacts.getComputeStages() });", ''],
            ['fixed: { offset: 13, mask: 2 }', 'fixed: { offset: 13, mask: 4 }'],
            ['contacts.destroy();', ''],
        ]) {
            assert.ok(examples[0].includes(before), 'mutation must match the actual guide');
            await assert.rejects(verify(examples[0].replace(before, after)), undefined, `broken example must fail: ${before}`);
        }
        if (process.env.NG_PHYSICS_EXAMPLE_GPU) {
            const { installNodeGPU } = await import('./gpu/node-gpu.mjs');
            const host = installNodeGPU();
            const { NullGraph } = await import('../dist/index.js');
            const engine = new NullGraph();
            const init = await engine.init(null, { width: 8, height: 8 });
            assert.ok(init.success, init.error);
            const { createContactDemo } = await load(examples[0]);
            const demo = createContactDemo(engine), dt = demo.agents.definition.dt;
            const initial = demo.agents.records.data.slice();
            for (let i = 0; i < 60; i++) demo.frame(dt);
            const d = engine.device, b = demo.agents.buffer;
            const dst = d.createBuffer({ size: b.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            try {
                const enc = d.createCommandEncoder(); enc.copyBufferToBuffer(b, 0, dst, 0, b.size); d.queue.submit([enc.finish()]);
                await dst.mapAsync(GPUMapMode.READ);
                const actual = new Float32Array(dst.getMappedRange().slice(0));
                const overlap = rec => {
                    let total = 0;
                    for (let i = 0; i < 256; i++) for (let j = i + 1; j < 256; j++) total += Math.max(0,
                        rec[i * 16 + 3] + rec[j * 16 + 3] - Math.hypot(rec[i * 16] - rec[j * 16], rec[i * 16 + 2] - rec[j * 16 + 2]));
                    return total;
                };
                assert.ok(actual.every(Number.isFinite));
                assert.ok(overlap(actual) < overlap(initial) * 0.1);
                assert.deepEqual(new Uint32Array(actual.buffer).slice(0, 3), new Uint32Array(initial.buffer).slice(0, 3));
                assert.equal((await demo.contacts.readCounters()).bodiesOverMaxRadius, 0);
            } finally { dst.destroy(); demo.destroy(); }
            await host.settle(); assert.deepEqual(host.events, []);
            for (const d of host.devices) d.destroy();
        }
    } finally { rmSync(dir, { recursive: true, force: true }); }
});