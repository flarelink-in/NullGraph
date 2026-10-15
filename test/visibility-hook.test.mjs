import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { WgslReflect } from '../node_modules/wgsl_reflect/wgsl_reflect.module.js';
import { buildInstanceDrawWGSL, VisibilitySystem } from '../dist/visibility.js';
import { MegabufferBuilder } from '../dist/geometry.js';
import { standIn } from './stand-in-engine.mjs';

const hook = {
    bindings: '@group(2) @binding(0) var<storage, read> shifts: array<vec4<f32>>;',
    wgsl: `fn deform(pos: vec3<f32>, normal: vec3<f32>, vi: u32, inst: Instance) -> Deformed {
        return Deformed(pos + shifts[inst.sourceIndex].xyz + vec3<f32>(f32(vi) * 0.001, 0, 0), normal);
    }`,
    entries: [{ binding: 0, resource: { buffer: {} } }],
};

test('no-hook WGSL remains byte-identical to D1.1', () => {
    const options = [{}, { fragmentBody: '// custom fragment', extraBindings: '// custom bindings', vertexAttributesWGSL: '// custom attrs' }];
    const hashes = ['4be6e071105e720579f4cca1c50e4cc6f9d6792d69f5d277f3b717223581e35e', 'b76f70bde3d6e8b1f0ab3343fc040ef5fb1a4c68b691fe804ceef1ca3785ed8c'];
    options.forEach((o, i) => assert.equal(createHash('sha256').update(buildInstanceDrawWGSL(o)).digest('hex'), hashes[i]));
});

test('hook WGSL parses, gets absolute vertex index and instance, and deforms before model transform', () => {
    const code = buildInstanceDrawWGSL({ vertexHook: hook });
    assert.match(code, /@builtin\(vertex_index\) vi: u32/);
    assert.match(code, /let deformed = deform\(pos, normal, vi, inst\)/);
    assert.match(code, /inst.model \* vec4<f32>\(deformed.position, 1.0\)/);
    assert.match(code, /normalMatrixFromModel\(inst.model\) \* deformed.normal/);
    const parsed = new WgslReflect(code);
    assert.equal(parsed.entry.vertex.length, 1);
    assert.equal(parsed.entry.fragment.length, 1);
});

function scene() {
    const { engine } = standIn(), bindings = [], configs = [];
    engine.ctx = { cameraUniformBuffer: {}, width: 64, height: 64 };
    engine.device.createBuffer = d => ({ ...d, getMappedRange: () => new ArrayBuffer(d.size), unmap() {}, destroy() {} });
    engine.createBatch = (pass, config) => { configs.push(config); const b = { storageBuffer: {}, indirectBuffer: {} }; pass.batches.push(b); return b; };
    engine.setBatchGeometry = () => {};
    engine.setRenderPassBindGroup = (batch, entries, group) => bindings.push({ entries, group });
    engine.clearBatch = () => {};
    const mega = new MegabufferBuilder(8);
    mega.addMesh('part', new Float32Array(24), [0, 1, 2]);
    const vis = new VisibilitySystem(engine, { maxInstances: 4 });
    vis.setGeometry(mega);
    return { vis, bindings, configs, pass: { batches: [] } };
}

test('hook binds caller resources, fills empty groups and refuses collisions before allocating a batch', () => {
    const { vis, bindings, configs, pass } = scene();
    vis.createInstanceDrawBatch(pass, 'part', 0, { vertexHook: hook });
    assert.deepEqual(bindings.map(b => b.group), [1, 2]);
    assert.deepEqual(bindings[0].entries, []);
    assert.equal(bindings[1].entries, hook.entries);
    assert.match(configs[0].shaderCode, /fn deform/);
    const before = configs.length;
    for (const options of [
        { shaderCode: 'override', vertexHook: hook },
        ...[0, -1, 1.5, NaN, 4].map(group => ({ vertexHook: { ...hook, group } })),
        { materialEntries: [], materialGroupIndex: 2, vertexHook: hook },
        { extraBindings: '@group(2) @binding(1) var<uniform> x: vec4<f32>;', vertexHook: hook },
        { vertexHook: { ...hook, group: 3 } },
        { material: { kind: 'toon', instance: { applyToBatch() {} },
            lighting: { wgslStructs: '@group(2) @binding(0) var<storage, read> lights: array<vec4<f32>>;', wgslLogic: '' } }, vertexHook: hook },
    ]) assert.throws(() => vis.createInstanceDrawBatch(pass, 'part', 0, options), /VisibilitySystem: vertexHook/);
    assert.equal(configs.length, before);
    vis.createInstanceDrawBatch(pass, 'part', 0, { vertexHook: { ...hook, group: 3, bindings: hook.bindings.replace('@group(2)', '@group(3)') } });
    assert.deepEqual(bindings.slice(2).map(b => b.group), [1, 2, 3]);
});

test('hook composes with every stock material family without changing the fragment contract', () => {
    for (const kind of ['basic', 'lambert', 'emissive', 'matcap', 'toon', 'pbr']) {
        const { vis, pass, configs } = scene();
        let applied = 0;
        vis.createInstanceDrawBatch(pass, 'part', 0, { vertexHook: hook,
            material: { kind, instance: { applyToBatch(_batch, group) { assert.equal(group, 1); applied++; } } } });
        assert.equal(applied, 1);
        const parsed = new WgslReflect(configs[0].shaderCode);
        assert.equal(parsed.entry.vertex.length, 1, kind);
        assert.equal(parsed.entry.fragment.length, 1, kind);
    }
});

test('visibility guide: actual vertex-hook example type-checks, bundles and binds the supplied buffer', async () => {
    const root = fileURLToPath(new URL('../', import.meta.url));
    const guide = readFileSync(join(root, 'docs/api/visibility.md'), 'utf8');
    const examples = [...guide.matchAll(/```ts\r?\n([\s\S]*?)```/g)].map(m => m[1]).filter(s => s.includes('export function addDeformedBatch'));
    assert.equal(examples.length, 1);
    const dir = mkdtempSync(join(root, 'test/.visibility-api-'));
    try {
        writeFileSync(join(dir, 'example.ts'), examples[0]);
        writeFileSync(join(dir, 'tsconfig.json'), JSON.stringify({ compilerOptions: {
            target: 'ES2022', module: 'ESNext', moduleResolution: 'Bundler', strict: true,
            noEmit: true, skipLibCheck: true, types: ['@webgpu/types'], lib: ['ES2022', 'DOM', 'DOM.Iterable'],
        }, files: ['example.ts'] }));
        try { execFileSync(process.execPath, [join(root, 'node_modules/typescript/bin/tsc'), '--project', join(dir, 'tsconfig.json')], { cwd: root, stdio: 'pipe' }); }
        catch (error) { throw new Error(`Visibility example TypeScript failed:\n${error.stdout}\n${error.stderr}`); }
        const output = await build({ stdin: { contents: examples[0], loader: 'ts', resolveDir: root },
            bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent' });
        const module = join(dir, 'example.mjs');
        writeFileSync(module, output.outputFiles[0].text);
        const { addDeformedBatch } = await import(pathToFileURL(module).href);
        const { vis, pass, bindings, configs } = scene(), buffer = {};
        addDeformedBatch(vis, pass, 'part', 0, buffer);
        assert.equal(bindings.find(b => b.group === 2).entries[0].resource.buffer, buffer);
        assert.equal(new WgslReflect(configs[0].shaderCode).entry.vertex.length, 1);
    } finally { rmSync(dir, { recursive: true, force: true }); }
});