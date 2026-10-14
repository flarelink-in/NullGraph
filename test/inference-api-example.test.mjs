// Step 8, phase 8.3 (docs/agents-onnx-reader-design.md §10), and step 9, phase 9.6
// (docs/agents-ort-design.md §13): the inference guide's examples are type-checked and bundled, as
// the agents guide's are (test/agents-api-example.test.mjs). The reader's runs on the committed
// networks; the ONNX Runtime one runs `useOrtBrain` against a stand-in for ONNX Runtime, on a
// stand-in engine. The export recipe is held to what phase 8.0 measured.
//
// Imports dist/: run `npm run build:lib` first.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { resolve, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { AgentSystem, RuleBrain } from '../dist/agents.js';
import { OrtRuntime } from '../dist/inference.js';
import { herdDefinition } from '../tools/agents/scenario/herd-wolves.mjs';
import { networkFromModel } from '../tools/agents/model.mjs';
import { onnxFromNetwork } from '../tools/agents/onnx.mjs';
import { chain } from './onnx-fixtures/build.mjs';
import { standIn } from './stand-in-engine.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const guideExamples = () => [...readFileSync(join(root, 'docs/api/inference.md'), 'utf8').matchAll(/```ts\r?\n([\s\S]*?)```/g)].map(match => match[1]);

test('inference guide: both examples type-check together, against the real onnxruntime-web', () => {
    const examples = guideExamples();
    assert.equal(examples.length, 2, 'validate every TypeScript example in the guide');
    const dir = mkdtempSync(join(root, 'test/.inference-api-'));
    try {
        examples.forEach((source, n) => writeFileSync(join(dir, `example${n}.ts`), source));
        writeFileSync(join(dir, 'tsconfig.json'), JSON.stringify({
            compilerOptions: { target: 'ES2022', module: 'ESNext', moduleResolution: 'Bundler',
                strict: true, noEmit: true, skipLibCheck: true, types: ['@webgpu/types'],
                lib: ['ES2022', 'DOM', 'DOM.Iterable'] },
            files: examples.map((_, n) => `example${n}.ts`),
        }));
        try {
            execFileSync(process.execPath, [resolve(root, 'node_modules/typescript/bin/tsc'),
                '--project', join(dir, 'tsconfig.json')], { cwd: root, stdio: 'pipe' });
        } catch (error) {
            throw new Error(`The inference guide's examples do not type-check:\n${error.stdout}\n${error.stderr}`);
        }
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});

test('inference guide: the reader\'s example bundles, loads the committed networks and refuses what it should', async () => {
    const guide = readFileSync(join(root, 'docs/api/inference.md'), 'utf8');
    const examples = guideExamples();

    // PyTorch's default call stores the weights outside the file (phase 8.0), and the reader refuses
    // that. Every export call the guide shows must keep them inside.
    const python = [...guide.matchAll(/```python\r?\n([\s\S]*?)```/g)].map(match => match[1]).join('\n');
    const calls = [...guide.matchAll(/torch\.onnx\.export\(([^\n]*)\)/g)].map(match => match[1]);
    assert.ok(/torch\.onnx\.export\(/.test(python), 'the guide shows the export call');
    assert.ok(calls.length >= 2 && calls.every(args => /external_data=False/.test(args)), `an export call without external_data=False: ${calls.join(' | ')}`);

    const dir = mkdtempSync(join(root, 'test/.inference-api-'));
    try {
        const result = await build({ stdin: { contents: examples[0], loader: 'ts', resolveDir: root },
            bundle: true, platform: 'browser', format: 'esm', write: false, logLevel: 'silent' });
        const module = join(dir, 'example.mjs');
        writeFileSync(module, result.outputFiles[0].text);
        const { useOnnxNetwork } = await import(pathToFileURL(module).href);

        const definition = herdDefinition(), { engine, stages, buffers } = standIn();
        const agents = new AgentSystem(engine, definition, { thinkEvery: 3 });
        const alive = () => buffers.filter(buffer => !buffer.destroyed).length;
        const models = Object.fromEntries(['prey', 'wolf'].map(species => {
            const file = `tools/agents/models/herd-wolves/${species}.mlp.json`, model = JSON.parse(readFileSync(join(root, file), 'utf8'));
            return [species, { model, network: networkFromModel(model, definition, file) }];
        }));
        // The two committed networks, as `npm run agents:export` writes them: with their names.
        const exported = (species, names = {}) => onnxFromNetwork(models[species].network, { name: `${species}-mlp`, metadata: {
            species, obsNames: JSON.stringify(models[species].model.obsNames), actNames: JSON.stringify(models[species].model.actNames), ...names } });
        const brains = ['prey', 'wolf'].map(species => useOnnxNetwork(engine, agents, exported(species), species));
        assert.deepEqual(brains.map(brain => brain.dims), [{ obs: 24, act: 2 }, { obs: 24, act: 2 }]);
        agents.advance(definition.dt);
        assert.equal(agents.getComputeStages().filter(stage => stage.name === 'AgentSystem think').length, 2, 'one think pass per species');

        // From here on every call is refused, and a refusal leaves nothing behind: no stage, no buffer.
        const built = stages.length, held = alive();
        const reversed = JSON.stringify([...models.prey.model.obsNames].reverse());
        assert.throws(() => useOnnxNetwork(engine, agents, exported('prey', { obsNames: reversed }), 'prey'), /^Error: The file was trained on other observations than "prey" has\.$/);
        assert.throws(() => useOnnxNetwork(engine, agents, exported('wolf', { actNames: '["turn"]' }), 'wolf'), /^Error: The file was trained on other actions than "wolf" has\.$/);
        // A file from another exporter, for other widths: the reader takes it, and setBrain refuses it.
        const fixtures = JSON.parse(readFileSync(join(root, 'test/onnx-fixtures/fixtures.json'), 'utf8')).cases;
        const bytesOf = name => Uint8Array.from(Buffer.from(fixtures.find(fixture => fixture.name === name).onnx, 'base64'));
        assert.throws(() => useOnnxNetwork(engine, agents, bytesOf('torch-chain'), 'prey'), /the brain maps 5→3; the layout is 24→2\./);
        // PyTorch's default export of anything but a tiny network is two files. The first alone is
        // refused by the reader, naming both ways out; with the second, it is read, and here setBrain
        // refuses its widths (step 9, phase 9.2).
        assert.throws(() => useOnnxNetwork(engine, agents, bytesOf('torch-outside'), 'prey'),
            /^Error: mlpFromOnnx: initializer "0\.weight" is stored outside the file, in "model\.onnx\.data"; pass that file, \{ externalData: \[\{ path: 'model\.onnx\.data', data \}\] \}, or export with the weights inside it \(PyTorch: external_data=False\)\.$/);
        const twoFiles = fixtures.find(fixture => fixture.name === 'torch-outside-batch');
        const externalData = Object.entries(twoFiles.externalData).map(([path, file]) => ({ path, data: Uint8Array.from(Buffer.from(file.base64, 'base64')) }));
        assert.throws(() => useOnnxNetwork(engine, agents, bytesOf('torch-outside-batch'), 'prey', externalData), /the brain maps 13→3; the layout is 24→2\./);
        assert.equal(stages.length, built, 'a refused file builds no stage');
        assert.equal(alive(), held, 'a refused file leaves no buffer behind: the brain made for it is destroyed');

        // A file without names loads: there is nothing to check them against. An ArrayBuffer is taken too.
        const plain = chain(models.prey.network, { form: 'matmul' });
        const again = useOnnxNetwork(engine, agents, plain.buffer.slice(plain.byteOffset, plain.byteOffset + plain.length), 'prey');
        assert.deepEqual(again.dims, { obs: 24, act: 2 });
        agents.destroy();
        for (const brain of [...brains, again]) brain.destroy();
        console.log('inference guide: the reader\'s example bundles, loads both committed networks from .onnx bytes, reads a second file when it is given, and refuses other names, other widths and weights outside the file without it.');
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});

/**
 * A stand-in for onnxruntime-web: its device is the stand-in engine's, and its sessions record
 * their options, report `shapes` (the first session is the runtime's anchor), and run at once.
 */
function fakeOrt(device) {
    const sessions = [], tensors = [];
    const tensor = (name, shape) => ({ name, isTensor: true, type: 'float32', shape });
    const fake = { sessions, tensors, shapes: { obs: ['N', 24], act: ['N', 2] } };
    fake.ort = {
        env: { webgpu: { device: Promise.resolve(device) } },
        InferenceSession: {
            async create(bytes, options) {
                const anchor = sessions.length === 0;
                const session = { options, runs: 0, released: false,
                    inputMetadata: [anchor ? tensor('x', [1]) : tensor('obs', fake.shapes.obs)],
                    outputMetadata: [anchor ? tensor('y', [1]) : tensor('act', fake.shapes.act)],
                    async run() { session.runs++; return {}; },
                    async release() { session.released = true; } };
                sessions.push(session);
                return session;
            },
        },
        Tensor: { fromGpuBuffer: (gpuBuffer, options) => { tensors.push({ gpuBuffer, ...options }); return { gpuBuffer, ...options }; } },
    };
    return fake;
}

test('inference guide: useOrtBrain loads a model, prepares its brain off the frame, and switches with no compile', async () => {
    // The example imports ONNX Runtime, which this test never starts: an empty module stands in.
    const result = await build({ stdin: { contents: guideExamples()[1], loader: 'ts', resolveDir: root },
        bundle: true, platform: 'browser', format: 'esm', write: false, logLevel: 'silent',
        plugins: [{ name: 'no-ort', setup(b) {
            b.onResolve({ filter: /^onnxruntime-web\/webgpu$/ }, () => ({ path: 'ort', namespace: 'no-ort' }));
            b.onLoad({ filter: /.*/, namespace: 'no-ort' }, () => ({ contents: 'export {};', loader: 'js' }));
        } }] });
    const dir = mkdtempSync(join(root, 'test/.inference-api-'));
    try {
        const module = join(dir, 'example1.mjs');
        writeFileSync(module, result.outputFiles[0].text);
        const { useOrtBrain, startWithOrt } = await import(pathToFileURL(module).href);
        assert.equal(typeof startWithOrt, 'function');

        const definition = herdDefinition(), { engine, stages, buffers } = standIn();
        const fake = fakeOrt(engine.device), runtime = await OrtRuntime.create(fake.ort);
        const agents = new AgentSystem(engine, definition);
        agents.setBrain(new RuleBrain('fn think(i: u32, species: u32, tick: u32) {}'));
        const alive = () => buffers.filter(buffer => !buffer.destroyed).length, built = stages.length;
        const { model, brain } = await useOrtBrain(runtime, agents, new Uint8Array([8, 8]), 'prey');
        const session = fake.sessions.at(-1);
        assert.deepEqual(session.options, { executionProviders: [{ name: 'webgpu' }], preferredOutputLocation: 'gpu-buffer', enableGraphCapture: false },
            'the WebGPU provider, outputs on the GPU, no graph capture');
        assert.deepEqual(stages.slice(built).map(stage => [stage.name, stage.async === true]), [['AgentSystem think', true]],
            'one pass, compiled asynchronously by prepareBrain; setBrain compiled nothing');
        const role = buffer => (buffer === agents.obs ? 'obs' : buffer === brain.buffers[0] ? 'results' : 'another buffer');
        assert.deepEqual(fake.tensors.map(t => `${role(t.gpuBuffer)} ${t.dims.join('×')}`), ['obs 4096×24', 'results 4096×2'],
            'the model is bound to the observations and to the brain\'s results');
        agents.advance(definition.dt);
        const thinks = agents.getComputeStages().filter(stage => stage.name === 'AgentSystem think');
        assert.ok(thinks.length === 2 && thinks.includes(stages[built]), 'the prepared pass runs beside the rules');
        assert.equal(session.runs, 1, 'a frame with a step starts a run');

        // A model whose batch axis is fixed at 1: refused when it is bound, and nothing is left behind.
        fake.shapes = { obs: [1, 24], act: [1, 2] };
        const before = stages.length, held = alive();
        await assert.rejects(useOrtBrain(runtime, agents, new Uint8Array([8, 8]), 'wolf'),
            /^Error: OrtBrain: input "obs" is \[1,24\]: its first axis is fixed at 1, and the system has 4096 agents\. Export it with a symbolic first axis/);
        assert.ok(fake.sessions.at(-1).released, 'the refused model is released');
        assert.equal(stages.length, before, 'a refused model builds no stage');
        assert.equal(alive(), held, 'a refused model leaves no buffer behind');

        agents.destroy();
        brain.destroy();
        await model.destroy();
        await runtime.destroy();
        assert.ok(fake.sessions.every(s => s.released), 'every session released, the anchor among them');
        console.log('inference guide: useOrtBrain loads a model without graph capture, prepares its brain asynchronously, switches with no compile, starts runs from frames, and leaves nothing behind when a model is refused.');
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});
