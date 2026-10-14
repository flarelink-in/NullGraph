// GPU integration test for null-graph/inference's ONNX Runtime adapter: step 9, phase 9.1
// (docs/agents-ort-design.md §5, §6, §10).
//
//   A29  the anchor. ONNX Runtime makes the device and the engine adopts it. A model is loaded,
//        run and destroyed, and the device outlives it: it still answers a read-back and runs the
//        engine's stages. A captured model loaded after that release runs 100 times, right at runs
//        1, 2 and 100 (without an anchor, spike S5 saw the second run fail on AMD). A model
//        without graph capture runs too. Only runtime.destroy() ends the device.
//   A32  (phase 9.2) PyTorch's two-file exports, through the runtime with ONNX Runtime's own
//        externalData: their outputs equal PyTorch's, and the reader's from the same two files.
//        The other half, the reader's MlpBrain against ONNX Runtime, is A28 in test/onnx-gpu.
//
// The runtime's own session is the only one alive across the release, so the device survives
// only if the anchor holds it.
//
// Two hosts, one file: `npm run test:gpu -- --only inference` (Node: Dawn as navigator.gpu, ORT's
// asyncify WASM with one thread), `npm run test:gpu:browser -- --only inference`.
import { NullGraph } from '../../dist/index.js';
import { mlpForward, rand01 } from '../../dist/agents.js';
import { OrtRuntime, mlpFromOnnx } from '../../dist/inference.js';
import { onnxFromNetwork } from '../../tools/agents/onnx.mjs';
import { harnessEnv } from '../gpu/harness-env.mjs';

const { log, check, fatal, finish, inBrowser } = harnessEnv('__INFERENCE_RESULTS__');
const ORT_DIST = new URL('../../node_modules/onnxruntime-web/dist/', import.meta.url);

// ---------------------------------------------------------------------------
// ORT bootstrap, per host (as test/onnx-gpu/run.js)
// ---------------------------------------------------------------------------
if (!globalThis.self) globalThis.self = globalThis;   // ORT's bundle expects a worker-style global
let ort;
try {
    ort = await import(new URL('ort.webgpu.bundle.min.mjs', ORT_DIST).href);
} catch (e) {
    fatal(`onnxruntime-web could not be loaded (${e.message}) — it is a devDependency; run npm install.`);
    throw e;
}
ort.env.wasm.numThreads = 1;                          // no SharedArrayBuffer / cross-origin isolation
if (inBrowser) {
    ort.env.wasm.wasmPaths = ORT_DIST.href;
} else {
    const { readFile } = await import('node:fs/promises');
    const { fileURLToPath } = await import('node:url');
    const mjs = new URL('ort-wasm-simd-threaded.asyncify.mjs', ORT_DIST);
    const wasm = new URL('ort-wasm-simd-threaded.asyncify.wasm', ORT_DIST);
    ort.env.wasm.wasmPaths = { mjs: mjs.href, wasm: wasm.href };
    ort.env.wasm.wasmBinary = new Uint8Array(await readFile(fileURLToPath(wasm)));
}

// A run that fails can leave ONNX Runtime waiting for ever (spike S5). Nothing here waits longer.
const LIMIT_MS = 20000;
function within(promise, what) {
    let timer;
    const limit = new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`${what}: no answer after ${LIMIT_MS / 1000} s`)), LIMIT_MS); });
    return Promise.race([promise, limit]).finally(() => clearTimeout(timer));
}
const message = e => `${e?.name ?? 'Error'}: ${String(e?.message ?? e).split('\n')[0].slice(0, 160)}`;

// ---------------------------------------------------------------------------
// The runtime, and the engine on its device
// ---------------------------------------------------------------------------
const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
let runtime;
const started = performance.now();
try {
    runtime = await within(OrtRuntime.create(ort, { adapter }), 'OrtRuntime.create');
} catch (e) {
    fatal(`OrtRuntime.create failed: ${message(e)}`);
    throw e;
}
const createMs = performance.now() - started;
const device = runtime.device;
const engine = new NullGraph();
const init = await engine.init(null, { device, width: 8, height: 8 });
if (!init.success) {
    fatal(init.error);
    throw new Error(init.error);
}
log(`A29: ONNX Runtime ${ort.env.versions?.web ?? ''} · OrtRuntime.create took ${createMs.toFixed(0)} ms`);
check('A29: the engine adopts the device ONNX Runtime made (one device, not the engine\'s to destroy)',
    engine.device === device && engine.ctx.ownsDevice === false && (await ort.env.webgpu.device) === device,
    `same device: ${engine.device === device}, ownsDevice: ${engine.ctx.ownsDevice}`);

// ---------------------------------------------------------------------------
// One network, three models of it
// ---------------------------------------------------------------------------
const N = 4096, ROWS = 256, OBS = 24, ACT = 2;
const layer = (input, output, stream) => ({ input, output, activation: 'tanh',
    weights: Float32Array.from({ length: input * output }, (_, k) => (rand01(k, stream, 6) * 2 - 1) * Math.sqrt(6 / (input + output))),
    biases: Float32Array.from({ length: output }, (_, k) => (rand01(k, stream, 7) * 2 - 1) * 0.1) });
const network = [layer(OBS, 32, 1), layer(32, 32, 2), layer(32, ACT, 3)];
const bytes = onnxFromNetwork(network, { name: 'a29' });
const observations = stream => Float32Array.from({ length: N * OBS }, (_, k) => rand01(k, stream, 0) * 2 - 1);

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const obsData = observations(11);
const obs = device.createBuffer({ label: 'A29 obs', size: N * OBS * 4, usage: STORAGE });
device.queue.writeBuffer(obs, 0, obsData);
const act = name => device.createBuffer({ label: `A29 act ${name}`, size: N * ACT * 4, usage: STORAGE });
const bindings = output => ({ inputs: { obs: { buffer: obs, dims: [N, OBS] } }, outputs: { act: { buffer: output, dims: [N, ACT] } } });

// Test-only read-backs (forbidden in a frame loop, legal here).
async function read(buffer, floats) {
    const staging = device.createBuffer({ size: floats * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, floats * 4);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
}
/** The largest difference from mlpForward over the first ROWS agents. */
async function worst(output, data) {
    const got = await read(output, ROWS * ACT);
    let w = 0;
    for (let i = 0; i < ROWS; i++) {
        mlpForward(network, data.subarray(i * OBS, (i + 1) * OBS)).forEach((v, j) => { w = Math.max(w, Math.abs(v - got[i * ACT + j])); });
    }
    return w;
}

// Model A: loaded, run, destroyed. After it, only the anchor holds the device.
const actA = act('A');
try {
    const a = await within(runtime.load(bytes), 'load A');
    a.bind(bindings(actA));
    await within(a.run(), 'A, run 1');
    const w = await worst(actA, obsData);
    check('A29: a captured model runs on the runtime\'s device and gives mlpForward\'s actions', w <= 1e-4, `worst ${w.toExponential(1)}`);
    await within(a.destroy(), 'A destroy');
} catch (e) {
    check('A29: a captured model runs on the runtime\'s device and gives mlpForward\'s actions', false, message(e));
}

let alive;
try {
    await within(read(actA, 4), 'a read-back after the release');
    alive = true;
} catch (e) {
    alive = message(e);
}
check('A29: with every model released, the anchor keeps the device: a read-back still works', alive === true, alive === true ? '' : alive);

// The engine's own work on the device, after the release.
const doubled = device.createBuffer({ label: 'A29 doubled', size: N * ACT * 4, usage: STORAGE });
const stage = engine.createComputeStage({
    name: 'A29 double', dispatchCount: [Math.ceil(N * ACT / 64), 1, 1],
    shaderCode: /* wgsl */ `
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
        @compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) g: vec3<u32>) {
            if (g.x < arrayLength(&a)) { o[g.x] = a[g.x] * 2.0; }
        }`,
});
engine.setComputePassBindGroup(stage, [{ binding: 0, resource: { buffer: actA } }, { binding: 1, resource: { buffer: doubled } }], 0);
let engineRan;
try {
    engine.render([stage]);
    const [twice, once] = [await read(doubled, ROWS * ACT), await read(actA, ROWS * ACT)];
    engineRan = twice.every((v, i) => v === once[i] * 2 && Number.isFinite(v)) ? true : 'a value differs';
} catch (e) {
    engineRan = message(e);
}
check('A29: and the engine\'s stages still run on it', engineRan === true, engineRan === true ? '' : engineRan);

// Model B: captured, loaded after the release. On AMD without an anchor, its second run failed.
const actB = act('B'), newObs = observations(12), checked = [];
try {
    const b = await within(runtime.load(bytes), 'load B');
    b.bind(bindings(actB));
    for (let r = 1; r <= 100; r++) {
        if (r === 100) device.queue.writeBuffer(obs, 0, newObs);   // a captured session reads its bound buffer as it is now
        await within(b.run(), `B, run ${r}`);
        if (r <= 2 || r === 100) checked.push([r, await worst(actB, r === 100 ? newObs : obsData)]);
    }
    await within(b.destroy(), 'B destroy');
} catch (e) {
    checked.push([0, message(e)]);
}
check('A29: a captured model loaded after that release runs 100 times, right at runs 1, 2 and 100 (new observations before the 100th)',
    checked.length === 3 && checked.every(([, w]) => typeof w === 'number' && w <= 1e-4),
    checked.map(([r, w]) => typeof w === 'number' ? `run ${r}: ${w.toExponential(1)}` : w).join('; '));

// Model C: without graph capture.
const actC = act('C');
try {
    const c = await within(runtime.load(bytes, { capture: false }), 'load C');
    c.bind(bindings(actC));
    for (let r = 1; r <= 3; r++) await within(c.run(), `C, run ${r}`);
    const w = await worst(actC, newObs);
    check('A29: a model without graph capture runs too', w <= 1e-4 && !c.capture, `worst ${w.toExponential(1)}`);
} catch (e) {
    check('A29: a model without graph capture runs too', false, message(e));
}

// ---------------------------------------------------------------------------
// A32: two files, given to ONNX Runtime as its own externalData, through the runtime
// ---------------------------------------------------------------------------
const readJson = async path => {
    const url = new URL(path, import.meta.url);
    return inBrowser ? (await fetch(url)).json() : JSON.parse((await import('node:fs')).readFileSync(url, 'utf8'));
};
const base64 = text => Uint8Array.from(atob(text), c => c.charCodeAt(0));
const fixtures = (await readJson('../onnx-fixtures/fixtures.json')).cases;
const padded = bytes => Math.ceil(bytes / 16) * 16;     // OrtModel's rule: ONNX Runtime reads and writes 16 B at a time
// Under capture, ONNX Runtime 1.29 can fail a model whose weights are in a second file from its
// second run on, depending on the model, the host and what ran before (phase 9.2's results;
// `npm run spike:ort-issues`). So `load` leaves capture off for such a model unless asked (decided
// with the user, the design's §14, item 5): A32 takes that default. Capture is probed after the
// checks, and reported.
const TWO_FILES = ['torch-outside', 'torch-outside-batch'];
async function twoFiles(name, capture) {
    const fixture = fixtures.find(c => c.name === name), bytes = base64(fixture.onnx);
    const externalData = Object.entries(fixture.externalData).map(([path, file]) => ({ path, data: base64(file.base64) }));
    const { inputs, outputs: { maker } } = fixture, width = inputs[0].length, outWidth = maker[0].length;
    const rows = fixture.holds.inputs[0].dims[0] === 1 ? 1 : inputs.length;   // a fixed batch of one runs row by row
    const model = await within(runtime.load(bytes, { externalData, ...(capture === undefined ? {} : { capture }) }), `load ${name}`);
    const input = device.createBuffer({ size: padded(rows * width * 4), usage: STORAGE }), output = device.createBuffer({ size: padded(rows * outWidth * 4), usage: STORAGE });
    try {
        model.bind({ inputs: { [model.inputs[0].name]: { buffer: input, dims: [rows, width] } },
            outputs: { [model.outputs[0].name]: { buffer: output, dims: [rows, outWidth] } } });
        const got = [];
        for (let r = 0; r < inputs.length; r += rows) {
            device.queue.writeBuffer(input, 0, Float32Array.from(inputs.slice(r, r + rows).flat()));
            await within(model.run(), `${name}, rows from ${r}`);
            got.push(...(await read(output, rows * outWidth)));
        }
        return { fixture, bytes, externalData, inputs, maker, outWidth, rows, got, captured: model.capture };
    } finally {
        await within(model.destroy(), `${name} destroy`);
    }
}
for (const name of TWO_FILES) {
    const label = `A32: ${name}, two files: through the runtime with ONNX Runtime's externalData, without graph capture by default, its outputs equal PyTorch's, and the reader's from the same files, within 1e-4`;
    try {
        const { bytes, externalData, inputs, maker, outWidth, rows, got, captured } = await twoFiles(name);
        if (captured) throw new Error('load captured a model whose weights are in a second file');
        const { network } = mlpFromOnnx(bytes, { externalData });
        let againstMaker = 0, againstReader = 0;
        inputs.forEach((row, r) => {
            const reader = mlpForward(network, Float32Array.from(row));
            for (let j = 0; j < outWidth; j++) {
                againstMaker = Math.max(againstMaker, Math.abs(got[r * outWidth + j] - maker[r][j]));
                againstReader = Math.max(againstReader, Math.abs(got[r * outWidth + j] - reader[j]));
            }
        });
        check(label, againstMaker <= 1e-4 && againstReader <= 1e-4 && got.length === inputs.length * outWidth,
            `${inputs.length} rows${rows === 1 ? ', one run each' : ''}; against PyTorch ${againstMaker.toExponential(1)}, against mlpFromOnnx and mlpForward ${againstReader.toExponential(1)}`);
    } catch (e) {
        check(label, false, message(e));
    }
}
// The known failure, probed with capture asked for: reported, never checked.
for (const name of TWO_FILES) {
    try {
        await twoFiles(name, true);
        log(`   probe: ${name} under graph capture ran its runs this time`);
    } catch (e) {
        log(`   probe: ${name} under graph capture failed, as ONNX Runtime 1.29 can: ${message(e)}`);
    }
}

// Only the runtime ends the device: its destroy releases model C, then the anchor. What ONNX
// Runtime then does with its device is its own; it is reported, not checked.
try {
    await within(runtime.destroy(), 'runtime destroy');
    let after;
    try { await within(read(actA, 4), 'a read-back after runtime.destroy()'); after = 'alive'; } catch (e) { after = `gone (${message(e)})`; }
    log(`   after runtime.destroy(), the device is ${after}`);
} catch (e) {
    check('A29: runtime.destroy() releases every model and the anchor', false, message(e));
}
await finish('A29: the anchor holds ONNX Runtime\'s device across model swaps.');
log('inference done');
