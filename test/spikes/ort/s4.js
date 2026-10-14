// Spike S4: what does ONNX Runtime cost to start? See docs/agents-design.md §13 (C0) and
// docs/agents-ort-design.md.
//
//   Node + Dawn:   npm run spike:s4
//   Chrome:        npm run drive -- --serve . "/test/spikes/ort/index.html?spike=s4" --wait-for "S4 done" --timeout 300
//
// One cold process or page per run: everything here happens once per page load. Timed, in order:
// loading the bundle; the first session (the WASM is fetched, compiled and started, a device is
// made); the first run (the pipelines are compiled); a second run; then a second session and its
// first run. Beside them, what the built-in path needs for the same network: reading the .onnx
// file with mlpFromOnnx, and compiling the WGSL kernel.
import { mlpFromOnnx } from '../../../dist/inference.js';
import { harnessEnv } from '../../gpu/harness-env.mjs';
import { LAYERS, describe, fmt, inBrowser, loadOrt, mlpForward, ms, onnxOf, random, readF32, tanhNetwork, wgslKernel } from './common.mjs';

const { log, check, finish } = harnessEnv('__ORT_SPIKE_RESULTS__');
const N = 20000, [OBS, ACT] = [LAYERS[0], LAYERS[LAYERS.length - 1]];
const t = {};
const timed = async (name, work) => { const t0 = performance.now(); const result = await work(); t[name] = performance.now() - t0; return result; };

const started = performance.now();
const { ort, timings } = await loadOrt();
Object.assign(t, { importMs: timings.importMs, ...(timings.readWasmMs === undefined ? {} : { readWasmMs: timings.readWasmMs }) });
const network = tanhNetwork(), bytes = onnxOf(network);
const options = { executionProviders: [{ name: 'webgpu' }], preferredOutputLocation: 'gpu-buffer', enableGraphCapture: true };
ort.env.webgpu.adapter = await timed('adapterMs', () => navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }));

const session = await timed('firstSessionMs', () => ort.InferenceSession.create(bytes, options));
const device = await ort.env.webgpu.device;
const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const obsData = Float32Array.from({ length: N * OBS }, () => random() * 2 - 1);
const obs = device.createBuffer({ size: N * OBS * 4, usage: STORAGE }), act = device.createBuffer({ size: N * ACT * 4, usage: STORAGE });
device.queue.writeBuffer(obs, 0, obsData);
const feeds = { obs: ort.Tensor.fromGpuBuffer(obs, { dataType: 'float32', dims: [N, OBS] }) };
const fetches = { act: ort.Tensor.fromGpuBuffer(act, { dataType: 'float32', dims: [N, ACT] }) };
const run = async s => { await s.run(feeds, fetches); await device.queue.onSubmittedWorkDone(); };
await timed('firstRunMs', () => run(session));
t.readyMs = performance.now() - started;                     // from the first line of this file to the first actions
await timed('secondRunMs', () => run(session));
const got = await readF32(device, act, 64 * ACT);
let worst = 0;
for (let i = 0; i < 64; i++) mlpForward(network, obsData.subarray(i * OBS, (i + 1) * OBS)).forEach((v, j) => { worst = Math.max(worst, Math.abs(v - got[i * ACT + j])); });
check('S4: the first runs give what mlpForward gives', worst <= 1e-4, `worst ${worst.toExponential(1)}`);

const second = await timed('secondSessionMs', () => ort.InferenceSession.create(bytes, options));
const act2 = device.createBuffer({ size: N * ACT * 4, usage: STORAGE });
const fetches2 = { act: ort.Tensor.fromGpuBuffer(act2, { dataType: 'float32', dims: [N, ACT] }) };
await timed('secondSessionFirstRunMs', async () => { await second.run(feeds, fetches2); await device.queue.onSubmittedWorkDone(); });

// The built-in path, for the same network on the same device.
const read = await timed('mlpFromOnnxMs', async () => mlpFromOnnx(bytes));
const kernel = wgslKernel(read.network);
await timed('wgslPipelineMs', () => device.createComputePipelineAsync({ layout: 'auto',
    compute: { module: device.createShaderModule({ code: kernel.code }), entryPoint: 'cs_main' } }));

const sizes = {};
if (inBrowser) {
    for (const entry of performance.getEntriesByType('resource')) {
        const name = entry.name.split('/').pop();
        if (/ort[-.]/.test(name)) sizes[name] = { bytes: entry.decodedBodySize, transferred: entry.transferSize, ms: Math.round(entry.duration) };
    }
} else {
    const { gzipSync } = await import('node:zlib'), { readFile } = await import('node:fs/promises'), { fileURLToPath } = await import('node:url');
    for (const name of ['ort-wasm-simd-threaded.asyncify.wasm', 'ort-wasm-simd-threaded.asyncify.mjs', 'ort.webgpu.bundle.min.mjs']) {
        const data = await readFile(fileURLToPath(new URL(`../../../node_modules/onnxruntime-web/dist/${name}`, import.meta.url)));
        sizes[name] = { bytes: data.length, gzip: gzipSync(data).length };
    }
}

log(`\n== S4 · ${inBrowser ? 'Chrome' : 'Node/Dawn'} · ${describe(device)} — ONNX Runtime Web ${ort.env.versions?.web ?? ''}, network ${LAYERS.join('→')}, ${fmt(N)} agents`);
const line = (label, value) => log(`   ${label.padEnd(64)}${ms(value).padStart(9)} ms`);
line('loading the bundle', t.importMs);
if (t.readWasmMs !== undefined) line('reading the WASM from disk (Node; a browser fetches it instead)', t.readWasmMs);
line('asking for an adapter', t.adapterMs);
line('the first session: the WASM compiled and started, a device made', t.firstSessionMs);
line('the first run: ONNX Runtime\'s pipelines compiled', t.firstRunMs);
line('from the first line to the first actions', t.readyMs);
line('a second run', t.secondRunMs);
line('a second session of the same model', t.secondSessionMs);
line('its first run', t.secondSessionFirstRunMs);
line('the built-in path: mlpFromOnnx on the same file', t.mlpFromOnnxMs);
line('the built-in path: the WGSL kernel\'s pipeline compiled', t.wgslPipelineMs);
for (const [name, s] of Object.entries(sizes)) {
    log(`   ${name}: ${fmt(s.bytes)} B${s.gzip ? `, ${fmt(s.gzip)} B gzipped` : ''}${s.transferred !== undefined ? `, ${fmt(s.transferred)} B transferred in ${s.ms} ms` : ''}`);
}
await device.queue.onSubmittedWorkDone();
await second.release();
await session.release();
log(`\nS4 JSON ${JSON.stringify({ spike: 'S4', host: inBrowser ? 'Chrome' : 'Node/Dawn', adapter: describe(device), ms: t, sizes })}`);
await finish('S4: measured.');
log('S4 done');
