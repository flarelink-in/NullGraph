// Four faults of ONNX Runtime's WebGPU provider, found in step 9's phases 9.2 and 9.4
// (docs/agents-ort-design.md, "Phase 9.2 results" and "Phase 9.4 results"), each in its smallest
// form. Measured on 1.29.0; 1.30.0 has the same code for faults 3 and 4.
//
//   1. Its kernels read and write 16 B at a time. Given an app's buffer whose values end inside
//      the last 16 B, a write that ends past the buffer is dropped whole on NVIDIA: the last values
//      stay as they were, with no error. OrtModel.bind therefore asks for the values rounded up to
//      16 B.
//   2. Under graph capture, a model whose input has a fixed batch, whose weights are in a second
//      file, and whose output a Gemm writes, fails from its second run with
//      "Cannot set properties of undefined (setting 'Symbol(gpuBufferMetadata)')". With a free batch,
//      with the weights inside the file, or with a node after the Gemm, it runs. Whether it fails
//      can depend on what ran before it in the process. Phase 9.4 found the cause, fault 3: this
//      is very likely one of its cases.
//   3. Under graph capture, from a session's second run on, ONNX Runtime's JavaScript looks up the
//      session's output buffer through a handle it released at the end of the first run
//      (lib/wasm/wasm-core-impl.ts, webgpuGetBuffer(dataOffset) on the output; the handle table
//      is in ort-wasm-simd-threaded.asyncify.mjs). It finds a buffer only when the allocator has
//      put another buffer's handle at that address since; otherwise the run fails with the error
//      of fault 2. So whether the second run fails depends on what ONNX Runtime allocated and freed
//      between the first two runs: another session created and released, or run and kept. Once
//      the lookup succeeds it stays right, but ONNX Runtime then takes the session's input buffer
//      for its output. Without capture every case runs. OrtRuntime.load leaves capture off unless
//      asked (decided with the user, 2026-10-14).
//   4. Under graph capture, every run from the second keeps an output tensor that
//      InferenceSession.run then drops, with the function that would release it: ONNX Runtime's
//      WASM memory grows with every run, and nothing gives it back. Measured only with `leak=1`
//      (NG_SPIKE_LEAK=1): a million runs with capture and a million without take about 9 minutes
//      under Node.
//
//   Node + Dawn:   npm run spike:ort-issues      (NG_DAWN_FLAGS=adapter=AMD or adapter=NVIDIA)
//   Chrome:        npm run drive -- --serve . "/test/spikes/ort/index.html?spike=issues" --wait-for "issues done" --timeout 120
//   One of fault 3's cases alone, in a fresh process (what ran before decides which case fails):
//                  NG_SPIKE_CASE=2 npm run spike:ort-issues, or &case=2 in Chrome
//
// Each model is 13 -> 5, built by hand. For faults 1 and 2 each run's output is read back and
// compared with the CPU forward pass. One runtime throughout: its anchor keeps the device, so no
// case runs on a re-made device, which is another way to reach fault 2 on AMD (spike S5).
import { mlpForward } from '../../../dist/agents.js';
import { OrtRuntime, mlpFromOnnx } from '../../../dist/inference.js';
import { harnessEnv } from '../../gpu/harness-env.mjs';
import { attr, floatBytes, model, node, tensor } from '../../onnx-fixtures/build.mjs';
import { describe, inBrowser, loadOrt, setting } from './common.mjs';

const { log, finish } = harnessEnv('__ORT_SPIKE_RESULTS__');
const { ort } = await loadOrt();
// Fault 4 reads ONNX Runtime's WASM memory, which it neither exports nor names: it is found among
// what ONNX Runtime passes to WebAssembly.instantiate, before the first session makes the instance.
let wasmMemory;
for (const name of ['instantiate', 'instantiateStreaming']) {
    const original = WebAssembly[name];
    WebAssembly[name] = async (...args) => {
        const result = await original.apply(WebAssembly, args);
        const found = [...Object.values((result.instance ?? result).exports),
            ...Object.values(args[1] ?? {}).flatMap(space => (space && typeof space === 'object' ? Object.values(space) : []))];
        wasmMemory ??= found.find(v => v instanceof WebAssembly.Memory);
        return result;
    };
}
const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const W = Array.from({ length: 65 }, (_, k) => Math.sin(k + 1)), B = Array.from({ length: 5 }, (_, k) => k / 10);
const firstLine = e => String(e?.message ?? e).split('\n')[0].slice(0, 80);

/** A 13 -> 5 Gemm, and a Tanh after it when `tanh`; a fixed batch of one or a free one; the weights inside or in w.bin. */
function build({ fixed, outside, tanh }) {
    const batch = fixed ? 1 : 'batch';
    const nodes = [node('Gemm', ['x', 'w', 'b'], [tanh ? 'h' : 'y'], { attributes: [attr('transB', 1)] }), ...(tanh ? [node('Tanh', ['h'], ['y'])] : [])];
    const bytes = model({ nodes, initializers: [tensor('w', [5, 13], outside ? [] : W, outside ? { external: { location: 'w.bin', offset: 0, length: 260 } } : {}), tensor('b', [5], B)],
        inputs: [['x', [batch, 13]]], outputs: [['y', [batch, 5]]] });
    return { bytes, externalData: outside ? [{ path: 'w.bin', data: floatBytes(W) }] : undefined };
}

const runtime = await OrtRuntime.create(ort, { adapter }), device = runtime.device;
/** A session straight from ONNX Runtime, as an app without OrtModel would make it, and GPU buffers for one row. */
async function session(spec, options = {}) {
    const { bytes, externalData } = build(spec);
    const s = await ort.InferenceSession.create(bytes, { executionProviders: [{ name: 'webgpu' }], ...(externalData ? { externalData } : {}), ...options });
    return { session: s, bytes, externalData };
}
const io = (size = n => Math.ceil(n / 16) * 16) => {
    const input = device.createBuffer({ size: size(52), usage: STORAGE }), output = device.createBuffer({ size: size(20), usage: STORAGE });
    return { input, output, feeds: { x: ort.Tensor.fromGpuBuffer(input, { dataType: 'float32', dims: [1, 13] }) },
        fetches: { y: ort.Tensor.fromGpuBuffer(output, { dataType: 'float32', dims: [1, 5] }) } };
};
const GPU = { preferredOutputLocation: 'gpu-buffer' };

/** Three runs of one model: 'right', 'kept' (our buffer not written where it should be), 'wrong', or the error. */
async function three(spec, capture, round) {
    const { session: s, bytes, externalData } = await session(spec, { ...GPU, enableGraphCapture: capture });
    const network = mlpFromOnnx(bytes, { externalData: externalData ?? [] }).network;
    // OrtModel refuses an unrounded buffer, so the unrounded case goes to ONNX Runtime directly.
    const { input, output, feeds, fetches } = io(round ? undefined : n => n);
    const out = [];
    for (let r = 0; r < 3; r++) {
        const x = Float32Array.from({ length: 13 }, (_, k) => Math.cos(k + r));
        device.queue.writeBuffer(output, 0, new Float32Array(5).fill(-7));
        device.queue.writeBuffer(input, 0, x);
        try {
            await s.run(feeds, fetches);
            const staging = device.createBuffer({ size: 20, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
            const encoder = device.createCommandEncoder();
            encoder.copyBufferToBuffer(output, 0, staging, 0, 20);
            device.queue.submit([encoder.finish()]);
            await staging.mapAsync(GPUMapMode.READ);
            const got = new Float32Array(staging.getMappedRange().slice(0));
            staging.unmap();
            const want = mlpForward(network, x), kept = [...got].findIndex(v => v === -7);
            out.push(want.every((v, j) => Math.abs(v - got[j]) <= 1e-4) ? 'right' : kept >= 0 ? `value ${kept} not written` : 'wrong');
        } catch (e) {
            out.push(`failed: ${firstLine(e)}`);
        }
    }
    await s.release();
    return out.join(' | ');
}

/** Fault 3: five runs of a captured 13 -> 5 Gemm and Tanh (weights inside, a free batch), with `between` done between runs 1 and 2. */
async function second(between, capture = true) {
    const { session: s } = await session({ tanh: true }, { ...GPU, enableGraphCapture: capture }), { feeds, fetches } = io(), out = [];
    const run = async () => { try { await s.run(feeds, fetches); out.push('ran'); } catch (e) { out.push(`failed: ${firstLine(e)}`); } };
    await run();
    const after = await between();
    for (let r = 2; r <= 5; r++) await run();
    await after?.();
    await s.release();
    return out.join(' | ');
}
const other = () => session({ tanh: true });
const CPU_FEEDS = { x: new ort.Tensor('float32', new Float32Array(13).fill(0.5), [1, 13]) };

log(`ONNX Runtime ${ort.env.versions?.web ?? ''} issues · ${inBrowser ? 'Chrome' : 'Node/Dawn'} · ${describe(device)}`);
// Fault 3 first, on a fresh runtime: what ran before decides which case fails.
const keep = async () => { const q = (await other()).session; await q.run(CPU_FEEDS); return () => q.release(); };
const CASES = [
    ['nothing', async () => undefined, true],
    ['another session created and released', async () => { await (await other()).session.release(); }, true],
    ['another session created, run and kept', keep, true],
    ['the same, capture off', keep, false],
];
const only = Number(setting('case', 0));
log(`\nGraph capture on, five runs; between the first two (fault 3${only ? `, case ${only} alone` : ''}):`);
for (const [label, between, capture] of only ? [CASES[only - 1]] : CASES) {
    log(`   ${label.padEnd(40)} ${await second(between, capture)}`);
}
async function faultsTwoAndOne() {
    log('\nGraph capture on, buffers rounded up to 16 B (fault 2):');
    for (const [label, spec] of [['fixed batch, weights in a second file', { fixed: true, outside: true }], ['free batch, weights in a second file', { outside: true }],
        ['fixed batch, weights inside', { fixed: true }], ['fixed batch, second file, Tanh after the Gemm', { fixed: true, outside: true, tanh: true }]]) {
        log(`   ${label.padEnd(48)} ${await three(spec, true, true)}`);
    }
    log('\nBuffers of exactly their values\' size, 20 B for 5 values, and rounded up to 16 B; capture off (fault 1):');
    for (const [label, spec] of [['Gemm', {}], ['Gemm, Tanh', { tanh: true }]]) {
        log(`   ${label.padEnd(12)} values' size:    ${await three(spec, false, false)}`);
        log(`   ${''.padEnd(12)} rounded to 16 B: ${await three(spec, false, true)}`);
    }
}
/** Fault 4: ONNX Runtime's WASM memory over many runs of one session, with capture and without. */
async function leak(runs) {
    log(`\nONNX Runtime's WASM memory over ${runs.toLocaleString('en-US')} runs (fault 4):`);
    for (const capture of [true, false]) {
        const { session: s } = await session({ tanh: true }, { ...GPU, enableGraphCapture: capture }), { feeds, fetches } = io();
        await s.run(feeds, fetches);
        await device.queue.onSubmittedWorkDone();
        const before = wasmMemory?.buffer.byteLength ?? NaN, t0 = performance.now();
        for (let r = 1; r <= runs; r++) {
            await s.run(feeds, fetches);
            if (r % 1000 === 0) await device.queue.onSubmittedWorkDone();    // a few submissions in flight at most (gotchas)
        }
        const after = wasmMemory?.buffer.byteLength ?? NaN;
        log(`   capture ${capture ? 'on ' : 'off'}: ${(before / 1e6).toFixed(2)} MB → ${(after / 1e6).toFixed(2)} MB in ${((performance.now() - t0) / 1000).toFixed(0)} s`);
        await s.release();
    }
}
if (!only) {
    await faultsTwoAndOne();
    if (setting('leak') === '1') await leak(Number(setting('runs', 1_000_000)));
}
await runtime.destroy();
await finish('issues: measured.');
log('issues done');
