// Two faults of ONNX Runtime 1.29's WebGPU provider, found in step 9's phase 9.2
// (docs/agents-ort-design.md, "Phase 9.2 results"), each in its smallest form:
//
//   1. Its kernels read and write 16 B at a time. Given an app's buffer whose values end inside
//      the last 16 B, a write that ends past the buffer is dropped whole on NVIDIA: the last values
//      stay as they were, with no error. OrtModel.bind therefore asks for the values rounded up to
//      16 B.
//   2. Under graph capture, a model whose input has a fixed batch, whose weights are in a second
//      file, and whose output a Gemm writes, fails from its second run with
//      "Cannot set properties of undefined (setting 'Symbol(gpuBufferMetadata)')". With a free batch,
//      with the weights inside the file, or with a node after the Gemm, it runs. Whether it fails
//      can depend on what ran before it in the process; alone, it fails on both GPUs here.
//
//   Node + Dawn:   npm run spike:ort-issues      (NG_DAWN_FLAGS=adapter=AMD or adapter=NVIDIA)
//   Chrome:        npm run drive -- --serve . "/test/spikes/ort/index.html?spike=issues" --wait-for "issues done" --timeout 120
//
// Each model is 13 -> 5, built by hand; each run's output is read back and compared with the CPU
// forward pass. One runtime throughout: its anchor keeps the device, so no case runs on a re-made
// device, which is another way to reach fault 2 on AMD (spike S5).
import { mlpForward } from '../../../dist/agents.js';
import { OrtRuntime, mlpFromOnnx } from '../../../dist/inference.js';
import { harnessEnv } from '../../gpu/harness-env.mjs';
import { attr, floatBytes, model, node, tensor } from '../../onnx-fixtures/build.mjs';
import { describe, inBrowser, loadOrt } from './common.mjs';

const { log, finish } = harnessEnv('__ORT_SPIKE_RESULTS__');
const { ort } = await loadOrt();
const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const W = Array.from({ length: 65 }, (_, k) => Math.sin(k + 1)), B = Array.from({ length: 5 }, (_, k) => k / 10);

/** A 13 -> 5 Gemm, and a Tanh after it when `tanh`; a fixed batch of one or a free one; the weights inside or in w.bin. */
function build({ fixed, outside, tanh }) {
    const batch = fixed ? 1 : 'batch';
    const nodes = [node('Gemm', ['x', 'w', 'b'], [tanh ? 'h' : 'y'], { attributes: [attr('transB', 1)] }), ...(tanh ? [node('Tanh', ['h'], ['y'])] : [])];
    const bytes = model({ nodes, initializers: [tensor('w', [5, 13], outside ? [] : W, outside ? { external: { location: 'w.bin', offset: 0, length: 260 } } : {}), tensor('b', [5], B)],
        inputs: [['x', [batch, 13]]], outputs: [['y', [batch, 5]]] });
    return { bytes, externalData: outside ? [{ path: 'w.bin', data: floatBytes(W) }] : undefined };
}

/** Three runs of one model on a fresh runtime: 'right', 'kept' (our buffer not written where it should be), 'wrong', or the error. */
const runtime = await OrtRuntime.create(ort, { adapter }), device = runtime.device;
async function three(spec, capture, round) {
    const { bytes, externalData } = build(spec), network = mlpFromOnnx(bytes, { externalData: externalData ?? [] }).network;
    const size = n => round ? Math.ceil(n / 16) * 16 : n;
    // OrtModel refuses an unrounded buffer, so the unrounded case goes to ONNX Runtime directly, as an app without OrtModel would.
    const input = device.createBuffer({ size: size(52), usage: STORAGE }), output = device.createBuffer({ size: size(20), usage: STORAGE });
    const session = await ort.InferenceSession.create(bytes, { executionProviders: [{ name: 'webgpu' }], preferredOutputLocation: 'gpu-buffer',
        enableGraphCapture: capture, ...(externalData ? { externalData } : {}) });
    const feeds = { x: ort.Tensor.fromGpuBuffer(input, { dataType: 'float32', dims: [1, 13] }) };
    const fetches = { y: ort.Tensor.fromGpuBuffer(output, { dataType: 'float32', dims: [1, 5] }) };
    const out = [];
    for (let r = 0; r < 3; r++) {
        const x = Float32Array.from({ length: 13 }, (_, k) => Math.cos(k + r));
        device.queue.writeBuffer(output, 0, new Float32Array(5).fill(-7));
        device.queue.writeBuffer(input, 0, x);
        try {
            await session.run(feeds, fetches);
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
            out.push(`failed: ${String(e.message ?? e).split('\n')[0].slice(0, 60)}`);
        }
    }
    await session.release();
    return out.join(' | ');
}

log(`ONNX Runtime ${ort.env.versions?.web ?? ''} issues · ${inBrowser ? 'Chrome' : 'Node/Dawn'} · ${describe(device)}`);
// Fault 2 first: it was found as the first model of a runtime.
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
await runtime.destroy();
await finish('issues: measured.');
log('issues done');
