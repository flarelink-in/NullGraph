// GPU integration test for the ONNX Runtime Web seam — Cycle A exit criterion 6 (docs/handoff.md).
//
//   O1  ORT runs a model on the WebGPU EP and keeps the output on the GPU ('gpu-buffer')
//   O2  NullGraph ADOPTS ORT's device — init(null, { device }) — instead of creating its own
//   O3  the model's output buffer, bound as-is (no copy, no wrapper), feeds a NullGraph compute
//       stage, and the numbers that come out are the model's numbers transformed by that stage
//   A27 the trained Herd & Wolves networks, written as .onnx by tools/agents/onnx.mjs, give in
//       ONNX Runtime the actions mlpForward gives (docs/agents-cloning-design.md, phase 7.10)
//
// The direction matters (ORT owns, NullGraph adopts): ORT requests the features and limits its
// kernels need when it creates the device, and its maintainers warn a device created with other
// parameters is not guaranteed to work. NullGraph's baseline needs no optional feature, so
// adopting is always safe from this side.
//
// Lifetime contract exercised here (same as vgpu's "reference mode"): keep the output tensor
// alive until the work that reads it has completed, then dispose it.
//
// Two hosts, one file: `npm run test:gpu` (Node: Dawn as navigator.gpu, ORT's asyncify WASM with
// one thread, per vercel-labs/vgpu experiments/ort-init-device), or `npm run dev:serve` →
// http://localhost:3000/test/onnx-gpu/.

import { NullGraph } from '../../dist/index.js';
import { harnessEnv } from '../gpu/harness-env.mjs';
import { addModel } from './tiny-model.mjs';
import { mlpForward } from '../../dist/agents.js';
import { createHerd, herdDefinition } from '../../tools/agents/scenario/herd-wolves.mjs';
import { lockstep, readBuffers } from '../../tools/agents/label-pass.mjs';
import { networkFromModel } from '../../tools/agents/model.mjs';
import { onnxFromNetwork } from '../../tools/agents/onnx.mjs';

const { log, check, fatal, finish, inBrowser } = harnessEnv('__ONNX_RESULTS__');

const ORT_DIST = new URL('../../node_modules/onnxruntime-web/dist/', import.meta.url);

// ---------------------------------------------------------------------------
// ORT bootstrap, per host
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
    // Node cannot fetch file: URLs, so ORT gets the asyncify build's module URL and the WASM bytes.
    // file:// URLs rather than paths: on Windows a "D:\…" path is not an import specifier.
    const { readFile } = await import('node:fs/promises');
    const { fileURLToPath } = await import('node:url');
    const mjs = new URL('ort-wasm-simd-threaded.asyncify.mjs', ORT_DIST);
    const wasm = new URL('ort-wasm-simd-threaded.asyncify.wasm', ORT_DIST);
    ort.env.wasm.wasmPaths = { mjs: mjs.href, wasm: wasm.href };
    ort.env.wasm.wasmBinary = new Uint8Array(await readFile(fileURLToPath(wasm)));
}
ort.env.webgpu.adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });

// ---------------------------------------------------------------------------
// The test
// ---------------------------------------------------------------------------
const N = 8;
let session;
try {
    session = await ort.InferenceSession.create(addModel(N), {
        executionProviders: ['webgpu'],
        preferredOutputLocation: 'gpu-buffer',
    });
} catch (e) {
    fatal(`ORT could not create a WebGPU session: ${e.message}`);
    throw e;
}

// ORT created the device while building the session; NullGraph adopts that exact device.
const ortDevice = await ort.env.webgpu.device;
const engine = new NullGraph();
const init = await engine.init(null, { device: ortDevice, width: 8, height: 8 });
if (!init.success) {
    fatal(init.error);
    throw new Error(init.error);
}
check('O2: NullGraph adopted ONNX Runtime\'s device (no second device, not owned)',
    engine.device === ortDevice && engine.ctx.ownsDevice === false,
    `same device: ${engine.device === ortDevice}, ownsDevice: ${engine.ctx.ownsDevice}`);

const x = Float32Array.from({ length: N }, (_, i) => i + 1);
const { Y } = await session.run({ X: new ort.Tensor('float32', x, [1, N]) });
check('O1: the model ran on the WebGPU EP and its output stayed on the GPU',
    Y.location === 'gpu-buffer' && Y.gpuBuffer && typeof Y.gpuBuffer.size === 'number',
    `location=${Y.location} gpuBuffer=${Y.gpuBuffer ? Y.gpuBuffer.size + ' bytes' : 'none'}`);

// NullGraph consumes the tensor's buffer directly — the binding IS Y.gpuBuffer.
const out = ortDevice.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const consume = engine.createComputeStage({
    name: 'consume model output', dispatchCount: [1, 1, 1],
    shaderCode: /* wgsl */ `
        @group(0) @binding(0) var<storage, read> y: array<f32>;
        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
        @compute @workgroup_size(${N}) fn cs_main(@builtin(global_invocation_id) g: vec3<u32>) {
            o[g.x] = y[g.x] * 10.0 + 0.5;
        }`,
});
engine.setComputePassBindGroup(consume, [
    { binding: 0, resource: { buffer: Y.gpuBuffer } },
    { binding: 1, resource: { buffer: out } },
], 0);
engine.render([consume]);

// Lifetime: the tensor must outlive the GPU work that reads it — fence, THEN dispose.
await ortDevice.queue.onSubmittedWorkDone();

// Test-only readback (forbidden in a frame loop, legal here).
const read = ortDevice.createBuffer({ size: N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
const enc = ortDevice.createCommandEncoder();
enc.copyBufferToBuffer(out, 0, read, 0, N * 4);
ortDevice.queue.submit([enc.finish()]);
await read.mapAsync(GPUMapMode.READ);
const got = Array.from(new Float32Array(read.getMappedRange().slice(0)));
read.unmap(); read.destroy();

const want = Array.from(x, v => (v + v) * 10 + 0.5);
check('O3: the model output, bound with zero copies, drives a NullGraph compute stage',
    got.every((v, i) => v === want[i]),
    `got=[${got.join(', ')}] want=[${want.join(', ')}]`);

// A27 runs before the first session is released. Run after it, the read-back of the observations
// aborted (AbortError): the device ONNX Runtime made does not outlive its sessions.
// Each committed network becomes an .onnx file in memory, and ONNX Runtime runs it on 256
// observations per species that the simulation itself produced: 120 steps of Herd & Wolves under
// the rule, on this same device. Its actions must equal mlpForward's within 1e-4 (the parent's A5).
{
    const TOL = 1e-4, ROWS = 256;
    const readJson = async path => {
        const url = new URL(path, import.meta.url);
        return inBrowser ? (await fetch(url)).json() : JSON.parse((await import('node:fs')).readFileSync(url, 'utf8'));
    };
    const herd = createHerd(engine), { capacity, obsDim, actDim } = herd.definition;
    const step = lockstep(engine, herd.agents);
    for (let s = 0; s < 120; s++) await step();
    const [words] = await readBuffers(ortDevice, [herd.agents.obs]);
    const obs = new Float32Array(words.buffer, words.byteOffset, words.length);
    herd.destroy();
    for (const [index, species] of herd.definition.species.map(s => s.name).entries()) {
        const file = `../../tools/agents/models/herd-wolves/${species}.mlp.json`, model = await readJson(file);
        const network = networkFromModel(model, herdDefinition(), file);
        const bytes = onnxFromNetwork(network, { name: `${species}-mlp`, metadata: { species } });
        // This species' first 256 agents, at a spread of indices so that several think blocks are in.
        const mine = Array.from({ length: capacity }, (_, i) => i).filter(i => herd.speciesOf(i) === index);
        const picked = Array.from({ length: ROWS }, (_, k) => mine[Math.floor(k * mine.length / ROWS)]);
        const input = new Float32Array(ROWS * obsDim);
        picked.forEach((i, k) => input.set(obs.subarray(i * obsDim, (i + 1) * obsDim), k * obsDim));
        let worst = 0, bad = 0, moved = 0, note = '';
        try {
            const run = await ort.InferenceSession.create(bytes, { executionProviders: ['webgpu'] });
            const { act } = await run.run({ obs: new ort.Tensor('float32', input, [ROWS, obsDim]) });
            const got = await act.getData(), cpu = new Float32Array(actDim);
            note = `output [${act.dims.join(', ')}]; `;
            for (let k = 0; k < ROWS; k++) {
                mlpForward(network, input.subarray(k * obsDim, (k + 1) * obsDim), cpu);
                for (let j = 0; j < actDim; j++) {
                    const d = Math.abs(got[k * actDim + j] - cpu[j]);
                    worst = Math.max(worst, d);
                    if (!(d <= TOL)) bad++;
                }
                if (Math.abs(cpu[0]) > 0.01) moved++;
            }
            act.dispose();
            await run.release();
        } catch (e) {
            bad = -1;
            note = `ONNX Runtime failed: ${e.message.split('\n')[0]}; `;
        }
        check(`A27: ONNX Runtime runs ${species}.onnx on ${ROWS} observations and its actions equal mlpForward's within ${TOL}`,
            bad === 0 && moved > ROWS / 4, `${note}${bad} of ${ROWS * actDim} actions out of tolerance; worst difference ${worst.toExponential(1)}; ` +
            `${moved} rows turn by more than 0.01; ${bytes.length.toLocaleString('en-US')} B`);
    }
}

Y.dispose();
out.destroy();
await session.release();

await finish('All ONNX seam GPU tests passed.');
