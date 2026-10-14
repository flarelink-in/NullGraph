// Spikes S1, S2, S4 and S5 (docs/agents-design.md §13, C0): what ONNX Runtime Web costs as the
// agents' think step. Measurements, not test suites: they are not registered in test/gpu/suites.mjs.
// This file holds what the four share: loading ONNX Runtime on either host, the networks, and the
// WGSL kernel that spike S3 measured, as the baseline.
//
// One process or page per run: ONNX Runtime keeps one device for its whole life, and whether it
// made that device or was handed the engine's is the question of S5.
import { mlpForward } from '../../../dist/agents.js';
import { onnxFromNetwork } from '../../../tools/agents/onnx.mjs';

export const inBrowser = typeof document !== 'undefined';
export const LAYERS = [24, 32, 32, 2];             // Herd & Wolves: obsDim 24 → actDim 2, as S3 measured
export const COUNTS = [1_000, 20_000, 100_000];
const ORT_DIST = new URL('../../../node_modules/onnxruntime-web/dist/', import.meta.url);

/** A setting: globalThis.NG_SPIKE[name] (set by a wrapper script), ?name=value in a browser, NG_SPIKE_NAME under Node. */
export function setting(name, fallback) {
    const value = globalThis.NG_SPIKE?.[name] ?? (inBrowser ? new URLSearchParams(location.search).get(name)
        : globalThis.process?.env?.[`NG_SPIKE_${name.toUpperCase()}`]);
    return value ?? fallback;
}

export const median = values => { const s = [...values].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
export const fmt = n => n.toLocaleString('en-US');
export const ms = n => (n < 10 ? n.toFixed(2) : n.toFixed(1));

/**
 * ONNX Runtime Web, loaded as test/onnx-gpu/run.js loads it. `timings` says how long each part took
 * (spike S4). Nothing is initialised yet: the WASM is compiled by the first session.
 *
 * `bundle` is 'webgpu' (the default: the WebGPU provider compiled into the WASM, which is what this
 * repository tests against) or 'all' (the older provider written in JavaScript, "JSEP").
 */
export async function loadOrt({ bundle = 'webgpu' } = {}) {
    const [script, build] = bundle === 'all' ? ['ort.all.bundle.min.mjs', 'jsep'] : ['ort.webgpu.bundle.min.mjs', 'asyncify'];
    if (!globalThis.self) globalThis.self = globalThis;        // the bundle expects a worker-style global
    const timings = {};
    let t = performance.now();
    const ort = await import(new URL(script, ORT_DIST).href);
    timings.importMs = performance.now() - t;
    ort.env.wasm.numThreads = 1;                               // no SharedArrayBuffer, no cross-origin isolation
    if (inBrowser) {
        ort.env.wasm.wasmPaths = ORT_DIST.href;
    } else {
        // Node cannot fetch file: URLs, so ONNX Runtime gets the module's URL and the WASM bytes.
        const { readFile, stat } = await import('node:fs/promises');
        const { fileURLToPath } = await import('node:url');
        const mjs = new URL(`ort-wasm-simd-threaded.${build}.mjs`, ORT_DIST), wasm = new URL(`ort-wasm-simd-threaded.${build}.wasm`, ORT_DIST);
        t = performance.now();
        ort.env.wasm.wasmPaths = { mjs: mjs.href, wasm: wasm.href };
        ort.env.wasm.wasmBinary = new Uint8Array(await readFile(fileURLToPath(wasm)));
        timings.readWasmMs = performance.now() - t;
        timings.wasmBytes = ort.env.wasm.wasmBinary.length;
        timings.bundleBytes = (await stat(fileURLToPath(new URL(script, ORT_DIST)))).size;
    }
    return { ort, timings };
}

let seed = 0x5eed;
export const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };

/** The network S3 measured: 24→32→32→2, tanh, Xavier-uniform weights and small biases. */
export function tanhNetwork() {
    const network = [];
    for (let l = 1; l < LAYERS.length; l++) {
        const input = LAYERS[l - 1], output = LAYERS[l], a = Math.sqrt(6 / (input + output));
        network.push({ input, output, activation: 'tanh',
            weights: Float32Array.from({ length: input * output }, () => (random() * 2 - 1) * a),
            biases: Float32Array.from({ length: output }, () => (random() * 2 - 1) * 0.1) });
    }
    return network;
}

/**
 * A network of the same shape and cost that carries a marker through: its first output is its
 * first input, for any input that is not negative (relu, relu, identity; one weight of 1 per layer).
 * Spike S2 uses it to see which frame's observations a run read.
 */
export function markerNetwork() {
    return LAYERS.slice(1).map((output, l) => {
        const input = LAYERS[l], weights = new Float32Array(input * output);
        weights[0] = 1;
        return { input, output, activation: l === LAYERS.length - 2 ? 'identity' : 'relu', weights, biases: new Float32Array(output) };
    });
}

export const onnxOf = network => onnxFromNetwork(network, { name: 'spike' });
export { mlpForward };

/**
 * The kernel S3 chose and MlpBrain ships in kind: weights in a storage buffer as vec4s, loops, one
 * invocation per agent, workgroup 64. Returns the WGSL and the packed weights.
 */
export function wgslKernel(network) {
    const layout = [];
    let quads = 0;
    for (const layer of network) {
        const in4 = Math.ceil(layer.input / 4), out4 = Math.ceil(layer.output / 4);
        layout.push({ layer, in4, out4, w: quads, bias: quads + out4 * 4 * in4 });
        quads += out4 * 4 * in4 + out4;
    }
    const packed = new Float32Array(quads * 4);
    for (const L of layout) {
        for (let r = 0; r < L.layer.output; r++) {
            for (let k = 0; k < L.layer.input; k++) packed[(L.w + r * L.in4) * 4 + k] = L.layer.weights[r * L.layer.input + k];
            packed[L.bias * 4 + r] = L.layer.biases[r];
        }
    }
    const ACT = { tanh: v => `tanh(clamp(${v}, vec4<f32>(-15.0), vec4<f32>(15.0)))`, relu: v => `max(${v}, vec4<f32>(0.0))`, identity: v => v };
    const first = layout[0], obsDim = network[0].input, actDim = network[network.length - 1].output;
    let body = `    var x0: array<vec4<f32>, ${first.in4}>;
    for (var k = 0u; k < ${first.in4}u; k++) {
        let b = i * ${obsDim}u + k * 4u;
        x0[k] = vec4<f32>(obs[b], obs[b + 1u], obs[b + 2u], obs[b + 3u]);
    }
`;
    layout.forEach((L, l) => {
        body += `    var x${l + 1}: array<vec4<f32>, ${L.out4}>;
    for (var o = 0u; o < ${L.out4}u; o++) {
        var v = weights[${L.bias}u + o];
        for (var r = 0u; r < 4u; r++) {
            let row = ${L.w}u + (o * 4u + r) * ${L.in4}u;
            var t = 0.0;
            for (var k = 0u; k < ${L.in4}u; k++) { t += dot(weights[row + k], x${l}[k]); }
            v[r] += t;
        }
        x${l + 1}[o] = ${ACT[L.layer.activation]('v')};
    }
`;
    });
    const y = `x${layout.length}[0]`;
    const code = /* wgsl */ `
@group(0) @binding(0) var<storage, read> obs: array<f32>;
@group(0) @binding(1) var<storage, read_write> act: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<vec4<f32>>;
@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&act) / ${actDim}u) { return; }
${body}${Array.from({ length: actDim }, (_, c) => `    act[i * ${actDim}u + ${c}u] = ${y}[${c}];`).join('\n')}
}`;
    return { code, packed };
}

/** Copies a buffer's first `floats` f32 out. Off the frame path: spikes may read back. */
export async function readF32(device, buffer, floats) {
    const staging = device.createBuffer({ size: floats * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, floats * 4);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
}

/** "vendor architecture (description)" of the adapter a device came from. */
export function describe(device) {
    const i = device?.adapterInfo ?? {};
    return [i.vendor, i.architecture].filter(Boolean).join(' ') + (i.description ? ` (${i.description})` : '') || 'unknown adapter';
}
