// Same Node bootstrap as test/inference-gpu/run.js; ORT owns the device the engine adopts.
import { readFileSync } from 'node:fs';
import { OrtRuntime } from '../../dist/inference.js';
export async function filmRuntime() {
    globalThis.self ??= globalThis;
    const base = new URL('../../node_modules/onnxruntime-web/dist/', import.meta.url);
    const ort = await import(new URL('ort.webgpu.bundle.min.mjs', base).href);
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = { mjs: new URL('ort-wasm-simd-threaded.asyncify.mjs', base).href, wasm: new URL('ort-wasm-simd-threaded.asyncify.wasm', base).href };
    ort.env.wasm.wasmBinary = new Uint8Array(readFileSync(new URL('ort-wasm-simd-threaded.asyncify.wasm', base)));
    return OrtRuntime.create(ort, { adapter: await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }) });
}