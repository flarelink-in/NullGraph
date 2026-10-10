// Run WebGPU in Node through Dawn (the `webgpu` devDependency) — the same engine code, no browser.
//
// installNodeGPU() makes Node look like a WebGPU host: the GPU* constant and error globals
// (GPUBufferUsage, GPUTextureUsage, GPUMapMode, GPUValidationError, …) plus navigator.gpu.
//
// Every device created through it is WATCHED from birth: uncaptured validation, out-of-memory and
// internal errors, and device loss, are recorded in `events` instead of only being printed. That
// is the half a browser console never hands to a test runner or an agent — a shader that fails
// to compile renders black and exits 0 unless something is listening.
//
// Nothing here ships: the npm package is `dist` only, and `webgpu` is a devDependency.

import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);

/**
 * @param {{ flags?: string[] }} [options] Dawn flags, e.g. `['backend=d3d12']`. `NG_DAWN_FLAGS`
 *   (comma-separated) is appended, so CI can pin a backend without touching code.
 * @returns {{ gpu: GPU, events: {kind: string, message: string}[], devices: GPUDevice[],
 *   watch(device: GPUDevice): GPUDevice, settle(): Promise<void> }}
 */
export function installNodeGPU({ flags = [] } = {}) {
    let dawn;
    try {
        dawn = require('webgpu');
    } catch (e) {
        throw new Error(
            `Dawn (the 'webgpu' devDependency) failed to load: ${e.message}\n` +
            `Run \`npm install\` in the repo root. Dawn ships prebuilt binaries for win32-x64, ` +
            `darwin and linux-x64; other platforms need a browser run instead (test/*-gpu/index.html).`);
    }

    const { create, globals } = dawn;
    Object.assign(globalThis, globals);

    const envFlags = (process.env.NG_DAWN_FLAGS ?? '').split(',').map(s => s.trim()).filter(Boolean);
    const gpu = create([...flags, ...envFlags]);

    const events = [];
    const devices = [];

    /** Attach error + loss listeners. Idempotent per device. */
    function watch(device) {
        if (devices.includes(device)) return device;
        devices.push(device);
        device.addEventListener('uncapturederror', (ev) => {
            const err = ev.error;
            events.push({ kind: errorKind(err), message: (err?.message ?? String(err)).trim() });
        });
        device.lost.then((info) => {
            // A deliberate destroy() at teardown is not a failure; anything else is.
            if (info.reason !== 'destroyed') {
                events.push({ kind: 'lost', message: `device lost (${info.reason}): ${info.message}` });
            }
        });
        return device;
    }

    // Watch devices the ENGINE creates (NullGraph.init requests its own), not only ones we make.
    const requestAdapter = gpu.requestAdapter.bind(gpu);
    gpu.requestAdapter = async (options) => {
        const adapter = await requestAdapter(options);
        if (!adapter) return adapter;
        const requestDevice = adapter.requestDevice.bind(adapter);
        adapter.requestDevice = async (descriptor) => watch(await requestDevice(descriptor));
        return adapter;
    };

    // Node 21+ has a navigator object (extensible); older Node has none.
    if (globalThis.navigator && Object.isExtensible(globalThis.navigator)) {
        Object.defineProperty(globalThis.navigator, 'gpu', { configurable: true, value: gpu });
    } else {
        Object.defineProperty(globalThis, 'navigator', { configurable: true, writable: true, value: { gpu } });
    }

    /**
     * Let queued GPU work finish and its error events arrive. Uncaptured errors are delivered
     * asynchronously, so reading `events` right after the last submit can miss the one that matters.
     */
    async function settle() {
        await Promise.all(devices.map(d => d.queue.onSubmittedWorkDone().catch(() => {})));
        await new Promise(r => setTimeout(r, 50));
    }

    return { gpu, events, devices, watch, settle };
}

function errorKind(err) {
    if (globalThis.GPUValidationError && err instanceof globalThis.GPUValidationError) return 'validation';
    if (globalThis.GPUOutOfMemoryError && err instanceof globalThis.GPUOutOfMemoryError) return 'out-of-memory';
    if (globalThis.GPUInternalError && err instanceof globalThis.GPUInternalError) return 'internal';
    return 'error';
}

/** One-line description of the adapter a device came from, for run headers. */
export function describeDevice(device) {
    const i = device?.adapterInfo ?? {};
    return [i.vendor, i.architecture, i.description].filter(Boolean).join(' ') || 'unknown adapter';
}
