// The runtime seam for GPU harnesses: one test file runs as a browser page AND under Node/Dawn.
//
// A harness asks this module for its canvas, its logger and its check() — the only three things
// that differed between the two hosts. In a browser the canvas is #gpu, the log is mirrored into
// #log, and results land on window[resultsKey] (what headless.mjs and a human read). Under Node
// the canvas is null (NullGraph renders into engine.screenTexture), logging is the console, and
// results land on globalThis[resultsKey] for test/gpu/run-harness.mjs.
//
// Every assertion stays numeric and stays in the harness; this file only decides where it goes.
//
// It also WATCHES every device the harness creates — the engine's, an adopted one, ONNX
// Runtime's — by wrapping navigator.gpu.requestAdapter → adapter.requestDevice before the harness
// asks for one. Uncaptured validation errors and device loss are published as `gpuErrors` beside
// the check results, on both hosts. That matters most in a browser: Chrome reports them outside the
// console API, so a runner listening to the console sees a red check with no cause — or, worse, a
// green check computed over a frame the device silently dropped.

const inBrowser = typeof document !== 'undefined';

const watched = [];
const gpuErrors = [];

function watchDevice(device) {
    if (watched.includes(device)) return device;
    watched.push(device);
    device.addEventListener('uncapturederror', (ev) => {
        gpuErrors.push((ev.error?.message ?? String(ev.error)).trim());
    });
    device.lost.then((info) => {
        if (info.reason !== 'destroyed') gpuErrors.push(`device lost (${info.reason}): ${info.message}`);
    });
    return device;
}

function installDeviceWatch() {
    const gpu = globalThis.navigator?.gpu;
    if (!gpu || gpu.__harnessWatched) return;
    const requestAdapter = gpu.requestAdapter.bind(gpu);
    gpu.requestAdapter = async (options) => {
        const adapter = await requestAdapter(options);
        if (!adapter) return adapter;
        const requestDevice = adapter.requestDevice.bind(adapter);
        adapter.requestDevice = async (descriptor) => watchDevice(await requestDevice(descriptor));
        return adapter;
    };
    gpu.__harnessWatched = true;
}

/**
 * @param {string} resultsKey Global the summary is published on, e.g. '__SHADOW_RESULTS__'.
 */
export function harnessEnv(resultsKey) {
    installDeviceWatch();
    const logEl = inBrowser ? document.getElementById('log') : null;
    const results = [];

    const log = (msg) => {
        if (logEl) logEl.textContent += `\n${msg}`;
        console.log(msg);
    };

    function check(name, cond, detail = '') {
        results.push({ name, pass: !!cond, detail });
        log(`  ${cond ? 'ok  ' : 'FAIL'}  ${name}${detail ? ' — ' + detail : ''}`);
    }

    return {
        inBrowser,
        /** The canvas to init against — `null` under Node, which selects headless rendering. */
        canvas: inBrowser ? document.getElementById('gpu') : null,
        log,
        check,
        results,

        /** Publish an environment failure (no WebGPU) — distinct from a failing check. */
        fatal(message) {
            log('FATAL: WebGPU init failed — ' + message);
            globalThis[resultsKey] = { fatal: message };
        },

        /**
         * Save an RGBA8 frame as `<NG_GPU_ARTIFACTS>/<name>.png` — Node only, and only when that
         * variable is set; otherwise a no-op. For LOOKING at what a check measured: a person, or
         * an agent that can read images, sees in one glance what a failing number cannot say.
         */
        async saveImage(name, rgba, width, height) {
            const dir = !inBrowser && globalThis.process?.env?.NG_GPU_ARTIFACTS;
            if (!dir) return;
            const [{ mkdirSync, writeFileSync }, { join }, { encodePNG }] = await Promise.all([
                import('node:fs'), import('node:path'), import('./png.mjs'),
            ]);
            mkdirSync(dir, { recursive: true });
            const file = join(dir, `${name.replace(/[^a-z0-9._-]+/gi, '-')}.png`);
            writeFileSync(file, encodePNG(rgba, width, height));
            log(`        (saved ${file})`);
        },

        /**
         * Let queued GPU work finish so late error events arrive, then print the summary line and
         * publish the machine-readable result — including `gpuErrors`, which fail the suite on
         * either host even when every check passed. Await it: the result is published after.
         */
        async finish(passLine) {
            await Promise.all(watched.map(d => d.queue.onSubmittedWorkDone().catch(() => {})));
            await new Promise(r => setTimeout(r, 50));
            const failed = results.filter(r => !r.pass);
            if (gpuErrors.length) {
                log(`\n${gpuErrors.length} uncaptured WebGPU error(s):`);
                for (const e of gpuErrors.slice(0, 5)) log(`  ${e.split('\n')[0]}`);
            }
            const clean = failed.length === 0 && gpuErrors.length === 0;
            log(clean ? `\n${passLine}` : `\n${failed.length} GPU test(s) FAILED, ${gpuErrors.length} WebGPU error(s).`);
            globalThis[resultsKey] = { total: results.length, failed: failed.length, results, gpuErrors: [...gpuErrors] };
            if (inBrowser) document.title = clean ? 'PASS' : 'FAIL';
        },
    };
}
