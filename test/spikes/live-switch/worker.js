// Spikes A and B of step 10's phase 10.0 (docs/agents-live-switch-design.md §4, §6): the worker.
//
// INIT makes the engine as the site's EngineCore.init makes it (its own device, timestamp-query,
// high-performance, samplable depth), with a GPUProfiler, and draws a ground grid every frame.
// SWITCH is decision D1's device switch (§4): ONNX Runtime imported on demand, its runtime
// started, the scene destroyed, the engine made again on ONNX Runtime's device over the same
// OffscreenCanvas, the profiler made again, the old device destroyed, the scene made again (a
// coarser grid, so that a screenshot tells the two engines apart). Then ONNX Runtime runs the
// trained prey network over a buffer the new engine made, and the canvas is resized.
// Every step is logged with "[spike]"; run.mjs reads the log.
import { Camera, NullGraph } from 'null-graph';
import { mlpForward } from 'null-graph/agents';
import { GPUProfiler } from 'null-graph/profiler';
import { groundGrid } from 'null-graph/debug-ui';
import { onnxFromNetwork } from '../../../tools/agents/onnx.mjs';
import preyFile from '../../../tools/agents/models/herd-wolves/prey.mlp.json';

const say = (text) => console.log(`[spike] ${text}`);
const now = () => performance.now();
const ms = (t) => Math.round(t * 10) / 10;
const median = (v) => [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)];
const describe = (device) => {
    const i = device.adapterInfo ?? {};
    return [i.vendor, i.architecture].filter(Boolean).join(' ') || 'unknown adapter';
};

let canvas, settings, engine, camera, pass, grid, profiler;
let drawing = false;
let frames = 0;
let waiting = null;
const gpuMs = [];

/** The demo's part: a pass, a ground grid, a camera. */
function scene(spacing) {
    pass = engine.createPass({ name: 'spike', isMainScreenPass: true });
    grid = groundGrid(engine, pass, { min: [-40, -40], max: [40, 40], minorSpacing: spacing,
        majorSpacing: spacing * 5, fadeStart: 80, fadeEnd: 160 });
    camera = new Camera(60, canvas.width / canvas.height, 0.1, 500);
    camera.updateView([34, 26, 34], [0, 0, 0]);
    drawing = true;
}

/** The site's RenderLoop and EngineCore.render, reduced to what the spike needs. */
function frame() {
    if (drawing) {
        profiler?.begin();
        engine.updateCamera(camera);
        engine.render([pass]);
        profiler?.end((gpu) => gpuMs.push(gpu));
        frames++;
        if (waiting && frames >= waiting.until) {
            const done = waiting.done;
            waiting = null;
            done();
        }
    }
    requestAnimationFrame(frame);
}
const framesFrom = (n) => new Promise((done) => { waiting = { until: frames + n, done }; });

async function init(msg) {
    ({ canvas, settings } = msg);
    canvas.width = msg.width;
    canvas.height = msg.height;
    engine = new NullGraph();
    const t0 = now();
    const result = await engine.init(canvas, { desiredFeatures: ['timestamp-query'], width: msg.width,
        powerPreference: 'high-performance', height: msg.height, samplableDepth: true });
    if (!result.success) throw new Error(`engine A: ${result.error}`);
    say(`engine A on ${describe(engine.device)}, its own device: ${ms(now() - t0)} ms`);
    profiler = engine.device.features.has('timestamp-query') ? new GPUProfiler(engine.device) : null;
    scene(2);
    requestAnimationFrame(frame);
    await framesFrom(60);
    say(`A rendered 60 frames; GPU ${gpuMs.length ? ms(median(gpuMs)) : '—'} ms per frame`);
}

async function deviceSwitch() {
    const t = {};
    let t0 = now();
    const all = now();
    const ort = await import('onnxruntime-web/webgpu');
    t.importOrt = now() - t0;
    ort.env.wasm.numThreads = 1;
    if (settings.wasm === 'prefix') ort.env.wasm.wasmPaths = '/vendor/onnxruntime-web/';
    else if (settings.wasm === 'wasm') ort.env.wasm.wasmPaths = { wasm: '/vendor/onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm' };
    t0 = now();
    const { OrtRuntime } = await import('null-graph/inference');
    t.importInference = now() - t0;
    t0 = now();
    const runtime = await OrtRuntime.create(ort);
    t.runtime = now() - t0;
    say(`ONNX Runtime ${ort.env.versions?.web ?? ''} started on ${describe(runtime.device)}: import ${ms(t.importOrt)} ms, ` +
        `null-graph/inference ${ms(t.importInference)} ms, OrtRuntime.create ${ms(t.runtime)} ms`);

    // §4, step 3: the demo destroyed, after its frames stop (cleanupDemo).
    drawing = false;
    grid.destroy();
    engine.clearPasses();
    const old = engine;
    // GPUProfiler.end's read has no catch: a device destroyed while it maps rejects uncaught. With
    // ?settle=1, the old device's queue is waited for first, so that the last read has landed.
    if (settings.settle === '1') {
        t0 = now();
        await old.device.queue.onSubmittedWorkDone();
        await new Promise((done) => setTimeout(done, 0));
        t.settle = now() - t0;
    }
    // Step 4: the engine again, on ONNX Runtime's device, over the same canvas. No powerPreference.
    t0 = now();
    engine = new NullGraph();
    const result = await engine.init(canvas, { device: runtime.device, desiredFeatures: ['timestamp-query'],
        width: canvas.width, height: canvas.height, samplableDepth: true });
    if (!result.success) throw new Error(`engine B: ${result.error}`);
    t.engine = now() - t0;
    // Step 5: the profiler again; the old device destroyed.
    t0 = now();
    profiler = new GPUProfiler(engine.device);
    old.device.destroy();
    t.destroyOld = now() - t0;
    // Step 6: the demo again.
    gpuMs.length = 0;
    scene(8);
    t.total = now() - all;
    say(`switched: ${Object.entries(t).map(([k, v]) => `${k} ${ms(v)} ms`).join(', ')}; the old device ` +
        `${old.device === engine.device ? 'is the new one' : 'is another'}`);
    await framesFrom(60);
    say(`B rendered 60 frames; GPU ${gpuMs.length ? ms(median(gpuMs)) : '—'} ms per frame, timed on ONNX Runtime's device`);

    // ONNX Runtime over a buffer the engine made: the prey network at 4,096 agents.
    const network = preyFile.layers.map((l) => ({ input: l.input, output: l.output, activation: l.activation,
        weights: Float32Array.from(l.weights), biases: Float32Array.from(l.biases) }));
    const n = 4096, obsDim = 24, actDim = 2, device = engine.device;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const obsData = Float32Array.from({ length: n * obsDim }, (_, k) => Math.sin(k * 0.37));
    const obs = device.createBuffer({ size: n * obsDim * 4, usage });
    const act = device.createBuffer({ size: n * actDim * 4, usage });
    device.queue.writeBuffer(obs, 0, obsData);
    t0 = now();
    const model = await runtime.load(onnxFromNetwork(network, { name: 'prey' }));
    model.bind({ inputs: { obs: { buffer: obs, dims: [n, obsDim] } }, outputs: { act: { buffer: act, dims: [n, actDim] } } });
    const loadMs = now() - t0;
    t0 = now();
    await model.run();
    await device.queue.onSubmittedWorkDone();
    const firstMs = now() - t0;
    const runs = [];
    for (let k = 0; k < 60; k++) {
        t0 = now();
        await model.run();
        runs.push(now() - t0);
    }
    const staging = device.createBuffer({ size: 64 * actDim * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(act, 0, staging, 0, 64 * actDim * 4);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const got = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    let worst = 0;
    for (let i = 0; i < 64; i++) {
        const want = mlpForward(network, obsData.subarray(i * obsDim, (i + 1) * obsDim));
        for (let c = 0; c < actDim; c++) worst = Math.max(worst, Math.abs(want[c] - got[i * actDim + c]));
    }
    say(`ONNX Runtime ran the prey network over the engine's buffer: load ${ms(loadMs)} ms, first run ${ms(firstMs)} ms, ` +
        `run ${ms(median(runs))} ms on the worker's thread; worst difference from mlpForward ${worst.toExponential(1)}` +
        (worst <= 1e-4 ? '' : ' — WRONG'));
    await model.destroy();
    [obs, act, staging].forEach((b) => b.destroy());

    // The page resized, as MessageRouter.handleResize does it.
    canvas.width = 640;
    canvas.height = 360;
    engine.resize(640, 360);
    camera.updateProjection(640 / 360);
    await framesFrom(20);
    const resources = performance.getEntriesByType('resource').map((e) => ({ name: new URL(e.name).pathname.replace(/^\/@fs\//, '').slice(-90),
        type: e.initiatorType, transferred: e.transferSize, body: e.encodedBodySize }));
    say(`resources the worker fetched: ${JSON.stringify(resources)}`);
    say(`done ${JSON.stringify({ switch: t, loadMs, firstMs, runMs: median(runs), worst })}`);
}

self.onmessage = (event) => {
    const msg = event.data;
    const task = msg.type === 'INIT' ? init(msg) : msg.type === 'SWITCH' ? deviceSwitch() : null;
    task?.catch((error) => console.error(`[spike] ${msg.type} failed: ${error?.stack ?? error}`));
};
