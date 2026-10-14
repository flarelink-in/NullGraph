// Step 10, phase 10.0 (docs/agents-live-switch-design.md §8 and §9, decision D6): ONNX Runtime at
// the Herd & Wolves demo's agent counts, 4,096 to 1,048,576, running the demo's own networks, the
// two trained 24→64→64→64→2 tanh networks of tools/agents/models/herd-wolves, each over every agent,
// as OrtBrain binds it. Measurements, not a suite.
//
//   Node + Dawn:   npm run spike:ort-counts          NG_DAWN_FLAGS=adapter=AMD or adapter=NVIDIA
//   Chrome:        npm run drive -- --serve . "/test/spikes/ort/index.html?spike=counts" --wait-for "counts done" --timeout 1200
//   NG_SPIKE_MAX (?max=) stops at a count. NG_SPIKE_LIMITMS (?limitms=, default 600, the demo's step
//   limit): a count whose run is predicted to take longer is not run. Windows resets a GPU that
//   stays busy for about 2 s.
//
// At each count, two models (prey, wolf), loaded through OrtRuntime as the demo's ORT choice for
// both species will load them, and bound to one observations buffer and a results buffer each:
//   create   OrtRuntime.load, per model
//   first    its first run, until the GPU is done: ONNX Runtime compiles its pipelines then
//   run      the main-thread time of `await model.run()`, median
//   done     from run() to onSubmittedWorkDone, median
//   GPU      spike S1's bracket: K runs queued behind a long WGSL pass, between two marker passes
//   memory   GPU bytes ONNX Runtime asked for on its device while the models were created, in their
//            first runs and in the runs after, and the bytes it destroyed; ONNX Runtime's WASM memory
//   marks    the overlay's way (§8): run() started, a frame's work of d ms submitted, a mark; another
//            mark when run() resolves. The span between the two marks, against the bracket, for d of
//            0, 2 and 8 ms.
// The runs' results are checked against mlpForward on their first rows.
import { OrtRuntime } from '../../../dist/inference.js';
import { harnessEnv } from '../../gpu/harness-env.mjs';
import { describe, fmt, inBrowser, loadOrt, median, mlpForward, ms, onnxOf, random, readF32, setting, tanhNetwork, wgslKernel } from './common.mjs';

const { log, check, finish } = harnessEnv('__ORT_SPIKE_RESULTS__');
const name = 'counts';
const COUNTS = [4_096, 16_384, 65_536, 262_144, 1_048_576];          // the demo's HERD_COUNTS
const MAX = Number(setting('max', COUNTS[COUNTS.length - 1]));
const LIMIT_MS = Number(setting('limitms', 600));
const SPECIES = ['prey', 'wolf'];
const OBS = 24, ACT = 2, CHECK_ROWS = 256, TOLERANCE = 1e-4, TRIALS = 9;
const FRAME_WORK_MS = [0, 2, 8];
const MB = bytes => (bytes / 1048576).toFixed(1);
const round16 = bytes => Math.ceil(bytes / 16) * 16;

// ONNX Runtime's WASM memory, which it neither exports nor names: found among what it passes to
// WebAssembly.instantiate, as test/spikes/ort/issues.js finds it.
let wasmMemory;
for (const fn of ['instantiate', 'instantiateStreaming']) {
    const original = WebAssembly[fn];
    WebAssembly[fn] = async (...args) => {
        const result = await original.apply(WebAssembly, args);
        const found = [...Object.values((result.instance ?? result).exports),
            ...Object.values(args[1] ?? {}).flatMap(space => (space && typeof space === 'object' ? Object.values(space) : []))];
        wasmMemory ??= found.find(v => v instanceof WebAssembly.Memory);
        return result;
    };
}

const { ort } = await loadOrt();
const runtime = await OrtRuntime.create(ort, { adapter: await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }) });
const device = runtime.device, L = device.limits;
const host = inBrowser ? 'Chrome' : 'Node/Dawn';
log(`${name}: ${host} · ${describe(device)} · ONNX Runtime Web ${ort.env.versions?.web ?? ''}`);
log(`   its device: maxBufferSize ${fmt(L.maxBufferSize)}, maxStorageBufferBindingSize ${fmt(L.maxStorageBufferBindingSize)}; ` +
    `timestamp-query ${device.features.has('timestamp-query') ? 'yes' : 'no'}`);

// Every buffer created on the device from here on is ONNX Runtime's, unless this spike made it
// through `own` (or `ours`, `readOurs`). ONNX Runtime's runtime anchor, made before, is left out.
const account = { created: 0, destroyed: 0, buffers: 0 };
let mine = false;
const createBuffer = device.createBuffer.bind(device);
device.createBuffer = descriptor => {
    const buffer = createBuffer(descriptor);
    if (!mine) {
        account.created += descriptor.size;
        account.buffers++;
        const destroy = buffer.destroy.bind(buffer);
        let gone = false;
        buffer.destroy = () => {
            if (!gone) { gone = true; account.destroyed += descriptor.size; }
            destroy();
        };
    }
    return buffer;
};
/** A buffer of this spike's own, left out of ONNX Runtime's account. */
function own(descriptor) {
    mine = true;
    try {
        return device.createBuffer(descriptor);
    } finally {
        mine = false;
    }
}
const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
function ours(bytes, data) {
    const buffer = own({ size: round16(bytes), usage: STORAGE });
    if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
}
const snapshot = () => ({ ...account, wasm: wasmMemory?.buffer.byteLength ?? NaN });
const since = before => ({ created: account.created - before.created, destroyed: account.destroyed - before.destroyed,
    buffers: account.buffers - before.buffers });

// The demo's networks, from the committed files.
async function network(species) {
    const url = new URL(`../../../tools/agents/models/herd-wolves/${species}.mlp.json`, import.meta.url);
    const text = inBrowser ? await (await fetch(url)).text() : await (await import('node:fs/promises')).readFile(url, 'utf8');
    return JSON.parse(text).layers.map(l => ({ input: l.input, output: l.output, activation: l.activation,
        weights: Float32Array.from(l.weights), biases: Float32Array.from(l.biases) }));
}
const networks = {}, bytes = {};
for (const species of SPECIES) {
    networks[species] = await network(species);
    bytes[species] = onnxOf(networks[species]);
}
log(`   networks: ${networks.prey.map(l => l.input).concat(ACT).join('→')}, ${networks.prey.map(l => l.activation).join(', ')}; ` +
    `${fmt(bytes.prey.length)} B of ONNX each`);

// ---------------------------------------------------------------------------------------------
// The timestamp machinery: spike S1's blocker, and the overlay's marks
// ---------------------------------------------------------------------------------------------
const stamp = {
    set: device.createQuerySet({ type: 'timestamp', count: 4 }),
    resolve: own({ size: 32, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
    read: own({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
};
async function stamps(encoder = device.createCommandEncoder()) {
    encoder.resolveQuerySet(stamp.set, 0, 4, stamp.resolve, 0);
    encoder.copyBufferToBuffer(stamp.resolve, 0, stamp.read, 0, 32);
    device.queue.submit([encoder.finish()]);
    await stamp.read.mapAsync(GPUMapMode.READ);
    const t = new BigUint64Array(stamp.read.getMappedRange().slice(0));
    stamp.read.unmap();
    return [...t].map(Number);
}
/** A pass that only writes timestamps `begin` (and `end`, when given) when the GPU reaches it. */
function mark(begin, end) {
    const encoder = device.createCommandEncoder();
    encoder.beginComputePass({ timestampWrites: { querySet: stamp.set, beginningOfPassWriteIndex: begin,
        ...(end === undefined ? {} : { endOfPassWriteIndex: end }) } }).end();
    device.queue.submit([encoder.finish()]);
}

// The blocker: S3's WGSL kernel over 100,000 rows of buffers of its own (Dawn makes a submission
// wait on a buffer that queued work still uses).
const BLOCK_N = 100_000, blockNet = tanhNetwork(), kernel = wgslKernel(blockNet);
const pipeline = await device.createComputePipelineAsync({ layout: 'auto',
    compute: { module: device.createShaderModule({ code: kernel.code }), entryPoint: 'cs_main' } });
const blockObs = ours(BLOCK_N * OBS * 4, Float32Array.from({ length: BLOCK_N * OBS }, () => random() * 2 - 1));
const blockBind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: blockObs } }, { binding: 1, resource: { buffer: ours(BLOCK_N * ACT * 4) } },
    { binding: 2, resource: { buffer: ours(kernel.packed.byteLength, kernel.packed) } }] });
function block(dispatches, timed = false) {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass(timed ? { timestampWrites: { querySet: stamp.set, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, blockBind);
    for (let k = 0; k < dispatches; k++) pass.dispatchWorkgroups(Math.ceil(BLOCK_N / 64));
    pass.end();
    if (timed) return encoder;
    device.queue.submit([encoder.finish()]);
    return null;
}
for (const t0 = performance.now(); performance.now() - t0 < 1000;) {                  // warm the GPU up
    block(20);
    await device.queue.onSubmittedWorkDone();
}
const [b0, b1] = await stamps(block(40, true));
const blockUs = (b1 - b0) / 40 / 1e3;                                                // µs per dispatch
const dispatchesFor = msOfWork => Math.max(1, Math.round((msOfWork * 1e3) / blockUs));
log(`   blocker: ${blockUs.toFixed(0)} µs per dispatch`);

/** µs of GPU time per call of `work`: K calls queued behind blockMs of the blocker, between two marks. */
async function bracket(blockMs, K, work) {
    block(dispatchesFor(blockMs));
    mark(0, 1);
    const t0 = performance.now();
    for (let k = 0; k < K; k++) await work();
    const queuedMs = performance.now() - t0;
    const encoder = device.createCommandEncoder();
    encoder.beginComputePass({ timestampWrites: { querySet: stamp.set, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 } }).end();
    const t = await stamps(encoder);
    return { us: (t[2] - t[1]) / K / 1e3, valid: queuedMs < blockMs * 0.8 };
}

/** The overlay's way: the run started, a frame's work of `frameMs`, a mark; a mark when the run resolves. µs between them. */
async function marks(model, frameMs) {
    const run = model.run();
    if (frameMs > 0) block(dispatchesFor(frameMs));
    mark(0);
    await run;
    mark(1);
    const t = await stamps();
    return (t[1] - t[0]) / 1e3;
}

const worst = (got, want) => want.reduce((w, v, k) => Math.max(w, Math.abs(got[k] - v)), 0);
/** readF32, its staging buffer left out of ONNX Runtime's account (it is created before readF32's first await). */
function readOurs(buffer, floats) {
    mine = true;
    try {
        return readF32(device, buffer, floats);
    } finally {
        mine = false;
    }
}

// ---------------------------------------------------------------------------------------------
// The counts
// ---------------------------------------------------------------------------------------------
const results = { spike: name, host, adapter: describe(device), ort: ort.env.versions?.web ?? null, blockUs, counts: {} };
const obsData = Float32Array.from({ length: Math.min(MAX, COUNTS[COUNTS.length - 1]) * OBS }, () => random() * 2 - 1);
const expected = {};
for (const species of SPECIES) {
    expected[species] = new Float32Array(CHECK_ROWS * ACT);
    for (let i = 0; i < CHECK_ROWS; i++) expected[species].set(mlpForward(networks[species], obsData.subarray(i * OBS, (i + 1) * OBS)), i * ACT);
}
let previous = null;
for (const n of COUNTS) {
    if (n > MAX) break;
    const entry = results.counts[n] = {};
    if (previous) {
        // Cost per agent grows past the GPU's caches; S1 saw up to linear. Predicted with some margin.
        const predictedMs = previous.ms * (n / previous.n) ** 1.1;
        if (predictedMs > LIMIT_MS) {
            entry.skipped = `a run is predicted to take ${ms(predictedMs)} ms, over the ${LIMIT_MS} ms limit`;
            log(`\n${fmt(n)} agents: not run: ${entry.skipped}`);
            break;
        }
    }
    log(`\n${fmt(n)} agents`);
    const obs = ours(n * OBS * 4, obsData.subarray(0, n * OBS));
    const out = Object.fromEntries(SPECIES.map(s => [s, ours(n * ACT * 4)]));
    const models = {};
    try {
        // Created and bound, as the demo's ORT choice for both species will.
        let before = snapshot();
        entry.createMs = {};
        for (const species of SPECIES) {
            const t0 = performance.now();
            models[species] = await runtime.load(bytes[species]);
            models[species].bind({ inputs: { obs: { buffer: obs, dims: [n, OBS] } }, outputs: { act: { buffer: out[species], dims: [n, ACT] } } });
            entry.createMs[species] = performance.now() - t0;
        }
        entry.memoryCreate = since(before);
        before = snapshot();
        entry.firstMs = {};
        for (const species of SPECIES) {
            const t0 = performance.now();
            await models[species].run();
            await device.queue.onSubmittedWorkDone();
            entry.firstMs[species] = performance.now() - t0;
        }
        entry.memoryFirst = since(before);
        entry.error = {};
        for (const species of SPECIES) {
            entry.error[species] = worst(await readOurs(out[species], CHECK_ROWS * ACT), expected[species]);
            check(`${name}: ${species} at ${fmt(n)} agents equals mlpForward on its first ${CHECK_ROWS} rows`,
                entry.error[species] <= TOLERANCE, `worst ${entry.error[species].toExponential(1)}`);
        }

        // Steady runs of one model (prey and wolf cost the same: one shape).
        const model = models.prey;
        before = snapshot();
        let done = [];
        for (let k = 0; k < 3; k++) {
            const t0 = performance.now();
            await model.run();
            await device.queue.onSubmittedWorkDone();
            done.push(performance.now() - t0);
        }
        const estimate = median(done);                                                    // ≥ the GPU time
        const per = Math.max(1, Math.min(20, Math.floor(100 / estimate)));                // runs per timed round
        const runMs = [];
        done = [];
        for (let round = 0; round < TRIALS; round++) {
            let t0 = performance.now();
            for (let r = 0; r < per; r++) await model.run();
            runMs.push((performance.now() - t0) / per);
            await device.queue.onSubmittedWorkDone();
            t0 = performance.now();
            await model.run();
            await device.queue.onSubmittedWorkDone();
            done.push(performance.now() - t0);
        }
        entry.runMs = median(runMs);
        entry.doneMs = median(done);
        // The bracket: K runs behind a blocker long enough to queue them all, the GPU kept under ~0.5 s.
        const K = Math.max(1, Math.min(16, Math.floor(150 / entry.doneMs)));
        const blockMs = Math.min(300, Math.max(20, K * entry.runMs * 4 + 10));
        const gpu = [];
        let invalid = 0;
        for (let round = 0; round < TRIALS; round++) {
            const b = await bracket(blockMs, K, () => model.run());
            if (b.valid) gpu.push(b.us); else invalid++;
        }
        entry.gpuUs = gpu.length ? median(gpu) : null;
        entry.bracket = { K, blockMs, invalid };
        entry.marksUs = {};
        for (const frameMs of FRAME_WORK_MS) {
            const spans = [];
            for (let round = 0; round < TRIALS; round++) {
                spans.push(await marks(model, frameMs));
                await device.queue.onSubmittedWorkDone();
            }
            entry.marksUs[frameMs] = { median: median(spans), min: Math.min(...spans), max: Math.max(...spans) };
        }
        entry.memorySteady = since(before);
        entry.wasmBytes = wasmMemory?.buffer.byteLength ?? null;
        previous = { n, ms: (entry.gpuUs ?? entry.doneMs * 1e3) / 1e3 };

        log(`   create ${SPECIES.map(s => ms(entry.createMs[s])).join(' / ')} ms · first run ${SPECIES.map(s => ms(entry.firstMs[s])).join(' / ')} ms (prey / wolf)`);
        log(`   run ${ms(entry.runMs)} ms on the main thread · done ${ms(entry.doneMs)} ms · GPU ${entry.gpuUs === null ? '—' : fmt(Math.round(entry.gpuUs))} µs per run ` +
            `(K ${K}, blocker ${ms(blockMs)} ms${invalid ? `, ${invalid} rounds dropped` : ''})`);
        log(`   marks (frame's work 0 / 2 / 8 ms): ${FRAME_WORK_MS.map(d => fmt(Math.round(entry.marksUs[d].median))).join(' / ')} µs` +
            (entry.gpuUs ? `; minus the bracket: ${FRAME_WORK_MS.map(d => fmt(Math.round(entry.marksUs[d].median - entry.gpuUs))).join(' / ')} µs` : ''));
        log(`   ONNX Runtime's GPU buffers: ${MB(entry.memoryCreate.created)} MB at creation, ${MB(entry.memoryFirst.created)} MB in the first runs ` +
            `(${entry.memoryFirst.buffers} buffers), ${MB(entry.memorySteady.created)} MB in the ${entry.memorySteady.buffers === 0 ? 'later runs (none)' : `later runs (${entry.memorySteady.buffers} buffers)`}; ` +
            `destroyed so far ${MB(account.destroyed)} MB of ${MB(account.created)} MB · WASM ${entry.wasmBytes === null ? '—' : MB(entry.wasmBytes)} MB · ` +
            `ours: obs ${MB(round16(n * OBS * 4))} MB, results ${MB(2 * round16(n * ACT * 4))} MB`);
    } catch (e) {
        entry.failed = String(e?.message ?? e).split('\n')[0].slice(0, 240);
        check(`${name}: ${fmt(n)} agents`, false, entry.failed);
    } finally {
        for (const species of SPECIES) await models[species]?.destroy();
        await device.queue.onSubmittedWorkDone();
        entry.destroyedAfterRelease = account.destroyed;
        obs.destroy();
        SPECIES.forEach(s => out[s].destroy());
    }
    if (entry.failed) break;
}

log(`\n${name} JSON ${JSON.stringify(results)}`);
await runtime.destroy();
await finish(`${name}: measured.`);
log(`${name} done`);
