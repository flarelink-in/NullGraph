// Spike S1: what does one ONNX Runtime run of the 24→32→32→2 network cost at 1,000, 20,000 and
// 100,000 agents, with the observations and the actions staying on the GPU, with graph capture on
// and off? See docs/agents-design.md §13 (C0) and docs/agents-ort-design.md.
//
//   Node + Dawn:   npm run spike:s1          NG_DAWN_FLAGS=adapter=AMD picks the other GPU
//   Chrome:        npm run drive -- --serve . "/test/spikes/ort/index.html?spike=s1" --wait-for "S1 done" --timeout 300
//
// What is timed, per agent count and per setting of graph capture:
//   run     the time `await session.run()` takes on the main thread
//   done    the time from calling run() until the GPU has finished its work
//   GPU     what the run occupies the GPU for. ONNX Runtime encodes its own passes, so they cannot
//           carry timestamps. Instead the GPU is first given a long pass of its own to chew on;
//           while it is busy, a marker pass, K runs and a second marker pass are queued behind it.
//           The GPU then executes them back to back, and the gap between the two markers' timestamps
//           is K runs of pure GPU time. The method is checked on the WGSL kernel, whose own pass can
//           be timed directly as spike S3 did.
// The WGSL kernel S3 measured runs beside it, on the same device, as the baseline.
//
// ONNX Runtime makes the device here. It cannot run on one the app made: that is spike S5.
import { harnessEnv } from '../../gpu/harness-env.mjs';
import { COUNTS, LAYERS, describe, fmt, loadOrt, median, mlpForward, ms, onnxOf, random, readF32, tanhNetwork, wgslKernel } from './common.mjs';

const { log, check, finish } = harnessEnv('__ORT_SPIKE_RESULTS__');
const TRIALS = 5, ROUNDS = 2, CHECK_ROWS = 256, TOLERANCE = 1e-4;
const [OBS, ACT] = [LAYERS[0], LAYERS[LAYERS.length - 1]];
const name = 'S1';

const { ort, timings } = await loadOrt();
ort.env.webgpu.adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
// Given as an object, not as the string "webgpu": only then is graph capture passed on to the provider.
const provider = { name: 'webgpu' };

const network = tanhNetwork(), bytes = onnxOf(network);
const options = capture => ({ executionProviders: [provider], preferredOutputLocation: 'gpu-buffer', enableGraphCapture: capture });
// ONNX Runtime's device does not outlive its last session, so one session is kept to the end.
const keep = await ort.InferenceSession.create(bytes, options(false));
const device = await ort.env.webgpu.device;
const L = device.limits;
log(`${name}: ${inBrowserName()} · ${describe(device)} · ONNX Runtime Web ${ort.env.versions?.web ?? ''}`);
log(`   its device: maxStorageBufferBindingSize ${fmt(L.maxStorageBufferBindingSize)}, maxBufferSize ${fmt(L.maxBufferSize)}, ` +
    `maxStorageBuffersPerShaderStage ${L.maxStorageBuffersPerShaderStage}, maxComputeInvocationsPerWorkgroup ${L.maxComputeInvocationsPerWorkgroup}, ` +
    `maxComputeWorkgroupStorageSize ${fmt(L.maxComputeWorkgroupStorageSize)}; features: ${[...device.features].join(', ') || 'none'}`);

function inBrowserName() { return typeof document !== 'undefined' ? 'Chrome' : 'Node/Dawn'; }

// ---------------------------------------------------------------------------------------------
// Buffers, the WGSL baseline, and the timestamp machinery
// ---------------------------------------------------------------------------------------------
const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const MAX_N = Math.max(...COUNTS);
const obsData = Float32Array.from({ length: MAX_N * OBS }, () => random() * 2 - 1);
const expected = new Float32Array(CHECK_ROWS * ACT);
for (let i = 0; i < CHECK_ROWS; i++) expected.set(mlpForward(network, obsData.subarray(i * OBS, (i + 1) * OBS)), i * ACT);
const buffer = (floats, data) => {
    const b = device.createBuffer({ size: floats * 4, usage: STORAGE });
    if (data) device.queue.writeBuffer(b, 0, data);
    return b;
};
const kernel = wgslKernel(network);
const weights = buffer(kernel.packed.length, kernel.packed);
const pipeline = await device.createComputePipelineAsync({ layout: 'auto',
    compute: { module: device.createShaderModule({ code: kernel.code }), entryPoint: 'cs_main' } });
const timestamps = device.features.has('timestamp-query');
const stamp = timestamps ? {
    set: device.createQuerySet({ type: 'timestamp', count: 4 }),
    resolve: device.createBuffer({ size: 32, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
    read: device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
} : null;
async function stamps(encoder) {
    encoder.resolveQuerySet(stamp.set, 0, 4, stamp.resolve, 0);
    encoder.copyBufferToBuffer(stamp.resolve, 0, stamp.read, 0, 32);
    device.queue.submit([encoder.finish()]);
    await stamp.read.mapAsync(GPUMapMode.READ);
    const t = new BigUint64Array(stamp.read.getMappedRange().slice(0));
    stamp.read.unmap();
    return [...t].map(Number);
}
const errorOf = (got, want = expected) => {
    let worst = 0;
    for (let k = 0; k < want.length; k++) worst = Math.max(worst, Math.abs(got[k] - want[k]));
    return worst;
};

/** The WGSL kernel on (obs, act): its bind group and three ways of timing it. */
function wgsl(n, obs, act, w = weights) {
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: obs } }, { binding: 1, resource: { buffer: act } }, { binding: 2, resource: { buffer: w } }] });
    const groups = Math.ceil(n / 64);
    const encode = (repeats = 1, timed = false) => {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass(timed ? { timestampWrites: { querySet: stamp.set, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind);
        for (let r = 0; r < repeats; r++) pass.dispatchWorkgroups(groups);
        pass.end();
        return encoder;
    };
    return {
        submit: (repeats = 1) => device.queue.submit([encode(repeats).finish()]),
        /** µs of GPU time per dispatch, from the pass's own timestamps: S3's method. */
        async direct(repeats) {
            const [begin, end] = await stamps(encode(repeats, true));
            return (end - begin) / repeats / 1e3;
        },
    };
}

/**
 * µs of GPU time per call of `work`, K calls: queued behind a long pass, between two marker passes.
 * `work` must only queue GPU work. Returns null when the queue was not ready before the long pass
 * ended, so that the gap held idle time.
 */
async function behindBlocker(blocker, blockerMs, K, work) {
    const mark = (begin, end) => {
        const encoder = device.createCommandEncoder();
        encoder.beginComputePass({ timestampWrites: { querySet: stamp.set, beginningOfPassWriteIndex: begin, endOfPassWriteIndex: end } }).end();
        return encoder;
    };
    blocker();
    device.queue.submit([mark(0, 1).finish()]);
    const t0 = performance.now();
    for (let k = 0; k < K; k++) await work();
    const queuedMs = performance.now() - t0;
    const t = await stamps(mark(2, 3));
    return { us: (t[2] - t[1]) / K / 1e3, queuedMs, valid: queuedMs < blockerMs * 0.8 };
}

// ---------------------------------------------------------------------------------------------
// The measurements
// ---------------------------------------------------------------------------------------------
const results = { spike: name, host: inBrowserName(), adapter: describe(device), timing: timestamps ? 'timestamps' : 'wall clock only', counts: {} };
// The blocking pass has buffers of its own. Under Dawn, a submit that uses a buffer which queued
// work still uses takes milliseconds instead of a tenth of one, and the queue would not be ready
// in time.
const block = { obs: buffer(MAX_N * OBS, obsData), act: buffer(MAX_N * ACT), weights: buffer(kernel.packed.length, kernel.packed) };
const blockerKernel = wgsl(MAX_N, block.obs, block.act, block.weights);

// Warm the GPU up, and learn how long the blocking pass takes per dispatch.
for (const t0 = performance.now(); performance.now() - t0 < 1000;) {
    blockerKernel.submit(20);
    await device.queue.onSubmittedWorkDone();
}
const blockUs = timestamps ? await blockerKernel.direct(40) : 0;       // µs per dispatch at 100,000 agents

for (const n of COUNTS) {
    const obs = buffer(n * OBS, obsData.subarray(0, n * OBS));
    const entry = results.counts[n] = { ort: {}, wgsl: {} };

    // --- the WGSL kernel -------------------------------------------------------------------
    const act = buffer(n * ACT), k = wgsl(n, obs, act);
    k.submit();
    const wgslError = errorOf(await readF32(device, act, CHECK_ROWS * ACT));
    check(`${name}: the WGSL kernel at ${fmt(n)} agents equals mlpForward`, wgslError <= TOLERANCE, `worst ${wgslError.toExponential(1)}`);
    {
        const submit = [], done = [], direct = [], bracket = [];
        for (let round = 0; round < ROUNDS * TRIALS; round++) {
            let t0 = performance.now();
            for (let r = 0; r < 20; r++) k.submit();
            submit.push((performance.now() - t0) / 20);
            await device.queue.onSubmittedWorkDone();
            t0 = performance.now();
            k.submit();
            await device.queue.onSubmittedWorkDone();
            done.push(performance.now() - t0);
            if (timestamps) {
                const repeats = Math.max(8, Math.ceil(5000 / Math.max(1, (blockUs * n) / MAX_N)));      // ≥ 5 ms per timed pass
                direct.push(await k.direct(repeats));
                const K = 16, blockDispatches = Math.ceil(120000 / blockUs);                              // about 120 ms of blocker; Dawn makes a submit wait once some thirty are queued
                const b = await behindBlocker(() => blockerKernel.submit(blockDispatches), blockDispatches * blockUs / 1e3, K, async () => k.submit());
                if (b.valid) bracket.push(b.us);
            }
        }
        Object.assign(entry.wgsl, { submitMs: median(submit), doneMs: median(done), gpuUs: direct.length ? median(direct) : null,
            gpuUsBehindBlocker: bracket.length ? median(bracket) : null });
    }

    // --- ONNX Runtime, graph capture off and on ----------------------------------------------
    for (const capture of [false, true]) {
        const out = buffer(n * ACT), slot = entry.ort[capture ? 'captureOn' : 'captureOff'] = {};
        let session;
        try {
            let t0 = performance.now();
            session = await ort.InferenceSession.create(bytes, options(capture));
            slot.createMs = performance.now() - t0;
            const feeds = { obs: ort.Tensor.fromGpuBuffer(obs, { dataType: 'float32', dims: [n, OBS] }) };
            const fetches = { act: ort.Tensor.fromGpuBuffer(out, { dataType: 'float32', dims: [n, ACT] }) };
            const run = () => session.run(feeds, fetches);
            t0 = performance.now();
            await run();
            await device.queue.onSubmittedWorkDone();
            slot.firstRunMs = performance.now() - t0;
            const got = await readF32(device, out, CHECK_ROWS * ACT);
            slot.error = errorOf(got);
            slot.againstWgsl = errorOf(got, await readF32(device, act, CHECK_ROWS * ACT));
            check(`${name}: ONNX Runtime at ${fmt(n)} agents, graph capture ${capture ? 'on' : 'off'}, writes our buffer what mlpForward gives`,
                slot.error <= TOLERANCE && slot.againstWgsl <= TOLERANCE, `worst ${slot.error.toExponential(1)}; against the WGSL kernel ${slot.againstWgsl.toExponential(1)}`);

            for (const t1 = performance.now(); performance.now() - t1 < 500;) await run();           // warm up
            await device.queue.onSubmittedWorkDone();
            const runMs = [], doneMs = [], gpu = [];
            let invalid = 0;
            for (let round = 0; round < ROUNDS * TRIALS; round++) {
                let t1 = performance.now();
                for (let r = 0; r < 20; r++) await run();
                runMs.push((performance.now() - t1) / 20);
                await device.queue.onSubmittedWorkDone();
                t1 = performance.now();
                await run();
                await device.queue.onSubmittedWorkDone();
                doneMs.push(performance.now() - t1);
                if (timestamps) {
                    // A blocker of 300 ms at most (Windows resets a GPU that stays busy for about 2 s), and
                    // as many runs as can be queued in half of it.
                    const wanted = 300, K = Math.max(3, Math.min(20, Math.floor((0.5 * wanted) / median(runMs))));
                    const blockDispatches = Math.ceil((wanted * 1e3) / blockUs);
                    const b = await behindBlocker(() => blockerKernel.submit(blockDispatches), blockDispatches * blockUs / 1e3, K, run);
                    if (b.valid) gpu.push(b.us); else invalid++;
                }
            }
            Object.assign(slot, { runMs: median(runMs), doneMs: median(doneMs), gpuUs: gpu.length ? median(gpu) : null, gpuInvalid: invalid });
            await device.queue.onSubmittedWorkDone();
            await session.release();
        } catch (e) {
            slot.failed = String(e.message ?? e).split('\n')[0].slice(0, 200);
            check(`${name}: ONNX Runtime at ${fmt(n)} agents, graph capture ${capture ? 'on' : 'off'}`, false, slot.failed);
            try { await session?.release(); } catch { /* already gone */ }
        }
        out.destroy();
    }
}

// ---------------------------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------------------------
const cell = v => (v === null || v === undefined ? '—' : typeof v === 'number' ? ms(v) : v).toString().padStart(9);
log(`\n== ${name} · ${results.host} · ${results.adapter} — network ${LAYERS.join('→')} tanh; medians of ${ROUNDS * TRIALS}; ${results.timing}`);
log(`   ${'agents'.padEnd(8)}${'what'.padEnd(34)}${'run ms'.padStart(9)}${'done ms'.padStart(9)}${'GPU µs'.padStart(9)}${'first ms'.padStart(9)}${'create ms'.padStart(10)}`);
for (const n of COUNTS) {
    const e = results.counts[n];
    log(`   ${fmt(n).padEnd(8)}${'WGSL kernel (one submit each)'.padEnd(34)}${cell(e.wgsl.submitMs)}${cell(e.wgsl.doneMs)}${cell(e.wgsl.gpuUs)}` +
        `${''.padStart(19)}   behind the blocker: ${e.wgsl.gpuUsBehindBlocker === null ? '—' : ms(e.wgsl.gpuUsBehindBlocker)} µs`);
    for (const [label, key] of [['ONNX Runtime, capture off', 'captureOff'], ['ONNX Runtime, capture on', 'captureOn']]) {
        const s = e.ort[key];
        log(s.failed ? `   ${''.padEnd(8)}${label.padEnd(34)} failed: ${s.failed}`
            : `   ${''.padEnd(8)}${label.padEnd(34)}${cell(s.runMs)}${cell(s.doneMs)}${cell(s.gpuUs)}${cell(s.firstRunMs)}${cell(s.createMs).padStart(10)}` +
              (s.gpuInvalid ? `   (${s.gpuInvalid} GPU samples dropped: the queue was not ready in time)` : ''));
    }
}

// What is left of ONNX Runtime's device when its last session is released?
await device.queue.onSubmittedWorkDone();
await keep.release();
let after;
try {
    after = `alive: a read-back gives ${(await readF32(device, block.act, 2)).length} values`;
} catch (e) {
    after = `gone: a read-back fails with ${e?.name ?? 'an error'}`;
}
results.deviceAfterLastSession = after;
log(`   after the last session was released, ONNX Runtime's device is ${after}`);
log(`   start-up seen here: ${JSON.stringify(timings)}`);
log(`\n${name} JSON ${JSON.stringify(results)}`);
await finish(`${name}: every run matched mlpForward and the WGSL kernel.`);
log(`${name} done`);
