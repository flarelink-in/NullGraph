// Spike S2: where does an ONNX Runtime run land, when it is started at the top of a frame?
// See docs/agents-design.md §13 (C0) and §9.4, and docs/agents-ort-design.md.
//
//   Node + Dawn:   npm run spike:s2
//   Chrome:        npm run drive -- --serve . "/test/spikes/ort/index.html?spike=s2" --wait-for "S2 done" --timeout 300
//
// The frame is the one the parent plan gives an ORT brain: advance() starts a run when none is in
// flight; then the frame's stages are submitted (sense writes the observations, act uses whatever
// actions the buffer holds). Here sense writes the frame's number into every agent's first
// observation, the network carries a first observation through to the first action unchanged, and
// a probe pass standing in for act records which number it found. So each frame tells which
// frame's observations its actions came from.
//
// Measured: the delay in frames, at 60 frames a second and with frames as fast as they come; that
// a lockstep step (sense, await the run, act) has no delay; and whether a session with graph
// capture ignores a different input buffer on a later run (the parent's fact 12).
import { harnessEnv } from '../../gpu/harness-env.mjs';
import { LAYERS, describe, fmt, inBrowser, loadOrt, markerNetwork, median, ms, onnxOf, readF32, setting } from './common.mjs';

const { log, check, finish } = harnessEnv('__ORT_SPIKE_RESULTS__');
const N = Number(setting('agents', 20000)), FRAMES = Number(setting('frames', 300)), LOCKSTEPS = 60;
const [OBS, ACT] = [LAYERS[0], LAYERS[LAYERS.length - 1]];

const { ort } = await loadOrt();
// ONNX Runtime makes the device (spike S5: it cannot run on one the app made).
ort.env.webgpu.adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
const provider = { name: 'webgpu' };
const bytes = onnxOf(markerNetwork());
const keep = await ort.InferenceSession.create(bytes, { executionProviders: [provider] });      // keeps ONNX Runtime's device alive
const device = await ort.env.webgpu.device;
log(`S2: ${inBrowser ? 'Chrome' : 'Node/Dawn'} · ${describe(device)} · ${fmt(N)} agents · network ${LAYERS.join('→')} carrying a marker`);

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
const pipelineOf = code => device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'cs_main' } });
// sense: every agent's first observation becomes the frame's marker.
const sense = await pipelineOf(/* wgsl */ `
struct Params { marker: f32, frame: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> obs: array<f32>;
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&obs) / ${OBS}u) { obs[id.x * ${OBS}u] = params.marker; }
}`);
// probe, in act's place: what do the actions hold, and what did sense just write?
const probe = await pipelineOf(/* wgsl */ `
struct Params { marker: f32, frame: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> obs: array<f32>;
@group(0) @binding(2) var<storage, read> act: array<f32>;
@group(0) @binding(3) var<storage, read_write> seen: array<f32>;
@compute @workgroup_size(1) fn cs_main() {
    seen[params.frame * 2u] = act[0];
    seen[params.frame * 2u + 1u] = obs[0];
}`);

// A run that fails can leave ONNX Runtime waiting for ever. Nothing here waits longer than this.
const LIMIT_MS = 15000;
const within = (promise, what) => Promise.race([promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(`${what}: no answer after ${LIMIT_MS / 1000} s`)), LIMIT_MS))]);

const nextFrame = pace => (pace === 'fast' ? new Promise(r => setTimeout(r, 0))
    : inBrowser ? new Promise(r => requestAnimationFrame(r)) : new Promise(r => setTimeout(r, 1000 / 60)));

async function variant(label, capture, epForm = 'object') {
    const session = await ort.InferenceSession.create(bytes, { executionProviders: [epForm === 'object' ? provider : 'webgpu'],
        preferredOutputLocation: 'gpu-buffer', enableGraphCapture: capture });
    const obs = device.createBuffer({ size: N * OBS * 4, usage: STORAGE }), act = device.createBuffer({ size: N * ACT * 4, usage: STORAGE });
    const other = device.createBuffer({ size: N * OBS * 4, usage: STORAGE });
    const seen = device.createBuffer({ size: (FRAMES + LOCKSTEPS + 8) * 2 * 4, usage: STORAGE });
    const feeds = { obs: ort.Tensor.fromGpuBuffer(obs, { dataType: 'float32', dims: [N, OBS] }) };
    const fetches = { act: ort.Tensor.fromGpuBuffer(act, { dataType: 'float32', dims: [N, ACT] }) };
    const senseBind = device.createBindGroup({ layout: sense.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer: obs } }] });
    const probeBind = device.createBindGroup({ layout: probe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer: obs } },
        { binding: 2, resource: { buffer: act } }, { binding: 3, resource: { buffer: seen } }] });
    const pass = (encoder, pipeline, bind, groups) => {
        const p = encoder.beginComputePass();
        p.setPipeline(pipeline);
        p.setBindGroup(0, bind);
        p.dispatchWorkgroups(groups);
        p.end();
    };
    const setFrame = (marker, slot) => {
        const data = new ArrayBuffer(16);
        new Float32Array(data, 0, 1)[0] = marker;
        new Uint32Array(data, 4, 1)[0] = slot;
        device.queue.writeBuffer(params, 0, data);
    };
    const groups = Math.ceil(N / 64);
    const out = { label, capture };

    // A first run compiles ONNX Runtime's pipelines; it is not part of any frame.
    await within(session.run(feeds, fetches), 'the first run');
    await device.queue.onSubmittedWorkDone();

    for (const pace of ['frame', 'fast']) {
        device.queue.writeBuffer(act, 0, new Float32Array(N * ACT).fill(-1));         // -1: no run has landed yet
        device.queue.writeBuffer(seen, 0, new Float32Array((FRAMES + LOCKSTEPS + 8) * 2).fill(-2));
        await device.queue.onSubmittedWorkDone();
        let inFlight = false, current = 0;
        const runs = [], frameMs = [];
        let last = performance.now();
        for (let f = 1; f <= FRAMES; f++) {
            // advance(): start a run when none is in flight.
            if (!inFlight) {
                inFlight = true;
                const record = { startedIn: f, t0: performance.now() };
                runs.push(record);
                session.run(feeds, fetches).then(() => {
                    record.resolvedAfter = current;
                    record.ms = performance.now() - record.t0;
                    inFlight = false;
                }, error => {
                    record.failed = String(error?.message ?? error).split('\n')[0].slice(0, 160);
                    inFlight = false;
                });
            }
            // render(): this frame's stages, in one submission.
            setFrame(f, f);
            const encoder = device.createCommandEncoder();
            pass(encoder, sense, senseBind, groups);
            pass(encoder, probe, probeBind, 1);
            device.queue.submit([encoder.finish()]);
            current = f;
            await nextFrame(pace);
            const now = performance.now();
            frameMs.push(now - last);
            last = now;
        }
        for (const t0 = performance.now(); inFlight && performance.now() - t0 < LIMIT_MS;) await new Promise(r => setTimeout(r, 1));
        if (inFlight) throw new Error('a run started in a frame never came back');
        await device.queue.onSubmittedWorkDone();
        const log2 = await readF32(device, seen, (FRAMES + 1) * 2);
        const delays = new Map(), firstSeen = new Map();
        let waiting = 0, future = 0, sensed = 0;
        for (let f = 1; f <= FRAMES; f++) {
            const from = log2[f * 2];
            if (log2[f * 2 + 1] === f) sensed++;
            if (from < 0) { waiting++; continue; }
            if (from > f) future++;
            delays.set(f - from, (delays.get(f - from) ?? 0) + 1);
            if (!firstSeen.has(from)) firstSeen.set(from, f);
        }
        // One run is in flight at a time, and a frame's probe sees each result before the next run can
        // land. So the markers, in the order they first appeared, are the runs in the order they started.
        // Which frame's observations did a run read, against the frame that started it?
        const read = new Map();
        const markers = [...firstSeen.keys()].sort((a, b) => firstSeen.get(a) - firstSeen.get(b));
        runs.forEach((run, k) => {
            const marker = markers[k];
            if (marker === undefined) return;                                          // the last run may land after the last frame
            read.set(marker - run.startedIn, (read.get(marker - run.startedIn) ?? 0) + 1);
            run.visibleIn = firstSeen.get(marker);
        });
        const visible = runs.filter(r => r.visibleIn !== undefined).map(r => r.visibleIn - r.startedIn);
        const failures = runs.filter(r => r.failed);
        out[pace] = { frameMs: median(frameMs), runs: runs.length, failed: failures.length, firstFailure: failures[0] ? `run ${runs.indexOf(failures[0]) + 1}: ${failures[0].failed}` : null,
            runMs: median(runs.map(r => r.ms ?? 0)),
            delays: Object.fromEntries([...delays].sort((a, b) => a[0] - b[0])), waiting, future, sensed,
            readOffset: Object.fromEntries([...read].sort((a, b) => a[0] - b[0])),
            visibleAfter: visible.length ? { min: Math.min(...visible), median: median(visible), max: Math.max(...visible) } : null };
    }

    // Lockstep: sense, await the run, then act. The actions must come from this step's observations.
    {
        const times = [];
        let exact = 0;
        for (let s = 0; s < LOCKSTEPS; s++) {
            const marker = 1000 + s, slot = FRAMES + 1 + s, t0 = performance.now();
            setFrame(marker, slot);
            let encoder = device.createCommandEncoder();
            pass(encoder, sense, senseBind, groups);
            device.queue.submit([encoder.finish()]);
            await within(session.run(feeds, fetches), 'a lockstep run');
            encoder = device.createCommandEncoder();
            pass(encoder, probe, probeBind, 1);
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
            times.push(performance.now() - t0);
        }
        const steps = await readF32(device, seen, (FRAMES + 1 + LOCKSTEPS) * 2);
        for (let s = 0; s < LOCKSTEPS; s++) if (steps[(FRAMES + 1 + s) * 2] === 1000 + s) exact++;
        out.lockstep = { exact, of: LOCKSTEPS, ms: median(times) };
    }

    // Fact 12: under graph capture the buffers are bound once. Give a later run another input buffer.
    {
        device.queue.writeBuffer(obs, 0, new Float32Array([4242]));
        device.queue.writeBuffer(other, 0, new Float32Array([7777]));
        let result;
        try {
            await within(session.run({ obs: ort.Tensor.fromGpuBuffer(other, { dataType: 'float32', dims: [N, OBS] }) }, fetches), 'the run on another buffer');
            await device.queue.onSubmittedWorkDone();
            const value = (await readF32(device, act, 1))[0];
            result = value === 7777 ? 'read the new buffer' : value === 4242 ? 'ignored the new buffer and read the first one again' : `gave ${value}`;
        } catch (e) {
            result = `threw: ${String(e.message ?? e).split('\n')[0].slice(0, 120)}`;
        }
        out.swap = result;
    }
    await device.queue.onSubmittedWorkDone();
    await within(session.release(), 'releasing the session');
    for (const b of [obs, act, other, seen]) b.destroy();
    return out;
}

/** A variant that fails says so, and leaves the others their turn. */
async function attempt(label, capture, epForm) {
    log(`S2: ${label}…`);
    try {
        return await variant(label, capture, epForm);
    } catch (error) {
        const failed = String(error?.message ?? error).split('\n')[0].slice(0, 220);
        check(`S2 ${label}: ran to its end`, false, failed);
        return { label, capture, failed };
    }
}

const variants = [
    await attempt('graph capture off', false),
    await attempt('graph capture on', true),
];

const show = o => Object.entries(o).map(([k, v]) => `${k}: ${v}`).join(', ') || 'none';
function report(v) {
    log(`\n== S2 · ${v.label}`);
    if (v.failed) { log(`   FAILED: ${v.failed}`); return; }
    for (const pace of ['frame', 'fast']) {
        const p = v[pace];
        log(`   ${pace === 'frame' ? 'at 60 frames a second' : 'frames as fast as they come'}: a frame took ${ms(p.frameMs)} ms; ${p.runs} runs in ${FRAMES} frames, ${ms(p.runMs)} ms each` +
            (p.failed ? `; ${p.failed} RUNS FAILED, the first: ${p.firstFailure}` : ''));
        log(`      frames by the age of their actions, in frames: {${show(p.delays)}}; ${p.waiting} frames before the first run landed`);
        log(`      a run read the observations of the frame that started it (0) or of the one before (-1): {${show(p.readOffset)}}; ` +
            `its actions were first used ${p.visibleAfter ? `${p.visibleAfter.min} to ${p.visibleAfter.max} frames later (median ${p.visibleAfter.median})` : 'never'}`);
        check(`S2 ${v.label}, ${pace}: no frame used actions from its own or a later frame's observations, and sense wrote every frame`,
            p.future === 0 && !(0 in p.delays) && p.sensed === FRAMES, `${p.future} from the future, ${p.delays[0] ?? 0} from the same frame, ${p.sensed} of ${FRAMES} frames sensed`);
    }
    log(`   lockstep: ${v.lockstep.exact} of ${v.lockstep.of} steps acted on their own observations; ${ms(v.lockstep.ms)} ms a step`);
    check(`S2 ${v.label}: a lockstep step acts on its own observations`, v.lockstep.exact === v.lockstep.of, `${v.lockstep.exact} of ${v.lockstep.of}`);
    log(`   a later run given another input buffer: ${v.swap}`);
}
variants.forEach(report);

// With the provider given as the string "webgpu", graph capture is asked for and not passed on to the
// provider: this build passes it on only with the provider given as an object. ONNX Runtime's
// JavaScript side then keeps its bindings as if the graph were captured, and the provider does not.
// On 2026-10-14 this variant stopped answering under Node/Dawn, in a way no timer could interrupt,
// so it runs only when asked for (string=1), and last: what is above is already printed.
if (setting('string') === '1') {
    log('\nS2: the provider as a string, with graph capture: this may never come back…');
    const v = await attempt('graph capture on, the provider given as the string "webgpu"', true, 'string');
    variants.push(v);
    report(v);
}
await device.queue.onSubmittedWorkDone();
await keep.release();
log(`\nS2 JSON ${JSON.stringify({ spike: 'S2', host: inBrowser ? 'Chrome' : 'Node/Dawn', adapter: describe(device), agents: N, frames: FRAMES, variants })}`);
await finish('S2: every frame and every lockstep step behaved as the plan assumes.');
log('S2 done');
