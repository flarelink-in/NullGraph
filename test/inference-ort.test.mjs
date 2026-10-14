// Step 9, phases 9.1, 9.3 and 9.4 (docs/agents-ort-design.md §5–§7, §10): OrtRuntime, OrtModel
// and OrtBrain, against a fake onnxruntime-web that records every call in order.
//
//   U32  the runtime makes ORT's device through an anchor session, created first and released
//        last; every session gets [{ name: 'webgpu' }], 'gpu-buffer', and graph capture only if
//        asked (phase 9.4), and the app cannot override them; bind checks names, types, ranks,
//        fixed axes, named axes, sizes and usage, once; one run at a time; destroy releases after
//        a run in flight
//   U36  (after step 10's phase 10.3) ONNX Runtime takes one call at a time: every create, run
//        and release of a runtime's models waits in line, in the order made; a call starts at
//        once when none is in line, and a call that fails does not hold the line
//   U33  OrtBrain (phase 9.3): dims and names from the model; a fixed first axis refused with
//        the fix; its results buffer seeded from act before the model is bound; the same buffers
//        re-bound, others refused; its copy pass parses inside the think module
//   U14  the refusals of the design's §11 that these phases build fire and name the offender
//   size the adapter's line of 3,100 B (3,000 B until U36); mlpFromOnnx alone carries none of it
//   types the real onnxruntime-web namespace is an OrtLike
//
// Imports dist/: run `npm run build:lib` first.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';
import { build } from 'esbuild';

globalThis.GPUBufferUsage ??= { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64 };
const { OrtRuntime } = await import('../dist/inference.js');
const root = fileURLToPath(new URL('../', import.meta.url)).replace(/\\/g, '/');

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const tensor = (name, shape, type = 'float32') => ({ name, isTensor: true, type, shape });
const OBS = tensor('obs', ['N', 24]), ACT = tensor('act', ['N', 2]);
const buffer = (size, usage = STORAGE) => ({ size, usage });
const N = 1000, obs = buffer(N * 24 * 4), act = buffer(N * 2 * 4);
const bound = () => ({ inputs: { obs: { buffer: obs, dims: [N, 24] } }, outputs: { act: { buffer: act, dims: [N, 2] } } });
const MODEL = new Uint8Array([8, 8]);

/**
 * A fake onnxruntime-web. Session 0 is the anchor. `log` holds every call in order; each session's
 * runs wait until the test settles them (`session.settle(error?)`).
 */
function fakeOrt({ inputs = [OBS], outputs = [ACT], failCreate = () => false, device = { label: 'ORT\'s device' } } = {}) {
    const log = [], sessions = [];
    const ort = {
        env: { webgpu: { get device() { log.push(['device']); return Promise.resolve(device); } } },
        InferenceSession: {
            async create(bytes, options) {
                const id = sessions.length;
                log.push(['create', id, options]);
                if (failCreate(id)) throw new Error('failed to call OrtCreateSession(). ERROR_CODE: 2\nsecond line');
                const runs = [];
                const session = {
                    bytes, inputMetadata: id ? inputs : [tensor('x', [1])], outputMetadata: id ? outputs : [tensor('y', [1])], runs,
                    run(feeds, fetches) {
                        log.push(['run', id]);
                        return new Promise((done, failed) => runs.push({ feeds, fetches, done, failed }));
                    },
                    settle(error) { const r = runs.find(x => !x.settled); r.settled = true; error ? r.failed(error) : r.done({}); },
                    async release() { log.push(['release', id]); },
                };
                sessions.push(session);
                return session;
            },
        },
        Tensor: { fromGpuBuffer: (gpuBuffer, options) => ({ gpuBuffer, ...options }) },
    };
    return { ort, log, sessions };
}
const refusal = (who, pattern) => error => error instanceof Error && error.message.startsWith(`${who}: `) && pattern.test(error.message);
const tick = () => new Promise(done => setTimeout(done, 0));

test('U32: the runtime starts ONNX Runtime with an anchor session, and takes its device', async () => {
    const { ort, log, sessions } = fakeOrt();
    const runtime = await OrtRuntime.create(ort);
    assert.equal(runtime.device.label, 'ORT\'s device');
    assert.deepEqual(log, [['create', 0, { executionProviders: [{ name: 'webgpu' }] }], ['device']],
        'the anchor session comes first, with the provider given as an object; the device is read after it');
    const anchor = sessions[0].bytes;
    assert.ok(anchor instanceof Uint8Array && anchor.length === 66 && anchor[0] === 8 && anchor[1] === 8, 'the anchor is the 66-byte model of ir_version 8');
    assert.ok(!('adapter' in ort.env.webgpu), 'no adapter is chosen for ONNX Runtime unless one is given');

    const adapter = { label: 'the app\'s adapter' }, second = fakeOrt();
    await OrtRuntime.create(second.ort, { adapter });
    assert.equal(second.ort.env.webgpu.adapter, adapter, 'a given adapter is set before the anchor is created');
});

test('U32: load sets the provider as an object, outputs on the GPU and graph capture only if asked, and the app cannot change them', async () => {
    const { ort, log, sessions } = fakeOrt();
    const runtime = await OrtRuntime.create(ort);
    const optionsOf = id => log.find(entry => entry[0] === 'create' && entry[1] === id)[2];
    const forced = capture => ({ executionProviders: [{ name: 'webgpu' }], preferredOutputLocation: 'gpu-buffer', enableGraphCapture: capture });

    // Under capture ONNX Runtime can fail a model from its second run, and its memory grows with
    // every run (phases 9.2 and 9.4): capture is off unless the app asks for it (decided with the
    // user, the design's §14, item 6).
    const model = await runtime.load(MODEL);
    assert.deepEqual(optionsOf(1), forced(false), 'graph capture is off by default');
    assert.equal(model.capture, false);
    assert.equal(sessions[1].bytes, MODEL, 'the bytes go to ONNX Runtime as they are');
    assert.equal((await runtime.load(MODEL, { capture: true })).capture, true);
    assert.deepEqual(optionsOf(2), forced(true), 'capture: true turns it on');
    await runtime.load(MODEL, { capture: false, executionProviders: ['webgpu'], preferredOutputLocation: 'cpu', enableGraphCapture: true });
    assert.deepEqual(optionsOf(3), forced(false), 'other session options are not taken from the app');
    const externalData = [{ path: 'model.onnx.data', data: new Uint8Array(4) }];
    const twoFiles = await runtime.load(MODEL, { externalData });
    assert.equal(optionsOf(4).externalData, externalData, 'externalData goes to ONNX Runtime as it is');
    assert.equal(optionsOf(4).enableGraphCapture, false, 'weights in a second file: capture off, as for any model');
    assert.equal(twoFiles.capture, false);
    assert.equal((await runtime.load(MODEL, { externalData, capture: true })).capture, true, 'unless the app asks for it');
    assert.equal(optionsOf(5).enableGraphCapture, true);
    assert.equal((await runtime.load(MODEL, { capture: false })).capture, false, 'capture: false is the default, said');
    await runtime.load(MODEL.buffer.slice(0));
    assert.deepEqual([...sessions[7].bytes], [...MODEL], 'an ArrayBuffer is taken as the same bytes');
    await assert.rejects(runtime.load([8, 8]), refusal('OrtRuntime', /^OrtRuntime: expected the model's bytes, as a Uint8Array or an ArrayBuffer\.$/));

    assert.deepEqual(model.inputs, [{ name: 'obs', type: 'float32', dims: ['N', 24] }]);
    assert.deepEqual(model.outputs, [{ name: 'act', type: 'float32', dims: ['N', 2] }]);
    const other = await OrtRuntime.create(fakeOrt({ inputs: [{ name: 'seq', isTensor: false }, { name: 'free', isTensor: true, type: 'float32', shape: [] }] }).ort);
    assert.deepEqual((await other.load(MODEL)).inputs, [{ name: 'seq', type: 'not a tensor', dims: [] }, { name: 'free', type: 'float32', dims: [] }]);
});

test('U32: bind checks every tensor and makes the feeds and fetches once', async () => {
    const { ort, sessions } = fakeOrt({ inputs: [OBS, tensor('free', [-1, 3]), tensor('any', [])], outputs: [ACT, tensor('pair', [2, 'N', ''])] });
    const runtime = await OrtRuntime.create(ort);
    const model = await runtime.load(MODEL);
    // Every buffer holds its values rounded up to 16 B: 60 B of values need 64 B, and 4 B need 16 B.
    const all = () => ({ inputs: { ...bound().inputs, free: { buffer: buffer(64), dims: [5, 3] }, any: { buffer: buffer(16), dims: [1] } },
        outputs: { ...bound().outputs, pair: { buffer: buffer(2 * N * 7 * 4), dims: [2, N, 7] } } });
    const refused = (change, pattern) => {
        const given = all();
        change(given);
        assert.throws(() => model.bind(given), refusal('OrtModel', pattern), String(pattern));
    };
    refused(g => { g.inputs.extra = g.inputs.obs; }, /^OrtModel: "extra" is not an input of this model; it has "obs", "free", "any"\.$/);
    refused(g => { g.outputs.obs = g.inputs.obs; }, /^OrtModel: "obs" is not an output of this model; it has "act", "pair"\.$/);
    refused(g => { delete g.inputs.free; }, /^OrtModel: input "free" has no buffer; every input and output is bound to one\.$/);
    refused(g => { delete g.outputs.act; }, /^OrtModel: output "act" has no buffer; every input and output is bound to one\.$/);
    refused(g => { g.inputs.obs = { buffer: obs, dims: [N, 23] }; }, /^OrtModel: input "obs" is \[N,24\]; it was given \[1000,23\]\.$/);
    refused(g => { g.inputs.obs = { buffer: obs, dims: [N * 24] }; }, /^OrtModel: input "obs" is \[N,24\]; it was given \[24000\]\.$/);
    refused(g => { g.inputs.free = { buffer: obs, dims: [0, 3] }; }, /^OrtModel: input "free" is \[-1,3\]; it was given \[0,3\]\.$/);
    refused(g => { g.inputs.free = { buffer: obs, dims: [2.5, 3] }; }, /given \[2\.5,3\]/);
    refused(g => { g.outputs.act = { buffer: act, dims: [N - 1, 2] }; }, /^OrtModel: the axis "N" is 1000 in "obs" and 999 in "act"\.$/);
    const rounded = 'with STORAGE usage: its values, rounded up to 16 B, as ONNX Runtime reads and writes 16 B at a time; its buffer has';
    refused(g => { g.inputs.obs = { buffer: buffer(N * 24 * 4 - 4), dims: [N, 24] }; }, new RegExp(`^OrtModel: "obs" needs 96000 B ${rounded} 95996 B\\.$`));
    refused(g => { g.outputs.act = { buffer: buffer(N * 2 * 4, GPUBufferUsage.COPY_SRC), dims: [N, 2] }; }, new RegExp(`^OrtModel: "act" needs 8000 B ${rounded} 8000 B, without STORAGE\\.$`));
    // Values that end inside the last 16 B: a buffer of exactly their size is too small (on NVIDIA,
    // ONNX Runtime's last write would be dropped whole: phase 9.2).
    refused(g => { g.inputs.free = { buffer: buffer(60), dims: [5, 3] }; }, new RegExp(`^OrtModel: "free" needs 64 B ${rounded} 60 B\\.$`));
    refused(g => { g.inputs.any = { buffer: buffer(4), dims: [1] }; }, new RegExp(`^OrtModel: "any" needs 16 B ${rounded} 4 B\\.$`));

    // A refused binding binds nothing: the model still takes a good one, once.
    model.bind(all());
    const run = model.run(), [first] = sessions[1].runs;
    assert.deepEqual(Object.keys(first.feeds), ['obs', 'free', 'any']);
    assert.deepEqual(first.feeds.obs, { gpuBuffer: obs, dataType: 'float32', dims: [N, 24] }, 'a tensor over the app\'s own buffer');
    assert.deepEqual(first.fetches.pair, { gpuBuffer: all().outputs.pair.buffer, dataType: 'float32', dims: [2, N, 7] }, 'every output is pre-allocated');
    sessions[1].settle();
    await run;
    assert.throws(() => model.bind(all()), refusal('OrtModel', /^OrtModel: bound already; buffers are fixed for life, because graph capture binds them once\. Load the model again for other buffers\.$/));

    const typed = await (await OrtRuntime.create(fakeOrt({ inputs: [tensor('ids', ['N', 4], 'int64')] }).ort)).load(MODEL);
    assert.throws(() => typed.bind({ inputs: { ids: { buffer: obs, dims: [N, 4] } }, outputs: bound().outputs }),
        refusal('OrtModel', /^OrtModel: input "ids" is int64; only float32 tensors can be bound\.$/));
    const sequence = await (await OrtRuntime.create(fakeOrt({ outputs: [{ name: 'list', isTensor: false }] }).ort)).load(MODEL);
    assert.throws(() => sequence.bind({ inputs: bound().inputs, outputs: { list: { buffer: act, dims: [N, 2] } } }),
        refusal('OrtModel', /^OrtModel: output "list" is not a tensor; only float32 tensors can be bound\.$/));
});

test('U32: one run at a time, over the same feeds and fetches, with ONNX Runtime\'s own errors', async () => {
    const { ort, sessions } = fakeOrt();
    const model = await (await OrtRuntime.create(ort)).load(MODEL);
    await assert.rejects(model.run(), refusal('OrtModel', /^OrtModel: run\(\) before bind\(\)\.$/));
    model.bind(bound());
    const first = model.run();
    // A second run that started would wait for ever on the fake; the race makes that fail at once.
    await assert.rejects(Promise.race([model.run(), tick().then(() => { throw new Error('a second run started'); })]),
        refusal('OrtModel', /^OrtModel: run\(\) while a run is in flight; await it first\.$/));
    assert.equal(sessions[1].runs.length, 1, 'the refused run reached nothing');
    sessions[1].settle();
    await first;
    const second = model.run();
    const [a, b] = sessions[1].runs;
    assert.ok(a.feeds === b.feeds && a.fetches === b.fetches, 'every run passes the objects bind made: a run allocates nothing on this side');
    sessions[1].settle(new Error('failed to call OrtRun(). ERROR_CODE: 1'));
    await assert.rejects(second, /^Error: failed to call OrtRun\(\)\. ERROR_CODE: 1$/, 'ONNX Runtime\'s error, unchanged');
    const third = model.run();
    sessions[1].settle();
    await third;
});

test('U32: destroy releases each model after the run in flight, and the anchor last', async () => {
    const { ort, log, sessions } = fakeOrt();
    const runtime = await OrtRuntime.create(ort);
    const [a, b, c] = [await runtime.load(MODEL), await runtime.load(MODEL), await runtime.load(MODEL)];
    a.bind(bound());
    const running = a.run();
    // Every release waits in line behind a's run, c's too: one call at a time (U36).
    const goneC = c.destroy();
    await c.destroy();
    const destroyed = runtime.destroy();
    await tick();
    const releases = () => log.filter(entry => entry[0] === 'release').map(entry => entry[1]);
    assert.deepEqual(releases(), [], 'nothing is released while a run is in flight; a second destroy returns at once');
    sessions[1].settle();
    await running;
    await goneC;
    await destroyed;
    assert.deepEqual(releases(), [3, 1, 2, 0], 'each session released once, in the order asked, the anchor last');
    await runtime.destroy();
    assert.equal(releases().length, 4, 'destroy twice releases nothing twice');

    await assert.rejects(runtime.load(MODEL), refusal('OrtRuntime', /^OrtRuntime: used after destroy\(\)\.$/));
    assert.throws(() => b.bind(bound()), refusal('OrtModel', /^OrtModel: used after destroy\(\)\.$/));
    await assert.rejects(a.run(), refusal('OrtModel', /^OrtModel: used after destroy\(\)\.$/));
});

// Step 10's phase 10.3 found that ONNX Runtime 1.29 fails two calls in flight, across sessions: a
// session created while another is being created, or a create or a run while another session runs
// (its WASM memory breaks). Decided with the user, 2026-10-15 (work/plan.md §5, item 14).
test('U36: ONNX Runtime takes one call at a time: every create, run and release waits in line, in the order made', async () => {
    const { ort, log, sessions } = fakeOrt();
    const runtime = await OrtRuntime.create(ort);
    const calls = () => log.filter(([what]) => what !== 'device').map(([what, id]) => `${what} ${id}`);

    // Two loads at once: the second session is created once the first create has settled. Each
    // create here waits until the test lets it go.
    const create = ort.InferenceSession.create, held = [];
    ort.InferenceSession.create = (...args) => new Promise(done => held.push(() => done(create(...args))));
    const loads = [runtime.load(MODEL), runtime.load(MODEL)];
    await tick();
    assert.equal(held.length, 1, 'the second load waits for the first create');
    held.shift()();
    const a = await loads[0];
    await tick();
    assert.equal(held.length, 1, 'then the second create starts');
    held.shift()();
    const b = await loads[1];
    ort.InferenceSession.create = create;

    // With nothing in line a call starts at once; another model's call waits for it.
    a.bind(bound());
    b.bind(bound());
    const ranA = a.run();
    assert.equal(sessions[1].runs.length, 1, 'with nothing in line, a run starts at once');
    const ranB = b.run(), loadingC = runtime.load(MODEL);
    await tick();
    assert.equal(sessions[2].runs.length, 0, 'b\'s run waits for a\'s run in flight');
    assert.equal(sessions.length, 3, 'so does a load');
    sessions[1].settle(new Error('failed to call OrtRun(). ERROR_CODE: 1'));
    await assert.rejects(ranA, /^Error: failed to call OrtRun\(\)\. ERROR_CODE: 1$/, 'ONNX Runtime\'s error, to the run that met it');
    await tick();
    assert.equal(sessions[2].runs.length, 1, 'a call that failed does not hold the line: b\'s run starts');
    assert.equal(sessions.length, 3, 'the load waits behind it');
    sessions[2].settle();
    await ranB;
    const c = await loadingC;

    // Releases wait in line too.
    c.bind(bound());
    const ranC = c.run(), destroyed = runtime.destroy();
    await tick();
    assert.deepEqual(calls().filter(call => call.startsWith('release')), [], 'nothing is released while a run is in flight');
    sessions[3].settle();
    await ranC;
    await destroyed;
    assert.deepEqual(calls(), ['create 0', 'create 1', 'create 2', 'run 1', 'run 2', 'create 3', 'run 3', 'release 1', 'release 2', 'release 3', 'release 0'],
        'one call at a time, in the order the app made them, the anchor\'s release last');
});

test('U14: the runtime\'s refusals name what went wrong', async () => {
    const notOrt = /^OrtRuntime: that is not onnxruntime-web \(no InferenceSession, Tensor\.fromGpuBuffer or env\.webgpu\); pass the namespace of import \* as ort from 'onnxruntime-web\/webgpu'\.$/;
    const { ort } = fakeOrt();
    for (const wrong of [undefined, {}, { ...ort, Tensor: {} }, { ...ort, env: {} }, { ...ort, InferenceSession: { create: 1 } }]) {
        await assert.rejects(OrtRuntime.create(wrong), refusal('OrtRuntime', notOrt));
    }
    await assert.rejects(OrtRuntime.create(fakeOrt({ failCreate: id => id === 0 }).ort),
        refusal('OrtRuntime', /^OrtRuntime: ONNX Runtime could not start its WebGPU provider: failed to call OrtCreateSession\(\)\. ERROR_CODE: 2\.$/));
    const deviceless = fakeOrt({ device: null });
    await assert.rejects(OrtRuntime.create(deviceless.ort),
        refusal('OrtRuntime', /^OrtRuntime: ONNX Runtime could not start its WebGPU provider: it made no device\.$/));
    assert.deepEqual(deviceless.log.at(-1), ['release', 0], 'the anchor is released again');

    const failing = await OrtRuntime.create(fakeOrt({ failCreate: id => id === 1 }).ort);
    await assert.rejects(failing.load(MODEL), refusal('OrtRuntime', /^OrtRuntime: ONNX Runtime could not load the model: failed to call OrtCreateSession\(\)\. ERROR_CODE: 2\.$/));
    const old = fakeOrt({ inputs: undefined });
    old.ort.InferenceSession.create = (create => async (...args) => Object.assign(await create(...args), { inputMetadata: undefined }))(old.ort.InferenceSession.create);
    const oldRuntime = await OrtRuntime.create(old.ort).catch(error => error);
    assert.ok(oldRuntime instanceof OrtRuntime, 'the anchor needs no metadata');
    await assert.rejects(oldRuntime.load(MODEL),
        refusal('OrtRuntime', /^OrtRuntime: this ONNX Runtime reports no input or output metadata; onnxruntime-web 1\.29 or later is needed\.$/));
    assert.deepEqual(old.log.at(-1), ['release', 1], 'that session is released again');
});

/** A device for OrtBrain's results buffer and its one copy; `log` holds what it was asked to do. */
function fakeDevice() {
    const log = [];
    const device = { label: 'ORT\'s device', log,
        createBuffer(o) { const b = { ...o, destroyed: false, destroy() { b.destroyed = true; log.push(['destroy', o.label]); } }; log.push(['buffer', o.label, o.size]); return b; },
        createCommandEncoder() { return { copyBufferToBuffer: (...args) => log.push(['copy', ...args]), finish: () => 'commands' }; },
        queue: { submit: list => log.push(['submit', ...list]) } };
    return device;
}
async function brainFixture({ inputs = [OBS], outputs = [ACT] } = {}) {
    const device = fakeDevice(), fake = fakeOrt({ inputs, outputs, device });
    const runtime = await OrtRuntime.create(fake.ort), model = await runtime.load(MODEL);
    return { device, fake, runtime, model, session: fake.sessions[1] };
}
const ioFor = (capacity = N, extra = {}) => ({ obs: buffer(capacity * 24 * 4), act: buffer(capacity * 2 * 4, STORAGE | GPUBufferUsage.COPY_SRC), capacity, obsDim: 24, actDim: 2, ...extra });

test('U33: OrtBrain takes its dims and names from the model, and refuses what it cannot run', async () => {
    const { OrtBrain } = await import('../dist/inference.js');
    const { model } = await brainFixture();
    const brain = new OrtBrain(model);
    assert.equal(brain.kind, 'async');
    assert.deepEqual(brain.dims, { obs: 24, act: 2 });
    assert.equal(brain.buffers, undefined, 'no results buffer before bind');
    assert.equal(brain.model, model);
    const free = new OrtBrain((await brainFixture({ inputs: [tensor('obs', ['N', 'width'])] })).model);
    assert.equal(free.dims, undefined, 'a free width is checked at bind, against the system');
    const refused = async (setup, pattern, options) => assert.throws(() => new OrtBrain(setup.model, options), refusal('OrtBrain', pattern));
    const two = await brainFixture({ inputs: [OBS, tensor('mask', ['N', 24])], outputs: [ACT, tensor('value', ['N', 1])] });
    await refused(two, /^OrtBrain: the model has 2 inputs \("obs", "mask"\); name the observations with \{ input: '<name>' \}\.$/);
    await refused(two, /^OrtBrain: the model has 2 outputs \("act", "value"\); name the actions with \{ output: '<name>' \}\.$/, { input: 'obs' });
    assert.deepEqual(new OrtBrain(two.model, { input: 'mask', output: 'act' }).dims, { obs: 24, act: 2 });
    await refused(two, /^OrtBrain: the model has no input "nope"; it has "obs", "mask"\.$/, { input: 'nope' });
    await refused(await brainFixture({ inputs: [tensor('obs', ['N', 24], 'int64')] }), /^OrtBrain: input "obs" is \[N,24\] int64; expected \[N, obsDim\] float32\.$/);
    await refused(await brainFixture({ outputs: [tensor('act', ['N', 1, 2])] }), /^OrtBrain: output "act" is \[N,1,2\] float32; expected \[N, actDim\] float32\.$/);
});

test('U33: bind makes a results buffer rounded up to 16 B, seeds it from act, then binds the model; once per system', async () => {
    const { OrtBrain } = await import('../dist/inference.js');
    const { device, model, session } = await brainFixture();
    const brain = new OrtBrain(model), io = ioFor(1001);           // 1001 × 2 × 4 = 8008 B of actions
    brain.assertDevice(device);
    brain.bind(io);
    const [results] = brain.buffers;
    assert.deepEqual(device.log.slice(0, 3), [['buffer', 'OrtBrain results', 8016], ['copy', io.act, 0, results, 0, 8008], ['submit', 'commands']],
        'a results buffer of the actions rounded up to 16 B, seeded with the actions agents have now');
    assert.equal(results.usage & GPUBufferUsage.STORAGE, GPUBufferUsage.STORAGE);
    const run = brain.run(), [first] = session.runs;
    assert.deepEqual(first.feeds.obs, { gpuBuffer: io.obs, dataType: 'float32', dims: [1001, 24] });
    assert.deepEqual(first.fetches.act, { gpuBuffer: results, dataType: 'float32', dims: [1001, 2] });
    session.settle();
    await run;
    // Again for the same system (a species given back to it): seeded again, bound once.
    brain.bind(io);
    assert.deepEqual(device.log.slice(3), [['copy', io.act, 0, results, 0, 8008], ['submit', 'commands']]);
    assert.throws(() => brain.bind(ioFor(1001)), refusal('OrtBrain', /^OrtBrain: bound to another system's buffers; buffers are fixed for life\. Load the model again for this system\.$/));
    assert.throws(() => brain.assertDevice({ label: 'the engine\'s own' }),
        refusal('OrtBrain', /^OrtBrain: this system's engine has another device\. ONNX Runtime 1\.29 runs only on a device it made: init the engine with \{ device: runtime\.device \}\.$/));
    // destroy waits for a run in flight before it releases the results buffer; the model stays the app's.
    const last = brain.run();
    brain.destroy();
    assert.ok(!results.destroyed, 'kept while a run may still write it');
    session.settle();
    await last;
    await Promise.resolve();
    assert.ok(results.destroyed);
    assert.ok(!session.released, 'the model stays the app\'s');
    assert.throws(() => brain.assertDevice(device), refusal('OrtBrain', /used after destroy/));
    await assert.rejects(brain.run(), refusal('OrtBrain', /used after destroy/));
});

test('U33: a fixed first axis must be the system\'s capacity; a refused bind leaves no buffer', async () => {
    const { OrtBrain } = await import('../dist/inference.js');
    const { device, model } = await brainFixture({ inputs: [tensor('obs', [1, 24])], outputs: [tensor('act', [1, 2])] });
    const brain = new OrtBrain(model);
    assert.throws(() => brain.bind(ioFor(1000)), refusal('OrtBrain',
        /^OrtBrain: input "obs" is \[1,24\]: its first axis is fixed at 1, and the system has 1000 agents\. Export it with a symbolic first axis \(PyTorch: dynamic_shapes, or dynamic_axes=\{'obs': \{0: 'N'\}\}\)\.$/));
    assert.equal(device.log.length, 0, 'refused before any buffer is made');
    brain.bind(ioFor(1));                                   // a fixed axis that is the capacity
    // The model's own refusal, after the buffer was made: the buffer is released again.
    const wide = await brainFixture({ inputs: [tensor('obs', ['N', 23])] });
    const narrow = new OrtBrain(wide.model);
    assert.throws(() => narrow.bind(ioFor(4)), refusal('OrtModel', /input "obs" is \[N,23\]; it was given \[4,24\]/));
    assert.deepEqual(wide.device.log.filter(([what]) => what !== 'copy' && what !== 'submit'), [['buffer', 'OrtBrain results', 32], ['destroy', 'OrtBrain results']]);
    assert.equal(narrow.buffers, undefined);
});

test('U33: the copy pass is a think pass: it parses inside the generated module, and an AgentSystem runs it as one', async () => {
    const { OrtBrain } = await import('../dist/inference.js');
    const A = await import('../dist/agents.js');
    const { WgslReflect } = await import('../node_modules/wgsl_reflect/wgsl_reflect.module.js');
    const { model } = await brainFixture({ inputs: [tensor('obs', ['N', 'width'])], outputs: [tensor('act', ['N', 'actions'])] });
    const brain = new OrtBrain(model);
    assert.ok(!brain.wgsl.includes('`'));
    const definition = A.defineAgents({ capacity: 64, world: { min: [-8, -8], max: [8, 8] }, cellSize: 4,
        species: [{ name: 'prey', maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 4, radius: 0.4 }],
        sensors: [A.Sensors.nearest({ name: 'near', k: 2 })] });
    const code = A.buildAgentThinkWGSL(definition, brain.wgsl);
    const reflect = new WgslReflect(code);
    assert.deepEqual(reflect.storage.filter(v => v.group === 1).map(v => [v.name, v.binding]), [['ortResults', 0]]);
    assert.match(code, /setAct\(i, k, ortResults\[i \* ACT_DIM \+ k\]\)/);
    // In a system on the mock engine: bound, then one think pass whose group 1 is the results buffer.
    const stages = [], engine = { device: Object.assign(model.device, { limits: { maxStorageBufferBindingSize: 1 << 27, maxBufferSize: 1 << 28,
        maxComputeWorkgroupsPerDimension: 65535, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256, maxStorageBuffersPerShaderStage: 8 } }),
        createComputeStage(o) { const s = { ...o, enabled: true, preClearBuffers: [] }; stages.push(s); return s; },
        setComputePassBindGroup(stage, entries, group) { (stage.groups ??= [])[group] = entries; }, render() {} };
    engine.device.queue.writeBuffer = () => {};
    engine.device.queue.onSubmittedWorkDone = async () => {};
    const agents = new A.AgentSystem(engine, definition);
    agents.setBrain(brain);
    const think = stages.find(s => s.name === 'AgentSystem think');
    assert.equal(think.groups[1][0].resource.buffer, brain.buffers[0]);
    assert.equal(brain.buffers[0].size, Math.ceil(64 * definition.actDim * 4 / 16) * 16);
    agents.destroy();
    brain.destroy();
});

test('size: the adapter keeps its line, and mlpFromOnnx alone carries none of it', async () => {
    const bundle = async contents => (await build({ stdin: { contents, resolveDir: root, loader: 'ts' }, bundle: true, minify: true,
        format: 'esm', platform: 'browser', external: ['gl-matrix'], write: false, logLevel: 'silent' })).outputFiles[0];
    const entry = `${root}src/extras/inference/index.ts`;
    // The line holds the adapter and OrtBrain (phase 9.3), which imports nothing of the agents entry but types.
    const adapter = await bundle(`export { OrtRuntime, OrtModel, OrtBrain } from '${entry}';`), gzip = gzipSync(adapter.contents).length;
    assert.ok(adapter.text.includes('could not start its WebGPU provider') && adapter.text.includes('ortResults') && !adapter.text.includes('mlpFromOnnx'),
        'the bundle is the adapter and OrtBrain, without the reader');
    assert.ok(!/MlpBrain|affine layers|AgentSystem:/.test(adapter.text), 'OrtBrain copies no code of the agents entry');
    // 3,000 B until 2026-10-15, when the line of calls (U36) took it to 3,064 B; the user approved a new line.
    assert.ok(gzip <= 3100, `the adapter and OrtBrain are ${gzip} B min+gzip; their line is 3,100 B (the design's §12.4)`);
    const reader = await bundle(`export { mlpFromOnnx } from '${entry}';`);
    assert.ok(!/OrtRuntime|OrtModel|OrtBrain|anchor|ortResults/.test(reader.text), 'the reader\'s bundle holds the adapter');
    console.log(`size: OrtRuntime, OrtModel and OrtBrain are ${adapter.contents.length} B min, ${gzip} B min+gzip, of 3,100 B; mlpFromOnnx alone is ${gzipSync(reader.contents).length} B min+gzip`);
});

test('types: the onnxruntime-web namespace, as an app imports it, is an OrtLike', () => {
    const dir = mkdtempSync(join(root, 'test/.inference-ort-'));
    try {
        writeFileSync(join(dir, 'example.ts'), `import * as ort from 'onnxruntime-web/webgpu';
import { OrtRuntime, type OrtLike, type OrtModel } from 'null-graph/inference';
export const like: OrtLike = ort;
export async function start(adapter: GPUAdapter, bytes: Uint8Array, obs: GPUBuffer, act: GPUBuffer): Promise<OrtModel> {
    const runtime = await OrtRuntime.create(ort, { adapter });
    const model = await runtime.load(bytes, { capture: false, externalData: [{ path: 'model.onnx.data', data: 'model.onnx.data' }] });
    model.bind({ inputs: { obs: { buffer: obs, dims: [1000, 24] } }, outputs: { act: { buffer: act, dims: [1000, 2] } } });
    await model.run();
    return model;
}
`);
        writeFileSync(join(dir, 'tsconfig.json'), JSON.stringify({
            compilerOptions: { target: 'ES2022', module: 'ESNext', moduleResolution: 'Bundler', strict: true, noEmit: true,
                skipLibCheck: true, types: ['@webgpu/types'], lib: ['ES2022', 'DOM', 'DOM.Iterable'] },
            files: ['example.ts'],
        }));
        try {
            execFileSync(process.execPath, [resolve(root, 'node_modules/typescript/bin/tsc'), '--project', join(dir, 'tsconfig.json')], { cwd: root, stdio: 'pipe' });
        } catch (error) {
            assert.fail(`onnxruntime-web's namespace is not an OrtLike:\n${error.stdout}\n${error.stderr}`);
        }
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});
