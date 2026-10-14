// Step 9, phase 9.1 (docs/agents-ort-design.md §5, §6, §10): OrtRuntime and OrtModel, against a
// fake onnxruntime-web that records every call in order.
//
//   U32  the runtime makes ORT's device through an anchor session, created first and released
//        last; every session gets [{ name: 'webgpu' }], 'gpu-buffer' and the capture setting, and
//        the app cannot override them; bind checks names, types, ranks, fixed axes, named axes,
//        sizes and usage, once; one run at a time; destroy releases after a run in flight
//   U14  the refusals of the design's §11 that phase 9.1 builds fire and name the offender
//   size the adapter's line of 3,000 B; mlpFromOnnx alone carries none of it
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

test('U32: load sets the provider as an object, outputs on the GPU and graph capture, and the app cannot change them', async () => {
    const { ort, log, sessions } = fakeOrt();
    const runtime = await OrtRuntime.create(ort);
    const optionsOf = id => log.find(entry => entry[0] === 'create' && entry[1] === id)[2];
    const forced = capture => ({ executionProviders: [{ name: 'webgpu' }], preferredOutputLocation: 'gpu-buffer', enableGraphCapture: capture });

    const model = await runtime.load(MODEL);
    assert.deepEqual(optionsOf(1), forced(true), 'graph capture is on by default');
    assert.equal(model.capture, true);
    assert.equal(sessions[1].bytes, MODEL, 'the bytes go to ONNX Runtime as they are');
    assert.equal((await runtime.load(MODEL, { capture: false })).capture, false);
    assert.deepEqual(optionsOf(2), forced(false), 'capture: false turns it off');
    await runtime.load(MODEL, { capture: true, executionProviders: ['webgpu'], preferredOutputLocation: 'cpu', enableGraphCapture: false });
    assert.deepEqual(optionsOf(3), forced(true), 'other session options are not taken from the app');
    const externalData = [{ path: 'model.onnx.data', data: new Uint8Array(4) }];
    await runtime.load(MODEL, { externalData });
    assert.equal(optionsOf(4).externalData, externalData, 'externalData goes to ONNX Runtime as it is');
    await runtime.load(MODEL.buffer.slice(0));
    assert.deepEqual([...sessions[5].bytes], [...MODEL], 'an ArrayBuffer is taken as the same bytes');
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
    const all = () => ({ inputs: { ...bound().inputs, free: { buffer: buffer(5 * 3 * 4), dims: [5, 3] }, any: { buffer: buffer(4), dims: [1] } },
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
    refused(g => { g.inputs.obs = { buffer: buffer(N * 24 * 4 - 4), dims: [N, 24] }; },
        /^OrtModel: "obs" needs 96000 B with STORAGE usage; its buffer has 95996 B\.$/);
    refused(g => { g.outputs.act = { buffer: buffer(N * 2 * 4, GPUBufferUsage.COPY_SRC), dims: [N, 2] }; },
        /^OrtModel: "act" needs 8000 B with STORAGE usage; its buffer has 8000 B, without STORAGE\.$/);

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

test('U32: destroy releases each model after its run in flight, and the anchor last', async () => {
    const { ort, log, sessions } = fakeOrt();
    const runtime = await OrtRuntime.create(ort);
    const [a, b, c] = [await runtime.load(MODEL), await runtime.load(MODEL), await runtime.load(MODEL)];
    a.bind(bound());
    const running = a.run();
    await c.destroy();
    await c.destroy();
    const destroyed = runtime.destroy();
    await tick();
    const releases = () => log.filter(entry => entry[0] === 'release').map(entry => entry[1]);
    assert.deepEqual(releases(), [3, 2], 'the model with a run in flight waits for it; the anchor waits for every model');
    sessions[1].settle();
    await running;
    await destroyed;
    assert.deepEqual(releases(), [3, 2, 1, 0], 'each session released once, the anchor last');
    await runtime.destroy();
    assert.equal(releases().length, 4, 'destroy twice releases nothing twice');

    await assert.rejects(runtime.load(MODEL), refusal('OrtRuntime', /^OrtRuntime: used after destroy\(\)\.$/));
    assert.throws(() => b.bind(bound()), refusal('OrtModel', /^OrtModel: used after destroy\(\)\.$/));
    await assert.rejects(a.run(), refusal('OrtModel', /^OrtModel: used after destroy\(\)\.$/));
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

test('size: the adapter keeps its line, and mlpFromOnnx alone carries none of it', async () => {
    const bundle = async contents => (await build({ stdin: { contents, resolveDir: root, loader: 'ts' }, bundle: true, minify: true,
        format: 'esm', platform: 'browser', external: ['gl-matrix'], write: false, logLevel: 'silent' })).outputFiles[0];
    const entry = `${root}src/extras/inference/index.ts`;
    const adapter = await bundle(`export { OrtRuntime, OrtModel } from '${entry}';`), gzip = gzipSync(adapter.contents).length;
    assert.ok(adapter.text.includes('could not start its WebGPU provider') && !adapter.text.includes('mlpFromOnnx'), 'the bundle is the adapter, without the reader');
    assert.ok(gzip <= 3000, `the adapter is ${gzip} B min+gzip; its line is 3,000 B (the design's §12.4)`);
    const reader = await bundle(`export { mlpFromOnnx } from '${entry}';`);
    assert.ok(!/OrtRuntime|OrtModel|anchor/.test(reader.text), 'the reader\'s bundle holds the adapter');
    console.log(`size: OrtRuntime and OrtModel are ${adapter.contents.length} B min, ${gzip} B min+gzip, of 3,000 B; mlpFromOnnx alone is ${gzipSync(reader.contents).length} B min+gzip`);
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
