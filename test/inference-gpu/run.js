// GPU integration test for null-graph/inference's ONNX Runtime adapter: step 9, phase 9.1
// (docs/agents-ort-design.md §5, §6, §10).
//
//   A29  the anchor. ONNX Runtime makes the device and the engine adopts it. A model is loaded,
//        run and destroyed, and the device outlives it: it still answers a read-back and runs the
//        engine's stages. A captured model loaded after that release runs 100 times, right at runs
//        1, 2 and 100 (without an anchor, spike S5 saw the second run fail on AMD). A model loaded
//        with the default, without graph capture since phase 9.4, runs too. Only
//        runtime.destroy() ends the device.
//   A32  (phase 9.2) PyTorch's two-file exports, through the runtime with ONNX Runtime's own
//        externalData: their outputs equal PyTorch's, and the reader's from the same two files.
//        The other half, the reader's MlpBrain against ONNX Runtime, is A28 in test/onnx-gpu.
//   A5   (phase 9.3, the parent's) Herd & Wolves in lockstep, prey on OrtBrain and wolves on
//        MlpBrain, graph capture on and off: every live agent's actions are its own species'
//        network's; the same step again with MlpBrain for the prey gives OrtBrain's actions
//        within 1e-4.
//   A30  (phase 9.3) two lockstep runs of 120 steps from one seed, prey on OrtBrain: the agent
//        buffers are byte-identical.
//   A31  (phase 9.3) interactive swaps, rules → OrtBrain → MlpBrain → the same OrtBrain again:
//        each attach starts the brain from the actions agents have (its results seeded from act),
//        its runs land, and nothing errs.
//   A6   (phase 9.4, the parent's) 600 interactive frames with OrtBrain, 300 at the display's pace
//        and 300 as fast as they come: zero mapAsync outside the inspector (GPUBuffer's mapAsync
//        wrapped; the count is shown to see the inspector's calls and ONNX Runtime's own), zero
//        GPU errors, and the latency: by spike S2's marker method, how old each frame's actions
//        are. Never the frame's own; one frame at the display's pace.
//
// The runtime's own session is the only one alive across the release, so the device survives
// only if the anchor holds it.
//
// Two hosts, one file: `npm run test:gpu -- --only inference` (Node: Dawn as navigator.gpu, ORT's
// asyncify WASM with one thread), `npm run test:gpu:browser -- --only inference`.
import { NullGraph } from '../../dist/index.js';
import { AGENT_FLAGS_OFFSET, AGENT_FLAG_ALIVE, AGENT_FLOATS, AgentSystem, MlpBrain, Sensors, defineAgents, mlpForward, rand01 } from '../../dist/agents.js';
import { OrtBrain, OrtRuntime, mlpFromOnnx } from '../../dist/inference.js';
import { readBuffers } from '../../tools/agents/label-pass.mjs';
import { networkFromModel } from '../../tools/agents/model.mjs';
import { onnxFromNetwork } from '../../tools/agents/onnx.mjs';
import { createHerd, herdDefinition } from '../../tools/agents/scenario/herd-wolves.mjs';
import { harnessEnv } from '../gpu/harness-env.mjs';

const { log, check, fatal, finish, inBrowser } = harnessEnv('__INFERENCE_RESULTS__');
// NG_ORT_DIST=<folder> (Node only): another ONNX Runtime build's dist folder, to try a version
// before upgrading to it. Phase 9.4 ran 1.30.0 this way.
const otherOrt = !inBrowser && globalThis.process?.env?.NG_ORT_DIST;
const ORT_DIST = otherOrt ? (await import('node:url')).pathToFileURL(`${otherOrt.replace(/[\\/]+$/, '')}/`)
    : new URL('../../node_modules/onnxruntime-web/dist/', import.meta.url);

// ---------------------------------------------------------------------------
// ORT bootstrap, per host (as test/onnx-gpu/run.js)
// ---------------------------------------------------------------------------
if (!globalThis.self) globalThis.self = globalThis;   // ORT's bundle expects a worker-style global
let ort;
try {
    ort = await import(new URL('ort.webgpu.bundle.min.mjs', ORT_DIST).href);
} catch (e) {
    fatal(`onnxruntime-web could not be loaded (${e.message}) — it is a devDependency; run npm install.`);
    throw e;
}
ort.env.wasm.numThreads = 1;                          // no SharedArrayBuffer / cross-origin isolation
if (inBrowser) {
    ort.env.wasm.wasmPaths = ORT_DIST.href;
} else {
    const { readFile } = await import('node:fs/promises');
    const { fileURLToPath } = await import('node:url');
    const mjs = new URL('ort-wasm-simd-threaded.asyncify.mjs', ORT_DIST);
    const wasm = new URL('ort-wasm-simd-threaded.asyncify.wasm', ORT_DIST);
    ort.env.wasm.wasmPaths = { mjs: mjs.href, wasm: wasm.href };
    ort.env.wasm.wasmBinary = new Uint8Array(await readFile(fileURLToPath(wasm)));
}

// A run that fails can leave ONNX Runtime waiting for ever (spike S5). Nothing here waits longer.
const LIMIT_MS = 20000;
function within(promise, what) {
    let timer;
    const limit = new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`${what}: no answer after ${LIMIT_MS / 1000} s`)), LIMIT_MS); });
    return Promise.race([promise, limit]).finally(() => clearTimeout(timer));
}
const message = e => `${e?.name ?? 'Error'}: ${String(e?.message ?? e).split('\n')[0].slice(0, 160)}`;

// ---------------------------------------------------------------------------
// The runtime, and the engine on its device
// ---------------------------------------------------------------------------
const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
let runtime;
const started = performance.now();
try {
    runtime = await within(OrtRuntime.create(ort, { adapter }), 'OrtRuntime.create');
} catch (e) {
    fatal(`OrtRuntime.create failed: ${message(e)}`);
    throw e;
}
const createMs = performance.now() - started;
const device = runtime.device;
const engine = new NullGraph();
const init = await engine.init(null, { device, width: 8, height: 8 });
if (!init.success) {
    fatal(init.error);
    throw new Error(init.error);
}
log(`A29: ONNX Runtime ${ort.env.versions?.web ?? ''} · OrtRuntime.create took ${createMs.toFixed(0)} ms`);
check('A29: the engine adopts the device ONNX Runtime made (one device, not the engine\'s to destroy)',
    engine.device === device && engine.ctx.ownsDevice === false && (await ort.env.webgpu.device) === device,
    `same device: ${engine.device === device}, ownsDevice: ${engine.ctx.ownsDevice}`);

// ---------------------------------------------------------------------------
// One network, three models of it
// ---------------------------------------------------------------------------
const N = 4096, ROWS = 256, OBS = 24, ACT = 2;
const layer = (input, output, stream) => ({ input, output, activation: 'tanh',
    weights: Float32Array.from({ length: input * output }, (_, k) => (rand01(k, stream, 6) * 2 - 1) * Math.sqrt(6 / (input + output))),
    biases: Float32Array.from({ length: output }, (_, k) => (rand01(k, stream, 7) * 2 - 1) * 0.1) });
const network = [layer(OBS, 32, 1), layer(32, 32, 2), layer(32, ACT, 3)];
const bytes = onnxFromNetwork(network, { name: 'a29' });
const observations = stream => Float32Array.from({ length: N * OBS }, (_, k) => rand01(k, stream, 0) * 2 - 1);

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const obsData = observations(11);
const obs = device.createBuffer({ label: 'A29 obs', size: N * OBS * 4, usage: STORAGE });
device.queue.writeBuffer(obs, 0, obsData);
const act = name => device.createBuffer({ label: `A29 act ${name}`, size: N * ACT * 4, usage: STORAGE });
const bindings = output => ({ inputs: { obs: { buffer: obs, dims: [N, OBS] } }, outputs: { act: { buffer: output, dims: [N, ACT] } } });

// Test-only read-backs (forbidden in a frame loop, legal here).
async function read(buffer, floats) {
    const staging = device.createBuffer({ size: floats * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, floats * 4);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
}
/** The largest difference from mlpForward over the first ROWS agents. */
async function worst(output, data) {
    const got = await read(output, ROWS * ACT);
    let w = 0;
    for (let i = 0; i < ROWS; i++) {
        mlpForward(network, data.subarray(i * OBS, (i + 1) * OBS)).forEach((v, j) => { w = Math.max(w, Math.abs(v - got[i * ACT + j])); });
    }
    return w;
}

// Model A: loaded, run, destroyed. After it, only the anchor holds the device.
const actA = act('A');
try {
    const a = await within(runtime.load(bytes, { capture: true }), 'load A');
    a.bind(bindings(actA));
    await within(a.run(), 'A, run 1');
    const w = await worst(actA, obsData);
    check('A29: a captured model runs on the runtime\'s device and gives mlpForward\'s actions', w <= 1e-4, `worst ${w.toExponential(1)}`);
    await within(a.destroy(), 'A destroy');
} catch (e) {
    check('A29: a captured model runs on the runtime\'s device and gives mlpForward\'s actions', false, message(e));
}

let alive;
try {
    await within(read(actA, 4), 'a read-back after the release');
    alive = true;
} catch (e) {
    alive = message(e);
}
check('A29: with every model released, the anchor keeps the device: a read-back still works', alive === true, alive === true ? '' : alive);

// The engine's own work on the device, after the release.
const doubled = device.createBuffer({ label: 'A29 doubled', size: N * ACT * 4, usage: STORAGE });
const stage = engine.createComputeStage({
    name: 'A29 double', dispatchCount: [Math.ceil(N * ACT / 64), 1, 1],
    shaderCode: /* wgsl */ `
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
        @compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) g: vec3<u32>) {
            if (g.x < arrayLength(&a)) { o[g.x] = a[g.x] * 2.0; }
        }`,
});
engine.setComputePassBindGroup(stage, [{ binding: 0, resource: { buffer: actA } }, { binding: 1, resource: { buffer: doubled } }], 0);
let engineRan;
try {
    engine.render([stage]);
    const [twice, once] = [await read(doubled, ROWS * ACT), await read(actA, ROWS * ACT)];
    engineRan = twice.every((v, i) => v === once[i] * 2 && Number.isFinite(v)) ? true : 'a value differs';
} catch (e) {
    engineRan = message(e);
}
check('A29: and the engine\'s stages still run on it', engineRan === true, engineRan === true ? '' : engineRan);

// Model B: captured, loaded after the release. On AMD without an anchor, its second run failed.
const actB = act('B'), newObs = observations(12), checked = [];
try {
    const b = await within(runtime.load(bytes, { capture: true }), 'load B');
    b.bind(bindings(actB));
    for (let r = 1; r <= 100; r++) {
        if (r === 100) device.queue.writeBuffer(obs, 0, newObs);   // a captured session reads its bound buffer as it is now
        await within(b.run(), `B, run ${r}`);
        if (r <= 2 || r === 100) checked.push([r, await worst(actB, r === 100 ? newObs : obsData)]);
    }
    await within(b.destroy(), 'B destroy');
} catch (e) {
    checked.push([0, message(e)]);
}
check('A29: a captured model loaded after that release runs 100 times, right at runs 1, 2 and 100 (new observations before the 100th)',
    checked.length === 3 && checked.every(([, w]) => typeof w === 'number' && w <= 1e-4),
    checked.map(([r, w]) => typeof w === 'number' ? `run ${r}: ${w.toExponential(1)}` : w).join('; '));

// Model C: the default, without graph capture (since phase 9.4; the design's §14, item 6).
const actC = act('C');
try {
    const c = await within(runtime.load(bytes), 'load C');
    c.bind(bindings(actC));
    for (let r = 1; r <= 3; r++) await within(c.run(), `C, run ${r}`);
    const w = await worst(actC, newObs);
    check('A29: a model loaded without options, so without graph capture, runs too', w <= 1e-4 && !c.capture, `worst ${w.toExponential(1)}`);
} catch (e) {
    check('A29: a model loaded without options, so without graph capture, runs too', false, message(e));
}

// ---------------------------------------------------------------------------
// A32: two files, given to ONNX Runtime as its own externalData, through the runtime
// ---------------------------------------------------------------------------
const readJson = async path => {
    const url = new URL(path, import.meta.url);
    return inBrowser ? (await fetch(url)).json() : JSON.parse((await import('node:fs')).readFileSync(url, 'utf8'));
};
const base64 = text => Uint8Array.from(atob(text), c => c.charCodeAt(0));
const fixtures = (await readJson('../onnx-fixtures/fixtures.json')).cases;
const padded = bytes => Math.ceil(bytes / 16) * 16;     // OrtModel's rule: ONNX Runtime reads and writes 16 B at a time
// Under capture, ONNX Runtime 1.29 can fail a model from its second run on, depending on the model,
// the host and what ran before (phases 9.2 and 9.4; `npm run spike:ort-issues`). So `load` leaves
// capture off unless asked (decided with the user, the design's §14, item 6): A32 takes that
// default. Capture is probed after the checks, and reported.
const TWO_FILES = ['torch-outside', 'torch-outside-batch'];
async function twoFiles(name, capture) {
    const fixture = fixtures.find(c => c.name === name), bytes = base64(fixture.onnx);
    const externalData = Object.entries(fixture.externalData).map(([path, file]) => ({ path, data: base64(file.base64) }));
    const { inputs, outputs: { maker } } = fixture, width = inputs[0].length, outWidth = maker[0].length;
    const rows = fixture.holds.inputs[0].dims[0] === 1 ? 1 : inputs.length;   // a fixed batch of one runs row by row
    const model = await within(runtime.load(bytes, { externalData, ...(capture === undefined ? {} : { capture }) }), `load ${name}`);
    const input = device.createBuffer({ size: padded(rows * width * 4), usage: STORAGE }), output = device.createBuffer({ size: padded(rows * outWidth * 4), usage: STORAGE });
    try {
        model.bind({ inputs: { [model.inputs[0].name]: { buffer: input, dims: [rows, width] } },
            outputs: { [model.outputs[0].name]: { buffer: output, dims: [rows, outWidth] } } });
        const got = [];
        for (let r = 0; r < inputs.length; r += rows) {
            device.queue.writeBuffer(input, 0, Float32Array.from(inputs.slice(r, r + rows).flat()));
            await within(model.run(), `${name}, rows from ${r}`);
            got.push(...(await read(output, rows * outWidth)));
        }
        return { fixture, bytes, externalData, inputs, maker, outWidth, rows, got, captured: model.capture };
    } finally {
        await within(model.destroy(), `${name} destroy`);
    }
}
for (const name of TWO_FILES) {
    const label = `A32: ${name}, two files: through the runtime with ONNX Runtime's externalData, without graph capture by default, its outputs equal PyTorch's, and the reader's from the same files, within 1e-4`;
    try {
        const { bytes, externalData, inputs, maker, outWidth, rows, got, captured } = await twoFiles(name);
        if (captured) throw new Error('load captured a model it was not asked to');
        const { network } = mlpFromOnnx(bytes, { externalData });
        let againstMaker = 0, againstReader = 0;
        inputs.forEach((row, r) => {
            const reader = mlpForward(network, Float32Array.from(row));
            for (let j = 0; j < outWidth; j++) {
                againstMaker = Math.max(againstMaker, Math.abs(got[r * outWidth + j] - maker[r][j]));
                againstReader = Math.max(againstReader, Math.abs(got[r * outWidth + j] - reader[j]));
            }
        });
        check(label, againstMaker <= 1e-4 && againstReader <= 1e-4 && got.length === inputs.length * outWidth,
            `${inputs.length} rows${rows === 1 ? ', one run each' : ''}; against PyTorch ${againstMaker.toExponential(1)}, against mlpFromOnnx and mlpForward ${againstReader.toExponential(1)}`);
    } catch (e) {
        check(label, false, message(e));
    }
}
// The known failure, probed with capture asked for: reported, never checked.
for (const name of TWO_FILES) {
    try {
        await twoFiles(name, true);
        log(`   probe: ${name} under graph capture ran its runs this time`);
    } catch (e) {
        log(`   probe: ${name} under graph capture failed, as ONNX Runtime 1.29 can: ${message(e)}`);
    }
}

// ---------------------------------------------------------------------------
// A5, A30, A31: OrtBrain in an AgentSystem, Herd & Wolves, the committed networks
// ---------------------------------------------------------------------------
const networks = {};
for (const species of ['prey', 'wolf']) {
    const file = `../../tools/agents/models/herd-wolves/${species}.mlp.json`;
    networks[species] = networkFromModel(await readJson(file), herdDefinition(), file);
}
const onnxOf = species => onnxFromNetwork(networks[species], { name: `${species}-mlp` });
const f32 = words => new Float32Array(words.buffer, words.byteOffset, words.length);
const isAlive = (records, i) => (records[i * AGENT_FLOATS + AGENT_FLAGS_OFFSET] & AGENT_FLAG_ALIVE) !== 0;

for (const capture of [true, false]) {
    const label = `A5: prey on OrtBrain (graph capture ${capture ? 'on' : 'off'}), wolves on MlpBrain, in lockstep: every live agent's actions are its species' network's, and OrtBrain's equal MlpBrain's within 1e-4`;
    let herd, model, brain, wolves, prey;
    try {
        herd = createHerd(engine, { thinkEvery: 1 });     // every live agent thinks every step: each row compares
        const { agents, definition } = herd, { obsDim, actDim, capacity } = definition, PREY = definition.speciesIndex('prey');
        for (let s = 0; s < 60; s++) await agents.step();  // the rule drives first, to a spread-out herd
        model = await within(runtime.load(onnxOf('prey'), { capture }), 'load prey.onnx');
        brain = new OrtBrain(model);
        wolves = new MlpBrain(engine, networks.wolf);
        agents.setBrain(brain, { species: 'prey' });
        agents.setBrain(wolves, { species: 'wolf' });
        const [records, state] = await readBuffers(device, [agents.buffer, agents.state]);
        await within(agents.step(), 'a lockstep step with OrtBrain');
        const [obsWords, actWords] = await readBuffers(device, [agents.obs, agents.act]);
        const obs = f32(obsWords), ours = f32(actWords).slice();
        let against = { prey: 0, wolf: 0 }, live = 0, moved = 0;
        for (let i = 0; i < capacity; i++) {
            if (!isAlive(records, i)) continue;
            live++;
            const species = herd.speciesOf(i) === PREY ? 'prey' : 'wolf', want = mlpForward(networks[species], obs.subarray(i * obsDim, (i + 1) * obsDim));
            for (let j = 0; j < actDim; j++) against[species] = Math.max(against[species], Math.abs(want[j] - ours[i * actDim + j]));
            if (species === 'prey' && Math.abs(ours[i * actDim]) > 0.01) moved++;
        }
        // The same step again from the same state, the prey on MlpBrain.
        device.queue.writeBuffer(agents.buffer, 0, records);
        device.queue.writeBuffer(agents.state, 0, state);
        prey = new MlpBrain(engine, networks.prey);
        agents.setBrain(prey, { species: 'prey' });
        await agents.step();
        const [againWords] = await readBuffers(device, [agents.act]), again = f32(againWords);
        let apart = 0, wolfSame = true;
        for (let i = 0; i < capacity; i++) {
            if (!isAlive(records, i)) continue;
            for (let j = 0; j < actDim; j++) {
                const d = Math.abs(again[i * actDim + j] - ours[i * actDim + j]);
                if (herd.speciesOf(i) === PREY) apart = Math.max(apart, d);
                else if (d !== 0) wolfSame = false;
            }
        }
        check(label, against.prey <= 1e-4 && against.wolf <= 1e-4 && apart <= 1e-4 && wolfSame && moved > live / 8 && model.capture === capture,
            `${live} live agents; against mlpForward: prey ${against.prey.toExponential(1)}, wolves ${against.wolf.toExponential(1)}; ` +
            `OrtBrain against MlpBrain on the same step ${apart.toExponential(1)}; wolves the same both times: ${wolfSame}; ${moved} prey turn by more than 0.01`);
    } catch (e) {
        check(label, false, message(e));
    } finally {
        herd?.destroy();
        for (const b of [brain, wolves, prey]) b?.destroy();
        await model?.destroy();
    }
}

{
    const label = 'A30: two lockstep runs of 120 steps from one seed, prey on OrtBrain, end in byte-identical agent buffers';
    const run = async () => {
        const herd = createHerd(engine), model = await within(runtime.load(onnxOf('prey')), 'load prey.onnx'), brain = new OrtBrain(model);
        try {
            herd.agents.setBrain(brain, { species: 'prey' });
            for (let s = 0; s < 120; s++) await within(herd.agents.step(), `step ${s}`);
            const [words] = await readBuffers(device, [herd.agents.buffer]);
            return words.slice();
        } finally {
            herd.destroy();
            brain.destroy();
            await model.destroy();
        }
    };
    try {
        const [a, b] = [await run(), await run()];
        let differ = 0;
        for (let k = 0; k < a.length; k++) if (a[k] !== b[k]) differ++;
        check(label, a.length === b.length && differ === 0, `${a.length.toLocaleString('en-US')} words, ${differ} differ`);
    } catch (e) {
        check(label, false, message(e));
    }
}

{
    const label = 'A31: interactive swaps, rules → OrtBrain → MlpBrain → the same OrtBrain: each attach starts from the actions agents have, and its runs land';
    let herd, model, brain, mlp;
    try {
        herd = createHerd(engine);
        const { agents, definition } = herd, words = definition.capacity * definition.actDim;
        const frames = async n => {
            for (let f = 0; f < n; f++) {
                agents.advance(definition.dt);
                engine.render(agents.getComputeStages());
                await device.queue.onSubmittedWorkDone();
            }
        };
        // The results buffer right after an attach, before any frame: it must hold the actions agents have.
        const seed = async () => {
            const [act, results] = await readBuffers(device, [agents.act, brain.buffers[0]]);
            return { same: results.length >= words && act.every((w, k) => w === results[k]), results: results.slice(0, words) };
        };
        // Runs landed: prey rows of the results moved away from the seed. Wolves' rows the brain computes too.
        const landed = async from => {
            const [results] = await readBuffers(device, [brain.buffers[0]]);
            let moved = 0;
            for (let k = 0; k < words; k++) if (results[k] !== from[k]) moved++;
            return moved;
        };
        await frames(10);
        model = await within(runtime.load(onnxOf('prey')), 'load prey.onnx');
        brain = new OrtBrain(model);
        agents.setBrain(brain, { species: 'prey' });
        const first = await seed();
        await frames(30);
        const firstMoved = await landed(first.results);
        mlp = new MlpBrain(engine, networks.prey);
        agents.setBrain(mlp, { species: 'prey' });
        await frames(30);
        agents.setBrain(brain, { species: 'prey' });
        const again = await seed();                         // re-attached: seeded again, from the actions now
        await frames(30);
        const againMoved = await landed(again.results);
        check(label, first.same && again.same && firstMoved > words / 4 && againMoved > words / 4,
            `seeded at the first attach: ${first.same}; again at the second: ${again.same}; after 30 frames, ${firstMoved} and ${againMoved} of ${words} result words had moved from the seed`);
    } catch (e) {
        check(label, false, message(e));
    } finally {
        herd?.destroy();
        for (const b of [mlp, brain]) b?.destroy();
        await model?.destroy();
    }
}

// ---------------------------------------------------------------------------
// A6 (phase 9.4): the interactive schedule on the GPU, and its latency
// ---------------------------------------------------------------------------
// Spike S2's marker method, through AgentSystem and OrtBrain. A custom sensor writes MARK + the
// step's tick into every agent's first observation. A marker network (relu, relu, identity, one
// weight of 1 per layer; the shape of Herd & Wolves' 24 → 32 → 32 → 2) carries a first
// observation to the first action unchanged. A step stage, after act, records what the first and
// the last agent act on, and what sense wrote. So each frame tells which step's observations its
// actions came from; a value under MARK is a seed, actions no run gave. One lockstep step before
// the frames runs the model once: it compiles ONNX Runtime's pipelines, as S2's first run did.
const nextFrame = pace => (pace === 'fast' ? new Promise(r => setTimeout(r, 0))
    : inBrowser ? new Promise(r => requestAnimationFrame(r)) : new Promise(r => setTimeout(r, 1000 / 60)));
const median = values => { const s = [...values].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
{
    const AGENTS = 20000, FRAMES = 300, MARK = 1000, LAST = AGENTS - 1, PACES = ['display', 'fast'];
    const SLOTS = PACES.length * FRAMES + 2;           // ticks 0 (the lockstep step before the frames) to the one after them
    const definition = defineAgents({ capacity: AGENTS, world: { min: [-100, -100], max: [100, 100] }, cellSize: 4,
        species: [{ name: 'dot', maxSpeed: 1, maxTurn: 1, accel: 1, senseRadius: 4, radius: 0.2 }],
        sensors: [Sensors.custom({ name: 'marker', width: OBS, wgsl: `fn marker(i: u32, a: Agent) -> array<f32, ${OBS}> {
    var v: array<f32, ${OBS}>;
    v[0] = ${MARK}.0 + f32(state.tick);
    return v;
}` })] });
    const widths = [OBS, 32, 32, ACT];
    const markerNetwork = widths.slice(1).map((output, l) => {
        const weights = new Float32Array(widths[l] * output);
        weights[0] = 1;
        return { input: widths[l], output, activation: l === widths.length - 2 ? 'identity' : 'relu', weights, biases: new Float32Array(output) };
    });
    const seen = device.createBuffer({ label: 'A6 seen', size: SLOTS * 16, usage: STORAGE });
    device.queue.writeBuffer(seen, 0, new Float32Array(SLOTS * 4).fill(-1));
    const proto = Object.getPrototypeOf(seen), mapAsync = proto.mapAsync;
    const maps = { control: 0, inspector: 0, other: [] }, errors = [], inspected = [], paces = [], runs = [];
    const onError = ev => errors.push(String(ev.error?.message ?? ev.error).split('\n')[0]);
    let agents, model, brain;
    try {
        agents = new AgentSystem(engine, definition);
        model = await within(runtime.load(onnxFromNetwork(markerNetwork, { name: 'a6-marker' })), 'load the marker network');
        brain = new OrtBrain(model);
        agents.setBrain(brain);
        agents.addStepStage({ name: 'probe', buffers: [seen], wgsl: `
@group(1) @binding(0) var<storage, read_write> seen: array<f32>;
fn step(i: u32, tick: u32) {
    if (i != 0u && i != ${LAST}u) { return; }
    let at = tick * 4u + select(0u, 2u, i == ${LAST}u);
    if (at + 1u < arrayLength(&seen)) {
        seen[at] = act[i * ACT_DIM];
        seen[at + 1u] = obs[i * OBS_DIM];
    }
}` });
        for (let i = 0; i < AGENTS; i++) {
            agents.spawn(i, { species: 'dot', position: [(rand01(i, 6, 0) * 2 - 1) * 99, 0, (rand01(i, 6, 1) * 2 - 1) * 99],
                heading: rand01(i, 6, 2) * Math.PI * 2 - Math.PI });
        }
        // Each run, with the frame that started it; frame 0 is the lockstep step before the frames.
        // The frame's tick is its number, so a run started in frame f should read marker MARK + f.
        let frame = 0;
        const run = brain.run.bind(brain);
        brain.run = () => {
            const record = { frame, t0: performance.now() };
            runs.push(record);
            const pending = run();
            pending.then(() => { record.ms = performance.now() - record.t0; }, e => { record.failed = message(e); });
            return pending;
        };
        await within(agents.step(), 'the lockstep step before the frames');

        let counting = 'control';
        proto.mapAsync = function (...args) {
            if (counting === 'control') maps.control++;
            else if (this.label === 'AgentSystem inspect staging') maps.inspector++;
            else maps.other.push(this.label || `a buffer without a label, usage ${this.usage}`);
            return mapAsync.apply(this, args);
        };
        device.addEventListener('uncapturederror', onError);
        try {
            // The control: ONNX Runtime reads an output back to the CPU with mapAsync, and the count sees it.
            const control = await within(ort.InferenceSession.create(bytes, { executionProviders: [{ name: 'webgpu' }] }), 'the control session');
            await within(control.run({ obs: new ort.Tensor('float32', obsData.slice(0, 8 * OBS), [8, OBS]) }), 'the control run');
            await within(control.release(), 'the control session\'s release');
            counting = 'frames';
            for (const pace of PACES) {
                const from = frame + 1, frameMs = [];
                let last = performance.now();
                for (let k = 0; k < FRAMES; k++) {
                    frame++;
                    agents.advance(definition.dt);               // one step; starts a run when none is in flight
                    engine.render(agents.getComputeStages());
                    if (k % 100 === 50) inspected.push(agents.inspect(k));    // the inspector reads back, off the frame path
                    await nextFrame(pace);
                    const now = performance.now();
                    frameMs.push(now - last);
                    last = now;
                }
                paces.push({ pace, from, to: frame, frameMs });
            }
        } finally {
            proto.mapAsync = mapAsync;
            device.removeEventListener('uncapturederror', onError);
        }
        frame++;
        await within(agents.step(), 'the lockstep step after the frames');    // it waits for the run in flight first
        const snapshots = await within(Promise.all(inspected), 'the inspector\'s read-backs');
        const words = await read(seen, SLOTS * 4);
        const slot = g => ({ act: words[g * 4], torn: words[g * 4] !== words[g * 4 + 2],
            sensed: words[g * 4 + 1] === MARK + g && words[g * 4 + 3] === MARK + g });

        // One run is in flight at a time, and the frame after one lands acts on it: so the markers,
        // in the order frames first acted on them, are the runs in the order they started.
        const firstSeen = new Set();
        for (let g = 1; g < frame; g++) if (slot(g).act >= MARK) firstSeen.add(slot(g).act - MARK);
        [...firstSeen].forEach((marker, k) => { if (runs[k]) runs[k].read = marker - runs[k].frame; });

        const outside = maps.other.length;
        check(`A6: across ${PACES.length * FRAMES} interactive frames with OrtBrain, zero mapAsync outside the inspector and zero GPU errors; the count saw the inspector's ${inspected.length} read-backs and ONNX Runtime's own`,
            outside === 0 && errors.length === 0 && inspected.length > 0 && maps.inspector === inspected.length && maps.control > 0 && snapshots.every(s => s !== null),
            `mapAsync: ${outside} outside the inspector${outside ? ` (${[...new Set(maps.other)].join(', ')})` : ''}, ${maps.inspector} by the inspector, ` +
            `${maps.control} by ONNX Runtime reading an output to the CPU (the control); GPU errors: ${errors.length}${errors.length ? ` (${errors[0]})` : ''}`);
        for (const p of paces) {
            const ages = new Map();
            let own = 0, seeds = 0, unsensed = 0, torn = 0;
            for (let g = p.from; g <= p.to; g++) {
                const s = slot(g);
                if (!s.sensed) unsensed++;
                if (s.torn) torn++;
                if (s.act < MARK) { seeds++; continue; }
                const age = g - (s.act - MARK);
                if (age <= 0) own++;
                ages.set(age, (ages.get(age) ?? 0) + 1);
            }
            const started = runs.filter(r => r.frame >= p.from && r.frame <= p.to), paired = started.filter(r => r.read !== undefined);
            const readOwn = paired.filter(r => r.read === 0).length, failed = started.filter(r => r.failed);
            const runMs = started.filter(r => r.ms !== undefined).map(r => r.ms), oneOld = ages.get(1) ?? 0;
            const name = p.pace === 'display' ? `at the display's pace (${inBrowser ? 'requestAnimationFrame' : 'a 60 Hz timer'})` : 'as fast as they come';
            log(`   A6 ${name}: a frame took ${median(p.frameMs).toFixed(1)} ms (median); ${started.length} runs in ${FRAMES} frames` +
                (runMs.length ? `; run() resolved in ${median(runMs).toFixed(2)} ms (median), ${Math.max(...runMs).toFixed(2)} ms at most` : ''));
            // One frame old: at the display's pace, 90% of frames at least; as fast as frames come, more
            // than half (spike S2's worst was 293 of 300), so that a schedule that stalls still fails.
            const typical = p.pace === 'display' ? 0.9 * FRAMES : FRAMES / 2 + 1;
            check(`A6 ${name}: no frame acts on its own or a later frame's observations, nor on a seed; the first and the last agent act on the same run; every run read the observations of the frame that started it; ` +
                (p.pace === 'display' ? '90% of frames or more act on actions one frame old' : 'more than half act on actions one frame old'),
                own === 0 && seeds === 0 && unsensed === 0 && torn === 0 && failed.length === 0 && readOwn === paired.length &&
                paired.length >= started.length - 1 && oneOld >= typical,
                `frames by the age of their actions, in frames: {${[...ages].sort((a, b) => a[0] - b[0]).map(([age, n]) => `${age}: ${n}`).join(', ')}}; ` +
                `${own} on their own or a later frame's, ${seeds} on a seed, ${unsensed} not sensed, ${torn} torn; ` +
                `runs that read the observations of the frame that started them: ${readOwn} of the ${paired.length} that frames acted on` +
                (failed.length ? `; ${failed.length} runs failed, the first: ${failed[0].failed}` : ''));
        }
    } catch (e) {
        check('A6: 600 interactive frames with OrtBrain ran to their end', false, message(e));
    } finally {
        agents?.destroy();
        brain?.destroy();
        await model?.destroy();
        seen.destroy();
    }
}

// ---------------------------------------------------------------------------
// A33 (phase 9.5): brains prepared in the background, with frames running
// ---------------------------------------------------------------------------
// Two Herd & Wolves systems, frames at the display's pace. In the first, setBrain compiles an
// MlpBrain's think pass inside a frame, then an OrtBrain's: those frames are the stall prepareBrain
// exists to remove. In the second, prepareBrain compiles the same two passes while frames go on,
// and the longest frame meanwhile is reported beside the stall. Then setBrain of each prepared
// brain, and of the MlpBrain again after it was detached, must take under 2 ms, and the prey must
// act as their network says. A pipeline cache makes a second compile of the same WGSL free, so each
// system has a thinkEvery no other test here uses (4 and 5): the WGSL of its passes is new.
{
    const frame = async (agents, before) => {
        const t0 = performance.now();
        before?.();
        agents.advance(agents.definition.dt);
        engine.render(agents.getComputeStages());
        await nextFrame('display');
        return performance.now() - t0;
    };
    const frames = async (agents, n) => { for (let f = 0; f < n; f++) await frame(agents); };
    /** prepareBrain called inside a frame, then frames until it settles: the longest of them, and how long it took. */
    const preparing = async (agents, brain) => {
        let promise, done, t0;
        const times = [await frame(agents, () => {
            t0 = performance.now();
            promise = agents.prepareBrain(brain);
            promise.then(() => { done = performance.now(); }, () => { done = performance.now(); });
        })];
        while (done === undefined && times.length < 2000) times.push(await frame(agents));
        await promise;
        return { longest: Math.max(...times), frames: times.length, ms: done - t0 };
    };
    /** After thinkEvery lockstep steps every agent has thought: each prey must act as `network` says on its observations. */
    const preyAs = async (herd, network) => {
        for (let s = 0; s < herd.options.thinkEvery; s++) await within(herd.agents.step(), 'a lockstep step');
        const { agents, definition: { obsDim, actDim, capacity } } = herd, PREY = herd.definition.speciesIndex('prey');
        const [records, obsWords, actWords] = await readBuffers(device, [agents.buffer, agents.obs, agents.act]);
        const obs = f32(obsWords), act = f32(actWords);
        let worst = 0, rows = 0;
        for (let i = 0; i < capacity; i++) {
            if (!isAlive(records, i) || herd.speciesOf(i) !== PREY) continue;
            rows++;
            mlpForward(network, obs.subarray(i * obsDim, (i + 1) * obsDim)).forEach((v, j) => { worst = Math.max(worst, Math.abs(v - act[i * actDim + j])); });
        }
        return { worst, rows };
    };
    const timed = action => { const t0 = performance.now(); action(); return performance.now() - t0; };
    const ms = v => `${v < 10 ? v.toFixed(2) : v.toFixed(0)} ms`;
    const label = 'A33: brains prepared while frames run; setBrain of a prepared MlpBrain and OrtBrain, and of the MlpBrain again after a detach, takes under 2 ms; the prey act as their network says; no GPU error';
    const errors = [], onError = ev => errors.push(String(ev.error?.message ?? ev.error).split('\n')[0]);
    const herds = [], brains = [], models = [];
    device.addEventListener('uncapturederror', onError);
    try {
        const [compiling, prepared] = [4, 5].map(thinkEvery => { const herd = createHerd(engine, { thinkEvery }); herds.push(herd); return herd; });
        for (let k = 0; k < 2; k++) models.push(await within(runtime.load(onnxOf('prey')), 'load prey.onnx'));
        const [mlpA, mlpB] = [0, 1].map(() => new MlpBrain(engine, networks.prey));
        const [ortA, ortB] = models.map(model => new OrtBrain(model));
        brains.push(mlpA, mlpB, ortA, ortB);
        const prey = { species: 'prey' };

        // setBrain compiling, inside a frame.
        await frames(compiling.agents, 20);
        const stallMlp = await frame(compiling.agents, () => compiling.agents.setBrain(mlpA, prey));
        await frames(compiling.agents, 10);
        const stallOrt = await frame(compiling.agents, () => compiling.agents.setBrain(ortA, prey));
        await frames(compiling.agents, 10);

        // prepareBrain while frames go on, then setBrain of the prepared brains.
        await frames(prepared.agents, 20);
        const prepMlp = await preparing(prepared.agents, mlpB);
        const setMlp = timed(() => prepared.agents.setBrain(mlpB, prey));
        const asMlp = await preyAs(prepared, networks.prey);
        const prepOrt = await preparing(prepared.agents, ortB);
        const setOrt = timed(() => prepared.agents.setBrain(ortB, prey));
        await frames(prepared.agents, 10);
        const asOrt = await preyAs(prepared, networks.prey);
        const back = timed(() => prepared.agents.setBrain(mlpB, prey));
        await frames(prepared.agents, 10);
        await device.queue.onSubmittedWorkDone();

        log(`   A33 setBrain compiling, the frame it ran in: MlpBrain ${ms(stallMlp)}, OrtBrain ${ms(stallOrt)}`);
        log(`   A33 prepareBrain while frames ran: MlpBrain ${ms(prepMlp.ms)} over ${prepMlp.frames} frames, the longest ${ms(prepMlp.longest)}; ` +
            `OrtBrain ${ms(prepOrt.ms)} over ${prepOrt.frames} frames, the longest ${ms(prepOrt.longest)}`);
        check(label, setMlp < 2 && setOrt < 2 && back < 2 && asMlp.worst <= 1e-4 && asOrt.worst <= 1e-4 && asMlp.rows > 0 && errors.length === 0,
            `setBrain of the prepared MlpBrain ${ms(setMlp)}, OrtBrain ${ms(setOrt)}, the MlpBrain again ${ms(back)}; ` +
            `${asMlp.rows} prey against their network: ${asMlp.worst.toExponential(1)} on MlpBrain, ${asOrt.worst.toExponential(1)} on OrtBrain; ` +
            `GPU errors: ${errors.length}${errors.length ? ` (${errors[0]})` : ''}`);
    } catch (e) {
        check(label, false, message(e));
    } finally {
        device.removeEventListener('uncapturederror', onError);
        for (const herd of herds) herd.destroy();
        for (const brain of brains) brain.destroy();
        for (const model of models) await model.destroy();
    }
}

// Only the runtime ends the device: its destroy releases model C, then the anchor. What ONNX
// Runtime then does with its device is its own; it is reported, not checked.
try {
    await within(runtime.destroy(), 'runtime destroy');
    let after;
    try { await within(read(actA, 4), 'a read-back after runtime.destroy()'); after = 'alive'; } catch (e) { after = `gone (${message(e)})`; }
    log(`   after runtime.destroy(), the device is ${after}`);
} catch (e) {
    check('A29: runtime.destroy() releases every model and the anchor', false, message(e));
}
await finish('A29: the anchor holds ONNX Runtime\'s device across model swaps.');
log('inference done');
