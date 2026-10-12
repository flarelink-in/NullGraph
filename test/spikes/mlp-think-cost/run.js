// Step 6, phase 6.3 (docs/agents-mlp-design.md §7): what the production MlpBrain think stage costs.
//
// A measurement, not a test suite: it is not registered in test/gpu/suites.mjs, and its numbers are
// recorded in agents-mlp-design.md. Spike S3 timed a bare 24→32→32→2 kernel; this times the shader
// AgentSystem actually generates (buildAgentThinkWGSL around brain.wgsl): the record read, the
// alive/cadence gate and the obs()/setAct()/setReason() accessors included, bound exactly as the
// think stage binds it. Every kernel's output is checked against mlpForward before it is timed, on
// every adapter, so the numbers come with a correctness check on GPUs the A4 suite did not reach.
//
//   Node + Dawn, on every adapter Dawn offers:  npm run spike:mlp-think
//   Chrome:  npm run drive -- --serve . test/spikes/mlp-think-cost/index.html --wait-for "MLP think done" --timeout 300
//
// Kernels, all with one invocation per agent and workgroup 64:
//   rule floor: a RuleBrain that copies two observations into the two actions and writes a reason —
//     the stage's own cost: gate, record read, accessors.
//   MLP 24→32→32→2 tanh: S3's network as an MlpBrain, every agent thinking.
//   the same with thinkEvery 3: the production block stagger, a third of the blocks thinking.
//   MLP 128→64→64→64→16: the largest network MlpBrain accepts.
//   rows (phase 7.2, agents-species-brains-design.md §7): a prey and a wolf network as rows of one
//     table (MlpBrain.bySpecies). Species contiguous is Herd & Wolves' layout (wolves first, 1 in
//     16); species alternating is the worst case, every wavefront reading both rows. Each agent is
//     checked against its own species' network.
//   rows or passes at 8 and 16 species (phase 7.2's decision): N species in contiguous blocks, each its
//     own network, as rows of one table in one pass, or as one pass per species.
import { harnessEnv } from '../../gpu/harness-env.mjs';
import * as A from '../../../dist/agents.js';

const { log, check, fatal, finish, inBrowser } = harnessEnv('__MLP_THINK_RESULTS__');

const COUNTS = [1_000, 20_000, 100_000];
const CHECK_N = 20_000;
const TRIALS = 5;
const ROUNDS = 2;        // the whole matrix twice; each cell keeps its lower median
const WARMUP_MS = 1000;  // a GPU idles at low clocks and boosts only under sustained load
const PASS_NS = 5e6;     // ≥ 5 ms of work per timed pass: Chrome may quantize timestamps to 100 µs
const TOL = 1e-5;        // A4's |gpu − cpu| ≤ 1e-5 + 1e-5·|cpu|
const SENTINEL = 7;

let seed = 0x6a3;
const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
const network = (...layers) => layers.map(([input, output, activation]) => ({ input, output, activation,
    weights: Float32Array.from({ length: input * output }, () => (random() * 2 - 1) * Math.sqrt(3 / input)),
    biases: Float32Array.from({ length: output }, () => (random() * 2 - 1) * 0.1) }));

const herdNet = network([24, 32, 'tanh'], [32, 32, 'tanh'], [32, 2, 'tanh']);
const maxNet = network([128, 64, 'relu'], [64, 64, 'tanh'], [64, 64, 'identity'], [64, 16, 'sigmoid']);
const wolfNet = network([24, 32, 'tanh'], [32, 32, 'tanh'], [32, 2, 'tanh']);
// N species for the rows-or-passes decision; the first two are prey and wolf so the herd's sensors apply.
const MANY = [8, 16];
const manyNames = Array.from({ length: 16 }, (_, k) => k === 0 ? 'prey' : k === 1 ? 'wolf' : `s${k}`);
const manyNets = [herdNet, wolfNet, ...Array.from({ length: 14 }, () => network([24, 32, 'tanh'], [32, 32, 'tanh'], [32, 2, 'tanh']))];
const species = [
    { name: 'prey', maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 8, radius: 0.4 },
    { name: 'wolf', maxSpeed: 4, maxTurn: 3, accel: 5, senseRadius: 8, radius: 0.6 },
];
// Herd & Wolves' 24 observations and 2 steering actions; the widest one fills 128 → 16 with custom slots.
const herdDef = capacity => A.defineAgents({ capacity, world: { min: [-200, -200], max: [200, 200] }, cellSize: 8, species,
    sensors: [A.Sensors.self(), A.Sensors.nearest({ name: 'wolf', species: ['wolf'], k: 1 }),
        A.Sensors.nearest({ name: 'prey', species: ['prey'], k: 3 }), A.Sensors.bounds()] });
const manyDef = N => capacity => A.defineAgents({ capacity, world: { min: [-200, -200], max: [200, 200] }, cellSize: 8,
    species: manyNames.slice(0, N).map((name, k) => ({ ...species[k === 1 ? 1 : 0], name })),
    sensors: [A.Sensors.self(), A.Sensors.nearest({ name: 'wolf', species: ['wolf'], k: 1 }),
        A.Sensors.nearest({ name: 'prey', species: ['prey'], k: 3 }), A.Sensors.bounds()] });
const wideDef = capacity => A.defineAgents({ capacity, world: { min: [-200, -200], max: [200, 200] }, cellSize: 8, species,
    sensors: [A.Sensors.custom({ name: 'wide', width: 128, wgsl: 'fn wide(i: u32, a: Agent) -> array<f32, 128> { var v: array<f32, 128>; return v; }' })],
    actions: A.Actions.custom({ name: 'wideAct', width: 16, wgsl: 'fn wideAct(i: u32) {}' }) });
const RULE = `fn think(i: u32, species: u32, tick: u32) {
    setAct(i, ACT_TURN, obs(i, 2u));
    setAct(i, ACT_THROTTLE, obs(i, 3u));
    setReason(i, 1u);
}`;
const FLOOR = { name: 'rule floor (2 reads, 2 writes)', def: herdDef, net: null, thinkEvery: 1 };
const HERD = { name: 'MLP 24→32→32→2 tanh', def: herdDef, net: herdNet, thinkEvery: 1 };
const WIDE = { name: 'MLP 128→64→64→64→16', def: wideDef, net: maxNet, thinkEvery: 1 };
const KERNELS = [
    FLOOR,
    HERD,
    { name: 'MLP 24→32→32→2, thinkEvery 3', def: herdDef, net: herdNet, thinkEvery: 3 },
    { name: 'rows: 2 nets, species contiguous', def: herdDef, net: herdNet, rows: wolfNet, layout: 'contiguous', thinkEvery: 1 },
    { name: 'rows: 2 nets, species alternating', def: herdDef, net: herdNet, rows: wolfNet, layout: 'alternating', thinkEvery: 1 },
    { name: 'rows: 2 nets, one species only', def: herdDef, net: herdNet, rows: wolfNet, layout: 'prey', thinkEvery: 1 },
    { name: 'switch: 2 nets, species contiguous', def: herdDef, net: herdNet, rows: wolfNet, layout: 'contiguous', thinkEvery: 1, constRows: true },
    { name: 'switch: 2 nets, species alternating', def: herdDef, net: herdNet, rows: wolfNet, layout: 'alternating', thinkEvery: 1, constRows: true },
    { name: 'dispatch per species, contiguous', def: herdDef, net: herdNet, rows: wolfNet, layout: 'contiguous', thinkEvery: 1, perSpecies: true },
    { name: 'dispatch per species, alternating', def: herdDef, net: herdNet, rows: wolfNet, layout: 'alternating', thinkEvery: 1, perSpecies: true },
    ...MANY.flatMap(N => [
        { name: `rows: ${N} species, contiguous`, many: N, net: herdNet, thinkEvery: 1 },
        { name: `passes: ${N} species, contiguous`, many: N, net: herdNet, thinkEvery: 1, perSpecies: true },
    ]),
    WIDE,
];
// The same table read through a switch whose branches pass a constant row offset, so each branch's
// weight addresses are uniform: tests whether the per-lane `row` term is what rows cost.
function constRowsWGSL(net, species) {
    const quads = A.mlpLayout(net).floats / 4;
    const body = A.buildMlpThinkWGSL(net)
        .replace('fn think(i: u32, species: u32, tick: u32) {', 'fn mlpRow(i: u32, base: u32) {')
        .replaceAll('mlpWeights[', 'mlpWeights[base + ');
    return `${body}
fn think(i: u32, species: u32, tick: u32) {
    switch species {
${species.map((name, r) => `        case SPECIES_${name.toUpperCase()}: { mlpRow(i, ${r * quads}u); }`).join('\n')}
        default: {}
    }
}`;
}
// Which species agent i is, per layout: Herd & Wolves' 1 in 16 wolves, interleaved or first.
const speciesAt = (layout, i, n) => layout === 'prey' ? 'prey' : layout === 'contiguous' ? (i < n / 16 ? 'wolf' : 'prey')
    : layout === 'alternating' ? (i % 2 ? 'wolf' : 'prey') : (i % 16 === 0 ? 'wolf' : 'prey');
const thinks = (k, i) => k.thinkEvery === 1 || Math.floor(i / 64) % k.thinkEvery === 0;   // tick 0, alive, age > 0

const median = xs => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
const fmt = n => n.toLocaleString('en-US');

async function runAdapter(adapter, name) {
    const timestamps = adapter.features.has('timestamp-query');
    const device = await adapter.requestDevice({ requiredFeatures: timestamps ? ['timestamp-query'] : [] });
    const made = [];
    const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const buffer = (bytes, usage, data) => {
        const b = device.createBuffer({ size: Math.max(16, bytes), usage });
        made.push(b);
        if (data) device.queue.writeBuffer(b, 0, data);
        return b;
    };
    async function readF32(src, floats) {
        const dst = device.createBuffer({ size: floats * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(src, 0, dst, 0, floats * 4);
        device.queue.submit([enc.finish()]);
        await dst.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(dst.getMappedRange().slice(0));
        dst.unmap();
        dst.destroy();
        return out;
    }
    const ts = timestamps ? {
        set: device.createQuerySet({ type: 'timestamp', count: 2 }),
        resolve: buffer(16, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC),
        read: buffer(16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ),
    } : null;
    const rawDeltas = [];
    const engineLike = { device };   // MlpBrain needs only its engine's device
    const brains = new Map([[herdNet, new A.MlpBrain(engineLike, herdNet)], [maxNet, new A.MlpBrain(engineLike, maxNet)]]);
    const rowsBrain = A.MlpBrain.bySpecies(engineLike, herdDef(64), { prey: herdNet, wolf: wolfNet });
    const manyBrains = new Map(MANY.map(N => [N, A.MlpBrain.bySpecies(engineLike, manyDef(N)(64),
        Object.fromEntries(manyNames.slice(0, N).map((name, r) => [name, manyNets[r]])))]));
    // Per-species passes as AgentSystem builds them (phase 7.3): one single-network MlpBrain per
    // species, each pass gated by its 16-byte species mask. Every other kernel passes every species.
    const speciesBrains = manyNets.map(net => new A.MlpBrain(engineLike, net));
    const mask = bits => buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, new Uint32Array([bits, 0, 0, 0]));
    const everySpecies = mask(0xffff), speciesMasks = manyNets.map((_, r) => mask(1 << r));

    // One system's worth of think-stage bindings for capacity n: every agent alive, age 1 (so the
    // cadence, not the newborn rule, decides), tick 0, observations in [-1, 1).
    const setups = new Map();
    function setup(k, n) {
        const key = `${k.many ? `many${k.many}` : k.def === herdDef ? 'herd' : 'wide'}:${k.layout ?? 'herd'}:${n}`;
        if (setups.has(key)) return setups.get(key);
        const def = k.many ? manyDef(k.many)(n) : k.def(n), records = new A.AgentRecords(def);
        for (let i = 0; i < n; i++) records.spawn(i, { position: [0, 0, 0], age: 1,
            species: k.many ? manyNames[Math.floor((i * k.many) / n)] : speciesAt(k.layout, i, n) });
        const obs = Float32Array.from({ length: n * def.obsDim }, () => random() * 2 - 1);
        const speciesData = A.packSpecies(def);
        const s = { def, obs, buffers: [
            buffer(records.data.byteLength, STORAGE, records.data),
            buffer(obs.byteLength, STORAGE, obs),
            buffer(16, STORAGE, new Uint32Array([0, 0, 0, 0])),
            buffer(n * def.actDim * 4, STORAGE),
            buffer(n * 4, STORAGE),
            buffer(speciesData.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, speciesData),
        ] };
        setups.set(key, s);
        return s;
    }
    async function pipeline(k, n) {
        const { def, buffers } = setup(k, n);
        const brain = k.many ? manyBrains.get(k.many) : k.rows ? rowsBrain : k.net ? brains.get(k.net) : new A.RuleBrain(RULE);
        // [WGSL, group-1 weights or null, species mask] per pass.
        const passes = k.perSpecies
            ? speciesBrains.slice(0, k.many ?? 2).map((b, r) => [b.wgsl, b.buffers[0], speciesMasks[r]])
            : [[k.constRows ? constRowsWGSL(herdNet, ['prey', 'wolf']) : brain.wgsl, k.net ? brain.buffers[0] : null, everySpecies]];
        const t0 = performance.now(), parts = [];
        for (const [wgsl, weights, gate] of passes) {
            const code = A.buildAgentThinkWGSL(def, wgsl, k.thinkEvery);
            const p = await device.createComputePipelineAsync({ label: k.name, layout: 'auto',
                compute: { module: device.createShaderModule({ label: k.name, code }), entryPoint: 'cs_main' } });
            const binds = [device.createBindGroup({ layout: p.getBindGroupLayout(0),
                entries: [...buffers, gate].map((b, binding) => ({ binding, resource: { buffer: b } })) })];
            if (weights) binds.push(device.createBindGroup({ layout: p.getBindGroupLayout(1),
                entries: [{ binding: 0, resource: { buffer: weights } }] }));
            parts.push({ p, binds });
        }
        return { parts, ms: performance.now() - t0, groups: Math.ceil(n / 64) };
    }
    async function timedPass(c, repeats) {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass(ts
            ? { timestampWrites: { querySet: ts.set, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
        for (let r = 0; r < repeats; r++) {
            for (const part of c.parts) {
                pass.setPipeline(part.p);
                part.binds.forEach((b, g) => pass.setBindGroup(g, b));
                pass.dispatchWorkgroups(c.groups);
            }
        }
        pass.end();
        if (ts) {
            enc.resolveQuerySet(ts.set, 0, 2, ts.resolve, 0);
            enc.copyBufferToBuffer(ts.resolve, 0, ts.read, 0, 16);
        }
        const t0 = performance.now();
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
        const wallNs = (performance.now() - t0) * 1e6;
        let gpuNs = null;
        if (ts) {
            await ts.read.mapAsync(GPUMapMode.READ);
            const [begin, end] = new BigUint64Array(ts.read.getMappedRange().slice(0));
            ts.read.unmap();
            if (end > begin) gpuNs = Number(end - begin);
        }
        return { gpuNs, wallNs };
    }
    async function measure(c) {
        const probe = await timedPass(c, 4);
        const per = (probe.gpuNs ?? probe.wallNs) / 4;
        const repeats = Math.max(8, Math.min(4000, Math.ceil(PASS_NS / Math.max(per, 250))));
        const gpu = [], wall = [];
        for (let t = 0; t < TRIALS; t++) {
            const r = await timedPass(c, repeats);
            if (r.gpuNs !== null) { gpu.push(r.gpuNs / repeats); rawDeltas.push(r.gpuNs); }
            wall.push(r.wallNs / repeats);
        }
        return gpu.length === TRIALS ? median(gpu) / 1e3 : median(wall) / 1e3;
    }
    // Every thinking row against the CPU (mlpForward, or the copied observations); skipped rows keep the sentinel.
    async function verify(k, c) {
        const s = setup(k, CHECK_N), { def } = s, act = s.buffers[3];
        device.queue.writeBuffer(act, 0, new Float32Array(CHECK_N * def.actDim).fill(SENTINEL));
        await timedPass(c, 1);
        const got = await readF32(act, CHECK_N * def.actDim);
        const input = new Float32Array(def.obsDim), out = new Float32Array(def.actDim);
        let maxAbs = 0, bad = 0, touched = 0;
        for (let i = 0; i < CHECK_N; i++) {
            const row = got.subarray(i * def.actDim, (i + 1) * def.actDim);
            if (!thinks(k, i)) { if (row.some(v => v !== SENTINEL)) touched++; continue; }
            input.set(s.obs.subarray(i * def.obsDim, (i + 1) * def.obsDim));
            if (k.many) A.mlpForward(manyNets[Math.floor((i * k.many) / CHECK_N)], input, out);
            else if (k.net) A.mlpForward(k.rows && speciesAt(k.layout, i, CHECK_N) === 'wolf' ? k.rows : k.net, input, out);
            else out.set([input[2], input[3]]);
            for (let a = 0; a < def.actDim; a++) {
                const e = Math.abs(row[a] - out[a]);
                if (!(e <= TOL + TOL * Math.abs(out[a]))) bad++;
                if (Number.isFinite(e)) maxAbs = Math.max(maxAbs, e);
            }
        }
        check(`${name}: ${k.name} = CPU on ${fmt(CHECK_N)} agents`, bad === 0 && touched === 0,
            `max abs error ${maxAbs.toExponential(2)}` + (k.thinkEvery > 1 ? `, skipped rows untouched: ${touched === 0}` : ''));
        return maxAbs;
    }

    const result = { adapter: name, host: inBrowser ? 'chrome' : 'node', timing: timestamps ? 'timestamps' : 'wall-clock',
        weightBytes: { [HERD.name]: A.mlpLayout(herdNet).floats * 4, [WIDE.name]: A.mlpLayout(maxNet).floats * 4 },
        kernels: {} };
    const compiled = [];
    for (const k of KERNELS) {
        const byCount = new Map();
        for (const n of COUNTS) byCount.set(n, await pipeline(k, n));
        const maxAbs = await verify(k, byCount.get(CHECK_N));
        compiled.push({ k, byCount });
        result.kernels[k.name] = { compileMs: byCount.get(CHECK_N).ms, maxAbs, us: {} };
    }
    const heavy = compiled[1].byCount.get(COUNTS[COUNTS.length - 1]), t0 = performance.now();
    while (performance.now() - t0 < WARMUP_MS) await timedPass(heavy, 20);
    for (let round = 0; round < ROUNDS; round++) {
        for (const n of COUNTS) {
            for (const { k, byCount } of compiled) {
                const entry = result.kernels[k.name];
                entry.us[n] = Math.min(entry.us[n] ?? Infinity, await measure(byCount.get(n)));
            }
        }
    }
    result.quantized = rawDeltas.length > 0 && rawDeltas.every(d => d % 100_000 === 0);
    for (const brain of brains.values()) brain.destroy();
    rowsBrain.destroy();
    for (const brain of manyBrains.values()) brain.destroy();
    for (const brain of speciesBrains) brain.destroy();
    for (const b of made) b.destroy();
    ts?.set.destroy();
    device.destroy();
    return result;
}

function report(r) {
    const timing = r.timing === 'timestamps' ? `GPU timestamps${r.quantized ? ', quantized to 100 µs' : ''}` : 'wall-clock';
    log(`\n== ${r.host === 'node' ? 'Node/Dawn' : 'Chrome'} · ${r.adapter} — µs per think dispatch, ` +
        `lower of ${ROUNDS} medians of ${TRIALS}, after a ${WARMUP_MS} ms warm-up (${timing})`);
    log(`   ${'kernel'.padEnd(32)}${COUNTS.map(n => fmt(n).padStart(10)).join('')}   compile   max abs error`);
    for (const [name, e] of Object.entries(r.kernels)) {
        log(`   ${name.padEnd(32)}${COUNTS.map(n => e.us[n].toFixed(1).padStart(10)).join('')}` +
            `   ${e.compileMs.toFixed(0).padStart(5)} ms   ${e.maxAbs.toExponential(1)}`);
    }
    const floor = r.kernels[FLOOR.name], herd = r.kernels[HERD.name], big = COUNTS[COUNTS.length - 1];
    log(`   at ${fmt(big)}: MLP 24→32→32→2 ${((herd.us[big] * 1e3) / big).toFixed(1)} ns per agent, ` +
        `stage floor ${((floor.us[big] * 1e3) / big).toFixed(1)} ns; weights ${Object.entries(r.weightBytes).map(([n, b]) => `${n.split(' ')[1]} ${fmt(b)} B`).join(', ')}`);
    for (const N of [2, ...MANY]) {
        const rowsN = r.kernels[N === 2 ? 'rows: 2 nets, species contiguous' : `rows: ${N} species, contiguous`];
        const passesN = r.kernels[N === 2 ? 'dispatch per species, contiguous' : `passes: ${N} species, contiguous`];
        log(`   ${N} species, rows ÷ passes at ${COUNTS.map(fmt).join(' / ')}: ` +
            COUNTS.map(c => `${(rowsN.us[c] / passesN.us[c]).toFixed(2)}`).join(' / ') + '  (below 1: rows faster)');
    }
    const rows = Object.entries(r.kernels).filter(([n]) => n.startsWith('rows:'));
    log(`   rows against one network, at ${COUNTS.map(fmt).join(' / ')}: ` +
        rows.map(([n, e]) => `${n.split(', ')[1]} ${COUNTS.map(c => `${Math.round((100 * e.us[c]) / herd.us[c])}%`).join(' / ')}`).join('; '));
}

const results = [];
const seen = new Set();
for (const powerPreference of ['high-performance', 'low-power']) {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference });
    if (!adapter) continue;
    const i = adapter.info ?? {};
    const key = [i.vendor, i.architecture, i.device, i.description].join('|');
    if (seen.has(key)) continue;
    seen.add(key);
    const name = [i.vendor, i.architecture].filter(Boolean).join(' ') || powerPreference;
    log(`MLP think: measuring on ${name}${i.description ? ` (${i.description})` : ''}…`);
    const r = await runAdapter(adapter, name);
    results.push(r);
    report(r);
}
if (!results.length) {
    fatal('No WebGPU adapter');
    throw new Error('No WebGPU adapter');
}
log(`\nMLP think JSON ${JSON.stringify(results)}`);
await finish(`MLP think: every kernel matched the CPU on ${results.length} adapter(s).`);
log('MLP think done');
