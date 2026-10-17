// Phase 2, spike S6 (docs/work/plan-2.md §5, milestone 2A): what a weight row per agent costs, measured
// before step 11 (MlpBrain.byGenome) is designed. A measurement, not a test suite: it is not registered in
// test/gpu/suites.mjs, and its numbers are recorded in step 11's design.
//
// Phase 7.2 found that a weight row chosen per lane costs 4.6× a single network in Chrome even when every
// agent reads the same row: "the rows penalty is the compiler's, not the data's"
// (docs/agents-species-brains-design.md, "Phase 7.2 results"). Evolution needs a row per agent. This spike
// measures what that costs, whether a row the compiler can prove uniform recovers the cost, and what
// share of a whole step it is.
//
//   Node + Dawn, on every adapter Dawn offers:  npm run spike:s6
//   Chrome:  npm run drive -- --serve . "test/spikes/genome-rows/index.html" --wait-for "S6 done" --timeout 1800
//   One part only:  NG_SPIKE_PART=kernels|steps under Node, ?part=kernels|steps in Chrome (default both)
//   Fewer agents:   NG_SPIKE_MAX=16384 under Node, ?max=16384 in Chrome (default: 4,096, 16,384 and 65,536)
//   Some kernels:   NG_SPIKE_VARIANTS=genome,staged,fallback under Node, ?variants=… in Chrome (one network always)
//
// Part 1, the think kernel alone: built as AgentSystem builds it (buildAgentThinkWGSL around the brain's
// WGSL), bound as the think stage binds it, every agent alive and thinking, one species. Each kernel is
// checked against mlpForward of every agent's own row before it is timed.
//   one network    today's single MlpBrain: the reference
//   (a) genome     each lane reads its row from its record's genome word: what step 11 would build first
//   (b) workgroup  the row derived from @builtin(workgroup_id): uniform by construction
//   (c) broadcast  lane 0 reads its genome word, and workgroupUniformLoad shares it with its block
//   (d) passes     one dispatch per row, over that row's agents, the row's offset in a uniform
//   (e) staged     the block's row (its first agent's genome, shared as in (c)) copied tile by tile into
//                  workgroup memory by all 64 lanes, coalesced; every lane then reads its weights there.
//                  Also run with one row, against today's kernel: does staging help a single network?
// Layouts of rows over agents: aligned (each 64-agent block reads one row), runs of 8 (each block reads 8
// rows, as wolves would in small arenas), shuffled (a row per agent at random). (b)–(e) need aligned rows;
// (a) is measured on all three.
//
// Part 2, a whole Herd & Wolves step (tools/agents/scenario/herd-wolves.mjs), as the films run it:
// today's brains (one MlpBrain per species, one pass each) against two rows brains with 64 rows per
// species, every row an identical copy of its species' network: (a), and since phase 11.3 the library's
// MlpBrain.byGenome (the staged kernel with its per-lane fallback). The simulation is then the same,
// word for word (checked), and only the weights' addressing differs. Two groups: the committed networks (C2's shape),
// and random networks of 7.2's smaller shape, the size lever of plan-2 §4.4. A group's configurations
// are built together, stepped in lockstep and timed in turn, so clock drift falls on all of them. Each
// submission is sized from a measured step and kept far below the ~2 s after which Windows resets a GPU.
import { harnessEnv } from '../../gpu/harness-env.mjs';
import * as A from '../../../dist/agents.js';
import { NullGraph } from '../../../dist/index.js';
import { herdDefinition, herdWolves } from '../../../tools/agents/scenario/herd-wolves.mjs';
import { networkFromModel } from '../../../tools/agents/model.mjs';

const { log, check, fatal, finish, inBrowser } = harnessEnv('__S6_RESULTS__');
const PART = (inBrowser ? new URLSearchParams(location.search).get('part') : globalThis.process?.env.NG_SPIKE_PART) || 'both';
if (!['kernels', 'steps', 'both'].includes(PART)) {
    fatal(`S6: unknown part "${PART}"; use kernels, steps or both`);
    throw new Error(`S6: unknown part "${PART}"`);
}

const SPIKE_MAX = Number((inBrowser ? new URLSearchParams(location.search).get('max') : globalThis.process?.env.NG_SPIKE_MAX) || Infinity);
const COUNTS = [4_096, 16_384, 65_536].filter(n => n <= SPIKE_MAX);
const ONLY = ((inBrowser ? new URLSearchParams(location.search).get('variants') : globalThis.process?.env.NG_SPIKE_VARIANTS) || '')
    .split(',').map(s => s.trim()).filter(Boolean);
const wanted = variant => variant === 'one' || !ONLY.length || ONLY.includes(variant);
const ROW_COUNTS = [16, 64, 256];
const TRIALS = 5;
const ROUNDS = 2;                // part 1: the whole matrix twice; each cell keeps its lower median
const WARMUP_MS = 1000;          // a GPU idles at low clocks and boosts only under sustained load
const PASS_NS = 5e6;             // ≥ 5 ms of work per timed pass: Chrome may quantize timestamps to 100 µs
const PROBE_LIMIT_NS = 200e6;    // a kernel slower than this in one dispatch is not timed (the GPU reset)
const TOL = 1e-5;                // A4's |gpu − cpu| ≤ 1e-5 + 1e-5·|cpu|
const SENTINEL = 7;
const STEP_ROWS = 64;            // part 2: rows per species
const STEP_TRIALS = 9;           // part 2: timed submissions per configuration, taken in turn
const SETTLE = { 4096: 300, 16384: 200, 65536: 120 };   // steps before timing: herds form, and a step's cost moves
const SUBMIT_MS = 100;           // part 2: steps per submission are chosen to take about this long
const STEP_LIMIT_MS = 500;       // part 2: a count whose single step takes longer is not run
const MAX_REPEAT = 64;
const RECORD_WORDS = 16, GENOME_WORD = 14;

let seed = 0x5e6;
const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
const network = layers => layers.map(([input, output, activation]) => ({ input, output, activation,
    weights: Float32Array.from({ length: input * output }, () => (random() * 2 - 1) * Math.sqrt(3 / input)),
    biases: Float32Array.from({ length: output }, () => (random() * 2 - 1) * 0.1) }));
// C2's networks (tools/agents/models/herd-wolves) and phase 7.2's test network; one random network per row.
const SHAPES = [
    { name: 'C2 24→64→64→64→2', layers: [[24, 64, 'tanh'], [64, 64, 'tanh'], [64, 64, 'tanh'], [64, 2, 'tanh']] },
    { name: '7.2 24→32→32→2', layers: [[24, 32, 'tanh'], [32, 32, 'tanh'], [32, 2, 'tanh']] },
].map(s => ({ ...s, nets: Array.from({ length: Math.max(...ROW_COUNTS) }, () => network(s.layers)) }));

const hash = i => {
    let h = Math.imul(i ^ 0x9e3779b9, 0x85ebca6b) >>> 0;
    h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
    return (h ^ (h >>> 16)) >>> 0;
};
/** Part 1's row of agent i of n: aligned rows own n / rows consecutive agents, a multiple of 64. */
const LAYOUTS = {
    'aligned': (i, n, rows) => Math.floor(i / (n / rows)),
    'runs of 8': (i, n, rows) => Math.floor(i / 8) % rows,
    'shuffled': (i, n, rows) => hash(i) % rows,
};
/** Part 2's: a row per 64-agent block, cycling; the scenario's wolves fill whole blocks (1 in 16 agents). */
const STEP_LAYOUTS = {
    'aligned': i => Math.floor(i / 64) % STEP_ROWS,
    'runs of 8': i => Math.floor(i / 8) % STEP_ROWS,
    'shuffled': i => hash(i) % STEP_ROWS,
};
const alignable = (n, rows) => (n / rows) % 64 === 0;

const species = [
    { name: 'prey', maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 8, radius: 0.4 },
    { name: 'wolf', maxSpeed: 4, maxTurn: 3, accel: 5, senseRadius: 8, radius: 0.6 },
];
// Herd & Wolves' 24 observations and 2 steering actions, as phase 7.2 used them.
const herdDef = capacity => A.defineAgents({ capacity, world: { min: [-200, -200], max: [200, 200] }, cellSize: 8, species,
    sensors: [A.Sensors.self(), A.Sensors.nearest({ name: 'wolf', species: ['wolf'], k: 1 }),
        A.Sensors.nearest({ name: 'prey', species: ['prey'], k: 3 }), A.Sensors.bounds()] });

// --- the think WGSL of each variant ----------------------------------------------------------------
const HEAD = 'fn think(i: u32, species: u32, tick: u32) {\n';
const ENTRY = 'fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {\n';
const INDEX = '    let i = id.x;\n';
function replaceOnce(text, from, to, what) {
    const at = text.indexOf(from);
    if (at < 0 || text.indexOf(from, at + 1) >= 0) throw new Error(`S6: ${what}: expected exactly one "${from.trim()}"`);
    return text.slice(0, at) + to + text.slice(at + from.length);
}
const quadsOf = net => A.mlpLayout(net).floats / 4;
/** The single network's think WGSL with every weight read offset by `base`, a WGSL expression. */
const offsetWGSL = (net, base) => A.buildMlpThinkWGSL(net).replaceAll('mlpWeights[', `mlpWeights[${base} + `);
/** (a): the row from the agent's own genome word, clamped to the table. */
const genomeWGSL = (net, rows) => replaceOnce(offsetWGSL(net, 'row'), HEAD,
    `${HEAD}    let row = min(agents[i].ids.z, ${rows - 1}u) * ${quadsOf(net)}u;\n`, 'think head');

const VARIANTS = {
    one: 'one network',
    genome: '(a) genome per lane',
    workgroup: '(b) workgroup index',
    broadcast: '(c) workgroup broadcast',
    passes: '(d) one pass per row',
    staged: '(e) staged in workgroup',
    fallback: '(e′) staged + per-lane',
};
const TILE_QUADS = 512;          // (e): 8 KB of workgroup memory per tile, half the default limit
const ACTIVATIONS = {
    relu: 'z = max(z, 0.0);',
    tanh: 'z = tanh(clamp(z, -15.0, 15.0));',
    sigmoid: 'let e = exp(-abs(z));\n            z = select(e / (1.0 + e), 1.0 / (1.0 + e), z >= 0.0);',
    identity: '',
};
/**
 * (e)'s entry, replacing the wrapper's: the same gate (alive, cadence, species mask), then each layer's
 * weights copied into workgroup memory a tile at a time by all 64 lanes, between barriers, and read
 * there by the lanes that think. The barriers stay in uniform control flow: no lane returns early.
 * The arithmetic is mlpShaders' own, in the same order. With `fallback` ((e′)), a lane whose genome
 * names another row than its block's first agent reads its own row from storage instead, as (a) does:
 * correct for any layout, and fast where blocks share a row.
 */
function stagedEntry(def, net, rows, fallback = false) {
    const layout = A.mlpLayout(net), quads = layout.floats / 4, last = net.length;
    let body = '';
    net.forEach((layer, L) => {
        const p = layout.layers[L], inQ = p.inputStride / 4, tileRows = Math.min(layer.output, Math.floor(TILE_QUADS / inQ));
        body += `    var x${L + 1}: array<vec4<f32>, ${p.outputStride / 4}>;\n`;
        for (let r0 = 0; r0 < layer.output; r0 += tileRows) {
            const r1 = Math.min(layer.output, r0 + tileRows);
            const rowLoop = (base, weight) => `for (var r = ${r0}u; r < ${r1}u; r++) {
            var z = mlpWeights[${base} + ${p.biasOffset / 4}u + r / 4u][r % 4u];
            for (var k = 0u; k < ${inQ}u; k++) { z += dot(${weight}, x${L}[k]); }
            ${ACTIVATIONS[layer.activation]}
            x${L + 1}[r / 4u][r % 4u] = z;
        }`;
            const fromTile = rowLoop('rowBase', `tile[(r - ${r0}u) * ${inQ}u + k]`);
            const fromRow = rowLoop('myBase', `mlpWeights[myBase + ${p.weightOffset / 4}u + r * ${inQ}u + k]`);
            body += `    workgroupBarrier();
    for (var q = lid; q < ${(r1 - r0) * inQ}u; q += 64u) { tile[q] = mlpWeights[rowBase + ${p.weightOffset / 4 + r0 * inQ}u + q]; }
    workgroupBarrier();
    if (thinking) {
        ${fallback ? `if (myBase == rowBase) { ${fromTile} } else { ${fromRow} }` : fromTile}
    }
`;
        }
    });
    return `var<workgroup> stagedRow: u32;
var<workgroup> tile: array<vec4<f32>, ${TILE_QUADS}>;
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32) {
    let i = id.x;
    if (lid == 0u) { stagedRow = min(agents[wg.x * 64u].ids.z, ${rows - 1}u) * ${quads}u; }
    let rowBase = workgroupUniformLoad(&stagedRow);
    var thinking = false;
    var myBase = 0u;
    if (i < ${def.capacity}u) {
        let a = agents[i];
        thinking = thinksNow(i, a) && (brainSpecies.x & (1u << a.ids.x)) != 0u;
        myBase = min(a.ids.z, ${rows - 1}u) * ${quads}u;
    }
    _ = arrayLength(&observations);
    _ = arrayLength(&act);
    _ = arrayLength(&reason);
    _ = species[0].maxSpeed;
    var x0: array<vec4<f32>, ${Math.ceil(net[0].input / 4)}>;
    if (thinking) {
        for (var k = 0u; k < ${net[0].input}u; k++) { x0[k / 4u][k % 4u] = obs(i, k); }
    }
${body}    if (thinking) {
        for (var k = 0u; k < ${net[last - 1].output}u; k++) { setAct(i, k, x${last}[k / 4u][k % 4u]); }
        setReason(i, 0u);
    }
}`;
}
function kernelCode(k, def) {
    const { net, rows, n } = k, quads = quadsOf(net);
    switch (k.variant) {
        case 'one': return A.buildAgentThinkWGSL(def, A.buildMlpThinkWGSL(net), 1);
        case 'genome': return A.buildAgentThinkWGSL(def, genomeWGSL(net, rows), 1);
        case 'workgroup': return replaceOnce(
            A.buildAgentThinkWGSL(def, `var<private> rowBase: u32;\n${offsetWGSL(net, 'rowBase')}`, 1), ENTRY,
            'fn cs_main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {\n' +
            `    rowBase = (wg.x / ${n / rows / 64}u) * ${quads}u;\n`, 'entry');
        case 'broadcast': return replaceOnce(
            A.buildAgentThinkWGSL(def, `var<private> rowBase: u32;\nvar<workgroup> sharedRow: u32;\n${offsetWGSL(net, 'rowBase')}`, 1), ENTRY,
            'fn cs_main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>, ' +
            '@builtin(local_invocation_index) lid: u32) {\n' +
            `    if (lid == 0u) { sharedRow = min(agents[wg.x * 64u].ids.z, ${rows - 1}u) * ${quads}u; }\n` +
            '    rowBase = workgroupUniformLoad(&sharedRow);\n', 'entry');
        case 'passes': return replaceOnce(
            A.buildAgentThinkWGSL(def, `@group(1) @binding(1) var<uniform> passRow: vec4<u32>;\n${offsetWGSL(net, 'passRow.y')}`, 1),
            INDEX, '    let i = id.x + passRow.x;\n', 'index');
        case 'staged': case 'fallback': {
            // The wrapper's module with its think left unused and its entry replaced; the weights binding kept.
            const code = A.buildAgentThinkWGSL(def, A.buildMlpThinkWGSL(net), 1);
            const at = code.indexOf('@compute @workgroup_size(64) fn cs_main');
            if (at < 0) throw new Error('S6: staged: the wrapper\'s entry was not found');
            return code.slice(0, at) + stagedEntry(def, net, rows, k.variant === 'fallback');
        }
        default: throw new Error(`S6: unknown variant ${k.variant}`);
    }
}

const median = xs => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
const fmt = n => n.toLocaleString('en-US');
const pct = (x, ref) => `${x >= ref ? '+' : '−'}${Math.abs((100 * (x - ref)) / ref).toFixed(1)}%`;
const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

// =============================================================================================
// Part 1: the think kernel alone
// =============================================================================================
async function runKernels(device, name, timestamps) {
    const made = [];
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
    const everySpecies = buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, new Uint32Array([0xffff, 0, 0, 0]));

    // Per count: observations, the think stage's other buffers; per count, rows and layout: the records.
    const perCount = new Map();
    function counted(n) {
        if (perCount.has(n)) return perCount.get(n);
        const def = herdDef(n);
        const obs = Float32Array.from({ length: n * def.obsDim }, () => random() * 2 - 1);
        const speciesData = A.packSpecies(def);
        const c = { def, obs, records: new Map(), buffers: {
            obs: buffer(obs.byteLength, STORAGE, obs),
            state: buffer(16, STORAGE, new Uint32Array([0, 0, 0, 0])),
            act: buffer(n * def.actDim * 4, STORAGE),
            reason: buffer(n * 4, STORAGE),
            species: buffer(speciesData.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, speciesData),
        } };
        perCount.set(n, c);
        return c;
    }
    /** The records of n agents, alive, age 1, all prey, each with its row in the genome word. */
    function recordsOf(n, rows, layout) {
        const c = counted(n), key = `${rows}:${layout}`;
        if (c.records.has(key)) return c.records.get(key);
        const records = new A.AgentRecords(c.def), genome = new Uint32Array(n);
        for (let i = 0; i < n; i++) {
            genome[i] = layout ? LAYOUTS[layout](i, n, rows) : 0;
            records.spawn(i, { position: [0, 0, 0], age: 1, species: 'prey', genome: genome[i] });
        }
        const r = { genome, buffer: buffer(records.data.byteLength, STORAGE, records.data) };
        c.records.set(key, r);
        return r;
    }
    // One weight table per shape and row count; the reference reads row 0 alone.
    const tables = new Map();
    function tableOf(shape, rows) {
        const key = `${shape.name}:${rows}`;
        if (tables.has(key)) return tables.get(key);
        const floats = A.mlpLayout(shape.nets[0]).floats, data = new Float32Array(floats * rows);
        for (let r = 0; r < rows; r++) A.packMlpWeights(shape.nets[r], data.subarray(r * floats, (r + 1) * floats));
        const b = buffer(data.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, data);
        tables.set(key, b);
        return b;
    }

    const kernels = [];
    for (const shape of SHAPES) {
        for (const n of COUNTS) {
            kernels.push({ shape, n, variant: 'one', net: shape.nets[0], rows: 1, layout: null });
            kernels.push({ shape, n, variant: 'staged', net: shape.nets[0], rows: 1, layout: null });
            for (const rows of ROW_COUNTS) {
                for (const variant of ['genome', 'fallback']) {
                    for (const layout of Object.keys(LAYOUTS)) {
                        if (layout !== 'aligned' || alignable(n, rows)) kernels.push({ shape, n, variant, net: shape.nets[0], rows, layout });
                    }
                }
                if (!alignable(n, rows)) continue;
                for (const variant of ['workgroup', 'broadcast', 'passes', 'staged']) kernels.push({ shape, n, variant, net: shape.nets[0], rows, layout: 'aligned' });
            }
        }
    }
    kernels.splice(0, kernels.length, ...kernels.filter(k => wanted(k.variant)));
    const label = k => `${k.shape.name} · ${VARIANTS[k.variant]}${k.layout ? ` · ${k.rows} rows ${k.layout}` : ''} · ${fmt(k.n)}`;

    async function build(k) {
        const c = counted(k.n), rec = recordsOf(k.n, k.rows, k.layout);
        const code = kernelCode(k, c.def), t0 = performance.now();
        const p = await device.createComputePipelineAsync({ label: label(k), layout: 'auto',
            compute: { module: device.createShaderModule({ label: label(k), code }), entryPoint: 'cs_main' } });
        const group0 = device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [rec.buffer, c.buffers.obs, c.buffers.state,
            c.buffers.act, c.buffers.reason, c.buffers.species, everySpecies].map((b, binding) => ({ binding, resource: { buffer: b } })) });
        const weights = tableOf(k.shape, k.rows);
        const parts = [];
        if (k.variant === 'passes') {
            // One uniform per row at 256-byte offsets: its first agent and its row's offset in vec4s.
            const per = k.n / k.rows, quads = quadsOf(k.net), data = new Uint32Array(k.rows * 64);
            for (let r = 0; r < k.rows; r++) data.set([r * per, r * quads], r * 64);
            const uniforms = buffer(data.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, data);
            for (let r = 0; r < k.rows; r++) {
                parts.push({ groups: per / 64, binds: [group0, device.createBindGroup({ layout: p.getBindGroupLayout(1), entries: [
                    { binding: 0, resource: { buffer: weights } },
                    { binding: 1, resource: { buffer: uniforms, offset: r * 256, size: 16 } }] })] });
            }
        } else {
            parts.push({ groups: k.n / 64, binds: [group0, device.createBindGroup({ layout: p.getBindGroupLayout(1),
                entries: [{ binding: 0, resource: { buffer: weights } }] })] });
        }
        return { k, p, parts, compileMs: performance.now() - t0 };
    }
    async function timedPass(c, repeats) {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass(ts
            ? { timestampWrites: { querySet: ts.set, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
        pass.setPipeline(c.p);
        for (let r = 0; r < repeats; r++) {
            for (const part of c.parts) {
                part.binds.forEach((b, g) => pass.setBindGroup(g, b));
                pass.dispatchWorkgroups(part.groups);
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
        const one = await timedPass(c, 1);
        const per = one.gpuNs ?? one.wallNs;
        if (per > PROBE_LIMIT_NS) return null;
        const repeats = Math.max(1, Math.min(4000, Math.ceil(PASS_NS / Math.max(per, 250))));
        const gpu = [], wall = [];
        for (let t = 0; t < TRIALS; t++) {
            const r = await timedPass(c, repeats);
            if (r.gpuNs !== null) { gpu.push(r.gpuNs / repeats); rawDeltas.push(r.gpuNs); }
            wall.push(r.wallNs / repeats);
        }
        return gpu.length === TRIALS ? median(gpu) / 1e3 : median(wall) / 1e3;
    }
    // The CPU's actions for every agent of n, by its row: kept per shape, count, rows and layout.
    const expected = new Map();
    function expectedOf(k) {
        const key = `${k.shape.name}:${k.n}:${k.variant === 'one' ? 'one' : `${k.rows}:${k.layout}`}`;
        if (expected.has(key)) return expected.get(key);
        const c = counted(k.n), { genome } = recordsOf(k.n, k.rows, k.layout), { obsDim, actDim } = c.def;
        const out = new Float32Array(k.n * actDim), input = new Float32Array(obsDim);
        const scratch = new Float32Array(2 * A.mlpLayout(k.net).scratchWidth);
        for (let i = 0; i < k.n; i++) {
            input.set(c.obs.subarray(i * obsDim, (i + 1) * obsDim));
            A.mlpForward(k.shape.nets[k.variant === 'one' ? 0 : genome[i]], input, out.subarray(i * actDim, (i + 1) * actDim), scratch);
        }
        expected.set(key, out);
        return out;
    }
    // The GPU's own actions from today's kernel and from (a), kept so (e) can be compared bit for bit: if
    // staging changes no bit, moving a brain onto it changes nothing but its speed.
    const bitsOf = new Map();
    const bitsKey = (k, variant) => `${k.shape.name}:${k.n}:${variant}:${k.rows}:${k.layout}`;
    async function verify(c) {
        const { k } = c, { def, buffers } = counted(k.n);
        device.queue.writeBuffer(buffers.act, 0, new Float32Array(k.n * def.actDim).fill(SENTINEL));
        await timedPass(c, 1);
        const got = await readF32(buffers.act, k.n * def.actDim), want = expectedOf(k);
        let maxAbs = 0, bad = 0;
        for (let j = 0; j < want.length; j++) {
            const e = Math.abs(got[j] - want[j]);
            if (!(e <= TOL + TOL * Math.abs(want[j]))) bad++;
            if (Number.isFinite(e)) maxAbs = Math.max(maxAbs, e);
        }
        check(`${name}: ${label(k)} = CPU on every agent`, bad === 0, `max abs error ${maxAbs.toExponential(2)}, ${bad} wrong`);
        const bits = new Uint32Array(got.buffer);
        if (k.variant === 'one' || k.variant === 'genome') bitsOf.set(bitsKey(k, k.variant), bits);
        if (k.variant === 'staged' || k.variant === 'fallback') {
            const ref = bitsOf.get(k.rows === 1 ? bitsKey({ ...k, layout: null }, 'one') : bitsKey(k, 'genome'));
            c.bitsDiffer = ref ? bits.reduce((n, w, j) => n + (w !== ref[j] ? 1 : 0), 0) : null;
        }
        return maxAbs;
    }

    const compiled = [];
    for (const k of kernels) {
        const c = await build(k);
        c.maxAbs = await verify(c);
        compiled.push(c);
    }
    const heavy = compiled.find(c => c.k.variant === 'one' && c.k.n === COUNTS[COUNTS.length - 1]), t0 = performance.now();
    while (performance.now() - t0 < WARMUP_MS) await timedPass(heavy, 4);
    const us = new Map();
    for (let round = 0; round < ROUNDS; round++) {
        for (const c of compiled) {
            const v = await measure(c);
            const was = us.get(c);
            us.set(c, v === null ? null : Math.min(was ?? Infinity, v));
        }
    }
    const result = { adapter: name, host: inBrowser ? 'chrome' : 'node', timing: timestamps ? 'timestamps' : 'wall-clock',
        quantized: rawDeltas.length > 0 && rawDeltas.every(d => d % 100_000 === 0), shapes: {} };
    for (const shape of SHAPES) {
        const rows = compiled.filter(c => c.k.shape === shape).map(c => ({ variant: c.k.variant, rows: c.k.rows, layout: c.k.layout,
            count: c.k.n, us: us.get(c), compileMs: c.compileMs, maxAbs: c.maxAbs, bitsDiffer: c.bitsDiffer }));
        result.shapes[shape.name] = { rowBytes: A.mlpLayout(shape.nets[0]).floats * 4, kernels: rows };
    }
    ts?.set.destroy();
    for (const b of made) b.destroy();
    return result;
}

function reportKernels(r) {
    const timing = r.timing === 'timestamps' ? `GPU timestamps${r.quantized ? ', quantized to 100 µs' : ''}` : 'wall-clock';
    log(`\n== S6 part 1 · ${r.host === 'node' ? 'Node/Dawn' : 'Chrome'} · ${r.adapter} — µs per think dispatch, ` +
        `lower of ${ROUNDS} medians of ${TRIALS}, after a ${WARMUP_MS} ms warm-up (${timing}); in brackets, against one network`);
    for (const [shapeName, s] of Object.entries(r.shapes)) {
        log(`   ${shapeName}, ${fmt(s.rowBytes)} B a row`);
        log(`   ${'kernel'.padEnd(26)}${'rows'.padStart(5)}  ${'layout'.padEnd(10)}${COUNTS.map(n => fmt(n).padStart(20)).join('')}`);
        const ref = new Map(s.kernels.filter(k => k.variant === 'one').map(k => [k.count, k.us]));
        const keyOf = k => `${k.variant}:${k.rows}:${k.layout}`;
        const seen = [];
        for (const k of s.kernels) if (!seen.includes(keyOf(k))) seen.push(keyOf(k));
        for (const key of seen) {
            const cells = s.kernels.filter(k => keyOf(k) === key), first = cells[0];
            const text = COUNTS.map(n => {
                const k = cells.find(x => x.count === n);
                if (!k) return '—'.padStart(20);
                if (k.us === null) return 'not timed'.padStart(20);
                return (first.variant === 'one' ? k.us.toFixed(1) : `${k.us.toFixed(1)} (${(k.us / ref.get(n)).toFixed(2)}×)`).padStart(20);
            }).join('');
            log(`   ${VARIANTS[first.variant].padEnd(26)}${String(first.rows).padStart(5)}  ${(first.layout ?? '—').padEnd(10)}${text}`);
        }
        const staged = s.kernels.filter(k => (k.variant === 'staged' || k.variant === 'fallback') && Number.isInteger(k.bitsDiffer));
        if (staged.length) {
            const same = staged.filter(k => k.bitsDiffer === 0).length;
            log(`   (e) and (e′) against today's kernel (one row) and (a) (rows), bit for bit: ${same} of ${staged.length} cells identical` +
                (same < staged.length ? `; at most ${fmt(Math.max(...staged.map(k => k.bitsDiffer)))} words differ` : ''));
        }
    }
}

// =============================================================================================
// Part 2: a whole Herd & Wolves step
// =============================================================================================
async function loadJSON(url) {
    return inBrowser ? (await fetch(url)).json() : JSON.parse((await import('node:fs')).readFileSync(url, 'utf8'));
}
const MODELS = new URL('../../../tools/agents/models/herd-wolves/', import.meta.url);

/** (a) as a brain object: `rows` identical copies of `net`, each agent reading the row its genome names. */
function rowsBrain(device, net, rows) {
    const floats = A.mlpLayout(net).floats, table = new Float32Array(floats * rows);
    for (let r = 0; r < rows; r++) A.packMlpWeights(net, table.subarray(r * floats, (r + 1) * floats));
    const weights = device.createBuffer({ label: 'S6 rows', size: table.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(weights, 0, table);
    return { kind: 'stage', wgsl: genomeWGSL(net, rows), buffers: [weights],
        dims: { obs: net[0].input, act: net[net.length - 1].output }, destroy() { weights.destroy(); } };
}

async function runSteps(device, name) {
    const engine = new NullGraph();
    const init = await engine.init(null, { device, width: 64, height: 64 });
    if (!init.success) throw new Error(`S6: engine: ${init.error}`);
    const nets = {};
    for (const s of ['prey', 'wolf']) nets[s] = networkFromModel(await loadJSON(new URL(`${s}.mlp.json`, MODELS)), herdDefinition(), `${s}.mlp.json`);

    const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
    const resolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const mark = index => {
        const enc = device.createCommandEncoder();
        enc.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: index } }).end();
        device.queue.submit([enc.finish()]);
    };
    /** Submits `sequence` alone between two timestamp marks and waits: GPU ms. */
    async function timed(sequence) {
        mark(0);
        engine.render(sequence);
        mark(1);
        const enc = device.createCommandEncoder();
        enc.resolveQuerySet(querySet, 0, 2, resolve, 0);
        enc.copyBufferToBuffer(resolve, 0, read, 0, 16);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const t = new BigUint64Array(read.getMappedRange().slice(0));
        read.unmap();
        return Number(t[1] - t[0]) / 1e6;
    }
    async function readU32(src, words) {
        const dst = device.createBuffer({ size: words * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(src, 0, dst, 0, words * 4);
        device.queue.submit([enc.finish()]);
        await dst.mapAsync(GPUMapMode.READ);
        const out = new Uint32Array(dst.getMappedRange().slice(0));
        dst.unmap();
        dst.destroy();
        return out;
    }
    /** Records that differ from `ref` in any word but the genome. */
    const differing = (got, ref) => {
        let off = 0;
        for (let i = 0; i < got.length; i += RECORD_WORDS) {
            for (let w = 0; w < RECORD_WORDS; w++) if (w !== GENOME_WORD && got[i + w] !== ref[i + w]) { off++; break; }
        }
        return off;
    };

    // Two groups: C2's committed networks, and phase 7.2's smaller shape (random networks, so the herd
    // behaves differently, but identically within the group). Within a group every configuration runs
    // the same simulation; all are built together, stepped in lockstep, and timed in turn, trial by
    // trial, so that a GPU's clocks drifting between configurations falls on all of them alike.
    // A configuration is today's brain (null), or a rows brain on a layout: (a), this spike's per-lane
    // prototype, or byGenome, the library's MlpBrain.byGenome (phase 11.2's staged kernel with its
    // per-lane fallback; timed here for phase 11.3).
    const rowsConfigs = layouts => [null, ...layouts.flatMap(layout => [{ kind: '(a)', layout }, { kind: 'byGenome', layout }])];
    const GROUPS = [
        { shape: 'C2 24→64→64→64→2 (committed)', nets, configs: rowsConfigs(Object.keys(STEP_LAYOUTS)) },
        { shape: '7.2 24→32→32→2 (random)', nets: { prey: SHAPES[1].nets[0], wolf: SHAPES[1].nets[1] }, configs: rowsConfigs(['aligned']) },
    ];
    const configName = config => (config ? `${config.kind} ${config.layout}` : 'today');
    function build(group, config, n, thinkEvery) {
        const sim = herdWolves.create(engine, { count: n, thinkEvery });
        const brains = [];
        try {
            for (const s of ['prey', 'wolf']) {
                const net = group.nets[s];
                const brain = !config ? new A.MlpBrain(engine, net)
                    : config.kind === '(a)' ? rowsBrain(device, net, STEP_ROWS)
                    : A.MlpBrain.byGenome(engine, Array.from({ length: STEP_ROWS }, () => net));
                brains.push(brain);
                sim.agents.setBrain(brain, { species: s });
            }
            if (config) {
                const layout = config.layout;
                // The spawn createHerd made, again, with each agent's row in its genome word.
                const inset = sim.definition.world.maxX - 1, s = sim.options.seed;
                for (let i = 0; i < n; i++) {
                    sim.agents.spawn(i, { position: [(A.rand01(i, s, 0) * 2 - 1) * inset, 0.08, (A.rand01(i, s, 1) * 2 - 1) * inset],
                        species: i < sim.wolves ? 'wolf' : 'prey', heading: A.rand01(i, s, 2) * Math.PI * 2 - Math.PI,
                        genome: STEP_LAYOUTS[layout](i) });
                }
            }
            if (sim.agents.advance(sim.definition.dt) !== 1) throw new Error('S6: advance(dt) did not select one step');
            sim.agents.upload();
            return { name: configName(config), sim, brains, chain: sim.agents.getComputeStages(),
                destroy() { sim.destroy(); brains.forEach(b => b.destroy()); } };
        } catch (error) {
            sim.destroy();
            brains.forEach(b => b.destroy());
            throw error;
        }
    }

    const rows = [];
    for (const group of GROUPS) {
        for (const thinkEvery of [1, 3]) {
            for (const n of COUNTS) {
                const row = { shape: group.shape, thinkEvery, count: n, repeat: 0, configs: {} };
                const sims = [];
                let skipped = false;
                try {
                    for (const config of group.configs) sims.push(build(group, config, n, thinkEvery));
                    for (const s of sims) await timed([...s.chain, ...s.chain]);      // pipelines compile on first use
                    const probe = await timed(sims[0].chain);
                    for (const s of sims.slice(1)) await timed(s.chain);           // the same steps for every one
                    if (probe > STEP_LIMIT_MS) {
                        log(`   ${group.shape}, thinkEvery ${thinkEvery}, ${fmt(n)}: not run, one step took ${probe.toFixed(0)} ms (limit ${STEP_LIMIT_MS})`);
                        skipped = true;
                    } else {
                        row.repeat = Math.max(1, Math.min(MAX_REPEAT, Math.round(SUBMIT_MS / Math.max(probe, 0.01))));
                        for (const s of sims) s.repeat = Array.from({ length: row.repeat }, () => s.chain).flat();
                        for (let k = 0; k < SETTLE[n]; k += row.repeat) for (const s of sims) await timed(s.repeat);
                        const ms = sims.map(() => []);
                        for (let t = 0; t < STEP_TRIALS; t++) {
                            for (const [j, s] of sims.entries()) ms[j].push((await timed(s.repeat)) / row.repeat);
                        }
                        let reference = null;
                        for (const [j, s] of sims.entries()) {
                            const records = await readU32(s.sim.agents.buffer, n * RECORD_WORDS);
                            reference ??= records;
                            row.configs[s.name] = { ms: median(ms[j]), min: Math.min(...ms[j]), max: Math.max(...ms[j]),
                                differing: differing(records, reference) };
                        }
                    }
                } finally {
                    sims.forEach(s => s.destroy());
                }
                if (skipped) break;
                rows.push(row);
                for (const config of group.configs.slice(1)) {
                    const c = row.configs[configName(config)];
                    check(`${name}: steps, ${group.shape}, thinkEvery ${thinkEvery}, ${fmt(n)}: ${configName(config)} runs the same simulation as today's brain`,
                        c.differing === 0, `${fmt(c.differing)} of ${fmt(n)} records differ after the same steps`);
                }
            }
        }
    }
    querySet.destroy();
    resolve.destroy();
    read.destroy();
    return { adapter: name, host: inBrowser ? 'chrome' : 'node', rowsPerSpecies: STEP_ROWS, settle: SETTLE, rows };
}

function reportSteps(r) {
    log(`\n== S6 part 2 · ${r.host === 'node' ? 'Node/Dawn' : 'Chrome'} · ${r.adapter} — a whole Herd & Wolves step, ` +
        `GPU ms (median of ${STEP_TRIALS} submissions, taken in turn, after the settle), (a) with ${r.rowsPerSpecies} rows per species`);
    for (const shape of [...new Set(r.rows.map(row => row.shape))]) {
        const mine = r.rows.filter(row => row.shape === shape), names = Object.keys(mine[0].configs);
        log(`   ${shape}`);
        log(`   ${'thinkEvery'.padEnd(11)}${'count'.padStart(8)}${'steps/sub'.padStart(11)}${names.map(nm => nm.padStart(22)).join('')}`);
        for (const row of mine) {
            const today = row.configs.today.ms;
            log(`   ${String(row.thinkEvery).padEnd(11)}${fmt(row.count).padStart(8)}${String(row.repeat).padStart(11)}` +
                names.map(nm => {
                    const c = row.configs[nm];
                    return (nm === 'today' ? c.ms.toFixed(3) : `${c.ms.toFixed(3)} (${pct(c.ms, today)})`).padStart(22);
                }).join(''));
        }
        for (const kind of ['(a)', 'byGenome']) {
            const aligned = mine.filter(row => row.thinkEvery === 1).map(row => row.configs[`${kind} aligned`].ms / row.configs.today.ms - 1);
            log(`   the worst cost of ${kind} aligned at thinkEvery 1, over counts: ${(100 * Math.max(...aligned)).toFixed(1)}% of a step`);
        }
    }
}

// =============================================================================================
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
    const timestamps = adapter.features.has('timestamp-query');
    if (PART !== 'kernels' && !timestamps) {
        fatal(`S6: ${name} has no timestamp-query, which part 2 needs`);
        throw new Error('S6: no timestamp-query');
    }
    log(`S6: measuring on ${name}${i.description ? ` (${i.description})` : ''} (${inBrowser ? 'Chrome' : 'Node/Dawn'}), part ${PART}…`);
    const device = await adapter.requestDevice({ requiredFeatures: timestamps ? ['timestamp-query'] : [] });
    const r = { adapter: name, description: i.description ?? '', host: inBrowser ? 'chrome' : 'node' };
    if (PART !== 'steps') reportKernels(r.kernels = await runKernels(device, name, timestamps));
    if (PART !== 'kernels') reportSteps(r.steps = await runSteps(device, name));
    device.destroy();
    results.push(r);
}
if (!results.length) {
    fatal('No WebGPU adapter');
    throw new Error('No WebGPU adapter');
}
log(`\nS6 JSON ${JSON.stringify(results)}`);
await finish(`S6: every kernel matched the CPU, and every rows brain ran the same simulation, on ${results.length} adapter(s).`);
log('S6 done');
