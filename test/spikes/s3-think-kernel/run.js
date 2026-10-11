// Spike S3: what does the 24→32→32→2 network cost as a WGSL think kernel?
//
// See docs/agents-design.md §13 (C0, S3) and docs/agents-gpu-loop-design.md phase 3.0. This is a
// measurement, not a test suite: it is not registered in test/gpu/suites.mjs, and its numbers are
// recorded in those two documents. It makes only the checks a benchmark needs to mean anything:
// every kernel's output equals a CPU forward pass, and a gated kernel leaves the rows it skips
// untouched.
//
//   Node + Dawn, on every adapter Dawn offers:  npm run spike:s3
//   Chrome:  npm run drive -- --serve . test/spikes/s3-think-kernel/index.html --wait-for "S3 done" --timeout 300
//
// The questions:
//   1. Where do the weights live: a storage buffer (runtime-sized or fixed-size array), a uniform
//      buffer, or staged in workgroup memory? Staging needs workgroupBarrier(), and therefore
//      uniform control flow in the think entry point (agents-gpu-loop-design §7.1).
//   2. Loops over private arrays, or fully unrolled code with constant indices?
//   3. With thinkEvery = 3, does staggering by agent index save time, or only staggering by
//      64-agent block? The lanes of one wavefront execute together (§7.3).
//   4. Is WGSL tanh finite for large inputs on this adapter? A NaN in a hidden layer becomes a
//      non-finite action, and the act stage then stops the agent without a sound.
import { harnessEnv } from '../../gpu/harness-env.mjs';

const { log, check, fatal, finish, inBrowser } = harnessEnv('__S3_RESULTS__');

const LAYERS = [24, 32, 32, 2];            // the Herd & Wolves layout: obsDim 24 → actDim 2
const COUNTS = [1_000, 20_000, 100_000];
const GATED_COUNTS = [20_000, 100_000];
const CHECK_N = 20_000;
const THINK_EVERY = 3;
const TRIALS = 5;
const ROUNDS = 2;        // the whole matrix is measured twice; each cell keeps its lower median
const WARMUP_MS = 1000;  // a GPU idles at low clocks and boosts only under sustained load
const PASS_NS = 5e6;     // ≥ 5 ms of work per timed pass: Chrome may quantize timestamps to 100 µs
const TOLERANCE = 1e-4;  // against an f64 forward pass over the same f32 weights; the maximum is logged
const SENTINEL = 7;      // written into act before a gated run; a skipped row must still hold it

let seed = 0x5eed;
const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };

// ---------------------------------------------------------------------------------------------
// The network: Xavier-uniform weights and small biases, the scale a trained network has. It is
// packed as vec4s: per layer, rows padded to a multiple of 4 and each row to whole vec4s, then the
// biases. Zero padding keeps the loop kernels branch-free.
// ---------------------------------------------------------------------------------------------
const layout = [];
let QUADS = 0;
for (let l = 1; l < LAYERS.length; l++) {
    const inp = LAYERS[l - 1], out = LAYERS[l], a = Math.sqrt(6 / (inp + out));
    const W = Float32Array.from({ length: out * inp }, () => (random() * 2 - 1) * a);
    const b = Float32Array.from({ length: out }, () => (random() * 2 - 1) * 0.1);
    const in4 = Math.ceil(inp / 4), out4 = Math.ceil(out / 4);
    layout.push({ inp, out, W, b, in4, out4, w: QUADS, bias: QUADS + out4 * 4 * in4 });
    QUADS += out4 * 4 * in4 + out4;
}
const packed = new Float32Array(QUADS * 4);
for (const L of layout) {
    for (let r = 0; r < L.out; r++) {
        for (let k = 0; k < L.inp; k++) packed[(L.w + r * L.in4) * 4 + k] = L.W[r * L.inp + k];
        packed[L.bias * 4 + r] = L.b[r];
    }
}

const MAX_N = Math.max(...COUNTS);
const obs = Float32Array.from({ length: MAX_N * LAYERS[0] }, () => random() * 2 - 1);

// The CPU forward pass every kernel is checked against: f64 arithmetic over the f32 weights.
const expected = new Float64Array(CHECK_N * 2);
for (let i = 0; i < CHECK_N; i++) {
    let v = Array.from(obs.subarray(i * LAYERS[0], (i + 1) * LAYERS[0]));
    for (const L of layout) {
        const y = new Array(L.out);
        for (let r = 0; r < L.out; r++) {
            let s = L.b[r];
            for (let k = 0; k < L.inp; k++) s += L.W[r * L.inp + k] * v[k];
            y[r] = Math.tanh(s);
        }
        v = y;
    }
    expected[i * 2] = v[0];
    expected[i * 2 + 1] = v[1];
}

// ---------------------------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------------------------
function weightDecl(source) {
    if (source === 'uniform') return `@group(0) @binding(2) var<uniform> weights: array<vec4<f32>, ${QUADS}>;`;
    if (source === 'fixed') return `@group(0) @binding(2) var<storage, read> weights: array<vec4<f32>, ${QUADS}>;`;
    const runtime = '@group(0) @binding(2) var<storage, read> weights: array<vec4<f32>>;';
    return source === 'workgroup' ? `${runtime}\nvar<workgroup> staged: array<vec4<f32>, ${QUADS}>;` : runtime;
}

function loopBody(w) {
    const L0 = layout[0];
    let s = `    var x0: array<vec4<f32>, ${L0.in4}>;
    for (var k = 0u; k < ${L0.in4}u; k++) {
        let b = i * ${L0.inp}u + k * 4u;
        x0[k] = vec4<f32>(obs[b], obs[b + 1u], obs[b + 2u], obs[b + 3u]);
    }
`;
    layout.forEach((L, l) => {
        s += `    var x${l + 1}: array<vec4<f32>, ${L.out4}>;
    for (var o = 0u; o < ${L.out4}u; o++) {
        var v = ${w}[${L.bias}u + o];
        for (var r = 0u; r < 4u; r++) {
            let row = ${L.w}u + (o * 4u + r) * ${L.in4}u;
            var t = 0.0;
            for (var k = 0u; k < ${L.in4}u; k++) { t += dot(${w}[row + k], x${l}[k]); }
            v[r] += t;
        }
        x${l + 1}[o] = tanh(v);
    }
`;
    });
    const y = `x${layout.length}[0]`;
    return `${s}    act[i * 2u] = ${y}.x;\n    act[i * 2u + 1u] = ${y}.y;`;
}

function unrolledBody(w) {
    const L0 = layout[0];
    let s = '';
    for (let k = 0; k < L0.in4; k++) {
        const at = (c) => `obs[i * ${L0.inp}u + ${4 * k + c}u]`;
        s += `    let x0_${k} = vec4<f32>(${at(0)}, ${at(1)}, ${at(2)}, ${at(3)});\n`;
    }
    layout.forEach((L, l) => {
        for (let o = 0; o < L.out4; o++) {
            const rows = [0, 1, 2, 3].map((r) => {
                const row = o * 4 + r;
                if (row >= L.out) return '0.0';
                return Array.from({ length: L.in4 }, (_, k) => `dot(${w}[${L.w + row * L.in4 + k}], x${l}_${k})`).join(' + ');
            });
            s += `    let x${l + 1}_${o} = tanh(${w}[${L.bias + o}] + vec4<f32>(\n        ${rows.join(',\n        ')}));\n`;
        }
    });
    const y = `x${layout.length}_0`;
    return `${s}    act[i * 2u] = ${y}.x;\n    act[i * 2u + 1u] = ${y}.y;`;
}

const GATES = {
    all: '',
    index: `if ((i % ${THINK_EVERY}u) != 0u) { return; }`,
    block: `if (((i / 64u) % ${THINK_EVERY}u) != 0u) { return; }`,
};
const thinks = (gate, i) => gate === 'all' || (gate === 'index' ? i % THINK_EVERY === 0 : Math.floor(i / 64) % THINK_EVERY === 0);

function mlpWGSL({ source, style, gate }) {
    const w = source === 'workgroup' ? 'staged' : 'weights';
    const stage = source === 'workgroup'
        ? `for (var k = li; k < ${QUADS}u; k += 64u) { staged[k] = weights[k]; }\n    workgroupBarrier();\n    ` : '';
    return /* wgsl */ `
@group(0) @binding(0) var<storage, read> obs: array<f32>;
@group(0) @binding(1) var<storage, read_write> act: array<f32>;
${weightDecl(source)}
@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    ${stage}let i = id.x;
    if (i >= arrayLength(&act) / 2u) { return; }
    ${GATES[gate]}
${style === 'loop' ? loopBody(w) : unrolledBody(w)}
}`;
}

// The floor: read the same 24 inputs, write the same 2 outputs, no network.
const COPY_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> obs: array<f32>;
@group(0) @binding(1) var<storage, read_write> act: array<f32>;
@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&act) / 2u) { return; }
    var s = 0.0;
    for (var k = 0u; k < ${LAYERS[0]}u; k++) { s += obs[i * ${LAYERS[0]}u + k]; }
    act[i * 2u] = s;
    act[i * 2u + 1u] = -s;
}`;

const TANH_INPUTS = [-1e4, -1000, -100, -89, -50, -20, -10, -5, -1, -0.5, 0, 0.5, 1, 5, 10, 20, 50, 89, 100, 1000, 1e4];
const TANH_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read_write> ys: array<f32>;
@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&xs)) { ys[id.x] = tanh(xs[id.x]); }
}`;

const SOURCES = ['storage', 'fixed', 'uniform', 'workgroup'];
const KERNELS = SOURCES.flatMap(source => ['loop', 'unrolled'].map(style => ({ source, style, gate: 'all' })));
const GATED = [{ source: 'storage', style: 'loop' }, { source: 'storage', style: 'unrolled' }]
    .flatMap(k => ['index', 'block'].map(gate => ({ ...k, gate })));
const kernelName = k => `${k.source} · ${k.style}${k.gate === 'all' ? '' : ` · ${k.gate}`}`;

const median = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
const fmt = (n) => n.toLocaleString('en-US');

// ---------------------------------------------------------------------------------------------
// One adapter
// ---------------------------------------------------------------------------------------------
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
    async function pipeline(code, label) {
        const t0 = performance.now();
        try {
            const p = await device.createComputePipelineAsync({ label, layout: 'auto',
                compute: { module: device.createShaderModule({ label, code }), entryPoint: 'cs_main' } });
            return { p, ms: performance.now() - t0 };
        } catch (e) {
            check(`${name}: ${label} compiles`, false, String(e?.message ?? e).split('\n')[0]);
            return null;
        }
    }

    const weightsStorage = buffer(QUADS * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, packed);
    const weightsUniform = buffer(QUADS * 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, packed);
    const io = new Map(COUNTS.map(n => [n, {
        obs: buffer(n * LAYERS[0] * 4, STORAGE, obs.subarray(0, n * LAYERS[0])),
        act: buffer(n * 2 * 4, STORAGE),
    }]));
    const bindings = new Map();
    const bindFor = (p, source, n) => {
        const key = `${source}:${n}`;
        let bg = bindings.get(p)?.get(key);
        if (bg) return bg;
        bg = device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [
            { binding: 0, resource: { buffer: io.get(n).obs } },
            { binding: 1, resource: { buffer: io.get(n).act } },
            ...(source ? [{ binding: 2, resource: { buffer: source === 'uniform' ? weightsUniform : weightsStorage } }] : []),
        ] });
        if (!bindings.has(p)) bindings.set(p, new Map());
        bindings.get(p).set(key, bg);
        return bg;
    };

    const ts = timestamps ? {
        set: device.createQuerySet({ type: 'timestamp', count: 2 }),
        resolve: buffer(16, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC),
        read: buffer(16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ),
    } : null;
    const rawDeltas = [];

    async function timedPass(p, bind, groups, repeats) {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass(ts
            ? { timestampWrites: { querySet: ts.set, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
        pass.setPipeline(p);
        pass.setBindGroup(0, bind);
        for (let r = 0; r < repeats; r++) pass.dispatchWorkgroups(groups);
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

    // Median per-dispatch time over TRIALS passes of `repeats` back-to-back dispatches, with
    // `repeats` chosen so one pass holds about PASS_NS of work. GPU timestamps when the adapter
    // has them, wall-clock submit→done otherwise.
    async function measure(p, source, n) {
        const bind = bindFor(p, source, n), groups = Math.ceil(n / 64);
        const probe = await timedPass(p, bind, groups, 4);
        const per = (probe.gpuNs ?? probe.wallNs) / 4;
        const repeats = Math.max(8, Math.min(4000, Math.ceil(PASS_NS / Math.max(per, 250))));
        const gpu = [], wall = [];
        for (let t = 0; t < TRIALS; t++) {
            const r = await timedPass(p, bind, groups, repeats);
            if (r.gpuNs !== null) { gpu.push(r.gpuNs / repeats); rawDeltas.push(r.gpuNs); }
            wall.push(r.wallNs / repeats);
        }
        return gpu.length === TRIALS ? median(gpu) / 1e3 : median(wall) / 1e3;
    }

    async function verify(p, k) {
        const { act } = io.get(CHECK_N);
        device.queue.writeBuffer(act, 0, new Float32Array(CHECK_N * 2).fill(SENTINEL));
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(p);
        pass.setBindGroup(0, bindFor(p, k.source, CHECK_N));
        pass.dispatchWorkgroups(Math.ceil(CHECK_N / 64));
        pass.end();
        device.queue.submit([enc.finish()]);
        const got = await readF32(act, CHECK_N * 2);
        let maxErr = 0, touched = 0, skipped = 0;
        for (let i = 0; i < CHECK_N; i++) {
            const on = thinks(k.gate, i);
            if (!on) skipped++;
            for (let c = 0; c < 2; c++) {
                const v = got[i * 2 + c];
                if (!on) { if (v !== SENTINEL) touched++; continue; }
                const e = Math.abs(v - expected[i * 2 + c]);
                maxErr = Number.isNaN(e) ? Infinity : Math.max(maxErr, e);
            }
        }
        check(`${name}: ${kernelName(k)} = CPU forward pass`, maxErr <= TOLERANCE && touched === 0,
            `max error ${maxErr.toExponential(1)}` + (k.gate === 'all' ? '' : `, ${fmt(skipped)} skipped rows, ${touched} touched`));
        return maxErr;
    }

    const result = { adapter: name, host: inBrowser ? 'chrome' : 'node', timing: timestamps ? 'timestamps' : 'wall-clock',
        quads: QUADS, weightBytes: QUADS * 16, kernels: {}, gated: {}, tanh: {} };

    const copy = await pipeline(COPY_WGSL, 'copy');
    if (copy) result.kernels.copy = { compileMs: copy.ms, us: {} };
    const compiled = [];
    for (const k of [...KERNELS, ...GATED]) {
        const c = await pipeline(mlpWGSL(k), kernelName(k));
        if (!c) continue;
        compiled.push({ k, ...c, maxErr: await verify(c.p, k) });
    }

    // Without a warm-up, the first run's 1,000-agent column on NVIDIA read 10–25× too slow until
    // the clocks rose partway through it.
    const heavy = compiled.find(c => c.k.gate === 'all');
    if (heavy) {
        const bind = bindFor(heavy.p, heavy.k.source, MAX_N), t0 = performance.now();
        while (performance.now() - t0 < WARMUP_MS) await timedPass(heavy.p, bind, Math.ceil(MAX_N / 64), 20);
    }
    const keepLower = (entry, n, us) => { entry.us[n] = Math.min(entry.us[n] ?? Infinity, us); };
    for (let round = 0; round < ROUNDS; round++) {
        for (const n of COUNTS) {
            if (copy) keepLower(result.kernels.copy, n, await measure(copy.p, null, n));
            for (const c of compiled) {
                if (c.k.gate !== 'all' && !GATED_COUNTS.includes(n)) continue;
                const slot = c.k.gate === 'all' ? result.kernels : result.gated;
                const entry = slot[kernelName(c.k)] ??= { compileMs: c.ms, maxErr: c.maxErr, us: {} };
                keepLower(entry, n, await measure(c.p, c.k.source, n));
            }
        }
    }
    result.quantized = rawDeltas.length > 0 && rawDeltas.every(d => d % 100_000 === 0);

    // tanh at large inputs
    const xs = buffer(TANH_INPUTS.length * 4, STORAGE, new Float32Array(TANH_INPUTS));
    const ys = buffer(TANH_INPUTS.length * 4, STORAGE);
    const probe = await pipeline(TANH_WGSL, 'tanh probe');
    if (probe) {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(probe.p);
        pass.setBindGroup(0, device.createBindGroup({ layout: probe.p.getBindGroupLayout(0), entries: [
            { binding: 0, resource: { buffer: xs } }, { binding: 1, resource: { buffer: ys } }] }));
        pass.dispatchWorkgroups(1);
        pass.end();
        device.queue.submit([enc.finish()]);
        const got = await readF32(ys, TANH_INPUTS.length);
        TANH_INPUTS.forEach((x, j) => { result.tanh[x] = got[j]; });
        result.tanhBad = TANH_INPUTS.filter((x, j) => !(Math.abs(got[j] - Math.tanh(x)) <= 1e-5));
    }

    for (const b of made) b.destroy();
    ts?.set.destroy();
    device.destroy();
    return result;
}

// ---------------------------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------------------------
function report(r) {
    const timing = r.timing === 'timestamps' ? `GPU timestamps${r.quantized ? ', quantized to 100 µs' : ''}` : 'wall-clock';
    log(`\n== ${r.host === 'node' ? 'Node/Dawn' : 'Chrome'} · ${r.adapter} — µs per dispatch, ` +
        `lower of ${ROUNDS} medians of ${TRIALS}, after a ${WARMUP_MS} ms warm-up (${timing})`);
    log(`   network ${LAYERS.join('→')}, tanh; ${r.quads} vec4 of weights (${fmt(r.weightBytes)} B); one invocation per agent, workgroup 64`);
    const head = (counts) => counts.map(n => fmt(n).padStart(10)).join('');
    log(`   ${'kernel'.padEnd(26)}${head(COUNTS)}   compile   max error`);
    for (const [name, e] of Object.entries(r.kernels)) {
        const err = e.maxErr === undefined ? '' : e.maxErr.toExponential(1);
        log(`   ${name.padEnd(26)}${COUNTS.map(n => (e.us[n] ?? NaN).toFixed(1).padStart(10)).join('')}` +
            `   ${e.compileMs.toFixed(0).padStart(5)} ms   ${err}`);
    }
    log(`   thinkEvery ${THINK_EVERY}, 1 in ${THINK_EVERY} agents thinks:`);
    for (const [name, e] of Object.entries(r.gated)) {
        const all = r.kernels[name.replace(/ · (index|block)$/, '')];
        log(`   ${name.padEnd(26)}${GATED_COUNTS.map(n => e.us[n].toFixed(1).padStart(10)).join('')}` +
            `   = ${GATED_COUNTS.map(n => `${Math.round((100 * e.us[n]) / all.us[n])}%`).join(' / ')} of all thinking`);
    }
    const best = Object.entries(r.kernels).filter(([n]) => n !== 'copy').sort((a, b) => a[1].us[20_000] - b[1].us[20_000])[0];
    if (best) {
        log(`   fastest at 20,000: ${best[0]} — ${best[1].us[20_000].toFixed(1)} µs, ` +
            `${((best[1].us[100_000] * 1e3) / 100_000).toFixed(2)} ns per agent at 100,000`);
    }
    log(r.tanhBad?.length
        ? `   FINDING: tanh is wrong for x = ${r.tanhBad.join(', ')} (got ${r.tanhBad.map(x => r.tanh[x]).join(', ')}) — clamp pre-activations`
        : `   tanh is finite and within 1e-5 of Math.tanh for every probe input, |x| up to 1e4`);
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
    log(`S3: measuring on ${name}${i.description ? ` (${i.description})` : ''}…`);
    const r = await runAdapter(adapter, name);
    results.push(r);
    report(r);
}
if (!results.length) {
    fatal('No WebGPU adapter');
    throw new Error('No WebGPU adapter');
}
log(`\nS3 JSON ${JSON.stringify(results)}`);
await finish(`S3: every kernel matched the CPU forward pass on ${results.length} adapter(s).`);
log('S3 done');
