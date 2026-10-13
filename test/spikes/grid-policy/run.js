// Step 5, phase 5.7 (docs/agents-ordered-grid-design.md): what each crowded-cell policy costs on the
// acceptance workload, and what production costs against the code before step 5.
// A measurement, not a suite; production code is not touched.
//
// Phase 5.7 measured the whole ordered-grid change hours after its 5.5 baseline, on a run that keeps
// evolving, so it could not say which part of the change costs what. Here every case starts from
// one frozen snapshot (the acceptance spike's state: Herd & Wolves' setup, three steps after spawn),
// and these chains are sampled in turn, restoring the snapshot before each sample:
//
//   old      the grid (atomic scatter) and query (first entries) before step 5, bundled from b7050b7
//   first    today's sorted grid; a crowded cell yields its first entries (no window)
//   agent    today's sorted grid; a window that rotates per querying agent and tick: phase 5.6's
//            query, bundled from da262ad
//   agent+   the same entries in the same order as agent, scanned by a tighter loop
//   cell     today's sorted grid; a window that rotates per cell and tick, in phase 5.6's loop
//   cell+    the same entries in the same order as cell, scanned by the tighter loop
//   now      today's production step, untouched. Since phase 5.7 its query is cell+'s but for one
//            parameter's name, so cell+ against now is also the noise between two identical chains
//
// Only the grid build and the two query stages differ between chains; sense, think, act and tick
// are the system's own stages in all of them. Before anything is timed, every query is checked
// against a CPU scan of the grid it read, and against the wrong grid being emptied.
//
// Timed: the grid build, the two queries on the frozen snapshot, a few steps from the snapshot,
// and a longer run from it. The longer run matters because neighbours differ by policy, so herds
// move differently, and a step costs what the crowding it meets costs.
//
//   node build.js && node test/spikes/grid-policy/prepare.mjs    (the earlier versions, into dist/)
//   node --test test/spikes/grid-policy/workload.test.mjs
//   node test/gpu/run-harness.mjs test/spikes/grid-policy/run.js __GRID_POLICY_RESULTS__
//   npm run drive -- --serve . test/spikes/grid-policy/index.html --wait-for "Grid policy done" --timeout 590
//
// NG_POLICY_ADAPTER=nvidia|amd (?adapter= in the page) picks one adapter. NG_POLICY_OUTPUT=<prefix>
// saves each adapter's JSON as soon as it completes. NG_POLICY_LAYOUTS=packed (?layouts=) runs the
// named layouts instead of the acceptance spike's two. Run one host at a time.
import { harnessEnv } from '../../gpu/harness-env.mjs';
import { NullGraph } from '../../../dist/index.js';
import * as A from '../../../dist/agents.js';
import * as B from '../../../dist/grid-policy-before.js';
import { buildAgentGridQueryWGSL as windowQuery } from '../../../dist/grid-policy-window.js';
import { RULES, herdDef, spawn, PAD, nearestSensors, policyQueryWGSL, gpuCell, sortedGrid, crowding, scanned,
    matches } from './workload.mjs';

const { log, check, fatal, finish, inBrowser } = harnessEnv('__GRID_POLICY_RESULTS__');
const env = globalThis.process?.env ?? {}, page = inBrowser ? new URLSearchParams(location.search) : null;
const only = inBrowser ? page.get('adapter') : env.NG_POLICY_ADAPTER;
const layouts = (inBrowser ? page.get('layouts') : env.NG_POLICY_LAYOUTS)?.split(',');

// spread and flocked are the acceptance spike's layouts. packed is not: it is the pile-up a rule
// without wall avoidance falls into, and the worst case for a crowded cell.
const CASES = [[4_096, 'spread'], [4_096, 'flocked'], [65_536, 'spread'], [65_536, 'flocked'], [65_536, 'packed']]
    .filter(([, layout]) => layouts ? layouts.includes(layout) : layout !== 'packed');
// Dawn quantizes timestamps to 65.5 µs, so a timed pass must be long; Windows resets a GPU that stays
// busy for about 2 s, so it must not be too long. By agent count:
const STEPS = { 4096: 16, 65536: 2 };        // steps in one "few steps" pass
const BUILDS = { 4096: 256, 65536: 32 };     // grid builds in one pass
const TRIALS = { 4096: 15, 65536: 9 };       // samples of each chain; the median is reported
// The longer run: `passes` passes of `each` steps, averaged over `trials` runs.
const RUN = { 4096: { passes: 16, each: 32, trials: 3 }, 65536: { passes: 32, each: 2, trials: 2 } };
const QUERY_PASS_MS = 48;    // the frozen queries repeat until a pass lasts about this, 64 times at most
const WARMUP_MS = 500;       // a GPU idles at low clocks and boosts only under sustained load

const stats = xs => {
    const s = [...xs].sort((a, b) => a - b);
    return { median: s[s.length >> 1], min: s[0], max: s.at(-1), samples: xs };
};
const same = (a, b) => a.length === b.length && a.every((w, i) => w === b[i]);
const view = words => ({ data: new Float32Array(words.buffer), u32: words });
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;

// packed: four knots of count / 4 agents, each 12 m across, so a few cells hold thousands.
function pack(agents, def) {
    const centre = (def.world.maxX - 1) / 2, r = A.rand01;
    for (let i = 0; i < def.capacity; i++) {
        const knot = i % 4;
        agents.spawn(i, { position: [(knot % 2 ? 1 : -1) * centre + (r(i, 11, 2) - 0.5) * 12, 0.08,
            (knot < 2 ? 1 : -1) * centre + (r(i, 11, 3) - 0.5) * 12],
        species: i < def.capacity / 16 ? 'wolf' : 'prey', heading: r(i, 11, 4) * Math.PI * 2 - Math.PI });
    }
    agents.upload();
}
const fmt = n => n.toLocaleString('en-US');
const versus = (s, base) => `${s.median >= base.median ? '+' : ''}${((s.median / base.median - 1) * 100).toFixed(1)}%`;

async function runAdapter(adapter, name) {
    const timestamps = adapter.features.has('timestamp-query');
    const device = await adapter.requestDevice({ requiredFeatures: timestamps ? ['timestamp-query'] : [] });
    const engine = new NullGraph();
    const init = await engine.init(null, { device, width: 64, height: 64 });
    if (!init.success) throw new Error(init.error);
    const set = timestamps ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null;
    const resolve = set && device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const staging = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    function stage(label, code, buffers, count) {
        const node = engine.createComputeStage({ name: label, shaderCode: code, dispatchCount: [Math.ceil(count / 64), 1, 1] });
        engine.setComputePassBindGroup(node, buffers.map((buffer, binding) => ({ binding, resource: { buffer } })), 0);
        return node;
    }
    /** GPU milliseconds of one compute pass replaying `chain` `repeats` times, batched as engine.render() batches it. */
    async function timed(chain, repeats) {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass(set ? { timestampWrites: { querySet: set, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
        for (let r = 0; r < repeats; r++) {
            for (const node of chain) {
                if (!node.enabled || node.preClearBuffers.length || node.indirectBuffer) {
                    throw new Error(`${node.name} needs more than a dispatch; the replay would not match`);
                }
                pass.setPipeline(node.computePipeline);
                for (const g in node.customBindGroups) pass.setBindGroup(Number(g), node.customBindGroups[g]);
                pass.dispatchWorkgroups(...node.dispatchCount);
            }
        }
        pass.end();
        if (set) {
            enc.resolveQuerySet(set, 0, 2, resolve, 0);
            enc.copyBufferToBuffer(resolve, 0, staging, 0, 16);
        }
        const t0 = performance.now();
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
        const wallMs = performance.now() - t0;
        if (!set) return wallMs;
        await staging.mapAsync(GPUMapMode.READ);
        const [begin, end] = new BigUint64Array(staging.getMappedRange().slice(0));
        staging.unmap();
        return Number(end - begin) / 1e6;
    }
    async function read(buffer) {
        const dst = device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(buffer, 0, dst, 0, buffer.size);
        device.queue.submit([enc.finish()]);
        await dst.mapAsync(GPUMapMode.READ);
        const words = new Uint32Array(dst.getMappedRange().slice(0));
        dst.unmap();
        dst.destroy();
        return words;
    }
    /** Milliseconds per repeat of every chain, sampled in turn: the order rotates and reverses, so
     * drift favours no chain. One unrecorded pass each comes first. */
    async function measure(chains, repeats, trials, prepare = () => {}, after = async () => {}) {
        const keys = Object.keys(chains), samples = Object.fromEntries(keys.map(key => [key, []]));
        for (const key of keys) {
            prepare();
            await timed(chains[key], repeats);
            await after(key);
        }
        for (let t = 0; t < trials; t++) {
            const order = [...keys.slice(t % keys.length), ...keys.slice(0, t % keys.length)];
            if (t % 2) order.reverse();
            for (const key of order) {
                prepare();
                samples[key].push(await timed(chains[key], repeats) / repeats);
                await after(key);
            }
        }
        return Object.fromEntries(keys.map(key => [key, stats(samples[key])]));
    }

    const rows = [];
    for (const [count, layout] of CASES) {
        const label = `${name} ${fmt(count)}/${layout}`;
        device.pushErrorScope('validation');
        const def = herdDef(count, 64), agents = new A.AgentSystem(engine, def, { thinkEvery: 3 });
        agents.setBrain(new A.RuleBrain(RULES));
        if (layout === 'packed') pack(agents, def);
        else spawn(agents, count, layout, 11);
        for (let s = 0; s < 3; s++) await agents.step();
        agents.advance(def.dt);
        const step = agents.getComputeStages(), ordered = agents.grid, sorted = ordered.getComputeStages();
        const atomic = new B.PointGrid(engine, agents.buffer, { ...def, stride: A.AGENT_FLOATS,
            flagsOffset: A.AGENT_FLAGS_OFFSET, aliveMask: A.AGENT_FLAG_ALIVE });
        const scatter = atomic.getComputeStages();

        // The snapshot every sample starts from. reason and neighbours cannot be written, and
        // need not be: no stage reads reason, and the queries rewrite neighbours before sense.
        const snap = { records: await read(agents.buffer), obs: await read(agents.obs),
            act: await read(agents.act), state: await read(agents.state) };
        const restore = (tick = snap.state[0]) => {
            device.queue.writeBuffer(agents.buffer, 0, snap.records);
            device.queue.writeBuffer(agents.obs, 0, snap.obs);
            device.queue.writeBuffer(agents.act, 0, snap.act);
            device.queue.writeBuffer(agents.state, 0, new Uint32Array([tick, snap.state[1], snap.state[2], snap.state[3]]));
        };

        const sensors = nearestSensors(def), stride = agents.neighbourStride;
        const queries = (policy, grid) => sensors.map(sensor => stage(`${policy} ${sensor.name}`,
            policyQueryWGSL(policy, def, sensor, stride, { before: B.buildAgentGridQueryWGSL, window: windowQuery }),
            [agents.buffer, grid.cellCounts, grid.cellOffsets, grid.cellBodies, agents.neighbours,
                ...(policy === 'first' ? [] : [agents.state])], count));
        const production = step.filter(n => n.name.startsWith('AgentSystem query '));
        const rest = step.filter(n => !sorted.includes(n) && !production.includes(n));
        const query = { old: queries('first', atomic), first: queries('first', ordered), agent: queries('agent', ordered),
            'agent+': queries('agent+', ordered), cell: queries('cell', ordered), 'cell+': queries('cell+', ordered),
            now: production };
        const policyOf = { old: 'first', first: 'first', agent: 'agent', 'agent+': 'agent', cell: 'cell', 'cell+': 'cell',
            now: 'cell' };
        const chains = Object.fromEntries(Object.keys(query).map(key =>
            [key, [...(key === 'old' ? scatter : sorted), ...query[key], ...rest]]));
        check(`${label}: the now chain is the production step, stage for stage`,
            production.length === sensors.length && same(chains.now, step));

        // 1. Each query returns what its policy says, at the snapshot's tick and at another.
        const rec = view(snap.records), mirror = sortedGrid(def, rec);
        // The library's CPU pointCell divides in f64 and the GPU in f32, so an agent within float
        // rounding of a cell edge is binned differently by the two. Counted, not hidden.
        const edge = [];
        for (let i = 0; i < count; i++) {
            const x = rec.data[i * A.AGENT_FLOATS], z = rec.data[i * A.AGENT_FLOATS + 2];
            if (gpuCell(def, x, z) !== A.pointCell(def, x, z)) edge.push(`#${i} at (${x}, ${z})`);
        }
        const readGrid = async grid => ({ cellCounts: await read(grid.cellCounts),
            cellOffsets: await read(grid.cellOffsets), cellBodies: await read(grid.cellBodies) });
        let crowd, changed;
        for (const tick of [snap.state[0], 37]) {
            restore(tick);
            await timed([...sorted, ...scatter], 1);
            const grids = { sorted: await readGrid(ordered), atomic: await readGrid(atomic) };
            check(`${label} tick ${tick}: the sorted grid lists each cell in index order`,
                same(grids.sorted.cellCounts, mirror.cellCounts) && same(grids.sorted.cellOffsets, mirror.cellOffsets) &&
                same(grids.sorted.cellBodies, mirror.cellBodies),
                edge.length ? `${edge.length} agent(s) on a cell edge that the library's f64 pointCell bins elsewhere: ${edge.slice(0, 3).join('; ')}` : '');
            crowd ??= crowding(def, rec, grids.sorted);
            const lists = {};
            for (const [key, nodes] of Object.entries(query)) {
                await timed(nodes, 1);
                lists[key] = await read(agents.neighbours);
                const grid = key === 'old' ? grids.atomic : grids.sorted, tally = { exact: 0, tie: 0, wrong: 0 };
                for (let i = 0; i < count; i++) {
                    for (const sensor of sensors) {
                        const at = i * stride + sensor.base;
                        tally[matches(scanned(def, rec, grid, i, sensor, policyOf[key], tick),
                            lists[key].subarray(at, at + sensor.k), sensor.k, def.cellSize ** 2)]++;
                    }
                }
                check(`${label} tick ${tick}: ${key} returns the "${policyOf[key]}" policy's neighbours`,
                    tally.wrong === 0 && tally.exact > 0,
                    `${fmt(tally.exact)} lists exact, ${tally.tie} float near-ties, ${tally.wrong} wrong`);
            }
            check(`${label} tick ${tick}: the production query equals cell+ word for word`, same(lists.now, lists['cell+']));
            check(`${label} tick ${tick}: cell+ equals cell, and agent+ equals agent, word for word`,
                same(lists['cell+'], lists.cell) && same(lists['agent+'], lists.agent));
            const differ = key => {
                let n = 0;
                for (let i = 0; i < count; i++) {
                    if (!same(lists[key].subarray(i * stride, (i + 1) * stride), lists.first.subarray(i * stride, (i + 1) * stride))) n++;
                }
                return n;
            };
            const moved = { old: differ('old'), cell: differ('cell'), agent: differ('agent') };
            changed ??= moved;
            if (!crowd.cells) {
                check(`${label} tick ${tick}: with no crowded cell, every policy and the old grid agree word for word`,
                    moved.old + moved.cell + moved.agent === 0);
            } else {
                check(`${label} tick ${tick}: with crowded cells, the windows change what agents see`,
                    moved.cell > 0 && moved.agent > 0,
                    `cell ${fmt(moved.cell)}, agent ${fmt(moved.agent)} of ${fmt(count)} agents`);
            }
        }

        // 2. Each query reads the grid its own chain builds: emptying the other grid changes nothing.
        restore();
        await timed([...sorted, ...scatter], 1);
        const before = {}, empty = new Uint32Array(def.grid.cellTotal);
        const ask = async key => { await timed(query[key], 1); return read(agents.neighbours); };
        for (const key of Object.keys(query)) before[key] = await ask(key);
        device.queue.writeBuffer(ordered.cellCounts, 0, empty);
        check(`${label}: old reads the atomic grid, not the sorted one`,
            same(await ask('old'), before.old) && (await ask('first')).every(w => w === PAD));
        await timed(sorted, 1);
        device.queue.writeBuffer(atomic.cellCounts, 0, empty);
        for (const key of Object.keys(query).filter(key => key !== 'old')) {
            check(`${label}: ${key} reads the sorted grid, not the atomic one`, same(await ask(key), before[key]));
        }
        check(`${label}: emptying the atomic grid empties the old query`, (await ask('old')).every(w => w === PAD));

        // 3. Timings. Warm up on the grid build alone: it moves nobody.
        restore();
        const t0 = performance.now();
        while (performance.now() - t0 < WARMUP_MS) await timed(sorted, 8);
        const builds = await measure({ atomic: scatter, sorted }, BUILDS[count], TRIALS[count]);
        await timed([...sorted, ...scatter], 1);
        const repeats = Math.max(1, Math.min(64, Math.round(QUERY_PASS_MS / await timed(query.now, 1))));
        const frozen = await measure(query, repeats, TRIALS[count]);
        const ends = Object.fromEntries(Object.keys(chains).map(key => [key, new Set()]));
        let overflowSteps = 0;
        const steps = await measure(chains, STEPS[count], TRIALS[count], restore, async key => {
            const state = await read(agents.state);
            ends[key].add(state[0]);
            if (key === 'now') overflowSteps = state[1] - snap.state[1];
        });
        for (const key of Object.keys(chains)) {
            check(`${label}: every ${key} sample ran ${STEPS[count]} steps from the snapshot`,
                ends[key].size === 1 && ends[key].has(snap.state[0] + STEPS[count]), `ended at tick ${[...ends[key]].join(', ')}`);
        }

        // 4. A longer run. Neighbours differ by policy, so the herds move differently, and a step
        // costs what the crowding it meets costs. Each chain runs on from the snapshot, pass by pass.
        const { passes, each, trials } = RUN[count], keys = Object.keys(chains), ran = passes * each;
        const series = Object.fromEntries(keys.map(key => [key, Array.from({ length: passes }, () => [])]));
        const final = {}, repeated = {}, endCrowd = {};
        for (let t = 0; t < trials; t++) {
            for (const key of t % 2 ? [...keys].reverse() : keys) {
                restore();
                for (let p = 0; p < passes; p++) series[key][p].push(await timed(chains[key], each) / each);
                const records = await read(agents.buffer), tick = (await read(agents.state))[0];
                if (tick !== snap.state[0] + ran) throw new Error(`${label}: the ${key} run ended at tick ${tick}`);
                if (final[key]) {
                    repeated[key] = (repeated[key] ?? true) && same(records, final[key]);
                } else {
                    final[key] = records;
                    endCrowd[key] = crowding(def, view(records), sortedGrid(def, view(records)));
                }
            }
        }
        check(`${label}: after ${ran} more steps, the production run holds the cell+ run's records word for word`,
            same(final.now, final['cell+']));
        check(`${label}: after ${ran} more steps, cell+ holds the cell run's records, and agent+ the agent run's, word for word`,
            same(final['cell+'], final.cell) && same(final['agent+'], final.agent));
        for (const key of ['first', 'agent', 'now']) {
            check(`${label}: the ${key} run repeats word for word over ${ran} steps`, repeated[key] === true);
        }
        log(`   ${label}: the old (atomic) run ${repeated.old ? 'also repeated' : 'did NOT repeat'} over ${ran} steps`);
        const run = { passes, stepsPerPass: each, trials, oldRepeated: repeated.old === true, end: endCrowd,
            ms: Object.fromEntries(keys.map(key => [key, series[key].map(mean)])), samples: series };

        rows.push({ count, layout, tick: snap.state[0], stepsPerSample: STEPS[count], queryRepeats: repeats,
            buildRepeats: BUILDS[count], trials: TRIALS[count], cells: def.grid.cellTotal, crowd, changed, edge,
            overflowSteps, builds, queries: frozen, step: steps, run });
        log(`${label}: step ms ` + Object.entries(steps).map(([key, s]) => `${key}=${s.median.toFixed(3)}`).join(' '));
        for (const key of Object.keys(query).filter(key => key !== 'now')) {
            for (const node of query[key]) node.enabled = false;
        }
        atomic.destroy();
        agents.destroy();
        const error = await device.popErrorScope();
        check(`${label}: no validation error`, error === null, error?.message ?? '');
    }
    set?.destroy(); resolve?.destroy(); staging.destroy();
    device.destroy();
    return { adapter: name, host: inBrowser ? 'chrome' : 'node', timing: timestamps ? 'timestamps' : 'wall-clock', rows };
}

function report(r) {
    log(`\n== ${r.host === 'node' ? 'Node/Dawn' : 'Chrome'} · ${r.adapter} (${r.timing})`);
    const note = { old: 'before step 5', first: 'sorted, first entries', agent: 'sorted, window per agent (phase 5.6)',
        'agent+': 'the same, tighter loop', cell: 'sorted, window per cell', 'cell+': 'the same, tighter loop',
        now: 'production today (cell+ again)' };
    for (const row of r.rows) {
        const c = row.crowd;
        log(`\n   ${fmt(row.count)} ${row.layout}: ${c.cells} of ${fmt(row.cells)} cells over budget (largest ${c.largest}), ` +
            `${(100 * c.agents).toFixed(0)}% of agents scan one; overflow in ${row.overflowSteps} of ${row.stepsPerSample} steps`);
        log(`   ${'chain'.padEnd(8)}${'step ms'.padStart(9)}${'min..max'.padStart(19)}${'vs old'.padStart(9)}` +
            `${'queries ms'.padStart(12)}${'vs old'.padStart(9)}   what`);
        for (const [key, s] of Object.entries(row.step)) {
            const q = row.queries[key];
            log(`   ${key.padEnd(8)}${s.median.toFixed(3).padStart(9)}${`${s.min.toFixed(3)}..${s.max.toFixed(3)}`.padStart(19)}` +
                `${(key === 'old' ? '' : versus(s, row.step.old)).padStart(9)}${q.median.toFixed(3).padStart(12)}` +
                `${(key === 'old' ? '' : versus(q, row.queries.old)).padStart(9)}   ${note[key]}`);
        }
        log(`   grid build: atomic ${row.builds.atomic.median.toFixed(3)} ms, sorted ${row.builds.sorted.median.toFixed(3)} ms; ` +
            `neighbour lists that differ from "first": cell ${fmt(row.changed.cell)}, agent ${fmt(row.changed.agent)}`);
        const run = row.run, total = run.passes * run.stepsPerPass, quarter = run.passes / 4;
        log(`   running on for ${total} steps (ms per step, by quarter of the run; then the crowding it ends in):`);
        log(`   ${'chain'.padEnd(8)}${[1, 2, 3, 4].map(q => `steps ≤${q * total / 4}`.padStart(12)).join('')}${'whole run'.padStart(11)}` +
            `${'vs old'.padStart(9)}   cells over budget, agents scanning one`);
        for (const [key, ms] of Object.entries(run.ms)) {
            const whole = mean(ms), end = run.end[key];
            log(`   ${key.padEnd(8)}${[0, 1, 2, 3].map(q => mean(ms.slice(q * quarter, (q + 1) * quarter)).toFixed(3).padStart(12)).join('')}` +
                `${whole.toFixed(3).padStart(11)}${(key === 'old' ? '' : versus({ median: whole }, { median: mean(run.ms.old) })).padStart(9)}` +
                `   ${end.cells} (largest ${end.largest}), ${(100 * end.agents).toFixed(0)}%`);
        }
    }
}

const results = [], seen = new Set();
for (const powerPreference of ['high-performance', 'low-power']) {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference });
    if (!adapter) continue;
    const info = adapter.info ?? {}, key = [info.vendor, info.architecture, info.device, info.description].join('|');
    if (seen.has(key)) continue;
    seen.add(key);
    const name = [info.vendor, info.architecture].filter(Boolean).join(' ') || powerPreference;
    if (only && !name.includes(only)) continue;
    log(`Grid policy: measuring on ${name}${info.description ? ` (${info.description})` : ''}…`);
    const result = await runAdapter(adapter, name);
    results.push(result);
    report(result);
    if (!inBrowser && env.NG_POLICY_OUTPUT) {
        const { writeFileSync } = await import('node:fs');
        writeFileSync(`${env.NG_POLICY_OUTPUT}-${name.replace(/\W+/g, '-')}.json`, JSON.stringify(result, null, 2));
    }
}
if (!results.length) {
    fatal('No WebGPU adapter');
    throw new Error('No WebGPU adapter');
}
log(`\nGrid policy JSON ${JSON.stringify(results)}`);
await finish(`Grid policy: measured on ${results.length} adapter(s).`);
log('Grid policy done');
