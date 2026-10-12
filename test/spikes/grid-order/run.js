// Step 5, phase 5.5 (docs/agents-ordered-grid-design.md §6, §9): today's grid, measured before a
// stable scatter replaces its atomic one, and measured again by the same script afterwards (5.7).
//
// A measurement, not a test suite: it is not registered in test/gpu/suites.mjs, and its numbers are
// recorded in the design. On every adapter:
//   1. Divergence (A23's scenario): two identical lockstep runs of Herd & Wolves' setup at 4,096
//      agents with a scan budget of 8, so most cells overflow. Counts the record, observation, action
//      and reason words that differ after 200 steps. Today's atomic scatter is expected to diverge;
//      a host that does not is reported, not hidden.
//   2. Cost: GPU time of a whole step and of the grid build alone, at 4,096 and 65,536 agents, with
//      Herd & Wolves' density, budget 64 and thinkEvery 3, spawned spread (the demo's spawn) or
//      flocked (herds that crowd cells). The engine's own stage nodes are replayed in one timed
//      compute pass, as engine.render() batches them, so CPU encoding is excluded.
//
//   Node + Dawn, on every adapter:
//     node test/gpu/run-harness.mjs test/spikes/grid-order/run.js __GRID_ORDER_RESULTS__
//   Chrome:
//     npm run drive -- --serve . test/spikes/grid-order/index.html --wait-for "Grid order done" --timeout 590
import { harnessEnv } from '../../gpu/harness-env.mjs';
import * as A from '../../../dist/agents.js';
import { NullGraph } from '../../../dist/index.js';

const { log, check, fatal, finish, inBrowser } = harnessEnv('__GRID_ORDER_RESULTS__');

// Herd & Wolves (the test-engine site's HerdWolvesGPU): a 160 m field at 4,096 agents that grows
// with the count, 8 m cells, one wolf in 16 (the lowest indices), the demo's default speeds.
const BASE_HALF = 80, CELL = 8, WOLF_SHARE = 16;
const COUNTS = [4_096, 65_536];
const LAYOUTS = ['spread', 'flocked'];
const DIVERGE = { count: 4_096, budget: 8, steps: 200 };
const TRIALS = 5;
const WARMUP_MS = 1000;      // a GPU idles at low clocks and boosts only under sustained load
const PASS_MS = 40;          // aim for about this much GPU work per timed pass
const MAX_PASS_MS = 300;     // and never more: Windows resets a GPU busy for about 2 s
// Prey flee the nearest wolf, wolves chase the nearest prey, and both turn away from a near wall,
// so a run keeps its density instead of piling into the corners.
const RULES = `fn think(i: u32, species: u32, tick: u32) {
    let wall = -3.0 * obs(i, OBS_BOUNDS + BOUNDS_SIDE);
    if (species == SPECIES_WOLF) {
        setAct(i, ACT_TURN, clamp(sign(obs(i, OBS_PREY_0 + NEAR_SIDE)) + wall, -1.0, 1.0));
        setAct(i, ACT_THROTTLE, 1.0);
        setReason(i, 2u);
    } else {
        let near = obs(i, OBS_WOLF_0 + NEAR_PRESENT);
        setAct(i, ACT_TURN, clamp(-sign(obs(i, OBS_WOLF_0 + NEAR_SIDE)) * near + wall, -1.0, 1.0));
        setAct(i, ACT_THROTTLE, 0.4 + 0.6 * near);
        setReason(i, 1u);
    }
}`;

const half = count => BASE_HALF * Math.sqrt(count / 4096);
const herdDef = (count, budget) => A.defineAgents({ capacity: count,
    world: { min: [-half(count), -half(count)], max: [half(count), half(count)] }, cellSize: CELL, maxScanPerCell: budget,
    species: [
        { name: 'prey', maxSpeed: 3.2, maxTurn: 4, accel: 6, senseRadius: CELL, radius: 0.4 },
        { name: 'wolf', maxSpeed: 4.2, maxTurn: 3, accel: 5, senseRadius: CELL, radius: 0.6 },
    ],
    sensors: [A.Sensors.self(), A.Sensors.nearest({ name: 'wolf', species: ['wolf'], k: 1 }),
        A.Sensors.nearest({ name: 'prey', species: ['prey'], k: 3 }), A.Sensors.bounds()] });

// spread: the demo's spawn. flocked: herds of 256 around random centres, a 4 m Gaussian each.
function spawn(agents, count, layout, seed) {
    const inset = half(count) - 1, herds = count / 256, r = A.rand01;
    for (let i = 0; i < count; i++) {
        let x = (r(i, seed, 0) * 2 - 1) * inset, z = (r(i, seed, 1) * 2 - 1) * inset;
        if (layout === 'flocked') {
            const h = i % herds, cx = (r(h, seed, 5) * 2 - 1) * (inset - 12), cz = (r(h, seed, 6) * 2 - 1) * (inset - 12);
            const radius = 4 * Math.sqrt(-2 * Math.log(1 - r(i, seed, 3))), angle = 2 * Math.PI * r(i, seed, 4);
            x = Math.max(-inset, Math.min(inset, cx + radius * Math.cos(angle)));
            z = Math.max(-inset, Math.min(inset, cz + radius * Math.sin(angle)));
        }
        agents.spawn(i, { position: [x, 0.08, z], species: i < count / WOLF_SHARE ? 'wolf' : 'prey',
            heading: r(i, seed, 2) * Math.PI * 2 - Math.PI });
    }
    agents.upload();
}

const median = xs => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
const fmt = n => n.toLocaleString('en-US');

async function runAdapter(adapter, name) {
    const timestamps = adapter.features.has('timestamp-query');
    const device = await adapter.requestDevice({ requiredFeatures: timestamps ? ['timestamp-query'] : [] });
    const engine = new NullGraph();
    const init = await engine.init(null, { device, width: 64, height: 64 });
    if (!init.success) throw new Error(init.error);
    async function read(src, words) {
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
    const ts = timestamps ? {
        set: device.createQuerySet({ type: 'timestamp', count: 2 }),
        resolve: device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
        read: device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
    } : null;
    const build = (def, layout, seed) => {
        const agents = new A.AgentSystem(engine, def, { thinkEvery: 3 });
        agents.setBrain(new A.RuleBrain(RULES));
        spawn(agents, def.capacity, layout, seed);
        return agents;
    };

    // 1. Divergence: the same seeded start, the same steps, twice.
    const def8 = herdDef(DIVERGE.count, DIVERGE.budget), words = { records: def8.capacity * 16,
        observations: def8.capacity * def8.obsDim, actions: def8.capacity * def8.actDim, reasons: def8.capacity };
    const runs = [];
    device.pushErrorScope('validation');
    for (let run = 0; run < 2; run++) {
        const agents = build(def8, 'spread', 11);
        for (let s = 0; s < DIVERGE.steps; s++) await agents.step();
        runs.push({ records: await read(agents.buffer, words.records), observations: await read(agents.obs, words.observations),
            actions: await read(agents.act, words.actions), reasons: await read(agents.reason, words.reasons),
            state: await read(agents.state, 4) });
        agents.destroy();
    }
    const divergeError = await device.popErrorScope();
    const differ = Object.fromEntries(Object.keys(words).map(k => [k, runs[0][k].reduce((n, w, j) => n + (w !== runs[1][k][j]), 0)]));
    const [tick, overflowSteps] = runs[0].state;
    check(`${name}: divergence runs completed ${DIVERGE.steps} steps each, overflowing in ${overflowSteps} of them`,
        divergeError === null && tick === DIVERGE.steps && runs[1].state[0] === DIVERGE.steps && overflowSteps > 0);

    // 2. Cost: replay the engine's own stage nodes, batched as engine.render() batches them.
    function encode(pass, chain) {
        for (const node of chain) {
            if (!node.enabled) continue;
            if (node.preClearBuffers.length) throw new Error(`${node.name} clears buffers; the replay would not match`);
            pass.setPipeline(node.computePipeline);
            for (const g in node.customBindGroups) pass.setBindGroup(Number(g), node.customBindGroups[g]);
            if (node.indirectBuffer) pass.dispatchWorkgroupsIndirect(node.indirectBuffer, node.indirectOffset);
            else pass.dispatchWorkgroups(...node.dispatchCount);
        }
    }
    async function timed(chain, repeats) {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass(ts ? { timestampWrites: { querySet: ts.set, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
        for (let r = 0; r < repeats; r++) encode(pass, chain);
        pass.end();
        if (ts) {
            enc.resolveQuerySet(ts.set, 0, 2, ts.resolve, 0);
            enc.copyBufferToBuffer(ts.resolve, 0, ts.read, 0, 16);
        }
        const t0 = performance.now();
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
        const wallMs = performance.now() - t0;
        if (!ts) return wallMs / repeats;
        await ts.read.mapAsync(GPUMapMode.READ);
        const [begin, end] = new BigUint64Array(ts.read.getMappedRange().slice(0));
        ts.read.unmap();
        return end > begin ? Number(end - begin) / 1e6 / repeats : wallMs / repeats;
    }
    async function measure(chain) {
        const once = await timed(chain, 1);
        const repeats = Math.max(1, Math.min(2000, Math.floor(Math.min(PASS_MS, MAX_PASS_MS) / Math.max(once, 1e-3))));
        const ms = [];
        for (let t = 0; t < TRIALS; t++) ms.push(await timed(chain, repeats));
        return { ms: median(ms), repeats };
    }
    const cost = {};
    device.pushErrorScope('validation');
    for (const count of COUNTS) {
        for (const layout of LAYOUTS) {
            const def = herdDef(count, 64), agents = build(def, layout, 11);
            for (let s = 0; s < 3; s++) await agents.step();
            agents.advance(def.dt);
            const step = agents.getComputeStages(), grid = agents.grid.getComputeStages();
            // Warm up on the grid build alone: it moves nobody, so every host measures the same state.
            const t0 = performance.now();
            while (performance.now() - t0 < WARMUP_MS) await timed(grid, 64);
            const before = await read(agents.state, 4);
            const g = await measure(grid), s = await measure(step);
            const after = await read(agents.state, 4);
            const steps = after[0] - before[0];
            cost[`${count}/${layout}`] = { stepMs: s.ms, gridMs: g.ms, gridShare: g.ms / s.ms,
                overflowShare: steps ? (after[1] - before[1]) / steps : 0, cells: def.grid.cellTotal,
                stagesPerStep: step.filter(n => n.enabled).length, gridStages: grid.filter(n => n.enabled).length };
            agents.destroy();
        }
    }
    const costError = await device.popErrorScope();
    check(`${name}: cost runs raised no validation error`, costError === null);

    ts?.set.destroy(); ts?.resolve.destroy(); ts?.read.destroy();
    device.destroy();
    return { adapter: name, host: inBrowser ? 'chrome' : 'node', timing: timestamps ? 'timestamps' : 'wall-clock',
        divergence: { ...DIVERGE, overflowSteps, differ }, cost };
}

function report(r) {
    const d = r.divergence;
    log(`\n== ${r.host === 'node' ? 'Node/Dawn' : 'Chrome'} · ${r.adapter} (${r.timing})`);
    log(`   divergence, ${fmt(d.count)} agents, budget ${d.budget}, ${d.steps} steps, overflow in ${d.overflowSteps}: ` +
        Object.entries(d.differ).map(([k, n]) => `${k} ${fmt(n)}`).join(', ') + ' words differ' +
        (Object.values(d.differ).every(n => n === 0) ? ' — the watch was NOT observed on this host' : ''));
    log(`   ${'agents/layout'.padEnd(18)}${'step ms'.padStart(10)}${'grid ms'.padStart(10)}${'grid share'.padStart(12)}` +
        `${'overflow'.padStart(10)}${'stages'.padStart(8)}`);
    for (const [key, c] of Object.entries(r.cost)) {
        log(`   ${key.padEnd(18)}${c.stepMs.toFixed(3).padStart(10)}${c.gridMs.toFixed(3).padStart(10)}` +
            `${(100 * c.gridShare).toFixed(1).padStart(11)}%${(100 * c.overflowShare).toFixed(0).padStart(9)}%` +
            `${String(c.stagesPerStep).padStart(8)}`);
    }
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
    log(`Grid order: measuring on ${name}${i.description ? ` (${i.description})` : ''}…`);
    const r = await runAdapter(adapter, name);
    results.push(r);
    report(r);
}
if (!results.length) {
    fatal('No WebGPU adapter');
    throw new Error('No WebGPU adapter');
}
log(`\nGrid order JSON ${JSON.stringify(results)}`);
await finish(`Grid order: measured on ${results.length} adapter(s).`);
log('Grid order done');
