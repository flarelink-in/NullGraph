import * as A from '../../../dist/agents.js';
import { resolveDiscContacts } from '../../../dist/physics.js';
import { buildLane, layout, readBuffers } from './arena.mjs';
import { Population, generation } from './experiment.mjs';
import { record } from './evidence.mjs';

const stationary = { kind: 'stage', wgsl: `fn think(i: u32, species: u32, tick: u32) {
    setAct(i, 0u, 0.0); setAct(i, 1u, -1.0);
}` };
async function read(engine, lane) {
    return readBuffers(engine.device, [{ buffer: lane.sys.buffer, words: lane.lay.count * 16 },
        { buffer: lane.sys.neighbours, words: lane.lay.count * lane.sys.neighbourStride },
        { buffer: lane.stats, words: 4 }]);
}
export async function extraFixtures(engine, networks, check) {
    const results = [];
    const checked = (name, pass, detail) => { check(name, pass, detail); results.push({ name, pass, detail }); record('fixture', { name, pass, detail }); };
    // Actual nearest-list catch semantics, including the resting-nearest occlusion case.
    // Brute-force oracle sorts by distance then index, BEFORE testing the chosen wolf's rest.
    for (const environment of ['legacy-v1', 'contact-v1']) for (const kind of ['coincident', 'tie', 'resting-nearest', 'outside', 'dead-nearest', 'multi-prey']) {
        const reach = environment === 'legacy-v1' ? 0.9 : 1.05;
        const lane = buildLane(engine, layout(1, 2, 6), { environment, fixture: (init, { i, wolf }) => {
            init.age = 0.25; init.heading = 0;
            init.position = [wolf ? (i === 0 ? -0.3 : 0.3) : 0, 0.08, wolf ? 0 : (i - 2) * 0.11];
            if (kind === 'coincident') init.position = [0, 0.08, 0];
            if (kind === 'outside' && !wolf) init.position = [0, 0.08, reach + 0.4];
            if (kind === 'resting-nearest' && i === 0) { init.user0 = 1; init.position[0] = -0.1; }
        } });
        try {
            lane.sys.setBrain(stationary);
            if (kind !== 'multi-prey' && kind !== 'coincident') for (let i = 3; i < 8; i++) lane.sys.kill(i);
            if (kind === 'dead-nearest') lane.sys.kill(0);
            const before = lane.sys.records.data.slice(), bits = new Uint32Array(before.buffer);
            let expected = 0; const nearest = [];
            for (let i = 2; i < 8; i++) {
                if (!(bits[i * 16 + 13] & 1)) continue;
                const sorted = [0, 1].filter(j => bits[j * 16 + 13] & 1).map(j => ({ j,
                    d: Math.hypot(before[j * 16] - before[i * 16], before[j * 16 + 2] - before[i * 16 + 2]) }))
                    .sort((a, b) => a.d - b.d || a.j - b.j);
                const w = sorted[0]; nearest.push({ i, wolf: w.j });
                if (w.d < reach && before[w.j * 16 + 11] <= 0) expected++;
            }
            await lane.sys.step();
            const [records, nb, stats] = await read(engine, lane);
            checked(`G1 brute-force ${environment}/${kind}`, stats[0] === expected &&
                nearest.every(({ i, wolf }) => nb[i * lane.sys.neighbourStride] === wolf), `expected ${expected}, actual ${stats[0]}`);
            checked(`G1 flags consumed ${environment}/${kind}`, records.every((v, i) => i % 16 !== 13 || (v & 768) === 0));
        } finally { lane.destroy(); }
    }
    // Off-cadence newborn: block zero at tick one must refresh only the respawned row.
    for (const broken of [false, true]) {
        const lane = buildLane(engine, layout(1, 1, 7));
        try {
            lane.sys.setBrain({ kind: 'stage', wgsl: `fn think(i: u32, species: u32, tick: u32) { setAct(i, 0u, f32(tick + 1u)); setAct(i, 1u, -1.0); }` });
            await lane.sys.step();
            lane.sys.spawn(1, { species: 'prey', position: [0, 0.08, 0], age: broken ? 1 : 0 });
            lane.sys.upload(); await lane.sys.step();
            const [act] = await readBuffers(engine.device, [{ buffer: lane.sys.act, words: 16 }]);
            const f = new Float32Array(act.buffer), predicate = f[2] === 2 && f[4] === 1;
            checked(`G1 ${broken ? 'watched-failing' : 'real'} off-cadence newborn`, broken ? !predicate : predicate);
        } finally { lane.destroy(); }
    }
    // Quantify, rather than hide, the pre-act nearest-list limitation. The chosen resting wolf
    // occludes an eligible second wolf; both policies are preserved, not silently changed here.
    const occluded = buildLane(engine, layout(1, 2, 6), { environment: 'contact-v1', fixture: (init, { i }) => {
        init.position = [i === 0 ? -0.1 : i === 1 ? 0.2 : 0, 0.08, 0];
        init.user0 = i === 0 ? 1 : 0;
    } });
    try {
        for (let i = 3; i < 8; i++) occluded.sys.kill(i);
        occluded.sys.setBrain(stationary); await occluded.sys.step();
        const [, , stats] = await read(engine, occluded);
        checked('G1 measured resting-nearest catch miss', stats[0] === 0, 'exhaustive eligible wolf oracle: 1; legacy/proposed nearest-list catch: 0');
        record('catch-limitation', { case: 'resting-nearest', exhaustive: 1, listActual: stats[0], environment: 'contact-v1' });
    } finally { occluded.destroy(); }
    // Inner tile wall: global solver clamp cannot replace tile containment. Fixed coincident
    // neighbours push a movable prey OUT of an internal wall in the omit-containment mutant.
    for (const broken of [false, true]) {
        const lay = layout(2, 1, 7), lane = buildLane(engine, lay, { environment: 'contact-v1', contacts: true,
            mutation: broken ? 'omit-containment' : undefined, fixture: (init, { a, wolf, slot, lo }) => {
                init.age = 1;
                init.position = [lo[0] + lay.L - 0.4 - (slot === 1 ? 0 : 0.15), 0.08, lo[1] + lay.L / 2];
                init.frozen = slot !== 1;
                if (wolf) init.position[2] = lo[1] + 0.6;
            } });
        try {
            lane.sys.setBrain(stationary); await lane.sys.step();
            const [rec] = await read(engine, lane), f = new Float32Array(rec.buffer);
            const i = lay.wolfTotal, hi = lay.origin[0] + lay.L - 0.4;
            const predicate = f[i * 16] <= hi + 0.0001;
            checked(`G1 ${broken ? 'watched-failing omit' : 'real'} tile containment`, broken ? !predicate : predicate,
                `x ${f[i * 16]}, wall ${hi}`);
        } finally { lane.destroy(); }
    }
    // Explicit dense overflow and advancing private solver phase, not just neutral end states.
    const lane = buildLane(engine, layout(1, 1, 255), { environment: 'contact-v1', contacts: true,
        contactOptions: { maxScanPerCell: 8 }, fixture: (init, { wolf }) => {
            init.position = [wolf ? 10 : 0, 0.08, 0]; init.age = 1;
        } });
    try {
        lane.sys.setBrain(stationary);
        const positions = [];
        for (let t = 0; t < 4; t++) {
            await lane.sys.step(); const [rec] = await read(engine, lane);
            positions.push(Array.from(rec));
            const counters = await lane.solver.readCounters();
            checked(`G1 dense contact overflow tick ${t}`, counters.overflow > 0 && counters.bodiesOverMaxRadius === 0,
                JSON.stringify(counters));
        }
        checked('G1 coincident dense solver progresses', positions.some(p => p.some((v, i) => i % 16 < 3 && v !== positions[0][i])));
    } finally { lane.destroy(); }
    // Independent CPU oracle for the EXACT dense private solver windows, including tick/pass
    // rotation and fixed agents. No catches/act/respawn mutate the oracle between these steps.
    const dense = buildLane(engine, layout(1, 1, 127), { environment: 'contact-v1', contacts: true,
        contactOptions: { maxScanPerCell: 8 }, fixture: (init, { i }) => {
            init.position = [0.0001 * (i % 4), 0.08, 0.0001 * (i % 3)]; init.frozen = i % 17 === 0;
        } });
    try {
        const [initial] = await read(engine, dense);
        let cpu = new Float32Array(initial.buffer);
        for (let tick = 0; tick < 4; tick++) {
            const options = dense.solver.options;
            const oracle = resolveDiscContacts({ capacity: dense.lay.count, stride: 16,
                alive: { offset: 13, mask: 1 }, fixed: { offset: 13, mask: 2 }, velocity: { offset: 4 } },
            { ...options, world: { min: [options.world.minX, options.world.minZ], max: [options.world.maxX, options.world.maxZ] } }, cpu, undefined, tick);
            await engine.device.__s7Prepare?.(); engine.render(dense.solver.getComputeStages());
            const [records, deltas] = await readBuffers(engine.device, [{ buffer: dense.sys.buffer, words: dense.lay.count * 16 },
                { buffer: dense.solver.deltas, words: dense.lay.count * 4 }]);
            const actual = new Float32Array(records.buffer), f = new Float32Array(deltas.buffer);
            const counts = oracle.contactCounts.every((n, i) => n === f[i * 4 + 2]);
            let error = 0;
            for (let i = 0; i < actual.length; i++) if ([0, 2, 4, 6].includes(i % 16)) error = Math.max(error, Math.abs(actual[i] - oracle.records[i]));
            checked(`G1 independent dense/contact CPU window tick ${tick}`, counts && error < 0.002,
                `counts ${counts}, max error ${error}`);
            // Rebase from observed GPU input for EACH independent step. A multi-step CPU rollout
            // amplifies tiny trig/normalization differences and eventually changes cell membership;
            // that is not a useful exact window-selection oracle for today's GPU input.
            cpu = actual;
        }
    } finally { dense.destroy(); }
    // Distinct archive content: verify GPU row bytes and per-agent genome assignments.
    const pop = new Population(engine, networks, 8);
    try {
        for (const s of Object.values(pop.species)) {
            s.archive = [s.x.slice(), Float64Array.from(s.x, x => x + 0.25)];
            s.brain.setRows(pop.pack(s, [s.x, ...s.archive]), { first: 16 });
            // MlpBrain intentionally does not grant COPY_SRC. Read through a spike storage kernel,
            // not an invalid copy or a library buffer-usage change.
            const count = s.floats * 19;
            const out = engine.device.createBuffer({ size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
            let words;
            try {
                const stage = engine.createComputeStage({ name: 'S7 archive identity read', dispatchCount: [Math.ceil(count / 64), 1, 1],
                    shaderCode: `@group(0) @binding(0) var<storage, read> src: array<u32>;
                    @group(0) @binding(1) var<storage, read_write> dst: array<u32>;
                    @compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
                        if (id.x < ${count}u) { dst[id.x] = src[id.x]; }
                    }` });
                engine.setComputePassBindGroup(stage, [s.brain.buffers[0], out].map((buffer, binding) => ({ binding, resource: { buffer } })));
                await engine.device.__s7Prepare?.(); engine.render([stage]);
                [words] = await readBuffers(engine.device, [{ buffer: out, words: count }]);
                stage.enabled = false;
            } finally { out.destroy(); }
            const expected = new Uint32Array(pop.pack(s, [s.x, ...s.archive]).buffer);
            checked(`G1 archive GPU identity ${s.name}`, expected.every((w, i) => words[16 * s.floats + i] === w));
        }
        const g = await generation(engine, pop, { K: 8, E: 2, P: 192, W: 16, T: 1, archiveShare: 1 }, 0,
            { neutral: true, update: false, diagnostic: async lane => {
                const [rec] = await read(engine, lane);
                return { rows: Array.from({ length: lane.lay.count }, (_, i) => [rec[i * 16 + 12], rec[i * 16 + 14], rec[i * 16 + 15]]) };
            } });
        const inverse = []; g.permutation.forEach((p, logical) => { inverse[p] = logical; });
        checked('G1 archive/permutation/replica assignment', g.diagnostics.every((history, sign) => history[0].rows.every(([species, row, tile]) =>
            row === (species === 0 ? 2 * Math.floor(inverse[tile] / 2) + sign : g.opponents[Math.floor(inverse[tile] / 2)]))) &&
            new Set(g.opponents).size === 2 && g.opponents.every(r => r === 17 || r === 18));
        const wrong = g.opponents.map(r => r + 1);
        checked('G1 watched-failing archive identity', g.diagnostics[0][0].rows.some(([species, row, tile]) =>
            species === 1 && row !== wrong[Math.floor(inverse[tile] / 2)]));
    } finally { pop.destroy(); }
    return results;
}