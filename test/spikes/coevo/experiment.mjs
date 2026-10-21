// Spike-only generation runner. Fresh systems, persistent caller-owned species tables.
import * as A from '../../../dist/agents.js';
import { flatParams, networkOf, shapeOf } from '../../../tools/agents/mlp-train.mjs';
import { modelText } from '../../../tools/agents/model.mjs';
import { herdDefinition, herdRule, DRIVER } from '../../../tools/agents/scenario/herd-wolves.mjs';
import { buildLane, layout, readBuffers, runSteps } from './arena.mjs';
import { hash, noise, ranks, gradient, adam, permutation, mean } from './math.mjs';
import { record } from './evidence.mjs';

export class Population {
    constructor(engine, networks, K, M = 32) {
        this.engine = engine; this.K = K; this.M = M; this.species = {}; this.tableBytes = 0;
        try {
            for (const name of ['prey', 'wolf']) {
                const net = networks[name], x = flatParams(net), floats = A.mlpLayout(net).floats;
                const brain = A.MlpBrain.byGenome(engine, Array.from({ length: 2 * K + 1 + M }, () => net));
                this.species[name] = { name, shape: shapeOf(net), x, start: x.slice(), m: new Float64Array(x.length),
                    v: new Float64Array(x.length), steps: 0, archive: [x.slice()], brain, floats };
                this.tableBytes += brain.buffers[0].size;
            }
        } catch (e) { this.destroy(); throw e; }
    }
    pack(s, params) {
        const p = new Float32Array(params.length * s.floats);
        params.forEach((x, r) => A.packMlpWeights(networkOf(s.shape, x), p.subarray(r * s.floats, (r + 1) * s.floats)));
        return p;
    }
    checkpoint() {
        const def = herdDefinition();
        return Object.fromEntries(Object.values(this.species).map(s => [s.name,
            modelText({ species: s.name, obsNames: def.obsNames, actNames: def.actNames,
                network: networkOf(s.shape, s.x), trainedOn: { tool: 's7-replacement-2', updates: s.steps } })]));
    }
    destroy() { for (const s of Object.values(this.species)) s.brain.destroy(); }
}

export function denseFixture(init, { slot, wolf, lo, lay }) {
    // Dense prey pile at a wall, wolves spread to avoid a rest-timer-only stress.
    if (!wolf) init.position = [lo[0] + 0.4, 0.08, lo[1] + lay.L / 2 + (slot % 5) * 0.00001];
}

export async function generation(engine, pop, cfg, t, { seed = cfg.seed ?? 18001, neutral = false,
    update = true, diagnostic, fixture, mutation, fixedNetworks, ruleSpecies = [], spawnDomain = 10,
    permute = true } = {}) {
    const start = performance.now(), K = cfg.K, E = cfg.E ?? 1, B = K * E;
    const lay = layout(B, cfg.W ?? 16, cfg.P ?? 192), evolving = t % 2 ? 'wolf' : 'prey';
    const s = pop.species[evolving], frozen = pop.species[evolving === 'prey' ? 'wolf' : 'prey'];
    const sigma = cfg.sigma ?? 0.02, eps = Array.from({ length: K }, (_, k) => noise(s.x.length, seed, t, k));
    const candidates = eps.flatMap(e => [Float64Array.from(s.x, (x, j) => x + (neutral ? 0 : sigma * e[j])),
        Float64Array.from(s.x, (x, j) => x - (neutral ? 0 : sigma * e[j]))]);
    if (fixedNetworks) {
        for (const name of ['prey', 'wolf']) {
            const sp = pop.species[name], x = flatParams(fixedNetworks[name]);
            sp.brain.setRows(pop.pack(sp, Array.from({ length: 2 * K + 1 + pop.M }, () => x)));
        }
    } else {
        s.brain.setRows(pop.pack(s, candidates));
        frozen.brain.setRows(pop.pack(frozen, [frozen.x, ...frozen.archive]), { first: 2 * K });
    }
    const opponents = Array.from({ length: K }, (_, k) => hash(11, seed, t, k) / 4294967296 < (cfg.archiveShare ?? 0.5)
        ? 2 * K + 1 + hash(12, seed, t, k) % frozen.archive.length : 2 * K);
    const perm = permute ? permutation(B, seed, t) : Array.from({ length: B }, (_, b) => b);
    const inverse = new Array(B); perm.forEach((physical, logical) => { inverse[physical] = logical; });
    const seeds = Array.from({ length: B }, (_, b) => hash(spawnDomain, seed, t, Math.floor(b / E), b % E));
    record('generation-start', { cfg, t, seed, spawnDomain, evolving, seeds, opponents, permutation: perm });
    const packedMs = performance.now() - start, lanes = [], timing = [];
    let setupMs = 0, episodeMs = 0, readbackMs = 0, diagResult;
    try {
        for (let sign = 0; sign < 2; sign++) {
            const setup = performance.now();
            const lane = buildLane(engine, lay, { environment: cfg.environment ?? 'legacy-v1', contacts: !!cfg.contacts,
                thinkEvery: mutation === 'cadence' && sign === 1 ? 1 : 3,
                seedOf: physical => seeds[inverse[physical]] + (mutation === 'seed' && sign === 1 ? 1 : 0),
                rowOf: (physical, name) => name === evolving ? 2 * Math.floor(inverse[physical] / E) + sign : opponents[Math.floor(inverse[physical] / E)],
                brains: Object.fromEntries(Object.values(pop.species).map(s => [s.name, s.brain])),
                fixture: fixture ?? (cfg.state === 'dense' ? denseFixture : undefined),
                mutation: sign === 1 ? mutation : undefined });
            lanes.push(lane);
            if (ruleSpecies.length) {
                for (const name of ruleSpecies) lane.sys.setBrain({ kind: 'stage',
                    wgsl: `const ACT_TURN = 0u;\nconst ACT_THROTTLE = 1u;\n${herdRule(DRIVER)}`,
                    buffers: [lane.settings] }, { species: name });
                // setBrain rebuilds chains; never execute the stale pre-override MLP chain.
                lane.chain = lane.sys.getComputeStages();
            }
            setupMs += performance.now() - setup;
        }
        for (const lane of lanes) {
            const time = performance.now();
            if (diagnostic) {
                const history = [];
                for (let done = 0; done < cfg.T;) {
                    const n = Math.min(50, cfg.T - done); timing.push(await runSteps(engine, lane, n)); done += n;
                    history.push(await diagnostic(lane, done));
                    record('diagnostic-sample', { cfg, t, seed, spawnDomain, sign: diagResult?.length ?? 0, done, sample: history.at(-1) });
                }
                (diagResult ??= []).push(history);
            } else timing.push(await runSteps(engine, lane, cfg.T));
            episodeMs += performance.now() - time;
        }
        const r0 = performance.now();
        const counters = await readBuffers(engine.device, lanes.map(lane => ({ buffer: lane.counters, words: B })));
        readbackMs = performance.now() - r0;
        const catches = counters.map(c => Array.from({ length: B }, (_, b) => c[perm[b]]));
        const scores = [], sums = [];
        for (let k = 0; k < K; k++) for (let sign = 0; sign < 2; sign++) {
            const sum = catches[sign].slice(k * E, (k + 1) * E).reduce((a, b) => a + b, 0);
            sums.push(sum); scores.push((evolving === 'prey' ? -1 : 1) * sum);
        }
        const u = ranks(scores), g = gradient(eps, u, sigma), opt = performance.now();
        if (update) {
            adam(s, g, cfg.rate ?? 0.01, cfg.decay ?? 0.005);
            if ((t + 1) % (cfg.archiveEvery ?? 5) === 0) for (const sp of Object.values(pop.species)) {
                sp.archive.push(sp.x.slice()); if (sp.archive.length > pop.M) sp.archive.splice(1, 1);
            }
        }
        let collisions = 0;
        for (let k = 0; k < K; k++) for (let j = 0; j < s.x.length; j++)
            if (Math.fround(candidates[2 * k][j]) === Math.fround(candidates[2 * k + 1][j])) collisions++;
        const result = { t, evolving, cfg, seeds, opponents, permutation: perm, catches, scores, sums,
            mean: mean(sums) / E, tieFraction: 1 - new Set(scores).size / scores.length,
            gradientNorm: Math.hypot(...g), f32CollisionFraction: collisions / (K * s.x.length),
            packedMs, setupMs, episodeMs, readbackMs, optimizerMs: performance.now() - opt,
            wallMs: performance.now() - start, mappings: 1, activeLaneAgents: lay.count,
            tableBytes: pop.tableBytes, timing, diagnostics: diagResult };
        record('generation-completed', result);
        return result;
    } catch (e) {
        record('generation-failed', { cfg, t, seed, spawnDomain, error: String(e.stack ?? e), timing });
        throw e;
    } finally { for (const lane of lanes) lane.destroy(); }
}

export async function diagnostics(engine, lane, includeNeighbours = true) {
    const entries = [{ buffer: lane.sys.buffer, words: lane.lay.count * 16 }, { buffer: lane.stats, words: 4 },
        { buffer: lane.counters, words: lane.lay.arenas }];
    if (includeNeighbours) entries.push({ buffer: lane.sys.neighbours, words: lane.lay.count * lane.sys.neighbourStride });
    const [records, stats, counters, nb] = await readBuffers(engine.device, entries);
    const f = new Float32Array(records.buffer), lay = lane.lay;
    let crossing = 0, sampled = 0, escapes = 0, walls = 0, nn = 0, overlaps = 0, speed = 0, rest = 0, nearestCount = 0;
    for (let i = 0; i < lay.count; i++) {
        const a = records[i * 16 + 15], x = f[i * 16], z = f[i * 16 + 2], radius = f[i * 16 + 3];
        const lo = [Math.fround(lay.origin[0]) + a % lay.cols * Math.fround(lay.pitch),
            Math.fround(lay.origin[1]) + Math.floor(a / lay.cols) * Math.fround(lay.pitch)];
        const d = Math.min(x - lo[0], lo[0] + lay.L - x, z - lo[1], lo[1] + lay.L - z);
        if (d < radius - 0.0001) escapes++;
        if (i >= lay.wolfTotal) {
            if (d < 1) walls++;
            let best = Infinity;
            for (let j = lay.wolfTotal + a * lay.prey; j < lay.wolfTotal + (a + 1) * lay.prey; j++) if (i !== j)
                best = Math.min(best, Math.hypot(x - f[j * 16], z - f[j * 16 + 2]));
            if (Number.isFinite(best)) { nn += best; nearestCount++; if (best < 0.8 * 0.95) overlaps++; }
            speed += Math.hypot(f[i * 16 + 4], f[i * 16 + 6]);
        } else if (f[i * 16 + 11] > 0) rest++;
        if (nb) for (let k = 0; k < lane.sys.neighbourStride; k++) {
            const j = nb[i * lane.sys.neighbourStride + k];
            if (j === 0xffffffff) continue;
            sampled++; if (records[j * 16 + 15] !== a) crossing++;
        }
    }
    return { records, stats: Array.from(stats), counters: Array.from(counters), crossing, sampled, escapes,
        wallShare: walls / (lay.arenas * lay.prey), nearestPrey: nn / nearestCount,
        overlapShare: overlaps / nearestCount, preySpeed: speed / nearestCount, wolfRestShare: rest / lay.wolfTotal };
}