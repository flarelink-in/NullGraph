import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { coevoProfile } from './fixtures/coevo-profile.mjs';
import { mlpLayout } from '../dist/agents.js';
import { flatParams, networkOf } from '../tools/agents/mlp-train.mjs';
import { NOISE_VERSION, MAX_PARAMETERS, tupleHash, noise, centeredRanks, gradient,
    createAdamState, adamAscent, masterParams, candidateNetworks } from '../tools/agents/es.mjs';

const near = (a, b, tolerance = 1e-14) => assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b}`);
const snapshot = state => ({ x: state.x.slice(), m: state.m.slice(), v: state.v.slice(), steps: state.steps });
const shape = [{ input: 2, output: 2, activation: 'tanh' }, { input: 2, output: 1, activation: 'identity' }];

test('U51 E4 versioned tuple hash: exact goldens and independent BigInt multiplication oracle', () => {
    assert.equal(NOISE_VERSION, 'es-tuple-box-muller-v1');
    const fixtures = [[[0], 2812203024], [[1, 2, 3], 995390908],
        [[0xffffffff, 0, 0xffffffff], 1340915906], [[0x45530101, 1, 1, 0, 0, 0], 2188908081]];
    const mask = 0xffffffffn;
    for (const [fields, golden] of fixtures) {
        let h = 0x85ebca6bn;
        for (const field of fields) {
            h = ((h ^ BigInt(field)) * 0x9e3779b1n) & mask;
            h ^= h >> 16n; h = (h * 0x7feb352dn) & mask; h ^= h >> 15n;
        }
        assert.equal(tupleHash(...fields), golden);
        assert.equal(tupleHash(...fields), Number(h));
    }
    for (const value of [-1, 2 ** 32, 0.5, NaN, Infinity, '1']) assert.throws(() => tupleHash(value), /^Error: coevolve:/);
});

test('U51 E4 Gaussian golden on Node 24, odd prefixes, deterministic tuple separation and budget refusals', () => {
    // Node v24.12.0 / V8; allow last-bit transcendental differences, not cross-engine bit promises.
    const golden = [0.6839898543014522, 0.9382111212418293, -0.6543989593212926,
        0.7900636767115494, -0.31766940287577583];
    const actual = noise(1, 0, 0, 5);
    actual.forEach((value, j) => near(value, golden[j], 2e-15));
    assert.deepEqual(actual, noise(1, 0, 0, 5));
    assert.deepEqual(actual, noise(1, 0, 0, 6).slice(0, 5));
    for (const args of [[0x80000001, 0, 0, 5], [1, 1, 0, 5], [1, 0, 1, 5]]) assert.notDeepEqual(actual, noise(...args));
    assert.notDeepEqual(noise(1, 1, 0, 5), noise(1, 0, 4096, 5));
    assert.ok(noise(0xffffffff, 0xffffffff, 0xffffffff, 1).every(Number.isFinite));
    for (const args of [[-1, 0, 0, 1], [0, 2 ** 32, 0, 1], [0, 0, 0.5, 1],
        [0, 0, 0, 0], [0, 0, 0, MAX_PARAMETERS + 1], [0, 0, 0, Infinity]])
        assert.throws(() => noise(...args), /^Error: coevolve:/);
    const sample = noise(191, 4, 7, 20000);
    near(sample.reduce((s, x) => s + x, 0) / sample.length, 0, 0.04);
    near(sample.reduce((s, x) => s + x * x, 0) / sample.length, 1, 0.06);
});

test('U51 E2/E3 average ranks: ties, permutations, monotone invariance and neutral paired gradients', () => {
    assert.deepEqual([...centeredRanks([7, 7, 7, 7])], [0, 0, 0, 0]);
    const ranks = centeredRanks([1, 1, 2, 3]);
    near(ranks[0], -1 / 3); near(ranks[1], -1 / 3); near(ranks[2], 1 / 6); near(ranks[3], 0.5);
    assert.deepEqual(ranks, centeredRanks([3, 3, 5, 7]));
    const p = [3, 1, 0, 2], scores = [1, 1, 2, 3];
    assert.deepEqual([...centeredRanks(p.map(i => scores[i]))], p.map(i => ranks[i]));
    const directions = [new Float64Array([1, -4]), new Float64Array([2, 5])];
    assert.deepEqual(gradient(directions, centeredRanks([7, 7, 7, 7]), 0.25), new Float64Array(2));
    assert.deepEqual(gradient(directions, centeredRanks([1, 1, 2, 2]), 0.25), new Float64Array(2));
    // Independent normalization fixture: K=2, sigma=.5, utility differences 1 and .5.
    assert.deepEqual([...gradient([[1, 2], [-1, 4]], [0.5, -0.5, 0.25, -0.25], 0.5)], [0.25, 2]);
    // Watched-failing index-ranked ties would produce a nonzero update.
    assert.notDeepEqual(gradient(directions, [-0.5, -1 / 6, 1 / 6, 0.5], 0.25), new Float64Array(2));
});

test('U51 rank/gradient refusals: empty, sparse, wrong dimensions and nonfinite arithmetic', () => {
    for (const scores of [[], [1], [1, NaN], [1, Infinity], [1, , 2], null])
        assert.throws(() => centeredRanks(scores), /^Error: coevolve:/);
    for (const args of [[[], [], 1], [[[1], [1, 2]], [0, 1, 2, 3], 1], [[[1]], [0], 1],
        [[[NaN]], [0, 1], 1], [[[1]], [0, 1], 0], [[[1]], [0, 1], -1],
        [[[1]], [0, 1], Infinity], [[[1]], [0, 1], Number.MAX_VALUE],
        [[[Number.MAX_VALUE]], [Number.MAX_VALUE, -Number.MAX_VALUE], 0.01]])
        assert.throws(() => gradient(...args), /^Error: coevolve:/);
});

test('U51 E1 Adam golden two steps: coupled decay, moments, correction and independent species', () => {
    const master = new Float64Array([1, -2]), state = createAdamState(master), wolf = createAdamState(master);
    assert.notEqual(master, state.x);
    adamAscent(state, [0.5, -0.25], { rate: 0.01, decay: 0.1 });
    [1.00999999975, -2.0099999980000005].forEach((x, j) => near(state.x[j], x));
    [0.04, -0.005].forEach((x, j) => near(state.m[j], x));
    [0.00016, 0.0000025].forEach((x, j) => near(state.v[j], x));
    adamAscent(state, [-0.125, 0.75], { rate: 0.01, decay: 0.1 });
    [1.012171214686597, -2.002920513332063].forEach((x, j) => near(state.x[j], x));
    [0.013400000002500002, 0.09059999997999998].forEach((x, j) => near(state.m[j], x));
    [0.00021091599998870024, 0.000906898499619601].forEach((x, j) => near(state.v[j], x));
    assert.equal(state.steps, 2); assert.equal(wolf.steps, 0); assert.deepEqual(wolf.x, master);
});

test('U51 E1/E2 quadratic ascent and zero-gradient behavior distinguish momentum, decay and neutrality', () => {
    const s = createAdamState([3, -2]);
    const loss = () => s.x.reduce((n, x) => n + x * x, 0), before = loss();
    for (let t = 0; t < 100; t++) adamAscent(s, Float64Array.from(s.x, x => -2 * x), { rate: 0.03, decay: 0 });
    assert.ok(loss() < before / 10);
    const old = s.x.slice(); adamAscent(s, [0, 0], { rate: 0.03, decay: 0 });
    assert.notDeepEqual(s.x, old, 'old momentum still moves with zero fitness gradient');
    const neutral = createAdamState([2]);
    adamAscent(neutral, [0], { rate: 0.01, decay: 0 }); assert.equal(neutral.x[0], 2);
    const decaying = createAdamState([2]);
    adamAscent(decaying, [0], { rate: 0.01, decay: 0.1 }); assert.ok(decaying.x[0] < 2);
    near(decaying.m[0], -0.02); near(decaying.v[0], 0.00004);
});

test('U51 optimizer refuses malformed/overflow states atomically before mutation', () => {
    const state = createAdamState([1, -2]), original = snapshot(state);
    for (const options of [{ rate: 0, decay: 0 }, { rate: 0.1, decay: -1 }, { rate: Infinity, decay: 0 },
        { rate: 0.1 }, {}]) {
        assert.throws(() => adamAscent(state, [0, 0], options), /^Error: coevolve:/);
        assert.deepEqual(snapshot(state), original);
    }
    for (const g of [[0], [0, NaN], [0, Number.MAX_VALUE]]) {
        assert.throws(() => adamAscent(state, g, { rate: 0.1, decay: 0 }), /^Error: coevolve:/);
        assert.deepEqual(snapshot(state), original);
    }
    for (const change of [{ steps: 2 ** 32 - 1 }, { steps: -1 }, { v: new Float64Array([-1, 0]) },
        { m: new Float64Array([Infinity, 0]) }, { x: new Float32Array([1, 2]) }, { m: state.x }])
        assert.throws(() => adamAscent({ ...state, ...change }, [0, 0], { rate: 0.1, decay: 0 }), /^Error: coevolve:/);
    assert.deepEqual(snapshot(state), original);
    assert.throws(() => createAdamState([Infinity]), /^Error: coevolve:/);
});

test('U51 parameter order and f32 candidates reuse canonical packing without mutating f64 masters', () => {
    const master = Float64Array.from({ length: 9 }, (_, i) => i / 10 + 1e-10);
    const direction = Float64Array.from({ length: 9 }, (_, i) => i % 2 ? -0.5 : 0.25);
    const saved = master.slice(), result = candidateNetworks(shape, master, direction, 0.2);
    const network = networkOf(shape, master);
    assert.deepEqual(masterParams(network), flatParams(network));
    const expected = Float64Array.from(master, (x, j) => Math.fround(x + 0.2 * direction[j]));
    assert.deepEqual(masterParams(result.plus), expected);
    assert.deepEqual(masterParams(result.minus), Float64Array.from(master, (x, j) => Math.fround(x - 0.2 * direction[j])));
    assert.doesNotThrow(() => mlpLayout(result.plus));
    assert.equal(result.diagnostics.collisions, 0);
    near(result.diagnostics.perturbationNorm, Math.sqrt([...direction].reduce((s, x) => s + (0.2 * x) ** 2, 0)));
    near(result.diagnostics.plusNorm, Math.sqrt([...expected].reduce((s, x, j) => s + (x - Math.fround(master[j])) ** 2, 0)));
    assert.deepEqual(master, saved);
    const collision = candidateNetworks([{ input: 1, output: 1, activation: 'identity' }], [1, 1], [1, 1], 1e-10);
    assert.equal(collision.diagnostics.collisions, 2);
    assert.equal(collision.diagnostics.plusNorm, 0); assert.equal(collision.diagnostics.minusNorm, 0);
    assert.deepEqual(masterParams(collision.plus), masterParams(collision.minus));
});

test('U51 parameter/candidate refusals reject shape drift, length errors and f32 overflow', () => {
    const basic = [{ input: 1, output: 1, activation: 'identity' }];
    for (const badShape of [[], [{ input: 1, output: 1, activation: 'linear' }],
        [{ input: 1, output: 0, activation: 'tanh' }], [...basic, ...shape],
        [{ input: MAX_PARAMETERS, output: MAX_PARAMETERS, activation: 'tanh' }],
        [{ input: 129, output: 1, activation: 'identity' }],
        [{ input: 1, output: 17, activation: 'identity' }], Array(5).fill(basic[0])])
        assert.throws(() => candidateNetworks(badShape, [1, 2], [0, 0], 1), /^Error: coevolve:/);
    for (const args of [[basic, [1], [0, 0], 1], [basic, [1, 2], [NaN, 0], 1],
        [basic, [1e40, 2], [0, 0], 1], [basic, [1, 2], [1e40, 0], 1],
        [basic, [1, 2], [0, 0], 0]])
        assert.throws(() => candidateNetworks(...args), /^Error: coevolve:/);
    assert.throws(() => masterParams([{ ...basic[0], weights: [1, 2], biases: [0] }]), /^Error: coevolve:/);
});

test('U51 repeatable CPU pipeline, separate timebases and no direct GPU/filesystem/scenario imports', () => {
    const run = () => {
        const species = [createAdamState([0.25, -0.5]), createAdamState([-0.25, 0.5])];
        for (let t = 0; t < 8; t++) {
            const s = species[t % 2], directions = Array.from({ length: 4 }, (_, k) => noise(123, t, k, 2));
            const scores = directions.flatMap(e => [1, -1].map(sign =>
                -s.x.reduce((sum, x, j) => sum + (x + sign * 0.1 * e[j]) ** 2, 0)));
            adamAscent(s, gradient(directions, centeredRanks(scores), 0.1), { rate: 0.01, decay: 0.005 });
        }
        return species.map(snapshot);
    };
    assert.deepEqual(run(), run()); assert.deepEqual(run().map(s => s.steps), [4, 4]);
    const source = readFileSync(new URL('../tools/agents/es.mjs', import.meta.url), 'utf8');
    const imports = [...source.matchAll(/^import .* from '([^']+)';/gm)].map(m => m[1]);
    assert.deepEqual(imports, ['./mlp-train.mjs']);
});

test('U50 canonical config validation, hashing, default geometry and strict refusals', async () => {
    const { validateConfig, configHash, canonicalJson, parseCliArgs } = await import('../tools/agents/coevolve-config.mjs');
    assert.equal(canonicalJson({ b: 1, a: 2 }), '{"a":2,"b":1}');
    assert.equal(canonicalJson([3, 2, 1]), '[3,2,1]');

    const valid = validateConfig(coevoProfile({ name: 'test-run-u50' }));
    assert.equal(valid.schema, 1);
    assert.equal(valid.K, 8); assert.equal(valid.E, 1);
    assert.equal(valid.arenasPerLane, 8); assert.equal(valid.totalArenas, 16);
    assert.equal(valid.P, 192); assert.equal(valid.W, 16);
    assert.equal(valid.L, 24); assert.equal(valid.cols, 2); assert.equal(valid.gap, 10);
    assert.equal(valid.T, 300); assert.equal(valid.thinkEvery, 3);
    assert.equal(valid.environment, 'contact-v2');
    assert.equal(valid.contacts, false);
    assert.equal(valid.catchMargin, 0.05);
    assert.equal(valid.generations, 10);
    assert.equal(valid.start, 'random');
    assert.deepEqual(valid.hidden, [16, 16]);
    assert.ok(typeof valid.configHash === 'string' && valid.configHash.length === 64);

    // Hash sensitivity
    const modified = validateConfig(coevoProfile({ name: 'test-run-u50-b', generations: 12 }));
    assert.notEqual(valid.configHash, modified.configHash);

    // Refusals
    const bad = obj => assert.throws(() => validateConfig(obj && !Array.isArray(obj) ? coevoProfile(obj) : obj), /^Error: coevolve:/);
    bad(null);
    bad([]);
    bad({ name: 'invalid name with spaces' });
    bad({ name: 'valid', unknownKey: 1 });
    bad({ name: 'valid', K: 0 });
    bad({ name: 'valid', E: -1 });
    bad({ name: 'valid', arenas: 30, K: 8, E: 1 }); // 30 != 2*8*1 = 16
    bad({ name: 'valid', dt: 0 });
    bad({ name: 'valid', L: 2 });
    bad({ name: 'valid', gap: -1 });
    bad({ name: 'valid', environment: 'unknown-env' });
    bad({ name: 'valid', environment: 'legacy-v1', contacts: true });
    bad({ name: 'valid', environment: 'contact-v2', catchMargin: -0.1 });
    bad({ name: 'valid', optimizer: { sigma: -1 } });
    bad({ name: 'valid', optimizer: { rate: 0 } });
    bad({ name: 'valid', optimizer: { decay: -0.01 } });
    bad({ name: 'valid', archive: { every: 0 } });
    bad({ name: 'valid', archive: { cap: 1 } });
    bad({ name: 'valid', archive: { share: 1.5 } });
    bad({ name: 'valid', seeds: { master: -1 } });
    bad({ name: 'valid', seeds: { master: 2 ** 32 } });
    bad({ name: 'valid', hidden: [] });
    bad({ name: 'valid', hidden: [128] });
    bad({ name: 'valid', start: 'unknown-start' });

    // parseCliArgs
    const parsed = parseCliArgs([
        '--name', 'my-cli-run',
        '--generations', '20',
        '--K', '4',
        '--E', '2',
        '--arenas', '16',
        '--hidden', '16x32',
        '--contacts',
        '--export-onnx'
    ]);
    assert.equal(parsed.name, 'my-cli-run');
    assert.equal(parsed.generations, 20);
    assert.equal(parsed.K, 4); assert.equal(parsed.E, 2);
    assert.equal(parsed.arenas, 16);
    assert.deepEqual(parsed.hidden, [16, 32]);
    assert.equal(parsed.contacts, true);
    assert.equal(parsed.exportOnnx, true);
});

test('U52 seeds manifest, cross-pair collision detection, tile permutation and archive manager', async () => {
    const {
        buildSeedManifest, permutationForGeneration, opponentForPair,
        createArchiveManager, evolvingSpecies, opponentSpecies, noiseChecksum
    } = await import('../tools/agents/coevolve-schedule.mjs');

    // Species alternation
    assert.equal(evolvingSpecies(0), 'prey');
    assert.equal(opponentSpecies(0), 'wolf');
    assert.equal(evolvingSpecies(1), 'wolf');
    assert.equal(opponentSpecies(1), 'prey');

    // Seed manifest
    const manifest = buildSeedManifest({
        masterSeed: 18001,
        domain: 1,
        generations: 4,
        K: 4,
        E: 2,
    });
    assert.equal(manifest.generations, 4);
    assert.equal(manifest.B, 8);
    assert.equal(manifest.seedsByGeneration.length, 4);
    assert.equal(manifest.totalEpisodes, 32);
    assert.ok(typeof manifest.manifestHash === 'string' && manifest.manifestHash.length === 64);

    // Verify all seeds are u32 and unique across the entire manifest
    const allSeeds = new Set();
    for (const genSeeds of manifest.seedsByGeneration) {
        assert.equal(genSeeds.length, 8);
        for (const s of genSeeds) {
            assert.ok(Number.isInteger(s) && s >= 0 && s <= 0xffffffff);
            assert.ok(!allSeeds.has(s), `duplicate episode seed ${s} detected`);
            allSeeds.add(s);
        }
    }

    // Permutation
    const perm0 = permutationForGeneration({ masterSeed: 18001, domain: 1, t: 0, B: 8 });
    const perm1 = permutationForGeneration({ masterSeed: 18001, domain: 1, t: 1, B: 8 });
    assert.equal(perm0.length, 8);
    assert.deepEqual([...perm0].sort((a, b) => a - b), [0, 1, 2, 3, 4, 5, 6, 7]);
    assert.notDeepEqual(perm0, perm1); // Changes with t

    // Opponent sampling
    const { policyRows } = await import('../tools/agents/coevolve-schedule.mjs');
    const rows = policyRows({ K: 8, cap: 32 });
    const archiveMock = [
        { id: 'wolf-0000', generation: 0, params: new Float64Array(10) },
        { id: 'wolf-0005', generation: 5, params: new Float64Array(10) },
    ];
    // h = 0: always current
    const oppCurrent = opponentForPair({ masterSeed: 18001, domain: 1, t: 0, k: 0, archive: archiveMock, h: 0, rows });
    assert.equal(oppCurrent.type, 'current');
    assert.equal(oppCurrent.row, 16);

    // h = 1: always archive
    const oppArchive = opponentForPair({ masterSeed: 18001, domain: 1, t: 0, k: 0, archive: archiveMock, h: 1, rows });
    assert.equal(oppArchive.type, 'archive');
    assert.ok(oppArchive.row >= 17 && oppArchive.row <= 18);
    assert.throws(() => opponentForPair({ masterSeed: 1, domain: 1, t: 0, k: 0, archive: archiveMock, h: 1, K: 8 }), /policyRows/);

    // Archive manager
    const mgr = createArchiveManager({ cap: 3, every: 2 });
    mgr.init({
        prey: { params: new Float64Array([1, 2]) },
        wolf: { params: new Float64Array([3, 4]) },
    });
    assert.equal(mgr.list('prey').length, 1);
    assert.equal(mgr.get('prey', 0).id, 'prey-0000');

    // t=0: (0+1)%2 !== 0 -> no snapshot
    assert.equal(mgr.maybeSnapshot(0, 'prey', new Float64Array([10, 20])), null);

    // t=1: (1+1)%2 === 0 -> snapshot 1
    const snap1 = mgr.maybeSnapshot(1, 'prey', new Float64Array([11, 22]));
    assert.equal(snap1.snapshotted.id, 'prey-0002');
    assert.equal(snap1.evicted, null);
    assert.equal(mgr.list('prey').length, 2);

    // t=3: snapshot 2 (length reaches cap = 3)
    const snap2 = mgr.maybeSnapshot(3, 'prey', new Float64Array([12, 24]));
    assert.equal(snap2.snapshotted.id, 'prey-0004');
    assert.equal(snap2.evicted, null);
    assert.equal(mgr.list('prey').length, 3);

    // t=5: snapshot 3 -> exceeds cap 3, evicts index 1, retaining index 0!
    const snap3 = mgr.maybeSnapshot(5, 'prey', new Float64Array([13, 26]));
    assert.equal(snap3.snapshotted.id, 'prey-0006');
    assert.equal(snap3.evicted.id, 'prey-0002'); // First inserted snapshot evicted
    assert.equal(mgr.list('prey').length, 3);
    assert.equal(mgr.get('prey', 0).id, 'prey-0000'); // Initial start retained!
    assert.equal(mgr.get('prey', 1).id, 'prey-0004');
    assert.equal(mgr.get('prey', 2).id, 'prey-0006');

    // Noise checksum
    const dirs = [new Float64Array([1, 2]), new Float64Array([3, 4])];
    const ck1 = noiseChecksum(dirs);
    const ck2 = noiseChecksum(dirs);
    assert.equal(ck1, ck2);
    assert.notEqual(ck1, noiseChecksum([new Float64Array([1, 2]), new Float64Array([3, 5])]));
});

test('U52 archive: both species append every H completed updates, whichever evolved', async () => {
    const { createArchiveManager, evolvingSpecies } = await import('../tools/agents/coevolve-schedule.mjs');
    // H = 2: due only at odd t, when wolves evolve. Appending only the evolving species (the
    // 14.3 re-review bug) would never archive prey past its start.
    const mgr = createArchiveManager({ cap: 3, every: 2 });
    mgr.init({ prey: { params: new Float64Array([1]) }, wolf: { params: new Float64Array([2]) } });
    const log = [];
    for (let t = 0; t < 6; t++) {
        const masters = { prey: new Float64Array([10 + t]), wolf: new Float64Array([20 + t]) };
        const changes = mgr.snapshotDue(t, masters);
        assert.equal(changes.length, (t + 1) % 2 ? 0 : 2);
        for (const c of changes) log.push([t, evolvingSpecies(t), c.species, c.snapshotted.id, c.evicted?.id ?? null]);
    }
    assert.deepEqual(log, [
        [1, 'wolf', 'prey', 'prey-0002', null], [1, 'wolf', 'wolf', 'wolf-0002', null],
        [3, 'wolf', 'prey', 'prey-0004', null], [3, 'wolf', 'wolf', 'wolf-0004', null],
        [5, 'wolf', 'prey', 'prey-0006', 'prey-0002'], [5, 'wolf', 'wolf', 'wolf-0006', 'wolf-0002']]);
    assert.deepEqual(mgr.list('prey').map(s => s.id), ['prey-0000', 'prey-0004', 'prey-0006']);
    // Snapshots copy the master: a later update does not change an archived member.
    assert.deepEqual([...mgr.get('prey', 2).params], [15]);
    assert.throws(() => mgr.snapshotDue(7, { prey: new Float64Array([1]) }), /Float64Array/);
    assert.deepEqual(mgr.list('prey').map(s => s.id), ['prey-0000', 'prey-0004', 'prey-0006'], 'refusal is atomic');
});

test('U52 policy rows: 2K candidates, the current opponent, then a full archive inside the table', async () => {
    const { policyRows, opponentForPair } = await import('../tools/agents/coevolve-schedule.mjs');
    const rows = policyRows({ K: 4, cap: 3 });
    assert.equal(rows.count, 2 * 4 + 1 + 3);
    assert.deepEqual([rows.candidate(0, 0), rows.candidate(0, 1), rows.candidate(3, 1)], [0, 1, 7]);
    assert.equal(rows.current, 8);
    assert.deepEqual([0, 1, 2].map(rows.archive), [9, 10, 11]);
    // The last archive row exists; without the +1 it would be the table's end.
    assert.ok(rows.archive(rows.cap - 1) < rows.count);
    for (const bad of [() => rows.archive(3), () => rows.archive(-1), () => rows.candidate(4, 0), () => rows.candidate(0, 2)])
        assert.throws(bad, /^Error: coevolve:/);
    for (const args of [{ K: 0, cap: 3 }, { K: 1, cap: 1 }, { K: 2 ** 31, cap: 2 }])
        assert.throws(() => policyRows(args), /^Error: coevolve:/);
    const full = Array.from({ length: 3 }, (_, j) => ({ id: `wolf-${j}`, params: new Float64Array(1) }));
    const sampled = new Set();
    for (let t = 0; t < 64; t++) sampled.add(opponentForPair({ masterSeed: 7, domain: 1, t, k: 0, archive: full, h: 1, rows }).row);
    assert.deepEqual([...sampled].sort((a, b) => a - b), [9, 10, 11]);
});

test('U52 random start seeds: own stream per species and domain, no arithmetic on the master seed', async () => {
    const { initSeed, STREAM_INIT } = await import('../tools/agents/coevolve-schedule.mjs');
    const prey = initSeed({ masterSeed: 18001, domain: 1, species: 'prey' });
    assert.equal(prey, tupleHash(18001, 1, STREAM_INIT, 0));
    assert.equal(initSeed({ masterSeed: 18001, domain: 1, species: 'wolf' }), tupleHash(18001, 1, STREAM_INIT, 1));
    assert.notEqual(prey, initSeed({ masterSeed: 18001, domain: 2, species: 'prey' }));
    assert.notEqual(prey, initSeed({ masterSeed: 18002, domain: 1, species: 'prey' }));
    assert.throws(() => initSeed({ masterSeed: 1, domain: 1, species: 'deer' }), /^Error: coevolve:/);
});

test('U53 checkpoint writing, atomic export, model identity and manifest verification', async () => {
    const {
        createRunDirectory, writeRunJson, appendLog,
        writeGenerationCheckpoint, writeArchiveSnapshot
    } = await import('../tools/agents/coevolve-checkpoint.mjs');
    const { existsSync, rmSync, readFileSync } = await import('node:fs');
    const { resolve, join } = await import('node:path');

    const testDir = resolve('tools/agents/out/coevo/u53-test-checkpoint');
    if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });

    try {
        createRunDirectory(testDir);
        assert.ok(existsSync(testDir));
        assert.ok(existsSync(join(testDir, 'archive')));

        // Test run.json write
        writeRunJson(testDir, {
            config: { schema: 1, algorithm: 'coevo-v1', configHash: 'abc', seeds: { master: 123 } },
            seedManifest: { masterSeed: 123, domain: 1, totalEpisodes: 10, manifestHash: 'def', retries: [] },
        });
        assert.ok(existsSync(join(testDir, 'run.json')));
        const runJson = JSON.parse(readFileSync(join(testDir, 'run.json'), 'utf8'));
        assert.equal(runJson.configHash, 'abc');

        // Test appendLog
        appendLog(testDir, { t: 0, species: 'prey', scores: [1, 2] });
        appendLog(testDir, { t: 1, species: 'wolf', scores: [3, 4] });
        const logLines = readFileSync(join(testDir, 'log.jsonl'), 'utf8').trim().split('\n');
        assert.equal(logLines.length, 2);
        assert.equal(JSON.parse(logLines[0]).t, 0);

        // Test generation checkpoint
        const net = [
            { input: 24, output: 16, activation: 'tanh', weights: new Float32Array(24 * 16), biases: new Float32Array(16) },
            { input: 16, output: 2, activation: 'tanh', weights: new Float32Array(16 * 2), biases: new Float32Array(2) }
        ];
        const obsNames = Array.from({ length: 24 }, (_, i) => `obs.${i}`);
        const actNames = ['turn', 'throttle'];

        writeGenerationCheckpoint(testDir, {
            generation: 0,
            preyNetwork: net,
            wolfNetwork: net,
            obsNames,
            actNames,
            configHash: 'hash123',
            experiment: { environment: 'contact-v2', dt: 0.1 },
            exportOnnx: true,
        });

        const gen0Dir = join(testDir, 'gen-0000');
        assert.ok(existsSync(gen0Dir));
        assert.ok(existsSync(join(gen0Dir, 'prey.mlp.json')));
        assert.ok(existsSync(join(gen0Dir, 'wolf.mlp.json')));
        assert.ok(existsSync(join(gen0Dir, 'manifest.json')));
        assert.ok(existsSync(join(gen0Dir, 'prey.onnx')));
        assert.ok(existsSync(join(gen0Dir, 'wolf.onnx')));

        // Manifest verification
        const manifest = JSON.parse(readFileSync(join(gen0Dir, 'manifest.json'), 'utf8'));
        assert.equal(manifest.generation, 0);
        assert.equal(manifest.configHash, 'hash123');
        assert.deepEqual(manifest.experiment, { environment: 'contact-v2', dt: 0.1 });
        assert.ok(manifest.models.prey.hash);
        assert.ok(manifest.models.wolf.hash);
        assert.throws(() => writeGenerationCheckpoint(testDir, { generation: 1, preyNetwork: net, wolfNetwork: net,
            obsNames, actNames, configHash: 'hash123' }), /experiment/);

        // Test archive snapshot
        writeArchiveSnapshot(testDir, {
            snapshot: { id: 'prey-0000', generation: 0 },
            species: 'prey',
            network: net,
            obsNames,
            actNames,
            configHash: 'hash123',
        });
        assert.ok(existsSync(join(testDir, 'archive/prey-0000.mlp.json')));
        const { createHash } = await import('node:crypto');
        const index = readFileSync(join(testDir, 'archive/index.jsonl'), 'utf8').trim().split('\n').map(line => JSON.parse(line));
        assert.deepEqual(index, [{ id: 'prey-0000', species: 'prey', generation: 0,
            hash: createHash('sha256').update(readFileSync(join(testDir, 'archive/prey-0000.mlp.json'))).digest('hex') }]);
    } finally {
        if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
    }
});

test('U55 guard submission calibration, batch size and 1500 ms safety threshold', async () => {
    const { checkStep, STEP_STOP_MS } = await import('../tools/film/guard.mjs');
    assert.equal(STEP_STOP_MS, 1500);

    // checkStep passes for safe timings
    assert.equal(checkStep(100, 1500), 100);
    assert.equal(checkStep(500, 1500), 500);
    assert.equal(checkStep(1500, 1500), 1500);

    // checkStep throws when exceeding safety stop
    assert.throws(() => checkStep(1501, 1500), /film: a step took .* ms, over the 1,500 ms stop/);
    assert.throws(() => checkStep(2000, 1500), /film: a step took .* ms, over the 1,500 ms stop/);
    assert.throws(() => checkStep(-1, 1500), /film: invalid step timing/);
    assert.throws(() => checkStep(NaN, 1500), /film: invalid step timing/);

    // Batch sizing arithmetic: target 500 ms, capped at 1000 ms, floor division
    const { createSubmissionGuard } = await import('../tools/agents/coevolve-guard.mjs');
    const calcBatch = (stepMs, T = 300) => createSubmissionGuard(stepMs).next(T);
    assert.equal(calcBatch(10), 50);    // 50 * 10 = 500 ms
    assert.equal(calcBatch(20), 25);    // 25 * 20 = 500 ms
    assert.equal(calcBatch(400), 1);    // 1 * 400 = 400 ms
    assert.equal(calcBatch(600), 1);    // 1 * 600 = 600 ms (single step <= 1000)
});

// Independent oracle for the bootstrap: plain arrays, integer tail ranks for the levels used here.
function bootstrapOracle(values, seed, resamples, tail) {
    const n = values.length, means = [];
    for (let r = 0; r < resamples; r++) {
        let sum = 0;
        for (let j = 0; j < n; j++) sum += values[Math.floor(tupleHash(0x45530601, seed, n, r, j) * n / 2 ** 32)];
        means.push(sum / n);
    }
    means.sort((x, y) => x - y);
    return { low: means[tail - 1], high: means[resamples - tail] };
}

test('U54 bootstrap: deterministic shared draws, nearest-rank bounds, constants and goldens', async () => {
    const { BOOTSTRAP, BOOTSTRAP_VERSION, resampleIndices, meanInterval } = await import('../tools/agents/coevolve-stats.mjs');
    assert.deepEqual(BOOTSTRAP, { resamples: 10000, level: 0.95 });
    assert.deepEqual([...resampleIndices({ seed: 7, n: 5, resamples: 3 })], [2, 2, 2, 4, 4, 3, 1, 0, 3, 4, 4, 0, 4, 0, 3]);
    assert.deepEqual(resampleIndices({ seed: 7, n: 5, resamples: 50 }), resampleIndices({ seed: 7, n: 5, resamples: 50 }));
    assert.notDeepEqual(resampleIndices({ seed: 7, n: 5, resamples: 50 }), resampleIndices({ seed: 8, n: 5, resamples: 50 }));
    const draws = resampleIndices({ seed: 3, n: 4, resamples: 10000 }), counts = [0, 0, 0, 0];
    for (const index of draws) counts[index]++;
    for (const count of counts) assert.ok(Math.abs(count / draws.length - 0.25) < 0.01, `draw shares ${counts}`);
    assert.ok(resampleIndices({ seed: 1, n: 1, resamples: 20 }).every(index => index === 0));

    // Golden (exact: integer hashing and fixed-order sums are the same in every engine).
    assert.deepEqual(meanInterval([1, 2, 3, 4, 5], { seed: 1, resamples: 1000 }),
        { n: 5, mean: 3, low: 1.8, high: 4.2, level: 0.95, resamples: 1000, seed: 1, version: BOOTSTRAP_VERSION });
    // Integer catches tie often; square roots make neighbouring ranks differ, so the rank is pinned.
    const integers = [12, 0, 7, 7, 3, 19, 4, 8, 1, 30, 2, 2, 6, 9, 11, 5, 0, 14, 3, 7];
    const distinct = Array.from({ length: 20 }, (_, j) => Math.sqrt(j + 2));
    for (const values of [integers, distinct]) for (const [resamples, level, tail] of [[1000, 0.95, 25], [10000, 0.95, 250], [400, 0.9, 20], [4, 0.5, 1]]) {
        const got = meanInterval(values, { seed: 99, resamples, level }), want = bootstrapOracle(values, 99, resamples, tail);
        assert.equal(got.low, want.low); assert.equal(got.high, want.high);
        assert.ok(got.low <= got.mean && got.mean <= got.high);
    }
    const next = bootstrapOracle(distinct, 99, 10000, 251);
    assert.notEqual(meanInterval(distinct, { seed: 99 }).low, next.low, 'the 250th and 251st means differ');
    // Constant samples and a single episode give a zero-width interval at the value.
    for (const constant of [[4, 4, 4, 4], [2.5]]) {
        const c = meanInterval(constant, { seed: 5 });
        assert.equal(c.low, constant[0]); assert.equal(c.high, constant[0]); assert.equal(c.mean, constant[0]);
    }
    for (const bad of [[[], { seed: 1 }], [[1, NaN], { seed: 1 }], [[1, Infinity], { seed: 1 }], [['1'], { seed: 1 }],
        [[1], {}], [[1], { seed: -1 }], [[1], { seed: 0.5 }], [[1], { seed: 1, resamples: 0 }],
        [[1], { seed: 1, level: 1 }], [[1], { seed: 1, level: 0 }], [null, { seed: 1 }]])
        assert.throws(() => meanInterval(...bad), /^Error: coevolve:/);
    assert.throws(() => resampleIndices({ seed: 1, n: 7000, resamples: 10000 }), /draws/);
});

test('U54 paired intervals: known shifts are exact, shared noise cancels, and decisions need the interval', async () => {
    const { pairedBootstrap, meanInterval, superior, noninferior } = await import('../tools/agents/coevolve-stats.mjs');
    const a = [3, 1, 4, 1, 5, 9, 2, 6], options = { seed: 42, resamples: 2000 };
    const shifted = pairedBootstrap(a, a.map(x => x + 3), options);
    assert.equal(shifted.difference.low, -3); assert.equal(shifted.difference.high, -3); assert.equal(shifted.difference.mean, -3);
    // The marginals use the same draws as the difference, and equal the one-arm interval.
    assert.deepEqual(shifted.a, meanInterval(a, options));
    assert.ok(Math.abs(shifted.b.low - (shifted.a.low + 3)) < 1e-12 && Math.abs(shifted.b.high - (shifted.a.high + 3)) < 1e-12);
    assert.equal(superior(shifted.difference, { better: 'lower' }), true);
    assert.equal(superior(shifted.difference, { better: 'higher' }), false);
    // Episode-to-episode spread shared by both arms overlaps the marginals; the paired interval is exact.
    const shared = [0, 40, 3, 25, 9, 31, 17, 2, 36, 12], paired = pairedBootstrap(shared.map(z => z + 2), shared, options);
    assert.ok(paired.a.low < paired.b.high, 'marginal intervals overlap');
    assert.deepEqual([paired.difference.low, paired.difference.high], [2, 2]);
    assert.equal(superior(paired.difference, { better: 'higher' }), true);
    // Strictly favorable: a bound at exactly zero is not superiority.
    const touching = pairedBootstrap([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], new Array(10).fill(0), options);
    assert.equal(touching.difference.low, 0);
    assert.equal(superior(touching.difference, { better: 'higher' }), false);
    // Golden mixed case: an interval spanning zero decides nothing either way.
    const mixed = pairedBootstrap(a, [2, 7, 1, 8, 2, 8, 1, 8], options);
    assert.deepEqual([mixed.difference.mean, mixed.difference.low, mixed.difference.high], [-0.75, -3.25, 1.625]);
    assert.equal(superior(mixed.difference, { better: 'higher' }), false);
    assert.equal(superior(mixed.difference, { better: 'lower' }), false);
    // Noninferiority needs a preregistered margin; no significance is not equivalence.
    assert.equal(noninferior(mixed.difference, { better: 'higher', margin: 4 }), true);
    assert.equal(noninferior(mixed.difference, { better: 'higher', margin: 3 }), false);
    assert.equal(noninferior(mixed.difference, { better: 'lower', margin: 2 }), true);
    for (const bad of [{ better: 'higher' }, { better: 'higher', margin: 0 }, { better: 'up', margin: 1 }])
        assert.throws(() => noninferior(mixed.difference, bad), /^Error: coevolve:/);
    assert.throws(() => superior(mixed.difference, {}), /^Error: coevolve:/);
    assert.throws(() => pairedBootstrap([1, 2], [1], options), /differ in length/);
    assert.throws(() => pairedBootstrap([1, 2], [1, NaN], options), /^Error: coevolve:/);
    const source = readFileSync(new URL('../tools/agents/coevolve-stats.mjs', import.meta.url), 'utf8');
    assert.deepEqual([...source.matchAll(/^import .* from '([^']+)';/gm)].map(m => m[1]), ['./es.mjs'], 'browser-loadable');
});

test('U54 held-out manifests: own stream per purpose, numerically distinct from training, recorded retries', async () => {
    const { buildSeedManifest, buildHeldOutManifest, manifestSeeds, STREAM_HELD_OUT } = await import('../tools/agents/coevolve-schedule.mjs');
    const training = buildSeedManifest({ masterSeed: 18001, domain: 1, generations: 3, K: 4, E: 2 });
    const validation = buildHeldOutManifest({ masterSeed: 18001, domain: 2, purpose: 'validation', episodes: 20, exclude: [training] });
    assert.equal(validation.seeds.length, 20);
    assert.equal(validation.seeds[0], tupleHash(18001, 2, STREAM_HELD_OUT.validation, 0, 0));
    assert.equal(validation.excluded, 24);
    const finalTest = buildHeldOutManifest({ masterSeed: 18001, domain: 2, purpose: 'final-test', episodes: 20,
        exclude: [training, validation] });
    const all = [...manifestSeeds(training), ...validation.seeds, ...finalTest.seeds];
    assert.equal(new Set(all).size, all.length, 'no seed is shared by training, validation and final test');
    // Separate streams, not the collision check, keep the purposes apart: no retries were needed.
    assert.deepEqual([validation.retries, finalTest.retries], [[], []]);
    assert.equal(finalTest.seeds[0], tupleHash(18001, 2, 8, 0, 0));
    assert.notEqual(validation.manifestHash, finalTest.manifestHash);
    assert.deepEqual(buildHeldOutManifest({ masterSeed: 18001, domain: 2, purpose: 'validation', episodes: 20, exclude: [training] }), validation);
    // A training seed equal to a held-out candidate is regenerated, deterministically, and recorded.
    const clash = { seeds: [validation.seeds[3]] };
    const regenerated = buildHeldOutManifest({ masterSeed: 18001, domain: 2, purpose: 'validation', episodes: 20, exclude: [clash] });
    assert.deepEqual(regenerated.retries, [{ e: 3, retry: 1, seed: tupleHash(18001, 2, STREAM_HELD_OUT.validation, 3, 1) }]);
    assert.ok(!regenerated.seeds.includes(validation.seeds[3]));
    assert.deepEqual(regenerated.seeds.filter((_, e) => e !== 3), validation.seeds.filter((_, e) => e !== 3));
    for (const bad of [{ purpose: 'test' }, { episodes: 0 }, { exclude: [{}] }, { exclude: 'training' }, { masterSeed: -1 }])
        assert.throws(() => buildHeldOutManifest({ masterSeed: 1, domain: 1, purpose: 'validation', episodes: 2, ...bad }), /^Error: coevolve:/);
});

test('U54 forgetting: maximum chronological loss, joint episode resampling, no loss on monotone curves', async () => {
    const { forgettingInterval } = await import('../tools/agents/coevolve-stats.mjs');
    const spread = [2, 9, 1, 8], options = { seed: 72, resamples: 1000 };
    const shifted = offsets => offsets.map(offset => spread.map(value => value + offset));
    const lost = forgettingInterval(shifted([0, 5, 2]), options);
    assert.deepEqual([lost.value, lost.low, lost.high], [3, 3, 3]);
    assert.deepEqual(forgettingInterval(shifted([0, 5, 2]), options), lost);
    for (const offsets of [[0], [0, 1, 2], [0, 0, 0]]) {
        const result = forgettingInterval(shifted(offsets), options);
        assert.deepEqual([result.value, result.low, result.high], [0, 0, 0]);
    }
    assert.equal(forgettingInterval(shifted([3, 0, 7, 2]), options).value, 5);
    assert.throws(() => forgettingInterval([], options), /ordered/);
    assert.throws(() => forgettingInterval([[1], [1, 2]], options), /share episodes/);
});

// Evaluation fixtures: network drivers need only an id and a layer list until the GPU plays them.
const net = id => ({ kind: 'network', id, network: [] });
const pairOf = id => ({ id, prey: net(id), wolf: net(id) });

test('U54 evaluation layout: the experiment\'s tile once per held-out episode, aligned, at least 20', async () => {
    const { evaluationGeometry, MIN_EPISODES } = await import('../tools/agents/coevolve-evaluate.mjs');
    const { validateConfig } = await import('../tools/agents/coevolve-config.mjs');
    const experiment = validateConfig(coevoProfile({ name: 'u54-layout', outDir: 'tools/agents/out/coevo/u54-never-created' }));
    assert.equal(MIN_EPISODES, 20);
    assert.deepEqual(evaluationGeometry(experiment, 20),
        { arenas: 20, wolves: 16, prey: 192, L: 24, gap: 10, cols: 5, contacts: null });
    assert.equal(evaluationGeometry(experiment, 36).cols, 6);
    assert.throws(() => evaluationGeometry(experiment, 16), /at least 20/);
    assert.throws(() => evaluationGeometry(experiment, 22), /multiple of 4/);
    const contacts = validateConfig(coevoProfile({ name: 'u54-contacts', outDir: 'tools/agents/out/coevo/u54-never-created',
        contacts: true, contactOptions: { iterations: 2, maxPush: 0.25, relaxation: 1, compliance: 0, damping: 0.05,
            mass: 'equal', maxScanPerCell: 64, cellSize: 2 } }));
    assert.equal(evaluationGeometry(contacts, 20).contacts.maxPush, 0.25);
});

test('U54 evaluation plans: baselines both ways with their own pairings, cross-play every cell', async () => {
    const { RULE_PAIR, LABEL_PAIR, matchup, baselineMatchups, baselineComparisons, crossplayMatchups } =
        await import('../tools/agents/coevolve-evaluate.mjs');
    const subject = pairOf('gen-0010'), clone = pairOf('clone'), baselines = [RULE_PAIR, LABEL_PAIR, clone];
    const plan = baselineMatchups(subject, baselines);
    assert.deepEqual(plan.map(m => m.id), ['gen-0010/gen-0010', 'gen-0010/rule', 'rule/gen-0010', 'rule/rule',
        'gen-0010/label', 'label/gen-0010', 'label/label', 'gen-0010/clone', 'clone/gen-0010', 'clone/clone']);
    assert.equal(plan[1].prey, subject.prey); assert.equal(plan[1].wolf.kind, 'rule');
    const comparisons = baselineComparisons(subject, baselines);
    assert.equal(comparisons.length, 6);
    assert.deepEqual(comparisons.slice(0, 2), [
        { species: 'prey', better: 'lower', subject: 'gen-0010', baseline: 'rule', minuend: 'gen-0010/rule', subtrahend: 'rule/rule' },
        { species: 'wolf', better: 'higher', subject: 'gen-0010', baseline: 'rule', minuend: 'rule/gen-0010', subtrahend: 'rule/rule' }]);
    const ids = new Set(plan.map(m => m.id));
    for (const c of comparisons) assert.ok(ids.has(c.minuend) && ids.has(c.subtrahend), `${c.minuend} - ${c.subtrahend} is planned`);
    const cross = crossplayMatchups([pairOf('gen-0000'), pairOf('gen-0010'), pairOf('gen-0020')]);
    assert.equal(cross.length, 9);
    assert.equal(cross[5].id, 'gen-0010/gen-0020');
    assert.equal(cross[5].prey.id, 'gen-0010'); assert.equal(cross[5].wolf.id, 'gen-0020');
    for (const bad of [[{ kind: 'network', id: '', network: [] }, RULE_PAIR.wolf], [{ kind: 'teacher', id: 't' }, RULE_PAIR.wolf], [RULE_PAIR.prey, null]])
        assert.throws(() => matchup(...bad), /^Error: coevolve:/);
});

test('U54 evaluation summaries: paired by episode, the right direction per species, anchors frozen', async () => {
    const { RULE_PAIR, baselineMatchups, baselineComparisons, summarize, crossplayMatchups, crossplaySummary } =
        await import('../tools/agents/coevolve-evaluate.mjs');
    const subject = pairOf('gen-0010'), plan = baselineMatchups(subject, [RULE_PAIR]);
    // Episode-to-episode spread shared by every matchup; the subject's prey suffer one catch fewer
    // against rule wolves on every episode, its wolves gain nothing.
    const spread = Array.from({ length: 20 }, (_, e) => (e * 7) % 13);
    const catches = plan.map(m => (m.id === 'gen-0010/rule' ? spread.map(x => x + 4) : spread.map(x => x + 5)));
    const options = { seed: 11, resamples: 2000 };
    const { matchups, comparisons } = summarize(plan, catches, baselineComparisons(subject, [RULE_PAIR]), options);
    assert.deepEqual(matchups.map(m => [m.id, m.prey, m.wolf]), plan.map(m => [m.id, m.prey.id, m.wolf.id]));
    assert.deepEqual(matchups[1].catches, catches[1]);
    const [prey, wolf] = comparisons;
    assert.deepEqual([prey.difference.low, prey.difference.high, prey.superior], [-1, -1, true]);
    assert.deepEqual([wolf.difference.low, wolf.difference.high, wolf.superior], [0, 0, false]);
    // The marginals of the same data overlap: only the pairing shows the improvement.
    assert.ok(matchups[1].interval.high > matchups[3].interval.low);
    assert.ok(!('noninferior' in prey), 'no margin, no noninferiority verdict');
    const margined = summarize(plan, catches, baselineComparisons(subject, [RULE_PAIR]), { ...options, margin: 0.5 }).comparisons;
    assert.deepEqual(margined.map(c => c.noninferior), [true, true]);
    const shortfall = plan.map(m => (m.id === 'rule/gen-0010' ? spread.map(x => x + 4) : spread.map(x => x + 5)));
    assert.equal(summarize(plan, shortfall, baselineComparisons(subject, [RULE_PAIR]), { ...options, margin: 0.5 }).comparisons[1].noninferior,
        false, 'wolves one catch short of the baseline are not within a half-catch margin');
    assert.throws(() => summarize(plan, catches.slice(1), [], options), /one catch list per matchup/);
    assert.throws(() => summarize(plan, catches, [{ minuend: 'x/y', subtrahend: 'rule/rule', better: 'lower' }], options), /not played/);

    const generations = [pairOf('gen-0000'), pairOf('gen-0010')], cross = crossplayMatchups(generations);
    // Prey generation i, wolf generation j: catches 10 + 2j - i on every episode.
    const cells = cross.map(m => Array(20).fill(10 + 2 * Number(m.wolf.id.slice(-2) === '10') - Number(m.prey.id.slice(-2) === '10')));
    const summary = crossplaySummary(generations, cross, cells, ['gen-0000'], options);
    assert.deepEqual(summary.mean, [[10, 12], [9, 11]]);
    assert.deepEqual(summary.anchorScores.prey.map(s => s.mean), [-10, -9], 'prey score minus catches against anchor wolves');
    assert.deepEqual(summary.anchorScores.wolf.map(s => s.mean), [10, 12], 'wolves score catches against anchor prey');
    assert.deepEqual(summary.forgetting.order, ['gen-0000', 'gen-0010']);
    assert.equal(summary.forgetting.prey.value, 0); assert.equal(summary.forgetting.wolf.value, 0);
    const reversed = crossplaySummary([...generations].reverse(), cross, cells, ['gen-0000'], options);
    assert.deepEqual(reversed.forgetting, summary.forgetting, 'forgetting order is chronological, not argument order');
    const both = crossplaySummary(generations, cross, cells, ['gen-0000', 'gen-0010'], options);
    assert.deepEqual(both.anchorScores.prey.map(s => s.mean), [-11, -10]);
    assert.throws(() => crossplaySummary(generations, cross, cells, ['gen-0020'], options), /anchors/);
    assert.throws(() => crossplaySummary(generations, cross, cells, [], options), /anchors/);
});

test('U54 raw-data replay: an evaluation file\'s statistics recompute from its raw catches alone', async () => {
    const { RULE_PAIR, LABEL_PAIR, baselineMatchups, baselineComparisons, summarize, crossplayMatchups, crossplaySummary, replay } =
        await import('../tools/agents/coevolve-evaluate.mjs');
    const subject = pairOf('gen-0010'), baselines = [RULE_PAIR, LABEL_PAIR], plan = baselineMatchups(subject, baselines);
    const catches = plan.map((m, k) => Array.from({ length: 20 }, (_, e) => (e * (k + 3)) % 11));
    const bootstrap = { seed: 5, resamples: 1000, level: 0.95 }, comparisons = baselineComparisons(subject, baselines);
    // As evaluation.json stores it, through JSON.
    const file = JSON.parse(JSON.stringify({ mode: 'baselines', plan: plan.map(m => ({ id: m.id, prey: m.prey.id, wolf: m.wolf.id })),
        comparisons, bootstrap, margin: 0.5, results: summarize(plan, catches, comparisons, { ...bootstrap, margin: 0.5 }) }));
    assert.deepEqual(replay(file), { results: file.results, crossplay: null });
    const altered = JSON.parse(JSON.stringify(file));
    altered.results.matchups[2].catches[7] += 1;
    assert.notDeepEqual(replay(altered).results, file.results, 'an altered raw catch changes the replay');

    const generations = [pairOf('gen-0000'), pairOf('gen-0010')], cross = crossplayMatchups(generations);
    const cells = cross.map((m, k) => Array.from({ length: 20 }, (_, e) => (e + 2 * k) % 7));
    const stored = JSON.parse(JSON.stringify({ mode: 'crossplay', plan: cross.map(m => ({ id: m.id, prey: m.prey.id, wolf: m.wolf.id })),
        comparisons: [], bootstrap, margin: null, results: summarize(cross, cells, [], bootstrap),
        crossplay: crossplaySummary(generations, cross, cells, ['gen-0000'], bootstrap) }));
    assert.deepEqual(replay(stored), { results: stored.results, crossplay: stored.crossplay });
    delete stored.results.matchups[1].catches;
    assert.throws(() => replay({ ...stored, results: { matchups: stored.results.matchups.slice(1) } }), /no raw catches/);
});

test('U54 evaluation command line: strict flags, whole numbers, one mode', async () => {
    const { parseEvalArgs } = await import('../tools/agents/coevolve-eval.mjs');
    assert.deepEqual(parseEvalArgs(['crossplay', '--run', 'r', '--generations', '0,10', '--anchors', '0', '--purpose', 'validation',
        '--seed', '7', '--domain', '3', '--analysis-seed', '99', '--episodes', '24']),
        { mode: 'crossplay', run: 'r', purpose: 'validation', name: undefined, outDir: undefined, generations: [0, 10], anchors: [0],
            seed: 7, domain: 3, analysisSeed: 99, episodes: 24, T: undefined, margin: undefined, clones: true });
    assert.equal(parseEvalArgs(['baselines', '--no-clones']).clones, false);
    assert.equal(parseEvalArgs(['baselines', '--margin', '0.75']).margin, 0.75);
    assert.equal(parseEvalArgs(['baselines', '--diagnostics-stride', '60']).diagnosticsStride, 60);
    for (const argv of [['evaluate'], [], ['baselines', '--seed'], ['baselines', '--seed', '-1'], ['baselines', '--seed', '1.5'],
        ['baselines', '--margin', 'half'], ['baselines', '--margin', '-1'],
        ['baselines', '--episodes', '20', '--episodes', '24'], ['baselines', '--generation', '10'], ['baselines', 'stray'],
        ['baselines', '--generations', '10,x'], ['baselines', '--run', '--seed', '1']])
        assert.throws(() => parseEvalArgs(argv), /^Error: coevolve:/, JSON.stringify(argv));
});