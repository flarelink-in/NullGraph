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