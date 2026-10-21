import * as A from '../../../dist/agents.js';
import { createHerd } from '../../../tools/agents/scenario/herd-wolves.mjs';
import { buildLane, layout, readBuffers, runSteps } from './arena.mjs';
import { Population, generation, diagnostics, denseFixture } from './experiment.mjs';
import { ranks, gradient, noise, mean, sd, pairedInterval, scoreQuality } from './math.mjs';
import { extraFixtures } from './fixtures.mjs';
import { record } from './evidence.mjs';

const wordsEqual = (a, b, normalizeGenome = false) => a.length === b.length &&
    a.every((v, i) => normalizeGenome && i % 16 === A.AGENT_GENOME_OFFSET || v === b[i]);
const compact = d => { const { records, obs, act, ...rest } = d; return rest; };
const adversarial = (init, c) => {
    denseFixture(init, c);
    init.age = 0.25; // non-newborn cadence, not only the unconditional newborn think
    if (!c.wolf && c.slot === c.lay.wolves) init.appFlags = 1; // force flag consumption / respawn
    if (c.wolf && c.slot === 0) { init.frozen = true; init.user0 = 1; }
};
async function snapshot(engine, lane) {
    const d = await diagnostics(engine, lane);
    [d.obs, d.act] = await readBuffers(engine.device, [
        { buffer: lane.sys.obs, words: lane.lay.count * lane.sys.definition.obsDim },
        { buffer: lane.sys.act, words: lane.lay.count * lane.sys.definition.actDim }]);
    return d;
}

export async function gate1(engine, networks, check) {
    const result = { equivalence: [], neutral: [], controls: [], catches: [] };
    result.extra = await extraFixtures(engine, networks, check);
    const herd = createHerd(engine), one = buildLane(engine, layout(1, 256, 3840));
    try {
        for (let done = 0; done < 300;) {
            const n = 50;
            // A public exact one-step path, not spike calibration or hidden warm-up.
            for (let j = 0; j < n; j++) { await herd.agents.step(); await one.sys.step(); }
            done += n;
            const [a, b, ac, bc] = await readBuffers(engine.device, [
                { buffer: herd.agents.buffer, words: 4096 * 16 }, { buffer: one.sys.buffer, words: 4096 * 16 },
                { buffer: herd.stats, words: 4 }, { buffer: one.stats, words: 4 }]);
            const equal = wordsEqual(a, b); result.equivalence.push({ done, equal, catches: [ac[0], bc[0]] });
            check(`G1 legacy equivalence step ${done}`, equal && ac[0] === bc[0], `${ac[0]} / ${bc[0]} catches`);
        }
    } finally { herd.destroy(); one.destroy(); }
    const pop = new Population(engine, networks, 8);
    try {
        const neutralCfg = { K: 8, E: 1, P: 192, W: 16, T: 600, environment: 'contact-v1', contacts: true };
        for (const environment of ['legacy-v1', 'contact-v1']) for (const state of ['ordinary', 'adversarial']) {
            const cfg = { ...neutralCfg, environment, contacts: environment === 'contact-v1' };
            const snapshots = [];
            const run = await generation(engine, pop, cfg, 0, { seed: 15001, neutral: true, update: false,
                fixture: state === 'adversarial' ? adversarial : undefined,
                diagnostic: async lane => { const d = await snapshot(engine, lane); snapshots.push(d); return compact(d); } });
            const n = cfg.T / 50, plus = snapshots.slice(0, n), minus = snapshots.slice(n);
            const equal = plus.every((a, i) => wordsEqual(a.records, minus[i].records, true) &&
                wordsEqual(a.obs, minus[i].obs) && wordsEqual(a.act, minus[i].act));
            const isolated = snapshots.every(d => d.crossing === 0 && d.escapes === 0);
            const counted = snapshots.every(d => d.counters.reduce((a, b) => a + b, 0) === d.stats[0]);
            const catches = run.catches.flat().reduce((a, b) => a + b, 0);
            check(`G1 neutral lanes ${environment}/${state}`, equal, `${catches} catches`);
            check(`G1 isolation/containment ${environment}/${state}`, isolated && snapshots.some(d => d.sampled > 0));
            check(`G1 counters ${environment}/${state}`, counted && catches > 0);
            result.neutral.push({ environment, state, equal, isolated, counted, run });
        }
        // E=2 mapping and both evolving species must also satisfy the neutral oracle.
        for (const t of [0, 1]) {
            const cfg = { ...neutralCfg, E: 2, T: 50 };
            const run = await generation(engine, pop, cfg, t, { neutral: true, update: false });
            check(`G1 replica/sign row mapping ${t % 2 ? 'wolf' : 'prey'}`,
                run.catches[0].every((v, b) => v === run.catches[1][b]) && run.sums.every((v, i) => i % 2 === 0 || v === run.sums[i - 1]));
        }
        // Known-bad versions are caught by the SAME neutral/counter predicates.
        for (const mutation of ['seed', 'cadence', 'respawn-id', 'double-count']) {
            const snapshots = [];
            await generation(engine, pop, { ...neutralCfg, T: 6 }, 0, { neutral: true, update: false,
                mutation, fixture: adversarial,
                diagnostic: async lane => { const d = await snapshot(engine, lane); snapshots.push(d); return compact(d); } });
            const caught = mutation === 'double-count'
                ? snapshots.some(d => d.counters.reduce((a, b) => a + b, 0) !== d.stats[0])
                : !wordsEqual(snapshots[0].records, snapshots[1].records, true);
            check(`G1 watched-failing ${mutation}`, caught); result.controls.push({ mutation, caught });
        }
        // Reset oracle: unrelated episodes on the same persistent brains cannot affect fresh state.
        const a = [], b = [];
        const capture = dest => async lane => { const d = await snapshot(engine, lane); dest.push(d); return compact(d); };
        await generation(engine, pop, { ...neutralCfg, T: 20 }, 0, { neutral: true, update: false, diagnostic: capture(a) });
        await generation(engine, pop, { ...neutralCfg, T: 17 }, 7, { neutral: true, update: false, fixture: adversarial });
        await generation(engine, pop, { ...neutralCfg, T: 20 }, 0, { neutral: true, update: false, diagnostic: capture(b) });
        check('G1 complete fresh-episode reset after unrelated work', a.every((d, i) => wordsEqual(d.records, b[i].records) && wordsEqual(d.obs, b[i].obs)));
        const u = ranks([7, 7, 7, 7]);
        check('G1 equal fitness gives exactly zero gradient', gradient([noise(9, 1, 0, 0), noise(9, 1, 0, 1)], u, 0.02).every(x => x === 0));
        // Last allocated archive row must be usable, and one beyond must be refused.
        for (const s of Object.values(pop.species)) {
            s.brain.setRows(pop.pack(s, [s.x]), { first: 2 * pop.K + pop.M });
            let refused = false;
            try { s.brain.setRows(pop.pack(s, [s.x]), { first: 2 * pop.K + pop.M + 1 }); } catch { refused = true; }
            check(`G1 table capacity ${s.name}`, refused);
        }
    } finally { pop.destroy(); }
    // Brute-force unambiguous boundary fixtures: exactly one live wolf and prey, zero movement.
    for (const environment of ['legacy-v1', 'contact-v1']) {
        const reach = environment === 'legacy-v1' ? 0.9 : 1.05;
        for (const distance of [reach - 0.001, reach + 0.001]) for (const contacts of [false, true]) {
            if (environment === 'legacy-v1' && contacts) continue;
            const lane = buildLane(engine, layout(1, 1, 3), { environment, contacts, fixture: (init, { wolf }) => {
                init.position = [wolf ? 0 : distance, 0.08, 0];
            } });
            try {
                for (let i = 2; i < 4; i++) lane.sys.kill(i);
                lane.sys.setBrain({ kind: 'stage', wgsl: 'fn think(i: u32, species: u32, tick: u32) { setAct(i, 0u, 0.0); setAct(i, 1u, -1.0); }' });
                await lane.sys.step();
                const d = await diagnostics(engine, lane);
                const expected = distance < reach ? 1 : 0;
                check(`G1 catch brute-force boundary ${environment}/${contacts}/${distance}`, d.stats[0] === expected && d.counters[0] === expected,
                    `expected ${expected}, actual ${d.stats[0]}`);
                result.catches.push({ environment, contacts, distance, expected, actual: d.stats[0] });
            } finally { lane.destroy(); }
        }
    }
    record('gate1-completed', result);
    return result;
}

export async function gate2(engine, networksByShape, protocol, check, log) {
    const results = [];
    for (const cfg of protocol.screen) {
        const pop = new Population(engine, networksByShape[cfg.shape], cfg.K);
        try {
            const settings = { ...cfg, ...protocol.rate };
            const ledger = engine.device.__s7Ledger;
            if (ledger) ledger.peakBytes = ledger.liveBytes;
            const warmup = await generation(engine, pop, settings, 0);
            const samples = [];
            for (let t = 1; t <= protocol.rate.measuredGenerations; t++) samples.push(await generation(engine, pop, settings, t));
            const wall = samples.map(g => g.wallMs).sort((a, b) => a - b);
            const summary = { medianMs: wall[Math.floor(wall.length / 2)], minMs: wall[0], maxMs: wall.at(-1),
                generationsPerMinute: 60000 / wall[Math.floor(wall.length / 2)] };
            check(`G2 ${JSON.stringify(cfg)}`, samples.length >= 5 && samples.every(s => s.mappings === 1));
            log(`G2 ${JSON.stringify(cfg)} ${JSON.stringify(summary)}`);
            results.push({ cfg, warmup, samples, summary, bufferMemory: ledger ? { ...ledger } : null });
            record('gate2-setting', results.at(-1));
        } finally { pop.destroy(); }
    }
    return results;
}

export async function gate3(engine, networks, protocol, check, log) {
    const p = protocol.noise, results = [];
    for (const species of ['prey', 'wolf']) for (const E of p.E) for (const T of p.T) for (const sigma of p.sigma) {
        const cfg = { ...p, E, T, sigma, species }, pop = new Population(engine, networks, p.K);
        try {
            const samples = [];
            for (let seedIndex = 0; seedIndex < p.validationSeeds; seedIndex++) {
                // t/perturbation seed fixed; spawnDomain alone varies the 20 independent validation worlds.
                samples.push(await generation(engine, pop, cfg, species === 'prey' ? 0 : 1, { seed: p.seed, spawnDomain: 100 + seedIndex, update: false }));
                record('gate3-sample', { cfg, seedIndex, sample: samples.at(-1) });
            }
            const directions = Array.from({ length: p.K }, (_, k) => {
                const differences = samples.map(g => (g.sums[2 * k] - g.sums[2 * k + 1]) / E);
                return { k, differences, mean: mean(differences), sd: sd(differences), interval: pairedInterval(differences) };
            });
            const summary = { meanAbsoluteSignal: mean(directions.map(d => Math.abs(d.mean))),
                residualPairedSD: mean(directions.map(d => d.sd)), tieFraction: mean(samples.map(g => g.tieFraction)),
                f32CollisionFraction: mean(samples.map(g => g.f32CollisionFraction)) };
            const quality = scoreQuality(directions, protocol.qualityGate);
            check(`G3 paired samples E${E}/T${T}/sigma${sigma}`, samples.length === 20 && directions.every(d => d.differences.length === 20));
            log(`G3 ${species}/E${E}/T${T}/sigma${sigma}: ${JSON.stringify(summary)}`);
            results.push({ cfg, samples, directions, summary, quality });
            record('gate3-setting', results.at(-1));
        } finally { pop.destroy(); }
    }
    const accepted = results.filter(r => r.quality.accepted && results.some(other => other.cfg.species !== r.cfg.species &&
        other.cfg.E === r.cfg.E && other.cfg.T === r.cfg.T && other.cfg.sigma === r.cfg.sigma && other.quality.accepted));
    results.qualityAccepted = accepted.length >= 2;
    record('gate3-verdict', { qualityAccepted: results.qualityAccepted, accepted: accepted.map(r => r.cfg), thresholds: protocol.qualityGate });
    check('G3 score-quality acceptance for both species', results.qualityAccepted, 'preregistered stability, signal/residual and interval bars; failure blocks Gate 4');
    return results;
}

export async function gate4(engine, networks, protocol, check, log) {
    const cfg = { ...protocol.smoke, environment: 'contact-v1', contacts: true };
    const run = async n => {
        const pop = new Population(engine, networks, cfg.K), trajectory = [], checkpoints = [];
        try {
            checkpoints.push(pop.checkpoint());
            for (let t = 0; t < n; t++) {
                trajectory.push(await generation(engine, pop, cfg, t)); checkpoints.push(pop.checkpoint());
                record('smoke-checkpoint', { repeat: n === cfg.repeatGenerations, t, checkpoint: checkpoints.at(-1) });
            }
            return { trajectory, checkpoints, pop };
        } catch (e) { pop.destroy(); throw e; }
    };
    const main = await run(cfg.generations);
    let repeat;
    try {
        repeat = await run(cfg.repeatGenerations);
        const exact = repeat.checkpoints.every((p, i) => JSON.stringify(p) === JSON.stringify(main.checkpoints[i]));
        check('G4 repeat three-generation checkpoints byte-for-byte', exact);
        const evolved = Object.fromEntries(Object.entries(main.pop.species).map(([name, s]) => [name,
            // Preserve f32 export boundary exactly as evaluation loads it.
            JSON.parse(main.checkpoints.at(-1)[name]).layers.map(l => ({ ...l, weights: Float32Array.from(l.weights), biases: Float32Array.from(l.biases) }))]));
        const evaluations = {};
        for (const matchup of ['clone-clone', 'evolved-clone', 'clone-evolved', 'evolved-evolved', 'evolved-rule', 'rule-evolved', 'rule-rule']) {
            const [prey, wolf] = matchup.split('-');
            const fixedNetworks = { prey: prey === 'evolved' ? evolved.prey : networks.prey,
                wolf: wolf === 'evolved' ? evolved.wolf : networks.wolf };
            const values = [], histories = [];
            const pop = new Population(engine, networks, cfg.K);
            try {
                // K*E spawns per generation; collect >=20 distinct episode seeds, use plus lane only.
                for (let group = 0; values.length < cfg.validationSeeds; group++) {
                    const g = await generation(engine, pop, cfg, 0, { seed: cfg.validationDomain + group, update: false,
                        fixedNetworks, ruleSpecies: ['prey', 'wolf'].filter(n => (n === 'prey' ? prey : wolf) === 'rule'),
                        diagnostic: async lane => compact(await diagnostics(engine, lane, false)) });
                    values.push(...g.catches[0]); histories.push(g.diagnostics[0]);
                }
                evaluations[matchup] = { catches: values.slice(0, cfg.validationSeeds), histories };
                record('anchor-evaluation', { matchup, ...evaluations[matchup] });
            } finally { pop.destroy(); }
        }
        const paired = {};
        for (const name of ['evolved-clone', 'clone-evolved', 'evolved-evolved'])
            paired[name] = pairedInterval(evaluations[name].catches.map((v, i) => v - evaluations['clone-clone'].catches[i]));
        for (const [name, result] of Object.entries(paired)) log(`G4 ${name} catch change vs clone/clone: ${JSON.stringify(result)}`);
        check('G4 all anchor evaluations have 20 validation episodes', Object.values(evaluations).every(e => e.catches.length === 20));
        return { cfg, trajectory: main.trajectory, checkpoints: main.checkpoints, repeatExact: exact, evaluations, paired };
    } finally { main.pop.destroy(); repeat?.pop.destroy(); }
}