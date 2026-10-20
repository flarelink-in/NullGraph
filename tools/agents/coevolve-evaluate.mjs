// Phase 14.4, design §10: arena-matched evaluation. Matchups play held-out episodes in the
// training environment, the checkpoint manifest's experiment: every episode of a matchup at once,
// episode e in tile e of one fresh lane from tick zero. All matchups of an evaluation share that
// layout and those seeds, so their results pair by episode. A species is driven by the rule, the
// label (the rule without wander and coin flips) or a network. No node: imports, so the GPU suites
// can run it in Chrome too; the command line is coevolve-eval.mjs.
import { arenaLayout } from './scenario/arena-layout.mjs';
import { buildLane, readBuffers } from './scenario/herd-wolves-arenas.mjs';
import { meanInterval, noninferior, pairedBootstrap, superior } from './coevolve-stats.mjs';

export const MIN_EPISODES = 20;     // design §10: at least 20 held-out episodes per pairing
const SPECIES = ['prey', 'wolf'];
const fail = message => { throw new Error(`coevolve: ${message}`); };
const gcd = (a, b) => (b ? gcd(b, a % b) : a);

/** Drivers. A network driver is { kind: 'network', id, network, ... }; ids name drivers in results. */
export const RULE = Object.freeze({ kind: 'rule', id: 'rule' });
export const LABEL = Object.freeze({ kind: 'label', id: 'label' });
/** Driver pairs, one driver per species: a checkpoint, the clones, or a rule for both. */
export const RULE_PAIR = Object.freeze({ id: 'rule', prey: RULE, wolf: RULE });
export const LABEL_PAIR = Object.freeze({ id: 'label', prey: LABEL, wolf: LABEL });

function checkDriver(driver, species) {
    if (driver?.kind === 'rule' || driver?.kind === 'label') return driver;
    if (driver?.kind !== 'network' || typeof driver.id !== 'string' || !driver.id || !Array.isArray(driver.network))
        fail(`${species} driver must be the rule, the label or a network with an id`);
    return driver;
}

/** A matchup: the prey's driver against the wolves'. Its id is "<prey id>/<wolf id>". */
export function matchup(prey, wolf) {
    checkDriver(prey, 'prey'); checkDriver(wolf, 'wolf');
    return Object.freeze({ id: `${prey.id}/${wolf.id}`, prey, wolf });
}

function unique(plan) {
    const seen = new Set();
    return plan.filter(m => !seen.has(m.id) && seen.add(m.id));
}

/** One tile of the experiment per held-out episode, in a near-square grid of the evaluation's own. */
export function evaluationGeometry(experiment, episodes) {
    if (!Number.isSafeInteger(episodes) || episodes < MIN_EPISODES)
        fail(`evaluation needs at least ${MIN_EPISODES} held-out episodes per pairing, not ${episodes}`);
    const unit = 64 / gcd(64, experiment.W);
    if (episodes % unit) fail(`${episodes} tiles of ${experiment.W} wolves break 64-agent alignment; use a multiple of ${unit}`);
    const geometry = { arenas: episodes, wolves: experiment.W, prey: experiment.P, L: experiment.L, gap: experiment.gap,
        cols: Math.ceil(Math.sqrt(episodes)), contacts: experiment.contacts ? experiment.contactOptions : null };
    arenaLayout(geometry);
    return geometry;
}

/**
 * The design's matchups for one checkpoint `subject` (a driver pair) against `baselines` (pairs):
 * its prey against each baseline's wolves, each baseline's prey against its wolves, each baseline
 * against itself, and the subject against itself.
 */
export function baselineMatchups(subject, baselines) {
    const plan = [matchup(subject.prey, subject.wolf)];
    for (const b of baselines) plan.push(matchup(subject.prey, b.wolf), matchup(b.prey, subject.wolf), matchup(b.prey, b.wolf));
    return unique(plan);
}

/**
 * Paired comparisons over the same episodes, per species and baseline. Prey are better with fewer
 * catches: subject prey against the baseline's wolves, minus the baseline's own pairing. Wolves
 * are better with more: the baseline's prey against the subject's wolves, minus the same pairing.
 */
export function baselineComparisons(subject, baselines) {
    return baselines.flatMap(b => [
        { species: 'prey', better: 'lower', subject: subject.id, baseline: b.id,
            minuend: matchup(subject.prey, b.wolf).id, subtrahend: matchup(b.prey, b.wolf).id },
        { species: 'wolf', better: 'higher', subject: subject.id, baseline: b.id,
            minuend: matchup(b.prey, subject.wolf).id, subtrahend: matchup(b.prey, b.wolf).id },
    ]);
}

/** Every prey generation against every wolf generation; `generations` are driver pairs. */
export function crossplayMatchups(generations) {
    return unique(generations.flatMap(p => generations.map(w => matchup(p.prey, w.wolf))));
}

function stats(options) {
    const { seed, resamples, level } = options ?? {};
    return { seed, resamples, level };
}

/** Raw catches per matchup with a marginal interval each, and the paired comparisons. With a
 * preregistered `options.margin` (catches per episode), each comparison also says noninferior. */
export function summarize(plan, catches, comparisons, options) {
    if (!Array.isArray(catches) || catches.length !== plan.length) fail('one catch list per matchup');
    const byId = new Map(plan.map((m, j) => [m.id, catches[j]]));
    return {
        matchups: plan.map((m, j) => ({ id: m.id, prey: m.prey.id, wolf: m.wolf.id, catches: [...catches[j]],
            interval: meanInterval(catches[j], stats(options)) })),
        comparisons: comparisons.map(c => {
            const a = byId.get(c.minuend), b = byId.get(c.subtrahend);
            if (!a || !b) fail(`comparison ${c.minuend} - ${c.subtrahend} names a matchup that was not played`);
            const { difference } = pairedBootstrap(a, b, stats(options));
            return { ...c, difference, superior: superior(difference, { better: c.better }),
                ...(options?.margin === undefined ? {} : { noninferior: noninferior(difference, { better: c.better, margin: options.margin }) }) };
        }),
    };
}

/**
 * Cross-play matrices (prey generation i, wolf generation j): mean catches and intervals, and each
 * generation's sign-adjusted catches against fixed `anchors` (driver-pair ids chosen before the
 * results are read, design §10): a prey generation scores minus its mean catches against the
 * anchor wolves, a wolf generation its mean catches against the anchor prey, averaged per episode
 * before resampling so the episodes stay paired.
 */
export function crossplaySummary(generations, plan, catches, anchors, options) {
    const ids = generations.map(g => g.id), byId = new Map(plan.map((m, j) => [m.id, catches[j]]));
    const cell = (i, j) => byId.get(`${generations[i].prey.id}/${generations[j].wolf.id}`) ?? fail('cross-play cell not played');
    const anchorIndex = anchors.map(id => ids.indexOf(id));
    if (!anchors.length || anchorIndex.some(k => k < 0)) fail(`anchors ${anchors.join(', ')} must be evaluated generations`);
    const average = lists => Array.from(lists[0], (_, e) => lists.reduce((sum, list) => sum + list[e], 0) / lists.length);
    const score = (values, sign) => meanInterval(values.map(v => sign * v), stats(options));
    const intervals = ids.map((_, i) => ids.map((__, j) => meanInterval(cell(i, j), stats(options))));
    return {
        generations: ids, anchors: [...anchors],
        mean: intervals.map(row => row.map(interval => interval.mean)),
        intervals,
        anchorScores: {
            prey: ids.map((_, i) => score(average(anchorIndex.map(j => cell(i, j))), -1)),
            wolf: ids.map((_, j) => score(average(anchorIndex.map(i => cell(i, j))), 1)),
        },
    };
}

/**
 * Recomputes an evaluation's statistics from its raw catches alone (the exit's raw-data replay):
 * `evaluation` is a parsed evaluation.json. Equal to its stored results when nothing was altered.
 */
export function replay(evaluation) {
    const plan = evaluation.plan.map(m => ({ id: m.id, prey: { id: m.prey }, wolf: { id: m.wolf } }));
    const raw = new Map(evaluation.results.matchups.map(m => [m.id, m.catches]));
    const catches = plan.map(m => raw.get(m.id) ?? fail(`no raw catches for ${m.id}`));
    const { seed, resamples, level } = evaluation.bootstrap;
    const options = { seed, resamples, level, ...(evaluation.margin == null ? {} : { margin: evaluation.margin }) };
    const results = summarize(plan, catches, evaluation.comparisons, options);
    if (evaluation.mode !== 'crossplay') return { results, crossplay: null };
    const ids = evaluation.crossplay.generations, generations = ids.map(id => ({ id, prey: { id }, wolf: { id } }));
    return { results, crossplay: crossplaySummary(generations, plan, catches, evaluation.crossplay.anchors, options) };
}

/**
 * Plays `game` (a matchup) on every held-out episode at once and returns each episode's catches.
 * `tables` maps a network driver's id to its caller-owned one-row MlpBrain.byGenome table per
 * species ({ prey, wolf }); `run(lane)` steps the lane exactly T times under the caller's guard.
 * `compile(make)` builds the lane and resolves its pipelines before any step is submitted.
 */
export async function playMatchup(engine, { experiment, geometry, seeds, game, tables, run, T,
    compile = async make => ({ value: make(), setupCompileMs: 0 }) }) {
    if (!Array.isArray(seeds) || seeds.length !== geometry.arenas) fail('one held-out seed per tile');
    const brains = {};
    for (const species of SPECIES) {
        const driver = checkDriver(game[species], species);
        brains[species] = driver.kind === 'network'
            ? tables.get(driver.id)?.[species] ?? fail(`no table for ${species} driver ${driver.id}`) : driver.kind;
    }
    const built = await compile(() => buildLane(engine, geometry, { environment: experiment.environment,
        contacts: experiment.contacts, contactOptions: experiment.contactOptions, catchMargin: experiment.catchMargin,
        spawnSeparation: experiment.spawnSeparation, thinkEvery: experiment.thinkEvery, dt: experiment.dt,
        monitorFinite: true, brains, seedOf: a => seeds[a], rowOf: () => 0 }));
    const lane = built.value;
    try {
        const observed = { ...await run(lane), setupCompileMs: built.setupCompileMs };
        // One mapping per matchup, after all T steps (I6).
        const [counters, laneStats] = await readBuffers(engine.device,
            [{ buffer: lane.counters, words: geometry.arenas }, { buffer: lane.stats, words: 4 }]);
        if (laneStats[3]) fail(`nonfinite runtime state in ${game.id}`);
        if (counters.reduce((sum, n) => sum + n, 0) !== laneStats[0] || counters.some(n => n > geometry.prey * T))
            fail(`catch counters of ${game.id} are inconsistent`);
        return { catches: Array.from(counters), observed };
    } finally { lane.destroy(); }
}
