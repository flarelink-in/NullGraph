// Step 7, part 2 (docs/agents-cloning-design.md §8): the copy in the running simulation.
//
// One run is a choice of driver per species: the teacher, the label's decisions (the rule without
// wander and coin-flips), or a student network. For each seed the scenario is stepped in lockstep
// with the label pass spliced in, as in a recording, so every observation the driver meets has a
// label. Every `stride` steps the observations, labels, tags and the driver's actions are copied
// out, and each decision is counted once.
//
// A run reports catches, the driver's mean absolute difference from the label on its own run (per
// species, action and reason code), and the scenario's own measures.
//
// No node: imports here: the GPU suite can run this in a browser as well. evaluate.mjs is the CLI.
import { TAG } from './scenario/herd-wolves.mjs';
import { createLabelPass, lockstep, readBuffers } from './label-pass.mjs';
import { isFresh } from './recorder.mjs';

export const EVALUATE_DEFAULTS = Object.freeze({
    seeds: Object.freeze([21, 22, 23, 24, 25]),     // none of them recorded: training saw 11–14, and 15 chose the pass
    steps: 1800,                                    // 30 s at 60 steps a second
    stride: 6,                                      // steps between copies, as in a recording
});

/** The design's five runs (§8), by driver per species of Herd & Wolves. */
export const RUNS = Object.freeze({
    T: Object.freeze({ prey: 'teacher', wolf: 'teacher' }),
    C: Object.freeze({ prey: 'label', wolf: 'label' }),
    S: Object.freeze({ prey: 'student', wolf: 'student' }),
    Sp: Object.freeze({ prey: 'student', wolf: 'teacher' }),
    Sw: Object.freeze({ prey: 'teacher', wolf: 'student' }),
});

/** Counts for one species: decisions, the driver's distance from the label, and the measures. */
export const newTally = (actDim, reasons, measures) => ({
    decisions: 0, frozen: 0, error: new Float64Array(actDim),
    reasons: Array.from({ length: reasons }, () => ({ rows: 0, error: new Float64Array(actDim) })),
    measures: Array.from({ length: measures }, () => ({ rows: 0, sum: 0 })),
});

/**
 * Counts one decision: `act` and `label` hold the driver's and the label's actions at `at`.
 * A frozen agent's action is never applied, so it is counted and compared with nothing.
 * Returns whether the agent was live.
 */
export function addDecision(tally, act, label, at, frozen, why) {
    tally.decisions++;
    if (frozen) {
        tally.frozen++;
        return false;
    }
    const group = tally.reasons[why];
    group.rows++;
    for (let k = 0; k < tally.error.length; k++) {
        const e = Math.abs(act[at + k] - label[at + k]);
        tally.error[k] += e;
        group.error[k] += e;
    }
    return true;
}

/** Adds `from`'s counts into `into`: the seeds of a run, pooled. */
export function addTally(into, from) {
    into.decisions += from.decisions;
    into.frozen += from.frozen;
    from.error.forEach((e, k) => { into.error[k] += e; });
    from.reasons.forEach((g, code) => {
        into.reasons[code].rows += g.rows;
        g.error.forEach((e, k) => { into.reasons[code].error[k] += e; });
    });
    from.measures.forEach((m, k) => {
        into.measures[k].rows += m.rows;
        into.measures[k].sum += m.sum;
    });
    return into;
}

/** A tally as means, by name. A mean over no rows is null. */
export function describeTally(tally, actNames, reasons, measures) {
    const mean = (sum, rows) => (rows ? sum / rows : null), live = tally.decisions - tally.frozen;
    const byAction = (error, rows) => Object.fromEntries(actNames.map((name, k) => [name, mean(error[k], rows)]));
    return {
        decisions: tally.decisions, frozen: tally.frozen, mae: byAction(tally.error, live),
        byReason: Object.fromEntries(tally.reasons.flatMap((g, code) => (g.rows ? [[reasons[code], { rows: g.rows, mae: byAction(g.error, g.rows) }]] : []))),
        measures: Object.fromEntries(measures.map((m, k) => [m.name, mean(tally.measures[k].sum, tally.measures[k].rows)])),
    };
}

/** Catches over a run's seeds: their mean, and the lowest and highest as a share of it. */
export function describeCatches(values) {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    return { values: [...values], mean, low: Math.min(...values) / mean - 1, high: Math.max(...values) / mean - 1 };
}

/** Refuses a driver that is not one of the three kinds, or a student without its network. */
function checkDriver(species, kind, students) {
    if (kind !== 'teacher' && kind !== 'label' && kind !== 'student') {
        throw new Error(`evaluate: the driver of "${species}" must be teacher, label or student; got "${kind}".`);
    }
    if (kind === 'student' && !students?.[species]) throw new Error(`evaluate: no student network for species "${species}".`);
}

/**
 * Runs `scenario` for each of `options.seeds` under `drivers` ({ <species>: 'teacher' | 'label' |
 * 'student' }). `options.students` maps a species to the brain that drives it as a student; the
 * caller owns those brains. Resolves to { drivers, episodes, catches, species }: `species` pools
 * the seeds, `episodes` has each.
 */
export async function evaluateRun(engine, scenario, drivers, options = {}) {
    const o = { ...EVALUATE_DEFAULTS, ...options }, scene = { ...scenario.defaults, ...options.scene };
    if (!(Number.isInteger(o.stride) && o.stride >= scene.thinkEvery)) {
        throw new Error(`evaluate: stride ${o.stride} must be a whole number of steps, at least thinkEvery (${scene.thinkEvery}).`);
    }
    if (!(o.seeds.length > 0 && Number.isInteger(o.steps) && o.steps >= o.stride)) {
        throw new Error(`evaluate: needs at least one seed, and steps (${o.steps}) of at least one stride (${o.stride}).`);
    }
    for (const [species, kind] of Object.entries(drivers)) checkDriver(species, kind, o.students);
    const device = engine.device, floats = words => new Float32Array(words.buffer, words.byteOffset, words.length);
    const episodes = [];
    let names, measures, pooled;
    for (const seed of o.seeds) {
        const sim = scenario.create(engine, { ...scene, seed });
        let label;
        try {
            const { definition } = sim, { capacity, obsDim, actDim } = definition;
            const rules = {};
            for (const s of definition.species) {
                const kind = drivers[s.name];
                checkDriver(s.name, kind, o.students);
                if (kind === 'label') sim.agents.setBrain(rules.label ??= sim.rule('label'), { species: s.name });
                else if (kind === 'student') sim.agents.setBrain(o.students[s.name], { species: s.name });
                // else the teacher, which the scenario starts with on every species
            }
            label = createLabelPass(engine, sim.agents, sim.label);
            const step = lockstep(engine, sim.agents, [label.stage]);
            names ??= definition.species.map(s => s.name);
            measures ??= scenario.measures(definition);
            const tallies = names.map(() => newTally(actDim, scenario.reasons.length, measures.length));
            let copied = -1;
            for (let s = 0; s < o.steps; s++) {
                await step();
                if ((s + 1) % o.stride) continue;
                const [obsWords, labelWords, tags, actWords] = await readBuffers(device, [sim.agents.obs, label.act, label.tags, sim.agents.act]);
                const obs = floats(obsWords), labels = floats(labelWords), act = floats(actWords);
                for (let i = 0; i < capacity; i++) {
                    if (!isFresh(TAG.tick(tags[i]), copied, s)) continue;
                    const species = sim.speciesOf(i), tally = tallies[species];
                    if (!addDecision(tally, act, labels, i * actDim, TAG.frozen(tags[i]) === 1, TAG.why(tags[i]))) continue;
                    for (let m = 0; m < measures.length; m++) {
                        const value = measures[m].of(obs, i * obsDim, species);
                        if (value !== null) {
                            tally.measures[m].rows++;
                            tally.measures[m].sum += value;
                        }
                    }
                }
                copied = s;
            }
            const [stats] = await readBuffers(device, [sim.stats]);
            pooled ??= names.map(() => newTally(actDim, scenario.reasons.length, measures.length));
            tallies.forEach((t, k) => addTally(pooled[k], t));
            episodes.push({ seed, steps: o.steps, catches: stats[0],
                species: Object.fromEntries(names.map((name, k) => [name, describeTally(tallies[k], definition.actNames, scenario.reasons, measures)])) });
        } finally {
            label?.destroy();
            sim.destroy();
        }
    }
    const actNames = Object.keys(episodes[0].species[names[0]].mae);
    return {
        drivers: Object.fromEntries(names.map(name => [name, drivers[name]])), episodes,
        catches: describeCatches(episodes.map(e => e.catches)),
        species: Object.fromEntries(names.map((name, k) => [name, describeTally(pooled[k], actNames, scenario.reasons, measures)])),
    };
}
