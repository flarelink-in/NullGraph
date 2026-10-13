// Step 7, part 2 (docs/agents-cloning-design.md §6): what a recording run does.
//
// For each episode: spawn the scenario for a seed, step it in lockstep with a label pass spliced in,
// and every `stride` steps copy out the observations, the labels, their tags and the driver's
// actions. A row is kept once per decision (its tag is newer than the last copy), and then by a
// hash of the agent, the tick and the species, so no species swamps the files.
//
// On one device, the same library build, scenario and seeds give the same rows, byte for byte
// (docs/agents-ordered-grid-design.md: lockstep reruns are bit-exact).
//
// No node: imports here: the GPU suite runs this in a browser as well. record.mjs is the CLI.
import { TAG } from './scenario/herd-wolves.mjs';
import { createLabelPass, lockstep, readBuffers } from './label-pass.mjs';
import { Rows, describeDefinition, keepRow, rowFlags, rowFloats } from './dataset.mjs';

export const RECORD_DEFAULTS = Object.freeze({
    seeds: Object.freeze([11, 12, 13, 14, 15]),
    heldOut: 1,                 // the last episodes, kept out of training
    steps: 1800,                // 30 s at 60 steps a second
    stride: 6,                  // steps between copies; at least thinkEvery, so no decision is missed twice
    keep: Object.freeze({ prey: 1 / 16, wolf: 1 }),
});

/**
 * Whether a decision tagged `tick` belongs to the stretch that ends at step `now`: newer than the
 * last copy, so it is kept once, and not a row no pass has written (tick -1) or an agent that has
 * stopped deciding.
 */
export const isFresh = (tick, copied, now) => tick > copied && tick <= now;

/**
 * Records `options.seeds` episodes of `scenario` on `engine`. Resolves to
 * { header, species: { <name>: { rows, f32, u32 } }, report }: `header` lacks only the files and
 * where it was made, which dataset.mjs and the CLI add. `scene` overrides the scenario's defaults.
 */
export async function recordEpisodes(engine, scenario, options = {}) {
    const o = { ...RECORD_DEFAULTS, ...options, keep: { ...RECORD_DEFAULTS.keep, ...options.keep } };
    const scene = { ...scenario.defaults, ...options.scene };
    if (!(Number.isInteger(o.stride) && o.stride >= scene.thinkEvery)) {
        throw new Error(`record: stride ${o.stride} must be a whole number of steps, at least thinkEvery (${scene.thinkEvery}).`);
    }
    if (!(o.seeds.length > 0 && Number.isInteger(o.steps) && o.steps >= o.stride)) {
        throw new Error(`record: needs at least one seed, and steps (${o.steps}) of at least one stride (${o.stride}).`);
    }
    const device = engine.device;
    let definition, rows, report;
    const episodes = [];
    for (const [episode, seed] of o.seeds.entries()) {
        const sim = scenario.create(engine, { ...scene, seed });
        let label;
        try {
            label = createLabelPass(engine, sim.agents, sim.label);
            const step = lockstep(engine, sim.agents, [label.stage]);
            if (!definition) {
                definition = sim.definition;
                rows = definition.species.map(() => new Rows(rowFloats(definition)));
                report = definition.species.map(s => ({ species: s.name, seen: 0, rows: 0, frozen: 0,
                    reasons: scenario.reasons.map(() => 0), hidden: definition.actNames.map(() => 0), compared: 0 }));
            }
            const { capacity, obsDim, actDim } = definition, rates = definition.species.map(s => o.keep[s.name] ?? 1);
            let copied = -1;
            for (let s = 0; s < o.steps; s++) {
                await step();
                if ((s + 1) % o.stride) continue;
                const [obs, labels, tags, act] = await readBuffers(device, [sim.agents.obs, label.act, label.tags, sim.agents.act]);
                const labelFloats = new Float32Array(labels.buffer, labels.byteOffset, labels.length);
                const actFloats = new Float32Array(act.buffer, act.byteOffset, act.length);
                for (let i = 0; i < capacity; i++) {
                    const tick = TAG.tick(tags[i]);
                    if (!isFresh(tick, copied, s)) continue;
                    const species = sim.speciesOf(i), tally = report[species];
                    tally.seen++;
                    if (!keepRow(i, tick, species, rates[species])) continue;
                    const frozen = TAG.frozen(tags[i]) === 1, why = TAG.why(tags[i]);
                    rows[species].push(obs.subarray(i * obsDim, (i + 1) * obsDim), labels.subarray(i * actDim, (i + 1) * actDim),
                        act.subarray(i * actDim, (i + 1) * actDim), episode, tick, i, rowFlags(frozen, why));
                    tally.rows++;
                    if (frozen) { tally.frozen++; continue; }
                    tally.reasons[why]++;
                    tally.compared++;
                    for (let k = 0; k < actDim; k++) tally.hidden[k] += Math.abs(actFloats[i * actDim + k] - labelFloats[i * actDim + k]);
                }
                copied = s;
            }
            const [stats] = await readBuffers(device, [sim.stats]);
            episodes.push({ seed, steps: o.steps, heldOut: episode >= o.seeds.length - o.heldOut,
                driver: Object.fromEntries(definition.species.map(s => [s.name, 'teacher'])), catches: stats[0] });
        } finally {
            label?.destroy();
            sim.destroy();
        }
    }
    const header = {
        scenario: { name: scenario.name, settings: scenario.settings(scene) },
        definition: describeDefinition(definition, scene.thinkEvery),
        reasons: [...scenario.reasons],
        episodes,
        sampling: { stride: o.stride, keep: Object.fromEntries(definition.species.map(s => [s.name, o.keep[s.name] ?? 1])) },
    };
    const species = Object.fromEntries(definition.species.map((s, k) => [s.name, rows[k].finish()]));
    return { header, species, report: report.map(r => ({ ...r, hidden: r.hidden.map(sum => (r.compared ? sum / r.compared : 0)) })) };
}

/** Lines that say what a recording holds: the measurements phase 7.6 reports. */
export function describeRecording({ header, species, report }) {
    const lines = [], actNames = header.definition.actNames, percent = (n, of) => (of ? `${(100 * n / of).toFixed(1)}%` : '—');
    for (const r of report) {
        const live = r.rows - r.frozen;
        lines.push(`${r.species}: ${species[r.species].rows.toLocaleString('en-US')} rows kept of ${r.seen.toLocaleString('en-US')} decisions seen; ` +
            `${percent(r.frozen, r.rows)} frozen`);
        lines.push(`   why the label acted: ${header.reasons.map((name, k) => `${name} ${percent(r.reasons[k], live)}`).join(', ')}`);
        lines.push(`   driver against label, mean absolute difference: ${actNames.map((name, k) => `${name} ${r.hidden[k].toFixed(4)}`).join(', ')}`);
    }
    const catches = header.episodes.map(e => e.catches), mean = catches.reduce((a, b) => a + b, 0) / catches.length;
    lines.push(`catches per episode of ${header.episodes[0].steps} steps: ${catches.join(', ')} ` +
        `(mean ${mean.toFixed(1)}, ${(100 * (Math.min(...catches) / mean - 1)).toFixed(1)}% to +${(100 * (Math.max(...catches) / mean - 1)).toFixed(1)}%)`);
    return lines;
}
