// Step 7, part 2 (docs/agents-cloning-design.md §10): the training tools on a real device.
// Runs on both Node/Dawn and Chrome through the shared GPU harness; build first.
//
//   A25  the label pass sees exactly what the system's own think sees
//   A24  a recording repeats, byte for byte
//   A26  the committed Herd & Wolves networks run as MlpBrains and compute what mlpForward computes
//   A8   cloning quality: the networks in the running simulation, against the label and its catches
import { NullGraph } from '../../dist/index.js';
import { MlpBrain, mlpForward } from '../../dist/agents.js';
import { harnessEnv } from '../gpu/harness-env.mjs';
import { DRIVER, LABEL, TAG, createHerd, herdDefinition, herdRule, herdWolves } from '../../tools/agents/scenario/herd-wolves.mjs';
import { createLabelPass, lockstep, readBuffers } from '../../tools/agents/label-pass.mjs';
import { recordEpisodes } from '../../tools/agents/recorder.mjs';
import { RUNS, evaluateRun } from '../../tools/agents/evaluator.mjs';
import { networkFromModel } from '../../tools/agents/model.mjs';

const { check, fatal, finish } = harnessEnv('__AGENT_TOOLS_RESULTS__');
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) { fatal('No WebGPU adapter'); throw new Error('No WebGPU adapter'); }
const device = await adapter.requestDevice();
const engine = new NullGraph();
const init = await engine.init(null, { device, width: 64, height: 64 });
if (!init.success) { fatal(init.error); throw new Error(init.error); }
const same = (a, b) => a.length === b.length && a.every((w, i) => w === b[i]);
const fmt = n => n.toLocaleString('en-US');
const FLAGS = 13, FROZEN = 2;      // the record's flags word, and its frozen bit

// A25. Given the driver's own text, a label pass must write the system's actions and reasons, word
// for word: the same gate, the same tick, the same inputs. Herd & Wolves respawns caught prey
// (a newborn thinks at once) and freezes resting wolves, so the run covers both.
async function watch(before, steps) {
    const herd = createHerd(engine), count = herd.definition.capacity, actDim = herd.definition.actDim;
    const pass = createLabelPass(engine, herd.agents, { wgsl: herdRule(DRIVER), buffers: [herd.settings] });
    const step = lockstep(engine, herd.agents, [pass.stage], before);
    let rows = 0, stepsOff = 0, frozen = 0;
    for (let s = 0; s < steps; s++) {
        await step();
        const [act, labels, reason, tags, records] = await readBuffers(device,
            [herd.agents.act, pass.act, herd.agents.reason, pass.tags, herd.agents.buffer]);
        let off = 0;
        for (let i = 0; i < count; i++) {
            if (reason[i] !== tags[i] || !same(act.subarray(i * actDim, (i + 1) * actDim), labels.subarray(i * actDim, (i + 1) * actDim))) off++;
            if (records[i * 16 + FLAGS] & FROZEN) frozen++;
        }
        if (off) { stepsOff++; rows += off; }
    }
    const [stats] = await readBuffers(device, [herd.stats]);
    pass.destroy();
    herd.destroy();
    return { rows, stepsOff, frozen, catches: stats[0] };
}
const placed = await watch(undefined, 300);
check('A25: given the driver\'s text, the label pass writes the system\'s own actions and reasons, word for word, over 300 steps',
    placed.rows === 0 && placed.catches > 0 && placed.frozen > 0,
    `${placed.rows} rows differ; ${placed.catches} catches, ${fmt(placed.frozen)} frozen agent-steps`);
const late = await watch('AgentSystem step catches', 60);
check('A25 watched failing: spliced after act, where a newborn\'s gate has already closed, the label pass differs',
    late.rows > 0, `${fmt(late.rows)} rows differ in ${late.stepsOff} of 60 steps`);
let refused = '';
try {
    const herd = createHerd(engine);
    try { lockstep(engine, herd.agents, [], 'AgentSystem acts'); } catch (error) { refused = error.message; } finally { herd.destroy(); }
} catch (error) { refused = `setup failed: ${error.message}`; }
check('A25: a step without the stage to splice at is refused by name',
    refused.startsWith('agents-tools: the step has no "AgentSystem acts" stage.'), refused);

// With the label's text: every decision is tagged with its step, the tag's frozen bit is the
// record's at think time, the label never rests, and its actions stay in [-1, 1].
{
    const herd = createHerd(engine), count = herd.definition.capacity, actDim = herd.definition.actDim, every = herd.agents.thinkEvery;
    const pass = createLabelPass(engine, herd.agents, herd.label);
    const step = lockstep(engine, herd.agents, [pass.stage]);
    let flags = new Uint32Array(count), untagged = 0, wrongFrozen = 0, rests = 0, outside = 0, frozenTags = 0, decisions = 0, preyFrozen = 0;
    for (let s = 0; s < 300; s++) {
        await step();
        const [tags, labels, records] = await readBuffers(device, [pass.tags, pass.act, herd.agents.buffer]);
        const actions = new Float32Array(labels.buffer, labels.byteOffset, labels.length);
        for (let i = 0; i < count; i++) {
            const tick = TAG.tick(tags[i]);
            if (tick > s || tick <= s - every) untagged++;          // every agent decides once per `every` steps
            if (tick !== s) continue;
            decisions++;
            if (TAG.frozen(tags[i]) !== ((flags[i] & FROZEN) ? 1 : 0)) wrongFrozen++;
            if (TAG.frozen(tags[i])) { frozenTags++; if (i >= herd.wolves) preyFrozen++; }
            if (TAG.why(tags[i]) > 4) rests++;
            for (let k = 0; k < actDim; k++) if (!(Math.abs(actions[i * actDim + k]) <= 1)) outside++;
        }
        flags = Uint32Array.from({ length: count }, (_, i) => records[i * 16 + FLAGS]);
    }
    check('A25: the label\'s tags name each decision\'s step, for every agent, once per think cycle',
        untagged === 0 && decisions >= 300 * Math.floor(count / every), `${untagged} stale tags; ${fmt(decisions)} decisions in 300 steps`);
    check('A25: a tag\'s frozen bit is the record\'s at think time; only wolves freeze; the label never rests; actions stay in [-1, 1]',
        wrongFrozen === 0 && frozenTags > 0 && preyFrozen === 0 && rests === 0 && outside === 0,
        `${wrongFrozen} wrong bits, ${fmt(frozenTags)} frozen decisions, ${preyFrozen} of them prey, ${rests} rests, ${outside} actions outside`);
    pass.destroy();
    herd.destroy();
}

// A24. A recording is a pure function of the build, the scenario, the seeds and the device.
{
    const options = { seeds: [11, 12], steps: 180, stride: 6 };
    const first = await recordEpisodes(engine, herdWolves, options), second = await recordEpisodes(engine, herdWolves, options);
    const other = await recordEpisodes(engine, herdWolves, { ...options, seeds: [21, 22] });
    const bits = recording => Object.values(recording.species).flatMap(s => [new Uint32Array(s.f32.buffer), s.u32]);
    const equal = (a, b) => bits(a).every((words, k) => same(words, bits(b)[k]));
    check('A24: two recordings of the same seeds are byte-identical, rows and words, for every species',
        equal(first, second) && JSON.stringify(first.header) === JSON.stringify(second.header) &&
        JSON.stringify(first.report) === JSON.stringify(second.report),
        Object.entries(first.species).map(([name, s]) => `${name} ${fmt(s.rows)} rows`).join(', '));
    check('A24: a recording of other seeds is not', !equal(first, other));
    const [prey, wolf] = first.report, catches = first.header.episodes.map(e => e.catches);
    check('A24: every decision is seen once per copy; prey are thinned to their share, wolves kept whole',
        prey.seen === 2 * 30 * 3840 && wolf.seen === 2 * 30 * 256 && wolf.rows === wolf.seen &&
        Math.abs(prey.rows / prey.seen - 1 / 16) < 0.005 && first.species.prey.rows === prey.rows && first.species.wolf.rows === wolf.rows,
        `prey ${fmt(prey.rows)} of ${fmt(prey.seen)}, wolves ${fmt(wolf.rows)} of ${fmt(wolf.seen)}`);
    check('A24: the run is the game: wolves catch and rest, and the last episode is held out',
        catches.every(n => n > 0) && wolf.frozen > 0 && prey.frozen === 0 &&
        first.header.episodes.map(e => e.heldOut).join() === 'false,true',
        `catches ${catches.join(', ')}; ${fmt(wolf.frozen)} frozen wolf rows`);
    check('A24: the driver differs from the label where the rule wanders, and only in the turn',
        prey.hidden[0] > 0.01 && prey.hidden[1] === 0 && wolf.hidden[1] === 0,
        `prey turn ${prey.hidden[0].toFixed(4)}, wolf turn ${wolf.hidden[0].toFixed(4)}`);
}

// The committed networks, read as an app reads them: parse, check the names, one MlpBrain per species.
const readJson = async path => {
    const url = new URL(path, import.meta.url);
    return typeof document !== 'undefined' ? (await fetch(url)).json() : JSON.parse((await import('node:fs')).readFileSync(url, 'utf8'));
};
const networks = {}, students = {};
for (const species of ['prey', 'wolf']) {
    const file = `../../tools/agents/models/herd-wolves/${species}.mlp.json`;
    networks[species] = networkFromModel(await readJson(file), herdDefinition(), file);
    students[species] = new MlpBrain(engine, networks[species]);
}

// A26. With a network on each species, every action written in a step is mlpForward of that agent's
// observation in that step. The label pass's tags say who thought.
// The bound is 1e-4, not A4's 1e-5 + 1e-5·|cpu|: trained weights are large (up to 178 here, against
// A4's unit-scale ones), and f32 rounding grows with them. How many actions pass A4's bound is reported.
{
    const herd = createHerd(engine), { capacity, obsDim, actDim } = herd.definition, names = herd.definition.species.map(s => s.name);
    for (const species of names) herd.agents.setBrain(students[species], { species });
    const pass = createLabelPass(engine, herd.agents, herd.label);
    const step = lockstep(engine, herd.agents, [pass.stage]);
    const floats = words => new Float32Array(words.buffer, words.byteOffset, words.length), cpu = new Float32Array(actDim);
    const compared = names.map(() => 0);
    const TOL = 1e-4;
    let bad = 0, beyondA4 = 0, worst = 0, swapped = 0;
    for (let s = 0; s < 60; s++) {
        await step();
        const [obsWords, actWords, tags] = await readBuffers(device, [herd.agents.obs, herd.agents.act, pass.tags]);
        const obs = floats(obsWords), act = floats(actWords);
        for (let i = 0; i < capacity; i++) {
            if (TAG.tick(tags[i]) !== s) continue;
            const species = herd.speciesOf(i), row = obs.subarray(i * obsDim, (i + 1) * obsDim);
            compared[species]++;
            mlpForward(networks[names[species]], row, cpu);
            for (let k = 0; k < actDim; k++) {
                const d = Math.abs(act[i * actDim + k] - cpu[k]);
                worst = Math.max(worst, d);
                if (!(d <= TOL)) bad++;
                if (!(d <= 1e-5 + 1e-5 * Math.abs(cpu[k]))) beyondA4++;
            }
            // The other species' network must not pass for this one's.
            mlpForward(networks[names[1 - species]], row, cpu);
            if (!(Math.abs(act[i * actDim] - cpu[0]) <= TOL)) swapped++;
        }
    }
    pass.destroy();
    herd.destroy();
    const rows = compared[0] + compared[1];
    check(`A26: the committed networks load into one MlpBrain per species, and each action equals mlpForward of its observation within ${TOL}`,
        bad === 0 && compared.every(n => n > 0), `${bad} of ${fmt(rows * actDim)} actions out of tolerance; worst difference ${worst.toExponential(1)}; ` +
        `${beyondA4} beyond A4's bound; ` +
        names.map((name, k) => `${name} ${fmt(compared[k])} decisions`).join(', '));
    check('A26 watched failing: judged by the other species\' network, the turns differ',
        swapped > rows / 2, `${fmt(swapped)} of ${fmt(rows)} turns out of tolerance`);
}

// A8 (the parent's cloning check, with the bars of docs/agents-cloning-design.md §8): 30 s on 5 seeds
// that no recording used. C drives with the label's own decisions; S with the networks.
{
    const BAR_ERROR = 0.05, BAR_CATCHES = 0.10;
    const C = await evaluateRun(engine, herdWolves, RUNS.C), S = await evaluateRun(engine, herdWolves, RUNS.S, { students });
    const errors = run => Object.entries(run.species).flatMap(([species, s]) => Object.entries(s.mae).map(([action, e]) => ({ species, action, e })));
    check('A8: run C drives with the label\'s decisions, so it differs from its label by nothing',
        errors(C).every(x => x.e === 0) && C.catches.mean > 0, `catches ${C.catches.values.join(', ')}`);
    check(`A8: the networks' error against the label on their own runs is at most ${BAR_ERROR}, per species and action`,
        errors(S).every(x => x.e <= BAR_ERROR), errors(S).map(x => `${x.species} ${x.action} ${x.e.toFixed(4)}`).join(', '));
    const share = S.catches.mean / C.catches.mean - 1;
    check(`A8: the networks' mean catches over 5 seeds are within ${100 * BAR_CATCHES}% of the label's`,
        Math.abs(share) <= BAR_CATCHES, `${S.catches.mean.toFixed(1)} against ${C.catches.mean.toFixed(1)} (${(100 * share).toFixed(1)}%); ` +
        `networks ${S.catches.values.join(', ')}; label ${C.catches.values.join(', ')}`);
}
for (const brain of Object.values(students)) brain.destroy();

await finish('All agent-tools GPU checks passed.');
device.destroy();
