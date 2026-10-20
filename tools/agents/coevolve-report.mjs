// Plain-text reports for people reading co-evolution results (docs/agents-coevo-runbook.md). Read-only.
//
//   node tools/agents/coevolve-report.mjs training <run directory> [--every 20]
//   node tools/agents/coevolve-report.mjs evaluation <evaluation directory>
//
// training: mean catches a tile over each block of generations, apart for the generations in which
// prey evolved and those in which wolves did. Training catches are against the current or an archived
// opponent, so they show the arms race, not skill against a fixed opponent.
// evaluation: each checkpoint's catches against the rule and the clones, paired by episode with 95%
// intervals; with --diagnostics-stride, the share of prey near a wall over time and what the wolves did.
import { existsSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const fail = message => { throw new Error(`coevolve: report: ${message}`); };
const fixed = (v, digits = 1) => (v >= 0 ? '+' : '') + v.toFixed(digits);

/** One line per block of `every` generations from a run's log.jsonl entries. */
export function trainingLines(entries, every = 20) {
    const gens = entries.filter(e => e.t !== undefined);
    if (!gens.length) fail('the log holds no completed generation');
    const lines = ['Catches a tile in training episodes, mean [lowest..highest candidate tile].',
        `${'generations'.padEnd(14)}${'when prey evolved'.padEnd(22)}when wolves evolved`];
    for (let b = 0; b < gens.length; b += every) {
        const block = gens.slice(b, b + every);
        const cell = species => {
            const rows = block.filter(g => g.species === species);
            if (!rows.length) return '—';
            const catches = rows.flatMap(g => g.scores.map(s => (species === 'prey' ? -s : s)));
            const mean = catches.reduce((a, c) => a + c, 0) / catches.length;
            return `${mean.toFixed(1).padStart(5)} [${Math.min(...catches)}..${Math.max(...catches)}]`;
        };
        lines.push(`${`${block[0].t + 1}–${block.at(-1).t + 1}`.padEnd(14)}${cell('prey').padEnd(22)}${cell('wolf')}`);
    }
    return lines;
}

/** A cross-play's catch matrix (rows: prey generations; columns: wolf generations) and its forgetting. */
export function crossplayLines({ generations, mean, anchors, forgetting }) {
    const lines = ['Cross-play: catches a tile; rows are prey generations, columns wolf generations.',
        `${'prey \\ wolves'.padEnd(14)}${generations.map(g => g.padStart(10)).join('')}`];
    generations.forEach((g, i) => lines.push(`${g.padEnd(14)}${mean[i].map(v => v.toFixed(1).padStart(10)).join('')}`));
    if (forgetting) {
        const f = x => `${x.value.toFixed(1)} [${x.low.toFixed(1)}, ${x.high.toFixed(1)}]`;
        lines.push(`Forgetting against ${anchors.join(', ')} (largest drop from an earlier to a later generation; 0 is none):`,
            `  prey ${f(forgetting.prey)}, wolves ${f(forgetting.wolf)}`);
    }
    return [...lines, ''];
}

/** The evaluation's comparisons against the rule and the clones, then its diagnostics if it has them. */
export function evaluationLines(evaluation) {
    const { results, diagnosticResults = [] } = evaluation;
    const mean = id => results.matchups.find(m => m.id === id)?.interval.mean ?? fail(`no matchup ${id}`);
    const adapter = evaluation.device?.adapter ?? {};
    const lines = [`${evaluation.evaluation.episodes} ${evaluation.heldOut?.purpose ?? ''} episodes of ${evaluation.evaluation.T} steps` +
        ` on ${[adapter.vendor, adapter.architecture].filter(Boolean).join(' ') || 'an unknown adapter'}; catches a tile.`,
    'Prey: fewer is better. Wolves: more is better. In brackets, the paired difference\'s 95% interval.', ''];
    const subjects = [...new Set(results.comparisons.map(c => c.subject))];
    if (evaluation.crossplay) lines.push(...crossplayLines(evaluation.crossplay));
    if (subjects.length) lines.push(`${'checkpoint'.padEnd(11)}${'against'.padEnd(10)}${'its prey caught (baseline prey)'.padEnd(40)}its wolves' catches (baseline wolves)`);
    for (const subject of subjects) for (const baseline of ['rule', 'clone']) {
        const cell = species => {
            const c = results.comparisons.find(x => x.subject === subject && x.baseline === baseline && x.species === species);
            if (!c) return '—';
            const d = c.difference, verdict = c.superior ? ' better' : (species === 'prey' ? d.low > 0 : d.high < 0) ? ' worse' : '';
            return `${mean(c.minuend).toFixed(1)} (${mean(c.subtrahend).toFixed(1)}) ${fixed(d.mean)} [${fixed(d.low)}, ${fixed(d.high)}]${verdict}`;
        };
        lines.push(`${subject.padEnd(11)}${baseline.padEnd(10)}${cell('prey').padEnd(40)}${cell('wolf')}`);
    }
    if (!diagnosticResults.length) return lines;
    const history = diagnosticResults[0].episodes[0].history;
    const marks = history.map((h, k) => [h.tick, k]).filter(([tick], k) => tick % 300 === 0 || k === history.length - 1);
    const dt = evaluation.experiment?.dt ?? 1 / 60;
    const average = (m, k, key) => {
        const v = m.episodes.map(e => e.history[k][key]).filter(x => x !== null);
        return v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN;
    };
    lines.push('', `Prey within 1 m of a wall (share of live prey) at ${marks.map(([tick]) => `${Math.round(tick * dt)} s`).join(', ')};`,
        'then, over the samples from 5 s on: wolves resting, prey mean speed (m/s), wolves to their nearest prey (m).');
    // Each checkpoint against itself and the rule both ways, and the rule and the clones as references;
    // a cross-play has no baselines, so all of its matchups.
    const shown = new Set(['rule/rule', 'clone/clone', ...subjects.flatMap(s => [`${s}/${s}`, `${s}/rule`, `rule/${s}`])]);
    for (const m of diagnosticResults.filter(m => !subjects.length || shown.has(m.id))) {
        const late = history.map((h, k) => k).filter(k => history[k].tick * dt >= 5);
        const over = key => late.reduce((s, k) => s + average(m, k, key), 0) / late.length;
        lines.push(`${m.id.padEnd(22)}${marks.map(([, k]) => average(m, k, 'preyWallShare').toFixed(2)).join('  ')}    ` +
            `${over('wolfRestFraction').toFixed(2)}  ${over('preyMeanSpeed').toFixed(2)}  ${over('nearestPreyMean').toFixed(2)}`);
    }
    return lines;
}

function main([mode, dir, ...rest]) {
    if (!dir) fail('usage: coevolve-report.mjs training <run directory> | evaluation <evaluation directory>');
    if (mode === 'training') {
        const file = join(dir, 'log.jsonl');
        if (!existsSync(file)) fail(`no log.jsonl in "${dir}"`);
        const i = rest.indexOf('--every'), every = i < 0 ? 20 : Number(rest[i + 1]);
        if (!(Number.isInteger(every) && every > 0)) fail('--every needs a whole number of generations');
        return trainingLines(readFileSync(file, 'utf8').trim().split('\n').map(line => JSON.parse(line)), every);
    }
    if (mode === 'evaluation') {
        const file = join(dir, 'evaluation.json');
        if (!existsSync(file)) fail(`no evaluation.json in "${dir}" (an evaluation still running has only plan.json)`);
        return evaluationLines(JSON.parse(readFileSync(file, 'utf8')));
    }
    fail(`mode must be training or evaluation, not "${mode}"`);
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
    try { console.log(main(process.argv.slice(2)).join('\n')); }
    catch (error) { console.error(error.message); process.exit(1); }
}
