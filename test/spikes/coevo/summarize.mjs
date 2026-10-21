// Derived report only; never substitutes for preserved seed-level raw evidence.
import { readFileSync, writeFileSync } from 'node:fs';
import { ranks, mean, sd } from './math.mjs';
const file = process.argv[2];
if (!file) throw new Error('Usage: node test/spikes/coevo/summarize.mjs <raw-evidence.json>');
const raw = JSON.parse(readFileSync(file, 'utf8'));
const report = { source: file, checks: raw.checks, adapters: [] };
const correlation = (a, b) => {
    const ma = mean(a), mb = mean(b), denom = Math.sqrt(a.reduce((s, v) => s + (v - ma) ** 2, 0) * b.reduce((s, v) => s + (v - mb) ** 2, 0));
    return denom ? a.reduce((s, v, i) => s + (v - ma) * (b[i] - mb), 0) / denom : null;
};
for (const a of raw.adapters) {
    const gates = a.gates, out = { name: a.name, gate1Passed: a.gate1Passed, error: a.error,
        memory: a.bufferMemory, pipelineCache: a.pipelineCache };
    out.rate = gates.gate2?.map(r => ({ cfg: r.cfg, ...r.summary, memory: r.bufferMemory,
        setupMedianMs: [...r.samples.map(s => s.setupMs)].sort((a, b) => a - b)[2],
        episodeMedianMs: [...r.samples.map(s => s.episodeMs)].sort((a, b) => a - b)[2] }));
    out.noise = (Array.isArray(gates.gate3) ? gates.gate3 : gates.gate3?.settings)?.map(r => {
        const first = r.directions.map(d => mean(d.differences.slice(0, 10)));
        const last = r.directions.map(d => mean(d.differences.slice(10)));
        const directionSD = sd(r.directions.map(d => d.mean));
        const residual = Math.sqrt(mean(r.directions.map(d => d.sd ** 2)));
        return { cfg: r.cfg, ...r.summary, splitHalfRankCorrelation: correlation([...ranks(first)], [...ranks(last)]),
            directionalSD: directionSD, residualPairedRMS: residual,
            signalToResidual: residual ? directionSD / residual : null,
            nonzeroIntervals: r.directions.filter(d => d.interval.low > 0 || d.interval.high < 0).length };
    });
    out.smoke = gates.gate4 && { cfg: gates.gate4.cfg, repeatExact: gates.gate4.repeatExact,
        paired: gates.gate4.paired, anchors: Object.fromEntries(Object.entries(gates.gate4.evaluations).map(([name, e]) =>
            [name, { catchesMean: mean(e.catches), histories: e.histories }])) };
    report.adapters.push(out);
}
const output = file.replace(/\.json$/, '-summary.json');
writeFileSync(output, JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));