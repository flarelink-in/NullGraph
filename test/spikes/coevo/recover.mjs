// Recover completed evidence after a safety stop/process kill. Never labels partial runs accepted.
import { readFileSync, writeFileSync } from 'node:fs';
const file = process.argv[2];
if (!file) throw new Error('Usage: node test/spikes/coevo/recover.mjs <events.jsonl>');
const lines = readFileSync(file, 'utf8').split('\n'), events = [];
let truncatedTail = false;
for (let i = 0; i < lines.length; i++) {
    if (!lines[i].trim()) continue;
    try { events.push(JSON.parse(lines[i])); }
    catch (error) {
        if (lines.slice(i + 1).some(l => l.trim())) throw error;
        truncatedTail = true;
    }
}
const report = { source: file, verdict: 'recovered-partial-not-acceptance', truncatedTail,
    eventCount: events.length, adapters: {} };
for (const event of events) {
    const adapter = event.data.adapter;
    if (!adapter) continue;
    const a = report.adapters[adapter] ??= { checks: [], generations: [], rate: [], noiseSamples: [], noiseSettings: [], failures: [] };
    if (event.type === 'check' || event.type === 'fixture') a.checks.push(event.data);
    if (event.type === 'generation-completed') a.generations.push(event.data);
    if (event.type === 'gate2-setting') a.rate.push(event.data);
    if (event.type === 'gate3-sample') a.noiseSamples.push(event.data);
    if (event.type === 'gate3-setting') a.noiseSettings.push(event.data);
    if (event.type.endsWith('-failed')) a.failures.push(event);
}
const output = file.replace(/\.jsonl$/, '-recovered.json');
if (output === file) throw new Error('Input must end in .jsonl');
writeFileSync(output, JSON.stringify(report, null, 1));
console.log(JSON.stringify({ output, eventCount: events.length, truncatedTail,
    adapters: Object.fromEntries(Object.entries(report.adapters).map(([name, a]) => [name,
        { generations: a.generations.length, noiseSamples: a.noiseSamples.length, failures: a.failures.length }])) }, null, 2));