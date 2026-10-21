// Replacement S7, protocol 2. Raw data in out/; original S7 results are withdrawn.
import { harnessEnv } from '../../gpu/harness-env.mjs';
import { NullGraph } from '../../../dist/index.js';
import { networkFromModel } from '../../../tools/agents/model.mjs';
import { herdDefinition } from '../../../tools/agents/scenario/herd-wolves.mjs';
import { initNetwork } from '../../../tools/agents/mlp-train.mjs';
import { gate1, gate2, gate3, gate4 } from './gates.mjs';
import { createJournal, installEvidenceSink, record } from './evidence.mjs';
const env = harnessEnv('__S7_RESULTS__');
const { log, check, finish, inBrowser } = env;
const part = (inBrowser ? new URLSearchParams(location.search).get('part') : process.env.NG_SPIKE_PART) ?? 'all';
if (!['all', 'gate1', 'gate2', 'gate3', 'gate4'].includes(part)) throw new Error('S7: part must be all or gate1..gate4');
const load = async url => inBrowser ? (await fetch(url)).json() : JSON.parse((await import('node:fs')).readFileSync(url, 'utf8'));
const protocol = await load(new URL('./protocol.json', import.meta.url));
const networks = {};
for (const name of ['prey', 'wolf']) networks[name] = networkFromModel(await load(new URL(`../../../tools/agents/models/herd-wolves/${name}.mlp.json`, import.meta.url)), herdDefinition(), name);
const h2 = Object.fromEntries(['prey', 'wolf'].map((name, i) => [name, initNetwork([
    { input: 24, output: 16, activation: 'tanh' }, { input: 16, output: 16, activation: 'tanh' },
    { input: 16, output: 2, activation: 'tanh' }], 1001 + i)]));
const evidence = { protocol, part, runtime: inBrowser ? navigator.userAgent : process.version,
    date: new Date().toISOString(), adapters: [] };
const journal = await createJournal({ browser: inBrowser,
    file: new URL(`./out/events-${evidence.date.replace(/[^0-9]/g, '')}.jsonl`, import.meta.url) });
let currentAdapter = null;
installEvidenceSink((type, data) => journal.write(type, { adapter: currentAdapter, ...data }));
globalThis.__S7_EVENTS__ = journal.events;
record('manifest', { protocol, part, runtime: evidence.runtime });
if (!inBrowser) {
    const { execFileSync } = await import('node:child_process');
    const { createHash } = await import('node:crypto');
    const { readFileSync } = await import('node:fs');
    evidence.commit = execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
    evidence.dirty = execFileSync('git', ['status', '--short'], { encoding: 'utf8' });
    evidence.command = { argv: process.argv, part, dawnFlags: process.env.NG_DAWN_FLAGS ?? '' };
    evidence.sources = {};
    for (const f of ['run.js', 'arena.mjs', 'experiment.mjs', 'math.mjs', 'gates.mjs', 'fixtures.mjs', 'evidence.mjs', 'protocol.json'])
        evidence.sources[f] = createHash('sha256').update(readFileSync(new URL(f, import.meta.url))).digest('hex');
}
record('source-manifest', { sources: evidence.sources, commit: evidence.commit, dirty: evidence.dirty });
const seen = new Set();
try {
    for (const powerPreference of ['high-performance', 'low-power']) {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference });
        if (!adapter) continue;
        const info = adapter.info ?? {}, key = JSON.stringify([info.vendor, info.architecture, info.device, info.description]);
        if (seen.has(key)) continue; seen.add(key);
        const name = [info.vendor, info.architecture, info.description].filter(Boolean).join(' ');
        currentAdapter = name;
        const device = await adapter.requestDevice();
        const adapterErrors = [];
        device.addEventListener('uncapturederror', event => {
            adapterErrors.push(event.error?.message ?? String(event.error));
            record('gpu-failed', { error: adapterErrors.at(-1) });
        });
        // Caller-visible GPUBuffer allocation ledger: includes private grids/scratch/tables.
        // Texture/pipeline memory is not exposed by WebGPU and is reported as unmeasured.
        const ledger = { liveBytes: 0, peakBytes: 0, allocations: 0, texturePipelineBytes: 'not exposed by WebGPU' };
        const createBuffer = device.createBuffer.bind(device);
        device.createBuffer = desc => {
            const b = createBuffer(desc), destroy = b.destroy.bind(b); let live = true;
            ledger.liveBytes += b.size; ledger.peakBytes = Math.max(ledger.peakBytes, ledger.liveBytes); ledger.allocations++;
            b.destroy = () => { if (live) { ledger.liveBytes -= b.size; live = false; } destroy(); };
            return b;
        };
        device.__s7Ledger = ledger;
        // Tool-local immutable pipeline cache. Every caller still receives a NEW stage and
        // bind groups; no episode buffer/tick/contact state is shared. Default auto layouts
        // are pipeline-owned and safe to reuse with that same cached pipeline.
        const modules = new WeakMap(), pipelines = new Map(), pending = [];
        const createModule = device.createShaderModule.bind(device), createPipeline = device.createComputePipeline.bind(device);
        const pipelineCache = { hits: 0, misses: 0, compileMs: 0, policy: 'same-device full-WGSL/entry/constant cache; immutable pipelines only' };
        device.createShaderModule = desc => { const m = createModule(desc); modules.set(m, desc.code); return m; };
        device.createComputePipeline = desc => {
            if (desc.layout !== 'auto' || !modules.has(desc.compute.module)) return createPipeline(desc);
            const key = JSON.stringify([modules.get(desc.compute.module), desc.compute.entryPoint, desc.compute.constants ?? {}]);
            if (pipelines.has(key)) { pipelineCache.hits++; return pipelines.get(key); }
            const time = performance.now(), pipeline = createPipeline(desc);
            pipelineCache.compileMs += performance.now() - time; pipelineCache.misses++;
            // Explicit non-scored compilation completion, never an extra simulation step.
            pending.push(device.createComputePipelineAsync(desc).then(() => {}));
            pipelines.set(key, pipeline); return pipeline;
        };
        device.__s7Prepare = async () => {
            if (!pending.length) return;
            const tasks = pending.splice(0), start = performance.now();
            await Promise.all(tasks);
            record('pipeline-preparation', { count: tasks.length, ms: performance.now() - start });
        };
        const result = { name, info: { vendor: info.vendor, architecture: info.architecture, description: info.description }, gates: {}, bufferMemory: ledger, pipelineCache };
        result.verdict = 'pending';
        evidence.adapters.push(result);
        const engine = new NullGraph();
        try {
            const init = await engine.init(null, { device, width: 64, height: 64 });
            if (!init.success) throw new Error(init.error);
            log(`S7 replacement protocol 2: ${name}, ${part}; gate 1 is always mandatory`);
            const at = env.results.length;
            const checked = (n, c, d) => { check(`${name}: ${n}`, c, d); record('check', { name: n, pass: !!c, detail: d }); };
            result.gates.gate1 = await gate1(engine, networks, checked);
            await device.queue.onSubmittedWorkDone();
            await new Promise(resolve => setTimeout(resolve, 50));
            checked('G1 no uncaptured GPU errors', adapterErrors.length === 0, adapterErrors.join('\n'));
            result.gate1Passed = env.results.slice(at).every(r => r.pass);
            if (!result.gate1Passed) { result.verdict = 'blocked-correctness'; log('S7: later gates BLOCKED by correctness failure'); continue; }
            if (part === 'all' || part === 'gate2') result.gates.gate2 = await gate2(engine, { C2: networks, H2: h2 }, protocol, check, log);
            if (part === 'all' || part === 'gate3' || part === 'gate4') {
                const screen = await gate3(engine, networks, protocol, check, log);
                result.gates.gate3 = { settings: screen, qualityAccepted: screen.qualityAccepted };
                const smokeQuality = ['prey', 'wolf'].every(species => screen.some(r => r.cfg.species === species && r.quality.accepted &&
                    r.cfg.E === protocol.smoke.E && r.cfg.T === protocol.smoke.T && r.cfg.sigma === protocol.smoke.sigma));
                if (!screen.qualityAccepted || ((part === 'all' || part === 'gate4') && !smokeQuality)) {
                    result.verdict = 'blocked-score-quality';
                    log('S7: Gate 4 BLOCKED by paired score-quality failure');
                    continue;
                }
            }
            if (part === 'all' || part === 'gate4') result.gates.gate4 = await gate4(engine, networks, protocol, check, log);
            result.verdict = 'selected-checks-completed-not-settings-acceptance';
        } catch (e) { result.verdict = 'blocked-error-or-safety-stop'; result.error = String(e.stack ?? e); record('adapter-failed', { error: result.error }); check(`${name}: completed selected gates`, false, result.error); }
        finally { await device.queue.onSubmittedWorkDone().catch(() => {}); device.destroy(); }
    }
    if (!seen.size) check('WebGPU adapter exists', false);
} finally {
    await finish('S7 replacement selected gates passed');
    evidence.checks = globalThis.__S7_RESULTS__;
    globalThis.__S7_EVIDENCE__ = evidence;
    if (!inBrowser) {
        const { mkdirSync, writeFileSync } = await import('node:fs');
        const out = new URL('./out/', import.meta.url); mkdirSync(out, { recursive: true });
        const tag = evidence.date.replace(/[^0-9]/g, '');
        writeFileSync(new URL(`node-${part}-${tag}.json`, out), JSON.stringify(evidence, null, 1));
    }
    log(`S7 replacement JSON ${JSON.stringify(evidence)}`);
    log('S7 done');
    record('finished', { checks: evidence.checks }); journal.close();
}
