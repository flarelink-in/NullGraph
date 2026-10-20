// Layout-aware conservative guard. Counts are real arena counts, never rounded body populations.
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { release } from 'node:os';
import { createRequire } from 'node:module';
import { checkStep, predictStep, readCosts, saveCosts } from '../film/guard.mjs';
import { canonicalJson } from './coevolve-config.mjs';

export const WORKLOADS = ['ordinary', 'dense', 'wall', 'coincident', 'frozen'];
const fail = message => { throw new Error(`coevolve: guard: ${message}`); };
const gcd = (a, b) => b ? gcd(b, a % b) : a;
// The films' stop, with the trainer's prefix (design §12).
const stop = ms => {
    try { return checkStep(ms); }
    catch (error) { fail(error.message.replace(/^film: /, '')); }
};

export function probeLayouts(geometry) {
    const unit = 64 / gcd(64, geometry.wolves), result = [];
    if (geometry.arenas % unit || geometry.prey % 64) fail('invalid calibration arena alignment');
    for (let arenas = unit; ; arenas = Math.min(geometry.arenas, arenas * 2)) {
        result.push({ ...geometry, arenas, cols: Math.min(geometry.cols, arenas) });
        if (arenas === geometry.arenas) break;
    }
    return result;
}

export function guardIdentity(config, adapter, networks) {
    const hash = createHash('sha256');
    const { name, targetDir, configHash, ...settings } = config;
    hash.update(canonicalJson({ settings, networks, adapter: Object.fromEntries(
        ['vendor', 'architecture', 'device', 'description', 'driver'].map(key => [key, adapter?.[key] ?? 'unknown'])),
        os: release(), node: process.version, platform: process.platform,
        dawn: createRequire(import.meta.url)('webgpu/package.json').version,
        flags: process.env.NG_DAWN_FLAGS ?? '', workloads: WORKLOADS,
        sensorGrid: { cellSize: 8, maxScanPerCell: 64 } }));
    for (const path of ['../../dist/index.js', '../../dist/agents.js', '../../dist/physics.js',
        './coevolve.mjs', './coevolve-guard.mjs', './coevolve-runtime.mjs', './es.mjs',
        './scenario/herd-wolves.mjs', './scenario/herd-wolves-arenas.mjs', './scenario/arena-layout.mjs', '../film/guard.mjs'])
        hash.update(readFileSync(new URL(path, import.meta.url)));
    return hash.digest('hex');
}

export async function calibrateArenaSteps({ geometry, measure, cacheFile, key = '', log = () => {}, record = () => {} }) {
    let worst = 0;
    const evidence = [];
    for (const workload of WORKLOADS) {
        const cacheKey = `${key}:${workload}`;
        const samples = cacheFile ? readCosts(cacheFile, cacheKey) : [];
        for (const layout of probeLayouts(geometry)) {
            const count = layout.arenas * (layout.prey + layout.wolves);
            if (count < 64 || count % 16) fail('calibration count must be >=64 and divisible by 16');
            const prediction = predictStep(samples, count);
            if (!prediction && count > 4096) fail(`initial ${workload} layout needs ${count} bodies without small-probe evidence`);
            if (prediction?.predictedMs > 1000) fail(`unsafe ${workload} probe predicted ${prediction.predictedMs.toFixed(1)} ms`);
            // Fresh probes remain mandatory even with cache: cache evidence can refuse, not waive tests.
            const sample = await measure(layout, workload);
            const ms = stop(sample.ms);
            if (!(ms > 0)) fail('invalid calibration timing');
            samples.push({ count, ms });
            if (cacheFile) saveCosts(cacheFile, cacheKey, samples);
            evidence.push({ workload, count, ...sample });
            record(evidence.at(-1));
            log(`guard: ${workload} ${count} bodies: maximum ${ms.toFixed(1)} ms/step`);
            if (ms > 1000) fail(`${workload} measured step exceeds 1000 ms admission limit`);
            worst = Math.max(worst, ms);
        }
    }
    return { stepMs: worst, evidence, targetMs: 500, stopMs: 1500 };
}

export function createSubmissionGuard(stepMs) {
    if (!Number.isFinite(stepMs) || stepMs <= 0 || stepMs > 1000) fail('unsafe step prediction');
    let worst = stepMs, cap = Infinity;
    return {
        next(remaining) {
            if (!Number.isSafeInteger(remaining) || remaining < 1) fail('invalid remaining steps');
            if (worst > 1000) fail('unsafe step prediction');
            return Math.min(remaining, cap, Math.max(1, Math.floor(500 / worst)));
        },
        observe(count, ms) {
            stop(ms);
            if (!Number.isSafeInteger(count) || count < 1) fail('invalid submitted count');
            worst = Math.max(worst, ms / count);
            if (ms > 500) cap = Math.max(1, Math.floor(count * 400 / ms));
        },
    };
}