// D2.3: conservative step predictions and disposable, workload-specific adapter costs.
// Wall timings include queue completion; they are deliberately not labelled GPU timestamps.
import { readFileSync, mkdirSync, writeFileSync, renameSync } from 'node:fs';
import { dirname } from 'node:path';

export const STEP_LIMIT_MS = 1000;
export const STEP_STOP_MS = 1500;

export function checkStep(ms, stopMs = STEP_STOP_MS) {
    if (!Number.isFinite(ms) || ms < 0) throw new Error('film: invalid step timing.');
    if (ms > stopMs) throw new Error(`film: a step took ${Math.round(ms).toLocaleString('en-US')} ms, over the ${stopMs.toLocaleString('en-US')} ms stop. Film fewer agents, or on a faster GPU.`);
    return ms;
}

const valid = s => Number.isSafeInteger(s?.count) && s.count > 0 && Number.isFinite(s.ms) && s.ms > 0;

/** Largest measured count <= N; extrapolation uses the demo's conservative N^1.25 rule. */
export function predictStep(samples, count) {
    const from = samples.filter(s => valid(s) && s.count <= count).sort((a, b) => b.count - a.count || b.ms - a.ms)[0];
    return from ? { predictedMs: from.ms * (count / from.count) ** 1.25, from } : null;
}

/** Bad or old caches are never evidence. Cache failure does not prevent fresh calibration. */
export function readCosts(file, key) {
    try {
        const cache = JSON.parse(readFileSync(file, 'utf8'));
        return cache.version === 1 && Array.isArray(cache.entries?.[key]) ? cache.entries[key].filter(valid) : [];
    } catch { return []; }
}

export function saveCosts(file, key, samples) {
    let cache = { version: 1, entries: {} };
    try {
        const old = JSON.parse(readFileSync(file, 'utf8'));
        if (old.version === 1 && old.entries && typeof old.entries === 'object' && !Array.isArray(old.entries)) cache = old;
    } catch { /* Missing or corrupt: start over. */ }
    // Keep the worst observation at each size, never a faster sample that weakens the guard.
    const worst = new Map();
    for (const s of [...readCosts(file, key), ...samples].filter(valid)) worst.set(s.count, Math.max(worst.get(s.count) ?? 0, s.ms));
    cache.entries[key] = [...worst].map(([count, ms]) => ({ count, ms }));
    mkdirSync(dirname(file), { recursive: true });
    const temp = `${file}.${process.pid}.tmp`;
    writeFileSync(temp, JSON.stringify(cache, null, 1));
    renameSync(temp, file);
}

/** Calibrate progressively; never submit a probe already predicted over the limit. */
export async function calibrateSteps({ count, measure, samples = [], limitMs = STEP_LIMIT_MS, stopMs = STEP_STOP_MS, log = () => {}, save = () => {} }) {
    if (!Number.isSafeInteger(count) || count < 64 || count % 16) throw new Error('film: guard count must be a multiple of 16, at least 64.');
    if (!(limitMs > 0 && limitMs <= STEP_LIMIT_MS && stopMs > 0 && stopMs <= STEP_STOP_MS)) throw new Error('film: guard limits must be positive and may only be lowered.');
    samples = samples.filter(valid);
    let cached = true;
    const refuse = prediction => {
        throw new Error(`film: ${count.toLocaleString('en-US')} agents: ~${prediction.predictedMs.toFixed(0)} ms per step predicted on this GPU ` +
            `(${(prediction.from.ms / prediction.from.count * 1e6).toFixed(0)} ns per agent at ${prediction.from.count.toLocaleString('en-US')}), ` +
            `over the ${limitMs} ms limit per submission (Windows resets a GPU busy for about 2 s).`);
    };
    let prediction = predictStep(samples, count);
    // Quarter-size evidence can be refined at a safely predicted half-size below.
    if (prediction?.predictedMs > limitMs && prediction.from.count < count / 4) refuse(prediction);
    // A compatible quarter-size (or larger) measurement is enough; smaller ones only gate probes.
    if (!prediction || prediction.from.count < count / 4) {
        const quarter = Math.max(64, Math.floor(count / 4 / 16) * 16);
        const probes = [];
        for (let size = Math.min(4096, quarter); size < quarter; size *= 4) probes.push(size);
        probes.push(quarter);
        for (const size of probes) {
            const gate = predictStep(samples, size);
            if (gate?.predictedMs > limitMs) break;
            const ms = checkStep(await measure(size, stopMs), stopMs);
            if (ms === 0) throw new Error('film: calibration returned zero step time.');
            const sample = { count: size, ms };
            samples.push(sample); cached = false; save(samples);
            log(`guard: ${size.toLocaleString('en-US')} agents step in ${ms.toFixed(1)} ms on this GPU (maximum queue-completion wall time)`);
            prediction = predictStep(samples, count);
            if (ms > limitMs) refuse(prediction);
        }
        prediction = predictStep(samples, count);
    }
    if (prediction?.predictedMs > limitMs && prediction.from.count >= count / 4 && prediction.from.count < count / 2) {
        const half = Math.floor(count / 2 / 16) * 16;
        const gate = predictStep(samples, half);
        if (gate && gate.predictedMs <= limitMs) {
            const ms = checkStep(await measure(half, stopMs), stopMs);
            if (ms === 0) throw new Error('film: calibration returned zero step time.');
            samples.push({ count: half, ms }); cached = false; save(samples);
            log(`guard: ${half.toLocaleString('en-US')} agents step in ${ms.toFixed(1)} ms on this GPU (safe half-size refinement)`);
            prediction = predictStep(samples, count);
        }
    }
    if (!prediction || prediction.predictedMs > limitMs) refuse(prediction ?? { predictedMs: Infinity, from: { ms: NaN, count: 0 } });
    log(`guard: ~${prediction.predictedMs.toFixed(0)} ms per step predicted from ${prediction.from.count.toLocaleString('en-US')} agents${cached ? ' (cached)' : ''}`);
    return { ...prediction, cached, timing: 'maximum queue-completion wall ms' };
}