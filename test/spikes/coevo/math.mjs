// S7 replacement protocol 2. Pure, deterministic spike math; no production optimizer claim.
import { TRAIN_DEFAULTS } from '../../../tools/agents/mlp-train.mjs';
export const mean = v => v.reduce((s, x) => s + x, 0) / v.length;
export const sd = v => Math.sqrt(v.reduce((s, x) => s + (x - mean(v)) ** 2, 0) / Math.max(1, v.length - 1));
export function correlation(a, b) {
    const ma = mean(a), mb = mean(b);
    const denom = Math.sqrt(a.reduce((s, x) => s + (x - ma) ** 2, 0) * b.reduce((s, x) => s + (x - mb) ** 2, 0));
    return denom ? a.reduce((s, x, i) => s + (x - ma) * (b[i] - mb), 0) / denom : null;
}
export function scoreQuality(directions, thresholds) {
    const half = Math.floor(directions[0].differences.length / 2);
    const first = directions.map(d => mean(d.differences.slice(0, half)));
    const last = directions.map(d => mean(d.differences.slice(half)));
    const splitHalfRankCorrelation = correlation([...ranks(first)], [...ranks(last)]);
    const residual = Math.sqrt(mean(directions.map(d => d.sd ** 2)));
    const directionalSD = sd(directions.map(d => d.mean));
    const signalToResidual = residual ? directionalSD / residual : null;
    const nonzeroIntervals = directions.filter(d => d.interval.low > 0 || d.interval.high < 0).length;
    return { splitHalfRankCorrelation, signalToResidual, nonzeroIntervals,
        accepted: splitHalfRankCorrelation !== null && signalToResidual !== null &&
            splitHalfRankCorrelation >= thresholds.splitHalfRankCorrelation &&
            signalToResidual >= thresholds.signalToResidual && nonzeroIntervals >= thresholds.nonzeroIntervals };
}
export function hash(...fields) {
    let h = 0x85ebca6b;
    for (const x of fields) {
        if (!Number.isInteger(x) || x < 0 || x > 0xffffffff) throw new Error('S7: hash field outside u32');
        h = Math.imul(h ^ x, 0x9e3779b1) >>> 0;
        h ^= h >>> 16; h = Math.imul(h, 0x7feb352d) >>> 0; h ^= h >>> 15;
    }
    return h >>> 0;
}
export function noise(n, seed, t, k) {
    const e = new Float64Array(n);
    for (let j = 0; j < n; j += 2) {
        const u = (hash(1, seed, t, k, j) + 0.5) / 4294967296;
        const v = (hash(2, seed, t, k, j) + 0.5) / 4294967296;
        const r = Math.sqrt(-2 * Math.log(u)), a = 2 * Math.PI * v;
        e[j] = r * Math.cos(a); if (j + 1 < n) e[j + 1] = r * Math.sin(a);
    }
    return e;
}
export function ranks(scores) {
    if (scores.length < 2 || scores.some(x => !Number.isFinite(x))) throw new Error('S7: invalid scores');
    const order = Array.from(scores, (_, i) => i).sort((a, b) => scores[a] - scores[b] || a - b);
    const u = new Float64Array(scores.length);
    for (let a = 0; a < order.length;) {
        let b = a + 1;
        while (b < order.length && scores[order[b]] === scores[order[a]]) b++;
        const value = ((a + b - 1) / 2) / (order.length - 1) - 0.5;
        for (let i = a; i < b; i++) u[order[i]] = value;
        a = b;
    }
    return u;
}
export function gradient(eps, u, sigma) {
    const g = new Float64Array(eps[0].length);
    for (let k = 0; k < eps.length; k++) for (let j = 0; j < g.length; j++)
        g[j] += (u[2 * k] - u[2 * k + 1]) * eps[k][j] / (2 * eps.length * sigma);
    return g;
}
export function adam(state, g, rate = 0.01, decay = 0.005) {
    const { beta1, beta2, epsilon } = TRAIN_DEFAULTS;
    state.steps++;
    for (let j = 0; j < g.length; j++) {
        const d = g[j] - decay * state.x[j];
        state.m[j] = beta1 * state.m[j] + (1 - beta1) * d;
        state.v[j] = beta2 * state.v[j] + (1 - beta2) * d * d;
        state.x[j] += rate * (state.m[j] / (1 - beta1 ** state.steps)) /
            (Math.sqrt(state.v[j] / (1 - beta2 ** state.steps)) + epsilon);
        if (!Number.isFinite(state.x[j])) throw new Error('S7: nonfinite update');
    }
}
export function permutation(n, seed, t) {
    const p = Array.from({ length: n }, (_, i) => i);
    for (let j = n - 1; j > 0; j--) { const k = hash(3, seed, t, j) % (j + 1); [p[j], p[k]] = [p[k], p[j]]; }
    return p;
}
export function pairedInterval(d, seed = 76123, samples = 10000) {
    const draws = [];
    for (let b = 0; b < samples; b++) {
        let sum = 0;
        for (let j = 0; j < d.length; j++) sum += d[hash(4, seed, b, j) % d.length];
        draws.push(sum / d.length);
    }
    draws.sort((a, b) => a - b);
    return { mean: mean(d), low: draws[Math.floor(samples * 0.025)], high: draws[Math.floor(samples * 0.975)] };
}