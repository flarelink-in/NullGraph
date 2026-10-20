// Phase 14.4, design §10. Seed-level statistics for arena evaluation and cross-play: deterministic
// percentile bootstrap of means and of paired differences. Pure; no node: imports, so the site's
// HUD (step 16) can recompute the same intervals from exported raw catches.
import { tupleHash } from './es.mjs';

export const BOOTSTRAP_VERSION = 'coevo-bootstrap-v1';
// Design §10: 10,000 resamples, 95% percentile bounds, a fixed analysis seed of its own.
export const BOOTSTRAP = Object.freeze({ resamples: 10000, level: 0.95 });
// Tuple tag for resampling draws: no tuple of the noise (0x45530101) or schedule shape matches it.
const RESAMPLE_TAG = 0x45530601;
const MAX_DRAWS = 2 ** 26;
const fail = message => { throw new Error(`coevolve: ${message}`); };

function samples(values, name) {
    if (!Array.isArray(values) && !ArrayBuffer.isView(values)) fail(`${name} must be an array of numbers`);
    if (values.length < 1) fail(`${name} needs at least one episode`);
    for (let j = 0; j < values.length; j++) if (typeof values[j] !== 'number' || !Number.isFinite(values[j]))
        fail(`${name}[${j}] must be a finite number`);
    return values;
}

function settings({ seed, resamples = BOOTSTRAP.resamples, level = BOOTSTRAP.level } = {}) {
    if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) fail('bootstrap seed must be an explicit u32');
    if (!Number.isSafeInteger(resamples) || resamples < 1) fail('resamples must be a positive integer');
    if (!(level > 0 && level < 1)) fail('level must be in (0, 1)');
    return { seed, resamples, level };
}

/** Draw r's slot j picks episode floor(u32 · n / 2^32), the same for every statistic of a seed. */
export function resampleIndices({ seed, n, resamples = BOOTSTRAP.resamples }) {
    settings({ seed, resamples });
    if (!Number.isSafeInteger(n) || n < 1) fail('n must be a positive integer');
    if (resamples * n > MAX_DRAWS) fail(`${resamples} resamples of ${n} episodes exceed ${MAX_DRAWS} draws`);
    const indices = new Uint32Array(resamples * n);
    for (let r = 0; r < resamples; r++) for (let j = 0; j < n; j++)
        indices[r * n + j] = Math.floor(tupleHash(RESAMPLE_TAG, seed, n, r, j) * n / 4294967296);
    return indices;
}

/** The rank-th smallest and rank-th largest of the sorted resampled means: nearest rank. */
function bounds(means, level) {
    means.sort();
    const rank = Math.max(1, Math.ceil((1 - level) / 2 * means.length - 1e-9));
    return { low: means[rank - 1], high: means[means.length - rank] };
}

const mean = values => {
    let sum = 0;
    for (let j = 0; j < values.length; j++) sum += values[j];
    return sum / values.length;
};

function interval(values, options, indices) {
    const { seed, resamples, level } = settings(options), n = values.length;
    indices ??= resampleIndices({ seed, n, resamples });
    const means = new Float64Array(resamples);
    for (let r = 0; r < resamples; r++) {
        let sum = 0;
        for (let j = 0; j < n; j++) sum += values[indices[r * n + j]];
        means[r] = sum / n;
    }
    return { n, mean: mean(values), ...bounds(means, level), level, resamples, seed, version: BOOTSTRAP_VERSION };
}

/** Percentile interval of one arm's mean over its episodes. */
export function meanInterval(values, options) {
    return interval(samples(values, 'values'), options);
}

/** Maximum drop from an earlier checkpoint's mean signed anchor score to a later one's.
 * Checkpoint order and anchors must be frozen before final testing. Resample whole episodes
 * jointly across the curve and recompute the maximum, not an interval of a selected cell.
 */
export function forgettingInterval(curve, options) {
    if (!Array.isArray(curve) || !curve.length) fail('forgetting needs an ordered checkpoint curve');
    curve.forEach(row => samples(row, 'checkpoint scores'));
    const n = curve[0].length;
    if (curve.some(row => row.length !== n)) fail('forgetting checkpoints must share episodes');
    const { seed, resamples, level } = settings(options), indices = resampleIndices({ seed, n, resamples });
    const drop = means => {
        let peak = means[0], worst = 0;
        for (const value of means) { worst = Math.max(worst, peak - value); peak = Math.max(peak, value); }
        return worst;
    };
    const draws = new Float64Array(resamples);
    for (let r = 0; r < resamples; r++) draws[r] = drop(curve.map(row => {
        let total = 0;
        for (let j = 0; j < n; j++) total += row[indices[r * n + j]];
        return total / n;
    }));
    return { value: drop(curve.map(mean)), ...bounds(draws, level), n, seed, resamples, level,
        version: 'coevo-forgetting-max-drawdown-v1' };
}

/**
 * Paired comparison over the same episodes (same held-out seeds and tiles, fresh state): the
 * interval of mean(a − b) resamples whole episodes, and both marginals use the same draws.
 */
export function pairedBootstrap(a, b, options) {
    samples(a, 'a'); samples(b, 'b');
    if (a.length !== b.length) fail(`paired samples differ in length: ${a.length} and ${b.length}`);
    const { seed, resamples } = settings(options);
    const indices = resampleIndices({ seed, n: a.length, resamples });
    const difference = Float64Array.from(a, (x, j) => x - b[j]);
    return { difference: interval(difference, options, indices),
        a: interval(a, options, indices), b: interval(b, options, indices) };
}

const sense = better => {
    if (better !== 'higher' && better !== 'lower') fail('better must be "higher" or "lower"');
    return better === 'higher' ? 1 : -1;
};

/** Design §10 primary superiority: the paired interval lies strictly on the favorable side of zero. */
export function superior(difference, { better }) {
    return sense(better) > 0 ? difference.low > 0 : difference.high < 0;
}

/** H2: the paired interval excludes a loss of `margin` or more; the margin is preregistered. */
export function noninferior(difference, { better, margin }) {
    if (!(Number.isFinite(margin) && margin > 0)) fail('noninferiority needs an explicit positive margin');
    return sense(better) > 0 ? difference.low > -margin : difference.high < margin;
}
