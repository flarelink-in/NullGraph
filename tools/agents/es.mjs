// Phase 14.2, design §7. CPU-only ES math; no device, filesystem, scenario or spike dependency.
// Packing and Adam constants are shared with the existing cloning trainer. There are deliberately
// no research defaults for sigma, rate or decay. Inputs/masters/moments/reductions stay float64.
import { TRAIN_DEFAULTS, flatParams, networkOf } from './mlp-train.mjs';

export const NOISE_VERSION = 'es-tuple-box-muller-v1';
// Explicit per-vector allocation/work bound (128 MiB f64), not an accepted population preset.
export const MAX_PARAMETERS = 2 ** 24;
const U32_MAX = 0xffffffff;
const fail = message => { throw new Error(`coevolve: ${message}`); };
function uint(value, name) {
    if (!Number.isInteger(value) || value < 0 || value > U32_MAX) fail(`${name} must be u32`);
}
function positive(value, name) {
    if (!Number.isFinite(value) || value <= 0) fail(`${name} must be finite and > 0`);
}
function vector(values, name, length) {
    if ((!Array.isArray(values) && !(values instanceof Float64Array) && !(values instanceof Float32Array)) ||
        values.length < 1 || values.length > MAX_PARAMETERS || (length !== undefined && values.length !== length))
        fail(`${name} has invalid length (1..${MAX_PARAMETERS}${length === undefined ? '' : `; expected ${length}`})`);
    for (let j = 0; j < values.length; j++) if (!Number.isFinite(values[j])) fail(`${name}[${j}] must be finite`);
}

/** Ordered u32 tuple mixer. Each field is mixed independently; no arithmetic packing or seed doubling.
 * Noise v1 tuple = [0x45530101, stream (1 radius / 2 angle), seed, generation, direction, even j].
 * Hashing is deterministic, not collision-free. Episode seed collision handling belongs to §8.
 */
export function tupleHash(...fields) {
    let h = 0x85ebca6b;
    for (const field of fields) {
        uint(field, 'hash field');
        h = Math.imul(h ^ field, 0x9e3779b1) >>> 0;
        h ^= h >>> 16;
        h = Math.imul(h, 0x7feb352d) >>> 0;
        h ^= h >>> 15;
    }
    return h >>> 0;
}

/** Standard Gaussian direction in fixed j order. Uniforms are strictly inside (0,1).
 * Floating fixtures pin Node's runtime; transcendental equality across engines is not promised.
 */
export function noise(seed, generation, direction, n) {
    uint(seed, 'seed'); uint(generation, 'generation'); uint(direction, 'direction');
    if (!Number.isInteger(n) || n < 1 || n > MAX_PARAMETERS) fail(`noise length must be 1..${MAX_PARAMETERS}`);
    const result = new Float64Array(n);
    for (let j = 0; j < n; j += 2) {
        const u = (tupleHash(0x45530101, 1, seed, generation, direction, j) + 0.5) / 4294967296;
        const v = (tupleHash(0x45530101, 2, seed, generation, direction, j) + 0.5) / 4294967296;
        const radius = Math.sqrt(-2 * Math.log(u)), angle = 2 * Math.PI * v;
        result[j] = radius * Math.cos(angle);
        if (j + 1 < n) result[j + 1] = radius * Math.sin(angle);
    }
    return result;
}

/** Ascending average ranks, normalized to [-.5,.5]. Equal values never gain index-based utility. */
export function centeredRanks(scores) {
    vector(scores, 'scores');
    if (scores.length < 2) fail('scores require at least two candidates');
    const order = Array.from(scores, (_, i) => i).sort((a, b) => scores[a] - scores[b] || a - b);
    const utilities = new Float64Array(scores.length);
    for (let a = 0; a < order.length;) {
        let b = a + 1;
        while (b < order.length && scores[order[b]] === scores[order[a]]) b++;
        const utility = ((a + b - 1) / 2) / (order.length - 1) - 0.5;
        for (let i = a; i < b; i++) utilities[order[i]] = utility;
        a = b;
    }
    return utilities;
}

/** Utilities are interleaved plus/minus by direction. g = sum (u+ - u-) eps / (2 K sigma). */
export function gradient(directions, utilities, sigma) {
    positive(sigma, 'sigma');
    if (!Array.isArray(directions) || !directions.length || directions.length > MAX_PARAMETERS / 2)
        fail('directions require a nonempty bounded array');
    vector(directions[0], 'direction 0');
    const n = directions[0].length;
    for (let k = 1; k < directions.length; k++) vector(directions[k], `direction ${k}`, n);
    vector(utilities, 'utilities', 2 * directions.length);
    const denominator = 2 * directions.length * sigma;
    if (!Number.isFinite(denominator) || denominator === 0) fail('gradient normalization overflow/underflow');
    const result = new Float64Array(n);
    for (let k = 0; k < directions.length; k++) {
        const weight = (utilities[2 * k] - utilities[2 * k + 1]) / denominator;
        for (let j = 0; j < n; j++) result[j] += weight * directions[k][j];
    }
    vector(result, 'gradient');
    return result;
}

/** Caller owns one independent state per species; copies master, never aliases caller storage. */
export function createAdamState(master) {
    vector(master, 'master');
    return { x: Float64Array.from(master), m: new Float64Array(master.length),
        v: new Float64Array(master.length), steps: 0 };
}

/** Coupled L2 ascent, NOT AdamW. Transactional: failed arithmetic leaves all state untouched.
 * A zero fitness gradient can still move through old momentum or decay; no implicit neutral skip.
 */
export function adamAscent(state, fitnessGradient, { rate, decay } = {}) {
    positive(rate, 'rate');
    if (!Number.isFinite(decay) || decay < 0) fail('decay must be finite and >= 0');
    if (!state || ![state.x, state.m, state.v].every(v => v instanceof Float64Array))
        fail('Adam state requires float64 x/m/v');
    vector(state.x, 'master');
    vector(state.m, 'first moment', state.x.length); vector(state.v, 'second moment', state.x.length);
    if (new Set([state.x.buffer, state.m.buffer, state.v.buffer]).size !== 3)
        fail('Adam x/m/v storage must be independent');
    if (state.v.some(v => v < 0)) fail('second moment must be nonnegative');
    uint(state.steps, 'Adam steps');
    if (state.steps === U32_MAX) fail('Adam steps overflow');
    vector(fitnessGradient, 'fitness gradient', state.x.length);
    const { beta1, beta2, epsilon } = TRAIN_DEFAULTS, steps = state.steps + 1;
    const x = new Float64Array(state.x.length), m = new Float64Array(x.length), v = new Float64Array(x.length);
    const c1 = 1 - beta1 ** steps, c2 = 1 - beta2 ** steps;
    for (let j = 0; j < x.length; j++) {
        const g = fitnessGradient[j] - decay * state.x[j];
        m[j] = beta1 * state.m[j] + (1 - beta1) * g;
        v[j] = beta2 * state.v[j] + (1 - beta2) * g * g;
        x[j] = state.x[j] + rate * (m[j] / c1) / (Math.sqrt(v[j] / c2) + epsilon);
    }
    vector(x, 'updated master'); vector(m, 'updated first moment'); vector(v, 'updated second moment');
    state.x.set(x); state.m.set(m); state.v.set(v); state.steps = steps;
    return state;
}

function shapeSize(shape) {
    if (!Array.isArray(shape) || shape.length < 1 || shape.length > 4) fail('network shape requires 1..4 affine layers');
    let size = 0;
    for (let i = 0; i < shape.length; i++) {
        const layer = shape[i];
        if (!layer || !Number.isInteger(layer.input) || layer.input < 1 || layer.input > (i === 0 ? 128 : 64) ||
            !Number.isInteger(layer.output) || layer.output < 1 || layer.output > (i === shape.length - 1 ? 16 : 64) ||
            !['identity', 'relu', 'tanh', 'sigmoid'].includes(layer.activation) ||
            (i > 0 && layer.input !== shape[i - 1].output)) fail(`invalid network layer ${i}`);
        size += layer.input * layer.output + layer.output;
        if (!Number.isSafeInteger(size) || size > MAX_PARAMETERS) fail('network parameter budget exceeded');
    }
    return size;
}

/** Existing layer order: weights then biases, per layer. Validate before existing packing helpers. */
export function masterParams(network) {
    shapeSize(network);
    for (let i = 0; i < network.length; i++) {
        vector(network[i].weights, `layer ${i} weights`, network[i].input * network[i].output);
        vector(network[i].biases, `layer ${i} biases`, network[i].output);
    }
    return flatParams(network);
}

/** Materialize only at the f32 boundary. Diagnostics count bit-equal plus/minus parameters,
 * including biases, and report f64 perturbation/f32 displacement norms. Masters are untouched.
 */
export function candidateNetworks(shape, master, direction, sigma) {
    const size = shapeSize(shape);
    vector(master, 'master', size); vector(direction, 'direction', size); positive(sigma, 'sigma');
    const plus = new Float64Array(size), minus = new Float64Array(size);
    let perturbationNorm = 0, plusNorm = 0, minusNorm = 0, collisions = 0;
    for (let j = 0; j < size; j++) {
        const delta = sigma * direction[j], base = Math.fround(master[j]);
        plus[j] = master[j] + delta; minus[j] = master[j] - delta;
        const p = Math.fround(plus[j]), m = Math.fround(minus[j]);
        if (![delta, plus[j], minus[j], base, p, m].every(Number.isFinite)) fail(`nonfinite candidate parameter ${j}`);
        if (Object.is(p, m)) collisions++;
        perturbationNorm = Math.hypot(perturbationNorm, delta);
        plusNorm = Math.hypot(plusNorm, p - base); minusNorm = Math.hypot(minusNorm, m - base);
    }
    if (![perturbationNorm, plusNorm, minusNorm].every(Number.isFinite)) fail('candidate norm overflow');
    return { plus: networkOf(shape, plus), minus: networkOf(shape, minus),
        diagnostics: { parameters: size, collisions, perturbationNorm, plusNorm, minusNorm } };
}