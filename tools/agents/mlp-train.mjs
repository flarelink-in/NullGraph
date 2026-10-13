// Step 7, part 2 (docs/agents-cloning-design.md §7): a small dense-network trainer in plain JS.
//
// Mean-squared error, Adam, mini-batches, and a step size that decays over the passes. Seeded
// through rand01 and run on one thread in a fixed order, so the same rows, shape and seed give the
// same weights, byte for byte, on one Node version (U11).
//
// It trains in f64 with MlpBrain's own activations (tanh's input clamped to ±15, the stable
// sigmoid) and returns MlpNetwork layers in f32. Errors it reports are measured with the library's
// mlpForward on those f32 weights: what is measured is what runs.
//
// Rows are read in place. A set is { data, stride, x, y, inputs, outputs, rows }: row r's input is
// data[rows[r] * stride + x …], and its target data[rows[r] * stride + y …].
// No Node imports.
import * as A from '../../dist/agents.js';

/** Phase 7.7 measured these to meet the 0.05 bar for both Herd & Wolves species (design, 7.7 results). */
export const TRAIN_DEFAULTS = Object.freeze({
    epochs: 200, batch: 256, rate: 1e-2, finalRate: 1e-4, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, seed: 1,
});
const INIT_STREAM = 0x696e6974, SHUFFLE_STREAM = 0x73687566;

/** Where each layer's weights and biases sit in one flat parameter array (MlpNetwork order). */
function layout(shape) {
    let size = 0;
    const layers = shape.map(({ input, output, activation }) => {
        const layer = { input, output, activation, w: size, b: size + input * output };
        size = layer.b + output;
        return layer;
    });
    return { layers, size };
}

function activate(kind, z) {
    if (kind === 'relu') return z > 0 ? z : 0;
    if (kind === 'tanh') return Math.tanh(z < -15 ? -15 : z > 15 ? 15 : z);
    if (kind === 'sigmoid') {
        const e = Math.exp(-Math.abs(z));
        return z >= 0 ? 1 / (1 + e) : e / (1 + e);
    }
    return z;
}

/** d activation / dz, from the input z and the output a. */
function slope(kind, z, a) {
    if (kind === 'relu') return z > 0 ? 1 : 0;
    if (kind === 'tanh') return z <= -15 || z >= 15 ? 0 : 1 - a * a;
    if (kind === 'sigmoid') return a * (1 - a);
    return 1;
}

/** Buffers for one batch: a[0] the inputs; for layer l, z[l] before and a[l + 1] after its activation. */
function workspace(shape, batch) {
    return {
        lay: layout(shape), batch,
        a: [new Float64Array(batch * shape[0].input), ...shape.map(l => new Float64Array(batch * l.output))],
        z: shape.map(l => new Float64Array(batch * l.output)),
        d: shape.map(l => new Float64Array(batch * l.output)),   // dLoss/dz
        target: new Float64Array(batch * shape[shape.length - 1].output),
    };
}

function gather(ws, set, order, from, n) {
    const { data, stride, x, y, inputs, outputs } = set, input = ws.a[0];
    for (let b = 0; b < n; b++) {
        const at = order[from + b] * stride;
        for (let k = 0; k < inputs; k++) input[b * inputs + k] = data[at + x + k];
        for (let j = 0; j < outputs; j++) ws.target[b * outputs + j] = data[at + y + j];
    }
}

function forward(ws, p, n) {
    const { layers } = ws.lay;
    for (let l = 0; l < layers.length; l++) {
        const L = layers[l], x = ws.a[l], z = ws.z[l], a = ws.a[l + 1];
        for (let b = 0; b < n; b++) {
            const xo = b * L.input, yo = b * L.output;
            for (let r = 0; r < L.output; r++) {
                const wo = L.w + r * L.input;
                let s = p[L.b + r];
                for (let k = 0; k < L.input; k++) s += p[wo + k] * x[xo + k];
                z[yo + r] = s;
                a[yo + r] = activate(L.activation, s);
            }
        }
    }
}

/** Mean-squared error over n rows; accumulates its gradient into g (which the caller zeroes). */
function backward(ws, p, n, g) {
    const { layers } = ws.lay, last = layers.length - 1, outputs = layers[last].output, scale = 2 / (n * outputs);
    const out = ws.a[last + 1], zl = ws.z[last], dl = ws.d[last];
    let loss = 0;
    for (let j = 0; j < n * outputs; j++) {
        const e = out[j] - ws.target[j];
        loss += e * e;
        dl[j] = scale * e * slope(layers[last].activation, zl[j], out[j]);
    }
    for (let l = last; l >= 0; l--) {
        const L = layers[l], x = ws.a[l], d = ws.d[l];
        for (let b = 0; b < n; b++) {
            const xo = b * L.input, yo = b * L.output;
            for (let r = 0; r < L.output; r++) {
                const dr = d[yo + r], wo = L.w + r * L.input;
                g[L.b + r] += dr;
                for (let k = 0; k < L.input; k++) g[wo + k] += dr * x[xo + k];
            }
        }
        if (l === 0) break;
        const P = layers[l - 1], dp = ws.d[l - 1], zp = ws.z[l - 1], ap = ws.a[l];
        for (let b = 0; b < n; b++) {
            const po = b * P.output, yo = b * L.output;
            for (let k = 0; k < L.input; k++) {
                let s = 0;
                for (let r = 0; r < L.output; r++) s += d[yo + r] * p[L.w + r * L.input + k];
                dp[po + k] = s * slope(P.activation, zp[po + k], ap[po + k]);
            }
        }
    }
    return loss / (n * outputs);
}

/** The shape of a network: its layers' input, output and activation. */
export const shapeOf = network => network.map(({ input, output, activation }) => ({ input, output, activation }));

/**
 * A network of `shape` from `seed`: weights uniform in ±√(6 / (in + out)), or ±√(6 / in) before a
 * relu; biases zero. Refused with MlpBrain's own messages if MlpBrain could not run it.
 */
export function initNetwork(shape, seed = TRAIN_DEFAULTS.seed) {
    const network = shape.map((l, i) => {
        const limit = Math.sqrt(6 / (l.activation === 'relu' ? l.input : l.input + l.output));
        return { input: l.input, output: l.output, activation: l.activation,
            weights: Float32Array.from({ length: l.input * l.output }, (_, k) => (A.rand01(k, i, INIT_STREAM + seed) * 2 - 1) * limit),
            biases: new Float32Array(l.output) };
    });
    A.mlpLayout(network);
    return network;
}

/** A network's weights and biases in one f64 array, layer by layer: weights, then biases. */
export function flatParams(network) {
    const { layers, size } = layout(network), p = new Float64Array(size);
    network.forEach((layer, i) => {
        p.set(layer.weights, layers[i].w);
        p.set(layer.biases, layers[i].b);
    });
    return p;
}

/** The f32 network those parameters describe. */
export function networkOf(shape, params) {
    return layout(shape).layers.map(l => ({ input: l.input, output: l.output, activation: l.activation,
        weights: Float32Array.from(params.subarray(l.w, l.b)), biases: Float32Array.from(params.subarray(l.b, l.b + l.output)) }));
}

/** Mean-squared error over the set's rows, and its gradient in flatParams order. f64; for U11. */
export function lossAndGradient(shape, params, set) {
    const n = set.rows.length, ws = workspace(shape, n), gradient = new Float64Array(params.length);
    gather(ws, set, set.rows, 0, n);
    forward(ws, params, n);
    return { loss: backward(ws, params, n, gradient), gradient };
}

/** Mean absolute error per output, in f64 at full parameters: what picks the best pass. */
function scoreF64(ws, p, set) {
    const outputs = set.outputs, sums = new Float64Array(outputs), last = ws.a.length - 1;
    for (let from = 0; from < set.rows.length; from += ws.batch) {
        const n = Math.min(ws.batch, set.rows.length - from);
        gather(ws, set, set.rows, from, n);
        forward(ws, p, n);
        for (let j = 0; j < n * outputs; j++) sums[j % outputs] += Math.abs(ws.a[last][j] - ws.target[j]);
    }
    return Array.from(sums, s => s / set.rows.length);
}

/**
 * Trains a network of `shape` on `set`. With `options.heldOut` (a set), every pass is scored on it
 * and the pass with the lowest mean absolute error is returned; otherwise the last pass is.
 * Resolves nothing asynchronously: one thread, a fixed order. `options.onEpoch(entry)` sees each pass.
 * Returns { network, last, best, history: [{ epoch, rate, loss, heldOut }] }, networks in f32.
 */
export function trainNetwork(shape, set, options = {}) {
    const o = { ...TRAIN_DEFAULTS, ...options };
    if (!(set.rows.length > 0)) throw new Error('train: the set has no rows.');
    if (shape[0].input !== set.inputs || shape[shape.length - 1].output !== set.outputs) {
        throw new Error(`train: the shape maps ${shape[0].input} → ${shape[shape.length - 1].output}; the set has ${set.inputs} → ${set.outputs}.`);
    }
    const params = flatParams(initNetwork(shape, o.seed)), ws = workspace(shape, o.batch);
    const gradient = new Float64Array(params.length), m = new Float64Array(params.length), v = new Float64Array(params.length);
    const order = Uint32Array.from(set.rows), history = [];
    let step = 0, best = -1, bestScore = Infinity, bestParams = params.slice();
    for (let epoch = 0; epoch < o.epochs; epoch++) {
        const rate = o.rate * (o.finalRate / o.rate) ** (epoch / Math.max(1, o.epochs - 1));
        for (let i = order.length - 1; i > 0; i--) {
            const j = Math.floor(A.rand01(i, epoch, SHUFFLE_STREAM + o.seed) * (i + 1)), t = order[i];
            order[i] = order[j];
            order[j] = t;
        }
        let loss = 0;
        for (let from = 0; from < order.length; from += o.batch) {
            const n = Math.min(o.batch, order.length - from);
            gather(ws, set, order, from, n);
            forward(ws, params, n);
            gradient.fill(0);
            loss += backward(ws, params, n, gradient) * n;
            step++;
            const c1 = 1 - o.beta1 ** step, c2 = 1 - o.beta2 ** step;
            for (let k = 0; k < params.length; k++) {
                const gk = gradient[k];
                m[k] = o.beta1 * m[k] + (1 - o.beta1) * gk;
                v[k] = o.beta2 * v[k] + (1 - o.beta2) * gk * gk;
                params[k] -= rate * (m[k] / c1) / (Math.sqrt(v[k] / c2) + o.epsilon);
            }
        }
        const entry = { epoch, rate, loss: loss / order.length, heldOut: o.heldOut ? scoreF64(ws, params, o.heldOut) : null };
        history.push(entry);
        o.onEpoch?.(entry);
        const score = entry.heldOut ? entry.heldOut.reduce((s, e) => s + e, 0) : -epoch;
        if (score < bestScore) {
            bestScore = score;
            best = epoch;
            bestParams = params.slice();
        }
    }
    return { network: networkOf(shape, bestParams), last: networkOf(shape, params), best, history };
}

/**
 * Mean absolute error per output of an f32 network on a set, computed with the library's
 * mlpForward: the number a trained file reports. `groupOf(row)`, if given, also splits the error
 * by a group, such as the teacher's reason code. Returns { rows, mae, groups: { <group>: { rows, mae } } },
 * and with `keep`, `errors`: each output's absolute errors, row by row, in the set's order.
 */
export function meanAbsoluteError(network, set, groupOf, { keep = false } = {}) {
    const outputs = set.outputs, out = new Float32Array(outputs), scratch = new Float32Array(2 * A.mlpLayout(network).scratchWidth);
    const total = new Float64Array(outputs), groups = new Map();
    const errors = keep ? Array.from({ length: outputs }, () => new Float32Array(set.rows.length)) : null;
    let index = 0;
    for (const row of set.rows) {
        const at = row * set.stride;
        A.mlpForward(network, set.data.subarray(at + set.x, at + set.x + set.inputs), out, scratch);
        let group;
        if (groupOf) {
            const key = groupOf(row);
            group = groups.get(key) ?? { rows: 0, sums: new Float64Array(outputs) };
            groups.set(key, group);
            group.rows++;
        }
        for (let j = 0; j < outputs; j++) {
            const e = Math.abs(out[j] - set.data[at + set.y + j]);
            total[j] += e;
            if (group) group.sums[j] += e;
            if (errors) errors[j][index] = e;
        }
        index++;
    }
    const n = set.rows.length;
    return { rows: n, mae: Array.from(total, s => s / n), ...(errors ? { errors } : {}),
        groups: Object.fromEntries([...groups].sort(([a], [b]) => a - b).map(([key, g]) => [key, { rows: g.rows, mae: Array.from(g.sums, s => s / g.rows) }])) };
}

/** The value at fraction q (0 to 1) of an array once sorted: the median is quantile(values, 0.5). */
export function quantile(values, q) {
    const sorted = Float64Array.from(values).sort();
    return sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] : NaN;
}
