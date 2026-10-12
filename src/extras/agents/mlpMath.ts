/** Logical dense-network parameters. Weights are row-major [output, input], never GPU padding. */
export type MlpActivation = 'identity' | 'relu' | 'tanh' | 'sigmoid';
export interface MlpLayer {
    readonly input: number;
    readonly output: number;
    readonly activation: MlpActivation;
    readonly weights: Float32Array;
    readonly biases: Float32Array;
}
export type MlpNetwork = readonly MlpLayer[];

/** Float offsets into the vec4-packed GPU buffer; widths include zero-padded lanes. */
export interface MlpLayerLayout {
    readonly inputStride: number;
    readonly outputStride: number;
    readonly weightOffset: number;
    readonly biasOffset: number;
}
export interface MlpLayout {
    readonly layers: readonly MlpLayerLayout[];
    readonly floats: number;
    readonly scratchWidth: number;
}

function fail(message: string): never { throw new Error(`MlpBrain: ${message}`); }

/** Validate without allocating: repeated CPU calls with caller scratch allocate no arrays. */
function validate(network: MlpNetwork): number {
    if (!Array.isArray(network) || network.length < 1 || network.length > 4) fail('expected 1–4 affine layers.');
    let width = 0;
    for (let i = 0; i < network.length; i++) {
        const layer = network[i];
        if (!layer) fail(`layer ${i} is missing.`);
        if (!Number.isInteger(layer.input) || layer.input < 1 || layer.input > (i === 0 ? 128 : 64)) {
            fail(`layer ${i} input must be 1–${i === 0 ? 128 : 64}.`);
        }
        const limit = i === network.length - 1 ? 16 : 64;
        if (!Number.isInteger(layer.output) || layer.output < 1 || layer.output > limit) {
            fail(`layer ${i} output must be 1–${limit}.`);
        }
        if (i > 0 && layer.input !== network[i - 1].output) fail(`layer ${i} input does not match the previous output.`);
        if (layer.activation !== 'identity' && layer.activation !== 'relu' &&
            layer.activation !== 'tanh' && layer.activation !== 'sigmoid') fail(`layer ${i} has unsupported activation.`);
        if (!(layer.weights instanceof Float32Array) || layer.weights.length !== layer.input * layer.output) {
            fail(`layer ${i} weights need ${layer.input * layer.output} float32 values.`);
        }
        if (!(layer.biases instanceof Float32Array) || layer.biases.length !== layer.output) {
            fail(`layer ${i} biases need ${layer.output} float32 values.`);
        }
        for (let k = 0; k < layer.weights.length; k++) {
            if (!Number.isFinite(layer.weights[k])) fail(`layer ${i} weights[${k}] must be finite.`);
        }
        for (let k = 0; k < layer.biases.length; k++) {
            if (!Number.isFinite(layer.biases[k])) fail(`layer ${i} biases[${k}] must be finite.`);
        }
        width = Math.max(width, layer.input, layer.output);
    }
    return width;
}

export function mlpLayout(network: MlpNetwork): MlpLayout {
    const scratchWidth = validate(network);
    let floats = 0;
    const layers = network.map(layer => {
        const inputStride = Math.ceil(layer.input / 4) * 4;
        const outputStride = Math.ceil(layer.output / 4) * 4;
        const weightOffset = floats;
        const biasOffset = weightOffset + inputStride * outputStride;
        floats = biasOffset + outputStride;
        return { inputStride, outputStride, weightOffset, biasOffset };
    });
    return { layers, floats, scratchWidth };
}

function overlaps(a: Float32Array, b: Float32Array): boolean {
    return a.buffer === b.buffer && a.byteOffset < b.byteOffset + b.byteLength &&
        b.byteOffset < a.byteOffset + a.byteLength;
}
function assertSeparate(network: MlpNetwork, a: Float32Array): void {
    for (const layer of network) {
        if (overlaps(a, layer.weights) || overlaps(a, layer.biases)) fail('writable views overlap model parameters.');
    }
}

/** Zero every padded lane, including when reusing an output buffer. Caller parameters are untouched. */
export function packMlpWeights(network: MlpNetwork, out?: Float32Array): Float32Array {
    const layout = mlpLayout(network);
    out ??= new Float32Array(layout.floats);
    if (!(out instanceof Float32Array) || out.length !== layout.floats) fail(`packed output needs ${layout.floats} floats.`);
    assertSeparate(network, out);
    out.fill(0);
    for (let i = 0; i < network.length; i++) {
        const layer = network[i], packed = layout.layers[i];
        for (let r = 0; r < layer.output; r++) {
            for (let k = 0; k < layer.input; k++) {
                out[packed.weightOffset + r * packed.inputStride + k] = layer.weights[r * layer.input + k];
            }
            out[packed.biasOffset + r] = layer.biases[r];
        }
    }
    return out;
}

/**
 * f32-aware CPU forward: round multiply/add operations and each activation output.
 * Pass output (exact final width) and scratch (at least 2 * mlpLayout().scratchWidth) to reuse
 * storage without allocations. All input/output/scratch views must be non-overlapping, and
 * writable views must not overlap model parameters. Non-finite runtime inputs are not sanitized.
 */
export function mlpForward(network: MlpNetwork, input: Float32Array,
    out?: Float32Array, scratch?: Float32Array): Float32Array {
    const width = validate(network), last = network[network.length - 1];
    if (!(input instanceof Float32Array) || input.length !== network[0].input) fail(`input needs ${network[0].input} floats.`);
    out ??= new Float32Array(last.output);
    scratch ??= new Float32Array(width * 2);
    if (!(out instanceof Float32Array) || out.length !== last.output) fail(`output needs ${last.output} floats.`);
    if (!(scratch instanceof Float32Array) || scratch.length < width * 2) fail(`scratch needs at least ${width * 2} floats.`);
    if (overlaps(input, out) || overlaps(input, scratch) || overlaps(out, scratch)) fail('input/output/scratch views overlap.');
    assertSeparate(network, out);
    assertSeparate(network, scratch);
    const f = Math.fround;
    let source = 0, destination = width;
    for (let k = 0; k < input.length; k++) scratch[k] = input[k];
    for (const layer of network) {
        for (let r = 0; r < layer.output; r++) {
            let z = layer.biases[r];
            for (let k = 0; k < layer.input; k++) {
                z = f(z + f(layer.weights[r * layer.input + k] * scratch[source + k]));
            }
            if (layer.activation === 'relu') z = Math.max(0, z);
            else if (layer.activation === 'tanh') z = Math.tanh(Math.max(-15, Math.min(15, z)));
            else if (layer.activation === 'sigmoid') {
                const e = Math.exp(-Math.abs(z));
                z = z >= 0 ? 1 / (1 + e) : e / (1 + e);
            }
            scratch[destination + r] = f(z);
        }
        const previous = source;
        source = destination;
        destination = previous;
    }
    for (let k = 0; k < last.output; k++) out[k] = scratch[source + k];
    return out;
}