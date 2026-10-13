// Step 8 (docs/agents-onnx-reader-design.md §10): ONNX graphs encoded by hand for the reader's
// tests. tools/agents/onnx.mjs writes one form of a dense network; this builds the other forms the
// reader takes, and files that are wrong in one chosen way. Plain JS, no node: imports, so the GPU
// suites can use it in a browser too.
import { int, msg, str, valueInfo, varint } from '../../tools/agents/onnx.mjs';

const f32 = value => {
    const bytes = new Uint8Array(4);
    new DataView(bytes.buffer).setFloat32(0, value, true);
    return [...bytes];
};
const OPS = { relu: 'Relu', tanh: 'Tanh', sigmoid: 'Sigmoid' };

/** An attribute: an integer, or a float with `{ float: true }`. `{ bare: true }` leaves its value out. */
export const attr = (name, value, { float = false, bare = false } = {}) =>
    [...str(1, name), ...(bare ? [] : float ? [0x15, ...f32(value)] : int(3, value)), ...int(20, float ? 1 : 2)];

/** A node. `domain` is written only when given. */
export const node = (op, inputs, outputs, { name = '', attributes = [], domain } = {}) => [
    ...inputs.flatMap(v => str(1, v)), ...outputs.flatMap(v => str(2, v)), ...(name ? str(3, name) : []), ...str(4, op),
    ...attributes.flatMap(a => msg(5, a)), ...(domain === undefined ? [] : str(7, domain))];

/**
 * A tensor. `as` says how its values are stored: 'raw' bytes, a packed float 'list', or an
 * 'unpacked' list of one field per value. `type` 1 is float32. `external` names the file that
 * would hold the values, and then none are written here.
 */
export function tensor(name, dims, values, { as = 'raw', packedDims = false, type = 1, external } = {}) {
    const floats = values.flatMap(f32);
    return [
        ...(packedDims ? msg(1, dims.flatMap(d => varint(d))) : dims.flatMap(d => int(1, d))),
        ...int(2, type),
        ...(external || as === 'raw' ? [] : as === 'list' ? msg(4, floats) : values.flatMap(v => [0x25, ...f32(v)])),
        ...str(8, name),
        ...(!external && as === 'raw' ? msg(9, floats) : []),
        ...(external ? [...msg(13, [...str(1, 'location'), ...str(2, external)]), ...int(14, 1)] : []),
    ];
}

/** The bytes of a model. `extra` is appended as it is: fields the reader must skip, or damage. */
export function model({ nodes = [], initializers = [], inputs = [['obs', ['batch', 5]]], outputs = [['act', ['batch', 3]]],
    opsets = [['', 13]], metadata = {}, ir = 8, graph = true, extra = [] } = {}) {
    const body = [
        ...nodes.flatMap(n => msg(1, n)), ...str(2, 'test'), ...initializers.flatMap(t => msg(5, t)),
        ...inputs.flatMap(([name, dims]) => msg(11, valueInfo(name, dims))),
        ...outputs.flatMap(([name, dims]) => msg(12, valueInfo(name, dims))),
    ];
    return new Uint8Array([
        ...int(1, ir), ...str(2, 'null-graph tests'), ...(graph ? msg(7, body) : []),
        ...opsets.flatMap(([domain, version]) => msg(8, [...(domain ? str(1, domain) : []), ...int(2, version)])),
        ...Object.entries(metadata).flatMap(([k, v]) => msg(14, [...str(1, k), ...str(2, v)])),
        ...extra,
    ]);
}

/** A layer's weights as [input][output]: what MatMul, and Gemm without transB, hold. */
export function transposed(layer) {
    const out = new Array(layer.weights.length);
    for (let r = 0; r < layer.output; r++) for (let k = 0; k < layer.input; k++) out[k * layer.output + r] = layer.weights[r * layer.input + k];
    return out;
}

/**
 * `network` (an MlpNetwork) as one chain, in a chosen form. Each option varies one thing:
 *   form      'gemm' (transB = 1, weights as they are) · 'gemm0' (transB = 0 written) ·
 *             'gemmDefault' (no transB at all) · 'matmul' (MatMul, then Add(result, bias)) ·
 *             'matmulBiasFirst' (Add(bias, result))
 *   bias      'vector' ([output]) · 'row' ([1, output]) · 'none' (left out: the layer's must be zero)
 *   tensors   options for tensor(): { as, packedDims }
 *   defaults  write Gemm's alpha = 1, beta = 1 and transA = 0 out, as PyTorch does
 *   bareTransB  with 'gemm0': the attribute is there and its value is left out
 *   asInputs  list every initializer among the graph's inputs, as files of IR version 3 do
 *   head, tail  nodes that change no value: head from 'Cast' and 'Flatten', tail ['Reshape']
 *   identity  each layer reads its weights and bias through that many Identity nodes, as PyTorch's
 *             older exporter writes tensors that hold equal values
 */
export function chain(network, { form = 'gemm', bias = 'vector', tensors = {}, defaults = false, bareTransB = false, asInputs = false,
    head = [], tail = [], identity = 0, opsets = [['', 13]], metadata = {}, domain } = {}) {
    const nodes = [], initializers = [], constants = [];
    let value = 'obs', count = 0;
    const fresh = () => `v${count++}`;
    const add = (op, inputs, options) => {
        const out = fresh();
        nodes.push(node(op, inputs, [out], { domain, ...options }));
        value = out;
    };
    /** Stores a tensor; returns the name a layer reads it by. */
    const constant = (name, dims, values) => {
        initializers.push(tensor(name, dims, values, tensors));
        constants.push([name, dims]);
        let read = name;
        for (let hop = 0; hop < identity; hop++) {
            nodes.push(node('Identity', [read], [`${name}~${hop}`], { name: `same${nodes.length}` }));
            read = `${name}~${hop}`;
        }
        return read;
    };
    for (const op of head) add(op, [value], { name: `head${count}`, attributes: op === 'Cast' ? [attr('to', 1)] : [] });
    network.forEach((layer, l) => {
        const asIs = form === 'gemm';
        const w = constant(`w${l}`, asIs ? [layer.output, layer.input] : [layer.input, layer.output], asIs ? [...layer.weights] : transposed(layer));
        const b = bias === 'none' ? null : constant(`b${l}`, bias === 'row' ? [1, layer.output] : [layer.output], [...layer.biases]);
        if (form.startsWith('gemm')) {
            const attributes = defaults ? [attr('alpha', 1, { float: true }), attr('beta', 1, { float: true }), attr('transA', 0)] : [];
            if (form === 'gemm') attributes.push(attr('transB', 1));
            if (form === 'gemm0') attributes.push(attr('transB', 0, { bare: bareTransB }));
            add('Gemm', [value, w, ...(b ? [b] : [])], { name: `gemm${l}`, attributes });
        } else {
            add('MatMul', [value, w], { name: `matmul${l}` });
            if (b) add('Add', form === 'matmulBiasFirst' ? [b, value] : [value, b], { name: `add${l}` });
        }
        if (OPS[layer.activation]) add(OPS[layer.activation], [value], { name: `${layer.activation}${l}` });
    });
    for (const op of tail) {
        // Reshape's second input is its target shape, [-1, 1]: an int64 tensor (type 7) as raw bytes.
        initializers.push([...int(1, 2), ...int(2, 7), ...str(8, 'shape'), ...msg(9, [255, 255, 255, 255, 255, 255, 255, 255, 1, 0, 0, 0, 0, 0, 0, 0])]);
        add(op, [value, 'shape'], { name: 'tail' });
    }
    return model({ nodes, initializers, opsets, metadata,
        inputs: [['obs', ['batch', network[0].input]], ...(asInputs ? constants : [])],
        outputs: [[value, ['batch', network[network.length - 1].output]]] });
}
