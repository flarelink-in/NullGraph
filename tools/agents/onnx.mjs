// Step 7, part 2 (docs/agents-cloning-design.md §9, phase 7.10): a network as an .onnx file.
//
// One Gemm per layer (Y = X·Wᵀ + b, with transB = 1, so W is the layer's row-major [output][input]
// weights as they are), followed by Tanh, Relu or Sigmoid; an identity layer has no second node.
// The input is "obs", float32 [batch, inputs]; the output is "act", float32 [batch, outputs].
// Weights are stored as raw little-endian f32, so they are the network's, bit for bit.
//
// MlpBrain clamps tanh's input to ±15. An ONNX Tanh does not, and need not: tanh(15) is 1 in f32.
//
// Hand-encoded protobuf, as test/onnx-gpu/tiny-model.mjs is: no dependency, and the same network
// gives the same bytes. Field numbers are from onnx.proto (IR v8):
//   ModelProto      1 ir_version · 2 producer_name · 7 graph · 8 opset_import · 14 metadata_props
//   OperatorSetId   1 domain · 2 version          StringStringEntry  1 key · 2 value
//   GraphProto      1 node · 2 name · 5 initializer · 11 input · 12 output
//   NodeProto       1 input · 2 output · 3 name · 4 op_type · 5 attribute
//   AttributeProto  1 name · 3 i · 20 type (2 = INT)
//   TensorProto     1 dims · 2 data_type (1 = FLOAT) · 8 name · 9 raw_data
//   ValueInfoProto  1 name · 2 type
//   TypeProto       1 tensor_type → { 1 elem_type, 2 shape → { 1 dim → { 1 dim_value | 2 dim_param } } }
//
// Plain JS, no node: imports. Reading ONNX is step 8 (MlpBrain.fromOnnx), not this file.

export const ONNX_OPSET = 13;
const OPS = Object.freeze({ identity: null, relu: 'Relu', tanh: 'Tanh', sigmoid: 'Sigmoid' });
const utf8 = new TextEncoder();

function varint(n) {
    const out = [];
    let v = BigInt(n);
    do {
        let b = Number(v & 0x7fn);
        v >>= 7n;
        if (v) b |= 0x80;
        out.push(b);
    } while (v);
    return out;
}
const key = (field, wire) => varint((field << 3) | wire);
const int = (field, n) => [...key(field, 0), ...varint(n)];
const msg = (field, bytes) => [...key(field, 2), ...varint(bytes.length), ...bytes];
const str = (field, s) => msg(field, utf8.encode(s));

/** A float tensor's value info; a string dimension is a named, free one. */
function valueInfo(name, dims) {
    const shape = dims.flatMap(d => msg(1, typeof d === 'string' ? str(2, d) : int(1, d)));
    return [...str(1, name), ...msg(2, msg(1, [...int(1, 1), ...msg(2, shape)]))];
}

function floatTensor(name, dims, values) {
    const bytes = new Uint8Array(values.length * 4), view = new DataView(bytes.buffer);
    values.forEach((v, k) => view.setFloat32(k * 4, v, true));
    return [...dims.flatMap(d => int(1, d)), ...int(2, 1), ...str(8, name), ...msg(9, bytes)];
}

/**
 * The bytes of an .onnx file for `network` (an MlpNetwork). `metadata` is stored as the model's
 * metadata_props, in key order as given: the names a reader should check, for instance.
 */
export function onnxFromNetwork(network, { name = 'null-graph-mlp', metadata = {} } = {}) {
    if (!(network?.length > 0)) throw new Error('onnx: a network needs at least one layer.');
    const nodes = [], initializers = [];
    let input = 'obs';
    network.forEach((layer, l) => {
        if (!(layer.activation in OPS)) {
            throw new Error(`onnx: layer ${l} has activation "${layer.activation}"; the writer takes ${Object.keys(OPS).join(', ')}.`);
        }
        if (layer.weights.length !== layer.input * layer.output || layer.biases.length !== layer.output) {
            throw new Error(`onnx: layer ${l} is ${layer.input}→${layer.output} but holds ${layer.weights.length} weights and ${layer.biases.length} biases.`);
        }
        if (l > 0 && layer.input !== network[l - 1].output) {
            throw new Error(`onnx: layer ${l} reads ${layer.input} values; layer ${l - 1} writes ${network[l - 1].output}.`);
        }
        const last = l === network.length - 1, op = OPS[layer.activation], out = last ? 'act' : `h${l}`, sum = op ? `z${l}` : out;
        initializers.push(floatTensor(`w${l}`, [layer.output, layer.input], layer.weights), floatTensor(`b${l}`, [layer.output], layer.biases));
        nodes.push([...str(1, input), ...str(1, `w${l}`), ...str(1, `b${l}`), ...str(2, sum), ...str(3, `gemm${l}`), ...str(4, 'Gemm'),
            ...msg(5, [...str(1, 'transB'), ...int(3, 1), ...int(20, 2)])]);
        if (op) nodes.push([...str(1, sum), ...str(2, out), ...str(3, `${layer.activation}${l}`), ...str(4, op)]);
        input = out;
    });
    const graph = [
        ...nodes.flatMap(node => msg(1, node)),
        ...str(2, name),
        ...initializers.flatMap(tensor => msg(5, tensor)),
        ...msg(11, valueInfo('obs', ['batch', network[0].input])),
        ...msg(12, valueInfo('act', ['batch', network[network.length - 1].output])),
    ];
    return new Uint8Array([
        ...int(1, 8),                                   // ir_version
        ...str(2, 'null-graph tools/agents'),           // producer_name
        ...msg(7, graph),
        ...msg(8, int(2, ONNX_OPSET)),                  // opset_import { version } (default domain)
        ...Object.entries(metadata).flatMap(([k, v]) => msg(14, [...str(1, k), ...str(2, String(v))])),
    ]);
}
