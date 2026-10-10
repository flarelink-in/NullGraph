// A tiny ONNX model built in memory: Y = X + X, X float32 [1, n], opset 13.
//
// Hand-encoded protobuf (ModelProto → GraphProto → NodeProto/ValueInfoProto), so the ONNX suite
// has no binary fixture, no download, no converter and no third-party weights — the model is
// these bytes, deterministically. It exists to prove the DEVICE AND BUFFER seam, not to be
// a useful network: one op is enough to show ORT computed on the GPU (2x, not x) and that the
// result reached NullGraph without leaving it.
//
// Field numbers are from onnx.proto (onnx/onnx, IR v8):
//   ModelProto      1 ir_version · 2 producer_name · 7 graph · 8 opset_import
//   OperatorSetId   1 domain · 2 version
//   GraphProto      1 node · 2 name · 11 input · 12 output
//   NodeProto       1 input · 2 output · 3 name · 4 op_type
//   ValueInfoProto  1 name · 2 type
//   TypeProto       1 tensor_type → { 1 elem_type (1 = FLOAT), 2 shape → { 1 dim → { 1 dim_value } } }

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
const str = (field, s) => msg(field, [...utf8.encode(s)]);

function tensorValueInfo(name, dims) {
    const shape = dims.flatMap(d => msg(1, int(1, d)));
    const tensorType = [...int(1, 1 /* FLOAT */), ...msg(2, shape)];
    return [...str(1, name), ...msg(2, msg(1, tensorType))];
}

/** ONNX bytes for Y = X + X with X, Y float32 [1, n]. */
export function addModel(n) {
    const node = [...str(1, 'X'), ...str(1, 'X'), ...str(2, 'Y'), ...str(3, 'add'), ...str(4, 'Add')];
    const graph = [
        ...msg(1, node),
        ...str(2, 'nullgraph-tiny-add'),
        ...msg(11, tensorValueInfo('X', [1, n])),
        ...msg(12, tensorValueInfo('Y', [1, n])),
    ];
    const model = [
        ...int(1, 8),                         // ir_version
        ...str(2, 'null-graph test suite'),   // producer_name
        ...msg(8, int(2, 13)),                // opset_import { version: 13 } (default domain)
        ...msg(7, graph),
    ];
    return new Uint8Array(model);
}
