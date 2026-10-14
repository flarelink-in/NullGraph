import type { MlpActivation, MlpLayer, MlpNetwork } from '../agents/mlpMath';

/** What `mlpFromOnnx` reads from an .onnx file. */
export interface OnnxMlp {
    /** The layers, ready for MlpBrain, MlpBrain.bySpecies, setWeights or mlpForward. */
    readonly network: MlpNetwork;
    /** The graph's input and output names, for instance "obs" and "act". */
    readonly input: string;
    readonly output: string;
    /** The model's metadata_props as written; empty when the file has none. */
    readonly metadata: Readonly<Record<string, string>>;
}

/** A file of weights that an exporter stored beside the .onnx file: `path` as the .onnx file names it. */
export interface OnnxExternalData {
    readonly path: string;
    readonly data: Uint8Array | ArrayBuffer;
}

export interface OnnxReadOptions {
    /** The files that hold weights stored outside the .onnx file: PyTorch's default call writes model.onnx.data. */
    readonly externalData?: readonly OnnxExternalData[];
}

function fail(message: string): never { throw new Error(`mlpFromOnnx: ${message}`); }
const CUT = 'the file ends inside a field; it is cut short or is not ONNX.';
const NOT_PROTOBUF = 'the file\'s bytes are not protobuf; it is not ONNX.';

/** One protobuf field: a varint's value in `v`, any other wire type's bytes in `b`. */
interface Field { readonly n: number; readonly w: number; readonly v: number; readonly b: Uint8Array }
const EMPTY = /* @__PURE__ */ new Uint8Array(0);
const utf8 = /* @__PURE__ */ new TextDecoder();

function cursor(b: Uint8Array) {
    let at = 0;
    return {
        more: () => at < b.length,
        /** Exact to 2^53. Ten bytes at most: beyond that the sum stops being a number. */
        varint(): number {
            let value = 0, scale = 1;
            for (let k = 0; k < 10; k++) {
                if (at >= b.length) fail(CUT);
                const byte = b[at++];
                value += (byte & 127) * scale;
                if (byte < 128) return value;
                scale *= 128;
            }
            return fail(NOT_PROTOBUF);
        },
        take(length: number): Uint8Array {
            if (length > b.length - at) fail(CUT);
            at += length;
            return b.subarray(at - length, at);
        },
    };
}

/** A message's fields in order. Nothing is copied: `b` is a view of the file. */
function fields(b: Uint8Array): Field[] {
    const c = cursor(b), out: Field[] = [];
    while (c.more()) {
        const tag = c.varint(), w = tag % 8, n = (tag - w) / 8;
        if (w === 0) out.push({ n, w, v: c.varint(), b: EMPTY });
        else out.push({ n, w, v: 0, b: c.take(w === 2 ? c.varint() : w === 5 ? 4 : w === 1 ? 8 : fail(NOT_PROTOBUF)) });
    }
    return out;
}
const all = (f: Field[], n: number) => f.filter(x => x.n === n);
const one = (f: Field[], n: number) => f.find(x => x.n === n);
const text = (f: Field[], n: number) => utf8.decode(one(f, n)?.b ?? EMPTY);
/** A repeated number field, packed or not. */
function numbers(f: Field[], n: number): number[] {
    const out: number[] = [];
    for (const x of all(f, n)) {
        if (x.w === 0) out.push(x.v);
        else for (const c = cursor(x.b); c.more();) out.push(c.varint());
    }
    return out;
}
const view = (b: Uint8Array) => new DataView(b.buffer, b.byteOffset, b.byteLength);
const bytesOf = (b: ArrayBufferView | ArrayBuffer) => ArrayBuffer.isView(b) ? new Uint8Array(b.buffer, b.byteOffset, b.byteLength) : new Uint8Array(b);
/** A whole number written as text, as ONNX writes an offset or a length; NaN for anything else. */
const whole = (s: string) => /^\d{1,15}$/.test(s) ? +s : NaN;
const defaultDomain = (domain: string) => domain === '' || domain === 'ai.onnx';

interface Node { op: string; name: string; inputs: string[]; outputs: string[]; attr: Map<string, number> }
interface Tensor { dims: number[]; data: Float32Array }
// A Map, not an object: an operator named "constructor" must not look like an activation.
const ACTIVATIONS = /* @__PURE__ */ new Map<string, MlpActivation>([['Relu', 'relu'], ['Tanh', 'tanh'], ['Sigmoid', 'sigmoid']]);

/**
 * The dense network in an .onnx file's bytes, for `new MlpBrain(engine, network)`.
 *
 * Takes one chain of layers from the graph's input to its output. A layer is `Gemm`, or `MatMul`
 * with an optional `Add`, then at most one of `Relu`, `Tanh`, `Sigmoid`; its weights and bias are
 * float32 tensors stored in the file, or an `Identity` of one. A `Cast` to float32 or a `Flatten`
 * before the first layer and a `Reshape` after the last are passed over. Anything else is refused,
 * naming the node.
 *
 * Weights stored outside the file are read from `externalData`, the same list ONNX Runtime takes,
 * with bytes as `data`. A file is found by its path exactly as the .onnx file writes it; nothing is
 * opened or fetched.
 *
 * The weights come back bit for bit, in arrays of their own. MlpBrain's limits (4 layers, the
 * widths, finite values) are checked by MlpBrain, not here.
 */
export function mlpFromOnnx(bytes: Uint8Array | ArrayBuffer, options: OnnxReadOptions = {}): OnnxMlp {
    const file = ArrayBuffer.isView(bytes) || bytes instanceof ArrayBuffer ? bytesOf(bytes) : fail('expected the file\'s bytes, as a Uint8Array or an ArrayBuffer.');
    // Exporters store equal tensors once, so a layer may share its weights with another. Each use
    // is copied, and the copies together may not outgrow the files: a network of MlpBrain's four
    // layers cannot, and a file built to exhaust memory is refused before it does.
    let budget = file.length;
    const files = new Map<string, Uint8Array>(), given = options.externalData ?? [];
    for (const x of Array.isArray(given) ? given : [0]) {
        const data = x?.data;
        if (typeof x?.path !== 'string' || !(ArrayBuffer.isView(data) || data instanceof ArrayBuffer)) {
            fail('externalData must be a list of { path, data }, with data a Uint8Array or an ArrayBuffer; a URL is for ONNX Runtime, which fetches it.');
        }
        files.set(x.path, bytesOf(data));
        budget += data.byteLength;
    }
    const model = fields(file);
    const graph = fields(one(model, 7)?.b ?? fail('the file has no graph; it is not an ONNX model.'));
    const set = all(model, 8).map(x => fields(x.b)).find(f => defaultDomain(text(f, 1)));
    const version = (set && one(set, 2)?.v) || 0;
    if (version < 7) fail(`the model's operator set is version ${version}; the reader takes version 7 or later.`);

    // First pass: nodes and tensor names. No tensor is copied before the graph has been checked.
    const tensors = new Map<string, Field[]>();
    for (const x of all(graph, 5)) {
        const f = fields(x.b);
        tensors.set(text(f, 8), f);
    }
    const nodes: Node[] = all(graph, 1).map(x => {
        const f = fields(x.b), domain = text(f, 7), attr = new Map<string, number>();
        for (const a of all(f, 5)) {
            // A value left out is zero, as protobuf defines; a float is 4 bytes, an integer a varint.
            const g = fields(a.b), float = one(g, 2);
            attr.set(text(g, 1), float ? (float.b.length === 4 ? view(float.b).getFloat32(0, true) : fail(NOT_PROTOBUF)) : one(g, 3)?.v ?? 0);
        }
        return { op: (defaultDomain(domain) ? '' : domain + '.') + text(f, 4), name: text(f, 3),
            inputs: all(f, 1).map(y => utf8.decode(y.b)), outputs: all(f, 2).map(y => utf8.decode(y.b)), attr };
    });
    const named = (n: number) => all(graph, n).map(x => text(fields(x.b), 1));
    // Files of IR version 3 and earlier list their initializers among the inputs.
    const inputs = named(11).filter(name => !tensors.has(name)), outputs = named(12);
    if (inputs.length !== 1 || outputs.length !== 1) {
        fail(`the graph has ${inputs.length} inputs and ${outputs.length} outputs; a dense network has one of each.`);
    }
    // PyTorch's older exporter stores equal tensors once and hands the other layers an Identity of it.
    const identity = new Map(nodes.filter(n => n.op === 'Identity').map(n => [n.outputs[0], n]));
    const readers = new Map<string, Node[]>();
    for (const node of nodes) {
        for (const name of node.inputs) {
            const list = readers.get(name);
            if (list) list.push(node);
            else readers.set(name, [node]);
        }
    }

    const refuse = (node: Node, why: string): never => fail(`node "${node.name || node.outputs[0]}" (${node.op}) ${why}`);
    const reached = new Set<Node>();
    /** The one node that reads `value`, or none: the end of the chain. */
    const next = (value: string): Node | undefined => {
        const list = readers.get(value);
        if (!list) return undefined;
        if (list.length > 1) refuse(list[1], `also reads "${value}"; a dense network is one chain of layers.`);
        if (reached.has(list[0])) refuse(list[0], 'is reached twice; the graph loops.');
        reached.add(list[0]);
        return list[0];
    };
    /** A node's input as f32 values, decoded only now. An Identity of an initializer stands for it. */
    const tensor = (node: Node, slot: number): Tensor => {
        let name = node.inputs[slot];
        for (let hops = 0, alias: Node | undefined; !tensors.has(name) && (alias = identity.get(name)) && hops < nodes.length; hops++) {
            reached.add(alias);
            name = alias.inputs[0];
        }
        const t = tensors.get(name) ?? refuse(node, `needs "${node.inputs[slot]}" to be a float32 initializer.`);
        const type = one(t, 2)?.v ?? 0, dims = numbers(t, 1), raw = one(t, 9), count = dims.reduce((a, d) => a * d, 1);
        if (type !== 1) fail(`initializer "${name}" has data type ${type}; only float32 (1) is read. Export the model as float32.`);
        // Raw bytes, or a float list: one packed field, or one field per value.
        let chunks = raw ? [raw.b] : all(t, 4).map(x => x.b);
        if (one(t, 14)?.v === 1) {
            // Stored outside the file: a location, an offset and a length, each written as text.
            const entry = new Map(all(t, 13).map(x => { const f = fields(x.b); return [text(f, 1), text(f, 2)]; }));
            const where = entry.get('location') ?? '';
            const data = files.get(where) ?? fail(`initializer "${name}" is stored outside the file, in "${where}"; pass that file, { externalData: [{ path: '${where}', data }] }, or export with the weights inside it (PyTorch: external_data=False).`);
            const at = whole(entry.get('offset') ?? '0'), length = entry.has('length') ? whole(entry.get('length')!) : data.length - at;
            if (!(length >= 0 && at + length <= data.length)) {
                fail(`initializer "${name}" reads ${entry.get('length') ?? 'the rest'} B at offset ${entry.get('offset') ?? 0} of "${where}", which holds ${data.length} B.`);
            }
            chunks = [data.subarray(at, at + length)];
        }
        let size = 0;
        for (const c of chunks) size += c.length % 4 ? NaN : c.length;
        if (size !== count * 4) fail(`initializer "${name}" is [${dims}] but holds ${chunks.reduce((a, c) => a + c.length, 0) / 4} values.`);
        if ((budget -= count) < 0) fail(`initializer "${name}" is used so often that the layers hold more values than the file has bytes.`);
        const data = new Float32Array(count);
        let k = 0;
        for (const c of chunks) for (let at = 0, v = view(c); at < c.length; at += 4) data[k++] = v.getFloat32(at, true);
        return { dims, data };
    };

    const network: MlpLayer[] = [];
    let value = inputs[0], node = next(value);
    // Before the first layer, nodes that change no value of one vector.
    while (node && (node.op === 'Cast' && node.attr.get('to') === 1 || node.op === 'Flatten' && (node.attr.get('axis') ?? 1) === 1)) {
        value = node.outputs[0];
        node = next(value);
    }
    while (node) {
        // After the last layer, a Reshape that nothing reads: the values keep their order.
        if (node.op === 'Reshape' && network.length && !readers.has(node.outputs[0])) {
            value = node.outputs[0];
            break;
        }
        const layer = node, gemm = node.op === 'Gemm';
        if (!gemm && node.op !== 'MatMul') {
            if (node.op === 'Add' || ACTIVATIONS.has(node.op)) refuse(node, 'is out of place: an Add follows a MatMul, and an activation follows a layer.');
            fail(`node "${node.name || node.outputs[0]}" is ${node.op}; supported: Gemm, MatMul, Add, Relu, Tanh, Sigmoid.`);
        }
        if (node.inputs[0] !== value) refuse(node, `must read "${value}" as its first input.`);
        if (gemm && ((node.attr.get('alpha') ?? 1) !== 1 || (node.attr.get('beta') ?? 1) !== 1 || node.attr.get('transA'))) {
            refuse(node, 'has alpha, beta or transA set; only the defaults are read.');
        }
        // Gemm with transB = 1 holds [output, input], as MlpLayer does; every other form holds [input, output].
        const w = tensor(node, 1), flip = !gemm || !node.attr.get('transB');
        let bias = gemm && node.inputs[2] ? tensor(node, 2) : undefined;
        value = node.outputs[0];
        node = next(value);
        if (!gemm && node?.op === 'Add') {
            bias = tensor(node, node.inputs[0] === value ? 1 : 0);
            value = node.outputs[0];
            node = next(value);
        }
        if (w.dims.length !== 2) refuse(layer, `has weights of shape [${w.dims}]; a dense layer's are [rows, columns].`);
        const input = w.dims[flip ? 0 : 1], output = w.dims[flip ? 1 : 0];
        if (bias && (bias.data.length !== output || bias.dims.length > 2 || bias.dims.length === 2 && bias.dims[0] !== 1)) {
            refuse(layer, `has a bias of shape [${bias.dims}] for ${output} outputs.`);
        }
        const weights = new Float32Array(input * output);
        for (let r = 0; r < output; r++) {
            for (let k = 0; k < input; k++) weights[r * input + k] = w.data[flip ? k * output + r : r * input + k];
        }
        const activation = node && ACTIVATIONS.get(node.op);
        if (activation) {
            value = node!.outputs[0];
            node = next(value);
        }
        network.push({ input, output, activation: activation || 'identity', weights, biases: bias ? bias.data : new Float32Array(output) });
    }
    if (!network.length) fail('the graph has no layers.');
    if (value !== outputs[0]) fail(`the layers end at "${value}", not at the graph's output "${outputs[0]}".`);
    for (const stray of nodes) if (!reached.has(stray)) refuse(stray, 'is not on the chain from the input to the output.');
    // fromEntries defines each key as its own property, "__proto__" included.
    return { network, input: inputs[0], output: outputs[0],
        metadata: Object.freeze(Object.fromEntries(all(model, 14).map(x => { const f = fields(x.b); return [text(f, 1), text(f, 2)]; }))) };
}
