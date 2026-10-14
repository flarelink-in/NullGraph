// Step 8 (docs/agents-onnx-reader-design.md §10): mlpFromOnnx, the reader of null-graph/inference.
//
//   U12  the reader half: what tools/agents/onnx.mjs writes comes back bit for bit
//   U29  every form of the dense subset gives the same network
//   U30  no file hangs the reader or breaks it: prefixes, changed bytes, named cases
//   U31  the files other exporters wrote (test/onnx-fixtures/fixtures.json) behave as listed here
//   U14  every refusal of the design's §11 fires and names the offender
//
// Imports dist/: run `npm run build:lib` first.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';
import { build } from 'esbuild';
import * as A from '../dist/agents.js';
import * as inference from '../dist/inference.js';
import { int, msg, onnxFromNetwork } from '../tools/agents/onnx.mjs';
import { attr, chain, model, node, tensor, transposed } from './onnx-fixtures/build.mjs';

const { mlpFromOnnx } = inference;
const root = fileURLToPath(new URL('../', import.meta.url)).replace(/\\/g, '/');
const bits = array => [...new Uint32Array(array.buffer, array.byteOffset, array.length)];
const layer = (input, output, activation, stream) => ({ input, output, activation,
    weights: Float32Array.from({ length: input * output }, (_, k) => A.rand01(k, stream, 6) * 4 - 2),
    biases: Float32Array.from({ length: output }, (_, k) => A.rand01(k, stream, 7) - 0.5) });
/** Two networks are the same layers, to the bit. */
function same(got, want, label = '') {
    assert.deepEqual(got.map(l => [l.input, l.output, l.activation]), want.map(l => [l.input, l.output, l.activation]), `${label}: shapes and activations`);
    got.forEach((l, k) => {
        assert.ok(l.weights instanceof Float32Array && l.biases instanceof Float32Array, `${label}: layer ${k} holds Float32Arrays`);
        assert.deepEqual(bits(l.weights), bits(want[k].weights), `${label}: layer ${k} weights`);
        assert.deepEqual(bits(l.biases), bits(want[k].biases), `${label}: layer ${k} biases`);
    });
}
/** Every refusal is an Error that says who refused. */
const refused = (bytes, pattern, label) => assert.throws(() => mlpFromOnnx(bytes),
    error => error instanceof Error && error.message.startsWith('mlpFromOnnx: ') && pattern.test(error.message), `${label ?? pattern}`);

// A middle layer that is square: a transpose that is forgotten keeps every shape and is still wrong.
const BASE = [layer(5, 7, 'tanh', 10), layer(7, 7, 'relu', 11), layer(7, 3, 'identity', 12)];
const NO_BIAS = BASE.map(l => ({ ...l, biases: new Float32Array(l.output) }));
const W = (name, rows, columns, options) => tensor(name, [rows, columns], Array.from({ length: rows * columns }, (_, k) => k / 8 - 1), options);
const B = (name, n, options) => tensor(name, [n], Array.from({ length: n }, (_, k) => k / 4), options);
const gemm = (input, w, b, output, options) => node('Gemm', b ? [input, w, b] : [input, w], [output], { attributes: [attr('transB', 1)], ...options });

test('the entry: null-graph/inference is built, declared and exports the reader and, since phase 9.1, the ONNX Runtime adapter', () => {
    assert.deepEqual(Object.keys(inference).sort(), ['OrtModel', 'OrtRuntime', 'mlpFromOnnx']);
    const entry = JSON.parse(readFileSync(root + 'package.json', 'utf8')).exports['./inference'];
    assert.deepEqual(entry, { types: './dist/extras/inference/index.d.ts', import: './dist/inference.js' });
    for (const path of Object.values(entry)) assert.ok(existsSync(root + path), `${path} is missing: run npm run build:lib`);
});

test('U12 (the reader): what the writer wrote comes back bit for bit, with its names and metadata', () => {
    const network = [layer(5, 7, 'tanh', 0), layer(7, 4, 'relu', 1), layer(4, 4, 'identity', 2), layer(4, 3, 'sigmoid', 3), layer(3, 2, 'identity', 4)];
    network[0].weights.set([1e-45, 3.4028234663852886e38, -0, -1 / 3]);
    const metadata = { species: 'prey', obsNames: '["a","b"]', actNames: '["turn","throttle"]' };
    const bytes = onnxFromNetwork(network, { name: 'five', metadata });
    const read = mlpFromOnnx(bytes);
    same(read.network, network, 'five layers');
    assert.equal(bits(read.network[0].weights)[2], 0x80000000, 'raw bytes keep even a -0');
    assert.deepEqual([read.input, read.output], ['obs', 'act']);
    assert.deepEqual(read.metadata, metadata);
    assert.ok(Object.isFrozen(read.metadata));

    // The same bytes as an ArrayBuffer, as a view at an odd offset, and as another kind of view.
    const shifted = new Uint8Array(bytes.length + 1);
    shifted.set(bytes, 1);
    same(mlpFromOnnx(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length)).network, network, 'an ArrayBuffer');
    same(mlpFromOnnx(shifted.subarray(1)).network, network, 'a view at byte 1');
    same(mlpFromOnnx(new DataView(shifted.buffer, 1)).network, network, 'a DataView');

    // The arrays are the reader's own: wiping the file afterwards changes nothing.
    shifted.fill(0);
    bytes.fill(0);
    same(read.network, network, 'after the file was wiped');
    const arrays = read.network.flatMap(l => [l.weights, l.biases]);
    assert.equal(new Set(arrays.map(a => a.buffer)).size, arrays.length, 'one buffer per array');

    // The reader has no limit of its own. MlpBrain's are MlpBrain's: five layers read, and are refused there.
    assert.throws(() => A.mlpLayout(read.network), /MlpBrain: expected 1–4 affine layers\./);
    const four = mlpFromOnnx(onnxFromNetwork(network.slice(0, 4))).network, x = Float32Array.from([0.5, -1, 2, 0.25, -0.75]);
    assert.deepEqual(bits(A.mlpForward(four, x)), bits(A.mlpForward(network.slice(0, 4), x)));

    // The parent's U12: an unsupported operator is refused by name.
    refused(model({ nodes: [node('Conv', ['obs', 'w'], ['act'], { name: 'conv1' })], initializers: [W('w', 3, 5)] }),
        /^mlpFromOnnx: node "conv1" is Conv; supported: Gemm, MatMul, Add, Relu, Tanh, Sigmoid\.$/);
});

test('U29: every form of the dense subset gives the same network, to the bit', () => {
    const forms = [
        ['Gemm with transB = 1', BASE, {}],
        ['Gemm with transB = 0', BASE, { form: 'gemm0' }],
        ['Gemm with transB left out', BASE, { form: 'gemmDefault' }],
        ['Gemm whose transB has no value', BASE, { form: 'gemm0', bareTransB: true }],
        ['Gemm with its defaults written out', BASE, { defaults: true }],
        ['MatMul, then Add(result, bias)', BASE, { form: 'matmul' }],
        ['MatMul, then Add(bias, result)', BASE, { form: 'matmulBiasFirst' }],
        ['MatMul alone', NO_BIAS, { form: 'matmul', bias: 'none' }],
        ['Gemm without a bias', NO_BIAS, { bias: 'none' }],
        ['Gemm without a bias, transB = 0', NO_BIAS, { form: 'gemm0', bias: 'none' }],
        ['a bias of [1, output]', BASE, { form: 'matmul', bias: 'row' }],
        ['a packed float list', BASE, { tensors: { as: 'list' } }],
        ['a float list of one field per value', BASE, { form: 'matmul', tensors: { as: 'unpacked' } }],
        ['packed dims', BASE, { tensors: { packedDims: true } }],
        ['initializers listed among the inputs', BASE, { asInputs: true }],
        ['another operator set declared and unused', BASE, { opsets: [['ai.onnx.ml', 3], ['', 21]] }],
        ['the default domain written as ai.onnx', BASE, { domain: 'ai.onnx', opsets: [['ai.onnx', 13]] }],
        ['a Cast to float32 first', BASE, { head: ['Cast'] }],
        ['a Flatten first', BASE, { head: ['Flatten'] }],
        ['a Reshape last', BASE, { tail: ['Reshape'] }],
        ['Cast and Flatten first, Reshape last, as scikit-learn writes its layers', BASE,
            { form: 'matmul', bias: 'row', tensors: { as: 'list' }, head: ['Cast', 'Flatten'], tail: ['Reshape'] }],
        ['weights and biases read through an Identity, as PyTorch\'s older exporter writes equal tensors', BASE, { identity: 1 }],
        ['through two Identity nodes, into MatMul and Add', BASE, { form: 'matmulBiasFirst', identity: 2 }],
    ];
    for (const [label, network, options] of forms) same(mlpFromOnnx(chain(network, options)).network, network, label);
    // The forms are different bytes: a builder that ignored its options would pass the loop above.
    assert.equal(new Set(forms.map(([, network, options]) => createHash('sha256').update(chain(network, options)).digest('hex'))).size, forms.length);
    // A transposed square layer really differs: the check above can tell the two apart.
    assert.notDeepEqual(bits(Float32Array.from(transposed(BASE[1]))), bits(BASE[1].weights));

    // One tensor read by two layers, as exporters store equal tensors: each layer gets its own copy.
    const twice = mlpFromOnnx(model({ nodes: [gemm('obs', 'w', 'b', 'h'), node('Tanh', ['h'], ['t']), gemm('t', 'w', 'b', 'act')],
        initializers: [W('w', 5, 5), B('b', 5)], outputs: [['act', ['batch', 5]]] })).network;
    assert.deepEqual(bits(twice[0].weights), bits(twice[1].weights));
    assert.ok(twice[0].weights !== twice[1].weights && twice[0].biases !== twice[1].biases);

    // The nodes that change no value are passed over at the ends only, and only in their harmless form.
    const two = (...middle) => {
        const nodes = [gemm('obs', 'w0', 'b0', 'h')];
        let value = 'h';
        middle.forEach((entry, k) => { nodes.push(node(entry[0], [value, ...(entry[2] ?? [])], [`m${k}`], { name: `mid${k}`, attributes: entry[1] ?? [] })); value = `m${k}`; });
        nodes.push(gemm(value, 'w1', 'b1', 'act'));
        return model({ nodes, initializers: [W('w0', 4, 5), B('b0', 4), W('w1', 3, 4), B('b1', 3), tensor('shape', [2], [], { type: 7 })] });
    };
    assert.equal(mlpFromOnnx(two()).network.length, 2);
    // An Identity is followed from a layer's weights or bias only: on the way of the data it is refused like the others.
    for (const op of ['Cast', 'Flatten', 'Reshape', 'Identity']) {
        refused(two([op, op === 'Cast' ? [attr('to', 1)] : [], op === 'Reshape' ? ['shape'] : []]), new RegExp(`node "mid0" is ${op}; supported: `), `${op} between two layers`);
    }
    const first = (op, attributes) => model({ nodes: [node(op, ['obs'], ['x'], { name: 'first', attributes }), gemm('x', 'w', 'b', 'act')], initializers: [W('w', 3, 5), B('b', 3)] });
    assert.equal(mlpFromOnnx(first('Cast', [attr('to', 1)])).network.length, 1);
    assert.equal(mlpFromOnnx(first('Flatten', [attr('axis', 1)])).network.length, 1);
    refused(first('Cast', [attr('to', 11)]), /node "first" is Cast; supported: /, 'a Cast to float64');
    refused(first('Cast', []), /node "first" is Cast; supported: /, 'a Cast that names no type');
    refused(first('Flatten', [attr('axis', 0)]), /node "first" is Flatten; supported: /, 'a Flatten that would merge the batch');
    refused(first('Reshape', []), /node "first" is Reshape; supported: /, 'a Reshape before any layer');
    // A Reshape whose result something reads is not the end of the chain.
    refused(model({ nodes: [gemm('obs', 'w', 'b', 'h'), node('Reshape', ['h', 'shape'], ['r'], { name: 'reshape' }), node('Tanh', ['r'], ['act'])],
        initializers: [W('w', 3, 5), B('b', 3)] }), /node "reshape" is Reshape; supported: /);
});

test('U30: no file hangs the reader or breaks it', () => {
    const fixtures = JSON.parse(readFileSync(new URL('./onnx-fixtures/fixtures.json', import.meta.url), 'utf8')).cases;
    const two = BASE.slice(0, 2), written = onnxFromNetwork(two, { metadata: { a: '1', b: '2', c: '3' } });

    // (a) Every strict prefix. A file cut exactly between two fields is valid protobuf, so the ones cut
    // after the operator set read, with the metadata that was left; every other one is refused by name.
    const ends = [];
    for (let at = 0; at < written.length;) {             // this test's own walk over the top-level fields
        const varint = () => { let value = 0, shift = 1; for (;;) { const b = written[at++]; value += (b & 127) * shift; if (b < 128) return value; shift *= 128; } };
        const tag = varint(), field = Math.floor(tag / 8), length = varint();   // a varint's value, or a field's length
        if (tag % 8 === 2) at += length;
        ends.push({ field, at });
    }
    assert.deepEqual(ends.map(e => e.field), [1, 2, 7, 8, 14, 14, 14], 'the writer\'s field order');
    const readable = new Map(ends.slice(3, -1).map((e, k) => [e.at, k]));   // after the operator set: 0, 1 and 2 entries of metadata
    let named = 0;
    for (let n = 0; n < written.length; n++) {
        if (readable.has(n)) {
            const read = mlpFromOnnx(written.subarray(0, n));
            same(read.network, two, `a prefix of ${n} bytes`);
            assert.equal(Object.keys(read.metadata).length, readable.get(n));
        } else {
            refused(written.subarray(0, n), /ends inside a field|has no graph|operator set is version 0/, `a prefix of ${n} bytes`);
            named++;
        }
    }
    assert.equal(named, written.length - 3);

    // (b) One byte changed, 10,000 times, over every fixture and two files of our own.
    const pool = [...fixtures.map(c => Uint8Array.from(Buffer.from(c.onnx, 'base64'))), written, chain(BASE, { form: 'matmul', tensors: { as: 'unpacked' }, head: ['Cast'], tail: ['Reshape'] })];
    const outcome = { returned: 0, refused: 0 }, other = [];
    let slowest = 0;
    for (let k = 0; k < 10000; k++) {
        const copy = pool[k % pool.length].slice(), at = Math.floor(A.rand01(k, 1, 80) * copy.length), value = Math.floor(A.rand01(k, 2, 80) * 256);
        copy[at] = copy[at] === value ? value ^ 0xff : value;
        const started = performance.now();
        try {
            const { network } = mlpFromOnnx(copy);
            assert.ok(network.length >= 1 && network.every(l => l.weights.length === l.input * l.output && l.biases.length === l.output), `change ${k}: a network that keeps the reader's promises`);
            outcome.returned++;
        } catch (error) {
            if (error instanceof Error && error.message.startsWith('mlpFromOnnx: ')) outcome.refused++;
            else other.push(`change ${k} (byte ${at} of file ${k % pool.length} set to ${copy[at]}): ${error?.constructor?.name}: ${error?.message}`);
        }
        slowest = Math.max(slowest, performance.now() - started);
    }
    assert.deepEqual(other, [], 'every failure is a refusal by name');
    assert.ok(outcome.returned > 1000 && outcome.refused > 1000, `both outcomes must be common, or this checks nothing: ${JSON.stringify(outcome)}`);
    assert.ok(slowest < 1000, `the slowest read took ${slowest.toFixed(1)} ms`);
    console.log(`U30: ${written.length} prefixes (${named} refused by name, 3 read); 10,000 changed bytes: ${outcome.returned} read, ${outcome.refused} refused by name, slowest ${slowest.toFixed(1)} ms`);

    // (c) Named cases.
    // A varint of eleven bytes: beyond ten, the sum is no longer a number, and a NaN length would end the reading silently.
    refused(new Uint8Array([0x08, ...new Array(10).fill(0x80), 0x01]), /^mlpFromOnnx: the file's bytes are not protobuf; it is not ONNX\.$/);
    refused(new Uint8Array([...new Array(10).fill(0xff), 0x01]), /not protobuf/, 'a tag of eleven bytes');
    // A graph that loops: the second layer writes the value the first one wrote.
    refused(model({ nodes: [gemm('obs', 'w', 'b', 'h', { name: 'one' }), node('Tanh', ['h'], ['t'], { name: 'squash' }), gemm('t', 'w', 'b', 'h', { name: 'two' })],
        initializers: [W('w', 5, 5), B('b', 5)] }), /^mlpFromOnnx: node "squash" \(Tanh\) is reached twice; the graph loops\.$/);
    // Identity nodes that feed each other: the walk back to an initializer gives up, and says which tensor it wanted.
    refused(model({ nodes: [node('Identity', ['a'], ['b'], { name: 'one' }), node('Identity', ['b'], ['a'], { name: 'two' }), gemm('obs', 'a', null, 'act', { name: 'dense' })] }),
        /^mlpFromOnnx: node "dense" \(Gemm\) needs "a" to be a float32 initializer\.$/);
    // Names from the file are never property names of a shared object.
    const odd = mlpFromOnnx(chain(two, { metadata: { ['__proto__']: 'x', constructor: 'y' } })).metadata;
    assert.deepEqual(Object.entries(odd), [['__proto__', 'x'], ['constructor', 'y']]);
    assert.equal(Object.getPrototypeOf(odd), Object.prototype);
    for (const op of ['constructor', 'toString', '__proto__']) {
        refused(model({ nodes: [gemm('obs', 'w', 'b', 'h'), node(op, ['h'], ['act'], { name: 'sly' })], initializers: [W('w', 3, 5), B('b', 3)] }),
            new RegExp(`node "sly" is ${op}; supported: `), `an operator named ${op} is no activation`);
    }
    // Dims that promise more than the file holds are refused before any array is made.
    refused(model({ nodes: [gemm('obs', 'w', null, 'act')], initializers: [tensor('w', [100000, 100000], [1, 2, 3, 4])] }),
        /^mlpFromOnnx: initializer "w" is \[100000,100000\] but holds 4 values\.$/);
    // The graph is judged before any tensor is decoded: a foreign operator is named, not the broken tensor beside it.
    refused(model({ nodes: [node('Conv', ['obs', 'w'], ['act'], { name: 'conv' })], initializers: [tensor('w', [100000, 100000], [1], { type: 11 })] }), /node "conv" is Conv/);
    // One tensor read by many layers cannot make the layers outgrow the file. Four layers, MlpBrain's most, always fit.
    const tied = count => model({ nodes: Array.from({ length: count }, (_, k) => gemm(k ? `h${k - 1}` : 'obs', 'w', null, k === count - 1 ? 'act' : `h${k}`, { name: `layer${k}` })),
        initializers: [W('w', 64, 64)], inputs: [['obs', ['batch', 64]]], outputs: [['act', ['batch', 64]]] });
    assert.equal(mlpFromOnnx(tied(4)).network.length, 4);
    refused(tied(40), /^mlpFromOnnx: initializer "w" is used so often that the layers hold more values than the file has bytes\.$/);
});

test('U31: files written by other exporters read as their makers hold them, or are refused as listed', () => {
    const document = JSON.parse(readFileSync(new URL('./onnx-fixtures/fixtures.json', import.meta.url), 'utf8'));
    assert.equal(document.format, 'null-graph/onnx-fixtures');
    // What the reader should do with each file is stated here, not stored with the file.
    const reads = ['helper-gemm0', 'sklearn-mlp', 'torch-acts', 'torch-chain', 'torch-flatten', 'torch-nobias', 'torch-shared', 'torch-vector',
        'torchscript-chain', 'torchscript-flatten', 'torchscript-nobias', 'torchscript-shared', 'torchscript-vector'];
    const refuses = {
        'torch-double': /^mlpFromOnnx: initializer "0\.weight" has data type 11; only float32 \(1\) is read\. Export the model as float32\.$/,
        'torch-heads': /^mlpFromOnnx: the graph has 1 inputs and 2 outputs; a dense network has one of each\.$/,
        'torch-outside': /^mlpFromOnnx: initializer "0\.weight" is stored outside the file, in "model\.onnx\.data"; export with the weights inside it \(PyTorch: external_data=False\)\.$/,
    };
    assert.deepEqual(document.cases.map(c => c.name).sort(), [...reads, ...Object.keys(refuses)].sort(), 'every fixture is listed here, and nothing else');
    let worst = 0;
    for (const fixture of document.cases) {
        const bytes = Uint8Array.from(Buffer.from(fixture.onnx, 'base64'));
        assert.equal(bytes.length, fixture.bytes, fixture.name);
        assert.equal(createHash('sha256').update(bytes).digest('hex'), fixture.sha256, `${fixture.name}: the bytes are not the ones that were recorded`);
        if (refuses[fixture.name]) {
            refused(bytes, refuses[fixture.name], fixture.name);
            continue;
        }
        const maker = fixture.network.map(l => ({ ...l, weights: Float32Array.from(l.weights), biases: Float32Array.from(l.biases) }));
        const { network } = mlpFromOnnx(bytes);
        same(network, maker, fixture.name);
        A.mlpLayout(network);                                    // and MlpBrain would run it
        fixture.inputs.forEach((input, row) => {
            const got = A.mlpForward(network, Float32Array.from(input)), want = fixture.outputs.maker[row];
            assert.equal(got.length, want.length, `${fixture.name}: outputs of row ${row}`);
            got.forEach((v, j) => {
                const apart = Math.abs(v - want[j]);
                worst = Math.max(worst, apart);
                assert.ok(apart <= 1e-5 + 1e-5 * Math.abs(want[j]), `${fixture.name}: row ${row}, output ${j}: ${v} against the maker's ${want[j]}`);
            });
        });
    }
    // Equal tensors that an exporter stored once come back as arrays of their own.
    const shared = mlpFromOnnx(Uint8Array.from(Buffer.from(document.cases.find(c => c.name === 'torch-shared').onnx, 'base64'))).network;
    assert.deepEqual(bits(shared[1].weights), bits(shared[2].weights));
    assert.notEqual(shared[1].weights, shared[2].weights);
    console.log(`U31: ${reads.length} files read with their makers' weights bit for bit, mlpForward within ${worst.toExponential(1)} of the makers' outputs; ${Object.keys(refuses).length} refused as listed`);
});

test('U14: every refusal fires and names the offender', () => {
    const good = chain(BASE.slice(0, 2));
    const one = (nodes, initializers = [W('w', 3, 5), B('b', 3)], rest = {}) => model({ nodes, initializers, ...rest });
    const cases = [
        ['a path instead of bytes', 'prey.onnx', /^mlpFromOnnx: expected the file's bytes, as a Uint8Array or an ArrayBuffer\.$/],
        ['a promise that was not awaited', Promise.resolve(good), /expected the file's bytes/],
        ['nothing', undefined, /expected the file's bytes/],
        ['a file cut short', good.subarray(0, good.length - 5), /^mlpFromOnnx: the file ends inside a field; it is cut short or is not ONNX\.$/],
        ['a web page', new TextEncoder().encode('<!DOCTYPE html><html><body>404 Not Found</body></html>'), /^mlpFromOnnx: the file's bytes are not protobuf; it is not ONNX\.$/],
        ['a .mlp.json file', new TextEncoder().encode('{\n "format": "null-graph/mlp"\n}'), /not protobuf/],
        ['a model without a graph', model({ graph: false }), /^mlpFromOnnx: the file has no graph; it is not an ONNX model\.$/],
        ['an empty file', new Uint8Array(0), /has no graph/],
        ['operator set 6', chain(BASE, { opsets: [['', 6]] }), /^mlpFromOnnx: the model's operator set is version 6; the reader takes version 7 or later\.$/],
        ['no operator set', chain(BASE, { opsets: [] }), /operator set is version 0/],
        ['only another operator set', chain(BASE, { opsets: [['ai.onnx.ml', 3]] }), /operator set is version 0/],
        ['two inputs', one([gemm('obs', 'w', 'b', 'act')], undefined, { inputs: [['obs', ['batch', 5]], ['mask', ['batch', 5]]] }),
            /^mlpFromOnnx: the graph has 2 inputs and 1 outputs; a dense network has one of each\.$/],
        ['no layers', one([], [], { outputs: [['obs', ['batch', 5]]] }), /^mlpFromOnnx: the graph has no layers\.$/],
        ['only a Cast', one([node('Cast', ['obs'], ['act'], { attributes: [attr('to', 1)] })], []), /the graph has no layers/],
        ['an operator of another set', one([node('Gelu', ['obs'], ['act'], { name: 'gelu', domain: 'com.microsoft' })]),
            /^mlpFromOnnx: node "gelu" is com\.microsoft\.Gelu; supported: Gemm, MatMul, Add, Relu, Tanh, Sigmoid\.$/],
        ['a node without a name', one([node('Softmax', ['obs'], ['act'])]), /^mlpFromOnnx: node "act" is Softmax; supported: /],
        ['an activation first', one([node('Relu', ['obs'], ['h'], { name: 'early' }), gemm('h', 'w', 'b', 'act')]),
            /^mlpFromOnnx: node "early" \(Relu\) is out of place: an Add follows a MatMul, and an activation follows a layer\.$/],
        ['two activations', one([gemm('obs', 'w', 'b', 'h'), node('Tanh', ['h'], ['t']), node('Relu', ['t'], ['act'], { name: 'again' })]), /node "again" \(Relu\) is out of place/],
        ['an Add after a Gemm', one([gemm('obs', 'w', 'b', 'h'), node('Add', ['h', 'b'], ['act'], { name: 'plus' })]), /node "plus" \(Add\) is out of place/],
        ['the data as second input', one([node('MatMul', ['w', 'obs'], ['act'], { name: 'backwards' })]),
            /^mlpFromOnnx: node "backwards" \(MatMul\) must read "obs" as its first input\.$/],
        ['a branch', one([gemm('obs', 'w', 'b', 'act', { name: 'policy' }), gemm('obs', 'w', 'b', 'value', { name: 'critic' })]),
            /^mlpFromOnnx: node "critic" \(Gemm\) also reads "obs"; a dense network is one chain of layers\.$/],
        ['a node off the chain', one([gemm('obs', 'w', 'b', 'act'), node('Tanh', ['b'], ['unused'], { name: 'idle' })]),
            /^mlpFromOnnx: node "idle" \(Tanh\) is not on the chain from the input to the output\.$/],
        ['alpha set', one([gemm('obs', 'w', 'b', 'act', { name: 'scaled', attributes: [attr('transB', 1), attr('alpha', 0.5, { float: true })] })]),
            /^mlpFromOnnx: node "scaled" \(Gemm\) has alpha, beta or transA set; only the defaults are read\.$/],
        ['beta set', one([gemm('obs', 'w', 'b', 'act', { name: 'scaled', attributes: [attr('transB', 1), attr('beta', 2, { float: true })] })]), /has alpha, beta or transA set/],
        ['alpha with its value left out, which is zero', one([gemm('obs', 'w', 'b', 'act', { attributes: [attr('transB', 1), attr('alpha', 0, { float: true, bare: true })] })]), /has alpha, beta or transA set/],
        ['transA set', one([gemm('obs', 'w', 'b', 'act', { attributes: [attr('transB', 1), attr('transA', 1)] })]), /has alpha, beta or transA set/],
        ['weights that are not in the file', one([gemm('obs', 'nope', 'b', 'act', { name: 'dense' })]),
            /^mlpFromOnnx: node "dense" \(Gemm\) needs "nope" to be a float32 initializer\.$/],
        ['a bias computed by a node', one([node('MatMul', ['obs', 'w'], ['z'], { name: 'mm' }), node('Add', ['z', 'obs2'], ['act'], { name: 'residual' })], [W('w', 5, 3)]),
            /^mlpFromOnnx: node "residual" \(Add\) needs "obs2" to be a float32 initializer\.$/],
        ['weights that are an Identity of nothing in the file', one([node('Identity', ['nope'], ['w2'], { name: 'same' }), gemm('obs', 'w2', 'b', 'act', { name: 'dense' })]),
            /^mlpFromOnnx: node "dense" \(Gemm\) needs "w2" to be a float32 initializer\.$/],
        ['an Identity that no layer reads', one([gemm('obs', 'w', 'b', 'act'), node('Identity', ['w'], ['w2'], { name: 'spare' })]),
            /^mlpFromOnnx: node "spare" \(Identity\) is not on the chain from the input to the output\.$/],
        ['float64 weights', one([gemm('obs', 'w', 'b', 'act')], [W('w', 3, 5, { type: 11 }), B('b', 3)]),
            /^mlpFromOnnx: initializer "w" has data type 11; only float32 \(1\) is read\. Export the model as float32\.$/],
        ['weights outside the file', one([gemm('obs', 'w', 'b', 'act')], [W('w', 3, 5, { external: 'weights.bin' }), B('b', 3)]),
            /^mlpFromOnnx: initializer "w" is stored outside the file, in "weights\.bin"; export with the weights inside it \(PyTorch: external_data=False\)\.$/],
        ['too few values', one([gemm('obs', 'w', 'b', 'act')], [tensor('w', [3, 5], new Array(14).fill(1)), B('b', 3)]), /^mlpFromOnnx: initializer "w" is \[3,5\] but holds 14 values\.$/],
        ['a float list that is too long', one([gemm('obs', 'w', 'b', 'act')], [tensor('w', [3, 5], new Array(16).fill(1), { as: 'list' }), B('b', 3)]), /initializer "w" is \[3,5\] but holds 16 values/],
        ['raw bytes that are not whole floats', one([gemm('obs', 'w', 'b', 'act')], [[...int(1, 1), ...int(1, 1), ...int(2, 1), ...msg(9, [1, 2, 3, 4, 5, 6]), ...msg(8, [119])], B('b', 1)]),
            /initializer "w" is \[1,1\] but holds 1\.5 values/],
        ['weights that are not a matrix', one([gemm('obs', 'w', 'b', 'act', { name: 'flat' })], [tensor('w', [15], new Array(15).fill(1)), B('b', 3)]),
            /^mlpFromOnnx: node "flat" \(Gemm\) has weights of shape \[15\]; a dense layer's are \[rows, columns\]\.$/],
        ['one bias for three outputs', one([gemm('obs', 'w', 'b', 'act', { name: 'dense' })], [W('w', 3, 5), B('b', 1)]),
            /^mlpFromOnnx: node "dense" \(Gemm\) has a bias of shape \[1\] for 3 outputs\.$/],
        ['a bias that is a column', one([gemm('obs', 'w', 'b', 'act')], [W('w', 3, 5), tensor('b', [3, 1], [1, 2, 3])]), /has a bias of shape \[3,1\] for 3 outputs/],
        ['an output the layers do not reach', one([gemm('obs', 'w', 'b', 'h')]), /^mlpFromOnnx: the layers end at "h", not at the graph's output "act"\.$/],
    ];
    for (const [label, bytes, pattern] of cases) refused(bytes, pattern, label);
    // The loop, the long varint and the tensor used too often are in U30; with them every message of §11 has fired.
    assert.equal(mlpFromOnnx(good).network.length, 2, 'and the file these were cut from reads');
});

test('size: the reader keeps its line, copies none of MlpBrain, and no other entry carries it', async () => {
    const bundle = async contents => (await build({ stdin: { contents, resolveDir: root, loader: 'ts' }, bundle: true, minify: true,
        format: 'esm', platform: 'browser', external: ['gl-matrix'], write: false, logLevel: 'silent' })).outputFiles[0];
    const reader = await bundle(`export { mlpFromOnnx } from '${root}src/extras/inference/index.ts';`), gzip = gzipSync(reader.contents).length;
    assert.ok(reader.text.includes('mlpFromOnnx: '), 'the bundle is the reader');
    assert.ok(gzip <= 2600, `the reader is ${gzip} B min+gzip; its line is 2,600 B (the design's §12)`);
    // Types only: MlpBrain validates what it runs, and none of that is copied here.
    assert.ok(!/affine layers|MlpBrain/.test(reader.text), 'the reader holds agents code');
    const agents = await bundle(`export * from '${root}src/extras/agents/index.ts';`);
    assert.ok(!agents.text.includes('mlpFromOnnx'), 'the agents entry holds the reader');
    // An app that imports the entry and uses nothing of it carries nothing.
    const unused = await bundle(`import '${root}src/extras/inference/index.ts';`);
    assert.equal(unused.text.trim(), '', `an unused import leaves ${unused.contents.length} B behind`);
    console.log(`size: mlpFromOnnx is ${reader.contents.length} B min, ${gzip} B min+gzip, of 2,600 B; the agents entry is ${gzipSync(agents.contents).length} B min+gzip and holds none of it`);
});
