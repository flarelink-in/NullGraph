"""Step 8, phase 8.0 (docs/agents-onnx-reader-design.md, section 9): dense networks written as
ONNX by exporters this repository did not write, with what each maker computed from them.

    venv/Scripts/python test/onnx-fixtures/make.py          rewrites fixtures.json beside this file
    venv/Scripts/python test/onnx-fixtures/make.py --list   also prints what each file holds

The tests read fixtures.json and need no Python; only regenerating does (requirements.txt).
Every network has seeded random weights. None is trained.

For each case fixtures.json records:
  onnx      the file's bytes, as base64 (text, so git cannot alter them), and their SHA-256
  holds     what the file holds, listed with the onnx package: versions, nodes, initializers
  network   the layers as the maker holds them: weights row-major [output][input], as f32
  inputs    four vectors; outputs.maker is what the maker itself computed from them
  judges    onnxruntime (CPU) and onnx's reference evaluator on the same bytes, and how far the
            three are apart
"""
import base64
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import warnings

sys.stdout.reconfigure(encoding='utf-8')   # torch.onnx prints a check mark; cp1252 cannot, and the export dies
sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import onnx
import onnx.reference
import onnxruntime
import sklearn
import skl2onnx
import torch
from onnx import TensorProto, helper
from sklearn.neural_network import MLPRegressor
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 8
ACTIVATIONS = {nn.Tanh: 'tanh', nn.ReLU: 'relu', nn.Sigmoid: 'sigmoid'}
STORAGE_FIELDS = ('raw_data', 'float_data', 'double_data', 'int32_data', 'int64_data')


def f32_list(array):
    """f32 values as JSON numbers that read back to the same f32: 9 significant digits."""
    flat = np.asarray(array, dtype=np.float32).ravel()
    out = [float('%.9g' % v) for v in flat]
    assert np.array_equal(np.asarray(out, dtype=np.float32).view(np.uint32), flat.view(np.uint32))
    return out


def rows(array):
    array = np.asarray(array)
    return [f32_list(r) if array.dtype == np.float32 else [float(v) for v in r] for r in array]


def inputs_for(width):
    """Four vectors: zeros, ordinary, small, and large enough to saturate tanh."""
    rng = np.random.default_rng(SEED + width)
    return np.stack([np.zeros(width), rng.normal(size=width), 0.1 * rng.normal(size=width),
                     8 * rng.normal(size=width)]).astype(np.float32)


def layers_of(modules):
    """nn.Linear and activation modules, in order, as MlpNetwork layers."""
    layers = []
    for module in modules:
        if isinstance(module, nn.Linear):
            weight = module.weight.detach().numpy()
            bias = module.bias.detach().numpy() if module.bias is not None else np.zeros(module.out_features)
            layers.append({'input': module.in_features, 'output': module.out_features, 'activation': 'identity',
                           'weights': f32_list(weight), 'biases': f32_list(bias)})
        elif type(module) in ACTIVATIONS:
            layers[-1]['activation'] = ACTIVATIONS[type(module)]
    return layers


def attribute_value(a):
    value = helper.get_attribute_value(a)
    if isinstance(value, bytes):
        return value.decode('utf-8', 'replace')
    if isinstance(value, (int, float, str)):
        return value
    return f'<{onnx.AttributeProto.AttributeType.Name(a.type)}>'


def dims_of(value_info):
    shape = value_info.type.tensor_type.shape
    # A dimension is a number, a name, or left open (None).
    return [getattr(d, d.WhichOneof('value')) if d.WhichOneof('value') else None for d in shape.dim]


def holds(model):
    """What a file holds, read with the onnx package."""
    graph, names = model.graph, {t.name for t in model.graph.initializer}
    def storage(t):
        if t.data_location == TensorProto.EXTERNAL:
            return 'outside the file'
        used = [f for f in STORAGE_FIELDS if len(getattr(t, f))]
        return used[0] if used else 'empty'
    return {
        'irVersion': model.ir_version,
        'opsets': {o.domain: o.version for o in model.opset_import},
        'producer': f'{model.producer_name} {model.producer_version}'.strip(),
        'nodes': [{'op': n.op_type, **({'domain': n.domain} if n.domain else {}), 'name': n.name,
                   'inputs': list(n.input), 'outputs': list(n.output),
                   'attributes': {a.name: attribute_value(a) for a in n.attribute}} for n in graph.node],
        'initializers': [{'name': t.name, 'dims': list(t.dims), 'type': TensorProto.DataType.Name(t.data_type),
                          'storage': storage(t), **({'external': {e.key: e.value for e in t.external_data}} if t.external_data else {})}
                         for t in graph.initializer],
        'inputs': [{'name': v.name, 'type': TensorProto.DataType.Name(v.type.tensor_type.elem_type), 'dims': dims_of(v),
                    **({'initializer': True} if v.name in names else {})} for v in graph.input],
        'outputs': [{'name': v.name, 'type': TensorProto.DataType.Name(v.type.tensor_type.elem_type), 'dims': dims_of(v)}
                    for v in graph.output],
        'nodeMetadataKeys': sorted({p.key for n in graph.node for p in n.metadata_props}),
        'modelMetadata': {p.key: p.value for p in model.metadata_props},
    }


def without_local_paths(model):
    """Drops the entries that hold this machine's file paths. Returns how many were dropped."""
    dropped = 0
    for node in model.graph.node:
        entries = [(p.key, p.value) for p in node.metadata_props]
        kept = [(key, value) for key, value in entries if 'stack_trace' not in key]
        if len(kept) != len(entries):
            dropped += len(entries) - len(kept)
            del node.metadata_props[:]
            for key, value in kept:
                node.metadata_props.add(key=key, value=value)
        if node.doc_string:
            node.doc_string, dropped = '', dropped + 1
    return dropped


def texts(model):
    """Every free-text field of a model: where a path could hide."""
    yield model.doc_string
    yield model.graph.doc_string
    for p in model.metadata_props:
        yield p.value
    for node in model.graph.node:
        yield node.doc_string
        for p in node.metadata_props:
            yield p.value


def judge(data, inputs, feeding):
    """The same bytes through onnxruntime and onnx's reference evaluator. First output only."""
    session = onnxruntime.InferenceSession(data, providers=['CPUExecutionProvider'])
    name = session.get_inputs()[0].name
    reference = onnx.reference.ReferenceEvaluator(onnx.load_from_string(data))
    # 'row': one [1, n] row at a time (a file with a fixed batch of 1); 'vector': one [n] vector; None: all rows.
    feeds = [inputs[k] if feeding == 'vector' else inputs[k:k + 1] for k in range(len(inputs))] if feeding else [inputs]
    def run(fn):
        outputs = [np.asarray(fn(feed)) for feed in feeds]
        return np.concatenate([o.reshape(-1, o.shape[-1]) for o in outputs])
    return (run(lambda feed: session.run(None, {name: feed})[0]),
            run(lambda feed: reference.run(None, {name: feed})[0]))


def torch_export(module, example, **options):
    """torch.onnx.export to a file, as a user would call it. Returns the bytes and the files written."""
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, 'model.onnx')
        torch.onnx.export(module, (example,), path, **options)
        files = {name: os.path.getsize(os.path.join(folder, name)) for name in sorted(os.listdir(folder))}
        with open(path, 'rb') as file:
            return file.read(), files


class Heads(nn.Module):
    """The shape of an actor-critic export: one observation, a policy head and a value head."""
    def __init__(self):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 3))
        self.vf = nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 1))

    def forward(self, obs):
        return self.pi(obs), self.vf(obs)


def torch_cases():
    """(name, exporter, how it was called, module, the modules that make the network, example input, options, feeding)"""
    def seq(*modules):
        return nn.Sequential(*modules).eval()
    chain = lambda: seq(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 3))
    cases = []
    def add(name, module, example, feeding, call, layers=None, **options):
        cases.append((name, module, layers if layers is not None else list(module), example, feeding, call, options))
    batch = {'dynamic_shapes': ({0: torch.export.Dim('batch')},)}
    add('torch-chain', chain(), torch.zeros(1, 5), 'row', 'torch.onnx.export(model, (zeros(1, 5),), path)')
    add('torch-acts', seq(nn.Linear(5, 7), nn.ReLU(), nn.Linear(7, 4), nn.Sigmoid(), nn.Linear(4, 3), nn.Tanh()),
        torch.zeros(2, 5), None, 'torch.onnx.export(model, (zeros(2, 5),), path, dynamic_shapes=({0: Dim("batch")},))', **batch)
    add('torch-nobias', seq(nn.Linear(5, 3, bias=False)), torch.zeros(1, 5), 'row', 'torch.onnx.export(model, (zeros(1, 5),), path)')
    add('torch-vector', chain(), torch.zeros(5), 'vector', 'torch.onnx.export(model, (zeros(5),), path)')
    flat = seq(nn.Flatten(), nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 3))
    add('torch-flatten', flat, torch.zeros(1, 5), 'row', 'torch.onnx.export(Sequential(Flatten(), ...), (zeros(1, 5),), path)')
    add('torch-double', seq(nn.Linear(5, 3)).double(), torch.zeros(1, 5, dtype=torch.float64), 'row',
        'torch.onnx.export(model.double(), (zeros(1, 5, dtype=float64),), path)')
    heads = Heads().eval()
    add('torch-heads', heads, torch.zeros(1, 5), 'row', 'torch.onnx.export(policy_and_value, (zeros(1, 5),), path)', layers=list(heads.pi))
    # 65 weights: one more than the default call keeps inside the file (see measuredOnly).
    add('torch-outside', seq(nn.Linear(13, 5)), torch.zeros(1, 13), 'row', 'torch.onnx.export(Linear(13, 5), (zeros(1, 13),), path)')
    old = {'dynamo': False, 'input_names': ['obs'], 'output_names': ['act']}
    add('torchscript-chain', chain(), torch.zeros(1, 5), None,
        'torch.onnx.export(model, (zeros(1, 5),), path, dynamo=False, input_names=["obs"], output_names=["act"], dynamic_axes={"obs": {0: "batch"}})',
        dynamic_axes={'obs': {0: 'batch'}}, **old)
    add('torchscript-vector', chain(), torch.zeros(5), 'vector',
        'torch.onnx.export(model, (zeros(5),), path, dynamo=False, input_names=["obs"], output_names=["act"])', **old)
    add('torchscript-nobias', seq(nn.Linear(5, 3, bias=False)), torch.zeros(1, 5), 'row',
        'torch.onnx.export(model, (zeros(1, 5),), path, dynamo=False, input_names=["obs"], output_names=["act"])', **old)
    add('torchscript-flatten', seq(nn.Flatten(), nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 3)), torch.zeros(1, 5), None,
        'torch.onnx.export(Sequential(Flatten(), ...), (zeros(1, 5),), path, dynamo=False, input_names=["obs"], output_names=["act"], dynamic_axes={"obs": {0: "batch"}})',
        dynamic_axes={'obs': {0: 'batch'}}, **old)

    # Tensors that hold the same values: every bias zero (as many reinforcement-learning libraries
    # start), and two equal 7x7 weight matrices. Each exporter stores such tensors once.
    def shared():
        net = seq(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 7), nn.Tanh(), nn.Linear(7, 7), nn.Tanh(), nn.Linear(7, 3))
        with torch.no_grad():
            for layer in net:
                if isinstance(layer, nn.Linear):
                    layer.bias.zero_()
            net[4].weight.copy_(net[2].weight)
        return net
    same = 'a 5-7-7-7-3 tanh network with zero biases and two equal 7x7 weight matrices'
    add('torch-shared', shared(), torch.zeros(1, 5), 'row', f'torch.onnx.export({same}, (zeros(1, 5),), path)')
    add('torchscript-shared', shared(), torch.zeros(1, 5), 'row',
        f'torch.onnx.export({same}, (zeros(1, 5),), path, dynamo=False, input_names=["obs"], output_names=["act"])', **old)
    return cases


def sklearn_case():
    rng = np.random.default_rng(SEED)
    x, y = rng.normal(size=(64, 5)), rng.normal(size=(64, 3))
    model = MLPRegressor(hidden_layer_sizes=(7,), activation='tanh', max_iter=20, random_state=SEED).fit(x, y)
    proto = skl2onnx.to_onnx(model, x[:1].astype(np.float32))
    layers = [{'input': w.shape[0], 'output': w.shape[1], 'activation': act,
               'weights': f32_list(w.T.astype(np.float32)), 'biases': f32_list(b.astype(np.float32))}
              for w, b, act in zip(model.coefs_, model.intercepts_, ['tanh', 'identity'])]
    return ('sklearn-mlp', f'skl2onnx {skl2onnx.__version__} (scikit-learn {sklearn.__version__})',
            'skl2onnx.to_onnx(MLPRegressor(hidden_layer_sizes=(7,), activation="tanh"), X[:1].astype(float32))',
            proto.SerializeToString(), layers, lambda inputs: model.predict(inputs.astype(np.float64)))


def helper_case():
    """Built by hand with onnx.helper: transB left at its default of 0, a float list, an older opset."""
    rng = np.random.default_rng(SEED + 1)
    w0, b0 = rng.normal(size=(5, 7)).astype(np.float32), rng.normal(size=7).astype(np.float32)   # [input, output]
    w1 = rng.normal(size=(3, 7)).astype(np.float32)                                              # [output, input], no bias
    tensor = lambda name, a: helper.make_tensor(name, TensorProto.FLOAT, a.shape, [float(v) for v in a.ravel()])
    graph = helper.make_graph(
        [helper.make_node('Gemm', ['x', 'w0', 'b0'], ['z0'], name='first'),
         helper.make_node('Tanh', ['z0'], ['h0'], name='squash'),
         helper.make_node('Gemm', ['h0', 'w1'], ['y'], name='second', transB=1)],
        'by-hand', [helper.make_tensor_value_info('x', TensorProto.FLOAT, ['n', 5])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, ['n', 3])],
        [tensor('w0', w0), tensor('b0', b0), tensor('w1', w1)])
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)], producer_name='onnx.helper', ir_version=6)
    layers = [{'input': 5, 'output': 7, 'activation': 'tanh', 'weights': f32_list(w0.T), 'biases': f32_list(b0)},
              {'input': 7, 'output': 3, 'activation': 'identity', 'weights': f32_list(w1), 'biases': f32_list(np.zeros(3))}]
    return ('helper-gemm0', f'onnx.helper {onnx.__version__}',
            'helper.make_model: Gemm without transB (so 0) on float_data, Tanh, Gemm(transB=1) without a bias; opset 11, IR 6',
            proto.SerializeToString(), layers, lambda inputs: np.tanh(inputs @ w0 + b0) @ w1.T)


def record(name, exporter, call, data, layers, maker_outputs, inputs, feeding, files=None):
    model = onnx.load_from_string(data)
    dropped = without_local_paths(model)
    if dropped:
        data = model.SerializeToString()
    for text in texts(model):
        assert ':\\' not in text and ':/' not in text and 'site-packages' not in text, f'{name}: a local path is left: {text[:120]!r}'
    for trace in ('site-packages', os.path.basename(os.path.expanduser('~'))):
        assert trace.encode() not in data, f'{name}: the file still holds {trace!r}'
    try:
        onnx.checker.check_model(model)
        checked = 'accepted'
    except Exception as error:
        checked = f'refused: {str(error).splitlines()[0]}'
    case = {
        'name': name, 'exporter': exporter, 'call': call,
        'onnx': base64.b64encode(data).decode('ascii'), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
        **({'edited': f'removed {dropped} node metadata entries that held this machine\'s file paths (stack traces)'} if dropped else {}),
        **({'filesWritten': files} if files else {}),
        'holds': holds(model), 'checker': checked,
        'network': layers, 'inputs': rows(inputs), 'outputs': {'maker': rows(maker_outputs)},
    }
    if any(t.data_location == TensorProto.EXTERNAL for t in model.graph.initializer):
        case['judges'] = {'skipped': 'weights are in model.onnx.data, which is not kept: the file alone cannot be run'}
        return case
    try:
        runtime, reference = judge(data, inputs.astype(np.float64) if model.graph.input[0].type.tensor_type.elem_type == TensorProto.DOUBLE else inputs, feeding)
        maker = np.asarray(maker_outputs, dtype=np.float64)
        # A reshape never reorders values: skl2onnx's regressor ends in one, to [-1, 1].
        runtime, reference = runtime.reshape(maker.shape), reference.reshape(maker.shape)
        case['outputs'].update(onnxruntime=rows(runtime), reference=rows(reference))
        case['judges'] = {'onnxruntime': onnxruntime.__version__,
                          'makerAgainstOnnxruntime': float(np.abs(maker - runtime).max()),
                          'makerAgainstReference': float(np.abs(maker - reference).max()),
                          'onnxruntimeAgainstReference': float(np.abs(runtime.astype(np.float64) - reference).max())}
    except Exception as error:
        case['judges'] = {'failed': f'{type(error).__name__}: {str(error).splitlines()[0][:200]}'}
    return case


def main():
    torch.manual_seed(SEED)
    cases, measured_only = [], []
    for name, module, layer_modules, example, feeding, call, options in torch_cases():
        kind = 'TorchScript exporter' if options.get('dynamo') is False else 'default exporter'
        inputs = inputs_for(example.shape[-1])
        with torch.no_grad():
            feed = torch.from_numpy(inputs).to(example.dtype)
            result = module(feed)
            maker = (result[0] if isinstance(result, tuple) else result).numpy()
        try:
            data, files = torch_export(module, example, **options)
        except Exception as error:
            measured_only.append({'name': name, 'exporter': f'PyTorch {torch.__version__}, {kind}', 'call': call,
                                  'failed': f'{type(error).__name__}: {str(error).splitlines()[0][:300]}'})
            continue
        cases.append(record(name, f'PyTorch {torch.__version__}, {kind}', call, data, layers_of(layer_modules), maker, inputs, feeding, files))

    # A network of the Herd & Wolves size, measured for where its weights are stored. Not kept.
    def big():
        return nn.Sequential(nn.Linear(24, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2), nn.Tanh()).eval()
    def stored(data, files):
        h = holds(onnx.load_from_string(data))
        return {'filesWritten': files, 'ops': ' '.join(n['op'] for n in h['nodes']),
                'initializers': ', '.join(f"{t['name']} {t['dims']} {t['storage']}" for t in h['initializers'])}
    exporter = f'PyTorch {torch.__version__}'
    for label, call, options in (
            ('torch-big', 'torch.onnx.export(model, (zeros(1, 24),), path)', {}),
            ('torch-big, external_data=False', 'torch.onnx.export(model, (zeros(1, 24),), path, external_data=False)', {'external_data': False}),
            ('torch-big, dynamo=False', 'torch.onnx.export(model, (zeros(1, 24),), path, dynamo=False)', {'dynamo': False})):
        measured_only.append({'name': label, 'exporter': exporter, 'network': '24->64->64->64->2 tanh', 'call': call,
                              **stored(*torch_export(big(), torch.zeros(1, 24), **options))})
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, 'model.onnx')
        torch.onnx.export(big(), (torch.zeros(1, 24),)).save(path)
        files = {name: os.path.getsize(os.path.join(folder, name)) for name in sorted(os.listdir(folder))}
        with open(path, 'rb') as file:
            measured_only.append({'name': 'torch-big, program.save', 'exporter': exporter, 'network': '24->64->64->64->2 tanh',
                                  'call': 'torch.onnx.export(model, (zeros(1, 24),)).save(path)', **stored(file.read(), files)})
    # Where the default call draws the line: one Linear(n, 1), so one weight tensor of n floats.
    line = {}
    for n in (63, 64, 65, 66):
        data, files = torch_export(nn.Linear(n, 1).eval(), torch.zeros(1, n))
        line[f'{n} floats ({n * 4} B)'] = holds(onnx.load_from_string(data))['initializers'][0]['storage']
    measured_only.append({'name': 'where the default call puts a weight tensor', 'exporter': exporter,
                          'call': 'torch.onnx.export(Linear(n, 1), (zeros(1, n),), path)', 'storage': line})

    for name, exporter, call, data, layers, maker in (sklearn_case(), helper_case()):
        inputs = inputs_for(5)
        cases.append(record(name, exporter, call, data, layers, maker(inputs), inputs, None))
    cases.sort(key=lambda case: case['name'])

    made = {'python': sys.version.split()[0],
            'packages': {m.__name__: m.__version__ for m in (torch, onnx, onnxruntime, sklearn, skl2onnx, np)}}
    try:
        import onnxscript
        made['packages']['onnxscript'] = onnxscript.__version__
    except ImportError:
        pass
    document = {'format': 'null-graph/onnx-fixtures', 'version': 1,
                'about': 'Written by test/onnx-fixtures/make.py. Do not edit by hand.',
                'made': made, 'cases': cases, 'measuredOnly': measured_only}
    path = os.path.join(HERE, 'fixtures.json')
    # One line per array of plain values: a layer's weights are one line, not one line per number.
    text = re.sub(r'\[\s*([^\[\]{}]*?)\s*\]', lambda m: '[' + ' '.join(m.group(1).split()) + ']', json.dumps(document, indent=1))
    assert json.loads(text) == document, 'putting arrays on one line changed the document'
    with io.open(path, 'w', encoding='utf-8', newline='\n') as file:
        file.write(text + '\n')
    print(f'\n{len(cases)} cases, {os.path.getsize(path):,} B: {path}')
    for case in cases:
        h, j = case['holds'], case['judges']
        ops = ' '.join(n['op'] for n in h['nodes'])
        stores = sorted({t['storage'] for t in h['initializers']})
        apart = j.get('failed') or j.get('skipped') or f"judges apart by at most {max(j['makerAgainstOnnxruntime'], j['makerAgainstReference']):.1e}"
        print(f"  {case['name']:19s} {case['bytes']:6,} B  IR {h['irVersion']}, opset {h['opsets'].get('', '?')}  [{ops}]  weights: {', '.join(stores)}  checker {case['checker'].split(':')[0]}  {apart}")
        if '--list' in sys.argv:
            for n in h['nodes']:
                print(f"      {n['op']}({', '.join(n['inputs'])}) -> {', '.join(n['outputs'])}  {n['attributes'] or ''}")
            for t in h['initializers']:
                print(f"      {t['name']} {t['dims']} {t['type']} as {t['storage']}")
            print(f"      inputs {h['inputs']}  outputs {h['outputs']}  files {case.get('filesWritten', '')}")
    for item in measured_only:
        print(f"  {item['name']:19s} (not kept) {item.get('failed') or item}")


if __name__ == '__main__':
    main()
