# Inference — networks from files

Import from `null-graph/inference`. The entry is for bringing a network that was trained elsewhere
into an app. Today it holds one function, `mlpFromOnnx`, which reads a small dense network from
the bytes of an `.onnx` file, for [`MlpBrain`](./agents.md#network-brains-mlpbrain) to run. Design:
[`agents-onnx-reader-design.md`](../agents-onnx-reader-design.md).

- **It does not run ONNX Runtime, and adds no dependency.** The file is read once, on the CPU, by
  this library. The network then runs on the GPU as `MlpBrain`'s think stage, like any other.
- **Not built yet:** running other kinds of model through ONNX Runtime, which this entry is
  planned to hold as well.

## Load a network

```ts
import type { NullGraph } from 'null-graph';
import { MlpBrain, type AgentSystem } from 'null-graph/agents';
import { mlpFromOnnx } from 'null-graph/inference';

/** Puts the dense network of an .onnx file on one species. The caller owns the returned brain. */
export function useOnnxNetwork(engine: NullGraph, agents: AgentSystem, bytes: ArrayBuffer | Uint8Array, species: string) {
    const { network, metadata } = mlpFromOnnx(bytes);   // throws, naming the node, when it is not a dense network
    // A file written by this repository's exporter carries the names it was trained with. Equal
    // widths cannot show that an input means what the network expects; equal names do.
    const { obsNames, actNames } = agents.definition;
    for (const [key, names] of [['obsNames', obsNames], ['actNames', actNames]] as const) {
        if (metadata[key] !== undefined && JSON.parse(metadata[key]).join() !== names.join()) {
            throw new Error(`The file was trained on other ${key === 'obsNames' ? 'observations' : 'actions'} than "${species}" has.`);
        }
    }
    const brain = new MlpBrain(engine, network);         // refuses more layers, or wider ones, than it runs
    try {
        agents.setBrain(brain, { species });             // refuses widths that are not the definition's
    } catch (error) {
        brain.destroy();
        throw error;
    }
    return brain;
}
```

Call it with the file's bytes, for example
`useOnnxNetwork(engine, agents, await (await fetch('prey.onnx')).arrayBuffer(), 'prey')`. Destroy
the system, or give the species another brain, before destroying the returned brain.

- **`mlpFromOnnx(bytes)`** takes a `Uint8Array` or an `ArrayBuffer` and returns
  `{ network, input, output, metadata }`:
  - `network` is an `MlpNetwork`. It also feeds `MlpBrain.bySpecies`, `brain.setWeights` and
    `mlpForward`.
  - `input` and `output` are the graph's names for its input and its output.
  - `metadata` is what the file says about itself, as text; it is empty for most files.
- **The weights come back bit for bit,** in arrays of their own. The bytes may be dropped after
  the call.
- **A file that is not a dense network is refused** with an error that names the node or the
  tensor and says what to do ([below](#when-a-file-is-refused)). Nothing is approximated.
- **`MlpBrain`'s limits are `MlpBrain`'s.** The reader takes a network of any depth and width;
  `new MlpBrain` refuses more than 4 layers, an input wider than 128, a hidden layer wider than
  64, an output wider than 16, and a weight that is not finite.

## What a file must hold

- **One chain of dense layers** from one input to one output. A layer is `Gemm`, or `MatMul` with
  an optional `Add` for the bias, followed by at most one of `Relu`, `Tanh`, `Sigmoid`.
- **Weights and biases stored in the file as float32.**
- **These are passed over, because they change no value:** a `Cast` to float32 or a `Flatten`
  before the first layer, a `Reshape` after the last, and an `Identity` on a layer's weights or
  bias. scikit-learn and PyTorch's older exporter write them.
- **Not read:** convolutions, recurrent layers, normalization layers, residual connections, and
  other activations (`LeakyRelu`, `Elu`, `Gelu`, `Clip`, `Softmax`); a graph with several inputs
  or outputs; float64 or float16 weights; weights stored in a second file.

## Exporting from PyTorch

```python
import torch
from torch import nn

policy = nn.Sequential(nn.Linear(24, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2), nn.Tanh()).eval()
example = torch.zeros(1, 24)                       # one row of observations: [1, obsDim]
torch.onnx.export(policy, (example,), "policy.onnx", external_data=False)
```

- **`external_data=False` is needed.** Without it, PyTorch stores every weight tensor larger
  than 256 bytes in a second file, `policy.onnx.data`, and `policy.onnx` alone holds the graph and
  no weights. The reader refuses such a file and names the argument. Measured with PyTorch 2.14.1:
  this 24→64→64→2 network is one file of 30,794 B with the argument, and two files of 7,927 B and
  23,040 B without it.
- **Export the policy alone.** A model that also estimates a value has two outputs, and the
  reader takes one. Export the part that maps observations to actions: for a model with a `pi`
  and a `vf` part, `torch.onnx.export(model.pi, (example,), "policy.onnx", external_data=False)`.
- **Keep it float32.** A model converted with `.double()` is refused.
- **The example's batch size does not matter.** The reader takes its shapes from the weights.
- **Your file paths are in the file.** PyTorch's default exporter writes a stack trace into every
  node, with the paths of your script and of your Python installation. The reader skips them.
  Before publishing a model, remove them:

  ```python
  import onnx

  model = onnx.load("policy.onnx")
  for node in model.graph.node:
      del node.metadata_props[:]
  onnx.save(model, "policy.onnx")
  ```

- **The older exporter** (`dynamo=False`) writes one file with no paths in it, and is read as
  well.

## Exporting from scikit-learn

```python
import numpy
from skl2onnx import to_onnx

with open("policy.onnx", "wb") as file:
    file.write(to_onnx(regressor, X[:1].astype(numpy.float32)).SerializeToString())
```

- **An `MLPRegressor` is read,** with any of its activations (`identity`, `logistic`, `tanh`,
  `relu`) and any number of outputs.
- **An `MLPClassifier` is not:** its file ends in operators that pick a class, and is refused.

## From this repository's trainer

`npm run agents:export` writes each trained Herd & Wolves network as a 41 KB `.onnx` file, with
the observation and action names it was trained with in the file's metadata
([`tools/agents/README.md`](../../tools/agents/README.md)). `mlpFromOnnx` reads such a file back
to the committed network bit for bit, and the example above checks its names.

## When a file is refused

Every refusal is an `Error` whose message starts with `mlpFromOnnx:`. It is thrown when the file is
loaded, never during a frame. The common ones:

| The message says | What to do |
|---|---|
| `initializer "…" is stored outside the file, in "….onnx.data"` | Export again with the weights inside the file. In PyTorch: `external_data=False` |
| `the graph has 1 inputs and 2 outputs` | Export the policy alone, not the policy with its value estimate |
| `node "…" is <Op>; supported: Gemm, MatMul, Add, Relu, Tanh, Sigmoid` | The model is not a plain dense network, or uses an activation `MlpBrain` does not have. Retrain with `tanh`, `relu` or `sigmoid`, or wait for ONNX Runtime support |
| `initializer "…" has data type 11; only float32 (1) is read` | Convert the model to float32 before exporting |
| `the file's bytes are not protobuf; it is not ONNX` | The bytes are something else: often a web page returned for a wrong address. Check the response before reading it |
| `the file ends inside a field; it is cut short or is not ONNX` | The download was cut short |
| `MlpBrain: expected 1–4 affine layers` or `MlpBrain: layer … must be 1–64` (from `new MlpBrain`) | The network is deeper or wider than the built-in kernel runs |
| `AgentSystem: the brain maps 5→3; the layout is 24→2` (from `setBrain`) | The network was trained for other observations or actions than this definition has |

## What nothing checks

- **What each input means.** A file from another tool carries no names. The reader cannot tell
  whether the network's third input is this definition's third observation. Train on the
  observations in `definition.obsNames`, in that order.
- **Scaling.** If the network was trained on normalized observations, the app must normalize them
  the same way; no file says so.
- **Behaviour.** The reader preserves the weights. `MlpBrain` then agrees with the CPU reference
  `mlpForward` within `1e-5 + 1e-5·|cpu|` for weights of unit scale
  ([agents.md](./agents.md#network-brains-mlpbrain)).

## Cost

- **Size:** `mlpFromOnnx` is 2.5 KB min+gzip. An app that does not import it carries none of it.
- **Time:** reading a 41 KB file takes about 2 ms.
- **Safety with files you did not make:** a damaged or hostile file is refused or read; it cannot
  hang the reader, and the reader never holds more than a few times the file's size.

## Current validation

- **Headless** (`npm run test:inference`): the reader gives back, bit for bit, what this
  repository's writer wrote, and one network written in 23 different ways. 13 files written by
  PyTorch 2.14.1 (both exporters), scikit-learn 1.7.2 and `onnx.helper` read with their makers'
  weights bit for bit; 3 are refused as they should be. 10,000 damaged files and every cut-short
  prefix of a file give a network or a named refusal, nothing else. This guide's example is
  type-checked, bundled and run on the committed networks.
- **On the GPU** (the `onnx` suite): for 17 files, the network read by `mlpFromOnnx` into an
  `MlpBrain` and ONNX Runtime running the same bytes give the same actions within 4.5e-6, on
  Node/Dawn (an NVIDIA and an AMD GPU) and in Chrome.
- **Not measured:** files from TensorFlow or Keras, and from reinforcement-learning libraries;
  export the policy as a plain PyTorch module.
