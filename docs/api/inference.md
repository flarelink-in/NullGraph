# Inference — networks from files, and ONNX Runtime

Import from `null-graph/inference`. The entry is for running, in an app, a network that was
trained elsewhere. It does that in two ways:

- **`mlpFromOnnx`** reads a small dense network from the bytes of an `.onnx` file, for
  [`MlpBrain`](./agents.md#network-brains-mlpbrain) to run. The file is read once, on the CPU, by
  this library, and the network then runs on the GPU as `MlpBrain`'s think stage, like any other.
  No ONNX Runtime, and no dependency. Design:
  [`agents-onnx-reader-design.md`](../agents-onnx-reader-design.md).
- **`OrtRuntime`, `OrtModel` and `OrtBrain`** run a model through ONNX Runtime Web's WebGPU
  provider, over GPU buffers ([below](#onnx-runtime)). `OrtBrain` makes such a model the brain of
  one or more species. ONNX Runtime is the app's dependency: an optional peer of this package,
  which the app installs and imports. Design: [`agents-ort-design.md`](../agents-ort-design.md).

**Use `mlpFromOnnx` and `MlpBrain` when the network fits them.** ONNX Runtime costs 3× to 10× the
built-in kernel's GPU time for the same network, a 6.4 MB download, and up to 1.6 s to start
([cost](#what-onnx-runtime-costs)). It is for the networks `MlpBrain` cannot run: deeper or wider,
or with other operators.

## Load a network

```ts
import type { NullGraph } from 'null-graph';
import { MlpBrain, type AgentSystem } from 'null-graph/agents';
import { mlpFromOnnx, type OnnxExternalData } from 'null-graph/inference';

/**
 * Puts the dense network of an .onnx file on one species. The caller owns the returned brain.
 * `externalData` holds the second file PyTorch writes beside the .onnx file by default.
 */
export function useOnnxNetwork(engine: NullGraph, agents: AgentSystem, bytes: ArrayBuffer | Uint8Array, species: string,
    externalData: OnnxExternalData[] = []) {
    const { network, metadata } = mlpFromOnnx(bytes, { externalData });   // throws, naming the node, when it is not a dense network
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

- **`mlpFromOnnx(bytes, { externalData })`** takes a `Uint8Array` or an `ArrayBuffer`, and, when
  the exporter stored weights in a second file, that file: `externalData: [{ path, data }]`, with
  `path` as the `.onnx` file names it (`policy.onnx.data`) and `data` its bytes. It is the list
  ONNX Runtime takes. It returns `{ network, input, output, metadata }`:
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
- **Weights and biases as float32,** stored in the file, or in a second file given as
  `externalData`. A second file is found by its name exactly as the `.onnx` file writes it; the
  reader opens and fetches nothing itself.
- **These are passed over, because they change no value:** a `Cast` to float32 or a `Flatten`
  before the first layer, a `Reshape` after the last, and an `Identity` on a layer's weights or
  bias. scikit-learn and PyTorch's older exporter write them.
- **Not read:** convolutions, recurrent layers, normalization layers, residual connections, and
  other activations (`LeakyRelu`, `Elu`, `Gelu`, `Clip`, `Softmax`); a graph with several inputs
  or outputs; float64 or float16 weights; weights in a second file that is not given.

## Exporting from PyTorch

```python
import torch
from torch import nn

policy = nn.Sequential(nn.Linear(24, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2), nn.Tanh()).eval()
example = torch.zeros(1, 24)                       # one row of observations: [1, obsDim]
torch.onnx.export(policy, (example,), "policy.onnx", external_data=False)
```

- **One file with `external_data=False`, or two files without it.** Without the argument,
  PyTorch stores every weight tensor larger than 256 bytes in a second file, `policy.onnx.data`,
  and `policy.onnx` alone holds the graph and the small tensors. Measured with PyTorch 2.14.1: this
  24→64→64→2 network is one file of 30,794 B with the argument, and two files of 7,927 B and
  23,040 B without it. To read two files, fetch both and pass the second:
  `mlpFromOnnx(bytes, { externalData: [{ path: 'policy.onnx.data', data }] })`. Given only the
  first, the reader refuses it and names both ways out.
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
| `initializer "…" is stored outside the file, in "….onnx.data"; pass that file, …` | Pass the second file as `externalData`, with the path the message names. Or export again with the weights inside the file; in PyTorch: `external_data=False` |
| `initializer "…" reads … B at offset … of "…", which holds … B` | The second file given is not the one the `.onnx` file was written with, or it is cut short |
| `the graph has 1 inputs and 2 outputs` | Export the policy alone, not the policy with its value estimate |
| `node "…" is <Op>; supported: Gemm, MatMul, Add, Relu, Tanh, Sigmoid` | The model is not a plain dense network, or uses an activation `MlpBrain` does not have. Retrain with `tanh`, `relu` or `sigmoid`, or run the model through [ONNX Runtime](#onnx-runtime) |
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

## What the reader costs

- **Size:** `mlpFromOnnx` is 2.5 KB min+gzip. An app that does not import it carries none of it.
- **Time:** reading a 41 KB file takes about 2 ms.
- **Safety with files you did not make:** a damaged or hostile file is refused or read; it cannot
  hang the reader, and the reader never holds more than a few times the file's size.

## ONNX Runtime

`OrtRuntime` starts ONNX Runtime Web's WebGPU provider and keeps its device. `OrtModel` is one
model on it, bound once to GPU buffers. `OrtBrain` runs a model as an agents brain. The app
installs `onnxruntime-web` (1.29 is tested) and passes its namespace; this package never imports
or bundles it.

```ts
import * as ort from 'onnxruntime-web/webgpu';
import { NullGraph } from 'null-graph';
import type { AgentSystem } from 'null-graph/agents';
import { OrtBrain, OrtRuntime } from 'null-graph/inference';

/** ONNX Runtime first: it makes the device, and the engine adopts it. */
export async function startWithOrt(canvas: HTMLCanvasElement) {
    const runtime = await OrtRuntime.create(ort);                        // loads the WASM, makes the device: 0.5–0.7 s
    const engine = new NullGraph();
    const init = await engine.init(canvas, { device: runtime.device });   // adopts it, and never destroys it
    if (!init.success) {
        await runtime.destroy();
        throw new Error(init.error);
    }
    return { runtime, engine };
}

/**
 * Gives `species` the model in `bytes`, run by ONNX Runtime. Its pass is compiled before the
 * switch, so that the switch itself compiles nothing. The caller owns the model and the brain.
 */
export async function useOrtBrain(runtime: OrtRuntime, agents: AgentSystem, bytes: Uint8Array, species: string) {
    const model = await runtime.load(bytes);        // one session: the WebGPU provider, outputs kept on the GPU
    let brain: OrtBrain | undefined;
    try {
        brain = new OrtBrain(model);                // refuses a model that does not map [N, obsDim] to [N, actDim] float32
        await agents.prepareBrain(brain);           // binds the model to the system's buffers; compiles off the frame
        agents.setBrain(brain, { species });        // compiles nothing
        return { model, brain };
    } catch (error) {
        brain?.destroy();
        await model.destroy();
        throw error;
    }
}
```

The frame does not change: `agents.advance(seconds)` also starts a run of the model when none is in
flight. To tear down, stop the frames, then destroy the system, each brain, each model
(`await model.destroy()`), and last the runtime (`await runtime.destroy()`): the device goes with
it, and the engine must not render after.

### The device

- **ONNX Runtime 1.29 runs only on a device it made.** Given the app's device, its sessions are
  created and every run fails, and after a few it can stop answering. So the engine is created
  after the runtime, on `runtime.device`. An app that makes its engine first must make it again
  to use ONNX Runtime.
- **`OrtRuntime.create(ort, { adapter? })`** starts the provider, on the given adapter or one
  ONNX Runtime asks for, and keeps one tiny session of its own for its whole life. ONNX Runtime
  destroys its device with its last session; this one keeps it alive while models come and go.
- **`await runtime.destroy()`** releases every model still loaded, then that session, and with it
  the device.

### Models

- **`await runtime.load(bytes, { capture?, externalData? })`** creates one session. The runtime
  sets its options, and the app cannot change them: the WebGPU provider, outputs kept on the GPU,
  and [graph capture](#graph-capture) only with `capture: true`. `externalData` is ONNX Runtime's
  own option, passed as it is: the same list `mlpFromOnnx` takes, where `data` may also be a URL
  that ONNX Runtime fetches.
- **`model.inputs` and `model.outputs`** list each tensor's `name`, `type` and `dims`: a number,
  or a string for a free axis.
- **`model.bind({ inputs: { name: { buffer, dims } }, outputs: { … } })`, once.** Every input and
  every output gets a GPU buffer: float32, of the model's rank, its fixed axes as the model says,
  and each free axis the same size wherever its name appears. Each buffer needs `STORAGE` usage,
  and its values' bytes rounded up to 16: ONNX Runtime reads and writes 16 B at a time, and on
  NVIDIA a write that ends past a buffer is lost without an error. A second `bind` is refused;
  load the model again for other buffers.
- **`await model.run()`** runs the session over the bound buffers. It resolves once ONNX Runtime
  has queued its GPU work; work submitted after it sees the results. One run of a model at a time.
- **`await model.destroy()`** releases the session, in line: after a run in flight.
- **One call into ONNX Runtime at a time.** ONNX Runtime 1.29 fails two calls in flight across its
  sessions: a session created while another is being created, or a load or a run while another
  model runs, which breaks its memory. So every load, run and release of a runtime's models waits
  in line, in the order made. A call starts at once when none is in line, so one model alone waits
  for nothing, and a call that fails does not hold the line. Two models' runs take turns, and
  `destroy()` can wait for another model's run. Calls an app makes on ONNX Runtime itself are not
  in this line, and neither are a second runtime's: use one runtime.
- **Not supported:** tensor types other than float32 (int64 token inputs, float16, quantized
  models), inputs or outputs on the CPU, and ONNX Runtime's older JavaScript provider (its `all`
  bundle, which ignores a device and cannot load under Node): import `onnxruntime-web/webgpu`.

### OrtBrain

`new OrtBrain(model, { input?, output? })` makes a model the brain of the species `setBrain` gives
it. The names default to the model's only input and only output; a model with several needs them
named.

- **Its shape:** the input `[N, obsDim]` and the output `[N, actDim]`, float32. `N` is a free axis,
  or fixed at the system's `capacity`. PyTorch fixes it at the example's batch size unless told
  otherwise: export with `dynamic_shapes`, or `dynamic_axes={'obs': {0: 'N'}}` with the older
  exporter.
- **How it runs:** each run reads every agent's observations and writes a results buffer of the
  brain's own; its think pass copies each agent's row into the actions, for its species' agents
  that think in that step. Its runs never write the actions themselves.
- **Interactive frames act on the actions of the frame before.** `advance()` starts a run when
  the brain has none in flight and `thinkEvery` steps have passed since its last one started,
  never in a frame without a step. The run reads the observations of that frame and lands before
  the next frame's work. Measured over 300 frames: every frame one frame old under Node, and 299
  of 300 in Chrome. When frames come faster than runs, a run can span several frames (up to 7 in
  Chrome), and the latest run wins.
- **`await agents.step()` is exact:** it senses, waits for the run, then acts. Prey on
  `OrtBrain` act within 1.1e-5 of the same network on `MlpBrain` in the same step, and two runs
  of 120 steps give the same bytes.
- **`thinkEvery` is the rate of runs.** At the default step of 1/60 s, `thinkEvery: 2` runs the
  model 30 times a second, at half the GPU cost ([cost](#what-onnx-runtime-costs)).
- **A brain is bound to one system's buffers,** at its first `setBrain` or `prepareBrain`. Each
  time `setBrain` attaches it, it copies the actions agents have into its results, so a species
  keeps acting as it did until the brain's first run lands (0.14 to 0.46 s for a model's first
  run, which compiles ONNX Runtime's pipelines).
- **A run that fails** is thrown by the next `advance()` or `step()`, once:
  `AgentSystem: the brain of species "prey" failed in a run: <ONNX Runtime's message>. Give the
  species another brain with setBrain().` No run of that brain starts after it, and its species
  keeps its last actions.
- **Prepare it:** `await agents.prepareBrain(brain)` binds the brain and compiles its pass
  ([agents.md](./agents.md#switching-brains-without-a-hitch)), as the example does.
- **A newborn agent** acts for one frame on its slot's last results, before a run has seen it.
- **Mixing:** `OrtBrain` serves its species, and `RuleBrain` or `MlpBrain` the others, in one
  system.
- **Teardown:** destroy the system, or give the brain's species another brain, before
  `brain.destroy()`. The brain releases its results buffer after a run in flight; the model stays
  the app's.

### Graph capture

ONNX Runtime can record a session's GPU work on its first run and replay it after. It costs the
same GPU time and saves main-thread time: 0.20–0.28 ms per run against 0.57–0.69 ms under Node,
and 0.06–0.09 ms against 0.74–0.83 ms in Chrome. **It is off unless `capture: true`,** because of
two faults of ONNX Runtime found here:

- **A captured model can fail from its second run,** with `Cannot set properties of undefined
  (setting 'Symbol(gpuBufferMetadata)')`, sometimes for every run after. Whether it fails depends
  on what ONNX Runtime did between the model's first two runs, such as loading or releasing
  another model, and on the host; on 1.30.0 the smallest case failed with nothing in between.
- **Memory grows with every captured run,** and is never given back while the page lives: 12 MB
  per million runs.

Turn capture on only for a model you measured it on, in the sequence your app runs.
`npm run spike:ort-issues` in this repository shows both faults.

### When a model is refused

Every refusal is thrown when the model is loaded, bound, or given to a system, never during a
frame. The message starts with the class that refused it.

| The message says | What to do |
|---|---|
| `OrtRuntime: that is not onnxruntime-web …` | Pass the namespace of `import * as ort from 'onnxruntime-web/webgpu'` |
| `OrtRuntime: ONNX Runtime could not start its WebGPU provider: …` | The browser or the host has no WebGPU, or ONNX Runtime could not reach its WASM files |
| `OrtRuntime: ONNX Runtime could not load the model: …` | ONNX Runtime's own reason follows; for weights in a second file, pass `externalData` |
| `OrtModel: input "…" is int64; only float32 tensors can be bound.` | The model takes another type: not supported |
| `OrtModel: output "…" is [N,2]; it was given [1000,3].` | The buffer's shape is not the model's |
| `OrtModel: "…" needs 8000 B with STORAGE usage: its values, rounded up to 16 B, …` | Make the buffer that large, with `GPUBufferUsage.STORAGE` |
| `OrtModel: bound already; buffers are fixed for life, …` | Load the model again for other buffers |
| `OrtModel: run() while a run is in flight; await it first.` | One run at a time |
| `OrtBrain: the model has 2 inputs ("obs", "mask"); name the observations with { input: '<name>' }.` | Name the input or the output that `OrtBrain` should use |
| `OrtBrain: input "…" is [N,24] float16; expected [N, obsDim] float32.` | The model takes another type, or its input or output is not a matrix of one row per agent |
| `AgentSystem: the brain maps 32→2; the layout is 24→2.` (from `setBrain`) | The model was trained for other observations or actions than this definition has |
| `OrtBrain: input "…" is [1,24]: its first axis is fixed at 1, and the system has 4096 agents. …` | Export with a free first axis, as the message says |
| `OrtBrain: this system's engine has another device. …` | Init the engine with `{ device: runtime.device }` |
| `OrtBrain: bound to another system's buffers; …` | One brain serves one system; load the model again for another |
| `AgentSystem: this brain is still being prepared; …` | Await `prepareBrain` before `setBrain` |

### What ONNX Runtime costs

- **Download:** ONNX Runtime's WASM is 6.4 MB gzipped (25.7 MB), and its JavaScript 58 KB
  gzipped. This package's part, `OrtRuntime`, `OrtModel` and `OrtBrain`, is 3.1 KB min+gzip, paid
  only by apps that import them.
- **Start:** 0.8–0.9 s from the first line to the first actions in Chrome, and 1.3–1.6 s under
  Node: loading ONNX Runtime, making the device (0.5–0.7 s), and a model's first run, which
  compiles ONNX Runtime's pipelines (0.14–0.46 s). A second model starts in tens of milliseconds.
- **Each run,** for a 24→32→32→2 network over 20,000 agents: 0.5–1.3 ms of GPU time on an
  NVIDIA GPU and 1.5–1.7 ms on an AMD iGPU, 3× to 8× the built-in kernel's (up to 10× at 100,000
  agents). Run every frame, that is 8–10% of a 60 fps frame on the iGPU; `thinkEvery: 2` halves
  it. The main thread spends 0.6–0.8 ms per run.

## Current validation

- **Headless** (`npm run test:inference`): the reader gives back, bit for bit, what this
  repository's writer wrote, and one network written in 23 different ways. 13 files written by
  PyTorch 2.14.1 (both exporters), scikit-learn 1.7.2 and `onnx.helper` read with their makers'
  weights bit for bit; 3 are refused as they should be. 10,000 damaged files and every cut-short
  prefix of a file give a network or a named refusal, nothing else. `OrtRuntime`, `OrtModel` and
  `OrtBrain` run against a stand-in for ONNX Runtime: the options every session gets, each
  binding refusal, one run at a time, release after a run in flight, one call into ONNX Runtime at
  a time across models, in the order made, and that the real `onnxruntime-web` namespace
  type-checks as what `OrtRuntime.create` takes. This guide's two examples are type-checked and
  bundled; the reader's runs on the committed networks, and `useOrtBrain` on a stand-in engine.
- **On the GPU** (the `onnx` and `inference` suites), on Node/Dawn with an NVIDIA and an AMD GPU,
  and in Chrome, with ONNX Runtime Web 1.29.0:
  - for 17 files, the network read by `mlpFromOnnx` into an `MlpBrain` and ONNX Runtime running
    the same bytes give the same actions within 4.5e-6;
  - the runtime's device outlives every model, and a model released and loaded again runs;
  - PyTorch's two-file exports run with `externalData`, and agree with PyTorch within 4.8e-7;
  - Herd & Wolves with its prey on `OrtBrain` (A5): within 1.1e-5 of `MlpBrain` on the same step,
    with graph capture on and off; reruns byte-identical; switching brains with frames running;
  - 600 interactive frames (A6): no read-back outside the inspector, no WebGPU error, and the
    frame ages above;
  - `prepareBrain` (A33): `setBrain` of a prepared brain takes under 1 ms;
  - two models at once (A34): prey and wolves each on an `OrtBrain`, both loaded together and both
    runs started together, act as their networks say within 1.1e-5, with no WebGPU error.
- **ONNX Runtime 1.30.0** ran the `inference` suite under Node on an NVIDIA GPU only.
- **Not measured:** files from TensorFlow or Keras, and from reinforcement-learning libraries;
  export the policy as a plain PyTorch module. Models other than dense networks through
  `OrtBrain`: they run as ONNX Runtime runs them, but none is tested here.
