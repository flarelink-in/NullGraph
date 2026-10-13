# NullGraph Agents — C2 step 8: reading ONNX — Technical Design

**Parent:** [`agents-design.md`](./agents-design.md) §9.3 (`MlpBrain.fromOnnx`), §9.4
(`null-graph/inference`), §10 (U12), §11 and §13, C2 step 8.
**Builds on:**
- [`agents-mlp-design.md`](./agents-mlp-design.md): `MlpBrain`, the `MlpNetwork` format, its limits;
- [`agents-cloning-design.md`](./agents-cloning-design.md) §9 and phase 7.10: the `.onnx` writer and
  what its files hold.

**Asked by the user:** 2026-10-13, "start designing phase 8"; 2026-10-14, "give one more round of
design on 8.0 if anything missed before starting".

**Decided with the user, 2026-10-14** (§14; [`work/plan.md`](./work/plan.md) §5, item 8):
1. **The reader goes in a new entry, `null-graph/inference`,** created in this step. The parent
   planned that entry for step 9.
2. **So the reader is a function, `mlpFromOnnx(bytes)`,** not a static method of `MlpBrain`: a
   static method cannot live in another entry.
3. **Files written by other exporters are committed as test fixtures,** made in the repository's
   `venv/`.
4. **After phase 8.0, the recommended option for each of:** the end nodes that scikit-learn and
   PyTorch's older exporter write are passed over; "Use OrtBrain" stays out of the refusals until
   step 9; no TensorFlow or Keras fixtures now.
5. **Weights outside the file are refused,** with a message that names the fix. PyTorch's default
   call writes them that way; the user will look at it again in step 9, where larger models
   arrive.
6. **An `Identity` of an initializer is followed** back to it, and **the reader's size line is
   2,600 B.**

**Nothing is open** (§14).

**Status:** design revised 2026-10-14 after a second round ("What the second round changed", at
the end of §14). Each phase's results are a section at the end of this file, added with the
phase's commit.
- **Phase 8.0 ran on 2026-10-14:** sixteen files from four makers.
- **Phase 8.1 was built on 2026-10-14:** the entry and the reader, 2,525 B min+gzip; every
  fixture behaves as listed; 34 of 34 mutations were caught.
- **Phase 8.2 was built on 2026-10-14:** on three host and GPU pairs, the reader's `MlpBrain` and
  ONNX Runtime give the same actions from the same bytes, within 4.5e-6, for 17 files.
- **Next: phase 8.3,** the guide and the README row.

**Target:** a new entry `src/extras/inference/` with one file, tests, fixtures, one guide. Core
and the agents entry unchanged. No new dependency.

---

## 1. Purpose

**Turn the bytes of an `.onnx` file into an `MlpNetwork`,** the value `MlpBrain` already runs. A
dense network trained anywhere (PyTorch, scikit-learn, this repository's trainer) then loads
without a converter and without ONNX Runtime.

**Words used here:**
- **reader:** the new function, bytes in, network out;
- **writer:** `tools/agents/onnx.mjs`, phase 7.10;
- **dense subset:** the ONNX graphs the reader takes (§5);
- **initializer:** a constant tensor stored in the file, here a layer's weights or biases;
- **fixture:** a small `.onnx` file kept for the tests, with what its maker computed from it.

### The single most important pre-existing fact

**Everything after the network already exists.** `new MlpBrain(engine, network)`,
`MlpBrain.bySpecies`, `setWeights` and `mlpForward` all take an `MlpNetwork` (facts 1–4). So the
reader is a pure function on the CPU: no device, no shader, no pipeline. Reading happens once,
when the file is loaded; the network then runs on the GPU as `MlpBrain`'s think stage, as today.

### The second fact: a round trip proves less than it seems

The parent's U12 is "exporter ↔ `fromOnnx` round trip is byte-exact". This repository wrote the
writer, and will write the reader. If both hold the same wrong idea, such as which way a weight
matrix is transposed, the round trip still passes. Judges that share nothing with the reader are
needed (§10):
- **ONNX Runtime,** running the same bytes;
- **files written by other exporters,** with the weights and outputs their own framework had.

### Non-goals

- **Anything beyond dense layers:** convolutions, recurrent layers, attention, normalization
  layers, residual connections. They are refused by name (§11). Step 9's `OrtBrain` is their way in.
- **Activations `MlpBrain` does not have:** `LeakyRelu`, `Elu`, `Gelu`, `Clip`, `Softmax`. Each
  needs a new activation in the kernel, which is a change to `MlpBrain`, not to the reader.
- **Networks larger than `MlpBrain` runs** (4 layers, hidden width 64). The reader does not shrink,
  split or approximate one.
- **Graphs with several outputs,** such as a policy exported together with its value estimate.
  The model's author exports the policy alone; the guide shows how (§13, phase 8.3).
- **Rewriting a graph.** No constant folding, no merging of scale and shift nodes into a layer.
- **Other number types.** Float32 weights only: no float16, float64 or quantized integers.
- **Weights outside the file** (ONNX "external data"). Refused with the fix named; to be looked
  at again in step 9 (§14, item 8).
- **Running ONNX Runtime,** and the parent's A5 (`OrtBrain` against `MlpBrain`): step 9.
- **Writing ONNX from the library.** The writer stays in `tools/agents/`.
- **The demo.** The site keeps loading `.mlp.json` until step 10.

### Design principles applied

- **Verify against the tree** ([`design-principles.md`](./design-principles.md) §1): §3's table.
  What other exporters emit is measured in phase 8.0, not assumed.
- **Fail loud, naming the offender** (§2): every refusal names the node or tensor and says what
  to do (§11). The reader runs once at load time, never per frame.
- **Detect, don't polyfill** (§12): an unsupported graph is refused, not approximated.
- **A second implementation drifts** (§13.1): the writer and the reader are judged by ONNX Runtime
  and by foreign files, not only by each other.
- **Tests pin numbers, and must be seen to fail** (§13): §10.

## 2. Architecture

```
 prey.onnx ──fetch──► bytes
                          │
                          ▼
             ┌── mlpFromOnnx (CPU, once) ─────────┐      null-graph/inference
             │ protobuf fields → nodes → walk the │
             │ chain from the graph's input →     │
             │ decode only the tensors it uses    │
             └───────────────┬────────────────────┘
                             ▼
        { network: MlpNetwork, input, output, metadata }
                             │                                  null-graph/agents
       ┌─────────────────────┼──────────────────────┬─────────────────┐
       ▼                     ▼                      ▼                 ▼
 new MlpBrain(engine,   MlpBrain.bySpecies(…)   brain.setWeights(…)  mlpForward(…)
     network)                                                        (CPU reference)
```

The frame is unchanged: `engine.render([...agents.getComputeStages(), pass])`. The reader adds no
stage and no buffer.

## 3. Integration facts (verified against this repository, 2026-10-14)

| # | Fact | Where | Consequence |
|---|---|---|---|
| 1 | `MlpBrain`'s constructor takes `(engine, network)` | `MlpBrain.ts`, the constructor | The parent's `MlpBrain.fromOnnx(bytes)` cannot build a brain as written: it has no engine |
| 2 | `MlpBrain.bySpecies` takes one `MlpNetwork` per species; `setWeights` takes one | `MlpBrain.ts`, `bySpecies`, `setWeights` | A reader that returns a network feeds both |
| 3 | A layer is `{ input, output, activation, weights, biases }`; weights are row-major `[output][input]`, as `Float32Array` | `mlpMath.ts`, `MlpLayer` | The reader's result type exists already |
| 4 | Every consumer validates the network itself: the constructor through `mlpLayout`, `setWeights` through `packMlpWeights`, `mlpForward` through `validate` (1–4 layers, input ≤ 128, hidden ≤ 64, output ≤ 16, chained widths, finite values) | `MlpBrain.ts`; `mlpMath.ts`, `validate` | A network the reader returns cannot run unchecked. The reader need not repeat those limits (§7) |
| 5 | An entry is one line in `build.js`'s `entryPoints` and one block in `package.json`'s `exports`. The README lists every entry in a table, with a guide in `docs/api/` | the agents entry was added that way (`7f4cdb5`: `build.js` +1, `package.json` +5); `README.md`, "null-graph/agents" row | What creating `null-graph/inference` touches (§12) |
| 6 | Each entry is bundled on its own, with no code splitting. Runtime code imported from another entry's folder is copied into the importing entry, as `debug-ui/agentDrawing.ts` does with `AGENT_STRUCT_WGSL` from `../agents/agentLayout`. A type import costs nothing | `build.js` (no `splitting`); that import | Importing `mlpLayout` would copy it into `inference.js`. The reader imports types only |
| 7 | Guide examples resolve `null-graph/<entry>` through the package's own `exports`: there is no alias and no copy in `node_modules`. They are type-checked and bundled from the repository root | `test/agents-api-example.test.mjs` (`moduleResolution: 'Bundler'`, esbuild with `resolveDir: root`) | An example that imports `null-graph/inference` is checked the same way once the entry is exported and built. That test expects exactly four examples in `agents.md` |
| 8 | The writer emits, per layer, `Gemm(input, w, b)` with `transB = 1`, then `Tanh`, `Relu` or `Sigmoid`; nothing after an identity layer | `tools/agents/onnx.mjs`, `onnxFromNetwork` | The simplest graph the reader must take |
| 9 | The writer stores weights as raw little-endian f32, dims unpacked, opset 13, IR version 8; the input is `obs`, the output `act` | `onnx.mjs`, `floatTensor`, the model fields | Raw bytes can come back bit for bit |
| 10 | The writer stores `species`, and `obsNames` and `actNames` as JSON text, in the model's metadata | `tools/agents/export.mjs`; `onnx.mjs`, `metadata` | The reader returns metadata, so an app can check names as it does for `.mlp.json` |
| 11 | The `.mlp.json` reader checks names against the definition, and lives in `tools/agents/`, not in the package | `tools/agents/model.mjs`, `networkFromModel` | The library has no name check today; the guide shows it as app code |
| 12 | U12's test carries its own small protobuf decoder | `test/agents-tools.test.mjs`, `protoFields` | A second decoder exists to check the reader's against |
| 13 | The `onnx` GPU suite runs ONNX Runtime on bytes built in memory and compares with `mlpForward` (A27). In the browser it reads repository files by `fetch` | `test/onnx-gpu/run.js`, the A27 block, `readJson` | ONNX Runtime can judge the reader there, on fixtures kept as JSON |
| 14 | The whole agents entry is 19,323 B min+gzip; C2's ceiling is 21,495 B: 2,172 B left | `node test/spikes/mlp-think-cost/size.mjs`, 2026-10-13 | With the reader elsewhere, the agents entry keeps all of it |
| 15 | A prototype reader (outside the repository), bundled as an entry of its own, is **1,946 B** min+gzip importing types only, and **2,453 B** with `mlpLayout` copied in. Inside the agents entry it added 1,734 B; as a static method it cost every `MlpBrain` app 1,767 B | measured 2026-10-13 and -14 with `size.mjs`'s settings | The size line of §12; types only (§7) |
| 16 | The prototype reads `prey.onnx` and `wolf.onnx` (41,186 B each) back to the committed networks bit for bit, in 1.5–2.6 ms | same run, against `tools/agents/models/herd-wolves/*.mlp.json` | The round trip works; reading is not a cost worth designing around |
| 17 | The package has no `sideEffects` field and ships `dist/` only | `package.json` | Tree-shaking is by export; fixtures add nothing to the package |
| 18 | `core.autocrlf` is `true` on this machine and the repository has no `.gitattributes`. No binary file is committed under `test/` or `tools/` | `git config`; `git ls-files`; `test/onnx-gpu/tiny-model.mjs` ("no binary fixture") | Git decides text or binary by looking for a zero byte. A tiny `.onnx` may hold none, be taken for text, and be altered on checkout. Fixtures are kept as base64 inside JSON (§9) |
| 19 | **Checked against the `onnx` package, 1.23.2:** every field number of §4. Operator versions: `Gemm` 1, 6, 7, 9, 11, 13; `MatMul` 1, 9, 13; `Add` 1, 6, 7, 13, 14; `Relu` 1, 6, 13, 14; `Tanh` and `Sigmoid` 1, 6, 13. `Gemm` lost `broadcast` at 7 and its third input became optional at 11. `Add` lost `axis` and `broadcast` at 7. The latest operator set is 28 | `onnx.defs` and the message descriptors, queried in `venv/` | The floor of 7 in §5. Nothing after 14 changes these six |
| 20 | The same package writes a tensor's dims unpacked and a float list packed; `numpy_helper` writes raw bytes; an attribute set to zero is written out | serialized by hand in `venv/` | The reader takes both encodings, and treats a field left out as zero (§6) |
| 21 | `venv/` (ignored by git, Python 3.10.6) holds, since 2026-10-14: `torch` 2.14.1, `onnx` 1.23.2, `onnxruntime` 1.23.2, `onnxscript` 0.7.2, `scikit-learn` 1.7.2, `skl2onnx` 1.20.0, `numpy` 2.2.6 | `pip list` | Phase 8.0's exporters. `pip` there needs `--cert venv/pip-ca-bundle.pem` ([`work/gotchas.md`](./work/gotchas.md)) |

**What those exporters write** was measured in phase 8.0 (§9; results at the end of this file).
It confirmed §5's forms and raised three questions, all decided since (§14, items 4, 8 and 9).

## 4. Data model

No GPU structure is added. The reader's result:

```ts
export interface OnnxMlp {
    /** The layers, ready for MlpBrain, MlpBrain.bySpecies, setWeights or mlpForward. */
    readonly network: MlpNetwork;
    /** The graph's input and output names, for instance "obs" and "act". */
    readonly input: string;
    readonly output: string;
    /** The model's metadata_props, as written. Empty when the file has none. */
    readonly metadata: Readonly<Record<string, string>>;
}
```

**The network owns its arrays.** Every `weights` and `biases` is a new `Float32Array`; none is a
view of the caller's bytes. The caller may reuse or drop the bytes.

**What is read from the file** (field numbers checked against `onnx` 1.23.2, fact 19):

| Message | Fields read | Everything else |
|---|---|---|
| `ModelProto` | 7 graph, 8 opset_import, 14 metadata_props | skipped |
| `OperatorSetIdProto` | 1 domain, 2 version | — |
| `GraphProto` | 1 node, 5 initializer, 11 input, 12 output | skipped |
| `NodeProto` | 1 input, 2 output, 3 name, 4 op_type, 5 attribute, 7 domain | skipped |
| `AttributeProto` | 1 name, 2 f, 3 i | skipped |
| `TensorProto` | 1 dims, 2 data_type, 4 float_data, 8 name, 9 raw_data, 14 data_location | skipped |
| `ValueInfoProto` | 1 name | skipped (shapes come from the weights) |

## 5. The dense subset

A graph is taken when it is **one chain** from its single input to its single output, made of
dense layers. A layer is one of:

| Form | Nodes | Weights in the file | Becomes | Written by (phase 8.0) |
|---|---|---|---|---|
| A | `Gemm(x, W, b)`, `transB = 1` | `W` is `[output, input]` | copied as it is | PyTorch, both exporters, for a batch; this repository's writer |
| B | `Gemm(x, W, b)`, `transB = 0` | `W` is `[input, output]` | transposed | no exporter measured; a file built by hand |
| C | `MatMul(x, W)` then `Add` with `b`, in either operand order | `W` is `[input, output]` | transposed | PyTorch for one vector (each exporter in its own operand order); scikit-learn |

followed by at most one of `Relu`, `Tanh`, `Sigmoid`. No activation node means `identity`.

- **The data is the first input** of a `Gemm` or a `MatMul`. `MatMul(W, x)` is another function
  and is refused.
- **A missing bias is zeros.** `Gemm`'s third input is optional from opset 11, and a `MatMul` may
  have no `Add`.
- **A bias is `[output]` or `[1, output]`.** Any other shape that ONNX would broadcast, a single
  value for instance, is refused.
- **`Gemm` with `alpha`, `beta` or `transA` away from their defaults is refused.** Folding a scale
  into the weights would change their bits, and no exporter is known to need it.
- **An `Add` is read only after a `MatMul`, and an activation only after a layer.** Elsewhere each
  is refused with its own message.
- **Weights and biases must be initializers:** float32, inside the file, as raw bytes or as a
  float list. A value computed by another node is refused.
- **An `Identity` of an initializer stands for it,** through any number of `Identity` nodes.
  PyTorch's older exporter writes that for tensors with equal values (§14, item 9). On the way of
  the data, between two layers, an `Identity` is refused like any other foreign node.
- **Two layers may read the same initializer.** PyTorch's default exporter stores equal tensors
  once (`torch-shared`). Each layer gets a copy of its own.
- **Shapes come from the weights.** The graph's declared input and output shapes, the batch
  dimension among them, are not read: the layers define a function of one vector.
- **The graph's input** is the one input that is not an initializer. Old files (IR version 3 and
  earlier) list initializers among the inputs too.
- **Every node must be on the chain.** A node the walk never reaches is refused by name, and so is
  a value that two nodes read.
- **Operator sets:** every node must be in the default domain (`""` or `ai.onnx`), and the model
  must declare that set at version 7 or later (fact 19). Other sets may be declared; only a node
  that uses one is refused. No upper bound: an unknown operator is refused by name anyway.
- **Nodes at the two ends that change no value** (decided after phase 8.0; §14, item 4):
  - before the first layer, a `Cast` to float32 and a `Flatten` with `axis = 1`;
  - after the last layer, a `Reshape` that nothing reads.

  They are passed over there and refused anywhere else. scikit-learn writes the `Cast` and the
  `Reshape` around every regressor, and PyTorch's older exporter keeps a `Flatten`. For one
  vector none of them changes a value or its order.
- **A `Constant` node is refused.** No exporter measured writes one around a dense network.

**`tanh` is clamped by `MlpBrain`, not by ONNX.** `MlpBrain` clamps tanh's input to ±15, and
`tanh(15)` is 1 in f32, so the results agree (7.10's results).

## 6. Reading the bytes

- **A hand-written protobuf reader,** as the writer and U12's decoder are. No dependency.
- **Varints are read as numbers,** exact to 2⁵³. **One longer than 10 bytes is refused:** past
  that, the arithmetic gives infinity or NaN, and a NaN length would end the reading silently with
  the rest of the file unread.
- **Wire types 0, 1, 2 and 5 are read or skipped;** the others are refused, since ONNX has none.
- **Repeated numbers are taken packed or unpacked** (fact 20).
- **A field left out is zero or empty,** as protobuf defines: an attribute named `transB` with no
  value is 0. An attribute that is absent has its ONNX default.
- **Every length is checked against the bytes left** before it is used, so a file cut short is
  refused.
- **The graph is checked before any weight is decoded.** The first pass collects nodes and the
  initializers' names, without copying a tensor. The walk then decodes only the tensors its layers
  use. A 200 MB model of another kind is refused at its first foreign operator, having copied
  nothing.
- **Memory and time are bounded by the file.**
  - A tensor's byte length must equal its dims before an array is made, so no single array is
    larger than the file.
  - **The layers together may not hold more values than the file has bytes.** Without this, a
    file of many layers that all read one large tensor would be copied once per layer. A network
    of `MlpBrain`'s four layers can never reach that limit, so nothing that could run is refused.
  - Readers of each value are found through a map built once, so the walk is linear.
- **A graph that loops is refused.** A node reached a second time ends the walk with an error; an
  invalid file cannot make it run forever.
- **Names from the file never become property names of a shared object.** Tensors and readers are
  kept in `Map`s; `metadata` is built with `Object.fromEntries`, so a key named `__proto__` is an
  ordinary entry.
- **Floats are read little-endian through a `DataView`,** so an unaligned byte offset and a
  big-endian host both give the right values.
- **Bits are preserved.** Raw weights reach the network unchanged, a `-0` included; a transpose
  moves values without arithmetic.

## 7. The public contract

```ts
// null-graph/inference
/** The dense network in an .onnx file's bytes. Throws, naming the node, when it is not one. */
export function mlpFromOnnx(bytes: Uint8Array | ArrayBuffer): OnnxMlp;
```

In a new file `src/extras/inference/mlpOnnx.ts`, exported from `src/extras/inference/index.ts`.

```ts
import { MlpBrain } from 'null-graph/agents';
import { mlpFromOnnx } from 'null-graph/inference';

const { network, metadata } = mlpFromOnnx(await (await fetch('prey.onnx')).arrayBuffer());
const brain = new MlpBrain(engine, network);
agents.setBrain(brain, { species: 'prey' });
```

- **What the reader guarantees about a network it returns:** at least one layer; each layer's
  `weights` and `biases` hold exactly `input × output` and `output` values; every array is its own.
- **What it leaves to `MlpBrain`:** the limits (4 layers, the widths), finite values, and that
  each layer reads what the one before writes. `MlpBrain`'s constructor, `setWeights` and
  `mlpForward` each check those already (fact 4), with `MlpBrain`'s messages. The refusal comes
  one line later in the app, still at load time. Repeating the checks in the reader would copy
  `mlpLayout` into `inference.js`: 2,453 B against 1,946 B (fact 15).
- **The names are the app's to check,** as for `.mlp.json`: `metadata.obsNames` against
  `definition.obsNames`. The guide's example shows it. A file from another exporter has no such
  metadata, and nothing can check what its inputs mean; the guide says so.
- **`setBrain` still checks the widths** against the definition, with its own message.

## 8. What the reader does not promise

- **That the network behaves as it did where it was trained.** The reader preserves the weights.
  Input order, scaling and normalization are the model author's. A policy trained on normalized
  observations needs the same normalization here, and no file says so.
- **Agreement to the bit with the training framework.** `MlpBrain` on the GPU agrees with
  `mlpForward` within 1e-5 + 1e-5·|cpu| for weights of unit scale, and within 4.4e-5 for the
  trained Herd & Wolves networks (7.9's results).

## 9. Fixtures from other exporters, and phase 8.0

**Phase 8.0 is a measurement. It writes no library code.** It answers: what do real exporters
write for a dense network, and does §5 take it?

### What is made

One Python script, `test/onnx-fixtures/make.py`, run with `venv/Scripts/python`. Every network
has seeded random weights; none is trained.

| Case | Made with | Network | The question |
|---|---|---|---|
| `torch-chain` | PyTorch, its default exporter | 5→7 tanh→3 | The common form: which operator, which `transB`, how weights are stored, which opset and IR version |
| `torch-acts` | the same | 5→7 relu→4 sigmoid→3 tanh | The three activation nodes, and an activation on the output |
| `torch-nobias` | the same | 5→3 without a bias | A missing bias |
| `torch-vector` | the same, example input without a batch dimension | 5→7 tanh→3 | Whether `MatMul` + `Add` appears, and `Add`'s operand order |
| `torch-flatten` | the same, `nn.Flatten()` first | 5→7 tanh→3 | Whether a `Flatten` that does nothing survives the export |
| `torch-double` | the same, float64 | 5→3 | The refusal for another number type |
| `torch-heads` | the same, a policy head and a value head | 5→7 tanh→(3, 1) | What an actor-critic export holds, and which refusal its author meets |
| `torch-outside` | the same | 13→5 (65 weights) | Added during 8.0: the smallest file whose weights the default call stores outside it |
| `torch-big` | the same, called four ways | 24→64→64→64→2 tanh | Where the weights of a real-sized network go. **Measured, not kept** |
| `torchscript-chain`, `-vector`, `-nobias`, `-flatten` | PyTorch's older exporter (`dynamo=False`) | as above | The forms older tutorials and files have |
| `torch-shared`, `torchscript-shared` | both PyTorch exporters | 5→7→7→7→3 tanh, every bias zero, two equal 7×7 matrices | Added after 8.0's first run: what each exporter does with tensors that hold equal values |
| `sklearn-mlp` | scikit-learn `MLPRegressor` through `skl2onnx` | 5→7 tanh→3 | A second exporter that shares no code with PyTorch |
| `helper-gemm0` | built by hand with `onnx.helper` | 5→7 tanh→3 | `transB = 0` and a float list, serialized by another library |

### What is recorded for each

- **The file's bytes, as base64,** and their SHA-256 (fact 18).
- **What the file holds,** listed with the `onnx` package, not with our code: IR version, operator
  sets, producer, each node (operator, domain, inputs, outputs, attributes), each initializer
  (dims, type, raw bytes or float list or outside the file), the graph's inputs and outputs.
- **The network as its framework holds it:** each layer's widths, activation, weights and biases,
  as f32. This allows a check to the bit, which outputs alone do not.
- **Four inputs** (seeded; one all zeros, one large enough to saturate `tanh`) **and the outputs
  from three judges:** the framework itself, `onnxruntime` on the CPU, and the `onnx` package's
  reference evaluator. They must agree with each other before the file is trusted.
- **Whether `onnx.checker` accepts the file.**
- **The versions of every package used,** and a `requirements.txt` beside the script.

All of it goes into one text file, `test/onnx-fixtures/fixtures.json`. Running the tests needs no
Python; only regenerating does. **What the reader should do with each file is not in it:** that
is the test's to state (U31), so the fixtures record only what their makers did.

- **Local paths are not committed.** If an exporter writes file paths of this machine into a
  file (stack traces in node metadata, for instance), those entries are removed before the file
  is stored, and the record says so.
- **`torch-big` is measured and not stored:** 41 KB per file adds nothing a 300-byte file does not
  show, unless its weights are stored differently. Then it is reported.

### What phase 8.0 reports

- **Per case:** the operators, `transB`, the storage, the opset and IR version, any extra node.
- **Whether the prototype reader takes each file,** and with which refusal when not. The
  prototype stays outside the repository.
- **The largest difference between the three judges,** and between `mlpForward` and the framework.
- **What §5 must change:** an operator to take, a refusal to reword. A larger subset goes to the
  user first.

### Not made, unless the user asks

- **TensorFlow and Keras** (through `tf2onnx`): about 1.5 GB, and its pins on `numpy` and
  `protobuf` may change packages that other scripts in `venv/` use (§14, item 7).
- **A real reinforcement-learning library** (Stable-Baselines3 and its dependencies).
  `torch-heads` imitates the shape of its export with plain PyTorch.

## 10. Test suite

**Headless** (a new file `test/inference-onnx.test.mjs`, run by a new `npm run test:inference`):

| # | Asserts |
|---|---|
| U12 | The reader half, completing the parent's check: writer → reader gives the network bit for bit (five layers, all four activations, a `-0`, the largest and smallest f32), with the names and the metadata. The same from an `ArrayBuffer`, from a view at an odd offset, and after the file's bytes are wiped: the arrays are the reader's own. Five layers read here and are refused by `mlpLayout`: the limits are `MlpBrain`'s. An unsupported operator is refused by name |
| U29 | Each form of §5, built in the test from the writer's encoding helpers: `Gemm` with `transB` 0 and 1, `MatMul` + `Add` in both operand orders, a missing bias, a `[1, output]` bias, a float list instead of raw bytes, packed dims, initializers listed among the inputs, an attribute with its value left out. Each gives the same network as form A, to the bit. **Watched failing** with the transpose removed, on a square layer. A `Cast`, a `Flatten` and a `Reshape` at the ends give the same network too, and each is refused between two layers, as are a `Cast` to another type and a `Flatten` on another axis. Weights and biases read through one or two `Identity` nodes give the same network; an `Identity` between two layers is refused. One tensor read by two layers gives each layer its own copy |
| U30 | No file hangs the reader or breaks it. (a) Every strict prefix of a valid file either is refused with a message of §11 or, when it ends exactly on a field boundary after the graph and the operator set, gives the same network with less metadata; the test counts those boundaries with its own decoder. (b) 10,000 files with one seeded random byte changed each return, or throw an `Error` whose message starts with `mlpFromOnnx: `; none throws anything else, and none takes more than a second. (c) Named cases: a varint of 11 bytes; a graph that loops; a metadata key `__proto__`, and operators named `constructor` and `toString`; a tensor whose dims promise more than the file holds; a foreign operator beside a broken tensor, refused by the operator's name; forty layers that read one tensor, refused, while four read; two `Identity` nodes that feed each other |
| U31 | Every fixture of §9: its bytes match its SHA-256. Each file the test lists as read gives its maker's network bit for bit, and `mlpForward` gives the maker's outputs within 1e-5 + 1e-5·\|expected\| (phase 8.0 measured 4.2e-7 at worst). Each file it lists as refused is refused with the message the test states: float64, two outputs, weights outside the file. No fixture is left unlisted |
| U14 | Every message of §11 fires and names the offender |

**GPU** (the `onnx` suite, Node + Dawn and Chrome):

| # | Asserts |
|---|---|
| A28 | For each of: the two Herd & Wolves files, a `MatMul` + `Add` rewrite of them, and each fixture that is read: the bytes read by `mlpFromOnnx` into an `MlpBrain` give, on 256 observations, the actions ONNX Runtime gives from **the same bytes**, within 1e-4 (A27's bound). **Watched failing** with a square layer's weights read the other way round, which no shape check can see. As built (phase 8.2): a file with a fixed batch of one, or with no batch dimension, is run row by row on 32 observations; a file ONNX Runtime cannot load or run fails its check; a first check states that the reader refuses exactly the three files it must |

The parent's A5 (`OrtBrain` against `MlpBrain`) stays with step 9. A28 is the same comparison
without `OrtBrain`.

**Also:**
- **The size line is a test,** in the same file: the reader as its own bundle is at most 2,600 B
  min+gzip; it holds none of `MlpBrain`'s code; a bundle of the agents entry holds none of the
  reader; an import of the entry that uses nothing leaves nothing behind. `size.mjs` reports the
  numbers and repeats the second check.
- **The guide's example** is type-checked, bundled and run on a stand-in engine, as the four
  examples of `agents.md` are (fact 7). Phase 8.3.
- **The entry is checked:** it exports `mlpFromOnnx` and nothing else, and the files that
  `package.json` names for it exist after `npm run build:lib`.

## 11. Errors

All thrown by the reader, at load time. `<node>` is the node's name, or its first output when it
has none. `<Op>` carries its domain when that is not the default.

| Condition | Message |
|---|---|
| Not bytes | `mlpFromOnnx: expected the file's bytes, as a Uint8Array or an ArrayBuffer.` |
| Cut short | `mlpFromOnnx: the file ends inside a field; it is cut short or is not ONNX.` |
| Not protobuf (wire type, varint) | `mlpFromOnnx: the file's bytes are not protobuf; it is not ONNX.` |
| No graph | `mlpFromOnnx: the file has no graph; it is not an ONNX model.` |
| Operator set too old or missing | `mlpFromOnnx: the model's operator set is version <v>; the reader takes version 7 or later.` |
| Not one input and one output | `mlpFromOnnx: the graph has <i> inputs and <o> outputs; a dense network has one of each.` |
| No layers | `mlpFromOnnx: the graph has no layers.` |
| Unsupported operator | `mlpFromOnnx: node "<node>" is <Op>; supported: Gemm, MatMul, Add, Relu, Tanh, Sigmoid.` |
| `Add` or an activation out of place | `mlpFromOnnx: node "<node>" (<Op>) is out of place: an Add follows a MatMul, and an activation follows a layer.` |
| The data is not the first input | `mlpFromOnnx: node "<node>" (<Op>) must read "<value>" as its first input.` |
| A branch | `mlpFromOnnx: node "<node>" (<Op>) also reads "<value>"; a dense network is one chain of layers.` |
| A node off the chain | `mlpFromOnnx: node "<node>" (<Op>) is not on the chain from the input to the output.` |
| A loop | `mlpFromOnnx: node "<node>" (<Op>) is reached twice; the graph loops.` |
| Scaled `Gemm` | `mlpFromOnnx: node "<node>" (Gemm) has alpha, beta or transA set; only the defaults are read.` |
| Weights computed by a node, or missing | `mlpFromOnnx: node "<node>" (<Op>) needs "<tensor>" to be a float32 initializer.` |
| Wrong number type | `mlpFromOnnx: initializer "<tensor>" has data type <t>; only float32 (1) is read. Export the model as float32.` |
| External data | `mlpFromOnnx: initializer "<tensor>" is stored outside the file, in "<file>"; export with the weights inside it (PyTorch: external_data=False).` The recipe is there because this is what PyTorch's default call produces (phase 8.0) |
| Size disagrees with shape | `mlpFromOnnx: initializer "<tensor>" is [<dims>] but holds <n> values.` |
| One tensor read by more layers than a file can pay for (§6) | `mlpFromOnnx: initializer "<tensor>" is used so often that the layers hold more values than the file has bytes.` |
| Weights not a matrix | `mlpFromOnnx: node "<node>" (<Op>) has weights of shape [<dims>]; a dense layer's are [rows, columns].` |
| Bias shape | `mlpFromOnnx: node "<node>" (<Op>) has a bias of shape [<dims>] for <o> outputs.` |
| The chain misses the output | `mlpFromOnnx: the layers end at "<value>", not at the graph's output "<name>".` |

**Thrown later, by `MlpBrain` or `mlpForward`,** when the network is used: too deep, too wide,
widths that do not chain, a value that is not finite
([`agents-mlp-design.md`](./agents-mlp-design.md) §6.1).

**Against the parent:** its message ends "Use OrtBrain." `OrtBrain` does not exist until step 9,
so this draft leaves the pointer out and step 9 adds it (§14, item 5).

## 12. Engine touchpoints

### Required: a new entry in the package

| File | Change |
|---|---|
| `src/extras/inference/index.ts`, `mlpOnnx.ts` | new |
| `build.js` | one line in `entryPoints` |
| `package.json` | an `exports` block for `./inference`; a `test:inference` script |
| `README.md` | one row in the entries table |
| `docs/api/inference.md` | new: the entry's guide |

**Core: none. The agents entry: none,** not one byte.

**Zero-change fallback:** the reader inside the agents entry. It works, and it spends 1,734 B of
the 2,172 B left under C2's ceiling (fact 15). The user chose the new entry.

### Bundle

| | Before step 8 | As built in 8.1 |
|---|---:|---:|
| Whole agents entry (ceiling 21,495 B) | 19,323 B | 19,323 B |
| An app with rules and `MlpBrain` | 15,517 B | 15,517 B |
| The same app, reading ONNX too | — | 17,840 B (+2,323 B) |
| `inference.js`, which is the reader alone until step 9 | — | 2,525 B |

**The reader's line: 2,600 B min+gzip as its own bundle** (§14, items 6 and 9).
- The first prototype was 1,946 B, without six of §11's messages and four of §6's checks.
- A second prototype, written to this revision, was 2,332 B; the end nodes of §5 added 68 B.
- The first build was 2,460 B, with the limit on reused tensors (§6) added: 40 B under a line
  of 2,500 B.
- **As built: 2,525 B,** with an `Identity` followed to its initializer (65 B). The user moved
  the line to 2,600 B for it. **75 B are left.**
- **Phase 8.1's exit is the reader at or under the line,** and a test holds it there (§10).

**Step 9 must keep the reader free of ONNX Runtime code:** an app that imports only `mlpFromOnnx`
must not carry the adapter. `size.mjs` checks it from step 9 on.

### Optional: none proposed

## 13. Build order

| Phase | Builds | Exit |
|---|---|---|
| **8.0** ✅ | `test/onnx-fixtures/make.py`, `requirements.txt`, `fixtures.json` (§9). No library code | **Measured:** per exporter, what the file holds; whether the prototype takes it; the judges' agreement. §5 is confirmed or amended, with the user's word if the subset grows. Run 2026-10-14, not committed; results at the end. The subset grows by three end nodes: §14, item 4 |
| **8.1** ✅ | The entry (§12) and `mlpOnnx.ts`; U12's reader half, U29, U30, U31, U14; the size lines | The Herd & Wolves files read back bit for bit; every fixture behaves as recorded; the reader is within its line and the agents entry is unchanged to the byte; the suite is green headless. Built 2026-10-14, not committed; results at the end. The README row and the guide are phase 8.3's |
| **8.2** ✅ | A28 in the `onnx` suite | ONNX Runtime agrees within 1e-4 on Node/NVIDIA, Node/AMD and Chrome/AMD. Built 2026-10-14: 23 of 23 checks on all three, worst difference 4.5e-6; results at the end |
| **8.3** | `docs/api/inference.md` with a checked example and "exporting a policy from PyTorch"; the agents guide's "As ONNX" line; the README row; `tools/agents/README.md`; the parent's §9.3, §11 and §12.5 brought in line; closure | Docs match what was measured; `docs/work` updated |

Every test is watched failing before it passes. One phase per commit.

### Deferred

- **`OrtBrain`, the generic ONNX Runtime adapter, the parent's A5:** step 9. Spikes S1, S2, S4 and
  S5 are still due before it; creating the entry now does not need them.
- **The demo loading `.onnx`:** step 10.
- **A name check inside the library** (for `.mlp.json` and `.onnx` alike): only if a second app
  needs it; the guide's example is four lines.
- **More activations** (`LeakyRelu`, `Elu`, `Clip`): a change to `MlpBrain`'s kernel; revisit when
  a real policy needs one.
- **Choosing one output of a graph with several:** a walk backwards from a named output would
  ignore the other branches. Not designed; phase 8.0's `torch-heads` shows how far such a file is
  from the subset.
- **Weights outside the file:** refused by name today. The user will look at it again in step 9,
  where larger models arrive (§14, item 8).
- **Float64 and float16 weights, graph rewriting:** refused by name today.

## 14. Decisions

### Decided by the user, 2026-10-14

1. **Where the reader lives: `null-graph/inference`,** created in this step ("yes create and use
   that entry"). The agents entry keeps its 2,172 B.
2. **The API's shape: a function, `mlpFromOnnx(bytes)`,** which follows from 1. It returns a
   network, so it feeds `bySpecies`, `setWeights` and `mlpForward` as well as the constructor.
   The parent's name, `MlpBrain.fromOnnx`, is retired; phase 8.3 updates the parent.
3. **Files from other exporters are committed,** made in the existing `venv/` ("yes that's
   fine"; "i meant same venv").

After phase 8.0, the user took the recommended option for each of these ("for rest u can choose
recommended options"):

4. **Nodes at the ends that change no value are passed over** (§5): a `Cast` to float32 or a
   `Flatten` before the first layer, a `Reshape` after the last. They are refused anywhere else.
   - **Reads:** every scikit-learn `MLPRegressor`, and a PyTorch model with `nn.Flatten()` from
     the older exporter. A scikit-learn user has no export option that avoids them.
   - **Cost:** 68 B, measured on the second prototype; three more operators to test.
   - **Not chosen:** the parent's six operators only.
5. **"Use OrtBrain" stays out of the refusals** until step 9 builds `OrtBrain`. A message should
   not point at a class that does not exist.
6. **The reader has a size line of its own,** as its own bundle (§12): first 2,500 B min+gzip,
   then 2,600 B with item 9. The parent says only "a few KB" for the whole of `inference.js`.
7. **No TensorFlow or Keras fixtures now.** PyTorch (two exporters), scikit-learn and a hand-built
   file cover the forms; TensorFlow costs about 1.5 GB and may disturb the other packages in
   `venv/`.

After phase 8.1's first build ("choose both recommended option since onnx is step 9 to test
running larger models then we can rethink this"):

8. **Weights outside the file are refused, with the fix named.** PyTorch's default call stores
   every weight tensor over 256 bytes in a second file, `model.onnx.data`. That file is not a
   description of the model: it is the weights themselves, as raw numbers. The `.onnx` then holds
   the graph and, for each such tensor, the second file's name, an offset and a length.
   - **What the model's author does:** one more argument when exporting, `external_data=False`;
     then one file. The refusal says so, and the guide leads with it.
   - **Not chosen now:** taking the second file's bytes as an argument. It would be an addition,
     not a change. **To be looked at again in step 9,** where larger models arrive.
9. **An `Identity` of an initializer is followed back to it** (§5). When two tensors hold the
   same values, PyTorch's older exporter stores one and gives the other layers an `Identity` of
   it (`torchscript-shared`). All biases zero is enough, and that is how many
   reinforcement-learning libraries start a network.
   - **Cost:** 65 B as built (63 B measured on a copy first): the reader is 2,525 B, so its size
     line moved from 2,500 B to 2,600 B.
   - **Not chosen:** refusing those files.

### Open

Nothing.

**Not proposed:** reading one output of a graph with several (`torch-heads`). Its author exports
the policy alone, which is one line; the guide shows it.

### What the second round changed (2026-10-14)

Found by re-reading the first draft against the repository, the prototype and the `onnx` package.

**The new entry**
- **Its wiring was not listed.** It needs `build.js`, `package.json`, the README table, a guide
  and a test script (§12).
- **The reader imports types only.** The first draft called `mlpLayout`; in a separate entry that
  copies it (2,453 B against 1,946 B). `MlpBrain` validates anyway (§7).

**Faults in the prototype that the design now rules out** (§6)
- **A varint longer than 10 bytes** could turn a length into NaN, and the reader would then stop
  silently with the rest of the file unread.
- **A graph that loops** would never end.
- **Names from the file were used as object keys.**
- **Every tensor was copied before the graph was checked,** so a large model of another kind
  would be copied whole before being refused.

**The subset, stated more exactly** (§5, §11)
- the bias shapes that are taken; `MatMul(W, x)` refused; `Add` and activations out of place;
- operator sets judged per node, with the floor of 7 checked against `onnx` (fact 19);
- a field left out of an attribute is zero; a missing operator set is refused;
- `ArrayBuffer` is accepted as well as `Uint8Array`.

**The tests**
- **U30's first wording was wrong.** A file cut exactly at a field boundary is valid protobuf, so
  "every prefix is refused" cannot hold. It now says which prefixes may be read.
- **U31 checks weights to the bit,** not outputs only, and refusals on real files.

**The fixtures** (§9)
- **Kept as base64 in JSON,** not as binary files: with `autocrlf` on and no `.gitattributes`, a
  tiny binary can be altered on checkout (fact 18). They move from `tools/agents/fixtures/` to
  `test/onnx-fixtures/`: they test the library, not the agent tools.
- **Three judges at export time,** and the framework's own weights recorded.
- **Exporters may write local file paths into a file;** those are removed before storing.
- **Two more cases:** an actor-critic shape (`torch-heads`), because reinforcement-learning
  libraries export a policy with its value estimate; and a large network, because weights may
  then be stored outside the file.

**The environment**
- **`pip` in `venv/` could not verify any download:** Avast's HTTPS scanning re-signs the
  connection with its own root, which Windows trusts and `pip` does not. Fixed by giving `pip` a
  certificate file that adds that one root (`venv/pip-ca-bundle.pem`); verification stays on.
  Recorded in `gotchas.md`.

## Phase 8.0 results — 2026-10-14: what exporters write

**Built** (no library code; `src/` untouched; not committed):
- `test/onnx-fixtures/make.py`: makes fourteen small networks with four makers and records each,
  as §9 says. **Two more were added the same day** ("Two more files", below): sixteen now.
- `test/onnx-fixtures/fixtures.json`: 104,303 B of text for the fourteen, whose files are
  32,077 B together; 127,818 B and 39,410 B with the two more.
- `test/onnx-fixtures/requirements.txt`: the versions used (fact 21).
- **Two runs wrote the same `fixtures.json`, byte for byte.**

### What each maker writes for a dense network

| Made with | The model | Nodes | Weights | IR, opset |
|---|---|---|---|---|
| PyTorch 2.14.1, default exporter | a batch, `[N, 5]` | `Gemm`, with `transA` 0, `transB` 1, `alpha` 1 and `beta` 1 all written out; then the activation | `[output, input]`, raw bytes | 10, 20 |
| the same | no bias | `Gemm` with two inputs | the same | the same |
| the same | one vector, `[5]` | `MatMul`, then `Add(result, bias)` | `[input, output]`, raw bytes | the same |
| the same | `nn.Flatten()` first | the `Flatten` is gone: as the first row | the same | the same |
| PyTorch 2.14.1, older exporter (`dynamo=False`) | a batch | `Gemm`, with `alpha`, `beta` and `transB` 1 | `[output, input]`, raw bytes | 9, 20 |
| the same | no bias | **`MatMul` alone** | `[input, output]`, raw bytes | the same |
| the same | one vector | `MatMul`, then **`Add(bias, result)`** | the same | the same |
| the same | `nn.Flatten()` first | **the `Flatten` stays** (`axis` 1), then as a batch | the same | the same |
| scikit-learn 1.7.2 `MLPRegressor`, through skl2onnx 1.20.0 | any | **`Cast` to float32**, then `MatMul`, `Add` and the activation per layer, then **`Reshape` to `[-1, 1]`** | `[input, output]`, a float list; biases are `[1, output]` | 10, 21 |
| `onnx.helper`, by hand | a batch | `Gemm` with no `transB`; `Gemm` with two inputs | a float list | 6, 11 |

- **Every form of §5 that an exporter writes was in the draft:** `Gemm` with `transB` 1, `MatMul`
  with `Add` in either order, `MatMul` alone, a `Gemm` without a bias, `[1, output]` biases, raw
  bytes and float lists.
- **`transB` 0 came from no exporter.** Only the hand-built file has it. It stays in the subset:
  it is ONNX's default.
- **Not in the draft: `Cast` and `Reshape` (scikit-learn) and `Flatten` (PyTorch's older
  exporter).** §14, item 4.
- **scikit-learn's regressors have that shape whatever the activation or the number of outputs**
  (checked: `tanh`, `relu`, `logistic`, `identity`; one, two and three outputs; one and two hidden
  layers). Its classifier is not a chain (`Softmax`, `ArgMax`, `ZipMap`) and stays refused.

### The largest finding: PyTorch's default call puts the weights outside the file

`torch.onnx.export(model, (x,), "model.onnx")`, with no other argument:

| A weight tensor of | Stored |
|---|---|
| 63 or 64 floats (up to 256 B) | inside `model.onnx` |
| 65 floats (260 B) or more | in `model.onnx.data` |

- **For 24→64→64→64→2:** `model.onnx` is 10,518 B and holds the graph and the biases;
  `model.onnx.data` is 39,424 B and holds all four weight matrices.
- **So nearly every real network exported the obvious way arrives as two files,** and the `.onnx`
  alone cannot be read. It is the first refusal a PyTorch user would meet.
- **One file comes from any of:**
  - `torch.onnx.export(model, (x,), path, external_data=False)`: 49,713 B;
  - `torch.onnx.export(model, (x,)).save(path)`: 49,713 B;
  - the older exporter, `dynamo=False`: 41,125 B.
- **An empty `model.onnx.data` is written beside every small model** too.

§14, item 8.

### What the reader would do with the fourteen files

Measured with the second prototype (outside the repository), which is written to this revision.

| Result | Files |
|---|---|
| **Read.** The weights equal the maker's bit for bit; `mlpForward` is within 4.2e-7 of the maker's outputs | 9: `torch-chain`, `-acts`, `-nobias`, `-vector`, `-flatten`; `torchscript-chain`, `-vector`, `-nobias`; `helper-gemm0` |
| **Read only if the end nodes are passed over** (§14, item 4, since decided); then also bit for bit, within 1.8e-7 | 2: `sklearn-mlp`, `torchscript-flatten` |
| **Refused, as designed,** each with its own message | 3: `torch-double` (float64), `torch-heads` (two outputs), `torch-outside` (weights outside the file) |

- **The first prototype refused `sklearn-mlp` for the wrong reason:** `initializer "shape_tensor"
  has data type 7`. It decoded every tensor before looking at the graph. The second round's rule
  (§6: check the graph, then decode only what a layer uses) names the `Cast` node instead.
- **The Herd & Wolves files** still read back bit for bit, in about 2 ms each, from a
  `Uint8Array` and from an `ArrayBuffer`.

### The second prototype under random damage

- **20,000 files with one random byte changed:** about half still read; every other one was
  refused with a message of §11. **None threw anything else,** and the slowest took 3.3 ms.
- **Every strict prefix of `torchscript-chain`** (697 of them) was refused by name: 692 as cut
  short, 4 for having no graph, 1 for having no operator set.
- **Its size:** 2,332 B min+gzip as its own bundle; 2,400 B with the end nodes passed over (§12).

### The judges

- **The maker, `onnxruntime` 1.23.2 and `onnx`'s reference evaluator agree within 2.7e-7** on
  every file that can be run.
- **ONNX Runtime Web 1.29.0, the judge A28 will use, loads all twelve files that can be run**
  (IR versions 6, 9 and 10; opsets 11, 20 and 21) and gives the maker's outputs within 2.4e-7.
  Checked on its CPU provider under Node.
- **It cannot run two files, of the same two kinds the reader refuses:** `torch-double` (it has
  no float64 `Gemm`) and `torch-outside` (the second file is missing).

### Also seen

- **PyTorch's default exporter writes the author's file paths into the file:** a stack trace per
  node, in the node's metadata. The fixtures have those entries removed, one per node, and each
  record says so; the rest of the metadata is kept, so the files still hold fields the reader
  must skip. The guide should warn anyone who publishes a model.
- **A tiny file can hold no zero byte.** `torchscript-nobias` is 217 B with none, and with 13
  line-feed bytes: git here would have taken it for text and changed it on checkout (fact 18).
- **On Windows, `torch.onnx.export` fails when the console is not UTF-8.** It prints a check mark
  that `cp1252` cannot encode, and the export stops with `UnicodeEncodeError`. `make.py` sets
  UTF-8 output first. In `gotchas.md`.
- **`onnx.checker` refuses `torch-outside`** without its second file, as it should.
- **Names:** the default exporter names the input after `forward`'s parameter (`input`, `obs`) and
  the output after its last operation (`linear_1`, `tanh`). The older one takes `input_names` and
  `output_names`.
- **Sizes:** the default exporter's files are 1.3 to 6.9 KB for these networks, mostly node
  metadata. The older exporter's are 217 to 775 B.

### Differs from §9's plan

- **Fourteen cases, not eleven:** `torch-outside` and `torchscript-flatten` were added when the
  first run showed what to ask.
- **The files are larger than the user was first told** ("a few hundred bytes to about 2 KB
  each"): the default exporter's node metadata. 32 KB of files, 104 KB of JSON with their records.
- **What the reader should do with each file is stated by the test,** not stored with the file.

### Two more files: tensors that hold equal values

Added on 2026-10-14, after the user's word on the end nodes. One network through both PyTorch
exporters: 5→7→7→7→3 `tanh`, every bias zero (as many reinforcement-learning libraries start a
network), and two equal 7×7 weight matrices. The fourteen earlier files did not change by a byte.

| File | What the exporter wrote |
|---|---|
| `torch-shared` (default exporter) | The zero biases are gone: each `Gemm` has two inputs. **The two equal matrices are stored once,** and two `Gemm` nodes read that one tensor |
| `torchscript-shared` (older exporter) | Each equal tensor is stored once, and **the other layers read an `Identity` of it:** `Identity(0.bias) → 2.bias`, `Identity(0.bias) → 4.bias`, `Identity(2.weight) → 4.weight` |

- **A layer may share its weights with another** in a real file. The reader gives each layer its
  own copy, and limits what the copies may add up to (§6).
- **The older exporter's file was refused** as the design then stood: a bias that is an `Identity`
  of an initializer is not an initializer. The user has since decided that such an `Identity` is
  followed (§14, item 9), and the file is read.
- **Both files run in `onnxruntime`** and agree with PyTorch within 6.0e-8.

### Not done

- **TensorFlow and Keras,** and any real reinforcement-learning library (§14, item 7).

## Phase 8.1 results — 2026-10-14: the entry and the reader

**Built:**
- `src/extras/inference/index.ts` and `mlpOnnx.ts` (226 lines): `mlpFromOnnx(bytes)` and the
  `OnnxMlp` type. It imports types only from the agents code.
- `build.js` and `package.json`: the entry `null-graph/inference`, and `npm run test:inference`.
- `test/inference-onnx.test.mjs`: seven tests. `test/onnx-fixtures/build.mjs`: graphs encoded by
  hand for them, on the writer's encoding helpers, which `tools/agents/onnx.mjs` now exports.
- `test/spikes/mlp-think-cost/size.mjs`: two lines for the new entry, and a check that no agents
  bundle holds the reader.

**As decided** (§14):
- weights outside the file are refused, with the fix named (item 8);
- a `Cast` or a `Flatten` before the first layer and a `Reshape` after the last are passed over
  (item 4);
- an `Identity` of an initializer is followed back to it (item 9);
- no refusal mentions `OrtBrain` (item 5).

**Tests** (`npm run test:inference`, 7 of 7):

| Test | What it showed |
|---|---|
| The entry | It exports `mlpFromOnnx` alone, and the files `package.json` names exist |
| U12 | Five layers written by `tools/agents/onnx.mjs` come back bit for bit, a `-0` included, with `obs`, `act` and the metadata; also from an `ArrayBuffer`, from a view at an odd offset, and after the file is wiped. `mlpLayout` then refuses the five layers: the limit is `MlpBrain`'s |
| U29 | 23 forms of one three-layer network give it back bit for bit; the 23 files are all different bytes. The end nodes and an `Identity` are refused between two layers, and so are a `Cast` to float64 and a `Flatten` on axis 0 |
| U30 | Of 718 prefixes of one file, 715 are refused by name and 3 read, exactly the three cut after the operator set. Of 10,000 files with one byte changed, 5,795 read and 4,205 are refused by name; **none throws anything else;** the slowest took 0.8 ms. The named cases of §10 hold, and two `Identity` nodes that feed each other are refused at once |
| U31 | **13 files from other makers read with their makers' weights bit for bit,** and `mlpForward` is within 4.2e-7 of the makers' outputs. 3 are refused with the stated messages: `torch-double`, `torch-heads`, `torch-outside` |
| U14 | 39 wrong files or arguments are each refused with their message. Among them: a web page and a `.mlp.json` file in place of the model, and a promise that was not awaited |
| Size | Below |

**Watched failing:** 34 mutations of `mlpOnnx.ts`, each rebuilt and run; **34 of 34 were caught.**
Among them: no transpose; `Gemm` ignoring `transB`; the bias always taken as `Add`'s second
operand; no limit on a varint; no check of a length against the bytes left; no notice of a loop;
no limit on reused tensors; any `Cast` passed over; an `Identity` not followed; layers sharing one
decoded array.
- **A loop cannot hang the reader even without its check:** with that check removed, the limit on
  reused tensors ended the walk, and the test failed on the message instead of running forever.
- **One mutation was caught by a hang:** with no limit on the `Identity` nodes followed, two that
  feed each other never end. The run was stopped after 20 s, and no process was left behind.

**Sizes** (min+gzip, `size.mjs`'s settings):

| | |
|---|---:|
| The reader as its own bundle (line 2,600 B) | 2,525 B |
| Whole agents entry, before and after | 19,323 B and 19,323 B |
| An app with rules and `MlpBrain` | 15,517 B |
| The same app with `mlpFromOnnx` | 17,840 B |

- The reader holds none of `MlpBrain`'s code, and the agents entry none of the reader.
- An import of the entry that uses nothing leaves no bytes behind.

**Also green after the change:** the headless agents suite (reference + 73), agent-kit (4 of 4),
core headless, and the `onnx` GPU suite on Node/NVIDIA (5 of 5). The library type-checks
(`npm run build:lib`).

**Differs from the design as it stood before this phase:**
- **A limit on reused tensors** (§6, §11), after `torch-shared` showed that real files share
  tensors: without a limit, the reader's memory was not bounded by the file.
- **An `Identity` of an initializer is followed,** and the size line is 2,600 B, after
  `torchscript-shared` (§14, item 9). The first build of this phase refused that file at 2,460 B.
- **U12 reads five layers from one file,** not from two: the reader has no layer limit, and the
  test shows where the limit is.
- **The size line is a test,** not only a line in `size.mjs`.
- **Module-level constants are marked pure,** so that an app importing the entry for something
  else carries none of the reader. Step 9 adds more to this entry.

**Not done, by plan:**
- **The README row, `docs/api/inference.md` and the agents guide's "As ONNX" line:** phase 8.3.
  Until then the guide still says that ONNX is not read.
- **A28,** the comparison with ONNX Runtime on the GPU: phase 8.2.
- **Not run:** the GPU suites in Chrome and on the AMD GPU; they are untouched by this phase.

**Next:** phase 8.2.

## Phase 8.2 results — 2026-10-14: the reader, judged by ONNX Runtime

**Built:** A28, in `test/onnx-gpu/run.js` (the `onnx` suite, now 23 checks). No library change.

**What A28 does.** One file's bytes go two ways:
- `mlpFromOnnx` reads them into an `MlpBrain`, which runs as the think stage of a small
  `AgentSystem` (256 agents) on the suite's device;
- ONNX Runtime's WebGPU provider runs the same bytes as they are, on the observations that
  system's own sense stage wrote: signed values from a hash, as the agents suite's A4 senses them.

The two must give the same actions within 1e-4.

**The 17 files:**
- the two Herd & Wolves networks as `tools/agents/onnx.mjs` writes them (24→64→64→64→2);
- each rewritten with `MatMul` and `Add`: one as `Add(result, bias)`, one as `Add(bias, result)`
  on float lists;
- the 13 fixtures the reader takes: PyTorch's two exporters, scikit-learn, `onnx.helper`.

A first check states the coverage: the reader refuses exactly `torch-double`, `torch-heads` and
`torch-outside`, and nothing else.

**Results:**

| | Node/Dawn NVIDIA | Node/Dawn AMD | Chrome AMD |
|---|---:|---:|---:|
| Checks, the whole suite | 23 of 23 | 23 of 23 | 23 of 23 |
| GPU errors | 0 | 0 | 0 |
| Worst difference between `MlpBrain` and ONNX Runtime, any file | 4.5e-6 | 3.0e-6 | 3.0e-6 |
| The same, among the 13 fixtures | 7.2e-7 | 6.0e-7 | 6.0e-7 |
| The suite's time | 5.8 s | 4.5 s | about 6 s |

- **The largest differences are the Herd & Wolves networks',** whose trained weights reach 178;
  7.9 measured 4.4e-5 between the GPU and `mlpForward` for them. The fixtures' weights are of
  unit scale.
- **On the AMD GPU the two files without a bias agree to the last bit.**

**Watched failing** (Node/NVIDIA; the reader mutated, `dist/inference.js` rebuilt, the suite run):

| The reader is broken so that | A28 checks that fail, of 18 |
|---|---|
| a square layer is read the other way round, every other layer as before | 6: exactly the files with a square layer (the four Herd & Wolves files, `torch-shared`, `torchscript-shared`) |
| no weight matrix is transposed | 7: every file in a form that needs it |
| the activation is dropped | 15 |
| the bias is dropped | 13: every file that has one |
| `Add`'s second operand is always the bias | 1: the coverage check, since two files are then refused |

- **The first row is what a round trip cannot see:** every shape is right, and the writer's own
  files fail only because a judge that shares nothing with the reader ran them.

**Differs from the design:**
- **One system and one brain per network shape** (five shapes), with `setWeights` between files:
  one pipeline compile per shape, not per file.
- **A file with a fixed batch of one, or with no batch dimension, is run row by row on 32
  observations,** not 256: eight files. The others take 256 at once.
- **A file that ONNX Runtime cannot load or run fails its check.** The design said "reported as
  not judged"; no file needed it.
- **The observations are hashed values in [−1, 1),** not the simulation's, so that every file is
  judged the same way. A27 keeps the simulation's observations for the two trained networks.

**Not run:** the other GPU suites; this phase changes one test file.

**Next:** phase 8.3.
