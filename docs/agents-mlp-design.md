# NullGraph Agents — C2 step 6: MlpBrain — Technical Design

**Parent:** [`agents-design.md`](./agents-design.md) §9.3 and §13, C2 step 6.
**Builds on:** [`agents-gpu-loop-design.md`](./agents-gpu-loop-design.md), the production brain
seam and S3 measurements; [`api/agents.md`](./api/agents.md).
**Status:** ✅ **Step 6 complete, 2026-10-12.** Phases 6.0 `e345921`, 6.1 `dbaf884` and 6.2
`b3424b1` are committed; 6.3 (guide example, cost and size report, closure) is implemented.
A4 passes on Node/Dawn and Chrome; the cost script adds NVIDIA agreement.
**Target:** `D:/orion-realms/null-graph/src/extras/agents/`, focused headless and real-GPU tests,
and API documentation. Core unchanged. No new dependency.
**Outcome:** run a supplied small dense network as the existing synchronous GPU think stage,
with a matching CPU forward pass and trustworthy correctness/lifecycle checks.

## 1. What changes—and what does not

Both brains use the same pipeline:

**grid/query → sense → think → act → app stages → tick → instances**.

Sense creates observations; think produces action values; act applies steering, safety limits and
integration. MlpBrain replaces only the observation-to-action calculation inside think.

| | RuleBrain today | MlpBrain in step 6 |
|---|---|---|
| Policy representation | Handwritten WGSL conditions and formulas | Dense-layer weights, biases and activations |
| Example | If a wolf is near, turn away | Evaluate the supplied network on sensed values |
| How behaviour changes | Edit rules or their parameters | Supply/update weights; a new shape requires a new brain/stage |
| Training | None | None at runtime; step 7 trains weights offline |
| GPU scheduling | Synchronous stage in the engine submission | Same stage/scheduling and cadence |
| Interpretability | Can emit explicit chase/flee/rest reason codes | Numerical outputs do not inherently explain a decision |
| Capability guarantee | Whatever the rules implement | Whatever the supplied weights encode; not automatically better |

An MLP is a multilayer perceptron: repeated affine transforms plus activations. It is not an LLM,
planner, autonomous learning loop or replacement for sense/act. Random weights produce arbitrary
behaviour. Step 6 proves inference, not that a network has learned Herd & Wolves.

## 2. Scope and dependencies

Implement network validation/layout, CPU forward, weight packing, generated WGSL, MlpBrain
ownership/update API, U10/A4 and integration regressions. Reuse AgentSystem.setBrain and its
block-staggered/newborn/dead gating. Existing buffers/observations/actions remain fixed for life.
The full step-5 scaling study is not a prerequisite and remains deferred, not complete.

Out of scope: training/dataset collection (7), ONNX parsing/export (7–8), ORT/async scheduling
(9), learned demo comparisons/cloning quality (10), per-agent genomes, recurrent networks,
automatic normalization, quantization and per-species brain routing.

## 3. Proposed public contract

These names/signatures are proposed, not existing APIs. Align final implementation with project
conventions and publish only after validation.

- `MlpActivation = 'identity' | 'relu' | 'tanh' | 'sigmoid'`.
- `MlpLayer`: input/output sizes, row-major Float32Array weights, Float32Array biases, activation.
- `MlpNetwork`: ordered readonly layer descriptions.
- `new MlpBrain(engine, network)`: validate/copy parameters, allocate/upload one owned weights
  buffer and generate a think contribution. Constructor requires the engine/device owning the
  target AgentSystem; buffers from another device are not portable.
- `brain.kind = 'stage'`; `brain.dims` declares first input and last output widths;
  `brain.wgsl` defines the required `think(i: u32, species: u32, tick: u32)`;
  `brain.buffers` contains the single owned read-only weights buffer.
- `brain.setWeights(parameters)`: same shape/activations only; validate the entire update before
  copying/uploading, preserve GPU buffer identity and compiled WGSL, one queue.writeBuffer.
  Structural changes require a new brain plus agents.setBrain, not hidden pipeline rebuilds.
- `brain.destroy()`: idempotently release its own buffer; updates after destruction fail.
- `mlpForward(network, input, out?, scratch?)`: CPU reference utility; supplied output and
  reusable scratch support allocation-free repeated evaluation. Input/output aliasing must be
  explicitly handled or refused; no mutable module-global scratch.

Use Float32Array view boundaries, not whole underlying ArrayBuffers. Copy caller data at setup
and update, so later source-array mutation cannot silently change CPU/GPU parameters. Validation
errors name the layer/field and expected shape. CPU helper data must be the same logical model,
not a separately hand-packed layout that can drift from the GPU contract.

### 3.1 Shape and parameter validation

Parent limits: at most four **affine layers** (input width does not count as a layer), hidden
width <=64, input width <=128 and output width <=16. Require positive integer widths, at least
one layer, adjacent matching widths, supported activations, exact weights/bias lengths and finite
f32 parameter values. Reject invalid models before allocating resources/registering stages.
Support one affine layer, odd widths, no hidden layer and all four activation types.
Do not silently clip unsupported networks to fit limits. Enforce device buffer limits too.

AgentSystem's existing dimension check refuses incompatible obs/act widths. Output order is the
layout's action order; input order is its observation order. Semantic compatibility is the user's
responsibility: matching dimensions alone cannot validate feature meaning or normalization.

## 4. Mathematics and memory

For output row r of each layer:

`z[r] = bias[r] + sum(weights[r * inputWidth + k] * x[k])`

`y[r] = activation(z[r])`.

Use row-major [output,input] logical weights. Define offsets/packing once in a shared layout helper.
Baseline GPU representation follows S3: vec4-packed input rows and biases in one storage buffer,
zero padding to whole vec4 groups. Padded lanes must be zero, never borrowed from the next agent
or layer. Publish/logical conversion rules for step-7 tooling; padding is internal, not model data.
For 24→32→32→2 the logical count is 1,922 f32 parameters (7,688 bytes); padded GPU storage may
be larger and must be reported separately. One shared model serves all agents, not one per agent.

Generate fixed shapes/offsets into WGSL with loops and bounded private intermediates. Reuse two
scratch regions or per-layer arrays with clear ownership. Read observations only within OBS_DIM;
write precisely ACT_DIM outputs. No GPU-to-CPU transfer is required for ordinary inference.

### 4.1 Activation safety and numerical contract

- identity: z; relu: max(z,0).
- tanh: clamp its input to [-15,15] before evaluating, following the measured S3 FXC issue.
- sigmoid: use a sign-stable expression to avoid unnecessary exponential overflow.

CPU and GPU implement the same activation definitions. CPU validation includes a high-precision
independent oracle and an f32-aware implementation, not just a copy of generated code.
GPU dot/FMA/transcendental results are not promised bit-identical to JS arithmetic or across GPUs.
Finite weights/inputs can still overflow intermediate sums. Do not hide that by arbitrarily
clipping identity/relu outputs: document the supported numerical operating range and preserve
existing act-stage handling of non-finite actions. Include extreme-input safety tests separately
from ordinary finite-range numerical-agreement tests.

Parent A4's relative-only 1e-5 bound is ill-defined near zero. Proposed step-6 refinement:
`abs(gpu - cpu) <= 1e-5 + 1e-5 * abs(cpu)` on declared finite fixtures. Report maximum absolute
and relative errors on both targets and retain S3's observed errors as historical—not a guarantee.
If a supported ordinary fixture fails, diagnose it; do not silently relax the tolerance.

## 5. Integration, lifecycle and inspection

WGSL weights use group 1 binding 0; AgentSystem retains group 0 and constructs the full stage.
No workgroup barriers/staging: the production entry returns early for capacity/dead/cadence gates.
The storage-loop baseline was selected by S3; hardware-specific alternatives are later measured
optimizations, not required step-6 features.

No changes to sense, act, app stages, tick, instances, core pipeline or synchronous frame scheduling
are planned. setWeights is an explicit caller operation, not an automatic per-frame upload.
Queue order defines when an update takes effect; it cannot retroactively change submitted work.

AgentSystem does not own/destroy a brain. Replace/detach its consuming stage or destroy the system
before destroying an attached MlpBrain. Retired stages must not access released weights, including
cached sequences; verify existing replacement/teardown behaviour on real GPUs. Constructor/update
failures must not leak resources or partially replace valid weights. Cross-device attachment needs
an actionable refusal: use a narrowly scoped ownership check if existing plumbing cannot express it,
and document that amendment rather than redesigning the Brain interface broadly.

MlpBrain should set a documented neutral reason code (proposed 0) for evaluated rows so rule reason
codes do not linger after replacement. Do not label network outputs 'fleeing' or 'chasing' without a
separately validated explanation. Skipped rows retain previous actions/reasons under existing cadence
semantics; paused brain replacement does not itself dispatch inference or refresh outputs.
Inspection can still show observations/actions and agent state; policy rationale is not guaranteed.

### 5.1 Species and teacher-policy limitation

The entry passes species/tick, but the baseline network consumes only declared observations.
Do not secretly append these fields or change obsDim. Identical observation vectors produce the
same outputs, even for different species. Thus one network cannot generally copy distinct prey/wolf
policies without a species feature, separate models/routing or another explicit encoding.
For step 6 use controlled fixtures/single-policy examples. Before step-7 cloning, design and validate
species encoding/routing and any needed rest/time state; do not promise teacher equivalence here.

## 6. Tests and acceptance

| Coverage | Required evidence |
|---|---|
| U10 mathematics | Hand-computable affine/two-layer fixtures, all activations, zero/bias-only/signed values, independent oracle |
| Layout | Odd widths, padding, offsets, byteOffset views, no cross-row reads/writes, maximum supported shape |
| Validation | Each limit/error, shape/finite-parameter failures before allocation, transactional failed update |
| WGSL/U9 | Full generated think parses, signature/binding/shape constants correct, dimension mismatch refused |
| CPU reuse | Output/scratch reuse, documented aliasing, repeated deterministic evaluations without hidden allocations |
| A4 GPU | Production AgentSystem execution matches CPU for affine, odd-size, mixed-activation and 24→32→32→2 fixtures on both hosts |
| Gating | Capacity tail, dead agents, cadence-skip sentinels, newborn first evaluation and block staggering |
| Updates | Same buffer/stage identities; new parameters change outputs as expected; failed update preserves prior results |
| Replacement | Rules→MLP→rules, neutral reasons, app/act/instances unchanged, no lingering retired-stage access |
| Lifecycle | Idempotent destroy, use-after-destroy refusal, dependent teardown/rebuild, zero GPU validation errors |

Use focused new headless tests (proposed `test/agents-mlp.test.mjs`) in existing test:agents, with
the existing mock-device/esbuild/WgslReflect conventions. Add A4 to the shared agents GPU suite,
not a benchmark-only S3 variant. Keep real-step act sanitization and existing replay regressions.
Do not claim cloning quality, held-out accuracy or cross-device byte-identical replay from A4.

Validate library/TypeScript and IDE build; agents/core and affected agent-kit/API example checks;
Node/Dawn and Chrome GPU runs where available. Missing hardware is reported as an unverified gate.
Provide a small known-weight executable usage example, not an untrained neural herd presented as
learned behaviour. Record test totals, numerical errors, commands and hardware in the handoff.

## 7. Bundle and performance boundaries

The 527 B remaining applies to C1, not all C2 additions. The parent allows **at most +4 KB min+gzip
for C2 MlpBrain plus fromOnnx combined**. Conservatively interpret as 4,000 B and measure a fresh
C1 baseline before implementation; reserve headroom for step 8 instead of spending it all now.
Report whole-entry, tree-shaken rule path, tree-shaken MLP path and parameter buffer sizes.
Rule-only imports should tree-shake neural code; core growth remains 0 B. No ORT import/download.

Report representative inference costs when practical, separately from full simulation timings,
without step-5 ceiling sweeps or population/millisecond success targets. Step 6's primary gate is
correct integration, not speed versus RuleBrain: their policies and computational work differ.

## 8. Build phases

| Phase   | Work                                              | Exit                                                                       |
|---------|---------------------------------------------------|----------------------------------------------------------------------------|
| **6.0** | Model/layout, validation and CPU forward          | U10 fixtures/packing/errors and CPU reuse verified                         |
| **6.1** | WGSL generator and owned weight buffer/update API | Parsing, dimensions, mock integration, ownership/update tests              |
| **6.2** | Production GPU integration and A4                 | Both-host numerical/gating/update/replacement/lifecycle checks             |
| **6.3** | Executable guide, size/timing report and closure  | Fresh regression/build checks, budget evidence and updated roadmap/handoff |

Each phase preserves existing uncommitted work. No commits, pushes or publishing unless requested.
Step 7 can then record teacher data and train networks; validate neighbour coverage/replay at the
chosen teacher workload and resolve policy input semantics first. Steps 8–10 remain separately scoped.

### Phase 6.0 implementation

`src/extras/agents/mlpMath.ts` exports `MlpLayer` (`input`, `output`, `weights`, `biases`,
`activation`), `MlpNetwork` (readonly layer array), `mlpLayout`, `packMlpWeights` and `mlpForward`.
Scratch is one Float32Array of at least twice `layout.scratchWidth`; output has exactly final width.
Overlapping input/output/scratch views and writable views overlapping model parameters are refused.
Parameters are borrowed for these pure helpers, not copied; future MlpBrain construction owns a copy.
Forward rounds each multiply/add and activation output to f32, without claiming GPU bit equality.
Tanh is clamped; sigmoid uses a sign-stable expression. Runtime inputs are not sanitized.
Packing honors view byte offsets and zeroes padded lanes. 24→32→32→2 has 7,688 logical parameter
bytes and 7,952 packed bytes. No GPU resource, scheduling or core changes in this phase.

Fresh pre-change whole-entry baseline: **15,495 B min+gzip** (esbuild browser ESM, default target,
gzipSync defaults), not the historical 15,473 B. Use the same measurement settings for C2 deltas.
U10 tests were observed failing before implementation. Phase 6.1 is next; A4 is not yet implemented.
Final phase-6.0 entry size: **16,569 B min+gzip**, +1,074 B over that baseline. Remaining combined
C2 allowance: 2,926 B. Focused U10 tests pass 8/8; builds and existing agents/core/agent-kit checks
pass. GPU numerical/integration checks remain outstanding, not inferred from CPU test success.

### Phase 6.1 implementation

Phase 6.0 committed as `e345921`. `MlpBrain.ts` implements the constructor, same-shape
`setWeights(network)` and idempotent destroy; `mlpShaders.ts` generates the think contribution.
One copied/packed GPU storage buffer is uploaded at construction and once per explicit update.
The class retains copied shape metadata, not caller arrays or a redundant CPU model copy.
Malformed models, device limits and incompatible updates are refused; initial upload failure
destroys its newly allocated buffer. Caller-side parameter mutations do not affect uploaded data.
GPU padding/intermediates are zero-initialized; observations/actions touch only logical widths.
All activations, odd shapes and the maximum shape parse in the full production think wrapper.

`Brain.assertDevice?` is the narrow optional resource-owner seam: AgentSystem calls it before
constructing a stage. MlpBrain rejects another device and attachment/update after destruction;
RuleBrain remains unchanged. Caller must still detach or destroy consumers before brain.destroy.
Evaluated rows receive reason 0, not a claimed explanation; shared cadence gating stays in the
production wrapper. Same-shape updates do not rebuild stages or cached chains.

Tests: four added phase-6.1 cases were observed failing before implementation; focused suite
12/12 passes, full agents run passes 24 plain + 47 node:test checks, core 17 and agent-kit 4/4.
Library/TypeScript and IDE builds pass. Existing Node/Dawn GPU regression suite passes 106/106,
zero WebGPU errors, AMD GCN-5 D3D12 driver 31.0.12002.1002. That suite does not yet run MlpBrain;
A4 production numerical/update/gating checks and Chrome MLP execution remain phase 6.2.

Whole-entry size is 17,283 B min+gzip (+1,788 B versus fresh C1; 2,212 B combined allowance left).
Representative tree-shaken exports of AgentSystem/defineAgents/Sensors/Actions plus RuleBrain
measure 12,369 B, versus 13,761 B with MlpBrain. These are named-export bundles, not frame-path
size claims. No core changes, new dependencies, training, ONNX, site changes or publication.

### Phase 6.2 implementation

**No library source changed:** A4 found nothing to fix. `test/agents-gpu/run.js` adds 14 checks to
the shared agents GPU suite (106 → **120**). Every numerical check runs the production chain:
sense writes each observation row, the generated think evaluates it, and each live agent's action
row is compared with `mlpForward` applied to the observation row the GPU itself wrote, under
`|gpu − cpu| ≤ 1e-5 + 1e-5·|cpu|`. Inputs come from a custom sensor that hashes (agent, field,
tick) into [−1, 1), or from Herd & Wolves' real sensors. Relative error is reported only where
|cpu| ≥ 1e-3: on near-zero outputs it is noise, and the absolute term covers them.

| Fixture | Live rows × actions | Max abs error, Node / Chrome | Max rel error, Node / Chrome |
|---|---|---|---|
| One affine layer, 3→2 identity | 95 × 2 | 1.19e-7 / 1.19e-7 | 7.77e-7 / 7.77e-7 |
| Odd widths, 5→7→3 (relu, sigmoid) | 95 × 3 | 5.96e-8 / 5.96e-8 | 1.75e-7 / 1.75e-7 |
| Four layers, 9→13→6→11→5 (all four activations) | 95 × 5 | 5.96e-8 / 5.96e-8 | 3.02e-7 / 3.02e-7 |
| Maximum shape, 128→64→64→64→16 | 95 × 16 | 2.09e-7 / 2.09e-7 | 7.96e-7 / 7.96e-7 |
| Herd & Wolves 24→32→32→2 tanh, 2,000 agents, 20 moving steps | 1,913 × 2 | 3.58e-7 / 2.98e-7 | 6.84e-6 / 6.77e-6 |

Every fixture has dead rows that keep a sentinel, and a partial last 64-agent block. All 24 of the
herd's observation fields are non-zero for some agent, so the comparison exercised every input. The
herd's neighbour lists depend on atomic scatter order, so its errors vary a little from run to run
(Node: 3.87e-7, 2.98e-7, 3.58e-7 in three runs); the hashed-input fixtures repeat exactly.

- **Extreme inputs (§4.1), separate from agreement:**
  - Tanh pre-activations reach |z| ≥ 89 in 95 of 95 agents and sigmoid saturates; every output is
    finite and in [0, 1]; the 90 fully saturated rows agree (max abs 4.17e-7).
  - Relu outputs up to 11,095 pass through unclipped. A sum of terms that cancel carries error
    proportional to the terms, not the result, so this check uses `1e-5 · (1 + |b| + Σ|w·x|)`.
- **Gating:** `thinkEvery` 3, capacity 200 (an 8-row tail block), 6 ticks. Exactly the due blocks
  and an off-cadence newborn are evaluated (396 rows, all agreeing). Every other row keeps its
  observations, actions and reason bit-exactly, and dead rows keep the sentinel until spawned.
- **Updates:** `setWeights` keeps the weights buffer, the cached chain and the think stage, and the
  next step uses the new weights. Editing the caller's arrays afterwards changes nothing. Three
  invalid updates (other activation, other depth, a NaN in the last bias) are refused, and the
  following step still matches the previous weights.
- **Replacement:** rules → MLP → rules on a system with an app stage and the instance stage.
  - Only the think stage changes. Sense, act, app, tick and instance stages are the same objects in
    the same order, and all run: an app counter and every agent's age count 6 steps, and the
    instance rows equal the CPU twin.
  - Reasons go 7 → 0 → 7. A swap while paused infers nothing.
  - After the detached MLP is destroyed, no cached chain (1–3 steps per frame) reaches a retired
    stage, and rendering them raises no validation error.
- **Lifecycle:** system-then-brain teardown and a second `destroy()` are clean. A destroyed brain
  and a brain from another device are refused (3/3) without disturbing the attached brain. A
  rebuilt system on the same engine matches.
- **Watched failing:**
  - In the suite: biases read from the neighbouring lane put 475 of 475 outputs out of tolerance.
  - Library mutations, each rebuilt, run on Node/Dawn and then restored byte for byte:
    - Tanh without the clamp fails only the extreme check on Node/Dawn, where FXC's `tanh` returns
      NaN. That confirms S3 on the production path. Chrome (DXC) still passes 120/120, so the clamp
      protects one compiler.
    - `setWeights` without the upload fails both update checks (outputs stay on the old weights,
      max abs 0.62).
    - No neutral reason fails 7 checks.
- **Hosts:** Node/Dawn and headless Chrome, both on the AMD GCN-5 iGPU (D3D12, driver
  31.0.12002.1002): two shader compilers, one GPU. NVIDIA was not exercised in this phase.
- **Totals:** 120/120 agents GPU checks on both hosts, zero WebGPU errors. Headless: agents 24
  reference + 47 node:test, core 17, agent-kit 4/4. Library build and whitespace pass. Tracked
  `dist/index.js` rebuilt byte-identical. Bundle unchanged at 17,283 B.

### Phase 6.3 implementation — step 6 closed

**API guide.** [`api/agents.md`](./api/agents.md) gains "Network brains: MlpBrain": a hand-set,
known-weight 4→2 tanh network that turns away from walls and slows when one is ahead. It is
presented as hand-set, not learned. It is followed by the contracts: shape and limits,
observation/action order, one network for every species, numerics, updates, swap cost, reasons,
ownership and cost. `test/agents-api-example.test.mjs` now validates **every** TypeScript block
in the guide:
- each is type-checked and bundled;
- the network example's bundle is run, and its documented CPU outputs (turn ≈ −0.905, throttle ≈
  0.762) and widths are checked;
- each example's generated think WGSL is parsed.

Watched failing: a flipped weight sign fails the number check, and `'softmax'` fails the type
check.

**Inference cost.** `npm run spike:mlp-think` (`test/spikes/mlp-think-cost/`, a measurement like
S3, not a suite) dispatches the think shader `AgentSystem` generates, gate and accessors included,
with S3's method: GPU timestamps, a 1 s warm-up, the lower of two medians of five. Before timing,
every kernel is checked against the CPU on 20,000 agents.

| µs per think dispatch | 1,000 | 20,000 | 100,000 | Pipeline compile | Max abs error |
|---|---|---|---|---|---|
| **NVIDIA Turing, Node/Dawn:** rule floor (2 reads, 2 writes) | 1.1 | 17.6 | 83.5 | 20 ms | 0 |
| MLP 24→32→32→2 tanh | 37.3 | 179.7 | 884.7 | 1,082 ms | 3.8e-7 |
| the same, `thinkEvery` 3 | 37.9 | 101.3 | 383.1 | 1,120 ms | 3.8e-7 |
| MLP 128→64→64→64→16 | 253.2 | 1,933.3 | 9,060.4 | 1,013 ms | 2.4e-7 |
| **AMD GCN-5, Node/Dawn:** rule floor | 1.3 | 133.2 | 655.4 | 12 ms | 0 |
| MLP 24→32→32→2 tanh | 12.6 | 184.0 | 827.4 | 183 ms | 3.7e-7 |
| the same, `thinkEvery` 3 | 12.6 | 111.8 | 470.7 | 176 ms | 3.0e-7 |
| MLP 128→64→64→64→16 | 258.3 | 4,522.0 | 21,848.1 | 1,641 ms | 2.1e-7 |
| **AMD GCN-5, Chrome:** rule floor | 1.9 | 132.3 | 627.6 | 17 ms | 0 |
| MLP 24→32→32→2 tanh | 28.9 | 231.8 | 861.6 | 76 ms | 3.6e-7 |
| the same, `thinkEvery` 3 | 31.1 | 147.3 | 489.5 | 80 ms | 3.3e-7 |
| MLP 128→64→64→64→16 | 429.2 | 6,030.9 | 28,797.9 | 221 ms | 2.4e-7 |

What the table says:
- **The production stage costs what S3's bare kernel cost.** At 100,000 agents the 24→32→32→2
  network takes 0.83–0.88 ms (8.3–8.8 ns per agent) on all three host/GPU pairs. S3 measured
  909 µs on the NVIDIA GPU, so the wrapper's gate and record read add nothing measurable there.
- **The AMD iGPU is bandwidth-bound.** Its stage floor alone, reading each 64-byte record and the
  observation row, is 6.3–6.6 ns per agent, against 0.8 ns on the NVIDIA GPU. The network
  therefore adds little on top of it on that machine.
- **`thinkEvery` 3** costs 43% of every agent thinking on the NVIDIA GPU, 57% on the AMD iGPU:
  the block stagger S3 predicted (32–42%), with the stage's fixed costs on top.
- **The largest network costs about 10× the herd network,** in line with its arithmetic (~17,400
  multiply-adds per agent against ~1,900).
- **Compiling is the cost to plan around.** `setBrain` builds a pipeline. `BatchManager` creates
  pipelines synchronously, so under Node/Dawn on the NVIDIA GPU the swap stalls the caller for
  about 1.1 s; in Chrome the GPU waits on first use instead (76 ms here). Same-shape `setWeights`
  never compiles.
- **NVIDIA numerics.** Dawn gives the agents GPU suite the AMD iGPU, so A4 never ran on NVIDIA.
  These checks did: the production MLP kernels agree with the CPU within 3.8e-7 there too, under
  A4's tolerance.

**Size** (`node test/spikes/mlp-think-cost/size.mjs`, phase 6.0's settings: esbuild browser ESM,
minified, gzipSync defaults):

| Import set | min | min+gzip |
|---|---|---|
| Whole `agents` entry | 54,171 B | 17,283 B |
| Rule path: `AgentSystem`, `defineAgents`, `Sensors`, `Actions`, `RuleBrain` | 40,501 B | 12,369 B |
| Rule path + `MlpBrain` | 44,999 B | 13,793 B |
| … + `mlpForward` | 45,942 B | 14,123 B |

- `MlpBrain` adds **1,424 B** to an app.
- The rule path carries **none** of the MLP's code (no WGSL, labels or errors); the script fails
  if it ever does.
- The whole entry grew +1,788 B over the fresh C1 baseline (15,495 B), leaving **2,212 B** of C2's
  +4,000 B allowance for step 8's `fromOnnx`.
- `src/core` is unchanged since before step 6 (0 B).
- Weights on the GPU: 24→32→32→2 is 7,688 B logical, 7,952 B packed; 128→64→64→64→16 is
  70,464 B (no padding).

**Regression at closure:**
- Library build (`node build.js && tsc`).
- Headless: agents 24 reference + 47 node:test (the API example included), core 17,
  agent-kit 4/4.
- Agents GPU suite 120/120 on Node/Dawn and in Chrome, zero WebGPU errors.
- `git diff --check`.

No library source changed in 6.3.

**Step 6 is complete.** It has two kinds of evidence:
- **Correctness:** A4 on the AMD iGPU through two compilers, and the cost script's agreement
  checks on the NVIDIA GPU.
- **Documentation:** the guide and its executable example.

**Carried forward, not done:**
- **Asynchronous pipeline creation for `setBrain`** (`createComputePipelineAsync`) would remove
  the swap stall before step 10's live Rules/Network switch. It changes core's stage creation, so
  it needs its own proposal.
- **The species/policy encoding** must be settled before step 7 records teacher data (§5.1).
- **Training, ONNX and ORT** stay steps 7–9.