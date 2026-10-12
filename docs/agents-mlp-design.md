# NullGraph Agents — C2 step 6: MlpBrain — Technical Design

**Parent:** [`agents-design.md`](./agents-design.md) §9.3 and §13, C2 step 6.
**Builds on:** [`agents-gpu-loop-design.md`](./agents-gpu-loop-design.md), the production brain
seam and S3 measurements; [`api/agents.md`](./api/agents.md).
**Status:** Phase 6.0 implemented, 2026-10-12. Remaining GPU/API phases are draft;
no MlpBrain class or new GPU results claimed.
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