# NullGraph — Per-Module Design Briefs

**Status:** planning document, companion to [`roadmap.md`](./roadmap.md). Non-binding except where
a linked spec exists.
**What a brief is:** a module's goal, the **real integration surface** it hangs off (named, in this
repository), its dependencies, and its explicit non-goals. A brief becomes a standalone spec when
its release is scheduled — the shape of that spec is fixed by
[`design-principles.md`](./design-principles.md) §14.

The point of writing the integration surface down before the code exists is that it is where plans
break. Every brief here names the function or file the module attaches to; when one of those stops
matching reality, the brief is wrong and the release is riskier than it looks.

| § | Module | Release | Spec |
|---|---|---|---|
| 1 | Geometry → meshlets & LOD | v1.0.5 ✅ | [geometry-meshlets-design.md](./geometry-meshlets-design.md) |
| 2 | Visibility → GPU culling | v1.0.5 ✅ | [visibility-design.md](./visibility-design.md) |
| 3 | Rendering → indirect / multi-draw / deferred | v1.1 ◐ | — |
| 4 | Shadows → CSM & atlas | v1.1 / v1.2 ✅ | [shadows-design.md](./shadows-design.md) · [shadows-atlas-design.md](./shadows-atlas-design.md) |
| 5 | Lighting → IBL & probes | v1.2 ✅ | [lighting-ibl-design.md](./lighting-ibl-design.md) · [lighting-probes-design.md](./lighting-probes-design.md) |
| 6 | **Environment → sky, atmosphere, ambient context** | v1.2 ◐ (IBL + capture ✅) | [lighting-ibl-design.md](./lighting-ibl-design.md) (the bake) |
| 7 | Materials → the shared fragment contract | ✅ ongoing | — |
| 8 | Effects → particles & decals | v1.2 ◻ | — |
| 9 | Animation → GPU skinning & morphs | v1.3 ◻ | — |
| 10 | Post → TAA & motion blur | v2.0 ◻ | [post-processing-design.md](./post-processing-design.md) |
| **11** | **Physics → the plugin** | 🔌 ◻ | — (this brief) |
| 12 | Scene & streaming | v2.0 ◻ | — |
| 13 | RayTracing | ⛔ | — |

---

## 1. Geometry → Meshlets & LOD — **v1.0.5 ✅ shipped** — [full spec](./geometry-meshlets-design.md)

- **Goal.** Turn a mesh into GPU-cullable clusters and discrete LOD chains; define the flat buffer
  layout the `VisibilitySystem` consumes.
- **Integration surface.** Extends `MegabufferBuilder` (`src/extras/geometry/MegaBufferBuilder.ts`),
  whose `MeshOffset { indexCount, firstIndex, baseVertex, meshID }` is already the right shape.
  Output buffers are plain `STORAGE` buffers.
- **Depends on.** Nothing new. Build-time only.
- **Non-goals.** The DAG cluster-LOD graph and runtime streaming (v2.0); vertex quantization (v1.1).

### 1.1 Still open in this branch

- **Vertex quantization (v1.1).** `snorm16` positions within a per-mesh AABB and octahedral normals
  roughly halve vertex bandwidth. Build-time only, and it composes with meshlets because the
  meshlet buffers store *indices*, not positions. The cost is a decode in every vertex shader that
  reads the mega pool — including the meshlet tier's vertex-pulling path.
- **Tangent generation (v1.3).** `getNormalFromMap` derives its TBN from screen-space derivatives
  (`materialFragments.ts`), which is robust but wrong on mirrored UVs. Authored tangents in the
  mega layout would fix it; the mega vertex layout is declarative, so this is additive.

---

## 2. Visibility → GPU culling — **v1.0.5 ✅ shipped** — [full spec](./visibility-design.md) · [API](./api/visibility.md)

- **Goal.** Make visible-set determination O(GPU) and scene-size-independent: frustum + normal cone
  + Hi-Z occlusion, producing indirect draw args.
- **Integration surface.** Emits `ComputeStageNode`s (the `LightSystem.getComputeStage` pattern)
  that write a batch's `indirectBuffer`; the scene batch is flagged `isIndirect` and drawn by
  `RenderPipeline.ts:137-140`. Consumes a **samplable** depth texture — `ctx.depthTexture` is
  `RENDER_ATTACHMENT`-only, so a depth pre-pass or the `PostChain` g-buffer depth supplies it.
- **Depends on.** §1's meshlet buffers for the fine tier. The coarse tier needs only per-instance
  bounds.
- **Non-goals.** Deferred visibility-buffer *shading* (§3); persistent GPU-side scene management.

### 2.1 What shadows proved about this module

The per-view options (`camera`, `occlusion`, `hizRect`, `screenSize`) were added for cascades and
immediately turned out to be the general capability: **any** off-main-view render — shadow cascade,
shadow atlas tile, probe capture face, and eventually a physics broadphase pass — is this same cull
chain pointed at a different `Camera` buffer. That generality is worth protecting; a change here
that assumes the main view now breaks four consumers.

### 2.2 Still open

- **Two-phase temporal culling** (reserved; throws today). Removes single-frame disocclusion pop-in
  at the cost of a second cull+draw. Ship only after single-phase is proven in a real scene.
- **Small-triangle rejection (v1.1).** A meshlet whose projected area is sub-pixel contributes
  nothing; cheap to test alongside the existing LOD math.
- **Visibility buffer (v1.1).** See §3.

---

## 3. Rendering → Indirect / Multi-draw / Deferred — **v1.1**

- **Indirect / GPU-driven (v1.0.5 ✅).** Not a separate module — it is the draw side of Visibility.
- **Multi-draw fast path (v1.1, detect-only).** `multiDrawIndexedIndirect` would let one call draw
  many meshlets with distinct index ranges, avoiding the index-compaction pass
  (`visibility-design.md` §7). It is **not** core WebGPU (Chromium-experimental). Per constraint #4:
  detect `capabilities.has('multi-draw-indirect')`, keep the compaction baseline, use multi-draw
  only when present.
- **Deferred / visibility-buffer shading (v1.1).** Write a triangle/meshlet-ID buffer instead of
  shaded color, then shade in a fullscreen compute pass. Reuses the `PostChain`/`TransientPool`
  target machinery and the material shader chunks. Depends on v1.0.5 meshlet IDs.
- **Transparency (v1.2).** There is **no ordering model today** — batches draw in array order and
  that is the whole story. Blended particles (§8) force the issue. Two candidates: back-to-front
  sorting of an indirect draw's instances (a GPU sort, which the broadphase work in §11 also wants),
  or weighted-blended OIT (order-independent, one extra target, no sort). The second composes better
  with GPU-driven drawing, because sorting fights the whole premise that the CPU never iterates.
- **Depends on.** v1.0.5 meshlet IDs; a g-buffer layout shared with Post.

---

## 4. Shadows → CSM & atlas — **v1.1 / v1.2 ✅ shipped** — [Phase 1](./shadows-design.md) · [Phase 2](./shadows-atlas-design.md)

- **Goal.** Directional shadow maps → CSM (Phase 1) → shadow atlas with punctual lights, tile
  caching, and GPU caster culling (Phase 2).
- **Integration surface.** A shadow map is an offscreen depth `RenderPassNode` rendered from the
  light's viewpoint with the same indirect batches as the main pass — so **caster culling is the
  `VisibilitySystem` run per cascade with a different frustum**, and a cascade is literally a
  108-float `Camera`. Receiver-side, shadowing **decorates** the `LightSystem` shader layout: it
  wraps `getIncidentLight` and folds the shadow term into `IncidentLight.attenuation`, so
  Lambert/Toon/PBR all gain shadows unedited.
- **Depends on.** v1.0.5 per-view culling; the five core additions in `roadmap.md` §1.1.
- **Non-goals.** Virtual shadow maps (v2.0), VSM/ESM/PCSS filtering (v1.3), ray-traced shadows (⛔).

### 4.1 Still open

- **PCSS / contact hardening (v1.3).** A blocker-search pass over the existing atlas, then a
  variable PCF radius. Needs no new resources — the atlas and comparison sampler already exist.
- **Filtered moments (VSM/ESM/MSM, v1.3).** Needs a *filterable color* atlas alongside the depth
  one, which costs a texture binding — check `lighting-ibl-design.md` §4.4 before committing.
- **Virtual shadow maps (v2.0).** Page-resident tiles driven by which texels visible receivers
  actually request. The v1.2 atlas, its tile table, and its per-view culling are the substrate that
  makes this a feature rather than a rewrite.

---

## 5. Lighting → IBL & Probes — **v1.2** — [Phase 1](./lighting-ibl-design.md) · [Phase 2](./lighting-probes-design.md)

- **Goal.** Give NullGraph an indirect lighting term. Today the *entire* indirect contribution in
  the PBR fragment is `let ambient = vec3<f32>(0.3) * albedo * ao;` — which is why metals read flat
  and why no work on the direct loop finishes a scene. Phase 1 replaces it with image-based
  lighting: prefiltered env cube + split-sum BRDF LUT + SH-L2 irradiance. Phase 2 makes it vary
  through space: SH-L1 irradiance volumes, GPU probe capture with multi-bounce, and local
  parallax-corrected reflection probes.
- **Integration surface.** A **sibling of `ShaderLightingLayout`**: `ShaderIndirectLayout` provides
  `getIndirectDiffuse(worldPos, N)` and `getIndirectSpecular(worldPos, N, V, roughness, F0)`,
  composed into materials exactly as `LightSystem.getShaderLayout()` already is
  (`materialFragments.ts:568-576`). Phase 2 re-emits the **same two functions** against a probe
  volume, so it costs zero material edits — which is why the contract takes `worldPos` in Phase 1
  even though Phase 1 ignores it.
- **The constraint that shapes both phases.** `maxBindGroups: 4`, all four spent (`roadmap.md`
  §6.1). Indirect lighting lives **inside group 2 alongside the lights**, never in a new group. The
  binding map and per-stage resource counts are in `lighting-ibl-design.md` §4.4 — it fits exactly,
  with no slack, and that budget is what forces SH-L1 volumes over L2 and a cube *array* for
  reflection probes.
- **Depends on.** Phase 1: nothing — the first module since v1.0.5 needing **no core-engine change**.
  Phase 2: the additions shadows already introduced, because probe capture *is* the shadow atlas
  rendered in color.
- **Non-goals.** DDGI per-probe visibility (v2.0), probe streaming, SSR (Post's job), RT GI (⛔).

### 5.1 Still open

- **Area lights via LTC (v1.3).** `RectAreaLightProxy` ships but its shading is a point-light
  approximation. Linearly-transformed cosines need two LUT textures — again, check the budget first.
- **Light cookies / IES (v1.3).** A texture-array lookup in the spot/point path. Cheap, and it
  costs one binding.
- **DDGI visibility (v2.0).** The correct fix for light leaking; needs two bindings that do not
  exist yet (`lighting-probes-design.md` §6.3).

---

## 6. Environment → sky, atmosphere, and the world's ambient context — **`null-graph/environment`** ✅ shipped (IBL + sky)

- **Goal.** Own everything that makes a world feel like a *place* rather than a set of objects: the
  sky, the light it throws, the air between the camera and the geometry, and the large-scale
  systems (weather, wind, water, terrain) that the rest of the engine reads as context.
- **Why it is its own module and not part of Lighting.** The IBL bake lives here because the sky is
  what gets baked — but `LightSystem` owns *direct* light and knows nothing about where the
  environment came from. Splitting on that line keeps the sky swappable: a procedural sky, an
  imported HDR, and a scene capture all produce the same prefiltered cube and the same SH
  coefficients, and neither the lights nor the materials can tell which.
- **Integration surface.**
  - **Into materials** through the `ShaderIndirectLayout` contract (§5) — the environment is the
    Phase 1 implementation of `getIndirectDiffuse` / `getIndirectSpecular`.
  - **Into the frame** as a bake chain of `ComputeStageNode`s (retiring, or amortized one mip per
    frame for an animated sky) plus one `RenderPassNode` for the skybox, drawn after opaque with
    `depthCompare: 'less-equal'` and `depthWriteEnabled: false`.
  - **Into `LightSystem`** in the other direction for time-of-day: the sun's orbit writes a
    direction and colour into the existing 16-float light record. No new plumbing — the sun is just
    a directional light whose fields someone else is filling.
  - **Into group 2** alongside the lights, because there is no fifth bind group (`roadmap.md` §6.1).
- **Depends on.** Nothing for the Phase 1 bake — compute stages, `rgba16float`/`rg16float` storage
  textures, and cube views are all core WebGPU. Scene capture (v2.0 atmosphere, water reflections)
  reuses the per-view rendering primitives shadows introduced.
- **Non-goals.** A weather *simulation*. Weather here is a set of blended parameters driving sky,
  fog, wind, and particle emitters — the simulation, if an application wants one, is application
  territory.

### 6.1 The sub-branches, and who owns what

The boundary with Effects is worth stating, because several features could plausibly live in
either: **Environment owns the ambient context; Effects owns discrete visual events.**

| Concern | Owner | Why |
|---|---|---|
| Sky, atmosphere, fog, clouds | Environment | ambient, always present, drives the IBL bake |
| Wind *field* | Environment | a global uniform many systems sample |
| Foliage response *to* wind | Environment | vertex-side, tied to the field |
| Rain/snow *particles* | Effects | discrete, pooled, indirect-drawn |
| Rain/snow *state* (how hard, wetness) | Environment | a weather parameter others read |
| Water surface | Environment | a large-scale world feature |
| Splashes, spray | Effects | discrete events |

### 6.2 The sequencing that matters

- **Sky before IBL.** The prefilter reads the sky cube, so a procedural sky is worth having first —
  it makes the whole bake chain testable with no asset, which is what keeps IBL from being blocked
  on finding an HDR file.
- **Fog after deferred, or before?** Exponential height fog is a fragment-side term and can ship in
  the forward path immediately (v1.2). Froxel volumetric fog (v2.0) wants the clustered grid shape
  and a depth buffer, so it follows the deferred work.
- **Time-of-day forces the amortized bake.** A static sky bakes once; a moving sun invalidates the
  prefiltered cube every frame. `updateMode: 'amortized'` (one mip per frame, full refresh in six)
  exists precisely so this is a config change rather than a redesign.
- **Water and terrain are v2.0 and should be designed together.** Both want a heightfield the
  physics plugin can also read (§11), and designing either alone will get that interface wrong.

---

## 7. Materials → the shared fragment contract — **✅ ongoing**

- **Goal.** One definition of each material family, usable from every draw path.
- **Integration surface.** `materialFragments.ts` is the single source of truth: each family is a
  `{ bindings, fragment }` pair **parameterized by bind-group index**, so the same material serves a
  normal ECS batch (group 1) and a visibility meshlet batch (group 2) without duplication. Lighting
  and — after v1.2 — indirect lighting arrive as injected WGSL layouts, so a material never knows
  which technique is in use.
- **Why it is listed as a module at all.** Every rendering feature since v1.0.5 has landed by
  *composing into* this contract rather than editing it: visibility reuses the fragments, shadows
  decorate the lighting layout, IBL adds a sibling layout. It is the most-extended interface in the
  engine and the one most worth keeping narrow.
- **Still open.** `ShaderConfig.tonemap` (the PBR fragment tonemaps and gamma-corrects inline, which
  double-applies whenever `PostChain`'s Tonemap runs — a pre-existing condition that IBL makes
  visible); a declared g-buffer output mode for §3's deferred path; `motionVectors: true` for §9.

---

## 8. Effects → Particles & Decals — **v1.2**

- **Goal.** Compute-simulated GPU particles; clustered/deferred decals.
- **Integration surface.** Particles are the canonical DOD case: a compute stage integrates a
  `Float32Array`-backed particle buffer, then an `isIndirect` batch draws the live count — the exact
  instance-culling shape from v1.0.5, with the emitter writing the indirect `instanceCount` instead
  of a cull. Decals reuse the froxel grid the clustered `LightSystem` already builds; a decal is a
  box that binds into the same spatial structure a light does.
- **Depends on.** v1.0.5 (indirect instancing); v1.1 (deferred, for deferred decals); §3's
  transparency model for *blended* particles specifically.
- **Non-goals.** A node-graph VFX authoring layer — that is application territory.
- **Shared surface with physics.** The particle buffer and the XPBD solver in §10 want the same
  layout. Design the particle record once, with room for a velocity and an inverse mass, and the
  physics plugin can simulate the same buffer the renderer draws.

---

## 9. Animation → GPU skinning & morphs — **v1.3**

- **Goal.** Move skinning and morph blending from the vertex shader (and the CPU) to a compute
  pre-pass.
- **Integration surface.** `GLBParser`/`SkeletonManager` already parse joints/weights and `Animator`
  samples poses. v1.3 adds a compute stage that reads bind-pose mega-vertices + a joint-matrix
  buffer and writes a **skinned mega-vertex buffer**, which then feeds the *same* meshlet/visibility
  path — no second draw path, because the output is layout-identical to the input.
- **The detail that makes it correct.** Skinned bounds must be refit per frame (loosened spheres),
  or culling rejects geometry that has animated outside its bind-pose bounds. Conservative direction:
  loosen, never tighten.
- **Bonus it unlocks.** Compute skinning dissolves the `@group(2) @binding(0)` collision between
  `SkinningBindings` and the clustered `LightSystem` (`roadmap.md` §6.1), because the skeleton
  uniform stops being a fragment-path binding at all.
- **Depends on.** v1.0.5 mega-buffer conventions.

---

## 10. Post → TAA & Motion blur — **v2.0**

- **Goal.** Temporal antialiasing and velocity-based motion blur.
- **Integration surface.** `TransientPool` **already supports** `history: 2` double-buffered targets
  (`transient-pool.md` §8) precisely so TAA needs no allocator change. Needs a velocity g-buffer
  target (`buildPBRShader({ motionVectors: true })`, anticipated in `post-processing-design.md` §13)
  and camera jitter.
- **Depends on.** A velocity buffer — a deferred/g-buffer output from §3.
- **Still open beyond TAA.** Depth of field currently ships as Bokeh without a real
  circle-of-confusion (no linearized scene depth is wired into it); upscaling wants the temporal
  history TAA introduces, so it follows rather than precedes.

---

## 11. Physics → the plugin — **🔌 `null-graph/physics`**

**Status:** brief only. Deliberately *not* scheduled into the core release train.

### 11.1 Why physics is a plugin, not a subsystem

NullGraph renders a caller-supplied array of nodes and reads instance transforms out of a flat
buffer it does not own. That is not an accident of the design — it is the design. A physics engine
is therefore a **producer of transforms**, exactly like a Web Worker computing an ECS layout, and it
attaches at the same seam.

The consequence is worth stating plainly: **most of what people want from "engine physics" needs no
engine code at all.** A worker running a mature solver (Rapier, Jolt, or a hand-rolled one) writing
model matrices into the instance array is a complete, correct integration — and it is faster to
ship, easier to replace, and does not put a physics dependency in the render path. Anything that
makes physics a *core* concern is a step away from the engine's premise.

So the plugin is phased by how much of the work moves to the GPU, and **P1 is a legitimate stopping
point** for most applications.

### 11.2 P1 — the transform bridge (zero engine change)

- **Goal.** A solver, wherever it runs, drives the instance buffer with no copies.
- **Integration surface.** `InstanceBuffer` (`visibility/cullMath.ts:47`) is a `Float32Array` view
  over a plain `ArrayBuffer`, 24 floats per instance, with the **model matrix at floats 0–15** and
  the **bounding sphere at 16–19**. A solver writes the matrix; the renderer adopts the array by
  reference through `setInstances` and never copies it. `VisibilitySystem.upload()` is one
  `writeBuffer` per frame.
- **The one additive change worth making.** `InstanceBuffer` allocates its own `ArrayBuffer`
  (`cullMath.ts:56`). Letting the caller supply the backing buffer is an optional constructor
  parameter and nothing else — and it is deliberately **transport-agnostic**.

#### 11.2.1 SharedArrayBuffer is an option, never a requirement

Worth stating plainly, because it is easy to read the zero-copy contract as implying it: **nothing
in NullGraph uses, needs, or checks for `SharedArrayBuffer`.** There are zero references to it in
`src/`, and the engine works identically without cross-origin isolation.

Two separate ideas get conflated here:

| Idea | What it means | What it costs |
|---|---|---|
| **Zero-copy** (the constraint) | the module uploads *from your array*; it never copies into its own | nothing — a plain `ArrayBuffer` on one thread satisfies it completely |
| **Worker transport** (your choice) | how the array gets filled off-thread | see below |

Only the second involves SAB, and it has a header-free option:

| Transport | Headers | Trade-off |
|---|---|---|
| Same thread | none | simplest; the solver competes with your frame |
| **Transferable `ArrayBuffer`** via `postMessage` | **none** | ownership *moves*, so double-buffer (send A, fill B, swap) or transfer back each frame |
| `SharedArrayBuffer` | **COOP + COEP required** | no transfer at all; both threads see one allocation |

The transferable path is the honest default: it needs no deployment change, and the double-buffer
it forces costs one extra instance array — which for 100 000 instances is 9.6 MB, not a reason to
require cross-origin isolation of every app that wants physics.

SAB is worth reaching for when the solver runs continuously at a fixed step and you want it writing
while the renderer reads, with no swap. `dev-server.js:13-14` serves the headers (with a `--no-sab`
opt-out) so that path is *testable* — that is a test-harness capability, not an engine dependency.
`InstanceBuffer(capacity, backing?)` accepts either kind of buffer and never asks which it got.
- **The detail that decides whether it feels good.** Physics runs at a fixed step; rendering does
  not. Without interpolation the result judders at any refresh rate that is not an exact multiple of
  the step. The bridge keeps two transform sets and interpolates into the instance buffer by the
  accumulator remainder — which is also the only per-frame work the bridge does.
- **Bounds.** The instance record's bounding sphere is already what the broadphase would want;
  keeping it in sync with the collider is the bridge's job, and getting it wrong shows up as
  *culling* artifacts, not physics ones. Conservative direction: loosen.
- **Non-goals for P1.** Shipping a solver. The plugin is an **adapter**; third-party WASM stays
  third-party, and the surface is small enough to write against Rapier and Jolt both.

### 11.3 P2 — GPU broadphase

- **Goal.** Make pair-finding scene-size-independent on the GPU, the way visibility made culling.
- **Integration surface — and the reason this is cheap.** A uniform-grid broadphase is *the same
  kernel shape* as the clustered light binner: bin N objects into a spatial grid with atomics, write
  a per-cell offset+count, then a compacted index list (`LightSystem.getComputeStage` +
  `preClearBuffers` to zero the counters). Emitting the candidate-pair list is the same compaction
  the visibility cull already does into `drawArgs[1]`. Both patterns exist and are tested.
- **Why the meshlet data helps.** Meshlet bounding spheres are already a one-level BVH over each
  mesh, built at import time. A midphase that tests a moving collider against a mesh's meshlet
  bounds gets a usable acceleration structure for free — the buffers are uploaded and the cull
  shaders already transform bounds by an instance matrix.
- **Depends on.** Nothing in the render path. Reads the instance buffer; writes its own.
- **Non-goals.** Replacing a CPU solver's broadphase when P1 is sufficient — this earns its place
  only at counts where the CPU broadphase dominates the frame.

### 11.4 P3 — GPU solver (XPBD)

- **Goal.** Simulate on the GPU the classes that suit it: particles, sand, debris, cloth, soft
  bodies, and eventually rigid contacts.
- **Why position-based.** XPBD is iterative constraint projection over positions — a sequence of
  compute dispatches over a flat buffer, with no per-frame allocation and no readback. That is
  precisely the shape the engine already executes.
- **The constraint that shapes the solver.** **WGSL has no `atomic<f32>`** (`roadmap.md` §6). A
  Gauss-Seidel solver accumulating contact impulses per body needs float atomics; without them the
  options are graph-colored constraint batches (dispatch per color, no conflicts), a **Jacobi**
  iteration that averages instead of accumulating, or fixed-point atomics on `atomic<u32>` with a
  scale factor. Jacobi converges slower but parallelizes cleanly and needs nothing exotic — it is
  the baseline, exactly as the atomic-free path is the baseline everywhere else here.
- **Shared with Effects.** Particles (§7) and the XPBD particle solver want the same buffer. Design
  the record once — position, velocity, inverse mass, flags — and the renderer draws what the solver
  integrates, with no transfer between them.
- **Non-goals.** A general rigid-body solver competing with Rapier/Jolt on correctness. GPU physics
  wins on *count*, not on fidelity; joints, stacking, and continuous collision belong on the CPU.

### 11.5 Queries, character control, and the readback rule

- **Raycasts and overlap queries are async.** Reading a GPU result back in the frame loop is a
  pipeline stall, which the engine forbids everywhere else and should forbid here. Queries are
  submitted into a request buffer, resolved by a compute stage, and read **one frame later** —
  which is acceptable for AI, targeting, and most gameplay, and unacceptable for a character
  controller.
- **So the character controller stays on the CPU.** A capsule sweep against a local subset of
  colliders, driven by P1's transform bridge. This is the honest split: the thing that must be
  immediate stays where immediacy is free.
- **Debug draw** (colliders, contact points, broadphase cells, sleeping islands) is a `debug-ui`
  consumer and is worth building first — a physics bug you cannot see is a physics bug you cannot
  fix.

### 11.6 Engine touchpoints, summarized

| Phase | Needs from the engine |
|---|---|
| P1 transform bridge | **Nothing required.** One optional additive change: `InstanceBuffer` accepting an external (Shared)ArrayBuffer. |
| P2 GPU broadphase | Nothing. Compute stages, atomics, and `preClearBuffers` all exist. Benefits from indirect dispatch (`roadmap.md` §6). |
| P3 GPU solver | Nothing. Benefits from indirect dispatch; blocked on nothing. |
| Queries | Nothing. Async by construction. |

That table is the argument for the phasing. A plugin that needs no core change can ship whenever
someone wants it, and can be abandoned without leaving anything behind in `src/core/`.

---

## 12. Scene & Streaming — **v2.0**

- **Goal.** Worlds larger than resident memory.
- **Integration surface.** The engine has no scene, so "streaming" means: which instances are in the
  instance buffer, and which mesh data is in the mega pool. Both are flat arrays with explicit caps,
  so streaming is a residency policy over ranges — not a graph traversal.
- **Depends on.** v2.0 virtual geometry shares the same residency problem for *clusters*; the two
  should be designed together or one will be rewritten.
- **Also here.** An offline asset pipeline (v1.3) that meshletizes, quantizes, and packs at build
  time rather than on load. Meshletization is already build-time; this makes it *pre*-build-time,
  which matters as soon as a scene has more than a handful of meshes.

---

## 13. RayTracing — **⛔ blocked on WebGPU**

- **Reality.** WebGPU has no acceleration-structure or ray-query API, and none is in the candidate
  spec as of this writing. This branch cannot be built, only prepared for.
- **The rule.** Per constraint #4 it is **detect-and-adopt**: when the platform ships it, surface
  the feature on `capabilities` and design then. Do **not** stub BLAS/TLAS types now — they would
  encode guesses about an API that does not exist, and they would have to be deleted.
- **In the meantime.** RT reflections/shadows/GI are approximated by screen-space effects (Post) and
  probes (§5). Those are legitimate techniques in their own right, not placeholders waiting to be
  swapped out — designing them as stand-ins would compromise both.
