# NullGraph — Engine Roadmap

**Status:** planning document. Non-binding except where a linked spec exists.
**Current release:** `v1.0.5`.
**This document:** the versioned plan — where the engine is, the target architecture, what ships
when, and the platform realities that shape all of it.
**Companion:** [`roadmap-modules.md`](./roadmap-modules.md) — the per-module design briefs. Each
module's goal, its **real integration surface**, its dependencies, and its explicit non-goals. A
brief becomes a standalone spec when its release is scheduled.

This document turns a loose module wishlist into a versioned plan that is **grounded in how
NullGraph actually renders today** — the linear `render(sequence)` model, the existing indirect
hook, the clustered `LightSystem` compute pattern, and the "detect capabilities, never polyfill"
stance. Every future module is mapped onto those primitives rather than onto a generic engine
architecture, because that is what makes the plan buildable instead of aspirational.

**Specs that exist today** — all implementation-ready. `✅` is shipped and exported; `✍` is designed
and not yet built. The five `environment-*` docs are one series, read in order, and each states what
the phase before it retires:

| Spec | Covers | Status |
|---|---|---|
| [`geometry-meshlets-design.md`](./geometry-meshlets-design.md) | clusters, bounds, cones, LOD chains | ✅ `extras/geometry/MeshletBuilder.ts` |
| [`visibility-design.md`](./visibility-design.md) | two-tier GPU culling, Hi-Z, indirect draw | ✅ `null-graph/visibility` |
| [`shadows-design.md`](./shadows-design.md) | CSM, caster bias, the lighting decorator | ✅ `null-graph/shadows` |
| [`shadows-atlas-design.md`](./shadows-atlas-design.md) | atlas, punctual shadows, tile caching | ✅ `null-graph/shadows` |
| [`lighting-ibl-design.md`](./lighting-ibl-design.md) | prefiltered env, BRDF LUT, SH irradiance | ✅ `null-graph/environment` |
| [`lighting-probes-design.md`](./lighting-probes-design.md) | irradiance volumes, probe capture, reflection probes | ✅ `null-graph/environment` |
| [`environment-sky-design.md`](./environment-sky-design.md) | the `SkySource` seam, per-pixel skybox, analytic daylight | ✅ 1.1 |
| [`environment-timeofday-design.md`](./environment-timeofday-design.md) | the clock; sun/moon → lights, shadows, sky, exposure | ✅ `null-graph/environment` |
| [`environment-night-design.md`](./environment-night-design.md) | stars, moon, twilight, `SkyComposite` | ✅ `null-graph/environment` |
| [`environment-atmosphere-design.md`](./environment-atmosphere-design.md) | Rayleigh/Mie/ozone LUT chain | ✅ `null-graph/environment` |
| [`environment-aerial-design.md`](./environment-aerial-design.md) | froxel aerial perspective, media unification | ✅ `null-graph/environment` |
| [`post-processing-design.md`](./post-processing-design.md) · [`transient-pool.md`](./transient-pool.md) | the `PostChain` and its allocator | ✅ `null-graph/post` |

---

## 0. Guiding constraints

These are not negotiable per release; they are why NullGraph exists and every module below
inherits them. A design that violates one of these is wrong even if it renders.

**How these become code:** [`design-principles.md`](./design-principles.md) — the practices that
turn these five constraints into implementations (verified integration facts, the direction of
"conservative", decorating contracts rather than forking them, `out`-parameter pooling, the
bind-group budget, pass-state ownership, and the shape every design document here follows). Each
one is written next to the case that produced it.

1. **Data-oriented, zero scene graph.** The engine renders a caller-supplied linear array of
   `ComputeStageNode | RenderPassNode` (`src/core/NullGraph.ts:192`, `RenderPipeline.ts:11`).
   New modules produce nodes to splice into that array. They do **not** introduce a retained
   scene, a traversal, or hidden ordering.
2. **Zero-copy, worker-drivable.** Bulk per-frame data is a `Float32Array`/`Uint32Array` the
   caller (often a Web Worker) fills and the module adopts by reference — the `LightSystem`
   `setRawData` pattern (`src/extras/lights/LightSystem.ts:92`). No per-frame allocation, no GC
   spikes. **"Adopts by reference" is the whole requirement** — a plain `ArrayBuffer` on one thread
   satisfies it. How the array reaches the renderer (same thread, a transferable `ArrayBuffer`, or a
   `SharedArrayBuffer`) is the caller's choice; the engine has zero references to `SharedArrayBuffer`
   and never requires cross-origin isolation.
3. **Everything expensive happens at build/compile.** Per-frame cost is uploads, a few buffer
   clears, dispatches, and draws. Structural work (meshletization, pipeline creation, bind-group
   assembly) happens once.
4. **Detect, don't polyfill.** Optional WebGPU features are negotiated once and surfaced on
   `engine.capabilities` (`src/core/types.ts:119`). A module may take a faster path when a feature
   is present, but the **baseline path must run on core WebGPU** with no feature flags. No
   dual-path bloat for a feature that isn't universal — detect it, keep the universal path, and
   adopt fully only when the feature is ubiquitous. See [`recommendation.md`](./recommendation.md).
5. **Fail loud at build time.** No silent fallbacks. A module that would render the wrong thing
   refuses to compile with the offending object named — the `PostChain`/`TransientPool` error
   discipline (`post-processing-design.md` §12, `transient-pool.md` §9).

---

## 1. Where NullGraph is today

Verified against the tree at the time of writing. This is the substrate every future module
builds on.

| Area | Module(s) | Status | Notes |
|---|---|---|---|
| **Core** | `WebGPUContext`, `NullGraph`, `RenderPipeline`, `BatchManager`, `BufferManager`, `TextureManager`, `Camera` | ✅ shipped | Linear compute+render execution; capability negotiation; indirect draw wired; per-batch viewport + camera override; gateable compute stages. |
| **Geometry** | `Primitives` (13 shapes), `GeometryBuilder`, `Geometry`, `MegabufferBuilder`, `MeshletBuilder`, `DynamicGeometryManager`, `VertexLayout` | ✅ shipped | `MegabufferBuilder` produces per-mesh `{indexCount, firstIndex, baseVertex, meshID}`; `MeshletBuilder` layers clusters + bounds + cones + LOD on it (§4). |
| **Visibility** | `VisibilitySystem`, `HiZPyramid`, `cullMath`, `visibilityShaders` | ✅ shipped | Two-tier GPU culling (instance + meshlet), Hi-Z occlusion, cone culling, LOD. `null-graph/visibility`. |
| **Shadows** | `ShadowSystem`, `ShadowAtlas`, `shadowMath`, `atlasMath`, shaders | ✅ shipped | CSM (sphere fit, texel snap, caster extrusion) + atlas with punctual lights, importance tiles, caching, light-view Hi-Z. `null-graph/shadows`. |
| **Cameras** | Orbital, Fly, Follow, Path (+ event proxies) | ✅ shipped | State objects, worker-friendly. |
| **Lighting** | `LightSystem` (standard + clustered-forward + tiled), 7 light proxy types | ✅ shipped | The reference compute-driven subsystem: builds buffers, emits a `ComputeStageNode`, subgroup-accelerated binning when available. **Direct light only** — the indirect term is still a constant (§2, Lighting). |
| **Materials** | PBR, Lambert, Toon, Matcap, Emissive, Basic (+ shader builders/chunks) | ✅ shipped | `materialFragments.ts` is the single source of truth: a portable `{bindings, fragment}` pair parameterized by group index, shared by the normal and GPU-culled paths. |
| **Post** | `PostChain`, `TransientPool`, `ParamArena`, `ShaderComposer`, 16 effects | ✅ shipped | Aliasing render-target pool; zero-copy param arena; the gold-standard spec pair in this repo. |
| **Loaders** | `GLBParser`, `Animator`, `SkeletonManager` | ✅ shipped | Parses skins/animations; vertex-shader skinning wired. **Compute** skinning is v1.3. |
| **Tooling** | `profiler`, `debug-ui` | ✅ shipped | Perf widgets, timestamp queries. |

**The fact that shaped this roadmap** (now discharged): GPU-driven rendering was already plumbed
and unused. `RenderPipeline.ts:137-140` issues `drawIndexedIndirect(batch.indirectBuffer, 0)` for
any batch flagged `isIndirect`, `RenderPipeline.ts:105` exempts such batches from the
"instanceCount === 0 → skip" rule, and `types.ts:57-58` exposes `isIndirect` +
`sharedSourceBuffer`. v1.0.5 is the release that connected a GPU culling compute stage to that
dormant draw path — **with zero core-engine changes**, exactly as designed.

### 1.1 What landed since, and what it cost the core

v1.0.5 needed no engine change. Shadows was the first release that did — five additive fields,
each defaulting to today's behavior, each with a documented (and worse) zero-change fallback:

| Addition | Where | Why it could not be avoided |
|---|---|---|
| `PipelineConfig.depthOnly` | `BatchManager.ts:37-61` | `createBatch` always built ≥1 color target; a depth-only pass has zero, and the mismatch is a validation error. |
| `PipelineConfig.cameraBuffer` | `BatchManager.ts:100` | Group 0 binding 0 was hard-wired to the one global camera, so nothing could point a cull+draw chain at a cascade. |
| `PipelineConfig.viewport` + `RenderBatch.viewport` | `RenderPipeline.ts:114` | The atlas needs tile-scoped rasterization; the zero-change alternative (clip-space remap) **bleeds** into neighbouring tiles. |
| `RenderPassConfig.size` | `RenderPass.ts` | Viewport is **pass** state — a full-pass batch after a tiled one inherits its tile unless the full attachment is restored, and a `GPUTextureView` does not expose its dimensions. |
| `ComputeStageNode.enabled` | `RenderPipeline.ts:28` | A cached tile that rasterizes nothing must not dispatch its caster cull either; a zero-area viewport suppresses only the draw. |

**Deliberately still open:** `VisibilitySystem` `mode: 'two-phase'` throws a named error
(visibility-design §9 — single-phase is the proven baseline); the GPU integration harnesses need a
browser; indirect *dispatch* (§6) remains the one optional additive change.

---

## 2. Target architecture

The full feature tree — every module, every planned feature, at whatever depth it needs — lives in
**[`architecture-tree.md`](./architecture-tree.md)**. It is the canonical "what exists and what is
coming" view, and it carries the inverted per-release index too ("what actually lands in v1.2").

Kept here: the top level, so this document stands on its own.

```
NullGraph
├── Core                  ✅  linear execution, indirect draw, per-view rendering primitives
├── Geometry              ✅  primitives, megabuffer, meshlets + LOD          → geometry-meshlets-design.md
├── Visibility            ✅  two-tier GPU culling, Hi-Z, per-view            → visibility-design.md
├── Rendering             ◐   forward + GPU-driven; deferred is v1.1
├── Materials             ✅  one portable fragment contract, injected layouts
├── Shadows               ✅  CSM + atlas, punctual, caching, caster culling  → shadows-*-design.md
├── Lighting              ✅  direct + indirect: IBL, probe volumes, refl probes → lighting-*-design.md
├── Environment           ◐   IBL + sky + probe capture ✅; fog, weather, water, terrain planned
├── Effects               ◻   GPU particles, decals, trails, screen-space
├── Physics            🔌 ◻   PLUGIN — transform bridge → GPU broadphase → XPBD
├── Animation             ◐   parsed + vertex skinning; compute skinning is v1.3
├── Post                  ✅  16 effects, aliased targets, param arena        → post-processing-design.md
├── Scene & Streaming     ◻   asset pipeline (v1.3), residency (v2.0)
├── Loaders & Assets      ✅  GLB, skins, textures; KTX2/Draco are v1.3
├── Tooling & Debug       ✅  profiler, widgets; frame-graph viz is v1.3
├── Platform              ✅  Worker + OffscreenCanvas, capability negotiation
└── RayTracing            ⛔  no WebGPU API exists — detect-and-adopt (§6)
```

Legend: `✅` shipped · `◐` partial · `◻` planned · `✍` designed · `⛔` blocked · `🔌` plugin

### 2.1 Reading the tree

Three things the shape is trying to say:

- **Core grew five entries in one release, and they were all off-main-view plumbing.** Camera
  override, viewport, pass size, depth-only, gateable compute — added for shadows, immediately
  reused by probe capture, and the reason `Environment`'s bake path and `Physics P2` look cheap.
  Generic primitives paid for once keep paying.
- **`Physics` is the only `🔌` branch.** Everything else is a renderer concern with a place in the
  release train. Physics is a *consumer* of the zero-copy instance contract, not a renderer
  feature, so it ships on its own cadence and can be skipped entirely
  ([`roadmap-modules.md`](./roadmap-modules.md) §11).
- **`◐` is where the honest work hides.** Depth pre-pass, distance culling, depth of field, and
  three detected-but-unused device features are all partial in ways the name does not admit.
  `architecture-tree.md` lists every one with what is actually missing.

---

## 3. Release plan

| Version | Theme | Headline modules | Specs | Depends on |
|---|---|---|---|---|
| **v1.0.5** ✅ | **GPU-Driven Visibility** | Meshlet geometry, two-tier GPU culling, Hi-Z occlusion | geometry-meshlets, visibility | the indirect hook (already present) |
| **v1.1** ✅/◻ | **Shadows & Deferred** | CSM + GPU caster culling ✅, visibility buffer, deferred shading, geometry compression | shadows ✅, rendering-deferred ◻ | v1.0.5 (per-view culling, meshlet IDs) |
| **v1.2** ◐ | **Light, Really** | Shadow atlas + punctual ✅, IBL, irradiance volumes + probes, GPU particles, decals | shadows-atlas ✅, lighting-ibl ✅, lighting-probes ✅, effects-particles ◻ | v1.1 (deferred for deferred decals) |
| **v1.3** ◻ | **Animation & Authoring** | Compute skinning, morph targets, asset pipeline, frame-graph viz, PCSS, area lights | animation-gpu | v1.0.5 mega-buffer conventions |
| **v1.4** ◻ | **Motion & Behaviour** | Blend trees, IK — arguably app-side; revisited when v1.3 lands | — | v1.3 |
| **v2.0** ◻ | **Virtual & Temporal** | Nanite-style cluster-LOD DAG + streaming, TAA, motion blur, volumetrics, virtual shadow maps, DDGI | virtual-geometry, temporal | v1.0.5, v1.1, v1.2 |
| **🔌** ◻ | **Physics plugin** | Transform bridge → GPU broadphase → XPBD solver | physics (brief only) | the zero-copy instance contract |
| — | RayTracing | ⛔ deferred until WebGPU exposes acceleration structures | — | the platform |

**Why Visibility was first.** It is the highest-leverage module and the one that connected an
already-built-but-dead engine capability (indirect draw) to a real consumer. Every later theme
depends on it: shadows wanted GPU caster culling, deferred wants meshlet visibility IDs, probe
capture wants per-view culling, virtual geometry is Visibility taken to its limit. Starting
anywhere else would have meant building those on a CPU-bound draw loop and reworking them later.

**Why Lighting is the next real one.** The entire indirect term in the PBR fragment is
`vec3<f32>(0.3) * albedo * ao`. No amount of work on the direct loop finishes a scene while that
constant is standing in for the environment, which is why v1.2 is the release with the largest
visible delta per line of code.

---

## 4. How Megabuffer, Meshlets, and Visibility chain

These three are levels of one pipeline, not alternatives. Top to bottom of the hierarchy
`meshes → meshlets → triangles → vertices`:

| Layer | Component | Role | Status |
|---|---|---|---|
| Vertex/index pool | **`MegabufferBuilder`** | Packs every mesh into one vertex + index buffer; records per-mesh `{indexCount, firstIndex, baseVertex, meshID}`. | ✅ |
| Sub-mesh clusters | **`MeshletBuilder`** | *Consumes* the megabuffer; partitions each mesh into ≤64v/≤124t meshlets with bounds + cones that **reference** the mega vertex pool. | ✅ v1.0.5 |
| Runtime culling/draw | **`VisibilitySystem`** | GPU-culls instances (coarse) and meshlets (fine), emits the indirect draws over that same pool. | ✅ v1.0.5 |

```
MegabufferBuilder ─▶ mega vertex pool + per-mesh offsets
        │                       │
        │  (consumed by)        └────────────┐ instance tier draws these directly
        ▼                                     ▼
MeshletBuilder ────▶ meshlet buffers ──▶ VisibilitySystem ──▶ drawIndexedIndirect × (few)
   (global indices into the mega pool)     (frustum + cone + Hi-Z + LOD)
```

The key that makes it one pipeline: **meshlets store *global* indices into the megabuffer's vertex
pool**, so a single vertex allocation backs both tiers. You can stop at any layer — megabuffer +
instance-tier culling is a complete meshlet-free path; add `MeshletBuilder` only when you want
sub-object granularity. See `geometry-meshlets-design.md` §2.1 for the exact ownership split.

For end-to-end scene assembly, see [`scene-composition.md`](./scene-composition.md).

---

## 5. Per-module design briefs → moved

The briefs now live in **[`roadmap-modules.md`](./roadmap-modules.md)** — goal, integration
surface, dependencies, and non-goals for each branch of §2, including the Physics plugin.

| Brief | Module | Brief | Module |
|---|---|---|---|
| §1 | Geometry → meshlets & LOD | §7 | Effects → particles & decals |
| §2 | Visibility → GPU culling | §8 | Animation → GPU skinning & morphs |
| §3 | Rendering → indirect / multi-draw / deferred | §9 | Post → TAA & motion blur |
| §4 | Shadows → CSM & atlas | §10 | **Physics → the plugin** 🔌 |
| §5 | Lighting → IBL & probes | §11 | Scene & streaming |
| §6 | Materials → the shared fragment contract | §12 | RayTracing ⛔ |

---

## 6. Cross-cutting: the WebGPU reality

The wishlist was written at the abstraction level of a native mesh-shader engine. Several of its
assumptions do not hold on WebGPU, and pretending otherwise is how a plan rots:

| Native assumption | WebGPU reality | Consequence for the plan |
|---|---|---|
| Mesh shaders expand meshlets | none | Meshlets are drawn via **index compaction + `drawIndexedIndirect`** (`visibility-design.md` §7), or the meshlet-as-instance vertex-pulling variant. Both are core. |
| `multiDrawIndexedIndirect` | optional, non-universal | Baseline is **one indirect draw per material** over a compacted index buffer. Multi-draw is a v1.1 *detected* fast path, never the baseline. |
| Ray tracing | none | The whole RT branch is ⛔ (`roadmap-modules.md` §13). |
| Indirect dispatch | core, and **now wired** | `ComputeStageConfig.indirectBuffer` — see below. Consumers move to it one at a time. |
| Bind groups are plentiful | **`maxBindGroups: 4`** | §6.1 — the budget is fully spent, and it is the hardest constraint in the plan. |
| Atomic floats for solvers | **no `atomic<f32>` in WGSL** | Any GPU accumulation (SH projection, physics contacts) uses a tiled reduction, fixed-point, or Jacobi iteration. Shapes the physics solver directly. |
| Readback is cheap | it is a pipeline stall | Nothing reads GPU counts back in the frame loop. Dispatches cover CPU-known caps with idle threads early-outing; physics queries are deliberately one frame late. |

**Indirect dispatch — shipped, consumers pending.** `ComputeStageConfig` now takes an
`indirectBuffer` (three `u32`, `[x, y, z]`) and `RenderPipeline` issues
`dispatchWorkgroupsIndirect` when one is present; `bufferManager.createDispatchIndirectBuffer()`
allocates it with the `INDIRECT | STORAGE | COPY_DST` mask, and `createComputeStage` refuses a
buffer missing a required flag rather than dropping the dispatch silently.

**The primitive is done; the waste is not gone yet.** Two-phase culling, per-view shadow culls, and
probe capture still over-dispatch against CPU-known caps, because moving each one means its
*producer* kernel must write its own dispatch triple — and clamp it to
`maxComputeWorkgroupsPerDimension`, since nothing on the CPU can validate a GPU-written count.
That is per-consumer work and each can move independently, which is why this is listed as wired
rather than finished.

Everything else in flight — storage buffers, atomics, `INDEX | STORAGE` buffers, `r32float` and
`rgba16float` storage textures, cube and 3D views, `drawIndexedIndirect` — is core WebGPU and
already reachable.

### 6.1 The bind-group budget — the ceiling nobody set out to hit

Core WebGPU guarantees **four** bind groups. NullGraph now spends all four:

| Group | Owner | Where it is fixed |
|---|---|---|
| 0 | camera + instance/ECS storage | `BatchManager.ts:94-102` — the engine binds it, unconditionally |
| 1 | material (instance tier) / meshlet geometry (meshlet tier) | `materialFragments.ts` `MaterialBindings`, `VisibilitySystem.ts:296-302` |
| 2 | lights — **hardcoded** `@group(2)` inside the WGSL `LightSystem` emits | `LightSystem.ts:134,250-252` |
| 3 | shadows | `shadows/shadowShaders.ts:44-46` |

There is no fifth group on the baseline device. Consequently **every future fragment-stage
subsystem must join an existing group**, and the only group with room is 2. IBL and probes
therefore live at group 2 bindings 3–13 (`lighting-ibl-design.md` §4.4, which also tallies the
per-stage counts against their core limits — storage buffers are the second-tightest at 7 of 8).

Three consequences worth stating before they are rediscovered:

- **Group 2 now has three owners** (lights, environment, probes), and exactly one of them may build
  the bind group. That is why `EnvironmentSystem.applyToBatch(batch, lights)` and
  `ProbeVolume.applyToBatch(batch, lights, env, refl)` take their peers as arguments.
- **A latent collision already exists:** `CommonChunks.SkinningBindings` and
  `PBRChunks.SkinningBindings` also declare `@group(2) @binding(0)`, so CPU-side skinning and
  clustered lighting cannot coexist in one shader today. v1.3's compute skinning dissolves it — but
  until then it is a real constraint, not a hypothetical one.
- **Physics does not compete for this budget.** Its buffers are consumed by compute stages and by
  the CPU, not by the fragment stage — which is a large part of why it can be a plugin at all.

---

## 7. Sequencing & dependency graph

```
                         ┌───────────────────────────┐
        v1.0.5           │  meshlet geometry (data)  │
                         └─────────────┬─────────────┘
                                       │ consumed by
                         ┌─────────────▼─────────────┐
                         │  VisibilitySystem (Hi-Z,  │
                         │  frustum, cone, indirect) │
                         └───┬───────────┬───────────┘
              per-view       │           │  meshlet IDs
              culling        │           │
             ┌───────────────▼──┐   ┌────▼─────────────────┐
      v1.1   │ Shadows P1 (CSM) │   │ Deferred / vis-buffer │ ── velocity ──┐
             │ ✅ shipped       │   │ shading               │               │
             └────────┬─────────┘   └────┬──────────────────┘               │
                      │ ShadowView + atlas     │                            │
             ┌────────▼─────────┐             │                            │
      v1.2   │ Shadows P2 ✅    │             │                            │
             │ atlas · punctual │             │                            │
             │ caching · Hi-Z   │             │                            │
             └────────┬─────────┘             │                            │
                      │  the atlas pass, reused in COLOR                   │
                      │  (cameraBuffer + viewport + size + enabled)        │
      ┌───────────────┼───────────────────────┼───────────┐                │
      ▼               ▼                       ▼           ▼                ▼
┌──────────────┐ ┌──────────────┐    ┌───────────────┐  ┌──────────┐ ┌──────────────┐
│ Lighting P1  │ │ Probe capture│    │ Effects (GPU  │  │ Physics  │ │ TAA / motion │
│ IBL: env,    │ │ (P2 volumes) │    │ particles,    │  │ P2 GPU   │ │ blur    v2.0 │
│ LUT, SH-L2   │ │ multi-bounce │    │ decals)       │  │ broadphase│ └──────────────┘
└──────┬───────┘ └──────────────┘    └───────┬───────┘  └────▲─────┘
       │  getIndirectDiffuse/Specular         │ shares the   │ shares the
       │  contract — Phase 2 reimplements it  │ particle buf │ binner shape
       └──────────────────────────────────────┴──────────────┘
                                     ┌──────────────┐
                            v1.3     │ GPU skinning │ ── skinned mega-buffer ──▶ virtual geo (v2.0)
                                     └──────────────┘    feeds the SAME vis path

   🔌 Physics P1 (transform bridge) hangs off the zero-copy instance contract alone —
      no dependency on any of the above, which is why it can ship at any time.
   ⛔ RayTracing — off the graph until WebGPU exposes acceleration structures.
```

The critical path is **meshlet data → Visibility**, and everything of value hangs off it. That is
the justification for the whole ordering: build the culling substrate once, correctly, and let
shadows, deferred shading, probe capture, particles, physics broadphase, and eventually virtual
geometry inherit it.

---

## 8. Scorecards — what "done" means

### 8.1 v1.0.5 — GPU-Driven Visibility

| # | Gate | Status |
|---|---|---|
| 1 | Draw calls proportional to **material/mesh count, not instance count** | ✅ **measured on a GPU** — 2 indirect draws for 2 meshes at both 1,000 and 13,000 instances, counted on the engine's own encoder; the GPU cull keeps exactly the instances the CPU mirror keeps (`test/visibility-gpu` V1, V2). |
| 2 | Geometry behind other geometry is **not drawn** (Hi-Z) | ✅ **measured on a GPU** — 400 spheres behind a wall culled, 25 in front kept, all 425 back when the wall moves (V3b, V3c). The run also found frame 0 culling *everything*; fixed (V3a). |
| 3 | Culling driven from **worker-fillable** buffers, **no per-frame allocation** | ✅ `InstanceBuffer` + `setInstances` adopt by reference. |
| 4 | Baseline on **core WebGPU**; optional features only accelerate | ✅ no feature gates anywhere. |
| 5 | **No core-engine changes** for the draw path | ✅ `src/core/` untouched by v1.0.5. |

### 8.2 v1.1 / v1.2 — Shadows

| # | Gate | Status |
|---|---|---|
| 1 | Caster culling reuses the visibility chain, not a copy of it | ✅ a cascade fills the same 108-float `Camera`; `CULL_CORE_WGSL` unmodified. |
| 2 | Stock materials gain shadows with **zero edits** | ✅ the `getIncidentLight` decorator; Lambert/Toon/PBR all unchanged. |
| 3 | Cascades do not shimmer on rotation or crawl on translation | ✅ **since Cycle A.** The snap was a no-op (its frame's eye sat at the cascade centre) and U3 passed tautologically; the first GPU run measured a 0.31-texel slide per degree of yaw. Fixed; U3 now measures in a fixed frame and `test/shadows-gpu` T4 shows whole-texel moves with 0 mismatched texels. |
| 4 | Every core addition is **additive and inert by default** | ✅ five fields, all optional; all pre-existing tests pass untouched. |
| 5 | A cached tile costs **neither a draw nor a cull** | ✅ viewport + `enabled` set together (U12). |
| 6 | The frame path allocates nothing | ✅ measured 0 typed-array constructions per frame. |
| 7 | Numeric proof of the visual gates (acne, peter-panning, seams, leaks) | ◐ seams (T5a–c), caster depth range (T3), shadow presence (T1) and beyond-distance (T7) measured on a GPU; peter-panning and leaks not yet. |

**The GPU layer now runs.** `npm run test:gpu` (Node + Dawn) and `npm run test:gpu:browser`
(headless Chrome) execute every harness on a real device and fail on any uncaptured WebGPU error;
both are green. See [`handoff.md`](./handoff.md) for the run log and what is not covered yet.
