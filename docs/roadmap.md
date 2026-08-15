# NullGraph — Engine Roadmap

**Status:** planning document. Non-binding except where a linked spec exists.
**Current release:** `v1.0.4`.
**Flagship release:** `v1.0.5` — *GPU-Driven Visibility* (meshlets + two-tier GPU culling) — **✅ implemented**,
see §1.1.

This document turns a loose module wishlist into a versioned plan that is **grounded in how
NullGraph actually renders today** — the linear `render(sequence)` model, the existing indirect
hook, the clustered `LightSystem` compute pattern, and the "detect capabilities, never polyfill"
stance. Every future module is mapped onto those primitives rather than onto a generic engine
architecture, because that is what makes the plan buildable instead of aspirational.

The two v1.0.5 modules have full implementation-ready specs — **both are now implemented**:

- **[`geometry-meshlets-design.md`](./geometry-meshlets-design.md)** — the data layer: how
  geometry becomes meshlets with bounds and cones, and the exact buffer contract.
  → `src/extras/geometry/MeshletBuilder.ts`
- **[`visibility-design.md`](./visibility-design.md)** — the runtime layer: GPU culling that
  consumes those buffers and emits the indirect draws the engine already renders.
  → `src/extras/visibility/`, exported as `null-graph/visibility`.

API docs: [`api/visibility.md`](./api/visibility.md) (usage) ·
[`api/visibility-shader-reference.md`](./api/visibility-shader-reference.md) (buffer layouts, WGSL
contracts, material bridge).

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
   spikes.
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

## 1. Where NullGraph is today — `v1.0.4` (+ v1.0.5 modules landed)

Verified against the tree at the time of writing. This is the substrate every future module
builds on.

| Area | Module(s) | Status | Notes |
|---|---|---|---|
| **Core** | `WebGPUContext`, `NullGraph`, `RenderPipeline`, `BatchManager`, `BufferManager`, `TextureManager`, `Camera` | ✅ shipped | Linear compute+render execution; capability negotiation; **indirect draw already wired**. |
| **Geometry** | `Primitives` (13 shapes), `GeometryBuilder`, `Geometry`, `MegabufferBuilder`, **`MeshletBuilder`**, `DynamicGeometryManager`, `VertexLayout` | ✅ shipped | `MegabufferBuilder` produces per-mesh `{indexCount, firstIndex, baseVertex, meshID}`; `MeshletBuilder` layers clusters + bounds + cones + LOD on it (§4.1). |
| **Visibility** | `VisibilitySystem`, `HiZPyramid`, `cullMath`, `visibilityShaders` | ✅ shipped | Two-tier GPU culling (instance + meshlet), Hi-Z occlusion, cone culling, LOD. Consumes the dormant indirect hook. `null-graph/visibility`. |
| **Cameras** | Orbital, Fly, Follow, Path (+ event proxies) | ✅ shipped | State objects, worker-friendly. |
| **Lighting** | `LightSystem` (standard + **clustered-forward** + tiled), 7 light proxy types | ✅ shipped | The reference compute-driven subsystem: builds buffers, emits a `ComputeStageNode`, subgroup-accelerated binning when available. |
| **Materials** | PBR, Lambert, Toon, Matcap, Emissive, Basic (+ shader builders/chunks) | ✅ shipped | Shader-composition pattern (`build*Shader`) is the template for generated pipelines. |
| **Post** | `PostChain`, `TransientPool`, `ParamArena`, `ShaderComposer`, 16 effects | ✅ shipped | Aliasing render-target pool; zero-copy param arena; the gold-standard spec pair in this repo. |
| **Loaders** | `GLBParser`, `Animator`, `SkeletonManager` | ✅ shipped | Already **parses** skins/animations. GPU skinning is not yet wired — see v1.3. |
| **Tooling** | `profiler`, `debug-ui` | ✅ shipped | Perf widgets. |

**The fact that shaped this roadmap** (now discharged): GPU-driven rendering was already plumbed
and unused. `RenderPipeline.ts:137-140` issues `drawIndexedIndirect(batch.indirectBuffer, 0)` for
any batch flagged `isIndirect`, `RenderPipeline.ts:105` exempts such batches from the
"instanceCount === 0 → skip" rule, and `types.ts:57-58` exposes `isIndirect` +
`sharedSourceBuffer`. v1.0.5 is the release that connected a GPU culling compute stage to that
dormant draw path — **and it did so with zero core-engine changes**, exactly as designed.

### 1.1 v1.0.5 status — what actually landed

| Piece | Where | Notes |
|---|---|---|
| Meshlet build (clusters, bounds, cones, LOD) | `extras/geometry/MeshletBuilder.ts` | Consumes a `MegabufferBuilder`; emits the flat `MeshletPack` buffers. |
| Instance tier (coarse, per-object) | `VisibilitySystem.createInstanceDrawBatch` | One `drawIndexedIndirect` per unique mesh, independent of instance count. |
| Meshlet tier (fine, per-cluster) | `VisibilitySystem.createMeshletDrawBatch` | Vertex-pulling over a static index buffer; one draw per material. |
| Hi-Z pyramid + occlusion | `HiZPyramid.ts`, `HIZ_OCCLUSION_WGSL` | `r32float` max-reduction mip chain; single-phase (last frame's depth). |
| Normal-cone (backface cluster) culling | `CULL_CORE_WGSL` | With the `cutoff >= 2.0` never-cull sentinel. |
| Screen-space-error LOD | `buildMeshletInstanceCullWGSL` | `LOD_AUTO`, or a per-instance override. |
| CPU mirror of the cull math | `cullMath.ts` | Pinned by `npm run test:visibility` (no GPU needed). |

**Deliberately still open:** `mode: 'two-phase'` throws a named error (design §9 — single-phase is
the proven baseline); the GPU integration tests (design §10 T1–T7) need a browser harness; indirect
*dispatch* (§6) is still the one optional additive engine change.

---

## 2. Target architecture, annotated

The wishlist tree, re-expressed against NullGraph's reality. Each leaf is tagged with a status
and the release that owns it. `⛔` marks work that is **blocked on WebGPU itself**, not on us.

```
NullGraph
├── Core                                         ✅ shipped
│   ├── Device / Context / Capabilities          ✅
│   ├── Linear execution (compute + render)       ✅
│   ├── Resource / Buffer / Pipeline mgmt          ✅
│   ├── Indirect draw hook                          ✅ consumed by Visibility (v1.0.5)
│   └── Indirect dispatch                           ◻ optional (small additive engine change)
│
├── Geometry                                     ✅ shipped
│   ├── Meshes / Primitives                        ✅
│   ├── Megabuffer (per-mesh offsets)             ✅
│   ├── Meshlets (build + bounds + cones)          ✅ v1.0.5   → geometry-meshlets-design.md
│   ├── Discrete LOD chains                         ✅ v1.0.5   → geometry-meshlets-design.md
│   ├── Geometry compression (quantized verts)      ◻ v1.1
│   └── Virtual geometry (DAG cluster-LOD)         ◻ v2.0     (Nanite-class; stretch)
│
├── Visibility                                   ✅ v1.0.5 flagship → visibility-design.md
│   ├── Frustum culling (instance + meshlet)       ✅ v1.0.5
│   ├── Hi-Z pyramid build                          ✅ v1.0.5
│   ├── Occlusion culling (Hi-Z test)              ✅ v1.0.5
│   ├── Meshlet cone (backface cluster) culling     ✅ v1.0.5
│   ├── Screen-space-error LOD selection            ✅ v1.0.5
│   ├── Two-phase temporal culling                  ◻ reserved (throws; design §9)
│   └── Visibility buffer (deferred IDs)            ◻ v1.1
│
├── Rendering                                    ◐ forward + GPU-driven
│   ├── Forward (materials)                        ✅
│   ├── Forward+ / Clustered                       ✅ (LightSystem)
│   ├── Indirect / GPU-driven                       ✅ v1.0.5 (the draw side of Visibility)
│   ├── Multi-draw fast path                        ◻ v1.1  (detect-only; not universal)
│   └── Deferred / Visibility-buffer shading        ◻ v1.1
│
├── Shadows                                      ✅ shipped → shadows-design.md (P1), shadows-atlas-design.md (P2)
│   ├── Shadow maps (single)                        ✅ v1.1     → shadows-design.md
│   ├── Cascaded shadow maps (CSM)                  ✅ v1.1     → shadows-design.md
│   ├── Shadow atlas + GPU caster culling            ✅ v1.2     → shadows-atlas-design.md
│   ├── Punctual shadows (spot + point cube)         ✅ v1.2     → shadows-atlas-design.md
│   └── Virtual shadow maps                          ◻ v2.0     (needs the v1.2 atlas)
│
├── Lighting                                     ◐ direct → direct + indirect
│   ├── Direct: PBR / clustered / tiled            ✅ (LightSystem)
│   ├── Indirect-lighting contract (the hook)       ◻ v1.2     → lighting-ibl-design.md §5
│   ├── Env cube (equirect / procedural sky)        ◻ v1.2     → lighting-ibl-design.md
│   ├── GGX prefiltered specular mip chain          ◻ v1.2     → lighting-ibl-design.md
│   ├── Split-sum BRDF LUT                          ◻ v1.2     → lighting-ibl-design.md
│   ├── SH-L2 diffuse irradiance (27 floats)        ◻ v1.2     → lighting-ibl-design.md
│   ├── Skybox pass                                 ◻ v1.2     → lighting-ibl-design.md
│   ├── Irradiance volumes (SH-L1 probe grid)       ◻ v1.2     → lighting-probes-design.md
│   ├── GPU probe capture + multi-bounce            ◻ v1.2     → lighting-probes-design.md
│   ├── Local reflection probes (parallax box)      ◻ v1.2     → lighting-probes-design.md
│   └── DDGI per-probe visibility (leak-free)       ◻ v2.0     (needs binding budget, §6.1)
│
├── Effects                                      ◻ NEW
│   ├── GPU particles (compute-simulated)           ◻ v1.2
│   ├── Decals (clustered / deferred)               ◻ v1.2
│   ├── Volumetrics (froxel fog)                     ◻ v2.0
│   └── Screen-space (SSR etc.)                      ◐ partial via Post
│
├── Animation                                    ◐ parsed → GPU
│   ├── GPU skinning (compute)                       ◻ v1.3 (loaders already parse skins)
│   ├── Morph targets                                ◻ v1.3
│   └── Animation sampling (worker)                  ◐ CPU today (Animator)
│
├── Post                                         ✅ shipped → expanded
│   ├── SSAO / Bloom / Tonemap / +13               ✅
│   ├── TAA + temporal history                       ◻ v2.0 (TransientPool history:2 already exists)
│   └── Motion blur (velocity buffer)               ◻ v2.0
│
└── RayTracing                                   ⛔ blocked on WebGPU
    └── BLAS / TLAS / ray queries / RT reflections   ⛔ no WebGPU RT extension exists;
                                                        detect-and-adopt when standardized
```

Legend: `✅` shipped · `◐` partial · `◻` planned · `⛔` blocked on the platform.

---

## 3. Release plan

| Version | Theme | Headline modules | New specs | Depends on |
|---|---|---|---|---|
| **v1.0.5** ✅ | **GPU-Driven Visibility** | Meshlet geometry, two-tier GPU culling, Hi-Z occlusion | geometry-meshlets, visibility | indirect hook (present) |
| v1.1 | Deferred & Shadows | Visibility buffer, deferred shading, shadow maps + CSM, geometry compression | rendering-deferred, **shadows (Phase 1) ✍** | v1.0.5 (caster culling, meshlet IDs) |
| v1.2 | Image-based lighting & Effects | IBL, irradiance volumes + probes, GPU particles, decals, shadow atlas + punctual shadows | **lighting-ibl (Phase 1) ✍**, **lighting-probes (Phase 2) ✍**, effects-particles, **shadows-atlas (Phase 2) ✍** | v1.1 (deferred, atlas) |
| v1.3 | Animation on the GPU | Compute skinning, morph targets | animation-gpu | v1.0.5 (mega-buffer skinned variant) |
| v2.0 | Virtual geometry & temporal | Nanite-style cluster-LOD DAG + streaming, TAA, motion blur, volumetrics | virtual-geometry, temporal | v1.0.5, v1.1 |
| — | RayTracing | ⛔ deferred until WebGPU exposes acceleration structures | — | the platform |

**Why Visibility is first.** It is the highest-leverage module and the one that connects an
already-built-but-dead engine capability (indirect draw) to a real consumer. Every later theme
depends on it: shadows want GPU caster culling, deferred wants meshlet visibility IDs, virtual
geometry is Visibility taken to its limit. Starting anywhere else means building those on a
CPU-bound draw loop and reworking them later.

---

## 4. v1.0.5 in one paragraph — ✅ shipped

Geometry gains a **meshlet build step** (`geometry-meshlets-design.md`): a mesh is partitioned
into clusters of ≤64 vertices / ≤124 triangles, each with a bounding sphere and a normal cone,
packed into flat GPU buffers. A new **`VisibilitySystem`** (`visibility-design.md`) runs before
the scene pass: it builds a Hi-Z depth pyramid from last frame's depth, then a compute stage
culls instances and meshlets against the frustum, the normal cone, and Hi-Z occlusion, expands
the survivors into a compacted index buffer, and writes the indexed-indirect draw arguments. The
scene pass draws that with **one `drawIndexedIndirect` per material** — the exact call
`RenderPipeline.ts:140` already makes. No core-engine change is required for the draw path; the
one optional additive change is indirect *dispatch* (§6). The result: draw-call count and
CPU-side culling become independent of scene size, which is the entire premise of "massive web
worlds."

### 4.1 How Megabuffer, Meshlets, and Visibility chain

These three are levels of one pipeline, not alternatives — and the first already ships. They are
easy to confuse, so, top to bottom of the hierarchy `meshes → meshlets → triangles → vertices`:

| Layer | Component | Role | Status |
|---|---|---|---|
| Vertex/index pool | **`MegabufferBuilder`** | Packs every mesh into one vertex + index buffer; records per-mesh `{indexCount, firstIndex, baseVertex, meshID}`. | ✅ ships today |
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
pool**, so a single vertex allocation backs both tiers. The instance tier (coarse, ships-shaped
today) reads the megabuffer's per-mesh offsets; the meshlet tier (fine) reads the meshlet buffers;
both pull the same vertices. You can stop at any layer: megabuffer + instance-tier culling is a
complete meshlet-free path; add `MeshletBuilder` only when you want sub-object granularity. See
`geometry-meshlets-design.md` §2.1 for the exact ownership split.

For end-to-end scene assembly (where these sit relative to lights, materials, and post), see
[`scene-composition.md`](./scene-composition.md).

---

## 5. Per-module design briefs

Every module in the tree gets a brief here: its goal, the **real integration surface** it hangs
off, its dependencies, and its explicit non-goals. A brief becomes a full standalone spec when
its release is scheduled — the two v1.0.5 modules already have theirs.

### 5.1 Geometry → Meshlets & LOD — **v1.0.5 ✅ shipped** — [full spec](./geometry-meshlets-design.md)

- **Goal.** Turn a mesh into GPU-cullable clusters and discrete LOD chains; define the flat
  buffer layout the `VisibilitySystem` consumes.
- **Integration surface.** Extends `MegabufferBuilder` (`src/extras/geometry/MegaBufferBuilder.ts`),
  whose `MeshOffset { indexCount, firstIndex, baseVertex, meshID }` is already the right shape.
  Output buffers are plain `STORAGE` buffers, uploaded via `BufferManager`.
- **Depends on.** Nothing new. Build-time only.
- **Non-goals (v1.0.5).** The DAG cluster-LOD graph and runtime streaming of virtual geometry
  (v2.0); vertex quantization/compression (v1.1).

### 5.2 Visibility → GPU culling — **v1.0.5 ✅ shipped** — [full spec](./visibility-design.md) · [API](./api/visibility.md)

- **Goal.** Make visible-set determination O(GPU) and scene-size-independent: frustum + normal
  cone + Hi-Z occlusion, producing indirect draw args.
- **Integration surface.** Emits `ComputeStageNode`s (the `LightSystem.getComputeStage` pattern,
  `LightSystem.ts:381`) that write a batch's `indirectBuffer`; the scene batch is flagged
  `isIndirect` and drawn by the existing `RenderPipeline.ts:137-140`. Consumes a **samplable**
  depth texture (note: `ctx.depthTexture` is `RENDER_ATTACHMENT`-only, `WebGPUContext.ts:126`, so
  a depth pre-pass or the `PostChain` g-buffer depth supplies it).
- **Depends on.** §5.1 meshlet buffers (for meshlet-tier culling). Instance-tier culling needs
  only per-instance bounds.
- **Non-goals (v1.0.5).** The deferred visibility-buffer *shading* model (v1.1); persistent
  GPU-side scene management.

### 5.3 Rendering → Indirect / Multi-draw / Deferred

- **Indirect / GPU-driven (v1.0.5).** Not a separate module — it is the draw side of Visibility.
  Already supported; §4.
- **Multi-draw fast path (v1.1, detect-only).** `multiDrawIndexedIndirect` would let one call
  draw many meshlets with distinct index ranges, avoiding the index-compaction pass in
  `visibility-design.md` §7. It is **not** core WebGPU (Chromium-experimental). Per constraint
  #4: detect `capabilities.has('multi-draw-indirect')`, keep the compaction baseline, use
  multi-draw only when present.
- **Deferred / Visibility-buffer shading (v1.1).** Write a triangle/meshlet-ID buffer instead of
  shaded color, then shade in a fullscreen compute pass. Reuses the `PostChain`/`TransientPool`
  target machinery and the material shader chunks. Depends on v1.0.5 meshlet IDs.

### 5.4 Shadows — **v1.1 / v1.2** — [Phase 1 spec](./shadows-design.md) · [Phase 2 spec](./shadows-atlas-design.md)

- **Goal.** Directional shadow maps → CSM (Phase 1, v1.1) → shadow atlas with punctual lights,
  tile caching, and GPU caster culling (Phase 2, v1.2).
- **Integration surface.** A shadow map is an offscreen depth `RenderPassNode` rendered from the
  light's viewpoint with the same indirect batches as the main pass — so **CSM caster culling is
  the `VisibilitySystem` run per cascade with a different frustum**, and a cascade is literally a
  108-float `Camera` (shadows-design §4.1). Receiver-side, shadowing **decorates** the
  `LightSystem` shader layout: it wraps `getIncidentLight` and folds the shadow term into
  `IncidentLight.attenuation`, so Lambert/Toon/PBR all gain shadows unedited (shadows-design §8.1).
- **Depends on.** v1.0.5 (per-frustum culling), depth-only pipeline variants.
- **The engine changes this needs** — the first release to need any, all additive and inert for
  existing callers: `PipelineConfig.depthOnly` (a fragment-less pipeline; `createBatch` always
  builds a color target today, `BatchManager.ts:37-61`) and `PipelineConfig.cameraBuffer` (group 0
  binding 0 is hard-wired to the one global camera, `BatchManager.ts:100`) in Phase 1; and for
  Phase 2's atlas, a per-batch `viewport`, a `RenderPassConfig.size` so `RenderPipeline` can
  restore the full viewport between tiled and full-pass batches (viewport is **pass** state — a
  full-pass batch after a tiled one otherwise inherits its tile), and
  `ComputeStageNode.enabled` so a cached tile can skip its caster cull as well as its draw. All
  are additive and default to today's behavior. Each has a documented zero-change fallback, and
  each fallback is worse — see shadows-design §12 and shadows-atlas-design §12.
- **Non-goals.** Virtual shadow maps (v2.0), VSM/ESM/PCSS filtering, ray-traced shadows (⛔).

### 5.5 Lighting → IBL & Probes — **v1.2** — [Phase 1 spec](./lighting-ibl-design.md) · [Phase 2 spec](./lighting-probes-design.md)

- **Goal.** Give NullGraph an indirect lighting term. Today the *entire* indirect contribution in
  the PBR fragment is `let ambient = vec3<f32>(0.3) * albedo * ao;` — which is why metals read flat
  and why no work on the direct loop finishes a scene. Phase 1 (v1.2) replaces it with image-based
  lighting: prefiltered env cube + split-sum BRDF LUT + SH-L2 irradiance. Phase 2 (v1.2) makes it
  vary through space: SH-L1 irradiance volumes, GPU probe capture with multi-bounce, and local
  parallax-corrected reflection probes.
- **Integration surface.** A **sibling of `ShaderLightingLayout`**: `ShaderIndirectLayout` provides
  `getIndirectDiffuse(worldPos, N)` and `getIndirectSpecular(worldPos, N, V, roughness, F0)`,
  composed into materials exactly as `LightSystem.getShaderLayout()` already is
  (`materialFragments.ts:568-576`). Phase 2 re-emits the **same two functions** against a probe
  volume, so it costs zero material edits — which is why the contract takes `worldPos` in Phase 1
  even though Phase 1 ignores it.
- **The constraint that shapes both phases.** Core WebGPU guarantees **`maxBindGroups: 4`**, and
  NullGraph already spends all four: 0 = camera + instances, 1 = material, 2 = lights (hardcoded
  `@group(2)` inside `LightSystem`'s WGSL), 3 = shadows. Indirect lighting therefore lives **inside
  group 2 alongside the lights**, never in a new group. The full binding map and the per-stage
  resource counts are in `lighting-ibl-design.md` §4.4 — it fits, exactly, with no slack. Any future
  fragment-stage subsystem must join an existing group and declare it there.
- **Depends on.** Phase 1: nothing — it is the first module since v1.0.5 needing **no core-engine
  change** (compute + `rgba16float`/`rg16float` storage textures + cube views are all core).
  Phase 2: the two additions shadows already introduced (`cameraBuffer`, per-batch `viewport`),
  because probe capture *is* the shadow atlas rendered in color.
- **Non-goals.** DDGI per-probe visibility, probe streaming, SSR (Post's job), ray-traced GI (⛔).

### 5.6 Effects → Particles & Decals — **v1.2**

- **Goal.** Compute-simulated GPU particles; clustered/deferred decals.
- **Integration surface.** Particles are the canonical DOD case: a compute stage integrates a
  `Float32Array`-backed particle buffer, then an `isIndirect` batch draws the live count — the
  exact instance-culling shape from v1.0.5. Decals reuse the clustered grid from `LightSystem`.
- **Depends on.** v1.0.5 (indirect instancing), v1.1 (deferred, for deferred decals).

### 5.7 Animation → GPU skinning & morphs — **v1.3**

- **Goal.** Move skinning and morph blending from CPU to a compute pre-pass.
- **Integration surface.** `GLBParser`/`SkeletonManager` already parse joints/weights and
  `Animator` samples poses (`src/extras/loaders/`). v1.3 adds a compute stage that reads bind-pose
  mega-vertices + a joint-matrix buffer and writes a skinned mega-vertex buffer, which then feeds
  the **same** meshlet/visibility path. Skinning bounds are refit per frame (loosened spheres) so
  culling stays valid.
- **Depends on.** v1.0.5 mega-buffer conventions.

### 5.8 Post → TAA & Motion blur — **v2.0**

- **Goal.** Temporal antialiasing and velocity-based motion blur.
- **Integration surface.** `TransientPool` **already supports** `history: 2` double-buffered
  targets (`transient-pool.md` §8) precisely so TAA needs no allocator change. Needs a velocity
  g-buffer target (`buildPBRShader({ motionVectors: true })`, anticipated in
  `post-processing-design.md` §13) and camera jitter.
- **Depends on.** A velocity buffer (a deferred/g-buffer output, v1.1).

### 5.9 RayTracing — **⛔ blocked on WebGPU**

- **Reality.** WebGPU has no acceleration-structure or ray-query API, and none is in the
  candidate spec as of this writing. This branch of the wishlist cannot be built, only prepared
  for. Per constraint #4 it is **detect-and-adopt**: when the platform ships it, surface the
  feature on `capabilities` and design then. Do not stub BLAS/TLAS types now — they would encode
  guesses about an API that does not exist. RT reflections/shadows/GI in the meantime are
  screen-space (Post) or probe-based (v1.2) approximations, not stand-ins to be swapped out.

---

## 6. Cross-cutting: the WebGPU reality, and the one engine change

The wishlist was written at the abstraction level of a native mesh-shader engine. Three of its
assumptions do not hold on WebGPU, and pretending otherwise is how a plan rots:

| Native assumption | WebGPU reality | Consequence for the plan |
|---|---|---|
| Mesh shaders expand meshlets | none | Meshlets are drawn via **index compaction + `drawIndexedIndirect`** (`visibility-design.md` §7), or the meshlet-as-instance vertex-pulling variant. Both are core. |
| `multiDrawIndexedIndirect` | optional, non-universal | Baseline is **one indirect draw per material** over a compacted index buffer. Multi-draw is a v1.1 *detected* fast path, never the baseline. |
| Ray tracing | none | The whole RT branch is ⛔ (§5.9). |
| Indirect dispatch | **core, but not yet wired in NullGraph** | The one additive engine change below. |
| Bind groups are plentiful | **`maxBindGroups: 4` is the core guarantee** | See §6.1 — the budget is already fully spent, and it is the hardest constraint in the plan. |

**The only engine change v1.0.5 wants** (and it is optional, an efficiency step, not a
correctness one): expose **indirect dispatch**. Today `RenderPipeline` calls
`dispatchWorkgroups(x,y,z)` from `node.dispatchCount` (`RenderPipeline.ts:58-59`) — a CPU-known
count. Two-phase culling is most efficient when phase 2 dispatches exactly the number of objects
phase 1 deferred, which is a GPU-side count. The baseline design **avoids needing this** by
dispatching over total instance capacity and early-outing idle threads (the same way the
clustered binner dispatches over all clusters, `LightSystem.ts:387`). Adding
`dispatchWorkgroupsIndirect` support to `ComputeStageNode` later removes that wasted dispatch.
It is filed as a v1.0.5-optional, not a blocker. Specced in `visibility-design.md` §11.

Everything else v1.0.5 needs — storage buffers, atomics, `INDEX | STORAGE` buffers, `r32float`
storage textures for Hi-Z, `drawIndexedIndirect` — is core WebGPU and already reachable through
the current API surface.

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
per-stage storage-buffer / sampled-texture / sampler / uniform counts against their core limits —
they fit, with no slack).

Two consequences worth stating before they are rediscovered:

- **Group 2 now has three owners** (lights, environment, probes), and exactly one of them may build
  the bind group. That is why `EnvironmentSystem.applyToBatch(batch, lights)` and
  `ProbeVolume.applyToBatch(batch, lights, env, refl)` take their peers as arguments.
- **A latent collision already exists:** `CommonChunks.SkinningBindings` and
  `PBRChunks.SkinningBindings` also declare `@group(2) @binding(0)`
  (`CommonChunks.ts`, `PBRShaderChunks.ts`), so CPU-side skinning and clustered lighting cannot
  currently coexist in one shader. v1.3's GPU skinning (§5.7) moves skinning to a compute pre-pass
  and dissolves the conflict — but until then, it is a real constraint, not a hypothetical one.

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
              per-frustum    │           │  meshlet IDs
              reuse          │           │
             ┌───────────────▼──┐   ┌────▼────────────────┐
      v1.1   │ Shadows P1 (CSM) │   │ Deferred / vis-buffer│  ── velocity buffer ──┐
             │ — free GPU       │   │ shading              │                       │
             │ caster culling   │   └────┬─────────────────┘                       │
             └────────┬─────────┘        │                                          │
                      │ ShadowView + atlas│                                          │
             ┌────────▼─────────┐        │                                          │
      v1.2   │ Shadows P2       │        │                                          │
             │ (atlas, punctual,│        │                                          │
             │  caching, Hi-Z)  │        │                                          │
             └────────┬─────────┘        │                                          │
                      │ the atlas pass, reused in COLOR                             │
                      │  (cameraBuffer + viewport)                                  │
      ┌───────────────┴───────────────┐  │                                          │
      │                               ▼  ▼                                          │
┌─────▼──────────┐            ┌───────────────┐                              ┌──────▼──────┐
│ Lighting P1    │  contract  │ Effects (GPU  │                         v2.0 │ TAA / motion │
│ IBL: env cube, │───────────▶│ particles,    │                              │ blur         │
│ BRDF LUT, SH-L2│  getIndirect│ decals)      │                              └─────────────┘
└─────┬──────────┘  Diffuse/  └───────────────┘
      │             Specular
┌─────▼──────────┐                ┌──────────────┐                           ┌──────────────┐
│ Lighting P2    │       v1.3     │ GPU skinning │ ── skinned mega-buffer ──▶ │ virtual geo  │ v2.0
│ probe volumes, │                └──────────────┘   feeds same vis path      │ (DAG LOD)    │
│ multi-bounce,  │                                                            └──────────────┘
│ refl probes    │
└────────────────┘
   ⛔ RayTracing — off the graph until WebGPU exposes acceleration structures.
```

The critical path is **meshlet data → Visibility**, and everything of value in v1.1–v2.0 hangs
off Visibility. That is the justification for the whole release ordering: build the culling
substrate once, correctly, and let shadows, deferred shading, particles, and eventually virtual
geometry inherit it.

---

## 8. What "done" means for v1.0.5 — scorecard

The acceptance gates the two specs elaborate, against what shipped:

| # | Gate | Status |
|---|---|---|
| 1 | A scene of N instances renders with **draw calls proportional to material/mesh count, not N** | ✅ by construction — one `drawIndexedIndirect` per mesh (instance tier) / per material (meshlet tier). Needs a GPU harness to assert numerically (§10 T2). |
| 2 | Meshlets behind other geometry are **not drawn** (Hi-Z) | ✅ implemented (`HIZ_OCCLUSION_WGSL`); the numeric scene test (§10 T4) awaits a browser harness. |
| 3 | Culling driven entirely from **worker-fillable** GPU buffers, **no per-frame allocation** | ✅ `InstanceBuffer` + `setInstances` adopt by reference; the frame path is one `writeBuffer`, the param refresh, clears, dispatches, draws. |
| 4 | Baseline runs on **core WebGPU**, optional features only accelerate | ✅ storage buffers, atomics, `r32float` storage textures, `drawIndexedIndirect` — all core. No feature gates. |
| 5 | **No core-engine changes** for the draw path | ✅ confirmed — `src/core/` untouched by v1.0.5; indirect dispatch remains optional (§6). |

**Remaining before the gates are fully *proven* (not just met):** a browser/WebGPU test harness for
§10 T1–T7. The pure math and every generated WGSL module are already covered headless by
`npm run test:visibility`.
