# NullGraph — Architecture Tree

**Status:** the canonical feature map. One tree per module, expanded to full depth.
**Companions:** [`roadmap.md`](./roadmap.md) (release plan, WebGPU realities, scorecards) ·
[`roadmap-modules.md`](./roadmap-modules.md) (why each branch is shaped the way it is)

`roadmap.md` says **when**, `roadmap-modules.md` says **how**, this file says **what** — every
module gets its own section and its own tree, so a branch can be read to whatever depth it actually
has without being squeezed into a single page-wide diagram.

**Legend**

| Mark | Meaning |
|---|---|
| `✅` | shipped and exported |
| `◐` | partial — works, but read the note; this is where half-truths hide |
| `◻` | planned, not started |
| `✍` | designed (implementation-ready spec exists), not implemented |
| `⛔` | blocked on WebGPU itself, not on us |
| `🔌` | plugin — ships on its own cadence, gates no core release |

**Index**

| § | Module | Package | Status |
|---|---|---|---|
| [1](#1-core) | Core | `null-graph` | ✅ |
| [2](#2-geometry) | Geometry | `null-graph/geometry` | ✅ |
| [3](#3-visibility) | Visibility | `null-graph/visibility` | ✅ |
| [4](#4-rendering) | Rendering | (core) | ◐ |
| [5](#5-materials) | Materials | `null-graph/materials` | ✅ |
| [6](#6-shadows) | Shadows | `null-graph/shadows` | ✅ |
| [7](#7-lighting) | Lighting | `null-graph/lights` | ✅ |
| [8](#8-environment) | Environment | `null-graph/environment` | ◐ · sky Phases 1–5 ✅, clouds/weather/wind ◻ |
| [9](#9-effects) | Effects | `null-graph/effects` | ◻ |
| [10](#10-physics) | **Physics** | `null-graph/physics` | ◻ 🔌 · [P1–P3 ✍ specced](./physics/) |
| [11](#11-animation) | Animation | `null-graph/loaders` | ◐ |
| [12](#12-post) | Post | `null-graph/post` | ✅ |
| [13](#13-scene--streaming) | Scene & Streaming | — | ◻ |
| [14](#14-loaders--assets) | Loaders & Assets | `null-graph/loaders` | ✅ |
| [15](#15-tooling--debug) | Tooling & Debug | `null-graph/profiler`, `/debug-ui` | ✅ |
| [16](#16-platform) | Platform | (core) | ✅ |
| [17](#17-raytracing) | RayTracing | — | ⛔ |
| [18](#18-picking) | Picking | `null-graph/picking` | ✅ |
| [19](#19-text) | Text | `null-graph/text` | ✅ |

---

## 1. Core

**Package:** `null-graph` · **Status:** ✅ shipped · **Brief:** —

```
Core
│
├── Device & context ──────────────────────────────────────────── ✅
│   ├── Adapter request (power preference, fallback)              ✅
│   ├── Feature negotiation
│   │   ├── PROGRESSION_FEATURES opt-in set                       ✅  never required, only enabled
│   │   ├── Capability snapshot (`engine.capabilities`)           ✅
│   │   └── Subgroup min/max size reporting                       ✅
│   ├── Canvas configure + resize                                 ✅
│   ├── Depth texture (samplableDepth: true opts into sampling)   ✅  opt-in TEXTURE_BINDING; off by default
│   ├── Error scopes / validation surfacing                       ◻ v1.3
│   └── Device-loss + context-restore                             ◻ v1.3  rebuild pipelines, reupload
│
├── Execution model ───────────────────────────────────────────── ✅  the whole scheduler
│   ├── Linear node array (`ComputeStageNode | RenderPassNode`)   ✅
│   ├── Single command encoder per frame                          ✅
│   ├── Automatic compute-pass batching                           ✅  consecutive stages share one pass
│   ├── Pass-boundary barriers (compute → draw)                   ✅  free, from the pass split
│   ├── Declarative buffer clears (`preClearBuffers`)             ✅  runs before its stage
│   ├── Gateable stages (`enabled`)                               ✅ v1.2  skips clears AND dispatch
│   ├── Dynamic dispatch count (callback form)                    ✅  CPU-known values only
│   ├── Indirect draw (`drawIndexedIndirect`)                     ✅  the hook v1.0.5 consumed
│   ├── Indirect dispatch (`dispatchWorkgroupsIndirect`)          ✅  exact counts; consumers not yet wired
│   └── Render bundles (encode once, replay)                      ◻ v2.0  static sub-scenes
│
├── Resources ─────────────────────────────────────────────────── ✅
│   ├── Buffer manager                                            ✅
│   ├── Texture manager
│   │   ├── sRGB-aware load + decode                              ✅
│   │   ├── Sampler cache                                         ✅
│   │   └── Solid-colour fallbacks (white / normal)               ✅
│   ├── Batch creation
│   │   ├── Pipeline from WGSL + layout:'auto'                    ✅
│   │   ├── Group-0 bind group (camera + storage)                 ✅
│   │   ├── Indirect buffers (5 × u32) + shared source            ✅
│   │   ├── Depth-only (fragment-less) pipelines                  ✅ v1.1
│   │   ├── Per-batch camera override                             ✅ v1.1  enables EVERY off-main view
│   │   ├── Per-batch viewport + scissor                          ✅ v1.2
│   │   └── Mixed-viewport guard (loud refusal)                   ✅ v1.2
│   ├── Render passes
│   │   ├── Offscreen colour + depth attachments                  ✅
│   │   ├── Main-screen swapchain path                            ✅
│   │   └── Pass size (viewport restore semantics)                ✅ v1.2
│   ├── Explicit pipeline cache / warmup                          ◻ v1.3  first-use compile hitches
│   └── Explicit bind-group layouts (drop `layout:'auto'`)        ◻ v1.3  would relax the group budget
│
└── Camera ────────────────────────────────────────────────────── ✅
    ├── 108-float uniform (viewProj/view/proj + 3 inverses)       ✅  every off-main view fills this
    ├── eye · zNear · zFar · fovRads · aspect · simTime           ✅
    ├── ZO clip convention (`perspectiveZO` / `orthoZO`)          ✅  the NO forms move the near plane
    └── Sub-pixel jitter (for TAA)                                ◻ v2.0
```

**Next up:** wiring the existing consumers to indirect dispatch. The primitive ships
(`ComputeStageConfig.indirectBuffer`, `bufferManager.createDispatchIndirectBuffer`), but
`VisibilitySystem` and the shadow caster culls still size themselves against a CPU-known cap —
`ceil(instanceCount / 64)` — because switching them means each cull's *producer* must write its own
dispatch triple. That is per-consumer work, not core work, and each one can move independently.

---

## 2. Geometry

**Package:** `null-graph/geometry` · **Status:** ✅ shipped ·
**Spec:** [`geometry-meshlets-design.md`](./geometry-meshlets-design.md) ·
**Brief:** [`roadmap-modules.md`](./roadmap-modules.md) §1

```
Geometry
│
├── Authoring ─────────────────────────────────────────────────── ✅
│   ├── Primitives (13)
│   │   ├── Cube · Sphere · Plane · Cylinder · Cone               ✅
│   │   ├── Torus · Ring · Capsule · Pyramid                      ✅
│   │   └── Tetra · Octa · Dodeca · Icosahedron                   ✅
│   ├── GeometryBuilder / Geometry                                ✅
│   ├── Declarative vertex layouts                                ✅
│   └── Dynamic (rebuildable) geometry                            ✅
│
├── Packing ───────────────────────────────────────────────────── ✅
│   ├── Megabuffer
│   │   ├── One vertex + index pool for every mesh                ✅
│   │   └── Per-mesh {indexCount, firstIndex, baseVertex, meshID} ✅
│   ├── Meshlets ─────────────────────────────────── ✅ v1.0.5
│   │   ├── Cluster partition (≤64 verts / ≤124 tris)             ✅
│   │   ├── Bounding sphere per meshlet                           ✅
│   │   ├── Normal cone per meshlet (+ never-cull sentinel)       ✅
│   │   ├── Global indices into the mega pool                     ✅  one allocation backs both tiers
│   │   └── Packed triangle bytes (3 × u8 per tri)                ✅
│   ├── LOD chains
│   │   ├── Discrete levels + geometric error                     ✅ v1.0.5
│   │   ├── Mesh → LOD → meshlet table                            ✅ v1.0.5
│   │   └── Simplification at build time                          ◐   caller supplies the LODs today
│   └── Compression
│       ├── Vertex quantization (snorm16 pos, oct normals)        ◻ v1.1  ~2× vertex bandwidth
│       └── Index compression (meshopt-style)                     ◻ v1.1
│
├── Attributes ────────────────────────────────────────────────── ◐
│   ├── Position / normal / UV                                    ✅
│   ├── Joints / weights (skinning)                               ✅
│   ├── Vertex colours                                            ◐   layout carries them; no material reads them
│   ├── Tangent generation (mikktspace-style)                     ◻ v1.3  TBN is derivative-based today
│   └── Multiple UV sets (lightmap / detail)                      ◻ v1.3
│
└── Virtual geometry ──────────────────────────────────────────── ◻ v2.0
    ├── DAG cluster-LOD graph                                     ◻ v2.0  Nanite-class; the stretch goal
    ├── Cluster group simplification                              ◻ v2.0
    └── Runtime cluster streaming                                 ◻ v2.0  shares residency with §13
```

**Next up:** vertex quantization (v1.1) — build-time only, and it composes with meshlets because
the meshlet buffers store indices, not positions.

---

## 3. Visibility

**Package:** `null-graph/visibility` · **Status:** ✅ shipped ·
**Spec:** [`visibility-design.md`](./visibility-design.md) ·
**Brief:** [`roadmap-modules.md`](./roadmap-modules.md) §2

```
Visibility
│
├── Culling tests ─────────────────────────────────────────────── ✅
│   ├── Frustum (Gribb–Hartmann from any viewProj)                ✅ v1.0.5
│   │   └── ZO near plane (r2, not r3+r2)                         ✅  the OpenGL form is wrong here
│   ├── Hi-Z occlusion
│   │   ├── r32float max-reduction mip pyramid                    ✅ v1.0.5
│   │   ├── Conservative projected-sphere test                    ✅ v1.0.5
│   │   ├── Near-plane crossing guard (keep visible)              ✅ v1.0.5
│   │   └── Single-phase (last frame's depth)                     ✅ v1.0.5  one frame of latency
│   ├── Meshlet normal cone (backface clusters)                   ✅ v1.0.5
│   ├── Small-triangle / sub-pixel rejection                      ◻ v1.1
│   └── Distance / per-instance flags                             ◐   flags exist; distance feeds LOD only
│
├── Tiers ─────────────────────────────────────────────────────── ✅
│   ├── Instance tier
│   │   ├── One indirect draw per unique mesh                     ✅ v1.0.5  independent of instance count
│   │   └── Compaction via atomicAdd on drawArgs[1]               ✅ v1.0.5
│   └── Meshlet tier
│       ├── Two-pass cull (instance → meshlet)                    ✅ v1.0.5
│       ├── Meshlet-as-instance vertex pulling                    ✅ v1.0.5  no mesh shaders on WebGPU
│       └── Static index buffer [0..371]                          ✅ v1.0.5
│
├── LOD ───────────────────────────────────────────────────────── ✅
│   ├── Screen-space-error selection                              ✅ v1.0.5
│   └── Per-instance override (LOD_AUTO sentinel)                 ✅ v1.0.5
│
├── Per-view culling ──────────────────────────────────────────── ✅ v1.1
│   ├── Arbitrary camera buffer per batch                         ✅  shadows proved it
│   ├── Per-batch occlusion toggle                                ✅
│   ├── Hi-Z UV remap (tiled / atlas views)                       ✅ v1.2
│   └── Per-view screen size (tile-correct footprints)            ✅ v1.2
│
├── Zero-copy inputs ──────────────────────────────────────────── ✅
│   ├── InstanceBuffer (24 floats: model, bounds, mesh, flags)    ✅
│   ├── Adopt-by-reference (`setInstances`)                       ✅
│   └── External buffer backing (caller-owned)                    ◻ 🔌  transport-agnostic; §10 P1
│
├── Two-phase temporal culling                                    ◻ reserved  throws today; design §9
├── Visibility buffer (deferred triangle/meshlet IDs)             ◻ v1.1
└── Persistent GPU scene (residency)                              ◻ v2.0   on-ramp to virtual geometry
```

**Next up:** the visibility buffer (v1.1) — it is what unlocks deferred shading in §4.

---

## 4. Rendering

**Package:** (core) · **Status:** ◐ forward + GPU-driven ·
**Brief:** [`roadmap-modules.md`](./roadmap-modules.md) §3

```
Rendering
│
├── Forward ───────────────────────────────────────────────────── ✅
│   ├── Material batches                                          ✅
│   ├── Forward+ / tiled                                          ✅
│   └── Clustered (froxel grid)                                   ✅  LightSystem compute stage
│
├── GPU-driven ────────────────────────────────────────────────── ✅ v1.0.5
│   ├── Indirect draw from a compute-filled buffer                ✅
│   ├── Draw count ∝ material/mesh count, not object count        ✅
│   └── Multi-draw fast path                                      ◻ v1.1  detect-only; NOT core WebGPU
│
├── Depth ─────────────────────────────────────────────────────── ◐
│   ├── Depth pre-pass                                            ◐   caller-assembled; Hi-Z needs it
│   ├── Depth-only pipelines                                      ✅ v1.1
│   └── Reverse-Z                                                 ◻ v1.3  precision; touches every projection
│
├── Deferred ──────────────────────────────────────────────────── ◻ v1.1
│   ├── Visibility-buffer (ID) pass                               ◻ v1.1
│   ├── Fullscreen compute shading                                ◻ v1.1  reuses PostChain targets
│   └── G-buffer layout
│       ├── Albedo / normal / ARM                                 ◻ v1.1
│       └── Velocity (motion vectors)                             ◻ v1.1  Post's TAA depends on it
│
├── Transparency ──────────────────────────────────────────────── ◐
│   ├── Alpha blend / additive states                             ✅
│   ├── No ordering model                                         ◐   batches draw in array order, full stop
│   ├── Sorted transparency (GPU sort)                            ◻ v1.2  fights GPU-driven drawing
│   └── Weighted-blended OIT                                      ◻ v1.2  order-independent; preferred
│
└── MSAA + resolve                                                ◻ v1.3
```

**Next up:** the g-buffer layout (v1.1) — it is the shared dependency of deferred shading, TAA, and
motion blur, so its shape should be decided once and early.

---

## 5. Materials

**Package:** `null-graph/materials` · **Status:** ✅ shipped ·
**Brief:** [`roadmap-modules.md`](./roadmap-modules.md) §7

```
Materials
│
├── Families ──────────────────────────────────────────────────── ✅
│   ├── Basic (albedo + tint)                                     ✅
│   ├── Lambert (+ half-lambert option)                           ✅
│   ├── Emissive (+ pulse)                                        ✅
│   ├── Matcap                                                    ✅
│   ├── Toon (ramp, rim, outline, hard specular)                  ✅
│   └── PBR (Cook-Torrance: GGX + Smith + Schlick)                ✅
│
├── The shared contract ───────────────────────────────────────── ✅  the most-extended interface here
│   ├── Portable {bindings, fragment} pair                        ✅  one source of truth
│   ├── Parameterized by bind-group index                         ✅  group 1 normal / group 2 meshlet tier
│   ├── Injected direct-lighting layout                           ✅  getVisibleLightCount/getIncidentLight
│   ├── Injected shadow decorator                                 ✅ v1.1  folds into `attenuation`
│   └── Injected indirect-lighting layout                         ✅ v1.2  getIndirect{Diffuse,Specular}
│
├── Shading options ───────────────────────────────────────────── ◐
│   ├── Inline tonemap + gamma                                    ◐   double-applies with PostChain
│   ├── Opt-out (`tonemap: false`)                                ◻ v1.2  IBL makes it visible
│   ├── Motion-vector output                                      ◻ v2.0
│   └── G-buffer output mode                                      ◻ v1.1
│
├── Texture inputs ────────────────────────────────────────────── ✅
│   ├── Albedo / normal / packed ARM|MRAO                         ✅
│   ├── Emissive map                                              ✅
│   ├── Derivative-based TBN                                      ◐   wrong on mirrored UVs
│   └── Authored tangents                                         ◻ v1.3
│
└── Advanced BRDF ─────────────────────────────────────────────── ◻
    ├── Parallax / height mapping                                 ◻ v1.3
    ├── Clear coat                                                ◻ v1.3
    ├── Sheen / cloth                                             ◻ v1.3
    ├── Anisotropy                                                ◻ v1.3
    ├── Transmission / refraction                                 ◻ v2.0
    └── Triplanar projection                                      ◻ v2.0  pairs with terrain
```

**Next up:** `tonemap: false` (v1.2) — a one-line branch, and IBL is what makes the double-apply
impossible to ignore.

---

## 6. Shadows

**Package:** `null-graph/shadows` · **Status:** ✅ shipped ·
**Specs:** [`shadows-design.md`](./shadows-design.md) ·
[`shadows-atlas-design.md`](./shadows-atlas-design.md) ·
**Brief:** [`roadmap-modules.md`](./roadmap-modules.md) §4

```
Shadows
│
├── Directional (Phase 1) ─────────────────────────────────────── ✅ v1.1
│   ├── Single shadow map                                         ✅
│   ├── Cascades
│   │   ├── Practical split scheme (log ↔ uniform, λ)             ✅
│   │   ├── Rotation-invariant sphere fit                         ✅  a box fit shimmers on yaw
│   │   ├── Texel snapping in light space                         ✅  unsnapped centres crawl
│   │   ├── Caster extrusion (off-screen casters)                 ✅  the pop-in fix
│   │   └── Cross-fade band between cascades                      ✅
│   ├── Virtual fov (LOD stays correct in an ortho view)          ✅
│   └── depth24plus 2D array, one layer per cascade               ✅
│
├── Punctual (Phase 2) ────────────────────────────────────────── ✅ v1.2
│   ├── Spot (fov derived from the cone cosine)                   ✅  cone and shadow cannot disagree
│   └── Point (6 cube faces, +X−X+Y−Y+Z−Z)                        ✅
│       ├── Widened fov margin (filter overlap)                   ✅
│       └── Major-axis face select                                ✅
│
├── Atlas (Phase 2) ───────────────────────────────────────────── ✅ v1.2
│   ├── Quad-tree free-list allocator                             ✅
│   ├── Importance heuristic + tile ladder                        ✅
│   ├── Resize hysteresis                                         ✅  a resize is a full re-render
│   ├── Eviction (fails open — never black)                       ✅
│   ├── Cascades pinned (never evicted)                           ✅
│   ├── One atlas pass, viewport-scoped batches                   ✅  clip-space remap would bleed
│   ├── Per-tile clear (3-vertex quad)                            ✅
│   └── Static/dynamic split maps                                 ◻ v2.0  removes reliance on invalidate()
│
├── Caching (Phase 2) ─────────────────────────────────────────── ✅ v1.2
│   ├── Declared dirtiness (`invalidate`)                         ✅  no scene graph to infer from
│   ├── Matrix-change detection                                   ✅
│   ├── Round-robin refresh                                       ✅  bounds a missed invalidate
│   └── Gates the DRAW and the CULL together                      ✅  viewport alone suppresses only the draw
│
├── Culling ───────────────────────────────────────────────────── ✅
│   ├── GPU caster culling per view                               ✅ v1.1  the visibility chain, re-pointed
│   └── Light-view Hi-Z                                           ✅ v1.2  atlas-wide is provably conservative
│
├── Bias ──────────────────────────────────────────────────────── ✅ v1.1
│   ├── Constant bias in clip space                               ✅  no depthBias in PipelineConfig
│   ├── Normal offset scaled by world texel size                  ✅  resolution-correct
│   └── Peter-panning guard (loud error)                          ✅
│
├── Filtering ─────────────────────────────────────────────────── ◐
│   ├── Hardware 2×2 comparison PCF                               ✅ v1.1
│   ├── N×N tap kernel (`pcfRadius`)                              ✅ v1.1
│   ├── Tap clamp inside the tile                                 ✅ v1.2  a tap must not read a neighbour
│   ├── PCSS / contact hardening                                  ◻ v1.3  blocker search over the same atlas
│   └── Filtered moments (VSM / ESM / MSM)                        ◻ v1.3  needs a filterable colour atlas
│
├── Receiver bridge ───────────────────────────────────────────── ✅
│   ├── `wrapLighting` decorator                                  ✅  zero material edits
│   ├── Cascade select by view depth                              ✅
│   └── Per-light shadow slot (`spot_params.z`)                   ✅ v1.2  free field on every light type
│
├── Translucent / coloured shadows                                ◻ v2.0  needs an ordering model
└── Virtual shadow maps                                           ◻ v2.0  needs the v1.2 atlas to virtualize
```

**Next up:** PCSS (v1.3) — needs no new resources; the atlas and comparison sampler already exist.

---

## 7. Lighting

**Package:** `null-graph/lights` · **Status:** ◐ direct ships, indirect is v1.2 ·
**Specs:** [`lighting-ibl-design.md`](./lighting-ibl-design.md) ·
[`lighting-probes-design.md`](./lighting-probes-design.md) ·
**Brief:** [`roadmap-modules.md`](./roadmap-modules.md) §5

```
Lighting
│
├── Direct ────────────────────────────────────────────────────── ✅
│   ├── Light proxies (7 types)
│   │   ├── Point · Directional · Spot                            ✅
│   │   ├── Ambient · Hemisphere                                  ✅
│   │   └── RectArea · Torch                                      ✅
│   ├── Zero-copy 16-float records                                ✅  worker-fillable, adopted by reference
│   ├── Techniques
│   │   ├── `standard` (loop every light)                         ✅
│   │   ├── `tiled-forward`                                       ✅
│   │   └── `clustered-forward`                                   ✅
│   │       ├── Froxel grid + logarithmic depth slices            ✅
│   │       ├── Atomic binning baseline                           ✅
│   │       └── Subgroup-accelerated binning                      ✅  when the device negotiated it
│   ├── Shader layout injection                                   ✅  materials never learn the technique
│   ├── Shadow slot field (`spot_params.z`)                       ✅ v1.2
│   ├── Area lights via LTC                                       ◻ v1.3  rectArea is approximated today
│   └── Light cookies / IES profiles                              ◻ v1.3
│
├── Indirect — image-based (Phase 1) ──────────────────── ✅ v1.2
│   ├── The `ShaderIndirectLayout` contract                       ✅  the hook everything else fills
│   │   ├── getIndirectDiffuse(worldPos, N)                       ✅  takes worldPos so Phase 2 needs no edits
│   │   └── getIndirectSpecular(worldPos, N, V, rough, F0)        ✅
│   ├── GGX prefiltered specular mip chain                        ✅  solid-angle mip select kills fireflies
│   ├── Split-sum BRDF LUT (rg16float 256²)                       ✅  scene-independent, baked once
│   ├── SH-L2 diffuse irradiance (27 floats)                      ✅  a uniform, not a texture — budget
│   └── Amortized re-bake (one mip per frame)                     ✅  animated skies
│
├── Indirect — spatial (Phase 2) ──────────────────────── ✅ v1.2
│   ├── Irradiance volume
│   │   ├── SH-L1 in 3 × rgba16float texture_3d                   ✅  hardware trilinear = free 8-probe blend
│   │   ├── Validity mask (leak control)                          ✅
│   │   └── Normal + view bias                                    ✅  the cheapest leak mitigation
│   ├── GPU probe capture
│   │   ├── Capture atlas (the shadow atlas, in colour)           ✅
│   │   ├── SH-L1 projection compute                              ✅
│   │   ├── Amortized scheduling (probesPerFrame)                 ✅
│   │   └── Multi-bounce (capture reads last frame's volume)      ✅  GI convergence for free
│   ├── Local reflection probes
│   │   ├── texture_cube_array + prefilter                        ✅
│   │   ├── Parallax-corrected box proxy                          ✅  reproject from the CAPTURE point
│   │   └── Influence blending → global fallback                  ✅  energy-preserving
│   ├── DDGI per-probe visibility                                 ◻ v2.0  the real leak fix; needs bindings
│   └── Streamed / cascaded volumes                               ◻ v2.0
│
└── Lightmaps (baked static GI)                                   ◻ —   not planned; probes cover it
```

**Next up:** the `ShaderIndirectLayout` contract — landing the seam first means Phase 2 costs zero
material edits later.

---

## 8. Environment

**Package:** `null-graph/environment` · **Status:** ◐ — IBL bake + sky ✅ shipped ·
**Spec:** [`lighting-ibl-design.md`](./lighting-ibl-design.md) (the bake) ·
**Brief:** [`roadmap-modules.md`](./roadmap-modules.md) §6

**The sky series** — five phases; **Phases 1–3 are shipped**, 4–5 ✍ designed. Phase 1 draws the seam every later phase drops into;
each doc states what the one before it retires:

| Phase | Spec | Ships | Rel |
|---|---|---|---|
| 1 | [`environment-sky-design.md`](./environment-sky-design.md) | the `SkySource` / `sampleSky` seam; per-pixel skybox; Preetham | ✅ 1.1 |
| 2 | [`environment-timeofday-design.md`](./environment-timeofday-design.md) | the clock; sun/moon → lights, shadows, sky, exposure; the re-bake gate | ✅ v1.3 |
| 3 | [`environment-night-design.md`](./environment-night-design.md) | stars, moon, twilight, `SkyComposite` | ✅ v1.3 |
| 4 | [`environment-atmosphere-design.md`](./environment-atmosphere-design.md) | Rayleigh/Mie/ozone LUT chain, as one more source | v2.0 |
| 5 | [`environment-aerial-design.md`](./environment-aerial-design.md) | froxel aerial perspective + media unification | v2.0 |

**The finding that ordered them:** the visible sky is a **128²-per-face** GGX-convolved cube
(`EnvironmentSystem.ts:136`, `envShaders.ts:603`) — **0.70° per texel**. The shipped `sunCos`
default draws a disc 6.8× the sun's true angular size because a real one would be sub-texel.
Nothing after Phase 1 is possible until the skybox stops sampling the bake.

```
Environment
│
├── Sky ───────────────────────────────────────────────────────── ◐ P1 ✅ 1.1
│   ├── Skybox pass
│   │   ├── Fullscreen triangle, no geometry                      ✅ v1.2  the draw(3,n) path
│   │   ├── Ray reconstruction from invViewProj                   ✅ v1.2
│   │   ├── depthCompare 'less-equal', no depth write             ✅ v1.2  drawn after opaque
│   │   ├── Params in the batch storage buffer                    ✅ v1.2  exposure / yaw / blur mip
│   │   └── Per-pixel sampleSky (not the prefiltered cube)        ✅ 1.1  DEFAULT; 0.266° disc measured back (T1)
│   ├── The evaluator seam ─────────────────────────── ✅ 1.1  sky §4
│   │   ├── ShaderSkyLayout (sibling of lighting/indirect)        ✅  one fn, bake + skybox consumers
│   │   ├── ONE sampleSky, injected twice                         ✅  byte-identical bodies (U2) + on-device (T2)
│   │   ├── skySunRadiance — the CPU-mirror seam                  ✅  drawn before Phase 2 needs it; U6
│   │   └── Zero material bindings                                ✅  group 2 is 0–13 spent already; U11
│   ├── Sources
│   │   ├── Equirect HDR import → cube                            ✅ v1.2
│   │   ├── Procedural gradient (zenith/horizon/ground + sun)     ✅ v1.2  works with NO asset
│   │   ├── Both moved behind the seam, numerically inert         ✅ 1.1  U1 pins it to 1e-6
│   │   ├── Analytic daylight (Preetham)                          ✅ 1.1  30 constants; Hosek is data, not design
│   │   ├── SkyComposite (day ↔ night ↔ weather)                  ✅ v1.3  night §6 — the wrapLighting pattern
│   │   └── Scene capture (reuses probe capture)                  ◻ v2.0
│   ├── Physical atmosphere ──────────────────────────── ✍ v2.0  atmosphere §5
│   │   ├── Rayleigh + Mie transmittance LUT (256×64)             ✍ v2.0  baked once
│   │   ├── Multi-scatter LUT (32×32)                             ✍ v2.0  the SH-projection kernel shape
│   │   ├── Sky-view LUT (192×108, per frame)                     ✍ v2.0  100× fewer evals than per-pixel
│   │   ├── Ozone (the reason twilight is blue, not grey)         ✍ v2.0  absorption only; deletes the §3 ramp
│   │   ├── Altitude support (falls out, not implemented)         ✍ v2.0
│   │   └── ~298 KB total                                         ✍ v2.0
│   └── Night sky ───────────────────────────────────── ✅ v1.3  night §4–5
│       ├── Stars — instanced draw(3,n), ONE call                 ✅  ~9,100-record capacity, caller data adopted
│       ├── Flux conservation, not radiance                       ✅  fixed-pixel radial sprite
│       ├── Scintillation from Phase 2's air mass                 ✅  one shared constant, not two
│       ├── Moon in sampleSky → free in the IBL bake              ✅  resolved (0.52° ≈ 3 source texels)
│       ├── Lommel-Seeliger limb + earthshine + opposition surge  ✅  a Lambert moon is visibly wrong
│       ├── Milky Way (baked band, celestial coords)              ✅
│       ├── Twilight ramp                                         ✅  dayFactor blend; authored, replaced by Phase 4
│       └── Purkinje shift (named artistic choice, on)            ✅  moonlight is not physically blue
│
├── IBL bake ──────────────────────────────────────── ✅ v1.2  (owned here, consumed by §7)
│   ├── Source cube + mip chain                                   ✅
│   ├── GGX prefilter per mip                                     ✅
│   ├── BRDF LUT                                                  ✅
│   ├── SH-L2 projection                                          ✅
│   └── Amortized refresh for a moving sun                        ✅
│
├── Time of day ───────────────────────────────────────────────── ✅ v1.3  timeofday
│   ├── The four-copy problem it exists to end                    ✅  sky/light/shadow/IBL each hold a sunDir
│   ├── Zero GPU resources — a pure producer                      ✅  no buffer, no stage, no bind group
│   ├── Clocks
│   │   ├── 'artistic' — a 0–24 dial, no geography                ✅  the default
│   │   └── 'astronomical' — NOAA solar (~0.01°), Meeus lunar     ✅
│   ├── Sun / moon orbit → direction + colour                     ✅  writes the existing light record
│   ├── Sky ↔ IBL coupling (re-bake on change)                    ✅ v1.2
│   ├── Re-bake gate on ANGULAR motion, not frames                ✅  sun + night-weighted moon motion
│   ├── Day/night shadow-slot handoff (one holder, ever)          ✅  slot + direction + invalidate, one call
│   ├── Exposure coupling                                         ✅  AutoExposure clamp or Tonemap absolute
│   └── GMST + dayFactor — consumed by Phase 3                    ✅  full celestial-state handoff
│
├── Fog & aerial perspective ─────────────────────────────────── ◐ v2.0  aerial; light injection deferred
│   ├── Exponential height fog (HeightFog, closed form)           ✅  fragment-side, cheap, ships alone
│   ├── Aerial perspective (froxel 32³, 256 KB)                   ✅  color*T + inScatter, one shared body
│   │   ├── Fullscreen apply over opaque                          ✅  1×/pixel, toggleable, depth-aware
│   │   ├── Injected layout for TRANSPARENTS only                 ✅  they have no single depth; U8 refuses both
│   │   └── samplable depth texture                               ✅  init({ samplableDepth: true }), opt-in
│   ├── Media unification (fog IS a row, not a system)            ✅  the two paths refuse each other
│   └── Froxel volumetric fog
│       ├── Froxel grid (the clustered shape again)               ✅  built by aerial perspective
│       ├── Light scattering injection                            ◻ v2.0
│       ├── Shadowed volumetrics                                  ◻ v2.0  one binding + a loop; atlas exists
│       └── Temporal reprojection                                 ◻ v2.0  TransientPool history: 2 exists
│
├── Clouds ────────────────────────────────────────────────────── ◻
│   ├── 2D scrolling layer                                        ◻ v1.3
│   ├── Volumetric raymarch                                       ◻ v2.0
│   └── Cloud shadows on the ground                               ◻ v2.0
│
├── Weather ───────────────────────────────────────────────────── ◻ v1.3
│   ├── State blending (clear ↔ overcast ↔ storm)                 ◻  drives sky + fog + wind together
│   ├── Precipitation state (drives §9 emitters)                  ◻
│   ├── Lightning / thunder flashes                               ◻ v2.0
│   └── Wetness / snow-cover material response                    ◻ v2.0
│
├── Wind ──────────────────────────────────────────────────────── ◻ v1.3
│   ├── Global wind field (direction, strength, gust)             ◻  one uniform, many readers
│   └── Foliage response (vertex-side, instanced)                 ◻
│
├── Water ─────────────────────────────────────────────────────── ◻
│   ├── Flat plane
│   │   ├── Planar reflection (a second camera view)              ◻ v1.3  reuses per-view rendering
│   │   ├── Refraction + depth absorption                         ◻ v1.3
│   │   └── Normal-map ripples                                    ◻ v1.3
│   ├── Gerstner waves                                            ◻ v2.0
│   ├── FFT ocean displacement                                    ◻ v2.0
│   ├── Shoreline foam                                            ◻ v2.0
│   └── Buoyancy sampling → physics                               ◻ v2.0 🔌
│
└── Terrain ───────────────────────────────────────────────────── ◻ v2.0
    ├── Heightmap + clipmap LOD                                   ◻ v2.0
    ├── Splat / triplanar texturing                               ◻ v2.0
    ├── Vegetation scatter (instanced, GPU-culled)                ◻ v2.0
    └── Heightfield shared with physics                           ◻ v2.0 🔌
```

**Next up:** [sky Phase 2](./environment-timeofday-design.md) — the clock.

[Phase 1](./environment-sky-design.md) shipped in 1.1 and unblocked the rest of this section. It
landed in the order its §11 specifies: step 1 moved the existing gradient behind the seam with a
test proving the image byte-identical, and *nothing improved* at that step — which was the point,
since a refactor is provably inert before it is asked to carry anything. The skybox is no longer a
128² lookup, so every other branch here now has somewhere to land.

---

## 9. Effects

**Package:** `null-graph/effects` (planned) · **Status:** ◻ ·
**Brief:** [`roadmap-modules.md`](./roadmap-modules.md) §8

```
Effects
│
├── GPU particles ─────────────────────────────────────────────── ◻ v1.2  the canonical DOD case
│   ├── Simulation
│   │   ├── Compute-integrated flat buffer                        ◻ v1.2
│   │   ├── Free-list pooling                                     ◻ v1.2
│   │   ├── Forces (gravity, drag, curl noise, wind)              ◻ v1.2  wind comes from §8
│   │   └── Depth-buffer collision (screen-space)                 ◻ v1.2  cheap, no physics dependency
│   ├── Emission
│   │   ├── Emitter shapes (point, sphere, box, mesh)             ◻ v1.2
│   │   ├── Burst + continuous rates                              ◻ v1.2
│   │   └── GPU-side spawn (indirect count)                       ◻ v1.2
│   ├── Rendering
│   │   ├── Indirect draw of the live count                       ◻ v1.2  emitter writes instanceCount
│   │   ├── Billboards / stretched / mesh particles               ◻ v1.2
│   │   ├── Lit particles (clustered light lookup)                ◻ v1.3
│   │   └── Sorting for blended particles                         ◻ v1.3  needs §4's ordering model
│   └── Shared buffer with the physics solver                     ◻ 🔌   design the record once
│
├── Decals ────────────────────────────────────────────────────── ◻ v1.2
│   ├── Clustered decals (reuse the froxel grid)                  ◻ v1.2
│   ├── Deferred decals (g-buffer blend)                          ◻ v1.2  needs v1.1 deferred
│   └── Projected normal blending                                 ◻ v1.2
│
├── Trails / ribbons ──────────────────────────────────────────── ◻ v1.3
│
└── Screen-space ──────────────────────────────────────────────── ◐
    ├── SSR                                                       ◐   partial via Post; no dedicated effect
    ├── SSGI                                                      ◻ v2.0
    └── Screen-space shadows (contact)                            ◻ v2.0
```

**Next up:** the particle record layout — §10's XPBD solver wants the same buffer, so it is worth
designing once with room for velocity and inverse mass.

---

## 10. Physics

**Package:** `null-graph/physics` · **Status:** ◻ 🔌 plugin — ships on its own cadence ·
**Specs:** [`physics/`](./physics/) — [P1](./physics/physics-p1-transform-bridge-design.md) ·
[P2](./physics/physics-p2-broadphase-design.md) · [P3](./physics/physics-p3-solver-design.md) ·
**Brief:** [`roadmap-modules.md`](./roadmap-modules.md) §11

The premise: NullGraph reads transforms out of a flat buffer it does not own, so a physics engine is
a **producer of transforms** and attaches at the same seam a worker ECS does. **P1 needs no engine
code at all**, and is a legitimate stopping point for most applications.

P1–P3 are ✍ **designed** — three specs in [`docs/physics/`](./physics/), each a stopping point.
Everything below them in this tree (queries, character control, integration points, debug draw) is
◻ **deferred to its own later document**, listed with its reason in
[`physics/README.md`](./physics/README.md).

```
Physics 🔌
│
├── P1 · Transform bridge ─────────────────────────── ✍ 🔌  ZERO engine change; P1 spec
│   ├── Instance buffer binding
│   │   ├── Model matrix write (floats 0–15)                      ✍ 🔌  the renderer adopts, never copies
│   │   ├── Bounding sphere sync (floats 16–19)                   ✍ 🔌  desync shows as CULLING artifacts
│   │   └── Caller-owned backing buffer (one optional ctor param) ✍ 🔌
│   │       ├── Transferable ArrayBuffer (no headers)             ✍ 🔌  works everywhere
│   │       └── SharedArrayBuffer                                 ✍ 🔌  OPTIONAL — needs COOP/COEP
│   ├── Solver adapters
│   │   ├── Rapier (WASM, worker)                                 ✍ 🔌  third-party stays third-party
│   │   ├── Jolt (WASM, worker)                                   ✍ 🔌
│   │   └── Custom / user solver                                  ✍ 🔌  the adapter is the whole API
│   ├── Stepping
│   │   ├── Fixed timestep + accumulator                          ✍ 🔌
│   │   ├── Render-step interpolation                             ✍ 🔌  without it, everything judders
│   │   └── Substepping                                           ✍ 🔌
│   └── Lifecycle (spawn / despawn / sleep sync)                   ✍ 🔌
│
├── P2 · GPU broadphase ───────────────────────────── ✍ 🔌  P2 spec
│   ├── Spatial binning
│   │   ├── Uniform grid (the light binner's kernel shape)        ✍ 🔌  atomics + preClearBuffers
│   │   ├── Per-cell offset + count                               ✍ 🔌
│   │   └── Compacted index list                                  ✍ 🔌
│   ├── Sweep-and-prune on sorted keys                            ◻ 🔌  P2 §12 — needs a GPU sort
│   ├── Pair list → compacted buffer                              ✍ 🔌  the visibility compaction pattern
│   ├── Midphase from meshlet bounds                              ◻ 🔌  P2 §8 — filed with its cost
│   └── Benefits from indirect dispatch                           ✅  the primitive ships; P2 uses it
│
├── P3 · GPU solver (XPBD) ────────────────────────── ✍ 🔌  P3 spec; particles + cloth only
│   ├── Position-based integration                                ✍ 🔌  compute over a flat buffer
│   ├── Constraint projection
│   │   ├── Jacobi iteration (baseline)                           ✍ 🔌  no atomic<f32> in WGSL
│   │   ├── Graph-coloured batches                                ◻ 🔌  P3 §6.2 — optimization over Jacobi
│   │   └── Fixed-point atomics on u32                            ◻ 🔌  P3 §6.3 — filed, NOT recommended
│   ├── Bodies
│   │   ├── Particles / sand / debris                             ✍ 🔌  shares the §9 particle buffer
│   │   ├── Cloth (distance, bending deferred)                    ✍ 🔌  P3 §7
│   │   ├── Soft bodies (shape matching)                          ◻ 🔌  P3 §12
│   │   └── Rigid contacts                                        ◻ 🔌  P6 — count over fidelity; P3 §7
│   └── Collision shapes (sphere, capsule, box, hull, heightfield) ◻ 🔌  P3 is positions only; P3 §7
│
├── Character controller ──────────────────────────── ◻ 🔌  CPU by design
│   ├── Capsule sweep against a local subset                      ◻ 🔌  must be immediate
│   ├── Slope / step handling                                     ◻ 🔌
│   └── Ground detection                                          ◻ 🔌
│
├── Queries ───────────────────────────────────────── ◻ 🔌  async by construction
│   ├── Request buffer → compute resolve                          ◻ 🔌
│   ├── Results read ONE FRAME LATE                               ◻ 🔌  readback is never a stall
│   ├── Raycast / spherecast                                      ◻ 🔌
│   └── Overlap / proximity                                       ◻ 🔌
│
├── Integration points ────────────────────────────── ◻ 🔌
│   ├── Terrain heightfield (from §8)                             ◻ 🔌
│   ├── Water buoyancy sampling (from §8)                         ◻ 🔌
│   └── Skinned-mesh colliders (from §11)                         ◻ 🔌
│
└── Debug draw ────────────────────────────────────── ◻ 🔌  build this FIRST
    ├── Collider wireframes                                       ◻ 🔌
    ├── Contact points + normals                                  ◻ 🔌
    ├── Broadphase cells                                          ◻ 🔌
    └── Sleeping islands                                          ◻ 🔌
```

**Engine cost, by phase**

| Phase | Needs from the engine |
|---|---|
| P1 transform bridge | **Nothing required.** One optional additive change: `InstanceBuffer` accepting an external (Shared)ArrayBuffer. |
| P2 GPU broadphase | Nothing. Compute stages, atomics, `preClearBuffers` all exist. Benefits from indirect dispatch. |
| P3 GPU solver | Nothing. Blocked on nothing. |
| Queries | Nothing. Async by construction. |

**Next up:** P1, and specifically the debug draw — a physics bug you cannot see is a physics bug you
cannot fix.

---

## 11. Animation

**Package:** `null-graph/loaders` · **Status:** ◐ parsed + vertex skinning ·
**Brief:** [`roadmap-modules.md`](./roadmap-modules.md) §9

```
Animation
│
├── Parsing ───────────────────────────────────────────────────── ✅
│   ├── GLB skins + joints + inverse bind matrices                ✅
│   ├── Animation clips + channels                                ✅
│   └── Skeleton hierarchy                                        ✅
│
├── Sampling ──────────────────────────────────────────────────── ◐
│   ├── Keyframe interpolation (step / linear / cubic)            ✅  CPU (`Animator`)
│   ├── Clip blending / crossfade                                 ◻ v1.3
│   └── Worker-side sampling                                      ◻ v1.3
│
├── Deformation ───────────────────────────────────────────────── ◐
│   ├── Vertex-shader skinning (≤70 bones)                        ✅  `useSkinning`
│   ├── Compute skinning → mega-vertex output                     ◻ v1.3  feeds the SAME visibility path
│   │   └── Dissolves the group-2 skinning/lighting collision     ◻ v1.3
│   ├── Morph targets / shape keys                                ◻ v1.3
│   └── Skinned bounds refit                                      ◻ v1.3  loosen, never tighten
│
└── Behaviour ─────────────────────────────────────────────────── ◻ v1.4  arguably app-side
    ├── Blend trees                                               ◻ v1.4
    ├── State machines                                            ◻ v1.4
    ├── IK (two-bone, look-at)                                    ◻ v1.4
    └── Root motion                                               ◻ v1.4
```

**Next up:** compute skinning (v1.3) — the output is layout-identical to the input, so there is no
second draw path, and it frees a bind-group slot as a side effect.

---

## 12. Post

**Package:** `null-graph/post` · **Status:** ✅ shipped ·
**Specs:** [`post-processing-design.md`](./post-processing-design.md) ·
[`transient-pool.md`](./transient-pool.md) ·
**Brief:** [`roadmap-modules.md`](./roadmap-modules.md) §10

```
Post
│
├── Infrastructure ────────────────────────────────────────────── ✅
│   ├── PostChain (explicit, pre-allocated)                       ✅
│   ├── TransientPool
│   │   ├── Lifetime analysis → target aliasing                   ✅
│   │   ├── History targets (`history: 2`)                        ✅  reserved for TAA
│   │   └── Over-aliasing refusal (loud)                          ✅
│   ├── ParamArena (zero per-frame allocation)                    ✅
│   ├── ShaderComposer                                            ✅
│   └── Chain ordering + presets                                  ✅
│
├── Effects (16) ──────────────────────────────────────────────── ✅
│   ├── Tone & colour
│   │   ├── Tonemap · AutoExposure · ColorGrade                   ✅
│   │   └── Halation · FilmGrain · Vignette                       ✅
│   ├── Lens
│   │   ├── Bloom (13-tap down / 9-tap tent up)                   ✅
│   │   ├── Bokeh · LensFlare · ChromaticAberration               ✅
│   ├── AA & sharpening
│   │   └── FXAA                                                  ✅
│   ├── Stylized
│   │   ├── Kuwahara · CelShade · Outline                         ✅
│   ├── SSAO                                                      ✅
│   └── CustomEffect (bring your own WGSL)                        ✅
│
├── Temporal ──────────────────────────────────────────────────── ◻ v2.0
│   ├── TAA + history reprojection                                ◻ v2.0  pool support already exists
│   ├── Camera jitter                                             ◻ v2.0
│   ├── Motion blur (velocity buffer)                             ◻ v2.0
│   └── Temporal upscaling                                        ◻ v2.0
│
├── Depth of field (true CoC from linear depth)                   ◐   Bokeh ships without real CoC
└── Spatial upscaling                                             ◻ v2.0
```

**Next up:** nothing until §4 produces a velocity buffer — TAA and motion blur both wait on it.

---

## 13. Scene & Streaming

**Package:** — · **Status:** ◻ · **Brief:** [`roadmap-modules.md`](./roadmap-modules.md) §12

```
Scene & Streaming
│
├── Asset pipeline ────────────────────────────────────────────── ◻ v1.3
│   ├── Offline meshletization                                    ◻ v1.3  build-time → pre-build-time
│   ├── Offline quantization + packing                            ◻ v1.3
│   └── Bundle format + manifest                                  ◻ v1.3
│
├── Residency ─────────────────────────────────────────────────── ◻ v2.0
│   ├── Chunk / region streaming                                  ◻ v2.0  a policy over buffer ranges
│   ├── Instance range management                                 ◻ v2.0
│   ├── Mega-pool suballocation + defrag                          ◻ v2.0
│   └── Cluster streaming                                         ◻ v2.0  shares the problem with §2
│
└── Persistent GPU scene                                          ◻ v2.0  no re-upload per frame
```

**Next up:** the asset pipeline (v1.3) — meshletization is already build-time; this makes it
pre-build-time, which matters as soon as a scene has more than a handful of meshes.

---

## 14. Loaders & Assets

**Package:** `null-graph/loaders` · **Status:** ✅ shipped

```
Loaders & Assets
│
├── glTF / GLB ────────────────────────────────────────────────── ✅
│   ├── Binary chunk parsing + resource unpacking                 ✅
│   ├── Meshes / materials / textures                             ✅
│   ├── Skins + animation clips                                   ✅
│   └── KHR_materials_* extensions                                ◻ v1.3
│
├── Textures ──────────────────────────────────────────────────── ✅
│   ├── sRGB-aware decode                                         ✅
│   ├── Cache + solid-colour fallbacks                            ✅
│   ├── Mip generation                                            ◻ v1.3
│   └── KTX2 / Basis (compressed)                                 ◻ v1.3  large VRAM + load-time win
│
└── Mesh compression ──────────────────────────────────────────── ◻ v1.3
    ├── Draco decode                                              ◻ v1.3
    └── meshopt decode                                            ◻ v1.3
```

**Next up:** KTX2/Basis (v1.3) — the single biggest VRAM and load-time win available.

---

## 15. Tooling & Debug

**Packages:** `null-graph/profiler`, `null-graph/debug-ui` · **Status:** ✅ shipped

```
Tooling & Debug
│
├── Profiling ─────────────────────────────────────────────────── ✅
│   ├── Timestamp-query GPU timings                               ✅
│   ├── Performance widgets (full + minimal)                      ✅
│   └── Per-pass attribution                                      ◻ v1.3
│
├── Inspection ────────────────────────────────────────────────── ◻ v1.3
│   ├── Frame-graph visualizer                                    ◻ v1.3  the render sequence IS the graph
│   ├── Shader dump / WGSL inspection                             ◻ v1.3
│   ├── Buffer inspector                                          ◻ v1.3
│   └── GPU capture helpers                                       ◻ v1.3
│
└── Validation ────────────────────────────────────────────────── ◐
    ├── Headless WGSL parse of every generated module             ✅  in the test suites
    ├── CPU mirrors of GPU math (cullMath, shadowMath)            ✅
    ├── GPU integration harnesses                                 ◐   run headless (Node+Dawn) + Chrome; 5 suites, GI/atmosphere/post not yet
    └── CI gate                                                   ◻ v1.3  not wired
```

**Next up:** the frame-graph visualizer (v1.3) — the render sequence is already a linear graph, so
this is a reader, not a new abstraction.

---

## 16. Platform

**Package:** (core) · **Status:** ✅ shipped

```
Platform
│
├── Threading ─────────────────────────────────────────────────── ✅
│   ├── Web Worker rendering (OffscreenCanvas)                    ✅  no special headers needed
│   ├── Worker-fillable buffer contracts                          ✅  lights, instances, params
│   └── Transport for those buffers — the CALLER's choice
│       ├── Same-thread (plain ArrayBuffer)                       ✅  the default; nothing special
│       ├── Transferable ArrayBuffer via postMessage              ✅  no headers; ownership moves
│       └── SharedArrayBuffer                                     ◻   OPTIONAL — needs COOP/COEP
│           └── dev-server serves the headers (`--no-sab` off)    ✅  for testing that path
│
└── Device features ───────────────────────────────────────────── ◐
    ├── Subgroups                                                 ✅  clustered light binning
    ├── shader-f16                                                ◐   detected, not yet used
    ├── bgra8unorm-storage                                        ◐   detected, not yet used
    ├── dual-source-blending                                      ◐   detected, not yet used
    ├── multi-draw-indirect                                       ◻ v1.1  detect-only; not core WebGPU
    └── timestamp-query                                           ✅  the profiler
```

---

## 17. RayTracing

**Status:** ⛔ blocked on the platform · **Brief:** [`roadmap-modules.md`](./roadmap-modules.md) §13

```
RayTracing
├── BLAS / TLAS acceleration structures                           ⛔  no WebGPU API exists
├── Ray queries in WGSL                                           ⛔
├── RT shadows                                                    ⛔  → §6 shadow maps meanwhile
├── RT reflections                                                ⛔  → §7 probes + SSR meanwhile
└── RT global illumination                                        ⛔  → §7 probe volumes meanwhile
```

**The rule:** detect-and-adopt. Do **not** stub BLAS/TLAS types now — they would encode guesses
about an API that does not exist. The screen-space and probe-based approximations are legitimate
techniques in their own right, not placeholders waiting to be swapped out.

---

## 18. Picking

**Package:** `null-graph/picking` · **Status:** ✅ Cycle B · **Spec:** [`picking-design.md`](./picking-design.md)

```
Picking
├── Pick camera (projection zoomed onto (2R+1)² pixels)          ✅  pure maths, pinned headless
├── Instance sources (VisibilitySystem, culled by its own kernel) ✅  cull stage rides the chain disabled
├── Plain-batch sources (stock 14-float layout, custom WGSL)      ✅  shares the source's storage
├── Hit: source, app instance index, depth, world position, radius ✅  ~0.5 ms, off the frame path
├── Meshlet tier                                                  ◻  needs {instanceId, meshletId}
├── Skinned meshes                                                ◻  pick shader needs the skeleton
└── Hover / rect selection, GPU-side highlight without readback   ◻
```

The one touchpoint: compacted instance records carry `sourceIndex` (§3 visibility) — also what lets
any custom instance-tier shader find its own per-instance data.

## 19. Text

**Package:** `null-graph/text` · **Status:** ✅ Cycle B · **Spec:** [`text-labels-design.md`](./text-labels-design.md)

```
Text
├── Layout (per codepoint, kerning, align, baseline, wrap)        ✅  pure, Worker-safe
├── Atlases: msdf-atlas-gen JSON · runtime Canvas SDF · raw       ✅  MSDF / MTSDF / SDF
├── GPU cull → one drawIndexedIndirect                            ✅  600k glyphs, one draw
├── Screen-space, pixel-snapped, halo, depth test at the anchor   ✅
├── Follow any GPU buffer (instances, simulation output)          ✅  zero CPU work per frame
├── Decluttering (label collision)                                ◻  a compute pass over the same records
├── Shaping (Arabic, Devanagari…), bidi                           ◻  non-goal for now
├── Picking labels, world-size labels, rich text                  ◻
└── Atlas generation in Node                                      ◻
```

---

## By release

The same data inverted — useful for "what is actually next".

### v1.1 — Deferred & the rest of shadows

Visibility buffer · deferred shading · g-buffer layout (incl. velocity) · multi-draw fast path
(detect-only) · vertex quantization + index compression · small-triangle rejection · indirect
dispatch · material g-buffer output mode

### v1.2 — Light, really

IBL (env cube, GGX prefilter, BRDF LUT, SH-L2) · irradiance volumes + probe capture + multi-bounce ·
local reflection probes · skybox + procedural sky · exponential height fog · GPU particles ·
clustered + deferred decals · transparency ordering (OIT preferred) · inline-tonemap opt-out

### v1.3 — Authoring, quality, and the world

Compute skinning + morphs + bounds refit · offline asset pipeline · KTX2/Basis, Draco, meshopt ·
**sky Phases 2–3** ([Phase 1](./environment-sky-design.md) — seam + per-pixel skybox + Preetham —
shipped early, in 1.1;
[time of day](./environment-timeofday-design.md) + the re-bake gate + shadow handoff,
[night sky](./environment-night-design.md) + `SkyComposite`) · weather blending · wind field +
foliage · flat water · 2D clouds · PCSS + filtered-moment shadows · area lights (LTC) +
cookies/IES · tangent generation · parallax + clear coat + sheen + anisotropy · frame-graph
visualizer + shader dump · MSAA · reverse-Z · device-loss handling · pipeline warmup · explicit
bind-group layouts

### v1.4 — Motion & behaviour

Blend trees · state machines · IK · root motion. Revisited once v1.3 lands — arguably application
territory, listed so the boundary is an explicit decision rather than an omission.

### v2.0 — Virtual & temporal

Virtual geometry (DAG cluster-LOD + streaming) · virtual shadow maps · static/dynamic split maps ·
DDGI per-probe visibility · streamed probe volumes · TAA + motion blur + temporal upscaling ·
**sky Phases 4–5** ([physical atmosphere](./environment-atmosphere-design.md),
[aerial perspective](./environment-aerial-design.md) + media unification — and with them the froxel
grid that volumetric fog needs) · light injection + shadowed volumetrics · volumetric clouds · FFT
ocean · terrain clipmaps · triplanar · transmission/refraction · SSGI · screen-space shadows ·
persistent GPU scene + chunk residency · render bundles

**Sequencing note.** Sky Phase 5 was the one item in this list gated on a *core* change — a samplable
depth texture. It landed as opt-in `init({ samplableDepth: true })`, off by default because
`TEXTURE_BINDING` can cost depth-compression fast paths every frame. `HiZPyramid` and the depth
pre-pass (§4) want the same flag and can now adopt it without a further core change.

### 🔌 Physics plugin — any time

P1 (transform bridge) depends on nothing but the zero-copy instance contract. P2 (GPU broadphase)
and P3 (XPBD solver) follow, and still need no core change. All three are ✍ specced in
[`docs/physics/`](./physics/); everything past P3 is deferred to its own later document, listed with
its reason in [`physics/README.md`](./physics/README.md).

P1 is the one worth reading even if physics is not on your roadmap: it documents what the instance
buffer's zero-copy contract actually guarantees, and it corrects a live assumption — bounds are
**object-space** and the cull scales them itself (`visibilityShaders.ts:113-119`), so a rigid body's
bounds are a spawn-time write, not per-frame sync.

---

## Where the `◐` are

Every partial entry, with what is actually missing:

| Entry | § | Works | Missing |
|---|---|---|---|
| LOD simplification | 2 | LOD chains + error | no built-in simplifier; caller supplies levels |
| Vertex colours | 2 | the layout carries them | no stock material reads them |
| Derivative TBN | 2, 5 | robust normal mapping | wrong on mirrored UVs; no authored tangents |
| Distance culling | 3 | distance drives LOD | no distance *rejection* |
| Depth pre-pass | 4 | callers can assemble one | no first-class helper; Hi-Z needs it |
| Transparency | 4 | blend states | **no ordering model at all** |
| Inline tonemap | 5 | tonemap + gamma in-material | double-applies with PostChain's Tonemap |
| Shadow filtering | 6 | PCF + tap clamp | PCSS, filtered moments |
| Exposure coupling | 8 | AutoExposure in Post | not coupled to sky / time-of-day — ✍ [timeofday §8](./environment-timeofday-design.md) |
| The sky | 8 | skybox pass, three sources behind one evaluator, the IBL bake | ✅ resolved in 1.1: the skybox evaluates `sampleSky` per pixel ([sky §1](./environment-sky-design.md)) |
| SSR | 9 | screen-space effects exist | no dedicated SSR effect |
| Animation sampling | 11 | CPU keyframe interpolation | no blending, no worker path |
| Depth of field | 12 | Bokeh effect | no true CoC from linear depth |
| GPU harnesses | 15 | shadows, sky, visibility, core, ONNX seam — green on Node+Dawn and Chrome (`npm run test:gpu`, `test:gpu:browser`) | probes/GI, atmosphere, aerial, post, shadow atlas have no GPU suite yet |
| f16 / bgra8-storage / dual-source | 16 | detected and surfaced | nothing consumes them |

---

## Not planned

Stated so their absence reads as a decision rather than an oversight.

| Thing | Why not |
|---|---|
| Retained scene graph | Constraint #1. The whole engine is the absence of this. |
| Baked lightmaps | Probe volumes (§7) cover static GI without a UV-unwrap pipeline. |
| Node-graph VFX / material editor | Application territory; the shader builders are the API. |
| A physics solver of our own | §10 is an *adapter*. Rapier and Jolt are better than anything we would write. |
| WebGL fallback | Constraint #4 — no polyfills, and the design assumes compute throughout. |
| Ray tracing | ⛔ the platform has no API to detect, let alone adopt. |
