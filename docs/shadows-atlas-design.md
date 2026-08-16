# NullGraph Shadows — Phase 2: Atlas, Punctual Lights & GPU Caster Culling — Technical Design

**Target:** `src/extras/shadows/` (`ShadowAtlas`, `atlasMath`, punctual views, tile caching)
**Package:** `null-graph/shadows` (same subpath; Phase 2 is additive)
**Status:** `v1.2` design, implementation-ready.
**Prerequisite:** [`shadows-design.md`](./shadows-design.md) — Phase 1. Read it first; this document
generalizes its `ShadowView` and assumes its §3 integration facts.
**Depends on:** Phase 1, [`visibility-design.md`](./visibility-design.md) §5 (`HiZPyramid`),
[`api/lights.md`](./api/lights.md).
**Roadmap:** [`roadmap.md`](./roadmap.md) §3 · [`roadmap-modules.md`](./roadmap-modules.md) §4.

---

## 1. Purpose

Phase 1 shadows **one** light with a fixed budget of textures. Phase 2 turns that into a **system**:
every light type can cast, shadow memory is one fixed allocation the engine partitions by
importance, tiles that did not change are not re-rendered, and casters hidden behind other casters
are culled on the GPU.

The unifying move is small and it is the whole design: Phase 1's `ShadowView` — a `Camera` plus a
render target — becomes a `ShadowView` **plus an atlas rectangle**. A cascade, a spot cone, and one
of six cube faces then differ only in how their matrix is built and how big a tile they get. Every
downstream mechanism — culling, drawing, sampling, caching — is written once against `ShadowView`.

### What Phase 1 deliberately left open, and where it lands here

| Phase 1 gap | Phase 2 answer |
|---|---|
| One directional light only | Per-light shadow slot in the light record (§4.3), any mix of directional/spot/point |
| Fixed `depth24plus` array, one layer per cascade | One atlas texture, tiles allocated by importance (§4, §5) |
| Shadow light identified by direction match (§8.1) | Explicit slot index in `spot_params.z` (§4.3, §9.1) |
| Every cascade re-rendered every frame | Dirty tracking + round-robin refresh (§8) |
| Frustum-only caster culling | Hi-Z occlusion on the light view, reusing `HiZPyramid` unchanged (§7) |
| N render passes | One atlas pass, N viewport-scoped batches (§5, §6) |

### Non-goals for Phase 2

- **Virtual shadow maps.** Page residency and streaming are v2.0; they are to this document what
  virtual geometry is to `geometry-meshlets-design.md`, and they are only sane *after* a real atlas
  exists to virtualize.
- **Filtered-moment shadows** (VSM/ESM/MSM) and **PCSS**. The atlas makes them possible; picking one
  is a separate quality decision, and each brings its own artifact class (light bleeding, halos).
- **Ray-traced shadows.** ⛔ blocked on WebGPU.
- **Translucent / colored shadows.** Needs a second colored target and an ordering model.

### Design principles

These extend Phase 1's; the first is the one that does the work.

1. **One `ShadowView` type, three constructors.** `cascadeView()`, `spotView()`, `cubeFaceView()`
   all produce the same struct: a 108-float `Camera`, an atlas rect, a texel size, a dirty flag.
   Nothing after construction branches on light type except the ~10 lines of sampling that pick
   *which* view to read (§9.2).
2. **The caller owns dirtiness.** NullGraph has no scene graph, so it cannot know a caster moved.
   Tiles are re-rendered unless the caller declares otherwise — `invalidate()` is explicit,
   `cached` is opt-in, and the default is always-correct (§8.1).
3. **Budget is fixed and failure is loud.** The atlas is allocated once. When demand exceeds it,
   the system refuses at build time naming the light, rather than silently dropping a shadow
   (§11) — the `TransientPool` discipline.
4. **Conservative always beats clever** — which, pleasantly, is *free* for atlas-wide Hi-Z (§7.2).

---

## 2. Architecture

```
   engine.render([
      ...vis.getComputeStages(),        // main-view culls
      ...shadows.getComputeStages(),    // per-view caster culls + light-view Hi-Z build
      shadows.getAtlasPass(),           // ONE depth-only pass, one viewport-scoped batch per (view × mesh)
      scenePass,
      ...post.getPassSequence(),
   ]);
```

```
        ┌──────────────────────────────────────────────────────────────────────┐
        │  ShadowSystem (Phase 2)                                              │
        │  registerLight / setBudget / invalidate / update(camera) / wrapLighting │
        └──┬──────────────┬────────────────┬────────────────┬──────────────────┘
           │              │                │                │
    ┌──────▼──────┐ ┌─────▼───────┐ ┌──────▼───────┐ ┌──────▼─────────┐
    │ ShadowAtlas │ │ view builders│ │ caster culls │ │ tile table     │
    │ alloc/evict │ │ cascade/spot/│ │ (VisibilitySys│ │ (storage buf) │
    │ (§4,§5)     │ │ cube (§9.1)  │ │  per view)   │ │ → receivers   │
    └──────┬──────┘ └─────┬────────┘ └──────┬───────┘ └──────┬─────────┘
           │              │                 │                │
      one depth24plus  432 B Camera    drawArgs[1]      atlas rect + viewProj
      atlas texture    per view        per view×mesh    per view, read at shading
           │                                                  │
      ┌────▼──────────┐                              ┌────────▼─────────┐
      │ HiZPyramid    │  ← last frame's atlas        │ getIncidentLight │
      │ (unchanged)   │                              │ slot → view → tap│
      └───────────────┘                              └──────────────────┘
```

---

## 3. Integration facts (additional to Phase 1 §3)

| Fact | Source | Consequence |
|---|---|---|
| **`RenderPipeline` never calls `setViewport` or `setScissorRect`.** Every batch in a pass rasterizes over the full attachment. | `RenderPipeline.ts:104-146` | An atlas needs a per-batch viewport (§6.1). The zero-change alternative — clip-space tile remap — **bleeds** (§6.2), which is the single most important correctness fact in this document. |
| Multiple batches in one `RenderPassNode`, each with its own bind groups, are drawn in array order into the same attachments. | `RenderPipeline.ts:104-146`, `RenderPass.ts:28` | One atlas pass can hold every view's draws — clear once, no per-tile pass overhead. |
| `HiZPyramid.configure(depthSource, w, h)` takes **any** samplable depth view and rebuilds its stages. Nothing in it is main-view specific. | `HiZPyramid.ts:42-81` | Light-view occlusion culling reuses it verbatim against last frame's atlas (§7). |
| `Light.spot_params` is `vec4<f32>` at floats 12–15 of the 16-float light record; only `.x`/`.y` (inner/outer cone cosines) are read, for spot and torch. | `LightSystem.ts:23,123,312-313,351-352` | `.z` is free on **every** light type — the natural home for the shadow slot (§4.3), with no `LightSystem` change and no change to the 16-float upload contract. |
| The clustered path resolves a loop index to a real light via `lightIndices[cluster.start_offset + i]`, and both are module-scope bindings in the composed shader. | `LightSystem.ts:250-252,289-292` | The Phase 1 decorator can read the real light record, so a slot lookup costs one extra indexed load (§9.2). |
| `CullParams` is a 32-byte uniform created per cull stage; the shader reads `screenSize` for both the Hi-Z footprint and LOD. | `visibilityShaders.ts:72-82`, `VisibilitySystem.ts:456-477` | Atlas Hi-Z needs a tile rect here (§7.1). Growing the struct to 48 bytes touches one `createBuffer` size and one `writeBuffer`. |
| Core WebGPU guarantees only **8** storage buffers per shader stage (`maxStorageBuffersPerShaderStage`); the meshlet cull already binds 6. | WebGPU limits; `VisibilitySystem.ts:323-332` | A "cull once, scatter to N tiles" kernel cannot bind many tiles' outputs. One dispatch per view stays the baseline (§7.3). |

---

## 4. Data model

### 4.1 `ShadowView` — Phase 1's cascade, generalized

```wgsl
struct ShadowView {
    viewProj   : mat4x4<f32>,   // 64 B  world → this view's clip
    rect       : vec4<f32>,     // 16 B  atlas UV transform: .xy = uv scale, .zw = uv offset
    texelWorld : f32,           //       world size of one texel at the view's reference depth
    splitFar   : f32,           //       cascades: view-space depth covered; punctual: light range
    kind       : u32,           //       0 = cascade, 1 = spot, 2 = cube face
    _pad       : u32,           // 16 B
};                              // 96 B, 16-aligned
```

The CPU-side twin keeps the full 432-byte `Camera` buffer (Phase 1 §4.1) that the cull stage and
the depth draw bind, plus allocation bookkeeping (tile index, resolution, dirty, last-used frame).
`ShadowView` above is only what the *receiver* needs — it is a `storage` array, not a uniform, so
the count is not baked into the shader.

### 4.2 `ShadowSlot` — one per shadow-casting light

```wgsl
struct ShadowSlot {
    firstView  : u32,   // index into the views array
    viewCount  : u32,   // cascades: N; spot: 1; point: 6
    kind       : u32,   // 0 = directional, 1 = spot, 2 = point
    lightRange : f32,   // punctual only: for the depth remap at sample time
};
```

Directional lights own `cascadeCount` consecutive views, spots one, points exactly six in
`+X,-X,+Y,-Y,+Z,-Z` order so face selection is arithmetic (§9.2).

### 4.3 The slot index on the light — the thing that replaces Phase 1's direction match

`spot_params.z` (float 14 of the 16-float light record) carries `slotIndex + 1`; `0` means "this
light does not cast." Free on every light type, requires no `LightSystem` edit, and the caller sets
it in the same `Float32Array` a worker already fills (`LightSystem.setRawData`,
`LightSystem.ts:92`). `ShadowSystem.registerLight()` returns the slot so the caller can write it:

```ts
const slot = shadows.registerLight({ kind: 'point', position, range: 30, importance: 1.0 });
lightData[lightIndex * 16 + 14] = slot + 1;      // spot_params.z
```

Phase 1's `dot(dir, shadow.lightDir) > 0.999` heuristic is deleted the day this lands (§9.2), along
with its "two identical directional lights collide" caveat.

### 4.4 Resources — allocated once

| Resource | Format / usage | Size |
|---|---|---|
| atlas | `depth24plus`, `RENDER_ATTACHMENT \| TEXTURE_BINDING` | `atlasSize²` (default 4096² ≈ 32 MB) |
| sampling view | `createView()` — plain `texture_depth_2d` | — |
| comparison sampler | `createSampler({ compare:'less', magFilter:'linear', minFilter:'linear' })` | — |
| `views` | `STORAGE`, `array<ShadowView>` | `96 × maxViews` |
| `slots` | `STORAGE`, `array<ShadowSlot>` | `16 × maxSlots` |
| per-view camera | `UNIFORM`, 432 B each | `432 × maxViews` |
| light-view Hi-Z | `HiZPyramid` over the atlas, `r32float` | `atlasSize²` × 4/3 |

At 4096² with 64 views: 32 MB atlas + 6 KB of tables + 27 KB of cameras + 21 MB Hi-Z. Nothing is
allocated per frame; the per-frame writes are `maxViews × 432` bytes of camera data and one
`views`/`slots` refresh.

---

## 5. Atlas allocation

### 5.1 Quad-tree tiles, power-of-two only

The atlas is partitioned into power-of-two square tiles by a simple quad-tree free-list. Tile size
comes from a light's **importance**, evaluated on the CPU once per frame:

```ts
// Screen-space footprint heuristic: how many pixels does this light's influence cover?
function importance(view: ShadowViewRequest, camera: Camera): number {
    if (view.kind === 'cascade') return 1.0 / (view.cascadeIndex + 1);      // cascade 0 dominates
    const dist = distance(camera.eye, view.position);
    const px   = (view.range / Math.max(dist, camera.near)) * camera.screenH;
    return clamp(px / camera.screenH, 0, 1) * view.userImportance;
}
```

Importance maps to a tile size by a fixed ladder (`2048, 1024, 512, 256, 128`) with hysteresis — a
tile only changes size when importance crosses a threshold by more than `resizeHysteresis` (default
25%), because every resize is a full re-render and a visible quality step.

### 5.2 Eviction

Sort requested views by importance, assign top-down until the atlas is full, and evict the
least-important resident views. An evicted view's light samples as **fully lit** — never as fully
shadowed, which would be a black artifact where a subtle one belongs. When eviction happens at all,
it is reported through `report()` (the profiler surface) rather than silently, and exceeding
`maxViews` is a hard error (§11).

### 5.3 Stability rules

Two rules keep the atlas from thrashing, and both exist because the failure they prevent looks like
a rendering bug rather than an allocator bug:

1. A view keeps its tile across frames unless resized or evicted — its rect is stable, so a cached
   tile stays valid (§8).
2. Cascade tiles are allocated **first and permanently**, at fixed sizes. The sun is not allowed to
   lose its shadow because forty torches got close to the camera.

---

## 6. The atlas pass

### 6.1 One pass, viewport-scoped batches

```ts
const atlasPass = engine.createPass({
    name: 'Shadow Atlas',
    isMainScreenPass: false,
    depthStencilAttachment: {
        view: atlas.createView(),
        depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store',
    },
});
// per (view × caster mesh):
const batch = engine.createBatch(atlasPass, {
    shaderCode: buildShadowCasterWGSL({ ... }),   // Phase 1 §6.2, unchanged
    isIndirect: true,
    depthOnly: true,                    // Phase 1 §12.1
    cameraBuffer: view.cameraBuffer,    // Phase 1 §12.1
    viewport: view.tileRect,            // §12.1 below — [x, y, w, h] in atlas pixels
    ...
});
```

The viewport is what makes the atlas correct: clipping happens against the view's own clip volume
in NDC, and the viewport transform then maps that clipped `[-1,1]²` into the tile rect. Nothing can
land outside the tile, by construction.

### 6.2 Why the zero-change alternative is not the baseline

The tempting zero-engine-change trick is a **clip-space tile remap** in the caster vertex shader:

```wgsl
// scale = tileSize / atlasSize;  offset = (2*tileOrigin + tileSize)/atlasSize - 1  (y mirrored)
out.clipPos = vec4<f32>(clip.xy * scale + offset * clip.w, clip.z, clip.w);
```

The algebra is right — the light's frustum edges land exactly on the tile's edges. But **clipping
still happens against the atlas's `[-1,1]`, not the tile's sub-rectangle.** A caster that straddles
the light frustum boundary keeps the geometry outside that boundary, and after the remap that
geometry rasterizes into **neighbouring tiles**, writing depth into another light's shadow map. The
symptom is a phantom shadow in an unrelated light, appearing only when a caster is partly
off-frustum — which is exactly the case caster culling does *not* remove.

Two ways to make it safe without `setViewport`:

- Give the pipeline a fragment stage with `targets: []` that `discard`s outside the tile rect
  (legal, and depth is not written on discard). Correct, but it reinstates a fragment stage — the
  thing Phase 1 §12.1 removed — and pays it on every caster triangle.
- Enable `clip-distances` and emit a per-tile clip plane. **Not core WebGPU**, so it violates
  roadmap constraint #4 as a baseline.

Both are strictly worse than six lines in `RenderPipeline`. The atlas is the feature that justifies
the change; §12.1 files it.

### 6.3 Draw-call count

`views × casterMeshes` indirect draws, in one pass, independent of instance count. Four cascades,
eight shadowed punctual lights (one spot at 1 view, seven points at 6), six caster meshes:
`(4 + 1 + 42) × 6 = 282` draws for an unbounded number of instances. §8's caching then removes the
ones whose tiles did not change — typically most of them.

---

## 7. GPU caster culling at atlas scale

### 7.1 Per-view culling reuses Phase 1's mechanism

Each view is one `VisibilitySystem` cull stage with `camera: view.cameraBuffer`
(Phase 1 §9.3). Orthographic and perspective shadow frusta both extract correctly through the
existing `frustumPlanes` (Phase 1 §3), so nothing in `CULL_CORE_WGSL` changes.

### 7.2 Hi-Z on the light view — and why atlas-wide is *safe*

Occlusion culling for casters: a caster fully behind another caster, **from the light's viewpoint**,
contributes nothing and can be dropped. `HiZPyramid` is reused unmodified over last frame's atlas
(`HiZPyramid.configure(atlasView, atlasSize, atlasSize)`), with the same one-frame latency argument
as `visibility-design.md` §5.2 — and a weaker consequence here, since a one-frame-stale shadow is
less visible than one-frame-stale geometry.

The obvious objection is contamination: at coarse mips, a tile's footprint pools texels from
**neighbouring tiles**, which belong to unrelated lights. That turns out to be safe, and the reason
is worth stating precisely because it is the opposite of the usual atlas-filtering answer:

> The pyramid is a **max** reduction, and the test is `ndcNear > occ`. Contamination can only ever
> *raise* `occ`, and a higher `occ` makes the test *harder* to pass. Neighbour bleed therefore
> causes **under-culling**, never over-culling. Cleared atlas regions hold 1.0 (far), the maximum,
> so borders cull nothing at all.

Under-culling costs a few wasted draws. Over-culling would delete shadows. One pyramid over the
whole atlas is correct, and the per-tile pyramids that would be "cleaner" are not worth their cost.

The one required change is the UV remap, because the pyramid's texel space is the atlas's, not the
tile's. `CullParams` grows one field:

```wgsl
struct CullParams {
    instanceCount : u32,
    meshIndex     : u32,
    hizMaxLevel   : u32,
    flags         : u32,
    screenSize    : vec2<f32>,
    lodErrorPx    : f32,
    _pad          : f32,
    hizRect       : vec4<f32>,   // NEW: .xy = uv scale, .zw = uv offset. Main view = (1,1,0,0).
};                               // 32 B → 48 B
```

and `occludedByHiZ` applies it just before sampling:

```wgsl
let uvMinA = uvMin * params.hizRect.xy + params.hizRect.zw;
let uvMaxA = uvMax * params.hizRect.xy + params.hizRect.zw;
```

`screenSize` for a shadow view is the **tile** size, so footprint-to-mip selection stays correct.
With `hizRect = (1,1,0,0)` the expression is the identity and the main view is bit-for-bit
unchanged — which is what §10 U5 asserts.

### 7.3 Why not one dispatch that scatters to every tile

A single cull kernel writing all views' outputs would need `2 × views` storage bindings (each
view's compacted list plus its indirect args). Core WebGPU guarantees only **8** storage buffers per
stage and the existing meshlet cull already uses 6. So the baseline stays **one dispatch per view**,
each over `maxInstances` threads with idle lanes early-outing — the same trade
`visibility-design.md` §7.2 makes.

Two documented escapes, neither on the baseline path:

- **A shared coarse pass.** A point light's six faces share one bounding-sphere test against the
  light range. One dispatch can reject the ~90% of instances outside the light's radius into a
  compacted per-light list; the six face culls then run over that much shorter list. Strictly a
  win when instance counts are large, and it needs only 3 extra bindings.
- **Indirect dispatch** (roadmap §6, `visibility-design.md` §12.1). With
  `dispatchWorkgroupsIndirect`, the per-face culls dispatch over the *actual* per-light survivor
  count instead of `maxInstances`. This is the release where that optional engine change starts
  paying for itself — 48 face culls over full capacity is where the waste stops being theoretical.

---

## 8. Caching — not re-rendering what did not change

### 8.1 Dirtiness is declared, not inferred

NullGraph has no retained scene, so it cannot detect that a caster moved. A view is re-rendered when
**any** of these holds, and the default configuration makes every frame dirty — correct first,
fast when the caller opts in:

| Cause | Detected by |
|---|---|
| the view's matrix changed (light moved, cascade re-snapped) | `ShadowSystem`, by comparing the packed matrix |
| the tile was resized, evicted, or newly allocated | `ShadowAtlas` |
| the caller moved a caster | `shadows.invalidate(slot)` — explicit |
| `updateMode: 'every-frame'` (default) | always |

`updateMode: 'on-demand'` renders only dirty views. `updateMode: 'round-robin'` additionally
refreshes `refreshPerFrame` clean views per frame (default 2), which bounds the damage from a
missed `invalidate()` to a few frames instead of forever — the pragmatic setting for a world where
some casters are "mostly static."

### 8.2 Clearing one tile without clearing the atlas

Caching needs `depthLoadOp: 'load'` on the atlas, which means the pass no longer clears — so each
dirty tile must clear itself. That is one extra batch per dirty tile, drawn first, scoped to the
same viewport:

```ts
engine.createBatch(atlasPass, {
    shaderCode: TILE_CLEAR_WGSL,      // one full-viewport triangle at z = 1.0, no vertex buffer
    depthOnly: true,
    depthWriteEnabled: true,
    depthCompare: 'always',           // overwrite whatever the tile held
    viewport: view.tileRect,
    ...
});
```

Three vertices per dirty tile. No engine change beyond §12.1's viewport, and no second pass.

### 8.3 What caching actually buys

A scene with 4 cascades and 8 shadowed static point lights: frame 1 renders 52 views; steady state
renders 4 (the cascades follow the camera and are always dirty) plus 2 round-robin refreshes. The
caster draws drop from 282 to ~36. This is the difference between "shadows for one light" and
"shadows for the scene," and it is why the atlas comes before any filtering upgrade.

**The cull must be gated too, not just the draw.** A suppressed tile is suppressed by giving its
batches a zero-area viewport — which stops rasterization and nothing else. The caster *cull* is a
`ComputeStageNode` riding the `VisibilitySystem`'s chain, and it will happily keep dispatching over
the entire instance buffer for a tile that draws nothing. At 52 views × 6 caster meshes × 100 000
instances that is ~600 000 wasted workgroups per frame — more than the draws caching was introduced
to remove. So a clean tile also sets `stage.enabled = false` (§12.1) on every cull feeding it,
reached through `VisibilitySystem.getCullStage(batch)`.

The symmetry is worth stating as a rule: **whatever suppresses a tile's draw must suppress its cull
in the same place.** They are set together in `retargetViewSlots`, and §10 U12 fails if the cull
gate is removed while the viewport gate stays.

### 8.4 Per-slot draw state is built per *registered view*, not per `maxViews`

`maxViews` is a capacity bound, not a workload. Building each slot's batches up front for all of
them costs a render pipeline + a cull stage + a params buffer per (slot × caster mesh) — for a
64-view atlas with 6 caster meshes, 384 pipeline compilations at startup, most for view slots no
light will ever occupy.

Slots are therefore materialized up to the number of views the **registered lights actually own**,
from both `registerLight` and `registerCaster` so the two orders converge (a mesh registered later
is appended to every existing slot; a light registered later materializes new slots with every
mesh registered so far). A lights-only system with no casters builds nothing at all — not even a
tile-clear, which exists only to make room for casters. §10 U11 pins the counts.

---

## 9. Views and sampling

### 9.1 The three view builders

| Kind | Matrix | Views | Notes |
|---|---|---|---|
| **cascade** | `orthoZO` fitted to a split sphere | `cascadeCount` | Phase 1 §5, unchanged |
| **spot** | `perspectiveZO(2 × outerConeAngle, 1.0, near, range)` | 1 | fov comes straight from `spot_params.y = cos(outer)`: `fov = 2·acos(outerCos)`, so cone and shadow can never disagree |
| **point** | `perspectiveZO(π/2, 1.0, near, range)` × 6 | 6 | axis order `+X,−X,+Y,−Y,+Z,−Z`; a 90° fov exactly tiles the sphere |

All three use the `ZO` variants — the WebGPU `[0,1]` clip volume, the trap `Camera.ts:62-66`
documents and Phase 1 §10 U4 pins.

**The 90° fov needs a margin.** PCF taps near a cube face edge read past the frustum. Each face is
therefore built with a slightly widened fov, `2·atan(1 + 2·filterTexels/tileSize)`, so the tap
kernel always has real data inside the tile — and sampling still clamps (§9.3).

### 9.2 The receiver: slot → view → tap

The Phase 1 decorator (§8.1 there) keeps its shape; only the middle changes — the direction match
becomes a slot lookup, and the view choice becomes type-dependent:

```wgsl
fn getIncidentLight(i: u32, worldPos: vec3<f32>) -> IncidentLight {
    var inc = getIncidentLightUnshadowed(i, worldPos);
    let rec = ${accessor};                          // lights[i] | globalLights[lightIndices[...]]
    let packed = u32(rec.spot_params.z);            // 0 = no shadow, slot+1 otherwise
    if (packed == 0u) { return inc; }

    let slot = shadowSlots[packed - 1u];
    var v = slot.firstView;
    if (slot.kind == 0u) {                          // directional: pick the cascade
        v = v + selectCascade(worldPos, slot);
    } else if (slot.kind == 2u) {                   // point: pick the cube face
        v = v + cubeFace(worldPos - rec.pos_type.xyz);
    }                                               // spot: the single view
    inc.attenuation = inc.attenuation * sampleView(v, worldPos);
    return inc;
}

// Major-axis face select, matching the +X,-X,+Y,-Y,+Z,-Z build order.
fn cubeFace(d: vec3<f32>) -> u32 {
    let a = abs(d);
    if (a.x >= a.y && a.x >= a.z) { return select(1u, 0u, d.x > 0.0); }
    if (a.y >= a.z)               { return select(3u, 2u, d.y > 0.0); }
    return select(5u, 4u, d.z > 0.0);
}
```

### 9.3 `sampleView` — the atlas tap

```wgsl
fn sampleView(vi: u32, worldPos: vec3<f32>) -> f32 {
    let v    = shadowViews[vi];
    let clip = v.viewProj * vec4<f32>(worldPos, 1.0);
    if (clip.w <= 0.0) { return 1.0; }                              // behind a punctual view
    let ndc = clip.xyz / clip.w;
    if (ndc.z > 1.0) { return 1.0; }
    var uv = vec2<f32>(ndc.x, -ndc.y) * 0.5 + vec2<f32>(0.5);       // clip +Y up → texture +V down
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) { return 1.0; }

    let texel = shadow.filterTexels / f32(shadow.atlasSize);        // in ATLAS uv units
    let ref   = ndc.z - shadow.constantBias;
    let r     = i32(shadow.pcfRadius);

    var sum = 0.0;
    var n   = 0.0;
    for (var y = -r; y <= r; y = y + 1) {
        for (var x = -r; x <= r; x = x + 1) {
            // Offset in the VIEW's uv, then map into the atlas, then clamp INSIDE the tile so a
            // tap can never read a neighbouring light's depth (§9.4).
            let local  = clamp(uv + vec2<f32>(f32(x), f32(y)) * (texel / v.rect.x),
                               vec2<f32>(0.0), vec2<f32>(1.0));
            let atlasUV = local * v.rect.xy + v.rect.zw;
            sum = sum + textureSampleCompareLevel(shadowAtlas, shadowSampler, atlasUV, ref);
            n   = n + 1.0;
        }
    }
    return sum / n;
}
```

`textureSampleCompareLevel`, for the reason Phase 1 §8.3 gives — every call here is inside a
branch and a loop. The `clamp` before the rect transform is the atlas's defining sampling rule and
§10 U3 pins it.

### 9.4 The two atlas sampling hazards

- **Tile bleed on filtering.** A PCF tap near a tile edge would otherwise land in a neighbour's
  tile — a *lit* surface picking up an unrelated light's occluders. The clamp above prevents it;
  the visual cost is a shadow that stops filtering at the tile edge, which is invisible because the
  view frustum edge is already the shadow's boundary.
- **Cube face seams.** Two adjacent faces disagree by up to one texel across the shared edge,
  producing a hairline. The widened fov (§9.1) supplies the overlap that hides it. `filterTexels`
  and the fov margin must be raised together — §11 errors when they disagree.

### 9.5 Bindings

```wgsl
@group(3) @binding(0) var shadowAtlas   : texture_depth_2d;
@group(3) @binding(1) var shadowSampler : sampler_comparison;
@group(3) @binding(2) var<uniform> shadow : ShadowParams;          // atlasSize, pcfRadius, biases, cascade splits
@group(3) @binding(3) var<storage, read> shadowViews : array<ShadowView>;
@group(3) @binding(4) var<storage, read> shadowSlots : array<ShadowSlot>;
```

Storage rather than uniform for the two tables, so view and slot counts are runtime values and the
shader is not recompiled when a light starts casting.

---

## 10. Test suite

Headless (`npm run test:shadows`, extending Phase 1's file) unless marked GPU.

**Pure-logic unit tests:**

- **U1 — allocation never overlaps.** 500 randomized size requests against a 4096² atlas; assert
  every returned rect is in bounds, power-of-two aligned, and pairwise non-overlapping. This is the
  test that catches quad-tree bugs before they look like §6.2's bleed.
- **U2 — cube face select is total and matches the build order.** All 6 axis directions plus the 8
  diagonals and 1000 random directions map to a face whose axis has the largest `|component|` and
  the matching sign; every face index is `< 6`.
- **U3 — the tap clamp.** For a tile at rect `(0.25, 0.25, 0.5, 0.5)`, a sample at the tile's uv
  corner with `pcfRadius: 2` produces atlas UVs all inside `[0.5, 0.75]²`. Remove the `clamp` and
  taps escape — the §9.4 bleed, caught by a number.
- **U4 — hysteresis.** Importance oscillating ±20% around a ladder threshold produces **zero** tile
  resizes; ±30% produces exactly one per crossing.
- **U5 — `hizRect` identity.** `occludedByHiZ` with `hizRect = (1,1,0,0)` returns bit-identical
  results to the Phase 1 / v1.0.5 kernel across 10 000 random spheres. The main view must not
  change when the atlas lands.
- **U6 — spot fov from the cone.** `spotView` built from `outerCos = cos(35°)` yields
  `fovRads ≈ 70°` to 1e-5, so a shadow can never be narrower than the cone it belongs to.
- **U7 — slot packing.** `registerLight` → `slot`, written to `spot_params.z` as `slot + 1`;
  the generated decorator reads `packed - 1u` and treats `0` as unshadowed. Round-trip 64 slots.
- **U8 — eviction is deterministic and safe.** Over-subscribe the atlas by 2×; assert the retained
  set is exactly the top-N by importance, cascades are never evicted (§5.3 rule 2), and evicted
  views report `rect = (0,0,0,0)`, which `sampleView` treats as fully lit.
- **U9 — dirty propagation.** Moving a light marks only its own views dirty; resizing a tile marks
  that view dirty; `updateMode: 'on-demand'` with nothing changed produces **zero** caster batches.

**Integration tests (GPU, one readback each).** Readback goes through a compute pass that
`textureLoad`s the atlas (`depth24plus` is not buffer-copyable — Phase 1 §3).

- **T1 — no tile bleed.** Two adjacent tiles; light A's caster deliberately extends far outside
  light A's frustum. **Expect** light B's tile depth unchanged from its cleared/rendered baseline
  at every texel. Swap the viewport for the §6.2 clip-remap and this test fails — which is its
  entire purpose.
- **T2 — point light shadows in all six directions.** A point light inside a box of six walls.
  **Expect** all six walls shadowed on their far side, and no seam pixel differing from its
  neighbour across a face edge by more than 15%.
- **T3 — spot cone alignment.** A spot with a 35° outer cone over a plane. **Expect** the shadowed
  region's boundary inside the lit cone's boundary everywhere (a shadow never exceeds its cone).
- **T4 — caching correctness.** Render 10 frames with `updateMode: 'on-demand'` and nothing moving;
  **expect** frames 2–10 issue zero caster draws and every frame's *image* is identical to
  `every-frame` mode. Then `invalidate()` one light and expect exactly its views to re-render.
- **T5 — light-view Hi-Z is conservative.** A dense caster field with occlusion on vs. off.
  **Expect** the two lit images identical within 1%, while the shadow-pass draw survivor count
  drops. Any *image* difference means over-culling — the failure §7.2 argues cannot happen, tested
  rather than assumed.
- **T6 — budget overflow is loud.** Request more views than `maxViews`. **Expect** the §11 error
  naming the light, not a dropped shadow.

---

## 11. Errors

| Condition | Message |
|---|---|
| more views than capacity | `ShadowAtlas: light "<name>" requests <n> views, exceeding maxViews <m>; raise maxViews or stop shadowing this light.` |
| a single view larger than the atlas | `ShadowAtlas: view "<name>" requests a <s>px tile in a <a>px atlas; raise atlasSize or lower maxTileSize.` |
| atlas full at the requested sizes | `ShadowAtlas: cannot place a <s>px tile — <f>px² free but fragmented across <k> regions; lower maxTileSize or raise atlasSize.` |
| slot index exceeds the encodable range | `ShadowSystem: shadow slot <n> exceeds 254; spot_params.z encodes slot+1 as a float.` |
| filter margin disagrees with the cube fov | `ShadowSystem: filterTexels <f> exceeds the cube-face fov margin (<m> texels); raise cubeFovMarginTexels or lower pcfRadius — face seams will appear.` |
| viewport unsupported and no fallback opted into | `ShadowSystem: the atlas requires per-batch viewport support (shadows-atlas-design.md §12.1); set allowClipRemap:true to use the bleeding clip-space remap instead.` |
| point light with a non-uniform tile ladder | `ShadowSystem: point light "<name>" got mismatched face tile sizes (<sizes>); cube faces must share a size or the seam filter is undefined.` |
| `invalidate()` on an unregistered slot | `ShadowSystem: invalidate(<n>) — no such shadow slot; registerLight() returns the valid range.` |

---

## 12. Engine touchpoints

### 12.1 Required — per-batch viewport

One addition, in two places, and it is the change the whole atlas rests on (§6.2):

```ts
// core/types.ts — PipelineConfig + RenderBatch
viewport?: [x: number, y: number, width: number, height: number];   // attachment pixels

// core/RenderPipeline.ts, inside the batch loop (after setPipeline)
if (batch.viewport) {
    const [x, y, w, h] = batch.viewport;
    passEncoder.setViewport(x, y, w, h, 0, 1);
    passEncoder.setScissorRect(x, y, w, h);
}
```

Six lines, additive, inert for every existing batch. `setScissorRect` is belt-and-braces — the
viewport transform already guarantees containment — but it makes the invariant explicit and costs
nothing. Beyond shadows, this is what any future tiled or split-screen pass needs.

**Viewport is pass state, not draw state.** Once a batch narrows the viewport it stays narrowed for
every following batch in that pass, so a full-pass batch after a tiled one silently renders into
the previous tile. `RenderPipeline` restores the full attachment between them — but a
`GPUTextureView` does not expose its dimensions, so the pass has to say how big it is:

```ts
// core/RenderPass.ts — RenderPassConfig
size?: [width: number, height: number];   // required only when a pass MIXES the two kinds
```

For `isMainScreenPass` the size is already known (`ctx.width/height`). For an offscreen pass that
mixes viewport-scoped and full-pass batches without declaring a size, `createBatch` throws naming
the pass — the combination it cannot fix is refused at build time rather than rendering into the
wrong rectangle.

### 12.1b Required — `ComputeStageNode.enabled`

```ts
// core/types.ts
public enabled: boolean = true;           // skips preClearBuffers AND the dispatch when false

// core/RenderPipeline.ts, top of the compute branch
if (!node.enabled) continue;
```

Two lines plus a field, defaulting to the current behavior. This is what makes §8.3's cull gating
possible without rebuilding the sequence array every frame, and it is generally useful: any system
owning many stages can now gate them per frame. `VisibilitySystem.getCullStage(batch)` is the
companion accessor that hands an owner the stage feeding a given draw.

### 12.2 Module changes (not core)

- `CullParams` grows `hizRect: vec4<f32>` (32 → 48 bytes), identity for the main view (§7.2).
- `HiZPyramid` — **no change**. It already accepts an arbitrary depth view.
- `VisibilitySystem` — no change beyond Phase 1 §9.3's `camera` / `occlusion` options, plus passing
  `hizRect` through to `writeParams`.
- `LightSystem` — **no change**. The slot rides in the existing 16-float record (§4.3).

### 12.3 Optional — filed, not needed

- **Indirect dispatch** (roadmap §6). Removes the idle lanes in `views × maxInstances` cull
  dispatches (§7.3). This is the release where it earns its keep.
- **`cullMode`** on `PipelineConfig`. Front-face-only caster rendering halves shadow-pass fragment
  work; Phase 1 §12.2 filed it first.
- **Multi-view / layered rendering.** No WebGPU equivalent of `gl_Layer` / view instancing exists;
  detect-and-adopt if one is standardized (roadmap constraint #4).

---

## 13. Build order

Each step renders correctly at its end. Steps 1–3 replace Phase 1's plumbing without changing a
single pixel — which is exactly how the swap should be verified.

1. **`ShadowView` generalization** — collapse Phase 1's cascade into `{ Camera, rect, texelWorld,
   kind }` with `rect = (1,1,0,0)` and the array target retained. **No visual change.** Assert
   Phase 1's T1–T7 still pass. This is a refactor, and doing it first keeps the atlas from hiding a
   regression.
2. **Atlas + viewport (§4–§6)** — one texture, quad-tree allocation, the atlas pass, cascades only.
   Assert U1, T1. T1 is the gate: without it, §6.2's bleed ships.
3. **Tile table + atlas sampling (§9.3)** — `shadowViews`/`shadowSlots` storage buffers and the
   rect-transformed tap. Assert U3, and Phase 1's T1/T3/T4/T5 unchanged.
4. **Slot indirection (§4.3, §9.2)** — `registerLight`, `spot_params.z`, and the deletion of Phase
   1's direction match. Assert U7. Directional shadows now go through the full Phase 2 path.
5. **Spot views (§9.1)** — the simplest punctual case: one view, one tile, a perspective matrix.
   Assert U6, T3.
6. **Point views (§9.1–9.2)** — six faces, the face select, the fov margin. Assert U2, T2. This is
   the step that stresses allocation, so it comes after allocation is proven.
7. **Importance, eviction, hysteresis (§5)** — Assert U4, U8, T6.
8. **Caching (§8)** — dirty tracking, `loadOp:'load'`, tile-clear batches, round-robin. Assert U9,
   T4. Last among the correctness features, because a caching bug looks like every other bug.
9. **Light-view Hi-Z (§7.2)** — `hizRect`, the atlas pyramid. Assert U5, T5. Purely an
   optimization, so it goes last and any regression is isolated to it.

### Deferred

- **Static/dynamic split maps** — a cached tile for static casters plus a per-frame tile for movers,
  combined with `min()` at sample time. Removes §8.1's reliance on the caller calling
  `invalidate()`, at the cost of two tiles per light.
- **Filtered-moment shadows** (VSM/ESM/MSM) and **PCSS** contact hardening — now possible, since the
  atlas can hold a filterable color format, but each brings its own artifact class.
- **Translucent and colored shadows.**
- **Virtual shadow maps (v2.0)** — page-resident tiles driven by which texels the visible receivers
  actually request, streamed like virtual geometry (`geometry-meshlets-design.md` §13). This atlas,
  its tile table, and its per-view culling are the substrate that makes VSMs a v2.0 feature rather
  than a rewrite.
