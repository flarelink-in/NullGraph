# Shadows — Shader & Buffer Reference

Companion to [`shadows.md`](./shadows.md) (usage),
[`shadows-design.md`](../shadows-design.md) (Phase 1 architecture), and
[`shadows-atlas-design.md`](../shadows-atlas-design.md) (Phase 2 architecture). This is the
**contract layer**: exact byte layouts, WGSL bindings, the generated kernels, and how the shadow
term reaches a material.

Read this when you are writing a custom receiver shader, debugging a bind-group error, computing
cascades in a worker, or wiring shadows into a hand-written material.

> **Phase 1 vs Phase 2.** Phase 1 (no `atlasSize`) uses a `depth24plus` **2D array**, one layer
> per cascade, and the `ShadowData` uniform — §1–§5 below. Phase 2 (`atlasSize` set) uses **one
> atlas texture** plus runtime `shadowViews`/`shadowSlots` storage tables — §A1–§A5. The caster
> vertex shader (§5) and the engine touchpoints (§7) are shared.

---

## 1. Buffer layouts

Every structure is 16-byte aligned and worker-fillable. Sizes are exact.

### 1.1 The cascade camera — 432 bytes (108 floats)

One `GPUBuffer` per cascade, filled with the **exact** `Camera` layout (`core/Camera.ts:35`) so
`CAMERA_STRUCT_WGSL` reads it unchanged. That is the whole trick: a cascade is indistinguishable
from a camera to every existing shader.

| Floats | Bytes | Field | Cascade value |
|---|---|---|---|
| 0–15 | 0–63 | `viewProj` | `orthoZO(l,r,b,t,0,2r+e) × lookAt(eye, center, up)` |
| 16–31 | 64–127 | `view` | the light's `lookAt` |
| 32–47 | 128–191 | `proj` | the `orthoZO` |
| 48–63 | 192–255 | `invViewProj` | inverse (unused by the cull path; filled for completeness) |
| 64–79 | 256–319 | `invView` | inverse |
| 80–95 | 320–383 | `invProj` | inverse |
| 96–98 | 384–395 | `eyePos` | ortho origin: `center − dir × (radius + extrusion)` |
| 99 | 396 | `simTime` | copied from the main camera |
| 100 | 400 | `zNear` | `0` (the extruded near plane, §5.4) |
| 101 | 404 | `zFar` | `2 × radius + extrusion` |
| 102 | 408 | `fovRads` | the **virtual** fov reproducing this cascade's texel density (§5.5) |
| 103 | 412 | `aspect` | `1.0` (cascades are square) |

`buildCascades()` / `fitCascade()` produce this array; `update()` writes it verbatim into the
cascade's buffer. A worker can fill the same 108 floats and hand them over via
`setRawCascades(views, shadowData)` — `views` is `cascades` packed 108-float blocks.

> **ZO, not NO.** The projection is `mat4.orthoZO` — WebGPU clip is `z ∈ [0,1]`. Swapping in the
> NO form (`ortho`) silently moves the near plane, the same trap `Camera.ts:62-66` documents for
> perspective. Pinned by test U4.

### 1.2 `ShadowCasterParams` — 16 bytes (uniform, per cascade)

One per cascade, written by `update()`. The caster's bias constants are **exact for the cascade
being rendered** — a single shared `ShadowData` cannot serve the caster, because the cascade index
is a per-*pass* property and a uniform-array index would be a per-*vertex* one.

| Float | Field | Source |
|---|---|---|
| 0 | `texelWorld` | `2 × radius / resolution` for THIS cascade |
| 1 | `normalOffsetTexels` | `config.normalOffsetTexels` (default 1.5) |
| 2 | `constantBias` | `config.constantBias` (default 0.0015) |
| 3 | `_pad` | 0 |

### 1.3 `ShadowData` — 352 bytes (uniform, one per system)

The receiver uniform, bound at the shadow group (default 3). `MAX_CASCADES = 4` sizes the fixed
array. Written once per frame from a persistent `Float32Array` by `packShadowData()`.

```wgsl
struct ShadowCascade {
    viewProj   : mat4x4<f32>,   // 64 B  world → cascade clip
    splitFar   : f32,           //       view-space depth this cascade covers up to
    texelWorld : f32,           //       world size of one shadow texel (bias + filter radius)
    _pad0      : f32,
    _pad1      : f32,           // 16 B
};                              // 80 B, 16-aligned

struct ShadowData {
    cascades      : array<ShadowCascade, MAX_CASCADES>,   // 80 × 4 = 320 B
    lightDir      : vec4<f32>,   // .xyz normalized light travel direction, .w unused
    cascadeCount  : u32,
    pcfRadius     : u32,         // taps per side: 1 → 3×3, 2 → 5×5, 0 → hardware 2×2 only
    blendTexels   : f32,         // cascade cross-fade band, in split-distance units
    constantBias  : f32,
};                              // 320 + 32 = 352 B
```

`cascadeCount` and `pcfRadius` occupy float slots but hold **raw u32 bits** — `packShadowData`
writes them through a `Uint32Array` view over the same memory (the `InstanceBuffer` contract).

### 1.4 The cascade array (texture)

| Resource | Format / usage | Size |
|---|---|---|
| cascade array | `depth24plus`, `RENDER_ATTACHMENT \| TEXTURE_BINDING`, `depthOrArrayLayers = cascades` | `resolution² × N` |
| per-layer views | `createView({ dimension:'2d', baseArrayLayer:i, arrayLayerCount:1 })` | the render targets |
| sampling view | `createView({ dimension:'2d-array' })` | bound as `texture_depth_2d_array` |
| comparison sampler | `createSampler({ compare:'less', magFilter:'linear', minFilter:'linear' })` | hardware 2×2 PCF |

4 × 2048² `depth24plus` ≈ **16 MB**, allocated once. `depth24plus` is samplable +
comparison-filterable (core WebGPU — no optional feature anywhere), but it is **not**
buffer-copyable, so the GPU tests read it back through a compute pass (`textureLoad` into an f32
storage buffer) — legal in a test, forbidden in the frame loop.

---

## 2. Bind-group contracts

`layout: 'auto'` prunes any binding the shader does not *use*, and supplying an entry for a pruned
binding throws. The generated modules therefore reference exactly what they bind.

### 2.1 The caster pass (depth-only)

Group 0 is engine-fixed by `createBatch`; the caster params ride the shadow group:

| Group | Binding | Resource |
|---|---|---|
| 0 | 0 | `camera` — the **cascade's** 432-byte Camera (`PipelineConfig.cameraBuffer`) |
| 0 | 1 | `instances` — the compacted survivors (`batch.storageBuffer`) |
| 3 | 3 | `casterParams` — the per-cascade `ShadowCasterParams` uniform |

The vertex stage genuinely reads both group-0 bindings (the culled instances and the cascade
camera), so nothing is pruned. The caster shader is vertex-only unless `allowColorFallback` adds
the throwaway `fs_main` (§6.3).

### 2.2 The receiver (scene material)

The decorator emits the shadow resources into the **same module** as the light bindings, at the
shadow group (default 3):

| Group | Binding | Resource |
|---|---|---|
| 0 | 0 | `camera` (the main view) |
| 1 | — | the material |
| 2 | — | `LightSystem`'s light buffers (hardcoded) |
| 3 | 0 | `shadowMaps : texture_depth_2d_array` |
| 3 | 1 | `shadowSampler : sampler_comparison` |
| 3 | 2 | `shadow : ShadowData` (uniform) |

> **Group 3 by default.** Group 0 is engine-fixed, group 1 is the material on the instance tier,
> group 2 is hardcoded into `LightSystem`'s WGSL. `shadowGroupIndex` is configurable for the
> meshlet tier, where group 1 already holds geometry — but it must be **≥ 3** (the constructor
> throws otherwise).

---

## 3. The decorator — how the shadow term reaches a material

`wrapLighting(lightSystem)` returns a `{ wgslStructs, wgslLogic }` of the same shape as
`LightSystem.getShaderLayout()`. It performs **one deterministic textual rename**, guarded by a
loud error if the marker ever moves:

```ts
const marker = 'fn getIncidentLight(';
wgslLogic.replace(marker, 'fn getIncidentLightUnshadowed(')   // the original, renamed
    + SHADOW_SAMPLING_WGSL                                    // sampleCascade / sampleShadow
    + SHADOW_BIASED_WGSL                                      // shadowFactorBiased (optional)
    + buildShadowDecoratorWGSL(accessor);                     // the new getIncidentLight
```

The replacement `getIncidentLight` calls the original and folds the shadow term into
`attenuation`, reading the light record directly to decide whether *that* light casts:

```wgsl
fn getIncidentLight(i: u32, worldPos: vec3<f32>) -> IncidentLight {
    var inc = getIncidentLightUnshadowed(i, worldPos);
    let rec = ${accessor};                  // standard:  lights[i]
                                            // clustered: globalLights[lightIndices[getClusterData(worldPos).start_offset + i]]
    if (rec.pos_type.w == 2.0 && dot(normalize(rec.dir_rad.xyz), shadow.lightDir.xyz) > 0.999) {
        inc.attenuation = inc.attenuation * sampleShadow(worldPos);
    }
    return inc;
}
```

Lambert, Toon, and PBR all consume `incident.attenuation`, so all three gain shadows with zero
edits. The `accessor` exists (rather than a direction-only test) because Phase 2 replaces the dot
product with an explicit per-light shadow slot carried in the light record's free `spot_params.z`.

> **The guard is the safety story.** U6 pins the rename against the real `LightSystem` output for
> both `standard` and `clustered-forward`, so a rename in `LightSystem.ts` fails a headless test
> rather than silently shipping unshadowed lighting.

### 3.1 Why the receiver has no normal

The material contract is `getIncidentLight(i, worldPos)` — no normal. Recovering the geometric
normal with `dpdx/dpdy(worldPos)` is illegal here: the call sits inside `for (i < lightCount)`
where `lightCount` comes from a cluster lookup and is **non-uniform across the quad**, which
WGSL's `derivative_uniformity` diagnostic rejects. That is the constraint that pushed all bias
onto the caster (§7). A hand-written fragment that *has* `N` in scope may call
`shadowFactorBiased(worldPos, N, L)` for receiver-side normal offset and drop `normalOffsetTexels`
to 0.

---

## 4. The sampling functions

```wgsl
fn sampleShadow(worldPos: vec3<f32>) -> f32 {
    let viewDepth = -(camera.view * vec4<f32>(worldPos, 1.0)).z;
    var c = shadow.cascadeCount;                       // sentinel: beyond the last split
    for (var k = 0u; k < shadow.cascadeCount; k = k + 1u) {
        if (viewDepth < shadow.cascades[k].splitFar) { c = k; break; }
    }
    if (c >= shadow.cascadeCount) { return 1.0; }      // past shadowDistance → fully lit
    let lit = sampleCascade(c, worldPos);
    // Cross-fade into the next cascade over the last blendTexels of this one.
    let edge = shadow.cascades[c].splitFar - shadow.blendTexels;
    if (c + 1u < shadow.cascadeCount && viewDepth > edge) {
        let t = (viewDepth - edge) / shadow.blendTexels;
        return mix(lit, sampleCascade(c + 1u, worldPos), clamp(t, 0.0, 1.0));
    }
    return lit;
}
```

`sampleCascade` projects into the cascade's clip space, flips Y (clip +Y up → texture +V down),
and PCF-filters:

```wgsl
let ref   = ndc.z - shadow.constantBias;
let r     = i32(shadow.pcfRadius);
if (r == 0) { return textureSampleCompareLevel(shadowMaps, shadowSampler, uv, k, ref); }
// else a (2r+1)² grid of hardware-2×2 taps
```

> **`textureSampleCompareLevel`, never the bare `textureSampleCompare`.** The latter takes implicit
> derivatives and is only valid in uniform control flow; every call above sits inside a cascade
> branch and a PCF loop. Using the bare form is a shader-compile error at best and a
> derivative-uniformity trap at worst — pinned by test U7, which asserts the generated WGSL never
> emits `textureSampleCompare(`.

Each tap is hardware 2×2-comparison-filtered by the comparison sampler, so `pcfRadius: 1` is a 3×3
grid of bilinear-PCF taps ≈ a 6×6 kernel.

---

## 5. The caster vertex shader — all bias lives here

`PipelineConfig` exposes neither `depthBias` nor `cullMode`, and the receiver has no normal in
scope (§3.1), so both bias terms live in the depth-only vertex shader, where the normal *is* in
scope and control flow *is* uniform:

```wgsl
@vertex
fn vs_main(@location(0) pos: vec3<f32>, @location(1) normal: vec3<f32>,
           @builtin(instance_index) ii: u32) -> @builtin(position) vec4<f32> {
    let inst = instances[ii];
    let n = normalize(normalMatrixFromModel(inst.model) * normal);
    let world = (inst.model * vec4<f32>(pos, 1.0)).xyz;

    // §7.2 — normal offset, scaled by THIS cascade's world texel size (resolution-correct).
    let biasedWorld = world + n * (casterParams.texelWorld * casterParams.normalOffsetTexels);

    var clipPos = camera.viewProj * vec4<f32>(biasedWorld, 1.0);
    // §7.1 — constant bias in clip space, × w so it is a plain NDC-z offset under ortho (w == 1).
    clipPos.z = clipPos.z + casterParams.constantBias * clipPos.w;
    return clipPos;
}
```

The tradeoff (§7.3): caster-side offset trades **acne** for **peter-panning** — contact shadows
detach by roughly `texelWorld × normalOffsetTexels`. At 2048 over a 40-unit cascade that is ~3 cm
(invisible); at 512 over 200 units it is ~1.2 m (obvious). The first cascade's world extent is what
sets contact-shadow quality — the real argument for cascades over one big map. `update()` throws
(§11) when the computed offset exceeds 2% of a cascade's extent.

---

## 6. Generated kernels

Every generator is exported, so you can print one and read it — the fastest way to debug a
bind-group mismatch.

| Function | Emits |
|---|---|
| `buildShadowCasterWGSL({ shadowGroupIndex, colorFallback })` | the depth-only caster VS (+ optional throwaway FS) |
| `buildShadowBindingsWGSL(group)` | the `@group(g)` shadow resource declarations |
| `buildShadowDecoratorWGSL(accessor)` | the shadowed `getIncidentLight` |

Reusable chunks: `SHADOW_STRUCTS_WGSL`, `SHADOW_SAMPLING_WGSL`, `SHADOW_BIASED_WGSL`. The caster
also reuses `CAMERA_STRUCT_WGSL`, `INSTANCE_STRUCT_WGSL`, and `CULL_CORE_WGSL` from
[`visibilityShaders`](./visibility-shader-reference.md#5-generated-kernels) unchanged — a cascade
is a camera, so `frustumPlanes(camera.viewProj)` and `worldSphereFromModel` work with no
shadow-specific branch.

```ts
import { buildShadowCasterWGSL } from 'null-graph/shadows';
console.log(buildShadowCasterWGSL({}));
```

---

## 7. The engine touchpoints (§12.1)

Phase 1 is the first NullGraph module that asks for a core change. Both are additive, opt-in, and
inert for every existing caller:

```ts
// PipelineConfig.depthOnly — omit the fragment stage and color targets entirely.
// A fragment-less render pipeline is core WebGPU.
engine.createBatch(pass, { /* … */, depthOnly: true });

// PipelineConfig.cameraBuffer — the group-0 binding-0 resource, defaulting to
// ctx.cameraUniformBuffer. One line in BatchManager.ts:
{ binding: 0, resource: { buffer: config.cameraBuffer || this.ctx.cameraUniformBuffer } }
```

`cameraBuffer` is what makes "a cascade is a camera" true end-to-end, and it is the enabling
change for every future off-main-view render: shadows, reflection probes, IBL capture, and any
editor picking pass. Without `depthOnly`, the §6.3 fallback renders with a throwaway `r8unorm`
color target at ~1.5–2× the pass cost (`allowColorFallback: true`).

---

## 8. Debugging checklist

| Symptom | Likely cause |
|---|---|
| No shadow at all | `shadows.getPasses()` not spliced **before** the scene pass; or `update(camera)` not called this frame; or the light's `dir_rad` doesn't match `setLight` (the direction match is `> 0.999`). |
| Shadow acne (striped self-shadowing) | `constantBias` / `normalOffsetTexels` too low for the resolution, or a cascade too coarse for its extent. |
| Contact shadows detached (peter-panning) | `normalOffsetTexels × texelWorld` too large — the §11 guard throws at 2% of the extent. Raise `resolution` or lower `normalOffsetTexels`. |
| Shadows pop in as objects enter view | `casterExtrusion` too small — a caster behind the cascade was culled (§5.4). This is the most common CSM bug. |
| Shimmering / crawling edges as the camera moves | Texel snapping broken — `resolution` not a power of two, or the sphere fit re-quantized (§5.3/§5.5). |
| A visible seam between cascades | `blendTexels: 0` — the cross-fade band is what removes the resolution change (§8.3). |
| Bind-group creation throws on the caster | A declared-but-unused binding was pruned by `layout:'auto'` — the caster VS must reference both `camera` and `instances` (it does). |
| Everything past `shadowDistance` is dark | By design shadows stop at `shadowDistance` (not `camera.far`); raise it or shrink the cascade set. |

---
---

# Phase 2 — the atlas contract

Everything below applies when `atlasSize` is set. The Phase 1 layouts above (the cascade camera,
the caster params, the caster VS) are unchanged — the atlas *generalizes* the cascade into a
`ShadowView` plus an atlas rectangle, it does not replace the math.

## A1. The atlas + the view/slot tables

| Resource | Format / usage | Size |
|---|---|---|
| atlas | `depth24plus`, `RENDER_ATTACHMENT \| TEXTURE_BINDING` | `atlasSize²` (default 4096² ≈ 32 MB) |
| sampling view | `createView()` — plain `texture_depth_2d` | — |
| comparison sampler | `createSampler({ compare:'less', magFilter:'linear', minFilter:'linear' })` | — |
| `shadowViews` | `STORAGE`, `array<ShadowView>` | `96 × maxViews` |
| `shadowSlots` | `STORAGE`, `array<ShadowSlot>` | `16 × maxSlots` |
| per-view camera | `UNIFORM`, 432 B each | `432 × maxViews` |
| light-view Hi-Z | `HiZPyramid` over the atlas, `r32float` | `atlasSize² × 4/3` |

The two tables are `storage`, not `uniform`, so view and slot counts are **runtime values** and
the receiver shader is not recompiled when a light starts casting (§9.5).

### A1.1 `ShadowView` — 96 bytes (24 floats, 16-aligned)

What the *receiver* needs of a view. The CPU side additionally keeps the full 432-byte `Camera`
(§1.1) for the cull stage and the depth draw.

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

`rect` is the atlas's defining field: the tap offsets in the *view's* uv, clamps to `[0,1]²`, then
maps into the atlas via `local * rect.xy + rect.zw` — so a PCF tap can never read a neighbouring
light's depth (§9.4, pinned by U3).

### A1.2 `ShadowSlot` — 16 bytes (4 floats)

One per shadow-casting light. Directional lights own `cascadeCount` consecutive views, spots one,
points exactly six in `+X,−X,+Y,−Y,+Z,−Z` order so face selection is arithmetic (§9.2).

```wgsl
struct ShadowSlot {
    firstView  : u32,   // index into the views array
    viewCount  : u32,   // cascades: N; spot: 1; point: 6
    kind       : u32,   // 0 = directional, 1 = spot, 2 = point
    lightRange : f32,   // punctual only: for the depth remap at sample time
};
```

### A1.3 `ShadowParams` — 48 bytes (uniform, one per system)

Smaller than Phase 1's `ShadowData`: the per-cascade matrices and splits moved into the
`shadowViews` table, leaving only the global filter/bias knobs and the cascade split distances
(for `selectCascade`).

```wgsl
struct ShadowParams {
    atlasSize     : f32,          // atlas edge in px — texel = filterTexels / atlasSize
    pcfRadius     : u32,          // taps per side: 1 → 3×3, 2 → 5×5, 0 → hardware 2×2 only
    constantBias  : f32,
    filterTexels  : f32,          // PCF kernel width in texels (drives the tap step)
    cascadeSplits : vec4<f32>,    // view-space split far distances, up to MAX_CASCADES
    cascadeCount  : u32,
    _pad0         : f32,
    _pad1         : f32,
    _pad2         : f32,
};
```

## A2. The slot index on the light — `spot_params.z`

`spot_params.z` (float 14 of the 16-float light record) carries `slotIndex + 1`; `0` means "this
light does not cast." Free on **every** light type, requires no `LightSystem` edit, and the caller
sets it in the same `Float32Array` a worker already fills (`LightSystem.setRawData`):

```ts
const slot = shadows.registerLight({ kind: 'point', position, range: 30 });
lightData[lightIndex * 16 + 14] = slot + 1;      // spot_params.z
```

The Phase 1 `dot(dir, shadow.lightDir) > 0.999` heuristic is **deleted** in the atlas path, along
with its "two identical directional lights collide" caveat. Slots top out at 254 (the field
encodes `slot+1` as a float) — exceeding it throws the §11 error.

## A3. The receiver bindings (§9.5)

```wgsl
@group(3) @binding(0) var shadowAtlas   : texture_depth_2d;
@group(3) @binding(1) var shadowSampler : sampler_comparison;
@group(3) @binding(2) var<uniform> shadow : ShadowParams;
@group(3) @binding(3) var<storage, read> shadowViews : array<ShadowView>;
@group(3) @binding(4) var<storage, read> shadowSlots : array<ShadowSlot>;
```

## A4. The atlas decorator — slot → view → tap

The Phase 2 `getIncidentLight` keeps the decorator's shape; only the middle changes — the
direction match becomes a slot lookup, and the view choice becomes type-dependent:

```wgsl
fn getIncidentLight(i: u32, worldPos: vec3<f32>) -> IncidentLight {
    var inc = getIncidentLightUnshadowed(i, worldPos);
    let rec = ${accessor};                          // lights[i] | globalLights[lightIndices[...]]
    let packed = u32(rec.spot_params.z);            // 0 = no shadow, slot+1 otherwise
    if (packed == 0u) { return inc; }

    let slot = shadowSlots[packed - 1u];
    var v = slot.firstView;
    if (slot.kind == 0u) {                          // directional: pick the cascade
        let viewDepth = -(camera.view * vec4<f32>(worldPos, 1.0)).z;
        v = v + selectCascade(viewDepth);
    } else if (slot.kind == 2u) {                   // point: pick the cube face
        v = v + cubeFace(worldPos - rec.pos_type.xyz);
    }                                               // spot: the single view
    inc.attenuation = inc.attenuation * sampleView(v, worldPos);
    return inc;
}
```

`cubeFace` is the major-axis select matching the build order; `sampleView` is the atlas tap (§9.3)
with the tile clamp (§9.4). Both use `textureSampleCompareLevel` — never the bare form (the same
derivative-uniformity rule as Phase 1, §4).

## A5. The atlas pass + the tile clear

One depth-only pass holds every view's draws, each scoped to its tile by `PipelineConfig.viewport`
(§12.1). Caching needs `depthLoadOp: 'load'` on the atlas, so each dirty tile clears itself with a
`TILE_CLEAR_WGSL` batch — one full-viewport triangle at `z = 1.0`, `depthCompare: 'always'`,
scoped to the same viewport, drawn first (§8.2). Three vertices per dirty tile, no second pass.

```wgsl
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(pos[vi], 1.0, 1.0);
}
```

## A6. The visibility touchpoints (§12.2)

The atlas reuses the `VisibilitySystem` cull chain per view; two fields ride the existing
`CullParams` uniform (32 → 48 bytes):

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
};                               // 48 B
```

`occludedByHiZ` applies `hizRect` just before sampling, so a shadow view samples only its own
atlas tile. With `hizRect = (1,1,0,0)` the expression is the identity and the main view is
bit-for-bit unchanged (§10 U5). `screenSize` for a shadow view is the **tile** size, so the
footprint→mip selection stays correct. `HiZPyramid` itself is **unchanged** — it already accepts
an arbitrary depth view, so the light-view pyramid is built over last frame's atlas verbatim.
