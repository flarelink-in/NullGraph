# NullGraph Shadows — Phase 1: Cascaded Shadow Maps — Technical Design

**Target:** `src/extras/shadows/` (`ShadowSystem`, `CascadeBuilder`, `shadowMath`, `shadowShaders`)
**Package:** `null-graph/shadows` (new subpath export)
**Status:** `v1.1` design, implementation-ready.
**Companion:** [`shadows-atlas-design.md`](./shadows-atlas-design.md) — Phase 2: the atlas, punctual
lights, caching, and shadow-view Hi-Z. Read this one first.
**Depends on:** [`visibility-design.md`](./visibility-design.md) (GPU caster culling),
[`api/lights.md`](./api/lights.md) (the lighting contract this decorates).
**Roadmap:** [`roadmap.md`](./roadmap.md) §3 · [`roadmap-modules.md`](./roadmap-modules.md) §4.

---

## 1. Purpose

Give NullGraph a **directional shadow**: a sun that occludes. Phase 1 builds the whole vertical
slice for the one light that matters most in an open world — depth-only caster passes, cascaded
frusta fitted to the view, GPU caster culling, and a filtered receiver-side lookup that drops into
every existing material without touching `LightSystem` or the material builders.

Phase 1 is deliberately **one light, N cascades**. Everything that makes shadows a *system* —
many lights, an atlas, tile caching, occlusion-culled casters — is Phase 2, and Phase 2 is built
by generalizing Phase 1's `ShadowView`, not by replacing it.

### The single most important pre-existing fact

**A shadow map is the same draw the main pass already makes, from a different matrix.** The
`VisibilitySystem` already turns "a camera + an instance buffer" into "a handful of
`drawIndexedIndirect` calls, GPU-culled." A cascade *is* a camera — an orthographic one. So caster
culling is not new machinery; it is the existing cull chain pointed at a different `Camera` uniform.

That reuse is what makes this release cheap, and §3 is where it stops being free: the engine binds
**one global camera buffer** into every batch (`BatchManager.ts:100`), so "point it at a different
camera" needs a one-line, additive engine change (§12). The roadmap's claim that shadows get GPU
culling "for free" is *almost* true; §12 is the exact price.

### Non-goals for Phase 1

- **Punctual shadows.** Spot and point lights are Phase 2 — they need the atlas, because six cube
  faces per point light is where fixed per-light targets stop scaling.
- **A shadow atlas.** Phase 1 renders to a `depth24plus` **2D array**, one layer per cascade. Simple,
  correct, and the thing Phase 2 replaces.
- **Occlusion-culled casters.** No Hi-Z on the light view (Phase 2 §8). Frustum-only caster culling.
- **Contact-hardening / area shadows.** PCSS, VSM/ESM/MSM are out. Fixed-radius PCF only.
- **Ray-traced shadows.** ⛔ blocked on WebGPU (roadmap-modules.md §12).

### Design principles

1. **A cascade is a Camera.** Every shadow view fills the *same* 108-float `Camera` layout
   (`Camera.ts:35`) the cull and draw shaders already consume. Nothing in `visibilityShaders.ts`
   learns what a shadow is; it is handed a different uniform and keeps working.
2. **All bias lives on the caster.** `PipelineConfig` exposes no `depthBias` and no `cullMode`
   (`types.ts:49-51`, `BatchManager.ts:66-71`), and the receiver-side lighting contract has no
   normal in scope (§8.2). So the constant and normal offsets are applied in the depth-only
   **vertex shader**, where the normal *is* in scope and control flow *is* uniform. This is a
   consequence of the engine as it exists, not a preference — and it is why Phase 1 needs no
   rasterizer-state change.
3. **Decorate the lighting contract; don't fork it.** Shadowing wraps `getIncidentLight` and folds
   the shadow term into `IncidentLight.attenuation` (§8). Lambert, Toon, and PBR all consume
   `incident.attenuation` (`materialFragments.ts:286,388,468+`), so all three gain shadows with
   zero edits to any material.
4. **Conservative always beats clever.** A caster wrongly culled is a missing shadow — a hole the
   user sees in the *lit* image, one indirection removed from the cause. Every cascade fit errs
   toward including geometry (§5.4 caster extrusion, §5.3 sphere fit).
5. **Detect, don't require.** Baseline is core WebGPU: `depth24plus` with `TEXTURE_BINDING`, a
   comparison sampler, a 2D-array depth attachment. No optional feature appears anywhere.

---

## 2. Architecture

Per frame, `ShadowSystem` contributes a cull chain and N depth-only passes, **before** the scene
pass that samples them:

```
   engine.render([
      ...vis.getComputeStages(),        // main-view culls (Hi-Z, frustum, LOD)
      ...shadows.getComputeStages(),    // per-cascade caster culls  ── batched into the same compute pass
      ...shadows.getPasses(),           // N depth-only render passes, one per cascade layer
      scenePass,                        // materials sample the cascade array
      ...post.getPassSequence(),
   ]);
```

```
        ┌───────────────────────────────────────────────────────────────────┐
        │  ShadowSystem              user-facing surface                    │
        │  setLight / registerCaster / update(camera) / wrapLighting(lights)│
        └───┬────────────────┬──────────────────┬──────────────────┬────────┘
            │                │                  │                  │
     ┌──────▼──────┐  ┌──────▼───────┐  ┌───────▼────────┐  ┌──────▼────────┐
     │CascadeBuilder│  │ cascade cams │  │ caster culls   │  │ depth passes  │
     │ split+fit+snap│  │ 108f × N     │  │ (VisibilitySys)│  │ (array layer) │
     └──────┬───────┘  └──────┬───────┘  └───────┬────────┘  └──────┬────────┘
            │                 │                  │                  │
      CPU, once/frame    GPUBuffer per      indirect draw args   depth24plus
      (worker-safe)      cascade (Camera)   (drawArgs[1])        array, N layers
                                                                      │
                                                          ┌───────────▼──────────┐
                                                          │ wrapLighting → WGSL  │
                                                          │ attenuation *= shadow│
                                                          └──────────────────────┘
```

Every box is an existing primitive: `createPass` for the depth passes (`NullGraph.ts:106`), the
`VisibilitySystem` cull stages for casters, `setRenderPassBindGroup` for the receiver bindings, and
`getShaderLayout()`-shaped WGSL for the material bridge.

---

## 3. Integration facts (verified against this repository)

Load-bearing. Each was read from source, not assumed. The first three are the ones that shape the
design.

| Fact | Source | Consequence |
|---|---|---|
| **`createBatch` always creates a fragment stage with ≥1 color target** — `formats = config.targetFormats \|\| [config.targetFormat \|\| ctx.format]`. A depth-only render pass has **zero** color attachments, and a pipeline whose color-target count disagrees with the pass is a WebGPU validation error. | `BatchManager.ts:37-61` | A true depth-only shadow pass needs `PipelineConfig.depthOnly` (§12.1). Zero-change fallback in §6.3. |
| **`createBatch` hard-binds the global camera** at group 0 binding 0: `{ binding: 0, resource: { buffer: this.ctx.cameraUniformBuffer } }`. One camera per frame, shared by every batch. | `BatchManager.ts:94-102` | A cascade cannot get its matrix through group 0 without `PipelineConfig.cameraBuffer` (§12.1). Zero-change fallback in §6.4. |
| **`layout:'auto'` prunes unused bindings**, and `createBindGroup` throws on an entry the pruned layout lacks. So *whatever* is bound at group 0 must be used by the shader. | `BatchManager.ts:94-102`, `visibility-design.md` §7.3 | The depth-only vertex shader must reference both camera@0 and storageBuffer@1. It genuinely uses @1 (the culled instances); @0 becomes real once `cameraBuffer` lands. |
| The depth-stencil state's **format is hardcoded `depth24plus`**; only `depthWriteEnabled` and `depthCompare` are configurable. No `depthBias`, no `cullMode`. | `BatchManager.ts:66-71`, `types.ts:49-51` | Shadow maps are `depth24plus` (samplable + comparison-filterable — fine). All bias moves to the caster VS (§7). `depth24plus` is **not** buffer-copyable, so GPU tests read it back through a compute pass (§10). |
| `expectsDepth` is true when `pass.depthStencilAttachment !== undefined`, so an offscreen depth-only pass does get a depth-stencil state. | `BatchManager.ts:69` | A shadow `RenderPassNode` needs only its `depthStencilAttachment` set. |
| `RenderPassNode.colorAttachments` defaults to `[]` and `RenderPipeline` passes attachments through verbatim for non-main passes. | `RenderPass.ts:24`, `RenderPipeline.ts:72-95` | The pass side of depth-only already works. Only the *pipeline* side is missing. |
| Consecutive `ComputeStageNode`s batch into one WebGPU compute pass; a `RenderPassNode` closes it. Pass boundaries give the compute→draw and draw→sample barriers for free. | `RenderPipeline.ts:46,61-66` | Cascade culls batch with the main-view culls at no extra cost; the shadow passes then read their results. |
| The camera uniform is **108 floats / 432 bytes**: `viewProj@0, view@16, proj@32, invViewProj@48, invView@64, invProj@80, eye@96, simTime@99, near@100, far@101, fovRads@102, aspect@103`. | `Camera.ts:35,76-88,102-111`, `WebGPUContext.ts:131-135` | A cascade fills this exact layout (§4.1) and is then indistinguishable from a camera to every existing shader. |
| The projection is **`perspectiveZO`** — WebGPU clip is `z ∈ [0,1]`, and the `NO` variants silently move the near plane. | `Camera.ts:62-66` | Cascade orthographic matrices must use **`mat4.orthoZO`**, and the cull's near plane stays the ZO form (`r2`, not `r3+r2`). |
| `frustumPlanes` extracts Gribb–Hartmann planes from any `viewProj`, with the ZO near-plane form already correct. | `visibilityShaders.ts:86-98`, `cullMath.ts` | Orthographic caster culling reuses `CULL_CORE_WGSL` **unmodified**. This part really is free. |
| Materials consume shading through `IncidentLight { color, direction, attenuation }`, looped by `getVisibleLightCount` / `getIncidentLight`, which `LightSystem.getShaderLayout()` emits into `@group(2)`. | `LightSystem.ts:117-378`, `materialFragments.ts:276-296,373-388,461-482` | Shadowing is a decorator over `getIncidentLight` (§8), and it must not use group 2. Default shadow group is **3**. |
| `VisibilitySystem` binds `ctx.cameraUniformBuffer` into every cull stage and gates occlusion system-wide (`this.occlusion`). | `VisibilitySystem.ts:244,324,411,101` | Caster culling needs two additive **module** options: `camera?: GPUBuffer` and `occlusion?: boolean` per draw batch (§9.3). Not a core-engine change. |

---

## 4. Data model

### 4.1 `ShadowView` — a cascade is a Camera

One `GPUBuffer` per cascade, 432 bytes, `UNIFORM | COPY_DST`, filled with the **exact** `Camera`
layout so `CAMERA_STRUCT_WGSL` reads it unchanged:

| Floats | Field | Cascade value |
|---|---|---|
| 0–15 | `viewProj` | `orthoZO(l,r,b,t,n,f) × lookAt(eye, center, up)` |
| 16–31 | `view` | the light's `lookAt` |
| 32–47 | `proj` | the `orthoZO` |
| 48–63 | `invViewProj` | inverse — unused by the cull path, filled for completeness |
| 64–79 | `invView` | inverse |
| 80–95 | `invProj` | inverse |
| 96–98 | `eyePos` | ortho origin: `center - dir × (radius + extrusion)` |
| 99 | `simTime` | copied from the main camera |
| 100 | `zNear` | `0` (the extruded near plane, §5.4) |
| 101 | `zFar` | `2 × radius + extrusion` |
| 102 | `fovRads` | the **virtual** fov that reproduces this cascade's texel density (§5.5) |
| 103 | `aspect` | `1.0` (cascades are square) |

Everything downstream — `frustumPlanes(camera.viewProj)`, `worldSphereFromModel`, `selectLodIndex` —
then works with no shadow-specific branch. That is the whole trick.

### 4.2 `ShadowData` — the receiver uniform

One uniform for the whole system, bound at the shadow group (default 3):

```wgsl
struct ShadowCascade {
    viewProj   : mat4x4<f32>,   // 64 B  world → cascade clip
    splitFar   : f32,           //       view-space depth this cascade covers up to
    texelWorld : f32,           //       world size of one shadow texel (bias + filter radius)
    _pad0      : f32,
    _pad1      : f32,           // 16 B
};                              // 80 B, 16-aligned

struct ShadowData {
    cascades      : array<ShadowCascade, MAX_CASCADES>,   // 80 × N
    lightDir      : vec4<f32>,   // .xyz normalized light travel direction, .w unused
    cascadeCount  : u32,
    pcfRadius     : u32,         // taps per side: 1 → 3×3, 2 → 5×5, 0 → hardware 2×2 only
    blendTexels   : f32,         // cascade cross-fade band, in split-distance units
    constantBias  : f32,
};
```

At `MAX_CASCADES = 4` that is 320 + 32 = **352 bytes** — one small `UNIFORM` buffer, rewritten once
per frame from a persistent `Float32Array` (no allocation).

### 4.3 Textures — allocated once

| Resource | Format / usage | Size |
|---|---|---|
| cascade array | `depth24plus`, `RENDER_ATTACHMENT \| TEXTURE_BINDING`, `depthOrArrayLayers = cascadeCount` | `resolution² × N` |
| per-layer views | `createView({ dimension:'2d', baseArrayLayer:i, arrayLayerCount:1 })` | the render targets |
| sampling view | `createView({ dimension:'2d-array' })` | bound as `texture_depth_2d_array` |
| comparison sampler | `createSampler({ compare:'less', magFilter:'linear', minFilter:'linear' })` | hardware 2×2 PCF |

4 × 2048² `depth24plus` ≈ **16 MB**. Nothing here is per-frame.

---

## 5. Cascade construction (CPU, once per frame)

Pure math, no GPU, no allocation — it lives in `shadowMath.ts` the way `cullMath.ts` mirrors the
cull kernels, and it is unit-tested headless (§10 U1–U5). It is also worker-safe: it consumes
matrices and numbers and produces a `Float32Array`.

### 5.1 Splits — practical split scheme

```ts
// Zhang et al.: blend logarithmic (good near) with uniform (good far) by lambda.
function cascadeSplits(near: number, shadowDistance: number, count: number, lambda: number): number[] {
    const splits: number[] = [];
    for (let i = 1; i <= count; i++) {
        const p   = i / count;
        const log = near * Math.pow(shadowDistance / near, p);
        const uni = near + (shadowDistance - near) * p;
        splits.push(lambda * log + (1 - lambda) * uni);
    }
    return splits;              // splits[count-1] === shadowDistance
}
```

`shadowDistance` — **not** `camera.far` — bounds the cascade set. A 1000-unit far plane with 4
cascades gives a useless first cascade; shadows stop at `shadowDistance` (default 200) and the
lighting falls back to unshadowed beyond it.

### 5.2 Split frustum corners

For split `[nearD, farD)`, the eight world-space corners come from the **main camera's**
`invViewProj` (already maintained, `Camera.ts:73`) by unprojecting the NDC box and re-scaling the
ray to the split distances — or, equivalently and with less numeric drift, directly from
`invView`, `fovRads`, and `aspect`:

```ts
const tanH = Math.tan(fovRads * 0.5), tanW = tanH * aspect;
for (const d of [nearD, farD])
    for (const sy of [-1, 1]) for (const sx of [-1, 1])
        corners.push(viewToWorld(invView, [sx * tanW * d, sy * tanH * d, -d]));
```

### 5.3 Fit a **sphere**, not a box

Fit a bounding *sphere* to the eight corners (centroid, then max radius). A box fitted to the
corners changes size as the camera rotates, and every size change re-quantizes the shadow map →
crawling edges. A sphere is rotation-invariant: rotate the camera in place and `radius` does not
move at all.

```ts
const center = mean(corners);
const radius = Math.ceil(max(dist(center, c) for c in corners) * 16) / 16;  // quantize to kill jitter
```

### 5.4 Caster extrusion — the conservative near plane

An object *behind* the cascade (between the light and the visible region) casts into it but is
outside a tightly-fitted ortho box. Culling it produces a shadow that pops in when its caster
enters view — the most common CSM bug, and one that reads as "the shadow is broken," not "the
culling is wrong."

So the ortho near plane is pushed back to the far side of the scene along the light direction:

```
eye        = center - lightDir * (radius + extrusion)
extrusion  = max(0, sceneBoundsExtentAlongLight)   // caller-supplied; default = shadowDistance
ortho      = mat4.orthoZO(out, -radius, radius, -radius, radius, 0, 2 * radius + extrusion)
```

The cost is depth range, and `depth24plus` has 24 bits of it — a 400-unit extruded range still
resolves ~2.4e-5 units. Cheap insurance. `ShadowConfig.casterExtrusion` overrides it for callers
who know their scene bounds.

### 5.5 Texel snapping, and the virtual fov

Snap the ortho centre to whole shadow texels **in light space**, or the cascade shimmers as the
camera translates:

```ts
const texelWorld = (2 * radius) / resolution;
const lightView  = lookAt([0, 0, 0], lightDir, up);   // FIXED in the world — see below
let  lightSpaceC = transform(lightView, center);
lightSpaceC[0] = Math.floor(lightSpaceC[0] / texelWorld) * texelWorld;
lightSpaceC[1] = Math.floor(lightSpaceC[1] / texelWorld) * texelWorld;
center = transform(invert(lightView), lightSpaceC);
```

**`lightView` must not move with the cascade.** The snap grid is the light view's texel grid, so
its origin has to be fixed in the world (the world origin is the simplest) while the cascade slides
across it. The tempting alternative — eye at the cascade `center` — maps `center` to `(0, 0)` every
frame, so the snap silently does nothing. That shipped: U3 measured it in the same moving frame and
passed tautologically, and the first real-device run (`test/shadows-gpu` T4) measured a 0.31-texel
slide per degree of yaw. Fixed in Cycle A ([`handoff.md`](./handoff.md)).

`texelWorld` also goes into `ShadowCascade.texelWorld`, where it sets the caster's normal offset
(§7.2) and the receiver's filter radius (§8.3) — one number, three consumers, so they can never
disagree.

**The virtual fov.** `selectLodIndex` computes `pxPerUnit = screenSize.y / (2·dist·tan(fov/2))`
(`visibilityShaders.ts:319`) — a perspective formula, meaningless for an ortho view. Rather than
fork the cull shader, pick the `fovRads` that reproduces the cascade's true texel density at its
centre, where `dist = radius + extrusion`:

```
fovRads = 2 · atan( radius / (radius + extrusion) )
```

LOD selection in the shadow view is then correct at the cascade centre and degrades smoothly
across its depth range — which is exactly the accuracy shadows need. `ShadowConfig.lodBias`
coarsens it further (shadow casters rarely need the finest LOD); `lodOverride` on the instance is
the hard escape hatch.

---

## 6. The shadow pass

### 6.1 Shape

One `RenderPassNode` per cascade, no color attachments, depth cleared to 1.0:

```ts
const pass = engine.createPass({
    name: `Shadow Cascade ${i}`,
    isMainScreenPass: false,
    // colorAttachments omitted → RenderPass.ts:24 defaults it to []
    depthStencilAttachment: {
        view: cascadeArray.createView({ dimension: '2d', baseArrayLayer: i, arrayLayerCount: 1 }),
        depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store',
    },
});
```

### 6.2 The depth-only pipeline

Each cascade × caster-mesh pair is one indirect batch, created exactly like a
`VisibilitySystem` instance-tier batch but with `depthOnly: true` and the cascade's camera:

```ts
const batch = engine.createBatch(pass, {
    shaderCode: buildShadowCasterWGSL({ ... }),   // vertex stage only
    strideFloats: INSTANCE_FLOATS,
    maxInstances: config.maxInstances,
    isIndirect: true,
    sharedSourceBuffer: vis.instanceBuffer,
    vertexLayouts: [StandardLayout.descriptor],
    depthOnly: true,                 // §12.1 — no fragment stage, no color targets
    cameraBuffer: cascade.buffer,    // §12.1 — this cascade's 432-byte Camera
    depthWriteEnabled: true,
    depthCompare: 'less',
});
```

Draw-call count is `cascades × casterMeshes`, **independent of instance count** — the same property
the main pass has (`visibility-design.md` §7.1 T2). A 4-cascade scene with 6 caster meshes and
200 000 instances is 24 draws.

### 6.3 Zero-change fallback: the dummy color target

If `depthOnly` is not adopted, the pass gains one `r8unorm` color attachment at shadow resolution
and the shader a one-line fragment that returns a constant. It is correct and it ships today. The
cost is honest and worth stating: `resolution² × cascades` bytes of pointless bandwidth (4 MB at
4×2048²) plus a full-rate fragment stage on every caster triangle — typically **1.5–2× the pass
cost**, in the pass most likely to be geometry-bound. Ship the fallback only if §12.1 is rejected.

### 6.4 Zero-change fallback: the cascade matrix in group 1

If `cameraBuffer` is not adopted, the cascade's `viewProj` moves to a group-1 uniform and the
caster VS uses that instead of `camera.viewProj`. The catch is `layout:'auto'`: group 0 binding 0
is bound unconditionally (`BatchManager.ts:94-102`) and would be pruned, making `createBindGroup`
throw. The shader must then keep a **named, deliberate** reference to the global camera, e.g.
carrying `camera.simTime` into an unused output — which is the kind of load-bearing weirdness that
gets "cleaned up" by the next reader and breaks the build. Prefer §12.1.

---

## 7. Bias — all of it on the caster

`PipelineConfig` exposes neither `depthBias` nor `cullMode`, so neither the classic slope-scaled
depth bias nor the front-face-culling trick is reachable. Both replacements live in the caster's
vertex shader, where the surface normal is in scope and control flow is uniform.

### 7.1 Constant bias in clip space

```wgsl
out.clipPos = camera.viewProj * world;
out.clipPos.z = out.clipPos.z + shadow.constantBias * out.clipPos.w;   // push casters away from the light
```

Multiplying by `w` applies the bias in NDC-z uniformly; for an orthographic cascade `w == 1`, so
`constantBias` is a plain NDC-z offset. Default `0.0015`.

### 7.2 Normal offset on the caster

```wgsl
let n     = normalize(normalMatrixFromModel(inst.model) * normal);
let world = inst.model * vec4<f32>(pos, 1.0);
let offset = n * (shadow.texelWorld * shadow.normalOffsetTexels);
out.clipPos = camera.viewProj * vec4<f32>(world.xyz + offset, 1.0);
```

Offsetting along the normal, scaled by the cascade's **world texel size**, makes the bias
resolution-correct: raise the shadow map to 4096 and acne does not return with a different
constant. Default `normalOffsetTexels = 1.5`.

### 7.3 The tradeoff, stated

Caster-side offset trades acne for **peter-panning**: contact shadows detach by roughly
`texelWorld × normalOffsetTexels`. At 2048 over a 40-unit cascade that is ~3 cm — invisible. At
512 over 200 units it is ~1.2 m — obvious. The pairing is not incidental: **the first cascade's
world extent is what sets the contact-shadow quality**, which is the real argument for cascades
over one big map. `§11` errors when the computed offset exceeds a configurable fraction of the
cascade extent, rather than letting it silently look broken.

A hand-written material fragment that has `N` in scope may additionally call
`shadowFactorBiased(worldPos, N, L)` (§8.4) for receiver-side normal offset and drop
`normalOffsetTexels` to 0. The stock material bridge cannot (§8.2).

---

## 8. Sampling and the material bridge

### 8.1 The decorator

`LightSystem.getShaderLayout()` returns `{ wgslStructs, wgslLogic }` and materials call the
`getIncidentLight(i, worldPos)` it defines. `ShadowSystem.wrapLighting(lightSystem)` returns a
layout of the same shape whose `getIncidentLight` calls the original and multiplies the shadow term
into `attenuation`:

```ts
wrapLighting(lights: LightSystem): ShaderLightingLayout {
    const inner = lights.getShaderLayout();
    if (!inner) throw new Error('ShadowSystem: wrapLighting requires a LightSystem with a shader layout ("standard" or "clustered-forward").');
    const marker = 'fn getIncidentLight(';
    if (!inner.wgslLogic.includes(marker))
        throw new Error('ShadowSystem: LightSystem WGSL no longer defines getIncidentLight(); the shadow decorator is out of date with LightSystem.getShaderLayout().');
    return {
        wgslStructs: inner.wgslStructs + this.bindingsWGSL(),         // shadow group, default 3
        wgslLogic:   inner.wgslLogic.replace(marker, 'fn getIncidentLightUnshadowed(')
                   + this.decoratorWGSL(lights.config.technique),
    };
}
```

One deterministic textual rename, guarded by a loud error if the marker ever moves. That guard is
the whole safety story — §10 U6 pins it against the real `LightSystem` output, so a rename in
`LightSystem.ts` fails a headless test rather than silently shipping unshadowed lighting.

The decorator is emitted **into the same module** as the light bindings, so it can read the light
record directly and decide whether that light casts:

```wgsl
fn getIncidentLight(i: u32, worldPos: vec3<f32>) -> IncidentLight {
    var inc = getIncidentLightUnshadowed(i, worldPos);
    let rec = ${accessor};                  // standard: lights[i]
                                            // clustered: globalLights[lightIndices[getClusterData(worldPos).start_offset + i]]
    if (rec.pos_type.w == 2.0 && dot(normalize(rec.dir_rad.xyz), shadow.lightDir.xyz) > 0.999) {
        inc.attenuation = inc.attenuation * sampleShadow(worldPos);
    }
    return inc;
}
```

Phase 1 shadows exactly one directional light, identified by direction match (two directional
lights pointing the same way *are* the same light). Phase 2 replaces the dot product with an
explicit per-light shadow slot carried in the light record's unused `spot_params.z`
(`LightSystem.ts:123` — `spot_params.zw` are free today), which is why the accessor exists here
rather than a direction-only test.

### 8.2 Why the receiver has no normal

The material contract is `getIncidentLight(i, worldPos)` — no normal. Recovering the geometric
normal with `dpdx/dpdy(worldPos)` is illegal here: the call sits inside `for (i < lightCount)` where
`lightCount` comes from a cluster lookup and is **non-uniform across the quad**, which WGSL's
`derivative_uniformity` diagnostic rejects. That is the constraint that pushed all bias onto the
caster (§7), and it is a fact about the shading loop, not an oversight.

### 8.3 Cascade selection, filtering, and the fade band

```wgsl
fn sampleShadow(worldPos: vec3<f32>) -> f32 {
    let viewDepth = -(camera.view * vec4<f32>(worldPos, 1.0)).z;

    var c = shadow.cascadeCount;                       // sentinel: beyond the last split
    for (var k = 0u; k < shadow.cascadeCount; k = k + 1u) {
        if (viewDepth < shadow.cascades[k].splitFar) { c = k; break; }
    }
    if (c >= shadow.cascadeCount) { return 1.0; }      // past shadowDistance → fully lit

    let lit = sampleCascade(c, worldPos);

    // Cross-fade into the next cascade over the last `blendTexels` of this one, so the
    // resolution change is a gradient rather than a visible seam.
    let edge = shadow.cascades[c].splitFar - shadow.blendTexels;
    if (c + 1u < shadow.cascadeCount && viewDepth > edge) {
        let t = (viewDepth - edge) / shadow.blendTexels;
        return mix(lit, sampleCascade(c + 1u, worldPos), clamp(t, 0.0, 1.0));
    }
    return lit;
}

fn sampleCascade(k: u32, worldPos: vec3<f32>) -> f32 {
    let clip = shadow.cascades[k].viewProj * vec4<f32>(worldPos, 1.0);
    let ndc  = clip.xyz / clip.w;
    if (ndc.z > 1.0) { return 1.0; }                                   // beyond the cascade far plane
    let uv   = vec2<f32>(ndc.x, -ndc.y) * 0.5 + vec2<f32>(0.5);        // clip +Y up → texture +V down
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) { return 1.0; }

    let ref   = ndc.z - shadow.constantBias;
    let texel = 1.0 / f32(textureDimensions(shadowMaps).x);
    let r     = i32(shadow.pcfRadius);
    if (r == 0) { return textureSampleCompareLevel(shadowMaps, shadowSampler, uv, k, ref); }

    var sum = 0.0;
    var n   = 0.0;
    for (var y = -r; y <= r; y = y + 1) {
        for (var x = -r; x <= r; x = x + 1) {
            let o = vec2<f32>(f32(x), f32(y)) * texel;
            sum = sum + textureSampleCompareLevel(shadowMaps, shadowSampler, uv + o, k, ref);
            n   = n + 1.0;
        }
    }
    return sum / n;
}
```

**`textureSampleCompareLevel`, not `textureSampleCompare`.** The latter takes implicit derivatives
and is only valid in uniform control flow; every call above sits inside a cascade branch and a PCF
loop. Using `textureSampleCompare` here is a shader-compile error at best and a
derivative-uniformity trap at worst — this is the shadow analogue of `visibility-design.md` §6.4's
flipped comparison, and §10 U7 pins it by asserting the generated WGSL never emits the bare form.

Each tap is hardware 2×4-comparison-filtered by the comparison sampler, so `pcfRadius: 1` is a 3×3
grid of bilinear-PCF taps ≈ a 6×6 kernel. Default `pcfRadius: 1`.

### 8.4 Bindings

```wgsl
@group(3) @binding(0) var shadowMaps    : texture_depth_2d_array;
@group(3) @binding(1) var shadowSampler : sampler_comparison;
@group(3) @binding(2) var<uniform> shadow : ShadowData;
```

Group 3 by default: group 0 is engine-fixed, group 1 is the material on the instance tier, group 2
is hardcoded into `LightSystem`'s WGSL (`LightSystem.ts:134,250`). `shadowGroupIndex` is
configurable for the meshlet tier, where group 1 already holds geometry. Bound with
`engine.setRenderPassBindGroup(batch, entries, shadowGroupIndex)`.

---

## 9. Public API

### 9.1 Construction

```ts
import { ShadowSystem } from 'null-graph/shadows';

const shadows = new ShadowSystem(engine, {
    cascades: 4,
    resolution: 2048,
    shadowDistance: 200,
    lambda: 0.6,              // split blend: 0 = uniform, 1 = logarithmic
    casterExtrusion: 200,     // §5.4; default = shadowDistance
    constantBias: 0.0015,
    normalOffsetTexels: 1.5,
    pcfRadius: 1,             // 1 → 3×3 taps of hardware PCF
    blendTexels: 4.0,
    lodBias: 1,               // casters render one LOD coarser
    visibility: vis,          // the VisibilitySystem whose instances are the casters
    shadowGroupIndex: 3,
});
```

### 9.2 Per frame

```ts
shadows.setLight([-0.5, -1.0, -0.3]);   // travel direction; normalized internally
shadows.update(camera);                 // splits, sphere fit, snap, write N cascade buffers + ShadowData
vis.upload();

engine.render([
    ...vis.getComputeStages(),
    ...shadows.getComputeStages(),
    ...shadows.getPasses(),
    scenePass,
    ...post.getPassSequence(),
]);
```

`update()` writes `cascades × 432 + 352` bytes from persistent typed arrays. No allocation, and it
takes only matrices and scalars — a worker can compute the cascade array and hand it over with
`shadows.setRawCascades(f32)`, the `LightSystem.setRawData` contract (`LightSystem.ts:92`).

### 9.3 Registering casters — and the two VisibilitySystem additions

```ts
shadows.registerCaster('rock', 0);      // meshName, meshIndex — mirrors createInstanceDrawBatch
```

Internally this is `cascades` calls into the `VisibilitySystem`, each needing a different camera and
no Hi-Z. Both are additive options on the existing draw-batch paths, defaulting to today's behavior:

```ts
interface InstanceDrawOptions {           // and MeshletDrawOptions
    /** Camera uniform for BOTH the cull stage and the draw. Default: engine.ctx.cameraUniformBuffer. */
    camera?: GPUBuffer;
    /** Override the system-wide Hi-Z setting for this batch. A shadow view has no pyramid. */
    occlusion?: boolean;
}
```

`VisibilitySystem` already builds a per-stage shader variant from `{ occlusion }`
(`VisibilitySystem.ts:239,313,406`) and binds the camera per stage (`:244,324,411`), so both are a
few lines each. `camera` on the *draw* side is what needs `PipelineConfig.cameraBuffer` (§12.1).

### 9.4 Receiver wiring

```ts
const shadowedLighting = shadows.wrapLighting(lights);

vis.createInstanceDrawBatch(scenePass, 'rock', 0, {
    material: { kind: 'pbr', instance: pbr, lighting: shadowedLighting },
});
shadows.applyToBatch(sceneBatch);      // binds shadowMaps + sampler + ShadowData at group 3
```

---

## 10. Test suite

Cascade math is pure logic and runs headless (`npm run test:shadows`), the way `cullMath` does.
GPU behavior is asserted numerically, never by eye, in `test/shadows-gpu` — run on a real device by
`npm run test:gpu` (Node + Dawn) or `npm run test:gpu:browser` (headless Chrome).

**Pure-logic unit tests (no GPU):**

- **U1 — splits.** `cascadeSplits(0.1, 200, 4, λ)` is strictly increasing, ends exactly at 200, and
  at `λ=0` matches the uniform series, at `λ=1` the logarithmic one, to 1e-6.
- **U2 — sphere fit is rotation-invariant.** Rotate the camera 360° in 1° steps at a fixed position;
  every cascade's `radius` varies by < 1e-4. A box fit fails this by ~40%.
- **U3 — texel snapping.** Translate the camera in small steps; the cascade centre, measured in ONE
  light frame fixed in the world, moves only in whole `texelWorld` increments — and it must move at
  least once, or the check is vacuous. (Measuring each centre in a frame built from that centre
  makes every delta 0, which passes for any fit.) The device half is `test/shadows-gpu` T4.
- **U4 — ZO round-trip.** A point at the cascade centre maps to `z ≈ 0.5` and the eight split
  corners all land in `z ∈ [0,1]`, `xy ∈ [-1,1]`. Swapping `orthoZO` for `ortho` (the NO form)
  fails this — the same trap `Camera.ts:62-66` documents for the perspective case.
- **U5 — caster extrusion.** An object placed `extrusion × 0.9` behind the cascade centre along the
  light direction passes `sphereInFrustum` against the cascade's planes; at `extrusion × 1.1` it
  does not. Set `casterExtrusion: 0` and the first case fails — the pop-in bug, caught by a number.
- **U6 — the decorator rename.** `wrapLighting` against a real `LightSystem` (both `standard` and
  `clustered-forward`) produces WGSL that defines `getIncidentLightUnshadowed` exactly once,
  defines `getIncidentLight` exactly once, and references `shadow.cascades`. Feed it a layout with
  the marker removed → it throws the §11 error.
- **U7 — no bare `textureSampleCompare`.** The generated receiver WGSL contains
  `textureSampleCompareLevel` and never `textureSampleCompare(` (§8.3).
- **U8 — cascade buffer packing.** A cascade's 108-float array has `viewProj` at 0, `view` at 16,
  `proj` at 32, `eyePos` at 96, `zNear/zFar/fovRads/aspect` at 100–103, matching `Camera.ts`.
  Byte-compare against a `Camera` instance configured to the same transform.

**Integration tests (GPU, one readback each).** `depth24plus` is not buffer-copyable, so readback
runs a small compute pass that `textureLoad`s the cascade layer into an `f32` storage buffer —
legal in a test, forbidden in the frame loop.

- **T1 — the shadow exists.** A unit sphere above a ground plane, sun overhead. Sample the lit
  image directly under the sphere and 5 units to the side. **Expect** the under-sphere sample
  darker by ≥ 60%. Disable shadows → the two match within 2%.
- **T2 — caster draw-call independence.** 10 000 instances of one caster mesh, 4 cascades.
  **Expect** exactly 4 `drawIndexedIndirect` calls and `sum(indirectBuffer[1]) > 0` for each.
  Raise to 100 000 → still 4.
- **T3 — no acne.** A large flat plane lit at 15° grazing incidence, no occluders. **Expect** zero
  pixels with `shadowFactor < 0.99`. Set `constantBias: 0` and `normalOffsetTexels: 0` → the test
  fails with a visible band, which is what makes it a regression test rather than a smoke test.
- **T4 — no peter-panning.** The sphere from T1 resting *on* the plane. **Expect** the shadow's
  boundary within `2 × texelWorld` of the contact point.
- **T5 — cascade seam.** Camera looking down the split between cascades 0 and 1 at a uniform
  surface. **Expect** no sample pair straddling `splitFar` differing by more than 20%. Set
  `blendTexels: 0` → the seam appears, confirming the fade band is what removes it.
- **T6 — caster outside the view frustum.** A tall pillar behind the camera whose shadow falls into
  view. **Expect** the shadow present. This is the §5.4 test in the rendered image; it is the one
  most likely to regress when someone "optimizes" the cascade fit.
- **T7 — beyond shadowDistance.** Geometry at `shadowDistance × 1.5` is fully lit, with no hard
  boundary artifact at the transition.

---

## 11. Errors

Build/compile time, naming the offender. No silent fallback.

| Condition | Message |
|---|---|
| `cascades` outside 1–4 | `ShadowSystem: cascades must be between 1 and 4 (got <n>); MAX_CASCADES sizes the ShadowData uniform.` |
| `resolution` not a power of two | `ShadowSystem: resolution <n> must be a power of two; texel snapping (§5.5) assumes it.` |
| `wrapLighting` on a technique with no layout | `ShadowSystem: wrapLighting requires a LightSystem with technique "standard" or "clustered-forward"; "<t>" returns no shader layout.` |
| `getIncidentLight` marker missing | `ShadowSystem: LightSystem WGSL no longer defines getIncidentLight(); the shadow decorator is out of date with LightSystem.getShaderLayout().` |
| `shadowGroupIndex` collides with the light group | `ShadowSystem: shadowGroupIndex 2 collides with LightSystem's hardcoded @group(2); use 3 or higher.` |
| normal offset swamps the cascade | `ShadowSystem: cascade <k> normal offset (<o> world units) exceeds 2% of its extent (<e>); raise resolution or lower normalOffsetTexels — contact shadows will detach.` |
| `getPasses()` before `update()` | `ShadowSystem: getPasses() called before update(camera); cascade matrices are unset.` |
| `registerCaster` without a VisibilitySystem | `ShadowSystem: registerCaster requires config.visibility; Phase 1 has no CPU caster path.` |
| depth-only unsupported and no fallback opted into | `ShadowSystem: depth-only pipelines require PipelineConfig.depthOnly (shadows-design.md §12.1); set allowColorFallback:true to render with a throwaway color target instead.` |

---

## 12. Engine touchpoints

### 12.1 Required — two additive `PipelineConfig` fields

Phase 1 is the first NullGraph module that asks for a core change. Both are additive, opt-in, and
inert for every existing caller.

1. **`depthOnly?: boolean`** — when set, `createBatch` omits the `fragment` stage and the color
   targets entirely. A fragment-less render pipeline is core WebGPU. ~6 lines in
   `BatchManager.ts:37-75`. Without it, §6.3's dummy color target is the fallback, at ~1.5–2× pass
   cost.

```ts
// BatchManager.createBatch
batch.pipeline = this.ctx.device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: shaderModule, entryPoint: 'vs_main', buffers: config.vertexLayouts || [] },
    ...(config.depthOnly ? {} : {
        fragment: { module: shaderModule, entryPoint: 'fs_main', targets: colorTargets }
    }),
    ...
});
```

2. **`cameraBuffer?: GPUBuffer`** — the group-0 binding-0 resource, defaulting to
   `ctx.cameraUniformBuffer`. **One line** at `BatchManager.ts:100`:

```ts
{ binding: 0, resource: { buffer: config.cameraBuffer || this.ctx.cameraUniformBuffer } }
```

This is the change that makes "a cascade is a camera" true end-to-end, and it is the enabling
change for every future off-main-view render: shadows, reflection probes, IBL capture (roadmap
§5.5), and any editor picking pass. Without it, §6.4's fallback works but leaves a fragile
anti-prune reference in generated WGSL.

### 12.2 Optional — filed, not needed

- **`depthBias` / `depthBiasSlopeScale` / `depthBiasClamp`** on `PipelineConfig`. Would let bias
  move back to rasterizer state; §7 works without it, and the caster-side offset is arguably better
  than constant depth bias anyway. **Cosmetic.**
- **`cullMode`** on `PipelineConfig`. Front-face culling in the shadow pass is a classic acne
  reduction and would halve caster fragment work. Blocked only by `primitive:` being hardcoded to
  `{ topology }` (`BatchManager.ts:63-65`). **Efficiency.**
- **`depthFormat`** on `PipelineConfig`. `depth32float` shadow maps are copyable (simpler GPU
  tests) and slightly more precise across the extruded range. `depth24plus` is sufficient.
  **Cosmetic.**

### 12.3 Module changes (not core)

`VisibilitySystem` gains `camera?: GPUBuffer` and `occlusion?: boolean` per draw batch (§9.3).
Both default to current behavior; no existing call site changes.

---

## 13. Build order

Each step renders correctly at its end.

1. **Cascade math (§5)** — splits, sphere fit, snap, extrusion, the 108-float packing. Pure, GPU-free.
   Assert U1–U5, U8. Nothing renders yet and that is fine: this is where CSM is usually wrong.
2. **One cascade, no culling (§6)** — a single depth-only pass drawing every caster instance with
   the existing non-indirect batch path. Assert T1 with `cascades: 1`. Proves the matrix, the
   attachment, the array layer, and the `depthOnly` change in isolation.
3. **Receiver sampling (§8.3–8.4)** — hardware 2×2 PCF, single cascade, hand-written fragment.
   Assert T1, T3, T4. Get `textureSampleCompareLevel` and the Y flip right here.
4. **The lighting decorator (§8.1)** — `wrapLighting`, so stock materials shadow. Assert U6, U7.
5. **N cascades + selection + fade (§5.1, §8.3)** — Assert T5, T7.
6. **GPU caster culling (§9.3)** — the `camera` / `occlusion` options and one indirect batch per
   cascade × mesh. Assert T2, T6. Everything before this renders correct shadows; this step makes
   them scale.
7. **PCF radius, LOD bias, tuning (§7.3, §5.5)** — the quality knobs, last, so any regression is
   isolated to a parameter rather than the structure.

Step 3 is the first frame with a real shadow in it. Steps 1–3 are the ones worth being slow about.

### Deferred

To **Phase 2** ([`shadows-atlas-design.md`](./shadows-atlas-design.md)): the atlas, spot and point
shadows, per-tile caching, Hi-Z occlusion on the light view, and the explicit per-light shadow slot
that replaces §8.1's direction match.

To **v2.0**: virtual shadow maps (roadmap §2), which are to this document what virtual geometry is
to `geometry-meshlets-design.md` — the same idea with residency and streaming, and only sane once
the atlas (Phase 2) exists to be virtualized.
