# Visibility — Shader & Buffer Reference

Companion to [`visibility.md`](./visibility.md) (usage) and
[`visibility-design.md`](../visibility-design.md) (architecture). This is the **contract layer**:
exact byte layouts, WGSL bindings, the generated kernels, and how to plug a material into a
GPU-culled draw.

Read this when you are writing a custom draw shader, debugging a bind-group error, filling
instance data from a worker, or wiring a built-in material into a visibility batch.

---

## 1. Buffer layouts

Every structure is 16-byte aligned, `vec4`-shaped, and worker-fillable. Sizes are exact.

### 1.1 `Instance` — 96 bytes (24 floats)

The per-frame input you fill. Written via `InstanceBuffer.write()`, or by a Web Worker writing
the raw arrays directly.

| Float | Bytes | Field | Type | Meaning |
|---|---|---|---|---|
| 0–15 | 0–63 | `model` | `mat4x4<f32>` | **Column-major** object→world matrix (gl-matrix `mat4`). |
| 16–19 | 64–79 | `boundsCenter` | `vec4<f32>` | Object-space bounding sphere: `.xyz` center, `.w` radius. |
| 20 | 80 | `meshIndex` | `u32` | Index into `meshTable`. Instance tier also uses it to route instances to the matching draw batch. |
| 21 | 84 | `lodOverride` | `u32` | Relative LOD level (0 = finest), or `LOD_AUTO` (`0xFFFFFFFF`). |
| 22 | 88 | `flags` | `u32` | Bit 0 `INSTANCE_FLAG_NO_OCCLUSION`, bit 1 `INSTANCE_FLAG_NO_CONE_CULL`. |
| 23 | 92 | `_pad` | `u32` | Must be 0. |

The integer fields occupy float slots but hold **raw u32 bits** — write them through a
`Uint32Array` view over the same `ArrayBuffer`, which is exactly what `InstanceBuffer` does. Writing
`20.0` as a float into slot 20 does *not* mean `meshIndex == 20`.

```ts
// Worker-side, no engine import needed — just the layout.
const INSTANCE_FLOATS = 24;
const f32 = new Float32Array(sab);          // shared with the main thread
const u32 = new Uint32Array(sab);           // SAME memory, integer view
const base = i * INSTANCE_FLOATS;
f32.set(modelMatrix, base);                 // 16 floats, column-major
f32[base + 16] = cx; f32[base + 17] = cy; f32[base + 18] = cz; f32[base + 19] = radius;
u32[base + 20] = meshIndex;
u32[base + 21] = 0xFFFFFFFF;                // LOD_AUTO
u32[base + 22] = 0;                         // flags
u32[base + 23] = 0;
```

### 1.2 `Meshlet` — 64 bytes

Produced by `MeshletBuilder`; read-only at runtime.

| Field | Type | Meaning |
|---|---|---|
| `boundsCenter` | `vec4<f32>` | Object-space sphere (`.xyz` center, `.w` radius) for frustum + Hi-Z. |
| `coneApex` | `vec4<f32>` | `.xyz` cone apex (the sphere center in v1). `.w` unused. |
| `coneAxis` | `vec4<f32>` | `.xyz` normalized axis, `.w` = **cutoff**. `cutoff >= 2.0` is the "never cull" sentinel. |
| `offsets` | `vec4<u32>` | `.x` vertexOffset, `.y` triangleOffset, `.z` vertexCount, `.w` triangleCount. |

### 1.3 `MeshEntry` / `MeshLOD` — 16 bytes each

| `MeshEntry` | Type | | `MeshLOD` | Type |
|---|---|---|---|---|
| `lodOffset` | `u32` | | `meshletOffset` | `u32` |
| `lodCount` | `u32` | | `meshletCount` | `u32` |
| `_pad0`, `_pad1` | `u32` | | `error` | `f32` (world-space geometric error) |
| | | | `_pad` | `f32` |

### 1.4 `VisibleInstance` — 16 bytes (internal)

Meshlet-tier intermediate, written by pass 1 and read by pass 2:
`{ instanceId: u32, lodMeshletOffset: u32, lodMeshletCount: u32, _pad: u32 }`.

### 1.5 `CullParams` — 32 bytes (uniform, per cull stage)

Refreshed every frame by `upload()`.

| Field | Type | Source |
|---|---|---|
| `instanceCount` | `u32` | `setInstances(..., count)` or `setInstanceCount(n)` |
| `meshIndex` | `u32` | the batch's mesh (instance tier); 0 for meshlet tier |
| `hizMaxLevel` | `u32` | `HiZPyramid.maxLevel` |
| `flags` | `u32` | reserved (0) |
| `screenSize` | `vec2<f32>` | `hiZ.width/height`, else canvas size |
| `lodErrorPx` | `f32` | `config.lodErrorPx` (default 1.0) |
| `_pad` | `f32` | 0 |

### 1.6 Indirect draw args — 20 bytes (5 × u32)

Engine-allocated (`BatchManager.ts:61-64`). The layout `drawIndexedIndirect` consumes:

```
[0] indexCount      written once by VisibilitySystem at batch creation
[1] instanceCount   ZEROED each frame via preClearBuffers, then atomically bumped by the cull
[2] firstIndex      per-mesh (instance tier) / 0 (meshlet tier)
[3] baseVertex      per-mesh (instance tier) / 0 (meshlet tier)
[4] firstInstance   0
```

Slot `[1]` is the whole GPU-driven trick: the cull kernel's `atomicAdd(&drawArgs[1], 1u)` both
allocates a compaction slot **and** grows the draw. No CPU readback.

---

## 2. Bind-group contracts

`layout: 'auto'` prunes any binding the shader does not *use*, and supplying an entry for a pruned
binding throws. The generated kernels therefore declare a binding **only** when the corresponding
feature is enabled — which is why occlusion-off variants omit `params`/`hiz` entirely.

### 2.1 Cull kernels (group 0)

| Binding | Instance cull | Meshlet instance cull (pass 1) | Meshlet cull (pass 2) |
|---|---|---|---|
| 0 | camera | camera | camera |
| 1 | instances (read) | instances (read) | instances (read) |
| 2 | outInstances (rw) | visibleInstances (rw) | visibleInstances (read) |
| 3 | drawArgs (atomic) | counters (atomic) | meshlets (read) |
| 4 | params | meshTable (read) | counters (atomic) |
| 5 | *hiz* † | meshLODs (read) | outMeshlets (rw) |
| 6 | — | params | drawArgs (atomic) |
| 7 | — | *hiz* † | *params* † |
| 8 | — | — | *hiz* † |

† emitted only when Hi-Z occlusion is enabled. Note pass 2 declares **`params` only under
occlusion** — nothing else in that kernel reads it.

### 2.2 Draw shaders

**Instance tier** — group 0 is fixed by `createBatch` (`BatchManager.ts:73-79`):

| Group | Binding | Resource |
|---|---|---|
| 0 | 0 | `camera` (uniform) |
| 0 | 1 | `instances` — the **compacted survivors** (`batch.storageBuffer`) |
| 1+ | — | your material (`materialEntries`, default group 1) |
| 2 | — | lighting, if you call `lights.applyToBatch(batch, 2)` |

**Meshlet tier** — group 1 is occupied by geometry, so materials move up:

| Group | Binding | Resource |
|---|---|---|
| 0 | 0 | `camera` |
| 0 | 1 | `visible: array<vec2<u32>>` — `{instanceId, meshletId}` (`batch.storageBuffer`) |
| 1 | 0–4 | `meshlets`, `meshletVertices`, `meshletTriangles`, `megaVertices`, `instances` |
| 2+ | — | your material (`materialGroupIndex`, default 2) |

> **Group 2 collision.** On the meshlet tier the material defaults to group 2, which is also where
> `LightSystem.applyToBatch(batch, 2)` binds. If you want both, pass an explicit
> `materialGroupIndex: 3` and bind lighting at 2. On the instance tier there is no collision:
> material at 1, lighting at 2, exactly as the stock material shaders expect.

---

## 3. The `VertexOut` contract

Both draw shaders emit a struct that is **field-identical to `materials/CommonChunks.ts`**:

```wgsl
struct VertexOut {
    @builtin(position) clipPos: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) worldNormal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) ecsColor: vec3<f32>,
};
```

This match is deliberate and load-bearing: it is what lets a stock material's fragment shader drop
into a visibility batch **unchanged** (§4). `ecsColor` is filled with white because the visibility
`Instance` record carries no per-instance color, so `material.baseColor * instanceColor` reduces to
`material.baseColor` — the correct no-tint result.

The normal is transformed by the inverse-transpose of the model's upper 3×3
(`normalMatrixFromModel`) and renormalized, so non-uniformly scaled instances shade correctly.

---

## 4. Using a built-in material with a culled batch

The default fragment is a neutral hemispheric shade — fine for bring-up, not for real rendering.

### 4.0 The one-liner (preferred)

Pass a stock material; the WGSL and the bind group are generated for you:

```ts
vis.createInstanceDrawBatch(scenePass, 'rock', rock.meshIndex, {
    vertexLayouts: StandardLayout.getWebGPUDescriptor(),
    material: { kind: 'pbr', instance: rockMat, lighting: lights.getShaderLayout() },
});
```

Under the hood this calls `buildMaterialFragment({ kind, groupIndex, lighting })` from
`materials/materialFragments.ts` — **the single source of truth** that the stock full-shader
builders (`buildPBRShader`, `buildLambertShader`, …) also compose from. One definition of each
material's bindings and `fs_main`, two consumers, so they cannot drift.

```ts
import { buildMaterialFragment } from 'null-graph/materials';

const { bindings, fragment } = buildMaterialFragment({ kind: 'lambert', groupIndex: 1 });
// bindings → struct + @group(1) declarations (+ helper math, + lighting when supplied)
// fragment → the material's @fragment fn fs_main
```

The rest of this section documents the **manual** route — what the one-liner generates. Use it
when you want a custom fragment, a non-stock material, or to understand the contract.

### 4.1 Instance tier — groups already line up

The stock material bindings sit at `@group(1)` and lighting at `@group(2)` — precisely the instance
tier's defaults. So the material's own WGSL needs no group edits:

```ts
import { StandardLayout } from 'null-graph/geometry';

const MATERIAL_BINDINGS = /* wgsl */ `
    struct LambertMaterialParams {
        baseColor: vec4<f32>,
        ambient: f32,
        hasAlbedoTexture: f32,
        _pad: vec2<f32>,
    };
    @group(1) @binding(0) var samp: sampler;
    @group(1) @binding(1) var texAlbedo: texture_2d<f32>;
    @group(1) @binding(2) var<uniform> material: LambertMaterialParams;
`;

const MATERIAL_FRAGMENT = /* wgsl */ `
@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    var baseColor = material.baseColor * vec4<f32>(in.ecsColor, 1.0);
    if (material.hasAlbedoTexture > 0.5) {
        baseColor = textureSample(texAlbedo, samp, in.uv) * baseColor;
    }
    let N = normalize(in.worldNormal);
    let L = normalize(vec3<f32>(1.0, 1.0, 0.5));
    let NdotL = max(dot(N, L), 0.0);
    let finalColor = material.ambient * baseColor.rgb + baseColor.rgb * NdotL;
    return vec4<f32>(finalColor, baseColor.a);
}`;

const batch = vis.createInstanceDrawBatch(scenePass, 'rock', rock.meshIndex, {
    vertexLayouts: StandardLayout.getWebGPUDescriptor(),
    extraBindings: MATERIAL_BINDINGS,      // declared in the generated module
    fragmentBody: MATERIAL_FRAGMENT,       // replaces the default fragment
    materialEntries: [                     // bound at group 1
        { binding: 0, resource: sampler },
        { binding: 1, resource: albedoView },
        { binding: 2, resource: { buffer: materialParamsBuffer } },
    ],
});
```

`extraBindings` and `fragmentBody` are two halves of one thing — the WGSL *declarations* and the
WGSL *body*. Supplying a fragment that references bindings you did not declare is a shader-creation
error, which fails loudly at batch creation.

### 4.2 Adding clustered lighting

The stock material fragments call `getVisibleLightCount()` / `getIncidentLight()`, which
`LightSystem.getShaderLayout()` emits. Concatenate them into `extraBindings`, and bind the light
buffers at group 2:

```ts
const layout = lights.getShaderLayout()!;   // clustered-forward or standard
const batch = vis.createInstanceDrawBatch(scenePass, 'rock', rock.meshIndex, {
    vertexLayouts: StandardLayout.getWebGPUDescriptor(),
    extraBindings: MATERIAL_BINDINGS + layout.wgslStructs + layout.wgslLogic,
    fragmentBody: LIT_FRAGMENT,             // the loop form, as in buildLambertShader
});
lights.applyToBatch(batch, 2);
```

### 4.3 Meshlet tier — shift the material group

Identical, except the material cannot use group 1 (geometry lives there). Pass
`materialGroupIndex: 2` (or 3 alongside lighting) and change `@group(1)` to the matching index in
`MATERIAL_BINDINGS`. This is the one place the stock material WGSL needs an edit.

### 4.4 How the two paths share one definition

```
                materials/materialFragments.ts
                  MaterialBindings[kind](group)
                  MaterialFragments[kind](config)
                  buildMaterialFragment(...)
                          │
            ┌─────────────┴──────────────┐
            ▼                            ▼
  buildPBRShader / buildLambertShader   VisibilitySystem
  (+ engine bindings, ECS vertex        (+ GPU-culled vertex stage reading
   stage reading the 14-float buffer)    a mat4 from the Instance record)
```

Both paths emit the **same** `VertexOut` (§3) and the **same** fragment stage; only the vertex
stage and the group index differ. `npm run test:visibility` asserts the `VertexOut` parity and that
every family composes into both tiers without a binding collision — so this cannot silently rot.

The remaining nuance is the meshlet tier's group 2 default (§4.3): pass `materialGroupIndex: 3` if
you want a material *and* clustered lighting on that tier.

---

## 5. Generated kernels

Every generator is exported, so you can print one and read it — the fastest way to debug a
bind-group mismatch.

| Function | Emits |
|---|---|
| `buildHiZMipWGSL(fromDepth)` | one Hi-Z mip reduction (max of 2×2) |
| `buildInstanceCullWGSL({occlusion})` | instance tier cull + compaction |
| `buildMeshletInstanceCullWGSL({occlusion})` | meshlet pass 1: coarse cull + LOD select |
| `buildMeshletCullWGSL({occlusion, cone})` | meshlet pass 2: per-cluster cull + compaction |
| `buildInstanceDrawWGSL(opts)` | instance tier draw (standard indexed-instanced) |
| `buildMeshletDrawWGSL(opts)` | meshlet tier draw (vertex pulling) |

Reusable chunks: `CAMERA_STRUCT_WGSL`, `INSTANCE_STRUCT_WGSL`, `MESHLET_STRUCT_WGSL`,
`MESH_TABLE_STRUCT_WGSL`, `CULL_PARAMS_WGSL`, `CULL_CORE_WGSL`, `HIZ_OCCLUSION_WGSL`.

```ts
import { buildInstanceCullWGSL } from 'null-graph/visibility';
console.log(buildInstanceCullWGSL({ occlusion: true }));
```

### 5.1 Dispatch shapes

| Stage | Workgroup | Dispatch |
|---|---|---|
| Hi-Z mip | `(8, 8, 1)` | `ceil(mipW/8) × ceil(mipH/8)` |
| Instance cull | `(64, 1, 1)` | `ceil(instanceCount/64)` — re-evaluated per frame |
| Meshlet pass 1 | `(64, 1, 1)` | `ceil(instanceCount/64)` |
| Meshlet pass 2 | `(8, 8, 1)` | `ceil(maxInstances/8) × ceil(maxMeshletsPerInstance/8)` |

Pass 2 dispatches over **CPU-known caps**, not the GPU-side visible count, and idle lanes early-out
against `counters[0]` / `lodMeshletCount`. That is the deliberate cost of never stalling on a
readback; indirect dispatch would remove it (design §12).

---

## 6. The cull math, and the signs that matter

Mirrored exactly between `cullMath.ts` (CPU, unit-tested) and `CULL_CORE_WGSL` (GPU). The CPU copy
exists so these can be pinned by numbers headlessly — run `npm run test:visibility`.

### 6.1 Frustum — the ZO near plane

NullGraph's camera uses `mat4.perspectiveZO`, so NDC z spans **[0, 1]**, not [-1, 1]:

```
L = r3 + r0    R = r3 - r0    B = r3 + r1    T = r3 - r1
N = r2         ← ZO form (clip z >= 0). NOT r3 + r2, which is the OpenGL [-1,1] form.
F = r3 - r2
```

Using the OpenGL near plane halves the near distance and culls geometry that is actually visible.

### 6.2 Normal cone

```wgsl
if (cutoff >= 2.0) { return false; }          // sentinel: too wide to ever cull
let d = normalize(apexW - eye);               // camera → apex, NOT the negation
return dot(normalize(axisW), d) >= cutoff;    // >=, not <=
```

Flip either the comparison or the direction and you cull exactly the **visible** clusters — holes
where geometry should be. Covered by test U4.

### 6.3 Hi-Z occlusion

```wgsl
return ndcNear > occ;   // sphere's nearest point is behind the farthest occluder
```

The pyramid stores the **max** (farthest) depth per 2×2 block, because the depth buffer holds the
*nearest* surface per pixel and an object is hidden only when it is behind the farthest of those.
No depth remap is needed: the pyramid and `ndcNear` come from the same ZO projection. A near-plane
guard keeps anything straddling the near plane visible.

---

## 7. Frame cost

The complete per-frame CPU work, with no allocation:

```ts
vis.setInstances(instances.data, instances.count);  // adopt by reference — no copy
vis.upload();                                       // 1 writeBuffer + params refresh per stage
engine.render([...vis.getComputeStages(), scenePass, ...post.getPassSequence()]);
```

`getComputeStages()` returns a **cached** array (rebuilt only when geometry or batches change).
`preClearBuffers` zeroes `drawArgs[1]` and `counters[0]` before each stage. Everything else —
plane extraction, culling, compaction, LOD — happens on the GPU.

---

## 8. Debugging checklist

| Symptom | Likely cause |
|---|---|
| Nothing renders | `getComputeStages()` not spliced **before** the scene pass; or `upload()` not called, so `instanceCount` is 0. |
| Everything renders (no culling) | Instances' `boundsCenter` radius is 0 or wrong — a zero-radius sphere fails the frustum test only when fully outside. Verify with `worldSphereFromModel`. |
| Holes / geometry vanishes when the camera turns | Cone sign inverted, or `coneAxis.w` not the `sin(α)` cutoff (geometry doc §6.2). |
| Everything vanishes once Hi-Z is on | `ndcNear > occ` flipped, or the depth source is not the *previous* frame's. |
| Bind-group creation throws | A declared-but-unused binding was pruned by `layout: 'auto'` — check the §2.1 table for the occlusion-off variant. |
| Meshlets flicker at distance | `maxVisibleMeshletInstances` too small; the compaction overflowed. Raise it. |
| Shading is black / inside-out | Custom `fragmentBody` reading `in.normal` — the field is **`worldNormal`** (§3). |
