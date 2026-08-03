# Core Reference

The low-level companion to the [Core API guide](./core.md). It documents the pieces you reach for
when writing custom shaders or wiring resources by hand: the **bind-group conventions** the engine
assumes, the **instance data layout**, the **`Camera`** uniform, **`ComputeStageConfig`** in full,
and the two resource managers (`BufferManager`, `TextureManager`) plus the shared context
(`engine.ctx`).

Read [core.md](./core.md) first for the high-level engine flow.

---

## 1. Bind-group conventions

Every shader NullGraph runs shares the same group layout. If you write custom WGSL, match it — the
engine binds these groups for you and a mismatch throws at bind-group creation.

| Group | Binding | Resource | Bound by |
|---|---|---|---|
| **0** | 0 | `camera` uniform (432-byte block) | `createBatch` (automatic) |
| **0** | 1 | instance **storage buffer** (`array<...>`, `strideFloats` per instance) | `createBatch` (automatic) |
| **1** | 0… | material params + textures/samplers | `createTextureBindGroup` / `setRenderPassBindGroup` / a `Material` |
| **2** | 0… | lights (and cluster grid/indices) | `LightSystem.applyToBatch` |

- **Group 0** is filled automatically for every batch: binding 0 is the global camera uniform,
  binding 1 is the batch's own instance storage buffer (what `updateBatchData` streams into).
- **Group 1** is the conventional "material" slot. The material classes and
  `createTextureBindGroup` default to group index 1.
- **Group 2** is the conventional "lights" slot (`LightSystem.applyToBatch` defaults to it).

Compute stages are independent: `setComputePassBindGroup` defaults to **group 0**, and you declare
whatever bindings your kernel needs.

---

## 2. Instance data layout

Each batch owns one pre-allocated storage buffer of `maxInstances * strideFloats * 4` bytes.
`updateBatchData(batch, data, count)` streams a flat `Float32Array` into it (Zero-Copy), and the
vertex shader indexes it by `@builtin(instance_index)`.

The stock shader builders assume a **14-float stride**:

| Float offset | Field | Notes |
|---|---|---|
| 0 | index / pad | free slot (the builders start reading at 1) |
| 1, 2, 3 | position | world XYZ |
| 4, 5, 6, 7 | rotation | quaternion XYZW |
| 8, 9, 10 | scale | XYZ |
| 11, 12, 13 | color | RGB |

```wgsl
// How the built-in vertex shaders read it:
let base = iIdx * 14u;
let ecsPos   = vec3<f32>(ecs[base + 1u], ecs[base + 2u], ecs[base + 3u]);
let ecsRot   = vec4<f32>(ecs[base + 4u], ecs[base + 5u], ecs[base + 6u], ecs[base + 7u]);
let ecsScale = vec3<f32>(ecs[base + 8u], ecs[base + 9u], ecs[base + 10u]);
```

```ts
// Populate one instance
const data = new Float32Array(14);
data[1] = x; data[2] = y; data[3] = z;   // position
data[7] = 1.0;                            // rotation quat W (identity)
data[8] = data[9] = data[10] = 1.0;       // scale
data[11] = 0.1; data[12] = 0.5; data[13] = 0.9; // color
engine.updateBatchData(batch, data, 1);
```

`strideFloats` is per-batch and fully custom — set it to whatever your own shader reads (post-process
passes use `8`, for example). The 14-float layout is only a convention the material builders adopt.

---

## 3. `Camera`

```ts
import { Camera } from 'null-graph';

const camera = new Camera(
    75,                              // fovDegrees
    canvas.width / canvas.height,    // aspect
    0.1,                             // near
    1000.0,                          // far
);
```

| Method | Description |
|---|---|
| `updateView(eye, target, up?)` | Recompute view + inverse-view and the combined matrices. `up` defaults to `[0,1,0]`. |
| `updateProjection(aspect)` | Recompute projection + inverse-projection (call on resize / aspect change). |
| `updateTime(simTime)` | Write a time value into the uniform's padding slot (float 99) — drives GPU animation (e.g. `EmissiveMaterial` pulse). |

Public fields include `fovDegrees`, `near`, `far`, the matrices, and `bufferData` (108 floats /
432 bytes — the contiguous block `engine.updateCamera` uploads).

### The Camera uniform (WGSL)

Custom shaders declare this struct at `@group(0) @binding(0)`. It matches `Camera.bufferData`
byte-for-byte:

```wgsl
struct Camera {
    viewProj:    mat4x4f,  // 0
    view:        mat4x4f,  // 16
    proj:        mat4x4f,  // 32
    invViewProj: mat4x4f,  // 48
    invView:     mat4x4f,  // 64
    invProj:     mat4x4f,  // 80
    eye:         vec3f,    // 96  (eye position)
    time:        f32,      // 99  (updateTime writes here)
    near:        f32,      // 100
    far:         f32,      // 101
    fovRads:     f32,      // 102
    aspect:      f32,      // 103
    _padEnd:     vec4f,    // 104–107 (reserved)
};
@group(0) @binding(0) var<uniform> camera: Camera;
```

> The inverse matrices exist for World-space reconstruction (SSR, decals, SSAO), billboarding, and
> clustered light culling. See [Deprecated API §3](../deprecated-api.md) for the migration note if you
> maintained a shader against the old 20-float camera.

### Frame usage

```ts
function frame() {
    const t = performance.now() * 0.001;
    camera.updateTime(t);
    camera.updateView([Math.sin(t) * 8, 3, Math.cos(t) * 8], [0, 0, 0]);
    engine.updateCamera(camera);   // uploads the 432-byte block
    engine.render([scenePass]);
    requestAnimationFrame(frame);
}

// On canvas resize:
engine.resize(w, h);
camera.updateProjection(w / h);
```

---

## 4. `ComputeStageConfig`

`engine.createComputeStage(config)` builds a standalone `ComputeStageNode`. The kernel's entry point
must be `cs_main`.

```ts
const stage = engine.createComputeStage({
    name: 'My Compute',
    shaderCode: myWGSL,                 // must define @compute fn cs_main(...)
    dispatchCount: [64, 1, 1],          // static, OR a function (see below)
    workgroupSize: [64, 1, 1],          // optional; usually declared in WGSL instead
    preClearBuffers: [
        { buffer: counterBuffer, offset: 0, data: new Uint32Array([0]) },
    ],
});
```

| Field | Type | Notes |
|---|---|---|
| `name` | `string` | Debug label. |
| `shaderCode` | `string` | WGSL; entry point `cs_main`. |
| `dispatchCount` | `[x,y,z]` **or** `() => [x,y,z]` | Workgroup counts. A function is re-evaluated each frame — use it for size-dependent dispatch (e.g. tied to canvas resolution). |
| `workgroupSize` | `[x,y,z]` | Optional; the `@workgroup_size(...)` in WGSL is the usual source of truth. |
| `preClearBuffers` | `BufferClearConfig[]` | Buffer resets run **immediately before** the dispatch, every frame. |

`BufferClearConfig` = `{ buffer: GPUBuffer, offset: number, data: Uint32Array | Float32Array }`.

The dynamic-dispatch form is why culling stages can react to a resize without recompiling:

```ts
dispatchCount: () => [Math.ceil(engine.width / 8), Math.ceil(engine.height / 8), 1],
```

Bind resources with `engine.setComputePassBindGroup(stage, entries, groupIndex = 0)`, then include
the stage in the `render(...)` sequence **before** whatever consumes its output. See
[Indirect Drawing](./core.md#indirect-drawing-gpu-compute-culling) and the
[Lights compute reference](./lights-shader-reference.md#4-rolling-your-own-compute-stage-over-the-light-buffers).

---

## 5. `BufferManager` (`engine.bufferManager`)

Thin helpers that create GPU buffers from typed arrays, mapped at creation (no staging copy).

```ts
const vbo = engine.bufferManager.createVertexBuffer(float32Vertices); // VERTEX | COPY_DST
const ibo = engine.bufferManager.createIndexBuffer(uint16OrUint32Indices); // INDEX | COPY_DST
engine.setBatchGeometry(batch, vbo, ibo, indices.length, 'uint16');
```

| Method | Returns | Notes |
|---|---|---|
| `createVertexBuffer(data: Float32Array)` | `GPUBuffer` | Vertex usage; sized to `data.byteLength`. |
| `createIndexBuffer(data: Uint16Array \| Uint32Array)` | `GPUBuffer` | Index usage; size rounded up to 4-byte alignment. Pass the matching `'uint16'`/`'uint32'` format to `setBatchGeometry`. |

(The [`Geometry`](./geometry.md) class wraps these behind `upload(engine)` — use the manager
directly when you have raw arrays, e.g. from `GeometryBuilder.build()` or a GLB parser.)

---

## 6. `TextureManager` (`engine.textureManager`)

Loads, decodes, caches, and generates textures, and hands out samplers. Cached by URL + color space,
so loading the same texture twice is free.

```ts
// Albedo/color maps: sRGB TRUE (GPU linearizes on sample).
const albedo = await engine.textureManager.load('/tex/albedo.png', { sRGB: true });
// Normal / ARM / data maps: sRGB FALSE.
const normal = await engine.textureManager.load('/tex/normal.png', { sRGB: false });
```

| Member | Description |
|---|---|
| `load(url, { sRGB?, samplerDescriptor? })` | Fetch → decode → upload → cached `GPUTextureView`. Falls back to a 1×1 white texture on failure. |
| `getSampler(descriptor)` | Cached `GPUSampler` (keyed by stringified descriptor). |
| `getSolidTexture(r, g, b, a?)` | 1×1 solid-color `GPUTextureView` (0–255 per channel), cached. |
| `fallbackWhite` | 1×1 white — default albedo/packed fallback. |
| `fallbackNormal` | 1×1 `(128,128,255)` — flat tangent-space normal. |
| `defaultSampler` | Linear filtering, `repeat` addressing — good default for 3D models. |
| `clearCache()` | Destroy all cached textures and drop samplers (call when tearing down a scene). |

> **`sRGB` matters.** Pass `true` only for color/albedo maps. Passing it for normal/roughness/metallic
> maps corrupts the data by applying gamma to values that aren't colors.

The materials system leans on `fallbackWhite` / `fallbackNormal` whenever a texture option is
omitted — see [Materials](./materials.md#how-materials-work).

---

## 7. The shared context (`engine.ctx`)

`engine.ctx` is the `WebGPUContext` — a `public readonly` surface for the low-level resources the
engine allocates. Reach for it when a custom compute stage or shader needs a shared buffer.

| Member | Type | Notes |
|---|---|---|
| `ctx.device` | `GPUDevice` | Same object as `engine.device`. |
| `ctx.format` | `GPUTextureFormat` | Swapchain format (may be `*-srgb`). |
| `ctx.cameraUniformBuffer` | `GPUBuffer` | The 432-byte global camera uniform — bind it into compute stages. |
| `ctx.depthTexture` | `GPUTexture` | Engine depth attachment (`depth24plus`, `RENDER_ATTACHMENT` only — not samplable). |
| `ctx.width` / `ctx.height` | `number` | Current drawing-buffer size (also `engine.width` / `engine.height`). |

> `ctx.depthTexture` is render-attachment only. For a **samplable** scene depth (e.g. SSAO), declare
> `gbuffer.depth` on a [`PostChain`](./post.md), which allocates a separate sampled depth target.
