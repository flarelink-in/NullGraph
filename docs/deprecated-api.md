# Deprecated & Breaking API Changes

This document tracks APIs that have been **renamed**, **removed**, or **restructured** in
NullGraph, along with drop-in migration steps. Entries are derived from the diff between the
last published surface and the current `src/` tree.

> NullGraph follows a "brutalist" philosophy: when an API stops reflecting the underlying
> data model, it is renamed rather than aliased. There are **no silent shims** — the old
> names are gone, so update call sites directly using the tables below.

---

## 1. Bind-group API renamed (`null-graph` core)

The bind-group helpers on `NullGraph` were renamed to say what they actually do (they build
and cache a `GPUBindGroup`), and the render/compute paths were split into two explicit methods.

### `attachTextureMaterial()` → `createTextureBindGroup()`

Same signature, same return value. Only the name changed.

```ts
// ❌ Before
const bg = engine.attachTextureMaterial(batch, [albedoView, normalView], sampler, extra, 1);

// ✅ After
const bg = engine.createTextureBindGroup(batch, [albedoView, normalView], sampler, extra, 1);
```

> Note: `StandardPBRMaterial` already calls the new method internally — if you only use the
> material classes you don't need to change anything.

### `attachCustomBindGroup()` → `setRenderPassBindGroup()` / `setComputePassBindGroup()`

The old method took a `target: 'render' | 'compute' | 'both'` argument because batches used to
own an embedded compute pipeline. Compute is now a **standalone stage** (see §2), so the method
was split so each target has a dedicated call.

| Old call | New call |
|---|---|
| `attachCustomBindGroup(batch, entries, group)` (default `'render'`) | `setRenderPassBindGroup(batch, entries, group)` |
| `attachCustomBindGroup(batch, entries, group, 'compute')` | `setComputePassBindGroup(stage, entries, group)` |
| `attachCustomBindGroup(batch, entries, group, 'both')` | Call both methods explicitly |

```ts
// ❌ Before — bound to the batch's hidden compute pipeline
engine.attachCustomBindGroup(meshBatch, entries, 2, 'render');

// ✅ After — render binding goes to the batch...
engine.setRenderPassBindGroup(meshBatch, entries, 2);

// ...and compute binding goes to a ComputeStageNode (default group index is 0)
engine.setComputePassBindGroup(computeStage, entries, 0);
```

---

## 2. Compute is no longer embedded in a batch (breaking)

Previously you enabled GPU-driven rendering by passing `computeShaderCode` into `createBatch`,
and the batch silently created and dispatched a compute pipeline. That coupling is removed.
Compute now lives in a **separate, explicit `ComputeStageNode`** that you sequence yourself.

**Removed from `PipelineConfig`:**

- `computeShaderCode`
- `extraBindGroup`

**Removed from `RenderBatch`:**

- `computePipeline`
- `computeBindGroup`
- `computeCustomBindGroups`

**New API:** `engine.createComputeStage(config)`, `engine.setComputePassBindGroup(stage, ...)`,
and an explicit `render(pipelineSequence)` sequence (see §4).

```ts
// ❌ Before — one batch that hid a compute cull pass
const batch = engine.createBatch(pass, {
    isIndirect: true,
    computeShaderCode: cullWGSL,
    shaderCode: drawWGSL,
    strideFloats: 14,
    maxInstances: 5000,
});
engine.render(); // compute ran implicitly

// ✅ After — create the indirect batch first, then a separate compute stage
const batch = engine.createBatch(pass, {
    isIndirect: true,
    shaderCode: drawWGSL,
    strideFloats: 14,
    maxInstances: 5000,
    sharedSourceBuffer: sourceBuffer, // instance data the compute stage reads/writes
});

const cull = engine.createComputeStage({
    name: 'Frustum Cull',
    shaderCode: cullWGSL,
    dispatchCount: [Math.ceil(5000 / 64), 1, 1],
    preClearBuffers: [
        { buffer: batch.indirectBuffer!, offset: 4, data: new Uint32Array([0]) }, // reset instanceCount
    ],
});

engine.setComputePassBindGroup(cull, [
    { binding: 0, resource: { buffer: engine.ctx.cameraUniformBuffer } },
    { binding: 1, resource: { buffer: sourceBuffer } },
    { binding: 2, resource: { buffer: batch.storageBuffer } },
    { binding: 3, resource: { buffer: batch.indirectBuffer! } },
], 0);

// Explicit ordering: compute first, then the pass that draws its results
engine.render([cull, pass]);
```

When `isIndirect: true`, `createBatch` still allocates `indirectBuffer` and (if you don't pass
`sharedSourceBuffer`) a `sourceStorageBuffer` for you — it just no longer builds the pipeline.

---

## 3. Camera uniform layout changed (breaking for custom WGSL)

The camera uniform buffer was expanded from **20 floats (80 bytes)** to **108 floats (432 bytes)**
to carry view/projection **inverse** matrices (for SSR, decals, clustered culling, billboarding)
plus scalar metadata. If you wrote your own WGSL `Camera` struct, it must be updated.

| Data | Old offset (floats) | New offset (floats) |
|---|---|---|
| `viewProj` (mat4) | 0 | 0 |
| `view` (mat4) | — | 16 |
| `proj` (mat4) | — | 32 |
| `invViewProj` (mat4) | — | 48 |
| `invView` (mat4) | — | 64 |
| `invProj` (mat4) | — | 80 |
| **`eyePosition` (vec3)** | **16** | **96** |
| time (padding slot, via `camera.updateTime`) | — | 99 |
| `zNear`, `zFar`, `fovRads`, `aspect` | — | 100–103 |

The largest breaking change is that **`eyePosition` moved from float offset 16 to 96**. Any
hand-written shader that read the eye position at the old location will now read a matrix row.
The stock shader builders (`buildPBRShader`, etc.) already reference the new `Camera` struct —
only fully custom shaders are affected. See the canonical struct in `src/core/Camera.ts`.

`updateCamera()` now writes `camera.bufferData.byteLength` (432 bytes) instead of a hardcoded
`20 * 4`, so no call-site change is needed as long as you use the `Camera` class.

---

## 4. `render()` takes an explicit pipeline sequence

`render()` still works with no arguments (it replays every tracked pass in creation order — the
legacy path). The modern, data-oriented path is to pass an **explicit ordered array** of compute
stages and render passes, which is required for GPU-driven culling and post-processing.

```ts
// ✅ Legacy (still supported): replays all createPass() nodes in order
engine.render();

// ✅ Modern: you control the exact execution order, including compute
engine.render([lightBinningStage, cullStage, scenePass, postPass]);
```

---

## 5. `init()` signature widened

`WebGPUContext.init(canvas, desiredFeatures[])` became `init(canvas, options)`. At the
`NullGraph` level, `init(canvas, { desiredFeatures })` still works, and the options object now
also accepts `powerPreference`, `width`, and `height`, and the canvas may be an `OffscreenCanvas`.

```ts
// ✅ Still valid
await engine.init(canvas, { desiredFeatures: ['timestamp-query'] });

// ✅ New options
await engine.init(offscreenCanvas, {
    desiredFeatures: ['timestamp-query'],
    powerPreference: 'high-performance',
    width: 1920,
    height: 1080,
});
```

---

## 6. Engine context is now public: `gpuCtx` → `ctx`

The engine's WebGPU context was a private field named `gpuCtx`. It is now exposed as a
`public readonly ctx` so you can reach shared resources (notably
`engine.ctx.cameraUniformBuffer`) when wiring custom compute stages. If you were reaching into
the private `gpuCtx` via a cast, switch to the supported `engine.ctx`.

```ts
// ❌ Before (unsupported internal access)
const camBuf = (engine as any).gpuCtx.cameraUniformBuffer;

// ✅ After
const camBuf = engine.ctx.cameraUniformBuffer;
```

New convenience getters `engine.width` and `engine.height` are also available.

---

## 7. Materials module: `ShaderBuilder.ts` → `PBRShaderBuilder.ts`

The file `src/extras/materials/ShaderBuilder.ts` was renamed to `PBRShaderBuilder.ts`. The
exported function name is **unchanged** (`buildPBRShader`), and the public subpath is still
`null-graph/materials`, so the standard import keeps working:

```ts
// ✅ Unchanged — import from the module, not the file
import { buildPBRShader, StandardPBRMaterial } from 'null-graph/materials';
```

Only breaks if you deep-imported the file path directly (e.g. `null-graph/materials/ShaderBuilder`),
which was never a supported entry point. The materials module now also ships `buildBasicShader`,
`buildLambertShader`, `buildToonShader`, `buildEmissiveShader`, and `buildMatcapShader` alongside
their matching material classes.

---

*Have a migration that isn't covered here? Open an issue on the
[NullGraph repository](https://github.com/Vikas593-cloud/NullGraph).*
