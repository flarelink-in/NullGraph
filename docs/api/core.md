# Core API Usage

The `null-graph` core package owns the WebGPU device, render passes, compute stages, batches, and
buffer streaming. This page covers the main `NullGraph` engine surface; see
[Deprecated API](../deprecated-api.md) if you are migrating from an older version.

> **Going lower level?** For the bind-group conventions, the instance-data layout, the `Camera`
> uniform (WGSL struct + offsets), the full `ComputeStageConfig`, and the `BufferManager` /
> `TextureManager` / `engine.ctx` surfaces, see the
> [Core Reference](./core-reference.md).

## Engine Initialization

```ts
import { NullGraph, Camera } from 'null-graph';

const engine = new NullGraph();
const result = await engine.init(canvas, {
    desiredFeatures: ['timestamp-query'],   // extra optional features, negotiated against the adapter
    powerPreference: 'high-performance',    // 'high-performance' (default) | 'low-power'
    autoNegotiate: true,                    // default: opportunistically enable modern features
    // width / height override — useful with an OffscreenCanvas
});

if (!result.success) throw new Error(result.error);
```

`canvas` may be an `HTMLCanvasElement` **or** an `OffscreenCanvas`, so the whole engine can run in a
Web Worker — **or `null`**, see below. After init you can read `engine.width` / `engine.height`,
reach shared GPU resources via `engine.ctx` (e.g. `engine.ctx.cameraUniformBuffer`), and access
`engine.device`, `engine.bufferManager`, `engine.textureManager`, and `engine.capabilities`.

## Headless rendering & sharing a device

Two options change *where* the engine runs rather than how it renders. Both are additive — a
canvas-only `init(canvas)` behaves exactly as before.

**No canvas — `init(null, { width, height })`.** The main-screen pass renders into an offscreen
texture, `engine.screenTexture` (`RENDER_ATTACHMENT | COPY_SRC`, format `engine.ctx.format`:
`'rgba8unorm'` unless you pass `format`). This is how the engine runs in Node (Dawn), in tests, and
in the render-to-PNG tool. `width` and `height` are required — there is no canvas to size from —
and `engine.resize(w, h)` recreates the screen texture. `engine.ctx.isHeadless` tells the two apart.

```ts
const engine = new NullGraph();
await engine.init(null, { width: 512, height: 512 });
// ...build passes; a pass with isMainScreenPass: true draws into engine.screenTexture
engine.render(sequence);
// copyTextureToBuffer(engine.screenTexture) to read the frame — in a test, never in the frame loop
```

**Adopt a device — `init(canvas | null, { device })`.** Use a `GPUDevice` another library already
created instead of requesting one, so the two share buffers with no copy. The motivating case is
ONNX Runtime Web: create the session first (ORT owns the device), then adopt it, and bind a model
output's `tensor.gpuBuffer` directly in a compute stage.

```ts
import * as ort from 'onnxruntime-web/webgpu';

const session = await ort.InferenceSession.create(model, {
    executionProviders: ['webgpu'], preferredOutputLocation: 'gpu-buffer',
});
await engine.init(canvas, { device: await ort.env.webgpu.device });

const { output } = await session.run({ input });              // stays on the GPU
engine.setComputePassBindGroup(stage, [{ binding: 0, resource: { buffer: output.gpuBuffer } }, /* … */]);
engine.render([stage, scenePass]);
await engine.device.queue.onSubmittedWorkDone();               // keep the tensor alive until read…
output.dispose();                                              // …then release it
```

- Adopt in that direction (the runtime owns, NullGraph adopts): the runtime requests the features
  and limits its kernels need, and NullGraph's baseline needs no optional feature. ONNX Runtime
  1.29 allows no other: given a device the app made, its runs fail.
- **ONNX Runtime destroys its device with its last session.** Keep a session alive while the
  engine renders on that device, or use `OrtRuntime` from `null-graph/inference`, which keeps one
  for its whole life and runs models over the engine's buffers
  ([inference.md](./inference.md#onnx-runtime)).
- Nothing is negotiated on an adopted device. `engine.capabilities` reports what it already has
  (subgroup sizes from `device.adapterInfo` where the browser exposes it); `powerPreference` and
  `autoNegotiate` are refused with a named error, and a missing `desiredFeatures` entry warns.
- `engine.ctx.ownsDevice` is `false`, and NullGraph never destroys a device it adopted.
- **`engine.destroy()` releases what the engine created, and only that.** That covers:
  - its batches' own buffers — never a `sharedSourceBuffer`, geometry, or a `cameraBuffer` you passed;
  - the texture manager's textures;
  - the screen and depth targets and the camera uniform;
  - the canvas's configuration;
  - the device, only if the engine requested it.

  `render()` then does nothing. Destroy extras first (`PostChain`, `LightSystem`, …): they own
  their resources. Idempotent, and safe after a failed `init`; every release is attempted and
  failures are thrown together. Destroying an engine's own device rejects reads still pending on
  it, so await `device.queue.onSubmittedWorkDone()` first if something is reading back.
- Pipelines and shader modules carry labels — a batch's `label` (default `"<pass> #<index>"`) and a
  compute stage's `name` — so a WebGPU validation error names what caused it.

Executable versions of both: `test/core-gpu` (C1–C6) and `test/onnx-gpu` (O1–O3), run by
`npm run test:gpu`.

## Feature Negotiation & Capabilities

To stay in sync with WebGPU's progression, `init()` **opportunistically enables** a curated set of
forward-looking features whenever the adapter supports them — currently `subgroups`, `shader-f16`,
`bgra8unorm-storage`, and `dual-source-blending` (`PROGRESSION_FEATURES`). This is purely additive:
every candidate is checked against the adapter's support list first, so it can only ever *enable* an
optional capability, never fail init. Set `autoNegotiate: false` to request only your explicit
`desiredFeatures`.

It also raises three **limits** to the adapter's own values (`PROGRESSION_LIMITS`):
`maxStorageBuffersPerShaderStage`, `maxStorageBufferBindingSize` and `maxBufferSize`. Without
that, a device gets WebGPU's defaults whatever the GPU offers: 8 storage buffers per stage and
128 MB bindings, where current desktop GPUs report 16 and about 2 GB. The list is curated, not
"every limit", because the spec warns that better limits can cost performance on some
implementations. Read the result on `engine.device.limits`. Modules size against those limits and
refuse above them with a named error, so code that needs more than the defaults fails loudly on a
baseline device. `autoNegotiate: false` gives a device with the defaults, which is how to check
that an app runs on the baseline. An adopted device keeps the limits its owner chose.

The negotiated result is exposed as `engine.capabilities` — a structured, queryable snapshot the
engine uses internally (e.g. the clustered `LightSystem` switches to a subgroup-accelerated binning
kernel when `capabilities.subgroups` is set) and that you can read for your own code paths:

```ts
const caps = engine.capabilities; // GPUCapabilities

caps.subgroups;           // boolean — subgroup ops (enable subgroups;) available
caps.shaderF16;           // boolean — 16-bit floats (enable f16;) available
caps.bgra8unormStorage;   // boolean — bgra8unorm usable as a storage texture
caps.dualSourceBlending;  // boolean — @blend_src(0/1) available
caps.subgroupMinSize;     // number  — hardware subgroup size range (0 if unsupported)
caps.subgroupMaxSize;     // number
caps.has('timestamp-query'); // membership test against the enabled feature set
caps.features;            // ReadonlySet<GPUFeatureName> — everything the device granted
```

> The `shaderF16`, `bgra8unormStorage`, and `dualSourceBlending` flags are detected as **groundwork**
> — they are surfaced now so those features can be adopted later without re-plumbing negotiation.

## Render Passes
NullGraph isolates rendering logic into distinct passes. You can render pristine 3D scenes offscreen and pipe them into post-processing passes.

```ts
// 1. Create an offscreen pass
const scenePass = engine.createPass({
    name: 'Offscreen Pass',
    isMainScreenPass: false,
    colorAttachments: [{
        view: offscreenTexture.createView(),
        clearValue: { r: 0.0, g: 0.01, b: 0.03, a: 1.0 },
        loadOp: 'clear', storeOp: 'store'
    }],
    // ... depth attachments
});

// 2. Create the final Post-Processing pass
const hudPass = engine.createPass({
    name: 'HUD Post Process',
    isMainScreenPass: true
});

// 3. Bind the offscreen texture to your post-process batch
const hudBatch = engine.createBatch(hudPass, { /* shader args */ });
engine.createTextureBindGroup(hudBatch, offscreenTexture.createView(), sampler);
```

## Indirect Drawing (GPU Compute Culling)
Stop relying on the CPU to figure out what to render. A **compute stage** evaluates thousands of
instances, writes an `IndirectDrawArgs` buffer, and commands the vertex shader without the CPU ever
knowing what happened.

> **Changed:** compute is no longer embedded in a batch via `computeShaderCode`. Create a standalone
> `ComputeStageNode` with `engine.createComputeStage(...)` and sequence it explicitly in `render()`.
> See [Deprecated API §2](../deprecated-api.md) for the migration.

```ts
// 1. An indirect batch allocates its indirect + source buffers (no pipeline)
const batch = engine.createBatch(scenePass, {
    isIndirect: true,
    shaderCode: renderShaderCode,
    strideFloats: 14,
    maxInstances: 5000,
    vertexLayouts: geometry.layout.getWebGPUDescriptor(),
});

// 2. A separate compute stage does the culling and writes drawArgs
const cull = engine.createComputeStage({
    name: 'Frustum Cull',
    shaderCode: cullWGSL, // atomicAdd(&drawArgs.instanceCount, 1u); ... cull & pack
    dispatchCount: [Math.ceil(5000 / 64), 1, 1],
    preClearBuffers: [
        { buffer: batch.indirectBuffer!, offset: 4, data: new Uint32Array([0]) },
    ],
});

engine.setComputePassBindGroup(cull, [
    { binding: 0, resource: { buffer: engine.ctx.cameraUniformBuffer } },
    { binding: 1, resource: { buffer: batch.sourceStorageBuffer! } },
    { binding: 2, resource: { buffer: batch.storageBuffer } },
    { binding: 3, resource: { buffer: batch.indirectBuffer! } },
], 0);

// 3. Execute compute first, then the draw pass
engine.render([cull, scenePass]);
```

## Advanced Alpha Blending
Whether you need standard transparency or intense additive glowing effects, NullGraph exposes WebGPU's blend states directly at the batch level.

```ts
// Example: Additive Blending for a particle system (Quantum Nebula)
const physicsBatch = engine.createBatch(scenePass, {
    // ... shaders and layout configs
    
    // ADDITIVE BLENDING: Colors sum together, creating intense glowing cores
    blend: {
        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }
    },
    depthWriteEnabled: false,
    depthCompare: 'less'
});
```

## Streaming Instance Data (Zero-Copy)
Batches are backed by a pre-allocated storage buffer. Blast a flat `Float32Array` — computed on the
main thread or in a Web Worker — straight into it. The default instance stride is 14 floats
(index/pad, position xyz, rotation quat xyzw, scale xyz, color rgb).

```ts
engine.setBatchGeometry(batch, geom.vertexBuffer!, geom.indexBuffer!, geom.indices.length);

const data = new Float32Array(instanceCount * 14);
// ... fill per-instance transforms ...
engine.updateBatchData(batch, data, instanceCount);
```

## Custom Bind Groups
Attach arbitrary resources to a batch's render pipeline (materials, skeletons, light buffers). For
compute stages, use the compute variant.

```ts
// Render pipeline (default group index 1)
engine.setRenderPassBindGroup(batch, [
    { binding: 0, resource: { buffer: skeletonManager.boneBuffer } },
], 2);

// Compute stage (default group index 0)
engine.setComputePassBindGroup(computeStage, entries, 0);
```

## Execution
`render()` accepts an explicit, ordered sequence of compute stages and render passes — the modern,
Data-Oriented path. Called with no arguments it replays every tracked pass in creation order.

```ts
engine.updateCamera(camera);            // upload camera uniforms (432-byte block)
engine.render([cullStage, scenePass, postPass]); // explicit order
// engine.render();                     // legacy: replays all createPass() nodes
```

## Camera & resource managers

A core `Camera` produces the view/projection matrices the engine uploads; `engine.bufferManager` and
`engine.textureManager` create geometry buffers and load/generate textures.

```ts
import { Camera } from 'null-graph';

const camera = new Camera(75, canvas.width / canvas.height, 0.1, 1000);
camera.updateView([0, 3, 8], [0, 0, 0]); // eye, target
engine.updateCamera(camera);

// Geometry buffers from raw typed arrays
const vbo = engine.bufferManager.createVertexBuffer(vertices);
const ibo = engine.bufferManager.createIndexBuffer(indices);

// Textures (sRGB TRUE for albedo/color, FALSE for normal/data maps)
const albedo = await engine.textureManager.load('/albedo.png', { sRGB: true });
```

Full signatures — the `Camera` methods, the WGSL camera struct, and every `BufferManager` /
`TextureManager` member — are in the [Core Reference](./core-reference.md).

## Cheat sheet

| Method | Purpose |
|---|---|
| `init(canvas, options)` | Boot WebGPU; returns `{ success, enabledFeatures, error? }`. |
| `init(null, { width, height })` | Boot headless — the main pass renders into `engine.screenTexture`. |
| `init(canvas \| null, { device })` | Adopt an existing device (e.g. ONNX Runtime's) instead of requesting one. |
| `resize(w, h)` | Reallocate the depth texture (and the headless screen texture) for a new size. |
| `createPass(config)` | Create a render pass (screen or offscreen). |
| `createBatch(pass, config)` | Create a render batch (pipeline + storage buffer). |
| `createComputeStage(config)` | Create a standalone GPU compute stage. |
| `createComputeStageAsync(config)` | The same stage, its pipeline from `createComputePipelineAsync`: the compile does not hold the caller. A shader that does not compile rejects, naming the stage, with the device's message about the WGSL. |
| `setBatchGeometry(batch, vb, ib, count, format?)` | Bind vertex/index buffers to a batch. |
| `updateBatchData(batch, data, count)` | Stream instance data into a batch (Zero-Copy). |
| `createTextureBindGroup(batch, views, sampler, extra?, group?)` | Build/cache a texture bind group. |
| `setRenderPassBindGroup(batch, entries, group?)` | Bind custom resources to a batch. |
| `setComputePassBindGroup(stage, entries, group?)` | Bind custom resources to a compute stage. |
| `updateCamera(camera)` | Upload the camera uniform block. |
| `render(sequence?)` | Execute the pipeline. |
| `destroy()` | Release what the engine created (batch buffers it allocated, textures, targets, camera uniform, canvas configuration; its device only if it requested it). `render()` does nothing afterwards. |

**Engine properties:** `device`, `ctx` (readonly `WebGPUContext`), `capabilities` (readonly
`GPUCapabilities`), `bufferManager`, `textureManager`, `width`, `height`, `isDestroyed`,
`screenTexture` (headless only), `canvas` (`null` when headless). On `ctx`: `isHeadless`,
`ownsDevice`, `getScreenView()`, and `context` (`undefined` when headless). See the
[Core Reference](./core-reference.md#7-the-shared-context-enginectx).
