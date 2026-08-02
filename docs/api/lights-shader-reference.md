# Lights: Shader & Compute Reference

This is the low-level companion to the [Lights API guide](./lights.md). It documents the exact
memory layout of a light, the WGSL contract the `LightSystem` injects, and how to consume that
layout **directly** — in a fully custom fragment shader or in your **own compute stage** — instead
of going through `buildPBRShader`.

Read [lights.md](./lights.md) first for the high-level flow. Everything here assumes you already
have a `LightSystem` and its `lightStorageBuffer`.

---

## 1. Light memory layout (16 floats / 64 bytes)

Every light is `LightSystem.FLOATS_PER_LIGHT` (16) floats — four `vec4`s. The proxies
([`PointLightProxy`](./lights.md) etc.) write into exactly these slots; if you build the buffer by
hand (e.g. in a Worker) you must match this layout byte-for-byte.

| vec4 | Float index | Field (WGSL) | Meaning |
|---|---|---|---|
| `pos_type` | 0, 1, 2 | `.xyz` | World position (point / spot) |
| | 3 | `.w` | **Light type**: `1.0` point, `2.0` directional, `3.0` spot |
| `col_int` | 4, 5, 6 | `.xyz` | Color RGB (linear) |
| | 7 | `.w` | Intensity (multiplied into color) |
| `dir_rad` | 8, 9, 10 | `.xyz` | Direction (directional / spot) |
| | 11 | `.w` | Radius / range (point / spot) |
| `spot_params` | 12 | `.x` | `cos(innerConeAngle)` |
| | 13 | `.y` | `cos(outerConeAngle)` |
| | 14, 15 | `.zw` | Reserved |

```wgsl
struct Light {
    pos_type: vec4<f32>,
    col_int: vec4<f32>,
    dir_rad: vec4<f32>,
    spot_params: vec4<f32>,
};
```

> **Cone angles are stored as cosines.** `SpotLightProxy.innerConeAngle`/`outerConeAngle` call
> `Math.cos()` on assignment. If you write the buffer manually, store `Math.cos(angle)`, not the
> angle — the shader compares against `dot(...)` directly.

### Capacity and index assignment

`capacity: { point, directional, spot }` only sizes the buffer — `totalLights` is their sum. There
is **no enforced partitioning**: proxies `seek(index)` to any slot, and the shaders identify a
light purely by its `pos_type.w` type flag. You own the index bookkeeping. A common scheme is to
reserve a fixed range per type (e.g. directional at `[0, 4)`, points after), but any convention
works as long as unused slots have a type your shader ignores.

---

## 2. The shader contract (`getShaderLayout`)

`lights.getShaderLayout()` returns `{ wgslStructs, wgslLogic } | undefined`. The material shader
builders splice these two strings in and then call the two functions the logic chunk defines. That
is the entire contract — anything that provides these functions is a valid lighting backend:

```wgsl
// Injected by wgslLogic — the two functions every material loop calls:
fn getVisibleLightCount(worldPos: vec3<f32>) -> u32;
fn getIncidentLight(index: u32, worldPos: vec3<f32>) -> IncidentLight;

struct IncidentLight {
    color: vec3<f32>,      // radiance (already color * intensity)
    direction: vec3<f32>,  // unit vector from surface toward the light
    attenuation: f32,      // distance + spot falloff, 1.0 for directional
};
```

The consuming loop in every lit material looks like this:

```wgsl
let count = getVisibleLightCount(in.worldPos);
for (var i = 0u; i < count; i = i + 1u) {
    let L = getIncidentLight(i, in.worldPos);
    // L.color, L.direction, L.attenuation → feed your BRDF
}
```

- **`standard`** — `getVisibleLightCount` returns the total light count; `getIncidentLight` indexes
  `lights[index]` directly.
- **`clustered-forward`** — `getVisibleLightCount` returns the per-cluster count for the pixel;
  `getIncidentLight` resolves `lightIndices[cluster.start_offset + i]` into `globalLights[...]`.

Both paths return the identical `IncidentLight`, so a material shader is agnostic to which technique
is active — swapping `standard` ↔ `clustered-forward` needs no shader change.

### Group 2 render bindings

`applyToBatch(batch, groupIndex = 2)` binds these into the fragment shader's group:

| Binding | `standard` | `clustered-forward` |
|---|---|---|
| 0 | `lights: array<Light>` | `globalLights: array<Light>` |
| 1 | — | `clusterGrid: array<Cluster>` |
| 2 | — | `lightIndices: array<u32>` |

---

## 3. Using the layout in a fully custom fragment shader

You don't have to use `buildPBRShader`. Any WGSL shader can consume the lighting layout — just paste
the struct/logic chunks in and honor the group-2 bindings. This is how you'd write a bespoke lit
shader (custom BRDF, cheaper lighting, debug view, etc.):

```ts
const layout = lights.getShaderLayout()!; // 'standard' or 'clustered-forward'

const shaderCode = /* wgsl */ `
    struct Camera { /* ...432-byte layout... */ };
    @group(0) @binding(0) var<uniform> camera: Camera;

    ${layout.wgslStructs}   // Light / IncidentLight / Cluster + @group(2) bindings
    ${layout.wgslLogic}     // getVisibleLightCount / getIncidentLight

    struct VertexOut { @builtin(position) clipPos: vec4f, @location(0) worldPos: vec3f, /* ... */ };

    @fragment
    fn fs_main(in: VertexOut) -> @location(0) vec4f {
        let N = normalize(in.worldNormal);
        var lit = vec3f(0.0);

        let count = getVisibleLightCount(in.worldPos);
        for (var i = 0u; i < count; i = i + 1u) {
            let L = getIncidentLight(i, in.worldPos);
            lit += max(dot(N, L.direction), 0.0) * L.color * L.attenuation; // your own model
        }
        return vec4f(lit, 1.0);
    }
`;

const batch = engine.createBatch(scenePass, {
    shaderCode,
    strideFloats: 14,
    maxInstances: 1000,
    vertexLayouts: geom.layout.getWebGPUDescriptor(),
});
lights.applyToBatch(batch, 2); // supplies the group-2 buffers the layout expects
```

> The **clustered** logic references `camera.view`, `camera.proj`, `camera.zNear`, `camera.zFar` —
> so your shader must declare the full `Camera` struct (432-byte layout — see `src/core/Camera.ts`
> or [Deprecated API §3](../deprecated-api.md)) and bind it at `@group(0) @binding(0)`.

---

## 4. Rolling your own compute stage over the light buffers

`getComputeStage(camera)` builds the stock clustered binning stage. If you need custom binning,
different culling math, or an extra pre-pass (e.g. clearing the cluster grid, which the stock stage
notes it does not do), build your own `ComputeStageNode` against the same buffers.

### Buffers the `LightSystem` exposes

| Property | WGSL type | Notes |
|---|---|---|
| `lightStorageBuffer` | `array<Light>` (read) | The 16-float-per-light data. |
| `clusterGridBuffer` | `array<Cluster>` (read_write) | 2×`u32` per cluster: `start_offset`, `light_count`. |
| `lightIndexBuffer` | `array<u32>` (read_write) | Flat list of light IDs, `totalClusters * maxLightsPerTile` long. |
| `globalCounterBuffer` | `atomic<u32>` (read_write) | Bump allocator into `lightIndexBuffer`. |

`clusterGridBuffer`, `lightIndexBuffer`, and `globalCounterBuffer` are only allocated when the
technique is `clustered-forward`. `engine.ctx.cameraUniformBuffer` gives you the camera uniform.

### Binding layout the stock stage uses (group 0)

```wgsl
@group(0) @binding(0) var<uniform> camera: Camera;                              // engine.ctx.cameraUniformBuffer
@group(0) @binding(1) var<storage, read>       globalLights: array<Light>;     // lightStorageBuffer
@group(0) @binding(2) var<storage, read_write> clusterGrid: array<Cluster>;    // clusterGridBuffer
@group(0) @binding(3) var<storage, read_write> lightIndices: array<u32>;       // lightIndexBuffer
@group(0) @binding(4) var<storage, read_write> globalCounter: atomic<u32>;     // globalCounterBuffer
```

### Building the stage

```ts
const grid = lights.config.clusterConfig!.grid;
const totalClusters = grid[0] * grid[1] * grid[2];

const myBinning = engine.createComputeStage({
    name: 'Custom Light Binning',
    shaderCode: myCullWGSL,                       // @workgroup_size(64,1,1), entry fn cs_main
    dispatchCount: [Math.ceil(totalClusters / 64), 1, 1],

    // Runs immediately before the dispatch — MUST reset the bump allocator each frame
    preClearBuffers: [
        { buffer: lights.globalCounterBuffer!, offset: 0, data: new Uint32Array([0]) },
    ],
});

engine.setComputePassBindGroup(myBinning, [
    { binding: 0, resource: { buffer: engine.ctx.cameraUniformBuffer } },
    { binding: 1, resource: { buffer: lights.lightStorageBuffer } },
    { binding: 2, resource: { buffer: lights.clusterGridBuffer! } },
    { binding: 3, resource: { buffer: lights.lightIndexBuffer! } },
    { binding: 4, resource: { buffer: lights.globalCounterBuffer! } },
], 0);

// Compute first, then the shading pass that reads the grid
engine.render([myBinning, scenePass]);
```

### What the stock binning kernel does (reference)

The default `cs_main` (one invocation per cluster, 64 per workgroup):

1. Reconstruct the cluster's `(x, y, z)` from `global_id.x`; skip out-of-range invocations.
2. Compute an approximate bounding sphere for the cluster in **view space** using logarithmic depth
   slices (`zNear * (zFar/zNear)^(z/GRID_Z)`) and the camera FOV/aspect.
3. Loop over all `TOTAL_LIGHTS`:
   - **Directional** (`type == 2.0`) → always added (they affect every cluster).
   - **Point / spot** → transform position to view space and do a sphere-vs-sphere test against
     `clusterRadius + lightRadius`.
4. `atomicAdd(&globalCounter, visibleCount)` to reserve a contiguous slice, write
   `Cluster(offset, visibleCount)` into `clusterGrid[clusterIndex]`, then append the light IDs into
   `lightIndices[offset + i]`.

The fragment side (§2, clustered) then reads `clusterGrid` for the pixel's cluster and walks
`lightIndices` to fetch each `Light`. Your custom kernel only has to keep the same output contract —
fill `clusterGrid` and `lightIndices` — and the stock fragment logic will consume it unchanged.

> **Depth slicing is logarithmic.** Both the compute binning and the fragment `getClusterData` use
> `log(depth / zNear) / log(zFar / zNear) * GRID_Z`. If you change the slicing in compute, mirror it
> in your fragment lookup or lights will bin into the wrong cluster.

---

## 5. Gotchas

- **Reset the counter every frame.** Without the `preClearBuffers` reset of `globalCounterBuffer`,
  `atomicAdd` keeps climbing and overflows `lightIndexBuffer` within a few frames.
- **`maxLightsPerTile` is a hard cap.** The kernel stops adding lights to a cluster once it hits the
  cap, and it also bounds the total index buffer size. Size it for your worst-case local density.
- **Only `clustered-forward` allocates the spatial buffers.** `standard` has `lightStorageBuffer`
  only; `getComputeStage` throws unless `clusterConfig` is present.
- **Directional lights ignore culling** by design — they are unconditionally added to every cluster,
  so a scene with many directionals will inflate every cluster's count.
