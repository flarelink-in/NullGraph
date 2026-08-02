# Lights API Usage

The `null-graph/lights` module is a Data-Oriented lighting system. Lights are not objects — they
are **16-float slices** in one contiguous `Float32Array` that you upload straight to VRAM. Thin
"cursor" proxies let you write those slices by field name without allocating anything, and the
`LightSystem` supplies the matching WGSL and (optionally) a GPU compute stage for clustered culling.

```ts
import {
    LightSystem,
    PointLightProxy, DirectionalLightProxy, SpotLightProxy,
} from 'null-graph/lights';
```

Supported techniques (`LightSystemConfig.technique`):

| Technique | Meaning |
|---|---|
| `'none'` | Lighting disabled. |
| `'standard'` | Every fragment loops over every light. Simple; best for small light counts. |
| `'tiled-forward'` | Screen-space tile culling (Forward+). |
| `'clustered-forward'` | Volumetric cluster culling via a compute stage — scales to thousands of lights. |

---

## 1. Create the system

```ts
const lights = new LightSystem(engine, {
    capacity: { point: 1000, directional: 4, spot: 256 },
    technique: 'clustered-forward',
    clusterConfig: {
        grid: [16, 9, 24],       // volumetric tiles (X, Y, depth slices)
        maxLightsPerTile: 100,
    },
    // For 'tiled-forward' instead:
    // tiledConfig: { tileSize: [16, 16], maxLightsPerTile: 64 },
});
```

Each light occupies `LightSystem.FLOATS_PER_LIGHT` (16) floats, laid out as four `vec4`s:
`pos_type`, `col_int`, `dir_rad`, `spot_params`.

## 2. Write light data with proxies

Proxies never allocate — they wrap your `Float32Array` and `seek()` to a light index, then you
assign fields. `seek()` also stamps the correct light *type* into the slice automatically.

```ts
// Your own pre-allocated buffer: capacity * 16 floats
const lightData = new Float32Array(1000 * LightSystem.FLOATS_PER_LIGHT);

const point = new PointLightProxy();
point.wrap(lightData);

point.seek(0);
point.position = [4, 3, 0];
point.color = [1.0, 0.6, 0.2];
point.intensity = 8.0;
point.radius = 12.0;

const sun = new DirectionalLightProxy();
sun.wrap(lightData);
sun.seek(1);
sun.direction = [-0.5, -1.0, -0.3];
sun.color = [1.0, 1.0, 0.95];
sun.intensity = 3.0;

const spot = new SpotLightProxy();
spot.wrap(lightData);
spot.seek(2);
spot.position = [0, 6, 0];
spot.direction = [0, -1, 0];
spot.radius = 20;
spot.innerConeAngle = 0.3;   // radians — stored as cosine internally
spot.outerConeAngle = 0.5;
```

Proxy fields at a glance:

| Proxy | Fields |
|---|---|
| `PointLightProxy` | `position`, `color`, `intensity`, `radius` |
| `DirectionalLightProxy` | `direction`, `color`, `intensity` |
| `SpotLightProxy` | `position`, `direction`, `radius`, `innerConeAngle`, `outerConeAngle`, `color`, `intensity` |

Because the data is just a flat array, you can compute all of it in a Web Worker and hand the
`Float32Array` to the system with zero copies.

## 3. Upload each frame

```ts
lights.setRawData(lightData); // marks the buffer dirty (Zero-Copy — no clone)
lights.upload();              // writeBuffer to VRAM once per frame, only if dirty
```

## 4. Wire it into your shader and batch

The system emits the WGSL structs and helper functions your material shader needs, which you feed
into the shader builder's `lighting` option:

```ts
import { buildPBRShader } from 'null-graph/materials';

const shaderLayout = lights.getShaderLayout(); // { wgslStructs, wgslLogic } | undefined

const meshBatch = engine.createBatch(scenePass, {
    shaderCode: buildPBRShader({ useSkinning: false, lighting: shaderLayout }),
    strideFloats: 14,
    maxInstances: 1000,
    vertexLayouts: geom.layout.getWebGPUDescriptor(),
});

// Bind the light buffers to the batch (group 2 by default)
lights.applyToBatch(meshBatch, 2);
```

## 5. Clustered culling (compute stage)

For `'clustered-forward'`, the system builds a compute stage that bins lights into the cluster grid
every frame. Add it to your explicit render sequence **before** the pass that shades:

```ts
const lightBinning = lights.getComputeStage(camera); // ComputeStageNode

function frame() {
    lights.upload();
    engine.updateCamera(camera);
    engine.render([lightBinning, scenePass]); // compute, then draw
    requestAnimationFrame(frame);
}
```

The stage resets its atomic counter via `preClearBuffers` before each dispatch, so binning starts
clean every frame. Call `lights.destroy()` to release all GPU buffers when tearing down.

| Method | Description |
|---|---|
| `setRawData(data)` | Stage a `Float32Array` of light slices (Zero-Copy). |
| `upload()` | Flush staged data to VRAM if dirty. Call once per frame. |
| `getShaderLayout()` | WGSL structs/logic for the shader builder's `lighting` option. |
| `applyToBatch(batch, groupIndex?)` | Bind the light (and cluster) buffers to a render batch. Default group `2`. |
| `getComputeStage(camera)` | Build the clustered light-binning `ComputeStageNode`. Clustered technique only. |
| `destroy()` | Free all GPU buffers. |
