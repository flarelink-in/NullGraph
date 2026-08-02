# Materials API Usage

NullGraph ships a family of materials, from cheap unlit shading up to Cook-Torrance PBR. Each one
comes in two halves that you combine:

1. A **shader builder** (`build*Shader`) that assembles WGSL for the batch's pipeline. Builders take
   a `ShaderConfig` so you can toggle skinning (`useSkinning`), inject a lighting layout
   (`lighting`), and set material-specific options.
2. A **material class** that owns the uniform buffer + textures and binds them to the batch via
   `applyToBatch(batch)`.

```ts
import {
    buildPBRShader,    StandardPBRMaterial,
    buildBasicShader,  BasicMaterial,
    buildLambertShader, LambertMaterial,
    buildToonShader,   ToonMaterial,
    buildEmissiveShader, EmissiveMaterial,
    buildMatcapShader, MatcapMaterial,
} from 'null-graph/materials';
```

> **Note:** the shader builders are plain functions — call `buildPBRShader({ ... })`, not
> `PBRShaderBuilder.build(...)`. (The source file was renamed from `ShaderBuilder.ts` to
> `PBRShaderBuilder.ts`, but the exported function name is unchanged. See
> [Deprecated API §7](../deprecated-api.md).)

---

## Physically Based Rendering (Cook-Torrance PBR)
Achieve photorealistic lighting with `StandardPBRMaterial`. Built on the Cook-Torrance BRDF, it
supports Albedo, Normal, and packed ARM (Ambient Occlusion, Roughness, Metallic) maps out of the box.

```ts
// 1. Generate a dynamic shader optimized for your mesh needs
const shaderCode = buildPBRShader({
    useSkinning: true,
    lighting: lightSystem.getShaderLayout(), // optional — omit for the built-in fallback light
});

const meshBatch = engine.createBatch(mainPass, {
    shaderCode,
    strideFloats: 14,
    maxInstances: 1000,
    vertexLayouts: geom.layout.getWebGPUDescriptor(),
});

// 2. Create the PBR Material with loaded textures
const pbrMaterial = new StandardPBRMaterial(engine, {
    albedoMap: albedoView,
    normalMap: normalView,
    packedMap: armView,         // AO, Roughness, Metallic
    packedMapFormat: "ARM",
    baseColor: [1.0, 1.0, 1.0, 1.0],
    metallicMultiplier: 1.0,
    roughnessMultiplier: 1.0
});

// 3. Apply it to your batch
pbrMaterial.applyToBatch(meshBatch);
```

## The full material set

Every material follows the same `new XMaterial(engine, options)` → `applyToBatch(batch)` pattern,
paired with its matching `buildXShader({ useSkinning, lighting? })`.

### `BasicMaterial` — unlit
The cheapest material: no lights, no normal reads. Flat color or a single albedo texture.

```ts
const mat = new BasicMaterial(engine, {
    baseColor: [0.1, 0.5, 0.9, 1.0],
    albedoMap: albedoView,   // optional
    uvTiling: [4, 4],        // optional — tile the albedo across large surfaces
});
```

### `LambertMaterial` — cheap diffuse
Classic N·L diffuse with an ambient floor. The right pick for thousands of lit objects that don't
need specular. Supports half-Lambert via the builder's `useHalfLambert` option.

```ts
const shader = buildLambertShader({ useSkinning: false, useHalfLambert: true, lighting });
const mat = new LambertMaterial(engine, {
    baseColor: [1, 1, 1, 1],
    albedoMap: albedoView,   // optional
    ambient: 0.15,           // prevents pitch-black shadows
});
```

### `ToonMaterial` — cel shading
Banded (stepped) lighting with rim light and a specular threshold. The builder's `useSmoothRamp`
option swaps hard bands for a smooth ramp.

```ts
const shader = buildToonShader({ useSkinning: false, useSmoothRamp: false, lighting });
const mat = new ToonMaterial(engine, {
    baseColor: [0.9, 0.2, 0.3, 1.0],
    rampSteps: 4,
    rimLightColor: [1, 1, 1],
    rimLightPower: 2.0,
    specularThreshold: 0.7,
    albedoMap: albedoView,   // optional
    normalMap: normalView,   // optional
});
```

### `EmissiveMaterial` — glow / HDR
Self-illuminating surfaces, optionally pulsing over time. `emissiveIntensity` can exceed `1.0` to
feed a [Bloom](./post.md) pipeline.

```ts
const mat = new EmissiveMaterial(engine, {
    baseColor: [0, 0, 0, 1],
    emissiveColor: [0.2, 0.9, 1.0],
    emissiveIntensity: 4.0,  // > 1.0 for HDR/Bloom
    emissiveMap: maskView,   // optional — masks which pixels glow
    pulseSpeed: 3.0,         // radians per simTime second
    pulseAmount: 0.5,        // 0 = steady, 1 = full pulse
});
```

> Pulsing reads `simTime` from the camera uniform — call `camera.updateTime(performance.now() * 0.001)`
> each frame before `engine.updateCamera(camera)`.

### `MatcapMaterial` — baked lighting
Material-capture shading: the whole lighting environment is baked into one sphere texture. No lights
needed, extremely cheap, great for previews and stylized looks.

```ts
const mat = new MatcapMaterial(engine, {
    matcapMap: matcapView,   // required — the sphere-lit capture
    albedoMap: albedoView,   // optional — multiplied under the matcap
    baseColor: [1, 1, 1, 1],
});
```

---

## Choosing a material

| Material | Lit? | Cost | Best for |
|---|---|---|---|
| `BasicMaterial` | No | Lowest | Flat colors, UI/debug, unlit props |
| `MatcapMaterial` | Baked | Very low | Stylized/preview shading, no light setup |
| `LambertMaterial` | Yes (diffuse) | Low | Massive counts of simple lit objects |
| `ToonMaterial` | Yes | Low–med | Cel-shaded / NPR looks |
| `EmissiveMaterial` | Self | Low | Glowing cores, HDR sources for Bloom |
| `StandardPBRMaterial` | Yes (PBR) | Highest | Photorealistic metal/dielectric surfaces |

To light any of the lit materials, pass a `lighting` layout from the
[Lights](./lights.md) system into the builder and `applyToBatch` the `LightSystem` to the same batch.
