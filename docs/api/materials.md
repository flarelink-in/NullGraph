# Materials API Usage

NullGraph ships a family of materials, from cheap unlit shading up to Cook-Torrance PBR. Each one
comes in two halves that you combine:

1. A **shader builder** (`build*Shader`) that assembles WGSL for the batch's pipeline. Builders take
   a `ShaderConfig` so you can toggle skinning (`useSkinning`), inject a lighting layout
   (`lighting`), and set material-specific options.
2. A **material class** that owns a uniform buffer + textures and binds them to the batch via
   `applyToBatch(batch)`.

```ts
import {
    buildPBRShader,     StandardPBRMaterial,
    buildBasicShader,   BasicMaterial,
    buildLambertShader, LambertMaterial,
    buildToonShader,    ToonMaterial,
    buildEmissiveShader, EmissiveMaterial,
    buildMatcapShader,  MatcapMaterial,
} from 'null-graph/materials';
```

> **Note:** the shader builders are plain functions — call `buildPBRShader({ ... })`, not
> `PBRShaderBuilder.build(...)`. (The source file was renamed from `ShaderBuilder.ts` to
> `PBRShaderBuilder.ts`, but the exported function name is unchanged. See
> [Deprecated API §7](../deprecated-api.md).)

---

## How materials work

Every material follows the same shape, so once you learn one you know them all:

```ts
// 1. Build the WGSL for the batch (skinning + optional lighting)
const shader = buildXShader({ useSkinning: false, lighting: lights?.getShaderLayout() });

// 2. Create the batch with that shader
const batch = engine.createBatch(pass, {
    shaderCode: shader,
    strideFloats: 14,
    maxInstances: 1000,
    vertexLayouts: geom.layout.getWebGPUDescriptor(),
});

// 3. Create the material and bind it (group index 1 by default)
const mat = new XMaterial(engine, { /* options */ });
mat.applyToBatch(batch);

// 4. Update params at runtime whenever you like
mat.updateUniforms({ /* changed fields */ });

// 5. Free the uniform buffer when done
mat.destroy();
```

**Texture fallbacks.** Every texture option is optional (except `MatcapMaterial.matcapMap`). When
omitted, the material substitutes a safe fallback from `engine.textureManager`
(`fallbackWhite` for color/packed maps, `fallbackNormal` for normal maps), and sets an internal
`hasX` flag so the shader ignores the fetch. A material never binds `null`.

**Updating parameters (the DOD way).** Materials keep their scalar params in a single small,
pre-allocated GPU uniform buffer. Mutating a public field (e.g. `mat.baseColor = [...]`) does **not**
touch the GPU on its own — call `updateUniforms(...)`, which re-packs the buffer and issues one
`writeBuffer`. There is no per-object allocation and no bind-group rebuild, so it's cheap to call
every frame. `applyToBatch` does **not** need to be called again after `updateUniforms`.

---

## Physically Based Rendering (Cook-Torrance PBR)

`StandardPBRMaterial` implements the Cook-Torrance BRDF with Albedo, Normal, and a **packed** map
carrying Ambient Occlusion / Roughness / Metallic.

```ts
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

const pbr = new StandardPBRMaterial(engine, {
    albedoMap: albedoView,
    normalMap: normalView,
    packedMap: armView,
    packedMapFormat: "ARM",        // 'ARM' (default) | 'MRAO'
    baseColor: [1.0, 1.0, 1.0, 1.0],
    metallicMultiplier: 1.0,
    roughnessMultiplier: 1.0,
});

pbr.applyToBatch(meshBatch);
```

### Packed map format: `ARM` vs `MRAO`

The single "packed" texture stores three grayscale channels. Because different DCC tools and asset
stores pack them in different channel orders, `packedMapFormat` tells the shader how to read the RGB
channels — no need to re-author the texture.

| Format | R channel | G channel | B channel |
|---|---|---|---|
| `'ARM'` (default) | **A**mbient Occlusion | **R**oughness | **M**etallic |
| `'MRAO'` | **M**etallic | **R**oughness | **A**mbient Occlusion |

Roughness sits in G for both, so ARM and MRAO differ only in whether AO and Metallic are swapped
(R↔B). Pick the one matching your texture; the fragment shader branches on the flag at runtime:

```wgsl
// packedData = textureSample(texMRAO, samp, uv);
if (material.mapFormat < 0.5) {          // ARM
    ao = packedData.r; roughness *= packedData.g; metallic *= packedData.b;
} else {                                  // MRAO
    metallic *= packedData.r; roughness *= packedData.g; ao = packedData.b;
}
```

`metallicMultiplier` and `roughnessMultiplier` scale the sampled values (so a fully white packed map
plus multiplier `1.0` gives raw metallic/roughness of `1.0`).

### PBR options

| Option | Type | Default | Description |
|---|---|---|---|
| `albedoMap` | `GPUTextureView` | `fallbackWhite` | Base color texture. |
| `normalMap` | `GPUTextureView` | `fallbackNormal` | Tangent-space normal map. |
| `packedMap` | `GPUTextureView` | `fallbackWhite` | Packed AO/Roughness/Metallic map. |
| `packedMapFormat` | `'ARM' \| 'MRAO'` | `'ARM'` | Channel layout of `packedMap`. |
| `baseColor` | `[r,g,b,a]` | `[1,1,1,1]` | Multiplied onto albedo; alpha is output alpha. |
| `metallicMultiplier` | `number` | `1.0` | Scales sampled metallic. |
| `roughnessMultiplier` | `number` | `1.0` | Scales sampled roughness. |

### Updating PBR params

PBR uses a **positional** update signature (the others use an options object):

```ts
pbr.updateUniforms(
    [1.0, 0.4, 0.1, 1.0], // baseColor
    0.9,                   // metallic
    0.3,                   // roughness
);
```

> Caveats: `updateUniforms` on PBR updates only `baseColor`, `metallic`, and `roughness`. It does
> **not** change the bound texture views, and it re-writes the packed-map format flag as `ARM`. If
> you rely on `MRAO`, prefer setting metallic/roughness through the multipliers at construction, or
> recreate the material to change textures/format.

### PBR bind-group layout (group 1)

| Binding | Resource |
|---|---|
| 0 | sampler |
| 1 | albedo texture |
| 2 | normal texture |
| 3 | packed (ARM/MRAO) texture |
| 4 | material uniform buffer (`baseColor`, `metallic`, `roughness`, `mapFormat`) |

---

## `BasicMaterial` — unlit

The cheapest material: no lights, no normal reads. Flat color or a single albedo texture, with
optional UV tiling for large surfaces.

```ts
const mat = new BasicMaterial(engine, {
    baseColor: [0.1, 0.5, 0.9, 1.0],
    albedoMap: albedoView,   // optional
    uvTiling: [4, 4],        // optional — tile the albedo across the surface
});

mat.updateUniforms({ baseColor: [1, 0, 0, 1], uvTiling: [8, 8] });
```

| Option / field | Type | Default | Notes |
|---|---|---|---|
| `baseColor` | `[r,g,b,a]` | `[1,1,1,1]` | Tint / flat color. |
| `albedoMap` | `GPUTextureView` | — (flat color) | Optional base texture. |
| `uvTiling` | `[u,v]` | `[1,1]` | Repeats the albedo map. |

`updateUniforms({ baseColor?, uvTiling? })` — omitted fields keep their current value.
**Bindings (group 1):** `0` sampler, `1` albedo, `2` uniform.

---

## `LambertMaterial` — cheap diffuse

Classic N·L diffuse with an ambient floor. The right pick for thousands of lit objects that don't
need specular. Pass a lighting layout into the builder; `useHalfLambert` softens the terminator.

```ts
const shader = buildLambertShader({ useSkinning: false, useHalfLambert: true, lighting });
const mat = new LambertMaterial(engine, {
    baseColor: [1, 1, 1, 1],
    albedoMap: albedoView,   // optional
    ambient: 0.15,           // prevents pitch-black shadows
});

mat.updateUniforms({ ambient: 0.25 });
```

| Option / field | Type | Default | Notes |
|---|---|---|---|
| `baseColor` | `[r,g,b,a]` | `[1,1,1,1]` | Diffuse albedo tint. |
| `albedoMap` | `GPUTextureView` | — (flat color) | Optional base texture. |
| `ambient` | `number` | `0.15` | Ambient light floor. |

`updateUniforms({ baseColor?, ambient? })` — omitted fields keep their current value.
**Bindings (group 1):** `0` sampler, `1` albedo, `2` uniform.

---

## `ToonMaterial` — cel shading

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

mat.updateUniforms({ baseColor: [0.9, 0.2, 0.3, 1.0], rampSteps: 3 });
```

| Option / field | Type | Default | Notes |
|---|---|---|---|
| `baseColor` | `[r,g,b,a]` | `[1,1,1,1]` | Surface color. |
| `rampSteps` | `number` | `3` | Number of shading bands. |
| `rimLightColor` | `[r,g,b]` | `[1,1,1]` | Fresnel rim tint. |
| `rimLightPower` | `number` | `2.0` | Rim falloff exponent. |
| `specularThreshold` | `number` | `0.5` | Cutoff for the toon specular blob. |
| `albedoMap` / `normalMap` | `GPUTextureView` | fallbacks | Optional textures. |

`updateUniforms({ baseColor?, rampSteps?, rimLightColor?, rimLightPower?, specularThreshold? })`.

> **Gotcha:** unlike the other materials, Toon's `updateUniforms` treats `baseColor` as
> **required** — if you omit it, the color resets to white `[1,1,1,1]`. Always pass `baseColor`
> when calling it.

**Bindings (group 1):** `0` sampler, `1` albedo, `2` normal, `3` uniform.

---

## `EmissiveMaterial` — glow / HDR

Self-illuminating surfaces, optionally pulsing over time. `emissiveIntensity` can exceed `1.0` to
feed a [Bloom](./post.md) pipeline. The pulse animates on the GPU from the camera's time slot — no
per-frame uniform write needed.

```ts
const mat = new EmissiveMaterial(engine, {
    baseColor: [0, 0, 0, 1],
    emissiveColor: [0.2, 0.9, 1.0],
    emissiveIntensity: 4.0,  // > 1.0 for HDR/Bloom
    emissiveMap: maskView,   // optional — masks which pixels glow
    pulseSpeed: 3.0,         // radians per simTime second
    pulseAmount: 0.5,        // 0 = steady, 1 = full pulse
});

mat.updateUniforms({ emissiveIntensity: 6.0 });
```

| Option / field | Type | Default | Notes |
|---|---|---|---|
| `baseColor` | `[r,g,b,a]` | `[0,0,0,1]` | Non-glowing base. |
| `emissiveColor` | `[r,g,b]` | `[1,1,1]` | Glow color, added on top. |
| `emissiveIntensity` | `number` | `1.0` | Glow strength; `>1.0` for HDR/Bloom. |
| `pulseSpeed` | `number` | `2.0` | Pulse rate (radians/sec). |
| `pulseAmount` | `number` | `0.0` | `0` steady … `1` full pulse. |
| `albedoMap` / `emissiveMap` | `GPUTextureView` | `fallbackWhite` | Optional textures. |

`updateUniforms({ baseColor?, emissiveColor?, emissiveIntensity?, pulseSpeed?, pulseAmount? })` —
omitted fields keep their current value.

> Pulsing reads `simTime` from the camera uniform — call
> `camera.updateTime(performance.now() * 0.001)` each frame before `engine.updateCamera(camera)`.

**Bindings (group 1):** `0` sampler, `1` albedo, `2` emissive, `3` uniform.

---

## `MatcapMaterial` — baked lighting

Material-capture shading: the whole lighting environment is baked into one sphere texture, looked up
by the view-space normal. Costs the same as an unlit material but reads as fully lit — great for
sculpt viewers, previews, and stylized crowds.

```ts
const mat = new MatcapMaterial(engine, {
    matcapMap: matcapView,   // required — the sphere-lit capture
    albedoMap: albedoView,   // optional — multiplied under the matcap
    baseColor: [1, 1, 1, 1],
});

mat.updateUniforms({ baseColor: [0.8, 0.9, 1.0, 1.0] });
```

| Option / field | Type | Default | Notes |
|---|---|---|---|
| `matcapMap` | `GPUTextureView` | **required** | The sphere-lit capture. |
| `albedoMap` | `GPUTextureView` | — | Optional, multiplied under the matcap. |
| `baseColor` | `[r,g,b,a]` | `[1,1,1,1]` | Tint. |

`updateUniforms({ baseColor? })`.
**Bindings (group 1):** `0` sampler, `1` matcap, `2` albedo, `3` uniform.

---

## Choosing a material

| Material | Lit? | Cost | Runtime-updatable params | Best for |
|---|---|---|---|---|
| `BasicMaterial` | No | Lowest | `baseColor`, `uvTiling` | Flat colors, UI/debug, unlit props |
| `MatcapMaterial` | Baked | Very low | `baseColor` | Stylized/preview shading, no light setup |
| `LambertMaterial` | Yes (diffuse) | Low | `baseColor`, `ambient` | Massive counts of simple lit objects |
| `ToonMaterial` | Yes | Low–med | `baseColor`, `rampSteps`, rim, specular | Cel-shaded / NPR looks |
| `EmissiveMaterial` | Self | Low | color, intensity, pulse | Glowing cores, HDR sources for Bloom |
| `StandardPBRMaterial` | Yes (PBR) | Highest | `baseColor`, `metallic`, `roughness` | Photorealistic metal/dielectric surfaces |

To light any of the lit materials (`Lambert`, `Toon`, `PBR`), pass a `lighting` layout from the
[Lights](./lights.md) system into the builder and `applyToBatch` the `LightSystem` onto the same
batch. Omit `lighting` on the PBR builder to fall back to a single hardcoded key light.
