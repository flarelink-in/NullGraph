# Environment API Usage

The `null-graph/environment` module is the indirect-lighting runtime. It replaces the constant that
currently stands in for the *entire* indirect term in the PBR fragment —

```wgsl
let ambient = vec3<f32>(0.3) * albedo * ao;   // what you get without this module
```

— with the split-sum image-based lighting model: a **GGX-prefiltered environment cube** for
specular, an **SH-L2 projection** for diffuse, and a **BRDF integration LUT** to join them.

The load-bearing idea (design §5): indirect lighting is a **sibling contract** to direct lighting.
`LightSystem.getShaderLayout()` injects `getVisibleLightCount` / `getIncidentLight`;
`EnvironmentSystem.getShaderLayout()` injects `getIndirectDiffuse` / `getIndirectSpecular`. Both are
`{ wgslStructs, wgslLogic }` pairs composed into the material by the same mechanism, so **no
material file knows either one exists**.

- **Architecture:** [`lighting-ibl-design.md`](../lighting-ibl-design.md)
- **Buffer layouts, WGSL contracts, the material bridge:** [`environment-shader-reference.md`](./environment-shader-reference.md)
- **Phase 2 (probe volumes, reflection probes):** [`probes.md`](./probes.md) · [`lighting-probes-design.md`](../lighting-probes-design.md) — ✅ shipped

```ts
import { EnvironmentSystem } from 'null-graph/environment';
import { LightSystem } from 'null-graph/lights';
```

---

## 1. What you can build with this

Phase 1 gives you **one environment for the whole scene**, from any of three sources, plus the sky
that matches it. That covers more than it sounds like — here is the honest range.

### 1.1 The zero-asset outdoor scene

A procedural gradient sky with a sun disc. No HDR file, no download, no asset pipeline.

```ts
const env = new EnvironmentSystem(engine, {});
env.setProceduralSky({
    zenith:  [0.10, 0.22, 0.52],   // linear HDR, not sRGB
    horizon: [0.65, 0.72, 0.85],
    ground:  [0.28, 0.24, 0.20],
    sunDir:  [-0.4, -1.0, -0.35],  // TRAVEL direction: where the light GOES
    sunIntensity: 18,
});
```

The sun disc in the source cube is what makes the prefiltered specular have a real highlight and
the SH have a dominant direction. Turn `sunIntensity` up and the whole scene keys off it.

**Point the `LightSystem`'s directional light along the same `sunDir`** and the direct and indirect
halves agree. Nothing enforces that — the two systems do not know about each other — so it is worth
storing the sun direction once in your own state and feeding both.

### 1.2 A photographed environment (HDR / LDR equirect)

```ts
await env.setEquirect('/hdr/sunset_2k.hdr');       // linear data
await env.setEquirect('/img/street.jpg', { sRGB: true });   // sRGB-encoded photo
```

Any equirectangular (lat-long) image. `sRGB: true` makes the GPU linearize on sample — use it for
ordinary photos and JPEGs, leave it off for true HDR data that is already linear.

Failure is **loud**: a 404 throws rather than silently shipping a black environment.

### 1.3 An indoor / studio look

Same API, different content. A studio HDRI gives you soft box reflections; a small `intensity` with
a bright `zenith` and dark `ground` gives a neutral turntable look:

```ts
env.setProceduralSky({
    zenith: [1, 1, 1], horizon: [0.5, 0.5, 0.5], ground: [0.05, 0.05, 0.05],
    sunDir: [0, -1, 0], sunIntensity: 0,          // no disc — pure ambient dome
});
env.setIntensity(0.6);
```

`sunIntensity: 0` disables the disc entirely, which is the "overcast / lightbox" case.

### 1.4 A day/night cycle (animated sky)

```ts
const env = new EnvironmentSystem(engine, {
    updateMode: 'amortized',
    keepSourceCube: true,        // required — the prefilter re-reads the source
});
```

`'amortized'` keeps the bake alive and rebuilds **one prefilter mip per frame**, cycling. At the
default 6 mips the environment fully refreshes every 6 frames, which is imperceptible for a sun
moving over minutes, and it bounds the per-frame cost to a single 128²×6 dispatch instead of the
whole chain.

Per frame: call `setProceduralSky` again with the new sun direction, then splice `getBakeStages()`.

### 1.5 A rotatable environment

```ts
env.setRotation(performance.now() * 0.0001);
```

Rotates the diffuse SH response and the specular lookup **and the skybox** by the same yaw, so the
three cannot drift apart. Cheap — it is one float in a uniform, not a re-bake.

### 1.6 Worker-computed / precomputed irradiance

The SH set is 27 floats. You can compute it off-thread (or ship it from a previous session) and
hand it over, skipping the projection entirely:

```ts
env.setRawSH(coefficients);    // 9 vec3, envMath order — the setRawData contract
```

`projectSHL2FromCube` in `envMath` is the same math, pure and GPU-free, so a Worker can produce
exactly what the GPU kernel would.

### 1.7 Where Phase 1 stops

Everything above is **one global environment**. If your scene is entirely outdoors, or entirely one
room, that is the correct amount of machinery. The moment you need "dark inside, bright outside,"
you need Phase 2 — which is **shipped**: see [`probes.md`](./probes.md).

| Want | Where |
|---|---|
| Indirect light that **varies through the scene** | ✅ Phase 2 — irradiance volumes ([`probes.md`](./probes.md)) |
| **Local** reflections (a mirror that reflects *this room*) | ✅ Phase 2 — reflection probes |
| Reflections of **scene geometry**, not just the sky | ✅ Phase 2 — probe capture |
| Multi-bounce colour bleeding | ✅ Phase 2 — captures read the previous volume |
| Analytic **atmospheric scattering**, volumetric clouds | ◻ v2.0 ([tree §8](../architecture-tree.md#8-environment)) |
| **Fog**, weather, wind, water, terrain | ◻ planned — the Environment module's other branches |
| Screen-space reflections | Post's job, complementary to IBL |

Phase 2 layers on top: it keeps this system as its **fallback outside the volume** and as the
**ambient its captures shade against**, so nothing here becomes redundant.

---

## 2. The minimum working setup

```ts
import { EnvironmentSystem } from 'null-graph/environment';
import { LightSystem } from 'null-graph/lights';

const lights = new LightSystem(engine, {
    technique: 'clustered-forward',
    capacity: { point: 32, directional: 1, spot: 8 },
    clusterConfig: { grid: [16, 9, 24], maxLightsPerTile: 32 },
});

const env = new EnvironmentSystem(engine, {
    sourceSize: 512,        // cube face size the bake works from
    prefilterMips: 6,       // roughness levels: mip L = roughness L/(mips-1)
    prefilterSamples: 128,  // GGX importance samples per texel
    intensity: 1.0,
});
env.setProceduralSky({ /* §1.1 */ });

// 1. Bake once. In 'once' mode the stages retire after this frame.
engine.render([...env.getBakeStages()]);
env.copySHToParams();       // moves the projected SH into the uniform (§3.3)

// 2. Build the material with BOTH lighting layouts.
const indirect = env.getShaderLayout();
const batch = vis.createInstanceDrawBatch(scenePass, 'rock', 0, {
    material: { kind: 'pbr', instance: pbr, lighting: lights.getShaderLayout(), indirect },
});

// 3. Bind the COMBINED group 2 — this call, not LightSystem.applyToBatch (§3.2).
env.applyToBatch(batch, lights);

// 4. Every frame.
function frame() {
    engine.updateCamera(camera);
    lights.upload();
    engine.render([
        ...vis.getComputeStages(),
        scenePass,
        env.getSkyPass({ colorAttachments, depthStencilAttachment }),
        ...post.getPassSequence(),
    ]);
    requestAnimationFrame(frame);
}
```

---

## 3. The three things that are easy to get wrong

### 3.1 Order the sky pass **after** the opaque pass

The skybox draws with `depthCompare: 'less-equal'` and `depthWriteEnabled: false`, at depth
`0.999999`. It fills only where nothing was drawn. Put it **before** the scene and it is
overdrawn — put it after and it is a background, which is what you want and also fewer shaded
pixels.

### 3.2 `env.applyToBatch(batch, lights)` replaces `lights.applyToBatch(batch)`

Core WebGPU guarantees **four** bind groups and NullGraph spends all four (0 engine, 1 material,
2 lights, 3 shadows). Indirect lighting therefore shares **group 2** with the lights — bindings 3–6,
alongside the lights at 0–2.

Because a bind group is built in one call, exactly one owner may build group 2. `EnvironmentSystem`
is that owner when IBL is in play, which is why it takes the `LightSystem` as an argument:

```ts
env.applyToBatch(batch, lights);      // ✅ builds lights 0–2 AND environment 3–6
```

```ts
lights.applyToBatch(batch);           // ❌ overwrites group 2 — the env bindings vanish
env.applyToBatch(batch, lights);      //    (this throws, naming the problem)
```

### 3.3 Call `copySHToParams()` after the bake

The SH projection is a compute shader, so it writes a *storage buffer*. The fragment shader reads
`EnvParams`, a *uniform*. `copySHToParams()` is the GPU-side copy that bridges them — one
144-byte `copyBufferToBuffer`, no readback, no stall.

Skip it and the **specular half works while the diffuse half is black**, which reads as "SH is
broken" rather than "nothing copied it." Call it after `engine.render([...getBakeStages()])`, and
every frame in `'amortized'` mode. It is a no-op before the first bake.

---

## 4. `EnvironmentConfig`

| Field | Default | Notes |
|---|---|---|
| `sourceSize` | `512` | Cube face size the bake works from. **Power of two ≥ 8** — the mip chain and the 8×8 SH mip assume it. |
| `prefilterMips` | `6` | Roughness levels in the specular cube. Must not exceed `log2(sourceSize)+1`. Base face is `2^(mips+1)` px, so 6 → 128². |
| `prefilterSamples` | `128` | GGX importance samples per prefiltered texel. Bake-time cost only. |
| `lutSize` | `256` | BRDF LUT resolution. Scene-independent; 256² is plenty. |
| `intensity` | `1.0` | Multiplies both indirect terms. |
| `horizonFade` | `1.0` | Fades specular reflections pointing below the horizon. `0` disables. |
| `updateMode` | `'once'` | `'once'` retires the bake; `'amortized'` refreshes one mip per frame. |
| `keepSourceCube` | `false`* | Retain the source cube after baking. *Defaults `true` when `updateMode: 'amortized'`, which **requires** it. |
| `environmentGroup` | `2` | Must equal the `LightSystem`'s group. Anything else throws (§3.2). |

---

## 5. API reference

### Sources

| Method | Notes |
|---|---|
| `setProceduralSky(sky)` | Analytic gradient + sun disc. No asset. Rebuilds the bake chain. |
| `setEquirect(url, { sRGB? })` | `async`. Loads a lat-long image. Throws loudly on failure. |
| `setRawSH(coeffs)` | Adopt 27 precomputed SH floats, skipping projection. |

`ProceduralSkyConfig`: `zenith`, `horizon`, `ground` (linear HDR `[r,g,b]`), `sunDir` (travel
direction, normalized internally), `sunIntensity`, optional `sunCos` (default `0.9995` ≈ 1.8°).

### Frame graph

| Method | Notes |
|---|---|
| `getBakeStages()` | The compute stages to splice in. Empty after the first frame in `'once'` mode. Throws if no source is set. |
| `copySHToParams()` | GPU copy: projected SH → `EnvParams`. Call after the bake (§3.3). |
| `getSkyPass({ colorAttachments, depthStencilAttachment?, size? })` | The skybox pass. Created once, cached. Splice **after** the opaque pass. |

### Material bridge

| Method | Notes |
|---|---|
| `getShaderLayout()` | The `ShaderIndirectLayout` to pass as `indirect`. Throws if no source is set. |
| `applyToBatch(batch, lights)` | Builds the **combined** group-2 bind group. Use instead of `lights.applyToBatch` (§3.2). |

### Runtime knobs

| Method | Notes |
|---|---|
| `setRotation(yawRadians)` | Rotates diffuse, specular, and sky together. |
| `setIntensity(x)` | Scales both indirect terms. |
| `setSkyParams({ exposure?, yaw?, blurMip? })` | Sky-only. `blurMip` samples a rougher mip for a soft backdrop. |

### Introspection

`specularView`, `brdfView`, `sampler`, `paramsBuffer`, `shStorageBuffer`, `isBaked`, `destroy()`.

---

## 6. Materials that consume it

`indirect` is optional everywhere. Without it, every fragment is **byte-identical** to what it was
before this module existed.

| Material | Uses |
|---|---|
| `pbr` | Diffuse **and** specular — the full split-sum term |
| `lambert` | Diffuse only |
| `toon` | Diffuse only |
| `basic`, `emissive`, `matcap` | Nothing — unlit by definition |

Both the normal batch path and the GPU-culled visibility path accept it:

```ts
// visibility
vis.createInstanceDrawBatch(pass, 'rock', 0, {
    material: { kind: 'pbr', instance: pbr, lighting, indirect },
});

// direct
buildPBRShader({ useSkinning: false, lighting, indirect });
```

**With shadows**, compose all three — shadows decorate the *direct* layout, the environment supplies
the *indirect* one, and they do not interact:

```ts
material: { kind: 'pbr', instance: pbr, lighting: shadows.wrapLighting(lights), indirect }
```

---

## 7. Cost

| | |
|---|---|
| Bake (`'once'`) | source cube + `log2(size)` mips + `prefilterMips` prefilter dispatches + LUT + SH. One frame, then **zero**. |
| Frame (`'once'`) | Nothing. No dispatch, no upload, no allocation. |
| Frame (`'amortized'`) | One source-cube fill + one prefilter mip + the SH projection. |
| Shading | Two texture samples (prefiltered cube + LUT) and a 9-term SH dot product per pixel. |
| Memory | 512² source cube ≈ 8 MB (releasable) · 128²×6 specular ≈ 200 KB · 256² LUT ≈ 256 KB · `EnvParams` 176 B |

---

## 8. Errors

Every failure names the offender and the way out.

| Condition | Message |
|---|---|
| Shading before a source is set | `getShaderLayout() called before a source was set; call setEquirect() or setProceduralSky() first.` |
| `sourceSize` not a power of two | `sourceSize <n> must be a power of two ≥ 8; the mip chain and the SH projection mip assume it.` |
| Too many prefilter mips | `prefilterMips <n> exceeds log2(sourceSize)+1 = <m>.` |
| Wrong bind group | `environmentGroup <n> must equal the LightSystem's group (2); core WebGPU guarantees only 4 bind groups and all four are in use.` |
| Group 2 already bound | `batch already has a group-2 bind group from LightSystem.applyToBatch(); use EnvironmentSystem.applyToBatch(batch, lights) to build the combined group instead.` |
| Equirect failed to load | `failed to load equirect "<url>" (...); falling back would silently ship a black environment — set a procedural sky explicitly instead.` |
| Amortized without a retained source | `updateMode "amortized" requires keepSourceCube: true; the prefilter reads the source every refresh.` |

---

## 9. Testing

`npm run test:environment` — 12 headless checks (U1–U10 + config validation): cube-face convention,
equirect round-trip, solid-angle exactness, SH packing order, ringing clamp, constant-environment
energy, rotation consistency, BRDF LUT endpoints, WGSL parse + binding budget, the material hook,
and the backtick guard.

GPU integration (white furnace, fireflies, mirror fidelity, rotation, sky depth test) needs a
browser harness — see [`lighting-ibl-design.md`](../lighting-ibl-design.md) §10 T1–T6.
