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
- **Sky sources (`setSky`, `PreethamSky`, per-pixel skybox):** [`environment-sky-design.md`](../environment-sky-design.md) — ✅ shipped in 1.1, §1.3b below
- **Time of day (sun/moon → sky, lights, shadows, exposure):** [`time-of-day.md`](./time-of-day.md) · [`environment-timeofday-design.md`](../environment-timeofday-design.md) — ✅ shipped
- **Night sky (`NightSky`, `SkyComposite`, `StarField`):** [`night-sky.md`](./night-sky.md) · [`environment-night-design.md`](../environment-night-design.md) — ✅ shipped
- **Physical atmosphere (`Atmosphere`, `AtmosphereSky`):** [`atmosphere.md`](./atmosphere.md) · [`environment-atmosphere-design.md`](../environment-atmosphere-design.md) — ✅ shipped

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
halves agree. For a static scene, storing that direction once is enough. For an animated sun, use
[`TimeOfDay`](./time-of-day.md): it writes one computed direction into the sky, direct-light,
shadow, and exposure contracts so those copies cannot drift.

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

### 1.3b Analytic daylight, and skies as objects

A sky is a **`SkySource`** — an object you hand to `setSky`, not a method on the system. The source
supplies one WGSL function, `sampleSky(dir)`, which the module injects into **both** the IBL bake
and the skybox fragment. The sky you are lit by and the sky you see are therefore the same function
by construction ([`environment-sky-design.md`](../environment-sky-design.md)).

```ts
import { EnvironmentSystem, PreethamSky } from 'null-graph/environment';

env.setSky(new PreethamSky({
    turbidity: 2.8,               // 1.8 = pristine alpine, 10 = urban haze
    groundAlbedo: [0.15, 0.14, 0.12],
    sunDir: [-0.4, -1.0, -0.35],  // TRAVEL direction, as everywhere in the engine
    sunAngularRadius: 0.266,      // degrees — the TRUE sun
}));

env.sky.setSunDirection(newDir);  // params only, no rebuild — call it per frame if you like
```

Preetham et al. 1999: the sky's luminance gradient, horizon brightening and circumsolar glow are
**derived** from turbidity and a sun direction rather than authored, which is what lets one number
drive a whole day instead of nine colour ramps. About 30 constants and 40 flops.

Its failure mode is documented rather than hidden: the fit **degrades below roughly 5° solar
elevation** and the model can dip negative near the horizon at high turbidity. The shipped path
clamps to non-negative, and sunset fidelity is what the later phases in the series are for.

**Why `sunAngularRadius: 0.266` is newly meaningful.** The real sun is 0.53° across. Until the
skybox evaluated `sampleSky` per pixel it sampled the 128²-per-face prefiltered cube — **0.70° per
texel** — so a physically sized disc was sub-texel and the shipped `sunCos` default drew one 6.8×
too wide on purpose. Per-pixel evaluation is the default now, and the GPU suite measures the disc
back at 0.265°.

**Writing your own source** is the whole extension story — no engine edit, no `if (procedural)`
branch, no new kernel:

```ts
class MySky implements SkySource {
    readonly name = 'MySky';
    readonly paramFloats = 8;
    getShaderLayout(base: number): ShaderSkyLayout { /* WGSL reading skyParams[base + i] */ }
    writeParams(out: Float32Array, base: number): void { /* out-parameter, no allocation */ }
}
```

Hosek–Wilkie is better than Preetham at sunset and is deliberately **not** shipped — it is ~2,700
lines of fitted coefficients, which is data rather than design. Because of this seam, you can add
it yourself without touching the module. See
[`environment-shader-reference.md`](./environment-shader-reference.md) §3.4 for the WGSL contract.

### 1.3c Where the skybox gets its colour

```ts
env.setSkyParams({ cheapSky: 0 });   // DEFAULT — evaluate sampleSky per pixel
env.setSkyParams({ cheapSky: 1, blurMip: 3 });   // sample the prefiltered cube instead
```

| | `cheapSky: 0` (default) | `cheapSky: 1` |
|---|---|---|
| Resolution | per pixel | 0.70°/texel at `prefilterMips: 6` |
| Cost | the source's evaluator, on background pixels only | one cube sample |
| `blurMip` | **ignored** | selects the roughness mip |
| Use it for | anything with detail: a sun disc, a horizon, a sharp imported sky | a deliberately soft backdrop, or a source with no analytic form |

The two agree to the last bit where they overlap — `cheapSky: 1` reproduces the pre-1.1 image
exactly (GPU test T4). **`blurMip` only has meaning in cheap mode**: under the default there is no
mip chain to select, so setting it alone does nothing. If you were using `blurMip` for a soft
background, pass `cheapSky: 1` alongside it.

`EquirectSky` benefits without any flag: its per-pixel path samples the **source** cube at mip 0
(512² by default, 0.176°/texel) instead of the prefiltered 128², a 4× angular improvement from
nothing but reading a different texture. That path needs `keepSourceCube: true`, and says so
loudly if it does not have it.

### 1.4 A complete day/night cycle

```ts
import {
    EnvironmentSystem, NightSky, PreethamSky, SkyComposite, StarField, TimeOfDay,
} from 'null-graph/environment';

const env = new EnvironmentSystem(engine, {
    updateMode: 'amortized',
    keepSourceCube: true,        // required — the prefilter re-reads the source
});

env.setSky(new SkyComposite({
    day: new PreethamSky({ turbidity: 2.8 }),
    night: new NightSky({ latitudeDeg: 45 }),
}));

const stars = new StarField(engine, { latitudeDeg: 45, maxStars: 9100 });

const tod = new TimeOfDay({
    mode: 'artistic',
    timeScale: 60,
    rebakeThresholdDeg: 0.25,
    exposureCoupling: 'off',
});

const skyPass = env.getSkyPass({
    colorAttachments,
    depthStencilAttachment,
    size: [width, height],
    targetFormat: 'rgba16float',
});
stars.attach(env);                    // second additive batch in the same pass
```

Per frame, update the clock before asking the environment for bake stages:

```ts
tod.advance(dt);
tod.applyTo({ environment: env, stars });

const bakeStages = env.getBakeStages();  // shared empty array while clean
engine.render([...bakeStages, ...restBeforeSky, skyPass, ...post.getPassSequence()]);
if (bakeStages.length > 0) env.copySHToParams();
```

`'amortized'` refreshes the source cube, **one prefilter mip**, and SH on each active bake frame.
At the default 6 mips a triggered environment fully refreshes over 6 frames. It does **not** run
unconditionally: `TimeOfDay` invalidates the bake after the sun, or a night-weighted moon, has
accumulated `rebakeThresholdDeg` of angular motion. Daytime lunar motion is suppressed by
`1 - dayFactor`. At the default `0.25°`, solar motion triggers roughly once per minute in real time
and about every four frames at 1000× speed.

Do not call `setProceduralSky` or `setSky` per frame. A source is installed once; `TimeOfDay`
updates it through the optional full `SkySource.setCelestialState` handoff, re-serializes its
parameters, and declares bake dirtiness. See [`time-of-day.md`](./time-of-day.md) for lights,
shadows, exposure, and astronomical mode, and [`night-sky.md`](./night-sky.md) for catalogue and
resize handling.

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
broken" rather than "nothing copied it." Call it after `engine.render([...getBakeStages()])` when
that call returned one or more stages. In amortized mode that means each **active bake frame**, not
each clean frame. It is a no-op before the first bake.

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
| `updateMode` | `'once'` | `'once'` retires the bake; `'amortized'` refreshes one mip per active invalidation frame and is idle while clean. |
| `keepSourceCube` | `false`* | Retain the source cube after baking. *Defaults `true` when `updateMode: 'amortized'`, which **requires** it. |
| `environmentGroup` | `2` | Must equal the `LightSystem`'s group. Anything else throws (§3.2). |

---

## 5. API reference

### Sources

| Method | Notes |
|---|---|
| `setSky(source)` | Install a `SkySource`. The mechanism; the two below forward to it. Rebuilds the bake chain. Must precede the first `getSkyPass()`. |
| `setProceduralSky(sky)` | `setSky(new GradientSky(sky))`. Analytic gradient + sun disc, no asset. |
| `setEquirect(url, { sRGB? })` | `async`. Loads a lat-long image, then `setSky(new EquirectSky(...))`. Throws loudly on failure. Needs `keepSourceCube: true`. |
| `setRawSH(coeffs)` | Adopt 27 precomputed SH floats, skipping projection. |
| `sky` | The installed source, typed — `env.sky.setSunDirection(dir)`. Throws before `setSky`. |

**Shipped sources and decorators:**

| Source | Config | Notes |
|---|---|---|
| `GradientSky` | `ProceduralSkyConfig` | The default. `zenith` / `horizon` / `ground` (linear HDR), `sunDir` (travel direction), `sunIntensity`, optional `sunCos` (default `0.9995` ≈ 1.8° radius). ~20 flops; numerically identical to the pre-1.1 sky. |
| `EquirectSky` | `{ equirect: GPUTextureView }` | Bake reads the 2D asset, skybox reads the source cube at mip 0. Requires `keepSourceCube: true`. |
| `PreethamSky` | `{ turbidity?, groundAlbedo?, sunDir?, sunAngularRadius?, sunDiscIntensity? }` | Analytic daylight (§1.3b). `setSunDirection` / `setTurbidity` rewrite params without a rebuild. |
| `AtmosphereSky` | `(atmosphere, { sunAngularRadius?, sunDiscIntensity? })` | Physical Rayleigh/Mie/ozone daylight backed by `Atmosphere` LUTs. Splice `atmosphere.getStages()` before the environment bake and sky pass. See [`atmosphere.md`](./atmosphere.md). |
| `NightSky` | `NightSkyConfig` | Resolved moon, earthshine, night dome, Milky Way, airglow, and horizon glow. Normally the night child of `SkyComposite`. |
| `SkyComposite` | `{ day, night, dayFactor? }` | Renames and blends two analytic `SkySource`s into one evaluator. `TimeOfDay` drives `dayFactor`. |

`StarField` is not a `SkySource`: it is the separate unresolved point-source draw. See
[`night-sky.md`](./night-sky.md).

### Frame graph

| Method | Notes |
|---|---|
| `getBakeStages()` | The compute stages to splice in. Returns a shared empty array while clean. After an amortized invalidation, emits source + one prefilter mip + SH per call until every mip is refreshed. Throws if no source is set. |
| `invalidateBake()` | Declare that source parameters outside `TimeOfDay` changed. In `'once'` mode, re-arms the full bake; in `'amortized'`, starts one bounded all-mip refresh cycle. Repeated calls during a cycle coalesce. |
| `copySHToParams()` | GPU copy: projected SH → `EnvParams`. Call after the bake (§3.3). |
| `getSkyPass({ colorAttachments, depthStencilAttachment?, size?, targetFormat? })` | The skybox pass. Created once, cached. Splice **after** the opaque pass. |
| `getCreatedSkyPass()` | Return the cached pass for an additive decorator such as `StarField`; throws if it has not been created. |
| `skyTargetFormat` | Colour format frozen into the cached sky-pass pipelines. |

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
| `setSkyParams({ exposure?, yaw?, blurMip?, cheapSky? })` | Every field **persists** — a later call that omits one leaves it alone. `yaw` is the environment rotation, so setting it here is equivalent to `setRotation` and keeps the SH, specular and skybox locked. `blurMip` applies only when `cheapSky` is 1 (§1.3c). |

Changes outside `TimeOfDay` must declare their own bake dirtiness. For example, changing
Preetham turbidity updates source-owned CPU state; re-serialize it and invalidate the IBL bake:

```ts
const sky = new PreethamSky({ turbidity: 2.8 });
env.setSky(sky);

sky.setTurbidity(5.0);
env.setSkyParams({});      // upload the source's changed params
env.invalidateBake();     // refresh source cube, prefilter mips, and SH
```

If the same sky is driven by `TimeOfDay`, its `turbidity` config should match the
`PreethamSky` value so the directional-light radiance and visible disc use the same extinction
curve. Runtime turbidity changes need the application to keep those two authored values aligned.

### Introspection

`specularView`, `sourceView`, `brdfView`, `sampler`, `paramsBuffer`, `shStorageBuffer`, `isBaked`,
`bakeInvalidated`, `sky`, `destroy()`.

`bakeInvalidated` is true before the initial bake and while an amortized invalidation still has
prefilter mips outstanding. `TimeOfDay.envDirty` is the celestial trigger for the current update;
the environment flag describes the whole multi-frame refresh cycle.

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
| Active bake frame (`'amortized'`) | One source-cube fill + one prefilter mip + the SH projection. |
| Clean frame (`'amortized'`) | Nothing. `getBakeStages()` returns a shared empty array until `invalidateBake()`. |
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

**Sky sources** — all build-time, none in the frame loop:

| Condition | Message |
|---|---|
| A cube-backed source on a system that releases the cube | `EnvironmentSystem: EquirectSky samples the source cube, but keepSourceCube is false — the cube is released after the bake. Construct with { keepSourceCube: true }.` |
| `setSky` after the skybox pipeline was compiled | `EnvironmentSystem: setSky(<name>) must be called before the first getSkyPass(); the compiled skybox pipeline has the previous source layout.` |
| `paramFloats` changed since `setSky` | `SkySource "<name>" declares paramFloats=<n>, but the sky batch was created with strideFloats=<m>. The stride is fixed at batch creation — paramFloats must not change after setSky().` |
| A source whose `wgslLogic` lacks `sampleSky` | `SkySource "<name>" does not define sampleSky(dir: vec3<f32>) -> vec3<f32> (its <site> wgslLogic lacks it).` |
| Generated WGSL contains a backtick | `SkySource "<name>" emitted a backtick, which terminates the template literal carrying it. Use single quotes in WGSL comments.` |
| `getSkyPass` / `env.sky` before a source | `getSkyPass() called before a sky source was set; call setSky() (or setProceduralSky/setEquirect) first.` |
| `StarField.attach(env)` before a sky pass exists | `EnvironmentSystem: no sky pass exists yet; call getSkyPass(target) before attaching a StarField, or pass target to StarField.attach().` |

---

## 9. Testing

`npm run test:environment` runs all six headless suites (IBL, probes, sky, time of day, night,
physical atmosphere).

- **`npm run test:sky`** — 14 headless checks (U1–U13 + the §8 error table). The gradient refactor
  is numerically inert to 1e-6; the bake kernel and the skybox carry byte-identical `sampleSky`
  bodies; every composed module parses; Preetham is monotone in turbidity and elevation and finite
  over 512 directions × 32 solar elevations; the CPU sun-radiance mirror matches the WGSL's own
  constants; `writeParams` allocates zero bytes; source bindings stay in the reserved 14–17 range.
- **`npm run test:sky:gpu`** — 13 numeric checks on a real device (headless Chrome; or open
  `/test/environment-gpu/` in a browser). Measured sun-disc radius vs `sunAngularRadius` within one
  pixel, skybox and bake against the CPU mirror, cube-edge continuity, `cheapSky: 1` against the
  pre-1.1 module, and the no-stale-mip source swap.
- **`npm run test:tod`** — headless celestial and integration checks: solar symmetry, polar
  day/night, the CPU radiance mirror, exclusive shadow-slot ownership and call order, sunset
  intensity, angular re-bake cadence at 1× and 1000×, lunar phase/illumination, normalized
  directions, zero typed-array allocation, `out` semantics, and named config errors. See
  [`time-of-day.md`](./time-of-day.md) §10.
- **`npm run test:night`** — celestial transforms, magnitude/flux, fixed-pixel sprite energy,
  Lommel–Seeliger lunar photometry, composite renaming, B−V colour, air-mass scintillation,
  by-reference catalogue upload, named errors, and WGSL parsing. See
  [`night-sky.md`](./night-sky.md) §8.
- **`npm run test:atmosphere`** — physical transmittance and wavelength ordering, the known vertical
  Rayleigh integral, phase normalization, LUT mappings, sky/composite binding propagation, named
  errors, and WGSL parsing. See [`atmosphere.md`](./atmosphere.md) §10.

`npm run test:environment` — 12 headless checks for the IBL half (U1–U10 + config validation):
cube-face convention, equirect round-trip, solid-angle exactness, SH packing order, ringing clamp,
constant-environment energy, rotation consistency, BRDF LUT endpoints, WGSL parse + binding budget,
the material hook, and the backtick guard.

The IBL GPU integration set (white furnace, fireflies, mirror fidelity) is still unwritten — see
[`lighting-ibl-design.md`](../lighting-ibl-design.md) §10 T1–T6.
