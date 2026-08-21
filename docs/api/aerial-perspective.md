# Aerial Perspective & Participating Media API Usage

`AerialPerspective`, `HeightFog`, and `mediaMath` are Phase 5 of `null-graph/environment`. Phases
1–4 render the air when you look **through** all of it. This renders the air **between the camera
and everything else**: distant mountains going blue and low-contrast, a valley filling with haze.

The model is not evaluated per screen pixel. It is cached in one 32×32×32 `rgba16float` froxel
volume — **262,144 bytes**, rebuilt every frame for ~32,768 texel updates against ~62,000,000 for a
per-pixel march at 1080p. Cost is independent of render resolution.

- **Physical atmosphere (the medium this shares):** [`atmosphere.md`](./atmosphere.md)
- **Environment and IBL frame graph:** [`environment.md`](./environment.md)
- **Sky-source binding contract:** [`environment-shader-reference.md`](./environment-shader-reference.md)
- **Technical design:** [`environment-aerial-design.md`](../environment-aerial-design.md)

```ts
import {
    AerialPerspective,
    Atmosphere,
    DEFAULT_HEIGHT_FOG,
    HeightFog,
} from 'null-graph/environment';
```

---

## 1. The one required engine change

This is the only phase in the environment series that touches `src/core/`, and the touch is one
usage flag.

```ts
await engine.init(canvas, { samplableDepth: true });
```

`WebGPUContext` creates the depth buffer `RENDER_ATTACHMENT`-only, so it cannot be sampled. Aerial
perspective is a function of distance to the surface behind each pixel — without depth there is no
distance.

**It is opt-in, and that is deliberate.** Adding `TEXTURE_BINDING` can disable some drivers'
depth-compression fast paths for that texture, and it costs that on every frame whether or not
anything samples it. Pay at construction for what you use.

**The zero-change fallback works today.** Render the scene into an offscreen pass with your own
depth attachment created with `TEXTURE_BINDING`, and pass that view to `getApplyPass({ depth })`.
`createPass` has always taken arbitrary attachments, and any scene using `PostChain` already renders
offscreen — so in practice only the main-screen convenience path is blocked.

```ts
const sceneDepth = engine.device.createTexture({
    size: [width, height], format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
});
```

Omitting both is a build-time refusal that names the flag *and* the fallback:

```
AerialPerspective: the apply pass needs a samplable depth texture. Create the context with
{ samplableDepth: true }, or render the scene into an offscreen pass with your own depth
attachment (design §9.1).
```

---

## 2. Complete setup

```ts
const atmosphere = new Atmosphere(engine);          // Phase 4 — supplies the medium and the LUT

const aerial = new AerialPerspective(engine, {
    atmosphere,
    fog: true,                    // adds the height-fog row; or pass a HeightFogMedium
    maxDistance: 32000,           // world units — 32 km at the default unitsPerKm
    distribution: 'linear',       // 'log' packs slices near the camera; see §4
    unitsPerKm: 1000,             // world units are metres
});

aerial.setSunDirection(sun.travelDirection);
```

Then splice it into the frame. **`getStages()` must be evaluated before `getApplyPass()`** — array
elements evaluate left to right, so writing them in this order is enough:

```ts
engine.render([
    ...atmosphere.getStages(),      // Phase 4: transmittance, multiscatter, sky-view
    ...aerial.getStages(),          // the 32³ froxel volume, per frame
    ...vis.getComputeStages(),
    scenePass,                      // opaque — untouched
    env.getSkyPass({ ... }),
    aerial.getApplyPass({           // one fullscreen pass, reads depth
        source: sceneColorView,
        colorAttachments: [{ view: foggedView, loadOp: 'clear', storeOp: 'store' }],
        covers: [scenePass, skyPass],
    }),
    transparentPass,                // materials with the injected layout (§5)
    ...post.getPassSequence(),
]);
```

`source` and the apply pass's colour attachment must be different textures — WebGPU forbids reading
an attachment you are writing. The transparent pass then draws into `foggedView`.

Calling `getApplyPass` before `getStages` refuses, because the volume would be sampled before
anything wrote it:

```
AerialPerspective: the froxel volume has never been built. Splice aerial.getStages() before
the scene pass.
```

---

## 3. Two application surfaces, one system

| | Fullscreen apply pass | Injected `ShaderAerialLayout` |
|---|---|---|
| For | **opaque** geometry | **transparent** geometry |
| Cost | once per pixel | once per fragment |
| Needs samplable depth | yes | no |
| Runtime toggle without rebuilding pipelines | yes | no |
| Depth-aware upsampling | yes | no — it sees one fragment, not a neighbourhood |

Both emit the **same `applyAerial` body**. The composition —

```wgsl
color * T + inScatter
```

— lives in exactly one function, `composeAerial`, in exactly one string, and a test asserts the two
consumers contain it byte-identically.

The fullscreen pass is chosen for opaque geometry because of correctness-per-pixel, toggleability,
and depth-aware upsampling — **not** because the injected layout would not fit in group 2. It fits,
with room (13 sampled textures of 16, 6 uniforms of 12, storage buffers unchanged). Saying so
plainly is what keeps the decision revisitable.

### Transparents

```ts
const batch = vis.createInstanceDrawBatch(pass, 'glass', 0, {
    material: {
        kind: 'pbr', instance: mat, lighting, indirect,
        aerial: aerial.getShaderLayout(),
    },
});
env.applyToBatch(batch, lights, aerial.getBindGroupEntries());   // group-2 bindings 18–19
aerial.registerBatch(batch, transparentPass, 'glass');
```

A transparent surface has **no single depth in the depth buffer** — it may not have written depth
at all, and several may overlap one pixel — so each needs aerial perspective at its own distance.

`registerBatch` is what makes the two paths' disjointness enforceable rather than merely true.
Registering a batch in a pass `getApplyPass({ covers })` names, in either order, refuses:

```
AerialPerspective: batch "glass" carries the injected aerial layout but is drawn in a pass
covered by getApplyPass(); aerial perspective would be applied twice. The injected layout is
for transparent passes drawn after the apply pass.
```

Every stock material family accepts the layout, lit or not — aerial perspective is a property of the
air between the camera and the surface, not of how the surface is shaded. On PBR it is injected
**before** the inline tonemap: aerial perspective is radiometric and belongs on linear radiance.
After the tonemap it fogs an already-compressed value and desaturates toward grey rather than toward
the sky's colour, which reads as "the fog colour is wrong" and points nowhere near the cause.

---

## 4. The froxel volume

Each texel holds in-scattered radiance in `rgb` and the mean transmittance from the camera in `a`.
The W axis maps to view distance, and the mapping is a config choice because the two media want
opposite things:

| `distribution` | Slices where | For |
|---|---|---|
| `'linear'` (default) | uniform to `maxDistance` | atmospheric haze — resolution matters at range |
| `'log'` | packed near the camera | ground fog, interiors — the near field is where it shows |

Beyond `maxDistance` the volume clamps to its last slice, which is correct: the medium's
transmittance is already near its asymptote there.

The march **continues** across slices rather than restarting from the camera, so the whole volume
costs one ray per lateral froxel with `dims[2]` accumulation steps. That is the entire cost
argument, and a version that restarts per slice renders an almost identical image for 32× the work.

**The characteristic artefact**, stated here rather than discovered: 32 lateral slices over a 60°
field of view is ~1.9° per froxel, so the volume is a coarse sampling of a signal with a sharp
forward-scatter peak. Near a hard occluder edge this shows as light bleeding around silhouettes.
The shipped mitigation is depth-aware upsampling in the apply pass (five taps, each sampling the
volume at its own view distance, with disagreeing taps rejected); temporal reprojection is the other
half and waits on TAA's history targets.

```ts
aerial.setApplyParams({
    enabled: true,          // runtime toggle — no pipeline recompilation
    intensity: 1.0,         // scales in-scatter only; extinction is physical
    depthTolerance: 0.05,   // bilateral rejection threshold, fraction of view distance
});
```

Sky pixels are skipped entirely: the sky already carries the whole atmospheric column from Phase 4's
sky-view LUT, and fogging it again would double-count the same medium.

---

## 5. Media, unified

A medium is a scattering coefficient, an absorption coefficient, a density profile, and a phase
function. Phase 4 defined three; this phase adds one and integrates all of them **together**:

| Medium | Profile | Origin |
|---|---|---|
| Rayleigh | exponential, H = 8 km | Phase 4 |
| Mie | exponential, H = 1.2 km | Phase 4 |
| Ozone | tent at 25 km | Phase 4 (absorption only) |
| **Height fog** | `σ₀ · exp(−(h − h₀)/H_f)` | **new** |

Height fog is Mie with different numbers. It is not a feature; it is a row.

```ts
aerial.setHeightFog({
    scattering: [2.0e-5, 2.0e-5, 2.0e-5],   // m^-1 at referenceHeight
    absorption: [1.0e-6, 1.0e-6, 1.0e-6],
    referenceHeight: 0,                      // km
    scaleHeight: 0.3,                        // km — small is ground fog, large is uniform haze
    g: 0.6,                                  // Cornette-Shanks anisotropy
});
```

### Why they must be one system

Running an analytic height fog *and* the froxel volume applies extinction twice:

```
wrong:   color · T_fog · T_aerial + inScatter_fog + inScatter_aerial
right:   color · T_combined       + inScatter_combined
```

The trap that lets this ship: extinction sums in the exponent, so `T_fog · T_aerial` already
**equals** `T_combined` exactly. Anything that checks how opaque the fog is passes. What is wrong is
the in-scatter — each separate march attenuates its own radiance by its own extinction only, so
summing them over-counts by whatever the other medium would have absorbed. And it is only wrong when
both are enabled, so it survives every test that enables one.

So the volume owns every medium while it is active, and the two refuse each other at construction,
in both orders:

```
AerialPerspective: HeightFog is active alongside the froxel volume; both integrate the same
medium and extinction would be applied twice (design §6.2). Move the fog parameters onto the
volume's media list.
```

---

## 6. `HeightFog` — the cheap path, shipped standalone

The closed-form exponential integral needs **no volume, no compute stage, and no samplable depth**.
For a single medium it is genuinely cheaper than a froxel volume, and it is a drop-in replacement at
the material site: both define `applyAerialAt(color, worldPos)`, so no material learns which it got.

```ts
const fog = new HeightFog(engine, { fog: DEFAULT_HEIGHT_FOG, unitsPerKm: 1000 });
fog.setSunDirection(sun.travelDirection);
fog.setSunIrradiance([2.0, 1.9, 1.7]);

const batch = vis.createInstanceDrawBatch(pass, 'terrain', 0, {
    material: { kind: 'pbr', instance: mat, lighting, aerial: fog.getShaderLayout() },
});
env.applyToBatch(batch, lights, fog.getBindGroupEntries());
```

Scattering and absorption share one density profile, so the single-scattering albedo is constant
along the ray and the in-scatter collapses to the familiar `fogColor * (1 - T)`. The only
approximation is that the sun's own transmittance is held constant along the ray — which is exactly
what makes it cheap. A test pins the two paths to within 2% for a matched single medium, over 64
(distance, altitude) pairs; that agreement is what makes switching between them safe.

---

## 7. God rays: screen-space vs volumetric

`post/GodRays.ts` marches radially outward in UV from a screen-space light position, weighting by
luminance above a threshold. It needs no depth, no volume, and no atmosphere. **It works today and
this phase does not replace it.**

| | Screen-space (`post/GodRays.ts`) | Volumetric (deferred) |
|---|---|---|
| Light off-screen | fails — the origin must be on screen | correct |
| Occluders behind the camera | invisible to it | correct |
| Shafts through a doorway from the side | not representable | correct |
| Cost | one radial blur | froxel volume + shadow taps |
| Depends on | nothing | samplable depth + shadow atlas |
| Physically coupled to the sky | no | yes — same medium as the horizon haze |

They are different effects with overlapping appearance. A stylized scene may prefer the screen-space
one permanently.

---

## 8. Configuration reference

| Option | Default | Notes |
|---|---|---|
| `dims` | `[32, 32, 32]` | each axis must be a multiple of 4 |
| `maxDistance` | `32000` | world units; must exceed `nearDistance` |
| `nearDistance` | `0.1` | world units; should match the camera near plane |
| `distribution` | `'linear'` | or `'log'` |
| `unitsPerKm` | `1000` | world units are metres |
| `groundLevel` | `0` | the world Y that is altitude 0 |
| `stepsPerSlice` | `1` | 1 gives exactly one ray per lateral froxel |
| `depthAwareUpsampling` | `true` | the silhouette-bleed mitigation |
| `upsampleDepthTolerance` | `0.05` | fraction of view distance |
| `sunIntensity` | `1` | linear scale on in-scatter |
| `aerialGroup` | `2` | must match the `LightSystem`'s group |

Group-2 bindings used: **18** (the volume, or `HeightFog`'s uniform) and **19** (its sampler).

---

## 9. Not in this phase

- **Clouds** — a medium with 3D spatial structure, its own advection, and its own lighting model.
- **Light injection into the froxel volume** — the clustered light list is the natural next step.
- **Volumetric shadowing** — one binding and a loop; the shadow atlas already exposes a public view
  and receivers already tap it by slot. Deferred because it makes the volume cost proportional to
  steps × lights and needs the noise mitigation that temporal reprojection provides.
- **Temporal reprojection** — depends on TAA's history machinery.
- **Local media volumes** (a room, a cave, underwater) — a placement problem; the medium and the
  integration are the same.
