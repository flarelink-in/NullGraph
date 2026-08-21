# Night Sky API Usage

Phase 3 of `null-graph/environment` adds three cooperating APIs:

- `NightSky` — the **resolved** night evaluator: lunar disc and phase, earthshine, night dome,
  airglow, Milky Way, and optional horizon light pollution;
- `SkyComposite` — decorates two `SkySource`s into the one `sampleSky` evaluator required by the
  environment bake and skybox;
- `StarField` — the **unresolved** point-source catalogue, rendered as one additive instanced draw.

The split is deliberate. The 0.52° moon is resolved by the source cube and therefore reaches SH
and specular IBL. A star is not resolved at any practical cube size, so it is drawn in screen space
with a fixed pixel footprint and does not enter the bake.

- **Environment and frame graph:** [`environment.md`](./environment.md)
- **Clock, moonlight, and sidereal rotation:** [`time-of-day.md`](./time-of-day.md)
- **Physical day atmosphere:** [`atmosphere.md`](./atmosphere.md)
- **Technical design:** [`environment-night-design.md`](../environment-night-design.md)
- **Sky shader contract:** [`environment-shader-reference.md`](./environment-shader-reference.md) §3.4

```ts
import {
    EnvironmentSystem,
    NightSky,
    PreethamSky,
    SkyComposite,
    StarField,
    TimeOfDay,
} from 'null-graph/environment';
```

---

## 1. Complete day/night setup

Install the resolved day/night evaluator before compiling the sky pass:

```ts
const env = new EnvironmentSystem(engine, {
    updateMode: 'amortized',
    keepSourceCube: true,
    prefilterMips: 6,
});

const night = new NightSky({
    nightGain: 1,
    latitudeDeg: 45,
    airglow: 0.00012,
    milkyWay: 0.0006,
    lightPollution: 0,
});

env.setSky(new SkyComposite({
    day: new PreethamSky({ turbidity: 2.8 }),
    night,
}));

const stars = new StarField(engine, {
    maxStars: 9100,
    latitudeDeg: 45,
    longitudeDeg: 0,
    radiusPx: 1.5,
    twinkle: 0.18,
});

const tod = new TimeOfDay({
    mode: 'artistic',
    latitudeDeg: 45,
    longitudeDeg: 0,
    timeScale: 60,
    rebakeThresholdDeg: 0.25,
    exposureCoupling: 'off',
    purkinje: true,
});
```

Create the environment sky pass, then attach the star batch. Batch insertion order is draw order,
so this puts the additive stars after the resolved sky evaluator:

```ts
const skyTarget = {
    colorAttachments,
    depthStencilAttachment,
    size: [width, height] as [number, number],
    targetFormat: 'rgba16float' as GPUTextureFormat,
};

const skyPass = env.getSkyPass(skyTarget);
stars.attach(env);                       // appends batch 1 after the sky evaluator
```

You can instead create and attach in one call:

```ts
stars.attach(env, skyTarget);            // creates the cached sky pass if needed
```

Per frame:

```ts
tod.advance(dt);
tod.applyTo({
    environment: env,
    stars,
    sun,
    moon,
    shadows,
    shadowSlot: celestialShadowSlot,
});

const bakeStages = env.getBakeStages();
engine.render([
    ...bakeStages,
    ...vis.getComputeStages(),
    scenePass,
    skyPass,
    ...post.getPassSequence(),
]);

if (bakeStages.length > 0) env.copySHToParams();
```

`TimeOfDay.applyTo()` sends one persistent `CelestialState` to the composite and star field. The
composite receives sun direction, moon direction, moon illumination, GMST, and `dayFactor`; the
star field receives GMST and uploads its small parameter uniform.

After a resize, update the star viewport dimensions:

```ts
stars.upload(width, height);
```

That call also uploads a newly installed catalogue if one is dirty.

---

## 2. `NightSky`

`NightSky` implements `SkySource`, so it can be installed alone for a static night or used as one
child of `SkyComposite`.

```ts
const night = new NightSky({
    nightGain: 1,
    moonAngularRadius: 0.26,             // degrees; 0.52° diameter
    moonColor: [1.0, 0.92, 0.78],       // slightly warm radiometric moon colour
    earthshine: [0.00012, 0.00016, 0.00022],
    domeColor: [0.00018, 0.00027, 0.00055],
    airglow: 0.00012,
    lightPollutionColor: [1.0, 0.34, 0.08],
    lightPollution: 0,
    milkyWay: 0.0006,
    latitudeDeg: 45,
    fixedGmstRad: 0,
});
```

### Configuration

| Field | Default | Meaning |
|---|---:|---|
| `nightGain` | `1` | Scene-relative gain for resolved night content. Must be greater than zero. |
| `moonAngularRadius` | `0.26` | Lunar angular radius in degrees. |
| `moonColor` | `[1, 0.92, 0.78]` | Linear colour multiplying lunar disc radiance. |
| `earthshine` | `[0.00012, 0.00016, 0.00022]` | Dark-limb radiance, scaled by `1 - moonIllumination`. |
| `domeColor` | `[0.00018, 0.00027, 0.00055]` | Base upper-hemisphere night radiance. |
| `airglow` | `0.00012` | Constant green-biased atmospheric emission strength. |
| `lightPollutionColor` | `[1, 0.34, 0.08]` | Linear horizon-glow colour. |
| `lightPollution` | `0` | Horizon-glow strength; zero disables it. |
| `milkyWay` | `0.0006` | Procedural celestial-band strength. |
| `latitudeDeg` | `45` | Stored observer latitude for night parameters. |
| `fixedGmstRad` | `0` | Static sidereal rotation when no `TimeOfDay` drives the source. |

The moon uses a geometric terminator and Lommel–Seeliger scattering rather than a Lambert sphere.
At full phase that keeps the disc nearly uniform to the limb. Lunar radiance also includes
air-mass extinction and an opposition-surge approximation. No lunar albedo texture is currently
used.

Because the moon is part of `sampleSky`, it is visible in the per-pixel skybox and enters the source
cube, GGX prefilter, and SH projection. Moonlit water can therefore receive a real environment
specular streak without a separate reflection path.

---

## 3. `SkyComposite`

```ts
const composite = new SkyComposite({
    day: new PreethamSky(),
    night: new NightSky(),
    dayFactor: 1,
});
```

| Config field | Meaning |
|---|---|
| `day` | First `SkySource`; evaluated when `dayFactor` is 1. |
| `night` | Second `SkySource`; evaluated when `dayFactor` is 0. |
| `dayFactor` | Initial blend in `[0, 1]`; defaults to 1. |

`setDayFactor(value)` clamps and updates the blend. Normally `TimeOfDay` owns it through
`setCelestialState`, where it follows the configured `civilDeg`/`nightDeg` twilight ramp.

The composite renames each child's entry points and every `sky_` helper, then emits exactly one
public `sampleSky` and one `skySunRadiance`. Its parameter count is sized by demand:

```text
day.paramFloats + night.paramFloats + 1 blend float
```

The current implementation expects analytic children that share the normal parameter-buffer
contract. It does not compose child-specific bake/skybox resource bindings such as an
`EquirectSky` texture facet.

---

## 4. `StarField`

### Built-in no-asset catalogue

With no `stars` option, `StarField` generates a deterministic catalogue of up to 2,000 stars:

```ts
const stars = new StarField(engine);
```

The batch still reserves the default `maxStars: 9100`, so a larger catalogue can be adopted later
without rebuilding the pipeline.

### Caller-supplied catalogue

Each star is four consecutive `f32` values:

| Float | Meaning | Unit |
|---:|---|---|
| 0 | right ascension | radians |
| 1 | declination | radians |
| 2 | apparent visual magnitude | magnitudes |
| 3 | B−V colour index | dimensionless |

```ts
const catalogue = new Float32Array([
    // ra,       dec,       magnitude, B-V
    1.767794,  -0.291751,  -1.46,      0.00,  // Sirius
    1.549729,   0.129277,   0.42,      1.85,  // Betelgeuse
]);

const stars = new StarField(engine, {
    maxStars: catalogue.length / 4,
    stars: catalogue,
    latitudeDeg: 45,
});
```

`setStars(data)` **adopts by reference** and marks the catalogue dirty. It does not clone:

```ts
stars.setStars(catalogue);
catalogue[2] = -1.5;       // the next upload observes this mutation
stars.upload();
```

### Configuration

| Field | Default | Meaning |
|---|---:|---|
| `maxStars` | `9100` | Fixed batch capacity. The catalogue cannot exceed it later. |
| `stars` | procedural fallback | Initial four-float records, adopted by reference. |
| `latitudeDeg` | `45` | Observer latitude used by the celestial-to-horizon transform. |
| `longitudeDeg` | `0` | East-positive observer longitude. |
| `fixedGmstRad` | `0` | Static sidereal rotation until a clock drives it. |
| `twinkle` | `0.18` | Scintillation strength; Kasten–Young air mass suppresses it near zenith. |
| `radiusPx` | `1.5` | Sprite radius in physical render-target pixels. |
| `intensity` | `0.035` | Global star flux multiplier. |

### Methods

| Method | Notes |
|---|---|
| `setStars(data)` | Validate and adopt a `Float32Array` by reference; upload is deferred. |
| `setCelestialState(state)` | Copy `state.gmstRad` into the persistent star parameter block. Normally called by `TimeOfDay`. |
| `attach(env, target?)` | Append the star batch to an existing sky pass, or create it from `target`. Idempotent. |
| `attachToPass(pass, targetFormat?)` | Lower-level attachment to a `RenderPassNode`. Defaults to `rgba16float`. |
| `upload(width?, height?)` | Upload a dirty catalogue, current GMST/twinkle parameters, and optional resized dimensions. |

The draw is one `draw(3, starCount)` call with no vertex or index buffer. Magnitude maps to flux as
`10^(-0.4m)`, B−V maps to a blackbody approximation, and the radial profile is smooth. The sprite
radius is in pixels rather than degrees, so its covered pixel count and total image energy do not
grow with render resolution.

### Current visibility limitation

`StarField` is a separate additive batch and currently has no `dayFactor` parameter. Attaching it
makes it draw whenever the sky pass draws; `TimeOfDay` rotates it but does not fade it during day or
twilight. Applications that need daylight suppression must currently control whether the star
batch is attached/drawn or swap to an empty catalogue. `intensity` is currently fixed at
construction; changing it later requires rebuilding the `StarField`.

---

## 5. Moonlight and Purkinje shift

The moon directional light is driven by `TimeOfDay`, not `NightSky`. By default:

```ts
const tod = new TimeOfDay({ purkinje: true });
```

The underlying radiometric moon colour is slightly warm. `purkinje: true` applies a named artistic
blue shift to the **directional moon light**, matching the expected scotopic appearance. Set it to
`false` for the warm radiometric weighting. It does not recolour the moon disc.

Moon shadow eligibility remains controlled by `moonShadowMinElevDeg` and
`moonMinIllumination`; see [`time-of-day.md`](./time-of-day.md) §3.

---

## 6. Errors

| Condition | Error |
|---|---|
| `nightGain <= 0` | `NightSky: nightGain must be > 0; 0 renders an unlit night, which is what omitting NightSky does (design §1).` |
| Catalogue length is not divisible by four | `StarField: star records are 4 floats (ra, dec, magnitude, bv); got <n>, which is not a multiple of 4.` |
| Catalogue exceeds fixed capacity | `StarField: <n> stars exceeds maxInstances <m> fixed at batch creation. Pass maxStars at construction.` |
| Attach without an existing pass or target | `EnvironmentSystem: no sky pass exists yet; call getSkyPass(target) before attaching a StarField, or pass target to StarField.attach().` |
| Missing composite child | `SkyComposite: child "day|night" has no source set.` |

All checks happen at construction/attachment or explicit upload boundaries, not in the render loop.

---

## 7. Pure helpers

`nightMath` is exported from `null-graph/environment` and is browser/GPU independent:

| Helper | Purpose |
|---|---|
| `celestialToHorizon(ra, dec, gmst, latitude, longitude?, out?)` | Catalogue coordinates to `(east, up, north)`. |
| `magnitudeToFlux(magnitude)` | Astronomical magnitude to relative flux. |
| `moonRadiance(illumination, elevationDeg, phaseAngleRad, nightGain?)` | Scene-relative lunar photometry. |
| `lunarOppositionBoost(phaseAngleRad)` | Full-moon opposition surge. |
| `lommelSeeliger(nDotL, nDotV)` | Lunar single-scattering term. |
| `bvToRGB(bv, out?)` | B−V index to normalized linear RGB approximation. |
| `starSpriteEnergy(magnitude, radiusPx?, samples?)` | CPU integral of the shipped radial profile. |
| `proceduralStarCatalogue(count?, seed?)` | Deterministic four-float fallback catalogue. |

Helpers with `out` parameters allocate only when `out` is omitted.

---

## 8. Testing

```bash
npm run test:night
```

The headless suite pins the celestial transform, magnitude scale, pixel-space flux conservation,
Lommel–Seeliger limb behaviour, lunar photometry, finite/non-negative radiometric parameters,
composite renaming, twilight monotonicity, B−V colours, shared air-mass scintillation, by-reference
catalogue uploads, named errors, and parse-valid generated WGSL.

It is included in:

```bash
npm run test:environment
```

The GPU readback tests proposed in the Phase 3 design—measured moon diameter, named-star positions,
one-draw profiler count, moonlit water streak, and full-cycle swapchain sampling—are not yet part of
the browser harness.