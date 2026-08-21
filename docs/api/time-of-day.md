# Time of Day API Usage

`TimeOfDay` is the allocation-free celestial clock in `null-graph/environment`. It renders
nothing and owns no GPU resources. Its job is to compute the sun and moon once, then write the
same **travel directions** into the sky, directional lights, shadows, and exposure controls.

That removes a subtle but common failure mode: the sky highlight, direct light, and shadow camera
each being individually valid but pointed in different directions.

- **Environment and sky setup:** [`environment.md`](./environment.md)
- **Night sky and star catalogue:** [`night-sky.md`](./night-sky.md)
- **Physical atmosphere:** [`atmosphere.md`](./atmosphere.md)
- **Architecture and mechanics:**
  [`environment-timeofday-design.md`](../environment-timeofday-design.md)
- **Lights:** [`lights.md`](./lights.md)
- **Shadows:** [`shadows.md`](./shadows.md)
- **Post-processing:** [`post.md`](./post.md)

```ts
import {
    EnvironmentSystem, NightSky, PreethamSky, SkyComposite, StarField, TimeOfDay,
} from 'null-graph/environment';
import { DirectionalLightProxy, LightSystem } from 'null-graph/lights';
import { ShadowSystem } from 'null-graph/shadows';
import { AutoExposure, Tonemap } from 'null-graph/post';
```

---

## 1. Minimum animated-sky setup

Use analytic day/night `SkySource`s and an amortized environment. `TimeOfDay` changes source
parameters every frame, but requests expensive IBL work only after the sun or night-weighted moon
has moved by the configured angular threshold.

```ts
const env = new EnvironmentSystem(engine, {
    updateMode: 'amortized',
    keepSourceCube: true,
    prefilterMips: 6,
});

env.setSky(new SkyComposite({
    day: new PreethamSky({
        turbidity: 2.8,
        groundAlbedo: [0.15, 0.14, 0.12],
        sunAngularRadius: 0.266,
    }),
    night: new NightSky({ latitudeDeg: 45 }),
}));

const stars = new StarField(engine, { latitudeDeg: 45, maxStars: 9100 });

const tod = new TimeOfDay({
    mode: 'artistic',
    timeScale: 60,              // 60 simulated seconds per real second
    rebakeThresholdDeg: 0.25,
    exposureCoupling: 'off',    // no post effect is bound in this minimal example
    prefilterMips: 6,           // warning text only; match EnvironmentSystem
    purkinje: true,             // named artistic blue shift on the moon directional light
});

tod.setTime(8.0);               // 08:00 on the 0–24 artistic dial

const skyPass = env.getSkyPass({
    colorAttachments,
    depthStencilAttachment,
    size: [width, height],
    targetFormat: 'rgba16float',
});
stars.attach(env);              // append the unresolved-star batch after the resolved sky
```

Per frame:

```ts
function frame(dt: number) {
    tod.advance(dt);
    tod.applyTo({ environment: env, stars });

    // getBakeStages contains the declared-dirtiness gate and returns a shared empty array while
    // idle. During a trigger it emits one amortized slice per frame until every mip is refreshed.
    const bakeStages = env.getBakeStages();

    engine.render([
        ...bakeStages,
        ...vis.getComputeStages(),
        scenePass,
        skyPass,
        ...post.getPassSequence(),
    ]);

    if (bakeStages.length > 0) env.copySHToParams();
}
```

`copySHToParams()` must follow a submitted bake. The prefiltered cube is updated directly by the
compute stages, but diffuse IBL reads a separate uniform and needs this GPU-side copy.

### Why both dirty flags appear

- `tod.envDirty` becomes true when the maximum of solar motion and night-weighted lunar motion
  since the last published bake reaches `rebakeThresholdDeg`. Lunar motion is multiplied by
  `1 - dayFactor`, so a daytime moon does not force environment work.
- `env.bakeInvalidated` remains true until the environment has emitted all outstanding amortized
  mip slices. Repeated invalidations coalesce rather than restarting the cycle.

At the default threshold, real-time motion triggers roughly once per minute. A 1000× simulation
triggers around every four frames. A frame counter cannot behave correctly at both speeds.

`TimeOfDay` owns celestial invalidation only. If another source parameter changes—turbidity,
weather blend, or a custom sky parameter—upload that source state with `env.setSkyParams({})` and
call `env.invalidateBake()` yourself. The environment coalesces that request with any active
celestial refresh.

---

## 2. Driving directional lights

`TimeOfDay` writes through existing `DirectionalLightProxy` cursors. You own the light array, its
indices, and the eventual `LightSystem` upload.

```ts
const lights = new LightSystem(engine, {
    technique: 'clustered-forward',
    capacity: { point: 32, directional: 2, spot: 8 },
    clusterConfig: { grid: [16, 9, 24], maxLightsPerTile: 32 },
});

const lightData = new Float32Array(
    (32 + 2 + 8) * LightSystem.FLOATS_PER_LIGHT,
);

const sun = new DirectionalLightProxy();
sun.wrap(lightData);
sun.seek(32);                    // index ownership remains yours

const moon = new DirectionalLightProxy();
moon.wrap(lightData);
moon.seek(33);

lights.setRawData(lightData);    // adopts the same array by reference
```

Then update and upload:

```ts
tod.advance(dt);
tod.applyTo({ environment: env, sun, moon });
lights.setRawData(lightData);    // marks the adopted array dirty
lights.upload();
```

The clock writes:

- normalized travel direction to both records;
- extinction-attenuated linear HDR colour;
- a sun intensity that is zero at and below geometric sunset;
- moon colour scaled by phase illumination, with intensity disabled below the horizon;
- a named optional Purkinje weighting on the moon light (`purkinje: true` by default).

If you do not want a moon light, omit `moon`. Lunar direction, phase, and illumination still drive
`NightSky` when the environment binding is present.

### Moonlight colour and `purkinje`

The moon's radiometric weighting is slightly warm. The familiar blue moonlit look is an observer
effect, not blue illumination, so it is exposed as a named artistic option:

```ts
const cinematic = new TimeOfDay({ purkinje: true });   // default: blue-biased moon light
const radiometric = new TimeOfDay({ purkinje: false }); // warm moon light weighting
```

This changes the directional moon light written to `DirectionalLightProxy`; it does not recolour
the lunar disc in `NightSky`.

---

## 3. Shadows and the sun/moon handoff

The current `ShadowSystem` has one directional shadow direction. Sun and moon therefore share one
registered directional atlas slot, and exactly one light record may hold it at a time.

At setup time, on the atlas shadow path:

```ts
const celestialShadowSlot = shadows.registerLight({
    kind: 'directional',
    name: 'sun-or-moon',
});
```

Per frame:

```ts
tod.applyTo({
    environment: env,
    sun,
    moon,
    shadows,
    shadowSlot: celestialShadowSlot,
});
```

`applyTo()` performs the related writes together:

1. removes the slot from the inactive light;
2. assigns it to the active light, if any;
3. updates the shadow direction and registered atlas light;
4. invalidates the slot's cached views.

The default handoff rules are:

- sun casts at elevations of at least `3°`;
- moon casts at elevations of at least `10°` and illumination of at least `0.35`;
- otherwise `activeBody` is `'none'` and neither record owns the slot.

The gap is intentional. Directional shadow depth precision degrades near the horizon, and a true
sun/moon shadow crossfade would require two live cascade sets.

Without the atlas path, passing `shadows` still drives `ShadowSystem.setLight()`. Omit
`shadowSlot`; registered-slot updates and invalidation apply only when an atlas slot is supplied.

---

## 4. Exposure coupling

Exposure coupling is explicit. `TimeOfDay` does not search a `PostChain` because the chain keeps
effect ownership private; pass the effect instance you already placed in the chain.

### Histogram adaptation with a day/night clamp

This is the default mode:

```ts
const autoExposure = new AutoExposure();

const tod = new TimeOfDay({
    exposureCoupling: 'clamp',
});

tod.applyTo({ environment: env, sun, moon, autoExposure });
```

`'clamp'` changes `AutoExposure.minExposure` and `maxExposure` from `dayFactor`; the existing
histogram and temporal adaptation remain authoritative inside that window. Omitting
`autoExposure` in this mode throws.

### Deterministic exposure

```ts
const tonemap = new Tonemap({ curve: 'aces' });
const tod = new TimeOfDay({ exposureCoupling: 'absolute' });

tod.applyTo({ environment: env, sun, moon, tonemap });
```

`'absolute'` drives `Tonemap.exposure` directly from `dayFactor`. Omitting `tonemap` throws.

### No clock-owned exposure

```ts
const tod = new TimeOfDay({ exposureCoupling: 'off' });
```

Use `'off'` when the application already owns exposure or when no post chain is present.

---

## 5. Clock modes

### Artistic mode — default

```ts
const tod = new TimeOfDay({
    mode: 'artistic',
    sunTiltDeg: 23.44,
    date: new Date('2026-06-21T00:00:00Z'),
});

tod.setTime(18.5);               // 18:30
```

The visible sun follows a geography-free great circle. Noon is the top of the arc; 06:00 and
18:00 touch the geometric horizon. The date still drives the real lunar phase and GMST used by
`NightSky` and `StarField`.

### Astronomical mode

```ts
const tod = new TimeOfDay({
    mode: 'astronomical',
    latitudeDeg: 51.5074,
    longitudeDeg: -0.1278,       // east-positive, west-negative
    date: new Date('2026-06-21T00:00:00Z'),
    timeScale: 600,
});
```

Astronomical mode uses the configured UTC date and time, latitude, and longitude. It includes
seasonal day length and polar day/night. Both coordinates are required; use artistic mode if you
do not want geography.

`advance(dt)` takes **real seconds** and applies `timeScale`. `setTime(hours)` sets the current
UTC/artistic 0–24 time while retaining the configured date.

---

## 6. Configuration reference

| Field | Default | Meaning |
|---|---:|---|
| `mode` | `'artistic'` | `'artistic'` or `'astronomical'`. |
| `latitudeDeg` | `45` | Observer latitude. Required explicitly in astronomical mode. |
| `longitudeDeg` | `0` | Observer longitude, east-positive. Required explicitly in astronomical mode. |
| `date` | current date/time | Initial UTC date and 0–24 time. Converted to Julian day at construction. |
| `sunTiltDeg` | `23.44` | Artistic great-circle tilt. |
| `timeScale` | `1` | Simulated seconds per real second. |
| `rebakeThresholdDeg` | `0.25` | Maximum solar/night-weighted lunar angular motion required to invalidate IBL. Must be greater than zero. |
| `sunShadowMinElevDeg` | `3` | Minimum sun elevation for shadow ownership. |
| `moonShadowMinElevDeg` | `10` | Minimum moon elevation for shadow ownership. |
| `moonMinIllumination` | `0.35` | Minimum illuminated fraction for moon shadows. |
| `civilDeg` | `-6` | Elevation at which `dayFactor` reaches 1. |
| `nightDeg` | `-18` | Elevation at which `dayFactor` reaches 0. Must be below `civilDeg`. |
| `exposureCoupling` | `'clamp'` | `'off'`, `'clamp'`, or `'absolute'`. |
| `turbidity` | `2.8` | Turbidity used by the CPU sun/moon radiance mirror. Match `PreethamSky`. |
| `prefilterMips` | `6` | Used only in the warning for a re-bake threshold above `2°`; match the environment. |
| `purkinje` | `true` | Apply the named artistic blue shift to the directional moon light. |

`rebakeThresholdDeg > 2` is legal but warns: an amortized cube is mixed-age for several frames,
which can make rough reflections step visibly.

---

## 7. Runtime API

### Methods

| Method | Notes |
|---|---|
| `setTime(hours)` | Set the wrapped 0–24 time and recompute state. |
| `advance(dtSeconds)` | Advance by real seconds multiplied by `timeScale`, then recompute. |
| `update()` | Recompute without advancing. Normally called indirectly by the two methods above. |
| `applyTo(bindings)` | Write all supplied consumers. The caller still uploads the light array. |
| `getSunDirection(out?)` | Return the persistent internal view, or fill and return `out`. |
| `getMoonDirection(out?)` | Return the persistent internal view, or fill and return `out`. |

The no-argument direction views are valid until the next `update()`, `advance()`, or `setTime()`.
Use an `out` array when the value must be retained independently.

### Read-only state

| Property | Meaning |
|---|---|
| `envDirty` | Solar or night-weighted lunar motion reached the IBL re-bake threshold this update. |
| `activeBody` | `'sun'`, `'moon'`, or `'none'` for directional shadow ownership. |
| `dayFactor` | Smooth twilight factor: 0 at/below `nightDeg`, 1 at/above `civilDeg`. |
| `sunElevationDeg` | Solar elevation above the horizon. |
| `moonElevationDeg` | Lunar elevation above the horizon. |
| `moonPhase` | Wrapped phase: 0 new, 0.5 full. |
| `moonIllumination` | Illuminated fraction in `[0, 1]`. |
| `gmstRad` | Greenwich mean sidereal time used to rotate `NightSky` and `StarField`. |

### `TimeOfDayBindings`

Every binding is optional, but the selected exposure mode may require its corresponding effect.

| Field | Written by `applyTo()` |
|---|---|
| `environment` | Full celestial state (`sunDir`, `moonDir`, illumination, GMST, `dayFactor`), source params, and bake invalidation. |
| `stars` | GMST through `setCelestialState`, followed by `StarField.upload()`. |
| `sun` | Direction, colour, intensity, and shadow slot. |
| `moon` | Direction, colour, intensity, and shadow slot. |
| `shadows` | Active directional shadow direction and atlas record/cache. |
| `shadowSlot` | Shared registered directional atlas slot. |
| `autoExposure` | Clamp pair in `'clamp'` mode. |
| `tonemap` | Exposure in `'absolute'` mode. |

---

## 8. Pure celestial helpers

`todMath` is exported from `null-graph/environment` and has no GPU or browser dependency:

```ts
import {
    createSolarPosition,
    julianDay,
    solarPosition,
} from 'null-graph/environment';

const out = createSolarPosition();             // allocate once
const jd = julianDay(new Date());

solarPosition(jd, 45, 0, out);                 // reuses out
console.log(out.elevationDeg, out.azimuthDeg, out.direction);
```

Exported helpers include `julianDay`, `gmstRadians`, `solarPosition`,
`artisticSolarPosition`, `lunarPosition`, `skySunRadiance`, `sunIntensity`, and
`angularDistanceDeg`. `createSolarPosition()` and `createLunarPosition()` create reusable output
records; omitting `out` from a position helper allocates for one-shot use.

Phase 3's pure catalogue and lunar helpers live in `nightMath`; see
[`night-sky.md`](./night-sky.md) §7.

---

## 9. Errors and warnings

| Condition | Result |
|---|---|
| Astronomical mode without both coordinates | Throws, naming artistic mode as the geography-free alternative. |
| Latitude outside `[-90, 90]` | Throws with the supplied value. |
| `rebakeThresholdDeg <= 0` | Throws; zero is equivalent to unconditional amortized work. |
| Moon shadow elevation below the sun threshold | Throws because the eligibility windows overlap. |
| `nightDeg >= civilDeg` | Throws because the twilight ramp would be empty or inverted. |
| Clamp coupling without `AutoExposure` | Throws and suggests absolute coupling. |
| Absolute coupling without `Tonemap` | Throws and names the missing effect. |
| `rebakeThresholdDeg > 2` | Warns about mixed-age prefilter mips and visible reflection stepping. |

---

## 10. Testing

```bash
npm run test:tod
```

The headless suite covers solar symmetry, the physical transmittance radiance mirror, polar
day/night, exclusive
shadow-slot ownership, handoff call order, sunset intensity, the re-bake gate at 1× and 1000×,
lunar phase/illumination, normalized directions, zero typed-array allocation over 1,000 frame
updates, `out` semantics, and configuration errors. It needs no GPU, device, or browser.

Phase 3's transform, lunar-disc, star-flux, composite, and generated-WGSL checks are in:

```bash
npm run test:night
```

It is also included in:

```bash
npm run test:environment
```