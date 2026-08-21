# Physical Atmosphere API Usage

`Atmosphere` and `AtmosphereSky` are the physical daylight implementation in
`null-graph/environment`. Together they replace a fitted day sky with a participating medium that
contains **Rayleigh scattering, Mie scattering and absorption, ozone absorption, and approximated
higher-order scattering**.

The model is not evaluated per screen pixel. It is cached in three filterable `rgba16float` lookup
textures. The visible sky performs one sky-view lookup and composites an analytic true-size sun
disc. At the default 192×108 sky-view resolution, atmosphere cost is independent of render
resolution.

- **Environment and IBL frame graph:** [`environment.md`](./environment.md)
- **Time-of-day driver:** [`time-of-day.md`](./time-of-day.md)
- **Night child and `SkyComposite`:** [`night-sky.md`](./night-sky.md)
- **Sky-source binding contract:** [`environment-shader-reference.md`](./environment-shader-reference.md)
- **Technical design:** [`environment-atmosphere-design.md`](../environment-atmosphere-design.md)

```ts
import {
    Atmosphere,
    AtmosphereSky,
    EnvironmentSystem,
    NightSky,
    SkyComposite,
    TimeOfDay,
} from 'null-graph/environment';
```

---

## 1. Complete setup

Construct one `Atmosphere` after the engine device is initialized. It owns the LUT textures and
compute stages. `AtmosphereSky` is the lightweight `SkySource` adapter installed in
`EnvironmentSystem`.

```ts
const atmosphere = new Atmosphere(engine);

const day = new AtmosphereSky(atmosphere, {
    sunAngularRadius: 0.266,   // degrees; radius, so diameter is 0.532°
    sunDiscIntensity: 20,
});

const env = new EnvironmentSystem(engine, {
    updateMode: 'amortized',
    keepSourceCube: true,
    prefilterMips: 6,
});

env.setSky(new SkyComposite({
    day,
    night: new NightSky({ latitudeDeg: 45 }),
}));

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
```

`AtmosphereSky` is analytic and does **not** need a retained source cube. The example enables
`keepSourceCube` because an amortized `EnvironmentSystem` requires it while refreshing IBL. With
`updateMode: 'once'`, the atmosphere source itself works with `keepSourceCube: false`.

### Per frame

```ts
function frame(dt: number, cameraHeightKm: number) {
    tod.advance(dt);
    tod.applyTo({ environment: env, stars, sun, moon, shadows });

    // Kilometres above the configured spherical ground radius.
    atmosphere.setCameraAltitude(cameraHeightKm);

    const atmosphereStages = atmosphere.getStages();
    const bakeStages = env.getBakeStages();

    engine.render([
        ...atmosphereStages,       // MUST precede consumers of the sky-view LUT
        ...bakeStages,
        ...vis.getComputeStages(),
        scenePass,
        skyPass,
        ...post.getPassSequence(),
    ]);

    if (bakeStages.length > 0) env.copySHToParams();
}
```

The ordering is load-bearing:

1. `atmosphere.getStages()` writes transmittance, multiple scattering, and sky view;
2. `env.getBakeStages()` samples `AtmosphereSky` into the source cube;
3. the sky pass samples the same sky-view LUT for the visible background.

Consecutive compute stages execute in order in one WebGPU compute pass. No explicit barrier or
readback is required.

### Without `TimeOfDay`

Directions in NullGraph are **light travel directions**—where light goes, not where the sun is.

```ts
const sunTravelDirection = new Float32Array([-0.4, -1.0, -0.35]);
day.setSunDirection(sunTravelDirection);
env.setSkyParams({});
env.invalidateBake();
```

`setSunDirection` updates both `AtmosphereSky` and its owning `Atmosphere`.
`setSkyParams({})` serializes the changed source state into the skybox and bake buffers;
`invalidateBake()` requests fresh IBL. The sky-view LUT updates when `atmosphere.getStages()` next
runs.

---

## 2. The LUT chain

| LUT | Default size | Rebuild cadence | Purpose |
|---|---:|---|---|
| Transmittance | 256×64 | First `getStages()` call | Fraction of light surviving from altitude and zenith angle to space. |
| Multiple scattering | 32×32 | First `getStages()` call | Isotropic higher-order scattering approximation. |
| Sky view | 192×108 | Every `getStages()` call by default | Radiance by azimuth and horizon-relative angle for the current sun and altitude. |

The first `getStages()` returns all three stages in dependency order. Later calls return only the
sky-view stage, subject to `skyViewInterval`. Returned arrays are persistent; an idle interval
returns a shared empty array rather than allocating.

The sky-view texture uses a nonlinear vertical mapping concentrated around the horizon. At the
default dimensions, at least 50 of 108 rows fall inside ±22.5°. The kernel uses an 8×4 workgroup so
192×108 dispatches exactly. Transmittance uses 8×8; each multiple-scattering texel is reduced by one
64-thread workgroup.

### Memory

| Texture | Default bytes |
|---|---:|
| Transmittance | 131,072 |
| Multiple scattering | 8,192 |
| Sky view | 165,888 |
| **Total** | **305,152 bytes (~298 KiB)** |

All three textures use baseline-core, filterable `rgba16float`. No optional WebGPU feature is
required.

---

## 3. `Atmosphere`

### Constructor

```ts
const atmosphere = new Atmosphere(engine, config?);
```

`engine.device` must already exist. Construction allocates the textures, sampler, one 16-byte
sky-view parameter uniform, and all three compute pipelines. It does not dispatch them; dispatch
begins when stages returned by `getStages()` are included in `engine.render()`.

### `AtmosphereConfig`

Distances are **kilometres**. Scattering and absorption coefficients are **m⁻¹**.

| Field | Default | Meaning |
|---|---:|---|
| `groundRadius` | `6360` | Spherical ground radius, km. |
| `atmosphereTopRadius` | `6460` | Atmosphere-top radius, km; default thickness is 100 km. |
| `rayleighScattering` | `[5.802e-6, 13.558e-6, 33.100e-6]` | RGB molecular scattering, m⁻¹. |
| `rayleighScaleHeight` | `8` | Rayleigh exponential scale height, km. |
| `mieScattering` | `3.996e-6` | Grey aerosol scattering, m⁻¹. |
| `mieAbsorption` | `4.40e-6` | Grey aerosol absorption, m⁻¹. |
| `mieScaleHeight` | `1.2` | Mie exponential scale height, km. |
| `mieG` | `0.8` | Cornette–Shanks anisotropy; strictly between −1 and 1. |
| `ozoneAbsorption` | `[0.650e-6, 1.881e-6, 0.085e-6]` | RGB absorption, m⁻¹. Ozone scatters nothing. |
| `ozoneCentre` | `25` | Centre of the triangular ozone profile, km. |
| `ozoneHalfWidth` | `15` | Half-width of the ozone profile, km. |
| `transmittanceSize` | `[256, 64]` | Must be divisible by its 8×8 workgroup. |
| `multipleScatteringSize` | `[32, 32]` | Validated against 8×8; dispatch is one 64-thread group per texel. |
| `skyViewSize` | `[192, 108]` | Must be divisible by its 8×4 workgroup. |
| `transmittanceSteps` | `40` | Optical-depth integration steps. |
| `multipleScatteringSteps` | `20` | Steps along each sampled direction. |
| `skyViewSteps` | `32` | View-ray steps per sky-view texel. |
| `skyViewInterval` | `1` | Run sky view every N calls to `getStages()`. |

`EARTH_ATMOSPHERE`, `DEFAULT_TRANSMITTANCE_SIZE`, `DEFAULT_MULTIPLE_SCATTERING_SIZE`, and
`DEFAULT_SKY_VIEW_SIZE` are exported for inspection and custom configuration.

### Methods

| Method | Notes |
|---|---|
| `setSunDirection(dir)` | Copies and normalizes a light travel direction. Normally reached through `AtmosphereSky` and `TimeOfDay`. |
| `setCameraAltitude(heightKm)` | Altitude above `groundRadius`; finite, non-negative, and no greater than atmosphere thickness. |
| `getStages()` | Uploads sun/altitude and returns ordered work. First call: 3 stages; normally thereafter: sky view only. |
| `sunTransmittanceUv(out)` | Writes the ground-to-current-sun LUT UV into a required two-float output. Allocation-free. |
| `destroy()` | Destroys all three LUT textures and the sky-view parameter buffer. |

### Read-only properties

| Property | Type | Purpose |
|---|---|---|
| `config` | `Required<AtmosphereConfig>` | Fully resolved configuration. |
| `medium` | `AtmosphereMedium` | The same configuration through its physics-only interface. |
| `sunDirection` | `Float32Array` | Persistent internal travel-direction view; treat as read-only. |
| `transmittanceView` | `GPUTextureView` | Sampled physical transmittance. |
| `multipleScatteringView` | `GPUTextureView` | Sampled higher-order scattering. |
| `skyView` | `GPUTextureView` | Current visible-sky radiance. |
| `sampler` | `GPUSampler` | Linear LUT sampler; U repeats and V clamps. |

The GPU resources are exposed for later atmosphere consumers such as aerial perspective. Ordinary
applications should use `AtmosphereSky` rather than binding them manually.

---

## 4. `AtmosphereSky`

```ts
const sky = new AtmosphereSky(atmosphere, {
    sunAngularRadius: 0.266,
    sunDiscIntensity: 20,
});
```

`AtmosphereSky` implements `SkySource` and owns no GPU resource. Its fixed `paramFloats` is 10. It
samples the owning atmosphere's sky-view LUT and composites the sun analytically because a 0.532°
disc cannot be represented by a 192-column full-azimuth texture.

### `AtmosphereSkyConfig`

| Field | Default | Meaning |
|---|---:|---|
| `sunAngularRadius` | `0.266` | Angular **radius** in degrees. |
| `sunDiscIntensity` | `20` | Scene-relative HDR multiplier after atmospheric transmittance. |

Both values are fixed at construction. To change either, install a new source before the sky pass
has been compiled.

### Source members

| Member | Notes |
|---|---|
| `name` | `'AtmosphereSky'` |
| `paramFloats` | `10` |
| `setSunDirection(dir)` | Forwards to `Atmosphere.setSunDirection`. |
| `setCelestialState(state)` | Copies `state.sunDir`; normally called through `TimeOfDay`. |
| `writeParams(out, base)` | Allocation-free direction, disc, visibility, and transmittance-UV serialization. |
| `getShaderLayout(base)` | Skybox facet using group 2 bindings 14–16. |
| `getBakeShaderLayout(base)` | Environment Stage-A facet using group 0 bindings 3–5. |

It does not set `needsSourceCube`, so unlike `EquirectSky` it is valid when the environment does not
retain its source cube.

### Direct source or day/night composite

```ts
env.setSky(new AtmosphereSky(atmosphere));

// Or:
env.setSky(new SkyComposite({
    day: new AtmosphereSky(atmosphere),
    night: new NightSky(),
}));
```

`SkyComposite` propagates each child's bake-specific layout and texture entries. The atmosphere is
therefore a drop-in day child; no material or `EnvironmentSystem` branch knows its concrete type.

### `cheapSky`

```ts
env.setSkyParams({ cheapSky: 0 });   // default: live sky-view LUT
env.setSkyParams({ cheapSky: 1 });   // EnvironmentSystem's prefiltered cube
```

`cheapSky: 1` bypasses `sampleSky` in the visible sky pass. Atmosphere stages are still required
when the IBL bake samples `AtmosphereSky`.

---

## 5. Time of day and altitude

`TimeOfDay.applyTo({ environment: env })` reaches the installed source. With a composite:

1. `SkyComposite` forwards celestial state to both children;
2. `AtmosphereSky` forwards `sunDir` to `Atmosphere`;
3. `NightSky` receives sun, moon, phase, illumination, and sidereal time;
4. the composite updates `dayFactor`;
5. `EnvironmentSystem` uploads source params and applies its angular IBL re-bake gate.

Camera altitude is separate because `TimeOfDay` has no camera dependency:

```ts
// If world Y is metres and sea level is world Y = 0:
atmosphere.setCameraAltitude(Math.max(camera.position[1], 0) / 1000);
```

The input is altitude above the atmosphere's spherical ground, not distance from the world origin
and not absolute radius. The module does not infer or read back the engine camera buffer. Above the
configured atmosphere top the LUT is undefined, so the setter throws instead of silently clamping.

`setCameraAltitude()` updates the live sky-view LUT but does not automatically dirty
`EnvironmentSystem`'s prefiltered cube or SH. If altitude-dependent IBL matters, declare it at an
application-chosen threshold rather than invalidating on every tiny camera movement:

```ts
if (Math.abs(cameraHeightKm - lastIblHeightKm) >= 0.25) {
    env.invalidateBake();
    lastIblHeightKm = cameraHeightKm;
}
```

The safe stale direction is intentional: the visible sky remains current every frame, while the
more expensive reflection refresh is caller-controlled like every other non-celestial source
change.

---

## 6. Pure CPU helpers

`atmosphereMath` is exported from `null-graph/environment` and needs no device or browser.

```ts
import { atmosphereSunRadiance, transmittance } from 'null-graph/environment';

const rgb = new Float32Array(3);
transmittance(0, 1, rgb);       // sea level, straight up
atmosphereSunRadiance(5, rgb);  // sun five degrees above the horizon
```

| Helper | Purpose |
|---|---|
| `resolveAtmosphereConfig(config?)` | Fill defaults and validate the complete configuration. |
| `validateAtmosphereConfig(config)` | Validate an already resolved `Required<AtmosphereConfig>`. |
| `rayleighDensity(heightKm, medium?)` | Molecular exponential density. |
| `mieDensity(heightKm, medium?)` | Aerosol exponential density. |
| `ozoneDensity(heightKm, medium?)` | Triangular ozone-layer density. |
| `raySphereDistance(radius, radialDistance, mu)` | First positive ray/sphere distance, or −1. |
| `distanceToAtmosphereBoundary(heightKm, mu, medium?)` | Distance to ground or atmosphere top. |
| `transmittance(heightKm, mu, out?, medium?, steps?)` | Trapezoidal RGB extinction integral; `mu` is cosine from local zenith. |
| `atmosphereSunRadiance(elevDeg, out?, medium?)` | Ground-level physical sun transmittance. |
| `rayleighPhase(cosTheta)` | Normalized Rayleigh phase function. |
| `cornetteShanksPhase(cosTheta, g?)` | Normalized Mie phase function; `g` defaults to 0.8. |
| `skyViewVFromHorizonAngle(angleRad)` | Horizon-relative angle to nonlinear sky-view V. |
| `horizonAngleFromSkyViewV(v)` | Exact inverse mapping. |
| `transmittanceUv(heightKm, mu, medium?, out?)` | Physical coordinates to Bruneton LUT UV. |
| `transmittanceUvToHeightMu(u, v, medium?, out?)` | LUT UV back to height and zenith cosine. |

Helpers with optional `out` allocate only when omitted. `Atmosphere.sunTransmittanceUv` instead
requires `out` because it is intended for the frame path.

`skySunRadiance(elevationDeg, out?, turbidity?)` remains exported from `todMath` with its historical
signature. The turbidity argument is accepted for compatibility but the physical body no longer
uses it; the function delegates to `atmosphereSunRadiance` with the default Earth medium.

---

## 7. Shader generators and bindings

Most applications do not call these directly, but all generators are public:

| Generator | Output and bindings |
|---|---|
| `buildTransmittanceWGSL(config?)` | Complete `cs_main`, 8×8. Binding 0: destination. |
| `buildMultipleScatteringWGSL(config?)` | Complete `cs_main`, 64 threads. Bindings 0–2: transmittance, sampler, destination. |
| `buildSkyViewWGSL(config?)` | Complete `cs_main`, 8×4. Bindings 0–4: params, transmittance, multiple scattering, sampler, destination. |
| `buildAtmosphereSkyLayout(base, group)` | `ShaderSkyLayout` using bindings 14–16 in the supplied group. |
| `buildAtmosphereBakeLayout(base)` | Stage-A layout using group 0 bindings 3–5. |

Skybox declarations:

```wgsl
@group(2) @binding(14) var sky_atm_trans: texture_2d<f32>;
@group(2) @binding(15) var sky_atm_view: texture_2d<f32>;
@group(2) @binding(16) var sky_atm_sampler: sampler;
```

These belong only to the dedicated sky batch and consume **zero material bindings**. The
environment bake binds the same resources at group 0 bindings 3–5.

---

## 8. Errors

| Condition | Error |
|---|---|
| Top radius does not exceed ground | `Atmosphere: atmosphereTopRadius (<top> km) must exceed groundRadius (<ground> km).` |
| Negative or non-finite coefficient | `Atmosphere: <term> coefficient <value> is negative; extinction must be non-negative or transmittance exceeds 1.` |
| `mieG` outside the open interval | `Atmosphere: mieG <value> is outside (-1, 1); Cornette-Shanks is singular at the endpoints.` |
| Non-positive scale height/ozone half-width | `Atmosphere: <field> <value> must be positive.` |
| Invalid LUT dimensions | `Atmosphere: <lut> dimensions <w>x<h> must be multiples of its <x>x<y> workgroup size.` |
| Invalid march count or interval | `Atmosphere: <field> <value> must be a positive integer.` |
| Negative/non-finite altitude | `Atmosphere: camera altitude <h> km must be finite and non-negative.` |
| Camera above atmosphere thickness | `Atmosphere: camera at <h> km is above atmosphereTopRadius <top> km; the sky-view LUT is undefined outside the medium. Clamp the camera or raise the top.` |

The accepted altitude is checked against `atmosphereTopRadius - groundRadius`. For the default
medium, the maximum is 100 km. All checks run at construction or explicit setter boundaries, not
inside `engine.render()`.

---

## 9. Lifetime and limitations

- Create one `Atmosphere` per independent medium/LUT chain and reuse it.
- Install its source before the first `EnvironmentSystem.getSkyPass()`; changing source layout
  after sky-pipeline compilation is forbidden.
- Call `Atmosphere.destroy()` during teardown. `EnvironmentSystem.destroy()` does not destroy
  source-owned atmosphere resources.
- Aerial perspective—the air between camera and scene geometry—is separate. This phase affects the
  sky and baked IBL, not geometry.
- Clouds, volumetric shadows, atmospheric refraction, and spectral rendering are not included.
- `skyViewInterval > 1` can visibly lag sun and altitude changes. The default is 1.
- The CPU `skySunRadiance` seam uses the default Earth medium. A custom `AtmosphereConfig` does not
  automatically replace `TimeOfDay`'s CPU medium.

---

## 10. Testing

```bash
npm run test:atmosphere
```

The headless suite pins transmittance bounds and wavelength order, the closed-form vertical
Rayleigh integral, planet occlusion, LUT mapping round trips, nonlinear horizon concentration,
phase-function normalization and forward peak, the `skySunRadiance` compatibility seam,
`SkyComposite` resource propagation, named errors, and parse-valid generated WGSL.

It is included in:

```bash
npm run test:environment
```

The current suite does not perform live-device LUT readback or image comparisons. Generated
kernels and both source injection sites are parsed with `wgsl_reflect`; GPU-vs-CPU readback,
24-hour range, banding, measured sun diameter, and altitude image tests remain browser-harness
work.