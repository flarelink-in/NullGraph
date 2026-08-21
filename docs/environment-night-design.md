# NullGraph Environment — Phase 3: Night Sky — Technical Design

**Target:** `src/extras/environment/` (`nightMath`, `nightShaders`, `NightSky`, `StarField`,
`SkyComposite`)
**Package:** `null-graph/environment`
**Status:** `v1.3` — ✅ **implemented.** Headless suite: `npm run test:night`; proposed Phase 3 GPU
readback tests remain pending.
**Series:** [1 · Sky sources](./environment-sky-design.md) →
[2 · Time of day](./environment-timeofday-design.md) → **3 · Night sky** →
[4 · Physical atmosphere](./environment-atmosphere-design.md) →
[5 · Aerial perspective](./environment-aerial-design.md)
**Depends on:** Phase 1's per-pixel `sampleSky` (without it a star is not representable at any cube
resolution) and Phase 2's `gmstRad`, `moonDir`, `moonPhase`, `dayFactor` — the four fields Phase 2
produces for a consumer that did not exist yet.

---

## 1. Purpose

Night is not day with the exposure pulled down. It is a different set of light sources, a different
angular scale, and a dynamic range that does not fit in the render target.

### The single most important pre-existing fact

**Phase 1 is what makes this phase possible, and it is worth restating why.** Before Phase 1 the
visible sky was a 128²-per-face cube at **0.70° per texel**
(`EnvironmentSystem.ts:136`, `envShaders.ts:603`). A star's angular diameter is under 0.01 arcsecond
— roughly **250,000×** smaller than one texel of the thing that would have displayed it.

That is not a resolution shortfall to be closed by allocating more memory. **A point source is not
a texture lookup.** Raising the cube to 4096² still puts every star inside one texel, and its
brightness then depends on where within that texel it happened to fall — so the whole sky
scintillates as the world rotates, in a pattern locked to the cube grid rather than to the
atmosphere.

Stars must be **drawn**, not sampled. §5 does that; §4 explains why the moon is the opposite case.

### The second fact: the range does not fit

| Source | Luminance (cd/m²) |
|---|---|
| Sun disc | ~1.6 × 10⁹ |
| Overcast daytime sky | ~10³ – 10⁴ |
| Full moon disc | ~2.5 × 10³ |
| Clear moonless night sky | ~1 × 10⁻⁴ |

Sun to night sky is **~13 orders of magnitude**. `rgba16float` — the format of every environment
texture and every post target in this engine (`lighting-ibl-design.md` §3) — spans `6.10 × 10⁻⁵` to
`65504` in normals: **~9 orders**.

Four orders short. This is not a tonemapping problem that a better curve solves; the values do not
survive being stored. So:

**Night content is authored against the scene-relative unit fixed in
[`environment-sky-design.md`](./environment-sky-design.md) §4.4** — `1.0` = a white Lambertian
surface in direct midday sun — with one explicit `nightGain` that lifts the night branch into
range. The gain is a stated artistic parameter with a physical default, not a fudge factor
discovered during tuning. Phase 1 fixed the reference in the first document precisely so this
phase is choosing a number rather than retrofitting a convention.

Ratios *within* night are preserved exactly, which is what actually matters: a full moon is
~125× integrated starlight (≈0.25 lux against ≈0.002 lux), and that ratio is what makes a moonlit
scene read as moonlit.

### Non-goals for Phase 3

- **Physically correct twilight.** §7's ramp is authored and is the weakest thing in this series.
  It is superseded by Phase 4, which gets twilight right by construction because ozone absorption
  is in the model. The *blend machinery* survives; the ramp does not.
- **Planets.** Five are naked-eye visible and they move against the stars, so they are ephemerides,
  not a catalogue. Filed (§12).
- **Aurora, zodiacal light, airglow structure.** Airglow appears only as a constant term in the
  night radiance.
- **Stars in reflections.** §5.5 shows the energy is negligible and the image would be a smear.
- **Light pollution as a simulation.** A horizon glow term, not a model.

### Design principles

1. **Conserve flux, not radiance.** The rule that makes a point source resolution-independent, and
   the one most often got wrong (§5.3).
2. **Resolved things go in the evaluator; unresolved things get drawn.** The moon is 0.52° — three
   texels in a 512² source cube, so it belongs in `sampleSky` and therefore lands in the IBL bake
   for free. A star is 0.00003° and belongs in a draw call. One sentence decides the architecture of
   both (§4, §5).
3. **The sky is a composite.** Day model + night model + blend, as a first-class `SkySource` built
   with the decorator pattern shadows already uses (design-principles §4). This is what lets Phase 4
   replace the day half without touching the night half — and later lets weather blend clear ↔
   overcast with the same machinery (§6).
4. **State the physics, then the artistic override.** Moonlight is not blue (§8.2). Ship the blue,
   name it, default it on.

---

## 2. Architecture

```
   env.setSky(new SkyComposite({
       day:   new PreethamSky({ ... }),          // or AtmosphereSky, Phase 4
       night: new NightSky({ ... }),
       blend: 'dayFactor',                       // driven by TimeOfDay
   }));

   engine.render([
      ...vis.getComputeStages(),
      scenePass,
      env.getSkyPass({ ... }),      // batch 0: sampleSky per pixel  (moon, milky way, night dome)
                                    // batch 1: draw(3, 9100) additive  (stars)
      ...post.getPassSequence(),
   ]);
```

```
   ┌──────────────┐   ┌──────────────┐
   │ PreethamSky  │   │ NightSky     │   moon disc + phase, milky way,
   │ sky_day_*    │   │ sky_night_*  │   night dome, light pollution
   └──────┬───────┘   └──────┬───────┘
          └────────┬─────────┘
                   ▼
          ┌──────────────────┐
          │ SkyComposite     │  fn sampleSky(dir) =
          │ renames children │    mix(sky_night_sampleSky(dir),
          │ emits ONE entry  │        sky_day_sampleSky(dir), dayFactor)
          └────────┬─────────┘
                   │  ONE evaluator — Phase 1's invariant survives composition
        ┌──────────┴──────────┐
        ▼                     ▼
   bake Stage A          skybox fragment
   (moon → IBL)               │
                              │   ┌────────────────────────────┐
                              └──▶│ StarField — SECOND batch   │
                                  │ draw(3, n), additive blend │
                                  │ never enters the bake      │
                                  └────────────────────────────┘
```

---

## 3. Integration facts

| Fact | Source | Consequence |
|---|---|---|
| `RenderPipeline` iterates `node.batches` **in array order** and draws a vertex-buffer-less batch as `draw(3, instanceCount)`. | `RenderPipeline.ts:103,147` | The star field is a second batch in the sky pass, drawn after it, with no geometry and no engine change. |
| `createBatch` binds `batch.storageBuffer` at group 0 binding 1, sized `strideFloats × maxInstances`. | `BatchManager.ts:94-102` | The star catalogue **is** that buffer: 4 floats × 9,100. No new binding, no new upload path. |
| `blend` is per-batch `PipelineConfig`. | `types.ts:22` | Additive stars over an already-drawn sky. Core. |
| The sky pass loads rather than clears (`loadOp: 'load'`). | `EnvironmentSystem.ts:431-441` | Both batches composite onto the opaque result correctly. |
| `Camera.updateTime(simTime)` writes camera-uniform float 99. | `Camera.ts:54-55` | Scintillation is free — no new uniform. |
| Phase 2 produces `gmstRad`, `moonDir`, `moonPhase`, `moonIllumination`, `dayFactor`. | [Phase 2](./environment-timeofday-design.md) §4.1 | Every input this phase needs already exists. |
| The source cube is `sourceSize`² per face, 512 by default → **0.176°/texel**. | `EnvironmentSystem.ts:121` | The moon (0.52°) is ~3 texels — resolved. This is the number that puts the moon in `sampleSky` and stars in a draw call. |
| A `SkySource`'s non-entry-point identifiers are prefixed `sky_`. | [Phase 1](./environment-sky-design.md) §4.2 | Composition is a mechanical rename. Phase 1 paid for this with one naming convention. |

---

## 4. The moon

### 4.1 Why it is in the evaluator

At 0.52° the moon covers ~3 texels of a 512² source cube face. It is **resolved**, so putting it in
`sampleSky` means:

- it is drawn per-pixel and sharp in the skybox;
- it enters the source cube, therefore the GGX prefilter, therefore the specular cube — so **a
  moonlit water surface gets a real specular streak** with zero extra machinery;
- it enters the SH-L2 projection, so moonlight contributes ambient without a second code path.

None of that is additional work. It is the payoff of Phase 1's decision that one function feeds
both the image and the bake.

### 4.2 The disc

```wgsl
fn sky_night_moon(dir: vec3<f32>) -> vec3<f32> {
    let cosAng = dot(dir, -p.moonDir);            // moonDir is TRAVEL direction
    if (cosAng < p.moonCosRadius) { return vec3<f32>(0.0); }

    // Surface normal of the lunar sphere at this point on the disc.
    let d      = normalize(dir - (-p.moonDir) * cosAng);
    let sinAng = sqrt(max(0.0, 1.0 - cosAng * cosAng));
    let r      = clamp(sinAng / p.moonSinRadius, 0.0, 1.0);
    let n      = normalize(-p.moonDir * sqrt(max(0.0, 1.0 - r * r)) + d * r);

    let nDotL  = dot(n, -p.moonSunDir);           // the terminator
    if (nDotL <= 0.0) { return p.earthshine; }    // the ashen light on the dark limb

    // Lommel-Seeliger: the Moon does NOT limb-darken like a Lambertian sphere.
    let nDotV  = max(dot(n, -dir), 1e-3);
    let brdf   = nDotL / (nDotL + nDotV);
    return p.moonRadiance * brdf + p.earthshine;
}
```

Three details that are the difference between a moon and a white circle:

- **Lommel–Seeliger, not Lambert.** A Lambertian sphere at full phase is markedly brighter at the
  centre than at the limb. The real full moon is **almost uniformly bright to the edge** — the
  regolith's single-scattering behaviour, which Lommel–Seeliger captures in one division. This is
  the single most recognisable property of the moon and the cheapest to get right.
- **Earthshine.** The dark limb of a crescent is faintly visible — sunlight reflected off Earth
  onto the moon and back. Scaled by `1 − moonIllumination`, since a thin crescent from here means a
  near-full Earth from there. One multiply; it is what makes a crescent read as a sphere rather
  than a decal.
- **The terminator is geometry, not a texture.** It comes from `moonSunDir`, which Phase 2 already
  computes as the sun's direction, so phase and position can never disagree.

**Not shipped:** an albedo map (maria/highlands). It is one 512² texture and one binding on a bind
group with room, and it changes nothing structurally — filed as content in §12, not design.

### 4.3 Moon radiance

`moonRadiance` is computed on the CPU by `nightMath`, from Phase 2's state:

```
E_moon ≈ E_sun · albedo(0.12) · illuminationFraction · phaseBoost(phaseAngle) · extinction(elev)
```

`phaseBoost` is the **opposition surge** — the moon at full is ~30% brighter than the
inverse-square-and-phase geometry predicts, because shadow-hiding in the regolith peaks when the
sun is directly behind the observer. A smooth-sphere model makes a full moon visibly dim. One
exponential term.

Everything here is the same shape as Phase 2 §5.2's `sunRadiance`, including its obligation: it is
a **CPU mirror of the WGSL that draws the disc**, and U5 pins them together (design-principles
§13.1).

### 4.4 The re-bake gate needs amending

Phase 2 §6 gates the IBL re-bake on the **sun's** angular motion. With the moon in `sampleSky`,
that is now incomplete: the moon moves ~14.5°/hour in the diurnal sense, and a moonlit specular
streak on water baked from a stale cube trails the visible disc.

**Amendment:** the gate is `max(sunMotion, moonMotion · moonWeight)`, where `moonWeight` is
`dayFactor`-inverted so a daytime moon does not force re-bakes for a body contributing nothing.
Pinned by U11. This is exactly the kind of coupling that only appears when the second phase lands —
recorded here rather than left as a surprise regression in the Phase 2 suite.

---

## 5. Stars

### 5.1 The catalogue

~9,100 stars to magnitude 6.5 — the naked-eye limit under a dark sky. Four floats each:

| Float | Field |
|---|---|
| 0 | right ascension (radians) |
| 1 | declination (radians) |
| 2 | apparent visual magnitude |
| 3 | colour index B−V |

4 × 4 B × 9,100 = **145 KB**, uploaded once. Stars are fixed in celestial coordinates, so the
buffer never changes after load — all motion is one matrix in the vertex shader (§6).

The catalogue is data, not code: `setStars(Float32Array)` adopts a caller-supplied array —
`LightSystem.setRawData`'s contract (roadmap constraint #2), so it can be fetched, generated, or
posted from a Worker. A built-in procedural fallback (~2,000 hash-placed stars with a plausible
magnitude distribution) means the module works with **no asset**, matching `GradientSky`'s role in
Phase 1.

### 5.2 The draw

A second batch in the sky pass. No vertex buffer, no index buffer:

```ts
this.starBatch = engine.createBatch(skyPass, {
    shaderCode: buildStarWGSL({ environmentGroup }),
    strideFloats: 4,
    maxInstances: starCount,
    blend: ADDITIVE,
    depthWriteEnabled: false,
    depthCompare: 'less-equal',
});
```

`RenderPipeline` takes its `draw(3, instanceCount)` path (`RenderPipeline.ts:147`) — one triangle
per star, circumscribing the disc, positioned in the vertex shader from `instance_index`.

**One draw call for the entire night sky.** This is the engine's default shape, not a trick: the
star field is exactly the case GPU-driven instancing exists for.

### 5.3 Flux conservation — the rule

A star has no angular extent worth resolving. What it has is a **flux**, and its screen footprint
is set by the point-spread function of the eye or lens, not by geometry.

So the sprite is sized in **pixels** — a constant ~1.5 px radius with a smooth radial falloff —
and its radiance is `k · 10^(−0.4 · magnitude)`.

The consequence is the property that matters: **covered pixel count is resolution-independent, so
total image energy is too.** Render at 720p or 4K and a magnitude-2 star contributes the same
amount of light to the frame. U3 asserts it across a 4× resolution change.

The alternative — a constant *angular* size — fails at both ends: at low resolution the sprite is
sub-pixel and stars flicker with sub-pixel motion; at high resolution they inflate into visible
discs. Both failures read as "the star field looks wrong at this resolution," which is not a
diagnosis anyone acts on.

The magnitude scale itself is definitional and pinned by U2: **Δm = 5 is exactly a factor of 100**
in flux.

### 5.4 Colour and scintillation

Colour comes from B−V through a blackbody approximation — a real sky is not white dots, and Betelgeuse
and Rigel are a recognisable pair. One polynomial.

Scintillation is **atmospheric**, so it scales with air mass: strong near the horizon, nearly absent
at the zenith. Reusing Phase 2 §5.2's Kasten–Young air mass makes it correct for free rather than
an authored "twinkle near the bottom" ramp:

```wgsl
let amp   = p.twinkle * clamp(airMass(elevDeg) / 38.0, 0.0, 1.0);
let flick = 1.0 + amp * sin(camera.simTime * 6.0 + starPhase);
```

`38` is air mass at the horizon under Kasten–Young — the same constant Phase 2 derives. Two systems
using one number rather than two tuned ones.

### 5.5 Stars are not in the bake, and that is correct

The star batch draws into the skybox only. It never enters the source cube, so it contributes
nothing to the IBL. Justified with a number rather than asserted: integrated starlight is
**~0.002 lux** against a full moon's **~0.25 lux** and direct sun's **~100,000 lux**. Its SH
contribution is ~1% of the moon's and ~2 × 10⁻⁸ of the sun's — below the quantisation of the
`rgba16float` it would be stored in.

Excluding them also avoids a real hazard: a magnitude-1 star landing in one texel of the source
cube is a **firefly** entering the GGX prefilter. The Karis solid-angle mip select
(`envShaders.ts:277-284`) exists to suppress exactly that, and there is no reason to spend it
suppressing energy that should not be there.

---

## 6. `SkyComposite` — the sky is always two skies

### 6.1 The mechanism

Two `SkySource`s in one module means two functions called `sampleSky`. The engine already has a
pattern for this: `ShadowSystem.wrapLighting` renames `getIncidentLight` and emits a decorated
replacement (design-principles §4). `SkyComposite` does the same thing:

```ts
const day   = rename(this.day.getShaderLayout(base),        'sky_day_');
const night = rename(this.night.getShaderLayout(base + dn), 'sky_night_');
```

`rename` rewrites the two entry points and every `sky_`-prefixed identifier — which is why Phase 1
mandated that prefix (§4.2 there). The composite emits one `sampleSky`, so **Phase 1's U2 invariant
survives composition**: the bake kernel and the skybox still contain byte-identical bodies. U6
asserts exactly one `fn sampleSky(` in the composed module and exactly two renamed children.

### 6.2 Why this is not just a night feature

The composite generalises past this phase:

| Blend | Children | Phase |
|---|---|---|
| day ↔ night | `PreethamSky`, `NightSky` | 3 |
| **physical** day ↔ night | `AtmosphereSky`, `NightSky` | **4 — one constructor argument** |
| clear ↔ overcast | any two | weather |
| a hand-authored hero sky ↔ the driven one | any two | content |

Phase 4 replaces the day half by changing one argument. That is the seam paying for itself a second
time, and it is why the composite is introduced here rather than being folded into `NightSky`.

### 6.3 The parameter budget

Each child writes into its own slice of the sky batch's storage buffer at its own `paramBase`;
the composite's `paramFloats` is the sum plus its own. The buffer is sized by demand
(design-principles §10) and lives on a bind group with room (Phase 1 §4.3). No pressure on the
material group.

---

## 7. Twilight — the honest part

Between solar elevation 0° and −18° the sun is below the horizon and the sky is still lit, by air
the observer cannot see. Preetham does not model it; `GradientSky` has no concept of it.

Phase 3's answer is an **authored ramp** on `dayFactor`, keyed to the conventional bands:

| Band | Solar elevation | What the ramp does |
|---|---|---|
| Civil | 0° → −6° | horizon warm, zenith deepening; brightest stars appear |
| Nautical | −6° → −12° | the blue hour; the horizon is still distinguishable |
| Astronomical | −12° → −18° | only a residual glow toward the sun's azimuth |
| Night | below −18° | `dayFactor` = 0 |

**This is the weakest thing in the series, and it is stated as such.** It is a fitted curve
standing in for radiative transfer, it has no altitude dependence, and it will look wrong from a
mountaintop or an aircraft. It ships because a day cycle that cuts to black at sunset is worse, and
because the alternative is blocking Phase 3 on Phase 4.

**What survives Phase 4:** `SkyComposite`, `dayFactor`, and every band threshold. What is deleted:
the ramp's colour table. Phase 4 produces twilight correctly because **ozone absorption** is in the
model — it is the term that keeps the zenith blue after sunset instead of grey, and the ramp is
approximating its output. That is not a coincidence, it is the ramp's job description.

---

## 8. Moonlight as a light

### 8.1 The handoff already exists

Phase 2 §7.2 built it: one holder of the shadow slot, direction and slot written together, cascades
invalidated in the same call. Phase 3 adds no mechanism — it supplies the colour and intensity that
Phase 2's `moonRadiance` field was reserved for.

Moon shadows are real and sharp-edged, and they are gated by `moonMinIllumination` (default 0.35):
a crescent does not cast a usable shadow, and letting it try produces a shadow map rendered for
light nobody can see.

### 8.2 Moonlight is not blue

The moon's regolith reflectance rises toward the red, so moonlight is spectrally **slightly redder
than sunlight**. The blue of a moonlit scene is the **Purkinje shift**: at scotopic light levels
vision moves to the rods, whose peak sensitivity sits at a shorter wavelength, so blues brighten
relative to reds.

It is a property of the observer, not of the light. The engine renders light.

So: `purkinje` is a **named artistic parameter, defaulting to on**, applied as a scotopic-weighted
desaturation-and-shift keyed to `nightGain`. Shipping it is right — every audience expects blue
moonlight and the effect is real, just not radiometric. Naming it is what stops it being
rediscovered as a bug in the moon's colour temperature two years from now.

---

## 9. Test suite

**Headless** (`npm run test:night`):

| # | Asserts |
|---|---|
| U1 | The celestial→horizon transform puts declination +90° at altitude exactly = latitude, for 64 values of GMST. Zero tolerance. Catches every sign and axis error in §6's matrix in one case. |
| U2 | Magnitude → flux: `f(m) / f(m + 5) == 100` within 1e-6, over m ∈ [−1.5, 6.5] |
| U3 | **Flux conservation.** Total star-sprite energy is invariant under a 4× resolution change within 1% (§5.3) |
| U4 | Lommel–Seeliger limb: at full phase, disc-centre and 0.95-radius radiance differ by < 15%; the Lambert form differs by > 60%. Both bounds — the second is what proves the test would catch a regression to Lambert |
| U5 | `moonRadiance` CPU mirror matches its WGSL at 32 (phase, elevation) pairs to 1e-5 (§4.3) |
| U6 | The composed module contains exactly one `fn sampleSky(`, and both children's entry points renamed. Phase 1's U2 (identical bodies in bake and skybox) still passes under composition |
| U7 | `dayFactor` is monotonically non-increasing over solar elevation +10° → −20°, and is exactly 0 below `nightDeg` |
| U8 | Earthshine → 0 at full moon and is maximal at new; the terminator's illuminated fraction matches Phase 2's `moonIllumination` within 2% across a synodic month |
| U9 | The night branch returns finite non-negative radiance for 512 directions × 32 times, including straight down |
| U10 | Scintillation amplitude → 0 at the zenith and is maximal at the horizon, using Phase 2's air mass — asserting the constant is *shared*, not duplicated |
| U11 | The amended re-bake gate (§4.4) triggers on moon motion when `dayFactor == 0`, and does **not** when a daytime moon is up |
| U12 | `setStars` adopts by reference: mutating the caller's array after the call changes what the next upload sends |
| U13 | No backtick in any generated WGSL; no `PI` redeclaration; no identifier collision between the two children |

**GPU integration:**

| # | Asserts |
|---|---|
| T1 | Moon disc angular diameter from a readback matches `moonAngularRadius × 2` within one pixel |
| T2 | A full moon's disc is uniformly bright to the limb (centre/edge < 1.15); switching to Lambert breaks it |
| T3 | Star positions match the catalogue under the §6 transform for 20 named stars within 0.1° |
| T4 | The star field costs exactly **one** draw call — read from the profiler, not asserted structurally |
| T5 | A moonlit water plane shows a specular streak — i.e. the moon really did reach the prefiltered cube (§4.1) |
| T6 | A full day/night cycle produces no NaN and no negative radiance in the swapchain, sampled every simulated 10 minutes |

T5 is the one that verifies the architectural claim rather than the maths: it fails if the moon
ended up in the skybox but not in the bake, which is exactly what a well-meaning "just draw the
moon as a sprite too" refactor would cause.

---

## 10. Errors

| Condition | Message |
|---|---|
| `setStars` with `length % 4 != 0` | `StarField: star records are 4 floats (ra, dec, magnitude, bv); got <n>, which is not a multiple of 4.` |
| star count > `maxInstances` | `StarField: <n> stars exceeds maxInstances <m> fixed at batch creation. Pass maxStars at construction.` |
| `NightSky` without a `TimeOfDay` supplying `gmstRad` | `NightSky: no sidereal time source. The star field needs TimeOfDay.gmstRad; a fixed skyRotation is available for a static night.` |
| `SkyComposite` children with different `environmentGroup` | `SkyComposite: day source targets group <a>, night source group <b>. Both share one bind group.` |
| `SkyComposite` with a child that is itself unresolved | `SkyComposite: child "<name>" has no source set.` |
| `nightGain <= 0` | `NightSky: nightGain must be > 0; 0 renders an unlit night, which is what omitting NightSky does (design §1).` |
| Moon configured, `capacity.directional < 2` | (raised by `TimeOfDay`, Phase 2 §10) |

---

## 11. Engine touchpoints

### 11.1 Core engine — **none**

Two batches in one pass, `draw(3, n)`, additive blend, a storage buffer sized at creation, and
`simTime` in the camera uniform. All shipped.

### 11.2 Module changes — additive

| Change | File | Nature |
|---|---|---|
| `nightMath.ts` — celestial transform, magnitude/flux, lunar photometry, blackbody | new | pure, worker-safe |
| `nightShaders.ts`, `NightSky.ts`, `StarField.ts`, `SkyComposite.ts` | new | |
| `rename()` on a `ShaderSkyLayout` | `SkySource.ts` | additive; the `wrapLighting` pattern |
| Re-bake gate takes moon motion | `TimeOfDay.ts` | **amends Phase 2 §6** (§4.4) |
| `EnvironmentSystem.getSkyPass` returns a pass a second batch can be added to | `EnvironmentSystem.ts` | additive accessor |

### 11.3 Optional — filed, not needed

- **A lunar albedo map.** One 512² texture on a bind group with room. Content, not design.
- **MSAA on the sky pass** would soften sub-pixel star edges. Not needed — the radial falloff is
  smooth by construction, and TAA (v2.0) covers it properly.
- **Separate star exposure.** Filed. `nightGain` plus `AutoExposure`'s clamp pair (Phase 2 §8) is
  already two controls; a third invites them to fight.

---

## 12. Build order

1. **`nightMath` + the celestial transform.** U1 first, before anything draws. A star field built on
   an unverified transform is a bug that looks like a content problem.
2. **`StarField`, fixed rotation.** One draw call, correct positions, correct relative brightness.
   U2, U3, T3, T4. Night has stars; nothing else has changed.
3. **Wire `gmstRad`.** The sky rotates correctly with Phase 2's clock. T3 now holds across time.
4. **`NightSky` dome + light pollution + airglow.** A night that is not pure black behind the stars.
5. **The moon.** Disc, phase, Lommel–Seeliger, earthshine. U4, U5, U8, T1, T2. **And T5** — this is
   the step where the moon appears in reflections, which is the proof it went through the bake.
6. **`SkyComposite` + the twilight ramp.** U6, U7. The cycle stops cutting to black.
7. **Moonlight + Purkinje.** The shadow slot changes hands with something visible on both sides.
8. **The re-bake gate amendment.** U11. Left until the moon is actually in the bake, because before
   step 5 there is nothing for it to gate.

### Deferred

- Planets (ephemerides, not a catalogue — five bodies, Meeus-style series, ~0.1° accuracy).
- Meteors, satellites, the ISS.
- Aurora — a genuinely different rendering problem (a volumetric emissive curtain), and closer to
  the volumetric branch than to the sky.
- Lunar libration and topographic limb detail.
- Atmospheric refraction lifting bodies near the horizon (~0.57° at elevation 0) — smaller than the
  lunar position error already accepted in Phase 2 §5.1, and it properly belongs to Phase 4.
