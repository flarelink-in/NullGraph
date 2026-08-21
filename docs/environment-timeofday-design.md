# NullGraph Environment — Phase 2: Time of Day — Technical Design

**Target:** `src/extras/environment/` (`todMath`, `TimeOfDay`)
**Package:** `null-graph/environment`
**Status:** `v1.3` — ✅ **implemented.** Headless suite: `npm run test:tod`.
**Series:** [1 · Sky sources](./environment-sky-design.md) → **2 · Time of day** →
[3 · Night sky](./environment-night-design.md) →
[4 · Physical atmosphere](./environment-atmosphere-design.md) →
[5 · Aerial perspective](./environment-aerial-design.md)
**Depends on:** Phase 1's `SkySource` seam and `skySunRadiance` — specifically, on the fact that a
sky source can be driven by a *direction* rather than by authored colours.
**Consumers:** `LightSystem`, `ShadowSystem`, `EnvironmentSystem`, `PostChain.AutoExposure` — all
existing, none modified.

---

## 1. Purpose

Make one clock the only place a celestial body's position exists.

### The single most important pre-existing fact

**The sun's direction is currently stored in four places, and nothing reconciles them.**

| Copy | Where | Form |
|---|---|---|
| Sky | `ProceduralSkyConfig.sunDir` → `skyUniformF32[12..14]` | travel direction, normalized on write (`EnvironmentSystem.ts:191-199`) |
| Direct light | `DirectionalLightProxy.direction` → light record floats 8–10 | travel direction (`DirectionalLightProxy.ts:12-16`) |
| Shadows | `ShadowSystem.setLight(direction)` → `this.lightDir` | travel direction, normalized (`ShadowSystem.ts:309-311`) |
| Indirect | baked into the SH from the sky cube | implicit |

The API documentation already concedes this, in `api/environment.md` §1.1:

> **Point the `LightSystem`'s directional light along the same `sunDir`** and the direct and
> indirect halves agree. Nothing enforces that — the two systems do not know about each other — so
> it is worth storing the sun direction once in your own state and feeding both.

**Phase 2 is that "your own state," promoted to a system.** Not because a helper is nice, but
because "nothing enforces that" is a bug generator with a long tail: a shadow cast from one
direction while the specular highlight sits at another is a defect that survives review, because
each system is individually correct.

The failure mode is worth naming precisely. Nothing about it is loud. There is no error, no
artefact with a name, no frame drop — just a scene that looks subtly wrong in a way that gets
blamed on the shadow bias, then on the normal maps, then on the tonemapper.

### What this document is *not*

`TimeOfDay` renders nothing. It owns **zero GPU resources**: no buffer, no texture, no pipeline, no
bind group, no compute stage, no render pass. It is a pure producer that writes into contracts that
already exist. If that seems like too little to warrant a document, note that it is the same shape
as `Camera` — and that the alternative is the four-copy problem above.

### Non-goals for Phase 2

- **Drawing anything.** No stars, no moon disc, no twilight gradient — Phase 3.
- **Atmospheric scattering.** The sun's colour here comes from Phase 1's `skySunRadiance`, an
  authored extinction curve. Phase 4 replaces the implementation behind that signature.
- **Weather, seasons as content.** A date gives you a declination, which gives you seasonal day
  length for free. Cloud cover, temperature, precipitation are the weather branch.
- **Astronomical rigour beyond ~0.01° solar / ~0.3° lunar.** Stated in §5; the error budget is set
  by a 0.53° disc, not by an ephemeris.
- **Interpolating between two shadow-casting bodies.** The sun→moon handoff is a hard switch, and
  §7.2 explains why a crossfade would cost a second cascade set.

### Design principles

1. **One producer, many consumers, no back-channel.** `TimeOfDay` writes; nothing reads back from
   it into the sky. A cycle here would make "who is authoritative" unanswerable.
2. **Dirtiness is declared, not inferred** — the discipline that made the shadow atlas affordable
   (`shadows-atlas-design.md` §8.1). Applied here to the IBL re-bake, it is the single largest
   performance decision in this document (§6).
3. **`out` parameters, not allocation.** A clock that runs every frame and allocates a `[x,y,z]`
   is a GC spike with a schedule (design-principles §5).
4. **Invariants over almanacs.** A test that hardcodes "sunset in London on 21 June" introduces a
   second source of truth that will disagree with the first. §9 pins derivable properties instead.

---

## 2. Architecture

```
                        ┌──────────────────────────────┐
     setTime(t) ───────▶│  TimeOfDay                   │
     advance(dt)        │  ── zero GPU resources ──    │
                        │  todMath: solar, lunar, GMST │
                        └──────────────┬───────────────┘
                                       │  state (pooled, non-escaping)
        ┌───────────────┬──────────────┼──────────────┬────────────────┐
        ▼               ▼              ▼              ▼                ▼
  ┌───────────┐  ┌────────────┐  ┌──────────┐  ┌───────────┐  ┌──────────────┐
  │ SkySource │  │ LightSystem│  │ Shadow   │  │ re-bake   │  │ AutoExposure │
  │ sun+moon  │  │ sun & moon │  │ System   │  │ gate      │  │ clamp pair   │
  │ direction │  │ records    │  │ setLight │  │ (§6)      │  │              │
  └───────────┘  └────────────┘  └──────────┘  └───────────┘  └──────────────┘
```

Per frame:

```ts
tod.advance(dt);                 // or tod.setTime(hours)
tod.applyTo(bindings);           // writes every consumer above, allocation-free
if (tod.envDirty) engine.render([...env.getBakeStages(), ...rest]);
else               engine.render([...rest]);
```

`envDirty` is the whole performance story and it is one boolean.

---

## 3. Integration facts (verified against this repository)

| Fact | Source | Consequence |
|---|---|---|
| A light record is 16 floats; direction is 8–10, colour 4–6, intensity 7, shadow slot 14. | `light-proxies.ts:6-56`, `DirectionalLightProxy.ts:12` | `TimeOfDay` writes through the existing cursor proxies. No new buffer, no new upload path. |
| `shadowSlot` is stored as `slot + 1`, with **0 meaning unshadowed**. | `light-proxies.ts:44` | The sun→moon handoff is two writes to one field. §7.2 shows why they must be adjacent. |
| `ShadowSystem` keeps exactly one directional `lightDir`, used by `buildCascades` for every cascade. | `ShadowSystem.ts:109,431,468` | One directional shadow caster at a time. That is a design fact, not a limitation to route around. |
| Cascades use a **rotation-invariant sphere fit**, so tile extent does not change with light yaw. | `shadows-design.md` §5.3 | A moving sun does **not** cause the cascade shimmer that a box fit would. The sphere fit was already the fix. |
| **Caster extrusion** pushes the light-space near plane back to catch off-screen casters. | `shadows-design.md` §5.4 | This is what degrades at grazing sun: the extruded depth range grows without bound as elevation → 0. §7.1 sets the clamp. |
| `updateMode: 'amortized'` re-runs Stage A + one prefilter mip + SH **every frame**, unconditionally. | `EnvironmentSystem.ts:356-365` | Unconditional is the problem §6 solves. The mode has no idea whether the sun moved. |
| `Camera.updateTime(simTime)` writes float 99 of the camera uniform. | `Camera.ts:54-55` | The GPU already has a clock. Phase 3's star twinkle reads it; Phase 2 does not need it. |
| `AutoExposure` clamps to `[minExposure, maxExposure]`, defaults `[0.1, 8.0]`, with an explicit note that the floor "stops night scenes being pulled to daylight." | `AutoExposure.ts:13-23,63` | The coupling point already exists and is already documented as a day/night concern. §8. |

---

## 4. Data model

### 4.1 `CelestialState` — what the clock produces

No GPU struct. This is a TypeScript object, allocated **once** and mutated in place.

| Field | Type | Meaning |
|---|---|---|
| `sunDir` | `Float32Array(3)` | **travel** direction — where the light goes. Matches every consumer's convention. |
| `sunElevationDeg` | `number` | above the horizon; negative below |
| `sunAzimuthDeg` | `number` | from north, clockwise |
| `sunRadiance` | `Float32Array(3)` | linear HDR, scene-relative (sky §4.4), already extinction-attenuated |
| `moonDir` | `Float32Array(3)` | travel direction |
| `moonElevationDeg` | `number` | |
| `moonPhase` | `number` | 0 = new, 0.5 = full, wrapping at 1 |
| `moonIllumination` | `number` | illuminated fraction, `(1 − cos(elongation)) / 2` |
| `moonRadiance` | `Float32Array(3)` | scaled by illumination and extinction |
| `gmstRad` | `number` | Greenwich mean sidereal time — **Phase 3 consumes this**; nothing in Phase 2 does |
| `activeBody` | `'sun' \| 'moon' \| 'none'` | who holds the shadow slot (§7.2) |
| `dayFactor` | `number` | 1 above `civilDeg`, 0 below `nightDeg`, smooth between — **Phase 3's blend parameter** |

Two fields are produced for a phase that does not exist yet. That is deliberate: `gmstRad` and
`dayFactor` are where Phase 3 attaches, and drawing the seam now is cheaper than discovering later
that sidereal time was never computed and the star field has to derive it from a different clock
(design-principles §4).

**The state does not escape.** `tod.state` is not exposed. Accessors follow the `out`-parameter
pattern:

```ts
tod.getSunDirection(out?: Float32Array): Float32Array;   // fills `out`, or an internal view
```

The internal view is documented as **valid until the next `update()`** — the same contract as a
`BaseLightCursor`, which the engine already relies on (design-principles §6).

### 4.2 `TimeOfDayConfig`

| Field | Default | Meaning |
|---|---|---|
| `mode` | `'artistic'` | `'artistic'` \| `'astronomical'` (§5.1) |
| `latitudeDeg` | `45` | astronomical mode |
| `longitudeDeg` | `0` | astronomical mode |
| `date` | now | astronomical mode; a `Date`, converted to Julian day once |
| `sunTiltDeg` | `23.44` | artistic mode: how far the sun's arc leans off vertical |
| `timeScale` | `1` | simulated seconds per real second |
| `rebakeThresholdDeg` | `0.25` | §6 — the whole performance argument |
| `sunShadowMinElevDeg` | `3` | below this the sun stops casting (§7.1) |
| `moonShadowMinElevDeg` | `10` | the moon casts only when high and bright enough |
| `moonMinIllumination` | `0.35` | a crescent does not cast a usable shadow |
| `civilDeg` / `nightDeg` | `−6` / `−18` | the `dayFactor` ramp |
| `exposureCoupling` | `'clamp'` | `'off'` \| `'clamp'` \| `'absolute'` (§8) |

---

## 5. Celestial mechanics

### 5.1 Two clock modes, one output

**`'artistic'` (default).** The sun rides a great circle tilted `sunTiltDeg` from vertical, driven
by a 0–24 dial. No geography, no date, no declination. `elevation = f(hour)` in closed form,
~10 flops.

This is the default because the question a scene actually asks is *"where is the sun at 18:30"*,
and answering it should not require a latitude. A dial that maps monotonically onto a sky is what
lighting artists want; a correct ephemeris is what they work around.

**`'astronomical'`.** Real latitude, longitude, and date.

- **Solar position** — the NOAA/Meeus low-precision algorithm: Julian day → mean longitude →
  mean anomaly → ecliptic longitude → declination δ and equation of time → hour angle H →

  ```
  sin(altitude) = sin φ · sin δ + cos φ · cos δ · cos H
  ```

  ~40 flops, accurate to about **0.01°** across 1950–2050. Against a 0.53° disc that is a 2%
  positional error — two orders of margin.

- **Lunar position** — Meeus's truncated series (five longitude terms, two latitude), ecliptic →
  equatorial → horizon. Accurate to roughly **0.2–0.3°**, which is a noticeable fraction of a
  0.52° disc. Stated rather than hidden: the moon will sit up to half its own width from where an
  ephemeris puts it, and if that ever matters, the fix is more terms in one pure function, not a
  redesign.

- **GMST** — from the Julian day, for Phase 3's star field.

Both modes produce a `CelestialState` and nothing downstream can tell which ran.

### 5.2 Sun colour — and the mirror

The sun's colour comes from Phase 1's `skySunRadiance(elevDeg)`, evaluated on the **CPU** by
`todMath`, mirroring the WGSL that draws the disc.

**This is a CPU mirror of a GPU kernel and it will drift** (design-principles §13.1). It is
inserted knowingly, because the alternative is a GPU readback of the sun's colour, and a readback
in the frame path is forbidden (design-principles §12). Phase 1 pins the two together with U6;
Phase 2 re-runs that assertion as U2 against its own consumption path.

The baseline model is air-mass extinction. Optical path length uses Kasten–Young:

```
AM(h) = 1 / (sin h + 0.50572 · (h + 6.07995)^(−1.6364))       h in degrees
```

which gives AM = 1 at the zenith and **AM ≈ 38 at the horizon** — the number that makes sunsets
red, because per-channel optical depth is wavelength-dependent and blue is scattered out first.

The whole model is ~20 flops and one three-element optical-depth constant. **Phase 4 replaces its
body with a transmittance-LUT evaluation** without changing the signature, which is exactly what
the seam was drawn for.

### 5.3 What "sunset" means, three times over

Three distinct thresholds, all of which someone will call sunset:

| Event | Condition | Governs |
|---|---|---|
| **Shadow handoff** | elevation < `sunShadowMinElevDeg` (3°) | who holds the shadow slot (§7) |
| **Geometric sunset** | elevation = 0 | the disc touches the horizon |
| **Light extinction** | intensity → 0 via air mass | when direct sun stops mattering |

They must fire in that order, and the ordering is asserted (U6). Backwards, the sun keeps casting
hard shadows while below the horizon — a bug that renders as "shadows point the wrong way at
dusk," which nobody attributes to a clock.

---

## 6. The re-bake gate

The largest decision in this document, and it is four lines of code.

### 6.1 The problem

`updateMode: 'amortized'` re-runs the source cube, one prefilter mip, and the SH projection **every
frame, unconditionally** (`EnvironmentSystem.ts:356-365`). It has no idea whether the sun moved.
With Phase 1's evaluator in Stage A, that is `512² × 6` = **1.57 M evaluations per frame** to
re-derive a sky that, in real time, changed by **0.00007°**.

### 6.2 The rule

Re-bake when the sun has accumulated more than `rebakeThresholdDeg` of angular motion since the
last bake. Not "every N frames" — **angular motion**, which is the quantity the bake is actually a
function of.

One rule, correct at every time scale:

| `timeScale` | Sun motion / frame @60fps | Frames between bakes @0.25° |
|---|---|---|
| 1× (real time) | 0.00007° | **~3,600** (one per minute) |
| 60× | 0.0042° | ~60 |
| 1000× | 0.069° | **~4** |
| 10000× | 0.69° | **every frame**, and the threshold is exceeded per frame |

A frame-counter heuristic is wrong at both ends: it burns 3,599 bakes out of 3,600 in real time,
and it under-bakes at high time scale. The angular rule needs no tuning per project.

### 6.3 The coupling nobody expects

`'amortized'` refreshes the six prefilter mips **round-robin**, so for six frames after a trigger
the specular cube is **mixed-age** — some mips describe the old sun, some the new.

At 0.25° that is invisible. At a threshold of, say, 10°, it renders as a rough metal surface whose
reflection *steps* through six brightness levels — and it will be reported as a mipmapping bug.

So the threshold is not a free knob:

```
rebakeThresholdDeg ≲ (the smallest sun motion visible in a mirror) / prefilterMips
```

The default 0.25° is half a sun diameter, one sixth of which is well below perceptible. A validator
warns above 2° (§10) rather than leaving it to be discovered.

### 6.4 What the gate does *not* cover

Anything else that invalidates the bake — turbidity changed, the source swapped, weather blended —
must also set the flag. `TimeOfDay` owns celestial motion only, so `env.invalidateBake()` is
public, and the gate is `envDirty || env.bakeInvalidated`. Declared dirtiness, exactly as the
shadow atlas does it, including its honest weakness: a caller who changes turbidity without saying
so gets a stale sky until the sun moves.

---

## 7. Driving the consumers

### 7.1 Direct light

Per frame, through the existing cursors — no allocation, no new upload path:

```ts
sun.seek(sunIndex);
sun.direction  = state.sunDir;              // travel direction, matching every other copy
sun.color      = state.sunRadiance;         // §5.2, already extinction-attenuated
sun.intensity  = sunIntensity(elevation);   // → 0 at or before elevation 0
```

`sunIntensity` reaching zero **at or before** geometric sunset is asserted (U6). A sun at exactly
0° elevation with full intensity rakes every surface at grazing incidence and produces shadows the
length of the scene — the single worst-looking frame in an ungated day cycle.

### 7.2 Shadows — the handoff

The hazard, precisely. `shadowSlot` writes `slot + 1` into light-record float 14
(`light-proxies.ts:44`). If sun and moon both hold slot 0, the receiver resolves whichever light
the shading loop reaches last. Shadows then come from one body while the key light is the other —
and it is intermittent, because it depends on light ordering.

So the handoff is **one function, three writes, no early return**:

```ts
private setShadowCaster(body: 'sun' | 'moon' | 'none'): void {
    if (body === this.activeBody) return;
    this.sunCursor.shadowSlot  = body === 'sun'  ? this.slot : -1;   // -1 → stores 0
    this.moonCursor.shadowSlot = body === 'moon' ? this.slot : -1;
    if (body !== 'none') this.shadows.setLight(this.directionOf(body));
    this.shadows.invalidate();          // cached cascades describe the old direction
    this.activeBody = body;
}
```

Three properties, none optional:

- **Exactly one holder.** Asserted across a full simulated day at one-minute steps (U4).
- **Direction and slot move together.** Splitting them across two call sites is how you get a frame
  of shadows cast from a body that no longer holds the slot — design-principles §9's rule, in its
  original form: *whatever suppresses a draw must suppress its compute, in the same place.*
- **The cache is invalidated.** Cascades are cached against declared dirtiness
  (`shadows-atlas-design.md` §8.1) and a light-direction change is not something they can infer.

**Why `sunShadowMinElevDeg = 3` and not 0.** Caster extrusion pushes the light-space near plane
back to catch off-screen casters (`shadows-design.md` §5.4). As elevation → 0 the extruded depth
range grows without bound, so depth precision collapses at exactly the moment shadows are longest
and most visible. The sphere fit keeps the cascade *tile* rotation-invariant, so there is no
shimmer — but nothing bounds the extrusion. 3° is where the extrusion is still bounded and the sun
contributes little direct light anyway.

**Why the handoff is a switch, not a crossfade.** `ShadowSystem` holds one `lightDir`
(`ShadowSystem.ts:109`). A crossfade needs two live cascade sets — double the atlas allocation,
double the caster culling, for a transition that happens when both bodies are near the horizon and
neither casts a usable shadow. The switch is scheduled into the window where the sun is below 3°
and the moon below 10°, so `activeBody` is `'none'` on both sides of it and there is nothing to
fade.

### 7.3 The sky

```ts
env.sky.setSunDirection(state.sunDir);
env.sky.setMoonDirection(state.moonDir);      // ignored by Phase 1 sources; Phase 3 reads it
```

Phase 1 sources declare which inputs they use; passing a moon direction to `PreethamSky` is a
no-op, not an error. That is what keeps Phase 3 from having to change this call.

---

## 8. Exposure

`AutoExposure` already exists and already clamps to `[minExposure, maxExposure]`, with a comment
naming the day/night case (`AutoExposure.ts:13-14`). Three modes:

| Mode | Behaviour |
|---|---|
| `'off'` | `TimeOfDay` writes nothing. The default for anyone already managing exposure. |
| `'clamp'` (default) | Drives the clamp *pair* from `dayFactor`, leaving adaptation to the existing histogram. Night gets a lower floor and ceiling; the eye still adapts within the window. |
| `'absolute'` | Sets a target exposure from solar elevation directly, adaptation off. Deterministic — for a cinematic or a regression test. |

`'clamp'` is the default because it is **additive to a shipped, working effect** rather than a
second exposure authority. Two systems both deciding exposure is the four-copy problem again, in a
place where it manifests as slow pumping that looks like a tonemapper bug.

The clamp pair is where the night's absolute darkness gets decided, and Phase 3 §3 shows why that
cannot be left to physical units.

---

## 9. Test suite

**Headless** (`npm run test:tod`). Pure functions — no GPU, no device, no browser.

**On method:** not one assertion hardcodes an almanac value. A published sunrise time is a *second
source of truth*, and when it disagrees you cannot tell which side is wrong. Every case below is
derivable from the algorithm's own definition, so a failure localizes.

| # | Asserts |
|---|---|
| U1 | Solar elevation is symmetric about solar noon: `elev(noon − t) == elev(noon + t)` within 0.02° for 100 offsets. Catches a sign error in the hour angle, which is the classic one. |
| U2 | `sunRadiance` CPU matches Phase 1's WGSL `skySunRadiance` at 32 elevations to 1e-5 — the §5.2 mirror (design-principles §13.1) |
| U3 | **Polar day.** At latitude 67°N on the June solstice, solar elevation > 0 for all 1,440 minutes. At 67°S the same day, < 0 for all of them. Pins declination sign, latitude sign, and the tilt magnitude in one case. |
| U4 | **Slot exclusivity.** Stepping a full simulated day at one-minute resolution, `sunCursor.shadowSlot >= 0` and `moonCursor.shadowSlot >= 0` are never both true (§7.2) |
| U5 | Every handoff writes a direction and invalidates the cache in the same call — verified with a recording stub, asserting call order, not just call count |
| U6 | **Threshold ordering.** For every minute of a simulated day: shadow handoff fires at elevation ≥ `sunShadowMinElevDeg`, and `sunIntensity == 0` at every elevation ≤ 0 (§5.3) |
| U7 | **Re-bake gate.** 3,600 frames at `timeScale: 1` triggers ≤ 2 bakes; the same at `timeScale: 1000` triggers ≥ 800. One rule, both ends (§6.2) |
| U8 | Synodic period: `moonPhase` returns to its initial value after 29.530588 days within 0.02 |
| U9 | Full moon ⟺ elongation ≈ 180°: `moonIllumination > 0.99` exactly when elongation > 172° |
| U10 | Both clock modes produce a normalized `sunDir` (‖·‖ = 1 ± 1e-6) for 10,000 sampled times |
| U11 | **Zero allocation.** 1,000 `advance()` + `applyTo()` calls allocate 0 bytes, measured with patched typed-array constructors — the harness that already exists |
| U12 | `getSunDirection(out)` writes `out` and returns it; the no-arg form returns the same object every call (no per-frame view allocation) |

**No GPU tests.** `TimeOfDay` produces no pixels. Its output is verified where it lands — Phase 1's
T-series for the sky, the shadow suite for cascade direction.

---

## 10. Errors

| Condition | Message |
|---|---|
| `'astronomical'` with no latitude | `TimeOfDay: mode 'astronomical' requires latitudeDeg/longitudeDeg. Use mode 'artistic' for a geography-free dial.` |
| `\|latitudeDeg\| > 90` | `TimeOfDay: latitudeDeg <v> is outside [-90, 90].` |
| `rebakeThresholdDeg <= 0` | `TimeOfDay: rebakeThresholdDeg must be > 0; 0 re-bakes every frame, which is what 'amortized' already does unconditionally.` |
| `rebakeThresholdDeg > 2` | **Warning:** `TimeOfDay: rebakeThresholdDeg <v>° exceeds 2°. With prefilterMips=<n>, the amortized cube stays mixed-age for <n> frames and rough reflections will step visibly (design §6.3).` |
| `applyTo` given a `LightSystem` with `capacity.directional < 2` while a moon is configured | `TimeOfDay: a moon light needs capacity.directional >= 2 (sun + moon); got <n>.` |
| `moonShadowMinElevDeg < sunShadowMinElevDeg` | `TimeOfDay: moon and sun shadow windows overlap; both bodies would qualify to cast simultaneously (design §7.2).` |
| `exposureCoupling: 'clamp'` with no `AutoExposure` in the chain | `TimeOfDay: exposureCoupling 'clamp' needs an AutoExposure effect in the PostChain; use 'absolute' to drive Tonemap.exposure directly.` |

The fourth is the only warning in this series. It earns it: the condition is legal, produces a
correct image at rest, and only misbehaves in motion — the kind of thing an error would over-police
and silence would under-report.

---

## 11. Engine touchpoints

### 11.1 Core engine — **none**

Nothing here touches `src/core/`. `TimeOfDay` allocates no GPU object and emits no node.

### 11.2 Module changes — additive

| Change | File | Nature |
|---|---|---|
| `todMath.ts` — solar, lunar, GMST, air mass, phase | new | pure, worker-safe, no imports |
| `TimeOfDay.ts` | new | |
| `EnvironmentSystem.invalidateBake()` | `EnvironmentSystem.ts` | additive, 3 lines (§6.4) |
| `SkySource.setSunDirection` / `setMoonDirection` | `SkySource.ts` | additive to the Phase 1 interface, default no-op |

`ShadowSystem`, `LightSystem`, `PostChain`: **unchanged**. Every write goes through a public
setter or an existing cursor proxy.

### 11.3 Optional — filed, not needed

- **A second directional shadow set** would allow a real sun/moon crossfade. Filed: it doubles
  atlas allocation and caster culling for a transition scheduled into a window where neither body
  casts (§7.2). Revisit only if a scene needs simultaneous sun and moon shadows, which is a real
  daytime-moon case and a genuinely different feature.
- **Worker-side clock.** `todMath` is pure, so a Worker can produce the state and post 16 floats.
  Not needed — the whole update is ~200 flops.

---

## 12. Build order

1. **`todMath` solar + artistic mode.** No consumers wired. U1, U3, U10 pass. Verifiable with
   `console.log`.
2. **Sun → `LightSystem` + `SkySource`.** A day cycle renders. Sun, sky, and specular now agree by
   construction — the four-copy problem is gone at step 2.
3. **The re-bake gate.** U7. This is where an animated sky becomes affordable; before it, step 2
   costs 1.57 M evaluations per frame.
4. **Shadow handoff for the sun alone** (`'sun'` ↔ `'none'`). U4, U5, U6. Sunset stops raking.
5. **Lunar position + phase.** U8, U9. Nothing draws the moon yet, and nothing needs to — the moon
   is a light and a direction before it is an image.
6. **Moon → light + shadow slot.** The handoff has two live bodies. U4 now covers a real switch.
7. **Exposure coupling.** Last, because it is the one piece that is a matter of taste, and it
   should be evaluated against a cycle that is already correct.

### Deferred

- Seasons as *content* (foliage colour, snow line) — reads `date`, belongs to weather.
- Eclipses. The geometry falls out of the sun and moon directions already computed; the rendering
  does not, and it is a novelty.
- Local sidereal time for observers off the prime meridian at high precision. `gmstRad` plus
  longitude is sufficient for a star field (Phase 3 §6.2).
- Refraction near the horizon (~0.57° lift at elevation 0). Smaller than the lunar error already
  accepted, and it interacts with Phase 4's atmosphere — it belongs there, if anywhere.
