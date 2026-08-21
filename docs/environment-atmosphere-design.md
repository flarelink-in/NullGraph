# NullGraph Environment — Phase 4: Physical Atmosphere — Technical Design

**Target:** `src/extras/environment/` (`atmosphereMath`, `atmosphereShaders`, `Atmosphere`,
`AtmosphereSky`)
**Package:** `null-graph/environment`
**Status:** `v2.0` — ✍ **designed, not implemented.**
**Series:** [1 · Sky sources](./environment-sky-design.md) →
[2 · Time of day](./environment-timeofday-design.md) →
[3 · Night sky](./environment-night-design.md) → **4 · Physical atmosphere** →
[5 · Aerial perspective](./environment-aerial-design.md)
**Depends on:** Phase 1's `SkySource` seam (this is a drop-in source), Phase 2's solar position and
`skySunRadiance` signature, Phase 3's `SkyComposite` (this replaces its *day* child, one constructor
argument).
**References:** Bruneton & Neyret 2008; Hillaire, *A Scalable and Production Ready Sky and
Atmosphere Rendering Technique*, EGSR 2020.

---

## 1. Purpose

Replace the fitted sky with the medium that produces it: Rayleigh scattering, Mie scattering, ozone
absorption, and the multiple scattering that makes a clear sky bright rather than black.

### The single most important pre-existing fact

**The physical model is evaluated 100× less often than the analytic model it replaces.**

Phase 1's `PreethamSky` runs per **pixel** — up to 2,073,600 evaluations per frame at 1080p. This
phase precomputes radiance into a **192×108 sky-view LUT** and reduces the per-pixel cost to one
bilinear fetch:

| | Evaluations / frame | Per-pixel cost |
|---|---|---|
| `PreethamSky` (Phase 1) | 2,073,600 | ~40 flops, closed form |
| `AtmosphereSky` (this phase) | **20,736** | one `textureSampleLevel` |

2,073,600 / 20,736 = **exactly 100**. The sky is smooth in angle, so sampling it at ~2° and
interpolating loses nothing — and the raymarch that would be unaffordable per pixel is trivial per
LUT texel.

This inverts the usual objection. "Physical atmosphere" is assumed to be the expensive option; at
1080p and above it is the **cheaper** one, and it gets cheaper as resolution rises because the LUT
does not.

The memory is equally unremarkable:

| LUT | Size | Format | Bytes | Rebuild |
|---|---|---|---|---|
| Transmittance | 256 × 64 | `rgba16float` | 131,072 | on atmosphere change |
| Multiple scattering | 32 × 32 | `rgba16float` | 8,192 | on atmosphere change |
| Sky-view | 192 × 108 | `rgba16float` | 165,888 | **per frame** |
| | | **total** | **~298 KB** | |

Under 300 KB — less than one 256² material texture.

### What this phase retires

| Retired | Replaced by | Survives |
|---|---|---|
| `PreethamSky` as the **default** day source | `AtmosphereSky` | Preetham stays available: 30 constants against 300 KB and a per-frame stage is still the right trade on a low-tier device |
| Phase 2 §5.2's authored air-mass extinction curve | a transmittance-LUT evaluation behind the **same** `skySunRadiance(elevDeg)` signature | the signature, the CPU mirror discipline, every caller |
| Phase 3 §7's authored twilight colour ramp | radiative transfer, which gets twilight right because **ozone is in the model** (§4.3) | `SkyComposite`, `dayFactor`, every band threshold |

Three deletions, zero caller changes. That is the seam from Phase 1 being spent, and it is the
reason this document is 400 lines rather than a rewrite.

### Non-goals for Phase 4

- **Aerial perspective** — the air between the camera and *geometry*. That is Phase 5, and it is
  separate because its integration surface is the material/post boundary, not the sky (§10.3).
- **Clouds.** A participating medium with 3D structure; a different branch entirely.
- **Volumetric shadows / god rays** through the atmosphere. Phase 5 §7.
- **Weather-driven turbidity as a system.** Mie density is an input here; who animates it is the
  weather branch's problem.
- **Spectral rendering.** Three channels at fixed representative wavelengths, like every real-time
  implementation of this model.
- **Other planets.** The constants are parameters, so it falls out — but it is not tested and not
  claimed.

### Design principles

1. **A LUT is a cache of a function, not a source of truth.** Every LUT here has a closed-form
   integrand the CPU can also evaluate. That is what makes §6's CPU mirror a mirror of *maths*
   rather than a forbidden GPU readback (design-principles §12).
2. **The seam holds or the phase is wrong.** If landing this requires editing a material, a light,
   or a caller, Phase 1 drew the contract in the wrong place (design-principles §14).
3. **Precompute what is constant, per-frame what is not.** Transmittance and multiple scattering
   depend only on the medium. Only the sky-view LUT depends on the sun and the camera's altitude.
4. **Detect nothing.** `rgba16float` storage, `textureSampleLevel` in compute, workgroup reduction.
   Core WebGPU throughout, as everywhere in this engine.

---

## 2. Architecture

```
   env.setSky(new SkyComposite({
       day:   new AtmosphereSky(atmosphere),    // ← the ONLY change from Phase 3
       night: new NightSky({ ... }),
   }));

   engine.render([
      ...atmosphere.getStages(),   // transmittance+multiscatter self-retire; sky-view is per-frame
      ...env.getBakeStages(),      // gated by TimeOfDay (Phase 2 §6)
      ...vis.getComputeStages(),
      scenePass,
      env.getSkyPass({ ... }),
      ...post.getPassSequence(),
   ]);
```

Ordering is load-bearing and free: `RenderPipeline` executes the array in order, and consecutive
compute stages share one pass with WebGPU's own sync between dispatches
(`RenderPipeline.ts:45-48`). The sky-view LUT is written before Stage A samples it, in the same
compute pass, with no explicit barrier.

```
   medium (Rayleigh, Mie, ozone, radii)
            │
            ▼
   ┌────────────────────┐        once, on change
   │ Transmittance LUT  │  256×64
   │ T(h, μ)            │───────────┐
   └─────────┬──────────┘           │
             ▼                      │
   ┌────────────────────┐           │
   │ Multi-scatter LUT  │  32×32    │
   │ Ψ(h, μ_s)          │───────┐   │
   └────────────────────┘       │   │
                                ▼   ▼
   sun dir ───────────▶ ┌────────────────────┐   PER FRAME (gateable)
   camera altitude ────▶│ Sky-view LUT       │   192×108, 32-step march
                        │ L(azimuth, zenith) │
                        └─────────┬──────────┘
                                  │
                   fn sampleSky(dir) = skyView(dir)
                                    + sunDisc(dir) · T(ground → sun)
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
              bake Stage A                skybox fragment
              (→ IBL, unchanged)          (per-pixel, unchanged)
```

---

## 3. Integration facts

| Fact | Source | Consequence |
|---|---|---|
| Consecutive `ComputeStageNode`s batch into **one** WebGPU compute pass, ordered by the sync model. | `RenderPipeline.ts:45-48` | The three LUTs and the bake chain the sky-view feeds are one pass. No barrier API needed. |
| `ComputeStageNode.enabled` skips clears **and** dispatch. | `RenderPipeline.ts:28` | The per-frame sky-view stage is gated by the same rule Phase 2 §6 gates the bake with. The mechanism already shipped, for shadows. |
| WGSL has **no `atomic<f32>`**; the SH projection therefore integrates in **one workgroup with a tree reduction**. | `envShaders.ts:377-421` | The multiple-scattering LUT has exactly this shape — 64 directions per texel, reduced in workgroup memory. A precedent in this repo, not a new pattern. |
| `rgba16float` is a core **write-only storage** format and is **filterable** when sampled. | WebGPU core format table | Every LUT is `rgba16float`. `rgba32float` would cost `float32-filterable` and lose the bilinear interpolation the whole approach depends on. |
| Phase 1's `wgslStructs` is emitted against a caller-supplied group index and binding base, and the sky batch's group 2 holds two entries. | [Phase 1](./environment-sky-design.md) §4.3 | Three LUT bindings on the sky batch and three on the bake kernel. Neither is a material. **Zero material bindings.** |
| `TimeOfDay` re-bake gating is angular, not per-frame. | [Phase 2](./environment-timeofday-design.md) §6 | The sky-view LUT can be gated by the same signal, though §5.4 argues for running it every frame anyway. |
| `EnvironmentSystem` keeps the source cube only when asked. | `EnvironmentSystem.ts:100` | `AtmosphereSky` is analytic — it does **not** need the source cube, unlike `EquirectSky`. |

---

## 4. The medium

### 4.1 Constants

Sea-level scattering/absorption coefficients at representative RGB wavelengths, in m⁻¹:

| Term | Coefficient | Profile |
|---|---|---|
| Rayleigh scattering | `(5.802, 13.558, 33.100) × 10⁻⁶` | exponential, scale height **8 km** |
| Mie scattering | `3.996 × 10⁻⁶` (grey) | exponential, scale height **1.2 km** |
| Mie absorption | `4.40 × 10⁻⁶` (grey) | same profile |
| Ozone absorption | `(0.650, 1.881, 0.085) × 10⁻⁶` | **tent**, centred 25 km, half-width 15 km |

Geometry: ground radius **6,360 km**, atmosphere top **6,460 km** (100 km thick).

The λ⁻⁴ signature in the Rayleigh triple — 33.1 blue against 5.8 red, a factor of 5.7 — is the sky
being blue, and it is *derived*, not authored. That is the whole difference between this phase and
every one before it.

### 4.2 Phase functions

```
Rayleigh:        p(θ) = 3/(16π) · (1 + cos²θ)
Mie (Cornette–Shanks, g = 0.8):
                 p(θ) = 3(1−g²)(1 + cos²θ) / ( 8π(2+g²)(1 + g² − 2g·cosθ)^{3/2} )
```

The Mie term's forward peak at `g = 0.8` is the white glare around the sun and the bright band
along the horizon. Turbidity is animated by scaling Mie density; the phase function is fixed.

### 4.3 Ozone — why it is not optional

Ozone contributes **no scattering at all**, only absorption, and its profile peaks at 25 km — far
above the air that makes daytime sky colour. Omitting it costs nothing during the day and is the
usual shortcut.

It is what makes **twilight blue**.

After sunset the light reaching the observer's zenith has travelled a long, nearly horizontal path
through the ozone layer. Ozone's Chappuis band absorbs in the **green-yellow** — note the triple
above: `0.650` red, `1.881` green, `0.085` blue. Green is absorbed ~2.9× more than red and ~22×
more than blue. The surviving light is blue, and that is the blue hour.

Without ozone, twilight in a Rayleigh+Mie model goes **grey then black** — which is exactly the
failure Phase 3 §7's authored ramp was hand-fitting around. The ramp was approximating a tent
function with three coefficients. This is why that ramp is deleted here and not merely improved.

---

## 5. The LUT chain

### 5.1 Transmittance — `T(h, μ)`, 256 × 64

Fraction of light surviving the path from altitude `h` toward zenith angle `μ` to the top of the
atmosphere. 40-step trapezoidal march of the combined extinction; the fundamental quantity
everything else calls.

Parameterization follows Bruneton's ρ/d mapping rather than linear (h, μ): it concentrates texels
where transmittance varies fastest — near-horizontal rays close to the ground, where a linear
mapping wastes most of its resolution on paths that are all ≈1.

Rebuilt only when the medium changes. In practice: once.

### 5.2 Multiple scattering — `Ψ(h, μ_s)`, 32 × 32

Single scattering alone gives a sky that is far too dark, especially away from the sun and near the
horizon. Hillaire's approximation treats higher orders as **isotropic and infinitely-scattering**,
which collapses them into a small 2D table.

**The kernel shape already exists in this repo.** Each texel integrates over a sphere of 64
directions; WGSL has no `atomic<f32>` — every atomic in this engine is a `u32` counter
(`clusteredComputeShader.ts:46,66`) — so it runs as **one workgroup of 64
threads with a tree reduction in workgroup memory** — structurally identical to
`buildSHProjectWGSL` (`envShaders.ts:377-421`), down to the `workgroupBarrier` placement. A reader
who has followed the SH projection has read this kernel.

1,024 texels × 64 directions × 20 steps ≈ 1.3 M samples, once.

### 5.3 Sky-view — `L(azimuth, zenith)`, 192 × 108, per frame

Radiance from the camera's altitude toward every direction, for the current sun. 32-step march
using both LUTs above.

**Vertical parameterization is non-linear about the horizon:**

```
v = 0.5 + 0.5 · sign(l) · sqrt( |l| / (π/2) )        l = view zenith angle relative to horizon
```

This puts **half the 108 rows inside ±22.5° of the horizon** — where a linear mapping would give
them 25%. The horizon is where the sky's gradient is steepest and where any banding is visible
against a hard silhouette.

Cost: 20,736 texels × 32 steps ≈ 663,552 samples. §1's headline.

### 5.4 Gate it, or run it every frame?

`ComputeStageNode.enabled` makes gating a one-line option, and Phase 2 §6 already computes the
signal. The recommendation is nonetheless to **run it every frame**:

- 663k samples is well under a millisecond on any device that runs this engine.
- Its inputs change more often than the bake's: the sun *and* camera altitude, so a camera climbing
  a hill invalidates it without the sun moving at all.
- A stale sky-view LUT is visible **directly** — it is what you look at. A stale prefiltered cube is
  visible only in reflections, which is why the bake tolerates the 0.25° gate and this does not.

The gate is exposed (`skyViewInterval`) for a device that needs it, defaulting to 1. Filed as a
knob, not used as a default.

### 5.5 The sun disc is not in the LUT

At 192 columns over 360° the sky-view LUT resolves **1.875° per column**. The sun is 0.53°. It
cannot be in there — and putting it there would smear a 3.5×-too-wide disc into the horizon band,
which is Phase 1 §1's original problem returning by a different route.

So `sampleSky` composites:

```wgsl
fn sampleSky(dir: vec3<f32>) -> vec3<f32> {
    var L = skyViewLookup(dir);                        // the atmosphere
    L += sunDisc(dir) * transmittanceToSun(dir);       // the disc, analytic, at TRUE angular size
    return L;
}
```

`sunDisc` is Phase 1's code, unchanged. What is new is that its colour is no longer authored: it is
the transmittance LUT evaluated along the sun ray, which is why a low sun reddens **and** dims by
the correct amount without a curve anyone tuned.

Phase 3's moon composites in the same place, through the same transmittance — so a moon near the
horizon goes orange for the same reason the sun does, from the same table.

---

## 6. The CPU mirror, resolved properly

Phase 2 fills a light record with the sun's colour, on the CPU, every time the gate fires. That
colour is now a transmittance-LUT value. A GPU readback in the frame path is forbidden
(design-principles §12).

**The resolution: the LUT is a cache of a closed-form integral, and the CPU evaluates the
integral.**

```ts
// atmosphereMath.ts — the same integrand the WGSL kernel marches.
export function transmittance(h: number, mu: number, out: Float32Array): Float32Array;
export function sunRadiance(elevDeg: number, out: Float32Array): Float32Array;
```

40 steps through the analytic density profile — a few microseconds, and only when the re-bake gate
fires (once per simulated minute at real-time rates), not per frame.

This is a **CPU mirror of a GPU kernel and it will drift** (design-principles §13.1). It is pinned
by U5: 64 (h, μ) pairs, CPU against a GPU readback of the LUT, to 1%. The 1% tolerance is honest —
the GPU value has been through `rgba16float` quantisation and the LUT's own parameterization, so
demanding 1e-5 would be pinning the interpolation rather than the physics.

The important structural point: the mirror is of **the medium**, not of the texture. If the LUT's
resolution or parameterization changes, the mirror does not, and U5 still passes. A mirror of the
sampling scheme would have to change with it — and that is the kind of mirror that drifts.

---

## 7. Altitude, and what comes free

The sky-view LUT is parameterized by the camera's altitude, so it is already correct for a camera
that is not on the ground. That yields, at no additional cost:

- The sky **darkens toward the zenith** with altitude, because there is less air above.
- The horizon **drops below eye level** and curves, because the ground sphere's tangent does.
- Above ~50 km the sky is essentially black with a thin bright limb.

None of this is a feature that was implemented; it is what the model does. Worth stating because
"physical" earns its name in exactly these cases — the ones nobody wrote code for.

**The boundary:** the sky-view LUT is valid for rays that reach the atmosphere's top or the ground
*with nothing in between*. A ray that hits **scene geometry** is not covered, and that is precisely
aerial perspective — [Phase 5](./environment-aerial-design.md).

---

## 8. Test suite

**Headless** (`npm run test:atmosphere`) — `atmosphereMath` is pure, so the physics is testable with
no device:

| # | Asserts |
|---|---|
| U1 | Transmittance is 1.0 at zero path length and **monotonically decreasing** in path length, for 64 (h, μ) pairs |
| U2 | Transmittance through a Rayleigh-only atmosphere, straight up from the ground, matches the closed-form `exp(−β · H)` within **1%** — the march is validated against an integral with a known answer |
| U3 | **Ordering by wavelength.** Vertical transmittance satisfies `T_r > T_g > T_b` (blue is scattered out first). Reversed, the sky is orange — and this catches a channel swap, which is otherwise invisible in a grey-sky test |
| U4 | Multiple-scattering orders decrease geometrically: order *n+1* is < 60% of order *n* for the first six. A model whose orders do not converge has an energy bug |
| U5 | **CPU ↔ GPU transmittance** at 64 (h, μ) pairs within 1% (§6, design-principles §13.1) |
| U6 | The sky-view V-parameterization is invertible over [0,1] to 1e-6, and ≥ 50 of 108 rows fall within ±22.5° of the horizon (§5.3) |
| U7 | **Phase-function normalization.** Both phase functions integrate to 1 over the sphere within 0.5%, by 4,096-sample quadrature. An unnormalized phase function is an energy leak that reads as "the sky is too bright," which gets fixed with an exposure knob |
| U8 | Cornette–Shanks peaks at θ = 0 and is monotonically decreasing to θ = π for g = 0.8 |
| U9 | `skySunRadiance` keeps Phase 1's signature and Phase 2's mirror test (U2 there) passes **unmodified** against this implementation — the seam claim, as an executable assertion |
| U10 | Ozone absorption is maximal in green: `β_o.g > β_o.r > β_o.b` (§4.3) |
| U11 | Zenith radiance is blue-dominant (`B > G > R`) at solar elevation 45°; horizon radiance toward the sun is red-dominant (`R > B`) at elevation 1°. **Sunsets are red** as a derived property, not an authored one |
| U12 | Twilight at −8° solar elevation is blue-dominant at the zenith **with** ozone and is not without it — the §4.3 claim, pinned by disabling one term |
| U13 | No backtick, no `PI` redeclaration, every generated module parses under `wgsl_reflect` |

U12 is the test that justifies shipping ozone, and it is written the way design-principles §13
requires: the failing variant is constructed and observed to fail.

**GPU integration:**

| # | Asserts |
|---|---|
| T1 | Sky-view LUT readback matches the CPU march at 32 directions within 3% |
| T2 | A 24-hour sweep produces no NaN, no negative radiance, and no value above 1e4 in the LUT |
| T3 | Sun disc angular diameter is the **true** 0.53°, and its colour matches `transmittanceToSun` within 2% at elevations 60°, 20°, 5°, 1° |
| T4 | No banding: adjacent sky-view texels differ by < 5% across the horizon band |
| T5 | Camera at 0 km, 10 km, and 80 km altitude produces monotonically decreasing zenith radiance (§7) |
| T6 | Swapping `PreethamSky` → `AtmosphereSky` in a live `SkyComposite` changes **no** caller code and requires no material rebuild — the whole series' architectural claim, executed |

---

## 9. Errors

| Condition | Message |
|---|---|
| `atmosphereTopRadius <= groundRadius` | `Atmosphere: atmosphereTopRadius (<a> km) must exceed groundRadius (<b> km).` |
| camera altitude above the atmosphere top | `Atmosphere: camera at <h> km is above atmosphereTopRadius <t> km; the sky-view LUT is undefined outside the medium. Clamp the camera or raise the top.` |
| any negative scattering/absorption coefficient | `Atmosphere: <term> coefficient <v> is negative; extinction must be non-negative or transmittance exceeds 1.` |
| `mieG` outside (−1, 1) | `Atmosphere: mieG <v> is outside (-1, 1); Cornette-Shanks is singular at the endpoints.` |
| sky-view LUT sampled before its stage ran | `AtmosphereSky: sky-view LUT is empty. Splice atmosphere.getStages() BEFORE env.getBakeStages() and the sky pass.` |
| `AtmosphereSky` with a `SkySource` slot expecting a source cube | `AtmosphereSky is analytic and needs no source cube; keepSourceCube is only required by EquirectSky.` |
| LUT dimensions not multiples of the workgroup size | `Atmosphere: <lut> dimensions <w>x<h> must be multiples of 8.` |

---

## 10. Engine touchpoints

### 10.1 Core engine — **none**

Compute stages, `rgba16float` storage textures, workgroup reduction, `textureSampleLevel`,
`ComputeStageNode.enabled`. Every one shipped, and the last was added for shadows in v1.2.

### 10.2 Module changes — additive

| Change | File | Nature |
|---|---|---|
| `atmosphereMath.ts` — density profiles, optical depth, phase functions, CPU transmittance | new | pure, worker-safe |
| `atmosphereShaders.ts` — three LUT kernels + the `sampleSky` layout | new | |
| `Atmosphere.ts` — LUT ownership, `getStages()` | new | |
| `AtmosphereSky.ts` — the `SkySource` | new | |
| `skySunRadiance` body | `skyMath.ts` | **replaced**, signature unchanged (§6) |
| Twilight ramp | `nightShaders.ts` | **deleted** — the model produces it (§4.3) |

`EnvironmentSystem`, `LightSystem`, `ShadowSystem`, `TimeOfDay`, every material: **unchanged**.

### 10.3 Optional — filed, not needed

- **A 3D aerial-perspective LUT.** Explicitly Phase 5, because its cost is not the LUT — it is the
  **integration surface**. Applying it per-fragment adds a 3D texture and a uniform to *every
  material's* group 2, and Phase 1 §4.3 shows that group already carries 7 sampled textures of a
  core 16 before group 1's material maps. Applying it as a fullscreen pass instead needs a samplable
  depth texture, which the core does not yet provide (`architecture-tree.md` §1). That trade is a
  document, not a section.
- **`float32-filterable` LUTs.** `rgba16float` banding has not been observed at these dimensions
  (T4 pins it). Adopting a feature to fix an unmeasured problem is what constraint #4 forbids.
- **Spectral (n > 3 wavelength) integration.** Filed. Visible mainly in twilight gradients, and it
  multiplies every LUT's cost.

---

## 11. Build order

1. **`atmosphereMath` — densities, optical depth, phase functions.** No GPU. U1, U2, U7, U8, U10
   pass. The physics is verified before a texture exists.
2. **Transmittance LUT + kernel.** U5 (CPU↔GPU) and T1. Nothing renders yet; the table is checkable
   against §5.1 by readback.
3. **`skySunRadiance` swapped to the LUT.** The **sun's colour becomes physical before the sky
   does** — Phase 2's day cycle immediately reddens correctly at sunset, with the old sky. U9 proves
   no caller changed. A visible win at step 3 of 7.
4. **Multiple-scattering LUT.** U4. Still not rendering — but a single-scattering-only sky, built
   next, would be diagnosed as "too dark" and the cause would be this table's absence.
5. **Sky-view LUT + `AtmosphereSky`.** The sky is physical. U6, U11, T1–T5.
6. **Composite swap.** `SkyComposite({ day: AtmosphereSky, night: NightSky })`. T6 — one constructor
   argument, no caller edits.
7. **Delete the twilight ramp.** U12 first, observed failing with ozone disabled, then the deletion.
   Deleting authored content is only safe once a test says the model replaced it.

### Deferred

- Aerial perspective → [Phase 5](./environment-aerial-design.md).
- Clouds as a participating medium in the same march.
- Volumetric shadowing of the atmosphere itself (mountain shadows in air).
- Atmospheric refraction near the horizon (~0.57° lift at elevation 0) — the model has the geometry
  to do it properly, and this is the phase where it stops being a hack. Filed; it is smaller than
  the lunar position error Phase 2 already accepts.
- Ground-bounce (the atmosphere lit from below by a bright surface — snow, desert). One more term
  in the multiple-scattering integral, gated on a ground-albedo input that already exists.
