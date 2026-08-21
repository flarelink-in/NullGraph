# NullGraph Environment — Phase 5: Aerial Perspective & Participating Media — Technical Design

**Target:** `src/extras/environment/` (`AerialPerspective`, `aerialShaders`, `mediaMath`)
**Package:** `null-graph/environment`
**Status:** `v2.0` — ✅ **implemented.** API usage: [`api/aerial-perspective.md`](./api/aerial-perspective.md).
**Series:** [1 · Sky sources](./environment-sky-design.md) →
[2 · Time of day](./environment-timeofday-design.md) →
[3 · Night sky](./environment-night-design.md) →
[4 · Physical atmosphere](./environment-atmosphere-design.md) → **5 · Aerial perspective**
**Depends on:** Phase 4's transmittance and multiple-scattering LUTs, and the medium definition.
**Engine change:** one core flag (§9.1) — **the only required engine change in this series** —
landed as opt-in `engine.init(canvas, { samplableDepth: true })`.

### What shipped

| Module | Contents |
|---|---|
| `src/extras/environment/mediaMath.ts` | media rows, combined extinction/scattering, both W distributions and their inverses, the analytic single-medium integral, the CPU march mirror |
| `src/extras/environment/aerialShaders.ts` | froxel build kernel, fullscreen apply, the shared `applyAerial` block, both injected layouts |
| `src/extras/environment/AerialPerspective.ts` | the volume, its stage, the apply pass, and `HeightFog` |
| `src/core/WebGPUContext.ts` | `samplableDepth` (§9.1), stored so it survives `resize()` |
| `src/extras/materials/`, `src/extras/visibility/` | `ShaderAerialLayout` threaded through all six families and both consumers |
| `test/environment-aerial.test.mjs` | U1–U12 plus the §10 error table, `npm run test:aerial` |

Deviations from the spec as written, and why:

- **§8's U11 is headless here.** It is listed under headless in §8 but its stated assertion is a
  CPU↔GPU comparison. The headless half pins the CPU mirror against invariants instead — column
  equivalence, an empty medium, and exact Beer-Lambert on a constant-density medium — which is what
  design-principles §13.1 actually asks of a mirror. The GPU readback half belongs with T1–T6.
- **§5.1's depth-aware upsampling** is implemented as a five-tap bilateral resolve in which each tap
  samples the volume at *its own* view distance. Rejecting taps that disagree about depth is the
  part §5.1 describes; sampling each at its own distance is what makes the accepted taps worth
  averaging.
- **U5 asserts both mis-orderings.** §4.1 says the reversed form makes distant objects glow, which
  is the *un-extinguished* form `color + inScatter` (brighter). Add-then-extinguish
  `(color + inScatter) * T` is a second, dimmer mistake; the test pins its residual at exactly
  `L * (1 - T)` so the diagnosis is one line rather than a debugging session.
- **U10's direction, measured rather than assumed.** §6.2 calls the double-integrated form "dimmer
  and bluer". The transmittances are in fact *identical* — extinction sums in the exponent, so
  `T_fog · T_aerial` is exactly `T_combined` — and the error is entirely in the in-scatter, which
  is over-counted. That is a sharper statement of why the bug survives testing, and the test asserts
  the identity as well as the over-count.

---

## 1. Purpose

Phases 1–4 render the air when you look *through* all of it. This phase renders the air between the
camera and everything else: distant mountains going blue and low-contrast, a valley filling with
haze, sunlight visibly picking out the volume it passes through.

### The single most important pre-existing fact

**This is the first phase in the series that cannot be built without touching `src/core/`.**

```ts
this.depthTexture = this.device.createTexture({
    size: [dWidth, dHeight],
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,      // WebGPUContext.ts:162
});
```

`RENDER_ATTACHMENT` only. The engine's depth buffer **cannot be sampled**, which is why
`HiZPyramid` allocates its own `r32float` pyramid and why `architecture-tree.md` §1 carries
"Depth texture (RENDER_ATTACHMENT only) ◐ not samplable" as a standing partial.

Aerial perspective is a function of **distance to the surface behind each pixel**. Without depth,
there is no distance. §9.1 states the change (one usage flag), its fallback, and that fallback's
real cost.

Every previous phase in this series reported "Core engine — none." This one does not, and the
difference is not incidental: it is why this phase is last, and why its scheduling is coupled to
`architecture-tree.md` §4's depth pre-pass rather than to anything in the sky branch.

### The second fact: the budget already decided the architecture

Phase 4 §10.3 deferred this document specifically because the cost is not the LUT — 256 KB — it is
**where the result gets applied**. §5 works that decision through, and the answer is *both* surfaces
for disjoint cases, sharing one function.

### Non-goals for Phase 5

- **Clouds.** A medium with 3D spatial structure, its own advection, and its own lighting model.
- **Full froxel volumetric fog with light injection.** This document builds the froxel volume and
  the application path; injecting the clustered light list into it is the natural next step and is
  explicitly `v2.0` (§7.2).
- **Temporal reprojection of the froxel volume.** The standard noise fix, and it depends on TAA's
  history machinery (`architecture-tree.md` §12).
- **Replacing screen-space god rays.** `post/GodRays.ts` needs no depth and works today. §7.1 says
  what each one is actually for.
- **Underwater / interior media as authored volumes.** Local media are a placement problem; the
  medium and the integration are the same.

### Design principles

1. **One medium, one integration.** Height fog and atmospheric haze are two density profiles, not
   two systems. Shipping them as two systems double-counts extinction (§6.2).
2. **Apply once per pixel where you can, per fragment where you must.** Opaque geometry gets a
   fullscreen pass; transparents get the injected layout, because they have no single depth.
3. **The required engine change is one flag, stated plainly, with its fallback costed.** This is
   the section design-principles §14 says authors are most tempted to soften.
4. **Every LUT stays a cache of a function the CPU can evaluate** — the discipline from Phase 4 §6.

---

## 2. Architecture

```
   engine.render([
      ...atmosphere.getStages(),      // Phase 4: transmittance, multiscatter, sky-view
      ...aerial.getStages(),          // 32×32×32 froxel volume, per frame
      ...vis.getComputeStages(),
      scenePass,                      // opaque — untouched
      env.getSkyPass({ ... }),
      aerial.getApplyPass({ ... }),   // ← one fullscreen pass, reads depth
      transparentPass,                // materials with the injected layout (§5.3)
      ...post.getPassSequence(),
   ]);
```

```
   Phase 4 LUTs ──┐
   media (§6)  ───┤     ┌──────────────────────────┐
   sun + shadows ─┴────▶│ Aerial-perspective volume │  32×32×32 rgba16float
                        │ (inScatter.rgb, T.a)      │  = 256 KB, per frame
                        └────────────┬──────────────┘
                                     │
                   fn applyAerial(color, uvw) = color * T + inScatter
                                     │
                      ┌──────────────┴──────────────┐
                      ▼                             ▼
        ┌──────────────────────────┐   ┌──────────────────────────────┐
        │ Fullscreen apply pass    │   │ ShaderAerialLayout            │
        │ opaque, 1× per pixel     │   │ injected into TRANSPARENT      │
        │ NEEDS samplable depth    │   │ materials only — per fragment  │
        │ toggleable at runtime    │   │ no depth needed                │
        └──────────────────────────┘   └──────────────────────────────┘
```

One `applyAerial` function, emitted once, injected into both. The composition invariant from Phase 1
§8 U2 applies here too: the two consumers must contain byte-identical bodies.

---

## 3. Integration facts

| Fact | Source | Consequence |
|---|---|---|
| `ctx.depthTexture` is created `RENDER_ATTACHMENT` only. | `WebGPUContext.ts:159-163` | §1. The required change, and its fallback (§9.1). |
| `createPass` accepts arbitrary caller-supplied attachments, including a caller-created depth texture. | `RenderPass.ts`, `EnvironmentSystem.ts:443-449` | **The fallback works today**: an offscreen scene pass with a caller-made samplable depth needs no engine change at all. Only the main-screen convenience path is blocked. |
| `texture_storage_3d<rgba16float, write>` is already used in this repo. | `probeShaders.ts:237-239` | The froxel volume is not a new capability. The precedent is the probe irradiance volume, written the same way. |
| The shadow atlas is a sampled texture with a public view, and receivers already tap it by slot. | `shadows-atlas-design.md` §9.3 | Volumetric shadowing (§7.2) costs one binding and a loop — **no new resource**. |
| `post/GodRays.ts` is a screen-space radial march from a UV light position, luminance-thresholded, with no depth input. | `src/extras/post/GodRays.ts:36-40` | It is a distinct effect, not a worse version of this. §7.1. |
| Consecutive compute stages share one pass; a render pass closes it. | `RenderPipeline.ts:45-48,66-70` | The froxel build slots ahead of the scene pass with no explicit barrier. |
| The PBR material tonemaps and gamma-corrects **inline**. | `lighting-ibl-design.md` §3 | The injected transparent path (§5.3) must apply aerial perspective **before** that, or it fogs an already-tonemapped value. A named hazard, pinned by U7. |
| Group 2 on a full material already carries 7 sampled textures and 3 uniforms, of core limits 16 and 12. | [Phase 1](./environment-sky-design.md) §4.3 | The injected path costs one 3D texture + one uniform. It **fits** — §5.2 explains why fit is not the deciding argument. |

---

## 4. The froxel volume

### 4.1 Shape

A 32×32×32 `rgba16float` 3D texture — **262,144 bytes**. Each texel holds:

| Channel | Meaning |
|---|---|
| `rgb` | in-scattered radiance accumulated from the camera to this froxel |
| `a` | mean transmittance from the camera to this froxel |

Application is one line, and it is the whole contract:

```wgsl
fn applyAerial(color: vec3<f32>, uvw: vec3<f32>) -> vec3<f32> {
    let ap = textureSampleLevel(aerialVolume, aerialSampler, uvw, 0.0);
    return color * ap.a + ap.rgb;
}
```

`color * T + inScatter` — extinction then addition, in that order. Reversed, distant objects glow
instead of fading, which is the single most common bug in a fog implementation and looks
superficially like "too much fog."

### 4.2 Depth distribution

The W axis maps to view distance, and the mapping is a config choice because the two media want
opposite things:

| Distribution | Slices where | For |
|---|---|---|
| `'linear'` (default) | uniform to `maxDistance` (32 km) | atmospheric haze — resolution matters at range |
| `'log'` | packed near the camera | ground fog, interiors — the near field is where it is visible |

Beyond `maxDistance` the volume clamps to its last slice, which is correct: at 32 km the medium's
transmittance is already near its asymptote, so extending the range buys nothing.

### 4.3 Cost

Each froxel continues the previous slice's march rather than restarting from the camera, so the
whole volume costs one ray per (x, y) with 32 accumulation steps:

| | Samples / frame |
|---|---|
| Froxel volume | 32 × 32 × 32 ≈ **32,768** texel updates |
| Per-pixel march at 1080p × 30 steps | 62,208,000 |

**~1,900× fewer.** The same argument as Phase 4 §1, and it is again the reason a technique with a
reputation for expense is affordable: it is evaluated where the signal is smooth, and interpolated
where the pixels are.

The froxel resolution is the price. 32 lateral slices over a 60° fov is ~1.9° per froxel, so a hard
occluder edge does not produce a hard fog edge. This is visible as **light bleeding around
silhouettes** and it is the technique's characteristic artefact — stated here rather than
discovered. The mitigations are depth-aware upsampling on apply (shipped, §5.1) and temporal
reprojection (deferred, §1).

---

## 5. The integration surface — the actual decision

### 5.1 Primary: one fullscreen pass over opaque

Reads the depth buffer, reconstructs view distance, samples the volume, writes `color * T + inScatter`.

- **Once per pixel.** With overdraw, the per-fragment alternative applies aerial perspective for
  every fragment and discards all but one — correct, but wasted.
- **Runtime-toggleable.** No pipeline recompilation. An editor viewport, a debug view, or a quality
  setting flips a boolean.
- **Depth-aware upsampling.** Because it owns the depth buffer, it can reject froxel samples whose
  depth disagrees with the pixel's, which is the mitigation for §4.3's silhouette bleeding. The
  injected path cannot do this — it sees one fragment and has no neighbourhood.
- **It needs samplable depth** (§9.1).

### 5.2 Why the budget is *not* the argument

Phase 4 §10.3 suggested group-2 pressure decides this. Counted precisely, it does not:

| Stage resource | Current on a full material | With the injected layout | Core limit |
|---|---|---|---|
| Sampled textures | 12 (7 in group 2, 4 material maps, 1 atlas) | 13 | 16 |
| Uniform buffers | 5 | 6 | 12 |
| Storage buffers | 5–6 | 5–6 | **8** |

It fits, with room. Stating that plainly matters more than winning the argument: **the fullscreen
pass is chosen for correctness-per-pixel, toggleability, and depth-aware upsampling — not because
the alternative does not fit.** A design that justifies itself with a constraint that turns out to
be false is a design nobody can revisit.

### 5.3 Secondary: the injected layout, for transparents

Transparents genuinely need the fragment path, because a transparent surface has **no single depth
in the depth buffer** — it may not have written depth at all, and several may overlap one pixel.
Each needs aerial perspective at its own distance.

```ts
const batch = vis.createInstanceDrawBatch(pass, 'glass', 0, {
    material: { kind: 'pbr', instance: mat, lighting, indirect, aerial: aerial.getShaderLayout() },
});
```

The fourth injected layout, the same shape as the three that exist. And it emits the **same
`applyAerial` body** as the fullscreen pass — one function, two consumers, exactly as Phase 1 §6
established for the sky. U6 asserts the bodies are byte-identical.

**Order hazard.** The PBR fragment tonemaps inline (`lighting-ibl-design.md` §3). Aerial perspective
is radiometric and must be applied to linear radiance, *before* that tonemap. Injecting it after
fogs an already-compressed value — which desaturates toward grey rather than toward the sky's
colour, and reads as "the fog colour is wrong." U7 pins the injection point.

### 5.4 Two paths is not two systems

They share the medium, the LUT, the volume, and `applyAerial`. What differs is which pixels each
one visits, and the sets are disjoint: the fullscreen pass runs before the transparent pass and the
transparent pass's own fragments are not in the depth buffer it read. Double application is
therefore not possible by construction — and U8 asserts it anyway, because "not possible by
construction" is a claim about code that changes.

---

## 6. Media, unified

### 6.1 The set

A medium is a scattering coefficient, an absorption coefficient, a density profile, and a phase
function. Phase 4 defined three; this phase adds one and integrates all of them together:

| Medium | Profile | Origin |
|---|---|---|
| Rayleigh | exponential, H = 8 km | Phase 4 §4.1 |
| Mie | exponential, H = 1.2 km | Phase 4 §4.1 |
| Ozone | tent at 25 km | Phase 4 §4.1 (absorption only — invisible at aerial-perspective range) |
| **Height fog** | exponential, `σ₀ · exp(−(h − h₀)/H_f)` | **new** |

Height fog is Mie with different numbers. It is not a feature; it is a row.

### 6.2 The unification claim, and the bug it prevents

`architecture-tree.md` §8 lists "Exponential height fog ◻ v1.2" as a separate item from "Froxel
volumetric fog ◻ v2.0". They are the same medium under two integrations, and if both ship
independently, a scene with both applies **extinction twice**:

```
wrong:   color · T_fog · T_aerial + inScatter_fog + inScatter_aerial
right:   color · T_combined       + inScatter_combined
```

The wrong form is dimmer and *bluer* than either alone — and it is only wrong when both are
enabled, so it survives every test that enables one.

**This is not an argument against shipping height fog early.** The closed-form analytic integral for
a single exponential medium is genuinely cheaper than a froxel volume, needs no depth texture, and
should ship at v1.2 as planned. The claim is narrower and enforceable: **when the froxel volume is
active, it owns every medium**, and the analytic path is disabled with a named error rather than
silently stacking (§8). U9 pins that the two produce the same result for a matched single medium —
which is what makes the switch safe.

---

## 7. Neighbours

### 7.1 God rays: screen-space vs volumetric

`post/GodRays.ts` marches radially outward in UV from a screen-space light position, weighting by
luminance above a threshold. It needs no depth, no volume, and no atmosphere. **It works today and
this phase does not replace it.**

| | Screen-space (`post/GodRays.ts`) | Volumetric (§7.2) |
|---|---|---|
| Light off-screen | fails — the origin must be on screen | correct |
| Occluders behind the camera | invisible to it | correct |
| Shafts through a doorway from the side | not representable | correct |
| Cost | one radial blur | froxel volume + shadow taps |
| Depends on | nothing | samplable depth + shadow atlas |
| Physically coupled to the sky | no | yes — same medium as the horizon haze |

They are different effects with overlapping appearance. A stylized scene may prefer the screen-space
one permanently. Documenting the boundary is what stops the volumetric version being filed as a
"proper" replacement and the working effect being deleted.

### 7.2 Volumetric shadowing — the deferred step, named precisely

The froxel volume already marches through the medium. Sampling the shadow atlas at each step turns
in-scattering into shadowed in-scattering — real light shafts. The atlas exists, its view is public,
and receivers already tap it by slot (`shadows-atlas-design.md` §9.3). So the cost is **one binding
and a loop**, not a new subsystem.

It is deferred because it changes the froxel volume from "cheap" to "proportional to steps × lights"
and needs the noise mitigation (temporal reprojection) that TAA has not landed yet. Naming it here
with its actual cost is the point: `architecture-tree.md` §8's "Froxel volumetric fog ◻ v2.0" with
its four sub-items is, after this phase, **one loop and a reprojection away** — not a separate
project.

---

## 8. Test suite

**Headless** (`npm run test:aerial`):

| # | Asserts |
|---|---|
| U1 | Transmittance → 1 at zero distance and is monotonically decreasing with distance, for 8 media configurations |
| U2 | In-scatter → 0 at zero distance |
| U3 | **Slice composition.** Integrating to distance `d`, then from `d` to `D`, equals integrating straight to `D` within 1% — over 64 (d, D) pairs. The correctness property of any slice-accumulated volume; it fails if the march restarts per slice instead of continuing |
| U4 | Both depth distributions are monotonic, cover `[near, maxDistance]`, and are invertible to 1e-6 |
| U5 | `color * T + inScatter` — the composition order, asserted against a reference where the reversed form is measurably brighter at range |
| U6 | The fullscreen pass and the injected layout contain **byte-identical** `applyAerial` bodies |
| U7 | In the composed transparent fragment, `applyAerial` appears **before** the inline tonemap (§5.3) |
| U8 | A batch cannot carry the injected layout **and** be drawn in a pass covered by the fullscreen apply — a build-time refusal, not a runtime check |
| U9 | **The unification claim.** The froxel volume and the analytic height-fog integral agree within 2% for a matched single exponential medium, over 64 (distance, height) pairs (§6.2) |
| U10 | Combining N media once equals applying them one at a time only under the *combined* form — i.e. the §6.2 double-extinction bug is constructed and observed to fail |
| U11 | CPU ↔ GPU volume agreement at 32 froxels within 3% (design-principles §13.1) |
| U12 | No backtick, no `PI` redeclaration, every generated module parses |

U10 is written the way design-principles §13 requires: the bug is built deliberately, seen to fail,
and then the correct form is asserted.

**GPU integration:**

| # | Asserts |
|---|---|
| T1 | A ground plane receding to 32 km shows monotonically increasing fog and converges to the horizon's sky colour within 2% — aerial perspective and the sky agree at the limit, which is the visual test that the two share a medium |
| T2 | Toggling the apply pass changes no other pixel than those with depth < far |
| T3 | Silhouette bleeding is bounded: with depth-aware upsampling on, fog error adjacent to a hard occluder edge is < 10%; with it off, > 30%. Both bounds — the second proves the mitigation is doing something |
| T4 | A transparent surface at distance `d` in front of an opaque surface at `2d` is fogged for `d`, not `2d` — the case the fullscreen path alone gets wrong |
| T5 | Volume rebuild cost, from the profiler, is under 0.3 ms at 32³ on the reference device |
| T6 | The fallback path (offscreen pass, caller-supplied samplable depth) produces an image identical to the main-screen path within 1e-3 (§9.1) |

---

## 9. Engine touchpoints

### 9.1 Required — one usage flag

```ts
// WebGPUContext.ts:162
usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
```

Sampling `depth24plus` needs a `texture_depth_2d` binding with a **non-filtering** sampler, or
`textureLoad`. Both are core WebGPU; no feature flag, no format change.

**Cost of the change.** Adding `TEXTURE_BINDING` can disable some drivers' depth-compression fast
paths for that texture. It is a real cost, it is not measured here, and it applies to every frame
whether or not aerial perspective is enabled — so the flag should be **opt-in at context creation**
(`{ samplableDepth: true }`) rather than unconditional. That keeps constraint #3's spirit: pay for
what you use, at construction, not per frame.

**Zero-change fallback, and its actual cost.** A caller can render the scene into an **offscreen
pass** with their own depth texture created with `TEXTURE_BINDING`, today, with no engine change —
`createPass` takes arbitrary attachments (`EnvironmentSystem.ts:443-449`), and any scene using
`PostChain` already renders offscreen.

So the fallback is mild and specific: **only the main-screen convenience path is blocked.** The cost
is that a scene wanting aerial perspective must render offscreen — which it almost certainly already
does, because post-processing requires it. T6 pins the two paths to the same image.

**Who else wants this.** `HiZPyramid` allocates a separate `r32float` pyramid because the depth
texture is not samplable, and `architecture-tree.md` §4 lists the depth pre-pass as caller-assembled
for the same reason. One flag serves three consumers, which is why this phase should be scheduled
**with** that core work rather than driving it alone.

### 9.2 Module changes — additive

| Change | File | Nature |
|---|---|---|
| `mediaMath.ts` — profiles, combined extinction, analytic single-medium integral | new | pure; U9's reference |
| `aerialShaders.ts`, `AerialPerspective.ts` | new | |
| `ShaderAerialLayout` + `aerial` in the material config | `materialFragments.ts` | additive, guarded by `if (config.aerial)` exactly as `lighting` and `indirect` are |
| Height-fog medium row | `atmosphereMath.ts` | additive |
| Analytic height fog refuses when the volume is active | `AerialPerspective.ts` | §6.2, §10 |

### 9.3 Optional — filed, not needed

- **A depth pre-pass helper** would give the apply pass depth without waiting for the opaque pass.
  Not needed — the apply pass runs after opaque anyway.
- **`r32float` linear-depth target** instead of sampling `depth24plus`. Filed: costs a full-resolution
  target to avoid one reconstruction, and reconstruction from the camera's `invProj` is already how
  the skybox builds its view ray (`envShaders.ts:593`).
- **16×16×16 volume for low-tier devices.** A config, not a design. The artefact scaling in §4.3
  gets worse linearly.

---

## 10. Errors

| Condition | Message |
|---|---|
| No samplable depth | `AerialPerspective: the apply pass needs a samplable depth texture. Create the context with { samplableDepth: true }, or render the scene into an offscreen pass with your own depth attachment (design §9.1).` |
| Analytic height fog enabled while the froxel volume is active | `AerialPerspective: HeightFog is active alongside the froxel volume; both integrate the same medium and extinction would be applied twice (design §6.2). Move the fog parameters onto the volume's media list.` |
| The injected layout on a batch inside the fullscreen apply's pass | `AerialPerspective: batch "<name>" carries the injected aerial layout but is drawn in a pass covered by getApplyPass(); aerial perspective would be applied twice. The injected layout is for transparent passes drawn after the apply pass.` |
| Volume dimensions not multiples of 4 | `AerialPerspective: volume dimensions <x>x<y>x<z> must be multiples of 4.` |
| `maxDistance <= near` | `AerialPerspective: maxDistance <d> must exceed the camera near plane <n>.` |
| `getApplyPass` before `getStages` was spliced | `AerialPerspective: the froxel volume has never been built. Splice aerial.getStages() before the scene pass.` |
| Injected layout without a Phase 4 `Atmosphere` | `AerialPerspective: no medium. Construct with an Atmosphere (Phase 4) or an explicit media list.` |

---

## 11. Build order

1. **`mediaMath` + the analytic single-medium integral.** No GPU, no depth. U1, U2, U4. It is also
   the v1.2 exponential height fog, shipped standalone and fragment-side — **useful before anything
   else in this document exists.**
2. **The froxel volume, built but unread.** Readback-verified against `mediaMath`. U3, U11. Nothing
   renders differently.
3. **The injected layout, on opaque materials.** Deliberately the "wrong" surface first — it needs
   **no depth texture**, so the whole medium and application chain is verified before §9.1's core
   change lands. U5, U6, U7, T1.
4. **`samplableDepth` + the fullscreen apply pass.** T2, T3, T6. Opaque moves to the fullscreen path.
5. **Restrict the injected layout to transparents.** U8, T4, and the refusals in §10.
6. **The unification.** Height fog becomes a medium row; the analytic path refuses when the volume
   is active. U9, U10.

Step 3 is the deliberate order inversion, and it is the point: it puts every piece that does *not*
need the engine change ahead of the one that does, so the core work is unblocked-by-verification
rather than blocking it.

### Deferred

- Light injection into the froxel volume (clustered light list → in-scatter) — `v2.0`.
- Volumetric shadowing from the shadow atlas (§7.2) — one binding and a loop.
- Temporal reprojection of the volume — needs TAA's history targets, which `TransientPool` already
  supports (`history: 2`).
- Local media volumes (a room, a cave, underwater) as placed boxes contributing to the same march.
- Cloud shadows modulating in-scatter — a 2D lookup in the same loop; it wants the cloud branch to
  exist first.
