# NullGraph Environment — Phase 1: Sky Sources & the Evaluator Seam — Technical Design

**Target:** `src/extras/environment/` (`skyMath`, `skyShaders`, `SkySource`, `EnvironmentSystem`)
**Package:** `null-graph/environment`
**Status:** `v1.1.0` — ✅ **implemented.** `npm run test:sky` (14 headless) ·
`npm run test:sky:gpu` (13 on-device). Two deviations from the text below, both deliberate and
both marked inline: `cheapSky` **ships defaulting to 0** (§4.5), and the harness lives in
`test/environment-gpu/` with a headless Chrome driver.
**Series:** **1 · Sky sources** → [2 · Time of day](./environment-timeofday-design.md) →
[3 · Night sky](./environment-night-design.md) →
[4 · Physical atmosphere](./environment-atmosphere-design.md) →
[5 · Aerial perspective](./environment-aerial-design.md)
**Depends on:** [`lighting-ibl-design.md`](./lighting-ibl-design.md) (the bake this feeds) —
✅ implemented. Nothing else.
**Roadmap:** [`architecture-tree.md`](./architecture-tree.md) §8 ·
[`roadmap-modules.md`](./roadmap-modules.md) §6

---

## 1. Purpose

The engine has two skies and neither of them is the sky.

**One** lives inside a compute kernel — `skyColor(dir)`, a three-colour gradient plus a sun disc,
written directly into the source-cube bake (`envShaders.ts:141-147`). **The other** is what you
actually see: the skybox fragment does not call `skyColor` at all, it samples the **prefiltered
specular cube** (`envShaders.ts:603`).

That second one is the finding this whole series rests on.

### The single most important pre-existing fact

**The visible sky is a 128²-per-face GGX-convolved cube.**

```ts
const specSize = 1 << (this.cfg.prefilterMips - 1 + 2);   // EnvironmentSystem.ts:136
```

At the default `prefilterMips: 6` that is `1 << 7` = **128**. Four cube faces span the horizon, so
the sky you see carries **512 texels over 360° — 0.70° per texel**.

Everything follows from that number:

| Thing | Angular size | In the visible sky |
|---|---|---|
| The real sun | **0.53°** | sub-texel — cannot be drawn |
| The shipped default sun (`sunCos: 0.9995`) | **3.62°** | ~5 texels |
| The moon | 0.52° | sub-texel |
| A star | < 0.01″ | not representable at any cube resolution |
| A cloud edge, a crepuscular ray, a horizon line | ≲ 0.1° | smeared |

The shipped `sunCos` default is not a stylistic choice — it is **6.8× the sun's true angular
diameter**, and it has to be, because a physically sized disc would vanish into a single texel of
the thing that displays it. The sky is currently wide enough to see only because the sun was made
wrong on purpose.

Mip 0 does not rescue this. `buildPrefilterWGSL` short-circuits roughness 0 to a straight copy
(`envShaders.ts:230-247`), so `blurMip: 0` is the source cube *resampled to 128²* — not the source.

So: **no phase after this one is possible without changing where the skybox gets its colour.** A
time-of-day system (Phase 2) whose sunset is five texels wide is not a time-of-day system. Stars
(Phase 3) are not a resolution problem, they are a representation problem. And Phase 4's physical
atmosphere would compute a horizon gradient that is destroyed on its way to the screen.

### The second fact: two skies drift

`skyColor` in the bake kernel and the prefiltered-cube lookup in the skybox are two pieces of WGSL
that both claim to be the sky, and **they already disagree** — one has a hard-edged disc at
`sunCos`, the other has a soft blob whose size is set by mip resolution. Nothing detects this. The
lighting is baked from one and the background is drawn from the other.

Every phase in this series adds a sky model. Adding a model to the current structure means editing
two kernels and hoping. Phase 1 makes that class of bug **unrepresentable**: one injected function,
two consumers.

### Non-goals for Phase 1

- **Any new sky content beyond one analytic daylight model.** Stars, moon, twilight (Phase 3);
  Rayleigh/Mie/ozone (Phase 4). Phase 1 ships the seam and moves what exists behind it.
- **Driving the sun.** Nothing here decides where the sun is; it takes a direction. Phase 2 owns
  the clock.
- **Fog, aerial perspective, anything between the camera and the geometry.** Phase 5.
- **Clouds.** A different branch of §8 entirely, and it wants the volumetric machinery.
- **Raising the prefiltered cube's resolution.** A mirror sphere still reflects a 128² sky; that is
  a `prefilterMips` config question (`prefilterMips: 8` → 512²), not a design question.

### Design principles

1. **One evaluator, every consumer.** `sampleSky(dir)` is emitted once and injected into both the
   bake kernel and the skybox fragment. The sky you are lit by and the sky you see are the same
   function *by construction*, not by review.
2. **A source is data, not a code path.** Adding Hosek–Wilkie, or a client's proprietary sky, is a
   new `SkySource` — no edit to `EnvironmentSystem`, no `if (procedural)` branch, no new kernel.
3. **Cost lands where it is affordable.** The evaluator runs per **cube texel** in the bake (once,
   or amortized) and per **pixel** in the skybox. It never enters a material fragment, so it costs
   **zero group-2 bindings** on the one bind group that is actually scarce (§4.3).
4. **Fail loud.** A sky source that needs the source cube, on a system configured to release it, is
   a named build-time error — not a black background (§11).

---

## 2. Architecture

Unchanged frame sequence. What changes is who supplies the WGSL.

```
   // once, at load:
   env.setSky(new GradientSky({ ... }));         // or PreethamSky, or EquirectSky, or yours
   engine.render([ ...env.getBakeStages(), ...rest ]);

   // every frame — identical to today:
   engine.render([
      ...vis.getComputeStages(),
      ...shadows.getComputeStages(),
      shadows.getAtlasPass(),
      scenePass,
      env.getSkyPass({ ... }),        // now evaluates sampleSky per pixel
      ...post.getPassSequence(),
   ]);
```

```
                          ┌───────────────────────────┐
                          │  SkySource                │
                          │  getShaderLayout()        │
                          │   → { wgslStructs,        │
                          │       wgslLogic }         │
                          │  writeParams(Float32Array)│
                          └─────────────┬─────────────┘
                                        │  fn sampleSky(dir) -> vec3<f32>
                        ┌───────────────┴───────────────┐
                        ▼                               ▼
          ┌──────────────────────────┐    ┌──────────────────────────┐
          │ Stage A — source cube    │    │ Skybox fragment          │
          │ 512² × 6, compute        │    │ per-pixel, view ray      │
          │ group 0 (free)           │    │ group 2 on ITS OWN batch │
          └────────────┬─────────────┘    └──────────────────────────┘
                       │                              ▲
                       ▼                              │ (fallback only:
        mip chain → GGX prefilter → specular cube ────┘  cheapSky mode)
                       │
                       └──▶ SH-L2 projection ──▶ EnvParams (materials)
```

The right-hand branch is the change: the skybox stops being a *consumer of the bake* and becomes a
*sibling consumer of the source*. The bake keeps feeding materials exactly as it does today.

---

## 3. Integration facts (verified against this repository)

**These were verified against the pre-1.1 tree — the state this design was written to change.** Four
of the nine describe code the phase then deleted or moved, so the *Now* column says where each one
went. Bare line numbers drift (design-principles §1 corollary), so live claims are pinned to a
distinctive symbol and the numbers below were re-swept after the implementation landed.

| Fact (pre-1.1) | Source | Consequence | Now |
|---|---|---|---|
| The skybox samples `envSpecular` — the **prefiltered** cube — at `blurMip`. | `buildSkyWGSL`'s `textureSampleLevel(envSpecular, ...)` | §1's whole argument. The visible sky is capped at `specSize`. | **Changed.** That lookup is now the `cheapSky` branch only (`envShaders.ts:562`); the default evaluates `sampleSky` per pixel. The pre-1.1 module survives verbatim as `buildSkyWGSL` *without* a layout (`:576`) — T4's reference. |
| `specSize = 1 << (prefilterMips - 1 + 2)` = **128** at the default. | `EnvironmentSystem.allocate`, `const specSize` (`:147`) | 0.70°/texel around the horizon. | Unchanged — and no longer the ceiling on the visible sky. |
| The gradient sky is hardcoded inside `buildSourceCubeWGSL`, selected by a `procedural: boolean`. | `envShaders.ts:128-152` (pre-1.1) | A third source means a third branch of a boolean. The seam replaces the boolean. | **Deleted.** `buildSourceCubeWGSL(layout)` takes the injected evaluator; the boolean is gone and `GradientSky` carries the body (U1 pins it inert to 1e-6). |
| The sky batch has **its own** bind group: only bindings 3 and 4 of group 2, no lights, no probes. | `getSkyPass`, the group-2 entry list (`:559-562`) | The evaluator's uniforms are free. Materials are untouched (§4.3). | Held. Sources may add 14–17 through `getSkyboxBindEntries`; U11 pins the range. |
| `createBatch` binds `batch.storageBuffer` at group 0 binding 1 unconditionally; the skybox already reads `[exposure, yaw, blurMip, _]` out of it. | `BatchManager.createBatch`'s group-0 bind group (`:94-102`) | Sky params widen by changing `strideFloats`. No engine change, no new binding. | Held, and used exactly as predicted: the stride is now `4 + source.paramFloats`. Float 3 became `cheapSky`. |
| A batch with no vertex/index buffer draws as `draw(3, instanceCount)`. | `RenderPipeline.ts:147` | The skybox is already a fullscreen triangle. Phase 3's star field reuses the same path. | Unchanged. |
| `keepSourceCube` defaults to `updateMode === 'amortized'` and is otherwise false. | `EnvironmentSystem` constructor `cfg` (`:111`) | A cube-backed sky source (equirect) needs the source retained. That is a new required-config relationship → a named error (§9). | Held; the named error ships and U9 pins it. |
| `updateMode: 'amortized'` rebuilds Stage A + **one** prefilter mip + the SH projection per frame. | `getBakeStages`, the `amortizeCursor` slice (`:437-445`) | Stage A re-runs every frame in amortized mode, so the evaluator's per-texel cost is a real per-frame cost: `sourceSize² × 6` invocations. §6.3 sizes it. | Unchanged. Phase 2 still owns the re-bake gate. |
| Group 2 bindings **0–13** are already spoken for on a fully-featured material. | `LightSystem.ts:78-88`, `buildIndirectStructsWGSL`, `probeShaders.ts:51-57` | The binding budget is the reason the evaluator must never be a material dependency. §4.3. | Held — the phase spent **zero** material bindings, as designed. |

---

## 4. Data model

### 4.1 `ShaderSkyLayout` — the contract

The third member of a family that already has two. `ShaderLightingLayout` (direct light),
`ShaderIndirectLayout` (indirect light), and now:

```ts
export interface ShaderSkyLayout {
    /** Bindings + structs. Emitted with a caller-supplied group and binding base. */
    wgslStructs: string;
    /** `fn sampleSky(dir: vec3<f32>) -> vec3<f32>` and its helpers. */
    wgslLogic: string;
}
```

Same shape, same injection mechanism, same composition story (design-principles §4). A reader who
has seen one has seen all three.

### 4.2 The two functions every source provides

```wgsl
/// Radiance arriving from world-space direction `dir` (normalized), in linear HDR,
/// in the engine's scene-relative unit (§4.4). Must be finite and non-negative for
/// every unit vector — including straight down.
fn sampleSky(dir: vec3<f32>) -> vec3<f32>;

/// Radiance of the sun disc alone at solar elevation `elevDeg`, WITHOUT the disc's
/// angular extent — i.e. what a directional light of that elevation should be coloured.
/// Phase 2's CPU driver mirrors THIS function and nothing else.
fn skySunRadiance(elevDeg: f32) -> vec3<f32>;
```

`skySunRadiance` looks redundant — the sun is in `sampleSky` already. It is not. It is the **named
seam for the CPU mirror**, drawn before there is a CPU consumer (design-principles §4). Phase 2
needs the sun's colour to fill a light record; Phase 4 replaces the implementation with a
transmittance lookup. Both work against this signature. Without it, Phase 2 would grow a private
extinction curve that drifts from the one on screen, and the failure — "the light is warm but the
disc is white" — is the kind nobody attributes to the right file.

**Constraints, checked at build time:**

- The body must not declare `PI` (the material/bake preambles own it) — U7.
- Names are prefixed `sky_` except the two entry points, so a composite (Phase 3) can rename
  without collision.
- No backtick anywhere in the generated string. This is a standing hazard in this repo — a backtick
  inside a WGSL comment terminates the JS template literal that carries it — and it is pinned by an
  assertion, not by care (U8).

### 4.3 Binding budget — the part that matters

Group 2 on a fully-featured material, today:

| Binding | Resource | Owner |
|---|---|---|
| 0–2 | `lights`, `clusterGrid`, `lightIndices` | `LightSystem` |
| 3–6 | `envSampler`, `envSpecular`, `brdfLUT`, `EnvParams` | IBL Phase 1 |
| 7–11 | `probeR/G/B`, `probeValidity`, `ProbeParams` | Probes Phase 2 |
| 12–13 | `reflProbes`, `reflProbeData` | Probes Phase 2 |

Per-stage totals on that fragment: **7 sampled textures** (of a core 16, before group 1's four
material maps and group 3's atlas), **3 uniform buffers** (of 12), **4 storage buffers** (of a core
**8**, and group 0 spends more). The storage-buffer count is the one with real headroom left, and
it is not much.

**Phase 1 adds zero of these.** The evaluator is injected into:

- the **bake kernel**, a compute stage whose group 0 holds two or three entries — effectively empty;
- the **skybox batch**, whose group 2 holds two entries.

Neither is the material. This is not an accident of implementation, it is the reason the design is
affordable, and it is the constraint that decides Phase 5's integration surface before Phase 5 is
written (see [`environment-aerial-design.md`](./environment-aerial-design.md) §5).

**Reserved on the sky batch only:** bindings 14–17 for source params. They collide with nothing
because that bind group has no lights and no probes in it.

### 4.4 Units

One scene-relative unit, stated once for the series: **`1.0` is the radiance of a Lambertian white
surface (albedo 1.0) under direct midday sun at normal incidence.** Every sky model, every light
colour, and every exposure default in Phases 1–5 is expressed against that reference.

It is not photometric, and that is deliberate — §3 of
[`environment-night-design.md`](./environment-night-design.md) shows that the day/night range does
not fit in `rgba16float` in absolute units. Fixing the reference *here*, in the first document,
means Phase 3 is choosing a gain rather than retro-fitting a convention.

### 4.5 `SkyParams` — the batch storage buffer, widened

Today: 4 floats. After Phase 1:

| Float | Field | Notes |
|---|---|---|
| 0 | `exposure` | unchanged |
| 1 | `yaw` | unchanged — environment rotation |
| 2 | `blurMip` | read only when `cheapSky` is 1; under evaluation there is no mip to pick |
| 3 | `cheapSky` | 0 = evaluate per pixel, 1 = the old cube lookup. **Ships defaulting to 0** |
| 4..N | source params | written by `SkySource.writeParams` |

Delivered through `updateBatchData` exactly as now (`EnvironmentSystem.ts:474`). `strideFloats`
comes from `source.paramFloats`, so the buffer is sized by **demand, not by a cap**
(design-principles §10).

---

## 5. The sources shipped in Phase 1

Three, covering the three ways a sky exists: authored, photographed, computed.

### 5.1 `GradientSky` — the zero-asset default

The existing `skyColor` body (`envShaders.ts:141-147`), moved behind the seam, **numerically
unchanged**. Pinned by U1: 64 sampled directions must match the pre-refactor CPU evaluation to
1e-6. A "pure refactor" that shifts the horizon by half a degree is a refactor nobody can review;
the test is what makes the claim checkable.

Cost: ~20 flops. It stays the default and stays the recommended source for stylized work.

### 5.2 `EquirectSky` — the imported environment

`sampleSky` samples the **source cube at mip 0** — `sourceSize`², default 512² = 0.176°/texel.
That is a **4× angular improvement** over the current skybox for free, from nothing but changing
which texture the fragment reads.

Requires `keepSourceCube: true`. The relationship is new and load-bearing, so it is a named error
(§11) rather than a silently black sky.

### 5.3 `PreethamSky` — analytic daylight

Preetham et al. 1999: five distribution coefficients (A–E) per channel, driven by turbidity `T`,
plus an absolute zenith term. ~30 embedded constants, ~40 flops per evaluation, closed form. Given
a sun direction and a turbidity it produces a sky whose luminance gradient, horizon brightening,
and circumsolar glow are *derived* rather than authored — which is exactly what a time-of-day
system needs to drive with one number instead of nine.

**Its failure is documented, not hidden.** Preetham's fit degrades below roughly 5° solar elevation
and its zenith term can go negative near the horizon at high turbidity — i.e. it is wrong precisely
at sunset. Phase 1 clamps to non-negative (U5) and states the boundary. Phase 3's twilight blend
covers the handoff; Phase 4 replaces the model outright.

**Why ship a model Phase 4 supersedes.** Three reasons, because this is the decision in this
document most likely to read as bloat:

1. It is **30 constants and 40 flops** against Phase 4's four LUTs, ~300 KB, and a per-frame
   compute stage. It remains the right choice for a low-tier device and for a scene whose sky is a
   backdrop.
2. It gives Phase 2 something real to drive **one phase earlier**. Driving a three-colour gradient
   with a solar position means authoring a colour ramp by hand — and that ramp is a second sky
   model that Phase 4 would then delete. Preetham is the model that makes the ramp unnecessary.
3. It is the **proof the seam works**. A contract with one implementation is an interface nobody
   has tested. Landing a genuinely different second source is what shows Phase 4 will drop in.

**Hosek–Wilkie** is strictly better at sunset and is *not* shipped: its dataset is ~2,700 lines of
fitted coefficients for a model that Phase 4 supersedes on quality. Because of the seam it is a
`SkySource` an application can add with no engine change — which is the point.

---

## 6. Mechanism

### 6.1 Injection into the bake kernel

`buildSourceCubeWGSL` loses its `procedural: boolean` and takes the layout:

```ts
export function buildSourceCubeWGSL(sky: ShaderSkyLayout): string {
    return `
${ENV_MATH_WGSL}
${sky.wgslStructs}
${sky.wgslLogic}
@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = textureDimensions(dstCube);
    if (gid.x >= dim.x || gid.y >= dim.y) { return; }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dim);
    textureStore(dstCube, gid.xy, gid.z, vec4<f32>(sampleSky(faceDirection(gid.z, uv)), 1.0));
}`;
}
```

The equirect path is no longer a separate kernel — `EquirectSky.wgslLogic` is the texture sample.
One kernel shape, N sources.

### 6.2 Injection into the skybox fragment

`buildSkyWGSL` takes the same layout and replaces its cube lookup:

```wgsl
let dir = normalize(far.xyz / far.w - camera.eyePos);
let R   = rotateYSky(dir, skyParams[1]);
var color: vec3<f32>;
if (skyParams[3] > 0.5) {
    color = textureSampleLevel(envSpecular, envSampler, R, skyParams[2]).rgb;   // cheapSky
} else {
    color = sampleSky(R);
}
return vec4<f32>(color * skyParams[0], 1.0);
```

The `cheapSky` branch keeps the current behaviour reachable — it is genuinely the right answer for
a heavily blurred backdrop, and it is what a source with no analytic form falls back to.

### 6.3 Cost, stated honestly

The evaluator is now in two hot places. Sized at the defaults:

| Site | Invocations | When |
|---|---|---|
| Bake Stage A | `512² × 6` = **1,572,864** | once (`'once'`), or **every frame** (`'amortized'`) |
| Skybox | ≤ 1 per pixel, only where nothing was drawn | every frame |

The amortized number is the one that surprises. Stage A already re-runs every frame in that mode
(`EnvironmentSystem.ts:361`); Phase 1 does not add the dispatch, it makes the dispatch's body a
real sky model. At ~40 flops (Preetham) that is ~63 Mflop/frame — measurable, not alarming, and
Phase 2 removes almost all of it by gating the re-bake on **how far the sun actually moved**
([Phase 2](./environment-timeofday-design.md) §6) rather than on the frame counter.

The skybox's cost is bounded by depth rejection: the sky draws with `depthCompare: 'less-equal'`
and `depthWriteEnabled: false` after opaque (`EnvironmentSystem.ts:455-456`), so an interior scene
evaluates it approximately nowhere.

### 6.4 What the skybox no longer needs

Once the skybox evaluates directly, its dependency on `envSpecular` exists **only** for `cheapSky`.
That matters for Phase 3: a star field must not be prefiltered into a specular cube, and after this
change it does not have to be.

---

## 7. Public API

```ts
import { EnvironmentSystem, GradientSky, PreethamSky, EquirectSky } from 'null-graph/environment';

const env = new EnvironmentSystem(engine, { keepSourceCube: true });

// A source is an object, not a method call.
env.setSky(new PreethamSky({
    turbidity: 2.8,             // 1.8 = pristine, 10 = urban haze
    groundAlbedo: [0.15, 0.14, 0.12],
    sunDir: [-0.4, -1.0, -0.35],// TRAVEL direction, as everywhere else in the engine
    sunAngularRadius: 0.266,    // degrees — the TRUE sun. Now representable.
}));

engine.render([...env.getBakeStages()]);
env.copySHToParams();

// Per frame, if anything moved:
env.sky.setSunDirection(dir);   // typed per source; writes params, no rebuild
env.setSkyParams({ exposure: 1.0 });
```

**Writing a source** is the whole extension story:

```ts
class MySky implements SkySource {
    readonly paramFloats = 8;
    getShaderLayout(base: number): ShaderSkyLayout { /* WGSL using skyParams[base + i] */ }
    writeParams(out: Float32Array, base: number): void { /* no allocation */ }
}
```

`writeParams` takes an `out` array and a base offset — the `out`-parameter discipline
(design-principles §5), so a per-frame sun update allocates nothing.

**Compatibility.** `setProceduralSky(cfg)` and `setEquirect(url)` remain, forwarding to
`new GradientSky(cfg)` and `new EquirectSky(...)`. Existing callers do not change. They are the
convenience layer, not the mechanism.

---

## 8. Test suite

**Headless** (`npm run test:sky`), each parse-validated with `wgsl_reflect`:

| # | Asserts |
|---|---|
| U1 | `GradientSky` matches the pre-refactor `skyColor` at 64 directions to 1e-6 — the refactor is numerically inert |
| U2 | The bake kernel and the skybox fragment contain **byte-identical** `sampleSky` bodies. The seam's entire value, as one string comparison |
| U3 | Every shipped source's composed bake kernel and composed skybox module parse |
| U4 | Preetham zenith luminance is monotonically increasing in turbidity at fixed solar elevation, and in `sin(elevation)` at fixed turbidity |
| U5 | Every shipped source returns finite, non-negative radiance for 512 directions × 32 solar elevations from −20° to +90° — including straight down |
| U6 | `skySunRadiance` CPU mirror matches its WGSL at 32 elevations to 1e-5 (design-principles §13.1) |
| U7 | No generated module declares `PI`; no identifier collides with `ENV_MATH_WGSL` |
| U8 | No generated WGSL contains a backtick |
| U9 | `EquirectSky` on a system with `keepSourceCube: false` throws, naming the flag |
| U10 | `writeParams` allocates zero bytes across 1,000 calls (patched typed-array constructors) |
| U11 | The sky batch's group-2 bindings do not intersect the material group-2 map of §4.3 |
| U12 | *(added in implementation)* The shipped `cheapSky` default is 0 — the phase's deliverable, as one number. Without it the seam can land switched off and every §1 claim is inert |
| U13 | *(added in implementation)* Every sky header float survives a later partial `setSkyParams`. Found by T4: `yaw` alone reverted to the system's rotation, silently turning the skybox back while the SH and specular lookup stayed put |

**GPU integration** (`test/environment-gpu/`, browser — or `npm run test:sky:gpu`, which drives
headless Chrome over the DevTools protocol and exits non-zero on failure per design-principles
§13.2):

| # | Asserts |
|---|---|
| T1 | Sun disc angular diameter, measured from a skybox readback, matches `sunAngularRadius × 2` within one pixel — at `0.266°`, which the current skybox cannot represent |
| T2 | Skybox and source cube agree within prefilter error at 64 directions — the two consumers really are one function |
| T3 | No seam artefact where cube faces meet: radiance is continuous across all 12 edges within 1% |
| T4 | `cheapSky: 1` reproduces the pre-Phase-1 image within 1e-3 |
| T5 | Switching sources at runtime rebuilds the bake and leaves no stale mip |

U2 and T2 are the same claim at two levels — one that a rebuild cannot skip, one that a driver
cannot fake.

**What the implementation had to add, and why.** Three of these needed a second half the spec did
not anticipate:

- **T2** is written as "skybox and source cube agree with each other." Compared directly, the cube
  contributes its own 512² quantization to every residual, so the tolerance would have had to
  absorb it — and a loose tolerance is where a real disagreement hides. Both are held against the
  **CPU mirror** instead (itself pinned to the pre-refactor kernel by U1): the skybox at 0.5%, the
  bake at 2%. Agreeing with one reference means agreeing with each other, without either one's
  error setting the other's bound.
- **T3** runs twice. A **constant** sky isolates the artefact — any variation across a face
  boundary at all is one, so 1% is a real bound — but it cannot notice a face that is rotated or
  flipped. A **gradient** sky catches that, and carries its own quadrature error near the horizon
  (`pow(|y|, 0.6)` has an infinite derivative at `y = 0`), so it is measured away from it at 2%.
- **T4** compares against `buildSkyWGSL` **without** a sky layout — the pre-Phase-1 module is still
  emitted, so the reference is the real old shader rather than a remembered image. It came out
  bit-identical. Its companion T4b asserts `cheapSky: 0` differs, because a T4 that passes with a
  dead branch proves nothing.

Measured on an AMD GCN-5: T1 reports **0.2647°** against the 0.266° asked, 67.8 px across at
0.0078°/px — the measurement §1 says the prefiltered cube's 0.70°/texel cannot represent.

---

## 9. Errors

| Condition | Message |
|---|---|
| `EquirectSky` with `keepSourceCube: false` | `EnvironmentSystem: EquirectSky samples the source cube, but keepSourceCube is false — the cube is released after the bake. Construct with { keepSourceCube: true }.` |
| `setSky` after `getShaderLayout` on a live batch | `EnvironmentSystem: sky source changed after materials were built; the skybox pipeline is compiled against the old source. Call setSky before the first getSkyPass().` |
| `paramFloats` changed since `setSky` | `SkySource "<name>" declares paramFloats=<n>, but the sky batch was created with strideFloats=<m>. The stride is fixed at batch creation — paramFloats must not change after setSky().` Checked in `getSkyPass`, where the stride is frozen: `setSky` already refuses a swap on a live batch, so a *mutated* source is the reachable form of this mistake, and its failure mode is `writeParams` running off the end of a silently bounds-clipped array. |
| A source whose `wgslLogic` lacks `fn sampleSky(` | `SkySource "<name>" does not define sampleSky(dir: vec3<f32>) -> vec3<f32>.` |
| Generated WGSL contains a backtick | `SkySource "<name>" emitted a backtick, which terminates the template literal carrying it. Use single quotes in WGSL comments.` |

All build-time. None run per frame.

---

## 10. Engine touchpoints

### 10.1 Core engine — **none**

Every mechanism exists: `createComputeStage` for the bake, `updateBatchData` for widened params,
`draw(3, n)` for the fullscreen triangle, `depthCompare`/`depthWriteEnabled` per batch.

### 10.2 Module changes — additive, one breaking signature

| Change | File | Nature |
|---|---|---|
| `ShaderSkyLayout`, `SkySource` | `types.ts` | new |
| `GradientSky`, `EquirectSky`, `PreethamSky` | `SkySource.ts` | new |
| `skyMath.ts` (CPU mirrors: gradient, Preetham, `sunRadiance`) | new | pinned by U1/U6 |
| `buildSourceCubeWGSL(opts)` → `buildSourceCubeWGSL(layout)` | `envShaders.ts` | **breaking**, internal — no exported caller |
| `buildSkyWGSL` gains the layout + `cheapSky` | `envShaders.ts` | additive |
| `EnvironmentSystem.setSky()` | `EnvironmentSystem.ts` | additive; `setProceduralSky`/`setEquirect` forward |

The one breaking change is to a function exported for testing, not for use. Public API is
preserved.

### 10.3 Optional — filed, not needed

- **A samplable depth texture** would let the skybox skip evaluation on covered pixels explicitly
  rather than relying on depth rejection. Not needed — depth rejection already does it. Recorded
  because Phase 5 requires this for a different reason.
- **`prefilterMips: 8`** (512² specular) sharpens mirror reflections. A config change, not a
  design change, and it costs 16× the prefilter time.

---

## 11. Build order

Each step renders correctly on its own.

1. **`ShaderSkyLayout` + `GradientSky`, bake only.** The kernel takes the layout; the skybox still
   samples the cube. U1 passes. The image is byte-identical to today. **Nothing has improved yet,
   and that is the point** — the refactor is provably inert before it is asked to do anything.
2. **Inject into the skybox.** `cheapSky` holds at 1 *during this step*, so the image is still
   identical; flipping it to 0 evaluates per pixel. T4 pins the old path, T1 measures the new one.
   This is the step where a true 0.53° sun becomes possible.

   **Shipped as 0**, not 1. Leaving the default at the cube lookup would have landed the seam with
   its own thesis switched off: §1's whole argument is that the visible sky must stop being a
   128²-per-face convolution, and §5.2's "4× sharper imported skies, for free" is not free if it
   needs a flag. The cost is stated where a user meets it (`api/environment.md` §1.3c): a scene
   that used `blurMip` for a soft backdrop now needs `cheapSky: 1` alongside it, because `blurMip`
   has no meaning when there is no mip chain in the path. U12 pins the default; T4 pins that the
   old image is still exactly reachable.
3. **`EquirectSky`.** The `procedural` boolean disappears. 4× sharper imported skies.
4. **`PreethamSky` + `skyMath` mirrors.** U4, U5, U6.
5. **`skySunRadiance` on every source.** Unused until Phase 2 — landed here because a seam added
   under pressure is a seam drawn in the wrong place.

### Deferred

- Hosek–Wilkie (data, not design — an application can add it).
- Raising the prefiltered cube for sharp mirror reflections (config).
- Sky in the reflection-probe capture path: probe captures currently see the prefiltered cube
  (`ProbeVolume.ts:283`). Aligning them costs a bind and belongs with Phase 4, when there is a
  sky worth the extra fidelity.
