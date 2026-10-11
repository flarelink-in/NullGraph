# Probes API Usage — Irradiance Volumes & Reflection Probes

Phase 2 of `null-graph/environment`. Phase 1 gives every surface in the scene the *same* indirect
light; that is right on an open plain and wrong everywhere else — inside a building, under an arch,
beside a red wall, in a cave. The missing quantity is **where**.

Phase 2 makes indirect light a function of position:

- an **irradiance volume** — a grid of SH-L1 probes, hardware-trilinearly blended — for diffuse;
- **local reflection probes** with parallax-corrected box proxies for specular;
- **GPU probe capture** that renders the scene from each probe, amortized across frames;
- and, because each capture reads the *previous* volume, **multi-bounce GI** that converges for free.

The load-bearing idea: **Phase 2 re-emits Phase 1's two functions and changes nothing else.**
`getIndirectDiffuse(worldPos, N)` took `worldPos` in Phase 1 and ignored it. This is the release
that uses it. No material file, no shader builder, and no draw batch changes — which is the whole
payoff of having drawn the seam before it was needed.

- **Phase 1 (the global environment):** [`environment.md`](./environment.md)
- **Architecture:** [`lighting-probes-design.md`](../lighting-probes-design.md)
- **Buffer layouts, WGSL contracts, capture atlas:** [`probes-shader-reference.md`](./probes-shader-reference.md)

```ts
import { EnvironmentSystem, ProbeVolume, ReflectionProbes } from 'null-graph/environment';
```

---

## 1. What Phase 2 adds

| | Phase 1 | Phase 2 |
|---|---|---|
| Diffuse indirect | one SH-L2 set for the whole scene | SH-L1 **per probe**, trilinearly blended |
| Specular indirect | one prefiltered sky cube | **local** cubes with parallax proxies, blending to the sky |
| Source | an HDR file or a procedural sky | **the rendered scene**, lit and shadowed |
| Bounces | none — the sky is the only source | **multi-bounce**, converging over refresh cycles |
| Cost | zero per frame after the bake | `probesPerFrame` captures per frame, bounded |

Concretely, these become possible:

**Colour bleeding.** A white sphere beside a red wall picks up red on its wall-facing side. This is
the headline result and it is impossible in Phase 1 by construction.

**Indoor/outdoor in one scene.** A room interior reads dark and locally-lit while the courtyard
outside reads sky-lit, with a smooth transition — because the probes near the door genuinely see
both.

**Local reflections.** A mirror floor in a room reflects *that room*, not the sky, with parallax
correction so reflected edges stay attached to real geometry as the camera moves.

**Lighting that responds.** Open a door, move a light, change the time of day → `invalidate()` the
affected bounds and the volume re-converges over the next few refresh cycles.

### 1.1 What is still out of scope

| Want | Status |
|---|---|
| Leak-free probes through thin walls | **mitigated, not solved** — §5. DDGI visibility is v2.0 |
| Volumes larger than one resident grid | v2.0 — streamed/cascaded volumes |
| Specular *from the volume* | by design: a grid cannot represent a mirror. Use reflection probes |
| Emissive geometry driving bounce beyond what a capture sees | v2.0 |

---

## 2. Minimum working setup

Phase 2 layers on a working Phase 1. The environment is **required** — it is both the fallback
outside the volume and the ambient the captures shade against.

```ts
// --- Phase 1 first ---
const env = new EnvironmentSystem(engine, {});
env.setProceduralSky({ /* ... */ });
engine.render([...env.getBakeStages()]);
env.copySHToParams();

// --- Phase 2 ---
const volume = new ProbeVolume(engine, {
    origin: [-64, 0, -64],
    spacing: 4,
    dims: [32, 16, 32],            // 16 384 probes → ~409 KB
    probesPerFrame: 16,
    environment: env,              // fallback + capture ambient
    visibility: vis,               // capture culling; there is no CPU draw path
    shadows,                       // captures must see shadows, or GI is flat and wrong
});

const refl = new ReflectionProbes(engine, { maxProbes: 16, faceSize: 128 });
refl.add({ center: [0, 2, 0], halfExtent: [8, 3, 8], capture: [0, 1.7, 0] });
volume.setReflectionProbes(refl);

// Meshes the CAPTURE draws (§3.2) — separate from your scene batches.
volume.registerCaptureMesh('rock', 0, { kind: 'pbr', instance: pbr,
    lighting: shadows.wrapLighting(lights) }, lights);

// Scene materials: the SAME contract as Phase 1, now backed by the volume.
const indirect = volume.getShaderLayout(refl);
const batch = vis.createInstanceDrawBatch(scenePass, 'rock', 0, {
    material: { kind: 'pbr', instance: pbr, lighting: shadows.wrapLighting(lights), indirect },
});
volume.applyToBatch(batch, lights, env, refl);   // the ONE group-2 bind group (§3.1)

// --- per frame ---
function frame() {
    engine.updateCamera(camera);
    lights.upload(); vis.upload();
    shadows.update(camera);
    volume.update(cameraPosition);               // schedules this frame's probes

    engine.render([
        ...vis.getComputeStages(),
        ...shadows.getComputeStages(),
        shadows.getAtlasPass(),

        ...volume.getCaptureStages(),             // capture-view caster culls
        volume.getCapturePass(),                  // ONE colour+depth pass, all tiles
        ...volume.getProjectStages(),             // tiles → SH-L1 → the volume textures
        ...refl.getPrefilterStages(),             // refl cubes → GGX mips

        scenePass,
        env.getSkyPass({ colorAttachments, depthStencilAttachment }),
        ...post.getPassSequence(),
    ]);
    requestAnimationFrame(frame);
}
```

Optionally fill the volume before the first frame:

```ts
await volume.prewarm((done, total) => setLoadingBar(done / total));
```

---

## 3. The four things that are easy to get wrong

### 3.1 `volume.applyToBatch` replaces *both* earlier binders

Group 2 now has **three** owners — lights (0–2), environment (3–6), probes (7–13) — and core WebGPU
guarantees only four bind groups, so they cannot be split. Exactly one call builds the group, and in
Phase 2 that call is the volume's:

```ts
volume.applyToBatch(batch, lights, env, refl);   // ✅ builds 0–13
```

```ts
lights.applyToBatch(batch);                      // ❌ overwrites everything
env.applyToBatch(batch, lights);                 // ❌ drops the probe bindings
```

The volume throws if it finds a group-2 bind group it did not build.

### 3.2 Capture meshes are registered separately from scene batches

`registerCaptureMesh()` is what the **capture pass** draws. It is a different set of batches from
your scene draws, because the capture renders from probe viewpoints into atlas tiles.

The material you pass **must** have `lighting` set — the capture shades the scene's *lit* response,
which is the whole point. Passing an unlit material captures albedo, and the GI comes out looking
like a flat colour wash. The volume refuses it:

```ts
volume.registerCaptureMesh('rock', 0,
    { kind: 'pbr', instance: pbr, lighting: shadows.wrapLighting(lights) },  // ← required
    lights);
```

`material.indirect` is auto-filled with this volume's own layout when you omit it — that is the
multi-bounce feedback path (§3.4), and getting it by default is deliberate.

### 3.3 Pass shadows, or the GI is wrong in a way that looks fine

A capture without shadows sees a fully lit scene from every probe, so every probe reports far too
much bounce and the volume washes the scene out uniformly. It does not look broken — it looks like
the ambient is too high — which is why the constructor refuses it rather than letting you discover
it later. `allowShadowlessCapture: true` opts in if you genuinely have no shadow system.

### 3.4 Multi-bounce converges — but it can also diverge

Each capture shades against the volume as it stood at the **start of the frame**, so frame *n*
captures direct light + bounce⁽ⁿ⁻¹⁾ and the volume ends up holding bounce⁽ⁿ⁾. After ~4 refresh
cycles the energy series has converged.

That feedback loop is only stable because the gain is < 1 (albedo × cosine < 1 per bounce). If a
scene brightens without bound, the cause is an albedo or cosine factor applied twice somewhere —
and a white room with one light is the test that shows it, because nothing else does.

---

## 4. Configuration

### 4.1 `ProbeVolumeConfig`

| Field | Default | Notes |
|---|---|---|
| `origin` | — | World position of probe (0,0,0), the volume's near corner. **Required.** |
| `spacing` | — | World distance between probes. **Required.** Drives leak behaviour (§5). |
| `dims` | — | Probe counts per axis. **Required.** Product ≤ `maxProbes`. |
| `environment` | — | **Required.** Phase 1 system: fallback + capture ambient. |
| `visibility` | — | **Required.** Capture culling; there is no CPU draw path. |
| `shadows` | — | Strongly recommended; see §3.3. |
| `captureFaceSize` | `32` | Capture cube face px. Four SH coefficients need no more. |
| `probesPerFrame` | `16` | Amortization budget. |
| `captureAtlasSize` | `1024` | Holds `probesPerFrame × 6` tiles + the reflection band. |
| `normalBiasTexels` | `0.5` | The cheapest leak mitigation (§5). |
| `borderFade` | `4` | World-unit fade at the volume edge, so the boundary is not a seam. |
| `intensity` | `1.0` | Multiplier on the volume lookup. |
| `maxProbes` | `65536` | Hard cap; a larger 3D allocation is refused. |
| `captureNear` / `captureFar` | `spacing×0.05` / `2×diagonal` | Capture frustum. |
| `distanceBias` | `16` | How much more often a probe at the camera refreshes vs. a far one. |
| `shBands` | `1` | Only L1. L2 needs 7 texture bindings where 4 exist. |
| `allowShadowlessCapture` | `false` | Opt in to §3.3. |
| `reflectionRefreshInterval` | `2` | Frames between reflection-probe refreshes. |
| `reflectionFaceSize` | `128` | Atlas band reserved for reflection captures. |

### 4.2 `ReflectionProbesConfig` / `ReflProbeConfig`

| Field | Default | Notes |
|---|---|---|
| `maxProbes` | `16` | Cube-array capacity, allocated once at `maxProbes × 6` layers. |
| `faceSize` | `128` | Specular carries high frequencies, so bigger than a diffuse probe. |
| `prefilterMips` | `6` | GGX roughness levels per cube. |
| `prefilterSamples` | `64` | Importance samples per texel. |

Per probe:

| Field | Default | Notes |
|---|---|---|
| `center` | — | Proxy box centre. **Required.** |
| `halfExtent` | — | Proxy half-extents. **Required.** Zero on any axis is rejected. |
| `capture` | `center` | Where the cube is captured from — **kept separate on purpose** (§6.2). |
| `blend` | `1` | Band inside the boundary over which weight fades 1 → 0. |
| `radius` | `\|halfExtent\|` | Influence radius; outside it the weight is 0. |
| `name` | — | Used in error messages. |

---

## 5. Light leaking — what you get and what you don't

A probe grid samples radiance at points that are not the shading point. When a wall separates the
two, light crosses it. **Phase 2 mitigates this and does not solve it**, and it is worth knowing
exactly which is which.

| Mitigation | Cost | Fixes |
|---|---|---|
| **Normal bias** — sample at `worldPos + N × bias` | free | A surface reading the probe on the other side of itself. The highest-value line in the module. |
| **Validity mask** — probes that captured themselves inside geometry are blended out | 1 binding | Probes buried in walls dumping interior darkness (or exterior daylight) into a neighbourhood. |
| **Capture-time enclosure detection** | free | Probes technically outside geometry but seeing mostly its inside. |

**How it still fails.** A wall thinner than `spacing`, bright on one side: all eight probes are
valid, none is inside geometry, and four genuinely see the bright side. Interior surfaces get a soft
glow through the wall. Halving `spacing` helps proportionally and costs memory cubically.

Practical guidance: **`spacing` should be smaller than your thinnest lit-both-sides wall.** That is
the single number that decides how much leaking you see.

The real fix is DDGI-style per-probe visibility (octahedral depth moments + a Chebyshev test). It is
v2.0 because it needs two more texture bindings than group 2 has —
[`lighting-probes-design.md`](../lighting-probes-design.md) §6.3.

---

## 6. Reflection probes

### 6.1 Blending

Probes are sorted by influence — smaller proxies win, since a small probe is a more specific
statement about a place — and the nearest few blend by a weight that falls to zero at the proxy
boundary. **Whatever weight remains unassigned goes to Phase 1's global cube**, so a scene with no
reflection probes behaves exactly as Phase 1 did, and total weight never exceeds 1 (which is what
keeps the split-sum recombination energy-preserving).

### 6.2 `capture` is not `center`

A cube captured at point *C* is only correct for a viewer at *C*. Parallax correction intersects the
reflection ray with the proxy box, then looks up the direction **from the capture point** to that
intersection. Conflating the two is the classic source of reflections that look right when the probe
sits at its box centre and slide visibly when it does not — so the two are separate fields, and U6
places the capture point off-centre precisely so they cannot be confused.

---

## 7. API reference

### `ProbeVolume`

| Member | Notes |
|---|---|
| `registerCaptureMesh(meshName, meshIndex, material, lights)` | Register a mesh the capture pass draws. `material.lighting` required (§3.2). |
| `setReflectionProbes(refl)` | Attach a `ReflectionProbes`; its `faceSize` must match `reflectionFaceSize`. |
| `update(cameraPos)` | Per frame: schedules probes, writes params. Allocation-free. |
| `getCaptureStages()` | Capture-view caster culls. Splice before the capture pass. |
| `getCapturePass()` | **The** capture pass — all active tiles + the reflection band, one pass. |
| `getProjectStages()` | Capture tiles → SH-L1 → the volume textures. |
| `getShaderLayout(refl?)` | The `ShaderIndirectLayout` — same two functions as Phase 1. |
| `applyToBatch(batch, lights, env?, refl?)` | Builds the whole group-2 bind group (§3.1). |
| `invalidate({ min, max })` | Mark probes in a world AABB dirty. Throws if the box misses the volume. |
| `prewarm(onProgress?)` | `async`. Captures every probe over as many internal frames as it takes. |
| `warmPending()` | Probes still never-captured. `0` once the volume is filled. |
| `probeCount`, `scheduledThisFrame`, `scheduledPicks`, `captureSlotCount` | Introspection. |
| `destroy()` | |

### `ReflectionProbes`

| Member | Notes |
|---|---|
| `add(cfg)` | Returns the layer index. Throws past `maxProbes` or on a zero half-extent. |
| `invalidate(layer)` | Force a re-capture of one probe. |
| `pickRefresh()` | The layer due for refresh this interval, or `-1`. |
| `capturePositionOf(layer)` | Where that probe captures from. |
| `primeRefresh(layer, skyIntensity)` / `markCaptured(layer)` | Capture-cycle bookkeeping. |
| `getPrefilterStages()` | GGX prefilter for the refreshed cube. |
| `intersects(layer, min, max)` | Whether a probe's proxy meets a world AABB — pairs with `invalidate`. |
| `layers()`, `probeCount`, `cubeView`, `probeDataBuffer` | Introspection. |
| `destroy()` | |

---

## 8. Cost

| | |
|---|---|
| Memory (volume) | `3 × probes × 8 B` + `probes × 1 B`. 32×16×32 → **409 KB** |
| Memory (capture atlas) | `captureAtlasSize²` rgba16float + depth. 1024² → ~8 MB |
| Memory (reflection) | `maxProbes × 6` faces at `faceSize²`, rgba16float, `prefilterMips` levels |
| Per frame | `probesPerFrame × 6` capture views: their culls, their tiles in one pass, one projection dispatch each |
| Full refresh | `probeCount / probesPerFrame` frames — 1 024 frames at the defaults, which is why the scheduler prioritises |
| Shading | 4 texture samples (3 volume + validity) + the reflection blend loop |

**The scheduler** prioritises never-captured probes first (so the volume fills in from nothing),
then near-camera, then explicitly invalidated, then age — so the far field cannot starve.

---

## 9. Errors

| Condition | Message |
|---|---|
| Volume too large | `dims <x>×<y>×<z> = <n> probes exceeds maxProbes <m>; a 3D texture of <bytes> would be allocated.` |
| Atlas too small | `probesPerFrame <n> needs <t> tiles of <s>px, exceeding the <a>px capture atlas; lower probesPerFrame or captureFaceSize.` |
| No environment | `config.environment is required — the volume falls back to the global environment outside its bounds and uses it as the capture ambient.` |
| No visibility system | `config.visibility is required; probe capture has no CPU draw path.` |
| Shadowless capture | `capturing without a ShadowSystem produces shadowless bounce light — pass shadows, or set allowShadowlessCapture:true to accept it.` |
| Unlit capture mesh | `capture mesh "<name>" has no material.lighting — the capture shades the scene's lit response.` |
| Group 2 already bound | `batch already has a group-2 bind group; group 2 has three owners (lights, environment, probes) and only ProbeVolume.applyToBatch(...) may build it.` |
| Reflection overflow | `probe <n> exceeds maxProbes <m>; the cube array is allocated once at <m> × 6 layers.` |
| Degenerate proxy | `probe "<name>" has a zero half-extent on axis <a>; parallax correction divides by the reflection ray and would produce NaN.` |
| `invalidate` misses | `invalidate() bounds do not intersect the volume; nothing was marked dirty.` |
| L2 requested | `shBands 2 needs 7 rgba16float 3D textures but only 4 group-2 texture bindings remain; use L1.` |

---

## 10. Testing

`npm run test:environment` runs Phase 1 and Phase 2 — **23 headless checks**. Phase 2's eleven cover
grid round-trips, the trilinear mirror, SH-L1 constant-environment energy, ringing clamp, the
validity fallback, parallax-from-capture-point, weight conservation, scheduler fairness, the
unchanged contract, the binding budget, and config validation.

**Colour bleeding and refresh convergence are GPU-verified:**
`npm run test:gpu -- --only probes` (Node/Dawn) or
`npm run test:gpu:browser -- --only probes` (Chrome). The shared `test/probes-gpu` harness has six
checks: finite/nonblank linear-HDR floor patches; near versus far red excess; disabled-volume and
neutral-wall controls; direct-only versus feedback captures; and bounded late-refresh change.
It uses 24 probes, one capture per frame, and 16 complete refresh cycles; the final four cycles
must change by less than 0.001 linear RGB and less than 10% of the initial step. No tone mapping
or byte quantization can hide divergence. Both NVIDIA/Dawn and AMD/Chrome pass.

The initial estimate includes global-environment fallback; convergence need not be monotonic.
This fixture proves feedback and convergence for a red-wall/open-floor scene, **not** the enclosed
white-room energy test (§10 T4), leak bounds, or reflection parallax. Those checks remain unwritten.
