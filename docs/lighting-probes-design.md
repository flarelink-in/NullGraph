# NullGraph Lighting — Phase 2: Irradiance Volumes & Probes — Technical Design

**Target:** `src/extras/environment/` (`ProbeVolume`, `ProbeCapture`, `ReflectionProbes`, `probeMath`)
**Package:** `null-graph/environment` ✅ exported (same subpath; Phase 2 is additive)
**Status:** `v1.2` — **✅ implemented.** Headless suite: `npm run test:environment`
(`environment-probes.test.mjs`, U1–U10 + §11). GPU integration (§10 T1–T7) needs a browser harness.
**Prerequisite:** [`lighting-ibl-design.md`](./lighting-ibl-design.md) — Phase 1. Read it first;
this document reimplements its §5 contract and inherits its §3 integration facts and §4.4 binding map.
**Depends on:** Phase 1; [`shadows-design.md`](./shadows-design.md) §12.1 (`depthOnly`,
`cameraBuffer`) and [`shadows-atlas-design.md`](./shadows-atlas-design.md) §12.1 (per-batch
`viewport`) — the capture path is the shadow atlas architecture pointed at a different target.
**Roadmap:** [`roadmap.md`](./roadmap.md) §3 · [`roadmap-modules.md`](./roadmap-modules.md) §5.
**API:** [`api/probes.md`](./api/probes.md) (usage) · [`api/probes-shader-reference.md`](./api/probes-shader-reference.md) (layouts, WGSL contracts) · Phase 1: [`api/environment.md`](./api/environment.md)

---

## 1. Purpose

Phase 1 gives every surface in the scene the *same* indirect light. That is right for an object on
an open plain and wrong everywhere else: inside a building, under an arch, beside a red wall, in a
cave. The missing quantity is **where** — indirect light varies through space, and a single
environment cannot express it.

Phase 2 makes indirect light a function of position:

- an **irradiance volume** — a grid of SH-L1 probes, hardware-trilinearly blended — for diffuse;
- **local reflection probes** with parallax-corrected proxies for specular;
- **GPU probe capture** that renders the scene from each probe and relights it, amortized across
  frames, so the volume is not a bake step but a living part of the frame;
- and, because captures feed on the previous capture's result, **multi-bounce** GI that converges
  over a handful of frames for free.

### The relationship to Phase 1, stated exactly

Phase 1 §5.2 defined:

```wgsl
fn getIndirectDiffuse(worldPos: vec3<f32>, N: vec3<f32>) -> vec3<f32>;
fn getIndirectSpecular(worldPos: vec3<f32>, N: vec3<f32>, V: vec3<f32>, roughness: f32, F0: vec3<f32>) -> vec3<f32>;
```

and passed `worldPos` even though it ignored it. **Phase 2 is the release that uses that
parameter.** It re-emits the same two functions with volume-sampling bodies, falling back to Phase
1's global environment outside the volume. No material, no shader builder, and no draw batch
changes — which is the entire payoff of having specified the contract before it was needed.

### Non-goals for Phase 2

- **DDGI-style per-probe visibility.** Octahedral depth moments with a Chebyshev test are the
  *correct* fix for light leaking and they are v2.0 (§6.3 explains the baseline mitigations and
  what they cost).
- **Probe streaming.** One volume, resident. Cascaded/streamed volumes for open worlds are v2.0.
- **Irradiance from dynamic emissive geometry** beyond what a capture happens to see.
- **Specular from the volume.** Diffuse comes from the grid, specular from reflection probes. A grid
  cannot represent a mirror.

### Design principles

1. **Reimplement the contract; touch nothing else.** §1 above. If a material file changes in this
   release, something has gone wrong.
2. **Capture is the shadow atlas, again.** Rendering the scene from N viewpoints into tiles of one
   texture, culled on the GPU, amortized by dirty tracking — that is `shadows-atlas-design.md`,
   verbatim, with a color target instead of a depth one. Phase 2 builds no new render architecture.
3. **The bind-group budget is already spent.** Phase 1 §4.4 allocated bindings 7–13 of group 2 to
   this release. That table is a hard ceiling, and §4.4 here shows the design living inside it.
4. **Leaking is the design problem.** Every irradiance-volume decision below is really a decision
   about light leaking through walls. §6.3 says what is mitigated, what is not, and how it fails.
5. **Conservative degradation.** A probe that was never captured reads as the global environment,
   never as black. A volume that does not cover a point falls back smoothly. The failure mode of
   this system is "looks like Phase 1," never "looks broken."

---

## 2. Architecture

```
   engine.render([
      ...vis.getComputeStages(),
      ...shadows.getComputeStages(),
      shadows.getAtlasPass(),

      ...probes.getCaptureStages(),   // caster culls for the probes scheduled this frame
      probes.getCapturePass(),        // ONE color+depth pass; 6 faces × probesPerFrame tiles
      ...probes.getProjectStages(),   // capture tiles → SH-L1 → the 3 volume textures
      ...refl.getPrefilterStages(),   // reflection-probe cubes → GGX mips (Phase 1 §6.3 reused)

      scenePass,                      // materials call the SAME two functions as Phase 1
      skyPass,
      ...post.getPassSequence(),
   ]);
```

```
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ ProbeVolume            getIndirectDiffuse(worldPos, N)                   │
   └───┬──────────────────────────────────────────────────┬───────────────────┘
       │                                                   │
  ┌────▼────────────┐                              ┌───────▼────────┐
  │ 3 × texture_3d  │  hardware trilinear          │ validity 3d    │  leak weight
  │ rgba16float     │  = 8-probe blend, free       │ r8unorm        │
  │ (L1xyz + L0)/ch │                              └────────────────┘
  └────▲────────────┘
       │ written by
  ┌────┴──────────────┐   ┌──────────────────┐   ┌───────────────────────────┐
  │ SH-L1 projection  │◀──│ capture atlas    │◀──│ capture pass (scene, lit) │
  │ (compute)         │   │ rgba16float 2D   │   │ = shadow atlas, in color  │
  └───────────────────┘   └──────────────────┘   └───────────┬───────────────┘
                                                              │ reads last frame's
                                                              │ volume as its ambient
                                                              └──▶ multi-bounce (§7.3)

   ┌──────────────────────────────────────────────────────────────────────────┐
   │ ReflectionProbes       getIndirectSpecular(worldPos, N, V, rough, F0)    │
   │ texture_cube_array + parallax box proxies + influence blend → Phase 1 cube│
   └──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Integration facts (additional to Phase 1 §3)

| Fact | Source | Consequence |
|---|---|---|
| Group 2 bindings 7–13 are the only slots left, and `maxBindGroups` is 4. | Phase 1 §4.4; `BatchManager.ts:94-102` | The volume gets **4 texture bindings + 1 uniform**, reflection probes **1 texture + 1 storage**. That budget is what forces SH-**L1** rather than L2 (§4.2) and a cube *array* rather than per-probe cubes. |
| `texture_3d<f32>` with `rgba16float` is core, filterable, and usable as `texture_storage_3d<rgba16float, write>`. | WebGPU core format/dimension tables | The 8-probe trilinear blend is **hardware**, and the projection compute writes the volume directly — no staging, no copy. |
| `texture_cube_array` is a core view dimension. | WebGPU `GPUTextureViewDimension` | All reflection probes live in one binding. Per-probe cube bindings would not fit (§3 row 1). |
| The shadow atlas already renders N off-camera views into tiles of one texture in **one pass**, GPU-culled, with dirty-tracked amortization. | `shadows-atlas-design.md` §5–§8 | Probe capture reuses it wholesale: same allocator, same `cameraBuffer`, same `viewport`, same round-robin. §7 is short because of this. |
| `VisibilitySystem` draw batches accept a `camera` override and per-batch `occlusion`. | `shadows-design.md` §9.3, `VisibilitySystem.ts:239,244` | Capture-view caster culling is free, exactly as cascade culling was. |
| The `Camera` struct is 108 floats and any view that fills it is indistinguishable from a camera to the cull and draw shaders. | `Camera.ts:35`; `shadows-design.md` §4.1 | A probe cube face is a `Camera` with a 90° `perspectiveZO`. The third use of this trick, and the reason it was worth generalizing. |
| Phase 1's GGX prefilter is a per-mip compute stage over a `2d-array` storage view of a cube. | `lighting-ibl-design.md` §6.3 | Reflection-probe prefiltering is the same kernel with `baseArrayLayer = probeIndex * 6`. No second implementation. |
| The capture pass needs materials to shade with *some* indirect term, and reading this frame's volume while writing it is a hazard. | `RenderPipeline.ts:65-70` (pass boundaries) | Capture shades against **last frame's** volume (§7.3) — which is what makes multi-bounce fall out for free rather than being a separate feature. |

---

## 4. Data model

### 4.1 The volume

An axis-aligned grid: `origin`, `spacing`, `dims: [x, y, z]`. Probe `(i,j,k)` sits at
`origin + spacing * (i,j,k)`. Regular, so the lookup is a single normalized coordinate and the
blend is a hardware sample — no acceleration structure, no traversal, in keeping with the engine's
no-scene-graph stance.

### 4.2 SH-L1, and why not L2

Diffuse irradiance is stored as **SH band 1** — four coefficients per color channel — packed one
channel per 3D texture:

| Texture | Format | Texel contents |
|---|---|---|
| `probeR` | `rgba16float` | `(L1x, L1y, L1z, L0)` for red |
| `probeG` | `rgba16float` | same, green |
| `probeB` | `rgba16float` | same, blue |
| `probeValidity` | `r8unorm` | per-probe confidence, `0` = never captured / inside geometry |

The packing is exact: four coefficients, four channels, one texture per channel. Nothing is wasted
and nothing is split across textures.

Phase 1 uses **L2** for its single global probe; the volume uses **L1**. That asymmetry is
deliberate and worth the paragraph:

> L2 needs 9 coefficients × 3 channels = 27 values, which packs into **7** `rgba16float` textures
> instead of 3. At 32×16×32 probes that is 917 KB versus 393 KB — affordable — but it is **7
> texture bindings** where §3 row 1 says 4 exist. One global probe can afford L2 because it lives
> in a uniform; a volume cannot, because it lives in bindings. L1 captures the dominant direction
> and the ambient level, which is what a diffuse bounce actually is.

Memory for the defaults (32×16×32 = 16 384 probes): `3 × 16384 × 8 B` + `16384 × 1 B` = **409 KB**.

### 4.3 `ProbeParams` — the uniform

```wgsl
struct ProbeParams {
    origin      : vec4<f32>,   // .xyz world origin of probe (0,0,0); .w = intensity
    invExtent   : vec4<f32>,   // .xyz = 1 / (spacing * (dims-1)); .w = spacing
    dims        : vec4<f32>,   // .xyz = grid dims as floats; .w = normalBias in texels
    fade        : vec4<f32>,   // .x = border fade width (world units), .yzw reserved
};                             // 64 bytes
```

### 4.4 Reflection probes

| Resource | Binding | Contents |
|---|---|---|
| `reflProbes` | 12 | `texture_cube_array<f32>`, `rgba16float`, `prefilterMips` levels, `maxReflProbes` cubes |
| `reflProbeData` | 13 | `array<ReflProbe>` storage |

```wgsl
struct ReflProbe {
    center     : vec4<f32>,   // .xyz proxy box centre (world); .w = blend distance
    halfExtent : vec4<f32>,   // .xyz proxy box half-extents; .w = influence radius
    capture    : vec4<f32>,   // .xyz capture position (≠ centre in general); .w = layer index
};                            // 48 bytes
```

The capture position is stored separately from the proxy centre because parallax correction
reprojects **from where the cube was captured** to where the shading point is; conflating the two is
a classic source of reflections that slide as the camera moves (§8.2).

---

## 5. The volume lookup — `getIndirectDiffuse`

```wgsl
fn getIndirectDiffuse(worldPos: vec3<f32>, N: vec3<f32>) -> vec3<f32> {
    // Normal bias: step off the surface toward the probe grid so a probe *behind* the wall
    // is not the nearest sample. §6.3 — this is the cheapest leak mitigation and the first
    // one to reach for.
    let biased = worldPos + N * (probe.dims.w * probe.invExtent.w);

    let uvw = (biased - probe.origin.xyz) * probe.invExtent.xyz;
    if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) {
        return getIndirectDiffuseGlobal(N);          // Phase 1, outside the volume
    }

    // Hardware trilinear = the 8-probe blend, for free.
    let r = textureSampleLevel(probeR, probeSampler, uvw, 0.0);
    let g = textureSampleLevel(probeG, probeSampler, uvw, 0.0);
    let b = textureSampleLevel(probeB, probeSampler, uvw, 0.0);
    let valid = textureSampleLevel(probeValidity, probeSampler, uvw, 0.0).r;

    let local = evalSHL1(r, g, b, N) * probe.origin.w;

    // Two blends, both toward Phase 1 and never toward black:
    //   validity  — how much of the 8-probe neighbourhood was usable
    //   border    — a smooth exit at the volume's edge so the seam is not a line
    let border = volumeBorderWeight(uvw, probe.fade.x * probe.invExtent.x);
    return mix(getIndirectDiffuseGlobal(N), local, valid * border);
}

fn evalSHL1(r: vec4<f32>, g: vec4<f32>, b: vec4<f32>, N: vec3<f32>) -> vec3<f32> {
    // Irradiance from L1, with the cosine-convolution constants already folded into
    // the stored coefficients at projection time (§7.2).
    let d = vec4<f32>(N, 1.0);
    return max(vec3<f32>(dot(r, d), dot(g, d), dot(b, d)), vec3<f32>(0.0));
}
```

Three properties are worth naming, because each is a bug that this shape prevents:

- **Both fallbacks blend toward Phase 1's global environment, never toward zero.** A half-captured
  volume looks like a slightly flat scene, not a scene with black holes in it.
- **`max(·, 0)` on the L1 evaluation.** L1 rings for the same reason L2 does (Phase 1 §7.4); an
  unclamped dot product goes negative on surfaces facing away from the dominant bounce direction.
- **The cosine constants live in the stored coefficients, not here.** One multiply saved per pixel,
  and — more importantly — it makes the stored value *irradiance*, so a CPU-side debug readout of a
  probe is directly interpretable.

---

## 6. Light leaking — the design problem, stated honestly

A probe grid samples radiance at points that are not the shading point. When a wall separates the
two, light crosses it. This is the defining artifact of every irradiance volume ever shipped, and
the honest position is: **Phase 2 mitigates it and does not solve it.**

### 6.1 What the baseline does

| Mitigation | Cost | What it fixes |
|---|---|---|
| **Normal bias** (§5) — sample at `worldPos + N × bias` | free | A surface reading the probe on the *other* side of itself. The single highest-value line in the module. |
| **Validity mask** — probes whose capture found itself inside geometry are marked invalid and blended out | 1 binding, `dims` bytes | Probes buried in walls dumping interior darkness (or exterior daylight) into a neighbourhood. |
| **Capture-time backface rejection** (§7.2) — a capture ray that hits a backface contributes 0 and decrements validity | free (inside the projection) | Probes that are *technically* outside geometry but see mostly its inside. |

### 6.2 How it still fails

A thin wall — thinner than `spacing` — with bright light on one side. All eight probes are valid,
none is inside geometry, and four of them genuinely see the bright side. Interior surfaces get a
soft glow through the wall. Raising `spacing` resolution helps proportionally and costs memory
cubically. This is the artifact to expect and to budget geometry around.

### 6.3 What actually fixes it, and why it is v2.0

DDGI's answer: store a small octahedral **depth-moment** map per probe (mean and mean-squared
distance), and weight each of the eight probes by a Chebyshev visibility test between the probe and
the shading point. It removes leaking almost entirely and it is the right eventual design.

It is deferred because it is not a small addition: a second probe atlas, a second projection kernel,
per-probe octahedral parameterization with border gutters, and — critically — **two more texture
bindings**, which Phase 1 §4.4 does not have. Landing it means first reclaiming binding budget
(merging validity into a spare channel, or moving reflection probes to their own pass). That is a
structural change and it deserves its own document rather than a footnote here.

---

## 7. Probe capture — the shadow atlas, in color

### 7.1 The capture pass

For each probe scheduled this frame, six 90° `perspectiveZO` views — the identical construction
`shadows-atlas-design.md` §9.1 uses for point-light shadows. Each view is a `Camera` buffer and an
atlas tile:

| Property | Value |
|---|---|
| atlas | `rgba16float`, `RENDER_ATTACHMENT \| TEXTURE_BINDING`, default 1024² |
| depth | `depth24plus`, same size, transient |
| tile | `captureFaceSize`² (default 32) → 6 tiles per probe, 16 probes per 1024² atlas at 32² |
| batches | `views × casterMeshes`, `isIndirect`, `cameraBuffer` per view, `viewport` per tile |
| culling | `VisibilitySystem` per view, Hi-Z off |

A 32² face is deliberately tiny. The output is four SH coefficients — a signal with no
high-frequency content whatsoever — so resolution buys nothing beyond suppressing sampling noise.
32² × 6 = 6 144 texels per probe is generous for four coefficients.

Because this is the shadow atlas architecture and not a new one, everything it already solved comes
along: tile allocation, the `setViewport` containment guarantee (and the bleed hazard that made it
required — `shadows-atlas-design.md` §6.2), dirty tracking, and GPU caster culling.

### 7.2 Projection to SH-L1

One compute stage per scheduled probe (or one dispatch with a probe index per workgroup): read the
probe's 6 tiles, project onto the four L1 basis functions weighted by true per-texel solid angle
(Phase 1 §7.2's `solidAngle` — the same function, the same reason), fold in the cosine-convolution
constants, and `textureStore` into `probeR/G/B` at `(i,j,k)`.

Validity is computed in the same pass: a capture texel whose depth is below `spacing * 0.25`
indicates the probe is inside or against geometry; the fraction of such texels drives the validity
value. A probe that is more than 60% enclosed is written as validity 0.

Both the reduction shape and the reason it is a single workgroup are Phase 1 §7.2's — WGSL has no
`atomic<f32>`, and 6 144 texels reduce comfortably in one workgroup.

### 7.3 Multi-bounce, for free

The capture pass renders the scene with its normal materials, which call `getIndirectDiffuse` —
which reads the volume. Reading the volume while writing it would be a hazard, so the capture shades
against the volume as it stood at the **start of the frame**.

That one-frame delay is not a compromise to be apologized for; it is the mechanism:

- frame *n* captures direct light + bounce ⁽ⁿ⁻¹⁾ → the volume holds bounce ⁽ⁿ⁾
- after ~4 refresh cycles the energy series has converged to within a few percent

So the volume converges to multi-bounce global illumination with no additional pass, no ray
tracing, and no second data structure. With `probesPerFrame: 16` and 16 384 probes, a full cycle is
1 024 frames — which is why §7.4 exists.

### 7.4 Scheduling

Capturing 16 384 probes uniformly is pointless: most are far away or in unchanged regions. The
scheduler picks `probesPerFrame` probes by priority:

| Priority input | Effect |
|---|---|
| never captured | highest — the volume fills in from nothing on first run |
| distance to camera | near probes refresh far more often |
| explicit `invalidate(bounds)` | the caller moved a light or opened a door |
| age since last capture | prevents starvation of the far field |

`prewarm()` captures the whole volume synchronously over as many frames as it takes, for editors
and for load screens, and reports progress so a loading bar is possible.

---

## 8. Reflection probes — `getIndirectSpecular`

### 8.1 Structure

`maxReflProbes` cubes in one `texture_cube_array`, each prefiltered with **Phase 1's GGX kernel**
(§6.3 there) at `baseArrayLayer = probeIndex * 6`. Capture uses §7.1's pass at a larger face size
(default 128²), since specular *does* carry high-frequency content.

### 8.2 Parallax correction

A cube captured at point `C` is only correct for a viewer at `C`. Everywhere else, the reflection
must be reprojected — intersect the reflection ray with the probe's proxy box, then look up the
direction from the capture point to that intersection:

```wgsl
fn parallaxCorrect(p: ReflProbe, worldPos: vec3<f32>, R: vec3<f32>) -> vec3<f32> {
    let invR   = 1.0 / R;
    let first  = (p.center.xyz + p.halfExtent.xyz - worldPos) * invR;
    let second = (p.center.xyz - p.halfExtent.xyz - worldPos) * invR;
    let furthest = max(first, second);
    let dist = min(min(furthest.x, furthest.y), furthest.z);
    let hit  = worldPos + R * dist;
    return normalize(hit - p.capture.xyz);      // FROM THE CAPTURE POINT, not the box centre
}
```

Using `p.center` in that last line instead of `p.capture` is the bug referenced in §4.4: the
reflection appears correct when the probe was captured at its box centre and slides visibly when it
was not. §10 U6 places the capture point off-centre precisely so the two cannot be confused.

### 8.3 Blending and fallback

Probes are sorted by influence — smaller proxy volumes win, since a small probe is a more specific
statement about a place — and the nearest few are blended by a weight that falls to zero over
`blend distance` at the proxy boundary. Whatever weight remains unassigned goes to **Phase 1's
global cube**, so a scene with no reflection probes at all behaves exactly as Phase 1 did.

```wgsl
fn getIndirectSpecular(worldPos: vec3<f32>, N: vec3<f32>, V: vec3<f32>, roughness: f32, F0: vec3<f32>) -> vec3<f32> {
    let R   = reflect(-V, N);
    let mip = roughness * env.specularMip;

    var acc = vec3<f32>(0.0);
    var w   = 0.0;
    for (var i = 0u; i < probeCount && w < 1.0; i = i + 1u) {
        let p = reflProbeData[i];
        let wi = min(influenceWeight(p, worldPos), 1.0 - w);
        if (wi <= 0.0) { continue; }
        acc += textureSampleLevel(reflProbes, envSampler, parallaxCorrect(p, worldPos, R), u32(p.capture.w), mip).rgb * wi;
        w   += wi;
    }
    acc += textureSampleLevel(envSpecular, envSampler, R, mip).rgb * (1.0 - w);   // Phase 1

    let dfg = textureSampleLevel(brdfLUT, envSampler, vec2<f32>(max(dot(N, V), 0.0), roughness), 0.0).rg;
    return acc * (F0 * dfg.x + dfg.y) * env.intensity;
}
```

The split-sum recombination at the end is byte-for-byte Phase 1's, which is what keeps the white
furnace test (Phase 1 §10 T1) valid in Phase 2 — and §10 T1 here re-runs it *with* probes to prove
the blend is energy-preserving.

---

## 9. Public API

```ts
import { ProbeVolume, ReflectionProbes } from 'null-graph/environment';

const volume = new ProbeVolume(engine, {
    origin: [-64, 0, -64],
    spacing: 4,
    dims: [32, 16, 32],           // 16 384 probes → 409 KB
    captureFaceSize: 32,
    probesPerFrame: 16,
    normalBiasTexels: 0.5,
    borderFade: 4,
    intensity: 1.0,
    environment: env,             // Phase 1 — the fallback and the capture's ambient
    visibility: vis,              // GPU caster culling for captures
    shadows,                      // captures see shadows; without them GI is flat and wrong
});

const refl = new ReflectionProbes(engine, { maxProbes: 16, faceSize: 128, environment: env });
refl.add({ center: [0, 2, 0], halfExtent: [8, 3, 8], capture: [0, 1.7, 0], blend: 1.0 });

// One layout, both systems, same contract as Phase 1
const indirect = volume.getShaderLayout(refl);
vis.createInstanceDrawBatch(scenePass, 'rock', 0, {
    material: { kind: 'pbr', instance: pbr, lighting: shadows.wrapLighting(lights), indirect },
});
volume.applyToBatch(batch, lights, env, refl);     // the ONE combined group-2 bind group

volume.invalidate({ min: [-10, 0, -10], max: [10, 4, 10] });   // a door opened
await volume.prewarm();                                        // optional: fill before first frame
```

`applyToBatch` taking all four owners is the same argument Phase 1 §9 made, escalated: group 2 now
has three owners and exactly one of them may build the bind group. §11 errors on any other order.

---

## 10. Test suite

Headless (`npm run test:environment`) unless marked GPU.

**Pure-logic unit tests:**

- **U1 — grid ↔ world round-trip.** `worldToUVW(probePosition(i,j,k))` recovers `(i,j,k)/(dims-1)`
  exactly for all corners and 1000 random interior probes; points outside the volume map outside
  `[0,1]³`.
- **U2 — trilinear weights.** The CPU mirror of the hardware blend at a point exactly between eight
  probes gives eight equal weights summing to 1; at a probe centre, one weight of 1.
- **U3 — SH-L1 of a constant environment.** A uniform radiance `c` capture projects to irradiance
  `c` in every direction within 1%. Same single-number validation as Phase 1 U6, one band down.
- **U4 — L1 clamping.** A capture that is bright in one hemisphere and black in the other evaluates
  `>= 0` in every direction.
- **U5 — validity fallback is toward the environment.** With `validity = 0` everywhere,
  `getIndirectDiffuse`'s CPU mirror equals Phase 1's global result **exactly** — not approximately,
  and not black.
- **U6 — parallax uses the capture point.** A probe with `capture` offset from `center` by half its
  extent: the corrected direction for a known ray differs from the naive `center`-based one by a
  measurable angle, and matches an analytic ray-box solve. Swap `capture` for `center` in the
  implementation → the test fails.
- **U7 — influence weights never exceed 1.** Overlapping probes at 100 random points: total probe
  weight ≤ 1, and the environment gets exactly the remainder. This is what makes §8.3
  energy-preserving.
- **U8 — the scheduler is fair.** Over 10 000 simulated frames with `probesPerFrame: 16`, every
  probe is captured at least once (no starvation) and near probes are captured at least 4× as often
  as far ones.
- **U9 — the contract is unchanged.** `volume.getShaderLayout(refl)` declares `getIndirectDiffuse`
  and `getIndirectSpecular` with **byte-identical signatures** to Phase 1's, and the generated
  material fragment is unchanged from Phase 1's given the same material. This is the test that
  proves the release cost no material edits.
- **U10 — binding map.** Generated WGSL uses group 2 bindings 7–13 only, and never 0–6 (lights and
  environment).

**Integration tests (GPU):**

- **T1 — white furnace, with probes.** Phase 1 T1 re-run inside a probe volume in a uniform
  environment. **Expect** the same flat energy response — the volume must not add or lose energy
  when it agrees with the environment.
- **T2 — the red wall.** A white sphere beside a strongly red diffuse wall, direct light only from
  above. **Expect** the sphere's wall-facing side reads measurably red (color bleeding), and its
  far side does not. The headline GI test: it fails on Phase 1 by construction.
- **T3 — leak bound.** A sealed box inside a bright environment, `spacing` at half the wall
  thickness. **Expect** interior surfaces below a stated leak threshold with normal bias and
  validity on; disable both and the interior brightens measurably. This test does not assert "no
  leak" — §6.2 says that would be false — it asserts the mitigations are wired and quantifies what
  they buy.
- **T4 — multi-bounce converges.** A white room, one light. Sample interior brightness across 20
  refresh cycles. **Expect** monotonic increase converging within 3% and **no divergence** — a
  feedback gain > 1 (an albedo or cosine constant applied twice) shows up here as a room that gets
  brighter forever, and nowhere else.
- **T5 — volume border.** A surface crossing the volume boundary shows no discontinuity: no adjacent
  sample pair differs by more than 5% across the border with `borderFade` at its default.
- **T6 — capture cost is bounded.** `probesPerFrame: 16` issues exactly `96` capture views' worth of
  draws and no more, regardless of volume size or instance count.
- **T7 — reflection probe parallax.** A mirror floor inside a box probe; move the camera. **Expect**
  reflected box edges stay attached to the real geometry. With parallax disabled they slide — the
  visual form of U6.

---

## 11. Errors

| Condition | Message |
|---|---|
| volume too large | `ProbeVolume: dims <x>×<y>×<z> = <n> probes exceeds maxProbes <m>; a 3D texture of <bytes> would be allocated.` |
| capture atlas too small | `ProbeVolume: probesPerFrame <n> needs <t> tiles of <s>px, exceeding the <a>px capture atlas; lower probesPerFrame or captureFaceSize.` |
| no environment | `ProbeVolume: config.environment is required — the volume falls back to the global environment outside its bounds and uses it as the capture ambient (lighting-probes-design.md §5, §7.3).` |
| no visibility system | `ProbeVolume: config.visibility is required; probe capture has no CPU draw path.` |
| capture without shadows | `ProbeVolume: capturing without a ShadowSystem produces shadowless bounce light — pass shadows, or set allowShadowlessCapture:true to accept it.` |
| group 2 bound by someone else | `ProbeVolume: batch already has a group-2 bind group; group 2 has three owners (lights, environment, probes) and only ProbeVolume.applyToBatch(batch, lights, env, refl) may build it.` |
| reflection probe overflow | `ReflectionProbes: probe <n> exceeds maxProbes <m>; the cube array is allocated once at <m> × 6 layers.` |
| degenerate proxy box | `ReflectionProbes: probe "<name>" has a zero half-extent on axis <a>; parallax correction divides by the reflection ray and would produce NaN.` |
| `invalidate` outside the volume | `ProbeVolume: invalidate() bounds do not intersect the volume; nothing was marked dirty.` |
| L2 requested | `ProbeVolume: shBands 2 needs 7 rgba16float 3D textures but only 4 group-2 texture bindings remain (lighting-probes-design.md §4.2); use L1.` |

---

## 12. Engine touchpoints

### 12.1 Core engine — none beyond what shadows already introduced

The capture pass needs `PipelineConfig.cameraBuffer`, the per-batch `viewport`, and
`RenderPassConfig.size` (`shadows-design.md` §12.1, `shadows-atlas-design.md` §12.1). All land with
shadows, one release earlier. Phase 2 adds **no new core requirement** — which is itself the
argument for having built the shadow atlas generally rather than as a shadow-specific mechanism.

`ComputeStageNode.enabled` is inherited too, and §7.4's scheduler is exactly what it is for: a
probe not captured this frame must not dispatch its caster culls either. The shadow atlas learned
this the expensive way — a zero-area viewport suppresses rasterization and nothing else
(`shadows-atlas-design.md` §8.3) — so the rule carries over verbatim: **whatever suppresses a
tile's draw suppresses its cull in the same place.** With `probesPerFrame: 16` out of 16 384
probes, getting that wrong would mean culling for all of them, every frame, forever.

`depthOnly` is not used here: the capture pass writes color.

### 12.2 Module changes — additive

- `ProbeVolume.getShaderLayout(refl?)` returns a `ShaderIndirectLayout` (Phase 1 §5.1). Nothing in
  `materials/` changes in this release.
- Phase 1's GGX prefilter kernel gains a `baseArrayLayer` parameter so it can target one cube of the
  reflection-probe array (§8.1). One uniform, no second kernel.
- Phase 1's `solidAngle` and `faceDirection` move to a shared `envMath` module — both phases need
  them and they must not diverge.

### 12.3 Optional — filed, not needed

- **Indirect dispatch** (roadmap §6). The per-view capture culls over-dispatch the same way the
  shadow atlas's do; the same optional change removes the same waste.
- **A binding-budget reclamation pass.** Folding `probeValidity` into a spare channel would free one
  binding — the down payment on §6.3's DDGI visibility. Not needed for Phase 2; recorded because
  v2.0 will need it.

---

## 13. Build order

Each step renders correctly at its end. Steps 1–3 produce a working volume from **synthetic** probe
data, so the lookup path is fully proven before capture — which is the expensive, hard-to-debug half.

1. **Volume storage + lookup (§4, §5)** — the three 3D textures, `ProbeParams`, and
   `getIndirectDiffuse` reimplemented against them, filled with a **constant** matching Phase 1's
   environment. Assert U1–U5, U9, U10, T1, T5. **The image must not change at all** — a volume that
   agrees with the environment is invisible, and that is the strongest possible proof the swap is
   clean.
2. **Synthetic gradients** — fill the volume procedurally (e.g. red on one side, blue on the other).
   Assert the shading varies spatially and smoothly. Still no capture, still fully debuggable.
3. **SH-L1 projection (§7.2)** — from a synthetic cube, not the scene. Assert U3, U4.
4. **Capture pass (§7.1)** — the shadow atlas in color, one probe, `prewarm()` only. Assert T6.
   This is the step that pulls in `cameraBuffer` and `viewport`; keeping it isolated means a failure
   here is unambiguously about capture, not about the lookup proven in step 1.
5. **The full volume + scheduler (§7.4)** — Assert U8, T2. **T2 is the gate**: color bleeding is the
   entire reason this release exists, and it is the first step where the scene looks different in a
   way a user would name.
6. **Validity + normal bias (§6.1)** — Assert T3. Leak mitigation after the light is correct, so
   any regression is isolated to the mitigation.
7. **Multi-bounce (§7.3)** — enable the volume as the capture's ambient. Assert T4. Watch for
   divergence; if the room brightens without bound, a cosine or albedo factor is applied twice.
8. **Reflection probes (§8)** — Assert U6, U7, T7. Last, because they are independent of everything
   above and their absence degrades cleanly to Phase 1.

### Deferred

- **DDGI per-probe visibility (v2.0)** — §6.3. The real leak fix; needs binding budget reclaimed
  first (§12.3).
- **Streamed / cascaded volumes (v2.0)** — for worlds larger than one resident grid. The scheduler
  in §7.4 is the hook: streaming is a priority function that also allocates.
- **Probes for specular occlusion** — using capture depth to occlude the specular term, which needs
  the same depth moments as DDGI and lands with it.
- **Emissive-driven bounce** beyond what a capture happens to see, e.g. explicit area-light probes.
