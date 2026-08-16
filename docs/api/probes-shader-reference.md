# Probes — Shader & Buffer Reference

Everything the GPU side of Phase 2 touches: the group-2 tail, the volume texture layout, the capture
atlas, the WGSL the decorator emits, and the conventions a custom shader or a CPU mirror has to
match.

Usage-level docs: [`probes.md`](./probes.md) ·
Phase 1: [`environment-shader-reference.md`](./environment-shader-reference.md) ·
Architecture: [`lighting-probes-design.md`](../lighting-probes-design.md)

---

## 1. The group-2 tail

Phase 2 completes the group-2 allocation. Core WebGPU guarantees four bind groups; this is the last
of them, and bindings 7–13 are the last slots in it.

| Binding | Resource | Owner |
|---|---|---|
| 0–2 | `globalLights`, `clusterGrid`, `lightIndices` | `LightSystem` |
| 3–6 | `envSampler`, `envSpecular`, `brdfLUT`, `env` | `EnvironmentSystem` (Phase 1) |
| **7** | `probeR` — `texture_3d<f32>` | **`ProbeVolume`** |
| **8** | `probeG` — `texture_3d<f32>` | **`ProbeVolume`** |
| **9** | `probeB` — `texture_3d<f32>` | **`ProbeVolume`** |
| **10** | `probeValidity` — `texture_3d<f32>` | **`ProbeVolume`** |
| **11** | `probe` — `ProbeParams` uniform | **`ProbeVolume`** |
| **12** | `reflProbes` — `texture_cube_array<f32>` | **`ReflectionProbes`** |
| **13** | `reflProbeData` — `ReflProbeBuffer` storage | **`ReflectionProbes`** |

```wgsl
@group(2) @binding(7)  var probeR: texture_3d<f32>;
@group(2) @binding(8)  var probeG: texture_3d<f32>;
@group(2) @binding(9)  var probeB: texture_3d<f32>;
@group(2) @binding(10) var probeValidity: texture_3d<f32>;
@group(2) @binding(11) var<uniform> probe: ProbeParams;
@group(2) @binding(12) var reflProbes: texture_cube_array<f32>;
@group(2) @binding(13) var<storage, read> reflProbeData: ReflProbeBuffer;
```

**The group is full.** `reflProbeData` is the last fragment-stage storage buffer available (7 of 8 —
`lighting-ibl-design.md` §4.4) and binding 13 is the last slot. Anything after this must displace
something; DDGI visibility is deferred for exactly this reason.

### 1.1 One builder, three owners

`ProbeVolume.applyToBatch(batch, lights, env, refl)` concatenates all three owners' entries into one
`setRenderPassBindGroup` call. A bind group is built atomically, so no other combination works — and
the volume throws if it finds a group-2 group it did not build.

### 1.2 The anti-prune, and why bindings 12–13 are touched by the diffuse path

`layout: 'auto'` prunes bindings a shader does not reference, and `createBindGroup` then throws on
the extra entries. A **Lambert or Toon** material only ever reaches `getIndirectDiffuse` — it never
calls `getIndirectSpecular`, so the reflection bindings would be pruned and the shared bind group
would fail to build for those materials only.

So the diffuse body deliberately references `reflProbes` and `reflProbeData` in a zero-weight term.
It is a real reference, not a comment, and test U10 asserts it stays:

```js
assert(diffuseBody.includes('reflProbes'),    'diffuse body touches the cube array (binding 12)');
assert(diffuseBody.includes('reflProbeData'), 'diffuse body touches the probe storage (binding 13)');
```

Remove it and PBR keeps working while Lambert and Toon break — the kind of asymmetric failure that
takes a while to attribute.

---

## 2. The volume

### 2.1 Three textures, one channel each

| Texture | Format | Texel |
|---|---|---|
| `probeR` | `rgba16float` | `(L1x, L1y, L1z, L0)` for red |
| `probeG` | `rgba16float` | same, green |
| `probeB` | `rgba16float` | same, blue |
| `probeValidity` | `r8unorm`-class | per-probe confidence; `0` = never captured / enclosed |

Four L1 coefficients, four channels, one texture per colour channel. Nothing wasted, nothing split.

**Why L1 and not L2.** L2 is 9 coefficients × 3 channels = 27 values → **7** `rgba16float` textures
instead of 3. At 32×16×32 that is 917 KB vs 393 KB — affordable — but it is 7 *texture bindings*
where 4 remain. Phase 1's single global probe affords L2 because it lives in a **uniform**; a volume
cannot, because it lives in **bindings**. The budget picked the band, not aesthetics.

### 2.2 The blend is hardware

`texture_3d<f32>` with `rgba16float` is core, filterable, and writable as
`texture_storage_3d<rgba16float, write>`. So:

- the 8-probe trilinear blend is a **single `textureSampleLevel`** — free, no manual weights;
- the projection kernel `textureStore`s into the volume directly — no staging, no copy.

**Texel-centre alignment is load-bearing.** A probe at grid coordinate `(i,j,k)` must sample at
`(i + 0.5) / dims`, not `i / (dims - 1)`. `probeSampleUVW` does that remap; without it, sampling
exactly on a probe would blend two texels and U2's "one weight of exactly 1" could never hold.

### 2.3 `ProbeParams` — 16 floats, 64 bytes, binding 11

```wgsl
struct ProbeParams {
    origin    : vec4<f32>,   // .xyz world origin of probe (0,0,0); .w = intensity
    invExtent : vec4<f32>,   // .xyz = 1 / (spacing * (dims-1)); .w = spacing
    dims      : vec4<f32>,   // .xyz = grid dims as floats;       .w = normalBias in texels
    fade      : vec4<f32>,   // .x = border fade width (world);   .yzw reserved
};
```

### 2.4 `ReflProbe` — 12 floats, 48 bytes each, binding 13

```wgsl
struct ReflProbe {
    center     : vec4<f32>,   // .xyz proxy box centre;   .w = blend distance
    halfExtent : vec4<f32>,   // .xyz proxy half-extents; .w = influence radius
    capture    : vec4<f32>,   // .xyz CAPTURE position;   .w = cube-array layer index
};
struct ReflProbeBuffer { header: vec4<f32>, probes: array<ReflProbe> };  // header.x = count
```

`capture` is stored separately from `center` on purpose — see §4.2.

---

## 3. The volume lookup

```wgsl
fn getIndirectDiffuse(worldPos: vec3<f32>, N: vec3<f32>) -> vec3<f32> {
    let biased = worldPos + N * (probe.dims.w * probe.invExtent.w);   // normal bias, §5
    let uvw    = (biased - probe.origin.xyz) * probe.invExtent.xyz;
    if (outside(uvw)) { return getIndirectDiffuseGlobal(N); }         // Phase 1 fallback

    let r = textureSampleLevel(probeR, probeSampler, sampleUVW, 0.0);
    let g = textureSampleLevel(probeG, probeSampler, sampleUVW, 0.0);
    let b = textureSampleLevel(probeB, probeSampler, sampleUVW, 0.0);
    let valid = textureSampleLevel(probeValidity, probeSampler, sampleUVW, 0.0).r;

    let local  = evalSHL1probe(r, g, b, N) * probe.origin.w;
    let border = volumeBorderWeight(uvw, probe.fade.x * probe.invExtent.x);
    return mix(getIndirectDiffuseGlobal(N), local, valid * border);
}

fn evalSHL1probe(r: vec4<f32>, g: vec4<f32>, b: vec4<f32>, N: vec3<f32>) -> vec3<f32> {
    let d = vec4<f32>(N, 1.0);      // cosine constants folded in at projection time
    return max(vec3<f32>(dot(r, d), dot(g, d), dot(b, d)), vec3<f32>(0.0));
}
```

Three properties, each preventing a specific bug:

- **Both fallbacks blend toward Phase 1, never toward zero.** A half-captured volume looks like a
  slightly flat scene, not one with black holes in it. U5 asserts the validity-0 case equals the
  global result *exactly*.
- **`max(·, 0)` is load-bearing.** L1 rings for the same reason L2 does; unclamped, the negative lobe
  reads as black patches facing away from the dominant bounce direction. U4 pins it.
- **The cosine constants live in the stored coefficients.** One multiply saved per pixel, and the
  stored value *is* irradiance — so a CPU readout of a probe is directly interpretable.

---

## 4. Reflection blending & parallax

### 4.1 Weight conservation

```wgsl
var acc = vec3<f32>(0.0);
var w   = 0.0;
for (var i = 0u; i < count && w < 1.0; i = i + 1u) {
    let wi = min(reflInfluenceWeight(p, worldPos), 1.0 - w);   // clamp keeps Σw ≤ 1
    ...
    w += wi;
}
acc += textureSampleLevel(envSpecular, envSampler, Rg, mip).rgb * horizon * (1.0 - w);  // remainder
```

The `min(·, 1 - w)` is what makes the blend energy-preserving: total probe weight never exceeds 1
and the environment gets exactly the remainder. U7 asserts it across overlapping probes. The
split-sum recombination that follows is byte-for-byte Phase 1's, which is what keeps the white
furnace test valid in Phase 2.

### 4.2 Parallax reprojects from the **capture point**

```wgsl
fn parallaxCorrect(p: ReflProbe, worldPos: vec3<f32>, R: vec3<f32>) -> vec3<f32> {
    let invR     = 1.0 / R;
    let first    = (p.center.xyz + p.halfExtent.xyz - worldPos) * invR;
    let second   = (p.center.xyz - p.halfExtent.xyz - worldPos) * invR;
    let furthest = max(first, second);
    let dist     = min(min(furthest.x, furthest.y), furthest.z);
    let hit      = worldPos + R * dist;
    return normalize(hit - p.capture.xyz);       // CAPTURE point, not center
}
```

Using `p.center` on the last line looks correct whenever a probe was captured at its box centre and
slides visibly when it was not. U6 places the capture point off-centre so the two cannot be
confused.

---

## 5. The capture atlas

Probe capture is the **shadow atlas rendered in colour** — same allocator, same `cameraBuffer`, same
per-batch `viewport`, same dirty-tracked amortization. It builds no new render architecture.

| Property | Value |
|---|---|
| colour | `rgba16float`, `RENDER_ATTACHMENT \| TEXTURE_BINDING`, `captureAtlasSize²` |
| depth | `depth24plus`, same size |
| diffuse tile | `captureFaceSize²` (default 32), 6 per probe |
| reflection band | reserved strip at `reflectionFaceSize²` (default 128) |
| views | 6 per probe, 90° `perspectiveZO`, each a 108-float `Camera` |
| culling | `VisibilitySystem` per view, Hi-Z off |

A 32² face is deliberately tiny: the output is four coefficients, a signal with no high-frequency
content. 32² × 6 = 6 144 texels per probe is generous.

**The viewport y-flip.** A captured tile is a rendered viewport — NDC +y lands on texture row 0 — so
tile row `y` sees cube-face `v = 1 - uv.y`. Both the kernel and the CPU mirror apply it:

```wgsl
let dir = faceDirection(face, vec2<f32>(uv.x, 1.0 - uv.y));
let sa  = cubeTexelSolidAngle(uv, FACE_SIZE);     // NOT flipped — see §6
```

**Whatever suppresses a draw suppresses its cull.** A probe not scheduled this frame must not
dispatch its capture culls either; a zero-area viewport stops rasterization and nothing else. With
16 of 16 384 probes scheduled, getting that wrong means culling for all of them, forever.

### 5.1 Projection to SH-L1

One workgroup of 64 threads strides the probe's 6 × `faceSize²` texels, accumulating 13 floats
(12 coefficients + validity) in workgroup memory, then tree-reduces. WGSL has no `atomic<f32>`, so
this is the same single-workgroup shape Phase 1 uses.

Sky texels (`ndcZ >= 1`) take the prefiltered environment instead of the captured colour — that is
how a probe that sees open sky gets the right answer without the sky being geometry.

Validity comes out of the same pass: a texel whose linear depth is under `spacing × 0.25` counts as
enclosed; past 60 % enclosed the probe is written invalid.

---

## 6. Conventions a CPU mirror must match

`probeMath.ts` mirrors `probeShaders.ts` so the math is testable headless. It is **a second
implementation**, and it drifts. Two traps, both of which have actually bitten:

### 6.1 `cubeTexelSolidAngle` takes different units on each side

| | Signature | Pass |
|---|---|---|
| **WGSL** | `cubeTexelSolidAngle(uv: vec2<f32>, size: u32)` | normalized texel-centre UV |
| **CPU** | `cubeTexelSolidAngle(x, y, size)` | **integer texel indices** |

Same math, different parameterisation, and **no type error** — `number` accepts `0.0625` as happily
as `0`. Passing UVs to the CPU form shrinks every texel's solid angle: Σ comes to 7.78 instead of
4π and the whole projection scales down ~1.6×, silently, with plausible-looking output.

The invariant that catches it: **a constant radiance `c` must project to L0 == `c`**, because the
solid angles sum to exactly 4π. That is U3, and it is worth more than reading the two side by side.

### 6.2 "No depth supplied" is not "everything is sky"

`ndcZ >= 1` is the sky sentinel. The CPU mirror's `depth` argument is optional (the GPU always has
one), and defaulting the absent case to `1` means every texel takes the sky path and the captured
colour is never read — the projection returns zeros. The guard is on *whether depth exists*, not on
its value:

```ts
const hasDepth = opts.depth !== undefined;
const ndcZ = hasDepth ? opts.depth![t] : 0;
if (hasDepth && ndcZ >= 1) { /* sky */ } else { /* colour */ }
```

### 6.3 Fix the mirror toward the kernel — after checking which is right

Both bugs above were in the **CPU** mirror; the WGSL was correct. Fixing them the other way round
would have broken the GPU path to make a test pass. Establish which side is right *first*.

---

## 7. Generated modules

| Builder | Produces |
|---|---|
| `buildProbeStructsWGSL(group)` | bindings 7–13 + `ProbeParams`/`ReflProbe` structs + helpers |
| `buildProbeIndirectLogicWGSL()` | `getIndirectDiffuse` / `getIndirectSpecular`, volume-backed |
| `buildProbeIndirectLayout(envLayout, group)` | the composed `ShaderIndirectLayout` handed to materials |
| `buildProbeProjectionWGSL({ faceSize })` | the capture → SH-L1 projection kernel |
| `buildReflAssembleWGSL()` | capture band → reflection cube face assembly |
| `buildPrefilterWGSL({ …, baseArrayLayer })` | **Phase 1's kernel**, targeting one cube of the array |

That last row is the point: reflection prefiltering is Phase 1's GGX kernel with a layer offset, not
a second implementation.

Every module is parsed with `wgsl_reflect` in the test suite, and the same **no-backtick** rule
applies as Phase 1 — a backtick in a WGSL comment terminates the JS template literal carrying it.

---

## 8. Test map

| Test | Pins |
|---|---|
| U1 | grid ↔ world round-trip; index/coord round-trip; outside maps outside |
| U2 | trilinear mirror: eight ⅛s between probes, one 1 on a centre (texel-centre alignment) |
| U3 | constant environment → L0 == c (**catches the solid-angle unit mismatch**) |
| U4 | L1 ringing clamped ≥ 0 everywhere |
| U5 | validity 0 ⇒ the lookup equals the global environment *exactly*, not black |
| U6 | parallax reprojects from the capture point, not the box centre |
| U7 | probe weights never exceed 1; the environment gets the remainder |
| U8 | scheduler fairness: no starvation, near ≥ 4× far |
| U9 | the contract is byte-identical to Phase 1's; the material fragment is unchanged |
| U10 | bindings 7–13 exactly once; the diffuse anti-prune holds |
| §11 | config validation messages |

`npm run test:environment` runs Phase 1 and Phase 2 (23 checks). GPU integration is
[`lighting-probes-design.md`](../lighting-probes-design.md) §10 T1–T7.
