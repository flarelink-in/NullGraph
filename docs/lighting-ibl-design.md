# NullGraph Lighting — Phase 1: Image-Based Lighting — Technical Design

**Target:** `src/extras/environment/` (`EnvironmentSystem`, `CubePrefilter`, `shProject`, `envShaders`)
**Package:** `null-graph/environment` (new subpath export)
**Status:** `v1.2` design, implementation-ready.
**Companion:** [`lighting-probes-design.md`](./lighting-probes-design.md) — Phase 2: irradiance
volumes, GPU probe capture, and local reflection probes. Read this one first — it defines the
contract Phase 2 reimplements.
**Depends on:** `LightSystem` ([`api/lights.md`](./api/lights.md)), the material shader chunks
(`materials/materialFragments.ts`), and — for the capture path only — the two engine additions
[`shadows-design.md`](./shadows-design.md) §12.1 already introduced.
**Roadmap:** [`roadmap.md`](./roadmap.md) §5.5, §3.

---

## 1. Purpose

Replace the constant that is currently standing in for the entire indirect lighting term. In the
PBR fragment today:

```wgsl
let ambient = vec3<f32>(0.3) * albedo * ao;      // materialFragments.ts, pbr fragment
```

Everything a surface receives that did not come from a light in the `LightSystem` loop is that
`0.3`. It is why metals read as flat grey, why a rough dielectric looks identical indoors and
outdoors, and why no amount of work on the direct lighting loop makes a scene look finished.

Phase 1 replaces it with the split-sum image-based lighting model: a **prefiltered environment
cube** for specular, a **spherical-harmonic irradiance projection** for diffuse, and a **BRDF
integration LUT** to join them. It also establishes the contract — `getIndirectDiffuse` /
`getIndirectSpecular` — that Phase 2 reimplements against a probe volume without any material
changing again.

### The single most important pre-existing fact

**`ShaderLightingLayout` is already the pattern for injecting lighting WGSL into materials.**
`LightSystem.getShaderLayout()` returns `{ wgslStructs, wgslLogic }`, `buildMaterialFragment`
splices it in, and every material calls the `getVisibleLightCount` / `getIncidentLight` functions it
defines (`LightSystem.ts:117-378`, `materialFragments.ts:568-576`). That is the *direct* lighting
contract.

Phase 1 adds its **sibling for indirect lighting** — same shape, same injection point, same
composition story. Nothing about how materials are built changes; one more layout gets composed in,
and the hardcoded `0.3` becomes a function call.

### Non-goals for Phase 1

- **Spatially varying indirect light.** One environment for the whole scene. Probe volumes and
  local reflection probes are Phase 2 — and they are the reason §5's contract is a *function of the
  shading point*, not a constant lookup.
- **Real-time capture from the scene.** Phase 1 takes an equirectangular HDR image or a procedural
  sky. Capturing the rendered scene into a cube is Phase 2's probe machinery, applied to one probe.
- **Screen-space reflections.** Post's business, and complementary — SSR handles the near field,
  IBL the far field.
- **Ray-traced GI.** ⛔ blocked on WebGPU.

### Design principles

1. **A sibling contract, not a fork.** `ShaderIndirectLayout` mirrors `ShaderLightingLayout` field
   for field. A material that wants indirect light asks for it the same way it asks for direct.
2. **Everything expensive is a bake.** Prefiltering is a one-time compute chain (roadmap constraint
   #3). Per-frame cost is zero unless the sky is animated, and then it is one amortized mip.
3. **The bind-group budget is the real constraint.** Core WebGPU guarantees **four** bind groups
   (§3), and NullGraph already spends all four. Indirect lighting therefore lives **inside the
   lighting group**, never in a new one. This is not a stylistic preference — a fifth group does
   not exist on the baseline device.
4. **Detect, don't require.** `rgba16float` and `rg16float` storage textures, `2d-array` storage
   views, cube sampling, and `textureSampleLevel` in compute are all core WebGPU. No optional
   feature appears anywhere.
5. **Fail loud.** An environment that never got prefiltered, a cube whose faces are not square, an
   SH set that was never projected — each is a named build-time error (§11), not a black scene.

---

## 2. Architecture

The bake runs once (or amortized); the frame path is a bind group and two function calls in the
fragment shader.

```
   // once, at load:
   await env.setEquirect('/hdr/sunset.hdr');   // or env.setProceduralSky({...})
   engine.render([ ...env.getBakeStages(), ...rest ]);   // bake stages self-retire after they run

   // every frame:
   engine.render([
      ...vis.getComputeStages(),
      ...shadows.getComputeStages(),
      shadows.getAtlasPass(),
      scenePass,          // materials call getIndirectDiffuse / getIndirectSpecular
      skyPass,            // env.getSkyPass() — depth-tested background, after opaque
      ...post.getPassSequence(),
   ]);
```

```
   equirect HDR ──┐
   procedural sky ─┼──▶ ┌──────────────┐   ┌────────────────┐   ┌──────────────┐
   (Phase 2: scene)┘    │ source cube  │──▶│ mip chain      │──▶│ GGX prefilter│──▶ specular cube
                        │ rgba16float  │   │ (box, 6 layers)│   │ per-mip      │    (6 mips)
                        └──────┬───────┘   └────────────────┘   └──────────────┘
                               │
                               │  8×8 mip, 384 texels
                               ▼
                        ┌──────────────┐                        ┌──────────────┐
                        │ SH-L2 project│──▶ 27 floats            │ BRDF LUT bake│──▶ rg16float 256²
                        │ (1 workgroup)│    (in EnvParams)       │ (split-sum)  │    (scene-independent)
                        └──────────────┘                        └──────────────┘
                               │                                        │
                               └──────────────┬─────────────────────────┘
                                              ▼
                              ┌───────────────────────────────────┐
                              │ ShaderIndirectLayout (WGSL)       │
                              │ getIndirectDiffuse(N)             │  ← composed into group 2,
                              │ getIndirectSpecular(N,V,rough,F0) │    alongside the lights
                              └───────────────────────────────────┘
```

Every box is an existing primitive: `createComputeStage` for the bakes, per-mip storage views
exactly as `HiZPyramid.ts:71-72` does them, and a `ShaderLightingLayout`-shaped WGSL pair for the
material bridge.

---

## 3. Integration facts (verified against this repository)

Load-bearing. The first is the one that dictates the whole binding design.

| Fact | Source | Consequence |
|---|---|---|
| **Core WebGPU guarantees `maxBindGroups: 4`** — groups 0–3 and no more. NullGraph already uses **all four**: 0 = camera + instance/ECS storage, 1 = material, 2 = lights (**hardcoded** `@group(2)` inside `LightSystem`'s emitted WGSL), 3 = shadows. | WebGPU limits; `BatchManager.ts:94-102`, `LightSystem.ts:134,250-252`, `shadows/shadowShaders.ts:44-46` | Indirect lighting **must** share group 2 with the lights. §4.4 is the resulting binding map, and it is not negotiable on the baseline device. |
| `LightSystem` occupies group 2 bindings **0–2** (`lights`, `clusterGrid`, `lightIndices`) and builds that bind group itself in `applyToBatch`. | `LightSystem.ts:74-86` | Bindings 3+ are free. `LightSystem` needs one additive change: expose its entries so a combined group-2 bind group can be built (§12.2). |
| The PBR fragment's entire indirect term is `let ambient = vec3<f32>(0.3) * albedo * ao;`, computed **after** the light loop and added to `Lo`. | `materialFragments.ts`, `pbr` fragment | The single injection point. §5.3's replacement is one line, guarded by an `if (config.indirect)` exactly like `if (config.lighting)`. |
| `PBR_MATH_WGSL` provides `DistributionGGX`, `GeometrySmith`, `fresnelSchlick`, `getNormalFromMap` — but **no** `fresnelSchlickRoughness` and no importance-sampling helpers. | `materialFragments.ts:135-173` | The indirect layout emits its own helpers in `wgslStructs`; it does not edit `PBR_MATH_WGSL`. |
| The PBR fragment **tonemaps and gamma-corrects inline**: `finalColor/(finalColor+1)` then `pow(·, 1/2.2)`. | `materialFragments.ts`, `pbr` fragment | With a real HDR environment this becomes the visible quality ceiling, and it double-applies if `PostChain`'s `Tonemap` also runs. §10 makes it a named config, not a surprise. |
| `HiZPyramid` writes one mip per compute stage using `createView({ baseMipLevel, mipLevelCount: 1 })` for the destination and the previous mip as the source. | `HiZPyramid.ts:58-79` | **This**, not `PostChain`, is the precedent for the prefilter chain. Roadmap §5.5 calls it "the PostChain pyramid pattern"; `PostChain` is render-pass based over 2D `TargetHandle`s (`PostChain.ts:458-466`, `Bloom.ts:79-125`) and has no cube or array concept. §6 corrects that. |
| `RenderPipeline` draws a batch with **no vertex/index buffer** as `draw(3, instanceCount)` — a fullscreen triangle. | `RenderPipeline.ts:143-144` | The skybox pass (§8) needs no geometry and no engine change. |
| `createBatch` binds `batch.storageBuffer` at group 0 binding 1 unconditionally, and `layout:'auto'` prunes it if unused — making `createBindGroup` throw. | `BatchManager.ts:94-102` | The skybox reads its exposure/rotation out of that buffer, which is a genuine use rather than an anti-prune trick (§8.2). |
| `depthWriteEnabled` and `depthCompare` are per-batch config. | `BatchManager.ts:66-71` | The skybox draws after opaque with `depthCompare:'less-equal'`, `depthWriteEnabled:false`. Core, no change. |
| WGSL has **no `atomic<f32>`**; the clustered binner's atomics are all `u32`. | `LightSystem.ts:397`, `clusteredComputeShader.ts` | SH projection cannot accumulate with float atomics. §7.2 projects from a small mip in **one workgroup** instead, which sidesteps the reduction entirely. |
| `rgba16float` and `rg16float` are core **write-only storage** formats, and `rgba16float` is **filterable** as a sampled texture (unlike `rgba32float`, which needs `float32-filterable`). | WebGPU core format table | Every texture in this document is `rgba16float`/`rg16float`. Using `rgba32float` would silently cost a feature flag and lose linear filtering. |

---

## 4. Data model

### 4.1 The source cube

`rgba16float`, 6 array layers, `TEXTURE_BINDING | STORAGE_BINDING`, with a full mip chain.
Default face size 512. Two views over the same texture:

| View | Dimension | Used for |
|---|---|---|
| `createView({ dimension: 'cube' })` | cube | sampling during prefilter, SH projection, and by the skybox |
| `createView({ dimension: '2d-array', baseMipLevel: L, mipLevelCount: 1 })` | 2d-array | the compute write target for mip `L` |

The same texture being both a cube sampler source and a 2d-array storage target is the crux of the
bake, and it is exactly the two-views-of-one-texture pattern `HiZPyramid.ts:69-72` already uses in
2D.

### 4.2 The prefiltered specular cube

`rgba16float`, 6 layers, `prefilterMips` levels (default 6), `TEXTURE_BINDING | STORAGE_BINDING`.
Mip `L` holds the environment convolved with a GGX lobe of roughness `L / (prefilterMips - 1)`, so
mip 0 is a mirror and mip 5 is fully rough. Sampled at runtime with
`textureSampleLevel(envSpecular, envSampler, R, roughness * (prefilterMips - 1))`.

Base 128² × 6 mips ≈ **200 KB**. The source cube can be released after the bake unless the skybox
draws it (§8) or the sky animates.

### 4.3 `EnvParams` — the uniform

```wgsl
struct EnvParams {
    shR         : array<vec4<f32>, 3>,   // SH-L2 red   — 9 coeffs packed 3×vec4 (§7.3)
    shG         : array<vec4<f32>, 3>,   // green
    shB         : array<vec4<f32>, 3>,   // blue
    rotation    : vec4<f32>,             // .xyz = yaw/pitch/roll cos-sin cache; .w = yaw radians
    intensity   : f32,                   // diffuse + specular multiplier
    specularMip : f32,                   // prefilterMips - 1, as a float
    horizonFade : f32,                   // occlusion of below-horizon specular; default 1.0
    _pad        : f32,
};                                       // 3×48 + 16 + 16 = 176 bytes
```

The nine SH coefficients per channel are the *entire* diffuse environment: 27 floats replacing what
would otherwise be an irradiance cubemap, a texture binding, and a sampler. That frugality is what
makes room for Phase 2 in the same bind group.

### 4.4 The group-2 binding map — the whole budget, laid out

Because §3's first fact leaves no slack, this table is the contract both phases hold to:

| Binding | Resource | Owner | Phase |
|---|---|---|---|
| 0 | `lights` (storage) | `LightSystem` | ships |
| 1 | `clusterGrid` (storage) | `LightSystem` | ships |
| 2 | `lightIndices` (storage) | `LightSystem` | ships |
| 3 | `envSampler` (filtering sampler) | environment | **P1** |
| 4 | `envSpecular` (`texture_cube<f32>`) | environment | **P1** |
| 5 | `brdfLUT` (`texture_2d<f32>`) | environment | **P1** |
| 6 | `env` (`EnvParams` uniform) | environment | **P1** |
| 7–9 | `probeR/G/B` (`texture_3d<f32>`) | probes | P2 |
| 10 | `probeValidity` (`texture_3d<f32>`) | probes | P2 |
| 11 | `probe` (uniform) | probes | P2 |
| 12 | `reflProbes` (`texture_cube_array<f32>`) | probes | P2 |
| 13 | `reflProbeData` (storage) | probes | P2 |

Per-stage resource counts at full Phase 2, against core WebGPU limits — this is why the plan fits:

| Resource | Used (fragment) | Core limit |
|---|---|---|
| storage buffers | **7** — `ecs`, lights, grid, indices, **shadow `views` + `slots`**, reflProbeData | 8 |
| sampled textures | 11 — 3 material, 1 shadow atlas, env cube, LUT, 3 probe, validity, refl array | 16 |
| samplers | 4 — material, shadow comparison, env, probe | 16 |
| uniform buffers | 5 — camera, material, `EnvParams`, probe, `ShadowParams` | 12 |
| bind groups | 4 | **4** |

Bind groups are the binding constraint, and they are exactly full. Any future subsystem that wants
GPU resources in the fragment stage must join an existing group; this table is where it declares
where.

**Storage buffers are the second-tightest, at 7 of 8.** The shadow atlas moved its view and slot
tables from a uniform to `var<storage, read>` when it shipped (`shadows/atlasShaders.ts:71-72`) —
the right call, since it decoupled view count from shader recompilation, but it spent two of this
budget. The consequence for Phase 2 is concrete: `reflProbeData` is the **last** fragment-stage
storage buffer available. Anything after it must be a texture, a uniform, or must displace an
existing binding.

---

## 5. The indirect-lighting contract

### 5.1 The type — a sibling of `ShaderLightingLayout`

```ts
// materials/types.ts — next to ShaderLightingLayout, deliberately identical in shape
export interface ShaderIndirectLayout {
    wgslStructs: string;   // bindings + helper declarations
    wgslLogic: string;     // getIndirectDiffuse / getIndirectSpecular
}
```

### 5.2 The two functions every implementation provides

```wgsl
/** Cosine-convolved irradiance arriving at `worldPos` with normal `N`, divided by PI.
 *  Multiply by albedo (and kD) to get the diffuse indirect term. */
fn getIndirectDiffuse(worldPos: vec3<f32>, N: vec3<f32>) -> vec3<f32>;

/** Pre-integrated specular radiance for the reflection lobe, already including the
 *  split-sum BRDF term — so the caller multiplies by nothing but occlusion. */
fn getIndirectSpecular(worldPos: vec3<f32>, N: vec3<f32>, V: vec3<f32>,
                       roughness: f32, F0: vec3<f32>) -> vec3<f32>;
```

Both take `worldPos` even though Phase 1 ignores it. That is the single most consequential line in
this document: it is what lets Phase 2 swap in a probe volume by re-emitting the same two functions,
with **zero** edits to any material. A contract that omitted `worldPos` would force a material
rewrite in Phase 2 — and it would be discovered late, after the materials had shipped.

### 5.3 The material change

`ShaderConfig` gains `indirect?: ShaderIndirectLayout` next to `lighting`, and the PBR fragment's
hardcoded ambient becomes a branch of exactly the shape the file already uses for `config.lighting`:

```wgsl
${config.indirect ? `
// --- IMAGE-BASED / PROBE INDIRECT LIGHT ---
let F_ibl  = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
let kD_ibl = (vec3<f32>(1.0) - F_ibl) * (1.0 - metallic);
let ambient = (kD_ibl * albedo * getIndirectDiffuse(in.worldPos, N)
               + getIndirectSpecular(in.worldPos, N, V, roughness, F0)) * ao;
` : `
let ambient = vec3<f32>(0.3) * albedo * ao;      // unchanged fallback
`}
```

`buildMaterialFragment` threads `indirect` through to `ShaderConfig` and appends
`indirect.wgslStructs` / `indirect.wgslLogic` to `bindings` alongside the lighting layout
(`materialFragments.ts:568-574`) — four added lines, and every existing call site is untouched
because `indirect` is optional.

**Lambert and Toon** get the diffuse half only (`ambient` currently `material.ambient * baseColor`
and `0.15 * baseColor`), which is a two-line change each and makes a scene look consistent when
materials are mixed. Basic, Emissive, and Matcap are unlit by definition and are left alone.

---

## 6. The bake chain

All four stages are `ComputeStageNode`s spliced in once. `getBakeStages()` returns them; after the
frame in which they run, `EnvironmentSystem` drops them from its chain (or keeps them, for an
animated sky — §6.5).

The correct precedent here is `HiZPyramid`, **not** `PostChain`. `PostChain` composes fullscreen
*render* passes over 2D pooled targets (`PostChain.ts:458-466`) and its `TargetSpec` has no notion
of a cube, an array layer, or a mip level (`post/types.ts`). Trying to bake a cube through it would
mean six chains and a manual copy per face. The per-mip compute + storage-view pattern is a direct
fit and already proven in this repo.

### 6.1 Stage A — source cube from equirect (or sky)

Dispatch `(size/8, size/8, 6)`; `z` is the cube face. Each thread turns its `(face, x, y)` into a
world direction, then samples the equirectangular source:

```wgsl
@group(0) @binding(0) var equirect : texture_2d<f32>;
@group(0) @binding(1) var srcSamp  : sampler;
@group(0) @binding(2) var dstCube  : texture_storage_2d_array<rgba16float, write>;

fn faceDirection(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let c = uv * 2.0 - 1.0;                 // [-1,1]
    switch face {
        case 0u: { return normalize(vec3<f32>( 1.0, -c.y, -c.x)); }   // +X
        case 1u: { return normalize(vec3<f32>(-1.0, -c.y,  c.x)); }   // -X
        case 2u: { return normalize(vec3<f32>( c.x,  1.0,  c.y)); }   // +Y
        case 3u: { return normalize(vec3<f32>( c.x, -1.0, -c.y)); }   // -Y
        case 4u: { return normalize(vec3<f32>( c.x, -c.y,  1.0)); }   // +Z
        default: { return normalize(vec3<f32>(-c.x, -c.y, -1.0)); }   // -Z
    }
}
```

That `switch` is the WebGPU cube-face convention, and getting one sign wrong mirrors the
environment in a way that looks plausible until a recognizable feature appears in the reflection.
§10 U1 pins all six against known directions.

The procedural-sky variant replaces the `textureSampleLevel` with an analytic gradient
(zenith/horizon/ground colors plus a sun disc), so the module works with **no asset at all** — which
matters, because "IBL requires you to find an HDR file" is how a feature goes unused.

### 6.2 Stage B — the source mip chain

`log2(size)` stages, each reading mip `L-1` as a cube and writing mip `L` as a 2d-array — a box
average, structurally identical to `buildHiZMipWGSL` (`visibilityShaders.ts:188-218`) with `max`
replaced by `mean`. These mips exist for two reasons: §6.3's firefly suppression needs them, and
§7.2 projects SH from the 8×8 level.

### 6.3 Stage C — GGX prefilter, one stage per mip

For mip `L`, roughness `r = L / (prefilterMips - 1)`, dispatch `(w/8, h/8, 6)`:

```wgsl
let N = faceDirection(face, uv);
let R = N; let V = N;                        // the standard split-sum approximation
var color = vec3<f32>(0.0);
var weight = 0.0;
for (var i = 0u; i < SAMPLE_COUNT; i = i + 1u) {
    let H = importanceSampleGGX(hammersley(i, SAMPLE_COUNT), N, roughness);
    let L = normalize(2.0 * dot(V, H) * H - V);
    let NdotL = dot(N, L);
    if (NdotL <= 0.0) { continue; }

    // Firefly suppression (Karis): pick a source mip whose texel solid angle matches the
    // sample's, so a single bright texel cannot survive as a hard dot in a rough reflection.
    let NdotH = max(dot(N, H), 0.0);
    let D     = DistributionGGX(N, H, roughness);
    let pdf   = (D * NdotH / (4.0 * max(dot(V, H), 1e-4))) + 1e-4;
    let saTexel  = 4.0 * PI / (6.0 * f32(srcSize) * f32(srcSize));
    let saSample = 1.0 / (f32(SAMPLE_COUNT) * pdf);
    let mip = select(0.5 * log2(saSample / saTexel), 0.0, roughness == 0.0);

    color  += textureSampleLevel(srcCube, srcSamp, L, mip).rgb * NdotL;
    weight += NdotL;
}
textureStore(dstMip, coord, face, vec4<f32>(color / max(weight, 1e-4), 1.0));
```

`SAMPLE_COUNT` defaults to 128 and mip 0 short-circuits to a straight copy (`roughness == 0` is a
mirror, and importance sampling it just adds noise). Without the solid-angle mip selection, a sun
disc in the source becomes a swarm of bright speckles at mid roughness — the classic IBL artifact,
and the reason §10 T2 measures variance rather than eyeballing it.

### 6.4 Stage D — the BRDF LUT

256×256 `rg16float`, `texture_storage_2d<rg16float, write>`, dispatch `(32, 32, 1)`. The split-sum
DFG integration over `(NdotV, roughness)`. It depends on **nothing** — not the environment, not the
scene — so it is baked once per device and could equally be shipped as data. It is computed rather
than shipped so the package carries no binary asset.

### 6.5 Animated skies

`updateMode: 'once'` (default) retires the bake stages after one frame. `updateMode: 'amortized'`
keeps them and rebuilds **one prefilter mip per frame**, cycling. A 6-mip chain fully refreshes in
6 frames — imperceptible for a day/night cycle, and it bounds the per-frame cost to a single
128²×6 dispatch. The SH projection (§7.2) is cheap enough to run every frame in that mode.

---

## 7. Diffuse irradiance as spherical harmonics

### 7.1 Why SH instead of an irradiance cube

Cosine-convolved irradiance is a very low-frequency signal — the L2 band captures it to within
~1% for all but pathological environments. Nine `vec3` coefficients therefore replace a 32²×6
cubemap plus its texture binding and sampler. In a design where §3's first fact says four bind
groups exist and all four are spoken for, spending 176 bytes of an existing uniform instead of a
new texture binding is not a micro-optimization — it is what leaves room for Phase 2.

### 7.2 Projection in a single workgroup

WGSL has no `atomic<f32>` (§3), so the usual "every texel atomically accumulates into nine
coefficients" does not exist here. Rather than build a two-pass reduction, project from the source
cube's **8×8 mip**: 6 faces × 64 texels = **384 samples**, well inside one workgroup.

```wgsl
var<workgroup> partial : array<array<vec3<f32>, 9>, 64>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    for (var k = 0u; k < 9u; k = k + 1u) { partial[t][k] = vec3<f32>(0.0); }

    // 384 texels / 64 threads = 6 texels each — one per face, same (x,y).
    for (var face = 0u; face < 6u; face = face + 1u) {
        let xy  = vec2<u32>(t % 8u, t / 8u);
        let uv  = (vec2<f32>(xy) + 0.5) / 8.0;
        let dir = faceDirection(face, uv);
        let dw  = solidAngle(uv, 8u);                       // per-texel solid angle
        let rgb = textureLoad(srcCube2DArray, xy, face, SH_MIP).rgb * dw;
        let sh  = shBasisL2(dir);                            // 9 scalars
        for (var k = 0u; k < 9u; k = k + 1u) { partial[t][k] += rgb * sh[k]; }
    }
    workgroupBarrier();

    // Tree reduction over 64 threads, then thread 0 applies the cosine-convolution constants
    // (A0=PI, A1=2PI/3, A2=PI/4) and writes EnvParams.shR/G/B.
    ...
}
```

`solidAngle` must be the true per-texel solid angle of a cube face, not `4π/384` — a cube's corner
texels subtend markedly less than its centre ones, and using a uniform weight tilts the whole
irradiance field toward the corners. §10 U3 catches it by checking that the weights sum to `4π`.

### 7.3 Packing 9 `vec3`s into `array<vec4, 3>` per channel

WGSL uniform arrays stride to 16 bytes per element, so `array<vec3<f32>, 9>` in a uniform costs 144
bytes and reads awkwardly. Storing each channel as `array<vec4<f32>, 3>` — 9 scalars packed
3-per-`vec4` with the twelfth slot unused — costs 48 bytes and unpacks with three loads. §10 U4
pins the packing order against the CPU-side writer, because a transposed coefficient produces
plausible-looking light coming from the wrong hemisphere.

### 7.4 Evaluation

```wgsl
fn getIndirectDiffuse(worldPos: vec3<f32>, N: vec3<f32>) -> vec3<f32> {
    let n = rotateY(N, -env.rotation.w);          // environment yaw
    let b = shBasisL2(n);
    var c = vec3<f32>(0.0);
    for (var k = 0u; k < 9u; k = k + 1u) { c += shCoeff(k) * b[k]; }
    return max(c, vec3<f32>(0.0)) * env.intensity;   // clamp: L2 can ring negative
}
```

The `max(·, 0)` is not cosmetic. An L2 projection of a high-contrast environment rings, and the
negative lobe shows up as black patches on surfaces facing away from the dominant light — a bug
that reads as "shadows in the wrong place." §10 U5 constructs exactly that environment.

---

## 8. The skybox pass

### 8.1 Shape

A fullscreen triangle drawn **after** the opaque pass, reconstructing a view ray per pixel from
`camera.invViewProj` (already maintained at float offset 48, `Camera.ts:79`):

```ts
const skyBatch = engine.createBatch(skyPass, {
    shaderCode: buildSkyWGSL({ environmentGroup: 2 }),
    strideFloats: 4,
    maxInstances: 1,
    depthWriteEnabled: false,
    depthCompare: 'less-equal',       // only where nothing was drawn (cleared depth = 1.0)
});
engine.updateBatchData(skyBatch, skyParams, 1);   // → draw(3, 1), RenderPipeline.ts:143-144
```

No vertex buffer, no index buffer, so `RenderPipeline` takes its `draw(3, instanceCount)` path. No
engine change.

### 8.2 The group-0 binding, used honestly

`createBatch` binds `batch.storageBuffer` at group 0 binding 1 unconditionally and `layout:'auto'`
prunes it if unused, which would make bind-group creation throw (§3). The skybox has no instances —
so rather than a dead reference, its four floats **are** the sky's parameters: `[exposure, yaw,
blurMip, _]`, written through `updateBatchData`. The binding is genuinely used, and animating the
sky's exposure needs no new buffer.

### 8.3 Which cube to sample

Sampling the **prefiltered** cube at `blurMip` rather than the source lets the same pass serve a
sharp sky and a deliberately blurred backdrop, and lets the source cube be released after the bake.
Default `blurMip: 0`.

---

## 9. Public API

```ts
import { EnvironmentSystem } from 'null-graph/environment';

const env = new EnvironmentSystem(engine, {
    sourceSize: 512,
    prefilterMips: 6,
    prefilterSamples: 128,
    lutSize: 256,
    intensity: 1.0,
    updateMode: 'once',
    environmentGroup: 2,        // §3 — must be the LightSystem's group
});

await env.setEquirect('/hdr/sunset.hdr');   // or:
env.setProceduralSky({ zenith: [0.2,0.4,0.9], horizon: [0.8,0.85,0.9], ground: [0.3,0.28,0.25], sunDir, sunIntensity: 12 });

// Bake (once). The stages self-retire in 'once' mode.
engine.render([...env.getBakeStages()]);

// Materials
const indirect = env.getShaderLayout();                   // ShaderIndirectLayout
vis.createInstanceDrawBatch(scenePass, 'rock', 0, {
    material: { kind: 'pbr', instance: pbr, lighting: shadows.wrapLighting(lights), indirect },
});
env.applyToBatch(batch, lights);        // builds the COMBINED group-2 bind group (§12.2)

// Sky
const skyPass = env.getSkyPass();       // splice after the opaque pass
env.setSkyParams({ exposure: 1.0, yaw: 0.0, blurMip: 0 });
```

`applyToBatch(batch, lights)` taking the `LightSystem` is deliberate and slightly unusual: group 2
has two owners, and a single call that builds the whole group is the only way to keep them from
overwriting each other's bind group. §11 errors if `LightSystem.applyToBatch` was called separately
on the same batch.

---

## 10. Test suite

Headless (`npm run test:environment`) unless marked GPU. The bake is pure math over known inputs, so
most of it is testable without a device.

**Pure-logic unit tests (no GPU):**

- **U1 — cube face directions.** `faceDirection(f, (0.5,0.5))` returns `+X,-X,+Y,-Y,+Z,-Z` exactly
  for `f = 0..5`; the four corners of each face are unit-length and land in the correct octant.
  This is the test that catches a mirrored environment.
- **U2 — equirect ↔ direction round-trip.** `dirToEquirectUV(faceDirection(f, uv))` recovers a
  direction within 1e-5 for 1000 random `(f, uv)` pairs.
- **U3 — solid angles sum to 4π.** Summing `solidAngle` over all 6×8×8 texels of the SH mip gives
  `4π` within 1e-4. A uniform `4π/384` weight fails.
- **U4 — SH packing round-trip.** Nine `vec3` coefficients → `array<vec4,3>` × 3 → back, exact. The
  generated WGSL's `shCoeff(k)` indexes the same order as the CPU writer.
- **U5 — SH ringing is clamped.** Project a synthetic environment that is white in one hemisphere
  and black in the other; the raw L2 evaluation goes negative somewhere, and
  `getIndirectDiffuse`'s CPU mirror returns `>= 0` everywhere.
- **U6 — SH of a constant environment.** A uniform radiance `c` environment projects to irradiance
  `c` in every direction, within 1%. This one number validates the cosine-convolution constants
  `A0/A1/A2` end to end.
- **U7 — BRDF LUT endpoints.** At `roughness = 0`, the LUT's `(scale, bias)` → `(1, 0)` within
  tolerance; the LUT is monotonic in `NdotV` at fixed roughness.
- **U8 — WGSL structure.** Every generated module parses, declares `getIndirectDiffuse` and
  `getIndirectSpecular` exactly once, and emits bindings only at group `environmentGroup`
  bindings 3–6 — never 0–2, which belong to the lights.
- **U9 — the material hook.** `buildMaterialFragment({ kind:'pbr', indirect })` emits the
  `getIndirectDiffuse` call and **not** the `vec3<f32>(0.3)` constant; without `indirect`, exactly
  the reverse. Byte-compare the no-`indirect` output against today's to prove nothing regressed.

**Integration tests (GPU, one readback each):**

- **T1 — white furnace.** A uniform radiance-1.0 environment, `metallic: 1`, `roughness` swept
  0→1, no direct lights, tonemapping off. **Expect** the surface reads ≈ 1.0 at every roughness
  (within 5%). This single test validates the prefilter, the LUT, and the split-sum recombination
  simultaneously — if any of the three is wrong, energy is lost or gained and the sweep is not flat.
- **T2 — no fireflies.** An environment that is black except a small very bright disc. **Expect**
  the variance of prefiltered mip 3 is below a threshold. Disable §6.3's solid-angle mip selection
  → the test fails, which is the point.
- **T3 — mirror fidelity.** `roughness: 0`, `metallic: 1` sphere. **Expect** the reflected image
  matches a direct cube sample of the same direction within 2% — mip 0 is not being convolved.
- **T4 — rotation.** `yaw: π/2` rotates both the diffuse SH response and the specular reflection by
  the same 90°. A common bug is rotating one and not the other; the test compares them against each
  other, not against absolutes.
- **T5 — no per-frame cost in `'once'` mode.** 100 frames after the bake. **Expect** zero compute
  dispatches from the environment system and no allocation.
- **T6 — the sky depth-tests.** A cube in front of the camera occludes the skybox exactly at its
  silhouette; the skybox writes no depth (a second sky draw is idempotent).

---

## 11. Errors

| Condition | Message |
|---|---|
| shading before the bake ran | `EnvironmentSystem: getShaderLayout() called before a source was set; call setEquirect() or setProceduralSky() first.` |
| non-power-of-two source | `EnvironmentSystem: sourceSize <n> must be a power of two ≥ 8; the mip chain and the SH projection mip assume it.` |
| `prefilterMips` exceeds the chain | `EnvironmentSystem: prefilterMips <n> exceeds log2(sourceSize)+1 = <m>.` |
| environment group ≠ light group | `EnvironmentSystem: environmentGroup <n> must equal the LightSystem's group (2); core WebGPU guarantees only 4 bind groups and all four are in use (lighting-ibl-design.md §3).` |
| binding collision with the lights | `EnvironmentSystem: binding <n> in group 2 is owned by LightSystem; environment resources start at binding 3.` |
| group 2 bound twice | `EnvironmentSystem: batch already has a group-2 bind group from LightSystem.applyToBatch(); use EnvironmentSystem.applyToBatch(batch, lights) to build the combined group instead.` |
| equirect load failed | `EnvironmentSystem: failed to load equirect "<url>"; falling back would silently ship a black environment — set a procedural sky explicitly instead.` |
| `rgba32float` requested | `EnvironmentSystem: format "rgba32float" needs the float32-filterable feature and cannot be linearly sampled on the baseline device; use rgba16float.` |
| amortized mode without a retained source | `EnvironmentSystem: updateMode "amortized" requires keepSourceCube: true; the prefilter reads the source every refresh.` |

---

## 12. Engine touchpoints

### 12.1 Core engine — **none**

Everything in Phase 1 is reachable today: compute stages, `rgba16float` / `rg16float` storage
textures, `2d-array` storage views, cube sampling views, `textureSampleLevel` in compute, and the
existing `draw(3, n)` fullscreen path for the sky. Phase 1 is the first module since v1.0.5 to need
no core change at all — the two additions shadows introduced (`depthOnly`, `cameraBuffer`) are not
required here, though Phase 2's capture path uses both.

### 12.2 Module changes — additive

1. **`LightSystem.getBindGroupEntries(): GPUBindGroupEntry[]`** — extract the array
   `applyToBatch` builds today (`LightSystem.ts:75-85`) into a public method; `applyToBatch` then
   calls it. Behavior identical, and it lets a combined group-2 bind group exist. ~5 lines.
2. **`ShaderConfig.indirect?: ShaderIndirectLayout`** and the PBR/Lambert/Toon ambient branch
   (§5.3). Optional, so every existing caller is unchanged — U9 proves it byte-for-byte.
3. **`MaterialFragmentOptions.indirect`** threaded through `buildMaterialFragment`
   (`materialFragments.ts:552-576`), and `VisibilityMaterial.indirect` alongside its existing
   `lighting` field (`visibility/types.ts:60-71`), so a GPU-culled batch can request indirect light
   the same one-liner way it requests clustered light.

### 12.3 Optional — filed, not needed

- **`ShaderConfig.tonemap?: boolean`** (default `true`). The PBR fragment tonemaps and gamma-corrects
  inline (§3), which with a real HDR environment both clips the range IBL exists to provide and
  double-applies whenever `PostChain`'s `Tonemap` runs. Turning it off is a one-line branch and is
  **strongly recommended** alongside any `PostChain` that tonemaps — but it is a pre-existing
  condition, not something IBL introduces, so it is filed rather than required.
- **A `TargetSpec` cube/array extension** for `TransientPool`. Would let the bake reuse the pool
  instead of owning its textures. The bake's textures are persistent, not transient, so the pool is
  the wrong home — filed only to record that it was considered and rejected.

---

## 13. Build order

Each step renders correctly at its end.

1. **The contract (§5)** — `ShaderIndirectLayout`, the `ShaderConfig.indirect` branch, and a stub
   implementation returning a constant. Assert U9 **and** that the no-`indirect` output is
   byte-identical to today's. Nothing looks different yet; the seam is now in place, and it is the
   seam Phase 2 plugs into.
2. **BRDF LUT (§6.4)** — scene-independent, verifiable in isolation. Assert U7.
3. **Source cube from a procedural sky (§6.1)** — no asset needed, so the whole pipeline is
   testable before any file I/O exists. Assert U1, U2.
4. **SH projection + `getIndirectDiffuse` (§7)** — Assert U3–U6. **This is the first step where the
   image changes**, and it is the one that removes the flat grey.
5. **Mip chain + GGX prefilter + `getIndirectSpecular` (§6.2–6.3)** — Assert T1, T2, T3. T1 is the
   gate: if the white-furnace sweep is not flat, the split-sum recombination is wrong and every
   later judgment about "how it looks" will be made against a broken baseline.
6. **Equirect loading (§6.1)** — real HDR input. Assert T4.
7. **Skybox pass (§8)** — Assert T6.
8. **Amortized mode (§6.5)** — animated skies, last, so a regression is isolated to it.

Steps 1–2 change no pixels. Step 4 is where the scene stops looking flat; step 5 is where it starts
looking correct.

### Deferred

To **Phase 2** ([`lighting-probes-design.md`](./lighting-probes-design.md)): spatially varying
indirect light — irradiance volumes, GPU probe capture and relighting, local parallax-corrected
reflection probes, and the volume↔global blend at volume borders.

To **v2.0**: DDGI-style per-probe visibility (the real fix for light leaking, Phase 2 §6.3);
screen-space specular occlusion; and a specular↔SSR handoff, which needs the deferred g-buffer
(roadmap §5.3, v1.1).
