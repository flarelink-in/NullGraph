# Environment — Shader & Buffer Reference

Everything the GPU side of `null-graph/environment` touches: the bind-group map, the exact byte
layout of every buffer and texture, the WGSL functions the material bridge injects, and the
conventions a custom shader has to match to interoperate.

Usage-level docs: [`environment.md`](./environment.md) ·
Architecture: [`lighting-ibl-design.md`](../lighting-ibl-design.md)

---

## 1. The bind-group map

Core WebGPU guarantees **four** bind groups (`maxBindGroups: 4`) and NullGraph spends all four.
Indirect lighting therefore lives **inside group 2 alongside the lights** — there is no fifth group
on the baseline device.

| Group | Owner | Bindings |
|---|---|---|
| 0 | engine | `0` camera uniform · `1` instance/ECS storage |
| 1 | material | `0` sampler · `1..n` textures · `n+1` params uniform |
| **2** | **lights + environment** | `0–2` lights · **`3–6` environment** |
| 3 | shadows | `0` atlas · `1` comparison sampler · `2` params · `3–4` view/slot tables |

### 1.1 Group 2 in detail

```wgsl
// LightSystem — bindings 0..2 (clustered-forward; `standard` uses 0 only)
@group(2) @binding(0) var<storage, read> globalLights: array<Light>;
@group(2) @binding(1) var<storage, read> clusterGrid:  array<Cluster>;
@group(2) @binding(2) var<storage, read> lightIndices: array<u32>;

// EnvironmentSystem — bindings 3..6
@group(2) @binding(3) var envSampler:  sampler;
@group(2) @binding(4) var envSpecular: texture_cube<f32>;
@group(2) @binding(5) var brdfLUT:     texture_2d<f32>;
@group(2) @binding(6) var<uniform> env: EnvParams;
```

Bindings **7–13 are reserved** for Phase 2 (probe volumes, reflection probes) — see
[`lighting-ibl-design.md`](../lighting-ibl-design.md) §4.4 for the full allocation and the
per-stage resource counts against core limits. A new subsystem that claims a binding without
updating that table is spending someone else's budget.

**One owner builds the group.** `EnvironmentSystem.applyToBatch(batch, lights)` concatenates
`lights.getBindGroupEntries()` with its own. Calling `lights.applyToBatch(batch)` separately
overwrites the whole group; the environment throws if it detects that.

---

## 2. Buffer & texture layouts

### 2.1 `EnvParams` — 44 floats, 176 bytes, group 2 binding 6

```wgsl
struct EnvParams {
    shR: array<vec4<f32>, 3>,   // floats  0..11   SH-L2 red,   9 coeffs packed 3-per-vec4
    shG: array<vec4<f32>, 3>,   // floats 12..23   green
    shB: array<vec4<f32>, 3>,   // floats 24..35   blue
    rotation: vec4<f32>,        // floats 36..39   .xyz reserved; .w = yaw radians
    intensity: f32,             // float  40
    specularMip: f32,           // float  41       prefilterMips - 1
    horizonFade: f32,           // float  42
    _pad: f32,                  // float  43
};
```

**This buffer has two writers, and the split matters.**

| Floats | Owner | Written by |
|---|---|---|
| `0..35` (the SH block) | **GPU** | the SH projection kernel → `copySHToParams()` |
| `36..43` (the tail) | **CPU** | `setRotation`, `setIntensity`, construction |

The CPU-side write is deliberately **partial** — offset `36*4`, length 8 floats. A full-struct write
would clobber the projected coefficients with the CPU's stale zeros every time someone nudged the
intensity, and the symptom is "the environment goes black when I move a slider."

`setRawSH()` is the one path that deliberately writes floats `0..35`, because there the CPU *is*
the source.

#### The SH packing (§7.3)

WGSL uniform arrays stride to 16 bytes per element, so `array<vec3<f32>, 9>` would cost 144 bytes
per channel. Packing 3 coefficients per `vec4` costs 48 and unpacks in three loads:

```
coefficient k of a channel lives at   channelBase + (k / 3) * 4 + (k % 3)
                                      ^^^^^^^^^^^   0 (R) · 12 (G) · 24 (B)
```

```wgsl
fn shCoeffR(k: u32) -> f32 { return env.shR[k / 3u][k % 3u]; }
```

The twelfth slot of each `vec4` is unused and written as `0`. `envMath.packSHChannel` is the CPU
mirror, and the SH projection kernel writes this **exact layout** so the transport is one copy
(§3.3).

### 2.2 The SH staging buffer — 36 floats, group 0 binding 1 of the projection kernel

```wgsl
@group(0) @binding(1) var<storage, read_write> shOut: array<f32, 36>;
```

**36, not 27.** The natural output would be 9 `vec3` interleaved, but that would need a repack
before it could reach the uniform. Writing the packed `EnvParams` layout directly makes the
transport a single `copyBufferToBuffer(shBuffer, 0, envParamsBuffer, 0, 144)`.

### 2.3 Textures

| Texture | Format | Size | Usage | Views |
|---|---|---|---|---|
| source cube | `rgba16float` | `sourceSize²` × 6, full mip chain | `TEXTURE_BINDING \| STORAGE_BINDING` | `cube` (sample) + per-mip `2d-array` (write) |
| specular cube | `rgba16float` | `2^(mips+1)`² × 6, `prefilterMips` levels | `TEXTURE_BINDING \| STORAGE_BINDING` | `cube` (sample) + per-mip `2d-array` (write) |
| BRDF LUT | `rg16float` | `lutSize²` | `TEXTURE_BINDING \| STORAGE_BINDING` | `2d` |

**Why `rgba16float` and not `rgba32float`.** `rgba16float` is a core storage format *and* core
filterable. `rgba32float` needs the `float32-filterable` feature and silently loses linear sampling
without it — which would break the roughness-mip interpolation. The config throws if you ask for it.

**Two views of one texture** is the crux of the bake: a cube view to sample from, a per-mip
`2d-array` view to `textureStore` into. Same pattern as `HiZPyramid`, one dimension up.

### 2.4 The environment sampler

```ts
{ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
  addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' }
```

`mipmapFilter: 'linear'` is load-bearing — the specular lookup interpolates **between roughness
mips**, and with `'nearest'` a roughness sweep steps visibly at each mip boundary.

### 2.5 The procedural-sky uniform — 16 floats, 64 bytes

```wgsl
struct SkyParams {
    zenith: vec3<f32>,       // floats  0..2
    sunCos: f32,             // float   3      cos(angular radius), default 0.9995
    horizon: vec3<f32>,      // floats  4..6
    sunIntensity: f32,       // float   7
    ground: vec3<f32>,       // floats  8..10
    sunDir: vec3<f32>,       // floats 12..14  TRAVEL direction, normalized
};
```

16 floats, not 13: every `vec3` aligns to 16 bytes in WGSL, so the scalars ride in the gaps and
float 11 / 15 are padding. Getting this wrong shifts every field after the first `vec3`.

### 2.6 Sky batch storage buffer — 4 floats

```wgsl
@group(0) @binding(1) var<storage, read> skyParams: array<f32>;   // [exposure, yaw, blurMip, _]
```

`createBatch` binds `batch.storageBuffer` at group 0 binding 1 unconditionally, and `layout:'auto'`
prunes unused bindings — so a shader that ignores it makes `createBindGroup` throw. The skybox
therefore puts its actual parameters there. The binding is genuinely used, not an anti-prune trick.

---

## 3. The WGSL contract

### 3.1 The two functions

`getShaderLayout()` returns `{ wgslStructs, wgslLogic }`. The logic half declares exactly these:

```wgsl
/** Cosine-convolved irradiance at worldPos with normal N, divided by PI.
 *  Multiply by albedo (and kD) for the diffuse indirect term. */
fn getIndirectDiffuse(worldPos: vec3<f32>, N: vec3<f32>) -> vec3<f32>;

/** Pre-integrated specular radiance for the reflection lobe, already including the split-sum
 *  BRDF term — the caller multiplies by nothing but occlusion. */
fn getIndirectSpecular(worldPos: vec3<f32>, N: vec3<f32>, V: vec3<f32>,
                       roughness: f32, F0: vec3<f32>) -> vec3<f32>;
```

**Both take `worldPos`, and Phase 1 ignores it.** That is deliberate: Phase 2 re-emits these same
two signatures against a probe volume, so swapping global IBL for spatially-varying probes costs
**zero material edits**. A contract that omitted the parameter would force a rewrite at exactly the
point where the code has the most users.

### 3.2 How a material consumes them

`ShaderConfig.indirect` is optional. Present, the PBR fragment's ambient becomes:

```wgsl
let F_ibl  = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
let kD_ibl = (vec3<f32>(1.0) - F_ibl) * (1.0 - metallic);
let ambient = (kD_ibl * albedo * getIndirectDiffuse(in.worldPos, N)
               + getIndirectSpecular(in.worldPos, N, V, roughness, F0)) * ao;
```

Absent, it is the original `vec3<f32>(0.3) * albedo * ao`, byte-for-byte. Lambert and Toon take the
diffuse half only.

`fresnelSchlickRoughness` is supplied by `wgslStructs` — `PBR_MATH_WGSL` has `fresnelSchlick` but
not the roughness-aware form, and the environment adds it rather than editing the shared chunk.

### 3.3 Name-collision discipline

The injected logic uses **suffixed** copies of the shared math — `shBasisL2Env`, `rotateYEnv` —
because it is composed into a material module that does *not* include `ENV_MATH_WGSL`. `PI` is
**not** redeclared; the PBR chunk set owns it.

If you write a custom fragment that composes this layout, you must supply `PI` and a `VertexOut`
with `worldPos` / `worldNormal` / `uv` / `ecsColor`.

---

## 4. Conventions a custom shader must match

### 4.1 The cube-face convention

```wgsl
fn faceDirection(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let c = uv * 2.0 - 1.0;
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

One wrong sign mirrors the environment in a way that looks entirely plausible until a recognizable
feature shows up in a reflection. `envMath.faceDirection` is the CPU mirror; test **U1** pins all
six against known directions and every corner's octant.

### 4.2 Per-texel solid angle — use the exact form

```wgsl
fn areaElement(x: f32, y: f32) -> f32 { return atan2(x * y, sqrt(x * x + y * y + 1.0)); }

// texel spans [x0,x1]×[y0,y1] in face-local [-1,1] coords
Ω = areaElement(x0,y0) - areaElement(x0,y1) - areaElement(x1,y0) + areaElement(x1,y1)
```

The tempting midpoint form `(4/size²)/(s²+t²+1)^1.5` is a *differential* approximation that
converges only as O(1/size²) — measured, it over-counts the sphere by **0.38 % at size 8**, which is
exactly the mip the SH projection reads. That error lands directly in the irradiance as lost or
gained energy. The exact form sums to `4π` at any size; **U3** asserts `1e-4`, a tolerance the
midpoint form cannot reach.

A uniform `4π/384` weight is worse still — a cube's corner texels subtend markedly less than its
centre ones, so a flat weight tilts the whole irradiance field toward the corners.

### 4.3 SH basis order

```
[ 1, y, z, x, xy, yz, 3z²−1, xz, x²−y² ]
```

with the L2 normalization constants folded in, and the **cosine-convolution constants**
(`A0 = π`, `A1 = 2π/3`, `A2 = π/4`) applied at projection time — so the stored value *is*
irradiance and a debug readout is directly interpretable.

### 4.4 Clamp the SH evaluation

```wgsl
return max(c, vec3<f32>(0.0)) * env.intensity;
```

Not cosmetic. An L2 projection of a high-contrast environment **rings**, and the negative lobe
appears as black patches on surfaces facing away from the dominant light — a bug that reads as
"shadows in the wrong place." **U5** constructs exactly that environment and asserts the raw
evaluation goes negative while the shipped one does not.

### 4.5 Rotation applies to both halves

`env.rotation.w` (yaw) is applied to the normal before the SH lookup **and** to the reflection
vector before the cube sample. Rotating one and not the other is a common bug; the skybox reads the
same yaw from its own params so all three agree.

---

## 5. The bake kernels

| Stage | Entry | Dispatch | Bindings (group 0) |
|---|---|---|---|
| A source cube | `buildSourceCubeWGSL({procedural})` | `(size/8, size/8, 6)` | procedural: `0` sky uniform, `2` dst · equirect: `0` texture, `1` sampler, `2` dst |
| B mip chain | `buildCubeMipWGSL()` | `(dim/8, dim/8, 6)` per mip | `0` src cube, `1` dst mip, `2` sampler |
| C prefilter | `buildPrefilterWGSL({level, prefilterMips, samples, srcSize})` | `(dim/8, dim/8, 6)` per mip | `0` src cube, `1` sampler, `2` dst mip |
| D BRDF LUT | `buildBRDFLUTWGSL({samples})` | `(lutSize/8, lutSize/8, 1)` | `0` dst LUT |
| SH | `buildSHProjectWGSL()` | `(1, 1, 1)` | `0` src as `2d-array` at the 8×8 mip, `1` `shOut` |

`gid.z` is the cube face throughout. All entry points are `cs_main`.

### 5.1 Firefly suppression in the prefilter

```wgsl
let saTexel  = 4.0 * PI / (6.0 * SRC_SIZE * SRC_SIZE);
let saSample = 1.0 / (f32(SAMPLE_COUNT) * pdf);
let mip = 0.5 * log2(saSample / saTexel);
color += textureSampleLevel(srcCube, srcSamp, L, mip).rgb * NdotL;
```

Sampling the source mip whose texel solid angle matches the sample's (Karis) is what stops a single
bright texel — a sun disc — surviving as a swarm of hard dots at mid roughness. Remove it and the
artifact is immediate and obvious; **T2** measures the variance rather than trusting the eye.

Mip 0 short-circuits to a straight copy: roughness 0 is a mirror, and importance-sampling it only
adds noise.

### 5.2 Why the SH projection is one workgroup

WGSL has **no `atomic<f32>`**, so "every texel atomically accumulates into nine coefficients" does
not exist. Rather than build a two-pass reduction, the projection reads the source cube's **8×8
mip** — 6 faces × 64 texels = 384 samples — which fits one 64-thread workgroup with a tree reduction
in workgroup memory. Irradiance is a very low-frequency signal, so 8×8 is ample.

---

## 6. The authoring hazard: backticks

Every WGSL generator here is a **JavaScript template literal**. A backtick inside a WGSL comment
*terminates the string*, and everything after it is parsed as TypeScript — producing a cascade of
syntax errors that point at WGSL and read as a corrupted file rather than a quoting mistake.

It cost two separate debugging sessions in this module. **U10** now asserts that no generator output
contains a backtick at all. When you add WGSL here, quote identifiers with `'single quotes'` in
comments.

---

## 7. Test map

| Test | Pins |
|---|---|
| U1 | cube-face convention, all six faces + corner octants |
| U2 | equirect ↔ direction round-trip |
| U3 | solid angles sum to 4π (catches the midpoint approximation) |
| U4 | SH packing order matches `shCoeff(k)`'s indexing |
| U5 | L2 ringing goes negative raw, clamped in the shipped path |
| U6 / U6b | constant-environment energy (validates A0/A1/A2); rotation consistency |
| U7 | BRDF LUT endpoints + monotonicity direction |
| U8 | every module parses; env claims exactly group-2 bindings 3–6 |
| U9 | the material hook is opt-in and byte-identical when absent |
| U10 | no backtick in any generated WGSL |
| §11 | config validation error messages |

`npm run test:environment`. GPU integration (T1–T6) awaits a browser harness.
