# Post-Processing API Usage

`null-graph/post` chains screen-space effects into your render sequence with an **explicit,
pre-allocated** design: no per-frame allocations, one `writeBuffer` per frame for every effect
param, and a memory layout a Web Worker can drive by reference.

`PostChain` is the façade. It orchestrates four internal systems so you never touch them directly:

| Internal system | Responsibility |
|---|---|
| **`TransientPool`** | Allocates and *aliases* intermediate render targets (a bloom mip, an AO buffer). Reuses VRAM across non-overlapping lifetimes. |
| **`ParamArena`** | One `UNIFORM` buffer holding a padded 256-byte slice per effect. All params live here; one `writeBuffer` flushes them. |
| **`ShaderComposer`** | Wraps each effect's fragment body into a complete WGSL module (camera, params struct, fullscreen-triangle VS, input samplers). |
| **`Effect`** | Owns **no GPU objects** — it's a param cursor plus a pass declaration. That's why `new Bloom()` before `new PostChain()` is legal. |

```ts
import { PostChain, Bloom, SSAO, Tonemap, CustomEffect } from 'null-graph/post';
```

> Design deep-dives live in [`post-processing-design.md`](../post-processing-design.md),
> [`transient-pool.md`](../transient-pool.md), and [`post-presets.md`](../post-presets.md). This page
> is the practical API reference.

### The effect catalog at a glance

Fifteen stock effects, plus `CustomEffect` for your own WGSL. "Depth" means the effect needs the
chain built with `gbuffer: { depth: 'depth24plus' }`; SSAO throws without it, the others degrade.
"Stage" is the effect's slot in the canonical pipeline order (see [Ordering](#ordering-rules)).

| Effect | Passes | Depth | Stage | One-liner |
|---|---|---|---|---|
| `SSAO` | 3 | **required** | pre-tonemap | Screen-space ambient occlusion. |
| `AutoExposure` | 5 | — | pre-tonemap | Auto eye-adaptation via a luminance histogram pyramid (temporal). |
| `Bokeh` | 3 | optional | pre-tonemap | Depth-of-field; uniform blur without depth. |
| `Bloom` | `2×mips` | — | pre-tonemap | Energy-conserving dual-filter glow. |
| `Halation` | 5 | — | pre-tonemap | Warm highlight bleed (film emulsion look). |
| `LensFlare` | 5 | — | pre-tonemap | Ghosts + halo from bright regions. |
| `ColorGrade` | 1 | — | pre-tonemap | Exposure/contrast/saturation/temperature/lift-gamma-gain. |
| `Tonemap` | 1 | — | **boundary** | Linear HDR → display range. |
| `CelShade` | 1 | — | post-tonemap | Toon luminance banding. |
| `Kuwahara` | 1 | — | post-tonemap | Edge-preserving painterly flattening. |
| `Outline` | 1 | optional | post-tonemap | Ink lines from depth + luma edges. |
| `FXAA` | 1 | — | post-tonemap | Fast edge anti-aliasing. |
| `ChromaticAberration` | 1 | — | post-tonemap | Radial RGB fringing. |
| `Vignette` | 1 | — | post-tonemap | Corner darkening / tint. |
| `FilmGrain` | 1 | — | post-tonemap | Animated film grain. |

Most people never touch individual effects — reach for a **preset** first:

```ts
import { PostChain, ChainPresets } from 'null-graph/post';

const post = new PostChain(engine, { gbuffer: { depth: 'depth24plus' } });
post.setChain(ChainPresets.cinematic());   // one line, correctly ordered
```

See [Presets](#presets) below and [`post-presets.md`](../post-presets.md) for all thirteen chains.

---

## Quick start

```ts
// 1. Build the chain (once)
const post = new PostChain(engine, {
    format: 'rgba16float',           // HDR working format (default)
    gbuffer: { depth: 'depth24plus' }, // needed only because SSAO reads depth
});

post.setChain([
    new SSAO({ radius: 0.6, intensity: 0.9 }),
    new Bloom({ threshold: 1.15, intensity: 0.7, mips: 6 }),
    new Tonemap({ curve: 'aces', exposure: 1.0 }),
]);
// setChain() calls compile() for you.

// 2. Render your scene INTO the chain's attachments
const scenePass = engine.createPass({
    name: 'Scene',
    isMainScreenPass: false,
    ...post.sceneAttachments({ clearValue: { r: 0, g: 0, b: 0, a: 1 } }),
});
// ... create your scene batches on scenePass as usual ...

// 3. Per-frame
function frame() {
    post.upload();                    // one writeBuffer for ALL effect params
    engine.updateCamera(camera);
    engine.render([scenePass, ...post.getPassSequence()]);
    requestAnimationFrame(frame);
}

// 4. Keep intermediates sized to the canvas
window.addEventListener('resize', () => post.resize(canvas.width, canvas.height));
```

---

## `PostChainConfig`

| Option | Type | Default | Description |
|---|---|---|---|
| `format` | `PostFormat` | `'rgba16float'` | HDR working format for every intermediate target. |
| `scale` | `number` | `1.0` | Internal resolution multiplier vs. canvas. `0.5` renders post at half res. |
| `gbuffer` | `{ depth?: GPUTextureFormat }` | — | Declares what the scene pass emits beyond color. Set `depth: 'depth24plus'` if any effect reads `'depth'` (e.g. SSAO). |
| `outputToScreen` | `boolean` | `true` | `true`: the last pass targets the swapchain and does linear→display encode. `false`: the result stays **linear** in `outputView`. |

> `gbuffer.depth` must be `'depth24plus'` — the engine's `BatchManager` hardcodes that format for any
> depth-tested pipeline, and a mismatch warns at construction. The depth here is the pool's
> *samplable* scene depth, distinct from the engine's internal `ctx.depthTexture`.

`PostFormat` is one of: `'rgba16float'`, `'rgba8unorm'`, `'rg11b10ufloat'`, `'r8unorm'`, `'r16float'`,
`'rg16float'`, `'depth24plus'`, `'depth32float'`.

---

## Lifecycle: build once vs. per frame

Everything expensive happens in `compile()` (called by `setChain()`). The per-frame path is just
param writes, one `writeBuffer`, and a cached array.

| Build-time (do once) | Per-frame (cheap) |
|---|---|
| `new PostChain(...)` | `upload()` |
| `setChain([...])` / `compile()` | reading `getPassSequence()` |
| `sceneAttachments(...)` | mutating runtime params (`bloom.intensity = …`) |
| `resize(w, h)` (on canvas resize) | `setRawParams(...)` (worker path) |

Calling `getPassSequence()`, `upload()`, `paramFloatCount`, etc. before `setChain()` throws.

### Render-sequence integration

`getPassSequence()` returns a **flat, fully-expanded** `RenderPassNode[]` — a 6-mip Bloom appears as
its ~12 individual passes. Concatenate it after your scene pass and hand the whole thing to
`engine.render()`:

```ts
engine.render([scenePass, ...post.getPassSequence()]);
```

The returned array is a single stable instance repopulated in place, so you may capture the spread
result each frame (as above) or cache `post.getPassSequence()` once and re-read it.

### Scene attachments

`sceneAttachments()` hands you the color (and, if declared, depth) attachments the chain owns, ready
to spread into `createPass`. It's safe to call **before** `setChain()` — the attachment objects are
created in the constructor and their `.view` is patched in at compile/resize, so your scene
`RenderPassNode` keeps working across resizes without recreation.

```ts
const { colorAttachments, depthStencilAttachment } = post.sceneAttachments({
    clearValue: { r: 0, g: 0, b: 0, a: 1 },
    depthClearValue: 1.0,
});
```

---

## Effects

Every effect is `enabled` by default; toggling `enabled` requires a recompile. Effects declare their
own passes and transient targets during `compile()`.

### `Bloom`

Energy-conserving dual-filter bloom: a soft-knee prefilter, a downsample pyramid, a tent-filter
upsample chain, and a composite. It emits **`2 × mips` passes**, each mip a separate pooled target.

```ts
const bloom = new Bloom({ threshold: 1.15, knee: 0.5, intensity: 0.7, mips: 6 });
```

| Option | Type | Default | Tier | Notes |
|---|---|---|---|---|
| `threshold` | `number` | `1.15` | **runtime** | Brightness cutoff feeding the glow. |
| `knee` | `number` | `0.5` | **runtime** | Soft-knee width around the threshold. |
| `intensity` | `number` | `0.7` | **runtime** | Bloom mix strength in the composite. |
| `mips` | `number` | `6` | **constructor** | Pyramid depth; clamped so the smallest mip stays ≥ 8 px/side. |

Runtime tier is exposed as live properties: `bloom.threshold`, `bloom.knee`, `bloom.intensity`.

### `Tonemap`

Maps linear HDR to display range in one full-resolution pass. The chain — not the effect — appends
the linear→sRGB encode when this is the final on-screen pass.

```ts
const tm = new Tonemap({ curve: 'aces', exposure: 1.0, whitePoint: 4.0 });
```

| Option | Type | Default | Tier | Notes |
|---|---|---|---|---|
| `curve` | `'aces' \| 'reinhard' \| 'agx' \| 'none'` | `'aces'` | **constructor** | Selects a WGSL branch; not a uniform. |
| `exposure` | `number` | `1.0` | **runtime** | Linear multiply before the curve. |
| `whitePoint` | `number` | `4.0` | **runtime** | Only meaningful for `reinhard`; ACES/AgX have built-in shoulders. |

Runtime properties: `tm.exposure`, `tm.whitePoint`.

### `SSAO`

Screen-space ambient occlusion in three passes: occlude (half-res `r8unorm`), 4×4 blur, and apply.
**Requires `gbuffer: { depth: 'depth24plus' }`** on the chain. Normals are reconstructed from depth
derivatives in v1 (faceted on curved surfaces).

```ts
const ssao = new SSAO({ radius: 0.6, bias: 0.02, intensity: 0.9, samples: 16 });
```

| Option | Type | Default | Tier | Notes |
|---|---|---|---|---|
| `radius` | `number` | `0.6` | **runtime** | View-space sampling radius. |
| `bias` | `number` | `0.02` | **runtime** | Depth bias to fight self-occlusion. |
| `intensity` | `number` | `0.9` | **runtime** | AO mix strength in the apply pass. |
| `samples` | `number` (`u32`) | `16` | **runtime** | Live sample count, clamped to the compile-time kernel size (`32`). |

All SSAO params are runtime; `ssao.samples = 24` is clamped to `[1, 32]`.

### `AutoExposure`

Eye adaptation. Meters scene luminance through a log-average pyramid, then eases a 1×1
double-buffered exposure value toward `key / avgLuminance` and multiplies the image by it. Five
passes; uses a `history: 2` target for temporal smoothing. Place **before** Tonemap.

```ts
const ae = new AutoExposure({ key: 0.18, smoothing: 0.95, minExposure: 0.1, maxExposure: 8 });
```

| Option | Default | Tier | Notes |
|---|---|---|---|
| `key` | `0.18` | runtime | Target mid-grey the average maps to. |
| `smoothing` | `0.95` | runtime | Per-frame retention of the previous exposure (frame-rate dependent by design). |
| `minExposure` / `maxExposure` | `0.1` / `8.0` | runtime | Clamp so night scenes aren't pulled to daylight. |

### `Bokeh`

Depth-of-field. Half-res circle-of-confusion from view-space distance to the focus plane, a
48-tap golden-angle disc gather, then a full-res composite (3 passes). With `gbuffer.depth` it
focuses by distance; **without depth it degrades to a uniform full-screen blur** rather than
throwing.

```ts
const dof = new Bokeh({ focusDistance: 10, focusRange: 6, maxRadius: 12, highlight: 1 });
```

| Option | Default | Tier | Notes |
|---|---|---|---|
| `focusDistance` | `10` | runtime | World-space distance to the sharp plane. |
| `focusRange` | `6` | runtime | Distance from focus at which blur maxes out. |
| `maxRadius` | `12` | runtime | Max blur radius in half-res pixels. |
| `highlight` | `1` | runtime | Extra weight for HDR highlights → visible bokeh discs. |

### `Halation`

Warm highlight bleed — bright light scattering through film emulsion. Soft-knee highlight
extract, a downsample, separable gaussian, additive tinted composite (5 passes). Place before
Tonemap (operates on linear HDR).

```ts
const hal = new Halation({ threshold: 1.0, intensity: 0.6, radius: 1.5, tint: [1, 0.3, 0.12] });
```

| Option | Default | Tier | Notes |
|---|---|---|---|
| `threshold` / `knee` | `1.0` / `0.5` | runtime | Same soft-knee shape as Bloom. |
| `intensity` | `0.6` | runtime | Glow strength. |
| `radius` | `1.5` | runtime | Blur spread multiplier. |
| `tint` | `[1, 0.3, 0.12]` | runtime¹ | Scatter color; set via `hal.setTint([r,g,b])`. |

### `LensFlare`

Screen-space pseudo flares: threshold extract, ghosts marching through screen center plus a
circular halo, blur, additive composite (5 passes). `ghosts` is constructor-only (loop bound).

```ts
const flare = new LensFlare({ threshold: 1.2, ghosts: 4, ghostDispersal: 0.3, intensity: 0.15 });
```

| Option | Default | Tier | Notes |
|---|---|---|---|
| `threshold` | `1.2` | runtime | Luminance that contributes to flares. |
| `ghostDispersal` / `haloWidth` / `chromaShift` | `0.3` / `0.25` / `0.01` | runtime | Ghost spacing, halo radius, per-sample fringing. |
| `intensity` | `0.15` | runtime | Flare strength. |
| `ghosts` | `4` | **constructor** | Ghost count (1–8). |

### `ColorGrade`

Parametric grade (no LUT — pass inputs can't bind an external texture). Order inside the pass:
exposure → white balance → log contrast → hue → saturation → lift/gamma/gain. One pass, all
runtime. Operates in linear; typically before Tonemap.

```ts
const grade = new ColorGrade({ exposure: 1, contrast: 1.1, saturation: 1.1, temperature: 0.2 });
grade.setLift([0.02, 0, 0]); grade.setGamma([1, 1, 1.1]); grade.setGain([1.05, 1, 1]);
```

| Option | Default | Notes |
|---|---|---|
| `exposure` / `contrast` / `saturation` | `1` / `1` / `1` | Scalars; contrast pivots at mid-grey. |
| `temperature` / `tint` | `0` / `0` | Warm↔cool and green↔magenta. |
| `hueShift` | `0` | Radians around the grey axis. |
| `lift` / `gamma` / `gain` | `[0,0,0]` / `[1,1,1]` / `[1,1,1]` | ASC-CDL; set via `setLift/setGamma/setGain`. |

### `CelShade`

Toon shading: quantizes luminance into flat bands while keeping hue, then boosts saturation. One
pass. Place **after** Tonemap — banding linear HDR crushes everything into the bottom band.
Pairs with `Outline`.

```ts
const cel = new CelShade({ bands: 4, softness: 0.08, saturation: 1.2 });
```

| Option | Default | Notes |
|---|---|---|
| `bands` | `4` | Band count; 3–4 comic, 5–6 anime. |
| `softness` | `0.08` | Transition width; `0` = hard posterize. |
| `saturation` | `1.2` | Post-band saturation. |

### `Outline`

Ink lines from a Sobel over linearized depth (silhouettes, distance-normalized) plus luma
(interior detail). One pass. **Without `gbuffer.depth` it falls back to luma-only edges** —
silhouettes against same-luminance backgrounds won't draw. Place after Tonemap/CelShade, before
FXAA (so lines get smoothed).

```ts
const ink = new Outline({ thickness: 1.5, depthSensitivity: 8, lumaSensitivity: 0.6, strength: 1 });
```

| Option | Default | Notes |
|---|---|---|
| `thickness` | `1.5` | Line width in pixels. |
| `depthSensitivity` / `lumaSensitivity` | `8` / `0.6` | Silhouette vs. interior-line response. |
| `strength` | `1` | Ink opacity. |
| `color` | `[0,0,0]` | Ink color; set via `ink.setColor([r,g,b])`. |

### `Kuwahara`

Edge-preserving painterly flattening: each pixel becomes the mean of whichever of four quadrant
windows has the lowest variance, so surfaces flatten into strokes while edges stay crisp. One
pass. `radius` is runtime but clamped to a compile-time max of 8.

```ts
const paint = new Kuwahara({ radius: 4 });   // paint.radius = 6 → clamped to [1, 8]
```

### `Vignette`

Radial blend toward a corner color, with the mask dithered to avoid 8-bit banding. One pass.

```ts
const vig = new Vignette({ radius: 0.75, softness: 0.45, intensity: 0.5, roundness: 1 });
vig.setColor([0, 0, 0]);   // corner tint (linear)
```

| Option | Default | Notes |
|---|---|---|
| `radius` / `softness` | `0.75` / `0.45` | Where darkening starts / falloff width. |
| `intensity` | `0.5` | Blend toward `color` at the corners. |
| `roundness` | `1` | `1` = circular, `0` = aspect-corrected ellipse. |
| `color` | `[0,0,0]` | Corner color; set via `setColor`. |

### `ChromaticAberration`

Spectral radial fringing: 12 taps along the dispersion axis weighted by a wavelength ramp, so
strong settings smear into a continuous rainbow rather than three ghosts. One pass.

```ts
const ca = new ChromaticAberration({ strength: 0.006, falloff: 2 });
```

| Option | Default | Notes |
|---|---|---|
| `strength` | `0.006` | Max channel offset at the screen edge (UV units). |
| `falloff` | `2` | Radial exponent; higher keeps the center clean. |

### `FilmGrain`

Integer-hashed, triangular-distributed grain, animated from `camera.time` (no per-frame work). One
pass. Place after Tonemap.

```ts
const grain = new FilmGrain({ intensity: 0.06, size: 1.5, colorAmount: 0.25 });
```

| Option | Default | Notes |
|---|---|---|
| `intensity` | `0.06` | Grain amplitude. |
| `size` | `1.0` | Cell size in px; `>1` = coarser stock. |
| `colorAmount` | `0.25` | `0` = mono, `1` = independent RGB. |
| `luminanceResponse` | `0.7` | Fade on bright pixels. |
| `seed` | `0.0` | Extra offset to decorrelate two chains. |

### `FXAA`

Full FXAA 3.11 quality preset (12-step edge walk). One pass. Place **after** Tonemap — its luma
estimate assumes display-referred values.

```ts
const aa = new FXAA({ edgeThreshold: 0.125, edgeThresholdMin: 0.0312, subpixel: 0.75 });
```

> **All params above are runtime unless a row says otherwise.** The only constructor-tier options in
> the whole library are `Bloom.mips`, `Tonemap.curve`, and `LensFlare.ghosts` — changing those means
> constructing a new effect and calling `setChain()` again (see the next section).

¹ `vec3f` params (`tint`, `color`, `lift`/`gamma`/`gain`) are set through a named method rather than a
scalar property, because the arena stores three floats.

---

## Presets

`null-graph/post` ships tuned option bundles so you rarely hand-pick numbers. Two tiers, named so a
recompile never surprises you mid-frame:

- **`*Presets.<name>`** — a full **constructor options** object. Pass it to the effect; full range,
  needs a rebuild.
- **`*Presets.<name>.runtime`** — the runtime-only subset, safe to `Object.assign` every frame.

```ts
import { Bloom, BloomPresets, ChainPresets, SSAOPresets, forSceneScale } from 'null-graph/post';

const bloom = new Bloom(BloomPresets.cinematic);          // build time — full preset
Object.assign(bloom, BloomPresets.dreamy.runtime);        // runtime — no recompile

post.setChain(ChainPresets.cinematic());                  // a whole correctly-ordered look
```

| Bundle | Names |
|---|---|
| `TonemapPresets` | `neutral`, `filmic`, `natural`, `soft`, `faithful` |
| `BloomPresets` | `subtle`, `natural`, `cinematic`, `dreamy`, `neon` |
| `SSAOPresets` | `performance`, `balanced`, `quality`, `contact`, `ambient` |
| `ChainPresets` | `faithful`, `neutral`, `product`, `archviz`, `cinematic`, `ethereal`, `synthwave`, `performance`, `anime`, `comic`, `painterly`, `analog`, `noir` |

Each `ChainPresets.*` is a **factory returning fresh effect instances** — call it, never store and
reuse the array (`PostChain` rejects the same instance twice, and each holds a live arena slice).

`SSAOPresets` values assume a ~1-unit-per-meter scene, because `radius`/`bias` are view-space world
units. For other conventions, scale them:

```ts
new SSAO(forSceneScale(SSAOPresets.balanced, 100));   // centimeter scene
```

Full tables, per-value rationale, and the resolution caveats live in
[`post-presets.md`](../post-presets.md).

---

## Updating parameters at runtime

Params split into two tiers:

- **Runtime** — plain properties (or a `setX()` method for `vec3f` fields). The setter writes
  **straight into the effect's arena slice** (a `Float32Array` subview). No allocation, no dirty
  flag, no bind-group rebuild. Change them every frame if you like; the next `upload()` flushes the
  whole arena in one `writeBuffer`.
- **Constructor** — only `Bloom.mips`, `Tonemap.curve`, and `LensFlare.ghosts`. These change the
  *shape* of the compiled shader/passes, so changing them means constructing a new effect and
  calling `setChain()` again. Everything else is runtime.

```ts
function frame(t) {
    bloom.intensity = 0.5 + 0.3 * Math.sin(t);  // arena write — free
    tm.exposure = exposureController.value;
    post.upload();                               // single flush for every effect
    engine.render([scenePass, ...post.getPassSequence()]);
}
```

---

## Custom effects

### Single-pass: `CustomEffect`

Bring your own WGSL as a first-class effect. Each schema field becomes a **live property** on the
instance (plus `get(name)` / `set(name, v)` for dynamic names).

```ts
const vignette = new CustomEffect({
    name: 'Vignette',
    params: [
        { name: 'strength', type: 'f32', default: 0.5 },
        { name: 'tint',     type: 'vec3f', default: [0, 0, 0] },
    ],
    inputs: ['chain'],   // default
    output: 'chain',     // default
    code: /* wgsl */ `
        fn effect(uv: vec2f) -> vec4f {
            let c = sampleInput0(uv);
            let d = distance(uv, vec2f(0.5));
            let v = 1.0 - smoothstep(0.3, 0.8, d) * params.strength;
            return vec4f(mix(params.tint, c.rgb, v), c.a);
        }
    `,
});

vignette.strength = 0.7;      // live property
vignette.set('strength', 0.7); // equivalent, dynamic name
```

`CustomEffectConfig`:

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | `string` | — | Must not collide with built-in members. |
| `code` | `string` | — | WGSL body; must define `fn effect(uv: vec2f) -> vec4f`. |
| `params` | `ParamSchema` | `[]` | Field list — becomes the `params` struct and live properties. |
| `inputs` | `ResourceRef[]` | `['chain']` | What the pass samples (see below). |
| `output` | `OutputRef` | `'chain'` | Where the pass writes. |

For **multi-pass** custom effects, subclass `PostEffectBase` and declare multiple `ctx.pass(...)`
calls in `declare()` (this is exactly how `Bloom` and `SSAO` are built).

### The shader authoring contract

You write only the fragment body. The `ShaderComposer` prepends the camera struct, the params
struct, a fullscreen-triangle vertex stage, and the input texture/sampler bindings. Available symbols
inside `fn effect(uv)`:

| Symbol | Type | Meaning |
|---|---|---|
| `params.<field>` | per schema | This effect's parameters. |
| `passInfo.resolution` | `vec2f` | Output target size in pixels. |
| `passInfo.texelSize` | `vec2f` | `1.0 / resolution`. |
| `camera.*` | struct | Full camera: `viewProj`, `view`, `proj`, `invViewProj`, `invView`, `invProj`, `eye`, `time`, `near`, `far`, `fovRads`, `aspect`. |
| `sampleInput0(uv)` … | `vec4f` (color) / `f32` (depth) | Filtered sample of the Nth declared input, in `inputs` order. |
| `loadInput0(px)` … | `vec4f` / `f32` | Integer-coordinate `textureLoad`, unfiltered. |

Notes that bite if ignored:

- **Y is flipped.** `uv` runs top-down (texture space). When reconstructing view/world position from
  depth you must undo it with `(1.0 - uv.y)` — see `SSAO.ts`'s `viewPosFromDepth`.
- **Depth inputs** bind as `texture_depth_2d`; `sampleInputN` returns a scalar `f32` and uses a
  non-filtering sampler (`depth24plus` is not filterable).
- **Never encode sRGB yourself.** The chain appends the linear→sRGB conversion on the final on-screen
  pass when the swapchain isn't `-srgb`. Effects always operate in linear space.
- The composer force-touches every declared binding, so unused inputs won't be pruned by
  `layout: 'auto'` — but only declare inputs you actually need.

### Input / output references

An effect pass reads `ResourceRef`s and writes an `OutputRef`:

| Ref | Meaning |
|---|---|
| `'chain'` | Output of the previous effect (or the scene color for the first effect). As an output, advances the chain cursor. |
| `'scene'` | The original scene color, always available. |
| `'depth'` | The scene depth. Requires `gbuffer.depth`; throws at compile if absent. |
| `TargetHandle` | An internal target you allocated via `ctx.target(spec, name)` (multi-pass effects). |

`TargetSpec` (for `ctx.target`): `{ scale?, format?, extraUsage?, history? }` — `scale` is relative to
the chain's internal size (`0.5` = half res), `history: 2` gives a never-aliased double buffer for
temporal effects.

---

## Param schema & alignment

A `ParamSchema` is an ordered `ParamField[]`; **order is the WGSL struct order**. Supported types and
their WGSL uniform alignment:

| `ParamType` | Size | Align |
|---|---|---|
| `f32` / `i32` / `u32` | 4 | 4 |
| `vec2f` | 8 | 8 |
| `vec3f` | 12 | **16** |
| `vec4f` | 16 | 16 |

`ParamArena` computes the padded layout from a single source of truth (`layoutFields`) shared by the
WGSL emitter and the JS accessors, so the struct and the buffer can never disagree — the classic
`vec3f`-aligns-to-16 corruption bug is handled for you. Each effect gets a 256-byte slice (or
`ceil(size/256)` consecutive slices if larger); two effects never share a slice.

---

## Driving from a Web Worker

The whole param set is one flat, compact `Float32Array`, so a worker can own it:

```ts
// Main thread, once: learn the layout
const floatCount = post.paramFloatCount;
const layout = post.describeLayout();
// e.g. { SSAO: { offsetFloats: 0, floats: 64, fields: ['radius','bias','intensity','samples'] },
//        Bloom: { offsetFloats: 64, floats: 64, fields: [...] }, ... }

// Worker: compute a Float32Array of length `floatCount`, write fields at their offsets, post it back.

// Main thread, per frame: adopt by reference (no copy) and flush
post.setRawParams(workerArray); // length must equal paramFloatCount
post.upload();
```

`setRawParams` adopts the array by reference and re-points every effect's accessor into it, so
subsequent property reads/writes still work. Pending constructor values were already flushed on first
bind, so adoption can't stomp worker-written data.

---

## Ordering rules

Order matters because `Tonemap` is the boundary between two color spaces. Everything **before** it
operates on linear HDR; everything **after** on the display-referred image. Put an effect on the
wrong side and it silently misbehaves — Bloom after Tonemap sees nothing above 1.0, CelShade before
Tonemap crushes into the bottom band.

The **canonical order** (exported as `EFFECT_ORDER`) is:

```
SSAO → AutoExposure → Bokeh → Bloom → Halation → LensFlare → ColorGrade
  → Tonemap →
CelShade → Kuwahara → Outline → FXAA → ChromaticAberration → Vignette → FilmGrain
```

Key adjacencies: SSAO before Bloom (occluded creases must not glow); AutoExposure before Bloom (so
the threshold sees normalized luminance); FXAA after Outline (so ink edges get anti-aliased);
FilmGrain last (so nothing smooths the grain). Full reasoning is in
[`post-presets.md`](../post-presets.md) §2.

### `orderChain()` — sort a chain for you

Don't memorize the table. Build the set in any order and let the helper place the built-ins:

```ts
import { orderChain, SSAO, Bloom, Tonemap, FilmGrain, Vignette } from 'null-graph/post';

post.setChain(orderChain([
    new FilmGrain(), new Tonemap(), new Bloom(), new Vignette(), new SSAO(),
]));
// → SSAO, Bloom, Tonemap, Vignette, FilmGrain
```

It's stable and non-mutating (returns a new array; never clones effects). **Unrecognized effects —
including every `CustomEffect` — are treated as unranked and kept, in their original relative order,
at the end.** If a custom effect belongs mid-chain, position it yourself instead of relying on the
helper. `orderChain` is a convenience, not a validator: it does not enforce the rules below.

### What the compiler actually enforces

- The chain must contain at least one enabled effect.
- No effect **instance** may appear twice (each instance is one slot; construct two if you need two).
- With `outputToScreen: true`, the **final pass must write to `'chain'`** — every stock effect's
  terminal pass does. (If you build a chain whose last effect writes to a private target, compile
  throws.)
- Any effect that reads `'depth'` (`SSAO` always; `Bokeh`/`Outline` when depth is present) requires
  the chain built with `gbuffer: { depth: 'depth24plus' }`. SSAO throws without it; Bokeh and Outline
  fall back to a no-depth path.

See [`post-presets.md`](../post-presets.md) for ready-made, correctly-ordered option bundles.

---

## Resizing, debugging & inspection

```ts
post.resize(canvas.width, canvas.height); // rebuild pooled targets; scene pass stays valid

post.describe();        // string: the pass graph, e.g. "#1 SSAO occlude: [sceneDepth] -> ao..."
post.describeLayout();  // arena offsets/fields per effect
post.memoryReport();    // TransientPool usage report (how much VRAM the pool aliased)
post.debugView(2);      // truncate after post pass index 2 and blit it to screen; -1 to disable
```

`debugView(passIndex)` is invaluable for isolating a stage — it stops the sequence after the given
pass (0-based into `getPassSequence()`) and blits that pass's output straight to the screen.

---

## `PostChain` method reference

| Member | Kind | Description |
|---|---|---|
| `new PostChain(engine, config?)` | build | Create the chain. |
| `sceneAttachments(opts?)` | build | Color/depth attachments for your scene pass. |
| `setChain(effects)` | build | Set the ordered effect list and `compile()`. |
| `compile()` | build | (Re)build pool, arena, and pass pipelines. |
| `getPassSequence()` | frame | Flat `RenderPassNode[]` to append to `engine.render()`. |
| `upload()` | frame | Flush the whole param arena (one `writeBuffer`) and advance the pool frame. |
| `setRawParams(data)` | frame | Adopt a worker-computed `Float32Array` by reference. |
| `paramFloatCount` | getter | Total float length of the arena. |
| `describeLayout()` | inspect | Per-effect `{ offsetFloats, floats, fields }`. |
| `describe()` | inspect | Human-readable pass graph. |
| `memoryReport()` | inspect | `TransientPool` usage report. |
| `debugView(passIndex)` | debug | Blit an intermediate pass to screen; `-1` disables. |
| `outputView` | getter | Final **linear** view — only when `outputToScreen: false`. |
| `resize(w, h)` | build | Reallocate intermediates for a new size. |
| `destroy()` | teardown | Free the pool and arena. |
