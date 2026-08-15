# NullGraph Post-Processing — Technical Design

**Package:** `null-graph/post`
**Status:** v1 design, implementation-ready.
**Companion:** [`transient-pool.md`](./transient-pool.md) — the resource allocator, specified separately.

---

## 1. Purpose

Give NullGraph a post-processing chain that matches the rest of the library's contract:
explicit sequence, pre-allocated memory, no per-frame garbage, and a data layout a Web Worker
can drive.

### Goals

- An ordered chain of effects that expands into the same flat `(ComputeStageNode | RenderPassNode)[]`
  array the engine already renders.
- One contiguous `Float32Array` holding every effect's parameters, uploaded in a single
  `writeBuffer`, and adoptable from a worker with no copy.
- Bring-your-own WGSL as a first-class path, not an escape hatch.
- v1 effects: **SSAO**, **Bloom**, **Tonemap**. Chosen because together they exercise depth
  sampling, multi-resolution pyramids, and the color-space boundary.

### Non-goals for v1

Shader fusion, TAA / temporal history, velocity buffers, LUT color grading, per-effect blend
modes. All are anticipated by the architecture; see §13.

### Design principles

1. **The chain owns targets; the user owns passes.** `PostChain` hands out attachment
   descriptors. The caller still calls `engine.createPass`.
2. **Everything expensive happens at compile.** The per-frame path is a param write, one buffer
   upload, and returning a cached array.
3. **The pass sequence is frozen after compile.** No hidden mutation. Structural changes go
   through an explicit recompile.
4. **Order is array order.** No dependency graph, no auto-resolution.

---

## 2. Architecture

```
        ┌──────────────────────────────────────────────┐
        │  PostChain            user-facing surface    │
        │  compile / sequence / upload / resize        │
        └───┬──────────┬──────────┬─────────────┬──────┘
            │          │          │             │
     ┌──────▼───┐ ┌────▼─────┐ ┌──▼─────────┐ ┌─▼──────────────┐
     │ Effect[] │ │ Transient│ │ ParamArena │ │ ShaderComposer │
     │ declare  │ │   Pool   │ │  layout    │ │  WGSL assembly │
     └──────────┘ └──────────┘ └────────────┘ └────────────────┘
                        │              │              │
        ┌───────────────▼──────────────▼──────────────▼─────────┐
        │  existing engine: createPass / createBatch /          │
        │  setRenderPassBindGroup / updateBatchData / render()  │
        └───────────────────────────────────────────────────────┘
```

Four internal modules, one public class. Each has a single responsibility and can be tested in
isolation:

| Module | Responsibility | Testable without a GPU? |
|---|---|---|
| `TransientPool` | logical targets → physical `GPUTexture` | yes, except `resolve()` |
| `ParamArena` | field list → byte offsets + WGSL struct text | yes, fully |
| `ShaderComposer` | effect body → complete WGSL module | yes, fully |
| `Effect` impls | declare passes + params | yes, fully |
| `PostChain` | orchestration, engine calls | no |

`ParamArena` and `ShaderComposer` must share one offset-computing function (§6.2). If they can
disagree, they eventually will, and the symptom is silently misread uniforms.

---

## 3. Engine integration

Verified against this repository at the time of writing.

| Fact | Source |
|---|---|
| A batch with no vertex/index buffer draws `draw(3, currentInstanceCount)` — a fullscreen triangle needs no geometry. | `core/RenderPipeline.ts:143-144` |
| A batch with `currentInstanceCount === 0` is **skipped**. `updateBatchData` must be called at least once per post batch. | `core/RenderPipeline.ts:105` |
| `createBatch` builds group 0 as `[camera@0, batch.storageBuffer@1]`, unconditionally. | `core/BatchManager.ts:94-102` |
| Pipelines use `layout: 'auto'`. A binding not *used* by the shader is pruned from the layout, and supplying an entry for it throws. | `core/BatchManager.ts:69` |
| `setRenderPassBindGroup(batch, entries, groupIndex)` accepts fully explicit entries — use this, not `createTextureBindGroup`, whose binding order is fixed. | `core/BatchManager.ts:172` |
| `RenderPassNode.colorAttachments` is public and mutable, and `createPass` stores the caller's array **by reference**. | `core/RenderPass.ts:18,24` |
| `isMainScreenPass` makes `RenderPipeline` **discard** the node's attachments and substitute the swapchain plus `ctx.depthTexture`, both `loadOp: 'clear'`. | `core/RenderPipeline.ts:72-82` |
| `expectsDepth` is true for any main-screen pass, and the pipeline depth format is hardcoded `depth24plus`. | `core/BatchManager.ts:69,48` |
| `ctx.depthTexture` has usage `RENDER_ATTACHMENT` only — it is **not samplable**. | `core/WebGPUContext.ts:126` |
| `Camera.bufferData` (108 floats): viewProj@0, view@16, proj@32, invViewProj@48, invView@64, invProj@80, eye@96, time@99, near@100, far@101, fovRads@102, aspect@103. | `core/Camera.ts:73-84` |

### 3.1 No engine changes are required

The awkward constraint is `layout: 'auto'` combined with `createBatch` always binding two
group-0 entries: a post shader that does not *use* both bindings will fail bind-group creation.

Rather than change the engine, **use both bindings for what they are good for**:

- **`@group(0) @binding(0)` — the camera uniform.** Genuinely needed: SSAO reconstructs
  view-space position from `invProj`, and linearizes depth with `near`/`far`. Every effect also
  gets `camera.time` free at float 99 via the existing `Camera.updateTime()`.
- **`@group(0) @binding(1)` — the batch storage buffer, repurposed as the per-pass header**
  (target resolution, texel size, pass index). These values change only on resize, so it is
  written **once at compile** with a single `updateBatchData(batch, header, 1)` — which also
  sets `currentInstanceCount = 1` and satisfies `RenderPipeline.ts:105`.

Result: both group-0 bindings are used, nothing is pruned, zero per-frame writes for headers,
and no engine modification.

### 3.2 Recommended (optional, additive, non-blocking)

Not required for v1, but each removes a real limitation:

1. **`RenderPassConfig.screenLoadOp` / `screenClearValue` / `screenDepth?: boolean`.** Today
   `isMainScreenPass` hardcodes a clear and force-attaches a full-resolution depth buffer
   (`RenderPipeline.ts:72-82`). Consequences: the final post pass pays a pointless full-res
   depth clear every frame, and **a HUD pass cannot draw over post output**, because a second
   main-screen pass would clear it. The README advertises exactly that composition.
2. **`NullGraph.removePass(pass)`.** `createPass` pushes into `this.passes` permanently
   (`NullGraph.ts:99`). Each `PostChain` recompile leaks 15+ nodes. `clearPasses()` is too blunt
   because it also drops the user's scene pass.
3. **Cache `ctx.depthTexture.createView()`** instead of recreating it per frame
   (`RenderPipeline.ts:79`). Small per-frame garbage in a library whose pitch is no GC spikes.

### 3.3 Unrelated landmine noticed while reading

`demos/WebGPUExperiments/AizawaCanvas/index.ts` writes `camera.bufferData[16]`, `[17]`, and
`[19]` to smuggle mouse and time. Floats 16–31 are the **view matrix** (`Camera.ts:100`). Use
`camera.updateTime()` (float 99) and the unused tail at floats 104–107 instead. Not a
post-processing issue; flagged because it will corrupt any effect that reads `camera.view`.

---

## 4. Public API

```ts
import { PostChain, Bloom, SSAO, Tonemap, CustomEffect } from 'null-graph/post';
```

### 4.1 `PostChain`

```ts
interface PostChainConfig {
  /** HDR working format for all intermediates. Default 'rgba16float'. */
  format?: GPUTextureFormat;
  /** Internal resolution multiplier vs. canvas. Default 1.0. */
  scale?: number;
  /** What the scene pass emits beyond color. Declared here, not inferred from the
   *  chain, so sceneAttachments() has no ordering dependency on setChain(). */
  gbuffer?: { depth?: GPUTextureFormat };
  /** true: the chain owns the final linear->display conversion and the last pass
   *  targets the swapchain. false: the result stays LINEAR in `outputView`. */
  outputToScreen?: boolean;
}

class PostChain {
  constructor(engine: NullGraph, config?: PostChainConfig);

  readonly format: GPUTextureFormat;
  readonly outputView: GPUTextureView;   // only when outputToScreen === false
  readonly paramFloatCount: number;

  /** Spread into engine.createPass(). Safe to call before setChain(). */
  sceneAttachments(opts?: { clearValue?: GPUColor; depthClearValue?: number }):
    Pick<RenderPassConfig, 'colorAttachments' | 'depthStencilAttachment'>;

  /** COMPILE. Not per-frame. Invalidates any previously returned sequence. */
  setChain(effects: Effect[]): void;
  compile(): void;

  /** Flat and fully expanded — Bloom appears as its individual passes. */
  getPassSequence(): RenderPassNode[];

  /** Once per frame. One writeBuffer for the whole arena. */
  upload(): void;

  /** Adopt a worker-computed arena by reference. Length must equal paramFloatCount. */
  setRawParams(data: Float32Array): void;
  describeLayout(): Record<string, { offsetFloats: number; floats: number; fields: string[] }>;

  describe(): string;
  memoryReport(): PoolReport;
  debugView(passIndex: number): void;   // -1 disables

  resize(width: number, height: number): void;
  destroy(): void;
}
```

### 4.2 Usage

```ts
const post = new PostChain(engine, {
  format: 'rgba16float',
  gbuffer: { depth: 'depth24plus' },
  outputToScreen: true,
});

const scenePass = engine.createPass({
  name: 'Scene',
  isMainScreenPass: false,
  ...post.sceneAttachments({ clearValue: { r: 0.02, g: 0.02, b: 0.04, a: 1 } }),
});

const ssao    = new SSAO({ radius: 0.6, intensity: 0.9, samples: 16 });
const bloom   = new Bloom({ threshold: 1.15, knee: 0.5, intensity: 0.7, mips: 6 });
const tonemap = new Tonemap({ curve: 'aces', exposure: 1.0 });

post.setChain([ssao, bloom, tonemap]);

// built once, never reallocated
const sequence = [lightSystem.getComputeStage(camera), scenePass, ...post.getPassSequence()];

// per frame
bloom.intensity = 0.5 + Math.sin(t) * 0.2;
post.upload();
engine.render(sequence);
```

Note `sceneAttachments()` is called before `setChain()`. That works because the g-buffer is
declared in the constructor, so the scene targets are allocated and resolved eagerly rather
than depending on chain contents.

### 4.3 The zero-copy path

```ts
const arena = new Float32Array(post.paramFloatCount);  // or a SharedArrayBuffer view
post.setRawParams(arena);   // adopted by reference, no copy
post.upload();

post.describeLayout();
// { SSAO:    { offsetFloats: 0,  floats: 64, fields: ['radius','bias','intensity','samples'] },
//   Bloom:   { offsetFloats: 64, floats: 64, fields: ['threshold','knee','intensity'] },
//   Tonemap: { offsetFloats: 128, floats: 64, fields: ['exposure','whitePoint'] } }
```

Offsets are **padded, not tightly packed** — see §6.1. A worker writing at packed offsets will
corrupt the adjacent effect.

---

## 5. Resource model

Delegated to [`transient-pool.md`](./transient-pool.md). Contract summary:

- Every intermediate, plus scene color and scene depth, is a pooled logical target.
- The final on-screen target is the sentinel `'screen'` and is **never** pool-allocated
  (`RenderPipeline` substitutes the swapchain — §3, `RenderPipeline.ts:72-82`).
- Scene depth is allocated by the pool with `RENDER_ATTACHMENT | TEXTURE_BINDING`. SSAO must
  sample **this**, not `ctx.depthTexture`, which lacks `TEXTURE_BINDING`.
- After `pool.resize()` + `resolve()`, `PostChain` must rewrite `.view` on every attachment
  object it handed out via `sceneAttachments()`. They are held by reference inside the user's
  `RenderPassNode`, so patching them in place keeps the user's scene pass and batches valid.

---

## 6. The param arena

One `GPUBuffer` (`UNIFORM | COPY_DST`) plus one `Float32Array` staging view. One slice per
**effect** — not per pass, so the worker-facing layout stays compact.

### 6.1 Slice alignment

Each effect's slice begins at a multiple of `minUniformBufferOffsetAlignment`, which is **256
bytes** on essentially all hardware. This is required because each pass's bind group binds its
effect's slice via `{ buffer, offset, size }`, and `GPUBufferBinding.offset` must satisfy that
alignment.

```
byteOffset(effect_i) = 256 * i          // v1: one 256-byte slice per effect
floatOffset(effect_i) = 64 * i
```

v1 caps a single effect's params at 256 bytes (64 floats). No v1 effect approaches this. If an
effect exceeds it, allocate ceil(size/256) consecutive slices; do not pack two effects into one.

**Static offsets, not dynamic offsets.** Offsets are known at compile time and never change, so
they go directly into the bind group entry. `RenderPipeline` calls `setBindGroup(index, group)`
with no offset array (`RenderPipeline.ts:95-101`), so dynamic offsets would require an engine
change. They are not needed.

### 6.2 Field layout inside a slice

WGSL uniform address space alignment rules:

| type | size | align |
|---|---|---|
| `f32` / `i32` / `u32` | 4 | 4 |
| `vec2f` | 8 | 8 |
| `vec3f` | 12 | **16** |
| `vec4f` | 16 | 16 |

Struct size is rounded up to a multiple of its largest member alignment.

`vec3f` aligning to 16 rather than 12 is the classic silent-corruption bug. One function must
compute offsets, and both the WGSL emitter and the JS accessor generator must call it:

```ts
function layoutFields(fields: ParamField[]): {
  offsets: number[];       // byte offset per field
  sizeBytes: number;
  wgsl: string;            // struct text with explicit _padN members
}
```

Emit explicit padding members rather than relying on implicit rules — it makes the generated
struct readable and diffable when something goes wrong.

### 6.3 Accessors

Each effect stores a `Float32Array` subarray of its slice. Accessors are plain index writes:

```ts
get intensity() { return this.slot[this.off.intensity]; }
set intensity(v: number) { this.slot[this.off.intensity] = v; }
```

No dirty flags, no change detection. `upload()` writes the whole arena unconditionally — at a
few kilobytes, one `writeBuffer` is cheaper than tracking what changed.

---

## 7. Shader composition

An effect author writes **one function**. Bindings, entry points, and the fullscreen triangle
are generated.

### 7.1 Effect-authored contract

```wgsl
fn effect(uv: vec2f) -> vec4f {
    let c = sampleInput0(uv);
    return c * params.intensity;
}
```

Available to the body (implementation note: the header variable is named
`passInfo`, not `pass` — `pass` is a WGSL reserved word and fails shader
creation):

| symbol | meaning |
|---|---|
| `params.<field>` | this effect's parameters |
| `passInfo.resolution` | `vec2f`, output target size in pixels |
| `passInfo.texelSize` | `vec2f`, `1.0 / resolution` |
| `camera.*` | full camera struct, incl. `invProj`, `near`, `far`, `time` |
| `sampleInput0(uv)` … `sampleInputN(uv)` | declared inputs, in declaration order |
| `loadInput0(px)` … | integer-coordinate `textureLoad`, unfiltered |

### 7.2 Generated preamble

```wgsl
struct Camera { /* 108-float layout, see Camera.ts */ };
@group(0) @binding(0) var<uniform> camera: Camera;

struct PassHeader { resolution: vec2f, texelSize: vec2f, passIndex: u32, _pad: vec3f };
@group(0) @binding(1) var<storage, read> passInfo: PassHeader;

struct Params { /* generated, §6.2 */ };
@group(1) @binding(0) var<uniform> params: Params;
@group(1) @binding(1) var samp: sampler;
@group(1) @binding(2) var tex0: texture_2d<f32>;
// @group(1) @binding(3) var tex1: texture_2d<f32>;   // per declared input
// depth inputs use: var texDepth: texture_depth_2d;

struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    var p = array<vec2f,3>(vec2f(-1.0,-1.0), vec2f(3.0,-1.0), vec2f(-1.0,3.0));
    var o: VSOut;
    o.pos = vec4f(p[vi], 0.0, 1.0);
    // Y FLIP: clip space is +Y up, texture space is +V down.
    o.uv  = vec2f((p[vi].x + 1.0) * 0.5, 1.0 - (p[vi].y + 1.0) * 0.5);
    return o;
}

fn sampleInput0(uv: vec2f) -> vec4f { return textureSample(tex0, samp, uv); }

// ... effect body inserted here ...

@fragment fn fs_main(in: VSOut) -> @location(0) vec4f { return effect(in.uv); }
```

**The UV Y-flip is mandatory.** Omitting it renders a vertically mirrored image that looks
plausible on symmetric content and is then extremely hard to spot. Add a test fixture with
deliberately asymmetric content.

### 7.3 Sampler policy

Two cached samplers, both `clamp-to-edge`:

- **linear** — default, used for color inputs and all bloom taps.
- **nearest** — required for depth. `depth24plus` is not filterable; sampling it with a
  filtering sampler is a validation error. `SSAO` must bind the nearest sampler and use
  `textureLoad` or a non-filtering sampler for its depth taps.

Address mode is `clamp-to-edge` everywhere. `repeat` on a bloom mip wraps the glow around the
screen edge.

### 7.4 Depth binding

Depth inputs bind as `texture_depth_2d`, not `texture_2d<f32>`, and sample via
`textureSampleLevel(texDepth, sampNearest, uv, 0.0)` returning a bare `f32`. Using the wrong
declaration is a shader-creation error, so it fails loudly — no special validation needed.

---

## 8. Compile pipeline

`setChain()` / `compile()` runs these in order. Each step is pure except the last two.

1. **Filter** — drop effects with `enabled === false`. Throw if the resulting list is empty.
2. **Declare** — call `effect.declare(ctx)` in order. The context records `TargetSpec`s and
   `PassRequest`s. Assign a global pass index to each request, in emission order.
3. **Resolve chain refs** — for each pass, map `'chain'` to the current chain target
   (`sceneColor` for the first read; thereafter the previous pass's chain output), `'scene'` to
   `sceneColor`, `'depth'` to `sceneDepth`. Throw a named error if a requested g-buffer resource
   was not declared in the constructor.
4. **Pool** — `markWrite` / `markRead` for every pass, then `pool.resolve()`. All pool
   validations run here (`transient-pool.md` §9).
5. **Arena** — lay out slices (§6.1), compute field offsets (§6.2), allocate the buffer and
   staging array, hand each effect its subarray.
6. **Compose** — generate one WGSL module per pass (§7).
7. **Instantiate** — for each pass, in order:
   - `engine.createPass({ name, isMainScreenPass: <last && outputToScreen>, colorAttachments: [...] })`
   - `engine.createBatch(pass, { shaderCode, strideFloats: <header floats>, maxInstances: 1,
     targetFormat: <pass output format>, depthWriteEnabled: false, depthCompare: 'always' })`
   - `engine.setRenderPassBindGroup(batch, [sampler, ...textures, paramsSlice], 1)`
   - `engine.updateBatchData(batch, headerFloats, 1)` — writes the static header **and** sets
     `currentInstanceCount = 1`, without which `RenderPipeline.ts:105` skips the batch.
8. **Freeze** — store the sequence array; set state to compiled.

### 8.1 Depth state on the final pass

`BatchManager.ts:69` sets `expectsDepth` for any main-screen pass, and hardcodes format
`depth24plus` (line 48), matching `ctx.depthTexture`. The final post batch **must** pass
`depthWriteEnabled: false, depthCompare: 'always'`. The defaults (`true` / `'less'`) happen to
pass because the fullscreen triangle sits at z=0 against a cleared depth of 1.0 — but relying
on that is fragile, and depth writes on a fullscreen pass are pure waste.

---

## 9. Frame execution

Everything below is the complete per-frame cost. There is no allocation and no branching on
chain structure.

```ts
// 1. user writes params — plain float stores into the staging array
bloom.intensity = 0.7;

// 2. one writeBuffer for the entire arena; advances pool history buffers
post.upload();

// 3. return the frozen array
engine.render(sequence);
```

`upload()`:

```ts
upload() {
  this.device.queue.writeBuffer(this.arenaBuffer, 0,
    this.staging.buffer, this.staging.byteOffset, this.staging.byteLength);
  this.pool.advanceFrame();
}
```

Per-pass headers are **not** rewritten — they are static between resizes (§3.1). `camera.time`
is already uploaded by the caller's existing `engine.updateCamera(camera)`.

---

## 10. Color management

The single most common way a post-processing library ships broken. Fix the rules once, here.

- Every intermediate is **linear** `rgba16float`. No sRGB intermediates, ever.
- The swapchain is `navigator.gpu.getPreferredCanvasFormat()` (`WebGPUContext.ts:87`), typically
  `bgra8unorm` — **not** an `-srgb` variant, so the hardware applies **no** gamma on write.
- Therefore the final pass **must** apply the linear→sRGB encode itself.

```wgsl
fn linearToSRGB(c: vec3f) -> vec3f {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3f(1.0/2.4)) - 0.055;
    return select(hi, lo, c <= vec3f(0.0031308));
}
```

Rules:

- `outputToScreen: true` → the chain appends the encode to the final pass's fragment output.
  Effects never do it themselves.
- `outputToScreen: false` → `outputView` is guaranteed **linear**, un-encoded. The caller owns
  the conversion.
- If the preferred canvas format ever resolves to an `-srgb` variant, the chain must **skip**
  the encode. Detect this at compile from `ctx.format.endsWith('-srgb')`; do not hardcode.

Applying the encode twice produces a washed-out image; applying it zero times produces a dark,
oversaturated one. Both are frequently mistaken for a tonemapping bug.

---

## 11. v1 effects

### 11.1 Tonemap

One pass, full resolution, `output: 'chain'` (or `'screen'` if last).

| param | type | default |
|---|---|---|
| `exposure` | f32 | 1.0 |
| `whitePoint` | f32 | 4.0 |

`curve` (`'aces' | 'reinhard' | 'agx' | 'none'`) is a **constructor-time** constant that selects
a WGSL branch — not a param. Making it a runtime uniform would force all curves into every
shader for no benefit.

### 11.2 Bloom

`2 * mips + 1` passes. With the default `mips: 6`, that is 13.

```
prefilter : chain            -> B0   (scale 0.5)   threshold/knee
down  x5  : B(i-1)           -> Bi   (scale 0.5^(i+1))
up    x5  : U(i+1), Bi       -> Ui   (additive, ascending)
composite : chain, U0        -> chain'  (full res)
```

| param | type | default |
|---|---|---|
| `threshold` | f32 | 1.15 |
| `knee` | f32 | 0.5 |
| `intensity` | f32 | 0.7 |

Notes:

- Each mip is a **separate logical target**, not a mip level of one texture. This is what lets
  the pool reason about them, and avoids mip-view plumbing.
- Downsample uses the 13-tap Jimenez filter; upsample uses the 9-tap tent. Both need
  `passInfo.texelSize` (§7.1) — which is the target's own texel size, differing per pass. This is
  precisely why the header is per-pass rather than per-effect.
- Clamp `mips` so the smallest target stays ≥ 8 px in both dimensions. At 1080p, `mips: 6`
  bottoms out at 30×16; `mips: 9` would reach 3×2 and produce visible pulsing.

### 11.3 SSAO

Three passes. The most demanding v1 effect and the reason it is in v1.

```
occlude : depth       -> AO_raw  (scale 0.5, r8unorm)
blur    : AO_raw      -> AO      (scale 0.5, r8unorm)
apply   : chain, AO   -> chain'  (full res)
```

| param | type | default |
|---|---|---|
| `radius` | f32 | 0.6 |
| `bias` | f32 | 0.02 |
| `intensity` | f32 | 0.9 |
| `samples` | u32 | 16 |

View-space reconstruction, using the camera layout confirmed in §3:

```wgsl
fn viewPosFromDepth(uv: vec2f, depth: f32) -> vec3f {
    let ndc = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let v = camera.invProj * ndc;
    return v.xyz / v.w;
}
```

Note the `(1.0 - uv.y)` — undoing the Y flip from §7.2. Getting this wrong tilts the ambient
occlusion the wrong way vertically, which reads as "the lighting looks slightly off" rather
than as an obvious bug.

Requirements and constraints:

- Declares `inputs: ['depth']`. Compile throws if `gbuffer.depth` was not configured.
- Must sample the **pool's** scene depth. `ctx.depthTexture` is `RENDER_ATTACHMENT` only
  (`WebGPUContext.ts:126`) and is not bindable.
- Depth must use the nearest sampler (§7.3).
- Normals are reconstructed from depth derivatives (`dpdx`/`dpdy` of view position) in v1. This
  produces faceted normals on curved surfaces. A real normal g-buffer target is v2 — the
  architecture already supports it via a `gbuffer.normal` entry.
- The kernel is a fixed hemisphere sample set generated at compile time and emitted as a WGSL
  `const array`, not a uniform. It never changes, so it does not belong in the arena.

---

## 12. Errors

All compile-time, all thrown with the offending effect and pass named. No silent fallbacks —
a post chain that quietly renders the wrong thing is worse than one that refuses to compile.

| Condition | Message |
|---|---|
| Effect requires `depth`, not configured | `PostChain: effect "SSAO" requires gbuffer.depth, which was not declared in the PostChain constructor.` |
| Empty chain after filtering | `PostChain: chain contains no enabled effects.` |
| Same effect instance twice | `PostChain: effect instance "Bloom" appears twice; each instance may occupy one slot.` |
| `setRawParams` length mismatch | `PostChain: setRawParams expected <n> floats, received <m>.` |
| Params exceed slice | `PostChain: effect "<x>" declares <n> bytes of params, exceeding the 256-byte slice.` |
| Sequence used before compile | `PostChain: getPassSequence() called before setChain().` |
| Pool validation | forwarded verbatim from `TransientPool` |

---

## 13. Build order

Each step ends at a state where the demo renders correctly.

1. **`ParamArena`** — pure logic, no GPU. Unit-test `vec3f` alignment and 256-byte slice
   offsets against hand-computed values.
2. **`ShaderComposer`** — pure string generation. Snapshot-test the full module for a
   two-param, one-input effect. Verify the Y flip against an asymmetric fixture.
3. **`TransientPool`, non-aliasing** — one texture per target (`transient-pool.md` §12 step 2).
4. **`PostChain` + a passthrough effect** — must render pixel-identical to no post at all.
   This is the correctness baseline everything else is diffed against.
5. **Tonemap** — proves §10. Compare against an offline-computed reference for a known HDR
   input value.
6. **Worker param streaming** — `setRawParams` from a real worker, before more effects exist.
   This is the design's actual differentiator; if it is awkward in practice, better to find out
   here than after six effects depend on the layout.
7. **Bloom** — the multi-pass, multi-resolution stress test for steps 1–4.
8. **SSAO** — depth sampling, nearest sampler, view-space reconstruction.
9. **Pool aliasing** (`transient-pool.md` §12 steps 5–7). Pixels must not change. If they do,
   the bug is in step 9, because nothing else moved.

### Deferred to v2

Fusion of adjacent single-tap effects; TAA and the `history: 2` path (the pool already supports
it); a velocity g-buffer target for motion blur (needs `buildPBRShader({ motionVectors: true })`
and a second scene color attachment, which `sceneAttachments()` is already shaped to return);
a normal g-buffer for SSAO; LUT color grading; per-effect blend modes.
