# TransientPool — Implementation Specification

**Target:** `src/extras/post/TransientPool.ts`
**Status:** v1 spec. Written to be implemented directly, without design decisions left open.
**Scope:** the render-target allocator only. Effect authoring, param arena layout, and WGSL
generation are specified elsewhere; this document defines the resource layer they sit on.

---

## 1. Purpose

A post-processing chain declares many intermediate render targets. Most are alive for one or
two passes and then dead. Allocating a `GPUTexture` per target wastes VRAM proportional to
chain length.

`TransientPool` accepts *logical* target declarations, computes each one's lifetime across the
pass sequence, and assigns them to a smaller set of *physical* `GPUTexture` objects such that
no two logical targets sharing a physical texture are ever alive at the same time.

**The property that matters:** with a correct pool, adding the Nth full-resolution effect to a
chain costs **zero** additional VRAM. Without it, each one costs a full-resolution target.

### Non-goals for v1

- Memory-heap aliasing. Not possible — see §2.
- Partial-resolution reuse (fitting a 960×540 target inside a 1920×1080 texture). Physical
  reuse requires an exact descriptor match.
- Mip-chain textures. Every bloom mip is its own logical target, not a mip level.
- Cross-frame aliasing beyond the explicit `history` mechanism in §8.

---

## 2. What "aliasing" means in WebGPU

**Read this before implementing.** WebGPU exposes no memory heaps, no `createPlacedResource`,
and no manual barriers. There is no way to overlap two textures in the same allocation the way
Vulkan or D3D12 FrameGraphs do.

Therefore, in this pool, aliasing means exactly one thing:

> Two logical targets are assigned **the same `GPUTexture` object**, and the pool hands out the
> same `GPUTextureView` for both.

Consequences, all of which simplify the implementation:

- Two aliased targets must have **byte-identical descriptors** (size, format, usage,
  sampleCount, dimension). A near-match is not reusable.
- **No barriers or layout transitions are needed.** WebGPU's implicit synchronisation handles
  the hazard between a pass writing a texture and a later pass sampling it.
- Contents are **not** preserved across an alias boundary. Any pass writing to a target must
  use `loadOp: 'clear'` or fully overwrite the target. A pass that uses `loadOp: 'load'` on an
  aliased target reads whatever the previous tenant left behind. See §9, validation V5.

---

## 3. Integration facts (verified against this repository)

These are load-bearing. Do not assume otherwise.

| Fact | Source |
|---|---|
| A batch with no vertex/index buffer issues `draw(3, currentInstanceCount)`. A fullscreen pass is a geometry-less batch with instance count 1. | `src/core/RenderPipeline.ts:143-144` |
| `RenderPassNode.colorAttachments` is a public mutable array of `GPURenderPassColorAttachment`. The pool may rewrite `.view` in place after the pass is created. | `src/core/RenderPass.ts:18` |
| `createPass` stores the caller's attachment array **by reference**. | `src/core/RenderPass.ts:24` |
| When `isMainScreenPass` is true, `RenderPipeline` **discards** the node's colorAttachments and depthStencilAttachment and substitutes the swapchain view plus `ctx.depthTexture`, both with `loadOp: 'clear'`. | `src/core/RenderPipeline.ts:72-82` |
| The engine renders a caller-supplied linear array: `render(pipelineSequence)`. | `src/core/NullGraph.ts:183-191` |
| `ctx.resize(w, h)` reallocates `ctx.depthTexture`. | `src/core/NullGraph.ts:73-75` |

### 3.1 The `isMainScreenPass` hazard

Because the engine overwrites attachments for the main screen pass, **the pool must never own
or be asked to allocate the target of the final on-screen pass.** The final pass in the chain
is marked `isMainScreenPass: true` and its `output` is the sentinel `'screen'`, which the pool
treats as external and does not allocate.

A second consequence: the engine unconditionally attaches `ctx.depthTexture` to that pass. The
fullscreen pipeline for the final effect must therefore declare a depth-stencil state matching
`ctx.depthTexture`'s format with `depthWriteEnabled: false` and `depthCompare: 'always'`, or
pipeline creation will fail against the pass layout. This is outside the pool's responsibility
but is the most likely place for a first-run crash.

---

## 4. Types

```ts
export type PoolFormat = GPUTextureFormat;

export interface TargetSpec {
  /** Resolution multiplier vs. the pool's base size. 0.5 = half res. Default 1.0. */
  scale?: number;
  /** Default: the pool's configured working format. */
  format?: PoolFormat;
  /** Extra usage bits ORed onto the default
   *  (RENDER_ATTACHMENT | TEXTURE_BINDING). Default 0. */
  extraUsage?: GPUTextureUsageFlags;
  /** 1 = transient, aliasable (default). 2 = double-buffered, survives one frame,
   *  never aliased. See §8. */
  history?: 1 | 2;
}

/** Opaque. Invalidated by resize() and by any new compile. Never cache the
 *  GPUTextureView obtained from one of these across a resize. */
export interface TargetHandle {
  readonly id: number;
  readonly debugName: string;
}

interface LogicalTarget {
  id: number;
  debugName: string;
  width: number;
  height: number;
  format: PoolFormat;
  usage: GPUTextureUsageFlags;
  history: 1 | 2;
  /** Lowest pass index that touches this target. -1 until markWrite/markRead. */
  firstTouch: number;
  /** Highest pass index that touches this target. -1 until markWrite/markRead. */
  lastTouch: number;
  /** True once markWrite has been called at least once. Used by validation V1. */
  written: boolean;
  /** Lowest pass index that WRITES. Used by validation V1. */
  firstWrite: number;
  /** Assigned during resolve(). */
  physicalIndex: number;
}

interface PhysicalTexture {
  key: string;
  texture: GPUTexture;
  view: GPUTextureView;
  /** Highest lastTouch among logical targets assigned so far. See §6.3. */
  maxLastTouch: number;
  byteSize: number;
}
```

---

## 5. Lifecycle

The pool is strictly two-phase. Mixing the phases is the single most likely source of bugs.

```
  ┌─ DECLARE ──────────────────────────────────────────────┐
  │  alloc()      — record intent, allocate nothing        │
  │  markWrite()  — record producer pass index             │
  │  markRead()   — record consumer pass index             │
  └────────────────────────────────────────────────────────┘
                            │
                        resolve()
                            │
  ┌─ RESOLVED ─────────────────────────────────────────────┐
  │  view(), historyView()   — legal                       │
  │  advanceFrame()          — once per frame              │
  │  report()                — legal                       │
  │  alloc/markWrite/markRead — THROW                      │
  └────────────────────────────────────────────────────────┘
                            │
                     resize() / reset()
                            │
                    back to DECLARE
```

Enforce the phase with an internal `private phase: 'declare' | 'resolved'`. Every public
method asserts its phase and throws a named error if violated. Do not make this a soft warning.

### Public surface

```ts
class TransientPool {
  constructor(device: GPUDevice, baseWidth: number, baseHeight: number,
              defaultFormat: PoolFormat);

  // DECLARE phase
  alloc(spec: TargetSpec, debugName: string): TargetHandle;
  markWrite(h: TargetHandle, passIndex: number): void;
  markRead(h: TargetHandle, passIndex: number): void;

  // Transition
  resolve(): void;

  // RESOLVED phase
  view(h: TargetHandle): GPUTextureView;
  historyView(h: TargetHandle): GPUTextureView;
  advanceFrame(): void;
  report(): PoolReport;

  // Teardown / rebuild
  resize(width: number, height: number): void;  // -> DECLARE, all handles invalid
  reset(): void;                                 // -> DECLARE, all handles invalid
  destroy(): void;
}
```

---

## 6. Algorithms

### 6.1 Dimension computation

```ts
width  = Math.max(1, Math.floor(baseWidth  * scale));
height = Math.max(1, Math.floor(baseHeight * scale));
```

Use `floor`, not `round`, and clamp to a minimum of 1. A bloom pyramid on a 1080p canvas
reaches 120×67 at 1/16 and would reach 0 in a dimension without the clamp. Floor is required
so that the same scale always yields the same size regardless of parity.

### 6.2 Lifetime intervals

For each logical target, the lifetime is the **closed interval** `[firstTouch, lastTouch]`
where a *touch* is any read **or** write.

```ts
markWrite(h, passIndex) {
  const t = this.targets[h.id];
  if (!t.written || passIndex < t.firstWrite) t.firstWrite = passIndex;
  t.written = true;
  t.firstTouch = t.firstTouch < 0 ? passIndex : Math.min(t.firstTouch, passIndex);
  t.lastTouch  = Math.max(t.lastTouch, passIndex);
}

markRead(h, passIndex) {
  const t = this.targets[h.id];
  t.firstTouch = t.firstTouch < 0 ? passIndex : Math.min(t.firstTouch, passIndex);
  t.lastTouch  = Math.max(t.lastTouch, passIndex);
}
```

Using *any touch* rather than *first write → last read* is deliberately conservative. It
handles a target written more than once (additive accumulation) without special-casing, at the
cost of occasionally missing an alias. Take the conservative version; the missed aliases are
rare and a wrong alias is a rendering corruption that is very hard to diagnose.

A target that is written but never read (a debug output, or the last chain target when the
final pass writes to the swapchain) ends up with `firstTouch === lastTouch`. That is legal.

### 6.3 Compatibility key

Two logical targets may share a physical texture only if this string is identical:

```ts
key = `${width}x${height}|${format}|${usage}|1|2d`;
```

Include `usage` in the key. Two targets of the same size and format but different usage bits
must not share, because the physical texture is created once with a fixed usage. Do **not**
"solve" this by ORing all usages together into one texture — that silently changes the
descriptor of every tenant and can push a texture into a usage combination the adapter rejects.

`history: 2` targets are excluded from the keyed pool entirely (§8).

### 6.4 Assignment — linear scan

```
sort transient targets by firstTouch ascending
   (tie-break on id ascending, so the result is deterministic)

for each target T:
    P = first physical texture with key(P) == key(T)
        and P.maxLastTouch < T.firstTouch
    if P exists:
        T.physicalIndex = P.index
        P.maxLastTouch = T.lastTouch
    else:
        create new physical texture P' with T's descriptor
        T.physicalIndex = P'.index
        P'.maxLastTouch = T.lastTouch
```

**Why this is correct.** Targets are processed in `firstTouch` order. A target is only assigned
to `P` when every target already in `P` finished strictly before `T` begins, so no two tenants
of `P` overlap. And because `P.maxLastTouch < T.firstTouch <= T.lastTouch`, the running maximum
is always `T.lastTouch` — no `Math.max` is needed on that line, though using one is harmless.

This is linear-scan register allocation. It is not optimal, but it is optimal for the
interval-graph structure a post chain produces, and it is trivially auditable.

### 6.5 THE PREDICATE — the one line most likely to be wrong

```ts
//  CORRECT
if (physical.maxLastTouch < target.firstTouch) { /* reuse */ }

//  WRONG — corrupts rendering
if (physical.maxLastTouch <= target.firstTouch) { /* reuse */ }
```

The comparison **must** be strictly less-than.

Consider bloom's upsample: pass 10 reads `B0` and writes `U0`. `B0.lastTouch === 10` and
`U0.firstTouch === 10`. With `<=`, the pool concludes they can share — and then a single pass
would sample the very texture it is rendering into. That is undefined behaviour in WebGPU, and
the validation layer will either reject the bind group or, worse, silently produce garbage.

The touch intervals of a post chain are dense and adjacent almost everywhere, so this off-by-one
does not cause a subtle regression. It causes catastrophic over-aliasing. See §10, test T2, for
the exact physical-texture count that detects it.

---

## 7. Byte accounting

`report()` must return both the achieved and the naive figure, since the naive figure is the
only way to see whether the pool is doing anything.

```ts
interface PoolReport {
  physicalTextures: number;
  logicalTargets: number;
  bytes: number;                 // sum over physical textures
  bytesWithoutAliasing: number;  // sum over logical targets
}
```

Byte size per texture: `width * height * bytesPerPixel(format)`.

```
rgba16float   8      rgba8unorm    4      rgba8unorm-srgb 4
rg16float     4      r8unorm       1      rg11b10ufloat   4
depth24plus   4      depth32float  4      r16float        2
```

`depth24plus` has an implementation-defined layout; 4 is the accepted estimate. Label the field
as an estimate in any UI that displays it.

---

## 8. History targets

A target declared `history: 2` must survive into the next frame (temporal AA, temporal
denoise, accumulation). Such targets:

- are **never** aliased, and are excluded from the linear scan entirely;
- allocate **two** physical textures, `a` and `b`;
- expose `view(h)` = the current frame's write target, `historyView(h)` = last frame's;
- swap on `advanceFrame()`.

```ts
advanceFrame() {
  assertPhase('resolved');
  for (const hb of this.historyBuffers) hb.parity ^= 1;
}
```

`view()` returns `parity === 0 ? a.view : b.view`; `historyView()` returns the other.

`historyView(h)` on a target declared `history: 1` throws. On the very first frame after
`resolve()`, `historyView()` returns a texture whose contents are undefined; effects that read
history must handle frame 0 (typically by clearing both buffers once in `resolve()`, or by
blending with a weight the effect ramps in). **`resolve()` must clear both history textures**
so that frame 0 reads zeroes rather than uninitialised VRAM — do this with a throwaway
`beginRenderPass({loadOp: 'clear', storeOp: 'store'})` per buffer.

No effect in v1 uses `history: 2`. Implement it anyway. Retrofitting cross-frame lifetimes into
an allocator built on a single-frame assumption is a rewrite of §6, whereas building it now is
about thirty lines.

---

## 9. Validation

All of these run inside `resolve()`, after assignment, before returning. They are cheap
(O(passes × inputs)) and they are the reason this pool will not produce mystery artifacts.
Do not gate them behind a debug flag in v1.

| ID | Check | Error |
|---|---|---|
| V1 | Every target has `written === true`, and `firstWrite === firstTouch`. | `PoolError: target "<name>" is read at pass <i> before it is written.` |
| V2 | Every target has `firstTouch >= 0`. | `PoolError: target "<name>" was allocated but never used.` |
| V3 | For every pass, no input's `physicalIndex` equals the output's `physicalIndex`. | `PoolError: pass <n> "<name>" would sample and render to the same physical texture (targets "<a>" and "<b>").` |
| V4 | No target has `history: 2` and a `physicalIndex` in the shared pool. | internal assertion |
| V5 | Any pass whose `loadOp` is `'load'` writes to a target with exactly one tenant. | `PoolError: pass <n> uses loadOp 'load' on aliased target "<a>"; contents are not preserved across aliasing.` |

**V3 is the most valuable check in this document.** It is the direct, independent confirmation
that §6.5 was implemented correctly. If the predicate is wrong, V3 fires immediately with the
offending pass named, instead of the user staring at a black screen. Implementing V3 requires
the pool to retain the pass→(inputs, output) mapping recorded via `markRead`/`markWrite`; keep
a parallel `passes: { index, name, inputs: number[], output: number }[]` array for this purpose.

---

## 10. Test suite

These are the acceptance criteria. Implement the pool against them directly; every number below
is exact and derived from the algorithm in §6.

Assume base size **1920×1080** and working format **`rgba16float`** (8 B/px) throughout.

Target byte sizes at this base:

| scale | dims | rgba16float | r8unorm | depth24plus |
|---|---|---|---|---|
| 1.0 | 1920×1080 | 16.59 MB | — | 8.29 MB |
| 0.5 | 960×540 | 4.15 MB | 0.52 MB | — |
| 0.25 | 480×270 | 1.04 MB | — | — |
| 0.125 | 240×135 | 0.26 MB | — | — |
| 0.0625 | 120×67 | 0.064 MB | — | — |

---

### T1 — Basic disjoint reuse

Two full-res `rgba16float` targets, `A` touched at passes [0,3], `B` at [5,6].

**Expect:** `physicalTextures === 1`, `logicalTargets === 2`.

---

### T2 — Adjacency must NOT alias (catches the `<=` bug)

Two full-res targets, `A` touched at [0,3], `B` at [3,6]. Pass 3 reads A and writes B.

**Expect:** `physicalTextures === 2`.

If this returns 1, §6.5 is implemented with `<=`. Fix it before continuing; every later test
will produce plausible-looking but wrong numbers.

---

### T3 — Descriptor mismatch blocks reuse

`A` at [0,1] `rgba16float`, `B` at [3,4] `rgba8unorm`, same dimensions.

**Expect:** `physicalTextures === 2`. Repeat with matching format but differing `extraUsage`;
also expect 2.

---

### T4 — History exclusion

`A` at [0,1] `history: 2`, `B` at [3,4] `history: 1`, otherwise identical descriptors.

**Expect:** three physical textures total (two for A's double buffer, one for B). `A` must not
share with `B` despite disjoint lifetimes. `historyView(B)` throws. After one `advanceFrame()`,
`view(A)` returns the texture that `historyView(A)` returned before the call.

---

### T5 — Read before write

`A` marked read at pass 2, marked write at pass 4.

**Expect:** `resolve()` throws V1 naming target `A`.

---

### T6 — SSAO + Bloom (the v1 integration case)

Chain: `SSAO → Bloom(4 mips) → Tonemap`, `outputToScreen: true`.

Full expansion, with the scene pass as index 0:

| # | Pass | reads | writes | target spec |
|---|---|---|---|---|
| 0 | Scene | — | `sceneColor`, `sceneDepth` | 1.0 rgba16float / 1.0 depth24plus |
| 1 | SSAO occlude | `sceneDepth` | `AO_raw` | 0.5 r8unorm |
| 2 | SSAO blur | `AO_raw` | `AO` | 0.5 r8unorm |
| 3 | SSAO apply | `sceneColor`, `AO` | `chain_1` | 1.0 rgba16float |
| 4 | Bloom prefilter | `chain_1` | `B0` | 0.5 |
| 5 | Bloom down | `B0` | `B1` | 0.25 |
| 6 | Bloom down | `B1` | `B2` | 0.125 |
| 7 | Bloom down | `B2` | `B3` | 0.0625 |
| 8 | Bloom up | `B3`, `B2` | `U2` | 0.125 |
| 9 | Bloom up | `U2`, `B1` | `U1` | 0.25 |
| 10 | Bloom up | `U1`, `B0` | `U0` | 0.5 |
| 11 | Bloom composite | `chain_1`, `U0` | `chain_2` | 1.0 |
| 12 | Tonemap | `chain_2` | `'screen'` | external, not allocated |

Resulting intervals:

```
pass:         0   1   2   3   4   5   6   7   8   9  10  11  12
sceneColor    W───────────R
sceneDepth    W───R
AO_raw            W───R
AO                    W───R
chain_1                   W───R───────────────────────R
B0                            W───────────────────R
B1                                W───────────R
B2                                    W───R
B3                                        W───R
U2                                            W───R
U1                                                W───R
U0                                                    W───R
chain_2                                                   W───R
```

**Expect exactly:**

```
logicalTargets       13
physicalTextures     12
bytesWithoutAliasing 70.07 MB
bytes                53.48 MB
```

The **only** alias pair is `sceneColor` ↔ `chain_2` (intervals [0,3] and [11,12]). Every other
adjacency touches and must stay separate. Assert the alias pair explicitly, not just the count.

**Diagnostic:** if `physicalTextures === 7`, the predicate in §6.5 is `<=`. That value comes
from every group collapsing to a single texture, and is the signature of this exact bug.

---

### T7 — Ping-pong emergence

Extend T6 by splitting the tail into three separate full-res passes — Tonemap → chain_3,
ChromaticAberration → chain_4, Vignette → `'screen'` (passes 12, 13, 14).

Full-res `rgba16float` group now holds five logical targets:
`sceneColor[0,3]`, `chain_1[3,11]`, `chain_2[11,12]`, `chain_3[12,13]`, `chain_4[13,14]`.

**Expect:** that group resolves to **2** physical textures.

This is the test that proves the pool's actual value. Classic A/B ping-ponging is not special-
cased anywhere in §6 — it is simply what linear-scan produces on a chain of full-res effects.
Adding a sixth or twentieth full-res effect keeps this number at 2.

---

### T8 — Resize invalidation

After `resolve()`, call `resize(1280, 720)`.

**Expect:** phase returns to `declare`; `view(h)` on any pre-resize handle throws; previously
created `GPUTexture` objects have had `.destroy()` called; `report()` throws until the next
`resolve()`.

---

## 11. Resize contract

```ts
resize(width, height) {
  for (const p of this.physical) p.texture.destroy();
  for (const hb of this.historyBuffers) { hb.a.destroy(); hb.b.destroy(); }
  this.physical = [];
  this.historyBuffers = [];
  this.targets = [];
  this.passes = [];
  this.nextId = 0;
  this.baseWidth = width;
  this.baseHeight = height;
  this.phase = 'declare';
}
```

The pool does **not** attempt to preserve handles across a resize. The owning `PostChain` must
re-run the declare phase — which it does anyway, because target dimensions change.

Because `RenderPassNode.colorAttachments` is mutable and stored by reference
(`src/core/RenderPass.ts:24`), `PostChain` can patch `.view` on the user's existing scene pass
after the new `resolve()`, so the user does not have to recreate their scene pass or their
batches. That is the reason scene color and depth are pooled rather than pinned. Document this
requirement wherever `PostChain.resize()` is implemented: **after `pool.resize()` and the
subsequent `resolve()`, every attachment object handed out by `sceneAttachments()` must have
its `.view` rewritten.** Forgetting this leaves the scene pass pointing at destroyed textures,
which surfaces as a validation error on the next `render()`.

---

## 12. Implementation order

1. Types, phase enum, `alloc` / `markWrite` / `markRead` recording. No GPU calls.
2. `resolve()` with **no aliasing** — one physical texture per logical target. Ship this.
   The chain renders correctly and `bytes === bytesWithoutAliasing`.
3. Validations V1, V2, V5.
4. T1–T5 against the non-aliasing implementation (T1 will report 2, not 1 — that is expected
   at this stage; assert `bytes === bytesWithoutAliasing` instead).
5. Add §6.3 keying and §6.4 linear scan. T1 now reports 1.
6. Add V3. Then T6 and T7.
7. Add §8 history support and T4.

Step 2 is a complete, correct, shippable pool. Everything after it is a pure memory
optimisation that changes no observable rendering. If a rendering bug appears after step 5,
it is in step 5 — because nothing else moved.
