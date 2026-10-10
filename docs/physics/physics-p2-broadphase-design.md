# NullGraph Physics — P2: GPU Broadphase — Technical Design

**Target:** `src/extras/physics/` (`broadphaseMath`, `broadphaseShaders`, `GpuBroadphase`)
**Package:** `null-graph/physics` 🔌
**Status:** `plugin` — ✍ **designed, not implemented.**
**Series:** [P1 · Transform bridge](./physics-p1-transform-bridge-design.md) →
**P2 · GPU broadphase** → [P3 · GPU solver](./physics-p3-solver-design.md) · [index](./README.md)
**Depends on:** P1's body record (§4.2 there) — floats 0–3 are one `vec4` load, by design.
**Engine cost:** **none.**

---

## 1. Purpose

Make pair-finding scene-size-independent on the GPU, the way `null-graph/visibility` made culling.

### The single most important pre-existing fact

**This kernel already exists in the engine, twice, and it is not called a broadphase.**

A uniform-grid broadphase is: bin N objects into cells with atomics, write a per-cell offset and
count, compact into an index list. That is `LightSystem`'s clustered binner, line for line — same
grid, same `atomicAdd` on a global counter, same `preClearBuffers` zeroing it before the dispatch
(`LightSystem.ts:399-419`). And emitting the candidate-pair list is the same compaction the
visibility cull already does:

```wgsl
let slot = atomicAdd(&drawArgs[1], 1u);          // visibilityShaders.ts:281
```

Both patterns ship, both are tested, and neither needed a core change. P2 is those two kernels
pointed at a different buffer.

**That is the argument for P2 being cheap. It is also the argument for P2 being optional** — a
technique that reuses two existing kernels is not a technique that earns a place in the frame until
the CPU broadphase actually dominates it.

### The second fact: indirect dispatch ships, and P2 is its first real consumer

`dispatchWorkgroupsIndirect` is ✅ in the core execution model
([`architecture-tree.md`](../architecture-tree.md) §1): a `ComputeStageConfig` takes an
`indirectBuffer` holding three `u32`, and `bufferManager.createDispatchIndirectBuffer()` allocates
one with the right usage mask.

That matters here more than anywhere else in the engine. Binning dispatches over *bodies*, a
CPU-known count — the existing `dispatchCount` callback covers it, exactly as the visibility cull
does:

```ts
dispatchCount: () => [Math.ceil(this.instanceCount / 64), 1, 1],   // VisibilitySystem.ts:262
```

But **anything downstream of the pair list has no CPU-known bound at all.** The pair count is
produced on the GPU, and `maxPairs` is a capacity, not a workload — sizing a dispatch against it
is the §10 rule ("size structures by demand, not by capacity") applied to dispatches. So Stage D's
consumers take their count from a buffer Stage D wrote, and §6.2 records what that removes.

### Non-goals for P2

- **Narrowphase.** P2 emits *candidate* pairs — overlapping bounding spheres. Deciding whether two
  boxes actually touch, and where, is contact generation, and it belongs to P3 or to the CPU solver.
- **Replacing a CPU broadphase when P1 is sufficient.** §7 gives the number at which this earns its
  place. Below it, P2 is a slower broadphase with a readback attached.
- **Sorted / SAP broadphase.** Filed (§12). A uniform grid is the right first answer because its
  kernel already exists; sweep-and-prune needs a GPU sort that does not.
- **Continuous collision.** Tunneling is a narrowphase and integration concern.
- **Queries.** Raycasts share the grid but are async and are their own phase (P4, deferred).

### Design principles

1. **Reuse the kernel shape, don't reinvent it.** The light binner and the cull compaction are the
   two halves of this, and both are already pinned by tests.
2. **A dropped pair is a missed collision.** §6.1. The conservative direction here is the opposite
   of culling's, and it is the most important sentence in this document.
3. **The pair list is a GPU product.** Whether it is *read back* is the caller's decision and it
   changes the phase's value completely (§5.3).
4. **Size structures by demand.** The grid is built against the registered body count, not against
   the cap ([`design-principles.md`](../design-principles.md) §10).

---

## 2. Architecture

```
   engine.render([
      ...broadphase.getStages(),      // clear counters -> bin -> emit pairs
      ...vis.getComputeStages(),
      scenePass,
      ...post.getPassSequence(),
   ]);
```

```
   BodySet (P1) ──upload──▶ ┌──────────────────────┐
   12 floats/body           │ Stage A: bin         │  atomicAdd per cell
   floats 0..3 = vec4       │ cellCounts[]         │  preClearBuffers zeroes it
                            └──────────┬───────────┘
                                       ▼
                            ┌──────────────────────┐
                            │ Stage B: prefix sum  │  cellCounts -> cellOffsets
                            └──────────┬───────────┘
                                       ▼
                            ┌──────────────────────┐
                            │ Stage C: scatter     │  bodyIndex -> cellBodies[]
                            └──────────┬───────────┘
                                       ▼
                            ┌──────────────────────┐
                            │ Stage D: emit pairs  │  atomicAdd(&pairCount, 1)
                            │ pairs[] (u32 x2)     │  sphere-overlap test
                            └──────────┬───────────┘
                                       │
                        ┌──────────────┴──────────────┐
                        ▼                             ▼
             P3 consumes it on the GPU      CPU solver reads it back
             (no stall, §5.3)               ONE FRAME LATE (§5.3)
```

Four consecutive compute stages. `RenderPipeline` batches consecutive compute stages into one WebGPU
pass and a render pass closes it (`RenderPipeline.ts:45-48,66-70`), so the whole chain costs no
explicit barriers and slots ahead of the scene pass for free.

---

## 3. Integration facts

| Fact | Source | Consequence |
|---|---|---|
| The clustered light binner bins into a 3D grid with `atomicAdd` on a global counter, zeroed by `preClearBuffers`. | `LightSystem.ts:399-419`, `clusteredComputeShader.ts:41-66` | Stages A and C are that kernel with a different key. **No new pattern.** |
| The visibility cull compacts survivors with `atomicAdd(&drawArgs[1], 1u)`. | `visibilityShaders.ts:281`, `:359`, `:439` | Stage D is that compaction with a pair instead of an index. |
| Consecutive compute stages share one pass; a render pass closes it. | `RenderPipeline.ts:45-48,66-70` | The four stages need no explicit barrier between them. |
| `preClearBuffers` runs before its own stage and closes any open compute pass to issue the write. | `RenderPipeline.ts:30-44` | Counter zeroing is declarative and correctly ordered. **It also forces a pass split** — §6.3. |
| `dispatchCount` accepts a callback, evaluated per frame, but only from **CPU-known** values. | `types.ts:156`, `VisibilitySystem.ts:262` | §1. Binning dispatches exactly from the body count. |
| `ComputeStageConfig.indirectBuffer` takes the workgroup count from a GPU buffer; `createComputeStage` refuses one missing `INDIRECT`, or missing both `STORAGE` and `COPY_DST`. | `types.ts` (`indirectBuffer`), `BatchManager.ts` (`validateIndirectDispatch`) | §6.2. Anything keyed on the **pair** count dispatches exactly. This is what makes `maxPairs` a capacity rather than a workload. |
| `bufferManager.createDispatchIndirectBuffer(count)` allocates `INDIRECT \| STORAGE \| COPY_DST`, zeroed. | `BufferManager.ts` | A stage whose producer has not run dispatches nothing — the safe direction. |
| Meshlet bounding spheres are 64-byte records already uploaded, with `boundsCenter` at offset 0. | `visibilityShaders.ts:46-53`, `MeshletBuilder.ts:476-479` | §8 — a free one-level BVH per mesh for the midphase. |
| P1's body record puts position at floats 0–2 and world radius at float 3. | [P1](./physics-p1-transform-bridge-design.md) §4.2 | One `vec4` load per body in the binning kernel. The record was laid out for this. |
| WGSL has no `atomic<f32>`. | [`roadmap.md`](../roadmap.md) §6 | Irrelevant to P2 — every atomic here is a `u32` counter. It is P3's problem (§1 there). |

---

## 4. Data model

### 4.1 Grid parameters — 32 bytes

```wgsl
struct GridParams {
    origin: vec3<f32>,     // world-space min corner
    cellSize: f32,         // uniform; §5.1 explains why one number
    dims: vec3<u32>,       // cells per axis
    bodyCount: u32,
};
```

### 4.2 Grid buffers, sized by demand

| Buffer | Size | Notes |
|---|---|---|
| `cellCounts` | `cellTotal × 4 B` | `atomic<u32>`, zeroed by `preClearBuffers` |
| `cellOffsets` | `cellTotal × 4 B` | prefix sum of the above |
| `cellBodies` | `maxBodyCellEntries × 4 B` | a body spanning cells appears once per cell (§5.2) |
| `pairs` | `maxPairs × 8 B` | two `u32` per pair |
| `pairCount` | `4 B` | `atomic<u32>`, zeroed per frame |
| `overflow` | `4 B` | saturating count of pairs that did not fit (§6.1) |

At 64³ cells and 100,000 bodies: `cellCounts` + `cellOffsets` = 2.1 MB, `cellBodies` ≈ 1.6 MB at an
average 4 cells per body, `pairs` at `maxPairs = 1M` = 8 MB. **~12 MB**, allocated once.

### 4.3 The pair record — 8 bytes

```wgsl
struct Pair { a: u32, b: u32 };
```

Body indices into the `BodySet`, always emitted with `a < b`, which is what makes deduplication a
comparison instead of a hash (§5.2).

---

## 5. The mechanism

### 5.1 One cell size, and why

A uniform grid degenerates when object sizes span orders of magnitude: a cell sized for the small
objects means the large ones span thousands of cells; a cell sized for the large ones puts every
small object in one cell and the pair test becomes O(n²) inside it.

P2 ships **one** cell size, defaulting to `2 × meanBodyRadius`, and refuses to guess. The refusal is
a build-time error naming the ratio (§10), because the failure mode is a frame-time cliff rather
than a wrong image — the kind of thing nobody diagnoses from a screenshot.

Hierarchical grids and a broadphase that adapts are filed in §12. Shipping one level first is the
same discipline as shipping single-phase occlusion before two-phase.

### 5.2 Binning a sphere that spans cells

A body is written into **every cell its bounding sphere touches**, so the pair test only ever
compares bodies in the same cell. That produces duplicate pairs — two bodies sharing three cells
generate the pair three times.

The standard fix is a **home-cell rule**: a pair is emitted only by the cell containing the midpoint
of the two centres. One comparison, no hash set, no sorting, and it is exact.

Emitting `a < b` always is what makes it exact rather than probabilistic — without the ordering, the
same unordered pair arrives as `(a,b)` and `(b,a)` and the home-cell rule cannot tell them apart.

### 5.3 The fork: who consumes the pair list

This is the decision that determines whether P2 is worth building at all.

| Consumer | Cost | When |
|---|---|---|
| **P3, on the GPU** | none — the buffer stays in VRAM | the whole point; the pair list never touches the CPU |
| **A CPU solver, read back** | one frame of latency, no stall | acceptable for AI and gameplay, **not** for a solver that must resolve this frame |

Reading back in the frame loop is a pipeline stall, which the engine forbids everywhere else
([`design-principles.md`](../design-principles.md) §13: "readback is legal in a test and forbidden
in the frame loop"). P2 does not make an exception. A CPU solver consuming this list is working with
last frame's pairs, and for a solver that is usually wrong.

**So P2's honest value is as P3's front end**, and as a query accelerator for P4. Bolting it onto a
CPU solver is possible and mostly not advisable — stated here rather than discovered.

### 5.4 The prefix sum

`cellTotal` is 262,144 at 64³. A single-workgroup scan does not cover it, so Stage B is the standard
three-pass scan (per-block scan → scan the block sums → add back). It is ~40 lines of WGSL and it is
the only kernel in P2 with no existing twin in the engine.

That makes it the one place a bug will hide, which is why U4 pins it against a CPU prefix sum over
random inputs rather than against a rendered result.

---

## 6. Failure modes, named

### 6.1 Pair overflow — the direction is the opposite of culling's

[`design-principles.md`](../design-principles.md) §3 requires every approximation to state its safe
direction. For visibility, over-culling deletes objects and under-culling wastes lanes, so the safe
direction is "draw something invisible."

**Here it inverts.** A dropped pair is a **missed collision** — an object falls through the floor,
and the cause is a counter that saturated three subsystems away. There is no conservative direction
available inside the kernel: once `maxPairs` is reached, pairs are lost.

So the mitigation is entirely outside it:

- `overflow` is a saturating counter the caller can read. It is a number, not an exception, because
  it can only be detected per frame (§5.2 in [P1](./physics-p1-transform-bridge-design.md) makes the
  same argument).
- `maxPairs` defaults to `8 × bodyCount`, which is generous for any plausible scene, and the
  constructor refuses a value below `bodyCount` outright (§10).
- **The debug view shows it.** An overflow that only appears in a counter nobody reads is an
  overflow nobody finds.

### 6.2 The dispatch triple, and what it removes

Stage D's workload is per *cell*, so it dispatches exactly. But any consumer keyed on the **pair
count** — P3's constraint solve, a narrowphase — has no CPU-known bound.

Without indirect dispatch that consumer sizes against `maxPairs` and early-outs: at
`maxPairs = 1M` and an actual 20,000 pairs, 15,625 workgroups of which 313 do work. The idle groups
cost a buffer read and a branch, so it is bandwidth rather than compute — measurable, not fatal.

The primitive ships, so P2 does not pay it. Stage D writes its own dispatch triple alongside the
pair count:

```wgsl
// after the last pair is emitted, one invocation publishes the next stage's workgroup count
if (globalId.x == 0u) {
    let pairs = atomicLoad(&pairCount);
    dispatchArgs[0] = min((pairs + 63u) / 64u, MAX_WORKGROUPS);   // clamp: see below
    dispatchArgs[1] = 1u;
    dispatchArgs[2] = 1u;
}
```

**The clamp is not optional.** Nothing on the CPU can validate a value the GPU wrote, so keeping
each component at or below `device.limits.maxComputeWorkgroupsPerDimension` is the producing
kernel's job. Omitting it turns a pair-count overflow (§6.1) from a dropped collision into an
invalid dispatch — a worse failure, in a place nobody is looking.

The buffer needs `INDIRECT | STORAGE`; `createDispatchIndirectBuffer()` supplies both, and
`createComputeStage` refuses a buffer missing either rather than silently dropping the dispatch.

### 6.3 `preClearBuffers` splits the compute pass

`RenderPipeline` closes any open compute pass to issue a `writeBuffer` (`RenderPipeline.ts:30-44`).
P2 has two counters to zero, so a naive stage layout splits the pass twice.

Both clears belong on **Stage A only**, as two entries in one `preClearBuffers` array, so the split
happens once ahead of the chain rather than between its stages.

---

## 7. When this earns its place

A CPU uniform-grid broadphase costs roughly 0.5–1 µs per body per frame in JS, so:

| Bodies | CPU broadphase | Verdict |
|---|---|---|
| 1,000 | ~0.5–1 ms | P1 alone. P2 is a readback attached to a rounding error. |
| 10,000 | ~5–10 ms | Borderline. Worth it only if the solver is already on the GPU. |
| 100,000+ | 50 ms+ | The CPU broadphase *is* the frame. P2 is the whole point. |

These are order-of-magnitude figures, not measurements — the honest number comes from a profiler on
the target scene, and T5 is where it gets pinned. Stating the shape of the curve is enough to make
the decision; pretending to a precision we have not measured would not be.

---

## 8. The midphase, for free

Meshlet bounding spheres are already a one-level BVH over every mesh, built at import time and
uploaded (`MeshletBuilder.ts:476-479`). The cull shaders already transform those bounds by an
instance matrix (`worldSphereFromModel`, `visibilityShaders.ts:113-119`).

So a midphase — testing a moving collider against a mesh's meshlet bounds instead of its whole
bounding sphere — costs **one binding and a loop**, and gets a usable acceleration structure with no
new build step and no new buffer.

It is filed rather than shipped because it only pays off for large concave static geometry, and
naming its actual cost is what stops it being scoped as a separate project later.

---

## 9. Test suite

**Headless** (`npm run test:physics`):

| # | Asserts |
|---|---|
| U1 | Cell index ↔ world position round-trips exactly for 4,096 positions across the grid, including all six boundary faces |
| U2 | **No missed pairs.** For 512 random scenes of 200 bodies, the GPU-mirrored pair set equals a brute-force O(n²) sphere-overlap set, exactly. Not a subset — equal |
| U3 | **No duplicate pairs.** The home-cell rule emits each unordered pair exactly once, over the same 512 scenes, including bodies spanning 8 cells |
| U4 | The prefix sum matches a CPU scan over 64 random `cellCounts` arrays including the all-zero and all-max cases (§5.4) |
| U5 | Overflow saturates rather than wrapping, and `pairCount` never exceeds `maxPairs` |
| U6 | Pairs are always emitted with `a < b` (§5.2) |
| U7 | Grid buffers are sized against the **registered** body count, not `maxBodies` ([`design-principles.md`](../design-principles.md) §10) |
| U8 | Both counter clears live on one stage, so the chain splits the compute pass once (§6.3) |
| U9 | No backtick in any generated WGSL, no `PI` redeclaration, every module parses |
| U10 | Every §10 error fires and names the offender |

U2 is the test that matters and it is written the way [§13](../design-principles.md) rule 2 requires:
the brute-force reference is the *definition*, so a kernel that drops pairs fails immediately rather
than producing a plausible list.

**GPU integration:**

| # | Asserts |
|---|---|
| T1 | The GPU pair list matches the CPU mirror exactly for a 10,000-body scene, by readback |
| T2 | A body spanning 8 cells generates its pairs exactly once |
| T3 | Deliberately overflowing `maxPairs` sets `overflow` and does not corrupt `pairs` |
| T4 | The four stages occupy **one** compute pass, read from the profiler (§6.3) |
| T5 | Binning cost at 100,000 bodies is under 0.5 ms on the reference device — the number §7 argues from |

---

## 10. Errors

| Condition | Message |
|---|---|
| `cellSize <= 0` | `GpuBroadphase: cellSize <n> must be positive.` |
| Body radius / cell size ratio extreme | `GpuBroadphase: the largest body radius <r> spans <n> cells at cellSize <c>; a uniform grid degenerates past 8. Raise cellSize, or split the large bodies out (design §5.1).` |
| `maxPairs < bodyCount` | `GpuBroadphase: maxPairs <n> is below bodyCount <c>; a scene where every body touches one neighbour already overflows. A dropped pair is a missed collision (design §6.1).` |
| Grid dimensions overflow the cell buffer | `GpuBroadphase: <x>x<y>x<z> is <n> cells, exceeding maxCells <m>. Raise maxCells or lower the grid resolution.` |
| `getStages()` before bodies were registered | `GpuBroadphase: no bodies registered. Call setBodies(bodySet) before splicing the stages.` |
| `BodySet` capacity mismatch | `GpuBroadphase: bound to a BodySet of capacity <a> but constructed for <b>. The grid is sized against the body count (design §10).` |

---

## 11. Engine touchpoints

### 11.1 Required — **none**

Compute stages, `atomic<u32>`, `preClearBuffers`, and storage buffers all ship. P2 imports
`createComputeStage` and nothing else.

### 11.2 Optional — filed

- **A GPU radix sort.** Would enable sweep-and-prune and a hierarchical grid. Nothing in the engine
  needs one today, so building one for a filed alternative would be speculative.

### 11.3 Module changes — additive

| Change | File | Nature |
|---|---|---|
| `broadphaseMath.ts` — cell indexing, the CPU mirror, brute-force reference | new | pure; U2's definition |
| `broadphaseShaders.ts`, `GpuBroadphase.ts` | new | |

---

## 12. Build order

1. **`broadphaseMath` + the brute-force reference.** No GPU. U1, U2's CPU half. The reference comes
   first because it is the definition every later step is checked against.
2. **Binning + prefix sum + scatter, read back in a test.** U4, U7, T2. Nothing collides yet; the
   grid is provably correct.
3. **Pair emission with the home-cell rule.** U3, U6, T1. The pair list exists and equals the
   brute-force set.
4. **Overflow handling + the debug view.** U5, T3. The failure mode becomes visible before anything
   depends on it not happening.
5. **Stage packing + the profiler number.** U8, T4, T5 — and §7's decision becomes a measurement
   rather than an estimate.

### Deferred

- **Hierarchical / multi-level grids** for wide size distributions (§5.1).
- **Sweep-and-prune** on sorted keys — needs a GPU sort.
- **The meshlet midphase** (§8) — one binding and a loop, filed with its cost.
- **Narrowphase and contact generation** — P3 or the CPU solver, never here.
- **Sleeping-island detection.** Needs P3's velocity state to be meaningful.
