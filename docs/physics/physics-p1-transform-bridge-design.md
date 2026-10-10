# NullGraph Physics — P1: The Transform Bridge — Technical Design

**Target:** `src/extras/physics/` (`bodyMath`, `TransformBridge`, `SolverAdapter`)
**Package:** `null-graph/physics` 🔌
**Status:** `plugin` — ✍ **designed, not implemented.**
**Series:** **P1 · Transform bridge** → [P2 · GPU broadphase](./physics-p2-broadphase-design.md) →
[P3 · GPU solver](./physics-p3-solver-design.md) · [index](./README.md)
**Depends on:** nothing. The zero-copy instance contract is already shipped (§3).
**Engine cost:** **none required.** One optional additive constructor parameter (§11.2).

---

## 1. Purpose

A solver — wherever it runs, whoever wrote it — drives the renderer's instance transforms with no
copies and no judder.

### The single most important pre-existing fact

**The zero-copy half of this is already done, and it is done on the read side.**

```ts
public setInstances(data: Float32Array, count: number): void {
    // ...
    this.instanceStaging = data;          // VisibilitySystem.ts:183 — adopted, never copied
    this.instanceCount = count;
}
```

`VisibilitySystem` holds the caller's array by reference and copies nothing until `upload()`, which
is one `writeBuffer` per frame straight from the caller's memory (`VisibilitySystem.ts:188-195`).

So a solver writing model matrices into that array **is already a complete integration**. There is
no bridge required to make physics work at all. What this document specifies is the part that makes
it *feel* right, and that part is one thing: interpolation.

### The second fact: fixed step vs. variable frame

Physics is integrated at a fixed timestep. Rendering is not. At a 60 Hz step on a 144 Hz display,
most frames show a transform that has not moved since the last one and some show two steps' worth —
so a body moving at constant velocity **visibly stutters**, and it stutters worse the smoother the
display. This is not a solver quality problem and no amount of substepping fixes it.

The bridge keeps two transform sets and interpolates between them by the accumulator's remainder.
**That interpolation is the only per-frame work the bridge does**, and it is why the bridge exists.

### The third fact: bounds are object-space, so they are not per-frame work

```wgsl
fn worldSphereFromModel(model: mat4x4<f32>, boundsCenter: vec4<f32>) -> vec4<f32> {
    let c = (model * vec4<f32>(boundsCenter.xyz, 1.0)).xyz;
    let sx = length(model[0].xyz);
    let sy = length(model[1].xyz);
    let sz = length(model[2].xyz);
    return vec4<f32>(c, boundsCenter.w * max(sx, max(sy, sz)));   // visibilityShaders.ts:113-119
}
```

The cull transforms the bounds itself, from the same model matrix, scaling the radius by the
**maximum** axis scale. Floats 16–19 of the instance record are therefore **object-space and
constant for a rigid body** — the bridge writes them once at spawn and never touches them again.

This corrects a reasonable assumption ([`roadmap-modules.md`](../roadmap-modules.md) §11.2 reads as
though bounds need per-frame syncing) and it *reduces* P1's scope: the per-frame path is
interpolation and nothing else.

What still matters is that the radius **covers the collider**, and the safe direction is to
**loosen** — an under-sized sphere makes the object vanish when it rotates, which reads as a
rendering bug and is diagnosed nowhere near physics.

### Non-goals for P1

- **Shipping a solver.** This is an adapter. Rapier and Jolt stay third-party, and the surface is
  small enough to write against both.
- **Colliders as engine types.** The bridge never learns what shape a body is. A collider lives in
  the solver; the bridge sees a position, an orientation, and a radius.
- **`SharedArrayBuffer` as a requirement.** §7. It is one transport of three and the other two need
  no deployment change.
- **Anything on the GPU.** P1 is pure CPU math. P2 is where compute enters.
- **Queries, character control, contacts, debug draw.** Deferred, with reasons, in
  [the index](./README.md).

### Design principles

1. **The renderer adopts, never copies — and the bridge must not break that.** The bridge writes
   *into* the caller's instance array. It does not own a second one.
2. **Interpolate, don't snap.** §5. The alternative is a stutter that gets worse on better hardware.
3. **The adapter is the whole API.** One method a solver must provide (§8). Anything larger is the
   plugin absorbing responsibilities that belong to the solver.
4. **Design the body record once.** P2's broadphase reads the same buffer (§4.2), so the record is
   laid out as three `vec4`s from the start rather than being repacked when the GPU wants it.

---

## 2. Architecture

```
   // --- fixed-step, wherever the solver runs ---
   bridge.advance(frameDeltaSeconds, (step) => {
       solver.step(step);                  // Rapier / Jolt / yours
       adapter.readBodies(bridge.next);    // solver -> the NEXT transform set
   });

   // --- once per rendered frame ---
   bridge.interpolate(instances);          // prev, next, alpha -> instance floats 0..15
   vis.setInstances(instances.data, instances.count);
   vis.upload();

   engine.render([
      ...vis.getComputeStages(),
      scenePass,
      ...post.getPassSequence(),
   ]);
```

```
   ┌──────────────┐   readBodies()   ┌───────────────────────────┐
   │ Solver       │ ───────────────▶ │ TransformBridge           │
   │ (worker/WASM)│                  │  prev  : BodySet (12f/body)│
   └──────────────┘                  │  next  : BodySet           │
                                     │  alpha : accumulator/step  │
                                     └────────────┬──────────────┘
                                                  │ interpolate()
                                                  ▼
                            ┌────────────────────────────────────────┐
                            │ InstanceBuffer (24 floats, caller-owned)│
                            │  0..15 model matrix   ← written per frame│
                            │  16..19 object bounds ← written at spawn │
                            │  20..23 mesh/LOD/flags← written at spawn │
                            └────────────────────┬───────────────────┘
                                                 │ setInstances (by reference)
                                                 ▼
                                          VisibilitySystem
```

The bridge touches floats 0–15 and nothing else. Everything below 16 belongs to whoever spawned the
instance, and the bridge overwriting it would be the plugin reaching into the renderer's data.

---

## 3. Integration facts

| Fact | Source | Consequence |
|---|---|---|
| `setInstances` stores the caller's `Float32Array` by reference and copies nothing. | `VisibilitySystem.ts:179-185` | The bridge writes into the caller's array and the renderer sees it. **Zero-copy needs no new mechanism.** |
| `upload()` is one `writeBuffer` from `instanceStaging.buffer` at its `byteOffset`. | `VisibilitySystem.ts:188-195` | A subarray view works. The bridge can drive a *slice* of a larger scene array. |
| The instance record is **24 floats / 96 bytes**: model 0–15, bounds 16–19, meshIndex 20, lodOverride 21, flags 22, pad 23. | `cullMath.ts:21-32` | §4.1. The bridge's write target is exactly floats 0–15. |
| Bounds are **object-space**; the cull transforms them and scales the radius by `max(len(col0), len(col1), len(col2))`. | `visibilityShaders.ts:113-119` | §1. Bounds are a spawn-time write, not per-frame. Conservative direction: loosen. |
| `InstanceBuffer` allocates its own `ArrayBuffer` in its constructor. | `cullMath.ts:55-59` | **The only gap.** A solver cannot supply the backing store. §11.2 — one optional parameter. |
| Matrices are **column-major** (gl-matrix / WGSL); element (col c, row r) is `m[c*4+r]`. | `cullMath.ts:7-8` | §5.3. Composing the matrix in the wrong order produces a plausible transform that is wrong under non-uniform scale. |
| Nothing in `src/` references `SharedArrayBuffer`. | (repo-wide) | §7. SAB is a transport choice, never an engine dependency. |
| `dev-server.js` sets COOP/COEP unless `--no-sab`. | `dev-server.js:9-18` | The SAB path is *testable* — a harness capability, not a requirement. |

---

## 4. Data model

### 4.1 What the bridge writes — instance floats 0–15

Column-major `mat4`, composed from the interpolated position, orientation and scale (§5.3). Floats
16–23 are untouched.

### 4.2 The body record — 12 floats / 48 bytes

```
float  0..2   position           (world)
float  3      worldBoundsRadius  (world; = objectRadius * max scale component)
float  4..7   orientation        (quaternion x, y, z, w)
float  8..10  scale
float  11     flags              (u32 bits, written as raw bits)
```

**Three `vec4`s, 16-byte aligned, and that is deliberate.** P2's broadphase bins bodies by world
position and radius — floats 0–3, exactly one `vec4` load. Laying the record out for a GPU that P1
does not use costs nothing now and saves a repack later; it is the same discipline that made
`getIndirectDiffuse` take a `worldPos` it ignored ([`design-principles.md`](../design-principles.md) §4).

`worldBoundsRadius` is stored rather than derived because P2 needs it per body and the max-scale
reduction is three `length()` calls the broadphase should not repeat. It is a **CPU mirror of
`worldSphereFromModel`'s radius term** and U7 pins it as such.

| Flag bit | Meaning |
|---|---|
| 0 | asleep — the solver reports no motion; the bridge may skip interpolation (§5.4) |
| 1 | static / kinematic — never interpolated; `prev` and `next` are the same by construction |

### 4.3 `BodySet` — two of them, allocated once

```ts
export class BodySet {
    readonly data: Float32Array;   // capacity * 12
    count: number;
}
```

The bridge holds `prev` and `next` and **swaps references** on each solver step — it never copies
one into the other. A copy is `capacity * 48` bytes per step and buys nothing.

---

## 5. The mechanism

### 5.1 The accumulator

```
accumulator += min(frameDelta, maxFrameDelta);
steps = 0;
while (accumulator >= step && steps < maxSubsteps) {
    swap(prev, next);
    solver.step(step);
    adapter.readBodies(next);
    accumulator -= step;
    steps++;
}
if (steps === maxSubsteps) { accumulator = 0; clampedFrames++; }   // §5.2
alpha = accumulator / step;
```

`maxFrameDelta` (default 0.25 s) is the tab-switch guard: without it, returning to a backgrounded
tab hands the loop a 30-second delta and the solver tries to simulate 1,800 steps in one frame.

### 5.2 The spiral of death, and why it is not a thrown error

If the solver cannot keep up, the accumulator grows every frame, which makes the next frame do more
work, which makes it grow faster. The fix is to cap substeps and **discard the excess time** —
`accumulator = 0`, not `accumulator -= steps * step`. Keeping the debt is what makes it a spiral.

Simulation time then runs slow, which is the correct failure: the alternative is a frame that never
returns.

This condition can only be detected per frame, and
[`design-principles.md`](../design-principles.md) §2 is explicit that throwing at 60 Hz is not
failing loud, it is failing continuously. So the bridge **counts** it — `bridge.clampedFrames` — and
the caller reads it. A number that only ever goes up is a diagnosis; an exception in the frame loop
is a crash.

### 5.3 Interpolation

Position and scale lerp. Orientation **nlerps with a shortest-arc sign fix**:

```ts
let d = qa.x*qb.x + qa.y*qb.y + qa.z*qb.z + qa.w*qb.w;
const s = d < 0 ? -1 : 1;      // ← the whole of §5.3.1
out.x = qa.x + (s*qb.x - qa.x) * alpha;
// ... y, z, w, then normalize
```

Then compose column-major, rotation scaled per column:

```
col0 = R[0] * scale.x    col1 = R[1] * scale.y    col2 = R[2] * scale.z    col3 = position
```

Scaling the *rows* instead produces a matrix that is correct for uniform scale and silently wrong
for everything else — and `worldSphereFromModel` reads column lengths, so the bounds go wrong with
it, in the same direction, which hides the error twice.

#### 5.3.1 The sign fix is not optional

`q` and `−q` are the same rotation. A solver is free to return either, and quaternion integrators
routinely flip hemispheres mid-motion. Without the sign fix the interpolation takes the **long way
around** — the body spins nearly 360° over one frame, once, at an unpredictable moment.

It reads as a physics glitch, it is not reproducible from the solver state, and it survives every
test that interpolates two nearby rotations. U2 constructs the opposite-hemisphere case
deliberately and asserts the un-fixed form fails ([`design-principles.md`](../design-principles.md)
§13, rule 1).

### 5.4 Sleeping bodies

A body flagged asleep has `prev == next` by definition, so interpolation is a no-op that still costs
a quaternion normalize and 16 float writes. Skipping it is worth it at scale — but the skip must be
**per body and unconditional**, not "skip if the transforms compare equal", because a float compare
over 12 floats costs more than the write it avoids.

The correctness direction: waking is the solver's job. If the bridge skipped a body the solver had
actually moved, the object would freeze in place — so the flag is read, never inferred.

### 5.5 Zero allocation

Every method takes an optional trailing `out` and writes into it; module-scope scratch holds the
quaternion and matrix temporaries ([`design-principles.md`](../design-principles.md) §5). None of
that scratch escapes into a return value (§6 there) — `interpolate` writes into the caller's
instance array and returns nothing.

Measured by counting typed-array constructor calls, not by heap deltas: U5.

---

## 6. Spawn and lifecycle

The bridge owns transforms. It does not own instances, and the split is worth stating because it is
where a plugin usually overreaches.

| Operation | Who | What happens |
|---|---|---|
| Create a renderable | caller | `instances.write(i, model, bounds, meshIndex)` — floats 16–23 fixed here |
| Register it with physics | caller | `bridge.link(instanceIndex, bodyIndex)` |
| Move it | solver → bridge | floats 0–15, every frame |
| Destroy it | caller | `bridge.unlink(instanceIndex)`, then the caller's own instance recycling |

`link` is an index pair, not an object. A body and an instance are both slots in flat arrays, and
the mapping is a `Uint32Array` — the same shape as everything else here.

**Bodies without instances are legal** (a trigger volume, a compound child) and so are instances
without bodies (static scenery). The mapping is sparse in both directions and the bridge iterates
the links, not the bodies.

---

## 7. Transport is the caller's choice

Zero-copy and cross-origin isolation get conflated. They are unrelated:

| Idea | What it means | What it costs |
|---|---|---|
| **Zero-copy** (the constraint) | the renderer uploads *from your array*, never into its own | nothing — a plain `ArrayBuffer` on one thread satisfies it completely |
| **Worker transport** (your choice) | how that array gets filled off-thread | see below |

| Transport | Headers | Trade-off |
|---|---|---|
| Same thread | none | simplest; the solver competes with your frame |
| **Transferable `ArrayBuffer`** | **none** | ownership *moves*, so double-buffer (send A, fill B, swap) |
| `SharedArrayBuffer` | COOP + COEP | no transfer; both threads see one allocation |

**The transferable path is the honest default.** It needs no deployment change, and the
double-buffer it forces costs one extra instance array — 9.6 MB at 100,000 instances, which is not
a reason to require cross-origin isolation of every app that wants physics.

SAB earns its headers when the solver runs continuously at a fixed step and you want it writing
while the renderer reads. `InstanceBuffer(capacity, backing?)` accepts either kind and **never asks
which it got** (§11.2).

---

## 8. The solver adapter

The entire contract:

```ts
export interface SolverAdapter {
    /** Write every live body into `out` in the §4.2 layout. Returns the count. Must not allocate. */
    readBodies(out: BodySet): number;
}
```

One method. A Rapier adapter is a loop over `world.bodies`; a Jolt adapter is a loop over its body
interface; a hand-rolled solver writes `out.data` directly and skips the copy entirely.

Stepping is deliberately **not** on the interface. `advance()` takes a callback (§2) because a
worker-hosted solver steps on the other side of a `postMessage` and cannot be called synchronously.
Putting `step()` on the adapter would force every worker integration to fake it.

---

## 9. Test suite

**Headless** (`npm run test:physics`) — all of P1 is pure math, so all of it is headless.

| # | Asserts |
|---|---|
| U1 | `interpolate` at `alpha = 0` reproduces `prev` and at `alpha = 1` reproduces `next`, **byte-identical**, over 64 random transforms |
| U2 | **Shortest arc.** For quaternion pairs with `dot < 0`, the interpolated rotation angle is ≤ 180°; the sign-fix-removed form exceeds it. Both bounds — the second proves the test would catch the regression (§5.3.1) |
| U3 | The composed matrix matches `mat4.fromRotationTranslationScale` to 1e-6, over 64 transforms **including non-uniform scale** — the case that distinguishes column scaling from row scaling |
| U4 | The accumulator runs exactly `floor(t / step)` steps over a 10,000-frame random-delta sequence; with `maxSubsteps` exceeded it clamps and **does not accumulate debt** (§5.2) |
| U5 | **Zero allocation.** 1,000 frames of `advance` + `interpolate` allocate 0 typed arrays, measured by patching the global typed-array constructors ([`design-principles.md`](../design-principles.md) §5) |
| U6 | Adopt-by-reference: mutating the caller's instance array after `setInstances` changes what the next `upload` would send |
| U7 | **The CPU mirror.** `worldBoundsRadius` equals `objectRadius * max(len(col0), len(col1), len(col2))` of the composed matrix, over 64 non-uniform scales — pinning the record against `worldSphereFromModel` rather than against inspection ([§13.1](../design-principles.md)) |
| U8 | A sleeping body's instance floats are bit-identical across 100 frames; clearing the flag resumes interpolation on the next frame |
| U9 | `link` / `unlink` leave floats 16–23 of every instance untouched — the bridge never writes outside its lane (§6) |
| U10 | Every §10 error fires, names the offender, and states the way out |

**No GPU harness.** P1 issues no dispatches and creates no resources; there is nothing a device
would tell us that the math does not. That is a property of the phase, and it is why P1 can ship
without waiting on a browser run.

---

## 10. Errors

| Condition | Message |
|---|---|
| `step <= 0` | `TransformBridge: step <n> must be positive; a zero or negative fixed timestep never advances the accumulator.` |
| `maxSubsteps < 1` | `TransformBridge: maxSubsteps <n> must be at least 1, or the solver never steps.` |
| Backing buffer wrong size | `BodySet: backing buffer is <n> bytes; capacity <c> needs <c*48>. A body record is 12 floats (design §4.2).` |
| `link` with an out-of-range instance index | `TransformBridge: link(instance <i>) exceeds the instance array's capacity <n>.` |
| `link` with an out-of-range body index | `TransformBridge: link(body <i>) exceeds BodySet capacity <n>.` |
| `interpolate` before any solver step | `TransformBridge: interpolate() called before the first step; there is nothing to interpolate between. Call advance() at least once.` |
| `readBodies` returns more than capacity | `TransformBridge: the adapter reported <n> bodies, exceeding BodySet capacity <c>. Construct the bridge with a larger capacity.` |
| Instance array shorter than the linked indices imply | `TransformBridge: instance array holds <n> instances but a link targets index <i>. Pass the array the links were built against.` |

All eight fire at construction or at link time. None can fire per frame — §5.2's clamp is a counter
precisely so it does not have to.

---

## 11. Engine touchpoints

### 11.1 Required — **none**

The instance array is adopted by reference, the record layout is public, and the bridge writes
plain floats into it. P1 imports nothing from `src/core/`.

### 11.2 Optional — one additive parameter

```ts
constructor(public readonly capacity: number, backing?: ArrayBuffer | SharedArrayBuffer)
```

`InstanceBuffer` allocates its own `ArrayBuffer` today (`cullMath.ts:55-59`). Accepting an external
one lets a worker-hosted solver own the allocation, and it is **transport-agnostic** — the parameter
is typed `ArrayBuffer | SharedArrayBuffer` and the constructor never branches on which it got.

**Zero-change fallback, and its actual cost.** Without it, a worker fills its own array and
`postMessage`s it, and the main thread copies into `InstanceBuffer.data`. That is one
`Float32Array.set` of `count * 96` bytes per frame — 9.6 MB at 100,000 instances, roughly 1–2 ms.
Real, but not fatal, and invisible below ~20,000 instances. **The fallback is the reason this
parameter is optional rather than blocking.**

### 11.3 Module changes — additive

| Change | File | Nature |
|---|---|---|
| `bodyMath.ts` — record layout, nlerp, matrix compose, accumulator | new | pure, worker-safe |
| `TransformBridge.ts`, `SolverAdapter.ts`, `BodySet.ts` | new | |
| `backing?` parameter | `visibility/cullMath.ts` | additive; every existing call site is unchanged |
| `null-graph/physics` subpath export | `package.json`, `build.js` | one entry point, following the existing pattern |

---

## 12. Build order

1. **`bodyMath` + the record.** No bridge, no solver. U2, U3, U7. The shortest-arc test comes first
   because every later step interpolates, and a rotation bug found at step 5 looks like a solver bug.
2. **`BodySet` + `TransformBridge.interpolate`, driven by a hand-written keyframe pair.** U1, U9.
   Nothing is simulated; a body moves because a test moves it. **Build the collider wireframe view
   here** — from the body record alone, one batch — because everything after this is easier to see
   than to reason about.
3. **The accumulator.** U4, U5. A body now moves at a fixed step and renders smoothly between them,
   with no solver in the picture.
4. **`SolverAdapter` + one real solver.** Rapier first, because its body iteration is the simplest.
   Nothing structural changes; the callback in step 3 stops being a test fixture.
5. **`link` / `unlink` + spawn/despawn.** U6, U8, U10, and the §10 refusals.
6. **The optional `backing` parameter + a worker integration.** U6 again, across a `postMessage`
   boundary. Left last deliberately: it is an optimization, and steps 1–5 are a complete, correct,
   shippable integration without it.

### Deferred

Everything in [the index](./README.md) — queries, character control, rigid contacts, integration
points, debug draw beyond the step-2 wireframe. Plus, specific to this phase:

- **Compound bodies** (one solver body driving several instances). The link map is already
  many-to-one in the right direction; what is missing is a per-link local offset, which is 12 more
  floats and no new mechanism.
- **Velocity output** for motion blur. Floats exist in the record's flags word only; a real velocity
  needs 3 more floats and a consumer, and `architecture-tree.md` §4 has no velocity buffer yet.
- **Interpolation of the bounding radius** for a body whose collider is animated. Rigid bodies do
  not need it (§1), and a soft body is P3's problem.
