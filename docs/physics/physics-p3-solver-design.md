# NullGraph Physics — P3: GPU Solver (XPBD) — Technical Design

**Target:** `src/extras/physics/` (`xpbdMath`, `xpbdShaders`, `ParticleSolver`)
**Package:** `null-graph/physics` 🔌
**Status:** `plugin` — ✍ **designed, not implemented.**
**Series:** [P1 · Transform bridge](./physics-p1-transform-bridge-design.md) →
[P2 · GPU broadphase](./physics-p2-broadphase-design.md) → **P3 · GPU solver** ·
[index](./README.md)
**Depends on:** P2's pair list for particle–particle contacts. Distance constraints need only P1.
**Engine cost:** **none.**

---

## 1. Purpose

Simulate on the GPU the classes that suit it — particles, sand, debris, and cloth — as a sequence of
compute dispatches over a flat buffer, with no per-frame allocation and no readback.

### The single most important pre-existing fact

**WGSL has no `atomic<f32>`** ([`roadmap.md`](../roadmap.md) §6), and that one absence decides the
solver.

A Gauss–Seidel solver accumulates constraint corrections into shared positions as it goes, which
needs float atomics. Without them there are exactly three options:

| Option | Convergence | Cost |
|---|---|---|
| **Jacobi** — compute all corrections against the old positions, then average | slower per iteration | nothing exotic; one extra buffer |
| **Graph-coloured batches** — dispatch per colour, no two constraints in a colour share a particle | Gauss–Seidel quality | a graph colouring at build time, and a dispatch per colour |
| **Fixed-point atomics on `atomic<u32>`** — scale, add, unscale | Gauss–Seidel quality | a scale factor that must not overflow or lose precision |

**Jacobi is the baseline**, exactly as the atomic-free path is the baseline everywhere else in this
engine (`clusteredComputeShader.ts` bins with plain `atomicAdd` and *accelerates* with subgroups
when the device has them; it never requires them). Graph colouring is filed as the acceleration, not
the design.

Stating this first is the point: the constraint is a platform fact, not a preference, and a solver
designed as though float atomics existed would be a solver that cannot ship.

### The second fact: the particle record is already spoken for

[`roadmap-modules.md`](../roadmap-modules.md) §8 says the Effects module's GPU particles and this
solver want the same buffer: *"Design the particle record once, with room for a velocity and an
inverse mass, and the physics plugin can simulate the same buffer the renderer draws."*

Neither exists yet, so P3 fixes the record (§4.1) and Effects adopts it. If Effects ships first, P3
adopts theirs. What must not happen is two records and a transfer between them — the whole argument
for GPU physics is that the thing being simulated and the thing being drawn are the same memory.

### Non-goals for P3

- **Rigid bodies.** Contacts, stacking, friction, and joints are where a CPU solver wins on
  correctness and this one would lose. Deferred to P6 with a reason, not omitted.
- **Competing with Rapier or Jolt on fidelity.** GPU physics wins on **count**. A scene with 40
  crates belongs on the CPU and always will.
- **Continuous collision.** Fast particles tunnel. Substepping mitigates it; CCD is not in scope.
- **Soft bodies / shape matching.** One more constraint type on the same machinery, filed (§12).
- **Self-collision for cloth.** Needs P2's grid at cloth-vertex density, which is a different tuning
  problem than rigid-body broadphase.

### Design principles

1. **The atomic-free path is the baseline.** §1. Optional capability may accelerate, never gate
   ([`design-principles.md`](../design-principles.md) §12).
2. **One buffer, two consumers.** The solver integrates what the renderer draws. No transfer.
3. **Everything expensive happens at build time.** Constraint topology, graph colouring, rest
   lengths — all precomputed. Per-frame cost is dispatches ([§11](../design-principles.md)).
4. **Substep, don't iterate.** §5.3. This is the one place XPBD's literature has a clear answer and
   it is not the intuitive one.

---

## 2. Architecture

```
   engine.render([
      ...broadphase.getStages(),      // P2 — pair list (particle contacts only)
      ...solver.getStages(),          // predict -> [solve x N] -> finalize
      ...vis.getComputeStages(),
      scenePass,                      // draws the SAME buffer the solver integrated
      ...post.getPassSequence(),
   ]);
```

```
   ┌───────────────────────────────────────────────┐
   │ ParticleBuffer — 12 floats/particle (§4.1)     │
   │  position · prevPosition · velocity · invMass  │
   └───────┬───────────────────────────────────────┘
           │
           ▼  Stage A: predict          x_pred = x + v*dt + g*dt²
   ┌───────────────────┐
   │ substep loop      │   for s in 0..substeps:
   │  ┌──────────────┐ │      Stage B: distance constraints  (Jacobi)
   │  │ solve pass   │ │      Stage C: contact constraints   (Jacobi, reads P2 pairs)
   │  └──────────────┘ │      Stage D: apply averaged deltas
   └───────┬───────────┘
           ▼  Stage E: finalize         v = (x - x_prev) / dt
   ┌───────────────────────────────────────────────┐
   │ the same buffer, drawn as an indirect batch   │
   └───────────────────────────────────────────────┘
```

Every stage is a `ComputeStageNode`. Consecutive ones share a WebGPU pass and the scene pass closes
it (`RenderPipeline.ts:45-48,66-70`) — the substep loop is `substeps × 3` stages spliced into the
array, not a GPU-side loop, because the dispatch count is CPU-known and the array is the scheduler.

---

## 3. Integration facts

| Fact | Source | Consequence |
|---|---|---|
| WGSL has no `atomic<f32>`. | [`roadmap.md`](../roadmap.md) §6 | §1. Jacobi baseline; colouring and fixed-point filed. |
| `atomic<u32>` binning with `preClearBuffers` zeroing ships and is tested. | `LightSystem.ts:399-419` | The fixed-point option (§6.3) needs no new engine capability if it is ever wanted. |
| Consecutive compute stages share one pass. | `RenderPipeline.ts:45-48,66-70` | `substeps × 3` stages cost one pass, not `substeps` passes. |
| `ComputeStageNode.enabled` skips a stage's clears **and** its dispatch. | `types.ts:171-180`, `RenderPipeline.ts:28` | A constraint type with zero instances this frame costs nothing, without rebuilding the array. |
| An `isIndirect` batch draws a compute-filled count via `drawIndexedIndirect`. | `RenderPipeline.ts:140-143` | The renderer draws the live particle count with no CPU involvement — the shape Effects §8 also wants. |
| A batch's storage buffer can be an externally-created buffer (`sharedSourceBuffer`). | `types.ts:58`, `BatchManager.ts:88-91` | **The solver's buffer is the draw's buffer.** §2's claim needs no engine change. |
| `dispatchCount` takes a callback over CPU-known values; `indirectBuffer` takes the count from a GPU buffer. | `types.ts:156`, `types.ts` (`indirectBuffer`) | Particle count is CPU-known, so predict/solve/finalize dispatch directly. The **contact** solve is keyed on P2's pair count and dispatches indirectly ([P2](./physics-p2-broadphase-design.md) §6.2). |
| P1's `BodySet` is 12 floats in three `vec4`s. | [P1](./physics-p1-transform-bridge-design.md) §4.2 | §4.1 deliberately mirrors that shape so one debug view reads both. |

---

## 4. Data model

### 4.1 The particle record — 12 floats / 48 bytes

```wgsl
struct Particle {
    position: vec3<f32>,      // floats 0..2   — current, solved in place
    invMass: f32,             // float  3      — 0 means pinned; §5.2
    prevPosition: vec3<f32>,  // floats 4..6   — XPBD's velocity is implicit in this
    _pad0: f32,               // float  7
    velocity: vec3<f32>,      // floats 8..10  — written by finalize, read by the renderer
    flags: f32,               // float  11     — u32 bits: asleep, pinned, colour group
};
```

Three `vec4`s, matching P1's body record byte-for-byte in shape. `invMass` sits in the position
vector's `w` because every constraint reads both together — one load, not two.

`invMass = 0` is the pin, and it is a *value*, not a branch: a constraint multiplies its correction
by `invMass`, so a pinned particle absorbs zero correction with no divergent code path. That is the
standard XPBD idiom and it is why the field is inverse mass rather than mass.

### 4.2 Constraint records, built once

```wgsl
struct DistanceConstraint {
    a: u32, b: u32,
    restLength: f32,
    compliance: f32,      // XPBD's alpha — the inverse of stiffness, in m/N
};
```

16 bytes. `compliance` is what makes this XPBD rather than PBD: stiffness becomes a physical
quantity independent of the iteration count, so a cloth does not get stiffer when you raise the
substep count. That property is the reason to pick XPBD at all, and losing it by hardcoding
compliance to zero would leave a solver whose behaviour is a function of its performance settings.

### 4.3 The Jacobi delta buffer

```
deltas:      maxParticles × 4 floats   (accumulated correction, xyz + weight)
```

Jacobi needs somewhere to accumulate corrections that is *not* the position being read. That is the
extra buffer §1 charges it: 1.6 MB at 100,000 particles.

The `w` component is the correction **count**, and dividing by it is the averaging that makes Jacobi
stable. Summing without averaging over-corrects and the cloth explodes — visibly and immediately,
which is at least an honest failure.

---

## 5. The mechanism

### 5.1 Predict

```wgsl
prevPosition = position;
velocity += gravity * dt;
position += velocity * dt;
```

### 5.2 Solve — one Jacobi pass

For a distance constraint between `a` and `b`:

```wgsl
let d = p.position[a] - p.position[b];
let len = length(d);
let w = invMass[a] + invMass[b];
if (w <= 0.0 || len <= 1e-9) { return; }              // both pinned, or coincident
let alpha = c.compliance / (dt * dt);                  // XPBD: compliance is time-scaled
let lambda = -(len - c.restLength) / (w + alpha);
let corr = d * (lambda / len);
// Jacobi: accumulate, do not write positions
atomicAddDelta(a,  corr * invMass[a]);
atomicAddDelta(b, -corr * invMass[b]);
```

`atomicAddDelta` is not a float atomic — it is a *scatter into the delta buffer*, and §6.1 explains
how it avoids needing one.

Then Stage D applies:

```wgsl
if (deltas[i].w > 0.0) { position[i] += deltas[i].xyz / deltas[i].w; }
```

### 5.3 Substeps beat iterations

The intuitive tuning is "more solver iterations per step." XPBD's result is the opposite: for a
fixed budget, **more substeps with one iteration each converges better than one substep with many
iterations** — because each substep re-integrates, so constraint corrections propagate through the
system rather than being re-projected against stale predictions.

The concrete default: `substeps: 8, iterations: 1`, not `substeps: 1, iterations: 8`. Same dispatch
count, materially better cloth.

This is stated because it is counter-intuitive, it is the single biggest quality lever in the
solver, and someone tuning it from intuition will turn the wrong knob.

### 5.4 Finalize

```wgsl
velocity = (position - prevPosition) / dt;
```

XPBD derives velocity from the position change rather than tracking it through the solve, which is
what makes the constraints unconditionally stable — a correction cannot inject energy it did not
account for.

---

## 6. The atomic-free scatter

### 6.1 Why the delta buffer needs no float atomics

Two constraints touching the same particle in the same dispatch would race on `deltas[i]`. The
Jacobi answer is to **invert the loop**: dispatch over *particles*, not constraints, and have each
particle gather the constraints that reference it.

That needs a per-particle constraint list — built once, at construction, as the same
offset+count+indices layout P2 builds per cell (§4.2 there) and `LightSystem` builds per cluster.
Three uses of one structure.

The cost is that a constraint's math runs twice, once from each endpoint. For a 16-byte constraint
that is cheaper than any synchronization would be, and it removes the race entirely rather than
managing it.

### 6.2 Graph colouring, filed with its actual cost

Colouring the constraint graph so no two constraints in a batch share a particle allows true
Gauss–Seidel: dispatch per colour, write positions directly, no delta buffer, faster convergence.

The cost is a colouring pass at build time (greedy is adequate; cloth grids colour into 4 sets by
construction) and one dispatch per colour instead of one per constraint type. **That is the entire
delta** — no new engine capability, no format change.

It is deferred because Jacobi ships correct and this is an optimization whose benefit should be
measured against a working baseline, not assumed. Filed here with its cost so it reads as a known
next step rather than a rewrite.

### 6.3 Fixed-point atomics, filed and not recommended

`atomic<u32>` with a scale factor gives true accumulation. It also gives an overflow that is silent,
a precision that depends on the scale factor, and a failure mode that looks like a physics
instability. Filed for completeness; §6.2 is the better trade.

---

## 7. What P3 simulates, precisely

| Class | Constraints | In scope |
|---|---|---|
| Particles, sand, debris | contacts (P2 pairs) + ground plane | ✅ |
| Cloth | distance + bending | ✅ |
| Ropes, chains | distance | ✅ — a 1D cloth |
| Soft bodies | shape matching | ◻ filed (§12) |
| Rigid contacts | contact + friction + stacking | ◻ **P6, deliberately** |

The line is drawn at **anything needing an orientation**. A particle has a position and a mass; the
moment a body has an inertia tensor, the constraint math changes shape and the CPU's advantages
start to matter. Keeping P3 to positions is what makes it a phase rather than a project.

---

## 8. Test suite

**Headless** (`npm run test:physics`):

| # | Asserts |
|---|---|
| U1 | **Energy does not grow.** A pendulum of 32 distance constraints, 10,000 steps: total energy is non-increasing within 1%. The property XPBD is chosen for; a sign error in §5.2 breaks it immediately |
| U2 | A single distance constraint at `compliance = 0` converges to `restLength` within 1e-4 in 8 substeps, from 64 random initial separations |
| U3 | **Compliance is iteration-independent.** The same compliance produces the same steady-state extension at 4, 8, and 16 substeps within 2% — the property §4.2 exists for, and the one PBD does not have |
| U4 | `invMass = 0` particles are bit-identically stationary after 1,000 steps with constraints pulling on them (§4.2) |
| U5 | **Jacobi averaging.** Removing the `/deltas[i].w` divide makes a 32×32 cloth diverge within 100 steps; with it, it does not. Both bounds ([§13](../design-principles.md) rule 1) |
| U6 | The per-particle constraint list (§6.1) references every constraint exactly twice, once per endpoint, over 64 random topologies |
| U7 | Determinism: the same initial state and step count produces bit-identical results across runs |
| U8 | The CPU mirror matches the WGSL kernel's correction for 64 constraint configurations to 1e-5 ([§13.1](../design-principles.md)) |
| U9 | Zero per-frame allocation across 1,000 steps, by typed-array constructor count |
| U10 | No backtick in any generated WGSL, no `PI` redeclaration, every module parses |
| U11 | Every §9 error fires and names the offender |

**GPU integration:**

| # | Asserts |
|---|---|
| T1 | GPU and CPU mirror agree to 1e-4 after 100 steps of a 1,024-particle cloth |
| T2 | A cloth pinned at two corners reaches a static hang and stays within 1e-3 for 1,000 further steps |
| T3 | The solver's buffer **is** the draw's buffer — mutating it in a compute stage changes the rendered image with no intervening copy (§3) |
| T4 | 100,000 particles with contacts solve in under 2 ms on the reference device, from the profiler |
| T5 | `substeps: 8, iterations: 1` converges measurably better than `substeps: 1, iterations: 8` at equal dispatch count (§5.3) |

T3 is the architectural test rather than the numeric one: it fails if a well-meaning refactor
introduces a staging copy between the solver and the renderer, which is precisely the thing this
phase exists to avoid.

---

## 9. Errors

| Condition | Message |
|---|---|
| `substeps < 1` | `ParticleSolver: substeps <n> must be at least 1.` |
| `compliance < 0` | `ParticleSolver: compliance <n> is negative; compliance is inverse stiffness in m/N and cannot be below zero. Use 0 for a rigid constraint.` |
| Constraint references an out-of-range particle | `ParticleSolver: constraint <i> references particle <p>, beyond the buffer's <n>.` |
| Constraint with `a == b` | `ParticleSolver: constraint <i> connects particle <p> to itself; its length is always 0 and the correction is undefined.` |
| Every particle in a constraint pinned | (allowed — the `w <= 0` guard in §5.2 makes it a no-op, by design) |
| `getStages()` before particles were set | `ParticleSolver: no particles. Call setParticles(buffer, count) before splicing the stages.` |
| Contacts enabled without a broadphase | `ParticleSolver: contact constraints need a pair list. Attach a GpuBroadphase (P2), or disable contacts for a constraint-only solve.` |
| Particle count exceeds the batch's `maxInstances` | `ParticleSolver: <n> particles exceeds the draw batch's maxInstances <m>, fixed at batch creation.` |

---

## 10. Engine touchpoints

### 10.1 Required — **none**

Compute stages, storage buffers, `sharedSourceBuffer`, indirect draw, and `enabled` all ship.

### 10.2 Optional — filed

- **Subgroup reductions** for the delta averaging. Would cut the delta buffer's bandwidth. Detect
  and accelerate, never require — `capabilities.subgroups` already reports it.
- **`shader-f16`** for the delta buffer. Detected today, consumed by nothing
  ([`architecture-tree.md`](../architecture-tree.md) §16); halving delta bandwidth is a plausible
  first consumer, and it is filed rather than assumed because the precision cost is unmeasured.

### 10.3 Module changes — additive

| Change | File | Nature |
|---|---|---|
| `xpbdMath.ts` — constraint projection, the CPU mirror, energy accounting | new | pure; U1/U8's reference |
| `xpbdShaders.ts`, `ParticleSolver.ts` | new | |
| The particle record | shared with Effects §8 | **fixed here, adopted there** (§1) |

---

## 11. Build order

1. **`xpbdMath` + the CPU solver.** No GPU. U1, U2, U3, U4. The energy property is asserted before
   anything runs on a device, because an unstable solver on the GPU is indistinguishable from a
   broken kernel.
2. **The particle buffer + predict/finalize on the GPU, no constraints.** Particles fall. T3 —
   proving the solver's buffer is the draw's buffer *before* anything depends on it.
3. **Distance constraints, Jacobi, single substep.** U5, U6, U8, T1. Cloth hangs.
4. **Substepping.** U7, T2, T5. The §5.3 claim becomes a measurement.
5. **Contacts from P2's pair list.** T4. Particles collide with each other rather than only with
   constraints.
6. **The `enabled` gate per constraint type.** A scene with no cloth pays nothing for the cloth
   stages.

### Deferred

- **Graph colouring** (§6.2) — filed with its full cost; an optimization over a working baseline.
- **Soft bodies** via shape matching — one more constraint type on this machinery.
- **Bending constraints** for cloth that resists folding. Distance constraints alone give a cloth
  that creases like paper.
- **Self-collision** — P2's grid at vertex density, a different tuning problem.
- **Rigid contacts, friction, stacking, joints** — P6, and possibly never on the GPU (§7).
- **Fixed-point atomics** (§6.3) — filed and not recommended.
