# NullGraph Physics — the plugin series

**Package:** `null-graph/physics` · **Status:** 🔌 plugin — ships on its own cadence, gates no core
release · **Brief:** [`roadmap-modules.md`](../roadmap-modules.md) §11 ·
**Tree:** [`architecture-tree.md`](../architecture-tree.md) §10

NullGraph renders a caller-supplied array of nodes and reads instance transforms out of a flat
buffer it does not own. That is not an accident of the design — it is the design. A physics engine
is therefore a **producer of transforms**, exactly like a Web Worker computing an ECS layout, and it
attaches at the same seam.

The consequence, stated plainly: **most of what people want from "engine physics" needs no engine
code at all.** The series is phased by how much of the work moves to the GPU, and **P1 is a
legitimate stopping point** for most applications.

---

## The three phases specified here

| Phase | Spec | Ships | Engine cost |
|---|---|---|---|
| **P1** | [`physics-p1-transform-bridge-design.md`](./physics-p1-transform-bridge-design.md) | the fixed-step accumulator, interpolation into the instance buffer, the solver-adapter contract | **none** (one *optional* additive parameter) |
| **P2** | [`physics-p2-broadphase-design.md`](./physics-p2-broadphase-design.md) | uniform-grid binning + compacted candidate-pair list on the GPU | none |
| **P3** | [`physics-p3-solver-design.md`](./physics-p3-solver-design.md) | XPBD particle integration + distance constraints (cloth), Jacobi baseline | none |

Each phase renders correctly on its own and each is a stopping point. P1 needs no GPU work at all;
P2 produces a pair list nobody is required to consume; P3 consumes it.

**Read them in order.** P1 fixes the body record every later phase reads, and P2's pair list is P3's
input. If P2 or P3 finds itself changing P1's record, the record was drawn in the wrong place.

---

## Deferred — specs to be written later, deliberately not started

Listed so their absence reads as a decision rather than an oversight. Each becomes its own document
when the phase before it has shipped, because each depends on a contract that does not exist yet.

| Phase | Why it waits |
|---|---|
| **P4 · Queries** (raycast, spherecast, overlap) | Async by construction — a request buffer resolved by a compute stage and read **one frame late**, because readback in the frame loop is a stall the engine forbids everywhere else. Needs P2's spatial grid to be worth building. |
| **P5 · Character controller** | Stays on the **CPU** by design: a capsule sweep against a local subset, driven by P1's bridge. It is the one thing that must be immediate, and immediacy is free on the CPU. Needs P1 shipped and a collider representation P4 settles. |
| **P6 · Rigid contacts** | Contact generation and stacking are where GPU solvers lose to Rapier and Jolt. Needs P3's constraint machinery to exist before the trade-off can be measured rather than argued. |
| **P7 · Integration points** | Terrain heightfields and water buoyancy sampling ([`architecture-tree.md`](../architecture-tree.md) §8), skinned-mesh colliders (§11). All three wait on branches that are `◻` today. |
| **P8 · Debug draw** | Collider wireframes, contact points, broadphase cells, sleeping islands. A `debug-ui` consumer. It is listed last but should be **built first** in practice — see the note below. |

### On debug draw

`architecture-tree.md` §10 says "build this FIRST", and that is right: a physics bug you cannot see
is a physics bug you cannot fix. It is filed as a later *document* rather than a later *task*
because it is not a design problem — it is a line renderer plus four data sources, and every one of
those sources is defined by P1–P3. Writing its spec before those records are fixed would be writing
a spec about nothing.

The practical guidance: **build the collider wireframe view during P1 step 2**, from the body record
alone. It costs one batch and it is the difference between debugging the interpolation and guessing
at it.

---

## What is not planned, at any phase

| Thing | Why not |
|---|---|
| A physics solver of our own | This series is an **adapter**. Rapier and Jolt are better than anything we would write, and a third-party WASM dependency staying third-party is the point. P3 is not an exception — it targets the classes a CPU solver is bad at (count, not fidelity). |
| Requiring `SharedArrayBuffer` | Zero-copy and cross-origin isolation are different ideas. Nothing in NullGraph uses, needs, or checks for SAB, and P1 keeps it that way — see [P1 §7](./physics-p1-transform-bridge-design.md#7-transport-is-the-callers-choice). |
| Physics in the render path | Anything that makes physics a *core* concern is a step away from the engine's premise. The plugin can be abandoned without leaving anything behind in `src/core/`. |
| Joints, motors, continuous collision on the GPU | CPU territory. GPU physics wins on **count**, not on correctness. |
