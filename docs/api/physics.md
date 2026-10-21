# Disc contacts on the XZ plane

Import `DiscContacts` from `null-graph/physics`. This optional entry separates overlapping
discs in a caller-owned GPU record buffer. It does not integrate motion or render bodies.
No physics code is included by importing only `null-graph/agents`.

## A complete fixed-step example

The host supplies an initialized `NullGraph`. This browser-bundler TypeScript example creates
256 moving bodies, including one frozen body, and runs contacts after movement on **every fixed
step**, not once per rendered frame. The returned `frame` performs compute only; a host can add
its own render passes. No GPU readback occurs in `frame`.

```ts
import type { NullGraph } from 'null-graph';
import { AgentSystem, defineAgents, RuleBrain, Sensors } from 'null-graph/agents';
import { DiscContacts } from 'null-graph/physics';

export function createContactDemo(engine: NullGraph) {
    const definition = defineAgents({
        capacity: 256,
        world: { min: [-20, -20], max: [20, 20] },
        cellSize: 4,
        species: [{ name: 'body', radius: 0.5, maxSpeed: 2, maxTurn: 3,
            accel: 3, senseRadius: 4 }],
        sensors: [Sensors.self()],
    });
    const agents = new AgentSystem(engine, definition);
    agents.setBrain(new RuleBrain(`
fn think(i: u32, species: u32, tick: u32) {
    setAct(i, ACT_TURN, 0.2);
    setAct(i, ACT_THROTTLE, 0.8);
}`));
    for (let i = 0; i < definition.capacity; i++) {
        agents.spawn(i, { species: 'body',
            position: [(i % 16 - 8) * 0.35, 0, (Math.floor(i / 16) - 8) * 0.35],
            heading: i * 0.15, frozen: i === 0 });
    }
    agents.upload();
    const contacts = new DiscContacts(engine, agents.buffer, {
        capacity: definition.capacity, stride: 16,
        alive: { offset: 13, mask: 1 },
        fixed: { offset: 13, mask: 2 },
        velocity: { offset: 4 },
        world: { min: [-20, -20], max: [20, 20] },
        maxRadius: 0.5, dt: definition.dt, clamp: true,
    });
    agents.addStepStage({ name: 'contacts', stages: contacts.getComputeStages() });
    let destroyed = false;
    return {
        agents,
        contacts,
        frame(deltaSeconds: number) {
            if (destroyed) return;
            agents.advance(deltaSeconds);
            agents.upload();
            engine.render(agents.getComputeStages());
        },
        destroy() {
            if (destroyed) return;
            destroyed = true;
            agents.destroy();
            contacts.destroy();
        },
    };
}
```

Stop submitting frames before calling `destroy()`. Destroy the system holding the external
stages first, then contacts; the system does not own those external stages. With an asynchronous
brain, first await outstanding `step()` calls/runs before releasing contacts. Never submit a
previously saved chain after its records or contacts have been destroyed.

## Any record buffer

Agents are not required. Construct `DiscContacts(engine, records, options)` with a buffer having
`GPUBufferUsage.STORAGE` and at least `capacity × stride × 4` bytes. The caller owns it.

| Declaration | Meaning |
|---|---|
| `capacity`, `stride` | Positive body count and floats per record; stride is a positive multiple of four |
| floats 0, 1, 2, 3 | Position x, y, z, then radius; contacts changes x and z only |
| `alive?: { offset, mask }` | Raw u32 word after the first four floats; without any mask bit set the body neither moves nor pushes. Omit to make every body live |
| `fixed?: { offset, mask }` | Raw u32 word; any mask bit set gives inverse mass zero. A live fixed body pushes others but is never written |
| `velocity?: { offset }` | Three floats vx, vy, vz after float 3; damping changes vx/vz only. Omit if not present |
| `world: { min, max }` | Finite XZ bounds with min below max; the grid clamps out-of-world entries to edge cells |
| `maxRadius` | Positive declared largest radius; oversized live bodies are counted, but completeness is no longer guaranteed |
| `dt` | Positive fixed timestep in seconds, default 1/60; use the same timestep as movement |
| `clamp` | Default false. When true, corrected centres are clamped to bounds inset by their radius |

Keep positions and radii finite; radii should be positive and at most `maxRadius`. Declare flags
as raw u32 bits, not floating-point values. Do not overlap flag words with writable velocity words.
Unrelated record words, dead records and fixed records are left untouched.

For a custom fixed-step loop, cache `contacts.getComputeStages()` and append it **after movement**
inside each step. The array is cached and reusable, including multiple steps in one submission.
Contacts builds its own stable grid once per step, then runs gather/apply pairs and advances its
window counter. Do not also append contacts outside an `AgentSystem` that already runs them.

## Ordering inside agents

`addStepStage({ name, stages })` copies the supplied array and inserts the stages after act,
among other app stages in registration order, before tick. It works in 1–3-step frame chains and
`await agents.step()`, including its asynchronous-brain acting half. Give each group a unique name.
An empty array is allowed. When both `stages` and `wgsl` are supplied, stages wins; WGSL and
`buffers` are ignored. Individual stage `enabled` flags remain live.

Register contacts **after** a stage that moves or respawns bodies so the grid sees those positions.
Register contacts **before** a stage that should inspect separated positions. Neighbour lists built
before movement are not refreshed by contacts; a later app stage must not treat them as an exact
post-contact query. Choose distance thresholds with the summed radii in mind: separated discs no
longer reach a threshold smaller than their contact distance.

## Tuning and the bounded grid

| Option | Default | Constraint / effect |
|---|---:|---|
| `iterations` | 2 | Integer 1–8, fixed at construction |
| `relaxation` | 1.75 | Strictly between 0 and 2; scales the sum divided by the square root of contact count |
| `compliance` | 0 | Nonnegative; denominator adds compliance / dt², making contacts softer |
| `maxPush` | 0.25 | Positive maximum correction per pass |
| `maxScanPerCell` | 32 | Positive integer budget per cell, fixed at construction; at most nine times this many candidate reads per body/pass |
| `damping` | 1 | From 0 to 1; removes the inward velocity component along the correction |
| `mass` | `'area'` | `'area'` weights mass by disc area; `'equal'` gives unfixed bodies equal mass |
| `cellSize` | derived | At least `2 × maxRadius + (iterations − 1) × maxPush`; default grows to fit device grid/radix limits |

`contacts.setParams({ relaxation?, compliance?, maxPush?, damping? })` validates and uploads
16 bytes without compiling stages. An increase in `maxPush` is refused if it breaks the existing
cell bound. Other construction options require a new solver. `contacts.options.cellSize` reports
the actual cell size; when comparing against the CPU mirror on a device-constrained grid, pass
that size explicitly to the mirror.

Set `contacts.enabled = false` to skip all its stages, including the grid and counter increment,
without rebuilding a chain. Re-enable to resume; disabled steps do not rotate the scan window.

Four iterations can improve dense crowds but cost more. In spike S9, two passes cost about
21–24% of a 16K step on NVIDIA and 25–34% on the AMD/browser paths; four cost about 40%.
The accepted two-pass setting kept ordinary herds near 1.8% of bodies over 5% penetration, with the
arena measurement at 2.13%. A continuously pushing jam still compressed: four passes reduced
the share over 20% penetration from about 50% to 2%, not the milder overlaps. Compliance `1e-4`
was a softer, steadier alternative. These are measured trade-offs, not universal performance
promises. See the [design's S9 results](../physics/physics-contacts-design.md#spike-s9-results--2026-10-18).

## Counters, reference and limits

Call `await contacts.readCounters()` **off the frame path**. It returns last-step `overflow`
(sum of entries beyond each cell's budget, once per grid build) and `bodiesOverMaxRadius`
(live bodies exceeding the declaration). Both reset each step. Overflow is not a count of
missed collisions: over-budget cells use deterministic rotating windows, so not every candidate
is visited in one pass. Bounded work protects dense piles from quadratic scans; it does not
guarantee rapid separation. The application still needs its own measured GPU-time guard.

`resolveDiscContacts(layout, options, records, out?, stepTick = 0)` is the CPU reference for one
step. Without `out` it modifies records in place; with `out` it first copies the active records.
It returns records, final-pass contact counts and counters. Increment `stepTick` between calls
to match GPU scan-window rotation. It is for tests/offline tools, not per-frame GPU readback.

Reruns are word-identical on one device, not across GPUs. Compliance resets its multiplier each
pass, so residual overlap depends on iteration count: this is not persistent-multiplier XPBD.
The cell completeness proof assumes declared radii, no over-budget cells, and initially in-bounds
bodies when clamping. Clamping an out-of-bounds body can exceed the push bound for that step;
an isolated body without contacts is not moved merely to enforce bounds.

This is discrete XZ disc separation, not 3D, friction, rigid stacking, wall/segment collision or
continuous collision detection. Fast bodies can tunnel. It supplies no collision groups and
does not promise zero overlap in a pile or a driven jam.