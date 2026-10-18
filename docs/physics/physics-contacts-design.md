# NullGraph Physics — Phase 2 step 13: disc contacts on the XZ plane — Technical Design

**Parent:** [`../work/plan-2.md`](../work/plan-2.md) §3 (the library rule), §5 (milestone 2B, step 13)
and §9 (decision 2, Q4: contacts get a new `null-graph/physics` entry); [`README.md`](./README.md)
(the physics series); [`../agents-queries-design.md`](../agents-queries-design.md) §14 (spike S8's
findings and what it left to this step).
**Status:** accepted with recommended answers by the user on 2026-10-18 (§15). Spike S9 (phase
13.0) measured and accepted; Q5–Q7 accepted with recommended options, Q8 deferred to scenarios. Phase 13.1 reviewed and committed (`8d0d1df`); phase 13.2 built and validated. Phase 13.3 next.
**Package:** a new entry, `null-graph/physics` 🔌.
**Builds on:**
- `PointGrid` ([`../api/agent-grid.md`](../api/agent-grid.md), `src/extras/spatial/`), the stable
  XZ grid the agents already use;
- P3's delta buffer and gathering loop ([P3](./physics-p3-solver-design.md) §4.3, §6.1), designed
  but not built;
- `AgentSystem`'s fixed step ([`../agents-gpu-loop-design.md`](../agents-gpu-loop-design.md) §8);
- spike S8's prototype and results (`test/spikes/contacts/`).

**Target:** `src/extras/physics/` (new: `contactMath.ts`, `contactShaders.ts`, `DiscContacts.ts`,
`index.ts`); one seam in `src/extras/agents/AgentSystem.ts` (§8); the entry in `build.js` and
`package.json`; headless U46–U49, GPU A37–A41 (a new `physics` GPU suite); `docs/api/physics.md`.
Core unchanged.

---

## 1. Purpose

Keep the bodies of any GPU record buffer from overlapping in the XZ plane. At each fixed step, after
the caller has moved them, a few Jacobi passes push every pair of overlapping discs apart. A body is
a disc: its centre is floats 0 and 2 of its record, its radius float 3. The passes are deterministic
(exact reruns on one device), use no float atomics, cost a bounded amount however the bodies crowd,
and take a compliance that makes contacts soft.

Who uses it (plan-2 §5's second consumers):
- **crowds:** the venue planner (I13), where pedestrians must keep their space for density maps and
  bottleneck times to mean anything;
- **horde games (G1, G2):** enemies that surround the player instead of stacking into one point;
- **herds** in films and in the co-evolution arenas (step 14). Spike S7 found that herds driven by
  the cloned networks stack without contacts: mean nearest neighbour 0.66–0.72 m against a body
  diameter of 0.80 m (1.28 m under the rule);
- **later, particles** (P3), once a 3D grid exists (deferred, §13).

### Pre-existing facts this design rests on

1. **The agent record and P1's body record agree on floats 0–3:** position x, y, z, then radius
   (`agentLayout.ts:12-19`; P1 §4.2). One `vec4` load per body serves both. P3's particle record,
   designed but not built, puts inverse mass at float 3 (P3 §4.1); §12 resolves it.
2. **WGSL has no `atomic<f32>`** (P3 §1). Each body therefore gathers its own contacts and writes
   only its own correction, P3 §6.1's inverted loop, and Jacobi averages the corrections (P3 §4.3).
3. **`PointGrid` is stable.** It lists each cell's live records by ascending index, through a stable
   radix sort (`PointGrid.ts:69-92`; A1, A23). A gather that walks the 3×3 cells, and each cell's
   entries, in that order adds in the same order on every run. Exact reruns follow by construction.
4. **Spike S8 measured a prototype** (`agents-queries-design.md` §14): averaged positional Jacobi, a
   0.25 m push limit, inward-velocity damping, no compliance.
   - With two passes over a rebinned 8 m grid, the worst penetration in an ordinary herd fell from
     0.784 m to 0.028 m (4,096 agents, final state, NVIDIA).
   - No configuration cleared a 256-agent pile: after 180 steps, 63–88% of its agents still
     overlapped a neighbour by more than 5% of their summed radii.
   - Its costs are single wall-clock samples, some contradictory (one pass costing more than two),
     so they cannot set defaults.
5. **`AgentSystem` has no place for another entry's stages inside each fixed step.**
   - A step is [grid, queries, sense] then [think…, act, app stages, tick], and the chains for 0–3
     steps per frame are cached (`AgentSystem.ts:190-206`).
   - `addStepStage` adds one per-agent dispatch with the system's buffers and at most 3 of the
     caller's on a default device (`AgentSystem.ts:208-213`, `346-356`).
   - S8 had to splice its stages into a copy of the chain, by stage name.
6. **The agents entry is 21,468 B min+gzip, 27 B under its 21,495 B ceiling** (U45). A minimal seam
   (§8) measured **+27 B: exactly the ceiling**, with nothing to spare.
7. **A steering agent's velocity lies along its heading,** and the act stage rederives speed from
   its length (`agentLayout.ts:20`; `actAgent`). A contact can usefully change a velocity's length
   (damping), not its direction.

### Non-goals

- **Rigid bodies:** friction, stacking, rotation. P6, and the CPU's territory ([README](./README.md)).
- **3D:** discs in XZ only. Spheres need a 3D grid (P2); deferred.
- **Avoidance:** steering away before contact is the brain's job (sensors, rules, networks).
- **Walls and obstacles that are not bodies** (NAV, in plan-2's backlog). A round obstacle can be a
  fixed body (§7).
- **Continuous collision:** a body that moves farther than its radius in a step can pass through
  another.
- **Equality across GPUs:** reruns are exact on one device only.
- **Readback in the frame loop:** counters are read off the frame path.

### Principles

- **The library rule** (plan-2 §3): no scenario or research words in the API, its docs or its WGSL;
  any record buffer, by declared layout.
- **Atomic-free and deterministic** (P3 §1, plan-2 §7).
- **The conservative direction** ([design-principles](../design-principles.md) §3) is to bound the
  work and count what was skipped. A missed contact leaves a visible overlap for a step; an
  unbounded scan can hold the GPU past Windows' reset at about 2 s ([gotchas](../work/gotchas.md)).
- **Measured before building:** phase 13.0 (spike S9, §16) sets the defaults.
- **Pay for what you use:** a separate entry. Agent apps without contacts pay only the seam.

---

## 2. The formal definition

**Symbols.**
- Bodies $i \in [0, N)$; record $i$ starts at float $iS$ (stride $S$, a multiple of 4).
- Centre $x_i = (r_i[0],\, r_i[2])$; radius $\rho_i = r_i[3]$; $\rho_{\max}$, the declared largest
  radius.
- Live $L(i)$: the alive mask on the declared flags word (every body is live without one). Fixed
  $F(i)$: the fixed mask.
- Inverse mass $w_i = 0$ when $F(i)$; otherwise $1$ (`'equal'`) or $(\rho_{\max}/\max(\rho_i, 10^{-4}))^2$
  (`'area'`: mass in proportion to the disc's area, the largest body at 1; non-positive $\rho_i$ guarded).
- Parameters: $K$ passes per step; relaxation $\omega \in (0, 2)$; compliance $\alpha \ge 0$, with
  $\tilde\alpha = \alpha / \Delta t^2$; push limit $m > 0$; per-cell scan budget $B$; damping
  $\beta \in [0, 1]$.
- The grid is built from $x^0$, the positions when the contact stages start, with cell edge
  $h \ge 2\rho_{\max} + (K - 1)\,m$.

**Pass $k = 0 \dots K-1$, from $x^k$:**
- **Candidates** $N_i^k$: live $j \ne i$ listed in the 3×3 cells around $\text{cell}(x_i^k)$. At
  most $B$ entries of each cell are scanned; a cell holding more yields a window that rotates with
  the cell, the step and $k$.
- **Contacts** $C_i^k = \{\, j \in N_i^k : d_{ij} < \rho_i + \rho_j,\ w_i + w_j > 0 \,\}$, with
  $d_{ij} = \lVert x_i^k - x_j^k \rVert$.
- **Per contact:** penetration $\delta_{ij} = \rho_i + \rho_j - d_{ij}$; normal
  $n_{ij} = (x_i^k - x_j^k)/d_{ij}$, or, when $d_{ij} \le \varepsilon$, a unit vector hashed from
  $(\min(i,j), \max(i,j))$ and signed so that $n_{ji} = -n_{ij}$;
  $$\Delta_{ij} = \frac{w_i}{w_i + w_j + \tilde\alpha}\,\delta_{ij}\,n_{ij}.$$
- **Correction:** $u_i = \dfrac{\omega}{\sqrt{|C_i^k|}} \sum_{j \in C_i^k} \Delta_{ij}$, added in grid
  order, then shortened to length $m$ if longer.
- **Move:** $x_i^{k+1} = \Pi_i(x_i^k + u_i)$ for a live, unfixed body with $|C_i^k| > 0$, else
  $x_i^{k+1} = x_i^k$. $\Pi_i$ clamps into the world inset by $\rho_i$ when clamping is asked for,
  as the agents' act stage does.
- **Velocity**, when its offset is declared and $u_i \ne 0$: with $\hat n = u_i/\lVert u_i\rVert$ in XZ,
  $(v_{i,x}, v_{i,z}) \leftarrow (v_{i,x}, v_{i,z}) - \beta \min(0,\ (v_{i,x}, v_{i,z}) \cdot \hat n)\,\hat n$;
  float `offset + 1` ($v_y$) is untouched.

The step's result is $x^K$.

**What "XPBD compliance" means here.** A pass is XPBD's projection with its multiplier reset to
zero: pairs are found from the grid on every pass, so no multiplier survives between passes. At
$K = 1$ that is exactly one XPBD iteration. At $K > 1$ the residual overlap depends on $K$, which
XPBD exists to avoid. S9 measures how much, and the API guide will state it.

**Properties, each a test (§10):**

| | Property | Means |
|---|---|---|
| **C1** | Complete within the bound | With no cell over $B$ and every $\rho_i \le \rho_{\max}$, $C_i^k$ holds every live body overlapping $i$ at every pass $k < K$ |
| **C2** | Exact reruns | The same records, stepped the same number of times, give the same buffers word for word on one device |
| **C3** | One pair, one pass | Two discs alone ($\tilde\alpha = 0$, $\omega = 1$, unclamped) end exactly touching, each moving its share $w_i/(w_i+w_j)$ of $\delta$ along $\pm n$; their mass-weighted centre does not move |
| **C4** | Fixed and dead bodies | A fixed body never moves (bit for bit), and the other body of each of its pairs takes the whole correction; a dead body neither moves nor pushes |
| **C5** | Compliance | $\alpha > 0$ removes less than the hard share per pass; under a constant push the steady overlap grows with $\alpha$, and $\alpha = 0$ is the hard contact |
| **C6** | Bounded work | Per body and pass at most $9B$ candidates. Skipped entries are counted, and windows rotate, so every entry is scanned over time |
| **C7** | The CPU mirror | The mirror and the GPU agree within $10^{-5}$ of the scale on random layouts, with equal contact sets |
| **C8** | Safe positions | With clamping, every live centre stays inside the world inset by its radius, and no pass moves a body more than $m$ |

---

## 3. Integration facts (verified against this repository, 2026-10-18)

| # | Fact | Where | Consequence |
|---|---|---|---|
| 1 | Floats 0–3 of the agent record are position and radius; velocity is 4–6; flags are the u32 at 13 (alive bit 1, frozen bit 2) | `agentLayout.ts:12-32` | Agents are bodies by declaration: stride 16, flags at 13, velocity at 4. No agents type enters the physics entry |
| 2 | P1's body record has position and radius at floats 0–3 | P1 §4.2 | The same declaration serves P1's bodies |
| 3 | `PointGrid` takes any records buffer, a stride in floats, an optional flags word and alive mask, and bins only live records, by cell then index | `PointGrid.ts:25-92`, `gridMath.ts` (`buildPointGrid`) | The solver builds its own grid from the caller's buffer with no new grid code; `buildPointGrid` is the mirror's grid |
| 4 | `cellOffsets` has `cellTotal + 1` words, the last the live total | `PointGrid.ts`, `agent-grid.md` | A cell's count is `offsets[c + 1] − offsets[c]`: the gather binds offsets and entries only |
| 5 | The agents' query walks the 3×3 cells clamped at the edges and starts an over-budget cell at `gridWindow(c, tick) % count` | `agentGridShaders.ts:40-48`, `agentGridWindow.ts` | The gather reuses the pattern and the hash. `gridWindow` lives in agents today; its home becomes `spatial` (§12) |
| 6 | A step's chain is sensing then acting; app stages run after act in insertion order; `step()` renders the same halves | `AgentSystem.ts:190-206`, `516-530` | A seam among the app stages reaches both `getComputeStages()` and `step()` (§8) |
| 7 | `addStepStage` binds 5 system buffers and allows `maxStorageBuffersPerShaderStage − 5` of the caller's | `AgentSystem.ts:208-213`, `346-356` | Contacts cannot be one step stage that also builds a grid; §8's fallback shows what they can be |
| 8 | `ComputeStageNode.enabled` skips a stage's dispatch | `core/types.ts:258` | Contacts switch on and off without rebuilding any chain |
| 9 | `NullGraph.createComputeStageAsync` exists | `core/NullGraph.ts:138` | An asynchronous build is possible later (deferred) |
| 10 | P2's pair list is compacted with `atomicAdd` on a pair counter | P2 §2 (stage D), §4.3 | Its order differs from run to run; a solver adding in pair-list order would lose exact reruns. Discs in XZ gather from the grid instead (§14) |
| 11 | The build has no `physics` entry; `package.json` exports none | `build.js:13-30`, `package.json` | Both gain one line (§12) |
| 12 | The agents ceiling is enforced by U45 | `agents-system.test.mjs:112-117` | The seam's bytes are a user decision (§15 Q3) |

---

## 4. Data model

| Buffer | Owner | Size | Contents |
|---|---|---|---|
| records | caller | capacity × stride × 4 B | x, y, z, radius at floats 0–3; flags (u32) and velocity at declared offsets |
| deltas | solver | capacity × 16 B | per pass: the summed correction (x, z), the contact count, unused |
| grid | solver (`PointGrid`) | `cellTotal + 1` + capacity words, plus radix scratch | as [`agent-grid.md`](../api/agent-grid.md) |
| params | solver | 16 B uniform | $\omega$, $\tilde\alpha$, $m$, $\beta$: changed by `setParams`, no recompile |
| counters | solver | 16 B | step counter (word 0, rotates window), bodies over $\rho_{\max}$ (word 1, zeroed each step); over-budget entries are the grid's own `overflow` |

At 65,536 bodies in a 640 m world with 1.5 m cells (182,329 cells), the grid holds 4.2 MB
(counts, offsets and entries, plus the radix sort's keys, indices, digit table and one scan per
pass; computed from `gridRadixLayout`) and the deltas 1 MB.

The layout, as declared:

```ts
interface DiscLayout {
    capacity: number;
    /** Floats per record, a multiple of 4. Floats 0–3 are x, y, z and radius. */
    stride: number;
    /** A u32 flags word: bodies without `mask` set are skipped (dead). */
    alive?: { offset: number; mask: number };
    /** Bodies with `mask` set push others and never move. */
    fixed?: { offset: number; mask: number };
    /** Floats offset..offset+2 are a velocity (offset+0: vx, offset+2: vz); the inward XZ normal part is removed (β), vy untouched. */
    velocity?: { offset: number };
}
```

---

## 5. API

### Recommended (option A): a `DiscContacts` object that owns its stages, and one agents seam

```ts
import { DiscContacts } from 'null-graph/physics';

const contacts = new DiscContacts(engine, records, {
    capacity: 16_384,
    stride: 16,                                 // floats per record; x, y, z, radius first
    alive: { offset: 13, mask: 1 },             // a u32 flags word: dead bodies are skipped
    fixed: { offset: 13, mask: 2 },             // these push others and never move
    velocity: { offset: 4 },                    // remove the inward normal part of the velocity
    world: { min: [-80, -80], max: [80, 80] },  // the grid's extent
    clamp: true,                                // keep centres inside the world, inset by radius
    maxRadius: 0.6,
    // Optional: cellSize defaults to 2 * maxRadius + (iterations - 1) * maxPush (clamped to device).
    // cellSize: 1.45,
    dt: 1 / 60,
    // Defaults from S9:
    iterations: 2, relaxation: 1.75, compliance: 0, maxPush: 0.25, maxScanPerCell: 32,
});
// Each fixed step, after the bodies have moved:
engine.render([...contacts.getComputeStages(), /* … */]);
// Inside an AgentSystem's step, in this position among its app stages (§8):
agents.addStepStage({ name: 'contacts', stages: contacts.getComputeStages() });
```

- **`getComputeStages()`** returns the same array every time: the grid's stages, then gather and
  apply for each pass, then the step counter.
- **`setParams({ relaxation?, compliance?, maxPush?, damping? })`** uploads 16 bytes; no recompile.
  Increasing `maxPush` is refused if it would violate the existing cell's completeness bound.
  `iterations`, `maxScanPerCell` and the cell size change the stages, so they are fixed at
  construction.
- **Switching off:** `contacts.enabled = false` disables its stages (fact 8); no chain changes.
- **`readCounters()`** resolves the grid's skipped entries and the bodies over `maxRadius` (from the
  last step), off the frame path, as `inspect` does.
- **Ownership:** the caller owns the records buffer, and `contacts` its grid and deltas. Destroy
  `contacts` after anything that runs its stages: an `AgentSystem` holding them, or the caller's
  sequence.
- **CPU mirror:** `resolveDiscContacts(layout, options, records, out?)` runs one step's passes over a
  `Float32Array`, with `buildPointGrid` as its grid. It is the definition the GPU is tested against
  (§10), and a reference for tools.

### Option B: WGSL builders for `addStepStage`, no seam

The entry exports WGSL that an app puts into two of its own step stages: gather (reading the
agents' 8 m grid: offsets and entries, plus a delta buffer, three caller buffers) and apply. No
agents change. It couples the physics entry to the agents' step-stage contract, cannot build its
own grid inside a step, and serves only agents; P1's bodies and plain buffers would need a second
API. It remains the zero-change fallback (§12).

### Option C: a pair list with stored multipliers (deferred)

A deterministic pair list (counted per body, prefix-summed, written in order) would allow XPBD's
multiplier to persist across passes, and later graph colouring (P3 §6.2). It costs a count pass, a
scan, a write pass and a pair buffer sized for the worst crowd, and needs a bound on pairs per body.
Filed until S9 shows that the reset multiplier is the limit.

**Why A over B:** one object serves agents, P1's bodies and any buffer; the agents change is a
three-line seam (+27 B) that knows nothing of physics; and the solver keeps its own small-cell
grid, which B cannot build.

---

## 6. The grid

**Completeness (C1).** Body $j$ is listed in the cell of $x_j^0$. Pass $k$ looks around
$\text{cell}(x_i^k)$. Between binning and pass $k$, a body moves at most $(K-1)m$. If
$\lVert x_i^k - x_j^k \rVert < \rho_i + \rho_j$, then
$\lVert x_i^k - x_j^0 \rVert < 2\rho_{\max} + (K-1)m \le h$. A Chebyshev distance at most $h$
means the cells differ by at most one on each axis, so $j$ is in the 3×3 block. No rebinning is
needed between passes; S8's rebinning was conservative. Clamping does not add to a body's move
while bodies start inside the world: clamping into an axis-aligned box is a projection onto a
convex set, which moves a point already inside it no farther than the push. A body the caller
leaves outside the world can jump in its first pass and miss contacts for that step.

**Options:**

| | Grid | Extra build per step | Candidates per body | Complete |
|---|---|---|---|---|
| **G1** | the agents' grid (8 m cells, built before act) | none | dense flocks fill 8 m cells: hundreds | if $2\rho_{\max} + $ act's move $+ (K-1)m \le$ 8 m; a respawn breaks it for one step |
| **G2** (recommended) | the solver's own, built after the caller's move, $h = 2\rho_{\max} + (K-1)m$ (raised to fit the device's cell limit) | one. The agents' 8 m grid builds in 0.04–0.08 ms at 4,096 and, at 65,536, 0.27 ms on NVIDIA and 1.2–1.4 ms on the AMD iGPU (`agent-grid.md`); smaller cells mean more cells to clear and scan | a few, unless bodies already overlap | always, within the bound |
| **G3** | rebuild before every pass (S8's) | $K$ | as G2 | always, with no margin |

**Recommendation: G2,** because it serves any buffer, holds few candidates where it matters (dense
crowds), and has no stale-grid condition. G1 is an optimization for agents. S9 compares G1 and G2
with GPU timestamps, and §16's rule decides whether G1 is also offered.

**The budget (C6).** A cell over $B$ yields a window of $B$ consecutive entries, its start hashed
from the cell, the solver's step counter and the pass. The grid's `overflow` counts the skipped
entries every build, as the agents' does. With G2, a cell over budget means bodies already overlap
heavily (a pile), the case where bounding the work matters most.

---

## 7. The solver

- **Prior art:** position-based crowds (Weiss, Litteneker, Jiang and Terzopoulos, 2017; extended
  in *Computers & Graphics*, 2019) treat agents as particles and project their positions out of
  collisions with PBD constraints, at interactive rates for up to about 100,000 agents. This design
  is the short-range (contact) part of that, made atomic-free and exact on one device.
- **Averaging** (constraint averaging, or mass splitting): dividing the summed corrections by the
  contact count keeps Jacobi from over-correcting a body with many contacts (P3 §4.3; Macklin et al.
  2014). The price is slow convergence in piles: a body with six contacts moves a sixth of their
  sum, as S8 saw.
- **Relaxation** $\omega$ multiplies the averaged correction. $\omega = 1$ is S8's. Over-relaxation
  ($1 < \omega < 2$) speeds convergence; too high, it oscillates. S9 measures 1, 1.25, 1.5 and
  1.75.
- **The push limit** $m$ caps a pass's move. It stops coincident or deeply buried bodies from
  jumping, and it is part of C1's bound.
- **Coincident centres:** a direction hashed from the pair's indices, opposite for the two bodies,
  so a pile spawned on one point opens deterministically.
- **Mass:** `'area'` (S8's) makes a larger body harder to push, and `'equal'` treats all bodies alike.
  The default comes from S9. A per-body inverse mass word is deferred.
- **Fixed bodies** have inverse mass 0, so the other body of the pair takes the whole correction.
  For agents, `fixed` can name the frozen bit: a held agent does not move, as `frozen` already
  promises.
- **Compliance** $\alpha$ adds $\tilde\alpha$ to the denominator: soft contacts that leave part of
  the overlap each pass, for crowds that compress before they stop. §2 states its limit.
- **Velocity damping** $\beta$ removes the part of the velocity that points back into the push. In
  S8's runs over the $k = 8$ list, it lowered displacement reversals from 11.92% to 6.23% (NVIDIA)
  and from 11.98% to 8.08% (AMD). For steering agents it slows a body that presses into a crowd, and
  the act stage keeps the heading (fact 7).
- **Order and exactness:** cells in a fixed order, entries in the grid's order, one write per body
  per stage, no float atomics: C2.

---

## 8. The agents seam

| | Seam | Bytes (measured) | Reaches `step()` | Notes |
|---|---|---|---|---|
| **S-A** (recommended) | `addStepStage({ name, stages })`: stages built elsewhere, run after act in this position, every step | **+27 B** (21,495 B, exactly the ceiling) | yes, through the acting half | 3 lines; the app orders contacts among its stages by call order |
| **S-B** | expose the per-step stage array; the app builds its own chains | not measured | no: `step()` would need another path | moves frame logic into every app |
| **S-C** | none: option B's WGSL in two step stages | 0 | yes | couples the entries, agents only, no own grid |

**S-A as measured** (2026-10-18, U45's method: the agents entry bundled and gzipped, with the
change applied to a copy of the source): `wgsl` becomes optional, `stages` is added, `addStepStage`
stores a copy of the array and returns, and `rebuildChains` flattens the app stages. The agents
entry goes from 21,468 B to 21,495 B, with no byte left. Refusing a call that gives both
`wgsl` and `stages` would cost more; as measured, `stages` wins and `wgsl` is ignored. Q3 asks
whether to keep the ceiling or raise it for that refusal.

```ts
export interface AgentStepStageOptions {
    name: string;
    wgsl?: string;
    /** External compute stages run in place of a compiled step WGSL kernel. */
    stages?: readonly ComputeStageNode[];
    buffers?: readonly GPUBuffer[];
}
```

**Order matters, and stays the app's.** Contacts placed before a stage that detects touching bodies
let it see separated positions; placed after a stage that moves bodies (a respawn), they separate
those too. The guide will say which to choose for what.

---

## 9. Cost and the GPU timeout guard

Per step: the grid build, then $K$ × (gather + apply), then one counter dispatch. The build is
about 20 small dispatches at 65,536 bodies: clear, bin, the prefix sums, keys, and a histogram,
scan and scatter per radix pass.

- **Worst case per pass:** $N \cdot 9B$ candidates, each one `vec4` (position, radius), plus one
  flags word when `fixed` is declared. At 65,536 bodies and $B = 32$: 18.9M candidates.
- **Without a budget a pile is quadratic:** 65,536 bodies on one point would read $9 \times
  65{,}536^2$ candidates per pass, enough to hold a GPU past its reset. The budget is what makes a
  pile a slow separation rather than a lost device.
- **The library does not guard; apps do,** with measured cost, as plan-2 §7 requires. S9 measures
  the worst case the guide will quote.

---

## 10. Test suite

IDs continue from step 12 (U45, A36).

**Headless** (`test/physics.test.mjs`, in a new `npm run test:physics`):
- **U46 — the declaration and its refusals:** every §11 error fires and names the offender. The
  cell rule $h \ge 2\rho_{\max} + (K-1)m$ is enforced. Defaults resolve as S9 set them.
- **U47 — the mirror's mechanics:** C3 (one pair exactly touching, the weighted centre fixed, for
  both mass rules), C4 (fixed bodies bit-identical; dead ones inert), C5 (compliance leaves part of
  the overlap, and more with $\alpha$), C8 (clamp and push limit), damping removes only the inward
  part, and coincident pairs open in opposite directions.
- **U48 — completeness and the budget in the mirror:** on 64 random layouts, contact sets equal a
  brute-force search when no cell is over budget (C1). Over budget, the skipped count equals
  $\sum_c \max(0, n_c - B)$ per build, and the windows rotate.
- **U49 — stages, size and the seam:** every generated module parses, and every generated name it
  uses is declared (the declared-name check of `agents-api-example.test.mjs`). The stage array is
  cached, with no typed array allocated per frame. `addStepStage({ name, stages })` puts the stages
  after act in call order, in the chains for 1–3 steps and in `step()`'s halves. The physics entry
  gets a size line; an agents-only app carries no physics code; the agents entry meets Q3's ceiling.

**GPU** (`test/physics-gpu/run.js`, a new suite on Node/NVIDIA, Node/AMD and Chrome/AMD):
- **A37 — GPU against the mirror:** 4,096 random bodies, one step at $K = 1$ and $K = 2$, both mass
  rules, compliance 0 and > 0: positions within $10^{-5}$ of the scale, contact counts equal (C3,
  C4, C7).
- **A38 — completeness:** a dense layout checked against a brute-force GPU oracle (S8's): zero
  missed contacts. **Watched failing:** a cell below the bound misses contacts.
- **A39 — exact reruns:** two 600-step runs of 16,384 steering bodies with contacts, word for word
  (C2). **Watched failing:** with corrections applied in place inside the gather (no delta buffer),
  invocations race on each other's positions and the runs differ.
- **A40 — the budget:** 65,536 bodies on one point. The step finishes within §16's limit, skipped
  entries are counted, and a few hundred steps open the pile (C6).
- **A41 — inside an `AgentSystem`:** contacts as an app-stage group. One step per frame, three
  steps per frame and `step()` give identical records. Overlap falls against the same run without
  contacts, and frozen agents stay put.

---

## 11. Errors (verbatim when built)

| Where | Condition | Message |
|---|---|---|
| `DiscContacts` | stride not a positive multiple of 4 | `DiscContacts: stride <s> must be a positive multiple of 4.` |
| `DiscContacts` | records too small or not `STORAGE` | `DiscContacts: records must have STORAGE usage and hold capacity × stride floats.` |
| `DiscContacts` | flags offset inside floats 0–3 or past the record | `DiscContacts: <field> offset <o> must be a word of the record after its first four (4–<stride − 1>).` |
| `DiscContacts` | velocity offset inside floats 0–3 or not leaving room for 3 floats | `DiscContacts: velocity offset <o> must leave room for 3 floats (4–<stride − 3>).` |
| `DiscContacts` | `maxRadius`, `maxPush` or `dt` not positive | `DiscContacts: <field> <v> must be positive.` |
| `DiscContacts` | `iterations` outside 1–8 | `DiscContacts: iterations <k> must be an integer from 1 to 8.` |
| `DiscContacts` | `relaxation` outside (0, 2) | `DiscContacts: relaxation <ω> must be above 0 and below 2.` |
| `DiscContacts` | negative compliance or damping outside [0, 1] | `DiscContacts: compliance <α> must be at least 0.` / `DiscContacts: damping <β> must be from 0 to 1.` |
| `DiscContacts` | a requested cell below the bound | `DiscContacts: cellSize <h> is below 2 × maxRadius + (iterations − 1) × maxPush = <b>; contacts would be missed.` |
| `DiscContacts` | the grid exceeds the device | `DiscContacts: the grid needs <n> cells; raise cellSize or shrink the world.` |
| `DiscContacts` | use after `destroy()` | `DiscContacts: used after destroy().` |
| `addStepStage` | neither `wgsl` nor `stages` | today's message: `AgentSystem: step stage "<name>" must define fn step(i: u32, tick: u32).` |

---

## 12. Engine touchpoints

- **Core:** none.
- **Agents — required for contacts inside an `AgentSystem`:** the seam S-A, +27 B as measured (§8).
  - **Zero-change fallback:** option B's WGSL in two step stages over the agents' 8 m grid. Its cost:
    the physics entry learns the agents' step-stage contract, the grid cannot be the solver's own
    (dense flocks scan hundreds of candidates), and P1's bodies need another API.
- **Spatial:** `gridWindow` and `GRID_WINDOW_WGSL` move from `agents` to `spatial`; agents
  re-exports them. Nothing changes for agents users; the bundle is unchanged within a byte or two.
- **Build:** a `physics` entry in `build.js`, and `./physics` in `package.json`'s exports.
- **The physics series:** the README gains this phase's line. P3 is not built; if it is, its
  particle record should carry the radius at float 3 and inverse mass elsewhere, or declare a radius
  offset that this solver would then accept (deferred with 3D).

---

## 13. Build order

| Phase | Delivers | Exit |
|---|---|---|
| **13.0** ✅ | Spike S9 (§16), scripts only: the solver's options and both grids, measured with GPU timestamps on the three pairs | Complete 2026-10-18: results after §16; G2 chosen; no configuration met T1–T6 on every pair, so the trade-offs went to the user (Q5–Q8) |
| **13.1** ✅ | The entry and the CPU mirror (`contactMath.ts`): the declaration, its refusals, `resolveDiscContacts`; U46–U48; the physics size line | Reviewed and committed `8d0d1df`; U48 strengthened to 64 random layouts and actual window differences |
| **13.2** ✅ | `DiscContacts` on the GPU (own grid, gather, apply, params, counters); the `physics` GPU suite, A37–A40 | Green on Node/NVIDIA, Node/AMD and Chrome/AMD, zero WebGPU errors; undersized-grid and racing-gather negative controls, zero-correction mutation caught |
| **13.3** | The agents seam S-A; U49's seam checks; A41; the agents size line at Q3's ceiling | Green on the three pairs |
| **13.4** | `docs/api/physics.md` with a checked example; a cross-link from the agents guide; the physics README's line | The example runs and is watched failing |

### Deferred

- **G1, the agents' grid,** unless §16's rule adopts it.
- **A statistics stage** (deepest penetration, contacts per step) for overlays.
- **Walls and segments** (NAV), and per-body bounds such as arena tiles.
- **3D spheres** on a 3D grid (P2), and P3's particles with them.
- **Stored multipliers** (option C) and graph colouring (P3 §6.2).
- **Subgroup reductions** in the gather: detect, never require (P3 §10.2).
- **Asynchronous construction** (`createComputeStageAsync`), for apps that enable contacts after load.
- **A per-body inverse mass** read from a record word.

---

## 14. Where this refines `plan-2.md` and the physics series

| Source | This design |
|---|---|
| plan-2 step 13: "over the point grid" | The solver builds its own point grid after the caller's move, with cells sized for contacts. Borrowing the agents' grid is an option only if S9 shows it pays (§6) |
| plan-2: "XPBD compliance" | XPBD's projection with the multiplier reset each pass: exact at one pass, residual depending on the pass count above that, measured (§2) |
| plan-2: "Jacobi with a correction buffer, no float atomics, exact reruns" | As stated, with averaging and a relaxation factor (§7) |
| plan-2 (silent) | A per-cell scan budget, counted: the GPU timeout guard's library side (§9) |
| plan-2 (silent) | One seam in `AgentSystem`, +27 B (§8) |
| plan-2 §5, 2A's spikes | A new spike, S9, as phase 13.0 (§16), if the user accepts it |
| Physics series: P3's contacts read P2's pair list | Discs in XZ gather from the grid, because P2's `atomicAdd` compaction is unordered. P2 and P3 stay the 3D path |

---

## 15. Questions for the user

**Decided 2026-10-18:** The user accepted all recommended options ("ok you could incorporate those minor things and also accept recommended answer in this design doc").
- **Q1:** (A) Run spike S9 before building (phase 13.0).
- **Q2:** Accept targets T1–T6 as proposed (§16).
- **Q3:** (A) Keep the ceiling at 21,495 B while the built seam fits; raise only if needed.
- **Q4:** `DiscContacts`, XZ discs now.

**Decided 2026-10-18 after S9:** The user accepted the recommended options for Q5–Q7 and deferred Q8 for later ("ok you could review the updated design after spikes result and if i should move ahead with recommendations,also i m thinking the q8 for later that i have decided already").
- **Q5:** (A) Divide summed correction by $\sqrt{n}$ ($\sqrt{|C_i^k|}$).
- **Q6:** (A) Defaults $K = 2$, $\omega = 1.75$, $B = 32$, $\beta = 1$, `'area'`, $\alpha = 0$.
- **Q7:** (A) Accept targets as measured (jam compression, AMD/Chrome shares, arenas at 2.13%) and proceed to 13.1.
- **Q8:** Defer to scenarios / Step 14 (library unchanged).

The questions as asked:

5. **Q5. The correction's divisor.**
   - **(A) Divide the summed correction by $\sqrt{n}$** (recommended). T1's herds and T2 at two
     passes; 2.8–3.0% jam reversals, bounded by the push limit. §2's $|C_i^k|$ becomes
     $\sqrt{|C_i^k|}$.
   - **(B) Keep §2's division by $n$.** T1 and T2 then need four passes, 42–45% of a 16K step.
6. **Q6. The defaults.**
   - **(A) $K = 2$, $\omega = 1.75$, $B = 32$, $\beta = 1$, `'area'`, $\alpha = 0$** (recommended):
     the cheapest that clears herds and the 256 pile. The guide names four passes (about 40% of a
     16K step) for dense crowds, which cuts a pushing jam's severe overlaps from 50% to 2%, and
     $\alpha = 10^{-4}$ for a softer, steadier jam.
   - **(B) Four passes by default.**
7. **Q7. The targets S9 missed.**
   - **(A) Accept them as measured and go on to 13.1** (recommended). A pushing jam compresses (T3).
     Two passes cost 21–24% of a 16K step on NVIDIA and 25–34% on the AMD iGPU (T4). The arenas
     reach 2.13% (T1). The guide states all three.
   - **(B) Design a stronger solver first:** graph colouring or stored multipliers (§5 option C,
     P3 §6.2), measured in another spike.
8. **Q8. Catches and contacts.** With contacts on, bodies no longer come closer than the sum of
   their radii.
   - **(A) Leave it to scenarios** (recommended). Herd & Wolves sets its catch distance from the
     radii when contacts are on, in step 14's arena scenario and its films. The library is
     unchanged.
   - **(B) Add contact groups now:** a flags mask naming which bodies collide, so that one kind can
     pass through another.

The questions as asked before S9:

1. **Q1. Run spike S9 before building (phase 13.0)?**
   - **(A) Yes** (recommended): S8's prototype never cleared a pile, and its costs are wall-clock
     samples. S9 sets the passes, relaxation, budget, mass rule and grid from GPU timestamps on the
     three pairs.
   - **(B) No:** build with S8's settings ($K = 2$, $\omega = 1$, own grid) and tune later.
2. **Q2. S9's targets (§16).** Accept T1–T6 as proposed, or change them? They decide when the
   solver is good enough, and what goes back to you when it is not.
3. **Q3. The agents seam and the ceiling.** S-A measured +27 B, landing exactly on 21,495 B.
   - **(A)** Keep the ceiling while the built seam fits, and raise it only if it does not, with the
     exact bytes reported (recommended; step 12's rule).
   - **(B)** Raise it now to 21,600 B, which also pays for refusing a step stage that gives both
     `wgsl` and `stages`.
4. **Q4. The name and the scope:** `DiscContacts`, XZ discs now, with 3D spheres deferred to a 3D
   grid (recommended), or a 3D-sounding name such as `SphereContacts` now?

---

## 16. Phase 13.0 — spike S9: what contacts should be, measured

**Scripts only** (`test/spikes/contact-solver/`, `npm run spike:s9`). It runs on Node/NVIDIA,
Node/AMD and Chrome/AMD, and takes its turn on the GPU (plan-2 §7). No library code.

### What S9 builds in spike code

The §2 pass as gather and apply stages over a solver-owned `PointGrid`, and over the agents' grid,
with every §7 option a switch. The worlds are S8's: Herd & Wolves on the rule, at 4,096, 16,384 and
65,536 agents, and S7's 64 arenas, plus three tests:
- **piles:** 256 bodies in a 4 m × 4 m box (S8's) and 4,096 in 16 m × 16 m;
- **a jam:** 4,096 bodies steering at full throttle toward one point for 600 steps, a bottleneck's
  steady state;
- **an extreme pile:** 65,536 bodies spawned on one point, for the cost of a step.

### What S9 measures

| # | Question | Metric |
|---|---|---|
| 1 | Cost | GPU timestamps around the contact stages and around the whole step, each work repeated in one pass until it lasts at least 5 ms and stays under 300 ms (gotchas). At 4K, 16K and 64K on the three pairs; 262K and 1M on NVIDIA, for films |
| 2 | Ordinary herds | Over 600 steps: the share of bodies overlapping by more than 5% (worst and final), and the worst penetration |
| 3 | Piles | Steps until fewer than 5% of bodies overlap by more than 5%, capped at 180 (S8: never) |
| 4 | The jam | The steady share over 5%, the worst penetration, and displacement reversals (S8's jitter metric) |
| 5 | The budget | The extreme pile's step time and skipped entries at $B$ = 16, 32, 64 |
| 6 | Exact reruns | Two 300-step runs per world, word for word |
| 7 | Completeness | A brute-force GPU oracle over 4,096 bodies against the gather, at the cell rule and below it |

**Configurations,** searched in stages so the matrix stays small:
1. **Grid and budget,** at $K = 2$, $\omega = 1$: G1 against G2, $B \in \{16, 32, 64\}$.
2. **Convergence,** at the chosen grid and budget: $K \in \{1, 2, 4\}$ × $\omega \in \{1, 1.25,
   1.5, 1.75\}$.
3. **Finish,** at the chosen point: damping $\beta \in \{0, 1\}$, both mass rules, and compliance
   0 against two soft values.

### Targets (proposed; Q2)

| | Target |
|---|---|
| **T1** | Ordinary herds: at most 2% of bodies over 5% overlap at any step of 600 (S8's best: 4.10%), and a final worst penetration under 10% of the summed radii |
| **T2** | The 256-body pile below 5% within 180 steps (S8: none) |
| **T3** | The jam's steady share over 5% at most 10%, with displacement reversals no higher than S8's damped figures (6–8%) |
| **T4** | Contact stages at most 25% of a Herd & Wolves step at 16,384 agents, on every pair |
| **T5** | The extreme pile's contact stages under 100 ms per step on every pair, at the chosen budget |
| **T6** | Zero differing words in every rerun |

### What S9 decides

- **The grid:** G2, unless G1 is at least 25% cheaper per step at 16K and 64K on every pair while
  meeting T1–T3. Then both are offered, G2 the default.
- **The defaults** ($K$, $\omega$, $B$, $\beta$, mass): the cheapest configuration meeting T1–T6 on
  every pair.
- **If none meets T2 or T3,** the measured trade-offs go to the user before phase 13.1. No silent
  acceptance: S8's lesson is that more neighbours do not by themselves converge a pile.

---

## Spike S9 results — 2026-10-18

`npm run spike:s9` (`test/spikes/contact-solver/`) ran its seven parts on Node/NVIDIA (Turing),
Node/AMD and Chrome/AMD (GCN-5 iGPU), with zero WebGPU errors and no library code.
- **How costs were taken:** GPU timestamps around each sequence, repeated from a restored snapshot
  to at least 8 ms, median of 3–5 rounds. NVIDIA's absolute times moved up to fivefold between
  runs with its clocks, so only shares within one run are compared.
- **Quality agrees across the three pairs:** herds and piles to two decimals, the jam within one
  percentage point.
- **Evidence:** compact results in `test/spikes/contact-solver/results.json`; raw files in its
  ignored `out/`.

### Checks (all three pairs)

- **C1:** at the cell rule, both passes find every contact brute force finds, in a dense pile (up
  to 74 contacts per body) and in a loose crowd of large discs. **Watched failing:** with cells of
  $2\rho_{\max}$, the loose crowd's second pass misses 57 contacts (the dense pile 1).
- **C2 (T6):** a 4,096 herd, a 256 pile and a 4,096 jam, two 300-step runs each, word for word, at
  S8's settings (NVIDIA) and at the candidate below (all three pairs).
- **C3:** one pair ends 1.0 m apart within $2 \times 10^{-8}$, each disc moving its share.

### The grid

Contact stages' GPU time as a share of the herd's own step, from flocked states after 600 steps,
$K = 2$:

| Grid | NVIDIA 16K | NVIDIA 64K | AMD 16K | AMD 64K | Chrome 16K | Chrome 64K |
|---|---:|---:|---:|---:|---:|---:|
| G1, the agents' 8 m grid, exhaustive | 65.0% | 60.7% | 60.0% | 92.7% | 51.9% | 95.0% |
| G1, $B = 64$ | 68.6% | 55.3% | 56.6% | 95.0% | 45.2% | 88.5% |
| G2, its own grid, $B$ = 16, 32, 64 | 25.1–25.2% | 12.8–13.1% | 26.7–27.6% | 18.4–20.2% | 26.9–28.8% | 19.7–20.9% |

G2 is 1.6–5.2 times cheaper everywhere, so by §16's rule **G2 alone; G1 is not offered.** The
budget does not change a herd's cost.

### Passes and relaxation, with §2's correction

$B = 32$, $\beta = 1$, `'area'`, $\alpha = 0$. The same on the three pairs:

| $K$ | $\omega$ | Herd: worst share over 5% | 256 pile clear at step | Jam: steady share over 5% (over 20%) |
|---:|---:|---:|---:|---:|
| 1 | 1–1.75 | 9.03–7.52% | never (87–70% left) | 94–91% (75–70%) |
| 2 | 1 | 4.13% | never (66% left) | 91% (67%) |
| 2 | 1.75 | 2.03% | never (19% left) | 87% (55%) |
| 4 | 1 | 1.27% | never (11% left) | 86% (50%) |
| 4 | 1.5 | 0.34% | 138 | 81% (37%) |
| 4 | 1.75 | 0.29% | 114 | 79% (31%) |

Contact stages at 16,384, as a share of a step (NVIDIA / AMD / Chrome): $K = 1$: 21.0 / 24.1 /
25.5%; $K = 2$: 24.3 / 27.8 / 33.1%; $K = 4$: 42.1 / 42.7 / 44.7%.

### Past the search's grid

| Point | Herd | 256 pile | Jam over 5% (over 20%) | 16K share (NVIDIA / AMD / Chrome) |
|---|---:|---:|---:|---:|
| $K = 2$, $\omega = 1.75$, correction divided by $\sqrt{n}$ | 1.81% | 99 | 83% (50%) | 21.0 / 28.0 / 27.1% |
| $K = 4$, $\omega = 1.5$, divided by $\sqrt{n}$ | 0.34% | 63 | 59% (2%) | 39.6 / 38.8 / 39.6% |
| $K = 8$, $\omega = 1.5$ | 0.00% | 69 | 64% (6%) | 97.0 / 80.9 / 86.5% |
| $K = 4$, $\omega = 1.75$, $m = 0.5$ | 0.10% | 117–120 | 78% (31–32%) | 48.1 / 41.5 / 40.9% |
| $K = 2$, $\omega = 1$, summed, not divided | 2.81% | 96–102 | 84% (54%) | 22.2 / 27.9 / 31.5% |

Dividing the summed correction by $\sqrt{n}$ rather than $n$ (the contact count) reaches T1's herds
and T2 at two passes. It reverses 2.8–3.0% of the jam's displacements, against 6–8% for S8's
damped prototype, and the push limit bounds every move.

### Damping, mass and compliance

At $K = 2$, $\omega = 1.75$, divided by $\sqrt{n}$:
- **Damping is essential.** Without it, herds reach 7.6–9.6% over 5%, and the jam reverses 8–29%
  of its moves.
- **The mass rules are alike:** 1.81% for both; jam reversals 2.8–3.0% (`'area'`) against 3.5–3.7%
  (`'equal'`).
- **Compliance $10^{-4}$** gives herds 2.03% and the 256 pile at 111–123 steps, but halves the jam's
  severe share (50% to 26–28% over 20%) with no reversals. $10^{-3}$ is too soft: herds 4.4–15%.

### Bodies on one point (T5)

- 65,536 bodies on one point, at $B$ = 16 to 64: 7.2–9.7 ms of contact stages on NVIDIA, 3.2–4.7 ms
  on AMD, 4.0–4.7 ms in Chrome.
- Exhaustive scans already cost 15–34 ms for 4,096 bodies on a point; at 65,536 they are predicted
  at 3.9–8.7 s, past Windows' reset. **The budget is required, not optional.**
- A 4,096-body point opens slowly: 95% of bodies are still over 5% after 300 steps.

### The candidate in the worlds

$K = 2$, $\omega = 1.75$, divided by $\sqrt{n}$, $B = 32$, $\beta = 1$, `'area'`, $\alpha = 0$; the
range covers the three pairs. The arenas run on the rule.

| World | Worst share over 5%: none → contacts | Mean nearest prey | Catches in 600 steps |
|---|---|---|---|
| Herd, 4,096 | 62.9–64.5% → 1.81% | 0.72–0.74 → 1.03–1.04 m | 429–440 → 2 |
| Herd, 16,384 | 58.7–59.3% → 1.82% | 0.78–0.79 → 1.05 m | 1,740–1,757 → 14 |
| 64 arenas | 64.1–64.2% → 2.13% | 0.73 → 0.96–0.97 m | 1,915–1,930 → 13 |

Contact stages as a share of a step (NVIDIA / AMD / Chrome):
- 4,096: 36.1 / 21.2 / 20.1%;
- 16,384: 23.9 / 33.5 / 24.8%;
- 65,536: 18.2 / 17.2 / 20.5%;
- 262,144 (NVIDIA): 9.8%;
- 1,048,576 (NVIDIA): 37.4%. There the radix sort's dispatch limit forces 5.53 m cells: buckets
  times blocks of 256 bodies must stay under 65,535 × 64.

### Against the targets

| | $K = 2$, $\div\sqrt{n}$ | $K = 4$, $\div\sqrt{n}$ | $K = 4$, $\omega = 1.5$, §2 as written |
|---|---|---|---|
| **T1** herds at most 2% | ✓ 1.81–1.82%; arenas 2.13% ✗ | ✓ 0.34% | ✓ 0.34% |
| **T2** 256 pile in 180 steps | ✓ 99 | ✓ 63 | ✓ 138 |
| **T3** jam at most 10% | ✗ 83% | ✗ 59% | ✗ 81% |
| **T4** at most 25% at 16K | NVIDIA ✓ 21–24%; AMD ✗ 28–34%; Chrome ✗ 25–33% | ✗ about 40% | ✗ 42–45% |
| **T5** one point under 100 ms | ✓ | ✓ by estimate (twice $K = 2$'s 3–10 ms) | ✓ by estimate |
| **T6** exact reruns | ✓ | not run | not run |

### What it shows

- **No configuration meets T1–T6 on every pair,** so the trade-offs go to the user (§15, Q5–Q8).
- **The grid is settled:** G2.
- **A jam that keeps pushing compresses at every affordable setting.** Each Jacobi pass carries a
  correction about one contact layer, and a 4,096-body jam is dozens of layers deep. More passes
  and the $\sqrt{n}$ division cut the severe share (over 20%) to 2%; nothing tested clears the
  mild one.
- **Contacts change scenarios.** With bodies kept 1.0 m apart (radii 0.6 + 0.4), Herd & Wolves'
  0.9 m catch almost never happens: 1,750 catches fall to 14 at 16,384.
- **At a million bodies, the device sets the cell,** not the bodies, and contacts cost a third of a
  step.
