# GPU agent grid — C1 step 2

Import `AgentGrid` / `PointGrid` from `null-graph/agents`; debug draws are opt-in
from `null-graph/debug-ui`. This is grid construction and querying, not yet the complete
sense → think → act `AgentSystem` (C1 step 3 onward).

```ts
import { AgentGrid } from 'null-graph/agents';
import { groundGrid, gridOccupancy } from 'null-graph/debug-ui';

// engine is initialized, definition comes from defineAgents(), and recordsBuffer is a
// caller-owned STORAGE buffer containing capacity × 64-byte agent records.
const grid = new AgentGrid(engine, recordsBuffer, definition, 0xffffffff, definition.cellSize, 8);
const pass = engine.createPass({ name: 'Grid debug', isMainScreenPass: true });
const floor = groundGrid(engine, pass, {
    min: [definition.world.minX, definition.world.minZ],
    max: [definition.world.maxX, definition.world.maxZ],
});
const occupancy = gridOccupancy(engine, pass, grid);
const sequence = [...grid.getComputeStages(), pass]; // Construct once, not every frame.

// In the host frame loop, after camera/spawn uploads:
engine.render(sequence);

// During teardown, remove debug draws before releasing the grid they read.
occupancy.destroy();
floor.destroy();
grid.destroy(); // Does not destroy recordsBuffer, engine or device.
```

## Construction and scheduling

`new AgentGrid(engine, records, definition, mask = 0xffffffff, radius = cellSize, k = 8)`
builds all buffers, pipelines, bind groups and stages once. `getComputeStages()` returns the same
array each time: reset → bin → hierarchical exclusive scan/add-back → scatter → query.
Repeat that array for multiple simulation steps, inserting movement stages between builds as needed.
The reset is GPU work, not queue-side `preClearBuffers`: two builds in one submission reset at
their actual execution points. No readback, pipeline creation, typed-array construction or CPU
occupancy update is needed in the frame path.

`PointGrid` exposes the shared construction without the agent query. Its options extend
`PointGridLayout` with `capacity`, `stride` (32-bit words), `maxScanPerCell`, optional
`flagsOffset` (default -1: all records live) and `aliveMask` (default 1). Position XYZ starts at
word zero. This is an XZ **point** grid; it does not yet implement physics sphere multi-cell
binning, pair generation or a broadphase solver. Input positions must be finite. Capacity and
layout are fixed: do not mutate the supplied definition/options after construction.

## Buffers and query contract

| Property | Contents |
|---|---|
| `cellCounts` | `cellTotal` u32 counts of live records |
| `cellOffsets` | Exclusive offsets, **cellTotal + 1** words including terminal live total |
| `cellBodies` | Capacity u32 indices; only `[0, terminalTotal)` is valid |
| `overflow` | One u32: sum of entries above each cell's scan budget, reset every build |
| `neighbours` (`AgentGrid`) | Capacity × k u32 indices, nearest-first; unused ranks = `0xffffffff` |

Queries visit a 3×3 cell neighbourhood, exclude self/dead candidates, apply the species mask,
include the radius boundary, and sort by squared XZ distance then lower index. Radius must be
finite, non-negative and ≤ cellSize; k is 1–8. Dead **query rows** follow the CPU grid-query
reference (can contain neighbours); future sensing skips dead rows. CPU uses double arithmetic,
WGSL uses f32: extremely close distance ordering/boundaries can differ due to floating-point
rounding. Ordinary exactly representable ties use the same index tie-break.

Scatter stores **all** live indices. Overflow limits only how many candidates each queried cell
scans. Atomic scatter order is unspecified; over-budget subsets can differ across builds/devices.
Do not interpret overflow queries as complete or deterministic neighbour sets. Buffers have
COPY_SRC for test/inspector copies; normal rendering/querying stays on-device.

## Debug draws

`groundGrid(engine, pass, options)` draws minor/major anti-aliased floor lines, field edge and
distance fade. Options: required XZ `min` / `max`; optional `height` (0), `minorSpacing` (2),
`majorSpacing` (10), `fadeStart` (30), `fadeEnd` (110). The floor line/fade logic is extracted
from the site preview shader, without changing the existing site demos.

`gridOccupancy(engine, pass, grid, height = 0.01)` draws the grid extent using `cellCounts`
directly in the fragment shader: empty cells dark, occupancy blue, over-budget cells red.
Draw after the grid stages. Both return `{ batch, destroy() }`; they use ordinary batches,
not visibility/culling systems. Update the engine camera before rendering. `destroy()` is
idempotent and unregisters/releases the draw's own resources. Destroying a grid disables its
cached stages; remove those stages from the host sequence during teardown. The engine/device
and source record buffer remain caller-owned.

## Validation

Build first: `npm run build:lib`. Then:

- `npm run test:agents`: CPU record/sensing/steering/layout and grid references (29 checks).
- `npm run test:gpu -- --only agents`: Node/Dawn real-device grid and drawing checks.
- `npm run test:gpu:browser -- --only agents`: same checks in headless Chrome.

The GPU suite tests prefix sizes 1/255/256/257/1025/65537, four 10,000-agent layouts,
species masks/deaths/ties, full membership under overflow, stable stages, multiple builds per
submission, empty rebuilds, input refusals and teardown. Image assertions verify floor variation,
red overflow occupancy, and removal of red after an empty rebuild. Set `NG_GPU_ARTIFACTS` to an
output directory for floor/overflow PNGs on the Node host. Readbacks occur only in the harness.
These are correctness checks, **not** a kernel-performance benchmark; S3 remains the next gate.