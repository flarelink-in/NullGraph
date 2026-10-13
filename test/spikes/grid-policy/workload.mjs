// Step 5, phase 5.7 (docs/agents-ordered-grid-design.md): the acceptance workload, the crowded-cell
// policies run.js compares, and a CPU check of what each policy must return.
// Test-only: nothing here is a production option.
import * as A from '../../../dist/agents.js';

// ---- Copied verbatim from test/spikes/grid-order/run.js, which stays unchanged because it is the
// ---- acceptance measurement. workload.test.mjs fails if the two drift apart.
const BASE_HALF = 80, CELL = 8, WOLF_SHARE = 16;
const RULES = `fn think(i: u32, species: u32, tick: u32) {
    let wall = -3.0 * obs(i, OBS_BOUNDS + BOUNDS_SIDE);
    if (species == SPECIES_WOLF) {
        setAct(i, ACT_TURN, clamp(sign(obs(i, OBS_PREY_0 + NEAR_SIDE)) + wall, -1.0, 1.0));
        setAct(i, ACT_THROTTLE, 1.0);
        setReason(i, 2u);
    } else {
        let near = obs(i, OBS_WOLF_0 + NEAR_PRESENT);
        setAct(i, ACT_TURN, clamp(-sign(obs(i, OBS_WOLF_0 + NEAR_SIDE)) * near + wall, -1.0, 1.0));
        setAct(i, ACT_THROTTLE, 0.4 + 0.6 * near);
        setReason(i, 1u);
    }
}`;

const half = count => BASE_HALF * Math.sqrt(count / 4096);
const herdDef = (count, budget) => A.defineAgents({ capacity: count,
    world: { min: [-half(count), -half(count)], max: [half(count), half(count)] }, cellSize: CELL, maxScanPerCell: budget,
    species: [
        { name: 'prey', maxSpeed: 3.2, maxTurn: 4, accel: 6, senseRadius: CELL, radius: 0.4 },
        { name: 'wolf', maxSpeed: 4.2, maxTurn: 3, accel: 5, senseRadius: CELL, radius: 0.6 },
    ],
    sensors: [A.Sensors.self(), A.Sensors.nearest({ name: 'wolf', species: ['wolf'], k: 1 }),
        A.Sensors.nearest({ name: 'prey', species: ['prey'], k: 3 }), A.Sensors.bounds()] });

// spread: the demo's spawn. flocked: herds of 256 around random centres, a 4 m Gaussian each.
function spawn(agents, count, layout, seed) {
    const inset = half(count) - 1, herds = count / 256, r = A.rand01;
    for (let i = 0; i < count; i++) {
        let x = (r(i, seed, 0) * 2 - 1) * inset, z = (r(i, seed, 1) * 2 - 1) * inset;
        if (layout === 'flocked') {
            const h = i % herds, cx = (r(h, seed, 5) * 2 - 1) * (inset - 12), cz = (r(h, seed, 6) * 2 - 1) * (inset - 12);
            const radius = 4 * Math.sqrt(-2 * Math.log(1 - r(i, seed, 3))), angle = 2 * Math.PI * r(i, seed, 4);
            x = Math.max(-inset, Math.min(inset, cx + radius * Math.cos(angle)));
            z = Math.max(-inset, Math.min(inset, cz + radius * Math.sin(angle)));
        }
        agents.spawn(i, { position: [x, 0.08, z], species: i < count / WOLF_SHARE ? 'wolf' : 'prey',
            heading: r(i, seed, 2) * Math.PI * 2 - Math.PI });
    }
    agents.upload();
}
// ---- End of the copy.
export { RULES, herdDef, spawn };

export const PAD = 0xffffffff;

/** The nearest sensors in declaration order, each with the slice of the neighbour list it owns. */
export function nearestSensors(def) {
    const out = [];
    let base = 0;
    for (const s of def.sensors) {
        if (s.kind !== 'nearest') continue;
        out.push({ name: s.name, k: s.k, mask: def.nearestMasks[out.length], base });
        base += s.k;
    }
    return out;
}

// Phase 5.6's scan of one cell (agentGridShaders.ts at da262ad), and the same scan with nothing left
// to recompute per candidate: the cell's offset is read once, and one index walks the cell and wraps.
const SCAN = new RegExp([
    String.raw`let count = counts\[c\];`,
    String.raw`var start = 0u;`,
    String.raw`if \(count > (\d+)u\) \{ start = gridWindow\(i, tick\) % count; \}`,
    String.raw`for \(var s = 0u; s < min\(count, \1u\); s\+\+\) \{`,
    String.raw`var at = start \+ s;`,
    String.raw`if \(at >= count\) \{ at -= count; \}`,
    String.raw`let j = bodies\[offsets\[c\] \+ at\];`,
].join(String.raw`\s*`));
const TIGHT = `let count = counts[c];
            let first = offsets[c];
            let stop = first + count;
            var at = first;
            if (count > $1u) { at += gridWindow(i, tick) % count; }
            for (var left = min(count, $1u); left > 0u; left--) {
                let j = bodies[at];
                at++;
                if (at == stop) { at = first; }`;

/**
 * One nearest sensor's query under a policy. Which entries a crowded cell yields:
 *   first  its first entries: the query before step 5 (`earlier.before`, b7050b7), unmodified
 *   agent  a window that rotates per querying agent and tick: phase 5.6's query (`earlier.window`,
 *          da262ad), unmodified
 *   cell   a window that rotates per cell and tick: every query of a cell reads the same entries
 * cell+ and agent+ scan the same entries in the same order as cell and agent, from a tighter loop.
 * cell+ is the query production ships since phase 5.7: workload.test.mjs holds the two texts equal.
 */
export function policyQueryWGSL(policy, def, sensor, stride, earlier) {
    const args = [def, sensor.mask, def.cellSize, sensor.k, { stride, base: sensor.base }];
    if (policy === 'first') return earlier.before(...args);
    const code = earlier.window(...args);
    if (policy === 'agent') return code;
    const perCell = text => text.replace('gridWindow(i, tick) % count', 'gridWindow(c, tick) % count');
    const tight = text => text.replace(SCAN, TIGHT);
    const changes = { cell: [perCell], 'cell+': [tight, perCell], 'agent+': [tight] }[policy];
    if (!changes) throw new Error(`Unknown policy ${policy}`);
    return changes.reduce((text, change) => {
        const next = change(text);
        if (next === text) throw new Error('Phase 5.6 query text not as expected; rerun prepare.mjs.');
        return next;
    }, code);
}

/** The cell the GPU puts a point in: pointCellWGSL's arithmetic, in f32. */
export function gpuCell(def, x, z) {
    const { cellsX, cellsZ } = def.grid, f = Math.fround;
    const cx = Math.max(0, Math.min(cellsX - 1, Math.floor(f(f(x - def.world.minX) / def.cellSize))));
    const cz = Math.max(0, Math.min(cellsZ - 1, Math.floor(f(f(z - def.world.minZ) / def.cellSize))));
    return cz * cellsX + cx;
}

/** What a stable sort by (cell, index) must produce, with each cell computed as the GPU computes it. */
export function sortedGrid(def, rec) {
    const F = A.AGENT_FLOATS, cellCounts = new Uint32Array(def.grid.cellTotal), cells = new Int32Array(def.capacity).fill(-1);
    for (let i = 0; i < def.capacity; i++) {
        if (!(rec.u32[i * F + A.AGENT_FLAGS_OFFSET] & A.AGENT_FLAG_ALIVE)) continue;
        cells[i] = gpuCell(def, rec.data[i * F], rec.data[i * F + 2]);
        cellCounts[cells[i]]++;
    }
    const cellOffsets = A.prefixOffsets(cellCounts), cursors = cellOffsets.slice(0, -1);
    const cellBodies = new Uint32Array(cellOffsets[cellCounts.length]);
    for (let i = 0; i < def.capacity; i++) {
        if (cells[i] >= 0) cellBodies[cursors[cells[i]]++] = i;
    }
    return { cellCounts, cellOffsets, cellBodies };
}

/** The 3×3 cells around agent i, as the query visits them. */
function stencil(def, rec, i) {
    const { cellsX, cellsZ } = def.grid, cells = [];
    const cell = gpuCell(def, rec.data[i * A.AGENT_FLOATS], rec.data[i * A.AGENT_FLOATS + 2]);
    const cx = cell % cellsX, cz = Math.floor(cell / cellsX);
    for (let z = Math.max(0, cz - 1); z <= Math.min(cellsZ - 1, cz + 1); z++) {
        for (let x = Math.max(0, cx - 1); x <= Math.min(cellsX - 1, cx + 1); x++) cells.push(z * cellsX + x);
    }
    return cells;
}

/** How crowded a grid is: the cells over the scan budget, and the share of agents that scan one. */
export function crowding(def, rec, grid) {
    let cells = 0, largest = 0, affected = 0;
    for (const n of grid.cellCounts) {
        if (n > def.maxScanPerCell) cells++;
        largest = Math.max(largest, n);
    }
    for (let i = 0; i < def.capacity; i++) {
        if (stencil(def, rec, i).some(c => grid.cellCounts[c] > def.maxScanPerCell)) affected++;
    }
    return { cells, largest, agents: affected / def.capacity };
}

/**
 * Every entry a policy scans for agent i that passes the sensor's alive and species filters, as
 * index → squared distance. `grid` is the grid the query read (counts, offsets and bodies, in the
 * order the GPU holds them), so this checks the query, not the grid build.
 */
export function scanned(def, rec, grid, i, sensor, policy, tick) {
    if (!['first', 'cell', 'agent'].includes(policy)) throw new Error(`Unknown policy ${policy}`);
    const F = A.AGENT_FLOATS, x = rec.data[i * F], z = rec.data[i * F + 2], out = new Map();
    for (const c of stencil(def, rec, i)) {
        const count = grid.cellCounts[c];
        let start = 0;
        if (count > def.maxScanPerCell && policy !== 'first') start = A.gridWindow(policy === 'cell' ? c : i, tick) % count;
        for (let s = 0; s < Math.min(count, def.maxScanPerCell); s++) {
            const j = grid.cellBodies[grid.cellOffsets[c] + (start + s) % count], b = j * F;
            if (j === i || !(rec.u32[b + A.AGENT_FLAGS_OFFSET] & A.AGENT_FLAG_ALIVE) ||
                !((sensor.mask >>> rec.u32[b + A.AGENT_SPECIES_OFFSET]) & 1)) continue;
            const dx = rec.data[b] - x, dz = rec.data[b + 2] - z;
            out.set(j, dx * dx + dz * dz);
        }
    }
    return out;
}

/**
 * Compares one GPU list with what its policy scans (`found`, from scanned()). The GPU ranks in f32
 * and this file in f64, so two neighbours whose distances round together may swap, and one on the
 * radius may drop out: those are 'tie'. Any other difference is 'wrong'.
 */
export function matches(found, gpu, k, r2) {
    const ranked = [...found].filter(([, d]) => d <= r2).sort((a, b) => a[1] - b[1] || a[0] - b[0]);
    let exact = true;
    for (let r = 0; r < k; r++) {
        if (gpu[r] !== (r < ranked.length ? ranked[r][0] : PAD)) exact = false;
    }
    if (exact) return 'exact';
    const near = (a, b) => Math.abs(a - b) <= 1e-4 * Math.max(1, b), seen = new Set();
    for (let r = 0; r < k; r++) {
        const j = gpu[r], want = r < ranked.length ? ranked[r][1] : Infinity;
        if (j === PAD) {
            if (want !== Infinity && !near(want, r2)) return 'wrong';
            continue;
        }
        const got = found.get(j);
        if (got === undefined || seen.has(j)) return 'wrong';     // not an entry this policy scans, or listed twice
        seen.add(j);
        if (!near(got, want === Infinity ? r2 : want)) return 'wrong';
    }
    return 'tie';
}
