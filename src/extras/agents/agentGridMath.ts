// CPU grid-query reference. Unlike nearestNeighbours (brute force), visits only the 3×3 stencil.
import { type PointGridMirror, pointCell } from '../spatial/gridMath';
import { AGENT_FLOATS, AGENT_FLAGS_OFFSET, AGENT_FLAG_ALIVE, AGENT_SPECIES_OFFSET, MAX_NEAREST_K,
    type AgentDefinition } from './agentLayout';
import { type RecordViews } from './agentMath';

/** Exact below the scan budget; overflowing cells intentionally expose the GPU's bounded scan. */
export function gridNearestNeighbours(
    def: AgentDefinition, rec: RecordViews, grid: PointGridMirror, i: number,
    mask: number, radius: number, k: number, out: { [n: number]: number },
): number {
    if (!(Number.isFinite(radius) && radius >= 0 && radius <= def.cellSize &&
        Number.isInteger(k) && k >= 1 && k <= MAX_NEAREST_K)) {
        throw new Error('AgentSystem: grid query radius must fit cellSize and k must fit MAX_NEAREST_K.');
    }
    const cell = pointCell(def, rec.data[i * AGENT_FLOATS], rec.data[i * AGENT_FLOATS + 2]);
    const { cellsX, cellsZ } = def.grid;
    const cx = cell % cellsX, cz = Math.floor(cell / cellsX);
    const distances = new Float64Array(k);
    let found = 0;
    for (let z = Math.max(0, cz - 1); z <= Math.min(cellsZ - 1, cz + 1); z++) {
        for (let x = Math.max(0, cx - 1); x <= Math.min(cellsX - 1, cx + 1); x++) {
            const c = z * cellsX + x, start = grid.cellOffsets[c];
            const end = start + Math.min(grid.cellCounts[c], def.maxScanPerCell);
            for (let at = start; at < end; at++) {
                const j = grid.cellBodies[at], b = j * AGENT_FLOATS;
                if (j === i || !(rec.u32[b + AGENT_FLAGS_OFFSET] & AGENT_FLAG_ALIVE) ||
                    !((mask >>> rec.u32[b + AGENT_SPECIES_OFFSET]) & 1)) continue;
                const dx = rec.data[b] - rec.data[i * AGENT_FLOATS];
                const dz = rec.data[b + 2] - rec.data[i * AGENT_FLOATS + 2];
                const d2 = dx * dx + dz * dz;
                if (d2 > radius * radius) continue;
                let p = 0;
                while (p < found && (distances[p] < d2 || (distances[p] === d2 && out[p] < j))) p++;
                if (p === k) continue;
                for (let n = Math.min(found, k - 1); n > p; n--) {
                    distances[n] = distances[n - 1]; out[n] = out[n - 1];
                }
                distances[p] = d2; out[p] = j;
                found = Math.min(k, found + 1);
            }
        }
    }
    return found;
}