// CPU definitions for the shared XZ point grid (agents-design §6, physics P2 stages A–C).
// Tests only: no readback or CPU grid rebuild belongs on the GPU frame path.

export interface PointGridLayout {
    readonly world: { readonly minX: number; readonly minZ: number };
    readonly cellSize: number;
    readonly grid: { readonly cellsX: number; readonly cellsZ: number; readonly cellTotal: number };
}

/** X-major cells; the closed maximum face and out-of-field points clamp to the edge cell. */
export function pointCell(layout: PointGridLayout, x: number, z: number): number {
    const { cellsX, cellsZ } = layout.grid;
    const cx = Math.max(0, Math.min(cellsX - 1, Math.floor((x - layout.world.minX) / layout.cellSize)));
    const cz = Math.max(0, Math.min(cellsZ - 1, Math.floor((z - layout.world.minZ) / layout.cellSize)));
    return cz * cellsX + cx;
}

/** Exclusive offsets, including a terminal total so even the last cell has a range. */
export function prefixOffsets(counts: Uint32Array): Uint32Array {
    const offsets = new Uint32Array(counts.length + 1);
    let sum = 0;
    for (let i = 0; i < counts.length; i++) {
        sum += counts[i];
        if (sum > 0xffffffff) throw new Error('SpatialGrid: prefix total exceeds u32 capacity.');
        offsets[i + 1] = sum;
    }
    return offsets;
}

export interface PointGridMirror {
    readonly cellCounts: Uint32Array;
    readonly cellOffsets: Uint32Array;
    readonly cellBodies: Uint32Array;
    /** Excess entries over the per-cell scan budget, counted once per grid build. */
    readonly overflow: number;
}

/** Bin → exclusive prefix → scatter. Stride is in floats; position starts at offset zero. */
export function buildPointGrid(
    layout: PointGridLayout, data: Float32Array, stride: number, count: number,
    alive: (index: number) => boolean, maxScanPerCell: number,
): PointGridMirror {
    if (!(Number.isInteger(stride) && stride >= 3 && Number.isInteger(count) && count >= 0 &&
        count * stride <= data.length && Number.isInteger(maxScanPerCell) && maxScanPerCell > 0)) {
        throw new Error('SpatialGrid: invalid record stride, count or scan budget.');
    }
    const cellCounts = new Uint32Array(layout.grid.cellTotal);
    const cells = new Int32Array(count).fill(-1);
    for (let i = 0; i < count; i++) {
        if (!alive(i)) continue;
        const x = data[i * stride], z = data[i * stride + 2];
        if (!(Number.isFinite(x) && Number.isFinite(z))) throw new Error(`SpatialGrid: point ${i} is non-finite.`);
        const cell = pointCell(layout, x, z);
        cells[i] = cell;
        cellCounts[cell]++;
    }
    const cellOffsets = prefixOffsets(cellCounts);
    const cellBodies = new Uint32Array(cellOffsets[cellCounts.length]);
    const cursors = cellOffsets.slice(0, -1);
    for (let i = 0; i < count; i++) {
        if (cells[i] >= 0) cellBodies[cursors[cells[i]]++] = i;
    }
    let overflow = 0;
    for (const n of cellCounts) overflow += Math.max(0, n - maxScanPerCell);
    return { cellCounts, cellOffsets, cellBodies, overflow };
}