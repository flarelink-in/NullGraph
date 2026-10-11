// Shared XZ point-grid kernels. Hosts prebuild the scan hierarchy and clear counts/cursors/overflow
// before binning. All records stay on device; scatter retains every live index, even on overflow.
import { type PointGridLayout } from './gridMath';

export const GRID_SCAN_BLOCK = 256;

export function pointCellWGSL(layout: PointGridLayout): string {
    const { cellsX, cellsZ } = layout.grid;
    return `fn pointCell(p: vec3<f32>) -> u32 {
    let c = clamp(vec2<i32>(floor((p.xz - vec2<f32>(${layout.world.minX}, ${layout.world.minZ})) / ${layout.cellSize})),
        vec2<i32>(0), vec2<i32>(${cellsX - 1}, ${cellsZ - 1}));
    return u32(c.y) * ${cellsX}u + u32(c.x);
}`;
}

/** Flags are a word offset, not a float value; -1 bins every record (e.g. a body set). */
export function buildGridBinWGSL(layout: PointGridLayout, count: number, stride: number,
    flagsOffset: number, aliveMask: number, budget: number): string {
    return `
@group(0) @binding(0) var<storage, read> records: array<u32>;
@group(0) @binding(1) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> overflow: atomic<u32>;
${pointCellWGSL(layout)}
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= ${count}u) { return; }
    let b = i * ${stride}u;
    ${flagsOffset < 0 ? '' : `if ((records[b + ${flagsOffset}u] & ${aliveMask}u) == 0u) { return; }`}
    let p = vec3<f32>(bitcast<f32>(records[b]), bitcast<f32>(records[b + 1u]), bitcast<f32>(records[b + 2u]));
    let previous = atomicAdd(&counts[pointCell(p)], 1u);
    if (previous >= ${budget}u) { atomicAdd(&overflow, 1u); }
}`;
}

/** One exclusive scan per block, plus block totals. Recurse on totals, then add back top-down.
 * Offsets have length count+1; the add pass also corrects the terminal total. */
export function buildGridScanWGSL(count: number): string {
    return `
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> totals: array<u32>;
var<workgroup> values: array<u32, ${GRID_SCAN_BLOCK}>;
@compute @workgroup_size(${GRID_SCAN_BLOCK}) fn cs_main(
    @builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
    let lane = local.x;
    let i = group.x * ${GRID_SCAN_BLOCK}u + lane;
    var value = 0u;
    if (i < ${count}u) { value = input[i]; }
    values[lane] = value;
    workgroupBarrier();
    for (var step = 1u; step < ${GRID_SCAN_BLOCK}u; step *= 2u) {
        var add = 0u;
        if (lane >= step) { add = values[lane - step]; }
        workgroupBarrier();
        values[lane] += add;
        workgroupBarrier();
    }
    if (i < ${count}u) { offsets[i] = values[lane] - value; }
    if (lane == ${GRID_SCAN_BLOCK - 1}u) { totals[group.x] = values[lane]; }
    if (i == ${count - 1}u) { offsets[${count}u] = values[lane]; }
}`;
}

export function buildGridScanAddWGSL(count: number): string {
    return `
@group(0) @binding(0) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(1) var<storage, read> blocks: array<u32>;
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i > ${count}u) { return; }
    let block = min(i, ${count - 1}u) / ${GRID_SCAN_BLOCK}u;
    offsets[i] += blocks[block];
}`;
}

export function buildGridScatterWGSL(layout: PointGridLayout, count: number, stride: number,
    flagsOffset: number, aliveMask: number): string {
    return `
@group(0) @binding(0) var<storage, read> records: array<u32>;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> cursors: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> bodies: array<u32>;
${pointCellWGSL(layout)}
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= ${count}u) { return; }
    let b = i * ${stride}u;
    ${flagsOffset < 0 ? '' : `if ((records[b + ${flagsOffset}u] & ${aliveMask}u) == 0u) { return; }`}
    let p = vec3<f32>(bitcast<f32>(records[b]), bitcast<f32>(records[b + 1u]), bitcast<f32>(records[b + 2u]));
    let cell = pointCell(p);
    bodies[offsets[cell] + atomicAdd(&cursors[cell], 1u)] = i;
}`;
}