// Shared XZ point-grid kernels. Hosts prebuild scan/radix stages and clear counts/overflow.
// All records stay on device; stable sorting retains every live index, even on overflow.
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

/** Fewest passes, then the smallest table that covers the key including the dead sentinel. */
export function gridRadixLayout(cellTotal: number): { bits: number; passes: number; buckets: number } {
    const keyBits = Math.floor(Math.log2(cellTotal)) + 1;
    const passes = Math.ceil(keyBits / 10), bits = Math.ceil(keyBits / passes);
    return { bits, passes, buckets: 2 ** bits };
}

export function buildGridKeysWGSL(layout: PointGridLayout, count: number, stride: number,
    flagsOffset: number, aliveMask: number): string {
    return `
@group(0) @binding(0) var<storage, read> records: array<u32>;
@group(0) @binding(1) var<storage, read_write> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;
${pointCellWGSL(layout)}
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= ${count}u) { return; }
    let b = i * ${stride}u;
    var cell = ${layout.grid.cellTotal}u;
    ${flagsOffset < 0 ? '' : `if ((records[b + ${flagsOffset}u] & ${aliveMask}u) != 0u)`} {
        cell = pointCell(vec3<f32>(bitcast<f32>(records[b]), bitcast<f32>(records[b + 1u]), bitcast<f32>(records[b + 2u])));
    }
    keys[i] = cell;
    indices[i] = i;
}`;
}

export function buildGridHistogramWGSL(count: number, bits: number, shift: number): string {
    const buckets = 2 ** bits, blocks = Math.ceil(count / GRID_SCAN_BLOCK);
    return `
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> table: array<u32>;
var<workgroup> histogram: array<atomic<u32>, ${buckets}>;
@compute @workgroup_size(${GRID_SCAN_BLOCK}) fn cs_main(
    @builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
    let lane = local.x;
    for (var d = lane; d < ${buckets}u; d += ${GRID_SCAN_BLOCK}u) { atomicStore(&histogram[d], 0u); }
    workgroupBarrier();
    let i = group.x * ${GRID_SCAN_BLOCK}u + lane;
    if (i < ${count}u) { atomicAdd(&histogram[(keys[i] >> ${shift}u) & ${buckets - 1}u], 1u); }
    workgroupBarrier();
    for (var d = lane; d < ${buckets}u; d += ${GRID_SCAN_BLOCK}u) {
        table[d * ${blocks}u + group.x] = atomicLoad(&histogram[d]);
    }
}`;
}

export function buildGridStableScatterWGSL(count: number, bits: number, shift: number): string {
    const blocks = Math.ceil(count / GRID_SCAN_BLOCK);
    return `
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> nextKeys: array<u32>;
@group(0) @binding(4) var<storage, read_write> nextIndices: array<u32>;
var<workgroup> digits: array<u32, ${GRID_SCAN_BLOCK}>;
@compute @workgroup_size(${GRID_SCAN_BLOCK}) fn cs_main(
    @builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
    let lane = local.x;
    let i = group.x * ${GRID_SCAN_BLOCK}u + lane;
    var key = 0u;
    if (i < ${count}u) { key = keys[i]; }
    let digit = (key >> ${shift}u) & ${2 ** bits - 1}u;
    digits[lane] = digit;
    workgroupBarrier();
    if (i >= ${count}u) { return; }
    var rank = 0u;
    for (var before = 0u; before < lane; before++) {
        if (digits[before] == digit) { rank++; }
    }
    let at = offsets[digit * ${blocks}u + group.x] + rank;
    nextKeys[at] = key;
    nextIndices[at] = indices[i];
}`;
}