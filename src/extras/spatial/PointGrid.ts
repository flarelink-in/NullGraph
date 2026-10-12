import type { NullGraph, ComputeStageNode } from '../../core';
import type { PointGridLayout } from './gridMath';
import { GRID_SCAN_BLOCK, buildGridBinWGSL, buildGridScanWGSL, buildGridScanAddWGSL,
    gridRadixLayout, buildGridKeysWGSL, buildGridHistogramWGSL, buildGridStableScatterWGSL } from './gridShaders';

export interface PointGridOptions extends PointGridLayout {
    capacity: number;
    stride: number;
    flagsOffset?: number;
    aliveMask?: number;
    maxScanPerCell: number;
}

/** GPU-owned XZ grid. Caller owns records. Stages/buffers are built once; repeat the same array
 * for each simulation step. Reset is a dispatch, so repeated steps in one submission are safe. */
export class PointGrid {
    public readonly cellCounts: GPUBuffer;
    public readonly cellOffsets: GPUBuffer;
    public readonly cellBodies: GPUBuffer;
    public readonly overflow: GPUBuffer;
    private readonly owned: GPUBuffer[] = [];
    private readonly stages: ComputeStageNode[] = [];
    private destroyed = false;

    constructor(public readonly engine: NullGraph, public readonly records: GPUBuffer,
        public readonly options: PointGridOptions) {
        const o = options, d = engine.device, flags = o.flagsOffset ?? -1;
        const integer = (v: number) => Number.isSafeInteger(v) && v > 0;
        if (!(integer(o.capacity) && integer(o.stride) && o.stride >= 3 && integer(o.maxScanPerCell) &&
            integer(o.grid.cellsX) && integer(o.grid.cellsZ) &&
            o.grid.cellTotal === o.grid.cellsX * o.grid.cellsZ &&
            Number.isFinite(o.cellSize) && o.cellSize > 0 &&
            Number.isFinite(o.world.minX) && Number.isFinite(o.world.minZ) &&
            Number.isInteger(flags) && flags >= -1 && flags < o.stride &&
            Number.isInteger(o.aliveMask ?? 1) && (o.aliveMask ?? 1) > 0 && (o.aliveMask ?? 1) <= 0xffffffff)) {
            throw new Error('PointGrid: invalid capacity, layout, stride, flags or scan budget.');
        }
        if (!(records.usage & GPUBufferUsage.STORAGE) || records.size < o.capacity * o.stride * 4) {
            throw new Error('PointGrid: records must have STORAGE usage and fit capacity × stride.');
        }
        for (const words of [o.capacity * o.stride, o.capacity, o.grid.cellTotal + 1]) {
            if (words * 4 > Math.min(d.limits.maxStorageBufferBindingSize, d.limits.maxBufferSize) ||
                Math.ceil(Math.max(o.capacity, o.grid.cellTotal + 1) / 64) > d.limits.maxComputeWorkgroupsPerDimension) {
                throw new Error('PointGrid: layout exceeds device buffer or dispatch limits.');
            }
        }
        if (d.limits.maxComputeInvocationsPerWorkgroup < GRID_SCAN_BLOCK ||
            d.limits.maxComputeWorkgroupSizeX < GRID_SCAN_BLOCK) {
            throw new Error('PointGrid: device requires a 256-lane scan workgroup.');
        }
        const radix = gridRadixLayout(o.grid.cellTotal), blocks = Math.ceil(o.capacity / GRID_SCAN_BLOCK);
        const tableWords = radix.buckets * blocks;
        for (const [name, words] of [['radix keys', o.capacity], ['radix indices', o.capacity],
            ['radix histogram', tableWords], ['radix offsets', tableWords + 1]] as const) {
            if (words * 4 > Math.min(d.limits.maxStorageBufferBindingSize, d.limits.maxBufferSize)) {
                throw new Error(`PointGrid: ${name} exceeds device buffer limits.`);
            }
        }
        if (Math.ceil((tableWords + 1) / 64) > d.limits.maxComputeWorkgroupsPerDimension) {
            throw new Error('PointGrid: radix scan exceeds device dispatch limit.');
        }
        if (radix.buckets * 4 > (d.limits.maxComputeWorkgroupStorageSize ?? 16384)) {
            throw new Error('PointGrid: radix histogram exceeds device workgroup storage limit.');
        }
        this.cellCounts = this.buffer(o.grid.cellTotal, 'counts');
        this.cellBodies = this.buffer(o.capacity, 'indices');
        this.overflow = this.buffer(1, 'overflow');
        this.stage('reset', `
@group(0) @binding(0) var<storage, read_write> counts: array<u32>;
@group(0) @binding(1) var<storage, read_write> overflow: u32;
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < ${o.grid.cellTotal}u) { counts[id.x] = 0u; }
    if (id.x == 0u) { overflow = 0u; }
}`, [this.cellCounts, this.overflow], Math.ceil(o.grid.cellTotal / 64));
        this.stage('bin', buildGridBinWGSL(o, o.capacity, o.stride, flags, o.aliveMask ?? 1, o.maxScanPerCell),
            [records, this.cellCounts, this.overflow], Math.ceil(o.capacity / 64));
        this.cellOffsets = this.scan(this.cellCounts, o.grid.cellTotal);
        const keys = [this.buffer(o.capacity, 'radix keys 0'), this.buffer(o.capacity, 'radix keys 1')];
        const values = [this.buffer(o.capacity, 'radix indices 0'), this.buffer(o.capacity, 'radix indices 1')];
        const table = this.buffer(radix.buckets * blocks, 'radix histogram');
        this.stage('keys', buildGridKeysWGSL(o, o.capacity, o.stride, flags, o.aliveMask ?? 1),
            [records, keys[0], values[0]], Math.ceil(o.capacity / 64));
        for (let pass = 0; pass < radix.passes; pass++) {
            const src = pass % 2, dst = 1 - src, shift = pass * radix.bits;
            this.stage('histogram', buildGridHistogramWGSL(o.capacity, radix.bits, shift), [keys[src], table], blocks);
            const offsets = this.scan(table, radix.buckets * blocks);
            this.stage('stable scatter', buildGridStableScatterWGSL(o.capacity, radix.bits, shift),
                [keys[src], values[src], offsets, keys[dst],
                    pass === radix.passes - 1 ? this.cellBodies : values[dst]], blocks);
        }
    }

    private buffer(words: number, label: string): GPUBuffer {
        const limits = this.engine.device.limits;
        if (words * 4 > Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize)) {
            throw new Error(`PointGrid: ${label} exceeds device buffer limits.`);
        }
        const b = this.engine.device.createBuffer({ label: `PointGrid ${label}`, size: words * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        this.owned.push(b);
        return b;
    }

    private stage(name: string, shaderCode: string, buffers: GPUBuffer[], groups: number): void {
        if (groups > this.engine.device.limits.maxComputeWorkgroupsPerDimension) {
            throw new Error(`PointGrid: ${name} exceeds device dispatch limit.`);
        }
        const stage = this.engine.createComputeStage({ name: `PointGrid ${name}`, shaderCode,
            dispatchCount: [groups, 1, 1] });
        this.engine.setComputePassBindGroup(stage, buffers.map((buffer, binding) => ({ binding, resource: { buffer } })), 0);
        this.stages.push(stage);
    }

    private scan(input: GPUBuffer, count: number): GPUBuffer {
        const offsets = this.buffer(count + 1, 'offsets'), blocks = Math.ceil(count / GRID_SCAN_BLOCK);
        const totals = this.buffer(blocks, 'block totals');
        this.stage('scan', buildGridScanWGSL(count), [input, offsets, totals], blocks);
        if (blocks > 1) {
            const blockOffsets = this.scan(totals, blocks);
            this.stage('scan add', buildGridScanAddWGSL(count), [offsets, blockOffsets], Math.ceil((count + 1) / 64));
        }
        return offsets;
    }

    public getComputeStages(): ComputeStageNode[] {
        if (this.destroyed) throw new Error('PointGrid: used after destroy().');
        return this.stages;
    }

    public destroy(): void {
        if (this.destroyed) return;
        this.destroyed = true;
        for (const s of this.stages) s.enabled = false;
        for (const b of this.owned) b.destroy();
    }
}