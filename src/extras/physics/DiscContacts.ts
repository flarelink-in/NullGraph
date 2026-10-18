import type { NullGraph, ComputeStageNode } from '../../core';
import { PointGrid } from '../spatial/PointGrid';
import { gridRadixLayout } from '../spatial/gridShaders';
import { validateDiscContacts, MAX_CELLS, type DiscContactsOptions, type ResolvedDiscContacts } from './contactMath';
import { buildContactGatherWGSL, buildContactApplyWGSL, buildContactCountWGSL,
    CONTACT_RESET_WGSL, CONTACT_TICK_WGSL } from './contactShaders';

export type DiscContactParams = Partial<Pick<DiscContactsOptions, 'relaxation' | 'compliance' | 'maxPush' | 'damping'>>;
export interface DiscContactCounters { overflow: number; bodiesOverMaxRadius: number }

/** Atomic-free Jacobi contacts on XZ. The caller owns records and runs these stages after moving them. */
export class DiscContacts {
    public readonly options: ResolvedDiscContacts;
    public readonly grid: PointGrid;
    public readonly deltas: GPUBuffer;
    private readonly owned: GPUBuffer[] = [];
    private readonly stages: ComputeStageNode[] = [];
    private readonly params: GPUBuffer;
    private readonly counters: GPUBuffer;
    private readonly upload = new Float32Array(4);
    private destroyed = false;
    private active = true;

    constructor(public readonly engine: NullGraph, public readonly records: GPUBuffer,
        options: DiscContactsOptions & { capacity: number; stride: number }) {
        const d = engine.device, limits = d.limits;
        // Validate all declarations before allocating resources. Default cells grow to fit the device.
        // A coarse provisional cell lets validation check the declaration before sizing a large default grid.
        const span = Math.max(options.world?.max?.[0] - options.world?.min?.[0],
            options.world?.max?.[1] - options.world?.min?.[1]);
        let o = validateDiscContacts(options, { ...options, cellSize: options.cellSize ?? Math.max(span,
            2 * options.maxRadius + ((options.iterations ?? 2) - 1) * (options.maxPush ?? 0.25)) });
        let cellSize = options.cellSize ?? 2 * o.maxRadius + (o.iterations - 1) * o.maxPush;
        const fits = (r: ResolvedDiscContacts) => {
            const table = gridRadixLayout(r.grid.cellTotal).buckets * Math.ceil(r.layout.capacity / 256) + 1;
            return r.grid.cellTotal <= MAX_CELLS && Math.max(table, r.grid.cellTotal + 1) * 4 <=
                Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize) &&
                Math.ceil(Math.max(table, r.grid.cellTotal + 1) / 64) <= limits.maxComputeWorkgroupsPerDimension;
        };
        if (options.cellSize === undefined) {
            const cells = (h: number) => Math.ceil((o.world.maxX - o.world.minX) / h) * Math.ceil((o.world.maxZ - o.world.minZ) / h);
            while (cells(cellSize) > MAX_CELLS) cellSize *= 1.25;
            o = validateDiscContacts(options, { ...options, cellSize });
            while (!fits(o) && o.grid.cellTotal > 1) {
                cellSize *= 1.25;
                o = validateDiscContacts(options, { ...options, cellSize });
            }
        }
        if (!fits(o)) throw new Error(`DiscContacts: the grid needs ${o.grid.cellTotal} cells; raise cellSize or shrink the world.`);
        if (!(records.usage & GPUBufferUsage.STORAGE) || records.size < o.layout.capacity * o.layout.stride * 4) {
            throw new Error('DiscContacts: records must have STORAGE usage and hold capacity × stride floats.');
        }
        const groups = Math.ceil(o.layout.capacity / 64);
        if (groups > limits.maxComputeWorkgroupsPerDimension || o.layout.capacity * 16 >
            Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize)) {
            throw new Error('DiscContacts: capacity exceeds device buffer or dispatch limits.');
        }
        this.options = o;
        this.grid = new PointGrid(engine, records, { ...o, ...o.layout,
            flagsOffset: o.layout.alive?.offset, aliveMask: o.layout.alive?.mask });
        try {
            this.deltas = this.buffer('deltas', o.layout.capacity * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
            this.params = this.buffer('params', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
            this.counters = this.buffer('counters', 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
            this.stages.push(...this.grid.getComputeStages());
            this.stage('reset', CONTACT_RESET_WGSL, [this.counters], 1);
            this.stage('count', buildContactCountWGSL(o), [records, this.counters], groups);
            for (let pass = 0; pass < o.iterations; pass++) {
                this.stage(`gather ${pass}`, buildContactGatherWGSL(o, pass),
                    [records, this.grid.cellOffsets, this.grid.cellBodies, this.deltas, this.params, this.counters], groups);
                this.stage(`apply ${pass}`, buildContactApplyWGSL(o), [records, this.deltas, this.params], groups);
            }
            this.stage('tick', CONTACT_TICK_WGSL, [this.counters], 1);
            this.setParams({});
        } catch (error) {
            this.destroy();
            throw error;
        }
    }

    private buffer(label: string, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
        const b = this.engine.device.createBuffer({ label: `DiscContacts ${label}`, size, usage });
        this.owned.push(b);
        return b;
    }

    private stage(name: string, shaderCode: string, buffers: GPUBuffer[], groups: number): void {
        const s = this.engine.createComputeStage({ name: `DiscContacts ${name}`, shaderCode, dispatchCount: [groups, 1, 1] });
        this.stages.push(s);
        this.engine.setComputePassBindGroup(s, buffers.map((buffer, binding) => ({ binding, resource: { buffer } })), 0);
    }

    private assertLive(): void {
        if (this.destroyed) throw new Error('DiscContacts: used after destroy().');
    }

    public getComputeStages(): ComputeStageNode[] { this.assertLive(); return this.stages; }
    public get enabled(): boolean { this.assertLive(); return this.active; }
    public set enabled(value: boolean) {
        this.assertLive();
        this.active = value;
        for (const s of this.stages) s.enabled = value;
    }

    public setParams(params: DiscContactParams): void {
        this.assertLive();
        const o = this.options;
        const next = validateDiscContacts(o.layout, { ...o, world: {
            min: [o.world.minX, o.world.minZ], max: [o.world.maxX, o.world.maxZ] }, ...params });
        this.upload.set([next.relaxation, next.tildeCompliance, next.maxPush, next.damping]);
        if (!this.upload.every(Number.isFinite)) throw new Error('DiscContacts: params must fit finite f32 values.');
        this.engine.device.queue.writeBuffer(this.params, 0, this.upload);
        o.relaxation = next.relaxation; o.compliance = next.compliance;
        o.tildeCompliance = next.tildeCompliance; o.maxPush = next.maxPush; o.damping = next.damping;
    }

    /** Off-frame readback of the last completed step. Safe for concurrent reads. */
    public async readCounters(): Promise<DiscContactCounters> {
        this.assertLive();
        const d = this.engine.device;
        const dst = d.createBuffer({ size: 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        try {
            const enc = d.createCommandEncoder();
            enc.copyBufferToBuffer(this.grid.overflow, 0, dst, 0, 4);
            enc.copyBufferToBuffer(this.counters, 4, dst, 4, 4);
            d.queue.submit([enc.finish()]);
            await dst.mapAsync(GPUMapMode.READ);
            const words = new Uint32Array(dst.getMappedRange());
            return { overflow: words[0], bodiesOverMaxRadius: words[1] };
        } finally { dst.destroy(); }
    }

    public destroy(): void {
        if (this.destroyed) return;
        this.destroyed = true;
        for (const s of this.stages) s.enabled = false;
        this.grid.destroy();
        for (const b of this.owned) b.destroy();
    }
}