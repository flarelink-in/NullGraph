import type { NullGraph, ComputeStageNode } from '../../core';
import { PointGrid } from '../spatial/PointGrid';
import { AGENT_BYTES, AGENT_FLOATS, AGENT_FLAGS_OFFSET, AGENT_FLAG_ALIVE,
    defineAgents, packSpecies, type AgentDefinition, type SpeciesOptions } from './agentLayout';
import { AgentRecords, type AgentInit } from './agentRecords';
import { FixedStep } from './agentMath';
import { buildAgentGridQueryWGSL } from './agentGridShaders';
import { buildAgentTickWGSL } from './agentSystemShaders';
import { buildAgentSenseWGSL } from './agentSenseShaders';
import { buildAgentActWGSL } from './agentActShaders';

export interface AgentSystemOptions {
    /** Fixed at construction; shared sense/think cadence, staggered by 64-agent block. */
    thinkEvery?: number;
}

/** Phase 3.3: GPU grid/query/sense/act/tick loop.
 * Records are a CPU staging area, not a read-back mirror. No readback on the frame path. */
export class AgentSystem {
    public readonly capacity: number;
    public readonly definition: AgentDefinition;
    public readonly records: AgentRecords;
    public readonly buffer: GPUBuffer;
    public readonly neighbours: GPUBuffer;
    public readonly obs: GPUBuffer;
    public readonly act: GPUBuffer;
    public readonly reason: GPUBuffer;
    public readonly state: GPUBuffer;
    public readonly species: GPUBuffer;
    public readonly grid: PointGrid;
    public readonly thinkEvery: number;
    public readonly neighbourStride: number;
    private readonly clock: FixedStep;
    private readonly owned: GPUBuffer[] = [];
    private readonly stages: ComputeStageNode[] = [];
    private readonly chains: ComputeStageNode[][] = [];
    private readonly speciesData: Float32Array;
    private speciesDirty = true;
    private destroyed = false;

    constructor(public readonly engine: NullGraph, def: AgentDefinition, options: AgentSystemOptions = {}) {
        this.thinkEvery = options.thinkEvery ?? 1;
        if (!(Number.isInteger(this.thinkEvery) && this.thinkEvery > 0 && this.thinkEvery <= 0xffffffff)) {
            throw new Error(`AgentSystem: thinkEvery ${this.thinkEvery} must be a positive integer.`);
        }
        this.capacity = def.capacity;
        this.definition = { ...def, species: def.species.map(s => ({ ...s })) };
        this.records = new AgentRecords(this.definition);
        this.clock = new FixedStep(def.dt, def.maxStepsPerFrame, def.maxFrameDelta);
        this.speciesData = packSpecies(this.definition);
        this.neighbourStride = def.sensors.reduce((n, s) => n + (s.kind === 'nearest' ? s.k : 0), 0);
        const senseCode = buildAgentSenseWGSL(def, this.thinkEvery);
        const actCode = buildAgentActWGSL(def);
        const sizes = [AGENT_BYTES * def.capacity, Math.max(4, def.capacity * this.neighbourStride * 4),
            Math.max(4, def.capacity * def.obsDim * 4), def.capacity * def.actDim * 4, def.capacity * 4, 16];
        const limit = Math.min(engine.device.limits.maxStorageBufferBindingSize, engine.device.limits.maxBufferSize);
        for (const [i, size] of sizes.entries()) {
            if (!Number.isSafeInteger(size) || size > limit) {
                throw new Error(`AgentSystem: ${['agents', 'neighbours', 'obs', 'act', 'reason', 'state'][i]} needs ${size} bytes; this device binds at most ${limit}. Lower capacity or the sensor count.`);
            }
        }
        let grid: PointGrid | undefined;
        try {
            this.buffer = this.allocate('agents', sizes[0], GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
            this.neighbours = this.allocate('neighbours', sizes[1], GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
            this.obs = this.allocate('obs', sizes[2], GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
            this.act = this.allocate('act', sizes[3], GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
            this.reason = this.allocate('reason', sizes[4], GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
            this.state = this.allocate('state', sizes[5], GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
            this.species = this.allocate('species', this.speciesData.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
            this.grid = grid = new PointGrid(engine, this.buffer, { ...def, stride: AGENT_FLOATS,
                flagsOffset: AGENT_FLAGS_OFFSET, aliveMask: AGENT_FLAG_ALIVE });
            let base = 0, sensorIndex = 0;
            for (const sensor of def.sensors) {
                if (sensor.kind !== 'nearest') continue;
                this.stage(`query ${sensor.name}`, buildAgentGridQueryWGSL(def, def.nearestMasks[sensorIndex++],
                    def.cellSize, sensor.k, { stride: this.neighbourStride, base }),
                [this.buffer, grid.cellCounts, grid.cellOffsets, grid.cellBodies, this.neighbours], Math.ceil(def.capacity / 64));
                base += sensor.k;
            }
            this.stage('sense', senseCode, [this.buffer, this.neighbours, this.state, this.obs, this.species], Math.ceil(def.capacity / 64));
            this.stage('act', actCode, [this.act, this.buffer, this.species], Math.ceil(def.capacity / 64));
            this.stage('tick', buildAgentTickWGSL(), [grid.overflow, this.state], 1);
            const step = [...grid.getComputeStages(), ...this.stages];
            for (let n = 0; n <= def.maxStepsPerFrame; n++) {
                const chain: ComputeStageNode[] = [];
                for (let j = 0; j < n; j++) chain.push(...step);
                this.chains.push(chain);
            }
        } catch (error) {
            grid?.destroy();
            for (const stage of this.stages) stage.enabled = false;
            for (const buffer of this.owned) buffer.destroy();
            throw error;
        }
    }

    private allocate(name: string, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
        const buffer = this.engine.device.createBuffer({ label: `AgentSystem ${name}`, size, usage });
        this.owned.push(buffer);
        return buffer;
    }

    private stage(name: string, shaderCode: string, buffers: GPUBuffer[], groups: number): void {
        const stage = this.engine.createComputeStage({ name: `AgentSystem ${name}`, shaderCode, dispatchCount: [groups, 1, 1] });
        this.stages.push(stage);
        this.engine.setComputePassBindGroup(stage, buffers.map((buffer, binding) => ({ binding, resource: { buffer } })), 0);
    }

    public spawn(i: number, init: AgentInit): void { this.assertAlive(); this.records.spawn(i, init); }
    public kill(i: number): void { this.assertAlive(); this.records.kill(i); }

    public setSpecies(name: string, partial: Partial<Omit<SpeciesOptions, 'name'>>): void {
        this.assertAlive();
        const def = this.definition, index = def.speciesIndex(name);
        const species = def.species.map(s => ({ ...s, ...(s.index === index ? partial : {}), name: s.name }));
        const checked = defineAgents({ ...def, species,
            world: { min: [def.world.minX, def.world.minZ], max: [def.world.maxX, def.world.maxZ] },
            maxCells: def.grid.cellTotal });
        // Keep the object AgentRecords holds; replace only the validated species table.
        Object.assign(def, { species: checked.species });
        packSpecies(def, this.speciesData);
        this.speciesDirty = true;
    }

    public upload(): void {
        this.assertAlive();
        const first = this.records.dirtyFirst;
        if (first >= 0) {
            const offset = first * AGENT_BYTES, size = (this.records.dirtyLast - first + 1) * AGENT_BYTES;
            this.engine.device.queue.writeBuffer(this.buffer, offset, this.records.data.buffer,
                this.records.data.byteOffset + offset, size);
            this.records.clearDirty();
        }
        if (this.speciesDirty) {
            this.engine.device.queue.writeBuffer(this.species, 0, this.speciesData as Float32Array<ArrayBuffer>);
            this.speciesDirty = false;
        }
    }

    public advance(frameSeconds: number): number { this.assertAlive(); return this.clock.advance(frameSeconds); }
    public getComputeStages(): ComputeStageNode[] { this.assertAlive(); return this.chains[this.clock.steps]; }

    public async step(): Promise<void> {
        this.assertAlive();
        this.upload();
        this.engine.render(this.chains[1]);
        await this.engine.device.queue.onSubmittedWorkDone();
    }

    public destroy(): void {
        if (this.destroyed) return;
        this.destroyed = true;
        this.grid.destroy();
        for (const stage of this.stages) stage.enabled = false;
        for (const buffer of this.owned) buffer.destroy();
    }

    private assertAlive(): void {
        if (this.destroyed) throw new Error('AgentSystem: used after destroy().');
    }

}