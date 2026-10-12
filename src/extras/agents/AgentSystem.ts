import type { NullGraph, ComputeStageNode } from '../../core';
import { PointGrid } from '../spatial/PointGrid';
import { AGENT_BYTES, AGENT_FLOATS, AGENT_FLAGS_OFFSET, AGENT_FLAG_ALIVE,
    defineAgents, packSpecies, assertBrainDims, type AgentDefinition, type SpeciesOptions } from './agentLayout';
import { AgentRecords, type AgentInit } from './agentRecords';
import { FixedStep } from './agentMath';
import { buildAgentGridQueryWGSL } from './agentGridShaders';
import { buildAgentTickWGSL } from './agentSystemShaders';
import { buildAgentSenseWGSL } from './agentSenseShaders';
import { buildAgentActWGSL } from './agentActShaders';
import type { Brain } from './RuleBrain';
import { buildAgentThinkWGSL, buildAgentStepWGSL } from './agentThinkShaders';
import { buildAgentInstanceWGSL, assertInstanceBounds, INSTANCE_ROW_FLOATS, HIGHLIGHT_NONE,
    type AgentHighlight } from './agentInstances';
import type { VisibilitySystem } from '../visibility/VisibilitySystem';
import { agentSnapshotLayout, decodeAgentSnapshot, type AgentSnapshot, type AgentSnapshotLayout } from './agentInspect';

export interface AgentSystemOptions {
    /** Fixed at construction; shared sense/think cadence, staggered by 64-agent block. */
    thinkEvery?: number;
}

export interface AgentStepStageOptions {
    name: string;
    wgsl: string;
    /** Caller-owned group-1 buffers; WGSL declares and uses these bindings in order. */
    buffers?: readonly GPUBuffer[];
}

export interface AgentInstanceOptions {
    /** Row of `vis.instanceBuffer` that agent 0 writes; agent i writes `firstInstance + i`. Default 0. */
    firstInstance?: number;
}

/** GPU grid/query/sense/think/act/app/tick loop.
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
    private thinkStage?: ComputeStageNode;
    private readonly appStages = new Map<string, ComputeStageNode>();
    private instanceStage?: ComputeStageNode;
    private highlightBuffer?: GPUBuffer;
    private readonly highlightData = new Uint32Array([HIGHLIGHT_NONE, HIGHLIGHT_NONE, 0, 0]);
    private readonly snapshotLayout: AgentSnapshotLayout;
    private inspectStaging?: GPUBuffer;
    private inspectQueue: Promise<unknown> = Promise.resolve();

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
        this.snapshotLayout = agentSnapshotLayout(this.definition);
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
            this.rebuildChains();
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

    private stage(name: string, shaderCode: string, buffers: GPUBuffer[], groups: number, extra: readonly GPUBuffer[] = []): ComputeStageNode {
        const stage = this.engine.createComputeStage({ name: `AgentSystem ${name}`, shaderCode, dispatchCount: [groups, 1, 1] });
        try {
            this.engine.setComputePassBindGroup(stage, buffers.map((buffer, binding) => ({ binding, resource: { buffer } })), 0);
            if (extra.length) this.engine.setComputePassBindGroup(stage,
                extra.map((buffer, binding) => ({ binding, resource: { buffer } })), 1);
        } catch (error) {
            stage.enabled = false;
            throw error;
        }
        this.stages.push(stage);
        return stage;
    }

    private rebuildChains(): void {
        const step = [...this.grid.getComputeStages(), ...this.stages.filter(s =>
            s.name.startsWith('AgentSystem query ') || s.name === 'AgentSystem sense'),
            ...(this.thinkStage ? [this.thinkStage] : []),
            this.stages.find(s => s.name === 'AgentSystem act')!, ...this.appStages.values(),
            this.stages.find(s => s.name === 'AgentSystem tick')!];
        this.chains.length = 0;
        for (let n = 0; n <= this.definition.maxStepsPerFrame; n++) {
            const chain: ComputeStageNode[] = [];
            for (let j = 0; j < n; j++) chain.push(...step);
            // Once per frame, after the last step — also with no step, so a highlight still shows.
            if (this.instanceStage) chain.push(this.instanceStage);
            this.chains.push(chain);
        }
    }

    private checkExtraBuffers(name: string, buffers: readonly GPUBuffer[]): void {
        const left = Math.min(3, this.engine.device.limits.maxStorageBuffersPerShaderStage - 5);
        if (buffers.length > left) throw new Error(`AgentSystem: ${name} binds ${buffers.length} buffers; ${left} are left after the system's 5.`);
    }

    public setBrain(brain: Brain): void {
        this.assertAlive();
        if (brain.kind !== 'stage') throw new Error('AgentSystem: async brains are deferred to C2; use a stage brain.');
        if (brain.dims) assertBrainDims(this.definition, brain.dims.obs, brain.dims.act);
        const code = buildAgentThinkWGSL(this.definition, brain.wgsl!, this.thinkEvery);
        const extra = brain.buffers ?? [];
        this.checkExtraBuffers('brain', extra);
        const next = this.stage('think', code, [this.buffer, this.obs, this.state, this.act, this.reason, this.species],
            Math.ceil(this.capacity / 64), extra);
        if (this.thinkStage) {
            this.thinkStage.enabled = false;
            this.stages.splice(this.stages.indexOf(this.thinkStage), 1);
        }
        this.thinkStage = next;
        this.rebuildChains();
    }

    /** App stages must write only their own record, never fields peers read in the same dispatch. */
    public addStepStage(options: AgentStepStageOptions): void {
        this.assertAlive();
        if (this.appStages.has(options.name)) throw new Error(`AgentSystem: step stage "${options.name}" is added twice.`);
        const code = buildAgentStepWGSL(this.definition, options.name, options.wgsl);
        const extra = options.buffers ?? [];
        this.checkExtraBuffers(`step stage "${options.name}"`, extra);
        const next = this.stage(`step ${options.name}`, code, [this.neighbours, this.obs, this.act, this.state, this.buffer, this.species],
            Math.ceil(this.capacity / 64), extra);
        this.appStages.set(options.name, next);
        this.rebuildChains();
    }

    /**
     * Step 4 (§9.2): write each agent's VisibilitySystem row on the GPU, once per frame after the last
     * step — `translate(position) · rotateY(heading)`, the species' bounds, and its mesh, hover or
     * selected variant by `setHighlight`. Call `vis.setInstanceCount(n)` first, with n ≥
     * `firstInstance + capacity`, and splice `vis.getComputeStages()` after this system's stages.
     * Calling again retargets. Destroy this system before `vis`: the stage binds its buffer.
     */
    public toInstances(vis: VisibilitySystem, options: AgentInstanceOptions = {}): void {
        this.assertAlive();
        if ((vis as unknown as { engine: NullGraph }).engine !== this.engine) {
            throw new Error('AgentSystem: that VisibilitySystem belongs to a different engine.');
        }
        const first = options.firstInstance ?? 0;
        if (!(Number.isInteger(first) && first >= 0)) {
            throw new Error(`AgentSystem: toInstances() firstInstance ${first} must be a whole number ≥ 0.`);
        }
        const room = Math.floor(vis.instanceBuffer.size / (INSTANCE_ROW_FLOATS * 4));
        if (first + this.capacity > room) {
            throw new Error(`AgentSystem: toInstances() needs maxInstances ≥ ${first + this.capacity}; this VisibilitySystem has ${room}.`);
        }
        if (vis.instanceSource !== 'gpu') {
            throw new Error('AgentSystem: toInstances() writes vis.instanceBuffer on the GPU; use vis.setInstanceCount(n), not setInstances() (it re-uploads every frame).');
        }
        assertInstanceBounds(this.definition);
        const highlight = this.highlightBuffer ?? this.allocate('highlight', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        if (!this.highlightBuffer) {
            this.highlightBuffer = highlight;
            this.engine.device.queue.writeBuffer(highlight, 0, this.highlightData);
        }
        const next = this.stage('instances', buildAgentInstanceWGSL(this.definition, first),
            [this.buffer, vis.instanceBuffer, this.species, highlight], Math.ceil(this.capacity / 64));
        if (this.instanceStage) {
            this.instanceStage.enabled = false;
            this.stages.splice(this.stages.indexOf(this.instanceStage), 1);
        }
        this.instanceStage = next;
        this.rebuildChains();
    }

    /**
     * Step 4 (§9.3): draw `selected` and `hovered` (agent indices, -1 for none) with their species'
     * selected and hover meshes. Writes the 16-byte uniform only when something changed; an omitted
     * field keeps its value.
     */
    public setHighlight(highlight: Partial<AgentHighlight>): void {
        this.assertAlive();
        const selected = this.highlightIndex(highlight.selected, this.highlightData[0]);
        const hovered = this.highlightIndex(highlight.hovered, this.highlightData[1]);
        if (selected === this.highlightData[0] && hovered === this.highlightData[1]) return;
        this.highlightData[0] = selected;
        this.highlightData[1] = hovered;
        if (this.highlightBuffer) this.engine.device.queue.writeBuffer(this.highlightBuffer, 0, this.highlightData);
    }

    private highlightIndex(v: number | undefined, current: number): number {
        if (v === undefined) return current;
        if (v === -1) return HIGHLIGHT_NONE;
        if (!(Number.isInteger(v) && v >= 0 && v < this.capacity)) {
            throw new Error(`AgentSystem: index ${v} is outside capacity ${this.capacity}.`);
        }
        return v;
    }

    /**
     * Step 4 (§9.6): what agent `i` senses, chose and why — its record, observation row, action
     * row, reason, neighbour indices and the system's counters — copied in one submission after
     * everything submitted so far, and read back off the frame path. Requests run one at a time.
     * Resolves null if the system is destroyed before the copy is read.
     */
    public inspect(i: number): Promise<AgentSnapshot | null> {
        this.assertAlive();
        if (!(Number.isInteger(i) && i >= 0 && i < this.capacity)) {
            throw new Error(`AgentSystem: index ${i} is outside capacity ${this.capacity}.`);
        }
        const run = this.inspectQueue.then(() => this.readSnapshot(i));
        this.inspectQueue = run.catch(() => undefined);
        return run;
    }

    private async readSnapshot(i: number): Promise<AgentSnapshot | null> {
        if (this.destroyed) return null;
        const def = this.definition, l = this.snapshotLayout, device = this.engine.device;
        const staging = this.inspectStaging ??= this.allocate('inspect staging', l.size,
            GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        const enc = device.createCommandEncoder({ label: 'AgentSystem inspect' });
        enc.copyBufferToBuffer(this.buffer, i * AGENT_BYTES, staging, l.record, AGENT_BYTES);
        enc.copyBufferToBuffer(this.obs, i * def.obsDim * 4, staging, l.obs, def.obsDim * 4);
        enc.copyBufferToBuffer(this.act, i * def.actDim * 4, staging, l.act, def.actDim * 4);
        enc.copyBufferToBuffer(this.reason, i * 4, staging, l.reason, 4);
        if (l.neighbourStride > 0) {
            enc.copyBufferToBuffer(this.neighbours, i * l.neighbourStride * 4, staging, l.neighbours, l.neighbourStride * 4);
        }
        enc.copyBufferToBuffer(this.state, 0, staging, l.state, 16);
        device.queue.submit([enc.finish()]);
        try {
            await staging.mapAsync(GPUMapMode.READ);
        } catch (error) {
            if (this.destroyed) return null;   // destroy() released the staging buffer mid-read
            throw error;
        }
        if (this.destroyed) return null;
        const data = staging.getMappedRange().slice(0);
        staging.unmap();
        return decodeAgentSnapshot(def, i, data, l);
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
        if (this.instanceStage) assertInstanceBounds(checked);
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
    public getComputeStages(): ComputeStageNode[] { this.assertAlive(); this.assertBrain(); return this.chains[this.clock.steps]; }

    public async step(): Promise<void> {
        this.assertAlive();
        this.assertBrain();
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

    private assertBrain(): void {
        if (!this.thinkStage) throw new Error('AgentSystem: no brain — call setBrain() before getComputeStages() or step().');
    }
}