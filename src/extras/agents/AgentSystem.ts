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
    /** The brain serving each species, by species index (phase 7.3). */
    private readonly brainOf: (Brain | undefined)[];
    /** One think pass per attached brain, in attachment order, gated by its species-mask uniform. */
    private readonly thinks = new Map<Brain, { stage: ComputeStageNode; mask: GPUBuffer; bits: number }>();
    /**
     * Step 9 (agents-ort-design.md §8.6): the think passes of prepared brains, with their masks,
     * kept while this system lives, attached or not; and each preparation's promise.
     */
    private readonly kept = new Map<Brain, { stage: ComputeStageNode; mask: GPUBuffer }>();
    private readonly preparing = new Map<Brain, Promise<void>>();
    private everySpecies = false;
    /**
     * Step 9 (agents-ort-design.md §8): each attached async brain's run in flight, the steps since
     * it last started one, and whether one failed. `inflight` holds every run not yet settled, of
     * brains attached or not, with its brain: destroy() releases `obs` only after them, and a brain
     * attached again keeps its run in flight.
     */
    private readonly runs = new Map<Brain, { pending?: Promise<void>; steps: number; failed?: boolean }>();
    private readonly inflight = new Map<Promise<void>, Brain>();
    private failure?: Error;
    /** Lockstep with an async brain: grid, queries and sense; then the passes, act, app stages, tick and instances. */
    private halves: [ComputeStageNode[], ComputeStageNode[]] = [[], []];
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
        this.brainOf = def.species.map(() => undefined);
        this.snapshotLayout = agentSnapshotLayout(this.definition);
        this.neighbourStride = def.sensors.reduce((n, s) => n + (s.kind === 'nearest' ? s.k : 0), 0);
        const senseCode = buildAgentSenseWGSL(def, this.thinkEvery);
        const actCode = buildAgentActWGSL(def);
        // obs is rounded up to 16 B: ONNX Runtime reads 16 B at a time (agents-ort-design.md §6).
        const sizes = [AGENT_BYTES * def.capacity, Math.max(4, def.capacity * this.neighbourStride * 4),
            Math.ceil(def.capacity * def.obsDim / 4) * 16, def.capacity * def.actDim * 4, def.capacity * 4, 16];
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
                    sensor.radius ?? def.cellSize, sensor.k, { stride: this.neighbourStride, base }),
                [this.buffer, grid.cellCounts, grid.cellOffsets, grid.cellBodies, this.neighbours, this.state], Math.ceil(def.capacity / 64));
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

    private release(buffer: GPUBuffer): void {
        this.owned.splice(this.owned.indexOf(buffer), 1);
        buffer.destroy();
    }

    private stage(name: string, shaderCode: string, buffers: GPUBuffer[], groups: number, extra: readonly GPUBuffer[] = []): ComputeStageNode {
        const stage = this.engine.createComputeStage({ name: `AgentSystem ${name}`, shaderCode, dispatchCount: [groups, 1, 1] });
        this.bindGroups(stage, buffers, extra);
        this.stages.push(stage);
        return stage;
    }

    /** Group 0, and group 1 when there are extra buffers; a stage that cannot be bound is disabled. */
    private bindGroups(stage: ComputeStageNode, buffers: GPUBuffer[], extra: readonly GPUBuffer[]): void {
        try {
            this.engine.setComputePassBindGroup(stage, buffers.map((buffer, binding) => ({ binding, resource: { buffer } })), 0);
            if (extra.length) this.engine.setComputePassBindGroup(stage,
                extra.map((buffer, binding) => ({ binding, resource: { buffer } })), 1);
        } catch (error) {
            stage.enabled = false;
            throw error;
        }
    }

    /**
     * A think pass and its species mask: the stage made now from `code`, or the one given (compiled
     * by prepareBrain), bound to the system's buffers, the mask, and the brain's own buffers.
     */
    private thinkPass(code: string, extra: readonly GPUBuffer[], made?: ComputeStageNode): { stage: ComputeStageNode; mask: GPUBuffer } {
        const mask = this.allocate('brain species', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        const buffers = [this.buffer, this.obs, this.state, this.act, this.reason, this.species, mask];
        try {
            if (!made) return { stage: this.stage('think', code, buffers, Math.ceil(this.capacity / 64), extra), mask };
            this.bindGroups(made, buffers, extra);
            return { stage: made, mask };
        } catch (error) {
            this.release(mask);
            throw error;
        }
    }

    private rebuildChains(): void {
        const sensing = [...this.grid.getComputeStages(), ...this.stages.filter(s =>
            s.name.startsWith('AgentSystem query ') || s.name === 'AgentSystem sense')];
        const acting = [...[...this.thinks.values()].map(t => t.stage),
            this.stages.find(s => s.name === 'AgentSystem act')!, ...this.appStages.values(),
            this.stages.find(s => s.name === 'AgentSystem tick')!];
        const step = [...sensing, ...acting];
        this.chains.length = 0;
        for (let n = 0; n <= this.definition.maxStepsPerFrame; n++) {
            const chain: ComputeStageNode[] = [];
            for (let j = 0; j < n; j++) chain.push(...step);
            // Once per frame, after the last step — also with no step, so a highlight still shows.
            if (this.instanceStage) chain.push(this.instanceStage);
            this.chains.push(chain);
        }
        this.halves = [sensing, this.instanceStage ? [...acting, this.instanceStage] : acting];
    }

    private checkExtraBuffers(name: string, buffers: readonly GPUBuffer[]): void {
        // 3 on a default device; more when the device was created with a higher limit (core's
        // PROGRESSION_LIMITS, or an adopted device's own).
        const left = this.engine.device.limits.maxStorageBuffersPerShaderStage - 5;
        if (buffers.length > left) throw new Error(`AgentSystem: ${name} binds ${buffers.length} buffers; ${left} are left after the system's 5.`);
    }

    /**
     * Phase 7.3 (agents-species-brains-design.md §6): `brain` thinks for the named species, or for
     * every species. Each attached brain is one think pass, gated to its species by a 16-byte mask;
     * moving species between attached brains rewrites masks and compiles nothing. A brain left
     * serving no species leaves every chain, and may then be destroyed. A prepared brain
     * (`prepareBrain`) compiles nothing here, and keeps its pass when it leaves.
     */
    public setBrain(brain: Brain, options: { readonly species?: string | readonly string[] } = {}): void {
        this.assertAlive();
        if (this.preparing.has(brain) && !this.kept.has(brain)) {
            throw new Error('AgentSystem: this brain is still being prepared; await prepareBrain(brain) before setBrain(brain).');
        }
        const def = this.definition, named = options.species ?? def.species.map(s => s.name);
        const indices = (typeof named === 'string' ? [named] : named).map(name => {
            const s = def.species.find(sp => sp.name === name);
            if (!s) throw new Error(`AgentSystem: "${name}" is not a declared species.`);
            return s.index;
        });
        if (this.thinks.has(brain)) brain.assertDevice?.(this.engine.device);
        else this.attach(brain);
        for (const s of indices) this.brainOf[s] = brain;
        for (const [b, t] of this.thinks) {
            let bits = 0;
            this.brainOf.forEach((x, s) => { if (x === b) bits |= 1 << s; });
            if (!bits) {
                t.stage.enabled = false;
                this.stages.splice(this.stages.indexOf(t.stage), 1);
                if (!this.kept.has(b)) this.release(t.mask);    // a prepared brain keeps its pass for its next attach
                this.thinks.delete(b);
                this.runs.delete(b);         // a run in flight lands in the brain's own buffer, which it keeps
            } else if (bits !== t.bits) {
                t.bits = bits;
                this.engine.device.queue.writeBuffer(t.mask, 0, new Uint32Array([bits, 0, 0, 0]));
            }
        }
        this.everySpecies = this.brainOf.every(Boolean);
        this.rebuildChains();
    }

    /**
     * Validates `brain` and returns its think WGSL. An async brain (step 9) is bound first: its runs
     * read `obs` and write a buffer of its own, which its pass, gated like any think pass, copies
     * into `act`. Binding again re-seeds that buffer from `act`.
     */
    private check(brain: Brain): string {
        const async = brain.kind === 'async';
        if (async ? typeof brain.bind !== 'function' || typeof brain.run !== 'function' : brain.kind !== 'stage') {
            throw new Error('AgentSystem: a brain is a stage brain, or an async brain with bind() and run().');
        }
        if (async && brain.cooperative) throw new Error('AgentSystem: only a stage brain can be cooperative.');
        brain.assertDevice?.(this.engine.device);
        const def = this.definition;
        if (brain.dims) assertBrainDims(def, brain.dims.obs, brain.dims.act);
        if (async) brain.bind!({ obs: this.obs, act: this.act, capacity: this.capacity, obsDim: def.obsDim, actDim: def.actDim });
        const code = buildAgentThinkWGSL(def, brain.wgsl!, this.thinkEvery, { cooperative: brain.cooperative });
        this.checkExtraBuffers('brain', brain.buffers ?? []);
        return code;
    }

    /** Validates `brain` and builds its pass, or takes its prepared one; nothing is assigned until this returns. */
    private attach(brain: Brain): void {
        const code = this.check(brain);
        let pass = this.kept.get(brain);
        if (pass) {
            pass.stage.enabled = true;
            this.stages.push(pass.stage);
        } else {
            pass = this.thinkPass(code, brain.buffers ?? []);
        }
        this.thinks.set(brain, { ...pass, bits: 0 });
        // The first frame with a step starts its first run.
        // A run still in flight from an earlier attach is awaited like one started now.
        if (brain.kind === 'async') this.runs.set(brain, { steps: this.thinkEvery, pending: [...this.inflight].find(([, b]) => b === brain)?.[0] });
    }

    /**
     * Step 9 (agents-ort-design.md §8.6): compiles `brain`'s think pass ahead, with
     * `engine.createComputeStageAsync`, and keeps it while this system lives, so that
     * `setBrain(brain)` compiles nothing, now or after a detach. Checks the brain, and binds an async
     * one, as `setBrain` does. Preparing again returns the first promise; `setBrain(brain)` before
     * it settles is refused. A preparation that fails is forgotten, and may be tried again.
     */
    public prepareBrain(brain: Brain): Promise<void> {
        this.assertAlive();
        let prepared = this.preparing.get(brain);
        if (prepared) return prepared;
        const attached = this.thinks.get(brain);
        if (attached) {
            this.kept.set(brain, attached);                // compiled already: kept from now on
            prepared = Promise.resolve();
        } else {
            const code = this.check(brain), extra = brain.buffers ?? [];
            prepared = this.engine.createComputeStageAsync({ name: 'AgentSystem think', shaderCode: code,
                dispatchCount: [Math.ceil(this.capacity / 64), 1, 1] }).then(stage => {
                stage.enabled = false;                     // in no chain until setBrain
                if (!this.destroyed) this.kept.set(brain, this.thinkPass(code, extra, stage));
            });
            prepared.catch(() => this.preparing.delete(brain));
        }
        this.preparing.set(brain, prepared);
        return prepared;
    }

    /** Starts a run of an async brain; a failure is kept for the next advance() or step(). */
    private start(brain: Brain, run: { pending?: Promise<void>; steps: number; failed?: boolean }): Promise<void> {
        run.steps = 0;
        let started: Promise<void>;
        try { started = brain.run!(); } catch (error) { started = Promise.reject(error); }
        const pending: Promise<void> = started.then(() => undefined, error => {
            run.failed = true;
            const names = this.definition.species.filter(s => this.brainOf[s.index] === brain).map(s => s.name);
            // A brain that serves no species any more failed for nobody.
            if (names.length) this.failure ??= new Error(`AgentSystem: the brain of species "${names.join('", "')}" failed in a run: ${String(error?.message ?? error).split('\n')[0].replace(/\.$/, '')}. Give the species another brain with setBrain().`);
        }).finally(() => {
            const now = this.runs.get(brain);
            if (now?.pending === pending) now.pending = undefined;
            this.inflight.delete(pending);
        });
        run.pending = pending;
        this.inflight.set(pending, brain);
        return pending;
    }

    /** A failed run is thrown once, by the next advance() or step(). */
    private throwFailure(): void {
        const failure = this.failure;
        this.failure = undefined;
        if (failure) throw failure;
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

    /**
     * The steps this frame. With an async brain (step 9, agents-ort-design.md §8.2) it also starts a
     * run when the brain has none in flight and `thinkEvery` steps have passed since its last one;
     * the run reads the observations this frame's stages write, and lands before the next frame's.
     */
    public advance(frameSeconds: number): number {
        this.assertAlive();
        this.throwFailure();
        const steps = this.clock.advance(frameSeconds);
        if (steps) {
            for (const [brain, run] of this.runs) {
                run.steps += steps;
                if (!run.pending && !run.failed && run.steps >= this.thinkEvery) this.start(brain, run);
            }
        }
        return steps;
    }
    public getComputeStages(): ComputeStageNode[] { this.assertAlive(); this.assertBrain(); return this.chains[this.clock.steps]; }

    /** One exact step. With an async brain: sense, await its run, then its pass, act and the rest (§8.4). */
    public async step(): Promise<void> {
        this.assertAlive();
        this.assertBrain();
        this.throwFailure();
        this.upload();
        if (!this.runs.size) {
            this.engine.render(this.chains[1]);
        } else {
            // An interactive run in flight first, so that two runs of one brain never overlap.
            await Promise.all([...this.runs.values()].map(run => run.pending));
            this.throwFailure();
            this.engine.render(this.halves[0]);
            await Promise.all([...this.runs].filter(([, run]) => !run.failed).map(([brain, run]) => this.start(brain, run)));
            this.throwFailure();
            this.engine.render(this.halves[1]);
        }
        await this.engine.device.queue.onSubmittedWorkDone();
    }

    public destroy(): void {
        if (this.destroyed) return;
        this.destroyed = true;
        this.grid.destroy();
        for (const stage of this.stages) stage.enabled = false;
        // A run not yet queued still reads obs: release the buffers once every run has settled.
        const release = () => { for (const buffer of this.owned) buffer.destroy(); };
        if (this.inflight.size) void Promise.all(this.inflight.keys()).then(release);
        else release();
    }

    private assertAlive(): void {
        if (this.destroyed) throw new Error('AgentSystem: used after destroy().');
    }

    private assertBrain(): void {
        if (this.everySpecies) return;
        const missing = this.definition.species.find(s => !this.brainOf[s.index])!;
        throw new Error(`AgentSystem: no brain for species "${missing.name}"; call setBrain(brain, { species: '${missing.name}' }).`);
    }
}