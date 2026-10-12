import type { NullGraph, ComputeStageNode } from '../../core';
import { PointGrid } from '../spatial/PointGrid';
import { AGENT_FLOATS, AGENT_FLAGS_OFFSET, AGENT_FLAG_ALIVE, type AgentDefinition } from './agentLayout';
import { buildAgentGridQueryWGSL } from './agentGridShaders';

/** Grid + fixed nearest query; no CPU updates/readbacks in the frame path. Query output is
 * capacity × k u32 indices, nearest-first, padded with 0xffffffff. */
export class AgentGrid extends PointGrid {
    public readonly neighbours: GPUBuffer;
    private readonly tick: GPUBuffer;
    private readonly sequence: ComputeStageNode[];
    private queryDestroyed = false;

    constructor(engine: NullGraph, records: GPUBuffer, public readonly definition: AgentDefinition,
        public readonly mask = 0xffffffff, public readonly radius = definition.cellSize, public readonly k = 8) {
        const code = buildAgentGridQueryWGSL(definition, mask, radius, k);
        if (definition.capacity * k * 4 > Math.min(engine.device.limits.maxStorageBufferBindingSize,
            engine.device.limits.maxBufferSize)) {
            throw new Error('AgentGrid: neighbour buffer exceeds device storage limit.');
        }
        if (engine.device.limits.maxStorageBuffersPerShaderStage < 6) {
            throw new Error('AgentGrid: query requires six storage buffer bindings.');
        }
        super(engine, records, { ...definition, stride: AGENT_FLOATS,
            flagsOffset: AGENT_FLAGS_OFFSET, aliveMask: AGENT_FLAG_ALIVE });
        const buffers: GPUBuffer[] = [], stages: ComputeStageNode[] = [];
        try {
            this.neighbours = engine.device.createBuffer({ label: 'AgentGrid neighbours',
                size: definition.capacity * k * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
            buffers.push(this.neighbours);
            this.tick = engine.device.createBuffer({ label: 'AgentGrid tick', size: 4, usage: GPUBufferUsage.STORAGE });
            buffers.push(this.tick);
            const query = engine.createComputeStage({ name: 'AgentGrid nearest', shaderCode: code,
                dispatchCount: [Math.ceil(definition.capacity / 64), 1, 1] });
            stages.push(query);
            engine.setComputePassBindGroup(query, [records, this.cellCounts, this.cellOffsets, this.cellBodies, this.neighbours, this.tick]
                .map((buffer, binding) => ({ binding, resource: { buffer } })), 0);
            const tick = engine.createComputeStage({ name: 'AgentGrid tick', shaderCode: `
@group(0) @binding(0) var<storage, read_write> tick: u32;
@compute @workgroup_size(1) fn cs_main() { tick++; }`, dispatchCount: [1, 1, 1] });
            stages.push(tick);
            engine.setComputePassBindGroup(tick, [{ binding: 0, resource: { buffer: this.tick } }], 0);
            this.sequence = [...super.getComputeStages(), query, tick];
        } catch (error) {
            for (const stage of stages) stage.enabled = false;
            for (const buffer of buffers) buffer.destroy();
            super.destroy();
            throw error;
        }
    }

    public override getComputeStages(): ComputeStageNode[] {
        if (this.queryDestroyed) throw new Error('AgentGrid: used after destroy().');
        return this.sequence;
    }

    public override destroy(): void {
        if (this.queryDestroyed) return;
        this.queryDestroyed = true;
        for (const stage of this.sequence) stage.enabled = false;
        this.neighbours.destroy();
        this.tick.destroy();
        super.destroy();
    }
}