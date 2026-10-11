import type { NullGraph, ComputeStageNode } from '../../core';
import { PointGrid } from '../spatial/PointGrid';
import { AGENT_FLOATS, AGENT_FLAGS_OFFSET, AGENT_FLAG_ALIVE, type AgentDefinition } from './agentLayout';
import { buildAgentGridQueryWGSL } from './agentGridShaders';

/** Grid + fixed nearest query; no CPU updates/readbacks in the frame path. Query output is
 * capacity × k u32 indices, nearest-first, padded with 0xffffffff. */
export class AgentGrid extends PointGrid {
    public readonly neighbours: GPUBuffer;
    private readonly sequence: ComputeStageNode[];
    private queryDestroyed = false;

    constructor(engine: NullGraph, records: GPUBuffer, public readonly definition: AgentDefinition,
        public readonly mask = 0xffffffff, public readonly radius = definition.cellSize, public readonly k = 8) {
        const code = buildAgentGridQueryWGSL(definition, mask, radius, k);
        if (definition.capacity * k * 4 > engine.device.limits.maxStorageBufferBindingSize) {
            throw new Error('AgentGrid: neighbour buffer exceeds device storage limit.');
        }
        super(engine, records, { ...definition, stride: AGENT_FLOATS,
            flagsOffset: AGENT_FLAGS_OFFSET, aliveMask: AGENT_FLAG_ALIVE });
        this.neighbours = engine.device.createBuffer({ label: 'AgentGrid neighbours',
            size: definition.capacity * k * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const query = engine.createComputeStage({ name: 'AgentGrid nearest', shaderCode: code,
            dispatchCount: [Math.ceil(definition.capacity / 64), 1, 1] });
        engine.setComputePassBindGroup(query, [records, this.cellCounts, this.cellOffsets, this.cellBodies, this.neighbours]
            .map((buffer, binding) => ({ binding, resource: { buffer } })), 0);
        this.sequence = [...super.getComputeStages(), query];
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
        super.destroy();
    }
}