import type { NullGraph, RenderPassNode } from '../../core';
import type { AgentSystem } from '../agents/AgentSystem';
import { AGENT_STRUCT_WGSL, AGENT_FLAG_ALIVE } from '../agents/agentLayout';
import type { GridDrawing } from './gridDrawing';

export interface AgentArrowsOptions {
    /** One RGB color per species; missing colors use the repeating default palette. */
    colors?: readonly (readonly [number, number, number])[];
}

const PALETTE = [[0.25, 0.8, 1], [1, 0.18, 0.45], [1, 0.8, 0.2], [0.35, 1, 0.5]] as const;

/** One procedural triangle per agent, reading live GPU records without uploads or readback. */
export function agentArrows(engine: NullGraph, pass: RenderPassNode, agents: AgentSystem,
    options: AgentArrowsOptions = {}): GridDrawing {
    if (agents.engine !== engine) throw new Error('agentArrows: agents must belong to this engine.');
    agents.grid.getComputeStages(); // Refuse a destroyed system without changing its selected chain.
    const colors = agents.definition.species.map((_, i) => options.colors?.[i] ?? PALETTE[i % PALETTE.length]);
    if (options.colors?.some(c => c.length !== 3 || !c.every(v => Number.isFinite(v) && v >= 0 && v <= 1))) {
        throw new Error('agentArrows: colors must be finite RGB triples in [0, 1].');
    }
    const shaderCode = AGENT_STRUCT_WGSL + `
@group(0) @binding(0) var<uniform> camera: array<vec4<f32>, 27>;
@group(0) @binding(1) var<storage, read> instances: array<vec4<f32>>;
@group(1) @binding(0) var<storage, read> agents: array<Agent>;
struct VertexOut { @builtin(position) position: vec4<f32>, @location(0) color: vec3<f32> };
@vertex fn vs_main(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> VertexOut {
    let a = agents[i];
    var out: VertexOut;
    out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    out.color = vec3<f32>(0.0);
    if ((a.ids.y & ${AGENT_FLAG_ALIVE}u) == 0u) { return out; }
    let colors = array<vec3<f32>, ${colors.length}>(${colors.map(c => `vec3<f32>(${c.join(',')})`).join(',')});
    let corners = array<vec2<f32>, 3>(vec2<f32>(0.0, 1.6), vec2<f32>(-0.7, -0.8), vec2<f32>(0.7, -0.8));
    let p = corners[v] * a.posRadius.w;
    let s = sin(a.velHeading.w); let c = cos(a.velHeading.w);
    let world = a.posRadius.xyz + vec3<f32>(p.x * c + p.y * s, 0.04 + instances[0].x, -p.x * s + p.y * c);
    out.position = mat4x4<f32>(camera[0], camera[1], camera[2], camera[3]) * vec4<f32>(world, 1.0);
    out.color = colors[min(a.ids.x, ${colors.length - 1}u)];
    return out;
}
@fragment fn fs_main(in: VertexOut) -> @location(0) vec4<f32> { return vec4<f32>(in.color, 1.0); }
`;
    const batch = engine.createBatch(pass, { label: 'Agent arrows', shaderCode, strideFloats: 4,
        maxInstances: agents.capacity, vertexLayouts: [{ arrayStride: 4, attributes: [] }],
        depthWriteEnabled: true, depthCompare: 'less-equal' });
    const vertices = engine.device.createBuffer({ label: 'Agent arrows procedural vertices', size: 12,
        usage: GPUBufferUsage.VERTEX });
    const indices = engine.device.createBuffer({ label: 'Agent arrows triangle', size: 8,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    engine.device.queue.writeBuffer(indices, 0, new Uint16Array([0, 1, 2, 0]));
    batch.vertexBuffer = vertices;
    batch.indexBuffer = indices;
    batch.indexCount = 3;
    batch.currentInstanceCount = agents.capacity;
    engine.setRenderPassBindGroup(batch, [{ binding: 0, resource: { buffer: agents.buffer } }], 1);
    let destroyed = false;
    return { batch, destroy() {
        if (destroyed) return;
        destroyed = true;
        engine.clearBatch(pass, batch);
        indices.destroy();
        vertices.destroy();
        batch.storageBuffer.destroy();
    } };
}