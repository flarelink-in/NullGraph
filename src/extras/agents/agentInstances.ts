// src/extras/agents/agentInstances.ts
//
// Step 4's instance stage (docs/agents-gpu-loop-design.md §9.2): agent records → VisibilitySystem
// rows, once per frame after the last step, and its CPU twin (U19), which is the definition A17
// compares the GPU against. Pure: no device, no visibility import — the row layout below is
// visibility's Instance (24 floats), pinned against it by U19.

import {
    AGENT_STRUCT_WGSL, AGENT_FLOATS, AGENT_POSITION_OFFSET, AGENT_HEADING_OFFSET, AGENT_SPECIES_OFFSET,
    AGENT_FLAGS_OFFSET, AGENT_FLAG_ALIVE, HIDDEN_MESH, type AgentDefinition,
} from './agentLayout';
import type { RecordViews } from './agentMath';

/** Floats per row of `vis.instanceBuffer`: model (16), bounds (4), meshIndex, LOD, flags, sourceIndex. */
export const INSTANCE_ROW_FLOATS = 24;
/** The row's meshIndex word. */
export const INSTANCE_ROW_MESH_OFFSET = 20;
/** Visibility's `LOD_AUTO`: the cull picks the level. */
export const INSTANCE_ROW_LOD_AUTO = 0xFFFFFFFF;
/** "No agent" in the highlight uniform. */
export const HIGHLIGHT_NONE = 0xFFFFFFFF;

/** Agent indices to draw with their species' selected and hover meshes; -1 for none. */
export interface AgentHighlight {
    selected: number;
    hovered: number;
}

export const HIGHLIGHT_WGSL = `
struct Highlight {
    selected: u32,
    hovered: u32,
    _pad: vec2<u32>,
};
`;

/** The species that would draw a mesh with a zero (or invalid) bounding radius — the cull drops those. */
export function assertInstanceBounds(def: AgentDefinition): void {
    for (const s of def.species) {
        const drawn = [s.mesh, s.hoverMesh, s.selectedMesh].find(m => m !== HIDDEN_MESH);
        if (drawn !== undefined && !(s.boundsRadius > 0)) {
            throw new Error(`AgentSystem: species "${s.name}" draws mesh ${drawn} with boundsRadius ${s.boundsRadius} — ` +
                `the cull would drop it while on screen. Set boundsRadius to the mesh's radius about its origin.`);
        }
    }
}

/** One invocation per agent writes row `firstInstance + i`. A dead agent gets only meshIndex = HIDDEN. */
export function buildAgentInstanceWGSL(def: AgentDefinition, firstInstance: number): string {
    return `${AGENT_STRUCT_WGSL}${HIGHLIGHT_WGSL}
struct InstanceRow {
    model: mat4x4<f32>,
    bounds: vec4<f32>,
    meshIndex: u32,
    lodOverride: u32,
    flags: u32,
    sourceIndex: u32,
};
@group(0) @binding(0) var<storage, read> agents: array<Agent>;
@group(0) @binding(1) var<storage, read_write> rows: array<InstanceRow>;
@group(0) @binding(2) var<uniform> species: array<Species, 16>;
@group(0) @binding(3) var<uniform> highlight: Highlight;
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= ${def.capacity}u) { return; }
    let a = agents[i];
    let r = ${firstInstance}u + i;
    if ((a.ids.y & ${AGENT_FLAG_ALIVE}u) == 0u) {
        rows[r].meshIndex = ${HIDDEN_MESH}u;
        return;
    }
    let sp = species[a.ids.x];
    let s = sin(a.velHeading.w);
    let c = cos(a.velHeading.w);
    rows[r].model = mat4x4<f32>(vec4<f32>(c, 0.0, -s, 0.0), vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(s, 0.0, c, 0.0), vec4<f32>(a.posRadius.xyz, 1.0));
    rows[r].bounds = vec4<f32>(0.0, 0.0, 0.0, sp.boundsRadius);
    var mesh = sp.meshIndex;
    if (i == highlight.selected) { mesh = sp.selectedMesh; }
    else if (i == highlight.hovered) { mesh = sp.hoverMesh; }
    rows[r].meshIndex = mesh;
    rows[r].lodOverride = ${INSTANCE_ROW_LOD_AUTO}u;
    rows[r].flags = 0u;
    rows[r].sourceIndex = 0u;
}`;
}

/**
 * U19: the instance stage's CPU twin. Writes row `row` of `out` from record `i`:
 * `translate(position) · rotateY(heading)`, so local +Z is the forward `(sin h, 0, cos h)` and local
 * +X the side `(cos h, 0, −sin h)`; bounds `[0, 0, 0, boundsRadius]`; the mesh by the highlight.
 * A dead agent's row gets only meshIndex = HIDDEN_MESH, as on the GPU.
 */
export function writeAgentInstanceRow(def: AgentDefinition, rec: RecordViews, i: number,
    highlight: AgentHighlight, out: Float32Array, row: number,
    outU32: Uint32Array = new Uint32Array(out.buffer, out.byteOffset, out.length)): void {
    const d = rec.data, u = rec.u32, b = i * AGENT_FLOATS, o = row * INSTANCE_ROW_FLOATS;
    if ((u[b + AGENT_FLAGS_OFFSET] & AGENT_FLAG_ALIVE) === 0) {
        outU32[o + INSTANCE_ROW_MESH_OFFSET] = HIDDEN_MESH;
        return;
    }
    const sp = def.species[u[b + AGENT_SPECIES_OFFSET]];
    const h = d[b + AGENT_HEADING_OFFSET], s = Math.sin(h), c = Math.cos(h);
    out.set([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0,
        d[b + AGENT_POSITION_OFFSET], d[b + AGENT_POSITION_OFFSET + 1], d[b + AGENT_POSITION_OFFSET + 2], 1,
        0, 0, 0, sp.boundsRadius], o);
    let mesh = sp.mesh;
    if (i === highlight.selected) mesh = sp.selectedMesh;
    else if (i === highlight.hovered) mesh = sp.hoverMesh;
    outU32[o + INSTANCE_ROW_MESH_OFFSET] = mesh;
    outU32[o + 21] = INSTANCE_ROW_LOD_AUTO;
    outU32[o + 22] = 0;
    outU32[o + 23] = 0;
}
