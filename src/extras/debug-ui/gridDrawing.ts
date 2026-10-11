import type { NullGraph, RenderPassNode, RenderBatch } from '../../core';
import type { PointGrid } from '../spatial/PointGrid';

export interface GroundGridOptions {
    min: [number, number];
    max: [number, number];
    height?: number;
    minorSpacing?: number;
    majorSpacing?: number;
    fadeStart?: number;
    fadeEnd?: number;
}

export interface GridDrawing {
    readonly batch: RenderBatch;
    /** Removes this batch and releases only its own resources. */
    destroy(): void;
}

// Camera words 0..15 = viewProj, 96..98 = eye. Group 0 binding 1 is engine-owned
// instance storage and must be referenced for the standard batch binding layout.
const HEADER = `
@group(0) @binding(0) var<uniform> camera: array<vec4<f32>, 27>;
@group(0) @binding(1) var<storage, read> instances: array<vec4<f32>>;
struct VertexOut { @builtin(position) position: vec4<f32>, @location(0) world: vec2<f32> };
`;

function vertex(min: [number, number], max: [number, number], height: number): string {
    return `
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> VertexOut {
    let corners = array<vec2<f32>, 6>(vec2<f32>(0,0), vec2<f32>(1,0), vec2<f32>(0,1),
        vec2<f32>(0,1), vec2<f32>(1,0), vec2<f32>(1,1));
    let p = mix(vec2<f32>(${min}), vec2<f32>(${max}), corners[i]);
    let vp = mat4x4<f32>(camera[0], camera[1], camera[2], camera[3]);
    var out: VertexOut;
    out.position = vp * vec4<f32>(p.x, ${height} + instances[0].x, p.y, 1.0);
    out.world = p;
    return out;
}`;
}

function drawing(engine: NullGraph, pass: RenderPassNode, code: string, label: string): GridDrawing {
    const batch = engine.createBatch(pass, { label, shaderCode: code, strideFloats: 4, maxInstances: 1,
        vertexLayouts: [{ arrayStride: 4, attributes: [] }],
        depthWriteEnabled: false, depthCompare: 'less-equal' });
    const vertices = engine.device.createBuffer({ label: `${label} procedural vertices`, size: 24,
        usage: GPUBufferUsage.VERTEX });
    const indices = engine.device.createBuffer({ label: `${label} triangles`, size: 12,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    engine.device.queue.writeBuffer(indices, 0, new Uint16Array([0, 1, 2, 3, 4, 5]));
    batch.indexBuffer = indices;
    batch.vertexBuffer = vertices;
    batch.indexCount = 6;
    batch.currentInstanceCount = 1;
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

/** Minor/major floor lines, derivative anti-aliasing, distance fade and field edge.
 * Extracted from the Herd & Wolves preview's floor shader; no DOM dependency. */
export function groundGrid(engine: NullGraph, pass: RenderPassNode, options: GroundGridOptions): GridDrawing {
    const { min, max } = options, height = options.height ?? 0;
    const minor = options.minorSpacing ?? 2, major = options.majorSpacing ?? 10;
    const start = options.fadeStart ?? 30, end = options.fadeEnd ?? 110;
    if (!(min.every(Number.isFinite) && max.every(Number.isFinite) && max[0] > min[0] && max[1] > min[1] &&
        Number.isFinite(height) && Number.isFinite(minor) && minor > 0 && Number.isFinite(major) && major > 0 &&
        Number.isFinite(start) && start >= 0 && Number.isFinite(end) && end > start)) {
        throw new Error('groundGrid: invalid bounds, spacing, height or fade range.');
    }
    return drawing(engine, pass, HEADER + vertex(min, max, height) + `
fn gridLine(p: vec2<f32>, spacing: f32) -> f32 {
    let q = p / spacing;
    let w = max(fwidth(q), vec2<f32>(1e-4));
    let g = abs(fract(q - 0.5) - 0.5) / w;
    return 1.0 - min(min(g.x, g.y), 1.0);
}
@fragment fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let p = in.world;
    let minor = gridLine(p, ${minor});
    let major = gridLine(p, ${major});
    let edgeDistance = min(min(p.x - ${min[0]}, ${max[0]} - p.x), min(p.y - ${min[1]}, ${max[1]} - p.y));
    let edge = 1.0 - smoothstep(0.0, 0.6, edgeDistance);
    let fade = 1.0 - smoothstep(${start}, ${end}, distance(camera[24].xz, p));
    let c = vec3<f32>(0.04, 0.05, 0.065) + vec3<f32>(0.07, 0.11, 0.15) * minor
        + vec3<f32>(0.14, 0.26, 0.36) * major + vec3<f32>(0.3, 0.5, 0.6) * edge;
    return vec4<f32>(mix(vec3<f32>(0.05), c, fade), 1.0);
}`, 'Ground grid');
}

/** Reads cellCounts directly in the fragment shader. Empty cells are dark, occupied cells blue,
 * over-budget cells red. Draw after the grid compute stages; no CPU occupancy uploads/readback. */
export function gridOccupancy(engine: NullGraph, pass: RenderPassNode, grid: PointGrid, height = 0.01): GridDrawing {
    if (grid.engine !== engine || !Number.isFinite(height)) {
        throw new Error('gridOccupancy: grid must belong to this engine and height must be finite.');
    }
    grid.getComputeStages(); // Refuse destroyed grids before creating a batch.
    const o = grid.options, { cellsX, cellsZ } = o.grid;
    const min: [number, number] = [o.world.minX, o.world.minZ];
    const max: [number, number] = [min[0] + cellsX * o.cellSize, min[1] + cellsZ * o.cellSize];
    const result = drawing(engine, pass, HEADER + vertex(min, max, height) + `
@group(1) @binding(0) var<storage, read> counts: array<u32>;
@fragment fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let q = (in.world - vec2<f32>(${min})) / ${o.cellSize};
    let c = clamp(vec2<i32>(floor(q)), vec2<i32>(0), vec2<i32>(${cellsX - 1}, ${cellsZ - 1}));
    let count = counts[u32(c.y) * ${cellsX}u + u32(c.x)];
    let border = min(min(fract(q).x, 1.0 - fract(q).x), min(fract(q).y, 1.0 - fract(q).y));
    let fill = smoothstep(0.0, 0.04, border);
    let intensity = min(1.0, f32(count) / ${o.maxScanPerCell}.0);
    var color = mix(vec3<f32>(0.025, 0.03, 0.04), vec3<f32>(0.05, 0.35, 0.8), intensity);
    if (count > ${o.maxScanPerCell}u) { color = vec3<f32>(0.9, 0.08, 0.04); }
    return vec4<f32>(color * mix(0.35, 1.0, fill), 1.0);
}`, 'Grid occupancy');
    engine.setRenderPassBindGroup(result.batch, [{ binding: 0, resource: { buffer: grid.cellCounts } }], 1);
    return result;
}