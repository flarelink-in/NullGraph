// src/data/engine-demos/demos/AgentsExperiments/HerdWolvesGPU/flashes.ts
//
// Catch flashes: one expanding ring per recent catch, read straight from the GPU. The respawn stage
// writes each catch into a ring buffer (x, z, step); this batch draws CATCH_RING screen-space quads
// and ages each against AgentSystem's GPU tick. Nothing about a catch ever reaches the CPU.

/* eslint-disable no-bitwise -- GPU buffer usage flags are bit masks */

import type { NullGraph, RenderBatch, RenderPassNode } from 'null-graph';

import { AGENT_STATE_WGSL } from 'null-graph/agents';

import { CATCH_RING } from './brain';

/** Seconds a flash stays on screen. */
const LIFE = 1.1;

export interface CatchFlashes {
  readonly batch: RenderBatch;
  /** Call when the canvas size changes: the rings are sized in pixels. */
  resize(width: number, height: number): void;
  destroy(): void;
}

export function catchFlashes(
  engine: NullGraph,
  pass: RenderPassNode,
  ring: GPUBuffer,
  state: GPUBuffer,
  dt: number
): CatchFlashes {
  const { device } = engine;
  const shaderCode = `${AGENT_STATE_WGSL}
@group(0) @binding(0) var<uniform> camera: array<vec4<f32>, 27>;
@group(0) @binding(1) var<storage, read> instances: array<vec4<f32>>;
@group(1) @binding(0) var<storage, read> catches: array<vec4<f32>, ${CATCH_RING}>;
@group(1) @binding(1) var<storage, read> agentState: AgentState;
@group(1) @binding(2) var<uniform> view: vec4<f32>;
struct VertexOut { @builtin(position) position: vec4<f32>, @location(0) local: vec2<f32>, @location(1) fade: f32 };
@vertex fn vs_main(@builtin(vertex_index) v: u32, @builtin(instance_index) k: u32) -> VertexOut {
    var out: VertexOut;
    out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    out.local = vec2<f32>(0.0);
    out.fade = 0.0;
    let e = catches[k];
    let age = (f32(agentState.tick) - e.z) * ${dt} + instances[0].x;
    if (age < 0.0 || age > ${LIFE}) { return out; }
    let clip = mat4x4<f32>(camera[0], camera[1], camera[2], camera[3]) * vec4<f32>(e.x, 0.1, e.y, 1.0);
    if (clip.w <= 0.0) { return out; }
    let corners = array<vec2<f32>, 4>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, 1.0));
    let t = age / ${LIFE};
    let radiusPx = mix(4.0, 26.0, sqrt(t));
    out.position = clip + vec4<f32>(corners[v] * radiusPx * 2.0 / view.xy * clip.w, 0.0, 0.0);
    out.local = corners[v];
    out.fade = 1.0 - t;
    return out;
}
@fragment fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let r = length(in.local);
    let a = smoothstep(0.55, 0.78, r) * (1.0 - smoothstep(0.84, 1.0, r)) * in.fade;
    return vec4<f32>(vec3<f32>(1.0, 0.74, 0.3) * a, a);
}`;
  const premultiplied: GPUBlendComponent = {
    srcFactor: 'one',
    dstFactor: 'one-minus-src-alpha',
    operation: 'add',
  };
  const batch = engine.createBatch(pass, {
    label: 'Herd catch flashes',
    shaderCode,
    strideFloats: 4,
    maxInstances: 1,
    vertexLayouts: [{ arrayStride: 4, attributes: [] }],
    blend: { color: premultiplied, alpha: premultiplied },
    depthWriteEnabled: false,
    depthCompare: 'always',
  });
  const view = device.createBuffer({
    label: 'Herd catch flashes view',
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const vertices = device.createBuffer({
    label: 'Herd catch flashes procedural vertices',
    size: 16,
    usage: GPUBufferUsage.VERTEX,
  });
  const indices = device.createBuffer({
    label: 'Herd catch flashes quad',
    size: 12,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(indices, 0, new Uint16Array([0, 1, 2, 2, 1, 3]));
  batch.vertexBuffer = vertices;
  batch.indexBuffer = indices;
  batch.indexCount = 6;
  batch.currentInstanceCount = CATCH_RING;
  engine.setRenderPassBindGroup(
    batch,
    [
      { binding: 0, resource: { buffer: ring } },
      { binding: 1, resource: { buffer: state } },
      { binding: 2, resource: { buffer: view } },
    ],
    1
  );
  const viewData = new Float32Array(4);
  let destroyed = false;
  return {
    batch,
    resize(width: number, height: number) {
      viewData[0] = Math.max(1, width);
      viewData[1] = Math.max(1, height);
      device.queue.writeBuffer(view, 0, viewData);
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      engine.clearBatch(pass, batch);
      indices.destroy();
      vertices.destroy();
      view.destroy();
      batch.storageBuffer.destroy();
    },
  };
}
