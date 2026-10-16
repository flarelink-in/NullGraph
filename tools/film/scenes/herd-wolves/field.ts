/* eslint-disable no-bitwise -- GPU usage flags */
import type { NullGraph, RenderPassNode } from 'null-graph';

import { rand01 } from 'null-graph/agents';
import { ToonMaterial } from 'null-graph/materials';
import { StandardLayout, MegabufferBuilder } from 'null-graph/geometry';
import { InstanceBuffer, VisibilitySystem } from 'null-graph/visibility';

import { fenceRows, fenceMeshes } from './fence';

import type { HerdModels } from './models';

export const FIELD_FRAGMENT = /* wgsl */ `
@fragment fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let p = in.worldPos.xz;
    let patches = sin(p.x * 0.13 + sin(p.y * 0.09)) * cos(p.y * 0.11);
    let bands = smoothstep(-0.15, 0.3, patches);
    let base = mix(vec3<f32>(0.14, 0.32, 0.09), vec3<f32>(0.32, 0.5, 0.16), bands);
    return vec4<f32>(base, 1.0);
}`;

export const GRASS_FRAGMENT = /* wgsl */ `
@fragment fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let sun = abs(dot(normalize(in.worldNormal), normalize(vec3<f32>(0.4, 1.0, 0.3))));
    let ramp = 0.6 + 0.4 * smoothstep(0.4, 0.48, sun);
    return vec4<f32>(vec3<f32>(0.25, 0.46, 0.1) * ramp, 1.0);
}`;

/**
 * Static scenery: populated once per rebuild, then frustum-culled on the GPU. The field, the grass,
 * and the fence around the agents' world (fence.ts), its posts and rails in stock toon wood.
 */
export function createHerdField(
  engine: NullGraph, pass: RenderPassNode, half: number, grass: HerdModels['grass'],
  targetFormat?: GPUTextureFormat
) {
  // Bounded decoration density, not an agent-count restriction. Large fields spread the same
  // patch budget farther apart instead of multiplying geometry with a million-agent stress run.
  const count = 768;
  const fence = fenceRows(half);
  const vis = new VisibilitySystem(engine, { maxInstances: count + 1 + fence.length });
  const mega = new MegabufferBuilder(StandardLayout.strideFloats);
  mega.addMesh('field', new Float32Array([
    -half, -0.02, -half, 0, 1, 0, 0, 0,
    half, -0.02, -half, 0, 1, 0, 1, 0,
    -half, -0.02, half, 0, 1, 0, 0, 1,
    half, -0.02, half, 0, 1, 0, 1, 1,
  ]), new Uint16Array([0, 2, 1, 1, 2, 3]));
  mega.addMesh('grass', grass.vertices, grass.indices);
  const { post, rail } = fenceMeshes();
  mega.addMesh('post', post.vertices, post.indices);
  mega.addMesh('rail', rail.vertices, rail.indices);
  vis.setGeometry(mega);
  const rows = new InstanceBuffer(count + 1 + fence.length);
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  rows.write(0, identity, [0, 0, 0, half * Math.SQRT2], 0);
  let radius = 0;
  for (let v = 0; v < grass.vertices.length; v += 8)
    radius = Math.max(radius, Math.hypot(...grass.vertices.subarray(v, v + 3)));
  for (let i = 1; i <= count; i += 1) {
    const a = rand01(i, 83, 0) * Math.PI * 2;
    const s = 0.65 + rand01(i, 83, 1) * 0.7;
    const x = (rand01(i, 83, 2) * 2 - 1) * (half - 2);
    const z = (rand01(i, 83, 3) * 2 - 1) * (half - 2);
    rows.write(i, [Math.cos(a) * s, 0, -Math.sin(a) * s, 0, 0, s, 0, 0,
      Math.sin(a) * s, 0, Math.cos(a) * s, 0, x, 0, z, 1], [0, 0, 0, radius], 1);
  }
  fence.forEach((row, k) => rows.write(count + 1 + k, row.model, row.bounds, row.mesh === 'post' ? 2 : 3));
  vis.setInstances(rows.data, rows.count);
  // Weathered wood, toon-ramped under the material's own sun; posts darker than their rails.
  const wood = (rgb: [number, number, number]) =>
    new ToonMaterial(engine, {
      baseColor: [...rgb, 1],
      rampSteps: 3,
      rimLightColor: [0.1, 0.08, 0.06],
      rimLightPower: 3,
      specularThreshold: 0.98,
    });
  const posts = wood([0.2, 0.13, 0.08]);
  const rails = wood([0.25, 0.18, 0.11]);
  const batches = [
    vis.createInstanceDrawBatch(pass, 'field', 0, { targetFormat, fragmentBody: FIELD_FRAGMENT }),
    vis.createInstanceDrawBatch(pass, 'grass', 1, { targetFormat, fragmentBody: GRASS_FRAGMENT }),
    vis.createInstanceDrawBatch(pass, 'post', 2, { targetFormat, material: { kind: 'toon', instance: posts } }),
    vis.createInstanceDrawBatch(pass, 'rail', 3, { targetFormat, material: { kind: 'toon', instance: rails } }),
  ];
  vis.upload();
  // Drop the CPU instance source after its first upload: scenery never moves.
  vis.setInstanceCount(rows.count);
  return {
    upload: () => vis.upload(),
    getComputeStages: () => vis.getComputeStages(),
    destroy() {
      batches.forEach((b) => vis.removeDrawBatch(b));
      vis.destroy();
      posts.destroy();
      rails.destroy();
    },
  };
}