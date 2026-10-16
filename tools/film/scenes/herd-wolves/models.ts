// D1.3: shared rigid poses; only one accumulated phase float per simulated animal.
/* eslint-disable no-bitwise -- WebGPU usage flags */
import type { NullGraph, RenderBatch } from 'null-graph';
import type { RigidAnimation } from 'null-graph/loaders';
import type { InstanceVertexHook } from 'null-graph/visibility';

import { INSTANCE_STRUCT_WGSL } from 'null-graph/visibility';
import { Primitives, StandardLayout } from 'null-graph/geometry';
import { GLBParser, BAKED_POSE_WGSL, BAKED_SAMPLE_WGSL, bakeRigidAnimation, BAKED_PALETTE_READ_WGSL, BAKED_PALETTE_WRITE_WGSL } from 'null-graph/loaders';

const CLIPS = ['idle', 'walk', 'run', 'eat'];

export async function loadHerdModels(engine: NullGraph) {
  // Start one shared public palette request, only when Models is first requested.
  const palette = engine.textureManager.load('/textures/colormap.png', { sRGB: true });
  const load = async (name: string, length: number) => {
    const response = await fetch(`/3dAssetsNotIncluded/animal-${name}.glb`);
    if (!response.ok) throw new Error(`Cannot load ${name}: HTTP ${response.status}`);
    const glb = GLBParser.parse(await response.arrayBuffer());
    try {
      const baked = bakeRigidAnimation(glb, { clips: CLIPS, fps: 30 });
      // These GLBs reference an absent external Textures/colormap.png; the parser reports an
      // empty URL. Use the installed site palette with the explicit swatch mapping below.
      const texture = await palette;
      let min = Infinity;
      let max = -Infinity;
      for (let v = 2; v < baked.vertices.length; v += 8) {
        min = Math.min(min, baked.vertices[v]);
        max = Math.max(max, baked.vertices[v]);
      }
      const scale = length / (max - min);
      // Bake into metre-sized geometry, so rest-pose picking uses the same scale as the draw.
      for (let v = 0; v < baked.vertices.length; v += 8)
        for (let k = 0; k < 3; k += 1) baked.vertices[v + k] *= scale;
      for (let k = 3; k < baked.transforms.length; k += 4) baked.transforms[k] *= scale;
      return { ...baked, texture, radius: baked.bounds.radius * scale };
    } finally {
      glb.imageUrls?.forEach((url) => URL.revokeObjectURL(url));
    }
  };
  const grass = async () => {
    const response = await fetch('/3dAssetsNotIncluded/patch-grass-foliage.glb');
    if (!response.ok) throw new Error(`Cannot load grass: HTTP ${response.status}`);
    const glb = GLBParser.parse(await response.arrayBuffer());
    try {
      const mesh = glb.meshes?.[0];
      if (!mesh) throw new Error('Grass asset has no geometry');
      // The stand-in has one rigid patch; no animation table or per-frame CPU updates.
      return { vertices: mesh.vertices, indices: mesh.indices };
    } finally {
      glb.imageUrls?.forEach((url) => URL.revokeObjectURL(url));
    }
  };
  const [prey, wolf, foliage] = await Promise.all([load('cow', 1.3), load('fox', 1.7), grass(), palette]);
  return { prey, wolf, grass: foliage };
}

export type HerdModels = Awaited<ReturnType<typeof loadHerdModels>>;

/** Below this size on screen (px, its bounding sphere's diameter) an animal draws as a cube. */
export const ANIMAL_LOD_PX = 22;
/** The far cubes' colours: the cow white, the fox orange. */
export const ANIMAL_CUBE_RGB = { prey: [0.93, 0.92, 0.88], wolf: [0.98, 0.52, 0.16] } as const;

/** The cube an animal becomes far away: its rest pose's box, shrunk to its bulk. */
export function animalCube(model: { vertices: Float32Array }) {
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let v = 0; v < model.vertices.length; v += 8)
    for (let k = 0; k < 3; k += 1) {
      lo[k] = Math.min(lo[k], model.vertices[v + k]);
      hi[k] = Math.max(hi[k], model.vertices[v + k]);
    }
  const cube = Primitives.createCube(
    StandardLayout,
    (hi[0] - lo[0]) * 0.8,
    (hi[1] - lo[1]) * 0.75,
    (hi[2] - lo[2]) * 0.8
  );
  for (let v = 0; v < cube.vertices.length; v += StandardLayout.strideFloats)
    for (let k = 0; k < 3; k += 1) cube.vertices[v + k] += (lo[k] + hi[k]) / 2;
  return cube;
}

// After respawn, use the actual speed and species' current top speed. Hash once at tick zero;
// thereafter wrap an accumulated phase, never derive it from age or wall-clock time.
export const ANIMAL_PHASE = /* wgsl */ `
@group(1) @binding(0) var<storage, read_write> animalPhase: array<f32>;
fn step(i: u32, tick: u32) {
    let a = agents[i];
    let speed = clamp(length(a.velHeading.xz) / max(0.001, species[a.ids.x].maxSpeed * a.state.z), 0.0, 1.0);
    var rate = mix(2.0, 4.0, smoothstep(0.15, 0.7, speed));
    if (speed < 0.15) { rate = mix(1.0, 2.0, smoothstep(0.0, 0.15, speed)); }
    if (a.state.w > 0.0) { rate = 2.0; }
    if (tick == 0u) { animalPhase[i] = rand01(i, 0u, 19u); }
    animalPhase[i] = fract(animalPhase[i] + DT * rate);
}`;

const DEFORM = /* wgsl */ `
struct Pose { a: vec4<f32>, b: vec4<f32>, c: vec4<f32> };
fn samplePose(clip: u32, phase: f32, part: u32) -> Pose {
    let first = animalClips[clip].x;
    let frames = animalClips[clip].y;
    let f = fract(phase) * f32(frames);
    let a = ((first + u32(f)) * animalParts + part) * 3u;
    let b = ((first + (u32(f) + 1u) % frames) * animalParts + part) * 3u;
    let w = fract(f);
    return Pose(mix(animalPoses[a], animalPoses[b], w),
                mix(animalPoses[a + 1u], animalPoses[b + 1u], w),
                mix(animalPoses[a + 2u], animalPoses[b + 2u], w));
}
fn deform(pos: vec3<f32>, normal: vec3<f32>, vi: u32, inst: Instance) -> Deformed {
    let row = inst.sourceIndex;
    let record = row * 4u;
    let state = animalAgents[record + 2u];
    let speed = clamp(length(animalAgents[record + 1u].xz) / max(0.001, animalTopSpeed[0] * state.z), 0.0, 1.0);
    let phase = animalPhase[row];
    let part = animalPartOf[vi];
    var p: Pose;
    if (state.w > 0.0) { p = samplePose(3u, phase, part); }
    else if (speed < 0.15) {
        let idle = samplePose(0u, phase, part);
        let walk = samplePose(1u, phase, part);
        let w = smoothstep(0.0, 0.15, speed);
        p = Pose(mix(idle.a, walk.a, w), mix(idle.b, walk.b, w), mix(idle.c, walk.c, w));
    }
    else {
        let walk = samplePose(1u, phase, part);
        let run = samplePose(2u, phase, part);
        let w = smoothstep(0.15, 0.7, speed);
        p = Pose(mix(walk.a, run.a, w), mix(walk.b, run.b, w), mix(walk.c, run.c, w));
    }
    let v = vec4<f32>(pos, 1.0);
    return Deformed(vec3<f32>(dot(p.a, v), dot(p.b, v), dot(p.c, v)),
                    normalize(vec3<f32>(dot(p.a.xyz, normal), dot(p.b.xyz, normal), dot(p.c.xyz, normal))));
}`;

export function animalHook(
  engine: NullGraph,
  model: RigidAnimation,
  partOf: Uint32Array,
  agents: GPUBuffer,
  phase: GPUBuffer,
  topSpeed: GPUBuffer,
  own: <T extends { destroy(): void }>(resource: T) => T
): InstanceVertexHook {
  const upload = (data: Uint32Array | Float32Array) => {
    const b = own(engine.device.createBuffer({
      size: Math.max(16, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }));
    engine.device.queue.writeBuffer(b, 0, data);
    return b;
  };
  return {
    wgsl: `const animalParts = ${model.parts}u;\nconst animalClips = array<vec2<u32>, 4>(${model.clips.map((c) => `vec2<u32>(${c.firstFrame}u, ${c.frames}u)`).join(', ')});\n${DEFORM}`,
    bindings: `
@group(2) @binding(0) var<storage, read> animalPartOf: array<u32>;
@group(2) @binding(1) var<storage, read> animalPoses: array<vec4<f32>>;
@group(2) @binding(2) var<storage, read> animalAgents: array<vec4<f32>>;
@group(2) @binding(3) var<storage, read> animalPhase: array<f32>;
@group(2) @binding(4) var<storage, read> animalTopSpeed: array<f32>;`,
    entries: [upload(partOf), upload(model.transforms), agents, phase, topSpeed].map((buffer, binding) => ({ binding, resource: { buffer } })),
  };
}

/** Worst-case allocation, not last frame's survivor count. Keep large stress tests on the hook. */
export function animalPaletteFits(engine: NullGraph, models: HerdModels, count: number, copies: number) {
  const bytes = [models.prey, models.wolf].map((model) => count * model.parts * 48);
  const limit = Math.min(engine.device.limits.maxStorageBufferBindingSize, engine.device.limits.maxBufferSize);
  return bytes.every((size) => size <= limit) && (bytes[0] + bytes[1]) * copies <= 64 * 1024 * 1024;
}

/** One palette per compacted draw list; attach only after the batch's cull has been created. */
export function animalPalette(
  engine: NullGraph,
  model: RigidAnimation,
  partOf: Uint32Array,
  agents: GPUBuffer,
  phase: GPUBuffer,
  topSpeed: GPUBuffer,
  capacity: number,
  own: <T extends { destroy(): void }>(resource: T) => T
) {
  const { device } = engine;
  const upload = (data: Uint32Array | Float32Array) => {
    const buffer = own(device.createBuffer({ size: Math.max(16, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }));
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const palette = own(device.createBuffer({ label: 'Herd visible animation palette', size: capacity * model.parts * 48, usage: GPUBufferUsage.STORAGE }));
  const table = upload(model.transforms);
  const constants = `const animalParts = ${model.parts}u;
const animalClips = array<vec2<u32>, 4>(${model.clips.map((clip) => `vec2<u32>(${clip.firstFrame}u, ${clip.frames}u)`).join(', ')});`;
  const hook: InstanceVertexHook = {
    instanceIndex: true,
    bindings: `@group(2) @binding(0) var<storage, read> animalPartOf: array<u32>;
@group(2) @binding(1) var<storage, read> bakedPalette: array<vec4<f32>>;`,
    entries: [upload(partOf), palette].map((buffer, binding) => ({ binding, resource: { buffer } })),
    wgsl: `${constants}\n${BAKED_POSE_WGSL}\n${BAKED_PALETTE_READ_WGSL}
fn deform(pos: vec3<f32>, normal: vec3<f32>, vi: u32, inst: Instance, slot: u32) -> Deformed {
    let p = bakedPalettePose(0u, animalParts, slot, animalPartOf[vi]);
    return Deformed(bakedPosition(p, pos), normalize(bakedNormal(p, normal)));
}`,
  };
  return {
    hook,
    attach(batch: RenderBatch) {
      // Two-dimensional dispatch stays within the workgroup limit even at the largest count.
      const stage = engine.createComputeStage({
        name: 'Herd visible animation palette',
        dispatchCount: [Math.ceil(capacity / 64), model.parts, 1],
        shaderCode: `${INSTANCE_STRUCT_WGSL}
@group(0) @binding(0) var<storage, read> visible: array<Instance>;
@group(0) @binding(1) var<storage, read> drawArgs: array<u32>;
@group(0) @binding(2) var<storage, read> bakedTransforms: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> bakedPalette: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> animalAgents: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> animalPhase: array<f32>;
@group(0) @binding(6) var<storage, read> animalTopSpeed: array<f32>;
${constants}\n${BAKED_POSE_WGSL}\n${BAKED_SAMPLE_WGSL}\n${BAKED_PALETTE_WRITE_WGSL}
fn samplePose(clip: u32, phase: f32, part: u32) -> BakedPose {
    return bakedClipPose(0u, animalParts, animalClips[clip], phase, true, part);
}
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Flatten the bounded 2D dispatch so adjacent lanes write adjacent part matrices.
    // The old slot=gid.x/part=gid.y order scattered writes by animalParts * 48 bytes.
    let linear = gid.x + gid.y * ${Math.ceil(capacity / 64) * 64}u;
    let slot = linear / animalParts;
    if (slot >= drawArgs[1] || slot >= ${capacity}u) { return; }
    let row = visible[slot].sourceIndex;
    let record = row * 4u;
    let state = animalAgents[record + 2u];
    let speed = clamp(length(animalAgents[record + 1u].xz) / max(0.001, animalTopSpeed[0] * state.z), 0.0, 1.0);
    let phase = animalPhase[row];
    let part = linear % animalParts;
    var p: BakedPose;
    if (state.w > 0.0) { p = samplePose(3u, phase, part); }
    else if (speed < 0.15) {
        p = bakedMix(samplePose(0u, phase, part), samplePose(1u, phase, part), smoothstep(0.0, 0.15, speed));
    } else {
        p = bakedMix(samplePose(1u, phase, part), samplePose(2u, phase, part), smoothstep(0.15, 0.7, speed));
    }
    bakedWritePalette(0u, animalParts, slot, part, p);
}`,
      });
      engine.setComputePassBindGroup(stage, [batch.storageBuffer!, batch.indirectBuffer!, table, palette, agents, phase, topSpeed]
        .map((buffer, binding) => ({ binding, resource: { buffer } })), 0);
      return stage;
    },
  };
}

export const ANIMAL_TEXTURE = `
@group(1) @binding(0) var animalTexture: texture_2d<f32>;
@group(1) @binding(1) var animalSampler: sampler;`;

export const animalFragment = (highlight: boolean) => /* wgsl */ `
@fragment fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // The installed site palette differs from the animals' missing external palette. Their UV
    // columns encode four swatches; map them to white, pink, orange and black in the site palette.
    var swatch = vec2<f32>(0.03, 0.95);
    if (in.uv.x > 0.9) { swatch = vec2<f32>(0.97, 0.05); }
    else if (in.uv.x > 0.4) { swatch = vec2<f32>(0.7, 0.3); }
    else if (in.uv.x > 0.3) { swatch = vec2<f32>(0.22, 0.95); }
    let base = textureSample(animalTexture, animalSampler, swatch).rgb;
    let n = normalize(cross(dpdx(in.worldPos), dpdy(in.worldPos)));
    let sun = abs(dot(n, normalize(vec3<f32>(0.4, 1.0, 0.3))));
    let light = 0.42 + 0.28 * smoothstep(0.25, 0.3, sun) + 0.3 * smoothstep(0.65, 0.7, sun);
    let color = ${highlight ? 'mix(base, vec3<f32>(1.0, 0.75, 0.1), 0.65)' : 'base'};
    return vec4<f32>(color * light, 1.0);
}`;