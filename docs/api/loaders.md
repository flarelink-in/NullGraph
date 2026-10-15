# Loaders API Usage

## First-Class GLTF / GLB Parsing
NullGraph includes a native `GLBParser` to ingest binary GLB assets. It unpacks vertex layouts,
indices, texture URLs, animation channels and skeletal data into engine-ready formats. It does
not currently preserve material colour factors or vertex colours.

```ts
import { GLBParser } from 'null-graph/loaders';

// Parse a GLB file directly into memory
const glbData = await GLBParser.load('./3dAssets/character.glb');

if (!glbData.meshes) throw new Error("No meshes found!");

// Easily map parsed mesh data to engine buffers
const vertexBuffer = engine.bufferManager.createVertexBuffer(glbData.meshes[0].vertices);
const indexBuffer = engine.bufferManager.createIndexBuffer(glbData.meshes[0].indices);
```

## Baked rigid animation (many instances, one shared table)

`bakeRigidAnimation` is for models animated by moving mesh nodes, not by skinning: a body,
head and legs as rigid parts. Use `GLBParser.load` or `parse`, **not `loadMerged`**: each mesh's
`node` identifies its entry in the parsed result's `nodes` hierarchy. Non-skinned vertices are
already transformed into model space at rest; those existing parser outputs are unchanged.

This complete preparation example is type-checked and run by `npm run test:loaders`:

```ts
import { GLBParser, bakeRigidAnimation } from 'null-graph/loaders';
import { MegabufferBuilder, StandardLayout } from 'null-graph/geometry';

export async function prepareRigidModel(url: string) {
    const glb = await GLBParser.load(url);
    const animation = bakeRigidAnimation(glb, {
        fps: 30,
        clips: ['idle', 'walk', 'run', { name: 'eat', loop: false }],
    });
    const mega = new MegabufferBuilder(StandardLayout.strideFloats);
    const mesh = mega.addMesh('animal', animation.vertices, animation.indices);
    const geometry = mega.build();

    // A vertex shader's vertex_index includes baseVertex. Pad to the whole mega vertex pool.
    const partOfVertex = new Uint32Array(geometry.megaVertices.length / StandardLayout.strideFloats);
    partOfVertex.set(animation.partOfVertex, mesh.baseVertex);
    const bounds = [0, 0, 0, animation.bounds.radius] as const;
    return { animation, mega, mesh, geometry, partOfVertex, bounds };
}
```

The result contains:
- `vertices`: rest-pose position, normal, uv (eight floats per vertex, `StandardLayout`);
  `indices`: `Uint32Array`; `vertexCount` and `indexCount`.
- `partOfVertex`: one `u32` per vertex; `parts`: the part count; `partNodes`: each part's node.
- `transforms`: frame-major, then part-major, **three rows of four floats** per transform.
  Part `p` in global frame `f` starts at `(f * parts + p) * 12`. Pose a position `v` with
  `dot(row0, vec4(v, 1))`, `dot(row1, vec4(v, 1))`, `dot(row2, vec4(v, 1))`.
  These are rest-to-pose transforms, not absolute node transforms.
- `clips`: `{ name, firstFrame, frames, seconds, loop }`; `frames`: total frames in the table.
- `bounds`: `min`, `max` and an origin-centred `radius` covering every sampled pose and convex
  blends of the table's poses. This does **not** bound the unsampled continuous glTF animation.

Options default to `fps: 30`, all clips in file order, looping, and `weld: true`. A looping clip
has `max(1, round(seconds * fps))` samples at `f * seconds / frames`; blend its last frame into
its first. A non-looping clip has `round(seconds * fps) + 1` samples including both endpoints
(one sample at zero if that count is one); clamp at the last frame. `firstFrame` is added after
selecting a frame within a clip. Linear blending of baked transforms approximates rotation
between samples: the test animals' run gait deviates by up to 37 mm at 30 fps. Raise `fps` for
closer motion, with a proportionally larger table.

Welding merges exact position, uv and part matches and averages/normalizes their normals; it
never joins different moving parts. Use `weld: false` to keep split face normals. A welded
flat-shaded model needs a fragment shader that reconstructs face normals from world-position
derivatives; otherwise it appears smooth. Under non-uniform scale, correct normal deformation
needs an inverse transpose: `GLBParser`'s existing rest-normal transform does not provide that.

The bake evaluates `LINEAR`, `STEP` and `CUBICSPLINE` channels and animated parents. It rejects
skinned meshes, missing mesh nodes or clips, empty clip selections, invalid fps, animated nodes
with a matrix, singular rest transforms and unknown interpolation modes. Morph-target weights
are not baked. The meshes become one material draw; separate material colours are not retained.

Upload `partOfVertex` and `animation.transforms` once as storage buffers shared by all instances.
Selecting clips/phases and deforming vertices is the application's shader's job; the bake alone
does not animate a draw. Use the instance-tier `vertexHook` in `null-graph/visibility` to apply
the table before the model transform (see the visibility guide). Skinned models use the
separate system below.

## Baked skinned animation (crowds)

`bakeSkinnedAnimation` is the four-influence companion to `bakeRigidAnimation`, exported from
`null-graph/loaders`. It evaluates the skeleton once at load time, not once per agent per frame.
The rigid bake and its one-influence format remain available unchanged.

```ts
import { GLBParser, bakeSkinnedAnimation } from 'null-graph/loaders';

export async function prepareSkinnedModel(url: string) {
    const glb = await GLBParser.load(url);
    try {
        const animation = bakeSkinnedAnimation(glb, { fps: 30 });
        return {
            animation,
            bounds: [0, 0, 0, animation.bounds.radius] as const,
        };
    } finally {
        // This example keeps geometry/animation only, not the source material textures.
        GLBParser.disposeImages(glb);
    }
}
```

Options and clip timing are the same as the rigid bake. The result has `vertices` (eight floats,
`StandardLayout`), `indices`, `vertexCount`, `indexCount`, `frames`, `clips` and `bounds`, plus:

- `influencesOfVertex`: four `u32` indices per vertex; unused slots have index and weight zero.
- `weightsOfVertex`: four `f32` weights per vertex, normalized to sum to one.
- `influences`: the table width; `influenceNodes`: the skin's joint nodes in their original order,
  followed by the nodes of rigid mesh attachments. An attachment has one influence of weight one.
- `transforms`: frame-major, influence-major, three rows of four floats. Influence `j` of frame `f`
  starts at `(f * influences + j) * 12`. Pose the vertex by summing the four transformed positions
  times their weights. Blend table rows linearly between frames, then blend clips as needed.

Skinning uses joint world transforms times inverse-bind matrices; the skinned mesh node's own
transform is ignored. To put bind-space geometry at the model's scale/orientation, the bake chooses
one fixed invertible transform `M` (the rest skinning matrix of the joint with the greatest total
weight), stores positions transformed by `M`, and stores `W(joint) * IBM(joint) * inverse(M)` in the
table. This preserves the exact weighted pose even if the file's default pose is not its bind pose;
in that case the stored vertices alone are **not** the default posed surface. Normals are transformed
by `M`'s inverse transpose and normalized; animated non-uniform scale still requires appropriate
normal deformation in the application. Bounds cover sampled weighted positions and convex blends,
not unsampled continuous motion. Welding requires exact position, UV, influence and weight matches.

The parser reads normalized `UNSIGNED_BYTE`/`UNSIGNED_SHORT` weights as floats and treats omitted
inverse-bind matrices as identity. The bake rejects multiple skins, additional influence sets
(`JOINTS_1`/`WEIGHTS_1`), zero-total, negative or non-finite weights, invalid joint indices, missing
nodes, malformed inverse binds, singular rebasing transforms, and the same invalid clip/timing
inputs as the rigid bake. Morph targets and separate material colours are not baked.

Upload influences, weights and the transform table once. When using a megabuffer, pad both per-vertex
arrays to its full vertex count and offset by `mesh.baseVertex * 4`, since `vertex_index` includes
the base vertex. This bake alone does not animate a draw. Use the palette snippets below to sample
once per **visible model**, after culling. The existing Herd & Wolves demo migration is phase 5.3;
the user's skinned animals follow when available.

## Baked pose / visible-model palette WGSL

Four independent strings are exported from `null-graph/loaders`:

| Export | Provides | Caller declares |
|---|---|---|
| `BAKED_POSE_WGSL` | `BakedPose { r0, r1, r2 }`, `bakedMix`, `bakedPosition`, `bakedNormal` | nothing |
| `BAKED_SAMPLE_WGSL` | `bakedFramePose`, `bakedClipPose` | storage-read `bakedTransforms: array<vec4<f32>>` |
| `BAKED_PALETTE_WRITE_WGSL` | `bakedWritePalette` | storage-read-write `bakedPalette: array<vec4<f32>>` |
| `BAKED_PALETTE_READ_WGSL` | `bakedPalettePose` (rigid), `bakedSkinPose` (four influences) | storage-read `bakedPalette: array<vec4<f32>>` |

Include the common pose string once in each shader module. The other strings require it and
declare no resources, entry points or bind groups. Apps own allocation, binding, clip selection,
dispatch, scheduling and teardown; no palette class or CPU work per visible agent is imposed.

Signatures (all functions are WGSL):

```wgsl
// tableBase and paletteBase are offsets in vec4 ROWS, not bytes or matrices.
// influences is the source table width; stride is the reserved influences per palette slot.
fn bakedFramePose(tableBase: u32, influences: u32, frame: u32, influence: u32) -> BakedPose
fn bakedClipPose(tableBase: u32, influences: u32, clip: vec2<u32>, phase: f32, looping: bool, influence: u32) -> BakedPose
fn bakedWritePalette(paletteBase: u32, stride: u32, slot: u32, influence: u32, pose: BakedPose)
fn bakedPalettePose(paletteBase: u32, stride: u32, slot: u32, influence: u32) -> BakedPose
fn bakedSkinPose(paletteBase: u32, stride: u32, slot: u32, joints: vec4<u32>, weights: vec4<f32>) -> BakedPose
fn bakedMix(a: BakedPose, b: BakedPose, weight: f32) -> BakedPose
fn bakedPosition(pose: BakedPose, position: vec3<f32>) -> vec3<f32>
fn bakedNormal(pose: BakedPose, normal: vec3<f32>) -> vec3<f32>
```

`clip` is `(firstFrame, frames)` from a baked clip. Phase is finite and measured in cycles, **not
seconds**: divide elapsed seconds by `clip.seconds` (use zero for a zero-duration clip). Looping
wraps negative/positive phases and interpolates across the seam. Non-looping clamps to `[0, 1]`
and includes the last frame. A one-frame clip holds its single pose. `bakedMix` does not clamp its
weight; keep blend weights in `[0, 1]` if relying on the bake's bounds. `bakedNormal` applies the
linear 3x3 part, without normalization or inverse transpose: normalize after transformation;
animated non-uniform scale needs application-specific normal handling or derivative face normals.

**Visible-slot scheduling with `VisibilitySystem`:** bind `batch.storageBuffer` as the compacted
`array<Instance>` and `batch.indirectBuffer` as storage-read `array<u32>`; `args[1]` is the count.
Dispatch enough threads for the batch capacity times its influence count; guard against that GPU
count before reading a survivor. Choose clips/phases from `visible[slot].sourceIndex`, sample each
influence and write its slot with `bakedWritePalette`. Put this stage **after that batch's cull and
before its draw**, e.g. `engine.render([...vis.getComputeStages(), paletteStage, scenePass])`.
Use separate palette regions/stages for batches (including separate camera/shadow culls).

In a vertex hook set **`instanceIndex: true`** and define the fifth argument `slot: u32`. Read the
palette by `slot`, never by `inst.sourceIndex`. For rigid vertices call
`bakedPalettePose(paletteBase, stride, slot, partOf[vi])`; for skinned vertices call
`bakedSkinPose(paletteBase, stride, slot, joints[vi], weights[vi])`, with indices and weights bound
as `array<vec4<u32>>` / `array<vec4<f32>>`. Return
`Deformed(bakedPosition(pose, pos), bakedNormal(pose, normal))`. Existing four-argument hooks remain
unchanged unless opted in. No vertex attributes, record layout or cull changes are needed.

**Memory and bounds are caller obligations:** reserve `capacity * stride * 48` bytes, plus any
base offset, with `STORAGE` usage (and `COPY_DST` if initializing). Capacity must cover the maximum
survivors, not just the previous frame's count; culling compacts the work but does not automatically
resize allocation. Check both `device.limits.maxStorageBufferBindingSize` and `maxBufferSize`, and
split batches/regions if required. Ensure all offsets/index products fit `u32`, `stride` covers all
influences, clips have at least one frame, and source/palette bounds are valid. Slots change every
cull and must be recomputed every frame; never reuse another view's slot assignment. Zero-weight
influences skip palette reads; weights should be non-negative and normalized. The snippets do not
validate malformed shader inputs or perform GPU allocation/dispatch for you.

**Executable examples:** `test/loaders-gpu/run.js` is a complete cull/palette/hook integration with
nonzero base offsets, clip blending, four-influence and rigid reads, shrinking/empty survivor lists,
and an independent CPU-rendered reference. Run `npm run test:gpu -- --only loaders`, or
`npm run test:gpu:browser -- --only loaders`. Build the library first.

## Hardware-Accelerated Skeletal Animations (one CPU-evaluated character)
NullGraph splits the heavy lifting: the CPU evaluates the animation timeline, and the GPU handles the vertex skinning. The `SkeletonManager` bridges the gap, allowing you to animate complex characters efficiently.

```ts
import { Animator, SkeletonManager } from 'null-graph/loaders';

// 1. Initialize animation systems with parsed GLB data
const skeletonManager = new SkeletonManager(engine.device, 70); // Max bones
const animator = new Animator(glbData.skin);
animator.play(glbData.animations[0]);

// 2. Bind the skeleton buffer to your mesh batch (Group 2)
engine.setRenderPassBindGroup(
    meshBatch,
    [{ binding: 0, resource: { buffer: skeletonManager.boneBuffer } }],
    2
);

// 3. Update loop
export function update(deltaTime: number) {
    // CPU computes the bone transforms ONCE
    animator.update(deltaTime);
    
    // GPU gets the updated bone matrices ONCE
    skeletonManager.updateFromAnimator(engine.device, animator);
}
```
