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
does not animate a draw. The visibility vertex hook is the next phase of the animated-agents
design, not part of this preparation API. Skinned models use the separate system below.

## Hardware-Accelerated Skeletal Animations
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
