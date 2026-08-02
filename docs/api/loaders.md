# Loaders API Usage

## First-Class GLTF / GLB Parsing
NullGraph includes a native `GLBParser` to effortlessly ingest 3D assets. It automatically unpacks vertex layouts, indices, materials, and skeletal data into engine-ready formats.

```ts
import { GLBParser } from 'null-graph/loaders';

// Parse a GLB file directly into memory
const glbData = await GLBParser.load('./3dAssets/character.glb');

if (!glbData.meshes) throw new Error("No meshes found!");

// Easily map parsed mesh data to engine buffers
const vertexBuffer = engine.bufferManager.createVertexBuffer(glbData.meshes[0].vertices);
const indexBuffer = engine.bufferManager.createIndexBuffer(glbData.meshes[0].indices);
```

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
