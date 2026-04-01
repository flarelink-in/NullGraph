# NullGraph Engine

**Zero scene graph. Zero copy. Infinite scale.**
> A Data-Oriented WebGPU rendering framework for massive web worlds.

NullGraph is a brutalist, high-performance rendering library designed specifically for **Web Workers** and **Data-Oriented Design (DOD)**.

It completely abandons the traditional Object-Oriented *Scene Graph* (`Root -> Node -> Mesh -> Geometry`) in favor of mapping raw, contiguous `ArrayBuffers` directly to **WebGPU Storage Buffers**.

If you are building an MMO, a voxel engine, or a multiverse with tens of thousands of dynamic entities, NullGraph ensures your main thread stays at a flat **0ms overhead**.

---

## ⚡ Why NullGraph?

Traditional WebGL frameworks (like Three.js or Babylon.js) are built for ease of use, heavily relying on the `new` keyword, dynamic memory, and Garbage Collection. When scaling up to massive open worlds, this OOP overhead causes main-thread stuttering and shader compilation lag.

**NullGraph solves this by doing less:**
- **Zero Scene Graph:** No `.traverse()`, no `.updateMatrixWorld()`. The GPU reads your flat array directly.
- **Zero-Copy Streaming:** Calculate your ECS layout in a Web Worker, pass the `Float32Array` to the main thread, and blast it straight to VRAM.
- **Render Queues (Batches):** Render thousands of unique object types simultaneously with minimal GPU state changes.
- **No GC Spikes:** Memory is pre-allocated. No runtime object creation or destruction.
- **Compute-Driven Indirect Drawing:** Offload culling entirely to the GPU. NullGraph supports WebGPU Compute Shaders that dynamically build IndirectDrawArgs, resulting in zero CPU overhead for visibility checks.
- **Multi-Pass Architecture:** Seamlessly chain offscreen render passes into screen-space post-processing pipelines (Bloom, CRT, HUD effects) by attaching textures directly to subsequent batches.

---
## Installation & Setup

NullGraph is distributed as a modular ESM package. To maintain its "Zero-Copy" philosophy, it requires `gl-matrix` as a peer dependency to ensure your application and the engine share the same math structures.

### 1. Install via NPM

```bash
# Install the core engine
npm install null-graph

# Install required peer dependencies
npm install gl-matrix

# Recommended: Install WebGPU types for IDE autocomplete
npm install @webgpu/types --save-dev
```
### 2. Module Architecture

NullGraph uses Subpath Exports to keep your production bundles lean. You only pay for the features you import.

* `null-graph`
The Core Engine. Handles WebGPU device initialization, Pass management, and Buffer streaming.
* `null-graph/geometry`
The Math & Primitive Toolbox. Contains dynamic generators for Cubes, Spheres, and custom Vertex Layouts.
---
##  The Architecture Demo Suite

**Play the Live Demo:** [null-graph.web.app](https://null-graph.web.app/)

**Github Source code:** [NullGraph-Test-Engine.git](https://github.com/Vikas593-cloud/NullGraph-Test-Engine.git)

NullGraph includes an interactive dashboard to test and benchmark different memory layouts and compute paradigms in real-time. Explore the following patterns:

### Memory Layout Benchmarks
* **AoS (Array of Structs):** The standard DOD baseline.
* **SoA (Struct of Arrays):** Cache-friendly contiguous memory maximizing CPU cache hits.
* **AoSoA (Chunked SoA):** The AAA industry standard aligning with CPU L1 cache lines for SIMD auto-vectorization.
* **OOP Scene Graph:** A purposeful stress-test demonstrating the CPU bottleneck of traditional pointer-chasing and recursive math.

### Advanced Rendering Demos
* **3D Lighting & Depth:** Demonstrates Z-Buffer occlusion and Dot-Product normal lighting entirely within the shader.
* **Space Fleet (Compute & Multi-Pass):** Showcases the engine's advanced pipelines. Thousands of asteroids and ships are culled via Compute Shaders using `Indirect Drawing`, rendered to an offscreen buffer, and then fed through a tactical HUD `Post-Processing Pass` (chromatic aberration, scanlines, and noise)
* **Voxel Fireworks (DOD Physics):** Simulates 15,000+ physics-driven particles by separating CPU-side physics state arrays from GPU-side render buffers.

---
#  Advanced Capabilities

## Multi-Pass Rendering & Post-Processing

NullGraph allows you to isolate rendering logic into distinct passes. You can render pristine 3D scenes offscreen and pipe them into post-processing passes.

```ts
// 1. Create an offscreen pass
const scenePass = engine.createPass({
    name: 'Offscreen Pass',
    isMainScreenPass: false,
    colorAttachments: [{
        view: offscreenTexture.createView(),
        clearValue: { r: 0.0, g: 0.01, b: 0.03, a: 1.0 },
        loadOp: 'clear', storeOp: 'store'
    }],
    // ... depth attachments
});

// 2. Create the final Post-Processing pass
const hudPass = engine.createPass({
    name: 'HUD Post Process',
    isMainScreenPass: true
});

// 3. Bind the offscreen texture to your post-process batch
const hudBatch = engine.createBatch(hudPass, { /* shader args */ });
engine.attachTextureMaterial(hudBatch, offscreenTexture.createView(), sampler);
```

## Indirect Drawing (GPU Compute Culling)

Stop relying on the CPU to figure out what to render. NullGraph batches can bind Compute Shaders to evaluate thousands of instances, write to an `IndirectDrawArgs buffer`, and command the vertex shader without the CPU ever knowing what happened.
```ts 
const batch = engine.createBatch(scenePass, {
isIndirect: true, // Tell NullGraph to use drawIndirect
computeShaderCode: `
        // Compute shader evaluates instance limits and writes to drawArgs
        let writeIdx = atomicAdd(&drawArgs.instanceCount, 1u);
        // ... cull and pack data
    `,
shaderCode: renderShaderCode,
strideFloats: 14,
maxInstances: 5000,
vertexLayouts: geometry.layout.getWebGPUDescriptor()
});
```
## Quick Start (Basic Batch Rendering)

NullGraph uses a Render Batch architecture. Define the 3D geometry once, set up your WGSL shader pipeline, and blast your ECS array to the GPU every frame.
```ts 
import { NullGraph, Camera } from 'null-graph';

async function init() {
const canvas = document.getElementById('gpuCanvas') as HTMLCanvasElement;
const engine = new NullGraph();
await engine.init(canvas);

    // 1. Upload Static Geometry via BufferManager
    const vbo = engine.bufferManager.createVertexBuffer(vertices);
    const ibo = engine.bufferManager.createIndexBuffer(indices);

    // 2. Create a Render Batch (Default Pass)
    const myBatch = engine.createBatch({
        shaderCode: `/* Your WGSL Code Here */`,
        strideFloats: 14,
        maxInstances: 10000,
        vertexLayouts: [ /* WebGPU layout descriptor */ ]
    });
    
    engine.setBatchGeometry(myBatch, vbo, ibo, indices.length);

    // 3. Generate DOD ArrayBuffer (From your ECS)
    const MAX_INSTANCES = 10000;
    const ecsBuffer = new Float32Array(MAX_INSTANCES * 14);

    // 4. Render Loop
    function frame() {
        engine.updateBatchData(myBatch, ecsBuffer, MAX_INSTANCES);
        engine.render(); 
        requestAnimationFrame(frame);
    }

    frame();
}
```
#### Notes:
* You have to manually define vertex buffers ,index buffers & indices buffers
* Refer to test-engine src/data/[Direct Git Link](https://github.com/Vikas593-cloud/NullGraph-Test-Engine/blob/main/src/data.ts) file for refrence of the above steps
## Option 2: Using Null-Graph Geometry (Recommended)

Instead of manually defining buffers, you can use the built-in Geometry Builder utilities for faster prototyping and cleaner code.

```ts
import { NullGraph, Camera } from 'null-graph';
import { Primitives, StandardLayout } from "null-graph/geometry";

async function init() {
    const canvas = document.getElementById('gpuCanvas') as HTMLCanvasElement;
    const engine = new NullGraph();
    await engine.init(canvas);

    // 1. Create Geometry using Primitives
    const cubeGeom = Primitives.createCube(StandardLayout, 1.0, 1.0, 1.0);

    // Uploads geometry to the engine's BufferManager
    cubeGeom.upload(engine);

    // 2. Create a Render Batch
    const myBatch = engine.createBatch({
        shaderCode: `/* Your WGSL Code Here */`,
        strideFloats: 14,
        maxInstances: 10000,
        vertexLayouts: cubeGeom.layout.getWebGPUDescriptor()
    });
    
    engine.setBatchGeometry(
        myBatch,
        cubeGeom.vertexBuffer!,
        cubeGeom.indexBuffer!,
        cubeGeom.indices.length
    );

    // 3. Generate ECS Data Buffer
    const MAX_INSTANCES = 10000;
    const ecsBuffer = new Float32Array(MAX_INSTANCES * 14);

    // 4. Render Loop
    function frame() {
        engine.updateBatchData(myBatch, ecsBuffer, MAX_INSTANCES);
        engine.render(); 
        requestAnimationFrame(frame);
    }

    frame();
}
```
## Roadmap
NullGraph is in active development for the Axion Engine.

-[x] Multi-Object Render Queue / Batching
-[x] Depth / Z-Buffer Integration (Proper 3D occlusion)
-[x] VBO/IBO Geometry Buffer Manager
-[x] Multi-Pass Rendering & Texture Attachments
-[x] GPU Compute Frustum Culling & Indirect Drawing
-[x] Geometry Builder And Intiation of `extras` library to improve DX experience
-[ ] PBR Textures & Material ID Injection
-[ ] Raw GLTF Mesh Parsing
-[ ] Directional Shadows / Cascaded Shadow Maps

### License
NullGraph is released under the MIT License.