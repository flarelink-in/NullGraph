# NullGraph

**Zero scene graph. Zero copy. Infinite scale.**
> A Data-Oriented WebGPU rendering framework for massive web worlds.

NullGraph is a brutalist, high-performance rendering library designed specifically for **Web Workers** and **Data-Oriented Design (DOD)**.

It completely abandons the traditional Object-Oriented *Scene Graph* (`Root -> Node -> Mesh -> Geometry`) in favor of mapping raw, contiguous `ArrayBuffers` directly to **WebGPU Storage Buffers**.

If you are building an MMO, a voxel engine, or a multiverse with tens of thousands of dynamic entities, NullGraph ensures your main thread stays at a flat **0ms overhead**.

---
![img.png](img.png)
## Why NullGraph?

Traditional WebGL frameworks (like Three.js or Babylon.js) are built for ease of use, heavily relying on:
- `new` keyword allocations
- Dynamic memory
- Garbage Collection (GC)

When building massive, chunk-streaming open worlds, this OOP overhead causes:
- Main-thread stuttering
- Shader compilation lag

### NullGraph solves this by doing less:

- **Zero Scene Graph**  
  No `.traverse()`, no `.updateMatrixWorld()`. The GPU reads your flat array.

- **Zero-Copy Streaming**  
  Calculate your ECS layout in a Web Worker, pass the `ArrayBuffer` to the main thread, and upload directly to VRAM.

- **No GC Spikes**  
  Memory is pre-allocated. No runtime object creation or destruction.

- **Predictable WebGPU Pipelines**  
  No mid-game shader compilation stutters.

---
## Prototype / Test Engine

You can explore a working prototype of NullGraph here:  
https://github.com/Vikas593-cloud/NullGraph-Test-Engine.git

This repository demonstrates the rendering pipeline, ECS buffer layout, and real-time WebGPU integration in action.


## Installation

> ⚠️ Currently in pre-release development

```bash
npm install null-graph gl-matrix
```
# NullGraph Engine

NullGraph is a high-performance, Data-Oriented WebGPU rendering engine designed for the Axion Engine.

It bypasses traditional object-oriented overhead by assuming a tight Entity Component System (ECS) architecture. Entity data is packed directly into flat `Float32Arrays` (or passed directly from a Rust/C++ WebAssembly module) and blasted straight to VRAM.

##  The Architecture Demo Suite

NullGraph now includes a built-in interactive dashboard to test and benchmark different memory layouts and compute paradigms in real-time.

**Play the Live Demo:**[null-graph.web.app](https://null-graph.web.app/)

Launch the UI to explore the following architectural patterns:

* **AoS (Array of Structs):** The standard DOD baseline. Data is packed as `[PosX, PosY, PosZ, ScaleX..., ColorR...]` per entity.
* **SoA (Struct of Arrays):** Cache-friendly contiguous memory. Arrays are separated by component type `[Pos1, Pos2...]`, `[Scale1, Scale2...]`, maximizing CPU cache hits.
* **AoSoA (Chunked SoA):** The AAA industry standard. Memory is chunked into blocks of 16 entities (64 bytes), perfectly aligning with CPU L1 cache lines and enabling SIMD auto-vectorization.
* **OOP Scene Graph:** A traditional hierarchical tree `Node -> Children`. Included specifically to benchmark and demonstrate the CPU bottleneck of pointer-chasing and recursive matrix math.
* **GPU Compute Animation:** Offloads simulation entirely to the GPU via Uniform buffers. The CPU handles zero per-frame math, achieving massive instance counts with almost no CPU time.

---

##  Quick Start (AoS Implementation)

Define your **Stride** (floats per entity) and **Offsets** (where position, scale, and color live). The WebGPU WGSL shader reads this storage buffer directly to instance your geometry.

```ts
import { NullGraph, Camera } from 'null-graph';

async function init() {
    const canvas = document.getElementById('gpuCanvas') as HTMLCanvasElement;

    // 1. Initialize the WebGPU Device
    const engine = new NullGraph();
    await engine.init(canvas);

    // 2. Setup Camera (Powered by gl-matrix)
    const camera = new Camera(75, canvas.width / canvas.height, 0.1, 1000.0);
    camera.updateView([0, 20, 80], [0, 0, 0]);
    engine.updateCamera(camera);

    // 3. Generate DOD ArrayBuffer
    const entityCount = 10000;
    const strideFloats = 14; 
    const ecsBuffer = new Float32Array(entityCount * strideFloats);

    // Populate buffer (In production, this comes from your ECS or WASM Worker)
    for (let i = 0; i < entityCount; i++) {
        const base = i * strideFloats;
        ecsBuffer[base + 1] = (Math.random() - 0.5) * 100; // X
        ecsBuffer[base + 2] = (Math.random() - 0.5) * 100; // Y
        ecsBuffer[base + 3] = (Math.random() - 0.5) * 100; // Z
        
        ecsBuffer[base + 8] = 1.0;  // Scale X
        ecsBuffer[base + 9] = 1.0;  // Scale Y
        ecsBuffer[base + 10] = 1.0; // Scale Z
    }

    // 4. Blast directly to VRAM
    engine.updateData(ecsBuffer, entityCount);

    // 5. Render Loop
    function frame() {
        engine.render();
        requestAnimationFrame(frame);
    }

    frame();
}

init();
```
### Roadmap
#### NullGraph is currently in active development. Upcoming features include:

-[ ] Depth / Z-Buffer Integration (Proper 3D occlusion)

-[ ] Custom WGSL Material Injection (Passing Material IDs via the ECS buffer)

-[ ] Raw GLTF Buffer Parsing (Extracting static vertex arrays from models)

-[ ] Directional Shadows
## License

NullGraph is released under the MIT License.
