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

---

##  The Architecture Demo Suite

**Play the Live Demo:** [null-graph.web.app](https://null-graph.web.app/)

NullGraph includes an interactive dashboard to test and benchmark different memory layouts and compute paradigms in real-time. Explore the following patterns:

### Memory Layout Benchmarks
* **AoS (Array of Structs):** The standard DOD baseline.
* **SoA (Struct of Arrays):** Cache-friendly contiguous memory maximizing CPU cache hits.
* **AoSoA (Chunked SoA):** The AAA industry standard aligning with CPU L1 cache lines for SIMD auto-vectorization.
* **OOP Scene Graph:** A purposeful stress-test demonstrating the CPU bottleneck of traditional pointer-chasing and recursive math.

### Advanced Rendering Demos
* **3D Lighting & Depth:** Demonstrates Z-Buffer occlusion and Dot-Product normal lighting entirely within the shader.
* **Space Fleet (Multi-Batch):** Showcases the Render Queue architecture, Object Pooling, and rendering distinct geometries (Ships, Asteroids, Lasers) in a single pass.
* **Voxel Fireworks (DOD Physics):** Simulates 15,000+ physics-driven particles by separating CPU-side physics state arrays from GPU-side render buffers.

---

##  Quick Start (Batch Rendering)

NullGraph uses a **Render Batch** architecture. You define the 3D geometry once, set up your WGSL shader pipeline, and then blast your ECS array to the GPU every frame.

```ts
import { NullGraph, Camera } from 'null-graph';

async function init() {
    const canvas = document.getElementById('gpuCanvas') as HTMLCanvasElement;

    // 1. Initialize the WebGPU Device
    const engine = new NullGraph();
    await engine.init(canvas);

    const camera = new Camera(75, canvas.width / canvas.height, 0.1, 1000.0);
    camera.updateView([0, 20, 80], [0, 0, 0]);
    engine.updateCamera(camera);

    // 2. Upload Static Geometry via BufferManager
    const vertices = new Float32Array([...]); // [X,Y,Z, NormalX,NormalY,NormalZ...]
    const indices = new Uint16Array([...]);
    
    const vbo = engine.bufferManager.createVertexBuffer(vertices);
    const ibo = engine.bufferManager.createIndexBuffer(indices);

    // 3. Create a Render Batch
    const myBatch = engine.createBatch({
        shaderCode: `/* Your WGSL Code Here */`,
        strideFloats: 14,
        maxInstances: 10000,
        vertexLayouts: [{
            arrayStride: 24, // 6 floats * 4 bytes
            attributes: [
                { shaderLocation: 0, offset: 0, format: 'float32x3' },  // Position
                { shaderLocation: 1, offset: 12, format: 'float32x3' }  // Normal
            ]
        }]
    });
    
    // Attach geometry to the batch
    engine.setBatchGeometry(myBatch, vbo, ibo, indices.length);

    // 4. Generate DOD ArrayBuffer (From your ECS)
    const MAX_INSTANCES = 10000;
    const ecsBuffer = new Float32Array(MAX_INSTANCES * 14);
    // ... Populate buffer with [PosX, PosY, PosZ, ...Scale, ...Color] ...

    // 5. Render Loop
    function frame() {
        // Update the specific batch with new data
        engine.updateBatchData(myBatch, ecsBuffer, MAX_INSTANCES);
        
        // Let NullGraph iterate the render queue
        engine.render(); 
        requestAnimationFrame(frame);
    }

    frame();
}

init();
```
## Roadmap
NullGraph is in active development for the Axion Engine.

-[x] Multi-Object Render Queue / Batching

-[x] Depth / Z-Buffer Integration (Proper 3D occlusion)

-[x] VBO/IBO Geometry Buffer Manager

-[ ] PBR Textures & Material ID Injection
-[ ] Raw GLTF Mesh Parsing

-[ ] Directional Shadows / Cascaded Shadow Maps

-[ ] GPU Compute Frustum Culling

### License
NullGraph is released under the MIT License.