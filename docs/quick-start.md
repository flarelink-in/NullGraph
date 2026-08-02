# Quick Start (Hello, Cube!)

NullGraph uses a heavily optimized Render Batch architecture. To get your first object on screen, we'll create a default pass, generate a primitive cube, apply a PBR material using the engine's fallback textures, and blast it to the GPU.

```ts
import { NullGraph, Camera } from 'null-graph';
import { Primitives, StandardLayout } from 'null-graph/geometry';
import { buildPBRShader, StandardPBRMaterial } from 'null-graph/materials';

async function main() {
    try {
        const canvas = document.getElementById('gpuCanvas');

        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        console.log("1. Initializing NullGraph...");
        const engine = new NullGraph();
        await engine.init(canvas);

        console.log("2. Setting up Camera...");
        const camera = new Camera(75, canvas.width / canvas.height, 0.1, 1000.0);

        console.log("3. Creating Pass and Geometry...");
        const mainPass = engine.createPass({ name: 'Main', isMainScreenPass: true });

        const cubeGeom = Primitives.createCube(StandardLayout, 2, 2, 2);
        cubeGeom.upload(engine);

        console.log("4. Creating Material and Batch...");
        const material = new StandardPBRMaterial(engine, {
            albedoMap: engine.textureManager.fallbackWhite,
            normalMap: engine.textureManager.fallbackNormal,
            packedMap: engine.textureManager.fallbackWhite,
            packedMapFormat: "ARM",
            baseColor: [0.1, 0.5, 0.9, 1.0], // Blue
            metallicMultiplier: 0.2,
            roughnessMultiplier: 0.5
        });

        const cubeBatch = engine.createBatch(mainPass, {
            shaderCode: buildPBRShader({ useSkinning: false }),
            strideFloats: 14,
            maxInstances: 1,
            vertexLayouts: cubeGeom.layout.getWebGPUDescriptor(),
            depthWriteEnabled: true
        });

        material.applyToBatch(cubeBatch);
        engine.setBatchGeometry(cubeBatch, cubeGeom.vertexBuffer, cubeGeom.indexBuffer, cubeGeom.indices.length);

        console.log("5. Setting Instance Data...");
        const initialData = new Float32Array(14);
        initialData[1] = 0.0; initialData[2] = 0.0; initialData[3] = 0.0; // Position XYZ
        initialData[7] = 1.0; // Rotation W
        initialData[8] = 1.0; initialData[9] = 1.0; initialData[10] = 1.0; // Scale XYZ
        initialData[11] = 1.0; initialData[12] = 1.0; initialData[13] = 1.0; // Color RGB

        engine.updateBatchData(cubeBatch, initialData, 1);

        console.log("6. Starting Render Loop...");
        function frame() {
            const simTime = performance.now() * 0.001;

            // Orbit the camera around the cube
            camera.updateView(
                [Math.sin(simTime) * 8, 3, Math.cos(simTime) * 8], // Eye
                [0, 0, 0] // Look Target
            );
            engine.updateCamera(camera);

            engine.render();
            requestAnimationFrame(frame);
        }

        frame();

    } catch (err) {
        console.error("CRITICAL ENGINE ERROR:", err);
    }
}

main();
```
