// docs/quick-start.md ("Hello, Cube!") as a render scene — the first code a new user, or an
// agent, copies is checked by an actual render:
//
//   npm run render -- test/gpu/scenes/quick-start.mjs --out cube.png
//
// Kept line-for-line equivalent to the doc. If one changes, change the other.

export default async function scene({ engine, width, height, lib }) {
    const { Camera } = lib.core;
    const { Primitives, StandardLayout } = lib.geometry;
    const { buildPBRShader, StandardPBRMaterial } = lib.materials;

    const camera = new Camera(75, width / height, 0.1, 1000.0);
    const mainPass = engine.createPass({ name: 'Main', isMainScreenPass: true });

    const cubeGeom = Primitives.createCube(StandardLayout, 2, 2, 2);
    cubeGeom.upload(engine);

    const material = new StandardPBRMaterial(engine, {
        albedoMap: engine.textureManager.fallbackWhite,
        normalMap: engine.textureManager.fallbackNormal,
        packedMap: engine.textureManager.fallbackWhite,
        packedMapFormat: 'ARM',
        baseColor: [0.1, 0.5, 0.9, 1.0], // Blue
        metallicMultiplier: 0.2,
        roughnessMultiplier: 0.5,
    });

    const cubeBatch = engine.createBatch(mainPass, {
        shaderCode: buildPBRShader({ useSkinning: false }),
        strideFloats: 14,
        maxInstances: 1,
        vertexLayouts: cubeGeom.layout.getWebGPUDescriptor(),
        depthWriteEnabled: true,
    });

    material.applyToBatch(cubeBatch);
    engine.setBatchGeometry(cubeBatch, cubeGeom.vertexBuffer, cubeGeom.indexBuffer, cubeGeom.indices.length);

    const initialData = new Float32Array(14);
    initialData[1] = 0.0; initialData[2] = 0.0; initialData[3] = 0.0; // Position XYZ
    initialData[7] = 1.0; // Rotation W
    initialData[8] = 1.0; initialData[9] = 1.0; initialData[10] = 1.0; // Scale XYZ
    initialData[11] = 1.0; initialData[12] = 1.0; initialData[13] = 1.0; // Color RGB
    engine.updateBatchData(cubeBatch, initialData, 1);

    // The doc's requestAnimationFrame loop, as a per-frame function (seconds = frame / 60).
    return (frame, simTime) => {
        camera.updateView([Math.sin(simTime) * 8, 3, Math.cos(simTime) * 8], [0, 0, 0]);
        engine.updateCamera(camera);
        return [mainPass];
    };
}
