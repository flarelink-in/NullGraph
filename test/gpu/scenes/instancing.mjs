// Asset-free batched instancing: one mesh, one batch, 10,000 records.
// Distilled from the frontend's data-oriented instance setup; no React/worker dependencies.
export default async function scene({ engine, width, height, lib }) {
    const { Camera } = lib.core;
    const { Primitives, StandardLayout } = lib.geometry;
    const { buildPBRShader, StandardPBRMaterial } = lib.materials;
    const count = 10_000;
    const camera = new Camera(60, width / height, 0.1, 500);
    camera.updateView([0, 90, 100], [0, 0, 0]);
    engine.updateCamera(camera);
    const pass = engine.createPass({ name: 'Instancing', isMainScreenPass: true });
    const geometry = Primitives.createCube(StandardLayout, 0.6, 0.6, 0.6);
    geometry.upload(engine);
    const material = new StandardPBRMaterial(engine, {
        albedoMap: engine.textureManager.fallbackWhite,
        normalMap: engine.textureManager.fallbackNormal,
        packedMap: engine.textureManager.fallbackWhite,
        packedMapFormat: 'ARM',
        baseColor: [0.1, 0.5, 0.9, 1],
        metallicMultiplier: 0,
        roughnessMultiplier: 0.8,
    });
    const batch = engine.createBatch(pass, {
        shaderCode: buildPBRShader({ useSkinning: false }),
        strideFloats: 14,
        maxInstances: count,
        vertexLayouts: geometry.layout.getWebGPUDescriptor(),
        depthWriteEnabled: true,
    });
    material.applyToBatch(batch);
    engine.setBatchGeometry(batch, geometry.vertexBuffer, geometry.indexBuffer, geometry.indices.length);
    const data = new Float32Array(count * 14);
    for (let i = 0; i < count; i++) {
        const offset = i * 14;
        data[offset + 1] = (i % 100) - 49.5;
        data[offset + 3] = Math.floor(i / 100) - 49.5;
        data[offset + 7] = 1; // Quaternion W.
        data[offset + 8] = data[offset + 9] = data[offset + 10] = 1;
        data[offset + 11] = data[offset + 12] = data[offset + 13] = 1;
    }
    engine.updateBatchData(batch, data, count);
    const sequence = [pass];
    return () => sequence;
}