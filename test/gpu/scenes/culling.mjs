// Instance-tier GPU frustum culling. No meshlets or Hi-Z depth history required.
// Inspired by the frontend VisibilityMeshlets demo; the first step deliberately omits meshlets.
export default async function scene({ engine, width, height, lib }) {
    const { Camera } = lib.core;
    const { Primitives, StandardLayout, MegabufferBuilder } = lib.geometry;
    const { VisibilitySystem, InstanceBuffer } = lib.visibility;
    const { mat4 } = lib.glMatrix;
    const count = 10_000;
    const camera = new Camera(60, width / height, 0.1, 500);
    camera.updateView([0, 25, 70], [0, 0, 0]);
    engine.updateCamera(camera);
    const pass = engine.createPass({ name: 'Culling', isMainScreenPass: true });
    const geometry = Primitives.createCube(StandardLayout, 1, 1, 1);
    const mega = new MegabufferBuilder(StandardLayout.strideFloats);
    mega.addMesh('cube', geometry.vertices, geometry.indices);
    const vis = new VisibilitySystem(engine, { maxInstances: count });
    vis.setGeometry(mega);
    vis.createInstanceDrawBatch(pass, 'cube', 0, {
        vertexLayouts: StandardLayout.getWebGPUDescriptor(),
    });
    const instances = new InstanceBuffer(count);
    const model = mat4.create();
    const bounds = [0, 0, 0, Math.sqrt(3) / 2]; // Object-space sphere enclosing the unit cube.
    for (let i = 0; i < count; i++) {
        mat4.fromTranslation(model, [((i % 100) - 49.5) * 4, 0, (Math.floor(i / 100) - 49.5) * 4]);
        instances.write(i, model, bounds, 0);
    }
    vis.setInstances(instances.data, instances.count);
    vis.upload();
    const sequence = [...vis.getComputeStages(), pass];
    return () => sequence;
}