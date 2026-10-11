# NullGraph agent guide

Target manifest version: **1.1.1**. This is an application guide, not an engine design spec.
Start small; fetch only the API pages needed by the task. The 50k-token budget is a ceiling,
not a reason to fill context. The initial quick path does not require the full documentation tree.

## 1. Quick path and discovery

NullGraph is a direct-to-GPU renderer with no retained scene graph. Build geometry, batches,
materials and a camera, then submit an ordered sequence of compute stages and render passes.
Do not invent `Scene`, parenting, automatic lights, or a model-to-scene loader.

There are three access modes:

1. **Repository checkout:** read the root [index](../../llms.txt), then this guide and
   [validation](validation.md). Examples and the runner are local.
2. **npm plus network:** the installed README points at the repository. Fetch the index as
   text, resolve its relative links against that document's URL, and fetch only needed pages.
   Downloading the guide does not download its examples or the runner. A checkout is the
   simplest way to obtain runnable validation tools.
3. **npm offline:** inspect README, package exports and declarations. The repository kit
   is not installed. You may build an application using public APIs, but must not claim
   that `npm run render` exists in the application's package.

The intended release anchor is **v1.1.1**. A manifest bump is not a published tag or npm release.
Until the tag is published, use a provided checkout or a known commit. For released packages,
match the kit's tag to the installed version; do not silently combine old packages with branch-head docs.

### Repository setup

Run from the library checkout, not the Vite frontend:

```sh
npm install
npm run build:lib
npm run test:agent-kit
npm run render -- test/gpu/scenes/quick-start.mjs --out cube.png --json cube.json --frames 2
```

Open the PNG and read the JSON. Exit zero means no recorded GPU errors and a non-uniform frame;
it does not prove the scene satisfies the user's request. See [validation](validation.md).

### The runner contract

Copy a complete starter to your scene file. Its default async function receives `engine`,
`width`, `height`, and `lib`, and returns a frame callback. `engine` is already initialized.
`lib` supplies core, geometry, materials, lights, cameras, visibility, shadows, environment,
post, loaders, picking, text and glMatrix. This injection is a **runner contract**, not a
global in ordinary browser projects. In an application, import public package subpaths instead.

The callback receives frame number and deterministic seconds (frame / 60). Return a render
sequence containing the main-screen pass; the runner submits it. Do not submit that same
sequence yourself as well. The starter scenes preallocate their sequence and static data.

## 2. Instancing: learn the plain batch first

[Executable source](../../test/gpu/scenes/instancing.mjs). A 100 by 100 grid, one cube mesh,
one material, one batch. This example performs no GPU visibility culling. Its PBR shader uses
the 14-float instance layout; this is not the 24-float VisibilitySystem record layout.
Keep scale and quaternion initialized. Static data needs only one upload.

<!-- scene: test/gpu/scenes/instancing.mjs -->
```js
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
```

## 3. Culling: add the instance tier, not every subsystem

[Executable source](../../test/gpu/scenes/culling.mjs). A wider deterministic grid lets the
frustum reject off-screen instances. Inspired by the frontend VisibilityMeshlets demo's
geometry/instance/upload flow, without its React worker bridge, post chain, assets or meshlets.
The instance tier needs a MegabufferBuilder but no MeshletBuilder. Bounding spheres are local
to the mesh; the GPU transforms them. Never substitute the plain batch's 14-float records.

<!-- scene: test/gpu/scenes/culling.mjs -->
```js
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
```

Read [Visibility](../api/visibility.md) for material integration, meshlets and Hi-Z. Hi-Z is
not demonstrated here: it needs a samplable depth source and previous-frame depth history.
The existing visibility GPU suite separately verifies frustum counts and Hi-Z controls.

## 4. Building a browser project from a starter

These headless scenes are not Vite entry points. In a browser app:

- Use the confirmed bundler/framework already in the project; do not add React or workers
  merely because the production demo site uses them.
- Import `NullGraph` and `Camera` from `null-graph`, and features from their public subpaths.
- Create or acquire the canvas, await initialization and check its success result before setup.
- Reproduce the starter's setup with the initialized engine; pass real canvas dimensions.
- Drive rendering with one animation loop, update the camera and resize dependent targets.
- Stop the loop before destroying its resources. Treat initialization failure and device loss
  as errors to expose, not a black frame to ignore.

The production frontend is a reference, not a dependency. Its worker hooks (`update`,
`afterRender`, camera hooks) belong to that application, not NullGraph's core public API.
Assets such as `/textures/colormap.png` do not arrive with npm install.

## 5. Ordering, layouts and lifetime

- Submit producers before consumers: lighting binning and visibility stages before geometry;
  shadow/environment passes before draws that sample them; post-processing after its inputs.
- Use explicit sequences. Do not assume creating a pass places it at the correct point in
  a subsystem's sequence. Rebuild a cached sequence when its topology actually changes.
- Preallocate application frame-path arrays and records where practical. Allocation-free
  starters do not prove every engine method or application path is allocation-free.
- Choose either plain batches or visibility records deliberately. Match the shader, vertex
  layouts, bind groups, target formats and instance stride; consult declarations when uncertain.
- Resource ownership is explicit. A shared VisibilitySystem can outlive individual draw batches;
  use its public removal API when removing their owners. Stop rendering before teardown yields.
- CPU readback is for tests/tooling, not an interactive frame loop.

### Environment and GI sequencing

Read [Environment](../api/environment.md) and [Probes](../api/probes.md) before adding GI.
Initialize camera, lights, geometry, environment and shadow prerequisites. Submit their initial
bakes/shadow atlas before startup `volume.prewarm()`. Prewarm submits its own frames; do not
launch it concurrently from an interactive loop. For UI full rebakes, invalidate the complete
volume bounds and let normal scheduling refresh progressively. Use public `volume.setIntensity()`.
Copy environment SH after bake submission so subsequent work consumes updated SH. Initial filling
is not proof of multi-bounce convergence. These are integration rules, not a new GI starter here.

## 6. Evidence and boundaries

Never label a compile-only check a rendering test. Never label a non-blank screenshot a proof
of correct culling. Report commands, exit status, GPU availability and what was actually checked.
The two new scenes are render smoke tests; [validation](validation.md) points to dedicated
draw-count/culling assertions. Fresh-agent evaluation remains separate from engine tests.

For unsupported or untested hosts, follow the open browser/GPU pointers in validation. Do not
invent platform support, automatic fallbacks, or a cross-platform guarantee.