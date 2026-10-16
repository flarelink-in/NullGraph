// D2.1 (docs/offline-films-design.md §10): the demo's Models look over a running Herd & Wolves simulation,
// composed from the copies as the demo's build() composes it (site index.ts:564–786): the post chain, the
// toon field, cows and foxes through the palette or the hook with far cubes by LOD, and the catch
// flashes drawn over the post chain's result. Without picking, labels, the overlay or the guards.
// D2.0's spike composed the same from the site's own files (test/spikes/films/look.mjs).
import { StandardLayout, MegabufferBuilder } from '../../../../dist/geometry.js';
import { VisibilitySystem } from '../../../../dist/visibility.js';
import { LambertMaterial } from '../../../../dist/materials.js';
import { PostChain, ChainPresets } from '../../../../dist/post.js';

const SPECIES = ['prey', 'wolf'];

/**
 * The look over `sim` (a scenario build). Adds the animal-phase stage and the instance stage to the
 * simulation: build it before taking the simulation's step chain. `copies` holds the copied modules
 * (models, field, flashes). Destroy the simulation first: its instance stage binds the look's rows.
 */
export function buildLook(engine, sim, copies, assets, { width, height, look = 'anime', scale = 1, flashes = true }) {
    const { device } = engine;
    const { agents } = sim, count = agents.capacity, half = sim.definition.world.maxX;
    const { animalHook, animalPalette, animalPaletteFits, animalFragment, animalCube, ANIMAL_PHASE, ANIMAL_TEXTURE,
        ANIMAL_LOD_PX, ANIMAL_CUBE_RGB } = copies.models;
    const owned = [];
    const own = thing => { owned.push(thing); return thing; };
    const buffer = (label, size, usage) => own(device.createBuffer({ label: `film Herd ${label}`, size, usage }));

    const post = own(new PostChain(engine, { gbuffer: { depth: 'depth24plus' }, scale }));
    post.setChain(look === 'anime' ? ChainPresets.anime() : ChainPresets.faithful());
    post.resize(width, height);
    const scene = engine.createPass({ name: 'film Herd toon field', isMainScreenPass: false,
        ...post.sceneAttachments({ clearValue: { r: 0.12, g: 0.2, b: 0.09, a: 1 } }) });
    const field = own(copies.field.createHerdField(engine, scene, half, assets.grass, post.format));

    // The animals: mesh 0 the cow, 1 the fox (the demo's DART_MESH order without its picking variants).
    const vis = new VisibilitySystem(engine, { maxInstances: count });
    const mega = new MegabufferBuilder(StandardLayout.strideFloats);
    const parts = [];
    SPECIES.forEach(shape => {
        mega.addMesh(shape, assets[shape].vertices, assets[shape].indices);
        parts.push(...assets[shape].partOfVertex);
    });
    // After every hooked mesh: the hook reads parts by absolute vertex index, and cubes have none.
    SPECIES.forEach(shape => {
        const cube = animalCube(assets[shape]);
        mega.addMesh(`${shape}-far`, cube.vertices, cube.indices);
    });
    vis.setGeometry(mega);
    SPECIES.forEach((shape, mesh) => agents.setSpecies(shape, { mesh, hoverMesh: mesh, selectedMesh: mesh, boundsRadius: assets[shape].radius }));

    const phase = buffer('animal phases', count * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    agents.addStepStage({ name: 'animal-phase', wgsl: ANIMAL_PHASE, buffers: [phase] });
    const usePalette = animalPaletteFits(engine, assets, count, 1);
    const vertexLayouts = StandardLayout.getWebGPUDescriptor();
    const hdr = 'rgba16float';
    const palettes = [], batches = [];
    SPECIES.forEach((shape, mesh) => {
        const top = buffer(`${shape} top speed`, 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        device.queue.writeBuffer(top, 0, new Float32Array([sim.definition.species[mesh].maxSpeed, 0, 0, 0]));
        const palette = usePalette ? animalPalette(engine, assets[shape], new Uint32Array(parts), agents.buffer, phase, top, count, own) : null;
        const hook = palette ? palette.hook : animalHook(engine, assets[shape], new Uint32Array(parts), agents.buffer, phase, top, own);
        const batch = vis.createInstanceDrawBatch(scene, shape, mesh, {
            targetFormat: hdr, vertexLayouts, fragmentBody: animalFragment(false), lod: { minPixels: ANIMAL_LOD_PX },
            vertexHook: hook, extraBindings: ANIMAL_TEXTURE,
            materialEntries: [{ binding: 0, resource: assets[shape].texture }, { binding: 1, resource: engine.textureManager.defaultSampler }],
        });
        batches.push(batch);
        if (palette) palettes.push(palette.attach(batch));
        const cube = own(new LambertMaterial(engine, { baseColor: [...ANIMAL_CUBE_RGB[shape], 1], ambient: 0.35 }));
        batches.push(vis.createInstanceDrawBatch(scene, `${shape}-far`, mesh, {
            targetFormat: hdr, vertexLayouts, lod: { maxPixels: ANIMAL_LOD_PX },
            material: { kind: 'lambert', instance: cube, useHalfLambert: true },
        }));
    });
    vis.setInstanceCount(count);
    agents.toInstances(vis);

    // The catch flashes, over the post chain's result, as the demo's shared pass draws them in Models mode.
    let overlay = null, rings = null;
    if (flashes) {
        overlay = engine.createPass({ name: 'film Herd flashes', isMainScreenPass: false,
            colorAttachments: [{ view: engine.ctx.getScreenView(), loadOp: 'load', storeOp: 'store' }],
            depthStencilAttachment: { view: engine.ctx.depthTexture.createView(), depthLoadOp: 'clear', depthClearValue: 1, depthStoreOp: 'store' } });
        rings = own(copies.flashes.catchFlashes(engine, overlay, sim.ring, agents.state, sim.definition.dt));
        rings.resize(width, height);
    }
    return {
        phase,
        usePalette,
        /** The frame's render sequence after its steps: culls, palettes, the scene, the post chain, the flashes. */
        sequence() {
            post.upload();
            field.upload();
            vis.upload();
            return [...field.getComputeStages(), ...vis.getComputeStages(), ...palettes, scene, ...post.getPassSequence(), ...(overlay ? [overlay] : [])];
        },
        destroy() {
            for (const b of batches) vis.removeDrawBatch(b);
            vis.destroy();
            for (const thing of owned.reverse()) thing.destroy();
        },
    };
}
