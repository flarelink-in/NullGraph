// D2.0 (docs/offline-films-design.md §10, §17): the Herd & Wolves demo's Models look, composed from the
// site's own modules over a running simulation, as the demo's build() composes it (site index.ts:564–786),
// without picking, labels, the overlay or the guards. The assets are read from the site's public/
// folder: loadHerdModels' bake is repeated here because it fetches root-relative URLs, and
// TextureManager.load needs createImageBitmap (browser only).
//
// The flashes are left out: the scenario does not hand out its catch ring (D2.1 adds that).
import { Camera } from '../../../dist/index.js';
import { StandardLayout, MegabufferBuilder } from '../../../dist/geometry.js';
import { VisibilitySystem } from '../../../dist/visibility.js';
import { GLBParser, bakeRigidAnimation } from '../../../dist/loaders.js';
import { LambertMaterial } from '../../../dist/materials.js';
import { PostChain, ChainPresets } from '../../../dist/post.js';
import { OrbitalCameraState } from '../../../dist/cameras.js';
import { readPNG } from '../anime-look/png-read.mjs';
import { publicFile } from './site.mjs';

const CLIPS = ['idle', 'walk', 'run', 'eat'];
const SPECIES = ['prey', 'wolf'];
const BASE_HALF = 80;

/** The demo's camera distances at 4,096 agents; `default` and `far` grow with the field. */
export const VIEWS = {
    close: { distance: 40, grows: false },
    default: { distance: 190, grows: true },
    far: { distance: 340, grows: true },
};

/** loadHerdModels (site models.ts:13–56), reading the site's public/ folder. */
export function loadAssets(engine) {
    const { device } = engine;
    const animal = (name, length) => {
        const baked = bakeRigidAnimation(GLBParser.parse(publicFile(`3dAssetsNotIncluded/animal-${name}.glb`)), { clips: CLIPS, fps: 30 });
        let min = Infinity, max = -Infinity;
        for (let v = 2; v < baked.vertices.length; v += 8) { min = Math.min(min, baked.vertices[v]); max = Math.max(max, baked.vertices[v]); }
        const scale = length / (max - min);
        for (let v = 0; v < baked.vertices.length; v += 8) for (let k = 0; k < 3; k++) baked.vertices[v + k] *= scale;
        for (let k = 3; k < baked.transforms.length; k += 4) baked.transforms[k] *= scale;
        return { ...baked, radius: baked.bounds.radius * scale };
    };
    const png = readPNG(new Uint8Array(publicFile('textures/colormap.png')));
    const palette = device.createTexture({ label: 'films spike palette', size: [png.width, png.height],
        format: 'rgba8unorm-srgb', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture: palette }, png.rgba, { bytesPerRow: png.width * 4 }, [png.width, png.height]);
    const texture = palette.createView();
    const grass = GLBParser.parse(publicFile('3dAssetsNotIncluded/patch-grass-foliage.glb')).meshes[0];
    return {
        prey: { ...animal('cow', 1.3), texture }, wolf: { ...animal('fox', 1.7), texture },
        grass: { vertices: grass.vertices, indices: grass.indices },
        destroy: () => palette.destroy(),
    };
}

/**
 * The Models look over `sim` (a scenario build: { agents, definition }), drawn into engine.screenTexture
 * through the anime chain. Call after the simulation's stages are attached and before its step chain is
 * taken: this adds the animal-phase stage and the instance stage.
 */
export function buildLook(engine, sim, site, assets, { width, height, chain = 'anime' }) {
    const { device } = engine;
    const { agents } = sim, count = agents.capacity;
    const { animalHook, animalPalette, animalPaletteFits, animalFragment, animalCube, ANIMAL_PHASE, ANIMAL_TEXTURE,
        ANIMAL_LOD_PX, ANIMAL_CUBE_RGB } = site.models;
    const owned = [];
    const own = thing => { owned.push(thing); return thing; };
    const buffer = (label, size, usage) => own(device.createBuffer({ label: `films ${label}`, size, usage }));
    const half = sim.definition.world.maxX;

    const post = own(new PostChain(engine, { gbuffer: { depth: 'depth24plus' } }));
    post.setChain(chain === 'anime' ? ChainPresets.anime() : ChainPresets.faithful());
    post.resize(width, height);
    const scene = engine.createPass({ name: 'films Herd toon field', isMainScreenPass: false,
        ...post.sceneAttachments({ clearValue: { r: 0.12, g: 0.2, b: 0.09, a: 1 } }) });
    const field = own(site.field.createHerdField(engine, scene, half, assets.grass, post.format));

    // The animals: mesh 0 the cow, 1 the fox, as the demo's DART_MESH without picking variants.
    const vis = new VisibilitySystem(engine, { maxInstances: count });
    const mega = new MegabufferBuilder(StandardLayout.strideFloats);
    const parts = [];
    SPECIES.forEach(shape => {
        mega.addMesh(shape, assets[shape].vertices, assets[shape].indices);
        parts.push(...assets[shape].partOfVertex);
    });
    SPECIES.forEach(shape => {
        const cube = animalCube(assets[shape]);
        mega.addMesh(`${shape}-far`, cube.vertices, cube.indices);
    });
    vis.setGeometry(mega);
    SPECIES.forEach((shape, mesh) => agents.setSpecies(shape, { mesh, hoverMesh: mesh, selectedMesh: mesh, boundsRadius: assets[shape].radius }));

    const phase = buffer('animal phases', count * 4, GPUBufferUsage.STORAGE);
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

    const camera = new Camera(60, width / height, 0.5, 1000);
    const scale = half / BASE_HALF;
    /** Points the demo's orbit (azimuth 0.3, polar 0.4) at the field's centre from one of VIEWS. */
    function view(name, seconds = 0) {
        const v = VIEWS[name];
        const distance = v.grows ? v.distance * scale : v.distance;
        const orbit = new OrbitalCameraState({ target: [0, 0, 0], distance, azimuthAngle: 0.3, polarAngle: 0.4,
            minDistance: 6, maxDistance: 340 * scale, maxPolarAngle: 1.45 });
        camera.near = Math.max(0.05, distance * 0.004);
        camera.far = distance * 2 + half * 3;
        camera.updateProjection(width / height);
        camera.updateView(orbit.getPosition(), orbit.target);
        camera.updateTime(seconds);
        engine.updateCamera(camera);
    }
    return {
        usePalette,
        view,
        /** The frame's render sequence after the steps: culls, palettes, the scene, the post chain. */
        sequence() {
            post.upload();
            field.upload();
            vis.upload();
            return [...field.getComputeStages(), ...vis.getComputeStages(), ...palettes, scene, ...post.getPassSequence()];
        },
        destroy() {
            for (const b of batches) vis.removeDrawBatch(b);
            vis.destroy();
            for (const thing of owned.reverse()) thing.destroy();
            engine.clearPasses();
        },
    };
}
