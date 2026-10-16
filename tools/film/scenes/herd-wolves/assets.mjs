// D2.1 (docs/offline-films-design.md §10): the assets the Herd & Wolves look draws, read from the test
// site's public/ folder (NG_SITE), never copied into this repository (the user's rule for models).
//
// This is the adapted part of the site's loadHerdModels (models.ts): the copy fetches root-relative URLs
// and loads its palette with TextureManager.load, which needs createImageBitmap, a browser API. The bake
// below is the same: the clips, 30 fps, the lengths (cow 1.3 m, fox 1.7 m), vertices and part
// translations scaled together, and the palette as an sRGB texture.
import { existsSync, readFileSync } from 'node:fs';
import { GLBParser, bakeRigidAnimation } from '../../../../dist/loaders.js';
import { readPNG } from '../../png-read.mjs';
import { SITE } from '../../check.mjs';

const CLIPS = ['idle', 'walk', 'run', 'eat'];

/** A file of the site's public/ folder, as an ArrayBuffer. */
export function publicFile(path) {
    const file = `${SITE}/public/${path}`;
    if (!existsSync(file)) throw new Error(`film: ${path} is missing from the site's public/ (${file}); set NG_SITE to the test site.`);
    const bytes = readFileSync(file);
    return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

/** { prey, wolf, grass, destroy() }: the shape loadHerdModels resolves to. */
export function loadHerdAssets(engine) {
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
    const palette = device.createTexture({ label: 'film Herd palette', size: [png.width, png.height],
        format: 'rgba8unorm-srgb', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture: palette }, png.rgba, { bytesPerRow: png.width * 4 }, [png.width, png.height]);
    const texture = palette.createView();
    const grass = GLBParser.parse(publicFile('3dAssetsNotIncluded/patch-grass-foliage.glb')).meshes?.[0];
    if (!grass) throw new Error('film: the grass asset has no geometry.');
    return {
        prey: { ...animal('cow', 1.3), texture }, wolf: { ...animal('fox', 1.7), texture },
        grass: { vertices: grass.vertices, indices: grass.indices },
        destroy: () => palette.destroy(),
    };
}
