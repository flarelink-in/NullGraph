// Phase 5.4 intake, without copying proprietary assets into this repository.
// node test/spikes/animated-models/check-skinned-asset.mjs <absolute.glb> [--herd]
// --herd requires the demo's four semantic clips; otherwise inspect all available clips.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { GLBParser, bakeSkinnedAnimation } from '../../../dist/loaders.js';

const args = process.argv.slice(2);
const file = args.find(arg => !arg.startsWith('--'));
assert.ok(file && args.every(arg => arg === file || arg === '--herd'),
    'usage: node check-skinned-asset.mjs <absolute.glb> [--herd]');
const bytes = readFileSync(resolve(file));
const glb = GLBParser.parse(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
try {
    const baked = bakeSkinnedAnimation(glb, {
        fps: 30,
        ...(args.includes('--herd') ? { clips: ['idle', 'walk', 'run', 'eat'] } : {}),
    });
    for (const key of ['vertices', 'weightsOfVertex', 'transforms'])
        assert.ok(baked[key].every(Number.isFinite), `${key} contains non-finite values`);
    for (let v = 0; v < baked.vertexCount; v++) {
        let sum = 0;
        for (let k = 0; k < 4; k++) {
            const i = v * 4 + k;
            assert.ok(baked.influencesOfVertex[i] < baked.influences, `vertex ${v}: invalid influence`);
            sum += baked.weightsOfVertex[i];
        }
        assert.ok(Math.abs(sum - 1) < 1e-5, `vertex ${v}: weights do not sum to one`);
    }
    console.log(JSON.stringify({
        file: resolve(file), vertices: baked.vertexCount, triangles: baked.indexCount / 3,
        influences: baked.influences, frames: baked.frames, clips: baked.clips,
        bounds: baked.bounds, tableBytes: baked.transforms.byteLength,
        paletteBytesPerDraw: Object.fromEntries([4096, 16384, 65536].map(n => [n, n * baked.influences * 48])),
        note: 'Bake/intake only: material, orientation, animated picking and GPU rendering still need visual validation.',
    }, null, 2));
} finally {
    glb.imageUrls?.forEach(url => URL.revokeObjectURL(url));
}