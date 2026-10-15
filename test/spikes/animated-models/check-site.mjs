// D1.3: exercise the site's real preparation and shader strings, without shipping the assets.
// node test/spikes/animated-models/check-site.mjs [site directory]
import assert from 'node:assert/strict';
import { readFileSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { createRequire } from 'node:module';
import { WgslReflect } from 'wgsl_reflect/wgsl_reflect.module.js';
import { buildAgentStepWGSL, defineAgents, Sensors } from '../../../dist/agents.js';
import { buildInstanceDrawWGSL } from '../../../dist/visibility.js';

const site = resolve(process.argv[2] ?? 'D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts');
const demo = join(site, 'src/data/engine-demos/demos/AgentsExperiments/HerdWolvesGPU');
const scratch = mkdtempSync(join(tmpdir(), 'ng-animal-site-'));
const esbuild = createRequire(join(site, 'package.json'))('esbuild');
try {
    const result = await esbuild.build({
        entryPoints: [join(demo, 'models.ts')], bundle: true, write: false, format: 'esm', platform: 'node',
        alias: { 'null-graph/loaders': resolve('dist/loaders.js') },
    });
    const m = await import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
    const urls = [];
    globalThis.fetch = async url => {
        urls.push(url);
        const bytes = readFileSync(join(site, 'public', url));
        return { ok: true, arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) };
    };
    const textures = [];
    const engine = { textureManager: { load: async url => { textures.push(url); return {}; } } };
    const models = await m.loadHerdModels(engine);
    assert.deepEqual(urls, ['/3dAssetsNotIncluded/animal-cow.glb', '/3dAssetsNotIncluded/animal-fox.glb']);
    assert.deepEqual(textures, ['/textures/colormap.png', '/textures/colormap.png']);
    globalThis.GPUBufferUsage = { STORAGE: 128, COPY_DST: 8 };
    engine.device = { createBuffer: config => ({ ...config, destroy() {} }), queue: { writeBuffer() {} } };
    for (const [name, model] of Object.entries(models)) {
        assert.deepEqual(model.clips.map(c => c.name), ['idle', 'walk', 'run', 'eat']);
        const zs = Array.from({ length: model.vertexCount }, (_, i) => model.vertices[i * 8 + 2]);
        assert.ok(Math.abs(Math.max(...zs) - Math.min(...zs) - (name === 'prey' ? 1.3 : 1.7)) < 1e-6);
        const hook = m.animalHook(engine, model, model.partOfVertex, {}, {}, {}, x => x);
        assert.equal(hook.entries.length, 5);
        assert.ok(hook.wgsl.includes('inst.sourceIndex'));
        assert.ok(!hook.wgsl.includes('camera.simTime'));
        const wgsl = buildInstanceDrawWGSL({ vertexHook: hook, extraBindings: m.ANIMAL_TEXTURE, fragmentBody: m.animalFragment(false) });
        const reflected = new WgslReflect(wgsl);
        assert.equal(reflected.entry.vertex.length, 1);
        assert.equal(reflected.entry.fragment.length, 1);
        console.log(`ok ${name}: correct asset/palette, metre-sized mesh, clips, public hook and fragment parse`);
    }
    const def = defineAgents({ capacity: 4, world: { min: [-8, -8], max: [8, 8] }, cellSize: 8,
        species: ['prey', 'wolf'].map(name => ({ name, radius: 0.4, maxSpeed: 4, maxTurn: 3, accel: 5, senseRadius: 8 })), sensors: [Sensors.self()] });
    assert.equal(new WgslReflect(buildAgentStepWGSL(def, 'animal-phase', m.ANIMAL_PHASE)).entry.compute.length, 1);
    assert.ok(m.ANIMAL_PHASE.includes('fract(animalPhase[i] + DT * rate)'));
    console.log('ok phase: real app-stage shader parses; accumulated fixed-step phase, not age/wall time');
} finally {
    rmSync(scratch, { recursive: true, force: true });
}