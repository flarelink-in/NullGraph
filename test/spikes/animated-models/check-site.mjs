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
        alias: { 'null-graph/loaders': resolve('dist/loaders.js'), 'null-graph/visibility': resolve('dist/visibility.js') },
    });
    const m = await import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
    const source = readFileSync(join(demo, 'index.ts'), 'utf8');
    assert.ok(!source.includes('await loadHerdModels(engine)'), 'startup must not await model downloads');
    assert.ok(source.includes("if (modelState !== 'idle') return"), 'repeat selections reuse the loaded assets');
    const options = readFileSync(join(demo, 'options.ts'), 'utf8');
    assert.ok(options.includes("['models', 'darts', 'arrows']"));
    const optionBundle = await esbuild.build({
        entryPoints: [join(demo, 'options.ts')], bundle: true, write: false, format: 'esm', platform: 'node',
    });
    const { HERD_DEFAULTS } = await import(`data:text/javascript;base64,${Buffer.from(optionBundle.outputFiles[0].text).toString('base64')}`);
    assert.equal(HERD_DEFAULTS.drawAs, 'models');
    assert.equal(HERD_DEFAULTS.preyBrain, 'network');
    assert.equal(HERD_DEFAULTS.wolfBrain, 'network');
    // Exercise the dashboard's actual store, not just the demo's fallback options.
    const storeBundle = await esbuild.build({
        stdin: {
            contents: `export { useDemoControlStore } from './src/store/useDemoControlStore';
                export { useEngineStore } from './src/store/useEngineStore';`,
            resolveDir: site,
        },
        absWorkingDir: site, bundle: true, write: false, format: 'esm', platform: 'node',
    });
    const { useDemoControlStore, useEngineStore } = await import(`data:text/javascript;base64,${Buffer.from(storeBundle.outputFiles[0].text).toString('base64')}`);
    const initial = useDemoControlStore.getState().configs.AGENTS_HERD_GPU;
    for (const [key, value] of Object.entries(HERD_DEFAULTS)) {
        assert.equal(initial[key], value, `dashboard default: ${key}`);
    }
    useDemoControlStore.getState().updateConfig('AGENTS_HERD_GPU', { paused: true });
    const sent = useEngineStore.getState().pendingMessages.at(-1).message;
    assert.equal(sent.type, 'UPDATE_STATE');
    assert.equal(sent.payload.id, 'AGENTS_HERD_GPU');
    assert.equal(sent.payload.state.drawAs, 'models');
    assert.equal(sent.payload.state.preyBrain, 'network');
    assert.equal(sent.payload.state.wolfBrain, 'network');
    useDemoControlStore.getState().updateConfig('AGENTS_HERD_GPU', { drawAs: 'arrows', preyBrain: 'rules' });
    useDemoControlStore.getState().updateConfig('AGENTS_HERD_GPU', { paused: false });
    assert.equal(useDemoControlStore.getState().configs.AGENTS_HERD_GPU.drawAs, 'arrows');
    assert.equal(useDemoControlStore.getState().configs.AGENTS_HERD_GPU.preyBrain, 'rules');
    console.log('ok dashboard: shared startup defaults reach worker updates; user choices remain editable');
    const urls = [];
    globalThis.fetch = async url => {
        urls.push(url);
        const bytes = readFileSync(join(site, 'public', url));
        return { ok: true, arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) };
    };
    const textures = [];
    const engine = { textureManager: { load: async url => { textures.push(url); return {}; } } };
    const models = await m.loadHerdModels(engine);
    assert.deepEqual(urls, ['/3dAssetsNotIncluded/animal-cow.glb', '/3dAssetsNotIncluded/animal-fox.glb', '/3dAssetsNotIncluded/patch-grass-foliage.glb']);
    assert.deepEqual(textures, ['/textures/colormap.png']);
    assert.ok(models.grass.vertices.length > 0);
    const field = await esbuild.build({
        entryPoints: [join(demo, 'field.ts')], bundle: true, write: false, format: 'esm', platform: 'node',
        external: ['null-graph/*'],
    });
    // Extract the actual exported shader templates without resolving the site's engine package.
    const fieldSource = readFileSync(join(demo, 'field.ts'), 'utf8');
    for (const name of ['FIELD_FRAGMENT', 'GRASS_FRAGMENT']) {
        const fragmentBody = fieldSource.match(new RegExp(`export const ${name} = /\\* wgsl \\*/ \x60([\\s\\S]*?)\x60;`))?.[1];
        assert.ok(fragmentBody, `${name} exists`);
        assert.equal(new WgslReflect(buildInstanceDrawWGSL({ fragmentBody })).entry.fragment.length, 1);
    }
    assert.ok(field.outputFiles[0].text.length > 0);
    console.log('ok field: real ground/grass shaders parse; Models/network defaults, lazy cached asset entry');
    globalThis.GPUBufferUsage = { STORAGE: 128, COPY_DST: 8 };
    engine.device = { createBuffer: config => ({ ...config, destroy() {} }), queue: { writeBuffer() {} } };
    engine.device.limits = { maxStorageBufferBindingSize: 128 * 1024 * 1024, maxBufferSize: 256 * 1024 * 1024 };
    assert.equal(m.animalPaletteFits(engine, models, 4096, 3), true);
    assert.equal(m.animalPaletteFits(engine, models, 1048576, 1), false);
    const stages = [];
    engine.createComputeStage = config => { stages.push(config); return config; };
    engine.setComputePassBindGroup = () => {};
    assert.ok(source.indexOf('...target.palettes') > source.indexOf('target.vis.getComputeStages()'));
    assert.ok(source.indexOf('...target.palettes') < source.indexOf('...target.names.getComputeStages()', source.indexOf('...target.palettes')));
    for (const [name, model] of [['prey', models.prey], ['wolf', models.wolf]]) {
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
        const palette = m.animalPalette(engine, model, model.partOfVertex, {}, {}, {}, 128, x => x);
        assert.equal(palette.hook.instanceIndex, true);
        assert.equal(palette.hook.entries.length, 2);
        const stage = palette.attach({ storageBuffer: {}, indirectBuffer: {} });
        assert.equal(new WgslReflect(stage.shaderCode).entry.compute.length, 1);
        assert.deepEqual(stage.dispatchCount, [2, model.parts, 1]);
        assert.ok(stage.shaderCode.includes('visible[slot].sourceIndex'));
        assert.ok(stage.shaderCode.includes('slot >= drawArgs[1]'));
        assert.ok(palette.hook.wgsl.includes('slot, animalPartOf[vi]'));
        assert.equal(new WgslReflect(buildInstanceDrawWGSL({ vertexHook: palette.hook })).entry.vertex.length, 1);
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