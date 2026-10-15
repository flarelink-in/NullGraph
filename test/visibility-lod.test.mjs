// Cycle D, step 1, phase 4 review (docs/agents-animated-models-design.md, "Phase 4 review"): level of
// detail by size on screen for instance batches, headless.
//
//   U41  without a band the cull WGSL is byte for byte what it was; banded culls parse, and test the
//        size after the frustum and before occlusion
//   U42  createInstanceDrawBatch takes a band into its cull and refuses a malformed one before it
//        allocates anything; adjoining bands keep each size in exactly one batch
//
// Imports dist/: run `npm run build:lib` first.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { WgslReflect } from '../node_modules/wgsl_reflect/wgsl_reflect.module.js';
import { buildInstanceCullWGSL, VisibilitySystem } from '../dist/visibility.js';
import { MegabufferBuilder } from '../dist/geometry.js';
import { standIn } from './stand-in-engine.mjs';

const sha = code => createHash('sha256').update(code).digest('hex');
// Captured from the build before the band existed (2026-10-15).
const BEFORE = {
    plain: '3ce6397f40b988317317234922eb8443686d68fe3508e0d19682d28512c2c8f6',
    occlusion: '3ed43b23637aeb4b44e9005b955d3882d7782d10e3a9c2dd839da3dcd14b5df8',
};

test('U41: no band leaves the cull as it was; a band is tested after the frustum, before occlusion', () => {
    assert.equal(sha(buildInstanceCullWGSL()), BEFORE.plain);
    assert.equal(sha(buildInstanceCullWGSL({ occlusion: false, lod: undefined })), BEFORE.plain);
    assert.equal(sha(buildInstanceCullWGSL({ lod: {} })), BEFORE.plain, 'an empty band is no band');
    assert.equal(sha(buildInstanceCullWGSL({ occlusion: true })), BEFORE.occlusion);
    for (const [lod, test_] of [[{ minPixels: 12 }, 'px < 12'], [{ maxPixels: 12.5 }, 'px >= 12.5'], [{ minPixels: 3, maxPixels: 40 }, 'px < 3 || px >= 40']]) {
        for (const occlusion of [false, true]) {
            const code = buildInstanceCullWGSL({ occlusion, lod });
            assert.equal(new WgslReflect(code).entry.compute.length, 1, `${JSON.stringify(lod)} parses`);
            assert.ok(code.includes(`if (${test_}) { return; }`), `${JSON.stringify(lod)}: ${test_}`);
            const at = name => code.indexOf(name);
            assert.ok(at('sphereInFrustum(planes') < at('let px') && at('let px') < at('atomicAdd(&drawArgs'), 'the band sits between the frustum test and the slot');
            if (occlusion) assert.ok(at('let px') < at('occludedByHiZ(ws'), 'and before the occlusion test');
        }
    }
    // The size is the meshlet tier's: the sphere's nearest distance, the vertical field of view, the screen height.
    assert.match(buildInstanceCullWGSL({ lod: { minPixels: 1 } }),
        /let px = ws\.w \* params\.screenSize\.y \/\s+\(max\(length\(ws\.xyz - camera\.eyePos\) - ws\.w, camera\.zNear\) \* tan\(camera\.fovRads \* 0\.5\)\);/);
});

function scene() {
    const { engine, stages } = standIn(), configs = [];
    engine.ctx = { cameraUniformBuffer: {}, width: 64, height: 64 };
    engine.device.createBuffer = d => ({ ...d, getMappedRange: () => new ArrayBuffer(d.size), unmap() {}, destroy() {} });
    engine.createBatch = (pass, config) => { configs.push(config); const b = { storageBuffer: {}, indirectBuffer: {} }; pass.batches.push(b); return b; };
    engine.setBatchGeometry = () => {};
    engine.setRenderPassBindGroup = () => {};
    engine.clearBatch = () => {};
    const mega = new MegabufferBuilder(8);
    mega.addMesh('model', new Float32Array(24), [0, 1, 2]);
    mega.addMesh('cube', new Float32Array(24), [0, 1, 2]);
    const vis = new VisibilitySystem(engine, { maxInstances: 4 });
    vis.setGeometry(mega);
    return { vis, stages, configs, pass: { batches: [] } };
}

test('U42: a band reaches the batch\'s cull; malformed bands are refused before anything is made', () => {
    const { vis, stages, configs, pass } = scene();
    vis.createInstanceDrawBatch(pass, 'model', 0, { lod: { minPixels: 12 } });
    vis.createInstanceDrawBatch(pass, 'cube', 0, { lod: { maxPixels: 12 } });
    vis.createInstanceDrawBatch(pass, 'model', 1);
    assert.deepEqual(stages.map(s => s.shaderCode.match(/if \((px[^)]*)\)/)?.[1] ?? null), ['px < 12', 'px >= 12', null]);
    assert.equal(sha(stages[2].shaderCode), BEFORE.plain, 'a batch without a band culls as before');

    const before = { stages: stages.length, configs: configs.length };
    for (const lod of [{}, { minPixels: -1 }, { maxPixels: NaN }, { minPixels: Infinity }, { maxPixels: '12' },
        { minPixels: 12, maxPixels: 12 }, { minPixels: 20, maxPixels: 10 }]) {
        assert.throws(() => vis.createInstanceDrawBatch(pass, 'model', 0, { lod }),
            /^Error: VisibilitySystem: lod needs minPixels or maxPixels, each a finite number ≥ 0, and minPixels below maxPixels; got /, JSON.stringify(lod));
    }
    assert.deepEqual({ stages: stages.length, configs: configs.length }, before, 'nothing was made for a refused band');

    // Adjoining bands split every size between them: [0, T) far, [T, ∞) near.
    // As the WGSL tests it: a bound left out is not tested at all.
    const inBand = (px, { minPixels, maxPixels }) => !(minPixels !== undefined && px < minPixels) && !(maxPixels !== undefined && px >= maxPixels);
    for (const px of [0, 0.5, 11.999, 12, 12.001, 400, Infinity]) {
        assert.equal([{ minPixels: 12 }, { maxPixels: 12 }].filter(band => inBand(px, band)).length, 1, `size ${px}`);
    }
});
