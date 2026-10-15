import { test } from 'node:test';
import assert from 'node:assert/strict';
import { WgslReflect } from '../node_modules/wgsl_reflect/wgsl_reflect.module.js';
import { BAKED_POSE_WGSL, BAKED_SAMPLE_WGSL, BAKED_PALETTE_WRITE_WGSL, BAKED_PALETTE_READ_WGSL } from '../dist/loaders.js';

test('palette snippets declare no bindings and compose into compute and vertex modules', () => {
    for (const snippet of [BAKED_POSE_WGSL, BAKED_SAMPLE_WGSL, BAKED_PALETTE_WRITE_WGSL, BAKED_PALETTE_READ_WGSL]) {
        assert.equal(typeof snippet, 'string');
        assert.doesNotMatch(snippet, /@group|@binding|@compute|@vertex/);
    }
    const compute = `${BAKED_POSE_WGSL}
        @group(0) @binding(0) var<storage, read> bakedTransforms: array<vec4<f32>>;
        @group(0) @binding(1) var<storage, read_write> bakedPalette: array<vec4<f32>>;
        ${BAKED_SAMPLE_WGSL} ${BAKED_PALETTE_WRITE_WGSL}
        @compute @workgroup_size(1) fn cs_main() {
            bakedWritePalette(0u, 2u, 0u, 1u, bakedClipPose(0u, 2u, vec2<u32>(0u, 1u), -0.25, true, 1u));
        }`;
    assert.equal(new WgslReflect(compute).entry.compute.length, 1);
    const vertex = `${BAKED_POSE_WGSL}
        @group(0) @binding(0) var<storage, read> bakedPalette: array<vec4<f32>>;
        ${BAKED_PALETTE_READ_WGSL}
        @vertex fn vs_main() -> @builtin(position) vec4<f32> {
            let p = bakedSkinPose(0u, 2u, 0u, vec4<u32>(0u, 1u, 0u, 0u), vec4<f32>(0.5, 0.5, 0.0, 0.0));
            return vec4<f32>(bakedPosition(p, vec3<f32>(0.0)), 1.0);
        }`;
    assert.equal(new WgslReflect(vertex).entry.vertex.length, 1);
});