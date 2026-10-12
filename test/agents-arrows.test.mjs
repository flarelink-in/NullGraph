import { test } from 'node:test';
import assert from 'node:assert/strict';
import { WgslReflect } from '../node_modules/wgsl_reflect/wgsl_reflect.module.js';
import { agentArrows } from '../dist/debug-ui.js';

globalThis.GPUBufferUsage = { VERTEX: 32, INDEX: 16, COPY_DST: 8 };
function fixture() {
    const buffers = [], writes = [], removed = [];
    const buffer = () => {
        const b = { destroys: 0, destroy() { this.destroys++; } };
        buffers.push(b);
        return b;
    };
    const engine = {
        device: { createBuffer: buffer, queue: { writeBuffer(...args) { writes.push(args); } } },
        createBatch(pass, config) { return { ...config, storageBuffer: buffer() }; },
        setRenderPassBindGroup(batch, entries, group) { batch.entries = entries; batch.group = group; },
        clearBatch(pass, batch) { removed.push(batch); },
    };
    const owned = { destroys: 0, destroy() { this.destroys++; } };
    let destroyed = false;
    const agents = { engine, capacity: 4096, buffer: owned, definition: { species: [{}, {}] },
        grid: { getComputeStages() { if (destroyed) throw new Error('used after destroy'); return []; } } };
    return { engine, agents, buffers, writes, removed, owned, destroySystem() { destroyed = true; } };
}

test('U9 arrows: generated shader parses, uses all bindings, clips dead records and transforms heading', () => {
    const f = fixture(), draw = agentArrows(f.engine, {}, f.agents);
    const shader = new WgslReflect(draw.batch.shaderCode);
    assert.equal(shader.entry.vertex.length, 1);
    assert.equal(shader.entry.fragment.length, 1);
    assert.equal(shader.storage.length, 2);
    assert.equal(shader.uniforms.length, 1);
    assert.match(draw.batch.shaderCode, /a\.ids\.y & 1u/);
    assert.match(draw.batch.shaderCode, /sin\(a\.velHeading\.w\)/);
    assert.match(draw.batch.shaderCode, /a\.posRadius\.xyz/);
    assert.match(draw.batch.shaderCode, /instances\[0\]\.x/);
    assert.ok(!draw.batch.shaderCode.includes('`'));
    assert.equal(draw.batch.currentInstanceCount, 4096);
    assert.equal(draw.batch.indexCount, 3);
    assert.equal(draw.batch.group, 1);
    assert.equal(draw.batch.entries[0].resource.buffer, f.owned);
    assert.equal(f.writes.length, 1); // Static geometry only; no agent uploads.
    assert.equal(f.writes[0][2].byteLength % 4, 0); // WebGPU writeBuffer alignment.
    draw.destroy(); draw.destroy();
    assert.equal(f.removed.length, 1);
    assert.ok(f.buffers.every(b => b.destroys === 1));
    assert.equal(f.owned.destroys, 0);
});

test('arrows: colors and engine/lifetime validation happen before allocating', () => {
    const f = fixture();
    for (const colors of [[[NaN, 0, 0]], [[1.1, 0, 0]], [[0, 0]], [[0, -1, 0]]]) {
        assert.throws(() => agentArrows(f.engine, {}, f.agents, { colors }), /RGB/);
    }
    assert.throws(() => agentArrows({}, {}, f.agents), /this engine/);
    assert.equal(f.buffers.length, 0);
    const draw = agentArrows(f.engine, {}, f.agents, { colors: [[1, 0, 0]] });
    assert.match(draw.batch.shaderCode, /vec3<f32>\(1,0,0\),vec3<f32>\(1,0.18,0.45\)/);
    draw.destroy();
    f.destroySystem();
    assert.throws(() => agentArrows(f.engine, {}, f.agents), /after destroy/);
});