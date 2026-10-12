// Step 4, phase 4.2 (docs/agents-gpu-loop-design.md §9.2, §9.3): the instance stage, its CPU twin,
// the highlight uniform and the refusals. Headless: the twin is the definition A17 holds the GPU to.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { WgslReflect } from '../node_modules/wgsl_reflect/wgsl_reflect.module.js';
import * as A from '../dist/agents.js';
import { INSTANCE_FLOATS, INSTANCE_MESH_INDEX_OFFSET, LOD_AUTO, INSTANCE_STRUCT_WGSL } from '../dist/visibility.js';

globalThis.GPUBufferUsage = { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64 };
// The real constructor, captured before any test counts allocations through a proxy.
const U32 = Uint32Array;

const MESH = { prey: [1, 2, 3], wolf: [4, 5, 6] };
const definition = (extra = {}) => A.defineAgents({ capacity: 96,
    world: { min: [-8, -8], max: [8, 8] }, cellSize: 4,
    species: [
        { name: 'prey', maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 4, radius: 0.4,
            mesh: MESH.prey[0], hoverMesh: MESH.prey[1], selectedMesh: MESH.prey[2], boundsRadius: 0.8 },
        { name: 'wolf', maxSpeed: 4, maxTurn: 3, accel: 5, senseRadius: 4, radius: 0.6,
            mesh: MESH.wolf[0], hoverMesh: MESH.wolf[1], selectedMesh: MESH.wolf[2], boundsRadius: 1.1 },
    ],
    sensors: [A.Sensors.self()], ...extra });

function mock() {
    const writes = [], buffers = [], stages = [];
    const engine = {
        device: { limits: { maxStorageBufferBindingSize: 128 * 1024 * 1024, maxBufferSize: 256 * 1024 * 1024,
            maxComputeWorkgroupsPerDimension: 65535, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256,
            maxStorageBuffersPerShaderStage: 8 },
        createBuffer(o) { const b = { ...o, destroyed: false, destroy() { this.destroyed = true; } }; buffers.push(b); return b; },
        // Copies the data: a caller may reuse (and later change) the array it wrote from.
        queue: { writeBuffer(target, offset, data) { writes.push({ target, offset, words: [...new U32(data.buffer, data.byteOffset, data.byteLength / 4)] }); },
            async onSubmittedWorkDone() {} } },
        createComputeStage(o) { const s = { ...o, enabled: true, preClearBuffers: [] }; stages.push(s); return s; },
        setComputePassBindGroup(stage, entries, group) { if (group === 0) stage.entries = entries; },
        render() {},
    };
    return { engine, writes, buffers, stages };
}
/** A stand-in VisibilitySystem: only what toInstances reads (its engine is TS-private there). */
const fakeVis = (engine, maxInstances, source = 'gpu') => ({
    engine, instanceSource: source,
    instanceBuffer: { size: maxInstances * INSTANCE_FLOATS * 4, label: 'vis rows', destroy() { this.destroyed = true; } },
});
function makeSystem(engine, def = definition()) {
    const s = new A.AgentSystem(engine, def);
    s.setBrain(new A.RuleBrain('fn think(i: u32, species: u32, tick: u32) {}'));
    return s;
}

test('U19: the row layout is visibility\'s Instance, word for word', () => {
    assert.equal(A.INSTANCE_ROW_FLOATS, INSTANCE_FLOATS);
    assert.equal(A.INSTANCE_ROW_MESH_OFFSET, INSTANCE_MESH_INDEX_OFFSET);
    assert.equal(A.INSTANCE_ROW_LOD_AUTO, LOD_AUTO);
    // The generated struct and visibility's agree on every offset the cull reads.
    const ours = new WgslReflect(A.buildAgentInstanceWGSL(definition(), 0)).structs.find(s => s.name === 'InstanceRow');
    const theirs = new WgslReflect(`${INSTANCE_STRUCT_WGSL}\n@compute @workgroup_size(1) fn cs_main() {}`).structs.find(s => s.name === 'Instance');
    assert.equal(ours.size, theirs.size);
    const at = (s, name) => s.members.find(m => m.name === name).offset;
    assert.equal(at(ours, 'model'), at(theirs, 'model'));
    assert.equal(at(ours, 'bounds'), at(theirs, 'boundsCenter'));
    for (const name of ['meshIndex', 'lodOverride', 'flags', 'sourceIndex']) assert.equal(at(ours, name), at(theirs, name), name);
});

test('U19: the CPU twin — local +Z is forward, +X is the side, bounds, LOD, and the dead get HIDDEN only', () => {
    const def = definition(), rec = new A.AgentRecords(def);
    const headings = [0, Math.PI / 2, -Math.PI / 2, 2.5, -3, 1];
    headings.forEach((heading, i) => rec.spawn(i, { position: [i - 2, 0.25 * i, 3 - i], species: i % 2 ? 'wolf' : 'prey', heading }));
    rec.kill(5);
    const out = new Float32Array(8 * A.INSTANCE_ROW_FLOATS).fill(7), u = new Uint32Array(out.buffer);
    for (let i = 0; i < 6; i++) A.writeAgentInstanceRow(def, rec, i, { selected: -1, hovered: -1 }, out, i);
    const f32 = Math.fround;
    for (let i = 0; i < 5; i++) {
        const o = i * 24, h = rec.data[i * 16 + 7], p = [rec.data[i * 16], rec.data[i * 16 + 1], rec.data[i * 16 + 2]];
        // A local point one metre to the side and two ahead lands where the act integrator's frame says.
        const local = [1, 0, 2, 1], world = [0, 1, 2].map(r => [0, 1, 2, 3].reduce((sum, c) => sum + out[o + c * 4 + r] * local[c], 0));
        const want = [p[0] + Math.cos(h) + 2 * Math.sin(h), p[1], p[2] - Math.sin(h) + 2 * Math.cos(h)];
        world.forEach((v, k) => assert.ok(Math.abs(v - want[k]) < 1e-5, `agent ${i} axis ${k}: ${v} vs ${want[k]}`));
        assert.deepEqual([...out.subarray(o + 12, o + 16)], [...p.map(f32), 1]);
        assert.deepEqual([out[o + 3], out[o + 7], out[o + 11], out[o + 4], out[o + 5], out[o + 6]], [0, 0, 0, 0, 1, 0]);
        assert.deepEqual([...out.subarray(o + 16, o + 20)], [0, 0, 0, f32(i % 2 ? 1.1 : 0.8)]);
        assert.deepEqual([u[o + 20], u[o + 21], u[o + 22], u[o + 23]], [i % 2 ? MESH.wolf[0] : MESH.prey[0], LOD_AUTO, 0, 0]);
    }
    const dead = 5 * 24;
    assert.equal(u[dead + 20], A.HIDDEN_MESH);
    assert.ok([...out.subarray(dead, dead + 20)].every(v => v === 7), 'a dead agent\'s other words must stay untouched');
});

test('U19: the highlight picks the selected, then the hovered variant; selected wins on the same agent', () => {
    const def = definition(), rec = new A.AgentRecords(def);
    for (let i = 0; i < 4; i++) rec.spawn(i, { position: [0, 0, 0], species: i < 2 ? 'prey' : 'wolf' });
    const out = new Float32Array(4 * 24), u = new Uint32Array(out.buffer);
    const meshes = (highlight) => [0, 1, 2, 3].map(i => { A.writeAgentInstanceRow(def, rec, i, highlight, out, i); return u[i * 24 + 20]; });
    assert.deepEqual(meshes({ selected: 1, hovered: 2 }), [1, 3, 5, 4]);
    assert.deepEqual(meshes({ selected: 3, hovered: 3 }), [1, 1, 4, 6]);
    assert.deepEqual(meshes({ selected: -1, hovered: -1 }), [1, 1, 4, 4]);
});

test('U9 instances: the module parses, binds agents/rows/species/highlight and bakes capacity and firstInstance', () => {
    const code = A.buildAgentInstanceWGSL(definition(), 37);
    const r = new WgslReflect(code);
    assert.equal(r.entry.compute.length, 1);
    assert.equal(r.entry.compute[0].name, 'cs_main');
    assert.deepEqual(r.storage.map(v => v.name).sort(), ['agents', 'rows']);
    assert.deepEqual(r.uniforms.map(v => v.name).sort(), ['highlight', 'species']);
    const body = code.slice(code.indexOf('fn cs_main'));
    for (const name of ['agents', 'rows', 'species', 'highlight']) assert.match(body, new RegExp(`\\b${name}[\\[.]`));
    assert.match(body, /i >= 96u/);
    assert.match(body, /let r = 37u \+ i;/);
    assert.ok(!code.includes('`'));
});

test('U14 toInstances: every refusal names the offender and happens before anything is allocated', () => {
    const m = mock(), s = makeSystem(m.engine);
    const before = () => [m.buffers.length, m.stages.length, s.getComputeStages().length];
    const start = before();
    const refuses = (vis, options, pattern) => {
        assert.throws(() => s.toInstances(vis, options), pattern);
        assert.deepEqual(before(), start, 'a refused toInstances allocated or registered something');
    };
    refuses(fakeVis({}, 200), {}, /^Error: AgentSystem: that VisibilitySystem belongs to a different engine\.$/);
    refuses(fakeVis(m.engine, 200), { firstInstance: -1 }, /firstInstance -1 must be a whole number/);
    refuses(fakeVis(m.engine, 200), { firstInstance: 1.5 }, /firstInstance 1\.5 must be a whole number/);
    refuses(fakeVis(m.engine, 100), { firstInstance: 5 }, /needs maxInstances ≥ 101; this VisibilitySystem has 100\./);
    refuses(fakeVis(m.engine, 200, 'cpu'), {}, /use vis\.setInstanceCount\(n\), not setInstances\(\)/);
    const bare = makeSystem(m.engine, definition({ species: [{ name: 'ghost', maxSpeed: 1, maxTurn: 1, accel: 1,
        senseRadius: 4, radius: 0.5, mesh: 9 }] }));
    assert.throws(() => bare.toInstances(fakeVis(m.engine, 200)),
        /species "ghost" draws mesh 9 with boundsRadius 0 — the cull would drop it while on screen/);
    // Hidden species need no bounds.
    const hidden = makeSystem(m.engine, definition({ species: [{ name: 'ghost', maxSpeed: 1, maxTurn: 1, accel: 1,
        senseRadius: 4, radius: 0.5 }] }));
    hidden.toInstances(fakeVis(m.engine, 200));
    [s, bare, hidden].forEach(x => x.destroy());
});

test('chains: the instance stage closes every chain once, also with no step; retargeting replaces it', () => {
    const m = mock(), s = makeSystem(m.engine), vis = fakeVis(m.engine, 200);
    s.advance(s.definition.dt);
    const step = s.getComputeStages().length;
    s.toInstances(vis, { firstInstance: 4 });
    const stage = m.stages.find(x => x.name === 'AgentSystem instances');
    assert.deepEqual(stage.entries.map(e => e.resource.buffer), [s.buffer, vis.instanceBuffer, s.species, m.buffers.at(-1)]);
    assert.equal(m.buffers.at(-1).size, 16);
    assert.equal(stage.dispatchCount[0], Math.ceil(96 / 64));
    for (let n = 0; n <= 3; n++) {
        s.advance(n * s.definition.dt);
        const chain = s.getComputeStages();
        assert.equal(chain.length, n * step + 1, `chains[${n}]`);
        assert.equal(chain.at(-1), stage);
        assert.equal(chain.filter(x => x === stage).length, 1);
    }
    s.toInstances(vis);
    const replacement = m.stages.filter(x => x.name === 'AgentSystem instances').at(-1);
    assert.equal(stage.enabled, false);
    s.advance(0);
    assert.deepEqual(s.getComputeStages(), [replacement]);
    assert.equal(m.buffers.filter(b => b.size === 16 && b.usage === (64 | 8)).length, 1, 'retargeting allocated a second highlight');
    s.destroy();
    assert.equal(replacement.enabled, false);
    assert.equal(m.buffers.find(b => b.label === 'AgentSystem highlight').destroyed, true);
    assert.ok(!vis.instanceBuffer.destroyed, 'destroy() must leave the VisibilitySystem\'s buffer alone');
});

test('setHighlight: validates indices, writes 16 bytes only on a change, allocates nothing per call', () => {
    const m = mock(), s = makeSystem(m.engine);
    s.setHighlight({ selected: 3 });   // before toInstances: kept, written when the buffer exists
    assert.equal(m.writes.length, 0);
    s.toInstances(fakeVis(m.engine, 200));
    const highlight = m.buffers.find(b => b.label === 'AgentSystem highlight');
    const written = () => m.writes.filter(w => w.target === highlight).map(w => w.words);
    assert.deepEqual(written(), [[3, A.HIGHLIGHT_NONE, 0, 0]]);
    for (const bad of [96, -2, 1.5, NaN]) assert.throws(() => s.setHighlight({ hovered: bad }), new RegExp(`index ${bad} is outside capacity 96`));
    const original = [globalThis.Float32Array, globalThis.Uint32Array];
    let allocations = 0;
    try {
        [globalThis.Float32Array, globalThis.Uint32Array] = original.map(T => new Proxy(T, {
            construct(target, args) { allocations++; return Reflect.construct(target, args); },
        }));
        s.setHighlight({ hovered: 7 });
        s.setHighlight({ hovered: 7 });          // unchanged: no write
        s.setHighlight({ selected: 3 });         // unchanged: no write
        s.setHighlight({ selected: -1, hovered: 7 });
        assert.equal(allocations, 0);
    } finally { [globalThis.Float32Array, globalThis.Uint32Array] = original; }
    assert.deepEqual(written(), [[3, A.HIGHLIGHT_NONE, 0, 0], [3, 7, 0, 0], [A.HIGHLIGHT_NONE, 7, 0, 0]]);
    s.destroy();
});

test('setSpecies keeps the zero-bounds refusal once instances are on, atomically', () => {
    const m = mock(), s = makeSystem(m.engine);
    s.setSpecies('prey', { boundsRadius: 0 });   // allowed: nothing draws yet
    assert.throws(() => s.toInstances(fakeVis(m.engine, 200)), /species "prey" draws mesh 1 with boundsRadius 0/);
    s.setSpecies('prey', { boundsRadius: 0.8 });
    s.toInstances(fakeVis(m.engine, 200));
    assert.throws(() => s.setSpecies('wolf', { boundsRadius: 0 }), /species "wolf" draws mesh 4 with boundsRadius 0/);
    assert.equal(s.definition.species[1].boundsRadius, 1.1, 'a refused setSpecies changed the table');
    s.setSpecies('wolf', { mesh: A.HIDDEN_MESH, hoverMesh: A.HIDDEN_MESH, selectedMesh: A.HIDDEN_MESH, boundsRadius: 0 });
    s.destroy();
});
