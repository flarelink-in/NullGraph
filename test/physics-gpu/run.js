import * as P from '../../dist/physics.js';
import * as A from '../../dist/agents.js';
import { NullGraph } from '../../dist/index.js';
import { harnessEnv } from '../gpu/harness-env.mjs';

const env = harnessEnv('__PHYSICS_RESULTS__');
const { check } = env;
const engine = new NullGraph();
const init = await engine.init(null, { width: 8, height: 8, desiredFeatures: ['timestamp-query'] });
if (!init.success) { env.fatal(init.error); throw new Error(init.error); }
const d = engine.device;
env.log(`Physics on ${JSON.stringify(d.adapterInfo)}`);
const base = { world: { min: [-32, -32], max: [32, 32] }, maxRadius: 0.6, maxPush: 0.25 };
function buffer(data) {
    const b = d.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(b, 0, data);
    return b;
}
async function read(b) {
    const dst = d.createBuffer({ size: b.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(b, 0, dst, 0, b.size);
    d.queue.submit([enc.finish()]);
    await dst.mapAsync(GPUMapMode.READ);
    const result = dst.getMappedRange().slice(0);
    dst.destroy();
    return result;
}
// Submit the actual library stages, without rendering a screen between simulation steps.
function encode(enc, stages, repeats = 1, timestampWrites) {
    const pass = enc.beginComputePass(timestampWrites ? { timestampWrites } : {});
    for (let step = 0; step < repeats; step++) for (const s of stages) {
        if (!s.enabled) continue;
        pass.setPipeline(s.computePipeline);
        for (const [group, bind] of Object.entries(s.customBindGroups)) pass.setBindGroup(Number(group), bind);
        pass.dispatchWorkgroups(...s.dispatchCount);
    }
    pass.end();
}
function run(stages, repeats = 1) {
    const enc = d.createCommandEncoder(); encode(enc, stages, repeats); d.queue.submit([enc.finish()]);
}
let seed = 12345;
const rand = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
function randomRecords(count, stride = 16) {
    const rec = new Float32Array(count * stride), words = new Uint32Array(rec.buffer);
    for (let i = 0; i < count; i++) {
        const b = i * stride;
        rec[b] = (rand() * 2 - 1) * 30; rec[b + 1] = 0.125;
        rec[b + 2] = (rand() * 2 - 1) * 30; rec[b + 3] = 0.3 + rand() * 0.3;
        rec[b + 4] = rand() - 0.5; rec[b + 5] = 17; rec[b + 6] = rand() - 0.5;
        words[b + 13] = i % 17 === 0 ? 0 : i % 19 === 0 ? 3 : 1;
    }
    return rec;
}
const equal = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
const layout = count => ({ capacity: count, stride: 16, alive: { offset: 13, mask: 1 }, fixed: { offset: 13, mask: 2 }, velocity: { offset: 4 } });

// A37: every combination, including arbitrary (not vec4-aligned) flags and velocity offsets below.
for (const iterations of [1, 2]) for (const mass of ['area', 'equal']) for (const compliance of [0, 1e-4]) {
    const count = 4096, rec = randomRecords(count), l = layout(count);
    const options = { ...base, ...l, iterations, mass, compliance, clamp: true, maxScanPerCell: 128 };
    const b = buffer(rec), c = new P.DiscContacts(engine, b, options);
    const cpu = P.resolveDiscContacts(l, { ...options, cellSize: c.options.cellSize }, rec.slice());
    run(c.getComputeStages());
    const actual = new Float32Array(await read(b)), deltas = new Float32Array(await read(c.deltas));
    let error = 0, counts = true, untouched = true;
    const originalWords = new Uint32Array(rec.buffer), actualWords = new Uint32Array(actual.buffer);
    for (let i = 0; i < count; i++) {
        for (const off of [0, 2, 4, 6]) error = Math.max(error, Math.abs(actual[i * 16 + off] - cpu.records[i * 16 + off]));
        counts &&= deltas[i * 4 + 2] === cpu.contactCounts[i];
        for (const off of [1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]) untouched &&= actualWords[i * 16 + off] === originalWords[i * 16 + off];
        if ((originalWords[i * 16 + 13] & 1) === 0 || (originalWords[i * 16 + 13] & 2) !== 0) {
            untouched &&= equal(actualWords.subarray(i * 16, i * 16 + 16), originalWords.subarray(i * 16, i * 16 + 16));
        }
    }
    check(`A37 K=${iterations} ${mass} alpha=${compliance}: mirror, counts, untouched words`, error < 1e-5 * 64 && counts && untouched, `max error ${error}`);
    c.destroy(); b.destroy();
}

{
    const rec = new Float32Array(3 * 12), words = new Uint32Array(rec.buffer);
    for (let i = 0; i < 3; i++) { rec[i * 12] = i * 0.3; rec[i * 12 + 3] = 0.7; words[i * 12 + 4] = 1; words[i * 12 + 5] = i === 0 ? 2 : 0; rec[i * 12 + 7] = -1; rec[i * 12 + 8] = 42; }
    const l = { capacity: 3, stride: 12, alive: { offset: 4, mask: 1 }, fixed: { offset: 5, mask: 2 }, velocity: { offset: 7 } };
    const b = buffer(rec), c = new P.DiscContacts(engine, b, { ...base, ...l });
    const cpu = P.resolveDiscContacts(l, base, rec.slice());
    run(c.getComputeStages());
    const actual = new Float32Array(await read(b)), counters = await c.readCounters();
    check('A37 arbitrary word offsets, fixed body, oversized radius counter', actual.every((v, i) => Math.abs(v - cpu.records[i]) < 1e-5) && counters.bodiesOverMaxRadius === 3);
    c.enabled = false; run(c.getComputeStages(), 3);
    check('A37 disabled stages leave records unchanged', equal(new Uint32Array(await read(b)), new Uint32Array(actual.buffer)));
    c.enabled = true; c.setParams({ compliance: 1e-4, damping: 0, maxPush: 0.1 });
    check('A37 params update uploads without replacing cached stages', c.getComputeStages() === c.getComputeStages() && c.options.compliance === 1e-4);
    const updated = P.resolveDiscContacts(l, { ...base, compliance: 1e-4, damping: 0, maxPush: 0.1,
        cellSize: c.options.cellSize }, actual.slice());
    run(c.getComputeStages());
    const afterUpdate = new Float32Array(await read(b));
    check('A37 updated params affect the next GPU step', afterUpdate.every((v, i) => Math.abs(v - updated.records[i]) < 1e-5));
    const smaller = afterUpdate.slice();
    for (let i = 0; i < 3; i++) smaller[i * 12 + 3] = 0.5;
    d.queue.writeBuffer(b, 0, smaller); run(c.getComputeStages(), 2);
    const [countsA, countsB] = await Promise.all([c.readCounters(), c.readCounters()]);
    check('A37 radius counter resets each step; concurrent readback agrees', countsA.bodiesOverMaxRadius === 0 && countsB.bodiesOverMaxRadius === 0);
    let refused = false; try { c.setParams({ maxPush: 100 }); } catch { refused = true; }
    check('A37 maxPush cannot invalidate completeness', refused && c.options.maxPush === 0.1);
    c.destroy(); c.destroy();
    let dead = false; try { c.getComputeStages(); } catch { dead = true; }
    check('A37 teardown disables stages and refuses reuse', dead);
    b.destroy();
}

// A38: independent brute-force GPU oracle checks every pass's contact counts.
async function completeness(cellSize, below = false) {
    const count = 4096, rec = randomRecords(count);
    // The loose large-disc layout is sensitive to stale cells on the second pass.
    for (let i = 0; i < count; i++) { rec[i * 16] *= 0.4; rec[i * 16 + 2] *= 0.4; rec[i * 16 + 3] = 0.6; }
    const b = buffer(rec), c = new P.DiscContacts(engine, b, { ...base, ...layout(count), iterations: 2, maxScanPerCell: count });
    const oracle = buffer(new Uint32Array(count));
    const code = `
@group(0) @binding(0) var<storage, read> records: array<f32>;
@group(0) @binding(1) var<storage, read_write> counts: array<u32>;
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
 let i = id.x; if (i >= ${count}u) { return; } counts[i] = 0u;
 let flags = bitcast<u32>(records[i * 16u + 13u]);
 if ((flags & 1u) == 0u || (flags & 2u) != 0u) { return; }
 let p = vec2<f32>(records[i * 16u], records[i * 16u + 2u]);
 for (var j = 0u; j < ${count}u; j++) {
  if (i == j || (bitcast<u32>(records[j * 16u + 13u]) & 1u) == 0u) { continue; }
  let delta = p - vec2<f32>(records[j * 16u], records[j * 16u + 2u]);
  let r = records[i * 16u + 3u] + records[j * 16u + 3u];
  if (dot(delta, delta) < r * r) { counts[i]++; }
 }
}`;
    const stage = engine.createComputeStage({ name: 'A38 brute force', shaderCode: code, dispatchCount: [count / 64, 1, 1] });
    engine.setComputePassBindGroup(stage, [b, oracle].map((buffer, binding) => ({ binding, resource: { buffer } })), 0);
    // Below-bound negative control uses a deliberately undersized grid and the real gather builder.
    let extraGrid;
    let stages = c.getComputeStages();
    if (below) {
        const h = cellSize;
        const grid = { cellsX: Math.ceil(64 / h), cellsZ: Math.ceil(64 / h) }; grid.cellTotal = grid.cellsX * grid.cellsZ;
        const opts = { ...c.options, cellSize: h, grid };
        extraGrid = new A.PointGrid(engine, b, { ...opts, ...opts.layout, flagsOffset: 13, aliveMask: 1 });
        const gathers = stages.filter(s => s.name.startsWith('DiscContacts gather'));
        for (let k = 0; k < 2; k++) {
            const s = engine.createComputeStage({ name: 'A38 undersized gather', shaderCode: P.buildContactGatherWGSL(opts, k), dispatchCount: [count / 64, 1, 1] });
            // Keep params and counters from the real stage; replace only the spatial bindings.
            // Binding resources are not introspectable: supply equivalent test-owned params/counter buffers.
            const params = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            d.queue.writeBuffer(params, 0, new Float32Array([1.75, 0, 0.25, 1]));
            const tick = buffer(new Uint32Array(4));
            engine.setComputePassBindGroup(s, [b, extraGrid.cellOffsets, extraGrid.cellBodies, c.deltas, params, tick]
                .map((buffer, binding) => ({ binding, resource: { buffer } })), 0);
            gathers[k] = s;
            s.testOwned = [params, tick];
        }
        stages = [...extraGrid.getComputeStages(), gathers[0], stages.find(s => s.name === 'DiscContacts apply 0'), gathers[1], stages.find(s => s.name === 'DiscContacts apply 1')];
    }
    let missed = 0, checks = 0;
    const prefix = [];
    for (const s of stages) {
        prefix.push(s);
        if (s.name.includes('gather')) {
            run([...prefix.splice(0), stage]);
            const expected = new Uint32Array(await read(oracle)), actual = new Float32Array(await read(c.deltas));
            for (let i = 0; i < count; i++) { missed += Math.max(0, expected[i] - actual[i * 4 + 2]); checks += expected[i]; }
        }
    }
    check(below ? 'A38 watched failing: below-bound cells miss contacts' : 'A38 complete at the cell bound in both passes', below ? missed > 0 : missed === 0, `${missed} missed / ${checks} contacts`);
    for (const s of stages) if (s.testOwned) { s.enabled = false; s.testOwned.forEach(b => b.destroy()); }
    stage.enabled = false; extraGrid?.destroy(); c.destroy(); b.destroy(); oracle.destroy();
}
await completeness(1.45);
await completeness(0.6, true);

// A39: exact reruns using the 13.3 addStepStage seam.
async function rerun() {
    const count = 16384;
    const def = A.defineAgents({ capacity: count, world: base.world, cellSize: 8,
        species: [{ name: 'body', radius: 0.4, maxSpeed: 1, maxTurn: 2, accel: 2, senseRadius: 8 }], sensors: [A.Sensors.self()] });
    const agents = new A.AgentSystem(engine, def);
    agents.setBrain({ kind: 'stage', wgsl: 'fn think(i: u32, species: u32, tick: u32) { setAct(i, ACT_TURN, 0.2); setAct(i, ACT_THROTTLE, 0.5); }', buffers: [], destroy() {} });
    for (let i = 0; i < count; i++) agents.spawn(i, { position: [(i % 128 - 64) * 0.45, 0, (Math.floor(i / 128) - 64) * 0.45], species: 'body', heading: i * 0.123 });
    agents.upload();
    const c = new P.DiscContacts(engine, agents.buffer, { ...base, ...layout(count), clamp: true });
    agents.addStepStage({ name: 'contacts', stages: c.getComputeStages() });
    agents.advance(def.dt);
    for (let k = 0; k < 600; k += 10) { run(agents.getComputeStages(), 10); await d.queue.onSubmittedWorkDone(); }
    const words = new Uint32Array(await read(agents.buffer));
    c.destroy(); agents.destroy(); return words;
}
check('A39 two 600-step runs of 16,384 steering bodies are word-identical', equal(await rerun(), await rerun()));

// Watched-failing negative control: deliberately race position writes against neighbour reads.
{
    const count = 4096, rec = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) { rec[i * 4] = (i % 64) * 0.02; rec[i * 4 + 2] = Math.floor(i / 64) * 0.02; rec[i * 4 + 3] = 0.4; }
    const b = buffer(rec), c = new P.DiscContacts(engine, b, { ...base, capacity: count, stride: 4, iterations: 1, maxScanPerCell: 128 });
    const params = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(params, 0, new Float32Array([1.75, 0, 0.25, 1]));
    const tick = buffer(new Uint32Array(4));
    const code = P.buildContactGatherWGSL(c.options, 0)
        .replace('var<storage, read> records', 'var<storage, read_write> records')
        .replace('deltas[i] = vec4<f32>(sum, f32(count), 0.0);', `
        deltas[i] = vec4<f32>(sum, f32(count), 0.0);
        if (count > 0u) {
            var u = sum * (params.x / sqrt(f32(count)));
            let len = length(u); if (len > params.z) { u *= params.z / len; }
            records[i * 4u] += u.x; records[i * 4u + 2u] += u.y;
        }`);
    const s = engine.createComputeStage({ name: 'A39 intentionally racing gather', shaderCode: code, dispatchCount: [count / 64, 1, 1] });
    engine.setComputePassBindGroup(s, [b, c.grid.cellOffsets, c.grid.cellBodies, c.deltas, params, tick]
        .map((buffer, binding) => ({ binding, resource: { buffer } })), 0);
    const chain = [...c.grid.getComputeStages(), s];
    const samples = [];
    for (let attempt = 0; attempt < 4; attempt++) {
        d.queue.writeBuffer(b, 0, rec); run(chain, 10); samples.push(new Uint32Array(await read(b)));
    }
    check('A39 watched failing: in-place gather produces differing reruns', samples.slice(1).some(words => !equal(words, samples[0])));
    s.enabled = false; c.destroy(); b.destroy(); params.destroy(); tick.destroy();
}

// A40: actual GPU timestamps (not wall time or shader compilation), bounded extreme pile.
{
    const count = 65536, rec = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) rec[i * 4 + 3] = 0.4;
    const b = buffer(rec), c = new P.DiscContacts(engine, b, { ...base, capacity: count, stride: 4 });
    run(c.getComputeStages()); await d.queue.onSubmittedWorkDone();
    d.queue.writeBuffer(b, 0, rec);
    const enc = d.createCommandEncoder();
    let queries, resolved;
    if (d.features.has('timestamp-query')) {
        queries = d.createQuerySet({ type: 'timestamp', count: 2 });
        resolved = d.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    }
    encode(enc, c.getComputeStages(), 1, queries ? { querySet: queries, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } : undefined);
    if (queries) enc.resolveQuerySet(queries, 0, 2, resolved, 0);
    d.queue.submit([enc.finish()]);
    const counters = await c.readCounters();
    check('A40 65,536 coincident bodies: skipped entries counted once per build', counters.overflow === count - 32);
    if (queries) {
        const times = new BigUint64Array(await read(resolved));
        const ms = Number(times[1] - times[0]) / 1e6;
        check('A40 extreme pile step below 100 ms', ms < 100, `${ms.toFixed(3)} ms`);
        queries.destroy(); resolved.destroy();
    } else check('A40 timestamp-query available for timeout target', false);
    for (let k = 0; k < 300; k += 5) { run(c.getComputeStages(), 5); await d.queue.onSubmittedWorkDone(); }
    const actual = new Float32Array(await read(b));
    let moved = 0, finite = true;
    for (let i = 0; i < count; i++) { moved += Math.hypot(actual[i * 4], actual[i * 4 + 2]) > 0.4; finite &&= Number.isFinite(actual[i * 4]) && Number.isFinite(actual[i * 4 + 2]); }
    check('A40 bounded pile opens over 300 steps without nonfinite positions', finite && moved > count * 0.5, `${moved}/${count} moved beyond a radius (not a claim of full separation)`);
    c.destroy(); b.destroy();
}

// A41: inside an AgentSystem: contacts as an app-stage group.
{
    const count = 256;
    const def = A.defineAgents({
        capacity: count,
        world: { min: [-20, -20], max: [20, 20] },
        cellSize: 4,
        species: [{ name: 'prey', radius: 0.5, maxSpeed: 2, maxTurn: 3, accel: 3, senseRadius: 4 }],
        sensors: [A.Sensors.self()],
    });

    function setup(useContacts) {
        const sys = new A.AgentSystem(engine, def);
        sys.setBrain({
            kind: 'stage',
            wgsl: 'fn think(i: u32, species: u32, tick: u32) { setAct(i, ACT_TURN, 0.2); setAct(i, ACT_THROTTLE, 0.8); }',
            buffers: [],
            destroy() {},
        });
        for (let i = 0; i < count; i++) {
            if (i === 0) {
                sys.spawn(i, { position: [0, 0, 0], species: 'prey', heading: 0, frozen: true });
            } else {
                sys.spawn(i, {
                    position: [(i % 16 - 8) * 0.35, 0, (Math.floor(i / 16) - 8) * 0.35],
                    species: 'prey',
                    heading: i * 0.15,
                });
            }
        }
        sys.upload();
        let c;
        if (useContacts) {
            c = new P.DiscContacts(engine, sys.buffer, {
                capacity: count,
                stride: 16,
                world: { min: [-20, -20], max: [20, 20] },
                maxRadius: 0.6,
                maxPush: 0.25,
                alive: { offset: 13, mask: 1 },
                fixed: { offset: 13, mask: 2 },
                velocity: { offset: 4 },
                clamp: true,
            });
            sys.addStepStage({ name: 'contacts', stages: c.getComputeStages() });
        }
        return { sys, c };
    }

    function overlap(rec) {
        let sum = 0;
        for (let i = 0; i < count; i++) {
            const xi = rec[i * 16], zi = rec[i * 16 + 2], ri = rec[i * 16 + 3];
            for (let j = i + 1; j < count; j++) {
                const xj = rec[j * 16], zj = rec[j * 16 + 2], rj = rec[j * 16 + 3];
                const d = Math.hypot(xi - xj, zi - zj);
                if (d < ri + rj) sum += (ri + rj - d);
            }
        }
        return sum;
    }

    // Run without contacts for 60 steps
    const { sys: sNo } = setup(false);
    sNo.advance(def.dt);
    for (let k = 0; k < 60; k++) run(sNo.getComputeStages());
    await d.queue.onSubmittedWorkDone();
    const noContactRec = new Float32Array(await read(sNo.buffer));
    const ovNo = overlap(noContactRec);
    sNo.destroy();

    // Mode A: 1 step per frame for 60 frames (60 steps)
    const { sys: s1, c: c1 } = setup(true);
    for (let k = 0; k < 60; k++) { s1.advance(def.dt); run(s1.getComputeStages()); }
    await d.queue.onSubmittedWorkDone();
    const words1 = new Uint32Array(await read(s1.buffer));
    const rec1 = new Float32Array(words1.buffer);
    const ovWith = overlap(rec1);

    // Mode B: 3 steps per frame for 20 frames (60 steps)
    const { sys: s3, c: c3 } = setup(true);
    for (let k = 0; k < 20; k++) { s3.advance(3 * def.dt); run(s3.getComputeStages()); }
    await d.queue.onSubmittedWorkDone();
    const words3 = new Uint32Array(await read(s3.buffer));

    // Mode C: await sys.step() for 60 steps
    const { sys: sStep, c: cStep } = setup(true);
    for (let k = 0; k < 60; k++) await sStep.step();
    const wordsStep = new Uint32Array(await read(sStep.buffer));

    // (a) Cadence equivalence
    check('A41 1 step/frame, 3 steps/frame and step() give identical records',
        equal(words1, words3) && equal(words1, wordsStep));

    // (b) Overlap falls with contacts vs without
    check('A41 overlap falls significantly with contacts vs without',
        ovWith < ovNo * 0.1,
        `overlap with: ${ovWith.toFixed(2)}, without: ${ovNo.toFixed(2)}`);

    // (c) Frozen agent stays put bit-for-bit
    check('A41 frozen agent stays put bit-for-bit',
        words1[0] === 0 && words1[1] === 0 && words1[2] === 0,
        `pos: [${rec1[0]}, ${rec1[1]}, ${rec1[2]}]`);

    s1.destroy(); c1.destroy();
    s3.destroy(); c3.destroy();
    sStep.destroy(); cStep.destroy();
}
await env.finish('Physics GPU checks passed.');