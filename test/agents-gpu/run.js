// A1 kernel slice. Runs on both Node/Dawn and Chrome through the shared GPU harness.
import * as A from '../../dist/agents.js';
import { NullGraph, Camera } from '../../dist/index.js';
import { groundGrid, gridOccupancy, agentArrows } from '../../dist/debug-ui.js';
import { VisibilitySystem } from '../../dist/visibility.js';
import { MegabufferBuilder, Primitives, StandardLayout } from '../../dist/geometry.js';
import { Picker } from '../../dist/picking.js';
import { FontAtlas, TextSystem } from '../../dist/text.js';
import { syntheticFont, SHAPES } from '../gpu/synthetic-font.mjs';
import { harnessEnv } from '../gpu/harness-env.mjs';

const { check, fatal, finish, saveImage } = harnessEnv('__AGENTS_RESULTS__');
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) { fatal('No WebGPU adapter'); throw new Error('No WebGPU adapter'); }
const device = await adapter.requestDevice();
const allocated = [];
function buffer(words, data) {
    const b = device.createBuffer({ size: Math.max(4, words * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    allocated.push(b);
    if (data) device.queue.writeBuffer(b, 0, data);
    return b;
}
async function read(b, words) {
    const dst = device.createBuffer({ size: words * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(b, 0, dst, 0, words * 4);
    device.queue.submit([enc.finish()]);
    await dst.mapAsync(GPUMapMode.READ);
    const result = new Uint32Array(dst.getMappedRange().slice(0));
    dst.unmap(); dst.destroy();
    return result;
}
function dispatch(enc, code, buffers, groups) {
    const pipeline = device.createComputePipeline({ label: 'A1 grid kernel', layout: 'auto',
        compute: { module: device.createShaderModule({ code }), entryPoint: 'cs_main' } });
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
        entries: buffers.map((b, binding) => ({ binding, resource: { buffer: b } })) });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(groups); pass.end();
}
function scan(enc, input, count) {
    const offsets = buffer(count + 1), blocks = Math.ceil(count / A.GRID_SCAN_BLOCK);
    const totals = buffer(blocks);
    dispatch(enc, A.buildGridScanWGSL(count), [input, offsets, totals], blocks);
    if (blocks > 1) {
        const blockOffsets = scan(enc, totals, blocks);
        dispatch(enc, A.buildGridScanAddWGSL(count), [offsets, blockOffsets], Math.ceil((count + 1) / 64));
    }
    return offsets;
}
const equal = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
let seed = 0x12345678;
const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };

// Partial blocks, exact block boundaries and recursive block-total scan.
for (const n of [1, 255, 256, 257, 1025, 65537]) {
    const counts = Uint32Array.from({ length: n }, () => Math.floor(random() * 64));
    const enc = device.createCommandEncoder(), offsets = scan(enc, buffer(n, counts), n);
    device.queue.submit([enc.finish()]);
    check(`U2 GPU prefix ${n}`, equal(await read(offsets, n + 1), A.prefixOffsets(counts)));
    allocated.splice(0).forEach(b => b.destroy());
}

for (const layout of ['uniform', 'boundaries', 'ties', 'crowded']) {
    const def = A.defineAgents({ capacity: 10000, world: { min: [-100, -100], max: [100, 100] }, cellSize: 4,
        maxScanPerCell: layout === 'crowded' ? 2 : 10000,
        species: ['prey', 'wolf'].map(name => ({ name, maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 4, radius: 0.4 })),
        sensors: [A.Sensors.nearest({ k: 8 })] });
    const rec = new A.AgentRecords(def);
    for (let i = 0; i < def.capacity; i++) {
        let x = -100 + random() * 200, z = -100 + random() * 200;
        if (layout === 'boundaries') { x = -100 + (i % 51) * 4; z = -100 + (Math.floor(i / 51) % 51) * 4; }
        if (layout === 'ties') { x = i % 5 - 2; z = Math.floor(i / 5) * 8 - 100; }
        if (layout === 'crowded') { x = 1; z = 1; }
        rec.spawn(i, { position: [x, 0, z], species: i % 2 });
        if (i % 17 === 0) rec.kill(i);
    }
    const cpu = A.buildPointGrid(def, rec.data, A.AGENT_FLOATS, def.capacity, i => rec.isAlive(i), def.maxScanPerCell);
    const records = buffer(rec.data.length, rec.data), counts = buffer(def.grid.cellTotal);
    const cursors = buffer(def.grid.cellTotal), bodies = buffer(def.capacity), overflow = buffer(1);
    const enc = device.createCommandEncoder();
    dispatch(enc, A.buildGridBinWGSL(def, def.capacity, 16, 13, 1, def.maxScanPerCell), [records, counts, overflow], Math.ceil(def.capacity / 64));
    const offsets = scan(enc, counts, def.grid.cellTotal);
    dispatch(enc, A.buildGridScatterWGSL(def, def.capacity, 16, 13, 1), [records, offsets, cursors, bodies], Math.ceil(def.capacity / 64));
    const queries = [1, 2, 3].map(mask => {
        const result = buffer(def.capacity * 8);
        dispatch(enc, A.buildAgentGridQueryWGSL(def, mask, 4, 8), [records, counts, offsets, bodies, result], Math.ceil(def.capacity / 64));
        return { mask, result };
    });
    device.queue.submit([enc.finish()]);
    const gpuCounts = await read(counts, def.grid.cellTotal), gpuOffsets = await read(offsets, def.grid.cellTotal + 1);
    const gpuBodies = await read(bodies, def.capacity), gpuOverflow = (await read(overflow, 1))[0];
    check(`A1 ${layout}: counts, offsets and overflow`, equal(gpuCounts, cpu.cellCounts) && equal(gpuOffsets, cpu.cellOffsets) && gpuOverflow === cpu.overflow);
    let membership = true;
    for (let c = 0; c < def.grid.cellTotal; c++) {
        const start = cpu.cellOffsets[c], end = cpu.cellOffsets[c + 1];
        if (!equal(gpuBodies.slice(start, end).sort(), cpu.cellBodies.slice(start, end).sort())) membership = false;
    }
    check(`A1 ${layout}: cell membership sets, all live indices retained`, membership);
    // Overflow is intentionally approximate; compare against the actual GPU scatter order.
    const mirror = { cellCounts: gpuCounts, cellOffsets: gpuOffsets, cellBodies: gpuBodies, overflow: gpuOverflow };
    for (const { mask, result } of queries) {
        const actual = await read(result, def.capacity * 8);
        let matches = true;
        for (let i = 0; i < def.capacity; i++) {
            const expected = [];
            const n = A.gridNearestNeighbours(def, rec, mirror, i, mask, 4, 8, expected);
            for (let j = 0; j < 8; j++) if (actual[i * 8 + j] !== (j < n ? expected[j] : 0xffffffff)) matches = false;
        }
        check(`A1 ${layout}: bounded nearest, species mask ${mask}`, matches);
    }
    allocated.splice(0).forEach(b => b.destroy());
}
// Production wrapper: a stable prebuilt stage sequence, two builds in one submission and
// rebuild after killing every record. The caller-owned records must survive grid teardown.
const engine = new NullGraph();
const init = await engine.init(null, { device, width: 128, height: 128 });
if (!init.success) { fatal(init.error); throw new Error(init.error); }
const def = A.defineAgents({ capacity: 10000, world: { min: [-8, -8], max: [8, 8] }, cellSize: 4,
    maxScanPerCell: 2, species: [{ name: 'prey', maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 4, radius: 0.4 }],
    sensors: [A.Sensors.nearest({ k: 8 })] });
const rec = new A.AgentRecords(def);
for (let i = 0; i < def.capacity; i++) rec.spawn(i, { position: [2, 0, 2], species: 0 });
const records = buffer(rec.data.length, rec.data);
const grid = new A.AgentGrid(engine, records, def, 1, 4, 8);
let badRadius = false, badBuffer = false, badLayout = false;
try { new A.AgentGrid(engine, records, def, 1, 5, 8); } catch { badRadius = true; }
try { new A.PointGrid(engine, grid.neighbours, { ...def, stride: 160 }); } catch { badBuffer = true; }
try { new A.PointGrid(engine, records, { ...def, stride: 16, cellSize: 0 }); } catch { badLayout = true; }
check('wrapper: invalid query, record capacity and layout refused', badRadius && badBuffer && badLayout);
const stages = grid.getComputeStages();
check('wrapper: stage array is stable and no queue-side clears', stages === grid.getComputeStages() && stages.every(s => s.preClearBuffers.length === 0));
engine.render([...stages, ...stages]);
check('wrapper: two builds in one submission reset counts/overflow',
    (await read(grid.cellCounts, 16))[10] === 10000 && (await read(grid.overflow, 1))[0] === 9998);
const counts = await read(grid.cellCounts, 16), offsets = await read(grid.cellOffsets, 17), bodies = await read(grid.cellBodies, def.capacity);
const actual = await read(grid.neighbours, def.capacity * 8);
let matches = true;
for (let i = 0; i < def.capacity; i++) {
    const expected = [];
    const n = A.gridNearestNeighbours(def, rec, { cellCounts: counts, cellOffsets: offsets, cellBodies: bodies }, i, 1, 4, 8, expected);
    for (let j = 0; j < 8; j++) if (actual[i * 8 + j] !== (j < n ? expected[j] : 0xffffffff)) matches = false;
}
check('wrapper: query buffer equals CPU bounded oracle for 10000 agents', matches);

const camera = new Camera(60, 1, 0.1, 100);
camera.updateView([0, 18, 0], [0, 0, 0], [0, 0, -1]);
engine.updateCamera(camera);
const floorPass = engine.createPass({ name: 'GPU floor debug', isMainScreenPass: true });
const floor = groundGrid(engine, floorPass, { min: [-8, -8], max: [8, 8] });
const overlayPass = engine.createPass({ name: 'GPU occupancy debug', isMainScreenPass: true });
const overlay = gridOccupancy(engine, overlayPass, grid);
async function image() {
    const dst = device.createBuffer({ size: 128 * 512, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: engine.screenTexture }, { buffer: dst, bytesPerRow: 512 }, [128, 128]);
    device.queue.submit([enc.finish()]);
    await dst.mapAsync(GPUMapMode.READ);
    const pixels = new Uint8Array(dst.getMappedRange().slice(0));
    dst.unmap(); dst.destroy();
    return pixels;
}
engine.render([floorPass]);
const floorPixels = await image();
await saveImage('agents-ground-grid', floorPixels, 128, 128);
check('groundGrid: anti-aliased floor renders a nonuniform frame', new Set(floorPixels.filter((_, i) => i % 4 === 0)).size > 4);
engine.render([overlayPass]);
const redPixels = await image();
await saveImage('agents-occupancy-overflow', redPixels, 128, 128);
const red = pixels => pixels.filter((v, i) => i % 4 === 0 && v > 150 && pixels[i + 1] < 60).length;
check('gridOccupancy: overflow cell draws red directly from GPU counts', red(redPixels) > 50);
for (let i = 0; i < def.capacity; i++) rec.kill(i);
device.queue.writeBuffer(records, 0, rec.data);
engine.render([...stages, overlayPass]);
check('wrapper: empty rebuild clears counts, terminal total, overflow and query padding',
    (await read(grid.cellCounts, 16)).every(v => v === 0) && (await read(grid.cellOffsets, 17)).every(v => v === 0) &&
    (await read(grid.overflow, 1))[0] === 0 && (await read(grid.neighbours, def.capacity * 8)).every(v => v === 0xffffffff));
check('gridOccupancy: empty rebuild removes red without CPU overlay upload', red(await image()) === 0);
floor.destroy(); floor.destroy(); overlay.destroy(); overlay.destroy(); grid.destroy(); grid.destroy();
let refused = false;
try { grid.getComputeStages(); } catch { refused = true; }
check('wrapper: teardown disables cached stages and refuses reuse', refused && stages.every(s => !s.enabled) && floorPass.batches.length === 0 && overlayPass.batches.length === 0);
check('wrapper: caller-owned records survive teardown', (await read(records, rec.data.length)).length === rec.data.length);
records.destroy();
// Phase 3.1 A13: a test-only brain seam before tick. It observes tick and moves records so
// subsequent grid/query builds must consume the preceding step, not only count dispatches.
const loopDef = A.defineAgents({ capacity: 128, world: { min: [-8, -8], max: [8, 8] }, cellSize: 4,
    maxScanPerCell: 128,
    species: ['prey', 'wolf'].map(name => ({ name, maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 4, radius: 0.4 })),
    sensors: [A.Sensors.nearest({ name: 'wolf', species: ['wolf'], k: 1 }), A.Sensors.nearest({ name: 'prey', species: ['prey'], k: 3 })] });
// Earlier grid/sense fixtures isolate those stages; phase-3.3 fixtures below run act normally.
function senseOnly(s) {
    s.advance(s.definition.dt);
    s.getComputeStages().find(stage => stage.name === 'AgentSystem act').enabled = false;
    return s;
}
function system(definition = loopDef) {
    const s = senseOnly(makeSystem(engine, definition));
    for (let i = 0; i < s.capacity; i++) s.spawn(i, { position: [-6 + (i % 16) * 0.75, 0, -3 + Math.floor(i / 16) * 0.75], species: i % 2 });
    s.upload();
    return s;
}
function stub(s, cpuTick = false) {
    const uniform = cpuTick ? device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }) : null;
    const code = `${A.AGENT_STRUCT_WGSL}${A.AGENT_STATE_WGSL}
@group(0) @binding(0) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(1) var<storage, read> state: AgentState;
@group(0) @binding(2) var<storage, read_write> act: array<f32>;
${cpuTick ? '@group(0) @binding(3) var<uniform> cpuState: vec4<u32>;' : ''}
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= ${s.capacity}u) { return; }
    let tick = ${cpuTick ? 'cpuState.x + (state.tick & 0u)' : 'state.tick'};
    act[i * 2u] = act[i * 2u] * 4.0 + f32(tick + 1u);
    agents[i].posRadius.x += f32(tick + 1u) * 0.125;
}`;
    const stage = engine.createComputeStage({ name: 'A13 test-only tick reader', shaderCode: code, dispatchCount: [2, 1, 1] });
    engine.setComputePassBindGroup(stage, [s.buffer, s.state, s.act, ...(uniform ? [uniform] : [])].map((buffer, binding) => ({ binding, resource: { buffer } })), 0);
    if (uniform) stage.preClearBuffers.push({ buffer: uniform, offset: 0, data: new Uint32Array([0, 0, 0, 0]) });
    s.advance(loopDef.dt);
    const one = s.getComputeStages();
    return { stage, uniform, chain: [...one.slice(0, -1), stage, one.at(-1)] };
}
const batch = system(), single = system(), broken = system();
const batchStub = stub(batch), singleStub = stub(single), brokenStub = stub(broken, true);
engine.render([...batchStub.chain, ...batchStub.chain, ...batchStub.chain]);
engine.render([...brokenStub.chain, ...brokenStub.chain, ...brokenStub.chain]);
for (let n = 0; n < 3; n++) { engine.render(singleStub.chain); await device.queue.onSubmittedWorkDone(); }
for (const [name, words] of [['buffer', 128 * 16], ['act', 128 * 2], ['neighbours', 128 * 4], ['obs', 128 * loopDef.obsDim], ['state', 4]]) {
    check(`A13: three chain copies equal three submissions (${name})`, equal(await read(batch[name], words), await read(single[name], words)));
}
check('A13 watched failing: submission-wide CPU tick is detected', !equal(await read(batch.act, 256), await read(broken.act, 256)));
const loopAct = new Float32Array((await read(batch.act, 256)).buffer);
check('A13: stub observed ticks 0,1,2 in order', loopAct.every((v, i) => i % 2 ? v === 0 : v === 27));
check('A13: tick state increments once per step, no overflow', equal(await read(batch.state, 4), new Uint32Array([3, 0, 0, 0])));
check('A13: grid counts and offsets agree across submissions',
    equal(await read(batch.grid.cellCounts, 16), await read(single.grid.cellCounts, 16)) &&
    equal(await read(batch.grid.cellOffsets, 17), await read(single.grid.cellOffsets, 17)));
// Verify both sensor slices against a fresh CPU record image, reflecting positions before the
// third stub ran (the queries are before it in the chain).
const gpuRecords = await read(batch.buffer, 128 * 16);
const queryRecords = new A.AgentRecords(loopDef, new Float32Array(gpuRecords.buffer));
for (let i = 0; i < 128; i++) queryRecords.data[i * 16] -= 3 * 0.125;
const shared = await read(batch.neighbours, 128 * 4);
let slicesMatch = true;
for (let i = 0; i < 128; i++) {
    for (const [mask, k, base] of [[2, 1, 0], [1, 3, 1]]) {
        const out = [], n = A.nearestNeighbours(loopDef, queryRecords, i, mask, 4, k, out);
        for (let j = 0; j < k; j++) if (shared[i * 4 + base + j] !== (j < n ? out[j] : 0xffffffff)) slicesMatch = false;
    }
}
check('U16 GPU: both shared query slices equal the CPU nearest oracle', slicesMatch);
const crowdedDef = A.defineAgents({ capacity: 8, world: { min: [-8, -8], max: [8, 8] }, cellSize: 4,
    maxScanPerCell: 2, species: [loopDef.species[0]], sensors: [A.Sensors.nearest({ k: 1 })] });
const crowded = makeSystem(engine, crowdedDef);
for (let i = 0; i < 8; i++) crowded.spawn(i, { position: [1, 0, 1], species: 0 });
crowded.upload(); crowded.advance(3 * crowdedDef.dt); engine.render(crowded.getComputeStages());
check('tick: overflow is folded once per build in a repeated chain', equal(await read(crowded.state, 4), new Uint32Array([3, 3, 18, 0])));
await crowded.step();
check('step: lockstep submits exactly one step', equal(await read(crowded.state, 4), new Uint32Array([4, 4, 24, 0])));
for (let i = 0; i < 8; i++) crowded.kill(i);
await crowded.step();
check('tick: empty rebuild preserves run-long overflow counters', equal(await read(crowded.state, 4), new Uint32Array([5, 4, 24, 0])));
check('upload: killed records leave query rows padded', (await read(crowded.neighbours, 8)).every(v => v === 0xffffffff));
const cached = crowded.getComputeStages();
for (const item of [batchStub, singleStub, brokenStub]) { item.stage.enabled = false; item.uniform?.destroy(); }
for (const s of [batch, single, broken, crowded]) { s.destroy(); s.destroy(); }
engine.render(cached);
check('destroy: cached skeleton stages are disabled', cached.every(stage => !stage.enabled));

// A2: 10,000 observers, all built-ins, two species masks, different observer radii and speeds.
const extraSensor = A.Sensors.custom({ name: 'extra', width: 2,
    wgsl: 'fn extra(i: u32, a: Agent) -> array<f32, 2> { return array<f32, 2>(a.state.w, f32(i % 7u)); }',
    mirror(d, u, i, out, at) { out[at] = d[i * 16 + 11]; out[at + 1] = i % 7; } });
const senseDef = A.defineAgents({ capacity: 10000, world: { min: [-100, -100], max: [100, 100] }, cellSize: 4,
    maxScanPerCell: 64,
    species: [{ name: 'prey', maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 2, radius: 0.4 },
        { name: 'wolf', maxSpeed: 5, maxTurn: 4, accel: 6, senseRadius: 4, radius: 0.4 }],
    sensors: [A.Sensors.self(), A.Sensors.nearest({ species: 'wolf', k: 1 }),
        A.Sensors.nearest({ species: 'prey', k: 3 }), A.Sensors.bounds(), extraSensor] });
const sensing = senseOnly(makeSystem(engine, senseDef));
for (let i = 0; i < senseDef.capacity; i++) {
    sensing.spawn(i, { position: [-99.9 + random() * 199.8, 0, -99.9 + random() * 199.8], species: i % 2,
        heading: -Math.PI + random() * 2 * Math.PI, speed: random() * (i % 2 ? 5 : 3),
        speedScale: 0.5 + random() * 0.5, energy: random() * 4 - 2, user0: random() * 2 - 1,
        frozen: i % 101 === 0, age: 1 });
    if (i % 97 === 0) sensing.kill(i);
}
// Skip/count random radius-boundary observers as required by §6.3. Use CPU cell membership
// for the boundary scan only; the observation oracle remains the brute-force senseAll mirror.
function boundaryObservers(def, rec) {
    const grid = A.buildPointGrid(def, rec.data, 16, def.capacity, i => rec.isAlive(i), def.maxScanPerCell);
    const skip = new Set();
    for (let i = 0; i < def.capacity; i++) {
        if (!rec.isAlive(i)) continue;
        const x = rec.data[i * 16], z = rec.data[i * 16 + 2], c = A.pointCell(def, x, z);
        const cx = c % def.grid.cellsX, cz = Math.floor(c / def.grid.cellsX);
        const radius = def.species[rec.u32[i * 16 + 12]].senseRadius;
        for (let zz = Math.max(0, cz - 1); zz <= Math.min(def.grid.cellsZ - 1, cz + 1); zz++) {
            for (let xx = Math.max(0, cx - 1); xx <= Math.min(def.grid.cellsX - 1, cx + 1); xx++) {
                const cell = zz * def.grid.cellsX + xx;
                for (let at = grid.cellOffsets[cell]; at < grid.cellOffsets[cell + 1]; at++) {
                    const j = grid.cellBodies[at];
                    if (j !== i && Math.abs(Math.hypot(rec.data[j * 16] - x, rec.data[j * 16 + 2] - z) - radius) < 1e-4) skip.add(i);
                }
            }
        }
    }
    return skip;
}
function compareSense(actual, expected, def, rec, skip = new Set()) {
    let maximum = 0, compared = 0, badPresence = 0;
    for (let i = 0; i < def.capacity; i++) {
        if (!rec.isAlive(i) || skip.has(i)) continue;
        compared++;
        for (let k = 0; k < def.obsDim; k++) {
            const error = Math.abs(actual[i * def.obsDim + k] - expected[i * def.obsDim + k]);
            if (!Number.isFinite(error)) return { ok: false, maximum: Infinity, compared, badPresence };
            maximum = Math.max(maximum, error);
            if (def.obsNames[k].endsWith('.present') && error !== 0) badPresence++;
        }
    }
    return { ok: maximum <= 7e-4 && badPresence === 0, maximum, compared, badPresence };
}
for (const tuned of [false, true]) {
    if (tuned) sensing.setSpecies('prey', { senseRadius: 3, maxSpeed: 4 });
    await sensing.step();
    const current = sensing.definition;
    const skip = boundaryObservers(current, sensing.records);
    const expected = A.senseAll(current, sensing.records);
    const actual = new Float32Array((await read(sensing.obs, current.capacity * current.obsDim)).buffer);
    const result = compareSense(actual, expected, current, sensing.records, skip);
    console.log(`A2 ${tuned ? 'tuned' : 'initial'}: maximum ${result.maximum}, compared ${result.compared}, boundary observers skipped ${skip.size}`);
    check(`A2 ${tuned ? 'tuned' : 'initial'}: GPU observations match CPU within 7e-4, presence exact`, result.ok && result.compared > 9800);
    check(`A2 ${tuned ? 'tuned' : 'initial'}: scene is overflow-free`, (await read(sensing.state, 4))[1] === 0);
}
// Compile the actual generated sense kernel with a flipped side basis: the oracle must reject it.
const flippedCode = A.buildAgentSenseWGSL(sensing.definition).replace('vec2<f32>(cs, -sn)', 'vec2<f32>(-cs, sn)');
const flippedEncoder = device.createCommandEncoder();
dispatch(flippedEncoder, flippedCode, [sensing.buffer, sensing.neighbours, sensing.state, sensing.obs, sensing.species], Math.ceil(senseDef.capacity / 64));
device.queue.submit([flippedEncoder.finish()]);
const flipped = new Float32Array((await read(sensing.obs, senseDef.capacity * senseDef.obsDim)).buffer);
check('A2 watched failing: flipped side axis is rejected', !compareSense(flipped, A.senseAll(sensing.definition, sensing.records),
    sensing.definition, sensing.records, boundaryObservers(sensing.definition, sensing.records)).ok);
sensing.destroy();

// Exact radius, absent/padded ranks, rotated axes, wall ties and speedScale > 1 are explicit fixtures.
const fixtureDef = A.defineAgents({ capacity: 6, world: { min: [-8, -8], max: [8, 8] }, cellSize: 4,
    species: [loopDef.species[0]], sensors: [A.Sensors.self(), A.Sensors.nearest({ k: 3 }), A.Sensors.bounds(), extraSensor] });
const fixture = senseOnly(makeSystem(engine, fixtureDef));
for (const [i, position, heading] of [[0, [0, 0, 0], 0], [1, [4, 0, 0], Math.PI / 2],
    [2, [0, 0, 4], -Math.PI / 2], [3, [-8, 0, -8], Math.PI], [4, [8, 0, 8], -Math.PI]]) {
    fixture.spawn(i, { position, heading, species: 0, speedScale: 2, speed: 6, age: 1 });
}
await fixture.step();
const fixtureObs = new Float32Array((await read(fixture.obs, 6 * fixtureDef.obsDim)).buffer);
check('A2 fixtures: exact radius, heading quadrants, corner ties, empty ranks and scaled speed',
    compareSense(fixtureObs, A.senseAll(fixtureDef, fixture.records), fixtureDef, fixture.records).ok && fixtureObs[6] === 1);
fixture.destroy();

// No act stage exists yet: age is supplied explicitly to exercise non-newborn cadence.
const cadenceDef = A.defineAgents({ ...loopDef, world: { min: [-8, -8], max: [8, 8] }, sensors: [A.Sensors.self()] });
const cadence = senseOnly(makeSystem(engine, cadenceDef, { thinkEvery: 3 }));
for (let i = 0; i < 128; i++) cadence.spawn(i, { position: [0, 0, 0], species: 0, age: 1, energy: 0.25 });
cadence.spawn(65, { position: [0, 0, 0], species: 0, age: 0, energy: 0.75 });
await cadence.step();
let cadenceObs = new Float32Array((await read(cadence.obs, 256)).buffer);
check('sense cadence: block 0 runs at tick 0; newborn in block 1 runs immediately',
    cadenceObs[1] === 0.25 && cadenceObs[129] === 0 && cadenceObs[131] === 0.75);
cadence.spawn(0, { position: [0, 0, 0], species: 0, age: 1, energy: 0.5 });
cadence.kill(65);
await cadence.step();
cadenceObs = new Float32Array((await read(cadence.obs, 256)).buffer);
check('sense cadence: off-cadence and dead rows retain their previous values', cadenceObs[1] === 0.25 && cadenceObs[131] === 0.75);
await cadence.step();
cadenceObs = new Float32Array((await read(cadence.obs, 256)).buffer);
check('sense cadence: block 1 runs at tick 2, block 0 retains its row', cadenceObs[129] === 0.25 && cadenceObs[1] === 0.25);
await cadence.step();
check('sense cadence: block 0 refreshes at tick 3', new Float32Array((await read(cadence.obs, 256)).buffer)[1] === 0.5);
cadence.destroy();
// These layouts must retain all auto-layout bindings even without any neighbour/custom use.
for (const sensors of [[A.Sensors.bounds()], [extraSensor]]) {
    const sparseDef = A.defineAgents({ ...fixtureDef, world: { min: [-8, -8], max: [8, 8] }, sensors });
    const sparse = makeSystem(engine, sparseDef);
    sparse.spawn(0, { position: [0, 0, 0], species: 0 });
    await sparse.step();
    check(`sense sparse: ${sensors[0]?.kind ?? 'empty'} layout submits with all bindings`, (await read(sparse.state, 4))[0] === 1);
    sparse.destroy();
}
// A3: CPU stepAll is the definition. Radius/wall and wrapped-heading fixtures accompany the
// 10,000-agent scene. Bounds include trig error and f32 output rounding, not guessed epsilons.
const actDef = A.defineAgents({ capacity: 10000, world: { min: [-20, -20], max: [20, 20] }, cellSize: 4,
    maxScanPerCell: 10000, dt: 1 / 60,
    species: loopDef.species, sensors: [A.Sensors.self()] });
const acting = makeSystem(engine, actDef);
const actions = new Float32Array(actDef.capacity * 2);
for (let i = 0; i < actDef.capacity; i++) {
    acting.spawn(i, { position: [-19.6 + random() * 39.2, random(), -19.6 + random() * 39.2], species: i % 2,
        heading: -Math.PI + random() * 2 * Math.PI, speed: random() * 3, speedScale: 0.5 + random() * 1.5,
        age: random() * 5, frozen: i % 101 === 0, user0: random(), user1: i });
    if (i % 97 === 0) acting.kill(i);
    actions[i * 2] = random() * 4 - 2;
    actions[i * 2 + 1] = random() * 4 - 2;
}
for (const [i, position, heading, radius] of [[1, [19.59, 2, 0], Math.PI / 2, 0.4],
    [2, [-19.59, 2, 0], -Math.PI / 2, 0.4], [3, [0, 2, 19.59], 0, 0.4],
    [4, [0, 2, -19.59], -Math.PI, 0.4], [5, [19.59, 2, 19.59], Math.PI / 4, 0.4],
    [6, [0, 2, 0], 0.7, 30], [7, [0, 2, 0], Math.PI - 1e-7, 0.4]]) {
    acting.spawn(i, { position, heading, radius, species: 0, speed: 3 });
    actions[i * 2] = i === 7 ? 1 : 0; actions[i * 2 + 1] = 1;
}
const initialActRecords = new Float32Array(acting.records.data);
const expectedAct = new A.AgentRecords(actDef, new Float32Array(initialActRecords));
A.stepAll(actDef, expectedAct, actions);
device.queue.writeBuffer(acting.act, 0, actions);
await acting.step();
const actualActWords = await read(acting.buffer, actDef.capacity * 16);
function compareAct(words, expected, def) {
    const floats = new Float32Array(words.buffer);
    let ok = true, maxHeading = 0, maxVelocity = 0, maxPosition = 0;
    const rounding = v => 8 * 2 ** -23 * Math.max(1, Math.abs(v));
    for (let i = 0; i < def.capacity; i++) {
        const b = i * 16, flags = expected.u32[b + 13];
        if (!(flags & 1)) {
            for (let k = 0; k < 16; k++) if (words[b + k] !== expected.u32[b + k]) ok = false;
            continue;
        }
        const speed = Math.hypot(expected.data[b + 4], expected.data[b + 6]);
        const velocityBound = speed * 2 ** -11;
        for (let k = 0; k < 16; k++) {
            if ([0, 2, 4, 6, 7].includes(k)) continue;
            if (k === A.AGENT_AGE_OFFSET) {
                if (Math.abs(floats[b + k] - expected.data[b + k]) > rounding(expected.data[b + k])) ok = false;
            } else if (words[b + k] !== expected.u32[b + k]) {
                console.log('A3 unexpected changed field', i, k, floats[b + k], expected.data[b + k]);
                ok = false;
            }
        }
        const headingError = Math.abs(A.wrapAngle(floats[b + 7] - expected.data[b + 7]));
        maxHeading = Math.max(maxHeading, headingError);
        if (headingError > rounding(expected.data[b + 7])) ok = false;
        for (const k of [4, 6]) {
            const error = Math.abs(floats[b + k] - expected.data[b + k]);
            maxVelocity = Math.max(maxVelocity, error);
            if (!Number.isFinite(error) || error > velocityBound + rounding(expected.data[b + k])) ok = false;
        }
        for (const k of [0, 2]) {
            const error = Math.abs(floats[b + k] - expected.data[b + k]);
            maxPosition = Math.max(maxPosition, error);
            if (!Number.isFinite(error) || error > velocityBound * def.dt + rounding(expected.data[b + k])) ok = false;
        }
    }
    return { ok, maxHeading, maxVelocity, maxPosition };
}
const actComparison = compareAct(actualActWords, expectedAct, actDef);
console.log('A3 measured maxima:', JSON.stringify(actComparison));
check('A3: 10000 GPU act rows match stepAll within derived trig/rounding bounds', actComparison.ok);
// Test the generated kernel directly against the same initial rows with X reflection removed.
device.queue.writeBuffer(acting.buffer, 0, initialActRecords);
const noReflection = A.buildAgentActWGSL(actDef).replaceAll('h = wrapAngle(-h);', 'h = h;');
const badEncoder = device.createCommandEncoder();
dispatch(badEncoder, noReflection, [acting.act, acting.buffer, acting.species], Math.ceil(actDef.capacity / 64));
device.queue.submit([badEncoder.finish()]);
check('A3 watched failing: missing wall reflection is rejected',
    !compareAct(await read(acting.buffer, actDef.capacity * 16), expectedAct, actDef).ok);
acting.destroy();

// A14: poison actions ON THE GPU, not via JS float conversion, then allow braking to finish.
const poisonDef = A.defineAgents({ capacity: 8, world: { min: [-20, -20], max: [20, 20] }, cellSize: 4,
    species: loopDef.species, sensors: [A.Sensors.self()] });
const poisoned = makeSystem(engine, poisonDef);
for (let i = 0; i < 8; i++) poisoned.spawn(i, { position: [i - 4, 0, 0], species: 0, speed: 1, heading: 0.5 });
poisoned.upload();
const poisonInitial = new Float32Array(poisoned.records.data);
const poisonEncoder = device.createCommandEncoder();
dispatch(poisonEncoder, `@group(0) @binding(0) var<storage, read_write> act: array<u32>;
@compute @workgroup_size(8) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let bits = select(select(0x7fc00000u, 0x7f800000u, i % 3u == 1u), 0xff800000u, i % 3u == 2u);
    act[i * 2u] = select(bits, bitcast<u32>(1.0), i % 2u != 0u);
    act[i * 2u + 1u] = select(bitcast<u32>(1.0), bits, i % 2u != 0u);
}`, [poisoned.act], 1);
device.queue.submit([poisonEncoder.finish()]);
for (let step = 0; step < 20; step++) await poisoned.step();
const poisonRecords = new Float32Array((await read(poisoned.buffer, 128)).buffer);
check('A14: GPU NaN/+Inf/-Inf in either action component brake to zero, records stay finite',
    Array.from({ length: 8 }, (_, i) => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].every(k => Number.isFinite(poisonRecords[i * 16 + k])) &&
        poisonRecords[i * 16 + 4] === 0 && poisonRecords[i * 16 + 6] === 0 && poisonRecords[i * 16 + 7] === Math.fround(0.5)).every(Boolean));
device.queue.writeBuffer(poisoned.buffer, 0, poisonInitial);
const noSanitize = A.buildAgentActWGSL(poisonDef).replace(
    /if \(\(bitcast<u32>\(a0\)[\s\S]*?\) \{/, 'if (true) {');
if (noSanitize === A.buildAgentActWGSL(poisonDef)) throw new Error('A14 sanitize injection did not change the shader');
const unsafeEncoder = device.createCommandEncoder();
dispatch(unsafeEncoder, noSanitize, [poisoned.act, poisoned.buffer, poisoned.species], 1);
device.queue.submit([unsafeEncoder.finish()]);
const unsafeRecords = new Float32Array((await read(poisoned.buffer, 128)).buffer);
check('A14 watched failing: removed sanitize violates the safe braking contract',
    Array.from({ length: 8 }, (_, i) => {
        const b = i * 16;
        return !Number.isFinite(unsafeRecords[b + 7]) || unsafeRecords[b + 7] !== Math.fround(0.5) ||
            Math.abs(Math.hypot(unsafeRecords[b + 4], unsafeRecords[b + 6]) - 0.9) > 1e-6;
    }).some(Boolean));
poisoned.destroy();

const agingDef = A.defineAgents({ capacity: 65, world: { min: [-8, -8], max: [8, 8] }, cellSize: 4,
    species: loopDef.species, sensors: [A.Sensors.self()] });
const aging = makeSystem(engine, agingDef, { thinkEvery: 3 });
aging.spawn(64, { position: [0, 0, 0], species: 0, energy: 0.75, frozen: true });
await aging.step();
const agingEncoder = device.createCommandEncoder();
dispatch(agingEncoder, `${A.AGENT_STRUCT_WGSL}
@group(0) @binding(0) var<storage, read_write> agents: array<Agent>;
@compute @workgroup_size(1) fn cs_main() { agents[64].state.x = 0.25; }`, [aging.buffer], 1);
device.queue.submit([agingEncoder.finish()]);
await aging.step();
const retainedEnergy = new Float32Array((await read(aging.obs, 130)).buffer)[129];
await aging.step();
check('act cadence: frozen newborn ages, skips tick 1 sensing, refreshes at tick 2',
    retainedEnergy === 0.75 && new Float32Array((await read(aging.obs, 130)).buffer)[129] === 0.25 &&
    new Float32Array((await read(aging.buffer, 65 * 16)).buffer)[64 * 16 + 9] > 0);
aging.destroy();

// A7: two 600-step lockstep runs, full production chains, identical seeded start and actions.
const replayDef = A.defineAgents({ capacity: 128, world: { min: [-8, -8], max: [8, 8] }, cellSize: 4,
    maxScanPerCell: 128, species: loopDef.species,
    sensors: [A.Sensors.self(), A.Sensors.nearest({ k: 3 }), A.Sensors.bounds()] });
const replayRecords = new A.AgentRecords(replayDef), replayActions = new Float32Array(256);
for (let i = 0; i < 128; i++) {
    replayRecords.spawn(i, { position: [-7 + random() * 14, 0, -7 + random() * 14], species: i % 2,
        heading: -Math.PI + random() * 2 * Math.PI, speed: random() * 3 });
    replayActions[i * 2] = random() * 2 - 1; replayActions[i * 2 + 1] = random() * 2 - 1;
}
const replayResults = [];
for (let run = 0; run < 2; run++) {
    const replay = makeSystem(engine, replayDef, { thinkEvery: 3 });
    replay.records.data.set(replayRecords.data); replay.records.markDirty(0, 127);
    device.queue.writeBuffer(replay.act, 0, replayActions);
    for (let step = 0; step < 600; step++) await replay.step();
    replayResults.push(await read(replay.buffer, 2048));
    check(`A7 run ${run + 1}: 600 ticks and zero overflow steps`, equal(await read(replay.state, 4), new Uint32Array([600, 0, 0, 0])));
    replay.destroy();
}
check('A7: two seeded 600-step runs produce byte-identical agent records', equal(...replayResults));

// Custom integration runs only after age/frozen checks, with the same bindings as steering.
const customDef = A.defineAgents({ capacity: 3, world: { min: [-8, -8], max: [8, 8] }, cellSize: 4,
    species: loopDef.species, sensors: [A.Sensors.self()], actions: A.Actions.custom({ name: 'customMove', width: 1,
        wgsl: 'fn customMove(i: u32) { agents[i].posRadius.x += act[i] * DT; }',
        mirror(d, u, i, act, at, dt) { d[i * 16] += act[at] * dt; } }) });
const customSystem = makeSystem(engine, customDef);
for (let i = 0; i < 3; i++) customSystem.spawn(i, { position: [0, 0, 0], species: 0, frozen: i === 1 });
customSystem.kill(2);
const customExpected = new A.AgentRecords(customDef, new Float32Array(customSystem.records.data));
const customActions = new Float32Array([2, 2, 2]); A.stepAll(customDef, customExpected, customActions);
device.queue.writeBuffer(customSystem.act, 0, customActions); await customSystem.step();
check('custom act: CPU twin agrees, frozen ages only and dead stays byte-identical',
    equal(await read(customSystem.buffer, 48), customExpected.u32));
customSystem.destroy();
// A15: invoke only production think on CPU-built observations (no sensing/trig in the oracle).
const ruleDef = A.defineAgents({ capacity: 10000, world: { min: [-100, -100], max: [100, 100] }, cellSize: 4,
    species: loopDef.species, sensors: [A.Sensors.self()] });
const rule = makeSystem(engine, ruleDef);
const ruleObs = new Float32Array(20000), ruleExpected = new Float32Array(20000), ruleReasons = new Uint32Array(10000);
for (let i = 0; i < 10000; i++) {
    rule.spawn(i, { position: [0, 0, 0], species: i % 2, age: 1 });
    ruleObs[i * 2] = Math.fround(random() * 2 - 1);
    ruleObs[i * 2 + 1] = Math.fround(random());
    ruleExpected[i * 2] = Math.fround(Math.fround(A.rand01(i, 0xffffffff, 7) * 2) - 1);
    ruleExpected[i * 2 + 1] = ruleObs[i * 2 + (i % 2)];
    ruleReasons[i] = i ^ 0xffffffff;
}
rule.setBrain(new A.RuleBrain(`fn think(i: u32, species: u32, tick: u32) {
    setAct(i, ACT_TURN, rand01(i, tick, 7u) * 2.0 - 1.0);
    setAct(i, ACT_THROTTLE, obs(i, select(SELF_SPEED, SELF_ENERGY, species == SPECIES_WOLF)));
    setReason(i, i ^ tick);
}`));
rule.upload(); device.queue.writeBuffer(rule.obs, 0, ruleObs);
device.queue.writeBuffer(rule.state, 0, new Uint32Array([0xffffffff, 0, 0, 0]));
rule.advance(ruleDef.dt);
engine.render([rule.getComputeStages().find(s => s.name === 'AgentSystem think')]);
check('A15: 10000 RuleBrain action rows match JS PCG/obs twin byte-exactly',
    equal(await read(rule.act, 20000), new Uint32Array(ruleExpected.buffer)));
check('A15: all reason words match integer twin exactly', equal(await read(rule.reason, 10000), ruleReasons));
rule.destroy();

// A16: full chain retains both actions and reason off-cadence; spawn refreshes a stale slot.
const thinkDef = A.defineAgents({ capacity: 130, world: { min: [-8, -8], max: [8, 8] }, cellSize: 4,
    maxScanPerCell: 130, species: loopDef.species, sensors: [A.Sensors.self()] });
const tickBrain = new A.RuleBrain(`fn think(i: u32, species: u32, tick: u32) {
    setAct(i, ACT_TURN, f32(tick)); setAct(i, ACT_THROTTLE, obs(i, SELF_ENERGY)); setReason(i, tick + 10u);
}`);
const thinking = makeSystem(engine, thinkDef, { thinkEvery: 3 }); thinking.setBrain(tickBrain);
for (let i = 0; i < 130; i++) thinking.spawn(i, { position: [0, 0, 0], species: 0, age: 1,
    energy: 0.25, frozen: true });
thinking.kill(129);
const retainedActs = new Float32Array(260).fill(-7), retainedReasons = new Uint32Array(130).fill(99);
device.queue.writeBuffer(thinking.act, 0, retainedActs);
const reasonEncoder = device.createCommandEncoder();
dispatch(reasonEncoder, `@group(0) @binding(0) var<storage, read_write> reason: array<u32>;
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < 130u) { reason[id.x] = 99u; }
}`, [thinking.reason], 3);
device.queue.submit([reasonEncoder.finish()]);
const cadenceExpected = new Float32Array(retainedActs), cadenceReasons = new Uint32Array(retainedReasons);
let thinkingMatches = true;
for (let tick = 0; tick < 6; tick++) {
    if (tick === 1) thinking.spawn(65, { position: [0, 0, 0], species: 0, frozen: true, energy: 0.75 });
    for (let i = 0; i < 129; i++) if ((tick + Math.floor(i / 64)) % 3 === 0 || (tick === 1 && i === 65)) {
        cadenceExpected[i * 2] = tick; cadenceExpected[i * 2 + 1] = i === 65 && tick >= 1 ? 0.75 : 0.25;
        cadenceReasons[i] = tick + 10;
    }
    await thinking.step();
    if (!equal(await read(thinking.act, 260), new Uint32Array(cadenceExpected.buffer)) ||
        !equal(await read(thinking.reason, 130), cadenceReasons)) thinkingMatches = false;
}
check('A16: tick/actions/reasons update only on block cadence, newborn refreshes immediately, dead retains', thinkingMatches);
// Watch failure using the generated think with its cadence early-return removed.
device.queue.writeBuffer(thinking.act, 0, retainedActs);
const ungated = A.buildAgentThinkWGSL(thinkDef, tickBrain.wgsl, 3).replace('if (!thinksNow(i, a)) { return; }',
    'if ((a.ids.y & 1u) == 0u) { return; }');
const ungatedEncoder = device.createCommandEncoder();
const everySpecies = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
allocated.push(everySpecies);
device.queue.writeBuffer(everySpecies, 0, new Uint32Array([0xffff, 0, 0, 0]));   // binding 6, the species gate (phase 7.3)
dispatch(ungatedEncoder, ungated, [thinking.buffer, thinking.obs, thinking.state, thinking.act, thinking.reason, thinking.species, everySpecies], 3);
device.queue.submit([ungatedEncoder.finish()]);
check('A16 watched failing: ungated think changes off-cadence block 1 at tick 6',
    new Float32Array((await read(thinking.act, 260)).buffer)[64 * 2] !== -7);
thinking.destroy();

// A stage brain can supply caller-owned group-1 data; swapping brain disables old cached stages.
const brainExtra = buffer(1, new Float32Array([0.625]));
const rebound = makeSystem(engine, thinkDef);
rebound.spawn(0, { position: [0, 0, 0], species: 0, frozen: true });
rebound.advance(thinkDef.dt);
const oldBrainStage = rebound.getComputeStages().find(s => s.name === 'AgentSystem think');
rebound.setBrain({ kind: 'stage', buffers: [brainExtra], dims: { obs: 2, act: 2 }, destroy() {},
    wgsl: `@group(1) @binding(0) var<storage, read> weights: array<f32>;
fn think(i: u32, species: u32, tick: u32) {
    setAct(i, ACT_TURN, weights[0]); setAct(i, ACT_THROTTLE, -1.0); setReason(i, 42u);
}` });
await rebound.step();
check('brain: group-1 data works and replacement disables the previous stage',
    !oldBrainStage.enabled && new Float32Array((await read(rebound.act, 260)).buffer)[0] === 0.625 &&
    (await read(rebound.reason, 130))[0] === 42);
rebound.destroy();
check('brain: caller-owned group-1 buffer survives destroy',
    (await read(brainExtra, 1))[0] === new Uint32Array(new Float32Array([0.625]).buffer)[0]);

// A13 with production RuleBrain, not only the earlier test stub. Brain depends on GPU tick/obs.
const movingBrain = new A.RuleBrain(`fn think(i: u32, species: u32, tick: u32) {
    setAct(i, ACT_TURN, rand01(i, tick, 2u) * 2.0 - 1.0);
    setAct(i, ACT_THROTTLE, obs(i, SELF_ENERGY)); setReason(i, tick);
}`);
const productionLoops = [];
for (let run = 0; run < 2; run++) {
    const s = makeSystem(engine, thinkDef, { thinkEvery: 3 }); s.setBrain(movingBrain);
    for (let i = 0; i < 130; i++) s.spawn(i, { position: [-6 + (i % 16) * 0.75, 0, -3 + Math.floor(i / 16) * 0.75],
        species: i % 2, heading: 0.5, energy: 0.25 });
    if (run === 0) { s.upload(); s.advance(3 * thinkDef.dt); engine.render(s.getComputeStages()); }
    else for (let n = 0; n < 3; n++) await s.step();
    productionLoops.push(s);
}
for (const [name, words] of [['buffer', 2080], ['obs', 260], ['act', 260], ['reason', 130], ['state', 4]]) {
    check(`A13 production brain: repeated vs separate chains agree (${name})`,
        equal(await read(productionLoops[0][name], words), await read(productionLoops[1][name], words)));
}
productionLoops.forEach(s => s.destroy());

// A18: catches writes flags/timers only. Respawn moves only self, in a distinct dispatch.
const gameDef = A.defineAgents({ capacity: 128, world: { min: [-8, -8], max: [8, 8] }, cellSize: 4,
    maxScanPerCell: 128, species: loopDef.species,
    sensors: [A.Sensors.self(), A.Sensors.nearest({ name: 'wolf', species: ['wolf'], k: 1 }),
        A.Sensors.nearest({ name: 'prey', species: ['prey'], k: 1 })] });
const gameBrain = new A.RuleBrain(`fn think(i: u32, species: u32, tick: u32) {
    setAct(i, ACT_TURN, rand01(i, tick / 8u, 3u) * 2.0 - 1.0);
    setAct(i, ACT_THROTTLE, -0.5); setReason(i, tick);
}`);
const catchesWGSL = `const CAUGHT = 1u << 8u;
fn step(i: u32, tick: u32) {
    let me = agents[i];
    if (me.ids.x == SPECIES_PREY) {
        let w = neighbours[i * NEIGHBOURS_PER_AGENT + NEIGHBOURS_WOLF];
        if (w != NO_AGENT && distance(agents[w].posRadius.xz, me.posRadius.xz) < 0.9) {
            agents[i].ids.y |= CAUGHT;
        }
    } else {
        let p = neighbours[i * NEIGHBOURS_PER_AGENT + NEIGHBOURS_PREY];
        if (p != NO_AGENT && distance(agents[p].posRadius.xz, me.posRadius.xz) < 0.9) {
            agents[i].state.w += 1.0;
        }
    }
}`;
const respawnWGSL = `const CAUGHT = 1u << 8u;
@group(1) @binding(0) var<storage, read_write> catches: atomic<u32>;
fn step(i: u32, tick: u32) {
    if ((agents[i].ids.y & CAUGHT) == 0u) { return; }
    respawn(i, randomPointInWorld(i, tick), rand01(i, tick, 2u) * 6.2831853 - 3.1415927);
    agents[i].ids.y &= ~CAUGHT;
    atomicAdd(&catches, 1u);
}`;
const games = [];
for (let run = 0; run < 2; run++) {
    const s = makeSystem(engine, gameDef, { thinkEvery: 3 }), score = buffer(1);
    s.setBrain(gameBrain);
    s.addStepStage({ name: 'catches', wgsl: catchesWGSL });
    s.addStepStage({ name: 'respawn', wgsl: respawnWGSL, buffers: [score] });
    for (let i = 0; i < 128; i++) s.spawn(i, { position: [-6 + (Math.floor(i / 2) % 8) * 1.5 + (i % 2) * 0.1,
        0, -6 + Math.floor(i / 16) * 1.5], species: i % 2, heading: 0 });
    await s.step();
    const first = new Float32Array((await read(s.buffer, 2048)).buffer);
    check(`A18 run ${run + 1}: all 64 prey caught/respawned on step 0, age/velocity reset`,
        (await read(score, 1))[0] === 64 && Array.from({ length: 64 }, (_, j) => {
            const b = j * 32;
            return first[b + 9] === 0 && first[b + 4] === 0 && first[b + 6] === 0 &&
                first[b] !== s.records.data[b] && first[b + 2] !== s.records.data[b + 2];
        }).every(Boolean));
    for (let step = 1; step < 600; step++) await s.step();
    const records = await read(s.buffer, 2048), floats = new Float32Array(records.buffer);
    check(`A18 run ${run + 1}: finite/inset records, cleared catch flags, 600 ticks and no overflow`,
        Array.from({ length: 128 }, (_, i) => {
            const b = i * 16;
            return Array.from({ length: 12 }, (_, k) => Number.isFinite(floats[b + k])).every(Boolean) &&
                Math.abs(floats[b]) <= 7.600001 && Math.abs(floats[b + 2]) <= 7.600001 && !(records[b + 13] & 256);
        }).every(Boolean) && equal(await read(s.state, 4), new Uint32Array([600, 0, 0, 0])));
    games.push({ records, obs: await read(s.obs, 128 * gameDef.obsDim), act: await read(s.act, 256),
        reason: await read(s.reason, 128), score: await read(score, 1) });
    s.destroy();
    check(`A18 run ${run + 1}: app-owned group-1 score survives system destroy`, equal(await read(score, 1), games[run].score));
}
check('A18/A7: two 600-step catches/respawn runs agree byte-exactly, including obs/actions/reasons/score',
    Object.keys(games[0]).every(key => equal(games[0][key], games[1][key])));
// Arrow rendering reads GPU records, including dead flags, species and heading.
const arrowDef = A.defineAgents({ capacity: 3, world: { min: [-8, -8], max: [8, 8] }, cellSize: 4,
    species: ['prey', 'wolf'].map(name => ({ name, maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 4, radius: 1 })),
    sensors: [A.Sensors.self()] });
const arrowSystem = makeSystem(engine, arrowDef);
arrowSystem.spawn(0, { position: [-3, 0, 0], species: 'prey', heading: 0 });
arrowSystem.spawn(1, { position: [3, 0, 0], species: 'wolf', heading: Math.PI / 2 });
arrowSystem.spawn(2, { position: [0, 0, 0], species: 'wolf' });
arrowSystem.kill(2); arrowSystem.upload();
const arrowPass = engine.createPass({ name: 'GPU agent arrows', isMainScreenPass: true });
const arrows = agentArrows(engine, arrowPass, arrowSystem, { colors: [[0, 1, 0], [1, 0, 0]] });
engine.render([arrowPass]);
const arrowPixels = await image();
await saveImage('agents-arrows', arrowPixels, 128, 128);
const colored = pixels => {
    let red = 0, green = 0;
    for (let p = 0; p < pixels.length; p += 4) {
        if (pixels[p] > 150 && pixels[p + 1] < 60) red++;
        if (pixels[p + 1] > 150 && pixels[p] < 60) green++;
    }
    return { red, green };
};
check('arrows: live species draw distinct colors from GPU records', colored(arrowPixels).red > 10 && colored(arrowPixels).green > 10);
arrowSystem.kill(1); arrowSystem.upload(); engine.render([arrowPass]);
const deadPixels = await image();
check('arrows: dead species clips completely, live species remains', colored(deadPixels).red === 0 && colored(deadPixels).green > 10);
// Mutate only the GPU record, leaving staging untouched: proves there is no CPU position mirror.
device.queue.writeBuffer(arrowSystem.buffer, 0, new Float32Array([-1, 0, 2, 1]));
device.queue.writeBuffer(arrowSystem.buffer, 28, new Float32Array([Math.PI / 2]));
engine.render([arrowPass]);
check('arrows: GPU-only position/heading changes alter pixels without staging upload', !equal(deadPixels, await image()));
arrows.destroy(); arrows.destroy();
check('arrows: destroying draw preserves caller-owned records', (await read(arrowSystem.buffer, 4))[0] === new Uint32Array(new Float32Array([-1]).buffer)[0]);
arrowSystem.destroy();

// A17 (phase 4.2): the instance stage. Rows the GPU writes into vis.instanceBuffer equal U19's CPU
// twin applied to the read-back records; rows it must not touch keep a sentinel; the visibility cull
// then draws exactly the live agents, each with its mesh or highlight variant.
async function readStorage(src, words) {   // vis.instanceBuffer and indirect args have no COPY_SRC
    const dst = buffer(words), enc = device.createCommandEncoder();
    dispatch(enc, `@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < ${words}u) { dst[id.x] = src[id.x]; }
}`, [src, dst], Math.ceil(words / 64));
    device.queue.submit([enc.finish()]);
    return read(dst, words);
}
const MESH = { PREY: 0, HOVER: 1, SELECTED: 2, WOLF: 3 };
const instDef = A.defineAgents({ capacity: 2000, world: { min: [-8, -8], max: [8, 8] }, cellSize: 4,
    species: [
        { name: 'prey', maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 4, radius: 0.4,
            mesh: MESH.PREY, hoverMesh: MESH.HOVER, selectedMesh: MESH.SELECTED, boundsRadius: 0.5 },
        { name: 'wolf', maxSpeed: 4, maxTurn: 3, accel: 5, senseRadius: 4, radius: 0.6,
            mesh: MESH.WOLF, hoverMesh: MESH.HOVER, selectedMesh: MESH.SELECTED, boundsRadius: 0.5 },
    ],
    sensors: [A.Sensors.self()] });
const FIRST = 3, ROWS = FIRST + instDef.capacity, HIGHLIGHT = { selected: 5, hovered: 9 };
const mega = new MegabufferBuilder(StandardLayout.strideFloats);
for (const name of ['prey', 'hover', 'selected', 'wolf']) {
    const g = Primitives.createSphere(StandardLayout, 0.5, 8, 6);
    mega.addMesh(name, g.vertices, g.indices);
}
const vis = new VisibilitySystem(engine, { maxInstances: ROWS + 5 });
vis.setGeometry(mega);
vis.setInstanceCount(ROWS);
const sentinel = new Float32Array(ROWS * 24).fill(7);   // rows the GPU must leave alone keep this
device.queue.writeBuffer(vis.instanceBuffer, 0, sentinel);
const instSystem = makeSystem(engine, instDef);
for (let i = 0; i < instDef.capacity; i++) {
    instSystem.spawn(i, { position: [random() * 15 - 7.5, random(), random() * 15 - 7.5],
        species: i % 3 === 0 ? 'wolf' : 'prey', heading: random() * 2 * Math.PI - Math.PI, speed: random() * 2 });
}
for (let i = 0; i < instDef.capacity; i += 7) instSystem.kill(i);
instSystem.toInstances(vis, { firstInstance: FIRST });
instSystem.setHighlight(HIGHLIGHT);
await instSystem.step();                                  // one step, then the instance stage
const after = await read(instSystem.buffer, instDef.capacity * 16);
const rowsGpu = await readStorage(vis.instanceBuffer, ROWS * 24);
const twin = new Float32Array(sentinel), twinU32 = new Uint32Array(twin.buffer);
const recAfter = { data: new Float32Array(after.buffer), u32: after };
for (let i = 0; i < instDef.capacity; i++) A.writeAgentInstanceRow(instDef, recAfter, i, HIGHLIGHT, twin, FIRST + i, twinU32);
// sin/cos: WGSL promises 2^-11 absolute on [-π, π]; the twin's f64 value is rounded to f32 once.
const rotationBound = 2 ** -11 + 2 ** -24;
const rowsF32 = new Float32Array(rowsGpu.buffer);
let rotationMax = 0, exactMismatch = 0, rotationOut = 0;
for (let w = 0; w < ROWS * 24; w++) {
    const row = Math.floor(w / 24), k = w % 24, agent = row - FIRST;
    const live = agent >= 0 && (after[agent * 16 + 13] & 1) !== 0;
    if (live && (k === 0 || k === 2 || k === 8 || k === 10)) {
        const e = Math.abs(rowsF32[w] - twin[w]);
        rotationMax = Math.max(rotationMax, e);
        if (!(e <= rotationBound)) rotationOut++;
    } else if (rowsGpu[w] !== twinU32[w]) exactMismatch++;
}
check(`A17: ${instDef.capacity} GPU instance rows equal the CPU twin (rotation within 2^-11, max ${rotationMax.toExponential(2)}; ` +
    'translation, bounds, mesh/highlight, LOD and untouched rows bit-exact)', rotationOut === 0 && exactMismatch === 0);
const pass = engine.createPass({ name: 'GPU agent instances', isMainScreenPass: true });
const batches = Object.entries(MESH).map(([name, mesh]) => [mesh, vis.createInstanceDrawBatch(pass, name.toLowerCase(), mesh)]);
camera.updateView([0, 18, 0], [0, 0, 0], [0, 0, -1]);
engine.updateCamera(camera);
vis.upload();
engine.render([...vis.getComputeStages(), pass]);
await saveImage('agents-instances', await image(), 128, 128);
async function drawnPerMesh() {
    const drawn = {};
    for (const [mesh, batch] of batches) drawn[mesh] = (await readStorage(batch.indirectBuffer, 5))[1];
    return drawn;
}
function expectedPerMesh(highlight) {
    const want = { [MESH.PREY]: 0, [MESH.HOVER]: 0, [MESH.SELECTED]: 0, [MESH.WOLF]: 0 };
    for (let i = 0; i < instDef.capacity; i++) {
        if ((after[i * 16 + 13] & 1) === 0) continue;
        if (i === highlight.selected) want[MESH.SELECTED]++;
        else if (i === highlight.hovered) want[MESH.HOVER]++;
        else want[after[i * 16 + 12] === 1 ? MESH.WOLF : MESH.PREY]++;
    }
    return want;
}
const same = (a, b) => Object.keys(b).every(m => a[m] === b[m]);
const drawn = await drawnPerMesh(), want = expectedPerMesh(HIGHLIGHT);
check(`A17: the visibility cull draws exactly the live agents through GPU rows (drawn ${JSON.stringify(drawn)})`,
    same(drawn, want) && want[MESH.SELECTED] === 1 && want[MESH.HOVER] === 1 && want[MESH.PREY] > 0 && want[MESH.WOLF] > 0);
instSystem.setHighlight({ selected: -1, hovered: -1 });
instSystem.advance(0);
engine.render([...instSystem.getComputeStages(), ...vis.getComputeStages(), pass]);   // no step: highlight only
const unlit = await drawnPerMesh();
check(`A17: with no step this frame, the instance stage still applies a highlight change (drawn ${JSON.stringify(unlit)})`,
    same(unlit, expectedPerMesh({ selected: -1, hovered: -1 })) && unlit[MESH.PREY] + unlit[MESH.WOLF] === want[MESH.PREY] + want[MESH.WOLF] + 2);
instSystem.destroy();                                     // dependents first: the stage binds vis.instanceBuffer
vis.destroy();

// A10–A12 (phase 4.3): picking and labels on GPU agents. The Picker culls the rows the instance
// stage wrote, so hit.index − firstInstance is the agent; a label follows agents.buffer with no
// CPU writes; a dead agent's row is HIDDEN_MESH, so it is neither drawn nor picked.
{
    const W = 128, H = 128, FIRST_ROW = 2, LABELLED = 1;
    const pickDef = A.defineAgents({ capacity: 32, world: { min: [-8, -8], max: [8, 8] }, cellSize: 4,
        species: instDef.species.map(s => ({ name: s.name, maxSpeed: s.maxSpeed, maxTurn: s.maxTurn, accel: s.accel,
            senseRadius: s.senseRadius, radius: s.radius, mesh: s.mesh, hoverMesh: s.hoverMesh,
            selectedMesh: s.selectedMesh, boundsRadius: s.boundsRadius })),
        sensors: [A.Sensors.self()] });
    const pvis = new VisibilitySystem(engine, { maxInstances: FIRST_ROW + pickDef.capacity });
    pvis.setGeometry(mega);
    pvis.setInstanceCount(FIRST_ROW + pickDef.capacity);
    const staticRows = new Float32Array(FIRST_ROW * 24);
    for (let r = 0; r < FIRST_ROW; r++) new Uint32Array(staticRows.buffer)[r * 24 + 20] = A.HIDDEN_MESH;
    device.queue.writeBuffer(pvis.instanceBuffer, 0, staticRows);
    const scene = engine.createPass({ name: 'GPU agent picking', isMainScreenPass: true });
    const drawBatches = Object.entries(MESH).map(([name, mesh]) => [mesh, pvis.createInstanceDrawBatch(scene, name.toLowerCase(), mesh)]);
    const picker = new Picker(engine, { maxRadius: 2 });
    Object.entries(MESH).forEach(([name, mesh]) => picker.addInstances(pvis, name.toLowerCase(), mesh, { label: 'agent' }));
    const sys = makeSystem(engine, pickDef);
    for (let i = 0; i < pickDef.capacity; i++) {
        sys.spawn(i, { position: [random() * 14 - 7, 0, random() * 14 - 7], species: i % 4 === 0 ? 'wolf' : 'prey',
            heading: random() * 2 * Math.PI - Math.PI });
    }
    sys.toInstances(pvis, { firstInstance: FIRST_ROW });
    const labelAtlas = FontAtlas.fromSDF(engine, syntheticFont());
    const labels = new TextSystem(engine, labelAtlas, { maxLabels: 1, maxGlyphs: 2, depthTest: false,
        follow: { buffer: sys.buffer, strideFloats: 16, positionOffset: 0 } });
    const labelPass = engine.createPass({ name: 'GPU agent label', isMainScreenPass: true });
    labels.createDrawBatch(labelPass);
    labels.setLabel(0, { text: 'B', follow: LABELLED, sizePx: 24 });
    camera.updateView([0, 18, 0], [0, 0, 0], [0, 0, -1]);
    engine.updateCamera(camera);

    const project = (p) => {   // the text shaders' anchor maths: clip → pixel-snapped screen position
        const v = camera.viewProjectionMatrix;
        const cx = v[0] * p[0] + v[4] * p[1] + v[8] * p[2] + v[12];
        const cy = v[1] * p[0] + v[5] * p[1] + v[9] * p[2] + v[13];
        const cw = v[3] * p[0] + v[7] * p[1] + v[11] * p[2] + v[15];
        return [Math.floor((cx / cw * 0.5 + 0.5) * W + 0.5), Math.floor((-cy / cw * 0.5 + 0.5) * H + 0.5)];
    };
    const labelRect = (p) => {   // where 'B' (filled box [0, 0, 0.5, 0.5] em) lands at 24 px
        const [ax, ay] = project(p), [l, b, r, t] = SHAPES.B.outer;
        return { x0: ax + l * 24, x1: ax + r * 24, y0: ay - t * 24, y1: ay - b * 24 };
    };
    const brightBox = (img) => {
        let x0 = Infinity, y0 = Infinity, x1 = -1, y1 = -1;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            const o = (y * W + x) * 4;
            if (img[o] > 128 && img[o + 1] > 128 && img[o + 2] > 128) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x + 1); y1 = Math.max(y1, y + 1); }
        }
        return x1 < 0 ? null : { x0, y0, x1, y1 };
    };
    const near = (got, want) => got && ['x0', 'y0', 'x1', 'y1'].every(k => Math.abs(got[k] - want[k]) <= 1);
    const records = async () => {
        const u = await read(sys.buffer, pickDef.capacity * 16);
        return { u, f: new Float32Array(u.buffer) };
    };
    const position = (r, i) => [r.f[i * 16], r.f[i * 16 + 1], r.f[i * 16 + 2]];
    const drawScene = () => {   // a frame with no step: the instance stage still writes every row
        sys.advance(0);
        pvis.upload();
        engine.render([...sys.getComputeStages(), ...pvis.getComputeStages(), scene]);
    };
    const drawLabel = async () => {
        labels.upload();
        engine.render([...labels.getComputeStages(), labelPass]);
        return brightBox(await image());
    };

    sys.upload();
    drawScene();
    const startRecords = await records();
    const box0 = await drawLabel(), want0 = labelRect(position(startRecords, LABELLED));
    let labelWrites = 0;
    const realWrite = device.queue.writeBuffer.bind(device.queue);
    device.queue.writeBuffer = (target, ...rest) => {
        if (target === labels.labelBuffer || target === labels.glyphBuffer) labelWrites++;
        return realWrite(target, ...rest);
    };
    for (let k = 0; k < 120; k++) await sys.step();
    const box1 = await drawLabel();
    device.queue.writeBuffer = realWrite;
    const moved = await records();
    const want1 = labelRect(position(moved, LABELLED));
    const shift = Math.hypot(want1.x0 - want0.x0, want1.y0 - want0.y0);
    check(`A11: a label follows its agent through agents.buffer for 120 steps with zero CPU label writes ` +
        `(moved ${shift.toFixed(0)} px; label/glyph writes ${labelWrites})`,
        near(box0, want0) && near(box1, want1) && shift >= 6 && labelWrites === 0);

    // A10: after 120 moving steps, every isolated live agent is picked as itself.
    drawScene();
    const live = [];
    for (let i = 0; i < pickDef.capacity; i++) if (moved.u[i * 16 + 13] & 1) live.push(i);
    const isolated = live.filter(i => {
        const [x, , z] = position(moved, i), [px, py] = project(position(moved, i));
        // 1.5 m apart: the 0.5 m spheres cannot overlap, so the pixel under an agent's centre is its own.
        const alone = live.every(j => j === i || Math.hypot(position(moved, j)[0] - x, position(moved, j)[2] - z) >= 1.5);
        return alone && px >= 3 && px < W - 3 && py >= 3 && py < H - 3;
    }).slice(0, 10);
    const meshOf = (i) => (moved.u[i * 16 + 12] === 1 ? MESH.WOLF : MESH.PREY);
    let picked = 0, wrong = [];
    for (const i of isolated) {
        const [px, py] = project(position(moved, i));
        const hit = await picker.pick(px, py, camera);
        if (hit && hit.index - FIRST_ROW === i && hit.meshIndex === meshOf(i)) picked++;
        else wrong.push(`${i}→${hit ? hit.index - FIRST_ROW : 'none'}`);
    }
    const chosen = isolated[0];
    sys.setHighlight({ selected: chosen });
    drawScene();
    const selectedHit = await picker.pick(...project(position(moved, chosen)), camera);
    check(`A10: after 120 moving steps, ${picked}/${isolated.length} isolated agents are picked as themselves ` +
        `(hit.index − firstInstance), the selected variant too${wrong.length ? `; wrong: ${wrong.join(', ')}` : ''}`,
        isolated.length >= 6 && picked === isolated.length &&
        selectedHit?.index - FIRST_ROW === chosen && selectedHit?.meshIndex === MESH.SELECTED);

    // A12: a killed agent is neither drawn nor picked.
    sys.setHighlight({ selected: -1 });
    const drawnOf = async (mesh) => (await readStorage(drawBatches.find(([m]) => m === mesh)[1].indirectBuffer, 5))[1];
    drawScene();
    const before = await drawnOf(meshOf(chosen));
    sys.kill(chosen);
    sys.upload();
    drawScene();
    const after = await drawnOf(meshOf(chosen));
    const ghost = await picker.pick(...project(position(moved, chosen)), camera);
    check(`A12: a dead agent draws nothing and is never picked (drawn ${before} → ${after}; pick at its spot: ${ghost ? ghost.index - FIRST_ROW : 'none'})`,
        after === before - 1 && ghost === null);

    // A19 (phase 4.4): the CPU frame path — advance, upload, a highlight change, getComputeStages —
    // constructs no typed arrays and makes at most two writeBuffer calls (here: the highlight's).
    const typed = [Float32Array, Float64Array, Uint32Array, Int32Array, Uint16Array, Uint8Array];
    let constructed = 0, maxWrites = 0, minWrites = Infinity;
    const realWrite19 = device.queue.writeBuffer.bind(device.queue);
    for (let frame = 0; frame < 60; frame++) {
        let frameWrites = 0;
        device.queue.writeBuffer = (...args) => { frameWrites++; return realWrite19(...args); };
        typed.forEach(T => { globalThis[T.name] = new Proxy(T, { construct(t, a) { constructed++; return Reflect.construct(t, a); } }); });
        sys.advance(pickDef.dt);
        sys.upload();
        sys.setHighlight({ hovered: isolated[1 + (frame % 2)] });
        const chain = sys.getComputeStages();
        typed.forEach(T => { globalThis[T.name] = T; });
        device.queue.writeBuffer = realWrite19;
        maxWrites = Math.max(maxWrites, frameWrites);
        minWrites = Math.min(minWrites, frameWrites);
        engine.render([...chain, ...pvis.getComputeStages(), scene]);   // outside the measured window
    }
    check(`A19: per frame the agent system constructs ${constructed} typed arrays and makes ${minWrites}–${maxWrites} ` +
        'writeBuffer call(s) (limit 2; one is the highlight change)', constructed === 0 && maxWrites <= 2 && minWrites === 1);

    // A20 (phase 4.5): remove dependents before their buffers, then replace the frame sequence.
    // Keep the old nodes to verify destroy disables them, but never render released draw batches.
    const retiredStages = [...sys.getComputeStages()];
    device.pushErrorScope('validation');
    picker.destroy();                                     // its pick batches live in pvis
    labels.destroy();
    labelAtlas.destroy();
    sys.destroy();                                        // before pvis: the instance stage binds its rows
    drawBatches.forEach(([, batch]) => pvis.removeDrawBatch(batch));
    pvis.destroy();
    for (let frame = 0; frame < 12; frame++) engine.render([scene, labelPass]);
    await device.queue.onSubmittedWorkDone();
    const teardownError = await device.popErrorScope();
    check('A20: destroy disables every retired agent stage, including grid and instances',
        retiredStages.length > 0 && retiredStages.every(stage => stage.enabled === false));
    check(`A20: dependents-first teardown and replacement sequence render 12 later frames without validation errors` +
        (teardownError ? `: ${teardownError.message}` : ''), teardownError === null);

    // Rebuild on the same engine: teardown must not destroy the caller's device or render pass.
    const replacement = makeSystem(engine, pickDef);
    replacement.spawn(0, { position: [0, 0, 0], species: 'prey' });
    replacement.upload();
    const replacementDraw = agentArrows(engine, scene, replacement);
    device.pushErrorScope('validation');
    for (let frame = 0; frame < 3; frame++) {
        replacement.advance(pickDef.dt);
        engine.render([...replacement.getComputeStages(), scene]);
    }
    const replacementSnapshot = await replacement.inspect(0);
    const rebuildError = await device.popErrorScope();
    check('A20: the same engine renders a rebuilt system and inspection observes its three ticks',
        rebuildError === null && replacementSnapshot?.alive && replacementSnapshot.tick === 3);
    replacementDraw.destroy();
    replacement.destroy();
}
// Phase 4.4: inspect(i) copies agent i's record, obs and act rows, reason, neighbour slots and the
// counters in one submission. Every field must equal the same buffers read directly.
{
    const inspDef = A.defineAgents({ capacity: 64, world: { min: [-8, -8], max: [8, 8] }, cellSize: 8, maxScanPerCell: 64,
        species: [
            { name: 'prey', maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 8, radius: 0.4 },
            { name: 'wolf', maxSpeed: 4, maxTurn: 3, accel: 5, senseRadius: 8, radius: 0.6 },
        ],
        sensors: [A.Sensors.self(), A.Sensors.nearest({ name: 'wolf', species: ['wolf'], k: 1 }),
            A.Sensors.nearest({ name: 'prey', species: ['prey'], k: 3 }), A.Sensors.bounds()] });
    const insp = new A.AgentSystem(engine, inspDef);
    // Every agent gets its own actions and reason, so a snapshot of the wrong row cannot pass.
    insp.setBrain(new A.RuleBrain(`fn think(i: u32, species: u32, tick: u32) {
    setAct(i, ACT_TURN, f32(i % 9u) * 0.1 - 0.4);
    setAct(i, ACT_THROTTLE, f32(i % 5u) * 0.2 - 0.4);
    setReason(i, 100u + i);
}`));
    for (let i = 0; i < inspDef.capacity; i++) {
        insp.spawn(i, { position: [random() * 14 - 7, 0, random() * 14 - 7], species: i % 4 === 0 ? 'wolf' : 'prey',
            heading: random() * 2 * Math.PI - Math.PI, speed: random() * 2 });
    }
    insp.kill(9);
    for (let k = 0; k < 40; k++) await insp.step();
    const agents = [3, 9, 17, 40, 63];
    const snaps = [];
    for (const i of agents) snaps.push(await insp.inspect(i));
    const K = 4, rec = await read(insp.buffer, inspDef.capacity * 16), recF = new Float32Array(rec.buffer);
    const obs = new Float32Array((await read(insp.obs, inspDef.capacity * inspDef.obsDim)).buffer);
    const act = new Float32Array((await read(insp.act, inspDef.capacity * inspDef.actDim)).buffer);
    const reason = await read(insp.reason, inspDef.capacity), neighbours = await read(insp.neighbours, inspDef.capacity * K);
    const counters = await read(insp.state, 4);
    const problems = [];
    snaps.forEach((s, n) => {
        const i = agents[n], b = i * 16;
        const expect = {
            index: i, alive: (rec[b + 13] & 1) !== 0, species: inspDef.species[rec[b + 12]].name, flags: rec[b + 13],
            position: [recF[b], recF[b + 1], recF[b + 2]], radius: recF[b + 3], heading: recF[b + 7], energy: recF[b + 8],
            age: recF[b + 9], speedScale: recF[b + 10], user0: recF[b + 11],
            speed: Math.hypot(recF[b + 4], recF[b + 5], recF[b + 6]),
            obs: Array.from(obs.subarray(i * inspDef.obsDim, (i + 1) * inspDef.obsDim)),
            act: Array.from(act.subarray(i * 2, i * 2 + 2)), reason: reason[i],
            neighbours: [Array.from(neighbours.subarray(i * K, i * K + 1)), Array.from(neighbours.subarray(i * K + 1, i * K + 4))]
                .map(slots => slots.filter(j => j !== 0xFFFFFFFF)),
            tick: counters[0], overflowSteps: counters[1], overflowTotal: counters[2],
        };
        const got = { ...s, obs: s.obs.map(o => o.value), act: s.act.map(o => o.value), neighbours: s.neighbours.map(x => x.indices) };
        for (const key of Object.keys(expect)) {
            if (JSON.stringify(got[key]) !== JSON.stringify(expect[key])) problems.push(`agent ${i} ${key}`);
        }
    });
    const live = snaps.filter(s => s.alive);
    check(`inspect (4.4): ${snaps.length} snapshots equal the buffers read directly — record, obs, act, reason, ` +
        `neighbours, counters${problems.length ? `; differs: ${problems.slice(0, 6).join(', ')}` : ''}`,
        problems.length === 0 && live.length === 4 && !snaps[1].alive && live.every(s => s.reason === 100 + s.index) &&
        live.some(s => s.neighbours.some(x => x.indices.length > 0)) && snaps[0].tick === 40);
    insp.destroy();
    check('inspect (4.4): a request after destroy() is refused', (() => { try { insp.inspect(0); return false; } catch { return true; } })());
}

// A4 (phase 6.2, docs/agents-mlp-design.md §4.1, §6): MlpBrain as the production think stage. Each
// live agent's action row must equal mlpForward() applied to the observation row the GPU's own sense
// stage wrote in that step: |gpu − cpu| ≤ 1e-5 + 1e-5·|cpu|. Then gating, same-shape updates,
// replacement and lifecycle, each inside a validation error scope.
{
    const TOL = 1e-5;
    const readF = async (b, n) => new Float32Array((await read(b, n)).buffer);
    const mlpNet = (...layers) => layers.map(([input, output, activation, scale = 1, biasScale = 0.5 * scale]) => ({
        input, output, activation,
        weights: Float32Array.from({ length: input * output }, () => (random() * 2 - 1) * scale * Math.sqrt(3 / input)),
        biases: Float32Array.from({ length: output }, () => (random() * 2 - 1) * biasScale) }));
    // Signed inputs in [-1, 1) from an integer hash of (agent, field, tick): every step senses new values.
    const probe = width => A.Sensors.custom({ name: 'probe', width, wgsl: `fn probe(i: u32, a: Agent) -> array<f32, ${width}> {
    var v: array<f32, ${width}>;
    for (var k = 0u; k < ${width}u; k++) {
        var h = (i * 0x9e3779b9u) ^ (k * 0x85ebca6bu) ^ (state.tick * 0xc2b2ae35u) ^ 0x27d4eb2fu;
        h = (h ^ (h >> 16u)) * 0x7feb352du;
        h = (h ^ (h >> 15u)) * 0x846ca68bu;
        h = h ^ (h >> 16u);
        v[k] = f32(h >> 8u) * (2.0 / 16777216.0) - 1.0;
    }
    return v;
}` });
    const mlpDef = (obs, act, capacity, species = {}) => A.defineAgents({ capacity, world: { min: [-8, -8], max: [8, 8] }, cellSize: 4,
        species: [{ name: 'prey', maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 4, radius: 0.4, ...species }],
        sensors: [probe(obs)],
        actions: act === 2 ? A.Actions.steering() : A.Actions.custom({ name: 'hold', width: act, wgsl: 'fn hold(i: u32) {}' }) });
    function populate(sys, def, kill = i => i % 23 === 5, extra = () => ({})) {
        const live = [];
        for (let i = 0; i < def.capacity; i++) {
            sys.spawn(i, { position: [random() * 14 - 7, 0, random() * 14 - 7], species: 0,
                heading: random() * 2 * Math.PI - Math.PI, speed: random() * 2, ...extra(i) });
            if (kill(i)) sys.kill(i); else live.push(i);
        }
        return live;
    }
    function fillReason(reason, capacity, value) {      // reason has no COPY_DST
        const enc = device.createCommandEncoder();
        dispatch(enc, `@group(0) @binding(0) var<storage, read_write> reason: array<u32>;
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < ${capacity}u) { reason[id.x] = ${value}u; }
}`, [reason], Math.ceil(capacity / 64));
        device.queue.submit([enc.finish()]);
    }
    // GPU actions against mlpForward of the GPU's observation row, for every listed row. NaN fails.
    function agree(net, def, obs, act, rows) {
        const input = new Float32Array(def.obsDim), out = new Float32Array(def.actDim);
        const scratch = new Float32Array(2 * A.mlpLayout(net).scratchWidth);
        const r = { maxAbs: 0, maxRel: 0, bad: 0, values: 0 };
        for (const i of rows) {
            input.set(obs.subarray(i * def.obsDim, (i + 1) * def.obsDim));
            A.mlpForward(net, input, out, scratch);
            for (let k = 0; k < def.actDim; k++) {
                const g = act[i * def.actDim + k], c = out[k], e = Math.abs(g - c);
                r.values++;
                if (!(e <= TOL + TOL * Math.abs(c))) r.bad++;
                if (Number.isFinite(e)) {
                    r.maxAbs = Math.max(r.maxAbs, e);
                    if (Math.abs(c) >= 1e-3) r.maxRel = Math.max(r.maxRel, e / Math.abs(c));   // relative error of ~0 is noise
                }
            }
        }
        return r;
    }
    const errors = r => `max abs ${r.maxAbs.toExponential(2)}, max rel ${r.maxRel.toExponential(2)} where |cpu| ≥ 1e-3`;
    async function runSystem(net, def, steps, extra, options) {
        const sys = new A.AgentSystem(engine, def, options), brain = new A.MlpBrain(engine, net);
        sys.setBrain(brain);
        const live = populate(sys, def, undefined, extra), liveSet = new Set(live);
        const dead = Array.from({ length: def.capacity }, (_, i) => i).filter(i => !liveSet.has(i));
        device.queue.writeBuffer(sys.act, 0, new Float32Array(def.capacity * def.actDim).fill(-7));
        fillReason(sys.reason, def.capacity, 99);
        device.pushErrorScope('validation');
        for (let s = 0; s < steps; s++) await sys.step();
        const error = await device.popErrorScope();
        return { sys, brain, live, dead, error, obs: await readF(sys.obs, def.capacity * def.obsDim),
            act: await readF(sys.act, def.capacity * def.actDim), reason: await read(sys.reason, def.capacity) };
    }
    async function fixture(name, net, def, steps, extra) {
        const run = await runSystem(net, def, steps, extra), { live, dead, act, reason } = run;
        const r = agree(net, def, run.obs, act, live);
        const untouched = dead.every(i => reason[i] === 99 && act.subarray(i * def.actDim, (i + 1) * def.actDim).every(v => v === -7));
        check(`A4 ${name}: ${live.length} live agents × ${def.actDim} actions match mlpForward of their GPU observations ` +
            `(${errors(r)}); reason 0; ${dead.length} dead rows untouched`,
            run.error === null && r.bad === 0 && live.every(i => reason[i] === 0) && untouched && dead.length > 0);
        return { ...run, r };
    }

    const affine = await fixture('one affine layer 3→2 identity', mlpNet([3, 2, 'identity']), mlpDef(3, 2, 100), 2);
    const oddNet = mlpNet([5, 7, 'relu'], [7, 3, 'sigmoid']);
    const odd = await fixture('odd widths 5→7→3 (relu, sigmoid)', oddNet, mlpDef(5, 3, 100), 2);
    const mixedNet = mlpNet([9, 13, 'tanh'], [13, 6, 'relu'], [6, 11, 'identity'], [11, 5, 'sigmoid']);
    const mixedDef = mlpDef(9, 5, 100);
    const mixed = await fixture('four layers 9→13→6→11→5 (tanh, relu, identity, sigmoid)', mixedNet, mixedDef, 2);
    const maxNet = mlpNet([128, 64, 'relu'], [64, 64, 'tanh'], [64, 64, 'identity'], [64, 16, 'sigmoid']);
    const widest = await fixture('maximum shape 128→64→64→64→16', maxNet, mlpDef(128, 16, 100), 2);

    // The 24→32→32→2 tanh network of S3 on Herd & Wolves' real sensors, through 20 moving steps.
    const herdDef = A.defineAgents({ capacity: 2000, world: { min: [-20, -20], max: [20, 20] }, cellSize: 8,
        species: [
            { name: 'prey', maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 8, radius: 0.4 },
            { name: 'wolf', maxSpeed: 4, maxTurn: 3, accel: 5, senseRadius: 8, radius: 0.6 },
        ],
        sensors: [A.Sensors.self(), A.Sensors.nearest({ name: 'wolf', species: ['wolf'], k: 1 }),
            A.Sensors.nearest({ name: 'prey', species: ['prey'], k: 3 }), A.Sensors.bounds()] });
    const herdNet = mlpNet([24, 32, 'tanh'], [32, 32, 'tanh'], [32, 2, 'tanh']);
    const herd = await fixture(`Herd & Wolves ${herdDef.obsDim}→32→32→2 tanh, 20 steps`, herdNet, herdDef, 20,
        i => ({ position: [random() * 38 - 19, 0, random() * 38 - 19], species: i % 16 === 0 ? 'wolf' : 'prey', energy: random() }));
    const seenFields = Array.from({ length: herdDef.obsDim }, (_, c) => herd.live.some(i => herd.obs[i * herdDef.obsDim + c] !== 0));
    check(`A4 Herd & Wolves: the comparison exercised every input — each of the ${herdDef.obsDim} observation fields is ` +
        `non-zero for some agent (${seenFields.filter(Boolean).length}/${herdDef.obsDim})`, herdDef.obsDim === 24 && seenFields.every(Boolean));

    // Watched failing: biases read from the neighbouring lane must put outputs out of tolerance.
    const shifted = mixed.brain.wgsl.replaceAll('][r % 4u];', '][(r + 1u) % 4u];');
    if (shifted === mixed.brain.wgsl) throw new Error('A4 bias-lane injection did not change the shader');
    mixed.sys.setBrain({ kind: 'stage', dims: mixed.brain.dims, buffers: mixed.brain.buffers, wgsl: shifted, destroy() {} });
    await mixed.sys.step();
    const injected = agree(mixedNet, mixedDef, await readF(mixed.sys.obs, 900), await readF(mixed.sys.act, 500), mixed.live);
    check(`A4 watched failing: biases read from the wrong lane put ${injected.bad} of ${injected.values} outputs out of tolerance`,
        injected.bad > 0);
    for (const run of [affine, odd, mixed, widest, herd]) { run.sys.destroy(); run.brain.destroy(); }

    // Extreme inputs, separate from agreement (§4.1). Tanh sees |z| far beyond 89, where FXC's tanh
    // returns NaN without the clamp, and sigmoid sees |z| in the hundreds. Outputs must stay finite and
    // in range; rows whose tanh layer saturated completely must still agree.
    {
        const net = mlpNet([3, 6, 'identity', 1000], [6, 6, 'tanh'], [6, 4, 'sigmoid', 100]);
        const def = mlpDef(3, 4, 100), run = await runSystem(net, def, 2);
        const pre = [net[0], { ...net[1], activation: 'identity' }], input = new Float32Array(3);
        let beyond89 = 0;
        const saturated = run.live.filter(i => {
            input.set(run.obs.subarray(i * 3, i * 3 + 3));
            const z = A.mlpForward(pre, input);
            if (z.some(v => Math.abs(v) >= 89)) beyond89++;
            return z.every(v => Math.abs(v) >= 15);
        });
        const inRange = run.live.every(i => run.act.subarray(i * 4, i * 4 + 4).every(v => v >= 0 && v <= 1));
        const r = agree(net, def, run.obs, run.act, saturated);
        check(`A4 extreme: tanh pre-activations reach |z| ≥ 89 in ${beyond89} of ${run.live.length} agents and sigmoid ` +
            `saturates; every output is finite and in [0, 1]; ${saturated.length} fully saturated rows agree (${errors(r)})`,
            run.error === null && beyond89 > 0 && inRange && saturated.length > 0 && r.bad === 0);
        run.sys.destroy(); run.brain.destroy();
    }
    {
        // Large finite relu outputs are not clipped. Cancellation makes |gpu − cpu| scale with the sum of
        // the terms' magnitudes, so the bound here is 1e-5 · (1 + |b| + Σ|w·x|).
        const net = mlpNet([3, 4, 'relu', 10000]), def = mlpDef(3, 4, 100), run = await runSystem(net, def, 2);
        const [{ weights: w, biases: b }] = net;
        let bad = 0, largest = 0;
        for (const i of run.live) {
            for (let r = 0; r < 4; r++) {
                let z = b[r], s = Math.abs(b[r]);
                for (let k = 0; k < 3; k++) { z += w[r * 3 + k] * run.obs[i * 3 + k]; s += Math.abs(w[r * 3 + k] * run.obs[i * 3 + k]); }
                const g = run.act[i * 4 + r];
                largest = Math.max(largest, g);
                if (!(Math.abs(g - Math.max(0, z)) <= TOL * (1 + s))) bad++;
            }
        }
        check(`A4 extreme: relu outputs up to ${largest.toFixed(0)} pass through unclipped and agree within the conditioning bound`,
            run.error === null && largest > 1000 && bad === 0);
        run.sys.destroy(); run.brain.destroy();
    }

    // Gating with thinkEvery 3 over 6 ticks: exactly the due 64-agent blocks (the last holds 8 rows) and
    // an off-cadence newborn are evaluated; every other row keeps its observations, actions and reason
    // bit-exactly, and dead rows keep the sentinel until spawned.
    {
        const def = mlpDef(5, 3, 200), net = mlpNet([5, 7, 'relu'], [7, 3, 'sigmoid']);
        const sys = new A.AgentSystem(engine, def, { thinkEvery: 3 }), brain = new A.MlpBrain(engine, net);
        sys.setBrain(brain);
        const alive = new Set(populate(sys, def, i => i === 7 || i === 130 || i === 199, () => ({ age: 1 })));
        device.queue.writeBuffer(sys.act, 0, new Float32Array(def.capacity * 3).fill(-7));
        fillReason(sys.reason, def.capacity, 99);
        let prevObs = await read(sys.obs, def.capacity * 5), prevAct = await read(sys.act, def.capacity * 3);
        let prevReason = await read(sys.reason, def.capacity);
        let ok = true, evaluated = 0, worst = { maxAbs: 0, maxRel: 0 };
        device.pushErrorScope('validation');
        for (let tick = 0; tick < 6; tick++) {
            if (tick === 2) { sys.spawn(130, { position: [0, 0, 0], species: 0 }); alive.add(130); }   // block 2 is due at 1 and 4
            await sys.step();
            const obsU = await read(sys.obs, def.capacity * 5), actU = await read(sys.act, def.capacity * 3);
            const reason = await read(sys.reason, def.capacity);
            const due = new Set([...alive].filter(i => (tick === 2 && i === 130) || (tick + Math.floor(i / 64)) % 3 === 0));
            const q = agree(net, def, new Float32Array(obsU.buffer), new Float32Array(actU.buffer), [...due]);
            worst = { maxAbs: Math.max(worst.maxAbs, q.maxAbs), maxRel: Math.max(worst.maxRel, q.maxRel) };
            if (q.bad) ok = false;
            for (let i = 0; i < def.capacity; i++) {
                const changed = [0, 1, 2, 3, 4].some(k => obsU[i * 5 + k] !== prevObs[i * 5 + k]);
                if (due.has(i)) { if (!changed || reason[i] !== 0) ok = false; continue; }
                if (changed || reason[i] !== prevReason[i] || [0, 1, 2].some(k => actU[i * 3 + k] !== prevAct[i * 3 + k])) ok = false;
            }
            evaluated += due.size;
            prevObs = obsU; prevAct = actU; prevReason = reason;
        }
        const error = await device.popErrorScope();
        const deadKept = [7, 199].every(i => prevReason[i] === 99 && new Float32Array(prevAct.buffer, i * 12, 3).every(v => v === -7));
        check(`A4 gating: over 6 ticks with thinkEvery 3, exactly the due blocks (8-row tail included) and an off-cadence ` +
            `newborn are evaluated (${evaluated} rows, ${errors(worst)}); all other rows keep obs, actions and reason bit-exactly; ` +
            'dead rows keep the sentinel', error === null && ok && deadKept);
        sys.destroy(); brain.destroy();
    }

    // Same-shape updates: one upload into the same buffer; no new stage or chain; the next step uses the
    // new weights; the caller's later edits do not leak; invalid updates change nothing.
    {
        const def = mlpDef(9, 5, 100), shape = [[9, 13, 'tanh'], [13, 6, 'relu'], [6, 11, 'identity'], [11, 5, 'sigmoid']];
        const w1 = mlpNet(...shape), w2 = mlpNet(...shape);
        const w2Copy = w2.map(l => ({ ...l, weights: l.weights.slice(), biases: l.biases.slice() }));
        const sys = new A.AgentSystem(engine, def), brain = new A.MlpBrain(engine, w1);
        sys.setBrain(brain);
        const live = populate(sys, def);
        sys.advance(def.dt);
        const chain = sys.getComputeStages(), think = chain.find(s => s.name === 'AgentSystem think'), weights = brain.buffers[0];
        device.pushErrorScope('validation');
        await sys.step();
        const first = agree(w1, def, await readF(sys.obs, 900), await readF(sys.act, 500), live);
        brain.setWeights(w2);
        for (const l of w2) { l.weights.fill(3); l.biases.fill(3); }
        await sys.step();
        const obs2 = await readF(sys.obs, 900), act2 = await readF(sys.act, 500);
        const second = agree(w2Copy, def, obs2, act2, live), stale = agree(w1, def, obs2, act2, live);
        let refused = 0;
        for (const bad of [
            w2Copy.map((l, k) => k === 3 ? { ...l, activation: 'tanh' } : l),
            w2Copy.slice(0, 3),
            w2Copy.map((l, k) => k === 3 ? { ...l, biases: Float32Array.of(0, 0, 0, 0, NaN) } : l),
        ]) { try { brain.setWeights(bad); } catch { refused++; } }
        await sys.step();
        const third = agree(w2Copy, def, await readF(sys.obs, 900), await readF(sys.act, 500), live);
        const error = await device.popErrorScope();
        check(`A4 updates: setWeights keeps the weights buffer, cached chain and think stage; the next step uses the new ` +
            `weights (${errors(second)}) and later edits to the caller's arrays do not leak`,
            error === null && first.bad === 0 && second.bad === 0 && stale.bad > 0 && brain.buffers[0] === weights &&
            sys.getComputeStages() === chain && chain.includes(think) && think.enabled);
        check(`A4 updates: ${refused} of 3 invalid updates (other activation, other depth, a NaN in the last bias) are ` +
            'refused, and the next step still matches the previous weights', refused === 3 && third.bad === 0);
        sys.destroy(); brain.destroy();
    }

    // Replacement: rules → MlpBrain → rules on a system with an app stage and the instance stage. Only
    // the think stage changes; reasons switch 7 → 0 → 7; the retired stages hold the destroyed MLP
    // weights but no cached chain reaches them.
    {
        const def = mlpDef(6, 2, 150, { mesh: 0, hoverMesh: 1, selectedMesh: 2, boundsRadius: 0.5 });
        const net = mlpNet([6, 8, 'tanh'], [8, 2, 'tanh']);
        const rules = () => new A.RuleBrain(`fn think(i: u32, species: u32, tick: u32) {
    setAct(i, ACT_TURN, 0.25); setAct(i, ACT_THROTTLE, -0.5); setReason(i, 7u);
}`);
        const sys = new A.AgentSystem(engine, def), brain = new A.MlpBrain(engine, net);
        sys.setBrain(rules());
        sys.addStepStage({ name: 'count', wgsl: 'fn step(i: u32, tick: u32) { agents[i].ids.w += 1u; }' });
        const rvis = new VisibilitySystem(engine, { maxInstances: def.capacity });
        rvis.setGeometry(mega);
        rvis.setInstanceCount(def.capacity);
        sys.toInstances(rvis);
        const live = populate(sys, def);
        sys.advance(def.dt);
        const others = sys.getComputeStages().filter(s => s.name !== 'AgentSystem think');
        const retired = [], problems = [];
        let steps = 0, pausedKept = true;
        device.pushErrorScope('validation');
        for (const [label, next] of [['rules', null], ['MLP', brain], ['rules again', rules()]]) {
            if (next) {
                const before = await read(sys.act, def.capacity * 2);
                retired.push(sys.getComputeStages().find(s => s.name === 'AgentSystem think'));
                sys.setBrain(next);
                if (!equal(await read(sys.act, def.capacity * 2), before)) pausedKept = false;   // no step, no inference
            }
            const chain = sys.getComputeStages(), at = chain.findIndex(s => s.name === 'AgentSystem think');
            const kept = chain.filter(s => s.name !== 'AgentSystem think');
            if (chain.filter(s => s.name === 'AgentSystem think').length !== 1 || !chain[at].enabled ||
                chain[at - 1].name !== 'AgentSystem sense' || chain[at + 1].name !== 'AgentSystem act' ||
                retired.some(s => s.enabled || chain.includes(s)) ||
                kept.length !== others.length || kept.some((s, k) => s !== others[k])) problems.push(`${label} chain`);
            for (let s = 0; s < 2; s++) { await sys.step(); steps++; }
            const act = await readF(sys.act, def.capacity * 2), reason = await read(sys.reason, def.capacity);
            if (next === brain) {
                const r = agree(net, def, await readF(sys.obs, def.capacity * 6), act, live);
                if (r.bad || live.some(i => reason[i] !== 0)) problems.push(`${label}: ${r.bad} outputs off`);
            } else if (live.some(i => act[i * 2] !== 0.25 || act[i * 2 + 1] !== -0.5 || reason[i] !== 7)) problems.push(`${label} actions`);
        }
        const rec = await read(sys.buffer, def.capacity * 16), recF = new Float32Array(rec.buffer);
        let age = 0;
        for (let s = 0; s < steps; s++) age = Math.fround(age + Math.fround(def.dt));
        if (live.some(i => rec[i * 16 + 15] !== steps || recF[i * 16 + 9] !== age)) problems.push('app/act stage counts');
        const rows = await readStorage(rvis.instanceBuffer, def.capacity * 24), rowsF = new Float32Array(rows.buffer);
        const twin = new Float32Array(def.capacity * 24), twinU = new Uint32Array(twin.buffer);
        for (let i = 0; i < def.capacity; i++) A.writeAgentInstanceRow(def, { data: recF, u32: rec }, i, { selected: -1, hovered: -1 }, twin, i, twinU);
        let rowsOff = 0;
        for (let w = 0; w < def.capacity * 24; w++) {
            const i = Math.floor(w / 24), k = w % 24, alive = (rec[i * 16 + 13] & 1) !== 0;
            if (alive && (k === 0 || k === 2 || k === 8 || k === 10)) { if (!(Math.abs(rowsF[w] - twin[w]) <= 2 ** -11 + 2 ** -24)) rowsOff++; }
            else if ((alive || k === 20) && rows[w] !== twinU[w]) rowsOff++;
        }
        if (rowsOff) problems.push(`${rowsOff} instance words`);
        const swapError = await device.popErrorScope();
        // The MLP is detached: destroy it, then render every cached chain length and keep stepping.
        brain.destroy();
        device.pushErrorScope('validation');
        for (const frames of [1, 2, 3]) {
            sys.advance(def.dt * frames);
            const chain = sys.getComputeStages();
            if (chain.some(s => !s.enabled || retired.includes(s))) problems.push(`chain of ${frames} reaches a retired stage`);
            engine.render(chain);
        }
        await sys.step();
        await device.queue.onSubmittedWorkDone();
        const afterError = await device.popErrorScope();
        check(`A4 replacement: rules → MLP → rules swaps only the think stage (sense, act, app, tick and instance stages ` +
            `identical, all running: ${steps} steps counted, instance rows equal the twin); reasons 7 → 0 → 7; a paused ` +
            `swap infers nothing; with the MLP destroyed, no cached chain reaches a retired stage${problems.length ? `; ${problems.join(', ')}` : ''}`,
            swapError === null && afterError === null && problems.length === 0 && pausedKept && retired.length === 2);
        sys.destroy(); rvis.destroy();
    }

    // Lifecycle: the documented order (system, then brain), idempotent destroy, refusals before any
    // stage exists, and a rebuilt system on the same engine.
    {
        const def = mlpDef(5, 3, 100), net = mlpNet([5, 7, 'relu'], [7, 3, 'sigmoid']);
        device.pushErrorScope('validation');
        const sys = new A.AgentSystem(engine, def), brain = new A.MlpBrain(engine, net);
        sys.setBrain(brain);
        populate(sys, def);
        await sys.step();
        sys.destroy();
        brain.destroy();
        brain.destroy();
        const rebuilt = await runSystem(net, def, 2);
        const r = agree(net, def, rebuilt.obs, rebuilt.act, rebuilt.live);
        let refused = 0;
        const refuses = (fn, pattern) => { try { fn(); } catch (e) { if (pattern.test(e.message)) refused++; } };
        refuses(() => rebuilt.sys.setBrain(brain), /after destroy/);
        refuses(() => brain.setWeights(net), /after destroy/);
        const otherAdapter = await navigator.gpu.requestAdapter(), otherDevice = await otherAdapter.requestDevice();
        const foreign = new A.MlpBrain({ device: otherDevice }, net);
        refuses(() => rebuilt.sys.setBrain(foreign), /same GPU device/);
        foreign.destroy();
        otherDevice.destroy();
        await rebuilt.sys.step();
        const still = agree(net, def, await readF(rebuilt.sys.obs, 500), await readF(rebuilt.sys.act, 300), rebuilt.live);
        rebuilt.sys.destroy();
        rebuilt.brain.destroy();
        await device.queue.onSubmittedWorkDone();
        const error = await device.popErrorScope();
        check(`A4 lifecycle: system-then-brain teardown and a second destroy() are clean; a destroyed brain and one from ` +
            `another device are refused (${refused}/3) and the system keeps its brain; a rebuilt system matches (${errors(r)})`,
            error === null && rebuilt.error === null && refused === 3 && r.bad === 0 && still.bad === 0);
    }

    // A21 (phase 7.2, docs/agents-species-brains-design.md §5, §10): one network per species as rows of
    // one weight table. Every live agent's actions must equal ITS species' network on its GPU
    // observations and fail every other species' network, on contiguous and interleaved layouts. A
    // one-species setWeights rewrites exactly that row; forcing every row to 0 is watched failing.
    {
        const names = ['prey', 'wolf', 'deer'];
        const def = A.defineAgents({ capacity: 300, world: { min: [-8, -8], max: [8, 8] }, cellSize: 4,
            species: names.map(name => ({ name, maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 4, radius: 0.4 })),
            sensors: [probe(5)], actions: A.Actions.custom({ name: 'hold', width: 3, wgsl: 'fn hold(i: u32) {}' }) });
        const shape = [[5, 7, 'tanh'], [7, 3, 'sigmoid']];
        const nets = Object.fromEntries(names.map(name => [name, mlpNet(...shape)]));
        const F = A.mlpLayout(nets.prey).floats;
        const words = net => new Uint32Array(A.packMlpWeights(net).buffer);
        const layouts = { contiguous: i => names[Math.floor(i / 100)], interleaved: i => names[i % 3] };
        const problems = [];
        let worst = { maxAbs: 0, maxRel: 0 }, injected = null;
        device.pushErrorScope('validation');
        for (const [layout, speciesOf] of Object.entries(layouts)) {
            const sys = new A.AgentSystem(engine, def), brain = A.MlpBrain.bySpecies(engine, def, nets);
            sys.setBrain(brain);
            const live = populate(sys, def, undefined, i => ({ species: speciesOf(i) }));
            const members = Object.fromEntries(names.map(name => [name, live.filter(i => speciesOf(i) === name)]));
            const judge = async (current, label) => {
                const obs = await readF(sys.obs, def.capacity * 5), act = await readF(sys.act, def.capacity * 3);
                const reason = await read(sys.reason, def.capacity);
                for (const name of names) {
                    const own = agree(current[name], def, obs, act, members[name]);
                    worst = { maxAbs: Math.max(worst.maxAbs, own.maxAbs), maxRel: Math.max(worst.maxRel, own.maxRel) };
                    if (own.bad) problems.push(`${layout} ${label}: ${name} ${own.bad}/${own.values} off its network`);
                    for (const other of names.filter(n => n !== name)) {
                        const cross = agree(current[other], def, obs, act, members[name]);
                        if (cross.bad < 0.99 * cross.values) problems.push(`${layout} ${label}: ${name} passes as ${other}`);
                    }
                    if (members[name].some(i => reason[i] !== 0)) problems.push(`${layout} ${label}: ${name} reason`);
                }
            };
            await sys.step();
            await sys.step();
            await judge(nets, 'rows');
            const wolf2 = mlpNet(...shape);
            brain.setWeights(wolf2, { species: 'wolf' });
            await sys.step();
            await judge({ ...nets, wolf: wolf2 }, 'after setWeights(wolf)');
            const table = await readStorage(brain.buffers[0], 3 * F);
            const expected = [words(nets.prey), words(wolf2), words(nets.deer)];
            if (!expected.every((w, r) => equal(table.subarray(r * F, (r + 1) * F), w))) problems.push(`${layout}: table rows`);
            if (layout === 'interleaved') {
                // Watched failing: every species reads row 0, prey's network.
                const flat = brain.wgsl.replace(/row = \d+u;/g, 'row = 0u;');
                if (flat === brain.wgsl) throw new Error('A21 row injection did not change the shader');
                sys.setBrain({ kind: 'stage', dims: brain.dims, buffers: brain.buffers, wgsl: flat, destroy() {} });
                await sys.step();
                const obs = await readF(sys.obs, def.capacity * 5), act = await readF(sys.act, def.capacity * 3);
                injected = { own: agree(wolf2, def, obs, act, members.wolf), asPrey: agree(nets.prey, def, obs, act, members.wolf) };
            }
            sys.destroy();
            brain.destroy();
        }
        const error = await device.popErrorScope();
        check(`A21: each of 3 species matches its own row of one weight table and fails the other two, contiguous and ` +
            `interleaved (${errors(worst)}); setWeights({ species: 'wolf' }) rewrites only that row` +
            (problems.length ? `; ${problems.slice(0, 4).join('; ')}` : ''), error === null && problems.length === 0);
        check(`A21 watched failing: with every row forced to 0, wolves read prey's network ` +
            `(${injected.own.bad} of ${injected.own.values} wolf outputs off their own network, ${injected.asPrey.bad} off prey's)`,
            injected.own.bad > 0 && injected.asPrey.bad === 0);
    }

    // A22 (phase 7.4, docs/agents-species-brains-design.md §6, §10): one think pass per brain, each gated
    // to its species by its own mask. Prey think with rules (constants, reason 7), wolves and deer with
    // networks of two different shapes (reason 0): every live agent must match its own brain and no
    // other, contiguous and interleaved. Moving wolves to the rules builds no stage, and the network
    // left serving nobody may then be destroyed. With the gate replaced by a plain read of the mask,
    // the last pass writes every species: watched failing.
    {
        const names = ['prey', 'wolf', 'deer'];
        const def = A.defineAgents({ capacity: 300, world: { min: [-8, -8], max: [8, 8] }, cellSize: 4,
            species: names.map(name => ({ name, maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 4, radius: 0.4 })),
            sensors: [probe(5)], actions: A.Actions.steering() });
        const nets = { wolf: mlpNet([5, 7, 'tanh'], [7, 2, 'tanh']), deer: mlpNet([5, 9, 'relu'], [9, 6, 'tanh'], [6, 2, 'sigmoid']) };
        const RULES = [0.25, -0.5];
        const ruleWGSL = `fn think(i: u32, species: u32, tick: u32) {
    setAct(i, ACT_TURN, ${RULES[0]}); setAct(i, ACT_THROTTLE, ${RULES[1]}); setReason(i, 7u);
}`;
        const GATE = 'if ((brainSpecies.x & (1u << a.ids.x)) == 0u) { return; }';
        const layouts = { contiguous: i => names[Math.floor(i / 100)], interleaved: i => names[i % 3] };
        const thinkPasses = sys => { sys.advance(def.dt); return sys.getComputeStages().filter(s => s.name === 'AgentSystem think'); };
        // Stages the engine builds while fn runs.
        const countBuilt = fn => {
            const own = Object.hasOwn(engine, 'createComputeStage'), original = engine.createComputeStage;
            let n = 0;
            engine.createComputeStage = function (...args) { n++; return original.apply(this, args); };
            try { fn(); } finally { if (own) engine.createComputeStage = original; else delete engine.createComputeStage; }
            return n;
        };
        // The production masks of the three passes, for the watched-failing dispatch.
        const masks = [0b001, 0b010, 0b100].map(bits => {
            const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            allocated.push(b);
            device.queue.writeBuffer(b, 0, new Uint32Array([bits, 0, 0, 0]));
            return b;
        });
        const problems = [];
        let worst = { maxAbs: 0, maxRel: 0 }, opened = null;
        device.pushErrorScope('validation');
        for (const [layout, speciesOf] of Object.entries(layouts)) {
            const sys = new A.AgentSystem(engine, def), rules = new A.RuleBrain(ruleWGSL);
            const wolfBrain = new A.MlpBrain(engine, nets.wolf), deerBrain = new A.MlpBrain(engine, nets.deer);
            const attached = [countBuilt(() => sys.setBrain(rules)), countBuilt(() => sys.setBrain(wolfBrain, { species: 'wolf' })),
                countBuilt(() => sys.setBrain(deerBrain, { species: ['deer'] }))];
            if (attached.join() !== '1,1,1') problems.push(`${layout}: attaching three brains built ${attached.join('/')} stages`);
            const live = populate(sys, def, undefined, i => ({ species: speciesOf(i) })), liveSet = new Set(live);
            const dead = Array.from({ length: def.capacity }, (_, i) => i).filter(i => !liveSet.has(i));
            const members = Object.fromEntries(names.map(name => [name, live.filter(i => speciesOf(i) === name)]));
            device.queue.writeBuffer(sys.act, 0, new Float32Array(def.capacity * 2).fill(-7));
            fillReason(sys.reason, def.capacity, 99);
            // brainOf: each species' brain, 'rules' or its network. Rules give the constants exactly and
            // reason 7; a network gives its own outputs within A4's tolerance, reason 0, and no other's.
            const judge = async (brainOf, label) => {
                const obs = await readF(sys.obs, def.capacity * 5), act = await readF(sys.act, def.capacity * 2);
                const reason = await read(sys.reason, def.capacity);
                const ruled = i => act[i * 2] === RULES[0] && act[i * 2 + 1] === RULES[1] && reason[i] === 7;
                for (const name of names) {
                    const own = brainOf[name], rows = members[name];
                    if (own === 'rules') {
                        if (!rows.every(ruled)) problems.push(`${layout} ${label}: ${name} off the rules`);
                    } else {
                        const r = agree(own, def, obs, act, rows);
                        worst = { maxAbs: Math.max(worst.maxAbs, r.maxAbs), maxRel: Math.max(worst.maxRel, r.maxRel) };
                        if (r.bad || rows.some(i => reason[i] !== 0)) problems.push(`${layout} ${label}: ${name} ${r.bad}/${r.values} off its network`);
                        if (rows.some(ruled)) problems.push(`${layout} ${label}: ${name} passes as rules`);
                    }
                    for (const net of Object.values(nets).filter(n => n !== own)) {
                        const cross = agree(net, def, obs, act, rows);
                        if (cross.bad < 0.99 * cross.values) problems.push(`${layout} ${label}: ${name} passes as another network`);
                    }
                }
                if (dead.some(i => reason[i] !== 99 || act[i * 2] !== -7 || act[i * 2 + 1] !== -7)) problems.push(`${layout} ${label}: dead rows written`);
            };
            await sys.step();
            await sys.step();
            const passes = thinkPasses(sys), [rulesPass, wolfPass, deerPass] = passes;
            const order = sys.getComputeStages().map(s => s.name), at = order.indexOf('AgentSystem think');
            if (passes.length !== 3 || !passes.every(p => p.enabled) || order.slice(at - 1, at + 4).join() !==
                ['AgentSystem sense', 'AgentSystem think', 'AgentSystem think', 'AgentSystem think', 'AgentSystem act'].join()) {
                problems.push(`${layout}: chain`);
            }
            await judge({ prey: 'rules', wolf: nets.wolf, deer: nets.deer }, 'three brains');
            const brains = [rules, wolfBrain, deerBrain];
            const codes = brains.map(brain => A.buildAgentThinkWGSL(sys.definition, brain.wgsl, sys.thinkEvery));
            if (layout === 'interleaved' && codes.every(code => code.includes(GATE))) {
                // Watched failing: the system's three passes, in its order, over its buffers and with its
                // masks, but each gate replaced by a plain read of the mask (the binding stays in use).
                const enc = device.createCommandEncoder();
                brains.forEach((brain, k) => {
                    const pipeline = device.createComputePipeline({ label: 'A22 ungated think', layout: 'auto',
                        compute: { module: device.createShaderModule({ code: codes[k].replace(GATE, '_ = brainSpecies.x;') }), entryPoint: 'cs_main' } });
                    const pass = enc.beginComputePass();
                    pass.setPipeline(pipeline);
                    pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
                        entries: [sys.buffer, sys.obs, sys.state, sys.act, sys.reason, sys.species, masks[k]]
                            .map((b, binding) => ({ binding, resource: { buffer: b } })) }));
                    if (brain.buffers?.length) pass.setBindGroup(1, device.createBindGroup({ layout: pipeline.getBindGroupLayout(1),
                        entries: brain.buffers.map((b, binding) => ({ binding, resource: { buffer: b } })) }));
                    pass.dispatchWorkgroups(Math.ceil(def.capacity / 64));
                    pass.end();
                });
                device.queue.submit([enc.finish()]);
                const obs = await readF(sys.obs, def.capacity * 5), act = await readF(sys.act, def.capacity * 2);
                opened = { asDeer: agree(nets.deer, def, obs, act, [...members.prey, ...members.wolf]),
                    ownWolf: agree(nets.wolf, def, obs, act, members.wolf), prey: members.prey.length,
                    preyRuled: members.prey.filter(i => act[i * 2] === RULES[0] && act[i * 2 + 1] === RULES[1]).length };
            }
            // Wolves back to the rules: the rules pass exists, so nothing is built and the wolf pass retires.
            const moved = countBuilt(() => sys.setBrain(rules, { species: 'wolf' })), after = thinkPasses(sys);
            if (moved !== 0 || after.length !== 2 || after[0] !== rulesPass || after[1] !== deerPass || wolfPass.enabled) {
                problems.push(`${layout}: moving wolves to the rules built ${moved} stage(s) or kept the wolf pass`);
            }
            // The wolf network serves nobody: destroy it, then run every cached chain length and a step.
            wolfBrain.destroy();
            for (const frames of [1, 2, 3]) {
                sys.advance(def.dt * frames);
                const chain = sys.getComputeStages();
                if (chain.includes(wolfPass) || chain.some(s => !s.enabled)) problems.push(`${layout}: a chain of ${frames} reaches the retired pass`);
                engine.render(chain);
            }
            await sys.step();
            await judge({ prey: 'rules', wolf: 'rules', deer: nets.deer }, 'wolves moved to the rules');
            sys.destroy();
            deerBrain.destroy();
        }
        const error = await device.popErrorScope();
        check(`A22: prey on rules, wolves and deer on networks of two shapes: every live agent matches its own brain and no ` +
            `other, contiguous and interleaved (${errors(worst)}); moving wolves to the rules builds no stage, and no chain ` +
            `reaches the destroyed wolf network` + (problems.length ? `; ${problems.slice(0, 4).join('; ')}` : ''),
            error === null && problems.length === 0);
        check(`A22 watched failing: with each gate replaced by a plain read of its mask, the last pass (deer's network) ` +
            `writes every species: ` + (opened ? `${opened.asDeer.values - opened.asDeer.bad} of ${opened.asDeer.values} prey ` +
            `and wolf outputs are deer's, ${opened.ownWolf.bad} of ${opened.ownWolf.values} wolf outputs left their own ` +
            `network, ${opened.preyRuled} of ${opened.prey} prey kept the rules` : 'the gate is missing from the think wrapper'),
            opened !== null && opened.asDeer.bad === 0 && opened.ownWolf.bad > 0 && opened.preyRuled === 0);
    }
}

allocated.splice(0).forEach(b => b.destroy());
await finish('All agent grid, sense, think, act, app-stage, arrow, instance, picking, label, inspect and MLP GPU checks passed.');
device.destroy();
function makeSystem(...args) { const s = new A.AgentSystem(...args); s.setBrain(new A.RuleBrain('fn think(i: u32, species: u32, tick: u32) {}')); return s; }
