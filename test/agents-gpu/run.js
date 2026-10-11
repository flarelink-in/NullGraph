// A1 kernel slice. Runs on both Node/Dawn and Chrome through the shared GPU harness.
import * as A from '../../dist/agents.js';
import { NullGraph, Camera } from '../../dist/index.js';
import { groundGrid, gridOccupancy } from '../../dist/debug-ui.js';
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
    const s = senseOnly(new A.AgentSystem(engine, definition));
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
const crowded = new A.AgentSystem(engine, crowdedDef);
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
const sensing = senseOnly(new A.AgentSystem(engine, senseDef));
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
const fixture = senseOnly(new A.AgentSystem(engine, fixtureDef));
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
const cadence = senseOnly(new A.AgentSystem(engine, cadenceDef, { thinkEvery: 3 }));
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
    const sparse = new A.AgentSystem(engine, sparseDef);
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
const acting = new A.AgentSystem(engine, actDef);
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
const poisoned = new A.AgentSystem(engine, poisonDef);
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
const aging = new A.AgentSystem(engine, agingDef, { thinkEvery: 3 });
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
    const replay = new A.AgentSystem(engine, replayDef, { thinkEvery: 3 });
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
const customSystem = new A.AgentSystem(engine, customDef);
for (let i = 0; i < 3; i++) customSystem.spawn(i, { position: [0, 0, 0], species: 0, frozen: i === 1 });
customSystem.kill(2);
const customExpected = new A.AgentRecords(customDef, new Float32Array(customSystem.records.data));
const customActions = new Float32Array([2, 2, 2]); A.stepAll(customDef, customExpected, customActions);
device.queue.writeBuffer(customSystem.act, 0, customActions); await customSystem.step();
check('custom act: CPU twin agrees, frozen ages only and dead stays byte-identical',
    equal(await read(customSystem.buffer, 48), customExpected.u32));
customSystem.destroy();

allocated.splice(0).forEach(b => b.destroy());
await finish('All phase 3.3 agent GPU checks passed.');
