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
// Phase-local fixtures do not yet include integration.
function senseOnly(s) { return s; }
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

allocated.splice(0).forEach(b => b.destroy());
await finish('All phase 3.2 agent GPU checks passed.');
