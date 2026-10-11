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
await finish('All agent grid GPU checks passed.');
device.destroy();