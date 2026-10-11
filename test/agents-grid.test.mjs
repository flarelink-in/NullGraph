// C1 step 2 starts with U2/U3 CPU definitions. GPU A1 remains a separate, pending gate.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import * as A from '../dist/agents.js';

// Shared spatial code is internal, not a new package subpath.
const bundle = await build({ entryPoints: [fileURLToPath(new URL('../src/extras/spatial/gridMath.ts', import.meta.url))],
    bundle: true, write: false, format: 'esm', platform: 'node' });
const G = await import(`data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString('base64')}`);
let seed = 0x12345678;
const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
const definition = (over = {}) => A.defineAgents({ capacity: 64, world: { min: [-20, -20], max: [20, 20] },
    cellSize: 4, species: ['prey', 'wolf'].map(name => ({ name, maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 4, radius: 0.4 })),
    sensors: [A.Sensors.nearest({ k: 8 })], ...over });
const gridFor = (def, rec) => G.buildPointGrid(def, rec.data, A.AGENT_FLOATS, def.capacity, i => rec.isAlive(i), def.maxScanPerCell);

test('U2: 4096 positions, every grid line and closed/clamped boundary faces', () => {
    const def = definition();
    for (let i = 0; i < 4096; i++) {
        const x = -20 + random() * 40, z = -20 + random() * 40;
        const c = G.pointCell(def, x, z), cx = c % 10, cz = Math.floor(c / 10);
        assert.ok(x >= -20 + cx * 4 && x < -20 + (cx + 1) * 4);
        assert.ok(z >= -20 + cz * 4 && z < -20 + (cz + 1) * 4);
        assert.equal(G.pointCell(def, -18 + cx * 4, -18 + cz * 4), c);
    }
    for (let n = 0; n <= 10; n++) {
        assert.equal(G.pointCell(def, -20 + n * 4, -20), Math.min(n, 9));
        assert.equal(G.pointCell(def, -20, -20 + n * 4), Math.min(n, 9) * 10);
    }
    assert.equal(G.pointCell(def, 100, 100), 99);
    assert.equal(G.pointCell(def, -100, -100), 0);
});

test('U2: exclusive prefix over 64 arrays, zero, maximum counts and overflow rejection', () => {
    for (let run = 0; run < 64; run++) {
        const counts = Uint32Array.from({ length: 1025 }, () => run === 0 ? 0 : run === 1 ? 64 : Math.floor(random() * 64));
        const offsets = G.prefixOffsets(counts);
        let sum = 0;
        for (let i = 0; i < counts.length; i++) { assert.equal(offsets[i], sum); sum += counts[i]; }
        assert.equal(offsets[counts.length], sum);
    }
    assert.deepEqual([...G.prefixOffsets(new Uint32Array([0xffffffff]))], [0, 0xffffffff]);
    assert.throws(() => G.prefixOffsets(new Uint32Array([0xffffffff, 1])), /u32/);
});

test('U3: grid nearest equals brute force over 512 seeded scenes, species masks and deaths', () => {
    for (let scene = 0; scene < 512; scene++) {
        const def = definition(), rec = new A.AgentRecords(def);
        for (let i = 0; i < def.capacity; i++) {
            rec.spawn(i, { position: [-20 + random() * 40, 0, -20 + random() * 40], species: i % 2 });
            if (random() < 0.15) rec.kill(i);
        }
        const grid = gridFor(def, rec);
        assert.equal(grid.overflow, 0);
        assert.equal(grid.cellBodies.length, Array.from({ length: 64 }, (_, i) => rec.isAlive(i)).filter(Boolean).length);
        assert.equal(new Set(grid.cellBodies).size, grid.cellBodies.length);
        for (const j of grid.cellBodies) assert.ok(rec.isAlive(j));
        for (let i = 0; i < def.capacity; i++) {
            const expected = [], actual = [], mask = 1 + scene % 3, k = 1 + scene % 8;
            const n = A.nearestNeighbours(def, rec, i, mask, 4, k, expected);
            assert.equal(A.gridNearestNeighbours(def, rec, grid, i, mask, 4, k, actual), n);
            assert.deepEqual(actual, expected);
        }
    }
});

test('U3: ties prefer lower index even with reversed atomic scatter order', () => {
    const def = definition(), rec = new A.AgentRecords(def);
    [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]].forEach(([x, z], i) => rec.spawn(i, { position: [x, 0, z], species: 0 }));
    const grid = gridFor(def, rec);
    for (let c = 0; c < grid.cellCounts.length; c++) grid.cellBodies.subarray(grid.cellOffsets[c], grid.cellOffsets[c + 1]).reverse();
    const out = [];
    assert.equal(A.gridNearestNeighbours(def, rec, grid, 0, 1, 4, 3, out), 3);
    assert.deepEqual(out, [1, 2, 3]);
});

test('U2/U3: crowded cells retain every index, count excess once, and bound the scan', () => {
    const def = definition({ maxScanPerCell: 2 }), rec = new A.AgentRecords(def);
    for (let i = 0; i < 6; i++) rec.spawn(i, { position: [1, 0, 1], species: 0 });
    const grid = gridFor(def, rec), out = [];
    assert.equal(grid.overflow, 4);
    assert.equal(grid.cellBodies.length, 6);
    assert.equal(A.gridNearestNeighbours(def, rec, grid, 0, 1, 4, 3, out), 1);
    assert.deepEqual(out, [1]);
    assert.throws(() => A.gridNearestNeighbours(def, rec, grid, 0, 1, 5, 3, []), /radius/);
});