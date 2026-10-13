// node --test test/spikes/grid-policy/workload.test.mjs      (after node build.js and prepare.mjs)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { WgslReflect } from '../../../node_modules/wgsl_reflect/wgsl_reflect.module.js';
import * as A from '../../../dist/agents.js';
import * as B from '../../../dist/grid-policy-before.js';
import * as W from '../../../dist/grid-policy-window.js';
import { herdDef, PAD, nearestSensors, policyQueryWGSL, gpuCell, sortedGrid, crowding, scanned, matches } from './workload.mjs';

const source = name => readFileSync(new URL(name, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const earlier = { before: B.buildAgentGridQueryWGSL, window: W.buildAgentGridQueryWGSL };

test('the workload is the acceptance spike\'s, character for character', () => {
    const acceptance = source('../grid-order/run.js'), mine = source('./workload.mjs');
    const slice = (from, to) => {
        const begin = acceptance.indexOf(from), end = acceptance.indexOf(to, begin);
        assert.ok(begin >= 0 && end > begin, `the acceptance spike no longer contains ${from}`);
        return acceptance.slice(begin, end + to.length);
    };
    for (const block of [slice('const BASE_HALF', ';'), slice('const RULES = `', '}`;'),
        slice('const half = count', '    agents.upload();\n}')]) {
        assert.ok(mine.includes(block), `workload.mjs differs from the acceptance spike near: ${block.slice(0, 60)}`);
    }
});

test('each policy\'s query parses and differs only in its scan; cell+ is the query production ships', () => {
    const def = herdDef(4096, 64), sensors = nearestSensors(def);
    assert.deepEqual(sensors.map(s => [s.name, s.k, s.mask, s.base]), [['wolf', 1, 2, 0], ['prey', 3, 1, 1]]);
    for (const sensor of sensors) {
        const code = policy => policyQueryWGSL(policy, def, sensor, 4, earlier);
        const args = [def, sensor.mask, def.cellSize, sensor.k, { stride: 4, base: sensor.base }];
        for (const policy of ['first', 'cell', 'cell+', 'agent', 'agent+']) new WgslReflect(code(policy));
        assert.equal(code('first'), B.buildAgentGridQueryWGSL(...args));
        assert.ok(!code('first').includes('gridWindow') && !code('first').includes('tick'));
        assert.equal(code('agent'), W.buildAgentGridQueryWGSL(...args));
        assert.equal(code('cell'), code('agent').replace('gridWindow(i, tick) %', 'gridWindow(c, tick) %'));
        assert.notEqual(code('cell'), code('agent'));
        // The tighter loop keeps the window's key and budget, and everything after the scan's first lines.
        const tight = code('cell+'), from = text => text.slice(text.indexOf('if (j == i'));
        assert.ok(tight.includes('if (count > 64u) { at += gridWindow(c, tick) % count; }'));
        assert.ok(tight.includes('for (var left = min(count, 64u); left > 0u; left--) {'));
        assert.ok(!tight.includes('offsets[c] + at') && !tight.includes('gridWindow(i, tick)'));
        assert.equal(from(tight), from(code('cell')));
        assert.equal(code('agent+'), tight.replace('gridWindow(c, tick) %', 'gridWindow(i, tick) %'));
        assert.notEqual(code('agent+'), tight);
        // What was measured as cell+ is what ships: the same text, but for the name of the hash's parameter.
        assert.equal(A.buildAgentGridQueryWGSL(...args).replace('fn gridWindow(cell: u32', 'fn gridWindow(i: u32')
            .replace('let seed = cell ^', 'let seed = i ^'), tight);
    }
    assert.throws(() => policyQueryWGSL('other', def, sensors[0], 4, earlier), /Unknown policy/);
});

// One crowded cell (40 agents, budget 8) beside a sparse one, so windows wrap and policies differ.
function scene() {
    const def = A.defineAgents({ capacity: 48, world: { min: [0, 0], max: [32, 32] }, cellSize: 8, maxScanPerCell: 8,
        species: [{ name: 'prey', maxSpeed: 1, maxTurn: 1, accel: 1, senseRadius: 8, radius: .4 },
            { name: 'wolf', maxSpeed: 1, maxTurn: 1, accel: 1, senseRadius: 8, radius: .4 }],
        sensors: [A.Sensors.nearest({ name: 'wolf', species: ['wolf'], k: 1 }), A.Sensors.nearest({ name: 'prey', species: ['prey'], k: 3 })] });
    const records = new A.AgentRecords(def);
    for (let i = 0; i < 48; i++) {
        const crowded = i < 40;
        records.spawn(i, { species: i % 5 === 0 ? 'wolf' : 'prey', position: crowded
            ? [9 + 6 * A.rand01(i, 3, 0), 0, 9 + 6 * A.rand01(i, 3, 1)] : [17 + 6 * A.rand01(i, 3, 0), 0, 9 + 6 * A.rand01(i, 3, 1)] });
    }
    const rec = { data: records.data, u32: new Uint32Array(records.data.buffer) };
    const grid = A.buildPointGrid(def, rec.data, A.AGENT_FLOATS, 48, () => true, 8);
    return { def, rec, grid, sensors: nearestSensors(def) };
}
const top = (found, k, r2) => {
    const ranked = [...found].filter(([, d]) => d <= r2).sort((a, b) => a[1] - b[1] || a[0] - b[0]).map(([j]) => j);
    return Array.from({ length: k }, (_, r) => ranked[r] ?? PAD);
};

test('the CPU scan agrees with each version\'s own mirror, and the cell window is shared by a cell', () => {
    const { def, rec, grid, sensors } = scene();
    assert.deepEqual(crowding(def, rec, grid), { cells: 1, largest: 40, agents: 1 });
    const { overflow, ...library } = grid;
    assert.equal(overflow, 32);
    assert.deepEqual(sortedGrid(def, rec), library);
    for (let i = 0; i < 48; i++) {
        assert.equal(gpuCell(def, rec.data[i * 16], rec.data[i * 16 + 2]), A.pointCell(def, rec.data[i * 16], rec.data[i * 16 + 2]));
    }
    // first: the mirror before step 5. agent: phase 5.6's. cell: today's.
    const mirrors = { first: B.gridNearestNeighbours, agent: W.gridNearestNeighbours, cell: A.gridNearestNeighbours };
    let moved = 0;
    for (const tick of [0, 3, 37]) {
        for (const sensor of sensors) {
            for (let i = 0; i < 48; i++) {
                const lists = {};
                for (const [policy, mirror] of Object.entries(mirrors)) {
                    const expected = new Array(sensor.k).fill(PAD);
                    mirror(def, rec, grid, i, sensor.mask, 8, sensor.k, expected, tick);
                    lists[policy] = top(scanned(def, rec, grid, i, sensor, policy, tick), sensor.k, 64);
                    assert.deepEqual(lists[policy], expected, `${policy}, agent ${i}, tick ${tick}`);
                }
                if (String(lists.cell) !== String(lists.first)) moved++;
            }
        }
        // Whoever scans the crowded cell under the cell policy scans the same eight of its forty entries.
        const crowded = grid.cellCounts.indexOf(40), members = new Set(grid.cellBodies.subarray(grid.cellOffsets[crowded], grid.cellOffsets[crowded] + 40));
        const all = { mask: 3 }, slices = new Set(), perAgent = new Set();
        for (let i = 40; i < 48; i++) {
            const inCell = policy => [...scanned(def, rec, grid, i, all, policy, tick).keys()].filter(j => members.has(j)).sort((a, b) => a - b);
            assert.equal(inCell('cell').length, 8);
            slices.add(String(inCell('cell')));
            perAgent.add(String(inCell('agent')));
        }
        assert.equal(slices.size, 1);
        assert.ok(perAgent.size > 1);
    }
    assert.ok(moved > 0, 'the cell window never changed a list');
    assert.throws(() => scanned(def, rec, grid, 0, sensors[0], 'cell+', 0), /Unknown policy/);
});

test('matches() accepts float near-ties and nothing else', () => {
    const found = new Map([[4, 1], [9, 2], [2, 2.00000001], [7, 63.99999], [5, 70]]);
    assert.equal(matches(found, [4, 9, 2], 3, 64), 'exact');
    assert.equal(matches(found, [4, 2, 9], 3, 64), 'tie');           // 9 and 2 round together in f32
    assert.equal(matches(found, [4, 9, 2, PAD], 4, 64), 'tie');      // 7 sits on the radius
    assert.equal(matches(found, [4, 9, 7], 3, 64), 'wrong');         // skipped a nearer neighbour
    assert.equal(matches(found, [4, 9, 9], 3, 64), 'wrong');         // listed twice
    assert.equal(matches(found, [4, 9, 11], 3, 64), 'wrong');        // not an entry the policy scans
    assert.equal(matches(found, [4, 9, 2, 5], 4, 64), 'wrong');      // outside the radius
    assert.equal(matches(found, [4, PAD, PAD], 3, 64), 'wrong');     // dropped real neighbours
    assert.equal(matches(new Map(), [PAD, PAD], 2, 64), 'exact');
});
