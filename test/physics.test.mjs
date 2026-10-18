import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { gzipSync } from 'node:zlib';
import * as P from '../dist/physics.js';
import * as A from '../dist/agents.js';
import { standIn } from './stand-in-engine.mjs';
import { WgslReflect } from '../node_modules/wgsl_reflect/wgsl_reflect.module.js';

test('U46: declaration and refusals match §11 and resolve S9 defaults', () => {
    const baseLayout = { capacity: 16, stride: 16 };
    const baseOptions = { maxRadius: 0.6, world: { min: [-10, -10], max: [10, 10] } };

    // 1. Stride not positive multiple of 4
    assert.throws(
        () => P.validateDiscContacts({ ...baseLayout, stride: 10 }, baseOptions),
        /DiscContacts: stride 10 must be a positive multiple of 4\./,
    );
    assert.throws(
        () => P.validateDiscContacts({ ...baseLayout, stride: -4 }, baseOptions),
        /DiscContacts: stride -4 must be a positive multiple of 4\./,
    );

    // 2. Capacity
    assert.throws(
        () => P.validateDiscContacts({ ...baseLayout, capacity: 0 }, baseOptions),
        /DiscContacts: capacity 0 must be a positive integer\./,
    );

    // 3. Flags offsets (alive / fixed)
    assert.throws(
        () => P.validateDiscContacts({ ...baseLayout, alive: { offset: 2, mask: 1 } }, baseOptions),
        /DiscContacts: alive offset 2 must be a word of the record after its first four \(4–15\)\./,
    );
    assert.throws(
        () => P.validateDiscContacts({ ...baseLayout, alive: { offset: 16, mask: 1 } }, baseOptions),
        /DiscContacts: alive offset 16 must be a word of the record after its first four \(4–15\)\./,
    );
    assert.throws(
        () => P.validateDiscContacts({ ...baseLayout, fixed: { offset: 3, mask: 2 } }, baseOptions),
        /DiscContacts: fixed offset 3 must be a word of the record after its first four \(4–15\)\./,
    );
    assert.throws(
        () => P.validateDiscContacts({ ...baseLayout, fixed: { offset: 16, mask: 2 } }, baseOptions),
        /DiscContacts: fixed offset 16 must be a word of the record after its first four \(4–15\)\./,
    );

    // 4. Velocity offset
    assert.throws(
        () => P.validateDiscContacts({ ...baseLayout, velocity: { offset: 2 } }, baseOptions),
        /DiscContacts: velocity offset 2 must leave room for 3 floats \(4–13\)\./,
    );
    assert.throws(
        () => P.validateDiscContacts({ ...baseLayout, velocity: { offset: 14 } }, baseOptions),
        /DiscContacts: velocity offset 14 must leave room for 3 floats \(4–13\)\./,
    );

    // 5. Positive parameters (maxRadius, maxPush, dt)
    assert.throws(
        () => P.validateDiscContacts(baseLayout, { ...baseOptions, maxRadius: 0 }),
        /DiscContacts: maxRadius 0 must be positive\./,
    );
    assert.throws(
        () => P.validateDiscContacts(baseLayout, { ...baseOptions, maxPush: -0.1 }),
        /DiscContacts: maxPush -0.1 must be positive\./,
    );
    assert.throws(
        () => P.validateDiscContacts(baseLayout, { ...baseOptions, dt: 0 }),
        /DiscContacts: dt 0 must be positive\./,
    );

    // 6. Iterations outside 1..8
    assert.throws(
        () => P.validateDiscContacts(baseLayout, { ...baseOptions, iterations: 0 }),
        /DiscContacts: iterations 0 must be an integer from 1 to 8\./,
    );
    assert.throws(
        () => P.validateDiscContacts(baseLayout, { ...baseOptions, iterations: 9 }),
        /DiscContacts: iterations 9 must be an integer from 1 to 8\./,
    );

    // 7. Relaxation outside (0, 2)
    assert.throws(
        () => P.validateDiscContacts(baseLayout, { ...baseOptions, relaxation: 0 }),
        /DiscContacts: relaxation 0 must be above 0 and below 2\./,
    );
    assert.throws(
        () => P.validateDiscContacts(baseLayout, { ...baseOptions, relaxation: 2 }),
        /DiscContacts: relaxation 2 must be above 0 and below 2\./,
    );

    // 8. Negative compliance or damping outside [0, 1]
    assert.throws(
        () => P.validateDiscContacts(baseLayout, { ...baseOptions, compliance: -0.01 }),
        /DiscContacts: compliance -0.01 must be at least 0\./,
    );
    assert.throws(
        () => P.validateDiscContacts(baseLayout, { ...baseOptions, damping: -0.1 }),
        /DiscContacts: damping -0.1 must be from 0 to 1\./,
    );
    assert.throws(
        () => P.validateDiscContacts(baseLayout, { ...baseOptions, damping: 1.1 }),
        /DiscContacts: damping 1.1 must be from 0 to 1\./,
    );

    // 9. Cell size below bound
    // bound = 2 * 0.6 + (2 - 1) * 0.25 = 1.45
    assert.throws(
        () => P.validateDiscContacts(baseLayout, { ...baseOptions, cellSize: 1.4 }),
        /DiscContacts: cellSize 1\.4 is below 2 × maxRadius \+ \(iterations − 1\) × maxPush = 1\.45; contacts would be missed\./,
    );

    // 10. Records too small in resolveDiscContacts
    assert.throws(
        () => P.resolveDiscContacts(baseLayout, baseOptions, new Float32Array(10)),
        /DiscContacts: records must have STORAGE usage and hold capacity × stride floats\./,
    );

    // 11. Default resolution (S9 defaults)
    const resolved = P.validateDiscContacts(baseLayout, baseOptions);
    assert.equal(resolved.iterations, 2);
    assert.equal(resolved.relaxation, 1.75);
    assert.equal(resolved.maxPush, 0.25);
    assert.equal(resolved.maxScanPerCell, 32);
    assert.equal(resolved.damping, 1);
    assert.equal(resolved.mass, 'area');
    assert.equal(resolved.compliance, 0);
    assert.equal(resolved.tildeCompliance, 0);
    assert.equal(resolved.clamp, false);
    assert.equal(resolved.cellSize, 1.45);
    assert.equal(resolved.dt, 1 / 60);
});

test('U47: CPU mirror mechanics (C3, C4, C5, C8, damping, coincident pairs)', () => {
    const stride = 16;
    const layout = {
        capacity: 4,
        stride,
        alive: { offset: 13, mask: 1 },
        fixed: { offset: 13, mask: 2 },
        velocity: { offset: 4 },
    };
    const options = {
        world: { min: [-10, -10], max: [10, 10] },
        maxRadius: 0.6,
        dt: 1 / 60,
    };

    // Helper to initialize records
    function makeRecords(count = 4) {
        const rec = new Float32Array(count * stride);
        const u32 = new Uint32Array(rec.buffer);
        for (let i = 0; i < count; i++) {
            u32[i * stride + 13] = 1; // alive
        }
        return { rec, u32 };
    }

    // --- C3: One pair alone (ends exactly touching, weighted centre fixed) ---
    {
        // 'area' mass rule:
        const { rec } = makeRecords(2);
        // Disc 0: small (r = 0.4) at [0, 0, 0]
        rec[0] = 0; rec[1] = 0; rec[2] = 0; rec[3] = 0.4;
        // Disc 1: large (r = 0.6) at [0.5, 0, 0.3]
        rec[16] = 0.5; rec[17] = 0; rec[18] = 0.3; rec[19] = 0.6;

        const d0 = Math.hypot(0.5, 0.3);
        const reach = 0.4 + 0.6;
        const pen = reach - d0;
        const w0 = (0.6 / 0.4) ** 2, w1 = 1;
        const share0 = w0 / (w0 + w1), share1 = w1 / (w0 + w1);

        const res = P.resolveDiscContacts(
            { capacity: 2, stride, alive: layout.alive },
            { ...options, iterations: 1, relaxation: 1, compliance: 0, maxPush: 1, mass: 'area', clamp: false },
            rec.slice(),
        );

        const out = res.records;
        const finalDist = Math.hypot(out[16] - out[0], out[18] - out[2]);
        assert.ok(Math.abs(finalDist - reach) < 1e-6, `Final distance ${finalDist} should equal reach ${reach}`);

        const move0 = Math.hypot(out[0], out[2]);
        const move1 = Math.hypot(out[16] - 0.5, out[18] - 0.3);
        assert.ok(Math.abs(move0 - pen * share0) < 1e-6, `Body 0 move ${move0} should equal ${pen * share0}`);
        assert.ok(Math.abs(move1 - pen * share1) < 1e-6, `Body 1 move ${move1} should equal ${pen * share1}`);

        // Mass-weighted centre: m0 = 1/w0, m1 = 1/w1. m0 * Delta0 + m1 * Delta1 = 0
        const m0 = 1 / w0, m1 = 1 / w1;
        const cmX = (m0 * out[0] + m1 * out[16]) - (m0 * 0 + m1 * 0.5);
        const cmZ = (m0 * out[2] + m1 * out[18]) - (m0 * 0 + m1 * 0.3);
        assert.ok(Math.hypot(cmX, cmZ) < 1e-6, 'Mass-weighted centre of mass should not move');
    }

    {
        // 'equal' mass rule:
        const { rec } = makeRecords(2);
        rec[0] = 0; rec[1] = 0; rec[2] = 0; rec[3] = 0.4;
        rec[16] = 0.5; rec[17] = 0; rec[18] = 0.3; rec[19] = 0.6;

        const res = P.resolveDiscContacts(
            { capacity: 2, stride, alive: layout.alive },
            { ...options, iterations: 1, relaxation: 1, compliance: 0, maxPush: 1, mass: 'equal', clamp: false },
            rec.slice(),
        );

        const out = res.records;
        const finalDist = Math.hypot(out[16] - out[0], out[18] - out[2]);
        assert.ok(Math.abs(finalDist - 1.0) < 1e-6, 'Equal mass discs end exactly touching');
        const move0 = Math.hypot(out[0], out[2]);
        const move1 = Math.hypot(out[16] - 0.5, out[18] - 0.3);
        assert.ok(Math.abs(move0 - move1) < 1e-6, 'Equal mass discs move equal amounts');
    }

    // --- C4: Fixed and dead bodies ---
    {
        const { rec, u32 } = makeRecords(3);
        // Disc 0: live, unfixed at [0, 0, 0], r = 0.4
        rec[0] = 0; rec[1] = 0; rec[2] = 0; rec[3] = 0.4;
        // Disc 1: live, fixed at [0.5, 0, 0], r = 0.4 (reach 0.8, penetration 0.3)
        rec[16] = 0.5; rec[17] = 0; rec[18] = 0; rec[19] = 0.4;
        u32[16 + 13] = 1 | 2; // alive + fixed
        // Disc 2: dead at [0, 0, 0.2], r = 0.4
        rec[32] = 0; rec[33] = 0; rec[34] = 0.2; rec[35] = 0.4;
        u32[32 + 13] = 0; // dead

        const res = P.resolveDiscContacts(
            { capacity: 3, stride, alive: layout.alive, fixed: layout.fixed },
            { ...options, iterations: 1, relaxation: 1, compliance: 0, maxPush: 1, clamp: false },
            rec.slice(),
        );

        const out = res.records;
        // Fixed body 1 never moves (bit-for-bit)
        assert.equal(out[16], rec[16]);
        assert.equal(out[17], rec[17]);
        assert.equal(out[18], rec[18]);

        // Dead body 2 never moves (bit-for-bit)
        assert.equal(out[32], rec[32]);
        assert.equal(out[33], rec[33]);
        assert.equal(out[34], rec[34]);

        // Body 0 takes 100% of the correction from fixed body 1 (pushed from x=0 to x=-0.3)
        assert.ok(Math.abs(out[0] - (-0.3)) < 1e-6, `Body 0 position ${out[0]} should be -0.3`);
        assert.ok(Math.abs(out[2] - 0) < 1e-6, `Body 0 z position should be untouched by dead body 2`);
    }

    // --- C5: Compliance leaves part of overlap ---
    {
        const { rec } = makeRecords(2);
        rec[0] = 0; rec[1] = 0; rec[2] = 0; rec[3] = 0.5;
        rec[16] = 0.6; rec[17] = 0; rec[18] = 0; rec[19] = 0.5; // pen = 1.0 - 0.6 = 0.4

        const resHard = P.resolveDiscContacts(
            { capacity: 2, stride, alive: layout.alive },
            { ...options, iterations: 1, relaxation: 1, compliance: 0, maxPush: 1, clamp: false, mass: 'equal' },
            rec.slice(),
        );
        const distHard = Math.abs(resHard.records[16] - resHard.records[0]);
        assert.ok(Math.abs(distHard - 1.0) < 1e-6, 'Hard compliance leaves 0 overlap');

        const resSoft1 = P.resolveDiscContacts(
            { capacity: 2, stride, alive: layout.alive },
            { ...options, iterations: 1, relaxation: 1, compliance: 0.0001, maxPush: 1, clamp: false, mass: 'equal' },
            rec.slice(),
        );
        const distSoft1 = Math.abs(resSoft1.records[16] - resSoft1.records[0]);
        assert.ok(distSoft1 < 1.0 && distSoft1 > 0.6, 'Soft compliance leaves part of overlap');

        const resSoft2 = P.resolveDiscContacts(
            { capacity: 2, stride, alive: layout.alive },
            { ...options, iterations: 1, relaxation: 1, compliance: 0.001, maxPush: 1, clamp: false, mass: 'equal' },
            rec.slice(),
        );
        const distSoft2 = Math.abs(resSoft2.records[16] - resSoft2.records[0]);
        assert.ok(distSoft2 < distSoft1, 'Higher compliance leaves larger overlap (smaller separated distance)');
    }

    // --- C8: Clamping and push limit ---
    {
        const { rec } = makeRecords(2);
        // Disc near positive edge of world [-10, 10]:
        rec[0] = 9.8; rec[1] = 0; rec[2] = 0; rec[3] = 0.5;
        // Pushed to the right by disc at 9.2:
        rec[16] = 9.2; rec[17] = 0; rec[18] = 0; rec[19] = 0.5;

        const res = P.resolveDiscContacts(
            { capacity: 2, stride, alive: layout.alive },
            { ...options, iterations: 1, relaxation: 1, maxPush: 1, clamp: true },
            rec.slice(),
        );
        // World max is 10, radius is 0.5, clamped inset is 9.5
        assert.ok(res.records[0] <= 9.5 + 1e-6, `Clamped position ${res.records[0]} must stay <= 9.5`);

        // Push limit:
        const { rec: recPush } = makeRecords(2);
        recPush[0] = 0; recPush[1] = 0; recPush[2] = 0; recPush[3] = 0.6;
        recPush[16] = 0.2; recPush[17] = 0; recPush[18] = 0; recPush[19] = 0.6; // pen = 1.0 m

        const resPush = P.resolveDiscContacts(
            { capacity: 2, stride, alive: layout.alive },
            { ...options, iterations: 1, relaxation: 1, maxPush: 0.15, clamp: false, mass: 'equal' },
            recPush.slice(),
        );
        const move = Math.abs(resPush.records[0] - 0);
        assert.ok(Math.abs(move - 0.15) < 1e-6, `Move ${move} should be capped at maxPush 0.15`);
    }

    // --- Damping: removes only inward velocity along normal, leaves vy untouched ---
    {
        const { rec } = makeRecords(2);
        // Body 0 at [0, 0, 0] with velocity moving into body 1 on right: vx = 5.0, vy = 1.23, vz = 0
        rec[0] = 0; rec[1] = 0; rec[2] = 0; rec[3] = 0.5;
        rec[4] = 5.0; rec[5] = 1.23; rec[6] = 0.0;
        // Body 1 at [0.6, 0, 0] (push normal on body 0 is [-1, 0])
        rec[16] = 0.6; rec[17] = 0; rec[18] = 0; rec[19] = 0.5;
        rec[20] = -5.0; rec[21] = 4.56; rec[22] = 0.0;

        const res = P.resolveDiscContacts(
            { capacity: 2, stride, alive: layout.alive, velocity: layout.velocity },
            { ...options, iterations: 1, relaxation: 1, damping: 1.0, clamp: false, mass: 'equal' },
            rec.slice(),
        );
        // Body 0 was pushed left (nx = -1). Its velocity vx = 5 was along +X, which is opposing push (-5 along normal).
        // With damping beta=1, the inward component is removed: vx becomes 0!
        assert.ok(Math.abs(res.records[4]) < 1e-6, `Damped vx ${res.records[4]} should be 0`);
        assert.equal(res.records[5], rec[5], 'vy must remain bit-for-bit untouched');

        // Check outward velocity is untouched:
        const { rec: recOut } = makeRecords(2);
        // Body 0 already moving left (vx = -3.0):
        recOut[0] = 0; recOut[1] = 0; recOut[2] = 0; recOut[3] = 0.5;
        recOut[4] = -3.0; recOut[5] = 9.99; recOut[6] = 0.0;
        recOut[16] = 0.6; recOut[17] = 0; recOut[18] = 0; recOut[19] = 0.5;

        const resOut = P.resolveDiscContacts(
            { capacity: 2, stride, alive: layout.alive, velocity: layout.velocity },
            { ...options, iterations: 1, relaxation: 1, damping: 1.0, clamp: false, mass: 'equal' },
            recOut.slice(),
        );
        assert.ok(Math.abs(resOut.records[4] - (-3.0)) < 1e-6, 'Outward velocity component should be untouched');
        assert.equal(resOut.records[5], recOut[5], 'vy must remain bit-for-bit untouched');
    }

    // --- Coincident pairs open in opposite directions deterministically ---
    {
        const { rec } = makeRecords(2);
        rec[0] = 1.0; rec[1] = 0; rec[2] = 2.0; rec[3] = 0.5;
        rec[16] = 1.0; rec[17] = 0; rec[18] = 2.0; rec[19] = 0.5;

        const res = P.resolveDiscContacts(
            { capacity: 2, stride, alive: layout.alive },
            { ...options, iterations: 1, relaxation: 1, maxPush: 1, clamp: false, mass: 'equal' },
            rec.slice(),
        );
        const out = res.records;
        const dist = Math.hypot(out[16] - out[0], out[18] - out[2]);
        assert.ok(dist > 0.99, `Coincident pair must open apart, got distance ${dist}`);

        // Vectors from initial centre [1, 2]:
        const v0 = [out[0] - 1.0, out[2] - 2.0];
        const v1 = [out[16] - 1.0, out[18] - 2.0];
        assert.ok(Math.abs(v0[0] + v1[0]) < 1e-6 && Math.abs(v0[1] + v1[1]) < 1e-6,
            'Coincident pair moves in exactly opposite directions; centre of mass is fixed');
    }
});

test('U48: completeness and budget in the mirror (C1, C6, rotated windows)', () => {
    const stride = 16;
    const count = 32;
    const layout = { capacity: count, stride };
    const options = {
        world: { min: [-20, -20], max: [20, 20] },
        maxRadius: 0.6,
        maxPush: 0.25,
        iterations: 1,
        maxScanPerCell: 32,
        clamp: false,
    };

    // 1. C1: Completeness on random layouts against brute-force search
    let pseudoRand = 12345;
    const rand = () => { pseudoRand = (pseudoRand * 1664525 + 1013904223) >>> 0; return pseudoRand / 0xffffffff; };

    for (let trial = 0; trial < 64; trial++) {
        const rec = new Float32Array(count * stride);
        for (let i = 0; i < count; i++) {
            rec[i * stride] = (rand() * 2 - 1) * 15;     // x
            rec[i * stride + 2] = (rand() * 2 - 1) * 15; // z
            rec[i * stride + 3] = 0.3 + rand() * 0.3;   // radius in [0.3, 0.6]
        }

        // Brute-force contact count for body i
        const bruteCounts = new Uint32Array(count);
        for (let i = 0; i < count; i++) {
            const xi = rec[i * stride], zi = rec[i * stride + 2], ri = rec[i * stride + 3];
            for (let j = 0; j < count; j++) {
                if (j === i) continue;
                const xj = rec[j * stride], zj = rec[j * stride + 2], rj = rec[j * stride + 3];
                const d2 = (xi - xj) ** 2 + (zi - zj) ** 2;
                if (d2 < (ri + rj) ** 2) {
                    bruteCounts[i]++;
                }
            }
        }

        const res = P.resolveDiscContacts(layout, options, rec);
        for (let i = 0; i < count; i++) {
            assert.equal(res.contactCounts[i], bruteCounts[i],
                `Trial ${trial}, body ${i}: contact count ${res.contactCounts[i]} must match brute force ${bruteCounts[i]}`);
        }
    }

    // 2. C6: Budget and overflow
    const crowdedCount = 50;
    const crowdedRec = new Float32Array(crowdedCount * stride);
    for (let i = 0; i < crowdedCount; i++) {
        crowdedRec[i * stride] = 0;     // all in cell (0, 0)
        crowdedRec[i * stride + 2] = 0;
        crowdedRec[i * stride + 3] = 0.5;
    }
    const resCrowd = P.resolveDiscContacts(
        { capacity: crowdedCount, stride },
        { ...options, maxScanPerCell: 16 },
        crowdedRec.slice(),
    );
    // Overflow should be 50 - 16 = 34
    assert.equal(resCrowd.counters.overflow, 34, 'Skipped count must equal excess entries over budget');

    // 3. Window rotation: stepTick rotates windows
    const tick0 = P.resolveDiscContacts(
        { capacity: crowdedCount, stride },
        { ...options, maxScanPerCell: 16 },
        crowdedRec.slice(),
        undefined,
        0,
    );
    const tick1 = P.resolveDiscContacts(
        { capacity: crowdedCount, stride },
        { ...options, maxScanPerCell: 16 },
        crowdedRec.slice(),
        undefined,
        1,
    );

    // Positions after tick 0 and tick 1 should both have moved
    assert.ok(tick0.records[0] !== 0 || tick0.records[2] !== 0, 'Tick 0 bodies moved');
    assert.ok(tick1.records[0] !== 0 || tick1.records[2] !== 0, 'Tick 1 bodies moved');
    assert.notDeepEqual(tick0.records, tick1.records, 'Different ticks must select different budget windows');
});

test('Physics size line: GPU entry size is measured (13.2; no accepted ceiling yet)', async () => {
    const output = await build({
        entryPoints: ['src/extras/physics/index.ts'],
        bundle: true,
        minify: true,
        platform: 'browser',
        format: 'esm',
        external: ['gl-matrix'],
        write: false,
        logLevel: 'silent',
    });
    const bytes = gzipSync(output.outputFiles[0].contents).length;
    console.log(`physics.js: ${bytes} B min+gzip (CPU mirror + GPU solver + own grid)`);
    assert.ok(bytes > 0);
});

test('13.2: generated modules parse and GPU owner validates, caches, updates and cleans up', () => {
    globalThis.GPUBufferUsage = { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64 };
    const buffers = [], stages = [], writes = [];
    let failBind = false;
    const engine = {
        device: {
            limits: { maxStorageBufferBindingSize: 128 * 1024 * 1024, maxBufferSize: 256 * 1024 * 1024,
                maxComputeWorkgroupsPerDimension: 65535, maxComputeInvocationsPerWorkgroup: 256,
                maxComputeWorkgroupSizeX: 256, maxComputeWorkgroupStorageSize: 16384 },
            createBuffer(o) { const b = { ...o, destroyed: false, destroy() { this.destroyed = true; } }; buffers.push(b); return b; },
            queue: { writeBuffer(b, offset, data) { writes.push([...data]); } },
        },
        createComputeStage(o) {
            // The existing PointGrid scan-add uses syntax this parser does not support; the GPU suite compiles it.
            if (o.name.startsWith('DiscContacts')) {
                try { new WgslReflect(o.shaderCode); } catch (e) { throw new Error(`${o.name}: ${e.message}`); }
            }
            const s = { ...o, enabled: true }; stages.push(s); return s;
        },
        setComputePassBindGroup(s, entries) { if (failBind && s.name.startsWith('DiscContacts')) throw new Error('test binding failure'); s.entries = entries; },
    };
    const options = { capacity: 128, stride: 16, maxRadius: 0.6, world: { min: [-80, -80], max: [80, 80] } };
    const records = { size: 128 * 16 * 4, usage: 128, destroy() { throw new Error('caller buffer destroyed'); } };
    assert.throws(() => new P.DiscContacts(engine, { ...records, usage: 0 }, options), /records must have STORAGE/);
    assert.equal(buffers.length, 0);
    for (const fields of [{}, { alive: { offset: 13, mask: 1 }, fixed: { offset: 13, mask: 2 }, velocity: { offset: 4 } },
        { alive: { offset: 4, mask: 0x80000000 }, fixed: { offset: 5, mask: 2 }, velocity: { offset: 7 } }]) {
        const c = new P.DiscContacts(engine, records, { ...options, ...fields, clamp: true });
        const chain = c.getComputeStages();
        assert.equal(chain, c.getComputeStages());
        assert.equal(c.options.cellSize, 1.45);
        assert.equal(chain.filter(s => s.name.includes('DiscContacts gather')).length, 2);
        c.setParams({ compliance: 1e-4, maxPush: 0.1, damping: 0 });
        assert.deepEqual(writes.at(-1), [1.75, Math.fround(0.36), Math.fround(0.1), 0]);
        const previous = writes.length;
        assert.throws(() => c.setParams({ maxPush: 0.3 }), /contacts would be missed/);
        assert.equal(writes.length, previous);
        assert.equal(c.options.maxPush, 0.1);
        c.enabled = false; assert.ok(chain.every(s => !s.enabled));
        c.enabled = true; assert.ok(chain.every(s => s.enabled));
        c.destroy(); c.destroy(); assert.ok(chain.every(s => !s.enabled));
        assert.throws(() => c.setParams({}), /used after destroy/);
        assert.throws(() => c.getComputeStages(), /used after destroy/);
    }
    assert.ok(buffers.every(b => b.destroyed));
    const large = new P.DiscContacts(engine, { ...records, size: 1048576 * 16 * 4 }, {
        ...options, capacity: 1048576, world: { min: [-1280, -1280], max: [1280, 1280] },
    });
    assert.ok(large.options.cellSize > 1.45, 'Default cell grows for the million-body radix dispatch');
    large.destroy();
    failBind = true;
    assert.throws(() => new P.DiscContacts(engine, records, options), /test binding failure/);
    assert.ok(buffers.every(b => b.destroyed));
    assert.ok(stages.every(s => !s.enabled));
});

test('U49: stages, size and the seam (declared names, stage order, agents size ceiling)', async () => {
    // 1. Shaders parse and declared names check
    const resolved = P.validateDiscContacts(
        { capacity: 64, stride: 16, alive: { offset: 13, mask: 1 }, fixed: { offset: 13, mask: 2 }, velocity: { offset: 4 } },
        { maxRadius: 0.6, world: { min: [-10, -10], max: [10, 10] }, cellSize: 2, clamp: true }
    );
    const shaders = [
        P.buildContactGatherWGSL(resolved, 0),
        P.buildContactGatherWGSL(resolved, 1),
        P.buildContactApplyWGSL(resolved),
        P.buildContactCountWGSL(resolved),
        P.CONTACT_RESET_WGSL,
        P.CONTACT_TICK_WGSL,
    ];
    for (const code of shaders) {
        new WgslReflect(code);
        for (const [name] of code.matchAll(/\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b/g)) {
            assert.ok(code.includes(`const ${name} = `) || code.includes(`let ${name} = `) || code.includes(`var ${name} = `),
                `${name} is used but never declared in shader`);
        }
    }

    // 2. Stages array caching and zero frame allocations on DiscContacts
    const { engine } = standIn();
    const recordsBuf = engine.device.createBuffer({ size: 64 * 16 * 4, usage: GPUBufferUsage.STORAGE });
    const contacts = new P.DiscContacts(engine, recordsBuf, {
        capacity: 64, stride: 16, maxRadius: 0.6, world: { min: [-10, -10], max: [10, 10] }
    });
    const stages1 = contacts.getComputeStages();
    const stages2 = contacts.getComputeStages();
    assert.equal(stages1, stages2, 'DiscContacts.getComputeStages() must return the cached stage array');

    // 3. AgentSystem seam S-A: addStepStage({ name, stages })
    const def = A.defineAgents({
        capacity: 64,
        world: { min: [-10, -10], max: [10, 10] },
        cellSize: 5,
        species: [{ name: 'prey', radius: 0.5, maxSpeed: 1, maxTurn: 1, accel: 1, senseRadius: 5 }],
        sensors: [A.Sensors.self()],
    });
    const agents = new A.AgentSystem(engine, def);
    agents.setBrain(new A.RuleBrain('fn think(i: u32, species: u32, tick: u32) {}'));

    // Test refusal when neither wgsl nor stages provided
    assert.throws(() => agents.addStepStage({ name: 'empty' }), /must define fn step/);

    const custom1 = engine.createComputeStage({ name: 'custom stage 1', shaderCode: '@compute @workgroup_size(1) fn cs_main() {}', dispatchCount: [1, 1, 1] });
    const custom2 = engine.createComputeStage({ name: 'custom stage 2', shaderCode: '@compute @workgroup_size(1) fn cs_main() {}', dispatchCount: [1, 1, 1] });
    const custom3 = engine.createComputeStage({ name: 'custom stage 3', shaderCode: '@compute @workgroup_size(1) fn cs_main() {}', dispatchCount: [1, 1, 1] });

    const supplied = [custom1, custom2];
    agents.addStepStage({ name: 'stageAB', stages: supplied, wgsl: 'ignored because stages wins' });
    supplied.length = 0;
    agents.addStepStage({ name: 'compiled', wgsl: 'fn step(i: u32, tick: u32) {}' });
    agents.addStepStage({ name: 'stageC', stages: [custom3] });

    // Duplicate name refusal
    assert.throws(() => agents.addStepStage({ name: 'stageAB', stages: [custom1] }), /is added twice/);

    // Verify stage order in chains for 1-3 steps
    for (let n = 1; n <= 3; n++) {
        agents.advance(n * def.dt);
        const chain = agents.getComputeStages();
        for (let step = 0; step < n; step++) {
            const stepLen = chain.length / n;
            const stepStages = chain.slice(step * stepLen, (step + 1) * stepLen);
            const actIdx = stepStages.findIndex(s => s.name === 'AgentSystem act');
            assert.ok(actIdx >= 0, 'AgentSystem act must be in step');
            assert.equal(stepStages[actIdx + 1], custom1, 'custom1 must immediately follow act');
            assert.equal(stepStages[actIdx + 2], custom2, 'custom2 must follow custom1');
            assert.equal(stepStages[actIdx + 3]?.name, 'AgentSystem step compiled');
            assert.equal(stepStages[actIdx + 4], custom3, 'external and compiled stages preserve call order');
            assert.equal(stepStages[actIdx + 5]?.name, 'AgentSystem tick', 'AgentSystem tick must follow custom app stages');
        }
    }

    // Verify order in step() acting half
    const halves = agents['halves'];
    assert.ok(halves && halves.length === 2, 'agents must have two halves for async brains');
    const actingHalf = halves[1];
    const actHalfIdx = actingHalf.findIndex(s => s.name === 'AgentSystem act');
    assert.ok(actHalfIdx >= 0);
    assert.equal(actingHalf[actHalfIdx + 1], custom1);
    assert.equal(actingHalf[actHalfIdx + 2], custom2);
    assert.equal(actingHalf[actHalfIdx + 3]?.name, 'AgentSystem step compiled');
    assert.equal(actingHalf[actHalfIdx + 4], custom3);
    assert.equal(actingHalf[actHalfIdx + 5]?.name, 'AgentSystem tick');

    // 4. Bundled agents size meets Q3 ceiling (21,495 B) and carries NO physics code
    const agentsBundle = await build({
        entryPoints: ['src/extras/agents/index.ts'],
        bundle: true,
        minify: true,
        platform: 'browser',
        format: 'esm',
        external: ['gl-matrix'],
        write: false,
        logLevel: 'silent',
    });
    const agentsCode = agentsBundle.outputFiles[0].text;
    assert.ok(!agentsCode.includes('DiscContacts'), 'agents bundle must not contain DiscContacts');
    assert.ok(!agentsCode.includes('resolveDiscContacts'), 'agents bundle must not contain resolveDiscContacts');
    assert.ok(!agentsCode.includes('tieBreak'), 'agents bundle must not contain tieBreak');
    assert.ok(!agentsCode.includes('CONTACT_TICK_WGSL'), 'agents bundle must not contain CONTACT_TICK_WGSL');

    const agentsGzip = gzipSync(agentsBundle.outputFiles[0].contents).length;
    console.log(`agents.js size with S-A seam: ${agentsGzip} B min+gzip (ceiling 21,495 B)`);
    assert.ok(agentsGzip <= 21495, `agents.js is ${agentsGzip} B, must be <= 21,495 B ceiling`);

    // Teardown
    contacts.destroy();
    agents.destroy();
    assert.ok(custom1.enabled && custom2.enabled && custom3.enabled, 'external stages remain caller-owned');
    recordsBuf.destroy();
});
