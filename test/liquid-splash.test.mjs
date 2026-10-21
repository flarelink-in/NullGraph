import test from 'node:test';
import assert from 'node:assert/strict';
import liquidSplash, { COUNT, GRID, DT, initialLiquid, refinedLiquid, liquidQuality, liquidPostEffects, INTEGRATE } from '../tools/film/scenes/liquid-splash.mjs';

test('receiving pool has a circular footprint and a radial collision boundary', () => {
    const particles = initialLiquid(), positions = new Set();
    for (let i = 0; i < 800; i++) {
        const [x, y, z] = particles.subarray(i * 8, i * 8 + 3);
        assert.ok(Math.hypot(x, z) <= 1.425001);
        positions.add(`${x},${y},${z}`);
    }
    assert.equal(positions.size, 800);
    assert.ok(Math.hypot(particles[0], particles[2]) > 1.42);
    assert.match(INTEGRATE, /let radius = length\(p.xz\)/);
    assert.match(INTEGRATE, /radialVelocity > 0.0/);
    assert.doesNotMatch(INTEGRATE, /for \(var axis/);
});

test('liquid post chain keeps bloom subtle and antialiasing after tonemapping', () => {
    const effects = liquidPostEffects();
    assert.deepEqual(effects.map(effect => effect.name), ['Bloom', 'Tonemap', 'FXAA']);
    assert.equal(effects[0].intensity, 0.12);
    assert.equal(effects[1].curve, 'aces');
    assert.equal(effects[2].subpixel, 0.4);
    assert.notEqual(effects[0], liquidPostEffects()[0]);
});

test('high quality refines volume elements deterministically and validates its preset', () => {
    const fine = refinedLiquid();
    assert.equal(fine.length, COUNT * 8 * 8);
    assert.ok(fine.every(Number.isFinite));
    assert.deepEqual(fine, refinedLiquid());
    assert.deepEqual(liquidQuality({ quality: 'high' }), { high: true, count: 11928, grid: 144, dt: DT / 2 });
    assert.throws(() => liquidQuality({ quality: 'ultra' }), /quality/);
    const base = initialLiquid();
    for (let p = 0; p < COUNT; p++) for (let axis = 0; axis < 3; axis++) {
        let mean = 0;
        for (let child = 0; child < 8; child++) mean += fine[(p * 8 + child) * 8 + axis] / 8;
        assert.ok(Math.abs(mean - base[p * 8 + axis]) < 1e-6);
    }
});

test('liquid initialization has finite deterministic complete particle records', () => {
    const a = initialLiquid();
    assert.equal(a.length, COUNT * 8);
    assert.deepEqual(a, initialLiquid());
    assert.ok(a.every(Number.isFinite));
    for (let i = 0; i < COUNT; i++) {
        const p = a.subarray(i * 8, i * 8 + 3);
        assert.ok(p[0] >= -1.52 && p[0] <= 1.52);
        assert.ok(p[1] >= 0.075 && p[1] <= 3);
        assert.ok(p[2] >= -1.52 && p[2] <= 1.52);
    }
    assert.equal((1 / 60) / DT, 4);
});

test('liquid separates persistent simulation from replay-safe surface reconstruction', async () => {
    const oldUsage = globalThis.GPUBufferUsage;
    globalThis.GPUBufferUsage = { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4 };
    const submissions = [], uploads = [], buffers = [];
    const engine = {
        device: {
            createBuffer(config) {
                const buffer = { ...config, destroy() { this.destroyed = true; } };
                buffers.push(buffer); return buffer;
            },
            queue: { writeBuffer(...args) { uploads.push(args); }, async onSubmittedWorkDone() {} },
        },
        createPass: config => config,
        createBatch: (_pass, config) => ({ ...config, storageBuffer: { size: config.maxInstances * 4 } }),
        updateBatchData(batch, _data, count) { batch.instanceCount = count; },
        createComputeStage: config => config,
        setComputePassBindGroup(stage, entries) { stage.entries = entries; },
        render(sequence) { submissions.push(sequence); },
        updateCamera() {},
        clearPasses() {},
    };
    try {
        const scene = await liquidSplash({ engine, width: 640, height: 640,
            film: { track: (name, buffer, config) => ({ name, buffer, ...config }) } });
        assert.equal(scene.tracks[0].buffer, scene.checkpoint[0].buffer);
        assert.equal(scene.tracks[0].stride, 32);
        assert.equal(scene.tracks[0].every, 1);
        assert.equal(uploads[0][2].length, COUNT * 8);
        const draw = scene.draw();
        assert.match(draw[0].name, /derived surface/);
        assert.deepEqual(draw[0].dispatchCount, [Math.ceil(GRID / 4), Math.ceil(GRID / 4), Math.ceil(GRID / 4)]);
        assert.equal(submissions.length, 0);
        scene.view(null, 0);
        await scene.step();
        assert.equal(submissions[0].length, 2);
        const [density, integrate] = submissions[0];
        assert.equal(density.entries[0].resource.buffer, scene.checkpoint[0].buffer);
        assert.equal(integrate.entries[1].resource.buffer, scene.checkpoint[0].buffer);
        assert.equal(density.entries[1].resource.buffer, integrate.entries[0].resource.buffer);
        scene.destroy();
        assert.ok(buffers.every(buffer => buffer.destroyed));
        submissions.length = 0;
        const high = await liquidSplash({ engine, width: 1920, height: 1080, settings: { quality: 'high', postProcess: false },
            film: { track: (name, buffer, config) => ({ name, buffer, ...config }) } });
        assert.equal(high.dt, DT / 2);
        assert.equal(uploads.at(-1)[2].length, COUNT * 8 * 8);
        assert.equal(high.draw().length, 1);
        await high.prepareSurface();
        assert.equal(submissions.length, 36);
        assert.ok(submissions.every(s => s.length === 1 && s[0].dispatchCount[2] === 1));
        assert.match(submissions[1][0].shaderCode, /let id = raw \+ vec3u\(0,0,4\)/);
        assert.match(submissions[0][0].shaderCode, /11928u/);
        assert.match(high.draw()[0].name, /surface/);
        high.destroy();
    } finally {
        if (oldUsage === undefined) delete globalThis.GPUBufferUsage;
        else globalThis.GPUBufferUsage = oldUsage;
    }
});