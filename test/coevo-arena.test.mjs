import { test } from 'node:test';
import assert from 'node:assert/strict';
import { arenaLayout, catchFlags, exhaustiveCatchWGSL } from '../tools/agents/scenario/arena-layout.mjs';
import { WgslReflect } from '../node_modules/wgsl_reflect/wgsl_reflect.module.js';
import { buildAgentStepWGSL } from '../dist/agents.js';
import { containmentWGSL, respawnWGSL, arenaDefinition } from '../tools/agents/scenario/herd-wolves-arenas.mjs';
const geometry = { arenas: 2, wolves: 32, prey: 64, cols: 2, L: 24, gap: 10 };
test('U54 diagnostics: bounded allocation, shader composition, time/episode indexing and denominators', async () => {
    const { diagnosticLayout, diagnosticStateWGSL, diagnosticCatchWGSL, decodeDiagnostics } = await import('../tools/agents/coevolve-diagnostics.mjs');
    const lay = arenaLayout(geometry), d = diagnosticLayout(lay, { steps: 5, stride: 2 });
    assert.equal(d.samples, 3); assert.equal(d.words, 96);
    for (const shader of [diagnosticStateWGSL(lay, d), diagnosticCatchWGSL(lay, d)])
        assert.doesNotThrow(() => new WgslReflect(buildAgentStepWGSL(arenaDefinition(lay), 'diagnostic', shader)));
    const samples = new Float32Array(d.words), catches = new Uint32Array(d.words);
    samples.set([4, 2, 1, 2, 12, 4, 1, 6, 2, 2, 0.5, 0.4, 15], 16 * 3);
    catches[16 * 3 + 7] = 3;
    const result = decodeDiagnostics(lay, d, new Uint32Array(samples.buffer), catches);
    const h = result.episodes[1].history[1];
    assert.equal(h.tick, 2); assert.equal(h.preyWallShare, 0.25); assert.equal(h.wolfRestFraction, 0.5);
    assert.equal(h.preyMeanSpeed, 3); assert.equal(h.nearestPreyMean, 3); assert.equal(h.meanPenetration, 0.25);
    assert.equal(result.episodes[1].catches, 3); assert.equal(result.episodes[0].catches, 0);
    assert.equal(result.episodes[0].history[0].preyMeanSpeed, null);
    for (const change of [{ stride: 0 }, { steps: 0 }, { steps: 2 ** 32 }, { wallDistance: 20 }, { steps: 10000000, stride: 1 }])
        assert.throws(() => diagnosticLayout(lay, { steps: 5, stride: 2, ...change }), /diagnostics:/);
    assert.throws(() => decodeDiagnostics(lay, d, new Uint32Array(1), catches), /incomplete/);
    samples[0] = NaN;
    assert.throws(() => decodeDiagnostics(lay, d, new Uint32Array(samples.buffer), catches), /nonfinite/);
});
test('U50 layout refuses unsafe geometry/alignment/products before allocation', () => {
    const lay = arenaLayout(geometry);
    assert.equal(lay.count, 192); assert.equal(lay.wolfTotal, 64);
    for (const change of [{ gap: 8 }, { prey: 65 }, { wolves: 31 }, { L: NaN }, { L: 2 },
        { arenas: 2 ** 30 }, { cols: 3 }, { L: 1e20 }, { contacts: { iterations: 8, maxPush: 1 } }])
        assert.throws(() => arenaLayout({ ...geometry, ...change }), /coevolve:/);
});
test('U50 catch oracle: resting occlusion, dead, multi-prey, strict boundary, tile isolation', () => {
    const lay = arenaLayout(geometry), f = new Float32Array(lay.count * 16), w = new Uint32Array(f.buffer);
    const spawn = (i, x, radius, tile, rest = 0) => {
        f[i * 16] = x; f[i * 16 + 3] = radius; f[i * 16 + 11] = rest;
        w[i * 16 + 13] = 1; w[i * 16 + 15] = tile;
    };
    spawn(0, 0, 0.5, 0, 1); spawn(1, 0.5, 0.5, 0); spawn(32, 0, 0.5, 1);
    spawn(64, 0, 0.5, 0); spawn(65, 0, 0.5, 0); spawn(66, 1.5, 0.5, 0);
    let flags = catchFlags(f, lay, 0);
    assert.equal(flags[0], 0); assert.equal(flags[1], 512); assert.equal(flags[32], 0);
    assert.equal(flags[64], 256); assert.equal(flags[65], 256); assert.equal(flags[66], 0);
    w[1 * 16 + 13] = 0;
    flags = catchFlags(f, lay, 0); assert.equal(flags[64], 0);
    w[1 * 16 + 13] = 1; f[1 * 16] = 1; f[1 * 16 + 3] = 0.6; f[64 * 16 + 3] = 0.4;
    assert.equal(catchFlags(f, lay, 0)[64], 0, 'f32 .4 + .6 is exactly 1, strict boundary excludes it');
    assert.throws(() => catchFlags(f, lay, -1), /coevolve:/);
});
test('U50 contact-v2 WGSL composes with canonical 24-observation definition', () => {
    const lay = arenaLayout(geometry), def = arenaDefinition(lay);
    assert.equal(def.obsDim, 24); assert.equal(def.actDim, 2);
    const code = buildAgentStepWGSL(def, 'catches', exhaustiveCatchWGSL(lay, 0.05));
    assert.doesNotThrow(() => new WgslReflect(code));
    assert.ok(code.includes('catchSnapshot[j]'));
    assert.throws(() => exhaustiveCatchWGSL(lay, NaN), /coevolve:/);
});
test('U50 containment and respawn WGSL reflect valid entry signatures', () => {
    const lay = arenaLayout(geometry), def = arenaDefinition(lay);
    const contCode = buildAgentStepWGSL(def, 'containment', containmentWGSL(lay));
    assert.doesNotThrow(() => new WgslReflect(contCode));
    const respawnCode = buildAgentStepWGSL(def, 'respawn', respawnWGSL(lay));
    assert.doesNotThrow(() => new WgslReflect(respawnCode));
});
test('U50 catch oracle multi-wolf and cross-tile isolation stress', () => {
    const lay = arenaLayout(geometry), f = new Float32Array(lay.count * 16), w = new Uint32Array(f.buffer);
    const spawn = (i, x, z, radius, tile, rest = 0) => {
        f[i * 16] = x; f[i * 16 + 2] = z; f[i * 16 + 3] = radius; f[i * 16 + 11] = rest;
        w[i * 16 + 13] = 1; w[i * 16 + 15] = tile;
    };
    // 2 wolves in tile 0 touching single prey 64
    spawn(0, 5, 5, 0.6, 0); spawn(1, 5.2, 5, 0.6, 0);
    spawn(64, 5.1, 5, 0.4, 0);
    // Wolf in tile 1 near boundary (x=15), prey in tile 0 near boundary (x=14)
    spawn(32, 15, 5, 0.6, 1);
    const flags = catchFlags(f, lay, 0.05);
    // Both wolves in tile 0 get rest bit 512, prey 64 gets caught bit 256
    assert.equal(flags[0], 512); assert.equal(flags[1], 512); assert.equal(flags[64], 256);
    // Wolf in tile 1 must NOT catch prey in tile 0 despite geometric proximity
    assert.equal(flags[32], 0);
});