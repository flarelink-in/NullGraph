// Headless aerial-perspective and participating-media tests (docs/environment-aerial-design.md §8).
// GPU readback integration (T1-T6) lives in the browser harness; everything here runs on a mock
// device or on pure math.
import { WgslReflect } from '../node_modules/wgsl_reflect/wgsl_reflect.module.js';

// Node has no WebGPU globals; AerialPerspective references the usage enums when it allocates the
// froxel volume. The values are the standard bitmasks — the mock device ignores them, but the
// samplable-depth check reads TEXTURE_BINDING out of a usage mask, so they must be real.
globalThis.GPUTextureUsage = globalThis.GPUTextureUsage ?? {
    COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16,
};
globalThis.GPUBufferUsage = globalThis.GPUBufferUsage ?? {
    MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, INDEX: 32, VERTEX: 64, UNIFORM: 128,
    STORAGE: 256, INDIRECT: 512,
};

const CAMERA_STRUCT = `struct Camera {
    viewProj: mat4x4<f32>, view: mat4x4<f32>, proj: mat4x4<f32>,
    invViewProj: mat4x4<f32>, invView: mat4x4<f32>, invProj: mat4x4<f32>,
    eyePos: vec3<f32>, simTime: f32, zNear: f32, zFar: f32, fovRads: f32, aspect: f32,
    _padEnd: vec4<f32>,
};`;

import {
    AERIAL_APPLY_FUNCTIONS_WGSL, AerialPerspective, DEFAULT_HEIGHT_FOG, EARTH_ATMOSPHERE, HeightFog,
    accumulateAerialSegment, analyticFogInScatter, analyticFogTransmittance, applyAerialCPU,
    buildAerialApplyWGSL, buildAerialAxisWGSL, buildAerialLayout, buildAerialVolumeCPU,
    buildAerialVolumeWGSL, buildHeightFogLayout, combinedExtinction, createAerialAccumulator,
    createAerialColumn, froxelDistance, froxelSliceDistance, froxelW, marchAerialColumn,
    resetAerialAccumulator, resolveAerialConfig,
} from '../dist/environment.js';
import { buildMaterialFragment } from '../dist/materials.js';

let failures = 0;
function check(name, fn) {
    try { fn(); console.log(`  ok    ${name}`); }
    catch (e) { failures++; console.log(`  FAIL  ${name}\n        ${e.message}`); }
}
function assert(v, m) { if (!v) throw new Error(m); }
function near(a, b, e, m) { if (Math.abs(a - b) > e) throw new Error(`${m}: expected ${b} ± ${e}, got ${a}`); }
function relative(a, b) { return Math.abs(a - b) / Math.max(Math.abs(b), 1e-12); }

const SUN = { direction: [0.3, -0.8, 0.5], intensity: 1 };
const FORWARD = [0, 0, -1];

/** March one ray straight through [from, to] world units and return the per-channel state. */
function marchTo(media, altitudeKm, dir, fromUnits, toUnits, steps, config, options = {}) {
    const state = resetAerialAccumulator(options.state ?? createAerialAccumulator());
    const toKm = 1 / config.unitsPerKm;
    accumulateAerialSegment(altitudeKm, dir, SUN, media, fromUnits * toKm, toUnits * toKm,
        steps, state, options);
    return state;
}

// ---------------------------------------------------------------------------

check('U1: transmittance starts at 1 and decreases monotonically with distance, over 8 media', () => {
    const configs = [
        { atmosphere: EARTH_ATMOSPHERE, fog: null },
        { atmosphere: null, fog: DEFAULT_HEIGHT_FOG },
        { atmosphere: EARTH_ATMOSPHERE, fog: DEFAULT_HEIGHT_FOG },
        { atmosphere: EARTH_ATMOSPHERE, fog: { ...DEFAULT_HEIGHT_FOG, scaleHeight: 4 } },
        { atmosphere: { ...EARTH_ATMOSPHERE, mieScattering: 0, mieAbsorption: 0 }, fog: null },
        { atmosphere: { ...EARTH_ATMOSPHERE, rayleighScattering: [0, 0, 0] }, fog: null },
        { atmosphere: null, fog: { ...DEFAULT_HEIGHT_FOG, scattering: [8e-5, 6e-5, 4e-5] } },
        { atmosphere: EARTH_ATMOSPHERE, fog: { ...DEFAULT_HEIGHT_FOG, referenceHeight: 0.5 } },
    ];
    const config = resolveAerialConfig({ maxDistance: 32000, stepsPerSlice: 2 });
    const column = createAerialColumn(config.dims[2]);
    for (const [index, media] of configs.entries()) {
        // Zero distance: nothing between the camera and the surface, so nothing has happened yet.
        const atZero = marchTo(media, 0.5, FORWARD, 0, 0, 1, config);
        for (let c = 0; c < 3; c++) near(atZero.transmittance[c], 1, 0, `media ${index} channel ${c} at zero distance`);

        marchAerialColumn(0.5, FORWARD, SUN, media, config, column);
        let previous = 1.0000001;
        for (let z = 0; z < config.dims[2]; z++) {
            const t = column[z * 4 + 3];
            assert(t >= 0 && t <= 1, `media ${index} slice ${z}: transmittance ${t} outside [0, 1]`);
            assert(t <= previous, `media ${index}: transmittance rose from ${previous} to ${t} at slice ${z}`);
            previous = t;
        }
        assert(previous < 1, `media ${index} never attenuated anything over 32 km`);
    }
});

check('U2: in-scatter is zero at zero distance and non-decreasing along the ray', () => {
    const config = resolveAerialConfig({ maxDistance: 32000 });
    const media = { atmosphere: EARTH_ATMOSPHERE, fog: DEFAULT_HEIGHT_FOG };
    const atZero = marchTo(media, 0.2, FORWARD, 0, 0, 1, config);
    for (let c = 0; c < 3; c++) near(atZero.inScatter[c], 0, 0, `channel ${c} at zero distance`);

    const column = marchAerialColumn(0.2, FORWARD, SUN, media, config);
    const previous = [0, 0, 0];
    for (let z = 0; z < config.dims[2]; z++) {
        for (let c = 0; c < 3; c++) {
            const value = column[z * 4 + c];
            assert(value >= previous[c] - 1e-9, `in-scatter fell at slice ${z} channel ${c}`);
            previous[c] = value;
        }
    }
    assert(previous[2] > previous[0], `Rayleigh in-scatter is not blue-dominant: ${previous}`);
});

check('U3: slice composition — marching [near,d] then [d,D] equals marching [near,D], 64 pairs', () => {
    // THE correctness property of any slice-accumulated volume. It fails if the march restarts
    // from the camera per slice instead of continuing, which renders an almost identical image
    // and costs dims[2] times more.
    const config = resolveAerialConfig({ maxDistance: 32000 });
    const media = { atmosphere: EARTH_ATMOSPHERE, fog: DEFAULT_HEIGHT_FOG };
    const dir = [0.2, 0.1, -0.97];
    const length = Math.hypot(...dir);
    const unit = dir.map(v => v / length);
    let worst = 0;
    for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 8; j++) {
            const d = 200 + i * 1800;
            const D = d + 500 + j * 3500;
            const direct = marchTo(media, 0.4, unit, 0.1, D, 96, config);
            const first = marchTo(media, 0.4, unit, 0.1, d, 48, config);
            const second = marchTo(media, 0.4, unit, d, D, 48, config,
                { state: createAerialAccumulator() });
            for (let c = 0; c < 3; c++) {
                // T composes multiplicatively; L composes as L1 + T1 * L2 — the second segment's
                // radiance still has to travel back through the first.
                const composedT = first.transmittance[c] * second.transmittance[c];
                const composedL = first.inScatter[c] + first.transmittance[c] * second.inScatter[c];
                worst = Math.max(worst, relative(composedT, direct.transmittance[c]),
                    relative(composedL, direct.inScatter[c]));
                assert(relative(composedT, direct.transmittance[c]) <= 0.01,
                    `T at (${d}, ${D}) channel ${c}: ${composedT} vs ${direct.transmittance[c]}`);
                assert(relative(composedL, direct.inScatter[c]) <= 0.01,
                    `L at (${d}, ${D}) channel ${c}: ${composedL} vs ${direct.inScatter[c]}`);
            }
        }
    }
    assert(worst > 0, 'composition was exact, which means the segments were not actually marched');

    // The property above is about the segment primitive. This is about the COLUMN: every slice of
    // marchAerialColumn must be the running state of one continued march, not a fresh march from
    // the camera. At stepsPerSlice = 1 the two are wildly different quadratures, so an
    // implementation that restarts per slice cannot fake this.
    // Deliberately a STEEPLY RISING ray through the shallow default fog. Along a near-horizontal
    // ray the density barely varies, so a one-step march over the whole range lands within a few
    // percent of a 32-step one and the check below could not tell the implementations apart.
    // Climbing out of a 300 m scale height in the first kilometre is where they diverge.
    const steep = [0, 0.98, -0.199];
    const steepLength = Math.hypot(...steep);
    const climbing = steep.map(v => v / steepLength);

    const oneStep = resolveAerialConfig({ maxDistance: 32000, stepsPerSlice: 1 });
    const column = marchAerialColumn(0.05, climbing, SUN, media, oneStep);
    const running = resetAerialAccumulator(createAerialAccumulator());
    let previousKm = oneStep.nearDistance / oneStep.unitsPerKm;
    for (let z = 0; z < oneStep.dims[2]; z++) {
        const targetKm = froxelSliceDistance(z, oneStep) / oneStep.unitsPerKm;
        accumulateAerialSegment(0.05, climbing, SUN, media, previousKm, targetKm, 1, running);
        previousKm = targetKm;
        const meanT = (running.transmittance[0] + running.transmittance[1] + running.transmittance[2]) / 3;
        near(column[z * 4 + 3], meanT, 1e-6, `slice ${z} transmittance is not the continued state`);
        for (let c = 0; c < 3; c++) {
            near(column[z * 4 + c], running.inScatter[c],
                Math.abs(running.inScatter[c]) * 1e-5 + 1e-9, `slice ${z} in-scatter channel ${c}`);
        }
    }

    // And prove that check discriminates: a march RESTARTED from the camera at the last slice —
    // the cheap-looking mistake — lands somewhere else entirely.
    const last = oneStep.dims[2] - 1;
    const restarted = marchTo(media, 0.05, climbing, oneStep.nearDistance,
        froxelSliceDistance(last, oneStep), 1, oneStep, { state: createAerialAccumulator() });
    const restartedMeanT = (restarted.transmittance[0] + restarted.transmittance[1] + restarted.transmittance[2]) / 3;
    assert(relative(restartedMeanT, column[last * 4 + 3]) > 0.05,
        `a restarted march gives ${restartedMeanT} against the continued ${column[last * 4 + 3]}; ` +
        'this assertion cannot tell the two implementations apart');
});

check('U4: both distributions are monotonic, cover [near, maxDistance], and invert to 1e-6', () => {
    for (const distribution of ['linear', 'log']) {
        const config = resolveAerialConfig({ distribution, nearDistance: 0.1, maxDistance: 32000 });
        near(froxelDistance(0, 0.1, 32000, distribution), 0.1, 1e-9, `${distribution} at w=0`);
        near(froxelDistance(1, 0.1, 32000, distribution), 32000, 1e-3, `${distribution} at w=1`);
        let previous = -Infinity;
        for (let i = 0; i <= 1000; i++) {
            const w = i / 1000;
            const d = froxelDistance(w, 0.1, 32000, distribution);
            assert(d > previous, `${distribution} distance fell at w=${w}`);
            previous = d;
            assert(Math.abs(froxelW(d, 0.1, 32000, distribution) - w) <= 1e-6,
                `${distribution} round trip at w=${w}`);
        }
        // Texel centres, which is what the sampler actually lands on.
        for (let z = 0; z < config.dims[2]; z++) {
            const d = froxelSliceDistance(z, config);
            near(froxelW(d, 0.1, 32000, distribution), (z + 0.5) / config.dims[2], 1e-6,
                `${distribution} texel ${z} centre`);
        }
    }
    // The 'log' distribution genuinely packs slices near the camera; if it did not, it would be a
    // second name for 'linear' and the config choice would be decoration.
    const logHalf = froxelDistance(0.5, 0.1, 32000, 'log');
    const linearHalf = froxelDistance(0.5, 0.1, 32000, 'linear');
    assert(logHalf < linearHalf / 100, `log midpoint ${logHalf} is not packed against linear ${linearHalf}`);
});

check('U5: composition is color*T + inScatter — both mis-orderings are measurably wrong at range', () => {
    // A medium thick enough that T is genuinely far from 1. With T near 1 EVERY form agrees, which
    // is exactly why a fog bug survives testing in thin air and only shows up in the valley.
    const fog = { scattering: [1.0e-4, 1.0e-4, 1.0e-4], absorption: [1e-5, 1e-5, 1e-5],
        referenceHeight: 0, scaleHeight: 1e9, g: 0.6 };
    const config = resolveAerialConfig({ maxDistance: 32000 });
    const sun = { direction: SUN.direction, intensity: 10 };
    const state = resetAerialAccumulator(createAerialAccumulator());
    accumulateAerialSegment(0.3, FORWARD, sun, { atmosphere: null, fog }, 0, 12, 256, state,
        { geometry: 'planar', sunTransmittance: [1, 1, 1] });

    const T = (state.transmittance[0] + state.transmittance[1] + state.transmittance[2]) / 3;
    const L = state.inScatter;
    const surface = [0.8, 0.8, 0.8];
    assert(T > 0.05 && T < 0.5, `the test medium is not thick enough to discriminate: T = ${T}`);

    const right = applyAerialCPU(surface, L, T);
    for (let c = 0; c < 3; c++) near(right[c], surface[c] * T + L[c], 1e-7, `channel ${c}`);

    // Mis-ordering 1: forgetting the extinction entirely. This is the one §4.1 names — distant
    // objects GLOW instead of fading, because they keep full brightness and gain the air's
    // radiance on top.
    let brightest = 0;
    for (let c = 0; c < 3; c++) {
        const glow = surface[c] + L[c];
        assert(glow > right[c], `channel ${c}: the un-extinguished form ${glow} is not brighter than ${right[c]}`);
        brightest = Math.max(brightest, relative(glow, right[c]));
    }
    assert(brightest > 0.2, `the un-extinguished form is only ${(brightest * 100).toFixed(1)}% off at range`);

    // Mis-ordering 2: adding first and extinguishing after. Dimmer, by exactly L*(1-T) — naming
    // the residual is what turns "the fog looks wrong" into a one-line diagnosis.
    for (let c = 0; c < 3; c++) {
        const addFirst = (surface[c] + L[c]) * T;
        near(right[c] - addFirst, L[c] * (1 - T), Math.abs(L[c]) * 1e-5 + 1e-7,
            `channel ${c}: the add-then-extinguish residual is not L*(1-T)`);
        assert(addFirst < right[c], `channel ${c}: add-then-extinguish was not dimmer`);
    }

    // At zero distance every form agrees — T = 1, L = 0 — which is why a near-field-only test
    // passes with the order inverted.
    const atZero = applyAerialCPU(surface, [0, 0, 0], 1);
    for (let c = 0; c < 3; c++) near(atZero[c], surface[c], 1e-7, `pass-through at zero distance, channel ${c}`);
});

check('U6: the fullscreen pass and the injected layout carry byte-identical applyAerial bodies', () => {
    const config = resolveAerialConfig({});
    const apply = buildAerialApplyWGSL(config);
    const injected = buildAerialLayout(config).wgslLogic;

    const extract = (source, name) => {
        const start = source.indexOf(`fn ${name}(`);
        assert(start >= 0, `${name} is missing`);
        let depth = 0, i = source.indexOf('{', start);
        assert(i >= 0, `${name} has no body`);
        for (let j = i; j < source.length; j++) {
            if (source[j] === '{') depth++;
            else if (source[j] === '}' && --depth === 0) return source.slice(start, j + 1);
        }
        throw new Error(`${name} body is unbalanced`);
    };
    for (const name of ['applyAerial', 'composeAerial', 'applyAerialAt', 'aerialUVW']) {
        const a = extract(apply, name);
        const b = extract(injected, name);
        assert(a === b, `${name} differs between the two consumers:\n--- fullscreen ---\n${a}\n--- injected ---\n${b}`);
    }
    // Both must reach it through the same shared constant, not by two authors typing the same
    // thing — that is what stops them drifting on the next edit.
    assert(apply.includes(AERIAL_APPLY_FUNCTIONS_WGSL), 'the fullscreen pass does not use the shared block');
    assert(injected.includes(AERIAL_APPLY_FUNCTIONS_WGSL), 'the injected layout does not use the shared block');
    assert(extract(apply, 'composeAerial').includes('color * ap.a + ap.rgb'),
        'composeAerial is not color * T + inScatter');
});

check('U7: applyAerial lands BEFORE the PBR inline tonemap', () => {
    const aerial = buildAerialLayout(resolveAerialConfig({}));
    const { fragment } = buildMaterialFragment({ kind: 'pbr', groupIndex: 1, aerial });
    const applyAt = fragment.indexOf('applyAerialAt(');
    const tonemap = fragment.indexOf('finalColor / (finalColor + vec3<f32>(1.0))');
    const gamma = fragment.indexOf('pow(finalColor, vec3<f32>(1.0 / 2.2))');
    assert(applyAt >= 0, 'the PBR fragment never calls applyAerialAt');
    assert(tonemap >= 0 && gamma >= 0, 'the PBR fragment no longer tonemaps inline; this test needs updating');
    assert(applyAt < tonemap && applyAt < gamma,
        `applyAerialAt is at ${applyAt}, after the tonemap at ${tonemap} — it would fog an already-compressed value`);

    // Without the layout the fragment is byte-for-byte what it was before Phase 5.
    const plain = buildMaterialFragment({ kind: 'pbr', groupIndex: 1 }).fragment;
    assert(!plain.includes('applyAerialAt'), 'the aerial line leaked into a material that did not ask for it');

    // Every family takes it: aerial perspective is a property of the air, not of the shading model.
    for (const kind of ['basic', 'lambert', 'emissive', 'matcap', 'toon', 'pbr']) {
        assert(buildMaterialFragment({ kind, groupIndex: 1, aerial }).fragment.includes('applyAerialAt('),
            `${kind} does not apply aerial perspective`);
    }
});

check('U8: a batch cannot carry the injected layout inside a pass the fullscreen apply covers', () => {
    const engine = mockEngine();
    const aerial = new AerialPerspective(engine, { fog: true });
    aerial.getStages();

    const opaquePass = { name: 'opaque', batches: [] };
    const transparentPass = { name: 'transparent', batches: [] };
    const glass = { customBindGroups: {} };
    const rock = { customBindGroups: {} };

    // Registered first, then the pass is declared covered.
    aerial.registerBatch(rock, opaquePass, 'rock');
    let message = '';
    try { aerial.addCoveredPass(opaquePass); } catch (e) { message = e.message; }
    assert(/batch "rock" carries the injected aerial layout/.test(message), `covered-after-register: ${message}`);
    assert(/applied twice/.test(message), `the message does not say what goes wrong: ${message}`);

    // And the other order, because either call can come first at build time.
    const second = new AerialPerspective(mockEngine(), { fog: true });
    second.addCoveredPass(opaquePass);
    message = '';
    try { second.registerBatch(rock, opaquePass, 'rock'); } catch (e) { message = e.message; }
    assert(/batch "rock" carries the injected aerial layout/.test(message), `register-after-covered: ${message}`);

    // A transparent pass drawn AFTER the apply is the supported case and must not throw.
    second.registerBatch(glass, transparentPass, 'glass');
    assert(true);
});

check('U9: the froxel march and the analytic height-fog integral agree within 2%, 64 pairs', () => {
    // The unification claim (§6.2): the two integrate the same medium, which is what makes
    // switching between the v1.2 analytic path and the v2.0 volume safe. Compared in PLANAR
    // geometry with a fixed sun transmittance, because that is exactly what the closed form
    // assumes — anything else would be measuring curvature, not agreement.
    const fog = { scattering: [1.2e-5, 1.2e-5, 1.2e-5], absorption: [4e-6, 4e-6, 4e-6],
        referenceHeight: 0, scaleHeight: 2, g: 0.6 };
    const media = { atmosphere: null, fog };
    const config = resolveAerialConfig({ maxDistance: 8000, unitsPerKm: 1000 });
    const options = { geometry: 'planar', sunTransmittance: [1, 1, 1] };
    const sunTo = SUN.direction.map(v => -v / Math.hypot(...SUN.direction));

    let worstT = 0, worstL = 0;
    for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 8; j++) {
            const altitude = i * 0.35;
            const distance = 400 + j * 1050;
            const pitch = -0.4 + j * 0.1;
            const dir = [0, pitch, -1];
            const length = Math.hypot(...dir);
            const unit = dir.map(v => v / length);
            const cosTheta = unit[0] * sunTo[0] + unit[1] * sunTo[1] + unit[2] * sunTo[2];

            const state = marchTo(media, altitude, unit, 0, distance, 128, config,
                { ...options, state: createAerialAccumulator() });
            const T = analyticFogTransmittance(altitude, unit[1], distance / 1000, fog);
            const L = analyticFogInScatter(altitude, unit[1], distance / 1000, cosTheta, [1, 1, 1], fog);
            for (let c = 0; c < 3; c++) {
                worstT = Math.max(worstT, relative(state.transmittance[c], T[c]));
                worstL = Math.max(worstL, relative(state.inScatter[c], L[c]));
            }
        }
    }
    assert(worstT <= 0.02, `worst transmittance disagreement was ${(worstT * 100).toFixed(2)}%`);
    assert(worstL <= 0.02, `worst in-scatter disagreement was ${(worstL * 100).toFixed(2)}%`);
});

check('U10: the double-extinction bug, constructed deliberately and observed to fail', () => {
    // Built the way design-principles §13 requires: make the wrong thing, watch it be wrong,
    // THEN assert the right thing.
    const fog = DEFAULT_HEIGHT_FOG;
    const config = resolveAerialConfig({ maxDistance: 16000 });
    const combined = { atmosphere: EARTH_ATMOSPHERE, fog };
    const air = { atmosphere: EARTH_ATMOSPHERE, fog: null };
    const justFog = { atmosphere: null, fog };
    const march = m => marchTo(m, 0.05, FORWARD, 0.1, 12000, 256, config, { state: createAerialAccumulator() });

    const right = march(combined);
    const a = march(air);
    const b = march(justFog);

    // The trap that makes this bug survive every test: extinction sums in the EXPONENT, so the
    // two transmittances multiply out to exactly the combined one. Anything that only checks how
    // opaque the fog is passes with both systems running.
    for (let c = 0; c < 3; c++) {
        assert(relative(a.transmittance[c] * b.transmittance[c], right.transmittance[c]) < 0.02,
            `channel ${c}: T_air * T_fog should already equal T_combined`);
    }

    // The in-scatter does NOT compose that way. Each separate march attenuates its own radiance
    // by its own extinction only, so summing them over-counts by whatever the other medium would
    // have absorbed.
    let overCount = 0;
    for (let c = 0; c < 3; c++) {
        const wrong = a.inScatter[c] + b.inScatter[c];
        overCount = Math.max(overCount, relative(wrong, right.inScatter[c]));
        assert(wrong > right.inScatter[c],
            `channel ${c}: the stacked form ${wrong} did not exceed the combined ${right.inScatter[c]}`);
    }
    assert(overCount > 0.05,
        `the stacked form is only ${(overCount * 100).toFixed(2)}% off — this test would not catch the bug`);

    // And the right form: one medium set, one march. Adding a row must not change what the other
    // rows contribute except through the shared extinction.
    const extinction = combinedExtinction(0.1, combined);
    const airOnly = combinedExtinction(0.1, air);
    const fogOnly = combinedExtinction(0.1, justFog);
    for (let c = 0; c < 3; c++) {
        near(extinction[c], airOnly[c] + fogOnly[c], 1e-12, `combined extinction channel ${c}`);
    }
});

check('U11: the CPU volume mirrors the column march and reproduces Beer-Lambert exactly', () => {
    // Pinned against INVARIANTS, not against the kernel (design-principles §13.1) — a mirror and
    // its kernel drift in ways that reading them side by side does not catch.
    const config = resolveAerialConfig({ dims: [8, 8, 32], maxDistance: 16000 });
    const media = { atmosphere: EARTH_ATMOSPHERE, fog: DEFAULT_HEIGHT_FOG };
    const basis = { right: [1, 0, 0], up: [0, 1, 0], forward: [0, 0, -1], tanHalfFovY: Math.tan(0.5), aspect: 16 / 9 };
    const volume = buildAerialVolumeCPU(0.25, basis, SUN, media, config);
    const [nx, ny, nz] = config.dims;

    // Invariant 1: every column of the volume equals the column march down the same ray.
    const column = createAerialColumn(nz);
    let checked = 0;
    for (const [x, y] of [[0, 0], [3, 5], [7, 7], [4, 2]]) {
        const sx = (((x + 0.5) / nx) * 2 - 1) * basis.tanHalfFovY * basis.aspect;
        const sy = (1 - ((y + 0.5) / ny) * 2) * basis.tanHalfFovY;
        const dir = [basis.right[0] * sx, basis.up[1] * sy, basis.forward[2]];
        const length = Math.hypot(...dir);
        marchAerialColumn(0.25, dir.map(v => v / length), SUN, media, config, column);
        for (let z = 0; z < nz; z++) {
            for (let c = 0; c < 4; c++) {
                const got = volume[((z * ny + y) * nx + x) * 4 + c];
                near(got, column[z * 4 + c], Math.abs(column[z * 4 + c]) * 1e-5 + 1e-9,
                    `froxel (${x}, ${y}, ${z}) channel ${c}`);
                checked++;
            }
        }
    }
    assert(checked === 4 * nz * 4, `expected ${4 * nz * 4} comparisons, made ${checked}`);

    // Invariant 2: an empty medium leaves everything untouched — T is 1 and nothing scatters.
    const empty = { atmosphere: null, fog: { ...DEFAULT_HEIGHT_FOG, scattering: [0, 0, 0], absorption: [0, 0, 0] } };
    const emptyColumn = marchAerialColumn(0, FORWARD, SUN, empty, config);
    for (let z = 0; z < nz; z++) {
        near(emptyColumn[z * 4 + 3], 1, 1e-6, `empty medium transmittance at slice ${z}`);
        near(emptyColumn[z * 4], 0, 1e-9, `empty medium in-scatter at slice ${z}`);
    }

    // Invariant 3: a constant-density medium must reproduce exp(-sigma * d) to numerical
    // precision, because the quadrature is exact when the integrand is constant. A scale error
    // anywhere in the unit conversion shows up here and nowhere else.
    const uniform = { scattering: [1e-5, 1e-5, 1e-5], absorption: [1e-5, 1e-5, 1e-5],
        referenceHeight: 0, scaleHeight: 1e9, g: 0 };
    const flat = resolveAerialConfig({ maxDistance: 10000, dims: [4, 4, 32] });
    const state = marchTo({ atmosphere: null, fog: uniform }, 0, FORWARD, 0, 10000, 64, flat,
        { geometry: 'planar', state: createAerialAccumulator() });
    const expected = Math.exp(-2e-5 * 10000);
    for (let c = 0; c < 3; c++) near(state.transmittance[c], expected, expected * 1e-4, `Beer-Lambert channel ${c}`);
});

check('U12: every generated module parses, contains no backtick, and does not redeclare PI', () => {
    const config = resolveAerialConfig({});
    const logConfig = resolveAerialConfig({ distribution: 'log', depthAwareUpsampling: false });
    const layout = buildAerialLayout(config);
    const fogLayout = buildHeightFogLayout(config);
    const modules = [
        buildAerialVolumeWGSL({ config, atmosphere: {} }),
        buildAerialVolumeWGSL({ config: logConfig, atmosphere: null }),
        buildAerialApplyWGSL(config),
        buildAerialApplyWGSL(logConfig),
        buildAerialAxisWGSL(config),
        layout.wgslStructs + layout.wgslLogic,
        fogLayout.wgslStructs + fogLayout.wgslLogic,
    ];
    for (const [index, source] of modules.entries()) {
        assert(!source.includes('`'), `module ${index} contains a backtick`);
        assert((source.match(/const PI\b/g) ?? []).length <= 1, `module ${index} redeclares PI`);
    }
    // Only the standalone modules are whole shaders; the layouts are fragments spliced into a
    // host, so they are parsed inside a minimal host that supplies what they reference.
    for (const source of modules.slice(0, 4)) new WgslReflect(source);
    const host = `${CAMERA_STRUCT}
@group(0) @binding(0) var<uniform> camera: Camera;
${layout.wgslStructs}
${layout.wgslLogic}
@fragment fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(applyAerialAt(vec3<f32>(1.0), vec3<f32>(0.0)), 1.0);
}`;
    new WgslReflect(host);
    new WgslReflect(`${CAMERA_STRUCT}
@group(0) @binding(0) var<uniform> camera: Camera;
${fogLayout.wgslStructs}
${fogLayout.wgslLogic}
@fragment fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(applyAerialAt(vec3<f32>(1.0), vec3<f32>(0.0)), 1.0);
}`);
    // The froxel kernel must be one ray per lateral froxel looping over z (§4.3), not one thread
    // per froxel. The difference is invisible in the output and costs dims[2] times more.
    const kernel = buildAerialVolumeWGSL({ config, atmosphere: {} });
    assert(/@workgroup_size\(4, 4\)/.test(kernel), 'the froxel kernel is not a 2D dispatch over lateral froxels');
    assert(/for \(var z = 0u; z < 32u/.test(kernel), 'the froxel kernel does not loop over slices');
});

check('errors: every §10 refusal fires, names the offender, and states the way out', () => {
    const cases = [
        [() => resolveAerialConfig({ dims: [32, 30, 32] }), /volume dimensions 32x30x32 must be multiples of 4/],
        [() => resolveAerialConfig({ maxDistance: 5, nearDistance: 10 }), /maxDistance 5 must exceed the camera near plane 10/],
        [() => new AerialPerspective(mockEngine(), {}), /no medium\. Construct with an Atmosphere/],
        [() => new AerialPerspective(mockEngine(), { fog: { ...DEFAULT_HEIGHT_FOG, g: 1 } }), /singular/],
        [() => new AerialPerspective(mockEngine(), { fog: { ...DEFAULT_HEIGHT_FOG, scattering: [-1, 0, 0] } }), /must be non-negative/],
    ];
    for (const [run, pattern] of cases) {
        let message = '';
        try { run(); } catch (e) { message = e.message; }
        assert(pattern.test(message), `expected ${pattern}, got "${message}"`);
    }

    // getApplyPass before getStages: the volume would be sampled before anything wrote it.
    const engine = mockEngine();
    const aerial = new AerialPerspective(engine, { fog: true });
    let message = '';
    try { aerial.getApplyPass({ source: {}, depth: {}, colorAttachments: [] }); } catch (e) { message = e.message; }
    assert(/froxel volume has never been built/.test(message), `unbuilt volume: ${message}`);
    assert(/Splice aerial\.getStages\(\) before the scene pass/.test(message), `no way out given: ${message}`);

    // No samplable depth: the §9.1 refusal must name the flag AND the zero-change fallback.
    aerial.getStages();
    message = '';
    try { aerial.getApplyPass({ source: {}, colorAttachments: [] }); } catch (e) { message = e.message; }
    assert(/needs a samplable depth texture/.test(message), `samplable depth: ${message}`);
    assert(/samplableDepth: true/.test(message) && /offscreen pass/.test(message),
        `the message omits the flag or the fallback: ${message}`);

    // A context created WITH the flag is accepted, which is what makes the refusal above a
    // configuration error rather than a dead end.
    const samplable = mockEngine({ depthSamplable: true });
    const ok = new AerialPerspective(samplable, { fog: true });
    ok.getStages();
    ok.getApplyPass({ source: {}, colorAttachments: [] });
});

check('§6.2: HeightFog and the froxel volume refuse each other in both construction orders', () => {
    const a = mockEngine();
    const volume = new AerialPerspective(a, { fog: true });
    let message = '';
    try { new HeightFog(a); } catch (e) { message = e.message; }
    assert(/extinction would be applied twice/.test(message), `fog after volume: ${message}`);
    assert(/Move the fog parameters onto the volume/.test(message), `no way out given: ${message}`);

    const b = mockEngine();
    const fog = new HeightFog(b);
    message = '';
    try { new AerialPerspective(b, { fog: true }); } catch (e) { message = e.message; }
    assert(/extinction would be applied twice/.test(message), `volume after fog: ${message}`);

    // Destroying one releases the claim, so a scene really can switch paths at runtime.
    volume.destroy();
    const replacement = new HeightFog(a);
    assert(replacement.getShaderLayout().wgslLogic.includes('applyAerialAt'),
        'the analytic layout does not define the seam');
    // Both implementations define the SAME entry point, which is what lets the cheap path ship
    // first and the volume replace it without touching a material.
    assert(fog.getShaderLayout().wgslLogic.includes('fn applyAerialAt(')
        && buildAerialLayout(resolveAerialConfig({})).wgslLogic.includes('fn applyAerialAt('),
        'the two aerial implementations do not share an entry point');
});

// ---------------------------------------------------------------------------

function mockEngine(ctx = {}) {
    const device = {
        createBuffer: () => ({ destroy() {} }),
        createTexture: () => ({ destroy() {}, createView: () => ({}), usage: 0 }),
        createSampler: () => ({}),
        queue: { writeBuffer() {} },
    };
    return {
        device,
        ctx: {
            cameraUniformBuffer: {},
            depthTexture: { createView: () => ({}), usage: 0 },
            depthSamplable: false,
            ...ctx,
        },
        createComputeStage: () => ({ enabled: true }),
        setComputePassBindGroup() {},
        createPass: config => ({ ...config, batches: [] }),
        createBatch: () => ({ customBindGroups: {} }),
        setRenderPassBindGroup() {},
        updateBatchData() {},
    };
}

console.log(failures === 0 ? '\nAll aerial-perspective tests passed.' : `\n${failures} test(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);
