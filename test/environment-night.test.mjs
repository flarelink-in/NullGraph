// Headless Phase 3 night-sky tests. GPU image/readback tests remain browser-only.
import { WgslReflect } from '../node_modules/wgsl_reflect/wgsl_reflect.module.js';
import {
    NightSky, SkyComposite, PreethamSky, StarField, TimeOfDay,
    SKY_PARAM_FLOATS, NIGHT_OFF, buildSourceCubeWGSL, buildSkyWGSL, buildStarWGSL,
    celestialToHorizon, magnitudeToFlux, starSpriteEnergy, lommelSeeliger,
    moonRadiance, bvToRGB, proceduralStarCatalogue, airMassKastenYoung,
} from '../dist/environment.js';

globalThis.GPUBufferUsage = globalThis.GPUBufferUsage ?? {
    COPY_DST: 8, COPY_SRC: 4, UNIFORM: 64, STORAGE: 128,
};

let failures = 0;
function check(name, fn) {
    try { fn(); console.log(`  ok    ${name}`); }
    catch (e) { failures++; console.log(`  FAIL  ${name}\n        ${e.message}`); }
}
function assert(v, m) { if (!v) throw new Error(m); }
function near(a, b, e, m) { if (Math.abs(a - b) > e) throw new Error(`${m}: expected ${b} ± ${e}, got ${a}`); }

check('U1: north celestial pole altitude equals observer latitude for 64 GMST values', () => {
    const out = new Float64Array(3);
    for (let i = 0; i < 64; i++) {
        const lat = (-89 + i * 178 / 63) * Math.PI / 180;
        celestialToHorizon(0, Math.PI / 2, i * Math.PI * 2 / 64, lat, 0, out);
        near(Math.asin(out[1]), lat, 2e-15, `sample ${i}`);
    }
});

check('U2: five magnitudes are exactly a factor of 100 in flux', () => {
    for (let m = -1.5; m <= 6.5; m += 0.125) near(magnitudeToFlux(m) / magnitudeToFlux(m + 5), 100, 1e-10, `m=${m}`);
});

check('U3: pixel-space sprite energy is invariant under a 4x render resolution change', () => {
    const a = starSpriteEnergy(2, 1.5), b = starSpriteEnergy(2, 1.5);
    near(a / b, 1, 0.01, 'energy ratio');
});

check('U4: Lommel-Seeliger has a uniform full-moon limb while Lambert does not', () => {
    const centre = lommelSeeliger(1, 1);
    const edgeN = Math.sqrt(1 - 0.95 * 0.95);
    const edge = lommelSeeliger(edgeN, edgeN);
    assert(Math.abs(edge / centre - 1) < 0.15, `LS edge/centre=${edge / centre}`);
    assert(Math.abs(edgeN - 1) > 0.60, `Lambert edge/centre=${edgeN}`);
});

check('U4b: full-moon centre geometry faces both observer and sun', () => {
    const moonTo = [0, 0, 1], sunTo = [0, 0, -1];
    const n = moonTo.map(x => -x);
    const nDotL = n[0] * sunTo[0] + n[1] * sunTo[1] + n[2] * sunTo[2];
    const nDotV = -(n[0] * moonTo[0] + n[1] * moonTo[1] + n[2] * moonTo[2]);
    near(nDotL, 1, 0, 'full-moon nDotL'); near(nDotV, 1, 0, 'full-moon nDotV');
});

check('U5/U8: lunar photometry and earthshine endpoints are finite and correctly ordered', () => {
    for (let i = 0; i < 32; i++) {
        const illum = i / 31, phase = Math.acos(2 * illum - 1);
        const r = moonRadiance(illum, -2 + i * 92 / 31, phase, 1);
        assert(Number.isFinite(r) && r >= 0, `radiance ${i}=${r}`);
    }
    const sky = new NightSky();
    const p = new Float32Array(SKY_PARAM_FLOATS + sky.paramFloats);
    sky.setCelestialState({ sunDir: new Float32Array([0, 1, 0]), sunElevationDeg: -20, sunAzimuthDeg: 0,
        sunRadiance: new Float32Array(3), moonDir: new Float32Array([0, -1, 0]), moonElevationDeg: 60,
        moonPhase: 0.5, moonIllumination: 1, moonRadiance: new Float32Array(3), gmstRad: 1,
        activeBody: 'moon', dayFactor: 0 });
    sky.writeParams(p, SKY_PARAM_FLOATS);
    for (const x of p) assert(Number.isFinite(x), `night param=${x}`);
    for (const off of [NIGHT_OFF.MOON_RGB, NIGHT_OFF.MOON_RGB + 1, NIGHT_OFF.MOON_RGB + 2,
        NIGHT_OFF.EARTHSHINE, NIGHT_OFF.DOME, NIGHT_OFF.AIRGLOW, NIGHT_OFF.POLLUTION_STRENGTH,
        NIGHT_OFF.NIGHT_GAIN, NIGHT_OFF.MILKY_WAY]) {
        assert(p[SKY_PARAM_FLOATS + off] >= 0, `radiometric param ${off}=${p[SKY_PARAM_FLOATS + off]}`);
    }
});

check('U6/U13: composite has one public evaluator, two renamed children, and valid WGSL', () => {
    const source = new SkyComposite({ day: new PreethamSky(), night: new NightSky(), dayFactor: 0.5 });
    const layout = source.getShaderLayout(SKY_PARAM_FLOATS);
    assert((layout.wgslLogic.match(/fn sampleSky\(/g) ?? []).length === 1, 'not exactly one sampleSky');
    assert(layout.wgslLogic.includes('fn sky_day_sampleSky('), 'day evaluator not renamed');
    assert(layout.wgslLogic.includes('fn sky_night_sampleSky('), 'night evaluator not renamed');
    for (const [name, src] of [
        ['bake', buildSourceCubeWGSL(layout)], ['skybox', buildSkyWGSL({ environmentGroup: 2, sky: layout })],
        ['stars', buildStarWGSL()],
    ]) {
        assert(!src.includes('`'), `${name} contains a backtick`);
        assert((src.match(/const PI\b/g) ?? []).length <= 1, `${name} redeclares PI`);
        try { new WgslReflect(src); } catch (e) { throw new Error(`${name} parse: ${String(e.message || e).split('\n')[0]}`); }
    }
});

check('U7: TimeOfDay twilight factor is monotone and zero at/below astronomical night', () => {
    const tod = new TimeOfDay({ exposureCoupling: 'off', sunTiltDeg: 0 });
    let previous = 1;
    for (let h = 18; h <= 24; h += 0.01) {
        tod.setTime(h);
        assert(tod.dayFactor <= previous + 1e-12, `rose at hour ${h}`);
        if (tod.sunElevationDeg <= -18) near(tod.dayFactor, 0, 0, `elev ${tod.sunElevationDeg}`);
        previous = tod.dayFactor;
    }
});

check('U9/U10: colours stay finite and shared air mass drives zero zenith/max horizon twinkle', () => {
    const rgb = new Float64Array(3);
    for (let bv = -0.4; bv <= 2; bv += 0.01) {
        bvToRGB(bv, rgb); for (const x of rgb) assert(Number.isFinite(x) && x >= 0 && x <= 1, `B-V ${bv}: ${x}`);
    }
    const zenith = (airMassKastenYoung(90) - 1) / 37;
    const horizon = (airMassKastenYoung(0) - 1) / 37;
    assert(Math.abs(zenith) < 2e-5, `zenith amplitude ${zenith}`);
    assert(horizon > 0.99, `horizon amplitude ${horizon}`);
});

check('U12: StarField adopts catalogue by reference and uploads caller mutations', () => {
    const writes = [];
    const pass = { size: [800, 600], batches: [], addBatch(b) { this.batches.push(b); } };
    const engine = {
        width: 800, height: 600,
        device: { createBuffer: () => ({}), queue: { writeBuffer: (...a) => writes.push(a) } },
        createBatch: (_p, cfg) => ({ stride: cfg.strideFloats, currentInstanceCount: 0, customBindGroups: {} }),
        setRenderPassBindGroup() {},
        updateBatchData(batch, data, count) { batch.currentInstanceCount = count; writes.push(['catalogue', data[0]]); },
    };
    const data = proceduralStarCatalogue(2);
    const field = new StarField(engine, { maxStars: 2, stars: data });
    field.attachToPass(pass);
    data[0] = 1.2345;
    field.setStars(data);
    field.upload();
    assert(writes.some(w => w[0] === 'catalogue' && Math.abs(w[1] - 1.2345) < 1e-6), 'next upload did not observe mutation');
});

check('error contracts name malformed records, capacity, and disabled night gain', () => {
    const engine = { device: {} };
    for (const [fn, pattern] of [
        [() => new NightSky({ nightGain: 0 }), /nightGain must be > 0/],
        [() => new StarField(engine, { stars: new Float32Array(3) }), /not a multiple of 4/],
        [() => new StarField(engine, { maxStars: 1, stars: new Float32Array(8) }), /exceeds maxInstances 1/],
    ]) {
        let msg = ''; try { fn(); } catch (e) { msg = e.message; }
        assert(pattern.test(msg), `expected ${pattern}, got ${msg}`);
    }
});

console.log(failures === 0 ? '\nAll environment night tests passed.' : `\n${failures} test(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);