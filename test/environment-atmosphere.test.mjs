// Headless physical-atmosphere tests. GPU readback integration remains in the browser harness.
import { WgslReflect } from '../node_modules/wgsl_reflect/wgsl_reflect.module.js';
import {
    EARTH_ATMOSPHERE, SKY_PARAM_FLOATS, NightSky, SkyComposite,
    atmosphereSunRadiance, buildAtmosphereBakeLayout, buildAtmosphereSkyLayout,
    buildMultipleScatteringWGSL, buildSkyViewWGSL, buildSourceCubeWGSL, buildSkyWGSL,
    buildTransmittanceWGSL, cornetteShanksPhase, horizonAngleFromSkyViewV,
    rayleighPhase, resolveAtmosphereConfig, skySunRadiance, skyViewVFromHorizonAngle,
    transmittance, transmittanceUv, transmittanceUvToHeightMu,
} from '../dist/environment.js';

let failures = 0;
function check(name, fn) {
    try { fn(); console.log(`  ok    ${name}`); }
    catch (e) { failures++; console.log(`  FAIL  ${name}\n        ${e.message}`); }
}
function assert(v, m) { if (!v) throw new Error(m); }
function near(a, b, e, m) { if (Math.abs(a - b) > e) throw new Error(`${m}: expected ${b} ± ${e}, got ${a}`); }

check('U1/U3/U10: transmittance is bounded, decreases toward the horizon, and has physical channel order', () => {
    const out = new Float32Array(3);
    let previous = [1, 1, 1];
    for (let i = 0; i < 64; i++) {
        const mu = 1 - i / 63;
        transmittance(0, mu, out);
        for (let c = 0; c < 3; c++) {
            assert(out[c] >= 0 && out[c] <= 1, `mu=${mu} channel ${c}: ${out[c]}`);
            assert(out[c] <= previous[c] + 1e-6, `transmittance rose toward horizon at ${i}, channel ${c}`);
        }
        assert(out[0] > out[1] && out[1] > out[2], `wavelength order at mu=${mu}: ${out}`);
        previous = [...out];
    }
    assert(EARTH_ATMOSPHERE.ozoneAbsorption[1] > EARTH_ATMOSPHERE.ozoneAbsorption[0]
        && EARTH_ATMOSPHERE.ozoneAbsorption[0] > EARTH_ATMOSPHERE.ozoneAbsorption[2], 'ozone channel order');
    transmittance(0, -0.1, out);
    assert(out.every(x => x === 0), `ground-occluded transmittance was ${out}`);
});

check('U2: vertical Rayleigh-only transmittance matches the exponential-profile closed form within 1%', () => {
    const medium = { ...EARTH_ATMOSPHERE, mieScattering: 0, mieAbsorption: 0, ozoneAbsorption: [0, 0, 0] };
    const got = transmittance(0, 1, new Float32Array(3), medium);
    const columnMetres = medium.rayleighScaleHeight * 1000
        * (1 - Math.exp(-(medium.atmosphereTopRadius - medium.groundRadius) / medium.rayleighScaleHeight));
    for (let c = 0; c < 3; c++) {
        const expected = Math.exp(-medium.rayleighScattering[c] * columnMetres);
        near(got[c], expected, expected * 0.01, `channel ${c}`);
    }
});

check('U6: sky-view mapping is invertible and concentrates at least 50 rows within 22.5 degrees', () => {
    for (let i = 0; i <= 1000; i++) {
        const a = -Math.PI / 2 + Math.PI * i / 1000;
        near(horizonAngleFromSkyViewV(skyViewVFromHorizonAngle(a)), a, 1e-6, `sample ${i}`);
    }
    let rows = 0;
    for (let y = 0; y < 108; y++) if (Math.abs(horizonAngleFromSkyViewV((y + 0.5) / 108)) <= Math.PI / 8) rows++;
    assert(rows >= 50, `${rows} rows were inside the horizon band`);
});

check('U7/U8: phase functions normalize and Cornette-Shanks has its global peak forward', () => {
    const N = 4096;
    for (const [name, phase] of [['Rayleigh', rayleighPhase], ['Cornette-Shanks', x => cornetteShanksPhase(x, 0.8)]]) {
        let integral = 0;
        for (let i = 0; i < N; i++) integral += phase(-1 + 2 * (i + 0.5) / N) * 4 * Math.PI / N;
        near(integral, 1, 0.005, `${name} normalization`);
    }
    const forward = cornetteShanksPhase(1, 0.8);
    let maximum = -Infinity;
    for (let i = 1; i <= 4096; i++) {
        const value = cornetteShanksPhase(1 - 2 * i / 4096, 0.8);
        maximum = Math.max(maximum, value);
    }
    assert(forward > maximum, `forward ${forward} was not the global maximum ${maximum}`);
    assert(forward > cornetteShanksPhase(-1, 0.8) * 100, 'forward lobe is not strongly anisotropic');
});

check('U5/U9: LUT parameterization round-trips and the TimeOfDay seam uses physical transmittance', () => {
    const uv = new Float32Array(2), hm = new Float32Array(2), a = new Float32Array(3), b = new Float32Array(3);
    for (let y = 1; y < 8; y++) for (let x = 0; x < 8; x++) {
        const h = y * 99 / 8, mu = x / 7;
        transmittanceUv(h, mu, EARTH_ATMOSPHERE, uv);
        transmittanceUvToHeightMu(uv[0], uv[1], EARTH_ATMOSPHERE, hm);
        near(hm[0], h, 2e-3, 'height round trip'); near(hm[1], mu, 2e-3, 'mu round trip');
    }
    for (const elevation of [-8, 0, 1, 5, 20, 60, 90]) {
        skySunRadiance(elevation, a, 9); atmosphereSunRadiance(elevation, b);
        for (let c = 0; c < 3; c++) near(a[c], b[c], 0, `elev ${elevation} channel ${c}`);
    }
});

check('U13: all generated modules parse, contain no backtick, and do not redeclare PI', () => {
    const sky = buildAtmosphereSkyLayout(SKY_PARAM_FLOATS, 2);
    const modules = [buildTransmittanceWGSL(), buildMultipleScatteringWGSL(), buildSkyViewWGSL(),
        buildSkyWGSL({ environmentGroup: 2, sky }), buildSourceCubeWGSL(buildAtmosphereBakeLayout(SKY_PARAM_FLOATS))];
    for (const source of modules) {
        assert(!source.includes('`'), 'generated WGSL contains a backtick');
        assert((source.match(/const PI\b/g) ?? []).length <= 1, 'generated WGSL redeclares PI');
        new WgslReflect(source);
    }
    assert(buildSkyViewWGSL().includes('sunVisible'), 'sky-view does not reject planet-occluded sunlight');
    assert(buildMultipleScatteringWGSL().includes('sunVisible'), 'multiple scattering does not reject planet-occluded sunlight');
});

check('T6 headless seam: SkyComposite propagates atmosphere bake facets and bind entries', () => {
    const resources = { transmittanceView: {}, skyView: {}, sampler: {}, sunDirection: new Float32Array([0, -1, 0]),
        setSunDirection() {}, sunTransmittanceUv(out) { out[0] = 0; out[1] = 0; return out; } };
    const day = {
        name: 'AtmosphereSky', paramFloats: 10,
        getShaderLayout: b => buildAtmosphereSkyLayout(b, 2), getBakeShaderLayout: buildAtmosphereBakeLayout,
        getBakeBindEntries: () => [{ binding: 3, resource: resources.transmittanceView }, { binding: 4, resource: resources.skyView }, { binding: 5, resource: resources.sampler }],
        getSkyboxBindEntries: () => [{ binding: 14, resource: resources.transmittanceView }, { binding: 15, resource: resources.skyView }, { binding: 16, resource: resources.sampler }],
        writeParams() {},
    };
    const composite = new SkyComposite({ day, night: new NightSky() });
    assert(composite.getBakeBindEntries({}).length === 3, 'bake entries were not propagated');
    assert(composite.getSkyboxBindEntries({}).length === 3, 'skybox entries were not propagated');
    const bake = composite.getBakeShaderLayout(SKY_PARAM_FLOATS);
    assert(bake.wgslStructs.includes('@group(0) @binding(3)'), 'day bake facet did not use group 0');
    new WgslReflect(buildSourceCubeWGSL(bake));
});

check('errors: malformed radii, coefficients, anisotropy, and dimensions fail loudly', () => {
    for (const [config, pattern] of [[{ atmosphereTopRadius: 6300 }, /must exceed groundRadius/],
        [{ mieAbsorption: -1 }, /extinction must be non-negative/], [{ mieG: 1 }, /singular/],
        [{ skyViewSize: [192, 107] }, /8x4 workgroup/]]) {
        let message = ''; try { resolveAtmosphereConfig(config); } catch (e) { message = e.message; }
        assert(pattern.test(message), `expected ${pattern}, got ${message}`);
    }
});

console.log(failures === 0 ? '\nAll atmosphere tests passed.' : `\n${failures} test(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);