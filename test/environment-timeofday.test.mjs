// Headless tests for docs/environment-timeofday-design.md. No GPU or browser is used.
//   npm run build:lib && npm run test:tod
import {
    TimeOfDay, angularDistanceDeg, artisticSolarPosition,
    createLunarPosition, createSolarPosition, julianDay, lunarPosition, preethamBeta,
    skySunRadiance, solarPosition, sunIntensity, atmosphereSunRadiance, SYNODIC_MONTH_DAYS,
} from '../dist/environment.js';
import { DirectionalLightProxy } from '../dist/lights.js';

let failures = 0;
function check(name, fn) {
    try { fn(); console.log(`  ok    ${name}`); }
    catch (e) { failures++; console.log(`  FAIL  ${name}\n        ${e.message}`); }
}
function assert(condition, message) { if (!condition) throw new Error(message); }
function near(a, b, eps, message) {
    if (Math.abs(a - b) > eps) throw new Error(`${message}: expected ${b} ± ${eps}, got ${a}`);
}
function norm(v) { return Math.hypot(v[0], v[1], v[2]); }
function phaseDistance(a, b) { const d = Math.abs(a - b); return Math.min(d, 1 - d); }

check('U1: artistic solar elevation is symmetric about noon', () => {
    const a = createSolarPosition(), b = createSolarPosition();
    for (let i = 0; i < 100; i++) {
        const offset = i * 11.9 / 99;
        artisticSolarPosition(12 - offset, 23.44, a);
        artisticSolarPosition(12 + offset, 23.44, b);
        near(a.elevationDeg, b.elevationDeg, 1e-10, `offset ${offset}`);
    }
});

check('U2: TimeOfDay sun radiance keeps its signature and matches physical transmittance', () => {
    const actual = new Float32Array(3), expected = new Float32Array(3);
    for (let i = 0; i < 32; i++) {
        const elevation = -2 + i * 92 / 31;
        skySunRadiance(elevation, actual, 2.8);
        atmosphereSunRadiance(elevation, expected);
        for (let c = 0; c < 3; c++) near(actual[c], expected[c], 1e-6, `elev ${elevation} channel ${c}`);
    }
});

check('U3: June solstice gives polar day at 67N and polar night at 67S', () => {
    const day = Math.floor(julianDay(new Date('2026-06-21T00:00:00Z')) - 0.5) + 0.5;
    const s = createSolarPosition();
    for (let minute = 0; minute < 1440; minute++) {
        solarPosition(day + minute / 1440, 67, 0, s);
        assert(s.elevationDeg > 0, `67N minute ${minute}: ${s.elevationDeg}`);
        solarPosition(day + minute / 1440, -67, 0, s);
        assert(s.elevationDeg < 0, `67S minute ${minute}: ${s.elevationDeg}`);
    }
});

check('U4/U6: shadow slot ownership is exclusive and sun intensity is zero at/below sunset', () => {
    const data = new Float32Array(32);
    const sun = new DirectionalLightProxy(), moon = new DirectionalLightProxy();
    sun.wrap(data); moon.wrap(data); sun.seek(0); moon.seek(1);
    const tod = new TimeOfDay({ date: new Date('2026-03-20T00:00:00Z'), exposureCoupling: 'off' });
    for (let minute = 0; minute < 1440; minute++) {
        tod.setTime(minute / 60);
        tod.applyTo({ sun, moon, shadowSlot: 0 });
        assert(!(sun.shadowSlot >= 0 && moon.shadowSlot >= 0), `both own slot at minute ${minute}`);
        if (tod.sunElevationDeg <= 0) near(sun.intensity, 0, 0, `sunset minute ${minute}`);
        if (sun.shadowSlot >= 0) assert(tod.sunElevationDeg >= 3, `sun shadows below threshold: ${tod.sunElevationDeg}`);
    }
});

check('U5: a shadow update moves direction then invalidates the same slot', () => {
    const calls = [];
    const shadows = {
        isAtlas: true,
        setLight: () => calls.push('direction'),
        updateLight: (slot) => calls.push(`slot:${slot}`),
        invalidate: (slot) => calls.push(`invalidate:${slot}`),
    };
    const tod = new TimeOfDay({ date: new Date('2026-03-20T12:00:00Z'), exposureCoupling: 'off' });
    tod.setTime(12);
    tod.applyTo({ shadows, shadowSlot: 4 });
    assert(calls.join(',') === 'direction,slot:4,invalidate:4', `call order was ${calls.join(',')}`);
});

check('U7: angular re-bake gate handles real-time and 1000x with one rule', () => {
    for (const [scale, min, max] of [[1, 0, 2], [1000, 800, 1000]]) {
        const tod = new TimeOfDay({ date: new Date('2026-03-20T12:00:00Z'), timeScale: scale, exposureCoupling: 'off' });
        let bakes = 0;
        for (let frame = 0; frame < 3600; frame++) {
            tod.advance(1 / 60);
            if (tod.envDirty) { bakes++; tod.applyTo({}); }
        }
        assert(bakes >= min && bakes <= max, `${scale}x produced ${bakes} bakes, expected ${min}..${max}`);
    }
});

check('U8/U9: lunar phase repeats after one synodic month and illumination follows elongation', () => {
    const jd = julianDay(new Date('2026-01-01T00:00:00Z'));
    const s = createSolarPosition(), moon = createLunarPosition();
    solarPosition(jd, 0, 0, s); lunarPosition(jd, 0, 0, s.eclipticLongitudeRad, moon);
    const initial = moon.phase;
    solarPosition(jd + SYNODIC_MONTH_DAYS, 0, 0, s);
    lunarPosition(jd + SYNODIC_MONTH_DAYS, 0, 0, s.eclipticLongitudeRad, moon);
    assert(phaseDistance(initial, moon.phase) < 0.02, `phase drift ${phaseDistance(initial, moon.phase)}`);
    for (let d = 0; d < 60; d += 0.1) {
        solarPosition(jd + d, 0, 0, s);
        lunarPosition(jd + d, 0, 0, s.eclipticLongitudeRad, moon);
        const expected = (1 - Math.cos(moon.elongationDeg * Math.PI / 180)) / 2;
        near(moon.illumination, expected, 1e-12, `day ${d}`);
        assert((moon.illumination > 0.99) === (moon.elongationDeg > 168.52),
            `full-moon threshold mismatch at elongation ${moon.elongationDeg}`);
    }
});

check('U10: both clock modes always produce normalized sun travel directions', () => {
    const artistic = createSolarPosition(), astro = createSolarPosition();
    const jd = julianDay(new Date('2026-01-01T00:00:00Z'));
    for (let i = 0; i < 10000; i++) {
        artisticSolarPosition(i * 24 / 10000, 23.44, artistic);
        solarPosition(jd + i * 365.25 / 10000, 45, 0, astro);
        near(norm(artistic.direction), 1, 1e-6, `artistic ${i}`);
        near(norm(astro.direction), 1, 1e-6, `astronomical ${i}`);
    }
});

check('U11: 1,000 advance/apply calls allocate no typed arrays', () => {
    const Original = globalThis.Float32Array;
    const tod = new TimeOfDay({ exposureCoupling: 'off' });
    let allocations = 0;
    globalThis.Float32Array = function (...args) { allocations++; return new Original(...args); };
    try {
        for (let i = 0; i < 1000; i++) { tod.advance(1 / 60); tod.applyTo({}); }
    } finally {
        globalThis.Float32Array = Original;
    }
    assert(allocations === 0, `${allocations} Float32Array allocations`);
});

check('U12: direction accessors reuse the internal view and honour out', () => {
    const tod = new TimeOfDay({ exposureCoupling: 'off' });
    assert(tod.getSunDirection() === tod.getSunDirection(), 'no-arg accessor returned different views');
    const out = new Float32Array(3);
    assert(tod.getSunDirection(out) === out, 'out accessor did not return out');
    near(angularDistanceDeg(out, tod.getSunDirection()), 0, 0.02, 'out direction');
});

check('config errors name the offending relationship and way out', () => {
    const cases = [
        [{ mode: 'astronomical' }, /requires latitudeDeg\/longitudeDeg/],
        [{ latitudeDeg: 91 }, /outside \[-90, 90\]/],
        [{ rebakeThresholdDeg: 0 }, /must be > 0/],
        [{ sunShadowMinElevDeg: 10, moonShadowMinElevDeg: 3 }, /windows overlap/],
    ];
    for (const [config, pattern] of cases) {
        let message = '';
        try { new TimeOfDay(config); } catch (e) { message = e.message; }
        assert(pattern.test(message), `expected ${pattern}, got ${message || 'no error'}`);
    }
    const tod = new TimeOfDay();
    let message = '';
    try { tod.applyTo({}); } catch (e) { message = e.message; }
    assert(/needs an AutoExposure/.test(message), `missing exposure error: ${message}`);
});

console.log(failures === 0 ? '\nAll environment time-of-day tests passed.' : `\n${failures} test(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);