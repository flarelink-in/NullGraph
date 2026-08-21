// Pure, worker-safe celestial math for docs/environment-timeofday-design.md.
// Directions are light TRAVEL directions in NullGraph's +Y-up, +Z-north frame.

import { DEG_TO_RAD } from './skyMath';
import { atmosphereSunRadiance } from './atmosphereMath';

export const RAD_TO_DEG = 180 / Math.PI;
export const J2000 = 2451545.0;
export const SYNODIC_MONTH_DAYS = 29.530588;

export interface SolarPosition {
    direction: Float32Array;
    elevationDeg: number;
    azimuthDeg: number;
    eclipticLongitudeRad: number;
    declinationRad: number;
}

export interface LunarPosition {
    direction: Float32Array;
    elevationDeg: number;
    eclipticLongitudeRad: number;
    illumination: number;
    phase: number;
    elongationDeg: number;
}

export function createSolarPosition(): SolarPosition {
    return { direction: new Float32Array(3), elevationDeg: 0, azimuthDeg: 0,
        eclipticLongitudeRad: 0, declinationRad: 0 };
}

export function createLunarPosition(): LunarPosition {
    return { direction: new Float32Array(3), elevationDeg: 0, eclipticLongitudeRad: 0,
        illumination: 0, phase: 0, elongationDeg: 0 };
}

function wrap(x: number, period: number): number {
    return ((x % period) + period) % period;
}

function clamp(x: number, lo: number, hi: number): number {
    return Math.min(hi, Math.max(lo, x));
}

/** Julian day from a Date. Milliseconds are retained; no Date is allocated. */
export function julianDay(date: Date): number {
    return date.getTime() / 86400000 + 2440587.5;
}

/** Greenwich mean sidereal time, radians in [0, 2π). */
export function gmstRadians(jd: number): number {
    const t = (jd - J2000) / 36525;
    const deg = 280.46061837 + 360.98564736629 * (jd - J2000)
        + 0.000387933 * t * t - t * t * t / 38710000;
    return wrap(deg * DEG_TO_RAD, Math.PI * 2);
}

function horizonDirection(
    rightAscension: number, declination: number, jd: number,
    latitudeDeg: number, longitudeDeg: number, out: Float32Array,
): number {
    const lat = latitudeDeg * DEG_TO_RAD;
    const hour = wrap(gmstRadians(jd) + longitudeDeg * DEG_TO_RAD - rightAscension + Math.PI,
        Math.PI * 2) - Math.PI;
    const sinAlt = clamp(Math.sin(lat) * Math.sin(declination)
        + Math.cos(lat) * Math.cos(declination) * Math.cos(hour), -1, 1);
    const alt = Math.asin(sinAlt);
    // atan2(east, north): north=0, east=90, clockwise when viewed from above.
    const east = -Math.cos(declination) * Math.sin(hour);
    const north = Math.sin(declination) * Math.cos(lat)
        - Math.cos(declination) * Math.sin(lat) * Math.cos(hour);
    const az = wrap(Math.atan2(east, north), Math.PI * 2);
    const cosAlt = Math.cos(alt);
    // Position is (east, up, north); light travel is its negative.
    out[0] = -cosAlt * Math.sin(az);
    out[1] = -Math.sin(alt);
    out[2] = -cosAlt * Math.cos(az);
    return alt * RAD_TO_DEG;
}

/** NOAA/Meeus low-precision apparent solar position. */
export function solarPosition(
    jd: number, latitudeDeg: number, longitudeDeg: number,
    out: SolarPosition = createSolarPosition(),
): SolarPosition {
    const t = (jd - J2000) / 36525;
    const meanLongitude = wrap((280.46646 + t * (36000.76983 + 0.0003032 * t)) * DEG_TO_RAD,
        Math.PI * 2);
    const anomaly = (357.52911 + t * (35999.05029 - 0.0001537 * t)) * DEG_TO_RAD;
    const centre = ((1.914602 - t * (0.004817 + 0.000014 * t)) * Math.sin(anomaly)
        + (0.019993 - 0.000101 * t) * Math.sin(2 * anomaly)
        + 0.000289 * Math.sin(3 * anomaly)) * DEG_TO_RAD;
    const omega = (125.04 - 1934.136 * t) * DEG_TO_RAD;
    const lambda = meanLongitude + centre - 0.00569 * DEG_TO_RAD
        - 0.00478 * DEG_TO_RAD * Math.sin(omega);
    const epsilon = (23.439291 - 0.0130042 * t) * DEG_TO_RAD
        + 0.00256 * DEG_TO_RAD * Math.cos(omega);
    const ra = Math.atan2(Math.cos(epsilon) * Math.sin(lambda), Math.cos(lambda));
    const dec = Math.asin(Math.sin(epsilon) * Math.sin(lambda));
    out.elevationDeg = horizonDirection(ra, dec, jd, latitudeDeg, longitudeDeg, out.direction);
    out.azimuthDeg = wrap(Math.atan2(-out.direction[0], -out.direction[2]), Math.PI * 2) * RAD_TO_DEG;
    out.eclipticLongitudeRad = wrap(lambda, Math.PI * 2);
    out.declinationRad = dec;
    return out;
}

/** Geography-free great-circle dial: noon is overhead, 06:00/18:00 touch the horizon. */
export function artisticSolarPosition(
    hour: number, tiltDeg: number, out: SolarPosition = createSolarPosition(),
): SolarPosition {
    const a = (wrap(hour, 24) - 12) * Math.PI / 12;
    const tilt = tiltDeg * DEG_TO_RAD;
    // Position before negation: eastward great circle, tilted around the east axis.
    const east = Math.sin(a);
    const up = Math.cos(a) * Math.cos(tilt);
    const north = Math.cos(a) * Math.sin(tilt);
    out.direction[0] = -east; out.direction[1] = -up; out.direction[2] = -north;
    out.elevationDeg = Math.asin(clamp(up, -1, 1)) * RAD_TO_DEG;
    out.azimuthDeg = wrap(Math.atan2(east, north), Math.PI * 2) * RAD_TO_DEG;
    out.eclipticLongitudeRad = wrap(a, Math.PI * 2);
    out.declinationRad = Math.asin(clamp(north, -1, 1));
    return out;
}

/** Meeus truncated lunar series: five longitude terms and two latitude terms. */
export function lunarPosition(
    jd: number, latitudeDeg: number, longitudeDeg: number, sunLongitudeRad: number,
    out: LunarPosition = createLunarPosition(),
): LunarPosition {
    const d = jd - J2000;
    const L = wrap((218.3164477 + 13.17639648 * d) * DEG_TO_RAD, Math.PI * 2);
    const D = wrap((297.8501921 + 12.19074912 * d) * DEG_TO_RAD, Math.PI * 2);
    const M = wrap((357.5291092 + 0.98560028 * d) * DEG_TO_RAD, Math.PI * 2);
    const Mp = wrap((134.9633964 + 13.06499295 * d) * DEG_TO_RAD, Math.PI * 2);
    const F = wrap((93.2720950 + 13.22935024 * d) * DEG_TO_RAD, Math.PI * 2);
    const lon = L + DEG_TO_RAD * (6.289 * Math.sin(Mp) + 1.274 * Math.sin(2 * D - Mp)
        + 0.658 * Math.sin(2 * D) + 0.214 * Math.sin(2 * Mp) - 0.186 * Math.sin(M));
    const lat = DEG_TO_RAD * (5.128 * Math.sin(F) + 0.280 * Math.sin(Mp + F));
    const epsilon = (23.439291 - 0.00000036 * d) * DEG_TO_RAD;
    const x = Math.cos(lon) * Math.cos(lat);
    const y = Math.sin(lon) * Math.cos(lat) * Math.cos(epsilon) - Math.sin(lat) * Math.sin(epsilon);
    const z = Math.sin(lon) * Math.cos(lat) * Math.sin(epsilon) + Math.sin(lat) * Math.cos(epsilon);
    const ra = Math.atan2(y, x);
    const dec = Math.asin(clamp(z, -1, 1));
    out.elevationDeg = horizonDirection(ra, dec, jd, latitudeDeg, longitudeDeg, out.direction);
    const elongation = wrap(lon - sunLongitudeRad, Math.PI * 2);
    const shortElongation = Math.acos(Math.cos(elongation));
    out.eclipticLongitudeRad = wrap(lon, Math.PI * 2);
    out.phase = elongation / (Math.PI * 2);
    out.elongationDeg = shortElongation * RAD_TO_DEG;
    out.illumination = (1 - Math.cos(shortElongation)) * 0.5;
    return out;
}

/** Physical ground-to-sun transmittance. The trailing turbidity argument survives for compatibility. */
export function skySunRadiance(
    elevationDeg: number, out: Float32Array = new Float32Array(3), _turbidity: number = 2.8,
): Float32Array {
    return atmosphereSunRadiance(elevationDeg, out);
}

/** Direct-light multiplier; deliberately reaches zero at geometric sunset. */
export function sunIntensity(elevationDeg: number): number {
    const t = clamp(elevationDeg / 3, 0, 1);
    return t * t * (3 - 2 * t);
}

export function angularDistanceDeg(a: ArrayLike<number>, b: ArrayLike<number>): number {
    return Math.acos(clamp(a[0] * b[0] + a[1] * b[1] + a[2] * b[2], -1, 1)) * RAD_TO_DEG;
}