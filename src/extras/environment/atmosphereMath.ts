// Pure CPU mirror of the physical-atmosphere kernels. Distances are kilometres; coefficients
// are supplied in m^-1 and converted at the integration boundary.

export type RGB = readonly [number, number, number];

export interface AtmosphereMedium {
    groundRadius: number;
    atmosphereTopRadius: number;
    rayleighScattering: RGB;
    rayleighScaleHeight: number;
    mieScattering: number;
    mieAbsorption: number;
    mieScaleHeight: number;
    mieG: number;
    ozoneAbsorption: RGB;
    ozoneCentre: number;
    ozoneHalfWidth: number;
}

export interface AtmosphereConfig extends Partial<AtmosphereMedium> {
    transmittanceSize?: [number, number];
    multipleScatteringSize?: [number, number];
    skyViewSize?: [number, number];
    transmittanceSteps?: number;
    multipleScatteringSteps?: number;
    skyViewSteps?: number;
    skyViewInterval?: number;
}

export const EARTH_ATMOSPHERE: Readonly<AtmosphereMedium> = Object.freeze({
    groundRadius: 6360,
    atmosphereTopRadius: 6460,
    rayleighScattering: [5.802e-6, 13.558e-6, 33.100e-6] as RGB,
    rayleighScaleHeight: 8,
    mieScattering: 3.996e-6,
    mieAbsorption: 4.40e-6,
    mieScaleHeight: 1.2,
    mieG: 0.8,
    ozoneAbsorption: [0.650e-6, 1.881e-6, 0.085e-6] as RGB,
    ozoneCentre: 25,
    ozoneHalfWidth: 15,
});

export const DEFAULT_TRANSMITTANCE_SIZE: readonly [number, number] = [256, 64];
export const DEFAULT_MULTIPLE_SCATTERING_SIZE: readonly [number, number] = [32, 32];
export const DEFAULT_SKY_VIEW_SIZE: readonly [number, number] = [192, 108];

function finiteNonNegative(name: string, value: number): void {
    if (!Number.isFinite(value) || value < 0) {
        throw new Error(`Atmosphere: ${name} coefficient ${value} is negative; extinction must be non-negative or transmittance exceeds 1.`);
    }
}

function size(name: string, value: [number, number], multiple: readonly [number, number]): void {
    if (!Number.isInteger(value[0]) || !Number.isInteger(value[1]) || value[0] <= 0 || value[1] <= 0
        || value[0] % multiple[0] !== 0 || value[1] % multiple[1] !== 0) {
        throw new Error(`Atmosphere: ${name} dimensions ${value[0]}x${value[1]} must be multiples of its ${multiple[0]}x${multiple[1]} workgroup size.`);
    }
}

export function resolveAtmosphereConfig(config: AtmosphereConfig = {}): Required<AtmosphereConfig> {
    const d = EARTH_ATMOSPHERE;
    const result: Required<AtmosphereConfig> = {
        groundRadius: config.groundRadius ?? d.groundRadius,
        atmosphereTopRadius: config.atmosphereTopRadius ?? d.atmosphereTopRadius,
        rayleighScattering: config.rayleighScattering ?? d.rayleighScattering,
        rayleighScaleHeight: config.rayleighScaleHeight ?? d.rayleighScaleHeight,
        mieScattering: config.mieScattering ?? d.mieScattering,
        mieAbsorption: config.mieAbsorption ?? d.mieAbsorption,
        mieScaleHeight: config.mieScaleHeight ?? d.mieScaleHeight,
        mieG: config.mieG ?? d.mieG,
        ozoneAbsorption: config.ozoneAbsorption ?? d.ozoneAbsorption,
        ozoneCentre: config.ozoneCentre ?? d.ozoneCentre,
        ozoneHalfWidth: config.ozoneHalfWidth ?? d.ozoneHalfWidth,
        transmittanceSize: config.transmittanceSize ?? [...DEFAULT_TRANSMITTANCE_SIZE],
        multipleScatteringSize: config.multipleScatteringSize ?? [...DEFAULT_MULTIPLE_SCATTERING_SIZE],
        skyViewSize: config.skyViewSize ?? [...DEFAULT_SKY_VIEW_SIZE],
        transmittanceSteps: config.transmittanceSteps ?? 40,
        multipleScatteringSteps: config.multipleScatteringSteps ?? 20,
        skyViewSteps: config.skyViewSteps ?? 32,
        skyViewInterval: config.skyViewInterval ?? 1,
    };
    validateAtmosphereConfig(result);
    return result;
}

export function validateAtmosphereConfig(c: Required<AtmosphereConfig>): void {
    if (c.atmosphereTopRadius <= c.groundRadius) {
        throw new Error(`Atmosphere: atmosphereTopRadius (${c.atmosphereTopRadius} km) must exceed groundRadius (${c.groundRadius} km).`);
    }
    for (let i = 0; i < 3; i++) finiteNonNegative(`rayleighScattering[${i}]`, c.rayleighScattering[i]);
    finiteNonNegative('mieScattering', c.mieScattering);
    finiteNonNegative('mieAbsorption', c.mieAbsorption);
    for (let i = 0; i < 3; i++) finiteNonNegative(`ozoneAbsorption[${i}]`, c.ozoneAbsorption[i]);
    if (!(c.mieG > -1 && c.mieG < 1)) {
        throw new Error(`Atmosphere: mieG ${c.mieG} is outside (-1, 1); Cornette-Shanks is singular at the endpoints.`);
    }
    for (const [name, value] of [['rayleighScaleHeight', c.rayleighScaleHeight], ['mieScaleHeight', c.mieScaleHeight],
        ['ozoneHalfWidth', c.ozoneHalfWidth]] as const) {
        if (!(value > 0)) throw new Error(`Atmosphere: ${name} ${value} must be positive.`);
    }
    size('transmittance LUT', c.transmittanceSize, [8, 8]);
    size('multiple-scattering LUT', c.multipleScatteringSize, [8, 8]);
    size('sky-view LUT', c.skyViewSize, [8, 4]);
    for (const [name, value] of [['transmittanceSteps', c.transmittanceSteps], ['multipleScatteringSteps', c.multipleScatteringSteps],
        ['skyViewSteps', c.skyViewSteps], ['skyViewInterval', c.skyViewInterval]] as const) {
        if (!Number.isInteger(value) || value < 1) throw new Error(`Atmosphere: ${name} ${value} must be a positive integer.`);
    }
}

export function rayleighDensity(heightKm: number, medium: AtmosphereMedium = EARTH_ATMOSPHERE): number {
    return Math.exp(-Math.max(heightKm, 0) / medium.rayleighScaleHeight);
}

export function mieDensity(heightKm: number, medium: AtmosphereMedium = EARTH_ATMOSPHERE): number {
    return Math.exp(-Math.max(heightKm, 0) / medium.mieScaleHeight);
}

export function ozoneDensity(heightKm: number, medium: AtmosphereMedium = EARTH_ATMOSPHERE): number {
    return Math.max(0, 1 - Math.abs(heightKm - medium.ozoneCentre) / medium.ozoneHalfWidth);
}

/** Positive distance to a sphere along a ray, or -1 when it is not hit in front of the origin. */
export function raySphereDistance(radius: number, radialDistance: number, mu: number): number {
    const b = radialDistance * mu;
    const discriminant = b * b - radialDistance * radialDistance + radius * radius;
    if (discriminant < 0) return -1;
    const s = Math.sqrt(discriminant);
    const near = -b - s, far = -b + s;
    return near > 1e-6 ? near : (far > 1e-6 ? far : (near >= 0 || far >= 0 ? 0 : -1));
}

export function distanceToAtmosphereBoundary(heightKm: number, mu: number,
    medium: AtmosphereMedium = EARTH_ATMOSPHERE): number {
    const r = medium.groundRadius + Math.max(0, heightKm);
    const top = raySphereDistance(medium.atmosphereTopRadius, r, mu);
    const ground = raySphereDistance(medium.groundRadius, r, mu);
    // A ray starting on the ground has a zero-distance self-intersection; it is not a hit ahead.
    return ground > 1e-6 && ground < top ? ground : top;
}

/** 40-step trapezoidal integration of extinction, matching the transmittance kernel. */
export function transmittance(heightKm: number, mu: number,
    out: Float32Array = new Float32Array(3), medium: AtmosphereMedium = EARTH_ATMOSPHERE,
    steps: number = 40): Float32Array {
    const r0 = medium.groundRadius + Math.max(0, heightKm);
    mu = Math.min(1, Math.max(-1, mu));
    const groundHit = raySphereDistance(medium.groundRadius, r0, mu);
    if (groundHit > 1e-6) { out[0] = out[1] = out[2] = 0; return out; }
    const d = distanceToAtmosphereBoundary(heightKm, mu, medium);
    if (!(d > 0)) { out[0] = out[1] = out[2] = 1; return out; }
    const ds = d / steps;
    let odR = 0, odM = 0, odO = 0;
    for (let i = 0; i <= steps; i++) {
        const s = ds * i;
        const r = Math.sqrt(Math.max(0, r0 * r0 + s * s + 2 * r0 * mu * s));
        const h = Math.max(0, r - medium.groundRadius);
        const w = (i === 0 || i === steps) ? 0.5 : 1;
        odR += rayleighDensity(h, medium) * w;
        odM += mieDensity(h, medium) * w;
        odO += ozoneDensity(h, medium) * w;
    }
    odR *= ds * 1000; odM *= ds * 1000; odO *= ds * 1000;
    const mieExt = medium.mieScattering + medium.mieAbsorption;
    for (let c = 0; c < 3; c++) {
        out[c] = Math.exp(-(medium.rayleighScattering[c] * odR + mieExt * odM + medium.ozoneAbsorption[c] * odO));
    }
    return out;
}

/** Physical replacement for Phase 2's authored air-mass colour, signature-compatible. */
export function atmosphereSunRadiance(elevDeg: number, out: Float32Array = new Float32Array(3),
    medium: AtmosphereMedium = EARTH_ATMOSPHERE): Float32Array {
    return transmittance(0, Math.sin(elevDeg * Math.PI / 180), out, medium, 40);
}

export function rayleighPhase(cosTheta: number): number {
    return 3 * (1 + cosTheta * cosTheta) / (16 * Math.PI);
}

export function cornetteShanksPhase(cosTheta: number, g: number = 0.8): number {
    const g2 = g * g;
    return 3 * (1 - g2) * (1 + cosTheta * cosTheta)
        / (8 * Math.PI * (2 + g2) * Math.pow(1 + g2 - 2 * g * cosTheta, 1.5));
}

/** Sky-view horizon mapping and its exact inverse (§5.3). */
export function skyViewVFromHorizonAngle(angleRad: number): number {
    const x = Math.min(Math.PI / 2, Math.max(-Math.PI / 2, angleRad));
    return 0.5 + 0.5 * Math.sign(x) * Math.sqrt(Math.abs(x) / (Math.PI / 2));
}

export function horizonAngleFromSkyViewV(v: number): number {
    const x = Math.min(1, Math.max(0, v)) * 2 - 1;
    return Math.sign(x) * x * x * (Math.PI / 2);
}

/** Bruneton rho/d parameterization used by the transmittance texture. */
export function transmittanceUv(heightKm: number, mu: number, medium: AtmosphereMedium = EARTH_ATMOSPHERE,
    out: Float32Array = new Float32Array(2)): Float32Array {
    const r = medium.groundRadius + Math.max(0, heightKm);
    const H = Math.sqrt(medium.atmosphereTopRadius ** 2 - medium.groundRadius ** 2);
    const rho = Math.sqrt(Math.max(0, r * r - medium.groundRadius ** 2));
    const d = raySphereDistance(medium.atmosphereTopRadius, r, mu);
    const dMin = medium.atmosphereTopRadius - r;
    const dMax = rho + H;
    out[0] = dMax === dMin ? 0 : (d - dMin) / (dMax - dMin);
    out[1] = rho / H;
    return out;
}

export function transmittanceUvToHeightMu(u: number, v: number, medium: AtmosphereMedium = EARTH_ATMOSPHERE,
    out: Float32Array = new Float32Array(2)): Float32Array {
    const H = Math.sqrt(medium.atmosphereTopRadius ** 2 - medium.groundRadius ** 2);
    const rho = Math.min(1, Math.max(0, v)) * H;
    const r = Math.sqrt(rho * rho + medium.groundRadius ** 2);
    const dMin = medium.atmosphereTopRadius - r;
    const dMax = rho + H;
    const d = dMin + Math.min(1, Math.max(0, u)) * (dMax - dMin);
    out[0] = r - medium.groundRadius;
    out[1] = d <= 0 ? 1 : Math.min(1, Math.max(-1,
        (medium.atmosphereTopRadius ** 2 - r * r - d * d) / (2 * r * d)));
    return out;
}