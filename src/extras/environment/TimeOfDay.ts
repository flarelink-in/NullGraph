// Allocation-free clock and consumer coordinator (docs/environment-timeofday-design.md).

import type { DirectionalLightProxy } from '../lights/DirectionalLightProxy';
import type { ShadowSystem } from '../shadows/ShadowSystem';
import type { AutoExposure } from '../post/AutoExposure';
import type { Tonemap } from '../post/Tonemap';
import type { EnvironmentSystem } from './EnvironmentSystem';
import type { StarField } from './StarField';
import {
    angularDistanceDeg, artisticSolarPosition, createLunarPosition, createSolarPosition,
    gmstRadians, julianDay, lunarPosition, skySunRadiance, solarPosition, sunIntensity,
} from './todMath';

export type TimeOfDayMode = 'artistic' | 'astronomical';
export type ExposureCoupling = 'off' | 'clamp' | 'absolute';
export type ActiveCelestialBody = 'sun' | 'moon' | 'none';

export interface TimeOfDayConfig {
    mode?: TimeOfDayMode;
    latitudeDeg?: number;
    longitudeDeg?: number;
    date?: Date;
    sunTiltDeg?: number;
    timeScale?: number;
    rebakeThresholdDeg?: number;
    sunShadowMinElevDeg?: number;
    moonShadowMinElevDeg?: number;
    moonMinIllumination?: number;
    civilDeg?: number;
    nightDeg?: number;
    exposureCoupling?: ExposureCoupling;
    /** Preetham turbidity used by the CPU radiance mirror. Default 2.8. */
    turbidity?: number;
    /** Used only to make the mixed-age warning name the actual cycle length. Default 6. */
    prefilterMips?: number;
    /** Named artistic scotopic shift for expected blue moonlight. Default true. */
    purkinje?: boolean;
}

export interface TimeOfDayBindings {
    environment?: EnvironmentSystem;
    sun?: DirectionalLightProxy;
    moon?: DirectionalLightProxy;
    shadows?: ShadowSystem;
    /** Registered directional atlas slot shared by the active sun or moon record. */
    shadowSlot?: number;
    autoExposure?: AutoExposure;
    tonemap?: Tonemap;
    /** Optional Phase 3 unresolved-star draw driven by the same sidereal clock. */
    stars?: StarField;
}

export interface CelestialState {
    readonly sunDir: Float32Array;
    sunElevationDeg: number;
    sunAzimuthDeg: number;
    readonly sunRadiance: Float32Array;
    readonly moonDir: Float32Array;
    moonElevationDeg: number;
    moonPhase: number;
    moonIllumination: number;
    readonly moonRadiance: Float32Array;
    gmstRad: number;
    activeBody: ActiveCelestialBody;
    dayFactor: number;
}

interface ResolvedConfig {
    mode: TimeOfDayMode; latitudeDeg: number; longitudeDeg: number; sunTiltDeg: number;
    timeScale: number; rebakeThresholdDeg: number; sunShadowMinElevDeg: number;
    moonShadowMinElevDeg: number; moonMinIllumination: number; civilDeg: number;
    nightDeg: number; exposureCoupling: ExposureCoupling; turbidity: number;
    purkinje: boolean;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
    const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
    return t * t * (3 - 2 * t);
}

export class TimeOfDay {
    private readonly cfg: ResolvedConfig;
    private readonly celestial: CelestialState;
    private readonly solar = createSolarPosition();
    /** Real solar longitude used only for lunar phase when the visible sun is on the artistic dial. */
    private readonly ephemerisSun = createSolarPosition();
    private readonly lunar = createLunarPosition();
    private readonly bakedSunDir = new Float32Array(3);
    private readonly bakedMoonDir = new Float32Array(3);
    private readonly initialJulianDay: number;
    private secondsOfDay: number;
    private elapsedDays = 0;
    private hasBakeDirection = false;
    private dirty = true;
    private appliedBody: ActiveCelestialBody = 'none';

    constructor(config: TimeOfDayConfig = {}) {
        const mode = config.mode ?? 'artistic';
        if (mode === 'astronomical' && (config.latitudeDeg === undefined || config.longitudeDeg === undefined)) {
            throw new Error(`TimeOfDay: mode 'astronomical' requires latitudeDeg/longitudeDeg. Use mode 'artistic' for a geography-free dial.`);
        }
        const latitudeDeg = config.latitudeDeg ?? 45;
        if (Math.abs(latitudeDeg) > 90) {
            throw new Error(`TimeOfDay: latitudeDeg ${latitudeDeg} is outside [-90, 90].`);
        }
        const threshold = config.rebakeThresholdDeg ?? 0.25;
        if (threshold <= 0) {
            throw new Error(`TimeOfDay: rebakeThresholdDeg must be > 0; 0 re-bakes every frame, which is what 'amortized' already does unconditionally.`);
        }
        const sunShadow = config.sunShadowMinElevDeg ?? 3;
        const moonShadow = config.moonShadowMinElevDeg ?? 10;
        if (moonShadow < sunShadow) {
            throw new Error(`TimeOfDay: moon and sun shadow windows overlap; both bodies would qualify to cast simultaneously (design §7.2).`);
        }
        if (threshold > 2) {
            console.warn(`TimeOfDay: rebakeThresholdDeg ${threshold}° exceeds 2°. With prefilterMips=${config.prefilterMips ?? 6}, the amortized cube stays mixed-age for ${config.prefilterMips ?? 6} frames and rough reflections will step visibly (design §6.3).`);
        }
        this.cfg = {
            mode, latitudeDeg, longitudeDeg: config.longitudeDeg ?? 0,
            sunTiltDeg: config.sunTiltDeg ?? 23.44, timeScale: config.timeScale ?? 1,
            rebakeThresholdDeg: threshold, sunShadowMinElevDeg: sunShadow,
            moonShadowMinElevDeg: moonShadow, moonMinIllumination: config.moonMinIllumination ?? 0.35,
            civilDeg: config.civilDeg ?? -6, nightDeg: config.nightDeg ?? -18,
            exposureCoupling: config.exposureCoupling ?? 'clamp', turbidity: config.turbidity ?? 2.8,
            purkinje: config.purkinje ?? true,
        };
        if (this.cfg.nightDeg >= this.cfg.civilDeg) {
            throw new Error(`TimeOfDay: nightDeg must be below civilDeg so dayFactor has a non-empty twilight ramp.`);
        }
        const date = config.date ?? new Date();
        this.initialJulianDay = julianDay(date);
        this.secondsOfDay = date.getUTCHours() * 3600 + date.getUTCMinutes() * 60
            + date.getUTCSeconds() + date.getUTCMilliseconds() / 1000;
        this.celestial = {
            sunDir: this.solar.direction, sunElevationDeg: 0, sunAzimuthDeg: 0,
            sunRadiance: new Float32Array(3), moonDir: this.lunar.direction, moonElevationDeg: 0,
            moonPhase: 0, moonIllumination: 0, moonRadiance: new Float32Array(3), gmstRad: 0,
            activeBody: 'none', dayFactor: 0,
        };
        this.update();
    }

    /** Set the 0–24 dial while retaining the configured date. */
    public setTime(hours: number): void {
        this.secondsOfDay = ((hours % 24) + 24) % 24 * 3600;
        this.update();
    }

    /** Advance by real seconds, scaled by timeScale. */
    public advance(dtSeconds: number): void {
        const oldDay = Math.floor(this.secondsOfDay / 86400);
        this.secondsOfDay += dtSeconds * this.cfg.timeScale;
        const newDay = Math.floor(this.secondsOfDay / 86400);
        this.elapsedDays += newDay - oldDay;
        this.secondsOfDay -= (newDay - oldDay) * 86400;
        if (this.secondsOfDay < 0) { this.secondsOfDay += 86400; this.elapsedDays--; }
        this.update();
    }

    /** Update the exposure coupling mode dynamically. */
    public setExposureCoupling(mode: ExposureCoupling): void {
        this.cfg.exposureCoupling = mode;
    }

    /** Recompute persistent state. Public for a worker or a caller that changed shared clock data. */
    public update(): void {
        // Dirtiness describes this update/frame. applyTo publishes it but does not erase it before
        // the caller can inspect the documented post-apply gate.
        this.dirty = false;
        const hour = this.secondsOfDay / 3600;
        const baseMidnight = Math.floor(this.initialJulianDay - 0.5) + 0.5;
        const jd = baseMidnight + this.elapsedDays + hour / 24;
        if (this.cfg.mode === 'artistic') {
            artisticSolarPosition(hour, this.cfg.sunTiltDeg, this.solar);
            solarPosition(jd, this.cfg.latitudeDeg, this.cfg.longitudeDeg, this.ephemerisSun);
        } else {
            solarPosition(jd, this.cfg.latitudeDeg, this.cfg.longitudeDeg, this.solar);
        }
        lunarPosition(jd, this.cfg.latitudeDeg, this.cfg.longitudeDeg,
            this.cfg.mode === 'artistic' ? this.ephemerisSun.eclipticLongitudeRad
                : this.solar.eclipticLongitudeRad, this.lunar);
        const s = this.celestial;
        s.sunDir.set(this.solar.direction);
        s.sunElevationDeg = this.solar.elevationDeg;
        s.sunAzimuthDeg = this.solar.azimuthDeg;
        skySunRadiance(s.sunElevationDeg, s.sunRadiance, this.cfg.turbidity);
        if (this.cfg.mode === 'artistic') {
            s.moonDir[0] = -this.solar.direction[0];
            s.moonDir[1] = -this.solar.direction[1];
            s.moonDir[2] = -this.solar.direction[2];
            s.moonElevationDeg = -this.solar.elevationDeg;
            s.moonPhase = 0.5;
            s.moonIllumination = 1.0;
        } else {
            s.moonDir.set(this.lunar.direction);
            s.moonElevationDeg = this.lunar.elevationDeg;
            s.moonPhase = this.lunar.phase;
            s.moonIllumination = this.lunar.illumination;
        }
        skySunRadiance(s.moonElevationDeg, s.moonRadiance, this.cfg.turbidity);
        // Slightly warm radiometric moonlight; optional Purkinje shift is explicitly artistic.
        const moonScale = s.moonIllumination * 0.039;
        const purkinjeR = this.cfg.purkinje ? 0.58 : 1.0;
        const purkinjeG = this.cfg.purkinje ? 0.76 : 0.92;
        const purkinjeB = this.cfg.purkinje ? 1.0 : 0.78;
        s.moonRadiance[0] *= moonScale * purkinjeR;
        s.moonRadiance[1] *= moonScale * purkinjeG;
        s.moonRadiance[2] *= moonScale * purkinjeB;
        s.gmstRad = gmstRadians(jd);
        s.dayFactor = smoothstep(this.cfg.nightDeg, this.cfg.civilDeg, s.sunElevationDeg);
        s.activeBody = s.sunElevationDeg >= this.cfg.sunShadowMinElevDeg ? 'sun'
            : s.moonElevationDeg >= this.cfg.moonShadowMinElevDeg
                && s.moonIllumination >= this.cfg.moonMinIllumination ? 'moon' : 'none';
        const moonWeight = 1 - s.dayFactor;
        if (!this.hasBakeDirection
            || Math.max(angularDistanceDeg(this.bakedSunDir, s.sunDir),
                angularDistanceDeg(this.bakedMoonDir, s.moonDir) * moonWeight)
                >= this.cfg.rebakeThresholdDeg) {
            this.dirty = true;
        }
    }

    public get envDirty(): boolean { return this.dirty; }
    public get activeBody(): ActiveCelestialBody { return this.celestial.activeBody; }
    public get dayFactor(): number { return this.celestial.dayFactor; }
    public get sunElevationDeg(): number { return this.celestial.sunElevationDeg; }
    public get moonElevationDeg(): number { return this.celestial.moonElevationDeg; }
    public get moonPhase(): number { return this.celestial.moonPhase; }
    public get moonIllumination(): number { return this.celestial.moonIllumination; }
    public get gmstRad(): number { return this.celestial.gmstRad; }

    /** No-arg form is an internal view, valid until the next update; out form writes and returns out. */
    public getSunDirection(out?: Float32Array): Float32Array {
        if (!out) return this.celestial.sunDir;
        out[0] = this.celestial.sunDir[0]; out[1] = this.celestial.sunDir[1]; out[2] = this.celestial.sunDir[2];
        return out;
    }

    public getMoonDirection(out?: Float32Array): Float32Array {
        if (!out) return this.celestial.moonDir;
        out[0] = this.celestial.moonDir[0]; out[1] = this.celestial.moonDir[1]; out[2] = this.celestial.moonDir[2];
        return out;
    }

    /** Write all configured consumers. The caller still owns LightSystem.setRawData/upload. */
    public applyTo(bindings: TimeOfDayBindings): void {
        const s = this.celestial;
        if (this.cfg.exposureCoupling === 'clamp' && !bindings.autoExposure) {
            throw new Error(`TimeOfDay: exposureCoupling 'clamp' needs an AutoExposure effect in the PostChain; use 'absolute' to drive Tonemap.exposure directly.`);
        }
        if (this.cfg.exposureCoupling === 'absolute' && !bindings.tonemap) {
            throw new Error(`TimeOfDay: exposureCoupling 'absolute' needs a Tonemap effect to drive Tonemap.exposure.`);
        }
        if (bindings.environment) {
            if (bindings.environment.sky.setCelestialState) bindings.environment.sky.setCelestialState(s);
            else {
                bindings.environment.sky.setSunDirection?.(s.sunDir);
                bindings.environment.sky.setMoonDirection?.(s.moonDir);
            }
            bindings.environment.setSkyParams({});
            if (this.dirty) bindings.environment.invalidateBake();
        }
        if (bindings.stars) {
            bindings.stars.setCelestialState(s);
            bindings.stars.upload();
        }
        if (bindings.sun) {
            bindings.sun.direction = s.sunDir;
            bindings.sun.color = s.sunRadiance;
            bindings.sun.intensity = sunIntensity(s.sunElevationDeg);
        }
        if (bindings.moon) {
            bindings.moon.direction = s.moonDir;
            bindings.moon.color = s.moonRadiance;
            bindings.moon.intensity = s.moonElevationDeg > 0 ? 1 : 0;
        }
        this.setShadowCaster(s.activeBody, bindings);
        if (this.cfg.exposureCoupling === 'clamp') {
            // Night window [0.05, 1.5], day window [0.1, 8]. Histogram adaptation remains authoritative.
            bindings.autoExposure!.minExposure = 0.05 + 0.05 * s.dayFactor;
            bindings.autoExposure!.maxExposure = 1.5 + 6.5 * s.dayFactor;
        } else if (this.cfg.exposureCoupling === 'absolute') {
            bindings.tonemap!.exposure = 0.25 + 1.75 * s.dayFactor;
        }
        if (this.dirty) {
            this.bakedSunDir.set(s.sunDir);
            this.bakedMoonDir.set(s.moonDir);
            this.hasBakeDirection = true;
        }
    }

    private setShadowCaster(body: ActiveCelestialBody, bindings: TimeOfDayBindings): void {
        if (!bindings.sun && !bindings.moon && !bindings.shadows) return;
        const slot = bindings.shadowSlot ?? -1;
        // Keep both slot writes adjacent on every call; external edits cannot leave two holders alive.
        if (bindings.sun) bindings.sun.shadowSlot = body === 'sun' ? slot : -1;
        if (bindings.moon) bindings.moon.shadowSlot = body === 'moon' ? slot : -1;
        if (!bindings.shadows) { this.appliedBody = body; return; }
        if (body !== 'none') {
            const direction = body === 'sun' ? this.celestial.sunDir : this.celestial.moonDir;
            bindings.shadows.setLight(direction);
            if (slot >= 0 && bindings.shadows.isAtlas) {
                bindings.shadows.updateLight(slot, { direction });
            }
        }
        // A handoff to none still invalidates the old cascade cache; slot ownership and cache
        // state move in this one function and cannot describe different bodies for a frame.
        if (slot >= 0 && bindings.shadows.isAtlas) {
            bindings.shadows.invalidate(slot);
        }
        this.appliedBody = body;
    }
}