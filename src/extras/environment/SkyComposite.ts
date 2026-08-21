import type { CelestialState } from './TimeOfDay';
import type { ShaderSkyLayout, SkySource } from './types';
import type { EnvironmentSystem } from './EnvironmentSystem';

export interface SkyCompositeConfig { day: SkySource; night: SkySource; dayFactor?: number; }

function renameLayout(layout: ShaderSkyLayout, prefix: string): ShaderSkyLayout {
    // Phase 1 reserves sky_-prefixed helpers. Entry points are renamed explicitly, then helpers.
    const rename = (src: string) => src
        .replace(/\bsky_/g, prefix)
        .replace(/\bsampleSky\b/g, `${prefix}sampleSky`)
        .replace(/\bskySunRadiance\b/g, `${prefix}skySunRadiance`);
    return { wgslStructs: rename(layout.wgslStructs), wgslLogic: rename(layout.wgslLogic) };
}

/** Decorates two SkySources into the one evaluator required by EnvironmentSystem. */
export class SkyComposite implements SkySource {
    public readonly name = 'SkyComposite';
    public readonly paramFloats: number;
    private readonly day: SkySource;
    private readonly night: SkySource;
    private dayFactor: number;

    constructor(cfg: SkyCompositeConfig) {
        if (!cfg.day) throw new Error('SkyComposite: child "day" has no source set.');
        if (!cfg.night) throw new Error('SkyComposite: child "night" has no source set.');
        this.day = cfg.day; this.night = cfg.night;
        this.paramFloats = this.day.paramFloats + this.night.paramFloats + 1;
        this.dayFactor = cfg.dayFactor ?? 1;
    }

    public setDayFactor(value: number): void { this.dayFactor = Math.min(1, Math.max(0, value)); }
    public setSunDirection(dir: ArrayLike<number>): void {
        this.day.setSunDirection?.(dir); this.night.setSunDirection?.(dir);
    }
    public setMoonDirection(dir: ArrayLike<number>): void {
        this.day.setMoonDirection?.(dir); this.night.setMoonDirection?.(dir);
    }
    public setCelestialState(state: CelestialState): void {
        this.day.setCelestialState?.(state); this.night.setCelestialState?.(state);
        this.setDayFactor(state.dayFactor);
    }

    public getShaderLayout(base: number): ShaderSkyLayout {
        const day = renameLayout(this.day.getShaderLayout(base), 'sky_day_');
        const nightBase = base + this.day.paramFloats;
        const night = renameLayout(this.night.getShaderLayout(nightBase), 'sky_night_');
        const factor = base + this.day.paramFloats + this.night.paramFloats;
        return {
            wgslStructs: `${day.wgslStructs}\n${night.wgslStructs}`,
            wgslLogic: `${day.wgslLogic}\n${night.wgslLogic}\n` + /* wgsl */ `
fn sampleSky(dir: vec3<f32>) -> vec3<f32> {
    return mix(sky_night_sampleSky(dir), sky_day_sampleSky(dir), skyParams[${factor}]);
}
fn skySunRadiance(elevDeg: f32) -> vec3<f32> {
    return mix(sky_night_skySunRadiance(elevDeg), sky_day_skySunRadiance(elevDeg), skyParams[${factor}]);
}
`,
        };
    }

    /** Compose the children at the bake site too; texture-backed children may use different groups. */
    public getBakeShaderLayout(base: number): ShaderSkyLayout {
        const dayLayout = this.day.getBakeShaderLayout?.(base) ?? this.day.getShaderLayout(base);
        const nightBase = base + this.day.paramFloats;
        const nightLayout = this.night.getBakeShaderLayout?.(nightBase) ?? this.night.getShaderLayout(nightBase);
        const day = renameLayout(dayLayout, 'sky_day_');
        const night = renameLayout(nightLayout, 'sky_night_');
        const factor = base + this.day.paramFloats + this.night.paramFloats;
        return {
            wgslStructs: `${day.wgslStructs}\n${night.wgslStructs}`,
            wgslLogic: `${day.wgslLogic}\n${night.wgslLogic}\n` + /* wgsl */ `
fn sampleSky(dir: vec3<f32>) -> vec3<f32> {
    return mix(sky_night_sampleSky(dir), sky_day_sampleSky(dir), skyParams[${factor}]);
}
fn skySunRadiance(elevDeg: f32) -> vec3<f32> {
    return mix(sky_night_skySunRadiance(elevDeg), sky_day_skySunRadiance(elevDeg), skyParams[${factor}]);
}
`,
        };
    }

    public getBakeBindEntries(env: EnvironmentSystem): GPUBindGroupEntry[] {
        return [...(this.day.getBakeBindEntries?.(env) ?? []), ...(this.night.getBakeBindEntries?.(env) ?? [])];
    }

    public getSkyboxBindEntries(env: EnvironmentSystem): GPUBindGroupEntry[] {
        return [...(this.day.getSkyboxBindEntries?.(env) ?? []), ...(this.night.getSkyboxBindEntries?.(env) ?? [])];
    }

    public writeParams(out: Float32Array, base: number): void {
        this.day.writeParams(out, base);
        const nightBase = base + this.day.paramFloats;
        this.night.writeParams(out, nightBase);
        out[nightBase + this.night.paramFloats] = this.dayFactor;
    }
}