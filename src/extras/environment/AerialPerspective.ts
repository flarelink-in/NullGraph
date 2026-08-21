// src/extras/environment/AerialPerspective.ts
//
// Phase 5 — aerial perspective and participating media (docs/environment-aerial-design.md).
//
// Phases 1-4 render the air when you look THROUGH all of it. This renders the air between the
// camera and everything else: distant mountains going blue and low-contrast, a valley filling
// with haze.
//
// It is the first phase in the series that touches `src/core/`, and the touch is exactly one
// usage flag (§9.1). `WebGPUContext` creates the depth buffer RENDER_ATTACHMENT-only, so it
// cannot be sampled, and aerial perspective is a function of distance to the surface behind each
// pixel. The flag is opt-in at context creation (`{ samplableDepth: true }`) because adding
// TEXTURE_BINDING can disable some drivers' depth-compression fast paths for every frame whether
// or not this system is enabled. The zero-change fallback — render the scene into an offscreen
// pass with your own depth attachment — works today and is what any scene using PostChain already
// does; only the main-screen convenience path is blocked.

import type { ComputeStageNode, NullGraph, RenderBatch } from '../../core';
import type { RenderPassNode } from '../../core/RenderPass';
import type { ShaderAerialLayout } from '../materials/types';
import type { AtmosphereConfig } from './atmosphereMath';
import { Atmosphere } from './Atmosphere';
import {
    type AerialConfig, type AerialMedia, type HeightFogMedium, type ResolvedAerialConfig,
    DEFAULT_HEIGHT_FOG, resolveAerialConfig, validateHeightFog,
} from './mediaMath';
import {
    AERIAL_APPLY_PARAM_FLOATS, AERIAL_PARAM_FLOATS, AERIAL_SAMPLER_BINDING, AERIAL_VOLUME_BINDING,
    buildAerialApplyWGSL, buildAerialLayout, buildAerialVolumeWGSL, buildHeightFogLayout,
} from './aerialShaders';

export interface AerialPerspectiveOptions extends AerialConfig {
    /**
     * Phase 4's atmosphere. Supplies the Rayleigh/Mie/ozone rows AND the transmittance LUT the
     * froxel march reads for sun visibility. Omit only for a fog-only volume, and then `fog` is
     * mandatory — a volume with no medium is refused (§10).
     */
    atmosphere?: Atmosphere | null;
    /** The height-fog row (§6.1). `true` uses {@link DEFAULT_HEIGHT_FOG}. */
    fog?: HeightFogMedium | boolean | null;
}

export interface AerialApplyTarget {
    /** The scene colour written by the opaque and sky passes. Sampled, not written. */
    source: GPUTextureView;
    /**
     * A SAMPLABLE depth view. Omit to use the engine's own depth texture, which requires the
     * context to have been created with `{ samplableDepth: true }` (§9.1) — otherwise this
     * refuses with the way out spelled in the message.
     */
    depth?: GPUTextureView;
    /** Where the fogged image goes. Must not alias `source`. */
    colorAttachments: GPURenderPassColorAttachment[];
    /**
     * The passes this apply covers — the opaque and sky passes. A batch carrying the injected
     * layout that is drawn in one of them would be fogged twice, and U8 makes that a build-time
     * refusal rather than a runtime check.
     */
    covers?: RenderPassNode[];
    size?: [number, number];
    targetFormat?: GPUTextureFormat;
}

export interface AerialApplyParams {
    /** Runtime toggle — no pipeline recompilation (§5.1). Default true. */
    enabled?: boolean;
    /** Multiplier on the in-scattered term only. Extinction is physical and is not scaled. */
    intensity?: number;
    /** Bilateral rejection threshold, as a fraction of view distance. */
    depthTolerance?: number;
}

// One froxel volume and one analytic height fog on the same engine integrate the same medium, and
// running both applies extinction twice (§6.2). The registry makes that a build-time refusal in
// BOTH directions, because either can be constructed first.
const volumesByEngine = new WeakMap<NullGraph, AerialPerspective>();
const fogsByEngine = new WeakMap<NullGraph, HeightFog>();

const DOUBLE_INTEGRATION_MESSAGE =
    'AerialPerspective: HeightFog is active alongside the froxel volume; both integrate the same ' +
    'medium and extinction would be applied twice (design §6.2). Move the fog parameters onto the ' +
    "volume's media list.";

/** Owns the froxel volume, its build stage, the fullscreen apply pass, and the injected layout. */
export class AerialPerspective {
    public readonly config: ResolvedAerialConfig;

    private readonly engine: NullGraph;
    private readonly device: GPUDevice;
    private readonly atmosphere: Atmosphere | null;
    private readonly volumeTexture: GPUTexture;
    private readonly volumeSampledView: GPUTextureView;
    private readonly volumeStorageView: GPUTextureView;
    private readonly volumeSampler: GPUSampler;
    private readonly paramsBuffer: GPUBuffer;
    private readonly params = new Float32Array(AERIAL_PARAM_FLOATS);
    private readonly applyParams = new Float32Array(AERIAL_APPLY_PARAM_FLOATS);
    private readonly buildStage: ComputeStageNode;
    private readonly stages: ComputeStageNode[];

    private fog: HeightFogMedium | null;
    private readonly sunDir = new Float32Array([0, -1, 0]);
    private sunIntensity: number;

    private applyPass: RenderPassNode | null = null;
    private applyBatch: RenderBatch | null = null;
    private readonly coveredPasses = new Set<RenderPassNode>();
    private readonly injectedBatches = new Map<RenderBatch, { pass: RenderPassNode; name: string }>();
    private stagesIssued = false;

    constructor(engine: NullGraph, options: AerialPerspectiveOptions = {}) {
        const existingFog = fogsByEngine.get(engine);
        if (existingFog) throw new Error(DOUBLE_INTEGRATION_MESSAGE);

        this.engine = engine;
        this.device = engine.device;
        this.config = resolveAerialConfig(options);
        this.atmosphere = options.atmosphere ?? null;

        const fogOption = options.fog;
        this.fog = fogOption === true ? { ...DEFAULT_HEIGHT_FOG }
            : (typeof fogOption === 'object' && fogOption !== null ? fogOption : null);
        if (this.fog) validateHeightFog(this.fog);

        if (!this.atmosphere && !this.fog) {
            throw new Error(
                'AerialPerspective: no medium. Construct with an Atmosphere (Phase 4) or an ' +
                'explicit media list.');
        }
        this.sunIntensity = this.config.sunIntensity;
        if (this.atmosphere) this.setSunDirection(this.atmosphere.sunDirection);

        const [nx, ny, nz] = this.config.dims;
        // 32x32x32 rgba16float = 262,144 bytes. `rgba16float` is a core storage format AND
        // filterable; rgba32float needs float32-filterable and would silently lose the trilinear
        // sampling this whole design rests on (design-principles §12).
        this.volumeTexture = this.device.createTexture({
            label: 'NullGraph_AerialVolume',
            size: [nx, ny, nz],
            dimension: '3d',
            format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
        });
        this.volumeSampledView = this.volumeTexture.createView({ dimension: '3d' });
        this.volumeStorageView = this.volumeTexture.createView({ dimension: '3d' });
        this.volumeSampler = this.device.createSampler({
            label: 'NullGraph_AerialSampler',
            magFilter: 'linear', minFilter: 'linear',
            addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge',
        });
        this.paramsBuffer = this.device.createBuffer({
            label: 'NullGraph_AerialParams',
            size: AERIAL_PARAM_FLOATS * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const atmosphereConfig: AtmosphereConfig | null = this.atmosphere ? this.atmosphere.config : null;
        this.buildStage = engine.createComputeStage({
            name: 'Aerial: froxel volume',
            shaderCode: buildAerialVolumeWGSL({ config: this.config, atmosphere: atmosphereConfig }),
            // One thread per LATERAL froxel; each loops over z. That shape is the §4.3 cost
            // argument, not an optimization detail.
            dispatchCount: [nx / 4, ny / 4, 1],
        });
        engine.setComputePassBindGroup(this.buildStage, [
            { binding: 0, resource: { buffer: engine.ctx.cameraUniformBuffer } },
            { binding: 1, resource: { buffer: this.paramsBuffer } },
            ...(this.atmosphere ? [
                { binding: 2, resource: this.atmosphere.transmittanceView },
                { binding: 3, resource: this.atmosphere.sampler },
            ] : []),
            { binding: 4, resource: this.volumeStorageView },
        ], 0);
        this.stages = [this.buildStage];

        this.applyParams[0] = 1;
        this.applyParams[1] = 1;
        this.applyParams[2] = this.config.upsampleDepthTolerance;
        this.writeParams();
        volumesByEngine.set(engine, this);
    }

    // ---------------------------------------------------------------------------
    // Per-frame state
    // ---------------------------------------------------------------------------

    /** The sun's TRAVEL direction (where the light goes) — the engine-wide convention. */
    public setSunDirection(dir: ArrayLike<number>): void {
        const length = Math.hypot(dir[0], dir[1], dir[2]) || 1;
        this.sunDir[0] = dir[0] / length;
        this.sunDir[1] = dir[1] / length;
        this.sunDir[2] = dir[2] / length;
    }

    /** Linear radiance scale on the sun's in-scatter contribution. */
    public setSunIntensity(intensity: number): void {
        this.sunIntensity = intensity;
    }

    /**
     * Replace the height-fog row (§6.1). `null` removes it; the atmosphere rows are unaffected.
     *
     * There is no separate fog system to enable — this IS the fog, integrated together with
     * everything else the volume marches through. That unification is what makes the §6.2
     * double-extinction bug unreachable rather than merely untested.
     */
    public setHeightFog(fog: HeightFogMedium | null): void {
        if (fog) validateHeightFog(fog);
        if (!fog && !this.atmosphere) {
            throw new Error(
                'AerialPerspective: no medium. Construct with an Atmosphere (Phase 4) or an ' +
                'explicit media list.');
        }
        this.fog = fog;
    }

    /** The active media, in the shape `mediaMath` integrates — the CPU mirror's input. */
    public get media(): AerialMedia {
        return { atmosphere: this.atmosphere ? this.atmosphere.medium : null, fog: this.fog };
    }

    /** Runtime knobs on the fullscreen apply. No pipeline recompilation (§5.1). */
    public setApplyParams(params: AerialApplyParams): void {
        if (params.enabled !== undefined) this.applyParams[0] = params.enabled ? 1 : 0;
        if (params.intensity !== undefined) this.applyParams[1] = params.intensity;
        if (params.depthTolerance !== undefined) this.applyParams[2] = params.depthTolerance;
        if (this.applyBatch) this.engine.updateBatchData(this.applyBatch, this.applyParams, 1);
    }

    // ---------------------------------------------------------------------------
    // §2 — the stage sequence
    // ---------------------------------------------------------------------------

    /**
     * The froxel build stage, to splice BEFORE the scene pass. Consecutive compute stages share
     * one WebGPU pass and a render pass closes it, so this needs no explicit barrier against the
     * draws that follow (RenderPipeline).
     */
    public getStages(): ComputeStageNode[] {
        this.params[0] = this.sunDir[0];
        this.params[1] = this.sunDir[1];
        this.params[2] = this.sunDir[2];
        this.params[3] = this.sunIntensity;
        const fog = this.fog;
        for (let c = 0; c < 3; c++) {
            this.params[4 + c] = fog ? fog.scattering[c] : 0;
            this.params[8 + c] = fog ? fog.absorption[c] : 0;
        }
        this.params[7] = fog ? fog.g : 0;
        // A zero scale height divides by zero in the density profile even when the coefficients
        // are zero, so an absent fog gets a harmless positive one rather than a sentinel.
        this.params[11] = fog ? fog.scaleHeight : 1;
        this.params[12] = fog ? fog.referenceHeight : 0;
        this.writeParams();
        this.stagesIssued = true;
        return this.stages;
    }

    private writeParams(): void {
        this.device.queue.writeBuffer(this.paramsBuffer, 0, this.params as Float32Array<ArrayBuffer>);
    }

    // ---------------------------------------------------------------------------
    // §5.1 — the primary surface: one fullscreen pass over opaque
    // ---------------------------------------------------------------------------

    /**
     * The fullscreen apply pass: reads depth, reconstructs view distance, writes
     * `color * T + inScatter` once per pixel.
     *
     * Cached after the first call, so the checks below run exactly once and never in the frame
     * loop (design-principles §2, "where the rule bends").
     */
    public getApplyPass(target: AerialApplyTarget): RenderPassNode {
        if (this.applyPass) return this.applyPass;

        if (!this.stagesIssued) {
            throw new Error(
                'AerialPerspective: the froxel volume has never been built. Splice ' +
                'aerial.getStages() before the scene pass.');
        }

        const depth = target.depth ?? this.engineDepthView();
        for (const pass of target.covers ?? []) this.addCoveredPass(pass);

        this.applyPass = this.engine.createPass({
            name: 'Aerial: apply',
            isMainScreenPass: false,
            colorAttachments: target.colorAttachments,
            size: target.size,
        });
        this.applyBatch = this.engine.createBatch(this.applyPass, {
            shaderCode: buildAerialApplyWGSL(this.config),
            strideFloats: AERIAL_APPLY_PARAM_FLOATS,
            maxInstances: 1,
            targetFormat: target.targetFormat ?? 'rgba16float',
        });
        this.engine.setRenderPassBindGroup(this.applyBatch, [
            { binding: 0, resource: target.source },
            { binding: 1, resource: depth },
            { binding: 2, resource: this.volumeSampledView },
            { binding: 3, resource: this.volumeSampler },
        ], 1);
        this.engine.updateBatchData(this.applyBatch, this.applyParams, 1);
        return this.applyPass;
    }

    /**
     * The engine's own depth texture, or the §9.1 refusal. This is the one place the required
     * core change is felt, and the message states the flag AND the zero-change fallback because
     * the fallback is the answer for most scenes: post-processing already renders offscreen.
     */
    private engineDepthView(): GPUTextureView {
        const ctx = this.engine.ctx as { depthTexture: GPUTexture; depthSamplable?: boolean };
        const samplable = ctx.depthSamplable
            ?? ((ctx.depthTexture.usage & GPUTextureUsage.TEXTURE_BINDING) !== 0);
        if (!samplable) {
            throw new Error(
                'AerialPerspective: the apply pass needs a samplable depth texture. Create the ' +
                'context with { samplableDepth: true }, or render the scene into an offscreen ' +
                'pass with your own depth attachment (design §9.1).');
        }
        return ctx.depthTexture.createView();
    }

    // ---------------------------------------------------------------------------
    // §5.3 — the secondary surface: the injected layout, for transparents
    // ---------------------------------------------------------------------------

    /**
     * The fourth injected layout, the same shape as `lighting`, `indirect` and `sky`. Emits the
     * same `applyAerial` body the fullscreen pass contains (U6).
     *
     * For TRANSPARENTS. A transparent surface has no single depth in the depth buffer — it may
     * not have written depth at all, and several may overlap one pixel — so each needs aerial
     * perspective at its own distance, which only the fragment path can give it.
     */
    public getShaderLayout(): ShaderAerialLayout {
        return buildAerialLayout(this.config);
    }

    /** The group-2 entries to splice into whichever system builds the combined bind group (§7). */
    public getBindGroupEntries(): GPUBindGroupEntry[] {
        return [
            { binding: AERIAL_VOLUME_BINDING, resource: this.volumeSampledView },
            { binding: AERIAL_SAMPLER_BINDING, resource: this.volumeSampler },
        ];
    }

    /**
     * Declare that `batch` carries the injected layout and is drawn in `pass`.
     *
     * The refusal below is §5.4's claim made enforceable. The two paths visit disjoint pixel sets
     * by construction — the fullscreen pass runs before the transparent pass, and the transparent
     * pass's own fragments are not in the depth buffer it read — but "not possible by
     * construction" is a claim about code that changes, so U8 asserts it anyway.
     */
    public registerBatch(batch: RenderBatch, pass: RenderPassNode, name = 'unnamed'): void {
        if (this.coveredPasses.has(pass)) throw doubleApplyError(name);
        this.injectedBatches.set(batch, { pass, name });
    }

    /** Record a pass as covered by the fullscreen apply, refusing any batch already registered. */
    public addCoveredPass(pass: RenderPassNode): void {
        for (const [, record] of this.injectedBatches) {
            if (record.pass === pass) throw doubleApplyError(record.name);
        }
        this.coveredPasses.add(pass);
    }

    // ---------------------------------------------------------------------------
    // Introspection
    // ---------------------------------------------------------------------------

    /** The froxel volume as a sampled view — for a custom material or a debug view. */
    public get volumeView(): GPUTextureView { return this.volumeSampledView; }
    /** The trilinear sampler the volume is read through. */
    public get sampler(): GPUSampler { return this.volumeSampler; }
    /** The `AerialParams` uniform buffer. */
    public get parametersBuffer(): GPUBuffer { return this.paramsBuffer; }
    /** True once the build stage has been handed out at least once. */
    public get isBuilt(): boolean { return this.stagesIssued; }

    public destroy(): void {
        this.volumeTexture.destroy();
        this.paramsBuffer.destroy();
        this.injectedBatches.clear();
        this.coveredPasses.clear();
        if (volumesByEngine.get(this.engine) === this) volumesByEngine.delete(this.engine);
    }
}

function doubleApplyError(name: string): Error {
    return new Error(
        `AerialPerspective: batch "${name}" carries the injected aerial layout but is drawn in a ` +
        'pass covered by getApplyPass(); aerial perspective would be applied twice. The injected ' +
        'layout is for transparent passes drawn after the apply pass.');
}

// ---------------------------------------------------------------------------
// §6.2 / §11 step 1 — the analytic path, shippable on its own
// ---------------------------------------------------------------------------

export interface HeightFogOptions {
    /** The medium. Defaults to {@link DEFAULT_HEIGHT_FOG}. */
    fog?: HeightFogMedium;
    /** World units in one kilometre. Default 1000. */
    unitsPerKm?: number;
    /** The world Y that is altitude 0. Default 0. */
    groundLevel?: number;
    /** The bind group the layout occupies — must match the LightSystem's. Default 2. */
    aerialGroup?: number;
    /** Sun TRAVEL direction. */
    sunDirection?: ArrayLike<number>;
    /** Sun irradiance reaching the fog, linear. Default [1, 1, 1]. */
    sunIrradiance?: ArrayLike<number>;
}

const HEIGHT_FOG_FLOATS = 20;

/**
 * Closed-form exponential height fog: no volume, no compute stage, and NO SAMPLABLE DEPTH — it is
 * fragment-side and needs nothing from `src/core/`. This is the v1.2 path, and it is genuinely
 * cheaper than a froxel volume for a single medium.
 *
 * It is a drop-in for {@link AerialPerspective.getShaderLayout} at the material site: both define
 * `applyAerialAt(color, worldPos)`, so no material learns which one it got.
 *
 * Constructing one alongside a froxel volume on the same engine is refused in both directions
 * (§6.2). The wrong form —
 *
 *     color * T_fog * T_aerial + inScatter_fog + inScatter_aerial
 *
 * — is dimmer and bluer than either alone, and it is only wrong when both are enabled, so it
 * survives every test that enables one. U10 constructs it deliberately and observes it fail.
 */
export class HeightFog {
    private readonly engine: NullGraph;
    private readonly device: GPUDevice;
    private readonly buffer: GPUBuffer;
    private readonly data = new Float32Array(HEIGHT_FOG_FLOATS);
    private readonly config: ResolvedAerialConfig;
    private fog: HeightFogMedium;

    constructor(engine: NullGraph, options: HeightFogOptions = {}) {
        if (volumesByEngine.get(engine)) throw new Error(DOUBLE_INTEGRATION_MESSAGE);

        this.engine = engine;
        this.device = engine.device;
        this.config = resolveAerialConfig({
            unitsPerKm: options.unitsPerKm,
            groundLevel: options.groundLevel,
            aerialGroup: options.aerialGroup,
        });
        this.fog = options.fog ?? { ...DEFAULT_HEIGHT_FOG };
        validateHeightFog(this.fog);

        this.buffer = this.device.createBuffer({
            label: 'NullGraph_HeightFogParams',
            size: HEIGHT_FOG_FLOATS * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.data[15] = this.config.unitsPerKm;
        this.data[16] = this.config.groundLevel;
        this.setSunDirection(options.sunDirection ?? [0, -1, 0]);
        this.setSunIrradiance(options.sunIrradiance ?? [1, 1, 1]);
        this.setFog(this.fog);
        fogsByEngine.set(engine, this);
    }

    public setFog(fog: HeightFogMedium): void {
        validateHeightFog(fog);
        this.fog = fog;
        for (let c = 0; c < 3; c++) {
            const extinction = fog.scattering[c] + fog.absorption[c];
            this.data[c] = extinction;
            this.data[4 + c] = extinction > 0 ? fog.scattering[c] / extinction : 0;
        }
        this.data[3] = fog.scaleHeight;
        this.data[7] = fog.referenceHeight;
        this.data[11] = fog.g;
        this.write();
    }

    public setSunDirection(dir: ArrayLike<number>): void {
        const length = Math.hypot(dir[0], dir[1], dir[2]) || 1;
        this.data[12] = dir[0] / length;
        this.data[13] = dir[1] / length;
        this.data[14] = dir[2] / length;
        this.write();
    }

    public setSunIrradiance(irradiance: ArrayLike<number>): void {
        this.data[8] = irradiance[0];
        this.data[9] = irradiance[1];
        this.data[10] = irradiance[2];
        this.write();
    }

    private write(): void {
        this.device.queue.writeBuffer(this.buffer, 0, this.data as Float32Array<ArrayBuffer>);
    }

    public getShaderLayout(): ShaderAerialLayout {
        if (volumesByEngine.get(this.engine)) throw new Error(DOUBLE_INTEGRATION_MESSAGE);
        return buildHeightFogLayout(this.config);
    }

    /** The single group-2 entry, at the same binding the froxel volume would occupy. */
    public getBindGroupEntries(): GPUBindGroupEntry[] {
        return [{ binding: AERIAL_VOLUME_BINDING, resource: { buffer: this.buffer } }];
    }

    /** The active medium, in the shape `mediaMath` integrates. */
    public get medium(): HeightFogMedium { return this.fog; }
    /** The `HeightFogParams` uniform buffer. */
    public get parametersBuffer(): GPUBuffer { return this.buffer; }

    public destroy(): void {
        this.buffer.destroy();
        if (fogsByEngine.get(this.engine) === this) fogsByEngine.delete(this.engine);
    }
}
