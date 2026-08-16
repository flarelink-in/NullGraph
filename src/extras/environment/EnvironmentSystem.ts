// src/extras/environment/EnvironmentSystem.ts
//
// The EnvironmentSystem (docs/lighting-ibl-design.md): Phase 1 image-based lighting. It replaces
// the constant that currently stands in for the entire indirect term in the PBR fragment
// (`let ambient = vec3<f32>(0.3) * albedo * ao;`) with the split-sum model — a prefiltered
// environment cube for specular, an SH-L2 projection for diffuse, and a BRDF integration LUT to
// join them.
//
// The bake is a chain of ComputeStageNodes spliced in once (design doc §6). The frame path is a
// bind group and two function calls in the fragment shader; in `updateMode: 'once'` the stages
// retire after they run and the per-frame cost is exactly zero.
//
// Everything here is core WebGPU: rgba16float / rg16float storage textures, 2d-array storage
// views, cube sampling views, textureSampleLevel in compute. No optional feature, no core-engine
// change (design doc §12.1).

import { NullGraph, RenderBatch, ComputeStageNode, RenderPassNode } from '../../core';
import type { LightSystem } from '../lights/LightSystem';
import {
    ENV_PARAMS_FLOATS, SH_SOURCE_SIZE, packEnvParams, validateEnvironmentConfig,
} from './envMath';
import {
    buildSourceCubeWGSL, buildCubeMipWGSL, buildPrefilterWGSL, buildBRDFLUTWGSL,
    buildSHProjectWGSL, buildIndirectStructsWGSL, buildIndirectLogicWGSL, buildSkyWGSL,
} from './envShaders';
import type { EnvironmentConfig, ProceduralSkyConfig, SkyParams } from './types';

/** The `{ wgslStructs, wgslLogic }` pair materials compose — the sibling of ShaderLightingLayout. */
export interface ShaderIndirectLayout {
    wgslStructs: string;
    wgslLogic: string;
}

/**
 * Floats of `EnvParams` the SH block occupies (0..35): 3 channels × 3 vec4. The GPU projection
 * owns this region; the CPU owns the tail. Splitting them is what lets a knob change and a bake
 * coexist without one overwriting the other.
 */
const SH_PACKED_FLOATS = 36;

/** Floats in the procedural-sky uniform. vec3 aligns to 16 in WGSL, so this is 16, not 13. */
const SKY_UNIFORM_FLOATS = 16;

/** Floats the sky batch's storage buffer carries: [exposure, yaw, blurMip, _] (§8.2). */
const SKY_PARAM_FLOATS = 4;

export class EnvironmentSystem {
    private readonly engine: NullGraph;
    private readonly device: GPUDevice;
    private readonly cfg: Required<Omit<EnvironmentConfig, never>>;

    // --- GPU resources, allocated once ---
    private sourceCube!: GPUTexture;
    private sourceCubeView!: GPUTextureView;          // dimension: 'cube', all mips
    private specularCube!: GPUTexture;
    private specularCubeView!: GPUTextureView;        // dimension: 'cube', all mips
    private brdfLUT!: GPUTexture;
    private brdfLUTView!: GPUTextureView;
    private envSampler!: GPUSampler;
    private envParamsBuffer!: GPUBuffer;
    private shBuffer!: GPUBuffer;                     // 27 f32, written by the SH projection
    private skyUniform: GPUBuffer | null = null;      // procedural sky only

    // --- bake stages ---
    private bakeStages: ComputeStageNode[] = [];
    private prefilterStages: ComputeStageNode[] = [];
    private shStage: ComputeStageNode | null = null;
    private sourceSet: boolean = false;
    private baked: boolean = false;
    private frame: number = 0;
    private amortizeCursor: number = 0;

    // --- sky pass ---
    private skyPass: RenderPassNode | null = null;
    private skyBatch: RenderBatch | null = null;

    // --- per-frame staging (persistent — the frame path allocates nothing) ---
    private readonly envParamsF32 = new Float32Array(ENV_PARAMS_FLOATS);
    private readonly skyParamsF32 = new Float32Array(SKY_PARAM_FLOATS);
    private readonly skyUniformF32 = new Float32Array(SKY_UNIFORM_FLOATS);
    /** SH coefficients as the CPU last knew them. The GPU owns the authoritative copy. */
    private readonly shCoeffs = new Float32Array(27);

    private yaw: number = 0;
    /** Batches whose group-2 bind group this system built, for the §11 double-bind guard. */
    private readonly boundBatches = new WeakSet<RenderBatch>();

    constructor(engine: NullGraph, config: EnvironmentConfig = {}) {
        this.engine = engine;
        this.device = engine.device;

        this.cfg = {
            sourceSize: config.sourceSize ?? 512,
            prefilterMips: config.prefilterMips ?? 6,
            prefilterSamples: config.prefilterSamples ?? 128,
            lutSize: config.lutSize ?? 256,
            intensity: config.intensity ?? 1.0,
            horizonFade: config.horizonFade ?? 1.0,
            updateMode: config.updateMode ?? 'once',
            keepSourceCube: config.keepSourceCube ?? (config.updateMode === 'amortized'),
            environmentGroup: config.environmentGroup ?? 2,
        };

        // Fail loud, naming the offender and the way out (design-principles §2).
        validateEnvironmentConfig({
            sourceSize: this.cfg.sourceSize,
            prefilterMips: this.cfg.prefilterMips,
            environmentGroup: this.cfg.environmentGroup,
            updateMode: this.cfg.updateMode,
            keepSourceCube: this.cfg.keepSourceCube,
        });

        this.allocate();
    }

    // ---------------------------------------------------------------------------
    // Allocation (§4)
    // ---------------------------------------------------------------------------

    private allocate(): void {
        const size = this.cfg.sourceSize;
        const sourceMips = Math.log2(size) + 1;

        // The source cube: sampled as a cube during prefilter/SH, written per-mip as a 2d-array.
        // Two views over one texture — the HiZPyramid pattern, in 3D (§4.1).
        this.sourceCube = this.device.createTexture({
            label: 'NullGraph_EnvSourceCube',
            size: [size, size, 6],
            format: 'rgba16float',
            mipLevelCount: sourceMips,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
        });
        this.sourceCubeView = this.sourceCube.createView({ dimension: 'cube' });

        // The prefiltered specular cube: mip L = roughness L/(prefilterMips-1).
        const specSize = 1 << (this.cfg.prefilterMips - 1 + 2); // 128 at the default 6 mips
        this.specularCube = this.device.createTexture({
            label: 'NullGraph_EnvSpecularCube',
            size: [specSize, specSize, 6],
            format: 'rgba16float',
            mipLevelCount: this.cfg.prefilterMips,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
        });
        this.specularCubeView = this.specularCube.createView({ dimension: 'cube' });

        this.brdfLUT = this.device.createTexture({
            label: 'NullGraph_EnvBRDFLUT',
            size: [this.cfg.lutSize, this.cfg.lutSize],
            format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
        });
        this.brdfLUTView = this.brdfLUT.createView();

        // Linear + mip filtering: the specular lookup interpolates BETWEEN roughness mips, so
        // mipmapFilter must be 'linear' or a roughness sweep steps visibly at the mip boundaries.
        this.envSampler = this.device.createSampler({
            label: 'NullGraph_EnvSampler',
            magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
            addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
        });

        this.envParamsBuffer = this.device.createBuffer({
            label: 'NullGraph_EnvParams',
            size: ENV_PARAMS_FLOATS * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        // 36 floats, matching EnvParams floats 0..35 exactly (the packed 3-per-vec4 SH layout),
        // so `copySHToParams` is one buffer-to-buffer copy rather than a readback.
        this.shBuffer = this.device.createBuffer({
            label: 'NullGraph_EnvSH',
            size: SH_PACKED_FLOATS * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        // EnvParams starts valid: a zeroed SH set is a black environment, which is a legitimate
        // "nothing baked yet" state rather than uninitialised memory in a uniform.
        this.writeEnvParams();
    }

    // ---------------------------------------------------------------------------
    // Sources (§6.1)
    // ---------------------------------------------------------------------------

    /**
     * Analytic gradient sky + sun disc. The zero-asset path: the whole bake chain is exercisable
     * without finding an HDR file, which is what keeps IBL from being blocked on asset sourcing.
     */
    public setProceduralSky(sky: ProceduralSkyConfig): void {
        const u = this.skyUniformF32;
        const d = sky.sunDir;
        const l = Math.hypot(d[0], d[1], d[2]) || 1;
        // WGSL uniform layout: each vec3 aligns to 16 bytes, so the scalars ride in the gaps.
        u[0] = sky.zenith[0];  u[1] = sky.zenith[1];  u[2] = sky.zenith[2];
        u[3] = sky.sunCos ?? 0.9995;
        u[4] = sky.horizon[0]; u[5] = sky.horizon[1]; u[6] = sky.horizon[2];
        u[7] = sky.sunIntensity;
        u[8] = sky.ground[0];  u[9] = sky.ground[1];  u[10] = sky.ground[2];
        u[11] = 0;
        u[12] = d[0] / l; u[13] = d[1] / l; u[14] = d[2] / l;
        u[15] = 0;

        if (!this.skyUniform) {
            this.skyUniform = this.device.createBuffer({
                label: 'NullGraph_EnvProceduralSky',
                size: SKY_UNIFORM_FLOATS * 4,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
        }
        this.device.queue.writeBuffer(this.skyUniform, 0, u as Float32Array<ArrayBuffer>);
        this.buildBakeChain({ procedural: true });
        this.sourceSet = true;
    }

    /**
     * Load an equirectangular HDR/LDR image and convert it to the source cube. Rejects loudly on
     * failure — silently falling back would ship a black environment attributed to the wrong
     * subsystem (§11).
     */
    public async setEquirect(url: string, options: { sRGB?: boolean } = {}): Promise<void> {
        let bitmap: ImageBitmap;
        try {
            const res = await fetch(url);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            bitmap = await createImageBitmap(await res.blob(), { colorSpaceConversion: 'none' });
        } catch (e) {
            throw new Error(
                `EnvironmentSystem: failed to load equirect "${url}" (${e instanceof Error ? e.message : e}); ` +
                `falling back would silently ship a black environment — set a procedural sky explicitly instead.`);
        }
        const tex = this.device.createTexture({
            label: 'NullGraph_EnvEquirect',
            size: [bitmap.width, bitmap.height],
            format: options.sRGB ? 'rgba8unorm-srgb' : 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.device.queue.copyExternalImageToTexture({ source: bitmap }, { texture: tex },
            [bitmap.width, bitmap.height]);
        this.buildBakeChain({ procedural: false, equirect: tex.createView() });
        this.sourceSet = true;
    }

    // ---------------------------------------------------------------------------
    // The bake chain (§6)
    // ---------------------------------------------------------------------------

    private buildBakeChain(src: { procedural: boolean; equirect?: GPUTextureView }): void {
        const size = this.cfg.sourceSize;
        const sourceMips = Math.log2(size) + 1;
        const stages: ComputeStageNode[] = [];

        // --- Stage A: source cube mip 0 ---
        const stageA = this.engine.createComputeStage({
            name: 'Env: source cube',
            shaderCode: buildSourceCubeWGSL({ procedural: src.procedural }),
            dispatchCount: [Math.ceil(size / 8), Math.ceil(size / 8), 6],
        });
        const dst0 = this.sourceCube.createView({
            dimension: '2d-array', baseMipLevel: 0, mipLevelCount: 1,
        });
        this.engine.setComputePassBindGroup(stageA, src.procedural
            ? [{ binding: 0, resource: { buffer: this.skyUniform! } },
               { binding: 2, resource: dst0 }]
            : [{ binding: 0, resource: src.equirect! },
               { binding: 1, resource: this.envSampler },
               { binding: 2, resource: dst0 }], 0);
        stages.push(stageA);

        // --- Stage B: the source mip chain (box average) ---
        // Each mip reads the level above through the cube view. Successive dispatches in one
        // compute pass are ordered by WebGPU's sync model, so the chain builds in one pass.
        for (let level = 1; level < sourceMips; level++) {
            const dim = Math.max(1, size >> level);
            const stage = this.engine.createComputeStage({
                name: `Env: source mip ${level}`,
                shaderCode: buildCubeMipWGSL(),
                dispatchCount: [Math.ceil(dim / 8), Math.ceil(dim / 8), 6],
            });
            this.engine.setComputePassBindGroup(stage, [
                { binding: 0, resource: this.sourceCube.createView({ dimension: 'cube', baseMipLevel: level - 1, mipLevelCount: 1 }) },
                { binding: 1, resource: this.sourceCube.createView({ dimension: '2d-array', baseMipLevel: level, mipLevelCount: 1 }) },
                { binding: 2, resource: this.envSampler },
            ], 0);
            stages.push(stage);
        }

        // --- Stage C: GGX prefilter, one stage per specular mip ---
        this.prefilterStages = [];
        const specBase = this.specularCube.width;
        for (let level = 0; level < this.cfg.prefilterMips; level++) {
            const dim = Math.max(1, specBase >> level);
            const stage = this.engine.createComputeStage({
                name: `Env: prefilter mip ${level}`,
                shaderCode: buildPrefilterWGSL({
                    level, prefilterMips: this.cfg.prefilterMips,
                    samples: this.cfg.prefilterSamples, srcSize: size,
                }),
                dispatchCount: [Math.ceil(dim / 8), Math.ceil(dim / 8), 6],
            });
            this.engine.setComputePassBindGroup(stage, [
                { binding: 0, resource: this.sourceCubeView },
                { binding: 1, resource: this.envSampler },
                { binding: 2, resource: this.specularCube.createView({ dimension: '2d-array', baseMipLevel: level, mipLevelCount: 1 }) },
            ], 0);
            this.prefilterStages.push(stage);
            stages.push(stage);
        }

        // --- Stage D: the BRDF LUT (scene-independent; baked once) ---
        if (!this.baked) {
            const lut = this.engine.createComputeStage({
                name: 'Env: BRDF LUT',
                shaderCode: buildBRDFLUTWGSL({ samples: 1024 }),
                dispatchCount: [Math.ceil(this.cfg.lutSize / 8), Math.ceil(this.cfg.lutSize / 8), 1],
            });
            this.engine.setComputePassBindGroup(lut, [{ binding: 0, resource: this.brdfLUTView }], 0);
            stages.push(lut);
        }

        // --- SH projection: one workgroup over the 8×8 mip (§7.2) ---
        const shMipLevel = Math.log2(size / SH_SOURCE_SIZE);
        const sh = this.engine.createComputeStage({
            name: 'Env: SH-L2 projection',
            shaderCode: buildSHProjectWGSL(),
            dispatchCount: [1, 1, 1],
        });
        this.engine.setComputePassBindGroup(sh, [
            { binding: 0, resource: this.sourceCube.createView({ dimension: '2d-array', baseMipLevel: shMipLevel, mipLevelCount: 1 }) },
            { binding: 1, resource: { buffer: this.shBuffer } },
        ], 0);
        this.shStage = sh;
        stages.push(sh);

        this.bakeStages = stages;
        this.baked = false;
        this.frame = 0;
        this.amortizeCursor = 0;
    }

    /**
     * The bake stages to splice into the render sequence. In `'once'` mode they run on the first
     * frame and this returns an empty array thereafter — the per-frame cost really is zero, not
     * "small" (§6.5, T5). In `'amortized'` mode it returns the SH stage plus ONE prefilter mip
     * per frame, cycling, so an animated sky fully refreshes in `prefilterMips` frames.
     */
    public getBakeStages(): ComputeStageNode[] {
        if (!this.sourceSet) {
            throw new Error(
                `EnvironmentSystem: getBakeStages() called before a source was set; ` +
                `call setEquirect() or setProceduralSky() first.`);
        }
        if (!this.baked) {
            this.baked = true;
            this.frame++;
            return this.bakeStages;
        }
        if (this.cfg.updateMode !== 'amortized') return EMPTY_STAGES;

        // Amortized: the source cube + one prefilter mip + the SH projection.
        const mip = this.prefilterStages[this.amortizeCursor % this.prefilterStages.length];
        this.amortizeCursor++;
        this.amortizedSlice[0] = this.bakeStages[0];      // stage A — the source, in case it moved
        this.amortizedSlice[1] = mip;
        this.amortizedSlice[2] = this.shStage!;
        return this.amortizedSlice;
    }

    private readonly amortizedSlice: ComputeStageNode[] = new Array(3);

    // ---------------------------------------------------------------------------
    // The material bridge (§5)
    // ---------------------------------------------------------------------------

    /**
     * The Phase 1 `ShaderIndirectLayout`. Phase 2 re-emits these same two functions against a
     * probe volume, which is why `getIndirectDiffuse`/`getIndirectSpecular` take `worldPos` even
     * though Phase 1 ignores it — the seam is drawn before it is needed (design-principles §4).
     */
    public getShaderLayout(): ShaderIndirectLayout {
        if (!this.sourceSet) {
            throw new Error(
                `EnvironmentSystem: getShaderLayout() called before a source was set; ` +
                `call setEquirect() or setProceduralSky() first.`);
        }
        return {
            wgslStructs: buildIndirectStructsWGSL(this.cfg.environmentGroup),
            wgslLogic: buildIndirectLogicWGSL(),
        };
    }

    /**
     * Build the COMBINED group-2 bind group — the lights' entries plus the environment's.
     *
     * Taking the `LightSystem` is deliberate and slightly unusual: group 2 has two owners (three
     * once probes land) and core WebGPU guarantees only four bind groups, so exactly one call must
     * build the whole group or they overwrite each other (§3, §9).
     */
    public applyToBatch(batch: RenderBatch, lights: LightSystem): void {
        const g = this.cfg.environmentGroup;
        if (batch.customBindGroups[g] && !this.boundBatches.has(batch)) {
            throw new Error(
                `EnvironmentSystem: batch already has a group-${g} bind group from ` +
                `LightSystem.applyToBatch(); use EnvironmentSystem.applyToBatch(batch, lights) to ` +
                `build the combined group instead.`);
        }
        this.engine.setRenderPassBindGroup(batch, [
            ...lights.getBindGroupEntries(),
            { binding: 3, resource: this.envSampler },
            { binding: 4, resource: this.specularCubeView },
            { binding: 5, resource: this.brdfLUTView },
            { binding: 6, resource: { buffer: this.envParamsBuffer } },
        ], g);
        this.boundBatches.add(batch);
    }

    // ---------------------------------------------------------------------------
    // The skybox pass (§8)
    // ---------------------------------------------------------------------------

    /**
     * A depth-tested background drawn AFTER the opaque pass. No vertex or index buffer, so
     * `RenderPipeline` takes its `draw(3, instanceCount)` fullscreen path — no engine change.
     */
    public getSkyPass(target: {
        colorAttachments: GPURenderPassColorAttachment[];
        depthStencilAttachment?: GPURenderPassDepthStencilAttachment;
        size?: [number, number];
        targetFormat?: GPUTextureFormat;
    }): RenderPassNode {
        if (this.skyPass) return this.skyPass;

        const skyColorAttachments: GPURenderPassColorAttachment[] = target.colorAttachments.map(att => ({
            ...att,
            loadOp: 'load',
            storeOp: 'store',
        }));

        const skyDepthAttachment: GPURenderPassDepthStencilAttachment | undefined = target.depthStencilAttachment ? {
            ...target.depthStencilAttachment,
            depthLoadOp: 'load',
            depthStoreOp: 'store',
        } : undefined;

        this.skyPass = this.engine.createPass({
            name: 'Env: Sky',
            isMainScreenPass: false,
            colorAttachments: skyColorAttachments,
            depthStencilAttachment: skyDepthAttachment,
            size: target.size,
        });
        this.skyBatch = this.engine.createBatch(this.skyPass, {
            shaderCode: buildSkyWGSL({ environmentGroup: this.cfg.environmentGroup }),
            strideFloats: SKY_PARAM_FLOATS,
            maxInstances: 1,
            targetFormat: target.targetFormat ?? 'rgba16float',
            depthWriteEnabled: false,
            depthCompare: 'less-equal',   // only where nothing was drawn (cleared depth = 1.0)
        });
        // Group 2 for the sky needs only the sampler + the cube; it does not read the lights.
        this.engine.setRenderPassBindGroup(this.skyBatch, [
            { binding: 3, resource: this.envSampler },
            { binding: 4, resource: this.specularCubeView },
        ], this.cfg.environmentGroup);
        this.setSkyParams({});
        return this.skyPass;
    }

    /** Exposure / yaw / blur mip for the sky, written through the batch's storage buffer (§8.2). */
    public setSkyParams(p: SkyParams): void {
        const s = this.skyParamsF32;
        s[0] = p.exposure ?? (s[0] || 1.0);
        s[1] = p.yaw ?? this.yaw;
        s[2] = p.blurMip ?? (s[2] || 0);
        s[3] = 0;
        if (this.skyBatch) this.engine.updateBatchData(this.skyBatch, s, 1);
    }

    // ---------------------------------------------------------------------------
    // Per-frame state
    // ---------------------------------------------------------------------------

    /** Rotate the environment about +Y. Applies to the diffuse SH and the specular lookup alike. */
    public setRotation(yawRadians: number): void {
        this.yaw = yawRadians;
        this.writeEnvParams();
        this.setSkyParams({ yaw: yawRadians });
    }

    /** Scale the whole indirect contribution. */
    public setIntensity(intensity: number): void {
        this.cfg.intensity = intensity;
        this.writeEnvParams();
    }

    /**
     * Adopt SH coefficients computed elsewhere (a Worker, or a previous session's bake) — the
     * `LightSystem.setRawData` contract. 27 floats, 9 vec3 in the envMath order.
     */
    public setRawSH(coeffs: Float32Array): void {
        if (coeffs.length < 27) {
            throw new Error(`EnvironmentSystem: setRawSH expects 27 floats (9 vec3), got ${coeffs.length}.`);
        }
        this.shCoeffs.set(coeffs.subarray(0, 27));
        // writeEnvParams deliberately skips the SH block (the GPU owns it), so a CPU-supplied set
        // is packed and pushed into floats 0..35 directly.
        packEnvParams(this.shCoeffs, this.yaw, this.cfg.intensity,
            this.cfg.prefilterMips - 1, this.cfg.horizonFade, this.envParamsF32);
        this.device.queue.writeBuffer(this.envParamsBuffer, 0,
            this.envParamsF32 as Float32Array<ArrayBuffer>, 0, SH_PACKED_FLOATS);
        this.writeEnvParams();
    }

    /**
     * Upload the NON-SH tail of `EnvParams` — floats 36..43 (rotation, intensity, specularMip,
     * horizonFade).
     *
     * Deliberately partial: floats 0..35 are the SH block, and the GPU projection owns those
     * (`copySHToParams`). Writing the whole struct here would clobber the projected coefficients
     * with the CPU's stale copy every time someone nudged the intensity — a bug whose symptom is
     * "the environment goes black when I change a knob."
     */
    private writeEnvParams(): void {
        packEnvParams(this.shCoeffs, this.yaw, this.cfg.intensity,
            this.cfg.prefilterMips - 1, this.cfg.horizonFade, this.envParamsF32);
        this.device.queue.writeBuffer(
            this.envParamsBuffer, SH_PACKED_FLOATS * 4,
            this.envParamsF32 as Float32Array<ArrayBuffer>,
            SH_PACKED_FLOATS, ENV_PARAMS_FLOATS - SH_PACKED_FLOATS);
    }

    /**
     * Move the GPU-projected SH coefficients into the `EnvParams` uniform.
     *
     * The projection kernel writes a storage buffer; the fragment shader reads a uniform. Rather
     * than stall the frame on a readback (forbidden — design-principles §12), the kernel writes
     * the **exact packed layout** of EnvParams floats 0..35, so this is a single 144-byte
     * `copyBufferToBuffer`. Queue ordering guarantees it lands after the dispatch that produced it.
     *
     * Call after `engine.render([...getBakeStages()])`. Cheap enough to call every frame; it is a
     * no-op before the first bake.
     */
    public copySHToParams(): void {
        if (!this.baked) return;
        const enc = this.device.createCommandEncoder({ label: 'NullGraph_EnvSHCopy' });
        enc.copyBufferToBuffer(this.shBuffer, 0, this.envParamsBuffer, 0, SH_PACKED_FLOATS * 4);
        this.device.queue.submit([enc.finish()]);
    }

    // ---------------------------------------------------------------------------
    // Introspection
    // ---------------------------------------------------------------------------

    /** The prefiltered specular cube view — for a custom material or a debug view. */
    public get specularView(): GPUTextureView { return this.specularCubeView; }
    /** The BRDF integration LUT view. */
    public get brdfView(): GPUTextureView { return this.brdfLUTView; }
    /** The environment sampler (linear + mip filtering). */
    public get sampler(): GPUSampler { return this.envSampler; }
    /** The configured HDR intensity — probe capture reads it to scale its sky term
     *  (lighting-probes-design.md §7.2: a capture ray that hits nothing sees the environment). */
    public get intensity(): number { return this.cfg.intensity; }
    /** The `EnvParams` uniform buffer. */
    public get paramsBuffer(): GPUBuffer { return this.envParamsBuffer; }
    /** The raw SH storage buffer the projection kernel writes (27 f32). */
    public get shStorageBuffer(): GPUBuffer { return this.shBuffer; }
    /** True once the bake stages have been handed out at least once. */
    public get isBaked(): boolean { return this.baked; }

    public destroy(): void {
        this.sourceCube.destroy();
        this.specularCube.destroy();
        this.brdfLUT.destroy();
        this.envParamsBuffer.destroy();
        this.shBuffer.destroy();
        this.skyUniform?.destroy();
        this.bakeStages = [];
        this.prefilterStages = [];
        this.shStage = null;
    }
}

/** Shared empty array — returning a fresh `[]` per frame would allocate (constraint #2). */
const EMPTY_STAGES: ComputeStageNode[] = [];
