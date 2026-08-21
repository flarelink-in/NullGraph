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
import { SKY_PARAM_FLOATS } from './skyMath';
import { GradientSky, EquirectSky } from './SkySource';
import type {
    EnvironmentConfig, ProceduralSkyConfig, SkyParams, SkySource, ShaderSkyLayout,
} from './types';

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
    /**
     * The Stage-A params buffer: the same `[header + source params]` layout as the sky batch's
     * storage buffer, so the injected evaluator reads identical indices in BOTH consumers
     * (environment-sky-design.md §6.1–6.2). Rebuilt by setSky when the source's size changes.
     */
    private bakeParamsBuffer: GPUBuffer | null = null;
    /** Lifetime owner of the equirect texture `setEquirect` loads (its view rides in the source). */
    private equirectTexture: GPUTexture | null = null;

    // --- the sky source (environment-sky-design.md §4) ---
    private skySource: SkySource | null = null;
    private skyLayout: ShaderSkyLayout | null = null;       // skybox site
    private skyBakeLayout: ShaderSkyLayout | null = null;   // bake site (== skyLayout unless overridden)
    private skyStrideFloats: number = SKY_PARAM_FLOATS;

    // --- bake stages ---
    private bakeStages: ComputeStageNode[] = [];
    private prefilterStages: ComputeStageNode[] = [];
    private shStage: ComputeStageNode | null = null;
    private sourceSet: boolean = false;
    private baked: boolean = false;
    private frame: number = 0;
    private amortizeCursor: number = 0;
    /** Remaining mixed-age prefilter slices after declared invalidation (Phase 2 time of day §6). */
    private invalidatedSlices: number = 0;

    // --- sky pass ---
    private skyPass: RenderPassNode | null = null;
    private skyBatch: RenderBatch | null = null;
    private skyPassTargetFormat: GPUTextureFormat = 'rgba16float';

    // --- per-frame staging (persistent — the frame path allocates nothing) ---
    private readonly envParamsF32 = new Float32Array(ENV_PARAMS_FLOATS);
    /** Widened by setSky to SKY_PARAM_FLOATS + source.paramFloats. */
    private skyParamsF32: Float32Array = new Float32Array(SKY_PARAM_FLOATS);
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
    // Sources (environment-sky-design.md §4, §7)
    // ---------------------------------------------------------------------------

    /**
     * The current sky source — the object, not a method call (design §7). Typed per source, so
     * `env.sky.setSunDirection(dir)` needs no cast on the sun-bearing ones.
     */
    public get sky(): SkySource {
        if (!this.skySource) {
            throw new Error(
                `EnvironmentSystem: sky accessed before a source was set; call setSky() first.`);
        }
        return this.skySource;
    }

    /**
     * Install a sky source. A source is data, not a code path (design principle 2): this method
     * never branches on the source's type — it injects the source's layout into the bake kernel
     * and the skybox, sizes the params buffers by demand, and rebuilds the bake chain.
     *
     * All failures are build-time and name the offender (§9): a source whose logic lacks
     * `sampleSky`, a backtick in generated WGSL, a cube-backed source on a system that releases
     * the cube, or a source swap after the skybox pipeline was compiled.
     */
    public setSky(source: SkySource): void {
        if (this.skyBatch) {
            throw new Error(
                `EnvironmentSystem: setSky(${source.name}) must be called before the first ` +
                `getSkyPass(); the compiled skybox pipeline has the previous source layout.`);
        }

        const base = SKY_PARAM_FLOATS;
        const layout = source.getShaderLayout(base);
        const bakeLayout = source.getBakeShaderLayout?.(base) ?? layout;
        for (const [which, l] of [['skybox', layout], ['bake', bakeLayout]] as const) {
            if (!l.wgslLogic.includes('fn sampleSky(')) {
                throw new Error(
                    `SkySource "${source.name}" does not define sampleSky(dir: vec3<f32>) -> ` +
                    `vec3<f32> (its ${which} wgslLogic lacks it).`);
            }
            if (l.wgslLogic.includes('`') || l.wgslStructs.includes('`')) {
                throw new Error(
                    `SkySource "${source.name}" emitted a backtick, which terminates the template ` +
                    `literal carrying it. Use single quotes in WGSL comments.`);
            }
        }
        // A cube-backed skybox reads the source cube after the bake; if this system releases it,
        // the background is black — fail loud, naming the flag, instead (§5.2, §9).
        if (source.needsSourceCube && !this.cfg.keepSourceCube) {
            throw new Error(
                `EnvironmentSystem: ${source.name} samples the source cube, but keepSourceCube ` +
                `is false — the cube is released after the bake. Construct with { keepSourceCube: true }.`);
        }

        this.skySource = source;
        this.skyLayout = layout;
        this.skyBakeLayout = bakeLayout;

        // Size the shared params staging + the Stage-A buffer by demand, not by a cap (§4.5).
        const stride = SKY_PARAM_FLOATS + source.paramFloats;
        this.skyStrideFloats = stride;
        this.skyParamsF32 = new Float32Array(stride);
        this.skyParamsF32[0] = 1.0;                    // exposure
        this.skyParamsF32[1] = this.yaw;               // yaw
        this.skyParamsF32[2] = 0.0;                    // blurMip (cheapSky mode only)
        // cheapSky 0 — the skybox EVALUATES sampleSky per pixel. This is the phase's deliverable
        // (§1): the visible sky is no longer capped at the prefiltered cube's 0.70°/texel, so a
        // true 0.266° sun disc is representable. `setSkyParams({ cheapSky: 1 })` restores the
        // pre-Phase-1 cube lookup, which is what `blurMip` blurs and the right answer for a
        // deliberately soft backdrop.
        this.skyParamsF32[3] = 0.0;
        source.writeParams(this.skyParamsF32, base);

        if (this.bakeParamsBuffer) this.bakeParamsBuffer.destroy();
        this.bakeParamsBuffer = this.device.createBuffer({
            label: 'NullGraph_EnvSkyBakeParams',
            size: stride * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.bakeParamsBuffer, 0,
            this.skyParamsF32 as Float32Array<ArrayBuffer>);

        this.buildBakeChain();
        this.sourceSet = true;
    }

    /**
     * Analytic gradient sky + sun disc. The zero-asset path: the whole bake chain is exercisable
     * without finding an HDR file. Forwards to `setSky(new GradientSky(cfg))` — the convenience
     * layer, not the mechanism (design §7).
     */
    public setProceduralSky(sky: ProceduralSkyConfig): void {
        this.setSky(new GradientSky(sky));
    }

    /**
     * Load an equirectangular HDR/LDR image and convert it to the source cube. Rejects loudly on
     * failure — silently falling back would ship a black environment attributed to the wrong
     * subsystem (§11). Forwards to `setSky(new EquirectSky(...))`; an equirect sky needs
     * `keepSourceCube: true` — its skybox samples the source cube (§5.2).
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
        this.equirectTexture?.destroy();
        this.equirectTexture = tex;
        this.setSky(new EquirectSky({ equirect: tex.createView() }));
    }

    // ---------------------------------------------------------------------------
    // The bake chain (§6)
    // ---------------------------------------------------------------------------

    private buildBakeChain(): void {
        const size = this.cfg.sourceSize;
        const sourceMips = Math.log2(size) + 1;
        const stages: ComputeStageNode[] = [];

        // --- Stage A: source cube mip 0, from the source's injected evaluator (§6.1) ---
        const stageA = this.engine.createComputeStage({
            name: 'Env: source cube',
            shaderCode: buildSourceCubeWGSL(this.skyBakeLayout!),
            dispatchCount: [Math.ceil(size / 8), Math.ceil(size / 8), 6],
        });
        const dst0 = this.sourceCube.createView({
            dimension: '2d-array', baseMipLevel: 0, mipLevelCount: 1,
        });
        // layout:'auto' prunes unread bindings: the params buffer is supplied only when the
        // source reads it (paramFloats > 0 — the documented SkySource contract).
        const entries: GPUBindGroupEntry[] = [];
        if (this.skySource!.paramFloats > 0) {
            entries.push({ binding: 0, resource: { buffer: this.bakeParamsBuffer! } });
        }
        entries.push({ binding: 2, resource: dst0 });
        const bakeExtras = this.skySource!.getBakeBindEntries?.(this);
        if (bakeExtras) entries.push(...bakeExtras);
        this.engine.setComputePassBindGroup(stageA, entries, 0);
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
        this.invalidatedSlices = 0;
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
            this.invalidatedSlices = 0;
            return this.bakeStages;
        }
        if (this.cfg.updateMode !== 'amortized' || this.invalidatedSlices === 0) return EMPTY_STAGES;

        // Amortized: the source cube + one prefilter mip + the SH projection.
        const mip = this.prefilterStages[this.amortizeCursor % this.prefilterStages.length];
        this.amortizeCursor++;
        this.invalidatedSlices--;
        this.amortizedSlice[0] = this.bakeStages[0];      // stage A — the source, in case it moved
        this.amortizedSlice[1] = mip;
        this.amortizedSlice[2] = this.shStage!;
        return this.amortizedSlice;
    }

    private readonly amortizedSlice: ComputeStageNode[] = new Array(3);

    /**
     * Declare that source parameters changed. Amortized mode refreshes every prefilter mip over a
     * bounded cycle; once mode re-arms the full bake. Repeated calls coalesce rather than extending
     * the cycle indefinitely.
     */
    public invalidateBake(): void {
        if (!this.baked) return;
        if (this.cfg.updateMode === 'amortized') {
            if (this.invalidatedSlices === 0) this.invalidatedSlices = this.cfg.prefilterMips;
        } else {
            this.baked = false;
        }
    }

    /** True while a declared source change still has bake work outstanding. */
    public get bakeInvalidated(): boolean {
        return !this.baked || this.invalidatedSlices > 0;
    }

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
     *
     * `extra` is for later contributors at higher bindings — sky sources at 14–17,
     * `AerialPerspective.getBindGroupEntries()` at 18–19. One optional parameter rather than
     * another N-argument escalation of who-owns-the-group; the rule that exactly one call builds
     * it is unchanged.
     */
    public applyToBatch(batch: RenderBatch, lights: LightSystem,
        extra: GPUBindGroupEntry[] = EMPTY_ENTRIES): void {
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
            ...extra,
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
        if (!this.skySource || !this.skyLayout) {
            throw new Error(
                `EnvironmentSystem: getSkyPass() called before a sky source was set; ` +
                `call setSky() (or setProceduralSky/setEquirect) first.`);
        }

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

        // The stride was fixed by setSky and is about to be frozen into the batch. A source that
        // changed its paramFloats since then would have writeParams run off the end of a staging
        // array that is silently bounds-clipped — the worst kind of failure, so name it (§9).
        const declared = SKY_PARAM_FLOATS + this.skySource.paramFloats;
        if (declared !== this.skyStrideFloats) {
            throw new Error(
                `SkySource "${this.skySource.name}" declares paramFloats=` +
                `${this.skySource.paramFloats}, but the sky batch was created with strideFloats=` +
                `${this.skyStrideFloats}. The stride is fixed at batch creation — paramFloats must ` +
                `not change after setSky().`);
        }

        this.skyPass = this.engine.createPass({
            name: 'Env: Sky',
            isMainScreenPass: false,
            colorAttachments: skyColorAttachments,
            depthStencilAttachment: skyDepthAttachment,
            size: target.size,
        });
        this.skyPassTargetFormat = target.targetFormat ?? 'rgba16float';
        this.skyBatch = this.engine.createBatch(this.skyPass, {
            shaderCode: buildSkyWGSL({ environmentGroup: this.cfg.environmentGroup, sky: this.skyLayout }),
            strideFloats: this.skyStrideFloats,
            maxInstances: 1,
            targetFormat: target.targetFormat ?? 'rgba16float',
            depthWriteEnabled: false,
            depthCompare: 'less-equal',   // only where nothing was drawn (cleared depth = 1.0)
        });
        // Group 2 for the sky needs the sampler + prefiltered cube (the cheapSky fallback), plus
        // any source-owned resources in the RESERVED 14–17 range (EquirectSky's source cube,
        // §4.3); it never reads the lights, so none of bindings 0–2 are touched.
        const group2: GPUBindGroupEntry[] = [
            { binding: 3, resource: this.envSampler },
            { binding: 4, resource: this.specularCubeView },
        ];
        const skyboxExtras = this.skySource.getSkyboxBindEntries?.(this);
        if (skyboxExtras) group2.push(...skyboxExtras);
        this.engine.setRenderPassBindGroup(this.skyBatch, group2, this.cfg.environmentGroup);
        this.setSkyParams({});
        return this.skyPass;
    }

    /** Existing sky pass for additive decorators such as StarField; enforces build order loudly. */
    public getCreatedSkyPass(): RenderPassNode {
        if (!this.skyPass) {
            throw new Error('EnvironmentSystem: no sky pass exists yet; call getSkyPass(target) before attaching a StarField, or pass target to StarField.attach().');
        }
        return this.skyPass;
    }

    /** Colour format frozen into every pipeline attached to the cached sky pass. */
    public get skyTargetFormat(): GPUTextureFormat { return this.skyPassTargetFormat; }

    /**
     * Exposure / yaw / blur mip / cheapSky for the sky, written through the batch's storage
     * buffer (§8.2) AND the Stage-A params buffer, so both consumers of the evaluator see the
     * same header. Also re-serializes the source's own params (so a `setSunDirection` before
     * this call lands without a rebuild).
     */
    public setSkyParams(p: SkyParams): void {
        // `yaw` is the ENVIRONMENT's rotation, not a sky-local knob — it turns the diffuse SH and
        // the specular lookup as well (api/environment.md §1.5, "the three cannot drift apart").
        // Routing it through the one owner is what keeps them locked AND what makes it persist:
        // exposure, blurMip and cheapSky already survive a later call that omits them, and a yaw
        // that silently reverted to `this.yaw` on the next call was a rotation nobody could keep.
        if (p.yaw !== undefined && p.yaw !== this.yaw) {
            this.yaw = p.yaw;
            this.writeEnvParams();
        }
        const s = this.skyParamsF32;
        s[0] = p.exposure ?? (s[0] || 1.0);
        s[1] = this.yaw;
        s[2] = p.blurMip ?? (s[2] || 0);
        s[3] = p.cheapSky ?? s[3];
        if (this.skySource) this.skySource.writeParams(s, SKY_PARAM_FLOATS);
        if (this.skyBatch) this.engine.updateBatchData(this.skyBatch, s, 1);
        if (this.bakeParamsBuffer) {
            this.device.queue.writeBuffer(this.bakeParamsBuffer, 0, s as Float32Array<ArrayBuffer>);
        }
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
    /** The SOURCE cube view at full resolution — EquirectSky's skybox samples it at mip 0
     *  (environment-sky-design.md §5.2); valid only while `keepSourceCube` is true. */
    public get sourceView(): GPUTextureView { return this.sourceCubeView; }
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
        this.bakeParamsBuffer?.destroy();
        this.equirectTexture?.destroy();
        this.bakeStages = [];
        this.prefilterStages = [];
        this.shStage = null;
    }
}

/** Shared empty array — returning a fresh `[]` per frame would allocate (constraint #2). */
const EMPTY_STAGES: ComputeStageNode[] = [];

/** Default for `applyToBatch`'s `extra` — a shared empty, for the same reason. */
const EMPTY_ENTRIES: GPUBindGroupEntry[] = [];
