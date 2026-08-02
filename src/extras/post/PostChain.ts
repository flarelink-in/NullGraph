// src/extras/post/PostChain.ts
import { NullGraph, RenderPassNode, RenderBatch, RenderPassConfig } from '../../core';
import type { Effect, DeclareContext, PassRequest } from './Effect';
import type { ParamSchema, TargetHandle, PostFormat } from './types';
import { TransientPool, PoolReport } from './TransientPool';
import { ParamArena, layoutFields } from './ParamArena';
import { composePassShader, ComposedShader, PASS_HEADER_FLOATS } from './ShaderComposer';

export interface PostChainConfig {
    /** HDR working format for all intermediates. Default 'rgba16float'. */
    format?: PostFormat;
    /** Internal resolution multiplier vs. canvas. Default 1.0. */
    scale?: number;
    /** What the scene pass emits beyond color. Declared here, not inferred from
     *  the chain, so sceneAttachments() has no ordering dependency on setChain(). */
    gbuffer?: { depth?: GPUTextureFormat };
    /** true (default): the chain owns the final linear->display conversion and
     *  the last pass targets the swapchain. false: the result stays LINEAR in
     *  outputView. */
    outputToScreen?: boolean;
}

const SCREEN = 'screen';

interface LogicalPass {
    /** Global pass index; the user's scene pass is 0, post passes start at 1. */
    index: number;
    name: string;
    effectIndex: number;
    inputs: TargetHandle[];
    output: TargetHandle | typeof SCREEN;
    body: string;
}

interface CompiledPass {
    node: RenderPassNode;
    batch: RenderBatch;
    composed: ComposedShader;
    /** null for the screen pass (RenderPipeline substitutes the swapchain). */
    attachment: GPURenderPassColorAttachment | null;
    headerF32: Float32Array;
    headerU32: Uint32Array;
}

/**
 * Orchestrates TransientPool, ParamArena, ShaderComposer, and the effects into
 * the engine's flat pass array. Everything expensive happens in compile();
 * the per-frame path is param stores, one writeBuffer, and a cached array.
 */
export class PostChain {
    public readonly format: PostFormat;

    private readonly engine: NullGraph;
    private readonly device: GPUDevice;
    private readonly scale: number;
    private readonly gbufferDepth?: GPUTextureFormat;
    private readonly outputToScreen: boolean;

    private baseWidth: number;
    private baseHeight: number;

    private pool: TransientPool | null = null;
    private arena: ParamArena | null = null;

    private effects: Effect[] = [];
    private enabledEffects: Effect[] = [];
    private schemas: ParamSchema[] = [];

    private logicalPasses: LogicalPass[] = [];
    private compiledPasses: CompiledPass[] = [];
    /** Single stable array instance — repopulated in place so a captured
     *  reference stays valid across debugView() toggles. */
    private readonly sequence: RenderPassNode[] = [];
    private compiled = false;

    private sceneColorH: TargetHandle | null = null;
    private sceneDepthH: TargetHandle | null = null;
    private finalTargetH: TargetHandle | null = null;

    /** Handed out by sceneAttachments() and held by reference inside the user's
     *  RenderPassNode; compile()/resize() patch .view in place. */
    private readonly sceneColorAttachment: GPURenderPassColorAttachment;
    private sceneDepthAttachment?: GPURenderPassDepthStencilAttachment;

    private linearSampler: GPUSampler | null = null;
    private nearestSampler: GPUSampler | null = null;

    private debugPassIndex = -1;
    private debugNode: RenderPassNode | null = null;
    private debugBatch: RenderBatch | null = null;
    private debugComposed: ComposedShader | null = null;

    constructor(engine: NullGraph, config: PostChainConfig = {}) {
        this.engine = engine;
        this.device = engine.device;
        this.format = config.format ?? 'rgba16float';
        this.scale = config.scale ?? 1.0;
        this.gbufferDepth = config.gbuffer?.depth;
        if (this.gbufferDepth && this.gbufferDepth !== 'depth24plus') {
            // BatchManager hardcodes 'depth24plus' for any depth-tested pipeline
            // (core/BatchManager.ts:48); a different scene depth format would
            // mismatch the user's scene batches.
            console.warn(`PostChain: gbuffer.depth "${this.gbufferDepth}" will mismatch the engine's hardcoded depth24plus pipeline state; use 'depth24plus'.`);
        }
        this.outputToScreen = config.outputToScreen ?? true;
        this.baseWidth = engine.width;
        this.baseHeight = engine.height;

        this.sceneColorAttachment = {
            view: undefined as unknown as GPUTextureView,
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
        };
        if (this.gbufferDepth) {
            this.sceneDepthAttachment = {
                view: undefined as unknown as GPUTextureView,
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            };
        }
    }

    private get internalWidth(): number {
        return Math.max(1, Math.floor(this.baseWidth * this.scale));
    }

    private get internalHeight(): number {
        return Math.max(1, Math.floor(this.baseHeight * this.scale));
    }

    // =========================================================
    // Public surface
    // =========================================================

    /** Spread into engine.createPass(). Safe to call before setChain() — the
     *  g-buffer is declared in the constructor, so the attachment objects exist
     *  now and their views are patched at compile. */
    public sceneAttachments(opts?: { clearValue?: GPUColor; depthClearValue?: number }):
        Pick<RenderPassConfig, 'colorAttachments' | 'depthStencilAttachment'> {
        if (opts?.clearValue !== undefined) this.sceneColorAttachment.clearValue = opts.clearValue;
        if (opts?.depthClearValue !== undefined && this.sceneDepthAttachment) {
            this.sceneDepthAttachment.depthClearValue = opts.depthClearValue;
        }
        return {
            colorAttachments: [this.sceneColorAttachment],
            ...(this.sceneDepthAttachment ? { depthStencilAttachment: this.sceneDepthAttachment } : {}),
        };
    }

    /** COMPILE. Not per-frame. Invalidates any previously returned sequence contents. */
    public setChain(effects: Effect[]): void {
        const seen = new Set<Effect>();
        for (const e of effects) {
            if (seen.has(e)) {
                throw new Error(`PostChain: effect instance "${e.name}" appears twice; each instance may occupy one slot.`);
            }
            seen.add(e);
        }
        this.effects = effects.slice();
        this.compile();
    }

    public compile(): void {
        if (this.effects.length === 0) {
            throw new Error('PostChain: chain contains no enabled effects.');
        }
        this.enabledEffects = this.effects.filter(e => e.enabled);
        if (this.enabledEffects.length === 0) {
            throw new Error('PostChain: chain contains no enabled effects.');
        }

        this.declareAndResolve();

        // Arena: one padded 256-byte slice per effect; effects flush their
        // pending constructor values into their slice on first bind.
        if (this.arena) this.arena.destroy();
        this.arena = new ParamArena(
            this.device,
            this.enabledEffects.map((e, i) => ({ name: e.name, schema: this.schemas[i] })),
        );
        this.enabledEffects.forEach((e, i) => e.bind(this.arena!.slotFor(i)));

        this.instantiate();
        this.compiled = true;
        this.debugPassIndex = -1;
        this.refreshSequence();
    }

    /** Flat and fully expanded — Bloom appears as its individual passes. */
    public getPassSequence(): RenderPassNode[] {
        if (!this.compiled) throw new Error('PostChain: getPassSequence() called before setChain().');
        return this.sequence;
    }

    /** Once per frame. One writeBuffer for the whole arena. */
    public upload(): void {
        if (!this.compiled) throw new Error('PostChain: upload() called before setChain().');
        this.arena!.upload(this.device.queue);
        this.pool!.advanceFrame();
    }

    /** Adopt a worker-computed arena by reference. Length must equal paramFloatCount. */
    public setRawParams(data: Float32Array): void {
        if (!this.compiled) throw new Error('PostChain: setRawParams() called before setChain().');
        this.arena!.adopt(data);
        // Re-point every effect's slot into the adopted array. Pending values
        // were already flushed on the first bind, so this cannot stomp
        // worker-written data.
        this.enabledEffects.forEach((e, i) => e.bind(this.arena!.slotFor(i)));
    }

    public get paramFloatCount(): number {
        if (!this.compiled) throw new Error('PostChain: paramFloatCount read before setChain().');
        return this.arena!.floatCount;
    }

    public describeLayout(): Record<string, { offsetFloats: number; floats: number; fields: string[] }> {
        if (!this.compiled) throw new Error('PostChain: describeLayout() called before setChain().');
        return this.arena!.describeLayout();
    }

    /** Only when outputToScreen === false: the final LINEAR result. */
    public get outputView(): GPUTextureView {
        if (this.outputToScreen) {
            throw new Error('PostChain: outputView is only available with outputToScreen: false.');
        }
        if (!this.compiled) throw new Error('PostChain: outputView read before setChain().');
        return this.pool!.view(this.finalTargetH!);
    }

    public describe(): string {
        if (!this.compiled) return 'PostChain: <not compiled>';
        const lines = this.logicalPasses.map(lp => {
            const ins = lp.inputs.map(h => h.debugName).join(', ') || '-';
            const out = lp.output === SCREEN ? 'screen' : lp.output.debugName;
            return `#${lp.index} ${lp.name}: [${ins}] -> ${out}`;
        });
        return lines.join('\n');
    }

    public memoryReport(): PoolReport {
        if (!this.compiled) throw new Error('PostChain: memoryReport() called before setChain().');
        return this.pool!.report();
    }

    /** Truncate the sequence after post pass `passIndex` (0-based into
     *  getPassSequence()) and blit that pass's output to the screen. -1 disables. */
    public debugView(passIndex: number): void {
        if (!this.compiled) throw new Error('PostChain: debugView() called before setChain().');
        if (passIndex >= this.compiledPasses.length) {
            throw new Error(`PostChain: debugView(${passIndex}) out of range (${this.compiledPasses.length} passes).`);
        }
        this.debugPassIndex = passIndex < 0 ? -1 : passIndex;
        this.refreshSequence();
    }

    /** Rebuild pooled targets for a new canvas size. The user's scene pass and
     *  batches stay valid: attachment .view fields are patched in place. */
    public resize(width: number, height: number): void {
        this.baseWidth = width;
        this.baseHeight = height;
        if (!this.compiled) return;

        this.declareAndResolve();

        if (this.logicalPasses.length !== this.compiledPasses.length) {
            // Effect declarations are size-dependent (e.g. Bloom clamps its mip
            // count) and the pass count changed — full re-instantiation.
            this.instantiate();
        } else {
            for (let i = 0; i < this.logicalPasses.length; i++) {
                const lp = this.logicalPasses[i];
                const cp = this.compiledPasses[i];
                if (cp.attachment && lp.output !== SCREEN) {
                    cp.attachment.view = this.pool!.view(lp.output);
                }
                this.bindPassResources(lp, cp);
            }
        }
        this.refreshSequence();
    }

    public destroy(): void {
        this.pool?.destroy();
        this.arena?.destroy();
        this.pool = null;
        this.arena = null;
        this.compiled = false;
        this.sequence.length = 0;
    }

    // =========================================================
    // Compile pipeline
    // =========================================================

    /** Steps 1-4 + 8 of the compile pipeline: declare targets, resolve chain
     *  refs, mark lifetimes, pool.resolve(). Pure pool work — re-runnable on
     *  resize because Effect.declare() is required to be deterministic. */
    private declareAndResolve(): void {
        const iw = this.internalWidth;
        const ih = this.internalHeight;

        if (!this.pool) {
            this.pool = new TransientPool(this.device, iw, ih, this.format);
        } else {
            this.pool.resize(iw, ih);
        }
        const pool = this.pool;

        this.sceneColorH = pool.alloc({ scale: 1, format: this.format }, 'sceneColor');
        pool.markWrite(this.sceneColorH, 0, { passName: 'Scene', loadOp: 'clear' });
        if (this.gbufferDepth) {
            this.sceneDepthH = pool.alloc({ scale: 1, format: this.gbufferDepth as PostFormat }, 'sceneDepth');
            pool.markWrite(this.sceneDepthH, 0, { passName: 'Scene', loadOp: 'clear' });
        } else {
            this.sceneDepthH = null;
        }

        // Declare: effects record targets and pass requests in order.
        const raw: { effectIndex: number; req: PassRequest }[] = [];
        this.schemas = [];
        this.enabledEffects.forEach((effect, ei) => {
            const ctx: DeclareContext = {
                target: (spec, debugName) => pool.alloc(spec, `${effect.name}/${debugName}`),
                pass: (req) => raw.push({ effectIndex: ei, req }),
                has: (resource) => resource === 'depth' ? this.sceneDepthH !== null : false,
                width: iw,
                height: ih,
            };
            const decl = effect.declare(ctx);
            this.schemas.push(decl.params);
        });
        if (raw.length === 0) {
            throw new Error('PostChain: enabled effects declared no passes.');
        }

        // Resolve chain refs and mark lifetimes.
        let cursor = this.sceneColorH;
        let chainCount = 0;
        this.logicalPasses = [];

        raw.forEach((r, i) => {
            const index = i + 1; // scene pass is 0
            const isLast = i === raw.length - 1;
            const effect = this.enabledEffects[r.effectIndex];

            const inputs: TargetHandle[] = r.req.inputs.map(ref => {
                if (ref === 'chain') return cursor;
                if (ref === 'scene') return this.sceneColorH!;
                if (ref === 'depth') {
                    if (!this.sceneDepthH) {
                        throw new Error(`PostChain: effect "${effect.name}" requires gbuffer.depth, which was not declared in the PostChain constructor.`);
                    }
                    return this.sceneDepthH;
                }
                if (ref === 'velocity') {
                    throw new Error(`PostChain: effect "${effect.name}" requires gbuffer.velocity, which is not supported in v1.`);
                }
                return ref;
            });

            let output: TargetHandle | typeof SCREEN;
            if (r.req.output === 'chain') {
                if (isLast && this.outputToScreen) {
                    output = SCREEN;
                } else {
                    output = pool.alloc({ scale: 1, format: this.format }, `chain_${++chainCount}`);
                }
            } else {
                output = r.req.output;
            }
            if (isLast && this.outputToScreen && output !== SCREEN) {
                throw new Error(`PostChain: with outputToScreen, the final pass ("${r.req.name}") must write to 'chain'.`);
            }

            for (const h of inputs) pool.markRead(h, index, { passName: r.req.name });
            if (output !== SCREEN) pool.markWrite(output, index, { passName: r.req.name, loadOp: 'clear' });
            if (r.req.output === 'chain' && output !== SCREEN) cursor = output;

            this.logicalPasses.push({
                index,
                name: r.req.name,
                effectIndex: r.effectIndex,
                inputs,
                output,
                body: r.req.shaderCode,
            });
        });

        this.finalTargetH = cursor;
        pool.resolve();

        // The pool's textures moved: re-point the scene pass attachments.
        this.sceneColorAttachment.view = pool.view(this.sceneColorH);
        if (this.sceneDepthH && this.sceneDepthAttachment) {
            this.sceneDepthAttachment.view = pool.view(this.sceneDepthH);
        }
    }

    /** Steps 6-7: compose WGSL, create engine passes/batches, bind, write headers. */
    private instantiate(): void {
        const pool = this.pool!;
        const arena = this.arena!;

        // Two cached samplers, both clamp-to-edge: 'repeat' on a bloom mip
        // wraps the glow around the screen edge, and depth24plus must be
        // sampled with a non-filtering sampler.
        if (!this.linearSampler) {
            this.linearSampler = this.device.createSampler({
                label: 'PostChain_linear',
                magFilter: 'linear', minFilter: 'linear',
                addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
            });
            this.nearestSampler = this.device.createSampler({
                label: 'PostChain_nearest',
                magFilter: 'nearest', minFilter: 'nearest',
                addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
            });
        }

        const swapchainIsSRGB = this.engine.ctx.format.endsWith('-srgb');
        this.compiledPasses = [];

        for (const lp of this.logicalPasses) {
            const isScreen = lp.output === SCREEN;
            const entry = arena.entries[lp.effectIndex];

            const composed = composePassShader({
                body: lp.body,
                paramsWgsl: entry.layout.wgsl,
                paramsTouchField: entry.layout.touchField,
                inputs: lp.inputs.map(h => ({ depth: pool.info(h).format.startsWith('depth') })),
                encodeToSRGB: isScreen && !swapchainIsSRGB,
                label: lp.name,
            });

            let attachment: GPURenderPassColorAttachment | null = null;
            let node: RenderPassNode;
            if (isScreen) {
                node = this.engine.createPass({ name: `Post/${lp.name}`, isMainScreenPass: true });
            } else {
                attachment = {
                    view: pool.view(lp.output as TargetHandle),
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                };
                node = this.engine.createPass({
                    name: `Post/${lp.name}`,
                    isMainScreenPass: false,
                    colorAttachments: [attachment],
                });
            }

            // The screen pass gets ctx.depthTexture force-attached by the
            // pipeline; depthWriteEnabled:false / 'always' keeps the fullscreen
            // triangle from touching it (relying on z=0 vs cleared 1.0 is fragile).
            const batch = this.engine.createBatch(node, {
                shaderCode: composed.code,
                strideFloats: PASS_HEADER_FLOATS,
                maxInstances: 1,
                targetFormat: isScreen ? this.engine.ctx.format : pool.info(lp.output as TargetHandle).format,
                depthWriteEnabled: false,
                depthCompare: 'always',
            });

            const headerBytes = new ArrayBuffer(PASS_HEADER_FLOATS * 4);
            const cp: CompiledPass = {
                node,
                batch,
                composed,
                attachment,
                headerF32: new Float32Array(headerBytes),
                headerU32: new Uint32Array(headerBytes),
            };
            this.bindPassResources(lp, cp);
            this.compiledPasses.push(cp);
        }

        this.createDebugBlit();
    }

    /** Group-1 bind group plus the static per-pass header. Re-run on resize:
     *  pool views changed, and updateBatchData also (re)sets
     *  currentInstanceCount = 1, without which the pipeline skips the batch. */
    private bindPassResources(lp: LogicalPass, cp: CompiledPass): void {
        const pool = this.pool!;
        const arena = this.arena!;
        const entry = arena.entries[lp.effectIndex];

        const entries: GPUBindGroupEntry[] = cp.composed.bindings.map(b => {
            switch (b.kind) {
                case 'params':
                    return {
                        binding: b.binding,
                        resource: { buffer: arena.buffer, offset: entry.sliceByteOffset, size: entry.sliceBytes },
                    };
                case 'linear-sampler':
                    return { binding: b.binding, resource: this.linearSampler! };
                case 'nearest-sampler':
                    return { binding: b.binding, resource: this.nearestSampler! };
                case 'texture':
                    return { binding: b.binding, resource: pool.view(lp.inputs[b.inputIndex!]) };
            }
        });
        this.engine.setRenderPassBindGroup(cp.batch, entries, 1);

        let w: number, h: number;
        if (lp.output === SCREEN) {
            w = this.baseWidth;
            h = this.baseHeight;
        } else {
            const info = pool.info(lp.output);
            w = info.width;
            h = info.height;
        }
        cp.headerF32[0] = w;
        cp.headerF32[1] = h;
        cp.headerF32[2] = 1 / w;
        cp.headerF32[3] = 1 / h;
        cp.headerU32[4] = lp.index;
        this.engine.updateBatchData(cp.batch, cp.headerF32, 1);
    }

    private createDebugBlit(): void {
        const minimal = layoutFields([], 'Params');
        this.debugComposed = composePassShader({
            body: `fn effect(uv: vec2f) -> vec4f { return sampleInput0(uv); }`,
            paramsWgsl: minimal.wgsl,
            paramsTouchField: minimal.touchField,
            inputs: [{ depth: false }],
            encodeToSRGB: !this.engine.ctx.format.endsWith('-srgb'),
            label: 'DebugView blit',
        });
        this.debugNode = this.engine.createPass({ name: 'Post/DebugView', isMainScreenPass: true });
        this.debugBatch = this.engine.createBatch(this.debugNode, {
            shaderCode: this.debugComposed.code,
            strideFloats: PASS_HEADER_FLOATS,
            maxInstances: 1,
            targetFormat: this.engine.ctx.format,
            depthWriteEnabled: false,
            depthCompare: 'always',
        });
    }

    private refreshSequence(): void {
        this.sequence.length = 0;
        const di = this.debugPassIndex;

        if (di < 0 || di >= this.compiledPasses.length) {
            for (const cp of this.compiledPasses) this.sequence.push(cp.node);
            return;
        }

        const lp = this.logicalPasses[di];
        for (let i = 0; i <= di; i++) this.sequence.push(this.compiledPasses[i].node);
        if (lp.output === SCREEN) return; // already on screen, nothing to blit

        // Point the blit at the debugged pass's output. Aliasing cannot corrupt
        // it: the sequence is truncated, so no later tenant runs this frame.
        const entries: GPUBindGroupEntry[] = this.debugComposed!.bindings.map(b => {
            switch (b.kind) {
                case 'params':
                    return {
                        binding: b.binding,
                        resource: { buffer: this.arena!.buffer, offset: 0, size: this.arena!.entries[0].sliceBytes },
                    };
                case 'linear-sampler':
                    return { binding: b.binding, resource: this.linearSampler! };
                case 'nearest-sampler':
                    return { binding: b.binding, resource: this.nearestSampler! };
                case 'texture':
                    return { binding: b.binding, resource: this.pool!.view(lp.output as TargetHandle) };
            }
        });
        this.engine.setRenderPassBindGroup(this.debugBatch!, entries, 1);

        const dbgBytes = new ArrayBuffer(PASS_HEADER_FLOATS * 4);
        const f = new Float32Array(dbgBytes);
        const u = new Uint32Array(dbgBytes);
        f[0] = this.baseWidth;
        f[1] = this.baseHeight;
        f[2] = 1 / this.baseWidth;
        f[3] = 1 / this.baseHeight;
        u[4] = this.logicalPasses.length + 1;
        this.engine.updateBatchData(this.debugBatch!, f, 1);

        this.sequence.push(this.debugNode!);
    }
}
