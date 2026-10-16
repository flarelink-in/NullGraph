// src/extras/picking/Picker.ts
//
// GPU picking (docs/picking-design.md). "What is under this pixel?" answered by rendering one more
// view — a camera zoomed onto the clicked pixels — through the culling and draw paths the scene
// already uses, then reading back a few texels. Off the frame path: a pick is its own submission,
// and a picker that is not picking costs nothing (its pass is never in the app's sequence, its
// cull stages ride the visibility chain disabled).
//
// Core is imported as TYPES only. Passes and batches are created through the engine, never with
// `new` — a second copy of a core class in this bundle would fail RenderPipeline's instanceof checks.

import type { NullGraph, RenderBatch, RenderPassNode, ComputeStageNode } from '../../core';
import type { VisibilitySystem } from '../visibility/VisibilitySystem';
import { StandardLayout } from '../geometry/VertexLayout';
import { buildInstancePickWGSL, buildStandardPickWGSL } from './pickShaders';
import {
    buildPickCamera, selectHit, unprojectPixel, CAMERA_FLOATS, CAM_INV_VIEW_PROJ, MAX_PICK_RADIUS,
} from './pickMath';

export interface PickerOptions {
    /**
     * Largest `radius` a pick may ask for, in pixels. The pick target is (2·maxRadius + 1)² texels
     * and is always rendered whole; `radius` only limits the search. Default 4 (9×9), max 7.
     */
    maxRadius?: number;
}

/** A registered pickable — returned by `addInstances` / `addBatch`, and carried by every hit. */
export interface PickSource {
    /** 1-based tag written into the pick target. */
    readonly tag: number;
    readonly label: string;
    readonly kind: 'instances' | 'batch';
}

export interface PickHit {
    source: PickSource;
    /** The app's own index: its InstanceBuffer index (instance sources) or the batch instance index. */
    index: number;
    /** meshIndex of the hit instance (instance sources); 0 for batch sources. */
    meshIndex: number;
    /** Clip-space depth of the hit, in [0, 1]. */
    depth: number;
    /** World-space position of the hit, reconstructed from `depth` through the camera passed to pick(). */
    position: [number, number, number];
    /** The pixel that was hit — differs from the requested one when `radius` > 0. */
    pixel: [number, number];
}

export interface PickOptions {
    /** Accept a hit up to this many pixels from (x, y) — the nearest wins. Default 0. */
    radius?: number;
}

/** What `pick()` needs from a camera: the engine's 108-float block (`Camera.bufferData`). */
export interface PickCamera { bufferData: ArrayLike<number> }

interface InstanceSource { kind: 'instances'; source: PickSource; vis: VisibilitySystem; batch: RenderBatch; stages: ComputeStageNode[] }
interface BatchSource { kind: 'batch'; source: PickSource; target: RenderBatch; batch: RenderBatch }

const ZERO_U32 = new Uint32Array([0]);

export class Picker {
    /** See {@link PickerOptions.maxRadius}. */
    public readonly maxRadius: number;
    /** Edge of the pick target in texels: 2·maxRadius + 1. */
    public readonly size: number;
    /** The pick camera's uniform — group 0 binding 0 of every pick batch. */
    public readonly cameraBuffer: GPUBuffer;

    private readonly device: GPUDevice;
    private readonly pass: RenderPassNode;
    private readonly idTexture: GPUTexture;
    private readonly depthTexture: GPUTexture;
    private readonly staging: GPUBuffer;
    private readonly rowStrideBytes: number;
    private readonly cameraData = new Float32Array(CAMERA_FLOATS);
    private readonly sources: Array<InstanceSource | BatchSource> = [];
    private queue: Promise<unknown> = Promise.resolve();
    private destroyed = false;

    constructor(private readonly engine: NullGraph, options: PickerOptions = {}) {
        const r = options.maxRadius ?? 4;
        if (!Number.isInteger(r) || r < 0 || r > MAX_PICK_RADIUS) {
            throw new Error(`Picker: maxRadius ${r} must be an integer in [0, ${MAX_PICK_RADIUS}].`);
        }
        this.maxRadius = r;
        this.size = 2 * r + 1;
        this.device = engine.device;

        this.cameraBuffer = this.device.createBuffer({
            label: 'NullGraph_PickCamera', size: CAMERA_FLOATS * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.idTexture = this.device.createTexture({
            label: 'NullGraph_PickTarget', size: [this.size, this.size], format: 'rgba32uint',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        });
        this.depthTexture = this.device.createTexture({
            label: 'NullGraph_PickDepth', size: [this.size, this.size], format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        // copyTextureToBuffer rows are 256-byte aligned; 16 bytes per rgba32uint texel.
        this.rowStrideBytes = Math.ceil((this.size * 16) / 256) * 256;
        this.staging = this.device.createBuffer({
            label: 'NullGraph_PickStaging', size: this.rowStrideBytes * this.size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        this.pass = engine.createPass({
            name: 'NullGraph Pick', isMainScreenPass: false, size: [this.size, this.size],
            colorAttachments: [{
                view: this.idTexture.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 },
                loadOp: 'clear', storeOp: 'store',
            }],
            depthStencilAttachment: {
                view: this.depthTexture.createView(), depthClearValue: 1,
                depthLoadOp: 'clear', depthStoreOp: 'discard',
            },
        });
    }

    /**
     * Make every instance of `meshIndex` in a VisibilitySystem pickable. A hit's `index` is the
     * instance's index in the app's InstanceBuffer. The pick batch is culled by the system's own
     * cull kernel against the pick camera — its stage rides the system's chain disabled, so the
     * app's frame does not pay for it.
     */
    public addInstances(vis: VisibilitySystem, meshName: string, meshIndex: number, opts: { label?: string } = {}): PickSource {
        this.assertLive();
        if ((vis as unknown as { engine: NullGraph }).engine !== this.engine) {
            throw new Error('Picker: that VisibilitySystem belongs to a different engine.');
        }
        const source: PickSource = { tag: this.sources.length + 1, label: opts.label ?? meshName, kind: 'instances' };
        const batch = vis.createInstanceDrawBatch(this.pass, meshName, meshIndex, {
            camera: this.cameraBuffer,
            occlusion: false,                        // the Hi-Z pyramid belongs to the main view
            shaderCode: buildInstancePickWGSL(source.tag),
            targetFormat: 'rgba32uint',
        });
        // Its cull and scatter (visibility-design §14), run in order at each pick.
        const stages = vis.getCullStages(batch);
        if (!stages.length) throw new Error('Picker: the VisibilitySystem did not register cull stages for the pick batch.');
        for (const stage of stages) stage.enabled = false;
        // Zero instances until a pick, so an app that calls engine.render() WITHOUT a sequence (which
        // renders every pass the engine tracks, this one included) draws nothing here.
        this.device.queue.writeBuffer(batch.indirectBuffer!, 4, ZERO_U32);
        this.sources.push({ kind: 'instances', source, vis, batch, stages });
        return source;
    }

    /**
     * Make a plain batch pickable. The pick batch shares the source's geometry AND its instance
     * storage buffer (no second upload); its instance count is copied from the source at pick time.
     * The default shader reads the stock 14-float layout the stock materials use. A custom layout
     * passes `shaderCode: (tag) => wgsl` — a vertex stage returning `PickVaryings` plus
     * `buildPickFragmentWGSL(tag)` — and its `vertexLayouts` when they are not StandardLayout's.
     */
    public addBatch(
        target: RenderBatch,
        opts: { label?: string; shaderCode?: (tag: number) => string; vertexLayouts?: GPUVertexBufferLayout[] } = {},
    ): PickSource {
        this.assertLive();
        const label = opts.label ?? `batch ${this.sources.length + 1}`;
        if (target.isIndirect) {
            throw new Error(`Picker: batch "${label}" is indirect — its instance count lives on the GPU; ` +
                `pick it through its VisibilitySystem (addInstances).`);
        }
        const source: PickSource = { tag: this.sources.length + 1, label, kind: 'batch' };
        const batch = this.engine.createBatch(this.pass, {
            label: `pick: ${label}`,
            shaderCode: opts.shaderCode ? opts.shaderCode(source.tag) : buildStandardPickWGSL(source.tag),
            strideFloats: target.stride,
            maxInstances: 1,                         // its own buffer is never read — group 0 binds the source's
            vertexLayouts: opts.vertexLayouts ?? StandardLayout.getWebGPUDescriptor(),
            targetFormat: 'rgba32uint',
            cameraBuffer: this.cameraBuffer,
            depthWriteEnabled: true,
            depthCompare: 'less',
        });
        batch.bindGroup = this.device.createBindGroup({
            label: `pick: ${label} group 0`,
            layout: batch.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.cameraBuffer } },
                { binding: 1, resource: { buffer: target.storageBuffer } },
            ],
        });
        batch.currentInstanceCount = 0;
        this.sources.push({ kind: 'batch', source, target, batch });
        return source;
    }

    /**
     * What is under render-target pixel (x, y)? Resolves after one GPU round trip with the nearest
     * hit within `radius`, or `null`. `camera` must be the camera the frame was drawn with; its
     * state is captured now, so moving it afterwards does not change this pick. Picks are
     * serialized; each is its own submission, outside the app's frame.
     *
     * Coordinates are render-target pixels — for a canvas, CSS pixels × devicePixelRatio.
     */
    public pick(x: number, y: number, camera: PickCamera, opts: PickOptions = {}): Promise<PickHit | null> {
        this.assertLive();
        if (this.sources.length === 0) {
            throw new Error('Picker: nothing is pickable — add sources with addInstances() or addBatch().');
        }
        const width = this.engine.width, height = this.engine.height;
        const px = Math.floor(x), py = Math.floor(y);
        if (!(px >= 0 && px < width && py >= 0 && py < height)) {
            throw new Error(`Picker: pixel (${x}, ${y}) is outside the ${width}×${height} target — pass ` +
                `render-target pixels (CSS px × devicePixelRatio).`);
        }
        const radius = opts.radius ?? 0;
        if (!Number.isInteger(radius) || radius < 0 || radius > this.maxRadius) {
            throw new Error(`Picker: radius ${radius} exceeds maxRadius ${this.maxRadius} set at construction ` +
                `(or is not a non-negative integer).`);
        }
        const snapshot = Float32Array.from(camera.bufferData);
        if (snapshot.length !== CAMERA_FLOATS) {
            throw new Error(`Picker: camera.bufferData has ${snapshot.length} floats; the engine camera block has ${CAMERA_FLOATS}.`);
        }
        const run = this.queue.then(() => this.run(px, py, radius, width, height, snapshot));
        this.queue = run.catch(() => undefined);
        return run;
    }

    /** Unregister every source and release the picker's GPU resources. */
    public destroy(): void {
        if (this.destroyed) return;
        for (const s of this.sources) {
            if (s.kind === 'instances') s.vis.removeDrawBatch(s.batch);
            else this.engine.clearBatch(this.pass, s.batch);   // destroys only the pick batch's own buffers
        }
        this.sources.length = 0;
        this.idTexture.destroy();
        this.depthTexture.destroy();
        this.staging.destroy();
        this.cameraBuffer.destroy();
        this.destroyed = true;
    }

    // -----------------------------------------------------------------------------------------

    private async run(px: number, py: number, radius: number, width: number, height: number, cam: Float32Array): Promise<PickHit | null> {
        // A pick queued before destroy() resolves null rather than touching released resources.
        if (this.destroyed) return null;
        buildPickCamera(cam, px, py, this.maxRadius, width, height, this.cameraData);
        this.device.queue.writeBuffer(this.cameraBuffer, 0, this.cameraData);

        const stages: ComputeStageNode[] = [];
        for (const s of this.sources) {
            if (s.kind === 'instances') {
                for (const stage of s.stages) stage.enabled = true;
                stages.push(...s.stages);
            } else {
                // Mirror the source as it is NOW — the app may have swapped geometry or counts.
                s.batch.vertexBuffer = s.target.vertexBuffer;
                s.batch.indexBuffer = s.target.indexBuffer;
                s.batch.indexCount = s.target.indexCount;
                s.batch.indexFormat = s.target.indexFormat;
                s.batch.currentInstanceCount = s.target.currentInstanceCount;
            }
        }
        try {
            this.engine.render([...stages, this.pass]);
        } finally {
            for (const s of this.sources) {
                if (s.kind === 'instances') {
                    for (const stage of s.stages) stage.enabled = false;
                    this.device.queue.writeBuffer(s.batch.indirectBuffer!, 4, ZERO_U32);
                } else {
                    s.batch.currentInstanceCount = 0;
                }
            }
        }

        const enc = this.device.createCommandEncoder({ label: 'NullGraph pick readback' });
        enc.copyTextureToBuffer(
            { texture: this.idTexture },
            { buffer: this.staging, bytesPerRow: this.rowStrideBytes },
            [this.size, this.size],
        );
        this.device.queue.submit([enc.finish()]);
        try {
            await this.staging.mapAsync(GPUMapMode.READ);
        } catch (error) {
            if (this.destroyed) return null;   // destroy() released the staging buffer mid-pick
            throw error;
        }
        if (this.destroyed) return null;
        const texels = new Uint32Array(this.staging.getMappedRange().slice(0));
        this.staging.unmap();

        const hit = selectHit(texels, this.rowStrideBytes / 4, this.maxRadius, radius, px, py, width, height);
        if (!hit) return null;
        const record = this.sources[hit.tag - 1];
        if (!record) {
            throw new Error(`Picker: the target held source tag ${hit.tag}, which is not registered — a pick ` +
                `shader wrote a tag of its own (custom shaderCode must use the tag it is given).`);
        }
        const position = unprojectPixel(
            cam.subarray(CAM_INV_VIEW_PROJ, CAM_INV_VIEW_PROJ + 16), hit.px, hit.py, hit.depth, width, height, [0, 0, 0],
        ) as [number, number, number];
        return {
            source: record.source,
            index: hit.index,
            meshIndex: record.kind === 'instances' ? hit.extra : 0,
            depth: hit.depth,
            position,
            pixel: [hit.px, hit.py],
        };
    }

    private assertLive(): void {
        if (this.destroyed) throw new Error('Picker: used after destroy().');
    }
}
