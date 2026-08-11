// src/extras/visibility/HiZPyramid.ts
//
// The Hi-Z depth pyramid (docs/visibility-design.md §5): an r32float mip chain where each texel
// holds the MAX (farthest) depth of the 2×2 region below it. Built once per frame from the
// app's samplable depth source (LAST frame's depth, single-phase §5.2), then read by the cull
// shaders' occlusion test.
//
// r32float is core WebGPU as both a write-only storage texture and an unfilterable
// `texture_2d<f32>` (textureLoad), so no optional feature is required. The build is a chain of
// per-mip compute stages; successive dispatches in one compute pass are ordered by WebGPU's sync
// model, so the whole pyramid builds in a single pass boundary.

import { NullGraph, ComputeStageNode } from '../../core';
import { buildHiZMipWGSL } from './visibilityShaders';
import { hiZMipCount, hiZMipSize } from './cullMath';

export class HiZPyramid {
    /** The pyramid texture (r32float, TEXTURE_BINDING | STORAGE_BINDING). */
    public texture!: GPUTexture;
    /** View over ALL mips, bound to the cull shaders as `texture_2d<f32>`. */
    public view!: GPUTextureView;
    /** Number of mip levels. */
    public mipLevels: number = 0;
    /** Highest mip level index (mipLevels - 1), passed to the cull shaders. */
    public maxLevel: number = 0;

    private device!: GPUDevice;
    private width: number = 0;
    private height: number = 0;
    private stages: ComputeStageNode[] = [];
    private built: boolean = false;

    constructor(private engine: NullGraph) {
        this.device = engine.device;
    }

    /**
     * (Re)allocate the pyramid for a depth source of `width`×`height` and build the per-mip
     * compute stages that fill it from `depthSource`. Called by the VisibilitySystem when Hi-Z
     * is enabled and again on resize.
     */
    public configure(depthSource: GPUTextureView, width: number, height: number): void {
        this.width = Math.max(1, width);
        this.height = Math.max(1, height);
        this.mipLevels = hiZMipCount(this.width, this.height);
        this.maxLevel = this.mipLevels - 1;

        if (this.texture) this.texture.destroy();
        this.texture = this.device.createTexture({
            label: 'NullGraph_HiZPyramid',
            size: [this.width, this.height],
            format: 'r32float',
            mipLevelCount: this.mipLevels,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
        });
        this.view = this.texture.createView();

        // One compute stage per mip. Mip 0 reads the depth source; mip L reads pyramid mip L-1.
        this.stages = [];
        for (let level = 0; level < this.mipLevels; level++) {
            const [w, h] = hiZMipSize(this.width, this.height, level);
            const fromDepth = level === 0;
            const stage = this.engine.createComputeStage({
                name: `Hi-Z mip ${level}`,
                shaderCode: buildHiZMipWGSL(fromDepth),
                dispatchCount: [Math.ceil(w / 8), Math.ceil(h / 8), 1],
            });

            const srcView = fromDepth
                ? depthSource
                : this.texture.createView({ baseMipLevel: level - 1, mipLevelCount: 1 });
            const dstView = this.texture.createView({ baseMipLevel: level, mipLevelCount: 1 });

            this.engine.setComputePassBindGroup(stage, [
                { binding: 0, resource: srcView },
                { binding: 1, resource: dstView },
            ], 0);
            this.stages.push(stage);
        }
        this.built = true;
    }

    /** The ordered compute stages that build the pyramid. Splice into the render sequence first. */
    public getBuildStages(): ComputeStageNode[] {
        if (!this.built) {
            throw new Error('HiZPyramid: getBuildStages() called before configure().');
        }
        return this.stages;
    }

    public destroy(): void {
        if (this.texture) {
            this.texture.destroy();
            this.texture = undefined as unknown as GPUTexture;
        }
        this.stages = [];
        this.built = false;
    }
}
