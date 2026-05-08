// null-graph/animations/SkeletonManager.ts
import { Animator } from "./Animator";

export class SkeletonManager {
    public boneBuffer: GPUBuffer;
    public maxBones: number;

    constructor(device: GPUDevice, maxBones: number = 70) {
        this.maxBones = maxBones;

        // Allocate the video memory on the GPU
        this.boneBuffer = device.createBuffer({
            size: this.maxBones * 16 * 4, // 16 floats per mat4, 4 bytes per float
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Initialize with Identity Matrices (Bind Pose)
        this.resetToBindPose(device);
    }

    resetToBindPose(device: GPUDevice) {
        const identityBones = new Float32Array(this.maxBones * 16);
        for (let i = 0; i < this.maxBones; i++) {
            identityBones[i * 16] = 1;
            identityBones[i * 16 + 5] = 1;
            identityBones[i * 16 + 10] = 1;
            identityBones[i * 16 + 15] = 1;
        }
        device.queue.writeBuffer(this.boneBuffer, 0, identityBones as BufferSource);
    }

    updateFromAnimator(device: GPUDevice, animator: Animator) {
        const currentFrameMatrices = animator.getSkinningMatrices();
        device.queue.writeBuffer(
            this.boneBuffer,
            0,
            currentFrameMatrices as BufferSource
        );
    }
}