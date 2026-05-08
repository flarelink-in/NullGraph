import { GLBAnimation, GLBSkin } from "./types";
import { MathUtils } from "./MathUtils";

export class Animator {
    private skin: GLBSkin;
    private currentAnimation: GLBAnimation | null = null;
    private time: number = 0;

    // We now track ALL nodes, not just joints!
    private nodeCount: number;
    private jointCount: number;
    private nodeParents: Int16Array;

    private localTranslations: Float32Array;
    private localRotations: Float32Array;
    private localScales: Float32Array;

    private localMatrices: Float32Array;
    private globalMatrices: Float32Array;
    private globalComputed: Uint8Array; // Flag to prevent redundant math

    private skinningMatrices: Float32Array;

    constructor(skin: GLBSkin) {
        this.skin = skin;
        this.nodeCount = skin.nodes.length;
        this.jointCount = skin.joints.length;

        // Allocate memory for the ENTIRE scene tree
        this.localTranslations = new Float32Array(this.nodeCount * 3);
        this.localRotations = new Float32Array(this.nodeCount * 4);
        this.localScales = new Float32Array(this.nodeCount * 3);

        this.localMatrices = new Float32Array(this.nodeCount * 16);
        this.globalMatrices = new Float32Array(this.nodeCount * 16);
        this.globalComputed = new Uint8Array(this.nodeCount);

        // The final output remains just the joints for the shader
        this.skinningMatrices = new Float32Array(this.jointCount * 16);

        // Pre-calculate the parent of every node so we can traverse up the tree instantly
        this.nodeParents = new Int16Array(this.nodeCount).fill(-1);
        for (let i = 0; i < this.nodeCount; i++) {
            const node = this.skin.nodes[i];
            if (node.children) {
                for (const child of node.children) {
                    this.nodeParents[child] = i;
                }
            }
        }

        this.resetToBindPose();
    }

    private resetToBindPose() {
        for (let i = 0; i < this.nodeCount; i++) {
            const node = this.skin.nodes[i];
            const t = node.translation || [0, 0, 0];
            const r = node.rotation || [0, 0, 0, 1];
            const s = node.scale || [1, 1, 1];

            this.localTranslations.set(t, i * 3);
            this.localRotations.set(r, i * 4);
            this.localScales.set(s, i * 3);
        }
    }

    public play(animation: GLBAnimation) {
        this.currentAnimation = animation;
        this.time = 0;
        this.resetToBindPose(); // Crucial: Reset bones before starting a new clip!
    }

    public update(deltaTime: number) {
        if (!this.currentAnimation) return;

        this.time += deltaTime;
        if (this.time > this.currentAnimation.maxTime) {
            this.time = this.time % this.currentAnimation.maxTime;
        }

        this.evaluateAnimation();
        this.computeMatrices();
    }

    public getSkinningMatrices(): Float32Array {
        return this.skinningMatrices;
    }

    private evaluateAnimation() {
        const anim = this.currentAnimation!;

        for (const channel of anim.channels) {
            // Note: We animate the targetNode directly now, no need to check if it's a joint!
            const nodeIndex = channel.targetNode;

            const sampler = anim.samplers[channel.samplerIndex];
            const times = sampler.input;
            const values = sampler.output;

            let frameIdx = 0;
            while (frameIdx < times.length - 1 && times[frameIdx + 1] < this.time) {
                frameIdx++;
            }

            const t0 = times[frameIdx];
            const t1 = times[frameIdx + 1] || t0;
            const factor = t1 > t0 ? (this.time - t0) / (t1 - t0) : 0;

            if (channel.path === 'translation') {
                const a = [values[frameIdx*3], values[frameIdx*3+1], values[frameIdx*3+2]];
                const b = [values[(frameIdx+1)*3], values[(frameIdx+1)*3+1], values[(frameIdx+1)*3+2]];
                MathUtils.lerp(a, b, factor, this.localTranslations, nodeIndex * 3);
            }
            else if (channel.path === 'rotation') {
                const a = [values[frameIdx*4], values[frameIdx*4+1], values[frameIdx*4+2], values[frameIdx*4+3]];
                const b = [values[(frameIdx+1)*4], values[(frameIdx+1)*4+1], values[(frameIdx+1)*4+2], values[(frameIdx+1)*4+3]];
                MathUtils.slerp(a, b, factor, this.localRotations, nodeIndex * 4);
            }
            else if (channel.path === 'scale') {
                const a = [values[frameIdx*3], values[frameIdx*3+1], values[frameIdx*3+2]];
                const b = [values[(frameIdx+1)*3], values[(frameIdx+1)*3+1], values[(frameIdx+1)*3+2]];
                MathUtils.lerp(a, b, factor, this.localScales, nodeIndex * 3);
            }
        }
    }

    private computeMatrices() {
        // 1. Build Local Matrices for EVERY node
        for (let i = 0; i < this.nodeCount; i++) {
            const t = [this.localTranslations[i*3], this.localTranslations[i*3+1], this.localTranslations[i*3+2]];
            const r = [this.localRotations[i*4], this.localRotations[i*4+1], this.localRotations[i*4+2], this.localRotations[i*4+3]];
            const s = [this.localScales[i*3], this.localScales[i*3+1], this.localScales[i*3+2]];

            const localMat = MathUtils.fromTRS(t, r, s);
            this.localMatrices.set(localMat, i * 16);
        }

        // 2. Clear computation flags for this frame
        this.globalComputed.fill(0);

        // 3. Helper function to recursively compute/fetch global matrices securely
        const getGlobal = (nodeIndex: number): Float32Array => {
            const offset = nodeIndex * 16;

            // Return cached matrix if already calculated this frame
            if (this.globalComputed[nodeIndex] === 1) {
                return this.globalMatrices.subarray(offset, offset + 16);
            }

            const local = this.localMatrices.subarray(offset, offset + 16);
            const parentIndex = this.nodeParents[nodeIndex];

            if (parentIndex === -1) {
                // Root node, global is just local
                this.globalMatrices.set(local, offset);
            } else {
                // global = parentGlobal * local
                const parentGlobal = getGlobal(parentIndex);
                const globalMat = MathUtils.multiply(parentGlobal as Float32Array, local as Float32Array);
                this.globalMatrices.set(globalMat, offset);
            }

            this.globalComputed[nodeIndex] = 1; // Mark as computed
            return this.globalMatrices.subarray(offset, offset + 16);
        };

        // 4. Compute Final Skinning Matrices ONLY for the joints sent to WebGPU
        for (let i = 0; i < this.jointCount; i++) {
            const nodeIndex = this.skin.joints[i];

            // Get the true global matrix (which now securely includes the Armature rotation!)
            const globalMat = getGlobal(nodeIndex);
            const invBindMat = this.skin.inverseBindMatrices.subarray(i * 16, i * 16 + 16);

            // skinMatrix = globalMat * inverseBindMatrix
            const skinMat = MathUtils.multiply(globalMat as Float32Array, invBindMat as Float32Array);
            this.skinningMatrices.set(skinMat, i * 16);
        }
    }
}