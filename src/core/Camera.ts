import { mat4, vec3 } from 'gl-matrix';

/*
struct Camera {
    viewProj: mat4x4<f32>,      // Used for standard vertex transformation
    view: mat4x4<f32>,          // Used for view-space lighting/normals
    proj: mat4x4<f32>,          // Used for depth buffer linearization
    invViewProj: mat4x4<f32>,   // Used for World-Space reconstruction (SSR, Decals)
    invView: mat4x4<f32>,       // Used for Billboarding and extracting camera Right/Up vectors
    invProj: mat4x4<f32>,       // Used for Clustered Culling (Screen to View space)

    eyePosition: vec3<f32>,     // Used for PBR Specular highlights
    _pad0: f32,                 // WGSL alignment padding

    zNear: f32,                 // Used for clustered depth slicing
    zFar: f32,                  // Used for clustered depth slicing
    fovRads: f32,               // Useful for LOD calculations
    aspect: f32                 // Useful for post-processing effects
    _padEnd: vec4<f32>          // Indices 104, 105, 106, 107 (Ready for future use!)
};
@group(0) @binding(0) var<uniform> camera: Camera;
 */
export class Camera {
    // Core Matrices
    public viewProjectionMatrix: mat4 = mat4.create();
    public viewMatrix: mat4 = mat4.create();
    public projectionMatrix: mat4 = mat4.create();

    // Inverse Matrices (The Future-Proofing)
    public inverseViewProjectionMatrix: mat4 = mat4.create();
    public inverseViewMatrix: mat4 = mat4.create();
    public inverseProjectionMatrix: mat4 = mat4.create();

    // The single, contiguous memory block (108 floats = 432 bytes)
    public bufferData: Float32Array = new Float32Array(108);

    public fovDegrees: number;
    public near: number;
    public far: number;
    private currentAspect: number;

    constructor(fovDegrees: number, aspect: number, near: number, far: number) {
        this.fovDegrees = fovDegrees;
        this.near = near;
        this.far = far;
        this.currentAspect = aspect;

        this.updateProjection(aspect);
    }

    /*
    to update time in the padding slot directly call this,it sets in camera buffer
     */
    public updateTime(simTime: number) {
        this.bufferData[99] = simTime; // Write directly into the padding slot!
    }

    public updateProjection(aspect: number) {
        this.currentAspect = aspect;
        const fovRads = (this.fovDegrees * Math.PI) / 180;

        // 1. Calculate Projection — perspectiveZO, because WebGPU's clip volume is z ∈ [0, 1],
        //    NOT OpenGL's [-1, 1]. A NO projection has every point with z_ndc < 0 clipped, which
        //    silently moves the near plane to 2×near (and the far plane to ~2×far): geometry
        //    within that shell vanishes/pops right at the camera.
        mat4.perspectiveZO(this.projectionMatrix, fovRads, aspect, this.near, this.far);

        // 2. Calculate Inverse Projection
        mat4.invert(this.inverseProjectionMatrix, this.projectionMatrix);

        // 3. Update Combined Matrices
        mat4.multiply(this.viewProjectionMatrix, this.projectionMatrix, this.viewMatrix);
        mat4.invert(this.inverseViewProjectionMatrix, this.viewProjectionMatrix);

        // --- MAP TO FLOAT32ARRAY ---
        this.bufferData.set(this.viewProjectionMatrix, 0);       // Offset 0
        // View Matrix handles offset 16-31
        this.bufferData.set(this.projectionMatrix, 32);          // Offset 32
        this.bufferData.set(this.inverseViewProjectionMatrix, 48); // Offset 48
        // Inverse View handles offset 64-79
        this.bufferData.set(this.inverseProjectionMatrix, 80);   // Offset 80

        // Scalar properties at the tail end
        this.bufferData[100] = this.near;
        this.bufferData[101] = this.far;
        this.bufferData[102] = fovRads;
        this.bufferData[103] = aspect;
    }

    public updateView(eye: vec3, target: vec3, up: vec3 = [0, 1, 0]) {
        // 1. Calculate View
        mat4.lookAt(this.viewMatrix, eye, target, up);

        // 2. Calculate Inverse View
        mat4.invert(this.inverseViewMatrix, this.viewMatrix);

        // 3. Update Combined Matrices
        mat4.multiply(this.viewProjectionMatrix, this.projectionMatrix, this.viewMatrix);
        mat4.invert(this.inverseViewProjectionMatrix, this.viewProjectionMatrix);

        // --- MAP TO FLOAT32ARRAY ---
        this.bufferData.set(this.viewProjectionMatrix, 0);       // Offset 0
        this.bufferData.set(this.viewMatrix, 16);                // Offset 16
        // Proj is at 32
        this.bufferData.set(this.inverseViewProjectionMatrix, 48); // Offset 48
        this.bufferData.set(this.inverseViewMatrix, 64);         // Offset 64
        // InvProj is at 80

        // Eye Position (vec3 at offset 96)
        this.bufferData.set(eye, 96);                            // Offset 96
        // Float 99 is implicit padding
    }
}