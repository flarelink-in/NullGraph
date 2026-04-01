import { mat4, vec3 } from 'gl-matrix';


export class Camera {
    public projectionMatrix: mat4 = mat4.create();
    public viewMatrix: mat4 = mat4.create();
    public viewProjectionMatrix: mat4 = mat4.create();

    public bufferData: Float32Array = new Float32Array(20);

    constructor(fovDegrees: number, aspect: number, near: number, far: number) {
        const fovRads = (fovDegrees * Math.PI) / 180;
        mat4.perspective(this.projectionMatrix, fovRads, aspect, near, far);
    }

    public updateView(eye: vec3, target: vec3, up: vec3 = [0, 1, 0]) {
        mat4.lookAt(this.viewMatrix, eye, target, up);
        mat4.multiply(this.viewProjectionMatrix, this.projectionMatrix, this.viewMatrix);


        this.bufferData.set(this.viewProjectionMatrix, 0);

        this.bufferData.set(eye, 16);
    }
}