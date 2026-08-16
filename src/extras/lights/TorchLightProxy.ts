import { BaseLightCursor } from "./light-proxies";

export class TorchLightProxy extends BaseLightCursor {
    public seek(index: number): void {
        super.seek(index);
        this.target[this.base + 3] = 7.0; // Auto-set Type: Torch
    }

    /** World-space position [x, y, z]. */
    public set position(v: [number, number, number] | Float32Array) {
        this.target[this.base + 0] = v[0];
        this.target[this.base + 1] = v[1];
        this.target[this.base + 2] = v[2];
    }
    public get position(): [number, number, number] {
        return [
            this.target[this.base + 0],
            this.target[this.base + 1],
            this.target[this.base + 2],
        ];
    }

    /** Direction of torch cone axis [dx, dy, dz]. */
    public set direction(v: [number, number, number] | Float32Array) {
        this.target[this.base + 8] = v[0];
        this.target[this.base + 9] = v[1];
        this.target[this.base + 10] = v[2];
    }
    public get direction(): [number, number, number] {
        return [
            this.target[this.base + 8],
            this.target[this.base + 9],
            this.target[this.base + 10],
        ];
    }

    /** Range of torch light in world units. */
    public set radius(r: number) {
        this.target[this.base + 11] = r;
    }
    public get radius(): number {
        return this.target[this.base + 11];
    }

    /** Inner full-illumination cone half-angle in radians. */
    public set innerConeAngle(angle: number) {
        this.target[this.base + 12] = Math.cos(angle);
    }
    public get innerConeAngle(): number {
        return Math.acos(this.target[this.base + 12]);
    }

    /** Outer cutoff cone half-angle in radians. */
    public set outerConeAngle(angle: number) {
        this.target[this.base + 13] = Math.cos(angle);
    }
    public get outerConeAngle(): number {
        return Math.acos(this.target[this.base + 13]);
    }
    
    /** Flicker amplitude multiplier. */
    public set flickerIntensity(f: number) {
        this.target[this.base + 14] = f;
    }
    public get flickerIntensity(): number {
        return this.target[this.base + 14];
    }
    
    /** Flicker frequency / speed. */
    public set flickerSpeed(s: number) {
        this.target[this.base + 15] = s;
    }
    public get flickerSpeed(): number {
        return this.target[this.base + 15];
    }
}
