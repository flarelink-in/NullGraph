import {BaseLightCursor} from "./light-proxies";

export class TorchLightProxy extends BaseLightCursor {
    public seek(index: number): void {
        super.seek(index);
        this.target[this.base + 3] = 7.0; // Auto-set Type: Torch
    }

    public set position(v: [number, number, number] | Float32Array) {
        this.target[this.base + 0] = v[0];
        this.target[this.base + 1] = v[1];
        this.target[this.base + 2] = v[2];
    }

    public set direction(v: [number, number, number] | Float32Array) {
        this.target[this.base + 8] = v[0];
        this.target[this.base + 9] = v[1];
        this.target[this.base + 10] = v[2];
    }

    public set radius(r: number) {
        this.target[this.base + 11] = r;
    }

    public set innerConeAngle(angle: number) {
        this.target[this.base + 12] = Math.cos(angle);
    }

    public set outerConeAngle(angle: number) {
        this.target[this.base + 13] = Math.cos(angle);
    }
    
    public set flickerIntensity(f: number) {
        this.target[this.base + 14] = f;
    }
    
    public set flickerSpeed(s: number) {
        this.target[this.base + 15] = s;
    }
}
