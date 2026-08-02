import {BaseLightCursor} from "./light-proxies";

export class PointLightProxy extends BaseLightCursor {
    public seek(index: number): void {
        super.seek(index);
        this.target[this.base + 3] = 1.0; // Auto-set Type: Point
    }

    public set position(v: [number, number, number] | Float32Array) {
        this.target[this.base + 0] = v[0];
        this.target[this.base + 1] = v[1];
        this.target[this.base + 2] = v[2];
    }

    public set radius(r: number) {
        this.target[this.base + 11] = r;
    }
}