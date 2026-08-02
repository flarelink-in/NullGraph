import {BaseLightCursor} from "./light-proxies";

export class SpotLightProxy extends BaseLightCursor {
    public seek(index: number): void {
        super.seek(index);
        this.target[this.base + 3] = 3.0; // pos_type.w: Type = Spot
    }

    public set position(v: [number, number, number] | Float32Array) {
        // pos_type.xyz
        this.target[this.base] = v[0];
        this.target[this.base + 1] = v[1];
        this.target[this.base + 2] = v[2];
    }

    public set direction(v: [number, number, number] | Float32Array) {
        // dir_rad.xyz
        this.target[this.base + 8] = v[0];
        this.target[this.base + 9] = v[1];
        this.target[this.base + 10] = v[2];
    }

    public set radius(r: number) {
        // dir_rad.w
        this.target[this.base + 11] = r;
    }

    public set innerConeAngle(angle: number) {
        // spot_params.x - Must be stored as Cosine!
        this.target[this.base + 12] = Math.cos(angle);
    }

    public set outerConeAngle(angle: number) {
        // spot_params.y - Must be stored as Cosine!
        this.target[this.base + 13] = Math.cos(angle);
    }
}