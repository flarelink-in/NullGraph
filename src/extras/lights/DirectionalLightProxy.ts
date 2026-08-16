import { BaseLightCursor } from "./light-proxies";

export class DirectionalLightProxy extends BaseLightCursor {
    public seek(index: number): void {
        super.seek(index);
        this.target[this.base + 3] = 2.0; // Auto-set Type: Directional
    }

    /** Direction of light travel [dx, dy, dz]. */
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
}