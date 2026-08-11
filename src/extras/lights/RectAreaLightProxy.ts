import {BaseLightCursor} from "./light-proxies";

export class RectAreaLightProxy extends BaseLightCursor {
    public seek(index: number): void {
        super.seek(index);
        this.target[this.base + 3] = 6.0; // Auto-set Type: RectArea
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

    public set width(w: number) {
        this.target[this.base + 12] = w;
    }

    public set height(h: number) {
        this.target[this.base + 13] = h;
    }
}
