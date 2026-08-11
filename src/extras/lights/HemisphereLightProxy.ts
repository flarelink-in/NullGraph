import {BaseLightCursor} from "./light-proxies";

export class HemisphereLightProxy extends BaseLightCursor {
    public seek(index: number): void {
        super.seek(index);
        this.target[this.base + 3] = 5.0; // Auto-set Type: Hemisphere
    }

    public set direction(v: [number, number, number] | Float32Array) {
        this.target[this.base + 8] = v[0];
        this.target[this.base + 9] = v[1];
        this.target[this.base + 10] = v[2];
    }

    // Ground color stored in spot parameters and radius space if needed, 
    // or we can store it in position since it has no position.
    public set groundColor(v: [number, number, number] | Float32Array) {
        this.target[this.base + 0] = v[0];
        this.target[this.base + 1] = v[1];
        this.target[this.base + 2] = v[2];
    }
}
