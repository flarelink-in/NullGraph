// light-proxies.ts

export class BaseLightCursor {
    protected target!: Float32Array;
    protected base: number = 0;
    public static readonly STRIDE = 16;

    public wrap(array: Float32Array): void {
        this.target = array;
    }

    public seek(index: number): void {
        this.base = index * BaseLightCursor.STRIDE;
    }

    // Shared properties across ALL lights
    public set color(v: [number, number, number] | Float32Array) {
        this.target[this.base + 4] = v[0];
        this.target[this.base + 5] = v[1];
        this.target[this.base + 6] = v[2];
    }

    public set intensity(i: number) {
        this.target[this.base + 7] = i;
    }
}