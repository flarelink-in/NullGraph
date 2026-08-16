// d:\orion-realms\null-graph\src\extras\lights\light-proxies.ts

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

    /** Light color [r, g, b] in linear HDR space. */
    public set color(v: [number, number, number] | Float32Array) {
        this.target[this.base + 4] = v[0];
        this.target[this.base + 5] = v[1];
        this.target[this.base + 6] = v[2];
    }
    public get color(): [number, number, number] {
        return [
            this.target[this.base + 4],
            this.target[this.base + 5],
            this.target[this.base + 6],
        ];
    }

    /** Luminous intensity / multiplier. */
    public set intensity(i: number) {
        this.target[this.base + 7] = i;
    }
    public get intensity(): number {
        return this.target[this.base + 7];
    }

    /**
     * Shadow Slot Index in ShadowSystem (Phase 2 Unified Atlas).
     * Pass the 0-based registered slot (e.g. `sunSlot`, `pointSlot`), or -1 / null / undefined to disable shadows for this light.
     * Stored on the GPU buffer at offset 14 (`spot_params.z`) as `slot + 1` (where 0 indicates unshadowed).
     */
    public set shadowSlot(slot: number | null | undefined) {
        this.target[this.base + 14] = (slot !== null && slot !== undefined && slot >= 0) ? (slot + 1) : 0;
    }
    public get shadowSlot(): number {
        const raw = this.target[this.base + 14];
        return raw > 0 ? (raw - 1) : -1;
    }

    /** Returns raw type ID of light (1=Point, 2=Directional, 3=Spot, 4=Ambient, 5=Hemisphere, 6=RectArea, 7=Torch). */
    public get lightType(): number {
        return this.target[this.base + 3];
    }
}