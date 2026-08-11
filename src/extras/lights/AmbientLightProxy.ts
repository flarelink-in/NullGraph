import {BaseLightCursor} from "./light-proxies";

export class AmbientLightProxy extends BaseLightCursor {
    public seek(index: number): void {
        super.seek(index);
        this.target[this.base + 3] = 4.0; // Auto-set Type: Ambient
    }
}
