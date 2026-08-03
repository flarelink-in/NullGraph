export interface FlyStateOptions {
    position?: [number, number, number];
    /** Horizontal look angle in radians. 0 looks toward +Z (matches the
     *  orbital azimuth convention). */
    yaw?: number;
    /** Vertical look angle in radians. Positive looks up. */
    pitch?: number;

    moveSpeed?: number;

    minPitch?: number;
    maxPitch?: number;
}

/**
 * First-person free-fly camera. Pure math — no DOM, no events — so it can
 * live on the main thread or inside a Web Worker, fed by FlyEventProxy (or
 * anything else that produces look deltas and move axes).
 */
export class FlyCameraState {
    public position: [number, number, number];
    public yaw: number;
    public pitch: number;
    public moveSpeed: number;

    private minPitch: number;
    private maxPitch: number;

    constructor(options: FlyStateOptions = {}) {
        this.position = options.position || [0, 2, -8];
        this.yaw = options.yaw || 0;
        this.pitch = options.pitch || 0;
        this.moveSpeed = options.moveSpeed || 6;

        // Just short of straight up/down so the look direction never becomes
        // parallel to the world up vector (which would break updateView).
        this.minPitch = options.minPitch ?? (-Math.PI / 2 + 0.01);
        this.maxPitch = options.maxPitch ?? (Math.PI / 2 - 0.01);
    }

    /**
     * Apply look deltas (from mouse/touch movement)
     */
    public applyLook(deltaYaw: number, deltaPitch: number) {
        this.yaw -= deltaYaw;
        this.pitch -= deltaPitch;
        this.pitch = Math.max(this.minPitch, Math.min(this.maxPitch, this.pitch));
    }

    /**
     * Move relative to the view. Axes are -1..1 (e.g. from held keys):
     * forward along the look direction, right perpendicular to it on the
     * ground plane, up along world Y. dt keeps speed frame-rate independent.
     */
    public applyMove(forward: number, right: number, up: number, dt: number) {
        const step = this.moveSpeed * dt;
        const cosP = Math.cos(this.pitch);
        const dirX = cosP * Math.sin(this.yaw);
        const dirY = Math.sin(this.pitch);
        const dirZ = cosP * Math.cos(this.yaw);
        // Right-hand perpendicular of the forward direction on the XZ plane.
        const rightX = Math.sin(this.yaw - Math.PI / 2);
        const rightZ = Math.cos(this.yaw - Math.PI / 2);

        this.position[0] += (dirX * forward + rightX * right) * step;
        this.position[1] += (dirY * forward + up) * step;
        this.position[2] += (dirZ * forward + rightZ * right) * step;
    }

    public getPosition(): [number, number, number] {
        return [this.position[0], this.position[1], this.position[2]];
    }

    /**
     * A point one unit along the look direction, for Camera.updateView().
     */
    public getTarget(): [number, number, number] {
        const cosP = Math.cos(this.pitch);
        return [
            this.position[0] + cosP * Math.sin(this.yaw),
            this.position[1] + Math.sin(this.pitch),
            this.position[2] + cosP * Math.cos(this.yaw),
        ];
    }
}
