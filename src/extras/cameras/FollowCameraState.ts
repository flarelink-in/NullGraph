export interface FollowStateOptions {
    /** World-space offset from the followed point to the camera. */
    offset?: [number, number, number];
    /** Added to the followed point to form the look-at target (e.g. aim at
     *  the character's head rather than their feet). */
    lookOffset?: [number, number, number];
    /** Damping rate per second; higher snaps faster. ~0 = never catches up. */
    stiffness?: number;
}

/**
 * Third-person chase camera. Game logic pushes the followed point in with
 * setTarget() (every frame is fine); update(dt) eases the camera toward
 * point + offset with exponential damping, so distant teleports glide and
 * small jitters are absorbed. Pure math — worker-safe like the other states.
 */
export class FollowCameraState {
    public offset: [number, number, number];
    public lookOffset: [number, number, number];
    public stiffness: number;

    private followed: [number, number, number] = [0, 0, 0];
    private eye: [number, number, number];
    private initialized: boolean = false;

    constructor(options: FollowStateOptions = {}) {
        this.offset = options.offset || [0, 3, -6];
        this.lookOffset = options.lookOffset || [0, 1, 0];
        this.stiffness = options.stiffness ?? 5;
        this.eye = [this.offset[0], this.offset[1], this.offset[2]];
    }

    /**
     * The point being followed (e.g. the player's position).
     */
    public setTarget(x: number, y: number, z: number) {
        this.followed = [x, y, z];
        if (!this.initialized) {
            this.snap();
        }
    }

    /**
     * Teleport the camera straight to its desired position (level start,
     * respawn) instead of easing across the map.
     */
    public snap() {
        this.eye = [
            this.followed[0] + this.offset[0],
            this.followed[1] + this.offset[1],
            this.followed[2] + this.offset[2],
        ];
        this.initialized = true;
    }

    /**
     * Ease toward the desired position. Exponential form keeps the damping
     * identical at any frame rate.
     */
    public update(dt: number) {
        const t = 1 - Math.exp(-this.stiffness * Math.max(dt, 0));
        this.eye[0] += (this.followed[0] + this.offset[0] - this.eye[0]) * t;
        this.eye[1] += (this.followed[1] + this.offset[1] - this.eye[1]) * t;
        this.eye[2] += (this.followed[2] + this.offset[2] - this.eye[2]) * t;
    }

    public getPosition(): [number, number, number] {
        return [this.eye[0], this.eye[1], this.eye[2]];
    }

    public getTarget(): [number, number, number] {
        return [
            this.followed[0] + this.lookOffset[0],
            this.followed[1] + this.lookOffset[1],
            this.followed[2] + this.lookOffset[2],
        ];
    }
}
