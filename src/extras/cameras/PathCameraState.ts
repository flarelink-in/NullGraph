export interface PathStateOptions {
    /** Waypoints the camera dollies through. At least 2. */
    points: [number, number, number][];
    /** Seconds for one full traversal. */
    duration?: number;
    /** true: the path closes into a loop and progress wraps. */
    loop?: boolean;
    /** Fixed look-at point. Omit to look along the direction of travel. */
    lookTarget?: [number, number, number];
}

/**
 * Cinematic dolly along a Catmull-Rom spline through waypoints. Pure math —
 * advance(dt) or set progress directly (e.g. tie it to scroll position).
 * The spline passes through every waypoint; open paths clamp their end
 * tangents, looped paths wrap them.
 */
export class PathCameraState {
    /** 0..1 position along the whole path. Writable for scrubbing. */
    public progress: number = 0;
    public duration: number;
    public loop: boolean;
    public lookTarget: [number, number, number] | null;

    private points: [number, number, number][];

    constructor(options: PathStateOptions) {
        if (options.points.length < 2) {
            throw new Error('PathCameraState: at least 2 waypoints are required.');
        }
        this.points = options.points.map(p => [p[0], p[1], p[2]]);
        this.duration = options.duration ?? 10;
        this.loop = options.loop ?? false;
        this.lookTarget = options.lookTarget ? [...options.lookTarget] : null;
    }

    /**
     * Move along the path. Wraps when looping, clamps at the end otherwise.
     */
    public advance(dt: number) {
        this.progress += dt / Math.max(this.duration, 1e-6);
        if (this.loop) {
            this.progress -= Math.floor(this.progress);
        } else {
            this.progress = Math.max(0, Math.min(1, this.progress));
        }
    }

    public getPosition(): [number, number, number] {
        return this.sample(this.progress);
    }

    /**
     * Fixed lookTarget if given; otherwise a point slightly ahead on the
     * path, so the camera faces its direction of travel.
     */
    public getTarget(): [number, number, number] {
        if (this.lookTarget) {
            return [this.lookTarget[0], this.lookTarget[1], this.lookTarget[2]];
        }
        const ahead = this.loop
            ? (this.progress + 0.01) % 1
            : Math.min(this.progress + 0.01, 1);
        const t = this.sample(ahead);
        if (ahead === this.progress) {
            // Clamped at the very end of an open path: keep the last heading.
            const back = this.sample(this.progress - 0.01);
            const p = this.sample(this.progress);
            return [p[0] * 2 - back[0], p[1] * 2 - back[1], p[2] * 2 - back[2]];
        }
        return t;
    }

    /** Catmull-Rom sample at t in 0..1 across the whole path. */
    private sample(t: number): [number, number, number] {
        const n = this.points.length;
        const segments = this.loop ? n : n - 1;
        const s = Math.max(0, Math.min(t, 1)) * segments;
        const i = Math.min(Math.floor(s), segments - 1);
        const u = s - i;

        const at = (idx: number): [number, number, number] => {
            if (this.loop) return this.points[((idx % n) + n) % n];
            return this.points[Math.max(0, Math.min(idx, n - 1))];
        };
        const p0 = at(i - 1);
        const p1 = at(i);
        const p2 = at(i + 1);
        const p3 = at(i + 2);

        const out: [number, number, number] = [0, 0, 0];
        for (let c = 0; c < 3; c++) {
            out[c] = 0.5 * (
                2 * p1[c]
                + (p2[c] - p0[c]) * u
                + (2 * p0[c] - 5 * p1[c] + 4 * p2[c] - p3[c]) * u * u
                + (3 * p1[c] - p0[c] - 3 * p2[c] + p3[c]) * u * u * u
            );
        }
        return out;
    }
}
