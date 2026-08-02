export interface OrbitalStateOptions {
    target?: [number, number, number];
    distance?: number;
    azimuthAngle?: number;
    polarAngle?: number;

    minDistance?: number;
    maxDistance?: number;

    minPolarAngle?: number;
    maxPolarAngle?: number;

    minAzimuthAngle?: number;
    maxAzimuthAngle?: number;
}

export class OrbitalCameraState {
    public target: [number, number, number];
    public distance: number;
    public azimuthAngle: number;
    public polarAngle: number;

    private minDistance: number;
    private maxDistance: number;
    private minPolarAngle: number;
    private maxPolarAngle: number;
    private minAzimuthAngle: number;
    private maxAzimuthAngle: number;

    constructor(options: OrbitalStateOptions = {}) {
        this.target = options.target || [0, 1.0, 0];
        this.distance = options.distance || 7;
        this.azimuthAngle = options.azimuthAngle || 0;

        // Default to roughly 68 degrees (looking slightly down)
        this.polarAngle = options.polarAngle || 1.2;

        this.minDistance = options.minDistance || 2;
        this.maxDistance = options.maxDistance || 20;

        // --- FULL SPHERICAL FREEDOM DEFAULTS ---

        // Vertical Limits (Polar)
        // 0.01 prevents Gimbal lock at the North Pole (top)
        this.minPolarAngle = options.minPolarAngle ?? 0.01;

        // Math.PI (180 degrees) allows going completely underneath.
        // We subtract 0.01 to prevent Gimbal lock at the South Pole (bottom)
        this.maxPolarAngle = options.maxPolarAngle ?? (Math.PI - 0.01);

        // Horizontal Limits (Azimuth)
        // -Infinity to Infinity allows endless spinning around the Y-axis
        this.minAzimuthAngle = options.minAzimuthAngle ?? -Infinity;
        this.maxAzimuthAngle = options.maxAzimuthAngle ?? Infinity;
    }

    /**
     * Apply rotation deltas (from mouse/touch movement)
     */
    public applyRotation(deltaAzimuth: number, deltaPolar: number) {
        this.azimuthAngle -= deltaAzimuth;
        this.polarAngle -= deltaPolar;

        this.azimuthAngle = Math.max(this.minAzimuthAngle, Math.min(this.maxAzimuthAngle, this.azimuthAngle));
        this.polarAngle = Math.max(this.minPolarAngle, Math.min(this.maxPolarAngle, this.polarAngle));
    }

    /**
     * Apply zoom deltas (from scroll wheel)
     */
    public applyZoom(deltaDistance: number) {
        this.distance += deltaDistance;
        this.distance = Math.max(this.minDistance, Math.min(this.maxDistance, this.distance));
    }

    /**
     * Calculates the final Cartesian coordinates for the camera eye
     */
    public getPosition(): [number, number, number] {
        return [
            this.distance * Math.sin(this.polarAngle) * Math.sin(this.azimuthAngle) + this.target[0],
            this.distance * Math.cos(this.polarAngle) + this.target[1],
            this.distance * Math.sin(this.polarAngle) * Math.cos(this.azimuthAngle) + this.target[2],
        ];
    }
}