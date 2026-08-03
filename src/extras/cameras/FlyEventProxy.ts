export interface FlyProxyOptions {
    canvas: HTMLCanvasElement;
    sensitivity?: number;
    // Same bridge pattern as OrbitalEventProxy: pure numbers out.
    onInput: (type: 'LOOK', x: number, y: number) => void;
}

/**
 * Input listener for FlyCameraState. Drag to look; movement keys are tracked
 * as state and POLLED once per frame via getMoveAxes(), because "key held"
 * is continuous — events alone can't express it. The polled axes are plain
 * numbers, so they can be posted to a worker that owns the state.
 *
 * Keys: W/S forward-back, D/A right-left, E/Q up-down.
 */
export class FlyEventProxy {
    private canvas: HTMLCanvasElement;
    private sensitivity: number;
    private onInput: (type: 'LOOK', x: number, y: number) => void;

    private isDragging: boolean = false;
    private lastX: number = 0;
    private lastY: number = 0;
    private keys: Set<string> = new Set();

    constructor(options: FlyProxyOptions) {
        this.canvas = options.canvas;
        this.sensitivity = options.sensitivity ?? 0.005;
        this.onInput = options.onInput;

        this.initEvents();
    }

    private initEvents() {
        this.canvas.addEventListener('pointerdown', this.onPointerDown);
        window.addEventListener('pointermove', this.onPointerMove);
        window.addEventListener('pointerup', this.onPointerUp);
        window.addEventListener('keydown', this.onKeyDown);
        window.addEventListener('keyup', this.onKeyUp);
    }

    /**
     * [forward, right, up], each -1..1. Poll once per frame and hand to
     * FlyCameraState.applyMove() together with the frame's dt.
     */
    public getMoveAxes(): [number, number, number] {
        const k = this.keys;
        return [
            (k.has('KeyW') ? 1 : 0) - (k.has('KeyS') ? 1 : 0),
            (k.has('KeyD') ? 1 : 0) - (k.has('KeyA') ? 1 : 0),
            (k.has('KeyE') ? 1 : 0) - (k.has('KeyQ') ? 1 : 0),
        ];
    }

    private onPointerDown = (e: PointerEvent) => {
        this.isDragging = true;
        this.lastX = e.clientX;
        this.lastY = e.clientY;
    };

    private onPointerMove = (e: PointerEvent) => {
        if (!this.isDragging) return;

        const deltaX = e.clientX - this.lastX;
        const deltaY = e.clientY - this.lastY;

        this.lastX = e.clientX;
        this.lastY = e.clientY;

        // Negated Y so dragging up looks up.
        this.onInput('LOOK', deltaX * this.sensitivity, -deltaY * this.sensitivity);
    };

    private onPointerUp = () => {
        this.isDragging = false;
    };

    private onKeyDown = (e: KeyboardEvent) => {
        this.keys.add(e.code);
    };

    private onKeyUp = (e: KeyboardEvent) => {
        this.keys.delete(e.code);
    };

    public destroy() {
        this.canvas.removeEventListener('pointerdown', this.onPointerDown);
        window.removeEventListener('pointermove', this.onPointerMove);
        window.removeEventListener('pointerup', this.onPointerUp);
        window.removeEventListener('keydown', this.onKeyDown);
        window.removeEventListener('keyup', this.onKeyUp);
        this.keys.clear();
    }
}
