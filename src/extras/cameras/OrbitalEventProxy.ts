export interface OrbitalProxyOptions {
    canvas: HTMLCanvasElement;
    sensitivity?: number;
    zoomSensitivity?: number;
    // This callback is the bridge! It passes pure numbers out.
    onInput: (type: 'ROTATE' | 'ZOOM', x: number, y: number) => void;
}

export class OrbitalEventProxy {
    private canvas: HTMLCanvasElement;
    private sensitivity: number;
    private zoomSensitivity: number;
    private onInput: (type: 'ROTATE' | 'ZOOM', x: number, y: number) => void;

    private isDragging: boolean = false;
    private lastX: number = 0;
    private lastY: number = 0;

    constructor(options: OrbitalProxyOptions) {
        this.canvas = options.canvas;
        this.sensitivity = options.sensitivity ?? 0.01;
        this.zoomSensitivity = options.zoomSensitivity ?? 0.05;
        this.onInput = options.onInput;

        this.initEvents();
    }

    private initEvents() {
        this.canvas.addEventListener('pointerdown', this.onPointerDown);
        window.addEventListener('pointermove', this.onPointerMove);
        window.addEventListener('pointerup', this.onPointerUp);
        this.canvas.addEventListener('wheel', this.onWheel, { passive: true });
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

        // Fire the abstract callback
        this.onInput('ROTATE', deltaX * this.sensitivity, deltaY * this.sensitivity);
    };

    private onPointerUp = () => {
        this.isDragging = false;
    };

    private onWheel = (e: WheelEvent) => {
        this.onInput('ZOOM', 0, e.deltaY * this.zoomSensitivity);
    };

    public destroy() {
        this.canvas.removeEventListener('pointerdown', this.onPointerDown);
        window.removeEventListener('pointermove', this.onPointerMove);
        window.removeEventListener('pointerup', this.onPointerUp);
        this.canvas.removeEventListener('wheel', this.onWheel);
    }
}