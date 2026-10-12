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

    private pointers = new Map<number, { x: number; y: number }>();
    private touchAction: string;

    constructor(options: OrbitalProxyOptions) {
        this.canvas = options.canvas;
        this.sensitivity = options.sensitivity ?? 0.01;
        this.zoomSensitivity = options.zoomSensitivity ?? 0.05;
        this.onInput = options.onInput;
        this.touchAction = this.canvas.style.touchAction;
        this.canvas.style.touchAction = 'none';

        this.initEvents();
    }

    private initEvents() {
        this.canvas.addEventListener('pointerdown', this.onPointerDown);
        window.addEventListener('pointermove', this.onPointerMove);
        window.addEventListener('pointerup', this.onPointerUp);
        window.addEventListener('pointercancel', this.onPointerUp);
        window.addEventListener('blur', this.onBlur);
        this.canvas.addEventListener('wheel', this.onWheel, { passive: true });
    }

    private onPointerDown = (e: PointerEvent) => {
        this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    };

    private onPointerMove = (e: PointerEvent) => {
        const pointer = this.pointers.get(e.pointerId);
        if (!pointer) return;
        const previousDistance = this.pinchDistance();
        const deltaX = e.clientX - pointer.x;
        const deltaY = e.clientY - pointer.y;
        pointer.x = e.clientX;
        pointer.y = e.clientY;

        if (this.pointers.size === 1) {
            this.onInput('ROTATE', deltaX * this.sensitivity, deltaY * this.sensitivity);
        } else if (this.pointers.size === 2) {
            // Spreading fingers reduces orbit distance; pinching increases it.
            this.onInput('ZOOM', 0, (previousDistance - this.pinchDistance()) * this.zoomSensitivity);
        }
    };

    private pinchDistance() {
        if (this.pointers.size !== 2) return 0;
        const [a, b] = this.pointers.values();
        return Math.hypot(a.x - b.x, a.y - b.y);
    }

    private onPointerUp = (e: PointerEvent) => {
        this.pointers.delete(e.pointerId);
    };

    private onBlur = () => {
        this.pointers.clear();
    };

    private onWheel = (e: WheelEvent) => {
        this.onInput('ZOOM', 0, e.deltaY * this.zoomSensitivity);
    };

    public destroy() {
        this.canvas.removeEventListener('pointerdown', this.onPointerDown);
        window.removeEventListener('pointermove', this.onPointerMove);
        window.removeEventListener('pointerup', this.onPointerUp);
        window.removeEventListener('pointercancel', this.onPointerUp);
        window.removeEventListener('blur', this.onBlur);
        this.canvas.removeEventListener('wheel', this.onWheel);
        this.pointers.clear();
        if (this.canvas.style.touchAction === 'none') this.canvas.style.touchAction = this.touchAction;
    }
}