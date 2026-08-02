export class MinimalPerfWidget {
    private container: HTMLElement;

    private canvasFps: HTMLCanvasElement;
    private ctxFps: CanvasRenderingContext2D | null;

    private canvasCpu: HTMLCanvasElement;
    private ctxCpu: CanvasRenderingContext2D | null;

    private canvasGpu: HTMLCanvasElement;
    private ctxGpu: CanvasRenderingContext2D | null;

    private elFpsVal: HTMLElement;
    private elCpuVal: HTMLElement;
    private elGpuVal: HTMLElement;

    private readonly HISTORY_SIZE = 60;
    private fpsHistory: number[];
    private cpuHistory: number[];
    private gpuHistory: number[];

    private frameCount = 0;
    private lastTextUpdate = performance.now();
    private lastFrameTime = performance.now();

    constructor(options: { draggable?: boolean } = {}) {
        const isDraggable = options.draggable !== false;

        this.fpsHistory = new Array(this.HISTORY_SIZE).fill(0);
        this.cpuHistory = new Array(this.HISTORY_SIZE).fill(0);
        this.gpuHistory = new Array(this.HISTORY_SIZE).fill(0);

        this.injectCSS();

        this.container = document.createElement('div');
        this.container.id = 'ng-unified-perf-widget';
        if (isDraggable) this.container.classList.add('ng-draggable');

        // Clean, cohesive stacked layout
        this.container.innerHTML = `
            <div class="ng-up-top-header">
                <span class="ng-up-title">LIVE PROFILER</span>
                <span class="ng-up-badge">ACTIVE</span>
            </div>

            <div class="ng-up-section">
                <div class="ng-up-header">
                    <span class="ng-up-label">FPS</span>
                    <span id="ng-up-fps-val" class="ng-up-val" style="color: #00E5FF;">0</span>
                </div>
                <canvas id="ng-up-fps-canvas" width="220" height="40"></canvas>
            </div>

            <div class="ng-up-section">
                <div class="ng-up-header">
                    <span class="ng-up-label">CPU Time</span>
                    <span id="ng-up-cpu-val" class="ng-up-val" style="color: #FFB300;">0.0ms</span>
                </div>
                <canvas id="ng-up-cpu-canvas" width="220" height="40"></canvas>
            </div>

            <div class="ng-up-section">
                <div class="ng-up-header">
                    <span class="ng-up-label">GPU Time</span>
                    <span id="ng-up-gpu-val" class="ng-up-val" style="color: #00ffcc;">0.0ms</span>
                </div>
                <canvas id="ng-up-gpu-canvas" width="220" height="40"></canvas>
            </div>
        `;
        document.body.appendChild(this.container);

        this.canvasFps = document.getElementById('ng-up-fps-canvas') as HTMLCanvasElement;
        this.ctxFps = this.canvasFps.getContext('2d');

        this.canvasCpu = document.getElementById('ng-up-cpu-canvas') as HTMLCanvasElement;
        this.ctxCpu = this.canvasCpu.getContext('2d');

        this.canvasGpu = document.getElementById('ng-up-gpu-canvas') as HTMLCanvasElement;
        this.ctxGpu = this.canvasGpu.getContext('2d');

        this.elFpsVal = document.getElementById('ng-up-fps-val') as HTMLElement;
        this.elCpuVal = document.getElementById('ng-up-cpu-val') as HTMLElement;
        this.elGpuVal = document.getElementById('ng-up-gpu-val') as HTMLElement;

        if (isDraggable) {
            this.makeDraggable(this.container);
        }
    }

    // --- PUBLIC API ---

    public tick() {
        const now = performance.now();

        // 1. Calculate Instantaneous FPS for a responsive, jitter-accurate graph
        const frameElapsed = now - this.lastFrameTime;
        this.lastFrameTime = now;

        const instantFps = frameElapsed > 0 ? 1000 / frameElapsed : 0;
        this.fpsHistory.push(instantFps);
        this.fpsHistory.shift();

        // 2. Calculate Averaged FPS for the text readout (1-second intervals)
        this.frameCount++;
        const elapsedSinceTextUpdate = now - this.lastTextUpdate;

        if (elapsedSinceTextUpdate >= 1000) {
            const trueFps = Math.round((this.frameCount * 1000) / elapsedSinceTextUpdate);
            this.elFpsVal.innerText = trueFps.toString();
            this.frameCount = 0;
            this.lastTextUpdate = now;
        }
    }

    public updateMetrics(cpuMs: number, gpuMs: number) {
        this.cpuHistory.push(cpuMs);
        this.cpuHistory.shift();

        this.gpuHistory.push(gpuMs);
        this.gpuHistory.shift();

        this.elCpuVal.innerText = `${cpuMs.toFixed(1)}ms`;

        if (gpuMs === 0) {
            this.elGpuVal.innerText = `N/A`;
            this.elGpuVal.style.color = '#555';
        } else {
            this.elGpuVal.innerText = `${gpuMs.toFixed(1)}ms`;
            this.elGpuVal.style.color = '#00ffcc';
        }

        this.drawGraphs();
    }

    // --- INTERNAL LOGIC ---

    private drawGraphs() {
        const dpr = window.devicePixelRatio || 1;
        const cssW = 220, cssH = 40;

        this.resizeCanvas(this.canvasFps, cssW, cssH, dpr);
        this.resizeCanvas(this.canvasCpu, cssW, cssH, dpr);
        this.resizeCanvas(this.canvasGpu, cssW, cssH, dpr);

        const step = cssW / (this.HISTORY_SIZE - 1);

        // Render FPS (Max expected target ~144)
        this.drawGraph(this.ctxFps, this.fpsHistory, cssW, cssH, step, 144, '#00E5FF', 'rgba(0, 229, 255, 0.15)');

        // Render CPU (Max expected target ~32ms for bad stutter scaling)
        this.drawGraph(this.ctxCpu, this.cpuHistory, cssW, cssH, step, 32, '#FFB300', 'rgba(255, 179, 0, 0.15)');

        // Render GPU (Max expected target ~32ms)
        this.drawGraph(this.ctxGpu, this.gpuHistory, cssW, cssH, step, 32, '#00ffcc', 'rgba(0, 255, 204, 0.15)');
    }

    private drawGraph(ctx: CanvasRenderingContext2D | null, data: number[], width: number, height: number, step: number, maxVal: number, strokeColor: string, fillColor: string) {
        if (!ctx) return;

        ctx.resetTransform();
        ctx.clearRect(0, 0, width * (window.devicePixelRatio || 1), height * (window.devicePixelRatio || 1));
        ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);

        // Pass 1: Draw the filled area
        ctx.beginPath();
        ctx.moveTo(0, height);
        data.forEach((val, i) => {
            const x = i * step;
            const normalized = Math.min(Math.max(val / maxVal, 0), 1);
            const y = height - (normalized * height);
            ctx.lineTo(x, y);
        });
        ctx.lineTo(width, height);
        ctx.closePath();
        ctx.fillStyle = fillColor;
        ctx.fill();

        // Pass 2: Draw the solid top line
        ctx.beginPath();
        data.forEach((val, i) => {
            const x = i * step;
            const normalized = Math.min(Math.max(val / maxVal, 0), 1);
            const y = height - (normalized * height);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }

    private resizeCanvas(canvas: HTMLCanvasElement, w: number, h: number, dpr: number) {
        if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
            canvas.width = w * dpr;
            canvas.height = h * dpr;
        }
    }

    private makeDraggable(el: HTMLElement) {
        let isDragging = false;
        let startX = 0, startY = 0, initialX = 0, initialY = 0;

        const start = (e: MouseEvent | TouchEvent) => {
            if ((e.target as HTMLElement).tagName.toLowerCase() === 'canvas') return; // Prevent drag if clicking canvas explicitly
            isDragging = true;
            document.body.style.userSelect = 'none';
            initialX = el.offsetLeft; initialY = el.offsetTop;
            el.style.right = 'auto'; el.style.left = `${initialX}px`; el.style.top = `${initialY}px`;
            startX = e.type === 'touchstart' ? (e as TouchEvent).touches[0].clientX : (e as MouseEvent).clientX;
            startY = e.type === 'touchstart' ? (e as TouchEvent).touches[0].clientY : (e as MouseEvent).clientY;
        };

        const drag = (e: MouseEvent | TouchEvent) => {
            if (!isDragging) return;
            e.preventDefault();
            const curX = e.type === 'touchmove' ? (e as TouchEvent).touches[0].clientX : (e as MouseEvent).clientX;
            const curY = e.type === 'touchmove' ? (e as TouchEvent).touches[0].clientY : (e as MouseEvent).clientY;
            el.style.left = `${initialX + (curX - startX)}px`;
            el.style.top = `${initialY + (curY - startY)}px`;
        };

        const end = () => { isDragging = false; document.body.style.userSelect = ''; };

        el.addEventListener('mousedown', start); document.addEventListener('mousemove', drag); document.addEventListener('mouseup', end);
        el.addEventListener('touchstart', start, { passive: false }); document.addEventListener('touchmove', drag, { passive: false }); document.addEventListener('touchend', end);
    }

    private injectCSS() {
        if (document.getElementById('ng-up-style')) return;
        const style = document.createElement('style');
        style.id = 'ng-up-style';
        style.textContent = `
            #ng-unified-perf-widget {
                position: fixed; top: 20px; right: 20px; width: 220px;
                background: rgba(15, 15, 20, 0.95); /* Deep dark background */
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-top: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 8px;
                padding: 14px 16px;
                z-index: 9999; 
                font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7);
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                color: #fff;
            }
            
            #ng-unified-perf-widget.ng-draggable { cursor: grab; }
            #ng-unified-perf-widget.ng-draggable:active { cursor: grabbing; }

            .ng-up-top-header {
                display: flex; justify-content: space-between; align-items: center;
                margin-bottom: 16px;
                padding-bottom: 8px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            }

            .ng-up-title {
                color: #e5e7eb; font-size: 10px; font-weight: 700;
                letter-spacing: 1.5px; text-transform: uppercase;
            }

            .ng-up-badge {
                color: #00E5FF; font-size: 9px; font-weight: 800;
                border: 1px solid #00E5FF; padding: 2px 6px; border-radius: 4px;
                letter-spacing: 1px;
            }

            .ng-up-section {
                display: flex; flex-direction: column; width: 100%; margin-bottom: 14px;
            }
            .ng-up-section:last-child { margin-bottom: 0; }

            .ng-up-header {
                display: flex; justify-content: space-between; align-items: flex-end;
                margin-bottom: 6px; padding: 0 2px;
            }

            .ng-up-label {
                color: #9ca3af; font-size: 11px; font-weight: 600;
                letter-spacing: 1px;
            }

            .ng-up-val {
                font-size: 14px; font-weight: 700;
                text-shadow: 0 0 10px currentColor; 
            }

            canvas[id^="ng-up-"] {
                width: 100%; height: 40px;
                background: transparent; /* Fixed the grey box issue */
                pointer-events: none;
            }
        `;
        document.head.appendChild(style);
    }
}