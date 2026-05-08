export class PerformanceWidget {
    private container: HTMLElement;
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D | null;

    private elFps: HTMLElement;
    private elCpu: HTMLElement;
    private elGpu: HTMLElement;

    private readonly HISTORY_SIZE = 60;
    private cpuHistory: number[];
    private gpuHistory: number[];

    private frameCount = 0;
    private lastUpdate = performance.now();

    constructor() {
        this.cpuHistory = new Array(this.HISTORY_SIZE).fill(0);
        this.gpuHistory = new Array(this.HISTORY_SIZE).fill(0);

        // 1. Inject Styles
        this.injectCSS();

        // 2. Inject HTML
        this.container = document.createElement('div');
        this.container.id = 'ng-perf-widget';
        this.container.innerHTML = `
            <div class="ng-metrics-grid">
                <div class="ng-metric">
                    <span class="ng-label">NET FPS</span>
                    <span id="ng-fps" class="ng-val" style="color: #fff;">60</span>
                </div>
                <div class="ng-metric">
                    <span class="ng-label">CPU</span>
                    <span id="ng-cpu" class="ng-val" style="color: #ffcc00;">0.0ms</span>
                </div>
                <div class="ng-metric">
                    <span class="ng-label">GPU</span>
                    <span id="ng-gpu" class="ng-val" style="color: #00ffcc;">0.0ms</span>
                </div>
            </div>
            <canvas id="ng-perf-canvas" width="200" height="60"></canvas>
        `;
        document.body.appendChild(this.container);

        // 3. Grab Elements
        this.canvas = document.getElementById('ng-perf-canvas') as HTMLCanvasElement;
        this.ctx = this.canvas.getContext('2d');
        this.elFps = document.getElementById('ng-fps') as HTMLElement;
        this.elCpu = document.getElementById('ng-cpu') as HTMLElement;
        this.elGpu = document.getElementById('ng-gpu') as HTMLElement;

        // 4. Initialize Dragging
        this.makeDraggable(this.container);
    }

    // --- PUBLIC API ---

    /** Call this synchronously every frame to calculate Net FPS */
    public tick() {
        this.frameCount++;
        const now = performance.now();
        const elapsed = now - this.lastUpdate;

        if (elapsed >= 1000) {
            const trueFps = Math.round((this.frameCount * 1000) / elapsed);
            this.elFps.innerText = Math.min(trueFps, 144).toString();
            this.frameCount = 0;
            this.lastUpdate += 1000;
        }
    }

    /** Call this asynchronously when the GPU profiler returns data */
    public updateMetrics(cpuMs: number, gpuMs: number) {
        this.cpuHistory.push(cpuMs); this.cpuHistory.shift();
        this.gpuHistory.push(gpuMs); this.gpuHistory.shift();

        this.elCpu.innerText = `${cpuMs.toFixed(1)}ms`;
        if (gpuMs === 0) {
            this.elGpu.innerText = `N/A`;
            this.elGpu.style.color = '#555';
        } else {
            this.elGpu.innerText = `${gpuMs.toFixed(1)}ms`;
            this.elGpu.style.color = '#00ffcc';
        }

        this.drawGraph();
    }

    // --- INTERNAL LOGIC ---

    private drawGraph() {
        const ctx = this.ctx; // <-- Grab a local reference
        if (!ctx) return;

        const dpr = window.devicePixelRatio || 1;
        const cssW = 200, cssH = 60;

        if (this.canvas.width !== cssW * dpr || this.canvas.height !== cssH * dpr) {
            this.canvas.width = cssW * dpr;
            this.canvas.height = cssH * dpr;
        }

        ctx.resetTransform();
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        ctx.scale(dpr, dpr);

        ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.beginPath();
        const targetY = cssH - (16.6 / 32) * cssH;
        ctx.moveTo(0, targetY);
        ctx.lineTo(cssW, targetY);
        ctx.stroke();

        const step = cssW / (this.HISTORY_SIZE - 1);
        this.drawPath(this.cpuHistory, '#ffcc00', step, cssH);
        this.drawPath(this.gpuHistory, '#00ffcc', step, cssH);
    }

    private drawPath(data: number[], color: string, step: number, height: number) {
        const ctx = this.ctx; // <-- Grab a local reference
        if (!ctx) return;

        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;

        data.forEach((ms, i) => {
            const x = i * step;
            const y = height - (Math.min(ms, 32) / 32) * height;
            // TypeScript now trusts 'ctx' completely inside this arrow function!
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });

        ctx.stroke();
    }
    private makeDraggable(el: HTMLElement) {
        let isDragging = false;
        let startX = 0, startY = 0, initialX = 0, initialY = 0;

        const start = (e: MouseEvent | TouchEvent) => {
            isDragging = true;
            document.body.style.userSelect = 'none';
            const rect = el.getBoundingClientRect();
            initialX = rect.left; initialY = rect.top;
            el.style.right = 'auto';
            el.style.left = `${initialX}px`; el.style.top = `${initialY}px`;

            if (e.type === 'touchstart') {
                startX = (e as TouchEvent).touches[0].clientX; startY = (e as TouchEvent).touches[0].clientY;
            } else {
                startX = (e as MouseEvent).clientX; startY = (e as MouseEvent).clientY;
            }
        };

        const drag = (e: MouseEvent | TouchEvent) => {
            if (!isDragging) return;
            e.preventDefault();
            let curX, curY;
            if (e.type === 'touchmove') {
                curX = (e as TouchEvent).touches[0].clientX; curY = (e as TouchEvent).touches[0].clientY;
            } else {
                curX = (e as MouseEvent).clientX; curY = (e as MouseEvent).clientY;
            }
            el.style.left = `${initialX + (curX - startX)}px`;
            el.style.top = `${initialY + (curY - startY)}px`;
        };

        const end = () => { isDragging = false; document.body.style.userSelect = ''; };

        el.addEventListener('mousedown', start); document.addEventListener('mousemove', drag); document.addEventListener('mouseup', end);
        el.addEventListener('touchstart', start, { passive: false }); document.addEventListener('touchmove', drag, { passive: false }); document.addEventListener('touchend', end);
    }

    private injectCSS() {
        if (document.getElementById('ng-perf-style')) return;
        const style = document.createElement('style');
        style.id = 'ng-perf-style';
        style.textContent = `
            #ng-perf-widget {
                position: fixed; top: 20px; right: 20px; width: 220px;
                background: rgba(10, 10, 12, 0.85); backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 12px;
                z-index: 9999; cursor: grab; font-family: monospace; user-select: none;
                box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            }
            #ng-perf-widget:active { cursor: grabbing; }
            .ng-metrics-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 5px; margin-bottom: 8px; }
            .ng-metric { display: flex; flex-direction: column; }
            .ng-label { color: #888; font-size: 10px; font-weight: bold; }
            .ng-val { font-size: 14px; font-weight: bold; text-shadow: 0 0 5px currentColor; }
            #ng-perf-canvas { width: 100%; height: 60px; background: rgba(0,0,0,0.4); border-radius: 4px; pointer-events: none; }
        `;
        document.head.appendChild(style);
    }
}