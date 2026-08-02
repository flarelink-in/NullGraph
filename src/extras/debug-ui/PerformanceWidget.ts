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

    // Added opacity and blur to the options
    constructor(options: { draggable?: boolean; opacity?: number; blur?: number } = {}) {
        const isDraggable = options.draggable !== false;

        // Defaults: 0.85 opacity, 12px blur
        const bgOpacity1 = options.opacity !== undefined ? options.opacity : 0.85;
        const bgOpacity2 = Math.max(0, bgOpacity1 - 0.1); // Make the bottom gradient slightly more transparent
        const blurAmt = options.blur !== undefined ? options.blur : 12;

        this.cpuHistory = new Array(this.HISTORY_SIZE).fill(0);
        this.gpuHistory = new Array(this.HISTORY_SIZE).fill(0);

        // 1. Inject Styles
        this.injectCSS();

        // 2. Inject HTML
        this.container = document.createElement('div');
        this.container.id = 'ng-perf-widget';

        // Apply custom glassmorphism styles inline
        this.container.style.background = `linear-gradient(135deg, rgba(15, 15, 20, ${bgOpacity1}), rgba(25, 25, 30, ${bgOpacity2}))`;
        this.container.style.backdropFilter = `blur(${blurAmt}px)`;
        this.container.style.setProperty('-webkit-backdrop-filter', `blur(${blurAmt}px)`); // Safari support

        // Add class if draggable so CSS can show the grab handle
        if (isDraggable) {
            this.container.classList.add('ng-draggable');
        }

        this.container.innerHTML = `
            <div class="ng-metrics-grid">
                <div class="ng-metric">
                    <span class="ng-label">NET FPS</span>
                    <span id="ng-fps" class="ng-val" style="color: #fff;">---</span>
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

        // 4. Initialize Dragging conditionally
        if (isDraggable) {
            this.makeDraggable(this.container);
        }
    }

    // --- PUBLIC API ---

    public tick() {
        this.frameCount++;
        const now = performance.now();
        const elapsed = now - this.lastUpdate;

        if (elapsed >= 1000) {
            const trueFps = Math.round((this.frameCount * 1000) / elapsed);
            this.elFps.innerText = trueFps.toString();
            this.frameCount = 0;
            this.lastUpdate += 1000;
        }
    }

    public updateMetrics(cpuMs: number, gpuMs: number, fps?: number) {
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

        // Add this block to override the internal tick FPS
        if (fps !== undefined) {
            this.elFps.innerText = Math.round(fps).toString();
        }

        this.drawGraph();
    }

    public bindWorker(worker: Worker) {
        worker.addEventListener('message', (event: MessageEvent) => {
            // 1. Check if the message is our specific Float32Array buffer
            if (event.data instanceof Float32Array) {
                const buffer = event.data;
                const opcode = buffer[0];

                // 2. If it's the metrics opcode, extract the data
                if (opcode === 1) {
                    const cpu = buffer[1];
                    const gpu = buffer[2];
                    const fps = buffer[3]; // The real WebGPU FPS we added earlier

                    // 3. Update the UI
                    this.updateMetrics(cpu, gpu, fps);

                    // 4. Instantly return the buffer to the worker so it can be reused
                    worker.postMessage(buffer, [buffer.buffer]);
                }
            }
        });
    }

    // --- INTERNAL LOGIC ---

    private drawGraph() {
        const ctx = this.ctx;
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
        const ctx = this.ctx;
        if (!ctx) return;

        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;

        data.forEach((ms, i) => {
            const x = i * step;
            const y = height - (Math.min(ms, 32) / 32) * height;
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

            initialX = el.offsetLeft;
            initialY = el.offsetTop;

            el.style.right = 'auto';
            el.style.left = `${initialX}px`;
            el.style.top = `${initialY}px`;

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
                position: fixed; top: 20px; right: 20px; width: 240px;
                /* Background and Blur are now handled via inline styles for customizability */
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-top: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 12px; padding: 14px;
                z-index: 9999; 
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255,255,255,0.1);
                transition: border-color 0.2s ease;
                color: #fff;
            }
            
            #ng-perf-widget.ng-draggable {
                cursor: grab;
                user-select: none;
            }
            #ng-perf-widget.ng-draggable:active { 
                cursor: grabbing; 
            }
            #ng-perf-widget:hover { border-color: rgba(255, 255, 255, 0.2); }

            #ng-perf-widget.ng-draggable::before {
                content: ''; display: block; width: 36px; height: 4px;
                background: rgba(255, 255, 255, 0.15); border-radius: 2px;
                margin: 0 auto 12px auto;
            }

            .ng-metrics-grid { 
                display: flex; justify-content: space-between; gap: 8px; margin-bottom: 12px; 
            }
            .ng-metric { 
                display: flex; flex-direction: column; flex: 1;
                background: rgba(0, 0, 0, 0.4); border-radius: 6px;
                padding: 6px 8px; border: 1px solid rgba(255,255,255,0.05);
            }
            
            .ng-label { 
                color: #9ca3af; font-size: 10px; font-weight: 600; 
                letter-spacing: 0.5px; margin-bottom: 4px; 
            }
            .ng-val { 
                font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
                font-size: 14px; font-weight: 700; 
                text-shadow: 0 0 10px currentColor; 
            }

            #ng-perf-canvas { 
                width: 100%; height: 60px; 
                background: rgba(0,0,0,0.5); border-radius: 6px; 
                border: 1px solid rgba(255, 255, 255, 0.05);
                pointer-events: none; 
            }
        `;
        document.head.appendChild(style);
    }
}