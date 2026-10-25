// Spike S1's page (docs/offline-films-browser-design.md §14). One case per load, from the query, e.g.
//   index.html?case=export&way=visible|offscreen|worker|readback&w=1920&h=1080&frames=300&target=memory|opfs&id=…
// It logs "[s1] done" or "[s1] failed: …" and leaves the result in globalThis.__FILM_S1__ for run.mjs.
// While a film runs it also watches its own main thread: rAF intervals and long tasks.

import { parseCase, runCase, surfaceFor, upload } from './film.js';

const logEl = document.querySelector('#log');
const log = (line, level = 'log') => {
    console[level](`[s1] ${line}`);
    logEl.textContent += `${line}\n`;
};

/**
 * rAF intervals and long tasks between start() and stop(): how free the page's own thread was. The spike's
 * own frame sampling (pause() … resume()) is left out: it copies and hashes whole frames on this thread.
 */
function mainThreadMonitor() {
    const intervals = [];
    const longTasks = [];
    const pauses = [];
    let on = false;
    let paused = false;
    let last = 0;
    let raf = 0;
    const observer = new PerformanceObserver(list => { if (on) longTasks.push(...list.getEntries()); });
    try { observer.observe({ type: 'longtask' }); } catch { /* not supported */ }
    const tick = t => {
        if (!on) return;
        if (!paused && last) intervals.push(t - last);
        last = paused ? 0 : t;
        raf = requestAnimationFrame(tick);
    };
    return {
        start() { on = true; last = 0; raf = requestAnimationFrame(tick); },
        pause(at) { paused = true; pauses.push([at, Infinity]); },
        resume(at) { paused = false; last = 0; if (pauses.length) pauses.at(-1)[1] = at; },
        stop() {
            on = false;
            cancelAnimationFrame(raf);
            const kept = longTasks.filter(e => !pauses.some(([a, b]) => e.startTime < b && e.startTime + e.duration > a));
            const sorted = [...intervals].sort((a, b) => a - b);
            const at = p => sorted.length ? Math.round(sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))] * 10) / 10 : null;
            return {
                rafCount: intervals.length, rafP50: at(0.5), rafP90: at(0.9), rafMax: at(1),
                longTasks: kept.length, longTaskMs: Math.round(kept.reduce((a, e) => a + e.duration, 0)),
                excluded: { pauses: pauses.length, longTasks: longTasks.length - kept.length },
            };
        },
    };
}

const monitor = mainThreadMonitor();
let mainThread = null;
const marks = [];
const mark = (name, at = performance.timeOrigin + performance.now()) => {
    marks.push({ name, at });
    console.log(`[s1] mark ${name} ${at.toFixed(1)}`);
    if (name === 'export-start' || name === 'pace-start') monitor.start();
    if (name === 'sample-start') monitor.pause(at - performance.timeOrigin);
    if (name === 'sample-end') monitor.resume(at - performance.timeOrigin);
    if (name === 'frames-done' || name === 'pace-end') mainThread = monitor.stop();
};

/** JS memory at a point of the film: the V8 heap, and the page's whole JS memory with its ArrayBuffers. */
const memory = [];
async function measure(label) {
    const entry = { label, at: performance.timeOrigin + performance.now(), heapUsed: performance.memory?.usedJSHeapSize ?? null };
    if (crossOriginIsolated && performance.measureUserAgentSpecificMemory) {
        try {
            const m = await Promise.race([performance.measureUserAgentSpecificMemory(), new Promise((_, no) => setTimeout(() => no(new Error('timeout')), 15000))]);
            entry.uaBytes = m.bytes;
        } catch (e) { entry.uaError = String(e?.message ?? e); }
    }
    memory.push(entry);
}

function runInWorker(cfg) {
    return new Promise((resolve, reject) => {
        const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
        worker.onmessage = ({ data }) => {
            if (data.type === 'log') log(data.line, data.level);
            else if (data.type === 'mark') mark(data.name, data.at);
            else if (data.type === 'done') { worker.terminate(); resolve(data.result); }
            else if (data.type === 'failed') { worker.terminate(); reject(new Error(data.error)); }
        };
        worker.onerror = e => { worker.terminate(); reject(new Error(`worker: ${e.message}`)); };
        worker.postMessage(cfg);
    });
}

try {
    const cfg = parseCase(new URLSearchParams(location.search));
    log(`${cfg.id}: ${cfg.case} ${cfg.way} ${cfg.w}×${cfg.h} ${cfg.frames} frames`);
    const result = cfg.way === 'worker'
        ? await runInWorker(cfg)
        : await runCase(cfg, { surfaceKind: surfaceFor(cfg.way), canvas: document.querySelector('#visible'), log, mark, upload, measure });
    globalThis.__FILM_S1__ = {
        ok: true, ...result, mainThread, marks, memory,
        browser: { userAgent: navigator.userAgent, crossOriginIsolated, hardwareConcurrency: navigator.hardwareConcurrency },
    };
    log('done');
} catch (e) {
    globalThis.__FILM_S1__ = { ok: false, error: String(e?.stack ?? e), marks, memory };
    log(`failed: ${e?.message ?? e}`, 'error');
}
