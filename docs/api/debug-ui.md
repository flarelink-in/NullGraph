# Debug UI API Usage

The `null-graph/debug-ui` module renders real-time telemetry (FPS, CPU/GPU timing) straight into
the DOM. It ships two widgets that share the same `tick()` / `updateMetrics()` contract, so you
can swap one for the other without touching your render loop.

- **`PerformanceWidget`** — the full HUD: styled panel with graphs, draggable, tunable opacity/blur,
  and an optional Web Worker binding so a worker-side engine can push metrics to a main-thread panel.
- **`MinimalPerfWidget`** — a lightweight counter for when you only need the numbers.

```ts
import { PerformanceWidget, MinimalPerfWidget } from 'null-graph/debug-ui';
```

---

## `PerformanceWidget`

```ts
const perf = new PerformanceWidget({
    draggable: true,   // adds a grab handle (default false)
    opacity: 0.85,     // panel opacity
    blur: 8,           // backdrop blur in px
});

function frame() {
    perf.tick(); // call once per frame — measures wall-clock FPS

    // ... your render work ...

    // Feed timings you already have (e.g. from the GPUProfiler)
    perf.updateMetrics(cpuMs, gpuMs /*, optional fps override */);
    requestAnimationFrame(frame);
}
```

### Worker binding

If your engine runs in a Web Worker, keep the widget on the main thread and bind it to the worker.
The widget listens for metric messages the worker posts and updates itself:

```ts
const worker = new Worker(new URL('./engine.worker.ts', import.meta.url), { type: 'module' });
perf.bindWorker(worker);
```

## `MinimalPerfWidget`

Same core loop, smaller footprint:

```ts
const perf = new MinimalPerfWidget({ draggable: true });

function frame() {
    perf.tick();
    perf.updateMetrics(cpuMs, gpuMs);
    requestAnimationFrame(frame);
}
```

| Method | Widget | Description |
|---|---|---|
| `tick()` | both | Advances the frame counter and computes FPS from wall-clock time. Call once per frame. |
| `updateMetrics(cpuMs, gpuMs, fps?)` | both (`fps` only on `PerformanceWidget`) | Pushes CPU/GPU millisecond readings into the display. |
| `bindWorker(worker)` | `PerformanceWidget` | Subscribes the panel to metric messages posted from a worker. |

> GPU timings come from the [`GPUProfiler`](./profiler.md) — pair the two to get true device-side
> milliseconds rather than CPU-submitted estimates.
