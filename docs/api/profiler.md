# Profiler API Usage

The `null-graph/profiler` module wraps WebGPU **timestamp queries** so you can measure how long the
GPU actually spent on your work — not just how long the CPU took to submit it. It resolves the
device-side timestamps and hands you the elapsed milliseconds through a callback, which you can pipe
straight into a [debug-ui widget](./debug-ui.md).

```ts
import { GPUProfiler } from 'null-graph/profiler';
```

> **Requirement:** timestamp queries are an optional WebGPU feature. Request it at init so the
> device enables it:
> ```ts
> await engine.init(canvas, { desiredFeatures: ['timestamp-query'] });
> ```
> If the feature isn't supported, `engine.init` logs a warning and the profiler is a no-op — always
> guard against that in production.

---

## `GPUProfiler`

```ts
const profiler = new GPUProfiler(engine.device);

function frame() {
    profiler.begin(); // opens a timestamp-writing scope for this frame

    engine.render();

    // Resolve last frame's timestamps; the callback gets elapsed GPU time in ms
    profiler.end((gpuTimeMs) => {
        perfWidget.updateMetrics(cpuMs, gpuTimeMs);
    });

    requestAnimationFrame(frame);
}
```

| Method | Description |
|---|---|
| `new GPUProfiler(device)` | Allocates the timestamp query set and resolve buffers. |
| `begin()` | Marks the start of a measured scope for the current frame. |
| `end(onUpdate)` | Resolves the timestamps and invokes `onUpdate(gpuTimeMs)` when the GPU results are ready. |

Because GPU results are read back asynchronously, the value delivered to `onUpdate` typically
reflects a frame or two earlier — which is exactly what you want for a smooth, non-blocking HUD.
Combine `gpuTimeMs` with a CPU-side `performance.now()` delta around your submission code to see
both halves of the frame budget at once.
