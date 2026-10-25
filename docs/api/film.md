# Film — offline browser rendering (F1/F2)

**Implementation status, 2026-10-25:**
- **Built:** F1's helpers and engine teardown; F2's managed runner, session, readback,
  video into memory, hashes, measurements and PNG snapshots.
- **Tested:** F2's browser G suite passed on an AMD iGPU and an NVIDIA GPU in Chrome
  ([validation](#validation-and-limits)).
- **Later phases:** OPFS, picked files, PNG sequences and workers.

The design and its measurements: `docs/offline-films-browser-design.md` ("Phase F2 results").

Import from `null-graph/film`. Video encoding needs the optional peer `mediabunny` (tested
with 1.61.3); pure helpers, hashes and snapshots do not. The runner dynamically imports
`null-graph` rather than bundling another copy of the core.

## Managed recording

`recordFilm(options)` creates an isolated engine on your borrowed `GPUDevice`, builds your
workload, calibrates, resets (or rebuilds), warms up, then records. Pause the preview before
calling it and restore the preview in your application's `finally`. Never lend mutable preview
simulation state to the workload. Device loss requires application recovery, not just resume.

Required options:

| Option | Contract |
|---|---|
| `device` | Borrowed GPUDevice; never destroyed by the film. Exports and snapshots serialize per device. |
| `film` | `{ width, height, fps, seconds, warmupSeconds? }`; even dimensions, positive integer fps and a whole number of frames. |
| `createWorkload` | Async factory receiving `{ engine, width, height, signal, defer }`. |

The factory returns `{ dt, step, draw, dispose, reset?, prepare?, info? }`:

- `dt` is a positive fixed timestep. Each frame must contain a whole number of steps.
- `step()` performs bounded work. The runner owns warm-up and filmed step counts.
- `draw(frame)` synchronously returns the engine render sequence; it must not submit or yield.
  `frame` contains `index`, `seconds = index / fps`, `timestampUs`, `durationUs` and `view`.
- `prepare(frame)` may perform bounded derived work before draw.
- `reset()` restores the exact initial state, including placement, without warm-up. If omitted,
  the calibrated workload is disposed and built again on the same film engine.
- `dispose()` releases workload-owned resources and is idempotent. Engine teardown runs afterwards.
- Register partially constructed resources using `defer(release)` immediately after acquisition.
  Those releases run if construction fails; after successful construction `dispose()` owns them.

With W warm-up steps and s steps per frame, frame j shows state W + s × (j + 1), including
frame zero. Calibration changes state, so a complete reset or deterministic rebuild is essential.

Optional runner settings:

| Option | Default / behavior |
|---|---|
| `camera` | Optional `(seconds) => CameraView`; omitted, the workload controls its camera. |
| `source` | `'auto'`: operationally probe offscreen canvas, otherwise readback, before frame zero. |
| `hashes` | `false`; when true forces exact RGBA readback. Explicit canvas + hashes is refused. |
| `output` | Video options; `false` disables video (hash-only or measurement-only). |
| `signal` | AbortSignal; cancellation discards and rejects with AbortError. |
| `partial` | `'discard'`; `'keep-if-finalizable'` may return a finalized partial after a non-cancel failure. |
| `guard` | `{ limitMs?, stopMs? }`, may only lower the 1000/1500 ms defaults. |
| `yieldEveryMs` | 8; real task yields occur after capture. |
| `onProgress` | Phase, frame count, elapsed ms, ms/frame, ETA, encoded bytes and background status. |
| `onReport` | Outcome report once the opened session settles; observer exceptions do not change outcome. |
| `provenance` | Application-supplied metadata for function-valued workloads/cameras. |

Invalid options and session-opening failures can reject before `onReport` is installed. Always
handle the returned promise; do not use `onReport` as the sole error channel.

**Cancelling.** Aborting `signal` rejects with an AbortError and keeps nothing, from any phase:
building, calibrating, resetting, warming up, recording, or a frame held by the encoder.
- **Latency:** in Chrome on both GPUs it was under 7 ms. It runs longer only while your own
  `createWorkload`, `step` or `prepare` is still running: the runner checks between your calls and
  cannot interrupt them.
- **A cancel during the muxer's finalize** joins it, and the film completes. The first terminal
  request wins, and by then the runner has asked to finish. Dispose the result if you no longer want
  it. A cancel that comes before finalizing begins wins.

**Device loss.** A lost device ends the film before its next submission:
`film: GPU device lost at frame <j> (<reason>: <message>); partial file <kept|discarded>.`
- **The encoder can go with the device.** Chrome's hardware encoder lives in its GPU process, so
  when that process crashes, the encoder goes too and never answers again.
- **After a loss** the runner waits up to `LOST_ENCODER_GRACE_MS` (2 s) for the encoder, then gives
  it up. The film ends rather than hanging.
- **Partials are best-effort:** with `partial: 'keep-if-finalizable'`, a partial is kept only if the
  encoder still finalizes it.
- **Recovery is the app's:** request a new adapter and device, rebuild what used the old one (your
  preview included), and film again. A lost device is never resumed.

**Hidden and frozen pages.** Film time comes from the frame number, never the wall clock, so a film
hashes the same whatever happened to the page.
- **Hidden:** a hidden page films at its usual pace (measured: within 5% on both GPUs). Its timers
  fire at most once a second, so a workload's `step()` should not wait on timers: the guard would
  read the wait as a slow operation, and could stop the film.
- **Frozen:** a frozen page (Page Lifecycle's `freeze`) runs no task, and the film waits. An
  operation measured across a freeze is not judged against the guard's limits (they would be timing
  the freeze). `report.guard.unjudged` counts such operations.

## Video output and results

Output options are `container`, `codecs`, `bitrate`, `target`, `name`, `memoryBudgetBytes`.
MP4 defaults to H.264 only; WebM tries VP9 then AV1. Bitrate defaults to
`round(width × height × fps × 0.16)`, hardware preference to `no-preference`. Requested bitrate
does not guarantee achieved bitrate or quality; report both without changing parked Q9 policy.

**F2 only implements memory:** `target: 'memory'` is explicit; `'auto'` also resolves to memory
until F3. OPFS and picked-file requests refuse rather than pretending to stream. The default
256 MiB budget limits encoded payload, not total process memory. Prediction is checked before
encoding, runtime growth after adds and after final flush. Encoder/muxer allocations may occur
before a budget stop; this is not a hard process-memory ceiling. Spike S1 measured roughly 3×
payload peak for the memory target, in addition to rendering/encoder costs.

A successful result has `{ status, file, name, codec, frames, report, dispose }`:

- `status` is `'completed'` or `'partial'`; `file` is a Blob, or null when video is disabled.
- A partial must have successfully finalized; an encoder failure or failed finalization is not
  returned as playable output. Inspect `report.failure` for why a kept partial stopped.
- `dispose()` is idempotent (a no-op for memory). The app owns object URLs and their revocation.
- Reports include source selection, planned/accepted/encoded frames, encoder config, bytes and
  the achieved bitrate, timings, hashes when requested, state history, GPU errors, a lost device's
  reason, and secondary cleanup failures. They also give the guard's evidence: calibration, fencing,
  where fencing switched, the worst fenced step and draw, and the operations left unjudged.

## App-driven sessions

`createFilmSession({ film, output?, mediabunny?, signal? })` supplies the frames-in/file-out
piece without creating an engine. Await `session.add(videoFrame)` for backpressure. Ownership
transfers on invocation, including rejected adds: do not close the frame again.

Stamp frame j with `frameStamp(j, fps)`: timestamp is `round(j × 1e6 / fps)` microseconds,
duration is the next timestamp minus this one. Do not pass seconds to VideoFrame.

- **`finish()`** requires the planned video frames and finalizes.
- **`stop()`** keeps a finalized nonempty partial; `{ keepPartial: false }` discards.
- **`cancel()`** discards and waits for teardown.
  - It cancels the encoder first, so it does not wait for a frame the encoder holds.
  - Frames queued behind that one are closed without being encoded.
- **The first terminal operation wins.** Calls made after finalization has begun join its outcome;
  they do not interrupt an in-progress finalization.
- **`abandon(error)`** is the exception: it stops waiting on an encoder that will not answer, even
  inside a `finish()` under way. The wait in flight, and every one after it, fails with `error`, and
  the session ends `failed`. `recordFilm` calls it after a device loss; call it yourself if your own
  loop finds its encoder gone. What mediabunny still holds of the dead encoder is left to the
  garbage collector.
- **A premature `finish()` refusal** leaves the session open to add more frames or stop/cancel.

Use `try/finally` to cancel an unfinished session if your app-driven loop fails.

## Readback, snapshots and pure helpers

`FrameReadback(device, texture, width, height)` reads rgba8unorm textures with 256-byte GPU
row padding removed. `grab(index)` queues a copy; keep at most two pending grabs. Await
`take(out?)` or `takeVideoFrame(timestampUs, durationUs)`. The latter uses RGBX, ignoring alpha
for opaque video. `take()` without an output buffer reuses its storage: copy bytes if retaining
them across calls. Call `destroy()` in `finally`.

`snapshot({ device, createWorkload, width, height, seconds?, view?, signal? })` draws exactly
once, steps nothing, and returns `{ png, rgba, width, height, limitations }`. Odd PNG dimensions
are allowed. `seconds` is a draw time, not an instruction to advance simulation state.

Pure helpers include `planFilm`, `frameStamp`, `readCamera`, `cameraPath`, `fitAspect`,
`containAspect`, `bitrateFor`, `predictBytes`, `encodePNG`, `crc32` and `sha256` (Web Crypto).
`probeFilm` is cheap capability evidence, not an export guarantee: operational capture,
encoding, allocation and storage can still fail at runtime.

## Validation and limits

After `npm run build:lib`:
- **`npm run test:film`** (95 tests) runs the whole runner headless, on stand-ins for the device,
  WebCodecs and mediabunny. It includes garbage-collection checks that no film keeps its engine
  alive.
- **`npm run test:core`.**
- **`node test/film-browser-smoke.mjs`** films with real Node/Dawn readback, reset, hashes and
  snapshots, on the adapter that `NG_DAWN_FLAGS=adapter=AMD` or `adapter=NVIDIA` selects.
- **`npm run test:film:browser`** is the G suite. It runs `recordFilm` in Chrome on each GPU
  (`--gpu amd,nvidia`, `--only <cases>`, `--headed`), loading `dist/film.js` through an import map,
  and reads every film back with ffprobe and ffmpeg. It needs both on `PATH`. Its 12 cases check:
  - **G1:** the same film twice hashes the same, and nothing is left on the device;
  - **G2:** exact readback bytes, and decoded colours per codec and range;
  - **G3:** J frames, timestamps on their ticks, key frames, `moov` first, and frame k showing k;
  - **G5:** the guard;
  - **G6:** canvas against readback;
  - **G9:** cancellation, the main thread and the preview, hidden and frozen pages, `destroy()`,
    and a crashed GPU process.

  On 2026-10-25 it passed 305 checks on each GPU.

**Limits:**
- **The guard measures time, not the GPU.** It measures queue-completion wall time, not GPU
  timestamps, and cannot interrupt a submission already in flight.
- **The first submission is unqualified:** it compiles the pipelines, and nothing qualifies it as
  safe.
- **The guard re-measures an isolated stall once (Q10).** When a fenced operation passes the stop
  but its calibrated operation was under a quarter of the stop, the runner measures the next fenced
  operation before stopping, so a single desktop stall does not end the film.
- **Untested:** Firefox and Safari, a real TDR, OPFS and 4K memory behaviour (phase F3).