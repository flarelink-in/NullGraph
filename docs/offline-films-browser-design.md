# NullGraph — offline films in the browser (D2.4) — Technical Design

**Parent:** [`offline-films-design.md`](./offline-films-design.md) §17, phase D2.4 ("Browser export:
its own design").

**Asked by the user** (2026-10-24): design specs for offline rendering in browsers and the related
web stack, with build steps, ready before the Meta hackathon (kickoff Mon 2026-10-26, 21:30 IST).
MetaForge Studio (`D:\orion-realms\metaforge`) is the first user and the reference: it already
exports deterministic MP4s in about 150 lines (`src/engine/exporter.ts`), and its readiness notes
list what is missing (`docs/null-graph-readiness.md`, gap 1).

**Status:** draft 3, 2026-10-24 (reviewed against `src/core`, `tools/film`, `metaforge` and
locally installed `mediabunny@1.61.3`). **Spike S1 measured 2026-10-24: results at the end
("Spike S1 results"); the design body above is unchanged and its proposed changes wait for the
user.** No library implementation yet.
This revision makes the **export session lifecycle**, not a scene abstraction, the baseline;
adds ownership, cancellation, partial-output and storage policies; and separates measured facts
from spike hypotheses. Existing user decisions remain in §16; new recommendations are not
silently treated as decisions. The design is to be ready before kickoff on 2026-10-26;
implementation during the hackathon is optional and separately gated (§14).

---

## 1. Purpose

**Render an application's offline workload to a fixed-timeline video on the viewer's own GPU,
without a server or real-time pacing, when the browser supports the required capture, encoding
and storage path.** Each unit of GPU work must remain bounded; arbitrarily slow submissions
are not safe just because the final video has a fixed frame rate.

This is the browser half of D2. The Node film tool (`tools/film/`) stays as it is: it has ffmpeg,
tracks, checkpoints, resume and replay. The browser gets the part every site needs: the clock, the
camera path, frame capture, encoding and muxing, the guard, and a file at the end.

**For whom:**
1. **MetaForge Studio** (hackathon, 2026-10-26 to 11-05): its Record button, in three shapes, up
   to 4K, without the compositor pacing the export and without long 4K clips filling RAM.
2. **The test site's demos** (Herd & Wolves, the film scenes): a Record setting, as D2.4 planned.
3. **Any app on `null-graph`**: one entry point, `null-graph/film`.

### Integration facts (verified against this repository)

| # | Claim | Verified at | Consequence for this design |
|---|---|---|---|
| 1 | The engine runs headless or on an `OffscreenCanvas`, and adopts an existing `GPUDevice` without owning its lifecycle. | `src/core/NullGraph.ts:48–56`; `src/core/WebGPUContext.ts:38`, `:57`, `:65`, `:101–144`, `:261–263`, `:286–294` | `init(null, { device, width, height })` draws into `screenTexture` (`rgba8unorm`, `RENDER_ATTACHMENT \| COPY_SRC`). `init(offscreenCanvas, { device })` configures `getPreferredCanvasFormat()` with `alphaMode: 'premultiplied'`. Neither destroys `device` when `ownsDevice === false`. |
| 2 | `NullGraph` has `isDestroyed` but no `destroy()` method yet; `depthTexture` (33.2 MB at 4K), `screenTexture` (33.2 MB at 4K) and `cameraUniformBuffer` live on `WebGPUContext`. | `src/core/NullGraph.ts:35`; `src/core/WebGPUContext.ts:49–57`, `:113`, `:286–313`; `metaforge/docs/null-graph-readiness.md:56–60` | Without teardown on the film engine, each 4K export leaks 33–66 MB of VRAM on the shared `device` until GC. §12 adds `engine.destroy()` (destroys engine-owned targets, never an adopted `device`). |
| 3 | `src/core` touches no DOM. Texture and font helpers use worker-available APIs. | `src/core/TextureManager.ts:48–62` (`fetch`, `createImageBitmap`); `src/extras/text/FontAtlas.ts:253` (`OffscreenCanvas`) | The film engine and `null-graph/film` can run on the main thread or in a Dedicated Worker. |
| 4 | MetaForge's exporter (`~155` lines) proves the WebCodecs + `mediabunny` path on this machine, and isolates the two faults to fix. | `metaforge/src/engine/exporter.ts:62–141`; `metaforge/docs/null-graph-readiness.md:21–23`, `:41`, `:48–53` | Lockstep `t = f / fps`, H.264 → VP9 → AV1, `new Quality({ bitrate })`. Fault 1: it resizes and draws into the *visible* canvas (`viewport.enterExport`), competing with the compositor. Fault 2: `BufferTarget` holds the whole MP4 in RAM. |
| 5 | `mediabunny@1.61.3` `VideoSampleSource.add` takes a `VideoSample` (timestamps in **seconds**), not a bare `VideoFrame` (timestamps in **µs**), and does **not** close the sample (`closeSample = false`). | `mediabunny` `media-source.js:1163–1168` vs `CanvasSource.add` at `:1204–1213`; `sample.js:177`, `:296–302`, `:363` | `createVideoSink` wraps frames in `VideoSample` and calls `sample.close()` in `finally` after `await source.add(sample)` so WebCodecs `VideoFrame` handles never leak. |
| 6 | `mediabunny@1.61.3` `Mp4OutputFormat` supports `fastStart: 'reserve'` when every track declares `maximumPacketCount`, and `StreamTarget` supports `chunked: true` (16 MiB chunks, up to 2 retained chunks in its chunk cache). | `mediabunny` `isobmff-muxer.js:143–152`; `target.js:153–154`, `:198–199`, `:260–310` | Known packet-count bounds permit front-reserved MP4 metadata on seekable storage. The 32 MiB chunk cache is **not** a bound on total export memory; mux metadata, temporary copies, encoder resources and GPU allocations are additional. |
| 7 | The pure planner and camera path in `tools/film/` have no Node imports; `FrameCapture` uses `Buffer` in one method; optional peer dependencies in `src/extras/` use structural types with no top-level import. | `tools/film/manifest.mjs:81`; `tools/film/camera.mjs:32`; `tools/film/capture.mjs:6`, `:41–44`; `src/extras/inference/OrtRuntime.ts:7–16` | `planFilm` and `cameraPath` move to `src/extras/film/` as the single source of truth. `mediabunny` is loaded via dynamic `await import('mediabunny')` so Node tools and `tsc` never require it when importing pure helpers. |
| 8 | The Node guard uses 1,000/1,500 ms thresholds; Chrome on this machine defaults to the AMD GCN-5 iGPU unless configured for the GTX 1650. Windows' default TDR timeout concerns task completion/preemption, not a universal WebGPU submission duration. | `tools/film/guard.mjs:6–7`; `docs/work/gotchas.md:30–36`; `metaforge/docs/null-graph-readiness.md:77–81`; Microsoft TDR reference (§17) | Reuse conservative thresholds as best-effort policy, not a reset-prevention guarantee. Device loss also affects a preview sharing that device; the app must recover it. |

### Non-goals

- **Tracks, checkpoints, replay and resume in the browser.** The core's pack and unpack stages
  would work there unchanged (offline-films §17), but storing and seeking records is a second
  project. Deferred to D2.5 (§15).
- **Real-time capture** (`MediaRecorder`, `captureStream`): it records the stutter.
- **Audio.** Not in the first slice. mediabunny can mux an audio track later.
- **The same bytes on different GPUs, browsers or encoders.** Frames are exact per machine and
  browser; encoded files may differ even there (hardware encoders are not required to be
  deterministic).
- **A server-side renderer.** Everything happens on the viewer's GPU.
- **A fallback for browsers without WebCodecs or WebGPU.** Detected and declined with a reason,
  never polyfilled (memory: modernization philosophy).

### Principles

The parent's principles hold: time from the frame number, never the clock; no engine hook; show
the cost before spending it; refuse unsupported or unsafe configurations with a reason. Three
more for the browser:

- **The export never shares a canvas with the preview.** It draws into its own target at its own
  size. The live view is untouched apart from being paused.
- **Detect, then offer.** Every capability (encoder, codec at that size, canvas frames, OPFS, save
  picker, workers) is probed; the UI offers only what works and says why something is missing.
- **Stream by default.** A file is written as it is encoded whenever the browser can; memory is
  an explicitly budgeted fallback, never an unlimited fallback after storage failure.
- **Lifecycle before scene shape.** The session owns transitions and resources it creates.
  Applications may supply frame callbacks, simulations, replay, procedural rendering or a
  scene-factory adapter. No scene graph, seed scheme or camera model is required by the sink.

## 2. The definition, restated for the browser

Unchanged from offline-films §2:

> I_j = R(S_((j+1)·s), C(τ_j), Q),  τ_j = j / r,  j = 0 … J−1

with s = 1/(r·dt) steps a frame, W warm-up steps before frame 0, and the invariant that I_j depends
only on the film's inputs and the machine. **"The machine" now includes the browser and its
version**, because Chrome's Dawn and Firefox's wgpu compile WGSL differently.

**The encoded film** is E = enc(I_0 … I_(J−1)), each I_j presented with timestamp τ_j and duration
1/r. Properties P1 (determinism) and P4 (capture fidelity) apply to I_j, measured from a readback;
E is only required to decode to J frames at r fps.

## 3. The stack

| Layer | Web API | Used for | Detected by |
|---|---|---|---|
| GPU | WebGPU | the scene, unchanged | `navigator.gpu`, `requestAdapter()` |
| Target | `OffscreenCanvas` + `GPUCanvasContext`, or the engine's headless `screenTexture` | where the frame is drawn | `typeof OffscreenCanvas === 'function'`, `new OffscreenCanvas(2, 2).getContext('webgpu')` |
| Frame | `VideoFrame` wrapped in mediabunny `VideoSample` | the bridge from GPU to encoder | `typeof VideoFrame === 'function'` |
| Encode | WebCodecs `VideoEncoder` | H.264 / VP9 / AV1 | `VideoEncoder.isConfigSupported` at the exact size (`getEncodableVideoCodecs`) |
| Mux | mediabunny (`Output`, `Mp4OutputFormat`, `WebMOutputFormat`, `VideoSampleSource`) | the container | dynamic `await import('mediabunny')` (§11) |
| Store | OPFS (`navigator.storage.getDirectory()`, `createWritable()`), File System Access (`showSaveFilePicker`), or memory | where the file goes | feature checks; `navigator.storage.estimate()` for room |
| PNG | `CompressionStream('deflate')` + CRC32 | exact unpremultiplied RGBA8 PNG frames and snapshots | `typeof CompressionStream === 'function'` |
| Thread | Dedicated `Worker` | export off the main thread (phase F4) | `typeof Worker === 'function' && typeof OffscreenCanvas === 'function'` |
| Hash | `crypto.subtle.digest('SHA-256')` | per-frame hashes for P1 tests | always present in secure contexts |
| Share | `navigator.share({ files })` | handing the file on, in the app | `navigator.canShare` (app's job, not the library's) |

## 4. The model

### 4.1 The clock (pure, shared with Node)

`planFilm` moves into the library as TypeScript, unchanged in behaviour: frames J, steps a frame s,
warm-up steps W, refusals for a non-whole number of steps or frames. The Node tool imports it from
`dist/film.js` instead of its own copy (one source of truth).

The loop matches the parent's formal definition (`I_j = R(S_((j+1)·s), C(τ_j), Q)` in §2 and
`tools/film/film.mjs:205, 229`):

```
workload starts at its declared initial state; W warm-up steps (not filmed)
for j in 0 … J−1:
    s steps                              (each step its own submission; frame 0 is drawn at S_s)
    camera ← C(j / r)
    await workload.prepare?.(frame)      (optional bounded derived reconstruction)
    sequence ← workload.draw(frame)      (synchronous; must not submit or yield)
    engine.render(sequence)              (its own submission)
    sample ← capture(frame)              (synchronous, before any yield or draw fence)
    await sink.add(sample)               (backpressure; sample closed in finally)
    fence and yield as required          (never inserted between render and capture)
```

- **Initial state is explicit:** initialization/placement defines the pre-warm-up state; it is
  not a filmed step. The runner then owns exactly W warm-up steps and s steps for every frame,
  including frame 0. MetaForge's current reset includes placement and warm-up; the Node Kerr
  factory also runs a placement dispatch before runner steps. Adapters must preserve those
  semantics deliberately. Tests compare step counts, simulation time and first-frame bytes
  before claiming Node/browser equivalence.
- **Not all workloads simulate:** procedural rendering and replay may expose a no-op step with
  a declared dt and use frame index/time in `prepare` and `draw`. No scene, seed or camera is
  required. §2's simulation equation applies when there is simulation state.

### 4.2 The camera (pure, shared with Node)

`cameraPath(keyframes)` moves into the library: orbit keyframes `{ t, target, distance, azimuth,
polar, fov? }`, Catmull-Rom slopes, log distance, eased ends.
- **`fov` compatibility:** Node film manifests omit `fov` (`tools/film/manifest.mjs:22`), whereas
  MetaForge keyframes include `fov` in degrees (`metaforge/src/engine/camera-path.ts:16`).
  `readCamera` and `cameraPath` accept optional `fov?: number` (finite degrees `> 0` and `< 180`),
  interpolating it with the same cubic Hermite when present and omitting it when absent so Node's U1
  and U4 tests pass unchanged.
- A film may instead pass any `(t: number) => CameraView`, or omit `camera` so the scene's own view
  holds.

**Aspect reframing** (MetaForge's `fitView`, `metaforge/src/engine/camera-path.ts:55`: for aspects
narrower than `designAspect = 16/9`, keep `keepWidth = 0.55` of the 16:9 width by widening vertical
`fov` up to `maxFov = 60°`, then pulling `distance` back; `containAspect` widens `fov` to fit a
guide rectangle on a narrower preview canvas) moves in as `fitAspect(view, aspect, options?)` and
`containAspect(view, frameAspect, canvasAspect, options?)` (defaulting `view.fov` to `45°` when
omitted). Optional to use.

### 4.3 Frame sources

Two, chosen by probe (§5), never both in one film:

**A. Canvas frames (provisional video default, Q1).** The film engine draws into its own
`OffscreenCanvas` (never the visible one). Immediately after render, construct a raw
`VideoFrame(offscreenCanvas, { timestamp: T(j), duration: D(j) })` and wrap it in `VideoSample`.
This avoids the pinned canvas constructor's seconds-to-microseconds truncation round-trip
(`sample.js:363`). A GPU-backed path may avoid CPU readback, but zero-copy is browser-dependent
and must be measured, not guaranteed.

- *Capture ordering matters.* Canvas textures have automatic expiry/presentation scheduling.
  Render and sample construction are synchronous and adjacent: no await, callback or GPU fence
  between them. Not every await crosses a task boundary, but correctness must not depend on that.
- *Timestamp units:* WebCodecs `VideoFrame` takes **microseconds** (`µs`); mediabunny `VideoSample`
  takes **seconds**. Passing seconds directly as integer microseconds produces near-zero times.
- *Quantization:* define `T(j) = Math.round(j * 1e6 / fps)` and `D(j) = T(j + 1) - T(j)`.
  Pass `T(j)`/`D(j)` to raw `VideoFrame`; `VideoSample` reads seconds from that frame.
  Use the same values for canvas and readback; do not independently truncate every duration.
  Container timebases may rescale them; tests allow one container timebase tick.
- *Whether an unattached `OffscreenCanvas` on the main thread is paced by anything* is spike S1's
  question (§14). If it is, the canvas source goes to the worker (F4) and the main-thread variant is
  dropped.

**B. Readback frames.** The film engine is headless (`init(null, { device, width, height })`) and
draws into `screenTexture` (`rgba8unorm`). Each frame is copied into one of three staging buffers
and mapped, with one frame in flight so frame `j − 1` maps and hashes/encodes while frame `j`'s
steps run (`tools/film/film.mjs:231`). `FrameReadback.take(out?: Uint8Array)` writes tightly packed
`RGBA8` into a caller-supplied or pooled buffer (`design-principles.md` §5: zero per-frame
allocation of 33 MB buffers at 4K).
- For the video sink, the readback bytes become a `VideoFrame` with `format: 'RGBX'` (ignoring the
  alpha byte so scenes whose fragment shaders leave `a < 1` in `screenTexture`, such as additive
  particle glow, encode opaque just as ffmpeg's `yuv420p` path does), wrapped in
  `new VideoSample(videoFrame)` using the quantized raw-frame timestamps. Passing a `VideoFrame` into
  `VideoSample` also avoids `mediabunny`'s internal `toUint8Array(data).slice()` copy on raw byte
  arrays (`sample.js:298`).
- Exact bytes, so it is the source for **hashes (P1), PNG frames, and the tests**.
- Costs a CPU round trip: 8.3 MB a frame at 1080p, 33.2 MB at 4K. Spike S1 measures it against A.

| | Canvas frames | Readback frames |
|---|---|---|
| CPU copy | none (browser-dependent) | one per frame (pipelined, pooled `Uint8Array`) |
| Exact bytes available | no | yes |
| Needs | `OffscreenCanvas` WebGPU context | nothing beyond WebGPU |
| Use | video export | hashes, PNGs, tests; video where A is unavailable |

### 4.4 Sinks

Each sink takes `add(frame, timestamp, duration)` and `close()`; the loop awaits `add`, so a slow
encoder or disk paces the GPU instead of filling memory.

| Sink | Takes | Writes |
|---|---|---|
| `video` | `VideoSample` (wrapping `VideoFrame`) | mediabunny `VideoSampleSource` → `Output` (MP4 or WebM) → a target (§4.5) |
| `hash` | readback `Uint8Array` | `crypto.subtle.digest('SHA-256', bytes)` per frame into the report |
| `png` | readback `Uint8Array` | one PNG per frame (`CompressionStream('deflate')` + CRC32, exact RGBA8) into an OPFS folder |
| `none` | anything | nothing: a run that only measures |

**Lifecycle rule for `VideoSampleSource`:** Unlike `CanvasSource.add` (which passes
`closeSample = true` internally, `media-source.js:1212`), `VideoSampleSource.add(sample)` passes
`closeSample = false` (`media-source.js:1167`). `createVideoSink`'s `add()` **must** call
`sample.close()` in a `try / finally` block after `await source.add(sample)` resolves; otherwise
WebCodecs `VideoFrame` handles leak and stall the encoder after ~16 frames.

**Encoding settings:**
- **Codec:** MP4/H.264 (`['avc']`) is the proposed default. Apps may opt into fallback lists,
  intersected with container support and probed at the actual size, fps, quality/bitrate and
  hardware preference. Surface the chosen codec: VP9/AV1 in MP4 are not a promise of H.264-like
  playback compatibility. WebM defaults to `['vp9', 'av1']`. Runtime encoder failure remains possible.
- **Bitrate:** MetaForge's measured rule, `Math.round(width × height × fps × 0.16)` (≈ 20 Mbit/s at
  1080p60, 80 at 4K60), overridable. Passed as `new Quality({ bitrate })`, not a bare number
  (MetaForge's gotcha: a number reads as a 0–1 quality level).
- **Keyframes** every 2 s (`keyFrameInterval: 2`). **Even** width and height, refused otherwise (as
  in Node).
- **Colour:** `alpha: 'discard'`, `latencyMode: 'quality'`. Declare input RGB colour space/range
  consistently with the actual render target (sRGB versus linear bytes must be established by
  tests), not by relabelling RGB input as BT.709 YUV. Target SDR BT.709 encoded output where the
  pipeline supports it; inspect decoded metadata and levels. The pinned mediabunny encoding
  options do not expose an arbitrary `colorSpace` setting. Canvas premultiplication/additive
  alpha behavior is a release gate, not assumed equivalent to raw RGBX readback.
- **Hardware preference:** `no-preference`. `onEncoderConfig` records the accepted
  `VideoEncoderConfig` (including `codec` and `hardwareAcceleration`) into the report.

### 4.5 Targets: where the file goes

| Target | When | How |
|---|---|---|
| **Memory** | explicitly selected, or within the fallback budget when OPFS is unavailable (§16 Q3) | `BufferTarget`, `Mp4OutputFormat({ fastStart: 'in-memory' })`; a `Blob` at the end |
| **OPFS** | recommended automatic target when writable | `StreamTarget(writable, { chunked: true })` over `FileSystemWritableFileStream`, `Mp4OutputFormat({ fastStart: 'reserve' })` with `maximumPacketCount: plan.frames`; a `File` at the end |
| **A file the user picked** | Chromium, when the app passes a `FileSystemFileHandle` | `handle.createWritable()` → `StreamTarget(writable, { chunked: true })`, `fastStart: 'reserve'`; written straight to the user's disk |

- **Why `fastStart: 'reserve'` and `chunked: true` on streams (`mediabunny` `isobmff-muxer.js:143`, `target.js:198`):**
  Because an offline film's frame count `J = plan.frames` is fixed before frame 0, passing
  `maximumPacketCount: plan.frames` to `output.addVideoTrack` lets `Mp4OutputFormat({ fastStart: 'reserve' })`
   reserve sufficient front metadata space for that packet-count bound and stream `mdat` in
   16 MiB chunks. Finalization fills the reserved metadata region; it does not overwrite `ftyp`
   at offset zero. This needs a seekable target and successful finalization. The retained chunk
   cache is at most 32 MiB in this version, **not total export RAM**: temporary copies, packet
   metadata (which may grow with frame count), encoder resources, readback pools and GPU targets
   are additional. Front metadata improves progressive loading, not codec/player compatibility.
- **Storage check:** Before starting, the predicted byte size (`Math.ceil(bitrate / 8 × seconds × 1.15)`)
  is compared with `navigator.storage.estimate()` (`quota - usage`) for OPFS, as the Node tool
   checks free disk. Refused with both numbers if it does not fit. This is an estimate, not a
   reservation or bitrate ceiling: quota, eviction, concurrent writes and actual encoder output
   can change the outcome. Handle write/finalize failures even after preflight succeeds.
- **Selection policy:** `auto` prefers writable OPFS. Without it, memory is offered only when
  predicted output fits `memoryBudgetBytes` (proposed default 256 MiB, Q3), with a runtime byte
  stop before committing further growth. This budgets encoded payload, not total process memory.
  Budget exhaustion aborts; it cannot guarantee avoiding all transient allocations. Never switch
  targets silently mid-export. Explicit memory selection obeys the same configured budget.
- **OPFS identity and retention:** use `null-graph-films/<exportId>/` with a random unique ID,
  a payload and a status/report manifest. The download name is metadata, not an internal path.
  Mark complete only after mux finalization and stream commit; renaming `.part` is optional.
  Failure cleanup removes only this session's uncommitted files. `dispose()` is idempotent and
  deletes only this result's backing directory; completed results are never swept by another
  export. The app retains backing storage through download/share consumption and revokes object
  URLs itself; a download click is not a reliable completion signal.
- **Orphans and concurrent tabs:** `probeFilm()` never sweeps files. An explicit cleanup API may
  remove incomplete exports only after acquiring an exclusive per-export Web Lock also held by
  every writer. Age/heartbeat alone is insufficient: background tabs can be suspended. If locks
  are unavailable, automatic cross-tab cleanup is disabled. Completed results require explicit
  app/user deletion, even if old. Orphan cleanup does not implement resume.
- **Picked files:** the app obtains the handle in a user gesture, before long async work; the
  library never opens the picker. Permission can still be denied at write time. On discard,
  abort rather than close/commit the unfinalized stream. A newly created empty directory entry
  may remain; do not delete user-owned files. A playable partial is committed only under §4.9.

### 4.6 Workload adapter and resource ownership

The baseline is an export session (§4.9), not a scene. `recordFilm` is a managed convenience
runner: it creates an isolated engine/target on the borrowed app device and calls `createWorkload`.
A workload may be simulation, replay, a procedural renderer or a scene adapter. It does not borrow
mutable preview state. Apps with their own loop use the session/sink pieces directly.

- The live preview keeps its canvas, size and quality ladder; it is paused by the app (one call)
  so the two do not contend for the GPU.
- Reproducibility requires the adapter to restore all relevant GPU and CPU state, RNG state,
  counters and temporal effects. A fresh engine alone does not establish determinism.
- Cost: the scene's GPU memory twice during the export, plus the film engine's screen/depth targets
  (33–66 MB at 4K). Scene construction and calibration run inside `'out-of-memory'` and
  `'validation'` error scopes. The session registers acquired resources immediately and unwinds
  them in reverse order, including partial construction. Workload disposal and engine teardown
  are both attempted even if one fails. Scopes/listeners on a shared device are device-wide:
  serialize exports per device and require the app to stop competing work during measured/scoped
  operations. Do not claim errors from unrelated device users can always be attributed correctly.

Proposed managed adapter contract (types finalized in F1; no dependency on a scene class):

```ts
type CreateFilmWorkload = (ctx: {
    engine: NullGraph;
    width: number;
    height: number;
    signal: AbortSignal;
    defer(cleanup: () => void | Promise<void>): void; // register before another fallible operation
}) => Promise<{
    dt: number;                 // fixed step; no-op steps allowed for stateless rendering
    reset?(): void | Promise<void>; // exact initial state, including placement, without W warm-up
    step(): void | Promise<void>;
    prepare?(frame: FilmFrame): void | Promise<void>;
    draw(frame: FilmFrame): RenderSequence; // synchronous, no submission; runner renders/captures
    dispose(): void | Promise<void>; // workload-owned resources only; idempotent
    info?: Record<string, unknown>;
}>;
type FilmFrame = {
    index: number;
    seconds: number;
    timestampUs: number;
    durationUs: number;
    view: CameraView | null;
};
```

An optional `fromFilmScene(factory, options)` adapter maps Node-style `prepareSurface`, `view`,
`draw` and `destroy` to this contract. Its browser `film.track()` records ignored declarations;
this does not make Node imports/assets automatically browser-safe. MetaForge pins quality to 1
in its adapter; the generic session never changes application quality or installs a seed policy.
Registered partial-build cleanups run if construction fails; once construction succeeds their
ownership transfers to the workload disposer, avoiding duplicate disposal.

**Ownership ledger:** the app owns the borrowed device, preview, file handle and object URLs;
the session owns its engine, capture pool, encoder/muxer, listeners and uncommitted storage;
the workload owns only resources it creates (borrowed assets are never destroyed). Sample
ownership transfers to `sink.add` on invocation, which closes it on success or rejection. If
sample wrapping throws, the creator closes the raw frame. No sample or mapped buffer survives
terminal cleanup. GPU resource release is explicit, but actual driver reclamation is not an
immediate or precisely measurable VRAM guarantee.

### 4.7 The guard

**Best-effort risk reduction, not reset prevention.** Reject work predicted above 1,000 ms and
stop submitting after a measured operation exceeds 1,500 ms. Measurements are queue-completion
wall milliseconds, not GPU timestamps. A watchdog may fire before a promise resolves; a timer
cannot interrupt submitted GPU work. These thresholds are conservative policy, not universal
browser/OS safety boundaries.

- **Drain before calibrating:** The live preview may have one `requestAnimationFrame` submission
  still in flight when paused. `calibrateFilm` first awaits `device.queue.onSubmittedWorkDone()` so
  live work is never charged to the film.
- **Initialization is work too:** require bounded factory/placement, reset, warm-up, step,
  reconstruction and draw operations. One callback is not necessarily one GPU submission; an
  adapter must declare its submission boundaries or accept conservative operation-level timing.
- **Calibration:** no unmeasured full-size warm-up batch before safety evidence. Where an adapter
  supports reduced workload probes, measure progressively (Node guard precedent), rejecting
  predicted unsafe increases. Record workload parameters and prediction provenance. Arbitrary
  shaders without scalable probes have **unverified first-submission risk**, disclosed before
  starting; do not fabricate a prediction or silently call that safe. Measure individual
  operations before optional bounded throughput batches. Calibration may take seconds or longer.
- **Restore after calibration:** exact `reset()` or destroy/rebuild the workload. Never count
  calibration steps as W warm-up. Rebuild/restore must itself satisfy bounded-work requirements.
- **While running (periodic vs per-submission fencing):**
  - When calibration measured `msPerFrame ≤ 150 ms`, the loop awaits `device.queue.onSubmittedWorkDone()`
    every 8 frames (`metaforge/src/engine/exporter.ts:125`) so the GPU queue stays fed without
    stalling on every frame.
  - **Why `> 150 ms` switches to per-submission fencing:** 8 frames at 200 ms/frame take 1,600 ms
    for the batch even though each operation is only 200 ms (inside the 1,500 ms policy stop).
    Comparing an 8-frame batch against 1,500 ms would falsely abort 4K exports! Therefore, when
    calibration is `> 150 ms` per frame — or on the frame
    immediately following any 8-frame batch whose wall time exceeded `1200 ms` — each step and
    draw operation is fenced and timed against `STEP_STOP_MS = 1500 ms`. Capture the draw's sample
    **before** awaiting its fence. Batch timing is a congestion signal, not evidence that any
    individual submission crossed the stop limit. Report worst-operation measurements only
    when actually measured; periodic mode cannot promise per-submission observation.
- **Uncaptured errors and device lost:** An `uncapturederror` listener is attached to `device` for
  the duration of `recordFilm` (`gotchas.md:28`) alongside `device.lost`. Either stops the film.
  Stop submitting immediately. Partial preservation follows §4.9 and may fail; the app cannot
  simply resume a preview on a lost shared device. Remove the event listener during teardown;
  mark the `device.lost` continuation inactive after the session terminates.
- **Costs are not cached** across sessions in this slice (no reliable per-adapter key in a
  browser). Record calibration wall time without promising a fixed duration.

### 4.8 The report

Every film returns a report, also written beside an OPFS file as `<name>.json`:
inputs (film settings, scene settings and seed, camera keyframes), the browser (`userAgent`), the
adapter (`device.adapterInfo`: vendor, architecture, description), the accepted `VideoEncoderConfig`
and codec string, the frame source, timings (calibration, warm-up, per-frame mean and worst
single-submission ms for steps and draws, and encode/mux wait ms), frame hashes when the hash sink
ran, ignored tracks, GPU errors, and whether the film is complete. Also record export ID, state
transitions, requested/submitted/encoded/finalized frame counts, last committed frame, target and
fallback reasons, timing quantization, guard evidence/limitations, approximate memory categories,
partial preservation status, write/commit status and cleanup errors. Configuration callbacks
describe the configuration requested/used by the library, not proof of hardware encoder selection.
On rejection, `onReport(report)` receives the terminal report; report callback errors are isolated
from export success/failure. Persisting the report itself can fail and must not hide the primary
error. Function-valued cameras/workloads need app-provided provenance; they cannot be JSON-serialized.

### 4.9 Export session lifecycle (baseline contract)

The session coordinates source, sink and target independently of workload shape. The managed
runner drives it, but an app may drive `createFilmSession` with its own frame producer. Its
`add(sample)` takes ownership, requires monotonically increasing planned timestamps, enforces
frame/byte limits and serializes writes; it rejects after stopping. `finish()` requires J accepted
samples. `stop({ keepPartial: true })` attempts an explicitly incomplete finalized result.
`cancel()` discards. Terminal calls are idempotent; the first terminal request wins, except that
an unrecoverable error overrides success. Teardown always runs before the operation settles.

| State | Work and exit invariant |
|---|---|
| `probing` | Validate options and capability evidence. No scene or preview mutation; no orphan sweep. |
| `initializing` | App pauses preview; serialize on borrowed device, allocate target/engine/workload with immediate cleanup registration. No unguarded full-size calibration batch. |
| `calibrating` | Gather bounded evidence; publish guard limitations and ETA confidence. |
| `resetting` | Exact restore or clean rebuild, then precisely W warm-up steps. |
| `recording` | Prepare, synchronous render/capture, serialized sink backpressure; bounded staging queue. |
| `stopping` | Stop accepting new frames; settle ownership of outstanding samples/readbacks. |
| `finalizing` | Flush encoder, finalize muxer, commit writable stream, then mark output/report complete or partial. |
| `disposing` | Remove listeners, close/unmap capture buffers, dispose workload and engine, release locks; attempt all cleanup. |
| `completed` / `partial` / `cancelled` / `failed` | Terminal report published; only successfully committed results retain backing storage. |

**Policies:**
- Default cancellation/failure policy is **discard**: cancel encoder/muxer and abort uncommitted
  writes. Cancel rejects with `AbortError`; other failures retain their original cause. Cleanup
  errors are secondary report entries, never replacements for the primary error.
- Partial output is **opt-in**, not automatic recovery. `partial: 'keep-if-finalizable'` permits
  recovery on a non-user failure; user cancellation still discards. Apps wanting “Stop and save”
  use explicit session `stop({ keepPartial: true })`. Only successfully flushed, mux-finalized
  and committed output is returned as `partial`, with `complete: false` and actual frame count.
  Zero-frame or unfinalizable output is discarded; device loss makes recovery best-effort.
- On a recoverable failure with opt-in preservation, `recordFilm` resolves a partial result with
  `report.failure` set; otherwise it rejects and publishes the report. No unfinalized MP4 is
  described as playable. Callback errors from workload code fail the export; progress/report
  observers are isolated and reported without invalidating an already committed file.
- Check cancellation before/after every await and between warm-up/step operations. Submitted GPU
  work cannot be cancelled. Yield an actual task after a configurable CPU budget (initial target
  8 ms), after sample capture, not only a chain of resolved promises. Measure cancellation latency
  rather than guaranteeing a bound for arbitrary application callbacks.
- Visibility/background transitions may throttle or suspend progress. Default policy continues
  without advancing film time from wall time; surface a stalled/background status. Do not promise
  survival of tab discard, sleep, navigation or storage eviction. Resume remains deferred.
- The app restores preview in its own `finally` after session cleanup; if the device was lost it
  rebuilds/reports recovery instead. The library never starts/stops an arbitrary application's UI.
- Source fallback is selected before frame 0. Failed operational canvas validation can select
  readback with an explicit reason; failures mid-film stop rather than silently changing paths.
  Retrying needs a new session and deterministic reset. Target fallback follows §4.5.

## 5. Capability probe

```ts
const support = await probeFilm({ device, width: 3840, height: 2160, fps: 60 });
// {
//   ok: true,                         // configuration-supported, not an end-to-end guarantee
//   codecs: ['avc', 'vp9'],           // encodable at this size & bitrate, in preference order
//   sources: { canvas: { api: true, operational: 'untested' }, readback: { api: true } },
//   targets: { memory: true, opfs: { api: true, writable: 'untested' }, picker: { api: true } },
//   worker: true,                      // F4
//   storage: { quota, usage, available }, // bytes
//   reasons: [],                       // plain words for anything false
// }
```

Distinguish API presence, supported encoder configuration, operational capture/storage smoke
tests, and end-to-end validation. A cheap probe draws no frames, deletes no files and requests no
picker permission. An explicit operational probe creates only uniquely named disposable test
resources and removes its own files. Re-probe when size, fps, bitrate, container or codec settings
change. Support checks use the same complete configuration as the sink, intersected with the
container's supported codecs; a successful support check does not guarantee encode success or
permission/quota. Only claim supported alternative sizes if they were actually probed. Worker
API presence is not proof that the adapter/module and encoder work in that worker.

## 6. API

**Entry point `null-graph/film`** (`src/extras/film/index.ts`), worker-safe, with no DOM imports.

```ts
import { recordFilm, probeFilm, planFilm, cameraPath, fitAspect, containAspect } from 'null-graph/film';

const result = await recordFilm({
    device,                                   // the app's device; the film engine adopts it
    createWorkload,                           // app callback with the lifecycle contract (§4.6)
    film: { width: 1920, height: 1080, fps: 60, seconds: 12, warmupSeconds: 2 },
    camera: cameraPath(keyframes),            // or (t) => CameraView, or omitted: the scene's view holds
    source: 'auto',                           // 'canvas' | 'readback' | 'auto' (canvas when probed true)
    output: {
        container: 'mp4',                     // 'mp4' | 'webm'
        codecs: ['avc'],                      // explicit opt-in list for VP9/AV1 fallback
        bitrate: undefined,                   // default: Math.round(w × h × fps × 0.16)
        target: 'auto',                       // 'memory' | 'opfs' | { file: FileSystemFileHandle } | 'auto'
        name: 'metaforge-kerr-1080p.mp4',
        memoryBudgetBytes: 256 * 1024 * 1024, // proposed fallback payload budget, Q3
    },
    hashes: false,                            // true forces the readback source
    signal,                                   // AbortSignal: cancel
    partial: 'discard',                        // or 'keep-if-finalizable' on non-user failure
    onProgress: p => ui.update(p),            // { phase, frame, frames, elapsedMs, msPerFrame, etaMs, bytes }
    onReport: report => ui.report(report),     // called on success, cancellation and failure
});
// result: { status: 'completed' | 'partial', file: Blob | File, codec, frames, report,
//           dispose(): Promise<void> }
```

**Smaller pieces, exported for apps that want their own loop** (MetaForge's photo, the test site's
custom UI):

| Export | Is |
|---|---|
| `planFilm(film, { dt })`, `readCamera(keyframes)` | the clock's plan and keyframe validator; pure |
| `cameraPath(keyframes)`, `fitAspect(view, aspect, options?)`, `containAspect(view, frameAspect, canvasAspect, options?)` | cameras; pure |
| `bitrateFor(width, height, fps)`, `predictBytes(bitrate, seconds)` | rate and size rules; pure |
| `probeFilm(options)` | §5 |
| `createFilmSession(options)` | baseline source/sink/target lifecycle: `add`, `finish`, `stop`, `cancel`; no scene required (§4.9) |
| `fromFilmScene(factory, options)` | optional Node-style scene compatibility adapter, not the baseline |
| `calibrateFilm(engine, workload, size, options?)` | bounded evidence and queue-completion wall measurements, including measurement coverage and unverified risks |
| `FrameReadback` | source B as a class (`grab(frame)`, `take(out?)`, `destroy()`) |
| `createVideoSink(options)` | the video sink and its target; `add(sample, encodeOptions?)`, `close()`, `cancel()` |
| `encodePNG(rgba, width, height)` | browser/Worker/Node RGBA8 → PNG (`CompressionStream('deflate')` + CRC32) |
| `snapshot(options & { createWorkload })` | isolated one-frame session; exact PNG uses readback + `encodePNG`; no preview resize or implicit simulation advancement |

**Phase F4 adds** `recordFilmInWorker(options & { workloadModule: URL })`: a dedicated worker
imports the adapter itself (functions do not cross `postMessage`). Only cloneable configuration
crosses the boundary; the borrowed main-thread device is replaced by a worker-owned device.

## 7. Threads

**F1–F3: the main thread, a separate target.** The film engine and its `OffscreenCanvas` (or
headless texture) live on the main thread beside the app. The visible canvas is not involved, so
compositor pacing is a spike hypothesis, not yet a guarantee. The loop explicitly yields tasks
after capture (§4.9); GPU/encoder awaits alone do not prove UI responsiveness. Long synchronous
application callbacks can still block the main thread and are a reason to choose F4.

**F4: a worker.** Moves the film engine, its workload adapter and encoder into a dedicated worker. Gains:
the main thread stays free for the UI even when the scene's own JavaScript is heavy (an ONNX brain,
a big `step()`). Costs: the scene must be a module the worker can import, and the worker needs its
own device unless the app's device can be shared (it cannot: a `GPUDevice` does not transfer). So
the worker requests its own adapter and device (`powerPreference: 'high-performance'`), and
explicitly destroys both the film engine and its worker `GPUDevice` in `finally`; the report names
both adapters, and the app pauses the preview as before. Taken only if S1 or F2 shows the
main-thread loop stuttering the UI.

## 8. Errors

Named refusals (`design-principles.md` §2), each in plain words naming the offender, the number
that caused it, and the way out:

| Condition | Message pattern |
|---|---|
| No WebGPU or device | `film: WebGPU is not available in this browser.` |
| `mediabunny` missing when video sink requested | `film: mediabunny is not installed; install mediabunny to encode video in the browser.` |
| No WebCodecs or no encoder at this size | `film: no WebCodecs encoder for <W>×<H> (<codecs>); supported sizes on this browser: <ladder>.` |
| Odd width or height | `film: width and height must be even for video encoding (yuv420p); got <W>×<H>.` |
| Non-whole steps or frames | `film: one frame at <fps> fps (<1/fps> s) is not a whole number of steps of <dt> s.` / `film: film.seconds (<s>) is not a whole number of frames at <fps> fps.` |
| Storage short | `film: not enough storage in OPFS — <need> MB needed, <avail> MB available.` |
| Calibration over 1 s limit | `film: a <step\|draw> at <W>×<H> took <ms> ms, over the 1,000 ms per-submission limit (Windows resets a GPU busy for ~2 s). Try a smaller size, fewer particles, or a faster GPU.` |
| Runtime submission over 1.5 s | `film: submission at frame <j> took <ms> ms, over the 1,500 ms safety stop.` |
| Out of GPU memory / validation error | `film: GPU <out-of-memory\|validation> error while building or filming the scene: <message>.` |
| Device lost | `film: GPU device lost at frame <j> (<reason>); partial file <kept\|discarded>.` |
| Encoder error | `film: video encoder failed at frame <j>: <message>.` |
| Cancelled | `DOMException('Export cancelled', 'AbortError')` (not shown as a failure). |

## 9. MetaForge adoption

MetaForge's `src/engine/exporter.ts` becomes a thin adapter:

| Today (MetaForge) | After |
|---|---|
| `viewport.enterExport(w, h)` resizes the visible canvas | `viewport.stop()` pauses the preview; the film draws into its own target and `viewport.start()` resumes in `finally` |
| `CanvasSource(viewport.canvas)` | `recordFilm({ source: 'auto' })` (`VideoSampleSource` over the film engine's `OffscreenCanvas` or `FrameReadback`) |
| `BufferTarget` | proposed `auto`: writable OPFS first; budgeted memory fallback only (§4.5) |
| `calibrate(...)` | `calibrateFilm(...)` |
| `runtime.reset()`, then steps on the live runtime | a fresh runtime adapter with explicit placement, exact restore and runner-owned warm-up |
| `snapshot(...)` | `snapshot(...)` from `null-graph/film` (no resize of the live canvas) |
| `fitView` / `containView` | imported from `null-graph/film` as `fitAspect` / `containAspect` (same math) |

The studio's export dialog keeps its UI. It gains: the probe's reasons for missing sizes, the
"Save to a file" option where `showSaveFilePicker` exists, and the storage check.

**`SceneRuntime` changes:**
1. Expose a `createWorkload` adapter that constructs `SceneRuntime` on the film engine passed by
   `recordFilm`; the runtime remains an app detail, not a library baseline.
2. Separate `seedReset()` (re-upload initial particle seed and run step 0 placement) from warm-up
   stepping so `recordFilm` owns the `W` warm-up steps and calibration reset without running
   warm-up twice. Document placement as initialization; restore RNG, time, frame counters and
   temporal post state as well as particle bytes. Test 30/60 fps and W = 0 before adoption.

## 10. Tests

**Headless** (`test/film-browser.test.mjs`, Node, no GPU):
- **B1:** `planFilm`, `readCamera`, `cameraPath` (with and without `fov`), `fitAspect` and
  `containAspect` (the Node tool's U1 and U4 cases run against `dist/film.js`, plus vertical 9:16
  and square 1:1 reframing invariants).
- **B2:** option checking: every refusal of §8 that needs no GPU (odd sizes, fps, seconds, unknown
  container or target, storage shortfall from a mock estimate).
- **B3:** `bitrateFor`, `predictBytes`, progress and ETA arithmetic, and the guard's periodic vs
  per-submission fence selection (`msPerFrame ≤ 150` vs `> 150`).
- **B4:** `FrameReadback`'s row unpadding (widths that need 256 B padding: 720, 1080, 1366) and
  optional `out` buffer reuse (`design-principles.md` §5, §6: zero allocation when `out` is
  supplied).
- **B5:** `encodePNG` round-trip (exact RGBA8 bytes, including pixels with `0 < alpha < 255` that
  Canvas 2D would premultiply).
- **B6:** session transition table; double finish/cancel/dispose; failure at each acquisition;
  original error retained when disposal also throws; observer exceptions isolated; sample close
  exactly once after successful or rejected add; raw frame closed if wrapping fails.
- **B7:** exact reset/rebuild after calibration, placement separate from W, 30/60 fps step counts;
  timestamp/duration quantization and safe-integer limits; frame limits and memory budget stop.
- **B8:** capability levels, container/codec intersection, target fallback without silent memory
  escalation; emitted declarations and pure-helper import without mediabunny installed.

**Browser** (`test/film-browser-gpu/`, run in Chrome through `npm run drive`, on the AMD iGPU and,
with `--force_high_performance_gpu`, on the GTX 1650):
- **G1 (P1):** Kerr plus a stateless callback workload, recorded twice with the hash sink: every
  frame hash equal. Track owned create/destroy calls across repeated exports; no growth in retained
  app references. Do not claim exact physical VRAM reclamation from JavaScript measurements.
- **G2 (P4):** a test pattern film (BT.709 colour bars, an orientation marker, and an `alpha = 0`
  additive region): first assert lossless readback bytes/orientation/channels, then encode/decode.
  Establish codec-specific lossy tolerances on flat patch interiors, excluding 4:2:0 edges; inspect
  range/transfer metadata. Additive RGB survives as opaque on both sources. No universal ±3 claim.
- **G3:** frame timing and fast-start layout: the decoded file has exactly J frames, timestamps
  quantized per §4.3 within one container tick (also checked with `ffprobe`). Parse top-level boxes
  to verify `moov` precedes `mdat`: memory uses `in-memory`, seekable streams use `reserve`.
- **G4:** OPFS target: a 4K, 10 s film streams to OPFS (`chunked: true`) with the page's JS heap
  measured against documented duration/size baselines (`performance.measureUserAgentSpecificMemory`
  where available). Account separately for retained chunk cache, temporary copies, growing mux
  metadata, encoder, readback and GPU allocations; do not equate chunk cache with total memory.
- **G5 (P5):** the guard, with a lowered limit, refuses by calibration and stops on a single
  submission over the stop limit (while allowing an 8-frame batch whose total exceeds the stop
  limit when each individual operation is under it); cancellation removes only the session's
  uncommitted OPFS payload. Use lowered limits/injected timing, never deliberately trigger TDR.
- **G6:** canvas source against readback source on one scene: decoded frames agree within the
  codec's loss (proves the canvas path draws the same image).
- **G7:** two concurrent tabs, same display name: no cross-export deletion; locks protect live
  writers; completed results survive later recordings; explicit orphan cleanup and idempotent dispose.
- **G8:** permission denial, mid-write quota failure, encoder/finalize failure and device-loss
  injection; discard aborts writes; opt-in partial returns only if finalized/committed and decodable.
- **G9:** cancellation during init/calibration/warm-up/add/finalization; visibility transition,
  explicit task yielding and UI responsiveness; preview restore or explicit device-loss recovery.

Relevant assertions are watched failing before trusted (break the timestamp unit on `VideoFrame`,
swap channels, skip a step, omit `sample.close()`).

## 11. Dependencies and the build

- **mediabunny** (MPL-2.0, pinned in MetaForge at `1.61.3`) becomes an **optional peer dependency**
  of `null-graph` (`peerDependencies` + `peerDependenciesMeta.mediabunny.optional = true`, and in
  `devDependencies` for tests), just as `onnxruntime-web` is.
- **Dynamic import and structural types (no top-level `import from 'mediabunny'`):**
  - `build.js` adds `'film': 'src/extras/film/index.ts'` and marks `'mediabunny'` in `external:
    ['gl-matrix', 'mediabunny']`.
  - `sinks.ts` and `probe.ts` load `mediabunny` via `await import('mediabunny')` (or accept an
    injected `mediabunny` namespace in options for unit tests), and declare minimal structural
    interfaces in `src/extras/film/` (matching `OrtLike` in `src/extras/inference/OrtRuntime.ts:12`).
  - **Why:** ES module static imports hoist to the top of `dist/film.js`. A static import of
    `mediabunny` would break `tools/film/film.mjs` in Node and any browser caller that only imports
    `planFilm`, `cameraPath`, `fitAspect` or `snapshot` without installing `mediabunny`, and would
    pull `@types/dom-webcodecs` into downstream `tsc` builds (`tsconfig.json` restricts `"types":
    ["@webgpu/types"]`). Dynamic import alone does not isolate declarations: use local public
    structural types and test emitted `.d.ts` files with a consumer lacking mediabunny. The ORT
    precedent is namespace injection, not dynamic loading; keep injection available here too.
- **Why not a muxer of our own:** MP4 boxes, `fastStart: 'reserve'`, WebM, codec strings and
  encoder backpressure are a project; mediabunny does them, and MetaForge has measured it on this
  machine.
- **Why `VideoSampleSource` over `CanvasSource`:** `CanvasSource` wraps `VideoSampleSource` for a
  single canvas (`media-source.js:1180`). Using `VideoSampleSource` with `VideoSample` lets source A
  (`OffscreenCanvas`) and source B (`FrameReadback` → `VideoFrame`) share one sink implementation.
- **Release:** proposed `null-graph@1.2.0` only after F-MF and package-isolation/lifecycle gates.
  Run build, core/film tests, declaration consumer checks and `npm pack --dry-run`; test the packed
  artifact in MetaForge before publishing. A date is not a release gate (Q8 remains open).

## 12. Engine touchpoints

Following `design-principles.md` §14, required vs optional changes to `src/core`:

| Change | Kind | Where | Why / Zero-change fallback and its cost |
|---|---|---|---|
| `NullGraph.destroy()` and `WebGPUContext.destroy()` | **Required** | `src/core/NullGraph.ts`, `src/core/WebGPUContext.ts` | Idempotently release engine-owned targets, camera uniform, cached textures and batch allocations; unconfigure canvas; clear retained passes/stages; set destroyed state. Never destroy an adopted device or borrowed assets. Support partially failed initialization and attempt all releases. *Zero-change fallback:* film reaches into context internals, coupling extras to core and leaving public teardown incomplete. |
| Track batch resource ownership | **Required where allocation/borrowing is ambiguous** | `src/core/BatchManager.ts`, batch types in `src/core/types.ts` | Internally allocated indirect/source storage must be released, but `sharedSourceBuffer`, caller geometry, camera buffers and external textures must not be destroyed. Existing `clearBatch` frees storage/indirect buffers but does not resolve all source ownership. Keep ownership records through clear/destroy paths and test sharing between engines. |

No render-loop hook is needed. Exact teardown changes follow the ownership audit, not a line-count
estimate. `depth24plus` allocation size is implementation-dependent; 33.2 MB is a nominal 4-byte
pixel estimate at 3840×2160, not measured VRAM. Snapshot and video use the same lifecycle ownership.

## 13. Files

| File | Holds |
|---|---|
| `src/core/NullGraph.ts`, `src/core/WebGPUContext.ts` | `destroy()` (frees engine-owned targets; respects `ownsDevice`) |
| `src/extras/film/index.ts` | the public exports |
| `src/extras/film/plan.ts` | `planFilm`, `readCamera`, `bitrateFor`, `predictBytes` (from `tools/film/manifest.mjs` + MetaForge) |
| `src/extras/film/camera.ts` | `cameraPath` (from `tools/film/camera.mjs`, with optional `fov`), `fitAspect`, `containAspect` (from MetaForge) |
| `src/extras/film/probe.ts` | non-destructive capability/configuration probe and explicit operational smoke checks |
| `src/extras/film/session.ts` | lifecycle state machine, sample ownership, terminal operations, cleanup ledger |
| `src/extras/film/storage.ts` | targets, export IDs, commit/abort, locks, explicit orphan cleanup and retention |
| `src/extras/film/readback.ts` | `FrameReadback` (from `tools/film/capture.mjs`, with `Uint8Array` and optional `out`) |
| `src/extras/film/png.ts` | `encodePNG` (`CompressionStream('deflate')` + CRC32), `snapshot` |
| `src/extras/film/sinks.ts` | the video (`VideoSampleSource` + `sample.close()`), hash, PNG and none sinks; memory, OPFS and file-handle targets |
| `src/extras/film/guard.ts` | `calibrateFilm`, the periodic and per-submission checks |
| `src/extras/film/record.ts` | `recordFilm`, the loop, error scopes, the report |
| `src/extras/film/worker.ts` | F4: `recordFilmInWorker` and the worker body |
| `test/film-browser.test.mjs` | B1–B8 |
| `test/film-browser-gpu/` | G1–G9, a page and its runner |
| `tools/film/manifest.mjs`, `camera.mjs` | re-export `planFilm`, `readCamera`, `cameraPath` from `dist/film.js` (one source of truth) |

## 14. Build order

Each phase has a validation gate and a work-log entry; commits remain an explicit implementation
workflow action. Estimates below are optimistic engineering effort, not delivery commitments.
Failure-injection, cross-tab storage and package checks are release work, not post-release polish.

| Phase | Builds | Exit | Est. |
|---|---|---|---|
| **S1** (spike) | A throwaway page in `test/spikes/film-browser/`: Kerr at 1080p and 4K, 300 frames, four ways: canvas frames from the visible canvas (today's MetaForge), from an unattached main-thread `OffscreenCanvas`, from an `OffscreenCanvas` in a worker; readback frames (`FrameReadback` → `VideoFrame('RGBX')`). Each into `VideoSampleSource` (`sample.close()` in `finally`) → MP4 in memory. Also: OPFS `StreamTarget({ chunked: true })` + `Mp4OutputFormat({ fastStart: 'reserve' })` at 4K. | ms a frame for each way on both GPUs; whether the unattached canvas is paced; heap peak with memory against OPFS; `ffprobe` verifies `moov` before `mdat`. Decides Q1 and Q2 (§16). **Measured 2026-10-24: "Spike S1 results".** | ½ day |
| **F1** | ownership-aware engine teardown (§12), pure `plan.ts`/`camera.ts`/`png.ts`, `probe.ts`, session state skeleton; build/export declarations and optional peer; Node pure-helper migration | B1, B2, B5 and dependency isolation portion of B8; core/Node film tests pass. B3/B4 wait for F2. | ½–1 day |
| **F2** | session lifecycle, readback/sinks/guard/runner/report; memory/hash/none sinks, explicit task yields, deterministic reset and cancellation | B1–B8 applicable to memory; G1–G3, guard portion of G5, G6 and G9 on both GPUs | 1–2 days |
| **F3** | storage/locks/IDs, OPFS and picked-file commit/abort, orphan cleanup, budget/quota failures, opt-in partial finalization and PNG sink | G4, storage portion of G5, G7–G9; 60 s 4K OPFS film | 1–2 days |
| **F-MF** | MetaForge adapter against local build, packed artifact then proposed published release | three shapes × three sizes; frame counts/initial-state tests; repeated teardown and cancel/partial gates; preview resumes or reports recovery; package isolation passes | ½–1 day |
| **F4** (if needed) | The worker: `worker.ts`, `recordFilmInWorker`, scene modules importable in a worker | the G tests pass in the worker; the UI's frame rate during export measured against F2 | 1 day |
| **F5** (later) | The test site: a Record setting on the demo pages; Herd & Wolves as a film scene in the browser | a Herd & Wolves film from the site, read by eye against the Node film | ½ day |

**For the hackathon:** finish/review the design before Monday 2026-10-26, 21:30 IST. S1 is the
first implementation gate, not evidence already obtained. F1–F3 and F-MF need several engineering
days plus measured validation; do not compress them into a quiet hour or a single performance day.
MetaForge retains its existing exporter until the replacement passes. Library-first remains a
recommendation (Q5), not an implementation-order decision or a demo dependency.

## 15. Deferred

- **D2.5: tracks and replay in the browser.** Record compute records into OPFS, replay with another
  camera or size without stepping. The pack and unpack stages run there unchanged; the store and
  the segment index are the work.
- **Resume** after a closed tab (needs checkpoints: D2.5).
- **Audio**, an end card, titles drawn on the GPU (the film-overlays design covers ffmpeg; a
  browser overlay pass is its own design).
- **Motion blur** by sub-frame accumulation (MetaForge lists it as optional): N draws a frame at
  τ_j + k/(N·r), averaged in a compute pass before capture.
- **PNG sequences zipped** for download.
- **Firefox and Safari** checks: the module detects and declines what they lack; nobody has run G1–G9
  there yet.

## 16. Questions for the user

| # | Question | Recommendation |
|---|---|---|
| Q1 | The default frame source: canvas frames or readback? | **Canvas provisionally**, only if S1 proves performance, pacing and additive-alpha/colour correctness; readback for exact bytes and operational fallback before frame 0. |
| Q2 | Worker in the first slice (F4), or main thread with a separate target? | **Main thread first.** The separate target already fixes the compositor problem; the worker costs a second device and scene modules. Build F4 only if F2 shows the UI stuttering. *(Decided 2026-10-24: main thread first.)* |
| Q3 | Automatic target and memory budget. | **Prefer writable OPFS**, otherwise permit memory only within configurable predicted/runtime payload budget (proposed 256 MiB). No unlimited fallback and no resolution-only rule. This does not cap total export RAM. |
| Q4 | mediabunny as an optional peer dependency of the library. | **Yes**, loaded via dynamic `await import('mediabunny')` with local structural types, as with `onnxruntime-web`. *(Decided 2026-10-24: yes.)* |
| Q5 | Library first, then MetaForge (this order), or build in MetaForge and lift into the library afterwards? | **Library first** for F1–F2, since the pieces already exist in two places (Node tool, MetaForge) and a third copy would drift. If the hackathon days are tight, keep MetaForge's exporter and land the library after the submission. *(Left for later.)* |
| Q6 | Should `fitAspect` and `containAspect` (MetaForge's vertical and square framing) live in the library? | **Yes**, small and general; the test site needs the same framing for Shorts. |
| Q7 | Default container and codec order. | **MP4/H.264**; VP9/AV1 fallback only when the app opts in and exposes the selected codec/compatibility distinction. WebM when requested; intersect codecs with container support. |
| Q8 | Release `null-graph@1.2.0` during the hackathon? | **Only after F-MF, lifecycle and packed-package gates pass**; never release solely to meet the calendar. |

**Decided by the user, 2026-10-24:** Q2 as recommended (main thread first; F4 only if F2 shows
the UI stuttering). Q4 as recommended (mediabunny an optional peer dependency). Q5 left for later.
MetaForge is the hackathon entry. Q1 waits for S1; Q3 and Q6–Q8 are open.

## 17. Review evidence and remaining validation

Draft 3 is a design review, not a new benchmark. Local implementation evidence is the core,
Node tools and `D:/orion-realms/metaforge/node_modules/mediabunny/` at 1.61.3. External behavior
should be checked against primary references during implementation:

- WebGPU specification: canvas current-texture expiry, queue completion and device error scopes.
- WebCodecs specification: integer microsecond timestamps, sample ownership and encoder behavior.
- Microsoft WDDM timeout detection and recovery documentation: default timeout and preemption.
- File System specification: writable-stream commit/abort and OPFS operations.
- Web Locks specification: exclusive ownership across same-origin contexts.
- Mediabunny's pinned source and documentation: sample closure, codec/container support,
  stream target buffering and front-reserved MP4 metadata.

Unproven until S1/G tests: unattached-canvas pacing, zero-copy capture, input/output colour
equivalence, worker encoder support, peak memory and partial recovery after real device loss.
Do not upgrade these hypotheses to capability guarantees based on API presence alone.

---

## Spike S1 results — 2026-10-24

**What ran.** `npm run spike:film-browser` (`test/spikes/film-browser/`, throwaway; its header lists
the options), Chrome 155 on this laptop: the AMD Radeon iGPU (GCN-5, Chrome's default) and the
GTX 1650 (`--force_high_performance_gpu`), headless (`--headless=new`) and, for pacing, headed.
- **The film:** the Node tool's own Kerr scene (`tools/film/scenes/kerr-black-hole.mjs`,
  1,048,576 particles, Bloom and Tonemap), bundled by esbuild with `node:crypto` shimmed. One step a
  frame at 60 fps after four fenced warm-up frames; the camera orbits slowly. Each case is a fresh
  page load with its own device.
- **The sink:** every frame goes `VideoFrame` → `VideoSample` → `VideoSampleSource.add`, the
  sample closed in `finally`.
  - The MP4 is H.264 at `w·h·fps·0.16` (19.9 Mbit/s at 1080p, 79.6 at 4K), keyframes every 2 s,
    `latencyMode: 'quality'`.
  - Into memory (`BufferTarget`, `fastStart: 'in-memory'`) or OPFS (`StreamTarget`,
    `chunked: true`, `fastStart: 'reserve'` with `maximumPacketCount`).
  - mediabunny 1.61.3 is now an exact dev dependency.
- **The loop:** canvas ways fence every 8 frames (every frame when a warm-up frame measured over
  150 ms: two AMD 4K cases). A `MessageChannel` task follows every frame (§4.9).
- **The runner:** samples the working sets of Chrome's renderer (picked by CPU time, through CDP
  `SystemInfo`) and GPU process. It ffprobes every MP4, decodes the colour pattern, and compares
  each way's frame 150 with the readback's.
- **Evidence:** 116 cases in the final runs: zero failures, zero WebGPU errors. Logs, JSON and
  kept MP4s are in `test/spikes/film-browser/out/` (git-ignored).

The four ways:
- **visible:** the engine on the page's `<canvas>` (MetaForge's exporter today),
  `new VideoFrame(canvas)`;
- **offscreen:** an `OffscreenCanvas` nobody shows, on the main thread;
- **worker:** the same in a dedicated worker, on the worker's own device;
- **readback:** a headless engine; `screenTexture` → three staging buffers, one frame in flight,
  `new VideoFrame(mappedRange, { format: 'RGBX', layout })`.

### Export: ms a frame (300 frames = 5 s, default encoder)

| Way | AMD 1080p | AMD 4K | GTX 1650 1080p | GTX 1650 4K |
|---|---|---|---|---|
| visible | 49.7 | 142.7 | 10.7 | 31.1 |
| offscreen | 49.6 | 127.5 | 8.6 | 30.3 |
| worker | 51.5 | 128.7 | 9.0 | 32.6 |
| readback | 44.7 | 124.1 | 10.3 | 37.0 |
| *headed:* visible / offscreen / readback | 51.3 / 43.8 / 46.7 | 145.2 / 137.5 / – | 12.1 / 11.2 / 11.2 | 38.5 / 33.3 / – |

- **No way is decisively faster.** The iGPU is GPU-bound: the readback's wait for its previous
  frame, 43 ms at 1080p and 118 ms at 4K, is the frame's GPU time. On the GTX 1650 the encoder
  sets the pace: a canvas way's `add` waits 7–8 ms at 1080p and about 20 ms at 4K.
- **The visible canvas costs 2–15% more** than the offscreen one, most at 4K.
- **MetaForge's loop shape (no task yield)** ran 1080p at 47.5 / 43.6 ms (visible / offscreen) on
  AMD and 9.2 / 8.8 on the GTX 1650. The yield costs up to 10%.
- **Canvas frames equal readback frames.** Frame 150 of every canvas way equals the readback's in
  RGB, byte for byte, on both GPUs, the worker's own device included.

### Pacing (Q1's "unattached canvas" question)

A light scene (65,536 particles, 1080p), a task between frames, no encoder; mean ms a frame:

| | headless engine | visible | offscreen | worker | readback |
|---|---|---|---|---|---|
| AMD, headless Chrome | 6.0 | 7.9 | 9.6 | 8.8 | 8.9 |
| AMD, headed (steady 8-frame blocks) | ~9.4 | ~8.6 | 9.3 | – | 11.3 |
| GTX 1650, headless | 2.7 | 2.3 | 2.5 | 2.5 | 4.7 |
| GTX 1650, headed | 1.5 | 1.6 | 1.6 | – | 4.0 |

- **Nothing is paced to the display:** not the unattached canvas, and not the visible one either,
  in headless or headed Chrome, on either GPU. The MetaForge slowness this design blamed on the
  compositor is the iGPU's GPU time.
- **The iGPU stalls in a real window.** In headed Chrome on the AMD iGPU (which also drives the
  desktop), two of four pace runs had isolated stalls of 0.4 s and 2.0 s, wherever the frame was
  drawn. Their means read 15.6 and 27.9 ms. Cause not established.
- **Queue completion is reported promptly:** `onSubmittedWorkDone` and a 4-byte `mapAsync` queued
  behind the same frame took the same time in every way and mode (AMD 7–12 ms, GTX 1650 3–4 ms).
  The compositor wait in MetaForge's `viewport.ts` comment did not appear.

### The main thread (Q2)

- **The page stays at 60 Hz:** rAF intervals were 16.7 / 16.8 ms (p50 / p90) in every export, on
  every way, with or without the task yield.
- **Encoder start-up blocks the main thread at 4K, once:** two to four long tasks (180–630 ms in
  all) around frames 0 and 3 on the main-thread ways; none in the worker way. No long tasks at
  1080p.

### Which encoder `no-preference` gets, and what it does with the bitrate

- **`no-preference` is the hardware path:** its `mdat` is byte for byte that of
  `prefer-hardware`, at both sizes, in both GPU runs.
- **One hardware encoder serves both GPU runs.** 120 JavaScript-made frames gave identical payloads
  in the AMD and the NVIDIA run, for `prefer-hardware` and `prefer-software` alike, and the
  hardware streams' SPS/PPS match: `--force_high_performance_gpu` moves WebGPU, not the video
  encoder. Which GPU's encoder it is was not established.
  - The streams differ in their headers: the hardware one is CABAC with POC type 0, the software
    one CAVLC with POC type 2, both declaring High.

4K, 5 s of Kerr:

| Setting | AMD run, Mbit/s | NVIDIA run | Notes |
|---|---|---|---|
| hardware VBR (= default) | **8.3** | **8.3** | 10× under the 79.6 asked; visibly blocky |
| hardware CBR | **8.6** | 82.4 | on target in one run only (seen twice) |
| hardware, constant QP 34 / 30 / 26 / 20 | – / – / – / 333 | 68 / 113 / 179 / 334 | `bitrateMode: 'quantizer'` |
| software VBR | 79.5 | 79.3 | renderer 0.6–1.1 GiB through the film |
| software + QP 20 | = software VBR | = software VBR | quantizer unsupported; mediabunny fell back |

- **At 1080p every bitrate setting lands near its target:** hardware VBR 20.6, CBR 20.0, software
  19.8. Hardware QP 34 / 30 / 26 cost 27.5 / 45.7 / 70.7 Mbit/s.
- **Read by eye** (frame 150, 800 × 450 crops at 4K): the 8.3 Mbit/s default smears the particle
  grain into macroblocks. Software VBR, hardware CBR at 82 and QP 34 keep it. This is what
  MetaForge's exporter (`CanvasSource`, no preference) records at 4K on this laptop, headed or not.
- **The software encoder is not bitrate-bound on hard content:** JavaScript noise frames came out
  at 97 Mbit/s at 1080p and 388 at 4K, against 19.9 and 79.6.
- **mediabunny's codec probe** (`getFirstEncodableVideoCodec`) takes `frameRate` but no hardware
  preference. `onEncoderConfig` is called once per candidate, before its support check; the last
  call is the config in use.

### Memory against OPFS (30 s at 4K: 1800 frames, 253 MiB of MP4)

Offscreen way, GTX 1650, hardware QP 34 (71 Mbit/s):

| Target | Renderer working set, base → peak | Page JS at frames-done / finished | V8 heap peak | Finalize |
|---|---|---|---|---|
| memory | 107 → **874 MiB** | 414 / 671 MiB | 254 MiB | 501 ms |
| OPFS | 134 → **198 MiB** | 180 / 161 MiB | 81 MiB | 255 ms |

- **The memory target peaked at about 3× its payload.** mediabunny's buffer doubles as it grows,
  finalize slices it, and the `Blob` is one more copy. OPFS stayed at +64 MiB: two 16 MiB chunks
  and overhead, whatever the length.
- **About 130 MiB of page JS is frames in flight** in both targets (`measureUserAgentSpecificMemory`
  counts in-flight 4K `VideoFrame`s). It is not the payload.
- **Each way's own cost, through a 4K film** (renderer median, small payloads): 40–150 MiB with a
  canvas on the hardware path, 420–530 MiB with readback, 0.6–1.1 GiB with the software encoder.
- **Single samples at a 4K film's start reached 0.7–1.0 GiB on the canvas ways too,** with the
  encoder's start-up long tasks. Not investigated.

### The file

- **`moov` comes first in every MP4:** `ftyp moov mdat` in memory, and `ftyp moov free mdat` with
  `reserve` (the `free` box holds 5.9 KB of headroom at 300 frames, 58.9 KB at 1800). ffprobe's own
  parse order agrees.
- **Exactly J packets, exactly timed:** at 30, 300 and 1800 frames, timestamps were exact (zero
  ticks of error, time base 1/60), with a key frame every 2 s. H.264 High, `yuv420p`.

### Alpha and colour (the alpha pattern, 1280 × 720)

- **In the video, the canvas ways match the readback:** each takes the canvas's premultiplied RGB
  as opaque, so a glow at alpha 0 survives. Decoded patches (a=0.5 / additive a=0 / clear) were
  100,49,23 / 203,100,49 / 0,0,0 against the readback's 100,50,24 / 203,102,49 / 0,0,0, on both
  GPUs.
- **But `VideoFrame.copyTo()` un-premultiplies a canvas frame:** the a=0.5 patch reads
  203,102,50, and the additive glow reads 0,0,0. Canvas frames cannot be the exact bytes for
  hashes or PNGs where alpha < 1 (§4.3 B stands).
- **The two sources are tagged differently:** canvas frames encode as limited range (`tv`), RGBX
  readback frames as full range (`pc`). Both carry BT.709 matrix and primaries, with transfer
  `iec61966-2-1` (sRGB).
- **Decode with the stream's tags.** ffmpeg decodes to RGB with BT.601 unless told otherwise, and
  reads BT.709 red 255,0,0 as 232,0,1.

### Cancel over OPFS (§4.5)

- **`output.cancel()` commits a partial file.** On a `StreamTarget` over a
  `FileSystemWritableFileStream` (1 MiB chunks, cancelled after 60 frames at 1080p), it left a
  committed 2,097,152-byte `film.mp4`: mediabunny 1.61.3 closes its writer on cancel
  (`target.js` `_close`), and closing commits.
- **An interposed `WritableStream` aborts instead:** it forwards writes and turns the close into an
  abort once the session has chosen to discard, leaving 0 bytes. Same on both GPUs.

### What changes in the design (proposed; the user decides)

1. **§4.3 and Q1:** the unattached `OffscreenCanvas` is not paced, and neither is the visible
   canvas. The separate target's case rests on these:
   - it leaves the preview's canvas and size alone;
   - it is 2–15% faster than the visible canvas;
   - it costs far less memory than readback at 4K;
   - it encodes limited range.

   Not on compositor pacing.
2. **§4.4, rate control:** a requested bitrate is not a dependable quality control at 4K on this
   machine. The default path delivered a tenth of it.
   - The report should carry the achieved bitrate, and the session should warn when it falls far
     below the request.
   - Constant QP on the hardware path is the predictable control measured here; whether to adopt it
     is Q9.
   - With QP the size depends on content, so §4.5's storage preflight needs an upper estimate, or a
     re-check after the first seconds.
3. **§4.4, the probe:** a hardware-preference-aware probe needs its own
   `VideoEncoder.isConfigSupported` call, since mediabunny's helper cannot express it. Record the
   last `onEncoderConfig` call, not the first.
4. **§4.5:** the OPFS writer must be wrapped so that discard aborts. Without the wrapper, mediabunny
   commits partial files on cancel.
5. **§4.5 and Q3:** a memory target holds about 3× its payload at peak. A 256 MiB payload budget
   means a renderer near 0.8 GiB on top of the film's own cost.
6. **§4.4, colour:** pick one range for both sources, or test both; full-range H.264 is the less
   compatible tag. G2 measures it.
7. **§4.7, the guard:** in a real window the iGPU stalled for up to 2.0 s under light work. Per
   submission fencing (frames over 150 ms) would read such a stall as the film's own cost and stop
   the export at 1.5 s. Report the evidence, and consider one re-measure before stopping.
8. **§7 and Q2:** the main thread kept 60 Hz rAF in every way. The worker's measured gain is the
   absence of the 4K encoder start-up hitch (0.2–0.6 s). That does not change "main thread first".

### Not run

- Firefox and Safari; Chrome on other machines or GPUs; a display other than 60 Hz.
- MetaForge's own scenes. The work is Kerr only, which leaves alpha at 1 after Tonemap; the pattern
  covered alpha < 1.
- The G tests themselves: device loss, quota, permission and lock cases; 60 s OPFS films; G2's
  per-codec tolerances.
- Which physical encoder serves the "hardware" path, and why its CBR differs between GPU runs.
- The cause of the iGPU stalls in the headed window.

### The questions, on these numbers

| # | Question | Recommendation |
|---|---|---|
| Q1 | The default frame source. | **Offscreen canvas for video, readback for exact bytes** (hashes, PNG, tests), as provisionally proposed. Same speed (±20%); limited range; 40–150 MiB of renderer through a 4K film where readback holds 420–530 MiB; colour in the video equal to readback's. The pacing argument is withdrawn. |
| Q3 | Automatic target and memory budget. | **OPFS by default** (+64 MiB flat at 4K for 30 s). A memory fallback within a payload budget, sized knowing the 3× peak. |
| Q9 (new) | Rate control and encoder preference. | **Ask for constant QP on the hardware path when the browser supports it, else bitrate; always report the achieved bitrate.** QP 34 gave 68 Mbit/s for Kerr at 4K. The QP and its size bound need the F2/G2 measurements. Software VBR hits the bitrate on Kerr but holds 0.6–1.1 GiB of renderer memory at 4K, and overshoots on noise. |
