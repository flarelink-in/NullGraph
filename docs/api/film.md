# Film — offline browser rendering (F1–F3)

**Implementation status, 2026-10-25:**
- **Built:** F1's helpers and engine teardown; F2's managed runner, session, readback, video into
  memory, hashes, measurements and PNG snapshots; F3's storage: video streamed to OPFS or to a file
  the user picked, PNG sequences, Web Locks, and the films an app lists, cleans up and removes.
- **Tested:** the browser G suite passed on an AMD iGPU and an NVIDIA GPU in Chrome
  ([validation](#validation-and-limits)).
- **Later phases:** a worker (F4), and MetaForge's adapter (F-MF).

The design and its measurements: `docs/offline-films-browser-design.md` ("Phase F2 results",
"Phase F3 results").

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

## Output, storage and results

Output options are `container`, `codecs`, `bitrate`, `target`, `name` and `memoryBudgetBytes`.
- **Containers:** `'mp4'` (H.264 alone unless you list more), `'webm'` (VP9, then AV1), or `'png'`,
  a PNG sequence: each frame's exact bytes as a file of its own.
- **Bitrate:** a video's bitrate defaults to `round(width × height × fps × 0.16)`, and its hardware
  preference to `no-preference`. The asked bitrate is not what an encoder delivers: the report gives
  both (rate control is the design's parked Q9).
- **`name`** is the download name: metadata, never a path.

**Where the file goes (`target`).** It is chosen before frame 0 and never changed during the film; the
report says where it went and why (`report.output.target`, `targetReason`).

| `target` | What happens |
|---|---|
| `'auto'` (default) | OPFS where it is there, has room for the film's predicted bytes, and takes a write. Otherwise memory, within `memoryBudgetBytes`, saying why. Otherwise a refusal giving both reasons: never a silent escalation. |
| `'opfs'` | OPFS, or a refusal: `film: not enough storage in OPFS — <need> MB needed, <avail> MB available.` |
| `'memory'` | A Blob, within `memoryBudgetBytes` (default 256 MiB of encoded payload, not of process memory: spike S1 measured about 3× the payload at peak). |
| `{ file: handle }` | The `FileSystemFileHandle` your app got from `showSaveFilePicker` in a user gesture. The library never opens a picker or asks for permission. |

A PNG sequence goes to OPFS only (`'auto'` or `'opfs'`), and is read back: its frames need exact bytes.

**In OPFS** a film is `null-graph-films/<exportId>/`: its payload (`film.mp4`, or a `frames/` folder of
`frame-000000.png`, …) beside its manifest, `film.json`.
- **While it is written:** the manifest says `writing`, and the film's Web Lock
  (`null-graph-film:<exportId>`) is held. Video writes wait in the browser's swap file, so the payload
  reads empty until committed.
- **When it is committed:** once the muxer has finalized, the library commits the file; only then does
  the manifest say `completed` (or `partial`), with the film's report. mediabunny also closes its
  stream when it cancels and when a finalize fails; neither close commits anything.
- **When it is discarded** (cancelled, failed, out of room): nothing is kept, and the folder is
  removed. The library aborts the file's writable, where closing it would commit a half film (spike
  S1's finding).
- **If the manifest cannot be marked complete**, the film is removed, and the error says so: an
  unmarked film would look abandoned to every later cleanup.
- **Memory:** an MP4 streamed to a file reserves its index at the front for the film's J frames
  (`ftyp moov free mdat`), so it plays as it loads. The page then holds the frames in flight and two
  16 MiB chunks, not the film: through a 60 s 4K film, its JS memory moved by under 2 MiB while the
  file grew to 57.7 MiB (Phase F3 results).
- **A storage check runs before frame 0**, comparing the predicted bytes with OPFS's estimate. It is
  an estimate, not a reservation: a write can still run out of room mid-film, which ends the film with
  `film: OPFS ran out of room at frame <j> (QuotaExceededError: …); free some storage, or film shorter
  or at a lower bitrate.`

**A picked file** is the user's. It is written through its handle and committed when the film is
finalized; a discarded film, one that fails at finalize included, leaves the file as it was (a file the
picker just made stays, empty), and `dispose()` never touches it. A handle the browser refuses at write time is refused in plain words:
`film: the chosen file "<name>" may not be written (NotAllowedError: …); ask for write permission in a
user gesture (handle.requestPermission({ mode: 'readwrite' })) and try again.`

A successful result has `{ status, file, name, codec, target, exportId, sequence, frames, report, dispose }`:

- **`status`** is `'completed'` or `'partial'`. A partial must have finalized, been committed and, in
  OPFS, been marked `partial`; `report.failure` says why it stopped.
- **`file`** is the video: a Blob in memory, or a `File` read from OPFS or the picked file, which holds
  no memory. It is null for a PNG sequence and when the output is off.
- **`target`** is `'memory'`, `'opfs'` or `'file'`. `exportId` names the film's OPFS folder (null
  elsewhere), as `listFilms` names it.
- **`sequence`** is a PNG sequence's frames folder (a `FileSystemDirectoryHandle`) and its file names,
  frame 0 first; null for a video.
- **`dispose()`** removes the film's OPFS folder, after which its `File` reads no more. It does nothing
  for memory (you revoke your own object URLs) or for a picked file. It is idempotent. Keep the film
  until it has been downloaded or shared: a download click is no sign that it is done.
- **Reports** include:
  - where the film went (`output.target`, `targetReason`, `storage`: path, lock, committed, and the
    storage check's numbers);
  - source selection, planned/accepted/encoded frames, encoder config, bytes and the achieved
    bitrate;
  - timings, hashes when requested, state history, GPU errors, a lost device's reason, and
    secondary cleanup failures;
  - the guard's evidence: calibration, fencing, where fencing switched, the worst fenced step and
    draw, and the operations left unjudged.

## Films in OPFS

An app sees the films it has made, in this tab or another, and decides what to remove.

- **`listFilms()`** lists every film, oldest first, and changes nothing. Each entry gives `exportId`,
  `status` (`writing`, `completed`, `partial`, or `unknown` when no manifest was written), `name`,
  `container`, `frames`, `bytes`, `created`, `report`, `getFile()` and `getFrames()`. It also gives
  `live`: a writer holds the film's lock now, in any tab (null where Web Locks are missing).
- **`cleanupFilms()`** removes the incomplete films no writer holds: those whose tab closed, crashed
  or was discarded mid-film. It removes a film only while holding that film's lock, so a film being
  written anywhere is kept, and only if the film's manifest, read again under that lock, still says
  it is incomplete, so a film that completes while cleanup runs is kept too. Completed and partial
  films are kept, even old ones: removing those is the app's call. Without Web Locks a live film cannot be told from an abandoned one, so it removes
  nothing and says why. It never resumes a film. It returns `{ removed, kept }`, each kept film with
  its reason.
- **`removeFilm(exportId)`** removes one film, completed or not. It refuses one being written now:
  `film: the film "<id>" is being written now (its lock is held); stop or cancel that film first.`

Two tabs never collide: every film has a random export id, whatever its name. Nothing is swept
automatically, not even by `probeFilm`.

OPFS is best-effort storage: under storage pressure the browser may evict it, films included.
Whether to ask for persistence (`navigator.storage.persist()`) is the app's decision. The library
never asks, and never promises a film survives eviction, a cleared site or a discarded tab.

## App-driven sessions

`createFilmSession({ film, output?, mediabunny?, signal? })` supplies the frames-in/file-out
piece without creating an engine; its `output` chooses a target as `recordFilm`'s does. Await
`session.add(videoFrame)` for backpressure. Ownership transfers on invocation, including rejected
adds: do not close the frame again. A PNG sequence's frames must be 8-bit RGB(A) VideoFrames at the
film's size (`'RGBA'` keeps alpha; `'RGBX'` writes it opaque).

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
- **What it does by default:** it writes nothing.
- **With `operational: true`** it writes a byte to OPFS under a folder of its own
  (`null-graph-films/.probe-<uuid>/`), holding that folder's lock as every writer does, and removes it
  again, so `targets.opfs.writable` says whether OPFS takes a write here.
- **`targets.locks`** says whether Web Locks exist, without which `cleanupFilms` cannot tell
  live films from abandoned ones.

## Validation and limits

After `npm run build:lib`:
- **`npm run test:film`** (114 tests) runs the whole runner headless, on stand-ins for the device,
  WebCodecs, mediabunny, OPFS and Web Locks. It includes garbage-collection checks that no film keeps
  its engine alive, and no result its session.
- **`npm run test:core`.**
- **`node test/film-browser-smoke.mjs`** films with real Node/Dawn readback, reset, hashes and
  snapshots, on the adapter that `NG_DAWN_FLAGS=adapter=AMD` or `adapter=NVIDIA` selects.
- **`npm run test:film:browser`** is the G suite. It runs `recordFilm` in Chrome on each GPU
  (`--gpu amd,nvidia`, `--only <cases>`, `--long`, `--headed`), loading `dist/film.js` through an
  import map, and reads every film back with ffprobe and ffmpeg. It needs both on `PATH`. Its 17
  cases check:
  - **G1:** the same film twice hashes the same, and nothing is left on the device;
  - **G2:** exact readback bytes, and decoded colours per codec and range; **G2-png:** a PNG
    sequence's frames, byte for byte;
  - **G3:** J frames, timestamps on their ticks, key frames, the MP4 layout for each target, and
    frame k showing k;
  - **G4:** Kerr at 4K streamed to OPFS, the page's JS memory flat while the file grows (G4-long, a
    60 s film, runs with `--long`);
  - **G5:** the guard; **G5-storage:** a discarded film leaves nothing in OPFS;
  - **G6:** canvas against readback;
  - **G7:** two tabs: a live writer's film is kept, an orphan is cleaned up;
  - **G8:** refused and picked files, a full quota, encoder and finalize failures, lost devices;
  - **G9:** cancellation, the main thread and the preview, hidden and frozen pages, `destroy()`,
    and a crashed GPU process.

  Every case fails if it leaves anything in OPFS. On 2026-10-25 it passed 422 checks on each GPU,
  and G4-long its 15.

**Limits:**
- **The guard measures time, not the GPU.** It measures queue-completion wall time, not GPU
  timestamps, and cannot interrupt a submission already in flight.
- **The first submission is unqualified:** it compiles the pipelines, and nothing qualifies it as
  safe.
- **The guard re-measures an isolated stall once (Q10).** When a fenced operation passes the stop
  but its calibrated operation was under a quarter of the stop, the runner measures the next fenced
  operation before stopping, so a single desktop stall does not end the film.
- **Untested:** Firefox and Safari, a real TDR, and a real `showSaveFilePicker` handle (the G suite
  writes through an OPFS file's handle, which has the same interface).