# Designing a filmable scene

The film runner records GPU state and renders frames on an explicit timeline. It does not supply
your simulation, physics solver or scene assembly. A film JSON describes the shoot; a scene module
implements the world and tells the runner what to step, save and draw.

This guide describes the repository tooling, not a published `null-graph/film` API. See
[setup and CLI commands](./README.md) before running a job.

## 1. Separate simulation, persistent state and drawing

Design around three operations:

- **Step:** advance the world by exactly `dt` simulated seconds, independent of wall time.
- **Restore:** load saved state into the same buffers the renderer normally reads.
- **Draw:** rebuild derived rendering data and render that state, without advancing the world.

The interactive app can schedule steps against its real-time clock. The offline runner waits for
each exact step. Replay restores tracks instead of calling the simulation. Use ticks, explicit
simulation time and seeded randomness, not `Date.now()` or render-frame timing, for world behaviour.
Seeded input alone does not guarantee identical results across GPU adapters or drivers.

Persistent visual state includes more than positions: animation phases, event ages, colours and
simulation clocks can all affect an image. Derived state, such as instance transforms built from
positions or camera-dependent visibility, can be regenerated while drawing.

## 2. Add a scene factory

For this checkout, a scene named `my-scene` is loaded from
`D:/orion-realms/null-graph/tools/film/scenes/my-scene.mjs`. Names use lowercase letters, digits and
hyphens. Export an async default factory. The runner passes one options object:

| Input | Meaning |
| --- | --- |
| `engine` | Initialized NullGraph engine and its GPU device |
| `width`, `height`, `scale` | Requested output dimensions and scene-controlled render scale |
| `settings` | Scene-specific values from the film JSON; validate and apply defaults yourself |
| `mode` | `record` or `replay` |
| `trackEvery` | Requested default sampling interval in simulation steps |
| `film.track` | Helper for declaring GPU-buffer tracks |
| `log` | Runner logging callback |
| `runtime` | Current ORT integration support; normally null for custom scenes |

Return this interface:

| Member | Responsibility |
| --- | --- |
| `dt` | Positive, fixed simulation timestep in seconds |
| `settings` | Resolved settings for logging and reports |
| `tracks` | Array of declarations made with `film.track` |
| `checkpoint` | Array of `{ name, buffer }` declarations for complete persistent state |
| `async step()` | Submit one simulation step and await completion |
| `view(state, seconds)` | Apply an orbit keyframe or handle null with a default view; update presentation time |
| `draw(frame)` | Return the sequence the runner will submit with `engine.render(...)` |
| `async prepareSurface(frame)` | Optional awaited derived reconstruction before drawing; split expensive work into bounded submissions. Called for record and replay, not tracks-only jobs. Must not advance persistent simulation state. |
| `destroy()` | Release scene-owned resources, not the runner's engine/device |
| `info` | Optional report metadata; `wgsl` and `networks` can carry dependency hashes |

The orbit state is `{ target: [x, y, z], distance, azimuth, polar }`, with angles in radians.
`draw()` returns a sequence: do not submit that same sequence yourself. Derived compute passes can
be in it, but physics, learning updates and other persistent world mutations must stay in `step()`.

Both modes must allocate the same declared buffers and layouts. In replay, avoid unnecessary brain
execution and simulation calibration; retain resources needed to build and draw the restored world.
The runner restores the initial full checkpoint before applying replay tracks.

## 3. Choose a clock the runner can schedule

Recording requires a whole number of simulation steps per output frame, at least one. For example,
`dt = 1 / 120` gives two steps per frame at 60 fps. `dt = 1 / 30` cannot record at 60 fps with the
current planner. Warm-up duration must also contain a whole number of steps. Duration and segment
duration must contain whole numbers of output frames.

The runner advances before drawing each frame. Its current presentation time argument is
`frame / fps`; the first image contains state after the first frame's simulation steps. Do not
infer a resumable world clock from calls to `view()` or `draw()`; keep it in declared state.

`view()` is called even for `--frames none`, but `draw()` is not. A tracks-only recording must
therefore produce all persistent state needed for later replay without depending on drawing.

## 4. Declare tracks for appearance, checkpoints for continuation

**Tracks answer: what must be restored to draw this moment?**
**Checkpoints answer: what must be restored to continue computing from this moment?**

For particles, tracks might contain positions, sizes, colours and ages. Checkpoints may additionally
need velocities, solver history, random state and simulation counters. Immutable geometry and
textures can load from assets; fields omitted from packed tracks retain their initial checkpoint
values unless explicitly rebuilt.

The following declaration examples assume a scene-owned GPU buffer named `particles`, records of
16 bytes (`x`, `y`, `z`, `age`, each f32), and factory inputs `film`, `trackEvery` and a local `dt`.
Use one alternative, not both for the same buffer:

```js
// Simple starting point: complete records held between samples.
const rawTrack = film.track('particles', particles, {
    stride: 16, every: 1, dt,
});

// Optional alternative: interpolate position, preserve discrete age values.
const packedTrack = film.track('particles', particles, {
    stride: 16, every: trackEvery, dt,
    fields: [
        { word: 0, store: 'f32', interp: 'linear' },
        { word: 1, store: 'f32', interp: 'linear' },
        { word: 2, store: 'f32', interp: 'linear' },
        { word: 3, store: 'f32', interp: 'step' },
    ],
});
```

Return the chosen declaration in `tracks`, and include `{ name: 'particles', buffer: particles }`
in `checkpoint` if the buffer is persistent. These are declaration examples, not a complete scene.
Buffer sizes must be whole multiples of record stride. Source offsets (`word`) count four-byte words;
`stride` counts bytes. Track names must be unique lowercase letters/digits/hyphens.

- Track and checkpoint buffers need `COPY_SRC | COPY_DST`; packed sources also need `STORAGE`.
- Raw tracks omit `fields` and hold the previous saved bytes; they do not interpolate.
- Storage kinds are `f32`, `f16`, `unorm16`, `u8` and `u32`. Quantization can change results visually.
- Interpolation kinds are `step`, `linear`, `angle`, `wrap` and `hermite`. Discrete packed values
  select the nearer sample at the midpoint; raw holds and packed `step` are not identical.
- Wrapping phase requires `maxRate * every * dt < 0.5`. Hermite requires a tracked velocity word;
  an optional `jump` threshold prevents smearing large discontinuities such as respawns.
- Short-lived events may disappear between samples. Save them frequently or retain event history.

Start with raw tracks every step and complete checkpoints. Reduce precision and sampling only
after comparing replay against the original run. Avoid overlapping tracks that restore the same fields.

Checkpoints currently store declared GPU buffers, not arbitrary JavaScript objects or GPU textures.
CPU counters, RNG state, ping-pong buffer selection and texture-backed solver history need explicit
representation/restoration; the runner has no generic scene serialization callback. Rebuild scratch
buffers only when they are fully overwritten before being read.

## 5. Describe and run a shoot

After implementing `my-scene`, a minimal film JSON is:

```json
{
  "name": "my-scene-trial",
  "scene": "my-scene",
  "settings": {},
  "film": { "seconds": 2, "fps": 60, "width": 640, "height": 360 },
  "segmentSeconds": 1
}
```

Save it as `D:/orion-realms/null-graph/tools/film/films/my-scene-trial.json`. From
`D:/orion-realms/null-graph`, run these in PowerShell, stopping if any command fails:

```powershell
npm run build:lib
npm run film -- 'D:/orion-realms/null-graph/tools/film/films/my-scene-trial.json' --frames none --out 'D:/orion-realms/null-graph/tools/film/out/my-scene-trial'
npm run film -- 'D:/orion-realms/null-graph/tools/film/films/my-scene-trial.json' --out 'D:/orion-realms/null-graph/tools/film/out/my-scene-trial' --replay --frames png
```

The second command records state only. The third draws the complete recording to the recording's
`replay/frames/` directory. Omit `--frames png` to encode video with ffmpeg instead. Choose a fresh
output directory for a new job; use `--resume` only for compatible interrupted jobs.

Titles, captions and credits need nothing from the scene: list them as `overlays` in the film JSON
and the runner burns them into `film-labelled.mp4` after encoding (see the README's
[Text overlays](./README.md#text-overlays-burned-in-after-encoding)). They sit at fixed places on
the screen. Text that must follow an agent or an object belongs in the scene, drawn by the engine.

Replay keeps recorded scene settings authoritative. Camera, output size and playback speed can be
changed using the [replay options](./README.md#compute-records-replay-and-long-jobs-d22). Keep the code,
assets and dependencies pinned: external custom assets are not automatically bundled or hashed.
Resume provenance includes adapter/build/tool inputs, so migrating an interrupted job to a different
GPU is not a promised workflow. Keep declared buffer sizes independent of output size for replay.

## 6. Validate before scaling up

1. Record a short, small direct-render trial and a tracks-only trial with identical inputs.
2. Replay and compare images at fixed views. Inspect motion, animation, event timing and teleports,
   not merely whether an image is nonblank. Packed tracks need explicit acceptable error bounds.
3. Stop after one committed segment with `--stop-after 1`, resume with `--resume`, and compare final
   state against an uninterrupted run. This catches missing actions, clocks and solver history.
4. Check each `report.json` for `failure: null` and `gpuErrors: []`, and verify the expected completed
   segments in `progress.json`. A deliberately paused job is not a complete recording.
5. Estimate GPU memory and disk use from the small trial before increasing size or duration.

Offline rendering removes the real-time budget, not safety limits. The runner currently stops when
a measured `step()` exceeds 1.5 seconds. General draw calibration and automatic disk-space refusal
are not implemented. The bundled agent calibration is scene-specific, not a guard for custom solvers.
Do not bypass safety checks to run unbounded GPU submissions. Split expensive algorithms into bounded
submissions and validate them; an extremely expensive solver may need further runner/scheduler work.

The existing integration is `D:/orion-realms/null-graph/tools/film/scenes/herd-wolves.mjs`.
The generic GPU-record tests are in `D:/orion-realms/null-graph/test/film-gpu/records.js`.
Use them as references for buffer declarations and codecs, not as dependencies your scene must adopt.