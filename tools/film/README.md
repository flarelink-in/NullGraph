# Film tools

These tools record **offline films**: a simulation and its look, stepped exactly and drawn frame by
frame, however long each frame takes, and encoded as a smooth 60 fps video. They live in the
repository and are not part of the `null-graph` package.

The recording core works with declared GPU buffers, not just agents. To make your own simulation,
procedural animation or other renderable scene recordable, start with the
[general-purpose scene authoring guide](./SCENE-AUTHORING.md). It covers the factory contract,
fixed-step scheduling, tracks versus checkpoints, replay-safe drawing, and validation.
The bundled Herd & Wolves scene is one integration example, not a requirement for custom scenes.

The design, with every measurement, is
[`docs/offline-films-design.md`](../../docs/offline-films-design.md) (Cycle D, step D2).

**Built:** D2.2: render records, compute tracks, checkpoints, replay, resume and detached jobs.
**D2.3 started:** persistent per-adapter step calibration is implemented. Resource estimates, draw
guarding, text and the stress film remain open.

Above 65,536 agents, Rules/Network calibration climbs from at most 4,096 agents to a quarter of the
requested count, predicting before each probe. Maximum steady-state queue-completion wall times
are saved in `tools/film/out/costs.json`. Warm-up is stop-checked but excluded from the prediction
because first-use pipeline compilation is not recurring GPU work. Compatible quarter-size evidence is reused; adapter/driver,
Node/Dawn/OS, Dawn flags, brain settings, seed, compiled code, scenario and network changes invalidate
it. Delete this disposable cache to force recalibration. The cache keeps the worst observation at a
size. A marginal quarter-size refusal can be refined at half-size only if that probe is itself
predicted below the unchanged 1 s limit. It is not a guarantee: AMD validation underpredicted the actual 128K step. Actual steps still
stop above 1.5 s. ORT above 65,536 remains refused. This is step calibration, not draw calibration.

## What you need

- **Node and a GPU that Dawn can use.** Pick the GPU with `NG_DAWN_FLAGS=adapter=NVIDIA` or
  `adapter=AMD`.
- **ffmpeg** on PATH, or its path in `NG_FFMPEG`. On Windows: `choco install ffmpeg`.
- **A current build.** The tools run `dist/`, and refuse to start when it is older than `src/`:
  `npm run build:lib`.
- **For Herd & Wolves, the test site** (`NG_SITE`; the default is the local checkout). Its look comes
  from copies of the site's files (`scenes/herd-wolves/*.ts`). Its models and palette are read from
  the site's `public/` folder, never copied here.

## Recording a film

```
NG_DAWN_FLAGS=adapter=NVIDIA npm run film -- tools/film/films/herd-showcase.json
npm run film -- <film.json> [--frames ffmpeg|png|none] [--out <dir>] [--seconds <n>] [--size <W>x<H>]
```

- **Each frame:**
  - the scene takes its steps exactly, each one its own GPU submission, waited for;
  - the camera follows the film's keyframes;
  - the frame is drawn, read back, hashed, and sent to ffmpeg.
- **Segments:** each segment (10 s by default) is one `.mp4`. At the end the segments are joined into
  `film.mp4` without encoding again.
- **Output:** `tools/film/out/<name>/` (ignored by git) holds:
  - `film.mp4` and `segments/`;
  - `progress.json` (the finished segments and their frame hashes);
  - `report.json` (provenance, timings, every frame's hash, GPU errors).
- **Other sinks:** `--frames png` writes `frames/NNNNNN.png` instead. `--frames none` records tracks
  and checkpoints only: it skips drawing and image readback.
- **Quick trials:** `--seconds` and `--size` override the film file, e.g. `--seconds 2 --size 640x360`.
- **Exit codes:**
  - 0: the film is written and the device reported no error;
  - 1: a GPU error, a failed encode, or a step over 1.5 s;
  - 2: a bad film file or argument, or a refusal before the first frame.

### The film file

```json
{
  "name": "herd-showcase",
  "scene": "herd-wolves",
  "settings": { "count": 65536, "seed": 11, "thinkEvery": 3, "brains": "network", "look": "anime", "flashes": true },
  "warmupSeconds": 10,
  "film": { "seconds": 60, "fps": 60, "width": 1920, "height": 1080, "scale": 1 },
  "camera": [
    { "t": 0,  "target": [0, 0, 0], "distance": 25,  "azimuth": 0.2, "polar": 0.6 },
    { "t": 60, "target": [0, 0, 0], "distance": 600, "azimuth": 1.0, "polar": 0.45 }
  ],
  "segmentSeconds": 10,
  "encode": { "codec": "libx264", "crf": 16, "preset": "medium", "tune": "animation" }
}
```

- **`settings`** belong to the scene.
- **`warmupSeconds`** are simulated before frame 0 and not filmed.
- **`film.scale`** renders larger and filters down (supersampling, through the post chain).
- **`camera`** keyframes use the demo's orbit terms. Each value moves smoothly between keyframes and
  starts and stops at rest; distance moves on a log scale. With no keyframes, the scene's own view
  holds.
- **Defaults:** 60 fps, 1920 × 1080, `libx264` CRF 16 `medium`, 10 s segments, no warm-up.
- **Refused by name:** odd sizes, a frame rate that is not a whole number of steps, keyframes out of
  order, and the like.

## The Herd & Wolves scene

- **Simulation:** the scenario of `tools/agents/scenario/herd-wolves.mjs` (the demo's rules), with
  both species on the committed networks (`brains: "network"`) or on the rules (`"rules"`).
- **Look:** the demo's Models mode: the toon field, cows and foxes animated by speed, far cubes, the
  catch flashes, and the anime or faithful post chain.
- **Settings:** `count` (a multiple of 16), `seed`, `thinkEvery`, `brains`, `look` and `flashes`.
- **The guard.** A count above 65,536 is first calibrated on this GPU at a sixteenth and a quarter of
  itself. It is refused when one step is predicted above 1 s of GPU work: Windows resets a GPU that
  stays busy for about 2 s. While a film runs, any step over 1.5 s stops it.

## One million agents: five seconds or a full minute

`films/herd-million.json` uses 1,048,576 agents, Network brains, `thinkEvery: 60`, no warm-up,
1080p60, anime, CRF 20. The camera pulls back over the first five seconds, then continues slowly.
This is a normal clip, not a seamless loop. Use NVIDIA on this machine, not AMD or ORT.

PowerShell, from `D:/orion-realms/null-graph` (run each command after the previous one succeeds):

```powershell
Set-Location 'D:/orion-realms/null-graph'
$env:NG_DAWN_FLAGS = 'adapter=NVIDIA'
npm run build:lib

# Five seconds: compute tracks first, then render those tracks into a video.
npm run film -- 'D:/orion-realms/null-graph/tools/film/films/herd-million.json' --seconds 5 --frames none --out 'D:/orion-realms/null-graph/tools/film/out/herd-million-5s-mine'
npm run film -- 'D:/orion-realms/null-graph/tools/film/films/herd-million.json' --out 'D:/orion-realms/null-graph/tools/film/out/herd-million-5s-mine' --replay

# Full minute: omit --seconds 5 and use a NEW recording directory.
npm run film -- 'D:/orion-realms/null-graph/tools/film/films/herd-million.json' --frames none --out 'D:/orion-realms/null-graph/tools/film/out/herd-million-60s-mine'
npm run film -- 'D:/orion-realms/null-graph/tools/film/films/herd-million.json' --out 'D:/orion-realms/null-graph/tools/film/out/herd-million-60s-mine' --replay
```

The video is `<recording directory>/replay/film.mp4`. Replay reads the recorded duration, so a
five-second recording stays five seconds even though the preset defaults to 60 seconds. Do not
change the preset or rebuild/edit tool code mid-recording if you need to resume.

For a background job add `--detach`, then use the same command with `--status` instead to check it.
Wait for completion before starting replay. Add `--resume` to the original compute command to
continue a paused recording; keep the original duration and `--frames none`. A replay can likewise
resume with `--replay --resume`. Existing outputs are not overwritten: choose another directory.
Do not bypass a guard refusal. Keep other GPU-heavy applications closed while calibrating/recording.

The verified five-second clip is `D:/orion-realms/null-graph/tools/film/out/herd-million-5s/replay/film.mp4`
(57,408,541 bytes, 300 frames, 1080p60, BT.709). Compute took 169.8 s, replay/encode 52.1 s,
worst measured step 814.2 ms; both reports have no failure or GPU errors. Tracks/checkpoints use
1.87 GB. Sample frames and full ffmpeg decoding were checked. Budget roughly 35–45 minutes plus replay
for a minute; it is an estimate, not a completed 60-second measurement. With one-second checkpoints,
reserve about 30 GB free for the minute's tracks/checkpoints and encoding headroom. No automatic
disk-space refusal is implemented yet. Inspect final `report.json` for `failure: null` and empty
`gpuErrors`.

## Keeping the copies honest

```
npm run film:check                 # each copy: the same as the site's, changed there, or missing
npm run film:check -- --update     # copy the site's files again
```

- The copies are kept byte for byte (`.gitattributes` keeps git from changing their line endings).
- A film warns when a copy differs, and its report records each copy's hash and state.
- After `--update`, record a short film and look at it against the demo.

## Compute records, replay and long jobs (D2.2)

```powershell
$env:NG_DAWN_FLAGS = 'adapter=NVIDIA'
npm run film -- tools/film/films/herd-showcase.json --out tools/film/out/herd-showcase-d22 --detach
npm run film -- tools/film/films/herd-showcase.json --out tools/film/out/herd-showcase-d22 --status
npm run film -- tools/film/films/herd-showcase.json --out tools/film/out/herd-showcase-d22 --resume
npm run film -- tools/film/films/herd-showcase.json --out tools/film/out/herd-showcase-d22 --replay --camera tools/film/films/herd-replay-camera.json --size 640x360
```

- `--out` selects the **recording** directory, also in replay. Replays go into its `replay/`
  subdirectory; `--replay-out <directory>` keeps another shot without overwriting the first.
- Replay never calls `scene.step()`. `--speed 0.25` makes a quarter-speed replay; change the film
  file's fps for another output rate. The saved simulation settings and duration stay authoritative.
- `--track-every 6` is the default; `1` saves every step (still quantized, not lossless).
  Resume reads the original track rate when the flag is omitted.
- Each track has a segment-local binary chunk and a `.bin.json` sample-step index. Layouts and
  resolved strides are saved in `film.json`. Packed records use generated GPU shaders; raw tracks
  are held between saves. Only two samples per track are read during replay, not the whole film.
- `checkpoints/0000.bin` is the post-warm-up initial state. Checkpoint N is the start of segment N;
  the last checkpoint is the final state. Checkpoints store complete bytes, never quantized tracks.
- Segment progress is atomically replaced **after** tracks, frames, and the next checkpoint finish.
  `--resume` rewrites an incomplete segment. Inputs include settings, layouts, scene/tool code,
  compiled library, scenario, network hashes and adapter. Changed inputs or sinks are refused.
- Ctrl+C / SIGTERM requests a stop after the current segment. `--stop-after N` deliberately stops
  after N total committed segments (useful for verifying resume). A killed process loses at most
  one segment; temporary files are ignored and overwritten. An output lock prevents concurrent writers.
- `--detach` starts an independent Node process, logs to `film.log`, and writes `job.json` with its
  PID. `--status` prints committed progress without initializing the GPU. Existing output is never
  deleted implicitly: resume it or choose another directory.
- `brains: "ort"` now runs both committed networks through ONNX Runtime, with the engine adopting
  ORT's device. Above 65,536 agents it is refused until D2.3 has ORT-specific guard calibration.

### Declaring a scene

See [Designing a filmable scene](./SCENE-AUTHORING.md) for the full workflow and current limitations.

The tool supplies `film.track(name, GPUBuffer, options)` to a scene factory. Return `tracks` and
`checkpoint: [{ name, buffer }]` alongside `dt`, `step`, `view`, `draw`, and `destroy`.
`options` take `stride` in bytes, `every` in steps, and `fields` with source word offsets,
`store`, and `interp`; or `preset: 'agents'` and `add`. Raw tracks omit fields.
Source buffers must support `COPY_SRC | COPY_DST`; packed sources also need `STORAGE`.
Declare every persistent changing buffer needed by stepping or drawing. Derived draw buffers must
be rebuilt in `draw()`. A `wrap` field declares `maxRate` so the half-cycle rule can be validated;
Hermite declares its tracked `velocity` word and optionally an absolute `jump` threshold.

The agents preset stores x/z, velocity x/z, heading and low-byte flags. Herd adds rest time, phase,
ring, stats **and GPU tick/state** (flashes age against that clock). Catch/respawn instants between
saves are deliberately stepped, not reconstructed; continuous-motion bounds do not apply across
those discontinuities. Packed m=1 replay is tested at ≤2% changed pixels and mean channel error
≤0.5 levels at the pinned view. Exact state continuation uses checkpoints, not packed replay.

## Tests

- `npm run test:film`: the film file and its plan, the camera path, ffmpeg's arguments, the drift
  check. Headless.
- `NG_DAWN_FLAGS=adapter=NVIDIA npm run test:gpu -- --only film`: GPU, Node only.
  - **F1, determinism.** Two films of 4,096 agents end with the same agents buffer, byte for byte.
    Every frame is byte-exact with flashes off, since the stable visibility draw order. Flashes
    retain the declared catch-order exception.
  - **F5, capture fidelity.** A four-colour pattern is read back exactly (also at a width whose rows
    need padding), then encoded and decoded within ±3 levels, tagged BT.709.
- **F2–F4 / U2:** generic points, every storage kind, CPU/GPU codecs, raw holds, checkpoint resume
  for Rules/Network/ORT, m=1 image error and m=6 per-field bounds.
- `node --test test/film-cli.test.mjs`: real GPU, fresh-process CLI resume, frame hashes and track
  bytes, replay, incompatible inputs, detach and status. Allow about a minute.
