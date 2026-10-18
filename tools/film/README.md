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

## CLI reference

Pass flags after npm's `--`. Supply exactly one film JSON file, including for replay, resume and
status. Value flags use a separate argument (`--seconds 5`, not `--seconds=5`). Unknown flags are
rejected; there is currently no `--help` flag. CLI overrides do not edit the film JSON.

| Argument / flag | Default | Usage |
| --- | --- | --- |
| `<film.json>` | Required | Film manifest: scene, settings, duration, frame rate, dimensions, camera and encoding settings. |
| `--frames ffmpeg\|png\|none` | `ffmpeg` | `ffmpeg` encodes segment MP4s and joins them into `film.mp4` on completion. `png` writes `frames/000000.png`, etc. `none` skips drawing/readback and records simulation tracks/checkpoints only; it does not save preview images. PNG/none do not require ffmpeg. This flag selects an output mode, **not a frame count**. |
| `--out <dir>` | `tools/film/out/<manifest name>/` | Recording destination. With `--replay`, this instead selects the existing recording to read. Relative paths resolve from the working directory. Existing films are not implicitly overwritten. |
| `--seconds <n>` | Manifest `film.seconds` | Override recording duration with a positive number of seconds. Duration times fps must be a whole number of frames. Replay uses the saved recording duration divided by `--speed`, so this does not trim replay. |
| `--size <W>x<H>` | Manifest dimensions; otherwise `1920x1080` | Override output dimensions, for example `--size 640x360`. Use lowercase `x` and positive, even integer dimensions (also required for PNG/none). Usable during replay. |
| `--track-every <n>` | `6` | Positive integer sampling interval in simulation steps, supplied to the scene. Smaller intervals save more samples and use more storage; scenes choose their track layouts. Resume inherits the saved interval when omitted; replay uses the recording's interval. |
| `--replay` | Off | Draw saved tracks without rerunning simulation. Scene and settings must match the recording. Default output is `<recording>/replay/`. |
| `--replay-out <dir>` | `<recording>/replay/` | Replay-only output destination, which must differ from the recording directory. Choose a new directory for each alternate render, or resume compatible output. |
| `--speed <n>` | `1` | Replay only; positive playback multiplier. `0.5` plays at half speed (twice the duration), `2` at double speed (half the duration). Output duration rounds up to a whole frame. |
| `--camera <file.json>` | Recorded camera | Replay only; replace the camera with a JSON **array of keyframes**, not a whole film manifest. Keyframes use `t`, `target`, `distance`, `azimuth` and `polar`, as described above. |
| `--resume` | Off | Continue compatible committed output; rewrite any incomplete segment. Keep the original manifest overrides and sink. Changed inputs are refused. Combine with `--replay` to resume replay output. |
| `--stop-after <n>` | No deliberate stop | Pause after a positive integer number of **total committed segments**, not additional segments or frames. Useful for testing resume. A paused successful run exits zero; it is not necessarily a finished film. |
| `--detach` | Off | Launch an independent Node process, append output to `film.log`, and save PID/start metadata in `job.json`. The launcher returning does not prove rendering finished. |
| `--status` | Off | Print JSON with output directory, process status, committed/total frames, ETA and segments, then exit without GPU initialization. Keep the original destination/replay flags. Manifest/build checks and sink discovery still run; for a PNG/none job, keep its `--frames` mode too. |

There are no CLI flags for fps, warm-up, segment duration, render scale, encoder quality or scene
settings: configure `film.fps`, `warmupSeconds`, `segmentSeconds`, `film.scale`, `encode` and
`settings` in the manifest. A frame must contain a whole positive number of simulation steps;
warm-up and segment durations must also align with the simulation/frame schedule.

### Environment variables and copy checking

These are environment variables, not film flags. In PowerShell set them before running npm:

| Variable | Usage |
| --- | --- |
| `NG_DAWN_FLAGS` | Select/configure Dawn, for example `$env:NG_DAWN_FLAGS = 'adapter=NVIDIA'` (or `adapter=AMD`). |
| `NG_FFMPEG` | Optional ffmpeg executable path; otherwise ffmpeg is searched for on PATH. Example: `$env:NG_FFMPEG = 'C:/ffmpeg/bin/ffmpeg.exe'`. |
| `NG_SITE` | Source test-site checkout for Herd & Wolves assets and scene-copy drift checks. Not required to author a standalone particle scene. |

`npm run film:check` compares bundled scene copies with the site. Its `--update` switch
(`npm run film:check -- --update`) **modifies those copies** to match available site sources;
review the site's changes first. It is a separate command, not a flag for `npm run film`.

### Single-frame PNG preview (PowerShell)

This renders one small GPU frame without encoding a video. At the Magnetic Star preset's 60 fps,
one frame is `1 / 60` seconds. Use a fresh output directory, or explicitly remove old output before
rerunning; the runner will not overwrite an existing film.

```powershell
Set-Location 'D:/orion-realms/null-graph'
$env:NG_DAWN_FLAGS = 'adapter=NVIDIA'
npm run build:lib
npm run film -- 'D:/orion-realms/null-graph/tools/film/films/magnetic-star.json' --frames png --seconds 0.016666666666666666 --size 640x360 --out 'D:/orion-realms/null-graph/tools/film/out/magnetic-star-single-preview'
```

Open `D:/orion-realms/null-graph/tools/film/out/magnetic-star-single-preview/frames/000000.png`.
This is the first rendered frame after warm-up and that frame's simulation steps, not an arbitrary
timestamp screenshot. Use `--seconds 0.1` for six PNG frames at 60 fps, or `--seconds 5` for 300.
PNG mode writes numbered frames; it does not automatically create `poster.png`.

### Common flag combinations (PowerShell)

After the setup above, these examples use independent destinations:

```powershell
# Five-second video, leaving the preset unchanged.
npm run film -- 'D:/orion-realms/null-graph/tools/film/films/magnetic-star.json' --seconds 5 --out 'D:/orion-realms/null-graph/tools/film/out/magnetic-star-video-example'

# Compute only, saving a sample every six simulation steps, in the background.
npm run film -- 'D:/orion-realms/null-graph/tools/film/films/magnetic-star.json' --seconds 5 --frames none --track-every 6 --out 'D:/orion-realms/null-graph/tools/film/out/magnetic-star-compute-example' --detach

# Check committed progress. Wait for all frames before starting replay.
npm run film -- 'D:/orion-realms/null-graph/tools/film/films/magnetic-star.json' --seconds 5 --frames none --out 'D:/orion-realms/null-graph/tools/film/out/magnetic-star-compute-example' --status

# Render the completed recording at half speed and a different size.
npm run film -- 'D:/orion-realms/null-graph/tools/film/films/magnetic-star.json' --out 'D:/orion-realms/null-graph/tools/film/out/magnetic-star-compute-example' --replay --speed 0.5 --size 1280x720 --replay-out 'D:/orion-realms/null-graph/tools/film/out/magnetic-star-slow-example'

# Pause after one committed segment, then continue with identical recording options.
npm run film -- 'D:/orion-realms/null-graph/tools/film/films/magnetic-star.json' --seconds 5 --frames none --out 'D:/orion-realms/null-graph/tools/film/out/magnetic-star-resume-example' --stop-after 1
npm run film -- 'D:/orion-realms/null-graph/tools/film/films/magnetic-star.json' --seconds 5 --frames none --out 'D:/orion-realms/null-graph/tools/film/out/magnetic-star-resume-example' --resume
```

Exit codes: `0` means successful completion **or a clean pause**; `1` means a runtime failure
(GPU error, encoding failure or step-time stop); `2` means invalid input or a pre-render refusal.
Check committed frame totals and the final `report.json` (`failure: null`, `gpuErrors: []`) rather
than treating a detached launch or zero exit code alone as proof of a completed film.

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
