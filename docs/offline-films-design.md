# NullGraph — offline films: compute records, render records, replay (D2) — Technical Design

**Parent:** [`work/plan.md`](./work/plan.md) §3, Cycle D: step D2, before Phase 2.

**Asked by the user** (2026-10-16):
- build "the recording infra" before Phase 2. A render or a brain may take a long time per frame,
  and the result should still play at 60 fps, stitched with ffmpeg;
- start with a formal definition;
- frames and state logs; Node first, with a browser export planned; Herd & Wolves first;
- after the first draft, the user asked for the design to be **general, not specific to agents**:
  "compute records and then render records". They also said 1M agents is a stress test, not the
  goal.

**Status:** revision 2, 2026-10-16. Revision 1 was agent-specific and aimed at 1M. Decided so far:
see §19.
- **D2.0** measured and committed 2026-10-16 (the spike `test/spikes/films/`; "Phase D2.0 results"
  at the end).
- **D2.1** built and committed 2026-10-16 (`tools/film/`; "Phase D2.1 results" at the end).
- **D2.2** implemented 2026-10-16 (layouts, tracks, checkpoints, replay, resume and detach;
  "Phase D2.2 results" below).

**Named films, not recordings:** `agents:record` already records the teacher's decisions (step 7).

---

## 1. Purpose

**Make a smooth, exact video of any NullGraph scene, however long each frame takes to compute,
and keep a record of the scene's changing GPU state that can be drawn again (any camera, size or
speed) without computing again.**

The core has two kinds of record on one timeline:
- **compute records:** declared GPU buffers, saved every m steps;
- **render records:** the frames `engine.render()` draws.

Agents are one user of compute records, not a special case.

**First targets:**
1. **A showcase film** of Herd & Wolves at the demo's counts (4K–64K), Models look, Network brains.
2. **A stress film** of 1M agents, last, as the demo treats 1M: a test of the limits, not the goal.

### Pre-existing facts this design rests on

1. **Headless rendering exists.** `npm run render` (`test/gpu/render.mjs`) renders a scene module
   under Node through Dawn into an offscreen `rgba8unorm` texture with `COPY_SRC`
   (`src/core/WebGPUContext.ts:34`, `:288–292`). It awaits the frame function (`render.mjs:92`) and
   passes `seconds = f / 60`, never the wall clock. `test/gpu/readback.mjs` reads the texture back.
2. **Agents already step exactly.** `AgentSystem` has a fixed-step clock (`FixedStep`,
   `agentMath.ts:226`): one `advance(dt)` selects exactly one step. `step()` (`AgentSystem.ts:514`)
   senses, **awaits** an ONNX Runtime brain's run, then acts, so no action is late. The tools step
   this way (`lockstep`, `tools/agents/label-pass.mjs:58`). Reruns are bit for bit on one GPU
   (step 5).
3. **Herd & Wolves runs headless** (`tools/agents/scenario/herd-wolves.mjs`; its rules are checked
   against the demo by `check-site.mjs`). The two networks are committed. ONNX Runtime runs under
   Node, and the engine adopts its device.
4. **The demo's look modules run under Node.** A probe on 2026-10-16 imported the site's `brain.ts`,
   `fence.ts`, `models.ts`, `field.ts`, `flashes.ts` and `options.ts` through Node 24's type
   stripping. Only `loadHerdModels` is browser-only: it uses `fetch` with root-relative URLs, and
   `TextureManager.load` needs `createImageBitmap` (`TextureManager.ts:56`).
5. **Text needs a font atlas, and Node cannot make one.** The demo builds its atlas with
   `FontAtlas.fromCanvas`, which needs `OffscreenCanvas`. Node can load one made beforehand
   (`FontAtlas.fromSDF`, `fromMSDF`; `src/extras/text/FontAtlas.ts:206`, `:223`).
6. **At 1M agents the neighbour queries dominate, and `thinkEvery` does not touch them.** The query
   stage runs for every agent on every step (`agentGridShaders.ts:31`). At 262,144 agents they were
   94% of a step (phase 10.5). An ONNX Runtime brain runs over every agent (`OrtBrain.ts:96`), and
   `step()` starts a run on every step (`AgentSystem.ts:527`).
7. **This machine:** a GTX 1650 (4 GB) and an AMD GCN-5 iGPU; a Ryzen 5 5600H with 7.3 GB of RAM;
   Node 24.12 (whose `node:zlib` has zstd); 64 GB free on D:. **ffmpeg 9.0.2** (gyan.dev essentials,
   installed by the user through Chocolatey on 2026-10-16), with `libx264` and `h264_nvenc`.
8. **A long run started from an agent's shell can be cut off** (gotchas). Long runs start detached
   and can resume.

### Non-goals

- Capturing the live demo in real time (screen recording, `MediaRecorder`): that records the stutter.
- **Camera authoring and input capture.** Scripted camera paths beyond simple keyframes, and
  recording a person's mouse moves to replay them, are tooling for later (the user, 2026-10-16).
- Audio and editing. Titles can be added with ffmpeg (§8).
- A library API in D2.0–D2.3: everything lives in `tools/film/`, outside the npm package. D2.4
  decides what, if anything, moves into the library.
- The same pixels on different GPUs or drivers. Determinism is per machine.
- Changing the simulation, the networks or the demo's look for a film.

### Principles

- **Time comes from the frame number, never the clock.**
- **Compute once, draw many times.** As Blender bakes a simulation and then renders it.
- **General core, presets on top.** The core tracks any buffer. Agents are a preset.
- **No engine hook.** The film tool calls `engine.render()` itself and reads the texture it drew.
- **Every long run can stop and resume,** losing at most one segment.
- **Show the cost before spending it.** Refuse only what would reset the GPU.

## 2. The formal definition

**Symbols.**
- **dt**, the scene's step length (1/60 s for Herd & Wolves).
- **r**, frames per second (60).
- **s** = 1/(r·dt) steps per frame, a whole number.
- **S_k**, the scene's GPU state after k steps: S_(k+1) = F(S_k). F is whatever the scene's
  `step()` does. S_0 is the scene after setup and W warm-up steps that are not filmed.

**A film** is the sequence of images

> I_j = R(S_((j+1)·s), C(τ_j), Q),  τ_j = j / r,  j = 0 … J−1

C is the camera, a function of film time. Q is the quality (size, look, supersampling). R is the
scene's draw.

- **Offline:** I_j has no deadline. The wall time of a frame never changes I_j.
- **Exact:** F runs as specified. An asynchronous brain is awaited inside the step, so actions are
  never late (in the live demo they land 1–2 frames late).
- **The invariant:** I_j depends on the film's inputs (settings, seed, W, j, C, Q) and on the
  machine (GPU, driver, build), and on nothing else.

**Records.** All are stamped with the step k and the frame j from one clock.
- **A compute record** (a *track*) of buffer B with layout L every m steps:
  T_B = { L(B at S_k) : k ≡ 0 (mod m) }. L lists B's fields and how each is stored and interpolated
  (§4). With no layout, the record is B's bytes, held between saves.
- **A checkpoint:** the full bytes of the declared buffers at S_c, at each segment's start.
- **A render record:** the frames I_j.

**A replay** draws I′_j = R(Π(T, k_j), C′(τ′_j), Q′) **without F.** Π writes every track,
interpolated at step k_j, back into its buffer. Checkpoint 0 restores everything that is not tracked
and never changes. C′, Q′, the frame rate and the speed may all differ from recording.

**A resume** restores S_c from a checkpoint and continues F. By determinism, it equals a run that
never stopped.

**Which scenes can be filmed, and which replayed:**
- **Filmed:** any scene that takes time only from the frame number (`npm run render`'s contract).
- **Replayed:** a scene in which every buffer that changes during the film is either recomputed
  from film time (camera, a sky or time of day driven by τ, a shader animated by the time uniform),
  or tracked. A scene declares its tracks: about 20 lines, not a new system.

**Properties, each one a test (§14):**

| | Property | Means |
|---|---|---|
| P1 | Determinism | The same film twice on one machine gives the same simulation state, tracks, checkpoints and frames, byte for byte, except where the scene declares otherwise. Herd & Wolves declares one exception (§10): which catches flash. D2.1 also found depth ties between overlapping animals changing winners, from the visibility cull's `atomicAdd` order. The stable draw order (visibility-design §14) removed them. |
| P2 | Resume | Stopped and resumed, a film equals an unbroken run. |
| P3 | Replay fidelity | A replay with every track at m = 1 matches the recorded frames within a pinned tolerance. At m = 6, each interpolated field is within a pinned error of m = 1's. |
| P4 | Capture fidelity | The video decodes to the captured image (orientation, channels, colour) within yuv420p's loss. |
| P5 | Safety | No submission predicted above 1 s of GPU work is sent (§12). |

## 3. What films give the brains (the MLP and ONNX Runtime)

1. **The policy as specified, at any cost.** Actions are never late, so Rules, Network and ORT are
   compared on what they decide. A bigger network, or Phase 2's ray sensors, can be filmed at 60 fps
   however slowly they run.
2. **Side by side on one seed.** Two films from one S_0 with different brains, and their tracks,
   show where a copy leaves its teacher.
3. **Measures at scale.** A track lets catches, speeds and herd sizes be measured over a whole film,
   on the CPU, after the run.
4. **Labelled images.** A frame and the tracks at its step are an image plus every agent's exact
   state: the start of the plan's "synthetic data factory", and of training data for Phase 2's
   agents that see.
5. **Phase 2's generations.** A checkpoint is a saved world: keep one per generation, and film any
   of them later.

Not: training data for today's networks. `agents:record` writes observations and labels.

## 4. The model: a clock, tracks, checkpoints and frames

### 4.1 The clock

The film tool owns the only clock: the frame j, its steps k, and its time τ. It calls the scene's
`step()` s times, then `draw(j)`. A warm-up runs W steps before frame 0.

### 4.2 Tracks (compute records)

A track names a GPU buffer, a record stride, a rate m, and optionally a **layout**: the fields to
keep. Each field has:
- its **word offset** in the source record;
- its **storage:** `f32`, `f16` (packed in pairs with `pack2x16float`), `unorm16`, `u8` (the low
  byte of a word), or `u32`;
- its **interpolation:**

| Kind | Between saves a and b (u = (k − a)/m) |
|---|---|
| `step` (the default) | a's value below u = ½, b's from ½ on |
| `linear` | a + u·(b − a) |
| `angle` | along the shorter arc |
| `wrap` | a + u·fract(b − a), for a value in [0, 1) that only moves forward, such as an animation phase. Exact while it moves less than half a cycle per save. |
| `hermite` | a cubic through a and b, with tangents m·dt times a named velocity field. A field may also declare a largest plausible jump, above which it is treated as `step`: a teleport, such as a respawn, is not smeared across the screen. |

**From a layout, the tool generates two compute stages:**
- **pack** reads the source buffer and writes the packed records, padded to whole words;
- **unpack** reads two packed saves and writes the interpolated fields back into the source buffer,
  touching no other word.

A CPU mirror of both is used by the tests and by analysis scripts.

**With no layout,** a track is a raw copy (`copyBufferToBuffer`), held between saves. This suits
small buffers saved every frame, such as a ring of events or counters.

**Presets** are named layouts. The first is `agents`, for any `AgentSystem` buffer (record of 64 B;
offsets from `agentLayout.ts`):

| Field | Word | Storage | Interpolation |
|---|---|---|---|
| position x, z | 0, 2 | f32 | hermite (velocity x, z; jump above top speed × m·dt) |
| velocity x, z | 4, 6 | f16 | linear |
| heading | 7 | f16 | angle |
| flags (`ids.y`, low byte) | 13 | u8 | step |

That is 15 B, padded to 16 B. A scene adds fields to a preset. Herd & Wolves adds the wolves' rest
timer (word 11, f16, step), which makes 20 B. The y position, radius, species and speed scale are
not tracked: they come from checkpoint 0.

### 4.3 Checkpoints

At the start of each segment, the declared buffers are copied whole. That includes everything F
reads before writing: for agents, the agents buffer, the actions (an agent not due keeps acting on
its last decision) and `state`, plus the scene's own buffers. Buffers that every step rewrites
before reading are not saved: observations, neighbours, the grid, reasons. Resume refuses a
checkpoint whose inputs hash (the simulation settings, the WGSL texts, the build) differs.

### 4.4 Render records (frames)

After each `engine.render(...)`:
1. `engine.screenTexture` is copied into one of three staging buffers;
2. the buffer is mapped;
3. the bytes go to the sink: ffmpeg, PNG, or none.

**`npm run drive` is a different tool, and it does not record films.** It runs the real site in
headless Chrome. It cannot step the demo exactly; its screenshots are of the whole page; and Chrome
here uses the AMD iGPU. It serves films in three ways:
- reference screenshots of the demo, to check the look by eye;
- baking the font atlas in a real browser (§8);
- testing the browser export later (D2.4).

## 5. API and commands

**A scene file** (`tools/film/scenes/<name>.mjs`):

```js
export default async function scene({ engine, lib, film, mode }) {   // mode: 'record' | 'replay'
    // Build what drawing needs; in 'record' also the simulation.
    return {
        step: async () => { /* one exact step; never called in replay */ },
        draw: (frame) => [ /* the render sequence; its last pass is the image */ ],
        camera,                                  // the Camera the film tool moves
        tracks: [
            film.track('agents', agents.buffer, { every: 6, preset: 'agents',
                add: [{ word: 11, store: 'f16', interp: 'step' }] }),
            film.track('phase', phases, { every: 6, fields: [{ word: 0, store: 'unorm16', interp: 'wrap' }] }),
            film.track('ring', ring, { every: 1 }),                    // raw, held
            film.track('stats', stats, { every: 1 }),
        ],
        checkpoint: [agents.buffer, agents.act, agents.state, phases, ring, stats],
        destroy() {},
    };
}
```

**A film file** (`tools/film/films/<name>.json`, committed):

```json
{
  "scene": "herd-wolves",
  "settings": { "count": 16384, "seed": 11, "thinkEvery": 3, "brains": "network", "look": "anime" },
  "warmupSeconds": 10,
  "film": { "seconds": 60, "fps": 60, "width": 1920, "height": 1080, "scale": 1 },
  "camera": [ { "t": 0,  "target": [0, 0, 0], "distance": 40,  "azimuth": 0.3, "polar": 0.6 },
              { "t": 60, "target": [0, 0, 0], "distance": 220, "azimuth": 0.9, "polar": 0.9 } ],
  "segmentSeconds": 10,
  "encode": { "codec": "libx264", "crf": 16, "preset": "slow" }
}
```

**Commands:**

```
npm run film -- tools/film/films/herd.json                 record: steps, tracks, checkpoints, frames
npm run film -- tools/film/films/herd.json --frames none   record the tracks only (a large count)
npm run film -- tools/film/films/herd.json --replay        replay: tracks → draw → frames
       --camera shots.json  --size 640x360  --speed 0.25   … with another camera, size, speed
npm run film -- <film> --resume | --detach | --status       long runs
npm run film:check                                         the copied site files against the site
```

Output goes to `tools/film/out/<name>/`, which git ignores.

## 6. Exact stepping (agents as the example)

- **Stage brains** (Rules, Network) step with the lockstep sequence: one `advance(dt)` selects the
  one-step chain, and the scene submits it once per `step()`.
- **ONNX Runtime brains** step with `await agents.step()`.
- **Steps and the draw are separate submissions** (§12).
- **`thinkEvery` keeps its meaning** (staggered by 64-agent block). Choosing it is §19, Q1.

## 7. Encoding

- **ffmpeg,** one process per segment, fed raw frames on stdin, respecting backpressure:

  ```
  ffmpeg -f rawvideo -pix_fmt rgba -s 1920x1080 -r 60 -i - -c:v libx264 -preset slow -crf 16 \
         -tune animation -vf scale=out_color_matrix=bt709,format=yuv420p \
         -colorspace bt709 -color_primaries bt709 -color_trc bt709 segments/0003.mp4
  ```

  Segments are joined by stream copy (`-f concat … -c copy -movflags +faststart film.mp4`).
  `h264_nvenc` is an option where encoding becomes the bottleneck, at small counts.
- Width and height must be even. Rows are padded to 256 B when the width needs it; 1920, 1280 and
  3840 do not.
- **PNG** (`test/gpu/png.mjs`) is the sink for checks. **None** is the sink for a tracks-only
  record.
- **Frame hashes:** a sha256 of each frame, stored in the segment's record (P1, P2).
- **Supersampling** uses `PostChain`'s `scale`: the scene renders larger, and the chain's last pass
  filters it down.

## 8. Text

The engine draws text on the GPU inside `engine.render()`, so it lands in the frame like everything
else.

| Text | In a film |
|---|---|
| World labels that follow a buffer (the wolves' names follow `agents.buffer`) | Work in record and replay. Replay writes the tracked positions into that same buffer. |
| The demo's live overlay (ms per step, runs a second) | Left out: in a film these measure the render. |
| A caption from the film's inputs and tracks (count, brains, catches, "rendered offline") | Optional, drawn in replay: change it and replay again. |
| Titles, credits, subtitles | Optional, with ffmpeg after encoding (`drawtext`, or a subtitle track). They cannot follow an agent. |

**The font atlas.** Node cannot draw glyphs (fact 5). The demo's atlas is baked once in Chrome:
`npm run drive` opens a small page that runs `fromCanvas` with the demo's settings (Inter SemiBold
600, 48 px em, range 16) and catches the description and bytes on their way into `fromSDF`. They are
saved outside git, like the models, and Node loads them with `fromSDF`.

## 9. Camera

- **Keyframes** `{ t, target, distance, azimuth, polar }`, in the demo's orbit terms, joined smoothly;
  or one fixed view. Near and far planes are set from the distance, as the demo sets them.
- Because replay is cheap, a camera is changed by editing the keyframes and replaying a 640 × 360
  preview.
- **Deferred** (non-goals): camera authoring tools, following an agent, and capturing a person's
  input to replay it.

## 10. The Herd & Wolves scene

- **Copies of the site's files,** decided by the user: `brain.ts`, `models.ts`, `field.ts`,
  `fence.ts`, `flashes.ts` and the font settings go to `tools/film/scenes/herd-wolves/`. They exist
  for development, in a folder the npm package never ships, as `tools/agents/` is.
  - Node runs them with type stripping. Imports of `null-graph/*` resolve to this repository's
    `dist/`.
  - `loadHerdModels` is adapted to read the GLBs and `colormap.png` from the site's `public/`
    (`NG_SITE`), never copied (the user's rule). Each adapted part is marked in the copy.
  - **`npm run film:check`** compares every unadapted copy with the site's file and lists the files
    that have changed there, so drift shows up instead of hiding.
- **Simulation:**
  - `herdWolves.create` (the scenario);
  - the brains: `MlpBrain` from the committed `.mlp.json`, or `OrtBrain` from `agents:export`;
  - the `animal-phase` stage;
  - `toInstances`.
- **Look:** Models mode, composed from the copies as the demo's `build()` composes it (site
  `index.ts:564–786`), without picking, the live overlay or the guards:
  - the post chain;
  - `createHerdField`;
  - cow and fox batches through the hook or the palette;
  - far cubes;
  - flashes;
  - optional names and caption (§8).
- **Tracks:**
  - `agents`, the preset plus the rest timer;
  - `phase`, with `wrap`. The phase moves at most 4 cycles a second, so m ≤ 7 keeps it exact;
  - `ring` and `stats`, raw, every frame.
- **One difference from the scenario to carry over.** The demo fills its catch ring with "caught
  long ago" (`-1e9`, site `index.ts:647–649`), so no flash appears at the origin. The scenario
  leaves the ring as zeros.
- **Declared exceptions to P1:**
  - **Which catches flash.** Above 4,096 agents, the respawn stage flashes one catch in N, chosen by
    `atomicAdd` order. Two recordings can therefore flash different catches, and the ring's slot
    order can differ at any count. The agents' state is unaffected, and a replay is exact because it
    plays the tracked ring.
  - **Depth ties, resolved.** D2.1 found that the visibility cull appended visible instances with
    `atomicAdd`. Where two animals overlapped with faces at the same depth, the winner changed from
    run to run: up to 0.53% of a frame. Since the stable draw order (visibility-design §14, the
    user's decision after D2.1), frames are exact; F1 requires it.

## 11. Costs

**Measured in D2.0** ("Phase D2.0 results" below). The steps held; the draw is 4–20× cheaper than
predicted here (14 ms at 1M).

Predictions for 1M use the demo's guard rule (ns per agent × N × (N/M)^0.25). D2.0 measures them.

**One step on the GTX 1650 under Node** (measured rows: ordered-grid design, "Whole step"):

| Agents | Field | Step | One minute (3,600 steps) | `agents` track at m = 6, per minute |
|---|---|---|---|---|
| 4,096 | 160 m | 0.3–0.8 ms, measured | 1–3 s | 49 MB |
| 16,384 | 320 m | — | — | 197 MB |
| 65,536 | 640 m | 14–26 ms, measured | 50–95 s | 786 MB |
| 1,048,576 (stress) | 2.56 km | 0.45–0.85 s, predicted | 26–50 min | 12.6 GB |

- **The showcase film** (4K–64K): record in minutes; replays at 1080p cost the draw (2–30 ms per
  frame at these counts, from phase 0) plus capture and encoding. Encoding likely dominates there,
  which is where `h264_nvenc` helps.
- **The stress film:**
  - 1M simulates only on the GTX 1650 here. The AMD iGPU is predicted at about 2.3 s per step, over
    the GPU reset; it could still *replay* a 1M track, since replay only draws.
  - `thinkEvery` 60 saves at most about 6% of a step at 1M, because the queries dominate.
  - ORT at 1M adds about 0.1 s per species per step (102 ms measured, phase 10.0) and over 1 GB of
    GPU memory.
- **RAM:** 7.3 GB. Tracks stream to disk as they are read back, never held a segment at a time.

## 12. The guard and long runs

- **The limit:** 1 s of GPU work per submission, half the time that resets the GPU. Each step and
  each draw is its own submission.
- **A count with no measurement on this adapter** is calibrated first at N/16 and N/4 (GPU
  timestamps). N is predicted from those, and refused above the limit with the prediction named.
  Costs are kept per adapter in `tools/film/out/costs.json`. D2.3's first slice uses maximum
  queue-completion wall timing, including warm-up, not GPU timestamps. Probes climb by at most
  fourfold from no more than 4K to N/4, predicting before every larger submission.
- **While running,** every submission is timed. One above 1.5 s stops the film after its
  checkpoint, with the reason.
- **Before starting,** the tool prints the predicted time, disk use (checked against free space with
  `statfs`) and GPU memory. Nothing is capped.
- **Long runs:** `--detach` restarts the tool as a detached process with its log in the film's
  folder; `--status` shows progress and the ETA; `--resume` continues.

## 13. Files

| File | Holds |
|---|---|
| `film.json` | the film file as run |
| `report.json` | provenance (commit, dirty tree, Node, Dawn and ffmpeg versions, adapter; the WGSL and copied-file hashes); timings per segment; GPU errors |
| `progress.json` | finished segments and their frame hashes |
| `segments/NNNN.mp4`, `film.mp4` | the render record |
| `tracks/<name>/NNNN.bin` | one chunk per segment: saves × records, little-endian. The layout is in `film.json`'s resolved form. |
| `checkpoints/NNNN.bin` | a header (buffer names, sizes, the inputs hash), then the bytes |

## 14. Tests

**Headless** (`test/film.test.mjs`):
- **U1:** the film file: defaults, and every refusal of §15.
- **U2:** layouts:
  - packing to words;
  - the pack and unpack WGSL against the CPU mirror;
  - each interpolation kind at its edges: hermite at u = 0 and 1, the shorter arc across ±π, `wrap`,
    the jump rule.
- **U3:** planning: frames to segments to checkpoints; resume from a `progress.json`; predictions.
- **U4:** the ffmpeg arguments; the camera keyframes.

**GPU** (`test/film-gpu/run.js`, `--only film`; Node on the NVIDIA GPU and on the AMD iGPU):
- **F1 (P1):** a 4K Herd & Wolves film recorded twice: the agents buffer after both, and every
  frame, equal byte for byte (flashes off; §10).
- **F2 (P2):** stopped after segment 1 and resumed, it equals F1, with Rules, Network and ORT.
- **F3:** a generic track. A scene of a few thousand moving points with its own made-up layout,
  recorded and replayed: proves the core needs no agent code.
- **F4 (P3):** a replay at m = 1 against the recorded frames; m = 6 against m = 1, with the error
  pinned per field.
- **F5 (P4):** a test pattern, encoded and decoded: colours within ±3 levels, not flipped or swizzled.
- **F6 (P5):** the guard, with a lowered limit, refuses by prediction and stops on a measurement.

Each is watched failing before it is trusted.

## 15. Errors

Named refusals; the exact texts are fixed in D2.1.
- ffmpeg not found (`NG_FFMPEG`, PATH), with `--frames png` as the alternative.
- fps not a whole number of steps; odd width or height.
- not enough disk: X GB needed, Y GB free.
- a count predicted over the 1 s limit, with the prediction and its source; a measured step over
  1.5 s, and where the film stopped.
- a layout with a field outside the record; an unknown storage or interpolation; a `wrap` field
  saved too rarely to be exact.
- the site or a model missing (`NG_SITE`).
- a resume whose inputs differ from the checkpoint's.
- `dist/` older than `src/` (the tools' existing check).

## 16. Engine touchpoints

**Required: none.** The tools use public API: `engine.render`, `engine.screenTexture`,
`createComputeStage`, the `AgentSystem` buffers, `step()` and `advance()`.

**Optional, later:**

| Change | Gives | Without it |
|---|---|---|
| A save and restore API on `AgentSystem` | checkpoints that cannot miss a buffer added later | the scene lists its buffers; F2 catches a missing one |
| Brain runs over only the agents that are due | ORT at `thinkEvery` n costs 1/n | every step runs all rows |
| `FontAtlas`: a browser-only bake that returns bytes without uploading | any tool can save an atlas for Node | the bake page catches them on their way into `fromSDF` |
| Flashes chosen by hash, not `atomicAdd` order | P1 with no exception for Herd & Wolves | §10's declared exception |
| A stable draw order in the visibility cull (compaction by visibility bits and counts, not `atomicAdd`) | bit-exact frames, and in the live demo no frame-to-frame flicker where overlapping animals tie in depth | **Done 2026-10-16** at the user's word (visibility-design §14); F1 now requires exact frames |
| A dispatch split across submissions | 1M simulated on the AMD iGPU | 1M only on the NVIDIA GPU |

## 17. Build order

| Phase | Builds | Exit |
|---|---|---|
| **D2.0** | A spike, no tool: on the NVIDIA GPU, steps at 16K/64K/256K/1M with `thinkEvery` 3 and 60; the Models draw at those counts; readback and the ffmpeg pipe at 1080p (`libx264` against `h264_nvenc`); a pack stage and its zstd ratio; peak GPU memory | Measured against §11; Q1 and Q6 decided on the numbers |
| **D2.1** | The core: clock, render records (capture, ffmpeg and PNG sinks, segments), the camera keyframes; the copies, `film:check`, and the Herd & Wolves scene in record mode | A 60 s showcase film at 16K, read by eye against the demo; F1, F5; U1, U4 |
| **D2.2** | Compute records: layouts, generated pack and unpack, raw tracks, the `agents` preset, checkpoints, `--replay`, `--resume`, `--detach` | F2, F3, F4; U2, U3; the showcase replayed with another camera |
| **D2.3** | The guard; text (the atlas baked through drive, names, caption); the 1M stress film by the record-tracks, then replay route | F6; the stress film and its report |
| **D2.4** (planned) | Browser export: a Record setting in site demos | Its own design (below) |

**D2.4's questions, for later:**
- `VideoEncoder`, detected with `isConfigSupported`; not offered where unsupported.
- `new VideoFrame(offscreenCanvas)` in the worker, so no readback.
- A muxer library in the site (Mediabunny, mp4-muxer's successor, to check then).
- Memory, or the origin-private file system, for long clips.
- The demo stepping with `step()` while recording.
- Whether tracks come to the browser too. If they do, the core's pack and unpack stages work there
  unchanged.

**Deferred:** camera authoring, following an agent, input capture; audio; films on a rented Linux
GPU; a library API.

## 18. Where this refines the plan

| The plan | This design |
|---|---|
| Cycle D has one step, D1 | Adds D2, before Phase 2. D1's phase 5.4 still waits for the user's animals. |
| §4 horizon: "synthetic data factory: not on this path" | Labelled frames are a by-product (§3); not claimed |
| Step 10's D3: the demo's rules are a checked copy of the scenario | The simulation is the scenario. The look is checked copies of the site's files (`film:check`). |
| Never commit 3D models | Models are read from the site's `public/` |
| §6: the npm package ships `dist/` only | Tools only |

## 19. Decisions

**Decided by the user, 2026-10-16:**
- **Network brains** for the films ("the brains one would choose network").
- **ffmpeg** installed by the user with Chocolatey (9.0.2, checked).
- **Tracks every 6 steps** by default ("keep track rate").
- **Copies of the site's files in this repository,** for development ("we should simply copy in that
  site path"; "this for development iterations only"), with `film:check` for drift.
- **General, not agent-specific:** compute records and render records, agents as a preset (accepted
  as proposed: "this sounds much better").
- **1M is a stress test,** not the goal: the first film is a showcase at the demo's counts.
- **Camera authoring and input capture** are left for later.

**Decided after D2.0's numbers** (the user, 2026-10-16: "thinkEvery can go for 3 and showcase can
go for 64k"):
- **`thinkEvery` 3** for the films.
- **The showcase at 65,536 agents,** with the rest of the bundle as recommended: 1080p at 60 fps,
  `libx264` medium at CRF 16, the anime look, a 10 s warm-up, and a two-keyframe camera from a close
  herd out to the whole field. Names and a caption come with text, in D2.3.

**Decided after D2.1** (the user, 2026-10-16, after watching the showcase: "it was nice"):
- **The camera's end distance is 600 m**; the path stays as it is.
- **The visibility cull gets a stable draw order** (§16's option), "that might solve other problems
  with cull too". This is a library change in `null-graph/visibility`, designed in its own section
  of [`visibility-design.md`](./visibility-design.md) before code.
- **Animals walking through each other** is a simulation matter: a collision or separation rule.
  It is not part of films, and not now.
- D2.1 committed; D2.2 started.

---

## Phase D2.0 results — 2026-10-16

**What ran.** `npm run spike:films` (`test/spikes/films/`), under Node/Dawn on the GTX 1650
(`NG_DAWN_FLAGS=adapter=NVIDIA`, driver 581.83), four cases, each in its own process.
- The simulation is the scenario with both species on the committed networks (`MlpBrain`). It is
  stepped as a film steps it: one step per submission, waited for, timed by GPU timestamps around it.
- The look is the demo's Models mode, composed from the site's own modules (imported through a
  resolve hook, `site.mjs`, spike-only), at 1920 × 1080 with the anime chain.
- Every run: zero WebGPU errors; every check passed. The case logs, JSON and pictures are in
  `test/spikes/films/out/` (git-ignored). Run with `NG_SPIKE_CASE=steps|draw|encode|pack`.

### One step (Network brains, GTX 1650)

GPU ms per step (wall time 0.1–0.5 ms more). Measured at spawn, then again after settling
(600 steps at 16K and 64K, 300 at 256K, none at 1M):

| Agents | `thinkEvery` 3 | `thinkEvery` 60 | 60 saves | Predicted (§11) | GPU allocated |
|---|---|---|---|---|---|
| 16,384 | 2.75 / 2.82 | 1.84 / 2.03 | ~30% | — | 3.4 MB |
| 65,536 | 16.71 / 16.25 | 15.27 / 12.52 | 9–23% | 14–26 | 13.5 MB |
| 262,144 | 127.3 / 91.8 | 121.4 / 94.4 | about none | 80–150 | 55.3 MB |
| 1,048,576 | 618.7 | 596.8 | 3.5% | 450–850 | 233 MB |

- **1M simulates at 0.62 s a step:** a minute is 37 minutes, plus 6 for a 10 s warm-up. The guard
  predicted 720 ms from 256K: 1.16× high, the safe side.
- **`thinkEvery` 60 is a real saving at small counts, and almost none at 1M.** With the network
  brain, thinking is about a third of a 16K step. At 1M the queries dominate, as §4 said. The
  revision-1 claim "at most about 6%" holds at 1M only.
- **A settled field steps faster than a fresh one** above 64K: 92 against 127 ms at 256K. The
  guard's prediction from spawn costs is therefore conservative.

### The draw (the Models look, 1920 × 1080)

GPU ms per frame. The views are the demo's orbit at 40 m (close), and 190 and 340 m scaled with
the field (default, far):

| Agents | close | default | far | Animal path | GPU allocated (simulation + look) |
|---|---|---|---|---|---|
| 16,384 | 3.9 | 4.0 | 3.9 | palette | 79 MB |
| 65,536 | 3.0 | 3.4 | 3.2 | palette | 140 MB |
| 262,144 | 3.3 | 5.5 | 5.1 | hook | 237 MB |
| 1,048,576 | 4.9 | 14.1 | 13.8 | hook | 781 MB |

- **The draw is cheap at every count: 14 ms at 1M,** against §11's prediction of 50–300 ms. LOD
  makes far animals cubes, and a close view culls almost everything.
- **So at 1M a film can draw its frames during recording for about 2% more time.** Replay is for
  changing the camera, not for saving time. §10.3's "tracks first, then replay" route stays an
  option, not a necessity.
- **Read by eye** (`draw-*.png`): the demo's cows and foxes, grass patches, toon field and anime
  ink, at every count. The far views show the field with animals as specks.

### Capture and encoding (16K, the default view, 300 frames)

Each frame is stepped, drawn, read back through three staging buffers (one frame in flight while
the previous is written), and piped to ffmpeg 9.0.2. Three runs:

| Sink | Frames a second | Per frame, waiting on readback / on the encoder | Output |
|---|---|---|---|
| none (readback only) | 159–168 | 3.9–4.2 / 0 ms | — |
| `libx264` slow, CRF 16 | 41–50 | 4.2–6.8 / 5.7–6.5 ms | 15.7 Mbit/s |
| `libx264` medium, CRF 16 | 55–66 | 4.0–5.2 / 3.5–4.3 ms | 15.7 Mbit/s |
| `h264_nvenc` | unavailable | — | — |

- **At showcase counts the CPU encoder is the bottleneck:** a 60 s film at 16K records in about 75 s
  with `slow`. Medium gives the same bitrate at this CRF, 30% faster.
- **`h264_nvenc` cannot run here.** ffmpeg 9's NVENC needs API 13.1, which comes with NVIDIA driver
  610 or newer; driver 581.83 has 13.0. The spike now reports ffmpeg's reason and goes on.
- **ffmpeg 9 tags colour from the frames.** The output flags `-color_primaries` and `-color_trc` left
  both "unknown". The `setparams` filter tags them, and `ffprobe` then shows BT.709 for the matrix,
  primaries and transfer.
- **A decoded frame matched the capture** (frame 150, `decoded-frame150.png`): orientation, colours,
  and the herds formed after the warm-up. This is the eye check that F5 will turn into a number.

### Packing a track (the 20-byte agents record of §4.2)

| Agents | A save | Pack (GPU) | Readback | zstd level 1 / 3 / 9 | Level 3 on saves XORed with the previous |
|---|---|---|---|---|---|
| 16,384 | 0.3 MB | 0.33 ms | 0.3 ms | — / ×1.56 / — | — |
| 65,536 (settled, 10 saves) | 1.3 MB | 0.26 ms | 0.6 ms | ×1.52 / ×1.56 / ×1.80 | ×1.78 |
| 262,144 | 5.0 MB | 1.2 ms | 1.9 ms | — / ×1.65 / — | — |
| 1,048,576 (2 saves) | 20 MB | 1.0 ms | 9.3 ms | ×1.53 / ×1.67 / ×1.88 | ×1.74 |

Speeds were 240, 120 and 50 MB/s for zstd levels 1, 3 and 9.
- **A save costs almost nothing on the GPU,** and its readback is 9 ms at 1M, every 6 steps of
  620 ms.
- **Compression helps a little:** 1.5–1.9×. The 1M minute's track is 12 GB raw, and about 7 GB at
  level 3. Level 3 costs 170 ms per 1M save on the main thread, and would go on a worker thread.
  XOR with the previous save adds little at level 3.

### What changes in the plan

1. §11's draw prediction was 4–20× too high; the step predictions held.
2. The showcase's count is free, so the choice is about the look:
   - 16K records a minute in about 75 s at the measured rate;
   - 64K is predicted at about 90 s (a 16 ms step, a 3 ms draw and 4 ms of readback per frame;
     not measured with the encoder).
3. NVENC needs a driver update. Until then, `libx264`.

### Not run

- The AMD iGPU (D2.0 measured the NVIDIA GPU, as §17 says), and Chrome.
- ORT brains: phase 10.0's 102 ms per run at 1M stands.
- A settled 1M field: 300 settle steps would take 3 minutes. The spawn cost is the higher one at
  256K.
- The flashes: the scenario does not hand out its catch ring yet. That comes in D2.1.

### The questions, on these numbers (decided: A, and 64K)

1. **`thinkEvery`:**
   - **A. 3** (recommended): the demo's default, and the networks' evaluated setting. What 60 saves
     comes where steps are already cheap (0.9 ms at 16K) and is about 3.5% at 1M, while agents
     would act on decisions up to a second old.
   - **B. 60,** your "one think a second".
2. **The showcase:**
   - **its count:** 16K or 64K, recorded in about the same time. 64K is recommended: denser herds in
     every view;
   - 1080p at 60 fps, `libx264` medium at CRF 16 (the same bitrate as slow here, 30% faster);
   - the anime look; a 10 s warm-up; a two-keyframe camera from a close herd out to the whole
     field; names in close shots and a caption.

---

## Phase D2.1 results — 2026-10-16

**Built** (`tools/film/`, its README; not committed yet):

| File | What it does |
|---|---|
| `film.mjs` | `npm run film`: records a film (clock, exact steps, camera, draw, capture, sink, segments, join, `progress.json`, `report.json`) |
| `manifest.mjs` | the film file, every default, every refusal by name; the plan (frames, steps per frame, warm-up, segments) |
| `camera.mjs` | keyframes in orbit terms, eased at the ends, Catmull-Rom slopes inside, distance on a log scale; keyframes hit exactly |
| `sinks.mjs` | ffmpeg (one process per segment, raw RGBA on stdin, `setparams` colour tags, ffmpeg's own reason on failure), PNG, none; the join by stream copy |
| `capture.mjs` | readback through three staging buffers, one frame in flight, padded rows unpacked |
| `check.mjs` | `npm run film:check [-- --update]`: the copies against the site, by hash |
| `resolve.mjs` | runs the copies as the site does: `null-graph/*` resolves to this repository's `dist/` |
| `scenes/herd-wolves.mjs` | the scene: the scenario, the networks, the look; the guard (below) |
| `scenes/herd-wolves/*.ts` | `brain`, `fence`, `field`, `flashes` and `models`, byte for byte from the site (its `brain.ts` and `flashes.ts` are not in the site's git: the check compares content, not commits) |
| `scenes/herd-wolves/assets.mjs` | the adapted half of `loadHerdModels`: GLBs and palette from the site's `public/` |
| `scenes/herd-wolves/look.mjs` | the Models look composed from the copies, flashes over the post chain |
| `films/herd-showcase.json`, `films/herd-test.json` | the decided showcase (64K); a 2 s, 640 × 360 test film |

Elsewhere:
- **The scenario** (`tools/agents/scenario/herd-wolves.mjs`) now returns its catch ring. Its texts
  and their hashes are unchanged.
- **`.gitattributes`:** the copies are `-text`. The site mixes LF and CRLF files, and a checkout must
  not change their bytes.
- **The GPU registry** gains `film`, marked `node`; the Chrome runner skips suites so marked.
- **npm scripts:** `film`, `film:check`, `test:film`.

**The guard came forward from D2.3,** because every size option must be guarded. A count above
65,536 is calibrated at a sixteenth and a quarter of itself, and refused above 1 s per step. Both
ways were run on the GTX 1650:
- **256K allowed:** 16K stepped in 4.0 ms and 64K in 19.9 ms, so about 113 ms was predicted; the
  film ran at 142 ms a frame, draw included.
- **4M refused, exit 2:** 256K stepped in 125.8 ms and 1M in 622.5 ms; "~3521 ms per step
  predicted on this GPU (594 ns per agent at 1,048,576), over the 1000 ms limit per submission".

D2.3 adds the per-adapter memory of costs.

**Tests:**
- **Headless, 7 of 7** (`npm run test:film`): U1 (defaults; 19 refusals; the plan; the scene's
  settings), U4 (ffmpeg's arguments; the camera path), and the drift check against a made-up site.
  Watched failing under four breakages, each caught: colour tags dropped, the path not eased at its
  ends, the drift check blind to content, odd sizes accepted.
- **GPU, 9 of 9 on Node/NVIDIA and on Node/AMD** (`--only film`), zero WebGPU errors. Watched failing
  under three breakages, each caught: padded rows not unpacked, colour tags dropped, a seed that
  changes between runs.
  - **F1:** the agents buffer is equal after two films. Frames differed in 37–53 of 90 by depth ties,
    the worst by 0.39–0.53%.
  - **F5:** the pattern reads back exactly, also at 330 px wide with padded rows. Decoded, the worst
    channel is 2 levels off; the stream is tagged BT.709.
- `npm run test:agents`: 83 of 83 after the scenario change.

**The finding: frames are not bit-exact, though the simulation is.** Two films ended with the same
agents buffer, but up to half their frames differed:
- the faithful chain by 1–4 pixels a frame, the anime chain by up to 237–307 at 320 × 180;
- the cause is the visibility cull, which appends visible instances with `atomicAdd`. Draw order
  changes from run to run, and so does the winner of a depth tie between overlapping animals (far
  cubes share a height). The anime chain draws its ink from depth, which widens each tie;
- P1 now says so (§2, §10). F1 holds the state exact and the frames under 1%;
- a stable draw order in the cull would make frames exact. It would also stop the matching flicker
  in the live demo (§16, optional).

**The showcase**, as decided: 65,536 agents, both species on the networks, `thinkEvery` 3, 60 s at
1080p60, `libx264` medium at CRF 16, anime, a 10 s warm-up, and the camera from 25 m out to 800 m.
- Recorded in 131.6 s on the GTX 1650: the warm-up in 10.8 s, then 25.7–38.1 frames a second. Steps
  took 21–26 ms a frame of wall time, which includes waiting for the previous frame's draw and copy.
- `film.mp4`: 3,600 frames, 60.000 s, BT.709, 274 MB (38 Mbit/s). Close views of 64K animals with
  ink lines are detailed; a shared copy would use CRF 20–23.
- **Read by eye** at 0, 15, 30, 45 and 60 s: herds forming and foxes hunting up close; herds as
  white swarms; the whole fenced field. The end distance overshoots: the field is whole from about
  45 s and small at 60 s. An end near 600 m would hold it.

**Not run:**
- Chrome: the suite is Node only, by design.
- The agent-tools GPU suite, after a change that only adds a returned field.
- Films on the AMD iGPU, other than F1 and F5.
- ORT brains (refused by name until a scene makes ONNX Runtime's device).

---

## Phase D2.2 results — 2026-10-16

**Handoff recovered:** the preceding agent stopped after implementing stable visibility order and
updating most of its documentation; no compute-record implementation had been left behind. Existing
visibility/library changes and unrelated working-tree files were preserved, not committed here.

**Built, tools only:**
- `tools/film/layout.mjs`: validated layouts, the agents preset, f32/f16/unorm16/u8/u32, CPU mirrors,
  generated GPU pack/unpack, all five interpolation kinds, untouched-word preservation, wrap-rate
  validation and teleport thresholds. Packed records align each field naturally and pad to words;
  Herd's agents record is 20 bytes.
- `tools/film/records.mjs`: GPU codecs; segment-local binary track chunks and sample-step sidecars;
  full-byte checkpoints with names, sizes, step and input hash; atomic metadata replacement;
  bounded-memory track playback. Resume validates a contiguous committed segment prefix.
- `tools/film/film.mjs`: records tracks alongside frames (or tracks only with `--frames none`);
  `--replay`, alternate camera/size/fps, `--speed`, `--replay-out`; `--resume`, `--detach`, `--status`,
  `--stop-after`. SIGINT/SIGTERM stop at a segment boundary. Output locks prevent concurrent writers;
  old recordings are never deleted implicitly. Detached logs and PID metadata remain in the output.
- Herd tracks agents, phase, ring, stats **and the GPU state/tick**: flashes need that clock during
  replay. Derived instance rows are rebuilt at draw time, without simulation steps. Checkpoints also
  include actions. Added COPY usage to scenario ring/stats and the look's phase buffer, no rule changes.
- `tools/film/ort.mjs`: existing Node ORT bootstrap pattern; the engine adopts its device and both
  committed networks run through OrtBrain in lockstep. ORT counts above 65,536 are explicitly refused
  pending ORT-specific guard calibration in D2.3 (the stage-brain prediction does not cover ORT).

**On-disk details:** checkpoint 0 is post-warm-up; checkpoint N is the start of segment N; the final
checkpoint stores the end state. Each track chunk has `.bin.json` sample steps, including both segment
boundaries (so a short final interval replays correctly). `film.json` stores resolved layouts and the
input hash. Progress commits only after frames, track chunks and the next checkpoint complete. A killed
run rewrites only the incomplete segment. Resume hashes compiled library, tool/scene and scenario
code, settings, layouts, networks and adapter; packed tracks are never used to continue simulation.

**Validation:**
- Headless U1–U4: **10/10**. Library/TypeScript and IDE builds pass; focused IDE inspections report no
  errors. `film:check`: all five copied files equal the site. Agent regression: **83/83** node tests.
- Node/NVIDIA: **31/31 film GPU checks**, zero GPU errors. Node/AMD: **31/31**, zero GPU errors
  (final log verified 2026-10-16, including the explicit respawn-endpoint assertion). F1 remains byte-exact with
  flashes off; F5 remains within 2 colour levels.
- U2 GPU pack matches CPU byte-for-byte for every storage kind over 2,048 made-up points. Unpack
  matches at u=0, .25, .5 and 1 within 0.0001, preserving untracked words and high flag bits. F3 streams
  generic point and raw tracks and replays without agent logic. Raw tracks hold the left save.
- F2 restores checkpoint state and continues identically for Rules, Network **and ORT**. Separate
  process CLI integration also compares resumed frame hashes and track bytes, refuses changed inputs,
  replays with zero stepping, and completes detached jobs with readable status.
- F4 m=1 packed replay is **not bit-exact**, as expected from quantization: NVIDIA 823 changed pixels
  out of 57,600 (1.43%), mean RGBA channel error 0.0154 levels; AMD 780 (1.35%), mean 0.0131. Pinned
  limits: ≤2% changed pixels and mean channel error ≤0.5 at this view.
- F4 six-step continuous midpoint error bounds: x/z ≤0.03 m (measured 0.00284/0.00340), velocity x/z
  ≤0.3 m/s (0.2951/0.2081), heading shortest-arc ≤0.3 rad (0.1997), rest timer ≤0.06 s (0.0516).
  Catch/respawn intervals are discontinuities, tested by endpoint selection instead of those bounds.
  The initial failing tests caught an incorrectly reused one-step jump threshold in the six-step
  test and an unrealistic bit/pixel-exact assumption for quantized replay; both are now explicit.

**Showcase:** a new complete 65,536-agent 60 s 1080p60 film with six-step tracks is at
`tools/film/out/herd-showcase-d22/film.mp4` (original D2.1 output preserved). Recorded in 201.4 s,
285.8 MB, zero GPU errors. This is a functional run, not an isolated benchmark: other validation ran
concurrently. The alternate camera file is `tools/film/films/herd-replay-camera.json`.
The alternate-camera replay completed in 44.6 s with zero simulation-step time and zero GPU errors;
`ffprobe` confirms 640×360, 60 fps, 60 s and 3,600 frames. The decoded 15 s PNG was visually inspected:
populated animals and catch rings, no obvious blank frame or corruption. CLI integration uses 4,096
agents for Rules, Network and ORT; its slow-replay assertion checks 96 frames at 120 fps and speed 0.25.
The fresh CLI rerun passed in 57.1 s (`tools/film/out/d22-validation/cli-verified.log`, empty stderr).
Final `npm run build:lib`, IDE build, `npm run test:film` (10/10) and `npm run film:check` (five copies)
passed on 2026-10-16. Phase acceptance/commit remains a separate decision; D2.3 scope is unchanged.

**Still D2.3:** persistent adapter costs and stronger guard calibration, disk/memory prediction and
free-space refusal, text/atlas/names/caption, the 1M stress film. Tracks are uncompressed; compression
and buffered/pipelined track I/O are optimizations, not required for correctness. Browser export
remains D2.4. No models or generated film artifacts are committed.

## Phase D2.3 started — 2026-10-16

First slice: `tools/film/guard.mjs`, shared measured step stop, persistent Rules/Network costs in
`tools/film/out/costs.json`. Cache identity includes adapter/driver, OS, Node/Dawn, flags, seed,
brain/think cadence, compiled core/agents, scenario, scene/guard code and networks. Costs retain
the maximum at each size. Probes start at no more than 4K and grow by at most fourfold to N/4;
compatible quarter-size evidence avoids calibration in a later process. Timings include warm-up
and queue completion, deliberately labelled wall measurements, not GPU timestamps.

Validation: library and IDE builds pass; headless film tests 14/14, including lowered-limit
prediction refusal before probing and measured stop with injected timings. Fresh-process CLI
regression passes in 57.0 s (Rules/Network/ORT resume, replay, detach/status).
Real 131,072-agent Network trials, six steps, tracks only, on both adapters:
- NVIDIA: predicted 324 ms from 32K; actual step averages 86/52 ms on first/cached runs.
- AMD: predicted 195 ms from 32K; actual averages 264/262 ms. **The rule underpredicts here.**
- Both subsequent processes reuse their own cache; all four reports have no failure/GPU errors.
- Evidence: `tools/film/out/d23-validation/` (logs and reports). These are smoke checks, not proof
  of safety for every scene or count. F6 policy tests inject timings; no deliberately unsafe GPU
  submission was made.

Still open: time/disk/GPU-memory estimates and free-space refusal, draw guard, ORT-specific
calibration (above 64K still refused), atlas/names/captions, then the 1M tracks-first stress film.
D2.3 is not complete or accepted. No commit or push.
