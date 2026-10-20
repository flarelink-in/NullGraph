# Agent training tools

These tools make a network that copies a rule brain: they record what the rule decides, train one
small network per species on the recording, and judge the copy in the running simulation. They
live in the repository and are not part of the `null-graph` package.

The design, with every measurement, is
[`docs/agents-cloning-design.md`](../../docs/agents-cloning-design.md). Loading a trained network
in an app is in [`docs/api/agents.md`](../../docs/api/agents.md), "A trained network".

## Co-evolution infrastructure (phase 14)

The cloning commands below remain unchanged. `scenario/herd-wolves-arenas.mjs` provides fresh
mirrored arenas with separate `legacy-v1` / exhaustive `contact-v2` identities. `es.mjs` is the
CPU-only optimizer foundation, not a training CLI. Design: `docs/agents-coevolution-design.md` §7.

- `noise(seed, generation, direction, n)` uses `es-tuple-box-muller-v1`: ordered u32 fields,
  separate radius/angle streams, open-interval Box–Muller uniforms and f64 outputs. Each vector
  is explicitly bounded to 2^24 parameters. Numerical episode-seed collision handling is separate.
- `centeredRanks(scores)` shares average ranks across ties; `gradient(directions, utilities, sigma)`
  expects interleaved plus/minus utilities and divides by `2 K sigma`.
- `createAdamState(master)` copies f64 state; keep one per species. `adamAscent(state, gradient,
  { rate, decay })` uses existing trainer beta/epsilon constants and coupled L2 ascent. Failed
  updates do not partially mutate state; zero gradients can still move via momentum or decay.
- `masterParams(network)` and `candidateNetworks(shape, master, direction, sigma)` use existing
  layer packing, validating canonical network dimensions and finite f32 candidates. Diagnostics
  include bit-equal plus/minus parameter collisions and perturbation/displacement norms.

No accepted training presets or optimizer hyperparameter defaults. Browser correctness and short
rollout smoke are not score-quality or learning acceptance. Schedule/archive/trainer/guard and
checkpoints are phase 14.3; evaluation (14.4) is below; research (14.5) remains.

### Evaluating a run (14.4)

```powershell
npm run agents:arena-eval -- --run tools/agents/out/coevo/<run> --generations 0,20 `
    --purpose validation --seed 7 --domain 14 --analysis-seed 99
npm run agents:crossplay -- --run tools/agents/out/coevo/<run> --generations 0,10,20 --anchors 0 `
    --purpose validation --seed 7 --domain 14 --analysis-seed 99
```

Both play a run's checkpoints in the environment they were trained in, read from the checkpoint's
manifest. Each matchup plays every held-out episode at once, episode *e* in tile *e* of one fresh
lane from tick zero, so all matchups of an evaluation pair by episode. `--episodes` defaults to 20,
the least the design allows; `--T` defaults to the run's episode length.

- **arena-eval** plays each listed checkpoint's prey against the rule's, the label's (the rule
  without wander) and the clones' wolves, those baselines' prey against its wolves, each baseline
  against itself, and the checkpoint against itself (`--no-clones` leaves the clones out). It
  compares, per species and baseline, the checkpoint's pairing with the baseline's own over the
  same episodes: prey are better with fewer catches, wolves with more. `--margin <catches>` adds a
  noninferiority verdict (H2) against a margin fixed before the results are read.
- **crossplay** plays every listed prey generation against every listed wolf generation and
  scores each generation against the `--anchors` (default: the first listed), fixed before the
  results are read.

Held-out seeds come from their own stream per `--purpose` (validation guides settings; final-test
only judges) and are checked against the run's training seeds. `--seed`, `--domain` and
`--analysis-seed` (the bootstrap's) are required, so a preregistration can name them. Checkpoint
models must match their manifest's hashes and the manifest the run's experiment. Output goes to a
fresh `tools/agents/out/coevo-eval/<name>/`: `plan.json` (drivers with model hashes, seeds, plan)
before the GPU is touched, `log.jsonl` with each matchup's raw catches as it finishes, and
`evaluation.json` with raw catches, 95% intervals (10,000 paired resamples) and, for crossplay,
the matrices and anchor scores. `status.json` says `complete` only after cleanup. `--name` or
`--out-dir` choose the directory; `replay()` in `coevolve-evaluate.mjs` recomputes every interval
and comparison from an `evaluation.json`'s raw catches.

Tests: `node --test test/agents-coevo.test.mjs test/coevo-arena.test.mjs` (also in `npm run test:agents`).

## What you need

- Node, and a GPU that Dawn can use: recording and evaluating run the simulation headless.
  Training runs on the CPU only.
- A current build. The tools run `dist/`, and refuse to start when it is older than `src/`:

  ```
  npm run build:lib
  ```

## The three steps

```
npm run agents:record                                # about 20 s; writes tools/agents/out/herd-wolves
npm run agents:train                                 # about an hour of CPU; writes …/herd-wolves-students
npm run agents:evaluate                              # about 2 minutes; writes evaluation.json beside the students
npm run agents:export                                # under a second; writes tools/agents/out/onnx/<species>.onnx
```

Everything under `tools/agents/out/` is ignored by git. The first three steps make a network; the
fourth writes it in another format.

### 1. Record

`npm run agents:record -- [--out <dir>] [--episodes 5] [--seed 11] [--steps 1800] [--stride 6] [--count 4096]`

- Runs Herd & Wolves in lockstep with the rule driving, 5 episodes of 30 s by default.
- Writes one row per decision: the observation, the label, and the action the rule took.
- Keeps every wolf row and one prey row in 16. The last episode is held out of training.
- On one GPU, the same build and seeds give the same files, byte for byte.

**The label is not the rule's raw action.** The rule also reads a random wander value, the agent's
index and the wolf's rest timer, which a network cannot see. The label is the rule's decision
without the wander and the index coin-flips. Rows of resting wolves are marked frozen and are not
trained on.

### 2. Train

`npm run agents:train -- [--data <dir>] [--out <dir>] [--species prey,wolf] [--hidden 64x64x64] [--epochs 200] [--seed 1] [--batch 256] [--rate 0.01] [--final-rate 0.0001]`

- Trains one `tanh` network per species, in plain JavaScript, on one core.
- The defaults are the settings that met the 0.05 error bar: 24→64→64→64→2 at 200 passes. They take
  about 45 minutes for prey and 20 for wolves. Pass `--species prey` and `--species wolf` to two
  processes to run them side by side.
- `--hidden 16x16 --epochs 2` finishes in seconds and is enough to check that the tools run.
- Writes `<species>.mlp.json` (the network) and `<species>.report.json` (errors per action and per
  reason code, and every pass's loss).
- The same recording and seed give the same network file, byte for byte.

**A run this long must not be tied to a terminal that may close.** Start it detached and send its
output to a file.

### 3. Evaluate

`npm run agents:evaluate -- [--students <dir>] [--runs T,C,S,Sp,Sw] [--episodes 5] [--seed 21] [--steps 1800] [--out <file>]`

Runs the simulation on seeds that no recording used, once per run:

| Run | Prey | Wolves | Answers |
|---|---|---|---|
| T | the rule | the rule | the reference, and how much seeds alone differ |
| C | the label's decisions | the same | what the label does when followed exactly |
| S | network | network | the copy |
| Sp | network | the rule | the prey's network alone |
| Sw | the rule | network | the wolves' network alone |

It reports catches in 30 s, each driver's mean absolute difference from the label on its own run,
mean speeds, and how tight the herds are.

- **The bars:** an error of at most 0.05 per species and action, and S's mean catches within 10% of
  C's.
- **Run C checks the tool:** it must differ from its own label by nothing. If it does not, the
  command exits 1.
- A missed bar is printed, and is not an error exit.

`--students tools/agents/models/herd-wolves` evaluates the committed networks.

### 4. Export to ONNX

`npm run agents:export -- [--students <dir>] [--out <dir>]`

- Reads every `<species>.mlp.json` in `--students`, by default the committed networks, and writes
  `<species>.onnx`: one `Gemm` and one activation per layer, opset 13.
- The input is `obs`, float32 `[batch, 24]`; the output is `act`, float32 `[batch, 2]`.
- The file's metadata holds the species and the observation and action names.
- The same network gives the same bytes. A Herd & Wolves network is 41 KB.
- **Nothing in the library reads these files yet.** ONNX Runtime does, and its actions equal
  `mlpForward`'s within 4.5e-6 on the measured GPUs.

## The committed networks

`models/herd-wolves/` holds the two trained Herd & Wolves networks, their training reports, and
their evaluation.

| File | Holds |
|---|---|
| `prey.mlp.json`, `wolf.mlp.json` | the networks, 24→64→64→64→2, about 180 KB each |
| `prey.report.json`, `wolf.report.json` | held-out error per action and reason code; every pass |
| `evaluation.json` | the five runs on seeds 21–25, on an AMD iGPU |

Measured: the prey's turn differs from the label by 0.034 on average and the wolves' by 0.011, and
the networks' catches are within 1% of the label's. Turning away from a wall is the weak branch:
0.105 for prey and 0.070 for wolves, on about 5% of decisions.

**They are tied to the scenario's label.** Each file stores a hash of the label's WGSL in
`trainedOn.teacher`. `npm run test:agents` fails when `scenario/herd-wolves.mjs` no longer gives
that hash; record and train again, then replace the files.

**They are valid at the demo's defaults only:** 4,096 agents, `thinkEvery` 3, speeds 3.2 and 4.2,
rest 5 s, flocking and fleeing on.

## The files

| File | Does |
|---|---|
| `scenario/herd-wolves.mjs` | The demo's definition, rule, app stages, spawn and settings. The rule's hidden inputs sit behind four functions, so the label text can leave them out |
| `label-pass.mjs` | A second think pass that writes labels into its own buffers; `lockstep()`; buffer read-back |
| `dataset.mjs` | A recording on disk: rows, the sampler, the held-out split, sizes and SHA-256 |
| `recorder.mjs`, `record.mjs` | What a recording run does, and its command line |
| `mlp-train.mjs`, `train.mjs` | The trainer (Adam, mean-squared error, seeded), and its command line |
| `evaluator.mjs`, `evaluate.mjs` | One run under a driver per species, and its command line |
| `model.mjs` | A `.mlp.json` written, and read back with its names checked |
| `onnx.mjs`, `export.mjs` | A network as `.onnx` bytes, and its command line |
| `cli.mjs` | What the command lines share |
| `scenario/arena-layout.mjs`, `scenario/herd-wolves-arenas.mjs` | Step 14's tiles: geometry, the contact-v2 catch rule, mirrored lane systems, the demo's model names |
| `es.mjs` | Step 14's evolution-strategies math: noise, ranks, the mirrored gradient, Adam |
| `coevolve.mjs` and `coevolve-*.mjs` | The co-evolution trainer: config, schedule and archive, guard, lifecycle, checkpoints, statistics |
| `coevolve-evaluate.mjs`, `coevolve-eval.mjs` | Arena-matched evaluation and cross-play (no Node imports), and their command line |
| `validate-coevolve.mjs` | The trainer's and the evaluator's GPU infrastructure regressions |

`scenario/`, `label-pass.mjs`, `dataset.mjs`'s row helpers, `recorder.mjs`, `evaluator.mjs`,
`model.mjs` and `onnx.mjs` import nothing from Node, so the GPU suites also run them in Chrome.

## Limits

### Co-evolution (step 14) — 14.3 closed 2026-10-20, 14.4 started

Trainer runs so far are infrastructure smoke only; no contact-v2 research settings or cross-GPU
equality are accepted, and 14.0's safety and quality gates are still open.

`npm run agents:coevolve -- --config <explicit-profile.json> --name <fresh-name>` requires every
simulation, optimizer, archive and seed field explicitly (see `test/fixtures/coevo-profile.mjs`
for a **test-only**, not recommended research profile). Contacts-on additionally requires all
eight solver overrides, including `mass: "equal"` or `"area"`; contacts-off uses `{}`.
Unknown/duplicate options, missing values, conflicting contacts flags and catch-counter overflow
are errors. `dt` controls both the agent kernels and contacts solver. Output labels/paths do not
affect experiment hashes; immutable models, generation manifests and ONNX are deterministic,
but provenance and wall-time logs intentionally differ between runs.

`environment` is `legacy-v1`, `contact-v2` or `contact-v3`. contact-v3 is contact-v2 with initial
prey spawns kept an explicit `spawnSeparation` (metres) from their tile's wolves, re-drawn from the
episode seed; it is the research environment the user chose, with 36 m tiles (design §5).

Models are written and read under the demo's feature names (`bounds.side`, `turn`, …), not the
arena stages' indexed ones, so `--start clones` loads the committed clones (pass `--hidden 64x64x64`,
their shape) and every checkpoint loads wherever the demo's networks do. Each `gen-NNNN/manifest.json`
holds the experiment its `configHash` hashes, so a checkpoint names its environment on its own.
After every `archive.every` completed updates **both** species append their current master;
`archive/index.jsonl` lists each snapshot's ID, generation and model hash, and the log records IDs
and evictions only. `run.json` adds the start models' hashes or seeds, device limits, OS, Dawn
version and the flags that picked the adapter.

`status.json` starts `incomplete` and becomes `complete` only after final GPU checks and cleanup.
Failures keep fsynced failure metadata/log entries; interruption leaves the original incomplete
marker. Checkpoints/logs are flushed with `fsync`; atomic rename avoids partially published
checkpoints. This is **not** a guarantee against filesystem or power-loss corruption (directory
metadata persistence is platform-dependent), nor a resume format. Existing output is never reused.

Calibration uses progressively doubled, layout-valid arena counts and separate ordinary/dense/
wall/coincident/frozen fixtures with candidate policy rows. Cache identity includes adapter,
driver, runtime, source/build, actual initial networks, geometry, grids and settings. Cached
maxima gate probes but never waive fresh calibration. Compilation completion is timed separately;
no hidden warm-up steps enter scored episodes. Predictions gate every probe, submissions target
500 ms using floor division, back off after overruns, and stop above 1500 ms. **No post-hoc guard
can prevent the first unexpected stall or guarantee protection from an OS GPU reset.**

Run `node tools/agents/validate-coevolve.mjs` for fresh contacts-on ten-generation infrastructure
regressions on the selected `NG_DAWN_FLAGS` adapter, byte repeatability across distinct directories,
GPU timestep/finite-state checks and a deliberate validation-error/incomplete-run test. It also runs
a clone start with two replicas and an archive small enough to evict, and counts mappable buffers on
the trainer's device (none while calibrating, exactly one per generation: I6). `--only clones` runs
that part alone. This does not establish research acceptance; its report identifies the
intentionally injected error.

`coevolve-stats.mjs` (14.4) holds the evaluation statistics: a percentile bootstrap of means and of
paired differences over the same held-out episodes (10,000 resamples, nearest-rank 95% bounds, an
explicit analysis seed), strict superiority and margin-based noninferiority. It imports nothing from
Node. `buildHeldOutManifest` in `coevolve-schedule.mjs` makes validation or final-test seeds,
checked against the evaluated runs' training seeds.

Known limits of the trainer:

- **The engine keeps every compute stage it creates.** Fresh lanes each generation leave 88 stages
  behind; process memory grew about 1.6 MB per generation on NVIDIA over 60 generations. Fine for
  hundreds of generations, not for unbounded runs; the remedy is a library change or pooling.
- **Randomness after spawn is the scenario's own.** The rule's wander and every respawn draw depend
  on body index and tick, not the episode seed, so two episodes in the same tile differ only in
  where bodies start. Held-out seeds hold out spawns, not those draws.
- **The finite-state check runs last in a step.** A position the contact solver made nonfinite can
  be clamped by its world clamp or tile containment before the check sees it; velocities and
  timers are not clamped.
- **Calibration's cache key includes seeds and weights,** so evidence carries over only to a rerun
  of the same experiment. It never waives fresh probes either way.
- **In the test profile's environment (24 m tiles) the prey's policy does not affect catches.**
  Prey standing still are caught as often as rule prey over 20 s, because the 5 s rest caps the
  wolves, and about 20 catches per tile happen on the first step from prey spawned inside a wolf's
  reach. The user chose `contact-v3` (separated spawns) at the reference density (36 m tiles) for
  research: fleeing then saves 16.5% over 20 s, but nothing over 5 s, and fewer prey per tile do
  not change that (design §10, `test/spikes/coevo-eval/spawn-share.mjs`). The test profile keeps
  24 m and contact-v2 so its infrastructure checks stay comparable.

- **One scenario.** The tools take a scenario module, but only Herd & Wolves exists.
- **The label pass finds its place by a stage's name,** `AgentSystem act`. If a library change
  renames that stage, the tools refuse to run and say so.
- **No equality across GPUs.** A recording made on one GPU differs from one made on another.
- **The site's demo has its own copy of the rule** until it moves onto `scenario/herd-wolves.mjs`.
  The two were compared once, by a scratch script: 0 rows differed over 600 steps.
- **Reading `.onnx`** is the library's, not these tools': `mlpFromOnnx` in `null-graph/inference`
  reads what `agents:export` writes back to the same network, bit for bit
  ([`docs/api/inference.md`](../../docs/api/inference.md)).

## Tests

- `npm run test:agents` runs `test/agents-tools.test.mjs`: the dataset, the sampler, the trainer's
  gradients, the evaluation's counts, the network files, and the `.onnx` writer's bytes.
- `npm run test:gpu -- --only agent-tools` and `npm run test:gpu:browser -- --only agent-tools` run
  the label pass, recording, the committed networks against `mlpForward`, and the two bars above,
  on a real GPU. They take one to two minutes.
- `npm run test:gpu -- --only onnx` runs each committed network as `.onnx` in ONNX Runtime against
  `mlpForward`.
