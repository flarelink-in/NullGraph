# Agent training tools

These tools make a network that copies a rule brain: they record what the rule decides, train one
small network per species on the recording, and judge the copy in the running simulation. They
live in the repository and are not part of the `null-graph` package.

The design, with every measurement, is
[`docs/agents-cloning-design.md`](../../docs/agents-cloning-design.md). Loading a trained network
in an app is in [`docs/api/agents.md`](../../docs/api/agents.md), "A trained network".

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
```

Everything under `tools/agents/out/` is ignored by git.

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
| `cli.mjs` | What the command lines share |

`scenario/`, `label-pass.mjs`, `dataset.mjs`'s row helpers, `recorder.mjs`, `evaluator.mjs` and
`model.mjs` import nothing from Node, so the GPU suite also runs them in Chrome.

## Limits

- **One scenario.** The tools take a scenario module, but only Herd & Wolves exists.
- **The label pass finds its place by a stage's name,** `AgentSystem act`. If a library change
  renames that stage, the tools refuse to run and say so.
- **No equality across GPUs.** A recording made on one GPU differs from one made on another.
- **The site's demo has its own copy of the rule** until it moves onto `scenario/herd-wolves.mjs`.
  The two were compared once, by a scratch script: 0 rows differed over 600 steps.
- **`.onnx` export** is the next phase (7.10) and is not built.

## Tests

- `npm run test:agents` runs `test/agents-tools.test.mjs`: the dataset, the sampler, the trainer's
  gradients, the evaluation's counts, and the network files.
- `npm run test:gpu -- --only agent-tools` and `npm run test:gpu:browser -- --only agent-tools` run
  the label pass, recording, the committed networks against `mlpForward`, and the two bars above,
  on a real GPU. They take one to two minutes.
