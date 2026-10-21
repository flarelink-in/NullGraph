# Co-evolution runs and films — a runbook

How to train your own Herd & Wolves co-evolution runs, score their checkpoints and make labelled
films of them, step by step. Every command runs from a PowerShell window in
`D:\orion-realms\null-graph`. Written 2026-10-20 for step 15's tools
([design](./agents-coevo-films-design.md)); the research protocol behind it is
[`agents-coevolution-design.md`](./agents-coevolution-design.md).

These are **exploratory runs**: good for seeing what evolves and for choosing settings, never evidence
for H1–H3. The preregistered study (14.5) has its own seeds and commands, and must not be touched by
exploration (§8).

## 0. Before you start

- **Close the IDE and other heavy apps.** The machine has 7.3 GB; a training run, an evaluation or a
  film each use about 0.5–1 GB, and Windows' memory pressure has stopped background jobs here before.
- **Keep the PC awake.** In Windows' power settings, set sleep to *Never* while you run. Sleep, or
  suspending the process, ends a training run (its 1.5 s safety stop or a lost GPU), and a training
  run cannot be continued: it has to start again.
- **Choose the GPU for this PowerShell window** (it stays set until you close the window):

  ```powershell
  $env:NG_DAWN_FLAGS = 'adapter=NVIDIA'   # the GTX 1650: faster
  # $env:NG_DAWN_FLAGS = 'adapter=AMD'    # the integrated Radeon: about 1.5× slower
  ```

  GPUs round differently, so the same run gives slightly different numbers on each. Compare runs
  scored on the same GPU. One job a GPU at a time; NVIDIA and AMD can each run one job at once.
- **Nothing to build.** `npm run build:lib` is only needed after a change in `src/`.

## 1. Train a run

Two starting points, both `contact-v3` pens of 36 m with 192 prey (cows) and 16 wolves (foxes),
20-second episodes, 32 mirrored pairs, 200 generations:

| Config | Start | Network |
|---|---|---|
| `tools/agents/explore/x150-clones.json` | the clones: networks trained to copy the hand-written rule | 24 → 64 → 64 → 64 → 2 |
| `tools/agents/explore/x150-random.json` | random weights | 24 → 16 → 16 → 2 |

```powershell
npm run agents:coevolve -- --config tools/agents/explore/x150-clones.json --master-seed 15004 --name x150-clones-15004
```

- **A new name every time.** The run's folder must not exist; nothing is ever overwritten.
- **Seeds:** use master seeds from 15000 to 15999 (the configs' domain is 150). Never 14501–14503
  with domain 145: those belong to the preregistered study. Used so far: 15001 (clones), 15002
  (random), 15003 (clones). The master seed is one of three kinds of seed, which never mix:

  | Seed | Set by | Decides |
  |---|---|---|
  | Master seed, 15000–15999 | `--master-seed` when training | the whole run: every generation's perturbations, every training episode's start positions, which older opponents are drawn, and a random start's first weights |
  | Evaluation seed, 15500 | `--seed` when scoring (§3) | the fresh test episodes, which training never played |
  | Film seed, 77001 by default | the film file's `"seed"` (§4) | only where the animals stand when the filmed episode starts, the same in every pen; nothing about the brains |

  A film's footer prints its film seed; its run's master seed is in the run's name.
- **One generation = one species learns:** prey in odd generations (1, 3, 5, …), wolves in even
  ones. 200 generations is 100 updates of each.
- **Time:** about 9 s a generation on NVIDIA (30 minutes for 200), about 15 s on AMD.

Each generation prints one line:

```text
coevolve: gen  12/200 (wolf) scores [51, 42, 55, 50...] |g| 29.011 |Δx| 0.4150 in 8952 ms
```

`(wolf)`: the wolves learned this generation. `scores`: each candidate's result in its pen (wolves:
catches; prey: minus catches). `|g|`: the size of the learning signal (large when each mirrored
pair's two candidates scored consistently apart). `|Δx|`: how far the network moved. Then the
generation's wall time.

**Other changes** are flags added to the same command:

| Flag | Default in the configs | What it changes |
|---|---|---|
| `--generations 400` | 200 | Longer runs. Keep under about 600: memory grows about 1.6 MB a generation |
| `--archive-share 0` | 0.5 | Archive off: always the latest opponent, never an older one (H3's question) |
| `--sigma 0.05` / `0.25` | 0.1 | How far candidates are perturbed from the current network |
| `--rate 0.03` | 0.01 | How big each learning step is |
| `--K 16` | 32 | Mirrored pairs a generation (fewer: cheaper, noisier) |

**Output**, in `tools/agents/out/coevo/<name>/` (not committed):
- `gen-0000/` … `gen-0200/` every 10 generations and the last: `prey.mlp.json` and `wolf.mlp.json` (the
  networks) and `manifest.json` (the environment they were trained in);
- `archive/`: the older opponents kept every 5 generations;
- `log.jsonl`: one line per generation; `run.json`: the full settings, seeds, GPU and code version;
- `status.json`: `"complete"` only when the run finished cleanly.

**Stopping:** Ctrl+C stops a run, and it cannot be resumed. Running the same command with a new
name, the same seed and the same GPU repeats it exactly, generation for generation.

## 2. Look at the training curve

```powershell
node tools/agents/coevolve-report.mjs training tools/agents/out/coevo/x150-clones-15004 --every 50
```

Mean catches a pen in the training episodes, in blocks of generations, apart for the generations in
which prey learned and those in which wolves did. These are catches against the *current* opponents,
so they show the arms race, not skill against a fixed opponent: for that, score the checkpoints (§3).

## 3. Score the checkpoints

```powershell
npm run agents:arena-eval -- --run tools/agents/out/coevo/x150-clones-15004 --generations 0,50,100,200 --purpose validation --seed 15500 --domain 150 --analysis-seed 15599 --episodes 52 --diagnostics-stride 60 --name x150-clones-15004-validation
node tools/agents/coevolve-report.mjs evaluation tools/agents/out/coevo-eval/x150-clones-15004-validation
```

The first command plays each listed generation against the rule, the rule without wander ("label")
and the clones, both ways, in 52 fresh 20-second episodes; the second prints the result.
- `--generations`: checkpoints that exist (multiples of 10, or the last).
- `--purpose validation`, always: `final-test` is reserved for the preregistered study.
- `--episodes`: a multiple of 4 (52, 100, 200). More episodes, narrower intervals.
- `--diagnostics-stride 60`: also measure, every second, how many prey hug the walls, how fast they
  run and what the wolves do. It doubles the time; leave it out for catches only.
- A new `--name` for every evaluation. Time: about 6–7 minutes on AMD with diagnostics.

**Reading the report:**

```text
checkpoint against   its prey caught (baseline prey)         its wolves' catches (baseline wolves)
gen-0100   rule      37.3 (55.1) -17.8 [-19.4, -16.3] better 55.0 (55.1) -0.1 [-1.2, +1.0]
```

Generation 100's prey were caught 37.3 times a pen by rule wolves; rule prey, 55.1 times. The
difference is −17.8, and its 95% interval [−19.4, −16.3] lies wholly below zero, hence *better*.
Generation 100's wolves caught 55.0 rule prey against the rule wolves' 55.1: an interval across
zero, no mark. The table after it gives the share of prey within 1 m of a wall at 0, 5, 10, 15 and
19 s (0.00 for the rule's prey), then the wolves' resting share, prey speed and wolves' distance
to their nearest prey.

**Archive on against off** (H3's question) needs cross-play, every listed generation against every
other:

```powershell
npm run agents:crossplay -- --run tools/agents/out/coevo/x150-clones-15004 --generations 0,50,100,150,200 --purpose validation --seed 15500 --domain 150 --analysis-seed 15599 --episodes 52 --name x150-clones-15004-crossplay
node tools/agents/coevolve-report.mjs evaluation tools/agents/out/coevo-eval/x150-clones-15004-crossplay
```

It prints the catch matrix (rows prey generations, columns wolf generations) and the forgetting: the
largest drop from an earlier to a later generation against generation 0. Train the same seed twice,
`--archive-share 0.5` and `0`, and compare their forgetting.

## 4. Make a film

**1. Write the film file.** Copy a preset to a folder git ignores and edit it:

```powershell
New-Item -ItemType Directory -Force tools/film/out/presets | Out-Null
Copy-Item tools/film/films/coevo-x150-clones-prey.json tools/film/out/presets/x150-clones-15004-prey.json
notepad tools/film/out/presets/x150-clones-15004-prey.json
```

```json
{
  "name": "x150-clones-15004-prey",
  "scene": "coevo-arena",
  "settings": { "run": "tools/agents/out/coevo/x150-clones-15004", "prey": [0, 50, 100, 200], "wolf": ["rule", "rule", "rule", "rule"] },
  "film": { "seconds": 20, "fps": 60, "width": 1920, "height": 1080, "scale": 1 },
  "segmentSeconds": 10,
  "encode": { "codec": "libx264", "crf": 18, "preset": "medium", "tune": "animation" }
}
```

- `run`: the run's folder. `prey` and `wolf`: what plays in each pen, in reading order (top left, top
  right, bottom left, bottom right). An entry is a generation, `"clone"`, or `"rule"`; the rule must
  then fill all four pens of that species. 4, 8, 12 or 16 pens; with 8, add `"cols": 4`.
- Three useful films: **prey against the rule** (as above), **wolves against the rule**
  (`"prey": ["rule", "rule", "rule", "rule"], "wolf": [0, 50, 100, 200]`) and **the arms race**
  (`"prey": [0, 50, 100, 200], "wolf": [0, 50, 100, 200]`).
- `"clone"` cannot share a film with a random start's generations (different network sizes).
- Optional settings: `"seed"` (the film seed, 77001 by default: another number films the same
  brains from other start positions), `"look": "faithful"` (the demo's plainer look). 20 seconds
  is exactly one training episode.

**2. Record, then label:**

```powershell
npm run film -- tools/film/out/presets/x150-clones-15004-prey.json --out tools/film/out/x150-clones-15004-prey
node tools/agents/coevolve-film-labels.mjs tools/film/out/x150-clones-15004-prey --title "Clone start, seed 15004: evolved prey vs rule wolves"
```

The film is `tools/film/out/x150-clones-15004-prey/film-labelled.mp4` (`film.mp4` is the same film
unlabelled). About 30–40 seconds a film on either GPU. If a recording stops, run the same `npm run
film` command again with `--resume` added: it continues exactly where it stopped.

## 5. Watch them live in the site

The site's page **Herd & Wolves (evolved)** (Agent Experiments in the menu,
`/dashboard/agents/herd-wolves-coevo`) runs checkpoints live
([design](./agents-coevo-demo-design.md)). It already holds the three runs so far, generations 0, 50,
100, 150 and 200. To add a run you trained:

```powershell
npm run agents:site -- export --run tools/agents/out/coevo/x150-clones-15004
```

- Without `--generations` it copies the multiples of 50 and the last; `--generations 0,100,200`
  chooses. A run exported again replaces its earlier copy; `npm run agents:site -- remove --name
  x150-clones-15004` takes it out.
- Every run in the page must have trained in the same pens (36 m, 192 and 16, `contact-v3`): the
  export refuses a run whose pens differ, and says which setting.
- Reload the page (or restart the site's dev server) to see a new run in the panel.

On the page:
- **Agents:** *training pens* (4, 16 or 64 pens of 36 m, run by the trainer's own pen code, so the
  brains meet exactly what they trained in; 64 pens is one whole training generation), or the *open
  field* (4K to 1M agents, Herd & Wolves' own rules: the transfer test).
- **Brains,** per species: *Rule*, *Clone*, or *Evolved* with a run and a generation.
- **Pens show:** *Generation ladder* puts the run's generations side by side, every pen from the
  same start, as the films do; *One matchup* runs one generation in every pen. A change of brains
  rebuilds the pens from the same start, so two brains are compared from the same positions;
  *Restart* moves to the next start.
- The overlay says when speeds, rest or think-every leave the values the brains trained with.

After editing the page's `arena.ts`, or the trainer's pens, run `npm run agents:site -- check`
(add `--gpu` to step both on the GPU): it must print `ok` on every line.

## 6. Ideas worth a run

| Question | Run | Compare |
|---|---|---|
| Is fence-hugging a fluke? | the clones config, new seeds (15001 and 15003 both found it) | each run's prey against the rule (§3), and films |
| Does the archive reduce forgetting? | one seed with `--archive-share 0.5` and the same with `0` | their cross-play forgetting (§3) |
| Where does learning stop? | `--generations 400` | generations 0, 100, 200, 300, 400 against the rule |
| How much does perturbation size matter? | `--sigma 0.05` and `--sigma 0.25` | prey and wolves against the rule |
| Do random starts reach the clones every time? | the random config, new seeds | generation 200 against the clones |

## 7. Where things are

| What | Where |
|---|---|
| Run configs | `tools/agents/explore/x150-clones.json`, `x150-random.json` |
| The site's page and its checkpoints | the site's `src/data/engine-demos/demos/AgentsExperiments/HerdWolvesCoevo/` (`checkpoints/`) |
| Runs | `tools/agents/out/coevo/<name>/` |
| Evaluations | `tools/agents/out/coevo-eval/<name>/` (`evaluation.json`, `plan.json`, `log.jsonl`) |
| Films | `tools/film/out/<name>/film-labelled.mp4` |
| Film files (presets) | `tools/film/films/coevo-x150-*.json` (committed); yours in `tools/film/out/presets/` |
| Reports | `tools/agents/coevolve-report.mjs` (`training`, `evaluation`) |
| Labels | `tools/agents/coevolve-film-labels.mjs` |
| Existing films | `tools/film/out/coevo-x150-{clones-prey,clones-wolves,random-pairs,random-prey}/` |

Everything under an `out/` folder is ignored by git: keep what you want elsewhere.

## 8. Keep the preregistered study clean

The 14.5 study ([prereg](./agents-coevolution-prereg.md)) will judge H1–H3 on fresh runs and fresh
test episodes. Until its freeze, and for exploration after it:
- never use master seeds 14501–14503 with domain 145;
- never pass `--purpose final-test`;
- call results exploratory wherever you write them down.

Its runs start only after the freeze (14.0's gates in `contact-v3`, the direction screen, the frozen
held-out lists). They will use the frozen configs and the commands in the prereg's §5; films of
them are made the same way as in §4 here.

## 9. When something goes wrong

| Message | What to do |
|---|---|
| `run directory already exists` / `evaluation directory already exists` | Use a new `--name` |
| `output already contains a film` | Add `--resume` to finish it, or use a new `--out` |
| `no checkpoint gen-0070` | List generations that exist: multiples of 10, or the last |
| `needs 4, 8, 12 or 16 tiles` / `runs the rule in every tile or in none` | Fix the film file's `prey`/`wolf` lists |
| `networks must share one shape` | Don't mix `"clone"` with a random start's generations |
| `film labels: the film follows a camera path` | Labels need the default camera: remove `"camera"` from the film file |
| `guard: … stopped` or `WebGPU failure` | Close other apps that use the GPU; run again with a new name |
| The PC slept, or the window closed | Training: run again with a new name. Film: same command with `--resume` |
| `status.json` says `incomplete` | The run did not finish; its checkpoints so far can still be scored and filmed |
| `coevolve-site: … pens differ from the catalog's in L, …` | The run trained in other pens: the page shows one kind at a time. Remove the catalog's runs first (`agents:site -- remove`) |
| The page's overlay says `a checkpoint did not load` | Export the run again, then `npm run agents:site -- check` |
