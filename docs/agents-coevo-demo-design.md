# NullGraph Agents — Phase 2 step 16: evolved brains in the site — Design

**Parent:** [`work/plan-2.md`](./work/plan-2.md) step 16, built on step 14's lanes
([`agents-coevolution-design.md`](./agents-coevolution-design.md)) and step 15's films
([`agents-coevo-films-design.md`](./agents-coevo-films-design.md)).
**Status (2026-10-21):** first slice built and validated (§7), at the user's word: "you told me that
the models can be tried out in site copy, with all that controls, so that i want too, along with one
more control of the count mentioned and used to create one". Asked the same day, the user chose the
site copy before S7, and **exact training pens** for the training count over a cheaper 208-animal
field under the site's rules. **Exploratory: the page shows behaviour, never evidence for H1–H3.**
The site's repository is not committed by this step: that stays the user's call.

**Target:** a new page in the test site, `HerdWolvesCoevo` (`/dashboard/agents/herd-wolves-coevo`),
beside `HerdWolvesGPU`, which stays as it is; here, `tools/agents/coevolve-site.mjs` (export and
check), U57 (headless) and A56 (GPU). No library change. Live evolution in the browser is not part
of it: plan-2 ties that to S7's rate, which is open.

## 1. What the page shows

**Brains,** per species: the rule (the demo's `HERD_BRAIN`), the clone (C2's network, the site's
`HerdWolvesGPU/networks/<species>.mlp.json`), or a saved generation of an exported run.

**Agents,** one control with two kinds of size:
- **Training pens:** 4, 16 or 64 pens of the run's experiment, each `L × L` with `P` prey and `W`
  wolves (today 36 m, 192 and 16), stepped by the trainer's lane: `contact-v3`, contacts on with the
  run's solver options, the exhaustive catch rule and its margin, spawns kept `spawnSeparation`
  from the pen's wolves, containment, a fence round every pen. 64 pens is one training generation
  (13,312 animals). Pens come in fours: `arenaLayout` needs `arenas · W` to fill 64-agent blocks.
- **Open field:** 4K to 1M agents, `HerdWolvesGPU`'s simulation with its own rules: one field
  160 m wide at 4K, one wolf in 16, the nearest-wolf catch, respawns anywhere, contacts optional.
  This is the transfer test: brains that learned in 36 m pens, in a field whose fences most animals
  never see.

**Pens show** the same matchup in every pen, each pen from its own seed, or a **generation
ladder**: each species set to a run plays that run's exported generations across the pens, every
pen from the same seed, as the films do. With more generations than pens they are spread from the
first, the last in the last pen (0, 50, 100, 200 of five, the films' four); with fewer, repeated
in order. A label over each pen's near fence names its brains and counts its catches since the
pens were built, read back every 0.5 s off the frame path.

**Episodes.** Pens run on past the training episode (`T`, 20 s); the overlay says when play is past
it. Restart, and any change of brain or ladder, builds the pens again from the same start, so two
brains are compared from the same positions; Restart moves to the next seed. Rebuilding on a timer
is left out: the engine keeps every compute stage it creates (about 44 a lane), so a page rebuilding
every 20 s would grow without bound (`tools/agents/README.md`, known limits).

## 2. The same pens as the trainer

The page holds a TypeScript port, `HerdWolvesCoevo/arena.ts`, of `scenario/arena-layout.mjs` and
`scenario/herd-wolves-arenas.mjs`: `arenaLayout`, the tile wall sensor and steering, the
exhaustive catch, the arena respawn, containment, `tileSpawns`, `arenaDefinition`, `modelNames` and
`buildLane` without the trainer's diagnostics, fixtures and finite monitor. The rule comes from
`HerdWolvesGPU/brain.ts`, which `test/spikes/live-switch/check-site.mjs` already holds to the
scenario's text byte for byte.

Checked from this repository by `node tools/agents/coevolve-site.mjs check`, which builds the port
with the site's own esbuild:
- **CPU (U57):** for layouts of 4, 16 and 64 pens, with and without contacts, the port's layout,
  definition (names, species, world, grid), generated sense and act WGSL, catch, respawn and
  containment WGSL, rule texts and spawns equal the trainer's.
- **GPU (A56, `--gpu`):** a lane from each, the same seed, brains and rows, stepped 600 steps on one
  device: records, held actions, state, catch counters and stats equal byte for byte.

Single networks: when every pen runs one network for a species, the page drives it with a plain
`MlpBrain`; the trainer uses a one-row `byGenome` table. The library guarantees the same actions
bit for bit on one GPU (the guide, "A network per agent"); A56 checks it on this pair.

What the page adds runs after the lane and writes only its own buffers: the Models look's animal
phases, and **catch flashes**: the lane's respawn stage keeps no flash ring, so a stage appended
after it records a wolf whose rest timer equals the rest setting (true only on the step of its
catch, since a resting wolf cannot catch) at the wolf's position. Pens catch about five times as
often an animal as the open field, so one catch in `pens / 4` flashes, about as many rings as the
field shows at 4K.

Not claimed: the page steps by wall time as every demo does (the same fixed steps, a different
number a frame), and another GPU or browser rounds differently, so a page's pens are the trainer's
code, not a film's numbers.

## 3. Checkpoints: the catalog

```powershell
node tools/agents/coevolve-site.mjs export --run tools/agents/out/coevo/x150-clones-15001 --generations 0,50,100,150,200
```

copies each named checkpoint's `prey.mlp.json` and `wolf.mlp.json` byte for byte into the site's
`HerdWolvesCoevo/checkpoints/<run>/gen-XXXX/`, after `loadRun` and `loadCheckpoint` have checked
them against the run's manifests, and rewrites `checkpoints/index.json`:
- `experiment`: the pens' settings (environment, `P`, `W`, `L`, `gap`, `T`, `dt`, `thinkEvery`,
  contacts and their options, catch margin, spawn separation), shared by every run in the catalog.
  An export whose run differs in any of them is refused: one catalog, one pen experiment.
- `runs`: per run its name, start, network shape, master seed, config hash and, per generation, the
  two files' SHA-256.

Exporting a run again replaces its entry and folder. `--site` (or `NG_SITE`) names the site; the
default is the one `film:check` uses. The page loads a file only when picked (Vite
`import.meta.glob`), checks its SHA-256 against the catalog where the page can hash, and its names
against the definition through `modelNames`.

## 4. Controls

From `HerdWolvesGPU`, kept: pause and restart, simulation speed, draw as (models, darts, arrows),
the anime post, think every, the rules' switches (flock, flee), the speeds, the wolves' rest, the
show toggles, hover, click and the inspector. Field only: contacts and their passes, the scan
budget. New: the agents control's pens, brains as rule, clone or run and generation, pens show
(same or ladder). Removed: ONNX Runtime, which stays on `HerdWolvesGPU`.

In the pens, what the experiment fixes (contacts, solver, scan budget, catch rule, spawns) is not a
control. Speeds, flocking, fleeing, rest and think-every stay live; set away from training's values
the overlay says the pens are outside training.

## 5. Validation

- U57 in `npm run test:agents`; A56 on Node/NVIDIA and Node/AMD; each watched failing with a changed
  port (one WGSL constant, one spawn draw).
- The site: `tsc` on the project and `eslint` on the new files, no new errors; `npm run drive` in
  Chrome on the AMD iGPU against a dev server: the pens at 4, 16 and 64, the field at 4K and 16K,
  every brain kind, the ladder; no WebGPU or uncaught error in the page or the worker; screenshots
  read by eye.
- `HerdWolvesGPU` untouched: `test/spikes/live-switch/check-site.mjs` and `npm run film:check` stay
  green.

## 6. Limits and next slices

- One pen experiment per catalog; runs trained in another experiment need their own page state.
- The page's catches are one device's; the evaluator's intervals stay the measure.
- Every rebuild of the pens leaves its compute stages in the engine (about 44 a lane), as every
  rebuild of `HerdWolvesGPU` does; a session of brain changes grows by about a megabyte each.
- The open field's catch is `HerdWolvesGPU`'s nearest-wolf rule; the pens' exhaustive rule loops over
  a pen's wolves, so it does not scale to one field of 16K.
- Next: the cross-play chart and intervals from `agents:crossplay` in the overlay (plan-2's HUD),
  a champion view (one pen, closer camera), and live evolution if S7's browser rate allows it.

## 7. Results, 2026-10-21

Built in one session at the user's word. **Exploratory, as everything the page shows.**

- **The catalog:** `x150-clones-15001`, `x150-random-15002` and `x150-clones-15003`, generations 0,
  50, 100, 150 and 200 each, 30 files, 3.7 MB; every file's SHA-256 matches its run's manifest
  (`check`).
- **U57** (`test/coevo-site.test.mjs`, in `npm run test:agents`, 149/149): export, refusals,
  re-export, removal and tampered files; the port check against the trainer itself, eight injected
  differences each named (respawn and containment WGSL, catch margin, a spawn draw, the definition's
  scan budget, a refusal, the rule's text, the rule drivers); the model reader. Against the site's own
  port: the five layouts' layouts, definitions, WGSL, model names and 2 × 5 × 3 spawns a layout, ten
  refusals, both rule texts, and the reader on the clones, all equal. Watched failing: four port
  mutations (respawn constant, spawn draw, prey `maxTurn`, the catch's resting test), a fifth (the
  gap rule) only after two refusal cases reaching it were added; three export mutations (no
  experiment refusal, no hash check, a removed run's folder kept).
- **A56** (`check --gpu`, 600 steps a case, one submission a step): the rule against the rule (4
  pens), the clone prey as one network against rule wolves, label prey against the clone wolves as
  one network, and a ladder of four prey rows and two wolf rows (16 pens): every record word, held
  action, state word, pen counter and stat equal, on Node/NVIDIA (118–124 catches a case) and
  Node/AMD (114–128). So a pen-wide plain `MlpBrain` matches the trainer's one-row table bit for bit
  on both. Watched failing: catch margin 0.051, spawn separation 2.5 and think-every 2 in the page's
  lane each fail all four cases.
- **The site:** `tsc` on the whole project clean (it was clean before); `eslint` clean on the new
  files (the two warnings left in `inspector-tab.tsx` and `useDemoControlStore.ts` predate them).
- **Drives** (`npm run drive`, Chrome on the AMD iGPU, three runs, 84 steps): 4, 16 and 64 pens on
  the ladder, one matchup, both species evolved, the random start on both species (the films'
  "pairs"), the clones on both, think-every 2, the open field at 4K, 16K and 64K, darts, and a click
  that selected "Prey 130 of pen 1 · decided by generation 0" with its senses. No uncaught error and
  no WebGPU error in the page or the worker. Read by eye: generation-200 prey against rule wolves
  bunch along the pens' fences; the overlay flags think-every 2 as outside training.
- **Costs** seen on the overlay (Chrome/AMD, Models, GPU time a step): 4 pens 4.6–5.1 ms, 16 pens
  4.7–5.9 ms, 64 pens 6.1–7.7 ms; the open field 4K 5.5 ms, 16K 9.0 ms, 64K 89 ms (0.17× real time,
  as `HerdWolvesGPU` measured at that size on this GPU). The first pens appear 3.8 s after load
  (networks, models and compiles); rebuilds take 0.2–2.4 s; a field network compiles ahead in
  185–192 ms at 4K and 16K, 1.4 s at 64K, then switches in 0.1 ms.
- **Not green, not this step's:** `npm run film:check` reports the films' copy of the site's
  `brain.ts` changed: the site's contacts work of 2026-10-18 (`herdCatchesWGSL(contacts)`) postdates
  the copy. `test/spikes/live-switch/check-site.mjs` passes.
