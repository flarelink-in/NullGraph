# NullGraph Agents — Phase 2 step 15: films of co-evolution checkpoints — Design

**Parent:** [`work/plan-2.md`](./work/plan-2.md) step 15, built on D2's scene contract
([`../tools/film/SCENE-AUTHORING.md`](../tools/film/SCENE-AUTHORING.md)) and step 14's lanes
([`agents-coevolution-design.md`](./agents-coevolution-design.md)).
**Status (2026-10-20):** first slice built at the user's word ("do step 15 first slice i wanna see
some results"): a scene that films checkpoints in their training arenas, tile labels, two
exploratory runs and four films of them, with their numbers on validation seeds (§5). **Exploratory: films are not evidence for H1–H3**, which only the frozen
14.5 protocol can give ([prereg](./agents-coevolution-prereg.md)).

**Target:** a scene in `tools/film/scenes/`, a label script, a report script and run configs in
`tools/agents/`, tests and docs. No library change and no film-runner change. The user's
step-by-step guide to runs, scores and films is [`agents-coevo-runbook.md`](./agents-coevo-runbook.md).

## 1. What a film shows

Each film is one episode in the arenas a run trained in: the run's own experiment (environment,
contacts and solver options, catch margin, spawn separation, tile size, prey and wolves per tile,
thinkEvery, dt), stepped by `buildLane`, the lane the trainer and the evaluator step. Tile a plays
`prey[a]` against `wolf[a]`. Every tile starts from the **same episode seed**, so the tiles differ
only in their brains (and, after spawn, in index-and-tick draws: design §11 of the prereg).

A film of 20 s at 60 fps is exactly one training episode (T = 1200 at dt = 1/60), one step a frame.
A longer film shows play past the training horizon.

Typical films:
- **pairs:** generations 0, 50, 100, 200 against each other, a row per tile: the arms race's state.
- **prey against the rule:** those generations' prey against rule wolves: prey change alone.
- **wolves against the rule:** rule prey against those generations' wolves.

## 2. The scene: `coevo-arena`

Settings: `run` (a run directory, relative to the repository), `prey` and `wolf` (one entry per tile:
a generation number, `"clone"` or `"rule"`), `seed` (77001), `cols` (2), `look` (`anime` or
`faithful`).

- **Drivers:** a species is the rule in every tile or in none, since a lane runs one brain per
  species. Network entries become rows of one `MlpBrain.byGenome` table per species, one row per
  distinct entry, read through the evaluator's `loadRun`/`loadCheckpoint`/`loadClones`: each model
  must match its manifest's hash and the manifest the run's experiment. One table needs one shape,
  so a 16 × 16 run's checkpoints cannot share tiles with the 64 × 64 × 64 clones.
- **Tiles:** 4, 8, 12 or 16, since 16 wolves a tile must fill 64-agent blocks.
- **Look:** Herd & Wolves' Models look (`herd-wolves/look.mjs`, the site's copies) over the lane:
  cows are prey, foxes wolves. The field is the world's width plus 6 m, and the field wrapper adds
  the site's fence (`fence.ts`) around every tile. The look's animal-phase stage writes only its
  own buffer, so the simulation is the lane's. No catch flashes (the arena's respawn stage keeps no
  flash ring).
- **Camera:** straight down by default, every tile in view with a tenth to spare, tile 0 top left
  and then reading order (`topDownView`). A film JSON may set a camera path; labels then refuse.
- **State:** tracks hold the records, the animals' phases and the agent state; checkpoints add the
  held actions, the catch counters and the contact solver's own tick (its window phase), which is
  everything a step reads from the previous one.

## 3. Labels

The film tool draws no text under Node yet (an atlas needs a canvas; captions are D2.3's next work).
`node tools/agents/coevolve-film-labels.mjs <film output>` burns two lines a tile ("Prey: generation
50", "Wolves: the rule") beside the tiles, plus a title (`--title`) and a footer (environment,
layout, seed, "exploratory, not a result"), with ffmpeg's drawtext. It finds each tile through the
scene's own `topDownView` and the engine's camera, reads tiles and grid from the film's
`report.json`, and re-encodes with the film runner's BT.709 tags to `film-labelled.mp4`.

## 4. Validation (A55)

On Node/AMD, 2 s of a four-tile film of a clone-start run (generations 0 and 3, the clones):
- direct recording: 120 frames, no failure, no WebGPU error, worst step 23 ms;
- tracks-only recording replayed against the direct frames: mean absolute difference at most
  0.41/255 a channel (packed, interpolated tracks; the worst single pixel 173);
- stop after one segment and resume: the final checkpoint equals an uninterrupted recording's
  byte for byte, so resume is exact;
- labels: the tile rectangles projected for 1920 × 1080 fall where the frames show the tiles.

Headless: U56 (`test/coevo-film.test.mjs`, in `npm run test:film`) checks the settings' refusals and
the tile plan against a real run directory (rows per distinct entry, the rule species-wide, model
checks, one shape a table), the label geometry and ffmpeg escaping; five injected faults (every tile
on row 0, no shape check, a mixable rule, one escaping level, right-column labels inside) each fail
it. `test/coevo-report.test.mjs` (in `npm run test:agents`) checks `coevolve-report.mjs`'s training,
comparison, diagnostics and cross-play lines.

## 5. The exploratory runs

`tools/agents/explore/x150-clones.json` and `x150-random.json`: `contact-v3` at 36 m, 192 prey and
16 wolves a tile, T = 1200, K = 32, E = 1, σ 0.1, rate 0.01, decay 0.005, archive on, step 13's
solver defaults, 200 generations; master seeds 15001 and 15002 in domain 150, apart from 14.5's
confirmatory seeds (14501–14503, domain 145). They are 14.0-style smoke runs at candidate settings
that Gate 3 has not chosen: their films and numbers inform, never confirm. Their numbers use
validation seeds only. Each took about 30 minutes on Node/NVIDIA (9.1 s a generation).

**Clone start (`x150-clones-15001`), 2026-10-20**, `agents:arena-eval` on 52 validation episodes
(`--seed 15500 --domain 150 --analysis-seed 15599 --diagnostics-stride 60`, Node/AMD; catches per
20 s tile, paired by episode against the baseline's own pairing, 95% intervals):

| Generation | Prey against rule wolves | vs rule prey | Prey within 1 m of a wall, 5 s / 19 s | Its wolves against rule prey, vs rule wolves |
|---|---|---|---|---|
| 0 (the clones) | 54.7 | −0.4 [−1.3, 0.5] | 0.00 / 0.00 | +0.8 [−0.1, 1.6] |
| 50 | 39.6 | −15.5 [−17.0, −14.0] | 0.26 / 0.44 | +1.5 [0.6, 2.3] |
| 100 | 37.3 | −17.8 [−19.4, −16.3] | 0.33 / 0.53 | −0.1 [−1.2, 1.0] |
| 200 | 40.7 | −14.3 [−15.7, −12.9] | 0.19 / 0.24 | +1.7 [0.7, 2.6] |

Rule prey are caught 55.1 times. Evolved prey moved to the fences (rule and clone prey: 0.00 after
5 s), ran about twice as fast (1.1–1.4 against 0.7 m/s) and kept rule wolves off rest (55–59% of
wolves resting against 77%). The rule turns both species away from a wall they near, blending the
chase out (`scenario/herd-wolves.mjs`, the bounds term), so prey at a fence may exploit rule-like
wolves rather than flee better: a hypothesis the diagnostics fit but do not prove. Generation 200,
co-evolved against wolves that met such prey, hugs the walls less and gains less. Evolved wolves
gained at most 1.7 catches (3%). One run, one device for the numbers, not H1 evidence.

**Random start (`x150-random-15002`)**, same evaluation (16 × 16 networks from seeded weights):

| Generation | Prey against rule wolves | vs rule prey | Prey within 1 m of a wall, 5 s / 19 s | Its wolves against rule prey | vs rule wolves |
|---|---|---|---|---|---|
| 0 (random) | 68.8 | +13.7 [12.6, 14.8] | 0.20 / 0.44 | 6.9 | −48.2 [−49.3, −47.0] |
| 50 | 42.2 | −12.9 [−14.6, −11.2] | 0.45 / 0.68 | 42.8 | −12.2 [−13.4, −11.0] |
| 100 | 43.3 | −11.8 [−13.4, −10.2] | 0.39 / 0.61 | 50.4 | −4.6 [−5.8, −3.6] |
| 200 | 44.9 | −10.1 [−11.8, −8.5] | 0.26 / 0.30 | 54.9 | −0.1 [−1.2, 0.9] |

From weights that catch 6.9 prey a tile, wolves reached the rule's 55.1 in 200 generations; prey
went from worse than the rule to better. Against the clones at generation 200, prey are
−8.8 [−10.1, −7.5] and wolves −0.5 [−1.8, 0.6]: within the prereg's provisional 5% margin
(about 2.6 catches) for both, so H2's rule would pass in this run. Random prey reach the walls by
drifting (0.44 at 19 s); evolved prey go there sooner, then, against co-evolving wolves, less.
Films: `tools/film/out/coevo-x150-{clones-prey,clones-wolves,random-pairs,random-prey}/film-labelled.mp4`
(the clone start's on Node/AMD, the random start's on Node/NVIDIA).

**Replication (`x150-clones-15003`)**, the clone start with master seed 15003, trained, scored and
filmed by following the runbook's commands as written (Node/NVIDIA training and films, Node/AMD
scores). Fence-hugging evolved again: prey caught 40.3, 36.9 and 41.6 times by rule wolves at
generations 50, 100 and 200 (−14.7, −18.2, −13.5, intervals within ±1.5), with 47%, 71% and 36% of
them within 1 m of a wall at 19 s; wolves gained 1.2–2.2 catches. Seed 15001 gave 39.6, 37.3 and
40.7 with 44%, 53% and 24%: the same peak at generation 100 and the same partial retreat by 200.

## 6. Limits and next slices

- One episode a film, one seed: a film shows behaviour, not an effect size.
- No cross-GPU equality: a film on another GPU shows a different episode of the same brains.
- The rule cannot share a species with networks in one film; one shape a species.
- Next: a champion arena (one tile, closer camera), films of 14.5's frozen checkpoints once they
  exist, and in-engine captions when D2.3 has them.
