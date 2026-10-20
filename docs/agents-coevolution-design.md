# NullGraph Agents — Phase 2 step 14: co-evolution in Herd & Wolves — Technical Design

**Parent:** [`work/plan-2.md`](./work/plan-2.md) §3–§5.
**Revision 2 — 2026-10-18:** complete revision after steps 12–13 and code review.
**Status (2026-10-20):** replacement 14.0 browser correctness verified (77/77); full-run safety and
both-species quality gates remain unaccepted. 14.1 infrastructure validated on all three pairs;
14.2 pure optimizer implemented and tested; **14.3 closed after re-review (§13); 14.4 next.**
Trainer runs so far are infrastructure smoke, not research or safety acceptance. Repairs and regression evidence are tracked in the work log. No training settings accepted. At the user's request, **all original
phase 14.0 / S7 results are discarded**: invariant acceptance, timing/noise tables, learning and
exploit conclusions, derived defaults and browser feasibility. **14.0 is reopened, not complete.**
The old spike source remains historical code, not a reference implementation. No new results are
claimed here, including acceptance based on rerunning the old checks.

**Target:** application code in `tools/agents/`, tests and docs. No core or library changes.
Builds on [genome rows](./agents-genome-rows-design.md), [training tools](./agents-cloning-design.md),
[query-only lists](./agents-queries-design.md), [contacts](./physics/physics-contacts-design.md), and
`tools/film/guard.mjs`. Steps 12–13 and their independent S8/S9 evidence are not discarded.

---

## 1. Purpose, decisions and scope

Train prey and wolves by alternating evolution strategies (ES), headless on the user's GPU. Export
ordinary `.mlp.json` checkpoints, optional ONNX, reproducible metadata and H1–H3 research data.
Contacts are an explicit environment configuration, never a hidden change to the baseline.

### Decisions carried forward from 2026-10-17

- Alternate prey/wolf updates, starting with prey; no simultaneous perturbation of both species.
- Fitness: prey negative catches, wolves positive catches; no shaping by default.
- Archive every 5 completed generations, cap 32 per species, retain the starting network;
  probability 0.5 of sampling an archived opponent. One generation is one species update.
- Checkpoint every 10 completed generations and the last; ONNX on request.
- H1 starts from committed clones; H2 from seeded `24 → 16 → 16 → 2` tanh networks.

These decisions are independent of S7 and remain recorded. Population, arena size, T, E, σ, learning
rate and throughput have **no accepted measured defaults**. Former settings are not carried forward
as accepted defaults merely because they occurred in an old run.

### Revision-2 choices

Mirrored signs use separate, identically indexed lane systems. Episodes start from fresh systems
and solvers. Equal scores get average ranks. Legacy and contact environments have separate versioned
identities. Contact catch semantics in §5 are a **proposed resolution** of step 13 Q8, not a claim
that the user previously accepted it; replacement 14.0 must validate it before settings freeze.

Non-goals: PPO, differentiable simulation, ecosystem births/deaths, obstacles, stamina, resume,
browser trainer, films and site work. Contacts are included. H4's combined contacts-and-obstacles
hypothesis remains deferred; a contacts-only ablation does not answer all of H4.

## 2. Integration constraints verified against the repository

| Existing behavior/API | Consequence |
|---|---|
| `genome` and `user1` are record words | Store policy row and tile ID without new library fields |
| Sense/think gate uses `(state.tick + i / 64) % thinkEvery`, except newborns | Corresponding signs need identical indices and initial tick |
| Rule and respawn randomness use i/tick | Equal initial spawn seeds alone are insufficient |
| Bounded query windows depend on cell ID/tick | Signs also need identical grid/world coordinates |
| Contact coincidence normals hash body indices; contact window phase is private | Fresh, identically indexed solvers; do not assume a reset API |
| `AgentSystem.upload()` uploads dirty records/species, not episode state | Re-spawning records alone is not a complete reset |
| Custom sensors/actions and ordered app stages exist | Arena walls, rules, respawn and containment stay in tools |
| `Sensors.query` does not widen obs | Extra diagnostic/rule queries need not invalidate models |
| `addStepStage({ name, stages })` accepts external stages | Contacts belong inside every fixed step through the shipped seam |
| `DiscContacts` has one world clamp, not per-tile bounds | Add tile containment after contacts |
| A compiled app stage allows at most 3 caller storage buffers on a default device | Respawn binds settings, stats, tile counters; omit the demo flash ring |
| Model loading checks obs/action names | Preserve canonical feature meanings and metadata, not merely widths |
| Current evaluation uses teacher/label/student and min/max spread | Explicit clone opponents and confidence intervals need implementation |

Relevant sources: `AgentSystem.ts`, `agentSenseShaders.ts`, `agentThinkShaders.ts`,
`contactShaders.ts`, `DiscContacts.ts`, `tools/agents/scenario/herd-wolves.mjs`, `model.mjs` and
`evaluator.mjs`. Exact reruns are a same-device/build/runtime requirement, not a cross-GPU promise.

## 3. Formal definition and mirrored lanes

### 3.1 Arenas and indexing

- K ≥ 1 mirrored perturbation pairs; E ≥ 1 replicas per candidate.
- B = KE arenas **per lane**; A = 2KE total evaluated arena episodes per generation.
  `--arenas` means A; require exact equality and no unscored spare arenas.
- Sign s = 0 is plus, s = 1 minus. Logical tile b = kE + e. Apply a seeded permutation of B tiles
  each generation, shared by both lanes, to avoid attaching directions permanently to tile biases.
- Each tile has W wolves and P prey. For physical tile b, wolves occupy `[bW,(b+1)W)` and prey
  `[BW+bP,BW+(b+1)P)`. The same body occupies the same global i in both signs.
- Production alignment requires P a multiple of 64 and BW a multiple of 64. Mixed-row wolf blocks
  use the existing fallback. Never correlate independent pairs' opponents just to share a block.
- Square tiles of side L on C columns have edge gap g. Require g greater than every enabled query
  radius and greater than `2 maxRadius + 2J maxPush`, conservatively protecting isolation during J
  contact passes before containment. Validate this bound in stress fixtures, not only on spawn.
- Require positive finite geometry, enough room for radii/spawn inset and legal world/grid/buffer/
  dispatch sizes. Validate integer products and device limits before allocation.

The plus and minus lanes have identical definitions, world origin, coordinates and indices, but
separate record buffers and candidate assignments. They can execute sequentially; concurrent GPU
execution is not required. Agents never query the other lane. Bounded-grid effects inside one lane
must be mirrored exactly, even if tiles differ from one another in alignment/cadence.

### 3.2 Policies, scores and update

At generation t, evolving master vector x produces `x ± σ ε(t,k)`. Opponent o(t,k) is drawn once per
pair, shared by signs and E replicas. Replica e has its own episode seed, shared by the signs.
With catch counter C(s,k,e) after exactly T fixed steps:

```
f(s,k) = speciesSign * sum_e C(s,k,e) / E     (prey −1, wolves +1)
g = sum_k (u(+,k) − u(−,k)) * ε(t,k) / (2 K σ)
x_next = AdamAscent(x, g − λx)
```

Rank integer catch sums since E is identical for all candidates; retain every replica's counter.
Candidate row is `2k+s`. Canonical logs order by k, sign, replica, independent of physical permutation.
Bodies are not independent statistical episodes. Archived policies are snapshots, not established
champions unless evaluation proves that claim.

### 3.3 Fairness contract

Both signs receive the same opponents, spawn streams, global i, tick zero, think cadence, contact
phase and grid coordinates. Policy-dependent trajectories may diverge; common randomness can
reduce noise but does **not** guarantee cancellation.

**Neutral oracle:** assign identical networks to both signs. On one supported runtime/device,
normalized records, obs/actions and catches must agree word for word. Normalize only intentional
candidate genome IDs, not positions or errors. Include forced respawns, rest timers, off-cadence
newborns, crowded-cell windows, coincidences and contacts. Any mismatch blocks acceptance. Adjacent
tiles with equal spawn seeds are not an acceptable substitute for this oracle.

## 4. Data model, files and ownership

Implement browser-loadable `tools/agents/scenario/herd-wolves-arenas.mjs` with no Node-only imports.
Record `genome` names the policy row; `user1` names the physical tile. Settings and stats retain the
scenario's existing meaning; each lane additionally owns B atomic u32 catch counters.

Persistent caller-owned `MlpBrain.byGenome` tables have **2K + 1 + M rows per species**:
`[0,2K)` candidates, row 2K current opponent, `[2K+1,2K+1+M)` archive. Both lane systems attach the
same species brains; replicas reference the same candidate row. One `setRows` per species per
generation packs all required rows. Tables never rebuild during a run; reject out-of-range rows.

Output: ignored `tools/agents/out/coevo/<name>/`. Refuse an existing run directory; no resume.

- `run.json`: schema/algorithm versions, canonical resolved config/hash, complete seed manifests,
  git commit **and** dirty status/relevant content hashes, build/model hashes, adapter/driver/limits,
  OS/runtime versions. A commit alone does not identify dirty code.
- `log.jsonl`: t, completed updates, evolved species, replica catches/aggregates, score summaries,
  tie share, opponent IDs, permutation, noise checksum, gradient/penalty/update/weight norms, f32
  perturbation collision count, archive changes, detailed timing and guard observations.
- `gen-0000/` always stores both starts. `gen-NNNN/` means after N completed species updates;
  write both networks every N = 10 and finally, without duplicating an already due final export.
  JSON through `modelText`; optional ONNX through `onnxFromNetwork`; environment/config/hash manifest.
- `archive/`: immutable snapshots with stable IDs, insertion generation and model hashes.

Checkpoint deterministic metadata excludes wall times, absolute paths and mutable output-directory
names. Run identity derives from canonical config/seeds/code/start hashes. Performance belongs in
logs. I5 compares checkpoint JSON/manifest bytes, not timing logs or machine metadata; ONNX byte
repeatability is a separate test. `modelText` rejects nonfinite/unrepresentable round trips including
negative zero: fail explicitly, never silently change weights. Write checkpoints to temporary paths
then rename; failures leave the run incomplete rather than publish partial final models.

## 5. Arena environments and contacts (step 13 Q8)

Custom wall sensor and steering integrator use tile bounds while retaining canonical input/output
order and meaning. Assert actual generated names against the canonical model names; custom names
are not automatically equivalent. Rule/settings/species values reuse existing scenario constants.
The library names custom fields by index, so `modelNames()` renames exactly `bounds.0/1` and
`steer.0/1` to the demo's names and refuses any other difference (14.3 re-review).

### `legacy-v1`, contacts off

Preserve catch distance `< 0.9 m`, rest 5 s, rule and respawn semantics. Tile-local respawn uses the
same lane i, episode tick and random streams. One tile covering the original `createHerd` world,
original count/species ratio, seed and thinkEvery 3 must match records and catches bitwise throughout
a rollout. This diagnostic layout may bypass production alignment via the test API only.
Equivalence is claimed only for this mode; other counts/cadences/worlds are distinct workloads.

### `contact-v1`, proposed contact-compatible rules

Initial explicit solver options come from completed step 13, **not S7**: J = 2, relaxation 1.75,
maxPush 0.25 m, budget 32, damping 1, area mass, compliance 0. Map alive/frozen masks and velocity
offset from exported agent layout constants; frozen wolves are fixed bodies. Record actual
resolved solver cell size and device-dependent grid choices.

Per fixed step, after built-in grid/queries/sense/think/act:

1. Catch detection on moved positions, before separation.
2. Consume flags, count once, respawn inside the tile, update wolf rest/frozen state.
3. External `DiscContacts` stages via `addStepStage`.
4. Tile containment: clamp centres to bounds inset by radius; remove outward wall-normal velocity
   components only. Keep heading/frozen semantics; do not reflect heading a second time.
5. Built-in tick finishes the step.

Proposed Q8 resolution: catch reach = prey radius + wolf radius + catchMargin, strict `<` on both
branches, provisional margin 0.05 m. Reach/order/rest are versioned environment parameters frozen
before research. Verify just-inside/outside, resting wolves, dense walls and multiple prey. Keep
legacy many-prey-per-wolf-step behavior; one-catch arbitration would be a separate rule change.

Catch candidate lists were queried before act, like the original scenario. They are not exhaustive
post-move nearest queries. Replacement 14.0 checks controlled cases against brute force and reports
misses; any needed rule/query change is versioned before accepting measurements. `Sensors.query`
can provide extra candidates without changing obs, but adding a query alone does not make a
bounded pre-act list exhaustive.

Global solver clamping is not tile containment. Containment can reintroduce wall overlap; measure
that limitation. The gap bound and per-step clamp must prevent any cross-tile influence during
contact passes. Never credit a catch to a tile reached by an escaped agent.

For a **controlled contacts ablation**, disable the solver while retaining contact-v1 catch rules
and containment/order semantics. Legacy-v1 versus contact-v1 changes multiple environment factors
and cannot isolate contacts alone. Contacts mode is explicit in every research config, not inferred
from discarded data. Obstacles and stamina remain deferred.

### `contact-v2`, user-selected exhaustive eligible-wolf rule (2026-10-18)

14.1 may begin as infrastructure despite 14.0 remaining **unaccepted**. Preserve the original
scenario and protocol-2.1 spike unchanged. `contact-v2` is not pooled with `contact-v1` evidence.
After act and before respawn/separation, copy the complete record buffer into a read-only snapshot.
For each live prey, test every live same-tile wolf whose snapshot rest timer is <= 0, using strict
distance < radius sum + explicit nonnegative catchMargin. Count each touching prey once; every
eligible touching wolf enters rest, including equal-distance ties. Frozen alone does not imply
ineligibility (rest timer does); dead bodies never catch or trigger rest. Multiple prey per wolf per
step remain permitted. Independent per-record writes read only the snapshot, not peers' mutable
flags. Respawn/rest then contacts then containment then tick retain their order. Containment applies
to resting bodies too. Margin 0.05 is a correctness fixture parameter, not an accepted training default.

New validation registration: `test/coevo-arena-gpu/protocol.json`. Phase 14.1 foundation verification
is complete across all three required runtime/device pairs (Node/NVIDIA 40/40, Node/AMD 40/40,
Chrome/AMD 40/40, zero WebGPU errors). Coverage verifies correctness (A42 legacy equivalence,
A46 exhaustive ties/boundaries/resting/dead), shared policy table allocation and candidate discrimination
on caller-owned `MlpBrain.byGenome` tables with in-place `setRows` updates (A43, A45), broader I1/I3/E5/C1
stress (4-arena 10m-gap cross-tile isolation, dense contact/coincidence mirrored exactness, multi-arena
catch counter sum equality, containment wall clamping), and contact-v2 60-step smoke rollout under
the 1500 ms safety stop (NVIDIA max 39.7 ms, AMD max 14.1 ms, Chrome max 133.8 ms) with exact reproducibility
(A47). 14.1 foundation is delivered; production module builds fresh lane systems and attaches shared brain
tables; training optimizer/CLI and full research screening remain for phases 14.2–14.5.

## 6. Complete episode lifecycle

Correctness-first: create fresh plus/minus `AgentSystem`s and fresh contact solvers for each
generation episode, attaching persistent caller-owned brains. This resets private solver phase and
GPU tick without a library change. Setup/compilation may be expensive and is included in rates.
Pooling is deferred until an explicit reset design passes the same tests.

1. Resolve noise, opponents, tile permutation and episode seeds on CPU, in fixed order.
2. Write policy tables. Build both identical lane definitions/systems and fresh settings/stats/
   counters/solvers, with exception-safe ownership.
3. Spawn every record in fixed global index order, using canonical initialization for age, flags,
   velocity, memory and timers; one contiguous record upload per lane. Fresh buffers start zero.
4. Select/cache a one-step chain via public `advance(dt)`/`getComputeStages`; assert one step was
   selected. Submit exactly T copies in guarded batches, not further accumulator-derived counts.
5. Copy both B-counter buffers into one staging buffer, submit copies, **map once after both lanes**.
   No map/inspect/`readCounters()`/stats readback inside the production step loop.
6. Aggregate and update only the evolving species; log/export. Destroy lane systems before buffers
   they bind; explicitly destroy solvers and settings/counters. Destroy brains only when run ends.

Use `try/finally` for mapping, failures and cancellation. Validation can inspect records/neighbours/
solver counters outside the production contract; diagnostic readback runs are not timing evidence.
Calibration uses disposable systems, never scored episode state. Fresh research systems follow
warm-up, so exactly T scored steps start at tick zero. R1 tests fresh episodes after unrelated
calibration/episodes against a fresh-process reference with the same brains.

## 7. Pure ES optimizer

`tools/agents/es.mjs`: pure functions, no GPU/filesystem/scenario imports. Reuse confirmed parameter
packing helpers and `TRAIN_DEFAULTS` Adam constants from `mlp-train.mjs`; no new dependencies.

**14.2 implemented, 2026-10-18:** `es.mjs` exports `noise`, `tupleHash`, `centeredRanks`,
`gradient`, `createAdamState`, `adamAscent`, `masterParams`, `candidateNetworks`. Noise version
`es-tuple-box-muller-v1` mixes [0x45530101, stream, seed, generation, direction, even parameter index];
stream 1 selects radius, 2 angle. Fields are u32; no seed arithmetic packing. Allocation/work per
vector capped at 2^24 parameters. Integer goldens have an independent BigInt oracle; floating
goldens recorded on Node v24.12.0 use 2e-15 tolerance. Candidate shape validation matches current
MlpBrain dimension/activation limits, with existing f32 layer packing. Masters/moments stay f64.
Adam updates are transactional and reject nonfinite intermediates; rate/decay/sigma are explicit.
U51 has ten headless tests; watched-failing mutations detect index-ranked ties, missing factor 2,
wrong ascent sign and omitted coupled decay. The CPU repeat test is not GPU training acceptance.

- Master vectors, noise, moments and reductions use float64, fixed k/j loop order. Convert only
  candidate networks at the f32 boundary; retain float64 archive masters. Log plus/minus f32 weight
  collisions and norms; stalled learning is not explained from weight magnitude alone.
- `noise(seed,t,k,n)`: documented versioned u32 tuple hash with separate fields; no unchecked
  `t*4096+k` packing or doubling seeds that aliases half the u32 domain. Check domain/budget bounds.
  Box–Muller uniforms avoid log(0). Golden integer-hash fixtures are exact; floating fixtures pin
  the supported runtime. JS transcendental bit equality across all engines is not promised.
- `centeredRanks(scores)`: deterministic ascending sort; equal scores share average occupied rank.
  For N = 2K, utility = averageRank/(N−1) − 0.5. All-equal scores yield exactly zero; reject nonfinite
  scores. Sorting index order may aid bookkeeping but must not invent utility differences.
- `gradient`: §3 formula in fixed order. Equal mirrored utilities contribute zero, including within
  partially tied populations. Rank invariance means strictly increasing transforms preserving ties.
- `adamAscent`: separate moments/timebase per species, increment on its turn only. Apply `g − λx`
  as coupled L2 penalty, **not AdamW**. Zero fitness gradient does not mean no movement with old
  momentum or nonzero decay. Test those separately.
- Refuse σ ≤ 0, rate ≤ 0, decay < 0, nonfinite vectors/candidates/moments/updates before GPU upload
  or export. Golden Adam steps and a known quadratic maximum test direction and normalization.

## 8. Schedule, seeds and archive

Even zero-based t evolves prey, odd t wolves. After `(t+1) % H == 0`, append each species' current
snapshot. Initial archive member 0 exists before t = 0. Choose current with probability 1−h,
otherwise uniform archive, independently per pair and shared by all E replicas/signs. This starts
immediately; start/current equality initially is harmless. h = 0 disables sampling. Require M ≥ 2;
over cap remove index 1, retaining the start. Log all IDs/evictions. No opponent grouping by wolf block.

Use versioned separate tuple fields for master seed, experiment domain, purpose, t, k and replica.
Separate perturbation/spawn/opponent/permutation streams. Precompute numerical training episode
seed manifests for the requested budget; compare with fixed validation/final-test manifests and
reject or deterministically regenerate collisions with recorded retry counters. Domains alone do
not prove numerical seeds differ. Detect unintended cross-pair episode-seed reuse; plus/minus reuse
is intentional. Reserve generation/u32 bounds; refuse overflow. Persist noise checksums.

Validation seeds guide settings. Final test seeds do not tune T/E/σ, margin, architecture, stopping
or checkpoint selection. Research manifests fix seeds and budgets before final outcomes are read.

## 9. Trainer and guarding

Implement `tools/agents/coevolve.mjs`, `npm run agents:coevolve`, and small pure config/schedule
helpers. Before replacement 14.0, examples are illustrative, not measured presets:

```text
npm run agents:coevolve -- --name h1 --start clones --config research-config.json
npm run agents:coevolve -- --name h2 --start random --hidden 16x16 --config research-config.json
```

Resolved config explicitly includes K/E/A/P/W/L/C/g/T/dt/thinkEvery/budgets, environment/catch
version and contacts options, optimizer, seed manifests, generations, archive/export settings.
Canonicalize before hashing. Refuse unknown options, missing explicit profile, incompatible clone
shape, existing output directory, row/device overflow and unsafe layouts. No silent population,
adapter, episode-length or environment fallback. Bound per-arena catches conservatively by PT;
refuse bounds ≥ 2^32. Nonfinite runtime state, device loss or any WebGPU error marks run incomplete.

Reuse film guard prediction/calibration/cache primitives through tools code. Valid calibration
counts must obey its ≥64/multiple-of-16 contract **and** actual arena layout constraints; arbitrary
quarter-population layouts are not automatically valid. Adapter/driver/build/network/layout/
thinkEvery/budget/environment/contact/grid-resolved settings form the cache identity.

Calibrate disposable ordinary, dense/wall/coincident and candidate/frozen-row workloads. Record
compilation warm-up separately; conservative step maxima gate probes, not mean timings. Initial
submission target 500 ms (never over 1,000 ms); use `floor`, not rounded-up batching. Wait/check
every batch; reduce on overruns. Stop on a measured single step or batch above 1,500 ms. A post-hoc
check cannot prevent the first timeout; never submit a probe predicted unsafe.

Report fresh setup/compile, CPU packing, both GPU lane episodes, readback, optimizer, exports and
end-to-end wall times. Queue-completion wall time is not GPU timestamp time. Record warm/cold/cache
conditions. A generation includes **both signs and all replicas**; checkpoint-inclusive research
wall time is separate. No browser/minutes-per-run promise before replacement measurements.

## 10. Evaluation and research protocol

Keep existing `agents:evaluate --students <dir>` backward compatible as a legacy diagnostic. Add
manifest-aware, arena-matched evaluation with explicit environment settings. Loading the same model
into the legacy 4K world is not a training-environment comparison merely because shapes match.

For selected generations evaluate evolved prey against rule wolves, clone wolves and evolved
wolves; rule/clone prey against evolved wolves; rule/rule and clone/clone references. Include the
label rule without wander where plan-2 asks. Store both opponent identities, environment and raw
seed-level catches. Use identical seeds in paired baseline comparisons and fresh episode state.

`agents:crossplay` plays prey generation i versus wolf generation j, in the same environment; export
raw data, mean/interval matrices and generation-0 anchors. Archive-on/off runs share environment,
budgets and master-seed lists. Preregister a scalar forgetting measure against fixed historical
anchors (sign-adjusted catches per species); a heatmap alone does not establish reduced cycling.
Freeze anchor selection before reading final results.

- At least 20 independent held-out episodes per pairing. Repeated training runs use multiple master
  seeds, report per-run results and between-run uncertainty; bodies/replicas are not independent
  training runs. Number of runs and budgets are frozen before 14.5.
- Fitness is catch count, **not mathematically equivalent to average survival time** with respawns.
  Report catches and rates; lifetime survival needs explicit counters before it can be claimed.
- Implement deterministic paired bootstrap intervals on seed-level differences: 10,000 resamples,
  fixed separate analysis seed, 95% percentile bounds. Also report marginal mean intervals. Test
  indexing/constant samples/known shifts; no new dependency needed.
- Primary superiority: paired improvement interval strictly favorable to zero. This explicitly
  refines plan-2's conservative marginal-nonoverlap rule; report marginal intervals too. H1 claims
  require both species' preregistered matchups, not a favorable subset.
- H2 requires a preregistered noninferiority margin against clone baselines. No significance is not
  equivalence. Checkpoint choice, generation budget, primary tests/margins and multiple-comparison
  handling are specified before final testing. Exploratory σ sweeps use validation seeds only.
- Exploit diagnostics include wall share **over time**, overlap/penetration, nearest prey, speeds,
  rest fraction and catch locations. A final-frame distance alone does not prove an episode-wide
  strategy or causal exploit mechanism.

## 11. Invariants, test allocation and negative controls

Existing steps own U43–U49 and A35–A41. Reserve **U50–U55, A42–A47** here; recheck concurrent
allocations before implementation. E/I labels name properties, not global test numbers.

| Property | Required evidence |
|---|---|
| I1 Independence | Every admitted layout, cross-tile sensing/catch/contact isolation including wall corrections/respawn |
| I2 Equivalence | One legacy-v1 tile vs `createHerd`, records/catches bitwise at spawn and during rollout |
| I3 Fair pairs | Identical-policy lane oracle with respawns, cadence, dense windows, coincidences and contacts |
| I4 Rows | Step-11 exact rows plus replica/opponent/table assignment checks |
| I5 Repeatability | Two fresh short same-runtime/device/config runs, identical checkpoint JSON/manifest bytes |
| I6 Readback | One score mapping after both lanes, none inside production loop |
| E1–E4 | Quadratic ascent, tied/neutral utilities, monotone invariance, deterministic noise/golden fixtures |
| E5 Counted once | Tile sum equals scenario catches, forced/reset cases nonvacuous |
| C1 Contacts | Catch boundary/rest, containment, frozen agents, limitations quantified |
| R1 Reset | Fresh episodes after unrelated calibration/episodes match fresh-process reference |

Headless `test/agents-coevo.test.mjs`: U50 config/refusals; U51 optimizer; U52 seeds/schedule/archive;
U53 checkpoint/model identity; U54 statistics/evaluation; U55 lifecycle/guard orchestration. Include
in `test:agents` when implemented, not in this documentation change.

GPU agent-tools suite, Node/NVIDIA, Node/AMD and Chrome/AMD: A42 isolation, A43 equivalence, A44
neutral/reset, A45 counters/row/replicas, A46 contacts/catches, A47 short-run repeatability/readback.
Three generations exercise both species; use small H deliberately to exercise archive operations.
Sampled isolation checks supplement, not replace, boundary fixtures and layout proofs.

Watched-failing controls: sign seed/cadence/tick skew; adjacent-tile sign mapping; mismatched respawn
IDs; index-ranked ties; omit row-capacity +1; world-wide respawn; dropped wall/containment; stale
contact phase; double catch counting; per-step readback. Each must fail the intended assertion and
be restored. No new tests are claimed as implemented by this design revision.

## 12. CLI errors and failure behavior

Pin implemented wording in tests; errors start `coevolve:` and name offending values/constraints.
Refuse nonfinite optimizer inputs, integer/seed overflow, A ≠ 2KE, invalid alignment/geometry,
unknown versions, unsafe gap, counter/device/row overflow, incompatible models, seed-manifest
collisions, invalid archive probabilities/caps and incomplete explicit research settings. Refuse
unsafe guard predictions before submission. Never silently change the experiment to make it run.
Checkpoint export failures are errors, not successful training artifacts. Preserve failure metadata
and cleanup all resources; do not overwrite unrelated or previously completed outputs.

## 13. Build order

| Phase | Delivers | Exit |
|---|---|---|
| **14.0 reopened** | Replacement revision-2 S7, scripts only, §16 | Correctness before measurements, raw evidence on three pairs, new settings proposed |
| **14.1 ✅** | Production lane/arena factory, legacy/contact modes, lifecycle | I1/I2/I3/E5/C1/R1, three pairs (Node NVIDIA/AMD, Chrome/AMD 40/40), watched failing; Q8 semantics settled |
| **14.2 ✅** | Pure optimizer and helpers | E1–E4, headless/golden fixtures (10/10) and refusals |
| **14.3 ✅** | Schedule/archive/trainer/guard/logs/checkpoints | Review findings closed with regressions (below); infrastructure evidence is not research acceptance |
| **14.4** | Arena evaluation, clone opponents, crossplay, intervals, README | At least 20 held-out episodes per validation pairing; raw-data replay |
| **14.5** | Preregistered H1/H2/H3, contacts ablation if scheduled | Results, uncertainty and verdicts recorded even on failure |

**14.3 closure, 2026-10-20.** The re-review found two defects the repairs had not reached, both
now fixed with regressions watched failing: the archive appended only the species that had just
evolved (half the snapshots at H = 5; with even H, prey never past its start), and models carried
the custom stages' indexed names (`bounds.0`, `steer.0`), so H1's clone start always failed and no
checkpoint loaded under the demo's definition. `modelNames()` maps the four custom fields onto the
demo's names and refuses any other drift (§5). Generation manifests now carry the hashed experiment
(an evaluable environment on their own), the archive keeps an index of IDs and model hashes, the
log holds archive IDs rather than weights, and one upload per species packs the opponent rows.
Evidence: headless agents 132/132; `validate-coevolve.mjs` on Node/NVIDIA and Node/AMD, each with
two byte-identical ten-generation runs, a clone start with E = 2 and archive evictions, and I6
instrumented (one mapping per generation; a per-batch readback injected on the GPU is caught).
Known limits, not blockers: the engine keeps every compute stage it creates (88 per generation;
about 1.6 MB of process memory per generation on NVIDIA), a library matter for long budgets; rule
wander and respawn draws depend on body index and tick only, so held-out episodes differ from
training in their spawn alone, to settle before 14.5's freeze.

Replacement 14.0 implements spike versions needed to measure end to end; it does not complete
production phases. Promote reviewed behavior, not copied bugs. Fresh-system setup costs may reject
an approach; report them instead of weakening reset guarantees. Resume/pooling, obstacles, stamina,
simultaneous updates, live browser training and films remain deferred designs.

## 14. Refinements of plan-2

| Parent formulation | Step-14 contract |
|---|---|
| Survival share / wolf catches | Chosen zero-sum catches, no survival-equivalence claim |
| Independent adjacent candidates | Separate signs at identical indices/coordinates |
| Same spawn/opponent | Complete matched conditions and neutral rollout oracle |
| Centered ranks | Equal average utilities for ties |
| Archive champions | Historical snapshots, no assumed quality |
| Nonoverlapping marginal intervals | Report marginals plus paired differences, preregister decisions |
| JSON and ONNX | JSON always, optional ONNX, explicit environment manifest |
| Contacts later | Step 13 exists; contacts/catch semantics live in scenario tools |

No library change follows. Scope expansion or rejected architecture returns to the user rather
than silently modifying the engine or research hypothesis.

## 15. Decision ledger and pending freeze

**2026-10-17:** Q1–Q5 accepted (§1); those decisions remain, not the old measurements.
**2026-10-18:** user requested full revision and discarding original phase 14.0 results. Old S7
acceptance/defaults/conclusions withdrawn; S8/S9 remain independent evidence.

Before 14.1: replacement S7 validates lane/reset architecture, proposes population/T/E and tuning
ranges, and tests proposed contact-v1 Q8 margin/order. Final settings and any trade-off requiring
scope change are recorded explicitly before production implementation/research acceptance. Browser
feasibility is undecided. No final research budget or obstacle/stamina design is implicitly approved.

## 16. Replacement phase 14.0 protocol — no accepted results

`npm run spike:s7` now invokes the **protocol-2 replacement**. Its first validation is recorded in
[`test/spikes/coevo/RESULTS.md`](../test/spikes/coevo/RESULTS.md): 14.0 remains open, with safety stops
on Node/AMD and Chrome/AMD, incomplete correctness coverage and no accepted settings. Original
source remains in git history, and original results remain withdrawn. Replacement scripts stay in `test/spikes/coevo/`, declare a
protocol version and save raw evidence, source hashes, commands, seeds, devices, timings and errors.

### Gate 1: correctness before performance

Neutral lanes, legacy equivalence, isolation, forced catch/respawn, contact boundaries/containment,
row/replica/archive mapping, full reset and tied ranks. Include coincidence, wall/dense/overflow
windows and off-cadence newborns. Compare after unrelated calibration and episodes. Check catch
lookup against brute-force controlled fixtures. Demonstrate negative controls. Any failed invariant
blocks performance/learning acceptance for that configuration.

### Gate 2: throughput, memory and setup costs

Exploratory screening values, **not defaults**: K in {8,16,32}, E in {1,2,4}, P in {192,256,384},
W = 16 where BW alignment permits, T in {300,600,1200}; thinkEvery initially 3. Reference density
`L = 160 sqrt((P+W)/4096)` is an explicit starting choice, not optimality evidence. Resolve a safe
gap from §3. Test C2/H2 shapes, both environments and ordinary/dense states. Preregister a staged
screening subset; a full Cartesian sweep is not required. Filter invalid layouts, not hidden padding.

Measure Node/NVIDIA, Node/AMD, Chrome/AMD: separate warm-up, at least five end-to-end generations
per reported setting, median/range and setup/episode breakdown. Report active lane B(P+W), total A,
peak two-lane memory, table memory, batch sizes and cache conditions. Include fresh construction/
compilation; do not transplant rates from another architecture or contact mode.

### Gate 3: paired score quality

Fixed actual perturbations/frozen opponents; ≥20 validation spawn seeds per direction on shortlisted
T/E/σ. Store replica catches, paired differences, tie fraction, seed variation and direction ranking
stability; compare directional signal against residual **paired** noise. Clone-only unperturbed CV
cannot choose E. Record f32 perturbation collisions and update norms. Tune only on validation seeds.

Follow-up protocol **2.1** screens both species at E=2/4, T=600 and sigma=.1/.25 on a new
seed domain, preserving 2.0 evidence separately. Before observing those outcomes, exploratory
quality bars are registered as split-half direction-rank correlation >=.7, direction SD / residual
paired RMS >=1 and at least four nonzero bootstrap intervals. Both species must pass the same
setting; Gate 4 additionally requires its preregistered E=2/T=600/sigma=.1 setting to pass.
These are conservative screening decisions, not a claim of research-level statistical proof.
Do not silently choose a different smoke setting from favorable results.

### Gate 4: smoke learning and diagnostics

Only after gates 1–3, run a short alternating trajectory, evaluate before/after against fixed clone/
rule anchors on validation seeds with paired intervals, and repeat a short run exactly. Collect
wall/overlap/rest histories and visual snapshots where feasible. A short trajectory is not H1/H2
proof; nearest-distance statistics alone are not causal exploit evidence.

### Evidence and verdict

Accept settings only from new raw evidence passing all gates; report unsafe/failed settings rather
than dropping them. Research runs, browser feasibility and learning conclusions remain pending.
**There is intentionally no retained original S7 result table in revision 2.**

Protocol 2.1 persists completed generations/samples/settings/checkpoints in fsynced append-only
Node JSONL and immediate browser console events. A later throw no longer erases earlier raw samples;
recovery labels interrupted runs partial, never accepted. Compilation completion is separately timed
without executing simulation steps. First four submissions are singles, subsequent batches are at most
two, with a 25 ms target, slowest-observed cost, host yield and the unchanged 1,500 ms hard stop.
Even a single submission can stall: this policy is not a safety guarantee.

Controlled brute-force fixtures now explicitly expose resting-nearest occlusion: an eligible second
wolf within reach can be hidden by the nearest resting wolf. `contact-v1` retains this limitation;
neither an exhaustive post-move catch rule nor Q8 acceptance is implied. Dense solver fixtures check
private overflow and exact CPU contact counts at advancing ticks, rebasing the CPU oracle on each
observed GPU input to avoid confusing accumulated floating-point drift with wrong window selection.
