# NullGraph Agents — C1 step 5: scaling limits and proven bottlenecks — Technical Design

**Parent:** [`agents-design.md`](./agents-design.md) §13, C1 step 5.
**Builds on:** [`agents-gpu-loop-design.md`](./agents-gpu-loop-design.md), phases 3.0–4.5;
[`api/agents.md`](./api/agents.md); [`api/agent-grid.md`](./api/agent-grid.md).
**Direction agreed with the user:** 2026-10-12 — characterize scaling and limits, not chase a
wishful population/timing target. Detailed experiment choices remain draft.
**Status:** Design only; no new measurements or step-5 implementation claimed.
**Sub-step, 2026-10-13:** the user resumed step 5 ahead of step 7, part 2, for deterministic
neighbour lookups, designed in [`agents-ordered-grid-design.md`](./agents-ordered-grid-design.md)
(phases 5.5–5.8, complete 2026-10-13). The study below (phases 5.0–5.4) stays deferred.
**Outcome:** explain how far the system scales, under which conditions, what limits it first,
and which finer agent/renderer/engine improvements the evidence supports. No core rewrite planned.

## 1. Purpose and scope amendment

The GPU rule loop, instances, picking, labels and inspection already work. The user reports good
frame rates at **65,536 agents**. Reproduce that configuration first: exact count, device, density,
cadence, overlays and sustained simulation throughput have not been independently verified for
that observation. Do not invent these settings or treat 65,536 as a proven ceiling.

Step 5 asks:
- How do GPU execution, CPU submission, rendering and memory scale with population?
- What changes with density, clustering, sensor workload and think cadence?
- Where does simulation fall behind even while presentation remains smooth?
- Where does query quality degrade, and what cost buys better coverage?
- Which limits are workload-specific, and which support reusable engine improvements?

The original 20,000-agent/A9 <=2 ms NVIDIA target is **historical comparison data, not a step-5
pass/fail condition**. 20K may be a comparison point; 65,536 is an observed starting point. Neither
is a required target or maximum supported count. Historical A9's network work belongs to C2;
this investigation uses the existing rule brain.

The previous draft's zero-overflow soak and timing gates are withdrawn. Measure latency and
approximation honestly, rather than forcing a selected preset to pass. Keep existing interactions;
an exact-20K preset, new starter or UI polish is not required to complete this investigation.
Remaining product packaging stays separately tracked, not silently marked complete.

### Constraints and non-goals

Phase 4.5 passed 106/106 agents GPU checks on each host, with zero GPU errors, and a reviewed
39-step/21-screenshot drive. Host update/camera/submission work measured 0.570 ms at 4,096 agents
and 0.599 ms at 16,384; those are not GPU or whole-process CPU timings. See the handoff.
Dense flocks have overflowed scan budget 64; smooth FPS does not establish query exactness.
Whole-entry agents.js is 15,473 B min+gzip, frame subset 12,330 B. Preserve the conservative
16,000 B budget (527 B remaining); instrumentation belongs outside the shipped entry.

No speculative optimization, universal agent-count/FPS promise, core rewrite, neural brains,
training/ORT, evolution, free lists, collisions, pathfinding, eyes, 3D motion or new label subsystem.
Keep public APIs and GPU-state ownership intact. Broader architectural/API changes require a
separate evidence-backed amendment. No commit, push, publication or deployment is authorized here.

## 2. Deliverables

| Deliverable | Useful result |
|---|---|
| Reproducible harness | Production stages, frozen settings, raw results and both-host support |
| Scaling report | Population/density curves; GPU/CPU/render, throughput and memory separated |
| Conditional capacity map | Highest safely tested healthy points and first observed limits per workload/device |
| Query-quality report | Overflow, nearest-set agreement and cost/quality tradeoffs |
| Bottleneck register | Evidence, confidence, subsystem, proposed improvement and priority |
| Optional narrow improvement | Like-for-like benefit and correctness proof, or justified no-change outcome |

Proposed harness: `D:/orion-realms/null-graph/test/agents-perf/`. Store samples, snapshots and
screenshots in a separate validation directory and link the eventual report. New commands/schema
are documented when implemented, not claimed to exist now. Discover the separate site's current
paths/Git status before work; preserve unrelated changes and the user's running server.

## 3. Experiment design

### Reproduce and freeze the observed workload

Record the actual 65,536 setting/count, adapter/backend/browser/OS, world bounds, seed, species split,
policy, speeds/radii, cell layout, scan budget, sensor list/k, fixed dt, thinkEvery, frame clamp/max
steps, time scale, canvas/DPR, meshes and overlays. Keep existing ordered catch/respawn stages.
Policy changes are separate experiments, not invisible tuning. Preserve IDs and instance mappings.

### Separate the axes

| Axis | Experiment | Isolates |
|---|---|---|
| Population, fixed density | Increase count and world area proportionally | Dispatch/data-size/locality scaling |
| Population, fixed world | Increase count without expanding bounds | Density and neighbour pressure |
| Distribution/state | Uniform, deliberately clustered and evolved flock states | Hot cells and state-dependent behaviour |
| Query workload | Valid cell sizes, budgets, sensor count and k | Spatial cost versus coverage |
| Think cadence | Observed setting, cadence 1 and selected alternatives | Throughput versus temporal responsiveness |
| Render workload | Simulation-only, instances, culling/drawing and overlays | Compute versus rendering/text cost |
| Interaction | No selection, hover, selection/inspection | Interaction and diagnostic overhead |
| Duration | Startup, warmed and sustained evolved operation | Startup cost, clustering and drift |

Start below/at/above the reproduced workload with geometric increments, then refine around
inflection points. Do not exhaust every combination: screen axes, then investigate material effects.
Changing density, policy or cadence is a different workload, not a like-for-like optimization.

### Safe exploration

Use bounded-duration trials, cleanup and conservative increments. Estimate cost and allocation
from completed smaller trials and respect actual device limits. Retain existing stress safeguards.
Do not knowingly launch unbounded dense scans or risky million-agent iGPU workloads.
Predeclare trial timeouts and stopping rules for slowdown, memory pressure and errors per device.
If no limit is safely reached, report **at least the highest tested count under these settings**.
Timeout/device loss is an observed failure, not proof of an exact ceiling or its underlying cause.

## 4. Measurement and attribution

Full rule-step interval: **grid/reset/prefix/scatter → all queries → sense → RuleBrain → act →
catch/respawn app stages → tick → instances**. The comparison sample includes one step and one
instance dispatch. Also measure production multi-step frames, where instances run once per frame;
do not imply identical per-step cost for these schedules.

Use GPU timestamps where available, following S3/profiler conventions. A test encoder may execute
public production nodes without core changes, but must respect bindings, enabled state, dispatch
and reset ordering, and match ordinary engine output before its timings are trusted. Include resets
and instance output; resolve/copy/map after the measured interval. Without timestamps, label
submit-to-completion wall time as fallback, not GPU execution time.

Measure the whole chain first, then grid, queries, sense/think, act/app/tick, instances, visibility
and rendering. Stage instrumentation can alter pass boundaries/cost: quantify overhead and compare
with uninstrumented totals. Isolated stage times need not sum exactly to production time.

Suggested initial sampling: at least 120 warmup steps, 200 valid samples and three independent runs
per selected configuration; adapt to timer resolution/cost and document deviations. Cover initial
and evolved states, repeat important comparisons across fixed seeds and report sustained-run duration.
No arbitrary zero-overflow five-minute acceptance gate is imposed.

Persist raw samples, median/p95/tails, variation, resolution, warmup/sample counts, power conditions,
features and source revision/dirty status. Do not merge devices or workloads into unexplained totals.
Use NVIDIA/Node-Dawn and AMD/Chrome where available; record actual adapters rather than assuming the
runner selects them. Missing targets/timestamps are explicit evidence gaps, never fabricated results.

Measure FPS/frame intervals separately from completed simulation steps per wall second, simulated-
time/wall-time ratio, steps per frame and clamp/dropped-time events. Derive intended simulation rate
from dt/time scale. Smooth rendering alone does not prove simulation keeps up.
Record CPU timer boundaries and separately identify excluded UI/inspection work. Report tracked
buffer allocations and device-limit headroom, not alleged total residency/driver memory. Check
interaction responsiveness at representative loads without routine full-state frame readbacks.

## 5. Density and query quality

Current coverage requires senseRadius <= cellSize for the surrounding 3×3 query. Smaller cells
require a separate coverage design. Grid indices are retained; bounded per-cell scans may omit
candidates. Distance/index ordering is exact only within the scanned subset; overflowing subsets
can vary with atomic scatter order. Do not claim global exactness or deterministic overflowing runs.

Overflow counters measure truncated grid entries/steps, not unique missed neighbours or personal
agent decision error. Report deltas per elapsed steps plus occupancy distributions. For tractable
sampled snapshots/fixtures, compare non-truncated results to brute force, bounded results to an
oracle using actual scatter order, and both to the full-neighbour reference for quality.
Measure nearest-set recall, missing nearest candidates and distance error, split by species/density;
define empty-set handling and sampling. Never brute-force a whole 65,536 population on the normal frame.

Publish **cost/quality curves**, not a mandate to eliminate overflow. Approximation stays visible.
Finite records, speed/bounds and lifecycle correctness remain hard safety properties. Observe
policy/catch behaviour, but screenshots alone do not prove behaviour equivalence.

## 6. Proven bottlenecks and conditional improvement

An expensive stage is a candidate, not automatically the scaling limit. Establish a bottleneck
with attribution, scaling trends, controlled changes/ablations, repeat runs and uncertainty analysis.
Removing required work is diagnostic, not an equivalent optimization. Exclude instrumentation or
reduced quality as explanations for apparent speedup.

Each record names workload/device range, evidence, uncertainty, affected subsystem, mechanism
hypothesis, proposed fix, priority and correctness/bundle/API risks. Separate **observed**, **supported
hypothesis** and **untested proposal**.

Only improve a narrow path when measurements justify useful benefit. Grid-ordered queries, fused
scans, dispatch overhead, locality or visibility/overlay cost are possible investigations, not
predetermined conclusions. (Grid order was taken up on 2026-10-13 for determinism rather than
speed: [`agents-ordered-grid-design.md`](./agents-ordered-grid-design.md).) Query gating is not automatically safe: app stages consume neighbours.
Validate identical workload/quality settings, multiple scales and both hosts where possible; report
regressions and variation. Preserve APIs, budget and ownership. Engine-wide opportunities become
separately scoped follow-ups. **No worthwhile optimization needed** is a valid result.

## 7. Completion and validation

Completion means a reproducible operating envelope and evidence-backed bottleneck register with
explicit limits/gaps, not reaching a required count, FPS or millisecond target. Required evidence:
- Reproduced user-observed workload, or exactly what could not be reproduced.
- Population/density/state/cadence results separating GPU, CPU, render and simulation throughput.
- Highest safe tested points, observed inflections and stop reasons by device/workload.
- Query cost/quality tradeoffs, raw samples, metadata and instrumentation/repeatability checks.
- Correctness/error status, allocations and representative interaction/lifecycle checks.
- Ranked finer improvements, or an evidence-backed no-change recommendation.

For implementation, run library/TypeScript builds, affected headless checks and both GPU hosts
where available; preserve picking/labels/teardown regressions. Measure bundle sizes and before/after
results. Diagnostic readbacks stay outside the ordinary frame path. Documentation-only work checks
links/whitespace without claiming GPU validation. Existing commands include build:lib, test:agents,
test:core, test:agent-kit, test:gpu, test:gpu:browser and render/drive; inspect runner options first.
Update the parent/handoff with actual findings.

## 8. Investigation phases

| Phase                               | Work                                                                       | Exit evidence                                        |
|-------------------------------------|----------------------------------------------------------------------------|------------------------------------------------------|
| **5.0 — reproduce and instrument**  | Capture observed workload; validate timing/output equivalence and overhead | Reproducible setup and measurement limits            |
| **5.1 — map scaling**               | Safe population, density, cadence and rendering sweeps                     | Conditional operating envelope and inflections       |
| **5.2 — explain limits**            | Query quality, stage attribution, ablations and repeat runs                | Proven bottlenecks versus hypotheses; tradeoffs      |
| **5.3 — improve only if justified** | Optional focused change or explicit no-change result                       | Like-for-like benefit/correctness proof or rationale |
| **5.4 — report findings**           | Evidence, ranked engine opportunities, gaps and follow-ups                 | Reproducible report and updated roadmap/handoff      |

No commits/publication unless requested. README/starter/showcase packaging remains separate product
follow-up; this study does not claim completion of undelivered artifacts.

## 9. What follows

C2 adds neural brains, record/train/export tools, ONNX loading, optional ORT and Rules/Network
comparisons. It can reuse this measurement harness, but rule timings do not prove network performance.
Evidence-backed agent/engine improvements can be prioritized separately before or alongside C2.
Evolution remains later work.
