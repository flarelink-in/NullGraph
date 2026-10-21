# Replacement S7 — first validation, 2026-10-18

**Current verdict (reviewed 2026-10-18):** Chrome/AMD protocol-2.1 Gate 1 numeric evidence
confirmed 77/77 with no WebGPU errors (`out/chrome-revision21/evidence.json`). Completed full-run
outcomes are in the superseding section below: safety/quality remain unaccepted. Later 14.1
contact-v2 correctness is separate (Node 80/80; Chrome/AMD 40/40), not a new S7 training result.
Historical pending/no-14.1 statements below describe those earlier runs only. Infrastructure
progression was explicitly authorized; no contact-v1 rates/settings transfer to contact-v2.

## Follow-up protocol 2.1 (2026-10-18)

The historical 2.0 observations below remain unchanged. New code completes controlled catch lookup
(coincident/tie/resting/dead/multiple prey/boundaries), watched-failing omitted containment and
off-cadence newborns, archive table byte identity and replica/permutation assignments, plus dense
overflow and independent CPU contact-count/window checks over four advancing solver ticks.
The CPU oracle is rebased on each actual GPU input; an interrupted development run using accumulated
CPU trajectories diverged numerically and is not acceptance evidence.

A real limitation is now quantified: in a controlled scene with one eligible wolf inside contact
reach, a nearer resting wolf causes **0 actual catches versus 1 exhaustive eligible-wolf catch**.
That is preserved nearest-list semantics, not a newly fixed exhaustive post-move rule. Q8 remains
provisional, and no catch-rule/environment change is silently mixed into old measurements.

Evidence durability: fsynced append-only Node events, immediate browser console checkpoints,
per-generation/per-noise-sample/per-setting records, smoke checkpoints and diagnostic samples.
Recovery of an interrupted development journal preserved **2 completed NVIDIA generations and 7 AMD
generations**, as well as the recorded unsafe submission. Pure durability/quality/math tests: **7/7**.
Library and IDE builds, syntax and whitespace checks passed during follow-up.

Submission growth is capped at two (first four singles), 25 ms target, slowest observed cost,
separate async compilation completion and host yield. The hard stop remains 1,500 ms. An early
follow-up without host yield hit **1,519.4 ms with n=1**, at scored step 42 of a chunk; the failed
batch and previous measurements survived. Conservative batching cannot guarantee no driver stall.

Exploratory 2.1 score screen is now both-species E=2/4, T=600, sigma=.1/.25 on a new seed domain.
Explicit quality bars gate learning, including the exact preregistered smoke configuration; completed
samples alone are not score-quality acceptance. **The new score screen and learning are not yet
accepted or fully measured.** Intermediate development runs were stopped for fixture fixes, retained
in `out/events-*.jsonl`, and are not acceptance runs. No concurrent timing experiments were used.

### Final correctness reruns

- Node/NVIDIA + Node/AMD: **154/154**, zero WebGPU errors, 154.6 s. Both adapters pass the
  complete implemented Gate 1 including new fixtures. Raw: `out/node-gate1-20261018110347558.json`,
  derived summary, `gate1-complete-console.log`, `gate1-complete-errors.log` and matching JSONL.
- Chrome/AMD: **77/77**, zero WebGPU errors, approximately 25 s. Numeric evidence explicitly
  inspected (not just drive exit): `out/chrome-revision21/evidence.json`; console/report/screenshot
  in that directory. Gate 1 completes without the earlier safety stop. This is correctness-only,
  **not** throughput/browser-training-feasibility acceptance.
- The interrupted all-gates follow-up journal `events-20261018110100889.jsonl` recovered **19
  completed NVIDIA generations**, zero recorded failures. It was deliberately stopped to run a
  final correctness-only validation with the intended command; it is not a completed screen.

The full 2.1 throughput and both-species quality/learning screen remains pending. New safety/host
yield/compilation policy materially changes overhead; do not reuse 2.0 rates as current rates.
**14.0 remains open; no 14.1.**

**14.0 remains open. No production settings or browser-feasibility acceptance.**
These are new protocol-2 observations, not reinstated original S7 results.

## Executed

- Library build and IDE build: pass. Pure spike math: 4/4 tests pass.
- Initial uncached correctness run: 64/64 checks across Node/NVIDIA and Node/AMD,
  zero WebGPU errors. Follow-up added E=2 neutral row/replica checks for both evolving species.
- Main Node run: **97/98 checks**, zero WebGPU errors, ~1,232 s. The one failure is
  the AMD measured-submission safety stop, not a GPU validation failure.
- Chrome/AMD: six legacy-equivalence checks pass, then safety stop during first neutral
  episode. Zero WebGPU errors. The drive report's exit 0 means clean browser plumbing,
  **not** numeric harness acceptance (the page reports FAIL).

## Gate 1

Both Node adapters pass rule legacy bitwise equivalence at 50-step intervals through 300;
neutral MLP lanes through 600 in legacy/contact environments, ordinary/dense-wall fixtures,
forced catches/respawns, isolation, containment, counters, reset after unrelated work, E=2
mapping for both species, tie neutrality, last archive-row capacity, and catch boundaries.
Deliberate seed/cadence/respawn-ID/double-count errors are detected.

This is **not yet full §16 acceptance**: exhaustive catch-lookup fixtures, a demonstrated
omit-containment failure, archive-selection identity fixtures and off-cadence newborn-specific
assertions still need expansion. Dense fixtures exercise contact overflow windows, but do not
yet assert their full coverage independently.

## Gate 2

Six preregistered settings each ran one warm-up and five measured generations per Node adapter.
Fresh systems/buffers/solvers are retained; only immutable pipelines are cached by complete
WGSL/entry/constants on the same device. No library changes. Cache misses and compile time
are recorded. Buffer totals are not total GPU memory (pipeline/texture allocation is unexposed).

Median end-to-end generation seconds (includes fresh construction):

| K/E/P/W/T, shape, environment | NVIDIA | AMD |
|---|---:|---:|
| 8/1/192/16/300, C2, legacy | 0.625 | 1.110 |
| 8/1/192/16/300, C2, contacts | 3.737 | 1.157 |
| 8/1/192/16/300, H2, dense contacts | 1.973 | 1.320 |
| 16/1/256/16/600, C2, contacts | 1.681 | 2.848 |
| 8/2/192/16/600, C2, contacts | 1.698 | 2.556 |
| 8/1/192/16/300, C2, contact-v1 solver off | 0.493 | 0.912 |

NVIDIA ranges include large first-use/outlier costs (up to 16.3 s at E=2); medians are **not**
a clean scaling curve or accepted defaults. Opponent species alternate across measured generations.
The geometry is identical within each mirrored pair, but unrelated candidates occupy different
physical tile coordinates and indices. No hidden 64-agent padding is used.

## Gate 3

NVIDIA completed eight settings (E=1/2, T=300/600, sigma=.02/.1), eight fixed actual directions
and 20 validation spawn seeds each. All observed perturbation collisions were zero.
Split-half direction-rank correlation ranges from -0.358 to 0.725; direction-SD/residual-paired-RMS
ratios range from 0.166 to 0.365. This is weak/noisy directional evidence, not accepted E/sigma.
Only prey perturbations were screened; wolf-direction screening remains required.

AMD completed seven settings; the eighth hit **2,166 ms** against the **1,500 ms** stop.
The console preserves completed setting summaries, but the current gate-level return loses the
partial gate-3 raw array when a later sample throws. Per-sample incremental persistence is a
required follow-up; do not claim those AMD summaries are complete raw acceptance evidence.

## Completed protocol-2.1 full run, 2026-10-18 (supersedes earlier partial status)

Evidence: `out/node-all-20261018111234870.json`, `out/phase14-full-report.json`, and authoritative
`out/events-20261018111234870.jsonl` (1124 complete events; no truncated tail). Full harness
172/174 checks, zero WebGPU errors, 3,519,662 ms. Both devices passed correctness and completed six
throughput settings. Both stopped in Gate 3: NVIDIA single-step 1933.6 ms; AMD single-step
1824.6 ms. Neither meets the unchanged 1500 ms submission requirement. No full-run Gate 4.

Completed Gate 3 screens (rank correlation / signal-to-residual / nonzero intervals):

| Device/species | E | T | sigma | Rank | Signal | Intervals | Quality only |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NVIDIA prey | 2 | 600 | .1 | .627 | .570 | 1 | fail |
| AMD prey | 2 | 600 | .1 | -.048 | .528 | 2 | fail |
| AMD prey | 2 | 600 | .25 | .970 | 1.708 | 6 | pass |
| AMD prey | 4 | 600 | .1 | .850 | .780 | 3 | fail |
| AMD prey | 4 | 600 | .25 | .857 | 2.913 | 7 | pass |
| AMD wolf | 2 | 600 | .1 | .595 | .474 | 3 | fail |

AMD's larger prey perturbations improve the measured score signal, but do not validate wolves or
establish improvement/training defaults. Remaining wolf screens were interrupted by the safety
stop. Evidence retains 20 NVIDIA and 116 AMD noise samples, including partial-screen samples.
**14.0 remains unaccepted.** User authorized starting 14.1 infrastructure with separately versioned
exhaustive `contact-v2` and fresh validation; none of these contact-v1 rates/quality results transfer.

### Historical Gate 4 (diagnostic only, earlier run)

NVIDIA completed ten alternating updates and an exact three-generation repeat (checkpoint
text byte-for-byte). Twenty validation episodes per matchup, clone/rule anchors, wall/overlap/rest
histories. No rendered simulation snapshots yet.

Paired catch change vs clone/clone (mean; bootstrap 95% interval):

- evolved prey / clone wolf: **-0.35 [-1.10, +0.45]**;
- clone prey / evolved wolf: **+0.20 [-0.50, +0.85]**;
- evolved / evolved: **-0.30 [-1.15, +0.55]**.

All include zero. No improvement or exploit-causality conclusion. This trajectory is diagnostic,
not accepted learning after a score-quality pass. AMD Gate 4 blocked by the safety stop.

## Chrome/AMD

Legacy equivalence passes through 300. A **1,709 ms** submission exceeds the stop during neutral
correctness. Gates 2–4 blocked. Next: more conservative submission growth and separately measured
first-use compilation; do not raise the stop or silently discard the unsafe observations.

## Raw evidence and follow-up

Ignored local evidence under `out/`:

- `node-all-20261018100129625.json` and its derived `-summary.json`;
- `node-all-console.log`, `node-all-errors.log`;
- `initial-gate1-console.log` and initial gate-1 JSON;
- `interrupted-pre-cache-console.log` (aborted review run, not acceptance);
- `chrome/console.log`, `chrome/report.json`, screenshot and extracted browser JSON.

Finish correctness fixture coverage; save partial results incrementally; improve conservative
submission/first-use handling; repeat AMD/Chrome without unsafe batches; screen both species and
stronger score signal, then reassess settings. Contact Q8 margin/order remain provisional. Do not
start production 14.1 from these observations.