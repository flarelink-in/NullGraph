# S7 replacement (protocol 2.1)

Original S7 results are withdrawn. This is spike code, not the production step-14 trainer.
`protocol.json` registers the staged screening subset before results are read. Design:
`../../../docs/agents-coevolution-design.md`.

From the repository root (PowerShell):

```powershell
npm run build:lib
node --test test/spikes/coevo/math.test.mjs test/spikes/coevo/evidence.test.mjs
npm run spike:s7
# Optional: a selected gate always runs gate 1 first.
$env:NG_SPIKE_PART='gate3'
npm run spike:s7
Remove-Item Env:NG_SPIKE_PART
```

Raw Node evidence is written under ignored `out/` with protocol, seed-level counters, timing batches,
checkpoint text, source hashes, runtime, adapter and check results. A failed correctness gate blocks
later gates on that adapter. WebGPU validation/device-loss events fail the harness.

Protocol 2.1 additionally fsyncs an append-only `out/events-<timestamp>.jsonl` after every completed
generation, noise sample, fixture, checkpoint and setting. Failures include the unsafe batch and prior
timings. The final JSON is convenient, but is **not** the only raw evidence. Recover an interrupted run:

```powershell
node test/spikes/coevo/recover.mjs test/spikes/coevo/out/events-<timestamp>.jsonl
```

Only a torn final line is tolerated; corrupt interior lines fail recovery. Recovered files are explicitly
partial, not acceptance. Browser checkpoints are published immediately as `S7 EVENT {...}` console
lines and in `__S7_EVENTS__`; preserve the drive console even if final JSON is never reached.

Submission policy: first four scored submissions are singles, maximum two copies, a 25 ms target,
slowest-observed per-step cost and a host-event-loop yield. The 1,500 ms hard stop is unchanged.
Async pipeline preparation is measured separately without advancing any simulation state; it cannot
guarantee the driver will not incur later first-use stalls.

The new exploratory noise screen tests **both species**, E=2/4, T=600, sigma=.1/.25 on a new seed
domain. This supersedes, rather than pools with, the prey-only 2.0 screen. The preregistered quality
bars in `protocol.json` must pass for both species at a common setting before Gate 4 runs; requesting
`gate4` also runs Gate 3. Failed quality does not become an accepted setting by completing samples.

Chrome (do not run concurrently with Node timing experiments):

```powershell
npm run drive -- --serve . /test/spikes/coevo/index.html --wait-for 'S7 done' --timeout 7200 --out test/spikes/coevo/out/chrome
```

Browser evidence is also published as `globalThis.__S7_EVIDENCE__` and emitted in the console as
`S7 replacement JSON ...`; preserve that raw console with the drive report. Drive exit zero alone
does not prove numeric checks passed: inspect `__S7_RESULTS__` / the published evidence.

Buffer memory comes from caller-side allocation tracking, including grid/scratch/tables. WebGPU
does not expose texture/pipeline memory usage; do not label buffer totals as total GPU memory.
Setup includes fresh stage/bind-group construction. A tool-local cache reuses immutable pipelines
only (full shader/entry/constants key on one device), never episode buffers or solver phase. Cold
cache misses/compile time are recorded; warm-up is reported separately. Diagnostic histories use readbacks and must
not be interpreted as production one-map throughput. `mappings` in generation results is the score
map count, not a claim that diagnostic maps did not occur.