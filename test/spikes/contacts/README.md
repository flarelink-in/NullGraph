# S8 — contact candidate sources

Scripts-only phase 12.0. Build the library before running:

```sh
npm run build:lib:prod
npm run spike:s8 -- --json test/spikes/contacts/out/node.json
node --test test/spikes/contacts/metrics.test.mjs
npm run drive -- --serve . test/spikes/contacts/index.html --wait-for "S8 done" --timeout 2400 --out test/spikes/contacts/out/chrome
```

Select `checks`, `cost`, `overlap`, `pileup`, or `jitter` with `NG_SPIKE_PART` under Node or
`?part=overlap` in the browser. Unknown parts are refused. Node writes the numeric measurements
to `out/node-<part>-measurements.json`; browser measurements and checks are available as
`globalThis.__S8_MEASUREMENTS__` and `globalThis.__S8_RESULTS__`. `out/` is local evidence, not shipped.

## Corrections from the initial spike

- Overlap is measured after **each of the 600 steps**, with the full history, run maxima, and
  final-state metrics reported separately. Exact X-sorted CPU sweeps prune only candidates that
  cannot overlap or improve the nearest distance. The headless test compares against all pairs.
- The standalone grid contact pass visits **every** entry in each cell and rebuilds the grid
  after motion and before every Jacobi iteration. Sensor queries intentionally retain their
  bounded scan budget. A 256-agent GPU check compares contact counts against brute force.
- All peer sources use the same contact-count averaging. The contact normal is normalized after
  max-push clamping. This is a positional Jacobi prototype with zero compliance, not a complete
  XPBD multiplier solver or a production dense-pile-up solver.
- Browser runs use the same harness and error watch as Node. Completion alone does not establish
  a pass: inspect `__S8_RESULTS__.failed`, `gpuErrors`, and the drive report.

## Interpretation

Cost is **queue-completed wall-clock milliseconds per step**, including host submission overhead,
not isolated GPU timestamp time. Sequential modes follow their own evolving trajectories, and
negative deltas or outliers should not be interpreted as a reliable speedup. Do not compare
timings while another GPU workload is running.

All tested contact modes reduce ordinary-herd stacking. The k=8 list does **not** establish a
dense-pile-up solution: none of these configurations clears the 256-agent stress pile within
180 steps. Two exhaustive-grid iterations improve ordinary-herd residual penetration, but the
production contact solver still needs a convergence design. Query-only lists are useful as the
general step-stage/custom-sensor contract, not as a guarantee of complete physical contacts.