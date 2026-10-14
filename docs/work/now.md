# Now — 2026-10-15

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2.** Steps 6–9 are built and committed (9.6, the last phase: `4e19a0f`).
  Step 5's scaling study is still deferred.
- **Step 10, the live switch** ([`../agents-live-switch-design.md`](../agents-live-switch-design.md);
  design `2b8f689`): Rules / Network / ONNX Runtime per species in the site's Herd & Wolves demo.
  - **10.0–10.2 committed** (`9320373`, `22b67f8`, `ab3d9e1`). After 10.0 the user revised D5 (Vite
    serves ONNX Runtime's files) and D6 (no cap on ORT's agents; the timeout guard still covers it).
  - **10.3 built, in the site, and recorded here** (the commit after `ab3d9e1`): the device switch on
    the first ORT pick (0.78–1.3 s); ORT as the third brain, its `.onnx` checked against the
    `.mlp.json`, prepared and warmed up; every switch compiled ahead (0.0–1.1 ms); a refused model
    and ORT not starting handled. ONNX Runtime 1.29 takes one call at a time, so the demo queues
    its calls.
- **Next action:** the user's word on `plan.md` §5, item 14: should `OrtRuntime` queue ONNX
  Runtime's calls (recommended), or each app? Then 10.4: the timings on the overlay.

## State of the repositories

- **null-graph** (`master`, not pushed).
- **The site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`, no commits). The design's
  10.1–10.3 results list every change there and what it replaced:
  - its null-graph is this repository's build at `a8d0107`, updated in the site alone (yalc
    publish, then update; `yalc:push` reaches five projects);
  - `node_modules/null-graph` is a copy (`gotchas.md`);
  - the user's dev server runs on 8081; it was down on 2026-10-15. No test server is left running.
- **ONNX Runtime 1.30.0 is out;** the repository and step 10 use 1.29.0 (D4).
- **`package-lock.json` is ignored by git here.** `dist/index.js` and its map are tracked although
  `dist` is ignored: commit them when core changes.
- **`venv/`** (ignored) needs `--cert venv/pip-ca-bundle.pem` for `pip` ([`gotchas.md`](./gotchas.md)).
- **Another worktree exists:** `D:/orion-realms/null-graph-56-validation` (detached at `9ea8a33`).
  Remove its `node_modules` junction first with `cmd /c rmdir`, then `git worktree remove`.
- **Green:**
  - on 2026-10-15: the headless suites (agents 83, inference 23, core, agent-kit 4, profiler);
    `check-site.mjs`, 16/16; the demo's drive, clean on the dev server and on a build;
  - on 2026-10-14: the core GPU suite, 8/8 on Node/NVIDIA, Node/AMD and Chrome/AMD.
- **Bundle:** core 6,191 B; agents 20,209 B (C2 ceiling 21,495 B); the inference entry: reader
  2,846 B (line 2,900 B), adapter with `OrtBrain` 2,988 B (line 3,000 B).
