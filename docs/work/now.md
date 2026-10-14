# Now — 2026-10-15

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2.** Steps 6–9 are built and committed (9.6, the last phase: `4e19a0f`).
  Step 5's scaling study is still deferred.
- **Step 10, the live switch** ([`../agents-live-switch-design.md`](../agents-live-switch-design.md);
  design `2b8f689`): Rules / Network / ONNX Runtime per species in the site's Herd & Wolves demo.
  - **10.0–10.3 committed** (`9320373`, `22b67f8`, `ab3d9e1`, `072224d`). After 10.0 the user
    revised D5 (Vite serves ONNX Runtime's files) and D6 (no cap; the timeout guard still covers it).
  - **`OrtRuntime` takes ONNX Runtime's calls one at a time** (`plan.md` §5, item 14, decided B;
    this commit): U36; A34 on three host/GPU pairs; the adapter's line 3,100 B. The site has it;
    the demo's own queue is gone.
  - **10.4 is built in the site too;** its records follow in the next commit.
- **Next action:** commit 10.4's records (the user asked for two commits); then 10.5: ORT at every
  count under the timeout guard, the site's documentation page, the guides' pointer.

## State of the repositories

- **null-graph** (`master`, not pushed).
- **The site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`, no commits). The design's
  10.1–10.3 results list every change there and what it replaced:
  - its null-graph is this repository's build with the queue (yalc signature `f89220ed…`), updated
    in the site alone (`yalc:push` reaches five projects); restart a dev server with `--force`;
  - `node_modules/null-graph` is a copy (`gotchas.md`);
  - the user's dev server runs on 8081; it was down on 2026-10-15. No test server is left running.
- **ONNX Runtime 1.30.0 is out;** the repository and step 10 use 1.29.0 (D4).
- **`package-lock.json` is ignored by git here.** `dist/index.js` and its map are tracked although
  `dist` is ignored: commit them when core changes.
- **`venv/`** (ignored) needs `--cert venv/pip-ca-bundle.pem` for `pip` ([`gotchas.md`](./gotchas.md)).
- **Another worktree exists:** `D:/orion-realms/null-graph-56-validation` (detached at `9ea8a33`).
  Remove its `node_modules` junction first with `cmd /c rmdir`, then `git worktree remove`.
- **Green on 2026-10-15:** the headless suites (agents 83, inference 24, core, agent-kit 4,
  profiler); the `inference` GPU suite, 17/17 on Node/NVIDIA, Node/AMD and Chrome/AMD;
  `check-site.mjs`, 16/16. The core GPU suite last ran on 2026-10-14 (8/8 on the same three pairs).
- **Bundle:** core 6,191 B; agents 20,209 B (C2 ceiling 21,495 B); the inference entry: reader
  2,847 B (line 2,900 B), adapter with `OrtBrain` 3,064 B (line 3,100 B).
