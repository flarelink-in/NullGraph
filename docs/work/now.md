# Now — 2026-10-15

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2: complete.** Steps 6–10 are built and committed. Step 5's scaling study
  (C1) is still deferred.
- **Step 10, the live switch** ([`../agents-live-switch-design.md`](../agents-live-switch-design.md)):
  10.0–10.4 committed (`9320373`, `22b67f8`, `ab3d9e1`, `072224d`, then 10.4's records after
  `f8c4935`). **10.5 built, in the site, and recorded in the commit after `fd9211b`** ("Phase 10.5
  results"):
  - ONNX Runtime at every count, under the timeout guard: its run predicted from the runs measured
    on this GPU, and measured first over 65,536 rows when nothing larger is measured. A refused
    species runs its network on the built-in kernel;
  - the steps' guard keeps its measurements across the device switch (a first ORT pick at 256K
    came back at 4K before);
  - overlay fixes found at 256K; the site's documentation page rewritten; `docs/api/agents.md` and
    `inference.md` point to the demo on nullgraph.dev.
- **Next action:** none scheduled. The user deploys the site, so that nullgraph.dev shows step 10
  where the guides now point. The user is thinking about the demo's next look (animated low-poly
  models, toon shading, a video of the capabilities) and Phase 2; nothing is decided. After C2,
  the user's direction for brains is in `plan.md` (scope changes, 2026-10-15): keep ORT as the
  reference, widen `MlpBrain` for the 3D cases. Not scheduled: step 5's scaling study.

## State of the repositories

- **null-graph** (`master`, not pushed). 10.5's records are committed: the design's results,
  `plan.md` (row 10, the horizon goal), `gotchas.md`, the parent's status and step 10 entry, the
  two guides' pointers.
- **The site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`, no commits). The design's
  10.1–10.5 results list every change there and what it replaced:
  - its null-graph is this repository's build with the queue (yalc signature `f89220ed…`); 10.5
    changed no library code. `node_modules/null-graph` is a copy (`gotchas.md`);
  - its menu and the page's tab title say "Herd & Wolves (GPU brains)" since 10.5 (was "GPU
    arrows");
  - the user's dev server runs on 8081 (down on 2026-10-15). No test server is left running.
- **ONNX Runtime 1.30.0 is out;** the repository and step 10 use 1.29.0 (D4).
- **`package-lock.json` is ignored by git here.** `dist/index.js` and its map are tracked although
  `dist` is ignored: commit them when core changes.
- **`venv/`** (ignored) needs `--cert venv/pip-ca-bundle.pem` for `pip` ([`gotchas.md`](./gotchas.md)).
- **Another worktree exists:** `D:/orion-realms/null-graph-56-validation` (detached at `9ea8a33`).
  Remove its `node_modules` junction first with `cmd /c rmdir`, then `git worktree remove`.
- **Green on 2026-10-15:** after 10.5's docs, agents 83, inference 24 and agent-kit 4 headless;
  `check-site.mjs` 16/16; the demo's drives clean on the dev server and on a build. Earlier the
  same day: core, profiler, and the `inference` GPU suite, 17/17 on three host/GPU pairs.
- **Bundle** (unchanged by 10.5): core 6,191 B; agents 20,209 B (C2 ceiling 21,495 B); the
  inference entry: reader 2,847 B (line 2,900 B), adapter with `OrtBrain` 3,064 B (line 3,100 B).
