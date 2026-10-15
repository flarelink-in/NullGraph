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
- **Cycle D, animated agents** (decided 2026-10-15, ahead of Phase 2;
  [`../agents-animated-models-design.md`](../agents-animated-models-design.md)). Step D1: baked
  rigid animation in loaders, a vertex hook in visibility (both public API), a Models draw mode and
  a toon field in Herd & Wolves; the cow for the prey, the fox for the wolves; one accumulated
  phase per agent. **Phase 0 measured, not committed:** `test/spikes/animated-models/`; posed,
  welded cows cost 3.7 / 8.1 / 28.5 ms at 4K / 16K / 64K in Chrome on the AMD iGPU.
- **Next action:** phase 1, `GLBParser`'s node per mesh and `bakeRigidAnimation` (welding by
  default). Open: the field in Models mode only, or in every mode (design §11, item 5; default
  Models only). The user deploys the site for step 10.

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
