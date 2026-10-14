# Now — 2026-10-14

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2.** Steps 6–9 are built and committed (9.6, the last phase: `4e19a0f`).
  Step 5's scaling study is still deferred.
- **Step 10, the live switch** ([`../agents-live-switch-design.md`](../agents-live-switch-design.md);
  design `2b8f689`): Rules / Network / ONNX Runtime per species in the site's Herd & Wolves demo.
  - **10.0 committed** (`9320373`). After it, the user revised two decisions: D5, Vite serves ONNX
    Runtime's files; D6, no cap on ORT's agents (the timeout guard still covers its run).
  - **10.1 built, in the site, and recorded here** (this commit): ORT 1.29.0 pinned and left out of
    Vite's pre-bundling; the two `.onnx` files beside the `.mlp.json`. The site builds, and the
    page loads as before.
  - **10.2 is built in the site too;** its records follow in the next commit.
- **Next action:** commit 10.2's records (the user asked for 10.1 and 10.2 as two commits); then
  10.3: the device switch in the site's `EngineCore`, ORT as the third choice, its brains prepared
  and warmed up, failures handled.

## State of the repositories

- **null-graph** (`master`, not pushed). Also committed this session, outside step 10: `a8d0107`
  (`GPUProfiler` drops a read that fails because its device was destroyed; C8 and
  `npm run test:profiler`) and `38c905c` (`.gitattributes`: a fresh checkout passes U26).
- **The site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`, no commits). The design's
  10.1 results list every change there and what it replaced:
  - its null-graph is this repository's build at `a8d0107`, updated in the site alone (yalc
    publish, then update; `yalc:push` reaches five projects);
  - `node_modules/null-graph` is a copy (`gotchas.md`);
  - the user's dev server runs on 8081.
- **ONNX Runtime 1.30.0 is out;** the repository and step 10 use 1.29.0 (D4).
- **`package-lock.json` is ignored by git here.** `dist/index.js` and its map are tracked although
  `dist` is ignored: commit them when core changes.
- **`venv/`** (ignored) needs `--cert venv/pip-ca-bundle.pem` for `pip` ([`gotchas.md`](./gotchas.md)).
- **Another worktree exists:** `D:/orion-realms/null-graph-56-validation` (detached at `9ea8a33`).
  Remove its `node_modules` junction first with `cmd /c rmdir`, then `git worktree remove`.
- **Green on 2026-10-14:**
  - headless suites: agents (reference + 83), inference (23), core, agent-kit (4), profiler (3);
  - the core GPU suite, 8/8 on Node/NVIDIA, Node/AMD and Chrome/AMD;
  - the demo's drive, clean on the dev server and on a build.
- **Bundle:** core 6,191 B; agents 20,209 B (C2 ceiling 21,495 B); the inference entry: reader
  2,846 B (line 2,900 B), adapter with `OrtBrain` 2,988 B (line 3,000 B).
