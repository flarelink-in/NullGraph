# Now — 2026-10-14

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2.** Steps 6–9 are built and committed (9.6, the last phase: `4e19a0f`).
  Step 5's scaling study is still deferred.
- **Step 10, the live switch** ([`../agents-live-switch-design.md`](../agents-live-switch-design.md);
  design `2b8f689`): Rules / Network / ONNX Runtime per species in the site's Herd & Wolves demo.
  - **Phase 10.0 is built and committed:** the spikes and their results (the design's "Phase 10.0
    results"). The device switch works over one `OffscreenCanvas`, in Vite's dev server and in a
    build, if the old device's queue is waited for first.
  - **After it the user revised D5 and D6** (`plan.md` §5, item 13): Vite serves ONNX Runtime's
    own files, and ORT has no cap on its agents; the timeout guard still covers its run.
- **Next action:** the two fixes the user asked for (a `.gitattributes` rule for the `.mlp.json`
  files; a `catch` in `GPUProfiler`'s read), then phase 10.1, the plumbing.

## State of the repositories

- **null-graph** (`master`, not pushed). Nothing uncommitted after phase 10.0.
- **A fresh checkout fails U26** (CRLF in the `.mlp.json` files; `gotchas.md`); the main tree passes.
- **ONNX Runtime 1.30.0 is out;** the repository and step 10 use 1.29.0 (D4).
- **`package-lock.json` is ignored by git here.** `dist/index.js` and its map are tracked although
  `dist` is ignored: commit them when core changes.
- **`venv/`** (ignored) needs `--cert venv/pip-ca-bundle.pem` for `pip` ([`gotchas.md`](./gotchas.md)).
- **The site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`, no commits): the user
  pushed null-graph there with yalc on 2026-10-14. Its `.yalc` and `node_modules` copies of `dist/`
  match this repository's build; its `package-lock.json` still names the registry's 1.0.5.
- **Another worktree exists:** `D:/orion-realms/null-graph-56-validation` (detached at `9ea8a33`).
  Remove its `node_modules` junction first with `cmd /c rmdir`, then `git worktree remove`.
- **Green on 2026-10-14:** headless agents (reference + 83), inference (23), core, agent-kit (4).
  The spikes, every check: `spike:ort-counts` 10/10 on Node/NVIDIA and Node/AMD, and clean in
  Chrome; the live-switch page clean in dev and build (with the wait). No library source changed
  since 9.5.
- **Bundle:** core 6,191 B; agents 20,209 B (C2 ceiling 21,495 B); the inference entry: reader
  2,846 B (line 2,900 B), adapter with `OrtBrain` 2,988 B (line 3,000 B).
