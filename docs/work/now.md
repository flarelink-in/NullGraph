# Now — 2026-10-15

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, C2 complete.** Steps 6–10 committed; step 5's scaling study remains deferred.
- **Cycle D, step D1: animated agents** ([design](../agents-animated-models-design.md)). The user
  calls its phases "10.1–10.5".
  - **Phases 0–4 committed.** Costs `edecdf4`; rigid bake `f538ffe`; vertex hook `75824a7`; Models
    mode (site `77c8e92`, checks `fdf3ede`); Cline's D1.4 field, grass, toon and anime (site
    `850ab4f`, records `e96af9d`).
  - **The phase 4 review is committed:** `ChainPresets.anime()` as soft colours with crisp depth ink
    (`8b68286`); `InstanceDrawOptions.lod`, LOD by size on screen (`c56de95`); far cubes, fences and
    stock materials in the site (`ca81605`). See "Phase 4 review" in the design.
- **Next:**
  1. the bug the user reported: in Models mode the agent count does not switch (Darts and Arrows
     switch);
  2. phase 5, skinned models (the user's "10.5"), designed before it is built.
- Open, not asked: the green clear colour reads as sky when the camera is low; catch rings dominate
  far views; the animals' palette is a stand-in.

## State of the repositories

- **null-graph** (`master`, not pushed): clean after this session's records commit. `dist/` is a
  development build.
- **Site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`): null-graph published to it
  alone through yalc (signature `d360757d…`, a production build of the code now committed). Eight
  unrelated files someone else staged (hosting files, two light demos) stay staged and uncommitted;
  commit by path there. Most of the agents demos' other files are still untracked.
- **Dev servers:** only the user's on 8082. Use another port for tests, and stop what you start.
- **Green on 2026-10-15:** visibility, post and loaders headless; visibility GPU 9/9 on Node/NVIDIA,
  Node/AMD, Chrome/AMD; each code commit validated alone in a temporary worktree; site `tsc`,
  ESLint (0 errors), `check-site.mjs` 4/4; Chrome drives of Herd & Wolves GPU, its preview and
  VisibilityMeshlets.
- **Bundle:** visibility 14,035 B min+gzip; post 18,609 B; loaders 6,463 B.
- **Another worktree exists:** `D:/orion-realms/null-graph-56-validation` (detached at `9ea8a33`).
- `package-lock.json` is ignored by git here; `venv/` needs `--cert venv/pip-ca-bundle.pem`.
