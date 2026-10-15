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
- **Phase 5.0 committed** (`86216cf`): palette adopted, one slot per visible model after culling;
  WGSL snippets in loaders, no class yet; move the demo's rigid cow/fox onto it before new animals.
- **Phase 5.1 committed `e84e308`:** skinned bake, shared sampling/welding refactor,
  normalized integer weights, named refusals, guide/tests. Resumed Claude's saved local session
  `20c36f59-ab5e-4974-8dfb-d4d710b7ad75`; no Claude quota used.
- **Phase 5.2 committed `49c5a4a`:** four public palette/pose WGSL snippets; opt-in
  `vertexHook.instanceIndex` passes compacted slot as fifth deform argument; existing hooks unchanged.
- **Phase 5.3 site committed `17cd4af`:** cow/fox per-visible-draw palettes after culling; Darts default/lazy assets
  unchanged; 64 MiB total palette budget/device limits, legacy hook fallback at 64K locally.
- No reliable rigid speedup: 4K NVIDIA hook/palette 1.901/2.032 ms; AMD 3.604/3.867 ms (512²).
- Actual-site GPU checks 6/6 each Node/NVIDIA and Node/AMD, no errors or pixel differences.
  Chrome/AMD: 32 actions, six reviewed screenshots, zero errors; Models 4K/16K/64K, pause, resize,
  anime/grid, mode switching, selected inspector, teardown. Count-switch bug not reproduced.
- **Follow-up site commit `f4566cd`:** contiguous slot/part producer layout fixes strided writes; AMD 16K old palette
  8.95–9.45 ms → 7.77–7.83 ms; NVIDIA near hook parity. Four final GPU runs 12/12 each.
- **5.4 started (intake only):** asset checker added; no skinned cow/fox found in site public GLBs.
  Need user's GLBs and idle/walk/run/eat clips; integration/visual drive remains blocked on assets.
- **Defaults follow-up:** Models + built-in Network for both species; README updated with animated
  crowd capabilities/boundaries; intake checker asserts the shared defaults. TypeScript, targeted
  ESLint, IDE build and site shader checks pass. Fresh isolated Chrome/AMD startup: both networks
  loaded/switched, six palettes, screenshot reviewed, clean teardown and zero errors. Not a full
  React/worker drive. Offline recording and 3D/evolution discussed only; no implementation.
- Open, not asked: the green clear colour reads as sky when the camera is low; catch rings dominate
  far views; the animals' palette is a stand-in.

## State of the repositories

- **null-graph** (`master`, not pushed): phases 5.1 and 5.2 committed in this session.
  `dist/` is a development build. No new assets, push or deployment.
- **Site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`): null-graph published to it
  alone through yalc (signature `d360757d…`, a production build of the code now committed). Eight
  unrelated files someone else staged (hosting files, two light demos) stay staged and uncommitted;
  commit by path there. 5.3 touches only HerdWolvesGPU/index.ts and models.ts. No yalc publish.
  Latest follow-up ran `npm run yalc:push` successfully: production build refreshed all five
  registered local consumers, including this site and its .yalc copy. Most other demo files are untracked.
- **Dev servers:** prior note says user's on 8082; no server started in 5.1.
- **5.3 site validation:** TypeScript, targeted ESLint, IDE build and production Vite build pass
  (51 s; existing chunk/deprecation warnings). React/worker host and ORT migration not retested.
  Artifacts in `%TEMP%/ng-herd-palette-selection/`, `ng-visible-palette-{nvidia,amd}.json`;
  reproducible scripts in `test/spikes/animated-models/`. Temporary public page removed after drive.
- **5.2 green:** production/development builds, TypeScript, IDE build; loaders **20/20**, visibility
  headless; palette GPU **12/12** Node/NVIDIA, Node/AMD, Chrome/AMD, zero errors; visibility GPU
  **9/9** Node/NVIDIA and Chrome/AMD. Dancers/butterfly via `NG_SKINNED_GLB_DIR` to
  site `public/`; direct glTF/Animator fixtures, guide examples and tree-shaking pass. Rigid
  cow/deer/fox byte-exact against `%TEMP%/ng-d15-pre-skinned-loaders.mjs`.
- **Bundle:** loaders entry 9,106 B min+gzip; snippets 786 B; parser 3,414 B; rigid 3,479 B;
  skinned 4,617 B. GPU example: `test/loaders-gpu/run.js`, registered as `--only loaders`.
- **Another worktree exists:** `D:/orion-realms/null-graph-56-validation` (detached at `9ea8a33`).
- `package-lock.json` is ignored by git here; `venv/` needs `--cert venv/pip-ca-bundle.pem`.
