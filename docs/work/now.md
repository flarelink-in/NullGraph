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
- **Phase 5.1 complete locally, uncommitted:** skinned bake, shared sampling/welding refactor,
  normalized integer weights, named refusals, guide/tests. Resumed Claude's saved local session
  `20c36f59-ab5e-4974-8dfb-d4d710b7ad75`; no Claude quota used.
- **Next:** phase 5.2 palette snippets + GPU visible-list/pose test; 5.3 rigid site migration and
  costs; 5.4 user's Blender animals when available. Palette snippets are not exported yet.
- Previously reported Models agent-count switching bug remains an open verification item; not
  investigated by this phase's headless bake tests.
- Open, not asked: the green clear colour reads as sky when the camera is low; catch rings dominate
  far views; the animals' palette is a stand-in.

## State of the repositories

- **null-graph** (`master`, not pushed): phase 5.1 and Claude's phase 5.0 decision records remain
  uncommitted. `dist/` is a development build. No new assets, commits, push or deployment.
- **Site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`): null-graph published to it
  alone through yalc (signature `d360757d…`, a production build of the code now committed). Eight
  unrelated files someone else staged (hosting files, two light demos) stay staged and uncommitted;
  commit by path there. Site untouched in 5.1; no yalc publish. Most other demo files are untracked.
- **Dev servers:** prior note says user's on 8082; no server started in 5.1.
- **5.1 green:** production/development builds, TypeScript and IDE build; final loaders run
  **19 pass, 0 fail, 0 skipped** (parser compatibility enabled). Dancers/butterfly via `NG_SKINNED_GLB_DIR` to
  site `public/`; direct glTF/Animator fixtures, guide examples and tree-shaking pass. Rigid
  cow/deer/fox byte-exact against `%TEMP%/ng-d15-pre-skinned-loaders.mjs`.
- **Bundle:** loaders entry 8,389 B min+gzip; parser 3,415 B; rigid 3,479 B; skinned 4,617 B.
- **Another worktree exists:** `D:/orion-realms/null-graph-56-validation` (detached at `9ea8a33`).
- `package-lock.json` is ignored by git here; `venv/` needs `--cert venv/pip-ca-bundle.pem`.
