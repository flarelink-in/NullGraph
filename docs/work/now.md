# Now — 2026-10-15

## Position

- **Cycle C, C2 complete.** Steps 6–10 committed; step 5's scaling study remains deferred.
- **Cycle D, step D1: animated agents.** Design and phase 0 committed as `edecdf4`.
  [Design/results](../agents-animated-models-design.md): cow prey, fox wolves, accumulated
  phase per agent; no machine-specific model-count cap.
- **Phase 1 built and validated, uncommitted.** Opus's rigid bake and parser node metadata;
  Cline completed Uint32 index typing, size/tree-shaking tests, checked loader guide,
  external-corpus comparison and regressions. No site changes or commits this session.
- **Next: phase 2**, public visibility `vertexHook`, no-hook WGSL identity, GPU pose parity
  and guide. Models mode/field remain phases 3–4. Field scope still open: default Models only
  (design §11, item 5). Skinned models wait for the user's model (phase 5).

## Validation and limitations

- Development build + TypeScript and IDE build pass. All existing headless scripts pass.
- Loaders **8/8** with corpus options: **193 site GLBs** preserve prior output apart from node
  metadata. Fox/deer/cow poses within 1.6e-7 of glTF at samples; no model files added.
- Node/Dawn GPU: core **8/8**, visibility **6/6**, agents **137/137**, zero GPU errors.
  No browser drives or new posed-model GPU parity test in phase 1.
- Min+gzip: parser 3,207 B; bake 3,261 B; full loaders 6,463 B (+2,267 B versus saved old entry).
  Regression ceilings 3,300 / 3,400 / 6,700 B. Parser-only apps tree-shake the bake away.
- Existing limitations unchanged: uneven-scale normals, no material/vertex colours; at 30 fps,
  blended run poses deviate up to 36.7 mm between samples. See the guide/design for constraints.

## Repository state

- **null-graph:** `master`; phase 1 uncommitted. `dist/` rebuilt in development mode;
  tracked core dist files unchanged. `package-lock.json` ignored.
- **Site:** `D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`, untouched this session.
  Installed library remains the step 10 build; no yalc push or deployment. Restart Vite after
  refresh (node_modules is a copy). The user deploys the site.
- Optional compatibility test needs `NG_GLB_COMPAT_DIR` and `NG_GLB_OLD_LOADER`; see design.
  Old build: local Claude temp session `20c36f59-ab5e-4974-8dfb-d4d710b7ad75`,
  `scratchpad/old/loaders.js`. Animal models remain local and ignored; never commit them.
