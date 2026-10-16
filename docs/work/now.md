# Now — 2026-10-16

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C complete** (C2 with step 10). Step 5's scaling study stays deferred.
- **Cycle D, D1 (animated agents):** phases 0–4 and 5.0–5.3 committed (last code `49c5a4a`, site
  `17cd4af`/`f4566cd`/`f16164d`, records `ba95184`). **Phase 5.4 waits for the user's skinned
  animals** (GLBs with idle/walk/run/eat); the intake checker is ready.
- **Cycle D, D2 (offline films), new 2026-10-16:** design revision 2,
  [`../offline-films-design.md`](../offline-films-design.md). The core is general: a clock,
  compute records (any GPU buffer by declared layout and interpolation; agents a preset),
  checkpoints, and render records (frames to ffmpeg); replay writes the tracks back and draws.
  First film: a Herd & Wolves showcase at the demo's counts; 1M is a stress film last. Copies of
  the site's look files go to `tools/film/scenes/herd-wolves/` with `film:check` for drift. No
  engine hook and no library change.
  - **D2.0 measured and committed 2026-10-16** (`npm run spike:films`, `test/spikes/films/`;
    results in the design). GTX 1650, Network brains:
    - 1M steps at 0.62 s;
    - `thinkEvery` 60 saves ~30% at 16K and 3.5% at 1M;
    - the Models draw is 4–14 ms at every count;
    - `libx264` runs at 41–66 fps and is the bottleneck at showcase counts;
    - NVENC needs NVIDIA driver ≥ 610;
    - a 1M track save is 20 MB, zstd ×1.5–1.9.
  - **D2.1 built and committed 2026-10-16** (`tools/film/`; results in the design):
    - `npm run film`, `film:check`, `test:film`, and GPU suite `film` (Node only);
    - the 60 s 64K showcase recorded (`tools/film/out/herd-showcase/film.mp4`);
    - frames differ by depth ties (the visibility cull's `atomicAdd` order), under 1%; the
      simulation is exact.

## Next action

**D2.3 started; first slice implemented, uncommitted.** Persistent Rules/Network step calibration
is isolated by adapter/driver and workload/build, with progressive probes, cached maxima and shared
measured-time stops. Film tests 14/14; library/IDE builds pass. NVIDIA and AMD 128K trials complete,
second processes reuse separate caches, zero reported GPU errors. CLI regression passes (57.0 s).
AMD underprediction remains explicit. A verified five-second 1,048,576-agent NVIDIA Network clip now
exists at `tools/film/out/herd-million-5s/replay/film.mp4` (1080p60, 300 frames, zero GPU errors).
Compute 169.8 s; replay/encode 52.1 s; worst measured step 814.2 ms. Preset `herd-million.json`
uses thinkEvery 60, no warm-up; reproduction commands are in the film README. Guard probes now
exclude stop-checked compilation warm-up and allow a safely predicted half-size refinement.
Film tests 16/16. Full-minute recording is NOT done. Next: resource estimates, draw guarding, ORT calibration,
atlas/names/captions, then the 1M tracks-first stress film. Evidence: `tools/film/out/d23-validation/`.
**D2.2 remains implemented, uncommitted, acceptance pending** (31/31 GPU checks on both adapters;
showcase/replay verified). Review the mixed working tree by path before any requested commit.
ORT above 65,536 agents remains refused; no phase accepted automatically.

ffmpeg 9.0.2 is installed; `libx264` only (see gotchas for NVENC and colour tags). `npm run drive`
checks the look and bakes the font atlas; it does not record.

## State of the repositories

- **null-graph** (`master`, level with `origin/master` at `ba95184`, pushed 2026-10-15; npm 1.1.1
  published the same day).
  - **The films session** committed D2.0 as `f945529` (the spike in `test/spikes/films/`, whose
    `site.mjs` is a spike-only resolve hook into the site; the design; `spike:films`; `.gitignore`).
    D2.1 committed after it.
  - **Found uncommitted and left alone:**
    - `dist/` is a production build (`dist/index.js` changed, `dist/index.js.map` deleted, from the
      last `yalc:push`; run `npm run build:lib` before committing);
    - `test/spikes/animated-models/check-site.mjs` has an uncommitted dashboard-store check (not
      this session's).
- **A second session the same day (docs only, uncommitted):** `docs/use-cases-current.md` and
  `docs/use-cases-after-phase-2.md`, every use case rated by effort, need and money, now and after
  Phase 2. Their proposed Phase 2 order (NAV first, then EVO) awaits the user; `plan.md` untouched.
- **Site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`): untouched this session. Eight
  unrelated staged files (hosting, two light demos) stay staged; commit there by path.
- **Axion** (`D:/orion-realms/orion-engine-test`, the engine null-graph exists for): its Phase 0–4
  work from August is **still uncommitted** (last commit 2026-03-17); Phases 3–4 await their GPU run.
  New, uncommitted: `docs/VERTICAL-SLICES.md`, the smallest games to build on it and their value. It
  imports only null-graph's core, geometry and cameras. **Since 2026-10-16 Axion has its own
  `docs/work/`** (plan; Cycle A, verification, designed in `docs/verification-design.md`): read
  those for Axion work. Its `.gitignore` hides 18 files, the Phase 4 harness among them.
- **Product** (`D:/orion-realms/vikaskumarsingh.com`, new 2026-10-16; not a git repository): the
  user's portfolio, the product Axion and null-graph are to serve ("design backwards"), built on
  Axion.
  - Its flow is Surface → Fall → Void (`docs/flow.md`), and the estimate is 10–14 weeks likely
    (`docs/estimate.md`).
  - Lab mock m1 (`lab/`) draws with null-graph 1.1.1 from npm. It found two candidate changes here:
    `Outline` inks sloped planes (a linear-depth Sobel), and `CelShade` lifts blacks to
    1/(2·bands). See `lab/README.md`.
  - Since 2026-10-17 the site runs **Axion** (through yalc, pure mode) in its hero; Axion's
    `docs/look-design.md` designs the post chain and custom shaders through Axion.
- **Plan change 2026-10-17:** **Cycle E, terrain for games**, proposed at the user's word (plan §3,
  §5 item 15, scope changes). Design first; its order against D2.3 and D1.5.4 is the user's call.
- **Another worktree:** `D:/orion-realms/null-graph-56-validation` (detached at `9ea8a33`).
- `package-lock.json` is ignored here; `venv/` needs `--cert venv/pip-ca-bundle.pem`.
- Open, not asked: the green clear colour reads as sky from a low camera; catch rings dominate far
  views; the animals' palette is a stand-in.
