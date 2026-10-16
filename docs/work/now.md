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
  - The tool (D2.1 onward) is not built.

## Next action

**D2.1**, in progress:
- the core: clock, render records, sinks, segments, camera keyframes;
- the copies, with `film:check`;
- the Herd & Wolves scene in record mode. The scenario must hand out its catch ring for the flashes.

Decided: `thinkEvery` 3; the showcase at 64K.

ffmpeg 9.0.2 is installed; `libx264` only (see gotchas for NVENC and colour tags). `npm run drive`
checks the look and bakes the font atlas; it does not record.

## State of the repositories

- **null-graph** (`master`, level with `origin/master` at `ba95184`, pushed 2026-10-15; npm 1.1.1
  published the same day).
  - **The films session** committed D2.0 (the spike in `test/spikes/films/`, whose `site.mjs` is a
    spike-only resolve hook into the site; the design; `spike:films`; `.gitignore`). D2.1 work is
    uncommitted while it is built.
  - **Found uncommitted and left alone:**
    - `dist/` is a production build (`dist/index.js` changed, `dist/index.js.map` deleted, from the
      last `yalc:push`; run `npm run build:lib` before committing);
    - `test/spikes/animated-models/check-site.mjs` has an uncommitted dashboard-store check (not
      this session's).
- **Site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`): untouched this session. Eight
  unrelated staged files (hosting, two light demos) stay staged; commit there by path.
- **Another worktree:** `D:/orion-realms/null-graph-56-validation` (detached at `9ea8a33`).
- `package-lock.json` is ignored here; `venv/` needs `--cert venv/pip-ca-bundle.pem`.
- Open, not asked: the green clear colour reads as sky from a low camera; catch rings dominate far
  views; the animals' palette is a stand-in.
