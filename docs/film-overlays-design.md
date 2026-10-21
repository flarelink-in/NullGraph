# Film overlays — burned-in text for any film (`film-overlays-design.md`)

**Status: proposed, 2026-10-21. Nothing built.** Questions Q1–Q5 (§6) wait on the user.

**Parent:** [`offline-films-design.md`](./offline-films-design.md) §8, row "Titles, credits,
subtitles: optional, with ffmpeg after encoding". This file designs that row for every scene.

**Why:** step 15's [`tools/agents/coevolve-film-labels.mjs`](../tools/agents/coevolve-film-labels.mjs)
burns tile labels into co-evolution films with ffmpeg `drawtext`. The idea works, but the script is
specific to one scene: it reads that scene's report, projects its tile rectangles through its default
camera, and refuses films shot along a camera path. The user, 2026-10-21, asked for the general part
to live under films. The film tool draws no text under Node (parent §8, fact 5), so burning after
encoding is still the only route, and everything below stays outside the engine.

---

## 1. What is general and what is not

| Piece | Today | General? |
|---|---|---|
| ffmpeg `drawtext` filter text, with the two-level escaping (`value`) | in `coevolve-film-labels.mjs` | Yes: any scene |
| Re-encode with the film runner's BT.709 tags and encode settings (`BT709`, the manifest's `encode`) | in the same script, duplicating `sinks.mjs` | Yes |
| Font choice (`C:/Windows/Fonts/segoeui.ttf`) | hard-coded default | Yes, as an option |
| Title and footer lines | a flag and a built-in sentence | Yes |
| Tile rectangles from the scene's top-down camera | `tileRects`, using `arenaLayout` and `topDownView` | **No**: a scene supplies anchors |
| The text of each tile ("Prey: generation 12") | `describe`, from the co-evolution report | **No**: a scene supplies the strings |

So the general tool is a **text pass**: items of text with a screen position and a time range go in,
a labelled video comes out. Where an item goes, and what it says, comes from the film file or from the
scene.

## 2. The overlay list

A film file (`tools/film/films/*.json`) may carry an `overlays` list. Each item:

| Field | Meaning |
|---|---|
| `text` | The string, drawn as written (no expansion, as `expansion=none` today) |
| `x`, `y` | Pixels from the top left, or ffmpeg expressions such as `w-text_w-40` (anchored to an edge). Default: top left, with padding |
| `size` | Font size in pixels; default `height / 30`, as the script uses |
| `from`, `to` | Seconds in the finished film; default: the whole film |
| `fade` | Optional seconds of fade in and out |

`readOverlays` in `overlay.mjs` (pure) checks the list, and `manifest.mjs` calls it as it does
`readCamera`: unknown fields, empty text, sizes at or below zero, `from` not below `to`, and a fade
longer than half the item's range are refused, each naming the item. **An item that ends after the
film is allowed** (decided while building O.1): `--seconds` shortens a film before its file is read,
so a 5-second preview of a film with a late credit must still read; the item just never shows.

## 3. Where the pass runs

A new `tools/film/overlay.mjs`, exporting:
- `overlayFilters({ overlays, width, height, font })`: the `drawtext` strings. Pure, headlessly tested.
- `burnOverlays(dir, options)`: reads `film.json` and `film.mp4`, runs ffmpeg with the filters and
  `BT709`, and writes `film-labelled.mp4` beside it. The same result file name as today, so existing
  paths keep working.

`coevolve-film-labels.mjs` shrinks to what is specific: it computes the tile rectangles and strings,
then calls `overlayFilters`. Its output, flags and refusals stay as they are.

The film runner can also call `burnOverlays` after the last segment when the manifest has
`overlays`, so one command gives the labelled film (see Q2).

## 4. Fitting the film tool's rules

- **No frame is touched by the simulation or the render.** Overlays are applied after encoding, so
  reruns of a film stay exact and the GPU timeout guard has nothing to do.
- **Cost:** one extra x264 encode of a finished video, measured in minutes for the long films. The
  quality loss of encoding twice is why Q3 asks about drawing during the first encode.
- **Library:** nothing in `src/`. The bundle and `size.mjs` are untouched. `HerdWolvesGPU`,
  `check-site.mjs` and `film:check` stay green.
- **No models, no new dependencies:** ffmpeg is already required (`findFfmpeg`, `NG_FFMPEG`).

## 5. Phases

| Phase | Delivers | Exit |
|---|---|---|
| O.0 ✅ | `overlay.mjs` with `overlayFilters`; the escaping and defaults tests, including the one-level escaping faults the label script's tests inject | Tests watched failing, then green (O1–O5; a dropped escaping level fails O1 and O2) |
| O.1 ✅ | Manifest `overlays` field and its refusals | Every refusal named in a test (O5, O6; U1 defaults) |
| O.2 ✅ | `burn.mjs` (`burnArgs`, `burnOverlays`, `burnFilm`, and `node tools/film/burn.mjs <dir>`) and the runner hook (per Q2: `film.mjs` burns after joining segments when the manifest has `overlays`; `--font` picks the font, checked before any frame renders). The encoder's quality arguments moved to `qualityArgs` in `sinks.mjs` so both passes share them | O7–O9 with the real ffmpeg (text only inside its range; a removed time range fails O8). A real 2 s Herd & Wolves film rendered with three overlays; frames at 0.5 s and 1.5 s show the title, the anchored footer and the timed caption as asked, BT.709 tags kept |
| O.3 ✅ | `coevolve-film-labels.mjs` rebased on it: `labelOverlays` (tile items) and `labelFilters` stay in the script, and `labelFilm` calls `burnOverlays`. The shared code accepts an unknown film length (an item then runs to the end, as `gte(t,from)`; a fade is refused, O10) | The three U56 tests pass unchanged. The filters for all four `coevo-x150` films, with and without a title (76 strings), are byte-identical to the script's before. Re-labelling `coevo-x150-random-pairs` gives the same ffmpeg arguments, the same decoded frames and the same file hash as the script it replaced. The footer wording was an uncommitted edit already in the working tree (`start positions from film seed …`), kept as it was |

## 6. Questions for the user

| # | Question | Recommendation |
|---|---|---|
| Q1 | Static, positioned text only, or also text that changes over time (a per-second caption, a running generation counter)? | Static items with `from`/`to` first. A changing value is a list of timed items, which the same fields already express. |
| Q2 | Does the film runner burn overlays itself when the manifest has them, or is it always a separate command? | Both: the runner calls it when `overlays` is present, and the command also works on a finished film. |
| Q3 | Draw during the first encode instead of encoding twice? | No. Keep the video unlabelled and exact; labelling stays optional and re-doable without re-rendering. |
| Q4 | Subtitle track (soft, toggleable) as well as burned text? | Not now. Burned text is what the label script needs. |
| Q5 | Should scene-specific anchors (tile rectangles) become a scene export the tool can ask for, or stay inside each scene's own label script? | Stay in the script: only one scene needs them today, and the tool's rule is a second consumer before generality. |
