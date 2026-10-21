// Step 15 (docs/agents-coevo-films-design.md §5): burns tile labels into a finished coevo-arena film.
// The text pass is the film tool's (tools/film/overlay.mjs, burn.mjs; docs/film-overlays-design.md); what is
// this scene's is where the labels go and what they say. This reads the film's own report (its tiles and grid)
// and finds each tile on screen through the scene's default top-down camera, so it refuses films shot along a
// camera path.
//
//   node tools/agents/coevolve-film-labels.mjs <film output directory> [--title <text>] [--font <ttf>]
//
// Writes film-labelled.mp4 beside film.mp4, colour-tagged as the film runner tags its films.
import { existsSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { OrbitalCameraState } from '../../dist/cameras.js';
import { Camera } from '../../dist/index.js';
import { arenaLayout } from './scenario/arena-layout.mjs';
import { topDownView } from '../film/scenes/coevo-arena.mjs';
import { burnOverlays, DEFAULT_FONT } from '../film/burn.mjs';
import { overlayFilters } from '../film/overlay.mjs';

const fail = message => { throw new Error(`coevolve: film labels: ${message}`); };

/** Each tile's screen rectangle [left, top, right, bottom] in pixels, under the scene's default camera. */
export function tileRects(grid, width, height) {
    const lay = arenaLayout({ ...grid, contacts: null }), view = topDownView(lay, width / height);
    const camera = new Camera(60, width / height, 0.5, 1000);
    const orbit = new OrbitalCameraState({ target: view.target, distance: view.distance, azimuthAngle: view.azimuth,
        polarAngle: view.polar, minDistance: 0.01, maxDistance: 1e9, maxPolarAngle: Math.PI - 1e-3 });
    camera.updateProjection(width / height);
    camera.updateView(orbit.getPosition(), orbit.target);
    const m = camera.viewProjectionMatrix;
    const pixel = (x, z) => {
        const w = m[3] * x + m[11] * z + m[15];
        return [((m[0] * x + m[8] * z + m[12]) / w * 0.5 + 0.5) * width, (0.5 - (m[1] * x + m[9] * z + m[13]) / w * 0.5) * height];
    };
    return Array.from({ length: lay.arenas }, (_, a) => {
        const x = lay.origin[0] + a % lay.cols * lay.pitch, z = lay.origin[1] + Math.floor(a / lay.cols) * lay.pitch;
        const [left, top] = pixel(x, z), [right, bottom] = pixel(x + lay.L, z + lay.L);
        return [left, top, right, bottom].map(Math.round);
    });
}

const describe = entry => (entry === 'rule' ? 'the rule' : entry === 'clone' ? 'the clones' : `generation ${entry}`);

/** The overlay items: two lines a tile, beside first- and last-column tiles, inside the others. */
export function labelOverlays({ settings, grid, width, height, title, footer }) {
    const rects = tileRects(grid, width, height), size = Math.round(height / 30), pad = Math.round(size * 0.6);
    const items = rects.flatMap(([left, top, right], a) => {
        const lines = [`Prey: ${describe(settings.prey[a])}`, `Wolves: ${describe(settings.wolf[a])}`];
        const col = a % grid.cols;
        const x = col === 0 ? `${left - pad}-text_w` : col === grid.cols - 1 ? `${right + pad}` : `${left + pad}`;
        return lines.map((text, k) => ({ text, x, y: top + pad + k * Math.round(size * 1.35), size }));
    });
    if (title) items.push({ text: title, x: pad, y: pad, size });
    if (footer) items.push({ text: footer, x: pad, y: `h-${pad}-text_h`, size: Math.round(size * 0.7) });
    return items;
}

/** The drawtext filters of those items. */
export function labelFilters({ font, ...film }) {
    return overlayFilters({ overlays: labelOverlays(film), width: film.width, height: film.height, font });
}

export function labelFilm(dir, { title, font = DEFAULT_FONT } = {}) {
    for (const file of ['film.json', 'report.json', 'film.mp4']) if (!existsSync(join(dir, file))) fail(`no ${file} in "${dir}"`);
    const manifest = JSON.parse(readFileSync(join(dir, 'film.json'), 'utf8'));
    const report = JSON.parse(readFileSync(join(dir, 'report.json'), 'utf8'));
    if (manifest.scene !== 'coevo-arena') fail(`"${dir}" holds a ${manifest.scene} film, not coevo-arena`);
    if (manifest.camera?.length) fail('the film follows a camera path; labels need the scene\'s default view');
    if (report.failure || report.gpuErrors?.length) fail('the film did not complete cleanly');
    const { width, height } = manifest.film, info = report.scene;
    const footer = `${info.environment}, ${info.layout}; start positions from film seed ${report.settings.seed}; exploratory, not a result`;
    return burnOverlays({ input: join(dir, 'film.mp4'), output: join(dir, 'film-labelled.mp4'), film: manifest.film,
        encode: manifest.encode ?? { codec: 'libx264', crf: 18, preset: 'medium' },
        overlays: labelOverlays({ settings: report.settings, grid: info.grid, width, height, title, footer }), font });
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
    const [dir, ...rest] = process.argv.slice(2);
    const flag = name => { const i = rest.indexOf(`--${name}`); return i < 0 ? undefined : rest[i + 1]; };
    try { console.log(`coevolve: wrote ${labelFilm(dir, { title: flag('title'), ...(flag('font') ? { font: flag('font') } : {}) })}`); }
    catch (error) { console.error(error.message); process.exit(1); }
}
