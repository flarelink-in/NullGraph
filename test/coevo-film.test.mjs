// Step 15 (docs/agents-coevo-films-design.md): the co-evolution film scene's settings and tile plan,
// headless. The GPU checks (record, replay, resume) are A55, run by hand (design §4).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { coevoProfile } from './fixtures/coevo-profile.mjs';
import { runCoevolution } from '../tools/agents/coevolve.mjs';
import { loadClones } from '../tools/agents/coevolve-eval.mjs';
import { readSettings, planTiles } from '../tools/film/scenes/coevo-arena.mjs';
import { labelFilters, tileRects } from '../tools/agents/coevolve-film-labels.mjs';

// A run writes run.json and gen-0000 before it asks for a GPU; a failing adapter ends it there.
async function startOnly(outDir, overrides) {
    const previous = Object.getOwnPropertyDescriptor(globalThis.navigator, 'gpu');
    Object.defineProperty(globalThis.navigator, 'gpu', { configurable: true,
        value: { async requestAdapter() { throw new Error('injected adapter failure'); } } });
    try {
        await assert.rejects(runCoevolution(coevoProfile({ outDir, environment: 'contact-v3', spawnSeparation: 2, L: 36, ...overrides }),
            { log() {} }), /injected adapter failure/);
    } finally {
        if (previous) Object.defineProperty(globalThis.navigator, 'gpu', previous);
        else delete globalThis.navigator.gpu;
    }
}

test('U56 coevo-arena settings: one driver per tile, the rule species-wide, refusals name the setting', () => {
    const base = { run: 'some/run', prey: [0, 10, 20, 30], wolf: ['rule', 'rule', 'rule', 'rule'] };
    assert.deepEqual(readSettings(base), { ...base, seed: 77001, cols: 2, look: 'anime' });
    for (const [change, message] of [
        [{ extra: 1 }, /no setting "extra"/],
        [{ run: '' }, /needs run/],
        [{ prey: [] }, /prey must list one driver per tile/],
        [{ prey: [0, -1, 2, 3] }, /prey entries are generations, "clone" or "rule"/],
        [{ prey: [0, 1.5, 2, 3] }, /prey entries/],
        [{ wolf: ['rule', 0, 'rule', 'rule'] }, /wolf runs the rule in every tile or in none/],
        [{ prey: [0, 1, 2], wolf: [0, 1, 2] }, /4, 8, 12 or 16 tiles/],
        [{ wolf: [0, 1, 2] }, /same number of tiles/],
        [{ prey: Array(20).fill(0), wolf: Array(20).fill(0) }, /4, 8, 12 or 16 tiles/],
        [{ seed: 2 ** 32 }, /seed must be a u32/],
        [{ cols: 5 }, /cols must be from 1 to the tile count/],
        [{ look: 'neon' }, /look must be "anime" or "faithful"/],
    ]) assert.throws(() => readSettings({ ...base, ...change }), message, JSON.stringify(change));
});

test('U56 coevo-arena tiles: the run\'s experiment, a row per distinct entry, models checked, shapes refused', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'coevo-film-'));
    try {
        const run = join(dir, 'clones'), small = join(dir, 'small');
        await startOnly(run, { start: 'clones', hidden: [64, 64, 64] });
        await startOnly(small, {});
        const plan = planTiles(readSettings({ run, prey: [0, 'clone', 0, 'clone'], wolf: ['rule', 'rule', 'rule', 'rule'], cols: 4 }));
        assert.deepEqual(plan.tiles, ['gen-0000/rule', 'clone/rule', 'gen-0000/rule', 'clone/rule']);
        assert.deepEqual(plan.geometry, { arenas: 4, wolves: 16, prey: 192, cols: 4, L: 36, gap: 10 });
        assert.equal(plan.lane.environment, 'contact-v3'); assert.equal(plan.lane.spawnSeparation, 2);
        assert.deepEqual(plan.rowOf, { prey: [0, 1, 0, 1], wolf: [0, 0, 0, 0] });
        assert.equal(plan.drivers.wolf, 'rule'); assert.equal(plan.drivers.prey.length, 2);
        // A clone start's generation 0 is the clones, bit for bit, and each row carries its model hash.
        const clones = loadClones();
        plan.drivers.prey.forEach(network => network.forEach((layer, j) => assert.deepEqual(layer.weights, clones.prey.network[j].weights)));
        assert.equal(plan.networks['prey:clone'], clones.prey.hash);
        assert.deepEqual(Object.keys(plan.networks).sort(), ['prey:clone', 'prey:gen-0000']);
        assert.throws(() => planTiles(readSettings({ run, prey: [0, 0, 0, 7], wolf: [0, 0, 0, 0] })), /no checkpoint gen-0007/);
        assert.throws(() => planTiles(readSettings({ run: join(dir, 'missing'), prey: [0, 0, 0, 0], wolf: [0, 0, 0, 0] })), /no run.json/);
        // One table per species: a 16 × 16 run's checkpoints and the 64 × 64 × 64 clones cannot share it.
        assert.throws(() => planTiles(readSettings({ run: small, prey: [0, 'clone', 0, 0], wolf: [0, 0, 0, 0] })),
            /prey networks must share one shape: gen-0000 is 24→16 tanh/);
    } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('U56 coevo-arena labels: tiles found through the scene camera, values escaped for both filter levels', () => {
    const grid = { arenas: 4, cols: 2, wolves: 16, prey: 192, L: 36, gap: 10 };
    // Top-down at 1080p: tile 0 top left, then reading order, each tile about 431 px a side.
    assert.deepEqual(tileRects(grid, 1920, 1080), [[469, 49, 900, 480], [1020, 49, 1451, 480], [469, 600, 900, 1031], [1020, 600, 1451, 1031]]);
    const filters = labelFilters({ settings: { prey: [0, 50, 'clone', 200], wolf: ['rule', 'rule', 'rule', 'rule'] }, grid,
        width: 1920, height: 1080, title: "x150's run, prey", footer: 'contact-v3; seed 7', font: 'C:/Windows/Fonts/segoeui.ttf' });
    assert.equal(filters.length, 10);
    // First-column labels end left of their tile; last-column labels start right of it.
    assert.match(filters[0], /text=Prey\\\\: generation 0:expansion=none:x=447-text_w:y=71:/);
    assert.match(filters[3], /text=Wolves\\\\: the rule:expansion=none:x=1473:/);
    assert.match(filters[4], /text=Prey\\\\: the clones/);
    // The drive's colon, the apostrophe, the comma and the semicolon are escaped for the options, then the graph.
    assert.ok(filters.every(f => f.startsWith('drawtext=fontfile=C\\\\:/Windows/Fonts/segoeui.ttf:text=')));
    assert.match(filters[8], /text=x150\\\\\\'s run\\, prey:/);
    assert.match(filters[9], /text=contact-v3\\; seed 7:.*y=h-22-text_h:fontsize=25:/);
});
