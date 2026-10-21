// Step 16 (docs/agents-coevo-demo-design.md): U57, the site's checkpoint catalog and the check that
// holds the page's port of the trainer's pens to the trainer, headless. The GPU half is A56, run by
// hand: `node tools/agents/coevolve-site.mjs check --gpu` (design §5).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { coevoProfile } from './fixtures/coevo-profile.mjs';
import { runCoevolution } from '../tools/agents/coevolve.mjs';
import { networkFromModel } from '../tools/agents/model.mjs';
import { SITE } from '../tools/film/check.mjs';
import { DEMO, TRAINER, bundlePort, checkCatalog, compareReader, comparePort, exportRun, readCatalog, removeRun }
    from '../tools/agents/coevolve-site.mjs';

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

test('U57 export: checkpoints copied byte for byte, one pen experiment a catalog, refusals name the cause', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'coevo-site-'));
    try {
        const site = join(dir, 'site'), runs = join(dir, 'runs');
        mkdirSync(join(site, DEMO), { recursive: true });
        const clones = join(runs, 'clones-1'), small = join(runs, 'small-2'), other = join(runs, 'other-3');
        await startOnly(clones, { start: 'clones', hidden: [64, 64, 64] });
        await startOnly(small, {});
        await startOnly(other, { L: 24, spawnSeparation: undefined, environment: 'contact-v2' });
        assert.deepEqual(readCatalog(site), { version: 1, about: readCatalog(site).about, experiment: null, runs: [] });

        const entry = exportRun({ run: clones, site });
        assert.equal(entry.name, 'clones-1'); assert.equal(entry.start, 'clones'); assert.equal(entry.shape, '24→64→64→64→2');
        assert.deepEqual(entry.checkpoints.map(c => c.generation), [0]);
        for (const species of ['prey', 'wolf'])
            assert.deepEqual(readFileSync(join(site, DEMO, 'checkpoints/clones-1/gen-0000', `${species}.mlp.json`)),
                readFileSync(join(clones, 'gen-0000', `${species}.mlp.json`)), `${species} copied byte for byte`);
        exportRun({ run: small, site });
        let catalog = readCatalog(site);
        assert.deepEqual(catalog.runs.map(r => [r.name, r.shape]), [['clones-1', '24→64→64→64→2'], ['small-2', '24→16→16→2']]);
        assert.deepEqual(catalog.experiment, { environment: 'contact-v3', P: 192, W: 16, L: 36, gap: 10, T: 300, dt: 1 / 60,
            thinkEvery: 3, contacts: false, contactOptions: {}, catchMargin: 0.05, spawnSeparation: 2 });
        assert.deepEqual(checkCatalog({ site, runs }).problems, []);

        // Another pen experiment, a generation the run lacks, a model that is not its manifest's.
        assert.throws(() => exportRun({ run: other, site }), /pens differ from the catalog's in environment, L, spawnSeparation/);
        assert.throws(() => exportRun({ run: clones, site, generations: [0, 10] }), /no checkpoint gen-0010 in .*; it has 0/);
        const prey = join(small, 'gen-0000/prey.mlp.json'), text = readFileSync(prey, 'utf8');
        writeFileSync(prey, text.replace('"tanh"', '"relu"'));
        assert.throws(() => exportRun({ run: small, site }), /does not match its manifest/);
        writeFileSync(prey, text);

        // Exported again: one entry, its folder replaced. A copy changed in the site is found.
        exportRun({ run: clones, site });
        assert.deepEqual(readCatalog(site).runs.map(r => r.name), ['clones-1', 'small-2']);
        const copy = join(site, DEMO, 'checkpoints/small-2/gen-0000/wolf.mlp.json');
        writeFileSync(copy, `${readFileSync(copy, 'utf8')} `);
        assert.match(checkCatalog({ site, runs }).problems.join('\n'), /small-2\/gen-0000\/wolf\.mlp\.json does not hash to the catalog's/);

        // Removed: entry and folder; an empty catalog has no experiment, so another may follow.
        removeRun({ name: 'small-2', site });
        assert.equal(existsSync(join(site, DEMO, 'checkpoints/small-2')), false);
        removeRun({ name: 'clones-1', site });
        catalog = readCatalog(site);
        assert.deepEqual([catalog.experiment, catalog.runs], [null, []]);
        assert.equal(exportRun({ run: other, site }).name, 'other-3');
        assert.throws(() => removeRun({ name: 'missing', site }), /holds no run "missing"/);
        assert.throws(() => exportRun({ run: clones, site: join(dir, 'nowhere') }), /has no .*HerdWolvesCoevo/);
    } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('U57 the port check: the trainer agrees with itself, and each injected difference is named', () => {
    assert.deepEqual(comparePort(TRAINER), []);
    const changed = [
        [{ respawnWGSL: lay => TRAINER.respawnWGSL(lay).replace('6.2831853', '6.2831850') }, /respawn: first difference/],
        [{ containmentWGSL: lay => TRAINER.containmentWGSL(lay).replace('a.velHeading.x < 0.0', 'a.velHeading.x <= 0.0') }, /containment/],
        [{ exhaustiveCatchWGSL: (lay, m) => TRAINER.exhaustiveCatchWGSL(lay, m + 1e-6) }, /catches, margin 0:/],
        [{ tileSpawns: (lay, lo, seed, s) => TRAINER.tileSpawns(lay, lo, seed ^ 1, s) }, /spawns, pen 0, seed 0, separation 0/],
        [{ arenaDefinition: (lay, o) => { const d = TRAINER.arenaDefinition(lay, o); return { ...d, maxScanPerCell: 32 }; } }, /definition/],
        [{ arenaLayout: spec => (spec.contacts?.iterations === 8 ? { refused: false } : TRAINER.arenaLayout(spec)) }, /refusal of .*"iterations":8/],
        [{ herdRule: seams => TRAINER.herdRule(seams).replace('2.5 * atan2', '2.4 * atan2') }, /the rule's text/],
        [{ RULE_DRIVERS: { ...TRAINER.RULE_DRIVERS, label: TRAINER.RULE_DRIVERS.rule } }, /the rule drivers/],
    ];
    for (const [change, message] of changed) {
        const problems = comparePort({ ...TRAINER, ...change });
        assert.ok(problems.length, `${message}: not caught`);
        assert.match(problems.join('\n'), message);
    }
});

test('U57 the model reader check: the trainer\'s reader passes, a reader blind to names is caught', () => {
    assert.deepEqual(compareReader({ networkFromFile: (file, def) => networkFromModel(file, def, 'test') }), []);
    const blind = { networkFromFile: file => networkFromModel({ ...file, obsNames: undefined, actNames: undefined },
        { obsNames: [], actNames: [] }, 'test') };
    assert.match(compareReader(blind).join('\n'), /reads a file trained on other names/);
});

// The site's own port, when the site is here (NG_SITE or film:check's default).
test('U57 the site\'s arena.ts and brains.ts against the trainer', { skip: !existsSync(join(SITE, DEMO, 'arena.ts')) && `no site at ${SITE}` }, async () => {
    const port = await bundlePort(SITE);
    assert.deepEqual(comparePort(port), []);
    assert.deepEqual(compareReader(port), []);
});
