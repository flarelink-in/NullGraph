// Step 16 (docs/agents-coevo-demo-design.md): co-evolution checkpoints in the test site's
// HerdWolvesCoevo page, and the page's port of the trainer's pens held to the trainer.
//
//   node tools/agents/coevolve-site.mjs export --run <run dir> [--generations 0,50,100,150,200]
//   node tools/agents/coevolve-site.mjs remove --name <run name>
//   node tools/agents/coevolve-site.mjs check [--gpu] [--steps 600] [--inject margin|separation|think]
//
// export copies checkpoints byte for byte into the site's HerdWolvesCoevo/checkpoints/ and rewrites
// its index.json, the page's catalog; the default generations are the multiples of 50 and the last.
// check builds the page's arena.ts and brains.ts with the site's own esbuild and compares them with
// scenario/herd-wolves-arenas.mjs and model.mjs (U57), then the catalog's files with their hashes
// and, where the run is here, its manifests. --gpu also steps a lane of each on one GPU and compares
// every record byte for byte (A56); --inject changes the page's lane, to watch that fail.
// --site <dir> or NG_SITE names the site, as for film:check.
import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { basename, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import * as layout from './scenario/arena-layout.mjs';
import * as arenas from './scenario/herd-wolves-arenas.mjs';
import { DRIVER, LABEL_DRIVING, herdDefinition, herdRule } from './scenario/herd-wolves.mjs';
import { loadCheckpoint, loadClones, loadRun } from './coevolve-eval.mjs';
import { networkFromModel } from './model.mjs';
import { SITE } from '../film/check.mjs';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
export const DEMO = 'src/data/engine-demos/demos/AgentsExperiments/HerdWolvesCoevo';
export const CATALOG_VERSION = 1;
/** What a pen is made of: every run in one catalog shares these (design §3). */
export const EXPERIMENT_KEYS = Object.freeze(['environment', 'P', 'W', 'L', 'gap', 'T', 'dt', 'thinkEvery', 'contacts',
    'contactOptions', 'catchMargin', 'spawnSeparation']);
const ABOUT = 'Co-evolution checkpoints for this page, written by null-graph\'s tools/agents/coevolve-site.mjs export: '
    + 'each file a byte-for-byte copy of a run\'s gen-XXXX/<species>.mlp.json. Exploratory, never evidence for H1–H3.';

const fail = message => { throw new Error(`coevolve-site: ${message}`); };
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const gen = g => `gen-${String(g).padStart(4, '0')}`;
/** JSON with object keys sorted, typed arrays as plain arrays: equal values give equal text. */
export const canonical = value => JSON.stringify(value, (_, v) => {
    if (ArrayBuffer.isView(v)) return Array.from(v);
    if (v && typeof v === 'object' && !Array.isArray(v)) return Object.fromEntries(Object.keys(v).sort().map(k => [k, v[k]]));
    return v;
});
export const experimentOf = config => Object.fromEntries(EXPERIMENT_KEYS.map(key => [key, config[key]]));
const shapeOf = network => [network[0].input, ...network.map(layer => layer.output)].join('→');

/** The generations a run directory holds a checkpoint for, in order. */
export function checkpointsOf(dir) {
    return readdirSync(dir).filter(name => /^gen-\d{4}$/.test(name) && existsSync(join(dir, name, 'manifest.json')))
        .map(name => Number(name.slice(4))).sort((a, b) => a - b);
}

export function readCatalog(site = SITE) {
    const path = join(site, DEMO, 'checkpoints', 'index.json');
    if (!existsSync(path)) return { version: CATALOG_VERSION, about: ABOUT, experiment: null, runs: [] };
    const catalog = JSON.parse(readFileSync(path, 'utf8'));
    if (catalog.version !== CATALOG_VERSION) fail(`${path} is catalog version ${catalog.version}; this tool writes ${CATALOG_VERSION}`);
    return catalog;
}

function writeCatalog(site, catalog) {
    const runs = [...catalog.runs].sort((a, b) => a.name.localeCompare(b.name));
    writeFileSync(join(site, DEMO, 'checkpoints', 'index.json'),
        `${JSON.stringify({ ...catalog, experiment: runs.length ? catalog.experiment : null, runs }, null, 1)}\n`);
}

/** Copies a run's checkpoints into the site's catalog; a run exported again replaces its entry and folder. */
export function exportRun({ run, generations, site = SITE }) {
    if (!existsSync(join(site, DEMO))) fail(`the site at ${site} has no ${DEMO} (--site or NG_SITE)`);
    const loaded = loadRun(resolve(run)), { config } = loaded.run, name = basename(resolve(run));
    if (!/^[A-Za-z0-9._-]+$/.test(name)) fail(`a run's folder name must be letters, digits, ".", "_" or "-"; got "${name}"`);
    const available = checkpointsOf(loaded.dir);
    if (!available.length) fail(`"${run}" holds no checkpoint`);
    const last = available[available.length - 1];
    const wanted = generations ?? [...new Set([...available.filter(g => g % 50 === 0), last])];
    for (const g of wanted) if (!available.includes(g)) fail(`no checkpoint ${gen(g)} in "${run}"; it has ${available.join(', ')}`);
    const catalog = readCatalog(site), experiment = experimentOf(config);
    const others = catalog.runs.filter(entry => entry.name !== name);
    if (others.length && canonical(catalog.experiment) !== canonical(experiment)) {
        const differ = EXPERIMENT_KEYS.filter(key => canonical(catalog.experiment[key]) !== canonical(experiment[key]));
        fail(`${name}'s pens differ from the catalog's in ${differ.join(', ')}: one catalog holds one pen experiment (remove its runs first)`);
    }
    // Every model checked against its manifest and the demo's names before anything is written.
    const pairs = wanted.map(g => [g, loadCheckpoint(loaded, g)]);
    const dir = join(site, DEMO, 'checkpoints');
    rmSync(join(dir, name), { recursive: true, force: true });
    for (const [g] of pairs) {
        mkdirSync(join(dir, name, gen(g)), { recursive: true });
        for (const species of ['prey', 'wolf'])
            copyFileSync(join(loaded.dir, gen(g), `${species}.mlp.json`), join(dir, name, gen(g), `${species}.mlp.json`));
    }
    const entry = { name, start: config.start, shape: shapeOf(pairs[0][1].prey.network), masterSeed: config.seeds.master,
        domain: config.seeds.domain, configHash: loaded.run.configHash, complete: loaded.status?.state === 'complete',
        lastGeneration: last, checkpoints: pairs.map(([g, pair]) => ({ generation: g, prey: pair.prey.hash, wolf: pair.wolf.hash })) };
    writeCatalog(site, { version: CATALOG_VERSION, about: ABOUT, experiment, runs: [...others, entry] });
    return entry;
}

export function removeRun({ name, site = SITE }) {
    const catalog = readCatalog(site);
    if (!catalog.runs.some(entry => entry.name === name)) fail(`the catalog holds no run "${name}"`);
    rmSync(join(site, DEMO, 'checkpoints', name), { recursive: true, force: true });
    writeCatalog(site, { ...catalog, runs: catalog.runs.filter(entry => entry.name !== name) });
}

/** The catalog against its files, and, for runs whose directory is here, against their manifests. */
export function checkCatalog({ site = SITE, runs = join(ROOT, 'tools/agents/out/coevo') } = {}) {
    const catalog = readCatalog(site), problems = [], notes = [];
    for (const entry of catalog.runs) {
        for (const point of entry.checkpoints) for (const species of ['prey', 'wolf']) {
            const file = join(site, DEMO, 'checkpoints', entry.name, gen(point.generation), `${species}.mlp.json`);
            if (!existsSync(file)) problems.push(`${entry.name}/${gen(point.generation)}/${species}.mlp.json is missing`);
            else if (sha256(readFileSync(file)) !== point[species]) problems.push(`${entry.name}/${gen(point.generation)}/${species}.mlp.json does not hash to the catalog's`);
        }
        const here = join(runs, entry.name);
        if (!existsSync(join(here, 'run.json'))) { notes.push(`${entry.name}: not in ${runs}; its files were checked against the catalog only`); continue; }
        const loaded = loadRun(here);
        if (loaded.run.configHash !== entry.configHash) problems.push(`${entry.name}: the run here has another experiment`);
        if (canonical(experimentOf(loaded.run.config)) !== canonical(catalog.experiment)) problems.push(`${entry.name}: the run's pens are not the catalog's`);
        for (const point of entry.checkpoints) {
            const pair = loadCheckpoint(loaded, point.generation);
            for (const species of ['prey', 'wolf']) if (pair[species].hash !== point[species])
                problems.push(`${entry.name}/${gen(point.generation)}/${species}: the catalog's hash is not the run's`);
        }
    }
    return { catalog, problems, notes };
}

/** The page's arena.ts and brains.ts, and the demo's rule, built with the site's esbuild; null-graph is this repository's dist. */
export async function bundlePort(site = SITE) {
    const esbuild = createRequire(join(site, 'package.json'))('esbuild');
    const scratch = mkdtempSync(join(tmpdir(), 'ng-coevo-site-'));
    const posix = path => path.replaceAll('\\', '/');
    try {
        const entry = join(scratch, 'entry.ts'), out = join(scratch, 'port.mjs');
        writeFileSync(entry, [
            `export * from ${JSON.stringify(posix(join(site, DEMO, 'arena.ts')))};`,
            `export * from ${JSON.stringify(posix(join(site, DEMO, 'brains.ts')))};`,
            `export { herdRule } from ${JSON.stringify(posix(join(site, DEMO, '../HerdWolvesGPU/brain.ts')))};`,
        ].join('\n'));
        const library = { name: 'null-graph', setup(build) {
            build.onResolve({ filter: /^null-graph(\/.*)?$/ }, args => ({ external: true,
                path: pathToFileURL(join(ROOT, 'dist', args.path === 'null-graph' ? 'index.js' : `${args.path.slice(11)}.js`)).href }));
        } };
        await esbuild.build({ entryPoints: [entry], outfile: out, bundle: true, format: 'esm', platform: 'neutral',
            resolveExtensions: ['.ts', '.js', '.json'], plugins: [library], logLevel: 'silent' });
        return await import(pathToFileURL(out).href);
    } finally { rmSync(scratch, { recursive: true, force: true }); }
}

/** The trainer's side of every comparison. */
export const TRAINER = Object.freeze({ ...layout, ...arenas, herdRule });

const CHECK_CONTACTS = Object.freeze({ iterations: 2, maxPush: 0.25, relaxation: 1.75, maxScanPerCell: 32, damping: 1,
    mass: 'area', compliance: 0, cellSize: 1.45 });
/** Pens the page builds (4, 16, 64) and two the trainer has used. */
export const CHECK_LAYOUTS = Object.freeze([
    { arenas: 4, wolves: 16, prey: 192, cols: 2, L: 36, gap: 10, contacts: CHECK_CONTACTS },
    { arenas: 16, wolves: 16, prey: 192, cols: 4, L: 36, gap: 10, contacts: CHECK_CONTACTS },
    { arenas: 64, wolves: 16, prey: 192, cols: 8, L: 36, gap: 10, contacts: CHECK_CONTACTS },
    { arenas: 32, wolves: 16, prey: 192, cols: 8, L: 36, gap: 10, contacts: null },
    { arenas: 8, wolves: 16, prey: 128, cols: 4, L: 24.5, gap: 12, contacts: { ...CHECK_CONTACTS, iterations: 4 } },
]);
const REFUSED = Object.freeze([
    { arenas: 4, wolves: 16, prey: 100, cols: 2, L: 36, gap: 10 }, { arenas: 3, wolves: 16, prey: 192, cols: 1, L: 36, gap: 10 },
    { arenas: 4, wolves: 16, prey: 192, cols: 5, L: 36, gap: 10 }, { arenas: 4, wolves: 16, prey: 192, cols: 2, L: 2, gap: 10 },
    { arenas: 4, wolves: 16, prey: 192, cols: 2, L: 36, gap: 8 }, { arenas: 4, wolves: 16, prey: 192, cols: 2, L: 36, gap: 0 },
    // The contact solver's reach sets the gap: 1.2 + 2 · 8 · 1 = 17.2 m here.
    { arenas: 4, wolves: 16, prey: 192, cols: 2, L: 36, gap: 12, contacts: { iterations: 8, maxPush: 1 } },
    { arenas: 4, wolves: 16, prey: 192, cols: 2, L: 36, gap: 17.5, contacts: { iterations: 8, maxPush: 1 } },
    { arenas: 4, wolves: 16, prey: 192, cols: 2, L: 36, gap: 10, contacts: { iterations: 9, maxPush: 0.25 } },
    { arenas: 4, wolves: 16, prey: 192, cols: 2, L: 36, gap: 10, contacts: { iterations: 2, maxPush: 0 } },
]);
const DEFINITION_FIELDS = ['capacity', 'world', 'cellSize', 'grid', 'species', 'sensors', 'actions', 'obsDim', 'actDim',
    'obsSlots', 'obsNames', 'actNames', 'constants', 'nearestMasks', 'dt', 'maxStepsPerFrame', 'maxFrameDelta', 'maxScanPerCell'];
const thrown = run => { try { run(); return 'no refusal'; } catch (error) { return error.message; } };
/** A result, or its refusal: either side refusing is compared, never thrown. */
const outcome = run => { try { return run(); } catch (error) { return `refused: ${error.message}`; } };

/** Every difference between the page's port and the trainer, as text; empty when they agree. */
export function comparePort(port, trainer = TRAINER) {
    const problems = [];
    const same = (what, mine, theirs) => {
        const a = canonical(mine), b = canonical(theirs);
        if (a === b) return;
        let k = 0;
        while (k < Math.min(a.length, b.length) && a[k] === b[k]) k++;
        problems.push(`${what}: first difference at ${k}: page ${JSON.stringify(a.slice(k, k + 60))}, trainer ${JSON.stringify(b.slice(k, k + 60))}`);
    };
    for (const spec of CHECK_LAYOUTS) {
        const name = `${spec.arenas} pens of ${spec.L} m, ${spec.prey} prey, contacts ${spec.contacts ? spec.contacts.iterations : 'off'}`;
        const lp = port.arenaLayout(spec), lt = trainer.arenaLayout(spec);
        same(`${name}: layout`, lp, lt);
        const dp = port.arenaDefinition(lp), dt = trainer.arenaDefinition(lt);
        same(`${name}: definition`, Object.fromEntries(DEFINITION_FIELDS.map(f => [f, dp[f]])),
            Object.fromEntries(DEFINITION_FIELDS.map(f => [f, dt[f]])));
        same(`${name}: the model names`, port.modelNames(dp), trainer.modelNames(dt));
        same(`${name}: respawn`, port.respawnWGSL(lp), trainer.respawnWGSL(lt));
        same(`${name}: containment`, port.containmentWGSL(lp), trainer.containmentWGSL(lt));
        for (const margin of [0, 0.05, 0.125]) same(`${name}: catches, margin ${margin}`, port.exhaustiveCatchWGSL(lp, margin), trainer.exhaustiveCatchWGSL(lt, margin));
        for (const a of [0, lt.arenas - 1]) {
            const lo = [lt.origin[0] + a % lt.cols * lt.pitch, lt.origin[1] + Math.floor(a / lt.cols) * lt.pitch];
            for (const seed of [0, 11, 15001, 77001, 0xffffffff]) for (const separation of [0, 2, 3])
                same(`${name}: spawns, pen ${a}, seed ${seed}, separation ${separation}`, outcome(() => port.tileSpawns(lp, lo, seed, separation)),
                    outcome(() => trainer.tileSpawns(lt, lo, seed, separation)));
        }
    }
    for (const spec of REFUSED) same(`refusal of ${JSON.stringify(spec)}`, thrown(() => port.arenaLayout(spec)), thrown(() => trainer.arenaLayout(spec)));
    same('the rule drivers', port.RULE_DRIVERS, trainer.RULE_DRIVERS);
    same('the environments', port.ENVIRONMENTS, trainer.ENVIRONMENTS);
    for (const seams of [DRIVER, LABEL_DRIVING]) same(`the rule's text, ${canonical(seams)}`, port.herdRule(seams), trainer.herdRule(seams));
    return problems;
}

/** The page's model reader (brains.ts) against the trainer's (model.mjs), on the committed clones. */
export function compareReader(port) {
    const problems = [], demo = herdDefinition();
    for (const species of ['prey', 'wolf']) {
        const path = join(ROOT, 'tools/agents/models/herd-wolves', `${species}.mlp.json`), file = JSON.parse(readFileSync(path, 'utf8'));
        const mine = port.networkFromFile(file, demo), theirs = networkFromModel(file, demo, path);
        if (canonical(mine) !== canonical(theirs)) problems.push(`${species}: the page reads the clone's layers differently`);
        const renamed = { ...file, obsNames: file.obsNames.map((n, k) => (k === 22 ? 'bounds.0' : n)) };
        if (!/obsNames\[22\]/.test(thrown(() => port.networkFromFile(renamed, demo)))) problems.push(`${species}: the page reads a file trained on other names`);
        if (!/version/.test(thrown(() => port.networkFromFile({ ...file, version: 2 }, demo)))) problems.push(`${species}: the page reads another format version`);
    }
    return problems;
}

/**
 * A56: the trainer's lane and the page's, built on one device from the same seeds, brains and rows,
 * stepped `steps` times one submission a step: every record word, held action, state word, pen
 * counter and stat equal. The page drives a pen-wide network with a plain MlpBrain, the trainer
 * with a one-row byGenome table (design §2).
 */
export async function gpuCheck(port, { steps = 600, inject = null, log = console.log } = {}) {
    const { installNodeGPU } = await import('../../test/gpu/node-gpu.mjs');
    const { NullGraph } = await import('../../dist/index.js');
    const { MlpBrain } = await import('../../dist/agents.js');
    const { initNetwork } = await import('./mlp-train.mjs');
    const { watchDevice } = await import('./coevolve-runtime.mjs');
    installNodeGPU();
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    const device = await adapter.requestDevice(), engine = new NullGraph(), watch = watchDevice(device);
    const info = adapter.info ?? {};
    log(`A56 on ${[info.vendor, info.architecture, info.description].filter(Boolean).join(' / ') || 'this GPU'}, ${steps} steps a case${inject ? `, page lane changed: ${inject}` : ''}`);
    if (!(await engine.init(null, { device, width: 8, height: 8 })).success) fail('the engine did not start');
    const clones = loadClones();
    const small = [1, 2].map(seed => initNetwork([{ input: 24, output: 16, activation: 'tanh' }, { input: 16, output: 16, activation: 'tanh' },
        { input: 16, output: 2, activation: 'tanh' }], seed));
    // Rows that differ: the clone, nudged a little more each row.
    const ladder = [0, 1, 2, 3].map(r => clones.prey.network.map(l => ({ ...l, weights: l.weights.map((w, k) => w + r * 0.02 * Math.sin(k)), biases: l.biases })));
    const owned = [], own = brain => { owned.push(brain); return brain; };
    const experiment = { environment: 'contact-v3', contacts: true, contactOptions: CHECK_CONTACTS, catchMargin: 0.05, spawnSeparation: 2, thinkEvery: 3 };
    const four = { arenas: 4, wolves: 16, prey: 192, cols: 2, L: 36, gap: 10 };
    const cases = [
        { name: 'the rule against the rule, 4 pens', geometry: four, rowOf: () => 0,
            brains: () => [{ prey: 'rule', wolf: 'rule' }, { prey: 'rule', wolf: 'rule' }] },
        { name: 'the clone prey (page: one network) against rule wolves, 4 pens', geometry: four, rowOf: () => 0,
            brains: () => [{ prey: own(MlpBrain.byGenome(engine, [clones.prey.network])), wolf: 'rule' },
                { prey: own(new MlpBrain(engine, clones.prey.network)), wolf: 'rule' }] },
        { name: 'label prey against the clone wolves (page: one network), 4 pens', geometry: four, rowOf: () => 0,
            brains: () => [{ prey: 'label', wolf: own(MlpBrain.byGenome(engine, [clones.wolf.network])) },
                { prey: 'label', wolf: own(new MlpBrain(engine, clones.wolf.network)) }] },
        { name: 'a ladder: four prey rows, two wolf rows, 16 pens', geometry: { ...four, arenas: 16, cols: 4 },
            rowOf: (a, species) => (species === 'prey' ? a % 4 : a % 2),
            brains: () => { const prey = own(MlpBrain.byGenome(engine, ladder)), wolf = own(MlpBrain.byGenome(engine, small)); return [{ prey, wolf }, { prey, wolf }]; } },
    ];
    const changed = { margin: { catchMargin: 0.051 }, separation: { spawnSeparation: 2.5 }, think: { thinkEvery: 2 } }[inject] ?? {};
    if (inject && !Object.keys(changed).length) fail(`--inject takes margin, separation or think; got ${inject}`);
    const results = [];
    try {
        for (const c of cases) {
            const [trainerBrains, pageBrains] = c.brains();
            const common = { ...experiment, seedOf: a => 77001 + a, rowOf: c.rowOf };
            const lanes = [arenas.buildLane(engine, c.geometry, { ...common, brains: trainerBrains }),
                port.buildLane(engine, c.geometry, { ...common, ...changed, brains: pageBrains })];
            try {
                for (let s = 0; s < steps; s++) for (const lane of lanes) engine.render(lane.chain);
                const words = buffer => buffer.size / 4;
                const read = lane => arenas.readBuffers(device, [lane.sys.buffer, lane.sys.act, lane.sys.state, lane.counters, lane.stats]
                    .map(buffer => ({ buffer, words: words(buffer) })));
                const [mine, theirs] = [await read(lanes[1]), await read(lanes[0])];
                await watch.check();
                const names = ['records', 'held actions', 'state', 'pen catches', 'stats'];
                const differences = names.flatMap((name, b) => {
                    const at = theirs[b].findIndex((w, k) => w !== mine[b][k]);
                    return at < 0 && mine[b].length === theirs[b].length ? [] : [`${name} differ first at word ${at} (page ${mine[b][at]}, trainer ${theirs[b][at]})`];
                });
                const catches = theirs[4][0];
                results.push({ name: c.name, ok: !differences.length, catches, differences });
                log(`  ${differences.length ? 'FAIL' : 'ok  '}  ${c.name}: ${catches} catches${differences.length ? `; ${differences.join('; ')}` : ', every word equal'}`);
            } finally { lanes.forEach(lane => lane.destroy()); }
        }
    } finally {
        owned.forEach(brain => brain.destroy());
        await watch.check({ final: true }).catch(error => results.push({ name: 'WebGPU', ok: false, differences: [error.message] }));
        watch.destroy();
        device.destroy();
    }
    return results;
}

function parseArgs(argv) {
    const [command, ...rest] = argv, args = { command };
    for (let k = 0; k < rest.length; k++) {
        const key = rest[k];
        if (key === '--gpu') { args.gpu = true; continue; }
        if (!['--run', '--generations', '--site', '--name', '--steps', '--inject'].includes(key)) fail(`unknown option ${key}`);
        if (rest[k + 1] === undefined) fail(`${key} needs a value`);
        args[key.slice(2)] = rest[++k];
    }
    return args;
}

async function main() {
    const args = parseArgs(process.argv.slice(2)), site = resolve(args.site ?? SITE);
    if (args.command === 'export') {
        if (!args.run) fail('export needs --run <run directory>');
        const generations = args.generations?.split(',').map(g => {
            if (!/^\d+$/.test(g.trim())) fail(`--generations takes whole numbers; got "${g}"`);
            return Number(g);
        });
        const entry = exportRun({ run: args.run, generations, site });
        console.log(`coevolve-site: ${entry.name} → ${join(site, DEMO, 'checkpoints', entry.name)}: generations ${entry.checkpoints.map(c => c.generation).join(', ')} (${entry.shape})`);
    } else if (args.command === 'remove') {
        if (!args.name) fail('remove needs --name <run name>');
        removeRun({ name: args.name, site });
        console.log(`coevolve-site: removed ${args.name}`);
    } else if (args.command === 'check') {
        if (!existsSync(join(site, DEMO, 'arena.ts'))) fail(`no ${DEMO}/arena.ts in ${site} (--site or NG_SITE)`);
        console.log(`The site: ${site}`);
        const port = await bundlePort(site);
        const problems = [...comparePort(port), ...compareReader(port)];
        console.log(`  ${problems.length ? 'FAIL' : 'ok  '}  U57: the page's pens and model reader against the trainer's (${CHECK_LAYOUTS.length} layouts)`);
        problems.forEach(p => console.log(`        ${p}`));
        const catalog = checkCatalog({ site });
        console.log(`  ${catalog.problems.length ? 'FAIL' : 'ok  '}  the catalog: ${catalog.catalog.runs.length} runs, ${catalog.catalog.runs.reduce((n, r) => n + r.checkpoints.length, 0)} checkpoints`);
        [...catalog.problems, ...catalog.notes].forEach(p => console.log(`        ${p}`));
        let failed = problems.length + catalog.problems.length;
        if (args.gpu) failed += (await gpuCheck(port, { steps: Number(args.steps ?? 600), inject: args.inject ?? null })).filter(r => !r.ok).length;
        process.exitCode = failed ? 1 : 0;
    } else fail(`the command is export, remove or check; got ${args.command}`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
    main().catch(error => { console.error(error.message); process.exitCode = 1; });
}
