// Phase 14.3, design §4, §9, §12. Configuration validation, canonicalization and hashing.
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { arenaLayout } from './scenario/arena-layout.mjs';

const fail = message => { throw new Error(`coevolve: ${message}`); };

export const CONFIG_SCHEMA = 1;
export const ALGORITHM_VERSION = 'null-graph/coevo-es-v1';

export function canonicalJson(obj) {
    if (obj === null || typeof obj !== 'object') return JSON.stringify(obj);
    if (Array.isArray(obj)) return '[' + obj.map(canonicalJson).join(',') + ']';
    const keys = Object.keys(obj).sort();
    return '{' + keys.map(k => JSON.stringify(k) + ':' + canonicalJson(obj[k])).join(',') + '}';
}

/** The settings a run's checkpoints depend on: everything but output location and human labels. */
export function experimentOf(config) {
    const { name, targetDir, outDir, configHash: ignored, ...experiment } = config;
    return experiment;
}

export function configHash(config) {
    return createHash('sha256').update(canonicalJson(experimentOf(config))).digest('hex');
}

const KNOWN_CONFIG_KEYS = new Set([
    'name', 'K', 'E', 'P', 'W', 'L', 'cols', 'gap', 'T', 'dt', 'thinkEvery',
    'environment', 'contacts', 'contactOptions', 'catchMargin', 'spawnSeparation', 'generations',
    'start', 'hidden', 'optimizer', 'archive', 'seeds', 'outDir', 'exportOnnx', 'arenas'
]);
const KNOWN_OPT_KEYS = new Set(['sigma', 'rate', 'decay']);
const KNOWN_ARCH_KEYS = new Set(['every', 'cap', 'share']);
const KNOWN_SEEDS_KEYS = new Set(['master', 'domain']);

export function validateConfig(raw) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) fail('config must be an object');

    for (const key of Object.keys(raw)) {
        if (!KNOWN_CONFIG_KEYS.has(key)) fail(`unknown config property "${key}"`);
    }

    for (const key of ['K', 'E', 'P', 'W', 'L', 'cols', 'gap', 'T', 'dt', 'thinkEvery',
        'environment', 'contacts', 'contactOptions', 'catchMargin', 'generations', 'start',
        'hidden', 'optimizer', 'archive', 'seeds']) {
        if (raw[key] === undefined) fail(`explicit research profile requires ${key}`);
    }
    for (const [key, fields] of [['optimizer', ['sigma', 'rate', 'decay']],
        ['archive', ['every', 'cap', 'share']], ['seeds', ['master', 'domain']]]) {
        if (!raw[key] || typeof raw[key] !== 'object' || Array.isArray(raw[key])) fail(`${key} must be an object`);
        for (const field of fields) if (raw[key][field] == null) fail(`explicit research profile requires ${key}.${field}`);
    }
    if (!raw.contactOptions || typeof raw.contactOptions !== 'object' || Array.isArray(raw.contactOptions))
        fail('contactOptions must be an object');
    const contactKeys = ['iterations', 'maxPush', 'relaxation', 'compliance', 'damping', 'mass', 'maxScanPerCell', 'cellSize'];
    for (const key of Object.keys(raw.contactOptions)) if (!contactKeys.includes(key)) fail(`unknown contactOptions property "${key}"`);
    if (raw.contacts) for (const key of contactKeys) {
        if (raw.contactOptions[key] === undefined) fail(`explicit research profile requires contactOptions.${key}`);
        if (key !== 'mass' && !Number.isFinite(raw.contactOptions[key])) fail(`contactOptions.${key} must be finite`);
    }
    if (!raw.contacts && Object.keys(raw.contactOptions).length) fail('contacts off requires empty contactOptions');
    if (raw.exportOnnx !== undefined && typeof raw.exportOnnx !== 'boolean') fail('exportOnnx must be boolean');

    const {
        name,
        K = 8,
        E = 1,
        P = 192,
        W = 16,
        L = 24,
        cols = 2,
        gap = 10,
        T = 300,
        dt = 1 / 60,
        thinkEvery = 3,
        environment = 'contact-v2',
        contacts = false,
        contactOptions = {},
        catchMargin = 0.05,
        generations = 10,
        start = 'random',
        hidden = [16, 16],
        optimizer = {},
        archive = {},
        seeds = {},
        outDir,
        exportOnnx = false,
    } = raw;

    if (typeof name !== 'string' || !/^[a-zA-Z0-9_-]+$/.test(name))
        fail(`invalid run name "${name}": must be alphanumeric, hyphen or underscore`);

    for (const [key, val] of Object.entries({ K, E, P, W, cols, T, thinkEvery, generations })) {
        if (!Number.isSafeInteger(val) || val < 1) fail(`${key} must be a positive integer`);
    }

    if (!Number.isFinite(dt) || dt <= 0) fail('dt must be finite and > 0');
    if (!Number.isFinite(L) || L <= 2) fail(`L ${L} leaves no room for spawn inset`);
    if (!Number.isFinite(gap) || gap <= 0) fail('gap must be positive');
    if (T > 0xffffffff || thinkEvery > 0xffffffff) fail('tick/cadence overflows u32');

    const arenas = K * E;
    const totalArenas = 2 * arenas; // 2KE total evaluated arena episodes
    if (!Number.isSafeInteger(totalArenas)) fail('arena product overflows safe integer');
    if (!Number.isSafeInteger(P * T) || P * T >= 2 ** 32) fail('per-arena catch bound P*T overflows u32');
    if (!Number.isSafeInteger(arenas * P * T) || arenas * P * T >= 2 ** 32)
        fail('lane catch bound arenas*P*T overflows u32');
    if (generations > 0xffffffff || K > 0xffffffff || E > 0xffffffff) fail('seed tuple fields overflow u32');
    if (raw.arenas !== undefined && raw.arenas !== totalArenas)
        fail(`--arenas ${raw.arenas} does not equal 2KE = ${totalArenas}`);

    if (!['contact-v2', 'contact-v3', 'legacy-v1'].includes(environment))
        fail(`unknown environment "${environment}"`);
    if (typeof contacts !== 'boolean') fail('contacts must be an explicit boolean');
    if (environment === 'legacy-v1' && contacts) fail('legacy-v1 requires contacts off');
    if (environment !== 'legacy-v1' && (!Number.isFinite(catchMargin) || catchMargin < 0))
        fail(`${environment} requires explicit nonnegative finite catchMargin`);
    // contact-v3 = contact-v2 with initial prey spawns at least spawnSeparation m from the tile's wolves.
    if (environment === 'contact-v3') {
        if (raw.spawnSeparation === undefined) fail('explicit research profile requires spawnSeparation for contact-v3');
        if (!Number.isFinite(raw.spawnSeparation) || raw.spawnSeparation <= 0 || raw.spawnSeparation >= L / 2)
            fail(`spawnSeparation ${raw.spawnSeparation} must be in (0, L/2 = ${L / 2}) m`);
    } else if (raw.spawnSeparation !== undefined) fail(`${environment} takes no spawnSeparation (contact-v3 does)`);
    if (!Number.isFinite(catchMargin) || catchMargin < 0) fail('catchMargin must be finite and nonnegative');
    if (contacts && (contactOptions.relaxation <= 0 || contactOptions.relaxation >= 2 ||
        contactOptions.compliance < 0 || contactOptions.damping < 0 || contactOptions.damping > 1 ||
        !['area', 'equal'].includes(contactOptions.mass) || contactOptions.cellSize <= 0 ||
        !Number.isSafeInteger(contactOptions.maxScanPerCell) || contactOptions.maxScanPerCell < 1))
        fail('invalid contactOptions solver settings');
    if (contacts && (!Number.isSafeInteger(contactOptions.iterations) || contactOptions.iterations < 1 ||
        contactOptions.iterations > 8 || contactOptions.maxPush <= 0)) fail('invalid contactOptions iterations/maxPush');

    // Layout verification verifies alignment: P % 64 === 0, arenas * wolves % 64 === 0, safe gap
    const layout = arenaLayout({
        arenas, wolves: W, prey: P, L, gap, cols,
        contacts: contacts ? { iterations: 2, maxPush: 0.25, ...contactOptions } : null
    });

    // Optimizer settings
    if (typeof optimizer !== 'object' || Array.isArray(optimizer)) fail('optimizer must be an object');
    for (const k of Object.keys(optimizer)) {
        if (!KNOWN_OPT_KEYS.has(k)) fail(`unknown optimizer property "${k}"`);
    }
    const sigma = optimizer.sigma ?? 0.1;
    const rate = optimizer.rate ?? 0.01;
    const decay = optimizer.decay ?? 0.005;
    if (!Number.isFinite(sigma) || sigma <= 0) fail('optimizer.sigma must be finite and > 0');
    if (!Number.isFinite(rate) || rate <= 0) fail('optimizer.rate must be finite and > 0');
    if (!Number.isFinite(decay) || decay < 0) fail('optimizer.decay must be finite and >= 0');

    // Archive settings
    if (typeof archive !== 'object' || Array.isArray(archive)) fail('archive must be an object');
    for (const k of Object.keys(archive)) {
        if (!KNOWN_ARCH_KEYS.has(k)) fail(`unknown archive property "${k}"`);
    }
    const H = archive.every ?? 5;
    const M = archive.cap ?? 32;
    const h = archive.share ?? 0.5;
    if (!Number.isSafeInteger(H) || H < 1) fail('archive.every must be integer >= 1');
    if (!Number.isSafeInteger(M) || M < 2) fail('archive.cap must be integer >= 2');
    if (!Number.isFinite(h) || h < 0 || h > 1) fail('archive.share must be in [0, 1]');

    // Seed master
    if (typeof seeds !== 'object' || Array.isArray(seeds)) fail('seeds must be an object');
    for (const k of Object.keys(seeds)) {
        if (!KNOWN_SEEDS_KEYS.has(k)) fail(`unknown seeds property "${k}"`);
    }
    const masterSeed = seeds.master ?? 18001;
    const domain = seeds.domain ?? 1;
    if (!Number.isInteger(masterSeed) || masterSeed < 0 || masterSeed > 0xffffffff)
        fail('seeds.master must be u32');
    if (!Number.isInteger(domain) || domain < 0 || domain > 0xffffffff)
        fail('seeds.domain must be u32');

    // Hidden layer widths
    if (!Array.isArray(hidden) || hidden.length < 1 || hidden.length > 3 || !hidden.every(w => Number.isInteger(w) && w > 0 && w <= 64))
        fail('hidden layers must be array of positive integers <= 64');

    if (!['clones', 'random'].includes(start)) fail(`unknown start mode "${start}"`);

    const targetDir = outDir ? resolve(outDir) : resolve('tools/agents/out/coevo', name);
    if (existsSync(targetDir)) fail(`run directory already exists: "${targetDir}" (no resume)`);

    const validated = {
        schema: CONFIG_SCHEMA,
        algorithm: ALGORITHM_VERSION,
        name,
        K,
        E,
        arenasPerLane: arenas,
        totalArenas,
        P,
        W,
        L,
        cols,
        gap,
        T,
        dt,
        thinkEvery,
        environment,
        contacts,
        contactOptions: contacts ? { iterations: 2, maxPush: 0.25, ...contactOptions } : {},
        catchMargin,
        // Only contact-v3 has the field, so earlier environments keep their experiment hashes.
        ...(environment === 'contact-v3' ? { spawnSeparation: raw.spawnSeparation } : {}),
        generations,
        start,
        hidden: [...hidden],
        optimizer: { sigma, rate, decay },
        archive: { every: H, cap: M, share: h },
        seeds: { master: masterSeed, domain },
        targetDir,
        exportOnnx: Boolean(exportOnnx),
    };

    validated.configHash = configHash(validated);
    return Object.freeze(validated);
}

export function parseCliArgs(argv) {
    const args = argv.slice();
    const values = new Set(['config', 'name', 'start', 'hidden', 'arenas', 'K', 'E', 'P', 'W', 'L',
        'gap', 'cols', 'T', 'dt', 'thinkEvery', 'environment', 'catchMargin', 'spawnSeparation', 'generations',
        'sigma', 'rate', 'decay', 'archive-every', 'archive-cap', 'archive-share', 'master-seed', 'domain', 'out-dir']);
    const flags = new Set(['contacts', 'no-contacts', 'export-onnx']);
    const seen = new Set();
    for (let i = 0; i < args.length; i++) {
        const token = args[i], key = token.startsWith('--') ? token.slice(2) : '';
        if (!values.has(key) && !flags.has(key)) fail(`unknown argument ${token}`);
        if (seen.has(key)) fail(`duplicate argument ${token}`);
        seen.add(key);
        if (values.has(key) && (!args[++i] || args[i].startsWith('--'))) fail(`${token} needs a value`);
    }
    if (seen.has('contacts') && seen.has('no-contacts')) fail('conflicting --contacts and --no-contacts');
    let configObj = {};

    const getOpt = name => {
        const idx = args.indexOf(`--${name}`);
        if (idx < 0) return undefined;
        return args[idx + 1];
    };
    const hasFlag = name => args.includes(`--${name}`);

    const configFile = getOpt('config');
    if (configFile) {
        try {
            configObj = JSON.parse(readFileSync(resolve(configFile), 'utf8'));
        } catch (e) {
            fail(`failed to read config file "${configFile}": ${e.message}`);
        }
    }
    if (!configObj || typeof configObj !== 'object' || Array.isArray(configObj)) fail('config file must contain an object');

    const name = getOpt('name');
    if (name) configObj.name = name;

    const start = getOpt('start');
    if (start) configObj.start = start;

    const hidden = getOpt('hidden');
    if (hidden) {
        configObj.hidden = hidden.split(/[x,]/).map(s => Number(s.trim()));
    }

    const arenas = getOpt('arenas');
    if (arenas !== undefined) configObj.arenas = Number(arenas);

    const K = getOpt('K');
    if (K !== undefined) configObj.K = Number(K);

    const E = getOpt('E');
    if (E !== undefined) configObj.E = Number(E);

    const P = getOpt('P');
    if (P !== undefined) configObj.P = Number(P);

    const W = getOpt('W');
    if (W !== undefined) configObj.W = Number(W);

    const L = getOpt('L');
    if (L !== undefined) configObj.L = Number(L);

    const gap = getOpt('gap');
    if (gap !== undefined) configObj.gap = Number(gap);

    const cols = getOpt('cols');
    if (cols !== undefined) configObj.cols = Number(cols);

    const T = getOpt('T');
    if (T !== undefined) configObj.T = Number(T);

    const dt = getOpt('dt');
    if (dt !== undefined) configObj.dt = Number(dt);

    const thinkEvery = getOpt('thinkEvery');
    if (thinkEvery !== undefined) configObj.thinkEvery = Number(thinkEvery);

    const environment = getOpt('environment');
    if (environment) configObj.environment = environment;

    if (hasFlag('contacts')) configObj.contacts = true;
    if (hasFlag('no-contacts')) configObj.contacts = false;

    const catchMargin = getOpt('catchMargin');
    if (catchMargin !== undefined) configObj.catchMargin = Number(catchMargin);

    const spawnSeparation = getOpt('spawnSeparation');
    if (spawnSeparation !== undefined) configObj.spawnSeparation = Number(spawnSeparation);

    const generations = getOpt('generations');
    if (generations !== undefined) configObj.generations = Number(generations);

    const sigma = getOpt('sigma');
    const rate = getOpt('rate');
    const decay = getOpt('decay');
    if (sigma !== undefined || rate !== undefined || decay !== undefined) {
        configObj.optimizer = {
            ...configObj.optimizer,
            ...(sigma !== undefined ? { sigma: Number(sigma) } : {}),
            ...(rate !== undefined ? { rate: Number(rate) } : {}),
            ...(decay !== undefined ? { decay: Number(decay) } : {}),
        };
    }

    const archiveEvery = getOpt('archive-every');
    const archiveCap = getOpt('archive-cap');
    const archiveShare = getOpt('archive-share');
    if (archiveEvery !== undefined || archiveCap !== undefined || archiveShare !== undefined) {
        configObj.archive = {
            ...configObj.archive,
            ...(archiveEvery !== undefined ? { every: Number(archiveEvery) } : {}),
            ...(archiveCap !== undefined ? { cap: Number(archiveCap) } : {}),
            ...(archiveShare !== undefined ? { share: Number(archiveShare) } : {}),
        };
    }

    const masterSeed = getOpt('master-seed');
    const domain = getOpt('domain');
    if (masterSeed !== undefined || domain !== undefined) {
        configObj.seeds = {
            ...configObj.seeds,
            ...(masterSeed !== undefined ? { master: Number(masterSeed) } : {}),
            ...(domain !== undefined ? { domain: Number(domain) } : {}),
        };
    }

    const outDir = getOpt('out-dir');
    if (outDir) configObj.outDir = outDir;

    if (hasFlag('export-onnx')) configObj.exportOnnx = true;

    return configObj;
}
