// Phase 14.3, design §4, §9. Checkpoints, logs and deterministic model exports.
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync, renameSync, rmSync, openSync, closeSync, fsyncSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { execSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { release } from 'node:os';
import { fileURLToPath } from 'node:url';
import { modelText } from './model.mjs';
import { onnxFromNetwork } from './onnx.mjs';

const fail = message => { throw new Error(`coevolve: ${message}`); };
const sha256 = content => createHash('sha256').update(content).digest('hex');
const repoRoot = fileURLToPath(new URL('../../', import.meta.url));
// Everything a run's results or checkpoint bytes depend on, beside the config and seeds.
const SOURCES = ['./coevolve.mjs', './coevolve-config.mjs', './coevolve-schedule.mjs',
    './coevolve-checkpoint.mjs', './coevolve-runtime.mjs', './coevolve-guard.mjs', './es.mjs',
    './mlp-train.mjs', './model.mjs', './onnx.mjs', '../film/guard.mjs',
    './scenario/arena-layout.mjs', './scenario/herd-wolves-arenas.mjs', './scenario/herd-wolves.mjs',
    '../../dist/index.js', '../../dist/agents.js', '../../dist/physics.js', '../../test/gpu/node-gpu.mjs'];
const LIMITS = ['maxBufferSize', 'maxStorageBufferBindingSize', 'maxStorageBuffersPerShaderStage',
    'maxComputeWorkgroupsPerDimension', 'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupStorageSize'];

function dawnVersion() {
    try { return createRequire(import.meta.url)('webgpu/package.json').version; }
    catch { return 'unknown'; }
}

function durableWrite(path, content, flags = 'w') {
    const fd = openSync(path, flags);
    try { writeFileSync(fd, content); fsyncSync(fd); }
    finally { closeSync(fd); }
}

export function writeStatus(targetDir, status) {
    atomicWriteFile(join(targetDir, 'status.json'), JSON.stringify(status, null, 1) + '\n');
}

export function atomicWriteFile(filePath, content) {
    const dir = resolve(filePath, '..');
    mkdirSync(dir, { recursive: true });
    const tmp = `${filePath}.${process.pid}.${Date.now()}.tmp`;
    try { durableWrite(tmp, content, 'wx'); renameSync(tmp, filePath); }
    finally { rmSync(tmp, { force: true }); }
}

export function atomicWriteDir(targetDir, files) {
    if (existsSync(targetDir)) fail(`checkpoint already exists: ${targetDir}`);
    const tmpDir = `${targetDir}.${process.pid}.${Date.now()}.tmp`;
    mkdirSync(tmpDir, { recursive: true });
    try {
        for (const [filename, content] of Object.entries(files)) {
            durableWrite(join(tmpDir, filename), content, 'wx');
        }
        renameSync(tmpDir, targetDir);
    } catch (e) {
        try { rmSync(tmpDir, { recursive: true, force: true }); } catch { /* ignore */ }
        throw e;
    }
}

export function getRepoMetadata() {
    let commit = 'unknown';
    let dirty = false;
    let gitAvailable = false;
    try {
        const options = { cwd: repoRoot, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] };
        commit = execSync('git rev-parse HEAD', options).trim();
        const status = execSync('git status --porcelain', options).trim();
        dirty = status.length > 0;
        gitAvailable = true;
    } catch {
        // Not in git or git failed
    }
    return { commit, dirty, gitAvailable };
}

export function createRunDirectory(targetDir) {
    if (existsSync(targetDir)) fail(`run directory already exists: "${targetDir}" (no resume)`);
    mkdirSync(resolve(targetDir, '..'), { recursive: true });
    mkdirSync(targetDir); // Exclusive claim, including against a concurrent process.
    writeStatus(targetDir, { state: 'incomplete', completedUpdates: 0 });
    mkdirSync(join(targetDir, 'archive'), { recursive: true });
}

/** `start` names the start networks' source files and hashes; `limits` is the device's. */
export function writeRunJson(targetDir, { config, seedManifest, start, adapterInfo, limits, repoMetadata }) {
    const sourceHashes = {};
    for (const path of SOURCES) sourceHashes[path] = sha256(readFileSync(new URL(path, import.meta.url)));
    const metadata = {
        schema: config.schema,
        algorithm: config.algorithm,
        config,
        configHash: config.configHash,
        sourceHashes,
        seedManifest,
        start: start ?? null,
        repo: repoMetadata ?? getRepoMetadata(),
        adapter: Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'driver'].map(key => [key, adapterInfo?.[key] ?? 'unknown'])),
        limits: limits ? Object.fromEntries(LIMITS.map(key => [key, limits[key] ?? null])) : null,
        platform: {
            node: process.version,
            platform: process.platform,
            arch: process.arch,
            os: release(),
            dawn: dawnVersion(),
            dawnFlags: process.env.NG_DAWN_FLAGS ?? '',
        },
    };
    atomicWriteFile(join(targetDir, 'run.json'), JSON.stringify(metadata, null, 1) + '\n');
}

export function appendLog(targetDir, entry) {
    const line = JSON.stringify(entry) + '\n';
    durableWrite(join(targetDir, 'log.jsonl'), line, 'a');
}

/** `experiment` (experimentOf(config)) makes a checkpoint evaluable in its own environment alone. */
export function writeGenerationCheckpoint(targetDir, {
    generation,
    preyNetwork,
    wolfNetwork,
    obsNames,
    actNames,
    configHash,
    experiment,
    exportOnnx = false,
}) {
    const dirName = `gen-${String(generation).padStart(4, '0')}`;
    const genDir = join(targetDir, dirName);
    if (existsSync(genDir)) fail(`checkpoint already exists: ${genDir}`);
    if (!experiment || typeof experiment !== 'object') fail('checkpoint manifest requires the experiment settings');

    // Deterministic model texts: no timestamps or mutable paths
    const preyJson = modelText({
        species: 'prey',
        obsNames,
        actNames,
        network: preyNetwork,
        generation,
        configHash,
    });
    const wolfJson = modelText({
        species: 'wolf',
        obsNames,
        actNames,
        network: wolfNetwork,
        generation,
        configHash,
    });

    const manifest = {
        generation,
        configHash,
        experiment,
        models: {
            prey: { hash: sha256(preyJson) },
            wolf: { hash: sha256(wolfJson) },
        },
    };
    const manifestJson = JSON.stringify(manifest, null, 1) + '\n';

    const files = {
        'prey.mlp.json': preyJson,
        'wolf.mlp.json': wolfJson,
        'manifest.json': manifestJson,
    };

    if (exportOnnx) {
        const preyOnnx = onnxFromNetwork(preyNetwork, {
            name: `prey-gen-${generation}`,
            metadata: { species: 'prey', generation: String(generation), configHash },
        });
        const wolfOnnx = onnxFromNetwork(wolfNetwork, {
            name: `wolf-gen-${generation}`,
            metadata: { species: 'wolf', generation: String(generation), configHash },
        });
        files['prey.onnx'] = Buffer.from(preyOnnx);
        files['wolf.onnx'] = Buffer.from(wolfOnnx);
    }

    atomicWriteDir(genDir, files);
}

export function writeArchiveSnapshot(targetDir, {
    snapshot,
    species,
    network,
    obsNames,
    actNames,
    configHash,
}) {
    const filename = `${snapshot.id}.mlp.json`;
    const dest = join(targetDir, 'archive', filename);
    if (existsSync(dest)) fail(`archive snapshot already exists: ${dest}`);

    const content = modelText({
        species,
        obsNames,
        actNames,
        network,
        generation: snapshot.generation,
        configHash,
    });
    atomicWriteFile(dest, content);
    // Append-only index: stable ID, insertion generation and model hash per snapshot (design §4).
    durableWrite(join(targetDir, 'archive', 'index.jsonl'), JSON.stringify({
        id: snapshot.id, species, generation: snapshot.generation, hash: sha256(content) }) + '\n', 'a');
}
