// Phase 14.3, design §4, §9. Checkpoints, logs and deterministic model exports.
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, writeFileSync, appendFileSync, renameSync, rmSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { execSync } from 'node:child_process';
import { modelText } from './model.mjs';
import { onnxFromNetwork } from './onnx.mjs';

const fail = message => { throw new Error(`coevolve: ${message}`); };

export function atomicWriteFile(filePath, content) {
    const dir = resolve(filePath, '..');
    mkdirSync(dir, { recursive: true });
    const tmp = `${filePath}.${process.pid}.${Date.now()}.tmp`;
    writeFileSync(tmp, content);
    renameSync(tmp, filePath);
}

export function atomicWriteDir(targetDir, files) {
    if (existsSync(targetDir)) return; // Already exists
    const tmpDir = `${targetDir}.${process.pid}.${Date.now()}.tmp`;
    mkdirSync(tmpDir, { recursive: true });
    try {
        for (const [filename, content] of Object.entries(files)) {
            writeFileSync(join(tmpDir, filename), content);
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
        commit = execSync('git rev-parse HEAD', { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
        const status = execSync('git status --porcelain', { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
        dirty = status.length > 0;
        gitAvailable = true;
    } catch {
        // Not in git or git failed
    }
    return { commit, dirty, gitAvailable };
}

export function createRunDirectory(targetDir) {
    if (existsSync(targetDir)) fail(`run directory already exists: "${targetDir}" (no resume)`);
    mkdirSync(targetDir, { recursive: true });
    mkdirSync(join(targetDir, 'archive'), { recursive: true });
}

export function writeRunJson(targetDir, { config, seedManifest, adapterInfo, repoMetadata }) {
    const metadata = {
        schema: config.schema,
        algorithm: config.algorithm,
        config,
        configHash: config.configHash,
        seedManifest: {
            masterSeed: seedManifest.masterSeed,
            domain: seedManifest.domain,
            totalEpisodes: seedManifest.totalEpisodes,
            manifestHash: seedManifest.manifestHash,
            retries: seedManifest.retries,
        },
        repo: repoMetadata ?? getRepoMetadata(),
        adapter: adapterInfo ?? {},
        platform: {
            node: process.version,
            platform: process.platform,
            arch: process.arch,
        },
    };
    atomicWriteFile(join(targetDir, 'run.json'), JSON.stringify(metadata, null, 1) + '\n');
}

export function appendLog(targetDir, entry) {
    const line = JSON.stringify(entry) + '\n';
    appendFileSync(join(targetDir, 'log.jsonl'), line, 'utf8');
}

export function writeGenerationCheckpoint(targetDir, {
    generation,
    preyNetwork,
    wolfNetwork,
    obsNames,
    actNames,
    configHash,
    exportOnnx = false,
}) {
    const dirName = `gen-${String(generation).padStart(4, '0')}`;
    const genDir = join(targetDir, dirName);
    if (existsSync(genDir)) return; // Already exported

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

    const preyHash = createHash('sha256').update(preyJson).digest('hex');
    const wolfHash = createHash('sha256').update(wolfJson).digest('hex');

    const manifest = {
        generation,
        configHash,
        models: {
            prey: { hash: preyHash },
            wolf: { hash: wolfHash },
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
    if (existsSync(dest)) return;

    const content = modelText({
        species,
        obsNames,
        actNames,
        network,
        generation: snapshot.generation,
        configHash,
    });
    atomicWriteFile(dest, content);
}
