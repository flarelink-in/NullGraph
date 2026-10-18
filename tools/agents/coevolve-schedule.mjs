// Phase 14.3, design §8. Schedule, seeds manifests and archive management.
import { createHash } from 'node:crypto';
import { tupleHash } from './es.mjs';

const fail = message => { throw new Error(`coevolve: ${message}`); };

export const STREAM_NOISE = 1;
export const STREAM_SPAWN = 2;
export const STREAM_OPPONENT = 3;
export const STREAM_PERMUTATION = 4;
export const STREAM_INIT = 5;

export function evolvingSpecies(t) {
    if (!Number.isInteger(t) || t < 0) fail('generation t must be a non-negative integer');
    return t % 2 === 0 ? 'prey' : 'wolf';
}

export function opponentSpecies(t) {
    if (!Number.isInteger(t) || t < 0) fail('generation t must be a non-negative integer');
    return t % 2 === 0 ? 'wolf' : 'prey';
}

/** Precompute numerical episode seed manifest for the requested budget.
 * Detects cross-pair/cross-episode collisions and regenerates deterministically with retry counters.
 */
export function buildSeedManifest({ masterSeed, domain, generations, K, E }) {
    if (!Number.isInteger(masterSeed) || masterSeed < 0 || masterSeed > 0xffffffff)
        fail('masterSeed must be u32');
    if (!Number.isInteger(domain) || domain < 0 || domain > 0xffffffff)
        fail('domain must be u32');
    if (!Number.isInteger(generations) || generations < 1)
        fail('generations must be >= 1');
    if (!Number.isInteger(K) || K < 1) fail('K must be >= 1');
    if (!Number.isInteger(E) || E < 1) fail('E must be >= 1');

    const B = K * E;
    const usedSeeds = new Set();
    const retries = [];
    const seedsByGeneration = [];

    for (let t = 0; t < generations; t++) {
        const genSeeds = [];
        for (let k = 0; k < K; k++) {
            for (let e = 0; e < E; e++) {
                let retry = 0;
                let seed;
                while (true) {
                    seed = tupleHash(masterSeed, domain, STREAM_SPAWN, t, k, e, retry);
                    if (!usedSeeds.has(seed)) {
                        usedSeeds.add(seed);
                        if (retry > 0) retries.push({ t, k, e, retry, seed });
                        genSeeds.push(seed);
                        break;
                    }
                    retry++;
                    if (retry > 10000) fail('excessive seed collisions in manifest');
                }
            }
        }
        seedsByGeneration.push(genSeeds);
    }

    const manifestData = JSON.stringify({ masterSeed, domain, generations, K, E, seedsByGeneration, retries });
    const manifestHash = createHash('sha256').update(manifestData).digest('hex');

    return {
        masterSeed,
        domain,
        generations,
        K,
        E,
        B,
        seedsByGeneration,
        retries,
        totalEpisodes: generations * B,
        manifestHash,
    };
}

/** Fisher-Yates seeded permutation of B arenas for generation t. */
export function permutationForGeneration({ masterSeed, domain, t, B }) {
    if (!Number.isInteger(masterSeed) || masterSeed < 0 || masterSeed > 0xffffffff) fail('masterSeed must be u32');
    if (!Number.isInteger(domain) || domain < 0 || domain > 0xffffffff) fail('domain must be u32');
    if (!Number.isInteger(t) || t < 0) fail('generation t must be >= 0');
    if (!Number.isInteger(B) || B < 1) fail('B must be >= 1');

    const perm = Array.from({ length: B }, (_, i) => i);
    for (let i = B - 1; i > 0; i--) {
        const r = tupleHash(masterSeed, domain, STREAM_PERMUTATION, t, i) % (i + 1);
        const temp = perm[i];
        perm[i] = perm[r];
        perm[r] = temp;
    }
    return perm;
}

/** Draws opponent for candidate pair k at generation t: with probability 1-h current (row 2K),
 * otherwise uniform sample from archive.
 */
export function opponentForPair({ masterSeed, domain, t, k, archive, h, K }) {
    if (!Number.isInteger(masterSeed) || masterSeed < 0 || masterSeed > 0xffffffff) fail('masterSeed must be u32');
    if (!Number.isInteger(domain) || domain < 0 || domain > 0xffffffff) fail('domain must be u32');
    if (!Number.isInteger(t) || t < 0) fail('t must be >= 0');
    if (!Number.isInteger(k) || k < 0) fail('k must be >= 0');
    if (!Number.isFinite(h) || h < 0 || h > 1) fail('h must be in [0, 1]');
    if (!Number.isInteger(K) || K < 1) fail('K must be >= 1');

    const u = (tupleHash(masterSeed, domain, STREAM_OPPONENT, t, k, 0) + 0.5) / 4294967296;
    if (h > 0 && u < h && Array.isArray(archive) && archive.length > 0) {
        const uIdx = (tupleHash(masterSeed, domain, STREAM_OPPONENT, t, k, 1) + 0.5) / 4294967296;
        const idx = Math.floor(uIdx * archive.length);
        const snap = archive[idx];
        return {
            type: 'archive',
            index: idx,
            id: snap.id,
            row: 2 * K + 1 + idx,
            params: snap.params,
        };
    }
    return {
        type: 'current',
        index: -1,
        id: 'current',
        row: 2 * K,
    };
}

/** Computes deterministic noise checksum across candidate directions for generation t. */
export function noiseChecksum(directions) {
    const hash = createHash('sha256');
    for (let k = 0; k < directions.length; k++) {
        const dir = directions[k];
        hash.update(Buffer.from(dir.buffer, dir.byteOffset, dir.byteLength));
    }
    return hash.digest('hex');
}

/** Archive manager: initial snapshot 0 before t=0, append current snapshot after (t+1)%H == 0,
 * cap M >= 2 with FIFO eviction at index 1 (retaining start at index 0).
 */
export function createArchiveManager({ cap = 32, every = 5 } = {}) {
    if (!Number.isSafeInteger(cap) || cap < 2) fail('archive cap must be integer >= 2');
    if (!Number.isSafeInteger(every) || every < 1) fail('archive every must be integer >= 1');

    const archives = {
        prey: [],
        wolf: [],
    };

    return {
        init({ prey, wolf }) {
            if (!prey?.params || !wolf?.params) fail('archive init requires prey and wolf params');
            archives.prey = [{
                id: 'prey-0000',
                generation: 0,
                params: Float64Array.from(prey.params),
            }];
            archives.wolf = [{
                id: 'wolf-0000',
                generation: 0,
                params: Float64Array.from(wolf.params),
            }];
        },

        maybeSnapshot(t, species, params) {
            if (!['prey', 'wolf'].includes(species)) fail(`unknown species "${species}"`);
            if (!params || !(params instanceof Float64Array)) fail('snapshot params must be Float64Array');
            if ((t + 1) % every !== 0) return null;

            const id = `${species}-${String(t + 1).padStart(4, '0')}`;
            const snap = {
                id,
                generation: t + 1,
                params: Float64Array.from(params),
            };
            archives[species].push(snap);

            let evicted = null;
            if (archives[species].length > cap) {
                // Remove index 1, keeping index 0 (initial start)
                evicted = archives[species].splice(1, 1)[0];
            }
            return { snapshotted: snap, evicted };
        },

        get(species, idx) {
            return archives[species]?.[idx];
        },

        list(species) {
            return archives[species]?.slice() ?? [];
        },
    };
}
