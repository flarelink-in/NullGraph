// Step 7, part 2 (docs/agents-cloning-design.md §4.1): a recording on disk.
//
// A dataset is a folder: header.json and, per species, two flat little-endian files.
//   <species>.f32   rows of obsDim + 2 × actDim floats: the observation, the label, the driver's action
//   <species>.u32   rows of 4 words: episode, tick, agent index, flags (bit 0 frozen, bits 4–7 reason)
// header.json holds what makes the files reproducible, and their sizes and SHA-256.
//
// The row helpers are plain JS and load in a browser; only writeDataset() and readDataset() need Node.
import * as A from '../../dist/agents.js';

export const DATASET_FORMAT = 'null-graph/agents-dataset';
export const DATASET_VERSION = 1;
/** Words per row of a .u32 file. */
export const ROW_WORDS = 4;
export const ROW = Object.freeze({ EPISODE: 0, TICK: 1, AGENT: 2, FLAGS: 3 });
export const FLAG_FROZEN = 1;
export const rowFlags = (frozen, why) => ((why & 15) << 4) | (frozen ? FLAG_FROZEN : 0);
export const flagsReason = flags => (flags >>> 4) & 15;

/** Floats per row of a .f32 file: the observation, the label and the driver's action. */
export const rowFloats = definition => definition.obsDim + 2 * definition.actDim;

/**
 * Whether a row is kept, by a hash of the agent, the tick and the species: the same on every run,
 * and independent of the order rows are visited in.
 */
export function keepRow(agent, tick, species, rate) {
    return rate >= 1 || A.rand01(agent, tick, 0x6b656570 + species) < rate;
}

/** Growable rows of one species. Floats are held as their words, so a row is copied bit for bit. */
export class Rows {
    constructor(floats) {
        this.floats = floats;
        this.rows = 0;
        this.values = new Uint32Array(floats * 1024);
        this.words = new Uint32Array(ROW_WORDS * 1024);
    }

    /** `obs`, `label` and `act` are Uint32Array views of one agent's f32 rows. */
    push(obs, label, act, episode, tick, agent, flags) {
        if (obs.length + label.length + act.length !== this.floats) {
            throw new Error(`dataset: a row has ${obs.length + label.length + act.length} floats; this dataset's rows have ${this.floats}.`);
        }
        if ((this.rows + 1) * this.floats > this.values.length) {
            const values = new Uint32Array(this.values.length * 2), words = new Uint32Array(this.words.length * 2);
            values.set(this.values);
            words.set(this.words);
            this.values = values;
            this.words = words;
        }
        const at = this.rows * this.floats, w = this.rows * ROW_WORDS;
        this.values.set(obs, at);
        this.values.set(label, at + obs.length);
        this.values.set(act, at + obs.length + label.length);
        this.words[w + ROW.EPISODE] = episode;
        this.words[w + ROW.TICK] = tick;
        this.words[w + ROW.AGENT] = agent;
        this.words[w + ROW.FLAGS] = flags;
        this.rows++;
    }

    /** The rows collected, in arrays of their own. */
    finish() {
        const values = this.values.slice(0, this.rows * this.floats);
        return { rows: this.rows, f32: new Float32Array(values.buffer), u32: this.words.slice(0, this.rows * ROW_WORDS) };
    }
}

/**
 * Row indices for training and for judging. A held-out episode's rows never train; frozen rows
 * train nothing either, because a frozen agent's action is never applied.
 */
export function splitRows(header, u32, { dropFrozen = true } = {}) {
    const heldOut = header.episodes.map(e => e.heldOut === true), train = [], test = [];
    for (let row = 0; row < u32.length / ROW_WORDS; row++) {
        const episode = u32[row * ROW_WORDS + ROW.EPISODE];
        if (episode >= heldOut.length) throw new Error(`dataset: row ${row} names episode ${episode}; header.json lists ${heldOut.length}.`);
        if (dropFrozen && (u32[row * ROW_WORDS + ROW.FLAGS] & FLAG_FROZEN)) continue;
        (heldOut[episode] ? test : train).push(row);
    }
    return { train: Uint32Array.from(train), heldOut: Uint32Array.from(test) };
}

/** What header.json records about a definition: enough to check a layout and to rebuild the scene. */
export function describeDefinition(definition, thinkEvery) {
    const w = definition.world;
    return {
        capacity: definition.capacity, world: { minX: w.minX, minZ: w.minZ, maxX: w.maxX, maxZ: w.maxZ },
        cellSize: definition.cellSize, maxScanPerCell: definition.maxScanPerCell, dt: definition.dt, thinkEvery,
        species: definition.species.map(s => ({ name: s.name, maxSpeed: s.maxSpeed, maxTurn: s.maxTurn, accel: s.accel,
            senseRadius: s.senseRadius, radius: s.radius })),
        obsDim: definition.obsDim, actDim: definition.actDim,
        obsNames: [...definition.obsNames], actNames: [...definition.actNames],
    };
}

const bytesOf = array => new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
const sha256 = async bytes => (await import('node:crypto')).createHash('sha256').update(bytes).digest('hex');

/** SHA-256 of a text, for the WGSL a dataset or a network was made with. Node only. */
export const hashText = text => sha256(new TextEncoder().encode(text));

/**
 * Writes a dataset folder. `header` is everything but the files; `species` maps a species name to
 * { rows, f32, u32 } as Rows.finish() returns them. Node only. Resolves to the written header.
 */
export async function writeDataset(dir, header, species) {
    const { mkdirSync, writeFileSync } = await import('node:fs'), { join } = await import('node:path');
    mkdirSync(dir, { recursive: true });
    const files = {};
    for (const [name, data] of Object.entries(species)) {
        for (const [extension, array] of [['f32', data.f32], ['u32', data.u32]]) {
            const file = `${name}.${extension}`, bytes = bytesOf(array);
            writeFileSync(join(dir, file), bytes);
            files[file] = { rows: data.rows, bytes: bytes.length, sha256: await sha256(bytes) };
        }
    }
    const written = { format: DATASET_FORMAT, version: DATASET_VERSION, ...header, files };
    writeFileSync(join(dir, 'header.json'), JSON.stringify(written, null, 2) + '\n');
    return written;
}

/**
 * Reads a dataset folder back, refusing a file whose size or hash is not the header's.
 * Resolves to { header, species: { <name>: { rows, f32, u32 } } }. Node only.
 */
export async function readDataset(dir) {
    const { readFileSync } = await import('node:fs'), { join } = await import('node:path');
    const header = JSON.parse(readFileSync(join(dir, 'header.json'), 'utf8'));
    if (header.format !== DATASET_FORMAT || header.version !== DATASET_VERSION) {
        throw new Error(`dataset: ${dir} is "${header.format}" version ${header.version}; this reader takes "${DATASET_FORMAT}" version ${DATASET_VERSION}.`);
    }
    const floats = header.definition.obsDim + 2 * header.definition.actDim, species = {};
    for (const [file, entry] of Object.entries(header.files)) {
        const [name, extension] = file.split('.'), stride = (extension === 'f32' ? floats : ROW_WORDS) * 4;
        const bytes = readFileSync(join(dir, file));
        if (bytes.length !== entry.bytes || entry.bytes !== entry.rows * stride) {
            throw new Error(`dataset: ${file} is ${bytes.length} bytes; header.json promises ${entry.bytes} (${entry.rows} rows × ${stride} bytes).`);
        }
        if (await sha256(bytes) !== entry.sha256) {
            throw new Error(`dataset: ${file} does not match header.json's SHA-256. It was changed or cut short.`);
        }
        const copy = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
        species[name] ??= { rows: entry.rows };
        species[name][extension] = extension === 'f32' ? new Float32Array(copy) : new Uint32Array(copy);
    }
    return { header, species };
}
