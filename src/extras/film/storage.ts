// D2.4 F3 (offline-films-browser-design.md §4.5): where a film goes when it does not stay in memory. A film
// written to OPFS lives in null-graph-films/<exportId>/ beside its manifest, film.json, and holds a Web Lock
// named for its export while it is written. A picked file is the app's: written through the handle it chose,
// never deleted. Either is written through a FileStreamGate, which commits the file only when the sink says the
// muxer has finalized, and otherwise aborts it (spike S1: mediabunny 1.61.3 closes its writer on cancel, and on
// a failed finalize too, and closing a FileSystemWritableFileStream commits it). listFilms, cleanupFilms and
// removeFilm are the app's view of OPFS: completed films stay until the app removes them, and an incomplete one
// is removed only once no writer holds its lock.

import { clause, fail, sentence } from './util';

/** The OPFS folder every film is written under, one folder per export. */
export const FILMS_DIRECTORY = 'null-graph-films';
/** Each export's manifest: its status, its name and, once committed, its report. */
export const MANIFEST = 'film.json';
/** A PNG sequence's frames, inside its export: frame-000000.png, frame-000001.png, … */
export const FRAMES_DIRECTORY = 'frames';

/** The Web Lock a film's writer holds for as long as it writes; cleanup removes a film only with it. */
export const filmLockName = (exportId: string) => `null-graph-film:${exportId}`;

/** frame-000000.png: frame j of a PNG sequence. */
export const frameFileName = (j: number) => `frame-${String(j).padStart(6, '0')}.png`;

const describe = (error: unknown) => (error instanceof Error ? `${error.name}: ${error.message}` : String(error));
const nav = () => (globalThis as { navigator?: Navigator }).navigator;

/**
 * A WritableStream for mediabunny's StreamTarget over a file's writable. Writes pass through, with their
 * backpressure. The stream's close commits nothing: mediabunny 1.61.3 closes it when it finalizes, but also when
 * it cancels and, in its finally, when a finalize fails. The sink commits the file with commit() once the muxer
 * has finalized; a discard aborts it, at once if the muxer has closed already. The first failed write is kept:
 * it, not the encoder, is what ended the film (§8).
 */
export class FileStreamGate {
    readonly stream: WritableStream;
    /** The first write the file refused (a full quota, a revoked permission). */
    writeError: unknown = null;
    /** Bytes the file holds once committed: the furthest any write reached. */
    bytes = 0;
    private readonly writer: WritableStreamDefaultWriter;
    private discarded = false;
    /** The muxer has closed its stream: every write it made has reached the file's writable. */
    private flushed = false;
    private settled: 'committed' | 'aborted' | null = null;

    constructor(file: WritableStream) {
        this.writer = file.getWriter();
        this.stream = new WritableStream({
            // mediabunny writes { type: 'write', position, data } (target.js), which a file's writable takes as is.
            write: async (chunk: { readonly type?: string; readonly position?: number; readonly size?: number; readonly data?: { readonly byteLength: number } }) => {
                try {
                    await this.writer.write(chunk);
                } catch (error) {
                    this.writeError ??= error;
                    throw error;
                }
                if (chunk.type === 'truncate') this.bytes = Math.min(this.bytes, chunk.size ?? this.bytes);
                else this.bytes = Math.max(this.bytes, (chunk.position ?? this.bytes) + (chunk.data?.byteLength ?? 0));
            },
            close: async () => {
                this.flushed = true;
                if (this.discarded) await this.abort();
            },
            abort: async reason => {
                await this.abort(reason);
            },
        });
    }

    get committed(): boolean {
        return this.settled === 'committed';
    }

    /** From now on the file can only be aborted; the muxer's close then aborts it at once. */
    discard(): void {
        this.discarded = true;
    }

    /**
     * Commit the file: what the muxer wrote becomes its content. Only after the muxer has finalized and closed its
     * stream, and never once discarded. Idempotent once committed.
     */
    async commit(): Promise<void> {
        if (this.settled === 'committed') return;
        if (this.settled === 'aborted' || this.discarded) fail('the film\'s file was discarded; it cannot be committed.');
        if (!this.flushed) fail('the muxer has not closed the film\'s file; it cannot be committed before the film is finalized.');
        await this.writer.close();
        this.settled = 'committed';
    }

    /** Abort the file's writable if it is still open: nothing it was given is kept. Idempotent; never throws. */
    async abort(reason: unknown = new DOMException('The film was discarded.', 'AbortError')): Promise<void> {
        if (this.settled) return;
        this.settled = 'aborted';
        try { await this.writer.abort(reason); } catch { /* an errored stream has nothing left to abort */ }
    }
}

/** What a manifest says of its film. */
export interface FilmManifest {
    readonly kind: 'null-graph-film';
    readonly version: 1;
    readonly exportId: string;
    /** `writing` until the payload is committed; then `completed` or `partial`. */
    readonly status: 'writing' | 'completed' | 'partial';
    /** The download name: metadata, never a path. */
    readonly name: string;
    readonly container: string;
    readonly mimeType: string;
    /** The payload's file name in the export's folder, or the frames' folder for a PNG sequence. */
    readonly payload: string;
    readonly created: string;
    readonly updated: string;
    readonly frames: number | null;
    readonly bytes: number | null;
    /** The film's report, once committed. */
    readonly report: unknown;
}

/** A held Web Lock, or none where the API is missing (§4.5: cleanup is then disabled). */
async function holdLock(name: string): Promise<{ readonly held: boolean; release(): void }> {
    const locks = nav()?.locks;
    if (typeof locks?.request !== 'function') return { held: false, release() {} };
    let release!: () => void;
    const held = new Promise<void>(resolve => { release = resolve; });
    const got = await new Promise<boolean>((resolve, reject) => {
        locks.request(name, { mode: 'exclusive', ifAvailable: true }, lock => {
            resolve(!!lock);
            return lock ? held : undefined;
        }).catch(reject);
    });
    // An export's id is a fresh random UUID: its lock is held elsewhere only if the id was reused.
    if (!got) fail(`the lock "${name}" is held elsewhere; an export id was used twice.`);
    return { held: true, release };
}

async function writeJSON(dir: FileSystemDirectoryHandle, name: string, value: unknown): Promise<void> {
    const handle = await dir.getFileHandle(name, { create: true });
    const writable = await handle.createWritable();
    try {
        await writable.write(JSON.stringify(value, null, 1));
        await writable.close();
    } catch (error) {
        try { await writable.abort(); } catch { /* the write's own error stands */ }
        throw error;
    }
}

async function readJSON(dir: FileSystemDirectoryHandle, name: string): Promise<unknown> {
    try {
        return JSON.parse(await (await (await dir.getFileHandle(name)).getFile()).text());
    } catch {
        return null;
    }
}

/** OPFS's root, or a refusal saying why there is none here. */
async function opfsRoot(): Promise<FileSystemDirectoryHandle> {
    const storage = nav()?.storage;
    if (typeof storage?.getDirectory !== 'function') fail('OPFS (navigator.storage.getDirectory) is not available here.');
    return storage.getDirectory();
}

/** Where a film is being written, and how it ends: committed, or discarded. */
export interface FilmStorage {
    readonly kind: 'opfs' | 'file';
    readonly exportId: string;
    /** `null-graph-films/<exportId>/film.mp4`, or null for a picked file (the app knows where it is). */
    readonly path: string | null;
    /** Whether the film's Web Lock is held while it is written (OPFS, where Web Locks exist). */
    readonly locked: boolean;
    /** The gate a video's muxer writes through; null for a PNG sequence. */
    readonly gate: FileStreamGate | null;
    /** A PNG sequence's frames folder; null for a video. */
    readonly frames: FileSystemDirectoryHandle | null;
    readonly committed: boolean;
    /**
     * After the payload is committed (the muxer has finalized and the sink committed the gate, or every frame is
     * written): mark the film complete or partial in its manifest, and give back the file. A manifest that cannot
     * be written leaves the film unmarked; it is then removed, and the error thrown (§4.5: complete only once marked).
     */
    commit(status: 'completed' | 'partial', fields: { readonly frames: number; readonly bytes: number; readonly report: unknown }): Promise<File | null>;
    /** Abort what is still open and, for OPFS, remove the export's folder, unless committed. Idempotent. */
    discard(): Promise<void>;
    /** After a commit: remove the film (OPFS), or nothing (a picked file is the app's). Idempotent. */
    dispose(): Promise<void>;
    /** Release the film's Web Lock. Idempotent. */
    release(): void;
}

export interface OpenStorageOptions {
    readonly exportId: string;
    readonly name: string;
    readonly container: string;
    readonly mimeType: string;
    /** A video's payload file ('film.mp4'), or a PNG sequence (its frames folder). */
    readonly payload: 'video' | 'png';
    readonly extension: string;
}

/**
 * Open a film's export in OPFS: take its lock, make null-graph-films/<exportId>/, write its manifest
 * (`writing`), and open the payload's writable (a video) or the frames' folder (a PNG sequence). What
 * fails part-way is undone before the error leaves.
 */
export async function openOpfsExport(o: OpenStorageOptions): Promise<FilmStorage> {
    const root = await opfsRoot();
    const lock = await holdLock(filmLockName(o.exportId));
    let films: FileSystemDirectoryHandle | null = null;
    let dir: FileSystemDirectoryHandle | null = null;
    let gate: FileStreamGate | null = null;
    const payloadName = o.payload === 'png' ? FRAMES_DIRECTORY : `film.${o.extension}`;
    const created = new Date().toISOString();
    const manifest = (status: FilmManifest['status'], more: Partial<FilmManifest> = {}): FilmManifest => ({
        kind: 'null-graph-film', version: 1, exportId: o.exportId, status, name: o.name, container: o.container, mimeType: o.mimeType,
        payload: payloadName, created, updated: new Date().toISOString(), frames: null, bytes: null, report: null, ...more,
    });
    const undo = async () => {
        await gate?.abort();
        if (films && dir) { try { await films.removeEntry(o.exportId, { recursive: true }); } catch { /* the open error stands */ } }
        lock.release();
    };
    let frames: FileSystemDirectoryHandle | null = null;
    let payload: FileSystemFileHandle | null = null;
    try {
        films = await root.getDirectoryHandle(FILMS_DIRECTORY, { create: true });
        // A released lock does not mean the id is unused: a completed film retains its directory.
        // Check under the writer's lock, before assigning dir, so undo never removes someone else's film.
        let exists = false;
        try {
            await films.getDirectoryHandle(o.exportId);
            exists = true;
        } catch (error) {
            if ((error as { name?: string } | null)?.name !== 'NotFoundError') throw error;
        }
        if (exists) fail(`the export id "${o.exportId}" already exists in OPFS; use a fresh export id.`);
        dir = await films.getDirectoryHandle(o.exportId, { create: true });
        await writeJSON(dir, MANIFEST, manifest('writing'));
        if (o.payload === 'png') {
            frames = await dir.getDirectoryHandle(FRAMES_DIRECTORY, { create: true });
        } else {
            payload = await dir.getFileHandle(payloadName, { create: true });
            gate = new FileStreamGate(await payload.createWritable());
        }
    } catch (error) {
        await undo();
        throw error;
    }
    const exportDir = dir;
    const filmsDir = films;
    let committed = false;
    let released = false;
    // One removal however many ask at once; a removal that fails may be asked for again.
    let removing: Promise<void> | null = null;
    const remove = () => (removing ??= filmsDir.removeEntry(o.exportId, { recursive: true }).catch(error => {
        removing = null;
        throw error;
    }));
    return {
        kind: 'opfs', exportId: o.exportId, path: `${FILMS_DIRECTORY}/${o.exportId}/${payloadName}`, locked: lock.held, gate, frames,
        get committed() { return committed; },
        async commit(status, fields) {
            if (gate && !gate.committed) fail('the film\'s payload is not committed; it cannot be marked complete.');
            // A video's size is its committed file's; a sequence's, the sum its sink counted.
            const file = payload ? await payload.getFile() : null;
            try {
                await writeJSON(exportDir, MANIFEST, manifest(status, { frames: fields.frames, bytes: file ? file.size : fields.bytes, report: fields.report }));
            } catch (error) {
                // Unmarked, the film would look incomplete to every later cleanup: remove it now, and say so.
                try { await remove(); } catch { /* the manifest's error stands */ }
                fail(`the film was written but could not be marked complete in OPFS (${clause(describe(error))}); it was removed.`);
            }
            committed = true;
            return file;
        },
        async discard() {
            if (committed) return;
            await gate?.abort();
            await remove();
        },
        async dispose() {
            if (!committed) return;
            await remove();
        },
        release() {
            if (released) return;
            released = true;
            lock.release();
        },
    };
}

/**
 * Open a file the user picked (§4.5): the app obtained `handle` in a user gesture; the library never opens a
 * picker, asks for permission, or deletes the file. A writable the browser refuses is refused in plain words.
 */
export async function openPickedFile(handle: FileSystemFileHandle, exportId: string): Promise<FilmStorage> {
    let writable: FileSystemWritableFileStream;
    try {
        writable = await handle.createWritable();
    } catch (error) {
        const denied = (error as { name?: string } | null)?.name === 'NotAllowedError' || (error as { name?: string } | null)?.name === 'SecurityError';
        fail(denied
            ? `the chosen file "${handle.name}" may not be written (${clause(describe(error))}); ask for write permission in a user gesture (handle.requestPermission({ mode: 'readwrite' })) and try again.`
            : `the chosen file "${handle.name}" could not be opened for writing: ${sentence(describe(error))}`);
    }
    const gate = new FileStreamGate(writable);
    return {
        kind: 'file', exportId, path: null, locked: false, gate, frames: null,
        get committed() { return gate.committed; },
        async commit() {
            if (!gate.committed) fail('the film\'s payload is not committed; it cannot be marked complete.');
            return handle.getFile();
        },
        async discard() {
            // The user's file keeps what it held before; a file the picker just made stays, empty.
            if (!gate.committed) await gate.abort();
        },
        async dispose() {},
        release() {},
    };
}

/** A film in OPFS, as listFilms finds it. */
export interface StoredFilm {
    readonly exportId: string;
    /** `unknown` when its manifest is missing or unreadable (it was cut off before writing one). */
    readonly status: FilmManifest['status'] | 'unknown';
    readonly name: string | null;
    readonly container: string | null;
    readonly mimeType: string | null;
    readonly created: string | null;
    readonly updated: string | null;
    readonly frames: number | null;
    readonly bytes: number | null;
    /** A writer holds the film's lock, so it is being written now; null where Web Locks are missing. */
    readonly live: boolean | null;
    readonly report: unknown;
    /** The payload, for a video; null for a PNG sequence, or when there is none. */
    getFile(): Promise<File | null>;
    /** A PNG sequence's frames folder, or null. */
    getFrames(): Promise<FileSystemDirectoryHandle | null>;
}

/** A manifest's status, `unknown` when it is missing, unreadable or not a film's. */
function statusOf(m: Partial<FilmManifest> | null): StoredFilm['status'] {
    const known = m && m.kind === 'null-graph-film';
    return known && (m.status === 'writing' || m.status === 'completed' || m.status === 'partial') ? m.status : 'unknown';
}

async function filmsFolder(): Promise<FileSystemDirectoryHandle | null> {
    try {
        return await (await opfsRoot()).getDirectoryHandle(FILMS_DIRECTORY);
    } catch {
        return null;
    }
}

/** The names of the Web Locks held now, by any context of this origin; null where Web Locks are missing. */
async function heldLockNames(): Promise<Set<string> | null> {
    const locks = nav()?.locks;
    if (typeof locks?.query !== 'function') return null;
    const { held = [] } = await locks.query();
    return new Set(held.map(l => l.name ?? ''));
}

/**
 * Every film in OPFS, oldest first: completed and partial ones (kept until the app removes them), and
 * incomplete ones, `live` when a writer, in this tab or another, holds its lock (§4.5). Lists; changes nothing.
 */
export async function listFilms(): Promise<StoredFilm[]> {
    const films = await filmsFolder();
    if (!films) return [];
    const held = await heldLockNames();
    const out: StoredFilm[] = [];
    for await (const [exportId, handle] of films.entries()) {
        if (handle.kind !== 'directory') continue;
        const dir = handle as FileSystemDirectoryHandle;
        const m = await readJSON(dir, MANIFEST) as Partial<FilmManifest> | null;
        const known = m && m.kind === 'null-graph-film';
        const payload = known && typeof m.payload === 'string' ? m.payload : null;
        out.push({
            exportId, status: statusOf(m),
            name: known && typeof m.name === 'string' ? m.name : null,
            container: known && typeof m.container === 'string' ? m.container : null,
            mimeType: known && typeof m.mimeType === 'string' ? m.mimeType : null,
            created: known && typeof m.created === 'string' ? m.created : null,
            updated: known && typeof m.updated === 'string' ? m.updated : null,
            frames: known && typeof m.frames === 'number' ? m.frames : null,
            bytes: known && typeof m.bytes === 'number' ? m.bytes : null,
            live: held ? held.has(filmLockName(exportId)) : null,
            report: known ? m.report ?? null : null,
            async getFile() {
                if (!payload || payload === FRAMES_DIRECTORY) return null;
                try { return await (await dir.getFileHandle(payload)).getFile(); } catch { return null; }
            },
            async getFrames() {
                if (payload !== FRAMES_DIRECTORY) return null;
                try { return await dir.getDirectoryHandle(FRAMES_DIRECTORY); } catch { return null; }
            },
        });
    }
    return out.sort((a, b) => (a.created ?? '').localeCompare(b.created ?? ''));
}

/**
 * Remove one film from OPFS, completed or not, once no writer holds its lock: the app's explicit deletion
 * (§4.5). Refuses, by name, a film being written now. Without Web Locks, a film whose writable is still open
 * in another tab cannot be removed either: the browser refuses that removal itself.
 */
export async function removeFilm(exportId: string): Promise<void> {
    if (typeof exportId !== 'string' || !exportId || exportId.includes('/')) fail(`removeFilm needs an export id; got ${JSON.stringify(exportId)}.`);
    const films = await filmsFolder();
    if (!films) fail(`there is no film "${exportId}" in OPFS.`);
    const remove = async () => {
        try {
            await films.removeEntry(exportId, { recursive: true });
        } catch (error) {
            if ((error as { name?: string } | null)?.name === 'NotFoundError') fail(`there is no film "${exportId}" in OPFS.`);
            throw error;
        }
    };
    const locks = nav()?.locks;
    if (typeof locks?.request !== 'function') return remove();
    let live = false;
    await locks.request(filmLockName(exportId), { mode: 'exclusive', ifAvailable: true }, async lock => {
        if (!lock) { live = true; return; }
        await remove();
    });
    if (live) fail(`the film "${exportId}" is being written now (its lock is held); stop or cancel that film first.`);
}

export interface FilmCleanup {
    readonly removed: readonly string[];
    readonly kept: readonly { readonly exportId: string; readonly reason: string }[];
}

/**
 * Remove the incomplete films no writer is writing: cut off when their tab closed, crashed or was discarded
 * (§4.5). A film is removed only while this holds its lock, so a live writer anywhere keeps its film, and only if
 * its manifest, read again under that lock, still says it is incomplete. Completed and partial films are kept:
 * they need the app's removeFilm. Without Web Locks a live film cannot be told from a dead one, so nothing is
 * removed. Never resumes a film.
 */
export async function cleanupFilms(): Promise<FilmCleanup> {
    const locks = nav()?.locks;
    if (typeof locks?.request !== 'function') {
        fail('Web Locks (navigator.locks) are not available here, so a film being written in another tab cannot be told from an abandoned one; nothing was removed.');
    }
    const removed: string[] = [];
    const kept: { exportId: string; reason: string }[] = [];
    const films = await filmsFolder();
    if (!films) return { removed, kept };
    for (const film of await listFilms()) {
        if (film.status === 'completed' || film.status === 'partial') {
            kept.push({ exportId: film.exportId, reason: `${film.status}: remove it with removeFilm` });
            continue;
        }
        await locks.request(filmLockName(film.exportId), { mode: 'exclusive', ifAvailable: true }, async lock => {
            if (!lock) {
                kept.push({ exportId: film.exportId, reason: 'being written now' });
                return;
            }
            // Listed before its lock was asked for: its writer may since have marked it complete and let go.
            let dir: FileSystemDirectoryHandle;
            try { dir = await films.getDirectoryHandle(film.exportId); } catch { return; } // gone meanwhile: nothing to remove
            const status = statusOf(await readJSON(dir, MANIFEST) as Partial<FilmManifest> | null);
            if (status === 'completed' || status === 'partial') {
                kept.push({ exportId: film.exportId, reason: `${status}: remove it with removeFilm` });
                return;
            }
            try {
                await films.removeEntry(film.exportId, { recursive: true });
                removed.push(film.exportId);
            } catch (error) {
                kept.push({ exportId: film.exportId, reason: `could not be removed (${clause(describe(error))})` });
            }
        });
    }
    return { removed, kept };
}
