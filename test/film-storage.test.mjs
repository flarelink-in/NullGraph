// D2.4, phase F3 (docs/offline-films-browser-design.md §4.5, §10 B9): null-graph/film's storage, headless, on
// the stand-ins of test/film-stand-ins.mjs: an OPFS whose writables commit on close and drop on abort, Web
// Locks, and mediabunny's StreamTarget. Where a film goes (auto, OPFS, memory, a picked file), when it is
// committed and marked complete, what a discard leaves (nothing), partials, quota and manifest failures, the
// films an app can list, clean up and remove, PNG sequences, and the probe's operational check.
// `npm run test:film` runs it.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import v8 from 'node:v8';
import vm from 'node:vm';
import {
    FILMS_DIRECTORY, MANIFEST, bitrateFor, cleanupFilms, createFilmSession, filmLockName, frameFileName, frameStamp, listFilms, openOpfsExport,
    predictBytes, probeFilm, recordFilm, removeFilm,
} from '../dist/film.js';
import { readPNG } from '../tools/film/png-read.mjs';
import {
    FakeVideoFrame, counterWorkload, fakeLocks, fakeMediabunny, fakeOPFS, filmDevice, frames, installStorage, installWebCodecs, pixels, sleep,
} from './film-stand-ins.mjs';

// The film's engine announces each device it adopts; against stand-in devices that is noise.
const log = console.log;
console.log = (...a) => { if (!/^NullGraph: Adopted/.test(String(a[0]))) log(...a); };

/** 6 frames at 60 fps, no warm-up: frame j shows j + 1 steps. */
const SMALL = { width: 64, height: 36, fps: 60, seconds: 0.1 };

/** A storage test: F2's stand-ins, with OPFS, Web Locks and a storage estimate (`storage`'s options) on the navigator. */
const f3 = (name, fn, storage = {}) => test(name, async () => {
    const uninstall = installWebCodecs();
    try { await fn(installStorage(storage)); } finally { uninstall(); }
});

/** recordFilm on stand-ins, and what happened, whether a film was made or not. */
async function film({ device = filmDevice(), workload = {}, mb = {}, ...options } = {}) {
    const work = counterWorkload(device, workload);
    const mediabunny = fakeMediabunny(mb);
    const reports = [];
    let result = null, error = null;
    try {
        result = await recordFilm({ device, createWorkload: work.create, film: SMALL, mediabunny, onReport: r => reports.push(r), ...options });
    } catch (e) {
        error = e;
    }
    return { result, error, report: reports[0] ?? null, device, workload: work.log, mb: mediabunny.log };
}

/** The films folder's tree: { exportId: { 'film.json': bytes, 'film.mp4': bytes } }. */
const films = async opfs => (await opfs.tree())[FILMS_DIRECTORY] ?? {};
const manifest = async (opfs, exportId) => JSON.parse(await (await (await (await opfs.folder(FILMS_DIRECTORY, exportId)).getFileHandle(MANIFEST)).getFile()).text());
/** The films folder's tree now, without awaiting: what an onProgress callback can look at. */
const peek = opfs => {
    const walk = node => Object.fromEntries([...node.children].map(([k, c]) => [k, c.kind === 'file' ? c.data.length : walk(c)]));
    const folder = opfs.root.children.get(FILMS_DIRECTORY);
    return folder ? walk(folder) : {};
};

// --- Where the film goes -------------------------------------------------------------------------------

f3('B9: auto films to OPFS where it is writable: committed only once finalized, marked complete, locked while written, removed by dispose', async ({ opfs, locks }) => {
    let during = null;
    const run = await film({
        onProgress: p => {
            if (p.phase !== 'recording' || p.frame !== 3) return;
            during = { tree: peek(opfs), held: [...locks.held] };
        },
    });
    assert.equal(run.result?.status, 'completed', String(run.error));
    const { result } = run;
    const id = result.report.exportId;
    assert.deepEqual([result.target, result.exportId, result.sequence], ['opfs', id, null]);
    assert.deepEqual(await frames(result.file), Array.from({ length: 6 }, (_, j) => [frameStamp(j, 60).timestampUs, frameStamp(j, 60).durationUs, j + 1]),
        'the file is not the film');
    assert.deepEqual(result.report.output.target, 'opfs');
    assert.equal(result.report.output.targetReason, 'OPFS is writable here (Q3)');
    assert.deepEqual(result.report.storage, {
        path: `${FILMS_DIRECTORY}/${id}/film.mp4`, locked: true, committed: true,
        needed: predictBytes(bitrateFor(64, 36, 60), 0.1), available: 1e10,
    });
    // While it was written: its lock held, its manifest "writing", its payload empty (writes wait in the swap).
    assert.deepEqual(during.held, [filmLockName(id)]);
    assert.equal(during.tree[id]['film.mp4'], 0, 'the payload was committed before the film was finalized');
    assert.ok(during.tree[id]['film.json'] > 0);
    assert.deepEqual([...locks.held], [], 'the lock outlived the film');
    const m = await manifest(opfs, id);
    assert.deepEqual([m.kind, m.status, m.exportId, m.name, m.container, m.mimeType, m.payload, m.frames, m.bytes],
        ['null-graph-film', 'completed', id, 'film.mp4', 'mp4', 'video/mp4', 'film.mp4', 6, result.file.size]);
    assert.equal(m.report.source, 'canvas', 'the manifest\'s report lacks the runner\'s side');
    assert.equal(m.report.exportId, id);
    // The muxer streamed into it: chunked, with MP4's index reserved at the front for J packets.
    const [output] = run.mb.outputs;
    assert.deepEqual([output.target.options, output.format.options.fastStart, output.tracks[0].options.maximumPacketCount], [{ chunked: true }, 'reserve', 6]);
    const listed = await listFilms();
    assert.deepEqual(listed.map(f => [f.exportId, f.status, f.live, f.frames]), [[id, 'completed', false, 6]]);
    assert.deepEqual(await frames(await listed[0].getFile()), await frames(result.file));
    await result.dispose();
    await result.dispose();
    assert.deepEqual(await films(opfs), {}, 'dispose left the film in OPFS');
    // Asked for OPFS: the same, said so.
    const asked = await film({ output: { target: 'opfs' } });
    assert.equal(asked.result.report.output.targetReason, 'asked for');
    await asked.result.dispose();
});

f3('B9: OPFS without room: asked for, refused with both numbers; through auto, memory within its budget, or both reasons', async ({ estimate, opfs }) => {
    estimate.quota = 100;
    const asked = await film({ output: { target: 'opfs' } });
    assert.equal(asked.error?.message, 'film: not enough storage in OPFS — 1 MB needed, 0 MB available.');
    assert.equal(asked.report, null);
    const auto = await film();
    assert.equal(auto.result?.status, 'completed', String(auto.error));
    assert.deepEqual([auto.result.target, auto.result.exportId], ['memory', null]);
    assert.equal(auto.result.report.output.targetReason, 'memory, within its 268.4 MB budget, because not enough storage in OPFS — 1 MB needed, 0 MB available');
    assert.deepEqual([auto.result.report.storage], [null]);
    const over = await film({ output: { memoryBudgetBytes: 100 } });
    assert.equal(over.error?.message, 'film: not enough storage in OPFS — 1 MB needed, 0 MB available; and in memory the film\'s predicted 0.0 MB is over its budget of 0.0 MB ' +
        '(raise output.memoryBudgetBytes, or shorten the film).');
    assert.deepEqual(await films(opfs), {}, 'a refused film wrote to OPFS');
});

f3('B9: OPFS that refuses to be written: asked for, refused in its own words; through auto, memory, saying why', async ({ opfs }) => {
    opfs.refuse('film.mp4', 'SecurityError');
    const asked = await film({ output: { target: 'opfs' } });
    assert.equal(asked.error?.message, 'film: OPFS could not be written here (SecurityError: The stand-in OPFS refused a writable for "film.mp4").');
    const auto = await film();
    assert.equal(auto.result?.target, 'memory');
    assert.match(auto.result.report.output.targetReason, /because OPFS could not be written here \(SecurityError: /);
    assert.deepEqual(await films(opfs), {}, 'the half-opened export was not undone');
});

// --- What a discard leaves: nothing --------------------------------------------------------------------

f3('B9: a film cancelled, failed or abandoned leaves nothing in OPFS, and touches no other film', async ({ opfs, locks }) => {
    const kept = await film();
    const keptId = kept.result.exportId;
    const commits = () => opfs.fs.writes.get('film.mp4') ?? 0;
    const before = commits();
    // Cancelled at frame 3: mediabunny's cancel closes its writer (1.61.3), which the gate turns into an abort.
    const controller = new AbortController();
    const cancelled = await film({ signal: controller.signal, onProgress: p => { if (p.phase === 'recording' && p.frame === 3) controller.abort(); } });
    assert.equal(cancelled.error?.name, 'AbortError');
    assert.equal(commits(), before, 'the cancelled film\'s payload was committed');
    // Failed at frame 2's draw, and given up after its device was lost with the encoder still holding a frame.
    const failed = await film({ workload: { fail: 'draw:7' } });
    assert.match(failed.error?.message, /failed at draw:7$/);
    const device = filmDevice();
    const abandoned = await film({ device, mb: { hangAddAt: 3, onHang: () => device.lose('unknown', 'Gone.') } });
    assert.match(abandoned.error?.message, /^film: GPU device lost at frame 3/);
    assert.equal(commits(), before);
    assert.deepEqual(Object.keys(await films(opfs)), [keptId], 'a discarded film left its folder, or another film\'s was touched');
    assert.equal((await manifest(opfs, keptId)).status, 'completed');
    assert.deepEqual([...locks.held], []);
    for (const run of [cancelled, failed, abandoned]) assert.equal(run.report.storage.committed, false);
});

f3('B9: "keep-if-finalizable" commits a finalized partial, marked partial with its frames; without it, nothing is kept', async ({ opfs }) => {
    const kept = await film({ partial: 'keep-if-finalizable', workload: { fail: 'draw:8' } });
    assert.equal(kept.result?.status, 'partial', String(kept.error));
    assert.deepEqual((await frames(kept.result.file)).map(f => f[2]), [1, 2, 3]);
    const m = await manifest(opfs, kept.result.exportId);
    assert.deepEqual([m.status, m.frames], ['partial', 3]);
    assert.match(m.report.failure, /failed at draw:8$/);
    const lost = await film({ workload: { fail: 'draw:8' } });
    assert.equal(lost.result, null);
    assert.deepEqual(Object.keys(await films(opfs)), [kept.result.exportId]);
});

f3('B9: OPFS running out of room mid-film ends it in plain words and keeps nothing', async ({ opfs }) => {
    // Room for the manifest and two packets; the preflight estimate says there is plenty (it is no reservation).
    opfs.fs.quotaBytes = 3000;
    const run = await film({ partial: 'keep-if-finalizable' });
    assert.match(run.error?.message, /^film: OPFS ran out of room at frame \d \(QuotaExceededError: The operation failed because it would cause the application to exceed its storage quota\); free some storage, or film shorter or at a lower bitrate\.$/);
    assert.equal(run.result, null, 'a film whose writes failed was kept');
    assert.deepEqual(await films(opfs), {});
    // The refused write errored the stream, so mediabunny's cancel cannot close it (1.61.3): that is the write's
    // error again, not a cleanup that failed.
    assert.deepEqual(run.report.cleanupErrors, [], 'the refused write was reported a second time, as a cleanup error');
});

f3('B9: a muxer that fails at finalize commits nothing: a picked file keeps what it held, and OPFS keeps nothing', async ({ opfs }) => {
    // mediabunny 1.61.3's finalize closes its target in a finally when it fails (an encoder failing at its flush).
    const old = await opfs.root.getFileHandle('old.mp4', { create: true });
    await old.createWritable().then(async w => { await w.write('what was here'); await w.close(); });
    const picked = await film({ output: { target: { file: old } }, mb: { failFinalize: true } });
    assert.equal(picked.error?.message, 'film: video encoder failed at frame 6: The stand-in muxer could not finalize.');
    assert.equal(await (await old.getFile()).text(), 'what was here', 'a film that failed at finalize overwrote the user\'s file');
    assert.equal(picked.report.storage.committed, false, 'the report calls a failed film\'s file committed');
    const kept = await film({ mb: { failFinalize: true }, partial: 'keep-if-finalizable' });
    assert.equal(kept.error?.message, 'film: video encoder failed at frame 6: The stand-in muxer could not finalize.');
    assert.equal(kept.report.storage.committed, false);
    assert.deepEqual(await films(opfs), {}, 'a film that failed at finalize left its folder');
});

f3('B9: an estimate that fails is no estimate: the film goes to OPFS all the same, its free bytes unknown', async () => {
    globalThis.navigator.storage.estimate = async () => { throw new DOMException('The stand-in estimate failed.', 'UnknownError'); };
    const run = await film();
    assert.equal(run.result?.target, 'opfs', run.result?.report.output.targetReason ?? String(run.error));
    assert.deepEqual([run.result.report.storage.needed, run.result.report.storage.available], [predictBytes(bitrateFor(64, 36, 60), 0.1), null]);
    await run.result.dispose();
});

f3('B9: the storage pieces alone: a gate commits only when told, after the muxer closed it, and aborts once discarded; commit marks the manifest, or removes an unmarkable film', async ({ opfs, locks }) => {
    const open = id => openOpfsExport({ exportId: id, name: `${id}.mp4`, container: 'mp4', mimeType: 'video/mp4', payload: 'video', extension: 'mp4' });
    // Committed: what went through the gate is the file, and the manifest says completed.
    const done = await open('done');
    assert.deepEqual([done.kind, done.locked, done.path], ['opfs', true, `${FILMS_DIRECTORY}/done/film.mp4`]);
    assert.deepEqual([...locks.held], [filmLockName('done')]);
    const writer = done.gate.stream.getWriter();
    await writer.write({ type: 'write', position: 0, data: new TextEncoder().encode('a film') });
    await assert.rejects(done.gate.commit(), /the muxer has not closed the film's file; it cannot be committed before the film is finalized/);
    // mediabunny closes its stream on a failed finalize too (1.61.3): the close alone commits nothing.
    await writer.close();
    assert.equal(done.gate.committed, false, 'the muxer\'s close committed the file');
    await assert.rejects(done.commit('completed', { frames: 1, bytes: 0, report: null }), /the film's payload is not committed; it cannot be marked complete/);
    await done.gate.commit();
    await done.gate.commit();
    const file = await done.commit('completed', { frames: 1, bytes: 0, report: { ok: true } });
    assert.deepEqual([await file.text(), done.committed, done.gate.bytes], ['a film', true, 6]);
    assert.deepEqual([(await manifest(opfs, 'done')).status, (await manifest(opfs, 'done')).bytes], ['completed', 6]);
    done.release();
    await sleep(0); // a lock goes once its holder's promise settles, as Web Locks let go
    assert.deepEqual([...locks.held], []);
    await done.discard(); // a committed film is not discarded
    assert.ok((await films(opfs)).done);
    await done.dispose();
    assert.equal((await films(opfs)).done, undefined);
    // Discarded: the muxer's close becomes an abort, the file can no longer be committed, and nothing of the folder is left.
    const dropped = await open('dropped');
    dropped.gate.discard();
    const w2 = dropped.gate.stream.getWriter();
    await w2.write({ type: 'write', position: 0, data: new Uint8Array(10) });
    await w2.close();
    await assert.rejects(dropped.gate.commit(), /the film's file was discarded; it cannot be committed/);
    assert.equal(dropped.gate.committed, false);
    await dropped.discard();
    dropped.release();
    assert.deepEqual(await films(opfs), {});
    // A manifest that cannot be marked: the film is removed by commit itself, whoever called it.
    opfs.failWrite(MANIFEST, opfs.fs.writes.get(MANIFEST) + 2);
    const unmarked = await open('unmarked');
    const w3 = unmarked.gate.stream.getWriter();
    await w3.close();
    await unmarked.gate.commit();
    await assert.rejects(unmarked.commit('completed', { frames: 0, bytes: 0, report: null }), /could not be marked complete in OPFS .*; it was removed\.$/);
    assert.deepEqual(await films(opfs), {}, 'commit left an unmarked film');
    unmarked.release();
});

f3('B9: opening a reused export id never overwrites or removes an existing film', async ({ opfs, locks }) => {
    const kept = (await film()).result;
    const before = await films(opfs);
    await assert.rejects(openOpfsExport({
        exportId: kept.exportId, name: 'replacement.mp4', container: 'mp4', mimeType: 'video/mp4', payload: 'video', extension: 'mp4',
    }), /already exists in OPFS; use a fresh export id/);
    assert.deepEqual(await films(opfs), before, 'opening a reused id changed the retained film');
    assert.deepEqual([...locks.held], [], 'the refused open kept its lock');
    await kept.dispose();
});

f3('B9: a film whose manifest cannot be marked complete is removed, and says so', async ({ opfs }) => {
    opfs.failWrite(MANIFEST, 2); // the first write is "writing"; the second, at commit, fails
    const run = await film();
    assert.equal(run.error?.message, 'film: the film was written but could not be marked complete in OPFS (InvalidStateError: The stand-in OPFS failed to commit "film.json"); it was removed.');
    assert.deepEqual(await films(opfs), {});
});

// --- The films an app can list, clean up and remove ------------------------------------------------------

/** A folder in OPFS as a writer left it: a manifest saying `status` (none when null). */
async function leftover(opfs, exportId, status) {
    const dir = await (await opfs.root.getDirectoryHandle(FILMS_DIRECTORY, { create: true })).getDirectoryHandle(exportId, { create: true });
    if (status) {
        const writable = await (await dir.getFileHandle(MANIFEST, { create: true })).createWritable();
        await writable.write(JSON.stringify({ kind: 'null-graph-film', version: 1, exportId, status, name: `${exportId}.mp4`, payload: 'film.mp4', created: '2026-10-25T00:00:00.000Z' }));
        await writable.close();
    }
    await (await dir.getFileHandle('film.mp4', { create: true })).createWritable().then(w => w.close());
}

f3('B9: listFilms names every film; cleanupFilms removes only the incomplete ones no writer holds; removeFilm the app\'s choice', async ({ opfs, locks }) => {
    const completed = (await film()).result.exportId;
    const partial = (await film({ partial: 'keep-if-finalizable', workload: { fail: 'draw:8' } })).result.exportId;
    await leftover(opfs, 'orphan', 'writing'); // its tab closed: no lock
    await leftover(opfs, 'live', 'writing'); // written in another tab now
    const releaseLive = locks.holdElsewhere(filmLockName('live'));
    await leftover(opfs, 'nameless', null); // cut off before its manifest
    const listed = Object.fromEntries((await listFilms()).map(f => [f.exportId, [f.status, f.live]]));
    assert.deepEqual(listed, {
        [completed]: ['completed', false], [partial]: ['partial', false], orphan: ['writing', false], live: ['writing', true], nameless: ['unknown', false],
    });
    const cleaned = await cleanupFilms();
    assert.deepEqual([...cleaned.removed].sort(), ['nameless', 'orphan']);
    assert.deepEqual(Object.fromEntries(cleaned.kept.map(k => [k.exportId, k.reason])), {
        [completed]: 'completed: remove it with removeFilm', [partial]: 'partial: remove it with removeFilm', live: 'being written now',
    });
    await assert.rejects(removeFilm('live'), /^Error: film: the film "live" is being written now \(its lock is held\); stop or cancel that film first\.$/);
    releaseLive(); // its tab closes
    assert.deepEqual((await cleanupFilms()).removed, ['live']);
    await removeFilm(completed);
    await assert.rejects(removeFilm(completed), new RegExp(`there is no film "${completed}" in OPFS`));
    await assert.rejects(removeFilm('../x'), /removeFilm needs an export id/);
    assert.deepEqual(Object.keys(await films(opfs)), [partial]);
});

f3('B9: cleanupFilms never removes a film its writer marked complete after cleanup listed it', async ({ opfs, locks }) => {
    await leftover(opfs, 'finishing', 'writing');
    const releaseFinishing = locks.holdElsewhere(filmLockName('finishing'));
    // Its writer, in another tab, marks it complete and lets its lock go just as cleanup comes to it.
    const request = locks.request.bind(locks);
    locks.request = async (name, options, callback) => {
        if (name === filmLockName('finishing')) {
            await leftover(opfs, 'finishing', 'completed');
            releaseFinishing();
        }
        return request(name, options, callback);
    };
    const cleaned = await cleanupFilms();
    assert.deepEqual(cleaned.removed, [], 'cleanup removed a film completed after it was listed');
    assert.deepEqual(cleaned.kept, [{ exportId: 'finishing', reason: 'completed: remove it with removeFilm' }]);
    assert.deepEqual(Object.keys(await films(opfs)), ['finishing']);
});

f3('B9: without Web Locks a live film cannot be told from an abandoned one: cleanupFilms removes nothing, and says why', async ({ opfs }) => {
    await leftover(opfs, 'orphan', 'writing');
    assert.deepEqual((await listFilms()).map(f => [f.exportId, f.live]), [['orphan', null]]);
    await assert.rejects(cleanupFilms(), /^Error: film: Web Locks \(navigator\.locks\) are not available here, so a film being written in another tab cannot be told from an abandoned one; nothing was removed\.$/);
    assert.deepEqual(Object.keys(await films(opfs)), ['orphan']);
    const run = await film();
    assert.deepEqual([run.result?.target, run.result.report.storage.locked], ['opfs', false], 'a film without Web Locks is written all the same, unlocked');
}, { locks: null });

// --- A file the user picked ----------------------------------------------------------------------------

f3('B9: a picked file is written through its handle, committed once finalized and never removed; a refused one is refused in plain words', async ({ opfs, locks }) => {
    const handle = await opfs.root.getFileHandle('mine.mp4', { create: true });
    const run = await film({ output: { target: { file: handle } } });
    assert.equal(run.result?.status, 'completed', String(run.error));
    assert.deepEqual([run.result.target, run.result.exportId, run.result.report.output.targetReason], ['file', null, 'asked for']);
    assert.deepEqual(run.result.report.storage, { path: null, locked: false, committed: true, needed: null, available: null });
    assert.deepEqual(await frames(await handle.getFile()), await frames(run.result.file));
    await run.result.dispose();
    assert.equal((await handle.getFile()).size, run.result.file.size, 'dispose touched the user\'s file');
    assert.deepEqual([...locks.held], []);
    // A cancel leaves the file as it was.
    const old = await opfs.root.getFileHandle('old.mp4', { create: true });
    await old.createWritable().then(async w => { await w.write('what was here'); await w.close(); });
    const controller = new AbortController();
    const cancelled = await film({ output: { target: { file: old } }, signal: controller.signal, onProgress: p => { if (p.frame === 3) controller.abort(); } });
    assert.equal(cancelled.error?.name, 'AbortError');
    assert.equal(await (await old.getFile()).text(), 'what was here', 'a cancelled film overwrote the user\'s file');
    opfs.refuse('denied.mp4', 'NotAllowedError');
    const denied = await film({ output: { target: { file: await opfs.root.getFileHandle('denied.mp4', { create: true }) } } });
    assert.equal(denied.error?.message, 'film: the chosen file "denied.mp4" may not be written (NotAllowedError: The stand-in OPFS refused a writable for "denied.mp4"); ' +
        'ask for write permission in a user gesture (handle.requestPermission({ mode: \'readwrite\' })) and try again.');
});

// --- PNG sequences -------------------------------------------------------------------------------------

f3('B9: a PNG sequence writes each frame\'s exact bytes as a file of its own, read back, and is marked complete', async ({ opfs }) => {
    const run = await film({ output: { container: 'png' } });
    assert.equal(run.result?.status, 'completed', String(run.error));
    const { result } = run;
    assert.deepEqual([result.file, result.codec, result.target, result.report.source, result.report.sourceReason], [null, 'png', 'opfs', 'readback', 'PNG frames need exact bytes']);
    assert.deepEqual(result.sequence.names, Array.from({ length: 6 }, (_, j) => frameFileName(j)));
    let bytes = 0;
    for (const [j, name] of result.sequence.names.entries()) {
        const file = await (await result.sequence.directory.getFileHandle(name)).getFile();
        bytes += file.size;
        const png = readPNG(new Uint8Array(await file.arrayBuffer()));
        assert.deepEqual([png.width, png.height], [64, 36]);
        assert.deepEqual(png.rgba, pixels(j + 1, 64, 36), `${name} is not frame ${j}'s exact bytes`);
    }
    assert.deepEqual([result.report.output.bitrate, result.report.achievedBitrate, result.report.frames.encoded, result.report.bytes], [null, null, 6, bytes]);
    const m = await manifest(opfs, result.exportId);
    assert.deepEqual([m.status, m.container, m.mimeType, m.payload, m.frames, m.bytes, m.name], ['completed', 'png', 'image/png', 'frames', 6, bytes, 'film-frames']);
    assert.equal(run.mb.outputs.length, 0, 'a PNG sequence made a muxer');
    assert.ok(FakeVideoFrame.made.every(f => f.closes === 1), 'a frame was not closed exactly once');
    const listed = await listFilms();
    assert.equal(await listed[0].getFile(), null);
    assert.deepEqual((await (await listed[0].getFrames()).getFileHandle(frameFileName(5))).kind, 'file');
    await result.dispose();
    assert.deepEqual(await films(opfs), {});
    // Cancelled, nothing is kept; a partial keeps the frames made, marked partial.
    const controller = new AbortController();
    const cancelled = await film({ output: { container: 'png' }, signal: controller.signal, onProgress: p => { if (p.frame === 3) controller.abort(); } });
    assert.equal(cancelled.error?.name, 'AbortError');
    assert.deepEqual(await films(opfs), {});
    const partial = await film({ output: { container: 'png' }, partial: 'keep-if-finalizable', workload: { fail: 'draw:8' } });
    assert.deepEqual([partial.result?.status, partial.result.sequence.names.length], ['partial', 2], 'readback keeps one frame in flight: the partial ends a frame early');
    assert.equal((await manifest(opfs, partial.result.exportId)).status, 'partial');
});

f3('B9: a PNG sequence whose frame file is refused ends in §8\'s words at that frame, and keeps nothing', async ({ opfs }) => {
    opfs.refuse(frameFileName(2), 'QuotaExceededError');
    const run = await film({ output: { container: 'png' }, partial: 'keep-if-finalizable' });
    assert.equal(run.error?.message, 'film: OPFS ran out of room at frame 2 (QuotaExceededError: The stand-in OPFS refused a writable for "frame-000002.png"); ' +
        'free some storage, or film shorter or at a lower bitrate.');
    assert.deepEqual(await films(opfs), {});
});

f3('B9: an app-driven session films to OPFS too, and its result keeps neither the session nor its sink alive', async ({ opfs }) => {
    v8.setFlagsFromString('--expose-gc');
    const collect = vm.runInNewContext('gc');
    const mb = fakeMediabunny();
    let ref = null;
    const result = await (async () => {
        const session = await createFilmSession({ film: { ...SMALL, seconds: 2 / 60 }, mediabunny: mb });
        ref = new WeakRef(session);
        for (let j = 0; j < 2; j++) {
            await session.add(new FakeVideoFrame(pixels(j, 64, 36), { format: 'RGBX', codedWidth: 64, codedHeight: 36, timestamp: frameStamp(j, 60).timestampUs }));
        }
        return session.finish();
    })();
    assert.deepEqual([result.status, result.target, result.frames], ['completed', 'opfs', 2]);
    for (let i = 0; i < 3; i++) { await sleep(5); collect(); }
    assert.equal(ref.deref(), undefined, 'the result kept its session, and with it the sink and the muxer, alive');
    assert.equal((await manifest(opfs, result.exportId)).status, 'completed');
});

f3('B9: the probe\'s operational check writes a file to OPFS under a name of its own, its lock held, and removes it', async ({ opfs }) => {
    const locks = fakeLocks();
    const scope = { navigator: { gpu: {}, storage: globalThis.navigator.storage, locks }, VideoFrame: class {}, CompressionStream, crypto: { randomUUID: () => 'probe-test' } };
    const untested = await probeFilm({ width: 64, height: 36, fps: 60, container: 'png', scope });
    assert.deepEqual([untested.ok, untested.targets.opfs, untested.targets.memory, untested.codecs], [true, { api: true, writable: 'untested' }, false, ['png']]);
    assert.deepEqual(await opfs.tree(), {}, 'the cheap probe wrote to OPFS');
    // A writer under null-graph-films/ like any other: cleanupFilms in another tab must not take its folder mid-write.
    let held = null;
    opfs.fs.onWritable = name => { if (name === 'probe.bin') held = [...locks.held]; };
    const tested = await probeFilm({ width: 64, height: 36, fps: 60, container: 'png', scope, operational: true });
    assert.deepEqual([tested.ok, tested.targets.opfs.writable, tested.targets.locks], [true, true, true]);
    assert.deepEqual(held, [filmLockName('.probe-probe-test')], 'the probe wrote without the lock every writer holds');
    assert.deepEqual([...locks.held], [], 'the probe kept its lock');
    assert.deepEqual(await films(opfs), {}, 'the probe left its test folder');
    opfs.refuse('probe.bin', 'NotAllowedError');
    const refused = await probeFilm({ width: 64, height: 36, fps: 60, container: 'png', scope, operational: true });
    assert.deepEqual([refused.ok, refused.targets.opfs.writable], [false, false]);
    assert.ok(refused.reasons.some(r => r.startsWith('film: OPFS refused a test write here (NotAllowedError: ')), refused.reasons.join(' | '));
    assert.deepEqual(await films(opfs), {});
});
