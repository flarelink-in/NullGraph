// F3's browser G tests (docs/offline-films-browser-design.md §10, §14 F3's exit): films written to OPFS and to a
// picked file, through recordFilm, as an app writes them.
//
//   G2-png      a PNG sequence of the pattern in OPFS: each frame's file decodes to its exact bytes.
//   G4          Kerr at 4K for 10 s streamed to OPFS: the payload stays empty until committed, the page's JS memory
//               stays flat while the film grows, and the file plays (moov first, its index reserved).
//   G4-long     the same for 60 s (F3's exit film); in the default run only with run.mjs --long.
//   G5-storage  a film cancelled, failed in its workload, or failed at its muxer's finalize leaves nothing in OPFS,
//               no swap file either, and touches no other film.
//   G7          two tabs (G7-peer is the other): a live writer's film survives cleanup and removal, two films of
//               one name stay apart, completed films survive later films, and a film whose tab closed mid-film
//               is an orphan that cleanup removes.
//   G8          a picked file refused at write time, one written, and one over what it held whose encoder fails at its
//               flush inside mediabunny's finalize; OPFS running out of room mid-film; the encoder failing mid-film
//               and the muxer at finalize; a lost device with and without a kept partial.

import { cleanupFilms, listFilms, removeFilm } from 'null-graph/film';
import { describe, film, keep, setup, summary } from './cases.js';
import { kerrWorkload, orbit, patternWorkload, sleep } from './workloads.js';

const ms = t => Math.round(t * 10) / 10;
const MiB = bytes => (bytes == null ? null : Math.round(bytes / 1048576 * 10) / 10);

/** The films folder's tree in OPFS: { exportId: [entries] }, swap files included. */
async function filmsTree() {
    const root = await navigator.storage.getDirectory();
    let films;
    try { films = await root.getDirectoryHandle('null-graph-films'); } catch { return {}; }
    const out = {};
    for await (const [name, handle] of films.entries()) {
        const entries = [];
        if (handle.kind === 'directory') for await (const [child] of handle.entries()) entries.push(child);
        out[name] = entries.sort();
    }
    return out;
}

/** The page's JS memory: measureUserAgentSpecificMemory where the page is cross-origin isolated, else the V8 heap. */
async function jsMemory() {
    if (crossOriginIsolated && performance.measureUserAgentSpecificMemory) {
        try { return { kind: 'ua', bytes: (await performance.measureUserAgentSpecificMemory()).bytes }; } catch { /* fall through */ }
    }
    return { kind: 'heap', bytes: performance.memory?.usedJSHeapSize ?? null };
}

// ---------------------------------------------------------------------------------------------------
// G2-png: a PNG sequence
// ---------------------------------------------------------------------------------------------------

async function g2Png(env) {
    const s = await setup(env);
    const W = 640, H = 360, fps = 30, J = 30;
    const work = patternWorkload({ dt: 1 / fps });
    const run = await film({ device: s.device, createWorkload: work.create, film: { width: W, height: H, fps, seconds: J / fps }, output: { container: 'png' } });
    env.check('a PNG sequence of the pattern films to OPFS', run.result?.status === 'completed' && run.result.target === 'opfs', describe(run));
    env.check('from the readback source: PNG frames need exact bytes', run.report?.source === 'readback' && run.report.sourceReason === 'PNG frames need exact bytes', run.report?.sourceReason);
    env.check(`${J} files, frame-000000.png first`, run.result?.sequence?.names.length === J && run.result.sequence.names[0] === 'frame-000000.png', run.result?.sequence?.names.slice(0, 2));
    const [listed] = await listFilms();
    env.check('listed: completed, a PNG sequence of J frames', listed?.status === 'completed' && listed.container === 'png' && listed.frames === J && !listed.live,
        listed && { status: listed.status, container: listed.container, frames: listed.frames });
    const pngs = [];
    for (const j of [0, 15, J - 1]) {
        const name = `G2-png-frame-${j}.png`;
        await env.upload(name, await (await run.result.sequence.directory.getFileHandle(run.result.sequence.names[j])).getFile());
        pngs.push({ file: name, width: W, height: H, frame: j });
    }
    await run.result.dispose();
    await s.settled('PNG sequence', work.log.engines);
    env.check('no uncaptured WebGPU error', s.uncaptured.length === 0, s.uncaptured);
    return { pngs, reports: [summary(run)] };
}

// ---------------------------------------------------------------------------------------------------
// G4: Kerr at 4K streamed to OPFS
// ---------------------------------------------------------------------------------------------------

async function opfsFilm(env, seconds) {
    const s = await setup(env);
    const work = kerrWorkload();
    const frames = 60 * seconds;
    const samples = [];
    const pending = [];
    let mid = null;
    const base = await jsMemory();
    const run = await film({
        device: s.device, createWorkload: work.create, camera: orbit, film: { width: 3840, height: 2160, fps: 60, seconds }, output: { target: 'opfs', name: 'kerr-4k.mp4' },
        onProgress: p => {
            if (p.phase !== 'recording') return;
            // JS memory each second of film, and once, mid-film, what OPFS holds: the payload empty, its writes in the swap.
            if (p.frame % 60 === 0) pending.push(jsMemory().then(m => samples.push({ frame: p.frame, bytes: m.bytes, kind: m.kind })));
            if (p.frame === frames / 2 && !mid) {
                mid = (async () => {
                    const [live] = (await listFilms()).filter(f => f.live);
                    return { live: !!live, status: live?.status, payloadBytes: (await live?.getFile())?.size ?? null, tree: await filmsTree() };
                })();
            }
        },
    });
    await Promise.all(pending);
    const during = await mid;
    env.check(`Kerr at 4K for ${seconds} s films to OPFS`, run.result?.status === 'completed' && run.result.target === 'opfs', describe(run));
    env.check('mid-film: the film is live, "writing", and its payload empty until committed', during?.live && during.status === 'writing' && during.payloadBytes === 0, during);
    env.check('mid-film: its writes wait in a swap file beside the payload', Object.values(during?.tree ?? {}).some(entries => entries.includes('film.mp4.crswap')), during?.tree);
    env.check('committed: the report says so, and the lock was held', run.report?.storage?.committed === true && run.report.storage.locked === true, run.report?.storage);
    const sorted = samples.sort((a, b) => a.frame - b.frame);
    const third = Math.max(1, Math.floor(sorted.length / 3));
    const mean = list => list.reduce((n, x) => n + x.bytes, 0) / list.length;
    const memory = {
        kind: base.kind, baseMiB: MiB(base.bytes), peakMiB: MiB(Math.max(...sorted.map(x => x.bytes))),
        firstThirdMiB: MiB(mean(sorted.slice(0, third))), lastThirdMiB: MiB(mean(sorted.slice(-third))), samples: sorted.length,
        fileMiB: MiB(run.result?.file?.size),
    };
    // Streaming, the page holds the frames in flight and two 16 MiB chunks, not the film: flat as it grows.
    env.check('the page\'s JS memory stays flat as the film grows (last third within 64 MiB of the first)', memory.lastThirdMiB - memory.firstThirdMiB < 64, memory);
    const video = await keep(env, `G4-kerr-4k-${seconds}s`, run, { id: `G4-kerr-4k-${seconds}s`, width: 3840, height: 2160, fps: 60, frames, container: 'mp4' });
    await s.settled(`Kerr 4K ${seconds} s`, work.log.engines);
    env.check('no uncaptured WebGPU error', s.uncaptured.length === 0, s.uncaptured);
    return { videos: video ? [video] : [], reports: [summary(run)], memory, during };
}

const g4 = env => opfsFilm(env, 10);
const g4Long = env => opfsFilm(env, 60);

// ---------------------------------------------------------------------------------------------------
// G5-storage: a discard leaves nothing, and touches no other film
// ---------------------------------------------------------------------------------------------------

async function g5Storage(env) {
    const s = await setup(env);
    const size = { width: 640, height: 360, fps: 30, seconds: 2 };
    const kept = await film({ device: s.device, createWorkload: patternWorkload({ dt: 1 / 30 }).create, film: size, output: { target: 'opfs', name: 'kept.mp4' } });
    const keptId = kept.result?.exportId;
    const keptSize = kept.result?.file?.size;
    env.check('a film to keep is made first', kept.result?.status === 'completed' && !!keptId, describe(kept));
    const mb = await import('mediabunny');
    class FailingOutput extends mb.Output {
        async finalize() { throw new Error('the muxer failed to finalize (injected)'); }
    }
    const outcomes = {};
    for (const [key, options] of [
        ['cancelled', controller => ({ signal: controller.signal, onProgress: p => { if (p.phase === 'recording' && p.frame === 20) controller.abort(); } })],
        ['failed in its workload', () => ({ createWorkload: patternWorkload({ dt: 1 / 30, stepMs: n => { if (n === 30) throw new Error('the workload failed (injected)'); return 0; } }).create })],
        ['failed at finalize', () => ({ mediabunny: { ...mb, Output: FailingOutput } })],
    ]) {
        const controller = new AbortController();
        const run = await film({ device: s.device, createWorkload: patternWorkload({ dt: 1 / 30 }).create, film: size, output: { target: 'opfs', name: `${key}.mp4` }, ...options(controller) });
        const tree = await filmsTree();
        outcomes[key] = { outcome: describe(run), exportId: run.report?.exportId, committed: run.report?.storage?.committed, tree };
        env.check(`${key}: nothing kept`, !run.result && run.report?.storage?.committed === false, outcomes[key]);
        env.check(`${key}: its folder is gone, swap file and all, and only the kept film remains`,
            JSON.stringify(Object.keys(tree)) === JSON.stringify([keptId]) && JSON.stringify(tree[keptId]) === JSON.stringify(['film.json', 'film.mp4']), tree);
        void controller.signal.reason?.stack;
    }
    const after = (await listFilms()).find(f => f.exportId === keptId);
    env.check('the kept film is as it was: completed, its file whole', after?.status === 'completed' && (await after.getFile())?.size === keptSize, after && { status: after.status });
    await kept.result?.dispose();
    await s.settled('discards');
    env.check('no uncaptured WebGPU error', s.uncaptured.length === 0, s.uncaptured);
    return { outcomes };
}

// ---------------------------------------------------------------------------------------------------
// G7: two tabs
// ---------------------------------------------------------------------------------------------------

const G7_CHANNEL = 'null-graph-g7';

/** G7's other tab: films what the main tab asks for, says when it is writing and when it is done. */
async function g7Peer(env) {
    const s = await setup(env);
    const channel = new BroadcastChannel(G7_CHANNEL);
    const post = message => channel.postMessage({ to: 'main', ...message });
    channel.onmessage = async ({ data }) => {
        if (data?.to !== 'peer' || data.do !== 'film') return;
        const run = await film({
            device: s.device, createWorkload: kerrWorkload({ count: 65536 }).create, camera: orbit,
            film: { width: 640, height: 360, fps: 60, seconds: data.seconds }, output: { target: 'opfs', name: data.name },
            onProgress: p => { if (p.phase === 'recording' && p.frame === 30) post({ event: 'writing', name: data.name }); },
        });
        post({ event: 'done', name: data.name, status: run.result?.status ?? null, error: run.error?.message ?? null, exportId: run.result?.exportId ?? null });
    };
    post({ event: 'ready' });
    await new Promise(() => {}); // until its tab is closed
}

async function g7(env) {
    const s = await setup(env);
    const channel = new BroadcastChannel(G7_CHANNEL);
    const inbox = [];
    let wake = null;
    channel.onmessage = ({ data }) => { if (data?.to === 'main') { inbox.push(data); wake?.(); } };
    /** The peer's next message of this kind (60 s at most). */
    const next = async (event, name = null) => {
        for (const deadline = performance.now() + 60000; performance.now() < deadline;) {
            const i = inbox.findIndex(m => m.event === event && (name === null || m.name === name));
            if (i >= 0) return inbox.splice(i, 1)[0];
            await Promise.race([new Promise(r => { wake = r; }), sleep(500)]);
        }
        throw new Error(`the other tab never said "${event}"`);
    };
    const tell = message => channel.postMessage({ to: 'peer', ...message });
    env.log('action open-peer G7-peer');
    await next('ready');

    // 1. The other tab films; while it writes, this one lists, cleans up, tries to remove it, and films under the same name.
    tell({ do: 'film', name: 'same-name.mp4', seconds: 8 });
    await next('writing', 'same-name.mp4');
    const live = (await listFilms()).filter(f => f.live);
    env.check('the other tab\'s film is listed live and "writing"', live.length === 1 && live[0].status === 'writing' && live[0].name === 'same-name.mp4', live);
    const peerId = live[0]?.exportId;
    const cleaned = await cleanupFilms();
    env.check('cleanup keeps a film written in another tab', cleaned.removed.length === 0 && cleaned.kept.some(k => k.exportId === peerId && k.reason === 'being written now'), cleaned);
    const refused = await removeFilm(peerId).then(() => null, e => e.message);
    env.check('removeFilm refuses a film written in another tab', /is being written now \(its lock is held\)/.test(refused ?? ''), refused);
    const mine = await film({ device: s.device, createWorkload: patternWorkload({ dt: 1 / 30 }).create, film: { width: 640, height: 360, fps: 30, seconds: 1 }, output: { target: 'opfs', name: 'same-name.mp4' } });
    env.check('this tab films under the same name meanwhile, as a film of its own', mine.result?.status === 'completed' && mine.result.exportId !== peerId, describe(mine));
    const done = await next('done', 'same-name.mp4');
    env.check('the other tab\'s film completes', done.status === 'completed' && done.exportId === peerId, done);
    const both = await listFilms();
    env.check('both are listed, completed, under one name and two ids', both.length === 2 && both.every(f => f.status === 'completed' && !f.live && f.name === 'same-name.mp4'),
        both.map(f => `${f.exportId}:${f.status}:${f.live}`));

    // 2. Completed films survive cleanup and later films; dispose is the app's, idempotent, and removes only its own.
    const later = await cleanupFilms();
    env.check('cleanup keeps completed films', later.removed.length === 0 && later.kept.length === 2, later);
    await mine.result.dispose();
    await mine.result.dispose();
    const left = await listFilms();
    env.check('dispose (twice) removed this tab\'s film alone', left.length === 1 && left[0].exportId === peerId, left.map(f => f.exportId));

    // 3. The other tab starts a film, and its tab is closed mid-film: an orphan, its lock let go with the tab.
    // 20 s of film, so that even a fast GPU is still writing it when the tab goes.
    tell({ do: 'film', name: 'orphan.mp4', seconds: 20 });
    await next('writing', 'orphan.mp4');
    env.log('action close-peer');
    let orphan = null;
    for (const deadline = performance.now() + 10000; performance.now() < deadline;) {
        orphan = (await listFilms()).find(f => f.name === 'orphan.mp4');
        if (orphan && orphan.live === false) break;
        await sleep(200);
    }
    env.check('the closed tab\'s film is left "writing", no longer live', orphan?.status === 'writing' && orphan.live === false, orphan && { status: orphan.status, live: orphan.live });
    const swept = await cleanupFilms();
    env.check('cleanup removes the orphan, and keeps the completed film', swept.removed.length === 1 && swept.removed[0] === orphan?.exportId && swept.kept.some(k => k.exportId === peerId), swept);
    await removeFilm(peerId);
    env.check('removeFilm removes a completed film the app is done with', (await listFilms()).length === 0);
    env.check('no swap file or folder is left', JSON.stringify(await filmsTree()) === '{}', await filmsTree());
    channel.close();
    await s.settled('two tabs');
    return { cleaned, later, swept };
}

// ---------------------------------------------------------------------------------------------------
// G8: failures
// ---------------------------------------------------------------------------------------------------

async function g8(env) {
    const s = await setup(env);
    const size = { width: 640, height: 360, fps: 30, seconds: 2 };
    const pattern = () => patternWorkload({ dt: 1 / 30 }).create;
    const outcomes = {};
    const videos = [];
    const root = await navigator.storage.getDirectory();
    const picked = await root.getDirectoryHandle('g8-picked', { create: true });

    // A picked file the browser refuses at write time (the app's handle, its permission revoked).
    const handle = await picked.getFileHandle('denied.mp4', { create: true });
    const denied = new Proxy(handle, {
        get(target, key) {
            if (key === 'createWritable') return () => Promise.reject(new DOMException('The request is not allowed by the user agent or the platform in the current context.', 'NotAllowedError'));
            const value = Reflect.get(target, key);
            return typeof value === 'function' ? value.bind(target) : value;
        },
    });
    const refused = await film({ device: s.device, createWorkload: pattern(), film: size, output: { target: { file: denied } } });
    outcomes.denied = describe(refused);
    env.check('a picked file refused at write time is refused in plain words', /^Error: film: the chosen file "denied\.mp4" may not be written \(NotAllowedError: .*\); ask for write permission in a user gesture/.test(outcomes.denied), outcomes.denied);
    // A picked file written: the film goes into the user's file, committed at the end, and stays after dispose.
    const mineHandle = await picked.getFileHandle('picked.mp4', { create: true });
    const written = await film({ device: s.device, createWorkload: pattern(), film: size, output: { target: { file: mineHandle } } });
    env.check('a picked file is written, committed and left to the user', written.result?.target === 'file' && written.report?.storage?.committed && (await mineHandle.getFile()).size === written.result.file.size, describe(written));
    const pickedVideo = await keep(env, 'G8-picked', written, { id: 'G8-picked', width: 640, height: 360, fps: 30, frames: 60, container: 'mp4', pattern: true });
    if (pickedVideo) videos.push(pickedVideo);
    env.check('dispose leaves the user\'s file', (await mineHandle.getFile()).size > 0);
    // An encoder failing at its flush, inside mediabunny's own finalize: 1.61.3 then closes the file's stream in its
    // finally, which on a file's writable would commit the half-made film. Over a picked file that already holds
    // something, the user's file must keep it.
    const mb = await import('mediabunny');
    class FlushFailingSource extends mb.VideoSampleSource {
        async _flushAndClose(forceClose) { // 1.61.3's internal flush, which Output.finalize awaits
            await super._flushAndClose(forceClose);
            if (!forceClose) throw new Error('EncodingError: the encoder failed at its flush (injected)');
        }
    }
    const heldHandle = await picked.getFileHandle('held.mp4', { create: true });
    await heldHandle.createWritable().then(async w => { await w.write('what was here'); await w.close(); });
    const flush = await film({ device: s.device, createWorkload: pattern(), film: size, output: { target: { file: heldHandle } }, mediabunny: { ...mb, VideoSampleSource: FlushFailingSource } });
    outcomes.flush = { outcome: describe(flush), committed: flush.report?.storage?.committed, held: await (await heldHandle.getFile()).text() };
    env.check('an encoder failing at its flush inside finalize: §8\'s words, nothing committed, and the picked file keeps what it held',
        /^Error: film: video encoder failed at frame 60: EncodingError: the encoder failed at its flush \(injected\)\.$/.test(outcomes.flush.outcome) &&
        outcomes.flush.committed === false && outcomes.flush.held === 'what was here', outcomes.flush);
    await root.removeEntry('g8-picked', { recursive: true });

    // OPFS running out of room mid-film: the origin's quota set 4 MB above what it uses (the estimate does not see it).
    const { usage } = await navigator.storage.estimate();
    env.log(`action quota ${Math.round(usage + 4e6)}`);
    await sleep(1000);
    const full = await film({ device: s.device, createWorkload: kerrWorkload({ count: 65536 }).create, camera: orbit, film: { width: 1920, height: 1080, fps: 60, seconds: 12 }, output: { target: 'opfs' }, partial: 'keep-if-finalizable' });
    env.log('action quota 0');
    await sleep(1000);
    outcomes.quota = { outcome: describe(full), tree: await filmsTree(), cleanupErrors: full.report?.cleanupErrors, bytes: full.report?.bytes };
    env.check('OPFS out of room mid-film ends the film in plain words', /^Error: film: OPFS ran out of room at frame \d+ \(QuotaExceededError: .*\); free some storage, or film shorter or at a lower bitrate\.$/.test(outcomes.quota.outcome), outcomes.quota);
    env.check('out of room: nothing kept, even with keep-if-finalizable', !full.result && JSON.stringify(outcomes.quota.tree) === '{}', outcomes.quota);
    env.check('out of room: the refused write is the film\'s one error, not repeated as a cleanup error', full.report?.cleanupErrors?.length === 0, outcomes.quota);

    // The encoder failing mid-film, and the muxer at finalize: the writes are aborted, nothing is kept.
    let added = 0;
    class FailingSource extends mb.VideoSampleSource {
        async add(sample, options) {
            if (++added === 20) throw new Error('EncodingError: the encoder failed (injected)');
            return super.add(sample, options);
        }
    }
    const encoder = await film({ device: s.device, createWorkload: pattern(), film: size, output: { target: 'opfs' }, partial: 'keep-if-finalizable', mediabunny: { ...mb, VideoSampleSource: FailingSource } });
    outcomes.encoder = { outcome: describe(encoder), tree: await filmsTree() };
    env.check('an encoder failing mid-film: §8\'s words, nothing kept', /^Error: film: video encoder failed at frame 19: EncodingError: the encoder failed \(injected\)\.$/.test(outcomes.encoder.outcome) && JSON.stringify(outcomes.encoder.tree) === '{}', outcomes.encoder);
    class FailingOutput extends mb.Output {
        async finalize() { throw new Error('the muxer failed to finalize (injected)'); }
    }
    const finalize = await film({ device: s.device, createWorkload: pattern(), film: size, output: { target: 'opfs' }, mediabunny: { ...mb, Output: FailingOutput } });
    outcomes.finalize = { outcome: describe(finalize), tree: await filmsTree() };
    env.check('a muxer failing at finalize: nothing kept', /the muxer failed to finalize \(injected\)/.test(outcomes.finalize.outcome) && JSON.stringify(outcomes.finalize.tree) === '{}', outcomes.finalize);

    // A lost device: with keep-if-finalizable a finalized partial is committed to OPFS, marked partial; without, nothing.
    for (const keepIt of [true, false]) {
        const lostSetup = await setup(env);
        const lost = await film({
            device: lostSetup.device, createWorkload: kerrWorkload({ count: 65536 }).create, camera: orbit, film: { width: 640, height: 360, fps: 60, seconds: 2 },
            output: { target: 'opfs' }, partial: keepIt ? 'keep-if-finalizable' : 'discard',
            onProgress: p => { if (p.phase === 'recording' && p.frame === 30) lostSetup.device.destroy(); },
        });
        const listed = await listFilms();
        const key = keepIt ? 'lost, kept' : 'lost, discarded';
        outcomes[key] = { outcome: describe(lost), failure: lost.report?.failure, listed: listed.map(f => `${f.status}:${f.frames}`) };
        if (keepIt) {
            env.check('a lost device with keep-if-finalizable: a partial committed to OPFS and marked partial', lost.result?.status === 'partial' && listed.length === 1 && listed[0].status === 'partial' && listed[0].frames === lost.result.frames, outcomes[key]);
            const video = await keep(env, 'G8-lost-partial', lost, { id: 'G8-lost-partial', width: 640, height: 360, fps: 60, frames: lost.result?.frames, container: 'mp4', partial: true });
            if (video) videos.push(video);
        } else {
            env.check('a lost device without it: nothing kept', !lost.result && listed.length === 0, outcomes[key]);
        }
    }
    await s.settled('failures');
    env.check('no uncaptured WebGPU error on the first device', s.uncaptured.length === 0, s.uncaptured);
    return { outcomes, videos };
}

export const STORAGE_CASES = {
    'G2-png': g2Png,
    G4: g4,
    'G4-long': g4Long,
    'G5-storage': g5Storage,
    G7: g7,
    'G7-peer': g7Peer,
    G8: g8,
};
