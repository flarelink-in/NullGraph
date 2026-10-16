// D2.1 (docs/offline-films-design.md §7): where a film's frames go. ffmpeg (one process per segment, raw
// RGBA on its stdin), PNG files, or nowhere. Node only, except encoderArgs, which is pure.
import { spawn, spawnSync } from 'node:child_process';
import { once } from 'node:events';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { encodePNG } from '../../test/gpu/png.mjs';

/**
 * The colour tags of an HD film. ffmpeg 9 tags the stream from its frames, so they are set in the filter
 * (setparams): as output flags they came out "unknown" (D2.0; docs/work/gotchas.md).
 */
export const BT709 = 'scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv';

/** ffmpeg's arguments for one segment: raw RGBA frames of `film`'s size on stdin, `encode`'s encoder, to `file`. Pure. */
export function encoderArgs(encode, film, file) {
    const quality = encode.codec === 'libx264'
        ? ['-preset', encode.preset, '-crf', String(encode.crf), ...(encode.tune ? ['-tune', encode.tune] : [])]
        : encode.codec === 'h264_nvenc'
            ? ['-preset', 'p5', '-tune', 'hq', '-rc', 'vbr', '-cq', String(encode.crf), '-b:v', '0']
            : [];
    return ['-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${film.width}x${film.height}`, '-r', String(film.fps), '-i', '-',
        '-c:v', encode.codec, ...quality, '-vf', BT709, '-movflags', '+faststart', file];
}

/** The ffmpeg to run: NG_FFMPEG, or `ffmpeg` on PATH. Resolves to { path, version }, or null when neither runs. */
export function findFfmpeg() {
    const path = process.env.NG_FFMPEG ?? 'ffmpeg';
    const probe = spawnSync(path, ['-hide_banner', '-version'], { encoding: 'utf8' });
    if (probe.status !== 0) return null;
    return { path, version: probe.stdout.split('\n')[0].trim() };
}

/** One segment through ffmpeg: write(bytes) per frame, then close(). A failure reports ffmpeg's own reason. */
export function ffmpegSegment(ffmpeg, encode, film, file) {
    const proc = spawn(ffmpeg.path, encoderArgs(encode, film, file), { stdio: ['pipe', 'ignore', 'pipe'] });
    let errors = '';
    proc.stderr.on('data', d => { errors += d; });
    proc.stdin.on('error', () => {});                  // an ffmpeg that quit is reported by write() and close()
    const exited = once(proc, 'close');
    const reason = () => errors.trim().split('\n').filter(l => !/Terminating thread|Task finished/.test(l)).slice(0, 3).join(' / ');
    const quit = async () => { await exited; throw new Error(`film: ffmpeg quit while encoding ${file}: ${reason() || `exit ${proc.exitCode}`}`); };
    return {
        async write(bytes) {
            if (proc.exitCode !== null || proc.stdin.destroyed) return quit();
            try {
                if (!proc.stdin.write(bytes)) await Promise.race([once(proc.stdin, 'drain'), exited]);
            } catch {
                return quit();
            }
        },
        async close() {
            proc.stdin.end();
            const [code] = await exited;
            if (code !== 0) throw new Error(`film: ffmpeg exited ${code} on ${file}: ${reason()}`);
        },
    };
}

/** Joins the segments, in order, without encoding again. */
export async function concatSegments(ffmpeg, files, out) {
    const list = `${out}.segments.txt`;
    writeFileSync(list, files.map(f => `file '${f.replace(/\\/g, '/').replace(/'/g, "'\\''")}'`).join('\n') + '\n');
    const proc = spawn(ffmpeg.path, ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', list,
        '-c', 'copy', '-movflags', '+faststart', out], { stdio: ['ignore', 'ignore', 'pipe'] });
    let errors = '';
    proc.stderr.on('data', d => { errors += d; });
    const [code] = await once(proc, 'close');
    if (code !== 0) throw new Error(`film: ffmpeg could not join the segments into ${out}: ${errors.trim()}`);
}

/** Each frame as a PNG in `dir`, named by its frame number. */
export function pngSink(dir, film) {
    mkdirSync(dir, { recursive: true });
    return {
        async write(bytes, frame) { writeFileSync(join(dir, `${String(frame).padStart(6, '0')}.png`), encodePNG(bytes, film.width, film.height)); },
        async close() {},
    };
}

/** No frames kept: a run that only measures, or (from D2.2) only records tracks. */
export const noSink = () => ({ async write() {}, async close() {} });
