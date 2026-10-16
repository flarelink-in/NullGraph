// Cycle D, step D2, phase D2.1 (docs/offline-films-design.md §14): the film tool on the GPU, under Node.
//   npm run test:gpu -- --only film          NG_DAWN_FLAGS=adapter=NVIDIA or adapter=AMD picks the GPU
// Needs ffmpeg (NG_FFMPEG or PATH) and the test site (NG_SITE) for the Herd & Wolves scene's assets.
//
// F1 (P1, determinism): the Herd & Wolves scene filmed twice, as tools/film/film.mjs films it (exact steps,
//    a moving camera, each frame read back). The simulation is exact: the agents buffer after the films is
//    equal byte for byte. The frames are equal except where two animals overlap with faces at the same
//    depth: the instance cull appends visible instances with atomicAdd (visibilityShaders.ts:313), so
//    the draw order, and with it the winner of a depth tie, changes from run to run (D2.1 found it; the
//    anime chain's depth ink widens each tie by a few pixels). Under 1% of a frame's pixels may differ
//    (measured worst: 0.41%). With the flashes on, which catches flash can differ too (§10), so those
//    frames are counted.
// F5 (P4, capture fidelity): a test pattern of four colours read back exactly, then encoded by the film's
//    ffmpeg settings and decoded again: each quadrant within ±3 levels, in its place (no flip, no swizzle).
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { hash } from 'node:crypto';
import { harnessEnv } from '../gpu/harness-env.mjs';
import { readBuffer } from '../gpu/readback.mjs';
import { NullGraph } from '../../dist/index.js';
import herdWolves from '../../tools/film/scenes/herd-wolves.mjs';
import { FrameCapture } from '../../tools/film/capture.mjs';
import { cameraPath } from '../../tools/film/camera.mjs';
import { ffmpegSegment, findFfmpeg } from '../../tools/film/sinks.mjs';

const { log, check, fatal, finish } = harnessEnv('__FILM_RESULTS__');
const W = 320, H = 180;                       // 1,280-byte rows: no padding (F5 checks a padded width too)
const ffmpeg = findFfmpeg();
if (!ffmpeg) { fatal('ffmpeg not found (NG_FFMPEG, PATH)'); throw new Error('ffmpeg not found'); }

// --- F1 ------------------------------------------------------------------------------------------
const path = cameraPath([{ t: 0, target: [0, 0, 0], distance: 25, azimuth: 0.2, polar: 0.6 },
    { t: 1.5, target: [0, 0, 0], distance: 120, azimuth: 0.8, polar: 0.45 }]);
async function film(flashes, frames = 90, warmup = 30) {
    const engine = new NullGraph();
    const init = await engine.init(null, { width: W, height: H });
    if (!init.success) throw new Error(init.error);
    const scene = await herdWolves({ engine, width: W, height: H, settings: { count: 4096, flashes } });
    const capture = new FrameCapture(engine, W, H);
    try {
        for (let k = 0; k < warmup; k++) await scene.step();
        const shots = [];
        for (let j = 0; j < frames; j++) {
            await scene.step();
            scene.view(path(j / 60), j / 60);
            engine.render(scene.draw(j));
            capture.grab(j);
            shots.push((await capture.take()).bytes);
        }
        const agents = hash('sha256', new Uint8Array(await readBuffer(engine.device, scene.agentsBuffer)));
        const hashes = shots.map(s => hash('sha256', s));
        return { shots, hashes, agents, distinct: new Set(hashes).size };
    } finally {
        scene.destroy();
        capture.destroy();
        await engine.device.queue.onSubmittedWorkDone();
        engine.device.destroy();
    }
}
/** Pixels whose RGB differ between two frames. */
const differingPixels = (x, y) => {
    let n = 0;
    for (let p = 0; p < x.length; p += 4) if (x[p] !== y[p] || x[p + 1] !== y[p + 1] || x[p + 2] !== y[p + 2]) n++;
    return n;
};
{
    const a = await film(false), b = await film(false);
    const pixels = a.shots.map((s, j) => differingPixels(s, b.shots[j]));
    const differ = pixels.filter(Boolean).length, worst = Math.max(...pixels);
    check('F1: two films of 4,096 agents, flashes off: the agents buffer after them equal, byte for byte', a.agents === b.agents);
    check('F1: … every frame equal but for depth ties between overlapping animals: under 1% of its pixels', worst < W * H / 100,
        `${differ} of ${a.shots.length} frames differ; the worst by ${worst} pixels (${(100 * worst / (W * H)).toFixed(2)}%)`);
    check('F1: … and the frames move (not one picture repeated)', a.distinct > a.hashes.length * 0.9, `${a.distinct} distinct of ${a.hashes.length}`);
    const c = await film(true), d = await film(true);
    const flashDiffer = c.hashes.filter((h, j) => h !== d.hashes[j]).length;
    check('F1: with the flashes on, the agents buffer still equal', c.agents === d.agents && c.agents === a.agents);
    log(`   F1: with the flashes on, ${flashDiffer} of ${c.hashes.length} frames differ between the two films (the declared exception: which catches flash)`);
}

// --- F5 ------------------------------------------------------------------------------------------
const PATTERN = [[200, 40, 40], [40, 200, 40], [40, 40, 200], [220, 220, 220]];   // top-left, top-right, bottom-left, bottom-right
const patternWGSL = (w, h) => /* wgsl */ `
@group(0) @binding(0) var<uniform> camera: array<vec4<f32>, 27>;
@group(0) @binding(1) var<storage, read> instances: array<vec4<f32>>;
struct VertexOut { @builtin(position) position: vec4<f32> };
@vertex fn vs_main(@builtin(vertex_index) v: u32) -> VertexOut {
    _ = camera[0];
    _ = instances[0];
    let corner = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var out: VertexOut;
    out.position = vec4<f32>(corner[v], 0.0, 1.0);
    return out;
}
@fragment fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let right = in.position.x >= ${w / 2}.0;
    let bottom = in.position.y >= ${h / 2}.0;
    var c = vec3<f32>(${PATTERN[0].join(', ')});
    if (right && !bottom) { c = vec3<f32>(${PATTERN[1].join(', ')}); }
    if (!right && bottom) { c = vec3<f32>(${PATTERN[2].join(', ')}); }
    if (right && bottom) { c = vec3<f32>(${PATTERN[3].join(', ')}); }
    return vec4<f32>(c / 255.0, 1.0);
}`;
/** The mean of a 16 × 16 block at each quadrant's centre, from `bytes` with `channels` per pixel. */
const quadrants = (bytes, w, h, channels) => [[w / 4, h / 4], [3 * w / 4, h / 4], [w / 4, 3 * h / 4], [3 * w / 4, 3 * h / 4]].map(c => c.map(Math.floor)).map(([cx, cy]) => {
    const sum = [0, 0, 0];
    for (let y = cy - 8; y < cy + 8; y++) for (let x = cx - 8; x < cx + 8; x++) for (let k = 0; k < 3; k++) sum[k] += bytes[(y * w + x) * channels + k];
    return sum.map(v => v / 256);
});
async function pattern(w, h) {
    const engine = new NullGraph();
    const init = await engine.init(null, { width: w, height: h });
    if (!init.success) throw new Error(init.error);
    const { device } = engine;
    const pass = engine.createPass({ name: 'film F5 pattern', isMainScreenPass: true });
    const batch = engine.createBatch(pass, { label: 'film F5 pattern', shaderCode: patternWGSL(w, h), strideFloats: 4, maxInstances: 1,
        vertexLayouts: [{ arrayStride: 4, attributes: [] }], depthWriteEnabled: false, depthCompare: 'always' });
    batch.vertexBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.VERTEX });
    batch.indexBuffer = device.createBuffer({ size: 8, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(batch.indexBuffer, 0, new Uint16Array([0, 1, 2, 0]));
    batch.indexCount = 3;
    batch.currentInstanceCount = 1;
    const capture = new FrameCapture(engine, w, h);
    engine.render([pass]);
    capture.grab(0);
    const { bytes } = await capture.take();
    capture.destroy();
    await device.queue.onSubmittedWorkDone();
    device.destroy();
    return bytes;
}
{
    const bytes = await pattern(W, H);
    const read = quadrants(bytes, W, H, 4);
    check('F5: the pattern reads back exactly, each quadrant in its place', read.every((q, i) => q.every((v, k) => v === PATTERN[i][k])),
        JSON.stringify(read));
    const padded = await pattern(330, 180);       // 1,320-byte rows, padded to 1,536 by the copy and unpacked by FrameCapture
    check('F5: … and at a width whose rows need padding', quadrants(padded, 330, 180, 4).every((q, i) => q.every((v, k) => v === PATTERN[i][k])));

    const dir = mkdtempSync(join(tmpdir(), 'film-f5-'));
    try {
        const file = join(dir, 'pattern.mp4');
        const sink = ffmpegSegment(ffmpeg, { codec: 'libx264', crf: 16, preset: 'veryfast', tune: null }, { width: W, height: H, fps: 60 }, file);
        for (let f = 0; f < 10; f++) await sink.write(bytes);
        await sink.close();
        const decoded = spawnSync(ffmpeg.path, ['-hide_banner', '-loglevel', 'error', '-i', file, '-frames:v', '1',
            '-vf', 'scale=in_color_matrix=bt709:in_range=tv,format=rgb24', '-f', 'rawvideo', '-'], { maxBuffer: 1 << 26 });
        check('F5: ffmpeg decodes the segment', decoded.status === 0 && decoded.stdout.length === W * H * 3, String(decoded.stderr ?? ''));
        const back = quadrants(decoded.stdout, W, H, 3);
        const worst = Math.max(...back.flatMap((q, i) => q.map((v, k) => Math.abs(v - PATTERN[i][k]))));
        check('F5: the decoded video matches the capture within ±3 levels, quadrant by quadrant', worst <= 3, `worst ${worst.toFixed(2)}: ${JSON.stringify(back.map(q => q.map(v => Math.round(v))))}`);
        const tags = spawnSync(ffmpeg.path.replace(/ffmpeg(\.exe)?$/i, 'ffprobe$1'), ['-v', 'error', '-show_entries',
            'stream=color_space,color_primaries,color_transfer', '-of', 'compact', file], { encoding: 'utf8' });
        check('F5: the video is tagged BT.709 (matrix, primaries, transfer)', /color_space=bt709\|color_transfer=bt709\|color_primaries=bt709/.test(tags.stdout ?? ''), (tags.stdout ?? '').trim());
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
}
await finish('film GPU checks done');
