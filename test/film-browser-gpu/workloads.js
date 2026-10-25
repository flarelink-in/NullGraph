// F2's browser G tests: the workloads the cases film, and the instruments they film them with.
// - patternWorkload: the test pattern (pattern.js), stateless; a step does nothing unless the guard's tests ask
//   it to take wall time (injected timing: never GPU work, never a reset, §10 G5).
// - kerrWorkload: the Node film tool's Kerr scene (tools/film/scenes/kerr-black-hole.mjs) through a browser
//   adapter. It has no reset(), so recordFilm rebuilds it after calibration.
// - trackDevice: every buffer, texture and query set made on the app's device, and whether it was destroyed, so a
//   case can say what a film left behind (G1).
// - capturingMediabunny: mediabunny with a VideoSampleSource that copies chosen frames' bytes before encoding
//   them, so G6 compares what each source handed the encoder, not only what came out.

import kerrBlackHole from '../../tools/film/scenes/kerr-black-hole.mjs';
import { PATTERN_WGSL } from './pattern.js';

export const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

/**
 * The pattern as a workload. `dt` is its fixed step; `stepMs(n)` waits that long in the nth step since the
 * workload was built (1-based, calibration's included), on a timer, or busy on the page's thread with
 * `busy` (a hidden page's timers fire at most once a second; work does not slow); `builds` counts constructions.
 */
export function patternWorkload({ dt = 1 / 60, stepMs = () => 0, busy = false } = {}) {
    const log = { builds: 0, steps: 0, draws: 0, disposals: 0, engines: [] };
    const create = async ({ engine, width, height }) => {
        log.builds++;
        log.engines.push(new WeakRef(engine));
        const pass = engine.createPass({ name: 'film G pattern', isMainScreenPass: true });
        const batch = engine.createBatch(pass, { label: 'film G pattern', shaderCode: PATTERN_WGSL, strideFloats: 4, maxInstances: 1 });
        const info = new Float32Array(4);
        let disposed = false;
        return {
            dt,
            info: { workload: 'pattern' },
            // Stateless: frame j shows j whatever was stepped, so an exact reset is no reset at all.
            reset() {},
            async step() {
                const ms = stepMs(++log.steps);
                if (ms && busy) for (const end = performance.now() + ms; performance.now() < end;);
                else if (ms) await sleep(ms);
            },
            draw(frame) {
                log.draws++;
                info.set([frame.index, width, height, 0]);
                engine.updateBatchData(batch, info, 1);
                return [pass];
            },
            dispose() {
                if (disposed) return;
                disposed = true;
                log.disposals++;
                engine.clearBatch(pass, batch);
            },
        };
    };
    return { create, log };
}

/** The Kerr scene (1,048,576 particles unless `count`) as a workload; `orbit` is S1's slow camera, for a film's `camera`. */
export function kerrWorkload({ count = 1048576 } = {}) {
    const log = { builds: 0, disposals: 0, engines: [] };
    const create = async ({ engine, width, height }) => {
        log.builds++;
        log.engines.push(new WeakRef(engine));
        // Construction places the particles (one fenced dispatch) before step zero: initialization, not a step (§4.1).
        const scene = await kerrBlackHole({ engine, width, height, settings: { count }, film: { track: () => null } });
        let disposed = false;
        return {
            dt: scene.dt,
            info: { workload: 'kerr-black-hole', particles: count },
            // The scene's step awaits its own fence (the Node tool's guard times it). Here the runner fences
            // (§4.7), so that promise settles by itself and the queue stays fed.
            step() { scene.step().catch(() => {}); },
            draw(frame) {
                scene.view(frame.view, frame.seconds);
                return scene.draw();
            },
            dispose() {
                if (disposed) return;
                disposed = true;
                log.disposals++;
                // The scene frees its post chain and forgets its passes; its batch buffers go with the engine.
                scene.destroy();
            },
        };
    };
    return { create, log };
}

/** S1's camera: a slow orbit, so the encoder sees motion as well as the particles' own. */
export const orbit = seconds => ({ target: [0, 0, 0], distance: 17.5, azimuth: -0.35 + 0.25 * seconds, polar: 1.32 });

/**
 * Count what is made on `device` and not yet destroyed: buffers, textures and query sets. `live()` gives the
 * counts and the labels still alive. The film adopts the app's device, so everything it makes passes here.
 */
export function trackDevice(device) {
    const alive = new Map();
    let made = 0;
    for (const [kind, method] of [['buffer', 'createBuffer'], ['texture', 'createTexture'], ['querySet', 'createQuerySet']]) {
        const create = device[method].bind(device);
        device[method] = descriptor => {
            const object = create(descriptor);
            made++;
            alive.set(object, `${kind} ${descriptor?.label ?? '(no label)'}`);
            const destroy = object.destroy.bind(object);
            object.destroy = () => {
                alive.delete(object);
                destroy();
            };
            return object;
        };
    }
    return {
        get made() { return made; },
        live() {
            const labels = [...alive.values()];
            return { count: labels.length, labels: labels.sort() };
        },
    };
}

/**
 * mediabunny, with a VideoSampleSource that copies the frames whose timestamps (µs) are in `wanted` into
 * `captured` as tightly packed RGBA before encoding them. Canvas frames copy un-premultiplied (S1), so this is
 * exact only where alpha is 1, as it is in Kerr after its tonemap.
 */
export async function capturingMediabunny(wanted, captured) {
    const mb = await import('mediabunny');
    class VideoSampleSource extends mb.VideoSampleSource {
        async add(sample, options) {
            if (wanted.has(sample.microsecondTimestamp)) {
                const bytes = new Uint8Array(sample.allocationSize());
                const [plane] = await sample.copyTo(bytes);
                captured.set(sample.microsecondTimestamp, { format: sample.format, ...toRGBA(bytes, plane, sample.format, sample.codedWidth, sample.codedHeight) });
            }
            return super.add(sample, options);
        }
    }
    return { ...mb, VideoSampleSource };
}

/** One plane of 8-bit RGB(A|X) or BGR(A|X) as tightly packed RGBA. */
function toRGBA(bytes, plane, format, width, height) {
    if (!['RGBA', 'RGBX', 'BGRA', 'BGRX'].includes(format)) return { rgba: null };
    const bgr = format.startsWith('B');
    const rgba = new Uint8Array(width * height * 4);
    for (let y = 0; y < height; y++) {
        for (let x = 0, s = plane.offset + y * plane.stride, d = y * width * 4; x < width; x++, s += 4, d += 4) {
            rgba[d] = bytes[s + (bgr ? 2 : 0)];
            rgba[d + 1] = bytes[s + 1];
            rgba[d + 2] = bytes[s + (bgr ? 0 : 2)];
            rgba[d + 3] = bytes[s + 3];
        }
    }
    return { rgba };
}

/** How two RGBA frames differ in RGB: pixels that differ, and the largest channel difference. */
export function compareRGB(a, b) {
    if (a.length !== b.length) return { pixels: a.length / 4, differing: -1, maxDiff: -1 };
    let differing = 0, maxDiff = 0;
    for (let i = 0; i < a.length; i += 4) {
        const d = Math.max(Math.abs(a[i] - b[i]), Math.abs(a[i + 1] - b[i + 1]), Math.abs(a[i + 2] - b[i + 2]));
        if (d) { differing++; if (d > maxDiff) maxDiff = d; }
    }
    return { pixels: a.length / 4, differing, maxDiff };
}
