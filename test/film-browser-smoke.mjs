// Real-device smoke for F2's managed runner, without WebCodecs: Node/Dawn, exact readback and PNG.
// Run after building: node test/film-browser-smoke.mjs (NG_DAWN_FLAGS=adapter=AMD or adapter=NVIDIA picks the GPU)
// This is not the browser G1–G9 gate (npm run test:film:browser): no canvas capture, encoder, muxer or
// storage is exercised.
import assert from 'node:assert/strict';
import { installNodeGPU, describeDevice } from './gpu/node-gpu.mjs';
import { recordFilm, sha256, snapshot } from '../dist/film.js';
import { readPNG } from '../tools/film/png-read.mjs';

const host = installNodeGPU();
let device;
try {
    const adapter = await host.gpu.requestAdapter({ powerPreference: 'high-performance' });
    assert.ok(adapter, 'WebGPU adapter is required for this smoke test');
    device = await adapter.requestDevice();
    const createWorkload = async ({ engine }) => {
        let steps = 0;
        const attachment = {
            view: engine.screenTexture.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store',
        };
        const pass = engine.createPass({ name: 'film smoke', isMainScreenPass: false, colorAttachments: [attachment] });
        return {
            dt: 1 / 60,
            step() { steps++; },
            reset() { steps = 0; },
            draw() {
                attachment.clearValue = { r: steps / 255, g: 0, b: 0, a: 1 };
                return [pass];
            },
            dispose() {},
        };
    };
    const options = {
        device, createWorkload, output: false, hashes: true,
        film: { width: 66, height: 10, fps: 30, seconds: 0.1, warmupSeconds: 1 / 60 },
    };
    const a = await recordFilm(options);
    const b = await recordFilm(options);
    const expected = await Promise.all([3, 5, 7].map(async steps => {
        const rgba = new Uint8Array(66 * 10 * 4);
        for (let i = 0; i < rgba.length; i += 4) rgba.set([steps, 0, 0, 255], i);
        return sha256(rgba);
    }));
    assert.deepEqual(a.report.hashes, expected, 'GPU frames must show W + s × (j + 1), including row unpadding');
    assert.deepEqual(b.report.hashes, expected, 'repeated recording must reset exactly');
    assert.equal(a.status, 'completed');
    assert.equal(a.frames, 3);
    const shot = await snapshot({ device, createWorkload, width: 33, height: 7 });
    assert.deepEqual(readPNG(shot.png).rgba, shot.rgba);
    assert.deepEqual([...shot.rgba.slice(0, 4)], [0, 0, 0, 255]);
    await host.settle();
    assert.deepEqual(host.events, [], 'uncaptured GPU errors or unexpected device loss');
    console.log(`F2 readback/hash/reset/snapshot smoke passed on ${describeDevice(device)}; no WebGPU errors.`);
} finally {
    for (const owned of host.devices) owned.destroy();
}