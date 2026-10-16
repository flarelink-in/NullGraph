// D2.2 F2–F4. Loaded by the film harness; uses the same checked host and result collector.
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { NullGraph } from '../../dist/index.js';
import { readBuffer } from '../gpu/readback.mjs';
import { resolveLayout, packCPU, unpackCPU } from '../../tools/film/layout.mjs';
import { track, GPUCodec, TrackWriter, TrackReader, saveCheckpoint, restoreCheckpoint } from '../../tools/film/records.mjs';
import herd from '../../tools/film/scenes/herd-wolves.mjs';
import { filmRuntime } from '../../tools/film/ort.mjs';
import { FrameCapture } from '../../tools/film/capture.mjs';

export async function recordsChecks(check, log) {
    const dir = mkdtempSync(join(tmpdir(), 'film-records-'));
    const engine = new NullGraph();
    const init = await engine.init(null, { width: 320, height: 180 });
    if (!init.success) throw new Error(init.error);
    const device = engine.device;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const equals = (a, b) => Buffer.from(a).equals(Buffer.from(b));
    try {
        // A made-up layout, 2,048 independent points, no agent imports in the recorder core.
        const count = 2048, source = new Float32Array(count * 8);
        for (let i = 0; i < count; i++) source.set([i / 8, 2, 3.1, 0.98, 0, 0, 999, 777], i * 8);
        const words = new Uint32Array(source.buffer);
        for (let i = 0; i < count; i++) { words[i*8+4] = 0x12345601; words[i*8+5] = 0xfedcba98; }
        const buffer = device.createBuffer({ size: source.byteLength, usage });
        const options = { stride: 32, every: 6, fields: [
            { word: 0, store: 'f32', interp: 'hermite', velocity: 1, jump: 10 },
            { word: 1, store: 'f16', interp: 'linear' }, { word: 2, store: 'f16', interp: 'angle' },
            { word: 3, store: 'unorm16', interp: 'wrap', maxRate: 4 }, { word: 4, store: 'u8' }, { word: 5, store: 'u32' },
        ] };
        const t = track('points', buffer, options), codec = new GPUCodec(device, t);
        device.queue.writeBuffer(buffer, 0, source);
        const a = await codec.save(), expected = packCPU(t.layout, source);
        check('U2 GPU: generated pack matches CPU, all storage kinds and padded records', equals(a, expected));
        const end = source.slice();
        for (let i = 0; i < count; i++) { end[i*8] += 0.2; end[i*8+2] = -3.1; end[i*8+3] = 0.02; }
        device.queue.writeBuffer(buffer, 0, end);
        const b = await codec.save();
        for (const u of [0, 0.25, 0.5, 1]) {
            device.queue.writeBuffer(buffer, 0, source); codec.restore(a, b, u, 0.1);
            const gpu = new Float32Array(await readBuffer(device, buffer)), cpu = source.slice();
            unpackCPU(t.layout, a, b, u, cpu);
            let worst = 0;
            for (let i = 0; i < gpu.length; i++) if (i % 8 < 4 || i % 8 > 5) worst = Math.max(worst, Math.abs(gpu[i] - cpu[i]));
            check(`U2 GPU: unpack matches CPU at u=${u}, preserving untouched words`, worst < 0.0001 && new Uint32Array(gpu.buffer)[4] === words[4], `worst ${worst}`);
        }
        const rawBuffer = device.createBuffer({ size: 16, usage });
        const raw = track('raw', rawBuffer, { every: 1 });
        const w = new TrackWriter(device, [t, raw], dir, 0);
        device.queue.writeBuffer(buffer, 0, source); device.queue.writeBuffer(rawBuffer, 0, new Uint32Array([1,2,3,4])); await w.save(0, true);
        device.queue.writeBuffer(buffer, 0, end); device.queue.writeBuffer(rawBuffer, 0, new Uint32Array([5,6,7,8])); await w.save(6, true); w.close();
        const r = new TrackReader(device, [t, raw], dir, [{ index: 0 }]);
        r.restore(3);
        const back = new Float32Array(await readBuffer(device, buffer));
        check('F3: generic points track streams to disk and interpolates back without stepping', Math.abs(back[0] - 0.1) < 1e-6 && back[6] === 999);
        check('F3: raw tracks hold their left sample', new Uint32Array(await readBuffer(device, rawBuffer))[0] === 1);
        r.restore(6);
        check('F3: raw tracks select the saved value at its exact step', new Uint32Array(await readBuffer(device, rawBuffer))[0] === 5);
        r.destroy(); codec.destroy(); buffer.destroy(); rawBuffer.destroy();

        for (const brains of ['rules', 'network', 'ort']) {
            const runtime = brains === 'ort' ? await filmRuntime() : null;
            const e = runtime ? new NullGraph() : engine;
            if (runtime) { const init = await e.init(null, { device: runtime.device, width: 320, height: 180 }); if (!init.success) throw new Error(init.error); }
            const s = await herd({ engine: e, width: 320, height: 180, settings: { count: 4096, brains, flashes: false }, trackEvery: 1, runtime });
            const file = join(dir, `${brains}.bin`);
            for (let i = 0; i < 5; i++) await s.step();
            await saveCheckpoint(e.device, s.checkpoint, file, 'same-inputs', 5);
            for (let i = 0; i < 7; i++) await s.step();
            const expected = await readBuffer(e.device, s.agentsBuffer);
            restoreCheckpoint(e.device, s.checkpoint, file, 'same-inputs');
            for (let i = 0; i < 7; i++) await s.step();
            check(`F2: checkpoint resume equals uninterrupted state (${brains})`, equals(expected, await readBuffer(e.device, s.agentsBuffer)));
            let refused = false; try { restoreCheckpoint(e.device, s.checkpoint, file, 'wrong-inputs'); } catch { refused = true; }
            check(`F2: incompatible checkpoint refused (${brains})`, refused);
            if (brains === 'network') {
                const capture = new FrameCapture(e, 320, 180), codecs = s.tracks.map(t => new GPUCodec(e.device, t));
                const samples = await Promise.all(codecs.map(c => c.save()));
                s.view(null, 0); e.render(s.draw(0)); capture.grab(0); const before = (await capture.take()).bytes;
                codecs.forEach((c, i) => c.restore(samples[i], samples[i], 0, s.dt));
                e.render(s.draw(0)); capture.grab(0); const after = (await capture.take()).bytes;
                let pixels = 0; for (let p = 0; p < before.length; p += 4) if (before[p] !== after[p] || before[p+1] !== after[p+1] || before[p+2] !== after[p+2]) pixels++;
                let mae = 0; for (let p = 0; p < before.length; p++) mae += Math.abs(before[p] - after[p]);
                mae /= before.length;
                check('F4: m=1 packed replay: <=2% changed pixels and mean channel error <=0.5', pixels / (320*180) <= 0.02 && mae <= 0.5, `${pixels} pixels, MAE ${mae}`);
                // Compare actual six-step trajectories against the exact midpoint; pin each stored field.
                const t = s.tracks[0], l = resolveLayout({ ...t.layout, fields: t.layout.fields, every: 6 });
                for (const f of l.fields) if (f.interp === 'hermite') f.jump = 4.2 * 6 / 60 * 1.05;
                const first = new Uint8Array(await readBuffer(e.device, t.buffer)); let middle;
                for (let k = 1; k <= 6; k++) { await s.step(); if (k === 3) middle = new Float32Array(await readBuffer(e.device, t.buffer)); }
                const last = new Uint8Array(await readBuffer(e.device, t.buffer)), interpolated = new Float32Array(first.buffer.slice(0));
                unpackCPU(l, packCPU(l, first), packCPU(l, last), 0.5, interpolated);
                const limits = { 0: 0.03, 2: 0.03, 4: 0.3, 6: 0.3, 7: 0.3, 11: 0.06 };
                const start = new Float32Array(first.buffer, first.byteOffset, first.byteLength / 4), finish = new Float32Array(last.buffer, last.byteOffset, last.byteLength / 4);
                let discrete = 0;
                for (let i = 0; i < middle.length; i += 16) if (Math.hypot(finish[i]-start[i], finish[i+2]-start[i+2]) > 4.2*6/60*1.05 || Math.abs(finish[i+11]-start[i+11]) > 0.11) discrete++;
                log(`F4: ${discrete} catch/respawn intervals checked by the jump/step rule, not continuous-motion bounds`);
                let jumpExact = true;
                for (let i = 0; i < middle.length; i += 16) for (const word of [0, 2]) if (Math.abs(finish[i+word]-start[i+word]) > 4.2*6/60*1.05) jumpExact &&= interpolated[i+word] === finish[i+word];
                check('F4: respawn jumps choose a saved endpoint, never a smeared position', jumpExact);
                for (const [word, limit] of Object.entries(limits)) {
                    let worst = 0; for (let base = 0; base < middle.length; base += 16) {
                        if (Math.hypot(finish[base]-start[base], finish[base+2]-start[base+2]) > 4.2*6/60*1.05 || Math.abs(finish[base+11]-start[base+11]) > 0.11) continue;
                        const i = base + Number(word), delta = interpolated[i] - middle[i];
                        const error = Number(word) === 7 ? Math.abs(Math.atan2(Math.sin(delta), Math.cos(delta))) : Math.abs(delta);
                        worst = Math.max(worst, error);
                    }
                    check(`F4: m=6 word ${word} midpoint error <= ${limit}`, worst <= limit, `worst ${worst}`);
                }
                codecs.forEach(c => c.destroy()); capture.destroy();
            }
            s.destroy(); await runtime?.destroy();
            log(`F2 ${brains} checked`);
        }
    } finally { device.destroy(); rmSync(dir, { recursive: true, force: true }); }
}