// D2.2: GPU codecs and bounded-memory, segment-local binary tracks/checkpoints.
import { closeSync, existsSync, mkdirSync, openSync, readFileSync, readSync, renameSync, writeFileSync, writeSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { hash } from 'node:crypto';
import { readBuffer } from '../../test/gpu/readback.mjs';
import { resolveLayout, layoutWGSL } from './layout.mjs';

export const number = i => String(i).padStart(4, '0');
export function atomicJSON(file, value) {
    mkdirSync(dirname(file), { recursive: true });
    writeFileSync(`${file}.tmp`, JSON.stringify(value, null, 1));
    renameSync(`${file}.tmp`, file);
}
export const inputHash = value => hash('sha256', JSON.stringify(value));
export function track(name, buffer, options = {}) {
    if (!/^[a-z0-9-]+$/.test(name)) throw new Error(`film: invalid track name ${name}.`);
    const layout = resolveLayout({ stride: options.stride ?? (options.preset ? 64 : buffer.size), ...options });
    if (buffer.size % layout.stride) throw new Error(`film: track ${name} size is not a whole number of records.`);
    if (globalThis.GPUBufferUsage && buffer.usage !== undefined) {
        const required = GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | (layout.fields ? GPUBufferUsage.STORAGE : 0);
        if ((buffer.usage & required) !== required) throw new Error(`film: track ${name} buffer needs COPY_SRC, COPY_DST${layout.fields ? ' and STORAGE' : ''} usage.`);
    }
    return { name, buffer, layout, size: buffer.size / layout.stride * layout.packedStride };
}

export class GPUCodec {
    constructor(device, t) {
        this.device = device; this.track = t; this.owned = [];
        if (!t.layout.fields) return;
        const alloc = (size, usage) => { const b = device.createBuffer({ size, usage }); this.owned.push(b); return b; };
        this.a = alloc(t.size, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
        this.b = alloc(t.size, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.params = alloc(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        const code = layoutWGSL(t.layout);
        const pipeline = text => device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: text }), entryPoint: 'main' } });
        this.pack = pipeline(code.pack); this.unpack = pipeline(code.unpack);
        const bind = (p, buffers) => device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
        this.packGroup = bind(this.pack, [t.buffer, this.a]);
        this.unpackGroup = bind(this.unpack, [this.a, this.b, t.buffer, this.params]);
    }
    dispatch(pipeline, group) {
        const enc = this.device.createCommandEncoder(), pass = enc.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(Math.ceil(this.track.buffer.size / this.track.layout.stride / 64)); pass.end();
        this.device.queue.submit([enc.finish()]);
    }
    async save() {
        if (!this.track.layout.fields) return new Uint8Array(await readBuffer(this.device, this.track.buffer));
        this.dispatch(this.pack, this.packGroup);
        return new Uint8Array(await readBuffer(this.device, this.a));
    }
    restore(a, b, u, interval) {
        if (!this.track.layout.fields) { this.device.queue.writeBuffer(this.track.buffer, 0, a); return; }
        this.device.queue.writeBuffer(this.a, 0, a); this.device.queue.writeBuffer(this.b, 0, b);
        this.device.queue.writeBuffer(this.params, 0, new Float32Array([u, interval, 0, 0]));
        this.dispatch(this.unpack, this.unpackGroup);
    }
    destroy() { for (const b of this.owned) b.destroy(); }
}

export async function saveCheckpoint(device, buffers, file, inputs, step) {
    mkdirSync(dirname(file), { recursive: true });
    const fd = openSync(`${file}.tmp`, 'w');
    try {
        const header = Buffer.from(JSON.stringify({ version: 1, inputs, step, buffers: buffers.map(({ name, buffer }) => ({ name, size: buffer.size })) }));
        const length = Buffer.alloc(4); length.writeUInt32LE(header.length);
        writeSync(fd, length); writeSync(fd, header);
        for (const { buffer } of buffers) writeSync(fd, new Uint8Array(await readBuffer(device, buffer)));
    } finally { closeSync(fd); }
    renameSync(`${file}.tmp`, file);
}
export function restoreCheckpoint(device, buffers, file, inputs) {
    const fd = openSync(file, 'r');
    try {
        const read = size => { const bytes = Buffer.alloc(size); if (readSync(fd, bytes, 0, size, null) !== size) throw new Error(`film: truncated checkpoint ${file}.`); return bytes; };
        const header = JSON.parse(read(read(4).readUInt32LE()).toString());
        if (header.version !== 1 || header.inputs !== inputs || JSON.stringify(header.buffers) !== JSON.stringify(buffers.map(({ name, buffer }) => ({ name, size: buffer.size })))) throw new Error('film: resume inputs differ from the checkpoint (settings, layouts, shaders or build).');
        // Validate and read all bytes before mutating any buffer. Checkpoints are bounded by scene state,
        // not the duration of the film; tracks remain streamed one sample at a time.
        const bytes = buffers.map(({ buffer }) => read(buffer.size));
        buffers.forEach(({ buffer }, i) => device.queue.writeBuffer(buffer, 0, bytes[i]));
        return header.step;
    } finally { closeSync(fd); }
}

export class TrackWriter {
    constructor(device, tracks, out, index) {
        this.items = tracks.map(t => {
            const file = join(out, 'tracks', t.name, `${number(index)}.bin`);
            mkdirSync(dirname(file), { recursive: true });
            return { t, file, fd: openSync(`${file}.tmp`, 'w'), steps: [], codec: new GPUCodec(device, t) };
        });
    }
    async save(step, force = false) {
        for (const item of this.items) if ((force || step % item.t.layout.every === 0) && item.steps.at(-1) !== step) {
            writeSync(item.fd, await item.codec.save()); item.steps.push(step);
        }
    }
    close(commit = true) {
        for (const item of this.items) {
            closeSync(item.fd); item.codec.destroy();
            if (commit) { renameSync(`${item.file}.tmp`, item.file); atomicJSON(`${item.file}.json`, { steps: item.steps, size: item.t.size }); }
        }
    }
}
export class TrackReader {
    constructor(device, tracks, out, segments) {
        this.items = tracks.map(t => {
            const samples = [];
            for (const s of segments) {
                const file = join(out, 'tracks', t.name, `${number(s.index)}.bin`);
                const meta = JSON.parse(readFileSync(`${file}.json`, 'utf8'));
                if (meta.size !== t.size) throw new Error(`film: track ${t.name} layout differs from recording.`);
                meta.steps.forEach((step, offset) => samples.push({ step, file, offset: offset * t.size }));
            }
            samples.sort((a, b) => a.step - b.step);
            return { t, samples, codec: new GPUCodec(device, t) };
        });
    }
    restore(step) {
        for (const { t, samples, codec } of this.items) {
            let lo = 0, hi = samples.length - 1;
            while (lo < hi) { const mid = Math.ceil((lo + hi) / 2); if (samples[mid].step <= step) lo = mid; else hi = mid - 1; }
            const a = samples[lo], b = samples[Math.min(lo + 1, samples.length - 1)];
            if (!a) throw new Error(`film: no samples for track ${t.name}.`);
            const read = s => {
                const fd = openSync(s.file, 'r'), bytes = Buffer.alloc(t.size);
                try { if (readSync(fd, bytes, 0, bytes.length, s.offset) !== bytes.length) throw new Error(`film: truncated track ${s.file}.`); }
                finally { closeSync(fd); }
                return bytes;
            };
            codec.restore(read(a), read(b), a.step === b.step ? 0 : (step - a.step) / (b.step - a.step), (b.step - a.step) * t.layout.dt);
        }
    }
    destroy() { for (const item of this.items) item.codec.destroy(); }
}

export function resumeSegments(progress, plan) {
    if (!progress || progress.frames !== plan.frames || !Array.isArray(progress.segments)) throw new Error('film: resume progress does not match the film plan.');
    progress.segments.forEach((s, i) => {
        const p = plan.segments[i];
        if (!p || s.index !== i || s.first !== p.first || s.count !== p.count || !Array.isArray(s.hashes) || s.hashes.length !== p.count) throw new Error('film: resume progress has a missing or invalid segment.');
        if (s.file && !existsSync(s.file)) throw new Error(`film: resume segment missing: ${s.file}.`);
    });
    return plan.segments.slice(progress.segments.length);
}