// Append-only checkpoints: a failed setting never overwrites earlier seed-level evidence.
export async function createJournal({ file, emit = console.log, browser = false } = {}) {
    let seq = 0, fs, fd;
    if (!browser) {
        fs = await import('node:fs');
        fs.mkdirSync(new URL('./out/', import.meta.url), { recursive: true });
        fd = fs.openSync(file, 'wx');
    }
    const events = [];
    return {
        events,
        write(type, data) {
            const event = { seq: seq++, at: new Date().toISOString(), type, data };
            // Serialize NOW, not later after mutable objects change. Each line is independently valid.
            const line = JSON.stringify(event, (_, v) => ArrayBuffer.isView(v) ? Array.from(v) : v);
            if (fs) { fs.writeSync(fd, line + '\n'); fs.fsyncSync(fd); }
            else { events.push(JSON.parse(line)); emit('S7 EVENT ' + line); }
            return event;
        },
        close() { if (fd !== undefined) { fs.closeSync(fd); fd = undefined; } },
    };
}

let sink = () => {};
export function installEvidenceSink(write) { sink = write; }
export function record(type, data) { sink(type, data); }