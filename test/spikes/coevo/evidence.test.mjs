import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createJournal } from './evidence.mjs';

test('completed samples and failing submission survive a later throw, with snapshot identity', async () => {
    const dir = mkdtempSync(join(tmpdir(), 's7-journal-')), file = join(dir, 'events.jsonl');
    const j = await createJournal({ file });
    try {
        const sample = { catches: new Uint32Array([1, 2]), seed: 42 };
        j.write('sample', sample); sample.catches[0] = 999;
        assert.throws(() => { j.write('submission-failed', { n: 1, ms: 1709 }); throw new Error('injected later failure'); });
        // Read while still open, as a different process would after abrupt termination.
        const events = readFileSync(file, 'utf8').trim().split('\n').map(JSON.parse);
        assert.deepEqual(events.map(e => e.seq), [0, 1]);
        assert.deepEqual(events[0].data.catches, [1, 2]);
        assert.equal(events[1].data.ms, 1709);
    } finally { j.close(); rmSync(dir, { recursive: true }); }
});
test('browser event checkpoints serialize immediately and publish through the console sink', async () => {
    const lines = [], j = await createJournal({ browser: true, emit: line => lines.push(line) });
    j.write('fixture', { pass: false });
    assert.equal(JSON.parse(lines[0].slice('S7 EVENT '.length)).data.pass, false);
    assert.equal(j.events.length, 1); j.close();
});