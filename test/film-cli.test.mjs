// Real CLI integration (GPU required): separate processes, segment commits, replay and detach.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const run = (...args) => {
    const p = spawnSync(process.execPath, ['tools/film/film.mjs', ...args], { encoding: 'utf8', timeout: 120000 });
    assert.equal(p.status, 0, `${p.stdout}\n${p.stderr}`); return p;
};
test('D2.2 CLI: fresh-process resume matches uninterrupted films; replay and detach work', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'film-cli-'));
    try {
        for (const brains of ['rules', 'network', 'ort']) {
            const file = join(dir, `${brains}.json`), full = join(dir, `${brains}-full`), resumed = join(dir, `${brains}-resume`);
            const manifest = { name: 'cli', scene: 'herd-wolves', settings: { count: 4096, brains, flashes: false }, warmupSeconds: 0.1,
                film: { seconds: 0.2, fps: 60, width: 160, height: 90 }, segmentSeconds: 0.1 };
            writeFileSync(file, JSON.stringify(manifest));
            const flags = ['--frames', 'png', '--track-every', '1'];
            run(file, '--out', full, ...flags);
            run(file, '--out', resumed, ...flags, '--stop-after', '1');
            assert.equal(JSON.parse(readFileSync(join(resumed, 'progress.json'))).segments.length, 1);
            run(file, '--out', resumed, ...flags, '--resume');
            const json = out => JSON.parse(readFileSync(join(out, 'progress.json')));
            assert.deepEqual(json(full).segments.map(s => s.hashes), json(resumed).segments.map(s => s.hashes), `${brains}: resumed frame hashes`);
            for (const name of ['agents', 'phase', 'stats', 'state']) {
                assert.deepEqual(readFileSync(join(full, 'tracks', name, '0001.bin')), readFileSync(join(resumed, 'tracks', name, '0001.bin')), `${brains}: resumed ${name} track`);
            }
            run(file, '--out', resumed, ...flags, '--replay');
            assert.equal(json(join(resumed, 'replay')).segments.length, 2);
            assert.equal(JSON.parse(readFileSync(join(resumed, 'replay', 'report.json'))).timing.worstStepMs, 0);
            if (brains === 'network') {
                manifest.film.fps = 120;
                writeFileSync(file, JSON.stringify(manifest));
                const slow = join(resumed, 'slow');
                run(file, '--out', resumed, '--frames', 'none', '--replay', '--speed', '0.25', '--replay-out', slow);
                assert.equal(json(slow).frames, 96, 'quarter speed at twice the fps');
                manifest.film.fps = 60;
            }
            manifest.settings.seed = 12; writeFileSync(file, JSON.stringify(manifest));
            const refusal = spawnSync(process.execPath, ['tools/film/film.mjs', file, '--out', resumed, ...flags, '--resume'], { encoding: 'utf8', timeout: 30000 });
            assert.equal(refusal.status, 2); assert.match(refusal.stderr, /inputs differ/);
        }
        const file = join(dir, 'detach.json'), out = join(dir, 'detached');
        writeFileSync(file, JSON.stringify({ scene: 'herd-wolves', settings: { count: 64 }, film: { seconds: 0.1, width: 160, height: 90 } }));
        run(file, '--out', out, '--frames', 'none', '--detach');
        const deadline = Date.now() + 30000;
        while (!existsSync(join(out, 'report.json')) && Date.now() < deadline) await new Promise(r => setTimeout(r, 100));
        assert.ok(existsSync(join(out, 'report.json')), 'detached child completes');
        assert.equal(JSON.parse(readFileSync(join(out, 'report.json'))).failure, null);
        assert.match(run(file, '--out', out, '--status').stdout, /segments/);
    } finally { rmSync(dir, { recursive: true, force: true }); }
});