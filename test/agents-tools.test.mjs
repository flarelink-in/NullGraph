// Step 7, part 2 (docs/agents-cloning-design.md §10): the training tools' headless checks.
// Part of `npm run test:agents`; build first (`npm run build:lib`).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as A from '../dist/agents.js';
import { WgslReflect } from '../node_modules/wgsl_reflect/wgsl_reflect.module.js';
import { DRIVER, LABEL, LABEL_DRIVING, TAG, HERD_DEFAULTS, herdDefinition, herdRule, herdWolves } from '../tools/agents/scenario/herd-wolves.mjs';
import { ROW, ROW_WORDS, Rows, flagsReason, keepRow, readDataset, rowFlags, rowFloats, splitRows, writeDataset } from '../tools/agents/dataset.mjs';
import { isFresh, recordEpisodes } from '../tools/agents/recorder.mjs';

const words = values => new Uint32Array(new Float32Array(values).buffer);

/** A small dataset with awkward values: negative zero, a denormal, the largest f32. */
function sample() {
    const definition = herdDefinition(), floats = rowFloats(definition), prey = new Rows(floats), wolf = new Rows(floats);
    for (let row = 0; row < 3000; row++) {
        const obs = Float32Array.from({ length: definition.obsDim }, (_, k) => A.rand01(row, k, 1) * 2 - 1);
        obs[0] = [-0, 1e-45, 3.4028234663852886e38][row % 3];
        const label = [A.rand01(row, 0, 2) * 2 - 1, A.rand01(row, 1, 2)], act = [A.rand01(row, 0, 3) * 2 - 1, A.rand01(row, 1, 3)];
        (row % 4 ? prey : wolf).push(words(obs), words(label), words(act), row % 3, row * 3, row, rowFlags(row % 7 === 0, row % 6));
    }
    const header = { scenario: { name: 'herd-wolves' }, definition: { obsDim: definition.obsDim, actDim: definition.actDim },
        episodes: [{ seed: 11, heldOut: false }, { seed: 12, heldOut: false }, { seed: 13, heldOut: true }] };
    return { definition, header, species: { prey: prey.finish(), wolf: wolf.finish() } };
}

test('U24: a dataset written and read back is byte-identical; a changed or short file is refused', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'ng-dataset-'));
    try {
        const { definition, header, species } = sample();
        assert.equal(rowFloats(definition), 28);
        const written = await writeDataset(dir, header, species), back = await readDataset(dir);
        assert.deepEqual(back.header, written);
        assert.deepEqual(Object.keys(back.species), ['prey', 'wolf']);
        for (const name of ['prey', 'wolf']) {
            assert.equal(back.species[name].rows, species[name].rows);
            // Compared as words: a float comparison would call -0 and 0, or two NaNs, equal or unequal wrongly.
            assert.deepEqual(new Uint32Array(back.species[name].f32.buffer), new Uint32Array(species[name].f32.buffer));
            assert.deepEqual(back.species[name].u32, species[name].u32);
        }
        assert.equal(species.prey.rows + species.wolf.rows, 3000);
        assert.equal(written.files['prey.f32'].bytes, species.prey.rows * 28 * 4);
        assert.equal(written.files['wolf.u32'].bytes, species.wolf.rows * ROW_WORDS * 4);
        assert.match(written.files['prey.f32'].sha256, /^[0-9a-f]{64}$/);

        const file = join(dir, 'prey.f32'), bytes = readFileSync(file);
        const flipped = Buffer.from(bytes);
        flipped[1234] ^= 1;
        writeFileSync(file, flipped);
        await assert.rejects(readDataset(dir), /dataset: prey\.f32 does not match header\.json's SHA-256\. It was changed or cut short\./);
        writeFileSync(file, bytes.subarray(0, bytes.length - 4));
        await assert.rejects(readDataset(dir), new RegExp(`dataset: prey\\.f32 is ${bytes.length - 4} bytes; header\\.json promises ` +
            `${bytes.length} \\(${species.prey.rows} rows × 112 bytes\\)\\.`));
        writeFileSync(file, bytes);
        await readDataset(dir);

        // A row is the observation, then the label, then the driver's action; its words are
        // episode, tick, agent, flags. A reader that swapped two of them would still round-trip.
        const rows = new Rows(28), obs = Float32Array.from({ length: 24 }, (_, k) => k + 1);
        rows.push(words(obs), words([-0.25, 0.5]), words([0.75, -1]), 2, 1799, 4095, rowFlags(true, 3));
        const one = rows.finish();
        assert.deepEqual([...one.f32], [...obs, -0.25, 0.5, 0.75, -1]);
        assert.deepEqual([...one.u32], [2, 1799, 4095, (3 << 4) | 1]);
        assert.deepEqual([one.u32[ROW.EPISODE], one.u32[ROW.TICK], one.u32[ROW.AGENT], one.u32[ROW.FLAGS] & 1, flagsReason(one.u32[ROW.FLAGS])],
            [2, 1799, 4095, 1, 3]);
        assert.throws(() => rows.push(words([1]), words([1, 2]), words([1, 2]), 0, 0, 0, 0), /a row has 5 floats; this dataset's rows have 28/);
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});

test('U25: the rule\'s texts parse in the think wrapper and differ only in their four functions; tags round-trip', () => {
    const definition = herdDefinition(), wrapped = seams => A.buildAgentThinkWGSL(definition, herdRule(seams), HERD_DEFAULTS.thinkEvery);
    for (const seams of [DRIVER, LABEL, LABEL_DRIVING]) new WgslReflect(wrapped(seams));
    // With the four functions taken out, every text is the same text.
    const body = seams => {
        let text = herdRule(seams);
        for (const name of ['wander', 'coin', 'resting', 'tag']) {
            const begin = text.indexOf(`fn ${name}(`), end = text.indexOf('\nfn ', begin);   // up to the next function
            assert.ok(begin >= 0 && end > begin, `fn ${name} is missing`);
            text = text.slice(0, begin) + text.slice(end + 1);
        }
        return text;
    };
    assert.equal(body(LABEL), body(DRIVER));
    assert.equal(body(LABEL_DRIVING), body(DRIVER));
    assert.ok(body(DRIVER).includes('fn think(i: u32, species: u32, tick: u32)') && !/fn (wander|coin|resting|tag)\(/.test(body(DRIVER)));
    // The driver keeps the demo's hidden inputs; the label has none of them and carries the step.
    const driver = herdRule(DRIVER), label = herdRule(LABEL);
    assert.ok(driver.includes('0x9e3779b1u') && driver.includes('(i % 2u) == 0u') && driver.includes('agents[i].state.w > 0.0'));
    assert.ok(driver.includes('fn tag(i: u32, tick: u32, why: u32) -> u32 { return why; }'));
    assert.ok(!label.includes('0x9e3779b1u') && !label.includes('% 2u') && !label.includes('state.w'));
    assert.ok(label.includes('fn wander(i: u32, tick: u32) -> f32 { return 0.0; }') && label.includes('fn coin(i: u32) -> bool { return true; }') &&
        label.includes('fn resting(i: u32) -> bool { return false; }'));
    assert.ok(label.includes('return ((tick + 1u) << 8u) | (((agents[i].ids.y >> 1u) & 1u) << 4u) | why;'));
    assert.throws(() => herdRule({ wander: true, coin: true, resting: true }), /herdRule needs a boolean "tag"/);
    assert.deepEqual(Object.keys(herdWolves.texts()), ['driver', 'label', 'catches', 'respawn']);
    assert.equal(herdWolves.texts().label, label);

    for (const [tick, frozen, why] of [[0, 0, 0], [0, 1, 5], [299, 1, 3], [1799, 0, 4], [(1 << 24) - 2, 1, 15]]) {
        const word = TAG.pack(tick, frozen, why);
        assert.deepEqual([TAG.tick(word), TAG.frozen(word), TAG.why(word)], [tick, frozen, why]);
        assert.notEqual(word, 0);
    }
    assert.equal(TAG.tick(0), -1);                                  // a row no pass has written
    assert.equal(flagsReason(rowFlags(true, 5)), 5);
    assert.equal(rowFlags(true, 5) & 1, 1);
    assert.equal(rowFlags(false, 3), 3 << 4);
});

test('U27: the row sampler repeats, keeps the asked share, and a held-out episode never trains', async () => {
    for (const [species, rate] of [[0, 1 / 16], [1, 1 / 4], [0, 0.9]]) {
        let kept = 0;
        const first = [];
        for (let row = 0; row < 100_000; row++) {
            const keep = keepRow(row % 4096, Math.floor(row / 4096) * 6 + 5, species, rate);
            first.push(keep);
            if (keep) kept++;
        }
        assert.ok(Math.abs(kept / 100_000 - rate) < 0.01, `kept ${kept} of 100,000 at rate ${rate}`);
        for (let row = 0; row < 100_000; row += 997) {
            assert.equal(keepRow(row % 4096, Math.floor(row / 4096) * 6 + 5, species, rate), first[row]);
        }
    }
    assert.ok(keepRow(7, 11, 0, 1) && keepRow(7, 11, 1, 2));
    // Another species draws from another stream: the two keep different rows.
    let differ = 0;
    for (let agent = 0; agent < 4096; agent++) if (keepRow(agent, 5, 0, 0.5) !== keepRow(agent, 5, 1, 0.5)) differ++;
    assert.ok(differ > 1500 && differ < 2600, `${differ} of 4,096 rows differ between species`);

    // A decision is kept once: in the stretch that follows the last copy, up to this step.
    assert.deepEqual([5, 6, 11, 12].map(tick => isFresh(tick, 5, 11)), [false, true, true, false]);
    assert.ok(isFresh(0, -1, 5) && !isFresh(TAG.tick(0), -1, 5), 'a row no pass has written is never fresh');
    // A copy rarer than a think cycle is fine; one more frequent would see some agents twice and is refused.
    await assert.rejects(recordEpisodes(null, herdWolves, { stride: 2 }),
        /record: stride 2 must be a whole number of steps, at least thinkEvery \(3\)\./);
    await assert.rejects(recordEpisodes(null, herdWolves, { seeds: [] }), /record: needs at least one seed/);
    await assert.rejects(recordEpisodes(null, herdWolves, { steps: 5 }), /steps \(5\) of at least one stride \(6\)/);

    const { header, species } = sample(), u32 = species.prey.u32, rows = species.prey.rows;
    const split = splitRows(header, u32), all = splitRows(header, u32, { dropFrozen: false });
    const episode = row => u32[row * ROW_WORDS + ROW.EPISODE], frozen = row => u32[row * ROW_WORDS + ROW.FLAGS] & 1;
    assert.ok(split.train.length > 0 && split.heldOut.length > 0);
    assert.ok(split.train.every(row => episode(row) !== 2 && !frozen(row)));
    assert.ok(split.heldOut.every(row => episode(row) === 2 && !frozen(row)));
    assert.equal(all.train.length + all.heldOut.length, rows);
    assert.ok(all.train.every(row => episode(row) !== 2) && all.heldOut.every(row => episode(row) === 2));
    assert.ok(all.train.length > split.train.length);
    assert.throws(() => splitRows({ episodes: header.episodes.slice(0, 2) }, u32), /names episode 2; header\.json lists 2/);
});
