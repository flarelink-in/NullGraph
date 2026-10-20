// The plain-text reports of tools/agents/coevolve-report.mjs (docs/agents-coevo-runbook.md), headless.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { crossplayLines, evaluationLines, trainingLines } from '../tools/agents/coevolve-report.mjs';

test('U56 report: training catches by evolving species, per block of generations', () => {
    const entries = [{ event: 'calibration' },
        { t: 0, species: 'prey', scores: [-50, -40] }, { t: 1, species: 'wolf', scores: [60, 50] },
        { t: 2, species: 'prey', scores: [-30, -20] }, { t: 3, species: 'wolf', scores: [44, 46] }];
    const lines = trainingLines(entries, 2);
    assert.equal(lines.length, 4);
    assert.match(lines[2], /^1–2 +45\.0 \[40\.\.50\] +55\.0 \[50\.\.60\]$/);
    assert.match(lines[3], /^3–4 +25\.0 \[20\.\.30\] +45\.0 \[44\.\.46\]$/);
    assert.throws(() => trainingLines([{ event: 'calibration' }]), /no completed generation/);
});

test('U56 report: paired comparisons with their verdicts, and diagnostics for the telling matchups', () => {
    const interval = mean => ({ interval: { mean } });
    const matchups = { 'gen-0010/rule': 40, 'rule/rule': 55, 'rule/gen-0010': 58, 'gen-0010/clone': 56, 'clone/clone': 54,
        'clone/gen-0010': 50, 'gen-0010/gen-0010': 48, 'gen-0010/label': 41, 'label/label': 55 };
    const comparison = (species, baseline, minuend, subtrahend, low, high, superior) => ({ species, subject: 'gen-0010', baseline,
        minuend, subtrahend, difference: { mean: matchups[minuend] - matchups[subtrahend], low, high }, superior });
    const history = Array.from({ length: 4 }, (_, k) => ({ tick: k * 300, preyWallShare: k / 10, wolfRestFraction: 0.5,
        preyMeanSpeed: 1, nearestPreyMean: 3 }));
    const evaluation = {
        evaluation: { episodes: 52, T: 1200 }, heldOut: { purpose: 'validation' }, experiment: { dt: 1 / 60 },
        device: { adapter: { vendor: 'nvidia', architecture: 'turing' } },
        results: {
            matchups: Object.entries(matchups).map(([id, mean]) => ({ id, ...interval(mean) })),
            comparisons: [comparison('prey', 'rule', 'gen-0010/rule', 'rule/rule', -16, -14, true),
                comparison('wolf', 'rule', 'rule/gen-0010', 'rule/rule', 1, 5, true),
                comparison('prey', 'clone', 'gen-0010/clone', 'clone/clone', 1, 3, false),
                comparison('wolf', 'clone', 'clone/gen-0010', 'clone/clone', -6, -2, false)],
        },
        diagnosticResults: ['gen-0010/rule', 'gen-0010/label', 'rule/rule'].map(id => ({ id, episodes: [{ history }] })),
    };
    const lines = evaluationLines(evaluation);
    assert.match(lines[0], /^52 validation episodes of 1200 steps on nvidia turing; catches a tile\.$/);
    assert.match(lines.find(l => l.startsWith('gen-0010   rule')), /40\.0 \(55\.0\) -15\.0 \[-16\.0, -14\.0\] better +58\.0 \(55\.0\) \+3\.0 \[\+1\.0, \+5\.0\] better$/);
    // Prey caught more than the clone prey's own pairing, wolves catching fewer: both worse.
    assert.match(lines.find(l => l.startsWith('gen-0010   clone')), /56\.0 \(54\.0\) \+2\.0 \[\+1\.0, \+3\.0\] worse +50\.0 \(54\.0\) -4\.0 \[-6\.0, -2\.0\] worse$/);
    // Wall shares at 0, 5, 10 and 15 s; the label's matchups stay in evaluation.json, not in the report.
    assert.match(lines.find(l => l.startsWith('gen-0010/rule')), /0\.00  0\.10  0\.20  0\.30 +0\.50  1\.00  3\.00$/);
    assert.ok(lines.some(l => l.startsWith('rule/rule')));
    assert.ok(!lines.some(l => l.startsWith('gen-0010/label')));
});

test('U56 report: a cross-play matrix with prey generations as rows, and its forgetting', () => {
    const drop = (value, low, high) => ({ value, low, high });
    const lines = crossplayLines({ generations: ['gen-0000', 'gen-0100'], anchors: ['gen-0000'], mean: [[52, 60.25], [40, 48]],
        forgetting: { prey: drop(0, 0, 0.5), wolf: drop(3.25, 1, 5) } });
    assert.match(lines[1], /^prey \\ wolves +gen-0000 +gen-0100$/);
    assert.match(lines[2], /^gen-0000 +52\.0 +60\.3$/);
    assert.match(lines[3], /^gen-0100 +40\.0 +48\.0$/);
    assert.equal(lines[5], '  prey 0.0 [0.0, 0.5], wolves 3.3 [1.0, 5.0]');
});
