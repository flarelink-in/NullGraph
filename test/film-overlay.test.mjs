// O.0 (docs/film-overlays-design.md §3, §5): the pure part of the film text pass. No ffmpeg runs here.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { filterValue, overlayFilters } from '../tools/film/overlay.mjs';

const base = { width: 1920, height: 1080, seconds: 10, font: 'C:/Windows/Fonts/segoeui.ttf' };

test('O1 overlay text: values are escaped for the options, then for the graph', () => {
    // The same two levels the co-evolution labels use (U56): \ : ' for the options, then \ ' [ ] , ; for the graph.
    assert.equal(filterValue('Prey: the clones'), 'Prey\\\\: the clones');
    assert.equal(filterValue("x150's run, prey: a;b[c]"), "x150\\\\\\'s run\\, prey\\\\: a\\;b\\[c\\]");
    assert.equal(filterValue('C:/Windows/Fonts/segoeui.ttf'), 'C\\\\:/Windows/Fonts/segoeui.ttf');
    assert.equal(filterValue('plain'), 'plain');
});

test('O2 overlay filters: an item with only text is drawn the way the label script draws one', () => {
    const [filter] = overlayFilters({ ...base, overlays: [{ text: 'A title' }] });
    // size = height / 30 = 36, pad = 0.6 of that = 22; the same look as coevolve-film-labels.
    assert.equal(filter, 'drawtext=fontfile=C\\\\:/Windows/Fonts/segoeui.ttf:text=A title:expansion=none:x=22:y=22:fontsize=36:fontcolor=white:borderw=3:bordercolor=black@0.75');
    assert.deepEqual(overlayFilters({ ...base, overlays: [] }), []);
});

test('O3 overlay filters: position, size and time range', () => {
    const [anchored, timed] = overlayFilters({ ...base, overlays: [
        { text: 'Footer', x: 'w-text_w-40', y: 'h-60-text_h', size: 20 },
        { text: 'Middle', x: 100, y: 200, from: 1, to: 3.5 },
    ] });
    assert.match(anchored, /:x=w-text_w-40:y=h-60-text_h:fontsize=20:/);
    assert.doesNotMatch(anchored, /enable=/);
    assert.match(timed, /:x=100:y=200:fontsize=36:.*:enable=between\(t\\,1\\,3\.5\)$/);
});

test('O4 overlay filters: a fade ramps the alpha over the item\'s own range', () => {
    const [fade] = overlayFilters({ ...base, overlays: [{ text: 'Fade', from: 2, to: 6, fade: 0.5 }] });
    assert.match(fade, /:enable=between\(t\\,2\\,6\):alpha=max\(0\\,min\(1\\,min\(\(t-2\)\/0\.5\\,\(6-t\)\/0\.5\)\)\)$/);
    // With no range given, the range is the whole film: 0 to `seconds`.
    const [whole] = overlayFilters({ ...base, overlays: [{ text: 'Whole', fade: 1 }] });
    assert.match(whole, /alpha=max\(0\\,min\(1\\,min\(\(t-0\)\/1\\,\(10-t\)\/1\)\)\)$/);
});

test('O10 overlay filters: with no film length an item runs to the end, and a fade has nothing to ramp over', () => {
    const [open] = overlayFilters({ overlays: [{ text: 'Open', from: 2 }], width: 1920, height: 1080, font: base.font });
    assert.match(open, /:enable=gte\(t\\,2\)$/);      // ffmpeg's expressions have no infinity
    const [plain] = overlayFilters({ overlays: [{ text: 'Plain' }], width: 1920, height: 1080, font: base.font });
    assert.doesNotMatch(plain, /enable=/);
    assert.throws(() => overlayFilters({ overlays: [{ text: 'Fade', fade: 1 }], width: 1920, height: 1080, font: base.font }),
        /overlay 0: fade needs the film's length or an end \(to\)/);
    assert.doesNotThrow(() => overlayFilters({ overlays: [{ text: 'Fade', fade: 1, to: 4 }], width: 1920, height: 1080, font: base.font }));
});

test('O5 overlay filters: refusals name the item', () => {
    for (const [overlays, message] of [
        [[{ text: '' }], /overlay 0: text/],
        [[{ text: 'a' }, { text: 'b', size: 0 }], /overlay 1: size/],
        [[{ text: 'a', from: 4, to: 2 }], /overlay 0: from must be below to/],
        [[{ text: 'a', colour: 'red' }], /overlay 0: unknown field "colour"/],
        [[{ text: 'a', fade: -1 }], /overlay 0: fade/],
        [[{ text: 'a', from: 0, to: 1, fade: 0.6 }], /overlay 0: fade .* half/],
    ]) assert.throws(() => overlayFilters({ ...base, overlays }), message, JSON.stringify(overlays));
});
