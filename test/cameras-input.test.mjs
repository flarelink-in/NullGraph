import { test } from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';

const { outputFiles } = await build({
    entryPoints: ['src/extras/cameras/OrbitalEventProxy.ts'],
    bundle: true, write: false, format: 'esm', platform: 'node',
});
const { OrbitalEventProxy } = await import(`data:text/javascript;base64,${Buffer.from(outputFiles[0].text).toString('base64')}`);

function setup(t) {
    const previous = globalThis.window;
    const window = new EventTarget();
    globalThis.window = window;
    const canvas = new EventTarget();
    canvas.style = { touchAction: 'pan-y' };
    const inputs = [];
    const proxy = new OrbitalEventProxy({ canvas, onInput: (...input) => inputs.push(input) });
    const send = (type, id, x = 0, y = 0) => {
        const event = new Event(type);
        Object.assign(event, { pointerId: id, clientX: x, clientY: y });
        (type === 'pointerdown' ? canvas : window).dispatchEvent(event);
    };
    t.after(() => {
        proxy.destroy();
        if (previous === undefined) delete globalThis.window;
        else globalThis.window = previous;
    });
    return { window, canvas, inputs, proxy, send };
}

test('single-pointer rotation and wheel zoom keep existing scales', t => {
    const { canvas, inputs, send } = setup(t);
    send('pointerdown', 1, 10, 20);
    send('pointermove', 99, 100, 100);
    send('pointermove', 1, 20, 40);
    send('pointerup', 1);
    send('pointermove', 1, 30, 50);
    const wheel = new Event('wheel');
    Object.assign(wheel, { deltaY: 40 });
    canvas.dispatchEvent(wheel);
    assert.deepEqual(inputs, [['ROTATE', 0.1, 0.2], ['ZOOM', 0, 2]]);
});

test('spread zooms in, pinch zooms out without rotating', t => {
    const { inputs, send } = setup(t);
    send('pointerdown', 1, 0, 0);
    send('pointerdown', 2, 100, 0);
    send('pointermove', 2, 120, 0);
    send('pointermove', 1, 20, 0);
    assert.deepEqual(inputs, [['ZOOM', 0, -1], ['ZOOM', 0, 1]]);
});

test('lifting either finger resumes rotation from its latest position', t => {
    const { inputs, send } = setup(t);
    for (const lifted of [1, 2]) {
        send('pointerdown', 1, 0, 0);
        send('pointerdown', 2, 100, 0);
        send('pointermove', 1, 10, 0);
        send('pointermove', 2, 110, 0);
        send('pointerup', lifted);
        inputs.length = 0;
        send('pointermove', lifted === 1 ? 2 : 1, lifted === 1 ? 115 : 15, 5);
        assert.deepEqual(inputs, [['ROTATE', 0.05, 0.05]]);
        send('pointerup', lifted === 1 ? 2 : 1);
    }
});

test('extra fingers, cancellation and focus loss do not leave a stuck gesture', t => {
    const { window, inputs, send } = setup(t);
    send('pointerdown', 1, 0, 0);
    send('pointerdown', 2, 100, 0);
    send('pointerdown', 3, 200, 0);
    send('pointermove', 2, 120, 0);
    assert.equal(inputs.length, 0);
    send('pointercancel', 3);
    send('pointermove', 2, 140, 0);
    assert.deepEqual(inputs, [['ZOOM', 0, -1]]);
    send('pointercancel', 1);
    send('pointercancel', 2);
    send('pointermove', 2, 150, 0);
    assert.equal(inputs.length, 1);
    send('pointerdown', 4);
    window.dispatchEvent(new Event('blur'));
    send('pointermove', 4, 20, 20);
    assert.equal(inputs.length, 1);
});

test('destroy removes listeners and restores touch-action without overwriting app changes', t => {
    const { canvas, inputs, proxy, send } = setup(t);
    assert.equal(canvas.style.touchAction, 'none');
    proxy.destroy();
    proxy.destroy();
    assert.equal(canvas.style.touchAction, 'pan-y');
    send('pointerdown', 1);
    send('pointermove', 1, 20, 20);
    assert.equal(inputs.length, 0);
    const second = new OrbitalEventProxy({ canvas, onInput: () => {} });
    canvas.style.touchAction = 'auto';
    second.destroy();
    assert.equal(canvas.style.touchAction, 'auto');
});