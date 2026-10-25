// Spike S1's worker way: the same film, on a device and an OffscreenCanvas of the worker's own (a
// GPUDevice does not transfer: design §7). Logs and marks go to the page, which prints them.

import { runCase, upload } from './film.js';

self.onmessage = async ({ data: cfg }) => {
    const log = (line, level = 'log') => self.postMessage({ type: 'log', line, level });
    const mark = name => self.postMessage({ type: 'mark', name, at: performance.timeOrigin + performance.now() });
    try {
        const result = await runCase(cfg, { surfaceKind: 'offscreen', log, mark, upload });
        self.postMessage({ type: 'done', result });
    } catch (e) {
        self.postMessage({ type: 'failed', error: String(e?.stack ?? e) });
    }
};
