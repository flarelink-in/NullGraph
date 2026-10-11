// The GPU suite registry — one list for both hosts:
//   npm run test:gpu           Node + Dawn (test/gpu/run-gpu-tests.mjs)
//   npm run test:gpu:browser   headless Chrome (test/gpu/browser-run.mjs)
//
// A new GPU harness is one line here plus its folder: test/<name>-gpu/{run.js,index.html}, with
// run.js publishing its summary on `key` through test/gpu/harness-env.mjs.

export const SUITES = [
    { name: 'shadows', file: 'test/shadows-gpu/run.js', key: '__SHADOW_RESULTS__' },
    { name: 'sky', file: 'test/environment-gpu/run.js', key: '__SKY_RESULTS__' },
    { name: 'visibility', file: 'test/visibility-gpu/run.js', key: '__VISIBILITY_RESULTS__' },
    { name: 'core', file: 'test/core-gpu/run.js', key: '__CORE_GPU_RESULTS__' },
    { name: 'onnx', file: 'test/onnx-gpu/run.js', key: '__ONNX_RESULTS__' },
    { name: 'picking', file: 'test/picking-gpu/run.js', key: '__PICKING_RESULTS__' },
];

/**
 * Render scenes smoke-tested through the CLI (test/gpu/render.mjs): exit 0, no WebGPU error, not a
 * uniform frame. The quick start is here because it is the first code every user and agent copies.
 * Node only — the CLI is the Node host.
 */
export const RENDER_SCENES = [
    { name: 'render:quick-start', file: 'test/gpu/scenes/quick-start.mjs' },
];
