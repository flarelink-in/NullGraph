// Spikes A and B of step 10's phase 10.0 (docs/agents-live-switch-design.md §4, §6 and §13):
//   A. ONNX Runtime in a module worker built by the site's Vite, on its dev server and in a
//      production build, with its WASM vendored as decision D5 says (public/vendor/onnxruntime-web/);
//   B. decision D1's device switch over one OffscreenCanvas (worker.js says how).
//
//   node test/spikes/live-switch/run.mjs [--mode dev|build] [--wasm prefix|wasm|default]
//        [--exclude-ort] [--extern-wasm] [--settle] [--site <dir>] [--out <dir>]
//
//   --wasm         where ONNX Runtime looks for its WASM: 'prefix' (default) sets wasmPaths to
//                  '/vendor/onnxruntime-web/', so it imports the loader and fetches the .wasm from
//                  there; 'wasm' sets only the .wasm's URL (the loader built into the bundle is used);
//                  'default' sets nothing.
//   --exclude-ort  adds onnxruntime-web to optimizeDeps.exclude, beside null-graph.
//   --extern-wasm  resolves onnxruntime-web with its 'onnxruntime-web-use-extern-wasm' condition:
//                  ort.webgpu.min.mjs, without the loader built in, instead of ort.webgpu.bundle.min.mjs.
//   --settle       the switch waits for the old device's queue before destroying it (worker.js).
//
// Vite is the site's (its node_modules), with the settings of the site's vite.config.ts that touch a
// worker: worker.format 'es', optimizeDeps.exclude ['null-graph']. null-graph is this repository's
// dist/ (what phase 10.1 pushes to the site); onnxruntime-web is this repository's 1.29.0 (D4); the
// two vendored files are copied into a scratch public folder. Nothing in the site changes. The page is
// driven by test/gpu/drive-page.mjs in headless Chrome: screenshots, and every console line of the
// page and the worker, land in <out>/drive-<mode>-<wasm>.
import { spawn } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { gzipSync } from 'node:zlib';

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, '../../..');
const argv = process.argv.slice(2);
const flag = (name) => argv.includes(`--${name}`);
const option = (name, fallback) => {
    const at = argv.indexOf(`--${name}`);
    return at >= 0 ? argv[at + 1] : fallback;
};
const mode = option('mode', 'dev');
const wasm = option('wasm', 'prefix');
const site = resolve(option('site', process.env.NG_SITE ?? 'D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts'));
const work = resolve(option('out', join(tmpdir(), 'nullgraph-live-switch')));
const viteEntry = join(site, 'node_modules/vite/dist/node/index.js');
if (!existsSync(viteEntry)) throw new Error(`no Vite at ${viteEntry}: pass --site <the site's folder>.`);
const vite = await import(pathToFileURL(viteEntry).href);
console.log(`live-switch spike: Vite ${vite.version} from ${site}; mode ${mode}; wasm ${wasm}` +
    `${flag('exclude-ort') ? '; onnxruntime-web excluded from optimizeDeps' : ''}${flag('extern-wasm') ? '; extern-wasm condition' : ''}`);

// D5: the two files, vendored. They are the site's public/vendor/onnxruntime-web/ to be.
const ortDist = join(repo, 'node_modules/onnxruntime-web/dist');
const publicDir = join(work, 'public');
mkdirSync(join(publicDir, 'vendor/onnxruntime-web'), { recursive: true });
for (const file of ['ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm']) {
    copyFileSync(join(ortDist, file), join(publicDir, 'vendor/onnxruntime-web', file));
}

const label = `${mode}-${wasm}${flag('exclude-ort') ? '-excluded' : ''}${flag('extern-wasm') ? '-extern' : ''}${flag('settle') ? '-settle' : ''}`;
const config = {
    configFile: false,
    root: here,
    publicDir,
    cacheDir: join(work, `vite-cache-${label}`),
    logLevel: 'info',
    resolve: {
        alias: [
            { find: /^null-graph$/, replacement: join(repo, 'dist/index.js') },
            { find: /^null-graph\/(.+)$/, replacement: join(repo, 'dist/$1.js') },
        ],
        ...(flag('extern-wasm') ? { conditions: ['onnxruntime-web-use-extern-wasm'] } : {}),
    },
    worker: { format: 'es' },
    optimizeDeps: { exclude: ['null-graph', ...(flag('exclude-ort') ? ['onnxruntime-web'] : [])] },
    server: { port: 5190, strictPort: false, fs: { allow: [repo, work] } },
    preview: { port: 5290, strictPort: false },
    build: { outDir: join(work, `build-${label}`), emptyOutDir: true, rollupOptions: { input: join(here, 'index.html') } },
};

/** Drive the page: screenshots before and after the switch, and the log. Returns drive-page's exit code. */
async function drive(url) {
    const out = join(work, `drive-${label}`);
    mkdirSync(out, { recursive: true });
    const actions = join(out, 'actions.json');
    writeFileSync(actions, JSON.stringify([
        { shot: 'a.png' },
        { eval: 'spikeSwitch()' },
        { waitFor: '\\[spike\\] (B rendered|SWITCH failed)', timeout: 120 },
        { shot: 'b.png' },
        { waitFor: '\\[spike\\] (done|SWITCH failed)', timeout: 120 },
        { shot: 'resized.png' },
    ]));
    const code = await new Promise((done) => {
        const child = spawn(process.execPath, [join(repo, 'test/gpu/drive-page.mjs'), url, '--wait-for', '\\[spike\\] (A rendered|INIT failed)',
            '--actions', actions, '--out', out, '--timeout', '180'], { stdio: ['ignore', 'pipe', 'pipe'] });
        child.stdout.on('data', (d) => process.stdout.write(d));
        child.stderr.on('data', (d) => process.stderr.write(d));
        child.on('exit', done);
    });
    const log = existsSync(join(out, 'console.log')) ? readFileSync(join(out, 'console.log'), 'utf8') : '';
    console.log(`\n--- [spike] lines and errors (${out}):`);
    for (const line of log.split('\n')) {
        if (/\[spike\]|error|exception|warning|optimized|reload/i.test(line)) console.log(line.length > 900 ? `${line.slice(0, 900)}…` : line);
    }
    return code;
}

/** Every file of a build, with its size, and gzip's size for scripts and WASM. */
function listing(dir) {
    const rows = [];
    const walk = (d) => {
        for (const name of readdirSync(d)) {
            const path = join(d, name);
            if (statSync(path).isDirectory()) walk(path);
            else {
                const bytes = statSync(path).size;
                const zipped = /\.(m?js|wasm)$/.test(name) ? gzipSync(readFileSync(path)).length : null;
                rows.push({ file: relative(dir, path).replaceAll('\\', '/'), bytes, gzip: zipped });
            }
        }
    };
    walk(dir);
    return rows.sort((a, b) => b.bytes - a.bytes);
}

let code;
if (mode === 'dev') {
    const server = await vite.createServer({ ...config, mode: 'development' });
    await server.listen();
    code = await drive(`${server.resolvedUrls.local[0]}index.html?wasm=${wasm}${flag('settle') ? '&settle=1' : ''}`);
    await server.close();
} else {
    const t0 = Date.now();
    await vite.build({ ...config, mode: 'production' });
    const files = listing(config.build.outDir);
    console.log(`\n--- the build (${((Date.now() - t0) / 1000).toFixed(1)} s, ${config.build.outDir}):`);
    for (const f of files) console.log(`   ${f.file.padEnd(70)} ${f.bytes.toLocaleString('en-US').padStart(12)} B${f.gzip === null ? '' : `, gzip ${f.gzip.toLocaleString('en-US')} B`}`);
    writeFileSync(join(work, `build-${label}.json`), JSON.stringify(files, null, 2));
    const server = await vite.preview({ ...config, mode: 'production' });
    code = await drive(`${server.resolvedUrls.local[0]}index.html?wasm=${wasm}${flag('settle') ? '&settle=1' : ''}`);
    await new Promise((done) => server.httpServer.close(done));
}
console.log(`\nlive-switch spike (${label}): drive-page exited ${code}`);
process.exit(code ?? 2);
