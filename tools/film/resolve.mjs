// D2.1 (docs/offline-films-design.md §10): lets Node run the Herd & Wolves copies as the site runs them.
// The copies are the site's TypeScript files, byte for byte, run by Node's type stripping: their bare
// `null-graph/*` imports resolve to this repository's dist/, and `.ts` is added to their extensionless
// relative imports. Keeping the copies unedited is what lets `npm run film:check` compare them by hash.
// Node only.
import { registerHooks } from 'node:module';
import { existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const DIST = fileURLToPath(new URL('../../dist/', import.meta.url));
const ENTRY = { '': 'index', geometry: 'geometry', loaders: 'loaders', materials: 'materials', visibility: 'visibility',
    agents: 'agents', post: 'post', cameras: 'cameras', lights: 'lights', text: 'text', inference: 'inference',
    picking: 'picking', shadows: 'shadows', environment: 'environment', profiler: 'profiler' };

let installed = false;

/** Installs the resolve hook once per process. */
export function hookCopies() {
    if (installed) return;
    installed = true;
    registerHooks({
        resolve(specifier, context, next) {
            const m = /^null-graph(?:\/(.*))?$/.exec(specifier);
            if (m) {
                const entry = ENTRY[m[1] ?? ''];
                if (!entry) throw new Error(`film: a copy imports "${specifier}", which is not a null-graph entry.`);
                return { url: pathToFileURL(`${DIST}${entry}.js`).href, shortCircuit: true };
            }
            if (specifier.startsWith('.') && !/\.[cm]?[jt]s$/.test(specifier) && context.parentURL?.endsWith('.ts')) {
                const url = new URL(`${specifier}.ts`, context.parentURL);
                if (existsSync(url)) return { url: url.href, shortCircuit: true };
            }
            return next(specifier, context);
        },
    });
}
