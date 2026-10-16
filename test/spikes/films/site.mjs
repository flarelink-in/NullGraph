// D2.0 only (docs/offline-films-design.md §17): the test site's Herd & Wolves look modules, imported as
// they are, under Node's type stripping. Bare `null-graph/*` imports resolve to this repository's dist/,
// and `.ts` is added to extensionless relative imports. D2.1 replaces this with copies in tools/film/
// (decided by the user, design §19); a spike may read the site directly.
//
//   NG_SITE  the site's folder (default: the user's local checkout)
import { registerHooks } from 'node:module';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

export const SITE = process.env.NG_SITE ?? 'D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts';
const DEMO = `${SITE}/src/data/engine-demos/demos/AgentsExperiments/HerdWolvesGPU`;
const DIST = fileURLToPath(new URL('../../../dist/', import.meta.url));
const ENTRY = { '': 'index', geometry: 'geometry', loaders: 'loaders', materials: 'materials', visibility: 'visibility',
    agents: 'agents', post: 'post', cameras: 'cameras', lights: 'lights', text: 'text', inference: 'inference' };

let hooked = false;
function hook() {
    if (hooked) return;
    hooked = true;
    registerHooks({
        resolve(specifier, context, next) {
            const m = /^null-graph(?:\/(.*))?$/.exec(specifier);
            if (m) {
                const entry = ENTRY[m[1] ?? ''];
                if (!entry) throw new Error(`films spike: no dist entry for "${specifier}"`);
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

/** The demo's look modules, by name: models, field, fence, flashes, brain. */
export async function siteModules() {
    if (!existsSync(DEMO)) throw new Error(`films spike: the site is not at ${SITE} (NG_SITE).`);
    hook();
    const load = name => import(pathToFileURL(`${DEMO}/${name}.ts`).href);
    const [models, field, flashes, brain] = await Promise.all(['models', 'field', 'flashes', 'brain'].map(load));
    return { models, field, flashes, brain };
}

/** A file of the site's public/ folder, as an ArrayBuffer. Models are read there, never copied. */
export function publicFile(path) {
    const file = `${SITE}/public/${path}`;
    if (!existsSync(file)) throw new Error(`films spike: ${path} is missing from the site's public/ (${file}).`);
    const bytes = readFileSync(file);
    return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}
