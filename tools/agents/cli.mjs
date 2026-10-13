// Step 7, part 2: what the tools' command lines share. Node only.
import { readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

/** The repository's root, with a trailing separator. */
export const root = fileURLToPath(new URL('../../', import.meta.url));

/** The tools run the BUILT library: a dist older than src would run old code under a new commit. */
export function assertDistCurrent() {
    const newest = dir => readdirSync(join(root, dir), { recursive: true })
        .map(f => join(root, dir, String(f))).filter(f => statSync(f).isFile())
        .reduce((t, f) => Math.max(t, statSync(f).mtimeMs), 0);
    if (newest('src') > statSync(join(root, 'dist/agents.js')).mtimeMs) {
        console.error('agents-tools: dist/ is older than src/. Run npm run build:lib first.');
        process.exit(1);
    }
}

/** `--name value` options. The number readers refuse a bad value by the tool's and the option's name. */
export function options(args, tool) {
    const option = (name, fallback) => {
        const at = args.indexOf(`--${name}`);
        return at < 0 ? fallback : args[at + 1];
    };
    const whole = (name, fallback) => {
        const value = Number(option(name, fallback));
        if (!(Number.isInteger(value) && value > 0)) throw new Error(`${tool}: --${name} must be a whole number above zero.`);
        return value;
    };
    const positive = (name, fallback) => {
        const value = Number(option(name, fallback));
        if (!(Number.isFinite(value) && value > 0)) throw new Error(`${tool}: --${name} must be a number above zero.`);
        return value;
    };
    return { option, whole, positive };
}
